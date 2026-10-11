"""Pure-Python GENERAL/SCRIPT serialization from already computed predictions.

Safe identifiers travel through the attributed procedural timing planner; the
original MOSS prose is quoted afterward. An independent scanner verifies the
final grammar, word sequence and elapsed budget.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from ._vendor.procedural_prompt_format import render_timed

FORMAT_ID = "humanness-general-script-v1"
_SPEECH = re.compile(r"^\(([^\n]*?); (\d+(?:\.\d+)?) seconds duration\) ?")
_PAUSE = re.compile(r"^\[(\d+(?:\.\d+)?) seconds pause\]$")
_BURST = re.compile(r'\(("(?:[^"\\]|\\.)*"), (\d+(?:\.\d+)?) seconds\)')
_OLD_PREFIX = re.compile(r"^(?:\(([^()]*)\) )?\[(\d+(?:\.\d+)?) seconds duration\] ?")


def sources():
    return json.loads((Path(__file__).parent / "assets" / "sources.json").read_text())


def verify_script(script, transcript, duration_s, expected_bursts=None):
    """Check words, event count, finite durations and independent elapsed budget."""
    errors, words, bursts, budget = [], [], [], 0.0
    for number, line in enumerate(script.splitlines(), 1):
        if not line.strip():
            continue
        pause = _PAUSE.fullmatch(line)
        if pause:
            value = float(pause[1])
            if value <= 0:
                errors.append(f"Nonpositive pause on line {number}")
            budget += value
            continue
        speech = _SPEECH.match(line)
        body = line[speech.end():] if speech else line
        matches = list(_BURST.finditer(body))
        for match in matches:
            caption, duration = json.loads(match[1]), float(match[2])
            if not caption.strip() or duration <= 0:
                errors.append(f"Empty caption or nonpositive burst on line {number}")
            bursts.append({"caption": caption, "duration_s": duration})
        spoken = _BURST.sub("", body).strip()
        if speech:
            value = float(speech[2])
            if value <= 0 or not speech[1].strip():
                errors.append(f"Invalid segment cue/budget on line {number}")
            budget += value
            words.extend(spoken.split())
        elif len(matches) == 1 and matches[0].span() == (0, len(line)):
            budget += float(matches[0][2])
        else:
            errors.append(f"Unrecognized script line {number}")
    if not math.isfinite(budget) or abs(budget - duration_s) > .015:
        errors.append("Printed elapsed budget differs from recording duration")
    if words != transcript.split():
        errors.append("ASR words were lost, duplicated or reordered")
    if expected_bursts is not None and len(bursts) != expected_bursts:
        errors.append("Some selected events are missing or duplicated")
    return {"verified": not errors, "errors": errors, "printed_budget_s": round(budget, 6),
            "word_count": len(words), "burst_count": len(bursts), "parsed_bursts": bursts}


def compose_annotation(acoustic, moss, *, pause_min_s=.2):
    """Compose the prompt and canonical endpoints using one selected event list."""
    duration = float(acoustic["dur"])
    if abs(duration - float(moss["audio_duration_s"])) > .002:
        raise ValueError("The two stages disagree about the audio duration")
    audio_hash = acoustic.get("audio_sha256")
    if not audio_hash or audio_hash != moss.get("audio_sha256"):
        raise ValueError("Both stages must refer to the same original audio SHA-256")
    general = moss["voice_caption"]["prediction"].strip()
    if not general:
        raise ValueError("The MOSS whole-recording caption is empty")
    for sentence in acoustic.get("sentences", []):
        if sentence.get("start") is None or sentence.get("end") is None:
            raise ValueError("Sentence alignment is unavailable; timestamps cannot be invented")
    events, safe_events = [], []
    for index, event in enumerate(moss["burst_captions"]):
        start, end = float(event["start_s"]), float(event["end_s"])
        caption = event["caption"]["prediction"]
        if not 0 <= start < end <= duration + .001 or not caption.strip():
            raise ValueError("Every burst needs valid start/end seconds and nonempty prose")
        identifier = f"HVA_EVENT_{index:04d}"
        events.append({"id": identifier, "start_s": start, "end_s": end,
                       "duration_s": end - start, "caption": caption,
                       "onset_probability": event.get("onset_probability"),
                       "top3_classes": copy.deepcopy(event.get("top3_classes", []))})
        safe_events.append({"start_s": start, "end_s": end, "label": identifier})
    lines, plan = render_timed(acoustic.get("sentences", []), acoustic.get("words", []),
                               safe_events, duration, pause_min=pause_min_s)
    sentence_sources = acoustic.get("sentences", [])
    segments = []
    for item in plan["timeline"]:
        if item["kind"] != "speech":
            continue
        candidates = [(max(0., min(item["end_s"], s["end"]) - max(item["start_s"], s["start"])), i, s)
                      for i, s in enumerate(sentence_sources)]
        matched = max(candidates, key=lambda x: x[0]) if candidates else None
        source = matched[2] if matched and matched[0] > 0 else {}
        cue = str(source.get("caption", "")).replace("(", "").replace(")", "").replace("\n", " ").strip()
        detail = copy.deepcopy(source.get("caption_detail", {}))
        segments.append({"id": f"segment-{len(segments)+1:03d}",
                         "start_s": item["start_s"], "end_s": item["end_s"],
                         "duration_s": item["end_s"] - item["start_s"],
                         "text": item["text"], "cue": cue or "delivery unspecified",
                         "source_sentence_index": matched[1] if source else None,
                         "selected_emotions": detail.get("emotions", []),
                         "selected_voice_properties": detail.get("voicenet", []),
                         "inline_event_ids": [e["label"] for e in item.get("inline_events", [])],
                         "duration_includes_inline_events": True})
    for index, line in enumerate(lines):
        prefix = _OLD_PREFIX.match(line)
        if prefix:
            lines[index] = f"({prefix[1] or 'delivery unspecified'}; {prefix[2]} seconds duration) " + line[prefix.end():]
    for event in events:
        quoted = json.dumps(event["caption"], ensure_ascii=False)
        count = 0
        pattern = re.compile(r"\(" + re.escape(event["id"]) + r", (\d+(?:\.\d+)?) seconds\)")
        for index, line in enumerate(lines):
            lines[index], n = pattern.subn(lambda m: f"({quoted}, {m[1]} seconds)", line)
            count += n
        if count != 1:
            raise ValueError(f"Event {event['id']} was not serialized exactly once")
    script = "\n".join(lines)
    transcript = " ".join(w["w"] for w in acoustic.get("words", []))
    if not transcript and str(acoustic.get("transcript", "")).strip():
        raise ValueError("Speech is present without verified word alignment")
    check = verify_script(script, transcript, duration, len(events))
    if not check["verified"]:
        raise ValueError("; ".join(check["errors"]))
    normalized_asr = " ".join(str(acoustic.get("transcript", "")).split())
    metadata = acoustic.get("source_metadata", {})
    public_metadata = {k: metadata[k] for k in ["id", "source", "source_key", "language", "voice", "title", "source_subset", "source_dataset"] if k in metadata}
    annotation = {
        "schema_version": FORMAT_ID, "format": "GENERAL / SCRIPT",
        "audio": {"sha256": audio_hash, "duration_s": duration,
                  "sample_id": acoustic.get("comparison_id", acoustic.get("id")),
                  "language": acoustic.get("language", acoustic.get("asr", {}).get("known_language")),
                  "source_metadata": public_metadata},
        "general": general, "script": script, "prompt": "GENERAL: " + general + "\nSCRIPT:\n" + script,
        "segments": segments, "burst_events": events, "words": copy.deepcopy(acoustic.get("words", [])),
        "asr_transcript": acoustic.get("transcript", ""), "aligned_transcript": transcript,
        "asr_and_aligned_words_equal": normalized_asr == transcript,
        "timing": {**plan, "format_id": FORMAT_ID, "duration_style": "round_combined",
                   "independent_verification": check,
                   "inline_position_policy": "Nearest word gap to event midpoint; exact overlap remains in JSON",
                   "overlap_policy": "Inline event durations are contained in the enclosing segment budget"},
        "models": sources()["models"], "settings": {
            "locator": copy.deepcopy(moss["locator_settings"]), "pause_min_s": pause_min_s,
            "sentence_cues": copy.deepcopy(acoustic.get("caption_settings", {})),
            "normalization": copy.deepcopy(acoustic.get("normalization_provenance", {})),
            "procedural_revision": sources()["procedural"]["revision"]},
        "raw_model_outputs": {
            "humaneness_ears_medium": copy.deepcopy(acoustic.get("humaneness_ears_medium", {})),
            "sentence_predictions": [{k: copy.deepcopy(v) for k, v in sentence.items()
                                      if k in {"start", "end", "text", "scores", "caption", "caption_detail", "humaneness_ears_medium"}}
                                     for sentence in sentence_sources],
            "moss": copy.deepcopy(moss)},
        "machine_predictions_not_ground_truth": True,
    }
    annotation["annotation_sha256"] = hashlib.sha256(json.dumps(annotation, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return annotation
