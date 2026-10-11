"""Inference worker: one environment for acoustic models, another for MOSS."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .formatting import sources
from .upstream import procedural_source


def acoustic(args):
    root = procedural_source(args.procedural_dir)
    from huggingface_hub import snapshot_download
    import burst_captions
    from humaneness_caption_settings import load_settings
    from refresh_saved_caption import refresh_saved_caption
    if args.language not in "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split():
        raise ValueError("This Parakeet checkpoint does not support the requested language")
    parakeet = sources()["models"]["parakeet"]
    burst_captions.PARAKEET_MODEL = args.parakeet_dir or snapshot_download(parakeet["repo"], revision=parakeet["revision"])
    cap = burst_captions.BurstCaptioner(device=args.device, score_backend="humaneness-medium",
        humaneness_dir=args.humaneness_dir, baseline=str(root/"baseline_stats_humaneness_medium_speech.json"),
        use_x2=False, threads=args.threads)
    settings = load_settings()
    settings.update(id="humanness-voice-annotator-v1", threshold=args.threshold)
    cap.medium_caption_settings = settings
    record = cap.process(str(args.audio), language=args.language)
    record = refresh_saved_caption(record, cap.baseline, settings)
    record.update(audio_sha256=hashlib.sha256(args.audio.read_bytes()).hexdigest(), language=args.language)
    record["asr"]["model"] = parakeet["repo"]
    record["asr"]["revision"] = parakeet["revision"]
    return record


def moss(args):
    import torch
    from huggingface_hub import snapshot_download
    models = sources()["models"]
    voice = args.voice_dir or snapshot_download(models["moss_voice"]["repo"], revision=models["moss_voice"]["revision"])
    burst = args.burst_dir or snapshot_download(models["moss_burst"]["repo"], revision=models["moss_burst"]["revision"], allow_patterns=["adapter/*"])
    sys.path.insert(0, str(voice))
    from inference import Captioner, load_audio
    from src.processing_moss_audio import MossAudioProcessor
    torch.set_num_threads(args.threads)
    record = json.loads(args.acoustic.read_text())
    waveform = load_audio(str(args.audio))
    cap = Captioner("caption", str(Path(voice)/"adapter"), str(voice), args.device)
    cap.model.load_adapter(str(Path(burst)/"adapter"), adapter_name="burst_caption", is_trainable=False)
    def generate(start=None, end=None):
        result = cap.caption(waveform, start, end, max_new_tokens=512 if start is None else 256,
                             temperature=0.0, seed=20261011)
        if result["hit_token_limit"]:
            result = cap.caption(waveform, start, end, max_new_tokens=1024, temperature=0.0, seed=20261011)
        if result["hit_token_limit"]:
            raise RuntimeError("Caption truncated even at 1024 generated tokens")
        return result
    global_caption = generate()
    cap.task = "burst_caption"
    cap.processor = MossAudioProcessor(cap.processor._base_tokenizer, enable_time_marker=True)
    cap.model.set_adapter("burst_caption")
    procedural_source(args.procedural_dir)
    from humaneness_caption_settings import filter_events, load_settings
    settings = load_settings()
    settings["threshold"] = args.threshold
    selected = filter_events(record["humaneness_ears_medium"]["burst_event_proposals"], settings)
    events = [dict(event, caption=generate(event["start_s"], event["end_s"])) for event in selected]
    return {"audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
            "audio_duration_s": len(waveform)/16000,
            "voice_caption": global_caption, "burst_captions": events,
            "locator_settings": {"threshold": args.threshold, "min_duration_s": .08,
                                 "nms_overlap_fraction": .5, "event_cap": None}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["acoustic", "moss"])
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--acoustic", type=Path)
    p.add_argument("--language", default="en")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--threshold", type=float, default=.6)
    for option in ["procedural-dir", "humaneness-dir", "parakeet-dir", "voice-dir", "burst-dir"]:
        p.add_argument("--"+option)
    args = p.parse_args()
    if not 0 <= args.threshold <= 1:
        p.error("threshold must lie in [0, 1]")
    if args.stage == "moss" and args.acoustic is None:
        p.error("MOSS needs the --acoustic stage sidecar")
    result = acoustic(args) if args.stage == "acoustic" else moss(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
