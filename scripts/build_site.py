"""Build the static site with original inline MP3s and exact evaluated outputs.

The local manifest supplies paths; none of those paths are emitted to the site.
The listening page contains its own CSS, JS, diagram and audio for offline use.
"""
import argparse
import base64
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanness_voice_annotator.formatting import sources
from site_sections import PAGES

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'composite': 'Weighted composite',
    'global_caption_fidelity': 'Whole-recording caption fidelity',
    'global_emotion_fit': 'Whole-recording emotion fit',
    'sentence_emotion_fit': 'Local speech emotion fit',
    'utterance_talking_style_fit': 'Whole-recording talking-style fit',
    'sentence_delivery_fit': 'Local speech delivery fit',
    'burst_description_fit': 'Burst-description fit',
    'burst_timing_fit': 'Burst-timing fit',
    'event_precision': 'Event precision rating',
    'event_coverage': 'Event coverage rating',
    'transcript_fidelity': 'Transcript fidelity (diagnostic)',
    'speech_timing_fit': 'Speech-timing fit (diagnostic)',
    'readability': 'Readability (diagnostic)',
}


def e(value):
    return escape(str(value), quote=True)


def document(title, body, *, listening=False):
    if listening:
        styles = '<style>'+ (ROOT/'docs/assets/site.css').read_text()+'</style>'
        javascript = '<script>'+ (ROOT/'docs/assets/site.js').read_text()+'</script>'
    else:
        styles = '<link rel="stylesheet" href="assets/site.css">'
        javascript = '<script src="assets/site.js" defer></script>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)} · Humanness Voice Annotator</title>
<meta name="description" content="Listen to reproducible voice annotations combining MOSS captions, Humaneness Ears scores, event locations and timed ASR words.">
{styles}</head><body><a href="#main" class="skip">Skip to content</a><header><nav aria-label="Main navigation">
<a class="brand" href="index.html">Humanness Voice Annotator</a><a href="index.html#listening">Listen</a>
<a href="format.html">Format</a><a href="models.html">Models &amp; training</a><a href="pipeline.html">Run the pipeline</a>
<a href="evaluation.html">Evaluation</a><a href="https://github.com/LAION-AI/Humanness-Voice-Annotator">GitHub ↗</a></nav></header>
<main id="main">{body}</main><footer>LAION · 11 October 2026 · Original code, documentation and new annotations: CC BY 4.0 ·
<a href="https://github.com/LAION-AI/Humanness-Voice-Annotator/blob/main/THIRD_PARTY.md">Upstream credits and retained licenses</a></footer>{javascript}</body></html>'''


def metrics_table(summary):
    rows = []
    rubric = summary['rubric']
    for key, value in summary['results'].items():
        lo, hi = value['bootstrap_95_ci']
        mean = value['mean']
        digits = 3 if key == 'composite' else 2
        description = ('Fixed weighted score, equal weight per recording.' if key == 'composite'
                       else rubric['definitions'][key])
        weight = rubric['composite_weights'].get(key)
        rows.append(f'<tr><td class="metric">{e(LABELS[key])}</td><td class="number"><strong>{mean:.{digits}f} / 5</strong>'
                    f'<div class="meter" aria-hidden="true"><span style="width:{mean/5*100:.2f}%"></span></div></td>'
                    f'<td class="number">{lo:.{digits}f}–{hi:.{digits}f}</td><td>{"100%" if key == "composite" else (f"{weight*100:.0f}%" if weight is not None else "Diagnostic")}</td>'
                    f'<td class="description">{e(description)}</td></tr>')
    return '<div class="table-wrap"><table><thead><tr><th scope="col">Criterion</th><th scope="col">Mean</th><th scope="col">95% clip interval</th><th scope="col">Weight</th><th scope="col">How to read it</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def timeline(annotation, review):
    duration = annotation['audio']['duration_s']
    x = lambda sec: 145 + 830 * sec / duration
    parts = ['<svg class="timeline" viewBox="0 0 1000 155" role="img" aria-label="Absolute speech and vocal-event timeline in seconds">']
    tracks = [(28, 'Speech segments', annotation['segments'], 'speech'),
              (66, 'Selected events', annotation['burst_events'], 'event'),
              (104, 'Reviewer proposals', review['answer']['audible_events'], 'review-event')]
    for y, label, intervals, style in tracks:
        parts.append(f'<text x="0" y="{y+15}">{e(label)}</text><rect class="track" x="145" y="{y}" width="830" height="24" rx="4"/>')
        for item in intervals:
            start, end = item['start_s'], item['end_s']
            text = item.get('text', item.get('caption', item.get('type', '')))
            hint = f'{start:.2f}–{end:.2f} s: {text}'
            parts.append(f'<rect class="{style}" x="{x(start):.3f}" y="{y}" width="{max(1.,x(end)-x(start)):.3f}" height="24" rx="2" data-start="{start:.6f}" tabindex="0" role="button" aria-label="Play from {e(hint)}"><title>{e(hint)}</title></rect>')
    for i in range(7):
        sec = duration * i / 6
        parts.append(f'<text x="{x(sec):.1f}" y="147" text-anchor="middle">{sec:.1f}s</text>')
    return ''.join(parts)+'</svg>'


def scalar_table(annotation):
    prediction = annotation['raw_model_outputs']['humaneness_ears_medium']
    raw, z = prediction['scores_raw'], prediction.get('scores_reference_z', {})
    names = sorted(raw, key=lambda key: (0 if key.startswith('emo_') else 1 if key.startswith('vn_') else 2, key))
    rows = [f'<tr><td><code>{e(name)}</code></td><td class="number">{raw[name]:.4f}</td><td class="number">{z[name]:+.3f}</td></tr>' if name in z
            else f'<tr><td><code>{e(name)}</code></td><td class="number">{raw[name]:.4f}</td><td>Unavailable</td></tr>' for name in names]
    for name in ['characters_per_second', 'estimated_burst_count']:
        value = prediction.get(name)
        normalized = prediction.get('auxiliary_reference_z', {}).get(name)
        if value is not None:
            rows.append(f'<tr><td><code>{e(name)}</code> (auxiliary)</td><td>{value:.4f}</td><td>{normalized:+.3f}</td></tr>' if isinstance(normalized, (int,float))
                        else f'<tr><td><code>{e(name)}</code> (auxiliary)</td><td>{value:.4f}</td><td>Unavailable</td></tr>')
    return '<p class="small">Raw values retain each target\'s own units. Reference z-scores use the speech normalizer, not the training-target normalizer. Negative values are preserved. Full embeddings, frames, local raw scores and all proposals are in the downloadable JSON.</p><div class="table-wrap"><table><thead><tr><th>Scalar target</th><th>Raw prediction</th><th>Speech-reference z</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def sample_card(row, annotation, review):
    sid = row['id']
    original = Path(row['audio_path']).read_bytes()
    assert hashlib.sha256(original).hexdigest() == annotation['audio']['sha256'] == review['audio_sha256']
    assert annotation['annotation_sha256'] == review['annotation_sha256']
    payload = base64.b64encode(original).decode()
    metadata = annotation['audio']['source_metadata']
    title = metadata.get('title', sid)
    language = {'en':'English', 'de':'German'}.get(row['language'], row['language'])
    sentence_reviews = {r['segment_id']:r for r in review['answer']['sentence_reviews']}
    segment_rows = []
    for segment in annotation['segments']:
        local = sentence_reviews[segment['id']]
        segment_rows.append(f'<tr><td class="number"><button data-start="{segment["start_s"]:.6f}">{segment["start_s"]:.2f}–{segment["end_s"]:.2f}s ▶</button><br>{e(segment["id"])}<br>Source sentence {segment["source_sentence_index"]+1 if segment["source_sentence_index"] is not None else "unavailable"}</td>'
                            f'<td>{e(segment["text"])}<br><span class="small">{e(segment["cue"])}</span></td>'
                            f'<td class="number">Emotion {local["emotion_fit"]}/5<br>Delivery {local["delivery_fit"]}/5</td><td>{e(local["note"])}</td></tr>')
    event_rows = []
    for event in annotation['burst_events']:
        classes = '; '.join(f'{item["label"]}: {item["softmax_probability"]*100:.1f}%' for item in event['top3_classes'])
        event_rows.append(f'<div class="event-row"><button data-start="{event["start_s"]:.6f}">{event["start_s"]:.2f}–{event["end_s"]:.2f} s ▶</button> <span class="small">{event["duration_s"]:.2f} s duration · {e(event["id"])}</span><p>{e(event["caption"])}</p><div class="classes">Top-three type softmax: {e(classes) if classes else "unavailable"}</div></div>')
    local_ratings = ''.join(f'<tr><td>{e(LABELS[key])}</td><td class="number">{value}/5</td></tr>' for key,value in review['answer']['ratings'].items())
    search_text = ' '.join([sid,title,row['language'],metadata.get('voice',''),annotation['general'],annotation['script']]).lower()
    return f'''<article class="sample" id="sample-{e(sid)}" data-language="{e(row['language'])}" data-search="{e(search_text)}">
<div class="sample-head"><div><span class="badge">{e(language)}</span><span class="badge">{len(annotation['burst_events'])} selected intervals</span><h2>{e(title)}</h2>
<div class="meta">{e(sid)} · {row['duration_s']:.2f} s · Voice: {e(metadata.get('voice','not recorded'))} · {e(metadata.get('source_subset',''))}</div></div>
<div class="review-score">{review['weighted_score']:.2f} / 5<small>Automated composite</small></div></div>
<audio controls preload="none" aria-label="Original audio for {e(sid)}"><source src="data:audio/mpeg;base64,{payload}" type="audio/mpeg">Your browser does not support HTML audio.</audio>
<p class="small">Original audio bytes, not regenerated. Select a colored interval or a timestamp to play from that point.</p>
{timeline(annotation,review)}<div class="legend"><span><i class="dot"></i>Speech</span><span><i class="dot burst"></i>Selected Ears intervals</span><span><i class="dot judge"></i>Reviewer event proposals (not ground truth)</span></div>
<h3>GENERAL — original whole-recording MOSS caption</h3><p>{e(annotation['general'])}</p>
<h3>SCRIPT — timed speech, local cues and free-form burst descriptions</h3><pre class="code-script">{e(annotation['script'])}</pre>
<div class="review-note"><strong>Automated review note</strong><br>{e(review['answer']['overall_note'])}</div>
<details><summary>Selected vocal events: exact starts/ends and top-three types</summary><p class="small">These are detector-selected intervals. MOSS prose describes the full-audio interval; softmax percentages indicate type ranking and are not calibrated correctness probabilities.</p><div class="event-list">{''.join(event_rows) or '<p>No interval selected.</p>'}</div></details>
<details><summary>Local emotion and delivery review — {len(annotation['segments'])} speech segments</summary><div class="table-wrap"><table><thead><tr><th>Absolute interval</th><th>Recognized speech and cue</th><th>Gemini ratings</th><th>Reviewer explanation</th></tr></thead><tbody>{''.join(segment_rows)}</tbody></table></div></details>
<details><summary>All recording-level review ratings</summary><div class="table-wrap"><table><thead><tr><th>Criterion</th><th>Rating</th></tr></thead><tbody>{local_ratings}</tbody></table></div></details>
<details><summary>All 192 raw scalar predictions, reference z-scores and auxiliary targets</summary>{scalar_table(annotation)}</details>
<details><summary>Provenance and format verification</summary><p>Schema: <code>{e(annotation['schema_version'])}</code><br>Audio SHA-256: <code>{e(annotation['audio']['sha256'])}</code><br>Annotation SHA-256: <code>{e(annotation['annotation_sha256'])}</code></p><p>{len(annotation['words'])} words preserved; {len(annotation['burst_events'])} selected events serialized; printed elapsed budget {annotation['timing']['independent_verification']['printed_budget_s']:.3f} s. Independent format proof: <strong>passed</strong>. This does not establish acoustic correctness.</p><p>Voice and source metadata are preserved. Full raw predictions and embeddings are in <a href="data/annotations.jsonl.gz">canonical annotations</a>; reviewer response and token usage are in <a href="data/reviews.jsonl">all reviews</a>.</p></details></article>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    out = ROOT/'docs'
    summary = json.loads((out/'data/evaluation-summary.json').read_text())
    reviews = {r['id']:r for r in map(json.loads,(out/'data/reviews.jsonl').read_text().splitlines())}
    rows = list(map(json.loads, args.manifest.read_text().splitlines()))
    assert len(rows) == len(reviews) == 50
    rows.sort(key=lambda row:(row['language']!='en',row['id']))
    annotations = [json.loads(Path(row['annotation_path']).read_text()) for row in rows]
    for key,(title,body) in PAGES.items():
        extra = metrics_table(summary) if key == 'evaluation' else ''
        (out/(key+'.html')).write_text(document(title,'<div class="hero"><div class="eyebrow">An open, reproducible voice annotation pipeline</div><h1>'+e(title)+'</h1></div>'+extra+body))
    shutil.copyfile(ROOT/'evaluation/rubric.json',out/'data/rubric.json')
    (out/'data/sources.json').write_text(json.dumps(sources(),indent=2)+'\n')
    diagram = (out/'assets/pipeline.svg').read_text().replace('<svg ', '<svg class="pipeline" ', 1)
    body = '''<div class="hero"><div class="eyebrow">LAION · listen, inspect, reproduce</div><h1>A voice has more to say<br>than its words.</h1><p class="lead">Humanness Voice Annotator turns audio into a whole-recording description and a timed performance script: spoken words, local delivery and emotion cues, free-form vocal events and pauses.</p><div class="buttons"><a class="button primary" href="#listening">Listen to all 50 examples</a><a class="button" href="format.html">Understand the format</a><a class="button" href="https://github.com/LAION-AI/Humanness-Voice-Annotator">Code and README ↗</a></div><p class="small">Original audio is embedded as Base64 MP3. No model or API runs in your browser. The downloadable listening HTML includes its own styles, diagram and players.</p></div>
<div class="stats"><div class="stat"><strong>50</strong><span>recordings · 45 English / 5 German</span></div><div class="stat"><strong>662</strong><span>timed speech segments</span></div><div class="stat"><strong>360</strong><span>selected vocal-event intervals</span></div><div class="stat"><strong>3.263 / 5</strong><span>automated review composite</span></div></div>
<section><h2>Four model roles, one documented output</h2>'''+diagram+'''<p><a href="https://huggingface.co/laion/humaneness-ears-base-medium">Humaneness Ears Medium</a> supplies acoustic scores and event proposals. <a href="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3">Parakeet v3</a> supplies recognized words and timestamps. <a href="https://huggingface.co/laion/moss-voice-captioner">MOSS voice-captioner</a> describes the whole recording, and <a href="https://huggingface.co/laion/moss-vocal-burst-captioner">MOSS burst-captioner</a> describes each selected interval. <a href="https://github.com/LAION-AI/procedural-voice-captions">Procedural rules</a> normalize and select local cues, plan the timeline and preserve overlap.</p><p>Humaneness Ears is a Whisper encoder with task heads, <strong>not an ASR decoder</strong>. MOSS event descriptions receive the full audio and both endpoints; MOSS does not discover those endpoints. This release connects existing models and preserves their raw predictions.</p><div class="buttons"><a class="button" href="models.html">Models and training details</a><a class="button" href="pipeline.html">Reproduce the pipeline</a><a class="button" href="https://laion-moss-voice-acting-captions.static.hf.space/format-comparison.html">Earlier MOSS/procedural comparison ↗</a></div></section>
<section><h2>How to read GENERAL / SCRIPT</h2><div class="columns"><div><h3>GENERAL</h3><p>The original full-recording MOSS caption: voice, performance and audible environment. It is preserved even when a claim is incorrect.</p><h3>SCRIPT</h3><p>Recognized speech with a round-bracket delivery cue and duration before each timed segment. Free-form vocal descriptions appear at their approximate word-gap positions. Square brackets mark alignment gaps.</p></div><div><pre>GENERAL: [Original whole-recording caption]
SCRIPT:
(quiet, hesitant; 1.20 seconds duration) I… wasn't sure.
[0.30 seconds pause]
("A short inhalation.", 0.40 seconds)
(warmer, relieved; 2.30 seconds duration) ("A soft laugh.", 0.50 seconds) But I'm glad you came.</pre><p class="small">Illustrative syntax; not an evaluated example. Total: 4.20 s. The inline laugh is included in its speech budget.</p></div></div><p>Use the JSON for exact event starts/ends and overlaps. A printed duration is a formatting budget, not proof that a detector proposal is real. Segments may be shorter than grammatical sentences. <a href="format.html">Read the complete contract →</a></p></section>
<section id="results"><h2>Audio-based evaluation of the final format</h2><p>Gemini 3.8 Flash listened to all 50 original recordings and reviewed the emitted annotations. It assessed global emotion, local emotion, talking style, vocal descriptions and timing separately. Intended TTS scripts and target emotions were withheld. Higher is better on a 0–5 scale.</p>'''+metrics_table(summary)+'''<div class="notice"><strong>What the scores show.</strong> Global acoustic descriptions and talking style are stronger than local emotion specificity and event timing. Several fine-grained emotion labels are misplaced; some event intervals are redundant or include unrelated speech. All original outputs and reviewer notes remain visible below.</div><p class="small">Single automated reviewer; small, reused synthetic listening set; no human adjudication. 95% intervals use 10,000 recording-level bootstrap resamples and exclude reviewer bias. Precision/coverage are ordinal ratings, not detection rates. New criteria/weights differ from earlier comparisons, so the composite is not a direct before/after number.</p><p><strong>Formatting checks:</strong> all 50 pass. Every recognized word is retained once (2,703 total), all selected events are serialized once, and elapsed budgets agree. Acoustic accuracy is assessed separately. The 662 script segments derive from 443 source sentences and reuse source-sentence scores after splitting.</p><div class="buttons"><a class="button" href="evaluation.html">Rubric, method and interpretation</a><a class="button" href="data/evaluation-summary.json">Full result JSON</a><a class="button" href="data/reviews.jsonl">All 50 reviews</a><a class="button" href="data/annotations.jsonl.gz">Canonical sidecars</a></div><p class="small">Estimated review list-price cost: $0.437 from returned usage. Annotation inference itself uses released local models and requires no Gemini API.</p></section>
<section id="listening"><h2>Listen and inspect the predictions</h2><p>English examples come first, followed by German. The players contain the original synthetic acting audio. Voice name, source subset, timestamps and hashes remain available. You can seek through the timeline and inspect every local review.</p><p><a href="https://huggingface.co/datasets/laion/german-english-voice-acting-challenges">Download the voice-acting dataset ↗</a> · <a href="https://github.com/LAION-AI/Humanness-Voice-Annotator/tree/main/examples">Reproduce a saved example ↗</a></p></section>
<div class="filters"><label for="language">Language</label><select id="language"><option value="all">All languages</option><option value="en">English</option><option value="de">German</option></select><label class="skip" for="search">Search examples</label><input id="search" type="search" placeholder="Search voice, challenge, caption or event" aria-label="Search examples"><span id="visible-count" aria-live="polite">50 of 50 recordings</span></div>'''
    body += ''.join(sample_card(row, annotation, reviews[row['id']]) for row,annotation in zip(rows,annotations))
    (out/'index.html').write_text(document('Listen to the default GENERAL / SCRIPT annotations',body,listening=True))
    (out/'.nojekyll').touch()
    print(json.dumps({'recordings':len(rows),'inline_mp3_players':len(rows),'segments':sum(len(a['segments']) for a in annotations),
                      'bytes':(out/'index.html').stat().st_size,'all_audio_and_annotation_hashes_match_reviews':True}))


if __name__ == '__main__':
    main()
