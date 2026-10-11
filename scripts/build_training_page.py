"""Publish audited tuning-data statistics as an explanatory static page."""
import argparse
from html import escape
import json
from pathlib import Path
import shutil
from build_site import document, ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',required=True,type=Path)
    p.add_argument('--preflight',required=True,type=Path)
    args=p.parse_args()
    summary=json.loads(args.summary.read_text())
    compute=json.loads((ROOT/'docs/data/training-compute-report.json').read_text())
    splits=''.join(f"<tr><td>{escape(k.title())}</td><td>{v['n_samples']:,}</td><td>{v['audio_hours']:.2f}</td><td>{v['challenge_groups']:,}</td></tr>" for k,v in summary['splits'].items())
    subsets=''.join(f"<tr><td>{escape(k.replace('_',' ').title())}</td><td>{v:,}</td></tr>" for k,v in summary['source_subsets'].items())
    body=f'''<div class="hero"><div class="eyebrow">LAION · inspect, reproduce, tune</div>
<h1>From an acting recording<br>to a training example.</h1>
<p class="lead">A timed GENERAL/SCRIPT instruction describes the performance. Twelve audio-codebook targets represent the sound. This additive release pairs both for every German/English acting recording.</p>
<div class="buttons"><a class="button primary" href="https://huggingface.co/datasets/laion/german-english-voice-acting-challenges/tree/main/tuning">Download tuning data ↗</a><a class="button" href="https://github.com/LAION-AI/Humanness-Voice-Annotator/blob/main/TRAINING_DATA.md">Code and full loading guide ↗</a><a class="button" href="index.html#listening">Hear the annotation format</a></div></div>
<div class="stats"><div class="stat"><strong>{summary['n_samples']:,}</strong><span>complete recordings</span></div><div class="stat"><strong>{summary['audio_hours']:.2f} h</strong><span>English and German</span></div><div class="stat"><strong>30</strong><span>preserved generation voices</span></div><div class="stat"><strong>{summary['n_shards']}</strong><span>additional WebDataset TARs</span></div></div>
<section><h2>What is being prepared?</h2><p>This is data preparation for <a href="https://huggingface.co/laion/humaneness-voice-small">Humaneness Voice Small</a>, a voice-generation model. It is separate from <a href="https://huggingface.co/laion/humaneness-ears-base-medium">Humaneness Ears</a>, the encoder used to estimate scores and locate vocal events. Preparing these examples does not report a new TTS training run or demonstrate a quality gain.</p>
<div class="table-wrap"><table><thead><tr><th>Instruction source</th><th>Role</th></tr></thead><tbody>
<tr><td>MOSS voice-captioner</td><td>Whole-recording GENERAL description</td></tr><tr><td>MOSS burst-captioner</td><td>Free-form descriptions of selected intervals, using the full audio and both endpoints</td></tr>
<tr><td>Humaneness Ears Medium + procedural rules</td><td>Event proposals, reference-relative emotion/style cues and local delivery descriptions</td></tr><tr><td>Parakeet v3</td><td>Recognized words and alignment endpoints</td></tr>
<tr><td>MOSS Audio Tokenizer v2</td><td>Discrete audio targets; a different model from the two captioners</td></tr></tbody></table></div>
<p>Existing, complete captions and acoustic predictions are reused. No new generative caption-model pass was needed. The original MP3s, annotation variants, generation prompts and exact voice IDs remain available. New local cue wording follows the frozen listening-demo format and retains all raw scores in the sidecar.</p></section>
<section><h2>One clip, two representations</h2><svg class="pipeline" viewBox="0 0 960 255" role="img" aria-label="Original recording branches to a timed performance prompt and codec targets, then joins into a training example"><defs><marker id="arrow-t" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8" fill="#7dd3fc"/></marker></defs>
<g fill="#14233d" stroke="#7dd3fc" stroke-width="2"><rect x="10" y="85" width="205" height="85" rx="14"/><rect x="290" y="20" width="305" height="85" rx="14"/><rect x="290" y="150" width="305" height="85" rx="14"/><rect x="690" y="85" width="250" height="85" rx="14"/></g>
<g fill="#f8fafc" font-family="sans-serif" font-size="18" text-anchor="middle"><text x="112" y="119">Original mono MP3</text><text x="112" y="148">voice + scenario metadata</text><text x="442" y="53">GENERAL / SCRIPT</text><text x="442" y="81">words, cues, events, durations</text><text x="442" y="181">MOSS v2 · 48 kHz</text><text x="442" y="211">[frames, 12] audio targets</text><text x="815" y="119">One training record</text><text x="815" y="148">JSON + NPY + sidecar</text></g>
<g fill="none" stroke="#7dd3fc" stroke-width="2" marker-end="url(#arrow-t)"><path d="M215,110 L290,65"/><path d="M215,145 L290,193"/><path d="M595,65 L690,110"/><path d="M595,193 L690,145"/></g></svg>
<p>The full GENERAL/SCRIPT is the model instruction. The SCRIPT also occupies the native Text field. The frame budget is <strong>T</strong>, while the audio targets have shape <strong>[T,12]</strong>. The native processor creates 13 channels—one text/continuation channel and twelve codebooks—and masks instruction/reference/padding positions from loss.</p></section>
<section><h2>Audio token contract</h2><div class="table-wrap"><table><thead><tr><th>Property</th><th>Value and meaning</th></tr></thead><tbody>
<tr><td>Codec</td><td><a href="https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2">MOSS Audio Tokenizer v2</a>; pinned revision in the export lock</td></tr><tr><td>Timing</td><td>48 kHz; one frame per 3,840 samples, or 80 ms</td></tr>
<tr><td>Depth</td><td>First 12 RVQ codebooks, matching Humaneness Voice Small</td></tr><tr><td>Storage</td><td>Little-endian int16 NumPy [T,12]; indices 0–1023</td></tr><tr><td>Tail handling</td><td>Explicit zero-padding of less than one frame; exact true length saved for trimming decoded audio</td></tr><tr><td>Compute</td><td>Persistent workers, four CPU threads/GPU, batch eight, 4.8-second streaming chunks, FP32 weights/quantizer and BF16 compute</td></tr></tbody></table></div>
<p>Mono is duplicated into two identical channels for the stereo codec. No recording ends are dropped and no loudness normalization is added. Original MP3 bytes remain unchanged in the source archives; codec reconstruction is lossy. STOP and PAD are created by the training processor rather than stored among physical code indices.</p></section>
<section><h2>Scenario-level partitions</h2><div class="table-wrap"><table><thead><tr><th>Split</th><th>Clips</th><th>Hours</th><th>Challenge groups</th></tr></thead><tbody>{splits}</tbody></table></div>
<p>Seeded hashing aims for 96% training, 2% validation and 2% test. The 50 earlier listening-demo scenarios are additionally held out for test. English/German counterparts stay together. These are scenario partitions; the 30 voices appear across splits. Related scenario semantics may still overlap, and the released recordings are already public. These preparation splits do not establish independent evaluation for the component annotation models.</p></section>
<section><h2>Contents and verification</h2><p><strong>{summary['codec_frames']:,} frames</strong> contain <strong>{summary['codec_indices']:,} codebook indices</strong>. The canonical captions retain <strong>{summary['selected_burst_events']:,} selected event intervals</strong> and <strong>{summary['asr_words']:,} recognized words</strong>. The additive TAR payload is {summary['bytes']/1e9:.2f} GB; original audio is joined by ID and SHA-256 rather than duplicated.</p>
<p>The encoder validated shapes/ranges and audio durations. Each caption passed independent word/event/timing checks. A 128-clip pilot passed, including four codec reconstruction canaries. Native packing was checked with and without reference audio, confirming T+1 target steps and excluding reference STOP labels from supervision. One empty ASR output is explicitly marked, preserving provenance rather than replacing it with an intended script.</p>
<p>Full source coverage, unchanged source metadata and absence of scenario split leakage were audited before release. Mechanical validity does not imply correct emotions or event timing. The earlier 50-example <a href="evaluation.html">audio-based automated review</a> describes the format's perceptual limitations.</p>
<div class="buttons"><a class="button" href="data/training-summary.json">Exact corpus statistics</a><a class="button" href="data/training-preflight.json">Packing checks</a></div></section>
<section><h2>Measured preparation time</h2><div class="table-wrap"><table><thead><tr><th>Measurement</th><th>Result</th><th>Scope</th></tr></thead><tbody>
<tr><td>Full production job</td><td>15 min 18 s</td><td>4 nodes × 4 GH200 GPUs; four CPU threads per GPU</td></tr>
<tr><td>Production allocation</td><td>{compute['production_allocated_gpu_hours']:.2f} GPU-hours</td><td>Includes startup and idle allocation</td></tr>
<tr><td>Including both pilots</td><td>{compute['total_allocated_gpu_hours_including_pilots']:.2f} GPU-hours</td><td>Includes the initial tail-padding failure and successful corrected pilot</td></tr>
<tr><td>End-to-end preparation throughput</td><td>{compute['production_clips_per_s_including_startup']:.2f} clips/s</td><td>All 16 GPUs together, including job startup; excludes publication</td></tr></tbody></table></div>
<p>These measurements encode audio and recompose/compress already-complete cached annotations. They do not measure fresh MOSS caption generation or TTS training. Allocated GPU-hours are resource accounting, not a monetary invoice. <a href="data/training-compute-report.json">Read the exact run record</a>.</p></section>
<section><h2>Acting coverage</h2><div class="table-wrap"><table><thead><tr><th>Source subset</th><th>Successful clips</th></tr></thead><tbody>{subsets}</tbody></table></div>
<p>New annotations, documentation and code are CC BY 4.0. Component-model licenses remain their own; the codec retains Apache 2.0. See the <a href="https://github.com/LAION-AI/Humanness-Voice-Annotator/blob/main/THIRD_PARTY.md">credits and source licenses</a>.</p></section>'''
    (ROOT/'docs/training.html').write_text(document('Humaneness Voice tuning data',body))
    shutil.copy2(args.summary,ROOT/'docs/data/training-summary.json')
    shutil.copy2(args.preflight,ROOT/'docs/data/training-preflight.json')


if __name__=='__main__':
    main()
