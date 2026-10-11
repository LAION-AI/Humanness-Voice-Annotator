"""Audit completed worker proofs and create a portable, additive token release."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    lock = json.loads((args.export/'export-lock.json').read_text())
    if lock['sample_limit'] or lock['shard_limit_per_rank']:
        p.error('A pilot cannot be published as the complete dataset')
    lockhash = hashlib.sha256(json.dumps(lock,ensure_ascii=False,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
    inventory = json.loads((args.source/'inventory.json').read_text())
    if hashlib.sha256((args.source/'inventory.json').read_bytes()).hexdigest()!=lock['input_inventory_sha256']:
        raise ValueError('Source inventory changed')
    expected = {r['id']:r for r in map(json.loads, (args.source/'index.jsonl').read_text().splitlines())}
    rows = []; output_inventory = []; totals = Counter(); groups = {}; seen = set()
    for i, source in enumerate(inventory):
        proof = json.loads((args.export/'state'/f'shard-{i:05d}.json').read_text())
        target = args.export/proof['path']
        if (proof['source'] != source or proof['export_lock_sha256'] != lockhash
                or proof['n'] != source['n_samples'] or len(proof['index']) != proof['n']
                or target.stat().st_size != proof['size_bytes']):
            raise ValueError('Incomplete or incompatible shard proof')
        for row in proof['index']:
            key = row['id']
            if key in seen or key not in expected:
                raise ValueError('Duplicate or unexpected sample')
            seen.add(key)
            original = expected[key]
            for field in ['language','voice','source_subset','audio_sha256','duration_s']:
                if row[field] != original[field]:
                    raise ValueError('Original sample metadata changed: '+field)
            group = row['challenge_group']
            if groups.setdefault(group, row['split']) != row['split']:
                raise ValueError('Scenario leakage across partitions')
            if group in lock['heldout_groups'] and row['split'] != 'test':
                raise ValueError('Listening demo scenario was not reserved')
            rows.append(row)
        output_inventory.append({k:proof[k] for k in ['path','n','sha256','size_bytes']})
        totals.update(proof['totals'])
    if seen != set(expected):
        raise ValueError('Corpus coverage mismatch')
    splits = {}
    for split in ['train','validation','test']:
        subset = [r for r in rows if r['split']==split]
        splits[split] = {'n_samples':len(subset),'audio_hours':sum(r['duration_s'] for r in subset)/3600,
                         'challenge_groups':len({r['challenge_group'] for r in subset})}
    summary = {'schema':'humanness-voice-tuning-summary-v1','n_samples':len(rows),'n_shards':len(inventory),
               'audio_hours':totals['seconds']/3600,'codec_frames':totals['frames'],
               'codec_indices':totals['frames']*12,'selected_burst_events':totals['events'],
               'asr_words':totals['words'],'asr_empty_samples':sum(r['asr_empty'] for r in rows),
               'bytes':sum(r['size_bytes'] for r in output_inventory),'splits':splits,
               'languages':dict(Counter(r['language'] for r in rows)),
               'voices':dict(sorted(Counter(r['voice'] for r in rows).items())),
               'source_subsets':dict(sorted(Counter(r['source_subset'] for r in rows).items())),
               'heldout_demo_challenge_groups':len(lock['heldout_groups']),
               'verified':{'full_source_coverage':True,'unchanged_audio_hashes':True,
                           'unchanged_voice_language_subset_duration':True,'no_scenario_split_leakage':True},
               'export_lock_sha256':lockhash}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'data').mkdir(exist_ok=True)
    for entry in output_inventory:
        target=args.output/entry['path']
        if target.exists():
            if target.stat().st_size!=entry['size_bytes']:
                raise ValueError('Release file already exists with different size')
        else:
            target.hardlink_to(args.export/entry['path'])
    rows.sort(key=lambda r:r['id'])
    (args.output/'index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))
    write(args.output/'inventory.json',output_inventory)
    write(args.output/'summary.json',summary)
    shutil.copy2(args.export/'export-lock.json',args.output/'export-lock.json')
    write(args.output/'heldout-groups.json',lock['heldout_groups'])
    split_table='\n'.join(f"| {s} | {v['n_samples']:,} | {v['audio_hours']:.2f} | {v['challenge_groups']:,} |" for s,v in splits.items())
    (args.output/'README.md').write_text(f'''# Humaneness Voice tuning annotations and MOSS v2 audio tokens

This is an **additive training export** for all {len(rows):,} synthetic German/English acting clips ({summary['audio_hours']:.2f} hours). It is ready for Humaneness Voice Small data loading; this export does **not** report a new TTS training run.

[Export code and guide](https://github.com/LAION-AI/Humanness-Voice-Annotator/blob/main/TRAINING_DATA.md) · [Listen to the format](https://projects.laion.ai/Humanness-Voice-Annotator/) · [Original dataset](https://huggingface.co/datasets/laion/german-english-voice-acting-challenges)

## Payload

Each of {len(inventory)} WebDataset TARs contains three members per sample:

| Member | Contents |
|---|---|
| `ID.json` | GENERAL/SCRIPT training prompt, ASR text, split, voice/generation provenance, frame budget and hashes |
| `ID.codes.npy` | Little-endian int16 `[frames, 12]` targets; codes 0–1023; first 12 RVQ codebooks |
| `ID.annotation.json.gz` | Full canonical caption, exact intervals, original raw acoustic scores/embeddings, local cues, MOSS outputs and independent format proof |

Original 128-kbit/s mono MP3s and original annotations remain in the parent `data/` archives. Join using `id`, source revision and audio SHA-256; audio is not duplicated here. All nine existing procedural variants remain available in the original records. `caption_transcript_prompt` additionally preserves the plain whole-recording CAPTION/TRANSCRIPT surface. The default new prompt uses GENERAL/SCRIPT, with round-bracket durations for speech/bursts and square-bracket pauses.

## Audio tokenizer

[OpenMOSS-Team/MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2), pinned revision `{lock['codec']['revision']}`. Decode/resample the original MP3 to 48 kHz, duplicate mono into stereo, and explicitly right-pad each waveform to a multiple of 3,840 samples. Store the true sample count and padding count. Encode 12 codebooks with fp32 weights/quantizer, bf16 compute and SDPA, using persistent GPU workers and 4.8-second streaming chunks. One frame is **80 ms**, and the Humaneness Voice `Tokens` field is the **frame count**, not twelve times that count. Decoded audio must be trimmed to `original_samples_48khz`. The arrays contain no stop/padding token; the native training processor adds and masks those tokens.

## Splits

| Split | Clips | Audio hours | Challenge groups |
|---|---:|---:|---:|
{split_table}

A deterministic challenge-level hash targets 96% train / 2% validation / 2% test, with the 50 previously reviewed listening scenarios additionally forced into test. Both languages of a challenge stay together. Voices occur in multiple splits: this is a **scenario holdout**, not a speaker holdout. The original synthetic clips are already public; these are preparation splits, not evidence of unseen-data generalization for a future model.

## Provenance and checks

The original dataset revision is `{lock['source_revision']}`. `export-lock.json` freezes code hashes, model artifacts, baseline/configuration and split inputs. `inventory.json` contains archive SHA-256; each row has token and annotation payload hashes. All source IDs, audio hashes, language, voice, subset and durations were checked for exact coverage. Each generated annotation passed the independent timing/word/event verifier. No intended generation script replaces the recognized words; the single empty ASR record is marked. Selected events are machine proposals, not human ground truth.

**{summary['codec_frames']:,} frames**, **{summary['codec_indices']:,} code indices**, **{summary['selected_burst_events']:,} selected events**, **{summary['asr_words']:,} ASR words**. See `summary.json` for exact per-language/per-voice counts. The release has {summary['bytes']/1e9:.2f} GB of additive TAR payloads.

## Licensing

New captions, metadata, code and released dataset annotations: **CC BY 4.0**. Original data licensing and source provenance are retained. Upstream neural model/code licenses remain their own; the MOSS Audio Tokenizer release uses Apache 2.0. Credit LAION and all linked component authors. Review synthetic generation metadata and upstream terms for your use case.
''')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
