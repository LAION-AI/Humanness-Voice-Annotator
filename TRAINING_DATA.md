# Preparing Humaneness Voice tuning data

[Listen to the format](https://projects.laion.ai/Humanness-Voice-Annotator/) · [Download the corpus](https://huggingface.co/datasets/laion/german-english-voice-acting-challenges/tree/main/tuning) · [Humaneness Voice Small](https://huggingface.co/laion/humaneness-voice-small)

This export joins complete, saved annotations with **MOSS Audio Tokenizer v2 targets**. It prepares data for later text-to-speech/voice-acting tuning. It does not start a neural-model training run or claim a resulting quality improvement.

## Two independent representations

1. **Human-readable performance conditioning:** the default `GENERAL / SCRIPT` format from the listening demo. MOSS supplies the whole-recording caption and interval descriptions. Humaneness Ears Medium plus procedural reference normalization supplies compact local delivery/emotion cues. Parakeet v3 supplies recognized words and their endpoints. Each speech segment and burst has a duration; pauses are square-bracket gaps. Absolute event starts/ends, overlaps and every raw prediction remain in JSON.
2. **Neural audio targets:** `[frames, 12]` integer codes from the first 12 RVQ codebooks of [MOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2). This codec is distinct from the two MOSS **captioning** adapters. The arrays supervise generated audio; the caption models supply instruction text.

Reusing completed predictions avoids running the captioning models again. Original MP3s, voice IDs, generation prompts and existing caption variants are preserved. The new fields are additive. Captions and events are machine annotations and may be wrong; mechanical format validity is not a perceptual quality score.

A real English training example is included under [`examples/tuning/`](examples/tuning/): lightweight JSON, 251 × 12 codec codes and its compressed complete annotation. It comes from the published source audio, with its original MP3 hash and dataset/shard/key recorded.

## Codec contract

| Property | Value |
|---|---|
| Model | `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` |
| Reviewed revision | `f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169` |
| Sampling rate | 48,000 Hz |
| Input channels | Original mono duplicated into two identical channels |
| Temporal frame | 3,840 source samples = 0.08 s = 12.5 frames/s |
| Exported RVQ depth | First **12** codebooks, matching Humaneness Voice Small |
| Codes | Integer 0–1023, little-endian int16 NumPy array, shape `[T,12]` |
| `Tokens` conditioning field | **T**, the number of codec frames |
| Streaming chunk | 4.8 s = 60 frames |
| Weight/quantizer dtype | FP32 |
| Non-quantizer compute | BF16 on GPU; SDPA |
| Audio tail | Explicit right-zero-padding to a multiple of 3,840 before encoding |

The pinned backend pads tensor storage while flooring valid length masks. Explicit per-clip padding prevents losing the partial final frame. `original_samples_48khz` and `right_padding_samples` are stored: after decoding, trim to the original sample count. Padding is always less than one frame. No normalization of loudness or removal of silence is applied. MP3 decoding and resampling retain the recording content but are not mathematically lossless; codec reconstruction is also lossy. The original MP3 SHA-256 remains the identity anchor.

The codec supports 32 codebooks; this release chooses 12 for compatibility with the target TTS model. Arrays contain physical codes only. PAD=1024 and audio STOP tokens are added/masked by the native training processor, not stored in `codes.npy`.

## WebDataset layout

```text
data/voice-acting-captioned-00000.tar   # existing MP3 + original complete JSON
tuning/data/voice-acting-tuning-00000.tar
  de-00001.json
  de-00001.codes.npy
  de-00001.annotation.json.gz
tuning/index.jsonl
tuning/inventory.json
tuning/summary.json
tuning/export-lock.json
tuning/heldout-groups.json
```

`json` is a lightweight training record: prompt, plain ASR text, language, voice, split, generation provenance, codec shape/sample counts, payload hashes and original dataset revision/shard/key. `annotation.json.gz` is the canonical annotation with raw global/local scores and embeddings, selected event captions, exact endpoints, cue-selection settings and timing proof. Original raw teacher annotations and all nine earlier procedural caption variants remain in the original dataset records. Join by ID and pinned source revision; token shards do not duplicate the MP3s.

The export lock records source inventory hash, code hashes, exact codec artifacts, batch/chunk/dtype policy and held-out challenge groups. It never contains API keys or machine-specific working paths. Shard hashes are in the release inventory. Resume requires the same contract; changing export code or configuration requires a new output directory.

## Splits and leakage

The hash unit is `source_subset:challenge_id`, with seed 20261011 and nominal proportions **96% train / 2% validation / 2% test**. Fifty scenarios used in the earlier listening review are additionally reserved for test. Both English and German recordings of the same scenario stay in the same partition. The realized counts are in `tuning/summary.json`.

This is a **scenario split**. The same 30 synthetic voices occur across partitions. Semantic similarity or related source scenarios may still cross the boundary. Reused recordings have been public and may have been used in previous component-model training; these splits apply to future Humaneness Voice tuning and are not a claim of unseen evaluation data for the caption/score models. The prior 50-example automated review is not a new evaluation of all 38,427 recordings.

The one empty ASR record is explicitly marked. The intended generation script remains provenance and is never substituted for recognized words. A downstream trainer can exclude it using `asr_empty`, or deliberately retain it for nonverbal generation after listening.

## Load records

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from humanness_voice_annotator.training import tuning_records, training_fields

root = Path(snapshot_download(
    "laion/german-english-voice-acting-challenges", repo_type="dataset",
    allow_patterns=["tuning/*"],
    revision="2f914a5081f6c0d6eb348b509ea50cf1d56aa7e3",
))
for shard in sorted((root / "tuning/data").glob("*.tar")):
    for row, codes in tuning_records(shard, split="train"):
        fields = training_fields(row)
        assert codes.shape == (fields["tokens"], 12)
        # fields: instruction, text, tokens (frames), language.
        # codes: target audio. Feed these to the native TTS processor.
```

Install `requirements-training.txt` for the tested encoder environment. A custom loader checks shapes, code range, payload SHA-256 and padding arithmetic. `include_annotation=True` loads the detailed sidecar. For corpus shuffling, shard and sample shuffle before batching; filtering by split is mandatory because each source TAR can contain multiple splits. Use group-level sampling when comparing voices or languages.

### Native TTS packing

`pack_target(processor, model_config, row, codes)` builds user instruction and assistant audio messages with the released Humaneness Voice Small processor. The processor can be constructed with `audio_tokenizer=None`: precomputed codes remove codec inference from training. Load the processor/config from the Humaneness Voice release, not the captioning model.

```python
from humanness_voice_annotator.training import pack_target

packed = pack_target(processor, model_config, row, codes)
# input_ids/labels have 13 channels: one text channel + 12 RVQ channels.
# Shifted targets: one continuation per audio frame, then one audio stop.
# Instruction/text/reference positions and PAD codes have label -100.
```

The `Text` field repeats the SCRIPT part, following the released structured-prompt packer. The full `GENERAL / SCRIPT` is the Instruction. `surface="caption-transcript"` selects the simpler preserved whole-recording caption with plain recognized text. `legacy=True` explicitly maps speech durations back to the older square-bracket surface; use it only when intentionally retaining that checkpoint's earlier format. The stored default remains round-bracket speech/burst durations and square-bracket pauses.

Optional `reference_codes` must come from a **distinct** recording, selected within the appropriate split and voice. Targets are not automatically used as their own reference. The packer guards against reference audio STOP labels leaking into target supervision. No identity-conditioning experiment is claimed by this preparation step.

For a model-free compatibility check using a downloaded Humaneness Voice Small bundle:

```bash
python scripts/verify_training_packing.py --bundle humaneness-voice-small \
  --shard tuning/data/voice-acting-tuning-00000.tar --output packing-check.json
```

This checks default/legacy/simple surfaces and reference masking. It does not load TTS or codec weights. The released tokenizer has no chat template; its native processor falls back to raw message content. This check preserves that historical behavior rather than silently introducing a new chat wrapper.

## Rebuild the export

The input is the parent dataset at revision `4e8a4f8ccd10ba5cf7965f461d47d213da463cba`, with its original inventory and TARs. Download the pinned codec and pinned procedural source on an internet-connected machine. The checked codec manifest lives in `humanness_voice_annotator/assets/training-codec.json`; the existing procedural source lock is also verified.

```bash
python scripts/download_training_codec.py --output models/moss_codec_v2

python scripts/export_training_dataset.py \
  --input original_dataset --output new_export \
  --codec-dir models/moss_codec_v2 --procedural-dir models/procedural \
  --source-revision 4e8a4f8ccd10ba5cf7965f461d47d213da463cba \
  --heldout-groups heldout-groups.json \
  --rank 0 --world-size 1 --batch-size 8 --threads 4 --decode-canary

python scripts/finalize_training_dataset.py \
  --source original_dataset --export new_export --output release
```

For multiple workers, use one GPU/process and distinct `--rank` values in `[0,world_size)`. SLURM rank variables are detected automatically. Each GPU loads the codec once; four CPU threads decode/compose ahead within each batch. Worker assignments are independent source TARs, with atomic completion proofs. The finalizer requires all source IDs exactly once, unchanged audio/voice/language/subset/duration, all expected shards and no scenario crossing partitions. Resume may use a different worker count after the previous job has stopped.

After auditing, `scripts/publish_training_dataset.py --release release --repo-id YOUR_DATASET` uploads one additive commit under `tuning/` and updates the dataset card. Original `data/` files are untouched. Use your own Hugging Face credential; never put it in the repository. The default Dataset Viewer remains restricted to the original audio archives. Use the custom loader for the tuning payload and its partitions.

## Attribution

Released code, new annotations and documentation use **CC BY 4.0**. Upstream model/code licenses remain unchanged: MOSS Audio Tokenizer v2 is Apache 2.0. See [THIRD_PARTY.md](THIRD_PARTY.md), the parent dataset card and linked component-model cards for source credits and their terms.
