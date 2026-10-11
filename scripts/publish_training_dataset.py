"""Publish an audited additive export, preserving the original dataset files."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',required=True,type=Path)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--token-file',type=Path,help='Optional private HF credential file; never copied into the release')
    args=p.parse_args()
    summary=json.loads((args.release/'summary.json').read_text())
    if not all(summary['verified'].values()):
        raise ValueError('Release audit failed')
    inventory=json.loads((args.release/'inventory.json').read_text())
    if len(inventory)!=summary['n_shards']:
        raise ValueError('Missing release shard inventory')
    for row in inventory:
        if (args.release/row['path']).stat().st_size!=row['size_bytes']:
            raise ValueError('Missing or incomplete release archive')
    token=args.token_file.read_text().strip() if args.token_file else None
    api=HfApi(token=token)
    parent=api.dataset_info(args.repo_id).sha
    card=Path(hf_hub_download(args.repo_id,'README.md',repo_type='dataset',revision=parent,token=token)).read_text()
    # Restrict the default Dataset Viewer to the original, uniform audio shards.
    # The additional token payload uses the documented custom split-aware loader.
    if 'configs:' not in card.split('---',2)[1]:
        card=card.replace('---\n','---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/voice-acting-captioned-*.tar\n',1)
    marker='## Humaneness Voice tuning export (11 October 2026)'
    if marker in card:
        card=card.split(marker,1)[0].rstrip()+'\n'
    card+=f'''

{marker}

**All {summary['n_samples']:,} recordings ({summary['audio_hours']:.2f} hours)** additionally have the Humanness Voice Annotator **GENERAL/SCRIPT** prompt and **MOSS Audio Tokenizer v2** targets for later Humaneness Voice Small tuning. This is an additive data-preparation release; no new TTS model training is reported.

[Download token shards and annotations](tuning/README.md) · [Exact statistics](tuning/summary.json) · [Training/loading guide](https://github.com/LAION-AI/Humanness-Voice-Annotator/blob/main/TRAINING_DATA.md) · [Explanatory page](https://projects.laion.ai/Humanness-Voice-Annotator/training.html) · [Listen to the default format](https://projects.laion.ai/Humanness-Voice-Annotator/)

The {summary['n_shards']} additive WebDataset TARs live under `tuning/data/`. Each sample has a lightweight JSON training record, an int16 `[frames,12]` `codes.npy`, and a compressed canonical annotation retaining raw acoustic scores/embeddings, exact intervals, MOSS descriptions and local delivery cues. Original MP3s and all previous annotations remain under `data/`; match by ID and audio SHA-256. Generation metadata, including the precise synthetic voice, is retained.

The separately loaded [MOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) encodes 48-kHz audio using its first **12 RVQ codebooks**, one frame per **80 ms**. Mono is duplicated to stereo, and the partial final frame is explicitly zero-padded. True length/padding and exact producer revisions are recorded. `Tokens` means the frame count, not the number of codebook indices. Decode and trim using `original_samples_48khz`.

Challenge-level splits keep English/German counterparts together; the 50 earlier listening scenarios are additionally reserved for test. These are future-TTS preparation splits, not speaker holdouts or independent component-model benchmark claims. Use the custom split-aware loader in the guide. The default Hugging Face Dataset Viewer continues to read the original audio archives; it does not expose the training splits.

New annotations and code are **CC BY 4.0**. Upstream model/code licenses remain their own; the neural codec is Apache 2.0. Model predictions remain estimates and should be reviewed before training.
'''
    operations=[CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card.encode())]
    for file in sorted(args.release.rglob('*')):
        if file.is_file() and '.cache' not in file.parts:
            operations.append(CommitOperationAdd(path_in_repo='tuning/'+file.relative_to(args.release).as_posix(),path_or_fileobj=file))
    result=api.create_commit(args.repo_id,repo_type='dataset',operations=operations,parent_commit=parent,
                             commit_message='Add verified GENERAL/SCRIPT captions and 12-codebook MOSS v2 tuning targets')
    print(result.commit_url)


if __name__=='__main__':
    main()
