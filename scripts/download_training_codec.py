"""Stage the reviewed codec artifacts for offline training-data export."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    pin=json.loads((root/'humanness_voice_annotator/assets/training-codec.json').read_text())
    snapshot_download(pin['repository'],revision=pin['revision'],
                      allow_patterns=[f['path'] for f in pin['files']],local_dir=args.output)
    print('Staged pinned codec:',pin['repository'],pin['revision'])


if __name__=='__main__':
    main()
