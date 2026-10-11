"""Stage pinned models and verified procedural source on a connected machine."""
import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanness_voice_annotator.formatting import sources
from humanness_voice_annotator.upstream import procedural_source


def main():
    from huggingface_hub import snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('models'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    patterns = {
        'humaneness_ears': ['model.py', 'training_normalization.json', 'classes.json',
                           'medium/config.json', 'medium/model.safetensors', 'medium/preprocessor_config.json',
                           'README.md', '*LICENSE*'],
        'parakeet': ['config.json', '*.safetensors', '*.safetensors.index.json',
                     '*token*', '*vocab*', '*processor*', '*.model', 'README.md', '*LICENSE*'],
        'moss_voice': None,
        'moss_burst': ['adapter/*', 'README.md', '*LICENSE*'],
    }
    for key, model in sources()['models'].items():
        snapshot_download(model['repo'], revision=model['revision'],
                          allow_patterns=patterns[key], local_dir=args.output/key)
        print('Staged', key, model['revision'], flush=True)
    source = procedural_source()
    target = args.output/'procedural'
    target.mkdir(exist_ok=True)
    for file in source.iterdir():
        if file.is_file():
            shutil.copy2(file, target/file.name)
    procedural_source(target)  # verify the staged copy as well
    print('Verified procedural source:', target)


if __name__ == '__main__':
    main()
