"""Fresh inference in isolated environments, or composition from cached predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .formatting import compose_annotation, sources, verify_script


def save(result, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    output.with_suffix(".txt").write_text(result["prompt"]+"\n")
    print(f"Saved {output} and {output.with_suffix('.txt')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    saved = sub.add_parser("from-saved", help="Reformat cached predictions without models or an API key")
    saved.add_argument("--acoustic", type=Path, required=True)
    saved.add_argument("--moss", type=Path, required=True)
    saved.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify", help="Independently check a GENERAL/SCRIPT annotation")
    check.add_argument("annotation", type=Path)
    fresh = sub.add_parser("annotate", help="Run two inference environments, then compose the prompt")
    fresh.add_argument("audio", type=Path)
    fresh.add_argument("--language", required=True)
    fresh.add_argument("--output", type=Path, required=True)
    fresh.add_argument("--acoustic-python", required=True)
    fresh.add_argument("--moss-python", required=True)
    fresh.add_argument("--device", default="cuda:0")
    fresh.add_argument("--threshold", type=float, default=.6)
    fresh.add_argument("--threads", type=int, default=4)
    fresh.add_argument("--cache", type=Path, default=Path("outputs/cache"))
    for option in ["procedural-dir", "humaneness-dir", "parakeet-dir", "voice-dir", "burst-dir"]:
        fresh.add_argument("--"+option)
    args = parser.parse_args()
    if args.command == "verify":
        annotation = json.loads(args.annotation.read_text())
        result = verify_script(annotation["script"], annotation["aligned_transcript"],
                               annotation["audio"]["duration_s"], len(annotation["burst_events"]))
        print(json.dumps(result, indent=2))
        if not result["verified"]:
            raise SystemExit(1)
        return
    if args.command == "annotate":
        if not 0 <= args.threshold <= 1 or args.threads < 1:
            parser.error("Invalid threshold or thread count")
        identity = {"sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(), "sources": sources(),
                    "language": args.language, "threshold": args.threshold, "threads": args.threads,
                    "device": args.device,
                    "local_overrides": {key: getattr(args, key) for key in ["procedural_dir", "humaneness_dir", "parakeet_dir", "voice_dir", "burst_dir"]}}
        work = args.cache/hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        work.mkdir(parents=True, exist_ok=True)
        (work/"identity.json").write_text(json.dumps(identity, indent=2)+"\n")
        paths = {}
        for stage, python in [("acoustic", args.acoustic_python), ("moss", args.moss_python)]:
            path = work/(stage+".json")
            paths[stage] = path
            if path.exists():
                continue
            command = [python, "-m", "humanness_voice_annotator.worker", stage,
                       "--audio", str(args.audio.resolve()), "--output", str(path.resolve()),
                       "--language", args.language, "--device", args.device,
                       "--threshold", str(args.threshold), "--threads", str(args.threads)]
            if stage == "moss":
                command += ["--acoustic", str(paths["acoustic"].resolve())]
            for key in ["procedural_dir", "humaneness_dir", "parakeet_dir", "voice_dir", "burst_dir"]:
                if getattr(args, key):
                    command += ["--"+key.replace("_", "-"), str(Path(getattr(args, key)).resolve())]
            subprocess.run(command, check=True)
        args.acoustic, args.moss = paths["acoustic"], paths["moss"]
    result = compose_annotation(json.loads(args.acoustic.read_text()), json.loads(args.moss.read_text()))
    save(result, args.output)


if __name__ == "__main__":
    main()
