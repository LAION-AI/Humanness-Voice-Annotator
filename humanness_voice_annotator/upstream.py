"""Fetch a small, hash-checked set of files from the pinned procedural source."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.request import urlopen


def procedural_source(directory=None):
    lock = json.loads((Path(__file__).parent / "assets" / "procedural-source-lock.json").read_text())
    root = Path(directory) if directory else Path(os.environ.get("HVA_CACHE", Path.home()/".cache"/"humanness-voice-annotator")) / "procedural" / lock["revision"]
    root.mkdir(parents=True, exist_ok=True)
    def fetch(item):
        name, expected = item
        path = root/name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return
        if directory is not None:
            raise ValueError(f"Pinned procedural source mismatch: {name}")
        with urlopen(f"https://raw.githubusercontent.com/{lock['repository']}/{lock['revision']}/{name}", timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Downloaded source hash mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, lock["files"].items()))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
