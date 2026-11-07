"""Download the Chen, Mathot & Akyurek (2026) OSF data (Exp 1: osf.io/ske56, Exp 2: osf.io/sk5zq).

Usage: python -I scripts/phase2_chen_download.py <substring-filter> [threads]
The manifest (data/chen/osf_manifest.json) was built from the OSF v2 API folder listings.
Files are saved to data/chen/exp1|exp2/<subfolder>/<name>; complete files are skipped.
"""
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "chen"
NODE = {"ske56": "exp1", "sk5zq": "exp2"}


def fetch(o):
    node, rest = o["path"].split("/", 1)
    rest = rest.replace("Data for the manuscript/", "").replace("Data/", "")
    dest = DEST / NODE[node] / rest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == o["size"]:
        return dest, "skip"
    err = None
    for _ in range(4):
        try:
            tmp = str(dest) + ".part"
            with urllib.request.urlopen(o["url"], timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if os.path.getsize(tmp) != o["size"]:
                raise IOError("size mismatch")
            os.replace(tmp, dest)
            return dest, "ok"
        except Exception as e:  # noqa: BLE001
            err = e
    return dest, f"FAIL {err}"


if __name__ == "__main__":
    pattern = sys.argv[1]
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    items = [o for o in json.loads((DEST / "osf_manifest.json").read_text()) if pattern in o["path"]]
    print(f"{len(items)} files, {sum(o['size'] for o in items) / 1e6:.0f} MB", flush=True)
    with ThreadPoolExecutor(threads) as ex:
        for d, s in ex.map(fetch, items):
            print(s, d, flush=True)
