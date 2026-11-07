"""Cache the Celniak Tobii exports and build the per-slide feature table.

Usage: python -I scripts/build_celniak_features.py <xlsx_dir> <cache_dir> <out_csv>
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from edd.celniak import cache_recording, load_cached, slide_features  # noqa: E402


def _cache_one(f, cache_dir):
    t0 = time.perf_counter()
    name = cache_recording(f, cache_dir)
    return f"cached {name} in {time.perf_counter() - t0:.1f}s"


def main(xlsx_dir, cache_dir, out_csv):
    xlsx_dir, cache_dir = Path(xlsx_dir), Path(cache_dir)
    files = sorted(xlsx_dir.glob("CIT Recording*.xlsx"), key=lambda p: int(p.stem.split("Recording")[1]))
    names = [f.stem.replace("CIT ", "") for f in files]
    todo = [f for f, n in zip(files, names) if not (cache_dir / f"{n}.csv.gz").exists()]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        for msg in pool.map(_cache_one, todo, [cache_dir] * len(todo)):
            print(msg, flush=True)
    tables = []
    for name in names:
        samples, meta = load_cached(cache_dir, name)
        tables.append(slide_features(samples, meta))
    df = pd.concat(tables, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    per_p = df.groupby("participant")["label"].first()
    print(f"{len(df)} slides, {df.participant.nunique()} participants "
          f"({int(per_p.sum())} guilty / {int((1 - per_p).sum())} innocent) -> {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
