"""Extract per-trial 100 Hz RSVP-phase traces from the Chen et al. (2026) EDF files.

Output: data/derived/chen_<exp>_traces.npz (p_dm, p_rec, p_raw, x, y, miss; trials x 650 samples)
and data/derived/chen_<exp>_meta.csv (OpenSesame variables logged to the EDF per trial).
Per-subject caches (also holding raw 1 kHz pupil, pa_1k) live in data/derived/chen_cache2/ and are
written atomically, so an interrupted run resumes safely.
Usage: python -I phase2_chen_extract.py [n_jobs]
"""
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edd import chen  # noqa: E402

DATA, DERIVED = ROOT / "data" / "chen", ROOT / "data" / "derived"
CACHE = DERIVED / "chen_cache2"
KEEP = ["subject_nr", "trialtype", "T1", "T1pos", "T1_direction", "width", "trialnumber", "blockno",
        "correct", "itemduration", "T1age", "phase_ms", "n_blinks_phase"]


def one(exp, edf):
    out = CACHE / f"{exp}_{edf.stem}.npz"
    if out.exists():
        return out, 0.0
    t = time.perf_counter()
    meta, arr = chen.extract_subject(edf)
    m = pd.DataFrame(meta)
    m = m[[c for c in KEEP if c in m]]
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=m.to_json(orient="records"), **arr)
    os.replace(tmp, out)
    return out, time.perf_counter() - t


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(exp, f) for exp in ("exp1", "exp2") for f in sorted((DATA / exp / "EDF").glob("*.edf"))]
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(e, f) for e, f in jobs)
    timing = pd.DataFrame([(str(p.name), s) for p, s in res], columns=["file", "seconds"])
    timing.to_csv(CACHE / "timing.csv", mode="a", index=False, header=not (CACHE / "timing.csv").exists())
    for exp in ("exp1", "exp2"):
        files = sorted(f for f in CACHE.glob(f"{exp}_*.npz") if ".tmp" not in f.name)
        metas, arrs = [], {k: [] for k in ("p_dm", "p_rec", "p_raw", "x", "y", "miss")}
        for f in files:
            z = np.load(f, allow_pickle=False)
            m = pd.read_json(io.StringIO(str(z["meta"])), orient="records")
            m["file"] = f.stem.split("_", 1)[1]
            metas.append(m)
            for k in arrs:
                arrs[k].append(z[k])
        meta = pd.concat(metas, ignore_index=True)
        np.savez_compressed(DERIVED / f"chen_{exp}_traces.npz", **{k: np.concatenate(v) for k, v in arrs.items()})
        meta.to_csv(DERIVED / f"chen_{exp}_meta.csv", index=False)
        print(exp, len(files), "subjects", len(meta), "trials")
