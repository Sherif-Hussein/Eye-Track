"""Task 2.2 (EXPLORATORY, not preregistered), step 1: Delmas raw EDF -> samples at 30/20/15/10 Hz, per-face features,
decimated sample caches and event-detection diagnostics.

Reuses UNCHANGED (imported from their files, nothing copied or edited):
  scripts/phase6_trackC_c2.py   load_halves (EDF -> 1000 Hz trials + parser fixations), parser_arrays, ivt_rate
  src/edd/delmas_raw.py         degrade (anti-aliased polyphase), trial_features (I-VT 30 deg/s, 60 ms), as in
                                scripts/phase2_delmas_raw_extract.py (which produced data/derived/delmas_raw_face_trials.csv.gz)
Output: cache/p<id>.npz (face rows + decimated x, y, pupil per rate), cache/ev_p<id>.pkl (event diagnostics per rate).
Resumable (skips finished participants). Single process, single thread.
"""
import os
import sys
import time
import importlib.util
import pickle
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

from edd import delmas_raw as dr  # noqa: E402
from edd.delmas import load_trials  # noqa: E402

spec = importlib.util.spec_from_file_location("trackC_c2", ROOT / "scripts" / "phase6_trackC_c2.py")
c2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c2)

OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
RATES = [30, 20, 15, 10]
BOOK64 = ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"


def process(pid, group):
    path = CACHE / f"p{pid:02d}.npz"
    if path.exists():
        return "cached"
    x, y, p, images, fix = c2.load_halves(pid)        # float32 (n_trials, 5500), same cutting as dr.load_session
    n_tr = len(images)
    X, Y, P = x.astype(np.float64), y.astype(np.float64), p.astype(np.float64)
    face_rows, dec = [], {}
    for fs in RATES:
        xs, ys, ps = [], [], []
        for k in range(n_tr):
            a, b, c = dr.degrade(x[k], y[k], p[k], fs)         # exactly as phase2_delmas_raw_extract.process
            xs.append(a); ys.append(b); ps.append(c)
            F = dr.trial_features(a, b, c, fs, with_msacc=True)
            fam_trial = int((images[k] >= 57).any())
            for q in range(4):
                face_rows.append([pid, group, fs, k, q, int(images[k, q]), fam_trial, np.nan, *F[q]])
        dec[f"x{fs}"], dec[f"y{fs}"], dec[f"p{fs}"] = np.stack(xs), np.stack(ys), np.stack(ps)
    # event diagnostics (Supplementary-Table-S10 style), via the Track C2 functions
    T = X.shape[0]
    par = c2.parser_arrays(fix, T)
    res = {"images": images, "parser": par, "rates": {}}
    for fs in RATES:
        iv = c2.ivt_rate(X, Y, P, fs, fix)
        del iv["fix"], iv["t"]
        res["rates"][fs] = iv
    res["missing_1000"] = float(np.mean(~np.isfinite(X[:, dr.PRE_MS:dr.PRE_MS + dr.POST_MS])))
    with open(CACHE / f"ev_p{pid:02d}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    tmp = CACHE / f"p{pid:02d}.tmp.npz"
    np.savez_compressed(tmp, face=np.array(face_rows, float), images=images, group=group, **dec)
    tmp.replace(path)
    return "extracted"


def main():
    book = load_trials(BOOK64)
    parts = {int(p): int(g) for p, g in book.groupby("participant").group.first().items()}
    t0 = time.time()
    for i, (pid, g) in enumerate(parts.items()):
        t1 = time.time()
        s = process(pid, g)
        print(f"{time.strftime('%H:%M:%S')} [{i + 1}/{len(parts)}] participant {pid} {s} {time.time() - t1:.1f}s (total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
