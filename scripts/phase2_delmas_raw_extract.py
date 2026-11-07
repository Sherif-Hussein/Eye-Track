"""Phase 2, track delmas_raw, step 1: EDF -> per-trial, per-face features at six sensor rates.

For every participant in the authors' trial tables, reads the main-session EDF(s) and Psychtoolbox design
file(s), cuts the 5 s parallel displays, degrades them to 1000/500/250/120/60/30 Hz, re-detects events with
fixed-parameter I-VT and computes per-face features (src/edd/delmas_raw.py), plus per-trial summaries for
the identity-leakage probe. Also records the EyeLink-parser fixation count per face, to audit how the
published per-image table was aggregated.

Usage: python -I scripts/phase2_delmas_raw_extract.py <data_raw_dir> <BookFIP_64.xlsx> <out_dir> [--n_jobs 2]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from edd import delmas_raw as dr  # noqa: E402
from edd.delmas import load_trials  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
RATES = [1000, 500, 250, 120, 60, 30]
SUMMARY = ["s_nfix", "s_fixdur_mean", "s_fixdur_sd", "s_sacc_amp", "s_fix_vel", "s_pupil_mean", "s_pupil_sd",
           "s_missing", "s_fix_x_med", "s_fix_y_med", "s_lat_first"]


def session_files(raw, pid):
    d = Path(raw) / f"Sj_{pid:02d}"
    halves = sorted(d.glob(f"I{pid:02d}c2_*.edf"))
    return [(e, d / f"FI{pid:02d}c2_{e.stem.split('_')[-1]}.mat") for e in halves]


def trial_summary(x, y, p, fs):
    vel, fid, st, en, xd, yd = dr.ivt(x, y, fs)
    t = np.arange(len(x)) * 1000.0 / fs - dr.PRE_MS
    w = (t[st] >= 0) & (t[st] < dr.POST_MS) if len(st) else np.array([], bool)
    st, en = st[w], en[w]
    dur = (en - st) * 1000.0 / fs
    cx = np.array([np.nanmean(xd[a:b]) for a, b in zip(st, en)]) if len(st) else np.array([np.nan])
    cy = np.array([np.nanmean(yd[a:b]) for a, b in zip(st, en)]) if len(st) else np.array([np.nan])
    amp = np.hypot(np.diff(cx), np.diff(cy)) if len(cx) > 1 else np.array([np.nan])
    win = (t >= 0) & (t < dr.POST_MS)
    fixm = (fid >= 0) & win
    return [len(st), np.mean(dur) if len(dur) else np.nan, np.std(dur) if len(dur) else np.nan, np.nanmean(amp),
            np.nanmean(vel[fixm]) if fixm.any() else np.nan, np.nanmean(p[win]), np.nanstd(p[win]),
            float(np.mean(~np.isfinite(x[win]))), np.nanmedian(cx), np.nanmedian(cy),
            t[st][0] if len(st) else np.nan]


def process(raw, pid, group, cache):
    """Features for one participant, cached so an interrupted run resumes where it stopped."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)      # loky workers do not inherit filters
    path = Path(cache) / f"p{pid:02d}.npz"
    if path.exists():
        z = np.load(path)
        return z["face"], z["trial"], len(z["trial"]) // len(RATES), 0.0, None
    t0 = time.perf_counter()
    try:
        segs = [dr.load_session(edf, mat) for edf, mat in session_files(raw, pid)]
        if not segs:
            raise FileNotFoundError(f"no main-session EDF for participant {pid}")
        seg = {k: np.concatenate([s[k] for s in segs]) for k in segs[0]}
    except Exception as e:                                           # report, do not lose the other 74
        return None, None, 0, 0.0, f"{pid}: {e}"
    el = seg["el_nfix"]
    n_tr = len(seg["images"])
    face_rows, trial_rows = [], []
    for fs in RATES:
        for k in range(n_tr):
            x, y, p = dr.degrade(seg["x"][k], seg["y"][k], seg["p"][k], fs)
            F = dr.trial_features(x, y, p, fs, with_msacc=True)
            fam_trial = int((seg["images"][k] >= 57).any())
            for q in range(4):
                face_rows.append([pid, group, fs, k, q, int(seg["images"][k, q]), fam_trial,
                                  el[k, q] if fs == 1000 else np.nan, *F[q]])
            trial_rows.append([pid, group, fs, k, fam_trial, *trial_summary(x, y, p, fs)])
    face, trial = np.array(face_rows, float), np.array(trial_rows, float)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, face=face, trial=trial)
    tmp.replace(path)
    return face, trial, n_tr, time.perf_counter() - t0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("book64")
    ap.add_argument("out_dir")
    ap.add_argument("--n_jobs", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out_dir)
    cache = out / "delmas_raw_cache"
    cache.mkdir(parents=True, exist_ok=True)
    book = load_trials(a.book64)
    parts = book.groupby("participant").group.first()
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(process)(a.raw, int(p), int(g), str(cache))
                                               for p, g in parts.items())
    errors = [r[4] for r in res if r[4]]
    ok = [r for r in res if r[4] is None]
    print("failed participants:", errors, flush=True)
    face = pd.DataFrame(np.vstack([r[0] for r in ok]),
                        columns=["participant", "group", "rate", "trial", "quad", "image", "fam_trial",
                                 "el_nfix"] + dr.TRIAL_FEATURES)
    trial = pd.DataFrame(np.vstack([r[1] for r in ok]),
                         columns=["participant", "group", "rate", "trial", "fam_trial"] + SUMMARY)
    for d, cols in ((face, ["participant", "group", "rate", "trial", "quad", "image", "fam_trial"]),
                    (trial, ["participant", "group", "rate", "trial", "fam_trial"])):
        d[cols] = d[cols].astype(int)
    face.to_csv(out / "delmas_raw_face_trials.csv.gz", index=False)
    trial.to_csv(out / "delmas_raw_trial_summary.csv.gz", index=False)
    print("participants", len(ok), "of", len(res), "trials per participant", sorted({r[2] for r in ok}),
          "seconds per computed participant (6 rates)", round(float(np.mean([r[3] for r in ok if r[3]] or [0])), 2))


if __name__ == "__main__":
    main()
