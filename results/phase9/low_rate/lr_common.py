"""Shared helpers for Task 2.2 (EXPLORATORY). Image-level AUC = repeat-averaged out-of-fold scores, AUC pooled within each of
the six Delmas group-sets, mean over sets (the definition of paper/figures/data_delmas_rate_curve.csv, scripts/phase5_figures.py
delmas_curve). Participant-cluster bootstrap resamples participants within each set (same scheme as phase5_figures.py and
scripts/phase6_trackC_c1.py stage_stats); all configurations are evaluated on the SAME resamples, so differences are paired.
Uses edd.webcam_noise.SetData / auc_matrix unchanged."""
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edd import webcam_noise as W  # noqa: E402

OUT = Path(__file__).resolve().parent
STORED = ROOT / "results" / "phase2" / "delmas_raw"
RATES_NEW = [30, 20, 15, 10]
SETS = ["64_g1", "64_g2", "64_g3", "32_g3", "32_g4", "32_g5"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def load_oof(path_or_df):
    """Repeat-averaged OOF scores per (set, participant, image)."""
    d = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df
    d = d.groupby(["set", "participant", "image"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
    return d.sort_values(["set", "participant", "image"]).reset_index(drop=True)


def pooled_auc(path_or_df):
    from sklearn.metrics import roc_auc_score
    d = load_oof(path_or_df)
    return float(np.mean([roc_auc_score(g.y, g.score) for _, g in d.groupby("set")]))


def paired_bootstrap(configs, ref, n_boot=2000, seed=20261102, n_ci_small=300, return_draws=False):
    """configs: {name: [oof DataFrame/path per noise draw]} (all share the same keys). A configuration's AUC is the mean over its draws.
    Returns {name: dict(auc, auc_lo, auc_hi [first n_ci_small resamples], delta_vs_ref, delta_lo, delta_hi [all n_boot])}."""
    names = list(configs)
    cols, owner = [], []
    base = None
    for n in names:
        for o in configs[n]:
            d = load_oof(o)
            if base is None:
                base = d
            else:
                assert len(d) == len(base) and (d[["set", "participant", "image", "y"]].to_numpy() == base[["set", "participant", "image", "y"]].to_numpy()).all(), n
            cols.append(d.score.to_numpy())
            owner.append(n)
    S = np.column_stack(cols)
    owner = np.array(owner)
    sets = {}
    for s in SETS:
        m = (base.set == s).to_numpy()
        sets[s] = W.SetData(base.y.to_numpy()[m], base.participant.to_numpy()[m], S[m])
    point = np.array([W.auc_matrix(sets[s].y, sets[s].S) for s in SETS])        # (sets, cols)
    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, len(SETS), S.shape[1]), np.nan)
    for b in range(n_boot):
        for si, s in enumerate(SETS):
            sd = sets[s]
            idx = sd.rows_of(rng.integers(0, sd.n_part, sd.n_part))
            y = sd.y[idx]
            if len(np.unique(y)) < 2:
                continue
            boot[b, si] = W.auc_matrix(y, sd.S[idx])
    bs = np.nanmean(boot, axis=1)                                                # (n_boot, cols)
    est = {n: bs[:, owner == n].mean(1) for n in names}
    out = {}
    for n in names:
        pt = float(point[:, owner == n].mean())
        d = est[n] - est[ref]
        out[n] = {"auc": pt, "auc_lo": float(np.percentile(est[n][:n_ci_small], 2.5)), "auc_hi": float(np.percentile(est[n][:n_ci_small], 97.5)),
                  "delta_vs_ref": pt - float(point[:, owner == ref].mean()),
                  "delta_lo": float(np.percentile(d, 2.5)), "delta_hi": float(np.percentile(d, 97.5)),
                  "n_draws": int((owner == n).sum()), "per_draw_auc": [float(point[:, k].mean()) for k in np.flatnonzero(owner == n)]}
    return (out, est) if return_draws else out
