"""Matched accuracy-and-cost measurement for the delmas_raw track (library part).

Nothing of the track is modified or copied: the track's own pipeline (`scripts/phase2_delmas_raw.py`) and its
non-inferiority script (`scripts/phase2_delmas_raw_noninf.py`) are loaded from their files and reused, so the
frozen folds, decimation, event detection, features and nested tuning are the very same code.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
TRACK_SCRIPT = ROOT / "scripts" / "phase2_delmas_raw.py"
NONINF_SCRIPT = ROOT / "scripts" / "phase2_delmas_raw_noninf.py"
TRACK_DIR = ROOT / "results" / "phase2" / "delmas_raw"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_track():
    """The track's pipeline module, loaded from its file (the file is not touched)."""
    mod = _load_module("phase2_delmas_raw_track", TRACK_SCRIPT)
    mod.progress_note = lambda name, r: None          # do not write progress.md into the new folder
    return mod


def load_noninf():
    return _load_module("phase2_delmas_raw_noninf_track", NONINF_SCRIPT)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


# ----------------------------------------------------------------------------- non-inferiority
def per_set_arrays(ref_oof, cand_oof):
    """Same structure as phase2_delmas_raw_noninf.main: {set: [(y, score_ref, score_cand) per participant]}."""
    m = ref_oof.merge(cand_oof, on=["set", "participant", "image", "y"], suffixes=("_r", "_c"))
    per = {}
    for s, g in m.groupby("set"):
        per[s] = [(gp.y.to_numpy(), gp.score_r.to_numpy(), gp.score_c.to_numpy()) for _, gp in g.groupby("participant")]
    return per, len(m)


def paired_cluster_bootstrap(per, rng, n_boot):
    """Exactly the loop of phase2_delmas_raw_noninf.main (one rng draw per set per resample, same order)."""
    auc_r = np.mean([roc_auc_score(np.concatenate([a[0] for a in v]), np.concatenate([a[1] for a in v])) for v in per.values()])
    auc_c = np.mean([roc_auc_score(np.concatenate([a[0] for a in v]), np.concatenate([a[2] for a in v])) for v in per.values()])
    deltas = []
    for _ in range(n_boot):
        dr_, dc_ = [], []
        for v in per.values():
            pick = rng.integers(0, len(v), len(v))
            y = np.concatenate([v[i][0] for i in pick])
            if len(np.unique(y)) < 2:
                continue
            dr_.append(roc_auc_score(y, np.concatenate([v[i][1] for i in pick])))
            dc_.append(roc_auc_score(y, np.concatenate([v[i][2] for i in pick])))
        deltas.append(np.mean(dc_) - np.mean(dr_))
    return float(auc_r), float(auc_c), np.array(deltas)


def model_auc_ci(oof, n_boot=2000, seed=1):
    """Participant-cluster bootstrap CI of ONE model's metric (image-level AUC pooled within each set,
    repeat-averaged scores, mean over the 6 sets), same resampling scheme as the pair test."""
    per = {}
    for s, g in oof.groupby("set"):
        per[s] = [(gp.y.to_numpy(), gp.score.to_numpy()) for _, gp in g.groupby("participant")]
    est = float(np.mean([roc_auc_score(np.concatenate([a[0] for a in v]), np.concatenate([a[1] for a in v])) for v in per.values()]))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        a = []
        for v in per.values():
            pick = rng.integers(0, len(v), len(v))
            y = np.concatenate([v[i][0] for i in pick])
            if len(np.unique(y)) < 2:
                continue
            a.append(roc_auc_score(y, np.concatenate([v[i][1] for i in pick])))
        out.append(np.mean(a))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"auc": est, "ci95": [float(lo), float(hi)], "n_boot": n_boot, "seed": seed}


def noninf_row(ref, cand, d, auc_r, auc_c, margin):
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"reference": ref, "candidate": cand, "auc_ref": auc_r, "auc_cand": auc_c, "delta": auc_c - auc_r,
            "ci95": [float(lo), float(hi)], "lower_one_sided_95": float(np.percentile(d, 5)),
            "p_inferior_raw": float((1 + np.sum(d <= -margin)) / (1 + len(d)))}


# ----------------------------------------------------------------------------- cost
def n_stored_parameters(clf, n_feat):
    """The track's convention for LR: coefficients + intercept + 4 stored numbers per feature (2*n*2 in
    phase2_delmas_raw.efficiency), generalised from 4 to n features. `strict` counts what inference really
    needs: coefficients, intercept, imputer median, scaler mean, scaler scale (3 per feature)."""
    return {"track_convention": int(clf.coef_.size + clf.intercept_.size + 4 * n_feat),
            "strict_minimum": int(clf.coef_.size + clf.intercept_.size + 3 * n_feat)}
