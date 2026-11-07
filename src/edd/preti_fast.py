"""Vectorised per-participant feature extraction for the Preti track (efficiency experiment).

Drop-in numerical replacement for `edd.preti.arm_subject_table(trials_from_fixations(f), arm)` of one
participant, for the arms `B_fusion`, `I7_lgbm_I1_rt` and `I7_lr_en_I1_fusion`, and for the trial matrix of
the eye-only arm `I3_trial_raw_aggtuned_ocular`. The input is a `FixArrays` (NumPy columns of one
participant, rows pre-sorted by participant and trial, each trial contiguous and in recording order).
No pandas, no per-row or per-trial Python loops; a handful of NumPy calls over the 200-300 fixation rows.

The formulas are the originals': pc_sub = mean(post-baseline pupil) - first-fixation pupil (NaN for a
one-fixation trial), blink_any = any blink in the trial, blink_per_s = blinks / RT * 1000, and subject
features = per-question-type means (NaN skipped), contrasts, ratios and deviance in units of the
participant's own SD (ddof = 1, over all of his/her trials).
"""
from typing import NamedTuple

import numpy as np

QT_CODE = {"C": 0, "E": 1, "U": 2}
QT = ("C", "E", "U")
_Q = np.arange(3)[:, None]


class FixArrays(NamedTuple):
    """Fixation rows of ONE participant as NumPy columns (rows sorted by trial; trials contiguous)."""
    trial: np.ndarray   # int64, trial code, constant within a trial
    qt: np.ndarray      # int8, 0 = C, 1 = E, 2 = U (per row)
    rt: np.ndarray      # float64, reaction time in ms (constant within a trial)
    pup: np.ndarray     # float64, CURRENT_FIX_PUP
    blink: np.ndarray   # float64, BLINK_0_1
    y: int


def arrays_from_frame(f):
    """One-time ingestion of the fixation DataFrame (`preti.fixations`) into {subject: FixArrays}.
    This is data loading, not per-participant inference, and is timed separately."""
    import pandas as pd
    qt = pd.Categorical(f["qt"], categories=list(QT)).codes.astype(np.int8)
    out = {}
    for s, idx in f.groupby("subject", sort=True).indices.items():
        g = f.iloc[idx]
        out[s] = FixArrays(trial=pd.factorize(g["trial"])[0].astype(np.int64), qt=qt[idx],
                           rt=g["RT"].to_numpy(float), pup=g["pup"].to_numpy(float),
                           blink=g["blink"].to_numpy(float), y=int(g["y"].iloc[0]))
    return out


# ----------------------------------------------------------------------------- trial level
def _starts(trial):
    n = trial.size
    chg = np.flatnonzero(trial[1:] != trial[:-1])
    starts = np.empty(chg.size + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = chg + 1
    return starts, n


class Trials(NamedTuple):
    qt: np.ndarray
    rt: np.ndarray
    pc_sub: np.ndarray | None = None
    blink_any: np.ndarray | None = None
    blink_per_s: np.ndarray | None = None
    blinks: np.ndarray | None = None


def trial_arrays(a, eye=True):
    """Per-trial RT and question type; with eye=True also pc_sub, blink_any, blink_per_s, blinks.
    The RT-only path (eye=False) touches no pupil or blink data."""
    starts, n = _starts(a.trial)
    rt, qt = a.rt[starts], a.qt[starts]
    if not eye:
        return Trials(qt, rt)
    blinks = np.add.reduceat(a.blink, starts)
    base = a.pup[starts]
    n_rest = np.diff(starts, append=n) - 1
    rest_sum = np.add.reduceat(a.pup, starts) - base          # sum of post-baseline fixations (exact: integers)
    with np.errstate(divide="ignore", invalid="ignore"):
        act = np.where(n_rest > 0, rest_sum / n_rest, np.nan)  # NaN for one-fixation trials
        blink_per_s = blinks / rt * 1000
    return Trials(qt, rt, act - base, (blinks > 0).astype(float), blink_per_s, blinks)


# ----------------------------------------------------------------------------- subject level
def _qt_mean(x, M):
    """Mean of x per question type, NaN skipped; NaN where a type has no valid trial. M: (3, T) one-hot."""
    ok = ~np.isnan(x)
    cnt = M @ ok
    with np.errstate(divide="ignore", invalid="ignore"):
        return (M @ np.where(ok, x, 0.0)) / cnt


def _sd(x):
    """SD with ddof = 1 over valid entries (NaN if fewer than two)."""
    ok = ~np.isnan(x)
    n = ok.sum()
    if n < 2:
        return np.nan
    xz = np.where(ok, x, 0.0)
    d = np.where(ok, x - xz.sum() / n, 0.0)
    return np.sqrt((d * d).sum() / (n - 1))


def _block(x, M, kinds):
    """Features of one measure in the order of `preti.subject_features`: level, diff, ratio, dev."""
    c, e, u = _qt_mean(x, M)
    out = []
    with np.errstate(divide="ignore", invalid="ignore"):
        if "level" in kinds:
            out += [c, e, u]
        if "diff" in kinds:
            out += [u - c, e - c, u - e]
        if "ratio" in kinds:
            out += [u / c, e / c]
        if "dev" in kinds:
            s = _sd(x)
            s = np.nan if s == 0 else s
            out += [np.nan_to_num((u - c) / s, nan=0.0, posinf=np.inf, neginf=-np.inf),
                    np.nan_to_num((e - c) / s, nan=0.0, posinf=np.inf, neginf=-np.inf)]
    return out


def _onehot(qt):
    return (qt[None, :] == _Q).astype(float)


_OC, _RT = ("pc_sub", "blink_per_s"), ("RT", "logRT")
_LDD, _LDRD = ("level", "diff", "dev"), ("level", "diff", "ratio", "dev")


def _finish(vals):
    v = np.asarray(vals, dtype=float)
    v[np.isinf(v)] = np.nan   # the original replaces +-inf with NaN after building the table
    return v[None, :]


def rt_features(a):
    """`I7_lgbm_I1_rt` / `I1_diff_rt` features (20), from one RT per trial."""
    t = trial_arrays(a, eye=False)
    M = _onehot(t.qt)
    return _finish(_block(t.rt, M, _LDRD) + _block(np.log(t.rt), M, _LDRD))


def fusion_features(a):
    """`I7_lr_en_I1_fusion` / `I1_diff_fusion` features (36): pc_sub and blink_per_s blocks, then the RT blocks."""
    t = trial_arrays(a, eye=True)
    M = _onehot(t.qt)
    return _finish(_block(t.pc_sub, M, _LDD) + _block(t.blink_per_s, M, _LDD)
                   + _block(t.rt, M, _LDRD) + _block(np.log(t.rt), M, _LDRD))


def b_fusion_features(a):
    """`B_fusion` features (9), in the order of the arm: RT_{C,E,U}, blink_rate_{C,E,U}, pupil_{C,E,U}."""
    t = trial_arrays(a, eye=True)
    M = _onehot(t.qt)
    with np.errstate(divide="ignore", invalid="ignore"):
        rt_m = (M @ t.rt) / M.sum(axis=1)                # mean RT per question type (NaN if the type is absent)
        blink_rate = (M @ t.blinks) / rt_m
    return _finish(list(rt_m) + list(blink_rate) + list(_qt_mean(t.pc_sub, M)))


def ocular_trial_matrix(a):
    """Trial design matrix of `I3_trial_raw_aggtuned_ocular`: pc_sub, blink_any, qU, qE (one row per trial)."""
    t = trial_arrays(a, eye=True)
    X = np.empty((t.qt.size, 4))
    X[:, 0], X[:, 1] = t.pc_sub, t.blink_any
    X[:, 2], X[:, 3] = t.qt == 2, t.qt == 1
    return X


FAST_FEATURES = {"B_fusion": b_fusion_features, "I7_lgbm_I1_rt": rt_features, "I7_lr_en_I1_fusion": fusion_features,
                 "I1_diff_rt": rt_features, "I1_diff_fusion": fusion_features}


def feature_names(arm):
    """Column names in the order the fast functions return them (checked against results.json)."""
    def blk(m, kinds):
        n = []
        if "level" in kinds:
            n += [f"{m}_{q}" for q in QT]
        if "diff" in kinds:
            n += [f"{m}_UmC", f"{m}_EmC", f"{m}_UmE"]
        if "ratio" in kinds:
            n += [f"{m}_UdC", f"{m}_EdC"]
        if "dev" in kinds:
            n += [f"{m}_devU", f"{m}_devE"]
        return n
    if arm == "B_fusion":
        return [f"RT_{q}" for q in QT] + [f"blink_rate_{q}" for q in QT] + [f"pupil_{q}" for q in QT]
    if arm.endswith("_rt"):
        return blk("RT", _LDRD) + blk("logRT", _LDRD)
    if arm.endswith("_fusion"):
        return blk("pc_sub", _LDD) + blk("blink_per_s", _LDD) + blk("RT", _LDRD) + blk("logRT", _LDRD)
    raise ValueError(arm)
