"""Phase 3 confirmatory test `chen_confirm`: pooled Exp 1 + Exp 2 of Chen, Mathot & Akyurek (2026).

Everything here is specified in results/phase3/chen_confirm/PREREG.md. The screening code
(scripts/phase2_chen.py) is imported unchanged (features, preprocessing, grid, paper rule, z statistics).
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
FAM, C1, C2 = 2, 3, 4
N_FOLDS, SEEDS = 10, (3031, 3032, 3033)
TARGET_FPR = 0.07
Z_ALPHA = float(stats.norm.isf(0.05))
EXP_OFFSET = {"exp1": 0, "exp2": 1000}
ARMS = ["rate_rule", "I2b", "I2b_noMiss", "MissOnly", "I2b_perexp"]
INNER_SPLITS = 5

_P2 = None


def p2():
    """The screening script, imported as a module (its main() is guarded; it only runs mkdir at import)."""
    global _P2
    if _P2 is None:
        spec = importlib.util.spec_from_file_location("phase2_chen_screening", ROOT / "scripts" / "phase2_chen.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["phase2_chen_screening"] = m
        spec.loader.exec_module(m)
        _P2 = m
    return _P2


# ----------------------------------------------------------------------------------------------
# Data and folds
# ----------------------------------------------------------------------------------------------
def build_pooled():
    """Pooled familiar/control trials of both experiments, screening preprocessing (authors' pipeline)."""
    P = p2()
    metas, chs = [], []
    for exp in ("exp1", "exp2"):
        meta, arr = P.load(exp)
        m, ch, _ = P.preprocess(meta, arr)
        mf, chf = P.fc_subset(m, ch)
        mf = mf.copy()
        mf["exp"] = 1 if exp == "exp1" else 2
        mf["unit"] = mf.subject_nr + EXP_OFFSET[exp]
        metas.append(mf)
        chs.append(chf)
    meta = pd.concat(metas, ignore_index=True)
    ch = {k: np.vstack([c[k] for c in chs]) for k in chs[0]}
    return meta, ch


def make_folds(meta):
    """3 repeats x 10 participant-disjoint folds; each experiment is permuted and dealt into 10 folds
    (stratified by experiment); exp-1 fold i is joined with exp-2 fold 9 - i to balance fold sizes."""
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        parts = {}
        for e in (1, 2):
            u = sorted(meta.loc[meta.exp == e, "unit"].unique())
            u = [int(x) for x in rng.permutation(u)]
            parts[e] = [u[i::N_FOLDS] for i in range(N_FOLDS)]
        out.append([[str(x) for x in parts[1][i] + parts[2][N_FOLDS - 1 - i]] for i in range(N_FOLDS)])
    return out


def feature_sets(ch):
    """125 features of I2b (screening: bin_features with the FUSION channel order) and column groups."""
    P = p2()
    X, names = P.bin_features(ch, P.FUSION)
    names = np.array(names)
    chan = np.array([n.split("_")[0] for n in names])
    return X, names, {"I2b": np.ones(len(names), bool), "I2b_noMiss": chan != "miss", "MissOnly": chan == "miss"}


# ----------------------------------------------------------------------------------------------
# Calibration and per-person statistics
# ----------------------------------------------------------------------------------------------
def person_z(stat, T1):
    """z statistics exactly as in screening's person_tests: familiar vs both controls, and control-1 vs
    control-2 in both directions (one-sided Student t converted to a z)."""
    P = p2()
    from edd import chen
    s = np.asarray(stat, float)
    _, p = chen.one_tailed_t(s[T1 == FAM], s[T1 != FAM])
    _, p12 = chen.one_tailed_t(s[T1 == C1], s[T1 == C2])
    _, p21 = chen.one_tailed_t(s[T1 == C2], s[T1 == C1])
    return {"z_fam": P._z_onesided(s[T1 == FAM], s[T1 != FAM]), "z_c1": P._z_onesided(s[T1 == C1], s[T1 == C2]),
            "z_c2": P._z_onesided(s[T1 == C2], s[T1 == C1]), "p": p, "p_c1c2": p12, "p_c2c1": p21}


def threshold_for_fpr(null_z, target=TARGET_FPR):
    """Smallest observed value tau such that the fraction of null z values strictly above tau is <= target."""
    z = np.sort(np.asarray(null_z, float)[np.isfinite(null_z)])[::-1]
    k = int(np.floor(target * len(z)))
    return float(z[k]) if len(z) > k else float("inf")


def null_from_oof(stat, T1, units):
    zs = []
    for u in np.unique(units):
        m = units == u
        d = person_z(stat[m], T1[m])
        zs += [d["z_c1"], d["z_c2"]]
    return np.array(zs, float)


# ----------------------------------------------------------------------------------------------
# One outer fold
# ----------------------------------------------------------------------------------------------
def _lr_arm(X, y, units, tr, te, T1):
    """Fit the screening LR pipeline (nested tuning on training participants), score the test rows, and
    return inner-out-of-fold training scores for the threshold calibration."""
    P = p2()
    est, params = P.fit_lr(X[tr], y[tr], units[tr], True)
    score_te = est.decision_function(X[te])
    oof = np.full(len(tr), np.nan)
    sgk = StratifiedGroupKFold(INNER_SPLITS, shuffle=True, random_state=0)
    for a, b in sgk.split(X[tr], y[tr], units[tr]):
        e = clone(est).fit(X[tr][a], y[tr][a])
        oof[b] = e.decision_function(X[tr][b])
    return score_te, score_te, oof, params


def _peak_arm(rate, y, units, tr, te):
    P = p2()
    ch = {"rate": rate}
    m = P.PeakRule("rate").fit(ch, tr, y[tr])
    score_te, stat_te = m.score(ch, te), m.raw(ch, te)
    oof = np.full(len(tr), np.nan)
    sgk = StratifiedGroupKFold(INNER_SPLITS, shuffle=True, random_state=0)
    for a, b in sgk.split(tr, y[tr], units[tr]):
        mi = P.PeakRule("rate").fit(ch, tr[a], y[tr][a])
        oof[b] = mi.raw(ch, tr[b])
    return score_te, stat_te, oof, {"peak_ms": m.peak * 10}


def fold_task(r, f, test_units, D):
    warnings.filterwarnings("ignore")
    with threadpool_limits(1):
        return _fold(r, f, test_units, D)


def _fold(r, f, test_units, D):
    units, exps, y, T1, X = D["units"], D["exps"], D["y"], D["T1"], D["X"]
    cols = D["cols"]
    te_all = np.isin(units.astype(str), test_units)
    trial_rows, person_rows = [], []

    def finish(arm, tr, te, score, stat, oof, params):
        """Per experiment: threshold from the training participants' inner-OOF control-1-vs-2 z values
        (FPR <= 7%), then every test participant's familiar z, control z values and detection."""
        for ee in (1, 2):
            trm, tm = exps[tr] == ee, exps[te] == ee
            if not tm.any():
                continue
            null = null_from_oof(oof[trm], T1[tr][trm], units[tr][trm])
            tau = threshold_for_fpr(null)
            for u in np.unique(units[te][tm]):
                m = units[te] == u
                d = person_z(stat[m], T1[te][m])
                person_rows.append({"arm": arm, "repeat": r, "fold": f, "unit": int(u), "exp": ee, **d, "tau": tau,
                                    "n_null": int(np.isfinite(null).sum()), "params": str(params),
                                    "detect_cal": float(d["z_fam"] > tau),
                                    "fpr_cal": float(((d["z_c1"] > tau) + (d["z_c2"] > tau)) / 2)})
            trial_rows.append(pd.DataFrame({"arm": arm, "repeat": r, "fold": f, "row": te[tm], "unit": units[te][tm],
                                            "exp": ee, "T1": T1[te][tm], "y": y[te][tm],
                                            "score": score[tm], "stat": stat[tm]}))

    # paper's rate rule: peak and calibration per experiment (as in screening)
    for e in (1, 2):
        tr = np.flatnonzero(~te_all & (exps == e))
        te = np.flatnonzero(te_all & (exps == e))
        score, stat, oof, params = _peak_arm(D["rate"], y, units, tr, te)
        finish("rate_rule", tr, te, score, stat, oof, params)
    # pooled LR arms: one fit on all training participants of both experiments, no experiment indicator
    tr, te = np.flatnonzero(~te_all), np.flatnonzero(te_all)
    for arm in ("I2b", "I2b_noMiss", "MissOnly"):
        score, stat, oof, params = _lr_arm(X[:, cols[arm]], y, units, tr, te, T1)
        finish(arm, tr, te, score, stat, oof, params)
    # sensitivity: screening-style I2b fits, one experiment at a time
    for e in (1, 2):
        tr = np.flatnonzero(~te_all & (exps == e))
        te = np.flatnonzero(te_all & (exps == e))
        score, stat, oof, params = _lr_arm(X[:, cols["I2b"]], y, units, tr, te, T1)
        finish("I2b_perexp", tr, te, score, stat, oof, params)
    return pd.concat(trial_rows, ignore_index=True), pd.DataFrame(person_rows)


# ----------------------------------------------------------------------------------------------
# Fast metrics for the bootstrap
# ----------------------------------------------------------------------------------------------
def fast_auc(y, s):
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_eer(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def fast_hter(y, s, thr=0.0):
    y = np.asarray(y)
    pred = np.asarray(s) >= thr
    pos, neg = y == 1, y == 0
    frr = 1 - pred[pos].mean() if pos.any() else np.nan
    far = pred[neg].mean() if neg.any() else np.nan
    return float((far + frr) / 2)


def person_level(zf, zc):
    """Screening's person-level set: positives = familiar z, negatives = control-1-vs-2 z, threshold z = 1.645."""
    y = np.r_[np.ones(len(zf)), np.zeros(len(zc))]
    s = np.r_[zf, zc] - Z_ALPHA
    ok = np.isfinite(s)
    return y[ok], s[ok]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj
