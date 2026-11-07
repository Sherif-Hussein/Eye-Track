"""Phase 2 screening, track `chen`: Chen, Mathot & Akyurek (2026) moving-RSVP familiar-face data.

Steps
 1. Reproduce the paper's individual detection rule (leave-one-participant-out peak, one-tailed
    t-test familiar vs averaged controls, alpha = .05) for pupil size and rate of pupil change.
 2. Freeze participant-disjoint outer folds (3 repeats x 10 grouped folds per experiment).
 3. Honest baseline = the paper's rule with the peak picked from training participants only.
 4. Screen ideas (window features, data-driven windows, within-person normalisation against own
    control trials, gaze+pupil fusion, MiniRocket, preprocessing ablations) on identical folds.
 5. Every model's held-out trial scores -> trial metrics (edd.metrics.binary_metrics), the paper's
    per-participant detection (one-tailed t-test of familiar vs control trial scores, alpha .05),
    false-positive rate of that rule on control-1 vs control-2 trials, and person-level metrics
    treating each participant's familiar test as a positive and their control-1-vs-control-2 test
    as a pseudo-innocent negative.
 6. Paired comparison vs the honest baseline (Wilcoxon on within-participant AUC, exact McNemar on
    detection, Holm), cluster-bootstrap CIs, permutation test, cross-experiment transfer, LOSO.

Usage: python -I scripts/phase2_chen.py [stage ...]   (stages: all | screen | best | cross)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edd import chen  # noqa: E402
from edd.folds import save_folds  # noqa: E402
from edd.metrics import binary_metrics, bootstrap_ci, summarise_repeats  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "results" / "phase2" / "chen"
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"exp1": {14, 15, 24, 30}, "exp2": {8, 15, 22, 28}}  # authors' scripts
PUBLISHED = {"exp1": {"N": 29, "pupil_size": 10, "pupil_rate": 16, "gaze_0_1000": 4, "gaze_1000_2500": 3},
             "exp2": {"N": 27, "pupil_size": 5, "pupil_rate": 9, "gaze_0_1000": 0, "gaze_1000_2500": 4}}
N_REPEATS, N_FOLDS, BASE_SEED = 3, 10, 2026
ALPHA = 0.05
N_JOBS = 2
FAM, C1, C2 = 2, 3, 4


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def load(exp):
    meta = pd.read_csv(DERIVED / f"chen_{exp}_meta.csv")
    z = np.load(DERIVED / f"chen_{exp}_traces.npz")
    keep = (meta.trialtype == 1) & ~meta.subject_nr.isin(EXCLUDE[exp]) & meta.T1.isin([1, 2, 3, 4])
    idx = np.flatnonzero(keep.to_numpy())
    return meta.iloc[idx].reset_index(drop=True), {k: z[k][idx].astype(float) for k in z.files}


def preprocess(meta, arr, pupil="p_dm", base="sub", base_n=5, length=250):
    """Authors' pipeline: baseline = mean of the 5 samples (50 ms) before critical-face onset;
    drop undefined baselines and per-participant baseline |z| > 2; subtractive correction; lock to
    onset; 0-2500 ms; rate = diff then Hanning smoothing (25 samples). Gaze: dx = (x - width/2) *
    T1_direction (toward the motion direction), dy = y - 540."""
    t1pos = meta.T1pos.to_numpy()
    P = arr[pupil]
    b = chen.baseline(P, t1pos, base_n)
    ok = np.isfinite(b)
    z = np.full(len(b), np.nan)
    for s in meta.subject_nr.unique():
        m = (meta.subject_nr.to_numpy() == s) & ok
        z[m] = (b[m] - b[m].mean()) / b[m].std()
    keep = ok & (np.abs(z) <= 2)
    ep = chen.epoch(P, t1pos, 0, length)
    pup = ep - b[:, None] if base == "sub" else ep / b[:, None] - 1.0
    rate = chen.smooth_hanning(np.diff(pup, axis=1), 25)
    x = chen.epoch(arr["x"], t1pos, 0, length)
    y = chen.epoch(arr["y"], t1pos, 0, length)
    dx = (x - meta.width.to_numpy()[:, None] / 2) * meta.T1_direction.to_numpy()[:, None]
    dy = y - 540.0
    miss = chen.epoch(arr["miss"], t1pos, 0, length)
    out = {"pupil": pup, "rate": rate, "dx": dx, "dy": dy, "miss": miss}
    info = {"n_trials_in": int(len(meta)), "n_undefined_baseline": int((~ok).sum()),
            "n_baseline_outliers": int((ok & ~keep).sum()), "n_kept": int(keep.sum())}
    m2 = meta.loc[keep].reset_index(drop=True)
    return m2, {k: v[keep] for k, v in out.items()}, info


def fc_subset(meta, ch):
    """Familiar vs control trials only (targets dropped)."""
    m = meta.T1.isin([FAM, C1, C2]).to_numpy()
    return meta.loc[m].reset_index(drop=True), {k: v[m] for k, v in ch.items()}


# ----------------------------------------------------------------------------------------------
# Paper's rule
# ----------------------------------------------------------------------------------------------
def peak_index(trace, is_fam, is_ctrl):
    d = np.abs(np.nanmean(trace[is_fam], 0) - np.nanmean(trace[is_ctrl], 0))
    return int(np.nanargmax(d))


def paper_loo(meta, ch, var):
    """Exact re-implementation of the authors' leave_one_out(): peak = argmax |mean familiar -
    mean control| over the other participants' pooled trials; one-tailed t-test at that sample."""
    subj = meta.subject_nr.to_numpy()
    T1 = meta.T1.to_numpy()
    fam, ctrl = T1 == FAM, np.isin(T1, [C1, C2])
    rows = []
    for s in np.unique(subj):
        tr, te = subj != s, subj == s
        pk = peak_index(ch[var], fam & tr, ctrl & tr)
        v = ch[var][:, pk]
        _, p = chen.one_tailed_t(v[te & fam], v[te & ctrl])
        _, p1 = chen.one_tailed_t(v[te & (T1 == C1)], v[te & (T1 == C2)])
        _, p2 = chen.one_tailed_t(v[te & (T1 == C2)], v[te & (T1 == C1)])
        rows.append({"subject": int(s), "peak_ms": pk * 10, "p": p, "p_c1c2": p1, "p_c2c1": p2,
                     "z_fam": _z_onesided(v[te & fam], v[te & ctrl]),
                     "z_c1": _z_onesided(v[te & (T1 == C1)], v[te & (T1 == C2)]),
                     "z_c2": _z_onesided(v[te & (T1 == C2)], v[te & (T1 == C1)])})
    r = pd.DataFrame(rows)
    dc, fc = calibrated(r)
    yp, sp = person_level_metrics(r)
    pm = binary_metrics(yp, sp)
    return {"detected": int((r.p < ALPHA).sum()), "N": len(r), "rate": float((r.p < ALPHA).mean()),
            "fpr_c1c2": float(((r.p_c1c2 < ALPHA).mean() + (r.p_c2c1 < ALPHA).mean()) / 2),
            "detection_calibrated": dc, "fpr_calibrated": fc,
            "fpr_random_split_controls": random_split_fpr([ch[var][(subj == u) & ctrl, int(r.peak_ms[r.subject == u].iloc[0]) // 10] for u in np.unique(subj)]),
            "fpr_excess_test": fpr_excess_test(r),
            "person_auc": pm["auc"], "person_eer": pm["eer"], "person_hter": pm["hter"],
            "peak_ms_median": float(r.peak_ms.median()), "per_subject": r.to_dict(orient="list")}


def paper_loo_gaze(meta, ch):
    """Gaze rule: dx in 0-1000 ms (familiar > control) and 1000-2500 ms (control > familiar)."""
    subj, T1 = meta.subject_nr.to_numpy(), meta.T1.to_numpy()
    fam, ctrl = T1 == FAM, np.isin(T1, [C1, C2])
    res = {}
    for name, a, b, sign in (("gaze_0_1000", 0, 100, 1), ("gaze_1000_2500", 100, 250, -1)):
        tr_ = ch["dx"][:, a:b] * sign
        det = 0
        for s in np.unique(subj):
            tr, te = subj != s, subj == s
            pk = peak_index(tr_, fam & tr, ctrl & tr)
            _, p = chen.one_tailed_t(tr_[te & fam, pk], tr_[te & ctrl, pk])
            det += p < ALPHA
        res[name] = int(det)
    return res


# ----------------------------------------------------------------------------------------------
# Folds
# ----------------------------------------------------------------------------------------------
def grouped_folds(units, n_folds, seed):
    rng = np.random.default_rng(seed)
    u = [str(x) for x in rng.permutation(sorted(units))]
    return [u[i::n_folds] for i in range(n_folds)]


# ----------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------
EDGES = np.arange(0, 251, 10)  # 100-ms bins over 0-2500 ms


def bin_features(ch, channels, edges=EDGES):
    feats, names = [], []
    for c in channels:
        e = edges if c != "rate" else np.minimum(edges, ch[c].shape[1])
        f = chen.bin_means(ch[c], e)
        feats.append(f)
        names += [f"{c}_{a * 10}_{b * 10}" for a, b in zip(e[:-1], e[1:])]
    return np.hstack(feats), names


def window_features(ch, windows):
    """Mean of a channel within fixed (ms) windows."""
    feats, names = [], []
    for c, a, b in windows:
        feats.append(np.nanmean(ch[c][:, a // 10:b // 10], 1)[:, None])
        names.append(f"{c}_{a}_{b}")
    return np.hstack(feats), names


def person_normalise(F, subj, is_ctrl):
    """Within-person z-scoring of each feature against the person's OWN control trials only (no
    familiar-trial information; trial types are known by design in a CIT)."""
    F = F.copy()
    for s in np.unique(subj):
        m = subj == s
        ref = F[m & is_ctrl]
        mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0)
        sd[~(sd > 0)] = 1.0
        F[m] = (F[m] - mu) / sd
    return F


# ----------------------------------------------------------------------------------------------
# Models (fit on training participants, score test participants)
# ----------------------------------------------------------------------------------------------
def lr_search(n_features, select=True):
    steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    grid = {"lr__C": [0.003, 0.03, 0.3]}
    if select:
        steps.append(("sel", SelectKBest(f_classif)))
        grid["sel__k"] = sorted({min(20, n_features), n_features})
    steps.append(("lr", LogisticRegression(class_weight="balanced", max_iter=2000)))
    return GridSearchCV(Pipeline(steps), grid, cv=StratifiedGroupKFold(3, shuffle=True, random_state=0),
                        scoring="roc_auc", n_jobs=1, refit=True)


def fit_lr(X, y, g, select=True):
    s = lr_search(X.shape[1], select)
    s.fit(X, y, groups=g)
    return s.best_estimator_, s.best_params_


class PeakRule:
    """Paper's rule as a model: peak picked on training participants; a 1-D logistic calibration
    (training trials only) maps the value at the peak to a signed score for threshold-0 metrics."""

    def __init__(self, var):
        self.var = var

    def fit(self, ch, idx, y):
        is_f, is_c = np.zeros(len(ch[self.var]), bool), np.zeros(len(ch[self.var]), bool)
        is_f[idx[y == 1]], is_c[idx[y == 0]] = True, True
        self.peak = peak_index(ch[self.var], is_f, is_c)
        v = ch[self.var][idx, self.peak]
        ok = np.isfinite(v)
        self.fill = float(np.nanmedian(v))
        self.cal = LogisticRegression(class_weight="balanced", C=1e6).fit(v[ok, None], y[ok])
        return self

    def raw(self, ch, idx):
        return ch[self.var][idx, self.peak]

    def score(self, ch, idx):
        v = self.raw(ch, idx)
        v = np.where(np.isfinite(v), v, self.fill)
        return self.cal.decision_function(v[:, None])


def rocket_input(ch, channels):
    X = np.stack([ch[c][:, :249] for c in channels], 1)  # (n, C, 249)
    n, C, L = X.shape
    t = np.arange(L)
    for i in range(n):
        for j in range(C):
            r = X[i, j]
            ok = np.isfinite(r)
            X[i, j] = np.interp(t, t[ok], r[ok]) if ok.sum() >= 2 else 0.0
    return X


def person_scale_channels(X, subj, is_ctrl):
    """Per-person, per-channel scalar z-scaling of traces using the person's control trials."""
    X = X.copy()
    for s in np.unique(subj):
        m = subj == s
        ref = X[m & is_ctrl]
        mu = ref.mean(axis=(0, 2), keepdims=True)
        sd = ref.std(axis=(0, 2), keepdims=True)
        sd[sd == 0] = 1
        X[m] = (X[m] - mu) / sd
    return X


# ----------------------------------------------------------------------------------------------
# Running one idea on frozen folds
# ----------------------------------------------------------------------------------------------
def _fold_task(spec, data, r, f, test_units):
    warnings.filterwarnings("ignore")
    with threadpool_limits(1):
        return _fold_task_inner(spec, data, r, f, test_units)


def _fold_task_inner(spec, data, r, f, test_units):
    meta, y, subj = data["meta"], data["y"], data["subj"]
    te = np.isin(subj.astype(str), test_units)
    tr_idx, te_idx = np.flatnonzero(~te), np.flatnonzero(te)
    kind = spec["kind"]
    params = None
    if kind == "peak":
        mdl = PeakRule(spec["var"]).fit(data["ch"], tr_idx, y[tr_idx])
        score, stat = mdl.score(data["ch"], te_idx), mdl.raw(data["ch"], te_idx)
        params = {"peak_ms": mdl.peak * 10}
    elif kind == "lr":
        X = data["X"]
        est, params = fit_lr(X[tr_idx], y[tr_idx], subj[tr_idx], spec.get("select", True))
        score = est.decision_function(X[te_idx])
        stat = score
    elif kind == "rocket":
        F = data["F"]
        pipe = Pipeline([("sc", StandardScaler()), ("rc", RidgeClassifier(class_weight="balanced"))])
        gs = GridSearchCV(pipe, {"rc__alpha": [1e2, 1e3, 1e4]}, scoring="roc_auc", n_jobs=1,
                          cv=StratifiedGroupKFold(3, shuffle=True, random_state=0))
        gs.fit(F[tr_idx], y[tr_idx], groups=subj[tr_idx])
        score = gs.best_estimator_.decision_function(F[te_idx])
        stat, params = score, gs.best_params_
    else:
        raise ValueError(kind)
    return pd.DataFrame({"repeat": r, "fold": f, "unit": subj[te_idx], "T1": meta.T1.to_numpy()[te_idx],
                         "row": te_idx, "y": y[te_idx], "score": score, "stat": stat, "params": str(params)})


def run_idea(spec, data, folds):
    tasks = [(r, f, t) for r, fl in enumerate(folds) for f, t in enumerate(fl)]
    out = Parallel(n_jobs=N_JOBS)(delayed(_fold_task)(spec, data, r, f, t) for r, f, t in tasks)
    return pd.concat(out, ignore_index=True)


def rocket_features(ch, subj, is_c, spec, fit_on):
    """MiniRocket's kernels and bias quantiles are unsupervised but data-dependent: they are fit on
    `fit_on` (the OTHER experiment's participants, or the training experiment for transfer), so no
    test participant ever shapes the transform; only the ridge head is fit per outer fold."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    R = rocket_input(ch, spec["channels"])
    if spec.get("pnorm"):
        R = person_scale_channels(R, subj, is_c)
    rk = MiniRocket(n_kernels=spec.get("n_kernels", 1000), random_state=0, n_jobs=1).fit(fit_on)
    F = rk.transform(R)
    if spec.get("pnorm_feats"):
        F = person_normalise(F, subj, is_c)
    return F, rk


def rocket_source(meta, ch, spec):
    y = (meta.T1.to_numpy() == FAM).astype(int)
    R = rocket_input(ch, spec["channels"])
    return person_scale_channels(R, meta.subject_nr.to_numpy(), y == 0) if spec.get("pnorm") else R


def build_data(meta, ch, spec, other=None):
    y = (meta.T1.to_numpy() == FAM).astype(int)
    subj = meta.subject_nr.to_numpy()
    d = {"meta": meta, "y": y, "subj": subj, "ch": ch}
    is_c = y == 0
    if spec["kind"] == "lr":
        if "windows" in spec:
            X, names = window_features(ch, spec["windows"])
        else:
            X, names = bin_features(ch, spec["channels"])
        if spec.get("pnorm"):
            X = person_normalise(X, subj, is_c)
        d["X"], d["names"] = X, names
    elif spec["kind"] == "rocket":
        d["F"], _ = rocket_features(ch, subj, is_c, spec, rocket_source(*other, spec))
    return d


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def _z_onesided(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    t, p2 = stats.ttest_ind(a, b)
    if not np.isfinite(t):
        return 0.0
    p = stats.t.sf(t, len(a) + len(b) - 2)
    return float(stats.norm.isf(np.clip(p, 1e-300, 1)))


def person_tests(tr):
    """Per participant: paper's detection (familiar vs both controls) and pseudo-innocent tests
    (control 1 vs control 2 in both directions) on the `stat` column."""
    rows = []
    for u, d in tr.groupby("unit"):
        s, T1 = d.stat.to_numpy(), d.T1.to_numpy()
        _, p = chen.one_tailed_t(s[T1 == FAM], s[T1 != FAM])
        _, p12 = chen.one_tailed_t(s[T1 == C1], s[T1 == C2])
        _, p21 = chen.one_tailed_t(s[T1 == C2], s[T1 == C1])
        y, sc = d.y.to_numpy(), d.score.to_numpy()
        auc = binary_metrics(y, sc)["auc"] if y.min() != y.max() else np.nan
        rows.append({"unit": u, "p": p, "p_c1c2": p12, "p_c2c1": p21,
                     "z_fam": _z_onesided(s[T1 == FAM], s[T1 != FAM]),
                     "z_c1": _z_onesided(s[T1 == C1], s[T1 == C2]),
                     "z_c2": _z_onesided(s[T1 == C2], s[T1 == C1]), "within_auc": auc})
    return pd.DataFrame(rows)


def random_split_fpr(stat_by_unit, n_splits=500, seed=0):
    """Rejection rate of the one-tailed rule when one person's CONTROL trials are randomly split
    into two equal halves. Trial types are randomised across the session, so the two halves are
    exchangeable and differ in no image: this measures the test's own calibration. A rate near
    alpha here, alongside an inflated control-1-vs-control-2 rate, implicates image identity."""
    rng = np.random.default_rng(seed)
    rej, n = 0, 0
    for v in stat_by_unit:
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        for _ in range(n_splits):
            ix = rng.permutation(len(v))
            a, b = v[ix[: len(v) // 2]], v[ix[len(v) // 2:]]
            rej += chen.one_tailed_t(a, b)[1] < ALPHA
            n += 1
    return float(rej / n)


def fpr_excess_test(pt):
    """A person is a false alarm if either direction rejects (the directions are mutually
    exclusive), so the count is Binomial(N, .10) under a calibrated test."""
    k = int(((pt.p_c1c2 < ALPHA) | (pt.p_c2c1 < ALPHA)).sum())
    return {"false_alarm_persons": k, "N": int(len(pt)),
            "binom_p_vs_nominal_10pct": float(stats.binomtest(k, len(pt), 2 * ALPHA, alternative="greater").pvalue)}


def calibrated(pt, q=0.95):
    """Detection at an empirically calibrated threshold: for each participant, the cut-off is the
    95th percentile of the OTHER participants' out-of-fold control-1-vs-control-2 statistics (both
    directions), i.e. a leave-one-participant-out estimate of the innocent null."""
    zf, z1, z2 = pt.z_fam.to_numpy(), pt.z_c1.to_numpy(), pt.z_c2.to_numpy()
    det, fp = [], []
    for i in range(len(pt)):
        o = np.arange(len(pt)) != i
        thr = np.nanquantile(np.r_[z1[o], z2[o]], q)
        det.append(zf[i] > thr)
        fp.append(((z1[i] > thr) + (z2[i] > thr)) / 2)
    return float(np.mean(det)), float(np.mean(fp))


def person_level_metrics(pt):
    """Positives: each participant's familiar-vs-control test; negatives: the same participant's
    control-1-vs-control-2 test (pseudo-innocent). Score = z - 1.645 so threshold 0 <=> p < .05."""
    y = np.r_[np.ones(len(pt)), np.zeros(len(pt))]
    s = np.r_[pt.z_fam.to_numpy(), pt.z_c1.to_numpy()] - stats.norm.isf(ALPHA)
    ok = np.isfinite(s)
    return y[ok], s[ok]


def evaluate_trials(trials):
    reps = []
    for r, t in trials.groupby("repeat"):
        m = binary_metrics(t.y, t.score)
        pt = person_tests(t)
        m["detection_rate"] = float((pt.p < ALPHA).mean())
        m["n_detected"] = int((pt.p < ALPHA).sum())
        m["fpr_c1c2"] = float(((pt.p_c1c2 < ALPHA).mean() + (pt.p_c2c1 < ALPHA).mean()) / 2)
        m["within_auc_mean"] = float(pt.within_auc.mean())
        m["detection_calibrated"], m["fpr_calibrated"] = calibrated(pt)
        yp, sp = person_level_metrics(pt)
        pm = binary_metrics(yp, sp)
        for k in ("auc", "eer", "hter", "balanced_accuracy", "mcc"):
            m[f"person_{k}"] = pm[k]
        reps.append(m)
    return summarise_repeats(reps), reps


def averaged(trials):
    """Repeat-averaged score per trial (each trial is tested once per repeat)."""
    return trials.groupby("row").agg(unit=("unit", "first"), T1=("T1", "first"), y=("y", "first"),
                                     score=("score", "mean"), stat=("stat", "mean")).reset_index()


def full_ci(trials, n_boot=1000, seed=0):
    """Point estimates on repeat-averaged scores + participant-cluster bootstrap 95% CIs."""
    avg = averaged(trials)
    pt = person_tests(avg)
    units = pt.unit.to_numpy()
    by_unit = {u: d for u, d in avg.groupby("unit")}
    pt_idx = pt.set_index("unit")
    point = binary_metrics(avg.y, avg.score)
    point["detection_rate"] = float((pt.p < ALPHA).mean())
    point["fpr_c1c2"] = float(((pt.p_c1c2 < ALPHA).mean() + (pt.p_c2c1 < ALPHA).mean()) / 2)
    yp, sp = person_level_metrics(pt)
    pm = binary_metrics(yp, sp)
    point.update({f"person_{k}": pm[k] for k in ("auc", "eer", "hter")})
    point["detection_calibrated"], point["fpr_calibrated"] = calibrated(pt)
    rng = np.random.default_rng(seed)
    keys = ["auc", "eer", "hter", "balanced_accuracy", "detection_rate", "fpr_c1c2",
            "person_auc", "person_eer", "person_hter", "detection_calibrated", "fpr_calibrated"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(units, len(units), replace=True)
        t = pd.concat([by_unit[u] for u in pick])
        m = binary_metrics(t.y, t.score)
        p = pt_idx.loc[pick]
        yp, sp = person_level_metrics(p.reset_index())
        pm = binary_metrics(yp, sp)
        vals = {"auc": m["auc"], "eer": m["eer"], "hter": m["hter"], "balanced_accuracy": m["balanced_accuracy"],
                "detection_rate": float((p.p < ALPHA).mean()),
                "fpr_c1c2": float(((p.p_c1c2 < ALPHA).mean() + (p.p_c2c1 < ALPHA).mean()) / 2),
                "person_auc": pm["auc"], "person_eer": pm["eer"], "person_hter": pm["hter"]}
        vals["detection_calibrated"], vals["fpr_calibrated"] = calibrated(p.reset_index())
        for k in keys:
            draws[k].append(vals[k])
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}
    # protocol's edd.metrics.bootstrap_ci on the person-level rows (positives/negatives resampled)
    yp, sp = person_level_metrics(pt)
    ci_person_edd = bootstrap_ci(yp, sp, n_boot=n_boot)
    point["fpr_random_split_controls"] = random_split_fpr([d.stat[d.T1 != FAM].to_numpy() for _, d in by_unit.items()], n_splits=200)
    return {"point": point, "ci95_cluster": ci, "ci95_person_rows_edd": ci_person_edd,
            "fpr_excess_test": fpr_excess_test(pt),
            "n_detected": int((pt.p < ALPHA).sum()), "N": int(len(pt)),
            "clopper_pearson_detection": list(map(float, stats.binomtest(int((pt.p < ALPHA).sum()), len(pt)).proportion_ci()))}, pt


def paired_vs(base_pt, idea_pt):
    """Wilcoxon on within-participant AUC; exact McNemar on detection (repeat-averaged scores)."""
    a = base_pt.set_index("unit").loc[idea_pt.unit]
    b = idea_pt.set_index("unit")
    d = b.within_auc.to_numpy() - a.within_auc.to_numpy()
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    da, db = a.p.to_numpy() < ALPHA, b.p.to_numpy() < ALPHA
    n01, n10 = int((~da & db).sum()), int((da & ~db).sum())
    mc = stats.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
    return {"delta_within_auc_mean": float(d.mean()), "wilcoxon_p": float(w.pvalue) if w else 1.0,
            "detect_gained": n01, "detect_lost": n10, "mcnemar_p": float(mc)}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ----------------------------------------------------------------------------------------------
# Idea catalogue
# ----------------------------------------------------------------------------------------------
PUP = ["pupil", "rate"]
FUSION = ["pupil", "rate", "dx", "dy", "miss"]
PAPER_WINDOWS = {  # FDR-significant familiar-vs-control clusters reported by the paper
    "exp1": [("pupil", 800, 1570), ("pupil", 1930, 2500), ("rate", 660, 860), ("rate", 1370, 1660)],
    "exp2": [("pupil", 1530, 2480), ("rate", 1510, 1730)],
}


def idea_specs(exp):
    return {
        "B0_size_peak": {"kind": "peak", "var": "pupil", "desc": "Paper rule: pupil size at training-set peak (honest baseline)"},
        "B0_rate_peak": {"kind": "peak", "var": "rate", "desc": "Paper rule: rate of pupil change at training-set peak (honest baseline, paper's best measure)"},
        "I2a_bins_pupil": {"kind": "lr", "channels": PUP, "desc": "100-ms bins of pupil size+rate, SelectKBest(k in {20, all}, training folds) + logistic"},
        "I2b_bins_fusion": {"kind": "lr", "channels": FUSION, "desc": "I2a + gaze dx, dy and missing-sample fraction"},
        "I2c_paperwin_pupil": {"kind": "lr", "windows": PAPER_WINDOWS[exp], "select": False,
                               "desc": "Mean pupil/rate in the paper's group-level significant windows + logistic (windows chosen on the full sample: optimistic)"},
        "I3a_pnorm_pupil": {"kind": "lr", "channels": PUP, "pnorm": True, "desc": "I2a with within-person z-scoring against own control trials"},
        "I3b_pnorm_fusion": {"kind": "lr", "channels": FUSION, "pnorm": True, "desc": "I2b with within-person z-scoring against own control trials"},
        "I4a_minirocket_fusion": {"kind": "rocket", "channels": ["pupil", "rate", "dx", "dy"], "pnorm": True,
                                  "desc": "MiniRocket (1k kernels; transform fit on the other experiment) on person-scaled pupil, rate, dx, dy + ridge (alpha tuned in grouped inner CV)"},
        "I4b_minirocket_fusion_pnormfeat": {"kind": "rocket", "channels": ["pupil", "rate", "dx", "dy"], "pnorm": True, "pnorm_feats": True,
                                            "desc": "I4a + within-person z-scoring of MiniRocket features against own control trials"},
    }


ABLATIONS = {  # preprocessing variants of the I3b pipeline (Mathot & Vilotijevic 2023 guidelines)
    "A0_gapinterp": {"pupil": "p_rec", "base": "sub", "base_n": 5, "desc": "Gap-interpolating blink handling (all gaps up to 500 ms filled) instead of the authors' advanced reconstruction"},
    "A1_noblinkrecon": {"pupil": "p_raw", "base": "sub", "base_n": 5, "desc": "No blink reconstruction (blink/missing samples left NaN, median-imputed after binning)"},
    "A2_divisive": {"pupil": "p_dm", "base": "div", "base_n": 5, "desc": "Divisive baseline (pupil / baseline - 1)"},
    "A3_base200": {"pupil": "p_dm", "base": "sub", "base_n": 20, "desc": "Subtractive baseline over 200 ms instead of 50 ms"},
}


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def summary_row(name, exp, summ, desc):
    row = {"exp": exp, "idea": name, "desc": desc}
    for k, v in summ.items():
        row[f"{k}_mean"], row[f"{k}_sd"] = v["mean"], v["sd"]
    return row


def main(stages):
    results = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    prep, folds = {}, {}
    for exp in ("exp1", "exp2"):
        meta, arr = load(exp)
        m, ch, info = preprocess(meta, arr)
        prep[exp] = (meta, arr, m, ch, info)
        units = sorted(m.subject_nr.unique())
        folds[exp] = [grouped_folds(units, N_FOLDS, BASE_SEED + r) for r in range(N_REPEATS)]
    fpath = OUT / "folds.json"
    if not fpath.exists():
        save_folds(fpath, {e: folds[e] for e in folds},
                   meta={"design": f"{N_REPEATS} repeats x {N_FOLDS} participant-disjoint folds per experiment",
                         "seeds": [BASE_SEED + r for r in range(N_REPEATS)],
                         "unit": "subject_nr", "excluded": {k: sorted(v) for k, v in EXCLUDE.items()}})
    frozen = json.loads(fpath.read_text())["repeats"]

    if "all" in stages or "repro" in stages:
        repro = {}
        for exp in ("exp1", "exp2"):
            meta, arr, m, ch, info = prep[exp]
            mf, chf = fc_subset(m, ch)
            repro[exp] = {"preprocessing": info, "n_subjects": int(m.subject_nr.nunique()),
                          "trials_per_type": {int(k): int(v) for k, v in mf.T1.value_counts().items()},
                          "pupil_size": paper_loo(mf, chf, "pupil"), "pupil_rate": paper_loo(mf, chf, "rate"),
                          "gaze": paper_loo_gaze(mf, chf), "published": PUBLISHED[exp]}
            m_a, ch_a, _ = preprocess(meta, arr, "p_rec")
            mf_a, chf_a = fc_subset(m_a, ch_a)
            repro[exp]["sensitivity_gap_interpolation"] = {k: {a: paper_loo(mf_a, chf_a, v)[a] for a in ("detected", "N", "fpr_c1c2")}
                                                           for k, v in (("pupil_size", "pupil"), ("pupil_rate", "rate"))}
            repro[exp]["gaze_range_px"] = {k: [float(np.nanmin(ch[k])), float(np.nanmax(ch[k]))] for k in ("dx", "dy")}
            print(exp, info, {k: repro[exp][k]["detected"] for k in ("pupil_size", "pupil_rate")}, repro[exp]["gaze"], repro[exp]["gaze_range_px"], flush=True)
            for k in ("pupil_size", "pupil_rate"):
                print("  ", k, {a: repro[exp][k][a] for a in ("rate", "fpr_c1c2", "fpr_random_split_controls", "fpr_excess_test", "detection_calibrated", "fpr_calibrated", "person_auc", "person_eer", "person_hter", "peak_ms_median")}, flush=True)
        results["reproduction_paper_protocol"] = repro

    def save():
        tmp = OUT / "results.json.tmp"
        tmp.write_text(json.dumps(results, indent=1, default=float))
        tmp.replace(OUT / "results.json")

    save()

    def checkpoint_idea(exp, name, desc, compute, extra=None):
        """Checkpointing: an idea whose metrics file exists is skipped; otherwise its out-of-fold
        scores and metrics are written the moment it finishes."""
        mpath = OUT / "ideas" / f"metrics_{exp}_{name}.json"
        mpath.parent.mkdir(exist_ok=True)
        if mpath.exists():
            row = json.loads(mpath.read_text())
            print(f"{exp} {name}: done earlier, skipped", flush=True)
        else:
            t0 = time.perf_counter()
            tr = compute()
            summ, reps = evaluate_trials(tr)
            row = summary_row(name, exp, summ, desc) | {"seconds": time.perf_counter() - t0} | (extra or {})
            tmp = OUT / f"scores_{exp}_{name}.tmp.csv.gz"
            tr.to_csv(tmp, index=False)
            tmp.replace(OUT / f"scores_{exp}_{name}.csv.gz")
            mpath.write_text(json.dumps(row | {"per_repeat": reps}, indent=1, default=float))
            print(f"{exp} {name}: AUC {summ['auc']['mean']:.3f}  det {summ['detection_rate']['mean']:.3f} "
                  f"fpr {summ['fpr_c1c2']['mean']:.3f}  detcal {summ['detection_calibrated']['mean']:.3f} "
                  f"personAUC {summ['person_auc']['mean']:.3f}  EER {summ['eer']['mean']:.3f} ({row['seconds']:.0f}s)", flush=True)
        row = {k: v for k, v in row.items() if k != "per_repeat"}
        rows = [r for r in results.get("screening", []) if not (r["exp"] == exp and r["idea"] == name)] + [row]
        results["screening"] = rows
        pd.DataFrame(rows).to_csv(OUT / "ideas_table.csv", index=False)
        save()

    if "all" in stages or "screen" in stages:
        for exp in ("exp1", "exp2"):
            meta, arr, m, ch, info = prep[exp]
            mf, chf = fc_subset(m, ch)
            specs = idea_specs(exp)
            other = fc_subset(*prep["exp2" if exp == "exp1" else "exp1"][2:4])
            for name, spec in specs.items():
                checkpoint_idea(exp, name, spec["desc"],
                                lambda spec=spec: run_idea(spec, build_data(mf, chf, spec, other), frozen[exp]))
            for name, ab in ABLATIONS.items():
                m2, ch2, info2 = preprocess(meta, arr, ab["pupil"], ab["base"], ab["base_n"])
                mf2, chf2 = fc_subset(m2, ch2)
                spec = dict(idea_specs(exp)["I3b_pnorm_fusion"])
                ufold = [[[u for u in f if int(u) in set(mf2.subject_nr)] for f in rep] for rep in frozen[exp]]
                checkpoint_idea(exp, name, "I3b with " + ab["desc"],
                                lambda spec=spec, mf2=mf2, chf2=chf2, ufold=ufold: run_idea(spec, build_data(mf2, chf2, spec), ufold),
                                {"n_kept": info2["n_kept"]})

    if "all" in stages or "compare" in stages:
        comp = {}
        for exp in ("exp1", "exp2"):
            names = [p.name[len(f"scores_{exp}_"):-len(".csv.gz")] for p in sorted(OUT.glob(f"scores_{exp}_*.csv.gz"))
                     if "LOSO" not in p.name and "perm" not in p.name and ".tmp" not in p.name]
            pts = {n: person_tests(averaged(pd.read_csv(OUT / f"scores_{exp}_{n}.csv.gz"))) for n in names}
            res = {}
            for base in ("B0_rate_peak", "B0_size_peak"):
                others = [n for n in names if not n.startswith("B0")]
                pv = [paired_vs(pts[base], pts[n]) for n in others]
                adj_w = holm([p["wilcoxon_p"] for p in pv])
                adj_m = holm([p["mcnemar_p"] for p in pv])
                for n, p, aw, am in zip(others, pv, adj_w, adj_m):
                    p["wilcoxon_p_holm"], p["mcnemar_p_holm"] = float(aw), float(am)
                res[f"vs_{base}"] = dict(zip(others, pv))
            comp[exp] = res
        results["paired_comparisons"] = comp
        save()

    best = next((s.split("=", 1)[1] for s in stages if s.startswith("best=")), "I3b_pnorm_fusion")
    n_perm = int(next((s.split("=", 1)[1] for s in stages if s.startswith("perm=")), "100"))

    if "all" in stages or "best" in stages:
        bres = {"best_candidate": best}
        for exp in ("exp1", "exp2"):
            meta, arr, m, ch, info = prep[exp]
            mf, chf = fc_subset(m, ch)
            other = fc_subset(*prep["exp2" if exp == "exp1" else "exp1"][2:4])
            e = {}
            for name in (best, "B0_rate_peak", "B0_size_peak"):
                ci, pt = full_ci(pd.read_csv(OUT / f"scores_{exp}_{name}.csv.gz"))
                e[name] = ci
                pt.to_csv(OUT / f"person_tests_{exp}_{name}.csv", index=False)
            # LOSO (the paper's leave-one-out design) for best + both baselines
            loso = [[[str(u)] for u in sorted(mf.subject_nr.unique())]]
            for name in (best, "B0_rate_peak", "B0_size_peak"):
                spec = idea_specs(exp)[name] if name in idea_specs(exp) else None
                tr = run_idea(spec, build_data(mf, chf, spec, other), loso)
                tr.to_csv(OUT / f"scores_LOSO_{exp}_{name}.csv.gz", index=False)
                ci, _ = full_ci(tr)
                e[f"LOSO_{name}"] = ci
                print(exp, "LOSO", name, ci["n_detected"], "/", ci["N"], "AUC", round(ci["point"]["auc"], 3),
                      "FPR", round(ci["point"]["fpr_c1c2"], 3), flush=True)
            # Within-participant label permutation (which 48 of the person's familiar/control trials
            # are "familiar" is shuffled; normalisation and the whole nested pipeline are rerun)
            spec = idea_specs(exp)[best]
            obs = evaluate_trials(pd.read_csv(OUT / f"scores_{exp}_{best}.csv.gz").query("repeat == 0"))[0]
            rng = np.random.default_rng(7)
            null_auc, null_det = [], []
            t0 = time.perf_counter()
            for i in range(n_perm):
                mp = mf.copy()
                T1 = mp.T1.to_numpy().copy()
                for s in mp.subject_nr.unique():
                    ix = np.flatnonzero(mp.subject_nr.to_numpy() == s)
                    T1[ix] = rng.permutation(T1[ix])
                mp["T1"] = T1
                tr = run_idea(spec, build_data(mp, chf, spec, other), [frozen[exp][0]])
                summ, _ = evaluate_trials(tr)
                null_auc.append(summ["auc"]["mean"])
                null_det.append(summ["detection_rate"]["mean"])
            null_auc, null_det = np.array(null_auc), np.array(null_det)
            e["permutation"] = {"n_perm": n_perm, "scheme": "within-participant shuffle of familiar/control/control labels, full pipeline rerun, repeat-0 folds",
                                "observed_auc": obs["auc"]["mean"], "null_auc_mean": float(null_auc.mean()), "null_auc_p95": float(np.percentile(null_auc, 95)),
                                "p_auc": float((1 + (null_auc >= obs["auc"]["mean"]).sum()) / (1 + n_perm)),
                                "observed_detection": obs["detection_rate"]["mean"], "null_detection_mean": float(null_det.mean()),
                                "null_detection_p95": float(np.percentile(null_det, 95)),
                                "p_detection": float((1 + (null_det >= obs["detection_rate"]["mean"]).sum()) / (1 + n_perm)),
                                "seconds": time.perf_counter() - t0}
            print(exp, "perm", e["permutation"], flush=True)
            # Mechanism: coefficients of the best model refit on all participants of this experiment
            if spec["kind"] == "lr":
                data = build_data(mf, chf, spec)
                est, params = fit_lr(data["X"], data["y"], data["subj"], spec.get("select", True))
                names = np.array(data["names"])
                if "sel" in est.named_steps:
                    names = names[est.named_steps["sel"].get_support()]
                coef = est.named_steps["lr"].coef_.ravel()
                imp = pd.DataFrame({"feature": names, "coef_std": coef})
                imp["channel"] = imp.feature.str.split("_").str[0]
                imp.sort_values("coef_std", key=np.abs, ascending=False).to_csv(OUT / f"coefficients_{exp}_{best}.csv", index=False)
                e["coefficients"] = {"params": params, "top10": imp.reindex(imp.coef_std.abs().sort_values(ascending=False).index).head(10).to_dict(orient="records"),
                                     "abs_coef_share_by_channel": (imp.groupby("channel").coef_std.apply(lambda c: c.abs().sum()) / imp.coef_std.abs().sum()).to_dict()}
                # group-level effect sizes of the person-normalised features (familiar vs control)
                X, y = data["X"], data["y"]
                d = (np.nanmean(X[y == 1], 0) - np.nanmean(X[y == 0], 0))
                eff = pd.DataFrame({"feature": data["names"], "mean_fam_minus_ctrl_z": d})
                eff.to_csv(OUT / f"effects_{exp}_{best}.csv", index=False)
            bres[exp] = e
            results["best"] = bres
            save()
        results["best"] = bres
        save()

    if "all" in stages or "cross" in stages:
        cres = {}
        for src, tgt in (("exp1", "exp2"), ("exp2", "exp1")):
            ms, cs = fc_subset(*prep[src][2:4])
            mt, ct_ = fc_subset(*prep[tgt][2:4])
            ms = ms.assign(subject_nr=ms.subject_nr + 1000)  # keep ids disjoint across experiments
            mm = pd.concat([ms, mt], ignore_index=True)
            cc = {k: np.vstack([cs[k], ct_[k]]) for k in cs}
            fold = [[[str(u) for u in sorted(mt.subject_nr.unique())]]]
            r = {}
            for name in (best, "B0_rate_peak", "B0_size_peak"):
                spec = idea_specs(src)[name]
                tr = run_idea(spec, build_data(mm, cc, spec, (ms, cs)), fold)
                tr.to_csv(OUT / f"scores_cross_{src}to{tgt}_{name}.csv.gz", index=False)
                ci, _ = full_ci(tr)
                r[name] = ci
                print("cross", src, "->", tgt, name, ci["n_detected"], "/", ci["N"], "AUC", round(ci["point"]["auc"], 3),
                      "FPR", round(ci["point"]["fpr_c1c2"], 3), flush=True)
            cres[f"{src}_to_{tgt}"] = r
            results["cross_experiment"] = cres
            save()
        results["cross_experiment"] = cres

    save()


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
