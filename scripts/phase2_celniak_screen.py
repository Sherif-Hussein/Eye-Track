"""Phase 2 (celniak track): idea screening on Celniak et al. (2026) under the mandatory protocol.

Frozen outer folds: one guilty + one innocent participant per test fold, matched on test order
(eye tracker first vs polygraph first) wherever possible; 3 repeats. Everything (scaling, imputation,
feature selection, hyperparameters, valid-fraction thresholds, order residualisation, MiniRocket kernels)
is fitted on training participants only, with participant-grouped inner folds.

Usage: python -I scripts/phase2_celniak_screen.py <derived_dir> <out_dir> [--repeats 3] [--n-perm 100] [--jobs 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin, clone  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from edd import efficiency as eff  # noqa: E402
from edd.celniak import PAPER_FEATURES, load_cached, slide_features  # noqa: E402
from edd.folds import save_folds  # noqa: E402
from edd.harness import (aggregate, evaluate, grouped_inner, make_search, nested_fit,  # noqa: E402
                         run_outer, scores_of)
from edd.metrics import binary_metrics, bootstrap_ci  # noqa: E402

LABEL, UNIT = "label", "recording"
SLIDE_KEYS = ["recording", "question", "repetition"]
# Reduced version of the paper's Table 2 SVM grid (6 x 6 x 6 = 216 in Phase 1): screening budget.
SVM_GRID = {"select__k": [5, 7, "all"], "clf__C": [0.1, 1, 10, 100], "clf__gamma": [0.001, 0.01, 0.1]}
LR_C = [0.1, 1.0]
INNER_SPLITS = 3  # participant-grouped inner folds; 3 (not 5) to fit the shared-CPU screening budget
DIFF_METRICS = ["dwell_share", "nfix", "nvisit", "entry_latency", "first_visit", "is_first", "pupil_on"]
QUALITY_THRESHOLDS = [0.0, 0.6, 0.8]
# Added after the full-sample univariate table was inspected, so they explain a result and are not candidates.
ABLATIONS = {"global_gaze_no_fixdur_lr", "fixdur_only_lr"}
# Ideas whose features depend on which item is the probe; everything else is a general state measure.
PROBE_SPECIFIC = ["paper_mean_lr", "orient_lr", "diff_lr", "diff_quality_lr", "diff_order_cov_lr", "diff_order_resid_lr"]


# ============================================================ folds

def order_matched_folds(per_p, order, seed):
    """Pairs of (guilty, innocent) participants with the same test order where possible, then across orders."""
    rng = np.random.default_rng(seed)
    folds, left_pos, left_neg = [], [], []
    for o in sorted(order.unique()):
        pos = list(rng.permutation(per_p.index[(per_p == 1) & (order == o)].to_numpy()))
        neg = list(rng.permutation(per_p.index[(per_p == 0) & (order == o)].to_numpy()))
        k = min(len(pos), len(neg))
        folds += [[str(p), str(n)] for p, n in zip(pos[:k], neg[:k])]
        left_pos += pos[k:]
        left_neg += neg[k:]
    left_pos, left_neg = list(rng.permutation(left_pos)), list(rng.permutation(left_neg))
    k = min(len(left_pos), len(left_neg))
    folds += [[str(p), str(n)] for p, n in zip(left_pos[:k], left_neg[:k])]
    folds += [[str(u)] for u in left_pos[k:] + left_neg[k:]]
    return folds


# ============================================================ participant-level feature tables

def _quadrants(items):
    """Item position per (slide, item) from gaze centroids pooled over all recordings (geometry only, no labels)."""
    c = items.groupby(["question", "repetition", "item"])[["cx", "cy"]].median()
    ymid = float(np.nanmedian(c["cy"]))
    q = (c["cx"] > 960).astype(int) + 2 * (c["cy"] > ymid).astype(int)
    return q.rename("quad")


def diff_features(items, trials, min_valid=0.0, suffix=""):
    """Within-person probe-minus-irrelevant effects, position-corrected and z-scored within participant.

    For each metric: mean signed effect over slides, its absolute value (concealers may over-attend or
    avoid), a t-like statistic, and consistency across the 3 repetitions of each question."""
    it = items.merge(_quadrants(items), left_on=["question", "repetition", "item"], right_index=True, how="left")
    it["entry_latency"] = it["entry_latency"].fillna(7500.0)
    tot = it.groupby(SLIDE_KEYS)["dwell"].transform("sum")
    it["dwell_share"] = np.where(tot > 0, it["dwell"] / tot.replace(0, np.nan), 0.25)
    # quality filter, chosen per participant: slides with valid_frac >= threshold (at least 5 slides kept)
    if min_valid > 0:
        vf = trials.set_index(SLIDE_KEYS)["valid_frac"]
        keep = []
        for rec, g in vf.groupby(level=0):
            ok = g[g >= min_valid]
            if len(ok) < 5:
                ok = g.sort_values(ascending=False).iloc[:5]
            keep += list(ok.index)
        it = it.set_index(SLIDE_KEYS).loc[lambda d: d.index.isin(keep)].reset_index()
    out = {}
    for m in DIFF_METRICS:
        v = it[m].astype(float)
        # position correction within participant, then z-score within participant
        v = v - it.groupby(["recording", "quad"])[m].transform("mean")
        z = (v - v.groupby(it["recording"]).transform("mean")) / v.groupby(it["recording"]).transform("std")
        tmp = it[SLIDE_KEYS + ["is_probe"]].assign(z=z.to_numpy())
        pr = tmp[tmp.is_probe == 1].set_index(SLIDE_KEYS)["z"]
        ir = tmp[tmp.is_probe == 0].groupby(SLIDE_KEYS)["z"].mean()
        d = (pr - ir).dropna().reset_index(name="d")
        g = d.groupby("recording")["d"]
        mean_d, sd_d, n_d = g.mean(), g.std(), g.size()
        sign_q = d.assign(s=np.sign(d["d"])).groupby(["recording", "question"])["s"].agg(lambda s: abs(s.sum()) / len(s))
        out[f"diff_{m}_mean{suffix}"] = mean_d
        out[f"diff_{m}_abs{suffix}"] = mean_d.abs()
        out[f"diff_{m}_t{suffix}"] = mean_d / (sd_d / np.sqrt(n_d))
        out[f"diff_{m}_cons{suffix}"] = sign_q.groupby(level=0).mean()
    # probe rank of dwell within slide (1 = most looked at), centred at chance 2.5, signed and absolute
    rk = it.assign(r=it.groupby(SLIDE_KEYS)["dwell_share"].rank(ascending=False))
    pr = rk[rk.is_probe == 1].groupby("recording")["r"].mean() - 2.5
    out[f"probe_rank{suffix}"] = pr
    out[f"probe_rank_abs{suffix}"] = pr.abs()
    return pd.DataFrame(out)


def pupil_features(trials, items):
    t = trials.copy()
    g = t.groupby("recording")
    f = pd.DataFrame({c: g[c].mean() for c in ["pup_w1", "pup_w2", "pup_w3", "pup_min_early", "pup_peak_late",
                                              "probe_pupil_minus_irr"]})
    f["pup_w3_hab"] = t[t.repetition == 1].groupby("recording")["pup_w3"].mean() - \
        t[t.repetition == 3].groupby("recording")["pup_w3"].mean()
    f["pupil_valid"] = g["pupil_valid_frac"].mean()
    d = diff_features(items, trials)
    f["diff_pupil_on_mean"] = d["diff_pupil_on_mean"]
    f["diff_pupil_on_abs"] = d["diff_pupil_on_abs"]
    return f


def prestim_features(trials):
    t = trials.copy()
    cols = ["vid_valid", "vid_disp", "vid_sdx", "vid_sdy", "vid_fixrate", "vid_fixdur", "vid_nsacc", "vid_blinkrate",
            "vid_pupil", "vid_pupil_slope", "cross_disp", "cross_valid", "cross_nblink", "cross_fixdur", "pre_dilation"]
    g = t.groupby("recording")
    f = pd.DataFrame({c: g[c].mean() for c in cols})
    by_rep = t.groupby(["recording", "repetition"])[["vid_pupil", "vid_disp", "vid_fixrate"]].mean().unstack()
    # CMPB-style changes across the three presentation blocks (Intervals 1-3)
    for c in ["vid_pupil", "vid_disp", "vid_fixrate"]:
        f[f"{c}_d13"] = by_rep[(c, 3)] / by_rep[(c, 1)] - 1
        f[f"{c}_d12"] = by_rep[(c, 2)] / by_rep[(c, 1)] - 1
    return f


def orient_features(trials, items):
    t = trials.copy()
    g = t.groupby("recording")
    f = pd.DataFrame({"first_is_probe": g["first_is_probe"].mean(), "probe_latency": g["probe_latency"].mean(),
                      "probe_latency_med": g["probe_latency"].median(),
                      "first_item_latency": g["first_item_latency"].mean()})
    f["first_visit_diff"] = (t["probe_first_visit"] - t["irr_first_visit"]).groupby(t["recording"]).mean()
    # first look on the probe relative to the participant's own quadrant bias (expected under no probe effect)
    it = items.merge(_quadrants(items), left_on=["question", "repetition", "item"], right_index=True, how="left")
    pq = it.groupby(["recording", "quad"])["is_first"].mean()
    exp = it[it.is_probe == 1].apply(lambda r: pq.get((r["recording"], r["quad"]), 0.25), axis=1)
    it_p = it[it.is_probe == 1].assign(exp=exp.to_numpy())
    f["first_probe_excess"] = (it_p["is_first"] - it_p["exp"]).groupby(it_p["recording"]).mean()
    f["first_probe_excess_abs"] = f["first_probe_excess"].abs()
    return f


def global_features(trials):
    g = trials.groupby("recording")
    return pd.DataFrame({c: g[c].mean() for c in ["sl_nfix", "sl_fixdur", "sl_nsacc", "sl_nblink", "sl_scanpath",
                                                 "sl_aoi_dwell", "valid_frac"]})


# ============================================================ custom transformers

class ColumnGroup(BaseEstimator, TransformerMixin):
    """Keep only the columns of one group (e.g. the features computed at one valid-fraction threshold)."""

    def __init__(self, groups=None, key=None):
        self.groups = groups
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.groups[self.key]]


class OrderResidualizer(BaseEstimator, TransformerMixin):
    """Last column is the test-order indicator. Subtract each feature's order-group mean estimated on the
    training participants (equivalent to OLS residuals on the binary order covariate); drop the order column."""

    def fit(self, X, y=None):
        o = X[:, -1] > 0.5
        F = X[:, :-1]
        self.means_ = np.vstack([np.nanmean(F[~o], axis=0), np.nanmean(F[o], axis=0)])
        self.means_ = np.where(np.isnan(self.means_), np.nanmean(F, axis=0), self.means_)
        return self

    def transform(self, X):
        o = (X[:, -1] > 0.5).astype(int)
        return X[:, :-1] - self.means_[o]


class MiniRocketRidge:
    """MiniRocket (unsupervised kernels fitted on training slides) + ridge with alpha chosen in
    participant-grouped inner folds on the transformed training set."""

    def __init__(self, n_channels, n_timepoints, n_kernels=2016, seed=0):
        self.shape = (n_channels, n_timepoints)
        self.n_kernels, self.seed = n_kernels, seed

    def fit(self, X, y, g):
        from aeon.transformations.collection.convolution_based import MiniRocket
        self.mr = MiniRocket(n_kernels=self.n_kernels, random_state=self.seed, n_jobs=1)
        Z = self.mr.fit_transform(X.reshape(len(X), *self.shape))
        self.search = GridSearchCV(Pipeline([("scale", StandardScaler()), ("clf", RidgeClassifier())]),
                                   {"clf__alpha": [10.0, 100.0, 1000.0, 10000.0]},
                                   cv=grouped_inner(INNER_SPLITS), scoring="roc_auc", n_jobs=1).fit(Z, y, groups=g)
        return self

    def decision_function(self, X):
        return self.search.best_estimator_.decision_function(self.mr.transform(X.reshape(len(X), *self.shape)))


def _clip_k(grid_k, n):
    ks = sorted({k for k in grid_k if isinstance(k, int) and k < n})
    return ks + ["all"]


def lr_search_factory(n_features, residualize=False, group_cols=None):
    steps = []
    if residualize:
        steps.append(("resid", OrderResidualizer()))
    steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
              ("select", SelectKBest(f_classif)),
              ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    grid = {"select__k": _clip_k([3, 10], n_features), "clf__C": LR_C}
    if group_cols is not None:
        steps.insert(0, ("group", ColumnGroup(groups=group_cols)))
        grid["group__key"] = list(group_cols)
        grid["select__k"] = _clip_k([3, 10], min(len(v) for v in group_cols.values()))

    def factory():
        return make_search(Pipeline(steps), grid, grouped_inner(INNER_SPLITS))
    return factory


def svm_pipe():
    return Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif)), ("clf", SVC(kernel="rbf"))])


# ============================================================ extra metrics and tests

def stratified_auc(parts, order):
    """AUC computed within each test-order stratum, pooled by the number of guilty x innocent pairs."""
    vals = []
    for r, p in parts.groupby("repeat"):
        o = p["unit"].map(order)
        num, den = 0.0, 0
        for s in o.unique():
            q = p[o == s]
            npos, nneg = int(q.y.sum()), int((1 - q.y).sum())
            if npos and nneg:
                num += roc_auc_score(q.y, q.score) * npos * nneg
                den += npos * nneg
        vals.append(num / den)
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}


def avg_scores(trials):
    p = aggregate(trials)
    return p.groupby("unit").agg(y=("y", "first"), score=("score", "mean"))


def _fast_auc(y, score):
    """Mann-Whitney AUC; equals roc_auc_score and is far cheaper inside a bootstrap loop."""
    r = rankdata(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """Two-sided participant-bootstrap test of AUC(a) - AUC(b) on repeat-averaged participant scores."""
    a, b = a.sort_index(), b.loc[a.sort_index().index]
    y = a.y.to_numpy(int)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    obs = roc_auc_score(y, a.score) - roc_auc_score(y, b.score)
    sa, sb = a.score.to_numpy(), b.score.to_numpy()
    diffs = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(_fast_auc(y[i], sa[i]) - _fast_auc(y[i], sb[i]))
    diffs = np.asarray(diffs)
    p = float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))
    return {"auc_diff": float(obs), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_two_sided": p}


def participant_permutation(df, features, folds, fit, observed_auc, n_perm, jobs, strata=None, seed=0):
    """Same test as edd.harness.permutation_test (labels shuffled between participants, full nested pipeline
    refitted), with an explicit job limit. With `strata`, labels are shuffled only within strata, which keeps
    the guilt-order association in the null distribution."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        d = df.copy()
        groups = [np.arange(len(d))] if strata is None else [np.flatnonzero(d[strata] == v) for v in d[strata].unique()]
        lab = d[LABEL].to_numpy().copy()
        for idx in groups:
            lab[idx] = rng.permutation(lab[idx])
        d[LABEL] = lab
        p = aggregate(run_outer(d, features, LABEL, UNIT, [folds], fit, n_jobs=jobs))
        null.append(binary_metrics(p.y, p.score)["auc"])
    null = np.asarray(null)
    return {"n_perm": n_perm, "observed_auc": observed_auc, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "strata": strata,
            "p_value": float((1 + np.sum(null >= observed_auc)) / (1 + n_perm)),
            "note": "repeat-0 folds; observed = repeat-0 participant AUC"}


def holm(pvals):
    names = sorted(pvals, key=pvals.get)
    m, out, running = len(names), {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[n]))
        out[n] = running
    return out


# ============================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max-folds", type=int, default=0, help="smoke test: truncate each repeat")
    ap.add_argument("--skip-existing", action="store_true", help="reuse trials_<name>.csv already in out_dir")
    a = ap.parse_args()
    der, out = Path(a.derived_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.perf_counter()

    slides = pd.read_csv(der / "celniak_slide_features.csv")
    trials = pd.read_csv(der / "celniak_p2_trials.csv")
    items = pd.read_csv(der / "celniak_p2_items.csv")
    ser = np.load(der / "celniak_p2_series.npz")
    per_p = slides.groupby(UNIT)[LABEL].first()
    first_exam = slides.groupby(UNIT)["first_exam"].first()
    order = (first_exam == "Eyetracker").astype(int)
    order.index = order.index.astype(str)

    # ---- frozen folds
    folds = [order_matched_folds(per_p, order.loc[per_p.index], 2026 + r) for r in range(a.repeats)]
    if a.max_folds:  # smoke tests only
        folds = [f[:a.max_folds] for f in folds]
    save_folds(out / "folds.json", folds, {
        "design": "1 guilty + 1 innocent participant per test fold, matched on test order (eye tracker / polygraph "
                  "first) where possible, remaining pairs across orders, leftover participant alone",
        "seeds": [2026 + r for r in range(a.repeats)], "n_participants": int(len(per_p)),
        "order_counts": {"guilty_ET_first": int(((per_p == 1) & (order == 1)).sum()),
                         "guilty_PG_first": int(((per_p == 1) & (order == 0)).sum()),
                         "innocent_ET_first": int(((per_p == 0) & (order == 1)).sum()),
                         "innocent_PG_first": int(((per_p == 0) & (order == 0)).sum())}})

    # ---- participant-level tables
    base = pd.DataFrame({LABEL: per_p, "order_et": order.loc[per_p.index].to_numpy(),
                         "val_acc": trials.groupby(UNIT)["val_acc"].first().loc[per_p.index]})
    tables = {
        "paper_mean": slides.groupby(UNIT)[PAPER_FEATURES].mean(),
        "global_gaze": global_features(trials),
        "pupil": pupil_features(trials, items),
        "prestim": prestim_features(trials),
        "orient": orient_features(trials, items),
        "diff": diff_features(items, trials),
    }
    gg = tables["global_gaze"]
    tables["global_gaze_no_fixdur"] = gg.drop(columns=["sl_fixdur"])
    tables["fixdur"] = pd.concat([gg[["sl_fixdur"]], prestim_features(trials)[["vid_fixdur", "cross_fixdur"]]], axis=1)
    tables["combined"] = pd.concat([tables[k] for k in ("pupil", "prestim", "orient", "diff")], axis=1)
    q_tabs = [diff_features(items, trials, thr, suffix=f"__q{thr}") for thr in QUALITY_THRESHOLDS]
    tables["diff_quality"] = pd.concat(q_tabs, axis=1)
    q_groups_cols = {thr: list(t.columns) for thr, t in zip(QUALITY_THRESHOLDS, q_tabs)}
    P = base.copy()
    for name, t in tables.items():
        t = t.loc[:, ~t.columns.duplicated()]
        tables[name] = t
        P = P.join(t[[c for c in t.columns if c not in P.columns]])
    P = P.reset_index().rename(columns={"index": UNIT})
    P.to_csv(Path(ROOT / "data" / "derived" / "celniak_p2_participant_features.csv"), index=False)

    results, trials_out = {"meta": {"n_participants": int(len(per_p)), "repeats": a.repeats}}, {}
    # checkpoint reuse: expensive stored stages (order diagnostic, permutations, reproduction) are not recomputed
    prev = json.loads((out / "results.json").read_text()) if a.skip_existing and (out / "results.json").exists() else {}

    def record(name, tr, kind, note=""):
        tr.to_csv(out / f"trials_{name}.csv", index=False)
        trials_out[name] = tr
        ev = evaluate(tr, n_boot=300)  # screening CI; recomputed with 2000 draws for baseline and best below
        ev["ci_n_boot"] = 300
        parts = aggregate(tr)
        ev["order_stratified_auc"] = stratified_auc(parts, order)
        ev["kind"], ev["note"] = kind, note
        results[name] = ev
        p = ev["participant"]
        print(f"{name:28s} AUC {p['auc']['mean']:.3f}±{p['auc']['sd']:.3f} acc {p['accuracy']['mean']:.3f} "
              f"EER {p['eer']['mean']:.3f} HTER {p['hter']['mean']:.3f} stratAUC {ev['order_stratified_auc']['mean']:.3f}"
              f"  [{time.perf_counter() - T0:.0f}s]", flush=True)

    def save():
        (out / "results.json").write_text(json.dumps(results, indent=1, default=str))

    def cached(name):
        f = out / f"trials_{name}.csv"
        return pd.read_csv(f) if a.skip_existing and f.exists() else None

    # ---- 0. order-only reference (no learning: score = +1 if eye tracker first)
    tr = pd.DataFrame([{"repeat": r, "fold": f, "unit": u, "y": int(per_p[u]), "score": order[u] - 0.5}
                       for r, fl in enumerate(folds) for f, us in enumerate(fl) for u in us])
    record("order_only", tr, "reference", "score = test-order indicator; the confound alone")
    # Tobii's post-calibration validation accuracy (degrees), recorded before the CIT starts: not a CIT signal
    tr = run_outer(P, ["val_acc"], LABEL, UNIT, folds, nested_fit(lr_search_factory(1)), n_jobs=a.jobs)
    record("validation_accuracy_lr", tr, "reference", "pre-test calibration-validation error only (metadata)")

    # ---- 1. honest baseline: the paper's slide-level SVM with in-fold ANOVA selection, nested
    tr = cached("baseline_svm_paper")
    if tr is None:
        fit = nested_fit(lambda: make_search(svm_pipe(), SVM_GRID, grouped_inner(INNER_SPLITS)))
        tr = run_outer(slides, PAPER_FEATURES, LABEL, UNIT, folds, fit, n_jobs=a.jobs)
    record("baseline_svm_paper", tr, "baseline", "paper features, slide-level RBF-SVM, mean of 21 slide scores")

    # ---- 2. participant-level ideas (L2 logistic regression, k and C tuned in grouped inner folds)
    ideas = {
        "paper_mean_lr": ("paper_mean", False, False, "paper features averaged per participant (model-effect control)"),
        "global_gaze_lr": ("global_gaze", False, False, "non-probe-specific looking totals (confound probe)"),
        "pupil_lr": ("pupil", False, False, "idea 2: cleaned pupil, cross baseline, event-locked windows, per-item pupil"),
        "prestim_lr": ("prestim", False, False, "idea 3: question-video and cross features (CMPB 2026 idea)"),
        "orient_lr": ("orient", False, False, "idea 4: first look, probe latency, first-visit duration"),
        "diff_lr": ("diff", False, False, "idea 5: within-person probe-minus-irrelevant z, |.|, t, consistency"),
        "diff_quality_lr": ("diff_quality", False, False, "idea 6: idea 5 with valid_frac threshold tuned in-fold"),
        "combined_lr": ("combined", False, False, "ideas 2-5 pooled"),
        "diff_order_cov_lr": ("diff", True, False, "idea 1: idea 5 + order covariate"),
        "combined_order_cov_lr": ("combined", True, False, "idea 1: combined + order covariate"),
        "diff_order_resid_lr": ("diff", False, True, "idea 1: idea 5 residualised on order in-fold"),
        "combined_order_resid_lr": ("combined", False, True, "idea 1: combined residualised on order in-fold"),
        "global_gaze_order_resid_lr": ("global_gaze", False, True, "idea 1: looking totals residualised on order"),
        "global_gaze_no_fixdur_lr": ("global_gaze_no_fixdur", False, False, "ablation: looking totals without slide fixation duration"),
        "fixdur_only_lr": ("fixdur", False, False, "ablation: mean fixation duration on slide, video and cross only"),
    }
    for name, (tab, add_order, resid, note) in ideas.items():
        tr = cached(name)
        if tr is None:
            feats = list(tables[tab].columns)
            groups = None
            if tab == "diff_quality":
                groups = {thr: [feats.index(c) for c in cols] for thr, cols in q_groups_cols.items()}
            if add_order or resid:
                feats = feats + ["order_et"]
            fac = lr_search_factory(len(feats) - int(resid), residualize=resid, group_cols=groups)
            tr = run_outer(P, feats, LABEL, UNIT, folds, nested_fit(fac), n_jobs=a.jobs)
        record(name, tr, "ablation" if name in ABLATIONS else "idea", note)

    # ---- 3. MiniRocket on per-slide pupil + gaze series (idea 7)
    tr = cached("minirocket_ridge")
    if tr is None:
        Xs = ser["X"]
        sdf = pd.DataFrame(Xs.reshape(len(Xs), -1), columns=[f"s{i}" for i in range(Xs[0].size)])
        sdf[UNIT] = ser["recording"]
        sdf[LABEL] = sdf[UNIT].map(per_p).astype(int)
        nc, nt = Xs.shape[1], Xs.shape[2]
        tr = run_outer(sdf, [c for c in sdf.columns if c.startswith("s")], LABEL, UNIT, folds,
                       lambda X, y, g: MiniRocketRidge(nc, nt).fit(X, y, g), n_jobs=a.jobs)
    record("minirocket_ridge", tr, "idea", "idea 7: MiniRocket (2016 kernels) on 30 Hz pupil/gaze/AOI-hit series")
    print(f"screening done [{time.perf_counter() - T0:.0f}s]", flush=True)
    save()

    # ---- 4. order diagnostics: can each feature set predict test order instead of guilt?
    diag = dict(prev.get("order_prediction_diagnostic", {}))
    Po = P.copy()
    Po["order_label"] = Po["order_et"]
    order_folds = [order_matched_folds(Po.set_index(UNIT)["order_label"], per_p.astype(int), 3026 + r) for r in range(1)]
    if a.max_folds:
        order_folds = [f[:a.max_folds] for f in order_folds]
    for tab in ("paper_mean", "global_gaze", "pupil", "prestim", "orient", "diff"):
        if tab in diag:
            continue
        feats = list(tables[tab].columns)
        tr = run_outer(Po, feats, "order_label", UNIT, order_folds, nested_fit(lr_search_factory(len(feats))), n_jobs=a.jobs)
        p = [binary_metrics(q.y, q.score)["auc"] for _, q in aggregate(tr).groupby("repeat")]
        diag[tab] = {"order_auc_mean": float(np.mean(p)), "order_auc_sd": float(np.std(p))}
        print(f"order-prediction {tab:12s} AUC {np.mean(p):.3f}±{np.std(p):.3f}", flush=True)
    results["order_prediction_diagnostic"] = diag
    # univariate: participant-level AUC of each feature for guilt and for order, full sample (descriptive only)
    uni = {}
    for c in [c for c in P.columns if c not in (UNIT, LABEL, "order_et")]:
        v = P[c].astype(float)
        ok = v.notna()
        if v[ok].nunique() < 2:
            continue
        both = ok & P["val_acc"].notna()
        uni[c] = {"auc_guilt": float(roc_auc_score(P.loc[ok, LABEL], v[ok])),
                  "spearman_with_val_acc": float(v[both].rank().corr(P.loc[both, "val_acc"].rank())),
                  "auc_order": float(roc_auc_score(P.loc[ok, "order_et"], v[ok])),
                  "auc_guilt_within_ET_first": float(roc_auc_score(P.loc[ok & (P.order_et == 1), LABEL], v[ok & (P.order_et == 1)])),
                  "auc_guilt_within_PG_first": float(roc_auc_score(P.loc[ok & (P.order_et == 0), LABEL], v[ok & (P.order_et == 0)]))}
    results["univariate_descriptive"] = uni
    save()

    # ---- 5. paired comparisons vs the honest baseline, Holm-corrected across ideas
    base_avg = avg_scores(trials_out["baseline_svm_paper"])
    raw_p, comps = {}, {}
    for name in [n for n, v in results.items() if isinstance(v, dict) and v.get("kind") in ("idea", "ablation")]:
        comps[name] = paired_bootstrap(avg_scores(trials_out[name]), base_avg)
        raw_p[name] = comps[name]["p_two_sided"]
    adj = holm(raw_p)
    for n in comps:
        comps[n]["p_holm"] = adj[n]
        comps[n]["per_repeat_auc_diff"] = [
            float(binary_metrics(x.y, x.score)["auc"] - binary_metrics(b.y, b.score)["auc"])
            for (_, x), (_, b) in zip(aggregate(trials_out[n]).groupby("repeat"),
                                      aggregate(trials_out["baseline_svm_paper"]).groupby("repeat"))]
    results["paired_vs_baseline"] = comps

    # ---- 6. best candidate: permutation test (labels shuffled between participants, full nested pipeline)
    idea_names = [n for n in comps if results[n]["kind"] == "idea"]
    best = max(idea_names, key=lambda n: results[n]["participant"]["auc"]["mean"])
    best_probe = max(PROBE_SPECIFIC, key=lambda n: results[n]["participant"]["auc"]["mean"])
    results["best_candidate"], results["best_probe_specific"] = best, best_probe
    print("best candidate:", best, "| best probe-specific:", best_probe, flush=True)
    results["best_vs_references"] = {r: paired_bootstrap(avg_scores(trials_out[best]), avg_scores(trials_out[r]))
                                     for r in ("order_only", "validation_accuracy_lr")}
    for n in {best, best_probe, "baseline_svm_paper", "validation_accuracy_lr", "order_only"}:
        av = avg_scores(trials_out[n])
        results[n]["participant_ci95"] = bootstrap_ci(av.y.to_numpy(), av.score.to_numpy(), n_boot=2000)
        results[n]["ci_n_boot"] = 2000
    def idea_setup(name):
        tab, add_order, resid, _ = ideas[name]
        feats = list(tables[tab].columns)
        groups = None
        if tab == "diff_quality":
            groups = {thr: [feats.index(c) for c in cols] for thr, cols in q_groups_cols.items()}
        if add_order or resid:
            feats = feats + ["order_et"]
        return feats, lr_search_factory(len(feats) - int(resid), residualize=resid, group_cols=groups)

    def observed(name):
        q = aggregate(trials_out[name]).query("repeat == 0")
        return float(binary_metrics(q.y, q.score)["auc"])

    if a.n_perm:
        # within-order shuffles first: they test whether the best result is only the test-order confound
        runs = [(best, "order_et", "permutation_within_order"), (best, None, "permutation")]
        if best_probe != best:
            runs.append((best_probe, None, "permutation"))
        for name, strata, key in runs:
            if name not in ideas:
                continue
            old = prev.get(name, {}).get(key)
            if old and old.get("n_perm") == a.n_perm and abs(old["observed_auc"] - observed(name)) < 1e-9:
                results[name][key] = old
                continue
            feats, fac = idea_setup(name)
            t0 = time.perf_counter()
            results[name][key] = participant_permutation(P, feats, folds[0], nested_fit(fac), observed(name),
                                                         a.n_perm, a.jobs, strata=strata)
            results[name][key]["seconds"] = time.perf_counter() - t0
            print(name, key, results[name][key], flush=True)
            save()

    # ---- 7. reproduction under the paper's (non-nested) protocol: pick the grid point with the best
    # participant-level CV accuracy on the very folds it is scored on (selection on test folds), repeat-0 folds.
    from itertools import product
    repro = []
    repro_csv = out / "reproduction_grid_paper_protocol.csv"
    X, y = slides[PAPER_FEATURES].to_numpy(float), slides[LABEL].to_numpy(int)
    g = slides[UNIT].astype(str).to_numpy()
    for k, C, gam in ([] if a.skip_existing and repro_csv.exists() else
                      product(SVM_GRID["select__k"], SVM_GRID["clf__C"], SVM_GRID["clf__gamma"])):
        est = svm_pipe().set_params(select__k=k, clf__C=C, clf__gamma=gam)
        rows = []
        for f, test_units in enumerate(folds[0]):
            te = np.isin(g, test_units)
            m = clone(est).fit(X[~te], y[~te])
            rows.append(pd.DataFrame({"repeat": 0, "unit": g[te], "y": y[te], "score": scores_of(m, X[te])}))
        pp = aggregate(pd.concat(rows))
        repro.append({"k": k, "C": C, "gamma": gam, **binary_metrics(pp.y, pp.score)})
    repro = pd.DataFrame(repro) if repro else pd.read_csv(repro_csv)
    repro.to_csv(repro_csv, index=False)
    best_row = repro.sort_values(["accuracy", "auc"], ascending=False).iloc[0]
    results["reproduction_paper_protocol"] = {
        "note": "grid point chosen by participant accuracy on the same outer folds (not nested), repeat-0 folds; "
                "reduced Table-2 grid",
        "best_by_accuracy": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_row.items()},
        "max_auc_over_grid": float(repro["auc"].max()), "median_auc_over_grid": float(repro["auc"].median()),
        "max_accuracy_over_grid": float(repro["accuracy"].max())}
    print("reproduction (non-nested):", results["reproduction_paper_protocol"], flush=True)
    save()

    # ---- 8. efficiency: honest baseline (paper SVM) vs best candidate, single CPU thread
    import phase2_celniak_features as fx
    samples, meta = load_cached(ROOT / "data" / "celniak_cache", "Recording10")  # 38 of 39 recordings are 250 Hz
    samples = samples.reset_index(drop=True)
    sensor_hz = 1e6 / float(samples["ts_us"].diff().median())
    effr = {"hardware": eff.hardware_info(), "recording": "Recording10", "recording_samples": int(len(samples)),
            "sensor_hz": sensor_hz}
    srch = make_search(svm_pipe(), SVM_GRID, grouped_inner(INNER_SPLITS)).fit(X, y, groups=g)
    svm = srch.best_estimator_
    ksel = int(svm.named_steps["select"].get_support().sum())
    one = X[:21]
    lat = eff.latency_ms(lambda: slide_features(samples, meta), n_runs=5, warmup=1)
    effr["baseline_svm_paper"] = {
        "final_params": srch.best_params_, "model_size_bytes": eff.model_size_bytes(svm),
        "parameter_count": eff.svm_parameter_count(svm.named_steps["clf"], ksel) + 2 * len(PAPER_FEATURES),
        "latency_per_participant_ms": eff.latency_ms(lambda: svm.decision_function(one), n_runs=500),
        "latency_per_slide_ms": eff.latency_ms(lambda: svm.decision_function(X[:1]), n_runs=500),
        "throughput_slides_per_s": eff.throughput_per_s(lambda: svm.decision_function(X), len(X)),
        "peak_memory_inference_bytes": eff.peak_memory_bytes(lambda: svm.decision_function(one)),
        "feature_extraction_per_recording_ms": lat,
        "feature_extraction_samples_per_s": float(len(samples) / (lat["p50"] / 1e3)),
        "realtime_factor_vs_sensor": float(len(samples) / (lat["p50"] / 1e3) / sensor_hz),
        "feature_extraction_peak_memory_bytes": eff.peak_memory_bytes(lambda: slide_features(samples, meta))}
    if best in ideas:
        tab, add_order, resid, _ = ideas[best]
        feats = list(tables[tab].columns) + (["order_et"] if add_order or resid else [])
        groups = None
        if tab == "diff_quality":
            fl = list(tables[tab].columns)
            groups = {thr: [fl.index(c) for c in cols] for thr, cols in q_groups_cols.items()}
        fac = lr_search_factory(len(feats) - int(resid), residualize=resid, group_cols=groups)
        Xp, yp, gp = P[feats].to_numpy(float), P[LABEL].to_numpy(int), P[UNIT].astype(str).to_numpy()
        s = fac().fit(Xp, yp, groups=gp)
        mdl = s.best_estimator_
        x1 = Xp[:1]
        lat2 = eff.latency_ms(lambda: fx.process_samples(samples, meta), n_runs=5, warmup=1)
        t0 = time.perf_counter()
        pd.concat([diff_features(items, trials), global_features(trials)], axis=1)
        table_ms = (time.perf_counter() - t0) * 1e3  # upper bound: the two table builders used by the candidates
        n_kept = int(mdl.named_steps["select"].get_support().sum())
        effr["best_candidate"] = {
            "name": best, "final_params": s.best_params_, "model_size_bytes": eff.model_size_bytes(mdl),
            "parameter_count": int(2 * mdl.named_steps["scale"].n_features_in_ + mdl.named_steps["impute"].n_features_in_
                                   + n_kept + 1),
            "selected_features": [f for f, keep in zip([f for f in feats if f != "order_et" or not resid],
                                                       mdl.named_steps["select"].get_support()) if keep]
            if groups is None else "group-dependent",
            "latency_per_participant_ms": eff.latency_ms(lambda: mdl.decision_function(x1), n_runs=500),
            "throughput_participants_per_s": eff.throughput_per_s(lambda: mdl.decision_function(Xp), len(Xp)),
            "peak_memory_inference_bytes": eff.peak_memory_bytes(lambda: mdl.decision_function(x1)),
            "feature_extraction_per_recording_ms": lat2,
            "feature_extraction_samples_per_s": float(len(samples) / (lat2["p50"] / 1e3)),
            "realtime_factor_vs_sensor": float(len(samples) / (lat2["p50"] / 1e3) / sensor_hz),
            "feature_extraction_peak_memory_bytes": eff.peak_memory_bytes(lambda: fx.process_samples(samples, meta)),
            "participant_table_build_ms_per_participant": table_ms / len(per_p),
            "note": "extraction = full raw-sample pipeline (pupil cleaning, events, all trial and item features)"}
        if hasattr(mdl.named_steps["clf"], "coef_") and groups is None:
            names = [f for f in feats if not (resid and f == "order_et")]
            kept = [f for f, keep in zip(names, mdl.named_steps["select"].get_support()) if keep]
            effr["best_candidate"]["coefficients"] = dict(zip(kept, mdl.named_steps["clf"].coef_[0].round(4).tolist()))
    results["efficiency"] = effr
    results["total_seconds"] = time.perf_counter() - T0
    save()
    print(f"done in {results['total_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
