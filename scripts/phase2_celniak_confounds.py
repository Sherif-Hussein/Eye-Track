"""Phase 2 (celniak track), takeover stage 3: confound controls for global_gaze_lr.

Every adjustment is fitted on training participants only (same frozen folds, 3 repeats, same nested LR pipeline):

  a1  gg_val_cov          validation error (log of Tobii average validation accuracy, degrees) as an always-kept covariate
  a2  gg_val_resid        every feature residualised on log validation error (OLS on the training participants)
  b   gg_qual_train       training participants with valid_frac below a threshold are dropped; the threshold
                          is chosen by participant-grouped inner CV from {0, .5, .6, .7}; ALL test participants are scored
      (b-eval)            the same scores, evaluated only on test participants whose valid_frac >= the threshold chosen
                          in their own fold (changes the evaluated population)
  ab  gg_val_resid_qual_train   a2 and b together
  c1  gg_drop_rec1        Recording1 (the only 120 Hz recording) removed from training and evaluation
  c2  gg_125hz            all 250 Hz recordings decimated by 2 (about 125 Hz), features recomputed from raw samples

Descriptive sensitivity analyses (no fitting, scores already out-of-fold): AUC of the stored nested scores on
subsets defined by fixed a-priori valid_frac cut-offs.

Outputs: results/phase2/celniak/trials_conf_<name>.csv and stage_confounds.json (results.json is not touched).
Usage: python -I scripts/phase2_celniak_confounds.py [--jobs 2] [--only name,name]
"""
import argparse
import ast
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import phase2_celniak_screen as S  # noqa: E402
from edd.celniak import load_cached  # noqa: E402
from edd.celniak_confounds import CovariateResidualizer, SelectPlusCovariate, quality_filtered_fit  # noqa: E402
from edd.harness import aggregate, evaluate, grouped_inner, make_search, nested_fit, run_outer  # noqa: E402
from edd.metrics import binary_metrics  # noqa: E402

OUT = ROOT / "results" / "phase2" / "celniak"
DER = ROOT / "data" / "derived"
LABEL, UNIT = S.LABEL, S.UNIT
GG = ["sl_nfix", "sl_fixdur", "sl_nsacc", "sl_nblink", "sl_scanpath", "sl_aoi_dwell", "valid_frac"]
QUALITY_COL = GG.index("valid_frac")
THRESHOLDS = [0.0, 0.5, 0.6, 0.7]
APRIORI_CUTOFFS = [0.5, 0.6, 0.7]
LR = lambda: LogisticRegression(max_iter=2000, class_weight="balanced")  # noqa: E731


def base_pipe(resid=False):
    steps = ([("resid", CovariateResidualizer())] if resid else []) + [
        ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif)), ("clf", LR())]
    return Pipeline(steps)


GRID = {"select__k": S._clip_k([3, 10], len(GG)), "clf__C": S.LR_C}


def lr_fit():
    return nested_fit(lambda: make_search(base_pipe(), GRID, grouped_inner(S.INNER_SPLITS)))


def resid_fit():
    return nested_fit(lambda: make_search(base_pipe(True), GRID, grouped_inner(S.INNER_SPLITS)))


def cov_fit():
    pipe = Pipeline([("cov", SelectPlusCovariate()), ("clf", LR())])
    grid = {"cov__k": S._clip_k([3, 10], len(GG)), "clf__C": S.LR_C}
    return nested_fit(lambda: make_search(pipe, grid, grouped_inner(S.INNER_SPLITS)))


def qual_fit(resid):
    return quality_filtered_fit(base_pipe(resid), GRID, QUALITY_COL, THRESHOLDS, S.INNER_SPLITS)


# ------------------------------------------------------------------ helpers

def auc_ci(y, s, n_boot=2000, seed=0):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = [S._fast_auc(y[i], s[i]) for i in (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                                           for _ in range(n_boot))]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def subset_summary(tr, keep_units=None, keep_rows=None):
    """Per-repeat participant AUC (mean, sd), repeat-averaged AUC with bootstrap CI, n by class, on a subset.
    keep_units: set of participants; keep_rows: boolean mask over aggregate(tr) rows (repeat-specific)."""
    parts = aggregate(tr)
    if keep_units is not None:
        parts = parts[parts.unit.isin(keep_units)]
    if keep_rows is not None:
        parts = parts[keep_rows(parts)]
    per = [binary_metrics(p.y, p.score)["auc"] for _, p in parts.groupby("repeat") if p.y.nunique() == 2]
    avg = parts.groupby("unit").agg(y=("y", "first"), score=("score", "mean"))
    n_rep = parts.groupby("repeat").size()
    return {"auc_mean": float(np.mean(per)), "auc_sd": float(np.std(per)),
            "n_mean": float(n_rep.mean()), "n_guilty_mean": float(parts.groupby("repeat").y.sum().mean()),
            "avg_scores_auc": float(S._fast_auc(avg.y.to_numpy(int), avg.score.to_numpy())),
            "avg_scores_auc_ci95": auc_ci(avg.y, avg.score)}


def thr_by_fold(tr):
    """Threshold chosen in each outer fold (from the stored params string) -> dict (repeat, fold) -> thr."""
    d = {}
    for (r, f), g in tr.groupby(["repeat", "fold"]):
        d[(r, f)] = ast.literal_eval(g["params"].iloc[0])["thr"]
    return d


def eval_variant(tr, order):
    ev = evaluate(tr, n_boot=2000)
    ev["ci_n_boot"] = 2000
    ev["order_stratified_auc"] = S.stratified_auc(aggregate(tr), order)
    return ev


# ------------------------------------------------------------------ 125 Hz extraction (stage c2)

def _decimated_trials(name):
    import phase2_celniak_features as fx
    samples, meta = load_cached(ROOT / "data" / "celniak_cache", name)
    samples = samples.reset_index(drop=True)
    fs = 1e6 / float(samples["ts_us"].diff().median())
    if fs > 200:
        samples = samples.iloc[::2].reset_index(drop=True)
    tr = fx.process_samples(samples, meta)[0]
    tr["fs_after"] = 1e6 / float(samples["ts_us"].diff().median())
    return tr


def build_125hz_table():
    f = DER / "celniak_conf_global_125hz.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0)
    names = sorted({p.name.split(".")[0] for p in (ROOT / "data" / "celniak_cache").glob("Recording*.csv.gz")},
                   key=lambda s: int(s.replace("Recording", "")))
    with ProcessPoolExecutor(max_workers=2) as pool:
        trials = pd.concat(list(pool.map(_decimated_trials, names)), ignore_index=True)
    trials.to_csv(DER / "celniak_conf_trials_125hz.csv", index=False)
    tab = S.global_features(trials)
    tab.to_csv(f)
    return tab


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    T0 = time.perf_counter()

    P = pd.read_csv(DER / "celniak_p2_participant_features.csv")
    P[UNIT] = P[UNIT].astype(str)
    P["log_val_acc"] = np.log(P["val_acc"])
    order = P.set_index(UNIT)["order_et"]
    folds = json.loads((OUT / "folds.json").read_text())["repeats"]
    stage_f = OUT / "stage_confounds.json"
    stage = json.loads(stage_f.read_text()) if stage_f.exists() else {}
    stage.setdefault("variants", {})

    def save():
        stage_f.write_text(json.dumps(stage, indent=1, default=str))

    def load_or_run(name, fn):
        f = OUT / f"trials_conf_{name}.csv"
        if f.exists():
            return pd.read_csv(f)
        tr = fn()
        tr.to_csv(f, index=False)
        return tr

    def run(name, desc, feats, fit, df=None, fl=None):
        if only and name not in only:
            return None
        df = P if df is None else df
        tr = load_or_run(name, lambda: run_outer(df, feats, LABEL, UNIT, folds if fl is None else fl, fit, n_jobs=a.jobs))
        ev = eval_variant(tr, order)
        ev["description"] = desc
        stage["variants"][name] = ev
        p = ev["participant"]
        print(f"{name:26s} AUC {p['auc']['mean']:.3f}±{p['auc']['sd']:.3f} acc {p['accuracy']['mean']:.3f} "
              f"EER {p['eer']['mean']:.3f} HTER {p['hter']['mean']:.3f} strat {ev['order_stratified_auc']['mean']:.3f} "
              f"[{time.perf_counter() - T0:.0f}s]", flush=True)
        save()
        return tr

    ref = pd.read_csv(OUT / "trials_global_gaze_lr.csv")
    val_ref = pd.read_csv(OUT / "trials_validation_accuracy_lr.csv")
    base_ref = pd.read_csv(OUT / "trials_baseline_svm_paper.csv")

    # ---- (a) validation error
    run("gg_val_cov", "global_gaze + log validation error as always-kept covariate", GG + ["log_val_acc"], cov_fit())
    run("gg_val_resid", "global_gaze residualised on log validation error (in-fold OLS)", GG + ["log_val_acc"], resid_fit())

    # ---- (b) quality exclusion, threshold chosen in training folds
    tr_b = run("gg_qual_train", "training participants with valid_frac < thr dropped; thr chosen by inner CV from "
               f"{THRESHOLDS}; all test participants scored", GG, qual_fit(False))
    tr_ab = run("gg_val_resid_qual_train", "a2 + b together", GG + ["log_val_acc"], qual_fit(True))
    stage["quality_threshold_choices"] = {}
    vf = P.set_index(UNIT)["valid_frac"]
    for name, tr in (("gg_qual_train", tr_b), ("gg_val_resid_qual_train", tr_ab)):
        if tr is None:
            continue
        thr = thr_by_fold(tr)
        counts = pd.Series(list(thr.values())).value_counts().sort_index()
        stage["quality_threshold_choices"][name] = {str(k): int(v) for k, v in counts.items()}
        # b-eval: evaluate only test participants with valid_frac >= the threshold chosen in their fold
        tr = tr.copy()
        tr["thr"] = [thr[(r, f)] for r, f in zip(tr["repeat"], tr["fold"])]
        tr["vf"] = tr["unit"].map(vf)
        keep = tr[tr.vf >= tr.thr].drop(columns=["thr", "vf"])
        sub = subset_summary(keep)
        sub["n_test_participants_dropped_mean"] = float(39 - sub["n_mean"])
        # identical population for the unadjusted model: same (repeat, unit) rows of the stored global_gaze_lr scores
        kk = set(zip(keep["repeat"], keep["unit"]))
        ref_same = ref[[(r, u) in kk for r, u in zip(ref["repeat"], ref["unit"])]]
        stage.setdefault("b_eval", {})[name] = {
            "adjusted_on_retained_test": sub,
            "global_gaze_lr_on_same_retained_test": subset_summary(ref_same),
            "validation_accuracy_lr_on_same_retained_test": subset_summary(
                val_ref[[(r, u) in kk for r, u in zip(val_ref["repeat"], val_ref["unit"])]])}
        save()

    # ---- (c1) drop the only 120 Hz recording
    fl1 = [[[u for u in f if u != "Recording1"] for f in rep] for rep in folds]
    fl1 = [[f for f in rep if f] for rep in fl1]
    df1 = P[P[UNIT] != "Recording1"]
    run("gg_drop_rec1", "global_gaze_lr with Recording1 (120 Hz) removed from training and evaluation", GG, lr_fit(), df=df1, fl=fl1)

    # ---- (c2) uniform ~125 Hz
    if not only or "gg_125hz" in only:
        tab = build_125hz_table()
        P125 = P.copy()
        for c in GG:
            P125[c] = P125[UNIT].map(tab[c])
        feat_shift = {c: {"spearman_with_original": float(spearmanr(P[c], P125[c]).correlation),
                          "median_relative_change": float(np.nanmedian((P125[c] - P[c]) / P[c].replace(0, np.nan)))}
                      for c in GG}
        stage["feature_shift_125hz"] = feat_shift
        run("gg_125hz", "global_gaze_lr with all 250 Hz recordings decimated by 2 (~125 Hz), features recomputed",
            GG, lr_fit(), df=P125)

    # ---- descriptive: stored nested scores on fixed a-priori subsets, and by sampling rate
    sens = {}
    for label, tr in (("global_gaze_lr", ref), ("validation_accuracy_lr", val_ref), ("baseline_svm_paper", base_ref)):
        sens[label] = {"all": subset_summary(tr),
                       "without_Recording1": subset_summary(tr, keep_units=set(P[UNIT]) - {"Recording1"})}
        for c in APRIORI_CUTOFFS:
            sens[label][f"valid_frac>={c}"] = subset_summary(tr, keep_units=set(P.loc[P.valid_frac >= c, UNIT]))
    sens["n_participants_by_cutoff"] = {str(c): {"n": int((P.valid_frac >= c).sum()),
                                                 "guilty": int(P.loc[P.valid_frac >= c, LABEL].sum())}
                                        for c in APRIORI_CUTOFFS}
    stage["apriori_quality_sensitivity"] = sens
    r1 = P[P[UNIT] == "Recording1"].iloc[0]
    stage["recording1"] = {"label": int(r1[LABEL]), "order_et": int(r1["order_et"]),
                           "sl_fixdur": float(r1["sl_fixdur"]), "valid_frac": float(r1["valid_frac"]),
                           "val_acc": float(r1["val_acc"]),
                           "sl_fixdur_rank_among_39_(1=shortest)": int(P["sl_fixdur"].rank().loc[r1.name]),
                           "oof_score_mean_over_repeats": float(ref[ref.unit == "Recording1"].score.mean()),
                           "oof_score_rank_among_39_(1=lowest)": int(ref.groupby("unit").score.mean().rank().loc["Recording1"])}
    # fixation duration and sampling-rate bound: fixation duration = last-first timestamp + one sample interval
    stage["fixdur_quantisation_bound_ms"] = {"250Hz_sample_ms": 4.0, "120Hz_sample_ms": 1000 / 120,
                                             "max_bias_of_120Hz_vs_250Hz_ms": 1000 / 120 - 4.0,
                                             "as_share_of_median_fixdur": float((1000 / 120 - 4.0) / P["sl_fixdur"].median())}
    # univariate: fixation duration vs log validation error, partial AUC picture
    stage["spearman_sl_fixdur_vs_log_val_acc"] = float(spearmanr(P["sl_fixdur"], P["log_val_acc"], nan_policy="omit").correlation)
    stage["spearman_sl_fixdur_vs_valid_frac"] = float(spearmanr(P["sl_fixdur"], P["valid_frac"]).correlation)
    stage["spearman_val_acc_vs_valid_frac"] = float(spearmanr(P["log_val_acc"], P["valid_frac"], nan_policy="omit").correlation)
    stage["seconds"] = time.perf_counter() - T0
    save()
    print("done", f"{stage['seconds']:.0f}s")


if __name__ == "__main__":
    main()
