"""Phase 1 on Celniak et al. (2026): nested participant-disjoint re-baseline, leakage audit,
permutation test and deployment-cost measurements.

Usage: python -I scripts/phase1_celniak.py <slide_features.csv> <cache_dir> <out_dir> [--repeats 5] [--n-perm 100] [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from edd import efficiency as eff  # noqa: E402
from edd.celniak import PAPER_FEATURES, RAW_TARGET_FEATURES, load_cached, slide_features  # noqa: E402
from edd.folds import make_repeated_folds, save_folds  # noqa: E402
from edd.harness import (evaluate, fixed_fit, grouped_inner, make_search, nested_fit,  # noqa: E402
                         permutation_test, random_trial_split, run_outer)

# Recordings 25 and 27 are both exported as "Participant25"; the paper reports 39 participants and the
# dataset has 39 recordings (20 guilty / 19 innocent), so the recording is the participant unit.
LABEL, UNIT = "label", "recording"

# Hyperparameter spaces from Celniak et al. Table 2; k for SelectKBest is not reported, so it is searched.
K_GRID = [3, 5, 7, 10, 13, "all"]
SVM_GRID = {"select__k": K_GRID, "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "clf__gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
RF_GRID = {"select__k": K_GRID, "clf__max_depth": [3, 5, 7, None], "clf__min_samples_split": [2, 5, 10],
           "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", "log2", None],
           "clf__n_estimators": [100, 200, 300]}
XGB_GRID = {"select__k": K_GRID, "clf__max_depth": [3, 4, 5, 6], "clf__min_child_weight": [1, 3, 5],
            "clf__n_estimators": [50, 100, 150], "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__subsample": [0.7, 0.8, 1.0], "clf__colsample_bytree": [0.7, 0.8, 1.0],
            "clf__reg_alpha": [0, 0.1, 0.5, 1], "clf__reg_lambda": [1, 2, 5], "clf__gamma": [0, 0.1, 0.2]}
QUICK_SVM_GRID = {"select__k": [7, "all"], "clf__C": [0.1, 1, 10], "clf__gamma": [0.01, 0.1]}


def svm_pipe():
    return Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif)), ("clf", SVC(kernel="rbf"))])


def rf_pipe():
    return Pipeline([("select", SelectKBest(f_classif)), ("clf", RandomForestClassifier(random_state=0, n_jobs=1))])


def xgb_pipe():
    return Pipeline([("select", SelectKBest(f_classif)),
                     ("clf", XGBClassifier(n_jobs=1, tree_method="hist", eval_metric="logloss", random_state=0))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_csv")
    ap.add_argument("cache_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-iter", type=int, default=30, help="random-search budget for RF/XGB")
    ap.add_argument("--quick", action="store_true", help="tiny grids for a smoke test")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.features_csv)
    per_p = df.groupby(UNIT)[LABEL].first()
    folds = make_repeated_folds(per_p.index.to_numpy(), per_p.to_numpy(), a.repeats)
    save_folds(out / "folds.json", folds, {"design": "1 guilty + 1 innocent participant per test fold",
                                           "n_participants": int(len(per_p)), "n_slides": int(len(df))})
    svm_grid = QUICK_SVM_GRID if a.quick else SVM_GRID
    n_iter = 3 if a.quick else a.n_iter
    inner = grouped_inner(5)
    results, timings = {}, {}

    def done(name):
        """Resume support: reuse out-of-fold scores already on disk (folds are seeded, so they match)."""
        f = out / f"trials_{name}.csv"
        if f.exists():
            print(f"{name}: loaded from {f.name}", flush=True)
            record(name, pd.read_csv(f))
            return True
        return False

    def record(name, trials):
        trials.to_csv(out / f"trials_{name}.csv", index=False)
        results[name] = evaluate(trials)
        p = results[name]["participant"]
        print(f"{name:28s} participant AUC {p['auc']['mean']:.3f}±{p['auc']['sd']:.3f}  "
              f"acc {p['accuracy']['mean']:.3f}  trial AUC {results[name]['trial']['auc']['mean']:.3f}", flush=True)

    # 1. Honest re-baselines: tuning and feature selection inside participant-disjoint folds.
    for name, feats, pipe, grid, it in (("svm_nested", PAPER_FEATURES, svm_pipe, svm_grid, None),
                                        ("svm_nested_raw_target", RAW_TARGET_FEATURES, svm_pipe, svm_grid, None),
                                        ("rf_nested", PAPER_FEATURES, rf_pipe, RF_GRID, n_iter),
                                        ("xgb_nested", PAPER_FEATURES, xgb_pipe, XGB_GRID, n_iter)):
        if done(name):
            continue
        t0 = time.perf_counter()
        fit = nested_fit(lambda p=pipe, g=grid, i=it: make_search(p(), g, inner, n_iter=i))
        record(name, run_outer(df, feats, LABEL, UNIT, folds, fit))
        timings[name] = time.perf_counter() - t0

    # 2. Leakage audit: the same SVM pipeline under three common shortcuts.
    if not done("leak_random_trial_split"):
        record("leak_random_trial_split", random_trial_split(
            df, PAPER_FEATURES, LABEL, UNIT, lambda: make_search(svm_pipe(), svm_grid, StratifiedKFold(5, shuffle=True, random_state=0))))

    tuned_all = make_search(svm_pipe(), svm_grid, inner, n_jobs=-1)
    tuned_all.fit(df[PAPER_FEATURES].to_numpy(float), df[LABEL].to_numpy(int), groups=df[UNIT].astype(str).to_numpy())
    if not done("leak_tuned_on_all_data"):
        record("leak_tuned_on_all_data", run_outer(df, PAPER_FEATURES, LABEL, UNIT, folds, fixed_fit(tuned_all.best_estimator_)))

    selector = SelectKBest(f_classif, k=7).fit(df[PAPER_FEATURES].to_numpy(float), df[LABEL].to_numpy(int))
    chosen = [f for f, keep in zip(PAPER_FEATURES, selector.get_support()) if keep]
    no_select_grid = {k: v for k, v in svm_grid.items() if not k.startswith("select")}
    fit = nested_fit(lambda: make_search(Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf"))]),
                                         no_select_grid, inner))
    if not done("leak_selection_on_all_data"):
        record("leak_selection_on_all_data", run_outer(df, chosen, LABEL, UNIT, folds, fit))
    results["leak_selection_on_all_data"]["selected_features"] = chosen

    # 3. Permutation test for the honest SVM.
    if a.n_perm:
        t0 = time.perf_counter()
        fit = nested_fit(lambda: make_search(svm_pipe(), svm_grid, inner))
        obs = results["svm_nested"]["participant"]["auc"]["mean"]
        results["svm_nested"]["permutation"] = permutation_test(df, PAPER_FEATURES, LABEL, UNIT, folds[0], fit, obs, a.n_perm)
        timings["permutation"] = time.perf_counter() - t0
        print("permutation", results["svm_nested"]["permutation"], flush=True)

    # 4. Deployment cost of the final SVM (fit on all data) and of feature extraction from raw samples.
    final = tuned_all.best_estimator_
    X = df[PAPER_FEATURES].to_numpy(float)
    one_participant = X[:21]
    k = int(final.named_steps["select"].get_support().sum())
    samples, meta = load_cached(a.cache_dir, "Recording1")
    feat_lat = eff.latency_ms(lambda: slide_features(samples, meta), n_runs=20, warmup=2)
    results["efficiency"] = {
        "hardware": eff.hardware_info(),
        "final_params": tuned_all.best_params_,
        "selected_features": int(k),
        "model_size_bytes": eff.model_size_bytes(final),
        "parameter_count": eff.svm_parameter_count(final.named_steps["clf"], k),
        "inference_per_participant_21_slides_ms": eff.latency_ms(lambda: final.decision_function(one_participant)),
        "inference_per_slide_ms": eff.latency_ms(lambda: final.decision_function(X[:1])),
        "inference_throughput_slides_per_s": eff.throughput_per_s(lambda: final.decision_function(X), len(X)),
        "inference_peak_memory_bytes": eff.peak_memory_bytes(lambda: final.decision_function(one_participant)),
        "feature_extraction_per_recording_ms": feat_lat,
        "feature_extraction_samples_per_s": float(len(samples) / (feat_lat["p50"] / 1e3)),
        "realtime_factor_vs_120Hz": float(len(samples) / (feat_lat["p50"] / 1e3) / 120.0),
        "feature_extraction_peak_memory_bytes": eff.peak_memory_bytes(lambda: slide_features(samples, meta)),
    }
    results["timings_s"] = timings
    (out / "summary.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results["efficiency"], indent=1, default=str))


if __name__ == "__main__":
    main()
