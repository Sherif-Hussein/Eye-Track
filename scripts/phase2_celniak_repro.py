"""Phase 2 (celniak track), takeover stage 4: the published (non-nested) protocol, to show how 0.95 / 1.00 can arise.

Paper protocol as far as the text fixes it: RBF-SVM on the 19 normalised slide features with ANOVA-F
SelectKBest, Table-2 grid (k is not reported, searched over {3,5,7,10,13,all}; C and gamma 6 values each = 216
points), GroupKFold with 2 participants per test fold (n_splits = 20), slide predictions aggregated to the
participant (rule unspecified; both mean slide score and majority vote are reported). The "tuned on all data"
step means that the grid point is the one with the best score on the very folds it is scored on.

Runs (all 216 grid points each; results in reproduction_*.csv, one per run, skipped when present):
  A_groupkfold         selection inside folds, sklearn GroupKFold(20)                      (the faithful reproduction)
  B_groupkfold_leaky   ANOVA selection fitted on ALL data before the CV (leaky), GroupKFold(20)
  A_pairing_s<seed>    A with random participant pairings (5 seeds, reduced 36-point grid): spread of "best of grid"
  Null_A_s<seed>       A with participant labels shuffled (5 seeds, reduced 36-point grid): best-of-grid optimism
                       when there is no signal at all

Usage: python -I scripts/phase2_celniak_repro.py [--jobs 2]
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from edd.celniak import PAPER_FEATURES  # noqa: E402
from edd.metrics import binary_metrics  # noqa: E402

OUT = ROOT / "results" / "phase2" / "celniak"
K_GRID = [3, 5, 7, 10, 13, "all"]
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
G_GRID = [0.0001, 0.001, 0.01, 0.1, 1, 10]
GRID = list(product(K_GRID, C_GRID, G_GRID))
# reduced grid (same as the screening baseline) for the repeated pairings and the label-shuffled nulls: cost
REDUCED = list(product([5, 7, "all"], [0.1, 1, 10, 100], [0.001, 0.01, 0.1]))


def folds_groupkfold(g):
    return [te for _, te in GroupKFold(n_splits=20).split(np.zeros(len(g)), groups=g)]


def folds_random_pairs(units, seed):
    rng = np.random.default_rng(seed)
    u = rng.permutation(units)
    return [list(u[i:i + 2]) for i in range(0, len(u), 2)]


def one_point(k, C, gam, X, y, g, fold_units, leaky_X):
    rows = []
    for test_units in fold_units:
        te = np.isin(g, test_units)
        pipe = Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif, k=k)), ("clf", SVC(C=C, gamma=gam))]) \
            if leaky_X is None else Pipeline([("scale", StandardScaler()), ("clf", SVC(C=C, gamma=gam))])
        Xf = X if leaky_X is None else leaky_X[k]
        pipe.fit(Xf[~te], y[~te])
        rows.append(pd.DataFrame({"unit": g[te], "y": y[te], "score": pipe.decision_function(Xf[te])}))
    s = pd.concat(rows)
    out = {"k": k, "C": C, "gamma": gam}
    mean_agg = s.groupby("unit").agg(y=("y", "first"), score=("score", "mean"))
    vote_agg = s.assign(v=(s.score >= 0).astype(float)).groupby("unit").agg(y=("y", "first"), score=("v", "mean"))
    for tag, a, thr in (("mean", mean_agg, 0.0), ("vote", vote_agg, 0.5)):
        m = binary_metrics(a.y, a.score - thr)
        out.update({f"{tag}_{n}": m[n] for n in ("accuracy", "sensitivity", "specificity", "f1", "auc", "eer", "hter")})
    out["slide_accuracy"] = float(((s.score >= 0).astype(int) == s.y).mean())
    return out


def run(name, y_by_unit, fold_units, X, y, g, leaky, jobs, grid=None):
    grid = GRID if grid is None else grid
    f = OUT / f"reproduction_{name}.csv"
    if f.exists():
        return pd.read_csv(f)
    t0 = time.perf_counter()
    leaky_X = None
    if leaky:
        leaky_X = {}
        Xs = StandardScaler().fit_transform(X)
        for k in K_GRID:
            sel = SelectKBest(f_classif, k=k).fit(Xs, y)  # fitted on ALL participants including the test folds
            leaky_X[k] = X[:, sel.get_support()]
    res = Parallel(n_jobs=jobs)(delayed(one_point)(k, C, gm, X, y, g, fold_units, leaky_X) for k, C, gm in grid)
    df = pd.DataFrame(res)
    df.to_csv(f, index=False)
    print(f"{name}: best acc {df.mean_accuracy.max():.3f}, best AUC {df.mean_auc.max():.3f} [{time.perf_counter() - t0:.0f}s]", flush=True)
    return df


def summarise(df):
    d = {}
    for tag in ("mean", "vote"):
        b = df.sort_values([f"{tag}_accuracy", f"{tag}_auc"], ascending=False).iloc[0]
        d[tag] = {
            "best_by_accuracy": {k: (v.item() if hasattr(v, "item") else v) for k, v in b.items()
                                 if k in ("k", "C", "gamma") or k.startswith(tag) or k == "slide_accuracy"},
            "max_accuracy": float(df[f"{tag}_accuracy"].max()), "max_auc": float(df[f"{tag}_auc"].max()),
            "median_accuracy": float(df[f"{tag}_accuracy"].median()), "median_auc": float(df[f"{tag}_auc"].median()),
            "share_grid_points_acc_ge_0.9": float((df[f"{tag}_accuracy"] >= 0.9).mean()),
            "n_grid_points": int(len(df))}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    slides = pd.read_csv(ROOT / "data" / "derived" / "celniak_slide_features.csv")
    X, y = slides[PAPER_FEATURES].to_numpy(float), slides["label"].to_numpy(int)
    g = slides["recording"].astype(str).to_numpy()
    units = np.array(sorted(set(g), key=lambda s: int(s.replace("Recording", ""))))
    ylab = slides.groupby(slides["recording"].astype(str))["label"].first()
    summary = {"protocol": __doc__, "n_slides": int(len(slides)), "n_participants": int(len(units)),
               "grid": {"k": K_GRID, "C": C_GRID, "gamma": G_GRID, "n_points": len(GRID)}}
    gk = folds_groupkfold(g)
    gk_units = [sorted(set(g[i])) for i in gk]
    summary["groupkfold_fold_sizes_participants"] = sorted({len(u) for u in gk_units})
    summary["groupkfold_n_folds"] = len(gk_units)
    summary["A_groupkfold"] = summarise(run("A_groupkfold", ylab, gk_units, X, y, g, False, a.jobs))
    (OUT / "stage_reproduction.json").write_text(json.dumps(summary, indent=1, default=str))
    summary["B_groupkfold_leaky"] = summarise(run("B_groupkfold_leaky", ylab, gk_units, X, y, g, True, a.jobs))
    (OUT / "stage_reproduction.json").write_text(json.dumps(summary, indent=1, default=str))
    dfA = pd.read_csv(OUT / "reproduction_A_groupkfold.csv")
    keyset = {(str(k), c, gm) for k, c, gm in REDUCED}
    summary["A_groupkfold_reduced_grid"] = summarise(dfA[[(str(k), c, gm) in keyset for k, c, gm in zip(dfA.k, dfA.C, dfA.gamma)]])
    summary["A_pairings"] = {}
    for s in range(5):
        summary["A_pairings"][str(s)] = summarise(run(f"A_pairing_s{s}", ylab, folds_random_pairs(units, 500 + s), X, y, g, False, a.jobs, REDUCED))
        (OUT / "stage_reproduction.json").write_text(json.dumps(summary, indent=1, default=str))
    summary["Null_A"] = {}
    for s in range(5):
        rng = np.random.default_rng(900 + s)
        perm = pd.Series(rng.permutation(ylab.to_numpy()), index=ylab.index)
        yn = perm.loc[g].to_numpy(int)
        summary["Null_A"][str(s)] = summarise(run(f"Null_A_s{s}", perm, gk_units, X, yn, g, False, a.jobs, REDUCED))
        (OUT / "stage_reproduction.json").write_text(json.dumps(summary, indent=1, default=str))
    print("done")


if __name__ == "__main__":
    main()
