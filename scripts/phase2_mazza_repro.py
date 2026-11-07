"""Data audit and reproduction of Colasanti et al. (2023, J Nonverbal Behav 48:47-71) classification:
WEKA 3.9 Logistic (ridge 1e-8) on the 36 target-AOI count/duration features, single 10-fold CV.
WEKA reports pooled out-of-fold predictions and class-weighted averages of precision, recall, F and
ROC AUC; we mimic that and repeat over 10 CV seeds (WEKA's default seed is 1) to show seed spread.

Usage: python -I scripts/phase2_mazza_repro.py <data.csv> <out_dir>
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from edd.metrics import binary_metrics  # noqa: E402
from edd.mazza import AOIS, NEUTRALS, PUB36, TARGETS, load, ncol, raw_columns, tcol  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Section "Classification Models", p. 63 of the article.
PUBLISHED = {
    "4class": {"auc": 0.63, "accuracy": 0.4013, "precision": 0.39, "recall": 0.40, "f": 0.40,
               "correct": {1: "10/40", 2: "15/38", 3: "19/40", 4: "19/39"}},
    "g2_vs_g3": {"auc": 0.72, "accuracy": 0.6538, "precision": 0.65, "recall": 0.65, "f": 0.65,
                 "correct": {2: "26/38", 3: "25/40"}},
    "g2_vs_g1": {"auc": 0.61, "accuracy": 0.6026, "precision": 0.60, "recall": 0.60, "f": 0.60,
                 "correct": {2: "23/38", 1: "24/40"}},
    "g2_vs_g4": {"auc": 0.66, "accuracy": 0.6104, "precision": 0.61, "recall": 0.61, "f": 0.61,
                 "correct": {2: "24/38", 4: "23/39"}},
}
SUBSET = {"4class": (1, 2, 3, 4), "g2_vs_g3": (2, 3), "g2_vs_g1": (1, 2), "g2_vs_g4": (2, 4)}

# Published group means (Table 2: counts; Table 3: summed durations), order G1, G2, G3, G4.
PUB_MEANS = {
    ("cnt", 1, "left"): [9.82, 9.11, 12.63, 10.69], ("cnt", 1, "central"): [13.88, 15.37, 10.50, 12.26],
    ("cnt", 4, "left"): [7.38, 5.13, 7.97, 4.92], ("cnt", 3, "central"): [16.10, 17.34, 13.55, 17.90],
    ("dur", 1, "left"): [2634.87, 2533.95, 3433.35, 2941.74], ("dur", 1, "central"): [3306.68, 3895.71, 2450.93, 2835.82],
    ("dur", 4, "central"): [399.50, 3720.37, 2875.35, 3896.31], ("dur", 6, "right"): [1872.28, 1402.84, 1862.70, 1849.39],
}


def audit(d):
    raw = raw_columns(d)
    X = d[raw]
    med, mad = X.median(), (X - X.median()).abs().median().replace(0, np.nan)
    rz = ((X - med) / (1.4826 * mad)).abs()
    out = {"rows": int(len(d)), "columns": int(d.shape[1] - 2), "separator": "comma", "encoding": "ASCII",
           "missing_values": int(X.isna().sum().sum()), "duplicate_ids": int(d.g_code.duplicated().sum()),
           "group_sizes": d.group.value_counts().sort_index().to_dict(),
           "cells_robust_z_gt_5": int((rz > 5).sum().sum()),
           "participants_with_any_robust_z_gt_5": int((rz > 5).any(axis=1).sum()),
           "columns_with_zero_MAD": [c for c in raw if pd.isna(mad[c])]}
    for k in ("cnt", "dur"):
        diff = sum(d[tcol(k, t, a)] for t in TARGETS for a in AOIS) - sum(d[tcol(k, t, "overall")] for t in TARGETS)
        ndiff = sum(d[ncol(k, n, a)] for n in NEUTRALS for a in AOIS) - sum(d[ncol(k, n, "overall")] for n in NEUTRALS)
        out[f"{k}_target_AOI_sum_minus_overall"] = {"mean": float(diff.mean()), "min": float(diff.min()),
                                                    "max": float(diff.max())}
        out[f"{k}_neutral_AOI_sum_minus_overall_max_abs"] = float(ndiff.abs().max())
    chk = {}
    for (k, t, a), pub in PUB_MEANS.items():
        ours = d.groupby("group")[tcol(k, t, a)].mean().round(2).tolist()
        chk[f"{k}_T{t}_{a}"] = {"published": pub, "data": ours}
    out["published_mean_check"] = chk
    return out


def weka_cv(d, task, seed):
    s = d[d.group.isin(SUBSET[task])].reset_index(drop=True)
    X, y = s[PUB36].to_numpy(float), s.group.to_numpy()
    P = np.zeros((len(y), len(np.unique(y))))
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1e8, max_iter=10000)).fit(X[tr], y[tr])
        P[te] = m.predict_proba(X[te])
    classes = np.unique(y)
    pred = classes[P.argmax(1)]
    pr, rc, f, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    if len(classes) == 2:
        auc = roc_auc_score(y == classes[1], P[:, 1])
        pos = 2
        bm = binary_metrics((y == pos).astype(int), P[:, list(classes).index(pos)] - 0.5)
    else:
        auc = roc_auc_score(y, P, multi_class="ovr", average="weighted")
        bm = None
    res = {"auc": auc, "accuracy": accuracy_score(y, pred), "precision": pr, "recall": rc, "f": f,
           "correct": {int(c): f"{int(((pred == c) & (y == c)).sum())}/{int((y == c).sum())}" for c in classes}}
    if bm:
        res["eer"], res["hter"], res["mcc"] = bm["eer"], bm["hter"], bm["mcc"]
    return res


def main():
    d = load(sys.argv[1])
    out = Path(sys.argv[2])
    res = {"audit": audit(d), "published": PUBLISHED, "reproduction": {}}
    for task in SUBSET:
        runs = [weka_cv(d, task, seed) for seed in range(1, 11)]
        keys = [k for k in runs[0] if k != "correct"]
        res["reproduction"][task] = {
            "seed1": runs[0],
            "seeds1_10": {k: {"mean": float(np.mean([r[k] for r in runs])), "sd": float(np.std([r[k] for r in runs])),
                              "min": float(np.min([r[k] for r in runs])), "max": float(np.max([r[k] for r in runs]))}
                          for k in keys}}
        r = res["reproduction"][task]["seeds1_10"]
        print(f"{task:9s} published AUC {PUBLISHED[task]['auc']:.2f} acc {PUBLISHED[task]['accuracy']:.4f} | "
              f"ours AUC {r['auc']['mean']:.3f}±{r['auc']['sd']:.3f} [{r['auc']['min']:.3f},{r['auc']['max']:.3f}] "
              f"acc {r['accuracy']['mean']:.3f}±{r['accuracy']['sd']:.3f}")
    (out / "reproduction.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["audit"], indent=1, default=str))


if __name__ == "__main__":
    main()
