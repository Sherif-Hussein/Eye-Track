"""Phase 2 (celniak track): complement to phase2_celniak_repro.py. Same SVM, same 216-point grid, same best-of-grid
selection on the scored folds, but folds are RANDOM SLIDE splits (StratifiedKFold, 5 splits, shuffled) that ignore
participants, so the same person's slides sit in train and test. Participant score = mean of the out-of-fold slide
scores. This is a leaky protocol, reported only to show the size of the inflation it can create.

Output: results/phase2/celniak/reproduction_C_random_slide_split_s<seed>.csv and stage_reproduction_leak.json
Usage: python -I scripts/phase2_celniak_repro_leak.py [--jobs 2]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from edd.celniak import PAPER_FEATURES  # noqa: E402
from edd.metrics import binary_metrics  # noqa: E402

OUT = ROOT / "results" / "phase2" / "celniak"
sys.path.insert(0, str(ROOT / "scripts"))
from phase2_celniak_repro import GRID  # noqa: E402


def one_point(k, C, gam, X, y, g, splits):
    parts = []
    for tr, te in splits:
        m = Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif, k=k)), ("clf", SVC(C=C, gamma=gam))])
        m.fit(X[tr], y[tr])
        parts.append(pd.DataFrame({"unit": g[te], "y": y[te], "score": m.decision_function(X[te])}))
    s = pd.concat(parts)
    a = s.groupby("unit").agg(y=("y", "first"), score=("score", "mean"))
    mm = binary_metrics(a.y, a.score)
    return {"k": k, "C": C, "gamma": gam, **{f"mean_{n}": mm[n] for n in ("accuracy", "sensitivity", "specificity", "f1", "auc", "eer", "hter")},
            "slide_accuracy": float(((s.score >= 0).astype(int) == s.y).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    sl = pd.read_csv(ROOT / "data" / "derived" / "celniak_slide_features.csv")
    X, y, g = sl[PAPER_FEATURES].to_numpy(float), sl["label"].to_numpy(int), sl["recording"].astype(str).to_numpy()
    summ = {}
    for seed in range(3):
        f = OUT / f"reproduction_C_random_slide_split_s{seed}.csv"
        if not f.exists():
            splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
            df = pd.DataFrame(Parallel(n_jobs=a.jobs)(delayed(one_point)(k, C, gm, X, y, g, splits) for k, C, gm in GRID))
            df.to_csv(f, index=False)
        df = pd.read_csv(f)
        b = df.sort_values(["mean_accuracy", "mean_auc"], ascending=False).iloc[0]
        summ[str(seed)] = {"best_by_accuracy": {k: (v.item() if hasattr(v, "item") else v) for k, v in b.items()},
                           "max_auc": float(df.mean_auc.max()), "median_auc": float(df.mean_auc.median()),
                           "median_accuracy": float(df.mean_accuracy.median())}
        print(seed, summ[str(seed)]["best_by_accuracy"], flush=True)
    (OUT / "stage_reproduction_leak.json").write_text(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
