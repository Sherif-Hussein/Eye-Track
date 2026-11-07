"""Selection-adjusted estimate of the "best config" for each task.

Picking the config with the highest mean AUC after seeing all 18 results is optimistic. Here, for
every outer fold of every repeat, the config is chosen by its pooled AUC on the *other nine folds*
of the same repeat (out-of-fold scores already on disk), and only that config's scores for the
held-out fold are kept. The kept scores are then evaluated as one model. This is cheap and much less
optimistic than hindsight selection. It is not fully nested: the other folds' scores came from
models whose training sets included the held-out participants.

Usage: python -I scripts/phase2_mazza_select.py <results_dir>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edd.mazza import TASKS, fast_auc  # noqa: E402
from edd.metrics import binary_metrics, bootstrap_ci, summarise_repeats  # noqa: E402
from phase2_mazza_configs import CONFIGS  # noqa: E402


def main():
    out = Path(sys.argv[1])
    res = {}
    for task in TASKS:
        oofs = {}
        for name in CONFIGS:
            p = out / "oof" / f"{task}__{name}.csv"
            if p.exists():
                o = pd.read_csv(p, dtype={"unit": str})
                # put every model on a common scale (keeps the sign, so threshold 0 is unchanged):
                # divide by the SD of that model's out-of-fold scores in the same repeat (labels not used)
                o["score"] = o.score / o.groupby("repeat").score.transform("std")
                oofs[name] = o
        if not oofs:
            continue
        kept, picks = [], []
        ref = next(iter(oofs.values()))
        for (r, f), _ in ref.groupby(["repeat", "fold"]):
            best, best_auc = None, -1.0
            for name, o in oofs.items():
                other = o[(o.repeat == r) & (o.fold != f)]
                a = fast_auc(other.y, other.score)
                if a > best_auc:
                    best, best_auc = name, a
            o = oofs[best]
            kept.append(o[(o.repeat == r) & (o.fold == f)][["repeat", "fold", "unit", "y", "score"]].assign(config=best))
            picks.append(best)
        k = pd.concat(kept, ignore_index=True)
        k.to_csv(out / "oof" / f"{task}__selected_leave_fold_out.csv", index=False)
        reps = [binary_metrics(t.y, t.score) for _, t in k.groupby("repeat")]
        avg = k.groupby("unit").agg(y=("y", "first"), score=("score", "mean"))
        hindsight = max(oofs, key=lambda n: np.mean([binary_metrics(t.y, t.score)["auc"]
                                                     for _, t in oofs[n].groupby("repeat")]))
        res[task] = {"n_candidates": len(oofs), "repeats": summarise_repeats(reps),
                     "avg_scores": binary_metrics(avg.y, avg.score),
                     "ci95": bootstrap_ci(avg.y.to_numpy(), avg.score.to_numpy(), n_boot=1000),
                     "pick_counts": pd.Series(picks).value_counts().to_dict(), "hindsight_best": hindsight,
                     "note": "per-fold choice by pooled AUC on the other 9 folds of the same repeat; scores of "
                             "each model divided by their within-repeat SD so mixed-model pooling is fair"}
        print(task, round(res[task]["repeats"]["auc"]["mean"], 3), res[task]["pick_counts"], flush=True)
    (out / "selection.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
