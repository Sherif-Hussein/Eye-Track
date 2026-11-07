"""Preview of single model vs voting vs stacking vs cascade, on saved out-of-fold (OOF) participant scores.

Every data-driven choice (member ranking, weights, stacker, cascade threshold) is fitted on the other outer
folds of the same repeat and applied to the held-out fold. This is a preview, not the final protocol:
member models from the other folds were trained on data that includes the held-out fold, so the stacker
and weighted vote are slightly optimistic. The final run must fit them in a fully nested loop.

Usage: python -I scripts/ensemble_preview.py <results/phase2> <out_dir>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from edd.metrics import binary_metrics  # noqa: E402


def load_member(path):
    d = pd.read_csv(path, usecols=lambda c: c in ("repeat", "fold", "unit", "y", "score"))
    return d.groupby(["repeat", "fold", "unit"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))


def load_pool(files):
    table = None
    for name, path in files.items():
        d = load_member(path).rename(columns={"score": name})
        table = d if table is None else table.merge(d, on=["repeat", "fold", "unit", "y"], how="inner")
    return table


def safe_auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else 0.5


def predict_fold(train, test, members, cheap, defer_frac=0.5):
    """Return {method: held-out scores} plus the share of test rows the cascade deferred."""
    mu, sd = train[members].mean(), train[members].std().replace(0, 1)
    ztr, zte = (train[members] - mu) / sd, (test[members] - mu) / sd
    auc = pd.Series({m: safe_auc(train.y, train[m]) for m in members}).sort_values(ascending=False)
    top3 = list(auc.index[:3])
    w = (auc[members] - 0.5).clip(lower=0)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1 / len(members), index=members)
    out = {"single_selected": zte[auc.index[0]].to_numpy(),
           "single_cheap": zte[cheap].to_numpy(),
           "hard_vote_top3": (zte[top3] > 0).mean(axis=1).to_numpy() - 0.5,
           "soft_vote_all": zte.mean(axis=1).to_numpy(),
           "soft_vote_top3": zte[top3].mean(axis=1).to_numpy(),
           "weighted_vote_all": (zte[members] * w).sum(axis=1).to_numpy()}
    stack = LogisticRegression(C=0.1, max_iter=500).fit(ztr[members], train.y)
    out["stacking_l2"] = stack.decision_function(zte[members])
    # Cascade: the cheap model keeps confident cases; the rest go to the top-3 soft vote.
    tau = np.quantile(np.abs(ztr[cheap]), defer_frac)
    defer = np.abs(zte[cheap].to_numpy()) < tau
    out["cascade"] = np.where(defer, zte[top3].mean(axis=1).to_numpy(), zte[cheap].to_numpy())
    return out, defer


def evaluate_pool(pool, members, cheap, n_boot=2000, seed=0):
    rows = []
    for r, rep in pool.groupby("repeat"):
        for f, test in rep.groupby("fold"):
            scores, defer = predict_fold(rep[rep.fold != f], test, members, cheap)
            for method, s in scores.items():
                rows.append(pd.DataFrame({"repeat": r, "unit": test.unit.to_numpy(), "y": test.y.to_numpy(),
                                          "method": method, "score": s, "deferred": defer}))
    long = pd.concat(rows, ignore_index=True)
    res = []
    for method, g in long.groupby("method"):
        reps = [binary_metrics(x.y, x.score) for _, x in g.groupby("repeat")]
        row = {"method": method}
        row.update({k: float(np.mean([m[k] for m in reps])) for k in ("auc", "accuracy", "balanced_accuracy", "eer", "hter")})
        row["auc_sd"] = float(np.std([m["auc"] for m in reps]))
        res.append(row)
    res = pd.DataFrame(res)

    # Paired participant bootstrap of the AUC gain over the honestly selected single model.
    wide = long.groupby(["method", "unit"]).agg(y=("y", "first"), score=("score", "mean")).reset_index()
    base = wide[wide.method == "single_selected"].set_index("unit")
    rng = np.random.default_rng(seed)
    pos, neg = base.index[base.y == 1].to_numpy(), base.index[base.y == 0].to_numpy()
    samples = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n_boot)]
    gains = {}
    for method, g in wide.groupby("method"):
        s = g.set_index("unit")
        d = [safe_auc(base.y.loc[i], s.score.loc[i]) - safe_auc(base.y.loc[i], base.score.loc[i]) for i in samples]
        gains[method] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    res["gain_ci_lo"] = res.method.map(lambda m: gains[m][0])
    res["gain_ci_hi"] = res.method.map(lambda m: gains[m][1])
    res["models_per_decision"] = res.method.map({
        "single_selected": 1, "single_cheap": 1, "hard_vote_top3": 3, "soft_vote_top3": 3,
        "soft_vote_all": len(members), "weighted_vote_all": len(members), "stacking_l2": len(members),
        "cascade": 1 + 3 * float(long[long.method == "cascade"].deferred.mean())})
    best_member = max(members, key=lambda m: np.mean([safe_auc(x.y, x[m]) for _, x in pool.groupby("repeat")]))
    corr = pool[members].corr(method="spearman").to_numpy()
    return res.sort_values("auc", ascending=False), best_member, float(corr[np.triu_indices_from(corr, 1)].mean())


def main(phase2, out_dir):
    phase2, out_dir = Path(phase2), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pools = {}
    for task in ("g2_vs_g3", "guilty_vs_innocent", "deceptive_vs_honest"):
        files = {p.name.split("__")[1][:-4]: p for p in sorted((phase2 / "mazza/oof").glob(f"{task}__*.csv"))
                 if "base_pub36" not in p.name}
        pools[f"mazza_{task}"] = (files, "i3_crime_lr")
    s = phase2 / "preti/scores"
    ocular = ["B_ocular", "I1_diff_ocular", "I1b_contrast_only_ocular", "I2_neutralz_subject_ocular",
              "I3_trial_neutralz_aggtuned_ocular"]
    pools["preti_ocular_only"] = ({k: s / f"{k}.csv" for k in ocular}, "B_ocular")
    pools["preti_all_signals"] = ({k: s / f"{k}.csv" for k in ocular + ["B_rt", "I1_diff_rt", "B_fusion", "I1_diff_fusion"]}, "B_rt")
    tables = []
    for name, (files, cheap) in pools.items():
        pool = load_pool(files)
        members = list(files)
        res, best, rho = evaluate_pool(pool, members, cheap)
        res.insert(0, "pool", name)
        tables.append(res)
        print(f"\n== {name}: {pool.unit.nunique()} participants, {len(members)} members, mean Spearman between members "
              f"{rho:.2f}, best member in hindsight: {best}")
        print(res.drop(columns=["pool"]).round(3).to_string(index=False))
    pd.concat(tables).to_csv(out_dir / "ensemble_preview.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
