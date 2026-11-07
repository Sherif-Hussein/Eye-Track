"""Paired participant-bootstrap of 4-class accuracy and macro one-vs-rest AUC (config minus baseline), on
repeat-averaged class probabilities, Holm across the 7 configs. Reads results/phase2/mazza/oof/4class__*.csv.
Usage: python -I scripts/phase2_mazza_4class_paired.py results/phase2/mazza"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

out = Path(sys.argv[1])
names = ["base_pub36_lr_unreg", "i1_pub36_lr", "i1_raw90_lr", "i1_raw90_lgbm", "i2_diffshare_lr",
         "i3_crime_lr", "i3_crime_diffshare_lr", "i4_all_lr_anova"]
P = ["p1", "p2", "p3", "p4"]


def avg(n):
    d = pd.read_csv(out / "oof" / f"4class__{n}.csv", dtype={"unit": str})
    return d.groupby("unit").agg(y=("y", "first"), **{p: (p, "mean") for p in P}).sort_index()


def stat(y, p):
    acc = (np.asarray([1, 2, 3, 4])[p.argmax(1)] == y).mean()
    return acc, roc_auc_score(y, p, multi_class="ovr", average="macro")


b = avg(names[0])
y = b.y.to_numpy()
rng = np.random.default_rng(0)
strata = [np.flatnonzero(y == c) for c in (1, 2, 3, 4)]
idxs = [np.concatenate([rng.choice(s, len(s)) for s in strata]) for _ in range(1000)]
res, praw = {}, {}
for n in names[1:]:
    a = avg(n).loc[b.index]
    pa, pb = a[P].to_numpy(), b[P].to_numpy()
    da, db = [], []
    for i in idxs:
        sa, sb = stat(y[i], pa[i]), stat(y[i], pb[i])
        da.append(sa[0] - sb[0]); db.append(sa[1] - sb[1])
    oa, ob = stat(y, pa), stat(y, pb)
    da, db = np.array(da), np.array(db)
    p = min(1, 2 * min((da <= 0).mean(), (da >= 0).mean())); p = max(p, 1 / len(idxs))
    res[n] = {"acc_avg_scores": oa[0], "macro_auc_avg_scores": oa[1],
              "delta_acc": oa[0] - ob[0], "delta_acc_ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
              "delta_macro_auc": oa[1] - ob[1], "delta_macro_auc_ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
              "p_boot_acc": float(p)}
    praw[n] = p
keys = sorted(praw, key=praw.get); run = 0
for i, k in enumerate(keys):
    run = max(run, min(1.0, (len(keys) - i) * praw[k])); res[k]["p_holm_acc"] = run
res["_baseline"] = {"acc_avg_scores": stat(y, b[P].to_numpy())[0], "macro_auc_avg_scores": stat(y, b[P].to_numpy())[1],
                    "note": "1000 stratified (by group) participant bootstrap resamples; accuracy-difference p, Holm over 7"}
(out / "fourclass_paired.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(k, {a: (round(x, 3) if isinstance(x, float) else x) for a, x in v.items() if a != "note"})
