"""Post hoc, EXPLORATORY: for LR + XGBoost, LR + CatBoost and all five models: plain average (AVG), stacking as fixed in mc_stack.py (STK), and stacking rescaled (STKZ).
STKZ is a check added after STK came out below both of its own models: the meta-learner's scores differ in scale and intercept between folds, which can lower an AUC pooled over folds.
STKZ z-scales each fold's meta-learner output with the mean and SD of its training-people output, the same treatment the base scores get before averaging.
Also reports the mean within-fold item AUC, which no between-fold scale can affect. No base model is retrained; repeat 0; paired bootstrap intervals only; no new test family.
Writes results_stack_pairs.csv/.json in this folder."""
import json
import pickle
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import mc_lib as M

L = M.L
import ms_eval as E
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
CK = HERE / "ck3"
reps = [0]
BASE = ["LR", "XGB", "LGBM", "CAT", "SVM"]
COMBOS = {"X": ["LR", "XGB"], "C": ["LR", "CAT"], "5": BASE}
NEW = [f"{k}{c}" for c in COMBOS for k in ("AVG", "STK", "STKZ")]
NAMES = BASE + NEW
CONDS = {"FULL": ["D30"], "CAMERA": ["W30_d1", "W30_d2", "W30_d3"]}


def scaled(d):
    m, s = d["inner_oof"].mean(), d["inner_oof"].std()
    return (d["inner_oof"] - m) / s, (d["score"] - m) / s


within = {}
weights = {}
for cond, tags in CONDS.items():
    for t in tags:
        (CK / t).mkdir(parents=True, exist_ok=True)
        for f in (ROOT / "ck" / t).glob("meta_*.pkl"):
            shutil.copy(f, CK / t / f.name)
        for c in NAMES:
            (CK / t / c).mkdir(exist_ok=True)
        for sk in L.SETS:
            y = pickle.load(open(CK / t / f"meta_{sk}.pkl", "rb"))["y"]
            o = {c: pickle.load(open(ROOT / "ck" / t / c / f"{sk}_r0.pkl", "rb"))[0] for c in BASE}
            res = {c: [] for c in NEW}
            for k in range(len(o["LR"])):
                for name, ms in COMBOS.items():
                    ds = [o[c][k] for c in ms]
                    Itr = np.column_stack([scaled(d)[0] for d in ds])
                    Ite = np.column_stack([scaled(d)[1] for d in ds])
                    base = dict(tr=ds[0]["tr"], te=ds[0]["te"], hp=("post hoc",), cv_auc=np.array([np.nan]))
                    res[f"AVG{name}"].append(dict(base, inner_oof=Itr.mean(1), score=Ite.mean(1)))
                    meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Itr, y[ds[0]["tr"]])
                    weights.setdefault((cond, name), []).append(meta.coef_.ravel())
                    tr_s, te_s = meta.decision_function(Itr), meta.decision_function(Ite)
                    res[f"STK{name}"].append(dict(base, inner_oof=tr_s, score=te_s))
                    res[f"STKZ{name}"].append(dict(base, inner_oof=(tr_s - tr_s.mean()) / tr_s.std(), score=(te_s - tr_s.mean()) / tr_s.std()))
            for c in BASE:
                pickle.dump({0: o[c]}, open(CK / t / c / f"{sk}.pkl", "wb"))
                within.setdefault((cond, c), []).append(np.mean([roc_auc_score(y[d["te"]], d["score"]) for d in o[c]]))
            for c in NEW:
                pickle.dump({0: res[c]}, open(CK / t / c / f"{sk}.pkl", "wb"))
                within.setdefault((cond, c), []).append(np.mean([roc_auc_score(y[d["te"]], d["score"]) for d in res[c]]))
L.CK = CK
if hasattr(E, "CK"):
    E.CK = CK
sets = L.SETS
rows = []
for cond, ts in CONDS.items():
    fr = {c: [E.item_frame(t, c, sets, reps) for t in ts] for c in NAMES}
    sd, ow, nm = E.to_sets(fr, sets)
    pt, bt = E.point(sd, ow, nm), E.boot(sd, ow, nm, 2000, E.BOOT_SEED)
    pf = {c: [E.person_frame(t, c, sets, "mean", reps)[1] for t in ts] for c in NAMES}
    sd2, ow2, nm2 = E.to_sets(pf, sets)
    p2, b2 = E.point(sd2, ow2, nm2), E.boot(sd2, ow2, nm2, 2000, E.BOOT_SEED + 1)
    for c in NAMES:
        row = dict(condition=cond, model=c, item_auc=pt[c], within_fold_auc=float(np.mean(within[(cond, c)])), person_auc=p2[c])
        refs = ["LR"] + ([f"AVG{c[-1]}"] if c.startswith("STK") else [])
        for ref in refs:
            if c != ref:
                d, dp = bt[c] - bt[ref], b2[c] - b2[ref]
                row.update({f"item_vs_{ref[:3]}": pt[c] - pt[ref], f"item_vs_{ref[:3]}_lo": np.percentile(d, 2.5), f"item_vs_{ref[:3]}_hi": np.percentile(d, 97.5),
                            f"person_vs_{ref[:3]}": p2[c] - p2[ref], f"person_vs_{ref[:3]}_lo": np.percentile(dp, 2.5), f"person_vs_{ref[:3]}_hi": np.percentile(dp, 97.5)})
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(HERE / "results_stack_pairs.csv", index=False)
w = {f"{c}_{n}": {"mean": np.round(np.mean(v, 0), 2).tolist(), "sd": np.round(np.std(v, 0), 2).tolist()} for (c, n), v in weights.items()}
(HERE / "results_stack_pairs.json").write_text(json.dumps({"meta_weights": w, "rows": json.loads(df.to_json(orient="records"))}, indent=1), encoding="utf-8")
shutil.rmtree(CK)
pd.set_option("display.width", 250)
print(df[["condition", "model", "item_auc", "within_fold_auc", "person_auc"]].round(4).to_string(index=False))
print(df[df.model.isin(NEW)][["condition", "model", "item_vs_LR", "item_vs_LR_lo", "item_vs_LR_hi", "person_vs_LR", "person_vs_LR_lo", "person_vs_LR_hi"]].round(3).to_string(index=False))
print(df[df.model.str.startswith("STK")][["condition", "model", "item_vs_AVG", "item_vs_AVG_lo", "item_vs_AVG_hi", "person_vs_AVG", "person_vs_AVG_lo", "person_vs_AVG_hi"]].round(3).to_string(index=False))
print(json.dumps(w))
