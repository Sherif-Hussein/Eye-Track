"""Post hoc, EXPLORATORY: logistic regression + XGBoost, averaged 50/50 (ENS) or stacked (STACK2), on the saved scores of Task M2 (repeat 0). No base model is retrained.
Same rules as mc_stack.py: z-scale from each model's inner-fold training scores; meta-learner L2 logistic regression, C = 1, class_weight balanced, no tuning, fitted only on the training people's inner-fold scores.
Paired bootstrap intervals only; no new test family. Writes results_stack2.csv/.json in this folder."""
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

sys.stdout.reconfigure(encoding="utf-8")
CK = HERE / "ck2"
reps = [0]
BASE = ["LR", "XGB"]
NAMES = BASE + ["ENS", "STACK2"]
CONDS = {"FULL": ["D30"], "CAMERA": ["W30_d1", "W30_d2", "W30_d3"]}


def scaled(d):
    m, s = d["inner_oof"].mean(), d["inner_oof"].std()
    return (d["inner_oof"] - m) / s, (d["score"] - m) / s


weights = {c: [] for c in CONDS}
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
            for c in BASE:
                pickle.dump({0: o[c]}, open(CK / t / c / f"{sk}.pkl", "wb"))
            ens, stk = [], []
            for a, b in zip(o["LR"], o["XGB"]):
                assert (a["te"] == b["te"]).all() and (a["tr"] == b["tr"]).all()
                Itr = np.column_stack([scaled(a)[0], scaled(b)[0]])
                Ite = np.column_stack([scaled(a)[1], scaled(b)[1]])
                base = dict(tr=a["tr"], te=a["te"], hp=("post hoc",), cv_auc=np.array([np.nan]))
                ens.append(dict(base, inner_oof=Itr.mean(1), score=Ite.mean(1)))
                meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Itr, y[a["tr"]])
                weights[cond].append(meta.coef_.ravel())
                stk.append(dict(base, inner_oof=meta.decision_function(Itr), score=meta.decision_function(Ite)))
            pickle.dump({0: ens}, open(CK / t / "ENS" / f"{sk}.pkl", "wb"))
            pickle.dump({0: stk}, open(CK / t / "STACK2" / f"{sk}.pkl", "wb"))
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
    P = np.array([E.W.auc_matrix(s.y, s.S) for s, _ in sd.values()])
    for c in NAMES:
        row = dict(condition=cond, model=c, item_auc=pt[c], person_auc=p2[c], item_eer=float(np.mean([E.item_metrics(t, c, sets, reps)["item_eer"] for t in ts])))
        if len(ts) == 1:
            row["per_set"] = " ".join(f"{v:.3f}" for v in P[:, ow == c].ravel())
        for ref in ("LR", "ENS"):
            if c != ref:
                d, dp = bt[c] - bt[ref], b2[c] - b2[ref]
                row.update({f"item_vs_{ref}": pt[c] - pt[ref], f"item_vs_{ref}_lo": np.percentile(d, 2.5), f"item_vs_{ref}_hi": np.percentile(d, 97.5),
                            f"person_vs_{ref}": p2[c] - p2[ref], f"person_vs_{ref}_lo": np.percentile(dp, 2.5), f"person_vs_{ref}_hi": np.percentile(dp, 97.5)})
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(HERE / "results_stack2.csv", index=False)
w = {c: {"LR": [float(np.mean(np.array(v)[:, 0])), float(np.std(np.array(v)[:, 0]))], "XGB": [float(np.mean(np.array(v)[:, 1])), float(np.std(np.array(v)[:, 1]))],
         "share_of_folds_LR_weight_negative": float(np.mean(np.array(v)[:, 0] < 0))} for c, v in weights.items()}
(HERE / "results_stack2.json").write_text(json.dumps({"meta_weights_mean_sd": w, "rows": json.loads(df.to_json(orient="records"))}, indent=1), encoding="utf-8")
shutil.rmtree(CK)
print(df[df.model == "STACK2"].drop(columns=["per_set"]).round(4).T.to_string())
print(df[["condition", "model", "item_auc", "item_eer", "person_auc"]].round(4).to_string(index=False))
print(df[df.condition == "FULL"][["model", "per_set"]].to_string(index=False))
print(json.dumps(w))
