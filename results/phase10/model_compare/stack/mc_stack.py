"""Post hoc, EXPLORATORY: five-model stacking on the saved scores of Task M2 (repeat 0). No base model is retrained. Development data only (Delmas); held-out data never read.
Rules, fixed before this script was run:
 - Base models: logistic regression, XGBoost, LightGBM, CatBoost, RBF-SVM, with the settings already chosen inside the training people.
 - Scale: within each outer fold, each model's scores are z-scaled with the mean and SD of its own inner-fold training scores.
 - STACK: L2 logistic regression, C = 1, class_weight balanced, no tuning, fitted only on the training people's inner-fold scores, applied to the test people's scores.
 - AVG5: plain average of the five scaled scores. ENS: logistic regression + XGBoost 50/50 (ADDENDUM_ensemble.md), shown for comparison and not in the Holm family.
 - Conditions: FULL (clean, 13 features) and CAMERA (simulated webcam noise, 10 features, mean over three draws).
 - Holm over 4 rows (STACK, AVG5 x FULL, CAMERA), separately at item and person level. Better = interval above zero and Holm p < .05; worse = interval below zero and Holm p < .05.
Thresholded rates are not reported for STACK, because its training scores are in-sample for the meta-learner.
Usage: .venv/Scripts/python.exe -I results/phase10/model_compare/stack/mc_stack.py   (writes only into this folder)"""
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
CK = HERE / "ck"
reps = [0]
BASE = ["LR", "XGB", "LGBM", "CAT", "SVM"]
NEW = ["ENS", "AVG5", "STACK"]
NAMES = BASE + NEW
CONDS = {"FULL": ["D30"], "CAMERA": ["W30_d1", "W30_d2", "W30_d3"]}
N_BOOT, N_PERM = 2000, 2000


def scaled(d):
    m, s = d["inner_oof"].mean(), d["inner_oof"].std()
    return (d["inner_oof"] - m) / s, (d["score"] - m) / s


def build():
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
                ens, avg, stk = [], [], []
                for k in range(len(o["LR"])):
                    ds = [o[c][k] for c in BASE]
                    for d in ds:
                        assert (d["te"] == ds[0]["te"]).all() and (d["tr"] == ds[0]["tr"]).all()
                        assert np.isfinite(d["inner_oof"]).all()
                    Itr = np.column_stack([scaled(d)[0] for d in ds])
                    Ite = np.column_stack([scaled(d)[1] for d in ds])
                    base = dict(tr=ds[0]["tr"], te=ds[0]["te"], hp=("post hoc",), cv_auc=np.array([np.nan]))
                    ens.append(dict(base, inner_oof=Itr[:, :2].mean(1), score=Ite[:, :2].mean(1)))
                    avg.append(dict(base, inner_oof=Itr.mean(1), score=Ite.mean(1)))
                    meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Itr, y[ds[0]["tr"]])
                    weights[cond].append(meta.coef_.ravel())
                    stk.append(dict(base, inner_oof=meta.decision_function(Itr), score=meta.decision_function(Ite)))
                for c, v in zip(NEW, (ens, avg, stk)):
                    pickle.dump({0: v}, open(CK / t / c / f"{sk}.pkl", "wb"))
    return {c: {b: [float(np.mean(np.array(v)[:, i])), float(np.std(np.array(v)[:, i]))] for i, b in enumerate(BASE)} for c, v in weights.items()}


def main():
    w = build()
    L.CK = CK
    if hasattr(E, "CK"):
        E.CK = CK
    sets = L.SETS
    rows = []
    for cond, ts in CONDS.items():
        fr = {c: [E.item_frame(t, c, sets, reps) for t in ts] for c in NAMES}
        sd, ow, nm = E.to_sets(fr, sets)
        pt, bt = E.point(sd, ow, nm), E.boot(sd, ow, nm, N_BOOT, E.BOOT_SEED)
        pf = {c: [E.person_frame(t, c, sets, "mean", reps)[1] for t in ts] for c in NAMES}
        sd2, ow2, nm2 = E.to_sets(pf, sets)
        p2, b2 = E.point(sd2, ow2, nm2), E.boot(sd2, ow2, nm2, N_BOOT, E.BOOT_SEED + 1)
        P = np.array([E.W.auc_matrix(s.y, s.S) for s, _ in sd.values()])
        for c in NAMES:
            im = [E.item_metrics(t, c, sets, reps) for t in ts]
            row = dict(condition=cond, model=c, item_auc=pt[c], item_lo=np.percentile(bt[c], 2.5), item_hi=np.percentile(bt[c], 97.5), item_eer=float(np.mean([m["item_eer"] for m in im])),
                       person_auc=p2[c], person_lo=np.percentile(b2[c], 2.5), person_hi=np.percentile(b2[c], 97.5))
            if c != "LR":
                d, dp = bt[c] - bt["LR"], b2[c] - b2["LR"]
                row.update(item_delta=pt[c] - pt["LR"], item_delta_lo=np.percentile(d, 2.5), item_delta_hi=np.percentile(d, 97.5), person_delta=p2[c] - p2["LR"], person_delta_lo=np.percentile(dp, 2.5), person_delta_hi=np.percentile(dp, 97.5))
            if c in ("AVG5", "STACK"):
                row["item_swap_p"] = E.swap_p(sd, ow, "LR", c, N_PERM, E.SWAP_SEED)[1]
                row["person_swap_p"] = E.swap_p(sd2, ow2, "LR", c, N_PERM, E.SWAP_SEED + 1)[1]
                for ref in ("ENS", "AVG5"):
                    if ref != c:
                        d, dp = bt[c] - bt[ref], b2[c] - b2[ref]
                        row.update({f"item_minus_{ref}": pt[c] - pt[ref], f"item_minus_{ref}_lo": np.percentile(d, 2.5), f"item_minus_{ref}_hi": np.percentile(d, 97.5),
                                    f"person_minus_{ref}": p2[c] - p2[ref], f"person_minus_{ref}_lo": np.percentile(dp, 2.5), f"person_minus_{ref}_hi": np.percentile(dp, 97.5)})
            if len(ts) == 1:
                row["per_set"] = " ".join(f"{v:.3f}" for v in P[:, ow == c].ravel())
            rows.append(row)
    df = pd.DataFrame(rows)
    fam = df[df.model.isin(["AVG5", "STACK"])].copy()
    fam["item_holm"], fam["person_holm"] = E.holm(fam.item_swap_p.to_numpy()), E.holm(fam.person_swap_p.to_numpy())
    fam["item_verdict"] = np.where((fam.item_delta_lo > 0) & (fam.item_holm < 0.05), "better", np.where((fam.item_delta_hi < 0) & (fam.item_holm < 0.05), "worse", "no reliable difference"))
    fam["person_verdict"] = np.where((fam.person_delta_lo > 0) & (fam.person_holm < 0.05), "better", np.where((fam.person_delta_hi < 0) & (fam.person_holm < 0.05), "worse", "no reliable difference"))
    df = df.merge(fam[["condition", "model", "item_holm", "person_holm", "item_verdict", "person_verdict"]], on=["condition", "model"], how="left")
    df.to_csv(HERE / "results_stack.csv", index=False)
    (HERE / "results_stack.json").write_text(json.dumps({"meta_weights_mean_sd": w, "rows": json.loads(df.to_json(orient="records"))}, indent=1), encoding="utf-8")
    shutil.rmtree(CK)
    show = ["condition", "model", "item_auc", "item_lo", "item_hi", "item_eer", "item_delta", "item_delta_lo", "item_delta_hi", "item_holm", "item_verdict"]
    print(df[show].round(4).to_string(index=False))
    print(df[["condition", "model", "person_auc", "person_lo", "person_hi", "person_delta", "person_delta_lo", "person_delta_hi", "person_holm", "person_verdict"]].round(4).to_string(index=False))
    ex = [c for c in df.columns if c.startswith(("item_minus", "person_minus"))]
    print(df[df.model.isin(["AVG5", "STACK"])][["condition", "model"] + ex].round(4).T.to_string())
    print("per set\n", df[df.condition == "FULL"][["model", "per_set"]].to_string(index=False))
    print("meta weights mean, SD over folds:", json.dumps({c: {b: [round(x, 2) for x in v] for b, v in d.items()} for c, d in w.items()}))


if __name__ == "__main__":
    main()
