"""Task M2 evaluation. EXPLORATORY. Source: results/phase10/model_search/ms_eval.py (metrics, paired participant bootstrap, paired swap, Holm) imported
unchanged; only the candidate list, tags and labels differ. Writes into results/phase10/model_compare/ only.
Usage: .venv/Scripts/python.exe -I results/phase10/model_compare/mc_eval.py [--repeats 0] [--nboot 2000] [--nperm 2000]
Primary metric: item AUC (pooled per group set, repeat-averaged out-of-fold scores, mean over six sets, mean over 3 draws for CAMERA).
Holm family = 5 non-reference models x 2 conditions = 10 rows (4 models + the 50/50 LR+XGB ensemble); item and person verdicts are given separately.
Verdict: better = bootstrap interval of the difference from logistic regression above zero AND Holm p < .05 (Holm over all model x condition rows);
worse = interval below zero AND Holm p < .05; otherwise no reliable difference.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_lib as M  # noqa: E402

L = M.L
import ms_eval as E  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CK = M.HERE / "ck"
M.LABEL["ENS"] = "Ensemble (logistic regression + XGBoost)"
MODELS = M.MODELS + ["ENS"]       # addendum: 50/50 ensemble of LR and XGB (mc_ens.py); NOT BLIND, see ADDENDUM_ensemble.md


def tags(cond):
    base = M.CONDS[cond][0]
    return [base] if cond == "FULL" else [f"{base}_d{d}" for d in M.CONDS[cond][1]]


def merge_ck(reps):
    """parts (<set>_r<r>.pkl) -> one {repeat: folds} file per tag x model x set in the layout ms_eval reads (ck/<tag>/<model>/<set>.pkl)."""
    for cond in M.CONDS:
        for t in tags(cond):
            for m in MODELS:
                for sk in L.SETS:
                    d = {}
                    for r in reps:
                        p = CK / t / m / f"{sk}_r{r}.pkl"
                        d.update(pickle.load(open(p, "rb")))
                    pickle.dump(d, open(L.ck_path(t, m, sk), "wb"))


def evaluate(reps, n_boot, n_perm):
    merge_ck(reps)
    L.CK = CK
    rows = []
    for cond in M.CONDS:
        ts = tags(cond)
        sets = L.SETS
        frames = {c: [E.item_frame(t, c, sets, reps) for t in ts] for c in MODELS}
        sdb, owner, names = E.to_sets(frames, sets)
        pt = E.point(sdb, owner, names)
        bt = E.boot(sdb, owner, names, n_boot, E.BOOT_SEED)
        pfr, pers = {}, {}
        for c in MODELS:
            tabs = [E.person_frame(t, c, sets, "mean", reps) for t in ts]
            pfr[c] = [x[1] for x in tabs]
            pers[c] = [x[0] for x in tabs]
        psd, powner, pnames = E.to_sets(pfr, sets)
        ppt = E.point(psd, powner, pnames)
        pbt = E.boot(psd, powner, pnames, n_boot, E.BOOT_SEED + 1)
        for c in MODELS:
            row = {"condition": cond, "model": c, "label": M.LABEL[c], "n_draws": len(ts), "repeats": ",".join(map(str, reps))}
            im = [E.item_metrics(t, c, sets, reps) for t in ts]
            row.update({k: float(np.mean([m[k] for m in im])) for k in im[0]})
            row["item_auc"] = pt[c]
            row["item_auc_lo"], row["item_auc_hi"] = float(np.percentile(bt[c], 2.5)), float(np.percentile(bt[c], 97.5))
            d = bt[c] - bt["LR"]
            row["item_delta"] = pt[c] - pt["LR"]
            row["item_delta_lo"], row["item_delta_hi"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            ms = [E.person_metrics(t) for t in pers[c]]
            row.update({k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
            row["person_auc"] = ppt[c]
            row["person_auc_lo"], row["person_auc_hi"] = float(np.percentile(pbt[c], 2.5)), float(np.percentile(pbt[c], 97.5))
            dp = pbt[c] - pbt["LR"]
            row["person_delta"] = ppt[c] - ppt["LR"]
            row["person_delta_lo"], row["person_delta_hi"] = float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5))
            if c != "LR":
                row["item_swap_obs"], row["item_swap_p"] = E.swap_p(sdb, owner, "LR", c, n_perm, E.SWAP_SEED)
                row["person_swap_obs"], row["person_swap_p"] = E.swap_p(psd, powner, "LR", c, n_perm, E.SWAP_SEED + 1)
            if c == "ENS":      # labelled extra: ensemble against XGBoost alone (paired interval only)
                dx = bt["ENS"] - bt["XGB"]
                row["extra_ens_vs_xgb_item_delta"] = pt["ENS"] - pt["XGB"]
                row["extra_ens_vs_xgb_item_lo"], row["extra_ens_vs_xgb_item_hi"] = float(np.percentile(dx, 2.5)), float(np.percentile(dx, 97.5))
                dpx = pbt["ENS"] - pbt["XGB"]
                row["extra_ens_vs_xgb_person_delta"] = ppt["ENS"] - ppt["XGB"]
                row["extra_ens_vs_xgb_person_lo"], row["extra_ens_vs_xgb_person_hi"] = float(np.percentile(dpx, 2.5)), float(np.percentile(dpx, 97.5))
            rows.append(row)
            L.log(cond, c, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)})
    df = pd.DataFrame(rows)
    fam = df[df.model != "LR"].copy()
    fam["item_swap_p_holm"] = E.holm(fam.item_swap_p.to_numpy())
    fam["better"] = (fam.item_delta_lo > 0) & (fam.item_swap_p_holm < 0.05)
    fam["worse"] = (fam.item_delta_hi < 0) & (fam.item_swap_p_holm < 0.05)
    fam["verdict"] = np.where(fam.better, "better", np.where(fam.worse, "worse", "no reliable difference"))
    fam["person_swap_p_holm"] = E.holm(fam.person_swap_p.to_numpy())
    fam["person_verdict"] = np.where((fam.person_delta_lo > 0) & (fam.person_swap_p_holm < 0.05), "better",
                                     np.where((fam.person_delta_hi < 0) & (fam.person_swap_p_holm < 0.05), "worse", "no reliable difference"))
    df = df.merge(fam[["condition", "model", "item_swap_p_holm", "verdict", "person_swap_p_holm", "person_verdict"]], on=["condition", "model"], how="left")
    df.loc[df.model == "LR", ["verdict", "person_verdict"]] = "reference"
    df.to_csv(M.HERE / "results_item_person.csv", index=False)
    (M.HERE / "results_item_person.json").write_text(json.dumps({"n_boot": n_boot, "n_perm": n_perm, "boot_seed": E.BOOT_SEED, "swap_seed": E.SWAP_SEED,
                                                                 "family_size": int(len(fam)), "repeats": reps, "rows": json.loads(df.to_json(orient="records"))}, indent=1))
    L.log("family size", len(fam), df[["condition", "model", "item_auc", "item_delta", "item_delta_lo", "item_delta_hi", "item_swap_p_holm", "verdict"]].to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", default="0")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--nperm", type=int, default=2000)
    a = ap.parse_args()
    evaluate([int(x) for x in a.repeats.split(",")], a.nboot, a.nperm)
