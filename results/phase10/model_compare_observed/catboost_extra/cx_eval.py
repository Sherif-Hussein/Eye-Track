"""Follow-up to Task 2.2c (LABELLED EXTRA, EXPLORATORY): evaluation of CatBoost and LR + CatBoost beside LR, XGBoost and LR + XGBoost.
Uses ms_eval's functions unchanged (through mco_lib / mc_eval): item_frame, person_frame, item_metrics, person_metrics, to_sets, boot, point, swap_p, with the
parent's seeds (bootstrap 20261106 / +1, swap 20261107 / +1, 2000 each), so the LR, XGBoost and LR + XGBoost rows must reproduce the parent's results_table.csv (checked).
Reads the parent's merged fold records of LR, XGB, ENS (copied into catboost_extra/ck) and this folder's CAT and ENSC parts. No Holm and no verdict: not in the family.
Writes results_cat_extra.csv/json, cx_repro_parent.json, tables_cat.md here.
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mco_lib as O  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

M, L, RN, EV = O.M, O.L, O.RN, O.EV
E = EV.E
sys.stdout.reconfigure(encoding="utf-8")
CK = HERE / "ck"
MAIN_CK = O.CK
M.HERE = HERE
L.OUT = HERE
L.CK = CK
RN.CK = CK
EV.CK = CK
assert CK != MAIN_CK
MODELS = ["LR", "XGB", "ENS", "CAT", "ENSC"]
NAME = {"LR": "Logistic regression (reference)", "XGB": "XGBoost", "ENS": "LR + XGBoost", "CAT": "CatBoost", "ENSC": "LR + CatBoost"}
TAGS = {"OBS": [f"OBS_d{d}" for d in (1, 2, 3)], "OBS_R": [f"OBSR_d{d}" for d in (1, 2, 3)]}


def prepare():
    for tags in TAGS.values():
        for t in tags:
            (CK / t).mkdir(parents=True, exist_ok=True)
            for sk in L.SETS:
                shutil.copy2(MAIN_CK / t / f"meta_{sk}.pkl", CK / t / f"meta_{sk}.pkl") if not (CK / t / f"meta_{sk}.pkl").exists() else None
                for m in ("LR", "XGB", "ENS"):
                    dst = L.ck_path(t, m, sk)
                    if not dst.exists():
                        d = pickle.load(open(MAIN_CK / t / m / f"{sk}.pkl", "rb"))
                        pickle.dump({0: d[0]}, open(dst, "wb"))
                for m in ("CAT", "ENSC"):
                    d = pickle.load(open(RN.part_path(t, m, sk, 0), "rb"))
                    pickle.dump(d, open(L.ck_path(t, m, sk), "wb"))


def main():
    prepare()
    reps, sets = [0], L.SETS
    rows = []
    for cond, ts in TAGS.items():
        frames = {c: [E.item_frame(t, c, sets, reps) for t in ts] for c in MODELS}
        sdb, owner, names = E.to_sets(frames, sets)
        pt = E.point(sdb, owner, names)
        bt = E.boot(sdb, owner, names, 2000, E.BOOT_SEED)
        pfr, pers = {}, {}
        for c in MODELS:
            tabs = [E.person_frame(t, c, sets, "mean", reps) for t in ts]
            pfr[c], pers[c] = [x[1] for x in tabs], [x[0] for x in tabs]
        psd, powner, pnames = E.to_sets(pfr, sets)
        ppt = E.point(psd, powner, pnames)
        pbt = E.boot(psd, powner, pnames, 2000, E.BOOT_SEED + 1)
        ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
        for c in MODELS:
            r = {"condition": cond, "model": c, "label": NAME[c]}
            im = [E.item_metrics(t, c, sets, reps) for t in ts]
            r.update({k: float(np.mean([m[k] for m in im])) for k in im[0]})
            r["item_auc"] = pt[c]
            r["item_auc_lo"], r["item_auc_hi"] = ci(bt[c])
            ms = [E.person_metrics(t) for t in pers[c]]
            r.update({k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
            r["person_auc"] = ppt[c]
            r["person_auc_lo"], r["person_auc_hi"] = ci(pbt[c])
            for a, b, key in (("CAT", "LR", "cat_vs_lr"), ("ENSC", "LR", "ensc_vs_lr"), ("ENSC", "ENS", "ensc_vs_ens"), ("ENSC", "CAT", "ensc_vs_cat"),
                              ("ENS", "LR", "ens_vs_lr"), ("XGB", "LR", "xgb_vs_lr")):
                if c == a:
                    r[f"item_{key}"] = pt[a] - pt[b]
                    r[f"item_{key}_lo"], r[f"item_{key}_hi"] = ci(bt[a] - bt[b])
                    r[f"person_{key}"] = ppt[a] - ppt[b]
                    r[f"person_{key}_lo"], r[f"person_{key}_hi"] = ci(pbt[a] - pbt[b])
                    if key in ("cat_vs_lr", "ensc_vs_lr", "ensc_vs_ens"):
                        r[f"item_{key}_swap_p_raw"] = E.swap_p(sdb, owner, b, a, 2000, E.SWAP_SEED)[1]
                        r[f"person_{key}_swap_p_raw"] = E.swap_p(psd, powner, b, a, 2000, E.SWAP_SEED + 1)[1]
            rows.append(r)
            L.log(cond, c, round(r["item_auc"], 4), round(r["person_auc"], 4))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results_cat_extra.csv", index=False)
    (HERE / "results_cat_extra.json").write_text(json.dumps(json.loads(df.to_json(orient="records")), indent=1))
    # the parent's rows must be reproduced
    par = pd.read_csv(HERE.parent / "results_table.csv")
    mx = 0.0
    for _, p in par.iterrows():
        r = df[(df.condition == p.condition) & (df.model == p.model)].iloc[0]
        for c in ("item_auc", "item_auc_lo", "item_auc_hi", "person_auc", "person_auc_lo", "person_auc_hi", "item_eer", "item_hter_fpr7", "person_eer", "person_hter_fpr7"):
            mx = max(mx, abs(r[c] - p[c]))
    (HERE / "cx_repro_parent.json").write_text(json.dumps({"max_abs_difference_parent_rows": mx, "passed": bool(mx < 1e-9)}, indent=1))
    L.log("parent rows reproduced, max abs difference", mx)
    tables(df)


def tables(df):
    f = lambda v: f"{v:.4f}"  # noqa: E731
    sg = lambda v: f"{v:+.4f}"  # noqa: E731
    out = []
    for lvl, nm in (("item", "Item level"), ("person", "Person level")):
        out.append(f"\n**{nm}**\n\n| Condition | Model | AUC [95% CI] | EER | HTER (7% threshold) |\n|---|---|---|---|---|")
        for _, r in df.iterrows():
            out.append(f"| {r.condition} | {r.label} | {f(r[lvl + '_auc'])} [{f(r[lvl + '_auc_lo'])}, {f(r[lvl + '_auc_hi'])}] | {r[lvl + '_eer']:.3f} | {r[lvl + '_hter_fpr7']:.3f} |")
        out.append(f"\n**{nm}: paired differences [95% CI] (labelled extra, raw swap p, no Holm, no verdict)**\n\n| Condition | Contrast | Difference [95% CI] | Raw swap p |\n|---|---|---|---|")
        for cond in ("OBS", "OBS_R"):
            for key, lab, who in (("cat_vs_lr", "CatBoost - LR", "CAT"), ("ensc_vs_lr", "LR + CatBoost - LR", "ENSC"), ("ensc_vs_ens", "LR + CatBoost - LR + XGBoost", "ENSC"), ("ensc_vs_cat", "LR + CatBoost - CatBoost (extra)", "ENSC")):
                r = df[(df.condition == cond) & (df.model == who)].iloc[0]
                pcol = f"{lvl}_{key}_swap_p_raw"
                pv = f"{r[pcol]:.4f}" if pcol in r and pd.notna(r[pcol]) else "--"
                out.append(f"| {cond} | {lab} | {sg(r[f'{lvl}_{key}'])} [{sg(r[f'{lvl}_{key}_lo'])}, {sg(r[f'{lvl}_{key}_hi'])}] | {pv} |")
    (HERE / "tables_cat.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
