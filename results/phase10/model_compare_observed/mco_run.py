"""Task 2.2c Part B driver (EXPLORATORY). Development data only; the held-out dataset is never read.
Usage (project root; one process, one thread):  .venv/Scripts/python.exe -I -B results/phase10/model_compare_observed/mco_run.py OBS|OBS_R [lr,xgb,ens]
  lr  : logistic regression through the engine, 3 repeats, six group sets, draws 1-3; reproduces the stored 2.2b pooled AUC of every draw
        (OBS: results/phase9/low_rate_observed/checkpoints/sbnp_obsnone_30Hz_d<draw>_oof.csv.gz; OBS_R: the Part A file of the same name in
        sensitivity/checkpoints). A pooled-AUC difference above 0.0005 stops the task (repro_FAILED_<cond>.json, exit 2).
  xgb : XGBoost (grid, tuning, seeds of Task M2), repeat 0 only, as M2 did.
  ens : 50/50 ensemble records from the stored LR and XGBoost fold records (ADDENDUM_ensemble.md), repeat 0.
Every unit (tag x model x set x repeat) is a checkpoint (ck/<tag>/<model>/<set>_r<repeat>.pkl) and is skipped on a restart.
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mco_lib as O  # noqa: E402

import json  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

M, L, RN, EN = O.M, O.L, O.RN, O.EN
sys.stdout.reconfigure(encoding="utf-8")
THRESH = 0.0005


def meta_of(tag, sk):
    return pickle.load(open(RN.CK / tag / f"meta_{sk}.pkl", "rb"))


def lr_stage(cond):
    spec = M.registry()["LR"]
    res = {}
    for d in M.CONDS[cond][1]:
        tag = O.tag_of(cond, d)
        todo = [sk for sk in L.SETS if not all(RN.part_path(tag, "LR", sk, r).exists() for r in (0, 1, 2))]
        cobj = O.build(cond, d) if todo else None
        for sk in todo:
            B = L.delmas_set_bundle(cobj, sk, want=("tc",))
            meta = RN.CK / tag / f"meta_{sk}.pkl"
            if not meta.exists():
                RN.save_part(meta, {"part": B.part, "y": B.y, "image": B.image})
            t = time.time()
            outs = L.run_set(spec, B)
            for r in outs:
                RN.save_part(RN.part_path(tag, "LR", sk, r), {r: outs[r]})
            L.log(f"{tag} LR {sk} 3 repeats done in {time.time() - t:.1f}s")
        frames = []
        for sk in L.SETS:
            m = meta_of(tag, sk)
            outs = {}
            for r in (0, 1, 2):
                outs.update(pickle.load(open(RN.part_path(tag, "LR", sk, r), "rb")))
            frames.append(L.oof_frame(SimpleNamespace(part=m["part"], y=m["y"], image=m["image"], key=sk), outs))
        mine = pd.concat(frames)
        refp = O.LOG_CKPT_DIR[cond] / f"sbnp_obsnone_30Hz_d{d}_oof.csv.gz"
        ref = pd.read_csv(refp)
        keys = ["set", "repeat", "participant", "image", "y"]
        mg = mine.merge(ref, on=keys, suffixes=("", "_ref"))
        auc, aucr = L.LRC.pooled_auc(mine), L.LRC.pooled_auc(ref)
        res[str(d)] = {"stored_file": str(refp), "rows": len(mine), "rows_stored": len(ref), "rows_matched": len(mg),
                       "max_abs_score_diff": float(np.abs(mg.score - mg.score_ref).max()), "pooled_auc_here": auc, "pooled_auc_stored": aucr,
                       "abs_auc_diff": abs(auc - aucr)}
        L.log("REPRO", cond, d, res[str(d)])
        if not (len(mg) == len(mine) == len(ref) and abs(auc - aucr) <= THRESH):
            (M.HERE / f"repro_FAILED_{cond}.json").write_text(json.dumps(res, indent=1))
            L.log("REPRODUCTION FAILED - stopping")
            sys.exit(2)
    res["max_abs_auc_diff_over_draws"] = max(v["abs_auc_diff"] for k, v in res.items() if k in ("1", "2", "3"))
    res["mean_pooled_auc_here"] = float(np.mean([res[k]["pooled_auc_here"] for k in ("1", "2", "3")]))
    res["mean_pooled_auc_stored"] = float(np.mean([res[k]["pooled_auc_stored"] for k in ("1", "2", "3")]))
    res["passed"] = True
    L.LRC.jdump(M.HERE / f"repro_{cond}.json", res)


def xgb_stage(cond):
    assert (M.HERE / f"repro_{cond}.json").exists(), "reproduction first"
    reg = M.registry()
    t_all = time.time()
    for d in M.CONDS[cond][1]:
        tag = O.tag_of(cond, d)
        todo = [sk for sk in L.SETS if not RN.part_path(tag, "XGB", sk, 0).exists()]
        if not todo:
            continue
        cobj = O.build(cond, d)
        for sk in todo:
            B = L.delmas_set_bundle(cobj, sk, want=("tc",))
            t = time.time()
            outs = L.run_set(reg["XGB"], B, repeats=(0,))
            RN.save_part(RN.part_path(tag, "XGB", sk, 0), outs)
            L.log(f"{tag} XGB {sk} r0 done in {time.time() - t:6.1f}s (elapsed {(time.time() - t_all) / 60:.1f} min)")


def ens_stage(cond):
    for d in M.CONDS[cond][1]:
        tag = O.tag_of(cond, d)
        for sk in L.SETS:
            lr = pickle.load(open(RN.part_path(tag, "LR", sk, 0), "rb"))
            xg = pickle.load(open(RN.part_path(tag, "XGB", sk, 0), "rb"))
            RN.save_part(RN.part_path(tag, "ENS", sk, 0), {0: EN.ens_records(lr[0], xg[0])})
        L.log("ensemble records written", tag)


if __name__ == "__main__":
    cond = sys.argv[1]
    assert cond in M.CONDS
    stages = (sys.argv[2] if len(sys.argv) > 2 else "lr,xgb,ens").split(",")
    assert M.HERE == O.HERE and RN.CK == O.CK and L.CK == O.CK
    for s in stages:
        {"lr": lr_stage, "xgb": xgb_stage, "ens": ens_stage}[s](cond)
    L.log("DONE", cond, stages)
