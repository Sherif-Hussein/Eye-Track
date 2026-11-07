"""Follow-up to Task 2.2c (docs/task_observed_followup_catboost.md), LABELLED EXTRA, EXPLORATORY. Development data only; no camera; held-out data not read.
CatBoost with the grid, tuning and seed rule of results/phase10/model_compare/PLAN.md (mc_lib.GBSpec kind 'cat': iterations {100, 300} x depth {2, 4} x
learning rate {0.03, 0.1}, auto_class_weights Balanced, thread_count=1 for fit AND predict (mc_lib.raw passes thread_count=1), random_seed = repeat, every
setting fitted directly), on the same OBS and OBS_R features, draws 1-3, repeat 0; then the 50/50 ensemble LR + CatBoost with mc_ens.ens_records (same scaling).
Engine and helpers imported unchanged through ../mco_lib.py; every write is redirected into this folder (catboost_extra/ck). The existing logistic regression
fold records of the parent folder are read only.
Usage: .venv/Scripts/python.exe -I -B results/phase10/model_compare_observed/catboost_extra/cx_run.py OBS|OBS_R [cat,ens]
Units (tag x model x set, repeat 0) are checkpoints and are skipped on a restart.
"""
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mco_lib as O  # noqa: E402

import pickle  # noqa: E402

M, L, RN, EN, EV = O.M, O.L, O.RN, O.EN, O.EV
sys.stdout.reconfigure(encoding="utf-8")
CK = HERE / "ck"
MAIN_CK = O.CK
M.HERE = HERE
L.OUT = HERE
L.CK = CK
RN.CK = CK
EV.CK = CK
EN.CK = CK


def cat_stage(cond):
    reg = M.registry()
    t_all = time.time()
    for d in M.CONDS[cond][1]:
        tag = O.tag_of(cond, d)
        todo = [sk for sk in L.SETS if not RN.part_path(tag, "CAT", sk, 0).exists()]
        if not todo:
            continue
        cobj = O.build(cond, d)
        for sk in todo:
            B = L.delmas_set_bundle(cobj, sk, want=("tc",))
            meta = CK / tag / f"meta_{sk}.pkl"
            if not meta.exists():
                RN.save_part(meta, {"part": B.part, "y": B.y, "image": B.image})
            t = time.time()
            outs = L.run_set(reg["CAT"], B, repeats=(0,))
            RN.save_part(RN.part_path(tag, "CAT", sk, 0), outs)
            L.log(f"{tag} CAT {sk} r0 done in {time.time() - t:6.1f}s (elapsed {(time.time() - t_all) / 60:.1f} min)")


def ens_stage(cond):
    for d in M.CONDS[cond][1]:
        tag = O.tag_of(cond, d)
        for sk in L.SETS:
            lr = pickle.load(open(MAIN_CK / tag / "LR" / f"{sk}_r0.pkl", "rb"))
            ct = pickle.load(open(RN.part_path(tag, "CAT", sk, 0), "rb"))
            RN.save_part(RN.part_path(tag, "ENSC", sk, 0), {0: EN.ens_records(lr[0], ct[0])})
        L.log("LR+CatBoost ensemble records written", tag)


if __name__ == "__main__":
    cond = sys.argv[1]
    assert cond in M.CONDS
    assert RN.CK == CK and L.CK == CK and M.HERE == HERE and CK != MAIN_CK
    for s in (sys.argv[2] if len(sys.argv) > 2 else "cat,ens").split(","):
        {"cat": cat_stage, "ens": ens_stage}[s](cond)
    L.log("DONE", cond)
