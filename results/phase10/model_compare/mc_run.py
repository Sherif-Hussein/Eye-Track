"""Task M2 driver. EXPLORATORY. Development data only; the held-out dataset is never read.
Source: results/phase10/model_search/ms_lib.py (engine: bundles, frozen folds, run_set/run_fold, oof_frame), reproduction logic of ms_run.py repro.
Usage (project root, one process, one thread):
  .venv/Scripts/python.exe -I results/phase10/model_compare/mc_run.py all [--repeats 0] [--models XGB,LGBM,CAT,SVM]
Stage 1 reproduces the frozen logistic regression (3 repeats, all six sets) through the engine and stops on a mismatch (writes repro_*.json here).
Stage 2 runs each model x condition x draw x group set x repeat; every finished unit is a checkpoint (ck/<tag>/<model>/<set>_r<repeat>.pkl) and is skipped on resume.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_lib as M  # noqa: E402

L = M.L
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CK = M.HERE / "ck"
REF_FULL, REF_CAMERA_MEAN = 0.7854480364491726, 0.7737


def tag_of(cond, draw):
    return M.CONDS[cond][0] if draw is None else f"{M.CONDS[cond][0]}_d{draw}"


def part_path(tag, model, sk, r):
    d = CK / tag / model
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sk}_r{r}.pkl"


def save_part(path, obj):
    tmp = Path(str(path) + ".tmp")
    pickle.dump(obj, open(tmp, "wb"))
    tmp.replace(path)


def build(cond, draw):
    t = time.time()
    cobj = L.DelmasCond(M.CONDS[cond][0], draw=draw)
    L.log(f"built {tag_of(cond, draw)} in {time.time() - t:.1f}s")
    return cobj


def repro_and_reference(cobjs):
    """LR through the engine, 3 repeats, all six sets; compare with the stored scores. Saves the LR checkpoints (all 3 repeats)."""
    spec = M.registry()["LR"]
    pooled = {}
    for (cond, draw), cobj in cobjs.items():
        frames = []
        for sk in L.SETS:
            B = L.delmas_set_bundle(cobj, sk, want=("tc",))
            tag = tag_of(cond, draw)
            outs = L.run_set(spec, B)
            for r in outs:
                save_part(part_path(tag, "LR", sk, r), {r: outs[r]})
            frames.append(L.oof_frame(B, outs))
        mine = pd.concat(frames)
        keys = ["set", "repeat", "participant", "image", "y"]
        if cond == "FULL":
            ref = pd.read_csv(L.LRC.STORED / "checkpoints" / "ALLz_lr_30_oof.csv.gz")
        else:
            ref = pd.read_csv(L.ROOT / "results" / "phase9" / "low_rate" / "checkpoints_webcam" / f"sbnp_anch_30Hz_d{draw}_oof.csv.gz")
        m = mine.merge(ref, on=keys, suffixes=("", "_ref"))
        md = float(np.abs(m.score - m.score_ref).max())
        auc = L.LRC.pooled_auc(mine)
        pooled[(cond, draw)] = {"rows": len(mine), "rows_matched": len(m), "max_abs_score_diff": md, "pooled_auc": auc, "stored_pooled_auc": L.LRC.pooled_auc(ref)}
        L.log("REPRO", cond, draw, pooled[(cond, draw)])
        ok = len(m) == len(mine) == len(ref) and md < 1e-9
        if cond == "FULL":
            ok = ok and abs(auc - REF_FULL) < 1e-9
        if not ok:
            L.log("REPRODUCTION FAILED - stopping")
            (M.HERE / "repro_FAILED.json").write_text(str(pooled))
            sys.exit(2)
    cam = [v["pooled_auc"] for (c, d), v in pooled.items() if c == "CAMERA"]
    res = {"FULL": pooled[("FULL", None)], "CAMERA_draws": {str(d): v for (c, d), v in pooled.items() if c == "CAMERA"},
           "CAMERA_mean_pooled_auc": float(np.mean(cam)), "stored_CAMERA_mean": REF_CAMERA_MEAN, "passed": bool(abs(np.mean(cam) - REF_CAMERA_MEAN) < 5e-4)}
    L.LRC.jdump(M.HERE / "repro.json", res)
    L.log("REPRO summary", res["CAMERA_mean_pooled_auc"], res["passed"])
    if not res["passed"]:
        L.log("CAMERA reproduction FAILED - stopping")
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--repeats", default="0")
    ap.add_argument("--models", default="XGB,LGBM,CAT,SVM")
    a = ap.parse_args()
    reps = [int(x) for x in a.repeats.split(",")]
    models = a.models.split(",")
    reg = M.registry()
    t_all = time.time()
    keys = [(c, d) for c in M.CONDS for d in M.CONDS[c][1]]
    if a.stage == "all" and not (M.HERE / "repro.json").exists():
        cobjs = {k: build(*k) for k in keys[:1]}
        # build and check one condition at a time to bound memory (CAMERA draws are built again below if needed)
        cobjs = {}
        for k in keys:
            cobjs[k] = build(*k)
        repro_and_reference(cobjs)
    else:
        cobjs = {}
    for k in keys:
        cond, draw = k
        tag = tag_of(cond, draw)
        cobj = cobjs.get(k) or build(*k)
        for sk in L.SETS:
            todo = [(m, r) for m in models for r in reps if not part_path(tag, m, sk, r).exists()]
            if not todo:
                continue
            B = L.delmas_set_bundle(cobj, sk, want=("tc",))
            meta = CK / tag / f"meta_{sk}.pkl"
            if not meta.exists():
                save_part(meta, {"part": B.part, "y": B.y, "image": B.image})
            for m, r in todo:
                t = time.time()
                outs = L.run_set(reg[m], B, repeats=(r,))
                save_part(part_path(tag, m, sk, r), outs)
                L.log(f"{tag} {m:5s} {sk} r{r} done in {time.time() - t:6.1f}s (elapsed {(time.time() - t_all) / 60:.1f} min)")
        if not (cond == "CAMERA" and False):
            cobjs.pop(k, None)
    L.log("ALL DONE", f"{(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
