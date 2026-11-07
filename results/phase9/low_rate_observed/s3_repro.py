"""Task 2.2b (EXPLORATORY): reproduction check BEFORE any new condition is run. Stops (exit 2) on a mismatch.

Through THIS folder's code (s2_sim.features_from_grid, s2_sim.table_for, the nested-CV run_grid of the track script on the frozen folds, written into
this folder's checkpoints/) it recomputes two stored values of results/phase9/low_rate (read only; they match results/phase7/trackD/d1_table.csv):
  dwell from samples, no pupil (10 features), clean, 30 Hz            AUC 0.7731838961824132
  dwell from samples, no pupil, measured noise 'anch', 30 Hz, mean of 3 draws   AUC 0.7737061625087726
Input: the 30 Hz arrays of results/phase9/low_rate/cache (anti-aliased resampling of the 1000 Hz recordings); noise: C1.noisy (unchanged).
Tolerance 5e-5 on each (the criterion of s4_webcam.py repro).
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import lro_common as C  # noqa: E402
import s2_sim as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

import lr_common as L  # noqa: E402
import s4_webcam as S4  # noqa: E402

EXP = {"clean": 0.7731838961824132, "anch": 0.7737061625087726}


def main():
    C.CK.mkdir(exist_ok=True)
    track = S4.M.load_track()
    track.CKPT = C.CK
    track.N_JOBS = 1
    frozen = C.json.loads((L.STORED / "folds.json").read_text())["repeats"]
    bk = S4.C1.books()
    segs = S4.load_rate(30)
    out = {"time": C.time.strftime("%Y-%m-%d %H:%M:%S"), "expected": EXP, "criterion_abs_diff": 5e-5}
    for cond, units in (("clean", [("clean", 0.0, 0)]), ("anch", [("anch", "A", d) for d in (1, 2, 3)])):
        aucs = []
        for cnd, Lv, d in units:
            tag = f"REPRO_sbnp_{cnd}_30Hz_d{d}"
            if not (C.CK / f"{tag}.json").exists():
                arr = {}
                for pid, s in segs.items():
                    x, y, p = S4.C1.noisy(s["x"], s["y"], s["p"], cnd, Lv, d, pid)
                    arr[pid] = (S.features_from_grid(x, y, p, 30)["sb"], s["images"], s["group"])
                tab = S.table_for(arr, "sb", 30, bk)
                track.run_grid(tab, frozen, S4.FEATS["sb"], "lr", 30, tag, {}, {})
            aucs.append(L.pooled_auc(C.CK / f"{tag}_oof.csv.gz"))
            C.log(tag, f"{aucs[-1]:.6f}")
        a = float(np.mean(aucs))
        out[cond] = {"auc_here": a, "per_draw": aucs, "abs_diff": abs(a - EXP[cond]), "passed": bool(abs(a - EXP[cond]) < 5e-5)}
    out["all_passed"] = all(out[c]["passed"] for c in EXP)
    C.jdump(C.HERE / "repro_values.json", out)
    C.log("repro", out)
    if not out["all_passed"]:
        C.log("REPRODUCTION FAILED - stopping")
        sys.exit(2)


if __name__ == "__main__":
    main()
