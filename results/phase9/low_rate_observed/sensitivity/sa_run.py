"""Task 2.2c Part A (EXPLORATORY): the observed-webcam run with the reduced slow jitter component.
The 2.2b code is imported UNCHANGED (lro_common.py, s2_sim.py, s4_run.py from the parent folder). The only things passed in from outside:
  * s2_sim.load_measured is wrapped so that meas["jm"][ax]["params"] are the reduced parameters of jitter_params_reduced.json (sa_params.py);
  * the output locations: lro_common.CK -> sensitivity/checkpoints, s4_run.FEATDIR -> sensitivity/features.
Everything else (frame times, offsets, loss, gains, seeds, folds, models) is the 2.2b code, so the random numbers of every draw are the same as in 2.2b
(the jitter noise is the same standard normals, only scaled differently).
Stages (resumable; one process, one thread):  feat -> features/r<rate>_p<id>.pkl ;  cv -> checkpoints/<route>np_obs<none|app>_<rate>Hz_d<draw>
cv is the loop of s4_run.cv_stage without the draw-1 'regular' pipeline check (the regular-timing run does not depend on the jitter and is not repeated).
Usage: python -I -B sa_run.py feat,cv
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LRO = HERE.parent
sys.path.insert(0, str(LRO))
import lro_common as C  # noqa: E402  (2.2b, unchanged)
import s2_sim as S  # noqa: E402
import s4_run as R  # noqa: E402

C.CK = HERE / "checkpoints"
R.FEATDIR = HERE / "features"
PAR = json.loads((HERE / "jitter_params_reduced.json").read_text(encoding="utf-8"))
_orig = S.load_measured


def load_measured_reduced():
    m = _orig()
    for ax in ("x", "y"):
        m["jm"][ax]["params"] = dict(PAR[ax]["params"])
    return m


S.load_measured = load_measured_reduced


def cv_stage():
    C.CK.mkdir(exist_ok=True)
    track = R.S4.M.load_track()
    track.CKPT = C.CK
    track.N_JOBS = 1
    frozen = C.json.loads((R.L.STORED / "folds.json").read_text())["repeats"]
    bk = R.S4.C1.books()
    for fs in R.RATES:
        feats = None
        for d in R.DRAWS:
            for corr in R.CORRS:
                for route in ("sb", "ivt"):
                    tag = f"{route}np_obs{corr}_{fs}Hz_d{d}"
                    if (C.CK / f"{tag}.json").exists():
                        continue
                    if feats is None:
                        feats = R.load_feats(fs)
                        assert len(feats) == 75
                    arr = {pid: (v["draws"][d][corr][route], v["images"], v["group"]) for pid, v in feats.items()}
                    tab = S.table_for(arr, route, fs, bk)
                    track.run_grid(tab, frozen, R.S4.FEATS[route], "lr", fs, tag, {}, {})
                    C.log(f"{tag}: pooled AUC {R.L.pooled_auc(C.CK / (tag + '_oof.csv.gz')):.4f}")


def main():
    st = (sys.argv[1] if len(sys.argv) > 1 else "feat,cv").split(",")
    assert C.CK == HERE / "checkpoints" and R.FEATDIR == HERE / "features"
    if "feat" in st:
        R.feat_stage()
    if "cv" in st:
        cv_stage()


if __name__ == "__main__":
    main()
