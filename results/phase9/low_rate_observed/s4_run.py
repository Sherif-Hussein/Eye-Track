"""Task 2.2b (EXPLORATORY): run the observed-webcam condition. Stages (each resumable; one process, one thread):
  feat  - for each rate (30, 20, 15, 10) and participant: 1000 Hz arrays -> observed-webcam frames -> features (both routes, both correction
          variants, 3 draws; plus draw-1 'regular' reference) -> features/r<rate>_p<id>.pkl
  cv    - nested CV on the frozen folds (edd.delmas_raw_matched track.run_grid, exactly as results/phase9/low_rate/s4_webcam.py) for every
          (route, correction, rate, draw) -> checkpoints/<tag>.json + <tag>_oof.csv.gz
Tags: <route>np_obs<none|app>_<rate>Hz_d<draw>  (route sb = dwell from samples, 10 features; ivt = I-VT, 11 features; both without pupil)
      <route>np_obsregular_<rate>Hz_d1  (regular frame times, no error: pipeline sanity check)
Usage: python -I s4_run.py feat,cv
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import lro_common as C  # noqa: E402
import s2_sim as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
import pickle  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

import lr_common as L  # noqa: E402
import s4_webcam as S4  # noqa: E402

RATES = [30, 20, 15, 10]
DRAWS = [1, 2, 3]
CORRS = ["none", "app"]
FEATDIR = C.HERE / "features"


def feat_stage():
    FEATDIR.mkdir(exist_ok=True)
    meas = S.load_measured()
    pids = sorted(int(f.stem[1:]) for f in C.CACHE1000.glob("p??.npz"))
    assert len(pids) == 75, len(pids)
    for fs in RATES:
        t0 = C.time.time()
        for k, pid in enumerate(pids):
            path = FEATDIR / f"r{fs}_p{pid:02d}.pkl"
            if path.exists():
                continue
            z = np.load(C.CACHE1000 / f"p{pid:02d}.npz")
            pre = S.prepare(z["x"].astype(float), z["y"].astype(float), z["p"].astype(float), fs)
            out = {"images": z["images"], "group": int(z["group"]), "draws": {}}
            for d in DRAWS:
                out["draws"][d] = S.simulate_participant(pid, z["x"].astype(float), z["y"].astype(float), z["p"].astype(float), fs, d, meas, pre=pre)
            tmp = FEATDIR / f"r{fs}_p{pid:02d}.tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(out, fh)
            tmp.replace(path)
            C.log(f"rate {fs} participant {pid} [{k + 1}/{len(pids)}] ({C.time.time() - t0:.0f}s)")


def load_feats(fs):
    return {int(f.stem.split("_p")[1]): pickle.load(open(f, "rb")) for f in sorted(FEATDIR.glob(f"r{fs}_p??.pkl"))}


def cv_stage():
    C.CK.mkdir(exist_ok=True)
    track = S4.M.load_track()
    track.CKPT = C.CK
    track.N_JOBS = 1
    frozen = C.json.loads((L.STORED / "folds.json").read_text())["repeats"]
    bk = S4.C1.books()
    for fs in RATES:
        feats = None
        units = [(d, c) for d in DRAWS for c in CORRS] + [(1, "regular")]
        for d, corr in units:
            for route in ("sb", "ivt"):
                tag = f"{route}np_obs{corr}_{fs}Hz_d{d}"
                if (C.CK / f"{tag}.json").exists():
                    continue
                if feats is None:
                    feats = load_feats(fs)
                    assert len(feats) == 75
                arr = {pid: (v["draws"][d][corr][route], v["images"], v["group"]) for pid, v in feats.items()}
                tab = S.table_for(arr, route, fs, bk)
                track.run_grid(tab, frozen, S4.FEATS[route], "lr", fs, tag, {}, {})
                C.log(f"{tag}: pooled AUC {L.pooled_auc(C.CK / (tag + '_oof.csv.gz')):.4f}")


def main():
    st = (sys.argv[1] if len(sys.argv) > 1 else "feat,cv").split(",")
    if "feat" in st:
        feat_stage()
    if "cv" in st:
        cv_stage()


if __name__ == "__main__":
    main()
