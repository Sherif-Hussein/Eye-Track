"""Task 2.2c Part B (EXPLORATORY): library. Logistic regression, XGBoost and the 50/50 ensemble on the observed-webcam data. Development data only.
Imported UNCHANGED: results/phase10/model_compare/mc_lib.py, mc_run.py (part_path, save_part), mc_ens.py (ens_records), mc_eval.py (evaluate),
results/phase10/model_search/ms_lib.py and ms_eval.py (through mc_lib), results/phase9/low_rate_observed/s2_sim.py (table_for) and lro_common.py.
What is passed in from outside: the output folder (mc_lib.HERE, ms_lib.OUT, ms_lib.CK, mc_run.CK, mc_eval.CK -> this folder), the condition list (mc_lib.CONDS),
the model list (mc_lib.MODELS and mc_eval.MODELS = LR, XGB, ENS) and a condition class that reads the stored 2.2b-style feature files (ObsCond below).
ObsCond is a subclass of ms_lib.DelmasCond whose constructor replaces the simulation of 'anch' noise by a read of the feature files of Part A / Task 2.2b:
the image-level table is built by the 2.2b function (s2_sim.table_for = C1.make_table, the same call as in s4_run.cv_stage), the trial-level base features are
the same feature array. The time-course and scan-path blocks of the engine (tc, scan) need the sample grids, which the feature files do not store; they are
filled with zeros. They are used only by candidate models that read them (PCA, catch22, scan-path); logistic regression and XGBoost here read the 'base'
block only, so nothing changes for them.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MC = ROOT / "results" / "phase10" / "model_compare"
LRO = ROOT / "results" / "phase9" / "low_rate_observed"
sys.path.insert(0, str(MC))
import mc_lib as M  # noqa: E402  (sets thread env vars first, then imports ms_lib)

L = M.L
sys.path.insert(0, str(LRO))
import lro_common as C  # noqa: E402
import s2_sim as S  # noqa: E402

import pickle  # noqa: E402

import numpy as np  # noqa: E402

import mc_ens as EN  # noqa: E402
import mc_eval as EV  # noqa: E402
import mc_run as RN  # noqa: E402

# ---------------------------------------------------------------- redirect every write to this folder
CK = HERE / "ck"
M.HERE = HERE
L.OUT = HERE
L.CK = CK
RN.CK = CK
EV.CK = CK
EN.CK = CK
M.MODELS = ["LR", "XGB"]
EV.MODELS = ["LR", "XGB", "ENS"]
M.CONDS = {"OBS": ("OBS", [1, 2, 3]), "OBS_R": ("OBSR", [1, 2, 3])}
M.LABEL["ENS"] = "Ensemble (logistic regression + XGBoost)"
FEAT_DIR = {"OBS": LRO / "features", "OBS_R": LRO / "sensitivity" / "features"}
LOG_CKPT_DIR = {"OBS": LRO / "checkpoints", "OBS_R": LRO / "sensitivity" / "checkpoints"}


class ObsCond(L.DelmasCond):
    def __init__(self, cond, draw, log_=L.log):
        self.name, self.draw = M.CONDS[cond][0], draw
        self.fs = self.rate = 30
        self.trials = {}
        fdir = FEAT_DIR[cond]
        C1 = S.S4.C1
        C1.FS = 30
        nonp = [n for n in C1.SB_NAMES if n not in ("pupil_face", "pupil_trial", "n_seg")]
        assert len(nonp) == 10
        cols = [C1.SB_NAMES.index(n) for n in nonp]
        pg = self._pids()
        pids = sorted(pg)
        arrays = {}
        for pid in pids:
            with open(fdir / f"r30_p{pid:02d}.pkl", "rb") as fh:
                v = pickle.load(fh)
            F = v["draws"][draw]["none"]["sb"]
            T = len(v["images"])
            assert F.shape[:2] == (T, 4) and int(v["group"]) == pg[pid]
            arrays[pid] = (F, v["images"], int(v["group"]))
            self.trials[pid] = {"images": v["images"], "group": int(v["group"]), "base": F[:, :, cols],
                                "tc": np.zeros((T, 4, L.NBIN)), "scan": np.zeros((T, 4, len(L.SCAN_NAMES)))}
        assert len(arrays) == 75 and len(list(fdir.glob("r30_p??.pkl"))) == 75
        self.C1 = C1
        self.df = S.table_for(arrays, "sb", 30, C1.books())
        self.feats = [f"{n}_z" for n in nonp]
        self.base_names = nonp


def tag_of(cond, draw):
    return f"{M.CONDS[cond][0]}_d{draw}"


def build(cond, draw):
    import time
    t = time.time()
    c = ObsCond(cond, draw)
    L.log(f"built {tag_of(cond, draw)} in {time.time() - t:.1f}s")
    return c
