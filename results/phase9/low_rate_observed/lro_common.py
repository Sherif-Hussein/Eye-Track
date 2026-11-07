"""Shared helpers for Task 2.2b (EXPLORATORY): observed-webcam condition for the sampling-rate curve.

Reuses UNCHANGED (imported, nothing copied or edited):
  results/phase9/low_rate/lr_common.py   (paired participant bootstrap, pooled image-level AUC, thread settings)
  results/phase9/low_rate/s4_webcam.py   (loads scripts/phase6_trackC_c1.py as C1: noisy, sb_features_batch, make_table,
                                          feature lists; df_ = edd.delmas_fast; M = edd.delmas_raw_matched)
  app/features.py                        (to_grid: the app's nearest-sample resampling onto the uniform grid)
  app/protocol.py                        (drift_from_frames: read, not imported - it needs pygame; its rule is re-stated in PLAN.md)
Writes only inside results/phase9/low_rate_observed/ (the low_rate cache is read only).
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOWRATE = ROOT / "results" / "phase9" / "low_rate"
APP = ROOT / "app"
for p in (str(LOWRATE), str(APP), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

# EMEET SmartCam C960 sessions only (user, 2026-11-06 addendum): test05, test06, test07, test08, test09
SESSIONS = ["test05_faces_K_emeet_glasses", "test06_faces_I_emeet_glasses", "test07_twofaces_K_dryrun", "test08_v1_18trials", "test09_v1_18trials"]
APPDATA = APP / "data"
MEAS = HERE / "measured"
CACHE1000 = HERE / "cache1000"
CK = HERE / "checkpoints"
PPD_DELMAS = 34.9868554165592
CENTER = np.array([960.0, 540.0])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)
