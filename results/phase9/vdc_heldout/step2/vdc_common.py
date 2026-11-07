"""Shared helpers for Task 2.1 step 2 (held-out recognition analysis on Van der Cruyssen et al. 2024 data).
Import AFTER the thread environment variables have been set by the calling script."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import re
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
STEP1 = ROOT / "results/phase9/vdc_heldout/step1"
STEP2 = ROOT / "results/phase9/vdc_heldout/step2"
RAW = ROOT / "data/external/vdc2024/raw/p562n/osfstorage/Data"
CACHE = Path("<local path>")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STEP1))

EXCL22 = [13, 14, 15, 20, 24, 35, 71, 74, 77, 88, 95, 104, 110, 114, 124, 127, 135, 138, 142, 148, 149, 150]
GROUPS = {"Guilty": "guilty", "Informed": "informed", "Naive": "naive"}
# Our Delmas-pipeline feature set minus the two pupil features (D2), same order as unified.DELMAS_FEATS
FEATS = ["dwell_1", "dwell_2", "n_fix", "n_visit", "dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4", "dwell_b5",
         "first_fix", "lat_first"]
# Same bin edges as src/edd/delmas_raw.BINS (ms); every edge falls on a 100 ms boundary -> 100 ms bin index ranges
BINS_MS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3500), (3500, 5000)]
BIN_IDX = [(a // 100, b // 100) for a, b in BINS_MS]          # [(0,5),(5,10),(10,20),(20,35),(35,50)] half-open
POST_MS = 5000

_FILES = {"log": ("Logboek.xlsx", "Blad1"), "d1": ("FirstPhaseParallel_DwellTime.xlsx", None),
          "d2": ("SecondPhaseParallel_DwellTime.xlsx", None), "fc": ("Parallel_FixationCount.xlsx", None),
          "vi": ("Parallel_Visits.xlsx", None), "sf": ("Single_FixationDuration.xlsx", None),
          "rt": ("RT.xlsx", None), "tc": ("TimeCourse.xlsx", "TimeCourseFinal")}


def pid_from(s):
    return int(re.match(r"\s*(\d+)", str(s)).group(1))


def load(key):
    """Raw table from the cache of parsed files (written by step 1) or, if missing, parsed from the raw .xlsx."""
    name, sheet = _FILES[key]
    sheet = sheet or name.replace(".xlsx", "")
    p = CACHE / f"{name}__{sheet}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    from inspect_vdc import open_xl
    xl = open_xl(str(RAW / name))
    df = xl.parse(xl.sheet_names[0], header=0)
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(df, open(p, "wb"))
    return df


def load_log():
    log = load("log").copy()
    log.columns = ["pid", "Condition", "Date", "Time", "Excl"]
    return log


def load_all():
    """Every table with the numeric participant id in column `pid` (session suffixes stripped, D6b)."""
    T = {"log": load_log()}
    for k in ("d1", "d2", "fc", "vi", "sf"):
        d = load(k).copy()
        d["pid"] = d["Participants"] if k == "d1" else d["RECORDING_SESSION_LABEL"].map(pid_from)
        T[k] = d
    rt = load("rt").copy()
    rt["pid"] = rt["Session_Name"]
    T["rt"] = rt
    tc = load("tc").copy()
    tc["pid"] = tc["RECORDING_SESSION_LABEL"]
    T["tc"] = tc
    return T


def keep_trial_index(T):
    """{pid: set(TRIAL_INDEX)} listed in the participant's response file (D6a); None where the participant has no RT
    trial index (17, 31, 43)."""
    rt = T["rt"]
    out = {}
    for p, g in rt[rt["trial"].notna() & rt["Trial_Index"].notna()].groupby("pid"):
        out[int(p)] = set(g["Trial_Index"].astype(int))
    return out


def drop_practice(df, kti, trial_col="trial"):
    """Keep only the TRIAL_INDEX values in each participant's RT file. For participants without them, of the rows
    labelled trial 64 keep only the highest TRIAL_INDEX (step-1 rule)."""
    keep = np.zeros(len(df), bool)
    pid = df["pid"].to_numpy()
    ti = df["TRIAL_INDEX"].to_numpy()
    tr = df[trial_col].astype(str).to_numpy()
    for p in np.unique(pid):
        m = pid == p
        if p in kti:
            keep[m] = np.isin(ti[m], list(kti[p]))
        else:
            m64 = m & (tr == "64")
            top = ti[m64].max() if m64.any() else None
            keep[m] = ~(tr[m] == "64") | (ti[m] == top)
    return df[keep]


def auc(y, s):
    from edd.unified import fast_auc
    return fast_auc(y, s)


def delong_ci(y, s, alpha=0.05):
    """DeLong (1988) AUC and CI as in pROC::ci.auc (normal approximation on the AUC, clipped to [0,1])."""
    from scipy.stats import norm
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    a = psi.mean()
    v10 = psi.mean(1)
    v01 = psi.mean(0)
    var = v10.var(ddof=1) / m + v01.var(ddof=1) / n
    se = np.sqrt(var)
    z = norm.ppf(1 - alpha / 2)
    return float(a), float(max(0, a - z * se)), float(min(1, a + z * se))
