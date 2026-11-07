"""Step 2.1 - cleaning (D6) and label-free item features (D2, D3). Writes features.csv and displays_per_participant.csv.
No group label is used here. Run: .venv\\Scripts\\python.exe -I s1_features.py"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from vdc_common import *

T = load_all()
log = T["log"]
keep = sorted(log.pid[log.Excl != 1].tolist())
assert len(keep) == 133, len(keep)                      # D6c: authors' 22 exclusions
assert set(log.pid[log.Excl == 1]) == set(EXCL22)
kti = keep_trial_index(T)
print("participants kept:", len(keep), "| with RT trial index:", len(kti & set(keep) if False else set(kti) & set(keep)))

# ---- clean every four-item table
def clean(d):
    d = d[d.pid.isin(keep)].copy()
    d = drop_practice(d, kti)
    d["criticalplace"] = pd.to_numeric(d["criticalplace"], errors="coerce")
    d["TRIAL_INDEX"] = d["TRIAL_INDEX"].astype(int)
    return d

d1, d2, fc, vi = (clean(T[k]) for k in ("d1", "d2", "fc", "vi"))
tc = T["tc"]
tc = tc[tc.pid.isin(keep)].copy()
tc = drop_practice(tc, kti)

# ---- displays that contain a crime-related item (D3)
key = ["pid", "TRIAL_INDEX", "IA_ID"]
crit = lambda d: d[d.criticalplace > 0]
d1c, d2c, fcc = crit(d1), crit(d2), crit(fc)
print("rows after cleaning (critical-present displays x 4 areas): d1", len(d1c), "d2", len(d2c), "fc", len(fcc))

base = d1c[key + ["criticalplace", "trial", "IA_DWELL_TIME"]].rename(columns={"IA_DWELL_TIME": "dwell_1"})
base = base.merge(d2c[key + ["IA_DWELL_TIME"]].rename(columns={"IA_DWELL_TIME": "dwell_2"}), on=key, how="inner")
base = base.merge(fcc[key + ["IA_FIXATION_COUNT"]].rename(columns={"IA_FIXATION_COUNT": "n_fix"}), on=key, how="inner")

# ---- visits and first fixation from the fixation sequence (rule: run of same-area fixations; '.' breaks the run)
vic = crit(vi)
vrows = []
for (p, ti), g in vic.groupby(["pid", "TRIAL_INDEX"], sort=False):
    ids = pd.to_numeric(g["CURRENT_FIX_INTEREST_AREA_ID"], errors="coerce").fillna(-1).astype(int).to_numpy()
    vis = np.zeros(5)
    prev = -2
    for a in ids:
        if a >= 1 and a != prev:
            vis[a] += 1
        prev = a
    first = next((a for a in ids if a >= 1), -1)
    for k in range(1, 5):
        vrows.append((p, ti, k, vis[k], float(first == k)))
vdf = pd.DataFrame(vrows, columns=key + ["n_visit", "first_fix"])
base = base.merge(vdf, on=key, how="left")
no_vis = base.n_visit.isna()
base.loc[no_vis, ["n_visit", "first_fix"]] = 0.0      # a display with no fixation rows has no visits (zero)
print("displays with no rows in the Visits file (set to zero visits):", int(no_vis.sum() / 4))

# ---- time-course features: mean share over the same bin edges as the Delmas features; latency = start of first bin with share > 0
tcc = tc[tc.criticalplace > 0].sort_values(["pid", "TRIAL_INDEX", "BIN_INDEX"])
rows = []
for (p, ti), g in tcc.groupby(["pid", "TRIAL_INDEX"], sort=False):
    if len(g) != 50:
        continue
    A = g[[f"AVERAGE_IA_{k}_SAMPLE_COUNT_%" for k in (1, 2, 3, 4)]].to_numpy(float)       # (50, 4)
    for k in range(4):
        v = A[:, k]
        nz = np.flatnonzero(v > 0)
        lat = float(g.BIN_INDEX.to_numpy()[nz[0]] * 100) if len(nz) else float(POST_MS)
        rows.append([p, ti, k + 1, *[v[a:b].mean() for a, b in BIN_IDX], lat])
tdf = pd.DataFrame(rows, columns=key + ["dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4", "dwell_b5", "lat_first"])
base = base.merge(tdf, on=key, how="inner")

# ---- valid displays per participant (all five sources present for the display)
nd = base.groupby("pid").apply(lambda g: g.groupby("TRIAL_INDEX").ngroups)
nd = nd.reindex(keep).fillna(0).astype(int)
print("valid critical-present displays per participant:", nd.value_counts().sort_index().to_dict())
print("participants with < 32:", nd[nd < 32].to_dict())
excluded_pids = nd[nd < 8].index.tolist()
print("participants with < 8 valid displays (excluded by D6d):", excluded_pids)
base = base[~base.pid.isin(excluded_pids)]
bad = base.groupby(["pid", "TRIAL_INDEX"]).size()
assert (bad == 4).all()

base["item"] = base.IA_ID
base["is_crime"] = (base.IA_ID == base.criticalplace).astype(int)
out = base[["pid", "TRIAL_INDEX", "trial", "item", "criticalplace", "is_crime"] + FEATS].rename(
    columns={"pid": "participant", "TRIAL_INDEX": "trial_index", "trial": "stimulus_trial", "item": "position",
             "is_crime": "item_label"}).sort_values(["participant", "trial_index", "position"])
out.to_csv(STEP2 / "features.csv", index=False)
nd.rename("n_valid_displays").to_csv(STEP2 / "displays_per_participant.csv")
print("features.csv:", out.shape, "participants:", out.participant.nunique())
print("item_label share:", out.item_label.mean())
print("missing values per feature:", out[FEATS].isna().sum().to_dict())
print(out[FEATS].describe().T[["mean", "std", "min", "max"]])
