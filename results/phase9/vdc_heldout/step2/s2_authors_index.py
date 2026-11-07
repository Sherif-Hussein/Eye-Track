"""Step 2.1 - the authors' equal-weight index (Van der Cruyssen et al. 2024), following the deposited R script
GuiltyNaive.R measure by measure, and the reproduction check against the published AUC .71 [.60, .82]
(guilty vs naive). Writes authors_index.csv and authors_index_check.json."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from vdc_common import *

T = load_all()
log = T["log"]
part = log[log.Excl != 1]
informed = set(part.pid[part.Condition == "Informed"])
final = part[part.Condition != "Informed"].sort_values("pid").reset_index(drop=True)
print("Final (guilty + naive):", final.Condition.value_counts().to_dict())


def zs(x):
    """R: (x - mean(x, na.rm)) / sd(x, na.rm); sd with n-1."""
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=1)


def mean_z_by_pid(df, valcol, crit_mask, sign):
    """z over each participant's rows, times sign, then the mean z over the crime-related rows (na.rm)."""
    out = {}
    for p, g in df.groupby("pid"):
        z = zs(g[valcol].to_numpy(float)) * sign
        m = crit_mask(g).to_numpy()
        out[p] = np.nanmean(z[m]) if m.any() else np.nan
    return pd.Series(out)


EX = set(EXCL22)
skip = EX | informed

# 1. RT (single display): z over the 64 rows, sign -1, crime-related = fourorcenter 2 or 3; 17, 31, 43 removed
rt = T["rt"]
rt = rt[~rt.pid.isin(EX | {17, 31, 43} | informed)]
rt = rt[rt.trial.notna()]
z_rt = mean_z_by_pid(rt, "RT_press", lambda g: g.fourorcenter.isin([2, 3]), -1.0)

# 2. mean fixation duration (single display): drop trial 64 and '.' fixations, z over fixations, sign +1
sf = T["sf"]
sf = sf[~sf.pid.isin(skip)]
sf = sf[sf.trial.astype(str) != "64"]
sf = sf[sf.CURRENT_FIX_INTEREST_AREA_LABEL != "."]
z_fd = mean_z_by_pid(sf, "CURRENT_FIX_DURATION", lambda g: g.fourorcenter.isin([2, 3]), +1.0)

# 3-5. four-item display: dwell 1 (+1), dwell 2 (-1), fixation count (-1); rows with criticalplace != 0 only
def area_measure(df, valcol, sign):
    df = df[~df.pid.isin(skip)].copy()
    df["criticalplace"] = pd.to_numeric(df["criticalplace"], errors="coerce")
    df = df[df.criticalplace != 0]
    return mean_z_by_pid(df, valcol, lambda g: g.IA_ID == g.criticalplace, sign)

z_d1 = area_measure(T["d1"], "IA_DWELL_TIME", +1.0)
z_d2 = area_measure(T["d2"], "IA_DWELL_TIME", -1.0)
z_fc = area_measure(T["fc"], "IA_FIXATION_COUNT", -1.0)

# 6. number of visits, replicating the R code: ID set to NA for '.' fixations and for displays without a crime-related
#    item; a row is a 'duplicate' if every column equals the previous row (an NA ID is ignored in that comparison);
#    duplicates dropped; visits counted per (participant, trial, correct), foil counts / 3, z over the rows, sign -1.
vi = T["vi"]
vi = vi[~vi.pid.isin(skip)].sort_values("pid", kind="stable").reset_index(drop=True)
idc = "CURRENT_FIX_INTEREST_AREA_ID"
vid = vi[idc].astype(str)
vid = vid.where(~((vi.CURRENT_FIX_INTEREST_AREA_LABEL == ".") | (vi.criticalplace == 0)), np.nan)
cols = [c for c in vi.columns if c not in ("pid", idc)]
same = np.ones(len(vi), bool)
for c in cols:
    a = vi[c].to_numpy()
    same &= np.concatenate([[False], a[1:] == a[:-1]])
va, vb = vid.to_numpy(object), np.concatenate([[None], vid.to_numpy(object)[:-1]])
na_either = pd.isna(va) | pd.isna(pd.Series(vb))
id_eq = np.where(na_either, True, va == vb)
dup = same & id_eq
dup[0] = False
vv = vi[~dup].copy()
vv["id"] = vid[~dup]
vv["correct"] = np.where(vv["id"].isna(), np.nan, (vv["id"].astype(str) == vv.criticalplace.astype(str)).astype(float))
res = vv.groupby(["pid", "trial", "correct"], dropna=False).size().rename("visits").reset_index()
res = res[res.correct.notna()].copy()
res["visits"] = res["visits"].astype(float)
res.loc[res.correct == 0, "visits"] = res.loc[res.correct == 0, "visits"] / 3
z_vis = mean_z_by_pid(res, "visits", lambda g: g.correct == 1, -1.0)

M = pd.DataFrame({"zmean_RT": z_rt, "zmean_FixationDuration": z_fd, "zmean_DwellTimeFirstPhase": z_d1,
                  "zmean_DwellTimeSecondPhase": z_d2, "zmean_Visits": z_vis, "zmean_FixationCount": z_fc})
M = M.reindex(final.pid)
M.index.name = "participant"
M["W_mean"] = M.iloc[:, :6].mean(axis=1, skipna=True)
M.to_csv(STEP2 / "authors_index.csv")

y = (final.Condition == "Guilty").astype(int).to_numpy()
w = M.W_mean.to_numpy()
a, lo, hi = delong_ci(y, w)
print(f"\nAuthors' index, guilty (n={y.sum()}) vs naive (n={(1 - y).sum()}): AUC = {a:.4f}  DeLong 95% CI [{lo:.3f}, {hi:.3f}]")
per = {}
for c in M.columns[:6]:
    m = M[c].notna().to_numpy()
    per[c] = {"n": int(m.sum()), "auc": float(auc(y[m], M[c].to_numpy()[m]))}
    print(f"   {c:30s} n={m.sum():3d}  AUC = {per[c]['auc']:.3f}")
ok = abs(a - 0.71) <= 0.01
rep = {"auc": a, "ci95_delong": [lo, hi], "published": {"auc": 0.71, "ci95": [0.60, 0.82]},
       "reproduces_within_0.01": bool(ok), "n_guilty": int(y.sum()), "n_naive": int((1 - y).sum()),
       "n_with_all_six": int(M.iloc[:, :6].notna().all(axis=1).sum()), "per_measure_auc_for_check_only": per}
(STEP2 / "authors_index_check.json").write_text(json.dumps(rep, indent=1))
print("\nREPRODUCES published .71 within +-0.01:", ok)
