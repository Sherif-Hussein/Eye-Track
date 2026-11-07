# Descriptive counts only (rows, distinct IDs, participants per condition label, trials per participant,
# exclusion flags, stimulus-design cross-tabs). NO dwell/RT/fixation statistic is split by condition.
# Reads pickles written by inspect_vdc.py (cache in scratchpad).
import sys, pickle, re, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import pandas as pd
CACHE = "<local path>"
def L(name, sh=None):
    sh = sh or name.replace(".xlsx", "")
    return pickle.load(open(f"{CACHE}{name}__{sh}.pkl", "rb"))

HARD22 = [13, 14, 15, 20, 24, 35, 71, 74, 77, 88, 95, 104, 110, 114, 124, 127, 135, 138, 142, 148, 149, 150]
log = L("Logboek.xlsx", "Blad1")
log.columns = ["pid", "Condition", "Date", "Time", "Excl_other"]
print("== Logboek: rows", len(log), "| distinct pid", log.pid.nunique(), "| pid range", log.pid.min(), log.pid.max())
print("Condition label counts (all 155):"); print(log.Condition.value_counts().to_string())
print("Excl_other flag counts:"); print(log.Excl_other.value_counts().to_string())
print("Flagged IDs (Excl_other==1):", sorted(log.pid[log.Excl_other == 1].tolist()))
print("Overlap of HARD22 with Logboek flag:", sorted(set(HARD22) & set(log.pid[log.Excl_other == 1])))
print("HARD22 not flagged in Logboek:", sorted(set(HARD22) - set(log.pid[log.Excl_other == 1])))
print("Flagged but not in HARD22:", sorted(set(log.pid[log.Excl_other == 1]) - set(HARD22)))
keep = log[(log.Excl_other != 1)]
print("After Logboek flag only: n =", len(keep)); print(keep.Condition.value_counts().to_string())
keep2 = log[(log.Excl_other != 1) & (~log.pid.isin(HARD22))]
print("After Logboek flag AND hard-coded 22: n =", len(keep2)); print(keep2.Condition.value_counts().to_string())
keep3 = log[~log.pid.isin(HARD22)]
print("After hard-coded 22 only: n =", len(keep3)); print(keep3.Condition.value_counts().to_string())
# Lists from TimecoursFigure.R (authors' condition lists), for ID/label consistency check only
guilty = [2, 4, 7, 22, 28, 29, 31, 32, 36, 37, 38, 46, 51, 52, 55, 56, 65, 66, 69, 80, 83, 84, 86, 89, 93, 94, 96, 98, 105, 106, 109, 111, 119, 121, 122, 126, 130, 131, 139, 144, 152, 154]
informed = [3, 5, 6, 10, 18, 19, 21, 26, 33, 34, 43, 44, 47, 53, 58, 59, 61, 62, 68, 73, 75, 76, 79, 81, 82, 85, 87, 92, 99, 100, 103, 112, 113, 115, 116, 118, 123, 128, 129, 133, 134, 136, 137, 147, 155]
naive = [1, 8, 9, 11, 12, 16, 17, 23, 25, 27, 30, 39, 40, 41, 42, 45, 48, 49, 50, 54, 57, 60, 63, 64, 67, 70, 72, 78, 90, 91, 97, 101, 102, 107, 108, 117, 120, 125, 132, 140, 141, 143, 145, 146, 151, 153]
print("TimecoursFigure.R lists sizes: guilty", len(guilty), "informed", len(informed), "naive", len(naive), "total", len(guilty) + len(informed) + len(naive))
lab = {**{p: "Guilty" for p in guilty}, **{p: "Informed" for p in informed}, **{p: "Naive" for p in naive}}
mism = [(r.pid, r.Condition, lab.get(r.pid)) for r in log.itertuples() if lab.get(r.pid) != r.Condition]
print("Logboek label vs TimecoursFigure.R list mismatches:", mism)

def pid_from(s):
    m = re.match(r"\s*(\d+)", str(s)); return int(m.group(1)) if m else None

files = {
 "FirstPhaseParallel_DwellTime": ("Participants", None),
 "SecondPhaseParallel_DwellTime": ("RECORDING_SESSION_LABEL", pid_from),
 "Parallel_FixationCount": ("RECORDING_SESSION_LABEL", pid_from),
 "Parallel_Visits": ("RECORDING_SESSION_LABEL", pid_from),
 "Single_FixationDuration": ("RECORDING_SESSION_LABEL", pid_from),
 "RT": ("Session_Name", None),
}
ids_per_file = {}
for f, (col, fn) in files.items():
    df = L(f + ".xlsx", f)
    raw = df[col]
    pid = raw.map(fn) if fn else raw
    df["pid"] = pid
    ids_per_file[f] = set(pid.dropna().astype(int))
    print("\n== ", f, "rows", len(df), "| distinct raw IDs", raw.nunique(), "| distinct numeric pid", pid.nunique())
    if fn:
        suf = raw.astype(str).str.replace(r"^\d+", "", regex=True)
        print("   ID suffix values:", suf.value_counts().to_dict())
    print("   pids not in Logboek:", sorted(set(pid.dropna().astype(int)) - set(log.pid)))
    print("   Logboek pids missing from file:", sorted(set(log.pid) - set(pid.dropna().astype(int))))
    print("   HARD22 present in file:", sorted(set(HARD22) & set(pid.dropna().astype(int))))
    print("   Logboek-flagged present in file:", sorted(set(log.pid[log.Excl_other == 1]) & set(pid.dropna().astype(int))))
    # participants per Logboek condition label (descriptive)
    m = df[["pid"]].drop_duplicates().merge(log[["pid", "Condition", "Excl_other"]], on="pid", how="left")
    print("   participants per Logboek condition in file:", m.Condition.value_counts(dropna=False).to_dict())
    m2 = m[(m.Excl_other != 1) & (~m.pid.isin(HARD22))]
    print("   ... after Logboek flag + hard-coded 22:", m2.Condition.value_counts(dropna=False).to_dict(), "n =", len(m2))
    # trials per participant
    if f in ("RT",):
        tcount = df.dropna(subset=["trial"]).groupby("pid").size()
        print("   rows per pid (non-null trial): value counts", tcount.value_counts().sort_index().to_dict())
        print("   rows with null trial:", int(df.trial.isna().sum()), "| pids with them:", sorted(df.pid[df.trial.isna()].unique().tolist())[:60])
    elif f.startswith("Parallel_Visits") or f.startswith("Single_Fix"):
        pt = df.groupby("pid")["TRIAL_INDEX"].nunique()
        print("   distinct TRIAL_INDEX per pid:", pt.value_counts().sort_index().to_dict())
        pt2 = df.groupby("pid")["trial"].nunique()
        print("   distinct trial per pid:", pt2.value_counts().sort_index().to_dict())
    else:
        pt = df.groupby("pid")["TRIAL_INDEX"].nunique()
        print("   distinct TRIAL_INDEX per pid:", pt.value_counts().sort_index().to_dict())
        pt2 = df.groupby("pid")["trial"].nunique()
        print("   distinct trial per pid:", pt2.value_counts().sort_index().to_dict())
        rows = df.groupby("pid").size()
        print("   rows per pid (4 per trial expected):", rows.value_counts().sort_index().to_dict())
    print("   'trial' value set (first 70 sorted as str):", sorted(set(map(str, df["trial"].unique())), key=lambda s: (len(s), s))[:75])
    print("   TRIAL_INDEX range:", df["TRIAL_INDEX"].min() if "TRIAL_INDEX" in df else None, df["TRIAL_INDEX"].max() if "TRIAL_INDEX" in df else None) if "TRIAL_INDEX" in df else None
    if "Exclusions" in df: print("   Exclusions column values:", df.Exclusions.value_counts().to_dict())

print("\n== ID set consistency across files")
base = ids_per_file["FirstPhaseParallel_DwellTime"]
for f, s in ids_per_file.items():
    print(f"   {f}: n={len(s)}; in Logboek={len(s & set(log.pid))}; only here vs Dwell1: {sorted(s - base)[:20]}; missing vs Dwell1: {sorted(base - s)[:20]}")
