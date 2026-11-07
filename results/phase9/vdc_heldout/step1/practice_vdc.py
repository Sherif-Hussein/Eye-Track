# Descriptive: how to tell practice rows (labelled trial 64) from the real trial 64; pid 23 and pids 17/31/43.
import sys, pickle, re, warnings
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
C = "<local path>"
L = lambda n, s=None: pickle.load(open(f"{C}{n}__{s or n.replace('.xlsx','')}.pkl", "rb"))
pf = lambda s: int(re.match(r"\s*(\d+)", str(s)).group(1))
log = L("Logboek.xlsx", "Blad1"); log.columns = ["pid", "Condition", "Date", "Time", "Excl"]
d2 = L("SecondPhaseParallel_DwellTime.xlsx"); d2["pid"] = d2.RECORDING_SESSION_LABEL.map(pf)
rt = L("RT.xlsx"); rt["pid"] = rt.Session_Name
cols = ["TRIAL_INDEX", "trial", "topleft", "topright", "bottomleft", "bottomright", "single", "fourorcenter", "criticalplace"]
x = d2[(d2.pid == 100) & (d2.IA_ID == 1) & (d2.trial.astype(str) == "64")][cols]
print("pid 100 rows labelled trial 64:"); print(x.to_string())
r = rt[(rt.pid == 100) & (rt.trial == 64)][["Trial_Index", "trial", "single", "topleft", "topright", "bottomleft", "bottomright", "fourorcenter", "criticalplace"]]
print("RT real trial 64 (pid 100):"); print(r.to_string())
print("\nBook configuration of stimulus trial 64 across all RT rows (distinct):")
print(rt[rt.trial == 64][["single", "topleft", "topright", "bottomleft", "bottomright", "fourorcenter", "criticalplace"]].drop_duplicates().to_string())
print("\nPractice rows: distinct configs among trial==64 & TRIAL_INDEX not in RT, pooled across pids with RT:")
rows = []
for p in sorted(set(rt.pid[rt.trial.notna()])):
    rti = set(rt[(rt.pid == p) & rt.trial.notna()].Trial_Index.astype(int))
    g = d2[(d2.pid == p) & (d2.IA_ID == 1) & (~d2.TRIAL_INDEX.isin(rti))]
    rows.append(g)
pr = pd.concat(rows)
print("n practice rows", len(pr), "| labelled trial values:", pr.trial.astype(str).value_counts().to_dict())
print(pr.groupby(["single", "topleft", "topright", "bottomleft", "bottomright", "fourorcenter", "criticalplace"]).size().to_string())
print("\nTRIAL_INDEX of practice rows (pooled): ", sorted(pr.TRIAL_INDEX.unique().tolist()))
print("\n== pid 23 (incomplete) and pids 17, 31, 43")
for p in (23, 17, 31, 43):
    print(p, log[log.pid == p][["Condition", "Excl"]].values.tolist(), "TRIAL_INDEX n:", d2[d2.pid == p].TRIAL_INDEX.nunique(), "rows trial==64:", d2[(d2.pid == p) & (d2.IA_ID == 1) & (d2.trial.astype(str) == "64")].TRIAL_INDEX.tolist(), "| RT rows:", len(rt[rt.pid == p]), "non-null trial:", int(rt[rt.pid == p].trial.notna().sum()), "non-null RT:", int(rt[rt.pid == p].RT_press.notna().sum()))
print("pid 23 real trial count (excluding trial 64 duplicates):", d2[(d2.pid == 23) & (d2.trial.astype(str) != "64")].trial.nunique())
print("RT rows pid 23:", len(rt[rt.pid == 23]), "non-null trial", int(rt[(rt.pid == 23)].trial.notna().sum()))
print("\nRT null-RT per retained pid, distribution of count of null RT (pooled, no condition):")
n = rt[rt.trial.notna()].groupby("pid").RT_press.apply(lambda s: int(s.isna().sum()))
print(n.value_counts().sort_index().to_dict())
