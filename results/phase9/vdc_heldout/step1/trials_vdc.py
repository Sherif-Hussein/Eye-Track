# Descriptive: identify practice / repeated rows, incomplete sessions, UNDEFINED rows, and confirm 64 usable
# trials per retained participant. Uses only IDs, trial indices and stimulus columns; no outcome by condition.
import sys, pickle, re, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
CACHE = "<local path>"
L = lambda n, s=None: pickle.load(open(f"{CACHE}{n}__{s or n.replace('.xlsx','')}.pkl", "rb"))
pid_from = lambda s: int(re.match(r"\s*(\d+)", str(s)).group(1))
log = L("Logboek.xlsx", "Blad1"); log.columns = ["pid", "Condition", "Date", "Time", "Excl"]
flag = set(log.pid[log.Excl == 1])
d2 = L("SecondPhaseParallel_DwellTime.xlsx"); d2["pid"] = d2.RECORDING_SESSION_LABEL.map(pid_from)
d1 = L("FirstPhaseParallel_DwellTime.xlsx"); d1["pid"] = d1.Participants
rt = L("RT.xlsx"); rt["pid"] = rt.Session_Name
sf = L("Single_FixationDuration.xlsx"); sf["pid"] = sf.RECORDING_SESSION_LABEL.map(pid_from)
vi = L("Parallel_Visits.xlsx"); vi["pid"] = vi.RECORDING_SESSION_LABEL.map(pid_from)
fc = L("Parallel_FixationCount.xlsx"); fc["pid"] = fc.RECORDING_SESSION_LABEL.map(pid_from)
tc = L("TimeCourse.xlsx", "TimeCourseFinal"); tc["pid"] = tc.RECORDING_SESSION_LABEL

print("== Incomplete sessions (TRIAL_INDEX count not in {69,74,79,84}) in SecondPhase")
n = d2.groupby("pid").TRIAL_INDEX.nunique()
odd = n[~n.isin([69, 74, 79, 84])]
print(odd.to_dict(), "| flagged by Logboek:", {p: (p in flag) for p in odd.index})
print("== UNDEFINED rows in SecondPhase/FixationCount: n rows", int((d2.trial.astype(str) == "UNDEFINED").sum()), "| pids", sorted(d2.pid[d2.trial.astype(str) == "UNDEFINED"].unique().tolist()))
print("   those pids flagged?", {p: (p in flag) for p in sorted(d2.pid[d2.trial.astype(str) == "UNDEFINED"].unique().tolist())})

print("\n== Practice rows: for pids with extra TRIAL_INDEX, are practice rows exactly those whose TRIAL_INDEX is absent from RT's Trial_Index?")
keep = sorted(set(log.pid[log.Excl != 1]))
res = {"match64": 0, "mismatch": []}
for name, df in [("FirstPhase", d1), ("SecondPhase", d2), ("FixationCount", fc), ("Visits", vi), ("Single", sf)]:
    cnt = {}
    bad = []
    for p in keep:
        ti = set(df[df.pid == p].TRIAL_INDEX.unique())
        rti = set(rt[(rt.pid == p) & rt.trial.notna()].Trial_Index.astype(int))
        if len(rti) != 64:
            bad.append((p, "RT trial_index n=%d" % len(rti))); continue
        matched = ti & rti
        cnt[len(matched)] = cnt.get(len(matched), 0) + 1
        extra = ti - rti
        # extras must all be labelled trial 64 (practice)
        ex_trials = set(map(str, df[(df.pid == p) & df.TRIAL_INDEX.isin(extra)].trial.unique()))
        if ex_trials - {"64"}:
            bad.append((p, "extras with trial", sorted(ex_trials)))
    print(f"   {name}: retained pids by #TRIAL_INDEX matching RT: {cnt}; problems: {bad[:10]}")

print("\n== After keeping only TRIAL_INDEX that appear in RT: trials per retained pid and rows per trial")
for name, df, per in [("FirstPhase", d1, 4), ("SecondPhase", d2, 4), ("FixationCount", fc, 4)]:
    k = []
    for p, g in df[df.pid.isin(keep)].groupby("pid"):
        rti = set(rt[(rt.pid == p) & rt.trial.notna()].Trial_Index.astype(int))
        g = g[g.TRIAL_INDEX.isin(rti)]
        k.append((p, g.TRIAL_INDEX.nunique(), g.shape[0] / max(1, g.TRIAL_INDEX.nunique())))
    k = pd.DataFrame(k, columns=["pid", "trials", "rows_per_trial"])
    print(f"   {name}: pids {len(k)}; trials per pid {k.trials.value_counts().to_dict()}; rows/trial {k.rows_per_trial.value_counts().to_dict()}")
k = []
for p, g in sf[sf.pid.isin(keep)].groupby("pid"):
    rti = set(rt[(rt.pid == p) & rt.trial.notna()].Trial_Index.astype(int))
    g = g[g.TRIAL_INDEX.isin(rti)]
    k.append((p, g.TRIAL_INDEX.nunique(), g.shape[0]))
k = pd.DataFrame(k, columns=["pid", "trials", "nfix"])
print(f"   Single: pids {len(k)}; trials per pid {k.trials.value_counts().to_dict()}; fixation rows per pid min/median/max {k.nfix.min()}/{int(k.nfix.median())}/{k.nfix.max()}")
k = []
for p, g in vi[vi.pid.isin(keep)].groupby("pid"):
    rti = set(rt[(rt.pid == p) & rt.trial.notna()].Trial_Index.astype(int))
    g = g[g.TRIAL_INDEX.isin(rti)]
    k.append((p, g.TRIAL_INDEX.nunique(), g.shape[0]))
k = pd.DataFrame(k, columns=["pid", "trials", "nfix"])
print(f"   Visits: pids {len(k)}; trials per pid {k.trials.value_counts().to_dict()}; fixation rows per pid min/median/max {k.nfix.min()}/{int(k.nfix.median())}/{k.nfix.max()}")

print("\n== TimeCourse: 'trial' (stimulus ID) per retained pid is the 32 critical-present trials?")
t = tc[tc.pid.isin(keep)]
print("   retained pids", t.pid.nunique(), "| trials per pid", t.groupby("pid").TRIAL_INDEX.nunique().value_counts().to_dict(), "| bins per trial", t.groupby(["pid", "TRIAL_INDEX"]).size().value_counts().to_dict())
print("   BIN_INDEX 0 = first 100 ms bin of the 4-item display; 50 bins = 0-5000 ms (per authors' TimecoursFigure.R)")
print("   Do TimeCourse TRIAL_INDEX values all appear in RT Trial_Index for the pid?")
bad = 0
for p, g in t.groupby("pid"):
    rti = set(rt[(rt.pid == p) & rt.trial.notna()].Trial_Index.astype(int))
    bad += len(set(g.TRIAL_INDEX.unique()) - rti)
print("   unmatched (pid, TRIAL_INDEX) pairs:", bad)
print("   range of 'single', 'IA_k_ID' strings:", t.IA_1_ID.unique().tolist(), t.IA_2_ID.unique().tolist(), t.IA_3_ID.unique().tolist(), t.IA_4_ID.unique().tolist(), t.IA_0_ID.unique().tolist())
print("   crimerelated/control/other: first rows of one trial (columns only, bins 0-4 and 20-24):")
x = t[(t.pid == 100) & (t.TRIAL_INDEX == 8)].set_index("BIN_INDEX")[["AVERAGE_IA_1_SAMPLE_COUNT_%", "AVERAGE_IA_2_SAMPLE_COUNT_%", "AVERAGE_IA_3_SAMPLE_COUNT_%", "AVERAGE_IA_4_SAMPLE_COUNT_%", "other", "crimerelated", "control"]]
print(x.iloc[list(range(0, 5)) + list(range(20, 25))].to_string())
print("   TimeCourse missing values anywhere:", int(tc.isna().sum().sum()))
