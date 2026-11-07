# Descriptive structure checks only (trial-index mapping, ID suffixes, stimulus coding, value ranges over ALL
# participants pooled). No measure is split by condition.
import sys, pickle, re, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
CACHE = "<local path>"
L = lambda n, s=None: pickle.load(open(f"{CACHE}{n}__{s or n.replace('.xlsx','')}.pkl", "rb"))
pid_from = lambda s: int(re.match(r"\s*(\d+)", str(s)).group(1))
log = L("Logboek.xlsx", "Blad1"); log.columns = ["pid", "Condition", "Date", "Time", "Excl"]

d1 = L("FirstPhaseParallel_DwellTime.xlsx"); d1["pid"] = d1.Participants
d2 = L("SecondPhaseParallel_DwellTime.xlsx"); d2["pid"] = d2.RECORDING_SESSION_LABEL.map(pid_from)
fc = L("Parallel_FixationCount.xlsx"); fc["pid"] = fc.RECORDING_SESSION_LABEL.map(pid_from)
vi = L("Parallel_Visits.xlsx"); vi["pid"] = vi.RECORDING_SESSION_LABEL.map(pid_from)
sf = L("Single_FixationDuration.xlsx"); sf["pid"] = sf.RECORDING_SESSION_LABEL.map(pid_from)
rt = L("RT.xlsx"); rt["pid"] = rt.Session_Name
tc = L("TimeCourse.xlsx", "TimeCourseFinal"); tc["pid"] = tc.RECORDING_SESSION_LABEL

print("== ID suffix vs Logboek condition label (consistency of session labels with the log)")
raw = d2[["RECORDING_SESSION_LABEL", "pid"]].drop_duplicates()
raw["suffix"] = raw.RECORDING_SESSION_LABEL.astype(str).str.replace(r"^\d+", "", regex=True)
print(raw.merge(log[["pid", "Condition"]], on="pid").groupby(["suffix", "Condition"]).size().unstack(fill_value=0).to_string())
print("Raw labels per pid (max):", raw.groupby("pid").size().max())

print("\n== pid 1, FirstPhase: TRIAL_INDEX -> trial (one row per trial, first 8 and last 8)")
x = d1[d1.pid == 1][["TRIAL_INDEX", "trial"]].drop_duplicates().sort_values("TRIAL_INDEX")
print(len(x), "distinct trials; head:"); print(x.head(8).T.to_string()); print("tail:"); print(x.tail(8).T.to_string())
print("\n== pid 100 (label 100_I), SecondPhase: TRIAL_INDEX -> trial")
x = d2[d2.pid == 100][["TRIAL_INDEX", "trial"]].drop_duplicates().sort_values("TRIAL_INDEX")
print(len(x), "distinct; "); print(x.T.to_string())
print("\n== TRIAL_INDEX rows with trial==64 per pid (SecondPhase), distribution of counts")
c = d2[d2.trial.astype(str) == "64"].groupby("pid").TRIAL_INDEX.nunique()
print(c.value_counts().sort_index().to_dict())
print("distinct non-numeric trial values in SecondPhase:", [v for v in d2.trial.unique() if not str(v).isdigit()])
print("distinct non-numeric fourorcenter:", [v for v in d2.fourorcenter.unique() if not str(v).isdigit()], " criticalplace:", [v for v in d2.criticalplace.unique() if not str(v).isdigit()])

print("\n== Item base names (digits stripped) across files")
def base(s): return re.sub(r"\d*\.jpg$", "", str(s))
for col in ["single", "topleft"]:
    print(col, sorted(set(map(base, d1[col]))))
print("\n== Number of distinct (trial) stimulus configurations in FirstPhase for pid 1, with criticalplace / fourorcenter")
x = d1[d1.pid == 1].drop_duplicates("trial")[["trial", "topleft", "topright", "bottomleft", "bottomright", "single", "fourorcenter", "criticalplace"]].sort_values("trial")
print(x.head(16).to_string())

print("\n== Design coding: fourorcenter x criticalplace>0 x single-is-critical (rows pooled over participants, RT file)")
r = rt.dropna(subset=["trial"]).copy()
r["single_crit"] = r.single.map(lambda s: not re.search(r"\d\.jpg$", str(s)))
r["cp_gt0"] = r.criticalplace > 0
print(r.groupby(["fourorcenter", "cp_gt0", "single_crit"]).size().to_string())
print("Trials per participant by fourorcenter (pid 1):", r[r.pid == 1].fourorcenter.value_counts().sort_index().to_dict())
print("criticalplace value counts (pooled RT rows):", r.criticalplace.value_counts().sort_index().to_dict())
print("Critical item is the file without a trailing digit in the four-display? check: for rows with criticalplace in 1..4, is the item at that position digit-free?")
pos = {1: "topleft", 2: "topright", 3: "bottomleft", 4: "bottomright"}
ok = tot = 0
for row in r[r.criticalplace > 0].itertuples():
    tot += 1
    ok += not re.search(r"\d\.jpg$", str(getattr(row, pos[int(row.criticalplace)])))
print("   consistent rows:", ok, "of", tot)
nocrit = r[r.criticalplace == 0]
print("   rows with criticalplace==0 where some four-display item is digit-free:", int(sum(any(not re.search(r"\d\.jpg$", str(getattr(row, p))) for p in pos.values()) for row in nocrit.itertuples())), "of", len(nocrit))

print("\n== Value ranges, all participants pooled")
print("FirstPhase IA_DWELL_TIME min/max:", d1.IA_DWELL_TIME.min(), d1.IA_DWELL_TIME.max(), "| SecondPhase:", d2.IA_DWELL_TIME.min(), d2.IA_DWELL_TIME.max())
print("IA_FIXATION_COUNT min/max:", fc.IA_FIXATION_COUNT.min(), fc.IA_FIXATION_COUNT.max())
print("CURRENT_FIX_DURATION min/max:", sf.CURRENT_FIX_DURATION.min(), sf.CURRENT_FIX_DURATION.max())
print("Single IA labels:", sf.CURRENT_FIX_INTEREST_AREA_LABEL.value_counts().to_dict())
print("Visits IA labels:", vi.CURRENT_FIX_INTEREST_AREA_LABEL.value_counts().to_dict())
print("RT: Key_response values:", rt.Key_response.value_counts(dropna=False).to_dict(), "| RT null:", int(rt.RT_press.isna().sum()), "of", len(rt), "| RT min/max:", rt.RT_press.min(), rt.RT_press.max())
print("RT 'correct' values (pooled):", rt.correct.value_counts(dropna=False).to_dict(), "| correcttrial:", rt.correcttrial.value_counts(dropna=False).to_dict())
print("RT other flag cols: Critical_items", rt.Critical_items.value_counts(dropna=False).to_dict(), "| filter_$", rt["filter_$"].value_counts(dropna=False).to_dict(), "| Critical_items_parallel_serial", rt.Critical_items_parallel_serial.value_counts(dropna=False).to_dict())
print("RT Trial_Recycled:", rt.Trial_Recycled.value_counts(dropna=False).to_dict())

print("\n== Dwell phase sums per trial (FirstPhase, pid 1, first 4 trials): total over 4 AOIs")
print(d1[d1.pid == 1].groupby("trial").IA_DWELL_TIME.sum().head(6).to_dict())
print("SecondPhase pid 1:", d2[d2.pid == 1].groupby("trial").IA_DWELL_TIME.sum().head(6).to_dict())
print("Fixations in Visits per trial (pid 1, first 6 trials):", vi[vi.pid == 1].groupby("trial").size().head(6).to_dict())
print("FixCount sum per trial (pid 1, first 6 trials):", fc[fc.pid == 1].groupby("trial").IA_FIXATION_COUNT.sum().head(6).to_dict())

print("\n== TimeCourse structure")
print("rows", len(tc), "| distinct pid", tc.pid.nunique(), "| BIN_INDEX range", tc.BIN_INDEX.min(), tc.BIN_INDEX.max())
g = tc.groupby("pid").agg(n_rows=("BIN_INDEX", "size"), n_trials=("TRIAL_INDEX", "nunique"))
print("rows per pid:", g.n_rows.value_counts().sort_index().to_dict())
print("trials per pid:", g.n_trials.value_counts().sort_index().to_dict())
print("fourorcenter values:", tc.fourorcenter.value_counts().sort_index().to_dict(), "| criticalplace values:", tc.criticalplace.value_counts().sort_index().to_dict())
print("pids in TimeCourse not in Logboek:", sorted(set(tc.pid) - set(log.pid)), "| Logboek flagged present:", sorted(set(tc.pid) & set(log.pid[log.Excl == 1])))
m = pd.DataFrame({"pid": sorted(set(tc.pid))}).merge(log, on="pid")
print("TimeCourse participants per Logboek label (all):", m.Condition.value_counts().to_dict(), "| after flag:", m[m.Excl != 1].Condition.value_counts().to_dict())
x = tc[tc.pid == 100].drop_duplicates("TRIAL_INDEX")[["TRIAL_INDEX", "trial", "single", "fourorcenter", "criticalplace", "topleft", "topright", "bottomleft", "bottomright"]].sort_values("TRIAL_INDEX")
print("pid 100 trials in TimeCourse:", len(x)); print(x.head(10).to_string())
cols = ["AVERAGE_IA_1_SAMPLE_COUNT_%", "AVERAGE_IA_2_SAMPLE_COUNT_%", "AVERAGE_IA_3_SAMPLE_COUNT_%", "AVERAGE_IA_4_SAMPLE_COUNT_%", "other"]
s = tc[cols].sum(axis=1)
print("sum of 4 IA % + other: min/max", round(s.min(), 4), round(s.max(), 4))
print("BIN width inferred from authors' script: 100 ms (bin = BIN_INDEX*100 ms, 0-5000 ms)")
