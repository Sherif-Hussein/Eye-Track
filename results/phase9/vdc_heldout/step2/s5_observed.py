"""Step 2.1 - the observed run: nested CV on the saved folds (item model trained on guilty participants of the training folds,
threshold from the training naive participants), participant scores averaged over the 3 repeats -> scores.csv."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, time, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vdc_model import *

f, X, y, g, idx = load_items()
folds, fd = load_folds()
grp = fd.drop_duplicates("participant").set_index("participant").group
G = [p for p in idx if grp[p] == "guilty"]
N = [p for p in idx if grp[p] == "naive"]
t0 = time.time()
st, tau = cv_stats(X, y, idx, folds, G, N, list(idx))
print("observed run: %.1f s" % (time.time() - t0))
rows = []
for p in sorted(idx):
    r = {"participant": p, "group": grp[p], "n_displays": len(idx[p]) // 4}
    for k in range(3):
        r[f"stat_r{k}"] = st[k][p]
        r[f"tau7_r{k}"] = tau[k][p]
    rows.append(r)
d = pd.DataFrame(rows)
d["stat_mean"] = d[[f"stat_r{k}" for k in range(3)]].mean(1)
d["tau7_mean"] = d[[f"tau7_r{k}" for k in range(3)]].mean(1)
d["margin_mean"] = d.stat_mean - d.tau7_mean       # mean over repeats of (statistic - fold threshold)
d.to_csv(STEP2 / "scores.csv", index=False)
print(d.groupby("group")[["stat_mean", "tau7_mean"]].agg(["mean", "std"]).round(3).to_string())
print("AUC guilty vs naive:", round(auc((d.group[d.group != "informed"] == "guilty").astype(int), d.stat_mean[d.group != "informed"]), 4))
