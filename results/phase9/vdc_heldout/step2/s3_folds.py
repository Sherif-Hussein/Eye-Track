"""Step 2.1 - outer folds, written BEFORE any model is fitted: 10 folds x 3 repeats, participant-disjoint,
stratified by group (guilty / informed / naive). Seeds fixed: 20261106 + repeat."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from vdc_common import *

f = pd.read_csv(STEP2 / "features.csv")
pids = np.array(sorted(f.participant.unique()))
log = load_log().set_index("pid")
grp = log.loc[pids, "Condition"].map(GROUPS).to_numpy()
print(pd.Series(grp).value_counts().to_dict())
rows = []
for r in range(3):
    skf = StratifiedKFold(10, shuffle=True, random_state=20261106 + r)
    for k, (_, te) in enumerate(skf.split(pids, grp)):
        for i in te:
            rows.append((int(pids[i]), grp[i], r, k))
d = pd.DataFrame(rows, columns=["participant", "group", "repeat", "fold"]).sort_values(["repeat", "fold", "participant"])
assert d.groupby("repeat").participant.nunique().eq(len(pids)).all() and not d.duplicated(["participant", "repeat"]).any()
d.to_csv(STEP2 / "folds.csv", index=False)
print(d.groupby(["repeat", "fold"]).size().unstack().to_string())
print(d.groupby(["repeat", "fold", "group"]).size().unstack().head(10).to_string())
