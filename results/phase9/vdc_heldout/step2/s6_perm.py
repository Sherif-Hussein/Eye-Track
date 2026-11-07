"""Step 2.1 - refit label permutations. usage: s6_perm.py H1|H2 N_PERMS [START]. Checkpoints every 25 permutations to perm/.
H1: guilty/naive labels permuted between the 88 guilty+naive participants; H2: guilty/informed labels permuted between the
87 guilty+informed participants. Each permutation refits the item model (trained on the participants labelled guilty in the
training folds) and the 7% threshold (from the participants labelled naive in the training folds; for H2 the naive participants
are unchanged) on all 3 repeats x 10 folds and returns the AUC of the repeat-averaged participant statistic."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vdc_model import *

test, nperm = sys.argv[1], int(sys.argv[2])
f, X, y, g, idx = load_items()
folds, fd = load_folds()
grp = fd.drop_duplicates("participant").set_index("participant").group
G = sorted(p for p in idx if grp[p] == "guilty")
N = sorted(p for p in idx if grp[p] == "naive")
I = sorted(p for p in idx if grp[p] == "informed")
other = I if test == "H2" else N
pool = np.array(G + other)
lab = np.r_[np.ones(len(G), int), np.zeros(len(other), int)]
out = STEP2 / "perm"
out.mkdir(exist_ok=True)
CH = 25
t0 = time.time()
for c0 in range(0, nperm, CH):
    fn = out / f"{test}_{c0:04d}.npy"
    if fn.exists():
        continue
    aucs = []
    for i in range(c0, min(c0 + CH, nperm)):
        rng = np.random.default_rng(20261106 * 10 + (0 if test == "H1" else 1) * 100000 + i)
        pl = rng.permutation(lab)
        Gp, Op = list(pool[pl == 1]), list(pool[pl == 0])
        negs = Op if test == "H1" else N
        st, _ = cv_stats(X, y, idx, folds, Gp, negs, list(pool))
        s = np.array([np.mean([st[r][p] for r in range(3)]) for p in pool])
        aucs.append(auc(pl, s))
    np.save(fn, np.array(aucs))
    print(f"{test}: perms {c0}-{min(c0 + CH, nperm) - 1} done, elapsed {time.time() - t0:.0f} s", flush=True)
