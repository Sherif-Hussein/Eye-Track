import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vdc_model import *
f, X, y, g, idx = load_items()
folds, fd = load_folds()
grp = fd.drop_duplicates("participant").set_index("participant").group
G = [p for p in idx if grp[p] == "guilty"]; N = [p for p in idx if grp[p] == "naive"]; I = [p for p in idx if grp[p] == "informed"]
print(len(G), len(N), len(I), X.shape)
rng = np.random.default_rng(1)
# observed-type run on one repeat for timing
t0 = time.time(); st, tau = cv_stats(X, y, idx, folds, G, N, list(idx), repeats=(0,)); t1 = time.time()
print("one repeat (10 folds) observed: %.1f s" % (t1 - t0))
# 10 permutations (guilty vs naive labels), all 3 repeats
GN = G + N
t0 = time.time()
for i in range(10):
    p = rng.permutation(GN); Gp, Np = list(p[:len(G)]), list(p[len(G):])
    cv_stats(X, y, idx, folds, Gp, Np, GN, repeats=(0, 1, 2))
t = (time.time() - t0) / 10
print("per permutation (3 repeats x 10 folds): %.1f s -> 1000 perms: %.2f h; two tests (H1+H2): %.2f h" % (t, 1000 * t / 3600, 2000 * t / 3600))
