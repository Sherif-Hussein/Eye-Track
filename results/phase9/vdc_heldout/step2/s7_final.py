"""Step 2.1 - final statistics from scores.csv, perm/*.npy, authors_index.csv. Single thread."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vdc_model import *
from scipy.stats import beta, rankdata
from edd.unified import holm, fast_auc
from edd.metrics import eer

sc = pd.read_csv(STEP2 / "scores.csv").set_index("participant")
ai = pd.read_csv(STEP2 / "authors_index.csv").set_index("participant")
grp = sc.group
S = sc.stat_mean
ids = {k: sc.index[grp == k].to_numpy() for k in ("guilty", "informed", "naive")}
rng = np.random.default_rng(20261107)
NB = 2000

def auc2(a, b):                      # a: known-group scores (positive), b: comparison group
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return fast_auc(y, np.r_[a, b])

def boot_ci(a, b):
    d = [auc2(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(NB)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

def load_perm(t):
    fs = sorted(glob.glob(str(STEP2 / "perm" / f"{t}_*.npy")))
    a = np.concatenate([np.load(f) for f in fs])
    return a

R = {"n": {k: int(len(v)) for k, v in ids.items()}}
# ---- H1, H2 (refit permutations)
for t, g2 in (("H1", "naive"), ("H2", "informed")):
    a, b = S[ids["guilty"]].to_numpy(), S[ids[g2]].to_numpy()
    obs = auc2(a, b)
    pm = load_perm(t)
    assert len(pm) == 1000, len(pm)
    p = float((1 + (pm >= obs).sum()) / (1 + len(pm)))
    R[t] = {"contrast": f"guilty (known, handled) vs {g2}", "auc": obs, "ci95_boot": boot_ci(a, b), "p_one_sided": p,
            "n_perm": int(len(pm)), "perm_auc_mean": float(pm.mean()), "perm_auc_sd": float(pm.std())}
# ---- H5 (informed vs naive; scores fixed, labels permuted)
a, b = S[ids["informed"]].to_numpy(), S[ids["naive"]].to_numpy()
obs = auc2(a, b)
pool = np.r_[a, b]; lab = np.r_[np.ones(len(a)), np.zeros(len(b))]
r5 = np.random.default_rng(20261107 + 5)
pm = np.array([fast_auc(r5.permutation(lab), pool) for _ in range(1000)])
R["H5"] = {"contrast": "informed (known, saw pictures) vs naive", "auc": obs, "ci95_boot": boot_ci(a, b),
           "p_one_sided": float((1 + (pm >= obs).sum()) / 1001), "n_perm": 1000,
           "note": "item model trained on guilty participants only, unaffected by this permutation"}
# ---- H3: our pipeline vs the authors' index, guilty vs naive, paired
gn = np.r_[ids["guilty"], ids["naive"]]
y = np.r_[np.ones(len(ids["guilty"])), np.zeros(len(ids["naive"]))].astype(int)
ours = S[gn].to_numpy(); theirs = ai.loc[gn, "W_mean"].to_numpy()
assert np.isfinite(theirs).all()
a_o, a_t = fast_auc(y, ours), fast_auc(y, theirs)
dobs = a_o - a_t
bd = []
n = len(gn)
for _ in range(NB):
    k = rng.integers(0, n, n)
    if len(np.unique(y[k])) < 2: continue
    bd.append(fast_auc(y[k], ours[k]) - fast_auc(y[k], theirs[k]))
ro, rt_ = rankdata(ours), rankdata(theirs)          # rank-transform each method's scores before swapping
r3 = np.random.default_rng(20261107 + 3)
pd_ = []
for _ in range(2000):
    sw = r3.random(n) < 0.5
    o2 = np.where(sw, rt_, ro); t2 = np.where(sw, ro, rt_)
    pd_.append(fast_auc(y, o2) - fast_auc(y, t2))
pd_ = np.array(pd_)
R["H3"] = {"contrast": "our pipeline minus authors' equal-weight index, guilty vs naive", "auc_ours": a_o, "auc_authors": a_t,
           "diff": dobs, "ci95_boot": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
           "p_two_sided": float((1 + (np.abs(pd_) >= abs(dobs) - 1e-12).sum()) / 2001), "n_swaps": 2000,
           "authors_ci95_delong": delong_ci(y, theirs)[1:]}
# ---- Holm over H2, H3, H5
ps = [R["H2"]["p_one_sided"], R["H3"]["p_two_sided"], R["H5"]["p_one_sided"]]
adj = holm(ps)
for k, v in zip(("H2", "H3", "H5"), adj):
    R[k]["p_holm"] = float(v)
R["H1"]["supported"] = bool(R["H1"]["p_one_sided"] < .05 and R["H1"]["ci95_boot"][0] > .5)

# ---- Estimation at the 7% threshold
dec = (sc.margin_mean > 0)
def cp(k, n_, al=.05):
    lo = 0.0 if k == 0 else beta.ppf(al / 2, k, n_ - k + 1)
    hi = 1.0 if k == n_ else beta.ppf(1 - al / 2, k + 1, n_ - k)
    return [float(lo), float(hi)]
est = {}
for g in ("guilty", "informed", "naive"):
    k, n_ = int(dec[ids[g]].sum()), len(ids[g])
    est[g] = {"flagged": k, "n": n_, "rate": k / n_, "ci95_clopper_pearson": cp(k, n_)}
dec_rep = {}
for r in range(3):
    m = (sc[f"stat_r{r}"] - sc[f"tau7_r{r}"]) > 0
    dec_rep[r] = {g: float(m[ids[g]].mean()) for g in ids}
fi, fn = dec[ids["informed"]].to_numpy().astype(float), dec[ids["naive"]].to_numpy().astype(float)
bdiff = [rng.choice(fi, len(fi)).mean() - rng.choice(fn, len(fn)).mean() for _ in range(NB)]
est["informed_minus_naive_fpr"] = {"diff": float(fi.mean() - fn.mean()), "ci95_boot": [float(np.percentile(bdiff, 2.5)), float(np.percentile(bdiff, 97.5))]}
est["per_repeat_rates"] = dec_rep
yy = (grp[gn] == "guilty").astype(int).to_numpy()
far, frr = est["naive"]["rate"], 1 - est["guilty"]["rate"]
est["HTER_at_7pct_threshold_guilty_vs_naive"] = (far + frr) / 2
est["EER_guilty_vs_naive"] = eer(yy, S[gn].to_numpy())
ge = np.r_[ids["guilty"], ids["informed"]]
est["EER_guilty_vs_informed"] = eer((grp[ge] == "guilty").astype(int).to_numpy(), S[ge].to_numpy())
ie = np.r_[ids["informed"], ids["naive"]]
est["EER_informed_vs_naive"] = eer((grp[ie] == "informed").astype(int).to_numpy(), S[ie].to_numpy())
est["HTER_note"] = "HTER = (FPR among naive + miss rate among guilty)/2 at the 7% threshold"
R["estimation"] = est

# ---- Exploratory D4: frozen pupil-free 30 Hz Delmas model on the closest features (sign as stored, not refitted)
fm = json.load(open(ROOT / "app/model/frozen_model.json"))["variants"]["sample_based"]
f, X, y_i, g_i, idx = load_items()
cols = [FEATS.index(c) for c in fm["base_features"]]
Z = X[:, cols]                                                  # already within-participant z (ddof 1, +1e-9)
Zi = np.where(np.isnan(Z), np.array(fm["impute_median"]), Z)
logit = ((Zi - np.array(fm["mean"])) / np.array(fm["scale"])) @ np.array(fm["coef"]) + fm["intercept"]
fz = pd.Series({p: cit_stat(logit[idx[p]], y_i[idx[p]]) for p in idx})
ex = {}
for nm, (g1, g2) in {"guilty_vs_naive": ("guilty", "naive"), "guilty_vs_informed": ("guilty", "informed"), "informed_vs_naive": ("informed", "naive")}.items():
    a, b = fz[ids[g1]].to_numpy(), fz[ids[g2]].to_numpy()
    ex[nm] = {"auc": auc2(a, b), "ci95_boot": boot_ci(a, b)}
ex["note"] = ("exploratory (D4), not in any family; frozen model applied unchanged (10 pupil-free features, stored sign: less dwell on the "
              "known item gives a higher score); no p-values; 11 of our features map to its 10 inputs by name; our dwell_1 window is the authors' 1-1000 ms")
R["exploratory_frozen_delmas_30hz"] = ex
fz.rename("frozen_stat").to_csv(STEP2 / "scores_frozen_exploratory.csv")
json.dump(R, open(STEP2 / "results.json", "w"), indent=1)
print(json.dumps(R, indent=1))
