"""Diagnostic for R10: why do Colasanti surrogate cases (neutral images standing in for the target images) score as guilty?
Compares the z features of real targets (innocent participants) with the surrogate z features (neutral-as-target), and the
person-level statistic (stored fold scores). No refit. Single thread."""
import json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import run_r10 as R
U = R.U
task = U.build_mazza(lambda *a: None)["mazza_guilty_vs_innocent"]
Xs, info = R.mazza_surrogates(task)
X, y = task["X"], np.asarray(task["y"])
inn = y == 0
names = task["names"]
S = Xs["leave_out"].mean(0)                          # mean over the 20 draws
out = {"n_features": len(names)}
real_mu, surr_mu = X[inn].mean(0), S[inn].mean(0)
sd = X[inn].std(0)
out["mean_abs_z_real_targets_innocent"] = float(np.abs(X[inn]).mean())
out["mean_abs_z_surrogate_innocent"] = float(np.abs(S[inn]).mean())
out["mean_z_real_targets_innocent"] = float(X[inn].mean())
out["mean_z_surrogate_innocent"] = float(S[inn].mean())
out["median_standardised_shift_surrogate_minus_real"] = float(np.median((surr_mu - real_mu) / (sd + 1e-9)))
out["share_features_shifted_more_than_1sd"] = float((np.abs(surr_mu - real_mu) / (sd + 1e-9) > 1).mean())
# stored model direction: mean coefficients of repeat 0 across folds -> score of surrogate vs real on the whole sample with that average model
import pickle
coefs, icpts, Cs = [], [], []
for f, d in enumerate(pickle.load(open(R.UNI / "checkpoints" / "mazza_guilty_vs_innocent__f100__r0.pkl", "rb"))):
    coefs.append(d["coef"]); icpts.append(d["intercept"])
coef, icpt = np.mean(coefs, 0), float(np.mean(icpts))
P = U.prep_fit(X)
s_real = U.prep_apply(P, X) @ coef + icpt
s_surr = np.stack([U.prep_apply(P, Xs["leave_out"][i]) @ coef + icpt for i in range(R.D)]).mean(0)
out["average_model_score_real_innocent_mean"] = float(s_real[inn].mean())
out["average_model_score_real_guilty_mean"] = float(s_real[~inn].mean())
out["average_model_score_surrogate_innocent_mean"] = float(s_surr[inn].mean())
out["average_model_score_surrogate_guilty_mean"] = float(s_surr[~inn].mean())
out["note"] = ("Descriptive, in-sample use of fold-averaged coefficients of repeat 0 (not an out-of-fold result): only to show the direction "
               "of the shift of surrogate cases relative to real cases.")
# z of neutral-as-target by variant: leave-out reference has 2 images (1 df), inflating |z|
out["mean_abs_z_surrogate_in_reference"] = float(np.abs(Xs["in_reference"].mean(0)[inn]).mean())
(HERE / "diag_mazza_shift.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
