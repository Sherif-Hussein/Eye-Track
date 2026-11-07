"""Task 2.2c Part A (EXPLORATORY): realism checks. Writes sens_realism.json here.
1. Drift-check failure rate and mean applied correction (rates 30, 20, 15, 10 Hz), 2.2b features against the reduced features, from the `diag` stored in
   the feature files (correction variant 'app', 75 participants x 3 draws x 64 trials). Recorded: 6 of 85 scored trials failed (7.06%), mean magnitude of
   the 79 passed logged offsets 2.64 deg (measured/measured_params.json).
2. Jitter statistics. The estimator and the simulator of s0b_fit_jitter.py are re-typed here (importing s0b would run its fit and overwrite
   measured/jitter_model.json). The re-typed code is verified by reproducing the stored 2.2b check_fresh_seed_1000_reps values with the 2.2b
   parameters (seed 991, 1000 replicates, as in s0b); then the reduced parameters are run the same way.
"""
import json
import pickle
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LRO = HERE.parent
sys.path.insert(0, str(LRO))
import lro_common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402

RATES = [30, 20, 15, 10]


def realism(fdir):
    out = {}
    for fs in RATES:
        ok, mean_off, grid = [], [], []
        for f in sorted(fdir.glob(f"r{fs}_p??.pkl")):
            v = pickle.load(open(f, "rb"))
            for d in (1, 2, 3):
                dg = v["draws"][d]["app"]["diag"]
                ok.append(dg["check_ok_share"]); mean_off.append(dg["mean_applied_offset_deg"]); grid.append(dg["grid_valid_share"])
        ok, mean_off = np.array(ok), np.array(mean_off)
        w = ok.copy()
        m = np.isfinite(mean_off)
        out[str(fs)] = {"n_participant_draws": int(len(ok)), "check_pass_share": float(ok.mean()), "check_failure_share": float(1 - ok.mean()),
                        "mean_applied_correction_deg_trial_weighted": float(np.sum(w[m] * mean_off[m]) / np.sum(w[m])),
                        "mean_applied_correction_deg_unweighted": float(mean_off[m].mean()), "grid_valid_share": float(np.mean(grid)),
                        "sd_of_failure_share_across_participant_draws": float((1 - ok).std())}
    return out


# ---- re-typed from s0b_fit_jitter.py (estimator and simulator, unchanged logic)
LAGS = 5
df = pd.read_csv(C.MEAS / "jitter_residuals.csv")
dots = [g for _, g in df.groupby("dot", sort=True)]
T_list = [g.t_ms.to_numpy(float) for g in dots]
F_list = [g.frame_idx.to_numpy(int) for g in dots]
n_dot, n_tot = len(dots), sum(len(t) for t in T_list)
off = np.cumsum([0] + [len(t) for t in T_list])
Fall = np.concatenate(F_list)
Did = np.concatenate([np.full(len(t), i) for i, t in enumerate(T_list)])
PAIR = {k: np.flatnonzero((Did[k:] == Did[:-k]) & (Fall[k:] - Fall[:-k] == k)) for k in range(1, LAGS + 1)}


def estimator(R):
    sd = np.sqrt((R ** 2).sum(1) / (n_tot - n_dot))
    v = (R ** 2).mean(1)
    rho = np.stack([(R[:, PAIR[k]] * R[:, PAIR[k] + k]).mean(1) / v for k in range(1, LAGS + 1)], 1)
    d = R[:, PAIR[1] + 1] - R[:, PAIR[1]]
    return sd, rho, np.sqrt((d ** 2).mean(1))


def centre(Rraw):
    out = np.empty_like(Rraw)
    for i in range(n_dot):
        s = slice(off[i], off[i + 1])
        out[:, s] = Rraw[:, s] - np.median(Rraw[:, s], axis=1, keepdims=True)
    return out


def simulate_dots(par, reps, rng):
    sig, w, t1, t2 = par
    out = np.empty((reps, n_tot))
    for i, t in enumerate(T_list):
        n = len(t)
        e = rng.standard_normal((2, reps, n))
        x1 = np.sqrt(w) * sig * e[0, :, 0]
        x2 = np.sqrt(1 - w) * sig * e[1, :, 0]
        out[:, off[i]] = x1 + x2
        for j in range(1, n):
            dt = t[j] - t[j - 1]
            p1, p2 = np.exp(-dt / t1), np.exp(-dt / t2)
            x1 = p1 * x1 + np.sqrt(1 - p1 ** 2) * np.sqrt(w) * sig * e[0, :, j]
            x2 = p2 * x2 + np.sqrt(1 - p2 ** 2) * np.sqrt(1 - w) * sig * e[1, :, j]
            out[:, off[i] + j] = x1 + x2
    return out


def run_jitter(par):
    S = centre(simulate_dots((par["sigma_deg"], par["share_fast"], par["tau_fast_ms"], par["tau_slow_ms"]), 1000, np.random.default_rng(991)))
    sd, rho, rms = estimator(S)
    return {"sigma_deg_mean": float(sd.mean()), "sigma_sd": float(sd.std()), "rho_mean": [float(v) for v in rho.mean(0)], "rho_sd": [float(v) for v in rho.std(0)],
            "rms_s2s_mean": float(rms.mean()), "rms_s2s_sd": float(rms.std())}


def main():
    res = {"recorded": {"drift_failures": 6, "scored_trials": 85, "failure_rate": 6 / 85,
                        "failure_rate_exact_95ci": [float(beta.ppf(0.025, 6, 80)), float(beta.ppf(0.975, 7, 79))], "mean_applied_magnitude_deg": 2.6432704275753522}}
    res["realism_2.2b"] = realism(LRO / "features")
    res["realism_reduced"] = realism(HERE / "features")
    jm = json.loads((LRO / "measured" / "jitter_model.json").read_text(encoding="utf-8"))
    red = json.loads((HERE / "jitter_params_reduced.json").read_text(encoding="utf-8"))
    res["jitter"] = {}
    for ax in ("x", "y"):
        mine_old = run_jitter(jm[ax]["params"])
        stored = jm[ax]["check_fresh_seed_1000_reps"]
        diff = max(abs(mine_old["sigma_deg_mean"] - stored["sigma_deg_mean"]), abs(mine_old["rms_s2s_mean"] - stored["rms_s2s_mean"]),
                   max(abs(a - b) for a, b in zip(mine_old["rho_mean"], stored["rho_mean"])))
        res["jitter"][ax] = {"measured": jm[ax]["measured"], "model_2.2b_process": mine_old, "model_reduced_process": run_jitter(red[ax]["params"]),
                             "retyped_code_vs_stored_2.2b_max_abs_diff": diff}
        print(ax, "retyped code reproduces stored 2.2b check, max abs diff", diff)
    (HERE / "sens_realism.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
