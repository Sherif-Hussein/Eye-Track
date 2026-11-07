"""Task 2.2b (EXPLORATORY), step 0b: fit a correlated jitter process to the measured residuals (measured/jitter_residuals.csv).

Model (per axis, independent axes; the measured x-y residual correlation at lag 0 is small and is reported):
  e(t) = a(t) + b(t), a and b independent Ornstein-Uhlenbeck processes (exponential autocovariance) with time constants tau1 (fast) and tau2
  (slow) and variance shares w and 1-w; total SD sigma. Sampled at the frame times t_k it is exactly
  x_k = phi * x_{k-1} + sqrt(1 - phi^2) * s * eps_k, phi = exp(-(t_k - t_{k-1}) / tau)   -> a sum of two AR(1) components (ARMA(2,1) on a
  regular grid), and works at any frame rate and for irregular frame times. A single AR(1) was tried first (`ar1_fit` below) and cannot match
  lags 1..5 of x (the measured autocorrelation decays more slowly than any single exponential).
Estimator (identical for the data and for the simulation): residual around each dot's median (settled frames), sigma = sqrt(sum r^2 / (N - dots)),
  rho_k = mean(r_i r_{i+k}) / mean(r^2) over pairs of valid frames exactly k frame indices apart, RMS-S2S = RMS of consecutive-frame differences.
Fit: parameters minimise the squared difference between the estimator's EXPECTED value on simulated dots (same number of dots, same frame times
  and same valid-frame pattern as the measured ones, 300 replicates, common random numbers) and the measured statistics, so the estimator's own
  small-sample bias is part of the match. A fresh-seed check (1000 replicates) is stored.
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import lro_common as C  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

LAGS = 5
df = pd.read_csv(C.MEAS / "jitter_residuals.csv")
dots = [g for _, g in df.groupby("dot", sort=True)]
T_list = [g.t_ms.to_numpy(float) for g in dots]
F_list = [g.frame_idx.to_numpy(int) for g in dots]
n_dot = len(dots)
n_tot = sum(len(t) for t in T_list)
off = np.cumsum([0] + [len(t) for t in T_list])
Fall = np.concatenate(F_list)
Did = np.concatenate([np.full(len(t), i) for i, t in enumerate(T_list)])
PAIR = {}
for k in range(1, LAGS + 1):
    PAIR[k] = np.flatnonzero((Did[k:] == Did[:-k]) & (Fall[k:] - Fall[:-k] == k))
v_dot_start = off[:-1]


def estimator(R):
    """R: (reps, n_tot) residuals ALREADY centred on each dot's median. Returns sigma, rho(1..5), rms_s2s per rep."""
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


def simulate_dots(par, reps, rng, comps=2):
    """par = (sigma, w, tau1, tau2) (ms); returns raw (not yet centred) processes at the measured frame times: (reps, n_tot)."""
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


def target(col):
    R = df[col].to_numpy(float)[None, :]
    sd, rho, rms = estimator(R)
    return sd[0], rho[0], rms[0]


def fit_axis(col, seed=7):
    sd0, rho0, rms0 = target(col)
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(seed + 100)
    # common random numbers: draw the raw normals once per call through a fixed generator state
    def obj(z):
        sig = np.exp(z[0]); w = 1 / (1 + np.exp(-z[1])); t1 = np.exp(z[2]); t2 = np.exp(z[3])
        if t2 < t1 or t2 > 5000 or t1 < 3:
            return 1e3
        r = np.random.default_rng(seed)
        S = centre(simulate_dots((sig, w, t1, t2), 120, r))
        sd, rho, rms = estimator(S)
        return float(np.sum((rho.mean(0) - rho0) ** 2) + ((sd.mean() - sd0) / sd0) ** 2 + ((rms.mean() - rms0) / rms0) ** 2)
    best = None
    for z0 in ([np.log(sd0 * 1.2), 0.0, np.log(60), np.log(600)], [np.log(sd0 * 1.4), 0.5, np.log(30), np.log(1500)], [np.log(sd0), -0.5, np.log(100), np.log(400)]):
        r = minimize(obj, z0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 220})
        if best is None or r.fun < best.fun:
            best = r
    z = best.x
    par = (float(np.exp(z[0])), float(1 / (1 + np.exp(-z[1]))), float(np.exp(z[2])), float(np.exp(z[3])))
    return par, float(best.fun)


def ar1_fit(col):
    """single exponential for comparison: choose sigma, tau to match rho1 and the s2s (reported only)."""
    sd0, rho0, rms0 = target(col)
    best = (1e9, None)
    for tau in np.exp(np.linspace(np.log(20), np.log(4000), 60)):
        r = np.random.default_rng(3)
        S = centre(simulate_dots((sd0, 0.9999, tau, tau * 1.0001), 80, r))
        sd, rho, rms = estimator(S)
        e = float(np.sum((rho.mean(0) - rho0) ** 2))
        if e < best[0]:
            best = (e, (tau, rho.mean(0).tolist(), float(sd.mean()), float(rms.mean())))
    return best


res = {"n_dots": n_dot, "n_samples": n_tot}
for ax, col in (("x", "res_x_deg"), ("y", "res_y_deg")):
    sd0, rho0, rms0 = target(col)
    par, f = fit_axis(col)
    r = np.random.default_rng(991)
    S = centre(simulate_dots(par, 1000, r))
    sd, rho, rms = estimator(S)
    a1 = ar1_fit(col)
    res[ax] = {"measured": {"sigma_deg": float(sd0), "rho": [float(v) for v in rho0], "rms_s2s_deg": float(rms0)},
               "params": {"sigma_deg": par[0], "share_fast": par[1], "tau_fast_ms": par[2], "tau_slow_ms": par[3]}, "objective": f,
               "check_fresh_seed_1000_reps": {"sigma_deg_mean": float(sd.mean()), "sigma_sd": float(sd.std()), "rho_mean": [float(v) for v in rho.mean(0)],
                                              "rho_sd": [float(v) for v in rho.std(0)], "rms_s2s_mean": float(rms.mean()), "rms_s2s_sd": float(rms.std())},
               "single_exponential_for_comparison": {"best_tau_ms": a1[1][0], "rho_mean": a1[1][1], "sigma": a1[1][2], "rms_s2s": a1[1][3], "sse_rho": a1[0]}}
    print(ax, json_ := __import__("json").dumps(res[ax], indent=1))
res["cross_axis_lag0_correlation"] = float(np.corrcoef(df.res_x_deg, df.res_y_deg)[0, 1])
res["note"] = "axes simulated independently (cross-axis residual correlation above); Gaussian marginals (measured excess kurtosis is reported in measured_params.json)"
C.jdump(C.MEAS / "jitter_model.json", res)
