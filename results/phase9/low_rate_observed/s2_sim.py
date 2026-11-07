"""Task 2.2b (EXPLORATORY): the observed-webcam simulation (library). Sources are stated per piece.

Reused UNCHANGED: edd.delmas_raw (_fill, degrade's missing-sample rule and Kaiser anti-alias filter design are re-stated here for frame times
that are not on a regular grid; degrade itself is used in the checks), app/features.py to_grid (the app's own nearest-sample resampling of the
time-stamped frames onto the uniform grid), scripts/phase6_trackC_c1.py (sb_features_batch, make_table, FS) and edd.delmas_fast.trial_features_batch
through results/phase9/low_rate/s4_webcam.py, results/phase9/low_rate/lr_common.py.

Pipeline for one participant (64 trials x 5500 samples at 1000 Hz, display onset at index 500) and one frame rate f:
  1. fill gaps linearly (dr._fill), low-pass with the SAME filter that dr.degrade (scipy resample_poly, Kaiser beta 5, cutoff f/2, 10 output periods
     half-length, 'line' edge extension) applies, built in the 1000 Hz domain (scipy upfirdn, up = down = 1);
  2. frame times: consecutive inter-frame gaps taken in blocks from the measured sessions (measured/gaps_*.npy), scaled so the mean gap is
     1000/f ms (f = 30, 20, 15 or 10), a random start phase; the filtered 1000 Hz signal is read at those times (linear interpolation);
     a frame is missing when any of the original samples within half a nominal period was missing (dr.degrade's rule);
  3. observed-webcam error on the frames: compression about the screen centre (gain x 0.79, y 0.65), per-trial offset (empirical joint
     distribution), correlated jitter (two-component exponential process fitted in s0b_fit_jitter.py, sampled at the frame times), and sample
     loss with the measured per-trial valid share;
  4. optional per-trial offset correction measured on the frames of [-500, -100) ms before display onset (the app's drift_from_frames rule);
  5. the app's to_grid puts the irregular frames on the uniform grid of the same rate; features and models are those of s4_webcam.py.
"""
import sys
from fractions import Fraction

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import lro_common as C  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import firwin, upfirdn  # noqa: E402

import lr_common as L  # noqa: E402
import s4_webcam as S4  # noqa: E402
from edd import delmas_raw as dr  # noqa: E402
from edd import delmas_fast as df_  # noqa: E402
from features import to_grid  # noqa: E402  (app/features.py)

C1 = S4.C1
SEED = S4.C1.W.SEED
GAIN = (0.88, 0.79)                 # x, y: EMEET-only medians (measured/measured_params.json, asserted in s0_measure.py)
MAX_DRIFT_DEG = 6.0                 # app/config.py drift_max_deg
WIN_PRE = (-500.0, -100.0)          # pre-trial fixation window available in the Delmas arrays (the app: 40% .. end - 0.1 s of 1 s)
MIN_SHARE = 8.0 / 15.0              # the app's 8 valid frames of ~15 in 0.5 s at 30 fps, kept as a share of the window
PPD = C.PPD_DELMAS


def n_min_frames(fs):
    return max(3, int(np.ceil(MIN_SHARE * (WIN_PRE[1] - WIN_PRE[0]) / 1000.0 * fs)))


# ------------------------------------------------------------------------- measured inputs
def load_measured():
    mp = C.json.loads((C.MEAS / "measured_params.json").read_text(encoding="utf-8"))
    jm = C.json.loads((C.MEAS / "jitter_model.json").read_text(encoding="utf-8"))
    import pandas as pd
    drift = pd.read_csv(C.MEAS / "drift_offsets.csv")
    vs = pd.read_csv(C.MEAS / "valid_share.csv")
    gaps = [np.load(C.MEAS / f"gaps_{s}.npy") for s in C.SESSIONS]
    allg = np.concatenate(gaps)
    return {"mp": mp, "jm": jm, "off": drift[["gaze_off_x_deg", "gaze_off_y_deg"]].to_numpy(float), "loss": 1.0 - vs.valid_share.to_numpy(float),
            "gaps": gaps, "gap_w": np.array([len(g) for g in gaps], float) / len(allg), "mean_gap": float(allg.mean())}


# ------------------------------------------------------------------------- 1. fill + low-pass (the degrade filter)
_KERNELS = {}


def kernel(fs):
    if fs not in _KERNELS:
        fr = Fraction(fs, 1000).limit_denominator(1000)
        up, down = fr.numerator, fr.denominator
        mr = max(up, down)
        H = int(round(10.0 * mr / up))
        h = firwin(2 * H + 1, up / down, window=("kaiser", 5.0))
        _KERNELS[fs] = (h, H)
    return _KERNELS[fs]


def lowpass_rows(A, fs):
    """A: (T, n) already gap-filled. Zero-phase filter with the degrade filter in the 1000 Hz domain."""
    h, H = kernel(fs)
    n = A.shape[1]
    out = np.empty_like(A, dtype=float)
    for i in range(A.shape[0]):
        y = upfirdn(h, A[i].astype(float), 1, 1, mode="line")
        out[i] = y[H:H + n]
    return out


def prepare(X, Y, P, fs):
    """X, Y, P: (T, 5500) float with NaN. Returns filtered (T, n) x, y, p and the cumulative count of bad input samples."""
    T, n = X.shape
    XF, YF, PF = np.empty((T, n)), np.empty((T, n)), np.empty((T, n))
    cb = np.zeros((T, n + 1), np.int32)
    for i in range(T):
        x, y, p = X[i].astype(float), Y[i].astype(float), P[i].astype(float)
        bad = ~(np.isfinite(x) & np.isfinite(y) & np.isfinite(p))
        cb[i, 1:] = np.cumsum(bad)
        XF[i], YF[i], PF[i] = dr._fill(x)[0], dr._fill(y)[0], dr._fill(p)[0]
    return lowpass_rows(XF, fs), lowpass_rows(YF, fs), lowpass_rows(PF, fs), cb


# ------------------------------------------------------------------------- 2. frame times
def frame_times(rng, T, fs, meas):
    """(T, n_gen) frame times in ms relative to display onset: measured gap blocks scaled to mean 1000/fs, random phase."""
    n_gen = int(np.ceil(5500 * fs / 1000.0 * 1.35)) + 20
    scale = (1000.0 / fs) / meas["mean_gap"]
    G = np.empty((T, n_gen))
    for i in range(T):
        s = rng.choice(len(meas["gaps"]), p=meas["gap_w"])
        g = meas["gaps"][s]
        st = int(rng.integers(0, len(g) - n_gen))
        G[i] = g[st:st + n_gen]
    G *= scale
    U = rng.random(T)
    t0 = -500.0 - U * G[:, 0]
    t = t0[:, None] + np.concatenate([np.zeros((T, 1)), np.cumsum(G[:, :-1], axis=1)], axis=1)
    assert (t[:, -1] > 5000 + 500.0 / fs).all()
    return t


def sample_frames(XF, YF, PF, cb, t_rel, fs):
    """Read the filtered signals at the frame times; returns xs, ys, ps (T, n_gen) and the missing mask (degrade's rule)."""
    T, n_gen = t_rel.shape
    n = XF.shape[1]
    idx = np.arange(n, dtype=float)
    ta = t_rel + dr.PRE_MS                      # ms from segment start = array index
    half = 500.0 / fs
    xs, ys, ps = np.empty((T, n_gen)), np.empty((T, n_gen)), np.empty((T, n_gen))
    miss = np.empty((T, n_gen), bool)
    for i in range(T):
        xs[i] = np.interp(ta[i], idx, XF[i]); ys[i] = np.interp(ta[i], idx, YF[i]); ps[i] = np.interp(ta[i], idx, PF[i])
        lo = np.clip(np.floor(ta[i] - half).astype(int), 0, n)
        hi = np.clip(np.ceil(ta[i] + half).astype(int) + 1, 0, n)
        miss[i] = (cb[i, hi] - cb[i, lo]) > 0
    return xs, ys, ps, miss


# ------------------------------------------------------------------------- 3. observed error
def ou2(rng, t_rel, par):
    """Two-component exponential (OU) process sampled at the frame times t_rel (T, n): x_k = phi x_{k-1} + sqrt(1-phi^2) s eps."""
    sig, w, t1, t2 = par["sigma_deg"], par["share_fast"], par["tau_fast_ms"], par["tau_slow_ms"]
    T, n = t_rel.shape
    e1, e2 = rng.standard_normal((n, T)), rng.standard_normal((n, T))
    s1, s2 = np.sqrt(w) * sig, np.sqrt(1 - w) * sig
    out = np.empty((T, n))
    x1, x2 = s1 * e1[0], s2 * e2[0]
    out[:, 0] = x1 + x2
    dt = np.diff(t_rel, axis=1)
    p1, p2 = np.exp(-dt / t1), np.exp(-dt / t2)
    q1, q2 = np.sqrt(1 - p1 ** 2) * s1, np.sqrt(1 - p2 ** 2) * s2
    for j in range(1, n):
        x1 = p1[:, j - 1] * x1 + q1[:, j - 1] * e1[j]
        x2 = p2[:, j - 1] * x2 + q2[:, j - 1] * e2[j]
        out[:, j] = x1 + x2
    return out


def trial_offsets(draw, pid, meas, T=64):
    """Per-trial offset (deg, x, y) and per-trial loss probability; the same at every rate (stream independent of the rate)."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(draw), int(pid), 1]))
    off = meas["off"][rng.integers(0, len(meas["off"]), T)]
    loss = meas["loss"][rng.integers(0, len(meas["loss"]), T)]
    return off, loss


def observed_frames(xs, ys, ps, miss, t_rel, off_deg, loss_p, rng, meas):
    """Apply gain, offset, jitter and loss to the sampled frames. Returns noisy x, y, p and the validity mask (T, n_gen)."""
    T, n = xs.shape
    gx, gy = GAIN
    x = C.CENTER[0] + gx * (xs - C.CENTER[0])
    y = C.CENTER[1] + gy * (ys - C.CENTER[1])
    jp = meas["jm"]
    jx = ou2(rng, t_rel, jp["x"]["params"])
    jy = ou2(rng, t_rel, jp["y"]["params"])
    x = x + (off_deg[:, 0:1] + jx) * PPD
    y = y + (off_deg[:, 1:2] + jy) * PPD
    u = rng.random((T, n))
    valid = ~miss & ~(u < loss_p[:, None])
    return x, y, ps, valid


# ------------------------------------------------------------------------- 4. correction (the app's rule)
def measure_correction(x, y, valid, t_rel, fs):
    """Per trial: (dx, dy, ok). Median of the valid frames in [-500, -100) ms; needs n_min_frames; refused when larger than 6 deg."""
    T = x.shape[0]
    dx, dy, ok = np.zeros(T), np.zeros(T), np.zeros(T, bool)
    nmin = n_min_frames(fs)
    for i in range(T):
        m = (t_rel[i] >= WIN_PRE[0]) & (t_rel[i] < WIN_PRE[1]) & valid[i]
        if m.sum() < nmin:
            continue
        a, b = C.CENTER[0] - np.median(x[i][m]), C.CENTER[1] - np.median(y[i][m])
        if np.hypot(a, b) / PPD > MAX_DRIFT_DEG:
            continue
        dx[i], dy[i], ok[i] = a, b, True
    return dx, dy, ok


# ------------------------------------------------------------------------- 5. grid + features
def to_grids(t_rel, x, y, p, valid, fs):
    """The app's to_grid per trial. app/features.py takes n = round(5500 fs / 1000) grid points; dr.degrade gives ceil (83 at 15 Hz, 82 by
    to_grid). To keep the same last grid point as the clean and measured-noise rows, POST_MS of the app module is raised by 34 ms for 15 Hz
    only during the call (module attribute; the file is not edited)."""
    import features as AF
    T = x.shape[0]
    n_g = int(np.ceil((dr.PRE_MS + dr.POST_MS) * fs / 1000.0 - 1e-9))
    old = AF.POST_MS
    AF.POST_MS = 5000 + (34 if fs == 15 else 0)
    try:
        return _to_grids(t_rel, x, y, p, valid, fs, T, n_g)
    finally:
        AF.POST_MS = old


def _to_grids(t_rel, x, y, p, valid, fs, T, n_g):
    Xg, Yg, Pg = np.full((T, n_g), np.nan), np.full((T, n_g), np.nan), np.full((T, n_g), np.nan)
    for i in range(T):
        keep = (t_rel[i] >= -dr.PRE_MS - 500.0 / fs) & (t_rel[i] < dr.POST_MS + 500.0 / fs)
        tt = t_rel[i][keep]
        _, a, b = to_grid(tt, x[i][keep], y[i][keep], valid[i][keep], fs)
        _, c, _ = to_grid(tt, p[i][keep], p[i][keep], valid[i][keep], fs)
        assert len(a) == n_g, (len(a), n_g, fs)
        Xg[i], Yg[i], Pg[i] = a, b, c
    return Xg, Yg, Pg


def features_from_grid(Xg, Yg, Pg, fs):
    """The two routes of s4_webcam.py (module attribute C1.FS set to the rate, as there)."""
    C1.FS = fs
    return {"sb": C1.sb_features_batch(Xg, Yg, Pg), "ivt": df_.trial_features_batch(Xg, Yg, Pg, fs, with_msacc=False)}


def simulate_participant(pid, X, Y, P, fs, draw, meas, pre=None):
    """Both correction variants for one (participant, rate, draw). Returns {corr: {"sb": F, "ivt": F, "diag": {...}}}."""
    if pre is None:
        pre = prepare(X, Y, P, fs)
    XF, YF, PF, cb = pre
    T = XF.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(draw), int(pid), 2, int(fs)]))
    t_rel = frame_times(rng, T, fs, meas)
    xs, ys, ps, miss = sample_frames(XF, YF, PF, cb, t_rel, fs)
    off, loss_p = trial_offsets(draw, pid, meas, T)
    x, y, p, valid = observed_frames(xs, ys, ps, miss, t_rel, off, loss_p, rng, meas)
    dx, dy, ok = measure_correction(x, y, valid, t_rel, fs)
    res = {}
    if draw == 1:               # reference: regular frame times, no error (sanity check of the timing/resampling code)
        tr = np.tile(-500.0 + np.arange(int(np.ceil(5500 * fs / 1000.0 - 1e-9)) + 3) * 1000.0 / fs, (T, 1))
        rx, ry, rp, rmiss = sample_frames(XF, YF, PF, cb, tr, fs)
        Xg, Yg, Pg = to_grids(tr, rx, ry, rp, ~rmiss, fs)
        res["regular"] = {**features_from_grid(Xg, Yg, Pg, fs), "diag": {"grid_valid_share": float(np.isfinite(Xg).mean())}}
    for corr in ("none", "app"):
        xx, yy = (x, y) if corr == "none" else (x + dx[:, None], y + dy[:, None])
        Xg, Yg, Pg = to_grids(t_rel, xx, yy, p, valid, fs)
        diag = {"grid_valid_share": float(np.isfinite(Xg).mean()), "check_ok_share": float(ok.mean()) if corr == "app" else float("nan"),
                "mean_applied_offset_deg": float(np.hypot(dx[ok], dy[ok]).mean() / PPD) if ok.any() else float("nan")}
        res[corr] = {**features_from_grid(Xg, Yg, Pg, fs), "diag": diag}
    return res


def table_for(F_by_pid, route, fs, bk):
    """F_by_pid: {pid: (F, images, group)} -> the s4 table (C1.make_table with the rate set)."""
    C1.FS = fs
    names = C1.dr.TRIAL_FEATURES if route == "ivt" else C1.SB_NAMES
    return C1.make_table(F_by_pid, names, bk)
