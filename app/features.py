"""Features per trial and per item, with the paper's names and time windows (src/edd/delmas_raw.py, scripts/phase6_trackC_c1.py).

Gaze trials are time-aligned to display onset with the same stored window as the paper (500 ms before, 5000 ms after) and
resampled to a uniform 30 Hz grid before feature extraction, so the paper's sample-based and I-VT definitions apply unchanged.
No pupil features (a webcam cannot measure pupil size reliably).

Sample-based (no event detection)        I-VT (velocity threshold 30 deg/s, min fixation 60 ms)
  dwell_1, dwell_2, dwell_b1..b5           dwell_1, dwell_2, dwell_b1..b5   (samples that are fixation samples)
  n_visit, first_fix, lat_first            n_fix, n_visit, first_fix, lat_first
Secondary channel (not in the frozen model): nearest-face dwell, `nearest_dwell` (each valid on-screen sample goes to the nearest
card centre, samples within 3 deg of the screen centre count as "centre"), windows 0.2-1 s and 1-5 s.
Reaction-time features: rt_ms, log_rt. Blink features: n_blinks, blink_rate_hz per trial, blink suppression per item.
Face item sets (memory task) add the RT to the test face when it is the probe versus an irrelevant (`memory_channels`) and the
memorisation measures of the "was this face one of the four?" answers (`memory_performance`: hit rate, false-alarm rate, d-prime,
criterion, percent correct, median RT).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRE_MS, POST_MS, GRID_FS = 500, 5000, 30
PHASE1, PHASE2 = (200, 1000), (1000, 5000)
BINS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3500), (3500, 5000)]
VEL_THR, MIN_FIX_MS = 30.0, 60.0
SB_NAMES = ["dwell_1", "dwell_2", "n_visit", "first_fix", "lat_first", "dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4", "dwell_b5"]
IVT_NAMES = ["dwell_1", "dwell_2", "n_fix", "n_visit", "first_fix", "lat_first", "dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4",
             "dwell_b5"]
MIN_RT_MS = 150.0
NN_CENTRE_DEG = 3.0                   # nearest-face channel: samples this close to the screen centre are "centre", not a face
MEMORY_RT_TIMEOUT_MS = 5000.0         # answer time of the memory question (face item sets)


# ----------------------------------------------------------------------------- blinks
def detect_blinks(t_ms: np.ndarray, ear: np.ndarray, face: np.ndarray, thr: float, min_frames: int = 2,
                  max_ms: float = 600.0) -> np.ndarray:
    """Blink onsets and durations (ms) from the eye-aspect-ratio trace: EAR below `thr` for at least `min_frames` consecutive
    frames and at most `max_ms` (longer closures are treated as gaze aversion / lost tracking, not blinks)."""
    low = face.astype(bool) & np.isfinite(ear) & (ear < thr)
    d = np.diff(np.concatenate([[0], low.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = []
    for a, b in zip(st, en):
        dur = t_ms[b - 1] - t_ms[a] + 1000.0 / GRID_FS
        if b - a >= min_frames and dur <= max_ms:
            out.append((t_ms[a], dur))
    return np.array(out, float).reshape(-1, 2)


def ear_threshold(ear: np.ndarray, face: np.ndarray, frac: float = 0.7) -> float:
    e = ear[face.astype(bool) & np.isfinite(ear)]
    return float(frac * np.percentile(e, 85)) if len(e) >= 30 else float("nan")


# ----------------------------------------------------------------------------- grid + AOI helpers
def to_grid(t_rel: np.ndarray, x: np.ndarray, y: np.ndarray, valid: np.ndarray, fs: int = GRID_FS):
    """Nearest-sample resampling onto the uniform grid t = -PRE_MS + k * 1000/fs (k = 0..n-1); a grid point with no valid sample
    within half a period is missing (NaN)."""
    n = int(round((PRE_MS + POST_MS) * fs / 1000.0))
    g = -PRE_MS + np.arange(n) * 1000.0 / fs
    xg, yg = np.full(n, np.nan), np.full(n, np.nan)
    if len(t_rel):
        j = np.clip(np.searchsorted(t_rel, g), 1, len(t_rel) - 1) if len(t_rel) > 1 else np.zeros(n, int)
        if len(t_rel) > 1:
            j = np.where(np.abs(t_rel[j - 1] - g) <= np.abs(t_rel[j] - g), j - 1, j)
        near = np.abs(t_rel[j] - g) <= 500.0 / fs
        ok = near & valid[j]
        xg[ok], yg[ok] = x[j][ok], y[j][ok]
    return g, xg, yg


def aoi_index(x: np.ndarray, y: np.ndarray, rects: np.ndarray) -> np.ndarray:
    q = np.full(len(x), -1)
    for k, (x0, y0, x1, y1) in enumerate(rects):
        q[(x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)] = k
    return q


def _window_dwell(q, valid, t, win, k):
    m = (t >= win[0]) & (t < win[1])
    nv = valid[m].sum()
    return float((q[m] == k).sum() / nv) if nv >= 0.2 * m.sum() and nv > 0 else float("nan")


# ----------------------------------------------------------------------------- sample-based features
def sb_features(xg: np.ndarray, yg: np.ndarray, rects: np.ndarray, fs: int = GRID_FS) -> np.ndarray:
    """(K, len(SB_NAMES)) sample-based features of one trial on the grid: dwell = share of VALID samples in the AOI over a
    window (needs >= 20% valid samples), first_fix / lat_first from the first sample inside any AOI, n_visit = entries into
    the AOI (gaps skipped). Same logic as `sb_features_batch` of scripts/phase6_trackC_c1.py without the pupil columns."""
    n, K = len(xg), len(rects)
    t = -PRE_MS + np.arange(n) * 1000.0 / fs
    valid = np.isfinite(xg) & np.isfinite(yg)
    q = aoi_index(xg, yg, rects)
    F = np.full((K, len(SB_NAMES)), np.nan)
    wins = {"dwell_1": PHASE1, "dwell_2": PHASE2, **{f"dwell_b{j + 1}": b for j, b in enumerate(BINS)}}
    cv = np.r_[0, np.cumsum(valid)]                                       # cumulative counts make every window O(1)
    ck = np.vstack([np.zeros(K), np.cumsum(q[:, None] == np.arange(K)[None, :], axis=0)])
    for name, w in wins.items():
        c = SB_NAMES.index(name)
        i0, i1 = int(np.searchsorted(t, w[0], "left")), int(np.searchsorted(t, w[1], "left"))
        nv = cv[i1] - cv[i0]
        if nv >= 0.2 * (i1 - i0) and nv > 0:
            F[:, c] = (ck[i1] - ck[i0]) / nv
    win = (t >= 0) & (t < POST_MS)
    tw, qw, vw = t[win], q[win], valid[win]
    lat = np.full(K, float(POST_MS))
    for k in range(K):
        hit = np.flatnonzero(qw == k)
        if len(hit):
            lat[k] = tw[hit[0]]
    F[:, SB_NAMES.index("lat_first")] = lat
    has = lat.min() < POST_MS
    F[:, SB_NAMES.index("first_fix")] = ((lat == lat.min()) & has).astype(float)
    # entries: a valid sample in AOI k whose previous valid sample (gaps skipped) was elsewhere / none
    qv = qw[vw]
    enter = (qv >= 0) & (qv != np.r_[-2, qv[:-1]])
    nvis = np.bincount(qv[enter], minlength=K).astype(float)
    F[:, SB_NAMES.index("n_visit")] = nvis
    return F


# ----------------------------------------------------------------------------- nearest-face dwell (secondary channel)
def nearest_dwell(xg: np.ndarray, yg: np.ndarray, rects: np.ndarray, screen_px: tuple, ppd: float,
                  centre_deg: float = NN_CENTRE_DEG, fs: int = GRID_FS) -> np.ndarray:
    """(K, 2) nearest-face dwell of one trial on the grid, windows 0.2-1 s and 1-5 s. A valid ON-SCREEN sample (finite, inside the
    screen) is assigned to the card whose centre is nearest, unless it lies within `centre_deg` of the screen centre, where it
    counts as "centre" (no card). Dwell = samples assigned to the card / valid on-screen samples of the window (centre samples
    included in the denominator); NaN when fewer than 20% of the window's grid points are valid on-screen samples (as for the
    AOI dwell). Unlike the AOI dwell, a look that lands between the cards (inside the gaps, in the webcam's error margin) still
    goes to the nearest face. It is a SECONDARY channel: the frozen-model statistic stays primary."""
    n, K = len(xg), len(rects)
    t = -PRE_MS + np.arange(n) * 1000.0 / fs
    W, H = screen_px
    with np.errstate(invalid="ignore"):
        on = np.isfinite(xg) & np.isfinite(yg) & (xg >= 0) & (xg <= W) & (yg >= 0) & (yg <= H)
        r = np.asarray(rects, float)
        cx, cy = (r[:, 0] + r[:, 2]) / 2, (r[:, 1] + r[:, 3]) / 2
        d = np.hypot(xg[:, None] - cx[None, :], yg[:, None] - cy[None, :])
        near = np.argmin(np.where(on[:, None], d, np.inf), axis=1)
        centre = np.hypot(xg - W / 2, yg - H / 2) <= centre_deg * ppd
    lab = np.where(on & ~centre, near, -1)
    F = np.full((K, 2), np.nan)
    for j, w in enumerate((PHASE1, PHASE2)):
        m = (t >= w[0]) & (t < w[1])
        nv = int(on[m].sum())
        if nv > 0 and nv >= 0.2 * m.sum():
            F[:, j] = [(lab[m] == k).sum() / nv for k in range(K)]
    return F


# ----------------------------------------------------------------------------- I-VT features
def _fill(v):
    v = v.astype(float).copy()
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.zeros_like(v), ok
    i = np.arange(len(v))
    v[~ok] = np.interp(i[~ok], i[ok], v[ok])
    return v, ok


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def ivt(x, y, fs, ppd, center, vel_thr=VEL_THR, min_fix_ms=MIN_FIX_MS):
    """I-VT as in `edd.delmas_raw.ivt`: gaze in degrees, fixed +-4 ms central difference after gap interpolation; samples in or
    next to gaps are not fixation samples. Returns fixation mask, run starts and ends (sample indices)."""
    xd, okx = _fill((x - center[0]) / ppd)
    yd, _ = _fill((y - center[1]) / ppd)
    ok = okx & np.isfinite(y)
    h = max(1, int(round(fs * 0.004)))
    n = len(xd)
    i = np.arange(n)
    lo, hi = np.clip(i - h, 0, n - 1), np.clip(i + h, 0, n - 1)
    span = (hi - lo) / fs
    vel = np.hypot((xd[hi] - xd[lo]) / span, (yd[hi] - yd[lo]) / span)
    okn = ok & np.concatenate([[True], ok[:-1]]) & np.concatenate([ok[1:], [True]])
    fixs = (vel < vel_thr) & okn
    st, en = _runs(fixs)
    keep = (en - st) * 1000.0 / fs >= min_fix_ms
    fid_mask = np.zeros(n, bool)
    for a, b in zip(st[keep], en[keep]):
        fid_mask[a:b] = True
    return fid_mask, st[keep], en[keep]


def ivt_features(xg, yg, rects, ppd, center, fs: int = GRID_FS) -> np.ndarray:
    """(K, len(IVT_NAMES)) I-VT features of one trial on the grid (paper's `trial_features` without pupil and microsaccades)."""
    n, K = len(xg), len(rects)
    t = -PRE_MS + np.arange(n) * 1000.0 / fs
    dt = 1000.0 / fs
    fixs, st, en = ivt(xg, yg, fs, ppd, center)
    q = aoi_index(xg, yg, rects)
    F = np.full((K, len(IVT_NAMES)), np.nan)
    if len(st):
        fx = np.array([np.nanmean(xg[a:b]) for a, b in zip(st, en)])
        fy = np.array([np.nanmean(yg[a:b]) for a, b in zip(st, en)])
        fq, ft = aoi_index(fx, fy, rects), t[st]
        inwin = (ft >= 0) & (ft < POST_MS)
        fq_w, ft_w = fq[inwin], ft[inwin]
    else:
        fq_w, ft_w = np.array([], int), np.array([])
    vis, prev = np.zeros(K), -2
    for a in fq_w:
        if a >= 0 and a != prev:
            vis[a] += 1
        prev = a
    first = fq_w[fq_w >= 0][0] if (fq_w >= 0).any() else -1
    c = {nm: i for i, nm in enumerate(IVT_NAMES)}
    for k in range(K):
        onk = fixs & (q == k)
        F[k, c["dwell_1"]] = onk[(t >= PHASE1[0]) & (t < PHASE1[1])].sum() * dt / (PHASE1[1] - PHASE1[0])
        F[k, c["dwell_2"]] = onk[(t >= PHASE2[0]) & (t < PHASE2[1])].sum() * dt / (PHASE2[1] - PHASE2[0])
        F[k, c["n_fix"]] = np.sum(fq_w == k)
        F[k, c["n_visit"]] = vis[k]
        F[k, c["first_fix"]] = float(first == k)
        F[k, c["lat_first"]] = ft_w[fq_w == k][0] if np.any(fq_w == k) else float(POST_MS)
        for j, (a, b) in enumerate(BINS):
            F[k, c[f"dwell_b{j + 1}"]] = onk[(t >= a) & (t < b)].sum() * dt / (b - a)
    return F


# ----------------------------------------------------------------------------- look-away rule
def onscreen_valid_share(obs: list, screen_px: tuple, margin_px: float, drift: tuple, ear_thr: float, pitch_ref: float,
                         pitch_max_deg: float, n_expected: float) -> float:
    """Share of the frames expected during a display that carry a valid ON-SCREEN sample: face found, eyes open, a finite gaze
    estimate (plus the trial's drift correction) inside the screen with `margin_px` to spare, and head pitch within
    `pitch_max_deg` of the calibration median `pitch_ref` (either direction: the sign of `pitch` depends on the face model, a head
    turned far up or down is looking away all the same). This is THE definition of a look-away trial: the session recorder flags a
    trial `lookaway` when the share is below cfg.min_trial_valid_share, and the report's usable-trial rate uses that flag."""
    W, H = screen_px
    n = 0
    for o in obs:
        if not o.face or not (np.isfinite(o.gaze_x) and np.isfinite(o.gaze_y)):
            continue
        if np.isfinite(ear_thr) and o.ear < ear_thr:
            continue
        x, y = o.gaze_x + drift[0], o.gaze_y + drift[1]
        if not (-margin_px <= x <= W + margin_px and -margin_px <= y <= H + margin_px):
            continue
        if np.isfinite(pitch_ref) and np.isfinite(o.pitch) and abs(o.pitch - pitch_ref) > pitch_max_deg:
            continue
        n += 1
    return float(n / max(float(n_expected), float(len(obs)), 1.0))


# ----------------------------------------------------------------------------- session -> trial gaze
def trial_gaze(frames: pd.DataFrame, trial: pd.Series, gaze_model, blink_thr: float):
    """Aligned samples of one trial: (t_rel_ms, x_px, y_px, valid), drift-corrected. `valid` = face found, eyes open, finite
    estimate. The gaze comes from the saved calibration applied to the recorded landmark features."""
    t0 = float(trial["onset_ms"])
    f = frames[(frames.t_ms >= t0 - PRE_MS - 50) & (frames.t_ms <= t0 + POST_MS + 50)]
    if len(f) == 0 or gaze_model is None:
        return (np.zeros(0),) * 3 + (np.zeros(0, bool),)
    from calibrate import GazeModel
    m = gaze_model if isinstance(gaze_model, GazeModel) else GazeModel.from_dict(gaze_model)
    from landmarks import FEATURE_NAMES
    feat = f[["f_" + n for n in FEATURE_NAMES]].to_numpy(float)
    with np.errstate(all="ignore"):
        xy = m.predict(np.where(np.isfinite(feat), feat, 0.0))
    xy[:, 0] += float(trial.get("drift_dx_px", 0.0) or 0.0)
    xy[:, 1] += float(trial.get("drift_dy_px", 0.0) or 0.0)
    face = f.face.to_numpy(bool) & np.isfinite(feat).all(1)
    blink = np.isfinite(blink_thr) & (f.ear.to_numpy() < blink_thr)
    valid = face & ~blink
    return f.t_ms.to_numpy() - t0, xy[:, 0], xy[:, 1], valid


def trial_rows(session, layout_rects: np.ndarray | None = None, ppd: float | None = None) -> pd.DataFrame:
    """One row per (trial, option): identifiers, roles, gaze features (sample-based and I-VT), RT, blink counts."""
    fr, tr, meta = session.frames, session.scored_trials, session.meta        # look-away trials are not scored
    thr = ear_threshold(fr.ear.to_numpy(), fr.face.to_numpy())
    blinks = detect_blinks(fr.t_ms.to_numpy(), fr.ear.to_numpy(), fr.face.to_numpy(), thr)
    center = np.array(meta.get("screen_px", [1920, 1080]), float) / 2
    rows = []
    # session-level baseline blink rate over all trial windows
    win = []
    for _, t in tr.iterrows():
        a, b = _blink_window(t)
        win.append((a, b))
    nb = [int(((blinks[:, 0] >= a) & (blinks[:, 0] < b)).sum()) if len(blinks) else 0 for a, b in win]
    base_rate = float(np.sum(nb) / max(np.sum([(b - a) / 1000.0 for a, b in win]), 1e-9)) if win else float("nan")
    for (_, t), (a, b), n_b in zip(tr.iterrows(), win, nb):
        dur = max((b - a) / 1000.0, 1e-3)
        common = {"session_id": meta["session_id"], "trial": int(t["trial"]), "question_id": t["question_id"],
                  "mode": t["mode"], "rt_ms": t.get("rt_ms", np.nan), "correct": t.get("correct", np.nan),
                  "n_blinks": n_b, "blink_rate_hz": n_b / dur, "blink_supp_hz": base_rate - n_b / dur,
                  "drift_deg": t.get("drift_deg", np.nan), "fixation_ok": t.get("fixation_ok", np.nan),
                  "test_is_probe": t.get("test_is_probe", np.nan), "test_in_display": t.get("test_in_display", np.nan),
                  "pool": t.get("pool", np.nan)}
        if t["mode"] == "A":
            model_t, rects, ppd_t = session.segment_context(t)       # model / layout of the calibration segment of this trial
            if layout_rects is not None:
                rects = np.asarray(layout_rects, float)
            ppd_t = ppd or ppd_t
            tt, x, y, v = trial_gaze(fr, t, model_t, thr)
            _, xg, yg = to_grid(tt, x, y, v)
            Fsb = sb_features(xg, yg, rects)
            Fiv = ivt_features(xg, yg, rects, ppd_t, center)
            Fnn = nearest_dwell(xg, yg, rects, tuple(center * 2), ppd_t)
            valid_frac = float(np.isfinite(xg[(np.arange(len(xg)) * 1000.0 / GRID_FS - PRE_MS) >= 0]).mean())
            for k in range(len(rects)):
                rows.append({**common, "quad": k, "item_id": t[f"item{k}"], "role": t[f"role{k}"], "valid_frac": valid_frac,
                             **{f"sb_{n}": Fsb[k, i] for i, n in enumerate(SB_NAMES)},
                             **{f"ivt_{n}": Fiv[k, i] for i, n in enumerate(IVT_NAMES)},
                             "nn_dwell_1": Fnn[k, 0], "nn_dwell_2": Fnn[k, 1]})
        else:
            rows.append({**common, "quad": -1, "item_id": t["item0"], "role": t["role0"], "valid_frac": np.nan})
    d = pd.DataFrame(rows)
    d["log_rt"] = np.log(d.rt_ms.astype(float).where(d.rt_ms.astype(float) > 0))
    return d


def _blink_window(t: pd.Series) -> tuple[float, float]:
    a = float(t["onset_ms"])
    if t["mode"] == "A":
        return a, a + POST_MS
    rt = t.get("rt_ms", np.nan)
    return a, a + (float(rt) if np.isfinite(rt) else 3000.0)


def rt_usable(d: pd.DataFrame, timeout_ms: float = 4000.0) -> pd.Series:
    """Trials whose RT is used: a response was given, within [150 ms, timeout], and (when an expected answer is defined) correct."""
    rt = d.rt_ms.astype(float)
    ok = rt.between(MIN_RT_MS, timeout_ms)
    c = d["correct"]
    return ok & (c.isna() | (c.astype(float) != 0.0))


def memory_channels(rows: pd.DataFrame, timeout_ms: float = MEMORY_RT_TIMEOUT_MS) -> dict:
    """Reaction-time channel of the memory question (face item sets): RT to the test face when it is the probe versus when it is an
    irrelevant, over correct answers within [150 ms, timeout]. The plan tests the probe once with yes and once with no and each
    irrelevant equally often, so the two means are balanced for the answer. Empty when the session has no memory task."""
    if "test_is_probe" not in rows or not (rows.quad >= 0).any():
        return {}
    t = rows[rows.quad >= 0].drop_duplicates("trial")
    t = t[t.test_is_probe.notna()]
    if t.empty:
        return {}
    use = rt_usable(t, timeout_ms)
    rt = t.rt_ms.astype(float)
    rp, ri = rt[use & (t.test_is_probe == 1)], rt[use & (t.test_is_probe == 0)]
    out = {"memory_rt_probe_ms": float(rp.mean()) if len(rp) else float("nan"),
           "memory_rt_irrelevant_ms": float(ri.mean()) if len(ri) else float("nan"),
           "memory_rt_diff_ms": float("nan"), "memory_rt_z": float("nan"), "memory_n_probe_tests": int(len(rp))}
    if len(rp) >= 1 and len(ri) >= 2:
        out["memory_rt_diff_ms"] = float(rp.mean() - ri.mean())
        out["memory_rt_z"] = float((rp.mean() - ri.mean()) / (ri.std(ddof=1) + 1e-9))
    return out


def dprime_c(hits: int, n_yes: int, fas: int, n_no: int) -> tuple[float, float]:
    """Sensitivity d' = z(H) - z(F) and criterion c = -(z(H) + z(F)) / 2 with the log-linear correction (Hautus 1995): 0.5 is added to
    the hit and false-alarm counts and 1 to the numbers of signal and noise trials, always, so rates of 0 and 1 give finite values.
    (Positive c = a conservative bias towards "no".) NaN when there are no signal or no noise trials."""
    from statistics import NormalDist
    if n_yes < 1 or n_no < 1:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf
    zh, zf = z((hits + 0.5) / (n_yes + 1)), z((fas + 0.5) / (n_no + 1))
    return float(zh - zf), float(-0.5 * (zh + zf))


def _memory_block(t: pd.DataFrame) -> dict:
    """Memory measures of the answered trials in `t`: hit rate (yes when the test face was in the display), false-alarm rate (yes when
    it was not), d', c (log-linear), percent correct and the median RT of correct answers (ms)."""
    nan = float("nan")
    a = t[t.response.isin(["yes", "no"])]
    sig, noi = a[a.test_in_display == 1], a[a.test_in_display == 0]
    hits, fas = int((sig.response == "yes").sum()), int((noi.response == "yes").sum())
    d, c = dprime_c(hits, len(sig), fas, len(noi))
    ok = a[a.response == np.where(a.test_in_display == 1, "yes", "no")]
    rt = ok.rt_ms.astype(float)
    return {"hit": hits / len(sig) if len(sig) else nan, "fa": fas / len(noi) if len(noi) else nan, "dprime": d, "c": c,
            "pc": 100 * len(ok) / len(a) if len(a) else nan, "rt_med": float(rt.median()) if rt.notna().any() else nan,
            "n": int(len(t)), "n_answered": int(len(a)), "n_signal": int(len(sig)), "n_noise": int(len(noi))}


def memory_performance(trials: pd.DataFrame) -> dict:
    """How well the person memorised the faces, from the answers to "was this test face one of the four?" in trials.csv (first
    presentations only: a repeated look-away trial is not a second answer; look-away trials count, memory does not need the gaze).
    Unanswered trials (timeout) are left out of every measure and counted (`mem_n_unanswered`). Keys `mem_hit`, `mem_fa`, `mem_dprime`,
    `mem_c`, `mem_pc` (percent), `mem_rt_med` (ms) overall and `mem_p1_*` / `mem_p2_*` for test pool 1 / 2 (when trials.csv has `pool`).
    Empty when the session has no memory task."""
    if "test_in_display" not in trials or "response" not in trials:
        return {}
    t = trials[trials.test_in_display.notna()]
    if "is_repeat" in t:
        t = t[t.is_repeat.fillna(0).astype(int) == 0]
    t = t.assign(rt_ms=t.rt_ms.astype(float))
    out = {f"mem_{k}": v for k, v in _memory_block(t).items()}
    out["mem_n_unanswered"] = out["mem_n"] - out["mem_n_answered"]
    if "pool" in t and t.pool.notna().any():
        for k in sorted(int(x) for x in t.pool.dropna().unique()):
            out.update({f"mem_p{k}_{m}": v for m, v in _memory_block(t[t.pool == k]).items()})
    return out


def item_table(rows: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Average the per-presentation features of each item (the paper's per-image aggregation). Targets and non-gaze rows are
    kept in the table but flagged by role; scoring drops them."""
    g = rows.groupby(["item_id", "role"], sort=False)
    out = g[feature_cols].mean().reset_index()
    out["n_pres"] = g.size().to_numpy()
    return out


def person_channels(rows: pd.DataFrame, timeout_ms: float = 4000.0) -> dict:
    """No-model person-level contrasts (probe/relevant minus irrelevant), the pre-specified channel scores.
      rt_diff_ms        mean RT probe - mean RT irrelevant (the paper's no-model RT benchmark, A7)
      rt_z              the same difference in SDs of the irrelevant trial RTs
      dwell_diff        mean dwell_2 probe - irrelevant (sample-based, raw sign: positive = looked MORE at the probe)
      blink_supp_diff   irrelevant blink rate - probe blink rate (positive = fewer blinks at the probe)
      nearest_dwell_early_diff / nearest_dwell_late_diff   the secondary nearest-face dwell (see `nearest_dwell`), probe minus
                        irrelevant, windows 0.2-1 s and 1-5 s (raw sign: positive = more of the gaze went to the probe's face);
                        NaN when there is no probe or no parallel display"""
    d = rows.copy()
    d["role"] = d.role.replace({"relevant": "probe"})
    pr, ir = d[d.role == "probe"], d[d.role == "irrelevant"]
    out = {"nearest_dwell_early_diff": float("nan"), "nearest_dwell_late_diff": float("nan")}
    if (d.quad >= 0).any():        # parallel display: RT and blinks belong to the trial, not to an option -> no item contrast
        if "sb_dwell_2" in d:
            a, b = pr.sb_dwell_2.dropna(), ir.sb_dwell_2.dropna()
            out["dwell_diff"] = float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")
        for col, name in (("nn_dwell_1", "nearest_dwell_early_diff"), ("nn_dwell_2", "nearest_dwell_late_diff")):
            if col in d:
                a, b = pr[col].dropna(), ir[col].dropna()
                out[name] = float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")
        out.update(rt_diff_ms=float("nan"), rt_z=float("nan"), log_rt_diff=float("nan"), blink_supp_diff=float("nan"),
                   blink_z=float("nan"))
        out.update(memory_channels(d))       # the memory question's RT contrast is reported on its own, not in rt_z / combined_z
        return out
    use = rt_usable(d, timeout_ms)
    rp, ri = pr.rt_ms.astype(float)[use[pr.index]], ir.rt_ms.astype(float)[use[ir.index]]
    if len(rp) >= 2 and len(ri) >= 2:
        out["rt_diff_ms"] = float(rp.mean() - ri.mean())
        out["rt_z"] = float((rp.mean() - ri.mean()) / (ri.std(ddof=1) + 1e-9))
        out["log_rt_diff"] = float(np.log(rp).mean() - np.log(ri).mean())
    else:
        out["rt_diff_ms"] = out["rt_z"] = out["log_rt_diff"] = float("nan")
    if len(pr) and len(ir):
        bs = ir.blink_rate_hz.mean() - pr.blink_rate_hz.mean()
        out["blink_supp_diff"] = float(bs)
        sd = np.concatenate([ir.blink_rate_hz.to_numpy()]).std(ddof=1) if len(ir) > 1 else np.nan
        out["blink_z"] = float(bs / (sd + 1e-9)) if np.isfinite(sd) else float("nan")
    return out
