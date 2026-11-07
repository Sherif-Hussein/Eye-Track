"""Raw-sample pipeline for Delmas et al. (2024): EyeLink EDF -> parallel-display segments -> sensor-rate
degradation -> I-VT events -> per-image features (the paper's four plus new ones).

Layout (from the authors' Psychtoolbox design files `FI<id>c2_<half>.mat`): four 300 x 400 px faces centred
at (640,270), (1280,270), (640,810), (1280,810) on a 1920 x 1080 screen, 34.99 px/deg. `idcible` is the face
at `ciblePosition`, `distractors[k]` at `dist{k+1}Position`. Display onset = message `EVENT_ParallelDisplay`.

Everything here works on one participant at a time and uses only that participant's samples, so it is
leakage-free by construction (no labels are touched).
"""
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

PRE_MS, POST_MS = 500, 5000          # stored window around parallel-display onset
PIX_PER_DEG = 34.9868554165592
CENTER = np.array([960.0, 540.0])
QUAD_RECTS = np.array([[490, 70, 790, 470], [1130, 70, 1430, 470],
                       [490, 610, 790, 1010], [1130, 610, 1430, 1010]], float)
VEL_THR = 30.0        # deg/s, the EyeLink parser threshold reported in the paper
MIN_FIX_MS = 60.0
PHASE1, PHASE2 = (200, 1000), (1000, 5000)


# ----------------------------------------------------------------------------- loading
def _quad_of_rect(rect):
    d = np.abs(QUAD_RECTS - np.asarray(rect, float)).sum(1)
    return int(np.argmin(d))


def read_edf_fast(edf_path):
    """Minimal EDF reader on SR Research's edfapi (bundled with the `eyelinkio` wheel): one ctypes pass,
    monocular gaze x/y and pupil per sample, messages, and EyeLink-parser fixations. eyelinkio's own reader
    makes two Python-level passes with per-field marshalling and is ~20x slower on these 15-min files."""
    import ctypes as ct

    from eyelinkio.edf import _edf2py as E

    err = ct.c_int(0)
    edf = E.edf_open_file(str(edf_path).encode(), 0, 1, 1, ct.byref(err))
    if not edf:
        raise OSError(f"cannot open {edf_path} ({err.value})")
    h = ct.cast(edf, ct.c_void_p).value
    gn = ct.CFUNCTYPE(ct.c_int, ct.c_void_p)(("edf_get_next_data", E.edfapi))
    gf = ct.CFUNCTYPE(ct.c_void_p, ct.c_void_p)(("edf_get_float_data", E.edfapi))
    S = E.FSAMPLE
    size = ct.sizeof(S)
    cap = 1 << 20
    raw = np.empty(cap * size, np.uint8)
    n = 0
    msgs, fix = [], []
    memmove = ct.memmove
    base = raw.ctypes.data
    # edf_get_float_data costs ~20 us per call, so samples are only fetched while a trial's parallel
    # display (or the fixation period just before it, which holds the pre-onset baseline) is pending
    gate, t_stop = False, None
    try:
        while True:
            ty = gn(h)
            if ty == 0:
                break
            if ty == 200:
                if not gate:
                    continue
                if n == cap:
                    cap *= 2
                    raw = np.resize(raw, cap * size)
                    base = raw.ctypes.data
                a = gf(h)
                memmove(base + n * size, a, size)
                n += 1
                if t_stop is not None and ct.c_uint.from_address(a).value >= t_stop:
                    gate, t_stop = False, None
            elif ty == 24 or ty == 8:
                e = ct.cast(gf(h), ct.POINTER(E.ALLF_DATA)).contents.fe
                if ty == 24:
                    txt = ct.string_at(ct.byref(e.message[0]), e.message.contents.len + 1)[2:]
                    txt = txt.split(b"\x00")[0]
                    msgs.append((int(e.sttime), txt))
                    if txt.startswith(b"EVENT_FixationDot2"):
                        gate = True
                    elif txt.startswith(b"EVENT_ParallelDisplay"):
                        gate, t_stop = True, int(e.sttime) + POST_MS
                else:
                    fix.append((int(e.sttime), int(e.entime), float(e.gavx), float(e.gavy)))
    finally:
        E.edf_close_file(edf)
    dt = np.dtype({"names": ["time", "gx", "gy", "pa"], "formats": ["<u4", ("<f4", 2), ("<f4", 2), ("<f4", 2)],
                   "offsets": [S.time.offset, S.gx.offset, S.gy.offset, S.pa.offset], "itemsize": size})
    rec = raw[:n * size].view(dt)
    gx, gy, pa = (rec[k].astype(np.float64) for k in ("gx", "gy", "pa"))
    valid = (np.abs(gx) < 1e7) & (gx > -30000)
    eye = int(np.argmax(valid.sum(0)))                      # monocular recording: the populated eye
    v = np.column_stack([gx[:, eye], gy[:, eye], pa[:, eye]])
    v[(v >= 1e7) | (v <= -30000)] = np.nan
    return rec["time"].astype(np.int64), v, msgs, np.array(fix, float).reshape(-1, 4)


def load_session(edf_path, design_path):
    """Return dict with x, y, pupil arrays (n_trials, PRE_MS+POST_MS) at 1000 Hz (NaN = missing),
    `images` (n_trials, 4): image id shown in each quadrant, and `el_nfix` (n_trials, 4): EyeLink-parser
    fixations per face (onset within the 5-s display)."""
    import scipy.io as sio

    times, samp, msgs, fix = read_edf_fast(edf_path)
    onsets = [t for t, m in msgs if m.startswith(b"EVENT_ParallelDisplay")]
    design = sio.loadmat(str(design_path), squeeze_me=True, struct_as_record=False)["design"]
    trials = np.atleast_1d(design.trial)
    if len(onsets) != len(trials):
        raise ValueError(f"{edf_path}: {len(onsets)} onsets vs {len(trials)} design trials")
    n = PRE_MS + POST_MS
    X = np.full((len(trials), 3, n), np.nan)
    images = np.zeros((len(trials), 4), int)
    el = np.zeros((len(trials), 4))
    for k, (t0, tr) in enumerate(zip(onsets, trials)):
        a = int(np.searchsorted(times, t0 - PRE_MS))
        b = int(np.searchsorted(times, t0 + POST_MS))
        idx = times[a:b] - (t0 - PRE_MS)
        X[k][:, idx] = samp[a:b].T
        f = fix[(fix[:, 0] >= t0) & (fix[:, 0] < t0 + POST_MS)]
        q = aoi_of(f[:, 2], f[:, 3])
        el[k] = [np.sum(q == j) for j in range(4)]
        images[k, _quad_of_rect(tr.ciblePosition)] = int(tr.idcible)
        for j, d in enumerate(np.atleast_1d(tr.distractors)):
            images[k, _quad_of_rect(getattr(tr, f"dist{j + 1}Position"))] = int(d)
    x, y, p = X[:, 0], X[:, 1], X[:, 2]
    bad = ~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(p) | (p <= 0)
    x[bad] = y[bad] = p[bad] = np.nan
    return {"x": x.astype(np.float32), "y": y.astype(np.float32), "p": p.astype(np.float32), "images": images,
            "el_nfix": el}


# ----------------------------------------------------------------------------- resampling
def _fill(v):
    """Linear interpolation over NaN gaps (edges held)."""
    v = v.astype(np.float64).copy()
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.zeros_like(v), ok
    idx = np.arange(len(v))
    v[~ok] = np.interp(idx[~ok], idx[ok], v[ok])
    return v, ok


def degrade(x, y, p, fs):
    """Anti-alias filter (polyphase FIR, Kaiser window) and resample one trial from 1000 Hz to `fs`.
    Gaps are interpolated before filtering; an output sample is marked missing if any input sample
    within half an output period of it was missing."""
    if fs == 1000:
        return x.astype(np.float64), y.astype(np.float64), p.astype(np.float64)
    fr = Fraction(fs, 1000).limit_denominator(1000)
    out = []
    ok_all = np.isfinite(x) & np.isfinite(y) & np.isfinite(p)
    for v in (x, y, p):
        f, _ = _fill(v)
        out.append(resample_poly(f, fr.numerator, fr.denominator, padtype="line"))
    m = len(out[0])
    t_out = np.arange(m) * 1000.0 / fs                        # ms from segment start
    half = 500.0 / fs
    bad = ~ok_all
    cbad = np.concatenate([[0], np.cumsum(bad)])
    lo = np.clip(np.floor(t_out - half).astype(int), 0, len(x))
    hi = np.clip(np.ceil(t_out + half).astype(int) + 1, 0, len(x))
    miss = (cbad[hi] - cbad[lo]) > 0
    for v in out:
        v[miss] = np.nan
    return out[0], out[1], out[2]


# ----------------------------------------------------------------------------- events
def _runs(mask):
    """Start/end (exclusive) indices of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def ivt(x, y, fs, vel_thr=VEL_THR, min_fix_ms=MIN_FIX_MS):
    """I-VT on gaze in degrees. Velocity by central difference after gap interpolation; samples in or
    adjacent to gaps are not fixation samples. Returns velocity (deg/s), per-sample fixation id (-1 = none)."""
    xd, okx = _fill((x - CENTER[0]) / PIX_PER_DEG)
    yd, _ = _fill((y - CENTER[1]) / PIX_PER_DEG)
    ok = okx & np.isfinite(y)
    # fixed-time differentiator: +-4 ms span (h samples each side, at least one), so the velocity
    # filter has the same temporal meaning at every sensor rate
    h = max(1, int(round(fs * 0.004)))
    n = len(xd)
    i = np.arange(n)
    lo, hi = np.clip(i - h, 0, n - 1), np.clip(i + h, 0, n - 1)
    span = (hi - lo) / fs
    vx = (xd[hi] - xd[lo]) / span
    vy = (yd[hi] - yd[lo]) / span
    vel = np.hypot(vx, vy)
    okn = ok & np.concatenate([[True], ok[:-1]]) & np.concatenate([ok[1:], [True]])
    fixs = (vel < vel_thr) & okn
    st, en = _runs(fixs)
    keep = (en - st) * 1000.0 / fs >= min_fix_ms
    fid = np.full(len(x), -1)
    for k, (a, b) in enumerate(zip(st[keep], en[keep])):
        fid[a:b] = k
    return vel, fid, st[keep], en[keep], xd, yd


def aoi_of(x, y):
    """Quadrant index 0-3 of each gaze sample (-1 = background / missing)."""
    q = np.full(len(x), -1)
    for k, (x0, y0, x1, y1) in enumerate(QUAD_RECTS):
        q[(x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)] = k
    return q


def microsaccades(xd, yd, fs, fix_mask, lam=6.0, min_ms=6.0, max_amp=1.0):
    """Engbert & Kliegl (2003) on one trial: 5-point velocity, median-based elliptic threshold computed
    on fixation samples, minimum duration, amplitude <= max_amp deg. Returns onset sample indices."""
    n = len(xd)
    if n < 5 or fix_mask.sum() < 50:
        return np.array([], int)
    vx = np.zeros(n)
    vy = np.zeros(n)
    vx[2:-2] = (xd[4:] + xd[3:-1] - xd[1:-3] - xd[:-4]) * fs / 6.0
    vy[2:-2] = (yd[4:] + yd[3:-1] - yd[1:-3] - yd[:-4]) * fs / 6.0
    sx = np.sqrt(max(np.median(vx[fix_mask] ** 2) - np.median(vx[fix_mask]) ** 2, 1e-12))
    sy = np.sqrt(max(np.median(vy[fix_mask] ** 2) - np.median(vy[fix_mask]) ** 2, 1e-12))
    above = ((vx / (lam * sx)) ** 2 + (vy / (lam * sy)) ** 2 > 1) & fix_mask
    st, en = _runs(above)
    keep = (en - st) * 1000.0 / fs >= min_ms
    st, en = st[keep], en[keep]
    amp = np.hypot(xd[np.minimum(en, n - 1)] - xd[st], yd[np.minimum(en, n - 1)] - yd[st])
    return st[amp <= max_amp]


# ----------------------------------------------------------------------------- features
TRIAL_FEATURES = ["dwell_1", "dwell_2", "n_fix", "n_visit",           # paper's four (per presentation)
                  "first_fix", "lat_first",                           # first-look capture and latency
                  "pupil_face", "pupil_trial",                       # pupil (blink-interpolated, baseline-subtracted)
                  "msacc_rate",                                       # microsaccades per s of dwell (>= 500 Hz only)
                  "dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4", "dwell_b5"]   # finer time course
BINS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3500), (3500, 5000)]


def _pupil_clean(p, fs, pad_ms=50):
    """Blink interpolation: widen gaps by pad_ms each side, then linear interpolation."""
    bad = ~np.isfinite(p)
    w = max(1, int(round(pad_ms * fs / 1000)))
    if bad.any():
        k = np.ones(2 * w + 1)
        bad = np.convolve(bad.astype(float), k, mode="same") > 0
    q = p.copy()
    q[bad] = np.nan
    f, ok = _fill(q)
    return f, ok


def trial_features(x, y, p, fs, with_msacc=True):
    """Per-quadrant feature matrix (4, len(TRIAL_FEATURES)) for one trial whose samples start PRE_MS
    before display onset."""
    n = len(x)
    t = np.arange(n) * 1000.0 / fs - PRE_MS
    vel, fid, st, en, xd, yd = ivt(x, y, fs)
    q = aoi_of(x, y)
    fixs = fid >= 0
    dt = 1000.0 / fs
    F = np.full((4, len(TRIAL_FEATURES)), np.nan)
    # fixation-level AOI (centroid) and onsets, restricted to the display window
    fx = np.array([np.nanmean(x[a:b]) for a, b in zip(st, en)]) if len(st) else np.array([])
    fy = np.array([np.nanmean(y[a:b]) for a, b in zip(st, en)]) if len(st) else np.array([])
    fq = aoi_of(fx, fy) if len(st) else np.array([], int)
    ft = t[st] if len(st) else np.array([])
    inwin = (ft >= 0) & (ft < POST_MS) if len(st) else np.array([], bool)
    fq_w, ft_w = fq[inwin], ft[inwin]
    # visits: runs of consecutive same-AOI fixations (background fixations break runs)
    vis = np.zeros(4)
    prev = -2
    for a in fq_w:
        if a >= 0 and a != prev:
            vis[a] += 1
        prev = a
    first = fq_w[(fq_w >= 0)][0] if (fq_w >= 0).any() else -1
    # pupil
    pc, pok = _pupil_clean(p, fs)
    base = (t >= -200) & (t < 0)
    b = np.nanmean(pc[base & pok]) if (base & pok).any() else np.nan
    pdil = pc - b
    late = (t >= 1000) & (t < POST_MS)
    pupil_trial = np.nanmean(pdil[late & pok]) if (late & pok).any() else np.nan
    ms_on = np.array([], int)
    if with_msacc and fs >= 500:
        ms_on = microsaccades(xd, yd, fs, fixs & (t >= -200))
    for k in range(4):
        onk = fixs & (q == k)
        F[k, 0] = onk[(t >= PHASE1[0]) & (t < PHASE1[1])].sum() * dt / (PHASE1[1] - PHASE1[0])
        F[k, 1] = onk[(t >= PHASE2[0]) & (t < PHASE2[1])].sum() * dt / (PHASE2[1] - PHASE2[0])
        F[k, 2] = np.sum(fq_w == k)
        F[k, 3] = vis[k]
        F[k, 4] = float(first == k)
        F[k, 5] = ft_w[fq_w == k][0] if np.any(fq_w == k) else float(POST_MS)
        sel = onk & late & pok
        F[k, 6] = np.nanmean(pdil[sel]) if sel.sum() >= 0.1 * fs else np.nan
        F[k, 7] = pupil_trial
        if with_msacc and fs >= 500:
            dwell_s = onk[(t >= 200) & (t < POST_MS)].sum() / fs
            cnt = np.sum(onk[ms_on] & (t[ms_on] >= 200) & (t[ms_on] < POST_MS)) if len(ms_on) else 0
            F[k, 8] = cnt / dwell_s if dwell_s > 0.2 else np.nan
        for j, (a, bb) in enumerate(BINS):
            F[k, 9 + j] = onk[(t >= a) & (t < bb)].sum() * dt / (bb - a)
    return F


def participant_features(seg, fs, trial_idx=None, with_msacc=True):
    """Average per-image features over the presentations in `trial_idx` (default: all trials).
    Returns rows (image, n_pres, features...)."""
    idx = np.arange(len(seg["images"])) if trial_idx is None else np.asarray(trial_idx)
    rows = {}
    for k in idx:
        x, y, p = degrade(seg["x"][k], seg["y"][k], seg["p"][k], fs)
        F = trial_features(x, y, p, fs, with_msacc=with_msacc)
        for qd in range(4):
            rows.setdefault(int(seg["images"][k, qd]), []).append(F[qd])
    out = []
    for img, fs_ in rows.items():
        a = np.vstack(fs_)
        with np.errstate(all="ignore"):
            m = np.nanmean(a, 0)
        out.append([img, len(fs_), *m])
    return np.array(out)
