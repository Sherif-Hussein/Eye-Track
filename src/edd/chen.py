"""Chen, Mathot & Akyurek (2026, J. Cognition 9(1):20) moving-RSVP familiar-face data.

Fast EDF reading (SR Research edfapi shipped with eyelinkio, raw structs parsed in bulk with
NumPy), eyelinkparser-style trial/phase parsing, pupil blink reconstruction and downsampling,
epoching as in the authors' analysis scripts, window features, and the per-participant
detection rule (one-tailed independent-samples t-test, alpha = .05).
"""
import ctypes as ct
import os
import warnings

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------------------------------
# EDF reading
# ----------------------------------------------------------------------------------------------
_SAMPLE, _MESSAGE, _ENDBLINK, _NO_PENDING = 200, 24, 4, 0


def _edf_api():
    from eyelinkio.edf import _edf2py as e
    lib = e.edfapi
    next_data = ct.CFUNCTYPE(ct.c_int, ct.c_void_p)(("edf_get_next_data", lib))
    float_data = ct.CFUNCTYPE(ct.c_void_p, ct.c_void_p)(("edf_get_float_data", lib))
    open_file = ct.CFUNCTYPE(ct.c_void_p, ct.c_char_p, ct.c_int, ct.c_int, ct.c_int,
                             ct.POINTER(ct.c_int))(("edf_open_file", lib))
    close_file = ct.CFUNCTYPE(ct.c_int, ct.c_void_p)(("edf_close_file", lib))
    return e, next_data, float_data, open_file, close_file


def read_edf_fast(path, phase="RSVP", max_samples=4_000_000):
    """Return dict with sample time (ms), pupil, gaze x/y for the recorded eye, messages and
    EyeLink blink events. Missing data are NaN.

    Samples are only converted while inside `start_phase <phase>` ... `end_phase <phase>` (the
    edfapi float conversion costs ~20 us per sample; phase=None keeps every sample). Converted
    structs are memmoved into a preallocated NumPy buffer and parsed in bulk."""
    e, next_data, float_data, open_file, close_file = _edf_api()
    size = ct.sizeof(e.FSAMPLE)
    err = ct.c_int(1)
    fid = open_file(os.path.normpath(os.path.abspath(path)).encode("ascii"), 2, 1, 1, ct.byref(err))
    if not fid or err.value != 0:
        raise OSError(f"cannot open {path}")
    buf = np.empty(max_samples, dtype=np.dtype(e.FSAMPLE))
    base, n = buf.ctypes.data, 0
    msgs, blinks = [], []
    memmove, cast, PFEV = ct.memmove, ct.cast, ct.POINTER(e.FEVENT)
    inside = phase is None
    start_msg, end_msgs = f"start_phase {phase}", (f"end_phase {phase}", f"stop_phase {phase}")
    try:
        while True:
            t = next_data(fid)
            if t == _NO_PENDING:
                break
            if t == _SAMPLE:
                if inside:
                    memmove(base + n * size, float_data(fid), size)
                    n += 1
            elif t == _MESSAGE:
                ev = cast(float_data(fid), PFEV).contents
                raw = ct.string_at(ct.byref(ev.message[0]), ev.message.contents.len + 1)[2:]
                m = raw.decode("utf-8", "ignore").strip("\x00").strip()
                msgs.append((int(ev.sttime), m))
                if phase is not None:
                    if m.startswith(start_msg):
                        inside = True
                    elif m.startswith(end_msgs):
                        inside = False
            elif t == _ENDBLINK:
                ev = cast(float_data(fid), PFEV).contents
                blinks.append((int(ev.sttime), int(ev.entime)))
    finally:
        close_file(fid)
    arr = buf[:n]
    pa = arr["pa"].astype(float)
    eye = int(np.argmax((pa > 0).sum(0)))  # monocular: pick the recorded eye
    out = {"t": arr["time"].astype(np.int64), "pa": pa[:, eye],
           "gx": arr["gx"][:, eye].astype(float), "gy": arr["gy"][:, eye].astype(float),
           "messages": msgs, "blinks": np.asarray(blinks, dtype=np.int64).reshape(-1, 2), "eye": eye}
    out["pa"][~(out["pa"] > 0)] = np.nan
    for k in ("gx", "gy"):
        bad = (np.abs(out[k]) > 1e7) | ~np.isfinite(out[k])
        out[k][bad] = np.nan
    return out


def parse_trials(rec, phase="RSVP"):
    """eyelinkparser semantics: `start_trial` opens a trial, `var <name> <value>` sets columns,
    `start_phase/end_phase <phase>` delimit the trace, `stop_trial` closes the trial."""
    trials, cur = [], None
    for t, m in rec["messages"]:
        w = m.split()
        if not w:
            continue
        if w[0] == "start_trial":
            cur = {"vars": {}, "t0": None, "t1": None}
        elif cur is None:
            continue
        elif w[0] == "var" and len(w) >= 2:
            cur["vars"][w[1]] = " ".join(w[2:])
        elif w[0] == "start_phase" and len(w) > 1 and w[1] == phase:
            cur["t0"] = t
        elif w[0] in ("end_phase", "stop_phase") and len(w) > 1 and w[1] == phase:
            cur["t1"] = t
        elif w[0] in ("stop_trial", "end_trial"):
            trials.append(cur)
            cur = None
    return trials


# ----------------------------------------------------------------------------------------------
# Pupil preprocessing
# ----------------------------------------------------------------------------------------------
def _runs(mask):
    """(start, stop) index pairs of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def blink_reconstruct(p, blink_mask=None, vt=5.0, maxdur=500, margin=10, max_ext=50, smooth=21):
    """Approximation of datamatrix.series.blinkreconstruct(mode='advanced') at 1000 Hz.

    Missing samples (and EyeLink blink events) form gaps. Each gap is extended backward while the
    smoothed pupil velocity shows the lid-closing drop (< -vt units/ms) and forward while it shows
    the re-opening transient (|v| > vt), up to `max_ext` ms, plus a fixed `margin`. Gaps up to
    `maxdur` ms are linearly interpolated; longer gaps stay NaN.
    """
    p = p.astype(float).copy()
    miss = ~np.isfinite(p)
    if blink_mask is not None:
        miss |= blink_mask
    if not miss.any():
        return p
    filled = p.copy()
    idx = np.arange(len(p))
    good = ~miss
    if good.sum() < 2:
        return np.full_like(p, np.nan)
    filled[miss] = np.interp(idx[miss], idx[good], p[good])
    k = np.ones(smooth) / smooth
    v = np.gradient(np.convolve(filled, k, mode="same"))
    starts, stops = _runs(miss)
    gap = miss.copy()
    n = len(p)
    for s, e in zip(starts, stops):
        a = s
        while a > 0 and s - a < max_ext and v[a - 1] < -vt:
            a -= 1
        b = e
        while b < n and b - e < max_ext and abs(v[b]) > vt:
            b += 1
        gap[max(0, a - margin):min(n, b + margin)] = True
    out = p.copy()
    out[gap] = np.nan
    good = np.isfinite(out)
    if good.sum() < 2:
        return np.full_like(p, np.nan)
    gs, ge = _runs(~good)
    for s, e in zip(gs, ge):
        if s == 0 or e >= n or (e - s) > maxdur:
            continue  # leading/trailing or too-long gaps stay missing
        out[s:e] = np.interp(np.arange(s, e), [s - 1, e], [out[s - 1], out[e]])
    return out


def dm_smooth(a, winlen=21):
    """datamatrix.series._smooth (Hanning, reflected edges); NaNs spread through the window."""
    if a.size < winlen:
        return a
    d = (winlen - 1) // 2
    s = np.r_[a[d:0:-1], a, a[-2:-d - 2:-1]]
    w = np.hanning(winlen)
    return np.convolve(w / np.nansum(w), s, mode="valid")


def _dm_blink_points(vtrace, vt_start, vt_end, maxdur, margin):
    astart = np.where(vtrace < -vt_start)[0]
    if len(astart) == 0:
        return None
    istart = astart[0]
    amid = np.where(vtrace[istart:] > vt_end)[0]
    if len(amid) == 0:
        return None
    imid = amid[0] + istart
    aend = np.where(vtrace[imid:] < np.nanstd(vtrace) / 100)[0]
    if len(aend) == 0:
        return None
    iend = aend[0] + imid
    if istart - margin >= 0:
        istart -= margin
    if iend + margin < len(vtrace):
        iend += margin
    if iend - istart > maxdur:
        pts = _dm_blink_points(vtrace[iend:], vt_start, vt_end, maxdur, margin)
        return None if pts is None else pts + iend
    return np.array([istart, iend])


def _dm_cubic_points(a, istart, iend):
    dur = iend - istart
    pts = np.array([istart - dur, istart, iend, iend + dur])
    for q in pts:
        if q < 0 or q >= len(a) or np.isnan(a[q]):
            return None
    return pts


def _dm_group(a):
    """Start and stop indices of contiguous runs (same output as datamatrix._group)."""
    if len(a) == 0:
        return []
    cut = np.where(np.diff(a) > 1)[0] + 1
    starts = np.r_[a[0], a[cut]]
    stops = np.r_[a[cut - 1] + 1, a[-1] + 1]
    return list(zip(starts, stops))


def _dm_trim(a, vtrace, std_thr, gap_margin, gap_vt):
    """Mark outliers/missing and velocity outliers NaN with a margin. As in the original, a.mean()
    is not NaN-aware, so the value-outlier terms are inert whenever any NaN remains; negative slice
    starts near the trace onset also behave as in the original."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        idx = np.where((a < (a.mean() - std_thr * a.std())) | (a.mean() > (a + std_thr * a.std()))
                       | np.isnan(a))[0]
    for s, e in _dm_group(idx):
        if e == len(a):
            continue
        a[s - gap_margin:e + gap_margin] = np.nan
    idx = np.where(np.abs(vtrace) > gap_vt)[0]
    for s, e in _dm_group(idx):
        if e == len(a):
            continue
        a[s - gap_margin:e + gap_margin] = np.nan
    return a


def dm_blinkreconstruct(a, vt_start=10, vt_end=5, maxdur=500, margin=10, gap_margin=20, gap_vt=10,
                        smooth_winlen=21, std_thr=3):
    """Port of datamatrix 1.0.20 blinkreconstruct(mode='advanced') with its default parameters, the
    call used by the authors (eyelinkparser.defaulttraceprocessor(blinkreconstruct=True,
    mode='advanced')). datamatrix is GPL-3.0 (Mathot et al.); this function is a port of its
    _blinkreconstruct_recursive, written iteratively, so the licence applies to any redistribution.
    Blinks: velocity onset, reversal, end; cubic spline if four valid points exist, else linear.
    What is left undetected is trimmed to NaN with a margin and is NOT interpolated."""
    a = np.array(a, dtype=float)
    seen = set()
    while True:
        strace = dm_smooth(a, smooth_winlen)
        vtrace = np.diff(strace)
        pts = _dm_blink_points(vtrace, vt_start, vt_end, maxdur, margin)
        if pts is None:
            return _dm_trim(a, vtrace, std_thr, gap_margin, gap_vt)
        key = (int(pts[0]), int(pts[1]))
        if key in seen:
            return a  # the original aborts reconstruction for this signal here
        seen.add(key)
        istart, iend = pts
        cubic = _dm_cubic_points(vtrace, istart, iend)
        x = pts if cubic is None else cubic
        f = interp1d(x, a[x], kind="linear" if cubic is None else "cubic")
        ix = np.arange(istart, iend)
        a[ix] = f(ix)


def downsample(x, factor=10):
    """Bin-mean (NaN-aware) downsampling, as eyelinkparser's downsample=10."""
    n = len(x) // factor * factor
    x = x[:n].reshape(-1, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=1)


def extract_subject(path, phase_len_ms=6500):
    """Per-trial 100 Hz RSVP-phase traces for one EDF.

    Returns (meta: list of dict, arrays: dict of (n_trials, phase_len_ms // 10) float32) with
    channels p_dm (authors' exact method: dm_blinkreconstruct on the phase trace), p_rec (simpler
    gap-interpolating approximation, kept as a sensitivity check), p_raw (missing kept NaN), x, y,
    miss (fraction of missing 1 kHz samples per 10 ms bin) and pa_1k (raw 1 kHz pupil, float32)."""
    rec = read_edf_fast(path)
    t, pa, gx, gy = rec["t"], rec["pa"], rec["gx"], rec["gy"]
    bl = rec["blinks"]
    trials = parse_trials(rec)
    L = phase_len_ms
    chans = {k: [] for k in ("p_dm", "p_rec", "p_raw", "x", "y", "miss")}
    raw_1k = []
    meta = []
    for tr in trials:
        if tr["t0"] is None:
            continue
        t0 = tr["t0"]
        sel = slice(np.searchsorted(t, t0), np.searchsorted(t, t0 + L))
        pos = t[sel] - t0
        arrs = {k: np.full(L, np.nan) for k in ("pa", "gx", "gy")}
        for k, src in (("pa", pa), ("gx", gx), ("gy", gy)):
            arrs[k][pos] = src[sel]
        bmask = np.zeros(L, bool)
        for s, e in bl[(bl[:, 1] >= t0) & (bl[:, 0] < t0 + L)]:
            bmask[max(0, s - t0):min(L, e - t0 + 1)] = True
        p_raw = arrs["pa"].copy()
        p_raw[bmask] = np.nan
        p_rec = blink_reconstruct(arrs["pa"], bmask)
        cut_dm = min(L, tr["t1"] - t0) if tr["t1"] is not None else L
        p_dm = np.full(L, np.nan)
        p_dm[:cut_dm] = dm_blinkreconstruct(arrs["pa"][:cut_dm])
        raw_1k.append(arrs["pa"].astype(np.float32))
        miss = (~np.isfinite(p_raw)).astype(float)
        if tr["t1"] is not None and tr["t1"] - t0 < L:  # beyond the phase end: missing
            cut = tr["t1"] - t0
            for a in (p_rec, p_raw, arrs["gx"], arrs["gy"]):
                a[cut:] = np.nan
            miss[cut:] = np.nan
        for k, a in (("p_dm", p_dm), ("p_rec", p_rec), ("p_raw", p_raw), ("x", arrs["gx"]), ("y", arrs["gy"]), ("miss", miss)):
            chans[k].append(downsample(a).astype(np.float32))
        m = dict(tr["vars"])
        m["phase_ms"] = (tr["t1"] - t0) if tr["t1"] is not None else -1
        m["n_blinks_phase"] = int(((bl[:, 0] >= t0) & (bl[:, 0] < t0 + L)).sum())
        meta.append(m)
    out = {k: np.stack(v) for k, v in chans.items()}
    out["pa_1k"] = np.stack(raw_1k)
    return meta, out


# ----------------------------------------------------------------------------------------------
# Epoching (authors' scripts) and features
# ----------------------------------------------------------------------------------------------
def epoch(tr, t1pos, start=0, length=250, soa=30):
    """Lock each trial's trace to the critical-face onset (T1onset = (T1pos - 1) * 30 samples)."""
    on = (np.asarray(t1pos, int) - 1) * soa
    out = np.full((len(tr), length), np.nan, dtype=float)
    for i, o in enumerate(on):
        a, b = o + start, o + start + length
        seg = tr[i, max(a, 0):min(b, tr.shape[1])]
        out[i, max(0, -a):max(0, -a) + len(seg)] = seg
    return out


def baseline(tr, t1pos, n=5, soa=30):
    on = (np.asarray(t1pos, int) - 1) * soa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.array([np.nanmean(tr[i, o - n:o]) for i, o in enumerate(on)])


def smooth_hanning(x, winlen=25):
    """datamatrix.series.smooth: Hanning-window convolution with reflected edges, row-wise."""
    w = np.hanning(winlen)
    w /= w.sum()
    pad = winlen // 2
    out = np.empty_like(x)
    for i, r in enumerate(x):
        r2 = np.concatenate([r[pad:0:-1], r, r[-2:-pad - 2:-1]])
        out[i] = np.convolve(r2, w, mode="valid")[: len(r)]
    return out


def bin_means(x, edges):
    """Mean over [a, b) sample windows -> (n, len(edges) - 1)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmean(x[:, a:b], axis=1) for a, b in zip(edges[:-1], edges[1:])], 1)


# ----------------------------------------------------------------------------------------------
# Detection rule
# ----------------------------------------------------------------------------------------------
def one_tailed_t(pos, neg):
    """Paper's rule: independent-samples (Student) t-test, one-tailed for pos > neg.
    Mirrors the authors' code: p/2 when the mean difference is positive, else p = 1."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), 1.0
    t, p = stats.ttest_ind(pos, neg)
    if not np.isfinite(p):
        return float(t), 1.0
    return float(t), float(p / 2 if pos.mean() > neg.mean() else 1.0)


def cluster_bootstrap(units, fn, n_boot=2000, seed=0):
    """Percentile 95% CI of fn(selected_units) resampling participants with replacement."""
    rng = np.random.default_rng(seed)
    units = np.asarray(units)
    vals = [fn(rng.choice(units, len(units), replace=True)) for _ in range(n_boot)]
    vals = np.asarray(vals, dtype=float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
