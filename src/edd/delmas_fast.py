"""Vectorised per-participant feature extraction for the Delmas raw-sample pipeline (efficiency experiment).

Drop-in numerical replacement for `edd.delmas_raw.degrade` + `edd.delmas_raw.trial_features` + the per-image
aggregation of `scripts/phase2_delmas_raw.raw_to_scores`, for one participant at a time. All trials of a
participant (or of a chunk of trials) are processed as one (n_trials, n_samples) NumPy batch: no per-trial,
per-fixation, per-quadrant or per-window Python loops, no pandas.

Same definitions as the original (see `delmas_raw.py`): gap interpolation with held edges, polyphase
anti-alias resampling and the half-period missing rule, I-VT with a fixed +-4 ms differentiator and a 60 ms
minimum fixation, AOI = the four picture rectangles, fixation-level AOI from the fixation centroid, visits =
runs of same-AOI fixations in the 0-5000 ms window, first look, latency, dwell fractions in the two phases and
five bins, blink-padded pupil interpolation with a [-200, 0) ms baseline, and (>= 500 Hz) Engbert-Kliegl
microsaccades. Differences from the original are limited to floating-point summation order (~1e-13).
"""
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from edd.delmas_raw import (BINS, CENTER, MIN_FIX_MS, PHASE1, PHASE2, PIX_PER_DEG, POST_MS, PRE_MS, QUAD_RECTS,
                            TRIAL_FEATURES, VEL_THR)

NF = len(TRIAL_FEATURES)
_X0, _X1 = QUAD_RECTS[0, 0], QUAD_RECTS[0, 2]        # left column  (rects 0 and 2)
_X2, _X3 = QUAD_RECTS[1, 0], QUAD_RECTS[1, 2]        # right column (rects 1 and 3)
_Y0, _Y1 = QUAD_RECTS[0, 1], QUAD_RECTS[0, 3]        # top row      (rects 0 and 1)
_Y2, _Y3 = QUAD_RECTS[2, 1], QUAD_RECTS[2, 3]        # bottom row   (rects 2 and 3)


# ----------------------------------------------------------------------------- building blocks
def fill_rows(v):
    """Row-wise `delmas_raw._fill`: linear interpolation over NaN gaps, edges held; rows with fewer than two
    finite samples become zeros (and keep their finite mask). Returns (filled, finite_mask)."""
    v = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(v)
    rows = np.flatnonzero(~ok.all(axis=1))
    if rows.size == 0:
        return v, ok
    out = v.copy()
    sub, okr = v[rows], ok[rows]
    n = v.shape[1]
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(okr, idx, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(okr, idx, n)[:, ::-1], axis=1)[:, ::-1]
    pc, nc = np.clip(prev, 0, n - 1), np.clip(nxt, 0, n - 1)
    vp, vn = np.take_along_axis(sub, pc, 1), np.take_along_axis(sub, nc, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (vn - vp) / (nxt - prev)
        interp = slope * (idx - prev) + vp                   # same formula as np.interp
    filled = np.where(prev < 0, vn, np.where(nxt >= n, vp, interp))
    out[rows] = np.where(okr, sub, filled)
    few = okr.sum(1) < 2
    if few.any():
        out[rows[few]] = 0.0
    return out, ok


def degrade_batch(X, Y, P, fs):
    """Batch `delmas_raw.degrade`: (n_trials, 5500) 1000 Hz samples -> sensor rate `fs`."""
    if fs == 1000:
        return X.astype(np.float64), Y.astype(np.float64), P.astype(np.float64)
    fr = Fraction(fs, 1000).limit_denominator(1000)
    n = X.shape[1]
    bad = ~(np.isfinite(X) & np.isfinite(Y) & np.isfinite(P))
    out = [resample_poly(fill_rows(v)[0], fr.numerator, fr.denominator, axis=1, padtype="line") for v in (X, Y, P)]
    m = out[0].shape[1]
    t_out = np.arange(m) * 1000.0 / fs
    half = 500.0 / fs
    cbad = np.zeros((bad.shape[0], n + 1), np.int32)
    np.cumsum(bad, axis=1, out=cbad[:, 1:])
    lo = np.clip(np.floor(t_out - half).astype(int), 0, n)
    hi = np.clip(np.ceil(t_out + half).astype(int) + 1, 0, n)
    miss = (cbad[:, hi] - cbad[:, lo]) > 0
    for v in out:
        v[miss] = np.nan
    return out[0], out[1], out[2]


def _runs2d(mask):
    """Runs of True per row: (trial, start, end_exclusive), row-major order, like `delmas_raw._runs`."""
    T, n = mask.shape
    m = np.zeros((T, n + 2), np.int8)
    m[:, 1:-1] = mask
    d = np.diff(m, axis=1).ravel()
    w = n + 1
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return s // w, s % w, e % w


def aoi_batch(x, y):
    """Quadrant index 0-3 of each sample (-1 = background / missing), same inclusive rectangles as `aoi_of`."""
    left, right = (x >= _X0) & (x <= _X1), (x >= _X2) & (x <= _X3)
    top, bot = (y >= _Y0) & (y <= _Y1), (y >= _Y2) & (y <= _Y3)
    q = np.full(x.shape, -1, np.int8)
    q[left & top] = 0
    q[right & top] = 1
    q[left & bot] = 2
    q[right & bot] = 3
    return q


def ivt_batch(X, Y, fs):
    """Batch I-VT. Returns xd, yd (gap-filled degrees), fixation-sample mask (>= 60 ms runs) and the kept runs
    (trial, start, end_exclusive)."""
    xd, okx = fill_rows((X - CENTER[0]) / PIX_PER_DEG)
    yd, _ = fill_rows((Y - CENTER[1]) / PIX_PER_DEG)
    ok = okx & np.isfinite(Y)
    T, n = X.shape
    h = max(1, int(round(fs * 0.004)))
    vel = np.empty((T, n))
    if n > 2 * h:        # interior samples: constant span, contiguous slices; the 2h edge samples use clipped indices
        cols = np.r_[0:h, n - h:n]
        parts = [(slice(h, n - h), slice(2 * h, n), slice(0, n - 2 * h), 2 * h / fs)]
    else:
        cols = np.arange(n)
        parts = []
    for sl, a, b, span in parts:
        vel[:, sl] = np.hypot((xd[:, a] - xd[:, b]) / span, (yd[:, a] - yd[:, b]) / span)
    lo, hi = np.clip(cols - h, 0, n - 1), np.clip(cols + h, 0, n - 1)
    span = (hi - lo) / fs
    vel[:, cols] = np.hypot((xd[:, hi] - xd[:, lo]) / span, (yd[:, hi] - yd[:, lo]) / span)
    okn = ok.copy()
    okn[:, 1:] &= ok[:, :-1]
    okn[:, :-1] &= ok[:, 1:]
    fixs = (vel < VEL_THR) & okn
    tr, st, en = _runs2d(fixs)
    keep = (en - st) * 1000.0 / fs >= MIN_FIX_MS
    tr, st, en = tr[keep], st[keep], en[keep]
    delta = np.zeros((T, n + 1), np.int16)
    delta[tr, st] = 1
    delta[tr, en] = -1
    fix = np.cumsum(delta, axis=1)[:, :n] > 0
    return xd, yd, fix, (tr, st, en)


def _row_median(A):
    """Median of the non-NaN entries of each row (rows with none -> NaN); equals np.median of those entries."""
    s = np.sort(A, axis=1)
    c = (~np.isnan(A)).sum(1)
    r = np.arange(A.shape[0])
    lo, hi = np.maximum(c - 1, 0) // 2, c // 2
    med = (s[r, lo] + s[r, np.minimum(hi, A.shape[1] - 1)]) / 2
    return np.where(c > 0, med, np.nan)


def _microsaccade_onsets(xd, yd, fs, fixm, t, lam=6.0, min_ms=6.0, max_amp=1.0):
    """Batch `delmas_raw.microsaccades`. Returns (trial, onset sample) of accepted microsaccades."""
    T, n = xd.shape
    empty = (np.array([], int), np.array([], int))
    if n < 5:
        return empty
    use = fixm.sum(1) >= 50
    if not use.any():
        return empty
    vx, vy = np.zeros((T, n)), np.zeros((T, n))
    vx[:, 2:-2] = (xd[:, 4:] + xd[:, 3:-1] - xd[:, 1:-3] - xd[:, :-4]) * fs / 6.0
    vy[:, 2:-2] = (yd[:, 4:] + yd[:, 3:-1] - yd[:, 1:-3] - yd[:, :-4]) * fs / 6.0
    sd = []
    for v in (vx, vy):
        vm = np.where(fixm, v, np.nan)
        sd.append(np.sqrt(np.maximum(_row_median(vm ** 2) - _row_median(vm) ** 2, 1e-12)))
    sx, sy = sd
    with np.errstate(invalid="ignore", divide="ignore"):
        above = ((vx / (lam * sx)[:, None]) ** 2 + (vy / (lam * sy)[:, None]) ** 2 > 1) & fixm
    above &= use[:, None]
    tr, st, en = _runs2d(above)
    keep = (en - st) * 1000.0 / fs >= min_ms
    tr, st, en = tr[keep], st[keep], en[keep]
    e1 = np.minimum(en, n - 1)
    amp = np.hypot(xd[tr, e1] - xd[tr, st], yd[tr, e1] - yd[tr, st])
    ok = amp <= max_amp
    return tr[ok], st[ok]


# ----------------------------------------------------------------------------- trial features
def trial_features_batch(X, Y, P, fs, with_msacc=True):
    """(n_trials, 4, len(TRIAL_FEATURES)) per-quadrant features for trials that start PRE_MS before display onset;
    batch equivalent of `delmas_raw.trial_features` applied to every row."""
    T, n = X.shape
    t = np.arange(n) * 1000.0 / fs - PRE_MS
    dt = 1000.0 / fs
    xd, yd, fix, (ftr, fst, fen) = ivt_batch(X, Y, fs)
    q = aoi_batch(X, Y)
    F = np.full((T, 4, NF), np.nan)

    # ---- fixation-level AOI (centroid), window, visits, first look, latency
    nfx = len(ftr)
    if nfx:
        flat = np.concatenate([X.ravel(), [0.0]]), np.concatenate([Y.ravel(), [0.0]])
        a, b = ftr * n + fst, ftr * n + fen
        idx = np.empty(2 * nfx, np.intp)
        idx[0::2], idx[1::2] = a, b
        cnt = fen - fst
        fx = np.add.reduceat(flat[0], idx)[0::2] / cnt
        fy = np.add.reduceat(flat[1], idx)[0::2] / cnt
        fq = aoi_batch(fx, fy).astype(np.intp)
        ft = t[fst]
        inw = (ft >= 0) & (ft < POST_MS)
        wtr, wq, wt = ftr[inw], fq[inw], ft[inw]
        on = wq >= 0
        F[:, :, 2] = np.bincount(wtr[on] * 4 + wq[on], minlength=T * 4).reshape(T, 4)
        newtr = np.ones(len(wtr), bool)
        newtr[1:] = wtr[1:] != wtr[:-1]
        prevq = np.empty_like(wq)
        prevq[0:1] = -2
        prevq[1:] = wq[:-1]
        visit = on & (newtr | (wq != prevq))
        F[:, :, 3] = np.bincount(wtr[visit] * 4 + wq[visit], minlength=T * 4).reshape(T, 4)
        first = np.full(T, -1)
        utr, uidx = np.unique(wtr[on], return_index=True)
        first[utr] = wq[on][uidx]
        F[:, :, 4] = first[:, None] == np.arange(4)[None, :]
        lat = np.full(T * 4, float(POST_MS))
        np.minimum.at(lat, wtr[on] * 4 + wq[on], wt[on])
        F[:, :, 5] = lat.reshape(T, 4)
    else:
        F[:, :, 2:4] = 0.0
        F[:, :, 4] = 0.0
        F[:, :, 5] = float(POST_MS)

    # ---- pupil: blink-padded gap interpolation, baseline [-200, 0), late window [1000, 5000)
    bad = ~np.isfinite(P)
    w = max(1, int(round(50 * fs / 1000)))
    cb = np.zeros((T, n + 1), np.int32)
    np.cumsum(bad, axis=1, out=cb[:, 1:])
    ii = np.arange(n)
    dil = (cb[:, np.minimum(ii + w + 1, n)] - cb[:, np.maximum(ii - w, 0)]) > 0
    pc, pok = fill_rows(np.where(dil, np.nan, P))
    b0, b1 = np.searchsorted(t, -200, "left"), np.searchsorted(t, 0, "left")
    nb = pok[:, b0:b1].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bl = np.where(nb > 0, np.where(pok[:, b0:b1], pc[:, b0:b1], 0.0).sum(1) / nb, np.nan)
    pdil = pc - bl[:, None]
    l0, l1 = np.searchsorted(t, 1000, "left"), np.searchsorted(t, POST_MS, "left")
    latep = np.zeros((T, n), bool)
    latep[:, l0:l1] = pok[:, l0:l1]
    nl = latep.sum(1)
    plate = np.where(latep, pdil, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pupil_trial = np.where(nl > 0, plate.sum(1) / nl, np.nan)
    F[:, :, 7] = pupil_trial[:, None]

    # ---- microsaccades (>= 500 Hz only)
    do_ms = with_msacc and fs >= 500
    if do_ms:
        mtr, mst = _microsaccade_onsets(xd, yd, fs, fix & (t >= -200), t)
        mt = t[mst]
        mwin = (mt >= 200) & (mt < POST_MS)

    # ---- per-quadrant dwell fractions and dwell-conditioned pupil
    lab = np.where(fix, q, -1)
    wins = [PHASE1, PHASE2] + list(BINS)
    cs = np.zeros((T, n + 1), np.int32)
    d0, d1 = np.searchsorted(t, 200, "left"), np.searchsorted(t, POST_MS, "left")
    lo_hi = [(np.searchsorted(t, a_, "left"), np.searchsorted(t, b_, "left")) for a_, b_ in wins]
    for k in range(4):
        onk = lab == k
        np.cumsum(onk, axis=1, out=cs[:, 1:])
        for col, (a_, b_), (lo, hi) in zip([0, 1, 9, 10, 11, 12, 13], wins, lo_hi):
            F[:, k, col] = (cs[:, hi] - cs[:, lo]) * dt / (b_ - a_)
        sel = onk & latep
        ns = sel.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            F[:, k, 6] = np.where(ns >= 0.1 * fs, np.where(sel, pdil, 0.0).sum(1) / ns, np.nan)
        if do_ms:
            dwell_s = (cs[:, d1] - cs[:, d0]) / fs
            sel_ms = mwin & (lab[mtr, mst] == k)
            c = np.bincount(mtr[sel_ms], minlength=T)
            with np.errstate(invalid="ignore", divide="ignore"):
                F[:, k, 8] = np.where(dwell_s > 0.2, c / dwell_s, np.nan)
    return F


# ----------------------------------------------------------------------------- per participant
def aggregate_images(F, images):
    """Mean over the presentations of each image (NaN skipped), images in order of first appearance, exactly the
    rows of `raw_to_scores` (trial-major, quadrant-minor)."""
    img = np.asarray(images).ravel()
    flat = F.reshape(-1, F.shape[-1])
    uniq, first, inv = np.unique(img, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty(len(uniq), int)
    rank[order] = np.arange(len(uniq))
    g = rank[inv]
    ok = ~np.isnan(flat)
    cnt = np.zeros((len(uniq), flat.shape[1]))
    s = np.zeros_like(cnt)
    np.add.at(cnt, g, ok)
    np.add.at(s, g, np.where(ok, flat, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / cnt, np.nan)


def participant_image_features(X, Y, P, images, fs, z=False, with_msacc=True, chunk=64):
    """Per-image feature matrix of one participant from sensor-rate samples (n_trials, n), in `TRIAL_FEATURES`
    order; `z` applies the per-participant z-scoring of ALL_z."""
    F = np.concatenate([trial_features_batch(X[a:a + chunk], Y[a:a + chunk], P[a:a + chunk], fs, with_msacc)
                        for a in range(0, len(X), chunk)])
    A = aggregate_images(F, images)
    if z:
        A = (A - np.nanmean(A, 0)) / (np.nanstd(A, 0, ddof=1) + 1e-9)
    return A


def raw_to_scores_fast(pre, images, fs, model, feats, scores_of, chunk=64):
    """Fast counterpart of `phase2_delmas_raw.raw_to_scores` (`pre` = (X, Y, P) arrays at the sensor rate)."""
    base = [f[:-2] if f.endswith("_z") else f for f in feats]
    idx = [TRIAL_FEATURES.index(f) for f in base]
    A = participant_image_features(*pre, images, fs, z=feats[0].endswith("_z"), with_msacc="msacc_rate" in base, chunk=chunk)
    return scores_of(model, A[:, idx])
