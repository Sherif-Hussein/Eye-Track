"""Vectorised feature extraction for the Celniak pipeline (efficiency experiment).

Drop-in numerical replacement, for one recording, of
  * `edd.celniak.slide_features(samples, meta)`  (the paper's AOI metrics: target / non-target-median / normalised
    fixation, visit and glance statistics and baseline-corrected pupil, one row per slide), and
  * the 7 global-gaze quantities of `scripts/phase2_celniak_efficiency.lean_global` (per-run fixation count and
    duration, saccades, blinks, scan path, AOI dwell, valid fraction; mean over the slide runs).
The input is a `Rec` (NumPy columns of the cached raw samples). Instead of one pandas group-by per slide, per
event and per AOI, all events of all slides are built at once with sorting and segment reductions (no per-slide,
per-event or per-AOI Python loops). Definitions are the originals': events are (kind, move_index) groups of a
slide's rows, sorted by onset; a fixation belongs to an AOI when more than half of its samples hit it; a visit is a
run of consecutive same-AOI fixations (saccades in between do not break it); a glance additionally includes the
saccade event that immediately precedes the visit.
"""
import re
from typing import NamedTuple

import numpy as np

from edd.celniak import PROBES, STATS

SLIDE_RE = re.compile(r"^(\d)_(\d)$")
KINDS = ("fixation", "visit", "glance")
FIX, SAC = 1, 2


class Rec(NamedTuple):
    """Raw samples of ONE recording as NumPy columns plus static task metadata."""
    ts: np.ndarray        # int64 timestamps (us)
    stim: np.ndarray      # int32 stimulus code per sample (-1 = no stimulus)
    kind: np.ndarray      # int8: 1 Fixation, 2 Saccade, 0 other / missing
    midx: np.ndarray      # float64 move index (NaN when missing)
    vl: np.ndarray        # bool, left eye "Valid"
    vr: np.ndarray        # bool, right eye "Valid"
    pf: np.ndarray        # pupil_f
    pl: np.ndarray        # pupil_l
    pr: np.ndarray        # pupil_r
    fx: np.ndarray        # fix_x
    fy: np.ndarray        # fix_y
    aoi: np.ndarray       # int8 (n, n_aoi_columns)
    # static metadata derived from the column / stimulus names
    slide_of_code: np.ndarray   # (n_codes,) slide number (index into the slide arrays) or -1
    slide_codes: np.ndarray     # (n_slides,) stimulus code of each slide, in order of first appearance
    slide_cols: np.ndarray      # (n_slides, K) aoi column indices of each slide's items
    slide_target: np.ndarray    # (n_slides,) position of the probe item among the K columns
    slide_q: np.ndarray         # (n_slides,) question number
    slide_rep: np.ndarray       # (n_slides,) repetition
    cross_code: int


def arrays_from_frame(samples):
    """One-time ingestion of the cached DataFrame (columns of `edd.celniak.SAMPLE_COLUMNS` + `aoi:*`). This is data
    loading, not extraction; it is timed separately."""
    import pandas as pd
    st_codes, names = pd.factorize(samples["stimulus"])
    names = [str(s) for s in names]
    aoi_names = [c for c in samples.columns if c.startswith("aoi:")]
    mt = samples["move_type"].to_numpy(object)
    kind = np.where(mt == "Fixation", FIX, np.where(mt == "Saccade", SAC, 0)).astype(np.int8)
    # slides in order of first appearance of their stimulus (= pandas groupby sort=False)
    first_pos = {}
    sl_codes = [c for c in range(len(names)) if SLIDE_RE.match(names[c])]
    for c in sl_codes:
        first_pos[c] = int(np.flatnonzero(st_codes == c)[0])
    sl_codes = sorted(sl_codes, key=first_pos.get)
    slide_of_code = np.full(len(names), -1, np.int32)
    cols, tgt, qs, reps = [], [], [], []
    for s, c in enumerate(sl_codes):
        slide_of_code[c] = s
        stim = names[c]
        ci = [j for j, a in enumerate(aoi_names) if a.startswith(f"aoi:{stim} - ")]
        q, rep = (int(g) for g in SLIDE_RE.match(stim).groups())
        cols.append(ci)
        tgt.append(ci.index(aoi_names.index(f"aoi:{stim} - {PROBES[q]}")))
        qs.append(q)
        reps.append(rep)
    k = {len(c) for c in cols}
    if len(k) != 1:
        raise ValueError(f"slides with different numbers of AOI columns: {sorted(k)}")
    return Rec(ts=samples["ts_us"].to_numpy(np.int64), stim=st_codes.astype(np.int32), kind=kind,
               midx=samples["move_index"].to_numpy(float), vl=samples["valid_l"].eq("Valid").to_numpy(),
               vr=samples["valid_r"].eq("Valid").to_numpy(), pf=samples["pupil_f"].to_numpy(float),
               pl=samples["pupil_l"].to_numpy(float), pr=samples["pupil_r"].to_numpy(float),
               fx=samples["fix_x"].to_numpy(float), fy=samples["fix_y"].to_numpy(float),
               aoi=samples[aoi_names].to_numpy(np.int8), slide_of_code=slide_of_code,
               slide_codes=np.array(sl_codes, np.int32), slide_cols=np.array(cols, np.intp),
               slide_target=np.array(tgt, np.intp), slide_q=np.array(qs), slide_rep=np.array(reps),
               cross_code=names.index("cross") if "cross" in names else -2)


# ----------------------------------------------------------------------------- shared pieces
def pupil_trace(r):
    """`celniak._pupil`: filtered pupil, falling back to the mean of the available eyes."""
    nl, nr = np.isnan(r.pl), np.isnan(r.pr)
    cnt = 2 - nl - nr
    with np.errstate(invalid="ignore", divide="ignore"):
        fb = (np.where(nl, 0.0, r.pl) + np.where(nr, 0.0, r.pr)) / cnt
    return np.where(np.isnan(r.pf), fb, r.pf)


def pupil_baseline_fast(r, pupil):
    idx = np.flatnonzero(r.stim == r.cross_code)
    if idx.size == 0:
        return float("nan")
    brk = np.flatnonzero(np.diff(idx) > 1)
    end = idx[brk[0]] if brk.size else idx[-1]
    seg = pupil[idx[0]:end + 1]
    return float(np.nanmean(seg)) if np.isfinite(seg).any() else float("nan")


def _runs1d(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def blink_onsets_fast(r, dt_ms):
    """Both eyes invalid for 75-500 ms (`phase2_celniak_features.blink_onsets`)."""
    a, b = _runs1d(~(r.vl | r.vr))
    keep = (b - a) * dt_ms >= 75
    keep &= (b - a) * dt_ms <= 500
    return a[keep]


class Events(NamedTuple):
    g: np.ndarray        # group (slide or run) of each event, sorted by (g, start)
    kind: np.ndarray
    start: np.ndarray
    end: np.ndarray
    aoi: np.ndarray      # item position (0..K-1) for fixations with a majority AOI, else -1
    cx: np.ndarray       # fixation centroid (mean of fix_x / fix_y over the event's rows, NaN skipped)
    cy: np.ndarray
    prev_sac_start: np.ndarray   # start of the immediately preceding event if it is a saccade, else own start
    rows_g: np.ndarray   # for fixation events: sample rows ... (see _events)
    row_ev: np.ndarray
    row_pos: np.ndarray


def _events(r, group_of_row, cols_of_group, dt_us):
    """All fixation and saccade events of every group (slide or run) at once, sorted by (group, onset)."""
    K = cols_of_group.shape[1]
    sel = np.flatnonzero((group_of_row >= 0) & (r.kind > 0) & ~np.isnan(r.midx))
    g, k, m = group_of_row[sel], r.kind[sel], r.midx[sel]
    order = np.lexsort((sel, m, k, g))
    sel, g, k, m = sel[order], g[order], k[order], m[order]
    new = np.ones(len(sel), bool)
    new[1:] = (g[1:] != g[:-1]) | (k[1:] != k[:-1]) | (m[1:] != m[:-1])
    st = np.flatnonzero(new)
    en = np.append(st[1:], len(sel))
    ev_g, ev_k, nrow = g[st], k[st], en - st
    start = r.ts[sel[st]].astype(float)
    end = r.ts[sel[en - 1]] + dt_us
    # AOI of fixation events: > half of the rows hit one item
    aoi = np.full(len(st), -1, np.intp)
    cx = np.full(len(st), np.nan)
    cy = np.full(len(st), np.nan)
    isfix_row = k == FIX
    fr = sel[isfix_row]
    fstarts = np.flatnonzero(new[isfix_row])
    hits = np.add.reduceat(r.aoi[fr[:, None], cols_of_group[g[isfix_row]]].astype(np.int32), fstarts, axis=0)
    fev = np.flatnonzero(ev_k == FIX)
    mx = hits.max(1)
    aoi[fev] = np.where(mx > nrow[fev] / 2, hits.argmax(1), -1)
    for src, dst in ((r.fx, cx), (r.fy, cy)):
        v = src[fr]
        fin = np.isfinite(v)
        s = np.add.reduceat(np.where(fin, v, 0.0), fstarts)
        c = np.add.reduceat(fin.astype(float), fstarts)
        with np.errstate(invalid="ignore", divide="ignore"):
            dst[fev] = np.where(c > 0, s / c, np.nan)
    # sort events by (group, onset); ties keep fixation before saccade (as in the original list order)
    o = np.lexsort((ev_k, start, ev_g))
    inv = np.empty(len(o), np.intp)
    inv[o] = np.arange(len(o))
    ev_g, ev_k, start, end, aoi, cx, cy = (a[o] for a in (ev_g, ev_k, start, end, aoi, cx, cy))
    prev_sac = start.copy()
    ps = np.zeros(len(o), bool)
    ps[1:] = (ev_g[1:] == ev_g[:-1]) & (ev_k[:-1] == SAC)
    prev_sac[ps] = start[np.flatnonzero(ps) - 1]
    # rows of fixation events (for the pupil of an item): position in the sorted event list
    row_ev = inv[np.cumsum(new)[isfix_row] - 1]
    return Events(ev_g, ev_k, start, end, aoi, cx, cy, prev_sac, g[isfix_row], row_ev, fr)


def _seg_stats(key, dur_ms, nkeys):
    """total, mean, min, max, count, first of `dur_ms` per key (entries in time order); 0 for empty keys."""
    cnt = np.bincount(key, minlength=nkeys).astype(float)
    tot = np.bincount(key, weights=dur_ms, minlength=nkeys)
    mn = np.full(nkeys, np.inf)
    mx = np.full(nkeys, -np.inf)
    np.minimum.at(mn, key, dur_ms)
    np.maximum.at(mx, key, dur_ms)
    first = np.zeros(nkeys)
    u, i = np.unique(key, return_index=True)
    first[u] = dur_ms[i]
    has = cnt > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(has, tot / cnt, 0.0)
    return np.stack([tot, mean, np.where(has, mn, 0.0), np.where(has, mx, 0.0), cnt, first], axis=1)


# ----------------------------------------------------------------------------- the paper's slide features
SLIDE_COLUMNS = None


def slide_features_fast(r):
    """Columns of `edd.celniak.slide_features` (numeric ones), one entry per slide, as a dict of arrays."""
    dt_us = float(np.median(np.diff(r.ts)))
    pupil = pupil_trace(r)
    baseline = pupil_baseline_fast(r, pupil)
    S, K = r.slide_cols.shape
    grp = np.where(r.stim >= 0, r.slide_of_code[np.maximum(r.stim, 0)], -1)
    ev = _events(r, grp, r.slide_cols, dt_us)
    isf = ev.kind == FIX
    # ---- fixation sequence per slide
    fg, fs_, fe, fa, fps = ev.g[isf], ev.start[isf], ev.end[isf], ev.aoi[isf], ev.prev_sac_start[isf]
    nf = len(fg)
    same_prev = np.zeros(nf, bool)
    same_prev[1:] = (fg[1:] == fg[:-1]) & (fa[1:] == fa[:-1])
    new_run = (fa >= 0) & ~same_prev
    last = np.zeros(nf, bool)
    last[:-1] = (fg[:-1] == fg[1:]) & (fa[:-1] == fa[1:])
    last = (fa >= 0) & ~last
    run_g, run_a = fg[new_run], fa[new_run]
    run_start, run_gl = fs_[new_run], fps[new_run]
    run_end = fe[last]
    onaoi = fa >= 0
    dur_f = (fe - fs_) / 1e3
    keys = {"fixation": (fg[onaoi] * K + fa[onaoi], dur_f[onaoi]),
            "visit": (run_g * K + run_a, (run_end - run_start) / 1e3),
            "glance": (run_g * K + run_a, (run_end - run_gl) / 1e3)}
    st = np.concatenate([_seg_stats(k, d, S * K).reshape(S, K, 6) for k, d in keys.values()], axis=2)   # (S, K, 18)
    # ---- pupil of the fixations on each item
    pm = ev.row_ev
    rows_aoi = ev.aoi[pm]
    ok = rows_aoi >= 0
    pv = pupil[ev.row_pos[ok]]
    fin = np.isfinite(pv)
    key = ev.rows_g[ok] * K + rows_aoi[ok]
    ps = np.bincount(key, weights=np.where(fin, pv, 0.0), minlength=S * K)
    pc = np.bincount(key, weights=fin.astype(float), minlength=S * K)
    with np.errstate(invalid="ignore", divide="ignore"):
        pup = np.where(pc > 0, ps / pc, np.nan).reshape(S, K) - baseline
    # ---- slide level
    rows = np.flatnonzero(grp >= 0)
    gg = grp[rows]
    n_samples = np.bincount(gg, minlength=S)
    both = (r.vl[rows] & r.vr[rows]).astype(float)
    valid_frac = np.bincount(gg, weights=both, minlength=S) / n_samples
    pr_ = pupil[rows]
    fr_ = np.isfinite(pr_)
    pcs = np.bincount(gg, weights=fr_.astype(float), minlength=S)
    with np.errstate(invalid="ignore", divide="ignore"):
        slide_pup = np.where(pcs > 0, np.bincount(gg, weights=np.where(fr_, pr_, 0.0), minlength=S) / pcs, np.nan) - baseline
    # ---- target vs median of the other items
    ar = np.arange(S)
    tpos = r.slide_target
    mask_other = np.ones((S, K), bool)
    mask_other[ar, tpos] = False
    others = np.where(mask_other[:, :, None], st, np.nan)                       # (S, K, 18)
    oth_pup = np.where(mask_other, pup, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ntm = np.nanmedian(others, axis=1)                                   # (S, 18)
            ntm_pup = np.nanmedian(oth_pup, axis=1)
    tgt = st[ar, tpos]
    tgt_pup = pup[ar, tpos]
    out = {"question": r.slide_q, "repetition": r.slide_rep, "valid_frac": valid_frac, "slide_pupil_bc": slide_pup,
           "n_samples": n_samples}
    names = [f"{k}_{s}" for k in KINDS for s in STATS]
    for j, nm in enumerate(names):
        out[f"tgt_{nm}"], out[f"ntm_{nm}"] = tgt[:, j], ntm[:, j]
        tot = tgt[:, j] + ntm[:, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"norm_{nm}"] = np.where(tot > 0, tgt[:, j] / tot, 0.5)
    out["tgt_pupil_bc"], out["ntm_pupil_bc"] = tgt_pup, ntm_pup
    out["pupil_bc"] = np.where(~np.isnan(tgt_pup), tgt_pup, slide_pup)
    return out


def paper_matrix(out, features):
    """(n_slides, len(features)) matrix in the order of `features` (e.g. `edd.celniak.PAPER_FEATURES`)."""
    return np.column_stack([out[f] for f in features])


# ----------------------------------------------------------------------------- global gaze
GG = ["sl_nfix", "sl_fixdur", "sl_nsacc", "sl_nblink", "sl_scanpath", "sl_aoi_dwell", "valid_frac"]


def global_gaze_fast(r):
    """The 7 global-gaze features of one recording (mean over its slide runs; NaN skipped), as in `lean_global`."""
    dt_us = float(np.median(np.diff(r.ts)))
    dt_ms = dt_us / 1e3
    blinks = blink_onsets_fast(r, dt_ms)
    # contiguous runs of named samples (`phase2_celniak_features.stimulus_runs`), restricted to slide stimuli
    named = np.flatnonzero(r.stim >= 0)
    nm = r.stim[named]
    run_of_named = np.cumsum(np.concatenate([[0], nm[1:] != nm[:-1]]))
    first = np.concatenate([[0], np.flatnonzero(nm[1:] != nm[:-1]) + 1])
    run_code = nm[first]
    run_slide = r.slide_of_code[run_code]
    is_slide = run_slide >= 0
    slide_run_idx = np.cumsum(is_slide) - 1                                       # index among slide runs
    G = int(is_slide.sum())
    cols_of_group = r.slide_cols[run_slide[is_slide]]
    grp = np.full(len(r.ts), -1, np.intp)
    keep = is_slide[run_of_named]
    grp[named[keep]] = slide_run_idx[run_of_named[keep]]
    ev = _events(r, grp, cols_of_group, dt_us)
    isf = ev.kind == FIX
    fg, fs_, fe, fa = ev.g[isf], ev.start[isf], ev.end[isf], ev.aoi[isf]
    dur = (fe - fs_) / 1e3
    nfix = np.bincount(fg, minlength=G).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fixdur = np.where(nfix > 0, np.bincount(fg, weights=dur, minlength=G) / nfix, np.nan)
    nsacc = np.bincount(ev.g[~isf], minlength=G).astype(float)
    nblink = np.bincount(grp[blinks][grp[blinks] >= 0], minlength=G).astype(float)
    cx, cy = ev.cx[isf], ev.cy[isf]
    step = np.hypot(np.diff(cx), np.diff(cy))
    same = fg[1:] == fg[:-1]
    ok = same & np.isfinite(step)
    scan = np.bincount(fg[1:][ok], weights=step[ok], minlength=G)
    dwell = np.bincount(fg[fa >= 0], weights=dur[fa >= 0], minlength=G)
    rows = np.flatnonzero(grp >= 0)
    n = np.bincount(grp[rows], minlength=G)
    vf = np.bincount(grp[rows], weights=(r.vl[rows] & r.vr[rows]).astype(float), minlength=G) / n
    M = np.column_stack([nfix, fixdur, nsacc, nblink, scan, dwell, vf])
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(M, axis=0)
