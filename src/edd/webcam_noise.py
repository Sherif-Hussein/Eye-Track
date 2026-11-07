"""Phase 3, experiment `webcam_noise` (library part): realistic webcam spatial error on 30 Hz gaze.

Nothing existing is modified. The Delmas raw-sample pipeline (`edd.delmas_raw`), the track script
(`scripts/phase2_delmas_raw.py`, loaded from its file by `edd.delmas_raw_matched`) and the frozen folds are reused.
Noise is applied to the decimated 30 Hz gaze (pixels) BEFORE event detection and AOI assignment.
"""
import contextlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from edd import delmas_raw as dr

ROOT = Path(__file__).resolve().parents[2]
FS = 30
SEED = 20261102                    # base seed of every noise draw: SeedSequence([SEED, draw, participant])
DROPOUT = 0.10                     # fraction of samples lost at every non-zero level
JITTER_BASE, JITTER_SLOPE = 0.5, 0.25       # per-axis jitter SD [deg] = 0.5 + 0.25 * offset
K_ADAPT = 3.0                      # adaptive I-VT: threshold = 30 + K_ADAPT * (noise-induced velocity SD per axis)
RAYLEIGH_MEAN = math.sqrt(math.pi / 2)      # mean of a unit-sigma Rayleigh distribution


# ----------------------------------------------------------------------------- geometry
def geometry_from_design(design_mat_path):
    """Viewing geometry from the authors' Psychtoolbox design file: distance [mm], screen width [mm], x resolution,
    and the px/deg they computed. Returns also the px/deg recomputed from distance and width (small-angle, screen centre)."""
    import scipy.io as sio
    m = sio.loadmat(str(design_mat_path), squeeze_me=True, struct_as_record=False)
    scr, vis = m["scr"], m["visual"]
    dist_mm = float(scr.subDist) * 10.0
    width_mm, xres = float(scr.width), float(scr.xres)
    recomputed = math.tan(math.radians(1.0)) * dist_mm / (width_mm / xres)
    return {"viewing_distance_cm": float(scr.subDist), "screen_width_mm": width_mm, "screen_height_mm": float(scr.height),
            "xres": xres, "yres": float(scr.yres), "refresh_hz": float(scr.hz),
            "pix_per_deg_design_file": float(vis.pixPerDeg), "pix_per_deg_recomputed": recomputed}


# ----------------------------------------------------------------------------- noise
def noise_params(level_deg):
    """Pre-registered parameters at offset level `level_deg` (mean offset magnitude across participants)."""
    if level_deg == 0:
        return {"offset_mean_deg": 0.0, "offset_sigma_per_axis_deg": 0.0, "jitter_sd_per_axis_deg": 0.0, "dropout": 0.0}
    return {"offset_mean_deg": float(level_deg), "offset_sigma_per_axis_deg": level_deg / RAYLEIGH_MEAN,
            "jitter_sd_per_axis_deg": JITTER_BASE + JITTER_SLOPE * level_deg, "dropout": DROPOUT}


def adaptive_threshold(level_deg, fs=FS):
    """Velocity threshold [deg/s] scaled with the known jitter: 30 + K * SD of the per-axis velocity noise of the
    +-h-sample differentiator used by edd.delmas_raw.ivt (SD = sqrt(2) * jitter / span)."""
    if level_deg == 0:
        return dr.VEL_THR
    h = max(1, int(round(fs * 0.004)))
    span = 2 * h / fs
    sd_v = math.sqrt(2.0) * noise_params(level_deg)["jitter_sd_per_axis_deg"] / span
    return dr.VEL_THR + K_ADAPT * sd_v


def apply_noise(x, y, p, level_deg, draw, pid):
    """x, y (pixels), p (pupil) of shape (trials, samples) at 30 Hz. Returns noisy copies and the realised parameters.
    All random numbers depend on (draw, participant) only; the level only scales them (common random numbers across
    levels, so the degradation curve is not blurred by independent re-draws).
    (a) one 2-D offset per participant for the whole session, components N(0, sigma_off^2) -> magnitude Rayleigh with
        mean = level, direction uniform; (b) iid Gaussian jitter per sample and axis; (c) iid sample loss (x, y, pupil)."""
    if level_deg == 0:
        return x.copy(), y.copy(), p.copy(), {"offset_deg": 0.0, "lost_fraction_added": 0.0}
    rng = np.random.default_rng([SEED, int(draw), int(pid)])
    z_off = rng.standard_normal(2)
    z_jit = rng.standard_normal((2,) + x.shape)
    u = rng.random(x.shape)
    prm = noise_params(level_deg)
    ppd = dr.PIX_PER_DEG
    off_deg = prm["offset_sigma_per_axis_deg"] * z_off
    xn = x + (off_deg[0] + prm["jitter_sd_per_axis_deg"] * z_jit[0]) * ppd
    yn = y + (off_deg[1] + prm["jitter_sd_per_axis_deg"] * z_jit[1]) * ppd
    pn = p.copy()
    drop = u < DROPOUT
    xn[drop] = np.nan
    yn[drop] = np.nan
    pn[drop] = np.nan
    return xn, yn, pn, {"offset_deg": float(np.hypot(*off_deg)), "lost_fraction_added": float(drop.mean())}


# ----------------------------------------------------------------------------- features
@contextlib.contextmanager
def ivt_threshold(thr):
    """Temporarily run edd.delmas_raw.trial_features with I-VT velocity threshold `thr` (module attribute swap in this
    process only; the file is untouched). thr = None keeps the clean pipeline's fixed 30 deg/s."""
    orig = dr.ivt
    if thr is not None:
        dr.ivt = lambda x, y, fs, _o=orig, _t=thr: _o(x, y, fs, vel_thr=_t)
    try:
        yield
    finally:
        dr.ivt = orig


def participant_face_rows(seg, group, pid, level_deg, draw, thr=None):
    """Per-(trial, quadrant) feature rows of one participant at 30 Hz after noise. Same columns as
    data/derived/delmas_raw_face_trials.csv.gz (el_nfix is NaN, as at every rate but 1000 Hz)."""
    X, Y, P = seg["x30"], seg["y30"], seg["p30"]
    xn, yn, pn, info = apply_noise(X, Y, P, level_deg, draw, pid)
    rows = []
    with ivt_threshold(thr):
        for k in range(len(X)):
            F = dr.trial_features(xn[k], yn[k], pn[k], FS, with_msacc=True)       # msacc is NaN below 500 Hz
            fam = int((seg["images"][k] >= 57).any())
            for q in range(4):
                rows.append([pid, group, FS, k, q, int(seg["images"][k, q]), fam, np.nan, *F[q]])
    cols = ["participant", "group", "rate", "trial", "quad", "image", "fam_trial", "el_nfix"] + dr.TRIAL_FEATURES
    df = pd.DataFrame(rows, columns=cols)
    info["missing_fraction_after_noise"] = float(np.mean(~np.isfinite(xn)))
    return df, info


def build_table_30(face, books):
    """Same aggregation as phase2_delmas_raw.build_tables for rate 30 (per-presentation means over every presentation in
    the set's trial window, merge with the authors' labels, per-participant z-scores), without the label-dependent
    published-rule columns (not used by any model here)."""
    out = []
    for s, book in books.items():
        f = face if s == "64" else face[face.trial >= 32]
        f = f[f.participant.isin(book.participant.unique())]
        agg = f.groupby(["rate", "participant", "image"])[dr.TRIAL_FEATURES].mean().reset_index()
        npres = f.groupby(["rate", "participant", "image"]).size().rename("n_pres").reset_index()
        agg = agg.merge(npres, on=["rate", "participant", "image"])
        agg = agg.merge(book[["participant", "image", "group", "face"]], on=["participant", "image"], how="inner")
        agg["set"] = s
        out.append(agg)
    df = pd.concat(out, ignore_index=True)
    cols = dr.TRIAL_FEATURES[:8] + dr.TRIAL_FEATURES[8:]
    z = df.groupby(["rate", "set", "participant"])[cols].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    for c in cols:
        df[c + "_z"] = z[c]
    return df


# ----------------------------------------------------------------------------- statistics
def auc_matrix(y, S):
    """AUC of each column of S against binary y (rank formula, ties averaged: identical to sklearn.roc_auc_score)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    r = rankdata(S, axis=0)
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


class SetData:
    """One group-set: rows ordered by participant; columns = score columns of every configuration."""

    def __init__(self, y, parts, S):
        self.y, self.S = y, S
        self.bounds = []
        start = 0
        for p in np.unique(parts):
            n = int((parts == p).sum())
            self.bounds.append((start, start + n))
            start += n
        self.n_part = len(self.bounds)

    def rows_of(self, pick):
        return np.concatenate([np.arange(*self.bounds[i]) for i in pick])
