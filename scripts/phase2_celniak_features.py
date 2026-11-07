"""Phase 2 (celniak track): feature extraction from the raw 120 Hz Tobii samples.

Every trial is question video (~6.5 s) -> fixation cross (~0.5 s) -> 4-item slide (7.5 s). For each trial this
script derives:

* properly preprocessed pupil (per-eye validity masking, dilation-speed outlier rejection after Kret &
  Sjak-Shie 2019, 50 ms blink-margin padding, eye merging with an inter-eye offset, linear interpolation of
  gaps <= 500 ms, 5-sample moving average), with a trial-locked subtractive baseline from the immediately
  preceding fixation cross and event-locked windows 0-800, 800-1500, 1500-7500 ms;
* pre-stimulus features during the question video and the cross (gaze dispersion, fixation rate and
  duration, saccades, blinks, pupil level and slope);
* orienting / first-look and dwell features for every item AOI (one row per slide x item), including
  per-item pupil while the item is looked at;
* a 30 Hz multichannel time series per slide (pupil, gaze x/y, probe and irrelevant hit) for MiniRocket.

Usage: python -I scripts/phase2_celniak_features.py <cache_dir> <out_dir> [--jobs 2]
Writes <out_dir>/celniak_p2_trials.csv, celniak_p2_items.csv, celniak_p2_series.npz
"""
import argparse
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edd.celniak import PROBES, load_cached  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices in invalid periods -> NaN by design

SLIDE_RE = re.compile(r"^(\d)_(\d)$")
SLIDE_MS = 7500.0
MAX_GAP_MS = 500.0
BLINK_PAD_MS = 50.0
SERIES_BIN_MS = 1000.0 / 30.0
N_BINS = int(round(SLIDE_MS / SERIES_BIN_MS))  # 225
PUPIL_WINDOWS = {"w1": (0, 800), "w2": (800, 1500), "w3": (1500, 7500)}


# ---------------------------------------------------------------- pupil preprocessing

def _runs(mask):
    """(start, stop) index pairs of True runs."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _dilate(mask, pad):
    if pad <= 0:
        return mask
    k = np.ones(2 * pad + 1)
    return np.convolve(mask.astype(float), k, mode="same") > 0


def clean_pupil(samples, dt_ms):
    """Blink-aware, artefact-rejected, merged and interpolated pupil trace (mm) for the whole recording."""
    eyes = []
    pad = int(round(BLINK_PAD_MS / dt_ms))
    for side in ("l", "r"):
        p = samples[f"pupil_{side}"].to_numpy(float).copy()
        valid = samples[f"valid_{side}"].eq("Valid").to_numpy()
        p[~valid | (p < 1.5) | (p > 9.0)] = np.nan
        # dilation-speed filter: max of backward/forward absolute change per ms, MAD threshold (n = 10)
        d = np.abs(np.diff(p)) / dt_ms
        back = np.concatenate([[np.nan], d])
        fwd = np.concatenate([d, [np.nan]])
        speed = np.fmax(back, fwd)
        med = np.nanmedian(speed)
        mad = np.nanmedian(np.abs(speed - med))
        p[speed > med + 10 * mad] = np.nan
        p[_dilate(np.isnan(p), pad)] = np.nan
        eyes.append(p)
    left, right = eyes
    both = ~np.isnan(left) & ~np.isnan(right)
    offset = float(np.nanmean(left[both] - right[both])) if both.any() else 0.0
    merged = np.where(both, (left + right) / 2,
                      np.where(~np.isnan(left), left - offset / 2, right + offset / 2))
    raw_valid = ~np.isnan(merged)
    max_gap = int(round(MAX_GAP_MS / dt_ms))
    idx = np.arange(len(merged))
    for a, b in _runs(np.isnan(merged)):
        if a == 0 or b >= len(merged) or (b - a) > max_gap:
            continue
        merged[a:b] = np.interp(idx[a:b], [a - 1, b], [merged[a - 1], merged[b]])
    smooth = pd.Series(merged).rolling(5, center=True, min_periods=1).mean().to_numpy().copy()
    smooth[np.isnan(merged)] = np.nan
    return smooth, raw_valid


def blink_onsets(samples, dt_ms):
    """Indices of blink onsets: both eyes invalid for 75-500 ms."""
    bad = ~(samples["valid_l"].eq("Valid").to_numpy() | samples["valid_r"].eq("Valid").to_numpy())
    return np.array([a for a, b in _runs(bad) if 75 <= (b - a) * dt_ms <= 500], int)


# ---------------------------------------------------------------- per-period descriptors

def _fixations(seg, aoi_cols, dt_us):
    """Fixation events in a segment: onset/offset (us), centroid, majority AOI (or None)."""
    fx = seg[seg["move_type"] == "Fixation"]
    out = []
    for _, g in fx.groupby("move_index", sort=False):
        ev = {"start": g["ts_us"].iloc[0], "end": g["ts_us"].iloc[-1] + dt_us,
              "x": float(g["fix_x"].mean()), "y": float(g["fix_y"].mean()), "aoi": None, "rows": g.index.to_numpy()}
        if aoi_cols:
            hits = g[aoi_cols].sum()
            if hits.max() > len(g) / 2:
                ev["aoi"] = hits.idxmax()
        out.append(ev)
    out.sort(key=lambda e: e["start"])
    return out


def _period_stats(seg, pupil, blinks_idx, dt_us, prefix):
    """Gaze dispersion, fixation rate, saccades, blinks and pupil for a non-slide period (video/cross)."""
    n = len(seg)
    dur_s = n * dt_us / 1e6
    x, y = seg["gaze_x"].to_numpy(float), seg["gaze_y"].to_numpy(float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    fx = _fixations(seg, [], dt_us)
    fdur = np.array([(e["end"] - e["start"]) / 1e3 for e in fx]) if fx else np.array([])
    nsacc = seg.loc[seg["move_type"] == "Saccade", "move_index"].nunique()
    rows = seg.index.to_numpy()
    nblink = int(np.isin(blinks_idx, rows).sum())
    p = pupil[rows]
    third = max(1, n // 3)
    return {
        f"{prefix}_dur_ms": n * dt_us / 1e3,
        f"{prefix}_valid": float(ok.mean()) if n else np.nan,
        f"{prefix}_disp": float(np.sqrt(np.nanvar(x[ok]) + np.nanvar(y[ok]))) if ok.sum() > 2 else np.nan,
        f"{prefix}_sdx": float(np.nanstd(x[ok])) if ok.sum() > 2 else np.nan,
        f"{prefix}_sdy": float(np.nanstd(y[ok])) if ok.sum() > 2 else np.nan,
        f"{prefix}_nfix": float(len(fx)),
        f"{prefix}_fixrate": float(len(fx) / dur_s) if dur_s else np.nan,
        f"{prefix}_fixdur": float(fdur.mean()) if len(fdur) else np.nan,
        f"{prefix}_nsacc": float(nsacc),
        f"{prefix}_nblink": float(nblink),
        f"{prefix}_blinkrate": float(nblink / dur_s) if dur_s else np.nan,
        f"{prefix}_pupil": float(np.nanmean(p)) if np.isfinite(p).any() else np.nan,
        f"{prefix}_pupil_slope": float(np.nanmean(p[-third:]) - np.nanmean(p[:third]))
        if np.isfinite(p[-third:]).any() and np.isfinite(p[:third]).any() else np.nan,
        f"{prefix}_pupil_late": float(np.nanmean(p[-int(round(1e6 / dt_us)):])) if np.isfinite(p).any() else np.nan,
    }


def _visits(fixations):
    """Consecutive fixations on the same AOI merged into visits: list of (aoi, start, end)."""
    visits, run = [], None
    for ev in fixations:
        if run is not None and ev["aoi"] == run[0]:
            run[2] = ev["end"]
            continue
        if run is not None and run[0] is not None:
            visits.append(tuple(run))
        run = [ev["aoi"], ev["start"], ev["end"]]
    if run is not None and run[0] is not None:
        visits.append(tuple(run))
    return visits


# ---------------------------------------------------------------- recording

def stimulus_runs(samples):
    """Contiguous runs of named stimuli: list of (name, row index array). Rows without a stimulus name
    (event rows; in some 250 Hz recordings they are interleaved within a stimulus) are skipped, and
    consecutive runs of the same stimulus separated only by such rows are merged."""
    st = samples["stimulus"]
    named = np.flatnonzero(st.notna().to_numpy())
    names = st.iloc[named].astype(str).to_numpy()
    brk = np.flatnonzero(names[1:] != names[:-1]) + 1
    starts = np.concatenate([[0], brk])
    stops = np.concatenate([brk, [len(named)]])
    return [(names[a], named[a:b]) for a, b in zip(starts, stops)]


def process_recording(cache_dir, name):
    t0 = time.perf_counter()
    samples, meta = load_cached(cache_dir, name)
    res = process_samples(samples, meta)
    return (*res, time.perf_counter() - t0, len(samples))


def process_samples(samples, meta):
    """All trial rows, item rows and slide series for one recording's raw samples."""
    samples = samples.reset_index(drop=True)
    dt_us = float(samples["ts_us"].diff().median())
    dt_ms = dt_us / 1e3
    pupil, _ = clean_pupil(samples, dt_ms)
    blinks = blink_onsets(samples, dt_ms)
    runs = stimulus_runs(samples)
    label = int(str(meta["group"]).strip().lower() == "guilty")
    base = {"recording": meta["Recording name"], "participant": meta["Participant name"], "label": label,
            "first_exam": meta.get("FirstExamination"),
            "val_acc": meta.get("Average validation accuracy (degrees)"), "fs_hz": round(1e6 / dt_us, 1)}
    trials, items, series, keys = [], [], [], []
    order = 0
    for i, (stim, rows) in enumerate(runs):
        m = SLIDE_RE.match(stim)
        if not m:
            continue
        q, rep = int(m.group(1)), int(m.group(2))
        cross = runs[i - 1] if i >= 1 and runs[i - 1][0] == "cross" else None
        video = runs[i - 2] if cross is not None and i >= 2 and not SLIDE_RE.match(runs[i - 2][0]) \
            and runs[i - 2][0] not in ("cross", "Eyetracker Calibration") else None
        slide = samples.iloc[rows]
        t_rel = (slide["ts_us"].to_numpy(float) - slide["ts_us"].iloc[0]) / 1e3
        row = {**base, "question": q, "repetition": rep, "order": order,
               "video": video[0] if video else None,
               "valid_frac": float((slide["valid_l"].eq("Valid") & slide["valid_r"].eq("Valid")).mean()),
               "pupil_valid_frac": float(np.isfinite(pupil[rows]).mean())}
        order += 1
        # ---- pre-stimulus periods
        if video is not None:
            row.update(_period_stats(samples.iloc[video[1]], pupil, blinks, dt_us, "vid"))
        if cross is not None:
            row.update(_period_stats(samples.iloc[cross[1]], pupil, blinks, dt_us, "cross"))
        # ---- trial-locked baseline: the cross; fall back to the last 500 ms of the video
        bl = np.nan
        if cross is not None and np.isfinite(pupil[cross[1]]).mean() >= 0.3:
            bl = float(np.nanmean(pupil[cross[1]]))
        elif video is not None:
            tail = video[1][-int(round(500 / dt_ms)):]
            if np.isfinite(pupil[tail]).mean() >= 0.3:
                bl = float(np.nanmean(pupil[tail]))
        row["baseline_pupil"] = bl
        row["pre_dilation"] = bl - row.get("vid_pupil_late", np.nan)
        p_bc = pupil[rows] - bl
        for w, (a, b) in PUPIL_WINDOWS.items():
            sel = (t_rel >= a) & (t_rel < b)
            row[f"pup_{w}"] = float(np.nanmean(p_bc[sel])) if np.isfinite(p_bc[sel]).any() else np.nan
        early = (t_rel < 1500) & np.isfinite(p_bc)
        late = (t_rel >= 1500) & (t_rel < 3500) & np.isfinite(p_bc)
        row["pup_min_early"] = float(p_bc[early].min()) if early.any() else np.nan
        row["pup_peak_late"] = float(p_bc[late].max()) if late.any() else np.nan
        # ---- slide gaze behaviour
        aoi_cols = [c for c in samples.columns if c.startswith(f"aoi:{stim} - ")]
        probe_col = f"aoi:{stim} - {PROBES[q]}"
        fx = _fixations(slide, aoi_cols, dt_us)
        onset = slide["ts_us"].iloc[0]
        nblink = int(np.isin(blinks, rows).sum())
        row.update({"sl_nfix": float(len(fx)),
                    "sl_fixdur": float(np.mean([(e["end"] - e["start"]) / 1e3 for e in fx])) if fx else np.nan,
                    "sl_nsacc": float(slide.loc[slide["move_type"] == "Saccade", "move_index"].nunique()),
                    "sl_nblink": float(nblink),
                    "sl_scanpath": float(np.nansum(np.hypot(np.diff([e["x"] for e in fx]), np.diff([e["y"] for e in fx]))))
                    if len(fx) > 1 else 0.0,
                    "sl_aoi_dwell": float(sum((e["end"] - e["start"]) / 1e3 for e in fx if e["aoi"] is not None))})
        on_aoi = [e for e in fx if e["aoi"] is not None]
        first = on_aoi[0] if on_aoi else None
        row["first_item_latency"] = (first["start"] - onset) / 1e3 if first else np.nan
        row["first_is_probe"] = float(first["aoi"] == probe_col) if first else np.nan
        visits = _visits(fx)
        # ---- per-item rows
        hit = slide[aoi_cols].to_numpy(bool)
        for j, c in enumerate(aoi_cols):
            f_c = [e for e in on_aoi if e["aoi"] == c]
            v_c = [v for v in visits if v[0] == c]
            gx = slide["fix_x"].to_numpy(float)[hit[:, j]]
            gy = slide["fix_y"].to_numpy(float)[hit[:, j]]
            if not np.isfinite(gx).any():
                gx = slide["gaze_x"].to_numpy(float)[hit[:, j]]
                gy = slide["gaze_y"].to_numpy(float)[hit[:, j]]
            on_late = hit[:, j] & (t_rel >= 1000) & np.isfinite(p_bc)
            items.append({**base, "question": q, "repetition": rep, "item": c.split(" - ", 1)[1],
                          "is_probe": int(c == probe_col),
                          "dwell": float(sum((e["end"] - e["start"]) / 1e3 for e in f_c)),
                          "dwell_gaze": float(hit[:, j].sum() * dt_ms),
                          "nfix": float(len(f_c)), "nvisit": float(len(v_c)),
                          "entry_latency": (f_c[0]["start"] - onset) / 1e3 if f_c else np.nan,
                          "first_visit": (v_c[0][2] - v_c[0][1]) / 1e3 if v_c else 0.0,
                          "is_first": float(first is not None and first["aoi"] == c),
                          "pupil_on": float(p_bc[on_late].mean()) if on_late.any() else np.nan,
                          "cx": float(np.nanmean(gx)) if np.isfinite(gx).any() else np.nan,
                          "cy": float(np.nanmean(gy)) if np.isfinite(gy).any() else np.nan,
                          "valid_frac": row["valid_frac"]})
        # probe-specific orienting summary on the slide row
        pr = items[-len(aoi_cols):]
        prb = [it for it in pr if it["is_probe"]][0]
        irr = [it for it in pr if not it["is_probe"]]
        row["probe_latency"] = prb["entry_latency"] if np.isfinite(prb["entry_latency"]) else SLIDE_MS
        row["probe_first_visit"] = prb["first_visit"]
        row["irr_first_visit"] = float(np.mean([it["first_visit"] for it in irr]))
        row["probe_pupil_minus_irr"] = prb["pupil_on"] - np.nanmean([it["pupil_on"] for it in irr]) \
            if np.isfinite([it["pupil_on"] for it in irr]).any() else np.nan
        trials.append(row)
        # ---- 30 Hz series for MiniRocket: pupil (bc), gaze x, gaze y, probe hit, irrelevant hit
        b = np.minimum((t_rel // SERIES_BIN_MS).astype(int), N_BINS - 1)
        chans = [p_bc, slide["gaze_x"].to_numpy(float) / 1920.0, slide["gaze_y"].to_numpy(float) / 1080.0,
                 hit[:, aoi_cols.index(probe_col)].astype(float),
                 hit[:, [k for k, c in enumerate(aoi_cols) if c != probe_col]].any(axis=1).astype(float)]
        ser = np.zeros((len(chans), N_BINS))
        for ci, ch in enumerate(chans):
            s = pd.Series(ch).groupby(b).mean().reindex(range(N_BINS))
            ser[ci] = s.interpolate(limit_direction="both").fillna(0.0).to_numpy()
        series.append(ser)
        keys.append((base["recording"], q, rep))
    return pd.DataFrame(trials), pd.DataFrame(items), np.stack(series), keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    names = sorted({p.name.split(".")[0] for p in Path(a.cache_dir).glob("Recording*.csv.gz")},
                   key=lambda s: int(s.replace("Recording", "")))
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        res = list(pool.map(process_recording, [a.cache_dir] * len(names), names))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trials = pd.concat([r[0] for r in res], ignore_index=True)
    items = pd.concat([r[1] for r in res], ignore_index=True)
    trials.to_csv(out / "celniak_p2_trials.csv", index=False)
    items.to_csv(out / "celniak_p2_items.csv", index=False)
    keys = [k for r in res for k in r[3]]
    np.savez_compressed(out / "celniak_p2_series.npz", X=np.concatenate([r[2] for r in res]),
                        recording=np.array([k[0] for k in keys]), question=np.array([k[1] for k in keys]),
                        repetition=np.array([k[2] for k in keys]))
    secs = [r[4] for r in res]
    n = [r[5] for r in res]
    print(f"{len(trials)} trials, {len(items)} item rows, {trials.recording.nunique()} recordings; "
          f"extraction {np.median(secs):.2f} s/recording (median), {np.sum(n) / np.sum(secs):,.0f} samples/s")


if __name__ == "__main__":
    main()
