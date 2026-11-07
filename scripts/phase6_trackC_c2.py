"""Phase 6, Track C, analysis C2: event-detection validity of the Delmas I-VT pipeline at 1000, 500, 250, 120, 60, 30 Hz.
Pre-registration: results/phase6/trackC/PREREG.md (C2). Independent of the stored feature tables: samples and EyeLink-parser
events are read again from the EDF files.

Stages:
  extract - EDF -> scratch npz per participant (1000 Hz x, y, pupil per trial; parser fixations per trial)
  rates   - per rate: fixations per trial, durations, agreement with the parser (counts and dwell), saccade detectability
Usage (project root):  .venv/Scripts/python.exe -I scripts/phase6_trackC_c2.py --stages extract,rates --n_jobs 2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from edd import delmas_fast as df_  # noqa: E402
from edd import delmas_raw as dr  # noqa: E402
from edd.delmas import load_trials  # noqa: E402

OUT = ROOT / "results" / "phase6" / "trackC"
SCRATCH = Path(os.environ.get("TRACKC_SCRATCH", r"<local path>"))
CACHE = SCRATCH / "c2_raw"
RAW = ROOT / "data" / "delmas" / "raw" / "data_raw"
BOOK64 = ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"
RATES = [1000, 500, 250, 120, 60, 30]
PPD = dr.PIX_PER_DEG
AMP_BINS = [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]
MAX_GAP_MS = 150


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


# ----------------------------------------------------------------------------- extract
def load_halves(pid):
    """Trials of both session halves: x, y, p (n_trials, 5500) at 1000 Hz, images (n_trials, 4), parser fixations
    (trial, start_rel, end_rel, x, y) with start in [-PRE, POST) ms around display onset, the same trial cutting as
    `edd.delmas_raw.load_session`."""
    import scipy.io as sio
    d = RAW / f"Sj_{pid:02d}"
    Xs, imgs, fixes = [], [], []
    n_prev = 0
    for edf in sorted(d.glob(f"I{pid:02d}c2_*.edf")):
        mat = d / f"FI{pid:02d}c2_{edf.stem.split('_')[-1]}.mat"
        times, samp, msgs, fix = dr.read_edf_fast(edf)
        onsets = [t for t, m in msgs if m.startswith(b"EVENT_ParallelDisplay")]
        design = sio.loadmat(str(mat), squeeze_me=True, struct_as_record=False)["design"]
        trials = np.atleast_1d(design.trial)
        assert len(onsets) == len(trials), (pid, edf)
        n = dr.PRE_MS + dr.POST_MS
        X = np.full((len(trials), 3, n), np.nan)
        images = np.zeros((len(trials), 4), int)
        for k, (t0, tr) in enumerate(zip(onsets, trials)):
            a = int(np.searchsorted(times, t0 - dr.PRE_MS))
            b = int(np.searchsorted(times, t0 + dr.POST_MS))
            idx = times[a:b] - (t0 - dr.PRE_MS)
            X[k][:, idx] = samp[a:b].T
            f = fix[(fix[:, 0] >= t0 - dr.PRE_MS) & (fix[:, 0] < t0 + dr.POST_MS)]
            fixes.append(np.column_stack([np.full(len(f), n_prev + k), f[:, 0] - t0, f[:, 1] - t0, f[:, 2], f[:, 3]]))
            images[k, dr._quad_of_rect(tr.ciblePosition)] = int(tr.idcible)
            for j, dd in enumerate(np.atleast_1d(tr.distractors)):
                images[k, dr._quad_of_rect(getattr(tr, f"dist{j + 1}Position"))] = int(dd)
        x, y, p = X[:, 0], X[:, 1], X[:, 2]
        bad = ~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(p) | (p <= 0)
        x[bad] = y[bad] = p[bad] = np.nan
        Xs.append(X)
        imgs.append(images)
        n_prev += len(trials)
    X = np.concatenate(Xs)
    return X[:, 0].astype(np.float32), X[:, 1].astype(np.float32), X[:, 2].astype(np.float32), np.concatenate(imgs), np.vstack(fixes)


def extract_one(pid):
    path = CACHE / f"p{pid:02d}.npz"
    if path.exists():
        return pid, "cached"
    x, y, p, images, fix = load_halves(pid)
    tmp = CACHE / f"p{pid:02d}.tmp.npz"
    np.savez_compressed(tmp, x=x, y=y, p=p, images=images, fix=fix)
    tmp.replace(path)
    return pid, "extracted"


def participants():
    book = load_trials(BOOK64)
    return {int(p): int(g) for p, g in book.groupby("participant").group.first().items()}


def stage_extract(n_jobs):
    CACHE.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(extract_one)(p) for p in participants())
    log("extract:", {s: sum(1 for r in res if r[1] == s) for s in ("cached", "extracted")})


# ----------------------------------------------------------------------------- per participant, per rate
def parser_arrays(fix, T):
    """Per trial and face: parser fixation count (onset in [0, 5000) ms) and dwell in [200, 1000) and [1000, 5000) ms
    (fixation time clipped to the window, assigned to the face containing the fixation's mean gaze position, divided by the window length);
    per trial: all-fixation count, mean duration (ms), share of the display window inside fixations."""
    cnt = np.zeros((T, 4))
    dw1, dw2 = np.zeros((T, 4)), np.zeros((T, 4))
    n_all, dur_sum, covered = np.zeros(T), np.zeros(T), np.zeros(T)
    durs = []
    q = df_.aoi_batch(fix[:, 3], fix[:, 4]).astype(int) if len(fix) else np.array([], int)
    for (tr, st, en, _, _), qq in zip(fix, q):
        tr = int(tr)
        d = en - st + 1
        if 0 <= st < dr.POST_MS:
            n_all[tr] += 1
            dur_sum[tr] += d
            durs.append(d)
            if qq >= 0:
                cnt[tr, qq] += 1
        ov = max(0.0, min(en + 1, dr.POST_MS) - max(st, 0.0))
        covered[tr] += ov
        if qq >= 0:
            dw1[tr, qq] += max(0.0, min(en + 1, 1000) - max(st, 200.0))
            dw2[tr, qq] += max(0.0, min(en + 1, 5000) - max(st, 1000.0))
    return {"cnt": cnt, "dw1": dw1 / (1000 - 200), "dw2": dw2 / (5000 - 1000), "n_all": n_all,
            "dur_sum": dur_sum, "covered": covered / dr.POST_MS, "durs": np.array(durs)}


def sb_dwell(X, Y, fs, win):
    """Share of valid samples inside each face area in the window (no event detection). (T, 4)."""
    t = np.arange(X.shape[1]) * 1000.0 / fs - dr.PRE_MS
    m = (t >= win[0]) & (t < win[1])
    valid = (np.isfinite(X) & np.isfinite(Y))[:, m]
    q = df_.aoi_batch(X, Y)[:, m]
    nv = valid.sum(1)
    out = np.full((X.shape[0], 4), np.nan)
    ok = nv >= 0.2 * m.sum()
    for k in range(4):
        out[ok, k] = (q[ok] == k).sum(1) / nv[ok]
    return out


def ivt_rate(X, Y, P, fs, fix):
    """Everything at one rate for one participant. Returns dict of arrays."""
    T = X.shape[0]
    xs, ys, ps = df_.degrade_batch(X, Y, P, fs)
    xd, yd, fixm, (ftr, fst, fen) = df_.ivt_batch(xs, ys, fs)
    n = xs.shape[1]
    t = np.arange(n) * 1000.0 / fs - dr.PRE_MS
    # fixation list: centroid on raw (decimated) samples, window by onset time
    cx = np.array([np.nanmean(xs[a, s:e]) for a, s, e in zip(ftr, fst, fen)])
    cy = np.array([np.nanmean(ys[a, s:e]) for a, s, e in zip(ftr, fst, fen)])
    fq = df_.aoi_batch(cx, cy).astype(int) if len(cx) else np.array([], int)
    onset = t[fst]
    inw = (onset >= 0) & (onset < dr.POST_MS)
    dur = (fen - fst) * 1000.0 / fs
    cnt = np.zeros((T, 4))
    on = inw & (fq >= 0)
    np.add.at(cnt, (ftr[on], fq[on]), 1)
    n_all = np.bincount(ftr[inw], minlength=T).astype(float)
    dur_sum = np.bincount(ftr[inw], weights=dur[inw], minlength=T)
    win = (t >= 0) & (t < dr.POST_MS)
    covered = (fixm & win[None, :]).sum(1) / win.sum()
    F = df_.trial_features_batch(xs, ys, ps, fs, with_msacc=False)       # dwell_1 (col 0), dwell_2 (col 1), n_fix (col 2)
    assert np.array_equal(F[:, :, 2], cnt), "per-face count mismatch with the pipeline feature n_fix"
    return {"cnt": cnt, "dw1": F[:, :, 0], "dw2": F[:, :, 1], "n_all": n_all, "dur_sum": dur_sum, "covered": covered,
            "durs": dur[inw], "sb1": sb_dwell(xs, ys, fs, dr.PHASE1), "sb2": sb_dwell(xs, ys, fs, dr.PHASE2),
            "fix": (ftr, fst, fen), "t": t, "missing": float(np.mean(~np.isfinite(xs[:, win])))}


def transitions(fix, ivt, fs, T):
    """Consecutive parser fixations (both onsets in the display window, gap <= MAX_GAP_MS): amplitude and the fate at rate fs.
    Returns rows (amplitude_deg, status) with status 0 = resolved, 1 = merged, 2 = lost."""
    ftr, fst, fen = ivt["fix"]
    order = np.lexsort((fst, ftr))
    ftr, fst, fen = ftr[order], fst[order], fen[order]
    starts = np.searchsorted(ftr, np.arange(T), "left")
    ends = np.searchsorted(ftr, np.arange(T), "right")
    rows = []
    n = len(ivt["t"])
    trials = np.unique(fix[:, 0]).astype(int)
    byt = {tr: fix[fix[:, 0] == tr] for tr in trials}
    for tr, f in byt.items():
        f = f[(f[:, 1] >= 0) & (f[:, 1] < dr.POST_MS)]
        f = f[np.argsort(f[:, 1])]
        if len(f) < 2:
            continue
        a, b = starts[tr], ends[tr]
        s_, e_ = fst[a:b], fen[a:b]

        def fid_at(mid):
            i = int(round((mid + dr.PRE_MS) * fs / 1000.0))
            if i < 0 or i >= n or len(s_) == 0:
                return -1
            j = int(np.searchsorted(s_, i, "right")) - 1
            return j if j >= 0 and i < e_[j] else -1
        for k in range(len(f) - 1):
            gap = f[k + 1, 1] - f[k, 2]
            if gap > MAX_GAP_MS:
                continue
            amp = float(np.hypot(f[k + 1, 3] - f[k, 3], f[k + 1, 4] - f[k, 4]) / PPD)
            i1, i2 = fid_at((f[k, 1] + f[k, 2]) / 2.0), fid_at((f[k + 1, 1] + f[k + 1, 2]) / 2.0)
            status = 2 if (i1 < 0 or i2 < 0) else (0 if i1 != i2 else 1)
            rows.append((amp, status))
    return rows


def process(pid):
    z = np.load(CACHE / f"p{pid:02d}.npz")
    X, Y, P, images, fix = (z[k].astype(np.float64) if k in ("x", "y", "p") else z[k] for k in ("x", "y", "p", "images", "fix"))
    T = X.shape[0]
    par = parser_arrays(fix, T)
    res = {"images": images, "parser": par, "rates": {}}
    for fs in RATES:
        iv = ivt_rate(X, Y, P, fs, fix)
        iv["trans"] = np.array(transitions(fix, iv, fs, T)).reshape(-1, 2)
        del iv["fix"], iv["t"]
        res["rates"][fs] = iv
    win = slice(dr.PRE_MS, dr.PRE_MS + dr.POST_MS)
    res["missing_1000"] = float(np.mean(~np.isfinite(X[:, win])))
    return pid, res


# ----------------------------------------------------------------------------- aggregation
def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def levels_r(pids, results, key_ivt, key_par):
    """Pearson r at the four units for one metric. key_ivt: field of the rate dict; key_par: parser field."""
    out = {}
    for fs in RATES:
        A, B, P, I = [], [], [], []
        for pid in pids:
            r = results[pid]
            A.append(r["rates"][fs][key_ivt])
            B.append(r["parser"][key_par])
            P.append(np.full(A[-1].shape, pid))
            I.append(r["images"])
        A, B, P, I = (np.concatenate(v) for v in (A, B, P, I))
        a, b, p, im = A.ravel(), B.ravel(), P.ravel(), I.ravel()
        row = {"trial_face": pear(a, b)}
        d = pd.DataFrame({"a": a, "b": b, "p": p, "im": im})
        pi = d.groupby(["p", "im"])[["a", "b"]].mean()
        row["participant_image"] = pear(pi.a, pi.b)
        row["image_level"] = pear(*pi.groupby("im")[["a", "b"]].mean().T.values)
        d[["a2", "b2"]] = d[["a", "b"]] - d.groupby("p")[["a", "b"]].transform("mean")
        row["within_participant"] = pear(d.a2, d.b2)
        row["mean_ivt"], row["mean_parser"] = float(np.nanmean(a)), float(np.nanmean(b))
        out[fs] = row
    return out


def stage_rates(n_jobs):
    pids = list(participants())
    ck = SCRATCH / "c2_results.pkl"
    import pickle
    if ck.exists():
        results = pickle.loads(ck.read_bytes())
        log("loaded cached per-participant results")
    else:
        out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(process)(p) for p in pids)
        results = dict(out)
        tmp = Path(str(ck) + ".tmp")
        tmp.write_bytes(pickle.dumps(results))
        tmp.replace(ck)
    groups = participants()
    R = {"n_participants": len(pids), "n_trials_per_participant": int(next(iter(results.values()))["images"].shape[0])}
    # ---- event statistics per rate
    rows = []
    par_n = np.concatenate([results[p]["parser"]["n_all"] for p in pids])
    par_d = np.concatenate([results[p]["parser"]["durs"] for p in pids])
    par_cov = np.concatenate([results[p]["parser"]["covered"] for p in pids])
    par_face = np.concatenate([results[p]["parser"]["cnt"] for p in pids])
    pm_dur = [np.mean(results[p]["parser"]["durs"]) for p in pids]
    rows.append({"rate_hz": "EyeLink parser", "fix_per_trial_all": float(par_n.mean()), "fix_per_trial_on_face": float(par_face.sum(1).mean()),
                 "mean_fix_dur_ms_pooled": float(par_d.mean()), "median_fix_dur_ms": float(np.median(par_d)),
                 "mean_fix_dur_ms_participant_means": float(np.mean(pm_dur)), "share_window_in_fixation": float(par_cov.mean()),
                 "pct_samples_missing_in_window": float(100 * np.mean([results[p]["missing_1000"] for p in pids]))})
    for fs in RATES:
        n_all = np.concatenate([results[p]["rates"][fs]["n_all"] for p in pids])
        cnt = np.concatenate([results[p]["rates"][fs]["cnt"] for p in pids])
        durs = np.concatenate([results[p]["rates"][fs]["durs"] for p in pids])
        cov = np.concatenate([results[p]["rates"][fs]["covered"] for p in pids])
        pmd = [np.mean(results[p]["rates"][fs]["durs"]) for p in pids]
        rows.append({"rate_hz": fs, "fix_per_trial_all": float(n_all.mean()), "fix_per_trial_on_face": float(cnt.sum(1).mean()),
                     "mean_fix_dur_ms_pooled": float(durs.mean()), "median_fix_dur_ms": float(np.median(durs)),
                     "mean_fix_dur_ms_participant_means": float(np.mean(pmd)), "share_window_in_fixation": float(cov.mean()),
                     "pct_samples_missing_in_window": float(100 * np.mean([results[p]["rates"][fs]["missing"] for p in pids]))})
    ev = pd.DataFrame(rows)
    # ---- agreement with the parser
    ag = {"count": levels_r(pids, results, "cnt", "cnt"), "dwell_2": levels_r(pids, results, "dw2", "dw2"),
          "dwell_1": levels_r(pids, results, "dw1", "dw1")}
    # sample-based dwell vs parser dwell
    ag["sample_dwell_2_vs_parser_dwell_2"] = levels_r(pids, results, "sb2", "dw2")
    ag["sample_dwell_1_vs_parser_dwell_1"] = levels_r(pids, results, "sb1", "dw1")
    arows = []
    for metric, d in ag.items():
        for fs, r in d.items():
            arows.append({"metric": metric, "rate_hz": fs, **r})
    agdf = pd.DataFrame(arows)
    # per-group r for counts at trial x face
    gr_rows = []
    for g in sorted(set(groups.values())):
        gp = [p for p in pids if groups[p] == g]
        r = levels_r(gp, results, "cnt", "cnt")
        for fs in RATES:
            gr_rows.append({"group": g, "rate_hz": fs, "n_participants": len(gp), **{k: r[fs][k] for k in ("trial_face", "participant_image")}})
    grdf = pd.DataFrame(gr_rows)
    # participant bootstrap CI of r (trial x face and participant x image), counts only
    rng = np.random.default_rng(0)
    boots = {fs: {"trial_face": [], "participant_image": []} for fs in RATES}
    for b in range(500):
        pick = rng.choice(len(pids), len(pids))
        # resample participants: stack their cells (a participant drawn twice is counted twice; image-level pairs keyed by draw index)
        for fs in RATES:
            A = np.concatenate([results[pids[i]]["rates"][fs]["cnt"].ravel() for i in pick])
            B = np.concatenate([results[pids[i]]["parser"]["cnt"].ravel() for i in pick])
            boots[fs]["trial_face"].append(pear(A, B))
            Ap = np.concatenate([results[pids[i]]["rates"][fs]["cnt"].reshape(-1) for i in pick])
            # participant x image: group by (draw slot, image)
            slot = np.concatenate([np.full(results[pids[i]]["images"].size, s) for s, i in enumerate(pick)])
            im = np.concatenate([results[pids[i]]["images"].ravel() for i in pick])
            d = pd.DataFrame({"s": slot, "im": im, "a": Ap, "b": B}).groupby(["s", "im"])[["a", "b"]].mean()
            boots[fs]["participant_image"].append(pear(d.a, d.b))
    ci = {fs: {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in d.items()} for fs, d in boots.items()}
    # ---- saccade detectability
    sac_rows = []
    for fs in RATES:
        h = max(1, int(round(fs * 0.004)))
        span_ms = 2 * h * 1000.0 / fs
        row = {"rate_hz": fs, "half_width_samples": h, "velocity_span_ms": span_ms, "min_displacement_deg_noise_free": 30.0 * span_ms / 1000.0,
               "min_fixation_ms_enforced": max(dr.MIN_FIX_MS, 1000.0 / fs * int(np.ceil(dr.MIN_FIX_MS * fs / 1000.0 - 1e-9)))}
        tr = np.vstack([results[p]["rates"][fs]["trans"] for p in pids])
        row["n_parser_transitions"] = int(len(tr))
        for lo, hi in AMP_BINS:
            m = (tr[:, 0] >= lo) & (tr[:, 0] < hi)
            lab = f"{lo:g}-{hi:g}" if hi < 1e8 else f">={lo:g}"
            row[f"n_{lab}"] = int(m.sum())
            row[f"resolved_pct_{lab}"] = float(100 * np.mean(tr[m, 1] == 0)) if m.any() else float("nan")
            row[f"merged_pct_{lab}"] = float(100 * np.mean(tr[m, 1] == 1)) if m.any() else float("nan")
            row[f"lost_pct_{lab}"] = float(100 * np.mean(tr[m, 1] == 2)) if m.any() else float("nan")
        sac_rows.append(row)
    sac = pd.DataFrame(sac_rows)
    # ---- write
    ev.to_csv(OUT / "c2_event_stats_by_rate.csv", index=False)
    agdf.to_csv(OUT / "c2_parser_agreement.csv", index=False)
    grdf.to_csv(OUT / "c2_parser_agreement_by_group.csv", index=False)
    sac.to_csv(OUT / "c2_saccade_detectability.csv", index=False)
    R.update({"event_stats": ev.to_dict("records"), "parser_agreement": ag, "parser_agreement_bootstrap_ci95_counts": ci,
              "saccade_detectability": sac.to_dict("records")})
    jdump(OUT / "results_c2.json", R)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(ev.round(3).to_string(index=False))
    print(agdf.round(3).to_string(index=False))
    print(sac.round(2).T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="extract,rates")
    ap.add_argument("--n_jobs", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    st = a.stages.split(",")
    if "extract" in st:
        stage_extract(a.n_jobs)
    if "rates" in st:
        stage_rates(a.n_jobs)


if __name__ == "__main__":
    main()
