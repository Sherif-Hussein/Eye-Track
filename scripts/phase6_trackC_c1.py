"""Phase 6, Track C, analysis C1: sample-based dwell features (no event detection) under the pre-registered webcam noise,
with a noise decomposition (offset only / jitter only / sample loss only / all three), I-VT versus sample-based.
Pre-registration: results/phase6/trackC/PREREG.md (written before any result).

Stages (each skips what is on disk):
  run    - features + frozen-fold nested CV for every configuration (checkpoints in the scratch folder)
  check  - the code-path checks listed in the PREREG
  stats  - paired participant bootstrap vs P4 SVM 1000 Hz, non-inferiority p, CSVs, results_c1.json

Usage (project root):
  OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I scripts/phase6_trackC_c1.py --stages run --shard 0/4
"""
import argparse
import json
import math
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

from edd import delmas_fast as df_  # noqa: E402
from edd import delmas_raw as dr  # noqa: E402
from edd import delmas_raw_matched as M  # noqa: E402
from edd import webcam_noise as W  # noqa: E402
from edd.delmas import load_trials  # noqa: E402

OUT = ROOT / "results" / "phase6" / "trackC"
SCRATCH = Path(os.environ.get("TRACKC_SCRATCH", r"<local path>"))
CK = SCRATCH / "c1_ck"
WEB = ROOT / "results" / "phase3" / "webcam_noise"
CACHE = WEB / "cache_30hz"
TRACK = M.TRACK_DIR
BOOK64 = ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"
BOOK32 = ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx"
LEVELS = [1.0, 1.5, 2.0, 3.0, 4.0]
DRAWS = [1, 2, 3]
FS = 30
N_BOOT, MARGIN, BOOT_SEED = 2000, 0.02, 0
REF = "P4_svm_1000"
ANCH = {"off_mean": 1.45, "jit_x": 1.66 / math.sqrt(2), "jit_y": 1.42 / math.sqrt(2), "loss": 0.03}

IVT_FEATS = ["dwell_1", "dwell_2", "n_fix", "n_visit", "first_fix", "lat_first", "pupil_face", "pupil_trial",
             "dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4", "dwell_b5"]
SB_NAMES = ["dwell_1", "dwell_2", "n_visit", "first_fix", "lat_first", "pupil_face", "pupil_trial",
            "dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4", "dwell_b5", "n_seg"]
FEATS = {"ivt": [f + "_z" for f in IVT_FEATS],
         "sb": [f + "_z" for f in SB_NAMES if f != "n_seg"],
         "sbseg": [f + "_z" for f in SB_NAMES]}
CM_SETS = ("64_g2", "32_g3", "32_g4", "32_g5")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def lv(L):
    return "A145" if L == "A" else ("000" if L == 0 else f"{int(round(L * 100)):03d}")


def tag_of(variant, cond, L, d):
    return f"{variant}_{cond}_L{lv(L)}_d{d}"


# ----------------------------------------------------------------------------- units of work
def units():
    """(cond, L, draw) -> variants to run. 'all' with I-VT is reused from the webcam study (except one recomputed check)."""
    out = [("clean", 0.0, 0, ["ivt", "sb", "sbseg"])]
    for L in LEVELS:
        for d in DRAWS:
            out.append(("all", L, d, ["sb", "sbseg"] + (["ivt"] if (L, d) == (1.0, 1) else [])))
            for c in ("off", "jit", "loss"):
                out.append((c, L, d, ["ivt", "sb"]))
    for d in DRAWS:
        out.append(("anch", "A", d, ["ivt", "sb", "sbseg"]))
    return out


def all_tags():
    return [(tag_of(v, c, L, d), v, c, L, d) for c, L, d, vs in units() for v in vs]


# ----------------------------------------------------------------------------- data
def participants():
    book = load_trials(BOOK64)
    return {int(p): int(g) for p, g in book.groupby("participant").group.first().items()}


def load_cache():
    segs = {}
    for pid, g in participants().items():
        z = np.load(CACHE / f"p{pid:02d}.npz")
        segs[pid] = {"x30": z["x30"], "y30": z["y30"], "p30": z["p30"], "images": z["images"], "group": g}
    return segs


def noisy(x, y, p, cond, L, draw, pid):
    """Same random numbers as `edd.webcam_noise.apply_noise` (offset, jitter, loss draws in that order); components
    switched on per `cond`. 'all' at level L reproduces apply_noise exactly."""
    if cond == "clean":
        return x.copy(), y.copy(), p.copy()
    rng = np.random.default_rng([W.SEED, int(draw), int(pid)])
    z_off = rng.standard_normal(2)
    z_jit = rng.standard_normal((2,) + x.shape)
    u = rng.random(x.shape)
    if cond == "anch":
        off_sd, jx, jy, loss = ANCH["off_mean"] / W.RAYLEIGH_MEAN, ANCH["jit_x"], ANCH["jit_y"], ANCH["loss"]
        use_off = use_jit = use_loss = True
    else:
        off_sd = L / W.RAYLEIGH_MEAN
        jx = jy = W.JITTER_BASE + W.JITTER_SLOPE * L
        loss = W.DROPOUT
        use_off, use_jit, use_loss = cond in ("all", "off"), cond in ("all", "jit"), cond in ("all", "loss")
    off = off_sd * z_off if use_off else np.zeros(2)
    jx, jy = (jx, jy) if use_jit else (0.0, 0.0)
    ppd = dr.PIX_PER_DEG
    xn = x + (off[0] + jx * z_jit[0]) * ppd
    yn = y + (off[1] + jy * z_jit[1]) * ppd
    pn = p.copy()
    if use_loss:
        drop = u < loss
        xn[drop] = np.nan
        yn[drop] = np.nan
        pn[drop] = np.nan
    return xn, yn, pn


# ----------------------------------------------------------------------------- sample-based features
def _pupil_block(P, t):
    """Blink-padded pupil interpolation and baseline [-200, 0) ms, as in delmas_fast.trial_features_batch.
    Returns pdil (T, n), pok (T, n), late-window mask latep, pupil_trial (T,)."""
    T, n = P.shape
    bad = ~np.isfinite(P)
    w = max(1, int(round(50 * FS / 1000)))
    cb = np.zeros((T, n + 1), np.int32)
    np.cumsum(bad, axis=1, out=cb[:, 1:])
    ii = np.arange(n)
    dil = (cb[:, np.minimum(ii + w + 1, n)] - cb[:, np.maximum(ii - w, 0)]) > 0
    pc, pok = df_.fill_rows(np.where(dil, np.nan, P))
    b0, b1 = np.searchsorted(t, -200, "left"), np.searchsorted(t, 0, "left")
    nb = pok[:, b0:b1].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bl = np.where(nb > 0, np.where(pok[:, b0:b1], pc[:, b0:b1], 0.0).sum(1) / nb, np.nan)
    pdil = pc - bl[:, None]
    l0, l1 = np.searchsorted(t, 1000, "left"), np.searchsorted(t, dr.POST_MS, "left")
    latep = np.zeros((T, n), bool)
    latep[:, l0:l1] = pok[:, l0:l1]
    nl = latep.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pupil_trial = np.where(nl > 0, np.where(latep, pdil, 0.0).sum(1) / nl, np.nan)
    return pdil, latep, pupil_trial


def sb_features_batch(X, Y, P):
    """(T, 4, len(SB_NAMES)) sample-based features of one participant (T trials x n samples at 30 Hz)."""
    T, n = X.shape
    t = np.arange(n) * 1000.0 / FS - dr.PRE_MS
    valid = np.isfinite(X) & np.isfinite(Y)
    q = df_.aoi_batch(X, Y)
    F = np.full((T, 4, len(SB_NAMES)), np.nan)
    col = {c: i for i, c in enumerate(SB_NAMES)}
    wins = {"dwell_1": dr.PHASE1, "dwell_2": dr.PHASE2, **{f"dwell_b{j + 1}": b for j, b in enumerate(dr.BINS)}}
    for name, (a, b) in wins.items():
        m = (t >= a) & (t < b)
        nv = valid[:, m].sum(1)
        enough = nv >= 0.2 * m.sum()
        for k in range(4):
            inside = (q[:, m] == k).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                F[:, k, col[name]] = np.where(enough, inside / np.maximum(nv, 1), np.nan)
    win = (t >= 0) & (t < dr.POST_MS)
    tw, qw, vw = t[win], q[:, win], valid[:, win]
    nw = qw.shape[1]
    # latency / first look
    lat = np.full((T, 4), float(dr.POST_MS))
    for k in range(4):
        hit = qw == k
        anyh = hit.any(1)
        lat[anyh, k] = tw[hit.argmax(1)][anyh]
    F[:, :, col["lat_first"]] = lat
    first = lat.argmin(1)
    has = lat.min(1) < dr.POST_MS
    F[:, :, col["first_fix"]] = ((first[:, None] == np.arange(4)[None, :]) & has[:, None]).astype(float)
    # entries into the area (previous valid sample, gaps skipped)
    idx = np.where(vw, np.arange(nw)[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)
    prev_last = np.concatenate([np.full((T, 1), -1), last[:, :-1]], axis=1)
    prevq = np.where(prev_last >= 0, np.take_along_axis(qw, np.clip(prev_last, 0, nw - 1), 1), -1)
    for k in range(4):
        inK = qw == k
        F[:, k, col["n_visit"]] = (inK & (prevq != k)).sum(1)
        pair = inK[:, :-1] & inK[:, 1:]
        F[:, k, col["n_seg"]] = pair[:, 0] + (pair[:, 1:] & ~pair[:, :-1]).sum(1)
    # pupil
    pdil, latep, ptrial = _pupil_block(P, t)
    F[:, :, col["pupil_trial"]] = ptrial[:, None]
    for k in range(4):
        sel = (q == k) & latep
        ns = sel.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            F[:, k, col["pupil_face"]] = np.where(ns >= 0.1 * FS, np.where(sel, pdil, 0.0).sum(1) / np.maximum(ns, 1), np.nan)
    return F


def sb_reference_trial(x, y, p):
    """Unvectorised reference implementation of one trial's sample-based features (4, len(SB_NAMES)), independent code."""
    n = len(x)
    t = np.arange(n) * 1000.0 / FS - dr.PRE_MS
    F = np.full((4, len(SB_NAMES)), np.nan)
    col = {c: i for i, c in enumerate(SB_NAMES)}
    valid = np.isfinite(x) & np.isfinite(y)
    q = np.full(n, -1)
    for k, (x0, y0, x1, y1) in enumerate(dr.QUAD_RECTS):
        for i in range(n):
            if valid[i] and x0 <= x[i] <= x1 and y0 <= y[i] <= y1:
                q[i] = k
    wins = {"dwell_1": dr.PHASE1, "dwell_2": dr.PHASE2, **{f"dwell_b{j + 1}": b for j, b in enumerate(dr.BINS)}}
    for name, (a, b) in wins.items():
        idx = [i for i in range(n) if a <= t[i] < b]
        nv = sum(valid[i] for i in idx)
        if nv >= 0.2 * len(idx):
            for k in range(4):
                F[k, col[name]] = sum(q[i] == k for i in idx) / nv
    widx = [i for i in range(n) if 0 <= t[i] < dr.POST_MS]
    lat = [dr.POST_MS] * 4
    for k in range(4):
        for i in widx:
            if q[i] == k:
                lat[k] = t[i]
                break
    best = min(lat)
    for k in range(4):
        F[k, col["lat_first"]] = lat[k]
        F[k, col["first_fix"]] = float(best < dr.POST_MS and lat[k] == best)
        nvis, nseg, prev = 0, 0, None
        run = 0
        for i in widx:
            if valid[i]:
                if q[i] == k and prev != k:
                    nvis += 1
                prev = q[i]
            if q[i] == k:
                run += 1
                if run == 2:
                    nseg += 1
            else:
                run = 0
        F[k, col["n_visit"]], F[k, col["n_seg"]] = nvis, nseg
    pc, pok = dr._pupil_clean(p, FS)
    base = (t >= -200) & (t < 0)
    b = np.nanmean(pc[base & pok]) if (base & pok).any() else np.nan
    pdil = pc - b
    late = (t >= 1000) & (t < dr.POST_MS)
    pt = np.nanmean(pdil[late & pok]) if (late & pok).any() else np.nan
    for k in range(4):
        sel = (q == k) & late & pok
        F[k, col["pupil_trial"]] = pt
        F[k, col["pupil_face"]] = np.nanmean(pdil[sel]) if sel.sum() >= 0.1 * FS else np.nan
    return F


# ----------------------------------------------------------------------------- tables
def books():
    return {"64": load_trials(BOOK64), "32": load_trials(BOOK32)}


def make_table(arrays, names, bk):
    """arrays: {pid: (F (T,4,nf), images (T,4), group)}; same aggregation as `webcam_noise.build_table_30`, generalised
    to a feature-name list."""
    rows = []
    for pid, (F, images, g) in arrays.items():
        T = F.shape[0]
        base = np.column_stack([np.full(T * 4, pid), np.full(T * 4, g), np.full(T * 4, FS), np.repeat(np.arange(T), 4),
                                np.tile(np.arange(4), T), images.reshape(-1)])
        rows.append(np.column_stack([base, F.reshape(T * 4, -1)]))
    face = pd.DataFrame(np.vstack(rows), columns=["participant", "group", "rate", "trial", "quad", "image"] + names)
    face[["participant", "group", "rate", "trial", "quad", "image"]] = face[["participant", "group", "rate", "trial", "quad", "image"]].astype(int)
    out = []
    for s, book in bk.items():
        f = face if s == "64" else face[face.trial >= 32]
        f = f[f.participant.isin(book.participant.unique())]
        agg = f.groupby(["rate", "participant", "image"])[names].mean().reset_index()
        agg = agg.merge(book[["participant", "image", "group", "face"]], on=["participant", "image"], how="inner")
        agg["set"] = s
        out.append(agg)
    d = pd.concat(out, ignore_index=True)
    z = d.groupby(["rate", "set", "participant"])[names].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    for c in names:
        d[c + "_z"] = z[c]
    return d


def build_tables(segs, cond, L, draw, variants, bk):
    ivt_arr, sb_arr = {}, {}
    for pid, seg in segs.items():
        x, y, p = noisy(seg["x30"], seg["y30"], seg["p30"], cond, L, draw, pid)
        if "ivt" in variants:
            ivt_arr[pid] = (df_.trial_features_batch(x, y, p, FS, with_msacc=False), seg["images"], seg["group"])
        if "sb" in variants or "sbseg" in variants:
            sb_arr[pid] = (sb_features_batch(x, y, p), seg["images"], seg["group"])
    out = {}
    if ivt_arr:
        out["ivt"] = make_table(ivt_arr, dr.TRIAL_FEATURES, bk)
    if sb_arr:
        tab = make_table(sb_arr, SB_NAMES, bk)
        out["sb"] = out["sbseg"] = tab
    return out


# ----------------------------------------------------------------------------- stage run
def stage_run(shard):
    assert (OUT / "PREREG.md").exists()
    CK.mkdir(parents=True, exist_ok=True)
    track = M.load_track()
    track.CKPT = CK
    frozen = json.loads((TRACK / "folds.json").read_text())["repeats"]
    segs = load_cache()
    bk = books()
    us = units()
    i0, n = shard
    for ui, (cond, L, d, vs) in enumerate(us):
        if ui % n != i0:
            continue
        need = [v for v in vs if not (CK / f"{tag_of(v, cond, L, d)}.json").exists()]
        if not need:
            continue
        t0 = time.time()
        tabs = build_tables(segs, cond, L, d, need, bk)
        log(f"unit {cond} L={L} d={d}: features built {time.time() - t0:.0f}s for {need}")
        for v in need:
            tag = tag_of(v, cond, L, d)
            t1 = time.time()
            res = track.run_grid(tabs[v], frozen, FEATS[v], "lr", FS, tag, {}, {})
            log(f"  {tag}: mean AUC {res['mean_over_sets']['image']['auc']:.4f} ({time.time() - t1:.0f}s)")


# ----------------------------------------------------------------------------- stage check
def stage_check():
    res = {}
    segs = load_cache()
    # (1) sample-based vectorised vs unvectorised reference, full noise L=1.5 draw 1, two participants, 40 trials each
    mx = 0.0
    for pid in list(segs)[:2]:
        seg = segs[pid]
        x, y, p = noisy(seg["x30"], seg["y30"], seg["p30"], "all", 1.5, 1, pid)
        Fv = sb_features_batch(x[:40], y[:40], p[:40])
        for k in range(40):
            Fr = sb_reference_trial(x[k], y[k], p[k])
            both = np.isfinite(Fv[k]) & np.isfinite(Fr)
            assert (np.isfinite(Fv[k]) == np.isfinite(Fr)).all(), (pid, k)
            mx = max(mx, float(np.abs(Fv[k][both] - Fr[both]).max()))
    res["sb_vectorised_vs_reference_max_abs_diff"] = mx
    # (2) noisy('all') equals webcam_noise.apply_noise
    seg = segs[2]
    xa, ya, pa, _ = W.apply_noise(seg["x30"], seg["y30"], seg["p30"], 1.5, 2, 2)
    xb, yb, pb = noisy(seg["x30"], seg["y30"], seg["p30"], "all", 1.5, 2, 2)
    res["noise_all_equals_apply_noise"] = bool(np.array_equal(xa, xb, equal_nan=True) and np.array_equal(ya, yb, equal_nan=True)
                                               and np.array_equal(pa, pb, equal_nan=True))
    # (3) I-VT OOF scores: level 0 and L=1 draw 1 recomputed here vs the stored webcam-study OOF
    for tag, ref in (("ivt_clean_L000_d0", "ALLz_fixed_lr_30_L000_d0"), ("ivt_all_L100_d1", "ALLz_fixed_lr_30_L100_d1")):
        f = CK / f"{tag}_oof.csv.gz"
        if not f.exists():
            res[tag] = "not run yet"
            continue
        a = pd.read_csv(f).sort_values(["set", "repeat", "participant", "image"]).reset_index(drop=True)
        b = pd.read_csv(WEB / "checkpoints" / f"{ref}_oof.csv.gz").sort_values(["set", "repeat", "participant", "image"]).reset_index(drop=True)
        same = len(a) == len(b) and (a[["set", "repeat", "participant", "image", "y"]].to_numpy() == b[["set", "repeat", "participant", "image", "y"]].to_numpy()).all()
        res[tag] = {"same_rows": bool(same), "max_abs_score_diff": float(np.abs(a.score.to_numpy() - b.score.to_numpy()).max()) if same else None}
    jdump(OUT / "c1_checks.json", res)
    log("checks:", res)
    return res


# ----------------------------------------------------------------------------- stage stats
def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def load_oof(path):
    d = pd.read_csv(path)
    d = d.groupby(["set", "participant", "image"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
    return d.sort_values(["set", "participant", "image"]).reset_index(drop=True)


def oof_path(tag):
    if tag == REF:
        return TRACK / "checkpoints" / f"{REF}_oof.csv.gz"
    if tag.startswith("ALLz_fixed"):
        return WEB / "checkpoints" / f"{tag}_oof.csv.gz"
    return CK / f"{tag}_oof.csv.gz"


def stage_stats():
    # conditions: (variant, cond, L) -> tags of its draws
    fam = {}
    for tag, v, c, L, d in all_tags():
        fam.setdefault((v, c, L), []).append(tag)
    for L in LEVELS:                       # reused full-noise I-VT results of the webcam study
        fam[("ivt", "all", L)] = [f"ALLz_fixed_lr_30_L{lv(L)}_d{d}" for d in DRAWS]
    fam[("ivt", "clean", 0.0)] = ["ALLz_fixed_lr_30_L000_d0"]       # stored level 0 (bit-identical to the recomputed one)
    names = [REF] + sorted({t for ts in fam.values() for t in ts})
    missing = [n for n in names if not oof_path(n).exists()]
    assert not missing, f"missing OOF: {missing[:5]} ({len(missing)})"
    base = load_oof(oof_path(REF))
    keys = base[["set", "participant", "image"]].to_numpy()
    S = np.empty((len(base), len(names)))
    S[:, 0] = base.score.to_numpy()
    for j, n in enumerate(names[1:], 1):
        o = load_oof(oof_path(n))
        assert len(o) == len(base) and (o[["set", "participant", "image"]].to_numpy() == keys).all() and (o.y.to_numpy() == base.y.to_numpy()).all(), n
        S[:, j] = o.score.to_numpy()
    col = {n: j for j, n in enumerate(names)}
    sets = {}
    for s in sorted(base.set.unique()):
        m = (base.set == s).to_numpy()
        sets[s] = W.SetData(base.y.to_numpy()[m], base.participant.to_numpy()[m], S[m])
    set_names = list(sets)
    point = np.array([W.auc_matrix(sets[s].y, sets[s].S) for s in set_names])           # (sets, cols)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.full((N_BOOT, len(set_names), len(names)), np.nan)
    for b in range(N_BOOT):
        for si, s in enumerate(set_names):
            sd = sets[s]
            pick = rng.integers(0, sd.n_part, sd.n_part)
            idx = sd.rows_of(pick)
            y = sd.y[idx]
            if len(np.unique(y)) < 2:
                continue
            boot[b, si] = W.auc_matrix(y, sd.S[idx])
    boot_auc = np.nanmean(boot, axis=1)
    log("bootstrap done")
    cm_idx = [set_names.index(s) for s in CM_SETS]
    pt_ref = float(point[:, 0].mean())
    clean_cols = {"ivt": [col["ALLz_fixed_lr_30_L000_d0"]], "sb": [col[tag_of("sb", "clean", 0.0, 0)]],
                  "sbseg": [col[tag_of("sbseg", "clean", 0.0, 0)]]}
    rows = {}
    for (v, c, L), tags in fam.items():
        cs = [col[t] for t in tags]
        cand_b = boot_auc[:, cs].mean(1)
        delta_b = cand_b - boot_auc[:, 0]
        pt_c = float(np.mean([point[:, k].mean() for k in cs]))
        own = clean_cols[v]
        own_b = boot_auc[:, own].mean(1)
        r = {"variant": v, "cond": c, "level": L, "tags": tags, "n_draws": len(tags), "auc": pt_c,
             "auc_ci95": [float(x) for x in np.percentile(cand_b, [2.5, 97.5])],
             "auc_per_draw": [float(point[:, k].mean()) for k in cs],
             "auc_draw_sd": float(np.std([point[:, k].mean() for k in cs])) if len(cs) > 1 else 0.0,
             "auc_cm_sets": float(np.mean([point[cm_idx, k].mean() for k in cs])),
             "auc_ref": pt_ref, "delta": pt_c - pt_ref,
             "delta_ci95": [float(x) for x in np.percentile(delta_b, [2.5, 97.5])],
             "delta_lower_one_sided_95": float(np.percentile(delta_b, 5)),
             "p_inferior_raw": float((1 + np.sum(delta_b <= -MARGIN)) / (1 + N_BOOT)),
             "delta_vs_own_clean": float(pt_c - np.mean([point[:, k].mean() for k in own])),
             "delta_vs_own_clean_ci95": [float(x) for x in np.percentile(cand_b - own_b, [2.5, 97.5])],
             "_cand_b": cand_b}
        rows[(v, c, L)] = r
    # SB minus I-VT contrasts at the same condition and level
    for (v, c, L), r in rows.items():
        if v == "ivt":
            continue
        partner = rows.get(("ivt", c, L))
        if partner is None:
            continue
        diff_b = r["_cand_b"] - partner["_cand_b"]
        r["delta_vs_ivt_same_cond"] = float(r["auc"] - partner["auc"])
        r["delta_vs_ivt_same_cond_ci95"] = [float(x) for x in np.percentile(diff_b, [2.5, 97.5])]
        r["ivt_auc_same_cond"] = partner["auc"]
    # Holm: primary family = sb / all + clean (6 tests); other families = one (variant, cond) across levels
    prim_keys = [("sb", "clean", 0.0)] + [("sb", "all", L) for L in LEVELS]
    for k, a in zip(prim_keys, holm([rows[k]["p_inferior_raw"] for k in prim_keys])):
        rows[k]["p_holm_primary_family"] = float(a)
        rows[k]["noninferior_primary_family"] = bool(a < 0.05)
    famgroups = {}
    for k in rows:
        v, c, L = k
        key = (v, "all" if c == "clean" else c)
        famgroups.setdefault(key, []).append(k)
    for key, ks in famgroups.items():
        for k, a in zip(ks, holm([rows[k]["p_inferior_raw"] for k in ks])):
            rows[k]["p_holm_own_family"] = float(a)
            rows[k]["noninferior_own_family_exploratory"] = bool(a < 0.05)
    # outputs
    out_rows = []
    for k, r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], -1 if kv[0][2] in (0.0,) else (999 if kv[0][2] == "A" else kv[0][2]))):
        r = {a: b for a, b in r.items() if a != "_cand_b"}
        out_rows.append(r)
    flat = []
    for r in out_rows:
        flat.append({"variant": r["variant"], "cond": r["cond"], "level_deg": r["level"], "auc": r["auc"], "auc_lo": r["auc_ci95"][0],
                     "auc_hi": r["auc_ci95"][1], "auc_draw_sd": r["auc_draw_sd"], "delta_vs_P4svm1000": r["delta"],
                     "delta_lo": r["delta_ci95"][0], "delta_hi": r["delta_ci95"][1], "p_noninferior_raw": r["p_inferior_raw"],
                     "p_holm_primary_family": r.get("p_holm_primary_family"), "p_holm_own_family": r.get("p_holm_own_family"),
                     "delta_vs_own_clean": r["delta_vs_own_clean"], "delta_vs_own_clean_lo": r["delta_vs_own_clean_ci95"][0],
                     "delta_vs_own_clean_hi": r["delta_vs_own_clean_ci95"][1],
                     "delta_vs_ivt_same_cond": r.get("delta_vs_ivt_same_cond"),
                     "delta_vs_ivt_lo": r.get("delta_vs_ivt_same_cond_ci95", [None, None])[0],
                     "delta_vs_ivt_hi": r.get("delta_vs_ivt_same_cond_ci95", [None, None])[1],
                     "auc_cm_sets": r["auc_cm_sets"]})
    pd.DataFrame(flat).to_csv(OUT / "c1_all_conditions.csv", index=False)
    R = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": REF, "reference_auc": pt_ref, "margin": MARGIN, "n_boot": N_BOOT,
         "bootstrap_seed": BOOT_SEED, "anchored_noise": ANCH, "checks": json.loads((OUT / "c1_checks.json").read_text()),
         "conditions": out_rows}
    jdump(OUT / "results_c1.json", R)
    log("stats written")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="run,check,stats")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CK.mkdir(parents=True, exist_ok=True)
    st = a.stages.split(",")
    i0, n = (int(v) for v in a.shard.split("/"))
    if "run" in st:
        stage_run((i0, n))
    if "check" in st:
        stage_check()
    if "stats" in st:
        stage_stats()


if __name__ == "__main__":
    main()
