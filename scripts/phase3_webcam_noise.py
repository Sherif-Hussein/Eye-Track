"""Phase 3, experiment `webcam_noise`: does "ALL_z LR at 30 Hz is non-inferior (margin 0.02 AUC) to P4 SVM at 1000 Hz"
survive realistic webcam spatial error? Pre-registration: results/phase3/webcam_noise/PREREG.md (written before any noisy run).

Stages (each checkpoints to results/phase3/webcam_noise/ and is skipped when its output exists):
  extract  - decode each participant's EDF once, decimate to 30 Hz (the track's `degrade`), cache x/y/pupil in cache_30hz/
  check    - clean 30 Hz features rebuilt from the cache vs the stored track features (must be identical)
  run      - for every noise level x draw x variant: noisy features -> frozen-fold nested CV (track's run_grid) -> OOF + metrics
  stats    - paired participant-cluster bootstrap non-inferiority vs P4 SVM 1000 Hz (clean), Holm, curve, card
Usage (project root):  OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I scripts/phase3_webcam_noise.py --stages extract,check,run,stats
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from edd import delmas_raw as dr  # noqa: E402
from edd import delmas_raw_matched as M  # noqa: E402
from edd import webcam_noise as W  # noqa: E402
from edd.delmas import load_trials  # noqa: E402

ROOT = M.ROOT
TRACK = M.TRACK_DIR
OUT = ROOT / "results" / "phase3" / "webcam_noise"
CK = OUT / "checkpoints"
CACHE = OUT / "cache_30hz"
RAW = ROOT / "data" / "delmas" / "raw" / "data_raw"
BOOK64 = ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"
BOOK32 = ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx"
SCRATCH = Path(os.environ.get("TEMP", ".")) / "webcam_noise_tmp"
LEVELS = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0]
DRAWS = [1, 2, 3]
N_BOOT, MARGIN, BOOT_SEED = 2000, 0.02, 0
REF = "P4_svm_1000"
ALLZ = [f for f in ("dwell_1", "dwell_2", "n_fix", "n_visit", "first_fix", "lat_first", "pupil_face", "pupil_trial",
                    "dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4", "dwell_b5")]
FEATS = {"ALLz": [f + "_z" for f in ALLZ],
         "P4": ["dwell_1", "dwell_2", "n_fix", "n_visit"],
         "ALLznopupil": [f + "_z" for f in ALLZ if not f.startswith("pupil")]}
# (model key, I-VT mode)
VARIANTS = [("ALLz", "fixed"), ("P4", "fixed"), ("ALLz", "adaptive")]
METRICS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "auc", "eer", "hter"]
CM_SETS = ("64_g2", "32_g3", "32_g4", "32_g5")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def lv(L):
    return f"{int(round(L * 100)):03d}"


def tag_of(model, mode, L, d):
    return f"{model}_{mode}_lr_30_L{lv(L)}_d{d}"


def configs():
    """All (tag, model, mode, L, draw) of the pre-registered design. Level 0 is noise-free: one run, draw 0, no adaptive."""
    out = []
    for L in LEVELS:
        for d in ([0] if L == 0 else DRAWS):
            for model, mode in VARIANTS:
                if L == 0 and mode == "adaptive":
                    continue
                out.append((tag_of(model, mode, L, d), model, mode, L, d))
    return out


# ----------------------------------------------------------------------------- stage extract
def session_files(pid):
    d = RAW / f"Sj_{pid:02d}"
    return [(e, d / f"FI{pid:02d}c2_{e.stem.split('_')[-1]}.mat") for e in sorted(d.glob(f"I{pid:02d}c2_*.edf"))]


def extract_one(pid, group):
    path = CACHE / f"p{pid:02d}.npz"
    if path.exists():
        return pid, "cached"
    segs = [dr.load_session(e, m) for e, m in session_files(pid)]
    seg = {k: np.concatenate([s[k] for s in segs]) for k in segs[0]}
    n = len(seg["images"])
    x30 = np.empty((n, 165))
    y30, p30 = np.empty((n, 165)), np.empty((n, 165))
    for k in range(n):
        x30[k], y30[k], p30[k] = dr.degrade(seg["x"][k], seg["y"][k], seg["p"][k], W.FS)
    geo = W.geometry_from_design(session_files(pid)[0][1])
    tmp = CACHE / f"p{pid:02d}.tmp.npz"
    np.savez_compressed(tmp, x30=x30, y30=y30, p30=p30, images=seg["images"], group=group,
                        geo=np.array([geo["viewing_distance_cm"], geo["screen_width_mm"], geo["xres"],
                                      geo["pix_per_deg_design_file"], geo["pix_per_deg_recomputed"]]))
    tmp.replace(path)
    return pid, "extracted"


def participants():
    book = load_trials(BOOK64)
    return {int(p): int(g) for p, g in book.groupby("participant").group.first().items()}


def stage_extract(n_jobs):
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = participants()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(extract_one)(p, g) for p, g in parts.items())
    log("extract:", {s: sum(1 for r in res if r[1] == s) for s in ("cached", "extracted")})


def load_cache():
    segs = {}
    for pid, g in participants().items():
        z = np.load(CACHE / f"p{pid:02d}.npz")
        segs[pid] = {"x30": z["x30"], "y30": z["y30"], "p30": z["p30"], "images": z["images"], "group": g, "geo": z["geo"]}
    return segs


# ----------------------------------------------------------------------------- tables
def books():
    return {"64": load_trials(BOOK64), "32": load_trials(BOOK32)}


def noisy_table(segs, L, d, mode):
    thr = None if mode == "fixed" else W.adaptive_threshold(L)
    frames, infos = [], []
    for pid, seg in segs.items():
        f, info = W.participant_face_rows(seg, seg["group"], pid, L, d, thr)
        frames.append(f)
        infos.append(info)
    face = pd.concat(frames, ignore_index=True)
    df = W.build_table_30(face, books())
    diag = {"ivt_threshold_deg_s": float(dr.VEL_THR if thr is None else thr),
            "realised_offset_deg_mean": float(np.mean([i["offset_deg"] for i in infos])),
            "realised_offset_deg_sd": float(np.std([i["offset_deg"] for i in infos])),
            "realised_offset_deg_max": float(np.max([i["offset_deg"] for i in infos])),
            "missing_fraction_after_noise_mean": float(np.mean([i["missing_fraction_after_noise"] for i in infos])),
            "mean_features_all_rows": {c: float(df[c].mean()) for c in ("dwell_1", "dwell_2", "n_fix", "n_visit", "first_fix", "lat_first")},
            "n_rows_64": int((df.set == "64").sum())}
    return df, face, diag


# ----------------------------------------------------------------------------- stage check
def stage_check(segs):
    f = CK / "clean_check.json"
    if f.exists():
        log("[skip, cached] clean_check")
        return
    df, face, diag = noisy_table(segs, 0.0, 0, "fixed")
    stored = pd.read_csv(ROOT / "data" / "derived" / "delmas_raw_face_trials.csv.gz")
    stored = stored[stored.rate == 30].sort_values(["participant", "trial", "quad"]).reset_index(drop=True)
    mine = face.sort_values(["participant", "trial", "quad"]).reset_index(drop=True)
    cols = dr.TRIAL_FEATURES
    a, b = stored[cols].to_numpy(float), mine[cols].to_numpy(float)
    same_keys = bool((stored[["participant", "trial", "quad", "image"]].to_numpy() == mine[["participant", "trial", "quad", "image"]].to_numpy()).all())
    nan_same = bool((np.isnan(a) == np.isnan(b)).all())
    mx = float(np.nanmax(np.abs(a - b)))
    # whole aggregated + z-scored table vs the track's own build_tables
    track = M.load_track()
    ref = track.build_tables(ROOT / "data" / "derived", BOOK64, BOOK32)
    ref = ref[ref.rate == 30].sort_values(["set", "participant", "image"]).reset_index(drop=True)
    mi = df.sort_values(["set", "participant", "image"]).reset_index(drop=True)
    zc = [c for c in mi.columns if c.endswith("_z") or c in dr.TRIAL_FEATURES]
    ta, tb = ref[zc].to_numpy(float), mi[zc].to_numpy(float)
    res = {"rows_compared": int(len(a)), "same_keys": same_keys, "nan_pattern_identical": nan_same, "max_abs_diff_trial_features": mx,
           "table_rows_ref": int(len(ref)), "table_rows_mine": int(len(mi)),
           "table_same_keys": bool((ref[["set", "participant", "image", "face", "group"]].to_numpy() == mi[["set", "participant", "image", "face", "group"]].to_numpy()).all()),
           "table_max_abs_diff_features_and_z": float(np.nanmax(np.abs(ta - tb))),
           "geometry": {"viewing_distance_cm": sorted({float(s['geo'][0]) for s in segs.values()}),
                        "screen_width_mm": sorted({float(s['geo'][1]) for s in segs.values()}),
                        "xres": sorted({float(s['geo'][2]) for s in segs.values()}),
                        "pix_per_deg_design_file": sorted({round(float(s['geo'][3]), 9) for s in segs.values()}),
                        "pix_per_deg_recomputed_tan1deg": sorted({round(float(s['geo'][4]), 9) for s in segs.values()}),
                        "pix_per_deg_used_by_pipeline": dr.PIX_PER_DEG}}
    log("clean check:", res)
    assert same_keys and nan_same and mx < 1e-9, "cached 30 Hz features differ from the stored ones"
    jdump(f, res)


# ----------------------------------------------------------------------------- stage run
def metric_summary(res):
    """Mean over the six sets of each image-level metric (mean over repeats), plus CM-set AUC and CIT AUC / HTER."""
    sets = [k for k in res if k not in ("mean_over_sets", "mean_over_cm_sets")]
    s = {m: float(np.nanmean([res[k]["image"][m]["mean"] for k in sets])) for m in METRICS}
    s["auc_cm_sets_track_def"] = float(res["mean_over_cm_sets"]["image"]["auc"])
    s["cit_auc"] = float(res["mean_over_sets"]["participant_cit"]["auc"])
    s["cit_hter"] = float(res["mean_over_sets"]["participant_cit"]["hter"])
    s["per_set_auc"] = {k: float(res[k]["image"]["auc"]["mean"]) for k in sets}
    return s


def stage_run(segs):
    track = M.load_track()
    track.CKPT = CK
    frozen = json.loads((TRACK / "folds.json").read_text())["repeats"]
    todo = configs()
    for L in LEVELS:
        t_level = time.time()
        for d in ([0] if L == 0 else DRAWS):
            for mode in ("fixed", "adaptive"):
                mine = [c for c in todo if c[3] == L and c[4] == d and c[2] == mode]
                missing = [c for c in mine if not (CK / f"{c[0]}.json").exists()]
                if not missing:
                    continue
                t0 = time.time()
                df, _, diag = noisy_table(segs, L, d, mode)
                jdump(CK / f"diag_{mode}_L{lv(L)}_d{d}.json", diag)
                log(f"L={L} draw={d} {mode}: features built in {time.time() - t0:.0f}s; thr {diag['ivt_threshold_deg_s']:.1f}, "
                    f"offset mean {diag['realised_offset_deg_mean']:.2f}, mean n_fix {diag['mean_features_all_rows']['n_fix']:.2f}")
                for tag, model, _, _, _ in missing:
                    t1 = time.time()
                    res = track.run_grid(df, frozen, FEATS[model], "lr", W.FS, tag, {}, {})
                    log(f"  {tag}: mean AUC {res['mean_over_sets']['image']['auc']:.4f} EER {res['mean_over_sets']['image']['eer']:.3f} "
                        f"({time.time() - t1:.0f}s)")
                write_resume()
        jdump(CK / f"level_{lv(L)}_done.json", {"level": L, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                 "seconds_this_session": time.time() - t_level})
        log(f"[level {L} done]")
        write_resume()


def write_resume(stage_note=None):
    done = []
    for tag, model, mode, L, d in configs():
        f = CK / f"{tag}.json"
        if f.exists():
            r = json.loads(f.read_text())["res"]
            done.append((tag, r["mean_over_sets"]["image"]["auc"], r["mean_over_cm_sets"]["image"]["auc"]))
    n_all = len(configs())
    lines = ["# webcam_noise: RESUME (auto-written by scripts/phase3_webcam_noise.py)", "",
             f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}. Configurations finished: {len(done)} of {n_all}.",
             "Pre-registration: `PREREG.md` (fixed before the first noisy run; do not edit the design, only append amendments).", ""]
    if stage_note:
        lines += [stage_note, ""]
    stages = {"extract": len(list(CACHE.glob("p*.npz"))) - len(list(CACHE.glob("p*.tmp.npz"))) if CACHE.exists() else 0}
    lines += [f"- 30 Hz cache: {stages['extract']} of 75 participants in `cache_30hz/`",
              f"- clean check: {'done' if (CK / 'clean_check.json').exists() else 'not yet'}",
              f"- stats / card: {'done' if (OUT / 'results.json').exists() else 'not yet'}", "",
              "## Finished configurations (mean image AUC over 6 sets, track definition; countermeasure-set AUC)", "",
              "| tag | AUC | CM-sets AUC |", "|---|---|---|"]
    lines += [f"| {t} | {a:.4f} | {c:.4f} |" for t, a, c in done]
    left = [c[0] for c in configs() if not (CK / f"{c[0]}.json").exists()]
    lines += ["", f"## Left ({len(left)})", ", ".join(left[:12]) + (" ..." if len(left) > 12 else ""), "",
              "## Resume (project root; every stage skips what is already on disk; run in the background, log to this folder)", "",
              "    OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I scripts/phase3_webcam_noise.py --stages extract,check,run,stats > results/phase3/webcam_noise/run_log.txt 2>&1", "",
              "To recompute one configuration delete `checkpoints/<tag>.json` (and its `_oof.csv.gz`). Never use dots in tags (Path.with_suffix).", ""]
    (OUT / "RESUME.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- stage stats
def load_oof(path):
    d = pd.read_csv(path)
    d = d.groupby(["set", "participant", "image"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
    return d.sort_values(["set", "participant", "image"]).reset_index(drop=True)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def stage_stats():
    cfgs = configs()
    names = [REF] + [c[0] for c in cfgs]
    paths = {REF: TRACK / "checkpoints" / f"{REF}_oof.csv.gz", **{c[0]: CK / f"{c[0]}_oof.csv.gz" for c in cfgs}}
    missing = [n for n in names if not paths[n].exists()]
    assert not missing, f"missing OOF files: {missing[:5]}"
    base = load_oof(paths[REF])
    keys = base[["set", "participant", "image"]].to_numpy()
    S = np.empty((len(base), len(names)))
    S[:, 0] = base.score.to_numpy()
    for j, n in enumerate(names[1:], 1):
        o = load_oof(paths[n])
        assert len(o) == len(base) and (o[["set", "participant", "image"]].to_numpy() == keys).all() and (o.y.to_numpy() == base.y.to_numpy()).all(), n
        S[:, j] = o.score.to_numpy()
    col = {n: j for j, n in enumerate(names)}
    sets = {}
    for s in sorted(base.set.unique()):
        m = (base.set == s).to_numpy()
        sets[s] = W.SetData(base.y.to_numpy()[m], base.participant.to_numpy()[m], S[m])
    set_names = list(sets)
    # point estimates (full sample), per set and column
    point = np.array([W.auc_matrix(sets[s].y, sets[s].S) for s in set_names])        # (sets, cols)
    # common bootstrap resamples (participants within each set), one stream, seed 0
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
    boot_auc = np.nanmean(boot, axis=1)                                              # (B, cols)
    log("bootstrap done")
    # model families: (model key, mode) -> {level: [tags]}
    fam = {}
    for tag, model, mode, L, d in cfgs:
        fam.setdefault((model, mode), {}).setdefault(L, []).append(tag)
    rows = []
    for (model, mode), lvls in fam.items():
        for L, tags in sorted(lvls.items()):
            cs = [col[t] for t in tags]
            cand_b = boot_auc[:, cs].mean(1)
            ref_b = boot_auc[:, 0]
            delta_b = cand_b - ref_b
            pt_c = point[:, cs].mean(0).mean() if False else float(np.mean([point[:, c].mean() for c in cs]))
            pt_r = float(point[:, 0].mean())
            cm_idx = [set_names.index(s) for s in CM_SETS]
            pt_cm = float(np.mean([point[cm_idx, c].mean() for c in cs]))
            clean_tag = tag_of(model, "fixed", 0.0, 0)
            r = {"model": model, "ivt_mode": mode, "level_deg": L, "tags": tags, "n_draws": len(tags),
                 "auc": pt_c, "auc_ci95": [float(x) for x in np.percentile(cand_b, [2.5, 97.5])],
                 "auc_per_draw": [float(point[:, c].mean()) for c in cs],
                 "auc_draw_sd": float(np.std([point[:, c].mean() for c in cs])) if len(cs) > 1 else 0.0,
                 "auc_cm_sets_noninf_def": pt_cm,
                 "auc_ref": pt_r, "delta": pt_c - pt_r,
                 "delta_ci95": [float(x) for x in np.percentile(delta_b, [2.5, 97.5])],
                 "delta_lower_one_sided_95": float(np.percentile(delta_b, 5)),
                 "delta_per_draw": [float(point[:, c].mean() - pt_r) for c in cs],
                 "p_inferior_raw": float((1 + np.sum(delta_b <= -MARGIN)) / (1 + N_BOOT))}
            ci_self = boot_auc[:, cs].mean(1) - boot_auc[:, col[clean_tag]]
            r["delta_vs_own_clean"] = float(pt_c - point[:, col[clean_tag]].mean())
            r["delta_vs_own_clean_ci95"] = [float(x) for x in np.percentile(ci_self, [2.5, 97.5])]
            rows.append(r)
    # Holm: pre-registered family F_all = ALLz fixed, P4 fixed, ALLz adaptive (all levels tested); ALLznopupil exploratory
    fam_all = [i for i, r in enumerate(rows) if (r["model"], r["ivt_mode"]) in (("ALLz", "fixed"), ("P4", "fixed"), ("ALLz", "adaptive"))]
    assert len(fam_all) == 17
    adj = holm([rows[i]["p_inferior_raw"] for i in fam_all])
    for i, a in zip(fam_all, adj):
        rows[i]["p_holm_family_all"] = float(a)
    prim = [i for i, r in enumerate(rows) if r["model"] == "ALLz" and r["ivt_mode"] == "fixed"]
    for i, a in zip(prim, holm([rows[i]["p_inferior_raw"] for i in prim])):
        rows[i]["p_holm_primary_model_only"] = float(a)
    nop = [i for i, r in enumerate(rows) if r["model"] == "ALLznopupil"]
    for i, a in zip(nop, holm([rows[i]["p_inferior_raw"] for i in nop])):
        rows[i]["p_holm_nopupil_own_family"] = float(a)
    for r in rows:
        p = r.get("p_holm_family_all")
        r["noninferior_holm_family_all"] = None if p is None else bool(p < 0.05)
    # primary outcome
    def largest(model, mode, key):
        rs = sorted([r for r in rows if r["model"] == model and r["ivt_mode"] == mode], key=lambda r: r["level_deg"])
        ok = [r["level_deg"] for r in rs if r.get(key) is not None and r[key] < 0.05]
        contiguous = None
        for r in rs:
            if r.get(key) is not None and r[key] < 0.05:
                contiguous = r["level_deg"]
            else:
                break
        return {"largest_noninferior_any_level": max(ok) if ok else None, "largest_contiguous_from_0": contiguous,
                "noninferior_levels": ok}
    primary = {f"{m}|{mo}|{k}": largest(m, mo, k) for m, mo in (("ALLz", "fixed"), ("P4", "fixed"), ("ALLz", "adaptive"), ("ALLznopupil", "fixed"))
               for k in ("p_holm_family_all", "p_holm_primary_model_only", "p_holm_nopupil_own_family")
               if any(r["model"] == m and r["ivt_mode"] == mo and r.get(k) is not None for r in rows)}
    # metrics tables (mean over draws), reference included
    def jres(tag, base_dir):
        return json.loads((base_dir / f"{tag}.json").read_text())["res"]
    met = {"P4_svm_1000(ref)": metric_summary(jres(REF, TRACK / "checkpoints"))}
    for (model, mode), lvls in fam.items():
        for L, tags in sorted(lvls.items()):
            ms = [metric_summary(jres(t, CK)) for t in tags]
            met[f"{model}|{mode}|{lv(L)}"] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0] if k != "per_set_auc"}
            met[f"{model}|{mode}|{lv(L)}"]["per_set_auc"] = {k: float(np.mean([m["per_set_auc"][k] for m in ms])) for k in ms[0]["per_set_auc"]}
    diags = {}
    for f in sorted(CK.glob("diag_*.json")):
        diags[f.stem] = json.loads(f.read_text())
    reps = {"clean_check": json.loads((CK / "clean_check.json").read_text())}
    # reproduction of the kept clean results: level-0 OOF vs the stored track / matched OOF (bit identity)
    repro = {}
    for name, tag, path in (("ALL_z LR 30 Hz", tag_of("ALLz", "fixed", 0.0, 0), TRACK / "checkpoints" / "ALLz_lr_30_oof.csv.gz"),
                            ("P4 LR 30 Hz", tag_of("P4", "fixed", 0.0, 0),
                             ROOT / "results" / "frozen" / "delmas_efficiency_matched_2026-11-02" / "checkpoints" / "P4_lr_30_oof.csv.gz")):
        a = pd.read_csv(CK / f"{tag}_oof.csv.gz").sort_values(["set", "repeat", "participant", "image"]).reset_index(drop=True)
        b = pd.read_csv(path).sort_values(["set", "repeat", "participant", "image"]).reset_index(drop=True)
        repro[name] = {"same_rows": bool(len(a) == len(b) and (a[["set", "repeat", "participant", "image", "y"]].to_numpy() == b[["set", "repeat", "participant", "image", "y"]].to_numpy()).all()),
                       "max_abs_score_diff": float(np.abs(a.score.to_numpy() - b.score.to_numpy()).max())}
    R = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "margin": MARGIN, "n_boot": N_BOOT, "bootstrap_seed": BOOT_SEED,
         "reference": REF, "reference_auc_noninf_def": float(point[:, 0].mean()),
         "reference_auc_ci95": [float(x) for x in np.percentile(boot_auc[:, 0], [2.5, 97.5])],
         "levels": LEVELS, "draws": DRAWS, "tests": rows, "primary_outcome": primary, "metrics_mean_over_draws": met,
         "noise_diagnostics": diags, "checks": {**reps, "level0_vs_stored_oof": repro}, "geometry": reps["clean_check"]["geometry"]}
    jdump(OUT / "results.json", R)
    write_tables(R)
    log("stats written")
    return R


def fmt_p(p):
    return "n/a" if p is None else f"{p:.4f}"


def write_tables(R):
    rows = R["tests"]
    names = {("ALLz", "fixed"): "ALL_z LR 30 Hz (primary)", ("P4", "fixed"): "P4 LR 30 Hz (secondary)",
             ("ALLz", "adaptive"): "ALL_z LR 30 Hz, adaptive I-VT (secondary)", ("ALLznopupil", "fixed"): "ALL_z without pupil features, LR (exploratory)"}
    lines = ["| model | offset (deg) | jitter SD/axis (deg) | AUC [95% CI] | AUC per draw | Delta vs P4 SVM 1000 Hz [95% CI] | raw p | Holm p (family of 17) | Holm p (primary model only, 6) | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    csv = ["model,ivt_mode,level_deg,jitter_sd_deg,auc,auc_lo,auc_hi,delta,delta_lo,delta_hi,delta_lower_one_sided,p_raw,p_holm_family_all,p_holm_primary_only,p_holm_nopupil_family,noninferior_family_all,auc_cm_noninf_def,delta_vs_own_clean,delta_vs_own_clean_lo,delta_vs_own_clean_hi"]
    for (model, mode), nm in names.items():
        for r in sorted([r for r in rows if r["model"] == model and r["ivt_mode"] == mode], key=lambda r: r["level_deg"]):
            jit = W.noise_params(r["level_deg"])["jitter_sd_per_axis_deg"]
            pf, pp, pn = r.get("p_holm_family_all"), r.get("p_holm_primary_model_only"), r.get("p_holm_nopupil_own_family")
            verdict = ("non-inferior" if pf < 0.05 else "not shown") if pf is not None else (("non-inferior (own family)" if pn < 0.05 else "not shown (own family)") if pn is not None else "")
            lines.append(f"| {nm} | {r['level_deg']:g} | {jit:.3f} | {r['auc']:.4f} [{r['auc_ci95'][0]:.4f}, {r['auc_ci95'][1]:.4f}] | "
                         f"{', '.join(f'{a:.4f}' for a in r['auc_per_draw'])} | {r['delta']:+.4f} [{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}] | "
                         f"{r['p_inferior_raw']:.4f} | {fmt_p(pf)} | {fmt_p(pp)} | {verdict} |")
            csv.append(",".join(str(x) for x in [model, mode, r["level_deg"], jit, r["auc"], *r["auc_ci95"], r["delta"], *r["delta_ci95"],
                                                  r["delta_lower_one_sided_95"], r["p_inferior_raw"], pf, pp, pn, r.get("noninferior_holm_family_all"),
                                                  r["auc_cm_sets_noninf_def"], r["delta_vs_own_clean"], *r["delta_vs_own_clean_ci95"]]))
    (OUT / "curve_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "curve_table.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    # all-metrics table
    m = R["metrics_mean_over_draws"]
    hdr = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "auc", "eer", "hter", "auc_cm_sets_track_def", "cit_auc", "cit_hter"]
    ml = ["| config | " + " | ".join(hdr) + " |", "|---|" + "---|" * len(hdr)]
    for k, v in m.items():
        ml.append(f"| {k} | " + " | ".join(f"{v[h]:.3f}" for h in hdr) + " |")
    (OUT / "metrics_table.md").write_text("\n".join(ml) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="extract,check,run,stats")
    ap.add_argument("--n_jobs", type=int, default=2)
    a = ap.parse_args()
    st = a.stages.split(",")
    CK.mkdir(parents=True, exist_ok=True)
    if "extract" in st:
        stage_extract(min(a.n_jobs, 2))
        write_resume()
    if {"check", "run"} & set(st):
        segs = load_cache()
    if "check" in st:
        stage_check(segs)
    if "run" in st:
        assert (OUT / "PREREG.md").exists(), "PREREG.md must exist before any noisy run"
        stage_run(segs)
    if "stats" in st:
        stage_stats()
        write_resume("Stats stage finished: results.json, curve_table.md/csv, metrics_table.md written.")


if __name__ == "__main__":
    main()
