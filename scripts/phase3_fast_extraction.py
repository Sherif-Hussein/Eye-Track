"""Phase 3 efficiency experiment `fast_extraction`: vectorised feature extraction for the Delmas raw-sample
pipeline (src/edd/delmas_fast.py) and the Celniak pipeline (src/edd/celniak_fast.py) against the original code.

Stages (each writes checkpoints/<stage>.json and is skipped when it exists; long stages also keep finer checkpoints):
  delmas_decode  - one-time EDF decoding of the 75 participants into cache/delmas_decoded (same code as the track)
  delmas_equiv   - fast vs original per participant at 1000 Hz and 30 Hz: degraded signals, per-trial features,
                   per-image P4 / ALL_z matrices; and vs the track's stored table; writes the fast face table
  delmas_auc     - replay of P4 SVM 1000, P4 LR 30, ALL_z LR 30 (+ P4 LR 1000, ALL_z LR 1000) on the frozen folds
  delmas_bench   - interleaved rounds, single thread, original vs fast
  celniak_equiv, celniak_auc, celniak_bench - the same for Celniak
  report         - results.json and card.md
A speed-up is claimed for a pipeline only if its equivalence and AUC stages pass.

Usage (repo root):
  OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I scripts/phase3_fast_extraction.py --stages delmas_equiv,delmas_auc,...
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = ROOT / "results" / "phase3" / "fast_extraction"
CK = OUT / "checkpoints"
CACHE = OUT / "cache"
RAW = ROOT / "data" / "delmas" / "raw" / "data_raw"
BOOK64 = ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"
BOOK32 = ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx"
DERIVED = ROOT / "data" / "derived"
TRACK = ROOT / "results" / "phase2" / "delmas_raw"
MATCHED = ROOT / "results" / "frozen" / "delmas_efficiency_matched_2026-11-02"
CEL = ROOT / "results" / "phase2" / "celniak"
CEL_CACHE = ROOT / "data" / "celniak_cache"
RATES = (1000, 30)
ROUNDS = 5
T0 = time.time()
LOAD_NOTE = ("Timings were taken while other jobs used the CPU (a very-low-priority phase1_celniak.py job, plus "
             "whatever else was running), so every absolute time is an upper bound; original and fast were measured "
             "back to back, interleaved, in the same session.")


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def save_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def cmp(orig, fast):
    """max abs diff, NaN-pattern equality, allclose(rtol 1e-9, atol 0) of two arrays."""
    o, f = np.asarray(orig, float), np.asarray(fast, float)
    assert o.shape == f.shape, (o.shape, f.shape)
    same_nan = bool((np.isnan(o) == np.isnan(f)).all())
    both = ~np.isnan(o) & ~np.isnan(f)
    d = float(np.abs(o[both] - f[both]).max()) if both.any() else 0.0
    return {"max_abs_diff": d, "nan_pattern_equal": same_nan,
            "allclose": bool(np.allclose(o, f, rtol=1e-9, atol=0, equal_nan=True)), "n_values": int(o.size)}


def merge_cmp(acc, c):
    if acc is None:
        return dict(c)
    return {"max_abs_diff": max(acc["max_abs_diff"], c["max_abs_diff"]),
            "nan_pattern_equal": acc["nan_pattern_equal"] and c["nan_pattern_equal"],
            "allclose": acc["allclose"] and c["allclose"], "n_values": acc["n_values"] + c["n_values"]}


# ============================================================================== Delmas: shared
def load_decoded(pid):
    z = np.load(CACHE / "delmas_decoded" / f"p{pid:02d}.npz")
    return {k: z[k] for k in ("x", "y", "p", "images", "el_nfix")}, int(z["group"])


def decoded_ids():
    return sorted(int(p.stem[1:]) for p in (CACHE / "delmas_decoded").glob("p*.npz") if ".tmp" not in p.name)


def _decode_one(pid, group):
    warnings.filterwarnings("ignore")
    from edd import delmas_raw as dr
    path = CACHE / "delmas_decoded" / f"p{pid:02d}.npz"
    if path.exists():
        return pid, "cached", 0.0
    t0 = time.perf_counter()
    d = RAW / f"Sj_{pid:02d}"
    halves = sorted(d.glob(f"I{pid:02d}c2_*.edf"))
    segs = [dr.load_session(e, d / f"FI{pid:02d}c2_{e.stem.split('_')[-1]}.mat") for e in halves]
    if not segs:
        return pid, "NO EDF", 0.0
    seg = {k: np.concatenate([s[k] for s in segs]) for k in segs[0]}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, x=seg["x"], y=seg["y"], p=seg["p"], images=seg["images"], el_nfix=seg["el_nfix"], group=group)
    tmp.replace(path)
    return pid, "ok", time.perf_counter() - t0


def stage_delmas_decode(n_jobs=2):
    from joblib import Parallel, delayed

    from edd.delmas import load_trials
    (CACHE / "delmas_decoded").mkdir(parents=True, exist_ok=True)
    parts = load_trials(BOOK64).groupby("participant").group.first()
    log("decoding", len(parts), "participants with n_jobs", n_jobs)
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_decode_one)(int(p), int(g)) for p, g in parts.items())
    bad = [r for r in res if r[1] not in ("ok", "cached")]
    save_json(CK / "delmas_decode.json", {"n_participants": len(res), "failed": bad})
    log("decode done; failed:", bad)


def orig_image_features(pre, images, fs, with_msacc, z):
    """The track's `raw_to_scores` up to (and excluding) model scoring: all 14 columns of `TRIAL_FEATURES`."""
    from edd import delmas_raw as dr
    rows = {}
    for k in range(len(pre)):
        F = dr.trial_features(*pre[k], fs, with_msacc=with_msacc)
        for q in range(4):
            rows.setdefault(int(images[k, q]), []).append(F[q])
    A = np.array([np.nanmean(v, 0) for v in rows.values()])
    if z:
        A = (A - np.nanmean(A, 0)) / (np.nanstd(A, 0, ddof=1) + 1e-9)
    return A


def fast_face_rows(pid, group, seg, fs, F):
    T = F.shape[0]
    trial = np.repeat(np.arange(T), 4)
    quad = np.tile(np.arange(4), T)
    img = seg["images"][trial, quad]
    fam = (seg["images"] >= 57).any(1).astype(int)[trial]
    el = seg["el_nfix"][trial, quad] if fs == 1000 else np.full(T * 4, np.nan)
    head = np.column_stack([np.full(T * 4, pid), np.full(T * 4, group), np.full(T * 4, fs), trial, quad, img, fam, el])
    return np.hstack([head, F.reshape(T * 4, -1)])


def stage_delmas_equiv():
    from edd import delmas_fast as df
    from edd import delmas_raw as dr
    ids = decoded_ids()
    assert len(ids) == 75, len(ids)
    acc = {fs: {} for fs in RATES}
    fast_rows, fast_t, orig_t = [], 0.0, 0.0
    for n, pid in enumerate(ids):
        seg, group = load_decoded(pid)
        for fs in RATES:
            t0 = time.perf_counter()
            pre = [dr.degrade(seg["x"][k], seg["y"][k], seg["p"][k], fs) for k in range(len(seg["x"]))]
            Fo = np.array([dr.trial_features(*o, fs, True) for o in pre])
            Ao = {z: orig_image_features(pre, seg["images"], fs, fs >= 500, z) for z in (False, True)}
            t1 = time.perf_counter()
            X, Y, P = df.degrade_batch(seg["x"], seg["y"], seg["p"], fs)
            Ff = df.trial_features_batch(X, Y, P, fs, True)
            Af = {z: df.participant_image_features(X, Y, P, seg["images"], fs, z=z, with_msacc=fs >= 500) for z in (False, True)}
            t2 = time.perf_counter()
            orig_t, fast_t = orig_t + (t1 - t0), fast_t + (t2 - t1)
            p4 = [dr.TRIAL_FEATURES.index(f) for f in ("dwell_1", "dwell_2", "n_fix", "n_visit")]
            nm = [j for j in range(14) if not (fs < 500 and j == 8)]      # ALL_z has no msacc_rate below 500 Hz
            checks = {"degraded_x": cmp(np.array([o[0] for o in pre]), X), "degraded_y": cmp(np.array([o[1] for o in pre]), Y),
                      "degraded_pupil": cmp(np.array([o[2] for o in pre]), P),
                      "trial_features_all14": cmp(Fo, Ff),
                      "trial_features_P4": cmp(Fo[..., p4], Ff[..., p4]),
                      "image_matrix_P4": cmp(Ao[False][:, p4], Af[False][:, p4]),
                      "image_matrix_ALL_z": cmp(Ao[True][:, nm], Af[True][:, nm])}
            for k, c in checks.items():
                acc[fs][k] = merge_cmp(acc[fs].get(k), c)
            fast_rows.append(fast_face_rows(pid, group, seg, fs, Ff))
        if n % 10 == 0:
            log(f"equiv {n + 1}/75 participants; orig {orig_t:.0f}s fast {fast_t:.0f}s;",
                {fs: acc[fs]["trial_features_all14"]["max_abs_diff"] for fs in RATES})
    cols = ["participant", "group", "rate", "trial", "quad", "image", "fam_trial", "el_nfix"] + dr.TRIAL_FEATURES
    fast = pd.DataFrame(np.vstack(fast_rows), columns=cols)
    for c in ["participant", "group", "rate", "trial", "quad", "image", "fam_trial"]:
        fast[c] = fast[c].astype(int)
    (CACHE / "derived").mkdir(exist_ok=True)
    fast.to_csv(CACHE / "derived" / "delmas_raw_face_trials.csv.gz", index=False)
    # versus the track's stored face table (what the cross-validation used)
    stored = pd.read_csv(DERIVED / "delmas_raw_face_trials.csv.gz")
    stored = stored[stored.rate.isin(RATES)]
    key = ["participant", "rate", "trial", "quad"]
    a, b = stored.set_index(key).sort_index(), fast.set_index(key).sort_index()
    assert a.index.equals(b.index)
    feat = dr.TRIAL_FEATURES
    vs_stored = {"n_rows": int(len(a)), "identifiers_equal": bool((a[["group", "image", "fam_trial"]].to_numpy() == b[["group", "image", "fam_trial"]].to_numpy()).all()),
                 "el_nfix": cmp(a.loc[a.index.get_level_values("rate") == 1000, "el_nfix"], b.loc[b.index.get_level_values("rate") == 1000, "el_nfix"])}
    for fs in RATES:
        m = a.index.get_level_values("rate") == fs
        vs_stored[f"rate_{fs}_all14"] = cmp(a.loc[m, feat], b.loc[m, feat])
        vs_stored[f"rate_{fs}_P4"] = cmp(a.loc[m, ["dwell_1", "dwell_2", "n_fix", "n_visit"]], b.loc[m, ["dwell_1", "dwell_2", "n_fix", "n_visit"]])
    res = {"n_participants": len(ids), "live_original_vs_fast": {str(fs): acc[fs] for fs in RATES}, "fast_vs_stored_table": vs_stored,
           "seconds_original_total": orig_t, "seconds_fast_total": fast_t}
    ok = all(c["allclose"] and c["nan_pattern_equal"] for fs in RATES for c in acc[fs].values()) and \
        all(v["allclose"] and v["nan_pattern_equal"] for k, v in vs_stored.items() if isinstance(v, dict) and "allclose" in v) and \
        vs_stored["identifiers_equal"]
    res["pass"] = bool(ok)
    save_json(CK / "delmas_equiv.json", res)
    log("DELMAS EQUIVALENCE", "PASS" if ok else "FAIL")


# ----------------------------------------------------------------------------- Delmas: AUC replay
DELMAS_AUC_TAGS = [  # (tag, features key, model, rate, stored OOF file, stored json)
    ("P4_svm_1000", "P4", "svm", 1000), ("P4_lr_30", "P4", "lr", 30), ("ALLz_lr_30", "ALL_z", "lr", 30),
    ("P4_lr_1000", "P4", "lr", 1000), ("ALLz_lr_1000", "ALL_z", "lr", 1000)]


def stage_delmas_auc():
    from edd import delmas_raw_matched as M
    track = M.load_track()
    ck = CK / "delmas_auc"
    ck.mkdir(exist_ok=True)
    track.CKPT = ck
    track.N_JOBS = 2
    d_fast = CACHE / "derived"
    df_fast = track.build_tables(d_fast, BOOK64, BOOK32)
    df_orig = track.build_tables(DERIVED, BOOK64, BOOK32)
    df_orig = df_orig[df_orig.rate.isin(RATES)].reset_index(drop=True)
    cols = track.P4 + track.BINS + track.NEW
    zc = [c + "_z" for c in cols]
    eq_tab = {"all_cols_incl_z": cmp(df_orig[cols + zc].to_numpy(float), df_fast[cols + zc].to_numpy(float)),
              "keys_equal": bool(df_orig[["rate", "set", "participant", "image", "face"]].equals(df_fast[["rate", "set", "participant", "image", "face"]]))}
    log("track table (per rate, set, participant, image) fast vs original:", eq_tab)
    frozen = json.loads((TRACK / "folds.json").read_text())["repeats"]
    out = {"table_equivalence": eq_tab, "models": {}}
    for tag, fkey, model, rate in DELMAS_AUC_TAGS:
        f = out_path = ck / f"{tag}_summary.json"
        if f.exists():
            out["models"][tag] = json.loads(f.read_text())
            log("skip (cached)", tag)
            continue
        feats = track.FEATURE_SETS[fkey] if rate >= 500 else [x for x in track.FEATURE_SETS[fkey] if x != "msacc_rate_z"]
        t0 = time.time()
        store, oofs = {}, {}
        res = track.run_grid(df_fast, frozen, feats, model, rate, tag, store, oofs)
        new = pd.read_csv(ck / f"{tag}_oof.csv.gz")
        stored_oof = MATCHED / "checkpoints" / f"{tag}_oof.csv.gz"
        stored_json = MATCHED / "checkpoints" / f"{tag}.json"
        if not stored_oof.exists():
            stored_oof, stored_json = TRACK / "checkpoints" / f"{tag}_oof.csv.gz", TRACK / "checkpoints" / f"{tag}.json"
        old = pd.read_csv(stored_oof)
        old_res = json.loads(stored_json.read_text())["res"]
        k = ["set", "repeat", "participant", "image"]
        m = old.merge(new, on=k, suffixes=("_old", "_new"), validate="1:1")
        assert len(m) == len(old) == len(new)
        from sklearn.metrics import roc_auc_score

        def pooled(d, c, y="y_old"):   # track table definition: per set, per repeat pooled image AUC, mean of repeats, mean over sets
            return float(np.mean([np.mean([roc_auc_score(g[y], g[c]) for _, g in s.groupby("repeat")]) for _, s in d.groupby("set")]))
        s = {"tag": tag, "model": model, "rate": rate, "features": feats, "n_oof_rows": int(len(m)),
             "auc_image_mean_over_sets_fast_run": res["mean_over_sets"]["image"]["auc"],
             "auc_image_mean_over_sets_stored_json": old_res["mean_over_sets"]["image"]["auc"],
             "auc_pooled_per_repeat_stored_scores": pooled(m, "score_old"), "auc_pooled_per_repeat_fast_scores": pooled(m, "score_new"),
             "per_set_auc_fast": {k_: v["image"]["auc"]["mean"] for k_, v in res.items() if isinstance(v, dict) and "image" in v and k_.startswith(("64", "32"))},
             "per_set_auc_stored": {k_: v["image"]["auc"]["mean"] for k_, v in old_res.items() if isinstance(v, dict) and "image" in v and k_.startswith(("64", "32"))},
             "max_abs_score_diff": float(np.abs(m.score_old - m.score_new).max()),
             "eer_hter_fast": {m_: res["mean_over_sets"]["image"][m_] for m_ in ("eer", "hter")},
             "eer_hter_stored": {m_: old_res["mean_over_sets"]["image"][m_] for m_ in ("eer", "hter")},
             "seconds": time.time() - t0}
        s["abs_auc_diff"] = abs(s["auc_image_mean_over_sets_fast_run"] - s["auc_image_mean_over_sets_stored_json"])
        s["pass"] = bool(s["abs_auc_diff"] < 1e-9 and s["max_abs_score_diff"] < 1e-6)
        save_json(f, s)
        out["models"][tag] = s
        log(tag, {k_: s[k_] for k_ in ("auc_image_mean_over_sets_fast_run", "auc_image_mean_over_sets_stored_json", "max_abs_score_diff", "pass")})
    out["pass"] = bool(all(v["pass"] for v in out["models"].values()) and eq_tab["all_cols_incl_z"]["allclose"])
    save_json(CK / "delmas_auc.json", out)
    log("DELMAS AUC", "PASS" if out["pass"] else "FAIL")


# ----------------------------------------------------------------------------- Delmas: benchmark
def delmas_models(track, df):
    """Models fitted on all 64-trial participants at the model's own rate (as in the matched cost measurement)."""
    out = {}
    for tag, fkey, model, rate in DELMAS_AUC_TAGS:
        feats = track.FEATURE_SETS[fkey] if rate >= 500 else [x for x in track.FEATURE_SETS[fkey] if x != "msacc_rate_z"]
        d = df[(df.rate == rate) & (df.set == "64")].reset_index(drop=True)
        m = track.model_factory(model, 0).fit(d[feats].to_numpy(float), d.face.to_numpy(), groups=d.participant.to_numpy()).best_estimator_
        out[tag] = (m, feats, rate)
    return out


def stage_delmas_bench(rounds=ROUNDS):
    from edd import delmas_fast as df
    from edd import delmas_raw as dr
    from edd import delmas_raw_matched as M
    from edd import efficiency as eff
    from edd.harness import scores_of
    track = M.load_track()
    tab = track.build_tables(DERIVED, BOOK64, BOOK32)
    models = delmas_models(track, tab)
    seg, _ = load_decoded(4)                                   # Sj_04 as in the track's cost measurement
    images = seg["images"]
    pre_o, pre_f = {}, {}
    for fs in RATES:
        pre_o[fs] = [dr.degrade(seg["x"][k], seg["y"][k], seg["p"][k], fs) for k in range(64)]
        pre_f[fs] = df.degrade_batch(seg["x"], seg["y"], seg["p"], fs)
    ck = CK / "delmas_bench"
    ck.mkdir(exist_ok=True)
    N = {"o_p": (5, 1), "o_t": (30, 3), "f_p": (60, 5), "f_t": (200, 10), "inf_p": (200, 10), "inf_t": (300, 10), "from1000_o": (4, 1)}

    def L(fn, key):
        n, w = N[key]
        r = eff.latency_ms(fn, n_runs=n, warmup=w)
        return {"p50": r["p50"], "p95": r["p95"], "runs": n}

    for rd in range(rounds):
        f = ck / f"round{rd + 1}.json"
        if f.exists():
            continue
        t0 = time.time()
        res = {}
        for tag, (m, feats, fs) in models.items():
            base = [x[:-2] if x.endswith("_z") else x for x in feats]
            idx = [dr.TRIAL_FEATURES.index(x) for x in base]
            msacc = "msacc_rate" in base
            z = feats[0].endswith("_z")
            po, (X, Y, P) = pre_o[fs], pre_f[fs]
            A_ex = orig_image_features(po, images, fs, msacc, z)[:, idx]

            def o_ext():
                return orig_image_features(po, images, fs, msacc, z)[:, idx]

            def o_e2e():
                return track.raw_to_scores(po, images, fs, m, feats)

            def f_ext(chunk=64):
                return df.participant_image_features(X, Y, P, images, fs, z=z, with_msacc=msacc, chunk=chunk)[:, idx]

            def f_e2e(chunk=64):
                return df.raw_to_scores_fast(pre_f[fs], images, fs, m, feats, scores_of, chunk=chunk)

            def o_tr_ext():
                return dr.trial_features(*po[0], fs, with_msacc=msacc)[:, idx]

            def o_tr_e2e():
                return scores_of(m, dr.trial_features(*po[0], fs, with_msacc=msacc)[:, idx])

            def f_tr_ext():
                return df.trial_features_batch(X[:1], Y[:1], P[:1], fs, msacc)[0][:, idx]

            def f_tr_e2e():
                return scores_of(m, df.trial_features_batch(X[:1], Y[:1], P[:1], fs, msacc)[0][:, idx])

            # sanity: the benchmarked fast and original paths give the same scores
            sc_o, sc_f = o_e2e(), f_e2e()
            r = {"max_abs_score_diff": float(np.abs(sc_o - sc_f).max()),
                 "max_abs_trial_score_diff": float(np.abs(o_tr_e2e() - f_tr_e2e()).max())}
            first_o = rd % 2 == 0
            for variant in (("o", "f") if first_o else ("f", "o")):
                if variant == "o":
                    r["orig"] = {"ext_p": L(o_ext, "o_p"), "e2e_p": L(o_e2e, "o_p"), "ext_t": L(o_tr_ext, "o_t"), "e2e_t": L(o_tr_e2e, "o_t")}
                else:
                    r["fast"] = {"ext_p": L(f_ext, "f_p"), "e2e_p": L(f_e2e, "f_p"), "ext_t": L(f_tr_ext, "f_t"), "e2e_t": L(f_tr_e2e, "f_t"),
                                 "e2e_p_chunk8": L(lambda: f_e2e(8), "f_p")}
            r["inference_only"] = {"p": L(lambda: scores_of(m, A_ex), "inf_p"), "t": L(lambda: scores_of(m, A_ex[:4]), "inf_t")}
            if fs == 30:      # including decimation from the 1000 Hz source samples
                def o_dec():
                    pre = [dr.degrade(seg["x"][k], seg["y"][k], seg["p"][k], fs) for k in range(64)]
                    return track.raw_to_scores(pre, images, fs, m, feats)

                def f_dec():
                    return df.raw_to_scores_fast(df.degrade_batch(seg["x"], seg["y"], seg["p"], fs), images, fs, m, feats, scores_of)
                r["from_1000hz_source"] = {"orig_e2e_p": L(o_dec, "from1000_o"), "fast_e2e_p": L(f_dec, "f_p")}
            res[tag] = r
        res["_seconds"] = time.time() - t0
        save_json(f, res)
        log(f"delmas bench round {rd + 1}/{rounds} done in {res['_seconds']:.0f}s")
    # peak memory (tracemalloc), once
    mem = {}
    for tag, (m, feats, fs) in models.items():
        po = pre_o[fs]
        mem[tag] = {"orig_e2e_p": eff.peak_memory_bytes(lambda: track.raw_to_scores(po, images, fs, m, feats)),
                    "fast_e2e_p": eff.peak_memory_bytes(lambda: df.raw_to_scores_fast(pre_f[fs], images, fs, m, feats, scores_of)),
                    "fast_e2e_p_chunk8": eff.peak_memory_bytes(lambda: df.raw_to_scores_fast(pre_f[fs], images, fs, m, feats, scores_of, chunk=8)),
                    "model_size_bytes": eff.model_size_bytes(m)}
    rds = [json.loads((ck / f"round{i + 1}.json").read_text()) for i in range(rounds)]
    summ = {"rounds": rounds, "hardware": eff.hardware_info(), "load_note": LOAD_NOTE, "peak_memory_bytes": mem,
            "participant": "Sj_04, 64 trials x 5.5 s (500 ms pre-onset + 5 s display)", "models": {}}
    med = lambda v: float(np.median(v))  # noqa: E731
    for tag, (m, feats, fs) in models.items():
        n_in = 64 * pre_f[fs][0].shape[1]
        rr = [r[tag] for r in rds]

        def agg(path):
            vals = []
            for r in rr:
                v = r
                for p in path:
                    v = v[p]
                vals.append(v)
            return {"p50": med([v["p50"] for v in vals]), "p95": med([v["p95"] for v in vals]),
                    "p50_round_min": min(v["p50"] for v in vals), "p50_round_max": max(v["p50"] for v in vals)}
        S = {"features": feats, "rate": fs, "input_samples_per_participant": int(n_in),
             "max_abs_score_diff_over_rounds": max(r["max_abs_score_diff"] for r in rr),
             "inference_only_participant_ms": agg(("inference_only", "p")), "inference_only_trial_ms": agg(("inference_only", "t"))}
        for v in ("orig", "fast"):
            S[v] = {k: agg((v, k)) for k in ("ext_p", "e2e_p", "ext_t", "e2e_t")}
            e = S[v]["e2e_p"]["p50"]
            S[v]["participants_per_s"] = 1000.0 / e
            S[v]["samples_per_s"] = n_in / (e / 1000.0)
            S[v]["real_time_factor"] = n_in / (e / 1000.0) / fs
        S["fast"]["e2e_p_chunk8"] = agg(("fast", "e2e_p_chunk8"))
        S["speedup_p50"] = {k: S["orig"][k]["p50"] / S["fast"][k]["p50"] for k in ("ext_p", "e2e_p", "ext_t", "e2e_t")}
        S["within_round_speedup_e2e_p"] = [r[tag]["orig"]["e2e_p"]["p50"] / r[tag]["fast"]["e2e_p"]["p50"] for r in rds]
        if fs == 30:
            S["from_1000hz_source"] = {"orig_e2e_p": agg(("from_1000hz_source", "orig_e2e_p")), "fast_e2e_p": agg(("from_1000hz_source", "fast_e2e_p"))}
        summ["models"][tag] = S
    save_json(CK / "delmas_bench.json", summ)
    log("DELMAS BENCH done")


# ============================================================================== Celniak
def celniak_names():
    return sorted({p.name.split(".")[0] for p in CEL_CACHE.glob("Recording*.csv.gz")}, key=lambda s: int(s.replace("Recording", "")))


def stage_celniak_equiv():
    import phase2_celniak_efficiency as E
    from edd import celniak_fast as cf
    from edd.celniak import PAPER_FEATURES, load_cached, slide_features
    names = celniak_names()
    assert len(names) == 39
    stored_slides = pd.read_csv(DERIVED / "celniak_slide_features.csv")
    stored_p = pd.read_csv(DERIVED / "celniak_p2_participant_features.csv").set_index("recording")
    acc = {}
    fast_tabs, gg_rows, orig_t, fast_t, per_rec = [], [], 0.0, 0.0, {}
    for i, name in enumerate(names):
        samples, meta = load_cached(CEL_CACHE, name)
        samples = samples.reset_index(drop=True)
        R = cf.arrays_from_frame(samples)
        t0 = time.perf_counter()
        O = slide_features(samples, meta)
        g0 = E.lean_global(samples, meta).to_numpy(float)
        t1 = time.perf_counter()
        F = cf.slide_features_fast(R)
        g1 = cf.global_gaze_fast(R)
        t2 = time.perf_counter()
        orig_t, fast_t = orig_t + t1 - t0, fast_t + t2 - t1
        num = [c for c in O.columns if c in F]
        assert [c for c in O.columns if c not in F] == ["participant", "recording", "label", "first_exam"]
        c_all = cmp(O[num].to_numpy(float), np.column_stack([F[c] for c in num]))
        c_pap = cmp(O[PAPER_FEATURES].to_numpy(float), cf.paper_matrix(F, PAPER_FEATURES))
        c_gg = cmp(g0, g1)
        # vs stored tables (what the cross-validation used)
        st = stored_slides[stored_slides.recording == name]
        c_st = cmp(st[PAPER_FEATURES].to_numpy(float), cf.paper_matrix(F, PAPER_FEATURES))
        c_stall = cmp(st[num].to_numpy(float), np.column_stack([F[c] for c in num]))
        c_stgg = cmp(stored_p.loc[name, cf.GG].to_numpy(float), g1)
        for k, c in (("slide_all_62_columns_vs_live_original", c_all), ("slide_paper19_vs_live_original", c_pap),
                     ("global_gaze7_vs_live_lean_original", c_gg), ("slide_paper19_vs_stored_table", c_st),
                     ("slide_all_62_columns_vs_stored_table", c_stall), ("global_gaze7_vs_stored_participant_table", c_stgg)):
            acc[k] = merge_cmp(acc.get(k), c)
        per_rec[name] = {"n_samples": int(len(samples)), "n_slides": int(len(O)), "sensor_hz": round(1e6 / float(np.median(np.diff(R.ts))), 1),
                         "max_abs_diff_slide_all": c_all["max_abs_diff"], "max_abs_diff_gg": c_gg["max_abs_diff"]}
        tab = pd.DataFrame(cf.paper_matrix(F, PAPER_FEATURES), columns=PAPER_FEATURES)
        tab.insert(0, "recording", meta["Recording name"])
        tab.insert(1, "label", int(str(meta["group"]).strip().lower() == "guilty"))
        fast_tabs.append(tab)
        gg_rows.append({"recording": meta["Recording name"], **dict(zip(cf.GG, g1))})
        if i % 5 == 0:
            log(f"celniak equiv {i + 1}/39; orig {orig_t:.0f}s fast {fast_t:.1f}s", {k: v["max_abs_diff"] for k, v in acc.items()})
    (CACHE / "derived").mkdir(exist_ok=True)
    pd.concat(fast_tabs, ignore_index=True).to_csv(CACHE / "derived" / "celniak_fast_slides.csv", index=False)
    pd.DataFrame(gg_rows).to_csv(CACHE / "derived" / "celniak_fast_gg.csv", index=False)
    ok = all(v["allclose"] and v["nan_pattern_equal"] for v in acc.values())
    save_json(CK / "celniak_equiv.json", {"n_recordings": len(names), "checks": acc, "per_recording": per_rec, "pass": bool(ok),
                                          "seconds_original_total": orig_t, "seconds_fast_total": fast_t})
    log("CELNIAK EQUIVALENCE", "PASS" if ok else "FAIL")


def stage_celniak_auc():
    import phase2_celniak_screen as S
    from edd.celniak import PAPER_FEATURES
    from edd.folds import load_folds
    from edd.harness import aggregate, grouped_inner, make_search, nested_fit, run_outer
    from edd.metrics import binary_metrics
    folds = load_folds(CEL / "folds.json")
    fs_ = pd.read_csv(CACHE / "derived" / "celniak_fast_slides.csv")
    gg = pd.read_csv(CACHE / "derived" / "celniak_fast_gg.csv").set_index("recording")
    slides = pd.read_csv(DERIVED / "celniak_slide_features.csv")
    P = pd.read_csv(DERIVED / "celniak_p2_participant_features.csv")
    P[S.UNIT] = P[S.UNIT].astype(str)
    slides_fast = slides.copy()
    # replace the model inputs by the fast ones (same row order, checked)
    assert (slides_fast.recording.to_numpy() == fs_.recording.to_numpy()).all()
    slides_fast[PAPER_FEATURES] = fs_[PAPER_FEATURES].to_numpy()
    P_fast = P.copy()
    GGc = ["sl_nfix", "sl_fixdur", "sl_nsacc", "sl_nblink", "sl_scanpath", "sl_aoi_dwell", "valid_frac"]
    P_fast[GGc] = gg.loc[P[S.UNIT], GGc].to_numpy()
    out = {"models": {}}
    runs = {"baseline_svm_paper": (slides, slides_fast, PAPER_FEATURES, lambda: nested_fit(lambda: make_search(S.svm_pipe(), S.SVM_GRID, grouped_inner(S.INNER_SPLITS)))),
            "global_gaze_lr": (P, P_fast, GGc, lambda: nested_fit(S.lr_search_factory(len(GGc))))}
    for name, (d_o, d_f, feats, fit) in runs.items():
        f = CK / "celniak_auc" / f"{name}.json"
        f.parent.mkdir(exist_ok=True)
        if f.exists():
            out["models"][name] = json.loads(f.read_text())
            continue
        t0 = time.time()
        new = run_outer(d_f, feats, S.LABEL, S.UNIT, folds, fit(), n_jobs=2)
        old = pd.read_csv(CEL / f"trials_{name}.csv")
        assert len(old) == len(new) and (old[["repeat", "fold", "unit", "y"]].to_numpy() == new[["repeat", "fold", "unit", "y"]].to_numpy()).all()

        def auc_rep(tr):
            return [float(binary_metrics(q.y, q.score)["auc"]) for _, q in aggregate(tr).groupby("repeat")]
        a_old, a_new = auc_rep(old), auc_rep(new)
        s = {"model": name, "n_oof_rows": int(len(new)), "auc_per_repeat_stored": a_old, "auc_per_repeat_fast": a_new,
             "auc_mean_stored": float(np.mean(a_old)), "auc_mean_fast": float(np.mean(a_new)),
             "max_abs_score_diff": float(np.abs(old.score - new.score).max()),
             "chosen_params_identical": bool((old.params.astype(str) == new.params.astype(str)).all()),
             "auc_equal_within_1e-12": bool(np.allclose(a_old, a_new, rtol=0, atol=1e-12)), "seconds": time.time() - t0}
        s["pass"] = bool(s["auc_equal_within_1e-12"] and s["chosen_params_identical"])
        save_json(f, s)
        out["models"][name] = s
        log(name, {k: s[k] for k in ("auc_mean_stored", "auc_mean_fast", "max_abs_score_diff", "chosen_params_identical", "pass")})
    out["pass"] = bool(all(v["pass"] for v in out["models"].values()))
    save_json(CK / "celniak_auc.json", out)
    log("CELNIAK AUC", "PASS" if out["pass"] else "FAIL")


def stage_celniak_bench(rounds=ROUNDS, rec="Recording10"):
    import phase2_celniak_efficiency as E
    import phase2_celniak_screen as S
    from edd import celniak_fast as cf
    from edd import efficiency as eff
    from edd.celniak import PAPER_FEATURES, load_cached, slide_features
    from edd.harness import grouped_inner, make_search
    slides = pd.read_csv(DERIVED / "celniak_slide_features.csv")
    P = pd.read_csv(DERIVED / "celniak_p2_participant_features.csv")
    P[S.UNIT] = P[S.UNIT].astype(str)
    samples, meta = load_cached(CEL_CACHE, rec)
    samples = samples.reset_index(drop=True)
    R = cf.arrays_from_frame(samples)
    n = len(samples)
    hz = 1e6 / float(np.median(np.diff(R.ts)))
    X, y, g = slides[PAPER_FEATURES].to_numpy(float), slides[S.LABEL].to_numpy(int), slides[S.UNIT].astype(str).to_numpy()
    svm = make_search(S.svm_pipe(), S.SVM_GRID, grouped_inner(S.INNER_SPLITS)).fit(X, y, groups=g).best_estimator_
    Xp, yp, gp = P[E.GG].to_numpy(float), P[S.LABEL].to_numpy(int), P[S.UNIT].to_numpy()
    lr = S.lr_search_factory(len(E.GG))().fit(Xp, yp, groups=gp).best_estimator_
    ck = CK / "celniak_bench"
    ck.mkdir(exist_ok=True)

    def L(fn, runs, warm):
        r = eff.latency_ms(fn, n_runs=runs, warmup=warm)
        return {"p50": r["p50"], "p95": r["p95"], "runs": runs}

    def svm_o():
        return svm.decision_function(slide_features(samples, meta)[PAPER_FEATURES].to_numpy(float))

    def svm_f():
        return svm.decision_function(cf.paper_matrix(cf.slide_features_fast(R), PAPER_FEATURES))

    def gg_o():
        return lr.decision_function(E.lean_global(samples, meta).to_numpy(float)[None, :])

    def gg_f():
        return lr.decision_function(cf.global_gaze_fast(R)[None, :])
    Xs = cf.paper_matrix(cf.slide_features_fast(R), PAPER_FEATURES)
    xg = cf.global_gaze_fast(R)[None, :]
    for rd in range(rounds):
        f = ck / f"round{rd + 1}.json"
        if f.exists():
            continue
        t0 = time.time()
        res = {"svm_diff": float(np.abs(svm_o() - svm_f()).max()), "gg_diff": float(np.abs(gg_o() - gg_f()).max())}
        for variant in (("o", "f") if rd % 2 == 0 else ("f", "o")):
            if variant == "o":
                res["orig"] = {"svm_ext": L(lambda: slide_features(samples, meta), 4, 1), "svm_e2e": L(svm_o, 4, 1),
                               "gg_ext": L(lambda: E.lean_global(samples, meta), 4, 1), "gg_e2e": L(gg_o, 4, 1)}
            else:
                res["fast"] = {"svm_ext": L(lambda: cf.slide_features_fast(R), 60, 5), "svm_e2e": L(svm_f, 60, 5),
                               "gg_ext": L(lambda: cf.global_gaze_fast(R), 100, 5), "gg_e2e": L(gg_f, 100, 5),
                               "ingestion_from_dataframe": L(lambda: cf.arrays_from_frame(samples), 8, 1)}
        res["inference_only"] = {"svm": L(lambda: svm.decision_function(Xs), 300, 10), "gg": L(lambda: lr.decision_function(xg), 300, 10)}
        res["_seconds"] = time.time() - t0
        save_json(f, res)
        log(f"celniak bench round {rd + 1}/{rounds} done in {res['_seconds']:.0f}s")
    mem = {"svm_orig": eff.peak_memory_bytes(svm_o), "svm_fast": eff.peak_memory_bytes(svm_f),
           "gg_orig": eff.peak_memory_bytes(gg_o), "gg_fast": eff.peak_memory_bytes(gg_f),
           "svm_model_size_bytes": eff.model_size_bytes(svm), "lr_model_size_bytes": eff.model_size_bytes(lr),
           "arrays_in_memory_bytes_fast_input": int(sum(a.nbytes for a in R if isinstance(a, np.ndarray)))}
    rds = [json.loads((ck / f"round{i + 1}.json").read_text()) for i in range(rounds)]
    med = lambda v: float(np.median(v))  # noqa: E731

    def agg(path):
        vals = []
        for r in rds:
            v = r
            for p in path:
                v = v[p]
            vals.append(v)
        return {"p50": med([v["p50"] for v in vals]), "p95": med([v["p95"] for v in vals]),
                "p50_round_min": min(v["p50"] for v in vals), "p50_round_max": max(v["p50"] for v in vals)}
    summ = {"recording": rec, "recording_samples": n, "sensor_hz": hz, "n_slides": 21, "rounds": rounds, "hardware": eff.hardware_info(),
            "load_note": LOAD_NOTE, "peak_memory_bytes": mem, "max_abs_decision_diff": {"svm": max(r["svm_diff"] for r in rds), "gg": max(r["gg_diff"] for r in rds)},
            "ingestion_from_dataframe_ms": agg(("fast", "ingestion_from_dataframe")), "models": {}}
    for key, label in (("svm", "baseline_svm_paper"), ("gg", "global_gaze_lr")):
        S_ = {"inference_only_ms": agg(("inference_only", key))}
        for v in ("orig", "fast"):
            S_[v] = {"ext": agg((v, f"{key}_ext")), "e2e": agg((v, f"{key}_e2e"))}
            e = S_[v]["e2e"]["p50"]
            S_[v]["participants_per_s"] = 1000.0 / e
            S_[v]["samples_per_s"] = n / (e / 1000.0)
            S_[v]["real_time_factor"] = n / (e / 1000.0) / hz
            S_[v]["per_slide_amortised_ms_p50"] = e / 21
        S_["speedup_p50"] = {k: S_["orig"][k]["p50"] / S_["fast"][k]["p50"] for k in ("ext", "e2e")}
        S_["within_round_speedup_e2e"] = [r["orig"][f"{key}_e2e"]["p50"] / r["fast"][f"{key}_e2e"]["p50"] for r in rds]
        summ["models"][label] = S_
    save_json(CK / "celniak_bench.json", summ)
    log("CELNIAK BENCH done")


# ============================================================================== report
def _pp(v, d=2):
    return f"{v['p50']:.{d}f} / {v['p95']:.{d}f}"


def _yn(x):
    return "yes" if x else "NO"


def stage_report():
    ck = lambda n: json.loads((CK / f"{n}.json").read_text()) if (CK / f"{n}.json").exists() else None  # noqa: E731
    D = {"equiv": ck("delmas_equiv"), "auc": ck("delmas_auc"), "bench": ck("delmas_bench")}
    C = {"equiv": ck("celniak_equiv"), "auc": ck("celniak_auc"), "bench": ck("celniak_bench")}
    claim = {"delmas": bool(D["equiv"] and D["auc"] and D["equiv"]["pass"] and D["auc"]["pass"] and D["bench"]),
             "celniak": bool(C["equiv"] and C["auc"] and C["equiv"]["pass"] and C["auc"]["pass"] and C["bench"])}
    save_json(OUT / "results.json", {"delmas": D, "celniak": C, "speedup_claimed": claim, "note": LOAD_NOTE})
    L = ["# Phase 3: fast_extraction (vectorised feature extraction), before / after", "",
         "Single thread (`edd.efficiency`, `threadpool_limits(1)`, `OMP_NUM_THREADS=1`). Original = the track's code; fast = `src/edd/delmas_fast.py` / "
         "`src/edd/celniak_fast.py`. Both measured back to back in the same session, in interleaved rounds (order alternated each round); tables give the "
         "median over rounds of each round's p50 and p95. " + LOAD_NOTE, "",
         f"**Verdict.** Delmas: features equivalent {_yn(D['equiv'] and D['equiv']['pass'])}; out-of-fold AUC unchanged "
         f"{_yn(D['auc'] and D['auc']['pass'])}; speed-up claimed: {_yn(claim['delmas'])}. "
         f"Celniak: features equivalent {_yn(C['equiv'] and C['equiv']['pass'])}; out-of-fold AUC unchanged "
         f"{_yn(C['auc'] and C['auc']['pass'])}; speed-up claimed: {_yn(claim['celniak'])}.", ""]
    # ------------------------------------------------------------------ Delmas
    L += ["## 1. Delmas raw-sample pipeline (Sj_04, 64 trials x 5.5 s; EDF decoding excluded)", "",
          "Timed path (as in the frozen matched-cost measurement): sensor-rate samples -> I-VT -> AOI -> dwell / fixation / visit / first-look / time-bin / "
          "pupil (and, at 1000 Hz for ALL_z, microsaccade) features -> per-image aggregation -> (per-participant z-scoring for ALL_z) -> model score. "
          "Decimation from 1000 Hz is simulated before the timer (as in the track) and reported separately below for 30 Hz."]
    if D["bench"]:
        b = D["bench"]
        L += ["", "### 1a. Latency (ms, p50 / p95)", "",
              "| Model | Unit | Stage | Original | Fast | Speed-up (p50) |", "|---|---|---|---|---|---|"]
        for tag, S in b["models"].items():
            for unit, ek in (("participant (64 images)", "p"), ("trial (4 faces)", "t")):
                o, f_ = S["orig"]["ext_" + ek], S["fast"]["ext_" + ek]
                L.append(f"| {tag} | {unit} | feature extraction only | {_pp(o)} | {_pp(f_, 3)} | {o['p50'] / f_['p50']:.1f}x |")
                inf = S["inference_only_participant_ms" if ek == "p" else "inference_only_trial_ms"]
                L.append(f"| {tag} | {unit} | inference only (same model code) | {_pp(inf, 3)} | {_pp(inf, 3)} | 1.0x |")
                o, f_ = S["orig"]["e2e_" + ek], S["fast"]["e2e_" + ek]
                L.append(f"| {tag} | {unit} | **end to end** | {_pp(o)} | {_pp(f_, 3)} | **{o['p50'] / f_['p50']:.1f}x** |")
        L += ["", "### 1b. Throughput, real-time factor, memory", "",
              "| Model | Participants/s orig -> fast | Samples/s orig -> fast | Real-time factor orig -> fast | Peak memory orig -> fast (chunk 64) -> fast (chunk 8) | Model size | Per-round e2e speed-up (min to max) |",
              "|---|---|---|---|---|---|---|"]
        for tag, S in b["models"].items():
            m = b["peak_memory_bytes"][tag]
            sp = S["within_round_speedup_e2e_p"]
            L.append(f"| {tag} | {S['orig']['participants_per_s']:.2f} -> {S['fast']['participants_per_s']:.1f} | "
                     f"{S['orig']['samples_per_s']:,.0f} -> {S['fast']['samples_per_s']:,.0f} | {S['orig']['real_time_factor']:,.0f}x -> {S['fast']['real_time_factor']:,.0f}x | "
                     f"{m['orig_e2e_p'] / 1e3:,.0f} kB -> {m['fast_e2e_p'] / 1e3:,.0f} kB -> {m['fast_e2e_p_chunk8'] / 1e3:,.0f} kB | {m['model_size_bytes']:,} B | {min(sp):.1f}x to {max(sp):.1f}x |")
        L += ["", "Throughput = 1000 / end-to-end p50 per participant (sequential, one thread). Samples/s = input samples per participant "
              "(64 x samples per trial, including the 500 ms pre-onset window) / end-to-end p50. Real-time factor = samples/s divided by the sensor rate. "
              "Peak memory is the Python-tracked (tracemalloc) peak of one participant's pass; the fast path holds whole (64, n) arrays, "
              "`chunk 8` processes 8 trials at a time (same features, lower peak).", "",
              "| Model | e2e per participant, fast, chunk 8 (p50 / p95 ms) |", "|---|---|"]
        for tag, S in b["models"].items():
            L.append(f"| {tag} | {_pp(S['fast']['e2e_p_chunk8'], 3)} |")
        L += ["", "### 1c. Including decimation from the 1000 Hz source samples (30 Hz models)", "",
              "| Model | Original e2e (p50 / p95 ms) | Fast e2e | Speed-up (p50) |", "|---|---|---|---|"]
        for tag, S in b["models"].items():
            if "from_1000hz_source" in S:
                o, f_ = S["from_1000hz_source"]["orig_e2e_p"], S["from_1000hz_source"]["fast_e2e_p"]
                L.append(f"| {tag} | {_pp(o)} | {_pp(f_, 3)} | {o['p50'] / f_['p50']:.1f}x |")
        L += ["", f"Rounds: {b['rounds']}. Per-round runs: original participant 5 (+1 warm-up) and trial 30; fast participant 60 and trial 200. "
              f"Max abs score difference original vs fast in the benchmarked paths: {max(S['max_abs_score_diff_over_rounds'] for S in b['models'].values()):.2e}."]
    if D["equiv"]:
        e = D["equiv"]
        L += ["", "### 1d. Equivalence of the features (all 75 participants)", "",
              "`numpy.allclose(rtol=1e-9, atol=0)` and identical NaN pattern. 'Live original' = the original code re-run on the decoded samples; "
              "'stored' = `data/derived/delmas_raw_face_trials.csv.gz`, the table the cross-validation used.", "",
              "| Comparison | Rate | Max abs diff | NaN pattern equal | allclose | Values |", "|---|---|---|---|---|---|"]
        for fs, chk in e["live_original_vs_fast"].items():
            for k, c in chk.items():
                L.append(f"| live original vs fast: {k} | {fs} Hz | {c['max_abs_diff']:.2e} | {_yn(c['nan_pattern_equal'])} | {_yn(c['allclose'])} | {c['n_values']:,} |")
        for k, c in e["fast_vs_stored_table"].items():
            if isinstance(c, dict):
                L.append(f"| stored table vs fast: {k} | | {c['max_abs_diff']:.2e} | {_yn(c['nan_pattern_equal'])} | {_yn(c['allclose'])} | {c['n_values']:,} |")
    if D["auc"]:
        a = D["auc"]
        te = a["table_equivalence"]
        L += ["", "### 1e. Out-of-fold AUC with the fast features (frozen folds, 5 folds x 3 repeats x 6 group-sets, nested tuning, n_jobs = 2)", "",
              "The track's own `build_tables` + `run_grid` were run on the fast face table. Fast vs original track table (all features and z-scores): "
              f"max abs diff {te['all_cols_incl_z']['max_abs_diff']:.2e}, allclose {te['all_cols_incl_z']['allclose']}, identical keys and labels {te['keys_equal']}.", "",
              "| Model | Image AUC mean over 6 sets: stored | fast | abs diff | Pooled-per-repeat AUC stored / fast | Max abs OOF score diff | EER / HTER stored | EER / HTER fast | Pass |",
              "|---|---|---|---|---|---|---|---|---|"]
        for tag, s in a["models"].items():
            L.append(f"| {tag} | {s['auc_image_mean_over_sets_stored_json']:.6f} | {s['auc_image_mean_over_sets_fast_run']:.6f} | {s['abs_auc_diff']:.1e} | "
                     f"{s['auc_pooled_per_repeat_stored_scores']:.6f} / {s['auc_pooled_per_repeat_fast_scores']:.6f} | {s['max_abs_score_diff']:.2e} | "
                     f"{s['eer_hter_stored']['eer']:.4f} / {s['eer_hter_stored']['hter']:.4f} | {s['eer_hter_fast']['eer']:.4f} / {s['eer_hter_fast']['hter']:.4f} | {_yn(s['pass'])} |")
    # ------------------------------------------------------------------ Celniak
    L += ["", "## 2. Celniak pipeline (cached raw Tobii samples)", "",
          "Original = `edd.celniak.slide_features` (the paper's AOI metrics, honest-baseline SVM input) and the track's lean global-gaze extractor "
          "(`phase2_celniak_efficiency.lean_global`). Fast = `slide_features_fast` / `global_gaze_fast` on NumPy columns of the same samples."]
    if C["bench"]:
        b = C["bench"]
        m = b["peak_memory_bytes"]
        L += ["", f"Recording {b['recording']}: {b['recording_samples']:,} samples at {b['sensor_hz']:.0f} Hz ({b['recording_samples'] / b['sensor_hz']:.0f} s), 21 slides. "
              "The decision unit is the participant (21 slides); per-slide values are the per-participant value divided by 21 (the original extractor "
              "processes the whole recording, including the cross-screen pupil baseline, so a standalone per-slide original call is not defined).", "",
              "### 2a. Latency (ms, p50 / p95)", "",
              "| Model | Stage | Original | Fast | Speed-up (p50) |", "|---|---|---|---|---|"]
        for tag, S in b["models"].items():
            o, f_ = S["orig"]["ext"], S["fast"]["ext"]
            L.append(f"| {tag} | feature extraction only | {_pp(o)} | {_pp(f_, 3)} | {o['p50'] / f_['p50']:.1f}x |")
            L.append(f"| {tag} | inference only (same model code) | {_pp(S['inference_only_ms'], 3)} | {_pp(S['inference_only_ms'], 3)} | 1.0x |")
            o, f_ = S["orig"]["e2e"], S["fast"]["e2e"]
            L.append(f"| {tag} | **end to end, per participant** | {_pp(o)} | {_pp(f_, 3)} | **{o['p50'] / f_['p50']:.1f}x** |")
            L.append(f"| {tag} | end to end, per slide (amortised, p50) | {S['orig']['per_slide_amortised_ms_p50']:.2f} | {S['fast']['per_slide_amortised_ms_p50']:.3f} | |")
        L += ["", "### 2b. Throughput, real-time factor, memory", "",
              "| Model | Participants/s orig -> fast | Samples/s orig -> fast | Real-time factor orig -> fast | Peak memory orig -> fast | Model size | Per-round e2e speed-up (min to max) |",
              "|---|---|---|---|---|---|---|"]
        for tag, S in b["models"].items():
            k = "svm" if tag.startswith("baseline") else "gg"
            sp = S["within_round_speedup_e2e"]
            L.append(f"| {tag} | {S['orig']['participants_per_s']:.3f} -> {S['fast']['participants_per_s']:.1f} | {S['orig']['samples_per_s']:,.0f} -> {S['fast']['samples_per_s']:,.0f} | "
                     f"{S['orig']['real_time_factor']:,.0f}x -> {S['fast']['real_time_factor']:,.0f}x | {m[k + '_orig'] / 1e6:.1f} MB -> {m[k + '_fast'] / 1e6:.1f} MB | "
                     f"{m['svm_model_size_bytes' if k == 'svm' else 'lr_model_size_bytes']:,} B | {min(sp):.1f}x to {max(sp):.1f}x |")
        L += ["", f"Fast path input: NumPy columns built once from the cached DataFrame (ingestion = data loading, {_pp(b['ingestion_from_dataframe_ms'], 1)} ms p50 / p95, "
              "not included above; the original's DataFrame is likewise built beforehand). The raw arrays held by the fast path occupy "
              f"{m['arrays_in_memory_bytes_fast_input'] / 1e6:.1f} MB, which the tracemalloc peak does not count. "
              f"Max abs decision-function difference original vs fast: SVM {b['max_abs_decision_diff']['svm']:.2e}, global-gaze LR {b['max_abs_decision_diff']['gg']:.2e}. Rounds: {b['rounds']}."]
    if C["equiv"]:
        e = C["equiv"]
        L += ["", f"### 2c. Equivalence of the features (all {e['n_recordings']} recordings)", "",
              "| Comparison | Max abs diff | NaN pattern equal | allclose | Values |", "|---|---|---|---|---|"]
        for k, c in e["checks"].items():
            L.append(f"| {k} | {c['max_abs_diff']:.2e} | {_yn(c['nan_pattern_equal'])} | {_yn(c['allclose'])} | {c['n_values']:,} |")
        hz = sorted({v["sensor_hz"] for v in e["per_recording"].values()})
        L += ["", f"Sensor rates present: {hz} Hz; 21 slides per recording."]
    if C["auc"]:
        L += ["", "### 2d. Out-of-fold participant AUC with the fast features (frozen folds, 3 repeats, nested tuning, n_jobs = 2)", "",
              "| Model | AUC per repeat, stored | AUC per repeat, fast | Mean stored / fast | Max abs OOF score diff | Same chosen params | Pass |", "|---|---|---|---|---|---|---|"]
        for tag, s in C["auc"]["models"].items():
            L.append(f"| {tag} | {', '.join(f'{x:.6f}' for x in s['auc_per_repeat_stored'])} | {', '.join(f'{x:.6f}' for x in s['auc_per_repeat_fast'])} | "
                     f"{s['auc_mean_stored']:.6f} / {s['auc_mean_fast']:.6f} | {s['max_abs_score_diff']:.2e} | {_yn(s['chosen_params_identical'])} | {_yn(s['pass'])} |")
    L += ["", "## 3. Caveats", "",
          "- " + LOAD_NOTE,
          "- Original and fast share the same fitted model and the same sklearn inference code, so inference-only is identical by construction.",
          "- Equivalence is to floating-point summation order (about 1e-12); event decisions (fixations, visits, first looks, counts) are identical.",
          "- Delmas per-participant timing is for one participant (Sj_04) with all 64 trials; Celniak for one 250 Hz recording (Recording10). Rates below the sensor "
          "rate are simulated from a research tracker by decimation, as in the track.",
          "- EDF decoding (Delmas) and CSV parsing (Celniak) are excluded from both pipelines; the fast path's DataFrame-to-array ingestion is reported separately.",
          "- Peak memory is the Python-tracked allocation (tracemalloc), not process RSS."]
    (OUT / "card.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("report written")


STAGES = {"delmas_decode": stage_delmas_decode, "delmas_equiv": stage_delmas_equiv, "delmas_auc": stage_delmas_auc,
          "delmas_bench": stage_delmas_bench, "celniak_equiv": stage_celniak_equiv, "celniak_auc": stage_celniak_auc,
          "celniak_bench": stage_celniak_bench}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--n_jobs", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    a = ap.parse_args()
    CK.mkdir(parents=True, exist_ok=True)
    for s in a.stages.split(","):
        if s == "report":
            stage_report()
            continue
        if (CK / f"{s}.json").exists():
            log("skip (checkpoint exists):", s)
            continue
        log("stage", s)
        if s == "delmas_decode":
            STAGES[s](a.n_jobs)
        elif s.endswith("bench"):
            STAGES[s](a.rounds)
        else:
            STAGES[s]()


if __name__ == "__main__":
    main()
