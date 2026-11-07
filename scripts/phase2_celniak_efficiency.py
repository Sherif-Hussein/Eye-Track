"""Phase 2 (celniak track), takeover stage 5: efficiency of the honest baseline (paper SVM) and global_gaze_lr.

Single CPU thread (edd.efficiency uses threadpool_limits(1)), models fitted on all 39 participants with the
screening grids, feature extraction timed from the cached raw samples of Recording10 (250 Hz). Besides the full
extraction used for the screening (all idea features), a lean extractor computes only the 7 global_gaze features;
it is verified to reproduce the stored global_gaze values for the recording.

Output: results/phase2/celniak/stage_efficiency.json
Usage: python -I scripts/phase2_celniak_efficiency.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import phase2_celniak_features as fx  # noqa: E402
import phase2_celniak_screen as S  # noqa: E402
from edd import efficiency as eff  # noqa: E402
from edd.celniak import PAPER_FEATURES, load_cached, slide_features  # noqa: E402
from edd.harness import grouped_inner, make_search  # noqa: E402

OUT = ROOT / "results" / "phase2" / "celniak"
DER = ROOT / "data" / "derived"
GG = ["sl_nfix", "sl_fixdur", "sl_nsacc", "sl_nblink", "sl_scanpath", "sl_aoi_dwell", "valid_frac"]
REC = "Recording10"


def lean_global(samples, meta):
    """Per-slide global_gaze quantities only (no pupil cleaning, no item tables, no pre-stimulus periods)."""
    samples = samples.reset_index(drop=True)
    dt_us = float(samples["ts_us"].diff().median())
    dt_ms = dt_us / 1e3
    blinks = fx.blink_onsets(samples, dt_ms)
    rows_out = []
    for stim, rows in fx.stimulus_runs(samples):
        if not fx.SLIDE_RE.match(stim):
            continue
        slide = samples.iloc[rows]
        aoi_cols = [c for c in samples.columns if c.startswith(f"aoi:{stim} - ")]
        ev = fx._fixations(slide, aoi_cols, dt_us)
        dur = [(e["end"] - e["start"]) / 1e3 for e in ev]
        xs, ys = [e["x"] for e in ev], [e["y"] for e in ev]
        rows_out.append({
            "sl_nfix": float(len(ev)), "sl_fixdur": float(np.mean(dur)) if ev else np.nan,
            "sl_nsacc": float(slide.loc[slide["move_type"] == "Saccade", "move_index"].nunique()),
            "sl_nblink": float(np.isin(blinks, rows).sum()),
            "sl_scanpath": float(np.nansum(np.hypot(np.diff(xs), np.diff(ys)))) if len(ev) > 1 else 0.0,
            "sl_aoi_dwell": float(sum(d for d, e in zip(dur, ev) if e["aoi"] is not None)),
            "valid_frac": float((slide["valid_l"].eq("Valid") & slide["valid_r"].eq("Valid")).mean())})
    return pd.DataFrame(rows_out).mean()[GG]


def main():
    T0 = time.perf_counter()
    slides = pd.read_csv(DER / "celniak_slide_features.csv")
    P = pd.read_csv(DER / "celniak_p2_participant_features.csv")
    P[S.UNIT] = P[S.UNIT].astype(str)
    samples, meta = load_cached(ROOT / "data" / "celniak_cache", REC)
    samples = samples.reset_index(drop=True)
    sensor_hz = 1e6 / float(samples["ts_us"].diff().median())
    n = len(samples)
    session_s = n / sensor_hz
    res = {"hardware": eff.hardware_info(), "recording": REC, "recording_samples": int(n), "sensor_hz": sensor_hz,
           "recording_duration_s": session_s, "n_slides_per_participant": 21,
           "note": "single CPU thread; another low-priority job was running on the machine; models fitted on all 39 participants"}

    # ---------------- lean extractor must reproduce the stored global features
    lean = lean_global(samples, meta)
    stored = P.set_index(S.UNIT).loc[REC, GG]
    res["lean_vs_stored_max_abs_rel_diff"] = float(np.max(np.abs((lean.to_numpy(float) - stored.to_numpy(float)) / stored.to_numpy(float))))
    print("lean extractor max relative difference to stored:", res["lean_vs_stored_max_abs_rel_diff"], flush=True)

    # ---------------- baseline: paper SVM
    X, y = slides[PAPER_FEATURES].to_numpy(float), slides[S.LABEL].to_numpy(int)
    g = slides[S.UNIT].astype(str).to_numpy()
    srch = make_search(S.svm_pipe(), S.SVM_GRID, grouped_inner(S.INNER_SPLITS)).fit(X, y, groups=g)
    svm = srch.best_estimator_
    ksel = int(svm.named_steps["select"].get_support().sum())
    one = X[:21]
    lat_ex = eff.latency_ms(lambda: slide_features(samples, meta), n_runs=10, warmup=1)
    base = {
        "final_params": srch.best_params_, "n_support_vectors": int(svm.named_steps["clf"].support_vectors_.shape[0]),
        "n_selected_features": ksel, "model_size_bytes": eff.model_size_bytes(svm),
        "parameter_count": eff.svm_parameter_count(svm.named_steps["clf"], ksel) + 2 * len(PAPER_FEATURES),
        "latency_per_participant_ms": eff.latency_ms(lambda: svm.decision_function(one), n_runs=500),
        "latency_per_slide_ms": eff.latency_ms(lambda: svm.decision_function(X[:1]), n_runs=500),
        "throughput_slides_per_s": eff.throughput_per_s(lambda: svm.decision_function(X), len(X)),
        "peak_memory_inference_bytes": eff.peak_memory_bytes(lambda: svm.decision_function(one)),
        "feature_extraction_per_recording_ms": lat_ex,
        "feature_extraction_samples_per_s": float(n / (lat_ex["p50"] / 1e3)),
        "realtime_factor_vs_sensor": float(n / (lat_ex["p50"] / 1e3) / sensor_hz),
        "feature_extraction_peak_memory_bytes": eff.peak_memory_bytes(lambda: slide_features(samples, meta))}
    base["throughput_participants_per_s"] = base["throughput_slides_per_s"] / 21
    base["end_to_end_per_participant_ms_p50"] = lat_ex["p50"] + base["latency_per_participant_ms"]["p50"]
    res["baseline_svm_paper"] = base
    print("baseline done", flush=True)

    # ---------------- global_gaze_lr
    Xp, yp, gp = P[GG].to_numpy(float), P[S.LABEL].to_numpy(int), P[S.UNIT].to_numpy()
    s = S.lr_search_factory(len(GG))().fit(Xp, yp, groups=gp)
    mdl = s.best_estimator_
    x1 = Xp[:1]
    lat_full = eff.latency_ms(lambda: fx.process_samples(samples, meta), n_runs=10, warmup=1)
    lat_lean = eff.latency_ms(lambda: lean_global(samples, meta), n_runs=10, warmup=1)
    n_kept = int(mdl.named_steps["select"].get_support().sum())
    kept = [f for f, k in zip(GG, mdl.named_steps["select"].get_support()) if k]
    gg = {
        "final_params": s.best_params_, "selected_features": kept,
        "coefficients_standardised": dict(zip(kept, mdl.named_steps["clf"].coef_[0].round(4).tolist())),
        "model_size_bytes": eff.model_size_bytes(mdl),
        "parameter_count": int(2 * mdl.named_steps["scale"].n_features_in_ + mdl.named_steps["impute"].n_features_in_ + n_kept + 1),
        "latency_per_participant_ms": eff.latency_ms(lambda: mdl.decision_function(x1), n_runs=500),
        "throughput_participants_per_s": eff.throughput_per_s(lambda: mdl.decision_function(Xp), len(Xp)),
        "peak_memory_inference_bytes": eff.peak_memory_bytes(lambda: mdl.decision_function(x1)),
        "feature_extraction_full_pipeline_per_recording_ms": lat_full,
        "feature_extraction_full_samples_per_s": float(n / (lat_full["p50"] / 1e3)),
        "feature_extraction_full_realtime_factor": float(n / (lat_full["p50"] / 1e3) / sensor_hz),
        "feature_extraction_full_peak_memory_bytes": eff.peak_memory_bytes(lambda: fx.process_samples(samples, meta)),
        "feature_extraction_lean_per_recording_ms": lat_lean,
        "feature_extraction_lean_samples_per_s": float(n / (lat_lean["p50"] / 1e3)),
        "feature_extraction_lean_realtime_factor": float(n / (lat_lean["p50"] / 1e3) / sensor_hz),
        "feature_extraction_lean_peak_memory_bytes": eff.peak_memory_bytes(lambda: lean_global(samples, meta)),
        "note": "full = every feature family of the screening (pupil cleaning, items, pre-stimulus, series); "
                "lean = only the 7 global_gaze features (verified identical to the stored values)"}
    gg["end_to_end_lean_per_participant_ms_p50"] = lat_lean["p50"] + gg["latency_per_participant_ms"]["p50"]
    res["global_gaze_lr"] = gg
    res["seconds"] = time.perf_counter() - T0
    (OUT / "stage_efficiency.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
