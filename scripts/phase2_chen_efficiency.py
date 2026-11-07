"""Efficiency of the paper's rule (re-implementation) vs the best candidate, single-threaded CPU.

Decision units: one trial (13-face RSVP stream, critical face scored) and one participant (all of
that person's familiar + control trials -> one-tailed t-test). Costs are measured from the rawest
data we hold: 1 kHz EyeLink samples of the RSVP phase decoded from the EDF (decoding itself is
reported separately because it is an edfapi/I/O cost, not part of the detector).
Usage: python -I scripts/phase2_chen_efficiency.py [best_idea]
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edd import chen  # noqa: E402
from edd.efficiency import (hardware_info, latency_ms, model_size_bytes, peak_memory_bytes,  # noqa: E402
                            throughput_per_s)

spec_ = importlib.util.spec_from_file_location("p2", ROOT / "scripts" / "phase2_chen.py")
p2 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(p2)
OUT = p2.OUT
BEST = sys.argv[1] if len(sys.argv) > 1 else "I3b_pnorm_fusion"
SUBJECT_EDF = ROOT / "data" / "chen" / "exp1" / "EDF" / "Face_16.edf"
FS = 1000.0


def raw_trials(path):
    """1 kHz RSVP-phase arrays per trial (as a live system would buffer them)."""
    rec = chen.read_edf_fast(path)
    t, bl = rec["t"], rec["blinks"]
    out = []
    for tr in chen.parse_trials(rec):
        if tr["t0"] is None or tr["vars"].get("trialtype") != "1" or tr["vars"].get("T1") not in ("2", "3", "4"):
            continue
        sel = slice(np.searchsorted(t, tr["t0"]), np.searchsorted(t, tr["t1"]))
        bm = np.zeros(sel.stop - sel.start, bool)
        for s, e in bl[(bl[:, 1] >= tr["t0"]) & (bl[:, 0] < tr["t1"])]:
            bm[max(0, s - tr["t0"]):max(0, min(len(bm), e - tr["t0"] + 1))] = True
        out.append({"pa": rec["pa"][sel], "gx": rec["gx"][sel], "gy": rec["gy"][sel], "blink": bm,
                    "T1": int(tr["vars"]["T1"]), "T1pos": int(tr["vars"]["T1pos"]),
                    "dir": int(tr["vars"]["T1_direction"]), "width": float(tr["vars"]["width"])})
    return out


def trial_channels(r, full=True):
    """Raw 1 kHz -> 100 Hz epochs (blink reconstruction, bin-mean downsampling, 50-ms subtractive
    baseline, onset lock, rate = smoothed diff); gaze channels only when `full`."""
    p100 = chen.downsample(chen.dm_blinkreconstruct(r["pa"]))[None]
    pos = np.array([r["T1pos"]])
    b = chen.baseline(p100, pos)
    pup = chen.epoch(p100, pos) - b[:, None]
    ch = {"pupil": pup, "rate": chen.smooth_hanning(np.diff(pup, axis=1), 25)}
    if full:
        x = chen.epoch(chen.downsample(r["gx"])[None], pos)
        y = chen.epoch(chen.downsample(r["gy"])[None], pos)
        miss = chen.epoch(chen.downsample((~np.isfinite(r["pa"]) | r["blink"]).astype(float))[None], pos)
        ch.update({"dx": (x - r["width"] / 2) * r["dir"], "dy": y - 540.0, "miss": miss})
    return ch


def main():
    res = {"hardware": hardware_info(), "best": BEST}
    t0 = time.perf_counter()
    rec_n = len(chen.read_edf_fast(SUBJECT_EDF)["t"])
    dec = time.perf_counter() - t0
    res["edf_decode"] = {"seconds_one_subject_file": dec, "rsvp_samples_decoded": rec_n,
                         "us_per_sample": dec / rec_n * 1e6,
                         "note": "edfapi float conversion; a live EyeLink link delivers samples without this cost"}
    trials = raw_trials(SUBJECT_EDF)
    n_samples = np.mean([len(r["pa"]) for r in trials])

    # Fit both models on all exp1 participants except the one used for timing
    meta, arr = p2.load("exp1")
    m, ch, _ = p2.preprocess(meta, arr)
    mf, chf = p2.fc_subset(m, ch)
    keep = mf.subject_nr.to_numpy() != 16
    mtr, chtr = mf.loc[keep].reset_index(drop=True), {k: v[keep] for k, v in chf.items()}
    spec = p2.idea_specs("exp1")[BEST]
    data = p2.build_data(mtr, chtr, spec)
    est, _ = p2.fit_lr(data["X"], data["y"], data["subj"], spec.get("select", True))
    rule = p2.PeakRule("rate").fit(chtr, np.arange(len(mtr)), (mtr.T1.to_numpy() == 2).astype(int))

    # per-person normalisation stats come from the person's own control trials at test time
    def person_feats(trs):
        F = np.vstack([p2.bin_features(trial_channels(r), spec["channels"])[0] for r in trs])
        is_c = np.array([r["T1"] != 2 for r in trs])
        return p2.person_normalise(F, np.zeros(len(trs)), is_c) if spec.get("pnorm") else F

    F_person = person_feats(trials)
    one = trials[0]

    def best_trial():
        f = p2.bin_features(trial_channels(one), spec["channels"])[0]
        return est.decision_function(f)

    def base_trial():
        c = trial_channels(one, full=False)
        return c["rate"][0, rule.peak]

    def best_person():
        F = person_feats(trials)
        s = est.decision_function(F)
        is_f = np.array([r["T1"] == 2 for r in trials])
        return chen.one_tailed_t(s[is_f], s[~is_f])

    def base_person():
        v = np.array([trial_channels(r, full=False)["rate"][0, rule.peak] for r in trials])
        is_f = np.array([r["T1"] == 2 for r in trials])
        return chen.one_tailed_t(v[is_f], v[~is_f])

    # pure model inference on precomputed features (no signal processing)
    Fp = F_person

    def best_model_only():
        return est.decision_function(Fp[:1])

    n_tr = len(trials)
    for name, ft, fp, full in (("paper_rule_rate", base_trial, base_person, False), (BEST, best_trial, best_person, True)):
        lt = latency_ms(ft, n_runs=300, warmup=20)
        lp = latency_ms(fp, n_runs=15, warmup=2)
        thr = throughput_per_s(ft, 1, seconds=2.0)
        fx = latency_ms(lambda: trial_channels(one, full=full), n_runs=300, warmup=20)
        res[name] = {
            "latency_trial_ms": lt, "latency_participant_ms": lp, "trials_per_participant": n_tr,
            "throughput_trials_per_s": thr, "throughput_raw_samples_per_s": thr * n_samples,
            "real_time_factor": (lt["p50"] / 1e3) / (n_samples / FS),
            "x_faster_than_real_time": (n_samples / FS) / (lt["p50"] / 1e3),
            "feature_extraction_trial_ms": fx,
            "peak_memory_participant_bytes": peak_memory_bytes(fp),
        }
    res["paper_rule_rate"].update({"model_size_bytes": model_size_bytes({"peak": rule.peak, "cal": rule.cal}),
                                   "parameter_count": 1 + 2, "parameters": "peak sample index + 1-D logistic calibration (slope, intercept); detection itself needs only the peak index"})
    sel = est.named_steps.get("sel")
    n_in = Fp.shape[1]
    n_used = int(sel.get_support().sum()) if sel is not None else n_in
    n_params = n_used + 1 + 2 * n_in + n_in  # LR weights+bias, scaler mean/scale, imputer medians
    res[BEST].update({"model_size_bytes": model_size_bytes(est), "parameter_count": int(n_params),
                      "n_input_features": int(n_in), "n_selected_features": n_used,
                      "model_only_latency_trial_ms": latency_ms(best_model_only, n_runs=500, warmup=20),
                      "note": "per-person normalisation statistics (2 x n_features) are computed at test time from the person's own control trials; they are not stored model parameters"})
    res["raw_samples_per_trial_mean"] = float(n_samples)
    (OUT / "efficiency.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}) for k, v in res.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
