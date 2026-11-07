"""Matched accuracy + cost measurement for the delmas_raw track (2026-11-02).

Why: the frozen claim combines accuracy of ALL_z LR with cost of P4 LR. This script measures
  acc     - P4 LR at 60 and 30 Hz (and a 1000 Hz re-run as a reproducibility check) with the track's own
            pipeline (frozen folds.json, same decimation / I-VT / features / nested tuning), no existing file changed;
  noninf  - non-inferiority (margin 0.02 AUC) of three new comparisons, same method as
            phase2_delmas_raw_noninf.py, Holm over the 3 new + the 6 existing = 9 tests;
  cost    - end-to-end raw-to-score cost of ALL_z LR and P4 LR at 30 and 1000 Hz, single thread, per participant
            (64 images), with edd.efficiency through the track's own cost_from_raw();
  report  - collect everything into results.json.
Every stage checkpoints into <out>/checkpoints/ and is skipped when its checkpoint exists.

Usage (project root):
  OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I scripts/phase2_delmas_raw_matched.py [--stages acc,noninf,cost,report]
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

from edd import delmas_raw_matched as M  # noqa: E402

ROOT = M.ROOT
TRACK = M.TRACK_DIR
OUT = ROOT / "results" / "frozen" / "delmas_efficiency_matched_2026-11-02"
CK = OUT / "checkpoints"
DERIVED = ROOT / "data" / "derived"
BOOK64 = ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"
BOOK32 = ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx"
RAW = ROOT / "data" / "delmas" / "raw" / "data_raw"
N_BOOT = 2000
MARGIN = 0.02
EXISTING_PAIRS = [("P4_svm_1000", "P4_svm_60"), ("P4_svm_1000", "P4_svm_30"), ("ALLz_lr_1000", "ALLz_lr_60"),
                  ("ALLz_lr_1000", "ALLz_lr_30"), ("P4_svm_1000", "ALLz_lr_1000"), ("P4_svm_1000", "ALLz_lr_30")]
NEW_PAIRS = [("P4_svm_1000", "P4_lr_30"), ("P4_svm_1000", "P4_lr_60"), ("P4_lr_1000", "P4_lr_30")]
COST_ROUNDS = 9


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


# ----------------------------------------------------------------------------- accuracy
def stage_acc():
    track = M.load_track()
    track.CKPT = CK
    log("building tables from", DERIVED)
    df = track.build_tables(DERIVED, BOOK64, BOOK32)
    # the folds used are the frozen ones; check that the track code would regenerate exactly the same
    frozen = json.loads((TRACK / "folds.json").read_text())["repeats"]
    scratch = Path(os.environ.get("TEMP", ".")) / "folds_check_matched.json"
    regen = track.make_folds(df, scratch)
    same = json.loads(json.dumps(regen)) == frozen
    log("frozen folds.json identical to regenerated folds:", same)
    assert same, "frozen folds differ from regenerated folds"
    store, oofs, out = {}, {}, {}
    for rate in (60, 30, 1000):
        tag = f"P4_lr_{rate}" if rate != 1000 else "P4_lr_1000_rerun"
        out[tag] = track.run_grid(df, frozen, track.P4, "lr", rate, tag, store, oofs)
        log(tag, "mean AUC", out[tag]["mean_over_sets"]["image"]["auc"])
    return out


# ----------------------------------------------------------------------------- non-inferiority
def oof_path(name):
    new = CK / f"{name}_oof.csv.gz"
    return new if new.exists() else TRACK / "checkpoints" / f"{name}_oof.csv.gz"


def load_oof(name):
    d = pd.read_csv(oof_path(name))
    return d.groupby(["set", "participant", "image"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))


def stage_noninf():
    ni = M.load_noninf()
    assert ni.MARGIN == MARGIN
    old = json.loads((TRACK / "noninferiority.json").read_text())
    cache = {}
    rows = []
    # one rng stream in the order of the track script: the 6 existing pairs first (reproduced, and compared with
    # the stored noninferiority.json), then the 3 new pairs
    rng = np.random.default_rng(0)
    for ref, cand in EXISTING_PAIRS + NEW_PAIRS:
        for n in (ref, cand):
            cache.setdefault(n, load_oof(n))
        per, nrows = M.per_set_arrays(cache[ref], cache[cand])
        t0 = time.time()
        ar, ac, d = M.paired_cluster_bootstrap(per, rng, N_BOOT)
        r = M.noninf_row(ref, cand, d, ar, ac, MARGIN)
        r["n_merged_rows"] = nrows
        r["source"] = "existing" if (ref, cand) in EXISTING_PAIRS else "new"
        rows.append(r)
        log(f"{cand} vs {ref}: {ar:.4f} -> {ac:.4f} delta {ac - ar:+.4f} CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
            f"p_raw {r['p_inferior_raw']:.4f} ({time.time() - t0:.0f}s)")
    # reproduction check of the 6 existing pairs
    chk = []
    for r, o in zip(rows[:6], old["pairs"]):
        chk.append({"pair": f"{o['candidate']} vs {o['reference']}",
                    "max_abs_diff": float(max(abs(r["delta"] - o["delta"]), abs(r["ci95"][0] - o["ci95"][0]),
                                              abs(r["ci95"][1] - o["ci95"][1]), abs(r["p_inferior_raw"] - o["p_inferior_raw"])))})
    log("reproduction of the 6 stored pairs, max abs diffs:", [round(c["max_abs_diff"], 12) for c in chk])
    adj9 = ni.holm(np.array([r["p_inferior_raw"] for r in rows]))
    for r, a in zip(rows, adj9):
        r["p_inferior_holm_over_9"] = float(a)
        r["verdict"] = "non-inferior (Holm over 9, margin 0.02)" if a < 0.05 else "non-inferiority not shown"
        r["ci_lower_ge_minus_margin"] = bool(r["ci95"][0] >= -MARGIN)
    for r, o in zip(rows[:6], old["pairs"]):
        r["p_inferior_holm_over_6_stored"] = o["p_inferior_holm"]
    # model-level CIs (each model's own metric)
    ci = {}
    for n in ("P4_svm_1000", "P4_lr_1000", "P4_lr_60", "P4_lr_30", "ALLz_lr_1000", "ALLz_lr_30"):
        ci[n] = M.model_auc_ci(cache.setdefault(n, load_oof(n)), N_BOOT)
        log("model CI", n, ci[n])
    return {"margin": MARGIN, "n_boot": N_BOOT, "holm_family_size": 9, "pairs": rows,
            "reproduction_of_stored_pairs": chk, "model_auc_ci": ci,
            "oof_inputs": {n: {"path": str(oof_path(n).relative_to(ROOT)), "sha256": M.sha256(oof_path(n))} for n in cache}}


# ----------------------------------------------------------------------------- cost
def stage_cost():
    from edd import efficiency as eff
    from edd.delmas_raw import load_session
    from edd.harness import scores_of
    track = M.load_track()
    track.CKPT = CK
    df = track.build_tables(DERIVED, BOOK64, BOOK32)
    pid = 4
    log("loading raw session of Sj_04 (EDF decode is excluded from the timings)")
    seg = load_session(RAW / f"Sj_{pid:02d}" / f"I{pid:02d}c2_1.edf", RAW / f"Sj_{pid:02d}" / f"FI{pid:02d}c2_1.mat")
    allz = track.FEATURE_SETS["ALL_z"]
    specs = {"P4_lr_30": ("P4", track.P4, 30), "ALLz_lr_30": ("ALL_z", [f for f in allz if f != "msacc_rate_z"], 30),
             "P4_lr_1000": ("P4", track.P4, 1000), "ALLz_lr_1000": ("ALL_z", allz, 1000)}
    models = {}
    for tag, (fsn, feats, rate) in specs.items():
        d = df[(df.rate == rate) & (df.set == "64")].reset_index(drop=True)
        m = track.model_factory("lr", 0).fit(d[feats].to_numpy(float), d.face.to_numpy(), groups=d.participant.to_numpy()).best_estimator_
        clf = m.named_steps["clf"]
        x64 = d[feats].to_numpy(float)[:64]
        npar = M.n_stored_parameters(clf, len(feats))
        models[tag] = (m, feats, rate, {
            "feature_set": fsn, "n_features": len(feats), "features": feats, "rate": rate,
            "fitted_on": f"all 64-trial participants at {rate} Hz (n={len(d)} image rows)",
            "lr_params": {"C": float(clf.C)}, "model_size_bytes": eff.model_size_bytes(m),
            "stored_parameters_track_convention": npar["track_convention"], "stored_parameters_strict": npar["strict_minimum"],
            "inference_per_participant_64_images_ms": eff.latency_ms(lambda m=m, x=x64: scores_of(m, x), n_runs=500),
            "inference_per_face_row_ms": eff.latency_ms(lambda m=m, x=x64: scores_of(m, x[:1]), n_runs=500),
            "inference_peak_memory_bytes": eff.peak_memory_bytes(lambda m=m, x=x64: scores_of(m, x))})
        log(tag, "size", models[tag][3]["model_size_bytes"], "params", npar)
    base_order = ["P4_lr_30", "ALLz_lr_30", "P4_lr_1000", "ALLz_lr_1000"]
    rounds = []
    for k in range(COST_ROUNDS):
        f = CK / f"cost_round{k + 1}.json"
        if f.exists():
            rounds.append(json.loads(f.read_text()))
            log("cached", f.name)
            continue
        order = base_order if k % 2 == 0 else base_order[::-1]      # alternate order to spread any load drift
        res = {"order": order, "hardware": eff.hardware_info(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        for tag in order:
            m, feats, rate, _ = models[tag]
            res[tag] = track.cost_from_raw(seg, rate, m, feats)
            log(f"round {k + 1} {tag}: per participant p50 {res[tag]['per_participant_ms']['p50']:.1f} ms "
                f"p95 {res[tag]['per_participant_ms']['p95']:.1f}, peak {res[tag]['peak_memory_bytes']} B")
        jdump(f, res)
        rounds.append(res)
    return {"model_static": {t: v[3] for t, v in models.items()}, "rounds": rounds,
            "protocol": "single thread (OMP_NUM_THREADS=1 + threadpool_limits(1)); simulated sensor output of one participant "
                        "(Sj_04, 64 trials x 5.5 s) -> anti-alias decimation excluded from timing -> I-VT -> features -> per-image "
                        "aggregation -> (per-participant z-score for ALL_z) -> LR; track function cost_from_raw: per-participant "
                        "n_runs=10 after 2 warm-ups, per-trial n_runs=200; EDF decoding excluded; tracemalloc peak."}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="acc,noninf,cost,report")
    st = ap.parse_args().stages.split(",")
    CK.mkdir(parents=True, exist_ok=True)
    for name, fn in (("acc", stage_acc), ("noninf", stage_noninf), ("cost", stage_cost)):
        f = CK / f"stage_{name}.json"
        if name in st:
            if f.exists():
                log("[skip, cached]", name)
            else:
                t0 = time.time()
                jdump(f, fn())
                log(f"[stage {name} done in {time.time() - t0:.0f}s]")
    if "report" in st:
        R = {"created": time.strftime("%Y-%m-%d %H:%M:%S")}
        for name in ("acc", "noninf", "cost"):
            f = CK / f"stage_{name}.json"
            if f.exists():
                R[name] = json.loads(f.read_text())
        # reproducibility of the 1000 Hz P4 LR re-run against the track's stored OOF scores
        if "acc" in R:
            a = pd.read_csv(CK / "P4_lr_1000_rerun_oof.csv.gz").sort_values(["set", "repeat", "participant", "image"]).reset_index(drop=True)
            b = pd.read_csv(TRACK / "checkpoints" / "P4_lr_1000_oof.csv.gz").sort_values(["set", "repeat", "participant", "image"]).reset_index(drop=True)
            R["rerun_check_P4_lr_1000"] = {"same_rows": bool(len(a) == len(b) and (a[["set", "repeat", "participant", "image", "y"]].values == b[["set", "repeat", "participant", "image", "y"]].values).all()),
                                           "max_abs_score_diff": float(np.abs(a.score.values - b.score.values).max()) if len(a) == len(b) else None,
                                           "track_mean_auc": json.loads((TRACK / "checkpoints" / "P4_lr_1000.json").read_text())["res"]["mean_over_sets"]["image"]["auc"],
                                           "rerun_mean_auc": R["acc"]["P4_lr_1000_rerun"]["mean_over_sets"]["image"]["auc"]}
            log("rerun check:", R["rerun_check_P4_lr_1000"])
        if "cost" in R:
            rounds = R["cost"]["rounds"]
            tags = ["P4_lr_30", "ALLz_lr_30", "P4_lr_1000", "ALLz_lr_1000"]
            summ = {}
            for t in tags:
                pp50 = [r[t]["per_participant_ms"]["p50"] for r in rounds]
                pp95 = [r[t]["per_participant_ms"]["p95"] for r in rounds]
                pt50 = [r[t]["per_trial_ms"]["p50"] for r in rounds]
                pt95 = [r[t]["per_trial_ms"]["p95"] for r in rounds]
                summ[t] = {"n_rounds": len(rounds),
                           "per_participant_p50_ms_median_over_rounds": float(np.median(pp50)),
                           "per_participant_p50_ms_min_max": [float(min(pp50)), float(max(pp50))],
                           "per_participant_p95_ms_median_over_rounds": float(np.median(pp95)),
                           "per_participant_p95_ms_min_max": [float(min(pp95)), float(max(pp95))],
                           "per_trial_p50_ms_median_over_rounds": float(np.median(pt50)),
                           "per_trial_p95_ms_median_over_rounds": float(np.median(pt95)),
                           "peak_memory_bytes_median_over_rounds": float(np.median([r[t]["peak_memory_bytes"] for r in rounds])),
                           "peak_memory_bytes_min_max": [int(min(r[t]["peak_memory_bytes"] for r in rounds)), int(max(r[t]["peak_memory_bytes"] for r in rounds))],
                           "input_samples_per_participant": rounds[0][t]["input_samples_per_participant"],
                           "real_time_factor_median": float(np.median([r[t]["real_time_factor"] for r in rounds]))}
            # within-round (same minute, same load) ratios ALL_z / P4
            for fs in (30, 1000):
                ratio = [r[f"ALLz_lr_{fs}"]["per_participant_ms"]["p50"] / r[f"P4_lr_{fs}"]["per_participant_ms"]["p50"] for r in rounds]
                summ[f"ratio_ALLz_over_P4_per_participant_p50_{fs}Hz"] = {"median": float(np.median(ratio)), "min": float(min(ratio)), "max": float(max(ratio)),
                                                                          "rounds_where_ALLz_slower": int(sum(x > 1 for x in ratio))}
            R["cost_summary"] = summ
        jdump(OUT / "results.json", R)
        log("wrote results.json")


if __name__ == "__main__":
    main()
