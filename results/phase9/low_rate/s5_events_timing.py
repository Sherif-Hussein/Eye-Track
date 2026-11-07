"""Task 2.2 (EXPLORATORY, not preregistered), step 5: (A) event-detection diagnostic at 30/20/15/10 Hz (Delmas, clean data, like
Supplementary Table S10 = results/phase6/trackC/c2_event_stats_by_rate.csv + c2_parser_agreement.csv) and (B) processing time per
participant at each rate.

(A) Uses scripts/phase6_trackC_c2.py UNCHANGED (parser_arrays, ivt_rate -> run in s1_extract.py; levels_r, pear here) on the
    per-participant results cached by s1_extract.py (cache/ev_p*.pkl). The tracker's own (EyeLink parser) fixations are the reference.
    Rows 30 Hz are compared with the stored c2 tables (reproduction check).
(B) One participant's session = 64 trials x 5.5 s of 1000 Hz samples. Time from the 1000 Hz samples to per-image model scores:
    anti-aliased resampling to the rate + I-VT + feature extraction + per-image aggregation + z-scoring + logistic-regression score
    (13 standardised features, no microsaccades). Two implementations that exist in the project: the original loop code
    (edd.delmas_raw.degrade/trial_features via scripts/phase2_delmas_raw.py raw_to_scores) and the vectorised code (edd.delmas_fast).
    Median of 3 rounds per participant after one warm-up, then median over 5 participants; one process, one thread; the 1000 Hz row is the
    no-resampling reference. Model: LogisticRegression fitted on this study's 30 Hz table (scoring cost is identical at every rate).
"""
import importlib.util
import pickle
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lr_common as L  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edd import delmas_fast as df_  # noqa: E402
from edd import delmas_raw as dr  # noqa: E402
from edd import delmas_raw_matched as M  # noqa: E402
from edd.harness import scores_of  # noqa: E402

spec = importlib.util.spec_from_file_location("trackC_c2", L.ROOT / "scripts" / "phase6_trackC_c2.py")
c2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c2)
OUT = L.OUT
RATES = [30, 20, 15, 10]
c2.RATES = RATES                      # levels_r loops over the module-level list (module attribute only; file untouched)
STORED_C2 = L.ROOT / "results" / "phase6" / "trackC"


def events():
    pids = sorted(int(f.stem[4:]) for f in (OUT / "cache").glob("ev_p??.pkl"))
    results = {p: pickle.load(open(OUT / "cache" / f"ev_p{p:02d}.pkl", "rb")) for p in pids}
    assert len(pids) == 75
    rows = []
    par_n = np.concatenate([results[p]["parser"]["n_all"] for p in pids])
    par_d = np.concatenate([results[p]["parser"]["durs"] for p in pids])
    par_face = np.concatenate([results[p]["parser"]["cnt"] for p in pids])
    rows.append({"rate_hz": "EyeLink parser (1000 Hz)", "fix_per_trial_all": par_n.mean(), "fix_per_trial_on_face": par_face.sum(1).mean(),
                 "mean_fix_dur_ms": par_d.mean(), "median_fix_dur_ms": np.median(par_d), "velocity_window_ms": np.nan, "min_detectable_saccade_deg": np.nan})
    for fs in RATES:
        n_all = np.concatenate([results[p]["rates"][fs]["n_all"] for p in pids])
        cnt = np.concatenate([results[p]["rates"][fs]["cnt"] for p in pids])
        durs = np.concatenate([results[p]["rates"][fs]["durs"] for p in pids])
        h = max(1, int(round(fs * 0.004)))
        span_ms = 2 * h * 1000.0 / fs
        rows.append({"rate_hz": fs, "fix_per_trial_all": n_all.mean(), "fix_per_trial_on_face": cnt.sum(1).mean(), "mean_fix_dur_ms": durs.mean(),
                     "median_fix_dur_ms": np.median(durs), "velocity_window_ms": span_ms, "min_detectable_saccade_deg": 30.0 * span_ms / 1000.0})
    ev = pd.DataFrame(rows)
    ag = {"count": c2.levels_r(pids, results, "cnt", "cnt"), "dwell_1": c2.levels_r(pids, results, "dw1", "dw1"),
          "dwell_2": c2.levels_r(pids, results, "dw2", "dw2"),
          "sample_dwell_1_vs_parser": c2.levels_r(pids, results, "sb1", "dw1"), "sample_dwell_2_vs_parser": c2.levels_r(pids, results, "sb2", "dw2")}
    arows = []
    for metric, d in ag.items():
        for fs, r in d.items():
            arows.append({"metric": metric, "rate_hz": fs, "r_trial_x_face": r["trial_face"], "r_participant_x_image": r["participant_image"],
                          "r_image_level": r["image_level"], "r_within_participant": r["within_participant"]})
    agdf = pd.DataFrame(arows)
    wide = ev.copy()
    for metric in ag:
        sub = agdf[agdf.metric == metric].set_index("rate_hz")
        wide[f"{metric}_r_participant_x_image"] = [np.nan if r not in sub.index else sub.loc[r, "r_participant_x_image"] for r in wide.rate_hz]
        wide[f"{metric}_r_trial_x_face"] = [np.nan if r not in sub.index else sub.loc[r, "r_trial_x_face"] for r in wide.rate_hz]
    wide.to_csv(OUT / "event_diag.csv", index=False)
    agdf.to_csv(OUT / "event_diag_agreement_long.csv", index=False)
    # reproduction of the 30 Hz row against the stored Track C2 tables
    se = pd.read_csv(STORED_C2 / "c2_event_stats_by_rate.csv")
    sa = pd.read_csv(STORED_C2 / "c2_parser_agreement.csv")
    rep = {}
    for hz in (30,):
        se_r = se[se.rate_hz.astype(str) == str(hz)].iloc[0]
        me_r = ev[ev.rate_hz == hz].iloc[0]
        rep[f"{hz}Hz_fix_per_trial_all"] = {"new": float(me_r.fix_per_trial_all), "stored": float(se_r.fix_per_trial_all)}
        rep[f"{hz}Hz_mean_fix_dur_ms"] = {"new": float(me_r.mean_fix_dur_ms), "stored": float(se_r.mean_fix_dur_ms_pooled)}
        for metric in ("count", "dwell_1", "dwell_2"):
            s_ = sa[(sa.metric == metric) & (sa.rate_hz == hz)].iloc[0]
            n_ = agdf[(agdf.metric == metric) & (agdf.rate_hz == hz)].iloc[0]
            rep[f"{hz}Hz_{metric}_r_participant_x_image"] = {"new": float(n_.r_participant_x_image), "stored": float(s_.participant_image)}
            rep[f"{hz}Hz_{metric}_r_trial_x_face"] = {"new": float(n_.r_trial_x_face), "stored": float(s_.trial_face)}
    ok = all(abs(v["new"] - v["stored"]) < 1e-9 for v in rep.values())
    rep["all_passed"] = bool(ok)
    L.jdump(OUT / "repro_events.json", rep)
    L.log("event diag written; 30 Hz reproduces stored C2:", ok)
    print(wide.round(3).to_string())
    print(agdf.round(3).to_string())
    if not ok:
        L.log("EVENT REPRODUCTION MISMATCH", rep)
        sys.exit(2)


def timing(n_part=5, rounds=3):
    track = M.load_track()
    track.N_JOBS = 1
    feats = [f for f in track.FEATURE_SETS["ALL_z"] if f != "msacc_rate_z"]
    df = track.build_tables(OUT / "derived", L.ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx", L.ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx")
    d = df[(df.rate == 30) & (df.set == "64")]
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(np.nan_to_num(d[feats].to_numpy(float)), d.face.to_numpy())
    pids = sorted(int(f.stem[1:]) for f in (OUT / "cache").glob("p??.npz"))[:n_part]
    rows = []
    for pid in pids:
        x, y, p, images, _ = c2.load_halves(pid)
        n_tr = len(images)
        X, Y, P = x.astype(np.float64), y.astype(np.float64), p.astype(np.float64)
        for fs in [1000, 30, 20, 15, 10]:
            def orig():
                t0 = time.perf_counter()
                pre = [dr.degrade(x[k], y[k], p[k], fs) for k in range(n_tr)]
                t1 = time.perf_counter()
                track.raw_to_scores(pre, images, fs, m, feats)
                return (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000

            def fast():
                t0 = time.perf_counter()
                pre = df_.degrade_batch(X, Y, P, fs)
                t1 = time.perf_counter()
                df_.raw_to_scores_fast(pre, images, fs, m, feats, scores_of)
                return (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000
            for impl, fn in (("original loop code", orig), ("vectorised code", fast)):
                fn()                                                          # warm-up
                r = np.array([fn() for _ in range(rounds)])                   # (rounds, 2): resample ms, features+score ms
                tot = r.sum(1)
                rows.append({"participant": pid, "rate_hz": fs, "implementation": impl, "resample_ms": float(np.median(r[:, 0])),
                             "features_score_ms": float(np.median(r[:, 1])), "total_ms": float(np.median(tot))})
        L.log("timing participant", pid)
    t = pd.DataFrame(rows)
    sess_s = n_tr * (dr.PRE_MS + dr.POST_MS) / 1000.0
    s = t.groupby(["implementation", "rate_hz"], as_index=False).agg(resample_ms=("resample_ms", "median"), features_score_ms=("features_score_ms", "median"),
                                                                     total_ms=("total_ms", "median"), total_ms_min=("total_ms", "min"), total_ms_max=("total_ms", "max"))
    s["n_participants"] = len(pids)
    s["rounds_per_participant"] = rounds
    s["session_seconds_of_data"] = sess_s
    s["input_samples_after_resampling"] = (sess_s * s.rate_hz).astype(int)
    s["real_time_factor"] = sess_s * 1000.0 / s.total_ms
    s["threads"] = 1
    s = s.sort_values(["implementation", "rate_hz"], ascending=[True, False])
    s.to_csv(OUT / "timing.csv", index=False)
    t.to_csv(OUT / "timing_per_participant.csv", index=False)
    print(s.round(2).to_string())


if __name__ == "__main__":
    st = (sys.argv[1] if len(sys.argv) > 1 else "events,timing").split(",")
    if "events" in st:
        events()
    if "timing" in st:
        timing()
