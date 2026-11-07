"""Efficiency experiment: original pandas feature extraction vs the vectorised NumPy one (edd.preti_fast).

Steps, in this order, all in one session on one machine:
  1. equivalence of the features, per participant, against the original code (numpy.allclose, rtol 1e-9, atol 0),
     for B_fusion, I7_lgbm_I1_rt, I7_lr_en_I1_fusion and the trial matrix of the eye-only arm I3
     (also against the full-table features that the cross-validation used);
  2. per-participant latency / throughput / peak memory, original vs fast, single thread (edd.efficiency);
  3. out-of-fold AUC replay of the frozen folds with the fast features, compared with the saved scores;
  4. writes efficiency_fast.json and efficiency_fast.md.
The speed-up is only claimed in the outputs when steps 1 and 3 pass.

Usage: python -I scripts/phase2_preti_fast_efficiency.py data/preti/preti2026.xlsx results/phase2/preti [--skip-auc]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from edd import efficiency as eff  # noqa: E402
from edd import preti as P  # noqa: E402
from edd import preti_fast as F  # noqa: E402
from edd.folds import load_folds  # noqa: E402

XLSX, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SKIP_AUC = "--skip-auc" in sys.argv
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


res = json.loads((OUT / "results.json").read_text())
cands = json.loads((OUT / "final.json").read_text())["candidates"]
fix = P.fixations(P.load_workbook(XLSX))
trials = P.trials_from_fixations(fix)
A = F.arrays_from_frame(fix)
subjects = sorted(A)
ARMS = {"published_reimpl_B_fusion": "B_fusion", "best_ocular": cands["ocular"], "best_rt": cands["rt"], "best_fusion": cands["fusion"]}
SUBJ_ARMS = {k: v for k, v in ARMS.items() if not v.startswith("I3")}
OCULAR = ARMS["best_ocular"]
OC_FEATS = res["arms"][OCULAR]["features"]
assert OC_FEATS == ["pc_sub", "blink_any", "qU", "qE"], OC_FEATS


def with_q(t):
    return t.assign(qU=(t.qt == "U").astype(int), qE=(t.qt == "E").astype(int))


# ----------------------------------------------------------------------------- 1. equivalence
def cmp(orig, fast):
    orig, fast = np.asarray(orig, float), np.asarray(fast, float)
    assert orig.shape == fast.shape, (orig.shape, fast.shape)
    same_nan = bool((np.isnan(orig) == np.isnan(fast)).all())
    both = ~np.isnan(orig) & ~np.isnan(fast)
    diff = float(np.abs(orig[both] - fast[both]).max()) if both.any() else 0.0
    return same_nan, diff, bool(np.allclose(orig, fast, rtol=1e-9, atol=0, equal_nan=True))


eq = {}
for label, arm in SUBJ_ARMS.items():
    feats = res["arms"][arm]["features"]
    assert F.feature_names(arm) == feats, arm
    eq[arm] = {"n_features": len(feats), "per_participant": {"max_abs_diff": 0.0, "all_allclose": True, "nan_pattern_equal": True},
               "full_table": {}}
eq[OCULAR] = {"n_features": 4, "per_participant": {"max_abs_diff": 0.0, "all_allclose": True, "nan_pattern_equal": True}, "full_table": {}}
for s in subjects:  # original per-participant path = what is timed: pandas on this participant's fixation rows only
    fs = fix[fix.subject == s]
    ts = P.trials_from_fixations(fs)
    for arm in [*SUBJ_ARMS.values(), OCULAR]:
        if arm == OCULAR:
            o, x = with_q(ts)[OC_FEATS].to_numpy(float), F.ocular_trial_matrix(A[s])
        else:
            o, x = P.arm_subject_table(ts, arm)[res["arms"][arm]["features"]].to_numpy(float), F.FAST_FEATURES[arm](A[s])
        nan_ok, d, ac = cmp(o, x)
        e = eq[arm]["per_participant"]
        e["max_abs_diff"], e["all_allclose"], e["nan_pattern_equal"] = max(e["max_abs_diff"], d), e["all_allclose"] and ac, e["nan_pattern_equal"] and nan_ok
for arm in [*SUBJ_ARMS.values(), OCULAR]:  # full-table features (the CV used these): all participants at once
    if arm == OCULAR:
        o = with_q(trials)[OC_FEATS].to_numpy(float)
        x = np.vstack([F.ocular_trial_matrix(A[s]) for s in subjects])  # trials are sorted by subject, trial index
    else:
        tab = P.arm_subject_table(trials, arm)
        assert list(tab.subject) == subjects
        o, x = tab[res["arms"][arm]["features"]].to_numpy(float), np.vstack([F.FAST_FEATURES[arm](A[s]) for s in subjects])
    nan_ok, d, ac = cmp(o, x)
    eq[arm]["full_table"] = {"max_abs_diff": d, "allclose": ac, "nan_pattern_equal": nan_ok, "shape": list(o.shape)}
    log("equivalence", arm, eq[arm])
EQUIV_OK = all(v["per_participant"]["all_allclose"] and v["per_participant"]["nan_pattern_equal"] and v["full_table"]["allclose"]
               and v["full_table"]["nan_pattern_equal"] for v in eq.values())
log("EQUIVALENCE", "PASS" if EQUIV_OK else "FAIL")


# ----------------------------------------------------------------------------- 2. benchmark
def fit_subject(arm):
    feats = res["arms"][arm]["features"]
    d = P.arm_subject_table(trials, arm)
    gs = GridSearchCV(P.pipe(LogisticRegression()), P.arm_grid(arm), cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
    gs.fit(d[feats].to_numpy(float), d.y.to_numpy(int))
    return gs.best_estimator_, feats, str({k: (type(v).__name__ if k == "clf" else v) for k, v in gs.best_params_.items()})


def fit_ocular():
    d = with_q(trials)
    return P.pipe(LogisticRegression(C=0.1, max_iter=5000)).fit(d[OC_FEATS].to_numpy(float), d.y.to_numpy(int))


one = fix[fix.subject == subjects[0]]
a1 = A[subjects[0]]
out = {"hardware": eff.hardware_info(),
       "decision_units": {"participant_fixation_rows": int(len(one)), "participant_trials": int(one.trial.nunique())},
       "protocol": {"threads": 1, "original": "exactly phase2_preti_efficiency.py (100 runs extraction/end-to-end, 500 inference, 2 s throughput)",
                    "fast": "same functions, but 2000 runs for extraction and end-to-end (the calls are 100x shorter), 500 inference, 2 s throughput",
                    "input_original": "pandas DataFrame of the participant's fixation rows",
                    "input_fast": "pre-sorted NumPy columns of the participant (FixArrays); ingestion from the DataFrame is timed separately as "
                                  "'fast_from_dataframe' (it is data loading, done once per recording)",
                    "model": "the same fitted estimator is used by both pipelines, so inference-only is identical code"},
       "equivalence": eq, "methods": {}}
log("benchmark start")
for label, arm in ARMS.items():
    if arm == OCULAR:
        est = fit_ocular()
        chosen = "LR C=0.1, mean aggregation"
        ext_o = lambda f: with_q(P.trials_from_fixations(f))[OC_FEATS].to_numpy(float)  # noqa: E731
        dec_o = lambda f: float(P.signed_score(est, ext_o(f)).mean())  # noqa: E731
        ext_f, dec_f = F.ocular_trial_matrix, (lambda a: float(P.signed_score(est, F.ocular_trial_matrix(a)).mean()))  # noqa: E731
    else:
        est, feats, chosen = fit_subject(arm)
        ext_o = lambda f, arm=arm, feats=feats: P.arm_subject_table(P.trials_from_fixations(f), arm)[feats].to_numpy(float)  # noqa: E731
        dec_o = lambda f, ext_o=ext_o: P.signed_score(est, ext_o(f))  # noqa: E731
        ext_f = F.FAST_FEATURES[arm]
        dec_f = lambda a, ext_f=ext_f: P.signed_score(est, ext_f(a))  # noqa: E731
    X1 = ext_o(one)
    # decisions identical on every participant (end-to-end equivalence)
    dmax = max(float(np.max(np.abs(np.asarray(dec_o(fix[fix.subject == s]), float) - np.asarray(dec_f(A[s]), float)))) for s in subjects)
    r = {"arm": arm, "chosen": chosen, "max_abs_decision_diff_all_participants": dmax}
    for i in range(30):  # warm both code paths before tracemalloc
        dec_f(a1), dec_o(one)
    for tag, ext, dec, inp, n in (("original", ext_o, dec_o, one, 100), ("fast", ext_f, dec_f, a1, 2000)):
        tp = eff.throughput_per_s(lambda: dec(inp), 1, seconds=2.0)
        r[tag] = {"feature_extraction_ms_per_participant": eff.latency_ms(lambda: ext(inp), n_runs=n, warmup=5 if tag == "original" else 100),
                  "inference_only_ms_per_participant": eff.latency_ms(lambda: P.signed_score(est, X1), n_runs=500, warmup=20),
                  "end_to_end_ms_per_participant": eff.latency_ms(lambda: dec(inp), n_runs=n, warmup=5 if tag == "original" else 100),
                  "peak_memory_bytes_end_to_end": eff.peak_memory_bytes(lambda: dec(inp)),
                  "throughput_participants_per_s": tp, "throughput_fixation_rows_per_s": tp * len(one)}
    r["fast_from_dataframe"] = {
        "ingestion_ms": eff.latency_ms(lambda: F.arrays_from_frame(one), n_runs=500, warmup=20),
        "end_to_end_ms_per_participant": eff.latency_ms(lambda: dec_f(F.arrays_from_frame(one)[subjects[0]]), n_runs=500, warmup=20),
        "peak_memory_bytes_end_to_end": eff.peak_memory_bytes(lambda: dec_f(F.arrays_from_frame(one)[subjects[0]]))}
    o, f_ = r["original"], r["fast"]
    r["speedup_p50"] = {k: o[k]["p50"] / f_[k]["p50"] for k in ("feature_extraction_ms_per_participant", "end_to_end_ms_per_participant")}
    out["methods"][label] = r
    log(label, arm, "extract p50 orig/fast", round(o["feature_extraction_ms_per_participant"]["p50"], 3), round(f_["feature_extraction_ms_per_participant"]["p50"], 4),
        "e2e", round(o["end_to_end_ms_per_participant"]["p50"], 3), round(f_["end_to_end_ms_per_participant"]["p50"], 4))
log("benchmark done")

# ----------------------------------------------------------------------------- 3. OOF AUC replay on the frozen folds
folds = load_folds(OUT / "folds.json")
auc = {}
if not SKIP_AUC:
    for label in ("best_rt", "best_fusion", "published_reimpl_B_fusion"):
        arm = ARMS[label]
        feats = res["arms"][arm]["features"]
        X = np.vstack([F.FAST_FEATURES[arm](A[s]) for s in subjects])
        df = pd.DataFrame(X, columns=feats)
        df.insert(0, "y", [A[s].y for s in subjects])
        df.insert(0, "subject", subjects)
        t0 = time.time()
        new = P.run_subject_level(df, feats, folds, P.arm_grid(arm), n_jobs=1)
        old = pd.read_csv(OUT / "scores" / f"{arm}.csv", dtype={"unit": str})
        k = ["repeat", "fold", "unit"]
        m = old.merge(new, on=k, suffixes=("_old", "_new"), validate="1:1")
        assert len(m) == len(old) == len(new)

        def per_rep(d, c):
            return [float(roc_auc_score(g.y, g[c])) for _, g in d.groupby("repeat")]
        a_old, a_new = per_rep(m.rename(columns={"y_old": "y"}), "score_old"), per_rep(m.rename(columns={"y_old": "y"}), "score_new")
        sd = lambda v: float(np.std(v, ddof=1))  # noqa: E731
        saved = res["arms"][arm]["participant"]["auc"]
        auc[arm] = {"auc_per_repeat_saved": a_old, "auc_per_repeat_fast": a_new,
                    "auc_mean_saved_scores": float(np.mean(a_old)), "auc_mean_fast": float(np.mean(a_new)),
                    "auc_mean_in_results_json": saved["mean"], "auc_sd_in_results_json": saved["sd"], "auc_sd_fast_ddof1": sd(a_new),
                    "max_abs_score_diff": float(np.abs(m.score_old - m.score_new).max()),
                    "max_abs_prob_diff": float(np.abs(m.prob_old - m.prob_new).max()),
                    "chosen_params_identical": bool((m.params_old == m.params_new).all()),
                    "auc_identical_per_repeat": bool(np.array_equal(a_old, a_new)),
                    "auc_equal_within_1e-12": bool(np.allclose(a_old, a_new, rtol=0, atol=1e-12)),
                    "n_oof_rows": int(len(m)), "replay_seconds": time.time() - t0}
        log("AUC replay", arm, auc[arm])
AUC_OK = (not SKIP_AUC) and all(v["auc_equal_within_1e-12"] and v["chosen_params_identical"] and abs(v["auc_mean_fast"] - v["auc_mean_in_results_json"]) < 1e-9
                                for v in auc.values())
out["oof_auc_replay"] = auc
out["verdict"] = {"features_equivalent": EQUIV_OK, "oof_auc_unchanged": bool(AUC_OK) if not SKIP_AUC else None,
                  "speedup_claimed": bool(EQUIV_OK and AUC_OK)}
out["note"] = ("Timings of both pipelines were taken in the same session, single thread, on a CPU that may be shared with other jobs "
               "(see the markdown file for the load observed), so all values are upper bounds.")
(OUT / "efficiency_fast.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- 4. markdown
def pp(d):
    return f"{d['p50']:.3f} / {d['p95']:.3f}"


names = {"published_reimpl_B_fusion": "Honest baseline `B_fusion`", "best_ocular": "Eye-only `I3_trial_raw_aggtuned_ocular`",
         "best_rt": "RT only `I7_lgbm_I1_rt`", "best_fusion": "Fusion `I7_lr_en_I1_fusion`"}
L = ["# Preti: vectorised feature extraction, before / after", "",
     f"Generated by `scripts/phase2_preti_fast_efficiency.py` (`src/edd/preti_fast.py`). Single thread, {out['hardware']['platform']}, "
     f"Python {out['hardware']['python']}. One participant = {out['decision_units']['participant_fixation_rows']} fixation rows in "
     f"{out['decision_units']['participant_trials']} trials. Latencies are p50 / p95 in ms; original and fast were measured in the same session. "
     "Original = pandas groupby path from the fixation DataFrame; fast = NumPy on pre-sorted arrays (same fitted model, same sklearn inference). ", "",
     f"**Verdict.** Features equivalent: {'yes' if EQUIV_OK else 'NO'}. Out-of-fold AUC unchanged: "
     f"{'not run' if SKIP_AUC else ('yes' if AUC_OK else 'NO')}. Speed-up claimed: {'yes' if out['verdict']['speedup_claimed'] else 'NO'}.", "",
     "| Method | Stage | Original | Fast | Speed-up (p50) |", "|---|---|---|---|---|"]
for k, r in out["methods"].items():
    for stage, key in (("feature extraction", "feature_extraction_ms_per_participant"), ("inference only", "inference_only_ms_per_participant"),
                       ("end to end", "end_to_end_ms_per_participant")):
        o, f_ = r["original"][key], r["fast"][key]
        L.append(f"| {names[k]} | {stage} | {pp(o)} | {pp(f_)} | {o['p50'] / f_['p50']:.1f}x |")
    o, f_, fd = r["original"], r["fast"], r["fast_from_dataframe"]
    L.append(f"| {names[k]} | participants/s | {o['throughput_participants_per_s']:.1f} | {f_['throughput_participants_per_s']:.0f} | "
             f"{f_['throughput_participants_per_s'] / o['throughput_participants_per_s']:.1f}x |")
    L.append(f"| {names[k]} | peak memory (B) | {o['peak_memory_bytes_end_to_end']:,} | {f_['peak_memory_bytes_end_to_end']:,} | "
             f"{o['peak_memory_bytes_end_to_end'] / f_['peak_memory_bytes_end_to_end']:.1f}x |")
    L.append(f"| {names[k]} | fast incl. DataFrame ingestion, end to end | - | {pp(fd['end_to_end_ms_per_participant'])} "
             f"(ingestion alone {fd['ingestion_ms']['p50']:.3f}) | {o['end_to_end_ms_per_participant']['p50'] / fd['end_to_end_ms_per_participant']['p50']:.1f}x |")
L += ["", "## Equivalence (fast vs original features)", "",
      "`numpy.allclose(rtol=1e-9, atol=0)` and identical NaN pattern, on all 57 participants, original code run per participant (the timed path) "
      "and on the full table the cross-validation used.", "",
      "| Arm | Features | Max abs diff, per participant | Max abs diff, full table | allclose |", "|---|---|---|---|---|"]
for arm, v in eq.items():
    L.append(f"| `{arm}` | {v['n_features']} | {v['per_participant']['max_abs_diff']:.2e} | {v['full_table']['max_abs_diff']:.2e} | "
             f"{'yes' if v['per_participant']['all_allclose'] and v['full_table']['allclose'] else 'NO'} |")
if auc:
    L += ["", "## Out-of-fold AUC with the fast features (frozen folds, 3 repeats, nested CV, n_jobs = 1)", "",
          "| Arm | AUC saved (mean over repeats) | AUC fast | Per-repeat AUC identical | Max abs score diff | Same chosen params |", "|---|---|---|---|---|---|"]
    for arm, v in auc.items():
        L.append(f"| `{arm}` | {v['auc_mean_saved_scores']:.6f} | {v['auc_mean_fast']:.6f} | "
                 f"{'yes' if v['auc_equal_within_1e-12'] else 'NO'} | {v['max_abs_score_diff']:.2e} | {'yes' if v['chosen_params_identical'] else 'NO'} |")
(OUT / "efficiency_fast.md").write_text("\n".join(L) + "\n", encoding="utf-8")
log("done", out["verdict"])
