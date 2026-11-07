"""Deployment cost of the published-method re-implementation and the Phase 2 candidates on Preti data,
single-threaded CPU, from the rawest data available (fixation-level rows of one participant).
Candidates are read from <out>/final.json, written by phase2_preti_final.py.

Usage: python -I scripts/phase2_preti_efficiency.py data/preti/preti2026.xlsx results/phase2/preti
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from edd import efficiency as eff  # noqa: E402
from edd import preti as P  # noqa: E402

XLSX, OUT = Path(sys.argv[1]), Path(sys.argv[2])
res = json.loads((OUT / "results.json").read_text())
cands = json.loads((OUT / "final.json").read_text())["candidates"]
fix = P.fixations(P.load_workbook(XLSX))
trials = P.trials_from_fixations(fix)


def n_params(est):
    clf, sc = est.named_steps["clf"], est.named_steps["sc"]
    pre = sc.mean_.size + sc.scale_.size + est.named_steps["imp"].statistics_.size
    if type(clf).__name__ == "LogisticRegression":
        return pre + clf.coef_.size + clf.intercept_.size
    if type(clf).__name__ == "SVC":
        return pre + eff.svm_parameter_count(clf, clf.support_vectors_.shape[1])
    if type(clf).__name__ == "LGBMClassifier":
        return pre + int(clf.booster_.trees_to_dataframe().shape[0] * 4)  # feature, threshold, value, child link per node
    raise ValueError(type(clf).__name__)


def subject_spec(arm):
    feats = res["arms"][arm]["features"]
    d = P.arm_subject_table(trials, arm)
    gs = GridSearchCV(P.pipe(LogisticRegression()), P.arm_grid(arm), cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
    gs.fit(d[feats].to_numpy(float), d.y.to_numpy(int))
    est = gs.best_estimator_

    def extract(f):
        return P.arm_subject_table(P.trials_from_fixations(f), arm)[feats].to_numpy(float)

    return est, feats, extract, lambda f: P.signed_score(est, extract(f)), \
        str({k: (type(v).__name__ if k == "clf" else v) for k, v in gs.best_params_.items()})


def trial_spec(arm):
    feats = res["arms"][arm]["features"]

    def table(t):
        t = t.assign(qU=(t.qt == "U").astype(int), qE=(t.qt == "E").astype(int))
        if "neutralz" in arm:
            t = P.per_subject_neutral_z(t, ["logRT", "pc_sub", "blink_any"])
            t = t[t.qt != "C"]
        return t

    d = table(trials)
    est = P.pipe(LogisticRegression(C=0.1, max_iter=5000)).fit(d[feats].to_numpy(float), d.y.to_numpy(int))

    def extract(f):
        return table(P.trials_from_fixations(f))[feats].to_numpy(float)

    return est, feats, extract, lambda f: float(P.signed_score(est, extract(f)).mean()), "LR C=0.1, mean aggregation"


one = fix[fix.subject == fix.subject.unique()[0]]
one_trial = one[one.trial == one.trial.iloc[0]]
out = {"hardware": eff.hardware_info(), "decision_units": {"participant_fixation_rows": int(len(one)), "participant_trials": int(one.trial.nunique())},
       "note": "Fixation-level export, so sensor samples/s and real-time factor cannot be computed; throughput is per "
               "fixation row. Timings were taken on a CPU shared with other jobs, so treat them as upper bounds.", "methods": {}}
arms = {"published_reimpl_B_fusion": "B_fusion", "best_ocular": cands["ocular"], "best_rt": cands["rt"], "best_fusion": cands["fusion"]}
for label, arm in arms.items():
    est, feats, extract, decide, chosen = trial_spec(arm) if arm.startswith("I3") else subject_spec(arm)
    X1 = extract(one)
    r = {"arm": arm, "chosen": chosen, "n_features": len(feats),
         "model_size_bytes": eff.model_size_bytes(est), "stored_parameters": int(n_params(est)),
         "feature_extraction_ms_per_participant": eff.latency_ms(lambda: extract(one), n_runs=100, warmup=5),
         "inference_only_ms_per_participant": eff.latency_ms(lambda: P.signed_score(est, X1), n_runs=500, warmup=20),
         "end_to_end_ms_per_participant": eff.latency_ms(lambda: decide(one), n_runs=100, warmup=5),
         "trial_feature_extraction_ms": eff.latency_ms(lambda: P.trials_from_fixations(one_trial), n_runs=200, warmup=10),
         "peak_memory_bytes_end_to_end": eff.peak_memory_bytes(lambda: decide(one))}
    tp = eff.throughput_per_s(lambda: decide(one), 1, seconds=2.0)
    r["throughput_participants_per_s"] = tp
    r["throughput_fixation_rows_per_s"] = tp * len(one)
    out["methods"][label] = r
    print(label, arm, {k: (round(v["p50"], 3) if isinstance(v, dict) else v) for k, v in r.items()}, flush=True)
(OUT / "efficiency.json").write_text(json.dumps(out, indent=1))
