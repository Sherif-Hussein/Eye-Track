"""Phase 2 screening on Preti et al. (2026) fake-identity lying data.

Stages: data audit -> frozen folds -> reproduction of a paper-style protocol -> honest baseline
(participant-disjoint nested CV) -> idea screening (ocular-only / RT-only / fusion) -> paired tests.
Participant scores per arm are cached in <out>/scores/ so reruns skip finished arms.

Usage: python -I scripts/phase2_preti.py data/preti/preti2026.xlsx results/phase2/preti
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from edd import preti as P  # noqa: E402
from edd.folds import load_folds, make_repeated_folds, save_folds  # noqa: E402
from edd.metrics import binary_metrics  # noqa: E402

XLSX, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "scores").mkdir(exist_ok=True)
DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
DERIVED.mkdir(exist_ok=True)
N_REPEATS = 3
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------- data
xl = P.load_workbook(XLSX)
fix = P.fixations(xl)
trials = P.trials_from_fixations(fix)
trials.to_csv(DERIVED / "preti_trials.csv", index=False)
subj = trials.groupby("subject").y.first()

audit = {
    "n_fixation_rows": int(len(fix)), "n_trials": int(len(trials)), "n_subjects": int(subj.size),
    "n_liars": int(subj.sum()), "n_truth": int((1 - subj).sum()),
    "label_coding": "CONDITION 0 = liar (positive class), 1 = truth-teller; inferred from the abstract: liars slower (raw U RT 3759 vs 2140 ms) and flat pupil profile",
    "trials_per_subject": trials.groupby("subject").size().value_counts().to_dict(),
    "trials_per_qt": {"C": 8, "E": 8, "U": 16},
    "fixations_per_trial": trials.n_fix.describe().round(2).to_dict(),
    "pc_sub_nan_trials": int(trials.pc_sub.isna().sum()),
    "rt_ms": trials.RT.describe().round(1).to_dict(),
    "pupil_units": "arbitrary EyeLink-style integer units (CURRENT_FIX_PUP, 1111-8849)",
    "corr_RT_nfix_trial": float(np.corrcoef(trials.RT, trials.n_fix)[0, 1]),
}
# sheet consistency
rt_cond = xl["RTs_aggregated"].groupby("SUBJECT_LABEL").CONDITION.first()
raw_cond = xl["raw data"].groupby("SUBJECT_LABEL").CONDITION.first()
pup_cond = xl["pupil_corrected"].groupby("SUBJECT_LABEL").CONDITION.first()
bl_cond = xl["blink_aggregated"].groupby("SUBJECT_LABEL").CONDITION.first()
err_cond = xl["errors"].groupby("SUBJECT_LABEL").CONDITION.first()
audit["sheet_label_mismatch_vs_raw_same_id"] = {
    "RTs_aggregated": int((rt_cond != raw_cond).sum()), "pupil_corrected": int((pup_cond != raw_cond).sum()),
    "blink_aggregated": int((bl_cond != raw_cond).sum()), "errors": int((err_cond != raw_cond).sum())}
shift = pd.Series({s: raw_cond.get(s + 1) for s in range(3, 57)})
audit["pupil_sheet_label_mismatch_vs_raw_id_plus1_(ids3-56)"] = int((pup_cond.loc[3:56] != shift).sum())
audit["subject3_label"] = {"raw": int(raw_cond[3]), "RTs_aggregated": int(rt_cond[3])}
log("audit", audit)

# ----------------------------------------------------------------------------- folds
fpath = OUT / "folds.json"
if not fpath.exists():
    folds = make_repeated_folds(subj.index.astype(str).to_numpy(), subj.to_numpy(), N_REPEATS, base_seed=2026)
    save_folds(fpath, folds, {"design": "paired participant folds (1 liar + 1 truth-teller per fold), 3 repeats",
                              "seeds": [2026, 2027, 2028], "unit": "subject (raw data SUBJECT_LABEL)",
                              "positive": "liar (CONDITION 0)"})
folds = load_folds(fpath)
log("folds", [len(f) for f in folds])

results = {"data_audit": audit, "arms": {}, "reproduction": {}, "paired_tests": {}}

# ----------------------------------------------------------------------------- reproduction (paper-style)
PUB_FEATS = [f"{m}_{q}" for m in ("RT", "blink_rate", "pupil") for q in P.QT]
SETS = {"fusion": PUB_FEATS, "ocular": [f for f in PUB_FEATS if not f.startswith("RT")],
        "rt": [f for f in PUB_FEATS if f.startswith("RT")]}
MODELS = {"LR": LogisticRegression(max_iter=5000), "SVM": SVC(), "RF": RandomForestClassifier(n_estimators=100, random_state=0),
          "NB": GaussianNB(), "Tree": DecisionTreeClassifier(random_state=0), "kNN": KNeighborsClassifier(5)}


def paper_style(df, feats, n_rep=10):
    """Non-nested repeated stratified 10-fold CV on one row per participant, default
    hyperparameters, several model families; the 'up to' number is the max over models."""
    X, y = df[feats].fillna(df[feats].median()).to_numpy(float), df.y.to_numpy(int)
    res = {}
    for name, m in MODELS.items():
        accs, aucs = [], []
        for r in range(n_rep):
            cv = StratifiedKFold(10, shuffle=True, random_state=r)
            pm = P.pipe(m)
            if isinstance(m, SVC):
                s = cross_val_predict(pm, X, y, cv=cv, method="decision_function")
            else:
                s = cross_val_predict(pm, X, y, cv=cv, method="predict_proba")[:, 1] - 0.5
            mm = binary_metrics(y, s)
            accs.append(mm["accuracy"]); aucs.append(mm["auc"])
        res[name] = {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)), "acc_max_over_seeds": float(np.max(accs)),
                     "auc_mean": float(np.mean(aucs))}
    best = max(res, key=lambda k: res[k]["acc_mean"])
    res["_best_model"] = best
    res["_up_to_acc_mean"] = res[best]["acc_mean"]
    res["_up_to_acc_any_seed"] = float(max(v["acc_max_over_seeds"] for k, v in res.items() if not k.startswith("_")))
    return res


def trial_random_split(t, feats, n_rep=3):
    """Leaky trial-level variant: trials pooled, random stratified 10-fold, trial accuracy."""
    X, y = t[feats].to_numpy(float), t.y.to_numpy(int)
    out = {}
    for name in ("LR", "RF"):
        accs = []
        for r in range(n_rep):
            s = cross_val_predict(P.pipe(MODELS[name]), X, y, cv=StratifiedKFold(10, shuffle=True, random_state=r),
                                  method="predict_proba")[:, 1] - 0.5
            accs.append(binary_metrics(y, s)["accuracy"])
        out[name] = float(np.mean(accs))
    return out


pub_raw = P.published_style_features(trials)
pub_sheets = P.published_sheets_merged(xl)
pub_raw.to_csv(DERIVED / "preti_published_style_from_raw.csv", index=False)
rep_cache = OUT / "scores" / "reproduction.json"
if rep_cache.exists():
    results["reproduction"] = json.loads(rep_cache.read_text())
else:
    rep = {}
    for dname, d in (("sheets_merged_on_SUBJECT_LABEL", pub_sheets), ("rebuilt_from_raw", pub_raw)):
        for sname, feats in SETS.items():
            rep[f"{dname}|{sname}"] = paper_style(d, feats)
            log("repro", dname, sname, rep[f"{dname}|{sname}"]["_best_model"], round(rep[f"{dname}|{sname}"]["_up_to_acc_mean"], 3))
    tfe = trials.assign(qU=(trials.qt == "U").astype(int), qE=(trials.qt == "E").astype(int))
    rep["trial_random_split_leaky|fusion"] = trial_random_split(tfe.fillna({"pc_sub": tfe.pc_sub.median()}), ["RT", "blink_any", "pc_sub", "qU", "qE"])
    rep["trial_random_split_leaky|ocular"] = trial_random_split(tfe.fillna({"pc_sub": tfe.pc_sub.median()}), ["blink_any", "pc_sub", "qU", "qE"])
    log("repro trial-leaky", rep["trial_random_split_leaky|fusion"], rep["trial_random_split_leaky|ocular"])
    results["reproduction"] = rep
    rep_cache.write_text(json.dumps(rep, indent=1))

# ----------------------------------------------------------------------------- arms
# Screening learner space: LR + RBF-SVM chosen in inner CV. Random forest costs ~0.8 s per fit on this
# shared CPU, so it is tested only as a model-family arm (I7) to stay inside the compute budget.
FAM3_RF = P.family_grid(("lr", "svm", "rf"))
FAM3 = P.family_grid(("lr", "svm"))
FAM2 = FAM3
LR_TRIAL = {"clf": [LogisticRegression(max_iter=5000)], "clf__C": [0.01, 0.1, 1.0]}
ARMS = {}


def sub_arm(name, df, feats, grid, group, modality, desc):
    ARMS[name] = dict(kind="subject", df=df, feats=feats, grid=grid, group=group, modality=modality, desc=desc)


def trial_arm(name, df, feats, rules, group, modality, desc):
    ARMS[name] = dict(kind="trial", df=df, feats=feats, grid=LR_TRIAL, rules=rules, group=group, modality=modality, desc=desc)


# Honest baselines: the paper's feature set, inner-selected among LR/SVM/RF
for mod, feats in SETS.items():
    sub_arm(f"B_{mod}", pub_raw, feats, FAM3, "baseline", mod, f"Published feature set ({mod}), rebuilt from raw, LR/SVM chosen in inner CV")
sub_arm("I7_lr_svm_rf_B_fusion", pub_raw, SETS["fusion"], FAM3_RF, "I7 model family", "fusion",
        "Published 9 features, LR/SVM/RF chosen in inner CV")
sub_arm("B_fusion_sheets", pub_sheets, SETS["fusion"], FAM3, "baseline", "fusion",
        "Published feature set from the deposited sheets merged on SUBJECT_LABEL (pupil/blink ids offset)")

# Idea 1: within-subject differential features
OC_M, RT_M = ["pc_sub", "blink_per_s"], ["RT", "logRT"]
d_oc = P.subject_features(trials, OC_M, ["level", "diff", "dev"])
d_rt = P.subject_features(trials, RT_M, ["level", "diff", "ratio", "dev"])
d_fu = d_oc.merge(d_rt.drop(columns="y"), on="subject")
for mod, d in (("ocular", d_oc), ("rt", d_rt), ("fusion", d_fu)):
    sub_arm(f"I1_diff_{mod}", d, [c for c in d.columns if c not in ("subject", "y")], FAM3, "I1 differential", mod,
            "Levels + within-subject contrasts (U-C, E-C, U-E), ratios (RT), deviance in person-SD units")
d_oc_only = P.subject_features(trials, OC_M, ["diff", "dev"])
d_rt_only = P.subject_features(trials, RT_M, ["diff", "ratio", "dev"])
d_fu_only = d_oc_only.merge(d_rt_only.drop(columns="y"), on="subject")
# Pruned to the ocular arm for compute (shared CPU: ~100 s per arm)
for mod, d in (("ocular", d_oc_only),):
    sub_arm(f"I1b_contrast_only_{mod}", d, [c for c in d.columns if c not in ("subject", "y")], FAM3, "I1 differential", mod,
            "Contrasts only (no absolute levels)")

# Idea 2: per-subject normalisation on the subject's own control (neutral) trials, subject level
d2_oc = P.subject_features(trials, OC_M, ["devc"])
d2_rt = P.subject_features(trials, RT_M, ["devc"])
d2_fu = d2_oc.merge(d2_rt.drop(columns="y"), on="subject")
for mod, d in (("ocular", d2_oc), ("rt", d2_rt), ("fusion", d2_fu)):
    sub_arm(f"I2_neutralz_subject_{mod}", d, [c for c in d.columns if c not in ("subject", "y")], FAM3, "I2 neutral norm", mod,
            "(E or U mean - C mean) / SD of the person's own control trials")

# Idea 3: trial-level model on control-normalised trials + inner-tuned aggregation rule
tz = P.per_subject_neutral_z(trials, ["logRT", "pc_sub", "blink_any", "pc_div"])
tz = tz.assign(qU=(tz.qt == "U").astype(int))
tz_eu = tz[tz.qt != "C"].reset_index(drop=True)
TZ = {"ocular": ["pc_sub_z", "blink_any_z", "qU"], "rt": ["logRT_z", "qU"], "fusion": ["logRT_z", "pc_sub_z", "blink_any_z", "qU"]}
for mod in ("ocular", "fusion"):
    trial_arm(f"I3_trial_neutralz_aggtuned_{mod}", tz_eu, TZ[mod], P.AGG_RULES, "I3 trial+aggregation", mod,
              "Trial LR on E/U trials z-scored by own control trials; aggregation rule (mean/median/trim20/vote) tuned in inner folds")
traw = trials.assign(qU=(trials.qt == "U").astype(int), qE=(trials.qt == "E").astype(int))
TR = {"ocular": ["pc_sub", "blink_any", "qU", "qE"], "rt": ["logRT", "qU", "qE"], "fusion": ["logRT", "pc_sub", "blink_any", "qU", "qE"]}
for mod in ("ocular", "rt", "fusion"):
    trial_arm(f"I3_trial_raw_aggtuned_{mod}", traw, TR[mod], P.AGG_RULES, "I3 trial+aggregation", mod,
              "Trial LR on all trials, no per-subject normalisation; aggregation tuned")
trial_arm("I3_trial_raw_mean_fusion", traw, TR["fusion"], ("mean",), "I3 trial+aggregation", "fusion",
          "Same as I3_trial_raw_aggtuned_fusion with aggregation fixed to mean (isolates the aggregation idea)")

# Idea 4: pupil preprocessing ablation (pupil-only, subject-level, levels+contrasts+deviance)
for pc in ("pc_sub", "pc_div", "pc_peak", "pc_clean", "pc_none"):
    d = P.subject_features(trials, [pc], ["level", "diff", "dev"])
    sub_arm(f"I4_pupil_{pc}", d, [c for c in d.columns if c not in ("subject", "y")], FAM2, "I4 pupil preprocessing", "pupil",
            f"Pupil-only, {pc}, published rejection")
for rule in ("minfix3", "z3"):
    d = P.subject_features(P.apply_rejection(trials, rule), ["pc_sub"], ["level", "diff", "dev"])
    sub_arm(f"I4_pupil_pc_sub_{rule}", d, [c for c in d.columns if c not in ("subject", "y")], FAM2, "I4 pupil preprocessing", "pupil",
            f"Pupil-only, pc_sub, rejection={rule}")
d = P.subject_features(trials, ["blink_per_s"], ["level", "diff", "dev"])
sub_arm("I4_blink_only", d, [c for c in d.columns if c not in ("subject", "y")], FAM2, "I4 pupil preprocessing", "blink", "Blink-only reference")

# Idea 5: eye-movement (fixation) measures; n_fix is an eye-tracker measure but tracks RT (r = .89)
d5 = P.subject_features(trials, ["pc_sub", "blink_per_s", "n_fix"], ["level", "diff", "dev"])
sub_arm("I5_ocular_plus_fixcount", d5, [c for c in d5.columns if c not in ("subject", "y")], FAM3, "I5 fixation measures", "ocular+fix",
        "Ocular incl. fixation count per trial (RT proxy)")
d5b = P.subject_features(trials, ["pc_sub", "blink_per_s", "fix_per_s"], ["level", "diff", "dev"])
sub_arm("I5_ocular_plus_fixrate", d5b, [c for c in d5b.columns if c not in ("subject", "y")], FAM3, "I5 fixation measures", "ocular+fix",
        "Ocular incl. fixations per second (RT-normalised)")
# Planned arm I6 (ANSWER_LABEL vs KEYPRESS share) was dropped unrun: it is outside the brief, the column is
# undocumented, and its mismatch rate is 0-6% in subjects 1-7 vs 31-72% in all others (an export artefact).

# Idea 7: model families with calibration, on the baseline and I1 fusion/ocular sets
for fam in ("lr_en", "lgbm"):
    g = P.family_grid((fam,))
    for mod, d in (("fusion", d_fu), ("rt", d_rt), ("ocular", d_oc)):
        sub_arm(f"I7_{fam}_I1_{mod}", d, [c for c in d.columns if c not in ("subject", "y")], g, "I7 model family", mod, f"{fam} on I1 {mod} features")


def run_arm(name, a):
    cache = OUT / "scores" / f"{name}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"unit": str})
    t0 = time.time()
    if a["kind"] == "subject":
        parts = P.run_subject_level(a["df"], a["feats"], folds, a["grid"], n_jobs=2)
    else:
        parts = P.run_trial_level(a["df"], a["feats"], folds, a["grid"], rules=a["rules"], n_jobs=2)
    parts.to_csv(cache, index=False)
    log(f"ran {name} in {time.time() - t0:.1f}s")
    return parts


avgs = {}
only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
for name, a in ARMS.items():
    if only and name not in only:
        continue
    parts = run_arm(name, a)
    # bootstrap CIs (slow on this shared CPU) only for baselines here; candidates get theirs in phase2_preti_final.py
    res, avg = P.evaluate_participants(parts, n_boot=2000, ci=name.startswith("B_"))
    res.update({k: a[k] for k in ("group", "modality", "desc", "kind")})
    res["n_features"] = len(a["feats"])
    res["features"] = a["feats"]
    res["chosen_params"] = parts.groupby(["repeat", "fold"]).params.first().value_counts().head(8).to_dict()
    results["arms"][name] = res
    avgs[name] = avg
    pm = res["participant"]
    log(f"{name:45s} AUC {pm['auc']['mean']:.3f}±{pm['auc']['sd']:.3f}  ACC {pm['accuracy']['mean']:.3f}  "
        f"EER {pm['eer']['mean']:.3f}  HTER {pm['hter']['mean']:.3f}")

# ----------------------------------------------------------------------------- paired tests vs honest baseline
BASE_OF = {"ocular": "B_ocular", "rt": "B_rt", "fusion": "B_fusion", "ocular+fix": "B_ocular", "pupil": "I4_pupil_pc_sub"}
pv = {}
for name, a in ARMS.items():
    if name.startswith("B_") or name not in avgs or a["modality"] not in BASE_OF:
        continue
    b = BASE_OF[a["modality"]]
    if b not in avgs or b == name:
        continue
    t = P.paired_auc_diff(avgs[name], avgs[b])
    t["vs"] = b
    results["paired_tests"][name] = t
    pv[name] = t["p_boot"]
for k, v in P.holm(pv).items():
    results["paired_tests"][k]["p_holm"] = v
# fusion vs ocular and RT within the baseline and the leading idea (does ocular add to RT?)
for x, y in (("B_fusion", "B_rt"), ("I1_diff_fusion", "I1_diff_rt"), ("I1_diff_fusion", "I1_diff_ocular"),
             ("I3_trial_raw_aggtuned_fusion", "I3_trial_raw_mean_fusion"), ("I7_lr_en_I1_fusion", "I7_lr_en_I1_rt"),
             ("I7_lgbm_I1_fusion", "I7_lgbm_I1_rt"), ("I7_lr_en_I1_rt", "I1_diff_rt"), ("I7_lr_en_I1_fusion", "I1_diff_fusion"),
             ("I3_trial_neutralz_aggtuned_fusion", "I3_trial_raw_aggtuned_fusion"), ("B_fusion_sheets", "B_fusion")):
    if x in avgs and y in avgs:
        results["paired_tests"][f"{x}__minus__{y}"] = P.paired_auc_diff(avgs[x], avgs[y])

prev = OUT / "results.json"
if prev.exists() and only:
    old = json.loads(prev.read_text())
    old["arms"].update(results["arms"])
    old["paired_tests"].update(results["paired_tests"])
    results = {**old, "data_audit": audit}
prev.write_text(json.dumps(results, indent=1, default=str))
log("done")
