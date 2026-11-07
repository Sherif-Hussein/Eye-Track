"""Final stage of the Preti track: best ocular-only / RT-only / fusion candidates chosen from results.json,
bootstrap CIs, pipeline-level permutation tests, subject-3 label sensitivity and mechanism evidence.

A permutation shuffles labels between participants, rebuilds paired folds for the shuffled labels (so the fold
design matches the real run) and reruns the full nested pipeline for one repeat.

Usage: python -I scripts/phase2_preti_final.py data/preti/preti2026.xlsx results/phase2/preti --n-perm 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from edd import preti as P  # noqa: E402
from edd.folds import load_folds, paired_participant_folds  # noqa: E402
from edd.metrics import binary_metrics, bootstrap_ci, eer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("xlsx")
ap.add_argument("out")
ap.add_argument("--n-perm", type=int, default=40)
ap.add_argument("--perm-budget-s", type=float, default=900, help="wall-clock cap per candidate")
A = ap.parse_args()
OUT = Path(A.out)
res = json.loads((OUT / "results.json").read_text())
trials = P.trials_from_fixations(P.fixations(P.load_workbook(A.xlsx)))


def m_auc(n):
    return res["arms"][n]["participant"]["auc"]["mean"]


def best(mods):
    pool = [n for n, a in res["arms"].items() if a["modality"] in mods and not n.startswith("B_")]
    return max(pool, key=m_auc)


def trial_table(name):
    t = trials.assign(qU=(trials.qt == "U").astype(int), qE=(trials.qt == "E").astype(int))
    if "neutralz" in name:
        t = P.per_subject_neutral_z(t, ["logRT", "pc_sub", "blink_any"])
        t = t[t.qt != "C"].reset_index(drop=True)
    return t


cands = {"ocular": best(("ocular", "pupil", "blink")), "rt": best(("rt",)), "fusion": best(("fusion",)),
         "ocular_plus_fixation": best(("ocular+fix",))}
print(cands, flush=True)
final = {"candidates": cands, "selection_note": "Selected by highest mean participant AUC over the arms tried; "
         "CIs and permutation p-values do not account for this selection.", "by_arm": {}}


def avg_scores(name):
    parts = pd.read_csv(OUT / "scores" / f"{name}.csv", dtype={"unit": str})
    return parts.groupby("unit").agg(y=("y", "first"), score=("score", "mean"))


# CIs + subject-3 sensitivity (scoring only; the models were trained with the raw label)
for name in sorted({*cands.values(), "B_fusion", "B_ocular", "B_rt"}):
    a = avg_scores(name)
    r = {"avg_score_metrics": binary_metrics(a.y, a.score),
         "ci95": bootstrap_ci(a.y.to_numpy(), a.score.to_numpy(), n_boot=2000)}
    d = a.drop(index="3")
    r["auc_drop_subject3"] = binary_metrics(d.y, d.score)["auc"]
    f = a.copy()
    f.loc["3", "y"] = 1 - f.loc["3", "y"]
    r["auc_flip_subject3"] = binary_metrics(f.y, f.score)["auc"]
    final["by_arm"][name] = r
    print(name, {k: round(v, 3) for k, v in r["avg_score_metrics"].items() if k in ("auc", "accuracy", "eer", "hter")}, flush=True)
(OUT / "final.json").write_text(json.dumps(final, indent=1))


# Validity of the ocular candidate: trial-level blink_any and pc_sub are not duration-normalised, and liars'
# trials last longer, so an "ocular" score can be a proxy for RT. These checks are diagnostics, not extra ideas.
folds = load_folds(OUT / "folds.json")
LR_GRID = {"clf": [LogisticRegression(max_iter=5000)], "clf__C": [0.01, 0.1, 1.0]}
raw_t = trial_table("raw")
DIAG = {"D_trial_pupil_only": ["pc_sub", "qU", "qE"], "D_trial_blink_any_only": ["blink_any", "qU", "qE"],
        "D_trial_pupil_blinkrate": ["pc_sub", "blink_per_s", "qU", "qE"]}
val = {"trial_corr_with_logRT": {}, "arms": {}}
for c in ("pc_sub", "blink_any", "blink_per_s", "n_fix"):
    x = raw_t[["subject", c, "logRT"]].dropna()
    w = x[[c, "logRT"]] - x.groupby("subject")[[c, "logRT"]].transform("mean")
    val["trial_corr_with_logRT"][c] = {"pooled": float(x[c].corr(x.logRT)), "within_person": float(w[c].corr(w.logRT))}
mean_logrt = raw_t.groupby("subject").logRT.mean()
mean_logrt.index = mean_logrt.index.astype(str)


def rt_dependence(a):
    """How much of a participant score is explained by the person's mean log RT."""
    lr = mean_logrt.loc[a.index].to_numpy()
    y, s = a.y.to_numpy(int), a.score.to_numpy()
    resid = s - np.polyval(np.polyfit(lr, s, 1), lr)
    return {"auc": float(roc_auc_score(y, s)), "spearman_score_vs_meanlogRT": float(spearmanr(s, lr)[0]),
            "spearman_within_liars": float(spearmanr(s[y == 1], lr[y == 1])[0]),
            "spearman_within_truth": float(spearmanr(s[y == 0], lr[y == 0])[0]),
            "auc_after_regressing_out_meanlogRT": float(roc_auc_score(y, resid))}


for name, feats in DIAG.items():
    cache = OUT / "scores" / f"{name}.csv"
    if not cache.exists():
        P.run_trial_level(raw_t, feats, folds, LR_GRID, n_jobs=2).to_csv(cache, index=False)
    parts = pd.read_csv(cache, dtype={"unit": str})
    r, _ = P.evaluate_participants(parts, ci=False)
    res["arms"][name] = {**r, "group": "ocular validity check", "modality": "diagnostic", "kind": "trial", "features": feats,
                         "desc": "Trial LR, aggregation tuned; ablation of the ocular trial-level arm"}
    val["arms"][name] = {"participant": r["participant"], **rt_dependence(avg_scores(name))}
    print(name, round(r["participant"]["auc"]["mean"], 3), flush=True)
for name in sorted({cands["ocular"], cands["rt"], cands["fusion"], "B_ocular", "I1_diff_ocular", "I2_neutralz_subject_ocular",
                    "I3_trial_neutralz_aggtuned_ocular", "I3_trial_raw_aggtuned_ocular", "I5_ocular_plus_fixrate", "I5_ocular_plus_fixcount"}):
    val["arms"][name] = {"participant": res["arms"][name]["participant"], **rt_dependence(avg_scores(name))}
final["ocular_validity"] = val
(OUT / "final.json").write_text(json.dumps(final, indent=1))
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: v for k, v in val["arms"].items()}, default=lambda o: None)[:100], flush=True)


# Permutation tests: ocular matters most, so it runs first
def one_perm(name, seed):
    rng = np.random.default_rng(seed)
    d = trial_table(name) if name.startswith("I3") else P.arm_subject_table(trials, name)
    per = d.groupby("subject").y.first()
    lab = pd.Series(rng.permutation(per.to_numpy()), index=per.index)
    d = d.assign(y=d.subject.map(lab).astype(int))
    fl = [paired_participant_folds(lab.index.astype(str).to_numpy(), lab.to_numpy(), 5000 + seed)]
    feats = res["arms"][name]["features"]
    if name.startswith("I3"):
        rules = ("mean",) if "_mean_" in name else P.AGG_RULES
        grid = {"clf": [LogisticRegression(max_iter=5000)], "clf__C": [0.01, 0.1, 1.0]}
        parts = P.run_trial_level(d, feats, fl, grid, rules=rules, n_jobs=2)
    else:
        parts = P.run_subject_level(d, feats, fl, P.arm_grid(name), n_jobs=2)
    return binary_metrics(parts.y, parts.score)["auc"]


# Mechanism: univariate AUCs (full sample, descriptive, no fitting). AUC > .5 means higher value = liar.
pub = P.published_style_features(trials)
extra = P.subject_features(trials, ["RT", "logRT", "pc_sub", "pc_none", "blink_per_s", "n_fix"], ["diff", "ratio", "devc"])
tab = pub.merge(extra.drop(columns="y"), on="subject")
tab["RT_all"] = trials.groupby("subject").RT.mean().to_numpy()
uni = {}
for c in tab.columns:
    if c in ("subject", "y"):
        continue
    v = tab[c].to_numpy(float)
    ok = ~np.isnan(v)
    uni[c] = float(roc_auc_score(tab.y[ok], v[ok]))
final["univariate_auc"] = dict(sorted(uni.items(), key=lambda kv: -abs(kv[1] - .5)))
y, v = tab.y.to_numpy(), tab["RT_U"].to_numpy()
final["no_model_RT_U"] = {"auc": float(roc_auc_score(y, v)), "eer": eer(y, v)}
final["no_model_RT_all"] = {"auc": float(roc_auc_score(y, tab.RT_all)), "eer": eer(y, tab.RT_all.to_numpy())}
s = trials.groupby("subject").agg(RT=("RT", "mean"), n_fix=("n_fix", "mean"))
final["corr_subject_mean_RT_vs_nfix"] = float(np.corrcoef(s.RT, s.n_fix)[0, 1])
final["group_means"] = {k: trials.groupby(["y", "qt"])[k].mean().unstack().round(3).to_dict()
                        for k in ("RT", "pc_sub", "pc_div", "blink_per_s", "n_fix")}

# Standardised LR weights on all participants (descriptive), for each candidate that has a subject table
final["lr_weights"] = {}
for key, name in cands.items():
    if name.startswith("I3"):
        continue
    d = P.arm_subject_table(trials, name)
    feats = res["arms"][name]["features"]
    if name.startswith("I7_lgbm"):
        continue
    clf = (LogisticRegression(solver="saga", l1_ratio=0.5, C=0.1, max_iter=20000) if name.startswith("I7_lr_en")
           else LogisticRegression(C=0.1, max_iter=5000))
    m = P.pipe(clf).fit(d[feats].to_numpy(float), d.y.to_numpy(int))
    w = pd.Series(m.named_steps["clf"].coef_[0], index=feats).sort_values(key=np.abs, ascending=False)
    final["lr_weights"][name] = {"n_nonzero": int((w != 0).sum()), "n_nonzero_ocular": int((w[w.index.str.startswith(("pc_", "blink"))] != 0).sum()),
                                 "top": w.head(10).round(3).to_dict()}
(OUT / "final.json").write_text(json.dumps(final, indent=1))
# Permutations last (slowest). Each null AUC is checkpointed to scores/perm_<key>.json and reused on restart.
final["permutation"] = {}
for key, name, n_perm in (("ocular", cands["ocular"], A.n_perm), ("ocular_fixrate", "I5_ocular_plus_fixrate", A.n_perm),
                          ("fusion", cands["fusion"], 20)):
    obs = float(np.mean([binary_metrics(p.y, p.score)["auc"] for _, p in
                         pd.read_csv(OUT / "scores" / f"{name}.csv", dtype={"unit": str}).groupby("repeat")]))
    ck = OUT / "scores" / f"perm_{key}.json"
    state = json.loads(ck.read_text()) if ck.exists() else {"arm": name, "null": [], "seconds": 0.0}
    t0 = time.time() - state["seconds"]
    for i in range(len(state["null"]), n_perm):
        if time.time() - t0 > A.perm_budget_s:
            break
        state["null"].append(one_perm(name, i))
        state["seconds"] = time.time() - t0
        ck.write_text(json.dumps(state))
    null = np.asarray(state["null"])
    final["permutation"][key] = {
        "arm": name, "n_perm": len(null), "observed_auc_mean_over_repeats": obs,
        "null_mean": float(null.mean()), "null_sd": float(null.std()), "null_p95": float(np.percentile(null, 95)),
        "p_value": float((1 + np.sum(null >= obs)) / (1 + len(null))),
        "null": null.tolist(), "seconds": state["seconds"]}
    print(key, {k: v for k, v in final["permutation"][key].items() if k != "null"}, flush=True)
    (OUT / "final.json").write_text(json.dumps(final, indent=1))
print("final done", flush=True)
