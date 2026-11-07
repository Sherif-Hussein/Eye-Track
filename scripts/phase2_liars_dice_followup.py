"""Phase 2 / liars_dice: mechanism and confound checks for the best ocular candidates, after the screening.

Ablation by feature family on the frozen folds, feature importance, how much of the ocular score is a proxy
for bid time, between-person base-rate contribution to pooled AUC, Holm across the "what the eyes add"
tests, and participant-bootstrap CIs on the per-participant metrics.

Usage: python -I scripts/phase2_liars_dice_followup.py <bids_csv> <results_dir> <oof_cache_dir>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import phase2_liars_dice as m  # noqa: E402
from edd.folds import load_folds  # noqa: E402
from edd.liars_dice import (OC_AOI, OC_COMMON, RT, add_past_stats, evaluate, paired_cluster_bootstrap_diff,  # noqa: E402
                            per_participant, run_cv)

FAMILIES = {"fixation_core": ["fix_rate", "fix_dur_mean", "fix_dur_median", "fix_dur_first"],
            "fixation_length_dependent": ["fix_dur_sd", "fix_dur_max", "fix_time_frac"],
            "blink": ["blink_rate", "blink_dur_mean", "any_blink"],
            "aoi": OC_AOI}


def spec_like(best, feats):
    """Same model family as the best candidate, on a feature subset (nested tuning retained)."""
    kind = "lgbm" if "lgbm" in best else "lr"
    if "online" in best:
        return m.model_spec(kind, online=feats, keep_raw="_raw" in best, k=5.0 if kind == "lgbm" else None)
    return m.model_spec(kind, raw=feats)


def importance(d, best):
    oc = OC_COMMON + (OC_AOI if d.exp.iloc[0] == "pvp" else [])
    _, cols, fit_fn = m.ideas_for(d.exp.iloc[0])[best]
    fitted = fit_fn(d[cols].to_numpy(float), d.y.to_numpy(int), d.group.astype(str).to_numpy())
    est = fitted.best_estimator_
    clf = est.named_steps["clf"]
    keep_raw = "_raw" in best
    names = [f"dev_{f}" for f in oc] + (list(oc) if keep_raw else []) if "online" in best else list(oc)
    if "lgbm" in best:
        v = clf.booster_.feature_importance("gain").astype(float)
        v = v / v.sum()
    else:
        v = clf.coef_.ravel()
    order = np.argsort(-np.abs(v))[:12]
    return {"kind": "lgbm_gain_share" if "lgbm" in best else "lr_standardised_coef",
            "top": {names[i]: float(v[i]) for i in order}}


def proxy_and_base_rate(oof, d):
    a = m.avg_scores(oof)
    t = d.iloc[a.index.to_numpy()]
    rt = t.log_bid_time.to_numpy()
    s = a.score.to_numpy()
    # Within-participant RT quartiles, so strata are not person effects.
    q = t.groupby("pid").log_bid_time.transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=False)).to_numpy()
    by = {}
    for k in range(4):
        i = q == k
        by[f"rt_q{k + 1}"] = float(roc_auc_score(a.y[i], s[i]))
    within = sum((q == k).sum() * by[f"rt_q{k + 1}"] for k in range(4)) / len(q)
    # Pooled AUC after removing each participant's mean score: no between-person level differences left.
    centred = a.score - a.groupby("pid").score.transform("mean")
    return {"spearman_score_vs_log_bid_time": float(spearmanr(s, rt).statistic),
            "auc_within_rt_quartiles": by, "auc_within_rt_quartiles_weighted": float(within),
            "pooled_auc": float(roc_auc_score(a.y, s)),
            "pooled_auc_after_person_mean_centring_scores": float(roc_auc_score(a.y, centred)),
            "mean_per_participant_auc": float(per_participant(oof).auc.mean())}


def boot_mean(v, n_boot=2000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(0)
    b = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n": int(len(v))}


def main(bids_csv, results_dir, cache_dir):
    res_dir = Path(results_dir)
    results = json.loads((res_dir / "results.json").read_text())
    folds = load_folds(res_dir / "folds.json")
    allb = pd.read_csv(bids_csv)
    on = add_past_stats(allb, OC_COMMON + OC_AOI + RT, roles=("Bidder",))
    on = on[allb.Role == "Bidder"].reset_index(drop=True)
    load = lambda e, n: pd.read_csv(Path(cache_dir) / f"{e}__{n}__r3.csv")  # noqa: E731
    out = {}
    for exp in ("pvc", "pvp"):
        d = on[on.exp == exp].reset_index(drop=True)
        best = results["best"][exp]["best_ocular_candidate"]["name"]
        base = results["best"][exp]["honest_baseline"]["name"]
        block = {"best": best, "importance": importance(d, best)}
        o_best = load(exp, best)
        s_best = m.avg_scores(o_best)
        block["proxy_base_rate"] = {n: proxy_and_base_rate(load(exp, n), d) for n in (best, base, "I4_lr_rt")}
        block["per_participant_ci95"] = {n: {k: boot_mean(per_participant(load(exp, n))[k]) for k in ("auc", "bacc", "hter", "eer")}
                                         for n in (best, base)}
        oc = OC_COMMON + (OC_AOI if exp == "pvp" else [])
        abl = {}
        for fam, feats in FAMILIES.items():
            feats = [f for f in feats if f in oc]
            if not feats:
                continue
            for label, use in ((f"drop_{fam}", [f for f in oc if f not in feats]), (f"only_{fam}", feats)):
                cols, fit_fn = spec_like(best, use)
                o = run_cv(d, cols, fit_fn, folds[exp])
                ev = evaluate(o)
                a = m.avg_scores(o).loc[s_best.index]
                abl[label] = {"auc": ev["trial"]["auc"], "pp_auc_mean": ev["per_participant"]["auc_mean"],
                              "delta_vs_full": paired_cluster_bootstrap_diff(s_best.y, a.score, s_best.score, s_best.pid, n_boot=1000)}
                print(exp, label, round(ev["trial"]["auc"]["mean"], 3), "pp", round(ev["per_participant"]["auc_mean"], 3), flush=True)
        block["ablation"] = abl
        ea = results["comparisons"][exp]["eyes_add"]
        p = {k: next(iter(v.values()))["p_two_sided"] for k, v in ea.items()}
        block["eyes_add_holm"] = m.holm(p)
        # Within-person measure: Holm over the same family as the pooled-AUC comparisons.
        cmp_oc = results["comparisons"][exp]["ocular_vs_honest_baseline"]
        block["ocular_vs_baseline_wilcoxon_holm"] = m.holm({k: v["wilcoxon_p"] for k, v in cmp_oc.items()})
        block["eyes_add_wilcoxon_holm"] = m.holm({k: next(iter(v.values()))["wilcoxon_p"] for k, v in ea.items()})
        pp = per_participant(o_best)
        t = d.groupby("pid").y.agg(["mean", "size"])
        block["participants_extreme_lie_rate"] = t[(t["mean"] > .8) | (t["mean"] < .1)].round(3).reset_index().to_dict("records")
        block["sessions_per_participant_lt3"] = {k: int(v) for k, v in d.groupby("pid").Session.nunique().items() if v < 3}
        block["pp_auc_quantiles"] = pp.auc.describe().round(3).to_dict()
        out[exp] = block
    (res_dir / "followup.json").write_text(json.dumps(out, indent=1, default=float))
    print("followup done")


if __name__ == "__main__":
    main(*sys.argv[1:4])
