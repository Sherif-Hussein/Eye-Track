"""Phase 6 Track A, A7: Preti reaction-time benchmark with no model and no fitting (PREREG section 7)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase6_common import *  # noqa: E402,F401,F403
from edd import preti as P  # noqa: E402

SEED = 20261207
NPERM = 10000


def eer_of(y, s):
    pos, neg = s[y == 1], s[y == 0]
    t = np.unique(s)
    cand = np.concatenate([[t[0] - 1], (t[:-1] + t[1:]) / 2, [t[-1] + 1]])
    fpr = np.array([(neg > c).mean() for c in cand])
    fnr = np.array([(pos <= c).mean() for c in cand])
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def main():
    t = P.trials_from_fixations(P.fixations(P.load_workbook(ROOT / "data" / "preti" / "preti2026.xlsx")))
    t["logRT"] = np.log(t["RT"])
    g = t.groupby(["subject", "qt"])
    mean = lambda col: g[col].mean().unstack().reindex(columns=["C", "E", "U"])  # noqa: E731
    y = t.groupby("subject").y.first()
    rt, lrt, bl, pu = mean("RT"), mean("logRT"), mean("blink_per_s"), mean("pc_sub")
    scores = {
        "RT_U_minus_C (primary)": rt["U"] - rt["C"],
        "RT_U": rt["U"], "RT_U_minus_E": rt["U"] - rt["E"], "RT_E_minus_C": rt["E"] - rt["C"],
        "logRT_U_minus_C": lrt["U"] - lrt["C"], "RT_U_over_C": rt["U"] / rt["C"],
        "blink_per_s_U_minus_C (per-second ocular)": bl["U"] - bl["C"],
        "pupil_pc_sub_U_minus_C (ocular)": pu["U"] - pu["C"],
    }
    subj = y.index
    yv = y.to_numpy()
    n = len(subj)
    log("participants", n, "liars", int(yv.sum()))
    # comparators: Phase 4 Level B / Level A pipelines and the honest baseline B_fusion (OOF, repeat-averaged unit means)
    uni_B = pd.read_csv(OOF / "preti_B_ocular_rt_rows.csv.gz").groupby("unit").score.mean()
    uni_A = pd.read_csv(OOF / "preti_A_ocular_rows.csv.gz").groupby("unit").score.mean()
    base_B = pd.read_csv(P2 / "preti" / "scores" / "B_fusion.csv").groupby("unit").score.mean()
    assert set(uni_B.index) == set(subj), (len(set(uni_B.index) & set(subj)), n)
    comp = {"pipeline_LevelB_phase4": uni_B.loc[subj].to_numpy(), "baseline_B_fusion": base_B.loc[subj].to_numpy(), "pipeline_LevelA_phase4": uni_A.loc[subj].to_numpy()}
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, n, n) for _ in range(B)]
    prng = np.random.default_rng(SEED + 1)
    out = {"definition": {"n": n, "liars": int(yv.sum()), "truth_tellers": int((1 - yv).sum()), "B": B, "seed": SEED, "n_perm": NPERM}, "scores": {}, "comparison": {}}
    rows = []
    for name, s in scores.items():
        v = s.loc[subj].to_numpy(float)
        ok = np.isfinite(v)
        a = auc(yv[ok], v[ok])
        bs = [auc(yv[i], v[i]) for i in draws]
        perms = np.array([auc(prng.permutation(yv[ok]), v[ok]) for _ in range(NPERM)])
        out["scores"][name] = {"auc": a, "ci95": pct_ci(bs), "eer_in_sample": eer_of(yv[ok], v[ok]), "p_perm_one_sided_greater": float((1 + np.sum(perms >= a)) / (1 + NPERM)),
                               "n_used": int(ok.sum()), "mean_liar": float(np.nanmean(v[yv == 1])), "mean_truth": float(np.nanmean(v[yv == 0]))}
        rows.append({"score": name, "auc": a, "lo": out["scores"][name]["ci95"][0], "hi": out["scores"][name]["ci95"][1], "eer": out["scores"][name]["eer_in_sample"]})
    prim = scores["RT_U_minus_C (primary)"].loc[subj].to_numpy(float)
    for cname, c in comp.items():
        a_c = auc(yv, c)
        d = np.array([auc(yv[i], prim[i]) - auc(yv[i], c[i]) for i in draws])
        out["comparison"][cname] = {"auc_comparator": a_c, "auc_RT_contrast": auc(yv, prim), "delta_RTcontrast_minus_comparator": auc(yv, prim) - a_c, "ci95": pct_ci(d)}
        rows.append({"score": cname, "auc": a_c, "lo": pct_ci([auc(yv[i], c[i]) for i in draws])[0], "hi": pct_ci([auc(yv[i], c[i]) for i in draws])[1], "eer": eer_of(yv, c)})
    out["stored_phase2_per_second_ablation"] = {"auc": 0.532, "source": "results/phase2/preti/comparison_card.md section 4a (pupil + blinks per second, fitted model, repeat-mean AUC)"}
    pd.DataFrame(rows).to_csv(OUT / "figdata_A7_preti_rt_benchmark.csv", index=False)
    save_section("A7", out)
    s = out["scores"]
    pr = s["RT_U_minus_C (primary)"]
    c = out["comparison"]
    para = (f"Preti (n = {n}: {int(yv.sum())} liars, {int((1 - yv).sum())} truth-tellers), no model, no fitting: the mean RT difference between unexpected and control questions (U - C) gives AUC {pr['auc']:.3f} {ci_u(pr['ci95'])}, in-sample EER {pr['eer_in_sample']:.3f}, permutation p {pr['p_perm_one_sided_greater']:.4f}. "
            f"Descriptive variants: RT_U {s['RT_U']['auc']:.3f}, U - E {s['RT_U_minus_E']['auc']:.3f}, E - C {s['RT_E_minus_C']['auc']:.3f}, log-RT contrast {s['logRT_U_minus_C']['auc']:.3f}, U/C {s['RT_U_over_C']['auc']:.3f}. "
            f"Phase 4 Level B pipeline (repeat-averaged OOF) {c['pipeline_LevelB_phase4']['auc_comparator']:.3f}; the single RT contrast minus the pipeline = {c['pipeline_LevelB_phase4']['delta_RTcontrast_minus_comparator']:+.3f} {ci_s(c['pipeline_LevelB_phase4']['ci95'])}; minus the honest baseline B_fusion ({c['baseline_B_fusion']['auc_comparator']:.3f}) = {c['baseline_B_fusion']['delta_RTcontrast_minus_comparator']:+.3f} {ci_s(c['baseline_B_fusion']['ci95'])}. "
            f"Per-second ocular, no model: blinks per second U - C AUC {s['blink_per_s_U_minus_C (per-second ocular)']['auc']:.3f} {ci_u(s['blink_per_s_U_minus_C (per-second ocular)']['ci95'])}, pupil U - C {s['pupil_pc_sub_U_minus_C (ocular)']['auc']:.3f} {ci_u(s['pupil_pc_sub_U_minus_C (ocular)']['ci95'])} "
            "(higher = more liar-like, no sign flipping); the stored Phase 2 fitted per-second ablation was 0.532. The primary contrast was fixed before the run; it is a one-number benchmark on one lying task and says nothing about recognition tests.")
    save_card("A7", para)
    log(para)


if __name__ == "__main__":
    main()
