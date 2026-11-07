"""Phase 6 Track A, A8: Nadeau-Bengio / Bouckaert-Frank corrected repeated-CV intervals for each primary delta (PREREG section 9)."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

from phase6_common import *  # noqa: E402,F401,F403

STORED = json.loads((RES / "phase4" / "unified" / "results.json").read_text())["primary_tests"]


def nb(d, rho, margin=None):
    """Corrected resampled t summary for per-fold differences d (J values) with test/train ratio rho."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    J = len(d)
    m, s2 = float(d.mean()), float(d.var(ddof=1))
    out = {"J": J, "mean_delta": m, "sd_fold_deltas": float(np.sqrt(s2)), "rho_ntest_over_ntrain": float(rho)}
    for lab, var in (("naive_1_over_J", s2 / J), ("corrected_NB_BF", (1 / J + rho) * s2)):
        se = float(np.sqrt(var))
        tc = tdist.ppf(0.975, J - 1)
        tt = m / se if se > 0 else np.nan
        e = {"se": se, "ci95": [m - tc * se, m + tc * se], "t": float(tt), "p_two_sided": float(2 * tdist.sf(abs(tt), J - 1)), "p_one_sided_delta_gt_0": float(tdist.sf(tt, J - 1)),
             "lower_one_sided_95": m - tdist.ppf(0.95, J - 1) * se}
        if margin is not None:
            tn = (m + margin) / se
            e["p_noninferior_one_sided_H0_delta_le_-margin"] = float(tdist.sf(tn, J - 1))
        out[lab] = e
    out["variance_inflation_NB_vs_naive"] = float((1 / J + rho) * J)
    return out


def across_repeat(dfr):
    """dfr: DataFrame with columns repeat, d. Each repeat's mean as one observation."""
    m = dfr.groupby("repeat").d.mean().to_numpy()
    R = len(m)
    se = float(m.std(ddof=1) / np.sqrt(R))
    tc = tdist.ppf(0.975, R - 1)
    return {"R": R, "mean": float(m.mean()), "ci95": [float(m.mean() - tc * se), float(m.mean() + tc * se)], "se": se}


def person_fold_table(base, uni_persons):
    b = base.assign(unit=base.unit.astype(str)).groupby(["repeat", "fold", "unit"], as_index=False).agg(y=("y", "first"), s_base=("score", "mean"))
    u = uni_persons.assign(unit=uni_persons.unit.astype(str)).groupby(["repeat", "fold", "unit"], as_index=False).agg(y=("y", "first"), s_uni=("stat", "mean"))
    m = u.merge(b[["repeat", "unit", "fold", "s_base"]], on=["repeat", "unit"], suffixes=("", "_b"))
    assert (m.fold == m.fold_b).all(), "fold assignment differs between arms"
    return m


def per_fold_auc(m):
    rows = []
    N = m.unit.nunique()
    for (r, f), d in m.groupby(["repeat", "fold"]):
        n_te = len(d)
        if d.y.nunique() < 2:
            rows.append({"repeat": r, "fold": f, "d": np.nan, "ratio": n_te / (N - n_te)})
            continue
        rows.append({"repeat": r, "fold": f, "d": auc(d.y, d.s_uni) - auc(d.y, d.s_base), "ratio": n_te / (N - n_te), "auc_uni": auc(d.y, d.s_uni), "auc_base": auc(d.y, d.s_base)})
    return pd.DataFrame(rows)


def person_test(name, base, uni_persons, stored_key):
    m = person_fold_table(base, uni_persons)
    reps = sorted(m["repeat"].unique())
    tab = per_fold_auc(m)
    ok = tab.dropna(subset=["d"])
    res = nb(ok.d, tab.ratio.mean())
    res.update({"n_units": int(m.unit.nunique()), "repeats_used": reps, "folds_total": len(tab), "folds_dropped_single_class": int(tab.d.isna().sum()),
                "mean_per_fold_auc_base": float(ok.auc_base.mean()), "mean_per_fold_auc_unified": float(ok.auc_uni.mean()),
                "across_repeat": across_repeat(ok), "phase4_pooled_delta": STORED[stored_key]["delta"], "phase4_bootstrap_ci95": STORED[stored_key]["ci95"]})
    return res, tab.assign(test=name)


def delmas_test(drop=()):
    base = pd.read_csv(P2 / "delmas_raw" / "checkpoints" / "P4_svm_1000_oof.csv.gz")
    sets = ["64_g1", "64_g2", "64_g3", "32_g3", "32_g4", "32_g5"]
    per_set, rows = {}, []
    for s in sets:
        if s in drop:
            continue
        u = pd.read_csv(OOF / f"delmas_{s}_rows.csv.gz").rename(columns={"unit": "participant"})
        b = base[base.set == s][["repeat", "participant", "image", "score"]].rename(columns={"score": "s_base"})
        m = u.merge(b, on=["repeat", "participant", "image"])
        N = m.participant.nunique()
        ds = []
        for (r, f), d in m.groupby(["repeat", "fold"]):
            n_te = d.participant.nunique()
            dd = auc(d.y, d.score) - auc(d.y, d.s_base)
            ds.append({"repeat": r, "fold": f, "d": dd, "ratio": n_te / (N - n_te), "set": s})
        dfm = pd.DataFrame(ds)
        per_set[s] = dfm
        rows.append(dfm)
    # combine: mean of sets, independent per-set variances
    S = len(per_set)
    means = np.array([d.d.mean() for d in per_set.values()])
    J = len(next(iter(per_set.values())))
    out = {"n_sets": S, "J_per_set": J, "mean_delta": float(means.mean()), "per_set_mean_delta": {s: float(d.d.mean()) for s, d in per_set.items()}}
    for lab in ("naive_1_over_J", "corrected_NB_BF"):
        var = 0.0
        for d in per_set.values():
            s2 = d.d.var(ddof=1)
            rho = d.ratio.mean()
            var += ((1 / J + rho) if lab.startswith("corr") else 1 / J) * s2
        se = float(np.sqrt(var) / S)
        df = J - 1
        tc = tdist.ppf(0.975, df)
        m = float(means.mean())
        out[lab] = {"se": se, "ci95": [m - tc * se, m + tc * se], "t": m / se, "p_two_sided": float(2 * tdist.sf(abs(m / se), df)), "p_one_sided_delta_gt_0": float(tdist.sf(m / se, df)),
                    "p_noninferior_one_sided_H0_delta_le_-0.02": float(tdist.sf((m + 0.02) / se, df)), "lower_one_sided_95": m - tdist.ppf(0.95, df) * se, "df_used": df}
    # across-repeat: each repeat's mean over sets of the set's mean per-fold delta
    rep = pd.concat(rows).groupby(["set", "repeat"]).d.mean().groupby("repeat").mean().to_numpy()
    se = float(rep.std(ddof=1) / np.sqrt(len(rep)))
    tc = tdist.ppf(0.975, len(rep) - 1)
    out["across_repeat"] = {"R": len(rep), "mean": float(rep.mean()), "ci95": [float(rep.mean() - tc * se), float(rep.mean() + tc * se)], "se": se}
    out["rho_ntest_over_ntrain"] = float(np.mean([d.ratio.mean() for d in per_set.values()]))
    out["variance_inflation_NB_vs_naive"] = float(out["corrected_NB_BF"]["se"] ** 2 / out["naive_1_over_J"]["se"] ** 2)
    return out, pd.concat(rows)


def chen_test():
    cps = pd.concat([pickle.load(open(RES / "phase3" / "chen_confirm" / "checkpoints" / f"repeat{r}.pkl", "rb"))["persons"] for r in range(3)], ignore_index=True)
    uni = pd.read_csv(OOF / "chen_unified_persons.csv.gz")
    res, tabs = {}, []
    for lab, arm in (("unified_minus_rate_rule", None), ("I2b_minus_rate_rule", "I2b")):
        if arm is None:
            a = uni[uni.kind == "fam"].assign(det=lambda d: (d.stat > d.tau_fpr7).astype(float))[["repeat", "fold", "unit", "det"]].rename(columns={"det": "d_a"})
        else:
            a = cps[cps.arm == arm][["repeat", "fold", "unit", "detect_cal"]].rename(columns={"detect_cal": "d_a"})
        b = cps[cps.arm == "rate_rule"][["repeat", "fold", "unit", "detect_cal"]].rename(columns={"detect_cal": "d_b"})
        m = a.merge(b, on=["repeat", "unit"], suffixes=("", "_b"))
        assert (m.fold == m.fold_b).all()
        N = m.unit.nunique()
        rows = []
        for (r, f), d in m.groupby(["repeat", "fold"]):
            rows.append({"repeat": r, "fold": f, "d": d.d_a.mean() - d.d_b.mean(), "ratio": len(d) / (N - len(d))})
        tab = pd.DataFrame(rows)
        e = nb(tab.d, tab.ratio.mean())
        e.update({"n_units": int(N), "across_repeat": across_repeat(tab), "metric": "per-fold detection rate difference (fraction of the fold's familiar participants detected at the calibrated FPR<=7% threshold)"})
        if arm is None:
            e["phase4_pooled_delta"] = STORED["chen_detection_fpr7"]["detection"]["delta_vs_rate_rule"]
            e["phase4_bootstrap_ci95"] = STORED["chen_detection_fpr7"]["detection"]["ci95"]
        res[lab] = e
        tabs.append(tab.assign(test=f"chen_{lab}"))
    return res, pd.concat(tabs)


if __name__ == "__main__":
    out = {"definition": {"rule": "Nadeau-Bengio corrected variance (1/J + n_test/n_train) s^2 over the J = repeats x folds per-fold differences, t with J-1 df; naive 1/J shown next to it; across-repeat t (R-1 df) as a conservative check",
                          "note": "per-fold metric = AUC of the fold's decision units (person AUC; Delmas image AUC per set); single-class folds dropped; Chen = per-fold detection rate",
                          "mazza_repeats": "baseline files hold 3 repeats, the unified run 5; only the 3 common repeats enter"}, "tests": {}}
    figs = []
    p2 = P2
    specs = {
        "celniak": (pd.read_csv(p2 / "celniak" / "trials_baseline_svm_paper.csv"), pd.read_csv(OOF / "celniak_unified_persons.csv.gz"), "celniak_participant_auc"),
        "preti_A": (pd.read_csv(p2 / "preti" / "scores" / "B_ocular.csv"), pd.read_csv(OOF / "preti_A_ocular_persons.csv.gz"), "preti_A_participant_auc"),
        "preti_B": (pd.read_csv(p2 / "preti" / "scores" / "B_fusion.csv"), pd.read_csv(OOF / "preti_B_ocular_rt_persons.csv.gz"), "preti_B_participant_auc"),
        "mazza_g2_vs_g3": (pd.read_csv(p2 / "mazza" / "oof" / "g2_vs_g3__base_pub36_lr_unreg.csv"), pd.read_csv(OOF / "mazza_g2_vs_g3_persons.csv.gz"), "mazza_g2_vs_g3_participant_auc"),
        "mazza_guilty_vs_innocent": (pd.read_csv(p2 / "mazza" / "oof" / "guilty_vs_innocent__base_pub36_lr_unreg.csv"), pd.read_csv(OOF / "mazza_guilty_vs_innocent_persons.csv.gz"), "mazza_guilty_vs_innocent_participant_auc"),
    }
    for k, (b, u, sk) in specs.items():
        res, tab = person_test(k, b, u, sk)
        out["tests"][k] = res
        figs.append(tab)
        log(k, round(res["mean_delta"], 3), res["corrected_NB_BF"]["ci95"], "dropped", res["folds_dropped_single_class"])
    out["tests"]["delmas_noninferiority_six_sets"], t1 = delmas_test()
    out["tests"]["delmas_noninferiority_G3_once"], _ = delmas_test(drop=("32_g3",))
    out["tests"]["delmas_noninferiority_six_sets"]["phase4_pooled_delta"] = STORED["delmas_noninferiority"]["delta"]
    out["tests"]["delmas_noninferiority_six_sets"]["phase4_bootstrap_ci95"] = STORED["delmas_noninferiority"]["ci95"]
    out["tests"]["delmas_noninferiority_G3_once"]["phase4_pooled_delta"] = STORED["delmas_sensitivity_group3_once"]["delta"]
    out["tests"]["delmas_noninferiority_G3_once"]["phase4_bootstrap_ci95"] = STORED["delmas_sensitivity_group3_once"]["ci95"]
    ch, tc = chen_test()
    out["tests"].update({f"chen_detection_{k}": v for k, v in ch.items()})
    rows = []
    for k, r in out["tests"].items():
        for lab in ("naive_1_over_J", "corrected_NB_BF"):
            rows.append({"test": k, "interval": lab, "mean_delta_per_fold": r["mean_delta"], "lo": r[lab]["ci95"][0], "hi": r[lab]["ci95"][1], "se": r[lab]["se"],
                         "phase4_pooled_delta": r.get("phase4_pooled_delta", np.nan), "phase4_lo": r.get("phase4_bootstrap_ci95", [np.nan, np.nan])[0], "phase4_hi": r.get("phase4_bootstrap_ci95", [np.nan, np.nan])[1]})
    pd.DataFrame(rows).to_csv(OUT / "figdata_A8_corrected_cv.csv", index=False)
    pd.concat(figs + [t1.assign(test="delmas_six_sets"), tc]).to_csv(OUT / "figdata_A8_per_fold_deltas.csv", index=False)
    save_section("A8", out)
    parts = []
    for k, r in out["tests"].items():
        p4 = r.get("phase4_bootstrap_ci95")
        parts.append(f"{k}: mean per-fold delta {r['mean_delta']:+.3f}, corrected 95% CI {ci_s(r['corrected_NB_BF']['ci95'])} (naive {ci_s(r['naive_1_over_J']['ci95'])}; Phase 4 bootstrap {ci_s(p4) if p4 else 'n/a'})")
    d6 = out["tests"]["delmas_noninferiority_six_sets"]
    para = ("Corrected repeated-CV (Nadeau-Bengio / Bouckaert-Frank) intervals from per-fold deltas (unified minus baseline; per-fold AUC for person tasks and Delmas image AUC per set, per-fold detection rate for Chen; "
            "variance (1/J + n_test/n_train) s^2, t with J - 1 df): " + "; ".join(parts) + ". "
            f"Delmas non-inferiority (H0: delta <= -0.02) with the corrected variance: one-sided p {d6['corrected_NB_BF']['p_noninferior_one_sided_H0_delta_le_-0.02']:.3f} (naive {d6['naive_1_over_J']['p_noninferior_one_sided_H0_delta_le_-0.02']:.3f}); lower one-sided 95% bound {d6['corrected_NB_BF']['lower_one_sided_95']:+.3f}. "
            "Where the corrected interval is wider than the bootstrap, the Phase 4 intervals understate fold-refit variance; the point estimate here is the mean of per-fold deltas, a different estimand from the pooled-AUC delta.")
    save_card("A8", para)
    log(para)
