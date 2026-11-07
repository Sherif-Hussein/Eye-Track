"""Phase 6 Track A, A6: power and minimum detectable effect for every primary comparison (PREREG section 8)."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, rankdata  # noqa: E402

from phase6_common import *  # noqa: E402,F401,F403
import phase6_a2_auc_definition as A2  # noqa: E402

SEED = 20261206
N_SIM = 2000
Z = {"one_sided_.05": norm.ppf(0.95), "holm_.05/7": norm.ppf(1 - 0.05 / 7), "one_sided_.025": norm.ppf(0.975)}
Z80 = norm.ppf(0.80)


def hm_se(A, n1, n0):
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    return float(np.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n0 - 1) * (Q2 - A * A)) / (n1 * n0)))


def min_detectable_auc(n1, n0, z_alpha):
    """Smallest true AUC detected against 0.5 with 80% power (Hanley-McNeil SE; null SE at 0.5)."""
    se0 = hm_se(0.5, n1, n0)
    for A in np.arange(0.5, 1.0, 0.0005):
        if A - 0.5 >= z_alpha * se0 + Z80 * hm_se(A, n1, n0):
            return float(A)
    return np.nan


def sim_sd_delta(A1_, A2_, n1, n0, rho, rng):
    """Binormal simulation of the SD of the AUC difference at the actual n1, n0, AUCs and within-class between-arm score correlation."""
    d1, d2 = np.sqrt(2) * norm.ppf(A1_), np.sqrt(2) * norm.ppf(A2_)
    L = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))
    out = np.empty(N_SIM)
    for i in range(N_SIM):
        p = rng.standard_normal((n1, 2)) @ L.T + np.array([d1, d2])
        q = rng.standard_normal((n0, 2)) @ L.T
        y = np.r_[np.ones(n1), np.zeros(n0)]
        out[i] = auc(y, np.r_[p[:, 1], q[:, 1]]) - auc(y, np.r_[p[:, 0], q[:, 0]])
    return float(out.std(ddof=1))


def power(delta, se, z_alpha):
    return float(norm.cdf(delta / se - z_alpha))


def person_row(name, task, rng):
    y = task.y
    Sb, Su = task.S["base"].mean(1), task.S["uni"].mean(1)
    n = len(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    ab, au = auc(y, Sb), auc(y, Su)
    brng = np.random.default_rng(SEED)
    bb, bu = [], []
    for _ in range(B):
        i = brng.integers(0, n, n)
        bb.append(auc(y[i], Sb[i]))
        bu.append(auc(y[i], Su[i]))
    bb, bu = np.array(bb), np.array(bu)
    se_boot = float((bu - bb).std(ddof=1))
    r_auc = float(np.corrcoef(bb, bu)[0, 1])
    se_b, se_u = hm_se(ab, n1, n0), hm_se(au, n1, n0)
    se_hm = float(np.sqrt(max(se_b ** 2 + se_u ** 2 - 2 * r_auc * se_b * se_u, 1e-12)))
    # within-class between-arm correlation of rank-normal scores
    rho = np.mean([np.corrcoef(rankdata(Sb[y == c]), rankdata(Su[y == c]))[0, 1] for c in (0, 1)])
    se_sim = sim_sd_delta(max(ab, 0.52), max(au, 0.52), n1, n0, float(np.clip(rho, -0.95, 0.95)), rng)
    row = {"n": n, "n_pos": n1, "n_neg": n0, "auc_base": ab, "auc_unified": au, "delta_observed": au - ab, "boot_se_auc_base": float(bb.std(ddof=1)), "boot_se_auc_unified": float(bu.std(ddof=1)),
           "boot_corr_of_aucs": r_auc, "within_class_rank_corr_between_arms": float(rho), "boot_se_delta": se_boot, "hanley_mcneil_se_delta": se_hm, "sim_binormal_sd_delta": se_sim}
    for k, z in Z.items():
        row[f"mde_delta_80pct_{k}"] = (z + Z80) * se_boot
        row[f"mde_delta_80pct_{k}_using_sim_se"] = (z + Z80) * se_sim
    row["min_single_auc_detectable_80pct_one_sided_.05"] = min_detectable_auc(n1, n0, Z["one_sided_.05"])
    row["min_single_auc_detectable_80pct_holm_.05/7"] = min_detectable_auc(n1, n0, Z["holm_.05/7"])
    row["power_at_true_delta"] = {str(d): {k: power(d, se_boot, z) for k, z in Z.items()} for d in (0.05, 0.10, 0.15, 0.20)}
    row["n_needed_for_80pct_at_true_delta_one_sided_.05"] = {str(d): float(n * ((Z["one_sided_.05"] + Z80) * se_boot / d) ** 2) for d in (0.05, 0.10, 0.15, 0.20)}
    return row


def main():
    rng = np.random.default_rng(SEED + 99)
    tasks = {
        "celniak": A2.person_task("celniak", pd.read_csv(P2 / "celniak" / "trials_baseline_svm_paper.csv"), pd.read_csv(OOF / "celniak_unified_rows.csv.gz")),
        "preti_A": A2.person_task("preti_A", pd.read_csv(P2 / "preti" / "scores" / "B_ocular.csv"), pd.read_csv(OOF / "preti_A_ocular_rows.csv.gz")),
        "preti_B": A2.person_task("preti_B", pd.read_csv(P2 / "preti" / "scores" / "B_fusion.csv"), pd.read_csv(OOF / "preti_B_ocular_rt_rows.csv.gz")),
        "mazza_g2_vs_g3": A2.person_task("mazza_g2_vs_g3", pd.read_csv(P2 / "mazza" / "oof" / "g2_vs_g3__base_pub36_lr_unreg.csv"), pd.read_csv(OOF / "mazza_g2_vs_g3_rows.csv.gz")),
        "mazza_guilty_vs_innocent": A2.person_task("mazza_guilty_vs_innocent", pd.read_csv(P2 / "mazza" / "oof" / "guilty_vs_innocent__base_pub36_lr_unreg.csv"), pd.read_csv(OOF / "mazza_guilty_vs_innocent_rows.csv.gz")),
    }
    out = {"definition": {"B": B, "seed": SEED, "n_sim": N_SIM, "note": "SEs condition on the fitted models (no refit variance), so MDEs are lower bounds and powers upper bounds"}, "tests": {}}
    for k, t in tasks.items():
        out["tests"][k] = person_row(k, t, rng)
        log(k, {a: round(b, 4) for a, b in out["tests"][k].items() if isinstance(b, float)})
    # Chen detection (points) and person AUC
    cps = pd.concat([pickle.load(open(RES / "phase3" / "chen_confirm" / "checkpoints" / f"repeat{r}.pkl", "rb"))["persons"] for r in range(3)], ignore_index=True)
    uni = pd.read_csv(OOF / "chen_unified_persons.csv.gz")
    units = sorted(cps.unit.unique())
    det_rr = cps[cps.arm == "rate_rule"].groupby("unit").detect_cal.mean().loc[units].to_numpy()
    det_i2 = cps[cps.arm == "I2b"].groupby("unit").detect_cal.mean().loc[units].to_numpy()
    f = uni[uni.kind == "fam"].assign(d=lambda d: (d.stat > d.tau_fpr7).astype(float))
    det_un = f.groupby("unit").d.mean().loc[units].to_numpy()
    n = 56
    brng = np.random.default_rng(SEED + 5)
    ds_un, ds_i2 = [], []
    for _ in range(B):
        i = brng.integers(0, n, n)
        ds_un.append(det_un[i].mean() - det_rr[i].mean())
        ds_i2.append(det_i2[i].mean() - det_rr[i].mean())
    chen = {"n": n, "detection": {"unified": float(det_un.mean()), "rate_rule": float(det_rr.mean()), "I2b": float(det_i2.mean())}}
    for nm, ds, dobs in (("unified_minus_rate_rule", ds_un, det_un.mean() - det_rr.mean()), ("I2b_minus_rate_rule", ds_i2, det_i2.mean() - det_rr.mean())):
        se = float(np.std(ds, ddof=1))
        chen[nm] = {"delta_observed": float(dobs), "boot_se": se, **{f"mde_80pct_{k}": (z + Z80) * se for k, z in Z.items()},
                    "power_at_true_delta": {str(d): {k: power(d, se, z) for k, z in Z.items()} for d in (0.10, 0.15, 0.20, 0.25)},
                    "n_needed_80pct_at_true_delta_.15_one_sided_.05": float(n * ((Z["one_sided_.05"] + Z80) * se / 0.15) ** 2)}
    pt, _ = A2.chen_tasks()
    brng = np.random.default_rng(SEED + 6)
    dd = []
    for _ in range(B):
        idx = {0: np.concatenate([pt.clusters[0][i] for i in brng.integers(0, n, n)])}
        dd.append(pt.metric("uni", "V0_repeat_avg_raw", idx) - pt.metric("base", "V0_repeat_avg_raw", idx))
    se_pa = float(np.std(dd, ddof=1))
    chen["person_auc_unified_minus_rate_rule"] = {"delta_observed": float(pt.metric("uni", "V0_repeat_avg_raw", pt.strata) - pt.metric("base", "V0_repeat_avg_raw", pt.strata)), "boot_se_delta": se_pa,
                                                  "mde_delta_80pct_one_sided_.05": (Z["one_sided_.05"] + Z80) * se_pa, "mde_delta_80pct_holm_.05/7": (Z["holm_.05/7"] + Z80) * se_pa}
    out["chen"] = chen
    # Delmas non-inferiority
    dt = A2.delmas_task()
    nm_sets = ["64_g1", "64_g2", "64_g3", "32_g3", "32_g4", "32_g5"]

    def delmas_se(drop=()):
        keep = [s for s in dt.strata if nm_sets[s] not in drop]
        brng = np.random.default_rng(SEED + 11)
        d = []
        for _ in range(B):
            idx = {s: np.concatenate([dt.clusters[s][i] for i in brng.integers(0, len(dt.clusters[s]), len(dt.clusters[s]))]) for s in keep}
            d.append(dt.metric("uni", "V0_repeat_avg_raw", idx) - dt.metric("base", "V0_repeat_avg_raw", idx))
        full = {s: dt.strata[s] for s in keep}
        return float(np.std(d, ddof=1)), float(dt.metric("uni", "V0_repeat_avg_raw", full) - dt.metric("base", "V0_repeat_avg_raw", full)), len(keep)
    ni = {}
    for nm, drop in (("six_sets_primary", ()), ("five_groups_G3_once", ("32_g3",))):
        se, dobs, ns = delmas_se(drop)
        margin = 0.02
        e = {"n_sets": ns, "n_participants_per_set": 15, "boot_se_delta": se, "delta_observed": dobs, "margin": margin, "power": {}, "n_participants_per_set_needed_for_80pct_at_true_delta_0": {}}
        for k, z in Z.items():
            e["power"][k] = {str(td): float(norm.cdf((margin + td) / se - z)) for td in (0.0, -0.005, -0.01)}
            e["n_participants_per_set_needed_for_80pct_at_true_delta_0"][k] = float(15 * ((z + Z80) * se / margin) ** 2)
        ni[nm] = e
        log("NI", nm, se, e["power"])
    out["delmas_noninferiority"] = ni
    # figure data
    rows = []
    for k, v in out["tests"].items():
        for d in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            rows.append({"test": k, "true_delta": d, **{f"power_{kk}": power(d, v["boot_se_delta"], z) for kk, z in Z.items()}, "boot_se": v["boot_se_delta"]})
    for nm in ("unified_minus_rate_rule", "I2b_minus_rate_rule"):
        for d in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            rows.append({"test": f"chen_detection_{nm}", "true_delta": d, **{f"power_{kk}": power(d, chen[nm]["boot_se"], z) for kk, z in Z.items()}, "boot_se": chen[nm]["boot_se"]})
    for nm, e in ni.items():
        for td in (0.0, -0.005, -0.01, -0.015):
            rows.append({"test": f"delmas_NI_{nm}", "true_delta": td, **{f"power_{kk}": float(norm.cdf((0.02 + td) / e["boot_se_delta"] - z)) for kk, z in Z.items()}, "boot_se": e["boot_se_delta"]})
    pd.DataFrame(rows).to_csv(OUT / "figdata_A6_power.csv", index=False)
    save_section("A6", out)
    T = out["tests"]
    mde = "; ".join(f"{k} n={v['n']} SE {v['boot_se_delta']:.3f}, MDE {v['mde_delta_80pct_one_sided_.05']:.2f} (Holm {v['mde_delta_80pct_holm_.05/7']:.2f}), min single AUC {v['min_single_auc_detectable_80pct_one_sided_.05']:.2f}" for k, v in T.items())
    c = chen["I2b_minus_rate_rule"]
    cu = chen["unified_minus_rate_rule"]
    n6, n5 = ni["six_sets_primary"], ni["five_groups_G3_once"]
    para = (f"Power at the actual n (80% power; SEs from the paired bootstrap of the stored scores, which exclude refit variance): {mde}. "
            f"Chen detection (n = 56): unified minus rate rule SE {cu['boot_se']*100:.1f} points, MDE {cu['mde_80pct_one_sided_.05']*100:.0f} points ({cu['mde_80pct_holm_.05/7']*100:.0f} Holm); I2b minus rate rule SE {c['boot_se']*100:.1f}, MDE {c['mde_80pct_one_sided_.05']*100:.0f} points ({c['mde_80pct_holm_.05/7']*100:.0f} Holm); the observed Phase 4 difference of {cu['delta_observed']*100:.1f} points is below its MDE. "
            f"Delmas non-inferiority (margin 0.02, true delta 0): SE {n6['boot_se_delta']:.4f}, power {n6['power']['one_sided_.05']['0.0']*100:.0f}% at one-sided alpha .05, {n6['power']['one_sided_.025']['0.0']*100:.0f}% at .025, {n6['power']['holm_.05/7']['0.0']*100:.0f}% at .05/7; "
            f"participants per set needed for 80%: {n6['n_participants_per_set_needed_for_80pct_at_true_delta_0']['one_sided_.05']:.0f} / {n6['n_participants_per_set_needed_for_80pct_at_true_delta_0']['one_sided_.025']:.0f} / {n6['n_participants_per_set_needed_for_80pct_at_true_delta_0']['holm_.05/7']:.0f} (now 15). "
            f"With G3 counted once (five groups): SE {n5['boot_se_delta']:.4f}, power {n5['power']['one_sided_.05']['0.0']*100:.0f}% / {n5['power']['one_sided_.025']['0.0']*100:.0f}% / {n5['power']['holm_.05/7']['0.0']*100:.0f}%. "
            "A 'not shown' verdict is therefore the expected outcome even if nothing is lost.")
    save_card("A6", para)
    log(para)


if __name__ == "__main__":
    main()
