"""Phase 6 Track A, A3: realised FPR at the nominal 7% threshold for every task (PREREG section 6)."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase6_common import *  # noqa: E402,F401,F403

SEED = 20261203
TASKS = [  # file stem, dataset, negative class description, role
    ("delmas_64_g1", "Delmas G1 (64 trials)", "simulated innocent: same person, random unknown faces as pseudo-probes", "primary"),
    ("delmas_64_g2", "Delmas G2 (64 trials)", "simulated innocent", "primary"),
    ("delmas_64_g3", "Delmas G3 (64 trials)", "simulated innocent", "primary"),
    ("delmas_32_g3", "Delmas G3 (last 32 trials; same people as 64_g3)", "simulated innocent", "primary (G3 twice)"),
    ("delmas_32_g4", "Delmas G4 (32 trials)", "simulated innocent", "primary"),
    ("delmas_32_g5", "Delmas G5 (32 trials)", "simulated innocent", "primary"),
    ("chen_unified", "Chen unified pipeline", "control-vs-control comparison of the same person (c1 and c2 mirror images)", "primary"),
    ("celniak_unified", "Celniak unified", "real innocent participants", "primary"),
    ("preti_A_ocular", "Preti Level A (ocular)", "real truth-tellers", "primary"),
    ("preti_B_ocular_rt", "Preti Level B (ocular + RT)", "real truth-tellers", "primary"),
    ("mazza_g2_vs_g3", "Colasanti G2 vs G3", "real group-3 participants", "primary"),
    ("mazza_guilty_vs_innocent", "Colasanti guilty vs innocent", "real innocent participants", "primary"),
    ("chen_stage1", "Chen cascade stage 1", "control-vs-control", "descriptive"),
    ("celniak_ctrl_valcov", "Celniak + validation-error covariate", "real innocent participants", "descriptive"),
    ("celniak_ref_valonly", "Celniak validation error only", "real innocent participants", "descriptive"),
    ("preti_A_contrast_only", "Preti A contrast-only", "real truth-tellers", "descriptive"),
    ("preti_B_contrast_only", "Preti B contrast-only", "real truth-tellers", "descriptive"),
]


def unit_table(d, neg_filter=None):
    """Per-participant realised decisions averaged over repeats."""
    d = d.copy()
    d["dec"] = (d.stat > d.tau_fpr7).astype(float)
    d.loc[~np.isfinite(d.stat) | ~np.isfinite(d.tau_fpr7), "dec"] = np.nan
    pos = d[d.y == 1].groupby("unit").dec.mean()
    neg = d[(d.y == 0) & (neg_filter(d) if neg_filter else True)].groupby("unit").dec.mean()
    return pos, neg


def summarise(pos, neg, rng):
    paired = set(pos.index) == set(neg.index) and len(pos) > 0
    out = {"n_pos_units": len(pos), "n_neg_units": len(neg), "detection": float(pos.mean()), "realised_fpr": float(neg.mean())}
    pb, nb = [], []
    if paired:
        us = np.array(sorted(pos.index))
        P, N = pos.loc[us].to_numpy(), neg.loc[us].to_numpy()
        for _ in range(B):
            i = rng.integers(0, len(us), len(us))
            pb.append(P[i].mean())
            nb.append(N[i].mean())
    else:
        P, N = pos.to_numpy(), neg.to_numpy()
        for _ in range(B):
            pb.append(P[rng.integers(0, len(P), len(P))].mean())
            nb.append(N[rng.integers(0, len(N), len(N))].mean())
    out["detection_ci95"] = pct_ci(pb)
    out["realised_fpr_ci95"] = pct_ci(nb)
    out["fpr_ci_contains_7pct"] = bool(out["realised_fpr_ci95"][0] <= 0.07 <= out["realised_fpr_ci95"][1])
    out["fpr_point_exceeds_7pct"] = bool(out["realised_fpr"] > 0.07)
    out["fpr_ci_lower_above_7pct"] = bool(out["realised_fpr_ci95"][0] > 0.07)
    return out


def main():
    rng = np.random.default_rng(SEED)
    res, rows = {}, []
    for stem, label, negdesc, role in TASKS:
        d = pd.read_csv(OOF / f"{stem}_persons.csv.gz")
        m = d.groupby(["repeat", "fold"]).n_train_neg.first()
        mneg = m.to_numpy()
        ks = np.floor(0.07 * mneg).astype(int)
        e = {"label": label, "role": role, "negative_class": negdesc,
             "threshold_rule": "tau = the (k+1)-th largest training negative statistic, k = floor(0.07 m), decide positive when stat > tau; "
                               "training negatives = inner out-of-fold statistics of the outer-training participants",
             "training_negatives_m": {"min": int(mneg.min()), "median": float(np.median(mneg)), "max": int(mneg.max())},
             "k": {"min": int(ks.min()), "max": int(ks.max())},
             "training_fpr_upper_bound_k_over_m": {"min": float((ks / mneg).min()), "max": float((ks / mneg).max())}}
        if stem.startswith("chen"):
            e["threshold_rule"] += "; Chen: negatives are the c1 and c2 statistics (mirror images), so m counts every comparison twice"
            pos, neg1 = unit_table(d, lambda t: t.kind == "c1")
            _, neg2 = unit_table(d, lambda t: t.kind == "c2")
            _, negb = unit_table(d)
            e["c1_once_primary"] = summarise(pos, neg1, rng)
            e["c2_once"] = summarise(pos, neg2, rng)
            e["both_directions_pool"] = summarise(pos, negb, rng)
            e.update({k: e["c1_once_primary"][k] for k in ("detection", "realised_fpr", "realised_fpr_ci95", "detection_ci95")})
        else:
            pos, neg = unit_table(d)
            e.update(summarise(pos, neg, rng))
        res[stem] = e
        log(stem, round(e["realised_fpr"], 3), e["realised_fpr_ci95"])
        rows.append({"task": stem, "label": label, "role": role, "detection": e["detection"], "det_lo": e["detection_ci95"][0], "det_hi": e["detection_ci95"][1],
                     "realised_fpr": e["realised_fpr"], "fpr_lo": e["realised_fpr_ci95"][0], "fpr_hi": e["realised_fpr_ci95"][1],
                     "m_train_neg_median": e["training_negatives_m"]["median"], "k_max": e["k"]["max"],
                     "training_fpr_bound_max": e["training_fpr_upper_bound_k_over_m"]["max"], "negative_class": negdesc})
    # Chen confirmatory arms from the checkpoints (their own rule: same as chen_confirm)
    cp = pd.concat([pickle.load(open(RES / "phase3" / "chen_confirm" / "checkpoints" / f"repeat{r}.pkl", "rb"))["persons"] for r in range(3)], ignore_index=True)
    for arm in ("rate_rule", "I2b"):
        c = cp[cp.arm == arm]
        pos = (c.z_fam > c.tau).astype(float).groupby(c.unit).mean()
        n1 = (c.z_c1 > c.tau).astype(float).groupby(c.unit).mean()
        n2 = (c.z_c2 > c.tau).astype(float).groupby(c.unit).mean()
        e = {"label": f"Chen confirmatory arm {arm}", "role": "stored comparator", "negative_class": "control-vs-control (c1 once / c2 once / both)",
             "threshold_rule": "chen_confirm.threshold_for_fpr on the training null z (c1 and c2 of training participants)",
             "training_negatives_m": {"min": int(c.n_null.min()), "median": float(c.n_null.median()), "max": int(c.n_null.max())},
             "c1_once_primary": summarise(pos, n1, rng), "c2_once": summarise(pos, n2, rng), "both_directions_pool": summarise(pos, (n1 + n2) / 2, rng)}
        e.update({k: e["c1_once_primary"][k] for k in ("detection", "realised_fpr", "realised_fpr_ci95", "detection_ci95")})
        res[f"chen_confirm_{arm}"] = e
        rows.append({"task": f"chen_confirm_{arm}", "label": e["label"], "role": "stored comparator", "detection": e["detection"], "det_lo": e["detection_ci95"][0], "det_hi": e["detection_ci95"][1],
                     "realised_fpr": e["realised_fpr"], "fpr_lo": e["realised_fpr_ci95"][0], "fpr_hi": e["realised_fpr_ci95"][1], "m_train_neg_median": e["training_negatives_m"]["median"],
                     "k_max": int(np.floor(0.07 * e["training_negatives_m"]["max"])), "training_fpr_bound_max": np.nan, "negative_class": e["negative_class"]})
    pd.DataFrame(rows).to_csv(OUT / "figdata_A3_realised_fpr.csv", index=False)
    prim = [res[s]["realised_fpr"] for s, _, _, r in TASKS if r.startswith("primary") and s != "delmas_32_g3"]
    prim_all = [res[s]["realised_fpr"] for s, _, _, r in TASKS if r.startswith("primary")]
    summ = {"primary_tasks_n": len(prim_all), "mean_realised_fpr_primary_incl_both_G3": float(np.mean(prim_all)), "range_primary": [float(min(prim_all)), float(max(prim_all))],
            "mean_realised_fpr_primary_G3_once": float(np.mean(prim)),
            "tasks_point_estimate_above_7pct": [s for s, _, _, r in TASKS if r.startswith("primary") and res[s]["realised_fpr"] > 0.07],
            "tasks_ci_lower_bound_above_7pct": [s for s, _, _, r in TASKS if r.startswith("primary") and (res[s].get("fpr_ci_lower_above_7pct") or res[s].get("c1_once_primary", {}).get("fpr_ci_lower_above_7pct"))],
            "note": "Delmas realised FPR is against simulated innocents, not real innocent people."}
    out = {"summary": summ, "tasks": res, "definition": {"B": B, "seed": SEED}}
    save_section("A3", out)
    txt = "; ".join(f"{stem.replace('delmas_', 'Delmas ') if stem.startswith('delmas') else label}: {res[stem]['realised_fpr']*100:.1f}% {ci_u([x*100 for x in res[stem]['realised_fpr_ci95']], 1)}" for stem, label, _, role in TASKS if role.startswith("primary"))
    para = (f"Realised FPR at the nominal 7% threshold (stored thresholds, training participants only; Chen with the control comparison counted once as z_c1), %: {txt}. "
            f"Mean over the {len(prim_all)} primary task rows {summ['mean_realised_fpr_primary_incl_both_G3']*100:.1f}% (range {summ['range_primary'][0]*100:.1f} to {summ['range_primary'][1]*100:.1f}%); "
            f"point estimate above 7% in: {', '.join(summ['tasks_point_estimate_above_7pct'])}. Rule: tau is the (k+1)-th largest training negative, k = floor(0.07 m); with m < 15 training negatives (Delmas, 12) k = 0 so the training FPR is 0 by construction and the held-out rate is driven by the simulated negatives. "
            "Delmas negatives are simulated innocents, not real people.")
    save_card("A3", para)
    log(para)


if __name__ == "__main__":
    main()
