"""Phase 6 Track A, A1: Chen detection against realised FPR across all thresholds (see results/phase6/trackA/PREREG.md section 2)."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase6_common import *  # noqa: E402,F401,F403

SEED = 20261201
ARMS = ["rate_rule", "I2b", "unified"]
XS = [k / 56 for k in range(1, 9)]
GRID = np.round(np.arange(0, 0.2501, 0.005), 4)

# ---------------------------------------------------------------- load
cps = [pickle.load(open(RES / "phase3" / "chen_confirm" / "checkpoints" / f"repeat{r}.pkl", "rb"))["persons"] for r in range(3)]
cp = pd.concat(cps, ignore_index=True)
uni = pd.read_csv(OOF / "chen_unified_persons.csv.gz")
casc = pd.read_csv(OOF / "chen_cascade_decisions.csv.gz")

units = sorted(cp.unit.unique())
assert len(units) == 56
R = 3


def arr(df, col, key="unit"):
    p = df.pivot_table(index=key, columns="repeat", values=col)
    return p.loc[units].to_numpy()          # (56, 3)


Z = {}
for arm in ("rate_rule", "I2b"):
    d = cp[cp.arm == arm]
    Z[arm] = {"fam": arr(d, "z_fam"), "c1": arr(d, "z_c1"), "c2": arr(d, "z_c2"), "tau": arr(d, "tau")}
Z["unified"] = {k: arr(uni[uni.kind == k], "stat") for k in ("fam", "c1", "c2")}
Z["unified"]["tau"] = arr(uni[uni.kind == "fam"], "tau_fpr7")
for arm in ARMS:   # check z_c1 = -z_c2
    assert np.nanmax(np.abs(Z[arm]["c1"] + Z[arm]["c2"])) < 1e-9, arm

cpick = casc.assign(unit=casc.unit.astype(int))
CD = {k: cpick[cpick.kind == k].pivot_table(index="unit", columns="repeat", values="dec").loc[units].to_numpy() for k in ("fam", "c1", "c2")}


def rep_mean(a):
    return np.nanmean(a, axis=1)


def stats_for(idx, arm, null="c1", per_repeat=False):
    """pAUC (3 flavours), full AUC and the detection-at-FPR vector for the participants idx."""
    z = Z[arm]
    if per_repeat:
        out = []
        for r in range(R):
            pos = z["fam"][idx, r]
            neg = z[null][idx, r] if null != "both" else np.concatenate([z["c1"][idx, r], z["c2"][idx, r]])
            out.append(pauc(pos[np.isfinite(pos)], neg[np.isfinite(neg)])[1])
        return float(np.mean(out))
    pos = rep_mean(z["fam"])[idx]
    neg = rep_mean(z[null])[idx] if null != "both" else np.concatenate([rep_mean(z["c1"])[idx], rep_mean(z["c2"])[idx]])
    a, an, mc = pauc(pos, neg)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    f, r = roc_points(pos, neg)
    dets = lambda xs: [float(r[f <= x + 1e-12].max()) for x in xs]  # noqa: E731
    return {"pauc_area": a, "pauc_norm": an, "pauc_mcclish": mc, "auc": auc(y, np.r_[pos, neg]),
            "det_at": dets(XS), "det_grid": dets(GRID)}


def own_threshold(idx, arm):
    """Detection and realised FPR at the stored calibrated FPR<=7% threshold; c1 counted once (and c2, and both)."""
    z = Z[arm]
    det = (z["fam"] > z["tau"]).astype(float)
    f1 = (z["c1"] > z["tau"]).astype(float)
    f2 = (z["c2"] > z["tau"]).astype(float)
    return {"detection": float(np.nanmean(det, axis=1)[idx].mean()), "fpr_c1_once": float(np.nanmean(f1, axis=1)[idx].mean()),
            "fpr_c2_once": float(np.nanmean(f2, axis=1)[idx].mean()), "fpr_both": float(((np.nanmean(f1, axis=1) + np.nanmean(f2, axis=1)) / 2)[idx].mean())}


def cascade_point(idx):
    return {"detection": float(rep_mean(CD["fam"])[idx].mean()), "fpr_c1_once": float(rep_mean(CD["c1"])[idx].mean()),
            "fpr_c2_once": float(rep_mean(CD["c2"])[idx].mean()), "fpr_both": float(((rep_mean(CD["c1"]) + rep_mean(CD["c2"])) / 2)[idx].mean())}


def run(null):
    n = 56
    full = np.arange(n)
    est = {a: stats_for(full, a, null) for a in ARMS}
    rng = np.random.default_rng(SEED)
    boots = {a: [] for a in ARMS}
    own_b = {a: [] for a in ARMS}
    cas_b, i2b_at_cas = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        for a in ARMS:
            boots[a].append(stats_for(idx, a, null))
            own_b[a].append(own_threshold(idx, a))
        cp_ = cascade_point(idx)
        cas_b.append(cp_)
        z = Z["I2b"]
        pos = rep_mean(z["fam"])[idx]
        neg = rep_mean(z[null])[idx] if null != "both" else np.concatenate([rep_mean(z["c1"])[idx], rep_mean(z["c2"])[idx]])
        i2b_at_cas.append(det_at_fpr(pos, neg, cp_["fpr_c1_once"] if null != "c2" else cp_["fpr_c2_once"]))
    res = {"estimates": {a: {k: v for k, v in est[a].items() if k not in ("det_at", "det_grid")} for a in ARMS}}
    for a in ARMS:
        res["estimates"][a].update({f"{m}_ci": pct_ci([b[m] for b in boots[a]]) for m in ("pauc_norm", "pauc_mcclish", "auc")})
        res["estimates"][a]["detection_at_realised_fpr_le_x"] = {f"{x:.4f}": est[a]["det_at"][i] for i, x in enumerate(XS)}
        res["estimates"][a]["own_threshold_fpr7"] = own_threshold(full, a)
        for k in ("detection", "fpr_c1_once", "fpr_both"):
            res["estimates"][a]["own_threshold_fpr7"][k + "_ci"] = pct_ci([o[k] for o in own_b[a]])
    pairs = [("I2b", "rate_rule"), ("unified", "rate_rule"), ("unified", "I2b")]
    res["paired"] = {}
    for a, b in pairs:
        r = {}
        for m in ("pauc_norm", "pauc_mcclish", "auc"):
            d = np.array([x[m] - y[m] for x, y in zip(boots[a], boots[b])])
            r[m] = {"delta": est[a][m] - est[b][m], "ci95": pct_ci(d), "p_tail_one_sided": p_one_sided(d),
                    "p_bonferroni_over_screening": {str(mm): min(1.0, p_one_sided(d) * mm) for mm in (9, 11, 13)}}
        r["detection_at_x"] = {}
        for i, x in enumerate(XS):
            d = np.array([u["det_at"][i] - v["det_at"][i] for u, v in zip(boots[a], boots[b])])
            r["detection_at_x"][f"{x:.4f}"] = {"delta": est[a]["det_at"][i] - est[b]["det_at"][i], "ci95": pct_ci(d), "p_tail_one_sided": p_one_sided(d)}
        d = np.array([x - y for x, y in zip((o["detection"] for o in own_b[a]), (o["detection"] for o in own_b[b]))])
        r["detection_at_own_threshold"] = {"delta": res["estimates"][a]["own_threshold_fpr7"]["detection"] - res["estimates"][b]["own_threshold_fpr7"]["detection"], "ci95": pct_ci(d)}
        d = np.array([x - y for x, y in zip((o["fpr_c1_once"] for o in own_b[a]), (o["fpr_c1_once"] for o in own_b[b]))])
        r["realised_fpr_at_own_threshold_c1_once_delta"] = {"delta": res["estimates"][a]["own_threshold_fpr7"]["fpr_c1_once"] - res["estimates"][b]["own_threshold_fpr7"]["fpr_c1_once"], "ci95": pct_ci(d)}
        # decision rule: CI of pAUC excludes 0 and detection positive at every matched FPR
        r["rule_robust_to_operating_point"] = bool(r["pauc_norm"]["ci95"][0] > 0 and all(v["delta"] > 0 for v in r["detection_at_x"].values()))
        res["paired"][f"{a}_minus_{b}"] = r
    # cascade
    cpt = cascade_point(full)
    fx = cpt["fpr_c1_once"] if null != "c2" else cpt["fpr_c2_once"]
    z = Z["I2b"]
    pos = rep_mean(z["fam"])
    neg = rep_mean(z[null]) if null != "both" else np.r_[rep_mean(z["c1"]), rep_mean(z["c2"])]
    i2b_here = det_at_fpr(pos, neg, fx)
    d = np.array([c["detection"] - v for c, v in zip(cas_b, i2b_at_cas)])
    res["cascade"] = {"point": cpt, "point_ci": {k: pct_ci([c[k] for c in cas_b]) for k in ("detection", "fpr_c1_once", "fpr_both")},
                      "i2b_detection_at_cascade_fpr": i2b_here, "cascade_minus_i2b_at_matched_fpr": {"delta": cpt["detection"] - i2b_here, "ci95": pct_ci(d)}}
    return res, boots


def curve_table(null="c1"):
    n = 56
    rng = np.random.default_rng(SEED + 7)
    full = np.arange(n)
    est = {a: stats_for(full, a, null)["det_grid"] for a in ARMS}
    bs = {a: [] for a in ARMS}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        for a in ARMS:
            bs[a].append(stats_for(idx, a, null)["det_grid"])
    rows = []
    for a in ARMS:
        bb = np.array(bs[a])
        for i, g in enumerate(GRID):
            rows.append({"arm": a, "fpr_le": g, "detection": est[a][i], "lo": np.percentile(bb[:, i], 2.5), "hi": np.percentile(bb[:, i], 97.5)})
    return pd.DataFrame(rows)


def make_card(out):
    e = out["primary_c1"]
    pa = e["paired"]
    c = e["cascade"]
    E = e["estimates"]
    k = pa["I2b_minus_rate_rule"]
    u = pa["unified_minus_rate_rule"]
    para = (f"Chen, 56 participants, control-vs-control comparison counted once (z_c1). Partial AUC over FPR 0 to 10% (normalised, chance 0.05): rate rule {E['rate_rule']['pauc_norm']:.3f} "
            f"{ci_u(E['rate_rule']['pauc_norm_ci'])}, I2b {E['I2b']['pauc_norm']:.3f} {ci_u(E['I2b']['pauc_norm_ci'])}, unified {E['unified']['pauc_norm']:.3f} {ci_u(E['unified']['pauc_norm_ci'])}. "
            f"Headline: pAUC I2b minus rate rule = {k['pauc_norm']['delta']:+.3f} {ci_s(k['pauc_norm']['ci95'])} (tail p {k['pauc_norm']['p_tail_one_sided']:.4f}; times 13 screened variants {k['pauc_norm']['p_bonferroni_over_screening']['13']:.3f}); "
            f"unified minus rate rule = {u['pauc_norm']['delta']:+.3f} {ci_s(u['pauc_norm']['ci95'])}; full AUC (c1 once) rate rule {E['rate_rule']['auc']:.3f}, I2b {E['I2b']['auc']:.3f}, unified {E['unified']['auc']:.3f}. "
            f"Detection difference at matched realised FPR (I2b minus rate rule) at 1.8% / 7.1% / 14.3%: {k['detection_at_x']['0.0179']['delta']:+.3f} / {k['detection_at_x']['0.0714']['delta']:+.3f} / {k['detection_at_x']['0.1429']['delta']:+.3f}, positive at every FPR in the grid. "
            f"Rule 'robust to operating point' (pAUC CI above 0 and positive detection difference at every matched FPR): I2b {k['rule_robust_to_operating_point']}, unified {u['rule_robust_to_operating_point']}. "
            f"Choice of the single control direction matters: with z_c2 instead of z_c1 the I2b-minus-rate-rule pAUC difference is {out['sensitivity_c2']['paired']['I2b_minus_rate_rule']['pauc_norm']['delta']:+.3f} {ci_s(out['sensitivity_c2']['paired']['I2b_minus_rate_rule']['pauc_norm']['ci95'])} "
            f"(rule satisfied: {out['sensitivity_c2']['paired']['I2b_minus_rate_rule']['rule_robust_to_operating_point']}), and with the original double-counted pool {out['sensitivity_both']['paired']['I2b_minus_rate_rule']['pauc_norm']['delta']:+.3f} {ci_s(out['sensitivity_both']['paired']['I2b_minus_rate_rule']['pauc_norm']['ci95'])} "
            f"(rule satisfied: {out['sensitivity_both']['paired']['I2b_minus_rate_rule']['rule_robust_to_operating_point']}); the per-repeat version gives {out['per_repeat_pauc_norm_mean']['I2b_minus_rate_rule']['delta']:+.3f} {ci_s(out['per_repeat_pauc_norm_mean']['I2b_minus_rate_rule']['ci95'])}. "
            f"Cascade point: detection {c['point']['detection']:.3f} at realised FPR {c['point']['fpr_c1_once']:.3f}; I2b at that FPR {c['i2b_detection_at_cascade_fpr']:.3f} (difference {c['cascade_minus_i2b_at_matched_fpr']['delta']:+.3f} {ci_s(c['cascade_minus_i2b_at_matched_fpr']['ci95'])}). "
            f"Confirmatory p .0105 times 9/11/13 = {out['bonferroni_over_screening']['m_candidates']['9']:.3f}/{out['bonferroni_over_screening']['m_candidates']['11']:.3f}/{out['bonferroni_over_screening']['m_candidates']['13']:.3f}: "
            "the confirmatory detection p does not survive treating the selection of I2b from the 13 screened variants as part of the test. "
            f"The pAUC tail p times 13 is {k['pauc_norm']['p_bonferroni_over_screening']['13']:.3f} with z_c1 but {out['sensitivity_c2']['paired']['I2b_minus_rate_rule']['pauc_norm']['p_bonferroni_over_screening']['13']:.3f} with z_c2, "
            "so the size of the gain depends on which of the two mirror-image control comparisons is counted (the null holds only 56 independent values).")
    save_card("A1", para)
    log(para)


if __name__ == "__main__" and "--card-only" in sys.argv:
    import json as _j
    make_card(_j.loads((OUT / "results.json").read_text())["A1"])
    sys.exit(0)

if __name__ == "__main__":
    out = {"definition": {"null_primary": "z_c1 counted once per participant (z_c1 = -z_c2)", "n": 56, "B": B, "seed": SEED}}
    log("primary (c1)")
    out["primary_c1"], _ = run("c1")
    log("sensitivity c2")
    out["sensitivity_c2"], _ = run("c2")
    log("sensitivity both (original double-counted pool)")
    out["sensitivity_both"], _ = run("both")
    # per-repeat sensitivity
    full = np.arange(56)
    rng = np.random.default_rng(SEED + 3)
    pr = {a: stats_for(full, a, "c1", per_repeat=True) for a in ARMS}
    bs = {a: [] for a in ARMS}
    for _ in range(B):
        idx = rng.integers(0, 56, 56)
        for a in ARMS:
            bs[a].append(stats_for(idx, a, "c1", per_repeat=True))
    out["per_repeat_pauc_norm_mean"] = {"estimates": pr, "I2b_minus_rate_rule": {"delta": pr["I2b"] - pr["rate_rule"], "ci95": pct_ci(np.array(bs["I2b"]) - np.array(bs["rate_rule"]))},
                                        "unified_minus_rate_rule": {"delta": pr["unified"] - pr["rate_rule"], "ci95": pct_ci(np.array(bs["unified"]) - np.array(bs["rate_rule"]))}}
    p_conf = 0.010494752623688156
    out["bonferroni_over_screening"] = {"confirmatory_p_one_sided_fpr7_detection": p_conf,
                                        "m_candidates": {"9": min(1, p_conf * 9), "11": min(1, p_conf * 11), "13": min(1, p_conf * 13)},
                                        "note": "bound, not a test of the selection; 13 = rows of phase2/chen/ideas_table.csv, 11 without the two baselines"}
    ct = curve_table("c1")
    ct.to_csv(OUT / "figdata_A1_chen_detection_vs_fpr.csv", index=False)
    rows = []
    for a in ARMS:
        for x in XS:
            e = out["primary_c1"]["estimates"][a]["detection_at_realised_fpr_le_x"][f"{x:.4f}"]
            rows.append({"arm": a, "fpr_le": x, "detection": e})
    for pair, r in out["primary_c1"]["paired"].items():
        for xk, v in r["detection_at_x"].items():
            rows.append({"arm": pair, "fpr_le": float(xk), "detection": v["delta"], "lo": v["ci95"][0], "hi": v["ci95"][1], "kind": "delta"})
    pd.DataFrame(rows).to_csv(OUT / "figdata_A1_matched_fpr_table.csv", index=False)
    save_section("A1", out)
    make_card(out)
