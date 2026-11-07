"""Phase 7 Track E, E2 (conservative threshold) and E3 (Chen corrected intervals). Definitions: results/phase7/trackE/PREREG.md.

    .venv/Scripts/python.exe -I scripts/phase7_trackE_e2e3.py --stage e2|e3|all [--smoke DIR]

No model is refitted. E2 re-applies the Phase 4 person-level statistic code to the stored inner out-of-fold scores
(checkpoints) to recover each fold's training negatives, checks that the nominal threshold is reproduced exactly, then adds the
Clopper-Pearson-conservative threshold. E3 builds per-fold Chen deltas (detection and person AUC) from stored checkpoints.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta, rankdata, t as tdist  # noqa: E402

from edd import unified as U  # noqa: E402

OOF = ROOT / "results" / "phase4" / "unified" / "oof"
CK = ROOT / "results" / "phase4" / "unified" / "checkpoints"
OUT = ROOT / "results" / "phase7" / "trackE"
TARGET = 0.07
ALPHA = 0.05            # one-sided 95% bound
B_BOOT = 2000
SEED_E2, SEED_E3 = 20261301, 20261302

# task -> (person function, role)
TASKS = {
    **{f"delmas_{k}": ("cit", "primary") for k in ("64_g1", "64_g2", "64_g3", "32_g3", "32_g4", "32_g5")},
    "chen_unified": ("chen", "primary"),
    "celniak_unified": ("mean", "primary"),
    "celniak_ctrl_valcov": ("mean", "control"),
    "celniak_ref_valonly": ("mean", "control"),
    "preti_A_ocular": ("mean", "primary"),
    "preti_B_ocular_rt": ("mean", "primary"),
    "preti_A_contrast_only": ("mean", "secondary"),
    "preti_B_contrast_only": ("mean", "secondary"),
    "mazza_g2_vs_g3": ("mean", "primary"),
    "mazza_guilty_vs_innocent": ("mean", "primary"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- statistics helpers
def cp_upper(k, m, alpha=ALPHA):
    """One-sided (1 - alpha) Clopper-Pearson upper bound for k events in m trials."""
    return 1.0 if k >= m else float(beta.ppf(1 - alpha, k + 1, m - k))


def cp_two_sided(k, n, level=0.95):
    if n == 0:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return (lo, hi)


def kstar(m, target=TARGET):
    """Largest k with CP upper bound <= target; -1 if none (infeasible)."""
    k = -1
    for j in range(0, m + 1):
        if cp_upper(j, m) <= target:
            k = j
        else:
            break
    return k


def tau_conservative(neg, target=TARGET):
    """Smallest tau with #{neg > tau} <= k*, i.e. the k*-th largest negative (0-indexed, descending). +inf if infeasible."""
    z = np.sort(np.asarray(neg, float)[np.isfinite(neg)])[::-1]
    m = len(z)
    k = kstar(m, target)
    if k < 0:
        return float("inf"), m, k
    if m <= k:
        return float("-inf"), m, k
    return float(z[k]), m, k


def tau_nominal(neg, target=TARGET):
    z = np.sort(np.asarray(neg, float)[np.isfinite(neg)])[::-1]
    k = int(np.floor(target * len(z)))
    return float(z[k]) if len(z) > k else float("inf")


# ----------------------------------------------------------------------------- E2: recover training negatives
class _Rec:
    """Monkeypatch edd.unified.thresholds to record the (pos, neg) passed in; the returned dict is unchanged."""
    def __init__(self):
        self.calls = []
        self.orig = U.thresholds

    def __call__(self, pos, neg, target=TARGET):
        self.calls.append((np.asarray(pos, float).copy(), np.asarray(neg, float).copy()))
        return self.orig(pos, neg, target)


def full_arrays(task):
    rows = pd.read_csv(OOF / f"{task}_rows.csv.gz")
    r0 = rows[rows["repeat"] == rows["repeat"].min()]
    n = int(r0.row.max()) + 1
    assert len(r0) == n and r0.row.nunique() == n, f"{task}: rows do not cover every index once"
    r0 = r0.sort_values("row")
    out = {"y": r0.y.to_numpy(int), "g": r0.unit.astype(str).to_numpy()}
    if "T1" in r0:
        out["T1"] = r0.T1.to_numpy(int)
        out["exps"] = r0.exp.to_numpy(int)
        out["g"] = r0.unit.astype(int).astype(str).to_numpy()
    return out


def recover(task, person, repeats, stored):
    """Per (repeat, fold[, exp]) -> dict(neg=training negative stats, tau_nom, tau_stored). Uses the stored inner scores."""
    arr = full_arrays(task)
    rec = _Rec()
    U.thresholds = rec
    recs = []
    try:
        for r in repeats:
            folds = pickle.load(open(CK / f"{task}__f100__r{r}.pkl", "rb"))
            for f, d in enumerate(folds):
                rec.calls.clear()
                res = {r: [d]}
                if person == "mean":
                    pt = U.persons_mean(arr, res)
                elif person == "cit":
                    pt = U.persons_cit(arr, res)
                else:
                    pt = U.persons_chen(arr, res)
                pt = pt.drop(columns=[c for c in ("train_pos", "train_null") if c in pt])
                if person == "chen":
                    exps = sorted(pt.exp.unique())
                    assert len(rec.calls) == len(exps)
                    for e, (pos, neg) in zip(exps, rec.calls):
                        tau_st = float(pt[pt.exp == e].tau_fpr7.iloc[0])
                        recs.append({"repeat": r, "fold": f, "exp": e, "neg": neg, "pos": pos, "tau_stored_recomputed": tau_st})
                else:
                    assert len(rec.calls) == 1
                    pos, neg = rec.calls[0]
                    recs.append({"repeat": r, "fold": f, "exp": 0, "neg": neg, "pos": pos,
                                 "tau_stored_recomputed": float(pt.tau_fpr7.iloc[0])})
    finally:
        U.thresholds = rec.orig
    return recs


def fold_thresholds(task, person, persons):
    repeats = sorted(persons["repeat"].unique())
    recs = recover(task, person, repeats, persons)
    rows = []
    for rc in recs:
        neg = rc["neg"]
        if person == "chen":
            variants = {"conservative": neg, "conservative_c1only": neg[0::2]}
        else:
            variants = {"conservative": neg}
        row = {"task": task, "repeat": rc["repeat"], "fold": rc["fold"], "exp": rc["exp"], "m_train_neg": int(np.isfinite(neg).sum()),
               "m_train_pos": int(np.isfinite(rc["pos"]).sum()), "tau_nominal": tau_nominal(neg)}
        for name, nn in variants.items():
            tau, m, k = tau_conservative(nn)
            row[f"tau_{name}"] = tau
            row[f"m_{name}"] = m
            row[f"kstar_{name}"] = k
            row[f"cp_upper_k0_{name}"] = cp_upper(0, m)
            row[f"feasible_{name}"] = bool(k >= 0)
        row["tau_stored_from_oof"] = rc["tau_stored_recomputed"]
        rows.append(row)
    return pd.DataFrame(rows)


def check_nominal(ft, persons):
    """Recomputed nominal tau == stored tau_fpr7 of the persons file, for every (repeat, fold[, exp])."""
    key = ["repeat", "fold"] + (["exp"] if "exp" in persons else [])
    st = persons.groupby(key, as_index=False).tau_fpr7.first()
    f = ft.copy()
    if "exp" not in persons:
        f = f.drop(columns="exp")
    m = f.merge(st, on=key, how="left")
    d = np.abs(m.tau_nominal - m.tau_fpr7)
    d2 = np.abs(m.tau_nominal - m.tau_stored_from_oof)
    both_inf = np.isinf(m.tau_nominal) & np.isinf(m.tau_fpr7) & (np.sign(m.tau_nominal) == np.sign(m.tau_fpr7))
    ok = bool(((d <= 1e-9) | both_inf).all()) and len(m) == len(st) and not m.tau_fpr7.isna().any()
    return ok, float(np.nanmax(np.where(both_inf, 0, d))), int(len(m))


def evaluate(persons, ft, variant, subset=None, rng=None, nboot=B_BOOT):
    """Detection and realised FPR with unit-level CP and cluster-bootstrap intervals for the threshold column `tau_<variant>`."""
    key = ["repeat", "fold"] + (["exp"] if "exp" in persons else [])
    f = ft.drop(columns="exp") if "exp" not in persons else ft
    tcol = {"nominal": "tau_nominal", "conservative": "tau_conservative", "conservative_c1only": "tau_conservative_c1only"}[variant]
    d = persons.merge(f[key + [tcol]], on=key, how="left")
    if subset is not None:
        d = d[subset(d)]
    d = d[np.isfinite(d.stat)]
    d["dec"] = (d.stat.to_numpy() > d[tcol].to_numpy()).astype(float)
    d["unit"] = d.unit.astype(str)
    out = {}
    unit_means = {}
    for role, yv in (("det", 1), ("fpr", 0)):
        s = d[d.y == yv].groupby(["unit", "repeat"]).dec.mean().groupby("unit").mean()
        unit_means[role] = s
        n = len(s)
        rate = float(s.mean()) if n else np.nan
        k = int(round(rate * n)) if n else 0
        lo, hi = cp_two_sided(k, n)
        out[role] = rate
        out[f"{role}_cp_lo"], out[f"{role}_cp_hi"] = lo, hi
        out[f"n_{'pos' if role == 'det' else 'neg'}_units"] = n
    units = sorted(set(d.unit))
    dm = pd.DataFrame({"det": unit_means["det"], "fpr": unit_means["fpr"]}).reindex(units)
    arr = dm.to_numpy()
    n = len(units)
    bd, bf = [], []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        a = arr[idx]
        bd.append(np.nanmean(a[:, 0]) if np.isfinite(a[:, 0]).any() else np.nan)
        bf.append(np.nanmean(a[:, 1]) if np.isfinite(a[:, 1]).any() else np.nan)
    for nm, v in (("det", bd), ("fpr", bf)):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        out[f"{nm}_boot_lo"], out[f"{nm}_boot_hi"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    return out


def run_e2(out_dir, tasks, nboot, repeats_limit=None):
    rng = np.random.default_rng(SEED_E2)
    all_rows, fold_rows, checks = [], [], {}
    for task in tasks:
        person, role = TASKS[task]
        persons = pd.read_csv(OOF / f"{task}_persons.csv.gz")
        if repeats_limit:
            persons = persons[persons["repeat"] < repeats_limit]
        t0 = time.time()
        ft = fold_thresholds(task, person, persons)
        ok, maxdiff, nf = check_nominal(ft, persons)
        checks[task] = {"nominal_reproduced": ok, "max_abs_diff": maxdiff, "n_fold_keys": nf}
        log(f"E2 {task}: nominal tau reproduced={ok} (max diff {maxdiff:.2e}, {nf} keys) in {time.time() - t0:.0f}s")
        fold_rows.append(ft)
        if not ok:
            continue
        subsets = {"all": None}
        if person == "chen":
            subsets = {"all": None, "exp1": lambda d: d.exp == 1, "exp2": lambda d: d.exp == 2}
        variants = ["nominal", "conservative"] + (["conservative_c1only"] if person == "chen" else [])
        for sname, sub in subsets.items():
            for v in variants:
                e = evaluate(persons, ft, v, sub, rng, nboot)
                f_ = ft if sname == "all" else ft[ft.exp == int(sname[-1])]
                vv = "conservative" if v == "nominal" else v
                feas = f_[f"feasible_{vv}"] if v != "nominal" else pd.Series([True] * len(f_))
                row = {"task": task, "subset": sname, "role": role, "threshold": v, "n_folds": int(len(f_)),
                       "folds_feasible": int(feas.sum()),
                       "mean_m_train_neg": float(f_.m_train_neg.mean()),
                       "mean_kstar": float(f_[f"kstar_{vv}"].mean()) if v != "nominal" else float(np.floor(TARGET * f_.m_train_neg).mean()),
                       "cp_upper_at_k0": cp_upper(0, int(round(f_.m_train_neg.mean()))),
                       "mean_tau": float(np.mean(np.where(np.isfinite(f_["tau_" + ("nominal" if v == "nominal" else v)]), f_["tau_" + ("nominal" if v == "nominal" else v)], np.nan))
                                         ) if np.isfinite(f_["tau_" + ("nominal" if v == "nominal" else v)]).any() else np.nan,
                       "infeasible_all_folds": bool(v != "nominal" and feas.sum() == 0), **e}
                all_rows.append(row)
    res = pd.DataFrame(all_rows)
    fo = pd.concat(fold_rows, ignore_index=True)
    return res, fo, checks


# ----------------------------------------------------------------------------- E3
def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def nb(d, rho):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    J = len(d)
    m, s2 = float(d.mean()), float(d.var(ddof=1))
    out = {"J": J, "mean_delta": m, "sd_fold_deltas": float(np.sqrt(s2)), "rho_ntest_over_ntrain": float(rho)}
    for lab, var in (("naive_1_over_J", s2 / J), ("corrected_NB_BF", (1 / J + rho) * s2)):
        se = float(np.sqrt(var))
        tc = tdist.ppf(0.975, J - 1)
        tt = m / se if se > 0 else np.nan
        out[lab] = {"se": se, "lo": m - tc * se, "hi": m + tc * se, "t": float(tt), "p_two_sided": float(2 * tdist.sf(abs(tt), J - 1)),
                    "p_one_sided_delta_gt_0": float(tdist.sf(tt, J - 1))}
    return out


def across_repeat(tab):
    m = tab.groupby("repeat").d.mean().to_numpy()
    R = len(m)
    se = float(m.std(ddof=1) / np.sqrt(R))
    tc = tdist.ppf(0.975, R - 1)
    return {"R": R, "mean": float(m.mean()), "lo": float(m.mean() - tc * se), "hi": float(m.mean() + tc * se), "se": se}


def run_e3(nboot):
    cps = pd.concat([pickle.load(open(ROOT / "results" / "phase3" / "chen_confirm" / "checkpoints" / f"repeat{r}.pkl", "rb"))["persons"]
                     for r in range(3)], ignore_index=True)
    uni = pd.read_csv(OOF / "chen_unified_persons.csv.gz")
    rs = cps[cps.arm == "rate_rule"].assign(unit=lambda d: d.unit.astype(str))
    i2 = cps[cps.arm == "I2b"].assign(unit=lambda d: d.unit.astype(str))
    uni = uni.assign(unit=uni.unit.astype(str))
    N = rs.unit.nunique()
    # per-unit per-repeat table
    key = ["repeat", "unit"]
    base = rs[key + ["fold", "z_fam", "z_c1", "z_c2", "detect_cal"]].rename(columns={"z_fam": "rf", "z_c1": "r1", "z_c2": "r2", "detect_cal": "rdet"})
    i2b = i2[key + ["fold", "z_fam", "z_c1", "z_c2", "detect_cal"]].rename(columns={"z_fam": "if_", "z_c1": "i1", "z_c2": "i2", "detect_cal": "idet"})
    up = uni.pivot_table(index=key, columns="kind", values="stat").reset_index().rename(columns={"fam": "uf", "c1": "u1", "c2": "u2"})
    ud = uni[uni.kind == "fam"].assign(udet=lambda d: (d.stat > d.tau_fpr7).astype(float))[key + ["udet", "fold"]].rename(columns={"fold": "fold_u"})
    m = base.merge(i2b, on=key, suffixes=("", "_i")).merge(up, on=key).merge(ud, on=key)
    assert (m.fold == m.fold_i).all() and (m.fold == m.fold_u).all(), "fold assignment differs between arms"
    assert len(m) == 3 * N
    arms = {"unified": ("uf", "u1", "u2", "udet"), "I2b": ("if_", "i1", "i2", "idet"), "rate_rule": ("rf", "r1", "r2", "rdet")}
    comps = [("unified_minus_rate_rule", "unified", "rate_rule"), ("I2b_minus_rate_rule", "I2b", "rate_rule"), ("unified_minus_I2b", "unified", "I2b")]
    per_fold, res, tabs = {}, {}, []
    for (r, f), d in m.groupby(["repeat", "fold"]):
        n_te = len(d)
        row = {"repeat": r, "fold": f, "n_test": n_te, "ratio": n_te / (N - n_te)}
        for an, (zf, z1, z2, det) in arms.items():
            row[f"det_{an}"] = d[det].mean()
            y = np.r_[np.ones(n_te), np.zeros(2 * n_te)]
            row[f"auc_{an}"] = auc(y, np.r_[d[zf], d[z1], d[z2]])
            row[f"auc_c1only_{an}"] = auc(y[: 2 * n_te], np.r_[d[zf], d[z1]])
        per_fold[(r, f)] = row
    pf = pd.DataFrame(per_fold.values())
    stored = json.loads((ROOT / "results" / "phase6" / "trackA" / "results.json").read_text())["A8"]["tests"]
    p3 = json.loads((ROOT / "results" / "phase3" / "chen_confirm" / "results.json").read_text())
    p4 = json.loads((ROOT / "results" / "phase4" / "unified" / "results.json").read_text())["primary_tests"]["chen_detection_fpr7"]
    rows, deltas = [], []
    for cname, a, b in comps:
        for metric, col in (("detection", "det"), ("person_auc", "auc"), ("person_auc_c1only", "auc_c1only")):
            tab = pf[["repeat", "fold", "ratio"]].assign(d=pf[f"{col}_{a}"] - pf[f"{col}_{b}"], metric=metric, comparison=cname)
            ok = tab.dropna(subset=["d"])
            e = nb(ok.d, tab.ratio.mean())
            e["across_repeat"] = across_repeat(ok)
            e["folds_dropped"] = int(tab.d.isna().sum())
            res[f"{cname}|{metric}"] = e
            deltas.append(tab)
            row = {"comparison": cname, "metric": metric, "J": e["J"], "mean_per_fold_delta": e["mean_delta"], "sd_fold_deltas": e["sd_fold_deltas"],
                   "rho": e["rho_ntest_over_ntrain"], "naive_lo": e["naive_1_over_J"]["lo"], "naive_hi": e["naive_1_over_J"]["hi"],
                   "corrected_lo": e["corrected_NB_BF"]["lo"], "corrected_hi": e["corrected_NB_BF"]["hi"],
                   "corrected_se": e["corrected_NB_BF"]["se"], "corrected_p_two_sided": e["corrected_NB_BF"]["p_two_sided"],
                   "across_repeat_lo": e["across_repeat"]["lo"], "across_repeat_hi": e["across_repeat"]["hi"], "folds_dropped": e["folds_dropped"]}
            rows.append(row)
    # check detection against Track A8
    chk = {}
    for cname in ("unified_minus_rate_rule", "I2b_minus_rate_rule"):
        s = stored[f"chen_detection_{cname}"]
        e = res[f"{cname}|detection"]
        chk[cname] = {"A8_mean": s["mean_delta"], "A8_corrected_ci": s["corrected_NB_BF"]["ci95"], "rederived_mean": e["mean_delta"],
                      "rederived_corrected_ci": [e["corrected_NB_BF"]["lo"], e["corrected_NB_BF"]["hi"]],
                      "max_abs_diff": float(max(abs(s["mean_delta"] - e["mean_delta"]), abs(s["corrected_NB_BF"]["ci95"][0] - e["corrected_NB_BF"]["lo"]),
                                                abs(s["corrected_NB_BF"]["ci95"][1] - e["corrected_NB_BF"]["hi"])))}
    # bootstrap companion: pooled person AUC (repeat-averaged), participant cluster
    ra = m.groupby("unit")[["rf", "r1", "r2", "if_", "i1", "i2", "uf", "u1", "u2"]].mean()
    units = ra.index.to_numpy()
    rng = np.random.default_rng(SEED_E3)

    def pooled_auc(df, f_, c1, c2):
        y = np.r_[np.ones(len(df)), np.zeros(2 * len(df))]
        return auc(y, np.r_[df[f_], df[c1], df[c2]])

    pt = {a: pooled_auc(ra, *cols[:3]) for a, cols in arms.items()}
    boots = {c[0]: [] for c in comps}
    for _ in range(nboot):
        idx = rng.integers(0, len(units), len(units))
        s = ra.iloc[idx]
        v = {a: pooled_auc(s, *cols[:3]) for a, cols in arms.items()}
        for cname, a, b in comps:
            boots[cname].append(v[a] - v[b])
    boot = {}
    for cname, a, b in comps:
        x = np.asarray(boots[cname])
        boot[cname] = {"delta": pt[a] - pt[b], "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5)), "auc_a": pt[a], "auc_b": pt[b]}
    quoted = {
        "I2b_minus_rate_rule_phase3_bootstrap": {k: p3["paired_I2b_minus_rate_rule"][k] for k in ("det_cal", "person_auc") if k in p3["paired_I2b_minus_rate_rule"]},
        "unified_minus_rate_rule_detection_bootstrap_phase4": {"delta": p4["detection"]["delta_vs_rate_rule"], "ci95": p4["detection"]["ci95"]},
    }
    return {"per_fold": pf, "deltas": pd.concat(deltas), "rows": pd.DataFrame(rows), "res": res, "check_vs_A8": chk, "person_auc_bootstrap": boot,
            "quoted": quoted, "n_units": int(N)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["e2", "e3", "all"])
    ap.add_argument("--smoke", default=None, help="scratch dir: 2 tasks, 1 repeat, few bootstrap draws")
    a = ap.parse_args()
    out = Path(a.smoke) if a.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    nboot = 50 if a.smoke else B_BOOT
    R = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    if a.stage in ("e2", "all"):
        tasks = ["delmas_64_g1", "preti_A_ocular", "chen_unified"] if a.smoke else list(TASKS)
        res, fo, checks = run_e2(out, tasks, nboot, repeats_limit=1 if a.smoke else None)
        res.to_csv(out / "E2_conservative_threshold.csv", index=False)
        fo.to_csv(out / "E2_per_fold_thresholds.csv", index=False)
        R["E2"] = {"nominal_check": checks, "rows": json.loads(res.to_json(orient="records")),
                   "rule": "k* = largest k with Clopper-Pearson one-sided 95% upper bound U(k, m) <= 0.07; tau = k*-th largest training negative; infeasible (flags nobody) when U(0, m) > 0.07 (m < 42)",
                   "min_m_feasible_k0": int(next(m for m in range(1, 500) if cp_upper(0, m) <= TARGET))}
        log("E2 written")
    if a.stage in ("e3", "all"):
        e3 = run_e3(nboot)
        e3["rows"].to_csv(out / "E3_chen_corrected_intervals.csv", index=False)
        e3["deltas"].to_csv(out / "E3_per_fold_deltas.csv", index=False)
        e3["per_fold"].to_csv(out / "E3_per_fold_metrics.csv", index=False)
        R["E3"] = {k: v for k, v in e3.items() if k not in ("per_fold", "deltas", "rows")}
        R["E3"]["rows"] = json.loads(e3["rows"].to_json(orient="records"))
        log("E3 written")
    (out / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else (o.item() if hasattr(o, "item") else str(o))))


if __name__ == "__main__":
    main()
