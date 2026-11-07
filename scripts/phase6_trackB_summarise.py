"""Phase 6 Track B: summarise B1 and B2 from perms/*.jsonl -> results.json, card.md, null CSVs (formulas: PREREG.md)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase6" / "trackB"
P4 = json.loads((ROOT / "results" / "phase4" / "unified" / "results.json").read_text())
TESTS = [("delmas_noninferiority", "delmas"), ("chen_detection_fpr7", "chen"), ("celniak_participant_auc", "celniak"),
         ("preti_A_participant_auc", "preti"), ("preti_B_participant_auc", "preti"),
         ("mazza_g2_vs_g3_participant_auc", "mazza_g2_vs_g3"), ("mazza_guilty_vs_innocent_participant_auc", "mazza_guilty_vs_innocent")]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# D5: only records completed before the laptop sleep (the first record with wall-clock seconds > 20,000 and everything after it is excluded);
# file order = completion order. Counts verified with scripts/_trackB_sleep_gap.py.
PRE = {"chen": 130, "delmas": 124, "mazza_g2_vs_g3": 103, "preti": 193, "celniak": None, "mazza_guilty_vs_innocent": None, "b2": 76}
EXCL = {}


def pre_deadline(recs, key):
    k = PRE.get(key)
    if k is None:
        assert all(r["seconds"] < 20000 for r in recs), key
        return recs
    assert all(r["seconds"] < 20000 for r in recs[:k]) and recs[k]["seconds"] > 20000, key
    return recs[:k]


def load(task):
    f = OUT / "perms" / f"{task}.jsonl"
    recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    n_all = len(recs)
    recs = pre_deadline(recs, task)
    EXCL[task] = {"records_on_disk": n_all, "pre_deadline_records": len(recs), "excluded_post_sleep": n_all - len(recs)}
    byi = {}
    for r in recs:
        byi[r["perm"]] = r
    # D3: only complete blocks of 25 permutation indices are used (a function of the indices only)
    blocks = {}
    for i in byi:
        blocks.setdefault(i // 25, []).append(i)
    keep = sorted(i for b, v in blocks.items() if len(v) == 25 for i in v)
    EXCL[task]["used_complete_blocks"] = len(keep) // 25
    EXCL[task]["excluded_partial_block_records"] = len(byi) - len(keep)
    EXCL[task]["N_used"] = len(keep)
    return [byi[i] for i in keep]


def rng_summary(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "p2.5": float(np.percentile(x, 2.5)), "p50": float(np.percentile(x, 50)),
            "p97.5": float(np.percentile(x, 97.5)), "min": float(x.min()), "max": float(x.max())}


def p_ge(null, obs):
    return float((1 + np.sum(np.asarray(null) >= obs - 1e-12)) / (1 + len(null)))


def p_le(null, obs):
    return float((1 + np.sum(np.asarray(null) <= obs + 1e-12)) / (1 + len(null)))


R0 = json.loads((OUT / "identity_check_repeat0.json").read_text())


def observed(tn):
    """Observed values of the R1 = 1 tasks are the repeat-0 values (identity run, cross-checked with stored repeat-0 scores); Mazza: Phase 4 values."""
    if tn in R0:
        c = R0[tn]["computed"]
        return c
    if tn == "chen_detection_fpr7":
        d = P4["primary_tests"][tn]["detection"]
        return {"base": d["rate_rule"], "ours": d["unified"], "delta": d["unified"] - d["rate_rule"],
                "auc_base": P4["primary_tests"][tn]["person_auc"]["rate_rule"]["auc"],
                "auc_ours": P4["primary_tests"][tn]["person_auc"]["unified"]}
    s = P4["primary_tests"][tn]
    return {"base": s["est_base"], "ours": s["est_unified"], "delta": s["delta"]}


def main():
    R = {"B1": {}, "B2": {}}
    cache = {}
    for tn, task in TESTS:
        if task not in cache:
            cache[task] = load(task)
        recs = cache[task]
        if not recs:
            continue
        rows = []
        for r in recs:
            v = r["tests"][tn]
            rows.append({"perm": r["perm"], **{k: x for k, x in v.items() if not isinstance(x, dict)}})
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"null_b1_{tn}.csv", index=False)
        n = len(df)
        o = observed(tn)
        res = {"N": n, "observed": o,
               "baseline": {"null": rng_summary(df.base), "p_single_upper": p_ge(df.base, o["base"]), "p_single_lower": p_le(df.base, o["base"])},
               "ours": {"null": rng_summary(df.ours), "p_single_upper": p_ge(df.ours, o["ours"]), "p_single_lower": p_le(df.ours, o["ours"])},
               "delta": {"null": rng_summary(df.delta), "p_delta_superiority": p_ge(df.delta, o["delta"]), "p_delta_lower": p_le(df.delta, o["delta"])}}
        if tn == "chen_detection_fpr7":
            res["baseline_person_auc"] = {"observed": o["auc_base"], "null": rng_summary(df.auc_base), "p_single_upper": p_ge(df.auc_base, o["auc_base"]),
                                          "p_single_lower": p_le(df.auc_base, o["auc_base"])}
            res["ours_person_auc"] = {"observed": o["auc_ours"], "null": rng_summary(df.auc_ours), "p_single_upper": p_ge(df.auc_ours, o["auc_ours"]),
                                      "p_single_lower": p_le(df.auc_ours, o["auc_ours"])}
            res["delta_person_auc"] = {"observed": o["auc_ours"] - o["auc_base"], "null": rng_summary(df.auc_delta),
                                       "p": p_ge(df.auc_delta, o["auc_ours"] - o["auc_base"])}
            res["null_realised_fpr"] = {"base": rng_summary(df.fpr_base), "ours": rng_summary(df.fpr_ours)}
        if tn == "delmas_noninferiority":
            res["p_NI_shift"] = float((1 + np.sum(df.delta.to_numpy() >= o["delta"] + 0.02 - 1e-12)) / (1 + n))
            res["lower_bound_delta_minus_1.645sd"] = float(o["delta"] - 1.645 * df.delta.std(ddof=1))
            res["refit_aware_p"] = res["p_NI_shift"]
        else:
            res["refit_aware_p"] = res["delta"]["p_delta_superiority"]
        res["phase4"] = {"p_raw_bootstrap": next(h["p_raw"] for h in P4["holm_family"] if h["test"] == tn),
                         "p_holm_bootstrap": next(h["p_holm"] for h in P4["holm_family"] if h["test"] == tn)}
        R["B1"][tn] = res
    ps = [R["B1"][tn]["refit_aware_p"] for tn, _ in TESTS if tn in R["B1"]]
    if len(ps) == len(TESTS):
        for (tn, _), a in zip(TESTS, holm(ps)):
            R["B1"][tn]["refit_aware_p_holm"] = float(a)

    # ---- B2
    f = OUT / "perms_b2" / "shuffles.jsonl"
    if f.exists():
        raw = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        pre = pre_deadline(raw, "b2")
        d = pd.DataFrame(pre).drop_duplicates("shuffle").sort_values("shuffle")
        EXCL["b2"] = {"records_on_disk": len(raw), "pre_deadline_records": len(pre), "excluded_post_sleep": len(raw) - len(pre)}
        cnt = d.shuffle.floordiv(25).value_counts()
        EXCL["b2"]["excluded_partial_block_records"] = int((d.shuffle.floordiv(25).map(cnt) != 25).sum())
        d = d[d.shuffle.floordiv(25).map(cnt) == 25]
        EXCL["b2"]["N_used"] = len(d)
        d.to_csv(OUT / "null_b2_celniak_grid.csv", index=False)
        ref = pd.read_csv(OUT / "b2_real_label_reference.csv") if (OUT / "b2_real_label_reference.csv").exists() else None
        N = len(d)
        acc_obs, auc_hi, auc_lo = 0.717948717948718, 0.7210526315789474, 0.6710526315789473
        R["B2"] = {"N": N, "real_reference_published_reproduction": {"acc": acc_obs, "auc_seed1": auc_hi, "auc_seed2": auc_lo},
                   "null_best_of_grid_accuracy": rng_summary(d.acc_max), "null_best_of_grid_auc": rng_summary(d.auc_max),
                   "null_best_of_grid_vote_accuracy": rng_summary(d.vote_acc_max), "null_best_of_grid_vote_auc": rng_summary(d.vote_auc_max),
                   "null_grid_median_auc": rng_summary(d.auc_median),
                   "null_p99_acc": float(np.percentile(d.acc_max, 99)), "null_p99_auc": float(np.percentile(d.auc_max, 99)),
                   "p_acc": p_ge(d.acc_max, acc_obs), "p_auc_high_0.7211": p_ge(d.auc_max, auc_hi), "p_auc_low_0.6711": p_ge(d.auc_max, auc_lo)}
        if ref is not None:
            R["B2"]["real_label_pairing_seeds_1_10"] = {"acc_max": rng_summary(ref.acc_max), "auc_max": rng_summary(ref.auc_max),
                                                        "p_acc_median_real": p_ge(d.acc_max, float(ref.acc_max.median())),
                                                        "p_auc_median_real": p_ge(d.auc_max, float(ref.auc_max.median())),
                                                        "table": ref[["pairing_seed", "acc_max", "auc_max", "vote_acc_max", "vote_auc_max"]].to_dict("records")}
    if R["B2"]:
        full = pd.DataFrame(pre_deadline([json.loads(x) for x in (OUT / "perms_b2" / "shuffles.jsonl").read_text().splitlines() if x.strip()], "b2")).drop_duplicates("shuffle")
        R["B2"]["supplement_all_pre_deadline_shuffles_incl_partial_blocks"] = {
            "N": len(full), "null_acc_mean": float(full.acc_max.mean()), "null_acc_p2.5_p97.5": [float(np.percentile(full.acc_max, 2.5)), float(np.percentile(full.acc_max, 97.5))],
            "null_auc_mean": float(full.auc_max.mean()), "null_auc_p2.5_p97.5": [float(np.percentile(full.auc_max, 2.5)), float(np.percentile(full.auc_max, 97.5))],
            "p_acc": p_ge(full.acc_max, 0.717948717948718), "p_auc_0.7211": p_ge(full.auc_max, 0.7210526315789474), "p_auc_0.6711": p_ge(full.auc_max, 0.6710526315789473),
            "note": "descriptive; includes shuffles from incomplete blocks of 25 (D3 primary analysis excludes them)"}
    R["inclusion"] = {"rule": "D3+D5: records completed before the sleep gap only, truncated to complete blocks of 25 permutation indices; excluded records stay on disk", "per_task": EXCL}
    (OUT / "results.json").write_text(json.dumps(R, indent=1))

    # ---- card
    L = ["# Phase 6 Track B card: refit-aware permutation tests (B1) and Celniak best-of-grid null (B2)", "",
         "Definitions, formulas and seeds: `PREREG.md`. p = (1 + #null >= observed) / (1 + N), one-sided as pre-registered; null ranges are 2.5-97.5 percentiles.", "",
         "## B1", "",
         "| Test | N | Baseline obs [null 95%] p_single | Ours obs [null 95%] p_single | Delta obs [null 95%] | refit-aware p (Holm) | Phase 4 bootstrap p (Holm) |", "|---|---|---|---|---|---|---|"]
    for tn, _ in TESTS:
        if tn not in R["B1"]:
            continue
        r = R["B1"][tn]
        o = r["observed"]

        def cell(k, ob):
            return f"{ob:.3f} [{r[k]['null']['p2.5']:.3f}, {r[k]['null']['p97.5']:.3f}] p={r[k]['p_single_upper']:.3f} (low {r[k]['p_single_lower']:.3f})"
        L.append(f"| {tn} | {r['N']} | {cell('baseline', o['base'])} | {cell('ours', o['ours'])} | {o['delta']:+.3f} [{r['delta']['null']['p2.5']:+.3f}, {r['delta']['null']['p97.5']:+.3f}] | "
                 f"{r['refit_aware_p']:.4f} ({r.get('refit_aware_p_holm', float('nan')):.4f}) | {r['phase4']['p_raw_bootstrap']:.4f} ({r['phase4']['p_holm_bootstrap']:.4f}) |")
    if "chen_detection_fpr7" in R["B1"]:
        r = R["B1"]["chen_detection_fpr7"]
        L += ["", f"Chen (detection rate at calibrated FPR <= 7%; N = {r['N']}): person AUC baseline {r['baseline_person_auc']['observed']:.3f} (p {r['baseline_person_auc']['p_single_upper']:.3f}), "
              f"ours {r['ours_person_auc']['observed']:.3f} (p {r['ours_person_auc']['p_single_upper']:.3f}); person-AUC delta p {r['delta_person_auc']['p']:.3f}."]
    if "delmas_noninferiority" in R["B1"]:
        r = R["B1"]["delmas_noninferiority"]
        L += ["", f"Delmas non-inferiority: shift p_NI = {r['p_NI_shift']:.4f}; delta - 1.645 x SD_null = {r['lower_bound_delta_minus_1.645sd']:+.4f} (margin -0.02)."]
    if R["B2"]:
        b = R["B2"]
        L += ["", "## B2 (Celniak best-of-216-grid null)", "",
              f"N = {b['N']} shuffles. Null best-of-grid accuracy: mean {b['null_best_of_grid_accuracy']['mean']:.3f}, 95% range [{b['null_best_of_grid_accuracy']['p2.5']:.3f}, {b['null_best_of_grid_accuracy']['p97.5']:.3f}], max {b['null_best_of_grid_accuracy']['max']:.3f}.",
              f"Null best-of-grid AUC: mean {b['null_best_of_grid_auc']['mean']:.3f}, 95% range [{b['null_best_of_grid_auc']['p2.5']:.3f}, {b['null_best_of_grid_auc']['p97.5']:.3f}], max {b['null_best_of_grid_auc']['max']:.3f}.",
              f"Real reproduction: acc 0.718, AUC 0.671 / 0.721. p_acc = {b['p_acc']:.4f}; p_auc(0.721) = {b['p_auc_high_0.7211']:.4f}; p_auc(0.671) = {b['p_auc_low_0.6711']:.4f}."]
    L += ["", "## Records used (D3 + D5)", "", "| Task | on disk | pre-sleep | excluded post-sleep | excluded partial block | N used | p resolution 1/(N+1) |", "|---|---|---|---|---|---|---|"]
    for k, v in EXCL.items():
        L.append(f"| {k} | {v['records_on_disk']} | {v['pre_deadline_records']} | {v['excluded_post_sleep']} | {v.get('excluded_partial_block_records', 0)} | {v['N_used']} | {1 / (1 + v['N_used']):.4f} |")
    if R["B2"]:
        sp = R["B2"]["supplement_all_pre_deadline_shuffles_incl_partial_blocks"]
        L += ["", f"B2 supplement (all {sp['N']} pre-sleep shuffles, descriptive): p_acc {sp['p_acc']:.4f}, p_auc(0.721) {sp['p_auc_0.7211']:.4f}, p_auc(0.671) {sp['p_auc_0.6711']:.4f}; null acc mean {sp['null_acc_mean']:.3f}, null AUC mean {sp['null_auc_mean']:.3f}."]
    (OUT / "card.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
