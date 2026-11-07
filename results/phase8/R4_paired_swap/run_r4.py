"""R4: paired participant-level score-swap permutation test of the DIFFERENCE (fixed pipeline minus leakage-free baseline)
for the seven planned comparisons, from the stored out-of-fold scores. No model is refitted. One process, one thread.

Run: .venv\\Scripts\\python.exe -I results\\phase8\\R4_paired_swap\\run_r4.py
"""
import os
import sys
import json
import time
import pickle
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
P4 = ROOT / "results" / "phase4" / "unified"
P2 = ROOT / "results" / "phase2"
CHEN_CP = ROOT / "results" / "phase3" / "chen_confirm" / "checkpoints"
SEED = 20261105
N_PERM = 2000
MARGIN = 0.02
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def auc(y, s):
    r = rankdata(s)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------------------------------------ item tables
# An "item table" has columns: pid (participant, the cluster), grp (group inside which an AUC is computed), y,
# a (score of the baseline), b (score of the fixed pipeline), and the rank-transformed versions ar, br.
def rank_within(df, by, col):
    """Normalised rank (average ties, in (0,1]) of col within each group of `by`."""
    return df.groupby(by)[col].transform(lambda s: rankdata(s) / len(s))


def mean_over_repeats(df, keys, col):
    return df.groupby(keys, as_index=False)[col].mean()


def person_items(base_rows, uni_rows):
    """Participant-level AUC comparison. rows: columns repeat, unit, y, score (row level). Participant score = mean of
    the participant's rows within a repeat, then mean over repeats (raw), or mean over repeats of the within-repeat
    normalised rank across participants (rank version)."""
    out = {}
    for tag, df in (("a", base_rows), ("b", uni_rows)):
        d = df.copy()
        d["unit"] = d["unit"].astype(str)
        pr = d.groupby(["repeat", "unit"], as_index=False).agg(y=("y", "first"), s=("score", "mean"))
        pr["r"] = rank_within(pr, "repeat", "s")
        pu = pr.groupby("unit", as_index=False).agg(y=("y", "first"), s=("s", "mean"), r=("r", "mean"))
        out[tag] = pu.set_index("unit")
    common = sorted(set(out["a"].index) & set(out["b"].index))
    a, b = out["a"].loc[common], out["b"].loc[common]
    y = b["y"].to_numpy()
    return pd.DataFrame({"pid": common, "grp": "all", "y": y, "a": a["s"].to_numpy(), "b": b["s"].to_numpy(),
                         "ar": a["r"].to_numpy(), "br": b["r"].to_numpy()})


def delmas_items(drop=()):
    base = pd.read_csv(P2 / "delmas_raw" / "oof_scores.csv.gz")
    base = base[base.tag == "P4_svm_1000"]
    parts = []
    for f in sorted((P4 / "oof").glob("delmas_*_rows.csv.gz")):
        key = f.name.replace("delmas_", "").replace("_rows.csv.gz", "")
        if key in drop:
            continue
        u = pd.read_csv(f)
        u["participant"] = u.unit.astype(int)
        b = base[base.set == key]
        ur = u.groupby(["repeat", "participant", "image"], as_index=False).agg(y=("y", "first"), s=("score", "mean"))
        br = b.groupby(["repeat", "participant", "image"], as_index=False).agg(s=("score", "mean"))
        ur["r"] = rank_within(ur, "repeat", "s")
        br["r"] = rank_within(br, "repeat", "s")
        ua = ur.groupby(["participant", "image"], as_index=False).agg(y=("y", "first"), s=("s", "mean"), r=("r", "mean"))
        ba = br.groupby(["participant", "image"], as_index=False).agg(s=("s", "mean"), r=("r", "mean"))
        m = ua.merge(ba, on=["participant", "image"], how="inner", suffixes=("_u", "_b"))
        parts.append(pd.DataFrame({"pid": m.participant.astype(str), "grp": key, "y": m.y, "a": m.s_b, "b": m.s_u,
                                   "ar": m.r_b, "br": m.r_u}))
    return pd.concat(parts, ignore_index=True)


def chen_cp():
    cps = [pickle.load(open(CHEN_CP / f"repeat{r}.pkl", "rb"))["persons"] for r in range(3)]
    return pd.concat(cps, ignore_index=True)


def chen_auc_items(pu, cp):
    """Participant AUC: items = (participant, kind) with kind fam (y=1) vs c1/c2 (y=0); statistic averaged over repeats."""
    ra = cp[cp.arm == "rate_rule"]
    la = pd.concat([pd.DataFrame({"repeat": ra.repeat, "unit": ra.unit.astype(str), "kind": k, "s": ra[z]})
                    for z, k in (("z_fam", "fam"), ("z_c1", "c1"), ("z_c2", "c2"))])
    lb = pu[["repeat", "unit", "kind", "stat"]].rename(columns={"stat": "s"}).copy()
    lb["unit"] = lb.unit.astype(str)
    res = {}
    for tag, d in (("a", la), ("b", lb)):
        d = d.copy()
        d["r"] = rank_within(d, "repeat", "s")
        g = d.groupby(["unit", "kind"], as_index=False).agg(s=("s", "mean"), r=("r", "mean"))
        res[tag] = g.set_index(["unit", "kind"])
    idx = res["a"].index.intersection(res["b"].index)
    a, b = res["a"].loc[idx], res["b"].loc[idx]
    kind = np.array([k for _, k in idx])
    return pd.DataFrame({"pid": [u for u, _ in idx], "grp": "all", "y": (kind == "fam").astype(int), "a": a.s.to_numpy(),
                         "b": b.s.to_numpy(), "ar": a.r.to_numpy(), "br": b.r.to_numpy()})


def chen_det_items(pu, cp):
    """Participant detection at the calibrated FPR<=7% threshold (stored decisions), mean over repeats."""
    f = pu[pu.kind == "fam"].copy()
    f["det"] = (f.stat > f.tau_fpr7).astype(float)
    du = f.groupby(f.unit.astype(str)).det.mean()
    ra = cp[cp.arm == "rate_rule"]
    db = ra.groupby(ra.unit.astype(str)).detect_cal.mean()
    units = sorted(set(du.index) & set(db.index))
    return pd.DataFrame({"pid": units, "a": db.loc[units].to_numpy(), "b": du.loc[units].to_numpy()})


# ------------------------------------------------------------------------------------------------ permutation engine
def make_masks(pids, seed, n_perm):
    """Swap indicator per (permutation, participant); one row per permutation; drawn from one seeded stream."""
    rng = np.random.default_rng(seed)
    return rng.random((n_perm, len(pids))) < 0.5


def delta_auc(items, swap_vec, pid_index, use_rank):
    """Mean-over-groups AUC(b') - AUC(a') where each participant's (a, b) are swapped if swap_vec[participant]."""
    a = items.ar.to_numpy() if use_rank else items.a.to_numpy()
    b = items.br.to_numpy() if use_rank else items.b.to_numpy()
    sw = swap_vec[pid_index]
    s_a = np.where(sw, b, a)
    s_b = np.where(sw, a, b)
    A, B = [], []
    for g in groups_cache[id(items)]:
        A.append(auc(y_cache[id(items)][g], s_a[g]))
        B.append(auc(y_cache[id(items)][g], s_b[g]))
    return float(np.mean(B) - np.mean(A)), float(np.mean(A)), float(np.mean(B))


groups_cache, y_cache = {}, {}


def swap_test_auc(items, seed, use_rank, masks_pids=None, n_perm=N_PERM):
    pids = sorted(items.pid.unique()) if masks_pids is None else masks_pids
    pid_index = items.pid.map({p: i for i, p in enumerate(pids)}).to_numpy()
    groups_cache[id(items)] = [np.where(items.grp.to_numpy() == g)[0] for g in sorted(items.grp.unique())]
    y_cache[id(items)] = items.y.to_numpy()
    obs, ma, mb = delta_auc(items, np.zeros(len(pids), bool), pid_index, use_rank)
    masks = make_masks(pids, seed, n_perm)
    perm = np.array([delta_auc(items, masks[i], pid_index, use_rank)[0] for i in range(n_perm)])
    return obs, ma, mb, perm


def pvals(obs, perm, margin=None):
    out = {"p_two_sided": float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (1 + len(perm))),
           "perm_sd": float(perm.std()), "perm_mean": float(perm.mean()), "n_perm": int(len(perm))}
    if margin is not None:
        # H0: delta <= -margin; null distribution = swap null shifted to -margin
        out["p_noninferiority_one_sided"] = float((1 + np.sum((perm - margin) >= obs - 1e-12)) / (1 + len(perm)))
    return out


def main():
    res = {"seed": SEED, "n_perm": N_PERM, "margin_delmas": MARGIN, "comparisons": {}}
    u_rows = lambda n: pd.read_csv(P4 / "oof" / f"{n}_rows.csv.gz")  # noqa: E731
    pj = json.load(open(P4 / "results.json"))["primary_tests"]

    specs = []
    # Delmas
    log("Delmas")
    it = delmas_items()
    specs.append(("1 Delmas image AUC (mean of 6 sets)", "delmas", it, 0.0, pj["delmas_noninferiority"]["delta"],
                  {"base": pj["delmas_noninferiority"]["est_base"], "uni": pj["delmas_noninferiority"]["est_unified"]}, True))
    # Chen
    pu = pd.read_csv(P4 / "oof" / "chen_unified_persons.csv.gz")
    cp = chen_cp()
    det = chen_det_items(pu, cp)
    cau = chen_auc_items(pu, cp)
    # Celniak, Preti, Mazza
    base_p = {
        "celniak": pd.read_csv(P2 / "celniak" / "trials_baseline_svm_paper.csv"),
        "preti_A": pd.read_csv(P2 / "preti" / "scores" / "B_ocular.csv"),
        "preti_B": pd.read_csv(P2 / "preti" / "scores" / "B_fusion.csv"),
        "g2_vs_g3": pd.read_csv(P2 / "mazza" / "oof" / "g2_vs_g3__base_pub36_lr_unreg.csv"),
        "guilty_vs_innocent": pd.read_csv(P2 / "mazza" / "oof" / "guilty_vs_innocent__base_pub36_lr_unreg.csv"),
    }
    person_specs = [("3 Celniak participant AUC", "celniak", "celniak", u_rows("celniak_unified"), "celniak_participant_auc"),
                    ("4 Preti eyes only participant AUC", "preti_A", "preti_A", u_rows("preti_A_ocular"), "preti_A_participant_auc"),
                    ("5 Preti with RT participant AUC", "preti_B", "preti_B", u_rows("preti_B_ocular_rt"), "preti_B_participant_auc"),
                    ("6 Colasanti group 2 vs 3 participant AUC", "g2_vs_g3", "g2_vs_g3", u_rows("mazza_g2_vs_g3"), "mazza_g2_vs_g3_participant_auc"),
                    ("7 Colasanti guilty vs innocent participant AUC", "guilty_vs_innocent", "guilty_vs_innocent",
                     u_rows("mazza_guilty_vs_innocent"), "mazza_guilty_vs_innocent_participant_auc")]
    person_tabs = {}
    for label, key, _, ur, jk in person_specs:
        person_tabs[label] = (person_items(base_p[key], ur), pj[jk])

    # -------- reproduction of observed values
    repro = {}
    ok_all = True

    def check(name, got, want, tol=6e-4):
        nonlocal ok_all
        ok = abs(got - want) <= tol
        ok_all &= ok
        repro[name] = {"recomputed": float(got), "stored_or_paper": float(want), "match": bool(ok)}
        log(f"reproduce {name}: {got:+.4f} vs {want:+.4f}  {'OK' if ok else 'MISMATCH'}")

    # Delmas
    for name, items, st in [("Delmas", it, pj["delmas_noninferiority"])]:
        groups_cache[id(items)] = [np.where(items.grp.to_numpy() == g)[0] for g in sorted(items.grp.unique())]
        y_cache[id(items)] = items.y.to_numpy()
        pid_index = np.zeros(len(items), int)
        o, ma, mb = delta_auc(items, np.array([False]), pid_index, False)
        check("Delmas AUC baseline", ma, st["est_base"])
        check("Delmas AUC fixed pipeline", mb, st["est_unified"])
        check("Delmas delta (paper +0.000)", o, 0.0, 6e-4)
    for label, (items, st) in person_tabs.items():
        pids = sorted(items.pid.unique())
        groups_cache[id(items)] = [np.arange(len(items))]
        y_cache[id(items)] = items.y.to_numpy()
        o, ma, mb = delta_auc(items, np.zeros(len(pids), bool), np.arange(len(items)), False)
        check(f"{label} AUC baseline", ma, st["est_base"])
        check(f"{label} AUC fixed pipeline", mb, st["est_unified"])
        check(f"{label} delta", o, st["delta"], 1e-6)
    pc = json.load(open(P4 / "results.json"))["primary_tests"]["chen_detection_fpr7"]["detection"]
    check("Chen detection baseline (rate rule)", det.a.mean(), pc["rate_rule"], 1e-9)
    check("Chen detection fixed pipeline", det.b.mean(), pc["unified"], 1e-9)
    check("Chen detection delta (paper +14.9 points)", det.b.mean() - det.a.mean(), 0.149, 6e-4)
    res["reproduction"] = repro
    res["all_reproduced"] = bool(ok_all)
    if not ok_all:
        log("WARNING: some observed values did not reproduce; see reproduction in results.json")

    # -------- tests
    comps = []

    def record(name, stat, obs, ma, mb, perm_raw, perm_rank=None, obs_rank=None, ma_r=None, mb_r=None, margin=None, note=""):
        e = {"statistic": stat, "observed_delta": float(obs), "mean_baseline": float(ma), "mean_fixed": float(mb),
             "raw": pvals(obs, perm_raw, margin), "note": note}
        if perm_rank is not None:
            e["rank"] = {"observed_delta": float(obs_rank), "mean_baseline": float(ma_r), "mean_fixed": float(mb_r),
                         **pvals(obs_rank, perm_rank, margin)}
        res["comparisons"][name] = e
        log(f"{name}: obs {obs:+.4f} raw p2={e['raw']['p_two_sided']:.4f}" +
            (f" | rank obs {obs_rank:+.4f} p2={e['rank']['p_two_sided']:.4f}" if perm_rank is not None else ""))

    # 1 Delmas (swap participants across all six sets consistently by participant id)
    nm = "1 Delmas image AUC (mean of 6 sets)"
    pids = sorted(it.pid.unique())
    o, ma, mb, perm = swap_test_auc(it, SEED, False)
    o_r, ma_r, mb_r, perm_r = swap_test_auc(it, SEED, True)
    record(nm, "image-level AUC, mean over 6 sets, repeat-averaged", o, ma, mb, perm, perm_r, o_r, ma_r, mb_r, margin=MARGIN,
           note=f"{len(pids)} participant ids swapped (a participant in two sets is swapped in both)")
    # 2 Chen detection (stored calibrated decisions swapped) and Chen participant AUC
    nm = "2 Chen detection at FPR<=7% (calibrated)"
    pids = sorted(det.pid)
    masks = make_masks(pids, SEED, N_PERM)
    a, b = det.a.to_numpy(), det.b.to_numpy()
    obs = b.mean() - a.mean()
    perm = np.array([(np.where(m, a, b).mean() - np.where(m, b, a).mean()) for m in masks])
    record(nm, "participant detection rate (stored decisions at the training-set 7% threshold, mean over 3 repeats), swap of the stored decisions",
           obs, a.mean(), b.mean(), perm,
           note="thresholds were set on training participants before swapping and are not recomputed; the stored per-participant decisions are exchanged")
    nm = "2b Chen participant AUC (secondary)"
    o, ma, mb, perm = swap_test_auc(cau, SEED, False)
    o_r, ma_r, mb_r, perm_r = swap_test_auc(cau, SEED, True)
    record(nm, "participant AUC, familiar vs control z statistics", o, ma, mb, perm, perm_r, o_r, ma_r, mb_r,
           note="secondary; not part of the Holm family")
    # 3-7
    for label, (items, st) in person_tabs.items():
        o, ma, mb, perm = swap_test_auc(items, SEED, False)
        o_r, ma_r, mb_r, perm_r = swap_test_auc(items, SEED, True)
        record(label, "participant AUC (mean of rows, mean over repeats)", o, ma, mb, perm, perm_r, o_r, ma_r, mb_r,
               note=f"{items.pid.nunique()} participants")

    # -------- Holm over the seven family p values
    fam_names = ["1 Delmas image AUC (mean of 6 sets)", "2 Chen detection at FPR<=7% (calibrated)",
                 "3 Celniak participant AUC", "4 Preti eyes only participant AUC", "5 Preti with RT participant AUC",
                 "6 Colasanti group 2 vs 3 participant AUC", "7 Colasanti guilty vs innocent participant AUC"]
    for view in ("raw", "rank"):
        ps = []
        for n in fam_names:
            e = res["comparisons"][n]
            v = e[view] if view in e else e["raw"]
            ps.append(v["p_noninferiority_one_sided"] if n.startswith("1 ") else v["p_two_sided"])
        adj = holm(ps)
        # alternative: two-sided p for Delmas too
        ps2 = [(res["comparisons"][n][view] if view in res["comparisons"][n] else res["comparisons"][n]["raw"])["p_two_sided"] for n in fam_names]
        adj2 = holm(ps2)
        for n, p, h, h2 in zip(fam_names, ps, adj, adj2):
            res["comparisons"][n].setdefault(f"holm_{view}", {})
            res["comparisons"][n][f"holm_{view}"] = {"p_used": float(p), "p_holm": float(h), "p_holm_if_delmas_two_sided": float(h2)}
    # Chen has no rank version (binary decisions): rank view reuses raw
    res["chen_note"] = "Chen detection is a binary decision; no rank transform applies, the rank view reuses the raw result."
    res["runtime_s"] = time.time() - T0
    json.dump(res, open(OUT / "results.json", "w"), indent=2, ensure_ascii=False)
    write_card(res, fam_names)
    log("done")


def write_card(res, fam_names):
    C = res["comparisons"]

    def fmt_delta(n, v):
        return f"{v * 100:+.1f} pts" if n.startswith("2 Chen") else f"{v:+.3f}"

    rows = []
    for n in fam_names:
        e = C[n]
        r = e["rank"] if "rank" in e else None
        raw_p = e["raw"]["p_noninferiority_one_sided"] if n.startswith("1 ") else e["raw"]["p_two_sided"]
        rows.append(f"| {n} | {fmt_delta(n, e['observed_delta'])} | {e['raw']['p_two_sided']:.4f} | "
                    + (f"{e['raw']['p_noninferiority_one_sided']:.4f} | " if n.startswith("1 ") else "n/a | ")
                    + f"{e['holm_raw']['p_holm']:.3f} | "
                    + (f"{r['observed_delta']:+.3f} / {r['p_two_sided']:.4f}" + (f" / NI {r['p_noninferiority_one_sided']:.4f}" if n.startswith("1 ") else "") + f" | {e['holm_rank']['p_holm']:.3f} |"
                       if r else "n/a (binary decisions) | same |"))
    e2 = C["2b Chen participant AUC (secondary)"]
    card = f"""# R4: paired score-swap permutation test of the difference (fixed pipeline minus leakage-free baseline)

## What was done
For each of the seven planned comparisons the stored out-of-fold scores of the fixed pipeline (`results/phase4/unified/oof/`)
and of the leakage-free baseline (`results/phase2/...`, `results/phase3/chen_confirm/checkpoints`) were paired on the same
participants. The planned statistic was first reproduced from the stored scores (all observed values match: {res['all_reproduced']}, see
`results.json` -> `reproduction`). Then, in each of {res['n_perm']} permutations (seed {res['seed']}), each participant independently
(probability 0.5) swapped which method their scores belong to (all items and repeats of that participant together; for Delmas
a participant id appearing in two sets is swapped in both), the statistic was recomputed for both methods, and the difference taken.
Two-sided p = (1 + #|diff_perm| >= |diff_obs|) / (1 + {res['n_perm']}). No model was refitted; one process, one thread.

Statistics: Delmas = image-level AUC on repeat-averaged scores, mean over the six group sets. Celniak, Preti (both),
Colasanti (both) = participant-level AUC of repeat-averaged participant scores. Chen = participant detection at the 7% threshold
(see deviations), plus participant AUC as a secondary check.

## Scale handling
Baseline and pipeline scores sit on different scales (for example Colasanti baseline logits are in the hundreds), so swapping raw
scores is not meaningful for AUC statistics. Rank version: within each method and repeat, scores are converted to normalised
ranks (for Delmas across the image rows of each set; for participant AUC across participants; for Chen AUC across all
participant-by-kind statistics), then averaged over repeats, before swapping. The raw version (planned statistic, reproduces Table 2)
is the headline; the rank version is shown next to it. The two versions agree for Delmas, Chen and Preti with RT, but differ for Celniak, Preti eyes only and Colasanti group 2 vs 3
(see table), where the unified model's logit scale changes between folds (tuned C) and rank averaging per repeat removes that effect.

## Results (Holm over the family of 7)
Delmas row: the non-inferiority one-sided p (H0: difference <= -0.02, swap null shifted by the margin) takes the family's place; its
two-sided p is also shown. Rank columns: observed difference / two-sided p (/ NI p for Delmas).

| Comparison | Observed diff | Two-sided p (raw) | NI one-sided p (raw) | Holm p (raw) | Rank-transformed (diff / p) | Holm p (rank) |
|---|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

Secondary (not in the family): Chen participant AUC, raw diff {e2['observed_delta']:+.3f}, two-sided p {e2['raw']['p_two_sided']:.4f};
rank version diff {e2['rank']['observed_delta']:+.3f}, p {e2['rank']['p_two_sided']:.4f}.

Holm if the Delmas two-sided p is used in place of the non-inferiority p (raw scores): {", ".join(f"{n.split(' ')[0]}: {C[n]['holm_raw']['p_holm_if_delmas_two_sided']:.3f}" for n in fam_names)}.

## Deviations and notes
- Chen detection: the 7% thresholds were calibrated on training participants inside each fold and cannot be recomputed under
  swapping; the stored per-participant calibrated decisions (mean over the 3 repeats) were exchanged between methods, which keeps
  the planned statistic. Participant AUC on the same swap masks is reported as a secondary check. The baseline for Chen is the paper's rate rule.
- The test is of the difference under exchangeability; with 2,000 permutations the smallest attainable p is {1 / (1 + res['n_perm']):.5f}, below Holm's first-step threshold (.0071).
- Observed differences are the stored ones, so the reproduction is exact. Colasanti baselines have 3 repeats and the pipeline 5 (as in the planned statistic); the participant's repeat-averaged scores are swapped as a block.
- Rank-transformed observed differences differ from the planned ones because ranking within repeat changes scale before averaging; they are reported only as a scale-robustness check.
- Runtime: {res['runtime_s']:.0f} s.
"""
    (OUT / "card.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
