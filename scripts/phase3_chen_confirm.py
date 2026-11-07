"""Phase 3 confirmatory test `chen_confirm` (see results/phase3/chen_confirm/PREREG.md).

Stages (python -I scripts/phase3_chen_confirm.py <stage>):
  folds    freeze folds.json (no model is run)
  run      fit all arms on the frozen folds, one checkpoint per repeat (skips finished repeats)
  analyse  primary + secondary + confound analyses -> results.json, card.md
Smoke test: EDD_CONFIRM_SMOKE=<scratch dir> writes there, shuffles labels within participants, uses 2 folds.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edd import chen_confirm as cc  # noqa: E402
from edd.folds import save_folds  # noqa: E402

SMOKE = os.environ.get("EDD_CONFIRM_SMOKE")
OUT = Path(SMOKE) if SMOKE else ROOT / "results" / "phase3" / "chen_confirm"
CK = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CK.mkdir(exist_ok=True)
N_JOBS = 2
N_BOOT = 40 if SMOKE else 2000
BOOT_SEED = 20261102
N_FLIPS = 500 if SMOKE else 20000


def load_pooled():
    meta, ch = cc.build_pooled()
    if SMOKE:  # label-shuffled smoke data: within-participant permutation of the trial types
        rng = np.random.default_rng(1)
        T1 = meta.T1.to_numpy().copy()
        for u in meta.unit.unique():
            ix = np.flatnonzero(meta.unit.to_numpy() == u)
            T1[ix] = rng.permutation(T1[ix])
        meta = meta.assign(T1=T1)
    return meta, ch


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_resume(text):
    (OUT / "RESUME.md").write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------------------------
def stage_folds():
    meta, _ = load_pooled()
    fp = OUT / "folds.json"
    if fp.exists():
        log("folds.json exists, not overwritten")
        return
    folds = cc.make_folds(meta)
    n = {e: int(meta.loc[meta.exp == e, "unit"].nunique()) for e in (1, 2)}
    save_folds(fp, folds, meta={"design": "3 repeats x 10 participant-disjoint folds, pooled Exp1+Exp2, stratified by experiment",
                                "seeds": list(cc.SEEDS), "unit": "exp1: subject_nr; exp2: subject_nr + 1000", "n_participants": n})
    log(f"folds frozen {n}")


def stage_run():
    meta, ch = load_pooled()
    folds = json.loads((OUT / "folds.json").read_text())["repeats"]
    X, names, cols = cc.feature_sets(ch)
    D = {"units": meta.unit.to_numpy(), "exps": meta.exp.to_numpy(), "y": (meta.T1.to_numpy() == cc.FAM).astype(int),
         "T1": meta.T1.to_numpy(), "X": X, "cols": cols, "rate": ch["rate"]}
    n_rep = 1 if SMOKE else len(folds)
    for r in range(n_rep):
        ck = CK / f"repeat{r}.pkl"
        if ck.exists():
            log(f"repeat {r}: done earlier, skipped")
            continue
        fl = folds[r][:2] if SMOKE else folds[r]
        t0 = time.time()
        res = Parallel(n_jobs=N_JOBS)(delayed(cc.fold_task)(r, f, t, D) for f, t in enumerate(fl))
        trials = pd.concat([a for a, _ in res], ignore_index=True)
        pers = pd.concat([b for _, b in res], ignore_index=True)
        tmp = ck.with_suffix(".tmp")
        pickle.dump({"trials": trials, "persons": pers, "names": list(names)}, open(tmp, "wb"))
        tmp.replace(ck)
        log(f"repeat {r} done in {time.time() - t0:.0f}s; persons {len(pers)}, trials {len(trials)}")
        done = sorted(int(p.stem[6:]) for p in CK.glob("repeat*.pkl"))
        write_resume(resume_text(done, n_rep))


def resume_text(done, n_rep):
    status = "COMPLETE (run stage); run `analyse` next" if len(done) == n_rep else f"running/partial: repeats done {done}"
    return f"""# chen_confirm RESUME (updated {time.strftime('%Y-%m-%d %H:%M')})

Status: {status}

PREREG.md is frozen (never edit). Deviations go in DEVIATIONS.md.

Resume commands (project root, Git Bash, n_jobs=2):
```
.venv/Scripts/python.exe -I scripts/phase3_chen_confirm.py run      > results/phase3/chen_confirm/run.log 2>&1   # skips finished repeats
.venv/Scripts/python.exe -I scripts/phase3_chen_confirm.py analyse  > results/phase3/chen_confirm/analyse.log 2>&1 # results.json, card.md
```
Checkpoints: `checkpoints/repeat{{0,1,2}}.pkl` (trial scores + per-person z statistics and thresholds, all arms).
"""


# ----------------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------------
class Pooled:
    """Aligned per-participant and per-trial arrays for all arms and repeats."""

    def __init__(self):
        cks = [pickle.load(open(CK / f"repeat{r}.pkl", "rb")) for r in range(3)] if not SMOKE else [pickle.load(open(CK / "repeat0.pkl", "rb"))]
        self.R = len(cks)
        pers = pd.concat([c["persons"] for c in cks], ignore_index=True)
        trials = pd.concat([c["trials"] for c in cks], ignore_index=True)
        self.units = np.array(sorted(pers.unit.unique()))
        self.exp = np.array([int(pers[pers.unit == u].exp.iloc[0]) for u in self.units])
        self.arms = [a for a in cc.ARMS if a in set(pers.arm)]
        self.P = {}
        for a in self.arms:
            d = {}
            for k in ("detect_cal", "fpr_cal", "z_fam", "z_c1", "z_c2", "p", "p_c1c2", "p_c2c1"):
                M = np.full((len(self.units), self.R), np.nan)
                for r in range(self.R):
                    s = pers[(pers.arm == a) & (pers.repeat == r)].set_index("unit")[k]
                    M[:, r] = s.reindex(self.units).to_numpy()
                d[k] = M
            self.P[a] = d
        self.T = {}
        self.rows = {}
        for a in self.arms:
            for r in range(self.R):
                t = trials[(trials.arm == a) & (trials.repeat == r)].sort_values("row")
                self.T[a, r] = (t.y.to_numpy(), t.score.to_numpy())
                if a == "rate_rule":
                    self.rows[r] = t.unit.to_numpy()
                else:
                    assert (self.rows[r] == t.unit.to_numpy()).all()
        self.pos = {r: [np.flatnonzero(self.rows[r] == u) for u in self.units] for r in range(self.R)}
        self.pers_raw = pers

    def metrics(self, arm, pick, trial=True):
        P = self.P[arm]
        out = {"det_cal": float(np.nanmean(P["detect_cal"][pick])), "fpr_cal": float(np.nanmean(P["fpr_cal"][pick])),
               "det05": float(np.nanmean(P["p"][pick] < 0.05)),
               "fpr05": float(np.nanmean(((P["p_c1c2"][pick] < 0.05) * 1.0 + (P["p_c2c1"][pick] < 0.05)) / 2))}
        pa, pe, ph, ta, te, th = [], [], [], [], [], []
        for r in range(self.R):
            y, s = cc.person_level(P["z_fam"][pick, r], P["z_c1"][pick, r])
            pa.append(cc.fast_auc(y, s)); pe.append(cc.fast_eer(y, s)); ph.append(cc.fast_hter(y, s))
            if trial:
                idx = np.concatenate([self.pos[r][i] for i in pick])
                yy, ss = self.T[arm, r]
                ta.append(cc.fast_auc(yy[idx], ss[idx])); te.append(cc.fast_eer(yy[idx], ss[idx])); th.append(cc.fast_hter(yy[idx], ss[idx]))
        out.update({"person_auc": float(np.mean(pa)), "person_eer": float(np.mean(pe)), "person_hter": float(np.mean(ph))})
        if trial:
            out.update({"trial_auc": float(np.mean(ta)), "trial_eer": float(np.mean(te)), "trial_hter": float(np.mean(th))})
        return out


HIGHER_BETTER = {"det_cal": True, "person_auc": True, "trial_auc": True, "person_eer": False, "person_hter": False,
                 "trial_eer": False, "trial_hter": False}


def boot_p_one_sided(delta_draws, higher_better=True):
    d = np.asarray(delta_draws) if higher_better else -np.asarray(delta_draws)
    return float((1 + (d <= 0).sum()) / (len(d) + 1))


def boot_p_two_sided(delta_draws):
    d = np.asarray(delta_draws)
    return float(min(1.0, 2 * min((1 + (d <= 0).sum()), (1 + (d >= 0).sum())) / (len(d) + 1)))


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def stage_analyse():
    pool = Pooled()
    U, n = pool.units, len(pool.units)
    log(f"{n} participants, arms {pool.arms}, repeats {pool.R}")
    allp = np.arange(n)
    res = {"n_participants": n, "n_exp1": int((pool.exp == 1).sum()), "n_exp2": int((pool.exp == 2).sum()), "n_boot": N_BOOT}
    point = {a: pool.metrics(a, allp) for a in pool.arms}
    # sanity: fast metrics agree with edd.metrics.binary_metrics
    from edd.metrics import binary_metrics
    y, s = pool.T["I2b", 0]
    bm = binary_metrics(y, s)
    assert abs(bm["auc"] - cc.fast_auc(y, s)) < 1e-9 and abs(bm["eer"] - cc.fast_eer(y, s)) < 1e-9 and abs(bm["hter"] - cc.fast_hter(y, s)) < 1e-9
    # participant bootstrap (same draws for all arms)
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(allp, n, replace=True) for _ in range(N_BOOT)]
    t0 = time.time()
    boot = {a: {k: [] for k in point[a]} for a in pool.arms}
    for i, pick in enumerate(draws):
        for a in pool.arms:
            m = pool.metrics(a, pick)
            for k, v in m.items():
                boot[a][k].append(v)
        if i % 500 == 0:
            log(f"bootstrap {i}/{N_BOOT} ({time.time() - t0:.0f}s)")
    boot = {a: {k: np.array(v) for k, v in d.items()} for a, d in boot.items()}
    res["point"] = point
    res["ci95"] = {a: {k: ci(boot[a][k]) for k in boot[a]} for a in pool.arms}

    def delta(a, b, k):
        dd = boot[a][k] - boot[b][k]
        return {"delta": point[a][k] - point[b][k], "ci95": ci(dd), "draws": dd}

    # ---- primary
    prim = delta("I2b", "rate_rule", "det_cal")
    prim_p = boot_p_one_sided(prim["draws"])
    di = np.nanmean(pool.P["I2b"]["detect_cal"], 1) - np.nanmean(pool.P["rate_rule"]["detect_cal"], 1)
    rng2 = np.random.default_rng(BOOT_SEED + 1)
    flips = rng2.choice([-1, 1], size=(N_FLIPS, n))
    perm_p = float((1 + ((flips * di).mean(1) >= di.mean() - 1e-12).sum()) / (N_FLIPS + 1))
    per_rep = [float(np.nanmean(pool.P["I2b"]["detect_cal"][:, r]) - np.nanmean(pool.P["rate_rule"]["detect_cal"][:, r])) for r in range(pool.R)]
    per_exp = {}
    for e in (1, 2):
        idx = np.flatnonzero(pool.exp == e)
        rr = np.random.default_rng(BOOT_SEED + 10 + e)
        dd = []
        for _ in range(N_BOOT):
            pk = rr.choice(idx, len(idx), replace=True)
            dd.append(np.nanmean(pool.P["I2b"]["detect_cal"][pk]) - np.nanmean(pool.P["rate_rule"]["detect_cal"][pk]))
        per_exp[f"exp{e}"] = {"n": int(len(idx)), "I2b": float(np.nanmean(pool.P["I2b"]["detect_cal"][idx])),
                              "rate_rule": float(np.nanmean(pool.P["rate_rule"]["detect_cal"][idx])),
                              "delta": float(di[idx].mean()), "ci95": ci(dd),
                              "fpr_I2b": float(np.nanmean(pool.P["I2b"]["fpr_cal"][idx])),
                              "fpr_rate_rule": float(np.nanmean(pool.P["rate_rule"]["fpr_cal"][idx]))}
    res["primary"] = {"outcome": "person-level detection at calibrated FPR <= 7%, I2b minus rate rule", "delta": prim["delta"],
                      "ci95": prim["ci95"], "ci90_lower": float(np.percentile(prim["draws"], 5)), "p_one_sided_bootstrap": prim_p,
                      "alpha": 0.05, "confirmed": bool(prim_p < 0.05), "sensitivity_sign_flip_p_one_sided": perm_p,
                      "per_repeat_delta": per_rep, "per_experiment": per_exp,
                      "participants_I2b_gain_loss_tie": [int((di > 0).sum()), int((di < 0).sum()), int((di == 0).sum())],
                      "mean_fpr_I2b": point["I2b"]["fpr_cal"], "mean_fpr_rate_rule": point["rate_rule"]["fpr_cal"],
                      "delta_fpr": delta("I2b", "rate_rule", "fpr_cal")["delta"],
                      "delta_fpr_ci95": delta("I2b", "rate_rule", "fpr_cal")["ci95"]}

    # ---- confound (b): missing-sample fraction familiar vs control
    meta, ch = load_pooled()
    miss = ch["miss"]
    isf, isc = (meta.T1.to_numpy() == cc.FAM), (meta.T1.to_numpy() != cc.FAM)
    un = meta.unit.to_numpy()

    def paired_by_unit(vals):
        f = np.array([np.nanmean(vals[(un == u) & isf]) for u in U])
        c = np.array([np.nanmean(vals[(un == u) & isc]) for u in U])
        return f, c

    mf = np.nanmean(miss[:, :250], 1)
    f, c = paired_by_unit(mf)
    d = f - c
    w = stats.wilcoxon(d)
    rr = np.random.default_rng(BOOT_SEED + 20)
    bd = [d[rr.choice(allp, n)].mean() for _ in range(N_BOOT)]
    miss_res = {"window": "0-2500 ms", "mean_fam": float(f.mean()), "mean_ctrl": float(c.mean()), "mean_diff": float(d.mean()),
                "ci95_diff": ci(bd), "wilcoxon_p_two_sided": float(w.pvalue), "n_fam_higher": int((d > 0).sum()), "n_ctrl_higher": int((d < 0).sum())}
    for e in (1, 2):
        m = pool.exp == e
        miss_res[f"exp{e}"] = {"mean_diff": float(d[m].mean()), "wilcoxon_p": float(stats.wilcoxon(d[m]).pvalue)}
    bins = []
    for b in range(25):
        fb, cb = paired_by_unit(np.nanmean(miss[:, b * 10:(b + 1) * 10], 1))
        db = fb - cb
        bins.append({"bin_ms": f"{b * 100}-{b * 100 + 100}", "mean_diff": float(db.mean()),
                     "wilcoxon_p_raw": float(stats.wilcoxon(db).pvalue) if np.any(db != 0) else 1.0})
    miss_res["per_bin_descriptive"] = bins
    for col in ("trialnumber", "blockno"):
        fo, co = paired_by_unit(meta[col].to_numpy(float))
        miss_res[f"order_{col}"] = {"mean_diff_fam_minus_ctrl": float((fo - co).mean()), "wilcoxon_p": float(stats.wilcoxon(fo - co).pvalue)}
    res["confound_b_missing_fraction"] = miss_res

    # ---- secondary tests (one-sided I2b > rate rule, except where stated) + confound tests
    fam = []  # (name, raw p, two-sided?)
    sec = {}
    for name, k in (("S1_person_auc", "person_auc"), ("S2_trial_auc", "trial_auc"), ("S3_trial_eer", "trial_eer"),
                    ("S4_trial_hter", "trial_hter"), ("S5_person_eer", "person_eer"), ("S6_person_hter", "person_hter")):
        dl = delta("I2b", "rate_rule", k)
        sec[name] = {"metric": k, "delta": dl["delta"], "ci95": dl["ci95"], "p": boot_p_one_sided(dl["draws"], HIGHER_BETTER[k]),
                     "alt": "I2b better than rate rule (one-sided)"}
    dl = delta("I2b_noMiss", "rate_rule", "det_cal")
    sec["C2_noMiss_vs_rate_det_cal"] = {"metric": "det_cal", "delta": dl["delta"], "ci95": dl["ci95"],
                                         "p": boot_p_one_sided(dl["draws"]), "alt": "I2b-noMiss better than rate rule (one-sided)"}
    dl = delta("I2b", "I2b_noMiss", "det_cal")
    sec["C3_I2b_vs_noMiss_det_cal"] = {"metric": "det_cal", "delta": dl["delta"], "ci95": dl["ci95"],
                                        "p": boot_p_two_sided(dl["draws"]), "alt": "I2b differs from I2b-noMiss (two-sided)"}
    sec["C1_missing_fraction_fam_vs_ctrl"] = {"metric": "mean missing fraction fam - ctrl", "delta": miss_res["mean_diff"],
                                              "ci95": miss_res["ci95_diff"], "p": miss_res["wilcoxon_p_two_sided"], "alt": "paired Wilcoxon (two-sided)"}
    names = list(sec)
    adj = cc.holm([sec[k]["p"] for k in names])
    for k, a in zip(names, adj):
        sec[k]["p_holm"] = float(a)
    res["secondary_holm_family"] = {"m": len(names), "tests": sec}
    res["confound_c_survival"] = {"I2b_gain": res["primary"]["delta"], "noMiss_gain": sec["C2_noMiss_vs_rate_det_cal"]["delta"],
                                  "noMiss_gain_p_holm": sec["C2_noMiss_vs_rate_det_cal"]["p_holm"],
                                  "noMiss_over_I2b_gain_ratio": float(sec["C2_noMiss_vs_rate_det_cal"]["delta"] / res["primary"]["delta"]) if res["primary"]["delta"] != 0 else None,
                                  "gain_survives_prespecified_rule": bool(sec["C2_noMiss_vs_rate_det_cal"]["p_holm"] < 0.05)}

    # ---- exploratory (descriptive): MissOnly and per-experiment fits vs rate rule
    expl = {}
    for a in ("MissOnly", "I2b_perexp"):
        if a in pool.arms:
            dl = delta(a, "rate_rule", "det_cal")
            expl[f"{a}_minus_rate_rule_det_cal"] = {"delta": dl["delta"], "ci95": dl["ci95"]}
            dl = delta(a, "rate_rule", "trial_auc")
            expl[f"{a}_minus_rate_rule_trial_auc"] = {"delta": dl["delta"], "ci95": dl["ci95"]}
    for k in ("det_cal", "trial_auc", "person_auc"):
        dl = delta("I2b", "I2b_noMiss", k)
        expl[f"I2b_minus_noMiss_{k}"] = {"delta": dl["delta"], "ci95": dl["ci95"]}
    res["exploratory_descriptive"] = expl

    # ---- paired deltas of all metrics, I2b vs rate rule (for the card)
    res["paired_I2b_minus_rate_rule"] = {k: {kk: v for kk, v in delta("I2b", "rate_rule", k).items() if kk != "draws"} for k in point["I2b"]}
    res["paired_noMiss_minus_rate_rule"] = {k: {kk: v for kk, v in delta("I2b_noMiss", "rate_rule", k).items() if kk != "draws"} for k in point["I2b"]}

    # ---- coefficient shares of the pooled I2b refit on all participants (descriptive)
    X, names_f, cols = cc.feature_sets(ch)
    P = cc.p2()
    y_all = (meta.T1.to_numpy() == cc.FAM).astype(int)
    est, params = P.fit_lr(X, y_all, un, True)
    nm = np.array(names_f)[est.named_steps["sel"].get_support()]
    coef = est.named_steps["lr"].coef_.ravel()
    chan = pd.Series([x.split("_")[0] for x in nm])
    share = (pd.Series(np.abs(coef)).groupby(chan.values).sum() / np.abs(coef).sum()).to_dict()
    res["pooled_refit_coefficient_share"] = {"params": str(params), "share_by_channel": {k: float(v) for k, v in share.items()}}

    res["fold_diagnostics"] = {a: {"mean_tau": float(pool.pers_raw[pool.pers_raw.arm == a].tau.mean()),
                                   "mean_n_null": float(pool.pers_raw[pool.pers_raw.arm == a].n_null.mean())} for a in pool.arms}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_card(res, pool)
    log(f"PRIMARY delta {res['primary']['delta']:.4f} CI {res['primary']['ci95']} p {res['primary']['p_one_sided_bootstrap']:.4f}")


def f3(x):
    return f"{x:.3f}"


def write_card(res, pool):
    pt, ci_ = res["point"], res["ci95"]
    L = ["# chen_confirm card (generated by scripts/phase3_chen_confirm.py; analysis exactly as in PREREG.md)\n",
         f"N = {res['n_participants']} participants (Exp 1: {res['n_exp1']}, Exp 2: {res['n_exp2']}); 3 repeats x 10 participant-disjoint folds (seeds 3031-3033); "
         f"{res['n_boot']} participant-bootstrap resamples; all CIs are 95% percentile intervals.\n"]
    p = res["primary"]
    L.append("## 1. Primary test\n")
    L.append(f"Outcome: person-level detection at calibrated FPR <= 7% (mean over participants and repeats), I2b minus the paper's rate rule.\n")
    L.append(f"- Delta = **{p['delta']:+.4f}** (95% CI {p['ci95'][0]:+.4f} to {p['ci95'][1]:+.4f}); one-sided bootstrap p = **{p['p_one_sided_bootstrap']:.4f}** (alpha .05); "
             f"lower bound of the 90% CI = {p['ci90_lower']:+.4f}.\n- **Hypothesis confirmed: {p['confirmed']}.**\n"
             f"- Sensitivity, sign-flip randomisation (one-sided): p = {p['sensitivity_sign_flip_p_one_sided']:.4f}. Per repeat delta: {', '.join(f'{x:+.4f}' for x in p['per_repeat_delta'])}. "
             f"Participants gaining / losing / tied: {p['participants_I2b_gain_loss_tie']}.\n"
             f"- Realised FPR on held-out control-1 vs control-2: I2b {p['mean_fpr_I2b']:.3f}, rate rule {p['mean_fpr_rate_rule']:.3f}; difference {p['delta_fpr']:+.3f} ({p['delta_fpr_ci95'][0]:+.3f}, {p['delta_fpr_ci95'][1]:+.3f}).\n")
    L.append("| Experiment | n | I2b detection | Rate-rule detection | Delta (95% CI) | FPR I2b | FPR rate |\n|---|---|---|---|---|---|---|")
    for e, v in p["per_experiment"].items():
        L.append(f"| {e} | {v['n']} | {v['I2b']:.3f} | {v['rate_rule']:.3f} | {v['delta']:+.3f} ({v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}) | {v['fpr_I2b']:.3f} | {v['fpr_rate_rule']:.3f} |")
    L.append("\n(Per-experiment rows are descriptive, not tested.)\n")
    L.append("## 2. All metrics by arm (pooled, mean over 3 repeats; 95% participant-bootstrap CI)\n")
    keys = [("det_cal", "Detection @ FPR<=7%"), ("fpr_cal", "Realised FPR"), ("det05", "Detection @ alpha .05"), ("fpr05", "FPR @ alpha .05 (ctrl1 vs 2)"),
            ("person_auc", "Person AUC"), ("person_eer", "Person EER"), ("person_hter", "Person HTER"),
            ("trial_auc", "Trial AUC"), ("trial_eer", "Trial EER"), ("trial_hter", "Trial HTER")]
    arms = [a for a in cc.ARMS if a in pt]
    L.append("| Metric | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for k, nm in keys:
        L.append(f"| {nm} | " + " | ".join(f"{pt[a][k]:.3f} ({ci_[a][k][0]:.3f}, {ci_[a][k][1]:.3f})" for a in arms) + " |")
    L.append("\nArms: `rate_rule` = paper's rate rule (peak and calibration per experiment); `I2b` = pooled fit, 125 features; `I2b_noMiss` = I2b without the missing-fraction channel (100 features); "
             "`MissOnly` = missing-fraction channel alone (exploratory); `I2b_perexp` = I2b fitted within each experiment as in screening (sensitivity). HTER is at the preset threshold 0 (trial) or z = 1.645 (person).\n")
    L.append("## 3. Paired differences vs the rate rule (identical folds and participants)\n")
    L.append("| Metric | I2b minus rate rule | I2b-noMiss minus rate rule |\n|---|---|---|")
    for k, nm in keys:
        a, b = res["paired_I2b_minus_rate_rule"][k], res["paired_noMiss_minus_rate_rule"][k]
        L.append(f"| {nm} | {a['delta']:+.3f} ({a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}) | {b['delta']:+.3f} ({b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}) |")
    L.append("\n## 4. Secondary and confound tests (Holm family)\n")
    L.append("| Test | Delta (95% CI) | raw p | Holm p | Alternative |\n|---|---|---|---|---|")
    for k, v in res["secondary_holm_family"]["tests"].items():
        L.append(f"| {k} | {v['delta']:+.4f} ({v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}) | {v['p']:.4f} | {v['p_holm']:.4f} | {v['alt']} |")
    m = res["confound_b_missing_fraction"]
    L.append(f"\n### Confound (b): missing-sample fraction, familiar vs control (paired by participant, 0-2500 ms)\n")
    L.append(f"Mean familiar {m['mean_fam']:.4f}, control {m['mean_ctrl']:.4f}, difference {m['mean_diff']:+.4f} (95% CI {m['ci95_diff'][0]:+.4f}, {m['ci95_diff'][1]:+.4f}); "
             f"Wilcoxon two-sided p = {m['wilcoxon_p_two_sided']:.4f}; familiar higher in {m['n_fam_higher']}, control higher in {m['n_ctrl_higher']} participants. "
             f"Exp 1: {m['exp1']['mean_diff']:+.4f} (p = {m['exp1']['wilcoxon_p']:.3f}); Exp 2: {m['exp2']['mean_diff']:+.4f} (p = {m['exp2']['wilcoxon_p']:.3f}).\n")
    L.append(f"Trial order (familiar minus control): trial number {m['order_trialnumber']['mean_diff_fam_minus_ctrl']:+.2f} (p = {m['order_trialnumber']['wilcoxon_p']:.3f}); block {m['order_blockno']['mean_diff_fam_minus_ctrl']:+.3f} (p = {m['order_blockno']['wilcoxon_p']:.3f}).\n")
    L.append("Per 100-ms bin (descriptive, uncorrected): " + "; ".join(f"{b['bin_ms']}: {b['mean_diff']:+.4f} (p={b['wilcoxon_p_raw']:.3f})" for b in m["per_bin_descriptive"]) + "\n")
    c = res["confound_c_survival"]
    L.append(f"### Confound (c): does the gain survive removing the missing channel?\nI2b gain {c['I2b_gain']:+.4f}; I2b-noMiss gain {c['noMiss_gain']:+.4f} (Holm p = {c['noMiss_gain_p_holm']:.4f}); ratio noMiss/I2b = {c['noMiss_over_I2b_gain_ratio']}. "
             f"Pre-specified rule (noMiss gain significant after Holm at .05): **{c['gain_survives_prespecified_rule']}**.\n")
    L.append("## 5. Exploratory and sensitivity (descriptive, no inference)\n")
    L.append("| Contrast | Delta (95% CI) |\n|---|---|")
    for k, v in res["exploratory_descriptive"].items():
        L.append(f"| {k} | {v['delta']:+.4f} ({v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}) |")
    L.append(f"\nChannel share of |coef| in the pooled I2b refit on all participants: {', '.join(f'{k} {v:.2f}' for k, v in res['pooled_refit_coefficient_share']['share_by_channel'].items())} ({res['pooled_refit_coefficient_share']['params']}).\n")
    (OUT / "card.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    for st in (sys.argv[1:] or ["folds", "run", "analyse"]):
        {"folds": stage_folds, "run": stage_run, "analyse": stage_analyse}[st]()
