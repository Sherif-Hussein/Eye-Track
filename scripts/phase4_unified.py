"""Phase 4: the unified pipeline run (pre-registered in results/phase4/unified/PREREG.md).

Stages (python -I scripts/phase4_unified.py <stage> [options]); every stage checkpoints and skips finished work:
  extract     label-free feature tables (Delmas 30 Hz, Celniak fast) + equivalence checks against the stored tables
  run         nested CV of every task on the frozen folds (one checkpoint per task x repeat)
  stats       person tables, metrics, the seven primary tests (Holm), sensitivities, controls, cascade -> results.json
  curves      learning curves (25/50/75% of training participants; the 100% point is the main run)
  figdata     tidy CSVs in figdata/ (ROC/DET, score distributions, learning/validation curves, weights, calibration)
  efficiency  baseline vs unified, same session, single thread, interleaved rounds -> figdata/efficiency.csv
  all         all of the above in order

Real labels are only touched after the coordinator creates results/phase4/unified/GO_AHEAD.txt. Until then use
`--smoke <scratch dir>`: outputs go to that folder, all labels are shuffled, 1 repeat and few bootstrap draws.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edd import unified as U  # noqa: E402

REAL_OUT = ROOT / "results" / "phase4" / "unified"
N_JOBS = 4
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=conv))
    tmp.replace(path)


class Ctx:
    def __init__(self, a):
        self.smoke = a.smoke is not None
        self.out = Path(a.smoke) if self.smoke else REAL_OUT
        if not self.smoke and not (REAL_OUT / "GO_AHEAD.txt").exists():
            sys.exit("REFUSED: real-label stages need results/phase4/unified/GO_AHEAD.txt (coordinator go-ahead). Use --smoke <dir>.")
        self.out.mkdir(parents=True, exist_ok=True)
        for d in ("cache", "checkpoints", "oof", "figdata"):
            (self.out / d).mkdir(exist_ok=True)
        self.n_boot = a.n_boot or (40 if self.smoke else 2000)
        self.fig_boot = 40 if self.smoke else 500
        self.n_jobs = a.n_jobs
        self.repeats = [0] if self.smoke else None
        self.tasks, self.chen_data = None, None

    def load_tasks(self):
        if self.tasks is not None:
            return self.tasks
        t = {}
        t.update(U.build_delmas(self.out / "cache" / "delmas_30hz_features.csv", log))
        ch, self.chen_data = U.build_chen(log)
        t.update(ch)
        t.update(U.build_celniak(self.out / "cache" / "celniak_fast_slides.csv", log))
        t.update(U.build_preti(log))
        t.update(U.build_mazza(log))
        if self.smoke:
            t = {k: U.shuffle_labels(v, seed=11 + i) for i, (k, v) in enumerate(t.items())}
        self.tasks = t
        return t

    def run_all(self, frac=1.0, only=None):
        tasks = self.load_tasks()
        res = {}
        for name, t in tasks.items():
            if only and name not in only:
                continue
            res[name] = U.run_task(name, t["X"], t["y"], t["g"], t["folds"], self.out / "checkpoints", n_jobs=self.n_jobs,
                                   frac=frac, val=(frac == 1.0), repeats=self.repeats, log=log)
        return res


# ----------------------------------------------------------------------------- stages
def stage_extract(c):
    from edd import delmas_fast  # noqa: F401
    tasks = c.load_tasks()
    rep = {"delmas": {}, "celniak": {}}
    # Delmas: our 30 Hz per-image features vs the Phase-2 stored face table (rate 30), aggregated per (set, participant, image)
    from edd.delmas import load_trials
    from edd.delmas_raw import TRIAL_FEATURES
    feat = pd.read_csv(c.out / "cache" / "delmas_30hz_features.csv")
    feat["set"] = feat["set"].astype(str)
    stored = pd.read_csv(ROOT / "data" / "derived" / "delmas_raw_face_trials.csv.gz")
    stored = stored[stored.rate == 30]
    for s, t0 in (("64", 0), ("32", 32)):
        sub = stored[stored.trial >= t0]
        agg = sub.groupby(["participant", "image"])[TRIAL_FEATURES].mean().reset_index()
        m = feat[feat.set == s].merge(agg, on=["participant", "image"], suffixes=("", "_st"))
        cols = [f for f in TRIAL_FEATURES if f != "msacc_rate"]
        diff = max(float(np.nanmax(np.abs(m[f] - m[f + "_st"]))) for f in cols)
        rep["delmas"][s] = {"rows_compared": int(len(m)), "rows_new": int((feat.set == s).sum()), "max_abs_diff": diff}
    cz = pd.read_csv(c.out / "cache" / "celniak_fast_slides.csv")
    st = pd.read_csv(ROOT / "data" / "derived" / "celniak_slide_features.csv")
    m = cz.merge(st, left_on=["recording", "question", "repetition"], right_on=["recording", "question", "repetition"], suffixes=("", "_st"))
    rep["celniak"] = {"rows_compared": int(len(m)), "rows_new": int(len(cz)),
                      "max_abs_diff": max(float(np.nanmax(np.abs(m[f] - m[f + "_st"]))) for f in U.CELNIAK_FEATS)}
    jdump(rep, c.out / "extract_equivalence.json")
    log("extract done", json.dumps(rep))


def stage_run(c):
    c.run_all(1.0)
    log("run done")


def collect(c):
    """Rows, persons for every task, from checkpoints."""
    tasks = c.load_tasks()
    res = c.run_all(1.0)
    rows, persons = {}, {}
    for name, t in tasks.items():
        rows[name] = U.oof_rows(res[name], t["meta"])
        persons[name] = U.PERSON_FN[t["person"]](t, res[name])
        rows[name].to_csv(c.out / "oof" / f"{name}_rows.csv.gz", index=False)
        persons[name].drop(columns=[x for x in ("train_pos", "train_null") if x in persons[name]]).to_csv(
            c.out / "oof" / f"{name}_persons.csv.gz", index=False)
    return tasks, res, rows, persons


def stage_stats(c):
    tasks, res, rows, persons = collect(c)
    R = {"recipe": {"C_grid": list(U.C_GRID), "inner_max_k": U.INNER_MAX_K, "target_fpr": U.TARGET_FPR, "n_boot": c.n_boot,
                    "boot_seed": U.BOOT_SEED, "smoke": c.smoke}}
    R["tasks"] = {}
    for name, t in tasks.items():
        R["tasks"][name] = {"features": t["names"], "n_features": len(t["names"]), "n_rows": int(len(t["y"])),
                            "n_units": int(len(np.unique(t["g"]))),
                            "row_level": U.row_summary(rows[name]), "person_level": U.person_summary(persons[name]),
                            "C_chosen_counts": {str(k): int(v) for k, v in rows[name].drop_duplicates(["repeat", "fold"]).C.value_counts().items()}}
        log(f"{name}: row AUC {R['tasks'][name]['row_level']['auc']['mean']:.3f}  person AUC {R['tasks'][name]['person_level']['auc']['mean']:.3f}")
    S = int(c.n_boot)
    seed = U.BOOT_SEED
    P = {}                                                                     # the seven pre-registered primary tests
    dl = {k: v for k, v in rows.items() if k.startswith("delmas_")}
    P["delmas_noninferiority"] = U.delmas_primary(dl, S, seed)
    P["delmas_sensitivity_group3_once"] = U.delmas_primary(dl, S, seed, drop=("32_g3",))
    P["chen_detection_fpr7"] = U.chen_primary(persons["chen_unified"], S, seed)
    P["chen_trial_auc_descriptive"] = U.chen_trial_auc(rows["chen_unified"], S, seed)
    P["celniak_participant_auc"] = U.test_person_auc(U.baseline_unit_scores("celniak"), U.uni_unit_scores(rows["celniak_unified"]), S, seed)
    P["preti_A_participant_auc"] = U.test_person_auc(U.baseline_unit_scores("preti_A"), U.uni_unit_scores(rows["preti_A_ocular"]), S, seed)
    P["preti_B_participant_auc"] = U.test_person_auc(U.baseline_unit_scores("preti_B"), U.uni_unit_scores(rows["preti_B_ocular_rt"]), S, seed)
    for tk in ("g2_vs_g3", "guilty_vs_innocent"):
        P[f"mazza_{tk}_participant_auc"] = U.test_person_auc(U.baseline_unit_scores("mazza", tk), U.uni_unit_scores(rows[f"mazza_{tk}"]), S, seed)
    # controls, references and sensitivities (descriptive, outside the Holm family)
    D = {}
    D["celniak_control_validation_error_covariate_vs_baseline"] = U.test_person_auc(
        U.baseline_unit_scores("celniak"), U.uni_unit_scores(rows["celniak_ctrl_valcov"]), S, seed)
    D["celniak_control_vs_unified"] = U.test_person_auc(
        U.uni_unit_scores(rows["celniak_unified"]), U.uni_unit_scores(rows["celniak_ctrl_valcov"]), S, seed)
    D["celniak_reference_validation_error_only_vs_baseline"] = U.test_person_auc(
        U.baseline_unit_scores("celniak"), U.uni_unit_scores(rows["celniak_ref_valonly"]), S, seed)
    D["celniak_unified_vs_validation_error_only"] = U.test_person_auc(
        U.uni_unit_scores(rows["celniak_ref_valonly"]), U.uni_unit_scores(rows["celniak_unified"]), S, seed)
    D["preti_A_contrast_only_vs_baseline"] = U.test_person_auc(U.baseline_unit_scores("preti_A"), U.uni_unit_scores(rows["preti_A_contrast_only"]), S, seed)
    D["preti_B_contrast_only_vs_baseline"] = U.test_person_auc(U.baseline_unit_scores("preti_B"), U.uni_unit_scores(rows["preti_B_contrast_only"]), S, seed)
    fam = [("delmas_noninferiority", P["delmas_noninferiority"]["p_inferior"]), ("chen_detection_fpr7", P["chen_detection_fpr7"]["detection"]["p_superior"]),
           ("celniak_participant_auc", P["celniak_participant_auc"]["p_superior"]), ("preti_A_participant_auc", P["preti_A_participant_auc"]["p_superior"]),
           ("preti_B_participant_auc", P["preti_B_participant_auc"]["p_superior"]),
           ("mazza_g2_vs_g3_participant_auc", P["mazza_g2_vs_g3_participant_auc"]["p_superior"]),
           ("mazza_guilty_vs_innocent_participant_auc", P["mazza_guilty_vs_innocent_participant_auc"]["p_superior"])]
    adj = U.holm([p for _, p in fam])
    R["holm_family"] = [{"test": k, "p_raw": p, "p_holm": float(a)} for (k, p), a in zip(fam, adj)]
    # cascade (optional, separate; stage-1 tiny LR vs the confirmed I2b), cost from the efficiency stage when it exists
    cost = None
    ef = c.out / "figdata" / "efficiency.csv"
    if ef.exists():
        e = pd.read_csv(ef)
        e = e[e.dataset == "chen"].set_index("variant")
        if {"cascade_stage1_tiny_LR", "I2b_confirmed_model"} <= set(e.index):
            cost = {"stage1_ms": float(e.loc["cascade_stage1_tiny_LR", "latency_p50_ms"]), "i2b_ms": float(e.loc["I2b_confirmed_model", "latency_p50_ms"])}
    casc, ctab = U.cascade_eval(persons["chen_stage1"], S, seed + 7, cost)
    ctab.to_csv(c.out / "oof" / "chen_cascade_decisions.csv.gz", index=False)
    R["primary_tests"] = {k: {kk: vv for kk, vv in v.items() if kk != "deltas"} for k, v in P.items()}
    R["controls_and_sensitivities"] = {k: {kk: vv for kk, vv in v.items() if kk != "deltas"} for k, v in D.items()}
    R["cascade_chen"] = casc
    jdump(R, c.out / "results.json")
    log("stats done; Holm family:", json.dumps(R["holm_family"]))


def stage_curves(c):
    for frac in U.LEARNING_FRACS:
        c.run_all(frac)
    log("learning-curve runs done")


def stage_figdata(c):
    tasks, res, rows, persons = collect(c)
    fd = c.out / "figdata"
    (fd / "README.md").write_text(U.FIGDATA_README, encoding="utf-8")
    specs, dist, calib, vals, learn, wts = [], [], [], [], [], []
    base_cache = {}

    def base_for(name):
        if name.startswith("celniak_"):
            return U.baseline_unit_scores("celniak")
        if name == "preti_A_ocular":
            return U.baseline_unit_scores("preti_A")
        if name == "preti_B_ocular_rt":
            return U.baseline_unit_scores("preti_B")
        if name.startswith("mazza_"):
            return U.baseline_unit_scores("mazza", name.replace("mazza_", ""))
        return None

    delmas_base = pd.read_csv(U.RESULTS / "phase2" / "delmas_raw" / "oof_scores.csv.gz")
    delmas_base = delmas_base[delmas_base.tag == "P4_svm_1000"]
    cp_pers, cp_trials = (U.chen_checkpoint_persons() if "chen_unified" in tasks else (None, None))
    for name, t in tasks.items():
        ds = t["ds"]
        rw, pr = rows[name], persons[name]
        pl = U.primary_level(t, rw)
        avg = pl.groupby("uid", as_index=False).agg(y=("y", "first"), score=("score", "mean"), unit=("unit", "first"))
        lvl_primary = "person" if t["person"] == "mean" else "row"
        # ---- ROC/DET specs
        specs.append({"dataset": ds, "task": name, "level": lvl_primary, "model": "unified", "groups": U.unit_groups(avg.assign(unit=avg.unit.astype(str)))})
        b = base_for(name)
        if b is not None:
            specs.append({"dataset": ds, "task": name, "level": "person", "model": "baseline", "groups": U.unit_groups(b.assign(unit=b.unit.astype(str)))})
        if name.startswith("delmas_"):
            key = name.replace("delmas_", "")
            bb = delmas_base[delmas_base.set == key].groupby(["participant", "image"], as_index=False).agg(score=("score", "mean"), y=("y", "first"))
            bb = bb.merge(avg.assign(participant=avg.unit.astype(int), image=avg.uid.str.split("_").str[1].astype(int))[["participant", "image", "y"]],
                          on=["participant", "image"], suffixes=("_b", ""))
            specs.append({"dataset": ds, "task": name, "level": "row", "model": "baseline", "groups": U.unit_groups(bb.assign(unit=bb.participant.astype(str)))})
        if name == "chen_unified":
            for arm in ("rate_rule", "I2b"):
                a = cp_trials[cp_trials.arm == arm].groupby("row", as_index=False).agg(score=("score", "mean"))
                m = avg[["uid", "y", "unit"]].merge(a, left_on="uid", right_on="row")
                specs.append({"dataset": ds, "task": name, "level": "row", "model": f"baseline_{arm}", "groups": U.unit_groups(m.assign(unit=m.unit.astype(str)))})
                long = pd.concat([pd.DataFrame({"unit": cp_pers[cp_pers.arm == arm].unit, "kind": k, "stat": cp_pers[cp_pers.arm == arm][zk]})
                                  for k, zk in (("fam", "z_fam"), ("c1", "z_c1"), ("c2", "z_c2"))])
                specs.append({"dataset": ds, "task": name, "level": "person", "model": f"baseline_{arm}", "groups": U._chen_person_groups(long)})
        # ---- person-level statistic curves for cit / chen
        if t["person"] in ("cit", "chen"):
            pa = pr.groupby(["unit", "kind"], as_index=False).agg(y=("y", "first"), stat=("stat", "mean"))
            pa["unit"] = pa["unit"].astype(str)
            specs.append({"dataset": ds, "task": name, "level": "person", "model": "unified",
                          "groups": {u: (d.y.to_numpy(), d.stat.to_numpy()) for u, d in pa.groupby("unit")}})
            d2 = pa.assign(uid=pa.unit + "_" + pa.kind).rename(columns={"stat": "score"})[["uid", "unit", "y", "score"]]
            dist.append(U.score_distribution_table(ds, name, "person", d2, pr))
            calib.append(U.calibration_bins(ds, name, "person", d2))
        dist.append(U.score_distribution_table(ds, name, lvl_primary, avg[["uid", "unit", "y", "score"]], pr if lvl_primary == "person" else None))
        calib.append(U.calibration_bins(ds, name, lvl_primary, avg))
        # ---- validation curve, weights
        vals.append(U.validation_curve_table(ds, name, t, res[name], t["meta"]))
        wts.append(U.feature_weight_table(ds, name, t["names"], res[name]))
        # ---- learning curve (needs `curves` stage checkpoints; 100% = main run)
        rf = {1.0: res[name]}
        for fr in U.LEARNING_FRACS:
            r = U.run_task(name, t["X"], t["y"], t["g"], t["folds"], c.out / "checkpoints", n_jobs=c.n_jobs, frac=fr, val=False,
                           repeats=c.repeats, log=log)
            rf[fr] = r
        learn.append(U.learning_curve_table(ds, name, t, rf, t["meta"], n_boot=c.fig_boot))
    curves, eers = U.roc_det_tables(specs, n_boot=c.fig_boot)
    curves.to_csv(fd / "roc_det_curves.csv", index=False)
    eers.to_csv(fd / "roc_det_eer.csv", index=False)
    pd.concat(dist).to_csv(fd / "score_distributions.csv", index=False)
    pd.concat(calib).to_csv(fd / "calibration_bins.csv", index=False)
    pd.concat(vals).to_csv(fd / "validation_curve.csv", index=False)
    pd.concat(learn).to_csv(fd / "learning_curve.csv", index=False)
    pd.concat(wts).to_csv(fd / "feature_weights.csv", index=False)
    log("figdata written to", fd)


def stage_efficiency(c):
    tasks = c.load_tasks()
    c.load_tasks()
    df, info = U.run_efficiency(tasks, c.chen_data, rounds=3 if c.smoke else 6, quick=c.smoke, log=log)
    df.to_csv(c.out / "figdata" / "efficiency.csv", index=False)
    jdump(info, c.out / "efficiency_info.json")
    log("efficiency written", len(df), "rows; errors:", info["errors"])


def stage_chen_rate(c):
    """Pre-registered analysis A (PREREG section 11): Chen at 100 / 60 / 30 Hz, unified recipe and I2b."""
    from edd import chen_confirm as cc
    tasks = c.load_tasks()
    meta, ch = c.chen_data
    base = tasks["chen_unified"]
    # feature identity at the reference rate: rebuilt I2b features vs the confirmatory run's stored feature matrix
    Xi, _ = U.chen_i2b_features(ch, 100)
    Xc, _, _ = cc.feature_sets(ch)
    ident = {"max_abs_diff_I2b_features_100Hz_vs_confirm": float(np.nanmax(np.abs(Xi - Xc))), "n_features": int(Xi.shape[1])}
    rt = U.build_chen_rate_tasks(c.chen_data, (100, 60))
    for t in rt.values():                       # same (possibly shuffled) labels as the main Chen task
        t["T1"], t["y"], t["meta"] = base["T1"], base["y"], base["meta"]
    allt = {"chen_unified_r100": rt["chen_unified_r100"], "chen_unified_r60": rt["chen_unified_r60"], "chen_unified": base}
    rows_by, persons_by = {}, {}
    for fs, name in ((100, "chen_unified_r100"), (60, "chen_unified_r60"), (30, "chen_unified")):
        t = allt[name]
        res = U.run_task(name, t["X"], t["y"], t["g"], t["folds"], c.out / "checkpoints", n_jobs=c.n_jobs, val=False, repeats=c.repeats, log=log)
        rows_by[("unified", fs)] = U.oof_rows(res, t["meta"])
        persons_by[("unified", fs)] = U.persons_chen(t, res)
    for fs in (60, 30):
        X, _ = U.chen_i2b_features(ch, fs)
        res = U.run_chen_i2b(f"chen_i2b_r{fs}", X, base, c.out / "checkpoints", n_jobs=c.n_jobs, repeats=c.repeats, log=log)
        rows_by[("I2b", fs)] = U.oof_rows(res, base["meta"])
        persons_by[("I2b", fs)] = U.persons_chen(base, res)
    cp, ct = U.chen_checkpoint_persons()
    for arm, key in (("I2b", ("I2b", 100)), ("rate_rule", ("rate_rule", 100))):
        rows_by[key] = ct[ct.arm == arm][["repeat", "fold", "row", "unit", "y", "score"]].reset_index(drop=True)
        persons_by[key] = U.cp_persons(cp, arm)
    if c.repeats is not None:                   # smoke: one repeat of the candidates; keep the stored arms to repeat 0 too
        for k in rows_by:
            rows_by[k] = rows_by[k][rows_by[k]["repeat"].isin(c.repeats)]
            persons_by[k] = persons_by[k][persons_by[k]["repeat"].isin(c.repeats)]
    res = U.chen_rate_analysis(rows_by, persons_by, rows_by[("unified", 30)], c.n_boot, U.BOOT_SEED + 11)
    res["feature_identity_check"] = ident
    res["rates_hz"] = list(U.CHEN_RATES)
    res["reference_rate_note"] = ("100 Hz is the highest rate cached for both pupil and gaze (gaze exists only as 100 Hz epochs; raw 1 kHz pupil "
                                  "is cached but not gaze). 60 and 30 Hz are anti-aliased polyphase decimations of the 100 Hz epochs.")
    jdump(res, c.out / "results_chen_rate.json")
    tab = [{"model": k.split("@")[0], "rate_hz": int(k.split("@")[1]), "trial_auc": res["trial_auc"][k], **res["secondary_descriptive"][k]}
           for k in res["trial_auc"] if k in res["secondary_descriptive"]]
    pd.DataFrame(tab).to_csv(c.out / "figdata" / "chen_rate_curve.csv", index=False)
    log("chen_rate done:", json.dumps(res["trial_auc"]))


def stage_webcam_adaptive(c):
    """Pre-registered analysis B (PREREG section 12): data-driven fixation threshold under simulated webcam noise. Primary rule
    `ek` (median-based, Engbert & Kliegl style); secondary rule `nh` (Nystrom & Holmqvist iterative); each its own Holm family."""
    cache = c.out / "cache" / "webcam_adaptive"
    cache.mkdir(parents=True, exist_ok=True)
    out, thr_all = {}, []
    for rule in ("ek",):                              # `nh` dropped (PREREG amendment 3)
        rows_by = {}
        for L in U.WEBCAM_LEVELS:
            for d in ([0] if L == 0 else U.WEBCAM_DRAWS):
                tag = f"{rule}_L{int(L * 100):03d}_d{d}"
                f, tf = cache / f"feat_{tag}.csv.gz", cache / f"thr_{tag}.csv"
                if f.exists() and tf.exists():
                    feat, thr = pd.read_csv(f), pd.read_csv(tf)
                else:
                    feat, thr = U.webcam_adaptive_features(L, d, rule)
                    feat.to_csv(f, index=False)
                    thr.to_csv(tf, index=False)
                feat["set"] = feat["set"].astype(str)
                thr_all.append(thr)
                tasks = U.delmas_tasks_from_features(feat)
                if c.smoke:
                    tasks = {k: U.shuffle_labels(v, seed=11 + i) for i, (k, v) in enumerate(tasks.items())}
                per = {}
                for tn, t in tasks.items():
                    res = U.run_task(f"{tn}__adapt_{tag}", t["X"], t["y"], t["g"], t["folds"], c.out / "checkpoints",
                                     n_jobs=c.n_jobs, val=False, repeats=c.repeats, log=lambda *a: None)
                    per[tn.replace("delmas_", "")] = U.oof_rows(res, t["meta"])
                rows_by[(L, d)] = per
                log(f"webcam adaptive [{rule}] level {L} draw {d} done")
        out[rule] = U.webcam_level_tests(rows_by, c.n_boot, U.BOOT_SEED + 23)
        log(f"webcam_adaptive [{rule}]:", json.dumps([(t["level_deg"], round(t["auc_level"], 3), round(t["p_holm"], 3)) for t in out[rule]]))
    thr_df = pd.concat(thr_all)
    thr_sum = thr_df.groupby(["rule", "level", "draw"]).threshold.agg(["mean", "std", "min", "max"]).reset_index().to_dict("records")
    jdump({"primary_rule_ek": out["ek"], "threshold_deg_per_s_by_rule_level_draw": thr_sum,
           "rules": {"ek": "max(30, 6 x sqrt((SDx^2+SDy^2)/2)), SD = sqrt(median(v^2)-median(v)^2) per axis (Engbert & Kliegl 2003 style)",
                     },
           "holm_family": "the six levels (0, 1, 1.5, 2, 3, 4 deg), separately per rule"}, c.out / "results_webcam_adaptive.json")
    thr_df.to_csv(c.out / "figdata" / "webcam_adaptive_thresholds.csv", index=False)
    cur = pd.concat([pd.DataFrame(v).drop(columns=["ci95"]).assign(rule=k, ci_lo=[t["ci95"][0] for t in v], ci_hi=[t["ci95"][1] for t in v])
                     for k, v in out.items()])
    cur.to_csv(c.out / "figdata" / "webcam_adaptive_curve.csv", index=False)


def stage_posthoc(c):
    """POST-HOC, descriptive only (not in the PREREG; logged in DEVIATIONS.md): the person-level primary tests average raw logits over
    repeats, and tuned C (hence logit scale) differs between folds, which can lower the repeat-averaged AUC. Here: per-repeat AUC
    (mean over repeats) for unified vs baseline with a paired participant bootstrap of the difference. No decision is based on it."""
    tasks, res, rows, persons = collect(c)
    p2 = U.RESULTS / "phase2"
    srcs = {"celniak": (pd.read_csv(p2 / "celniak" / "trials_baseline_svm_paper.csv"), "celniak_unified"),
            "preti_A": (pd.read_csv(p2 / "preti" / "scores" / "B_ocular.csv"), "preti_A_ocular"),
            "preti_B": (pd.read_csv(p2 / "preti" / "scores" / "B_fusion.csv"), "preti_B_ocular_rt"),
            "mazza_g2_vs_g3": (pd.read_csv(p2 / "mazza" / "oof" / "g2_vs_g3__base_pub36_lr_unreg.csv"), "mazza_g2_vs_g3"),
            "mazza_guilty_vs_innocent": (pd.read_csv(p2 / "mazza" / "oof" / "guilty_vs_innocent__base_pub36_lr_unreg.csv"), "mazza_guilty_vs_innocent")}
    out = {}
    rng = np.random.default_rng(U.BOOT_SEED + 31)
    for k, (b, tn) in srcs.items():
        b = b.assign(unit=b.unit.astype(str)).groupby(["repeat", "unit"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
        u = rows[tn].assign(unit=rows[tn].unit.astype(str)).groupby(["repeat", "unit"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
        reps = sorted(set(b["repeat"]) & set(u["repeat"]))
        units = np.array(sorted(set(u.unit)))
        ymap = u.drop_duplicates("unit").set_index("unit").y
        def per_rep(df, uu):
            d = df.set_index(["repeat", "unit"])
            a = []
            for r in reps:
                sub = d.loc[r]
                yy = ymap.loc[uu].to_numpy()
                ss = sub.score.reindex(uu).to_numpy()
                a.append(U.fast_auc(yy, ss))
            return float(np.mean(a))
        ub, uu_ = per_rep(b, units), per_rep(u, units)
        # bootstrap over participants (resampled units keep their duplicates: handled by indexing with repeats)
        dl = []
        for _ in range(c.n_boot // 4):
            pick = units[rng.integers(0, len(units), len(units))]
            dl.append(per_rep(u, pick) - per_rep(b, pick))
        out[k] = {"per_repeat_mean_auc_baseline": ub, "per_repeat_mean_auc_unified": uu_, "delta": uu_ - ub,
                  "ci95": [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))], "n_boot": len(dl)}
        log("posthoc", k, json.dumps(out[k]))
    jdump(out, c.out / "results_posthoc_per_repeat_auc.json")


STAGES = {"extract": stage_extract, "run": stage_run, "stats": stage_stats, "curves": stage_curves,
          "figdata": stage_figdata, "efficiency": stage_efficiency,
          "chen_rate": stage_chen_rate, "posthoc": stage_posthoc, "webcam_adaptive": stage_webcam_adaptive}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--smoke", default=None, help="scratch dir; shuffled labels, 1 repeat, few bootstrap draws")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    a = ap.parse_args()
    if a.n_jobs > 4:
        sys.exit("n_jobs must be <= 4")
    c = Ctx(a)
    order = ["extract", "run", "efficiency", "stats", "chen_rate", "webcam_adaptive", "curves", "figdata"] if a.stage == "all" else [a.stage]
    for s in order:
        log("=== stage", s)
        STAGES[s](c)
    log("done")


if __name__ == "__main__":
    main()
