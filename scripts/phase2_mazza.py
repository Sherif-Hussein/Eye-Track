"""Phase 2 screening on the Mazza/Colasanti et al. (2023) RealEye webcam crime-scene dataset.

Stages (run in order):
  folds   freeze participant-disjoint outer folds (5 repeats x stratified 10-fold, stratified by group)
  screen  every config in phase2_mazza_configs.CONFIGS on every binary task; OOF scores to results dir
  extras  paired tests (Holm), bootstrap CIs, calibration + selective prediction, clipping ablation,
          4-class task, permutation tests, feature importance, efficiency

Usage: python -I scripts/phase2_mazza.py <data.csv> <out_dir> <stage> [--repeats 5] [--perm 200]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,  # noqa: E402
                             roc_auc_score, brier_score_loss)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from edd import efficiency as eff  # noqa: E402
from edd.folds import load_folds, save_folds  # noqa: E402
from edd.metrics import binary_metrics, bootstrap_ci, eer, summarise_repeats  # noqa: E402
from edd.mazza import TASKS, ece, fast_auc, load, raw_columns, run_binary, run_multiclass, selective, signed_score  # noqa: E402
from phase2_mazza_configs import CONFIGS, build  # noqa: E402

SUBSETS = {"all": (1, 2, 3, 4), "g23": (2, 3), "g12": (1, 2), "g24": (2, 4)}
TASK_SUBSET = {"guilty_vs_innocent": "all", "deceptive_vs_honest": "all", "g2_vs_g3": "g23",
               "g2_vs_g1": "g12", "g2_vs_g4": "g24"}


def task_data(d, task):
    groups, pos, _ = TASKS[task]
    s = d[d.group.isin(groups)].reset_index(drop=True)
    y = s.group.isin(pos).astype(int).to_numpy()
    return s, y


def stage_folds(d, out, repeats):
    folds = {}
    for name, groups in SUBSETS.items():
        s = d[d.group.isin(groups)]
        reps = []
        for r in range(repeats):
            skf = StratifiedKFold(10, shuffle=True, random_state=2026 + r)
            reps.append([s.unit.iloc[te].tolist() for _, te in skf.split(s, s.group)])
        folds[name] = reps
    save_folds(out / "folds.json", folds, meta={
        "design": "participant-disjoint outer folds; one row per participant; StratifiedKFold(10) stratified by "
                  "experimental group (so merged tasks keep subgroup composition), seeds 2026..2026+R-1",
        "subsets": {k: list(v) for k, v in SUBSETS.items()}, "task_subset": TASK_SUBSET,
        "inner": "StratifiedGroupKFold(5, shuffle, seed=1000*repeat+fold) on outer-training participants"})


def per_repeat_metrics(oof):
    return [binary_metrics(t.y, t.score) for _, t in oof.groupby("repeat")]


def avg_scores(oof):
    return oof.groupby("unit").agg(y=("y", "first"), score=("score", "mean")).reset_index()


PAIR_CONFIGS = ["base_pub36_lr_unreg", "i1_pub36_lr", "i1_raw90_lgbm", "i2_diffshare_lr", "i3_crime_lr",
                "i3_crime_diffshare_lr", "i4_all_lr_anova"]
SCREEN_REPEATS = 3   # repeats 0-2 of the frozen folds are used for screening; 3-4 are reserved


def stage_screen(d, out, only=None, tasks=None):
    folds = {k: v[:SCREEN_REPEATS] for k, v in load_folds(out / "folds.json").items()}
    raw = raw_columns(d)
    oof_dir = out / "oof"
    oof_dir.mkdir(exist_ok=True)
    timing = json.loads((out / "screen_timing.json").read_text()) if (out / "screen_timing.json").exists() else {}
    for task in tasks or TASKS:
        s, y = task_data(d, task)
        X, g = s[raw].to_numpy(float), s.unit.to_numpy()
        for name, (idea, sets, model, select, _) in CONFIGS.items():
            if only and name not in only:
                continue
            if task in ("g2_vs_g1", "g2_vs_g4") and name not in PAIR_CONFIGS:
                continue
            path = oof_dir / f"{task}__{name}.csv"
            if path.exists():
                continue
            t0 = time.time()
            oof = run_binary(X, y, g, folds[TASK_SUBSET[task]], build(raw, sets, model, select), n_jobs=2)
            oof.to_csv(path, index=False)
            timing[f"{task}__{name}"] = time.time() - t0
            m = summarise_repeats(per_repeat_metrics(oof))
            print(f"{task:22s} {name:24s} AUC {m['auc']['mean']:.3f}±{m['auc']['sd']:.3f} "
                  f"BA {m['balanced_accuracy']['mean']:.3f} EER {m['eer']['mean']:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
            (out / "screen_timing.json").write_text(json.dumps(timing, indent=1))


def paired_auc_test(y, s_a, s_b, n_boot=2000, seed=0):
    """Participant bootstrap (stratified) of AUC(a) - AUC(b) on repeat-averaged scores; two-sided p."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    obs = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(fast_auc(y[idx], s_a[idx]) - fast_auc(y[idx], s_b[idx]))
    diffs = np.asarray(diffs)
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"delta_auc": float(obs), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_boot": float(max(p, 1 / n_boot))}


def holm(pvals):
    keys = sorted(pvals, key=pvals.get)
    m, adj, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def load_oof(out, task, name):
    p = out / "oof" / f"{task}__{name}.csv"
    return pd.read_csv(p, dtype={"unit": str}) if p.exists() else None


def summarise_screen(out):
    res = {}
    for task in TASKS:
        base = load_oof(out, task, "base_pub36_lr_unreg")
        if base is None:
            continue
        b_avg = avg_scores(base)
        rows, pvals = {}, {}
        for name, (idea, sets, model, select, desc) in CONFIGS.items():
            oof = load_oof(out, task, name)
            if oof is None:
                continue
            a = avg_scores(oof)
            a = a.set_index("unit").loc[b_avg.unit].reset_index()
            rep = per_repeat_metrics(oof)
            rows[name] = {"idea": idea, "description": desc, "repeats": summarise_repeats(rep),
                          "avg_scores": binary_metrics(a.y, a.score),
                          "selective": selective(a.y.to_numpy(), a.score.to_numpy()),
                          "params": oof.params.value_counts().head(3).to_dict()}
            if name != "base_pub36_lr_unreg":
                rows[name]["vs_baseline"] = paired_auc_test(a.y.to_numpy(), a.score.to_numpy(), b_avg.score.to_numpy())
                rep_b = per_repeat_metrics(base)
                dif = [x["auc"] - z["auc"] for x, z in zip(rep, rep_b)]
                rows[name]["vs_baseline"]["per_repeat_delta_auc"] = {"mean": float(np.mean(dif)), "sd": float(np.std(dif))}
                pvals[name] = rows[name]["vs_baseline"]["p_boot"]
        for k, v in holm(pvals).items():
            rows[k]["vs_baseline"]["p_holm"] = v
        res[task] = rows
    return res


def pick_best(screen):
    """Best = highest mean participant AUC over repeats (screening choice, stated as such)."""
    return {t: max(rows, key=lambda k: rows[k]["repeats"]["auc"]["mean"]) for t, rows in screen.items()}


def permutation(d, out, task, name, n_perm, seed=0):
    folds = load_folds(out / "folds.json")[TASK_SUBSET[task]][:1]
    s, y = task_data(d, task)
    raw = raw_columns(d)
    X, g = s[raw].to_numpy(float), s.unit.to_numpy()
    idea, sets, model, select, _ = CONFIGS[name]
    obs_oof = load_oof(out, task, name)
    obs = binary_metrics(*obs_oof[obs_oof.repeat == 0][["y", "score"]].to_numpy().T)["auc"]
    rng = np.random.default_rng(seed)
    null, t0, budget_s = [], time.time(), 360.0
    i = 0
    while i < n_perm:
        yp = rng.permutation(y)
        o = run_binary(X, yp, g, folds, build(raw, sets, model, select), n_jobs=2)
        null.append(roc_auc_score(o.y, o.score))
        i += 1
        if i == 3:   # CPU budget: cap the number of permutations, but never below 50
            n_perm = int(min(n_perm, max(30, budget_s / ((time.time() - t0) / 3))))
    null = np.asarray(null)
    return {"n_perm": n_perm, "repeat": 0, "observed_auc": float(obs), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm))}


def calib_selective(d, out, task, name):
    folds = load_folds(out / "folds.json")[TASK_SUBSET[task]][:SCREEN_REPEATS]
    s, y = task_data(d, task)
    raw = raw_columns(d)
    X, g = s[raw].to_numpy(float), s.unit.to_numpy()
    idea, sets, model, select, _ = CONFIGS[name]
    oof = run_binary(X, y, g, folds, build(raw, sets, model, select), n_jobs=2, calibrate=True)
    oof.to_csv(out / "oof" / f"{task}__{name}__calibrated.csv", index=False)
    res = {"raw_prob": [], "platt_prob": [], "sel_raw": [], "sel_platt": []}
    for _, t in oof.groupby("repeat"):
        if "prob" in t:
            res["raw_prob"].append({"brier": brier_score_loss(t.y, t.prob), "ece": ece(t.y, t.prob)})
        res["platt_prob"].append({"brier": brier_score_loss(t.y, t.prob_cal), "ece": ece(t.y, t.prob_cal),
                                  "auc": roc_auc_score(t.y, t.prob_cal)})
        res["sel_raw"].append(selective(t.y.to_numpy(), t.score.to_numpy()))
        res["sel_platt"].append(selective(t.y.to_numpy(), t.prob_cal.to_numpy() - 0.5))
    summ = {}
    for k in ("raw_prob", "platt_prob"):
        if res[k]:
            summ[k] = {m: {"mean": float(np.mean([r[m] for r in res[k]])), "sd": float(np.std([r[m] for r in res[k]]))}
                       for m in res[k][0]}
    for k in ("sel_raw", "sel_platt"):
        summ[k] = {c: {m: {"mean": float(np.mean([r[c][m] for r in res[k]])), "sd": float(np.std([r[c][m] for r in res[k]]))}
                       for m in ("accuracy", "balanced_accuracy")} for c in res[k][0]}
    summ["note"] = ("selective = per repeat, keep the c most confident participants by |signed score| (sel_raw) or "
                    "|Platt prob - 0.5| (sel_platt); risk-coverage evaluation, not a deployed threshold")
    return summ


def clip_ablation(d, out, task, name):
    folds = load_folds(out / "folds.json")[TASK_SUBSET[task]][:SCREEN_REPEATS]
    s, y = task_data(d, task)
    raw = raw_columns(d)
    X, g = s[raw].to_numpy(float), s.unit.to_numpy()
    idea, sets, model, select, _ = CONFIGS[name]
    oof = run_binary(X, y, g, folds, build(raw, sets, model, select, clip=False), n_jobs=2)
    return summarise_repeats(per_repeat_metrics(oof))


def multiclass_metrics(t):
    classes = [1, 2, 3, 4]
    P = t[[f"p{c}" for c in classes]].to_numpy()
    y = t.y.to_numpy()
    pred = np.asarray(classes)[P.argmax(1)]
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"), "weighted_f1": f1_score(y, pred, average="weighted"),
            "macro_auc_ovr": roc_auc_score(y, P, multi_class="ovr", average="macro"),
            "weighted_auc_ovr": roc_auc_score(y, P, multi_class="ovr", average="weighted"),
            "macro_eer_ovr": float(np.mean([eer((y == c).astype(int), P[:, i]) for i, c in enumerate(classes)])),
            "macro_hter_ovr_argmax": float(np.mean([
                0.5 * (((pred == c) & (y != c)).sum() / (y != c).sum() + ((pred != c) & (y == c)).sum() / (y == c).sum())
                for c in classes])),
            "correct_per_group": {int(c): int(((pred == c) & (y == c)).sum()) for c in classes}}


def stage_multiclass(d, out):
    folds = load_folds(out / "folds.json")["all"][:SCREEN_REPEATS]
    raw = raw_columns(d)
    X, y, g = d[raw].to_numpy(float), d.group.to_numpy(), d.unit.to_numpy()
    res = {}
    for name in ["base_pub36_lr_unreg", "i1_pub36_lr", "i1_raw90_lr", "i1_raw90_lgbm", "i2_diffshare_lr",
                 "i3_crime_lr", "i3_crime_diffshare_lr", "i4_all_lr_anova"]:
        idea, sets, model, select, desc = CONFIGS[name]
        t0 = time.time()
        oof = run_multiclass(X, y, g, folds, build(raw, sets, model, select, multiclass=True), n_jobs=2)
        oof.to_csv(out / "oof" / f"4class__{name}.csv", index=False)
        reps = [multiclass_metrics(t) for _, t in oof.groupby("repeat")]
        keys = [k for k in reps[0] if not k.startswith("_") and k != "correct_per_group"]
        res[name] = {"description": desc,
                     "repeats": {k: {"mean": float(np.mean([r[k] for r in reps])), "sd": float(np.std([r[k] for r in reps]))}
                                 for k in keys},
                     "correct_per_group_repeat0": reps[0]["correct_per_group"], "seconds": time.time() - t0}
        print(f"4class {name:24s} macroAUC {res[name]['repeats']['macro_auc_ovr']['mean']:.3f} "
              f"acc {res[name]['repeats']['accuracy']['mean']:.3f}", flush=True)
    return res


def feature_importance(d, task, name):
    """Refit the pipeline on all participants of the task (descriptive only) and report the selected
    hyper-parameters, standardised LR coefficients (if linear) and univariate AUCs of the top features."""
    s, y = task_data(d, task)
    raw = raw_columns(d)
    idea, sets, model, select, _ = CONFIGS[name]
    search = build(raw, sets, model, select)(0)
    search.fit(s[raw].to_numpy(float), y, groups=s.unit.to_numpy())
    pipe = search.best_estimator_
    names = list(pipe.named_steps["feat"].get_feature_names_out())
    if "sel" in pipe.named_steps:
        sel = pipe.named_steps["sel"]
        mask = sel.get_support() if hasattr(sel, "get_support") else sel.support_
        names = [n for n, k in zip(names, mask) if k]
    clf = pipe.named_steps["clf"]
    out = {"best_params": search.best_params_}
    if hasattr(clf, "coef_"):
        coef = clf.coef_.ravel()
        order = np.argsort(-np.abs(coef))[:15]
        out["top_coefficients"] = [[names[i], float(coef[i])] for i in order]
    F = pipe.named_steps["feat"].transform(s[raw].to_numpy(float))
    fn = list(pipe.named_steps["feat"].get_feature_names_out())
    uni = sorted(((fn[j], float(roc_auc_score(y, F[:, j]))) for j in range(F.shape[1])),
                 key=lambda x: -abs(x[1] - 0.5))[:12]
    out["top_univariate_auc"] = uni
    return out


def efficiency(d, task, name):
    s, y = task_data(d, task)
    raw = raw_columns(d)
    idea, sets, model, select, _ = CONFIGS[name]
    search = build(raw, sets, model, select)(0)
    X = s[raw].to_numpy(float)
    t0 = time.perf_counter()
    search.fit(X, y, groups=s.unit.to_numpy())
    fit_s = time.perf_counter() - t0
    pipe = search.best_estimator_
    one = X[:1]
    feat = pipe.named_steps["feat"]
    clf = pipe.named_steps["clf"]
    n_feat = len(feat.get_feature_names_out())
    if "sel" in pipe.named_steps:
        sel = pipe.named_steps["sel"]
        n_feat = int((sel.get_support() if hasattr(sel, "get_support") else sel.support_).sum())
    if hasattr(clf, "coef_"):
        params = int(clf.coef_.size + clf.intercept_.size)
    elif hasattr(clf, "support_vectors_"):
        params = eff.svm_parameter_count(clf, n_feat)
    elif hasattr(clf, "booster_"):
        params = int(clf.booster_.trees_to_dataframe().shape[0])
    else:
        params = int(sum(t.tree_.node_count for t in clf.estimators_))
    n_scaled = pipe.named_steps["scale"].mean_.size   # scaler mean/scale, imputer medians, clip bounds
    pre = n_scaled * (3 + (2 if "clip" in pipe.named_steps else 0))
    return {"config": name, "task": task, "n_model_inputs": n_feat,
            "fit_seconds_full_nested_search": fit_s,
            "latency_ms_per_participant_end_to_end": eff.latency_ms(lambda: signed_score(pipe, one), n_runs=500),
            "latency_ms_feature_extraction_only": eff.latency_ms(lambda: feat.transform(one), n_runs=500),
            "throughput_participants_per_s_batch157": eff.throughput_per_s(lambda: signed_score(pipe, d[raw].to_numpy(float)), len(d)),
            "peak_memory_bytes_inference_one": eff.peak_memory_bytes(lambda: signed_score(pipe, one)),
            "model_size_bytes": eff.model_size_bytes(pipe),
            "stored_parameters_classifier": params, "stored_parameters_preprocessing_approx": pre}


def block_ablation(d, out, task):
    """L2-LR on one raw block at a time. `neutral` is the negative control."""
    folds = load_folds(out / "folds.json")[TASK_SUBSET[task]][:SCREEN_REPEATS]
    s, y = task_data(d, task)
    raw = raw_columns(d)
    X, g = s[raw].to_numpy(float), s.unit.to_numpy()
    res = {}
    for blk in ("neutral", "target_overall", "scanpath", "no_neutral"):
        oof = run_binary(X, y, g, folds, build(raw, [blk], "lr"), n_jobs=2)
        oof.to_csv(out / "oof" / f"{task}__block_{blk}.csv", index=False)
        a = avg_scores(oof)
        yy, ss = a.y.to_numpy(), a.score.to_numpy()
        rng = np.random.default_rng(0)
        pos, neg = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
        boots = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            boots.append(fast_auc(yy[idx], ss[idx]))
        res[blk] = {"repeats": summarise_repeats(per_repeat_metrics(oof)),
                    "ci95_auc": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return res


def reserved_repeats(d, out, task, name):
    """Same participants, the two fold seeds held back from screening: controls fold-seed luck only."""
    folds = load_folds(out / "folds.json")[TASK_SUBSET[task]][SCREEN_REPEATS:]
    s, y = task_data(d, task)
    raw = raw_columns(d)
    X, g = s[raw].to_numpy(float), s.unit.to_numpy()
    idea, sets, model, select, _ = CONFIGS[name]
    oof = run_binary(X, y, g, folds, build(raw, sets, model, select), n_jobs=2)
    return summarise_repeats(per_repeat_metrics(oof))


def stage_extras(d, out, n_perm):
    screen_path = out / "screen_summary.json"
    screen = json.loads(screen_path.read_text()) if screen_path.exists() else summarise_screen(out)
    (out / "screen_summary.json").write_text(json.dumps(screen, indent=1, default=str))
    best = pick_best(screen)
    print("best per task:", best, flush=True)
    cache_path = out / "extras_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def cached(key, fn):
        if key not in cache:
            t0 = time.time()
            cache[key] = fn()
            cache_path.write_text(json.dumps(cache, indent=1, default=str))
            print(f"done {key} ({time.time() - t0:.0f}s)", flush=True)
        return cache[key]

    main_tasks = ("guilty_vs_innocent", "deceptive_vs_honest", "g2_vs_g3")
    extras = {"best": best, "ci95": {}, "permutation": {}, "calibration_selective": {}, "clip_ablation": {},
              "feature_importance": {}, "block_ablation": {}, "reserved_repeats": {}}
    base = "base_pub36_lr_unreg"
    for task, name in best.items():
        names = list(dict.fromkeys([name, base]))
        for nm in names:
            a = avg_scores(load_oof(out, task, nm))
            extras["ci95"][f"{task}__{nm}"] = cached(f"ci|{task}|{nm}", lambda a=a: bootstrap_ci(
                a.y.to_numpy(), a.score.to_numpy(), n_boot=2000))
        extras["calibration_selective"][task] = {
            nm: cached(f"calib|{task}|{nm}", lambda nm=nm: calib_selective(d, out, task, nm)) for nm in names}
        extras["clip_ablation"][task] = {"config": name, "no_clip": cached(f"noclip|{task}|{name}",
                                                                           lambda: clip_ablation(d, out, task, name))}
        extras["feature_importance"][task] = {name: cached(f"fi|{task}|{name}", lambda: feature_importance(d, task, name))}
        if task in main_tasks:
            extras["block_ablation"][task] = cached(f"blocks|{task}", lambda: block_ablation(d, out, task))
            extras["reserved_repeats"][task] = {nm: cached(f"reserved|{task}|{nm}",
                                                           lambda nm=nm: reserved_repeats(d, out, task, nm)) for nm in names}
            extras["permutation"][task] = cached(f"perm|{task}|{name}", lambda: permutation(d, out, task, name, n_perm))
    extras["multiclass"] = cached("multiclass", lambda: stage_multiclass(d, out))
    extras["efficiency"] = {"hardware": eff.hardware_info(),
                            "baseline": efficiency(d, "g2_vs_g3", base)}
    for task, name in best.items():
        extras["efficiency"][f"best_{task}"] = efficiency(d, task, name)
    (out / "extras.json").write_text(json.dumps(extras, indent=1, default=str))
    timing = json.loads((out / "screen_timing.json").read_text())
    results = {"dataset": "Colasanti/Mazza et al. RealEye crime scene, Zenodo 10.5281/zenodo.7194229",
               "unit_of_decision": "participant (one aggregate row per participant)",
               "outer_cv": "stratified 10-fold by group; 3 repeats for screening, 2 further frozen fold seeds held back",
               "tasks": {k: v[2] for k, v in TASKS.items()}, "configs": {k: v[4] for k, v in CONFIGS.items()},
               "screen": screen, **extras, "screen_seconds_total": float(sum(timing.values())),
               "screen_seconds": timing}
    repro = out / "reproduction.json"
    if repro.exists():
        results["reproduction"] = json.loads(repro.read_text())
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("stage", choices=["folds", "screen", "extras", "summary"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(a.data)
    if a.stage == "folds":
        stage_folds(d, out, a.repeats)
    elif a.stage == "screen":
        stage_screen(d, out, a.only, a.tasks)
    elif a.stage == "extras":
        stage_extras(d, out, a.perm)
    else:
        print(json.dumps({t: {k: round(v["repeats"]["auc"]["mean"], 3) for k, v in r.items()}
                          for t, r in summarise_screen(out).items()}, indent=1))


if __name__ == "__main__":
    main()
