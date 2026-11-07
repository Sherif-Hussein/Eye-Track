"""Phase 2 / liars_dice: idea screening on Foucher & Huckauf (PsyArXiv 2026) Liar's Dice data.

Trial (bid)-level label, participant-disjoint outer folds (PvP: dyad-disjoint), nested grouped tuning,
3 repeats. Ideas: (1) the paper's effects as a classifier, (2) LR/LightGBM on ocular features,
(3) online within-person baselines, (4) ocular vs RT vs context vs fusion, (5) PvC <-> PvP transfer.

Usage: python -I scripts/phase2_liars_dice.py <bids_csv> <out_dir> <oof_csv_gz> [--quick] [--n-perm N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from edd import efficiency as eff  # noqa: E402
from edd.folds import save_folds  # noqa: E402
from edd.harness import fixed_fit, grouped_inner, make_search, nested_fit, scores_of  # noqa: E402
from edd.liars_dice import (CONTEXT, OC_AOI, OC_COMMON, PAPER_AOI, PAPER_OCULAR, PAPER_RT, RT,  # noqa: E402
                            OnlineBaseline, SignedZScore, add_past_stats, bid_features_arrays, cluster_bootstrap,
                            evaluate, online_cols, paired_cluster_bootstrap_diff, per_participant,
                            person_offline_center, run_cv)
from edd.metrics import binary_metrics, bootstrap_ci  # noqa: E402

N_REPEATS, K_OUTER, BASE_SEED = 3, 5, 2026


# ---------------------------------------------------------------- folds
def make_folds(bids, n_repeats):
    """Per experiment, K_OUTER folds over participants (PvP: over dyads, both members in the same fold)."""
    out = {}
    for exp, t in bids.groupby("exp"):
        members = t.groupby("group")["pid"].unique()
        groups = np.array(sorted(members.index))
        reps = []
        for r in range(n_repeats):
            perm = np.random.default_rng(BASE_SEED + r).permutation(groups)
            reps.append([sorted(str(p) for gname in chunk for p in members[gname])
                         for chunk in np.array_split(perm, K_OUTER)])
        out[exp] = reps
    return out


# ---------------------------------------------------------------- model factories
def _clf(kind):
    if kind == "lr":
        return LogisticRegression(class_weight="balanced", max_iter=3000)
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, class_weight="balanced", n_jobs=1, verbose=-1, random_state=0)


def model_spec(kind, online=(), raw=(), keep_raw=True, k=None, seed=0):
    """Returns (columns, fit_fn). `online` features get the online person baseline (deviation from the
    shrunken running past mean, plus the raw value when keep_raw); `raw` features pass through."""
    online, raw = list(online), list(raw)
    cols = online_cols(online) + raw
    steps = []
    if online:
        p = len(online)
        steps.append(("ct", ColumnTransformer([("ob", OnlineBaseline(k=k or 5.0, keep_raw=keep_raw), list(range(3 * p))),
                                               ("raw", "passthrough", list(range(3 * p, 3 * p + len(raw))))])))
    if kind == "lr":
        steps += [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    steps.append(("clf", _clf(kind)))
    grid = ({"clf__C": [0.01, 0.1, 1.0]} if kind == "lr"
            else {"clf__num_leaves": [7, 15], "clf__min_child_samples": [20, 60]})
    if online and kind == "lr" and k is None:
        grid["ct__ob__k"] = [2.0, 10.0]
    pipe = Pipeline(steps)
    return cols, nested_fit(lambda: make_search(pipe, grid, grouped_inner(3, seed)))


def paper_spec(signs, online=False):
    feats = list(signs)
    cols = online_cols(feats) if online else feats
    return cols, fixed_fit(SignedZScore(signs=tuple(signs.values()), online=online, k=5.0))


def ideas_for(exp):
    oc = OC_COMMON + (OC_AOI if exp == "pvp" else [])
    pc_oc = [f"pc__{f}" for f in oc]
    S = {}
    # Idea 1: the paper's reported effects as a classifier (honest baseline = paper ocular measures).
    S["I1_paper_ocular"] = ("ocular", *paper_spec(PAPER_OCULAR))
    S["I1_paper_ocular_rt"] = ("ocular+rt", *paper_spec({**PAPER_OCULAR, **PAPER_RT}))
    if exp == "pvp":
        S["I1_paper_ocular_aoi"] = ("ocular", *paper_spec({**PAPER_OCULAR, **PAPER_AOI}))
    S["I1_paper_ocular_online"] = ("ocular", *paper_spec(PAPER_OCULAR if exp == "pvc" else {**PAPER_OCULAR, **PAPER_AOI}, online=True))
    # Idea 2: ML baselines on ocular features.
    S["I2_lr_ocular"] = ("ocular", *model_spec("lr", raw=oc))
    S["I2_lgbm_ocular"] = ("ocular", *model_spec("lgbm", raw=oc))
    if exp == "pvp":
        S["I2_lr_ocular_common"] = ("ocular", *model_spec("lr", raw=OC_COMMON))
    # Idea 3: online within-person normalisation (past bids only), and a transductive reference.
    S["I3_lr_ocular_online"] = ("ocular", *model_spec("lr", online=oc, keep_raw=False))
    S["I3_lr_ocular_online_raw"] = ("ocular", *model_spec("lr", online=oc, keep_raw=True))
    S["I3_lgbm_ocular_online_raw"] = ("ocular", *model_spec("lgbm", online=oc, keep_raw=True, k=5.0))
    S["I3_lr_ocular_offline_TRANSDUCTIVE"] = ("ocular", *model_spec("lr", raw=pc_oc + oc))
    S["I3_lr_ocular_online_allroles"] = ("ocular", *model_spec("lr", online=oc, keep_raw=True))  # run on all-role baseline
    if exp == "pvp":
        S["I3_lr_ocular_common_online_raw"] = ("ocular", *model_spec("lr", online=OC_COMMON, keep_raw=True))
    # Idea 4: RT, context, fusion.
    S["I4_lr_rt"] = ("rt", *model_spec("lr", online=RT, keep_raw=True))
    S["I4_lr_context"] = ("context", *model_spec("lr", raw=CONTEXT))
    S["I4_lgbm_context"] = ("context", *model_spec("lgbm", raw=CONTEXT))
    S["I4_lgbm_context_rt"] = ("context+rt", *model_spec("lgbm", online=RT, raw=CONTEXT, k=5.0))
    S["I4_lr_ocular_rt"] = ("ocular+rt", *model_spec("lr", online=oc + RT, keep_raw=True))
    S["I4_lr_fusion"] = ("fusion", *model_spec("lr", online=oc + RT, raw=CONTEXT, keep_raw=True))
    S["I4_lgbm_context_ocular"] = ("context+ocular", *model_spec("lgbm", online=oc, raw=CONTEXT, k=5.0))
    S["I4_lgbm_fusion"] = ("fusion", *model_spec("lgbm", online=oc + RT, raw=CONTEXT, k=5.0))
    return S


# ---------------------------------------------------------------- analysis helpers
def avg_scores(oof):
    return oof.groupby("row").agg(pid=("pid", "first"), y=("y", "first"), score=("score", "mean"))


def auc_within_bidpos(oof, d):
    """Deception signal not explained by bid position: AUC within strata bid 1 / 2 / >=3, n-weighted."""
    a = avg_scores(oof)
    pos = d["bid_pos"].to_numpy()[a.index.to_numpy()]
    strata = np.where(pos >= 3, 3, pos)
    res, w = {}, []
    for s in (1, 2, 3):
        m = strata == s
        if a.y[m].nunique() == 2:
            res[f"bid{s}{'+' if s == 3 else ''}"] = float(roc_auc_score(a.y[m], a.score[m]))
            w.append((m.sum(), res[f"bid{s}{'+' if s == 3 else ''}"]))
    res["weighted"] = float(sum(n * v for n, v in w) / sum(n for n, _ in w))
    return res


def holm(pvals):
    names = sorted(pvals, key=pvals.get)
    m, out, running = len(names), {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[n]))
        out[n] = running
    return out


def compare(oofs, d, base, names, n_boot):
    """Paired comparisons vs `base` on identical rows: participant-cluster bootstrap of the trial-AUC
    difference and Wilcoxon on per-participant AUCs; Holm across `names`."""
    ab = avg_scores(oofs[base])
    pb = per_participant(oofs[base]).set_index("pid").auc
    res, pv = {}, {}
    for n in names:
        a = avg_scores(oofs[n]).loc[ab.index]
        r = paired_cluster_bootstrap_diff(ab.y, a.score, ab.score, ab.pid, n_boot=n_boot)
        pp = per_participant(oofs[n]).set_index("pid").auc
        j = pd.concat([pp, pb], axis=1, keys=["a", "b"]).dropna()
        r["per_participant_auc_diff_mean"] = float((j.a - j.b).mean())
        r["wilcoxon_p"] = float(wilcoxon(j.a, j.b).pvalue)
        res[n] = r
        pv[n] = r["p_two_sided"]
    for n, p in holm(pv).items():
        res[n]["holm_p"] = p
        res[n]["reliable"] = bool(p < 0.05)
    return res


def within_participant_permutation(d, cols, fit_fn, folds, n_perm, seed=0):
    """Labels shuffled within each participant (keeps each person's lie rate; destroys the bid-wise link
    between eyes and honesty). For CPU budget the hyperparameters are frozen at the values the nested search
    picks on all data; the observed AUC is recomputed with the same frozen pipeline on the same folds, so
    observed and null are computed identically."""
    m = fit_fn(d[cols].to_numpy(float), d["y"].to_numpy(int), d["group"].astype(str).to_numpy())
    frozen = fixed_fit(getattr(m, "best_estimator_", m))
    o = run_cv(d, cols, frozen, [folds])
    observed = float(roc_auc_score(o.y, o.score))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        dd = d.copy()
        dd["y"] = dd.groupby("pid")["y"].transform(lambda s: rng.permutation(s.to_numpy()))
        o = run_cv(dd, cols, frozen, [folds])
        null.append(roc_auc_score(o.y, o.score))
    null = np.asarray(null)
    return {"n_perm": n_perm, "frozen_params": str(getattr(m, "best_params_", "")),
            "observed_trial_auc_repeat0_frozen": observed, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + np.sum(null >= observed)) / (1 + n_perm))}


# ---------------------------------------------------------------- efficiency
def param_count(est):
    if isinstance(est, SignedZScore):
        n = est.mu_.size + est.sd_.size + 1 + len(est.signs)
        return int(n + (est.ob_.m0_.size if est.online else 0))
    n = 0
    for _, step in est.steps:
        if isinstance(step, ColumnTransformer):
            n += step.named_transformers_["ob"].m0_.size
        elif isinstance(step, SimpleImputer):
            n += step.statistics_.size
        elif isinstance(step, StandardScaler):
            n += step.mean_.size + step.scale_.size
        elif isinstance(step, LogisticRegression):
            n += step.coef_.size + step.intercept_.size
        elif isinstance(step, LGBMClassifier):
            dump = step.booster_.dump_model()
            for t in dump["tree_info"]:
                leaves = t["num_leaves"]
                n += leaves + 3 * (leaves - 1)  # leaf values + (feature, threshold, default-direction) per split
    return int(n)


def efficiency_block(d, cols, fit_fn, raw_bid):
    m = fit_fn(d[cols].to_numpy(float), d["y"].to_numpy(int), d["group"].astype(str).to_numpy())
    est = getattr(m, "best_estimator_", m)
    X = d[cols].to_numpy(float)
    pid_rows = d.groupby("pid").indices
    p_med = sorted(pid_rows, key=lambda p: len(pid_rows[p]))[len(pid_rows) // 2]
    Xp = X[pid_rows[p_med]]
    x1 = X[:1]
    fix_d, blk_d, bt, aoi = raw_bid
    return {
        "latency_ms_per_bid": eff.latency_ms(lambda: scores_of(est, x1), n_runs=500),
        "latency_ms_per_participant": eff.latency_ms(lambda: scores_of(est, Xp), n_runs=200),
        "bids_in_that_participant": int(len(Xp)),
        "throughput_bids_per_s": eff.throughput_per_s(lambda: scores_of(est, X), len(X), seconds=1.0),
        "peak_memory_bytes_inference_participant": eff.peak_memory_bytes(lambda: scores_of(est, Xp)),
        "model_size_bytes": eff.model_size_bytes(est),
        "parameter_count": param_count(est),
        "feature_extraction_ms_per_bid_from_events": eff.latency_ms(lambda: bid_features_arrays(fix_d, blk_d, bt, aoi), n_runs=500),
        "params": str(getattr(m, "best_params_", "")),
    }


# ---------------------------------------------------------------- transfer
def transfer(bids_on, n_boot):
    """Train on all of one experiment, test on all of the other (common ocular features only)."""
    res = {}
    specs = {"paper_ocular": paper_spec(PAPER_OCULAR), "paper_ocular_online": paper_spec(PAPER_OCULAR, online=True),
             "lr_ocular_common": model_spec("lr", raw=OC_COMMON),
             "lr_ocular_common_online_raw": model_spec("lr", online=OC_COMMON, keep_raw=True),
             "lr_ocular_common_online": model_spec("lr", online=OC_COMMON, keep_raw=False)}
    for src, tgt in (("pvc", "pvp"), ("pvp", "pvc")):
        S, T = bids_on[bids_on.exp == src], bids_on[bids_on.exp == tgt]
        for name, (cols, fit_fn) in specs.items():
            m = fit_fn(S[cols].to_numpy(float), S.y.to_numpy(int), S.group.astype(str).to_numpy())
            est = getattr(m, "best_estimator_", m)
            s = scores_of(est, T[cols].to_numpy(float))
            oof = pd.DataFrame({"row": np.arange(len(T)), "repeat": 0, "fold": 0, "score": s,
                                "y": T.y.to_numpy(), "pid": T.pid.to_numpy()})
            ev = evaluate(oof)
            res[f"{src}_to_{tgt}/{name}"] = {"trial": binary_metrics(T.y, s),
                                             "trial_cluster_ci95": cluster_bootstrap(T.y, s, T.pid, n_boot=n_boot),
                                             "per_participant": ev["per_participant"],
                                             "params": str(getattr(m, "best_params_", ""))}
            print(f"transfer {src}->{tgt} {name}: AUC {res[f'{src}_to_{tgt}/{name}']['trial']['auc']:.3f}", flush=True)
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bids_csv"); ap.add_argument("out_dir"); ap.add_argument("oof_csv")
    ap.add_argument("--raw-dir", default=None, help="Rda folder, for feature-extraction cost")
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--n-perm", type=int, default=100)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_rep = 1 if a.quick else N_REPEATS
    n_boot = 300 if a.quick else 2000

    allb = pd.read_csv(a.bids_csv)
    all_feats = OC_COMMON + OC_AOI + RT
    on_bidder = add_past_stats(allb, all_feats, roles=("Bidder",))
    on_all = add_past_stats(allb, all_feats, roles=("Bidder", "Challenger"))
    keep = allb.Role == "Bidder"
    on_bidder = on_bidder[keep].reset_index(drop=True)
    on_all = on_all[keep].reset_index(drop=True)
    for f in OC_COMMON + OC_AOI:
        on_bidder[f"pc__{f}"] = np.nan
    for exp in ("pvc", "pvp"):
        oc = OC_COMMON + (OC_AOI if exp == "pvp" else [])
        sub = on_bidder.exp == exp
        on_bidder.loc[sub, [f"pc__{f}" for f in oc]] = person_offline_center(on_bidder[sub], oc)[[f"pc__{f}" for f in oc]].to_numpy()

    folds = make_folds(on_bidder, n_rep)
    folds_path = out / "folds.json"
    if not a.quick:
        save_folds(folds_path, folds, meta={"unit": "pid", "k_outer": K_OUTER, "base_seed": BASE_SEED,
                                            "note": "per experiment; PvP folds are dyad-disjoint (both players of a dyad in the same fold)"})

    results = {"meta": {"hardware": eff.hardware_info(), "n_repeats": n_rep, "k_outer": K_OUTER, "quick": a.quick,
                        "label_recode": "deceptive = released BidHonesty 'Truth' (see data_audit.json)"},
               "ideas": {}, "within_bidpos_auc": {}, "comparisons": {}}
    oof_all, oofs = [], {}
    # Checkpoint: post-CV stages already finished for an experiment are reused from results_partial.json.
    partial_path = out / "results_partial.json"
    partial = json.loads(partial_path.read_text()) if (partial_path.exists() and not a.quick) else {}
    for exp in ("pvc", "pvp"):
        d = on_bidder[on_bidder.exp == exp].reset_index(drop=True)
        d_all = on_all[on_all.exp == exp].reset_index(drop=True)
        results["ideas"][exp], results["within_bidpos_auc"][exp] = {}, {}
        oofs[exp] = {}
        specs = ideas_for(exp)
        if a.quick:
            specs = {k: v for k, v in specs.items() if "lgbm" not in k or k == "I4_lgbm_fusion"}
        for name, (block, cols, fit_fn) in specs.items():
            t0 = time.time()
            data = d_all if name.endswith("allroles") else d
            cache = Path(a.oof_csv).parent / "liars_dice_oofcache" / f"{exp}__{name}__r{n_rep}.csv"
            if cache.exists():
                oof = pd.read_csv(cache)
            else:
                oof = run_cv(data, cols, fit_fn, folds[exp])
                cache.parent.mkdir(parents=True, exist_ok=True)
                oof.to_csv(cache, index=False)
            ev = evaluate(oof)
            ev["block"] = block
            ev["seconds"] = round(time.time() - t0, 1)
            ev["params_chosen"] = oof.groupby(["repeat", "fold"]).params.first().value_counts().head(3).to_dict()
            results["ideas"][exp][name] = ev
            results["within_bidpos_auc"][exp][name] = auc_within_bidpos(oof, d)
            oofs[exp][name] = oof
            oof_all.append(oof.assign(exp=exp, idea=name))
            tr, pp = ev["trial"], ev["per_participant"]
            print(f"[{exp}] {name:38s} {block:14s} AUC {tr['auc']['mean']:.3f}±{tr['auc']['sd']:.3f} "
                  f"BA {tr['balanced_accuracy']['mean']:.3f} EER {tr['eer']['mean']:.3f} HTER {tr['hter']['mean']:.3f} | "
                  f"ppAUC {pp['auc_mean']:.3f} det {pp['detected_alpha05']}/{pp['n_participants']} | "
                  f"bidpos-AUC {results['within_bidpos_auc'][exp][name]['weighted']:.3f} | {ev['seconds']}s", flush=True)

        if exp in partial.get("best", {}) and exp in partial.get("efficiency", {}) and (a.n_perm == 0 or
                "permutation_within_participant" in partial["best"][exp]["best_ocular_candidate"]):
            for key in ("comparisons", "best", "efficiency"):
                results.setdefault(key, {})[exp] = partial[key][exp]
            print(f"[{exp}] post-CV stages reused from results_partial.json", flush=True)
            (out / "results_partial.json").write_text(json.dumps(results, indent=1, default=float))
            continue

        # Ocular-only ideas vs the honest baseline (paper ocular measures as a classifier); Holm across them.
        base = "I1_paper_ocular"
        oc_names = [n for n in oofs[exp] if results["ideas"][exp][n]["block"] == "ocular" and n != base
                    and "TRANSDUCTIVE" not in n]
        results["comparisons"][exp] = {"ocular_vs_honest_baseline": compare(oofs[exp], d, base, oc_names, n_boot)}
        # What the eyes add beyond game context (+RT).
        ctx_pairs = {}
        if "I4_lgbm_context_rt" in oofs[exp] and "I4_lgbm_fusion" in oofs[exp]:
            ctx_pairs["fusion_vs_context_rt"] = compare(oofs[exp], d, "I4_lgbm_context_rt", ["I4_lgbm_fusion"], n_boot)
        if "I4_lgbm_context" in oofs[exp] and "I4_lgbm_context_ocular" in oofs[exp]:
            ctx_pairs["context_ocular_vs_context"] = compare(oofs[exp], d, "I4_lgbm_context", ["I4_lgbm_context_ocular"], n_boot)
        if "I4_lr_rt" in oofs[exp] and "I4_lr_ocular_rt" in oofs[exp]:
            ctx_pairs["ocular_rt_vs_rt"] = compare(oofs[exp], d, "I4_lr_rt", ["I4_lr_ocular_rt"], n_boot)
        results["comparisons"][exp]["eyes_add"] = ctx_pairs

        # Best ocular-only (online-deployable) candidate by mean trial AUC.
        best = max(oc_names, key=lambda n: results["ideas"][exp][n]["trial"]["auc"]["mean"])
        bo, bb = avg_scores(oofs[exp][best]), avg_scores(oofs[exp][base])
        block_best = {"name": best,
                      "trial_cluster_ci95": cluster_bootstrap(bo.y, bo.score, bo.pid, n_boot=n_boot),
                      "trial_row_bootstrap_ci95_edd": bootstrap_ci(bo.y.to_numpy(), bo.score.to_numpy(), n_boot=min(n_boot, 1000)),
                      "trial_metrics_repeat_avg_scores": binary_metrics(bo.y, bo.score)}
        block_base = {"name": base,
                      "trial_cluster_ci95": cluster_bootstrap(bb.y, bb.score, bb.pid, n_boot=n_boot),
                      "trial_row_bootstrap_ci95_edd": bootstrap_ci(bb.y.to_numpy(), bb.score.to_numpy(), n_boot=min(n_boot, 1000)),
                      "trial_metrics_repeat_avg_scores": binary_metrics(bb.y, bb.score)}
        # Per-participant CI on mean per-participant AUC (participant bootstrap).
        for blk, nm in ((block_best, best), (block_base, base)):
            pp = per_participant(oofs[exp][nm]).auc.dropna().to_numpy()
            rng = np.random.default_rng(0)
            bs = [rng.choice(pp, len(pp)).mean() for _ in range(n_boot)]
            blk["per_participant_auc_mean_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        if a.n_perm and not a.quick:
            _, cols, fit_fn = ideas_for(exp)[best]
            data = d_all if best.endswith("allroles") else d
            t0 = time.time()
            block_best["permutation_within_participant"] = within_participant_permutation(
                data, cols, fit_fn, folds[exp][0], a.n_perm)
            block_best["permutation_within_participant"]["seconds"] = round(time.time() - t0, 1)
        results.setdefault("best", {})[exp] = {"best_ocular_candidate": block_best, "honest_baseline": block_base}
        print(f"[{exp}] best ocular candidate: {best}", flush=True)

        # Efficiency on this laptop, single-threaded.
        if a.raw_dir:
            raw_bid = load_raw_bid(a.raw_dir, exp)
            results.setdefault("efficiency", {})[exp] = {}
            for nm in (base, best):
                _, cols, fit_fn = ideas_for(exp)[nm]
                data = d_all if nm.endswith("allroles") else d
                results["efficiency"][exp][nm] = efficiency_block(data, cols, fit_fn, raw_bid)
        (out / "results_partial.json").write_text(json.dumps(results, indent=1, default=float))

    results["transfer"] = transfer(on_bidder, n_boot)
    results["meta"]["total_seconds"] = round(time.time() - t_start, 1)
    (out / ("results_quick.json" if a.quick else "results.json")).write_text(json.dumps(results, indent=1, default=float))
    pd.concat(oof_all).to_csv(a.oof_csv, index=False)
    print("total seconds", results["meta"]["total_seconds"])


def load_raw_bid(raw_dir, exp):
    """A median-length bidder bid's raw event rows (fixation durations, AOIs, blink durations)."""
    from edd.liars_dice import EXPS, KEYS, _read_rda
    fx = _read_rda(f"{raw_dir}/{EXPS[exp]}_fixations.Rda")
    bl = _read_rda(f"{raw_dir}/{EXPS[exp]}_blinks.Rda")
    sizes = fx[fx.Role == "Bidder"].groupby(KEYS).size()
    key = sizes.index[np.argsort(sizes.to_numpy())[len(sizes) // 2]]
    g = fx.set_index(KEYS).loc[key]
    b = bl.set_index(KEYS).loc[[key]]["BlinkDuration"].to_numpy(float) if key in bl.set_index(KEYS).index else np.empty(0)
    aoi = g["AOI"].astype(str).to_numpy() if "AOI" in g.columns else None
    return g["FixDuration"].to_numpy(float), b, float(g["BidTime"].iloc[0]), aoi


if __name__ == "__main__":
    main()
