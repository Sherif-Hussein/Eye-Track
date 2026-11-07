"""R10 (exploratory, post-review): are surrogate-innocent false-positive rates a fair proxy for real-innocent FPRs?

Datasets with real innocents: Celniak et al. (task celniak_unified) and Colasanti/Mazza guilty-vs-innocent.
Pipeline = the stored fixed pipeline of results/phase4/unified (features, within-person scaling, L2 LR balanced, C tuned
in inner folds, frozen outer folds, threshold flagging 7% of training innocent cases). Stored outer-fold models are
rescored (coef/intercept/C from the checkpoints; imputation + standardisation refitted by the pipeline's own prep_fit on
the stored training rows). The only refits are the inner-fold LR fits at the stored C (needed to get inner out-of-fold
scores of surrogate cases for the surrogate-calibrated threshold), on the frozen folds, <= 3 repeats.

One process, single thread. Run:  .venv\\Scripts\\python.exe -I results\\phase8\\R10_surrogate_check\\run_r10.py [--repro-only]
"""
import json
import os
import pickle
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from edd import unified as U  # noqa: E402

OUT = Path(__file__).resolve().parent
UNI = ROOT / "results" / "phase4" / "unified"
SEED_SURR = 20261110
SEED_BOOT = 20261111
D = 20                      # random substitutions
NBOOT = 1000
MAX_REFIT_REPEATS = 3       # inner refits only on repeats 0..2
VARIANTS = ("leave_out", "in_reference")
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def cp_ci(k, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return [lo, hi]


# ============================================================================= Celniak surrogate features
def celniak_parts():
    """Per slide: item statistics st (K,18), item pupil (K,), slide pupil, probe position; in the row order of the cache."""
    from edd import celniak, celniak_fast as cf
    cache = UNI / "cache" / "celniak_fast_slides.csv"
    df = pd.read_csv(cache)
    recs = list(dict.fromkeys(df.recording))
    ST, PUP, SP, TP = [], [], [], []
    for name in recs:
        samples, meta = celniak.load_cached(ROOT / "data" / "celniak_cache", name)
        r = cf.arrays_from_frame(samples)
        dt_us = float(np.median(np.diff(r.ts)))
        pupil = cf.pupil_trace(r)
        baseline = cf.pupil_baseline_fast(r, pupil)
        S, K = r.slide_cols.shape
        grp = np.where(r.stim >= 0, r.slide_of_code[np.maximum(r.stim, 0)], -1)
        ev = cf._events(r, grp, r.slide_cols, dt_us)
        isf = ev.kind == cf.FIX
        fg, fs_, fe, fa, fps = ev.g[isf], ev.start[isf], ev.end[isf], ev.aoi[isf], ev.prev_sac_start[isf]
        nf = len(fg)
        same_prev = np.zeros(nf, bool)
        same_prev[1:] = (fg[1:] == fg[:-1]) & (fa[1:] == fa[:-1])
        new_run = (fa >= 0) & ~same_prev
        last = np.zeros(nf, bool)
        last[:-1] = (fg[:-1] == fg[1:]) & (fa[:-1] == fa[1:])
        last = (fa >= 0) & ~last
        run_g, run_a = fg[new_run], fa[new_run]
        run_start, run_gl = fs_[new_run], fps[new_run]
        run_end = fe[last]
        onaoi = fa >= 0
        dur_f = (fe - fs_) / 1e3
        keys = {"fixation": (fg[onaoi] * K + fa[onaoi], dur_f[onaoi]),
                "visit": (run_g * K + run_a, (run_end - run_start) / 1e3),
                "glance": (run_g * K + run_a, (run_end - run_gl) / 1e3)}
        st = np.concatenate([cf._seg_stats(k, d, S * K).reshape(S, K, 6) for k, d in keys.values()], axis=2)
        pm = ev.row_ev
        rows_aoi = ev.aoi[pm]
        ok = rows_aoi >= 0
        pv = pupil[ev.row_pos[ok]]
        fin = np.isfinite(pv)
        key = ev.rows_g[ok] * K + rows_aoi[ok]
        ps = np.bincount(key, weights=np.where(fin, pv, 0.0), minlength=S * K)
        pc = np.bincount(key, weights=fin.astype(float), minlength=S * K)
        with np.errstate(invalid="ignore", divide="ignore"):
            pup = np.where(pc > 0, ps / pc, np.nan).reshape(S, K) - baseline
        rows = np.flatnonzero(grp >= 0)
        gg = grp[rows]
        pr_ = pupil[rows]
        fr_ = np.isfinite(pr_)
        pcs = np.bincount(gg, weights=fr_.astype(float), minlength=S)
        with np.errstate(invalid="ignore", divide="ignore"):
            slide_pup = np.where(pcs > 0, np.bincount(gg, weights=np.where(fr_, pr_, 0.0), minlength=S) / pcs, np.nan) - baseline
        sub = df[df.recording == name]
        # align the slides of this recording to the cache rows by (question, repetition)
        order = [int(np.flatnonzero((r.slide_q == q) & (r.slide_rep == rep))[0]) for q, rep in zip(sub.question, sub.repetition)]
        ST.append(st[order]); PUP.append(pup[order]); SP.append(slide_pup[order]); TP.append(r.slide_target[order])
    return df, np.concatenate(ST), np.concatenate(PUP), np.concatenate(SP), np.concatenate(TP)


def celniak_features(st, pup, slide_pup, probe, pseudo, variant):
    """Same formulas as celniak_fast.slide_features_fast, for target position `pseudo` (== probe for the real features).
    leave_out: reference = median of the items other than the target and the real probe; in_reference: median of the
    non-probe items (includes the pseudo-target itself)."""
    S, K = probe.shape[0], st.shape[1]
    ar = np.arange(S)
    mask = np.ones((S, K), bool)
    mask[ar, probe] = False
    if variant == "leave_out" and not np.array_equal(pseudo, probe):
        mask[ar, pseudo] = False
    others = np.where(mask[:, :, None], st, np.nan)
    ntm = np.nanmedian(others, axis=1)
    tgt = st[ar, pseudo]
    tot = tgt + ntm
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(tot > 0, tgt / tot, 0.5)
    tp = pup[ar, pseudo]
    pupil = np.where(~np.isnan(tp), tp, slide_pup)
    return np.column_stack([norm, pupil])


def celniak_surrogates(parts, X_real, y, g):
    df, st, pup, sp, probe = parts
    X_chk = celniak_features(st, pup, sp, probe, probe, "leave_out")
    diff = float(np.nanmax(np.abs(X_chk - X_real)))
    log(f"celniak real features rebuilt, max abs diff vs stored task matrix = {diff:.3g}")
    assert diff < 1e-9
    n, K = len(probe), st.shape[1]
    irr = np.array([[j for j in range(K) if j != p] for p in probe])          # (n, 3) irrelevant positions per slide
    Xs = {v: [] for v in VARIANTS}
    for d in range(D):
        rng = np.random.default_rng([SEED_SURR, 1, d])
        pseudo = irr[np.arange(n), rng.integers(0, irr.shape[1], n)]
        for v in VARIANTS:
            Xs[v].append(celniak_features(st, pup, sp, probe, pseudo, v))
    return {v: np.stack(Xs[v]) for v in VARIANTS}, {"rebuild_max_abs_diff": diff, "n_irrelevant_per_slide": int(irr.shape[1]),
                                                       "probe_set_size": 1}


# ============================================================================= Mazza surrogate features
def mazza_surrogates(task):
    from edd import mazza as mz
    d = mz.load(ROOT / "data" / "mazza" / "data.csv")
    raw = mz.raw_columns(d)
    fz = mz.Featurizer(raw_cols=raw, sets=("z",)).fit(d[raw].to_numpy(float))
    m = d.group.isin((1, 2, 3, 4)).to_numpy()
    Rw = d[raw].to_numpy(float)[m]
    assert (d.unit.astype(str).to_numpy()[m] == task["g"]).all()
    B = fz._blocks(Rw)
    eps = {"cnt": 1.0, "dur": 100.0}

    def z_feats(slot_choice, variant):
        """slot_choice None = real targets. Else (n,6) index (0..2) of the neutral image standing in for each target slot."""
        cols = []
        n = Rw.shape[0]
        for k in ("cnt", "dur"):
            T, N = B[k]                                                    # (n,6,3), (n,3,3)
            if slot_choice is None:
                mu = N.mean(1)
                sd = np.sqrt(((N - mu[:, None, :]) ** 2).sum((1, 2)) / (3 * 2))
                v = (T - mu[:, None, :]) / (sd + eps[k])[:, None, None]
            else:
                v = np.zeros_like(T)
                for t in range(6):
                    j = slot_choice[:, t]
                    Tp = N[np.arange(n), j, :]                              # (n,3) pseudo target image
                    if variant == "leave_out":
                        keep = np.ones((n, 3), bool)
                        keep[np.arange(n), j] = False                       # the other two neutrals
                        Nr = np.stack([N[np.arange(n), i, :] for i in range(3)], 1)   # (n,3,3)
                        Rf = np.where(keep[:, :, None], Nr, np.nan)
                        mu = np.nanmean(Rf, axis=1)                        # (n,3)
                        resid = np.where(keep[:, :, None], Nr - mu[:, None, :], 0.0)
                        sd = np.sqrt((resid ** 2).sum((1, 2)) / (3 * (2 - 1)))
                    else:
                        mu = N.mean(1)
                        sd = np.sqrt(((N - mu[:, None, :]) ** 2).sum((1, 2)) / (3 * 2))
                    v[:, t, :] = (Tp - mu) / (sd + eps[k])[:, None]
            cols.append(v.reshape(n, -1))
        return np.column_stack(cols)

    X_chk = z_feats(None, "leave_out")
    diff = float(np.nanmax(np.abs(X_chk - task["X"])))
    log(f"mazza real z features rebuilt, max abs diff vs stored task matrix = {diff:.3g}")
    assert diff < 1e-9
    n = Rw.shape[0]
    Xs = {v: [] for v in VARIANTS}
    for dd in range(D):
        rng = np.random.default_rng([SEED_SURR, 2, dd])
        ch = rng.integers(0, 3, (n, 6))
        for v in VARIANTS:
            Xs[v].append(z_feats(ch, v))
    return {v: np.stack(Xs[v]) for v in VARIANTS}, {"rebuild_max_abs_diff": diff, "n_irrelevant_images": 3, "probe_set_size": 6,
                                                       "note": "6 target slots, only 3 neutral images: each slot is filled by one neutral image drawn with replacement"}


# ============================================================================= scoring over the stored folds
def run_dataset(name, task, Xs, refit_repeats):
    """Returns long table: one row per (repeat, fold, unit) with the real statistic, nominal tau, and per-draw surrogate
    statistics and surrogate-calibrated taus (lists over D, per variant), plus reproduction diagnostics."""
    X, y, g = task["X"], np.asarray(task["y"]).astype(int), np.asarray(task["g"]).astype(str)
    rows, diag = [], {"max_abs_score_diff": 0.0, "max_abs_inner_oof_diff": 0.0, "n_tau_mismatch": 0, "n_stat_mismatch": 0}
    for r in range(len(task["folds"])):
        ck = pickle.load(open(UNI / "checkpoints" / f"{name}__f100__r{r}.pkl", "rb"))
        for f, d in enumerate(ck):
            tr, te = d["tr"], d["te"]
            P = U.prep_fit(X[tr])
            coef, icpt, C = d["coef"], d["intercept"], d["C"]
            sc = U.prep_apply(P, X[te]) @ coef + icpt
            diag["max_abs_score_diff"] = max(diag["max_abs_score_diff"], float(np.max(np.abs(sc - d["score"]))))
            # real thresholds exactly as persons_mean
            ut, yt, st_tr = U._stats_mean(g[tr], y[tr], d["inner_oof"])
            th = U.thresholds(st_tr[yt == 1], st_tr[yt == 0])
            ue, ye, se = U._stats_mean(g[te], y[te], sc)
            # surrogate statistics of the test participants (rescoring only)
            surr_te = {}
            for v in VARIANTS:
                S = np.stack([(U.prep_apply(P, Xs[v][i][te]) @ coef + icpt) for i in range(D)])      # (D, n_te)
                tmp = pd.DataFrame(S.T, index=g[te]).groupby(level=0).mean()
                surr_te[v] = tmp.loc[ue].to_numpy().T                                                   # (D, n_units)
            tau_g = {v: np.full(D, np.nan) for v in VARIANTS}
            tau_i = {v: np.full(D, np.nan) for v in VARIANTS}
            if r in refit_repeats:
                with threadpool_limits(1):
                    seed = 1000 * r + f
                    ug = np.unique(g[tr])
                    k = U.inner_k(len(ug))
                    cv = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
                    oof_real = np.full(len(tr), np.nan)
                    oof_s = {v: np.full((D, len(tr)), np.nan) for v in VARIANTS}
                    for a, b in cv.split(X[tr], y[tr], g[tr]):
                        Pa = U.prep_fit(X[tr][a])
                        m = U.fit_lr(U.prep_apply(Pa, X[tr][a]), y[tr][a], C)
                        oof_real[b] = m.decision_function(U.prep_apply(Pa, X[tr][b]))
                        for v in VARIANTS:
                            for i in range(D):
                                oof_s[v][i, b] = m.decision_function(U.prep_apply(Pa, Xs[v][i][tr][b]))
                diag["max_abs_inner_oof_diff"] = max(diag["max_abs_inner_oof_diff"], float(np.max(np.abs(oof_real - d["inner_oof"]))))
                for v in VARIANTS:
                    for i in range(D):
                        _, ytr, ss = U._stats_mean(g[tr], y[tr], oof_s[v][i])
                        tau_g[v][i] = U.thresholds(st_tr[yt == 1], ss[ytr == 1])["tau_fpr7"]     # surrogates from training guilty
                        tau_i[v][i] = U.thresholds(st_tr[yt == 1], ss[ytr == 0])["tau_fpr7"]     # surrogates from training innocents
            for j, u in enumerate(ue):
                rows.append({"repeat": r, "fold": f, "unit": u, "y": int(ye[j]), "stat": float(se[j]), "tau_nom": th["tau_fpr7"],
                             "n_train_pos": int((yt == 1).sum()), "n_train_neg": int((yt == 0).sum()),
                             "surr": {v: surr_te[v][:, j] for v in VARIANTS}, "tau_g": tau_g, "tau_i": tau_i})
        log(f"{name}: repeat {r} done")
    return rows, diag


# ============================================================================= summaries
def unit_means(vals, units):
    return pd.Series(vals).groupby(np.asarray(units)).mean()


def summarise(name, rows, diag, stored_persons, refit_repeats, rng):
    res = {"reproduction_diag": diag}
    # ---- (a) real innocents at the nominal threshold, all repeats, from my rescoring and from the stored persons table
    R = pd.DataFrame([{k: v for k, v in x.items() if k in ("repeat", "fold", "unit", "y", "stat", "tau_nom")} for x in rows])
    R["dec"] = (R.stat > R.tau_nom).astype(float)
    sp = stored_persons.copy()
    sp["unit"] = sp.unit.astype(str)
    sp["dec"] = (sp.stat > sp.tau_fpr7).astype(float)
    mm = R.merge(sp[["repeat", "fold", "unit", "stat", "tau_fpr7"]], on=["repeat", "fold", "unit"], suffixes=("", "_st"))
    res["reproduction_diag"]["max_abs_person_stat_diff_vs_stored"] = float(np.max(np.abs(mm.stat - mm.stat_st)))
    res["reproduction_diag"]["max_abs_tau_diff_vs_stored"] = float(np.max(np.abs(mm.tau_nom - mm.tau_fpr7)))

    def unit_tab(df, y):
        return df[df.y == y].groupby("unit").dec.mean()

    def block(df):
        pos, neg = unit_tab(df, 1), unit_tab(df, 0)
        kp, kn = int(round(pos.mean() * len(pos))), int(round(neg.mean() * len(neg)))
        return {"detection": float(pos.mean()), "detection_cp95": cp_ci(kp, len(pos)), "n_pos_units": len(pos),
                "fpr": float(neg.mean()), "fpr_cp95": cp_ci(kn, len(neg)), "n_neg_units": len(neg)}

    res["a_real_nominal_all_repeats"] = block(R)
    res["a_real_nominal_stored_table_all_repeats"] = block(sp)
    R3 = R[R.repeat.isin(refit_repeats)]
    res["a_real_nominal_repeats_0_2"] = block(R3)
    return res, R


def surrogate_summaries(rows, refit_repeats, rng, label_pos="guilty"):
    out = {}
    nrep_all = sorted({x["repeat"] for x in rows})
    for scope, reps in (("all_repeats", nrep_all), ("repeats_0_2", [r for r in nrep_all if r in refit_repeats])):
        sub = [x for x in rows if x["repeat"] in reps]
        units = np.array([x["unit"] for x in sub])
        yy = np.array([x["y"] for x in sub])
        stat = np.array([x["stat"] for x in sub])
        tau = np.array([x["tau_nom"] for x in sub])
        real = (stat > tau).astype(float)
        uy = pd.Series(yy).groupby(units).first()
        gu = uy.index[uy == 1].to_numpy()
        iu = uy.index[uy == 0].to_numpy()
        R_unit = unit_means(real, units)                                         # real decision per unit (mean over repeats)
        sc = {"n_guilty_units": len(gu), "n_innocent_units": len(iu)}
        for v in VARIANTS:
            S = np.stack([x["surr"][v] for x in sub])                           # (rows, D)
            sdec = (S > tau[:, None]).astype(float)                              # nominal threshold
            # per-draw FPR: unit mean over repeats, then mean over units
            per_draw = {}
            for lab, ids in (("guilty", gu), ("innocent", iu)):
                arr = np.array([unit_means(sdec[:, d], units).loc[ids].mean() for d in range(D)])
                per_draw[lab] = arr
            U_unit = {lab: pd.DataFrame(sdec).groupby(units).mean().mean(1).loc[ids] for lab, ids in (("guilty", gu), ("innocent", iu))}
            blk = {"nominal": {}}
            Rin, Rgu = R_unit.loc[iu].to_numpy(), R_unit.loc[gu].to_numpy()
            Sg, Si = U_unit["guilty"].to_numpy(), U_unit["innocent"].to_numpy()
            for lab, arr, U_ in (("surrogate_from_guilty", per_draw["guilty"], Sg), ("surrogate_from_innocent", per_draw["innocent"], Si)):
                kk = int(round(U_.mean() * len(U_)))
                blk["nominal"][lab] = {"fpr_mean_over_draws": float(U_.mean()), "fpr_sd_over_draws": float(arr.std(ddof=1)),
                                       "fpr_min_draw": float(arr.min()), "fpr_max_draw": float(arr.max()), "n_units": len(U_),
                                       "cp95_approx": cp_ci(kk, len(U_))}
            # surrogate-calibrated thresholds
            cal = {}
            for lab, key in (("calibrated_on_training_guilty_surrogates", "tau_g"), ("calibrated_on_training_innocent_surrogates", "tau_i")):
                T = np.stack([x[key][v] for x in sub])                           # (rows, D)
                if not np.isfinite(T).all():
                    cal[lab] = None
                    continue
                dec_real = (stat[:, None] > T).astype(float)                      # real test statistic vs surrogate-calibrated tau
                dec_s = (np.stack([x["surr"][v] for x in sub]) > T).astype(float)  # surrogate test cases vs same tau
                df_r = pd.DataFrame(dec_real).groupby(units).mean()
                df_s = pd.DataFrame(dec_s).groupby(units).mean()
                Cin = df_r.loc[iu].mean(1).to_numpy()
                Cgu = df_r.loc[gu].mean(1).to_numpy()
                Csg = df_s.loc[gu].mean(1).to_numpy()
                per_draw_in = df_r.loc[iu].mean(0).to_numpy()
                per_draw_gu = df_r.loc[gu].mean(0).to_numpy()
                tau_ref = float(np.mean([x["tau_nom"] for x in sub]))
                cal[lab] = {"real_innocent_fpr": float(Cin.mean()), "real_innocent_fpr_sd_over_draws": float(per_draw_in.std(ddof=1)),
                            "real_innocent_fpr_min_draw": float(per_draw_in.min()), "real_innocent_fpr_max_draw": float(per_draw_in.max()),
                            "real_innocent_fpr_cp95": cp_ci(int(round(Cin.mean() * len(Cin))), len(Cin)),
                            "real_guilty_detection": float(Cgu.mean()), "real_guilty_detection_sd_over_draws": float(per_draw_gu.std(ddof=1)),
                            "real_guilty_detection_cp95": cp_ci(int(round(Cgu.mean() * len(Cgu))), len(Cgu)),
                            "surrogate_guilty_fpr_at_same_threshold": float(Csg.mean()),
                            "mean_tau_surrogate_calibrated": float(np.nanmean(T)), "mean_tau_nominal": tau_ref,
                            "_Cin": Cin, "_Csg": Csg, "_Cgu": Cgu}
            blk["surrogate_calibrated"] = cal
            blk["real_innocent_fpr_nominal"] = float(Rin.mean())
            blk["real_innocent_fpr_nominal_cp95"] = cp_ci(int(round(Rin.mean() * len(Rin))), len(Rin))
            blk["real_guilty_detection_nominal"] = float(Rgu.mean())
            # ---- bootstrap over participants, stored scores, no refits
            b_rng = np.random.default_rng(SEED_BOOT)
            Bd = {k: [] for k in ("real", "sg", "si", "d_g", "d_i", "cal_g", "cal_i", "d_calg_real", "d_cali_real", "calsg")}
            Cg = cal.get("calibrated_on_training_guilty_surrogates")
            Ci = cal.get("calibrated_on_training_innocent_surrogates")
            for _ in range(NBOOT):
                ii = b_rng.integers(0, len(iu), len(iu))
                gg = b_rng.integers(0, len(gu), len(gu))
                r_ = Rin[ii].mean()
                sg_ = Sg[gg].mean()
                si_ = Si[ii].mean()
                Bd["real"].append(r_); Bd["sg"].append(sg_); Bd["si"].append(si_)
                Bd["d_g"].append(sg_ - r_); Bd["d_i"].append(si_ - r_)
                if Cg:
                    Bd["cal_g"].append(Cg["_Cin"][ii].mean()); Bd["d_calg_real"].append(Cg["_Cin"][ii].mean() - r_)
                    Bd["calsg"].append(Cg["_Csg"][gg].mean())
                if Ci:
                    Bd["cal_i"].append(Ci["_Cin"][ii].mean()); Bd["d_cali_real"].append(Ci["_Cin"][ii].mean() - r_)

            def ci(a):
                return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else None
            blk["differences_nominal"] = {
                "surrogate_from_guilty_minus_real_innocent": {"diff": float(Sg.mean() - Rin.mean()), "boot95": ci(Bd["d_g"]), "design": "unpaired (different people)"},
                "surrogate_from_innocent_minus_real_innocent": {"diff": float(Si.mean() - Rin.mean()), "boot95": ci(Bd["d_i"]), "design": "paired (same innocent people)"},
                "real_innocent_fpr_boot95": ci(Bd["real"]), "surrogate_from_guilty_fpr_boot95": ci(Bd["sg"]), "surrogate_from_innocent_fpr_boot95": ci(Bd["si"])}
            if Cg:
                blk["differences_surrogate_calibrated"] = {
                    "guilty_surrogate_calibrated_real_innocent_fpr_boot95": ci(Bd["cal_g"]),
                    "minus_real_innocent_fpr_at_nominal_threshold": {"diff": float(Cg["_Cin"].mean() - Rin.mean()), "boot95": ci(Bd["d_calg_real"])},
                    "minus_7pct_target": {"diff": float(Cg["_Cin"].mean() - 0.07), "boot95": ci(np.array(Bd["cal_g"]) - 0.07)},
                    "surrogate_guilty_fpr_at_same_threshold_boot95": ci(Bd["calsg"])}
            if Ci:
                blk["differences_surrogate_calibrated_innocent_surrogates"] = {
                    "real_innocent_fpr_boot95": ci(Bd["cal_i"]),
                    "minus_real_innocent_fpr_at_nominal_threshold": {"diff": float(Ci["_Cin"].mean() - Rin.mean()), "boot95": ci(Bd["d_cali_real"])}}
            for c in cal.values():
                if c:
                    for k_ in ("_Cin", "_Csg", "_Cgu"):
                        c.pop(k_, None)
            sc[v] = blk
        out[scope] = sc
    return out


def main():
    repro_only = "--repro-only" in sys.argv
    final = {"meta": {"status": "exploratory, post-review; not pre-registered", "D": D, "n_boot": NBOOT, "seed_surrogate": SEED_SURR,
                      "seed_boot": SEED_BOOT, "variants": {"leave_out": "primary: the pseudo-probe item(s) (and the real probe) are removed from the reference set",
                                                           "in_reference": "sensitivity: the pseudo-probe stays inside the reference set"},
                      "refit_repeats": "inner-fold LR fits at the stored C on repeats 0-2 only (frozen folds)"}}
    rng = np.random.default_rng(SEED_BOOT)
    # ------------------------------------------------------------------ Celniak
    log("building tasks")
    ct = U.build_celniak(UNI / "cache" / "celniak_fast_slides.csv", log)["celniak_unified"]
    mt = U.build_mazza(log)["mazza_guilty_vs_innocent"]
    if repro_only:
        for nm, t in (("celniak_unified", ct), ("mazza_guilty_vs_innocent", mt)):
            sp = pd.read_csv(UNI / "oof" / f"{nm}_persons.csv.gz")
            sp["dec"] = (sp.stat > sp.tau_fpr7).astype(float)
            pos = sp[sp.y == 1].groupby("unit").dec.mean(); neg = sp[sp.y == 0].groupby("unit").dec.mean()
            kn = int(round(neg.mean() * len(neg)))
            print(nm, "detection", round(pos.mean() * 100, 2), "FPR", round(neg.mean() * 100, 2), [round(x * 100, 1) for x in cp_ci(kn, len(neg))], "n", len(pos), len(neg))
        return
    for nm, task, build in (("celniak_unified", ct, "celniak"), ("mazza_guilty_vs_innocent", mt, "mazza")):
        log("=====", nm)
        if build == "celniak":
            parts = celniak_parts()
            assert (parts[0].recording.astype(str).to_numpy() == task["g"]).all()
            Xs, info = celniak_surrogates(parts, task["X"], task["y"], task["g"])
        else:
            Xs, info = mazza_surrogates(task)
        rows, diag = run_dataset(nm, task, Xs, set(range(MAX_REFIT_REPEATS)))
        sp = pd.read_csv(UNI / "oof" / f"{nm}_persons.csv.gz")
        res, R = summarise(nm, rows, diag, sp, set(range(MAX_REFIT_REPEATS)), rng)
        res["surrogate_construction"] = info
        res["surrogate"] = surrogate_summaries(rows, set(range(MAX_REFIT_REPEATS)), rng)
        final[nm] = res
        pd.DataFrame([{k: v for k, v in x.items() if k in ("repeat", "fold", "unit", "y", "stat", "tau_nom", "n_train_pos", "n_train_neg")} |
                      {"surr_mean_leave_out": float(np.mean(x["surr"]["leave_out"])), "tau_g_mean_leave_out": float(np.nanmean(x["tau_g"]["leave_out"]))}
                      for x in rows]).to_csv(OUT / f"scores_{nm}.csv.gz", index=False)
        log(nm, "diag", json.dumps(diag))
        (OUT / "results.json").write_text(json.dumps(final, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    log("done")


if __name__ == "__main__":
    main()
