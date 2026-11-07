"""Phase 4 unified pipeline: one fixed recipe applied identically to every dataset (see
results/phase4/unified/PREREG.md, which is the authority for every choice made here).

Lightweight, label-free eye-tracking concealed-information detection:
  1 input at the lowest supported rate (30 Hz where raw samples exist), 2 label-free per-item / per-trial features
  (existing, vectorised extraction code), 3 within-participant standardisation (label-free), 4 L2 logistic regression
  with balanced class weights and C tuned in participant-grouped inner folds, 5 vectorised extraction, 6 person-level
  decision at thresholds calibrated on training participants only.

Nothing here modifies an existing file. The tracks' frozen folds and stored baseline scores are only read.
"""
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# ----------------------------------------------------------------------------- the fixed recipe
C_GRID = tuple(float(c) for c in np.logspace(-3, 1, 9))      # 0.001 ... 10, half-decade steps, same for every dataset
INNER_MAX_K = 5                                               # inner participant-grouped folds (fewer for tiny training sets)
TARGET_FPR = 0.07                                             # false-positive target of the second operating point
LEARNING_FRACS = (0.25, 0.5, 0.75)                            # (1.0 is the main run)
BOOT_SEED = 20261103


def inner_k(n_units):
    return int(min(INNER_MAX_K, max(2, n_units // 3)))


def fast_auc(y, s):
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(y, s):
    """Unweighted log-loss of sigmoid(score)."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    return float(np.mean(np.logaddexp(0.0, s) - y * s))


# ----------------------------------------------------------------------------- model primitives
def prep_fit(X):
    """Median imputation and column standardisation, fitted on the given (training) rows only."""
    X = np.asarray(X, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isnan(X), med, X)
    mu, sd = Xi.mean(0), Xi.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return med, mu, sd


def prep_apply(P, X):
    med, mu, sd = P
    X = np.asarray(X, float)
    return (np.where(np.isnan(X), med, X) - mu) / sd


def fit_lr(X, y, C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)


class DeployLR:
    """Everything the deployed model stores: imputation medians, means, scales, coefficients, intercept."""

    def __init__(self, P, coef, intercept, names=None, C=None):
        self.med, self.mu, self.sd = P
        self.coef_ = np.asarray(coef, float)
        self.intercept_ = float(intercept)
        self.names = list(names) if names is not None else None
        self.C = C

    def decision_function(self, X):
        return prep_apply((self.med, self.mu, self.sd), X) @ self.coef_ + self.intercept_

    @property
    def n_parameters(self):
        return int(self.coef_.size + 1 + 3 * self.coef_.size)      # coef + intercept + median, mean, scale per feature


class Fit:
    pass


def tune_fit(X, y, g, seed, grid=C_GRID):
    """C chosen by mean per-fold AUC over participant-grouped inner folds (ties go to the smaller C);
    refit on all rows. Also returns the inner out-of-fold scores at the chosen C (for threshold calibration)."""
    ug = np.unique(g)
    k = inner_k(len(ug))
    cv = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    n = len(y)
    oof = np.full((n, len(grid)), np.nan)
    aucs = np.full((k, len(grid)), np.nan)
    for f, (a, b) in enumerate(cv.split(X, y, g)):
        P = prep_fit(X[a])
        Xa, Xb = prep_apply(P, X[a]), prep_apply(P, X[b])
        two = len(np.unique(y[b])) == 2
        for j, C in enumerate(grid):
            s = fit_lr(Xa, y[a], C).decision_function(Xb)
            oof[b, j] = s
            if two:
                aucs[f, j] = fast_auc(y[b], s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cv_auc = np.nanmean(aucs, axis=0)
    j = int(np.nanargmax(cv_auc)) if np.isfinite(cv_auc).any() else len(grid) // 2
    P = prep_fit(X)
    m = fit_lr(prep_apply(P, X), y, grid[j])
    r = Fit()
    r.C, r.j, r.P, r.coef, r.intercept = grid[j], j, P, m.coef_[0], float(m.intercept_[0])
    r.oof, r.cv_auc = oof[:, j], cv_auc
    return r


def decide(fit, X):
    return prep_apply(fit.P, X) @ fit.coef + fit.intercept


# ----------------------------------------------------------------------------- outer CV engine
def _subsample_units(units, y, frac, rng):
    """Random subset of the training participants (stratified by class when each participant has one label)."""
    ug = np.unique(units)
    lab = {u: np.unique(y[units == u]) for u in ug}
    pure = all(len(v) == 1 for v in lab.values())
    if pure:
        keep = []
        for c in (0, 1):
            uc = np.array([u for u in ug if lab[u][0] == c])
            if len(uc):
                keep += list(rng.choice(uc, max(2, int(round(frac * len(uc)))), replace=False))
        return np.array(keep)
    return rng.choice(ug, max(4, int(round(frac * len(ug)))), replace=False)


def _fold(X, y, g, test_units, seed, frac, rseed, val):
    with threadpool_limits(1):
        warnings.filterwarnings("ignore")
        te_mask = np.isin(g, test_units)
        tr = np.flatnonzero(~te_mask)
        te = np.flatnonzero(te_mask)
        if frac < 1.0:
            keep = _subsample_units(g[tr], y[tr], frac, np.random.default_rng(rseed))
            tr = tr[np.isin(g[tr], keep)]
        fit = tune_fit(X[tr], y[tr], g[tr], seed)
        out = {"te": te, "tr": tr, "score": decide(fit, X[te]), "C": fit.C, "coef": fit.coef, "intercept": fit.intercept,
               "inner_oof": fit.oof, "cv_auc": fit.cv_auc, "n_train_units": int(len(np.unique(g[tr])))}
        if val:                                       # validation curve: every C fitted on the whole outer-training set
            Xtr, Xte = prep_apply(fit.P, X[tr]), prep_apply(fit.P, X[te])
            te_s = np.empty((len(te), len(C_GRID)))
            tr_ll, tr_auc = [], []
            for j, C in enumerate(C_GRID):
                m = fit_lr(Xtr, y[tr], C)
                te_s[:, j] = m.decision_function(Xte)
                s_tr = m.decision_function(Xtr)
                tr_ll.append(logloss(y[tr], s_tr))
                tr_auc.append(fast_auc(y[tr], s_tr))
            out.update({"val_te": te_s, "val_tr_ll": np.array(tr_ll), "val_tr_auc": np.array(tr_auc)})
        return out


def run_task(name, X, y, g, folds_per_repeat, ck_dir, n_jobs=2, frac=1.0, val=True, repeats=None, log=print):
    """Nested-CV one task on its frozen folds. One checkpoint per repeat (skipped when present)."""
    ck_dir = Path(ck_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    g = np.asarray(g).astype(str)
    y = np.asarray(y).astype(int)
    out = {}
    for r, folds in enumerate(folds_per_repeat):
        if repeats is not None and r not in repeats:
            continue
        ck = ck_dir / f"{name}__f{int(round(frac * 100))}__r{r}.pkl"
        if ck.exists():
            out[r] = pickle.load(open(ck, "rb"))
            continue
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(
            delayed(_fold)(X, y, g, [str(u) for u in tu], 1000 * r + f, frac, 7000 + 1000 * r + f, val and frac == 1.0)
            for f, tu in enumerate(folds))
        tmp = ck.with_suffix(".tmp")
        pickle.dump(res, open(tmp, "wb"))
        tmp.replace(ck)
        out[r] = res
        log(f"  [{name} f{frac:.2f}] repeat {r}: {len(res)} folds in {time.time() - t0:.0f}s")
    return out


def oof_rows(res_by_repeat, meta):
    """Long table of out-of-fold row scores: repeat, fold, row, unit, y + the task's meta columns."""
    parts = []
    for r, folds in res_by_repeat.items():
        for f, d in enumerate(folds):
            df = meta.iloc[d["te"]].copy()
            df.insert(0, "row", d["te"])
            df.insert(0, "fold", f)
            df.insert(0, "repeat", r)
            df["score"] = d["score"]
            df["C"] = d["C"]
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- person-level decisions
def thresholds(pos, neg, target=TARGET_FPR):
    """Calibration on training participants: tau_fpr7 = the floor(target*m)-th largest negative statistic
    (decide 'positive' when stat > tau); tau_eer = the cut where training FPR and FNR are closest."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return {"tau_eer": float("nan"), "tau_fpr7": float("nan")}
    z = np.sort(neg)[::-1]
    k = int(np.floor(target * len(z)))
    tau7 = float(z[k]) if len(z) > k else float("inf")
    allv = np.unique(np.concatenate([pos, neg]))
    cand = np.concatenate([[allv[0] - 1.0], (allv[:-1] + allv[1:]) / 2, [allv[-1] + 1.0]])
    fpr = (neg[None, :] > cand[:, None]).mean(1)
    fnr = (pos[None, :] <= cand[:, None]).mean(1)
    i = int(np.argmin(np.abs(fpr - fnr)))
    return {"tau_eer": float(cand[i]), "tau_fpr7": tau7}


def _stats_mean(units, y, score):
    d = pd.DataFrame({"u": units, "y": y, "s": score}).groupby("u").agg(y=("y", "first"), s=("s", "mean"))
    return d.index.to_numpy(), d.y.to_numpy(), d.s.to_numpy()


def _cit_z(s, y, rng):
    """Delmas CIT statistic (as the Phase 2 track): z of famous vs irrelevant scores; simulated innocent = the same
    person with |famous| random unknown faces as pseudo-probes."""
    sf, su = s[y == 1], s[y == 0]
    zk = (sf.mean() - su.mean()) / (su.std() + 1e-9)
    pick = rng.choice(len(su), len(sf), replace=False)
    rest = np.delete(su, pick)
    zi = (su[pick].mean() - rest.mean()) / (rest.std() + 1e-9)
    return float(zk), float(zi)


def persons_mean(task, res_by_repeat):
    """Person statistic = mean row score; thresholds from the training participants' inner out-of-fold means."""
    g, y = task["g"], task["y"]
    rows = []
    for r, folds in res_by_repeat.items():
        for f, d in enumerate(folds):
            ut, yt, st = _stats_mean(g[d["tr"]], y[d["tr"]], d["inner_oof"])
            th = thresholds(st[yt == 1], st[yt == 0])
            ue, ye, se = _stats_mean(g[d["te"]], y[d["te"]], d["score"])
            for u, yy, s in zip(ue, ye, se):
                rows.append({"repeat": r, "fold": f, "unit": u, "kind": "person", "y": int(yy), "stat": float(s), **th,
                             "n_train_pos": int((yt == 1).sum()), "n_train_neg": int((yt == 0).sum())})
    return pd.DataFrame(rows)


def persons_cit(task, res_by_repeat):
    """Delmas person-level CIT decision (positive = known-face statistic, negative = simulated innocent)."""
    g, y = task["g"], task["y"]
    rows = []
    for r, folds in res_by_repeat.items():
        for f, d in enumerate(folds):
            pos, neg = [], []
            tr_g = g[d["tr"]]
            for u in np.unique(tr_g):
                m = tr_g == u
                zk, zi = _cit_z(d["inner_oof"][m], y[d["tr"]][m], np.random.default_rng(500 + int(u)))
                pos.append(zk)
                neg.append(zi)
            th = thresholds(pos, neg)
            te_g = g[d["te"]]
            for u in np.unique(te_g):
                m = te_g == u
                zk, zi = _cit_z(d["score"][m], y[d["te"]][m], np.random.default_rng(r + int(u)))
                for kind, yy, z in (("known", 1, zk), ("sim_innocent", 0, zi)):
                    rows.append({"repeat": r, "fold": f, "unit": u, "kind": kind, "y": yy, "stat": z, **th,
                                 "n_train_pos": len(pos), "n_train_neg": len(neg)})
    return pd.DataFrame(rows)


def persons_chen(task, res_by_repeat):
    """Chen person statistics exactly as chen_confirm (one-sided t of familiar vs controls converted to a z; negatives =
    control-1-vs-2 z in both directions); threshold per experiment from the training participants' inner out-of-fold
    scores. Also stores the training z statistics (needed by the cascade rule)."""
    from edd import chen_confirm as cc
    g, T1, exps = task["g"], task["T1"], task["exps"]
    rows = []
    for r, folds in res_by_repeat.items():
        for f, d in enumerate(folds):
            tr, te = d["tr"], d["te"]
            for ee in (1, 2):
                trm, tm = exps[tr] == ee, exps[te] == ee
                if not tm.any():
                    continue
                pos, null = [], []
                for u in np.unique(g[tr][trm]):
                    m = trm & (g[tr] == u)
                    z = cc.person_z(d["inner_oof"][m], T1[tr][m])
                    pos.append(z["z_fam"])
                    null += [z["z_c1"], z["z_c2"]]
                th = thresholds(pos, null)
                th["tau_fpr7"] = cc.threshold_for_fpr(np.array(null, float))     # identical to the confirmatory test
                ptr = np.array(pos, float)
                for u in np.unique(g[te][tm]):
                    m = tm & (g[te] == u)
                    z = cc.person_z(d["score"][m], T1[te][m])
                    for kind, yy, key in (("fam", 1, "z_fam"), ("c1", 0, "z_c1"), ("c2", 0, "z_c2")):
                        rows.append({"repeat": r, "fold": f, "unit": u, "exp": ee, "kind": kind, "y": yy, "stat": z[key],
                                     **th, "n_train_pos": int(np.isfinite(ptr).sum()), "n_train_neg": int(np.isfinite(null).sum()),
                                     "train_pos": json.dumps([float(x) for x in ptr]),
                                     "train_null": json.dumps([float(x) for x in null])})
    return pd.DataFrame(rows)


PERSON_FN = {"mean": persons_mean, "cit": persons_cit, "chen": persons_chen}


def person_summary(pdf):
    """Metrics of a person table: per repeat, then mean and SD over repeats (AUC, EER, metrics at the EER-calibrated
    threshold incl. HTER, detection and realised FPR at the FPR<=7% threshold)."""
    from edd.metrics import binary_metrics
    per = []
    for r, d in pdf.groupby("repeat"):
        y, s = d.y.to_numpy(), d.stat.to_numpy()
        ok = np.isfinite(s)
        y, s, d = y[ok], s[ok], d[ok]
        m = binary_metrics(y, s - d.tau_eer.to_numpy(), threshold=0.0)         # per-row tau (calibrated per fold)
        row = {f"eer_cal_{k}": v for k, v in m.items() if k not in ("n", "auc", "eer")}
        m0 = binary_metrics(y, s)                                              # threshold-free metrics on the raw statistic
        row["auc"], row["eer"] = m0["auc"], m0["eer"]
        dec = s > d.tau_fpr7.to_numpy()
        row["fpr7_detection"] = float(dec[y == 1].mean()) if (y == 1).any() else float("nan")
        row["fpr7_realised_fpr"] = float(dec[y == 0].mean()) if (y == 0).any() else float("nan")
        row["n"] = int(len(y))
        per.append(row)
    keys = [k for k in per[0] if k != "n"]
    return {k: {"mean": float(np.nanmean([p[k] for p in per])), "sd": float(np.nanstd([p[k] for p in per]))} for k in keys}


def row_summary(rows):
    """Row-level (trial / image / slide) metrics per repeat (threshold 0 on the logit), mean and SD over repeats."""
    from edd.metrics import binary_metrics, summarise_repeats
    return summarise_repeats([binary_metrics(d.y, d.score) for _, d in rows.groupby("repeat")])


# ----------------------------------------------------------------------------- statistics
def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def p_superior(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float((1 + np.sum(d <= 0)) / (1 + len(d)))


def p_inferior(d, margin):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float((1 + np.sum(d <= -margin)) / (1 + len(d)))


def auc_items(items):
    y = np.concatenate([i[0] for i in items])
    s = np.concatenate([i[1] for i in items])
    return fast_auc(y, s)


def paired_unit_bootstrap(ga, gb, metric, n_boot, seed):
    """Paired participant-cluster bootstrap of metric(b) - metric(a). ga, gb: {unit: (y, score)} with identical units."""
    units = sorted(set(ga) & set(gb))
    rng = np.random.default_rng(seed)
    A, B = [ga[u] for u in units], [gb[u] for u in units]
    est_a, est_b = metric(A), metric(B)
    d, ma, mb = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        a, b = metric([A[i] for i in pick]), metric([B[i] for i in pick])
        ma.append(a)
        mb.append(b)
        d.append(b - a)
    d = np.array(d)
    return {"est_base": est_a, "est_unified": est_b, "delta": est_b - est_a, "n_units": len(units),
            "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
            "ci_base": [float(np.nanpercentile(ma, 2.5)), float(np.nanpercentile(ma, 97.5))],
            "ci_unified": [float(np.nanpercentile(mb, 2.5)), float(np.nanpercentile(mb, 97.5))],
            "lower_one_sided_95": float(np.nanpercentile(d, 5)), "deltas": d}


def unit_groups(df, unit="unit", y="y", score="score"):
    """{unit: (y array, score array)} from a long table of (already repeat-averaged) scores."""
    return {u: (d[y].to_numpy(), d[score].to_numpy()) for u, d in df.groupby(unit)}


def avg_over_repeats(df, keys, score="score"):
    """Repeat-averaged score per decision unit (keys identify the unit; y must be constant within them)."""
    return df.groupby(keys, as_index=False).agg(y=("y", "first"), score=(score, "mean"))


# ============================================================================= dataset adapters (fixed feature lists)
DELMAS_FEATS = ["dwell_1", "dwell_2", "n_fix", "n_visit", "dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4", "dwell_b5",
                "first_fix", "lat_first", "pupil_face", "pupil_trial"]            # 13: msacc_rate needs >= 500 Hz
DELMAS_SETS = [("64", 1), ("64", 2), ("64", 3), ("32", 3), ("32", 4), ("32", 5)]
DELMAS_DECODED = RESULTS / "phase3" / "fast_extraction" / "cache" / "delmas_decoded"
CHEN_CONFIRM = RESULTS / "phase3" / "chen_confirm"
CELNIAK_CACHE = ROOT / "data" / "celniak_cache"
CELNIAK_FEATS = [f"norm_{k}_{s}" for k in ("fixation", "visit", "glance") for s in ("total", "mean", "min", "max", "count", "first")] + ["pupil_bc"]


def zscore_within(X, g):
    """Label-free within-participant standardisation: each column against that participant's own rows (ddof 1)."""
    X = np.asarray(X, float)
    Z = np.full_like(X, np.nan)
    g = np.asarray(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for u in np.unique(g):
            m = g == u
            mu = np.nanmean(X[m], 0)
            sd = np.nanstd(X[m], 0, ddof=1)
            Z[m] = (X[m] - mu) / (sd + 1e-9)
    return Z


# ----------------------------------------------------------------------------- Delmas (30 Hz, fast extraction)
def delmas_image_rows(pid, F, images):
    """Per-image means of the trial features (n_trials, 4, 14) in each set's trial window (set 64: all 64 trials, set 32: the
    last 32). No labels are touched."""
    from edd import delmas_fast as dfast
    out = []
    for s, t0 in (("64", 0), ("32", 32)):
        img = images[t0:].ravel()
        uniq, first = np.unique(img, return_index=True)
        ids = uniq[np.argsort(first)]
        A = dfast.aggregate_images(F[t0:], images[t0:])
        for i, a in zip(ids, A):
            out.append([pid, s, int(i), *a])
    return out


def delmas_extract(out_csv, log=print):
    """Decoded 1000 Hz sessions -> anti-aliased 30 Hz -> vectorised trial features -> per-image means in each set's trial window
    (set 64: all 64 trials, set 32: the last 32). No labels are touched. Cached to `out_csv`."""
    from edd import delmas_fast as dfast
    from edd.delmas_raw import TRIAL_FEATURES
    out_csv = Path(out_csv)
    if out_csv.exists():
        d = pd.read_csv(out_csv)
        d["set"] = d["set"].astype(str)
        return d
    rows = []
    files = sorted(DELMAS_DECODED.glob("p*.npz"))
    if not files:
        raise FileNotFoundError(f"{DELMAS_DECODED} missing: run the phase-3 decode stage first")
    for fpath in files:
        pid = int(fpath.stem[1:])
        z = np.load(fpath)
        X, Y, P = dfast.degrade_batch(z["x"], z["y"], z["p"], 30)
        F = dfast.trial_features_batch(X, Y, P, 30, with_msacc=False)
        images = z["images"]
        rows += delmas_image_rows(pid, F, images)
    df = pd.DataFrame(rows, columns=["participant", "set", "image"] + TRIAL_FEATURES)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"delmas 30 Hz features extracted: {len(df)} rows")
    return df


def build_delmas(cache_csv, log=print):
    return delmas_tasks_from_features(delmas_extract(cache_csv, log))


def delmas_tasks_from_features(feat):
    from edd.delmas import load_trials
    folds_all = json.loads((RESULTS / "phase2" / "delmas_raw" / "folds.json").read_text())["repeats"]
    books = {"64": load_trials(ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"),
             "32": load_trials(ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx")}
    tasks = {}
    for s, g in DELMAS_SETS:
        book = books[s]
        d = feat[feat.set == s].merge(book[["participant", "group", "face", "image"]], on=["participant", "image"], how="inner")
        d = d.reset_index(drop=True)
        d[DELMAS_FEATS] = zscore_within(d[DELMAS_FEATS].to_numpy(), d.participant.to_numpy())
        d = d[d.group == g].reset_index(drop=True)
        key = f"{s}_g{g}"
        folds = [[[str(p) for p in fold] for fold in rep] for rep in folds_all[key]]
        tasks[f"delmas_{key}"] = {
            "name": f"delmas_{key}", "ds": "delmas", "X": d[DELMAS_FEATS].to_numpy(float), "y": d.face.to_numpy(int),
            "g": d.participant.astype(str).to_numpy(), "names": DELMAS_FEATS, "folds": folds, "person": "cit",
            "meta": pd.DataFrame({"unit": d.participant.astype(str), "y": d.face.astype(int), "image": d.image, "set": key}),
            "label_scope": "within_unit"}
    return tasks


# ----------------------------------------------------------------------------- Chen (30 Hz from the 100 Hz epochs)
CHEN_CHANNELS = ("pupil", "dx", "dy")


def decimate_to_rate(A, fs):
    """Epochs (n, 250) at 100 Hz -> sensor rate `fs` (Hz): gap interpolation, polyphase anti-alias resampling by fs/100
    (the Delmas track's `resample_poly`); fs = 100 returns the epochs unchanged (gaps stay NaN)."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    from edd.delmas_fast import fill_rows
    A = np.asarray(A, float)
    if fs == 100:
        return A
    fr = Fraction(fs, 100)
    few = np.isfinite(A).sum(1) < 2
    out = resample_poly(fill_rows(A)[0], fr.numerator, fr.denominator, axis=1, padtype="line")
    out[few] = np.nan
    return out


def decimate_to_30(A):
    return decimate_to_rate(A, 30)


def chen_features(ch, channels=CHEN_CHANNELS, fs=30):
    """25 bins of 100 ms over 0-2500 ms (fs/10 samples per bin) for pupil, horizontal gaze toward the motion direction (dx),
    vertical gaze (dy). The missing-sample channel and the derived pupil-rate channel are not used."""
    feats, names = [], []
    spb = fs // 10
    for c in channels:
        A = decimate_to_rate(ch[c], fs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            feats.append(np.nanmean(A.reshape(len(A), 25, spb), axis=2))
        names += [f"{c}_{100 * i}_{100 * (i + 1)}" for i in range(25)]
    return np.hstack(feats), names


def build_chen(log=print):
    from edd import chen_confirm as cc
    meta, ch = cc.build_pooled()
    Xraw, names = chen_features(ch)
    g = meta.unit.astype(int).astype(str).to_numpy()
    X = zscore_within(Xraw, g)
    T1 = meta.T1.to_numpy().astype(int)
    y = (T1 == cc.FAM).astype(int)
    folds = json.loads((CHEN_CONFIRM / "folds.json").read_text())["repeats"]
    names = np.array(names)
    # stage 1 of the cascade: pupil and horizontal gaze at 500-ms resolution (5 + 5 features), standardised within participant
    coarse, cn = [], []
    for c in ("pupil", "dx"):
        raw = Xraw[:, [i for i, n in enumerate(names) if n.startswith(c + "_")]]
        coarse.append(raw.reshape(len(raw), 5, 5).mean(2))
        cn += [f"{c}_{500 * i}_{500 * (i + 1)}" for i in range(5)]
    Xs1 = zscore_within(np.hstack(coarse), g)
    common = {"g": g, "y": y, "folds": folds, "person": "chen", "T1": T1, "exps": meta.exp.to_numpy().astype(int),
              "meta": pd.DataFrame({"unit": g, "y": y, "T1": T1, "exp": meta.exp.to_numpy().astype(int)}),
              "label_scope": "within_unit"}
    return {"chen_unified": {"name": "chen_unified", "ds": "chen", "X": X, "names": list(names), **common},
            "chen_stage1": {"name": "chen_stage1", "ds": "chen", "X": Xs1, "names": cn, **common}}, (meta, ch)


# ----------------------------------------------------------------------------- Celniak (fast extraction of the probe-normalised features)
def celniak_extract(out_csv, log=print):
    """Vectorised extraction (edd.celniak_fast) of the paper's 19 probe-vs-irrelevant slide features from the cached raw
    Tobii samples; one row per slide. Cached to `out_csv`."""
    from edd import celniak, celniak_fast as cf
    out_csv = Path(out_csv)
    if out_csv.exists():
        return pd.read_csv(out_csv)
    rows = []
    for f in sorted(CELNIAK_CACHE.glob("Recording*.csv.gz")):
        name = f.name[:-len(".csv.gz")]
        samples, meta = celniak.load_cached(CELNIAK_CACHE, name)
        rec = cf.arrays_from_frame(samples)
        out = cf.slide_features_fast(rec)
        d = pd.DataFrame({"recording": name, "question": out["question"], "repetition": out["repetition"],
                          "valid_frac": out["valid_frac"]})
        for c in CELNIAK_FEATS:
            d[c] = out[c]
        d["label"] = int(str(meta["group"]).strip().lower() == "guilty")
        d["first_exam"] = meta.get("FirstExamination")
        d["val_acc"] = meta.get("Average validation accuracy (degrees)", np.nan)
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"celniak fast features extracted: {len(df)} slides, {df.recording.nunique()} recordings")
    return df


def build_celniak(cache_csv, log=print):
    df = celniak_extract(cache_csv, log)
    folds = json.loads((RESULTS / "phase2" / "celniak" / "folds.json").read_text())["repeats"]
    g = df.recording.astype(str).to_numpy()
    y = df.label.to_numpy(int)
    base = {"g": g, "y": y, "folds": folds, "person": "mean", "label_scope": "unit",
            "meta": pd.DataFrame({"unit": g, "y": y, "question": df.question, "repetition": df.repetition})}
    lv = np.log(pd.to_numeric(df.val_acc, errors="coerce").to_numpy(float))
    tasks = {
        "celniak_unified": {"name": "celniak_unified", "ds": "celniak", "X": df[CELNIAK_FEATS].to_numpy(float),
                            "names": CELNIAK_FEATS, **base},
        # labelled control: the same 19 features + the tracker's own validation error (log degrees) as a covariate
        "celniak_ctrl_valcov": {"name": "celniak_ctrl_valcov", "ds": "celniak",
                                "X": np.column_stack([df[CELNIAK_FEATS].to_numpy(float), lv]),
                                "names": CELNIAK_FEATS + ["log_validation_error"], **base},
        # reference: the validation error alone (one number per recording, repeated on its slides)
        "celniak_ref_valonly": {"name": "celniak_ref_valonly", "ds": "celniak", "X": lv[:, None],
                                "names": ["log_validation_error"], **base},
    }
    return tasks


# ----------------------------------------------------------------------------- Preti (fast extraction, one row per participant)
def build_preti(log=print):
    from edd import preti, preti_fast as pf
    xl = preti.load_workbook(ROOT / "data" / "preti" / "preti2026.xlsx")
    arrays = pf.arrays_from_frame(preti.fixations(xl))
    subj = sorted(arrays)
    X = np.vstack([pf.fusion_features(arrays[s]) for s in subj])
    y = np.array([arrays[s].y for s in subj], int)
    names = pf.feature_names("I1_diff_fusion")
    ocular = list(range(16))
    contrast_a = list(range(3, 8)) + list(range(11, 16))
    contrast_b = contrast_a + list(range(19, 26)) + list(range(29, 36))
    folds = json.loads((RESULTS / "phase2" / "preti" / "folds.json").read_text())["repeats"]
    g = np.array([str(s) for s in subj])
    meta = pd.DataFrame({"unit": g, "y": y})
    base = {"g": g, "y": y, "folds": folds, "person": "mean", "meta": meta, "label_scope": "unit", "ds": "preti"}
    out = {}
    for nm, idx in (("preti_A_ocular", ocular), ("preti_B_ocular_rt", list(range(36))),
                    ("preti_A_contrast_only", contrast_a), ("preti_B_contrast_only", contrast_b)):
        out[nm] = {"name": nm, "X": X[:, idx], "names": [names[i] for i in idx], **base}
    return out


# ----------------------------------------------------------------------------- Mazza (within-person z against own neutral images)
def build_mazza(log=print):
    from edd import mazza as mz
    d = mz.load(ROOT / "data" / "mazza" / "data.csv")
    raw = mz.raw_columns(d)
    fz = mz.Featurizer(raw_cols=raw, sets=("z",)).fit(d[raw].to_numpy(float))
    X = fz.transform(d[raw].to_numpy(float))
    names = list(fz.names_)
    folds_all = json.loads((RESULTS / "phase2" / "mazza" / "folds.json").read_text())["repeats"]
    tasks = {}
    for task, (groups, pos, _) in mz.TASKS.items():
        if task not in ("g2_vs_g3", "guilty_vs_innocent"):
            continue
        sub = {"g2_vs_g3": "g23", "guilty_vs_innocent": "all"}[task]
        m = d.group.isin(groups).to_numpy()
        y = d.group.isin(pos).astype(int).to_numpy()[m]
        g = d.unit.astype(str).to_numpy()[m]
        folds = [[[str(u) for u in fold] for fold in rep] for rep in folds_all[sub]]
        tasks[f"mazza_{task}"] = {"name": f"mazza_{task}", "ds": "mazza", "X": X[m], "y": y, "g": g, "names": names,
                                  "folds": folds, "person": "mean", "label_scope": "unit",
                                  "meta": pd.DataFrame({"unit": g, "y": y, "group": d.group.to_numpy()[m]})}
    return tasks


def shuffle_labels(task, seed=0):
    """Smoke-test only: person-labelled tasks get their labels permuted across participants; item-labelled tasks permute
    the labels within each participant (Chen: the trial types). Returns a shallow copy."""
    t = dict(task)
    rng = np.random.default_rng(seed)
    g, y = t["g"], t["y"].copy()
    if t["label_scope"] == "unit":
        ug = np.unique(g)
        lab = np.array([y[g == u][0] for u in ug])
        perm = dict(zip(ug, rng.permutation(lab)))
        y = np.array([perm[u] for u in g])
        t["meta"] = t["meta"].assign(y=y)
    else:
        if "T1" in t:
            T1 = t["T1"].copy()
            for u in np.unique(g):
                ix = np.flatnonzero(g == u)
                T1[ix] = rng.permutation(T1[ix])
            t["T1"] = T1
            y = (T1 == 2).astype(int)
            t["meta"] = t["meta"].assign(T1=T1, y=y)
        else:
            for u in np.unique(g):
                ix = np.flatnonzero(g == u)
                y[ix] = rng.permutation(y[ix])
            t["meta"] = t["meta"].assign(y=y)
    t["y"] = y
    return t


# ============================================================================= stored baselines (read only) and primary tests
MARGIN_DELMAS = 0.02
MARGIN_CASCADE = 0.05


def _unit_mean_scores(df, unit="unit"):
    """Per decision unit: mean over rows within a repeat, then over repeats (y constant per unit)."""
    a = df.groupby(["repeat", unit], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
    return a.groupby(unit, as_index=False).agg(y=("y", "first"), score=("score", "mean"))


def baseline_unit_scores(ds, task=None):
    """Participant-level honest-baseline scores (the Phase 2 out-of-fold scores on the same frozen folds)."""
    p2 = RESULTS / "phase2"
    if ds == "celniak":
        return _unit_mean_scores(pd.read_csv(p2 / "celniak" / "trials_baseline_svm_paper.csv"))
    if ds == "preti_A":
        return _unit_mean_scores(pd.read_csv(p2 / "preti" / "scores" / "B_ocular.csv"))
    if ds == "preti_B":
        return _unit_mean_scores(pd.read_csv(p2 / "preti" / "scores" / "B_fusion.csv"))
    if ds == "mazza":
        return _unit_mean_scores(pd.read_csv(p2 / "mazza" / "oof" / f"{task}__base_pub36_lr_unreg.csv"))
    raise KeyError(ds)


def uni_unit_scores(rows):
    r = rows.copy()
    r["unit"] = r["unit"].astype(str)
    return _unit_mean_scores(r)


def test_person_auc(base, uni, n_boot, seed, margin=None):
    """Participant-level AUC, unified minus baseline, paired participant-cluster bootstrap (decision units are participants)."""
    b, u = base.copy(), uni.copy()
    b["unit"], u["unit"] = b["unit"].astype(str), u["unit"].astype(str)
    common = set(b.unit) & set(u.unit)
    ga = unit_groups(b[b.unit.isin(common)].assign(y=lambda d: d.unit.map(dict(zip(u.unit, u.y)))))
    gb = unit_groups(u[u.unit.isin(common)])
    r = paired_unit_bootstrap(ga, gb, auc_items, n_boot, seed)
    d = r.pop("deltas")
    r["p_superior"] = p_superior(d)
    if margin is not None:
        r["margin"] = margin
        r["p_inferior"] = p_inferior(d, margin)
    return r


def delmas_primary(rows_by_task, n_boot, seed, drop=(), baseline_tag="P4_svm_1000"):
    """Image-level AUC pooled within each of the six group-sets, mean over sets; repeat-averaged scores; paired participant
    bootstrap within each set; non-inferiority margin 0.02 vs P4 SVM 1000 Hz (stored scores). `drop` removes sets (the
    sensitivity analysis that counts group 3 once drops '32_g3')."""
    base = pd.read_csv(RESULTS / "phase2" / "delmas_raw" / "oof_scores.csv.gz")
    base = base[base.tag == baseline_tag]
    per = {}
    for name, rows in rows_by_task.items():
        key = name.replace("delmas_", "")
        if key in drop:
            continue
        u = rows.assign(participant=rows.unit.astype(int)).groupby(["participant", "image"], as_index=False).agg(
            y=("y", "first"), score=("score", "mean"))
        b = base[base.set == key].groupby(["participant", "image"], as_index=False).agg(score_b=("score", "mean"))
        m = u.merge(b, on=["participant", "image"], how="inner")
        per[key] = [(d.y.to_numpy(), d.score_b.to_numpy(), d.score.to_numpy()) for _, d in m.groupby("participant")]
    rng = np.random.default_rng(seed)

    def mean_auc(col, idxs=None):
        a = []
        for k, v in per.items():
            sel = v if idxs is None else [v[i] for i in idxs[k]]
            y = np.concatenate([x[0] for x in sel])
            if len(np.unique(y)) < 2:
                continue
            a.append(fast_auc(y, np.concatenate([x[col] for x in sel])))
        return float(np.mean(a))

    ab, au = mean_auc(1), mean_auc(2)
    d = []
    for _ in range(n_boot):
        idx = {k: rng.integers(0, len(v), len(v)) for k, v in per.items()}
        d.append(mean_auc(2, idx) - mean_auc(1, idx))
    d = np.array(d)
    per_set = {}
    for k, v in per.items():
        y = np.concatenate([x[0] for x in v])
        per_set[k] = {"auc_base": fast_auc(y, np.concatenate([x[1] for x in v])),
                      "auc_unified": fast_auc(y, np.concatenate([x[2] for x in v])), "n_participants": len(v)}
    return {"est_base": ab, "est_unified": au, "delta": au - ab, "sets": sorted(per),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "lower_one_sided_95": float(np.percentile(d, 5)), "margin": MARGIN_DELMAS,
            "p_inferior": p_inferior(d, MARGIN_DELMAS), "per_set": per_set, "n_boot": n_boot}


# ----------------------------------------------------------------------------- Chen
def chen_checkpoint_persons():
    """The confirmatory run's stored person z statistics (arms rate_rule, I2b) on the same frozen folds."""
    pers, trials = [], []
    for r in range(3):
        c = pickle.load(open(CHEN_CONFIRM / "checkpoints" / f"repeat{r}.pkl", "rb"))
        pers.append(c["persons"])
        trials.append(c["trials"])
    return pd.concat(pers, ignore_index=True), pd.concat(trials, ignore_index=True)


def chen_detection_matrix(pdf):
    """Per participant (rows) x repeat (columns): calibrated detection indicator from a unified person table."""
    f = pdf[pdf.kind == "fam"].copy()
    f["det"] = (f.stat > f.tau_fpr7).astype(float)
    return f.pivot_table(index="unit", columns="repeat", values="det")


def chen_realised_fpr(pdf):
    f = pdf[pdf.kind.isin(["c1", "c2"])].copy()
    f["fp"] = (f.stat > f.tau_fpr7).astype(float)
    return float(f.fp.mean())


def chen_primary(persons_u, n_boot, seed):
    """Detection at the calibrated FPR<=7% threshold, pooled over both experiments (56 participants), unified minus the
    paper's rate rule, one-sided paired participant bootstrap."""
    cp, _ = chen_checkpoint_persons()
    det_u = chen_detection_matrix(persons_u)
    out = {}
    base = cp[cp.arm == "rate_rule"].pivot_table(index="unit", columns="repeat", values="detect_cal")
    i2b = cp[cp.arm == "I2b"].pivot_table(index="unit", columns="repeat", values="detect_cal")
    base.index = base.index.astype(str)
    i2b.index = i2b.index.astype(str)
    det_u.index = det_u.index.astype(str)
    units = sorted(set(det_u.index) & set(base.index))
    du, db, di = (m.loc[units].mean(1).to_numpy() for m in (det_u, base, i2b))
    rng = np.random.default_rng(seed)
    n = len(units)
    deltas, d_i2b = [], []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        deltas.append(du[k].mean() - db[k].mean())
        d_i2b.append(du[k].mean() - di[k].mean())
    deltas, d_i2b = np.array(deltas), np.array(d_i2b)
    out["detection"] = {"unified": float(du.mean()), "rate_rule": float(db.mean()), "I2b_confirmed": float(di.mean()),
                        "delta_vs_rate_rule": float(du.mean() - db.mean()),
                        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                        "p_superior": p_superior(deltas), "n_units": n,
                        "delta_vs_I2b_descriptive": float(du.mean() - di.mean()),
                        "ci95_vs_I2b": [float(np.percentile(d_i2b, 2.5)), float(np.percentile(d_i2b, 97.5))],
                        "realised_fpr_unified": chen_realised_fpr(persons_u),
                        "realised_fpr_rate_rule": float(cp[cp.arm == "rate_rule"].fpr_cal.mean()),
                        "realised_fpr_I2b": float(cp[cp.arm == "I2b"].fpr_cal.mean())}
    out["per_experiment_detection"] = {
        f"exp{e}": {"unified": float(chen_detection_matrix(persons_u[persons_u.exp == e]).mean().mean()),
                    "rate_rule": float(cp[(cp.arm == "rate_rule") & (cp.exp == e)].detect_cal.mean()),
                    "I2b": float(cp[(cp.arm == "I2b") & (cp.exp == e)].detect_cal.mean())} for e in (1, 2)}
    out["person_auc"] = _chen_person_auc(persons_u, cp, n_boot, seed + 1)
    return out


def _chen_person_groups(df, stat="stat", kind="kind"):
    d = df.groupby(["unit", kind], as_index=False).agg(s=(stat, "mean"))
    d["y"] = (d[kind] == "fam").astype(int)
    d["unit"] = d.unit.astype(str)
    return {u: (g.y.to_numpy(), g.s.to_numpy()) for u, g in d.groupby("unit")}


def _chen_person_auc(persons_u, cp, n_boot, seed):
    long = []
    for arm in ("rate_rule", "I2b"):
        d = cp[cp.arm == arm]
        for key, kind in (("z_fam", "fam"), ("z_c1", "c1"), ("z_c2", "c2")):
            long.append(pd.DataFrame({"arm": arm, "unit": d.unit, "kind": kind, "stat": d[key]}))
    long = pd.concat(long)
    gu = _chen_person_groups(persons_u)
    res = {"unified": auc_items(list(gu.values()))}
    for arm in ("rate_rule", "I2b"):
        ga = _chen_person_groups(long[long.arm == arm])
        r = paired_unit_bootstrap(ga, gu, auc_items, n_boot, seed)
        d = r.pop("deltas")
        res[arm] = {"auc": r["est_base"], "delta_unified_minus": r["delta"], "ci95": r["ci95"], "p_superior": p_superior(d)}
    return res


def chen_trial_auc(rows_u, n_boot, seed):
    """Descriptive: trial-level AUC (repeat-averaged scores), unified vs rate rule vs I2b, paired participant bootstrap."""
    _, ct = chen_checkpoint_persons()
    out = {}
    ru = rows_u.groupby("row", as_index=False).agg(y=("y", "first"), score=("score", "mean"), unit=("unit", "first"))
    gu = unit_groups(ru.assign(unit=ru.unit.astype(str)))
    out["unified"] = auc_items(list(gu.values()))
    for arm in ("rate_rule", "I2b"):
        a = ct[ct.arm == arm].groupby("row", as_index=False).agg(score=("score", "mean"))
        m = ru[["row", "y", "unit"]].merge(a, on="row", how="inner")
        ga = unit_groups(m.assign(unit=m.unit.astype(str)))
        gb = unit_groups(ru[ru.row.isin(m.row)].assign(unit=lambda d: d.unit.astype(str)))
        r = paired_unit_bootstrap(ga, gb, auc_items, n_boot, seed)
        d = r.pop("deltas")
        out[arm] = {"auc": r["est_base"], "delta_unified_minus": r["delta"], "ci95": r["ci95"], "p_superior": p_superior(d)}
    return out


def cascade_eval(persons_s1, n_boot, seed, cost=None):
    """Stage 1 = the tiny unified LR (10 features), stage 2 = the confirmed I2b model (read from the confirmatory run's
    stored statistics, same folds). Deferral rule fixed on TRAINING participants only: the stage-1 statistic must exceed
    tau_hi (training null: at most 2% above) for a confident 'detected', or fall at or below tau_lo (10th percentile of the
    training participants' familiar statistics) for a confident 'not detected'; everything between is deferred to I2b, whose
    own calibrated FPR<=7% decision is used."""
    cp, _ = chen_checkpoint_persons()
    i2 = cp[cp.arm == "I2b"]
    dec2 = {}
    for _, r in i2.iterrows():
        u, rep = str(r.unit), int(r["repeat"])
        dec2[(rep, u, "fam")] = float(r.z_fam > r.tau)
        dec2[(rep, u, "c1")] = float(r.z_c1 > r.tau)
        dec2[(rep, u, "c2")] = float(r.z_c2 > r.tau)
    rows = []
    for (rep, fold, exp), d in persons_s1.groupby(["repeat", "fold", "exp"]):
        pos = np.array(json.loads(d.train_pos.iloc[0]), float)
        null = np.array(json.loads(d.train_null.iloc[0]), float)
        pos, null = pos[np.isfinite(pos)], null[np.isfinite(null)]
        zs = np.sort(null)[::-1]
        tau_hi = float(zs[int(np.floor(0.02 * len(zs)))]) if len(zs) > int(np.floor(0.02 * len(zs))) else float("inf")
        tau_lo = float(np.quantile(pos, 0.10))
        for _, r in d.iterrows():
            s = r.stat
            if not np.isfinite(s):
                continue
            key = (int(r["repeat"]), str(r.unit), r.kind)
            if s > tau_hi:
                dec, deferred = 1.0, 0
            elif s <= tau_lo:
                dec, deferred = 0.0, 0
            else:
                dec, deferred = dec2[key], 1
            rows.append({"repeat": rep, "unit": str(r.unit), "kind": r.kind, "dec": dec, "deferred": deferred,
                         "dec_i2b": dec2[key], "tau_hi": tau_hi, "tau_lo": tau_lo})
    c = pd.DataFrame(rows)
    fam = c[c.kind == "fam"].pivot_table(index="unit", columns="repeat", values="dec").mean(1)
    fam_i = c[c.kind == "fam"].pivot_table(index="unit", columns="repeat", values="dec_i2b").mean(1)
    rng = np.random.default_rng(seed)
    du, di = fam.to_numpy(), fam_i.loc[fam.index].to_numpy()
    n = len(du)
    d = np.array([du[k].mean() - di[k].mean() for k in (rng.integers(0, n, n) for _ in range(n_boot))])
    res = {"detection_cascade": float(du.mean()), "detection_I2b": float(di.mean()), "delta": float(du.mean() - di.mean()),
           "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "margin": MARGIN_CASCADE,
           "p_inferior": p_inferior(d, MARGIN_CASCADE), "lower_one_sided_95": float(np.percentile(d, 5)),
           "realised_fpr_cascade": float(c[c.kind.isin(["c1", "c2"])].dec.mean()),
           "realised_fpr_I2b": float(c[c.kind.isin(["c1", "c2"])].dec_i2b.mean()),
           "defer_rate_all_cases": float(c.deferred.mean()), "defer_rate_familiar_cases": float(c[c.kind == "fam"].deferred.mean()),
           "defer_rate_control_cases": float(c[c.kind != "fam"].deferred.mean()), "n_units": n}
    if cost and "stage1_ms" in cost and "i2b_ms" in cost:
        res["cost_ms_per_decision"] = {"stage1": cost["stage1_ms"], "I2b": cost["i2b_ms"],
                                       "cascade_mean": cost["stage1_ms"] + res["defer_rate_all_cases"] * cost["i2b_ms"],
                                       "I2b_alone": cost["i2b_ms"]}
    return res, c


# ============================================================================= figure data (tidy CSVs; the charts are made elsewhere)
ROC_GRID = np.linspace(0, 1, 101)


def primary_level(task, rows):
    """Out-of-fold scores at the dataset's primary decision level: participants (mean row score) for person-labelled tasks,
    images (Delmas) or trials (Chen) otherwise. Columns: repeat, uid, unit, y, score."""
    if task["person"] == "mean":
        d = rows.groupby(["repeat", "unit"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
        d["uid"] = d["unit"]
        return d
    d = rows.copy()
    d["uid"] = d["row"] if task["person"] == "chen" else d["unit"].astype(str) + "_" + d["image"].astype(str)
    return d[["repeat", "uid", "unit", "y", "score"]]


def _curve(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    tg = np.interp(ROC_GRID, fpr, tpr)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return tg, float((fpr[i] + fnr[i]) / 2), float(thr[i]), fast_auc(y, s)


def roc_det_tables(specs, n_boot=500, seed=BOOT_SEED):
    """specs: [{dataset, task, level, model, groups: {unit: (y, score)}}]. ROC/DET on a 101-point FPR grid with participant-
    cluster bootstrap bands (2.5/97.5 percentiles) and the EER point."""
    curves, eers = [], []
    for sp in specs:
        units = sorted(sp["groups"])
        A = [sp["groups"][u] for u in units]
        y = np.concatenate([a[0] for a in A])
        s = np.concatenate([a[1] for a in A])
        if len(np.unique(y)) < 2:
            continue
        tpr, eer, thr, auc = _curve(y, s)
        rng = np.random.default_rng(seed)
        bt, be, ba = [], [], []
        for _ in range(n_boot):
            k = rng.integers(0, len(units), len(units))
            yb = np.concatenate([A[i][0] for i in k])
            if len(np.unique(yb)) < 2:
                continue
            t, e, _, a = _curve(yb, np.concatenate([A[i][1] for i in k]))
            bt.append(t)
            be.append(e)
            ba.append(a)
        bt = np.array(bt)
        lo, hi = np.percentile(bt, 2.5, axis=0), np.percentile(bt, 97.5, axis=0)
        curves.append(pd.DataFrame({"dataset": sp["dataset"], "task": sp["task"], "level": sp["level"], "model": sp["model"],
                                    "fpr": ROC_GRID, "tpr": tpr, "tpr_lo": lo, "tpr_hi": hi,
                                    "fnr": 1 - tpr, "fnr_lo": 1 - hi, "fnr_hi": 1 - lo}))
        eers.append({"dataset": sp["dataset"], "task": sp["task"], "level": sp["level"], "model": sp["model"], "n_units": len(units),
                     "n_decisions": len(y), "eer": eer, "eer_lo": float(np.percentile(be, 2.5)), "eer_hi": float(np.percentile(be, 97.5)),
                     "eer_score_threshold": thr, "auc": auc, "auc_lo": float(np.percentile(ba, 2.5)), "auc_hi": float(np.percentile(ba, 97.5)),
                     "n_boot": len(bt)})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(eers)


def score_distribution_table(dataset, task, level, df, persons=None):
    """One row per decision unit: repeat-averaged out-of-fold score, label and the (fold-averaged) calibrated thresholds.
    `df` columns: uid, unit, y, score (already repeat-averaged). `persons` (person table) supplies the thresholds."""
    d = df.copy()
    d.insert(0, "level", level)
    d.insert(0, "task", task)
    d.insert(0, "dataset", dataset)
    d["threshold_logit_zero"] = 0.0 if level == "row" else np.nan
    if persons is not None and len(persons):
        d["threshold_eer_calibrated"] = float(persons.tau_eer.replace([np.inf, -np.inf], np.nan).mean())
        d["threshold_fpr7_calibrated"] = float(persons.tau_fpr7.replace([np.inf, -np.inf], np.nan).mean())
    else:
        d["threshold_eer_calibrated"] = d["threshold_fpr7_calibrated"] = np.nan
    return d


def calibration_bins(dataset, task, level, df, n_bins=10):
    """Equal-count bins of repeat-averaged out-of-fold probabilities sigmoid(logit) vs the observed positive rate (Wilson 95%)."""
    d = df.copy()
    d["p"] = 1 / (1 + np.exp(-d.score))
    nb = int(min(n_bins, max(2, len(d) // 8)))
    d["bin"] = pd.qcut(d.p.rank(method="first"), nb, labels=False)
    rows = []
    for b, g in d.groupby("bin"):
        n, k = len(g), int(g.y.sum())
        ph = k / n
        z = 1.96
        den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        rows.append({"dataset": dataset, "task": task, "level": level, "bin": int(b), "n": n, "mean_predicted": float(g.p.mean()),
                     "observed_positive_rate": ph, "obs_lo": float(c - h), "obs_hi": float(c + h)})
    return pd.DataFrame(rows)


def validation_curve_table(dataset, task, task_obj, res_by_repeat, meta):
    """Per C: held-out and training log-loss and AUC. Held-out log-loss is at row level; held-out AUC at the primary level.
    Mean and SD over repeats; train values are averaged over outer folds."""
    recs = []
    for j, C in enumerate(C_GRID):
        ho_ll, ho_auc, tr_ll, tr_auc, inner, chosen = [], [], [], [], [], []
        for r, folds in res_by_repeat.items():
            rows = []
            for f, d in enumerate(folds):
                df = meta.iloc[d["te"]][["unit", "y"] + [c for c in ("image",) if c in meta.columns]].copy()
                df["row"], df["repeat"], df["score"] = d["te"], r, d["val_te"][:, j]
                rows.append(df)
            rows = pd.concat(rows)
            ho_ll.append(logloss(rows.y, rows.score))
            pl = primary_level(task_obj, rows)
            ho_auc.append(fast_auc(pl.y, pl.score) if len(np.unique(pl.y)) == 2 else float("nan"))
            tr_ll.append(np.mean([d["val_tr_ll"][j] for d in folds]))
            tr_auc.append(np.mean([d["val_tr_auc"][j] for d in folds]))
            inner.append(np.nanmean([d["cv_auc"][j] for d in folds]))
            chosen.append(np.mean([d["C"] == C for d in folds]))
        recs.append({"dataset": dataset, "task": task, "C": C,
                     "heldout_logloss": np.mean(ho_ll), "heldout_logloss_sd": np.std(ho_ll),
                     "heldout_auc": np.nanmean(ho_auc), "heldout_auc_sd": np.nanstd(ho_auc),
                     "train_logloss": np.mean(tr_ll), "train_auc": np.mean(tr_auc),
                     "inner_cv_auc": np.nanmean(inner), "fraction_of_folds_choosing_C": np.mean(chosen)})
    return pd.DataFrame(recs)


def learning_curve_table(dataset, task, task_obj, res_by_frac, meta, n_boot=300, seed=BOOT_SEED):
    """AUC vs fraction of training participants. Per fraction: mean and SD over repeats of the primary-level AUC, and a
    participant-cluster bootstrap CI on the repeat-averaged scores."""
    recs = []
    for frac, res in sorted(res_by_frac.items()):
        rows = oof_rows(res, meta)
        pl = primary_level(task_obj, rows)
        per = [fast_auc(d.y, d.score) for _, d in pl.groupby("repeat") if len(np.unique(d.y)) == 2]
        avg = pl.groupby("uid", as_index=False).agg(y=("y", "first"), score=("score", "mean"), unit=("unit", "first"))
        groups = {u: (d.y.to_numpy(), d.score.to_numpy()) for u, d in avg.groupby("unit")}
        units = sorted(groups)
        rng = np.random.default_rng(seed)
        b = []
        for _ in range(n_boot):
            k = rng.integers(0, len(units), len(units))
            yb = np.concatenate([groups[units[i]][0] for i in k])
            if len(np.unique(yb)) == 2:
                b.append(fast_auc(yb, np.concatenate([groups[units[i]][1] for i in k])))
        ntr = np.mean([d["n_train_units"] for folds in res.values() for d in folds])
        recs.append({"dataset": dataset, "task": task, "fraction_of_training_participants": frac,
                     "n_train_participants_mean": float(ntr), "auc_mean": float(np.mean(per)), "auc_sd_over_repeats": float(np.std(per)),
                     "auc_repeat_avg_scores": fast_auc(avg.y, avg.score), "auc_lo": float(np.percentile(b, 2.5)),
                     "auc_hi": float(np.percentile(b, 97.5)), "n_repeats": len(per)})
    return pd.DataFrame(recs)


def feature_weight_table(dataset, task, names, res_by_repeat):
    """Standardised coefficients (inputs are standardised, so coefficients are directly comparable), across all outer folds x
    repeats: mean, 2.5/97.5 percentiles and the share of models with a positive sign. Positive = towards the 'deceptive/known'
    class."""
    W = np.array([d["coef"] for folds in res_by_repeat.values() for d in folds])
    return pd.DataFrame({"dataset": dataset, "task": task, "feature": names, "coef_mean": W.mean(0),
                         "coef_lo": np.percentile(W, 2.5, axis=0), "coef_hi": np.percentile(W, 97.5, axis=0),
                         "share_positive": (W > 0).mean(0), "n_models": len(W), "abs_rank": (-np.abs(W.mean(0))).argsort().argsort() + 1})


FIGDATA_README = """# Figure data for the unified pipeline (Phase 4)

All files are tidy CSVs written by `scripts/phase4_unified.py --stage figdata`. Every number comes from out-of-fold scores (the
decision unit never saw its own participant in training). `dataset` is one of delmas, chen, celniak, preti, mazza; `task` names the
frozen-fold task (Delmas: one per group-set; Celniak: unified plus labelled control/reference; Preti: Level A/B plus contrast-only
sensitivity; Mazza: g2_vs_g3, guilty_vs_innocent). No charts are produced here; the senior analyst draws them.

`level`: `row` = the trial/image/slide row scored by the model (Delmas image, Chen trial, Celniak slide); `person` = the participant-
level decision statistic (Celniak, Preti, Mazza: mean of the participant's row scores; Delmas: CIT z statistic of known vs
irrelevant images, negatives are simulated innocents; Chen: one-sided-t z of familiar vs control trials, negatives are control-1-vs-2).
`model`: `unified` = this pipeline; `baseline` = the stored honest baseline of the track on the same folds (person-level only).

## roc_det_curves.csv
One row per (dataset, task, level, model, fpr grid point), 101 points from 0 to 1. `tpr` = ROC point estimate (interpolated on the
FPR grid), `tpr_lo/tpr_hi` = 2.5/97.5 percentile of a participant-cluster bootstrap (resampling participants with replacement,
500 draws, the same draws for the same dataset/task). DET = `fpr` against `fnr` (= 1 - tpr; bands `fnr_lo = 1 - tpr_hi`,
`fnr_hi = 1 - tpr_lo`). Scores are repeat-averaged out-of-fold scores.

## roc_det_eer.csv
One row per curve: `eer` = equal error rate point (mean of FPR and FNR at the closest crossing), `eer_lo/eer_hi` bootstrap 95% interval,
`eer_score_threshold` = score at that point (units of the score; logit for row level), `auc` with bootstrap interval, `n_units`
(participants), `n_decisions` (rows), `n_boot` valid draws.

## score_distributions.csv
One row per decision unit (`uid`; for Delmas `participant_image`, Chen the trial row index, otherwise the participant): repeat-averaged
out-of-fold `score`, label `y` (1 = positive class), `unit` (participant), `level`. `threshold_eer_calibrated` and
`threshold_fpr7_calibrated` = mean over outer folds and repeats of the thresholds calibrated on training participants only (person level
only; statistic units); `threshold_logit_zero` = 0, the LR decision threshold, for row level.

## learning_curve.csv
`fraction_of_training_participants` 0.25/0.5/0.75/1.0 (each outer fold's training participants are subsampled, stratified by class when
the label belongs to the person; C is re-tuned in nested folds). `auc_mean`/`auc_sd_over_repeats`: primary-level AUC per repeat, mean and
SD. `auc_repeat_avg_scores` with `auc_lo/auc_hi`: participant-cluster bootstrap (300 draws) on repeat-averaged scores. `n_train_participants_mean` =
mean number of training participants actually used.

## validation_curve.csv
One row per (task, C) on the fixed grid. `heldout_logloss` (unweighted log-loss of sigmoid(logit), row level) and `heldout_auc` (primary
level), mean and SD over repeats, from models refitted at each C on the outer-training participants. `train_logloss`/`train_auc` =
resubstitution values averaged over outer folds. `inner_cv_auc` = the selection criterion (mean per-fold inner AUC) averaged over outer
folds; `fraction_of_folds_choosing_C` = how often nested tuning picked that C.

## feature_weights.csv
Standardised coefficients (the model is fitted on standardised inputs): `coef_mean`, `coef_lo/coef_hi` = 2.5/97.5 percentile across all outer
folds x repeats, `share_positive`, `n_models`, `abs_rank` (1 = largest |mean|). Positive = towards the positive class (known face /
guilty / liar / familiar).

## calibration_bins.csv
Equal-count bins (up to 10, fewer for small n) of repeat-averaged out-of-fold probability sigmoid(logit): `mean_predicted`,
`observed_positive_rate` with Wilson 95% `obs_lo/obs_hi`, `n`. The model uses balanced class weights, so probabilities refer to a balanced
prior, not to the sample's base rate (supplementary plot).

## chen_rate_curve.csv (analysis A)
One row per (model, rate): `model` unified or I2b or rate_rule, `rate_hz` 100/60/30, `trial_auc` (repeat-averaged out-of-fold scores pooled over trials), `person_auc`, `detection_fpr7` (person detection at the calibrated FPR <= 7% threshold), `realised_fpr`, and for non-rate-rule rows the detection difference to the rate rule at 100 Hz with its bootstrap 95% interval. The non-inferiority and superiority tests are in `results_chen_rate.json`.

## webcam_adaptive_curve.csv / webcam_adaptive_thresholds.csv (analysis B)
Curve: one row per (rule, level_deg): `auc_level` (image-level AUC, mean over sets and draws), `auc_reference` (clean P4 SVM 1000 Hz), `delta`, `ci_lo/ci_hi` (paired bootstrap 95%), `lower_one_sided_95`, `p_inferior` (H0: delta <= -0.02), `p_holm` (Holm over the six levels within the rule), `non_inferior_holm`. `rule` ek = primary median-based threshold, nh = secondary Nystrom-Holmqvist. Thresholds: one row per (rule, level, draw, participant): the data-driven fixation velocity threshold (deg/s), the peak threshold (nh only), and the realised calibration offset of that participant's noise draw (deg).

## efficiency.csv
Same session, single thread, interleaved rounds, baseline vs unified, per dataset: `latency_p50_ms`/`latency_p95_ms` per decision unit
(median over rounds of each round's percentile), `throughput_per_s`, `samples_per_s` and `real_time_factor` (where a raw sensor signal is
processed; real-time factor = samples per second / sensor rate), `peak_memory_kb` (tracemalloc), `model_size_bytes` (pickle), `parameters`,
`rounds`, `runs_per_round`. `stage` says what is timed (extraction + model, from which input).
"""


# ============================================================================= efficiency: baseline vs unified, same session
def fit_deploy(task_or_arrays, seed=0):
    """Final deployable model: nested-tuned C on all rows of the task(s), refit, packaged as `DeployLR`."""
    ts = task_or_arrays if isinstance(task_or_arrays, (list, tuple)) else [task_or_arrays]
    X = np.vstack([t["X"] for t in ts])
    y = np.concatenate([t["y"] for t in ts])
    g = np.concatenate([t["g"] for t in ts])
    with threadpool_limits(1):
        f = tune_fit(X, y, g, seed)
    return DeployLR(f.P, f.coef, f.intercept, None, f.C)       # numbers only: feature names are not part of the stored model


def interleaved(fns, rounds=5, runs=None, warmup=3):
    """Single-thread, back-to-back rounds with the order alternated each round. Returns, per name, the median over rounds of
    each round's p50 and p95 latency (ms) and the per-round p50 list."""
    runs = runs or {}
    per = {k: [] for k in fns}
    with threadpool_limits(1):
        for fn in fns.values():
            for _ in range(warmup):
                fn()
        for r in range(rounds):
            order = list(fns) if r % 2 == 0 else list(fns)[::-1]
            for k in order:
                n = runs.get(k, 20)
                t = np.empty(n)
                for i in range(n):
                    t0 = time.perf_counter()
                    fns[k]()
                    t[i] = time.perf_counter() - t0
                per[k].append((np.percentile(t, 50) * 1e3, np.percentile(t, 95) * 1e3))
    return {k: {"p50_ms": float(np.median([a for a, _ in v])), "p95_ms": float(np.median([b for _, b in v])),
                "p50_rounds_ms": [float(a) for a, _ in v], "rounds": rounds} for k, v in per.items()}


def _eff_rows(dataset, stage, fns, meta, rounds, runs):
    """fns: {name: callable}; meta: {name: dict(model=..., params=int, samples=int|None, sensor_hz=float|None, mem_fn=callable)}."""
    from edd import efficiency as eff
    lat = interleaved(fns, rounds=rounds, runs=runs)
    rows = []
    for k, v in lat.items():
        m = meta[k]
        mem = eff.peak_memory_bytes(m.get("mem_fn", fns[k]))
        sps = m["samples"] / (v["p50_ms"] / 1e3) if m.get("samples") else float("nan")
        rows.append({"dataset": dataset, "variant": k, "stage": stage, "latency_p50_ms": v["p50_ms"], "latency_p95_ms": v["p95_ms"],
                     "throughput_per_s": 1e3 / v["p50_ms"], "samples_per_s": sps,
                     "real_time_factor": sps / m["sensor_hz"] if m.get("samples") and m.get("sensor_hz") else float("nan"),
                     "peak_memory_kb": mem / 1024, "model_size_bytes": eff.model_size_bytes(m["model"]),
                     "parameters": int(m["params"]), "rounds": rounds, "runs_per_round": runs.get(k, 20),
                     "sensor_hz": m.get("sensor_hz", float("nan")), "samples_per_decision": m.get("samples", float("nan")),
                     "p50_rounds_ms": json.dumps(v["p50_rounds_ms"])})
    return rows


def _svm_params(pipe_or_svc, n_feat):
    from edd import efficiency as eff
    svc = getattr(pipe_or_svc, "named_steps", {}).get("clf", pipe_or_svc)
    return eff.svm_parameter_count(svc, n_feat) + 3 * n_feat


def eff_delmas(tasks, rounds, runs_orig, runs_fast, pid=4):
    from edd import delmas_fast as dfast, delmas_raw as dr
    from edd.delmas_raw_matched import load_track
    track = load_track()
    z = np.load(DELMAS_DECODED / f"p{pid:02d}.npz")
    seg = {k: z[k] for k in ("x", "y", "p", "images")}
    df = track.build_tables(ROOT / "data" / "derived", ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx",
                            ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx")
    d = df[(df.rate == 1000) & (df.set == "64")].reset_index(drop=True)
    svm = track.model_factory("svm", 0).fit(d[track.P4].to_numpy(float), d.face.to_numpy(int), groups=d.participant.to_numpy()).best_estimator_
    from edd.harness import scores_of
    pre1000 = [dr.degrade(seg["x"][k], seg["y"][k], seg["p"][k], 1000) for k in range(64)]
    X30, Y30, P30 = dfast.degrade_batch(seg["x"], seg["y"], seg["p"], 30)       # simulated 30 Hz sensor output, outside the timer
    dep = fit_deploy([tasks[f"delmas_64_g{g}"] for g in (1, 2, 3)])
    feats = [f + "_z" for f in DELMAS_FEATS]
    n1000, n30 = 64 * seg["x"].shape[1], 64 * X30.shape[1]
    sc = lambda m, A: m.decision_function(A)
    fns = {"baseline_P4_SVM_1000Hz_original_code": lambda: track.raw_to_scores(pre1000, seg["images"], 1000, svm, track.P4),
           "unified_30Hz_fast_chunk64": lambda: dfast.raw_to_scores_fast((X30, Y30, P30), seg["images"], 30, dep, feats, sc, chunk=64),
           "unified_30Hz_fast_chunk8": lambda: dfast.raw_to_scores_fast((X30, Y30, P30), seg["images"], 30, dep, feats, sc, chunk=8)}
    meta = {"baseline_P4_SVM_1000Hz_original_code": {"model": svm, "params": _svm_params(svm, 4), "samples": n1000, "sensor_hz": 1000.0},
            "unified_30Hz_fast_chunk64": {"model": dep, "params": dep.n_parameters, "samples": n30, "sensor_hz": 30.0},
            "unified_30Hz_fast_chunk8": {"model": dep, "params": dep.n_parameters, "samples": n30, "sensor_hz": 30.0}}
    runs = {"baseline_P4_SVM_1000Hz_original_code": runs_orig, "unified_30Hz_fast_chunk64": runs_fast, "unified_30Hz_fast_chunk8": runs_fast}
    return _eff_rows("delmas", "samples at sensor rate -> I-VT -> features -> model scores for 64 images (one participant; decimation excluded)",
                     fns, meta, rounds, runs)


def _chen_p_arrays(meta, ch, unit):
    m = (meta.unit.astype(int).astype(str) == str(unit)).to_numpy()
    return {k: v[m] for k, v in ch.items()}, meta.T1.to_numpy()[m].astype(int)


def eff_chen(tasks, chen_data, rounds, runs, unit=None):
    """From the preprocessed 100 Hz epochs of one participant (the raw EDF read and blink reconstruction are the same for every
    method and are excluded): the paper's rate rule, the unified LR (30 Hz bins), the cascade's tiny stage 1, and I2b."""
    from edd import chen_confirm as cc
    meta, ch = chen_data
    unit = unit or str(int(meta.unit.iloc[0]))
    chp, T1 = _chen_p_arrays(meta, ch, unit)
    P = cc.p2()
    dep = fit_deploy(tasks["chen_unified"])
    dep1 = fit_deploy(tasks["chen_stage1"])
    # baselines fitted on all rows (peak sample for the rate rule; I2b pipeline with its tuning)
    allm = np.arange(len(meta))
    y_all = (meta.T1.to_numpy() == cc.FAM).astype(int)
    peak = P.PeakRule("rate").fit(ch, allm, y_all).peak
    X125, _ = P.bin_features(ch, P.FUSION)
    i2b, _ = P.fit_lr(X125, y_all, meta.unit.to_numpy(), True)

    def unified():
        X, _ = chen_features(chp)
        Xz = zscore_within(X, np.zeros(len(X)))
        return cc.person_z(dep.decision_function(Xz), T1)

    def stage1():
        Xr, nm = chen_features(chp, channels=("pupil", "dx"))
        nm = np.array(nm)
        co = [Xr[:, [i for i, n in enumerate(nm) if n.startswith(c + "_")]].reshape(len(Xr), 5, 5).mean(2) for c in ("pupil", "dx")]
        return cc.person_z(dep1.decision_function(zscore_within(np.hstack(co), np.zeros(len(Xr)))), T1)

    def rate_rule():
        return cc.person_z(chp["rate"][:, peak], T1)

    def i2b_fn():
        X, _ = P.bin_features(chp, P.FUSION)
        return cc.person_z(i2b.decision_function(X), T1)

    n_samp = len(T1) * 250 * 3
    fns = {"baseline_rate_rule_100Hz": rate_rule, "unified_30Hz_LR": unified, "cascade_stage1_tiny_LR": stage1, "I2b_confirmed_model": i2b_fn}
    meta_e = {"baseline_rate_rule_100Hz": {"model": {"peak": peak}, "params": 2, "samples": len(T1) * 250, "sensor_hz": 100.0},
              "unified_30Hz_LR": {"model": dep, "params": dep.n_parameters, "samples": n_samp, "sensor_hz": 100.0},
              "cascade_stage1_tiny_LR": {"model": dep1, "params": dep1.n_parameters, "samples": len(T1) * 250 * 2, "sensor_hz": 100.0},
              "I2b_confirmed_model": {"model": i2b, "params": int(i2b.named_steps["lr"].coef_.size + 1), "samples": len(T1) * 250 * 5, "sensor_hz": 100.0}}
    rows = _eff_rows("chen", f"preprocessed 100 Hz epochs ({len(T1)} trials of one participant) -> features -> trial scores -> person z (raw EDF read excluded; samples = epoch samples over all channels used)",
                     fns, meta_e, rounds, {k: runs for k in fns})
    return rows


def eff_celniak(tasks, rounds, runs_orig, runs_fast, name="Recording10"):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from edd import celniak, celniak_fast as cf
    samples, meta = celniak.load_cached(CELNIAK_CACHE, name)
    rec = cf.arrays_from_frame(samples)                 # one-time ingestion, excluded (as in the phase-3 benchmark)
    stored = pd.read_csv(ROOT / "data" / "derived" / "celniak_slide_features.csv")
    svm = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                    ("clf", SVC(C=1.0, gamma=0.1, class_weight="balanced"))]).fit(stored[CELNIAK_FEATS].to_numpy(float), stored.label.to_numpy(int))
    dep = fit_deploy(tasks["celniak_unified"])
    n = len(samples)
    fns = {"baseline_paper_features_SVM_original_code": lambda: svm.decision_function(
               celniak.slide_features(samples, meta)[CELNIAK_FEATS].to_numpy(float)).mean(),
           "unified_fast_extraction_LR": lambda: dep.decision_function(cf.paper_matrix(cf.slide_features_fast(rec), CELNIAK_FEATS)).mean()}
    meta_e = {"baseline_paper_features_SVM_original_code": {"model": svm, "params": _svm_params(svm, 19), "samples": n, "sensor_hz": 250.0},
              "unified_fast_extraction_LR": {"model": dep, "params": dep.n_parameters, "samples": n, "sensor_hz": 250.0}}
    return _eff_rows("celniak", f"cached raw Tobii samples of {name} (~{n} samples, 250 Hz) -> slide features -> participant score (CSV ingestion excluded)",
                     fns, meta_e, rounds, {"baseline_paper_features_SVM_original_code": runs_orig, "unified_fast_extraction_LR": runs_fast})


def eff_preti(tasks, rounds, runs_orig, runs_fast):
    from sklearn.linear_model import LogisticRegression

    from edd import preti, preti_fast as pf
    xl = preti.load_workbook(ROOT / "data" / "preti" / "preti2026.xlsx")
    f = preti.fixations(xl)
    arrays = pf.arrays_from_frame(f)
    t_all = preti.trials_from_fixations(f)
    tab = preti.arm_subject_table(t_all, "B_fusion")
    cols = [c for c in tab.columns if c not in ("subject", "y")]
    base_m = preti.pipe(LogisticRegression(C=1.0, max_iter=2000)).fit(tab[cols].to_numpy(float), tab.y.to_numpy(int))
    s = sorted(arrays)[0]
    f_one = f[f.subject == s]
    dep = fit_deploy(tasks["preti_B_ocular_rt"])
    n_fix = len(f_one)
    fns = {"baseline_B_fusion_LR_original_code": lambda: base_m.decision_function(
               preti.arm_subject_table(preti.trials_from_fixations(f_one), "B_fusion")[cols].to_numpy(float)),
           "unified_LevelB_fast_extraction_LR": lambda: dep.decision_function(pf.fusion_features(arrays[s]))}
    meta_e = {"baseline_B_fusion_LR_original_code": {"model": base_m, "params": int(cols and len(cols) + 1 + 3 * len(cols)), "samples": None},
              "unified_LevelB_fast_extraction_LR": {"model": dep, "params": dep.n_parameters, "samples": None}}
    rows = _eff_rows("preti", f"fixation table of one participant ({n_fix} fixations, 32 trials) -> participant features -> score (workbook read excluded)",
                     fns, meta_e, rounds, {"baseline_B_fusion_LR_original_code": runs_orig, "unified_LevelB_fast_extraction_LR": runs_fast})
    return rows


def eff_mazza(tasks, rounds, runs):
    from sklearn.linear_model import LogisticRegression

    from edd import mazza as mz
    d = mz.load(ROOT / "data" / "mazza" / "data.csv")
    raw = mz.raw_columns(d)
    row = d[raw].to_numpy(float)[:1]
    sub = d[d.group.isin([2, 3])]
    base_f = mz.Featurizer(raw_cols=raw, sets=("pub36",)).fit(row)
    base_m = LogisticRegression(penalty=None, class_weight="balanced", max_iter=3000).fit(
        base_f.transform(sub[raw].to_numpy(float)), sub.group.eq(2).astype(int))
    uni_f = mz.Featurizer(raw_cols=raw, sets=("z",)).fit(row)
    dep = fit_deploy(tasks["mazza_g2_vs_g3"])
    fns = {"baseline_pub36_LR": lambda: base_m.decision_function(base_f.transform(row)),
           "unified_z_LR": lambda: dep.decision_function(uni_f.transform(row))}
    meta_e = {"baseline_pub36_LR": {"model": base_m, "params": 37, "samples": None},
              "unified_z_LR": {"model": dep, "params": dep.n_parameters, "samples": None}}
    return _eff_rows("mazza", "one participant's 90 aggregate columns -> features -> score (webcam extraction not part of the data)",
                     fns, meta_e, rounds, {k: runs for k in fns})


def run_efficiency(tasks, chen_data, rounds=5, quick=False, log=print):
    from edd import efficiency as eff
    ro, rf = (3, 20) if quick else (6, 60)
    rows, errors = [], {}
    jobs = {"delmas": lambda: eff_delmas(tasks, rounds, 2 if quick else 4, 6 if quick else 30),
            "chen": lambda: eff_chen(tasks, chen_data, rounds, 3 if quick else 15),
            "celniak": lambda: eff_celniak(tasks, rounds, 1 if quick else 3, 3 if quick else 15),
            "preti": lambda: eff_preti(tasks, rounds, ro, 10 * rf),
            "mazza": lambda: eff_mazza(tasks, rounds, 20 * rf)}
    for k, fn in jobs.items():
        try:
            rows += fn()
            log(f"  efficiency {k}: done")
        except Exception as e:                                     # one dataset failing must not stop the others
            errors[k] = repr(e)
            log(f"  efficiency {k}: FAILED {e!r}")
    return pd.DataFrame(rows), {"hardware": eff.hardware_info(), "errors": errors, "rounds": rounds}


# ============================================================================= A. Chen sampling-rate test
CHEN_REF_RATE = 100                      # highest rate cached for BOTH pupil and gaze (gaze exists only as 100 Hz epochs)
CHEN_RATES = (100, 60, 30)
MARGIN_RATE = 0.02


def build_chen_rate_tasks(chen_data, rates=(100, 60)):
    """Unified-recipe Chen tasks at other sensor rates (30 Hz is the main task `chen_unified`)."""
    from edd import chen_confirm as cc
    meta, ch = chen_data
    g = meta.unit.astype(int).astype(str).to_numpy()
    T1 = meta.T1.to_numpy().astype(int)
    y = (T1 == cc.FAM).astype(int)
    exps = meta.exp.to_numpy().astype(int)
    folds = json.loads((CHEN_CONFIRM / "folds.json").read_text())["repeats"]
    out = {}
    for fs in rates:
        X, names = chen_features(ch, fs=fs)
        out[f"chen_unified_r{fs}"] = {
            "name": f"chen_unified_r{fs}", "ds": "chen", "X": zscore_within(X, g), "names": names, "g": g, "y": y, "folds": folds,
            "person": "chen", "T1": T1, "exps": exps, "label_scope": "within_unit",
            "meta": pd.DataFrame({"unit": g, "y": y, "T1": T1, "exp": exps})}
    return out


def chen_i2b_features(ch, fs):
    """The I2b feature set (pupil, rate, dx, dy, missing fraction; 100-ms bins, 125 features) rebuilt at sensor rate `fs`: every
    channel decimated from the 100 Hz epochs with the anti-aliased polyphase filter (gaps interpolated first), the rate channel
    recomputed as diff of the decimated pupil smoothed with a Hanning window of the same 250 ms. fs = 100 is the stored
    confirmatory feature set."""
    from edd import chen, chen_confirm as cc
    P = cc.p2()
    d = {c: decimate_to_rate(ch[c], fs) for c in ("pupil", "dx", "dy", "miss")}
    if fs == 100:
        d["rate"] = ch["rate"]
    else:
        d["rate"] = chen.smooth_hanning(np.diff(d["pupil"], axis=1), int(round(0.25 * fs)))
    edges = np.round(np.arange(26) * fs * 0.1).astype(int)
    X, names = P.bin_features(d, P.FUSION, edges)
    return X, names


def _i2b_fold(X, y, units, T1, te_units):
    from edd import chen_confirm as cc
    with threadpool_limits(1):
        warnings.filterwarnings("ignore")
        te_all = np.isin(units.astype(str), te_units)
        tr, te = np.flatnonzero(~te_all), np.flatnonzero(te_all)
        score, _, oof, params = cc._lr_arm(X, y, units, tr, te, T1)
        return {"te": te, "tr": tr, "score": score, "inner_oof": oof, "C": params.get("lr__C", np.nan), "params": str(params)}


def run_chen_i2b(name, X, task, ck_dir, n_jobs=2, repeats=None, log=print):
    """The confirmed I2b pipeline (screening `fit_lr`: SelectKBest, C grid {0.003, 0.03, 0.3}, nested participant-grouped
    tuning) on the frozen confirmatory folds, pooled-training arm only. One checkpoint per repeat."""
    ck_dir = Path(ck_dir)
    units = task["g"].astype(int)
    out = {}
    for r, folds in enumerate(task["folds"]):
        if repeats is not None and r not in repeats:
            continue
        ck = ck_dir / f"{name}__f100__r{r}.pkl"
        if ck.exists():
            out[r] = pickle.load(open(ck, "rb"))
            continue
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(delayed(_i2b_fold)(X, task["y"], units, task["T1"], [str(u) for u in tu]) for tu in folds)
        pickle.dump(res, open(ck.with_suffix(".tmp"), "wb"))
        ck.with_suffix(".tmp").replace(ck)
        out[r] = res
        log(f"  [{name}] repeat {r}: {len(res)} folds in {time.time() - t0:.0f}s")
    return out


def cp_persons(cp, arm):
    """Confirmatory run's stored person statistics in the unified person-table layout."""
    d = cp[cp.arm == arm]
    parts = []
    for kind, y, key in (("fam", 1, "z_fam"), ("c1", 0, "z_c1"), ("c2", 0, "z_c2")):
        parts.append(pd.DataFrame({"repeat": d["repeat"].to_numpy(), "unit": d.unit.astype(str).to_numpy(), "exp": d.exp.to_numpy(),
                                   "kind": kind, "y": y, "stat": d[key].to_numpy(), "tau_fpr7": d.tau.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def _trial_groups(rows, y_from):
    a = rows.groupby("row", as_index=False).agg(score=("score", "mean"))
    m = y_from[["row", "y", "unit"]].merge(a, on="row", how="inner")
    return {u: (d.y.to_numpy(), d.score.to_numpy()) for u, d in m.assign(unit=m.unit.astype(str)).groupby("unit")}


def chen_rate_analysis(rows_by, persons_by, ref_rows, n_boot, seed):
    """rows_by / persons_by: {(model, fs): out-of-fold trial rows / person table} for model in {unified, I2b}, fs in 100, 60, 30
    plus ('rate_rule', 100) read from the confirmatory run. `ref_rows` supplies row identity, label and unit for pairing.
    Primary outcome: trial AUC. Tests (paired participant-cluster bootstrap, the same resamples throughout):
      A1 (non-inferiority, margin 0.02): for model in {unified, I2b}, fs in {60, 30}: AUC(model, fs) - AUC(model, 100).
      A2 (superiority): AUC(model, fs) - AUC(paper's rate rule, 100) for the same four low-rate models.
    Holm within A1 and within A2 separately; secondary (descriptive, CIs only): person AUC and detection at FPR <= 7%."""
    groups = {k: _trial_groups(v, ref_rows) for k, v in rows_by.items()}
    common = sorted(set.intersection(*[set(g) for g in groups.values()]))
    keys = list(groups)
    A = {k: [groups[k][u] for u in common] for k in keys}
    n = len(common)
    res = {"trial_auc": {f"{k[0]}@{k[1]}": auc_items(A[k]) for k in keys}}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        for k in keys:
            draws[k].append(auc_items([A[k][i] for i in pick]))
    draws = {k: np.array(v) for k, v in draws.items()}

    def comp(a, b):
        d = draws[b] - draws[a]
        return d, {"delta": float(res["trial_auc"][f"{b[0]}@{b[1]}"] - res["trial_auc"][f"{a[0]}@{a[1]}"]),
                   "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                   "lower_one_sided_95": float(np.nanpercentile(d, 5))}
    a1, a2 = [], []
    for m in ("unified", "I2b"):
        for fs in (60, 30):
            d, r = comp((m, 100), (m, fs))
            r.update({"test": f"A1 {m} {fs}Hz vs {m} 100Hz", "margin": MARGIN_RATE, "p_inferior": p_inferior(d, MARGIN_RATE)})
            a1.append(r)
            d, r = comp(("rate_rule", 100), (m, fs))
            r.update({"test": f"A2 {m} {fs}Hz vs rate rule 100Hz", "p_superior": p_superior(d)})
            a2.append(r)
    for fam, key in ((a1, "p_inferior"), (a2, "p_superior")):
        adj = holm([r[key] for r in fam])
        for r, a in zip(fam, adj):
            r["p_holm"] = float(a)
    res["A1_noninferiority_vs_own_reference_rate"] = a1
    res["A2_superiority_vs_rate_rule_at_100Hz"] = a2
    sec = {}
    dets = {}
    for k, p in persons_by.items():
        pa = p.groupby(["unit", "kind"], as_index=False).agg(y=("y", "first"), stat=("stat", "mean"))
        det = p[p.kind == "fam"].assign(det=lambda d: (d.stat > d.tau_fpr7).astype(float)).groupby("unit").det.mean()
        det.index = det.index.astype(str)
        dets[k] = det
        fp = float(p[p.kind != "fam"].assign(fp=lambda d: (d.stat > d.tau_fpr7).astype(float)).fp.mean())
        sec[f"{k[0]}@{k[1]}"] = {"person_auc": fast_auc(pa.y, pa.stat), "detection_fpr7": float(det.mean()), "realised_fpr": fp}
    ref = dets[("rate_rule", 100)]
    for k, d in dets.items():
        if k[0] == "rate_rule":
            continue
        idx = sorted(set(d.index) & set(ref.index))
        dv, rv = d.loc[idx].to_numpy(), ref.loc[idx].to_numpy()
        bs = [dv[q].mean() - rv[q].mean() for q in (rng.integers(0, len(idx), len(idx)) for _ in range(n_boot))]
        sec[f"{k[0]}@{k[1]}"]["detection_delta_vs_rate_rule"] = float(dv.mean() - rv.mean())
        sec[f"{k[0]}@{k[1]}"]["detection_delta_vs_rate_rule_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res["secondary_descriptive"] = sec
    res["n_units"] = n
    return res


# ============================================================================= B. data-driven adaptive fixation threshold (webcam noise)
NH_START, NH_K_PEAK, NH_K_ONSET, NH_TOL, NH_MAX = 100.0, 6.0, 3.0, 1.0, 1000.0
WEBCAM_LEVELS = (0.0, 1.0, 1.5, 2.0, 3.0, 4.0)
WEBCAM_DRAWS = (1, 2, 3)
WEBCAM_CACHE = RESULTS / "phase3" / "webcam_noise" / "cache_30hz"


def sample_velocities(xn, yn, fs=30):
    """All finite sample velocities (deg/s) of one participant, from its own gaze only: central difference over +-1 sample (the
    pipeline's differentiator at 30 Hz) on the pixel coordinates, no gap interpolation, NaN where a neighbour is missing."""
    from edd.delmas_raw import PIX_PER_DEG
    dx = (xn[:, 2:] - xn[:, :-2]) / PIX_PER_DEG
    dy = (yn[:, 2:] - yn[:, :-2]) / PIX_PER_DEG
    v = np.hypot(dx, dy) / (2.0 / fs)
    return v[np.isfinite(v)]


def nystrom_holmqvist_threshold(vel):
    """Velocity threshold of one participant after Nystrom & Holmqvist (2010, Behav Res Methods 42:188-204), as described in the
    saccadr package documentation (alexander-pastukhov.r-universe.dev/saccadr, 'Implemented Methods'; the paper itself was not
    accessible to me): start the peak threshold PT at 100 deg/s; repeat PT = mean + 6 SD of the velocities below PT until the
    change is below 1 deg/s; the saccade onset/offset threshold is then mean + 3 SD of the velocities below the converged PT.
    That onset threshold is used as the I-VT fixation threshold. No labels, no noise level, no floor or ceiling except the
    physiological cap of 1000 deg/s."""
    v = np.asarray(vel, float)
    pt = NH_START
    for _ in range(200):
        sub = v[v < pt]
        if len(sub) < 10:
            return 30.0, pt
        new = sub.mean() + NH_K_PEAK * sub.std()
        done = abs(new - pt) < NH_TOL
        pt = new
        if done:
            break
    sub = v[v < pt]
    return float(min(sub.mean() + NH_K_ONSET * sub.std(), NH_MAX)), float(pt)


EK_LAMBDA = 6.0


def velocity_components(xn, yn, fs=30):
    from edd.delmas_raw import PIX_PER_DEG
    vx = (xn[:, 2:] - xn[:, :-2]) / PIX_PER_DEG / (2.0 / fs)
    vy = (yn[:, 2:] - yn[:, :-2]) / PIX_PER_DEG / (2.0 / fs)
    ok = np.isfinite(vx) & np.isfinite(vy)
    return vx[ok], vy[ok]


def engbert_kliegl_threshold(vx, vy):
    """PRIMARY adaptive rule. Median-based noise estimate of Engbert & Kliegl (2003) as described in the saccadr documentation
    ('Implemented Methods'; the paper itself was not read): per axis SD = sqrt(median(v^2) - median(v)^2) (mean-based
    if that is not positive), threshold = 6 SDs (their default velocity threshold). Here the two axes are combined to one scalar
    speed threshold 6 x sqrt((SDx^2 + SDy^2) / 2) for the I-VT classifier (the elliptic two-axis criterion is not used).
    The result is floored at the pipeline's standard 30 deg/s (EyeLink parser / I-VT value), so clean data keeps the standard
    threshold and noise can only raise it (decided after seeing label-free thresholds: the unfloored median rule gives ~8 deg/s on
    clean 30 Hz data). Per participant, from its own samples only; no labels, no noise level."""
    def sd(v):
        a = np.median(v ** 2) - np.median(v) ** 2
        return float(np.sqrt(a)) if a > 1e-12 else float(np.sqrt(max(np.mean(v ** 2) - np.mean(v) ** 2, 1e-12)))
    sx, sy = sd(vx), sd(vy)
    from edd.delmas_raw import VEL_THR
    return float(max(VEL_THR, EK_LAMBDA * np.sqrt((sx ** 2 + sy ** 2) / 2.0)))


def webcam_participants():
    """{participant: group} from the Delmas 64-trial book (as the webcam-noise experiment)."""
    from edd.delmas import load_trials
    b = load_trials(ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx")
    return {int(p): int(g) for p, g in b.groupby("participant").group.first().items()}


def webcam_adaptive_features(level, draw, rule="ek"):
    """Noisy 30 Hz gaze (the webcam-noise experiment's cache, noise model, seeds and draws, unchanged) -> per-participant data-driven
    threshold -> vectorised trial features -> per-image rows. Returns (feature table, threshold table)."""
    from edd import delmas_fast as dfast, webcam_noise as W
    from edd.delmas_raw import TRIAL_FEATURES
    rows, thr_rows = [], []
    for pid in webcam_participants():
        z = np.load(WEBCAM_CACHE / f"p{pid:02d}.npz")
        xn, yn, pn, info = W.apply_noise(z["x30"], z["y30"], z["p30"], level, draw, pid)
        if rule == "ek":
            thr, pt = engbert_kliegl_threshold(*velocity_components(xn, yn)), float("nan")
        else:
            thr, pt = nystrom_holmqvist_threshold(sample_velocities(xn, yn))
        old = dfast.VEL_THR
        dfast.VEL_THR = thr                                   # module attribute read inside ivt_batch; restored below
        try:
            F = dfast.trial_features_batch(xn, yn, pn, 30, with_msacc=False)
        finally:
            dfast.VEL_THR = old
        rows += delmas_image_rows(pid, F, z["images"])
        thr_rows.append({"rule": rule, "level": level, "draw": draw, "participant": pid, "threshold": thr, "peak_threshold": pt,
                         "offset_deg": info["offset_deg"]})
    return pd.DataFrame(rows, columns=["participant", "set", "image"] + TRIAL_FEATURES), pd.DataFrame(thr_rows)


def webcam_level_tests(rows_by, n_boot, seed, margin=MARGIN_DELMAS, baseline_tag="P4_svm_1000"):
    """rows_by: {(level, draw): {set_key: oof rows}}. AUC of a level = mean over its draws of the image-level AUC pooled within each set
    (repeat-averaged scores), mean over the six sets; reference = clean P4 SVM 1000 Hz (stored). Paired participant bootstrap
    within each set, the same resampled participants for the reference and every level; H0: delta <= -margin; Holm over levels."""
    base = pd.read_csv(RESULTS / "phase2" / "delmas_raw" / "oof_scores.csv.gz")
    base = base[base.tag == baseline_tag]
    levels = sorted({k[0] for k in rows_by})
    cols = {k: f"L{k[0]}_d{k[1]}" for k in rows_by}
    per = {}
    for sk in next(iter(rows_by.values())):
        m = None
        for k, rs in rows_by.items():
            u = rs[sk].assign(participant=rs[sk].unit.astype(int)).groupby(["participant", "image"], as_index=False).agg(
                y=("y", "first"), score=("score", "mean")).rename(columns={"score": cols[k]})
            m = u if m is None else m.merge(u.drop(columns="y"), on=["participant", "image"])
        b = base[base.set == sk].groupby(["participant", "image"], as_index=False).agg(ref=("score", "mean"))
        m = m.merge(b, on=["participant", "image"])
        per[sk] = [(d.y.to_numpy(), d["ref"].to_numpy(), {k: d[c].to_numpy() for k, c in cols.items()}) for _, d in m.groupby("participant")]

    def auc_sets(idx):
        out = {"ref": [], **{L: [] for L in levels}}
        for sk, v in per.items():
            sel = v if idx is None else [v[i] for i in idx[sk]]
            y = np.concatenate([x[0] for x in sel])
            if len(np.unique(y)) < 2:
                continue
            out["ref"].append(fast_auc(y, np.concatenate([x[1] for x in sel])))
            for L in levels:
                out[L].append(np.mean([fast_auc(y, np.concatenate([x[2][k] for x in sel])) for k in rows_by if k[0] == L]))
        return {k: float(np.mean(v)) for k, v in out.items()}
    est = auc_sets(None)
    rng = np.random.default_rng(seed)
    bs = {k: [] for k in est}
    for _ in range(n_boot):
        bb = auc_sets({sk: rng.integers(0, len(v), len(v)) for sk, v in per.items()})
        for k in est:
            bs[k].append(bb[k])
    bs = {k: np.array(v) for k, v in bs.items()}
    res = []
    for L in levels:
        d = bs[L] - bs["ref"]
        res.append({"level_deg": L, "auc_level": est[L], "auc_reference": est["ref"], "delta": est[L] - est["ref"],
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "lower_one_sided_95": float(np.percentile(d, 5)), "p_inferior": p_inferior(d, margin)})
    adj = holm([r["p_inferior"] for r in res])
    for r, a in zip(res, adj):
        r["p_holm"] = float(a)
        r["non_inferior_holm"] = bool(a < 0.05)
    return res
