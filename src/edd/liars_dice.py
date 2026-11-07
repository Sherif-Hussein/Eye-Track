"""Liar's Dice (Foucher & Huckauf, PsyArXiv 2026): bid-level features from the released fixation/blink
tables, online within-person baselines, the paper's effect turned into a classifier, a participant-disjoint
CV runner that keeps row identity, and trial- plus per-participant metrics.

LABEL NOTE: in the released Rda files the `BidHonesty` coding is inverted with respect to the paper.
Contested bids labelled "Lie" are won by the bidder in 1057/1059 (PvC) and 936/998 (PvP) cases, which the
game rules only allow for truthful bids, and the paper's descriptive statistics are reproduced only after
swapping the labels. We therefore define deceptive (y = 1) as released `BidHonesty == "Truth"`.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.metrics import roc_auc_score

from .harness import scores_of
from .metrics import binary_metrics, summarise_repeats

KEYS = ["Subject", "Session", "Round", "Bid"]
EXPS = {"pvc": "PlayerComputer", "pvp": "PlayerPlayer"}
DECEPTIVE_RELEASED_CODE = "Truth"   # see module docstring

# Ocular, time-normalised (no raw counts: those mostly re-encode bid time).
OC_COMMON = ["fix_rate", "fix_dur_mean", "fix_dur_median", "fix_dur_sd", "fix_dur_max", "fix_time_frac",
             "fix_dur_first", "blink_rate", "blink_dur_mean", "any_blink"]
OC_AOI = ["rate_face", "rate_dice", "rate_out", "dur_face", "dur_dice", "dur_out", "dwell_face", "dwell_dice",
          "dwell_out", "share_face", "switch_rate", "first_face", "first_dice"]
RT = ["log_bid_time"]
CONTEXT = ["bid_pos", "own_bid_idx", "opened_round", "session", "round", "prev_round_won", "session_win_rate"]
# The paper's measures and the direction of their reported lie effects (signs fixed a priori from the paper).
PAPER_OCULAR = {"fix_rate": -1, "fix_dur_mean": +1, "blink_rate": -1, "blink_dur_mean": -1}
PAPER_RT = {"log_bid_time": +1}
PAPER_AOI = {"rate_face": -1, "dur_dice": +1, "dur_out": +1}


def _read_rda(path):
    import pyreadr
    df = next(iter(pyreadr.read_r(str(path)).values()))
    for c in KEYS:
        df[c] = df[c].astype(float).astype(int)
    return df


def _won(series):
    s = series.astype(str).str.upper()
    return s.isin(["PLAYER", "TRUE"]).astype(float)


def build_bids(data_dir):
    """One row per bid (both roles) for both experiments, with ocular, RT and context features."""
    frames = []
    for exp, stem in EXPS.items():
        fx, bl = _read_rda(f"{data_dir}/{stem}_fixations.Rda"), _read_rda(f"{data_dir}/{stem}_blinks.Rda")
        has_aoi = "AOI" in fx.columns
        meta_cols = ["Role", "BidHonesty", "ChalDecision", "BidTime", "Winner"]
        meta = pd.concat([fx[KEYS + meta_cols].drop_duplicates(KEYS),
                          bl[KEYS + meta_cols].drop_duplicates(KEYS)]).drop_duplicates(KEYS, keep="first")
        fgroups = {k: g for k, g in fx.groupby(KEYS, sort=False)}
        bgroups = {k: g["BlinkDuration"].to_numpy(float) for k, g in bl.groupby(KEYS, sort=False)}
        rows = []
        for r in meta.itertuples(index=False):
            k = (r.Subject, r.Session, r.Round, r.Bid)
            g = fgroups.get(k)
            d = g["FixDuration"].to_numpy(float) if g is not None else np.empty(0)
            f = bid_features_arrays(d, bgroups.get(k, np.empty(0)), float(r.BidTime),
                                    g["AOI"].astype(str).to_numpy() if (has_aoi and g is not None) else
                                    (np.empty(0, object) if has_aoi else None))
            f.update(dict(zip(KEYS, k)))
            f.update({"Role": r.Role, "BidHonesty": r.BidHonesty, "ChalDecision": r.ChalDecision,
                      "bid_time": float(r.BidTime), "round_won": float(_won(pd.Series([r.Winner])).iloc[0])})
            rows.append(f)
        df = pd.DataFrame(rows)
        df["exp"] = exp
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["y"] = (df["BidHonesty"] == DECEPTIVE_RELEASED_CODE).astype(int)
    df["pid"] = df["exp"] + "_" + df["Subject"].astype(str)
    # PvP players were seated in dyads (1,2), (3,4), ... (verified from the role/bid alignment).
    df["group"] = np.where(df["exp"] == "pvp", "pvp_d" + ((df["Subject"] + 1) // 2).astype(str), df["pid"])
    df["log_bid_time"] = np.log(df["bid_time"])
    df = df.sort_values(["exp", "Subject", "Session", "Round", "Bid"]).reset_index(drop=True)
    df["order"] = df.groupby("pid").cumcount()
    # Context available at bid time (no current-round outcome, no challenge decision, no LastBid).
    df["bid_pos"] = df["Bid"].astype(float)
    df["own_bid_idx"] = ((df["Bid"] + 1) // 2).astype(float)
    df["opened_round"] = (df["Bid"] % 2 == 1).astype(float)
    df["session"], df["round"] = df["Session"].astype(float), df["Round"].astype(float)
    rnd = df.groupby(["pid", "Session", "Round"])["round_won"].first().reset_index()
    rnd["prev_round_won"] = rnd.groupby(["pid", "Session"])["round_won"].shift(1)
    cum = rnd.groupby(["pid", "Session"])["round_won"].transform(lambda s: s.shift(1).expanding().mean())
    rnd["session_win_rate"] = cum
    df = df.merge(rnd[["pid", "Session", "Round", "prev_round_won", "session_win_rate"]],
                  on=["pid", "Session", "Round"], how="left")
    return df


def bid_features_arrays(d, b, bid_time_ms, aoi=None):
    """Features of one bid from its fixation durations (file order), their AOIs and its blink durations."""
    t = bid_time_ms / 1000.0
    n = len(d)
    out = {"n_fix": n, "fix_rate": n / t, "fix_time_frac": d.sum() / bid_time_ms if n else 0.0,
           "fix_dur_mean": d.mean() if n else np.nan, "fix_dur_median": float(np.median(d)) if n else np.nan,
           "fix_dur_sd": d.std() if n > 1 else np.nan, "fix_dur_max": d.max() if n else np.nan,
           "fix_dur_first": d[0] if n else np.nan,
           "n_blink": len(b), "blink_rate": len(b) / t, "blink_dur_mean": b.mean() if len(b) else np.nan,
           "any_blink": float(len(b) > 0)}
    if aoi is not None:
        for a, name in (("Face", "face"), ("Dice", "dice"), ("Out", "out")):
            m = aoi == a
            out[f"rate_{name}"] = m.sum() / t
            out[f"dur_{name}"] = d[m].mean() if m.any() else np.nan
            out[f"dwell_{name}"] = d[m].sum() / bid_time_ms
        out["share_face"] = (aoi == "Face").mean() if n else np.nan
        out["switch_rate"] = (np.sum(aoi[1:] != aoi[:-1]) / t) if n > 1 else 0.0
        out["first_face"] = float(n > 0 and aoi[0] == "Face")
        out["first_dice"] = float(n > 0 and aoi[0] == "Dice")
    return out


def add_past_stats(df, feats, roles=("Bidder",)):
    """Per participant, running mean and count of each feature over that participant's *earlier* events
    of the given roles (strictly past, label-free): the online within-person baseline."""
    df = df.copy()
    src = df["Role"].isin(roles)
    for f in feats:
        v = df[f].where(src)
        g = v.groupby(df["pid"])
        s = g.transform(lambda x: x.fillna(0).cumsum().shift(1).fillna(0))
        n = v.notna().astype(float).groupby(df["pid"]).transform(lambda x: x.cumsum().shift(1).fillna(0))
        df[f"pm__{f}"] = np.where(n > 0, s / n.replace(0, np.nan), np.nan)
        df[f"pn__{f}"] = n
    return df


def person_offline_center(df, feats):
    """Transductive reference (not online): subtract the participant's mean over ALL their bidder bids."""
    out = df.copy()
    for f in feats:
        out[f"pc__{f}"] = df[f] - df.groupby("pid")[f].transform("mean")
    return out


class OnlineBaseline(BaseEstimator, TransformerMixin):
    """Input columns: [raw (p), past_mean (p), past_n (p)]. Output: raw - baseline, where the baseline
    shrinks the participant's running past mean towards the training-population mean with `k` pseudo-bids.
    With keep_raw, the raw features are appended."""

    def __init__(self, k=5.0, keep_raw=False):
        self.k, self.keep_raw = k, keep_raw

    def fit(self, X, y=None):
        p = X.shape[1] // 3
        self.m0_ = np.nanmean(X[:, :p], axis=0)
        return self

    def transform(self, X):
        p = X.shape[1] // 3
        raw, pm, pn = X[:, :p], np.nan_to_num(X[:, p:2 * p]), X[:, 2 * p:]
        base = (pn * pm + self.k * self.m0_) / (pn + self.k)
        dev = raw - base
        return np.hstack([dev, raw]) if self.keep_raw else dev


def online_cols(feats):
    return list(feats) + [f"pm__{f}" for f in feats] + [f"pn__{f}" for f in feats]


class SignedZScore(BaseEstimator, ClassifierMixin):
    """The paper's reported effects as a classifier: mean of sign_k * z_k over the paper's measures, signs
    fixed from the paper, z from training participants; threshold at the midpoint of the training class means.
    Missing values contribute 0. Optional online person baseline (columns as in OnlineBaseline)."""

    def __init__(self, signs=(), online=False, k=5.0):
        self.signs, self.online, self.k = signs, online, k

    def _prep(self, X):
        return self.ob_.transform(X) if self.online else X

    def fit(self, X, y):
        y = np.asarray(y)
        if self.online:
            self.ob_ = OnlineBaseline(self.k).fit(X)
        Z = self._prep(X)
        self.mu_, self.sd_ = np.nanmean(Z, axis=0), np.nanstd(Z, axis=0) + 1e-9
        s = self._raw(Z)
        self.thr_ = (s[y == 1].mean() + s[y == 0].mean()) / 2
        self.classes_ = np.array([0, 1])
        return self

    def _raw(self, Z):
        z = np.nan_to_num((Z - self.mu_) / self.sd_)
        return z @ np.asarray(self.signs, float) / len(self.signs)

    def decision_function(self, X):
        return self._raw(self._prep(X)) - self.thr_


def _fold_task(X, y, g, te, r, f, fit_fn):
    m = fit_fn(X[~te], y[~te], g[~te])
    est = getattr(m, "best_estimator_", m)
    return pd.DataFrame({"row": np.flatnonzero(te), "repeat": r, "fold": f, "score": scores_of(est, X[te]),
                         "params": str(getattr(m, "best_params_", ""))})


def run_cv(df, cols, fit_fn, folds_per_repeat, group_col="group", n_jobs=2):
    """Outer participant-disjoint CV that keeps row indices. `folds_per_repeat[r]` lists test pids per fold.
    `fit_fn(X, y, groups)` returns a fitted model (inner tuning must use `groups`)."""
    from joblib import Parallel, delayed
    X, y = df[cols].to_numpy(float), df["y"].to_numpy(int)
    g, pid = df[group_col].astype(str).to_numpy(), df["pid"].to_numpy()
    tasks = [(np.isin(pid, t), r, f) for r, folds in enumerate(folds_per_repeat) for f, t in enumerate(folds)]
    out = Parallel(n_jobs=n_jobs)(delayed(_fold_task)(X, y, g, te, r, f, fit_fn) for te, r, f in tasks)
    oof = pd.concat(out, ignore_index=True)
    oof["y"] = y[oof["row"]]
    oof["pid"] = pid[oof["row"]]
    return oof


def per_participant(oof):
    """Within-participant discrimination from out-of-fold scores averaged over repeats."""
    avg = oof.groupby("row").agg(pid=("pid", "first"), y=("y", "first"), score=("score", "mean"))
    rows = []
    for p, t in avg.groupby("pid"):
        both = t.y.nunique() == 2
        auc = roc_auc_score(t.y, t.score) if both else np.nan
        pval = mannwhitneyu(t.score[t.y == 1], t.score[t.y == 0], alternative="greater").pvalue if both else np.nan
        m = binary_metrics(t.y, t.score)
        rows.append({"pid": p, "n": len(t), "n_lie": int(t.y.sum()), "auc": auc, "p_mw": pval,
                     "bacc": m["balanced_accuracy"], "hter": m["hter"], "eer": m["eer"] if both else np.nan})
    return pd.DataFrame(rows)


def evaluate(oof):
    trial_rep = [binary_metrics(t.y, t.score) for _, t in oof.groupby("repeat")]
    pp = per_participant(oof)
    v = pp.dropna(subset=["auc"])
    return {
        "trial": summarise_repeats(trial_rep),
        "per_participant": {
            "n_participants": int(len(v)),
            "auc_mean": float(v.auc.mean()), "auc_median": float(v.auc.median()), "auc_sd": float(v.auc.std()),
            "frac_auc_gt_0.5": float((v.auc > 0.5).mean()),
            "detected_alpha05": int((v.p_mw < 0.05).sum()), "detected_frac": float((v.p_mw < 0.05).mean()),
            "bacc_mean": float(v.bacc.mean()), "hter_mean": float(v.hter.mean()), "eer_mean": float(v.eer.mean()),
        },
        "n_repeats": len(trial_rep),
    }


def cluster_bootstrap(y, score, pid, n_boot=2000, seed=0, alpha=0.05):
    """Participant (cluster) bootstrap of pooled trial-level metrics: resample participants with
    replacement, keep all their trials. Honest for trial-level data with repeated measures."""
    y, score, pid = np.asarray(y), np.asarray(score), np.asarray(pid)
    units = np.unique(pid)
    idx_of = {u: np.flatnonzero(pid == u) for u in units}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[u] for u in rng.choice(units, len(units))])
        draws.append(binary_metrics(y[idx], score[idx]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.nanpercentile([d[k] for d in draws], lo)), float(np.nanpercentile([d[k] for d in draws], hi))]
            for k in draws[0] if k != "n"}


def paired_cluster_bootstrap_diff(y, s_a, s_b, pid, n_boot=2000, seed=0):
    """CI of AUC(a) - AUC(b) on identical rows, resampling participants."""
    y, s_a, s_b, pid = map(np.asarray, (y, s_a, s_b, pid))
    units = np.unique(pid)
    idx_of = {u: np.flatnonzero(pid == u) for u in units}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[u] for u in rng.choice(units, len(units))])
        d.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    d = np.asarray(d)
    return {"diff": float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
