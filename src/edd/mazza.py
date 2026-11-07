"""Mazza/Colasanti et al. (2023) RealEye webcam crime-scene dataset (Zenodo 10.5281/zenodo.7194229).

One row per participant (157), 90 aggregate gaze features. Everything here is either a stateless,
row-wise transform (uses only the same participant's own numbers, so it cannot leak across people)
or an sklearn transformer whose state is learned inside the training fold only.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

AOIS = ("left", "central", "right")
TARGETS = (1, 2, 3, 4, 5, 6)
NEUTRALS = (2, 3, 4)
# AOI in which the desk with the exam paper is visible (asterisks in Colasanti et al. 2023, Tables 2-3).
# Target 2 does not show the desk.
DESK = {1: "left", 3: "left", 4: "left", 5: "central", 6: "right"}
GROUP_NAMES = {1: "Guilty Honest", 2: "Guilty Deceptive", 3: "Innocent Honest", 4: "Naive"}

TASKS = {
    # name: (groups included, positive groups, description)
    "guilty_vs_innocent": ((1, 2, 3, 4), (1, 2), "G1+G2 (stole the exam) vs G3+G4"),
    "deceptive_vs_honest": ((1, 2, 3, 4), (2,), "G2 (lied) vs G1+G3+G4"),
    "g2_vs_g3": ((2, 3), (2,), "Guilty Deceptive vs Innocent Honest (both visited the room)"),
    "g2_vs_g1": ((1, 2), (2,), "Guilty Deceptive vs Guilty Honest (published pair)"),
    "g2_vs_g4": ((2, 4), (2,), "Guilty Deceptive vs Naive (published pair)"),
}


def tcol(kind, t, aoi):
    k = "fixations_sum" if kind == "cnt" else "fixations_duration_sum"
    return f"{k}_Target {t} - {aoi}"


def ncol(kind, n, aoi):
    k = "fixations_sum" if kind == "cnt" else "fixations_duration_sum"
    return f"{k}_Neutral_recognized {n} - {aoi}"


def load(path):
    """Comma-separated ASCII; `condition` is 'Group k'."""
    d = pd.read_csv(path)
    d["group"] = d["condition"].str.extract(r"(\d)").astype(int)
    d["unit"] = d["g_code"].astype(str)
    return d


def raw_columns(d):
    return [c for c in d.columns if c not in ("g_code", "condition", "group", "unit")]


PUB36 = [tcol(k, t, a) for k in ("cnt", "dur") for t in TARGETS for a in AOIS]
# Raw-column blocks for the ablation. Neutral images carry no crime content, so a neutral-only model is
# the negative control: signal there means group differences unrelated to the crime scene.
BLOCKS = {
    "neutral": lambda c: "Neutral_recognized" in c,
    "target_overall": lambda c: "Target" in c and c.endswith("overall") and c.startswith("fixations_") and
                                ("fixations_sum" in c or "fixations_duration_sum" in c),
    "scanpath": lambda c: "euclidean" in c or "time_distance" in c,
    "no_neutral": lambda c: "Neutral_recognized" not in c,
}


class Featurizer(BaseEstimator, TransformerMixin):
    """Stateless row-wise feature engineering from the 90 raw columns.

    sets: any of 'raw', 'pub36', 'diffshare', 'logratio', 'z', 'crime'.
    """

    def __init__(self, raw_cols=None, sets=("raw",)):
        self.raw_cols = raw_cols
        self.sets = sets

    def fit(self, X, y=None):
        self._idx = {c: i for i, c in enumerate(self.raw_cols)}
        self.names_ = None
        return self

    def _g(self, X, c):
        return X[:, self._idx[c]].astype(float)

    def _blocks(self, X):
        """Per kind: target AOI array (n, 6, 3), neutral AOI array (n, 3, 3)."""
        out = {}
        for k in ("cnt", "dur"):
            T = np.stack([np.stack([self._g(X, tcol(k, t, a)) for a in AOIS], 1) for t in TARGETS], 1)
            N = np.stack([np.stack([self._g(X, ncol(k, n, a)) for a in AOIS], 1) for n in NEUTRALS], 1)
            out[k] = (T, N)
        return out

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        B = self._blocks(X)
        feats, names = [], []

        def add(arr, nm):
            feats.append(np.asarray(arr, float).reshape(len(X), -1))
            names.extend(nm)

        eps = {"cnt": 1.0, "dur": 100.0}
        for s in self.sets:
            if s == "raw":
                add(X, list(self.raw_cols))
            elif s == "pub36":
                add(np.column_stack([self._g(X, c) for c in PUB36]), PUB36)
            elif s in BLOCKS:
                cols = [c for c in self.raw_cols if BLOCKS[s](c)]
                add(np.column_stack([self._g(X, c) for c in cols]), cols)
            elif s in ("diffshare", "logratio", "z"):
                for k in ("cnt", "dur"):
                    T, N = B[k]
                    if s == "diffshare":
                        # share of the image's fixations in each third, minus the same person's share on neutral images
                        ts = T / np.maximum(T.sum(2, keepdims=True), eps[k])
                        ns = N.sum(1) / np.maximum(N.sum((1, 2))[:, None], eps[k])
                        v = ts - ns[:, None, :]
                    elif s == "logratio":
                        v = np.log((T + eps[k]) / (N.mean(1)[:, None, :] + eps[k]))
                    else:
                        mu = N.mean(1)
                        resid = N - mu[:, None, :]
                        sd = np.sqrt((resid ** 2).sum((1, 2)) / (3 * (3 - 1)))
                        v = (T - mu[:, None, :]) / (sd + eps[k])[:, None, None]
                    add(v, [f"{s}_{k}_T{t}_{a}" for t in TARGETS for a in AOIS])
            elif s == "crime":
                for k in ("cnt", "dur"):
                    T, N = B[k]
                    ts = T / np.maximum(T.sum(2, keepdims=True), eps[k])            # (n,6,3)
                    ns = N.sum(1) / np.maximum(N.sum((1, 2))[:, None], eps[k])      # (n,3)
                    ai = {a: i for i, a in enumerate(AOIS)}
                    desk = np.column_stack([ts[:, t - 1, ai[a]] for t, a in DESK.items()])
                    desk_ref = np.column_stack([ns[:, ai[a]] for a in DESK.values()])
                    cen = ts[:, :, 1]
                    add(desk, [f"crime_{k}_deskshare_T{t}" for t in DESK])
                    add(cen, [f"crime_{k}_centralshare_T{t}" for t in TARGETS])
                    add(np.column_stack([desk.mean(1), desk.std(1), cen.mean(1), cen.std(1),
                                         (desk - desk_ref).mean(1), (cen - ns[:, [1]]).mean(1),
                                         desk[:, :3].mean(1) - cen[:, [0, 2, 3]].mean(1)]),
                        [f"crime_{k}_{n}" for n in ("desk_mean", "desk_sd", "central_mean", "central_sd",
                                                     "desk_vs_neutral", "central_vs_neutral",
                                                     "avoid_T134")])
            else:
                raise ValueError(s)
        self.names_ = names
        return np.column_stack(feats)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.names_)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise each column at training-fold quantiles (outlier handling inside folds)."""

    def __init__(self, lo=0.01, hi=0.99):
        self.lo = lo
        self.hi = hi

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_ = np.nanquantile(X, self.lo, axis=0)
        self.hi_ = np.nanquantile(X, self.hi, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


class StabilitySelector(BaseEstimator, TransformerMixin):
    """Stability selection (Meinshausen & Buehlmann): L1-logistic on n_boot half-samples of the
    training fold; keep features selected in >= threshold of them (at least `min_keep`)."""

    def __init__(self, n_boot=30, C=0.1, threshold=0.6, min_keep=3, seed=0):
        self.n_boot = n_boot
        self.C = C
        self.threshold = threshold
        self.min_keep = min_keep
        self.seed = seed

    def fit(self, X, y):
        X = StandardScaler().fit_transform(np.asarray(X, float))
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        freq = np.zeros(X.shape[1])
        for _ in range(self.n_boot):
            idx = np.concatenate([rng.choice(np.flatnonzero(y == c), max(1, (y == c).sum() // 2), replace=False)
                                  for c in np.unique(y)])
            m = LogisticRegression(penalty="l1", C=self.C, solver="liblinear", class_weight="balanced")
            m.fit(X[idx], y[idx])
            freq += (np.abs(m.coef_).max(0) > 1e-8)
        self.freq_ = freq / self.n_boot
        keep = self.freq_ >= self.threshold
        if keep.sum() < self.min_keep:
            keep = np.zeros_like(keep)
            keep[np.argsort(-self.freq_)[: self.min_keep]] = True
        self.support_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.support_]


def fast_auc(y, score):
    """Rank-based AUC, equal to roc_auc_score (ties included) and about 100x faster, for bootstraps."""
    y = np.asarray(y).astype(bool)
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def signed_score(est, X):
    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        if s.ndim == 1:
            return s
    return est.predict_proba(X)[:, 1] - 0.5


def _outer(r, f, X, y, g, test_units, factory, calibrate):
    test = np.isin(g, test_units)
    search = factory(r * 1000 + f)
    search.fit(X[~test], y[~test], groups=g[~test])
    est = getattr(search, "best_estimator_", search)
    out = {"repeat": r, "fold": f, "unit": g[test], "y": y[test], "score": signed_score(est, X[test]),
           "params": str(getattr(search, "best_params_", None))}
    if hasattr(est, "predict_proba"):
        out["prob"] = est.predict_proba(X[test])[:, 1]
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold
        cal = CalibratedClassifierCV(clone(est), method="sigmoid",
                                     cv=StratifiedKFold(5, shuffle=True, random_state=r * 1000 + f))
        cal.fit(X[~test], y[~test])
        out["prob_cal"] = cal.predict_proba(X[test])[:, 1]
    return pd.DataFrame(out)


def run_binary(X, y, g, folds_per_repeat, factory, n_jobs=2, calibrate=False):
    """`factory(seed)` returns an unfitted search (or estimator accepting groups=). One parallel task per
    outer fold. Returns one row per (repeat, participant)."""
    tasks = [(r, f, t) for r, folds in enumerate(folds_per_repeat) for f, t in enumerate(folds)]
    out = Parallel(n_jobs=n_jobs)(delayed(_outer)(r, f, X, y, g, t, factory, calibrate) for r, f, t in tasks)
    return pd.concat(out, ignore_index=True)


def _outer_multi(r, f, X, y, g, test_units, factory):
    test = np.isin(g, test_units)
    search = factory(r * 1000 + f)
    search.fit(X[~test], y[~test], groups=g[~test])
    est = getattr(search, "best_estimator_", search)
    P = est.predict_proba(X[test])
    df = pd.DataFrame(P, columns=[f"p{c}" for c in est.classes_])
    df.insert(0, "y", y[test])
    df.insert(0, "unit", g[test])
    df.insert(0, "fold", f)
    df.insert(0, "repeat", r)
    return df


def run_multiclass(X, y, g, folds_per_repeat, factory, n_jobs=2):
    tasks = [(r, f, t) for r, folds in enumerate(folds_per_repeat) for f, t in enumerate(folds)]
    out = Parallel(n_jobs=n_jobs)(delayed(_outer_multi)(r, f, X, y, g, t, factory) for r, f, t in tasks)
    return pd.concat(out, ignore_index=True)


def ece(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def selective(y, score, coverages=(1.0, 0.8, 0.6)):
    """Risk-coverage: keep the `c` fraction of participants with the largest |score| (most confident)."""
    y, score = np.asarray(y), np.asarray(score)
    order = np.argsort(-np.abs(score))
    out = {}
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        idx = order[:k]
        pred = (score[idx] >= 0).astype(int)
        acc = float((pred == y[idx]).mean())
        tpr = (pred[y[idx] == 1] == 1).mean() if (y[idx] == 1).any() else np.nan
        tnr = (pred[y[idx] == 0] == 0).mean() if (y[idx] == 0).any() else np.nan
        out[f"{int(c * 100)}"] = {"accuracy": acc, "balanced_accuracy": float(np.nanmean([tpr, tnr])), "n": int(k)}
    return out
