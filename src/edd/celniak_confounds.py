"""Confound-control building blocks for the celniak track (all fitted on training participants only).

* CovariateResidualizer: last column is a covariate (e.g. log validation error). Each feature is regressed on it
  by OLS on the training participants and the fitted linear part is removed (the feature mean is kept).
* SelectPlusCovariate: ANOVA-F selection on the feature columns only, the covariate (last column) is always kept.
* QualityFilteredSearch: chooses the minimum recording quality for TRAINING participants by participant-grouped
  inner CV (inner validation folds keep every participant, so the score reflects generalisation to all
  recordings), then refits on the retained training participants.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


class CovariateResidualizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        c = X[:, -1].astype(float)
        self.c_med_ = float(np.nanmedian(c))
        c = np.where(np.isnan(c), self.c_med_, c)
        self.c_mean_ = float(c.mean())
        F = X[:, :-1]
        cc = c - self.c_mean_
        denom = float((cc ** 2).sum()) or 1.0
        self.slope_ = np.array([np.nansum(np.where(np.isnan(F[:, j]), 0.0, F[:, j] - np.nanmean(F[:, j])) * cc) / denom
                                for j in range(F.shape[1])])
        return self

    def transform(self, X):
        c = X[:, -1].astype(float)
        c = np.where(np.isnan(c), self.c_med_, c)
        return X[:, :-1] - np.outer(c - self.c_mean_, self.slope_)


class SelectPlusCovariate(BaseEstimator, TransformerMixin):
    """Impute + scale all columns; ANOVA-F top-k on the feature columns; append the covariate (last column)."""

    def __init__(self, k="all"):
        self.k = k

    def fit(self, X, y):
        self.imp_ = SimpleImputer(strategy="median").fit(X)
        self.sc_ = StandardScaler().fit(self.imp_.transform(X))
        Z = self.sc_.transform(self.imp_.transform(X))
        n = Z.shape[1] - 1
        self.sel_ = SelectKBest(f_classif, k="all" if self.k == "all" or self.k >= n else self.k).fit(Z[:, :-1], y)
        return self

    def transform(self, X):
        Z = self.sc_.transform(self.imp_.transform(X))
        return np.hstack([self.sel_.transform(Z[:, :-1]), Z[:, -1:]])

    def get_support_features(self):
        return self.sel_.get_support()


def filtered_inner_splits(y, g, drop, n_splits=3, seed=0):
    """Participant-grouped inner folds; training indices of low-quality participants (`drop`) are removed,
    validation indices are kept in full."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for tr, va in cv.split(np.zeros(len(y)), y, g):
        tr = tr[~drop[tr]]
        out.append((tr, va))
    return out


class _Fitted:
    def __init__(self, est, params):
        self.best_estimator_, self.best_params_ = est, params

    def decision_function(self, X):
        return self.best_estimator_.decision_function(X)


def quality_filtered_fit(pipeline, grid, quality_col, thresholds, n_splits=3, min_per_class=5):
    """fit(X, y, g) for harness.run_outer. `quality_col` indexes the recording-quality column of X (higher is
    better). For each threshold the training set keeps participants with quality >= threshold; the threshold with
    the best inner-CV AUC (ties go to the smaller threshold) is used for the final refit."""
    def fit(X, y, g):
        q = X[:, quality_col]
        best = None
        for thr in sorted(thresholds):
            keep = q >= thr
            # a threshold is admissible only if enough participants of both classes remain
            if min(len(set(g[keep & (y == 1)])), len(set(g[keep & (y == 0)]))) < min_per_class:
                continue
            splits = filtered_inner_splits(y, g, ~keep, n_splits)
            if any(len(set(y[tr])) < 2 for tr, _ in splits):
                continue
            gs = GridSearchCV(clone(pipeline), grid, cv=splits, scoring="roc_auc", n_jobs=1, refit=False).fit(X, y)
            if best is None or gs.best_score_ > best[0] + 1e-12:
                best = (gs.best_score_, thr, gs.best_params_)
        score, thr, params = best
        keep = q >= thr
        est = clone(pipeline).set_params(**params).fit(X[keep], y[keep])
        return _Fitted(est, {"thr": float(thr), "n_train_participants": int(len(set(g[keep]))),
                             "inner_auc": float(score), **{k: (v if isinstance(v, (int, float, str)) else str(v))
                                                           for k, v in params.items()}})
    return fit
