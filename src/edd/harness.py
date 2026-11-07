"""Participant-grouped evaluation: nested tuning inside frozen outer folds, trial-to-participant
aggregation, leakage variants for the audit, and a participant-level permutation test."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold

from .metrics import binary_metrics, bootstrap_ci, summarise_repeats


def scores_of(model, X):
    """Signed score with 0 as the decision threshold, for any sklearn classifier."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1] - 0.5


def make_search(pipeline, grid, inner_cv, n_iter=None, seed=0, scoring="roc_auc", n_jobs=1):
    """Searches default to one job: parallelism is applied over outer folds instead, because
    dispatching thousands of millisecond-long inner fits costs more than the fits themselves."""
    if n_iter:
        return RandomizedSearchCV(pipeline, grid, n_iter=n_iter, cv=inner_cv, scoring=scoring,
                                  n_jobs=n_jobs, random_state=seed, refit=True)
    return GridSearchCV(pipeline, grid, cv=inner_cv, scoring=scoring, n_jobs=n_jobs, refit=True)


def _outer_task(r, f, X, y, g, test_units, fit_fn):
    test = np.isin(g, test_units)
    model = fit_fn(X[~test], y[~test], g[~test])
    params = getattr(model, "best_params_", None)
    est = getattr(model, "best_estimator_", model)
    return pd.DataFrame({"repeat": r, "fold": f, "unit": g[test], "y": y[test],
                         "score": scores_of(est, X[test]), "params": str(params)})


def run_outer(df, features, label, unit, folds_per_repeat, fit_fn, n_jobs=-1):
    """Fit `fit_fn(X_train, y_train, groups_train) -> fitted model` on each outer training split and
    score the held-out participants, one parallel task per outer fold. Returns one row per (repeat, trial)."""
    X, y, g = df[features].to_numpy(float), df[label].to_numpy(int), df[unit].astype(str).to_numpy()
    tasks = [(r, f, test_units) for r, folds in enumerate(folds_per_repeat) for f, test_units in enumerate(folds)]
    out = Parallel(n_jobs=n_jobs)(delayed(_outer_task)(r, f, X, y, g, t, fit_fn) for r, f, t in tasks)
    return pd.concat(out, ignore_index=True)


def nested_fit(search_factory):
    """Tune with participant-grouped inner CV on the outer training split only."""
    def fit(X, y, g):
        search = search_factory()
        search.fit(X, y, groups=g)
        return search
    return fit


def fixed_fit(estimator):
    def fit(X, y, g):
        return clone(estimator).fit(X, y)
    return fit


def aggregate(trials, how="mean"):
    """Participant score from out-of-fold trial scores."""
    agg = trials.groupby(["repeat", "unit"]).agg(y=("y", "first"), score=("score", how)).reset_index()
    return agg


def evaluate(trials, n_boot=2000):
    """Trial- and participant-level metrics per repeat (mean/SD) plus a bootstrap CI on the
    repeat-averaged participant scores."""
    parts = aggregate(trials)
    trial_rep = [binary_metrics(t.y, t.score) for _, t in trials.groupby("repeat")]
    part_rep = [binary_metrics(p.y, p.score) for _, p in parts.groupby("repeat")]
    avg = parts.groupby("unit").agg(y=("y", "first"), score=("score", "mean"))
    return {
        "trial": summarise_repeats(trial_rep),
        "participant": summarise_repeats(part_rep),
        "participant_avg_scores": binary_metrics(avg.y, avg.score),
        "participant_ci95": bootstrap_ci(avg.y.to_numpy(), avg.score.to_numpy(), n_boot=n_boot),
        "n_repeats": len(trial_rep),
    }


def random_trial_split(df, features, label, unit, estimator_search, n_splits=5, seed=0):
    """Leakage variant: ignore participants entirely (the common, flawed protocol)."""
    X, y = df[features].to_numpy(float), df[label].to_numpy(int)
    units = df[unit].astype(str).to_numpy()
    out = []
    for f, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)):
        search = estimator_search()
        search.fit(X[tr], y[tr])
        est = getattr(search, "best_estimator_", search)
        out.append(pd.DataFrame({"repeat": 0, "fold": f, "unit": units[te], "y": y[te], "score": scores_of(est, X[te])}))
    return pd.concat(out, ignore_index=True)


def permutation_test(df, features, label, unit, folds, fit_fn, observed_auc, n_perm, seed=0):
    """Shuffle labels between participants (all of a participant's trials move together) and rerun
    the full nested pipeline. p = (1 + #null >= observed) / (1 + n_perm)."""
    rng = np.random.default_rng(seed)
    units = df[unit].astype(str)
    per_unit = df.groupby(units)[label].first()
    null = []
    for _ in range(n_perm):
        perm = pd.Series(rng.permutation(per_unit.to_numpy()), index=per_unit.index)
        d = df.copy()
        d[label] = units.map(perm).to_numpy()
        trials = run_outer(d, features, label, unit, [folds], fit_fn)
        p = aggregate(trials)
        null.append(binary_metrics(p.y, p.score)["auc"])
    null = np.asarray(null)
    return {"n_perm": n_perm, "observed_auc": observed_auc, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + np.sum(null >= observed_auc)) / (1 + n_perm))}


def grouped_inner(n_splits=5, seed=0):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
