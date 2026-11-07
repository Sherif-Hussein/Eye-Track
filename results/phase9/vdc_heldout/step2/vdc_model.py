"""Item model + participant CIT statistic + threshold, reusing the project's unified pipeline (src/edd/unified.py):
tune_fit (L2 logistic regression, balanced weights, C from 9 log-spaced values by mean AUC over <= 5 inner
participant-disjoint folds), decide (decision function), thresholds (7% rule), zscore_within (within-participant z).
CIT statistic = unified._cit_z's known-vs-unknown part: (mean score of the known item - mean score of the unknown items)
/ SD of the unknown items' scores (SD with ddof 0, +1e-9), computed on the participant's displays."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from vdc_common import *
from edd import unified as U


def load_items():
    f = pd.read_csv(STEP2 / "features.csv")
    f = f.sort_values(["participant", "trial_index", "position"]).reset_index(drop=True)
    X = U.zscore_within(f[FEATS].to_numpy(float), f.participant.to_numpy())      # label-free, within participant
    y = f.item_label.to_numpy(int)
    g = f.participant.to_numpy(int)
    idx = {int(p): np.flatnonzero(g == p) for p in np.unique(g)}
    return f, X, y, g, idx


def cit_stat(s, y):
    su, sk = s[y == 0], s[y == 1]
    return float((sk.mean() - su.mean()) / (su.std() + 1e-9))


def load_folds():
    d = pd.read_csv(STEP2 / "folds.csv")
    return {r: [sorted(g.participant[g.fold == k].tolist()) for k in range(10)] for r, g in d.groupby("repeat")}, d


def cv_stats(X, y, idx, folds, trainers, negs, eval_pids, repeats=(0, 1, 2), want_tau=True):
    """Nested CV. trainers: participants whose items train the item model (when in the training folds); negs: participants
    whose statistics set the 7% threshold (when in the training folds); eval_pids: participants scored when in the test fold.
    Returns stat[r][pid] and tau[r][pid] (the threshold of the fold in which the participant was tested)."""
    trainers, negs, ev = set(trainers), set(negs), set(eval_pids)
    stat = {r: {} for r in repeats}
    tau = {r: {} for r in repeats}
    with threadpool_limits(1):
        warnings.filterwarnings("ignore")
        for r in repeats:
            for k, test in enumerate(folds[r]):
                tr_pids = [p for p in trainers if p not in set(test)]
                rows = np.concatenate([idx[p] for p in tr_pids])
                gtr = np.concatenate([np.full(len(idx[p]), p) for p in tr_pids])
                fit = U.tune_fit(X[rows], y[rows], gtr, 1000 * r + k)
                for p in test:
                    if p in ev:
                        stat[r][p] = cit_stat(U.decide(fit, X[idx[p]]), y[idx[p]])
                if want_tau:
                    ns = [cit_stat(U.decide(fit, X[idx[p]]), y[idx[p]]) for p in negs if p not in set(test)]
                    t7 = U.thresholds(np.array([0.0]), np.array(ns))["tau_fpr7"]
                    for p in test:
                        tau[r][p] = t7
    return stat, tau
