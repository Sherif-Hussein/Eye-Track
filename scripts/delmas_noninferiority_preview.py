"""Preliminary non-inferiority check for the Delmas sampling-rate curve (agreed margin: 0.02 AUC).

For each pair (reference, candidate), the metric is image-level AUC pooled within each group set,
averaged over the 6 sets and over CV repeats. CIs come from a paired cluster bootstrap that resamples
participants within each set. Non-inferior if the lower 95% bound of (candidate - reference) >= -0.02.

Usage: python -I scripts/delmas_noninferiority_preview.py <checkpoints_dir> [n_boot]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MARGIN = 0.02
PAIRS = [("P4_svm_1000", "P4_svm_30"), ("P4_svm_1000", "P4_svm_60"), ("ALLz_lr_1000", "ALLz_lr_30"),
         ("ALLz_lr_1000", "ALLz_lr_60"), ("P4_svm_1000", "ALLz_lr_1000"), ("P4_svm_1000", "ALLz_lr_30")]


def load(ckpt, name):
    d = pd.read_csv(Path(ckpt) / f"{name}_oof.csv.gz")
    # Average over repeats per (set, participant, image), so each image has one out-of-fold score.
    return d.groupby(["set", "participant", "image"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))


def mean_set_auc(d):
    return float(np.mean([roc_auc_score(g.y, g.score) for _, g in d.groupby("set")]))


def main(ckpt, n_boot=1000):
    rng = np.random.default_rng(0)
    cache = {}
    print(f"{'reference':>14s} -> {'candidate':<14s} {'ref AUC':>7s} {'cand AUC':>8s} {'delta':>7s}  95% CI            verdict (margin {MARGIN})")
    for ref, cand in PAIRS:
        for n in (ref, cand):
            if n not in cache:
                cache[n] = load(ckpt, n)
        m = cache[ref].merge(cache[cand], on=["set", "participant", "image", "y"], suffixes=("_r", "_c"))
        groups = {s: g for s, g in m.groupby("set")}
        parts = {s: g.participant.unique() for s, g in groups.items()}
        deltas = []
        for _ in range(n_boot):
            aucs_r, aucs_c = [], []
            for s, g in groups.items():
                pick = rng.choice(parts[s], len(parts[s]))
                b = pd.concat([g[g.participant == p] for p in pick])
                if b.y.nunique() < 2:
                    continue
                aucs_r.append(roc_auc_score(b.y, b.score_r))
                aucs_c.append(roc_auc_score(b.y, b.score_c))
            deltas.append(np.mean(aucs_c) - np.mean(aucs_r))
        r = mean_set_auc(m.rename(columns={"score_r": "score"}))
        c = mean_set_auc(m.rename(columns={"score_c": "score"}))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        verdict = "superior" if lo > 0 else ("non-inferior" if lo >= -MARGIN else "NOT shown non-inferior")
        print(f"{ref:>14s} -> {cand:<14s} {r:7.3f} {c:8.3f} {c - r:+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
