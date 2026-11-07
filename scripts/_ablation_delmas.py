"""Delmas feature-engineering ablation: paired step-by-step image-level AUC differences on repeat-averaged scores.

Same convention as the frozen matched run (results/frozen/delmas_efficiency_matched_2026-11-02): scores averaged over repeats
per (participant, image), AUC per group set, mean over the six sets; paired participant-cluster bootstrap within each set
(2000 resamples). Steps whose paired CI is not stored in the result files are computed here; stored pairs are recomputed
as a check. Writes paper/figures/data_delmas_ablation.csv.

Usage: python -I scripts/_ablation_delmas.py
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

CK = "results/phase2/delmas_raw/checkpoints"
SETS = ["64_g1", "64_g2", "64_g3", "32_g3", "32_g4", "32_g5"]
rng = np.random.default_rng(20261102)


def auc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load(model):
    d = pd.read_csv(f"{CK}/{model}_oof.csv.gz")
    return {s: d[d.set == s].groupby(["participant", "image"]).agg(y=("y", "first"), s=("score", "mean")) for s in SETS}


MODELS = ["P4_svm_1000", "P4_lr_1000", "ALL_lr_1000", "ALLz_lr_1000", "P4_z_lr_1000", "ALLz_lr_30", "P4_svm_30"]
D = {m: load(m) for m in MODELS}

# common (participant, image) rows per set, and participant index lists for the cluster bootstrap
common = {}
for s in SETS:
    idx = None
    for m in MODELS:
        idx = D[m][s].index if idx is None else idx.intersection(D[m][s].index)
    common[s] = idx
packs = {}
for s in SETS:
    y = D[MODELS[0]][s].loc[common[s], "y"].to_numpy()
    parts = common[s].get_level_values(0).to_numpy()
    groups = [np.flatnonzero(parts == p) for p in np.unique(parts)]
    packs[s] = (y, groups, {m: D[m][s].loc[common[s], "s"].to_numpy() for m in MODELS})

point = {m: np.mean([auc(packs[s][0], packs[s][2][m]) for s in SETS]) for m in MODELS}
NB = 2000
boot = {m: np.empty(NB) for m in MODELS}
for b in range(NB):
    vals = {m: [] for m in MODELS}
    for s in SETS:
        y, groups, sc = packs[s]
        take = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        yt = y[take]
        for m in MODELS:
            vals[m].append(auc(yt, sc[m][take]))
    for m in MODELS:
        boot[m][b] = np.mean(vals[m])

STEPS = [  # label, candidate, reference
    ("model change: SVM -> logistic regression (published 4 features, 1000 Hz)", "P4_lr_1000", "P4_svm_1000"),
    ("+ time bins, pupil, first look (not standardised, 1000 Hz)", "ALL_lr_1000", "P4_lr_1000"),
    ("+ within-person standardisation (1000 Hz)", "ALLz_lr_1000", "ALL_lr_1000"),
    ("check: standardisation of the published 4 features alone (1000 Hz)", "P4_z_lr_1000", "P4_lr_1000"),
    ("all steps vs published model (1000 Hz) [stored: +0.0118 [-0.0044, +0.0255]]", "ALLz_lr_1000", "P4_svm_1000"),
    ("1000 -> 30 Hz, our features [stored: -0.0075 [-0.0203, +0.0043]]", "ALLz_lr_30", "ALLz_lr_1000"),
    ("ours at 30 Hz vs published model at 1000 Hz [stored: +0.0044 [-0.0153, +0.0230]]", "ALLz_lr_30", "P4_svm_1000"),
]
rows = []
for lab, c, r in STEPS:
    d = boot[c] - boot[r]
    est = point[c] - point[r]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p2 = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    rows.append(dict(step=lab, candidate=c, reference=r, auc_candidate=point[c], auc_reference=point[r], delta=est,
                     ci_lo=lo, ci_hi=hi, p_two_sided=max(p2, 1 / NB)))
out = pd.DataFrame(rows)
print({m: round(v, 4) for m, v in point.items()})
print(out.drop(columns=["candidate", "reference"]).round(4).to_string(index=False))
out.to_csv("paper/figures/data_delmas_ablation.csv", index=False)
