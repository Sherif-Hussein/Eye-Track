"""Participant-disjoint fold construction, frozen to JSON so every experiment uses identical splits."""
import json
from pathlib import Path

import numpy as np


def paired_participant_folds(units, labels, seed):
    """Folds of one positive and one negative participant each (Celniak's 2-out GroupKFold design).

    Unpaired leftovers become single-participant folds.
    """
    units, labels = np.asarray(units), np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    pos = list(rng.permutation(units[labels == 1]))
    neg = list(rng.permutation(units[labels == 0]))
    folds = [[str(p), str(n)] for p, n in zip(pos, neg)]
    k = min(len(pos), len(neg))
    folds += [[str(u)] for u in pos[k:] + neg[k:]]
    return folds


def make_repeated_folds(units, labels, n_repeats, base_seed=2026):
    return [paired_participant_folds(units, labels, base_seed + r) for r in range(n_repeats)]


def save_folds(path, folds_per_repeat, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "repeats": folds_per_repeat}, indent=1))


def load_folds(path):
    return json.loads(Path(path).read_text())["repeats"]
