"""Delmas et al. (2024) famous-faces short-term-memory CIT, EyeLink 1000 at 1000 Hz (OSF vygfz).

The authors' trial tables have no header row; their import code reads the first data row as a header
and so drops one trial per file. We read with `header=None` to keep it.

Groups: 1 control (memory only), 2 concealment, 3 simple countermeasure, 4 enhanced countermeasure,
5 countermeasure with gaze feedback. `face` = 1 for a famous (known) face.
"""
import random

import pandas as pd

COLUMNS = ["participant", "group", "face", "image", "dwell_1", "dwell_2", "n_fix", "n_visit"]
FEATURES = ["dwell_1", "dwell_2", "n_fix", "n_visit"]


def load_trials(xlsx_path):
    df = pd.read_excel(xlsx_path, header=None, names=COLUMNS).dropna()
    for c in ("participant", "group", "face", "image"):
        df[c] = df[c].astype(int)
    return df.reset_index(drop=True)


def add_control_label(df, seed=42):
    """The authors' chance baseline: 8 of the 64 images, drawn with Python's `random`, relabelled as 'known'."""
    random.seed(seed)
    chosen = set(random.sample(range(1, 65), 8))
    df = df.copy()
    df["control"] = df["image"].isin(chosen).astype(int)
    return df
