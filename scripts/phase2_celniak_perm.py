"""Phase 2 (celniak track), takeover stage 1-2: label-permutation tests with per-permutation checkpoints.

* global_gaze_lr, labels shuffled only WITHIN test-order groups (first_exam), 100 shuffles
* orient_lr (best probe-specific idea), labels shuffled between all participants, 50 shuffles

Same test as `participant_permutation` in phase2_celniak_screen.py (full nested pipeline refitted on
repeat-0 folds, observed = repeat-0 participant AUC), but permutation i uses its own RNG seed (1000 + i), so a
stopped run resumes exactly where it stopped. State is kept in results/phase2/celniak/stage_perm_<name>_<key>.json
and results.json is NOT touched here (merged later by phase2_celniak_finalize.py).

Usage: python -I scripts/phase2_celniak_perm.py [--jobs 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import phase2_celniak_screen as S  # noqa: E402
from edd.harness import aggregate, nested_fit, run_outer  # noqa: E402
from edd.metrics import binary_metrics  # noqa: E402

OUT = ROOT / "results" / "phase2" / "celniak"
DER = ROOT / "data" / "derived"
JOBS = [("global_gaze_lr", "order_et", "permutation_within_order", 100),
        ("orient_lr", None, "permutation", 50)]


def feature_columns(P, name):
    """Feature columns of an idea, from the same table builders as the screening script."""
    trials = pd.read_csv(DER / "celniak_p2_trials.csv")
    items = pd.read_csv(DER / "celniak_p2_items.csv")
    if name == "global_gaze_lr":
        return list(S.global_features(trials).columns)
    if name == "orient_lr":
        return list(S.orient_features(trials, items).columns)
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    P = pd.read_csv(DER / "celniak_p2_participant_features.csv")
    P[S.UNIT] = P[S.UNIT].astype(str)
    folds = json.loads((OUT / "folds.json").read_text())
    folds0 = folds["repeats"][0]
    for name, strata, key, n_perm in JOBS:
        f = OUT / f"stage_perm_{name}_{key}.json"
        feats = feature_columns(P, name)
        fit = nested_fit(S.lr_search_factory(len(feats)))
        tr = pd.read_csv(OUT / f"trials_{name}.csv")
        obs = float(binary_metrics(*[aggregate(tr).query("repeat == 0")[c] for c in ("y", "score")])["auc"])
        state = json.loads(f.read_text()) if f.exists() else {"name": name, "strata": strata, "n_perm": n_perm,
                                                              "observed_auc": obs, "null": [], "seconds": 0.0}
        assert abs(state["observed_auc"] - obs) < 1e-9
        t_start = time.perf_counter()
        while len(state["null"]) < n_perm:
            i = len(state["null"])
            rng = np.random.default_rng(1000 + i)
            d = P.copy()
            lab = d[S.LABEL].to_numpy().copy()
            groups = [np.arange(len(d))] if strata is None else [np.flatnonzero(d[strata] == v) for v in sorted(d[strata].unique())]
            for idx in groups:
                lab[idx] = rng.permutation(lab[idx])
            d[S.LABEL] = lab
            p = aggregate(run_outer(d, feats, S.LABEL, S.UNIT, [folds0], fit, n_jobs=a.jobs))
            state["null"].append(float(binary_metrics(p.y, p.score)["auc"]))
            state["seconds"] += time.perf_counter() - t_start
            t_start = time.perf_counter()
            f.write_text(json.dumps(state))
            if (i + 1) % 10 == 0:
                print(f"{name} {key}: {i + 1}/{n_perm}  [{state['seconds']:.0f}s]", flush=True)
        null = np.asarray(state["null"])
        state.update({"null_mean": float(null.mean()), "null_sd": float(null.std()),
                      "null_p95": float(np.percentile(null, 95)),
                      "p_value": float((1 + np.sum(null >= obs)) / (1 + n_perm)), "done": True,
                      "note": "repeat-0 folds; observed = repeat-0 participant AUC; permutation i uses seed 1000+i"})
        f.write_text(json.dumps(state))
        print(name, key, {k: v for k, v in state.items() if k != "null"}, flush=True)


if __name__ == "__main__":
    main()
