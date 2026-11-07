"""Paired participant-bootstrap CI of (best candidate - honest baseline) on identical repeat-averaged
out-of-fold scores, for trial AUC, person AUC and the number of participants detected at alpha .05.
Reads results/phase2/chen/scores_*.csv.gz + person_tests_*.csv; writes delta_ci.json and merges it into
results.json under `best_vs_baseline_paired` (run only when no phase2_chen.py process is running)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase2" / "chen"
BEST = sys.argv[1] if len(sys.argv) > 1 else "I2b_bins_fusion"
N_BOOT = 2000
res = {}
for exp in ("exp1", "exp2"):
    rng = np.random.default_rng(11)
    sc = {}
    for n in (BEST, "B0_rate_peak", "B0_size_peak"):
        t = pd.read_csv(OUT / f"scores_{exp}_{n}.csv.gz")
        a = t.groupby("row").agg(unit=("unit", "first"), y=("y", "first"), score=("score", "mean")).reset_index()
        sc[n] = a.set_index("row")
    pts = {n: pd.read_csv(OUT / f"person_tests_{exp}_{n}.csv").set_index("unit") for n in sc}
    units = pts[BEST].index.to_numpy()
    by_unit = {n: {u: d for u, d in sc[n].groupby("unit")} for n in sc}

    def stats_(n, pick):
        t = pd.concat([by_unit[n][u] for u in pick])
        p = pts[n].loc[pick]
        yp = np.r_[np.ones(len(p)), np.zeros(len(p))]
        sp = np.r_[p.z_fam.to_numpy(), p.z_c1.to_numpy()]
        ok = np.isfinite(sp)
        return (roc_auc_score(t.y, t.score), roc_auc_score(yp[ok], sp[ok]), int((p.p < 0.05).sum()))

    out = {}
    for base in ("B0_rate_peak", "B0_size_peak"):
        pt = np.array(stats_(BEST, units)) - np.array(stats_(base, units))
        d = []
        for _ in range(N_BOOT):
            pick = rng.choice(units, len(units), replace=True)
            d.append(np.array(stats_(BEST, pick)) - np.array(stats_(base, pick)))
        d = np.array(d)
        out[f"vs_{base}"] = {k: {"delta": float(pt[i]), "ci95": [float(np.percentile(d[:, i], 2.5)), float(np.percentile(d[:, i], 97.5))]}
                             for i, k in enumerate(("trial_auc", "person_auc", "n_detected_alpha05"))}
    res[exp] = out
    print(exp, json.dumps(out, indent=1))
(OUT / "delta_ci.json").write_text(json.dumps({"best": BEST, "n_boot": N_BOOT, **res}, indent=1))
r = json.loads((OUT / "results.json").read_text())
r["best_vs_baseline_paired"] = {"best": BEST, "n_boot": N_BOOT, **res}
(OUT / "results.json.tmp").write_text(json.dumps(r, indent=1))
(OUT / "results.json.tmp").replace(OUT / "results.json")
