"""Label-permutation test for the RT-only candidate (read from final.json), using the same procedure as
`one_perm` in phase2_preti_final.py (shuffle labels between participants, rebuild paired folds, rerun the full
nested pipeline for one repeat). Checkpointed to scores/perm_rt.json; reruns resume. Result goes to
perm_rt_result.json (final.json is not touched).

Usage: python -I scripts/phase2_preti_perm_rt.py data/preti/preti2026.xlsx results/phase2/preti --n-perm 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edd import preti as P  # noqa: E402
from edd.folds import paired_participant_folds  # noqa: E402
from edd.metrics import binary_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("xlsx")
ap.add_argument("out")
ap.add_argument("--n-perm", type=int, default=20)
A = ap.parse_args()
OUT = Path(A.out)
res = json.loads((OUT / "results.json").read_text())
name = json.loads((OUT / "final.json").read_text())["candidates"]["rt"]
trials = P.trials_from_fixations(P.fixations(P.load_workbook(A.xlsx)))
feats = res["arms"][name]["features"]


def one_perm(seed):
    rng = np.random.default_rng(seed)
    d = P.arm_subject_table(trials, name)
    per = d.groupby("subject").y.first()
    lab = pd.Series(rng.permutation(per.to_numpy()), index=per.index)
    d = d.assign(y=d.subject.map(lab).astype(int))
    fl = [paired_participant_folds(lab.index.astype(str).to_numpy(), lab.to_numpy(), 5000 + seed)]
    parts = P.run_subject_level(d, feats, fl, P.arm_grid(name), n_jobs=2)
    return binary_metrics(parts.y, parts.score)["auc"]


ck = OUT / "scores" / "perm_rt.json"
state = json.loads(ck.read_text()) if ck.exists() else {"arm": name, "null": [], "seconds": 0.0}
t0 = time.time() - state["seconds"]
for i in range(len(state["null"]), A.n_perm):
    state["null"].append(one_perm(i))
    state["seconds"] = time.time() - t0
    ck.write_text(json.dumps(state))
    print(i, round(state["null"][-1], 3), round(state["seconds"], 1), flush=True)
obs = float(np.mean([binary_metrics(p.y, p.score)["auc"] for _, p in
                     pd.read_csv(OUT / "scores" / f"{name}.csv", dtype={"unit": str}).groupby("repeat")]))
null = np.asarray(state["null"])
r = {"arm": name, "n_perm": len(null), "observed_auc_mean_over_repeats": obs, "null_mean": float(null.mean()),
     "null_sd": float(null.std()), "null_p95": float(np.percentile(null, 95)),
     "p_value": float((1 + np.sum(null >= obs)) / (1 + len(null))), "null": null.tolist(), "seconds": state["seconds"]}
(OUT / "perm_rt_result.json").write_text(json.dumps(r, indent=1))
print({k: v for k, v in r.items() if k != "null"}, flush=True)
