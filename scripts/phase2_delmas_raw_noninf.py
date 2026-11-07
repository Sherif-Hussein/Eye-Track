"""Phase 2, track delmas_raw: non-inferiority of lower sampling rates / cheaper models (margin 0.02 AUC).

Same comparisons as the coordinator's preview (scripts/delmas_noninferiority_preview.py), with a larger
bootstrap and Holm correction across the pairs. Metric: image-level AUC pooled within each of the six
group-sets (OOF scores averaged over the 3 repeats), averaged over the six sets. CI: paired participant-cluster
bootstrap, participants resampled within each set. H0: delta <= -margin; one-sided bootstrap p = share of
bootstrap deltas <= -margin; Holm across the pairs. Non-inferior when the Holm-adjusted p < 0.05
(equivalently, for the family, the lower one-sided bound is above -margin).

Usage: python -I scripts/phase2_delmas_raw_noninf.py <checkpoints_dir> <out_json> [n_boot]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MARGIN = 0.02
PAIRS = [("P4_svm_1000", "P4_svm_60"), ("P4_svm_1000", "P4_svm_30"), ("ALLz_lr_1000", "ALLz_lr_60"),
         ("ALLz_lr_1000", "ALLz_lr_30"), ("P4_svm_1000", "ALLz_lr_1000"), ("P4_svm_1000", "ALLz_lr_30")]


def load(ckpt, name):
    d = pd.read_csv(Path(ckpt) / f"{name}_oof.csv.gz")
    return d.groupby(["set", "participant", "image"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))


def holm(p):
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def main(ckpt, out, n_boot=2000):
    rng = np.random.default_rng(0)
    cache, rows = {}, []
    for ref, cand in PAIRS:
        for n in (ref, cand):
            cache.setdefault(n, load(ckpt, n))
        m = cache[ref].merge(cache[cand], on=["set", "participant", "image", "y"], suffixes=("_r", "_c"))
        per = {}
        for s, g in m.groupby("set"):
            per[s] = [(gp.y.to_numpy(), gp.score_r.to_numpy(), gp.score_c.to_numpy()) for _, gp in g.groupby("participant")]
        auc_r = np.mean([roc_auc_score(np.concatenate([a[0] for a in v]), np.concatenate([a[1] for a in v])) for v in per.values()])
        auc_c = np.mean([roc_auc_score(np.concatenate([a[0] for a in v]), np.concatenate([a[2] for a in v])) for v in per.values()])
        deltas = []
        for _ in range(n_boot):
            dr_, dc_ = [], []
            for v in per.values():
                pick = rng.integers(0, len(v), len(v))
                y = np.concatenate([v[i][0] for i in pick])
                if len(np.unique(y)) < 2:
                    continue
                dr_.append(roc_auc_score(y, np.concatenate([v[i][1] for i in pick])))
                dc_.append(roc_auc_score(y, np.concatenate([v[i][2] for i in pick])))
            deltas.append(np.mean(dc_) - np.mean(dr_))
        d = np.array(deltas)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"reference": ref, "candidate": cand, "auc_ref": float(auc_r), "auc_cand": float(auc_c),
                     "delta": float(auc_c - auc_r), "ci95": [float(lo), float(hi)],
                     "lower_one_sided_95": float(np.percentile(d, 5)),
                     "p_inferior_raw": float((1 + np.sum(d <= -MARGIN)) / (1 + len(d))),
                     "p_superior_raw": float((1 + np.sum(d <= 0)) / (1 + len(d))) if False else None})
        print(f"{ref} -> {cand}: {auc_r:.4f} -> {auc_c:.4f} delta {auc_c - auc_r:+.4f} CI [{lo:+.4f}, {hi:+.4f}]", flush=True)
    adj = holm(np.array([r["p_inferior_raw"] for r in rows]))
    for r, a in zip(rows, adj):
        r["p_inferior_holm"] = float(a)
        r["verdict"] = ("non-inferior (Holm, margin 0.02)" if a < 0.05 else "non-inferiority not shown")
        r["ci_lower_ge_minus_margin"] = bool(r["ci95"][0] >= -MARGIN)
        del r["p_superior_raw"]
        print(f"  {r['candidate']} vs {r['reference']}: p_raw {r['p_inferior_raw']:.4f} holm {a:.4f} -> {r['verdict']}"
              f" (two-sided 95% CI lower >= -0.02: {r['ci_lower_ge_minus_margin']})", flush=True)
    Path(out).write_text(json.dumps({"margin": MARGIN, "n_boot": n_boot, "pairs": rows}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
