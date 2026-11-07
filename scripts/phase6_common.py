"""Shared helpers for Phase 6 Track A (re-analyses from stored scores). Import with sys.path set to scripts/ and src/."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
OUT = RES / "phase6" / "trackA"
OUT.mkdir(parents=True, exist_ok=True)
OOF = RES / "phase4" / "unified" / "oof"
P2 = RES / "phase2"
B = 2000
CARD_ORDER = ["A1", "A4", "A5", "A2", "A3", "A7", "A6", "A8"]


def log(*a):
    print(*a, flush=True)


def _conv(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(type(o))


def save_section(key, obj):
    """Write results.json section `key` (merging with other sections)."""
    p = OUT / "results.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = json.loads(json.dumps(obj, default=_conv))
    p.write_text(json.dumps(d, indent=1))
    (OUT / f"checkpoint_{key}.json").write_text(json.dumps(d[key], indent=1))


def save_card(key, paragraph):
    p = OUT / "card_parts.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = paragraph.strip()
    p.write_text(json.dumps(d, indent=1))
    lines = ["# Phase 6 Track A card", "",
             "One paragraph per analysis, in the order they were run. Definitions are in `PREREG.md`; deviations in `DEVIATIONS.md`. "
             "All intervals are 95% participant-cluster bootstrap (B = 2000) unless stated; no model was refitted.", ""]
    for k in CARD_ORDER:
        if k in d:
            lines += [f"**{k}.** {d[k]}", ""]
    (OUT / "card.md").write_text("\n".join(lines), encoding="utf-8")


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pct_ci(x, q=(2.5, 97.5)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, q[0])), float(np.percentile(x, q[1]))]


def p_one_sided(d, null_dir="le0"):
    """Descriptive bootstrap tail probability (1 + #{d <= 0}) / (B + 1)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float((1 + np.sum(d <= 0)) / (1 + len(d)))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def roc_points(pos, neg):
    """Empirical ROC polygon: (fpr, tpr) at every distinct threshold (stat >= t), starting at (0, 0)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    t = np.unique(np.concatenate([pos, neg]))[::-1]
    f = np.concatenate([[0.0], [(neg >= x).mean() for x in t]])
    r = np.concatenate([[0.0], [(pos >= x).mean() for x in t]])
    return f, r


def pauc(pos, neg, a=0.10):
    """Partial AUC over FPR in [0, a] under the empirical ROC polygon. Returns (area, area/a, McClish standardised)."""
    f, r = roc_points(pos, neg)
    keep = f < a
    ff = np.append(f[keep], a)
    # tpr at fpr = a: highest tpr among points with fpr <= a, interpolated within the polygon
    j = np.searchsorted(f, a, side="right")
    if j >= len(f):
        ta = r[-1]
    else:
        f0, f1, r0, r1 = f[j - 1], f[j], r[j - 1], r[j]
        ta = r0 + (r1 - r0) * (a - f0) / (f1 - f0) if f1 > f0 else r1
    tt = np.append(r[keep], ta)
    area = float(np.trapezoid(tt, ff))
    mn, mx = a * a / 2, a
    mcc = 0.5 * (1 + (area - mn) / (mx - mn))
    return area, area / a, float(mcc)


def det_at_fpr(pos, neg, x):
    """Highest detection rate with realised FPR <= x (step function over thresholds stat > t)."""
    f, r = roc_points(pos, neg)
    ok = f <= x + 1e-12
    return float(r[ok].max())


def wilson(k, n, z=1.959964):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def ci_s(ci, d=3):
    return f"[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"


def ci_u(ci, d=3):
    return f"[{ci[0]:.{d}f}, {ci[1]:.{d}f}]"
