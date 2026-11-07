"""Phase 6 Track B, analysis B2: best-of-216-grid null of the Celniak published-protocol reproduction.

Pre-registration: results/phase6/trackB/PREREG.md (section 2). The protocol is the class-paired reproduction of scripts/audit_followup.py
(part A), with the full 216-point grid. The StandardScaler + SelectKBest step is cached per (fold, k); this gives bit-identical inputs to
the SVC and is verified against the verbatim function (`verify` mode).

Usage (python -I scripts/phase6_trackB_b2.py <mode> [options]):
  verify                      cached implementation vs verbatim audit function on one pairing seed (all 216 points), plus real seeds 1,2
  real     [--workers 7]      real-label reference, pairing seeds 1..10 -> b2_real_label_reference.csv
  time     --scratch DIR      one shuffle, timing only
  run      [--n 300] [--workers 7]   null shuffles, resumable -> perms_b2/shuffles.jsonl
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from edd.celniak import PAPER_FEATURES  # noqa: E402
from edd.metrics import binary_metrics  # noqa: E402

OUT = ROOT / "results" / "phase6" / "trackB"
T0 = time.time()
GRID = list(product([3, 5, 7, 10, 13, "all"], [0.001, 0.01, 0.1, 1, 10, 100], [0.0001, 0.001, 0.01, 0.1, 1, 10]))
SEED_BASE = 20261202
PAIR_SEED0 = 5000


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:7.0f}s]", *a, flush=True)


_D = {}


def data():
    if not _D:
        sl = pd.read_csv(ROOT / "data" / "derived" / "celniak_slide_features.csv")
        _D["X"] = sl[PAPER_FEATURES].to_numpy(float)
        _D["y"] = sl.label.to_numpy(int)
        _D["g"] = sl.recording.to_numpy()
        _D["per"] = sl.groupby("recording").label.first()
    return _D


def paired(labels, seed):
    """Class-paired folds (verbatim from audit_followup.py)."""
    r = np.random.default_rng(seed)
    pos = list(r.permutation(labels.index[labels == 1]))
    neg = list(r.permutation(labels.index[labels == 0]))
    folds = [[p, n] for p, n in zip(pos, neg)]
    k = min(len(pos), len(neg))
    return folds + [[u] for u in pos[k:] + neg[k:]]


def point_verbatim(k, C, gam, folds, yy):
    """audit_followup.py `point`, unchanged."""
    d = data()
    X, g = d["X"], d["g"]
    rows_ = []
    for test in folds:
        te = np.isin(g, test)
        m = Pipeline([("s", StandardScaler()), ("k", SelectKBest(f_classif, k=k)), ("c", SVC(C=C, gamma=gam))]).fit(X[~te], yy[~te])
        rows_.append(pd.DataFrame({"u": g[te], "y": yy[te], "s": m.decision_function(X[te])}))
    a = pd.concat(rows_).groupby("u").agg(y=("y", "first"), s=("s", "mean"))
    v = pd.concat(rows_).assign(v=lambda d: (d.s >= 0).astype(float)).groupby("u").agg(y=("y", "first"), s=("v", "mean"))
    m1, m2 = binary_metrics(a.y, a.s), binary_metrics(v.y, v.s - 0.5)
    return {"k": str(k), "C": C, "gamma": gam, "acc": m1["accuracy"], "auc": m1["auc"], "vote_acc": m2["accuracy"], "vote_auc": m2["auc"]}


def grid_cached(folds, yy):
    """All 216 grid points with the scale+select step computed once per (fold, k). Same SVC inputs as `point_verbatim`."""
    d = data()
    X, g = d["X"], d["g"]
    prep = []
    for test in folds:
        te = np.isin(g, test)
        sc = StandardScaler().fit(X[~te])
        Xtr, Xte = sc.transform(X[~te]), sc.transform(X[te])
        per_k = {}
        for k in (3, 5, 7, 10, 13, "all"):
            sel = SelectKBest(f_classif, k=k).fit(Xtr, yy[~te])
            per_k[k] = (sel.transform(Xtr), sel.transform(Xte))
        prep.append((te, per_k))
    out = []
    for k, C, gam in GRID:
        rows_ = []
        for te, per_k in prep:
            Xtr, Xte = per_k[k]
            m = SVC(C=C, gamma=gam).fit(Xtr, yy[~te])
            rows_.append(pd.DataFrame({"u": g[te], "y": yy[te], "s": m.decision_function(Xte)}))
        cat = pd.concat(rows_)
        a = cat.groupby("u").agg(y=("y", "first"), s=("s", "mean"))
        v = cat.assign(v=lambda dd: (dd.s >= 0).astype(float)).groupby("u").agg(y=("y", "first"), s=("v", "mean"))
        m1, m2 = binary_metrics(a.y, a.s), binary_metrics(v.y, v.s - 0.5)
        out.append({"k": str(k), "C": C, "gamma": gam, "acc": m1["accuracy"], "auc": m1["auc"], "vote_acc": m2["accuracy"], "vote_auc": m2["auc"]})
    return pd.DataFrame(out)


def summarise_grid(r):
    return {"acc_max": float(r.acc.max()), "auc_max": float(r.auc.max()), "vote_acc_max": float(r.vote_acc.max()),
            "vote_auc_max": float(r.vote_auc.max()), "auc_median": float(r.auc.median()), "acc_median": float(r.acc.median()),
            "auc_min": float(r.auc.min())}


def real_job(seed):
    d = data()
    t0 = time.time()
    r = grid_cached(paired(d["per"], seed), d["y"])
    return {"pairing_seed": seed, **summarise_grid(r), "seconds": time.time() - t0}


def null_job(j):
    d = data()
    t0 = time.time()
    rng = np.random.default_rng([SEED_BASE, j])
    perm = pd.Series(rng.permutation(d["per"].to_numpy()), index=d["per"].index)
    yy = pd.Series(d["g"]).map(perm).to_numpy(int)
    for attempt in (1, 2):
        try:
            r = grid_cached(paired(perm, PAIR_SEED0 + j), yy)
            return {"shuffle": j, "pairing_seed": PAIR_SEED0 + j, **summarise_grid(r), "seconds": time.time() - t0}
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"shuffle": j, "error": repr(e)}


def mode_verify(a):
    d = data()
    log_ref = {1: (0.717948717948718, 0.7210526315789474), 2: (0.717948717948718, 0.6710526315789473)}
    rep = {}
    for seed in (1, 2):
        t0 = time.time()
        r = grid_cached(paired(d["per"], seed), d["y"])
        s = summarise_grid(r)
        rep[f"real_seed{seed}"] = {**s, "stored_acc_max": log_ref[seed][0], "stored_auc_max": log_ref[seed][1],
                                   "diff_acc": s["acc_max"] - log_ref[seed][0], "diff_auc": s["auc_max"] - log_ref[seed][1], "seconds": time.time() - t0}
        log("verify real seed", seed, json.dumps(rep[f"real_seed{seed}"]))
    # verbatim vs cached, all 216 points, one shuffled-label pairing (scratch seed, not used in production)
    perm = pd.Series(np.random.default_rng([SEED_BASE, 99999]).permutation(d["per"].to_numpy()), index=d["per"].index)
    yy = pd.Series(d["g"]).map(perm).to_numpy(int)
    folds = paired(perm, 99999)
    t0 = time.time()
    c = grid_cached(folds, yy)
    tc = time.time() - t0
    t0 = time.time()
    v = pd.DataFrame([point_verbatim(k, C, gm, folds, yy) for k, C, gm in GRID])
    tv = time.time() - t0
    mx = float(max(np.abs(c[col] - v[col]).max() for col in ("acc", "auc", "vote_acc", "vote_auc")))
    rep["cached_vs_verbatim"] = {"max_abs_diff_all_216_points_4_metrics": mx, "seconds_cached": tc, "seconds_verbatim": tv}
    log("verify cached vs verbatim", json.dumps(rep["cached_vs_verbatim"]))
    out = Path(a.scratch) if a.scratch else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "b2_verify.json").write_text(json.dumps(rep, indent=1))


def mode_real(a):
    seeds = list(range(1, 11))
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(real_job, seeds, chunksize=1)
    df = pd.DataFrame(res)
    df.to_csv(OUT / "b2_real_label_reference.csv", index=False)
    log("real reference\n" + df.to_string())


def mode_time(a):
    t0 = time.time()
    r = null_job(99998)
    log("one null shuffle:", json.dumps(r))


def mode_run(a):
    pdir = OUT / "perms_b2"
    pdir.mkdir(parents=True, exist_ok=True)
    f = pdir / "shuffles.jsonl"
    done = set()
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["shuffle"])
    jobs = [j for j in range(a.n) if j not in done]
    if a.reverse:
        jobs = jobs[::-1]
    log(f"B2: N target {a.n}, done {len(done)}, to run {len(jobs)}")
    n = len(done)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(null_job, jobs, chunksize=1):
            if "error" in r:
                log("ERROR", r)
                continue
            with open(f, "a") as fh:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
            n += 1
            if n % 10 == 0:
                log(f"B2 progress {n}/{a.n}; last best-of-grid acc {r['acc_max']:.3f} auc {r['auc_max']:.3f} ({r['seconds']:.0f}s)")
                (OUT / "progress_b2.json").write_text(json.dumps({"done": n, "target": a.n, "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
    log("B2 finished", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "real", "time", "run"])
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    {"verify": mode_verify, "real": mode_real, "time": mode_time, "run": mode_run}[a.mode](a)


if __name__ == "__main__":
    main()
