"""Phase 7 Track E, E1: like-for-like timing, 2 (rate) x 2 (code) x 2 (model) on one Delmas participant. Definitions: results/phase7/trackE/PREREG.md.

    .venv/Scripts/python.exe -I scripts/phase7_trackE_e1_timing.py --smoke SCRATCH_DIR          # tiny test, any machine state
    .venv/Scripts/python.exe -I scripts/phase7_trackE_e1_timing.py --wait-quiet --poll-min 15   # real run, waits for a quiet machine

The real run refuses to time while a python command line contains `phase6_trackB` (checked before, between rounds and after).
`phase7_trackD` processes are recorded and the load is stored; the run goes ahead with them (documented in DEVIATIONS.md).
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

OUT = ROOT / "results" / "phase7" / "trackE"
PID = 4
SEED_BOOT = 20261303
ROUNDS = 8
RUNS = {"orig_1000": 4, "orig_30": 8, "vec": 30}        # calls per cell per round


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- machine state
def python_cmdlines():
    cmd = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | "
           "ForEach-Object { $_.CommandLine }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def machine_state(tag):
    import psutil
    cl = python_cmdlines()
    return {"tag": tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_python_processes": len(cl),
            "n_phase6_trackB": sum("phase6_trackB" in c for c in cl),
            "n_phase7_trackD": sum("phase7_trackD" in c for c in cl),
            "cpu_percent_system_1s": psutil.cpu_percent(interval=1.0),
            "python_cmdlines": [c[:220] for c in cl]}


# ----------------------------------------------------------------------------- build the eight (+2) cells
def build_cells(log=log):
    from edd import delmas_fast as dfast, delmas_raw as dr, unified as U
    from edd.delmas_raw_matched import load_track
    from edd.harness import scores_of
    track = load_track()
    z = np.load(U.DELMAS_DECODED / f"p{PID:02d}.npz")
    seg = {k: z[k] for k in ("x", "y", "p", "images")}
    images = seg["images"]
    log("building Delmas tables (track code) ...")
    df = track.build_tables(ROOT / "data" / "derived", ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx",
                            ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx")
    P4 = list(track.P4)
    feats30 = [f + "_z" for f in U.DELMAS_FEATS]                    # 13, as Phase 4
    feats1000 = feats30 + ["msacc_rate_z"]                          # 14 incl. microsaccade rate (ALL_z at 1000 Hz)
    models = {}
    for fs in (1000, 30):
        d = df[(df.rate == fs) & (df.set == "64")].reset_index(drop=True)
        models[("pub", fs)] = track.model_factory("svm", 0).fit(d[P4].to_numpy(float), d.face.to_numpy(int), groups=d.participant.to_numpy()).best_estimator_
    log("published SVMs fitted")
    tasks = U.build_delmas(ROOT / "results" / "phase4" / "unified" / "cache" / "delmas_30hz_features.csv", log)
    models[("ours", 30)] = U.fit_deploy([tasks[f"delmas_64_g{g}"] for g in (1, 2, 3)])
    d = df[(df.rate == 1000) & (df.set == "64") & (df.group.isin([1, 2, 3]))].reset_index(drop=True)
    for nm, fl in (("ours", feats1000), ("ours13", feats30)):
        with threadpool_limits(1):
            f = U.tune_fit(d[fl].to_numpy(float), d.face.to_numpy(int), d.participant.astype(str).to_numpy(), 0)
        models[(nm, 1000)] = U.DeployLR(f.P, f.coef, f.intercept, None, f.C)
    log("our LR models fitted")
    # data at both rates, decimation outside the timer
    pre = {1000: [dr.degrade(seg["x"][k], seg["y"][k], seg["p"][k], 1000) for k in range(64)],
           30: [dr.degrade(seg["x"][k], seg["y"][k], seg["p"][k], 30) for k in range(64)]}
    vecdata = {fs: dfast.degrade_batch(seg["x"], seg["y"], seg["p"], fs) for fs in (1000, 30)}
    sc = lambda m, A: scores_of(m, A)                              # noqa: E731
    feat_of = {"pub": P4, "ours": {30: feats30, 1000: feats1000}, "ours13": {1000: feats30}}

    def fl(model, fs):
        f = feat_of[model]
        return f if isinstance(f, list) else f[fs]

    cells = {}
    spec = [("pub", 1000), ("pub", 30), ("ours", 1000), ("ours", 30), ("ours13", 1000)]
    for model, fs in spec:
        for code in ("orig", "vec"):
            if model == "ours13" and fs != 1000:
                continue
            m, feats = models[(model, fs)], fl(model, fs)
            if code == "orig":
                fn = (lambda p=pre[fs], fs=fs, m=m, feats=feats: track.raw_to_scores(p, images, fs, m, feats))
            else:
                fn = (lambda p=vecdata[fs], fs=fs, m=m, feats=feats: dfast.raw_to_scores_fast(p, images, fs, m, feats, sc, chunk=64))
            cells[f"{model}|{fs}|{code}"] = {"fn": fn, "model": model, "rate": fs, "code": code, "n_features": len(feats),
                                             "samples": int(64 * vecdata[fs][0].shape[1]), "runs": RUNS["vec"] if code == "vec" else RUNS[f"orig_{fs}"]}
    # equivalence of the two code paths
    eq = {}
    for model, fs in spec:
        if model == "ours13" and fs != 1000:
            continue
        a, b = cells[f"{model}|{fs}|orig"]["fn"](), cells[f"{model}|{fs}|vec"]["fn"]()
        eq[f"{model}|{fs}"] = float(np.nanmax(np.abs(np.asarray(a) - np.asarray(b))))
    return cells, eq


# ----------------------------------------------------------------------------- timing
class Invalid(Exception):
    pass


def time_cells(cells, rounds, runs_scale=1.0, check_between=True, log=log):
    names = list(cells)
    times = {k: [] for k in names}                                  # per round: array of call times (ms)
    states = []
    with threadpool_limits(1):
        for k in names:
            for _ in range(3):
                cells[k]["fn"]()
        for r in range(rounds):
            if check_between:
                st = machine_state(f"before_round_{r}")
                states.append({k: v for k, v in st.items() if k != "python_cmdlines"})
                if st["n_phase6_trackB"] > 0:
                    raise Invalid(f"phase6_trackB process appeared before round {r}")
            order = names if r % 2 == 0 else names[::-1]
            for k in order:
                n = max(2, int(round(cells[k]["runs"] * runs_scale)))
                t = np.empty(n)
                for i in range(n):
                    t0 = time.perf_counter()
                    cells[k]["fn"]()
                    t[i] = (time.perf_counter() - t0) * 1e3
                times[k].append(t)
            log(f"round {r + 1}/{rounds} done")
    return times, states


def summarise(cells, times):
    rows = []
    for k, c in cells.items():
        per = times[k]
        p50r = [float(np.median(t)) for t in per]
        pooled = np.concatenate(per)
        rows.append({"cell": k, "model": c["model"], "rate_hz": c["rate"], "code": c["code"], "n_features": c["n_features"],
                     "samples_per_call": c["samples"], "rounds": len(per), "calls_per_round": len(per[0]), "n_calls": int(len(pooled)),
                     "p50_ms": float(np.median(p50r)), "p95_ms_pooled": float(np.percentile(pooled, 95)),
                     "p95_ms_median_of_rounds": float(np.median([np.percentile(t, 95) for t in per])),
                     "p50_round_min_ms": float(min(p50r)), "p50_round_max_ms": float(max(p50r)), "mean_ms": float(pooled.mean()),
                     "p50_rounds_ms": json.dumps([round(x, 4) for x in p50r])})
    return pd.DataFrame(rows)


def decompose(cells_df, times, nboot=2000):
    """log-additive decomposition of T(1000, orig, pub) / T(30, vec, ours); Shapley over the six orders; bootstrap over rounds."""
    names = list(times)
    per50 = {k: np.array([np.median(t) for t in times[k]]) for k in names}
    R = len(next(iter(per50.values())))
    rng = np.random.default_rng(SEED_BOOT)
    idxs = [np.arange(R)] + [rng.integers(0, R, R) for _ in range(nboot)]
    base = ("pub", 1000, "orig")
    final = ("ours", 30, "vec")
    factors = ("rate", "code", "model")

    def T(cell, idx):
        return float(np.median(per50["|".join(map(str, cell))][idx]))

    def setting(sw):
        m = final[0] if "model" in sw else base[0]
        r = final[1] if "rate" in sw else base[1]
        c = final[2] if "code" in sw else base[2]
        return (m, r, c)

    def stats(fn):
        vals = np.array([fn(i) for i in idxs])
        return float(vals[0]), float(np.percentile(vals[1:], 2.5)), float(np.percentile(vals[1:], 97.5))

    out = []
    est, lo, hi = stats(lambda i: T(base, i) / T(final, i))
    out.append({"quantity": "total speed-up (pub|1000|orig / ours|30|vec)", "value": est, "ci_lo": lo, "ci_hi": hi})
    # sequential paths and Shapley
    shap = {f: [] for f in factors}
    for order in itertools.permutations(factors):
        sw = set()
        for f in order:
            prev = setting(sw)
            sw.add(f)
            cur = setting(sw)
            shap[f].append((prev, cur))
    for order in itertools.permutations(factors):
        sw, ratio_parts = set(), []
        for f in order:
            prev = setting(sw)
            sw.add(f)
            cur = setting(sw)
            e, l_, h_ = stats(lambda i, p=prev, c=cur: T(p, i) / T(c, i))
            ratio_parts.append((f, e, l_, h_, prev, cur))
        for f, e, l_, h_, prev, cur in ratio_parts:
            out.append({"quantity": f"sequential order {'>'.join(order)}: factor {f} (from {'|'.join(map(str, prev))} to {'|'.join(map(str, cur))})",
                        "value": e, "ci_lo": l_, "ci_hi": h_})
    for f in factors:
        def fn(i, f=f):
            return float(np.exp(np.mean([np.log(T(p, i) / T(c, i)) for p, c in shap[f]])))
        e, l_, h_ = stats(fn)
        out.append({"quantity": f"Shapley (mean log over the six orders) factor: {f}", "value": e, "ci_lo": l_, "ci_hi": h_})
    # the 12 edges of the cube
    for f in factors:
        others = [g for g in factors if g != f]
        lv = {"rate": (1000, 30), "code": ("orig", "vec"), "model": ("pub", "ours")}
        for oa in lv[others[0]]:
            for ob in lv[others[1]]:
                def cell(lev, f=f, oa=oa, ob=ob, others=others):
                    d = {others[0]: oa, others[1]: ob, f: lev}
                    return (d["model"], d["rate"], d["code"])
                a, b = lv[f]
                e, l_, h_ = stats(lambda i: T(cell(a), i) / T(cell(b), i))
                out.append({"quantity": f"edge {f}: {a} -> {b} at {others[0]}={oa}, {others[1]}={ob}", "value": e, "ci_lo": l_, "ci_hi": h_})
    # sensitivity: our 13-feature model at 1000 Hz
    for code in ("orig", "vec"):
        if f"ours13|1000|{code}" in per50 and f"ours|1000|{code}" in per50:
            e, l_, h_ = stats(lambda i, code=code: T(("ours", 1000, code), i) / T(("ours13", 1000, code), i))
            out.append({"quantity": f"microsaccade cost: ours 14 features / 13 features at 1000 Hz, {code} code", "value": e, "ci_lo": l_, "ci_hi": h_})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--wait-quiet", action="store_true")
    ap.add_argument("--poll-min", type=float, default=15.0)
    ap.add_argument("--max-wait-h", type=float, default=8.0)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    a = ap.parse_args()
    out = Path(a.smoke) if a.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    rounds = 1 if a.smoke else a.rounds
    scale = 0.34 if a.smoke else 1.0
    cells, eq = build_cells()
    log("code-path equivalence (max abs score diff):", json.dumps(eq))
    state0 = None
    if not a.smoke:
        t_start = time.time()
        while True:
            state0 = machine_state("start_check")
            log(f"check: trackB={state0['n_phase6_trackB']} trackD={state0['n_phase7_trackD']} python={state0['n_python_processes']} cpu={state0['cpu_percent_system_1s']}%")
            if state0["n_phase6_trackB"] == 0:
                break
            if not a.wait_quiet or (time.time() - t_start) / 3600 > a.max_wait_h:
                sys.exit("REFUSED: Track B processes are running (phase6_trackB in a python command line)")
            time.sleep(a.poll_min * 60)
    t0 = time.time()
    try:
        times, states = time_cells(cells, rounds, scale, check_between=not a.smoke)
        invalid = None
    except Invalid as e:
        sys.exit(f"INVALID run: {e}")
    state1 = machine_state("end_check") if not a.smoke else None
    if state1 is not None and state1["n_phase6_trackB"] > 0:
        invalid = "phase6_trackB process present at the end of the run"
    tab = summarise(cells, times)
    dec = decompose(tab, times, nboot=50 if a.smoke else 2000)
    tab.to_csv(out / "E1_timing_cells.csv", index=False)
    dec.to_csv(out / "E1_decomposition.csv", index=False)
    import platform
    meta = {"participant": PID, "rounds": rounds, "calls_per_round": {k: c["runs"] for k, c in cells.items()}, "equivalence_max_abs_score_diff": eq,
            "elapsed_s": time.time() - t0, "state_start": state0, "state_end": state1, "state_between_rounds": states,
            "invalid": invalid, "platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
            "single_thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    R = {"E1": {"cells": json.loads(tab.to_json(orient="records")), "decomposition": json.loads(dec.to_json(orient="records")), "meta": meta}}
    (out / ("results_E1.json")).write_text(json.dumps(R, indent=1, default=str))
    pd.set_option("display.width", 250)
    log("\n" + tab[["cell", "n_features", "p50_ms", "p95_ms_pooled", "p50_round_min_ms", "p50_round_max_ms"]].to_string())
    log("\n" + dec.head(1).to_string())


if __name__ == "__main__":
    main()
