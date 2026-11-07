"""Task M2 cost measurements. EXPLORATORY. Source: mc_lib / model_search engine (bundle, frozen folds). Writes into results/phase10/model_compare/ only.
Laptop: Intel Core i7-8550U, one thread. The processes running during the measurement (CPU share over 3 s before and after) are recorded in cost.json (key "other_load").
Usage: .venv/Scripts/python.exe -I results/phase10/model_compare/mc_cost.py run      (parent: data, timings, sizes, work counts; spawns `mem` children one at a time)
       .venv/Scripts/python.exe -I results/phase10/model_compare/mc_cost.py mem <MODEL> <setting-json>
Method (also written to card.md):
 - setting = most often chosen setting over the 30 outer folds (6 sets x 5 folds, repeat 0) in FULL; ties: first in grid order.
 - scoring time = predict from precomputed features for ALL rows of one person (the test people of fold 0, set 64_g1: 3 people, plan said 5) and for the 4 items of one display;
   each measurement is a batch of calls lasting >= 0.25 s (process_time has 15.6 ms resolution on Windows) divided by the calls; 6 rounds, models alternated in every round;
   median over rounds x people. CPU = time.process_time, wall = time.perf_counter.
 - fit time = one fit (imputer + scaler + model) on the training people of fold 0 / repeat 0 of set 64_g1; batch >= 0.5 s; 3 rounds alternated; median.
 - peak memory = fresh child process per model: growth of the Windows peak working set (peak_wset) and of the current working set over one fit + one scoring call
   after imports and data loading, plus the tracemalloc peak of Python allocations (native allocations of the libraries are in the working set only).
 - core share = CPU ms per display / 5000 ms. (a) scoring the 4 items of one display, (b) rescoring all items of the person once per display (conservative).
"""
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_lib as M  # noqa: E402

L = M.L
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CK = M.HERE / "ck"
MODELS = M.MODELS + ["ENS"]      # ENS = logistic regression + XGBoost (ADDENDUM_ensemble.md): times, sizes, work and memory are the two models together
DISPLAY_S = 5.0


def modal_hp():
    out = {}
    for m in M.MODELS:
        hps = []
        for sk in L.SETS:
            for f in pickle.load(open(CK / "D30" / m / f"{sk}_r0.pkl", "rb"))[0]:
                hps.append(f["hp"])
        grid = list(L.C_SK) if m == "LR" else M.registry()[m].grid(None)
        counts = {h: hps.count(h) for h in grid}
        best = max(counts.values())
        out[m] = {"hp": next(h for h in grid if counts[h] == best), "counts": {str(h): c for h, c in counts.items() if c}}
    out["ENS"] = {"hp": [out["LR"]["hp"], list(out["XGB"]["hp"])], "counts": {"LR": out["LR"]["counts"], "XGB": out["XGB"]["counts"]}}
    return out


def fit_model(m, X, y, hp, seed=0):
    """Plain fit with the stated setting: the full pipeline object (imputer, scaler, model)."""
    if m == "ENS":
        return (fit_model("LR", X, y, hp[0], seed), fit_model("XGB", X, y, tuple(hp[1]), seed))
    if m == "LR":
        return L.sk_pipe(hp).fit(X, y)
    if m == "SVM":
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
        return Pipeline(M.prep() + [("clf", SVC(kernel="rbf", C=hp[0], gamma=hp[1], class_weight="balanced"))]).fit(X, y)
    return M.registry()[m].fit_direct(X, y, hp, seed)


def score(m, pipe, X):
    if m == "ENS":      # two scores, z-scaled (constants fixed here only for timing; in the study they come from the inner-fold scores), averaged 50/50
        a, b = score("LR", pipe[0], X), score("XGB", pipe[1], X)
        return 0.5 * (a - 0.0) / 1.0 + 0.5 * (b - 0.0) / 1.0
    if m in ("LR", "SVM"):
        return pipe.decision_function(X)
    return M.registry()[m].raw(pipe, X)


def batch(fn, min_wall):
    """Run fn repeatedly for >= min_wall seconds of wall time; return (cpu_ms_per_call, wall_ms_per_call)."""
    fn()
    n = 1
    while True:
        c0, w0 = time.process_time(), time.perf_counter()
        for _ in range(n):
            fn()
        c, w = time.process_time() - c0, time.perf_counter() - w0
        if w >= min_wall:
            return c / n * 1000, w / n * 1000
        n = max(n * 2, int(n * min_wall / max(w, 1e-4) * 1.2) + 1)


def work_counts(m, pipe, Xp):
    if m == "ENS":
        a, b = work_counts("LR", pipe[0], Xp), work_counts("XGB", pipe[1], Xp)
        return {"n_params": a["n_params"] + b["n_params"], "n_trees": b["n_trees"], "mean_leaf_depth": b["mean_leaf_depth"],
                "ops_per_item": a["ops_per_item"] + b["ops_per_item"] + 6.0,
                "ops_note": "logistic regression operations + XGBoost nodes visited + 6 (two z-scalings and the average); mixed units, a count of operations"}
    p = Xp.shape[1]
    clf = pipe[-1]
    Z = pipe[:-1].transform(Xp)
    if m == "LR":
        return {"n_params": 4 * p + 1, "ops_per_item": float(4 * p), "ops_note": f"standardise {p} sub + {p} div; dot product {p} mul + {p} add = {4 * p} operations"}
    if m == "SVM":
        nsv = int(clf.n_support_.sum())
        return {"n_params": nsv * (p + 1) + 2 * p, "n_sv": nsv, "ops_per_item": float(nsv * p),
                "ops_note": f"{nsv} support vectors x {p} features (each term = one subtract, multiply, add; about 3x this in raw arithmetic)"}
    if m == "XGB":
        import xgboost
        bo = clf.get_booster()
        df = bo.trees_to_dataframe()
        by_id = df.set_index("ID")
        dep = {}
        for t in sorted(df.Tree.unique()):
            stack = [(f"{t}-0", 0)]
            while stack:
                nid, d = stack.pop()
                dep[nid] = d
                r = by_id.loc[nid]
                if r.Feature != "Leaf":
                    stack += [(r.Yes, d + 1), (r.No, d + 1)]
        leaves = bo.predict(xgboost.DMatrix(Z), pred_leaf=True).astype(int)       # (n, trees): node index within tree
        n_trees = leaves.shape[1]
        depths = np.array([[dep[f"{t}-{leaves[i, t]}"] for t in range(n_trees)] for i in range(len(leaves))])
        return {"n_params": int(len(df)), "n_trees": int(n_trees), "mean_leaf_depth": float(depths.mean()), "ops_per_item": float(depths.sum(1).mean()),
                "ops_note": "split nodes visited per item = trees x mean leaf depth, measured on the person's items"}
    if m == "LGBM":
        bo = clf.booster_
        df = bo.trees_to_dataframe()
        leafdf = df[df.split_feature.isna()]
        lookup = {(int(r.tree_index), int(str(r.node_index).split("-L")[1])): int(r.node_depth) - 1 for r in leafdf.itertuples()}
        leaves = bo.predict(Z, pred_leaf=True).astype(int)
        n_trees = leaves.shape[1]
        depths = np.array([[lookup[(t, int(leaves[i, t]))] for t in range(n_trees)] for i in range(len(leaves))])
        return {"n_params": int(len(df)), "n_trees": int(n_trees), "mean_leaf_depth": float(depths.mean()), "ops_per_item": float(depths.sum(1).mean()),
                "ops_note": "split nodes visited per item = trees x mean leaf depth, measured on the person's items"}
    it, d = clf.get_param("iterations"), clf.get_param("depth")
    return {"n_params": int(it * (d + 2 ** d)), "n_trees": int(it), "mean_leaf_depth": float(d), "ops_per_item": float(it * d),
            "ops_note": f"oblivious trees: {it} trees x depth {d} (one comparison per level)"}


def other_load(seconds=3.0):
    """Processes using CPU for `seconds` (psutil), excluding this process; returns the busiest ones and the machine-wide CPU percent."""
    import psutil
    me = psutil.Process().pid
    ps = [p for p in psutil.process_iter(["pid", "name"]) if p.pid != me]
    for p in ps:
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    tot = psutil.cpu_percent(interval=seconds)
    rows = []
    for p in ps:
        try:
            c = p.cpu_percent(None)
            if c > 0.5:
                rows.append({"pid": p.pid, "name": p.name(), "cpu_percent_of_one_core": c})
        except Exception:
            pass
    return {"machine_cpu_percent_all_cores": tot, "busy_processes": sorted(rows, key=lambda r: -r["cpu_percent_of_one_core"])[:8]}


def run():
    load_before = other_load()
    L.log("other load before", load_before)
    hpi = modal_hp()
    L.log("modal settings", hpi)
    cobj = L.DelmasCond("D30")
    B = L.delmas_set_bundle(cobj, "64_g1", want=("tc",))
    X = B.X["base"]
    te_units = B.folds[0][0]
    tr = np.flatnonzero(~np.isin(B.part, te_units))
    people = [int(u) for u in te_units][:5]
    rows_of = {u: np.flatnonzero(B.part == u) for u in people}
    n_items = {u: len(r) for u, r in rows_of.items()}
    Xtr, ytr = X[tr], B.y[tr]
    np.savez(M.HERE / "cost_data.npz", Xtr=Xtr, ytr=ytr, Xp=X[rows_of[people[0]]])
    res = {m: {"model": m, "setting": str(hpi[m]["hp"]), "setting_counts": hpi[m]["counts"], "n_features": int(X.shape[1]), "n_train_rows": int(len(tr))} for m in MODELS}
    fit_c = {m: [] for m in MODELS}
    fit_w = {m: [] for m in MODELS}
    for rnd in range(3):
        for m in MODELS:
            c, w = batch(lambda: fit_model(m, Xtr, ytr, hpi[m]["hp"]), 0.5)
            fit_c[m].append(c)
            fit_w[m].append(w)
    pipes = {}
    for m in MODELS:
        pipes[m] = fit_model(m, Xtr, ytr, hpi[m]["hp"])
        res[m]["fit_cpu_ms"], res[m]["fit_wall_ms"] = float(np.median(fit_c[m])), float(np.median(fit_w[m]))
        res[m]["pickle_bytes"] = len(pickle.dumps(pipes[m]))   # ENS: pickle of the pair = sum of both models
        res[m].update(work_counts(m, pipes[m], X[rows_of[people[0]]]))
    sc_c = {m: [] for m in MODELS}
    sc_w = {m: [] for m in MODELS}
    d_c = {m: [] for m in MODELS}
    d_w = {m: [] for m in MODELS}
    for rnd in range(6):
        for m in MODELS:
            for u in people:
                Xu = X[rows_of[u]]
                c, w = batch(lambda: score(m, pipes[m], Xu), 0.25)
                sc_c[m].append(c)
                sc_w[m].append(w)
                X4 = Xu[:4]
                c, w = batch(lambda: score(m, pipes[m], X4), 0.25)
                d_c[m].append(c)
                d_w[m].append(w)
        L.log("round", rnd, {m: round(float(np.median(sc_c[m])), 3) for m in MODELS})
    for m in MODELS:
        r = res[m]
        r["items_per_person_mean"] = float(np.mean(list(n_items.values())))
        r["score_person_cpu_ms"], r["score_person_wall_ms"] = float(np.median(sc_c[m])), float(np.median(sc_w[m]))
        r["score_display4_cpu_ms"], r["score_display4_wall_ms"] = float(np.median(d_c[m])), float(np.median(d_w[m]))
        r["core_share_display4_pct"] = r["score_display4_cpu_ms"] / (DISPLAY_S * 1000) * 100
        r["core_share_person_pct"] = r["score_person_cpu_ms"] / (DISPLAY_S * 1000) * 100
    for m in MODELS:
        out = subprocess.run([sys.executable, "-I", str(Path(__file__)), "mem", m, json.dumps(hpi[m]["hp"])], capture_output=True, text=True, timeout=600)
        L.log("mem", m, out.stdout.strip(), out.stderr.strip()[-300:])
        res[m].update(json.loads(out.stdout.strip().splitlines()[-1]))
    load_after = other_load()
    L.log("other load after", load_after)
    df = pd.DataFrame([res[m] for m in MODELS])
    df.to_csv(M.HERE / "cost.csv", index=False)
    (M.HERE / "cost.json").write_text(json.dumps({"machine": "Intel Core i7-8550U, one thread", "other_load": {"before": load_before, "after": load_after}, "display_s": DISPLAY_S,
                                                   "people": people, "items_per_person": n_items, "rows": [res[m] for m in MODELS]}, indent=1, default=str))
    L.log(df[["model", "setting", "pickle_bytes", "fit_cpu_ms", "score_person_cpu_ms", "score_display4_cpu_ms", "core_share_display4_pct", "ops_per_item"]].to_string())


def mem(m, hp):
    import psutil
    import tracemalloc
    hp = json.loads(hp)
    hp = tuple(hp) if isinstance(hp, list) and m != "ENS" else hp
    z = np.load(M.HERE / "cost_data.npz")
    Xtr, ytr, Xp = z["Xtr"], z["ytr"], z["Xp"]
    p = psutil.Process()
    a0, w0 = p.memory_info().peak_wset, p.memory_info().wset
    tracemalloc.start()
    pipe = fit_model(m, Xtr, ytr, hp)
    score(m, pipe, Xp)
    py_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    mi = p.memory_info()
    print(json.dumps({"mem_peak_wset_growth_mib": (mi.peak_wset - a0) / 2 ** 20, "mem_wset_growth_mib": (mi.wset - w0) / 2 ** 20,
                      "mem_python_peak_mib": py_peak / 2 ** 20, "mem_baseline_wset_mib": w0 / 2 ** 20}))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run()
    else:
        mem(sys.argv[2], sys.argv[3])
