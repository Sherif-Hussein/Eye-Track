"""Task M2 pilot timing (not a result). Source: model_search engine via mc_lib. One outer fold of one group set, per model, CPU time."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_lib as M
L = M.L
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, json

sk = sys.argv[1] if len(sys.argv) > 1 else "64_g1"
t0 = time.time()
cobj = L.DelmasCond("D30")
print("build D30", round(time.time() - t0, 1), "s", flush=True)
tot_rows = {s: len(L.delmas_set_bundle(cobj, s, want=("tc",)).y) for s in L.SETS}
B = L.delmas_set_bundle(cobj, sk, want=("tc",))
print("rows per set", tot_rows, flush=True)
reg = M.registry()
res = {}
for name in M.MODELS:
    spec = reg[name]
    te_units = B.folds[0][0]
    te_m = np.isin(B.part, te_units)
    te, tr = np.flatnonzero(te_m), np.flatnonzero(~te_m)
    c0, w0 = time.process_time(), time.perf_counter()
    rec = L.run_fold(spec, B, tr, te, 0)
    c, w = time.process_time() - c0, time.perf_counter() - w0
    res[name] = {"cpu_s_per_fold": c, "wall_s_per_fold": w, "hp": str(rec["hp"]), "n_grid": len(spec.grid(B)), "n_train": len(tr)}
    print(name, res[name], flush=True)
# estimate: 15 folds x scale by rows over six sets, per pass (FULL=1 pass, CAMERA=3 passes) => 4 passes
tr_rows = len(B.y)
tot = {}
for name, r in res.items():
    per_pass = r["wall_s_per_fold"] * 15 * sum(tot_rows.values()) / tr_rows
    tot[name] = {"per_pass_min_3rep": per_pass / 60, "all_4_passes_min_3rep": 4 * per_pass / 60}
print(json.dumps(tot, indent=1))
print("TOTAL 3 repeats (min):", sum(v["all_4_passes_min_3rep"] for v in tot.values()))
json.dump({"fold_timing": res, "estimate": tot, "rows": tot_rows, "set": sk}, open(Path(__file__).parent / "pilot_timing.json", "w"), indent=1)
