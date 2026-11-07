"""Check: first 100 trees of the 300-tree model == the 100-tree model (scores equal). Source: mc_lib / model_search engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_lib as M
import numpy as np
L = M.L
sys.stdout.reconfigure(encoding="utf-8")
cobj = L.DelmasCond("D30")
B = L.delmas_set_bundle(cobj, "64_g1", want=("tc",))
te = np.flatnonzero(np.isin(B.part, B.folds[0][0])); tr = np.flatnonzero(~np.isin(B.part, B.folds[0][0]))
reg = M.registry()
out = {}
for name in ("XGB", "LGBM", "CAT"):
    sp = reg[name]
    for a, lr in ((sp.grid(B)[0][1], 0.03), (sp.grid(B)[2][1], 0.1)):
        s_prefix = sp.fit(B, tr, (100, a, lr), 3)(te)
        s_100 = sp.raw(sp.fit_direct(B.X["base"][tr], B.y[tr], (100, a, lr), 3), B.X["base"][te])
        s_prefix300 = sp.fit(B, tr, (300, a, lr), 3)(te)
        s_300 = sp.raw(sp.fit_direct(B.X["base"][tr], B.y[tr], (300, a, lr), 3), B.X["base"][te])
        out[f"{name} a={a} lr={lr}"] = (float(np.abs(s_prefix - s_100).max()), float(np.abs(s_prefix300 - s_300).max()))
        print(name, a, lr, out[f"{name} a={a} lr={lr}"], flush=True)
