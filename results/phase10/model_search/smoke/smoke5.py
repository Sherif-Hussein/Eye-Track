import sys, time, pickle
sys.path.insert(0, "./results/phase10/model_search")
sys.stdout.reconfigure(encoding="utf-8")
import ms_lib as L
import numpy as np, pandas as pd
t=time.time(); B=L.chen_bundle(); print("chen bundle",round(time.time()-t,1),{k:v.shape for k,v in B.X.items()},flush=True)
R=L.spec_registry("chen")
t=time.time(); outs=L.run_set(R["F0"],B,repeats=(0,),folds_limit=1); print("F0 1 fold",round(time.time()-t,1),flush=True)
ref=pickle.load(open(L.ROOT/"results/phase4/unified/checkpoints/chen_unified__f100__r0.pkl","rb"))[0]
d=outs[0][0]
print("same te",np.array_equal(ref["te"],d["te"]),"max abs diff score",np.abs(ref["score"]-d["score"]).max(),"C",ref["C"],d["hp"], "inner oof diff",np.abs(ref["inner_oof"]-d["inner_oof"]).max())
for n in ("F4a_pca","F5a_c22","C8_cnn"):
    t=time.time(); L.run_set(R[n],B,repeats=(0,),folds_limit=1); print(n,"1 fold",round(time.time()-t,1),flush=True)
