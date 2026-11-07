import sys, time
sys.path.insert(0, "./results/phase10/model_search")
sys.stdout.reconfigure(encoding="utf-8")
import ms_lib as L
import numpy as np, pandas as pd
cond=L.DelmasCond("D30"); B=L.delmas_set_bundle(cond,"64_g1")
R=L.spec_registry("delmas")
for n in ("F4a_pca","F4b_pca+base","C7_rf"):
    t=time.time(); L.run_set(R[n],B,repeats=(0,),folds_limit=2); print(n,"2 folds",round(time.time()-t,1),"s",flush=True)
t=time.time(); cw=L.DelmasCond("W15",draw=1); print("W15 cond build",round(time.time()-t,1),"s", cw.df.shape, cw.feats)
Bw=L.delmas_set_bundle(cw,"64_g1"); print({k:v.shape for k,v in Bw.X.items()})
t=time.time(); o=L.run_set(R["F0"],Bw,repeats=(0,),folds_limit=5); print("F0 web 5 folds",round(time.time()-t,1))
