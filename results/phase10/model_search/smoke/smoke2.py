import sys, time
sys.path.insert(0, "./results/phase10/model_search")
sys.stdout.reconfigure(encoding="utf-8")
import ms_lib as L
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
t=time.time(); cond=L.DelmasCond("D30"); print("cond", round(time.time()-t,1))
t=time.time(); B=L.delmas_set_bundle(cond,"64_g1"); print("bundle",round(time.time()-t,1))
R=L.spec_registry("delmas")
res={}
for name,spec in R.items():
    t=time.time()
    try:
        outs=L.run_set(spec,B,repeats=(0,),folds_limit=5)
    except Exception as e:
        import traceback; traceback.print_exc(); print(name,"FAILED",e); continue
    o=L.oof_frame(B,outs)
    auc=roc_auc_score(o.y,o.score)
    print(f"{name:16s} item AUC (set 64_g1, r0) {auc:.4f}  {time.time()-t:5.1f}s  hp={[d['hp'] for d in outs[0]][:2]}",flush=True)
