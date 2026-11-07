import sys, time
sys.path.insert(0, "./results/phase10/model_search")
sys.stdout.reconfigure(encoding="utf-8")
import ms_lib as L
B=L.chen_bundle(want=("tc",))
R=L.spec_registry("chen")
for n in ("C6_enet","C7_rf"):
    t=time.time(); o=L.run_set(R[n],B,repeats=(0,),folds_limit=1); print(n,"1 fold",round(time.time()-t,1),o[0][0]["hp"],flush=True)
