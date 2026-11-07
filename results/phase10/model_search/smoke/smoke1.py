import sys, time
sys.path.insert(0, "./results/phase10/model_search")
sys.stdout.reconfigure(encoding="utf-8")
import ms_lib as L
import numpy as np, pandas as pd
t=time.time(); cond=L.DelmasCond("D30"); print("cond built", round(time.time()-t,1),"s", cond.df.shape, len(cond.feats))
t=time.time(); B=L.delmas_set_bundle(cond,"64_g1",want=("scan","tc","c22")); print("bundle",round(time.time()-t,1),B.n,B.nd,B.d_ok.sum(), {k:v.shape for k,v in B.X.items()})
# aggregation check: mean of trial base vs frozen raw columns
raw = L.agg_to_rows(B.t_raw_base, B.t_rowmap, B.n)
d = cond.df[(cond.df.set=="64")&(cond.df.group==1)&(cond.df.rate==30)].reset_index(drop=True)
print("agg check max abs diff", np.nanmax(np.abs(raw - d[cond.base_names].to_numpy(float))))
spec=L.spec_registry("delmas")["F0"]
t=time.time(); outs=L.run_set(spec,B,repeats=(0,)); print("F0 r0",round(time.time()-t,1),"s")
o=L.oof_frame(B,outs)
ref=pd.read_csv(L.LRC.STORED/"checkpoints"/"ALLz_lr_30_oof.csv.gz")
ref=ref[(ref.set=="64_g1")&(ref.repeat==0)]
m=o.merge(ref,on=["repeat","participant","image","y"],suffixes=("","_ref"))
print(len(o),len(ref),len(m),"max abs diff vs stored", np.abs(m.score-m.score_ref).max())
