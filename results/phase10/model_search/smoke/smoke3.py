import sys, time
sys.path.insert(0, "./results/phase10/model_search")
sys.stdout.reconfigure(encoding="utf-8")
import ms_lib as L
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
cond=L.DelmasCond("D30"); B=L.delmas_set_bundle(cond,"64_g1")
# zwithin vs pandas
d=cond.df[(cond.df.set=="64")&(cond.df.group==1)&(cond.df.rate==30)].reset_index(drop=True)
z=L.zwithin(d[cond.base_names].to_numpy(float), d.participant.to_numpy())
print("zwithin vs frozen z, max abs", np.nanmax(np.abs(z-d[cond.feats].to_numpy(float))))
R=L.spec_registry("delmas")
o0=L.oof_frame(B,L.run_set(R["F0"],B,repeats=(0,),folds_limit=5)); o1=L.oof_frame(B,L.run_set(R["M1_cl"],B,repeats=(0,),folds_limit=5))
print("corr F0 vs CL", np.corrcoef(o0.score,o1.score)[0,1], np.abs(o0.score-o1.score).max())
for n in ("F4a_pca","F4b_pca+base"):
    t=time.time(); L.run_set(R[n],B,repeats=(0,),folds_limit=1); print(n,"1 fold",round(time.time()-t,1),"s",flush=True)
t=time.time(); L.run_set(R["C7_rf"],B,repeats=(0,),folds_limit=1); print("rf 1 fold",round(time.time()-t,1),"s",flush=True)
t=time.time(); outs=L.run_set(R["C8_cnn"],B,repeats=(0,),folds_limit=5); o=L.oof_frame(B,outs); print("cnn 5 folds",round(time.time()-t,1),"s auc",roc_auc_score(o.y,o.score),"params",R["C8_cnn"].n_params(B),flush=True)
