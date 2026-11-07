import sys, pickle
sys.path.insert(0, "./results/phase10/model_search")
sys.stdout.reconfigure(encoding="utf-8")
import ms_lib as L, numpy as np, pandas as pd
from types import SimpleNamespace
L.CK = L.OUT/"smoke_ck"
m=pickle.load(open(L.CK/"D30"/"meta_64_g1.pkl","rb")); meta=SimpleNamespace(part=m["part"],y=m["y"])
outs=pickle.load(open(L.ck_path("D30","F0","64_g1"),"rb"))
a=L.persons_cit2(meta,outs,"mean")
task={"g":m["part"],"y":m["y"]}
b=U=L.U.persons_cit(task,outs)
a=a.sort_values(["repeat","fold","unit","kind"]).reset_index(drop=True); b=b.sort_values(["repeat","fold","unit","kind"]).reset_index(drop=True)
print(len(a),len(b), np.abs(a.stat.to_numpy()-b.stat.to_numpy()).max(), np.abs(a.tau_fpr7-b.tau_fpr7).max(), np.abs(a.tau_eer-b.tau_eer).max())
o1=pd.concat([L.oof_frame(SimpleNamespace(part=m["part"],y=m["y"],image=m["image"]),pickle.load(open(L.ck_path("D30",c,"64_g1"),"rb")),"64_g1") for c in ("F0",)])
o2=L.oof_frame(SimpleNamespace(part=m["part"],y=m["y"],image=m["image"]),pickle.load(open(L.ck_path("D30","M1_cl","64_g1"),"rb")),"64_g1")
from sklearn.metrics import roc_auc_score
print(roc_auc_score(o1.y,o1.score), roc_auc_score(o2.y,o2.score))
