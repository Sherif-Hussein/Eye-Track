import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import mc_lib as M
import numpy as np
rng=np.random.default_rng(0)
X=rng.normal(size=(700,13));y=(rng.random(700)<0.3).astype(int);X[:,0]+=y*0.5
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
def t(f,n=3):
    r=[]
    for _ in range(n):
        c=time.process_time();w=time.perf_counter();f();r.append((time.process_time()-c,time.perf_counter()-w))
    return r
for n in (100,300):
    print("xgb",n,t(lambda:XGBClassifier(n_estimators=n,max_depth=3,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,tree_method="hist",n_jobs=1,verbosity=0).fit(X,y)))
    print("xgb exact-free nthread",n,t(lambda:XGBClassifier(n_estimators=n,max_depth=3,learning_rate=0.1,tree_method="hist",n_jobs=1,verbosity=0,max_bin=64).fit(X,y)))
    print("lgb",n,t(lambda:LGBMClassifier(n_estimators=n,num_leaves=8,verbose=-1,n_jobs=1,force_row_wise=True,deterministic=True).fit(X,y)))
    print("cat",n,t(lambda:CatBoostClassifier(iterations=n,depth=4,thread_count=1,allow_writing_files=False,verbose=0).fit(X,y)))
