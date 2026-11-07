"""Check: CatBoost predict thread use (CPU > wall suggests several threads) and that thread_count=1 gives identical scores. Source: mc_lib."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_lib as M
import numpy as np
from catboost import CatBoostClassifier
rng = np.random.default_rng(0)
X = rng.normal(size=(700, 13)); y = (rng.random(700) < 0.3).astype(int)
m = CatBoostClassifier(iterations=100, depth=4, thread_count=1, allow_writing_files=False, verbose=0).fit(X, y)
Xp = X[:60]
for tc in (None, 1):
    kw = {} if tc is None else {"thread_count": tc}
    c0, w0 = time.process_time(), time.perf_counter()
    for _ in range(300): s = m.predict(Xp, prediction_type="RawFormulaVal", **kw)
    print("thread_count", tc, "cpu/wall ms", (time.process_time() - c0) / 300 * 1000, (time.perf_counter() - w0) / 300 * 1000)
print("identical:", np.array_equal(m.predict(Xp, prediction_type="RawFormulaVal"), m.predict(Xp, prediction_type="RawFormulaVal", thread_count=1)))
