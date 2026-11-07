"""Task M2 library: XGBoost, LightGBM, CatBoost and RBF-SVM as engine specs. EXPLORATORY. Development data only (Delmas).
Source: imports results/phase10/model_search/ms_lib.py UNCHANGED (engine run_set / run_fold, folds, bundles, F0 frozen LR spec). Writes only
into results/phase10/model_compare/. The held-out dataset (data/external/vdc2024, results/phase9/vdc_heldout) is never read.
Grids are fixed in PLAN.md (SHA-256 in PLAN.sha256.txt) before any full run.
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "results" / "phase10" / "model_search"))
sys.path.append(str(HERE / "site_pkgs"))        # CatBoost 1.2.10 (+ graphviz, plotly) installed with --no-deps --target; .venv untouched
import ms_lib as L  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

L.OUT = HERE                       # redirect every write of the imported engine into this folder
L.CK = HERE / "ck"

GRID_N = (100, 300)
SEEDLESS = None


def prep():
    return [("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]


class GBSpec:
    """kind: xgb | lgbm | cat. The score is the raw margin."""

    def __init__(self, name, kind):
        self.name, self.kind = name, kind
        self.inner = True

    def grid(self, B):
        if self.kind == "xgb":
            return [(n, d, lr) for n in (100, 300) for d in (2, 3) for lr in (0.03, 0.1)]
        if self.kind == "lgbm":
            return [(n, nl, lr) for n in (100, 300) for nl in (4, 8) for lr in (0.03, 0.1)]
        return [(n, d, lr) for n in (100, 300) for d in (2, 4) for lr in (0.03, 0.1)]

    def make(self, hp, y, seed):
        n, a, lr = hp
        if self.kind == "xgb":
            from xgboost import XGBClassifier
            spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
            return XGBClassifier(n_estimators=n, max_depth=a, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                 tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)
        if self.kind == "lgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=n, num_leaves=a, learning_rate=lr, min_child_samples=20, class_weight="balanced", n_jobs=1,
                                  random_state=seed, verbose=-1, force_row_wise=True, deterministic=True)
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=n, depth=a, learning_rate=lr, auto_class_weights="Balanced", thread_count=1, allow_writing_files=False,
                                  random_seed=seed, verbose=0)

    def raw(self, pipe, X, n=None):
        Z = pipe[:-1].transform(X)
        m = pipe[-1]
        n = n or pipe[-1].get_params().get("n_estimators", None) or pipe[-1].get_params().get("iterations")
        if self.kind == "xgb":
            return m.get_booster().predict(__import__("xgboost").DMatrix(Z), output_margin=True, iteration_range=(0, n))
        if self.kind == "lgbm":
            return m.booster_.predict(Z, raw_score=True, num_iteration=n)
        return m.predict(Z, prediction_type="RawFormulaVal", ntree_end=n, thread_count=1)

    _cache = {"key": None, "models": {}}

    def fit(self, B, tr, hp, seed):
        """Boosting is sequential with a fixed seed and learning rate, so the first 100 trees of the 300-tree model ARE the 100-tree model
        (XGBoost and LightGBM, verified numerically, mc_check_prefix.py: identical scores; CatBoost is NOT prefix-consistent and is fitted directly). The 300-tree model is fitted once per (training rows, depth/leaves, learning rate, seed)
        and the 100-tree setting scores with its first 100 trees. Results are identical; the run is a quarter shorter."""
        n, a, lr = hp
        X = B.X["base"]
        if self.kind == "cat":          # CatBoost: the prefix of the longer model is NOT the shorter model (checked), so every setting is fitted directly
            pipe = self.fit_direct(X[tr], B.y[tr], hp, seed)
            f = lambda rows: self.raw(pipe, X[rows], n)  # noqa: E731
            f.pipe, f.n = pipe, n
            return f
        key = (hash(tr.tobytes()), seed)
        c = GBSpec._cache
        if c["key"] != key or c.get("kind") != self.kind:
            c["key"], c["kind"], c["models"] = key, self.kind, {}
        if (a, lr) not in c["models"]:
            c["models"][(a, lr)] = Pipeline(prep() + [("clf", self.make((300, a, lr), B.y[tr], seed))]).fit(X[tr], B.y[tr])
        pipe = c["models"][(a, lr)]
        f = lambda rows: self.raw(pipe, X[rows], n)  # noqa: E731
        f.pipe, f.n = pipe, n
        return f

    def fit_direct(self, X, y, hp, seed):
        """Plain fit with the stated n_estimators (used for cost measurements)."""
        return Pipeline(prep() + [("clf", self.make(hp, y, seed))]).fit(X, y)


class SVMSpec:
    def __init__(self, name):
        self.name = name
        self.inner = True

    def grid(self, B):
        return [(C, g) for C in (0.1, 1.0, 10.0) for g in ("scale", 0.01, 0.1)]

    def fit(self, B, tr, hp, seed):
        C, g = hp
        X = B.X["base"]
        pipe = Pipeline(prep() + [("clf", SVC(kernel="rbf", C=C, gamma=g, class_weight="balanced"))]).fit(X[tr], B.y[tr])
        f = lambda rows: pipe.decision_function(X[rows])  # noqa: E731
        f.pipe = pipe
        return f


def registry():
    R = L.spec_registry("delmas")
    return {"LR": R["F0"], "XGB": GBSpec("XGB", "xgb"), "LGBM": GBSpec("LGBM", "lgbm"), "CAT": GBSpec("CAT", "cat"), "SVM": SVMSpec("SVM")}


MODELS = ["LR", "XGB", "LGBM", "CAT", "SVM"]
LABEL = {"LR": "Logistic regression (reference)", "XGB": "XGBoost", "LGBM": "LightGBM", "CAT": "CatBoost", "SVM": "RBF-kernel SVM"}
# condition -> engine condition name and noise draws
CONDS = {"FULL": ("D30", [None]), "CAMERA": ("W30", [1, 2, 3])}
