"""Pipelines screened on the Mazza/Colasanti RealEye dataset. Every pipeline starts from the 90 raw
columns; feature engineering is row-wise (within-person), and imputation, clipping, scaling,
feature selection and tuning are fitted on outer-training participants only."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from edd.harness import grouped_inner  # noqa: E402
from edd.mazza import Featurizer, QuantileClipper, StabilitySelector  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

C_GRID = [0.01, 0.1, 1, 10]


def _clf(model, multiclass=False):
    cw = "balanced"
    if model == "lr_unreg":   # WEKA Logistic default: ridge 1e-8, i.e. practically unpenalised
        return LogisticRegression(C=1e8, max_iter=1000, class_weight=cw), {}
    if model == "lr":
        if multiclass:
            return LogisticRegression(max_iter=2000, class_weight=cw), {"clf__C": C_GRID}
        return LogisticRegression(solver="liblinear", max_iter=2000, class_weight=cw), {"clf__C": C_GRID}
    if model == "l1":
        return (LogisticRegression(penalty="l1", solver="saga" if multiclass else "liblinear", max_iter=5000,
                                   class_weight=cw), {"clf__C": [0.03, 0.1, 0.3, 1]})
    if model == "svm":
        return SVC(kernel="rbf", class_weight=cw), {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.003]}
    if model == "rf":
        return (RandomForestClassifier(n_estimators=150, max_features="sqrt", class_weight=cw, n_jobs=1,
                                       random_state=0), {"clf__min_samples_leaf": [3]})
    if model == "lgbm":
        return (LGBMClassifier(learning_rate=0.05, num_leaves=4, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, class_weight=cw, n_jobs=1, verbose=-1, random_state=0),
                {"clf__n_estimators": [50, 150], "clf__min_child_samples": [5, 15]})
    raise ValueError(model)


def build(raw_cols, sets, model, select=None, clip=True, multiclass=False):
    """Return factory(seed) -> unfitted GridSearchCV with participant-grouped inner 5-fold CV."""
    def factory(seed):
        clf, grid = _clf(model, multiclass)
        steps = [("feat", Featurizer(raw_cols=raw_cols, sets=tuple(sets))),
                 ("impute", SimpleImputer(strategy="median"))]
        if clip:
            steps.append(("clip", QuantileClipper()))
        steps.append(("scale", StandardScaler()))
        grid = dict(grid)
        if select and "clf__C" in grid:
            grid["clf__C"] = [0.1, 1]
        if select == "anova":
            steps.append(("sel", SelectKBest(f_classif)))
            grid["sel__k"] = [5, 10, 20]
        elif select == "stability":
            steps.append(("sel", StabilitySelector(n_boot=20, seed=seed)))
            grid["sel__threshold"] = [0.5, 0.7]
        steps.append(("clf", clf))
        pipe = Pipeline(steps)
        scoring = "roc_auc_ovr" if multiclass else "roc_auc"
        if not grid:
            grid = {"clf__class_weight": ["balanced"]}
        return GridSearchCV(pipe, grid, cv=grouped_inner(5, seed), scoring=scoring, n_jobs=1, refit=True)
    return factory


# name: (idea, feature sets, model, selector, description)
CONFIGS = {
    "base_pub36_lr_unreg": ("honest baseline", ["pub36"], "lr_unreg", None,
                            "Published method: logistic regression (ridge 1e-8) on the 36 target-AOI count/duration features"),
    "i1_pub36_lr": ("1 baseline models", ["pub36"], "lr", None, "Same 36 features, L2-LR with tuned C"),
    "i1_raw90_lr": ("1 baseline models", ["raw"], "lr", None, "All 90 raw columns, L2-LR"),
    "i1_raw90_svm": ("1 baseline models", ["raw"], "svm", None, "All 90 raw columns, RBF-SVM"),
    "i1_raw90_rf": ("1 baseline models", ["raw"], "rf", None, "All 90 raw columns, random forest"),
    "i1_raw90_lgbm": ("1 baseline models", ["raw"], "lgbm", None, "All 90 raw columns, LightGBM"),
    "i2_diffshare_lr": ("2 within-person differential", ["diffshare"], "lr", None,
                        "AOI share on each target minus own AOI share on neutral images (36), L2-LR"),
    "i2_logratio_lr": ("2 within-person differential", ["logratio"], "lr", None,
                       "log ratio of target AOI value to own neutral AOI mean (36), L2-LR"),
    "i2_z_lr": ("2 within-person differential", ["z"], "lr", None,
                "target AOI value z-scored against own neutral images (36), L2-LR"),
    "i2_diffshare_svm": ("2 within-person differential", ["diffshare"], "svm", None, "diffshare, RBF-SVM"),
    "i3_crime_lr": ("3 crime-area features", ["crime"], "lr", None,
                    "desk-AOI and central-third shares per target, cross-target mean/SD, neutral-referenced (36), L2-LR"),
    "i3_crime_svm": ("3 crime-area features", ["crime"], "svm", None, "crime set, RBF-SVM"),
    "i3_crime_lgbm": ("3 crime-area features", ["crime"], "lgbm", None, "crime set, LightGBM"),
    "i3_crime_diffshare_lr": ("3 crime-area features", ["crime", "diffshare"], "lr", None,
                              "crime + diffshare (72), L2-LR"),
    "i4_all_lr_none": ("4 feature selection", ["raw", "diffshare", "crime"], "lr", None,
                       "raw + diffshare + crime (162), L2-LR, no selection"),
    "i4_all_lr_anova": ("4 feature selection", ["raw", "diffshare", "crime"], "lr", "anova",
                        "162 features, ANOVA-F SelectKBest k in {5,10,20} tuned inside, L2-LR"),
    "i4_all_l1": ("4 feature selection", ["raw", "diffshare", "crime"], "l1", None,
                  "162 features, L1-LR (embedded selection)"),
    "i4_all_lr_stability": ("4 feature selection", ["raw", "diffshare", "crime"], "lr", "stability",
                            "162 features, stability selection (30 L1 half-samples) then L2-LR"),
}
