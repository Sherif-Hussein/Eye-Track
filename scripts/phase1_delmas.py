"""Phase 1 on Delmas et al. (2024): faithful reproduction of the between-participant SVM, an honest
variant, a random-trial-split leakage variant, and EER/HTER plus deployment cost.

Usage: python -I scripts/phase1_delmas.py <BookFIP_64_trials.xlsx> <BookFIP_32_trials.xlsx> <out_dir> [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from edd import efficiency as eff  # noqa: E402
from edd.delmas import FEATURES, add_control_label, load_trials  # noqa: E402
from edd.metrics import binary_metrics, bootstrap_ci  # noqa: E402

GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
PAPER_WEIGHTS = {1: 8 / 64, 0: 56 / 64}  # as in the authors' code
# Published between-participant results: Table 6 (64 trials, groups 1-3) and Table 8 (last 32 trials, groups 3-5).
PUBLISHED = {("64", 1): {"auc": 1.00, "ba": .98, "f1": .97}, ("64", 2): {"auc": .93, "ba": .89, "f1": .83},
             ("64", 3): {"auc": 1.00, "ba": .96, "f1": .93}, ("32", 3): {"auc": .82, "ba": .94, "f1": .73},
             ("32", 4): {"auc": .99, "ba": 1.00, "f1": .98}, ("32", 5): {"auc": 1.00, "ba": 1.00, "f1": .99}}


def lopo(data, target, variant, grid):
    """Leave-one-participant-out. 'paper' = scaler outside the search, non-grouped inner 4-fold,
    authors' class weights. 'honest' = scaler inside, participant-grouped inner folds, balanced weights."""
    per_p, pooled = [], []
    for p in sorted(data.participant.unique()):
        tr, te = data[data.participant != p], data[data.participant == p]
        Xtr, ytr, Xte, yte = tr[FEATURES].to_numpy(), tr[target].to_numpy(), te[FEATURES].to_numpy(), te[target].to_numpy()
        if variant == "paper":
            scaler = StandardScaler().fit(Xtr)
            gs = GridSearchCV(SVC(class_weight=PAPER_WEIGHTS), {"C": grid, "gamma": grid},
                              cv=StratifiedKFold(4, shuffle=True, random_state=42), n_jobs=-1)
            gs.fit(scaler.transform(Xtr), ytr)
            model, Xte_ = gs.best_estimator_, scaler.transform(Xte)
        else:
            pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC(class_weight="balanced"))])
            gs = GridSearchCV(pipe, {"clf__C": grid, "clf__gamma": grid}, scoring="roc_auc", n_jobs=-1,
                              cv=StratifiedGroupKFold(4, shuffle=True, random_state=42))
            gs.fit(Xtr, ytr, groups=tr.participant.to_numpy())
            model, Xte_ = gs.best_estimator_, Xte
        score = model.decision_function(Xte_)
        pred = (score >= 0).astype(int)
        per_p.append({"participant": int(p), "ba": balanced_accuracy_score(yte, pred),
                      "auc": roc_auc_score(yte, score), "f1": f1_score(yte, pred, zero_division=0)})
        pooled.append(pd.DataFrame({"participant": p, "y": yte, "score": score}))
    return pd.DataFrame(per_p), pd.concat(pooled, ignore_index=True), model


def random_split(data, grid):
    """Leakage variant: trial-level 4-fold ignoring who the trials belong to."""
    out = []
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=42).split(data, data.face):
        pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC(class_weight="balanced"))])
        gs = GridSearchCV(pipe, {"clf__C": grid, "clf__gamma": grid}, scoring="roc_auc", n_jobs=-1,
                          cv=StratifiedKFold(4, shuffle=True, random_state=0))
        gs.fit(data.iloc[tr][FEATURES].to_numpy(), data.iloc[tr].face.to_numpy())
        out.append(pd.DataFrame({"y": data.iloc[te].face.to_numpy(),
                                 "score": gs.best_estimator_.decision_function(data.iloc[te][FEATURES].to_numpy())}))
    return pd.concat(out, ignore_index=True)


def per_participant_summary(t):
    return {m: {"mean": float(t[m].mean()), "sd": float(t[m].std(ddof=0))} for m in ("ba", "auc", "f1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx64")
    ap.add_argument("xlsx32")
    ap.add_argument("out_dir")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = [0.1, 1, 10] if a.quick else GRID
    tables = {"64": add_control_label(load_trials(a.xlsx64)), "32": add_control_label(load_trials(a.xlsx32))}
    results, last_model = {}, None
    for (n_trials, group), pub in PUBLISHED.items():
        data = tables[n_trials][tables[n_trials].group == group].reset_index(drop=True)
        key = f"{n_trials}trials_group{group}"
        res = {"published": pub, "n_participants": int(data.participant.nunique()), "n_trials": int(len(data))}
        for variant in ("paper", "honest"):
            per_p, pooled, last_model = lopo(data, "face", variant, grid)
            pooled.to_csv(out / f"scores_{key}_{variant}.csv", index=False)
            res[variant] = {"per_participant": per_participant_summary(per_p),
                            "pooled": binary_metrics(pooled.y, pooled.score),
                            "pooled_ci95": bootstrap_ci(pooled.y.to_numpy(), pooled.score.to_numpy(), n_boot=1000)}
        ctrl, _, _ = lopo(data, "control", "paper", grid)
        res["paper_chance_baseline"] = per_participant_summary(ctrl)
        leak = random_split(data, grid)
        res["leak_random_trial_split"] = binary_metrics(leak.y, leak.score)
        results[key] = res
        r, h = res["paper"]["per_participant"], res["honest"]["per_participant"]
        print(f"{key}: published AUC {pub['auc']:.2f} BA {pub['ba']:.2f} | reproduced AUC {r['auc']['mean']:.3f} "
              f"BA {r['ba']['mean']:.3f} | honest AUC {h['auc']['mean']:.3f} BA {h['ba']['mean']:.3f} | "
              f"random-split AUC {res['leak_random_trial_split']['auc']:.3f}", flush=True)

    X = tables["32"][FEATURES].to_numpy()[:32]
    results["efficiency"] = {
        "hardware": eff.hardware_info(),
        "model_size_bytes": eff.model_size_bytes(last_model),
        "parameter_count": eff.svm_parameter_count(last_model.named_steps["clf"], len(FEATURES)),
        "inference_per_participant_32_trials_ms": eff.latency_ms(lambda: last_model.decision_function(X)),
        "inference_per_trial_ms": eff.latency_ms(lambda: last_model.decision_function(X[:1])),
        "inference_peak_memory_bytes": eff.peak_memory_bytes(lambda: last_model.decision_function(X)),
    }
    (out / "summary.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
