"""Task M2 report tables. Source: results_item_person.csv (mc_eval.py) and cost.csv (mc_cost.py), both in this folder.
Writes results_table.csv, results.json and table_models.tex into results/phase10/model_compare/ only. No model is re-run here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")
ev = pd.read_csv(HERE / "results_item_person.csv")
co = pd.read_csv(HERE / "cost.csv")
co = co.rename(columns={"model": "model"})
keep = ["model", "setting", "n_features", "pickle_bytes", "n_params", "fit_cpu_ms", "fit_wall_ms", "score_person_cpu_ms", "score_person_wall_ms",
        "score_display4_cpu_ms", "score_display4_wall_ms", "core_share_display4_pct", "core_share_person_pct", "ops_per_item", "ops_note",
        "mem_peak_wset_growth_mib", "mem_wset_growth_mib", "mem_python_peak_mib", "items_per_person_mean"]
df = ev.merge(co[keep].add_prefix("cost_").rename(columns={"cost_model": "model"}), on="model", how="left")
df.to_csv(HERE / "results_table.csv", index=False)
(HERE / "results.json").write_text(json.dumps({"note": "Exploratory. Cost columns were measured on FULL features (13) on an Intel Core i7-8550U, one thread, with one light statistics job running (see cost.json other_load); "
                                               "they are repeated on the CAMERA rows for convenience (CAMERA has 10 features, so work per item is smaller).",
                                               "rows": json.loads(df.to_json(orient="records"))}, indent=1))


def ci(d, lo, hi):
    return f"${d:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]"


def size(b):
    return f"{b / 1024:.1f} kB" if b < 1e6 else f"{b / 1e6:.2f} MB"


def num(x):
    return f"{x:,.0f}".replace(",", "\\,")


rows = []
full = df[df.condition == "FULL"].set_index("model")
cam = df[df.condition == "CAMERA"].set_index("model")
names = {"LR": "Logistic regression", "XGB": "XGBoost", "LGBM": "LightGBM", "CAT": "CatBoost", "SVM": "RBF-kernel SVM", "ENS": "Ensemble (LR + XGBoost)"}
lines = []
for m in ["LR", "XGB", "LGBM", "CAT", "SVM", "ENS"]:
    f, c = full.loc[m], cam.loc[m]
    mark = lambda r: {"better": r"$^{*}$", "worse": r"$^{\dagger}$"}.get(r.verdict, "")  # noqa: E731
    chg_f = "reference" if m == "LR" else ci(f.item_delta, f.item_delta_lo, f.item_delta_hi) + mark(f)
    chg_c = "reference" if m == "LR" else ci(c.item_delta, c.item_delta_lo, c.item_delta_hi) + mark(c)
    lines.append(" & ".join([names[m], f"{f.item_auc:.3f}", f"{c.item_auc:.3f}", chg_f, chg_c, size(f.cost_pickle_bytes),
                             f"{f.cost_score_person_cpu_ms:.2f}", f"{f.cost_fit_cpu_ms / 1000:.2f}", f"{f.cost_core_share_display4_pct:.3f}",
                             num(f.cost_ops_per_item)]) + r" \\")
tex = r"""% Task M2 (exploratory). AUC = item AUC, one repeat of the frozen participant-disjoint folds. Change = difference from logistic regression, 95% paired participant bootstrap.
% Times: CPU, one thread, Intel Core i7-8550U, one light statistics job running.
\begin{table}[t]
\centering
\caption{Other classifiers compared with the logistic regression on the same folds (item AUC; full model: clean 30\,Hz, 13 features; camera model: webcam-level noise, 10 features, mean of 3 noise draws).
Change = difference from logistic regression with a 95\% participant-cluster bootstrap interval. Size = pickled model. Score = CPU ms to score all items of one person from features; fit = CPU s for one fit;
core = share of one CPU core if the four items of each 5\,s display are scored in real time; work = operations (logistic regression), support vectors $\times$ features (SVM) or tree nodes visited (boosted trees) per item (full model).
The ensemble averages the z-scaled scores of logistic regression and XGBoost with equal weights; it was examined after an initial look at the data and is exploratory. $^{*}$ better and $^{\dagger}$ worse than logistic regression (interval excludes zero and Holm-adjusted $p<.05$ over 10 comparisons); no mark = no reliable difference.
Times were measured on a laptop (Intel Core i7-8550U, one thread) with one light statistics job running on the machine.}
\label{tab:models}
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccrrrrr}
\toprule
Model & AUC full & AUC camera & Change, full & Change, camera & Size & Score (ms) & Fit (s) & Core (\%) & Work per item \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
(HERE / "table_models.tex").write_text(tex, encoding="utf-8")
print(df[["condition", "model", "item_auc", "item_delta", "item_delta_lo", "item_delta_hi", "item_swap_p_holm", "verdict"]].to_string())
print(tex)
