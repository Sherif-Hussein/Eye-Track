"""Task 2.2c Part B report tables (EXPLORATORY). Reads results_item_person.csv (mco_eval.py) and repro_*.json in this folder; writes results_table.csv,
results.json and table_observed.tex here. No model is re-run."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")
ev = pd.read_csv(HERE / "results_item_person.csv")
order = {"LR": 0, "XGB": 1, "ENS": 2}
ev["o"] = ev.model.map(order)
ev["c"] = ev.condition.map({"OBS": 0, "OBS_R": 1})
ev = ev.sort_values(["c", "o"]).drop(columns=["o", "c"]).reset_index(drop=True)
cols = ["condition", "model", "label", "n_draws", "repeats", "item_auc", "item_auc_lo", "item_auc_hi", "item_delta", "item_delta_lo", "item_delta_hi", "item_swap_p", "item_swap_p_holm",
        "verdict", "item_eer", "item_hter_fpr7", "item_detection_fpr7", "item_realised_fpr7", "person_auc", "person_auc_lo", "person_auc_hi", "person_delta", "person_delta_lo",
        "person_delta_hi", "person_swap_p", "person_swap_p_holm", "person_verdict", "person_eer", "person_hter_fpr7", "person_detection_fpr7", "person_realised_fpr7",
        "extra_ens_vs_xgb_item_delta", "extra_ens_vs_xgb_item_lo", "extra_ens_vs_xgb_item_hi", "extra_ens_vs_xgb_person_delta", "extra_ens_vs_xgb_person_lo", "extra_ens_vs_xgb_person_hi"]
ev = ev[cols]
ev.to_csv(HERE / "results_table.csv", index=False)

e = ev[ev.model == "ENS"].set_index("condition")
ok = {c: bool(e.loc[c, "person_delta_lo"] > 0 and e.loc[c, "person_swap_p_holm"] < 0.05) for c in ("OBS", "OBS_R")}
app = "worth considering for the app later" if all(ok.values()) else "logistic regression stays the only live model"
repro = {c: json.loads((HERE / f"repro_{c}.json").read_text()) for c in ("OBS", "OBS_R")}
(HERE / "results.json").write_text(json.dumps({"note": "Exploratory, development data only. Repeat 0 for all three models; item AUC = mean over 3 noise draws; Holm over 4 rows, item and person separately.",
                                               "ensemble_person_level_interval_above_zero_and_holm_below_.05": ok, "app_reading": app, "reproduction": repro,
                                               "rows": json.loads(ev.to_json(orient="records"))}, indent=1))

names = {"LR": "Logistic regression", "XGB": "XGBoost", "ENS": "Ensemble (LR + XGBoost)"}
cond_name = {"OBS": "Observed webcam (full slow jitter)", "OBS_R": "Observed webcam (reduced slow jitter)"}


def ci(d, lo, hi):
    return f"${d:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]"


def p(x):
    return "--" if not np.isfinite(x) else ("$<.001$" if x < 0.001 else f"{x:.3f}")


def vt(v):
    return {"better": "better", "worse": "worse", "no reliable difference": "none", "reference": "ref."}.get(v, v)


lines = []
for c in ("OBS", "OBS_R"):
    lines.append(r"\multicolumn{11}{l}{\textit{" + cond_name[c] + r"}} \\")
    for m in ("LR", "XGB", "ENS"):
        r = ev[(ev.condition == c) & (ev.model == m)].iloc[0]
        ref = m == "LR"
        lines.append(" & ".join([names[m], f"{r.item_auc:.3f}", "reference" if ref else ci(r.item_delta, r.item_delta_lo, r.item_delta_hi), "--" if ref else p(r.item_swap_p_holm),
                                 vt(r.verdict), f"{r.item_eer:.3f}", f"{r.item_hter_fpr7:.3f}", f"{r.person_auc:.3f}",
                                 "reference" if ref else ci(r.person_delta, r.person_delta_lo, r.person_delta_hi), "--" if ref else p(r.person_swap_p_holm),
                                 vt(r.person_verdict)]) + r" \\")
tex = r"""% Task 2.2c Part B (exploratory). Item AUC = mean over 3 noise draws, repeat 0 of the frozen participant-disjoint folds, 30 Hz, dwell from samples, 10 features, no per-trial correction.
\begin{table}[t]
\centering
\caption{Logistic regression, XGBoost and their 50/50 ensemble on the simulated observed-webcam data (30\,Hz, 10 features, mean of 3 noise draws). Item AUC = known item against unknown items; person AUC = known against simulated-unknown person statistic.
Change = difference from logistic regression with a 95\% paired participant-cluster bootstrap interval. $p$ = paired score-swap $p$ after Holm adjustment over 4 comparisons (XGBoost and ensemble in two conditions), item and person level separately.
Verdict: better or worse = interval excludes zero and Holm $p<.05$; none = no reliable difference. EER = equal error rate; HTER = half total error rate at the threshold set on the training participants for 7\% false alarms.
Reduced slow jitter: the slow component of the simulated webcam jitter scaled down so that its total SD equals the measured residual SD. The ensemble was examined earlier on other noise and is exploratory.}
\label{tab:models_observed}
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccccc}
\toprule
Model & Item AUC & Change & $p$ & Verdict & EER & HTER & Person AUC & Change & $p$ & Verdict \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
(HERE / "table_observed.tex").write_text(tex, encoding="utf-8")
print(ev[["condition", "model", "item_auc", "item_delta", "item_delta_lo", "item_delta_hi", "item_swap_p_holm", "verdict", "person_auc", "person_delta", "person_delta_lo",
          "person_delta_hi", "person_swap_p_holm", "person_verdict"]].round(4).to_string())
print(ok, app)
