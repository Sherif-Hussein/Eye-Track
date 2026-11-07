"""Render markdown tables for the chen comparison card from results/phase2/chen (ideas_table.csv,
results.json). Writes results/phase2/chen/tables_screening.md (UTF-8). No computation of its own."""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results" / "phase2" / "chen"
d = pd.read_csv(OUT / "ideas_table.csv")
pc = json.load(open(OUT / "results.json"))["paired_comparisons"]
order = ['B0_size_peak', 'B0_rate_peak', 'I2a_bins_pupil', 'I2b_bins_fusion', 'I2c_paperwin_pupil', 'I3a_pnorm_pupil',
         'I3b_pnorm_fusion', 'I4a_minirocket_fusion', 'I4b_minirocket_fusion_pnormfeat', 'A0_gapinterp',
         'A1_noblinkrecon', 'A2_divisive', 'A3_base200']


def f(r, k):
    return f"{r[k + '_mean']:.3f} ± {r[k + '_sd']:.3f}"


lines = []
for e in ('exp1', 'exp2'):
    x = d[d.exp == e].set_index('idea')
    lines += [f"\n**{e}, trial level** (pooled out-of-fold trials, familiar vs control; mean ± SD over 3 repeats)\n",
              "| Idea | Acc | BalAcc | Sens | Spec | MCC | AUC | EER | HTER |", "|---|---|---|---|---|---|---|---|---|"]
    for i in order:
        lines.append(f"| {i} | " + " | ".join(f(x.loc[i], k) for k in ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'mcc', 'auc', 'eer', 'hter']) + " |")
    lines += [f"\n**{e}, per participant** (paper's one-tailed t-test on the model's scores; innocent surrogate = control 1 vs control 2; mean ± SD over 3 repeats)\n",
              "| Idea | Detected at alpha .05 | FPR ctrl 1 vs 2 | Detection at calibrated threshold | FPR at calibrated threshold | Person AUC | Person EER | Person HTER |", "|---|---|---|---|---|---|---|---|"]
    for i in order:
        lines.append(f"| {i} | " + " | ".join(f(x.loc[i], k) for k in ['detection_rate', 'fpr_c1c2', 'detection_calibrated', 'fpr_calibrated', 'person_auc', 'person_eer', 'person_hter']) + " |")
    for base in ('B0_rate_peak', 'B0_size_peak'):
        lines += [f"\n**{e}, paired against {base}** (repeat-averaged scores; Holm over the 11 non-baseline entries; p raw (Holm))\n",
                  "| Idea | Δ within-person AUC | Wilcoxon p | detections gained/lost | McNemar p |", "|---|---|---|---|---|"]
        for i, v in pc[e][f'vs_{base}'].items():
            lines.append(f"| {i} | {v['delta_within_auc_mean']:+.3f} | {v['wilcoxon_p']:.3f} ({v['wilcoxon_p_holm']:.3f}) | {v['detect_gained']}/{v['detect_lost']} | {v['mcnemar_p']:.3f} ({v['mcnemar_p_holm']:.3f}) |")
(OUT / "tables_screening.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("written", len(lines))
