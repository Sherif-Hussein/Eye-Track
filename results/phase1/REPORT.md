# Phase 1 results: evaluation harness and re-baselines

## Delmas et al. 2024 (famous-faces CIT, EyeLink 1000 Hz, 15 participants per group)

Leave-one-participant-out within each group, as in the paper. Per-participant metrics are means over held-out participants; EER/HTER/MCC are pooled over all held-out trials.

| Set | Published AUC / BA | Reproduced AUC / BA | Honest AUC / BA | Honest EER | Honest HTER | Honest MCC | Random-split AUC | Chance-label AUC |
|---|---|---|---|---|---|---|---|---|
| 64trials_group1 | 1.00 / 0.98 | 1.000 / 0.979 | 1.000 / 0.994 | 0.002 | 0.006 | 0.981 | 0.997 | 0.600 |
| 64trials_group2 | 0.93 / 0.89 | 0.919 / 0.889 | 0.996 / 0.981 | 0.019 | 0.020 | 0.910 | 0.927 | 0.644 |
| 64trials_group3 | 1.00 / 0.96 | 1.000 / 0.957 | 1.000 / 0.998 | 0.002 | 0.002 | 0.985 | 0.999 | 0.726 |
| 32trials_group3 | 0.82 / 0.94 | 0.937 / 0.828 | 0.975 / 0.917 | 0.090 | 0.088 | 0.774 | 0.981 | 0.613 |
| 32trials_group4 | 0.99 / 1.00 | 1.000 / 0.986 | 1.000 / 0.997 | 0.007 | 0.003 | 0.979 | 1.000 | 0.613 |
| 32trials_group5 | 1.00 / 1.00 | 1.000 / 0.995 | 1.000 / 0.893 | 0.104 | 0.112 | 0.699 | 0.846 | 0.816 |

Deployment cost (last fitted SVM, 4 features): 12.0 KiB, 1,026 stored parameters, p50 1.49 ms per participant (32 trials), peak memory 9.0 KiB.
