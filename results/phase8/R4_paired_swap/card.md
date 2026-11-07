# R4: paired score-swap permutation test of the difference (fixed pipeline minus leakage-free baseline)

## What was done
For each of the seven planned comparisons the stored out-of-fold scores of the fixed pipeline (`results/phase4/unified/oof/`)
and of the leakage-free baseline (`results/phase2/...`, `results/phase3/chen_confirm/checkpoints`) were paired on the same
participants. The planned statistic was first reproduced from the stored scores (all observed values match: True, see
`results.json` -> `reproduction`). Then, in each of 2000 permutations (seed 20261105), each participant independently
(probability 0.5) swapped which method their scores belong to (all items and repeats of that participant together; for Delmas
a participant id appearing in two sets is swapped in both), the statistic was recomputed for both methods, and the difference taken.
Two-sided p = (1 + #|diff_perm| >= |diff_obs|) / (1 + 2000). No model was refitted; one process, one thread.

Statistics: Delmas = image-level AUC on repeat-averaged scores, mean over the six group sets. Celniak, Preti (both),
Colasanti (both) = participant-level AUC of repeat-averaged participant scores. Chen = participant detection at the 7% threshold
(see deviations), plus participant AUC as a secondary check.

## Scale handling
Baseline and pipeline scores sit on different scales (for example Colasanti baseline logits are in the hundreds), so swapping raw
scores is not meaningful for AUC statistics. Rank version: within each method and repeat, scores are converted to normalised
ranks (for Delmas across the image rows of each set; for participant AUC across participants; for Chen AUC across all
participant-by-kind statistics), then averaged over repeats, before swapping. The raw version (planned statistic, reproduces Table 2)
is the headline; the rank version is shown next to it. The two versions agree for Delmas, Chen and Preti with RT, but differ for Celniak, Preti eyes only and Colasanti group 2 vs 3
(see table), where the unified model's logit scale changes between folds (tuned C) and rank averaging per repeat removes that effect.

## Results (Holm over the family of 7)
Delmas row: the non-inferiority one-sided p (H0: difference <= -0.02, swap null shifted by the margin) takes the family's place; its
two-sided p is also shown. Rank columns: observed difference / two-sided p (/ NI p for Delmas).

| Comparison | Observed diff | Two-sided p (raw) | NI one-sided p (raw) | Holm p (raw) | Rank-transformed (diff / p) | Holm p (rank) |
|---|---|---|---|---|---|---|
| 1 Delmas image AUC (mean of 6 sets) | +0.000 | 0.9880 | 0.0275 | 0.185 | -0.001 / 0.9350 / NI 0.0360 | 0.216 |
| 2 Chen detection at FPR<=7% (calibrated) | +14.9 pts | 0.0590 | n/a | 0.295 | n/a (binary decisions) | same |
| 3 Celniak participant AUC | -0.147 | 0.1124 | n/a | 0.450 | -0.154 / 0.0550 | 0.275 |
| 4 Preti eyes only participant AUC | +0.189 | 0.0265 | n/a | 0.185 | +0.184 / 0.0070 | 0.049 |
| 5 Preti with RT participant AUC | +0.036 | 0.2259 | n/a | 0.678 | +0.038 / 0.0915 | 0.275 |
| 6 Colasanti group 2 vs 3 participant AUC | -0.089 | 0.2589 | n/a | 0.678 | +0.009 / 0.9015 | 0.902 |
| 7 Colasanti guilty vs innocent participant AUC | +0.059 | 0.2904 | n/a | 0.678 | +0.073 / 0.1674 | 0.335 |

Secondary (not in the family): Chen participant AUC, raw diff +0.085, two-sided p 0.0315;
rank version diff +0.085, p 0.0310.

Holm if the Delmas two-sided p is used in place of the non-inferiority p (raw scores): 1: 0.988, 2: 0.354, 3: 0.562, 4: 0.185, 5: 0.904, 6: 0.904, 7: 0.904.

## Deviations and notes
- Chen detection: the 7% thresholds were calibrated on training participants inside each fold and cannot be recomputed under
  swapping; the stored per-participant calibrated decisions (mean over the 3 repeats) were exchanged between methods, which keeps
  the planned statistic. Participant AUC on the same swap masks is reported as a secondary check. The baseline for Chen is the paper's rate rule.
- The test is of the difference under exchangeability; with 2,000 permutations the smallest attainable p is 0.00050, below Holm's first-step threshold (.0071).
- Observed differences are the stored ones, so the reproduction is exact. Colasanti baselines have 3 repeats and the pipeline 5 (as in the planned statistic); the participant's repeat-averaged scores are swapped as a block.
- Rank-transformed observed differences differ from the planned ones because ranking within repeat changes scale before averaging; they are reported only as a scale-robustness check.
- Runtime: 24 s.
