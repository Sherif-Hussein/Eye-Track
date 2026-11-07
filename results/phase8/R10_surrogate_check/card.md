# R10: are surrogate-innocent false-positive rates a fair proxy for real-innocent rates?

**Status: exploratory, post-review analysis. Not pre-registered. Nothing in the paper was changed.** Script: `run_r10.py` (diagnostic: `diag_mazza_shift.py`); numbers: `results.json`, `diag_mazza_shift.json`; per-participant scores: `scores_*.csv.gz`; log: `log_run.txt`.

## Question and design
Delmas et al. and Chen et al. have no innocent participants, so their FPR is measured on surrogate innocents built from the same person. Do surrogates agree with real innocents? Tested where real innocents exist: Celniak et al. (`celniak_unified`, 20 guilty / 19 innocent) and Colasanti et al. guilty vs innocent (`mazza_guilty_vs_innocent`, 78 / 79).

Pipeline reused unchanged (`src/edd/unified.py`, stored checkpoints, frozen folds): label-free features, within-person scaling, L2 logistic regression with balanced weights, C from the 9-point grid chosen in inner folds, threshold = the (floor(0.07 m)+1)-th largest training-negative statistic (flags 7% of training innocent cases), person statistic = mean item score.

**Reproduction first (passed).** From the stored fold models, test scores, inner out-of-fold scores and thresholds were recomputed; max abs differences vs the stored tables: scores 0, inner out-of-fold scores 0 (inner refits at the stored C on repeats 0-2 reproduce exactly), person statistics and thresholds below 3e-16. Table S15 reproduced: Celniak detection 8.3% [1.2, 32], FPR 8.8% [1.3, 33]; Colasanti detection 28.5% [18.6, 39.5] (S15: 19-40), FPR 9.1% [3.6, 17].

**Surrogate construction (as for Delmas).** For each participant and each of 20 random draws (seed 20261110), the probe item(s) are replaced by irrelevant item(s) of the same number, which then act as the probe; the case is scored with the same model and the same threshold. Primary variant `leave_out`: the pseudo-probe and the real probe are removed from the reference set (the reference then uses the remaining irrelevant items; this is the analogue of Delmas, where the pseudo-known faces are scored against the remaining unknown faces). Sensitivity `in_reference`: the pseudo-probe stays inside the reference.
- Celniak: 1 probe + 3 irrelevant items per slide; one irrelevant item drawn per slide as pseudo-probe, features rebuilt with the paper's formulas (rebuilt real features match the stored matrix to 2e-16).
- Colasanti: 6 target images + 3 neutral images; the z features of each target slot are rebuilt with one randomly drawn neutral image (with replacement, because there are only 3) in place of the target (rebuilt real features match exactly).
- Surrogates from guilty participants and, for comparison, from innocent participants (for innocents all items are irrelevant already, so this is the pure "same-person reuse" analogue).

**Surrogate-calibrated threshold.** As on Delmas/Chen, the training negatives are replaced by training-participant surrogates (inner out-of-fold scores, same inner folds and C; one threshold per draw, flagging 7% of them), then applied to the REAL test participants. Two versions: training surrogates from guilty participants (the Delmas situation: only guilty people exist) and from innocent participants. This needed inner refits; done for repeats 0-2 only (Celniak has exactly 3 repeats; Colasanti has 5, so its calibrated rows use repeats 0-2, where the real-innocent FPR at the nominal threshold is 8.0% [2.8, 15.8] instead of 9.1%).

Intervals: Clopper-Pearson over participants for the FPRs (surrogate rows use k rounded to an integer, so they are approximate); differences have 95% percentile bootstrap intervals over participants (1,000 resamples, seed 20261111, stored scores, no refit; guilty-surrogate vs real-innocent differences resample the two groups independently, innocent-surrogate vs real-innocent differences are paired). Draw spread = SD over the 20 draws.

## Results (primary variant `leave_out`; FPR/detection in %)

| | Celniak (n 20 / 19) | Colasanti guilty vs innocent (n 78 / 79) |
|---|---|---|
| (a) Real innocents, nominal threshold: FPR [CP] | 8.8 [1.3, 33] | 9.1 [3.6, 17] |
| (a) Detection of real guilty | 8.3 [1.2, 32] | 28.5 [18.6, 39.5] |
| (b) Surrogate FPR from guilty participants, mean over draws (draw SD; range) [CP] | 6.7 (1.8; 3.3-10.0) [0.1, 25] | 80.3 (2.8; 76.2-85.1) [70, 89] |
| (b) Surrogate FPR from innocent participants | 9.8 (2.0; 5.3-12.3) [1.3, 33] | 83.0 (3.3; 76.5-91.1) [74, 91] |
| Difference, surrogate(guilty) minus real innocent [95% bootstrap] | -2.1 [-15.2, +9.2] | +71.2 [+64.8, +77.1] |
| Difference, surrogate(innocent) minus real innocent (paired) | +1.1 [-1.0, +3.9] | +73.9 [+67.9, +79.2] |
| Real-innocent FPR, threshold calibrated on guilty surrogates [CP] (repeats 0-2) | 9.5 [1.3, 33] (draw SD 2.8) | 0.1 [0, 4.6] (draw SD 0.2) |
| ... difference from FPR at the nominal threshold | +0.7 [-4.5, +5.5] | -7.9 [-13.3, -3.2] |
| ... detection of real guilty at that threshold | 11.4 [1.2, 32] | 1.8 [0, 6.9] |
| Real-innocent FPR, threshold calibrated on innocent surrogates [CP] | 8.2 [1.3, 33] (draw SD 1.6) | 0.4 [0, 4.6] (draw SD 0.3) |
| ... detection of real guilty | 6.7 [0.1, 25] | 2.4 [0.3, 9.0] |
| Mean threshold, surrogate-calibrated vs nominal (statistic units) | 16.0 vs 22.9 (guilty surr.); 24.7 (innocent surr.) | 59.4 vs 28.1; 57.2 |

Sensitivity `in_reference` (pseudo-probe kept in the reference): Celniak surrogate FPR 5.7 (guilty) and 9.3 (innocent), real-innocent FPR under surrogate calibration 7.9 and 9.4; Colasanti surrogate FPR 92.4 and 93.3, real-innocent FPR under surrogate calibration 1.2 and 1.6 (repeats 0-2), detection 4.9 and 5.3. Conclusions are the same.

## Why the two datasets disagree
- **Celniak**: the unified model has no usable signal (person AUC 0.46, detection 8.3% at a 7% flag rate, i.e. at chance). When the score carries no information about the probe, any substitution of items leaves the score distribution unchanged, so a surrogate FPR close to 7% is guaranteed. The agreement (-2.1 pts guilty-based, +1.1 pts paired innocent-based; spread across draws 2 pts) is therefore real but uninformative: it shows the surrogate machinery is unbiased when exchangeability holds trivially, not that it works when a model has signal. The intervals (about +/-12 pts, from 19-20 people) are also too wide to exclude a difference of several points.
- **Colasanti**: surrogate FPR is 80-83% against 9% for real innocents (difference about +71 to +74 pts, interval far from 0, spread across draws 3 pts). This is not a property of the threshold but of the construction: the six target images draw far more fixation than the three neutral images (mean z of the real target features in innocents +2.0, of the neutral-as-target surrogate +0.01; diagnostic in `diag_mazza_shift.json`), so a neutral image placed in a target slot is out of distribution and is scored on the guilty side by the model (average-model score of surrogate cases +0.44 vs -0.16 real innocent and +0.16 real guilty; in-sample descriptive check). Neutral images are not exchangeable with target images in this dataset, so no valid surrogate exists from the available items; the failure is of this surrogate, not evidence on how Delmas surrogates behave. Calibrating the threshold on such surrogates pushes it far above the usable range (mean 59 vs 28), and then the real-innocent FPR is only 0.1-0.4% and detection collapses to 2%: the surrogate-calibrated threshold would be hugely too conservative.

## Verdict
Where surrogates are exchangeable with the real irrelevant cases, they track real-innocent FPR closely (Celniak: -2 to +1 pts, and a threshold calibrated on them gives 8-9.5% real-innocent FPR against 7% target and 8.8% nominal), but this test is non-diagnostic because the model has no signal there; the interval allows a few points either way. Where the substitute items differ systematically from the probe items (Colasanti, neutral images vs crime-scene images), surrogates grossly OVERSTATE the real FPR (about 80% vs 9%), and a threshold calibrated on them UNDERSTATES the real FPR (0.1-0.4% vs 7%) at the cost of near-zero detection. So the surrogate FPR is a fair proxy only if the unknown/irrelevant items are exchangeable with the probes except for knowledge; this analysis cannot confirm that for Delmas (faces; a surrogate-known set of unknown faces is plausibly exchangeable but untested here) or for Chen (two unfamiliar control faces; their mirror-image comparison). Surrogate FPRs on Delmas and Chen should be described as FPR under a same-person null that is not validated against real innocents, not as an estimate of the rate among real innocent people. Neither dataset here tests the Delmas design with a signalling model and truly exchangeable items.

## Deviations and caveats
- Inner refits (needed for surrogate calibration) only on repeats 0-2, at the stored C (C not re-tuned per surrogate draw); they reproduced the stored inner out-of-fold scores exactly. No permutation refits, one process, one thread.
- Colasanti calibrated rows use repeats 0-2 only; nominal surrogate rows use all 5 repeats (card table, first four rows); the real-innocent FPR on repeats 0-2 is 8.0% [2.8, 15.8].
- Draws: 20 per participant, shared across the guilty and innocent surrogate sets; draw SD describes substitution randomness only, not sampling uncertainty (that is the CP/bootstrap interval).
- Colasanti has only 3 neutral images for 6 target slots; neutral images are drawn with replacement. Celniak probe set size is 1 (3 irrelevant items available). Delmas draws as many unknown faces as known faces from a large pool, which these datasets cannot mimic.
- The Delmas design also uses a person-level z statistic; here the stored mean-score statistic is used, as in the paper's pipeline for these datasets. Surrogate CP intervals use rounded counts (approximate). Percentile bootstrap intervals at n=19-20 are coarse.
- Reading the verdict: a negative or null result for Colasanti reflects the surrogate construction, not a failure of the pipeline.
