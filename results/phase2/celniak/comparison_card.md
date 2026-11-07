# Comparison card, Celniak et al. 2026 (eye-tracking CIT vs polygraph)

> FINAL (2026-11-02). Phase 2 screening track `celniak` is complete: 17 screening settings (3 repeats), the within-order and probe-specific permutation tests, confound controls, the non-nested reproduction and the efficiency table. All numbers are in `results.json` (keys `confound_controls`, `reproduction_paper_protocol`, `reproduction_random_slide_split_leaky`, `efficiency`, and `<idea>.permutation*`). Note on uncertainty: the "± SD" over the 3 repeats only reflects different pairings of the same 39 people into test folds, so it is small (0.003 to 0.05) and says nothing about sampling variation across people. The participant-bootstrap CIs (about ±0.15 on AUC) are the honest uncertainty. N = 39 and no difference in this card should be read as established unless its CI excludes zero.

## 1. The published study

**Citation.** Celniak W. et al. (2026). "Evaluating eye-tracking as a method for deception detection against the polygraph as deployed in practice." *Scientific Reports* 16:25352. DOI 10.1038/s41598-026-55344-9. Data: Zenodo 10.5281/zenodo.18525952.

**Task.** Mock-crime Concealed Information Test (CIT). Guilty participants carried out 7 steps (cap and gloves, key, office entry, laptop, Gmail login, copying exam questions, hiding the copy in a black bag). Innocent participants waited 15 min.

**Participants.** N = 39, aged 19–60 (M = 28.05, SD = 9.5), 56% female. Randomly assigned: 20 guilty, 19 innocent. No eye-tracking exclusions; 5 polygraph exclusions.

**Sensor.** Tobii Pro Fusion with Tobii Pro Lab, 27-inch full-HD screen at about 70 cm. The paper does not report the sampling rate. **In the released exports, 38 of 39 recordings are sampled at 250 Hz and only Recording1 at 120 Hz** (median inter-sample interval 4.0 ms vs 8.3 ms).

**Procedure.** There are 7 questions, each shown as a 6 s avatar video (in the export about 5.3–6.6 s), then a fixation cross (about 0.5 s), then a 7.5 s slide with 4 images (1 probe and 3 irrelevant). The 7-question block is repeated 3 times with item positions shuffled, giving 21 slides per participant and 819 slides in total. Half the participants did the polygraph first and half the eye tracker first. In the released data, the order is unbalanced across groups: 14/20 guilty vs 9/19 innocent did the eye tracker first.

**Features.** Per slide, for fixations, visits and glances: the total, mean, minimum and maximum duration, the count, and the duration of the first event, for the probe AOI. Each is normalised as target / (target + median of the 3 non-targets). Pupil is the mean baseline-corrected diameter, with the baseline taken from the first fixation cross of the whole session. That gives 19 features.

**Model and evaluation.** RBF-SVM, random forest and XGBoost, with and without ANOVA-F SelectKBest inside each fold (Table 2 grids). Evaluation used GroupKFold with 2 participants per test fold. Slide predictions were aggregated to the participant (the rule is not specified). Nested tuning is not described.

**Reported metrics (participant level, Table 4).**

| Configuration | Model | Accuracy | Sensitivity | Specificity | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| No feature selection | SVM-RBF | 0.90 | 0.80 | 1.00 | 0.89 | 0.90 |
| No feature selection | Random forest | 0.65 | 0.90 | 0.40 | 0.72 | 0.62 |
| No feature selection | XGBoost | 0.72 | 0.90 | 0.53 | 0.77 | 0.67 |
| With selection | **SVM-RBF** | **0.95** | **0.90** | **1.00** | **0.95** | **1.00** |
| With selection | Random forest | 0.64 | 0.90 | 0.40 | 0.72 | 0.62 |
| With selection | XGBoost | 0.74 | 1.0 | 0.5 | 0.67 | 0.75 |

Precision, MCC, EER and HTER are not reported. The text gives 94.9% (37/39) and elsewhere 92.3%. The polygraph reached 0.88 (30/34). Feature tests (Table 3, Mann–Whitney) report no statistics.

**Second 2026 comparison from the same group.** Celniak W., Słapczyńska D., Augustyniak P. (2026). "Pre-stimulus oculomotor signatures of deception in the Concealed Information Test." *Comput. Methods Programs Biomed.* 285:109472, DOI 10.1016/j.cmpb.2026.109472. It reports a participant-level SVM accuracy of 0.729 from anticipatory (avatar-question) eye movements. We read the public script as data only; it was never executed. Its features are pupil per repetition block and ratios between blocks, fixation dispersion, fixation duration on the avatar, and eye/mouth and face fixation counts. Its CV is a single StratifiedGroupKFold(5, seed 42) by recording. Hyperparameters are chosen by the mean participant AUC on the same outer folds that are reported (no inner loop), so the 0.729 is optimistic. AUC, sensitivity and specificity are not given in the abstract; we could not access the full text.

## 2. Reproduction under the paper's protocol

**What we ran** (`scripts/phase2_celniak_repro.py`, `scripts/phase2_celniak_repro_leak.py`; per-grid-point results in `reproduction_*.csv`). The paper's SVM (RBF, StandardScaler, ANOVA-F SelectKBest) on our 19 slide features (rebuilt from the exported samples), the full Table-2 grid (C and gamma 6 values each; k, which the paper does not report, in {3, 5, 7, 10, 13, all}; 216 points), `GroupKFold(20)` (19 test folds of 2 participants, 1 of 1), slide scores aggregated to the participant by mean score (and, separately, by majority vote because the paper does not say). "Tuned on all data" is implemented as picking the grid point with the best accuracy on the very folds it is scored on (no inner loop). Nothing is nested here, so every best-of-grid number below is optimistic by construction.

| Protocol variant (participant level) | Best-of-grid accuracy | Best-of-grid AUC | Median over grid (acc / AUC) |
|---|---|---|---|
| **Published (Table 4, SVM + selection)** | **0.95** | **1.00** | not applicable |
| A. GroupKFold(20), selection inside folds, tuned on the scored folds, 216 points, mean-score aggregation | 0.641 (k=all, C=10, γ=10; AUC at that point 0.553) | 0.632 | 0.205 / 0.196 |
| A, majority-vote aggregation | 0.667 | 0.718 | 0.205 / 0.200 |
| B. As A, but ANOVA selection fitted on ALL participants first (leaky selection), mean score | 0.641 | 0.634 | 0.205 / 0.204 |
| A on 5 random participant pairings (reduced 36-point grid) | 0.538 to 0.590 (5 seeds) | 0.555 to 0.621 | 0.256 to 0.436 (acc) |
| Null: A with participant labels shuffled (5 seeds, reduced 36-point grid) | 0.385 to 0.538 | 0.332 to 0.539 | 0.231 to 0.410 (acc) |
| C. Leaky: random 5-fold SLIDE split (same person in train and test), 216 points, best of grid, 3 seeds | 0.744, 0.795, 0.744 | 0.768, 0.768, 0.726 | not reported |
| Honest baseline of this card (nested, section 3) | 0.564 | 0.597 | not applicable |

**Result.** We do **not** reproduce 0.95 / 1.00 under any variant. The best optimistic participant-disjoint number (tuning on the test folds, 216 points) is accuracy 0.64 to 0.67 and AUC 0.63 to 0.72; no grid point reaches accuracy 0.90. Only a protocol that puts the same person's slides on both sides of the split (variant C, a leaky slide-level split) gets near the published level, at accuracy 0.74 to 0.80 and AUC 0.73 to 0.77, which is still below 0.95 / 1.00. Two further observations: (1) selecting the best grid point on the scored folds is worth roughly +0.08 accuracy over the honest nested figure (0.64 vs 0.56) with 216 points, but with shuffled labels the same best-of-grid selection already reaches 0.54 accuracy on a 36-point grid, so selection alone produces "above-chance" numbers from noise; (2) the **median** grid point is far *below* chance (accuracy 0.205, AUC 0.196). Many grid points (small gamma or tiny C) collapse to a near-constant predictor, and holding out whole participants shifts the training class balance against the held-out class, which turns such a predictor into a systematic anti-predictor. We did not investigate this further. It means best-of-grid on the scored folds picks a point from a distribution centred well under chance, which is another reason these numbers should not be read as accuracy. **Caveats:** our features are rebuilt from the public export and may differ from the authors' (aggregation rule, k, scaling, and feature details are not fully specified), so we cannot rule out that their exact pipeline gives higher numbers under the same protocol. We did not find a way to get to 0.95 and do not claim to know how the authors obtained it. Phase 1 (different folds and tuning) measured a +0.13 to +0.23 AUC inflation from leaky protocols, which is consistent in size with variant C here.

## 3. Honest baseline under our protocol

The paper's 19 features, slide-level RBF-SVM with SelectKBest, and a reduced Table-2 grid (k ∈ {5, 7, all}, C ∈ {0.1, 1, 10, 100}, γ ∈ {0.001, 0.01, 0.1}) are tuned in participant-grouped inner folds (StratifiedGroupKFold, 3 splits). Outer folds each hold out 1 guilty + 1 innocent participant, matched on test order where possible (`folds.json`, 3 repeats). The participant score is the mean of its 21 slide scores.

- Participant AUC **0.597 ± 0.021**, accuracy 0.564, EER 0.445, HTER 0.436. The repeat-averaged scores give AUC 0.608, 95% CI [0.41, 0.79]. The order-stratified AUC is 0.559.
- Phase 1 (full 216-point grid, different folds) gave 0.497 ± 0.021. Both are compatible with chance and far below the published AUC of 1.00.

## 4. Ideas screened (3 repeats, identical folds, participant level, mean ± SD)

"Order-stratified AUC" is computed separately within the eye-tracker-first and polygraph-first participants and then pooled, so test order cannot contribute. ΔAUC and p come from a participant bootstrap on repeat-averaged scores vs the honest baseline. Holm correction is over all 17 ideas and ablations.

| Setting | Kind | AUC | Acc | BalAcc | Sens | Spec | Prec | F1 | MCC | EER | HTER | Order-stratified AUC | ΔAUC vs baseline [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_svm_paper | baseline | 0.597 ± 0.021 | 0.564 ± 0.021 | 0.564 ± 0.022 | 0.550 ± 0.041 | 0.579 ± 0.074 | 0.582 ± 0.027 | 0.564 ± 0.017 | 0.130 ± 0.044 | 0.445 ± 0.048 | 0.436 ± 0.022 | 0.559 ± 0.024 | n/a | n/a |
| validation_accuracy_lr | reference (metadata) | 0.799 ± 0.003 | 0.684 ± 0.012 | 0.690 ± 0.012 | 0.450 ± 0.000 | 0.930 ± 0.025 | 0.873 ± 0.039 | 0.594 ± 0.009 | 0.431 ± 0.034 | 0.214 ± 0.032 | 0.310 ± 0.012 | 0.828 ± 0.004 | n/a | n/a |
| order_only | reference (confound) | 0.613 ± 0.000 | 0.615 ± 0.000 | 0.613 ± 0.000 | 0.700 ± 0.000 | 0.526 ± 0.000 | 0.609 ± 0.000 | 0.651 ± 0.000 | 0.230 ± 0.000 | 0.387 ± 0.000 | 0.387 ± 0.000 | 0.500 ± 0.000 | n/a | n/a |
| global_gaze_lr | idea (state, not probe-specific) | 0.765 ± 0.007 | 0.752 ± 0.012 | 0.751 ± 0.012 | 0.783 ± 0.024 | 0.719 ± 0.025 | 0.746 ± 0.014 | 0.764 ± 0.013 | 0.504 ± 0.024 | 0.239 ± 0.024 | 0.249 ± 0.012 | 0.713 ± 0.015 | +0.168 [-0.045, +0.400] | 0.138 (1.000) |
| minirocket_ridge | idea 7 | 0.748 ± 0.011 | 0.709 ± 0.012 | 0.707 ± 0.011 | 0.800 ± 0.041 | 0.614 ± 0.025 | 0.686 ± 0.007 | 0.738 ± 0.017 | 0.423 ± 0.028 | 0.368 ± 0.012 | 0.293 ± 0.011 | 0.695 ± 0.010 | +0.150 [-0.058, +0.368] | 0.158 (1.000) |
| global_gaze_order_resid_lr | idea 1 | 0.711 ± 0.015 | 0.667 ± 0.000 | 0.666 ± 0.001 | 0.683 ± 0.047 | 0.649 ± 0.050 | 0.674 ± 0.015 | 0.677 ± 0.015 | 0.334 ± 0.000 | 0.368 ± 0.012 | 0.334 ± 0.001 | 0.690 ± 0.023 | +0.121 [-0.082, +0.347] | 0.270 (1.000) |
| combined_order_cov_lr | idea 1 | 0.636 ± 0.053 | 0.590 ± 0.036 | 0.589 ± 0.036 | 0.633 ± 0.062 | 0.544 ± 0.025 | 0.593 ± 0.030 | 0.612 ± 0.044 | 0.179 ± 0.072 | 0.419 ± 0.044 | 0.411 ± 0.036 | 0.577 ± 0.055 | +0.034 [-0.211, +0.279] | 0.812 (1.000) |
| combined_lr | ideas 2–5 | 0.625 ± 0.046 | 0.590 ± 0.042 | 0.589 ± 0.041 | 0.633 ± 0.094 | 0.544 ± 0.050 | 0.592 ± 0.034 | 0.610 ± 0.060 | 0.180 ± 0.082 | 0.385 ± 0.042 | 0.411 ± 0.041 | 0.577 ± 0.048 | +0.037 [-0.208, +0.282] | 0.804 (1.000) |
| orient_lr | idea 4 | 0.616 ± 0.010 | 0.538 ± 0.036 | 0.537 ± 0.036 | 0.583 ± 0.047 | 0.491 ± 0.025 | 0.546 ± 0.032 | 0.564 ± 0.039 | 0.075 ± 0.073 | 0.454 ± 0.032 | 0.463 ± 0.036 | 0.591 ± 0.013 | +0.013 [-0.232, +0.253] | 0.914 (1.000) |
| combined_order_resid_lr | idea 1 | 0.572 ± 0.052 | 0.564 ± 0.063 | 0.564 ± 0.062 | 0.583 ± 0.103 | 0.544 ± 0.025 | 0.569 ± 0.056 | 0.575 ± 0.079 | 0.128 ± 0.124 | 0.411 ± 0.001 | 0.436 ± 0.062 | 0.532 ± 0.050 | -0.032 [-0.269, +0.213] | 0.815 (1.000) |
| diff_order_resid_lr | idea 1 | 0.556 ± 0.016 | 0.538 ± 0.036 | 0.538 ± 0.035 | 0.550 ± 0.071 | 0.526 ± 0.000 | 0.548 ± 0.030 | 0.548 ± 0.050 | 0.077 ± 0.071 | 0.419 ± 0.044 | 0.462 ± 0.035 | 0.568 ± 0.029 | -0.039 [-0.290, +0.205] | 0.780 (1.000) |
| diff_lr | idea 5 | 0.532 ± 0.030 | 0.487 ± 0.063 | 0.486 ± 0.062 | 0.517 ± 0.085 | 0.456 ± 0.050 | 0.497 ± 0.061 | 0.506 ± 0.073 | -0.027 ± 0.125 | 0.487 ± 0.075 | 0.514 ± 0.062 | 0.507 ± 0.024 | -0.074 [-0.316, +0.161] | 0.531 (1.000) |
| diff_order_cov_lr | idea 1 | 0.514 ± 0.032 | 0.479 ± 0.064 | 0.478 ± 0.064 | 0.500 ± 0.071 | 0.456 ± 0.066 | 0.491 ± 0.060 | 0.495 ± 0.064 | -0.044 ± 0.128 | 0.513 ± 0.021 | 0.522 ± 0.064 | 0.473 ± 0.040 | -0.082 [-0.311, +0.150] | 0.492 (1.000) |
| pupil_lr | idea 2 | 0.498 ± 0.035 | 0.436 ± 0.036 | 0.438 ± 0.036 | 0.350 ± 0.041 | 0.526 ± 0.043 | 0.437 ± 0.044 | 0.389 ± 0.041 | -0.126 ± 0.074 | 0.487 ± 0.095 | 0.562 ± 0.036 | 0.473 ± 0.035 | -0.118 [-0.418, +0.205] | 0.440 (1.000) |
| paper_mean_lr | model control | 0.449 ± 0.023 | 0.444 ± 0.048 | 0.445 ± 0.048 | 0.417 ± 0.062 | 0.474 ± 0.043 | 0.453 ± 0.052 | 0.434 ± 0.057 | -0.110 ± 0.096 | 0.547 ± 0.065 | 0.555 ± 0.048 | 0.444 ± 0.024 | -0.163 [-0.418, +0.111] | 0.226 (1.000) |
| diff_quality_lr | idea 6 | 0.441 ± 0.049 | 0.427 ± 0.060 | 0.427 ± 0.060 | 0.450 ± 0.082 | 0.404 ± 0.066 | 0.441 ± 0.059 | 0.445 ± 0.070 | -0.147 ± 0.121 | 0.564 ± 0.075 | 0.573 ± 0.060 | 0.414 ± 0.058 | -0.168 [-0.392, +0.055] | 0.132 (1.000) |
| prestim_lr | idea 3 (CMPB idea) | 0.430 ± 0.017 | 0.427 ± 0.024 | 0.426 ± 0.024 | 0.483 ± 0.062 | 0.368 ± 0.043 | 0.445 ± 0.024 | 0.462 ± 0.041 | -0.150 ± 0.047 | 0.598 ± 0.012 | 0.574 ± 0.024 | 0.407 ± 0.031 | -0.174 [-0.439, +0.103] | 0.209 (1.000) |
| fixdur_only_lr | ablation (post hoc) | 0.777 ± 0.024 | 0.709 ± 0.024 | 0.711 ± 0.024 | 0.667 ± 0.024 | 0.754 ± 0.025 | 0.741 ± 0.026 | 0.702 ± 0.025 | 0.422 ± 0.049 | 0.273 ± 0.044 | 0.289 ± 0.024 | 0.733 ± 0.018 | +0.187 [-0.016, +0.411] | 0.073 (1.000) |
| global_gaze_no_fixdur_lr | ablation (post hoc) | 0.634 ± 0.012 | 0.632 ± 0.024 | 0.630 ± 0.024 | 0.733 ± 0.047 | 0.526 ± 0.000 | 0.619 ± 0.015 | 0.671 ± 0.028 | 0.267 ± 0.052 | 0.469 ± 0.011 | 0.370 ± 0.024 | 0.570 ± 0.022 | +0.037 [-0.129, +0.197] | 0.677 (1.000) |

**What each idea is** (code: `scripts/phase2_celniak_features.py`, `scripts/phase2_celniak_screen.py`). All participant-level ideas use median imputation, scaling, SelectKBest (k ∈ {3, 10, all}) and L2 logistic regression (C ∈ {0.1, 1}, balanced), all tuned in grouped inner folds.

- **Idea 1, order control.** Folds are matched on order. We add order as a covariate, or residualise features on order within training folds (in-fold group means). We also report the order-stratified AUC for every row.
- **Idea 2, pupil.** Per-eye validity masking, a dilation-speed MAD filter, 50 ms blink padding, eye merging with an inter-eye offset, interpolation of gaps ≤ 500 ms, and 5-sample smoothing. The subtractive baseline comes from each trial's own fixation cross. Windows are 0–800, 800–1500 and 1500–7500 ms, plus early minimum, late peak, habituation from repetition 1 to 3, and per-item pupil (probe minus irrelevant) while the item is fixated.
- **Idea 3, pre-stimulus (video + cross).** Gaze dispersion, fixation rate and duration, saccades, blink rate, pupil level and slope, anticipatory dilation from cross to video, and CMPB-style ratios between repetition blocks.
- **Idea 4, orienting.** First fixation on the probe, an excess over the participant's own quadrant bias and its absolute value, probe entry latency, and the probe-minus-irrelevant first-visit duration.
- **Idea 5, within-person differential.** Position-corrected and within-person z-scored probe-minus-irrelevant effects for 7 metrics: mean, absolute value (over- or under-attention), t statistic, and sign consistency across the 3 repetitions. Also the probe dwell rank and its absolute deviation.
- **Idea 6, data quality.** Idea 5 recomputed after dropping slides with valid fraction below a threshold of 0, 0.6 or 0.8, with the threshold tuned in inner folds.
- **Idea 7, MiniRocket.** 2016 kernels on 30 Hz per-slide series (pupil, gaze x/y, probe hit, irrelevant hit), with ridge alpha tuned in grouped inner folds. Slide scores are averaged per participant.
- **global_gaze.** Probe-agnostic looking totals per slide: fixation count and mean duration, saccades, blinks, scan path, total AOI dwell, valid fraction. We added it to test the Phase 1 "total looking" confound.

**Probe-specific (CIT) result.** Every feature set that depends on which item is the probe stays at chance under the honest protocol. The best is orient_lr at AUC 0.616 (order-stratified 0.591), Δ vs baseline +0.013 with p = 0.91. The within-person differentials (idea 5) give 0.53 and the pupil 0.50. Pre-stimulus features (idea 3, the CMPB idea) give 0.43, so we do not reproduce the CMPB 0.729 under nested evaluation.

**Order diagnostic.** We predicted the test order instead of guilt, with the same pipeline and 1 repeat. Order AUC by feature set: paper features 0.72, global gaze 0.71, pre-stimulus 0.75, diff 0.64, orient 0.49, pupil 0.45. Session-level looking behaviour therefore carries order information, which is exactly why order has to be controlled.

## 5. Best candidate vs published vs honest baseline

**Best candidate: global_gaze_lr** (probe-agnostic looking totals with L2 logistic regression).

| | Participant AUC | Accuracy | EER | HTER |
|---|---|---|---|---|
| Published SVM (non-nested) | 1.00 | 0.95 | not reported | not reported |
| Honest baseline (paper method, nested) | 0.597 ± 0.021; repeat-avg 0.608 [0.41, 0.79] | 0.564 | 0.445 | 0.436 |
| **global_gaze_lr** | **0.765 ± 0.007; repeat-avg 0.776 [0.61, 0.92]** | **0.752 ± 0.012** (repeat-avg 0.769 [0.64, 0.90]) | **0.239 ± 0.024** (0.205 [0.10, 0.41]) | **0.249 ± 0.012** (0.230 [0.10, 0.36]) |
| Order alone (confound) | 0.613 | 0.615 | 0.387 | 0.387 |
| Validation-accuracy metadata alone | 0.799 ± 0.003 | 0.684 | 0.214 | 0.310 |

Bracketed values are participant-bootstrap 95% CIs (2000 draws) on repeat-averaged scores.

- **Delta vs honest baseline:** +0.168 AUC, 95% CI [−0.045, +0.400], p = 0.138, Holm p = 1.0. Under the protocol's winner rule this is **no reliable difference**.
- **Delta vs order alone:** +0.163 [−0.036, +0.364], p = 0.097. **Delta vs validation metadata:** −0.029 [−0.255, +0.200], p = 0.83.
- **Permutation (labels shuffled between participants, full nested pipeline refitted, 50 shuffles, repeat-0 folds):** observed AUC 0.771, null mean 0.459, null 95th percentile 0.677, **p = 0.020**.
- **Permutation within test-order strata** (labels shuffled only inside the eye-tracker-first and polygraph-first groups, so the guilt-order association is kept in the null; 100 shuffles, repeat-0 folds, full nested pipeline refitted): observed AUC 0.771, null mean 0.470 (SD 0.137), null 95th percentile 0.635, **p = 0.020** (the smallest possible p with 100 shuffles is 0.0099). The null is not shifted upward by order (0.470 vs 0.459 for the unrestricted shuffle), so the guilt effect is not explained by the group imbalance in test order. This is one 20-fold run on repeat-0 folds.
- **Best probe-specific idea, orient_lr, permutation** (labels shuffled between participants, 50 shuffles, repeat-0 folds): observed AUC 0.629 (repeat-0; 0.616 ± 0.010 over the 3 repeats), null mean 0.507 (SD 0.090), null 95th percentile 0.646, **p = 0.098**. It is not distinguishable from chance; its Holm-adjusted p against the honest baseline is 1.0 (section 4).
- **Versus published:** −0.235 AUC and −0.20 accuracy. The published number is not reachable under participant-disjoint nested evaluation with any feature family we tried.

**Is it just the test-order confound? Evidence: no, order does not explain it.**

1. Within-order AUC stays at 0.713 ± 0.015 (order alone scores 0.500 on this metric by construction).
2. Residualising every feature on order inside the training folds leaves AUC at 0.711 ± 0.015 (strat 0.690).
3. Univariate and descriptive: mean slide fixation duration separates the groups inside both order strata (guilty lower: AUC 0.66 among eye-tracker-first, 1.00 among polygraph-first; full sample 0.83). It also separates them in each repetition (0.77, 0.82, 0.72).

4. The within-order permutation (above) gives p = 0.020 with a null that is not inflated by order.

The order diagnostic shows the same features also predict order (0.71). So order and guilt both shift this behaviour, and the guilt shift survives conditioning on order.

**Confound controls for global_gaze_lr** (takeover stage; `scripts/phase2_celniak_confounds.py`, `src/edd/celniak_confounds.py`, `confound_table.md`, `stage_confounds.json`). Same frozen folds and 3 repeats. Every adjustment is fitted on the training participants of each outer fold only: the OLS residualisation, the covariate scaling and selection, the imputation (training median; Recording7 has no validation value) and the quality threshold. Validation error enters as its logarithm because it is heavy-tailed (0.2 to 2.84 degrees). The quality measure is the recording's mean valid-sample fraction (`valid_frac`); the exclusion threshold is chosen from {0, 0.5, 0.6, 0.7} in the training fold by participant-grouped inner CV (3 splits; the candidate threshold is applied to the inner training part only, and the inner score is AUC on all inner validation participants).

| Variant | AUC (mean ± SD, 3 repeats) | Repeat-avg AUC [95% CI] | Acc | EER | HTER | Order-stratified AUC | Δ vs global_gaze_lr [95% CI], p | Share of above-chance AUC kept |
|---|---|---|---|---|---|---|---|---|
| global_gaze_lr (unadjusted) | 0.765 ± 0.007 | 0.776 [0.611, 0.924] | 0.752 | 0.239 | 0.249 | 0.713 | n/a | 100% |
| validation error alone (reference) | 0.799 ± 0.003 | 0.805 [0.642, 0.942] | 0.684 | 0.214 | 0.310 | 0.828 | +0.029 [−0.200, +0.255], p=0.83 (val minus gaze) | 113% |
| (a1) + validation error as covariate (`gg_val_cov`) | 0.798 ± 0.003 | 0.818 [0.658, 0.947] | 0.786 | 0.205 | 0.214 | 0.754 | +0.042 [−0.079, +0.150], p=0.50 | 113% |
| (a2) residualised on validation error (`gg_val_resid`) | 0.717 ± 0.023 | 0.726 [0.553, 0.876] | 0.718 | 0.325 | 0.284 | 0.661 | −0.050 [−0.145, +0.042], p=0.31 | 82% |
| (b) low-quality training recordings dropped, threshold chosen in-fold; all test participants scored (`gg_qual_train`) | 0.765 ± 0.019 | 0.747 [0.568, 0.905] | 0.778 | 0.223 | 0.225 | 0.681 | −0.029 [−0.113, +0.042], p=0.49 | 100% |
| (a2 + b) (`gg_val_resid_qual_train`) | 0.731 ± 0.046 | 0.734 [0.555, 0.895] | 0.744 | 0.333 | 0.259 | 0.670 | −0.042 [−0.137, +0.039], p=0.34 | 87% |
| (c1) Recording1 (120 Hz) removed from training and evaluation (`gg_drop_rec1`; 38 participants) | 0.735 ± 0.003 | 0.740 [0.576, 0.895] | 0.746 | 0.254 | 0.254 | 0.684 | −0.036 [−0.086, +0.000], p=0.06 | 89% |
| (c2) all 250 Hz recordings decimated to about 125 Hz, features recomputed (`gg_125hz`) | 0.764 ± 0.010 | 0.768 [0.603, 0.918] | 0.726 | 0.248 | 0.274 | 0.708 | −0.008 [−0.053, +0.026], p=0.73 | 100% |

"Share of above-chance AUC kept" is (AUC − 0.5) / (0.765 − 0.5), a rough summary and not a statistic. No row differs reliably from unadjusted global_gaze_lr (all CIs include zero, apart from c1, whose upper bound is exactly 0.000).

**(b) in detail: does excluding low-quality recordings explain the effect? No, the effect gets stronger.** The threshold chosen over the 60 outer folds (3 repeats x 20) was 0 in 11 folds, 0.5 in 16, 0.6 in 22 and 0.7 in 11. Evaluating only the test participants whose `valid_frac` is at least the threshold chosen in their own fold (on average 33 of 39 retained, 17.3 of them guilty): the quality-trained model reaches AUC **0.816 ± 0.012**, the unadjusted global_gaze_lr on those same participants and repeats 0.795 ± 0.015, and the validation-error reference 0.776 ± 0.001. The combined a2 + b model on its retained 32 participants gives 0.788 ± 0.015 against 0.805 ± 0.024 for the unadjusted model on the same people. As a fixed, a-priori sensitivity (no fitting; the nested out-of-fold scores are restricted afterwards; the cut-offs were fixed before looking and not tuned):

| Subset of participants | n (guilty) | global_gaze_lr AUC [95% CI] | validation error alone | honest baseline SVM |
|---|---|---|---|---|
| all | 39 (20) | 0.765 [0.61, 0.92] | 0.799 [0.64, 0.94] | 0.597 [0.41, 0.79] |
| valid_frac ≥ 0.5 | 33 (17) | 0.817 [0.65, 0.96] | 0.760 [0.59, 0.92] | 0.566 [0.37, 0.78] |
| valid_frac ≥ 0.6 | 30 (15) | 0.803 [0.64, 0.95] | 0.757 [0.56, 0.93] | 0.566 [0.36, 0.78] |
| valid_frac ≥ 0.7 | 25 (12) | 0.803 [0.63, 0.97] | 0.716 [0.49, 0.92] | 0.532 [0.28, 0.76] |

(The first number in each cell is the mean per-repeat AUC; the bracket is the participant-bootstrap CI of the repeat-averaged scores.) Among clean recordings the gaze effect stays at about 0.80 while the validation-error signal drops to 0.72 to 0.76 (the two worst validations, Recordings 17 and 20 at 2.05 and 2.84 degrees, are also among the lowest-quality recordings). Dropping the 6 to 14 lowest-quality recordings therefore does not remove the gaze effect, and the earlier concern that low-validity recordings drive it is **not supported** by this test. CIs at n = 25 to 33 are about ±0.16, so the cut-offs cannot be ranked against each other.

**(c) sampling rate.** Recording1 is the only 120 Hz recording (guilty, eye-tracker first, mean slide fixation 284 ms, rank 18 of 39 from the shortest, valid fraction 0.83, out-of-fold score rank 21 of 39): it is not an outlier on any variable that carries the effect. Removing it from training and evaluation lowers AUC by 0.03 (0.735, against 0.765 for the stored scores of the same 38 people; CI of the paired difference [−0.086, +0.000]). We read this as the sensitivity of a 39-person fit to losing one person and not as a sampling-rate artefact, but we did not run the matching leave-one-other-out comparison, so that reading is untested. Recomputing every feature from raw samples after decimating all 250 Hz recordings by 2 (the exported Tobii fixation and saccade labels are kept, so this tests our feature code and timing arithmetic and not Tobii's own I-VT classification at 120 Hz) changes the seven global features very little (Spearman with the originals ≥ 0.991; slide fixation duration 0.9996 with a median change of +0.001%; blink count has a median +20% change because brief blinks are missed) and leaves the nested AUC at 0.764 ± 0.010 (Δ −0.008 [−0.053, +0.026]). The worst-case timing bias of one 120 Hz against one 250 Hz fixation is the sample-interval difference, 4.3 ms, or 1.5% of the median fixation duration. **Sampling rate does not matter for this analysis.**

**(d) how much AUC remains.** After the validation-error adjustment (a2): 0.717 (82% of the above-chance AUC; repeat-avg 0.726 [0.553, 0.876]). After quality exclusion (b): 0.765 on all test participants and 0.816 on the clean ones. After both (a2 + b): 0.731 on all test participants (87%) and 0.788 on the 32 retained. Adding validation error as a covariate (a1) gives 0.798, the same as validation error alone (0.799), so **at N = 39 the gaze features add no measurable information beyond the pre-test validation error** (Δ +0.042 [−0.079, +0.150]). Conversely the gaze signal survives removal of a linear validation-error component and of low-quality recordings with a loss of about 0.03 to 0.05 AUC, so it is not reducible to those two covariates in this sample either. Both statements fit if the two measures reflect partly the same session-level state; these data cannot separate them.

**Why it works (mechanism; ablation and `mechanism_checks.json`).**

- The ablations isolate one variable. Mean fixation duration alone (slide, video, cross) gives 0.777 ± 0.024. Removing slide fixation duration from global_gaze drops AUC to 0.634 ± 0.012.
- Guilty participants have shorter mean slide fixations (median 257 vs 321 ms) and also shorter fixations during the question video (289 vs 399 ms).
- The difference is **not on the items**. Fixation duration on items is the same (246 vs 248 ms) and identical on the probe (239 vs 238 ms). It comes from **off-item viewing**:
  - innocent participants spend 28% of fixation time off the four items vs 16% for guilty (about 5.3 vs 3.2 long off-item fixations per slide);
  - guilty participants make more on-item fixations (15.8 vs 11.9 per slide).
- The pattern is consistent with (it was not tested directly) guilty participants **actively searching all four items for the one they know**, while innocent participants dwell longer on the centre or empty space. That is a knowledge- or arousal-related state effect. It is **not** a probe-specific CIT effect: it cannot tell which item is the crime item, and the probe-vs-irrelevant features are at chance.

**Caveats (important for the paper).**

1. **Session-level confounds are strong in this dataset.** The Tobii post-calibration validation error, recorded before the CIT begins, predicts guilt at AUC 0.80–0.82 (median 0.63° guilty vs 0.41° innocent; 0.84 within each order stratum). The best eye-movement candidate does not beat it (Δ −0.03), and adding it as a covariate leaves the gaze model at 0.798 (no gain). Anything that differs by session (time since the mock crime, physical activity, arousal, posture, experimenter or setup) can drive group separation without any concealed-information mechanism.
2. **Data quality (tested).** Within guilty participants, fixation duration correlates with valid-sample fraction (Spearman 0.71); three of the four shortest-fixation recordings are guilty with pupil validity ≤ 14% (17, 20, 12). We tested whether this drives the result (section 5, controls (b) and the a-priori cut-offs): excluding low-quality recordings with an in-fold threshold, or at fixed valid-fraction cut-offs of 0.5 to 0.7, leaves the AUC at 0.79 to 0.82. Quality does not explain the effect in this sample.
3. **Selection.** global_gaze was added to test the Phase 1 "total looking" confound, not as a pre-registered CIT idea. The fixation-duration ablations were defined after the full-sample univariate table was inspected. They explain the result and are not candidates.
4. **Generalisation.** A state signal would also fire for anxious innocent suspects in the field. It is not evidence of recognising crime details.
5. Small N (39) and wide CIs ([0.61, 0.92]). Recordings 25 and 27 share the label "Participant25" (treated as distinct, as in the paper's N = 39).

## 6. Efficiency

Measured with `edd.efficiency` (`scripts/phase2_celniak_efficiency.py`, `stage_efficiency.json`). Hardware: Intel64 Family 6 Model 142 (8 logical CPUs), Windows 11, Python 3.14.6, **one CPU thread** (threadpoolctl limit 1). Another low-priority job was running on the machine, so absolute times are likely a little pessimistic. Both models were fitted on all 39 participants with the screening grids; feature extraction starts from the cached raw samples of Recording10 (250 Hz, 101,546 samples = 406 s of recording). The unit of decision is the participant, which needs the whole session (21 slides, about 6.8 minutes of recording), so no decision can be faster than the session length; the table gives the processing cost after the data exist. "FPS" is the sample-processing rate; the real-time factor is that rate divided by the 250 Hz sensor rate.

| Measure | Published method (our SVM re-implementation, honest baseline) | global_gaze_lr (best candidate) |
|---|---|---|
| Final model | RBF-SVM, C=1, γ=0.1, all 19 features, 717 support vectors | L2 logistic regression, C=1, 3 of 7 features (slide fixation duration, blink count, scan path; standardised coefficients −1.13, +0.16, +0.62) |
| Inference latency per participant, p50 / p95 (ms) | 8.46 / 9.46 (21 slides) | 3.44 / 4.33 |
| Inference latency per slide, p50 / p95 (ms) | 3.06 / 3.26 | not defined (one decision per session) |
| Throughput | 3,913 slides/s = 186 participants/s | 11,193 participants/s |
| Peak inference memory (Python allocations, KiB) | 14.2 | 10.1 |
| Model size (pickled, KiB) | 122.9 | 1.8 |
| Stored parameters | 14,379 | 25 |
| Feature extraction per recording, p50 / p95 (ms) | 3,049 / 3,124 | full screening pipeline: 4,604 / 4,867; lean (the 7 global features only): 2,658 / 3,235 |
| Extraction rate, samples/s (FPS-equivalent) | 33,307 | full: 22,055; lean: 38,203 |
| Real-time factor vs the 250 Hz sensor | 133x | full: 88x; lean: 153x |
| Extraction peak memory (Python allocations, MiB) | 83.5 | full: 9.6; lean: 4.1 |
| End-to-end per participant, p50 (ms) | 3,057 | lean: 2,661 |

Notes. Peak memory is the peak of Python/NumPy allocations measured with `tracemalloc` and excludes the interpreter and library baseline. The lean extractor (in the efficiency script) reproduces the stored global features exactly (maximum relative difference 0.0, checked on Recording10); the screening used the full pipeline, which also builds pupil, item and pre-stimulus features that global_gaze_lr does not need. The final-fit model above (3 features) is the all-data refit; inside the CV, k and C were tuned per fold. Extraction dominates the cost for both methods (99.7% or more of the end-to-end time); the classifiers themselves are negligible. Measured on one recording on one machine; the p95 over 10 repeats of the extraction is a rough tail estimate.

## 7. Recommendation (final)

- **For the paper's claim (eye-tracking CIT detection):** nothing probe-specific beats chance reliably. This is a valid negative result. The published 0.95 / 1.00 does not survive participant-disjoint nested evaluation (honest 0.50 to 0.60), is not reproduced even when the grid point is chosen on the scored folds (best accuracy 0.64 to 0.67, AUC 0.63 to 0.72), and is approached only by a leaky slide-level split (accuracy 0.74 to 0.80). The CMPB 0.729 pre-stimulus result is not reproduced (0.43). The best probe-specific idea, orient_lr, has permutation p = 0.098 and AUC 0.62.
- **The only above-chance signal** is general looking behaviour (global gaze, mainly shorter fixations in guilty participants, from off-item viewing): AUC 0.765, permutation p = 0.020 (also 0.020 when shuffling within test-order groups), order-stratified AUC 0.71. It is not significantly better than the honest baseline after Holm correction (Δ +0.168, CI [−0.045, +0.400]) and it does not beat the pre-test validation error (0.799), which it adds nothing to (covariate model 0.798). It is not explained by test order, by low-quality recordings or by the single 120 Hz recording, and it keeps 82% of its above-chance AUC after removing the linear effect of validation error. It remains a probe-agnostic state or search-behaviour difference that cannot identify the crime item, and it could also fire for anxious innocent suspects.
- **Worth a full run?** Only as a confound analysis, not as a CIT detector, and only with more data. At N = 39 the CIs are about ±0.15 and no control comparison can be resolved. The decisive missing information is why guilty and innocent sessions differ in validation error (0.63 vs 0.41 degrees, recorded before the test).
- **Next:** (1) ask the authors for session timestamps, time since the mock crime, and experimenter and set-up logs, and for the exact feature pipeline and aggregation rule behind Table 4; (2) a larger or independent sample in which guilt is assigned to sessions in counterbalanced order, so that session-level state and guilt are not confounded; (3) if a CIT claim is wanted, evaluate probe-specific features on that sample under the same nested, participant-disjoint protocol with the within-order permutation as the null.
