# Preti et al. (2026) fake-identity lying, comparison card (FINAL)

Status: complete. Screening (34 arms + 3 diagnostic ablations, 3 repeats, frozen folds), the ocular validity check, label-permutation tests for four candidates, and the efficiency measurements are done. Eye-only, reaction-time-only and fusion results are reported separately throughout. Run history and exact commands are in RESUME.md.

## 1. The published study

**Citation.** Preti AN, Melis G, Barbieri B, Sartori G, Crippa F, Zago S, Bolognini N. "The lie in the eyes: Ocular indicators and machine-learning models with a cognitive load paradigm to detect lying." *Legal and Criminological Psychology* (2026), published online 22 Oct 2026. DOI [10.1111/lcrp.70074](https://doi.org/10.1111/lcrp.70074).

**Access.** The full text returned HTTP 403 from Wiley (article page, PDF, `pdfdirect`, full XML). Unpaywall lists it as "bronze" open access on the publisher site only. No preprint was found on PsyArXiv/OSF, Europe PMC or Semantic Scholar. Everything below comes from the abstract (Crossref and Semantic Scholar), the two Zenodo records, and the data. Where a detail is not in those sources, it is marked "not reported (full text inaccessible)".

| Item | What the accessible sources say |
|---|---|
| Design | Induced fake-identity task. Participants answer Control, Expected and Unexpected questions while assuming a fake identity (liars) or their true identity (truth-tellers): a cognitive-load / unexpected-question paradigm |
| Participants | 57 healthy adults, randomised to lying or truth-telling. Group sizes not reported in the abstract. Data: 27 liars and 30 truth-tellers (`raw data`), or 26/31 (`RTs_aggregated`, which disagrees on subject 3) |
| Trials | Data: 32 per participant (C1-C8, E1-E8, U1-U16); one participant has 29 |
| Eye tracker, sampling rate | Not reported (full text inaccessible). Column names (`CURRENT_FIX_PUP`, `CURRENT_FIX_BLINK_AROUND`, "Fixation: 0 ms" labels in v1) are SR Research Data Viewer fixation-report fields, so likely an EyeLink. In v1, 50% of fixation durations are odd numbers of ms, which fits ≥ 1000 Hz rather than 500 Hz. This is an inference, not a stated fact |
| Measures | Pupil dilation, eye blinks, reaction times (RT), errors |
| Pupil preprocessing (from data) | Baseline = first fixation's pupil; activation = mean of the remaining fixations; corrected = activation − baseline (subtractive). Arbitrary EyeLink units |
| Blink measure (from data) | `blink_rate` = blink-flagged fixations / mean RT (ms), per participant × question type |
| ML models, CV scheme, participant-disjoint? | Not reported (full text inaccessible) |
| Reported results (abstract) | Liars had significantly longer RTs than truth-tellers, especially to Unexpected questions. Truth-tellers' pupils dilated less to Unexpected questions; liars' pupil responses were relatively stable across question types. No group differences in blink rate or errors. ML models reached **up to 84% accuracy**. RT to Unexpected questions was the strongest predictor, and ocular measures added limited predictive value |

| Published metric | Value |
|---|---|
| Accuracy (best model, "up to") | 0.84 |
| AUC, sensitivity, specificity, F1, EER, HTER | not reported (full text inaccessible) |
| Model and CV behind the 0.84 | not reported (full text inaccessible) |

The research note "ML up to .84, ocular adds little beyond reaction time" is **verified** against the abstract.

**Data facts that matter for anyone reusing this dataset** (`data/preti/preti2026.xlsx`, Zenodo v2 [21451382](https://zenodo.org/records/21451382), CC BY 4.0):
- `raw data` is fixation-level: 13,335 fixations in 1,821 trials, about 7 per trial. RT and key press are constant within a trial.
- Coding: `CONDITION` 0 = liar, 1 = truth-teller, inferred from the abstract. Raw mean RT to Unexpected questions is 3,759 ms for group 0 vs 2,140 ms for group 1, and group 0 has the flat pupil profile.
- **The aggregate sheets use different subject numbering.** In `pupil_corrected`, sheet id *s* is raw id *s*+1 for *s* = 3..56: matched exactly on the per-trial baseline pupil, with labels agreeing after the shift. `blink_aggregated` is offset in the same way for 50 of 57 ids. Pupil-sheet id 57 matches no raw participant. `RTs_aggregated` and `raw data` share ids, but disagree on subject 3's label.
- Merging the sheets on `SUBJECT_LABEL` therefore pairs most participants' RTs with someone else's pupil and blink values. Under our protocol this costs fusion AUC 0.06 (0.864 vs 0.927; paired p = .14). We cannot tell whether the authors' analysis did this.
- Zenodo v1 ([18093935](https://zenodo.org/records/18093935), 60 participants, extra fields such as fixation duration) matches v2 participant for participant, with v2 dropping v1 ids 8, 13 and 49 (all liars) and renumbering. v1 confirms subject 3 as a liar, so the raw label is used. Dropping or flipping subject 3 changes AUC by ≤ 0.008.
- All our features are rebuilt from `raw data`.

## 2. Reproduction under a paper-like protocol

The authors' protocol is not available, so we ran the most common one for a study of this kind. Each participant is one row with 9 aggregates (mean RT, blink rate and corrected pupil for each of C, E and U). We then ran stratified 10-fold CV repeated 10 times, with default hyperparameters and no nested tuning, across six model families, and report the best one as "up to".

| Features | Data source | Best model: mean accuracy | Best single seed | Paper |
|---|---|---|---|---|
| Fusion (RT + pupil + blink) | Sheets merged on `SUBJECT_LABEL` | 0.896 (RF) | 0.930 | "up to 0.84" |
| Fusion | Rebuilt from raw | 0.907 (decision tree) | 0.930 | |
| RT only | Sheets | 0.921 (NB) | 0.930 | |
| RT only | Raw | 0.921 (tree) | 0.947 | |
| Ocular only (pupil + blink) | Sheets | 0.570 (LR) | 0.649 | "limited additional value" |
| Ocular only | Raw | 0.544 (RF) | 0.632 | |
| Trial-level random split (leaky), fusion | Raw | 0.665 trial accuracy (LR) | | |

Every model's accuracy is in `results.json` → `reproduction`. Under this protocol we get **above** the published 0.84, not below it. The published number is therefore not an inflated outlier. With 57 participants, best-of-six-models selection and seed choice move accuracy by ±0.05. Ocular aggregates alone sit at chance in every variant, which agrees with the paper's conclusion.

## 3. Honest baseline under our protocol

Folds are participant-disjoint and paired (1 liar + 1 truth-teller per fold), 30 folds × 3 repeats, frozen in `folds.json`. The learner is chosen in inner `StratifiedGroupKFold(5)` CV between logistic regression (C ∈ {0.01, 0.1, 1, 10}) and RBF-SVM (C ∈ {0.3, 1, 3}), with median imputation and scaling fitted on training participants only. The decision unit is the participant; the threshold is 0 on the signed score. 95% CIs come from a participant bootstrap of repeat-averaged scores.

| Baseline (paper's 9 aggregates) | AUC | Accuracy | EER | HTER | Sens | Spec | MCC |
|---|---|---|---|---|---|---|---|
| **Eye only** `B_ocular` (pupil + blink, 6 features) | 0.399 ± 0.023 [CI 0.236, 0.532] | 0.462 ± 0.017 | 0.591 ± 0.022 | 0.548 ± 0.016 | 0.272 | 0.633 | −0.101 |
| **RT only** `B_rt` (3 features) | 0.909 ± 0.015 [0.816, 0.988] | 0.860 ± 0.014 | 0.134 ± 0.017 | 0.140 ± 0.013 | 0.864 | 0.856 | 0.720 |
| **Fusion** `B_fusion` (9 features) | 0.927 ± 0.010 [0.832, 0.995] | 0.877 ± 0.014 [0.825, 0.982] | 0.129 ± 0.017 | 0.124 ± 0.015 | 0.852 | 0.900 | 0.754 |
| Fusion with RF added to the learner space | 0.915 ± 0.009 | 0.836 ± 0.030 | 0.146 ± 0.030 | 0.164 ± 0.029 | 0.827 | 0.844 | 0.672 |
| Fusion from sheets merged on `SUBJECT_LABEL` | 0.864 ± 0.042 [0.752, 0.958] | 0.830 ± 0.022 | 0.158 ± 0.029 | 0.177 ± 0.019 | 0.744 | 0.903 | 0.663 |

The honest fusion accuracy (0.877) is about the published 0.84. The published fusion claim holds up under participant-disjoint nested CV. The paper's ocular aggregates carry no out-of-sample information. Their AUC below 0.5 is the usual small-sample anti-learning of a null signal, not a reversed effect: the 95% CI reaches 0.53.

## 4. Ideas screened

All arms use the same frozen folds. Δ is the paired participant-bootstrap ΔAUC vs the matching honest baseline. Holm correction runs across the 30 arm-vs-baseline tests. The full table with all 12 metrics, including precision, F1, balanced accuracy, Brier and ECE, is in `tables.md` and `results.json`.

### 4a. Eye only (no RT column used anywhere)

| Arm | Idea | AUC | Accuracy | EER | HTER | ΔAUC vs `B_ocular` [95% CI] | Holm p |
|---|---|---|---|---|---|---|---|
| `B_ocular` | paper's 6 ocular aggregates | 0.399 ± 0.023 | 0.462 | 0.591 | 0.548 | | |
| `I1_diff_ocular` | + within-subject contrasts (U−C, E−C, U−E) and deviance in person-SD units | 0.501 ± 0.025 | 0.456 | 0.473 | 0.554 | +0.115 [−0.019, +0.257] | 0.99 |
| `I1b_contrast_only_ocular` | contrasts only, no absolute levels | 0.561 ± 0.019 | 0.520 | 0.433 | 0.488 | +0.191 [+0.053, +0.325] | 0.095 |
| `I2_neutralz_subject_ocular` | (U or E − C) / SD of own control trials | 0.588 ± 0.032 | 0.526 | 0.392 | 0.480 | +0.206 [+0.058, +0.368] | 0.126 |
| `I3_trial_neutralz_aggtuned_ocular` | trial LR on control-normalised trials + inner-tuned aggregation | 0.626 ± 0.018 | 0.556 | 0.404 | 0.466 | +0.281 [+0.109, +0.453] | 0.023 |
| **`I3_trial_raw_aggtuned_ocular`** | trial LR on raw trial pupil + blink, inner-tuned aggregation (vote chosen) | **0.694 ± 0.004** | **0.637 ± 0.008** | **0.416 ± 0.009** | **0.374 ± 0.008** | +0.312 [+0.170, +0.453] | <0.001 |
| `I7_lr_en_I1_ocular` | elastic-net LR on I1 ocular | 0.569 ± 0.041 | 0.532 | 0.450 | 0.476 | +0.191 [+0.048, +0.340] | 0.126 |
| `I7_lgbm_I1_ocular` | LightGBM on I1 ocular | 0.578 ± 0.020 | 0.509 | 0.456 | 0.494 | +0.200 [+0.063, +0.348] | 0.080 |
| `I4_blink_only` | blink rate only | 0.571 ± 0.006 | 0.567 | 0.497 | 0.438 | | |

Pupil preprocessing ablation (pupil only, levels + contrasts + deviance; Δ vs published subtractive `I4_pupil_pc_sub`):

| Pupil variant | AUC | Accuracy | EER | HTER | ΔAUC vs subtractive | Holm p |
|---|---|---|---|---|---|---|
| Subtractive, published (`pc_sub`) | 0.556 ± 0.030 | 0.526 | 0.480 | 0.477 | | |
| Divisive (`pc_div`) | 0.554 ± 0.024 | 0.526 | 0.433 | 0.477 | +0.000 | 1.00 |
| Peak − baseline | 0.550 ± 0.013 | 0.520 | 0.538 | 0.489 | −0.022 | 1.00 |
| Blink-flagged fixations and baselines removed | 0.467 ± 0.044 | 0.474 | 0.510 | 0.534 | −0.114 | 0.99 |
| No baseline (raw level) | 0.403 ± 0.037 | 0.497 | 0.578 | 0.524 | −0.241 [−0.426, −0.063] | 0.135 |
| Subtractive, reject trials with < 3 fixations | 0.576 ± 0.036 | 0.544 | 0.463 | 0.459 | +0.012 | 1.00 |
| Subtractive, reject \|z\| > 3 within person | 0.459 ± 0.028 | 0.474 | 0.556 | 0.530 | −0.115 | 0.95 |

Eye movement measures (eye-tracker data, but fixation count tracks RT: trial r = 0.89):

| Arm | AUC | Accuracy | EER | HTER | ΔAUC vs `B_ocular` | Holm p |
|---|---|---|---|---|---|---|
| `I5_ocular_plus_fixcount` (pupil, blink, fixations per trial) | 0.792 ± 0.011 | 0.725 | 0.281 | 0.278 | +0.410 [+0.248, +0.573] | <0.001 |
| `I5_ocular_plus_fixrate` (pupil, blink, fixations per second) | 0.747 ± 0.021 | 0.678 | 0.293 | 0.332 | +0.370 [+0.240, +0.511] | <0.001 |

**Validity check (done): the best eye-only arm is a reaction-time proxy.** All trial-level ocular arms below use the same frozen folds.

| Check | AUC (mean ± SD) | ρ(score, mean log RT) | AUC after regressing out mean log RT |
|---|---|---|---|
| `I3_trial_raw_aggtuned_ocular` (pupil + blink occurrence per trial) | 0.694 ± 0.004 | +0.48 | 0.519 |
| Ablation: blink occurrence per trial only | 0.699 ± 0.002 | +0.48 | 0.521 |
| Ablation: corrected pupil only | 0.400 ± 0.036 | −0.20 | 0.644 |
| Ablation: pupil + blinks **per second** (duration-normalised) | 0.532 ± 0.019 | +0.17 | 0.465 |
| `I3_trial_neutralz_aggtuned_ocular` | 0.626 ± 0.018 | +0.22 | 0.589 |
| `I5_ocular_plus_fixcount` | 0.792 ± 0.011 | +0.54 | 0.641 |
| `I5_ocular_plus_fixrate` | 0.747 ± 0.021 | +0.50 | 0.556 |
| RT-only candidate `I7_lgbm_I1_rt`, for reference | 0.969 ± 0.002 | +0.87 | 0.793 |

At trial level, blink occurrence correlates with log RT at r = 0.41 within a person. Fixation count correlates at r = 0.86; corrected pupil at r = 0.01. All of the best eye-only arm's signal comes from "did a blink happen in this trial", which is more likely in longer trials. Once blinks are counted per second, AUC falls to 0.53. Once the person's mean RT is regressed out, AUC falls to 0.52.

**Conclusion for the eye-only claim: no eye-only signal independent of RT was found in this dataset.** That agrees with the paper. Fixation-count arms keep some AUC after removing mean RT (0.56–0.64), but so does RT itself (0.79), because the RT profile across question types carries more than mean RT. Fixation counts cannot be credited to the eyes rather than to timing.

### 4b. Reaction time only

| Arm | Idea | AUC | Accuracy | EER | HTER | ΔAUC vs `B_rt` [95% CI] | Holm p |
|---|---|---|---|---|---|---|---|
| `B_rt` | paper's 3 RT means | 0.909 ± 0.015 | 0.860 | 0.134 | 0.140 | | |
| `I1_diff_rt` | levels + contrasts + U/C ratios + deviance, RT and log RT | 0.959 ± 0.009 | 0.901 ± 0.008 | 0.076 ± 0.008 | 0.101 ± 0.008 | +0.049 [+0.009, +0.100] | **0.042** |
| `I2_neutralz_subject_rt` | control-normalised only | 0.702 ± 0.010 | 0.667 | 0.333 | 0.341 | −0.217 [−0.369, −0.064] | 0.128 |
| `I3_trial_raw_aggtuned_rt` | trial LR + tuned aggregation | 0.933 ± 0.004 | 0.883 | 0.129 | 0.119 | +0.019 | 1.00 |
| `I7_lr_en_I1_rt` | elastic-net LR on I1 RT | 0.967 ± 0.008 | 0.930 ± 0.000 | 0.059 ± 0.021 | 0.072 ± 0.000 | +0.052 [+0.010, +0.107] | **0.023** |
| `I7_lgbm_I1_rt` | LightGBM on I1 RT | 0.969 ± 0.002 | 0.936 ± 0.008 | 0.059 ± 0.017 | 0.064 ± 0.009 | +0.052 [+0.004, +0.128] | 0.43 |

### 4c. Fusion (RT + eye)

| Arm | Idea | AUC | Accuracy | EER | HTER | Brier | ECE | ΔAUC vs `B_fusion` [95% CI] | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| `B_fusion` | paper's 9 aggregates | 0.927 ± 0.010 | 0.877 | 0.129 | 0.124 | 0.104 | 0.078 | | |
| `I1_diff_fusion` | I1 features | 0.937 ± 0.006 | 0.889 | 0.106 | 0.114 | 0.109 | 0.120 | +0.020 | 1.00 |
| `I2_neutralz_subject_fusion` | control-normalised only | 0.641 ± 0.035 | 0.608 | 0.374 | 0.399 | 0.244 | 0.108 | −0.265 | <0.001 (worse) |
| `I3_trial_neutralz_aggtuned_fusion` | control-normalised trials + aggregation | 0.590 ± 0.032 | 0.532 | 0.427 | 0.486 | 0.245 | — | −0.327 | <0.001 (worse) |
| `I3_trial_raw_aggtuned_fusion` | raw trials + tuned aggregation (mean chosen) | 0.923 ± 0.002 | 0.877 | 0.123 | 0.122 | 0.182 | 0.189 | −0.005 | 1.00 |
| `I3_trial_raw_mean_fusion` | same, aggregation fixed to mean | 0.923 ± 0.002 | 0.877 | 0.123 | 0.122 | 0.182 | 0.189 | −0.005 | 1.00 |
| **`I7_lr_en_I1_fusion`** | elastic-net LR on I1 fusion | **0.971 ± 0.002** | **0.924 ± 0.017** | **0.076 ± 0.034** | **0.076 ± 0.017** | 0.071 | 0.048 | +0.051 [+0.004, +0.106] | 0.43 |
| `I7_lgbm_I1_fusion` | LightGBM on I1 fusion | 0.969 ± 0.002 | 0.930 | 0.098 | 0.070 | 0.046 | 0.050 | +0.040 | 1.00 |

Does the eye add to RT? Paired ΔAUC, fusion minus RT-only, on identical folds:
- Paper features (`B_fusion − B_rt`): +0.014 [−0.020, +0.048], p = .39.
- I1 with LR/SVM: −0.016, p = .34.
- I1 with elastic net: +0.012, p = .39.
- I1 with LightGBM: +0.001, p = 1.0.

None of these is a reliable gain. The eye adds nothing measurable to RT, which agrees with the paper.

**What did not work, and why.** Within-subject normalisation on the person's own control questions (I2, I3-neutralz) destroys the RT signal: fusion AUC drops from 0.93 to 0.59–0.64. In this design liars are slower on **all** question types, including control questions (raw mean RT, liars vs truth-tellers: control 1,728 vs 1,274 ms; expected 1,594 vs 1,206 ms; unexpected 3,716 vs 2,124 ms). Normalising to control trials removes the person-level slowing that carries most of the information. The inner-tuned aggregation rule chose "mean" for fusion and RT, identical to fixing the mean (ΔAUC 0.000). It chose "vote" for the eye-only trial models.

## 5. Best candidates vs published vs honest baseline

**Label-permutation tests** (labels shuffled between participants, paired folds rebuilt, the full nested pipeline rerun for one repeat per draw; p = (1 + #null >= observed) / (1 + n); `final.json`, `perm_rt_result.json`). The observed value is the mean AUC over the 3 repeats.

| Candidate | Observed AUC | Null AUC mean ± SD | Null p95 | Draws | p |
|---|---|---|---|---|---|
| Eye only: `I3_trial_raw_aggtuned_ocular` | 0.694 | 0.474 ± 0.098 | 0.604 | 30 | 0.032 |
| Eye + fixation rate: `I5_ocular_plus_fixrate` | 0.747 | 0.486 ± 0.094 | 0.616 | 30 | 0.032 |
| RT only: `I7_lgbm_I1_rt` | 0.969 | 0.489 ± 0.115 | 0.643 | 20 | 0.048 |
| Fusion: `I7_lr_en_I1_fusion` | 0.971 | 0.464 ± 0.105 | 0.637 | 20 | 0.048 |

0.032 and 0.048 are the smallest p-values 30 and 20 draws allow (1/31 and 1/21): no null draw reached the observed AUC. They show only that each score beats chance on these folds. For the two eye arms, beating chance is exactly what the validity check explains away: their signal is reaction time (section 4a). The permutation null uses a single repeat per draw, so it is somewhat wider than the 3-repeat observed mean, and it does not account for choosing the best arm of many.

**Mechanism (`final.json`).**
- Univariate, full-sample AUCs (descriptive, no fitting): mean RT to Unexpected questions 0.960, U−E RT contrast 0.960, mean RT over all trials 0.942, control-question RT 0.893.
- Mean RT to Unexpected questions used alone as a score, with no model, reaches AUC 0.960 and EER 0.072.
- The gain of the I1/I7 RT features over `B_rt` (+0.05 AUC, Holm-significant) comes from the log transform and the within-person U−E and U−C contrasts. Elastic-net fusion weights: log RT_U 0.43, RT_U−E 0.33, RT_U 0.31, RT_U−C 0.21, and zero weight on all 12 ocular features.
- Ocular univariate AUCs top out at pupil U−C 0.68: liars' pupils dilate relatively more to Unexpected vs Control (corrected pupil, liars −605 vs truth-tellers −661 on U), which matches the abstract. This group-level effect does not survive out-of-sample: contrast-only eye arms reach 0.56–0.59.
- Fixation count, subject level, correlates r = 0.82 with mean RT. Bootstrap CIs on repeat-averaged scores are already computed (`final.json`):

| Candidate | AUC [95% CI] | Accuracy [95% CI] | EER [95% CI] | HTER [95% CI] |
|---|---|---|---|---|
| Eye only: `I3_trial_raw_aggtuned_ocular` | 0.693 [0.547, 0.816] | 0.649 [0.544, 0.754] | 0.422 [0.263, 0.528] | 0.363 [0.254, 0.474] |
| Eye only, honest baseline `B_ocular` | 0.380 [0.236, 0.532] | 0.439 [0.333, 0.544] | 0.631 [0.474, 0.754] | 0.572 [0.465, 0.680] |
| Eye + fixation count: `I5_ocular_plus_fixcount` | 0.790 [0.658, 0.906] | 0.702 [0.579, 0.825] | 0.263 [0.141, 0.387] | 0.300 [0.180, 0.422] |
| RT only: `I7_lgbm_I1_rt` | 0.967 [0.895, 1.000] | 0.947 [0.894, 1.000] | 0.070 [0.000, 0.141] | 0.052 [0.000, 0.108] |
| RT only, honest baseline `B_rt` | 0.915 [0.816, 0.988] | 0.877 [0.789, 0.947] | 0.106 [0.035, 0.228] | 0.122 [0.050, 0.209] |
| Fusion: `I7_lr_en_I1_fusion` | 0.979 [0.942, 1.000] | 0.930 [0.860, 0.982] | 0.070 [0.000, 0.141] | 0.069 [0.017, 0.139] |
| Fusion, honest baseline `B_fusion` | 0.928 [0.832, 0.995] | 0.912 [0.825, 0.982] | 0.122 [0.017, 0.193] | 0.089 [0.019, 0.174] |
| Published (fusion, "up to") | n/r | 0.84 | n/r | n/r |

These rows use repeat-averaged scores, so they differ slightly from the mean-over-repeats values in section 4. Candidates were picked as the highest-AUC arm in each modality, so the CIs do not account for that selection.

## 6. Efficiency

Hardware (`hardware_info()`): Intel64 Family 6 Model 142 Stepping 10 (8 logical CPUs), Windows 11, Python 3.14.6. Single-threaded, on a CPU shared with other jobs, so all timings are upper bounds. The decision unit is one participant (276 fixation rows in 32 trials), starting from the rawest data available, the fixation-level export. Sensor samples per second and the real-time factor cannot be computed from a fixation export, so throughput is given per participant and per fixation row. Latencies are p50 / p95 in ms.

| Method | Feats | Feature extraction | Inference only | End to end | Participants/s | Fixation rows/s | Peak memory (B) | Model size (B) | Stored parameters |
|---|---|---|---|---|---|---|---|---|---|
| Honest baseline `B_fusion` (LR C = 1; re-implementation of the published method) | 9 | 117.3 / 127.3 | 2.37 / 2.57 | 124.4 / 149.2 | 8.0 | 2,216 | 140,133 | 1,650 | 37 |
| Eye-only `I3_trial_raw_aggtuned_ocular` (timed as LR C = 0.1 with mean aggregation, approximate) | 4 | 85.0 / 92.9 | 2.37 / 2.94 | 91.9 / 110.4 | 10.3 | 2,841 | 140,561 | 1,450 | 17 |
| RT only `I7_lgbm_I1_rt` (LightGBM, 3 leaves) | 20 | 169.4 / 203.5 | 4.07 / 4.48 | 170.5 / 195.1 | 5.7 | 1,566 | 159,997 | 54,136 | 2,060 |
| Fusion `I7_lr_en_I1_fusion` (elastic net C = 0.1, l1_ratio 0.5) | 36 | 238.5 / 256.9 | 2.44 / 2.55 | 236.3 / 242.3 | 4.0 | 1,114 | 155,638 | 2,729 | 145 |

Per trial, feature extraction costs about 42-43 ms (p50) for every method. Model inference is 2-4 ms, so cost is dominated by feature extraction, not by the model. Every candidate decides one participant in under 0.25 s with under 0.2 MB of peak memory, so efficiency does not separate the methods here. Stored parameters count the imputer, scaler and classifier.

**Addendum (vectorised extraction).** The extraction cost above is mostly pandas overhead, not inherent. A NumPy re-implementation (`src/edd/preti_fast.py`, benchmark `scripts/phase2_preti_fast_efficiency.py`) reproduces all four feature sets (maximum absolute difference 1.1e-13, `numpy.allclose` rtol 1e-9 on all 57 participants) and leaves the out-of-fold AUC unchanged on the frozen folds. Measured in one session under heavy CPU load (so upper bounds), end-to-end p50 per participant fell from 133 / 83 / 170 / 198 ms to 2.7 / 2.4 / 4.9 / 3.8 ms (`B_fusion` / eye-only / RT only / fusion), starting from pre-sorted arrays; starting from the DataFrame adds about 3.4-4.2 ms of ingestion. Details: `efficiency_fast.md`, `efficiency_fast.json`. The table above is unchanged.

## 7. Recommendation

**(a) Eye-only conclusion.** No eye-only signal independent of reaction time was found in this dataset.
- The best eye-only arm (AUC 0.694, accuracy 0.637, EER 0.416, HTER 0.374; permutation p = 0.032) is a reaction-time proxy. Its signal is blink occurrence per trial, which rises with trial length (within-person r = 0.41 with log RT). With blinks per second, AUC falls to 0.532. After regressing out the person's mean log RT, it falls to 0.519.
- The fixation-count arm (AUC 0.792) and fixation-rate arm (0.747, p = 0.032) are also RT proxies: fixation count correlates with RT at r = 0.85-0.89 per trial and 0.82 per subject. They cannot be credited to the eyes rather than to timing.
- Pupil preprocessing variants (subtractive, divisive, peak, blink-cleaned, outlier-rejected) are all at chance (0.40-0.58). Control-question normalisation does not rescue the eye signal, and it destroys the RT signal.
- This agrees with the authors' abstract ("ocular measures added limited predictive value"). For the paper this dataset is a negative control for eye-only claims. An eye-only full run is not worth the compute.

**(b) Comparison with the published number.** The published result is "up to 84% accuracy" (abstract only). The full text was inaccessible, so their model, CV scheme and whether folds were participant-disjoint are **not reported**. Under our participant-disjoint nested protocol (3 repeats, paired folds):

| | Accuracy | AUC | EER | HTER |
|---|---|---|---|---|
| Published, fusion, "up to" | 0.84 | not reported | not reported | not reported |
| Honest baseline `B_fusion` | 0.877 | 0.927 | 0.129 | 0.124 |
| Honest baseline `B_rt` | 0.860 | 0.909 | 0.134 | 0.140 |
| Best RT only `I7_lgbm_I1_rt` | 0.936 | 0.969 | 0.059 | 0.064 |
| Best fusion `I7_lr_en_I1_fusion` | 0.924 | 0.971 | 0.076 | 0.076 |

Accuracy deltas versus the published 0.84 are +0.037 (honest baseline), +0.096 (RT only) and +0.084 (fusion). Because their protocol is unknown, these deltas are indicative only, not a like-for-like win. The honest baseline already reaches the published accuracy, so the published claim is not inflated. The RT-only gain over its own baseline (delta AUC +0.05; `I1_diff_rt` Holm p = 0.042, `I7_lr_en_I1_rt` p = 0.023) holds after Holm correction. The fusion gain over `B_fusion` (Holm p = 0.43) and fusion minus RT-only (|delta AUC| <= 0.016) are no reliable difference. The elastic net gives every ocular feature zero weight. A single no-model statistic, mean RT to Unexpected questions, already reaches AUC 0.960 and EER 0.072. Caveats: 57 participants; candidates were picked as the best arm of many, so CIs and permutation p-values do not account for that selection; the permutation null uses a single repeat.

**(c) Data-quality issue to report to the authors.** The deposited `pupil_corrected` and `blink_aggregated` sheets number subjects one off from `raw data` and `RTs_aggregated` (sheet id s is raw id s+1 for ids >= 3; pupil-sheet id 57 matches no raw participant). Merging the sheets on `SUBJECT_LABEL` pairs most participants' RTs with another participant's pupil and blink values. Under our protocol this costs fusion AUC about 0.06 (0.864 vs 0.927). Subject 3's label also disagrees between `RTs_aggregated` and `raw data` (Zenodo v1 supports the raw label; dropping or flipping subject 3 changes AUC by <= 0.008), and v2 drops v1 ids 8, 13 and 49 (all liars). We cannot tell whether the authors' own analysis merged on the shifted ids.

**Next steps.** RT and fusion are near ceiling (AUC 0.97). A short final run with more repeats is justified only as an RT/fusion comparison point, not as an eye-movement result. An independent dataset is needed to say whether any ocular signal exists beyond RT.
