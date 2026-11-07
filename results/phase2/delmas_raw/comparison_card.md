# Comparison card, track `delmas_raw` (Delmas et al., 2024, raw EyeLink samples)

> FINAL (2026-11-02). Every number is copied from `results.json`, `noninferiority.json`, `tables.md` and `checkpoints/*.json`. Sections 3b and 5 describe observations on the released data, pending an answer from the authors.

## 1. The published study

**Citation.** Delmas, H., Ciocan, C., Novopashyna, M., & Paeye, C. (2024). Resistance of a short-term memory concealed information test with famous faces to countermeasures. *Memory & Cognition*. DOI 10.3758/s13421-023-01489-1. Data: OSF vygfz.

| Item | Published |
|---|---|
| Task | Short-term-memory CIT (STM-CIT). Each trial shows a 2 x 2 parallel display of four same-gender faces for 5,000 ms. A single face follows and the participant answers whether it was in the display. 64 trials after 4 practice trials. |
| Items | 64 faces: 56 unknown and 8 celebrities famous in France. Each famous face appears 4 times in the parallel displays and 4 times as the single face. Unknown faces appear 3 or 4 times in parallel displays and at most once as the single face. |
| Participants | 15 per group, 75 in total. Exp. 1 (n = 45, 4 excluded): G1 control, G2 concealment, G3 simple countermeasure (CM). Exp. 2 (n = 30, 2 excluded): G4 enhanced CM (slideshow, repeated at mid-session) and G5 CM with feedback (own gaze time course after 32 trials). |
| Sensor | EyeLink 1000, 1,000 Hz, monocular, chin rest, 57 cm, 1920 x 1080 at 144 Hz, faces 11.5 x 8.6 deg centred at 18 deg eccentricity. |
| Preprocessing | EyeLink parser (30 deg/s, 8,000 deg/s^2). Blinks removed. Fixations off the pictures ignored. Misidentified faces relabelled from the post-test, and trials with ambiguous labels dropped. |
| Features (classifier) | 4 per face: dwell 200 to 1,000 ms, dwell 1,001 to 5,000 ms, total fixation count, total visit count, all from the parallel displays. |
| Model | RBF-SVM, StandardScaler, GridSearchCV over C and gamma in {1e-3 to 1e3} (7 x 7), 4-fold *non-grouped* inner CV, class weights {1: 8/64, 0: 56/64}. |
| Evaluation | Unit = face (famous vs unknown) within each group. "Between-participants" = leave-one-participant-out. Exp. 2 classification uses the last 32 trials (after the CM manipulation). |
| Not reported | EER, HTER, MCC, pooled metrics, CIs, any sampling rate other than 1,000 Hz, latency, memory, model size. |

**Published between-participant results** (mean over held-out participants, Table 6 for 64 trials and Table 8 for the last 32 trials):

| Set | AUC | Balanced acc. | F1 | Chance-label AUC (Table 4) |
|---|---|---|---|---|
| G1 control, 64 | 1.00 (0) | .98 (.03) | .97 (.03) | .60 |
| G2 concealment, 64 | .93 (.07) | .89 (.09) | .83 (.13) | .64 |
| G3 simple CM, 64 | 1.00 (0) | .96 (.09) | .93 (.15) | .73 |
| G3 simple CM, last 32 | .82 (.08) | .94 (.04) | .73 (.13) | .56 |
| G4 enhanced CM, last 32 | .99 (.03) | 1.00 (0) | .98 (.03) | .61 |
| G5 CM + feedback, last 32 | 1.00 (.02) | 1.00 (0) | .99 (.02) | .82 |

The paper's headline is that the STM-CIT "could still be used to distinguish between familiar and unfamiliar faces with high accuracy" even in the feedback group, where the ANOVA showed that the orientation-avoidance gaze pattern had disappeared.

## 2. Reproduction under their protocol

Phase 1 (`results/phase1/REPORT.md`) reproduced all six published values within 0.01 to 0.02 from the authors' per-face table (`BookFIP_*.xlsx`). The only exception is G3-32, where we get AUC 0.937 vs 0.82 published.

## 3. Honest baseline (published table + paper's SVM, our protocol)

Protocol: within each group, 5 participant-disjoint outer folds x 3 repeats (frozen in `folds.json`); nested StratifiedGroupKFold(4) tuning on training participants only; scaler and imputer inside the pipeline; balanced class weights. Metrics come from `edd.metrics.binary_metrics` (threshold 0 on SVM scores), pooled over held-out faces, as mean ± SD over repeats. "CIT" is a participant-level decision: the z-score of famous vs unknown model scores, compared against the same person with 8 random unknown faces as pseudo-probes, using the a-priori threshold z >= 1.645.

| Set | AUC | BA | EER | HTER | MCC | per-participant AUC | CIT AUC / EER / HTER |
|---|---|---|---|---|---|---|---|
| G1-64 | 0.999 ± 0.000 | 0.991 | 0.006 | 0.009 | 0.962 | 1.000 | 1.000 / 0.000 / 0.000 |
| G2-64 | 0.994 ± 0.003 | 0.977 | 0.023 | 0.023 | 0.905 | 0.993 | 1.000 / 0.000 / 0.000 |
| G3-64 | 0.999 ± 0.000 | 0.997 | 0.002 | 0.003 | 0.978 | 1.000 | 1.000 / 0.000 / 0.000 |
| G3-32 | 0.978 ± 0.004 | 0.941 | 0.056 | 0.059 | 0.801 | 0.977 | 1.000 / 0.000 / 0.011 |
| G4-32 | 1.000 ± 0.000 | 0.992 | 0.010 | 0.008 | 0.956 | 1.000 | 1.000 / 0.000 / 0.000 |
| G5-32 | 1.000 ± 0.000 | 0.984 | 0.000 | 0.016 | 0.906 | 1.000 | 1.000 / 0.000 / 0.000 |
| **Mean** | **0.995** | 0.980 | 0.016 | 0.020 | 0.918 | | 1.000 / 0.000 / 0.002 |

## 3b. Key observation: we could not rebuild the released per-face table from the raw data, and the label-free rebuild is much weaker

We rebuilt every per-face feature from the raw EDF samples (`Data_raw.zip`, 940,857,791 bytes, size-verified) and the authors' Psychtoolbox design files, which give the identity and quadrant of all four faces in every trial. The pipeline is validated in three ways. Our raw-sample reader matches `eyelinkio` bit for bit. Our I-VT fixation counts correlate r = 0.97 with the EyeLink parser at 1,000 Hz. EyeLink-parser fixations mapped to our AOIs reproduce the published `n_fix` exactly for 80% (64-trial table) and 91% (32-trial table) of faces, under the rule below.

**Observation on the released data (pending the authors' answer): the counts match a label-dependent aggregation.** For the published `n_fix` and `n_visit`, famous faces are summed over all their presentations, and unknown faces only over presentations in trials *without* a famous face (r = 0.925 and 0.965 against this rule, vs 0.377 and 0.640 against all presentations). So the number of presentations counted depends on the label: 4.0 for famous vs 2.37 for unknown in the 64-trial table. That count *alone* separates the classes with AUC 0.943. A deployed system cannot use such a rule, because it would need to know which faces are famous before choosing which presentations to count. Per presentation, famous faces receive *fewer* fixations in our rebuild (EyeLink n_fix per presentation, famous-vs-unknown AUC 0.21), whereas the released table shows more fixations on famous faces. This is an observation on the released data, pending an answer from the authors; we do not know how the table was constructed.

**Observation on the released data (pending the authors' answer): the dwell columns could not be reproduced.** No aggregation rule we tried reproduces the published dwell columns (r about 0.3 for every rule, including pooling over the co-presented unknown faces). The published dwell values are also far less variable within class (SD 0.018 vs 0.05 to 0.06 for the true per-face means), and published `dwell_2` alone gives AUC 0.998, vs 0.818 for the rebuilt value. We cannot tell from the shared files how these columns were computed. **This needs a question to the authors before any claim about it goes into the paper.**

**Consequence (honest protocol, same folds, image-level AUC):**

| Features | G1-64 | G2-64 | G3-64 | G3-32 | G4-32 | G5-32 | Mean |
|---|---|---|---|---|---|---|---|
| Published table (honest baseline) | 0.999 | 0.994 | 0.999 | 0.978 | 1.000 | 1.000 | 0.995 |
| Own detector, published label-dependent rule | 0.930 | 0.921 | 0.876 | 0.774 | 0.585 | 0.599 | 0.781 |
| Counts only, published rule | 0.666 | 0.739 | 0.601 | 0.590 | 0.591 | 0.578 | 0.627 |
| Counts only, all presentations (label-free) | 0.850 | 0.820 | 0.796 | 0.791 | 0.592 | 0.620 | 0.745 |
| **P4, label-free, 1,000 Hz (SVM)** | 0.916 | 0.888 | 0.838 | 0.825 | 0.572 | 0.618 | 0.776 |

With features that a deployed system could actually compute, detection stays good without countermeasures (AUC 0.84 to 0.92). It drops to near chance under the enhanced and feedback countermeasures (AUC about 0.57 to 0.62, EER about 0.42 to 0.46). With deployable features, we do not reproduce the paper's conclusion that the STM-CIT resists countermeasures; whether the released features are comparable to ours is the open question for the authors.

## 4. Ideas screened

All numbers: within-group, 5 participant-disjoint outer folds x 3 repeats (frozen in `folds.json`, identical for every row), nested participant-grouped tuning, scaler and imputer fitted on training participants only. Image-level pooled metrics, mean over the six group-sets, mean ± SD over the 3 repeats. Per-set values for every row are in `tables.md`; raw numbers in `results.json` and `checkpoints/`. "CM sets" = mean over G2, G3-32, G4-32, G5-32 (AUC / EER / HTER). CIT = participant-level decision (see section 3).

| Config | BA | F1 | MCC | AUC | EER | HTER | CM sets AUC / EER / HTER | per-participant AUC | CIT AUC / EER / HTER |
|---|---|---|---|---|---|---|---|---|---|
| Published table + SVM (honest baseline) | 0.980±0.003 | 0.929±0.008 | 0.918±0.009 | 0.995±0.001 | 0.016±0.003 | 0.020±0.003 | 0.994 / 0.018 / 0.022 | 0.995 | 1.000 / 0.000 / 0.002 |
| Own detector, published label-dependent aggregation, SVM | 0.732±0.007 | 0.464±0.008 | 0.365±0.011 | 0.781±0.005 | 0.268±0.005 | 0.268±0.007 | 0.751 / 0.291 / 0.291 | 0.800 | 0.895 / 0.148 / 0.302 |
| Counts only, published aggregation, SVM | 0.604±0.014 | 0.300±0.012 | 0.148±0.021 | 0.627±0.015 | 0.403±0.013 | 0.396±0.014 | 0.620 / 0.408 / 0.403 | 0.647 | 0.803 / 0.244 / 0.478 |
| Counts only, all presentations, SVM | 0.681±0.008 | 0.394±0.011 | 0.271±0.014 | 0.745±0.012 | 0.314±0.009 | 0.319±0.008 | 0.724 / 0.329 / 0.335 | 0.775 | 0.876 / 0.128 / 0.330 |
| P4 SVM 1000 Hz | 0.724±0.011 | 0.463±0.014 | 0.359±0.018 | 0.776±0.009 | 0.278±0.013 | 0.276±0.011 | 0.748 / 0.300 / 0.299 | 0.794 | 0.882 / 0.139 / 0.267 |
| P4 LR 1000 Hz | 0.726±0.006 | 0.458±0.007 | 0.353±0.010 | 0.779±0.007 | 0.275±0.008 | 0.274±0.006 | 0.753 / 0.296 / 0.296 | 0.798 | 0.883 / 0.146 / 0.287 |
| P4+bins LR 1000 Hz | 0.730±0.007 | 0.464±0.009 | 0.361±0.012 | 0.781±0.008 | 0.271±0.010 | 0.270±0.007 | 0.755 / 0.292 / 0.292 | 0.793 | 0.877 / 0.144 / 0.281 |
| NEW LR 1000 Hz | 0.519±0.016 | 0.233±0.014 | 0.026±0.023 | 0.523±0.017 | 0.476±0.014 | 0.481±0.016 | 0.511 / 0.487 / 0.493 | 0.536 | 0.554 / 0.456 / 0.500 |
| pupil_only LR 1000 Hz | 0.505±0.010 | 0.225±0.008 | 0.005±0.014 | 0.511±0.014 | 0.492±0.013 | 0.495±0.010 | 0.496 / 0.504 / 0.505 | 0.521 | 0.539 / 0.470 / 0.493 |
| firstlook_only LR 1000 Hz | 0.524±0.015 | 0.240±0.012 | 0.033±0.020 | 0.532±0.012 | 0.474±0.014 | 0.476±0.015 | 0.518 / 0.487 / 0.488 | 0.544 | 0.553 / 0.469 / 0.500 |
| P4+NEW LR 1000 Hz | 0.730±0.008 | 0.463±0.009 | 0.359±0.013 | 0.776±0.008 | 0.273±0.009 | 0.270±0.008 | 0.748 / 0.295 / 0.295 | 0.796 | 0.885 / 0.157 / 0.281 |
| ALL LR 1000 Hz | 0.730±0.009 | 0.463±0.010 | 0.360±0.014 | 0.779±0.009 | 0.273±0.009 | 0.270±0.009 | 0.750 / 0.296 / 0.294 | 0.795 | 0.884 / 0.148 / 0.281 |
| P4_z LR 1000 Hz | 0.743±0.006 | 0.472±0.007 | 0.375±0.010 | 0.790±0.006 | 0.260±0.008 | 0.257±0.006 | 0.766 / 0.281 / 0.279 | 0.795 | 0.884 / 0.126 / 0.281 |
| P4_z SVM 1000 Hz | 0.743±0.004 | 0.490±0.006 | 0.392±0.007 | 0.789±0.007 | 0.259±0.010 | 0.257±0.004 | 0.765 / 0.280 / 0.277 | 0.794 | 0.875 / 0.146 / 0.243 |
| ALL_z LR 1000 Hz | 0.740±0.004 | 0.474±0.005 | 0.376±0.007 | 0.789±0.008 | 0.264±0.006 | 0.260±0.004 | 0.763 / 0.284 / 0.282 | 0.793 | 0.882 / 0.159 / 0.280 |
| ALL_z SVM 1000 Hz | 0.736±0.007 | 0.486±0.010 | 0.386±0.010 | 0.781±0.011 | 0.266±0.011 | 0.264±0.007 | 0.753 / 0.290 / 0.288 | 0.786 | 0.874 / 0.157 / 0.263 |
| P4 LightGBM 1000 Hz | 0.707±0.010 | 0.451±0.015 | 0.341±0.019 | 0.757±0.010 | 0.302±0.012 | 0.293±0.010 | 0.728 / 0.326 / 0.317 | 0.774 | 0.862 / 0.169 / 0.263 |
| ALL_z LightGBM 1000 Hz | 0.727±0.008 | 0.503±0.010 | 0.407±0.012 | 0.780±0.008 | 0.278±0.006 | 0.273±0.008 | 0.755 / 0.298 / 0.294 | 0.787 | 0.890 / 0.143 / 0.246 |

Reading guide: P4 = the paper's four features, rebuilt per presentation (label-free). `_z` = z-scored within participant (uses only that participant's own images). bins = five finer time-course dwell bins. NEW = first-fixation capture, first-look latency, pupil (blink-interpolated, baseline-subtracted), microsaccade rate (>= 500 Hz only). ALL_z = P4 + bins + NEW, z-scored. All are label-free and use only the participant's own samples. "Published label-dependent aggregation" rows use our detector but the presentation-selection rule that matches the released table (see 3b); they are labelled as such because that rule needs the labels.

### Idea 1: sampling-rate degradation (anti-alias FIR, decimate, re-run I-VT, rebuild features)

Mean image-level AUC / EER / HTER over the six sets; CIT AUC of the SVM in the last column.

| Rate | P4 + SVM (paper's model) | ALL_z + LR | CIT AUC (P4 SVM) |
|---|---|---|---|
| 1000 Hz | 0.776 / 0.278 / 0.276 | 0.789 / 0.264 / 0.260 | 0.882 |
| 500 Hz | 0.780 / 0.273 / 0.273 | 0.791 / 0.261 / 0.260 | 0.876 |
| 250 Hz | 0.778 / 0.276 / 0.274 | 0.789 / 0.262 / 0.259 | 0.883 |
| 120 Hz | 0.773 / 0.272 / 0.271 | 0.789 / 0.261 / 0.258 | 0.882 |
| 60 Hz | 0.770 / 0.279 / 0.271 | 0.790 / 0.261 / 0.258 | 0.874 |
| 30 Hz | 0.769 / 0.280 / 0.276 | 0.781 / 0.266 / 0.262 | 0.857 |

The curve is flat from 1000 to 60 Hz and loses about 0.01 AUC at 30 Hz. The ALL_z features below 500 Hz exclude the microsaccade feature. Foucher et al. (ETRA 2025) report a participant-disjoint accuracy drop from 0.736 at 1000 Hz to 0.638 at 200 Hz on a different task; we do not see a comparable drop here. Our I-VT fixation counts agree less with the EyeLink parser at low rates (r = 0.97 at >= 120 Hz, 0.88 at 60 Hz, 0.77 at 30 Hz), yet the classification result barely moves.

**Non-inferiority (margin 0.02 AUC; `noninferiority.json`, `scripts/phase2_delmas_raw_noninf.py`).** Metric = image-level AUC pooled within each set, averaged over the six sets (OOF scores averaged over repeats). CI = paired participant-cluster bootstrap (2000 resamples, participants resampled within each set). One-sided bootstrap p for H0: delta <= -0.02, Holm-corrected across the six comparisons.

| Candidate vs reference | Ref AUC | Cand. AUC | Delta | 95% CI | p (H0: delta <= -0.02) raw / Holm | Verdict |
|---|---|---|---|---|---|---|
| P4 SVM 60 Hz vs P4 SVM 1000 Hz | 0.7811 | 0.7771 | -0.0040 | [-0.0154, +0.0075] | 0.0040 / 0.0160 | non-inferior |
| P4 SVM 30 Hz vs P4 SVM 1000 Hz | 0.7811 | 0.7715 | -0.0096 | [-0.0221, +0.0033] | 0.0530 / 0.0590 | not shown (borderline) |
| ALL_z LR 60 Hz vs ALL_z LR 1000 Hz | 0.7929 | 0.7935 | +0.0006 | [-0.0106, +0.0107] | 0.0005 / 0.0030 | non-inferior |
| ALL_z LR 30 Hz vs ALL_z LR 1000 Hz | 0.7929 | 0.7854 | -0.0075 | [-0.0203, +0.0043] | 0.0295 / 0.0590 | not shown (borderline) |
| ALL_z LR 1000 Hz vs P4 SVM 1000 Hz | 0.7811 | 0.7929 | +0.0118 | [-0.0044, +0.0255] | 0.0005 / 0.0030 | non-inferior (superiority not shown) |
| ALL_z LR 30 Hz vs P4 SVM 1000 Hz | 0.7811 | 0.7854 | +0.0044 | [-0.0153, +0.0230] | 0.0045 / 0.0160 | non-inferior |

Cross-check with the coordinator's preliminary preview (`scripts/delmas_noninferiority_preview.py`, margin 0.02, 1000 resamples): it found SVM and LR 1000 -> 60 Hz non-inferior, 30 Hz borderline (lower CI about -0.021 to -0.023), and LR 30 Hz vs SVM 1000 Hz non-inferior (delta +0.004, CI [-0.015, +0.022]). Our final run agrees on every verdict. The lower bounds at 30 Hz are -0.0221 (SVM) and -0.0203 (LR); LR 30 Hz vs SVM 1000 Hz is +0.0044 [-0.0153, +0.0230]. After Holm correction the within-model 30 Hz comparisons stay unproven (Holm p = 0.059 each). Caveat: this is AUC within groups of 14 to 15 participants, so the CIs are wide (about ±0.01 to 0.02) and of similar size to the margin.

A second, independent paired test (Wilcoxon on per-participant AUC, 75 participants, group 3 averaged over its two sets, Holm over 14 candidates vs P4 SVM 1000 Hz, main script) found no difference for 120 Hz (-0.0006), 60 Hz (-0.0091), 30 Hz (-0.0098) and ALL_z LR 60 Hz (-0.0022); all Holm p = 1.0. That test cannot show equivalence, only absence of a detectable difference; the non-inferiority table above is the equivalence evidence.

### Idea 2: countermeasure transfer (LR; train on one block of groups, test on the other; participants disjoint)

Train on groups 1-2 (64 trials), test on groups 3-5 (last 32 trials), and the reverse. The SD over repeats reflects only inner-fold seeds, because train and test sets are fixed, so it is not shown.

| Features / rate | Direction | Test AUC | EER | HTER | per-group AUC | CIT AUC / HTER |
|---|---|---|---|---|---|---|
| P4 LR 1000 Hz | no-CM -> CM | 0.696 | 0.348 | 0.336 | G3 0.842, G4 0.598, G5 0.634 | 0.793 / 0.378 |
| P4 LR 1000 Hz | CM -> no-CM | 0.889 | 0.188 | 0.187 | G1 0.899, G2 0.879 | 0.985 / 0.167 |
| P4 LR 60 Hz | no-CM -> CM | 0.704 | 0.342 | 0.343 | G3 0.843, G4 0.612, G5 0.639 | 0.809 / 0.400 |
| P4 LR 60 Hz | CM -> no-CM | 0.897 | 0.161 | 0.185 | G1 0.915, G2 0.877 | 0.989 / 0.156 |
| P4_z LR 1000 Hz | no-CM -> CM | 0.706 | 0.346 | 0.325 | G3 0.846, G4 0.609, G5 0.649 | 0.787 / 0.378 |
| P4_z LR 1000 Hz | CM -> no-CM | 0.903 | 0.181 | 0.199 | G1 0.908, G2 0.896 | 0.983 / 0.178 |
| P4_z LR 60 Hz | no-CM -> CM | 0.705 | 0.346 | 0.329 | G3 0.847, G4 0.611, G5 0.646 | 0.794 / 0.389 |
| P4_z LR 60 Hz | CM -> no-CM | 0.906 | 0.161 | 0.201 | G1 0.917, G2 0.894 | 0.989 / 0.150 |
| ALL_z LR 1000 Hz | no-CM -> CM | 0.713 | 0.333 | 0.322 | G3 0.853, G4 0.615, G5 0.657 | 0.799 / 0.378 |
| ALL_z LR 1000 Hz | CM -> no-CM | 0.901 | 0.178 | 0.203 | G1 0.913, G2 0.887 | 0.983 / 0.161 |
| ALL_z LR 60 Hz | no-CM -> CM | 0.712 | 0.334 | 0.330 | G3 0.853, G4 0.614, G5 0.652 | 0.805 / 0.389 |
| ALL_z LR 60 Hz | CM -> no-CM | 0.908 | 0.158 | 0.191 | G1 0.920, G2 0.893 | 0.991 / 0.183 |
| Published table + SVM (released aggregation, for comparison) | no-CM -> CM | 0.993 | 0.028 | 0.113 | G3 0.980, G4 0.999, G5 0.999 | 1.000 / 0.011 |
| Published table + SVM | CM -> no-CM | 0.995 | 0.023 | 0.030 | G1 0.999, G2 0.990 | 1.000 / 0.011 |

Result: training on the no-countermeasure groups gives AUC 0.70 to 0.71 on the countermeasure groups. G3 (simple CM) transfers well (0.84 to 0.85), but G4 (0.60 to 0.62) and G5 (0.63 to 0.66) stay at the within-group level (0.56 to 0.63 in section 3b). Training on countermeasure participants and testing on no-countermeasure participants works about as well as within-group training (0.89 to 0.91 vs 0.87 to 0.93 for G1/G2 within-group). So the weak G4 and G5 result does not come from a training-set mismatch: with these label-free features, the signal in those groups is weak whichever group trains the model. The rate (60 vs 1000 Hz) changes none of this. Caveats: the two blocks use different trial windows (64 vs the last 32 trials), and each block has about 30 participants.

### Idea 3: new raw features (feature-set ablation, LR unless noted, 1000 Hz)

See the main table above. Mean AUC: P4 LR 0.779, P4+bins 0.781, P4+NEW 0.776, ALL 0.779; NEW alone 0.523, pupil only 0.511, first-look only 0.532. The new pupil, first-look and microsaccade features carry essentially no famous-vs-unknown signal on their own (AUC 0.51 to 0.53, MCC 0.005 to 0.033) and add nothing when combined with P4 (Holm p = 1.0 in every case). Within-participant z-scoring gives the only visible gain: P4_z LR 0.790 and ALL_z LR 0.789 vs P4 LR 0.779, with the largest differences in G4 (P4_z LR 0.625 vs 0.601) and G5 (ALL_z LR 0.661 vs 0.620). The gain is within noise after Holm correction (P4_z LR: +0.0017 per-participant AUC vs P4 SVM, Holm p = 1.0). Microsaccades were extracted only at 1000 and 500 Hz (>= 500 Hz rule). The pupil feature is a simple late-window, baseline-subtracted mean; a richer pupil model was not tried.

### Idea 4: model comparison at equal AUC (1000 Hz)

| Model | Features | AUC | EER | HTER | Paired vs P4 SVM (per-participant AUC), Holm |
|---|---|---|---|---|---|
| SVM (paper's) | P4 | 0.776 | 0.278 | 0.276 | reference |
| LR | P4 | 0.779 | 0.275 | 0.274 | +0.0033, Holm p = 1.0 |
| LightGBM (small: 50 or 150 trees, 7 leaves) | P4 | 0.757 | 0.302 | 0.293 | -0.0222, raw p = 0.0013, Holm p = 0.018 (worse) |
| LR | ALL_z | 0.789 | 0.264 | 0.260 | -0.0021, Holm p = 1.0 |
| SVM | ALL_z | 0.781 | 0.266 | 0.264 | -0.0116, Holm p = 1.0 |
| LightGBM | ALL_z | 0.780 | 0.278 | 0.273 | -0.0073, Holm p = 1.0 |

LR is statistically indistinguishable from the SVM at equal AUC and its model is about 76 times smaller (section 6). LightGBM on the four paper features is reliably worse (Holm p = 0.018); with the larger z-scored set it matches the SVM.

## 5. Best candidate vs published vs honest baseline

**Selection.** The stats stage picks the configuration with the highest mean per-participant AUC among the 14 candidates, which is `P4_lr_1000` (LR on the four label-free P4 features, 1000 Hz). The choice is weak: per-participant AUC differences among the label-free candidates other than LightGBM are at most 0.012 and none is significant after Holm correction. By pooled image AUC, the highest are P4_z LR and ALL_z LR (0.790 and 0.789 at 1000 Hz; ALL_z LR 0.791 at 500 Hz). We therefore state the result as "no reliable winner among label-free options", with P4 LR the simplest and cheapest member. We do not claim a winning idea.

| Quantity | Published (paper) | Honest baseline (published table + their SVM, our protocol) | Label-free P4 SVM 1000 Hz | Best candidate P4 LR 1000 Hz |
|---|---|---|---|---|
| Mean AUC over the six sets (image level, pooled) | 0.957 (mean of the six published AUCs, per-participant means) | 0.995 | 0.776 | 0.779 |
| Mean per-participant AUC | 0.957 | 0.995 | 0.794 | 0.798 |
| Mean EER / HTER | not reported | 0.016 / 0.020 | 0.278 / 0.276 | 0.275 / 0.274 |
| AUC in G4 / G5 (enhanced CM / CM + feedback) | 0.99 / 1.00 | 1.000 / 1.000 | 0.572 / 0.618 | 0.601 / 0.620 |

Deltas: best candidate minus honest baseline = -0.216 (per-participant AUC, 75 participants, Wilcoxon, p < 0.0001, Holm p < 0.0001; the same holds for all 14 candidates). Best candidate minus published = about -0.16 (per-participant) to -0.18 (pooled). The label-free pipeline does not beat the published result; the reason is described in 3b.

**Participant-cluster bootstrap 95% CI of the image-level AUC (500 resamples; repeat-averaged OOF scores).**

| Set | Published table + SVM | P4 SVM 1000 Hz | P4 LR 1000 Hz (best candidate) |
|---|---|---|---|
| G1-64 | [0.999, 1.000] | [0.867, 0.957] | [0.855, 0.954] |
| G2-64 | [0.987, 0.999] | [0.848, 0.934] | [0.818, 0.930] |
| G3-64 | [0.998, 1.000] | [0.691, 0.952] | [0.697, 0.950] |
| G3-32 | [0.963, 0.994] | [0.719, 0.911] | [0.716, 0.914] |
| G4-32 | [1.000, 1.000] | [0.469, 0.678] | [0.502, 0.697] |
| G5-32 | [1.000, 1.000] | [0.536, 0.701] | [0.536, 0.704] |

For P4 SVM, the G4 CI includes 0.5 and the G5 CI barely excludes it. The participant-level CIT bootstrap CIs saved in `results.json` (`ci`) are degenerate (14 to 15 participants per set; each yields one decision per class) and are not interpreted.

**Permutation test (best candidate).** Famous/unknown labels were shuffled within each participant (30 permutations; LR on P4 at 1000 Hz; repeat 0 of every set; statistic = mean pooled AUC over the six sets). Observed 0.777; null mean 0.497; null 95th percentile 0.522; p = 0.032, the minimum attainable with 30 permutations (1/31). The signal is real, but 30 permutations give only a coarse bound.

**Why it works (mechanism).** The signal is carried by the paper's own four features: dwell, fixation count and visit count on the face. Adding finer time-course bins, pupil, first-look or microsaccade features gives no reliable gain (idea 3), and the new features alone are at chance. Count features alone give AUC 0.745 (all presentations, SVM), so most of the label-free signal is how often the face is looked at. We did not run a formal feature-importance or ablation beyond these feature-set comparisons.

**Observations on the released data, pending an answer from the authors.** (a) In the released per-face table, the `n_fix` and `n_visit` columns agree with a rule that sums famous faces over all of their presentations but unknown faces only over presentations in trials without a famous face (exact match for 80% of faces in the 64-trial table and 91% in the 32-trial table; r = 0.925 and 0.965). Under that rule the number of presentations counted depends on the label (4.0 for famous vs 2.37 for unknown in the 64-trial table), and this count alone gives AUC 0.943. (b) We could not reproduce the released dwell columns from the raw samples under any aggregation rule we tried (r about 0.3; within-class SD 0.018 released vs 0.05 to 0.06 rebuilt; released `dwell_2` alone gives AUC 0.998 vs 0.818 rebuilt). We do not know why and make no claim about the cause. Both points are observations on the released files; we have not verified them with the authors, and the shared files do not document how presentations were selected or how the dwell windows were defined. A question to the authors is needed before anything is said about this in the paper.

Caveats: labels and inclusion decisions (misidentified faces, excluded trials) are taken from the authors' BookFIP tables, not re-derived. Groups have 14 to 15 participants, so every set-level CI is wide. The 3 repeats share the same participants, so SDs over repeats understate uncertainty; the bootstrap CIs are the better guide.

## 6. Efficiency

Hardware (`hardware_info()`): Windows 11 (10.0.26200), Intel64 Family 6 Model 142 Stepping 10, 8 logical CPUs, Python 3.14.6; single-threaded (OMP_NUM_THREADS=1) and measured while another experiment was using the machine, so absolute times are indicative. Inference uses the model fitted on all 64-trial participants at 1000 Hz (SVM C=1, gamma=0.1; LR C=0.01; LightGBM 150 trees x 7 leaves). Features are the four P4 features.

| Item | SVM (paper's model) | LR (best candidate) | LightGBM |
|---|---|---|---|
| Model size (pickled pipeline, bytes) | 110,953 | 1,463 | 138,794 |
| Stored parameters (incl. scaler/imputer) | 10,497 | 21 | 3,900 |
| Latency per image row, p50 / p95 (ms) | 3.70 / 6.27 | 2.36 / 3.93 | 2.46 / 3.22 |
| Latency per trial (4 faces), p50 / p95 (ms) | 5.88 / 17.47 | 2.37 / 4.60 | 2.49 / 2.80 |
| Latency per participant (64 images), p50 / p95 (ms) | 60.5 / 78.6 | 2.41 / 2.76 | 3.73 / 4.08 |
| Throughput (images/s, model only) | 875 | 26,684 | 17,163 |
| Peak memory during inference (bytes) | 12,006 | 12,052 | 13,238 |

Model-only latency for LR and LightGBM is dominated by the fixed overhead of the scikit-learn pipeline call (about 2.4 ms), not by the arithmetic.

**End-to-end from raw samples** (simulated sensor output of one participant, 64 trials x 5.5 s: anti-alias and decimation, I-VT, feature extraction, per-image aggregation, model). Excludes EDF decoding. Input samples include the 500 ms pre-onset window; real-time factor = (samples/s) / sensor rate. Peak memory is the extra allocation during one participant.

| Model | Rate | Input samples | Per trial p50 / p95 (ms) | Per participant p50 / p95 (ms) | Samples/s | Real-time factor | Peak memory (bytes) |
|---|---|---|---|---|---|---|---|
| SVM | 1000 Hz | 352,000 | 23.2 / 32.5 | 827.7 / 939.6 | 425,274 | 425x | 627,124 |
| SVM | 250 Hz | 88,000 | 13.5 / 17.4 | 525.0 / 633.9 | 167,615 | 670x | 210,367 |
| SVM | 60 Hz | 21,120 | 12.3 / 16.2 | 489.2 / 582.0 | 43,172 | 720x | 106,344 |
| SVM | 30 Hz | 10,560 | 11.7 / 16.6 | 425.1 / 482.1 | 24,839 | 828x | 94,832 |
| LR | 1000 Hz | 352,000 | 12.3 / 14.4 | 643.5 / 713.0 | 547,017 | 547x | 627,080 |
| LR | 250 Hz | 88,000 | 10.6 / 15.1 | 520.9 / 714.3 | 168,954 | 676x | 210,499 |
| LR | 60 Hz | 21,120 | 9.3 / 11.7 | 431.3 / 659.5 | 48,967 | 816x | 106,388 |
| LR | 30 Hz | 10,560 | 9.0 / 28.2 | 402.2 / 498.6 | 26,255 | 875x | 94,788 |
| LightGBM | 1000 Hz | 352,000 | 12.8 / 19.1 | 628.3 / 839.8 | 560,258 | 560x | 627,036 |
| LightGBM | 60 Hz | 21,120 | 9.5 / 10.2 | 441.7 / 617.1 | 47,813 | 797x | 107,048 |
| Best candidate (P4 LR) | 1000 Hz | 352,000 | 12.3 / 12.8 | 614.2 / 729.5 | 573,147 | 573x | 626,992 |

The pipeline runs 425 to 900 times faster than real time on a laptop CPU at every rate. The cost is dominated by raw-sample processing (I-VT, pupil cleaning, and the microsaccade detector at >= 500 Hz), not by the classifier. Peak memory scales with the input samples and falls about six-fold from 1000 Hz to 60 Hz (627 kB to 106 kB). The real-time factor is higher at low rates because part of the per-trial cost is fixed. Because that fixed part is large, the LR advantage in end-to-end time is only about 22% at 1000 Hz, while its advantage in model-only size and latency is large. Efficiency was measured with the P4 features only: ALL_z LR (the highest-AUC variant) needs the extra pupil and first-look features, and its raw-to-score time was not measured separately. Per-participant z-scoring needs all 64 images of a participant, so P4_z and ALL_z cannot be applied trial by trial.

Compute note: the whole screening took about 4.9 h wall clock (`results.json`, `timing_s`), of which about 4.3 h was the stats stage (permutation test plus bootstraps). That is far more than the compute would need; we did not investigate the cause (shared CPU, and the machine may have been suspended).

## 6b. Identity-leakage probe (does trial-level gaze identify the participant, and does the sampling rate matter?)

Multinomial LR on 9 per-trial gaze summaries (and + 2 pupil summaries) to identify one of 75 participants; train on odd trials, test on even and back. Chance = 0.013.

| Rate | Gaze only (top-1) | Gaze + pupil (top-1) |
|---|---|---|
| 1000 Hz | 0.352 | 0.605 |
| 500 Hz | 0.355 | 0.607 |
| 250 Hz | 0.353 | 0.597 |
| 120 Hz | 0.300 | 0.550 |
| 60 Hz | 0.250 | 0.483 |
| 30 Hz | 0.199 | 0.429 |

Trial-level gaze summaries identify the participant 15 to 45 times above chance, and identification falls steadily with the sampling rate (gaze + pupil 0.605 at 1000 Hz to 0.429 at 30 Hz). Lower sampling rates therefore reduce, but do not remove, identity leakage. This is a probe of what a trial-level summary can reveal, not a privacy guarantee; it uses within-session splits of the same participants.

## 7. Recommendation

1. **The main finding is the audit, not a new model.** On the released data the per-face table behaves differently from what the paper's text describes: the count columns follow a label-dependent rule and the dwell columns do not match the raw data we rebuilt. These are observations on the released files, pending an answer from the authors. With label-free features rebuilt from the raw samples the same task gives mean AUC 0.78 (EER 0.27), and detection is weak to near chance under the enhanced and feedback countermeasures (AUC 0.57 to 0.66, EER 0.38 to 0.46). The paper's conclusion that the STM-CIT resists countermeasures is not supported by label-free features computed from the raw samples; it rests on features whose construction we could not reproduce.
2. **Sampling rate:** 60 Hz is non-inferior to 1000 Hz (margin 0.02 AUC, Holm-corrected) for both the SVM on the paper's features and LR on the larger set; 30 Hz is borderline and not proven. A 60 Hz sensor is a reasonable target for these features. Identity leakage also falls with the rate.
3. **Model:** use LR rather than the SVM or LightGBM for these features: same accuracy, 1.5 kB, 21 parameters, about 25x faster model-only inference. Do not expect gains from the new pupil, first-look or microsaccade features.
4. **Worth a full run:** yes, but of the audit and the data-provenance question, not of a new classifier. Priorities: (i) obtain the authors' table-construction code or a description of the dwell windows and presentation selection; (ii) rerun with full-run settings (more repeats, a larger permutation count, selection on held-out data); (iii) test countermeasure-robust features on larger groups; the only hint so far is within-participant z-scoring (up to +0.04 AUC in G4 and G5), which is not yet reliable. Nothing in this track justifies a claim that a new method beats the published numbers.

## Files

`results.json` (all numbers; `noninferiority_margin_0.02_holm` holds the non-inferiority table), `noninferiority.json`, `tables.md` (per-set tables for every configuration), `folds.json`, `checkpoints/` (per-configuration metrics and OOF scores), `oof_scores.csv.gz`, `progress.md`, `run_log_part1.txt` and `run_log.txt`, `noninf_log.txt`, and the scripts `scripts/phase2_delmas_raw_extract.py`, `scripts/phase2_delmas_raw.py`, `scripts/phase2_delmas_raw_tables.py`, `scripts/phase2_delmas_raw_noninf.py`, with `src/edd/delmas_raw.py`.
