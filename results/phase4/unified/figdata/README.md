# Figure data for the unified pipeline (Phase 4)

All files are tidy CSVs written by `scripts/phase4_unified.py --stage figdata`. Every number comes from out-of-fold scores (the
decision unit never saw its own participant in training). `dataset` is one of delmas, chen, celniak, preti, mazza; `task` names the
frozen-fold task (Delmas: one per group-set; Celniak: unified plus labelled control/reference; Preti: Level A/B plus contrast-only
sensitivity; Mazza: g2_vs_g3, guilty_vs_innocent). No charts are produced here; the senior analyst draws them.

`level`: `row` = the trial/image/slide row scored by the model (Delmas image, Chen trial, Celniak slide); `person` = the participant-
level decision statistic (Celniak, Preti, Mazza: mean of the participant's row scores; Delmas: CIT z statistic of known vs
irrelevant images, negatives are simulated innocents; Chen: one-sided-t z of familiar vs control trials, negatives are control-1-vs-2).
`model`: `unified` = this pipeline; `baseline` = the stored honest baseline of the track on the same folds (person-level only).

## roc_det_curves.csv
One row per (dataset, task, level, model, fpr grid point), 101 points from 0 to 1. `tpr` = ROC point estimate (interpolated on the
FPR grid), `tpr_lo/tpr_hi` = 2.5/97.5 percentile of a participant-cluster bootstrap (resampling participants with replacement,
500 draws, the same draws for the same dataset/task). DET = `fpr` against `fnr` (= 1 - tpr; bands `fnr_lo = 1 - tpr_hi`,
`fnr_hi = 1 - tpr_lo`). Scores are repeat-averaged out-of-fold scores.

## roc_det_eer.csv
One row per curve: `eer` = equal error rate point (mean of FPR and FNR at the closest crossing), `eer_lo/eer_hi` bootstrap 95% interval,
`eer_score_threshold` = score at that point (units of the score; logit for row level), `auc` with bootstrap interval, `n_units`
(participants), `n_decisions` (rows), `n_boot` valid draws.

## score_distributions.csv
One row per decision unit (`uid`; for Delmas `participant_image`, Chen the trial row index, otherwise the participant): repeat-averaged
out-of-fold `score`, label `y` (1 = positive class), `unit` (participant), `level`. `threshold_eer_calibrated` and
`threshold_fpr7_calibrated` = mean over outer folds and repeats of the thresholds calibrated on training participants only (person level
only; statistic units); `threshold_logit_zero` = 0, the LR decision threshold, for row level.

## learning_curve.csv
`fraction_of_training_participants` 0.25/0.5/0.75/1.0 (each outer fold's training participants are subsampled, stratified by class when
the label belongs to the person; C is re-tuned in nested folds). `auc_mean`/`auc_sd_over_repeats`: primary-level AUC per repeat, mean and
SD. `auc_repeat_avg_scores` with `auc_lo/auc_hi`: participant-cluster bootstrap (300 draws) on repeat-averaged scores. `n_train_participants_mean` =
mean number of training participants actually used.

## validation_curve.csv
One row per (task, C) on the fixed grid. `heldout_logloss` (unweighted log-loss of sigmoid(logit), row level) and `heldout_auc` (primary
level), mean and SD over repeats, from models refitted at each C on the outer-training participants. `train_logloss`/`train_auc` =
resubstitution values averaged over outer folds. `inner_cv_auc` = the selection criterion (mean per-fold inner AUC) averaged over outer
folds; `fraction_of_folds_choosing_C` = how often nested tuning picked that C.

## feature_weights.csv
Standardised coefficients (the model is fitted on standardised inputs): `coef_mean`, `coef_lo/coef_hi` = 2.5/97.5 percentile across all outer
folds x repeats, `share_positive`, `n_models`, `abs_rank` (1 = largest |mean|). Positive = towards the positive class (known face /
guilty / liar / familiar).

## calibration_bins.csv
Equal-count bins (up to 10, fewer for small n) of repeat-averaged out-of-fold probability sigmoid(logit): `mean_predicted`,
`observed_positive_rate` with Wilson 95% `obs_lo/obs_hi`, `n`. The model uses balanced class weights, so probabilities refer to a balanced
prior, not to the sample's base rate (supplementary plot).

## chen_rate_curve.csv (analysis A)
One row per (model, rate): `model` unified or I2b or rate_rule, `rate_hz` 100/60/30, `trial_auc` (repeat-averaged out-of-fold scores pooled over trials), `person_auc`, `detection_fpr7` (person detection at the calibrated FPR <= 7% threshold), `realised_fpr`, and for non-rate-rule rows the detection difference to the rate rule at 100 Hz with its bootstrap 95% interval. The non-inferiority and superiority tests are in `results_chen_rate.json`.

## webcam_adaptive_curve.csv / webcam_adaptive_thresholds.csv (analysis B)
Curve: one row per (rule, level_deg): `auc_level` (image-level AUC, mean over sets and draws), `auc_reference` (clean P4 SVM 1000 Hz), `delta`, `ci_lo/ci_hi` (paired bootstrap 95%), `lower_one_sided_95`, `p_inferior` (H0: delta <= -0.02), `p_holm` (Holm over the six levels within the rule), `non_inferior_holm`. `rule` ek = primary median-based threshold, nh = secondary Nystrom-Holmqvist. Thresholds: one row per (rule, level, draw, participant): the data-driven fixation velocity threshold (deg/s), the peak threshold (nh only), and the realised calibration offset of that participant's noise draw (deg).

## efficiency.csv
Same session, single thread, interleaved rounds, baseline vs unified, per dataset: `latency_p50_ms`/`latency_p95_ms` per decision unit
(median over rounds of each round's percentile), `throughput_per_s`, `samples_per_s` and `real_time_factor` (where a raw sensor signal is
processed; real-time factor = samples per second / sensor rate), `peak_memory_kb` (tracemalloc), `model_size_bytes` (pickle), `parameters`,
`rounds`, `runs_per_round`. `stage` says what is timed (extraction + model, from which input).
