# Task 2.2: lowest usable sampling rate (card)

**Status: EXPLORATORY. Not preregistered.** Plan: `docs/PLAN.md` Task 2.2. Output folder `results/phase9/low_rate/`. Scripts: `s1_extract.py` (EDF -> 30/20/15/10 Hz samples and features), `s2_delmas_curve.py`, `s3_chen_curve.py`, `s4_webcam.py`, `s5_events_timing.py`, `s6_assemble.py`, helper `lr_common.py`, chain `run_all.sh`; logs `log_*.txt`; per-stage outputs `rows_*.csv`, `results_*.json`, `repro_*.json`. One process, single thread throughout; no existing script or result was edited (existing code is imported unchanged; the source of each reused piece is stated at the top of each script).

## What was run

All settings other than the sampling rate are those of the originals: same frozen folds, seeds, tuning grids and nested cross-validation.

1. **Delmas et al. (EyeLink, 1000 Hz recordings, 75 participants, six group sets).** Source: `scripts/phase2_delmas_raw.py` stage `rate`, drawn by `scripts/phase5_figures.py` into `paper/figures/data_delmas_rate_curve.csv`. New rates 20, 15 and 10 Hz made by anti-aliased polyphase resampling of the 1000 Hz samples (`edd.delmas_raw.degrade`), I-VT 30 deg/s with 60 ms minimum, the same per-face features. Two models: the published four features with an RBF-SVM (`P4_svm`), and our 13 standardised features (no microsaccades) with L2 logistic regression (`ALLz_lr`). Metric: image-level AUC, pooled within each of the six sets, mean over sets (repeat-averaged out-of-fold scores). The 1000 Hz row is the stored result.
2. **Chen et al. (released 100 Hz epochs, trial-level AUC).** Source: `scripts/phase4_unified.py` stage `chen_rate` (`results/phase4/unified/results_chen_rate.json`). New rates 20, 15, 10 Hz by anti-aliased polyphase resampling of the 100 Hz epochs (`edd.unified.decimate_to_rate`). Two models: our pipeline (`unified`: 25 x 100 ms bins of pupil and gaze, within-participant z-scores, L2 logistic regression) and `I2b` (pupil bins + gaze, 125 features). The 100 Hz row is the stored result.
3. **Delmas, measured-webcam-noise condition** (`scripts/phase6_trackC_c1.py` condition `anch`: 1.45 deg mean offset per participant, per-sample jitter SD 1.66/sqrt(2) deg (x) and 1.42/sqrt(2) deg (y), 3% sample loss; three noise draws), at 30, 20, 15 and 10 Hz, no pupil features: dwell computed directly from gaze samples on each face area (`sb`, 10 features) against fixed-threshold event detection (I-VT 30 deg/s, 60 ms, 11 features). A noise-free (clean) run of each route at each rate is the reference. Offsets are the same per participant at every rate; jitter and loss are drawn per sample at that rate.
4. **Event-detection diagnostic** (Delmas, clean data) like Supplementary Table S10 (`scripts/phase6_trackC_c2.py` functions): fixations per trial, mean fixation duration, smallest saccade the velocity window can detect (30 deg/s x window), and Pearson r of per-item fixation count and per-item dwell with the tracker's own (EyeLink parser) values.
5. **Processing time per participant** (Delmas, 64 trials x 5.5 s of 1000 Hz samples = 352 s of data) from the 1000 Hz samples to per-image scores, median of 3 rounds per participant after a warm-up, median over 5 participants, one thread.

## Reproduction check (done before any new rate; all passed with zero difference)

| Quantity at 30 Hz | Stored | Re-run here | Max abs difference in out-of-fold scores |
|---|---|---|---|
| Delmas, 13 features + LR, AUC (repeat-averaged, pooled per set, mean over sets) | 0.7854 | 0.7854 | 0.0 |
| Same model, mean of per-fold AUCs (screening log, "0.781") | 0.7811 | 0.7811 | |
| Delmas, published 4 features + SVM, AUC | 0.7715 | 0.7715 | 0.0 |
| Chen, our pipeline, trial AUC | 0.6016 | 0.6016 | 0.0 |
| Chen, pupil bins + gaze (I2b), trial AUC | 0.6044 | 0.6044 | 0.0 |
| Webcam study, I-VT, no pupil, clean, AUC (Phase 7 Track D `d1_table.csv`) | 0.7863 | 0.7863 | |
| Webcam study, sample dwell, no pupil, clean, AUC (Phase 7 Track D `d1_table.csv`) | 0.7732 | 0.7732 | |
| Webcam study, I-VT, no pupil, measured noise, AUC (Phase 7 Track D `d1_table.csv`) | 0.7344 | 0.7344 | |
| Webcam study, sample dwell, no pupil, measured noise, AUC (Phase 7 Track D `d1_table.csv`) | 0.7737 | 0.7737 | |
| Event diagnostic: fixations per trial / mean fixation duration (Track C2) | 9.739 / 392.3 ms | 9.739 / 392.3 ms | |

Also identical: the 19,200 per-face feature rows at 30 Hz against `data/derived/delmas_raw_face_trials.csv.gz` (max difference 4.5e-13, floating-point only), the 30 Hz decimated samples against the earlier webcam cache, and the edge-based Chen feature builders used for 20/15/10 Hz against `unified.chen_features` / `chen_i2b_features` at 30 Hz (difference 0.0).

## Result 1. Delmas, clean EyeLink data, image-level AUC

AUC CI: participant-cluster bootstrap, first 300 of 2000 resamples (as the original curve). Delta CI: all 2000 resamples, paired (all rates evaluated on the same resamples).

**published 4-feature SVM**

| Rate (Hz) | AUC [95% CI] | Delta vs 30 Hz [95% CI] | Within 0.02 of 30 Hz |
|---|---|---|---|
| 1000 (stored) | 0.7811 [0.7335, 0.8212] | +0.0096 [-0.0035, +0.0223] | yes (CI also) |
| 30 | 0.7715 [0.7240, 0.8129] | (reference) |  |
| 20 | 0.7748 [0.7250, 0.8180] | +0.0033 [-0.0059, +0.0129] | yes (CI also) |
| 15 | 0.7700 [0.7231, 0.8124] | -0.0015 [-0.0101, +0.0073] | yes (CI also) |
| 10 | 0.7547 [0.7108, 0.7969] | -0.0168 [-0.0274, -0.0052] | yes |

**our 13 standardised features, L2 logistic regression**

| Rate (Hz) | AUC [95% CI] | Delta vs 30 Hz [95% CI] | Within 0.02 of 30 Hz |
|---|---|---|---|
| 1000 (stored) | 0.7929 [0.7470, 0.8307] | +0.0075 [-0.0049, +0.0204] | yes (CI also) |
| 30 | 0.7854 [0.7393, 0.8244] | (reference) |  |
| 20 | 0.7777 [0.7321, 0.8161] | -0.0077 [-0.0160, +0.0010] | yes (CI also) |
| 15 | 0.7709 [0.7218, 0.8108] | -0.0146 [-0.0246, -0.0021] | yes |
| 10 | 0.7505 [0.7045, 0.7881] | -0.0349 [-0.0505, -0.0190] | NO |

## Result 2. Chen, trial-level AUC

2000 paired participant-cluster resamples (same seed and scheme as the stored analysis).

**our pipeline (unified)**

| Rate (Hz) | AUC [95% CI] | Delta vs 30 Hz [95% CI] | Within 0.02 of 30 Hz |
|---|---|---|---|
| 100 (stored) | 0.6016 [0.5813, 0.6219] | +0.0001 [-0.0035, +0.0037] | yes (CI also) |
| 30 | 0.6016 [0.5806, 0.6217] | (reference) |  |
| 20 | 0.6015 [0.5805, 0.6222] | -0.0000 [-0.0015, +0.0016] | yes (CI also) |
| 15 | 0.6014 [0.5811, 0.6220] | -0.0001 [-0.0034, +0.0033] | yes (CI also) |
| 10 | 0.6042 [0.5829, 0.6252] | +0.0027 [-0.0003, +0.0058] | yes (CI also) |

**pupil bins + gaze (I2b)**

| Rate (Hz) | AUC [95% CI] | Delta vs 30 Hz [95% CI] | Within 0.02 of 30 Hz |
|---|---|---|---|
| 100 (stored) | 0.6064 [0.5851, 0.6278] | +0.0020 [-0.0024, +0.0061] | yes (CI also) |
| 30 | 0.6044 [0.5850, 0.6248] | (reference) |  |
| 20 | 0.6054 [0.5858, 0.6255] | +0.0009 [-0.0003, +0.0022] | yes (CI also) |
| 15 | 0.6046 [0.5855, 0.6244] | +0.0002 [-0.0020, +0.0025] | yes (CI also) |
| 10 | 0.6054 [0.5854, 0.6261] | +0.0009 [-0.0009, +0.0028] | yes (CI also) |

Adaptations needed for rates that do not split 100 ms into whole samples (the originals only handle rates that do): bin means use sample edges round(k x rate / 10) (one or two samples per 100 ms bin at 15 Hz; identical to the original reshape at the other rates), and the I2b pupil-rate smoothing window (250 ms, `round(0.25 x rate)` samples) would be 2 samples at 10 Hz, which makes the Hanning weights 0/0, so a 3-sample window (300 ms) is used at 10 Hz.

## Result 3. Delmas, webcam noise, dwell from samples versus I-VT, no pupil features

Mean over 3 noise draws; AUC CI first 300 of 2000 paired resamples, delta CI all 2000 (resampling the mean over draws).

**I-VT event detection 30 deg/s, 60 ms (no pupil); clean, no added noise**

| Rate (Hz) | AUC [95% CI] | Delta vs 30 Hz [95% CI] | Within 0.02 of 30 Hz |
|---|---|---|---|
| 30 | 0.7863 [0.7409, 0.8241] | (reference) |  |
| 20 | 0.7786 [0.7292, 0.8208] | -0.0077 [-0.0155, +0.0005] | yes (CI also) |
| 15 | 0.7718 [0.7221, 0.8170] | -0.0145 [-0.0251, -0.0018] | yes |
| 10 | 0.7520 [0.7049, 0.7929] | -0.0342 [-0.0511, -0.0181] | NO |

**sb: dwell from samples, no event detection (no pupil); clean, no added noise**

| Rate (Hz) | AUC [95% CI] | Delta vs 30 Hz [95% CI] | Within 0.02 of 30 Hz |
|---|---|---|---|
| 30 | 0.7732 [0.7208, 0.8160] | (reference) |  |
| 20 | 0.7743 [0.7215, 0.8170] | +0.0012 [-0.0021, +0.0045] | yes (CI also) |
| 15 | 0.7747 [0.7230, 0.8187] | +0.0015 [-0.0034, +0.0067] | yes (CI also) |
| 10 | 0.7682 [0.7178, 0.8111] | -0.0050 [-0.0103, +0.0007] | yes (CI also) |

**I-VT event detection 30 deg/s, 60 ms (no pupil); measured webcam values (1.45 deg offset, ~1.1 deg jitter, 3% loss), mean of 3 draws**

| Rate (Hz) | AUC [95% CI] | Delta vs 30 Hz [95% CI] | Within 0.02 of 30 Hz |
|---|---|---|---|
| 30 | 0.7344 [0.6976, 0.7658] | (reference) |  |
| 20 | 0.7502 [0.7038, 0.7863] | +0.0158 [+0.0024, +0.0282] | yes (CI also) |
| 15 | 0.7673 [0.7201, 0.8072] | +0.0330 [+0.0172, +0.0473] | yes (CI also) |
| 10 | 0.7510 [0.7068, 0.7880] | +0.0166 [+0.0020, +0.0311] | yes (CI also) |

**sb: dwell from samples, no event detection (no pupil); measured webcam values (1.45 deg offset, ~1.1 deg jitter, 3% loss), mean of 3 draws**

| Rate (Hz) | AUC [95% CI] | Delta vs 30 Hz [95% CI] | Within 0.02 of 30 Hz |
|---|---|---|---|
| 30 | 0.7737 [0.7229, 0.8155] | (reference) |  |
| 20 | 0.7743 [0.7249, 0.8184] | +0.0006 [-0.0040, +0.0050] | yes (CI also) |
| 15 | 0.7764 [0.7259, 0.8212] | +0.0027 [-0.0012, +0.0068] | yes (CI also) |
| 10 | 0.7689 [0.7217, 0.8086] | -0.0048 [-0.0109, +0.0015] | yes (CI also) |

**Sample dwell minus I-VT at the same rate (paired)**

| Condition | Rate (Hz) | AUC difference [95% CI] |
|---|---|---|
| clean | 30 | -0.0131 [-0.0226, -0.0022] |
| clean | 20 | -0.0043 [-0.0133, +0.0060] |
| clean | 15 | +0.0029 [-0.0041, +0.0097] |
| clean | 10 | +0.0162 [+0.0030, +0.0318] |
| measured noise | 30 | +0.0394 [+0.0216, +0.0562] |
| measured noise | 20 | +0.0241 [+0.0121, +0.0365] |
| measured noise | 15 | +0.0091 [+0.0022, +0.0169] |
| measured noise | 10 | +0.0179 [+0.0059, +0.0308] |

## Result 4. Event-detection diagnostic (Delmas, clean data)

Reference: the tracker's own parser at 1000 Hz. r = Pearson correlation with the parser's per-item value over participant x item cells (trial x face and image-level values are in `event_diag_agreement_long.csv`).

| Rate (Hz) | Fixations / trial | Mean fix. duration (ms) | Median (ms) | Velocity window (ms) | Smallest saccade detectable (deg) | r fixation count | r dwell 0.2-1 s | r dwell 1-5 s | r sample dwell 0.2-1 s | r sample dwell 1-5 s |
|---|---|---|---|---|---|---|---|---|---|---|
| EyeLink parser (1000 Hz) | 16.37 | 256 | 219 |  |  |  |  |  |  |  |
| 30 | 9.74 | 392 | 333 | 67 | 2 | 0.682 | 0.993 | 0.989 | 0.989 | 0.982 |
| 20 | 7.77 | 467 | 400 | 100 | 3 | 0.577 | 0.984 | 0.978 | 0.987 | 0.981 |
| 15 | 7.56 | 459 | 400 | 133 | 4 | 0.544 | 0.977 | 0.967 | 0.986 | 0.980 |
| 10 | 6.81 | 450 | 400 | 200 | 6 | 0.538 | 0.949 | 0.930 | 0.982 | 0.974 |

**Per-item fixation counts and dwell, last two columns:** "sample dwell" is the share of valid samples on the face area with no event detection; "dwell" columns use the I-VT fixations.

## Result 5. Processing time per participant (Delmas, from 1000 Hz samples to per-image scores)

`resample` = anti-aliased resampling from 1000 Hz (a simulation step: a real sensor delivers the low rate directly, so it is not a deployed cost); `features + score` = I-VT, features, aggregation, z-scoring and the logistic-regression score. Median over 5 participants of the median of 3 rounds; min-max of the 5 participants' totals in brackets. This laptop's timings are noisy (the min-max range is wide, and the 15 and 10 Hz resampling medians vary between runs), so only large differences should be read.

**original loop code**

| Rate (Hz) | Resample (ms) | Features + score (ms) | Total (ms) [range] | Real-time factor (x) |
|---|---|---|---|---|
| 1000 (no resampling) | 4.6 | 237.7 | 242 [211-546] | 1452 |
| 30 | 231.0 | 113.0 | 359 [329-437] | 982 |
| 20 | 198.6 | 121.3 | 321 [289-392] | 1098 |
| 15 | 398.9 | 150.5 | 540 [462-1347] | 652 |
| 10 | 344.8 | 157.0 | 518 [334-744] | 679 |

**vectorised code**

| Rate (Hz) | Resample (ms) | Features + score (ms) | Total (ms) [range] | Real-time factor (x) |
|---|---|---|---|---|
| 1000 (no resampling) | 2.5 | 90.8 | 93 [64-161] | 3781 |
| 30 | 74.2 | 5.1 | 79 [70-95] | 4443 |
| 20 | 93.2 | 7.0 | 98 [73-132] | 3589 |
| 15 | 119.8 | 5.7 | 125 [87-157] | 2806 |
| 10 | 111.2 | 4.5 | 117 [78-174] | 3002 |

## Plain summary (5 lines)

1. Delmas, clean EyeLink: the published four-feature SVM stays within 0.02 AUC of 30 Hz down to 10 Hz (delta -0.017, CI lower limit -0.027, so 10 Hz is not shown within the margin; 15 Hz is, by the CI); our 13-feature LR stays within 0.02 down to 15 Hz (delta -0.015 at 15 Hz, CI lower limit -0.025; by the CI only 20 Hz) and loses 0.035 at 10 Hz.
2. Chen (100 Hz epochs, trial-level AUC): both models stay within 0.02 of 30 Hz at every rate tested, down to 10 Hz for both models; the largest change is 0.0027 AUC and the lower CI limits are above -0.004 (Chen's pupil signal is slow; it does not depend on the rate in this range).
3. Dwell computed from samples (no pupil) under the measured webcam noise stays within 0.02 of 30 Hz down to 10 Hz (CI-supported to 10 Hz; delta -0.005 at 10 Hz); it is not below the I-VT route at any rate under noise (+0.039 at 30 Hz, +0.018 at 10 Hz).
4. Event detection (I-VT) breaks at 10 Hz on clean data (delta -0.034; within 0.02 down to 15 Hz), and the fixation count it produces already agrees poorly with the tracker's parser at 30 Hz (r 0.68, falling to 0.54 at 10 Hz) while the smallest detectable saccade grows from 2 to 6 deg; under measured webcam noise I-VT stays within 0.02 of its own 30 Hz value (down to 10 Hz) only because its 30 Hz value is already low (it is higher at 20, 15, 10 Hz).
5. Dwell per item agrees with the tracker's parser at r >= 0.93 even at 10 Hz (sample dwell r >= 0.97), so dwell-based recognition measures need far fewer samples than fixation counts; the cost of the whole pipeline from resampled samples is about 5 ms (vectorised code) or 0.1-0.16 s (original loop code) per participant for features and score at every rate here, so rate is limited by accuracy, not compute.

## What this does and does not show

- Exploratory only: no preregistration; the 0.02 margin is the one used in the project's earlier rate analyses and is applied here to the point estimate and, separately, to the lower CI limit (columns `within_0.02_of_30Hz_point` and `lower_CI_above_-0.02` in `rate_table.csv`). "Lowest rate within 0.02" is the lowest tested rate such that all tested rates from 30 Hz down to it are within the margin.
- The low rates are simulated by anti-aliased resampling of high-rate recordings (Delmas 1000 Hz, Chen 100 Hz) from chin-rest research trackers; webcam noise is simulated, not recorded. A real 15 Hz webcam stream may differ in timing jitter, dropped frames and signal filtering; these are not simulated.
- Differences between neighbouring rates are mostly inside the intervals (Delmas 75 participants, six group sets that include strong-countermeasure groups with AUC near 0.55); the 10 Hz losses of the 13-feature LR (-0.035) and of I-VT on clean data (-0.034) are the only ones whose point estimate is beyond the margin; their intervals exclude zero but still reach the margin edge (upper limits -0.019 and -0.018). The Delmas SVM curve is non-monotonic (20 Hz above 30 Hz), which is noise.
- Under measured webcam noise the I-VT route is better at 20/15/10 Hz than at 30 Hz. A plausible reason (not tested here) is that per-sample jitter produces velocity of jitter/window, and the velocity window widens as the rate falls (67 ms at 30 Hz, 133 ms at 15 Hz), so fewer noise samples cross the 30 deg/s threshold. It is not a recommendation to lower the rate for I-VT; under noise the sample-dwell route is higher at every rate (on clean data I-VT is higher at 30 Hz and 20 Hz, sample dwell at 10 Hz).
- Timing: the laptop's timings fluctuate between runs (a first run of the same script gave 78-106 ms vectorised totals across rates against 79-125 ms here); the resample column is an artefact of simulating a low-rate sensor from 1000 Hz data.

## Files

`rate_table.csv` (dataset, model, condition, rate, AUC, CI, delta vs 30 Hz with CI, within-margin flags, sample-minus-I-VT columns), `event_diag.csv` (+ `event_diag_agreement_long.csv`), `timing.csv` (+ `timing_per_participant.csv`), `results.json`, `repro_*.json`, the scripts above, and the per-stage checkpoints (`checkpoints_delmas/`, `checkpoints_chen/`, `checkpoints_webcam/`, `cache/`, `derived/`).