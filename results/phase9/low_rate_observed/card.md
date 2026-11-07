# Task 2.2b: observed-webcam condition for the frame-rate curve (card)

**Status: EXPLORATORY. Not preregistered.** Answers reviewer comment M2 and parts of M1, M3, M4 (`paper_rec/review/review_1.md`). Plan and parameter sources: `PLAN.md` (SHA-256 `f735ed8c5f68fbf036a6f39cdf70b89d32c4cc89a965050bfe6998521323dbc1`, written 2026-11-07 00:24:10, after the reproduction check and before any observed-condition result; `PLAN.sha256.txt`). Output folder `results/phase9/low_rate_observed/`. Scripts: `lro_common.py`, `s0_measure.py`, `s0b_fit_jitter.py`, `s1_extract1000.py`, `s2_sim.py`, `s3_repro.py`, `s4_run.py`, `s5_stats.py`, `s6_tables.py`, `fig_rate_observed.py`. Existing code is imported unchanged (sources at the top of each script); nothing outside this folder was written; one process, one thread; the two `s6_perm.py` jobs were not touched. Numbers only: no camera frames or images were read or saved. Camera: the EMEET SmartCam C960 sessions only (user decision of 2026-11-06). Wording: known item / unknown item.

## Reproduction check (done before the full run; zero difference)

| Quantity (30 Hz, no pupil, dwell from samples) | Stored (`results/phase9/low_rate`, Phase 7 Track D) | Through this folder's code | Difference |
|---|---|---|---|
| Clean | 0.7731838961824132 | 0.7731838961824132 | 0.0 |
| Measured noise 'anch', mean of 3 draws | 0.7737061625087726 | 0.7737061625087726 | 0.0 |

Further checks, all passed: the 1000 Hz arrays reloaded here and passed through `edd.delmas_raw.degrade` at 30 Hz equal the earlier 30 Hz cache for all 75 participants (max difference 0.0); the bootstrap run here reproduces every stored row of `rows_webcam.csv` (clean and 'anch', both measures; max difference 1e-16, same seed and resamples); the pipeline with regular frame times and no error gives, for dwell from samples, exactly the clean AUC at every rate, and for I-VT the clean AUC at 20/15/10 Hz and 0.7852 against 0.7863 at 30 Hz (-0.0010 [-0.0026, 0.0000]; cause not investigated).

## Parameters measured from our own EMEET recordings (`measured/`)

Sessions: `test05_faces_K_emeet_glasses`, `test06_faces_I_emeet_glasses`, `test07_twofaces_K_dryrun`, `test08_v1_18trials`, `test09_v1_18trials` (`app/data/`; one person; `frames.csv`, `trials.csv`, `events.csv`, `meta.json`, numbers only).

| Piece | Measured value | Source and rule |
|---|---|---|
| (a) Per-trial offset | 79 scored trials with a passed drift check (6 of 85 failed: 3 above 6 deg, 3 too few frames; not in the distribution). Gaze offset = minus the logged `drift_dx_px`, `drift_dy_px` divided by each session's `px_per_deg`. Magnitude median 2.53, mean 2.64, 10th-90th percentile 1.12-4.24, range 0.32-5.90 deg. x mean -0.76 (SD 1.92), y mean +0.61 (SD 2.02), correlation 0.26. Per-session median magnitude 1.73, 3.01, 2.66, 2.02, 2.90 deg. | `trials.csv`; `drift_offsets.csv`. Drawn jointly, independently per trial. (`results/phase9/figures/fig_drift.csv`, test06 only, median 3.04 deg for scored trials, is consistent: it is one of the five sessions.) |
| (b) Gains about the screen centre | horizontal 0.88, vertical 0.79 (unrounded 0.877, 0.788; 11 calibration segments) | median over EMEET segments of the slope of target+error on target, the rule of `vertical_check.py`, on `results/app_pilot/vertical_check/vertical_check_targets.csv` (base variant). With the built-in-camera session included the values were 0.79 / 0.65. |
| (c) Jitter | 156 settled dots (first 0.7 s dropped, valid frames; 6,131 samples; one test07 dot with residual RMS above 20 deg, a tracking failure, excluded). Residual SD around each dot's median: x 2.09, y 1.53 deg. Lag 1-5 autocorrelation x .683 .627 .531 .455 .361; y .580 .540 .415 .350 .248. RMS sample-to-sample: x 1.587, y 1.359 deg (Kaduk et al. 2024: 1.66 / 1.42). Excess kurtosis 16-22 (heavy tails). Per session (x SD, lag-1, RMS-S2S): test05 1.22/.27/1.44, test06 1.41/.12/1.84, test07 1.79/.48/1.76, test08 3.09/.87/1.43, test09 1.04/.54/.98. | `jitter_residuals.csv`, `jitter_pairs.csv.gz`, `measured_params.json` |
| (c) Jitter model | per axis, a sum of two independent exponential (Ornstein-Uhlenbeck) processes sampled at the frame times (exact AR(1) recursion with phi = exp(-gap/tau), valid for irregular times). x: total SD 3.57 deg, fast share 0.074 (tau 3.9 ms), slow tau 1242 ms. y: total SD 1.92 deg, fast share 0.192 (tau 3.0 ms), slow tau 460 ms. Fitted by simulation to the estimator on the same dots and frame pattern. Simulated (1,000 fresh replicates) against measured: x SD 2.101 vs 2.095, lags .692 .607 .525 .447 .372 vs .683 .627 .531 .455 .361, RMS-S2S 1.584 vs 1.587; y SD 1.539 vs 1.526, lags .596 .504 .418 .340 .267 vs .580 .540 .415 .350 .248, RMS-S2S 1.349 vs 1.359. | `jitter_model.json`. A single exponential cannot match the lags and the RMS together (best tau 270 / 158 ms gives RMS-S2S 1.01 / 0.94). Axes independent (measured lag-0 correlation 0.07); Gaussian marginals. |
| (d) Sample loss | 1 - `valid_share` of 85 scored trials: mean 5.2% (validation-dot loss 3.7% on average, 4 sessions with a validation) | `trials.csv`; drawn per trial from this empirical set, independent per frame |
| (e) Frame timing | 50,620 inter-frame gaps: mean 33.39 ms (29.95 fps), median 32.0, 5th-95th percentile 30.0-48.0 ms, maximum 81.9 ms; 8.0% of gaps exceed 1.4 x the median (dropped or delayed frames; 18-58 backlog drops per session in `meta.json`) | `frames.csv` `t_ms`; gap blocks taken contiguously, scaled so the mean gap is 1000/rate ms (20, 15, 10 Hz keep the same relative pattern: an assumption), random phase. The 17 fps condition was dropped (user, 2026-11-06). |

Simulated observed-webcam frames (checks stored in the feature files): share of grid points with a valid sample 0.89 at 30 Hz and 0.885 at 10 Hz (5.2% sample loss plus missing-sample and delayed-frame effects).

## How the condition is built (`s2_sim.py`)

Per participant and trial, from the 1000 Hz recordings: gaps filled and low-pass filtered with the anti-alias design of `degrade`; frame times from the measured gap blocks; the filtered signal read at those times (timing applied to the 1000 Hz signal, not decimation of a resampled signal); gain about the screen centre (960, 540); per-trial offset; correlated jitter; loss. With the app's correction: offset = cross minus the median of the valid frames of the pre-trial window, measured on the already noisy frames, refused (offset 0, trial kept) if fewer than 8/15 of the expected frames or above 6 deg (`app/protocol.py drift_from_frames`, `drift_max_deg`); added to every frame of the trial. Frames go onto the uniform grid with the app's own `to_grid`; features and nested-CV models are those of `results/phase9/low_rate/s4_webcam.py` (dwell from samples, 10 features; I-VT 30 deg/s, 60 ms, 11 features; no pupil; frozen folds; three noise draws; AUC = mean over draws). Conditions: observed webcam without correction, with correction; 30, 20, 15, 10 Hz.

## Results (all exploratory; simulated frame rates)

AUC is image-level (pooled within each of six group sets, mean over sets, repeat-averaged). AUC interval: first 300 of 2000 paired participant-cluster resamples (as in the earlier tables); every difference: all 2000 resamples, paired (one bootstrap for all configurations, seed 20261102). Non-inferiority: one-sided against 0.02 AUC, lower bound = 5th percentile of the paired change from 30 Hz; p = (1 + #{change <= -0.02}) / 2001 (minimum 0.0005); no multiplicity correction. "Within 0.02 (point / bound)": the point change is above -0.02 / the lower one-sided bound is above -0.02. The clean and simulated-noise rows come from Task 2.2 and are repeated (recomputed here, identical) to complete the 2 x 2 design.


**Dwell from samples (camera model)**

| Condition | Rate (Hz) | AUC [95% CI] | Change from 30 Hz [95% CI] | Lower one-sided 95% bound | p (inferior by 0.02) | Within 0.02 (point / bound) |
|---|---|---|---|---|---|---|
| clean | 30 | 0.7732 [0.7208, 0.8160] | (reference) | | | |
| clean | 20 | 0.7743 [0.7215, 0.8170] | +0.0012 [-0.0021, +0.0045] | -0.0016 | 0.0005 | yes / yes |
| clean | 15 | 0.7747 [0.7230, 0.8187] | +0.0015 [-0.0034, +0.0067] | -0.0026 | 0.0005 | yes / yes |
| clean | 10 | 0.7682 [0.7178, 0.8111] | -0.0050 [-0.0103, +0.0007] | -0.0093 | 0.0005 | yes / yes |
| simulated measured noise | 30 | 0.7737 [0.7229, 0.8155] | (reference) | | | |
| simulated measured noise | 20 | 0.7743 [0.7249, 0.8184] | +0.0006 [-0.0040, +0.0050] | -0.0032 | 0.0005 | yes / yes |
| simulated measured noise | 15 | 0.7764 [0.7259, 0.8212] | +0.0027 [-0.0012, +0.0068] | -0.0006 | 0.0005 | yes / yes |
| simulated measured noise | 10 | 0.7689 [0.7217, 0.8086] | -0.0048 [-0.0109, +0.0015] | -0.0098 | 0.0005 | yes / yes |
| observed webcam, no correction | 30 | 0.7278 [0.6934, 0.7601] | (reference) | | | |
| observed webcam, no correction | 20 | 0.7250 [0.6889, 0.7561] | -0.0028 [-0.0131, +0.0075] | -0.0116 | 0.0005 | yes / yes |
| observed webcam, no correction | 15 | 0.7240 [0.6895, 0.7579] | -0.0037 [-0.0139, +0.0062] | -0.0124 | 0.0010 | yes / yes |
| observed webcam, no correction | 10 | 0.7227 [0.6857, 0.7553] | -0.0051 [-0.0148, +0.0051] | -0.0135 | 0.0025 | yes / yes |
| observed webcam, with correction | 30 | 0.7282 [0.6919, 0.7622] | (reference) | | | |
| observed webcam, with correction | 20 | 0.7252 [0.6896, 0.7563] | -0.0030 [-0.0131, +0.0066] | -0.0117 | 0.0005 | yes / yes |
| observed webcam, with correction | 15 | 0.7243 [0.6927, 0.7542] | -0.0040 [-0.0146, +0.0071] | -0.0126 | 0.0020 | yes / yes |
| observed webcam, with correction | 10 | 0.7184 [0.6809, 0.7505] | -0.0098 [-0.0215, +0.0017] | -0.0193 | 0.0410 | yes / yes |

**I-VT fixation detection**

| Condition | Rate (Hz) | AUC [95% CI] | Change from 30 Hz [95% CI] | Lower one-sided 95% bound | p (inferior by 0.02) | Within 0.02 (point / bound) |
|---|---|---|---|---|---|---|
| clean | 30 | 0.7863 [0.7409, 0.8241] | (reference) | | | |
| clean | 20 | 0.7786 [0.7292, 0.8208] | -0.0077 [-0.0155, +0.0005] | -0.0142 | 0.0005 | yes / yes |
| clean | 15 | 0.7718 [0.7221, 0.8170] | -0.0145 [-0.0251, -0.0018] | -0.0234 | 0.1334 | yes / no |
| clean | 10 | 0.7520 [0.7049, 0.7929] | -0.0342 [-0.0511, -0.0181] | -0.0474 | 0.9495 | no / no |
| simulated measured noise | 30 | 0.7344 [0.6976, 0.7658] | (reference) | | | |
| simulated measured noise | 20 | 0.7502 [0.7038, 0.7863] | +0.0158 [+0.0024, +0.0282] | +0.0049 | 0.0005 | yes / yes |
| simulated measured noise | 15 | 0.7673 [0.7201, 0.8072] | +0.0330 [+0.0172, +0.0473] | +0.0195 | 0.0005 | yes / yes |
| simulated measured noise | 10 | 0.7510 [0.7068, 0.7880] | +0.0166 [+0.0020, +0.0311] | +0.0040 | 0.0005 | yes / yes |
| observed webcam, no correction | 30 | 0.6707 [0.6418, 0.6968] | (reference) | | | |
| observed webcam, no correction | 20 | 0.6771 [0.6473, 0.7017] | +0.0064 [-0.0082, +0.0207] | -0.0055 | 0.0005 | yes / yes |
| observed webcam, no correction | 15 | 0.7028 [0.6731, 0.7352] | +0.0321 [+0.0173, +0.0470] | +0.0197 | 0.0005 | yes / yes |
| observed webcam, no correction | 10 | 0.6804 [0.6504, 0.7105] | +0.0097 [-0.0044, +0.0234] | -0.0025 | 0.0005 | yes / yes |
| observed webcam, with correction | 30 | 0.6695 [0.6417, 0.6960] | (reference) | | | |
| observed webcam, with correction | 20 | 0.6735 [0.6410, 0.7016] | +0.0039 [-0.0089, +0.0170] | -0.0074 | 0.0005 | yes / yes |
| observed webcam, with correction | 15 | 0.6975 [0.6698, 0.7256] | +0.0279 [+0.0133, +0.0435] | +0.0153 | 0.0005 | yes / yes |
| observed webcam, with correction | 10 | 0.6753 [0.6485, 0.7028] | +0.0058 [-0.0075, +0.0193] | -0.0054 | 0.0005 | yes / yes |

**Dwell from samples minus I-VT at the same rate (paired)**

| Condition | 30 Hz | 20 Hz | 15 Hz | 10 Hz |
|---|---|---|---|---|
| clean | -0.0131 [-0.0226, -0.0022] | -0.0043 [-0.0133, +0.0060] | +0.0029 [-0.0041, +0.0097] | +0.0162 [+0.0030, +0.0318] |
| simulated measured noise | +0.0394 [+0.0216, +0.0562] | +0.0241 [+0.0121, +0.0365] | +0.0091 [+0.0022, +0.0169] | +0.0179 [+0.0059, +0.0308] |
| observed webcam, no correction | +0.0571 [+0.0431, +0.0698] | +0.0479 [+0.0352, +0.0590] | +0.0213 [+0.0129, +0.0291] | +0.0422 [+0.0283, +0.0560] |
| observed webcam, with correction | +0.0587 [+0.0437, +0.0743] | +0.0518 [+0.0435, +0.0599] | +0.0268 [+0.0173, +0.0355] | +0.0431 [+0.0314, +0.0549] |

**EER and HTER (nested-CV track values, mean over the six group sets and over draws; image level | participant level)**

| Measure | Condition | Rate (Hz) | EER image | HTER image | EER participant | HTER participant |
|---|---|---|---|---|---|---|
| Dwell from samples (camera model) | clean | 30 | 0.273 | 0.272 | 0.154 | 0.285 |
| Dwell from samples (camera model) | clean | 20 | 0.272 | 0.271 | 0.165 | 0.294 |
| Dwell from samples (camera model) | clean | 15 | 0.271 | 0.271 | 0.156 | 0.291 |
| Dwell from samples (camera model) | clean | 10 | 0.278 | 0.274 | 0.156 | 0.304 |
| Dwell from samples (camera model) | simulated measured noise | 30 | 0.272 | 0.269 | 0.157 | 0.288 |
| Dwell from samples (camera model) | simulated measured noise | 20 | 0.273 | 0.271 | 0.157 | 0.290 |
| Dwell from samples (camera model) | simulated measured noise | 15 | 0.269 | 0.267 | 0.177 | 0.291 |
| Dwell from samples (camera model) | simulated measured noise | 10 | 0.278 | 0.276 | 0.175 | 0.305 |
| Dwell from samples (camera model) | observed webcam, no correction | 30 | 0.317 | 0.317 | 0.199 | 0.391 |
| Dwell from samples (camera model) | observed webcam, no correction | 20 | 0.323 | 0.327 | 0.212 | 0.391 |
| Dwell from samples (camera model) | observed webcam, no correction | 15 | 0.321 | 0.323 | 0.205 | 0.397 |
| Dwell from samples (camera model) | observed webcam, no correction | 10 | 0.325 | 0.328 | 0.218 | 0.404 |
| Dwell from samples (camera model) | observed webcam, with correction | 30 | 0.321 | 0.320 | 0.197 | 0.388 |
| Dwell from samples (camera model) | observed webcam, with correction | 20 | 0.324 | 0.324 | 0.195 | 0.399 |
| Dwell from samples (camera model) | observed webcam, with correction | 15 | 0.321 | 0.322 | 0.198 | 0.403 |
| Dwell from samples (camera model) | observed webcam, with correction | 10 | 0.331 | 0.331 | 0.231 | 0.406 |
| I-VT fixation detection | clean | 30 | 0.262 | 0.261 | 0.148 | 0.289 |
| I-VT fixation detection | clean | 20 | 0.274 | 0.272 | 0.135 | 0.302 |
| I-VT fixation detection | clean | 15 | 0.274 | 0.271 | 0.161 | 0.306 |
| I-VT fixation detection | clean | 10 | 0.294 | 0.292 | 0.165 | 0.313 |
| I-VT fixation detection | simulated measured noise | 30 | 0.312 | 0.310 | 0.193 | 0.352 |
| I-VT fixation detection | simulated measured noise | 20 | 0.301 | 0.298 | 0.196 | 0.329 |
| I-VT fixation detection | simulated measured noise | 15 | 0.284 | 0.282 | 0.154 | 0.310 |
| I-VT fixation detection | simulated measured noise | 10 | 0.296 | 0.293 | 0.161 | 0.329 |
| I-VT fixation detection | observed webcam, no correction | 30 | 0.373 | 0.371 | 0.264 | 0.473 |
| I-VT fixation detection | observed webcam, no correction | 20 | 0.363 | 0.363 | 0.244 | 0.463 |
| I-VT fixation detection | observed webcam, no correction | 15 | 0.339 | 0.339 | 0.237 | 0.445 |
| I-VT fixation detection | observed webcam, no correction | 10 | 0.357 | 0.359 | 0.267 | 0.463 |
| I-VT fixation detection | observed webcam, with correction | 30 | 0.372 | 0.373 | 0.291 | 0.475 |
| I-VT fixation detection | observed webcam, with correction | 20 | 0.363 | 0.363 | 0.260 | 0.469 |
| I-VT fixation detection | observed webcam, with correction | 15 | 0.346 | 0.347 | 0.226 | 0.452 |
| I-VT fixation detection | observed webcam, with correction | 10 | 0.366 | 0.368 | 0.269 | 0.469 |

**Other paired contrasts (AUC difference [95% CI])**

| Measure | Contrast | 30 Hz | 20 Hz | 15 Hz | 10 Hz |
|---|---|---|---|---|---|
| Dwell from samples (camera model) | correction effect (with - without) | +0.0005 [-0.0072, +0.0084] | +0.0003 [-0.0090, +0.0089] | +0.0002 [-0.0089, +0.0093] | -0.0043 [-0.0125, +0.0040] |
| Dwell from samples (camera model) | observed (no correction) - simulated measured noise | -0.0459 [-0.0628, -0.0287] | -0.0493 [-0.0679, -0.0313] | -0.0524 [-0.0690, -0.0363] | -0.0463 [-0.0613, -0.0314] |
| Dwell from samples (camera model) | observed (with correction) - simulated measured noise | -0.0455 [-0.0610, -0.0293] | -0.0490 [-0.0657, -0.0324] | -0.0521 [-0.0702, -0.0342] | -0.0505 [-0.0673, -0.0349] |
| Dwell from samples (camera model) | simulated measured noise - clean | +0.0005 [-0.0059, +0.0067] | -0.0001 [-0.0058, +0.0062] | +0.0017 [-0.0028, +0.0064] | +0.0007 [-0.0062, +0.0074] |
| Dwell from samples (camera model) | observed (with correction) - clean | -0.0450 [-0.0625, -0.0270] | -0.0491 [-0.0679, -0.0298] | -0.0504 [-0.0693, -0.0318] | -0.0498 [-0.0687, -0.0317] |
| Dwell from samples (camera model) | observed (no correction) - clean | -0.0454 [-0.0636, -0.0269] | -0.0494 [-0.0692, -0.0300] | -0.0506 [-0.0682, -0.0335] | -0.0455 [-0.0625, -0.0281] |
| Dwell from samples (camera model) | regular timing through this pipeline - clean (pipeline check) | +0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| I-VT fixation detection | correction effect (with - without) | -0.0012 [-0.0103, +0.0081] | -0.0037 [-0.0142, +0.0060] | -0.0053 [-0.0143, +0.0039] | -0.0051 [-0.0156, +0.0055] |
| I-VT fixation detection | observed (no correction) - simulated measured noise | -0.0637 [-0.0811, -0.0466] | -0.0730 [-0.0919, -0.0533] | -0.0645 [-0.0827, -0.0464] | -0.0706 [-0.0889, -0.0513] |
| I-VT fixation detection | observed (with correction) - simulated measured noise | -0.0648 [-0.0830, -0.0472] | -0.0767 [-0.0945, -0.0579] | -0.0699 [-0.0896, -0.0491] | -0.0757 [-0.0952, -0.0557] |
| I-VT fixation detection | simulated measured noise - clean | -0.0519 [-0.0682, -0.0345] | -0.0284 [-0.0384, -0.0187] | -0.0045 [-0.0128, +0.0031] | -0.0010 [-0.0092, +0.0064] |
| I-VT fixation detection | observed (with correction) - clean | -0.1167 [-0.1406, -0.0917] | -0.1051 [-0.1267, -0.0826] | -0.0744 [-0.0971, -0.0511] | -0.0767 [-0.0998, -0.0545] |
| I-VT fixation detection | observed (no correction) - clean | -0.1156 [-0.1386, -0.0912] | -0.1015 [-0.1234, -0.0790] | -0.0690 [-0.0911, -0.0475] | -0.0716 [-0.0936, -0.0500] |
| I-VT fixation detection | regular timing through this pipeline - clean (pipeline check) | -0.0010 [-0.0026, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0001] | +0.0000 [+0.0000, +0.0000] |

`rate_table_observed.csv` has every number above plus the regular-timing check rows (`condition_key` = regular); `paired_contrasts_observed.csv` the contrasts; `fig_rate_observed.{pdf,png,csv}` the figure (a: AUC by rate, dwell from samples in blue and I-VT in orange, each on clean data, under simulated noise and under the observed webcam without and with correction; b: change in AUC from 30 Hz for dwell from samples with 95% intervals and the 0.02 margin; the clean and simulated-noise dwell lines nearly coincide in panel a).

**Figure description (for the paper).** Panel a plots, for frame rates 30, 20, 15 and 10 Hz, the accuracy (AUC) of telling a known item from unknown items. Blue lines use looking time computed directly from gaze samples, orange lines use fixation detection. Each is drawn for clean tracker data, for the earlier simulated webcam noise, and for the observed-webcam condition built from our own recordings without and with the per-trial offset correction. Looking time from samples stays near 0.77 on clean data and under simulated noise and near 0.72-0.73 under the observed-webcam condition across all four rates, whereas fixation detection falls from 0.79 to 0.75 on clean data, rises with falling rate under both noise conditions, and is about 0.67-0.70 under the observed-webcam condition. Panel b shows the change in AUC from 30 Hz for looking time from samples with 95% intervals; all intervals have their lower end above the dashed -0.02 margin except the observed-webcam condition with correction at 10 Hz, whose lower end is -0.0215 (the one-sided 95% bound, -0.0193, is above the margin).

## Plain summary (5 lines)

1. Dwell from samples under the observed-webcam condition scores AUC 0.728 at 30 Hz, 0.725 at 20, 0.724 at 15 and 0.723 at 10 Hz without correction (0.728, 0.725, 0.724, 0.718 with correction): about 0.045 below clean data (-0.045 [-0.064, -0.027]), where the earlier simulated noise cost nothing (+0.0005), so the earlier simulation understated the error of this measure.
2. Its change from 30 to 10 Hz is -0.005 [-0.015, +0.005] without and -0.010 [-0.022, +0.002] with correction; the lower one-sided 95% bounds (-0.0135, -0.0193) are above the -0.02 margin (p 0.0025 and 0.041), so it is within the margin (exploratory), narrowly so for the corrected condition at 10 Hz; the same holds for clean data and simulated noise (p 0.0005).
3. I-VT under the observed webcam scores 0.671 at 30 Hz, 0.677 at 20, 0.703 at 15, 0.680 at 10 Hz (no correction), as under simulated noise it does not fall with rate; dwell minus I-VT is +0.057 [+0.043, +0.070] at 30 Hz and +0.042 [+0.028, +0.056] at 10 Hz (all four intervals above zero; with correction +0.059 to +0.027), while on clean data the sign changes with rate (-0.013 at 30 Hz, +0.016 [+0.003, +0.032] at 10 Hz); under simulated noise at 10 Hz it is +0.0179 [+0.0059, +0.0308] (as in Task 2.2).
4. The per-trial correction made no measurable difference to either measure (dwell -0.004 to +0.0005, I-VT -0.005 to -0.001, all intervals include 0); in the simulation only about 80% (30 Hz) and 76% (10 Hz) of trials passed the app's drift check against 93% in the recordings.
5. Caution: the simulated slow wander of the jitter (x SD 3.4 deg, tau 1.2 s) may overlap with the measured per-trial offsets (which already contain slow wander), so the observed condition may be harsher than the recordings; the parameters come from one person with one EMEET camera; everything is exploratory.

## Deviations, assumptions and caveats

- `PLAN.md` was written after the user's changes (EMEET sessions test05-test09, gains 0.88 / 0.79, no 17 fps condition), so it has no separate addendum; the changes are in its first section. An earlier measurement pass made with the first parameter set (four sessions without test07, gains 0.79 / 0.65) was overwritten before any simulation or result was computed.
- The session was interrupted and `s4_run.py` restarted from its checkpoints (all draws are seeded by participant, rate and draw, so results do not depend on the restart). All 56 planned checkpoints exist (4 rates x 2 measures x (2 corrections x 3 draws + 1 regular check)), plus the 4 reproduction runs.
- Possible double counting in the jitter model (the main caveat). The jitter is fitted on residuals around each dot's median over about 1.3 s; the slow component (x tau 1.24 s) then has a variance that is extrapolated beyond that window (SD x 3.4, y 1.7 deg) and is added on top of the per-trial offsets, whose measured spread (SD 1.9 / 2.0 deg per axis) was taken from a median over a short fixation and already contains some slow wander. The simulated observed condition therefore may be harsher than the recordings; this is not tested. Supporting signs: the mean applied correction was 3.3 deg against 2.64 deg logged, and the drift check failed in about 20-25% of simulated trials against 7% (6 of 85) of recorded trials. A sensitivity run with a reduced slow component is not done; it can be added on request.
- Frame timing at 20, 15 and 10 Hz: the 30 fps gap pattern (including delayed frames) is scaled to the lower mean gap; a real camera at those rates could behave differently. Rates are simulated from 1000 Hz recordings of other people (Delmas viewers), not recorded on the EMEET.
- Pre-trial correction window: [-500, -100) ms before display onset of the Delmas arrays (400 ms; the app uses 0.4-0.9 s of a 1 s fixation), minimum frames ceil(8/15 x expected). The Delmas viewers fixate a central cross before the display, but their own gaze is not exactly on the cross.
- Loss is independent per frame (real loss is bursty); empirical offsets exclude the 3 trials above 6 deg and 3 with too few frames, so the uncorrected condition is slightly optimistic.
- Jitter excess kurtosis (16-22) is not reproduced (Gaussian marginals); axes simulated independently; test07 (an aborted two-face dry run) is included in all measurements as instructed, with one gross-failure dot excluded (rule: residual RMS above 20 deg on either axis, fixed before the fit).
- At 15 Hz the module constant POST_MS of `app/features.py` is raised by 34 ms for the call so the grid has 83 points as in `degrade` (module attribute only; file unchanged).
- EER and HTER are the nested-CV track's own image-level and participant-level values (mean over the six group sets, mean over three draws); no intervals were computed for them.
- Non-inferiority p values have a floor of 0.0005 (2000 resamples) and no multiplicity correction; the margin of 0.02 is the project's earlier one, chosen after screening (M1/M4).
- Interval conventions follow Task 2.2 (AUC interval from 300 resamples), so AUC intervals are slightly less stable than the difference intervals.
- The folder cleanup of 2026-11-07 removed `paper/`; nothing here reads it. `results/phase9/figures/fig_drift.csv` was read only for the consistency note above.

## Files

`PLAN.md`, `PLAN.sha256.txt`, `rate_table_observed.csv`, `paired_contrasts_observed.csv`, `card_tables.md` (the tables above), `fig_rate_observed.pdf/.png/.csv`, `measured/` (drift, valid share, jitter residuals and pairs, gap arrays, `measured_params.json`, `jitter_model.json`), `repro_values.json`, `repro_raw_reload.json`, `repro_stats_vs_stored.json`, `checkpoints/` (nested-CV outputs per tag), `features/` (per participant and rate), `cache1000/` (1000 Hz arrays of the Delmas recordings), logs `log_*.txt`.
