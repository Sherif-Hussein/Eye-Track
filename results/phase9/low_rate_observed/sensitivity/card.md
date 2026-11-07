# Task 2.2c Part A: sensitivity of the observed-webcam condition to the slow jitter component (card)

**Status: EXPLORATORY, development data only (Delmas et al.), simulated frame rates. Not preregistered.** Rules fixed 2026-11-07 before any of these runs (`docs/task_observed_sensitivity_and_models.md`). Plan: `PLAN.md` (SHA-256 `d00085cb7aa711ca7c0cb76173eeb1c2a313f691705da6dfe3933dcffd856315`, written 2026-11-07 06:12:49 before the run; `PLAN.sha256.txt`). Output folder `results/phase9/low_rate_observed/sensitivity/`. Scripts: `sa_params.py`, `sa_run.py`, `sa_checks.py`, `sa_stats.py`. The 2.2b code (`lro_common.py`, `s2_sim.py`, `s4_run.py`) was imported unchanged; the changed jitter parameters and the output folders were passed from outside; nothing was written in the 2.2b folders. No camera, numbers only; the held-out data were not read. Wording: known item / unknown item.

## Why
Task 2.2b gave dwell from samples AUC 0.728 at 30 Hz under the observed webcam, 0.045 below clean data, but its slow jitter may double count the per-trial offset (2.2b card, "Possible double counting"; simulated drift check failed in 20-25% of trials against 7% recorded; mean applied correction 3.3 deg against 2.64 deg logged). This run changes one thing and nothing else.

## The one change (`jitter_params_reduced.json`)
Per axis, tau_fast, tau_slow and the fast component's absolute variance are those of `measured/jitter_model.json`; the slow component's variance is rescaled so that the process's total long-run SD equals the measured residual SD. The fast variance is below the target on both axes (no stop).

| Axis | tau_fast (ms) | tau_slow (ms) | fast SD (deg) | slow SD, 2.2b -> now (deg) | total SD, 2.2b -> now (deg) |
|---|---|---|---|---|---|
| x | 3.9 | 1242 | 0.971 | 3.433 -> 1.857 | 3.567 -> 2.095 |
| y | 3.0 | 460 | 0.838 | 1.721 -> 1.275 | 1.915 -> 1.526 |

Gains (0.88, 0.79), per-trial offsets, loss, frame times, seeds and folds are exactly those of 2.2b. The jitter noise is the same set of standard normals scaled differently, so every draw has the same frame times, offsets and losses as in 2.2b and the comparison is paired. Runs: both routes (dwell from samples 10 features; I-VT 11 features), correction none and app, 30/20/15/10 Hz, draws 1-3, frozen folds, nested L2 logistic regression: 48 nested-CV runs, all complete. The 2.2b rows were recomputed in the same bootstrap call and reproduce `rate_table_observed.csv` (maximum absolute difference 1e-16; `sens_repro_check.json`).

## Realism checks (`sens_realism.json`)

**Drift-check failure rate and mean applied correction** (75 participants x 3 draws x 64 trials per rate; correction variant 'app'). Recorded: 6 of 85 scored trials failed = 7.1% (exact 95% interval 2.6% to 14.7%; one person, one camera); mean magnitude of the 79 passed logged offsets 2.64 deg.

| Rate | Failure rate, 2.2b | Failure rate, reduced | Recorded | Mean applied correction (deg), 2.2b | Mean applied correction (deg), reduced | Recorded |
|---|---|---|---|---|---|---|
| 30 Hz | 20.5% | 8.0% | 7.1% | 3.32 | 3.03 | 2.64 |
| 20 Hz | 21.7% | 9.0% | | 3.31 | 3.03 | |
| 15 Hz | 22.3% | 9.5% | | 3.34 | 3.06 | |
| 10 Hz | 24.8% | 12.7% | 7.1% | 3.35 | 3.06 | 2.64 |

How close: at 30 Hz the reduced simulation's failure rate (8.0%) is inside the interval of the recordings and about one percentage point from them; at 10 Hz (12.7%) it is inside the interval but nearly twice the recorded point value. The 2.2b simulation (20.5% and 24.8%) is far outside the interval at both rates. The mean applied correction is still above the recorded value in both simulations: 3.03 deg (reduced) and 3.32 deg (2.2b) against 2.64 deg; so reducing the slow component closes about 40% of the gap in the applied correction (and nearly all of the gap in the failure rate), not all of it. Part of the remaining gap is expected from construction: the applied correction is measured on noisy frames (jitter of a median of about 12 frames, the gain compression of the Delmas viewers' own gaze, their own deviation from the cross) added to an offset drawn from the logged distribution. The recorded failure rate rests on 6 failed trials of 85.

**Jitter statistics against the measurements** (1000 fresh replicates on the same 156 dots and frame pattern, estimator of `s0b_fit_jitter.py`, seed 991; the re-typed code reproduces the stored 2.2b check exactly, difference 0.0). The reduced process is not re-fitted, so it fits the measured shape less well, as expected.

| Axis, source | SD (deg) | Lag 1 | Lag 2 | Lag 3 | Lag 4 | Lag 5 | RMS sample-to-sample (deg) |
|---|---|---|---|---|---|---|---|
| x, measured | 2.095 | .683 | .627 | .531 | .455 | .361 | 1.587 |
| x, 2.2b process | 2.101 | .692 | .607 | .525 | .447 | .372 | 1.584 |
| x, reduced process | 1.398 | .447 | .391 | .337 | .287 | .237 | 1.437 |
| y, measured | 1.526 | .580 | .540 | .415 | .350 | .248 | 1.359 |
| y, 2.2b process | 1.539 | .596 | .504 | .418 | .340 | .267 | 1.349 |
| y, reduced process | 1.272 | .474 | .401 | .331 | .269 | .211 | 1.278 |

The reduced process fits the measured jitter clearly worse: its settled-dot SD is 1.40 deg against 2.10 (x) and 1.27 against 1.53 (y), lag-1 autocorrelation 0.45 against 0.68 (x) and 0.47 against 0.58 (y), RMS sample-to-sample 1.44 against 1.59 (x) and 1.28 against 1.36 (y). The reason is that the measured SD is taken around each dot's median over about 1.3 s, which removes part of the slow wander, so a process whose total SD equals the measured value has too little slow wander to give that residual. On the jitter statistics alone the reduced run is therefore milder than the recordings; the 2.2b process matches them but is harsher on the drift check.

## Results (all exploratory; simulated frame rates)
Conventions as in 2.2b: AUC is image-level (pooled within each of six group sets, mean over sets), mean over 3 noise draws; AUC interval from the first 300 of 2000 paired participant-cluster resamples, every difference from all 2000 (one bootstrap call for all configurations, seed 20261102); non-inferiority one-sided against 0.02, lower bound = 5th percentile of the paired change from 30 Hz, p = (1 + #{change <= -0.02}) / 2001, no multiplicity correction. Clean and simulated-noise rows are in `sens_rate_table.csv`.

**Dwell from samples (camera model), no correction** (2.2b = full slow jitter; reduced = slow jitter variance scaled down)

| Rate (Hz) | 2.2b AUC [95% CI] | 2.2b change from 30 Hz [95% CI] | Reduced AUC [95% CI] | Reduced change from 30 Hz [95% CI] | Reduced lower one-sided 95% bound | p (inferior by 0.02) | Within 0.02 (point / bound) |
|---|---|---|---|---|---|---|---|
| 30 | 0.7278 [0.6934, 0.7601] | (reference) | 0.7481 [0.7091, 0.7813] | (reference) | | | |
| 20 | 0.7250 [0.6889, 0.7561] | -0.0028 [-0.0131, +0.0075] | 0.7444 [0.7047, 0.7798] | -0.0038 [-0.0118, +0.0042] | -0.0104 | 0.0005 | yes / yes |
| 15 | 0.7240 [0.6895, 0.7579] | -0.0037 [-0.0139, +0.0062] | 0.7429 [0.7032, 0.7777] | -0.0053 [-0.0146, +0.0037] | -0.0131 | 0.0025 | yes / yes |
| 10 | 0.7227 [0.6857, 0.7553] | -0.0051 [-0.0148, +0.0051] | 0.7373 [0.6971, 0.7700] | -0.0109 [-0.0192, -0.0020] | -0.0179 | 0.0160 | yes / yes |

**Dwell from samples (camera model), with the app's correction** (2.2b = full slow jitter; reduced = slow jitter variance scaled down)

| Rate (Hz) | 2.2b AUC [95% CI] | 2.2b change from 30 Hz [95% CI] | Reduced AUC [95% CI] | Reduced change from 30 Hz [95% CI] | Reduced lower one-sided 95% bound | p (inferior by 0.02) | Within 0.02 (point / bound) |
|---|---|---|---|---|---|---|---|
| 30 | 0.7282 [0.6919, 0.7622] | (reference) | 0.7581 [0.7146, 0.7959] | (reference) | | | |
| 20 | 0.7252 [0.6896, 0.7563] | -0.0030 [-0.0131, +0.0066] | 0.7518 [0.7083, 0.7889] | -0.0063 [-0.0141, +0.0005] | -0.0126 | 0.0005 | yes / yes |
| 15 | 0.7243 [0.6927, 0.7542] | -0.0040 [-0.0146, +0.0071] | 0.7538 [0.7102, 0.7888] | -0.0042 [-0.0140, +0.0043] | -0.0125 | 0.0015 | yes / yes |
| 10 | 0.7184 [0.6809, 0.7505] | -0.0098 [-0.0215, +0.0017] | 0.7518 [0.7108, 0.7875] | -0.0062 [-0.0159, +0.0028] | -0.0145 | 0.0050 | yes / yes |

**I-VT fixation detection, no correction** (2.2b = full slow jitter; reduced = slow jitter variance scaled down)

| Rate (Hz) | 2.2b AUC [95% CI] | 2.2b change from 30 Hz [95% CI] | Reduced AUC [95% CI] | Reduced change from 30 Hz [95% CI] | Reduced lower one-sided 95% bound | p (inferior by 0.02) | Within 0.02 (point / bound) |
|---|---|---|---|---|---|---|---|
| 30 | 0.6707 [0.6418, 0.6968] | (reference) | 0.6945 [0.6608, 0.7247] | (reference) | | | |
| 20 | 0.6771 [0.6473, 0.7017] | +0.0064 [-0.0082, +0.0207] | 0.7039 [0.6697, 0.7340] | +0.0093 [-0.0003, +0.0187] | +0.0013 | 0.0005 | yes / yes |
| 15 | 0.7028 [0.6731, 0.7352] | +0.0321 [+0.0173, +0.0470] | 0.7142 [0.6785, 0.7497] | +0.0196 [+0.0071, +0.0320] | +0.0094 | 0.0005 | yes / yes |
| 10 | 0.6804 [0.6504, 0.7105] | +0.0097 [-0.0044, +0.0234] | 0.7003 [0.6652, 0.7316] | +0.0058 [-0.0064, +0.0177] | -0.0043 | 0.0005 | yes / yes |

**I-VT fixation detection, with the app's correction** (2.2b = full slow jitter; reduced = slow jitter variance scaled down)

| Rate (Hz) | 2.2b AUC [95% CI] | 2.2b change from 30 Hz [95% CI] | Reduced AUC [95% CI] | Reduced change from 30 Hz [95% CI] | Reduced lower one-sided 95% bound | p (inferior by 0.02) | Within 0.02 (point / bound) |
|---|---|---|---|---|---|---|---|
| 30 | 0.6695 [0.6417, 0.6960] | (reference) | 0.7041 [0.6716, 0.7361] | (reference) | | | |
| 20 | 0.6735 [0.6410, 0.7016] | +0.0039 [-0.0089, +0.0170] | 0.7078 [0.6691, 0.7381] | +0.0037 [-0.0056, +0.0125] | -0.0041 | 0.0005 | yes / yes |
| 15 | 0.6975 [0.6698, 0.7256] | +0.0279 [+0.0133, +0.0435] | 0.7281 [0.6890, 0.7623] | +0.0240 [+0.0131, +0.0353] | +0.0148 | 0.0005 | yes / yes |
| 10 | 0.6753 [0.6485, 0.7028] | +0.0058 [-0.0075, +0.0193] | 0.7154 [0.6822, 0.7466] | +0.0114 [-0.0002, +0.0224] | +0.0020 | 0.0005 | yes / yes |

**Dwell from samples minus I-VT at the same rate (paired)**

| Condition | 30 Hz | 20 Hz | 15 Hz | 10 Hz |
|---|---|---|---|---|
| observed webcam (2.2b), no correction | +0.0571 [+0.0431, +0.0698] | +0.0479 [+0.0352, +0.0590] | +0.0213 [+0.0129, +0.0291] | +0.0422 [+0.0283, +0.0560] |
| observed webcam, reduced slow jitter, no correction | +0.0536 [+0.0393, +0.0667] | +0.0405 [+0.0279, +0.0519] | +0.0287 [+0.0204, +0.0365] | +0.0370 [+0.0259, +0.0478] |
| observed webcam (2.2b), with correction | +0.0587 [+0.0437, +0.0743] | +0.0518 [+0.0435, +0.0599] | +0.0268 [+0.0173, +0.0355] | +0.0431 [+0.0314, +0.0549] |
| observed webcam, reduced slow jitter, with correction | +0.0540 [+0.0382, +0.0700] | +0.0440 [+0.0308, +0.0567] | +0.0257 [+0.0172, +0.0335] | +0.0364 [+0.0240, +0.0498] |

**Paired AUC differences [95% CI]**

| Measure | Contrast | 30 Hz | 20 Hz | 15 Hz | 10 Hz |
|---|---|---|---|---|---|
| Dwell from samples (camera model) | reduced, no correction - clean | -0.0250 [-0.0394, -0.0109] | -0.0300 [-0.0428, -0.0172] | -0.0318 [-0.0463, -0.0182] | -0.0309 [-0.0462, -0.0157] |
| Dwell from samples (camera model) | reduced, with correction - clean | -0.0151 [-0.0266, -0.0030] | -0.0226 [-0.0346, -0.0104] | -0.0209 [-0.0311, -0.0106] | -0.0164 [-0.0285, -0.0045] |
| Dwell from samples (camera model) | reduced - 2.2b observed, no correction | +0.0204 [+0.0127, +0.0281] | +0.0194 [+0.0092, +0.0290] | +0.0188 [+0.0114, +0.0262] | +0.0146 [+0.0064, +0.0233] |
| Dwell from samples (camera model) | reduced - 2.2b observed, with correction | +0.0298 [+0.0198, +0.0405] | +0.0265 [+0.0156, +0.0375] | +0.0296 [+0.0172, +0.0415] | +0.0334 [+0.0233, +0.0436] |
| Dwell from samples (camera model) | reduced, no correction - simulated measured noise | -0.0256 [-0.0376, -0.0133] | -0.0299 [-0.0414, -0.0198] | -0.0335 [-0.0471, -0.0211] | -0.0316 [-0.0445, -0.0195] |
| Dwell from samples (camera model) | correction effect, reduced (with - without) | +0.0099 [+0.0020, +0.0188] | +0.0074 [+0.0006, +0.0144] | +0.0110 [+0.0021, +0.0198] | +0.0145 [+0.0065, +0.0228] |
| Dwell from samples (camera model) | 2.2b observed, no correction - clean | -0.0454 [-0.0636, -0.0269] | -0.0494 [-0.0692, -0.0300] | -0.0506 [-0.0682, -0.0335] | -0.0455 [-0.0625, -0.0281] |
| Dwell from samples (camera model) | 2.2b observed, with correction - clean | -0.0450 [-0.0625, -0.0270] | -0.0491 [-0.0679, -0.0298] | -0.0504 [-0.0693, -0.0318] | -0.0498 [-0.0687, -0.0317] |
| I-VT fixation detection | reduced, no correction - clean | -0.0917 [-0.1118, -0.0696] | -0.0747 [-0.0939, -0.0561] | -0.0576 [-0.0751, -0.0404] | -0.0517 [-0.0703, -0.0345] |
| I-VT fixation detection | reduced, with correction - clean | -0.0822 [-0.1003, -0.0632] | -0.0708 [-0.0891, -0.0524] | -0.0437 [-0.0599, -0.0276] | -0.0366 [-0.0535, -0.0206] |
| I-VT fixation detection | reduced - 2.2b observed, no correction | +0.0238 [+0.0129, +0.0342] | +0.0268 [+0.0168, +0.0360] | +0.0114 [+0.0026, +0.0204] | +0.0199 [+0.0113, +0.0286] |
| I-VT fixation detection | reduced - 2.2b observed, with correction | +0.0345 [+0.0231, +0.0455] | +0.0343 [+0.0250, +0.0431] | +0.0307 [+0.0180, +0.0436] | +0.0401 [+0.0286, +0.0513] |
| I-VT fixation detection | reduced, no correction - simulated measured noise | -0.0398 [-0.0539, -0.0254] | -0.0463 [-0.0603, -0.0319] | -0.0531 [-0.0666, -0.0393] | -0.0507 [-0.0657, -0.0362] |
| I-VT fixation detection | correction effect, reduced (with - without) | +0.0095 [+0.0015, +0.0176] | +0.0039 [-0.0040, +0.0117] | +0.0139 [+0.0056, +0.0225] | +0.0151 [+0.0076, +0.0227] |
| I-VT fixation detection | 2.2b observed, no correction - clean | -0.1156 [-0.1386, -0.0912] | -0.1015 [-0.1234, -0.0790] | -0.0690 [-0.0911, -0.0475] | -0.0716 [-0.0936, -0.0500] |
| I-VT fixation detection | 2.2b observed, with correction - clean | -0.1167 [-0.1406, -0.0917] | -0.1051 [-0.1267, -0.0826] | -0.0744 [-0.0971, -0.0511] | -0.0767 [-0.0998, -0.0545] |

**EER and HTER (nested-CV track values, mean over the six group sets and over draws)**

| Measure | Condition | Rate (Hz) | EER image | HTER image | EER participant | HTER participant |
|---|---|---|---|---|---|---|
| Dwell from samples (camera model) | observed webcam (2.2b), no correction | 30 | 0.317 | 0.317 | 0.199 | 0.391 |
| Dwell from samples (camera model) | observed webcam (2.2b), no correction | 20 | 0.323 | 0.327 | 0.212 | 0.391 |
| Dwell from samples (camera model) | observed webcam (2.2b), no correction | 15 | 0.321 | 0.323 | 0.205 | 0.397 |
| Dwell from samples (camera model) | observed webcam (2.2b), no correction | 10 | 0.325 | 0.328 | 0.218 | 0.404 |
| Dwell from samples (camera model) | observed webcam, reduced slow jitter, no correction | 30 | 0.300 | 0.300 | 0.187 | 0.358 |
| Dwell from samples (camera model) | observed webcam, reduced slow jitter, no correction | 20 | 0.305 | 0.304 | 0.180 | 0.365 |
| Dwell from samples (camera model) | observed webcam, reduced slow jitter, no correction | 15 | 0.304 | 0.304 | 0.170 | 0.368 |
| Dwell from samples (camera model) | observed webcam, reduced slow jitter, no correction | 10 | 0.307 | 0.307 | 0.201 | 0.369 |
| Dwell from samples (camera model) | observed webcam (2.2b), with correction | 30 | 0.321 | 0.320 | 0.197 | 0.388 |
| Dwell from samples (camera model) | observed webcam (2.2b), with correction | 20 | 0.324 | 0.324 | 0.195 | 0.399 |
| Dwell from samples (camera model) | observed webcam (2.2b), with correction | 15 | 0.321 | 0.322 | 0.198 | 0.403 |
| Dwell from samples (camera model) | observed webcam (2.2b), with correction | 10 | 0.331 | 0.331 | 0.231 | 0.406 |
| Dwell from samples (camera model) | observed webcam, reduced slow jitter, with correction | 30 | 0.293 | 0.293 | 0.180 | 0.331 |
| Dwell from samples (camera model) | observed webcam, reduced slow jitter, with correction | 20 | 0.296 | 0.296 | 0.193 | 0.344 |
| Dwell from samples (camera model) | observed webcam, reduced slow jitter, with correction | 15 | 0.293 | 0.291 | 0.186 | 0.343 |
| Dwell from samples (camera model) | observed webcam, reduced slow jitter, with correction | 10 | 0.293 | 0.293 | 0.173 | 0.350 |
| I-VT fixation detection | observed webcam (2.2b), no correction | 30 | 0.373 | 0.371 | 0.264 | 0.473 |
| I-VT fixation detection | observed webcam (2.2b), no correction | 20 | 0.363 | 0.363 | 0.244 | 0.463 |
| I-VT fixation detection | observed webcam (2.2b), no correction | 15 | 0.339 | 0.339 | 0.237 | 0.445 |
| I-VT fixation detection | observed webcam (2.2b), no correction | 10 | 0.357 | 0.359 | 0.267 | 0.463 |
| I-VT fixation detection | observed webcam, reduced slow jitter, no correction | 30 | 0.351 | 0.350 | 0.240 | 0.438 |
| I-VT fixation detection | observed webcam, reduced slow jitter, no correction | 20 | 0.338 | 0.338 | 0.238 | 0.425 |
| I-VT fixation detection | observed webcam, reduced slow jitter, no correction | 15 | 0.331 | 0.333 | 0.213 | 0.411 |
| I-VT fixation detection | observed webcam, reduced slow jitter, no correction | 10 | 0.339 | 0.340 | 0.237 | 0.423 |
| I-VT fixation detection | observed webcam (2.2b), with correction | 30 | 0.372 | 0.373 | 0.291 | 0.475 |
| I-VT fixation detection | observed webcam (2.2b), with correction | 20 | 0.363 | 0.363 | 0.260 | 0.469 |
| I-VT fixation detection | observed webcam (2.2b), with correction | 15 | 0.346 | 0.347 | 0.226 | 0.452 |
| I-VT fixation detection | observed webcam (2.2b), with correction | 10 | 0.366 | 0.368 | 0.269 | 0.469 |
| I-VT fixation detection | observed webcam, reduced slow jitter, with correction | 30 | 0.340 | 0.342 | 0.240 | 0.425 |
| I-VT fixation detection | observed webcam, reduced slow jitter, with correction | 20 | 0.334 | 0.334 | 0.231 | 0.415 |
| I-VT fixation detection | observed webcam, reduced slow jitter, with correction | 15 | 0.320 | 0.321 | 0.201 | 0.388 |
| I-VT fixation detection | observed webcam, reduced slow jitter, with correction | 10 | 0.333 | 0.332 | 0.215 | 0.416 |

`sens_rate_table.csv` has every number above (plus clean and simulated measured noise); `sens_contrasts.csv` the paired contrasts.

## Statement on the level (no end is claimed correct)
For dwell from samples at 30 Hz the observed-webcam condition costs between about 0.015 and 0.025 AUC (reduced slow jitter: AUC 0.748 without and 0.758 with correction, -0.025 [-0.039, -0.011] and -0.015 [-0.027, -0.003] from clean) and about 0.045 AUC (2.2b: 0.728, -0.045 [-0.064, -0.027]) from clean data, so the true cost under this camera most likely lies between roughly 0.015 and 0.045, that is a dwell-from-samples AUC of roughly 0.73 to 0.76 at 30 Hz, and the same range holds at 20-10 Hz within about 0.01 (reduced minus clean -0.015 to -0.032; 2.2b -0.045 to -0.051). Neither end can be called correct: the 2.2b run matches the measured jitter statistics but fails the drift check three to four times as often as the recordings do (20-25% against 7%), whereas the reduced run is close to the recorded failure rate (8% at 30 Hz) but under-reproduces the measured jitter (settled SD 1.40 against 2.10 deg in x), and both apply a larger correction than was logged (3.0-3.3 against 2.64 deg). The reduced condition is also not a fit: it removes slow variance until a total SD matches, which is a bound rather than an estimate. The conclusions that do not depend on the end: dwell from samples is lower than on clean data under both (all intervals below zero), its change from 30 to 10 Hz stays within the 0.02 margin by the one-sided bound under both (worst lower bounds, both at 10 Hz: -0.0135 without and -0.0193 with correction for 2.2b; -0.0179 and -0.0145 for reduced), and dwell from samples beats I-VT at every rate under both (+0.054 [+0.039, +0.067] at 30 Hz and +0.037 [+0.026, +0.048] at 10 Hz for the reduced run, no correction).

## Plain summary (5 lines)
1. With the slow jitter scaled down to the measured residual SD, dwell from samples scores AUC 0.748 at 30 Hz without and 0.758 with the app's correction (2.2b: 0.728 and 0.728); the cost against clean data is -0.025 [-0.039, -0.011] and -0.015 [-0.027, -0.003] (2.2b: -0.045).
2. The change from 30 to 10 Hz is -0.011 [-0.019, -0.002] without and -0.006 [-0.016, +0.003] with correction; the lower one-sided bounds (-0.018, -0.015) are above the -0.02 margin (p 0.016, 0.005), as in 2.2b (exploratory).
3. Dwell minus I-VT stays positive at all four rates (30 Hz +0.054 [+0.039, +0.067], 10 Hz +0.037 [+0.026, +0.048] without correction), as in 2.2b.
4. The correction now helps a little (dwell +0.010 [+0.002, +0.019] at 30 Hz; 2.2b: +0.0005), the drift check fails in 8% (30 Hz) and 13% (10 Hz) of simulated trials against 7% recorded (2.2b: 20-25%), and the mean applied correction is 3.0 deg against 2.64 recorded (2.2b: 3.3).
5. The reduced run under-reproduces the measured jitter (SD 1.40 against 2.10 deg), so the true level lies between the two runs, roughly 0.73 to 0.76 AUC at 30 Hz; neither end is claimed correct; one person, one camera, simulated rates, exploratory.

## Deviations, assumptions and caveats
- `sa_run.py cv` re-types the loop of `s4_run.cv_stage` (same calls) without the draw-1 regular-timing pipeline check, which has no jitter and is identical to 2.2b; `s4_run.feat_stage` was used as is and also computed the regular-timing features for draw 1 (unused).
- `s5_stats.py` writes into the 2.2b folder, so its method was re-typed in `sa_stats.py`; the 2.2b rows are reproduced exactly (difference 1e-16). `s0b_fit_jitter.py` fits when imported, so its estimator and simulator were re-typed in `sa_checks.py` (verified against the stored 2.2b check, difference 0.0).
- The reduced process is a rule-based bound, not a refit; the jitter statistics are worse than 2.2b's by construction (table above).
- The "recorded" drift-failure rate and applied correction come from one person (85 trials, 6 failures); its interval is wide (2.6% to 14.7%).
- Mean applied correction is trial-weighted over participants and draws; the unweighted values differ by less than 0.002 deg.
- All caveats of 2.2b apply (frame timing of 20-10 Hz scaled from 30 fps, bursty loss not simulated, Gaussian marginals, Delmas viewers are other people, margin chosen after screening).
- Part B (`results/phase10/model_compare_observed/`) ran its OBS_R condition on the 30 Hz, no-correction features of this run, after this run had finished.

## Files
`PLAN.md`, `PLAN.sha256.txt`, `jitter_params_reduced.json`, `sa_params.py`, `sa_run.py`, `sa_checks.py`, `sa_stats.py`, `sens_rate_table.csv`, `sens_contrasts.csv`, `sens_tables.md`, `sens_realism.json`, `sens_repro_check.json`, `features/` (per participant and rate), `checkpoints/` (nested-CV outputs per tag), logs `log_sa_run.txt`, `log_sa_checks.txt`, `log_sa_stats.txt`.
