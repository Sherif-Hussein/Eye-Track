# webcam_noise: does the 30 Hz non-inferiority claim survive webcam spatial error? (2026-11-02)

**Answer: no.** Once realistic calibration offset, jitter and dropout are added, ALL_z LR at 30 Hz is no longer non-inferior to P4 SVM at 1000 Hz (margin 0.02 AUC). At the first noise level (1 deg mean offset) the difference to the reference is -0.028 (95% CI [-0.049, -0.007]), i.e. 0.033 AUC below the clean 30 Hz result. The largest non-inferior noise level is **0 deg (clean data only)**. An I-VT velocity threshold scaled with the known jitter recovers most of the loss, but non-inferiority is still not shown at any noise level, and the CIs are too wide to call it inferior up to 2 deg.

The design was fixed in `PREREG.md` before the first noisy run (only the extraction of the clean 30 Hz cache preceded it; one typo in the adaptive thresholds was corrected before any noisy run). Code: `scripts/phase3_webcam_noise.py`, `src/edd/webcam_noise.py`. Every number below comes from `results.json`, `curve_table.md/csv` and `metrics_table.md`.

## 1. Setup (summary of PREREG.md)
- Geometry (read from the authors' Psychtoolbox design files for all 75 participants, and the paper): 57 cm, 546 x 304 mm screen, 1920 x 1080, 34.9869 px/deg (1 deg = 35 px; picture half-width 4.3 deg, gap between pictures 4.0 to 9.7 deg).
- Noise on the decimated 30 Hz gaze, before event detection: (a) one random-direction calibration offset per participant for the whole session, magnitude Rayleigh with mean = level; (b) Gaussian jitter per sample and axis, SD 0.5 + 0.25 x level deg; (c) 10% random sample loss at every non-zero level. Levels 0, 1, 1.5, 2, 3, 4 deg; 3 noise draws per non-zero level (seeded; the same standard-normal draws are rescaled across levels). Realised offsets: mean 0.95 to 1.06 deg at level 1, 1.43 to 1.59 at level 1.5, 3.82 to 4.25 at level 4; participant maxima 2.3 to 10.7 deg (`checkpoints/diag_*.json`). Realised missing fraction 11.2 to 11.3% (1.4% clean).
- Grounding: Kaduk et al. 2024 (Labvanced webcam, 30 Hz, Large Grid task, n = 19; full text read): accuracy 1.45 deg, precision 1.23 deg (RMS-S2S 1.4 to 1.7 deg), calibration error 1.44 deg, sample loss 1.5 to 3%. So **level 1.5 deg is the best published webcam case**; levels 3 to 4 deg are the WebGazer regime (the ">4 deg" figure is Kaduk's second-hand statement; the original WebGazer papers were not read). The 10% loss is deliberately above the measured 1.5 to 3%.
- Models: ALL_z LR (13 per-participant z-scored features) primary; P4 LR (4 features) secondary; ALL_z LR with an adaptive I-VT threshold secondary (30 + 3 SD of the jitter-induced velocity: 77.7, 85.7, 93.6, 109.5, 125.5 deg/s at levels 1 to 4). Otherwise fixed I-VT (30 deg/s, +-4 ms differentiator, 60 ms minimum fixation), no re-tuning per level.
- Evaluation: frozen `results/phase2/delmas_raw/folds.json`, nested participant-grouped tuning (track code), image-level AUC pooled per set on repeat-averaged OOF scores, mean over 6 sets; a noise level's AUC is the mean over its 3 draws. Paired participant-cluster bootstrap, 2000 resamples (seed 0), versus clean P4 SVM 1000 Hz (AUC 0.7811, 95% CI [0.7402, 0.8192]). H0: Delta <= -0.02, one-sided; Holm over a pre-specified family of 17 tests (ALL_z fixed x6 levels, P4 fixed x6, ALL_z adaptive x5).
- Checks passed: cached 30 Hz features equal the stored clean features (max abs diff 4.5e-13); level-0 OOF scores equal the stored files (ALL_z LR max abs score diff 3.5e-12, P4 LR 5.3e-15; same rows).

## 2. Degradation curve (primary model: ALL_z LR, 30 Hz, fixed I-VT)
| offset (deg) | jitter SD (deg) | AUC [95% CI] | Delta vs P4 SVM 1000 Hz [95% CI] | raw p | Holm p (17) | Holm p (6 primary tests) | Delta vs own clean 30 Hz [95% CI] |
|---|---|---|---|---|---|---|---|
| 0 (clean) | 0 | 0.7854 [0.7432, 0.8231] | +0.0044 [-0.0159, +0.0231] | 0.0070 | 0.119 | 0.042 | n/a |
| 1 | 0.750 | 0.7527 [0.7165, 0.7858] | -0.0284 [-0.0492, -0.0071] | 0.823 | 1.000 | 1.000 | -0.0327 [-0.048, -0.0165] |
| 1.5 | 0.875 | 0.7418 [0.7080, 0.7722] | -0.0393 [-0.0615, -0.0189] | 0.968 | 1.000 | 1.000 | -0.0436 [-0.0636, -0.0242] |
| 2 | 1.000 | 0.7267 [0.6945, 0.7576] | -0.0544 [-0.0767, -0.0337] | 0.9995 | 1.000 | 1.000 | -0.0587 [-0.0796, -0.0376] |
| 3 | 1.250 | 0.7005 [0.6756, 0.7252] | -0.0806 [-0.1077, -0.0542] | 1.000 | 1.000 | 1.000 | -0.0849 [-0.1107, -0.058] |
| 4 | 1.500 | 0.6416 [0.6194, 0.6651] | -0.1395 [-0.1697, -0.1097] | 1.000 | 1.000 | 1.000 | -0.1438 [-0.1741, -0.1133] |

AUC per draw (draw SD 0.006 to 0.013): 1 deg 0.7366 / 0.7689 / 0.7527; 1.5 deg 0.7285 / 0.7562 / 0.7407; 2 deg 0.7145 / 0.7351 / 0.7305; 3 deg 0.6993 / 0.7082 / 0.6941; 4 deg 0.6318 / 0.6452 / 0.6479. At 1 deg the spread between draws is about 0.03, as large as the margin.

**Primary outcome: the largest non-inferior level is 0 deg.** In the pre-registered 17-test family not even level 0 passes (Holm p = 0.119; raw p = 0.007), because the large family charges a heavy multiplicity adjustment. Within the six primary-model tests alone, only level 0 passes (Holm p = 0.042). No noisy level is non-inferior under either family (smallest raw p 0.823). From 1.5 deg upward the upper 95% CI bound of Delta is -0.019 or lower, i.e. at the margin or beyond it; at 1 deg the upper bound is -0.007, so that interval does not exclude a true loss smaller than 0.02.

## 3. Secondary models
| offset (deg) | P4 LR fixed: AUC [95% CI] | Delta vs P4 SVM 1000 Hz [95% CI] | ALL_z LR adaptive I-VT: AUC [95% CI] | Delta vs P4 SVM 1000 Hz [95% CI] | raw p | Holm p (17) |
|---|---|---|---|---|---|---|
| 0 | 0.7772 [0.7346, 0.8184] | -0.0039 [-0.0184, +0.0095] | (same as fixed) | | | |
| 1 | 0.7444 [0.7085, 0.7791] | -0.0367 [-0.0527, -0.0225] | 0.7693 [0.7298, 0.8039] | -0.0118 [-0.0302, +0.0055] | 0.218 | 1.00 |
| 1.5 | 0.7370 [0.7029, 0.7692] | -0.0440 [-0.0598, -0.0292] | 0.7704 [0.7311, 0.8054] | -0.0107 [-0.0300, +0.0070] | 0.186 | 1.00 |
| 2 | 0.7189 [0.6860, 0.7494] | -0.0622 [-0.0795, -0.0457] | 0.7661 [0.7279, 0.8013] | -0.0149 [-0.0354, +0.0044] | 0.368 | 1.00 |
| 3 | 0.6624 [0.6389, 0.6898] | -0.1186 [-0.1423, -0.0911] | 0.7573 [0.7200, 0.7906] | -0.0238 [-0.0446, -0.0043] | 0.693 | 1.00 |
| 4 | 0.6122 [0.5935, 0.6356] | -0.1688 [-0.1973, -0.1371] | 0.7341 [0.7001, 0.7656] | -0.0469 [-0.0696, -0.0258] | 0.991 | 1.00 |

- P4 LR (4 features, no z-scoring) is below ALL_z LR at every noisy level and is not non-inferior at any of them (P4 LR clean: Delta -0.0039, raw p 0.0125, Holm p (17) 0.200).
- The adaptive threshold keeps ALL_z LR at AUC 0.757 to 0.770 up to 3 deg (Delta -0.011 to -0.024). Its CIs contain both 0 and the -0.02 margin up to 2 deg, so the data neither show non-inferiority nor exclude a loss under 0.02. At the best-webcam point (1.5 deg) it gives Delta -0.0107 [-0.0300, +0.0070]. The adaptive threshold uses the known jitter of the simulation; in practice it would need a validation measurement of the noise.
- Diagnostic (an observation, not a tested mechanism): mean n_fix per face is 2.36 clean; with the fixed 30 deg/s threshold it is 3.52 at 1 deg, 3.28 at 1.5, 2.85 at 2, 1.8 at 3 and 1.0 at 4 deg; with the adaptive threshold 2.96 at 1 deg, 2.9 at 1.5 and 2, 2.5 to 2.7 at 3, and 2.0 to 2.35 at 4. The jitter-induced velocity noise (about 16 deg/s SD per axis at 1 deg) first fragments fixations, and at larger noise fixed-threshold fixations become rare.

## 4. All metrics
Image level, mean over the 6 sets, over repeats, over noise draws; threshold 0 on LR scores; CIT = participant-level decision. AUC here is the track definition (mean of per-repeat pooled AUC), which differs from the non-inferiority AUC above by up to 0.008. CM-sets AUC = mean of G2, G3-32, G4-32, G5-32. Reference row: P4 SVM 1000 Hz, clean. Row labels are model|I-VT mode|offset x 100.

| config | accuracy | balanced_accuracy | sensitivity | specificity | precision | f1 | mcc | auc | eer | hter | CM-sets auc | cit_auc | cit_hter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P4_svm_1000 (ref) | 0.763 | 0.724 | 0.668 | 0.780 | 0.359 | 0.463 | 0.359 | 0.776 | 0.278 | 0.276 | 0.726 | 0.882 | 0.267 |
| ALLz fixed 000 | 0.751 | 0.738 | 0.719 | 0.756 | 0.356 | 0.474 | 0.373 | 0.781 | 0.266 | 0.262 | 0.731 | 0.875 | 0.293 |
| ALLz fixed 100 | 0.720 | 0.702 | 0.677 | 0.727 | 0.315 | 0.428 | 0.311 | 0.751 | 0.298 | 0.298 | 0.691 | 0.842 | 0.332 |
| ALLz fixed 150 | 0.709 | 0.692 | 0.669 | 0.716 | 0.302 | 0.415 | 0.294 | 0.741 | 0.308 | 0.308 | 0.682 | 0.834 | 0.347 |
| ALLz fixed 200 | 0.697 | 0.680 | 0.657 | 0.704 | 0.287 | 0.398 | 0.272 | 0.725 | 0.320 | 0.320 | 0.664 | 0.821 | 0.369 |
| ALLz fixed 300 | 0.663 | 0.658 | 0.650 | 0.666 | 0.252 | 0.362 | 0.230 | 0.698 | 0.343 | 0.342 | 0.642 | 0.824 | 0.433 |
| ALLz fixed 400 | 0.612 | 0.606 | 0.599 | 0.614 | 0.210 | 0.310 | 0.152 | 0.641 | 0.393 | 0.394 | 0.589 | 0.753 | 0.481 |
| P4 fixed 000 | 0.733 | 0.732 | 0.731 | 0.733 | 0.337 | 0.458 | 0.357 | 0.774 | 0.270 | 0.268 | 0.723 | 0.866 | 0.296 |
| P4 fixed 100 | 0.709 | 0.700 | 0.688 | 0.712 | 0.309 | 0.424 | 0.306 | 0.744 | 0.301 | 0.300 | 0.682 | 0.855 | 0.333 |
| P4 fixed 150 | 0.701 | 0.693 | 0.681 | 0.705 | 0.298 | 0.412 | 0.292 | 0.737 | 0.307 | 0.307 | 0.678 | 0.854 | 0.352 |
| P4 fixed 200 | 0.681 | 0.677 | 0.671 | 0.683 | 0.274 | 0.388 | 0.262 | 0.718 | 0.323 | 0.323 | 0.659 | 0.829 | 0.372 |
| P4 fixed 300 | 0.636 | 0.632 | 0.626 | 0.638 | 0.229 | 0.334 | 0.190 | 0.661 | 0.367 | 0.368 | 0.612 | 0.792 | 0.446 |
| P4 fixed 400 | 0.582 | 0.593 | 0.607 | 0.578 | 0.195 | 0.294 | 0.130 | 0.613 | 0.408 | 0.407 | 0.570 | 0.743 | 0.479 |
| ALLz adaptive 100 | 0.738 | 0.719 | 0.692 | 0.745 | 0.337 | 0.451 | 0.342 | 0.767 | 0.281 | 0.281 | 0.707 | 0.862 | 0.311 |
| ALLz adaptive 150 | 0.737 | 0.719 | 0.694 | 0.744 | 0.337 | 0.451 | 0.342 | 0.767 | 0.282 | 0.281 | 0.709 | 0.863 | 0.312 |
| ALLz adaptive 200 | 0.733 | 0.716 | 0.691 | 0.740 | 0.332 | 0.446 | 0.335 | 0.763 | 0.285 | 0.284 | 0.704 | 0.851 | 0.312 |
| ALLz adaptive 300 | 0.721 | 0.709 | 0.692 | 0.726 | 0.316 | 0.431 | 0.319 | 0.755 | 0.292 | 0.291 | 0.700 | 0.856 | 0.344 |
| ALLz adaptive 400 | 0.694 | 0.683 | 0.667 | 0.699 | 0.285 | 0.397 | 0.275 | 0.732 | 0.317 | 0.317 | 0.678 | 0.834 | 0.378 |

Per-set AUCs are in `results.json` under `metrics_mean_over_draws`.

## 5. Caveats
- This is a simulation: a research tracker (EyeLink 1000) decimated to 30 Hz plus modelled noise, not real webcam recordings. Real webcams also have spatially varying error (worse at the screen edge), head-movement drift within a session, irregular frame timing and different blink handling. The offset here is constant for the whole session by design, so drift is not covered.
- The levels rest on one webcam study (Kaduk et al.; n = 19, grid task, 30 Hz). The WebGazer figure is second-hand. The 10% loss exceeds the measured 1.5 to 3% and is the same at all non-zero levels, so its effect is not separated from the offset and jitter effects (a no-loss run was not part of the pre-registration).
- The jitter rule couples jitter to offset, so the curve is one path through the noise space, not separate effects of offset and jitter. At 1.5 deg the rule gives a per-axis jitter SD of 0.875 deg (RMS-S2S about 1.24 deg), somewhat below Kaduk's RMS-S2S of 1.4 to 1.7 deg, so the rule is not conservative for that point.
- Only 3 noise draws per level, with a draw-to-draw AUC spread up to 0.03 at 1 deg. The participant bootstrap CI does not include noise-draw variance, so the CIs are somewhat too narrow.
- 75 participants (14 to 15 per group); CIs are about +-0.02, the size of the margin. A failure to show non-inferiority is not proof of inferiority, except from 1.5 deg upward where the Delta CI sits at or beyond -0.02.
- The Holm family here (17 tests) is larger than the kept result's (9 tests), so level-0 p-values differ from the kept ones (raw 0.0070 here vs 0.0045 kept; this run uses the seed-0 stream from its start, the kept run used the 6th pair of one shared stream). The level-0 point estimates are the same as the kept result (Delta +0.0044; ALL_z AUC 0.7854, reference 0.7811). With the 6-test family the clean claim holds (Holm p = .042); with the 17-test family it does not (p = .119). That difference reflects multiplicity, not noise.
- Labels and inclusion follow the authors' BookFIP tables, as in the kept result. ALL_z needs all 64 images of a participant (per-participant z-scoring). The cost figures of the kept result are unchanged by this experiment (noise was not timed).

## 6. Files
`PREREG.md`, `results.json` (all tests, primary outcome, metrics, diagnostics, checks), `curve_table.md/csv`, `metrics_table.md`, `checkpoints/` (OOF scores and metrics for the 47 configurations, `diag_*.json`, `level_*_done.json`, `clean_check.json`), `cache_30hz/` (decimated signals), `run_log.txt`, `stats_log.txt`, `finisher.sh`, `RESUME.md`. Command: `OMP_NUM_THREADS=1 .venv/Scripts/python.exe -W ignore -I scripts/phase3_webcam_noise.py --stages extract,check,run,stats`.
