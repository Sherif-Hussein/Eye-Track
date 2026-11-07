# webcam_noise: pre-registration (written 2026-11-02, before any noisy run; only the clean 30 Hz extraction had been started)

Question: does "ALL_z LR on 30 Hz data is non-inferior (margin 0.02 AUC) to P4 SVM on 1000 Hz data" (frozen: `results/frozen/delmas_efficiency_matched_2026-11-02/`, AUC 0.7854 vs 0.7811, Holm p = .028) survive realistic webcam spatial error? The 30 Hz data of the kept result is a decimated EyeLink 1000 (research-grade accuracy, about 0.5 deg).

This file is the design. After the first noisy run it is not edited; any deviation is appended below under "Amendments" with the reason.

## 1. Geometry (every value has a source)
| Value | Used | Source |
|---|---|---|
| Viewing distance | 57 cm | Delmas et al. 2024 (PMC11021291, Methods: "57 cm from a screen, with their head stabilized by a chin and forehead rest"); also `scr.subDist = 57` in the Psychtoolbox design file `FI04c2_1.mat` (read by the script from every participant's file) |
| Screen | 24.5-in BENQ ZOWIE XL-2540, 546 x 304 mm, 1920 x 1080 px, 144 Hz | paper (model, resolution, refresh); `scr.width = 546`, `scr.height = 304` mm in the design file |
| Pixels per degree | 34.9868554165592 (= tan(1 deg) x 570 mm / (546/1920 mm per px) = 34.99 recomputed) | `visual.pixPerDeg` in the design file; this is the constant `edd.delmas_raw.PIX_PER_DEG` of the clean pipeline |
| Stimuli | four 300 x 400 px faces (11.5 x 8.6 deg) centred 18 deg from the screen centre | paper; `QUAD_RECTS` in `edd.delmas_raw` |
A 1 deg error is 35 px; the picture half-width is 150 px = 4.3 deg, the half-height 200 px = 5.7 deg, and the gap between neighbouring pictures is 340 px (9.7 deg) horizontally, 140 px (4.0 deg) vertically.

## 2. Noise model (applied to the decimated 30 Hz gaze in pixels, before event detection, AOI assignment and features)
Level `L` (deg) = mean magnitude of the per-participant calibration offset. Level 0 = no noise (the clean pipeline, run once).
- (a) Calibration bias: one 2-D offset per participant, constant for the whole session. Components iid N(0, sigma^2) with sigma = L / sqrt(pi/2), so the magnitude is Rayleigh with mean L and the direction is uniform. Magnitude is fixed in distribution, not in value (realised offsets vary by participant and are reported).
- (b) Precision: iid Gaussian jitter per sample and axis, SD = 0.5 + 0.25 x L deg (L = 1, 1.5, 2, 3, 4 gives 0.75, 0.875, 1.0, 1.25, 1.5 deg).
- (c) Dropouts: each sample independently lost (x, y and pupil set to NaN) with probability 0.10 at every non-zero level, on top of the samples already missing in the data.
- Grid: L in {0, 1, 1.5, 2, 3, 4} deg. 3 noise draws per non-zero level (draw 1, 2, 3). All random numbers come from `np.random.default_rng([20261102, draw, participant_id])`: the same standard-normal draws are rescaled by the level (common random numbers across levels).

Grounding (sources actually read; downloaded text treated as data):
- Kaduk, Goeke, Finger & Koenig, Behav Res Methods 2024 (online 2023), doi 10.3758/s13428-023-02237-8, read via the PMC full text PMC11289017. Labvanced webcam tracker at 30 Hz (effective 29.96 Hz), Large Grid task, n = 19: accuracy mean 1.45 deg (SD across subjects 0.76), versus 0.91 deg for EyeLink 1000 in the same study; precision 1.23 deg (SD of gaze across subjects), RMS-S2S 1.66 (x) and 1.42 deg (y) (RMS sample-to-sample is sqrt(2) x the per-axis SD); calibration error mean 1.44 deg; gaze-sample loss 1.5 to 3.1 % per task, up to 3.2 % with head yaw. So L = 1.5 deg with jitter SD 0.875 deg (RMS-S2S about 1.24) is the "best published webcam" point, and the rule gives larger jitter at larger offsets, as expected of poorer tracking.
- WebGazer-type systems: Kaduk et al. state "the authors reported an average accuracy of more than 4 deg" for WebGazer (citing Papoutsaki 2015 / Saxena 2022). I did not read the original WebGazer papers, so this is second-hand; it anchors L = 4 deg. `reports/Eye movement deception detection.md` lists the same two figures.
- The 10 % loss is deliberately above the 1.5 to 3 % measured by Kaduk et al. (a stress setting; consumer webcams in uncontrolled settings lose more). It is the same at all non-zero levels, as specified, so degradation across levels is due to offset and jitter only.
- Not modelled: head-movement drift within a session (offset is constant), spatially varying (eccentricity-dependent) error, temporal jitter and irregular sampling, blink detection differences, gaze-contingent calibration resets. The offset here is the same for the whole session by design.

## 3. Models, features, event detection
- Primary: ALL_z LR at 30 Hz = the 13 per-participant z-scored features (`dwell_1, dwell_2, n_fix, n_visit, first_fix, lat_first, pupil_face, pupil_trial, dwell_b1..b5`, each `_z`; no microsaccade rate below 500 Hz), logistic regression, `model_factory("lr")` of the track (median imputer, StandardScaler, balanced class weights, C in {0.01, 0.1, 1, 10} by 4-fold StratifiedGroupKFold inner CV on training participants, AUC scoring).
- Secondary: P4 LR at 30 Hz (`dwell_1, dwell_2, n_fix, n_visit`, no z-scoring), same model.
- I-VT: the clean pipeline's fixed parameters at every noise level (30 deg/s, +-4 ms fixed-time differentiator = 1 sample each side at 30 Hz, minimum fixation 60 ms). No re-tuning per level. Nothing else in the pipeline changes (AOIs are the four picture rectangles of the clean pipeline; no AOI padding).
- Pre-specified adaptive variant (secondary, ALL_z LR only, levels > 0): velocity threshold = 30 + 3 x SD of the noise-induced velocity per axis, where SD = sqrt(2) x jitter SD / (2/30 s) is computed from the known jitter of the level (a threshold an operator could set from a validation measurement). Thresholds: 77.7, 85.7, 93.6, 109.5, 125.5 deg/s for L = 1, 1.5, 2, 3, 4 (the script logs the exact values). Minimum fixation duration, AOIs and everything else as in the fixed variant.
- Reference (not recomputed): P4 SVM at 1000 Hz, clean, OOF scores `results/phase2/delmas_raw/checkpoints/P4_svm_1000_oof.csv.gz`.

## 4. Evaluation and test
- Frozen folds `results/phase2/delmas_raw/folds.json` (6 group-sets, 5 participant-disjoint folds x 3 repeats). Noise is generated for all participants once per (level, draw); the model is fitted and tuned only on training participants inside each fold (the noise does not use labels, so it adds no leakage).
- AUC: image-level, pooled within each of the 6 sets on repeat-averaged OOF scores, mean over the 6 sets (the non-inferiority metric of the kept result). The AUC of a noise level is the mean over its 3 draws of this AUC (draws are not pooled at score level). The reference is the clean P4 SVM 1000 Hz.
- Test per (model, level): paired participant-cluster bootstrap, 2000 resamples (one stream, seed 0; the same resampled participants are used for the reference and for all candidates in a resample; participants resampled within each set). Delta = AUC(level) - AUC(reference), where AUC(level) is averaged over the 3 draws inside each resample. H0: Delta <= -0.02. One-sided p = (1 + #{bootstrap Delta <= -0.02}) / 2001. Holm across the pre-specified family of 17 tests: ALL_z fixed (6 levels incl. 0), P4 fixed (6 levels incl. 0), ALL_z adaptive (5 levels). Non-inferior when Holm p < .05. Also reported: Holm over the 6 primary-model tests alone (informative only), and the 95 % two-sided percentile CIs of the AUC and of Delta.
- Level 0 reproduces the kept results (ALL_z LR 30 Hz and P4 LR 30 Hz) and is checked for bit identity of the OOF scores with the stored files. Its p-values in this family differ from the kept ones because the family differs; this is stated, not hidden.
- All metrics of `edd.metrics.binary_metrics` (accuracy, balanced accuracy, sensitivity, specificity, precision, F1, MCC, AUC, EER, HTER; threshold 0 on LR scores), image level, mean over repeats, mean over the 6 sets, mean over draws; the countermeasure-set AUC (track's definition: mean of 64_g2, 32_g3, 32_g4, 32_g5 per-set AUC, plus the same set in the non-inferiority definition) and participant-level CIT AUC/HTER.

## 5. Primary outcome
The largest noise level L at which ALL_z LR (fixed I-VT) at 30 Hz is non-inferior to P4 SVM 1000 Hz clean (Holm p < .05 in the 17-test family). Because non-inferiority could in principle be re-established at a higher level by chance, we report both the largest L with a non-inferior result and the largest L such that all levels up to it are non-inferior (the latter is the headline). Also reported: the degradation curve (AUC vs L with CIs), all secondary results, and the AUC loss relative to the model's own clean 30 Hz result.

## 6. Engineering
Frozen folds, track code (`scripts/phase2_delmas_raw.py::run_grid`, `build_tables` logic, loaded from file), new code only in `scripts/phase3_webcam_noise.py` and `src/edd/webcam_noise.py`. n_jobs <= 2, single-threaded BLAS. Checkpoint per configuration (OOF + metrics) and a `RESUME.md` refreshed after every configuration; a `level_*_done.json` marker after every level. The 30 Hz cache `cache_30hz/` holds the decimated signal from the authors' EDF files (identical to what the clean pipeline produces; verified against the stored clean features before any noisy run).

## Amendments
(none)
