# Phase 3: fast_extraction (vectorised feature extraction), before / after

Single thread (`edd.efficiency`, `threadpool_limits(1)`, `OMP_NUM_THREADS=1`). Original = the track's code; fast = `src/edd/delmas_fast.py` / `src/edd/celniak_fast.py`. Both measured back to back in the same session, in interleaved rounds (order alternated each round); tables give the median over rounds of each round's p50 and p95. Timings were taken while other jobs used the CPU (a very-low-priority phase1_celniak.py job, plus whatever else was running), so every absolute time is an upper bound; original and fast were measured back to back, interleaved, in the same session.

**Verdict.** Delmas: features equivalent yes; out-of-fold AUC unchanged yes; speed-up claimed: yes. Celniak: features equivalent yes; out-of-fold AUC unchanged yes; speed-up claimed: yes.

## 1. Delmas raw-sample pipeline (Sj_04, 64 trials x 5.5 s; EDF decoding excluded)

Timed path (as in the frozen matched-cost measurement): sensor-rate samples -> I-VT -> AOI -> dwell / fixation / visit / first-look / time-bin / pupil (and, at 1000 Hz for ALL_z, microsaccade) features -> per-image aggregation -> (per-participant z-scoring for ALL_z) -> model score. Decimation from 1000 Hz is simulated before the timer (as in the track) and reported separately below for 30 Hz.

### 1a. Latency (ms, p50 / p95)

| Model | Unit | Stage | Original | Fast | Speed-up (p50) |
|---|---|---|---|---|---|
| P4_svm_1000 | participant (64 images) | feature extraction only | 542.76 / 559.21 | 116.280 / 122.707 | 4.7x |
| P4_svm_1000 | participant (64 images) | inference only (same model code) | 38.136 / 42.411 | 38.136 / 42.411 | 1.0x |
| P4_svm_1000 | participant (64 images) | **end to end** | 554.97 / 562.07 | 157.292 / 176.168 | **3.5x** |
| P4_svm_1000 | trial (4 faces) | feature extraction only | 7.88 / 9.14 | 4.217 / 4.657 | 1.9x |
| P4_svm_1000 | trial (4 faces) | inference only (same model code) | 4.703 / 5.617 | 4.703 / 5.617 | 1.0x |
| P4_svm_1000 | trial (4 faces) | **end to end** | 13.12 / 14.54 | 9.442 / 10.482 | **1.4x** |
| P4_lr_30 | participant (64 images) | feature extraction only | 328.81 / 332.11 | 9.145 / 9.507 | 36.0x |
| P4_lr_30 | participant (64 images) | inference only (same model code) | 2.168 / 2.460 | 2.168 / 2.460 | 1.0x |
| P4_lr_30 | participant (64 images) | **end to end** | 338.60 / 364.50 | 11.394 / 13.527 | **29.7x** |
| P4_lr_30 | trial (4 faces) | feature extraction only | 5.24 / 6.10 | 2.563 / 3.050 | 2.0x |
| P4_lr_30 | trial (4 faces) | inference only (same model code) | 2.134 / 2.497 | 2.134 / 2.497 | 1.0x |
| P4_lr_30 | trial (4 faces) | **end to end** | 7.72 / 8.80 | 4.957 / 5.856 | **1.6x** |
| ALLz_lr_30 | participant (64 images) | feature extraction only | 320.29 / 340.96 | 9.604 / 10.583 | 33.3x |
| ALLz_lr_30 | participant (64 images) | inference only (same model code) | 2.109 / 2.524 | 2.109 / 2.524 | 1.0x |
| ALLz_lr_30 | participant (64 images) | **end to end** | 326.51 / 341.25 | 12.605 / 13.905 | **25.9x** |
| ALLz_lr_30 | trial (4 faces) | feature extraction only | 5.01 / 5.57 | 2.870 / 3.273 | 1.7x |
| ALLz_lr_30 | trial (4 faces) | inference only (same model code) | 2.105 / 2.482 | 2.105 / 2.482 | 1.0x |
| ALLz_lr_30 | trial (4 faces) | **end to end** | 7.06 / 8.51 | 4.996 / 5.863 | **1.4x** |
| P4_lr_1000 | participant (64 images) | feature extraction only | 532.29 / 560.16 | 118.288 / 128.207 | 4.5x |
| P4_lr_1000 | participant (64 images) | inference only (same model code) | 2.231 / 2.487 | 2.231 / 2.487 | 1.0x |
| P4_lr_1000 | participant (64 images) | **end to end** | 542.82 / 569.01 | 120.197 / 127.322 | **4.5x** |
| P4_lr_1000 | trial (4 faces) | feature extraction only | 7.97 / 8.55 | 4.141 / 4.836 | 1.9x |
| P4_lr_1000 | trial (4 faces) | inference only (same model code) | 2.120 / 2.612 | 2.120 / 2.612 | 1.0x |
| P4_lr_1000 | trial (4 faces) | **end to end** | 10.54 / 12.13 | 6.724 / 8.141 | **1.6x** |
| ALLz_lr_1000 | participant (64 images) | feature extraction only | 667.66 / 682.71 | 217.070 / 225.190 | 3.1x |
| ALLz_lr_1000 | participant (64 images) | inference only (same model code) | 2.137 / 2.545 | 2.137 / 2.545 | 1.0x |
| ALLz_lr_1000 | participant (64 images) | **end to end** | 648.21 / 659.27 | 219.643 / 230.115 | **3.0x** |
| ALLz_lr_1000 | trial (4 faces) | feature extraction only | 9.75 / 11.51 | 6.378 / 7.601 | 1.5x |
| ALLz_lr_1000 | trial (4 faces) | inference only (same model code) | 2.126 / 2.862 | 2.126 / 2.862 | 1.0x |
| ALLz_lr_1000 | trial (4 faces) | **end to end** | 12.80 / 14.56 | 8.772 / 10.011 | **1.5x** |

### 1b. Throughput, real-time factor, memory

| Model | Participants/s orig -> fast | Samples/s orig -> fast | Real-time factor orig -> fast | Peak memory orig -> fast (chunk 64) -> fast (chunk 8) | Model size | Per-round e2e speed-up (min to max) |
|---|---|---|---|---|---|---|
| P4_svm_1000 | 1.80 -> 6.4 | 634,268 -> 2,237,875 | 634x -> 2,238x | 628 kB -> 28,757 kB -> 3,706 kB | 110,953 B | 3.3x to 3.8x |
| P4_lr_30 | 2.95 -> 87.8 | 31,187 -> 926,832 | 1,040x -> 30,894x | 95 kB -> 984 kB -> 160 kB | 1,463 B | 25.6x to 31.5x |
| ALLz_lr_30 | 3.06 -> 79.3 | 32,342 -> 837,793 | 1,078x -> 27,926x | 112 kB -> 984 kB -> 160 kB | 1,823 B | 25.2x to 27.7x |
| P4_lr_1000 | 1.84 -> 8.3 | 648,471 -> 2,928,520 | 648x -> 2,929x | 627 kB -> 28,757 kB -> 3,706 kB | 1,463 B | 4.2x to 4.6x |
| ALLz_lr_1000 | 1.54 -> 4.6 | 543,037 -> 1,602,602 | 543x -> 1,603x | 675 kB -> 40,724 kB -> 5,201 kB | 1,863 B | 2.7x to 3.1x |

Throughput = 1000 / end-to-end p50 per participant (sequential, one thread). Samples/s = input samples per participant (64 x samples per trial, including the 500 ms pre-onset window) / end-to-end p50. Real-time factor = samples/s divided by the sensor rate. Peak memory is the Python-tracked (tracemalloc) peak of one participant's pass; the fast path holds whole (64, n) arrays, `chunk 8` processes 8 trials at a time (same features, lower peak).

| Model | e2e per participant, fast, chunk 8 (p50 / p95 ms) |
|---|---|
| P4_svm_1000 | 137.195 / 144.085 |
| P4_lr_30 | 32.094 / 34.855 |
| ALLz_lr_30 | 32.201 / 34.745 |
| P4_lr_1000 | 99.405 / 104.841 |
| ALLz_lr_1000 | 171.312 / 179.130 |

### 1c. Including decimation from the 1000 Hz source samples (30 Hz models)

| Model | Original e2e (p50 / p95 ms) | Fast e2e | Speed-up (p50) |
|---|---|---|---|
| P4_lr_30 | 786.24 / 800.33 | 121.257 / 130.454 | 6.5x |
| ALLz_lr_30 | 786.18 / 798.46 | 120.564 / 129.603 | 6.5x |

Rounds: 5. Per-round runs: original participant 5 (+1 warm-up) and trial 30; fast participant 60 and trial 200. Max abs score difference original vs fast in the benchmarked paths: 2.22e-16.

### 1d. Equivalence of the features (all 75 participants)

`numpy.allclose(rtol=1e-9, atol=0)` and identical NaN pattern. 'Live original' = the original code re-run on the decoded samples; 'stored' = `data/derived/delmas_raw_face_trials.csv.gz`, the table the cross-validation used.

| Comparison | Rate | Max abs diff | NaN pattern equal | allclose | Values |
|---|---|---|---|---|---|
| live original vs fast: degraded_x | 1000 Hz | 0.00e+00 | yes | yes | 26,400,000 |
| live original vs fast: degraded_y | 1000 Hz | 0.00e+00 | yes | yes | 26,400,000 |
| live original vs fast: degraded_pupil | 1000 Hz | 0.00e+00 | yes | yes | 26,400,000 |
| live original vs fast: trial_features_all14 | 1000 Hz | 1.36e-12 | yes | yes | 268,800 |
| live original vs fast: trial_features_P4 | 1000 Hz | 0.00e+00 | yes | yes | 76,800 |
| live original vs fast: image_matrix_P4 | 1000 Hz | 0.00e+00 | yes | yes | 19,200 |
| live original vs fast: image_matrix_ALL_z | 1000 Hz | 2.22e-15 | yes | yes | 67,200 |
| live original vs fast: degraded_x | 30 Hz | 0.00e+00 | yes | yes | 792,000 |
| live original vs fast: degraded_y | 30 Hz | 0.00e+00 | yes | yes | 792,000 |
| live original vs fast: degraded_pupil | 30 Hz | 0.00e+00 | yes | yes | 792,000 |
| live original vs fast: trial_features_all14 | 30 Hz | 1.36e-12 | yes | yes | 268,800 |
| live original vs fast: trial_features_P4 | 30 Hz | 0.00e+00 | yes | yes | 76,800 |
| live original vs fast: image_matrix_P4 | 30 Hz | 0.00e+00 | yes | yes | 19,200 |
| live original vs fast: image_matrix_ALL_z | 30 Hz | 2.00e-15 | yes | yes | 62,400 |
| stored table vs fast: el_nfix | | 0.00e+00 | yes | yes | 19,200 |
| stored table vs fast: rate_1000_all14 | | 1.14e-12 | yes | yes | 268,800 |
| stored table vs fast: rate_1000_P4 | | 0.00e+00 | yes | yes | 76,800 |
| stored table vs fast: rate_30_all14 | | 1.36e-12 | yes | yes | 268,800 |
| stored table vs fast: rate_30_P4 | | 1.11e-16 | yes | yes | 76,800 |

### 1e. Out-of-fold AUC with the fast features (frozen folds, 5 folds x 3 repeats x 6 group-sets, nested tuning, n_jobs = 2)

The track's own `build_tables` + `run_grid` were run on the fast face table. Fast vs original track table (all features and z-scores): max abs diff 9.09e-13, allclose True, identical keys and labels True.

| Model | Image AUC mean over 6 sets: stored | fast | abs diff | Pooled-per-repeat AUC stored / fast | Max abs OOF score diff | EER / HTER stored | EER / HTER fast | Pass |
|---|---|---|---|---|---|---|---|---|
| P4_svm_1000 | 0.776201 | 0.776201 | 0.0e+00 | 0.776201 / 0.776201 | 0.00e+00 | 0.2783 / 0.2762 | 0.2783 / 0.2762 | yes |
| P4_lr_30 | 0.773964 | 0.773964 | 0.0e+00 | 0.773964 / 0.773964 | 0.00e+00 | 0.2696 / 0.2678 | 0.2696 / 0.2678 | yes |
| ALLz_lr_30 | 0.781130 | 0.781130 | 0.0e+00 | 0.781130 / 0.781130 | 1.16e-12 | 0.2657 / 0.2621 | 0.2657 / 0.2621 | yes |
| P4_lr_1000 | 0.779161 | 0.779161 | 0.0e+00 | 0.779161 / 0.779161 | 0.00e+00 | 0.2748 / 0.2736 | 0.2748 / 0.2736 | yes |
| ALLz_lr_1000 | 0.789460 | 0.789460 | 0.0e+00 | 0.789460 / 0.789460 | 1.20e-11 | 0.2636 / 0.2599 | 0.2636 / 0.2599 | yes |

## 2. Celniak pipeline (cached raw Tobii samples)

Original = `edd.celniak.slide_features` (the paper's AOI metrics, honest-baseline SVM input) and the track's lean global-gaze extractor (`phase2_celniak_efficiency.lean_global`). Fast = `slide_features_fast` / `global_gaze_fast` on NumPy columns of the same samples.

Recording Recording10: 101,546 samples at 250 Hz (406 s), 21 slides. The decision unit is the participant (21 slides); per-slide values are the per-participant value divided by 21 (the original extractor processes the whole recording, including the cross-screen pupil baseline, so a standalone per-slide original call is not defined).

### 2a. Latency (ms, p50 / p95)

| Model | Stage | Original | Fast | Speed-up (p50) |
|---|---|---|---|---|
| baseline_svm_paper | feature extraction only | 2482.29 / 2536.06 | 26.620 / 28.700 | 93.2x |
| baseline_svm_paper | inference only (same model code) | 6.693 / 7.594 | 6.693 / 7.594 | 1.0x |
| baseline_svm_paper | **end to end, per participant** | 2439.66 / 2501.77 | 33.941 / 37.066 | **71.9x** |
| baseline_svm_paper | end to end, per slide (amortised, p50) | 116.17 | 1.616 | |
| global_gaze_lr | feature extraction only | 1839.04 / 1871.05 | 19.937 / 21.407 | 92.2x |
| global_gaze_lr | inference only (same model code) | 2.455 / 2.812 | 2.455 / 2.812 | 1.0x |
| global_gaze_lr | **end to end, per participant** | 1900.24 / 1952.54 | 22.649 / 24.611 | **83.9x** |
| global_gaze_lr | end to end, per slide (amortised, p50) | 90.49 | 1.079 | |

### 2b. Throughput, real-time factor, memory

| Model | Participants/s orig -> fast | Samples/s orig -> fast | Real-time factor orig -> fast | Peak memory orig -> fast | Model size | Per-round e2e speed-up (min to max) |
|---|---|---|---|---|---|---|
| baseline_svm_paper | 0.410 -> 29.5 | 41,623 -> 2,991,795 | 166x -> 11,967x | 87.6 MB -> 4.5 MB | 125,858 B | 68.3x to 85.8x |
| global_gaze_lr | 0.526 -> 44.2 | 53,438 -> 4,483,515 | 214x -> 17,934x | 4.3 MB -> 6.1 MB | 1,865 B | 61.4x to 84.6x |

Fast path input: NumPy columns built once from the cached DataFrame (ingestion = data loading, 119.2 / 132.6 ms p50 / p95, not included above; the original's DataFrame is likewise built beforehand). The raw arrays held by the fast path occupy 14.9 MB, which the tracemalloc peak does not count. Max abs decision-function difference original vs fast: SVM 8.75e-14, global-gaze LR 4.44e-16. Rounds: 5.

### 2c. Equivalence of the features (all 39 recordings)

| Comparison | Max abs diff | NaN pattern equal | allclose | Values |
|---|---|---|---|---|
| slide_all_62_columns_vs_live_original | 4.55e-13 | yes | yes | 50,778 |
| slide_paper19_vs_live_original | 1.38e-14 | yes | yes | 15,561 |
| global_gaze7_vs_live_lean_original | 1.82e-12 | yes | yes | 273 |
| slide_paper19_vs_stored_table | 1.38e-14 | yes | yes | 15,561 |
| slide_all_62_columns_vs_stored_table | 9.09e-13 | yes | yes | 50,778 |
| global_gaze7_vs_stored_participant_table | 2.73e-12 | yes | yes | 273 |

Sensor rates present: [120.0, 250.0] Hz; 21 slides per recording.

### 2d. Out-of-fold participant AUC with the fast features (frozen folds, 3 repeats, nested tuning, n_jobs = 2)

| Model | AUC per repeat, stored | AUC per repeat, fast | Mean stored / fast | Max abs OOF score diff | Same chosen params | Pass |
|---|---|---|---|---|---|---|
| baseline_svm_paper | 0.623684, 0.571053, 0.597368 | 0.623684, 0.571053, 0.597368 | 0.597368 / 0.597368 | 1.97e-11 | yes | yes |
| global_gaze_lr | 0.771053, 0.755263, 0.768421 | 0.771053, 0.755263, 0.768421 | 0.764912 / 0.764912 | 1.30e-12 | yes | yes |

## 3. Caveats

- Timings were taken while other jobs used the CPU (a very-low-priority phase1_celniak.py job, plus whatever else was running), so every absolute time is an upper bound; original and fast were measured back to back, interleaved, in the same session.
- Original and fast share the same fitted model and the same sklearn inference code, so inference-only is identical by construction.
- Equivalence is to floating-point summation order (about 1e-12); event decisions (fixations, visits, first looks, counts) are identical.
- Delmas per-participant timing is for one participant (Sj_04) with all 64 trials; Celniak for one 250 Hz recording (Recording10). Rates below the sensor rate are simulated from a research tracker by decimation, as in the track.
- EDF decoding (Delmas) and CSV parsing (Celniak) are excluded from both pipelines; the fast path's DataFrame-to-array ingestion is reported separately.
- Peak memory is the Python-tracked allocation (tracemalloc), not process RSS.
