# Phase 7, Track D (sensing robustness): pre-registration

Written 2026-11-02 before any no-pupil or shrunken-area result was computed on real labels. Later changes are logged in `DEVIATIONS.md`. Plan: `docs/phase7_plan.md` (Track D), source: `docs/review_7_q1_editor.md` M3. Code: `scripts/phase7_trackD_*.py` (it imports the Track C harness `scripts/phase6_trackC_c1.py` unchanged: sample-based features, the noise model that is bit-identical to `edd.webcam_noise.apply_noise`, table building, the frozen-fold nested CV `run_grid`). Checkpoints (derived scores of CC BY-NC-ND data) are kept outside `results/`.

## Common set-up
- Data: the decimated 30 Hz gaze cache of the webcam study (75 participants), frozen folds `results/phase2/delmas_raw/folds.json` (all six group sets), model ALL_z LR (median imputer, scaler, balanced logistic regression, C in {0.01, 0.1, 1, 10}, 4-fold inner participant-grouped CV, AUC scoring), per-participant z-scoring within set, per-image means over presentations, exactly as Track C.
- Noise levels (full noise = offset + jitter + 10% sample loss): L = 0 (clean), 1, 1.5, 2, 3, 4 deg, draws 1, 2, 3 for L > 0 (the pre-registered `rng([20261102, draw, participant])` draws); anchored level (Kaduk et al. 2024: offset Rayleigh mean 1.45 deg, jitter SD 1.174 / 1.004 deg per axis, loss 3%), draws 1, 2, 3. A level's AUC is the mean over its draws.
- Two feature routes: I-VT (event-based, `edd.delmas_fast.trial_features_batch`, 30 deg/s) and sample-based dwell (Track C `sb_features_batch`, no event detection).
- Feature sets. With pupil = the Track C models: I-VT 13 features (the full ALL_z set without `msacc_rate`, which does not exist at 30 Hz), sample-based 12 features (`n_fix` dropped). No pupil = the same models minus `pupil_face` and `pupil_trial`: I-VT 11 features, sample-based 10 features. The with-pupil results at full-size AOI are NOT recomputed; the stored Track C / webcam-study out-of-fold scores are reused (`ALLz_fixed_lr_30_*` and `c1_ck` scratch files).
- Metric: image-level AUC pooled within each of the six sets on repeat-averaged out-of-fold scores, mean over sets (the non-inferiority definition). Reference: P4 SVM at 1000 Hz, AUC 0.7811.
- Bootstrap: paired participant-cluster bootstrap, 2000 resamples, seed 0, participants resampled within each set, identical resamples for the reference and every candidate (the procedure of Track C `stage_stats`; the resample sequence does not depend on the number of columns). 95% percentile intervals.
- Harness check (before any new result is read): the new code path with full-size AOI and with pupil must reproduce Track C's clean sample-based AUC 0.7723 and clean I-VT AUC 0.7854 (OOF scores equal to the stored files to 1e-9); the AOI-patching code at scale 1.0 must change nothing; the vectorised features at scales 0.75 and 0.5 must match the unvectorised reference implementations (`sb_reference_trial`, `edd.delmas_raw.trial_features`) on a subset of trials.

## D1. Webcam simulation without pupil features
Conditions: {I-VT, sample-based} x {clean, 1, 1.5, 2, 3, 4 deg full noise, anchored} = 14 no-pupil configurations (full-size AOI).
Reported per configuration: AUC [95% CI], Delta = AUC - 0.7811 with its paired-bootstrap 95% CI, raw one-sided non-inferiority p = (1 + #{Delta <= -0.02}) / 2001 (margin 0.02), and Holm-adjusted p.
- **D1 decision family (confirmatory in form, exploratory in status as the plan labels the webcam result exploratory): the 14 no-pupil configurations; Holm over 14.** A Holm p within each route (7 tests) is also reported. Non-inferior when Holm p < 0.05.
- Pupil contribution: at each level and route, AUC(with pupil) - AUC(no pupil) with a paired-bootstrap 95% CI (no test, no margin). With pupil = stored Track C results at the same level and draws.
- Also reported: Delta of each no-pupil configuration vs its own clean no-pupil result, and sample-based minus I-VT (no-pupil) at each level, with paired-bootstrap CIs.

## D2. Face-area (AOI) size sensitivity
Each face rectangle is shrunk to 75% and 50% of its width and height around its centre (full: 300 x 400 px = 8.57 x 11.43 deg at 34.987 px/deg; 75%: 225 x 300 px = 6.43 x 8.57 deg; 50%: 150 x 200 px = 4.29 x 5.71 deg; centres unchanged, rectangles inclusive as in the pipeline, floating-point bounds). The shrunken rectangle replaces the face area in every AOI-dependent feature (dwell, visits, latency, first look, `pupil_face`, fixation-centroid assignment in I-VT; sample position in the sample-based route). The I-VT fixation detection and the pupil pre-processing are unchanged. No other parameter changes.
Conditions: scales {0.75, 0.5} x {clean 30 Hz, anchored (draws 1-3)} x {I-VT, sample-based} x {with pupil, no pupil} = 16 configurations per scale. Full-size comparators (scale 1.0): with pupil = Track C stored results; no pupil = D1.
Reported per configuration: AUC [95% CI]; Delta vs the full-size AOI of the same configuration (same route, same pupil setting, same condition and draws) with a paired-bootstrap 95% CI (the primary D2 quantity); Delta vs 0.7811 with CI and raw one-sided non-inferiority p at margin 0.02 (descriptive; D2 has no decision family and no multiplicity claim). The AOI size in degrees is given for every scale.
No offset-by-AOI-size grid is run (not requested by the plan).

## Outputs
`results/phase7/trackD/`: `results.json`, `card.md`, `d1_table.csv`, `d1_pupil_contribution.csv`, `d2_table.csv`, `checks.json`, `DEVIATIONS.md`.
