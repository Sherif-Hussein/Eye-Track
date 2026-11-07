# RESUME: Phase 2, celniak track

## STATUS: COMPLETE (2026-11-02). Nothing is running. Nothing is left.

All remaining items are finished, saved, and merged into `results.json` (meta status "final"); `comparison_card.md` is final (INTERIM removed). Headline numbers:

1. Within-order permutation, global_gaze_lr: observed AUC 0.771, null mean 0.470, p95 0.635, p = 0.020 (100 shuffles). File `stage_perm_global_gaze_lr_permutation_within_order.json`.
2. orient_lr permutation: observed 0.629, null mean 0.507, p95 0.646, p = 0.098 (50 shuffles). File `stage_perm_orient_lr_permutation.json`.
3. Confound controls (global_gaze_lr = 0.765): + validation-error covariate 0.798 (no gain over validation alone, 0.799); residualised on validation error 0.717 (82% of the above-chance AUC kept); in-fold quality exclusion 0.765 on all test participants and 0.816 on the retained high-quality ones; both together 0.731 (0.788 on retained); Recording1 (120 Hz) removed 0.735; all recordings decimated to about 125 Hz 0.764. Files `stage_confounds.json`, `confound_table.md`, `trials_conf_*.csv`.
4. Non-nested reproduction (GroupKFold(20), 216-point grid, tuned on the scored folds): best accuracy 0.641, max AUC 0.632; majority-vote aggregation 0.667 / 0.718; leaky ANOVA selection the same; leaky random slide split 0.74 to 0.80 accuracy. The published 0.95 / 1.00 is not reproduced by any variant. Files `stage_reproduction.json`, `stage_reproduction_leak.json`, `reproduction_*.csv`.
5. Efficiency table: `stage_efficiency.json` (card section 6).
6. `comparison_card.md` final; `results.json` final. Backups: `results_checkpoint_2340.json` (state at handover), `results_final_backup.json` (final copy), `results_before_finalize_*.json` (pre-merge copies).

Scripts written in this takeover (in `scripts/`): `phase2_celniak_perm.py`, `phase2_celniak_confounds.py`, `phase2_celniak_confound_summary.py`, `phase2_celniak_repro.py`, `phase2_celniak_repro_leak.py`, `phase2_celniak_efficiency.py`, `phase2_celniak_finalize.py`; library `src/edd/celniak_confounds.py`. Rerun order if ever needed (repo root, `set OMP_NUM_THREADS=1`, `.venv\Scripts\python.exe -I -W ignore scripts\<name>.py --jobs 2`): perm, confounds, repro, efficiency, repro_leak, confound_summary, finalize. Each skips outputs that exist; delete a stage file to recompute it. Do not use `tasklist` on this machine (it hangs); use `Get-CimInstance Win32_Process`.

Limits to remember: permutation tests use repeat-0 folds only; the "± SD over repeats" is small because repeats only re-pair the same 39 people (use the bootstrap CIs); the decimation and exclusion variants are single-sample sensitivity checks; efficiency was measured while another job used the CPU; our reproduction features are rebuilt from the public export. An earlier version of this file was accidentally truncated while editing and was rebuilt from the handover text, so the old-handover section below is a faithful copy of what was read, not the original file.

## Pitfalls and decisions (carried over from the handover)

- **Keep a copy of `results.json` before rerunning `phase2_celniak_screen.py`.** That script rewrites results.json at several checkpoints and an interruption loses the stored diagnostic, paired-test and permutation sections. Restore from `results_checkpoint_2340.json` plus `phase2_celniak_finalize.py`. `results_partial_resume_0002.json` is an incomplete write from the stopped handover run; ignore it.
- **Sampling rate:** 38 of 39 recordings are 250 Hz and only Recording1 is 120 Hz. All extraction is time-based. Efficiency uses Recording10.
- **Stimulus segmentation:** event rows without a stimulus name interleave inside stimuli in some recordings; `stimulus_runs` merges same-name runs, giving 819 trials (an earlier version gave 837).
- **Folds:** 1 guilty + 1 innocent per fold, matched on test order where possible, seeds 2026 to 2028 (`folds.json`). Not the Phase 1 folds.
- **Budget choices:** inner CV uses 3 grouped splits. The LR grid is k in {3, 10, all} x C in {0.1, 1}; the SVM grid is reduced to 36 points. Screening CIs use 300 bootstrap draws; baseline, best and references use 2000.
- **Permutation:** `edd.harness.permutation_test` uses n_jobs=-1, which breaks the CPU limit; the screening script uses its own `participant_permutation`, and the takeover `phase2_celniak_perm.py` uses a per-shuffle seed (1000 + i) with a checkpoint after every shuffle.
- **Labels:** global_gaze and the two fixdur ablations are probe-agnostic state measures, not CIT detectors. The ablations were defined after the full-sample univariate table was inspected, so they are kept out of best-candidate selection (kind "ablation").
- **Strongest confound:** Tobii post-calibration validation error (metadata, before the CIT) predicts guilt at AUC 0.80 to 0.82 (0.84 inside each order stratum). `mechanism_checks.json` (`scripts/phase2_celniak_mechanism.py`).
- **CMPB 2026 code** was read as text only, never run: hyperparameters were selected on the reported outer folds (not nested).
- **Old files:** `interim_1repeat/` is an older 1-repeat run with a larger LR grid; reference only.
- **Scripts from the first agent:** `phase2_celniak_features.py` (raw to `data/derived/celniak_p2_*`, about 3 min with 2 jobs), `phase2_celniak_screen.py` (screening, order diagnostic, bootstrap and Holm, permutation, an older reproduction/efficiency stage superseded by the takeover scripts), `phase2_celniak_tables.py`, `phase2_celniak_mechanism.py`.

## Screening headline numbers (first agent; participant level, mean +/- SD over 3 repeats; "Strat" = AUC within test-order strata)

| Setting | AUC | Acc | EER | HTER | Strat AUC |
|---|---|---|---|---|---|
| order_only (test order alone) | 0.613 | 0.615 | 0.387 | 0.387 | 0.500 |
| validation_accuracy_lr (Tobii validation error, metadata) | 0.799 +/- 0.003 | 0.684 | 0.214 | 0.310 | 0.828 |
| baseline_svm_paper (honest baseline) | 0.597 +/- 0.021 (CI 0.41-0.79) | 0.564 | 0.445 | 0.436 | 0.559 |
| global_gaze_lr (best idea; probe-agnostic) | 0.765 +/- 0.007 (repeat-avg 0.776, CI 0.61-0.92) | 0.752 | 0.239 | 0.249 | 0.713 |
| global_gaze_order_resid_lr | 0.711 +/- 0.015 | 0.667 | 0.368 | 0.334 | 0.690 |
| fixdur_only_lr (post-hoc ablation) | 0.777 +/- 0.024 | 0.709 | 0.273 | 0.289 | 0.733 |
| global_gaze_no_fixdur_lr (post-hoc ablation) | 0.634 +/- 0.012 | 0.632 | 0.469 | 0.370 | 0.570 |
| minirocket_ridge | 0.748 +/- 0.011 | 0.709 | 0.368 | 0.293 | 0.695 |
| orient_lr (best probe-specific) | 0.616 +/- 0.010 | 0.538 | 0.454 | 0.463 | 0.591 |

Full table with all metrics: `comparison_card.md` section 4. global_gaze_lr vs the baseline: +0.168 AUC [-0.045, +0.400], p = 0.138, Holm p = 1.0 (no reliable difference; no idea survives Holm).
