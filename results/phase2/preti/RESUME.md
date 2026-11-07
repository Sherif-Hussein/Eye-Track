# Preti track: resume notes (handover)

**STATUS: COMPLETE (2026-11-02).** All work is finished and nothing is left to run. Permutations: eye-only p = 0.032 (30 draws), fixrate p = 0.032 (30), RT-only p = 0.048 (20, `scripts/phase2_preti_perm_rt.py`, `perm_rt_result.json`), fusion p = 0.048 (20). Efficiency: `efficiency.json`, `efficiency.log`. `comparison_card.md` is FINAL (sections 5-7 written, INTERIM removed). `results.json` now also holds the keys `final` (candidates, permutations, CIs) and `efficiency`; rerunning `scripts/phase2_preti.py` would overwrite them, so rerun `phase2_preti_final.py` and the efficiency script afterwards. `tables.md` is regenerated. Section "Left" and the resume commands below are historical.

Earlier note (2026-11-02 00:15): no preti processes were running; all were stopped for the handover.

## Handover 2 (2026-11-02, second agent)
- Started the permutations in the background: `final.py --n-perm 30 --perm-budget-s 1800` (log `final.log`; the default cap of 900 s would have stopped the eye-only run at about 19 draws because `seconds` is cumulative). A backup of the pre-permutation `final.json` is `final_pre_perm_backup.json`. Next: efficiency (run only after the permutations finish, so timings are not contended by my own job), then card sections 5-7, results.json, completion mark.

- Run 1 finished (log kept as `final_run1.log`, JSON as `final_run1_backup.json`): eye-only permutation 30/30, p = 0.032; fixrate 30/30, p = 0.032. Fusion got only 1 draw because the wall clock jumped (the machine was suspended; recorded `seconds` = 7219) and the cumulative 1800 s cap tripped. Re-running with `--perm-budget-s 100000` to finish fusion (20 draws). Ocular and fixrate are cached and skipped.

- Run 2 finished ("final done"): fusion permutation 20/20, observed AUC 0.971, null 0.464 ± 0.105, p95 0.637, p = 0.048. Next: efficiency (started, log efficiency.log), RT permutation (new small script `scripts/phase2_preti_perm_rt.py`, 20 draws, run after efficiency), tables, card sections 5-7.

## Done (all on disk)
- **Paper facts.** Full text is inaccessible (Wiley returns 403; no preprint found). Facts come from the abstract (Crossref and Semantic Scholar), Zenodo v1 and v2, and the data. Verified: "ML up to 84% accuracy; RT to Unexpected questions is the strongest predictor; ocular measures add limited value; no group difference in blinks or errors." Not reported in accessible sources: eye tracker, sampling rate, ML models, CV scheme. Field names point to an EyeLink at ≥ 1000 Hz (inferred).
- **Data audit** (`results.json` → `data_audit`; comparison card section 1):
  - `raw data` is fixation-level. `CONDITION` 0 = liar (positive class): 27 liars, 30 truth-tellers.
  - The pupil and blink aggregate sheets are offset by one subject id from raw/RT for ids ≥ 3.
  - Zenodo v1 (`data/preti/v1/`, 60 participants) maps exactly onto v2; v2 drops v1 ids 8, 13 and 49, all liars.
  - Subject 3's label disagrees in `RTs_aggregated`; v1 confirms the raw label. Dropping or flipping subject 3 changes AUC by ≤ 0.008.
- **Folds** (`folds.json`): paired participant folds, 3 repeats × 30.
- **Paper-style reproduction** (`scores/reproduction.json`; card section 2): non-nested 10 × 10-fold CV. Best accuracy is 0.90–0.92 (fusion or RT only) vs the published 0.84. Ocular-only accuracy is 0.54–0.57.
- **Screening**: 34 arms plus 3 diagnostic `D_*` arms, scores in `scores/<arm>.csv`. Metrics, all 12 per arm, are in `results.json` with Holm-corrected paired tests. Headlines are participant level, mean ± SD over 3 repeats.
  - **Eye only.** Baseline `B_ocular` AUC 0.399. Best `I3_trial_raw_aggtuned_ocular`: AUC 0.694 ± 0.004, accuracy 0.637, EER 0.416, HTER 0.374 (CI on averaged scores: AUC [0.547, 0.816]). **It is an RT proxy.** Its signal comes from blink occurrence per trial, which rises with trial length (within-person r = 0.41 with log RT). With blinks per second the AUC is 0.532; after regressing out mean log RT it is 0.519.
  - **Pitfall, flagged by the coordinator.** The highest eye-tracker arm, `I5_ocular_plus_fixcount` (AUC 0.792), uses fixation count. Fixation count is an RT proxy: trial r = 0.85–0.89, subject-level r = 0.82. It must not be presented as an eye-only result. `I5_ocular_plus_fixrate` (0.747) is also RT-dependent (ρ = 0.50 with mean log RT). With strict pupil and blink measures, no eye-only signal independent of RT exists. The best non-RT-proxy eye arm is about 0.56–0.63 and not reliable.
  - **RT only.** `B_rt` AUC 0.909. `I7_lgbm_I1_rt` AUC 0.969 ± 0.002, accuracy 0.936, EER 0.059, HTER 0.064. `I1_diff_rt` (0.959) and `I7_lr_en_I1_rt` (0.967) beat `B_rt` after Holm (p = .042 and .023). A single no-model statistic, mean RT to Unexpected questions, already gives AUC 0.960 / EER 0.072.
  - **Fusion.** `B_fusion` AUC 0.927, accuracy 0.877, EER 0.129, HTER 0.124. `I7_lr_en_I1_fusion` AUC 0.971 ± 0.002, accuracy 0.924, EER 0.076, HTER 0.076; Holm p = .43 vs `B_fusion`, so no reliable difference. Elastic net keeps 0 ocular features. Fusion minus RT-only is never reliable (ΔAUC between −0.016 and +0.014).
  - Failures: control-question normalisation (I2, I3-neutralz) collapses the RT signal (fusion 0.59–0.64), because liars are slower on all question types. Tuned aggregation equals the mean. Pupil preprocessing variants are all at chance (0.40–0.58).
- **`final.json`**: candidates, bootstrap CIs, subject-3 sensitivity, ocular validity, univariate AUCs, group means, LR weights.
- **Permutation, partial.** `scores/perm_ocular.json` holds 8 of 30 draws for `I3_trial_raw_aggtuned_ocular`: null mean 0.42 ± 0.11, null p95 0.53, observed 0.694, so p = 1/9 = 0.11 so far (8 draws are too few). This test only shows the score beats chance. The validity check above shows that its signal is RT.

## Left
1. Finish the permutations: ocular to 30 draws, then `I5_ocular_plus_fixrate` (30) and `I7_lr_en_I1_fusion` (20). The script resumes from `scores/perm_*.json`, and each candidate is capped at 900 s by `--perm-budget-s`.
2. Run the efficiency script (not run yet; no `efficiency.json`).
3. Fill comparison card sections 5 (permutation p), 6 (efficiency) and 7, then remove INTERIM. `scripts/phase2_preti_tables.py` renders every table from the JSON files.

## Resume commands
```
.venv\Scripts\python.exe -I -u scripts\phase2_preti_final.py data\preti\preti2026.xlsx results\phase2\preti --n-perm 30 > results\phase2\preti\final.log 2>&1
.venv\Scripts\python.exe -I -u scripts\phase2_preti_efficiency.py data\preti\preti2026.xlsx results\phase2\preti > results\phase2\preti\efficiency.log 2>&1
.venv\Scripts\python.exe -I scripts\phase2_preti_tables.py results\phase2\preti results\phase2\preti\tables.md
```

## Pitfalls
- The final script first recomputes CIs (~3 min, 2000 bootstraps per arm on a shared CPU). The `D_*` arms are cached.
- Do **not** rerun `scripts/phase2_preti.py` unless an arm changes. It overwrites `results.json`, dropping the `D_*` diagnostic arms that the final script adds. If you do rerun it, rerun the final script afterwards.
- The efficiency script reads its candidates from `final.json`. The ocular candidate is a trial-level arm and is timed with LR C = 0.1 and mean aggregation (approximate). Timings are taken on a shared CPU, so they are upper bounds.
- ECE of about 0.001 for the trial-level vote arms is an artefact: their probabilities cluster at the base rate.
- The random forest costs about 0.8 s per fit on this machine, so it appears only in `I7_lr_svm_rf_B_fusion`.
- The planned I6 arm (`ANSWER_LABEL`) was dropped unrun: the column is undocumented and looks like an export artefact.
