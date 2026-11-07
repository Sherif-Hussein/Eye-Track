# Task 2.2c Part B: logistic regression, XGBoost and the 50/50 ensemble on the observed-webcam data (card)

**Status: EXPLORATORY, development data only (Delmas et al.), simulated webcam error. Not preregistered.** Rules fixed 2026-11-07 before any run (`docs/task_observed_sensitivity_and_models.md`). Plan: `PLAN.md` (SHA-256 `ee13b402696cb7100e3d7e2fb2d867f0f0360e3295c1b60f28410df5a01a1d59`, written before the run; `PLAN.sha256.txt`). Output folder `results/phase10/model_compare_observed/`. Engine of Task M2 (`results/phase10/model_search/`, `results/phase10/model_compare/mc_lib.py`, `mc_run.py`, `mc_ens.py`, `mc_eval.py`) imported unchanged; every write redirected here. No camera, numbers only, held-out data not read. Wording: known item / unknown item. The ensemble was looked at earlier on other noise (Task M2), so it stays exploratory here.

## Conditions and models
Both at 30 Hz, dwell from samples, 10 features (no pupil), no per-trial correction, noise draws 1-3, item AUC = mean over draws.
- OBS = the 2.2b observed-webcam features (`results/phase9/low_rate_observed/features/`).
- OBS_R = the Part A reduced-slow-jitter features (`results/phase9/low_rate_observed/sensitivity/features/`), run after Part A had finished.
Models: logistic regression (reference), XGBoost (M2 grid, tuning inside training participants, one thread, seed = repeat, repeat 0), 50/50 ensemble (z-scale from each model's own inner-fold training scores, no retraining). Nothing else was run.

## Reproduction (done first; zero difference)
The logistic regression through this engine at 3 repeats reproduces the stored 2.2b pooled AUC of OBS for each draw (same definition, `lr_common.pooled_auc`), limit 0.0005:

| Draw | Pooled AUC here | Stored 2.2b (`sbnp_obsnone_30Hz_d<draw>_oof.csv.gz`) | Difference | Max score difference |
|---|---|---|---|---|
| 1 | 0.720100 | 0.720100 | 0.0 | 9e-16 |
| 2 | 0.732093 | 0.732093 | 0.0 | 9e-16 |
| 3 | 0.731110 | 0.731110 | 0.0 | 9e-16 |

Mean over draws 0.7278 (2.2b: 0.7278). For OBS_R the same check against the Part A files (extra): 0.754068, 0.737340, 0.753024, mean 0.7481, difference 0.0, maximum score difference 2e-15 (`repro_OBS.json`, `repro_OBS_R.json`). All primary numbers below use repeat 0 (as in M2), so the logistic regression's item AUC (OBS 0.7237, OBS_R 0.7462) differs slightly from the 3-repeat values.

## Results
Item AUC: pooled per group set, repeat-averaged scores, mean over six sets, mean over three draws. Paired participant-cluster bootstrap (2000, seed 20261106), paired score swap (2000, seed 20261107, two-sided). Holm over the 4 rows (XGBoost and ensemble x OBS and OBS_R), item level and person level separately. Verdict: better = interval above zero and Holm p < .05; worse = interval below zero and Holm p < .05; otherwise no reliable difference. EER = equal error rate; HTER = half total error rate at the threshold set on the training participants for 7% false alarms (realised false-alarm rates 0.073-0.077 at item level and 0.056-0.085 at person level). Person level = Delmas known-versus-unknown person statistic.

**Item level**

| Condition | Model | Item AUC [95% CI] | Change from LR [95% CI] | Swap p (raw) | Holm p | Verdict | EER | HTER (7% threshold) |
|---|---|---|---|---|---|---|---|---|
| OBS | Logistic regression (reference) | 0.7237 [0.6919, 0.7561] | reference | -- | -- | reference | 0.319 | 0.370 |
| OBS | XGBoost | 0.7120 [0.6834, 0.7415] | -0.0116 [-0.0225, +0.0004] | 0.0645 | 0.1514 | no reliable difference | 0.335 | 0.368 |
| OBS | Ensemble (LR + XGBoost) | 0.7294 [0.6990, 0.7605] | +0.0057 [+0.0003, +0.0109] | 0.0505 | 0.1514 | no reliable difference | 0.323 | 0.358 |
| OBS_R | Logistic regression (reference) | 0.7462 [0.7109, 0.7814] | reference | -- | -- | reference | 0.299 | 0.344 |
| OBS_R | XGBoost | 0.7352 [0.7052, 0.7662] | -0.0109 [-0.0231, +0.0025] | 0.1254 | 0.1514 | no reliable difference | 0.310 | 0.341 |
| OBS_R | Ensemble (LR + XGBoost) | 0.7578 [0.7235, 0.7912] | +0.0116 [+0.0059, +0.0175] | 0.0005 | 0.0020 | better | 0.295 | 0.338 |

**Person level**

| Condition | Model | Person AUC [95% CI] | Change from LR [95% CI] | Swap p (raw) | Holm p | Verdict | EER | HTER (7% threshold) |
|---|---|---|---|---|---|---|---|---|
| OBS | Logistic regression (reference) | 0.8385 [0.7899, 0.8812] | reference | -- | -- | reference | 0.183 | 0.241 |
| OBS | XGBoost | 0.8025 [0.7526, 0.8464] | -0.0360 [-0.0664, -0.0044] | 0.0220 | 0.0880 | no reliable difference | 0.241 | 0.257 |
| OBS | Ensemble (LR + XGBoost) | 0.8249 [0.7751, 0.8682] | -0.0136 [-0.0324, +0.0037] | 0.0790 | 0.2369 | no reliable difference | 0.217 | 0.250 |
| OBS_R | Logistic regression (reference) | 0.8607 [0.8099, 0.9057] | reference | -- | -- | reference | 0.176 | 0.213 |
| OBS_R | XGBoost | 0.8469 [0.7958, 0.8933] | -0.0138 [-0.0398, +0.0151] | 0.3068 | 0.6137 | no reliable difference | 0.196 | 0.226 |
| OBS_R | Ensemble (LR + XGBoost) | 0.8620 [0.8128, 0.9082] | +0.0012 [-0.0133, +0.0160] | 0.8586 | 0.8586 | no reliable difference | 0.183 | 0.202 |

**Extra (labelled): ensemble minus XGBoost, paired interval only**

| Condition | Item: ensemble minus XGBoost [95% CI] | Person: ensemble minus XGBoost [95% CI] |
|---|---|---|
| OBS | +0.0174 [+0.0085, +0.0250] | +0.0225 [+0.0030, +0.0417] |
| OBS_R | +0.0225 [+0.0122, +0.0318] | +0.0151 [-0.0047, +0.0328] |

`results_table.csv` has every number above plus the detection and false-alarm rates; `results.json` the rows, the reproduction and the app reading; `table_observed.tex` the LaTeX table; `results_item_person.csv/json` the raw output of `mc_eval.py`.

## Pre-set reading for the app
Rule (fixed before the run): "worth considering for the app later" needs an ensemble person-level interval above zero with person-level Holm p < .05 in both OBS and OBS_R.
- OBS: ensemble minus logistic regression at person level -0.0136 [-0.0324, +0.0037], Holm p 0.237: not met.
- OBS_R: +0.0012 [-0.0133, +0.0160], Holm p 0.859: not met.

**Result: the rule is not met, so logistic regression stays the only live model.** At item level the ensemble is better than the logistic regression in OBS_R (+0.0116 [+0.0059, +0.0175], Holm p 0.002) but not in OBS (+0.0057 [+0.0003, +0.0109], Holm p 0.151); XGBoost alone is never reliably different from the logistic regression (item or person). The ensemble beats XGBoost alone in both conditions at item level (interval above zero, extra, unadjusted). The ensemble result is exploratory: it was examined earlier on other noise.

## Plain summary (5 lines)
1. Reproduction is exact: the logistic regression through the engine gives the stored 2.2b observed-webcam AUC for all three draws (difference 0.0).
2. Item level: XGBoost is not reliably different from the logistic regression in either condition (-0.012 [-0.022, +0.000] in OBS, -0.011 [-0.023, +0.003] in OBS_R; Holm p 0.151).
3. The ensemble is better than the logistic regression in OBS_R at item level (+0.012 [+0.006, +0.017], Holm p 0.002) but not in OBS (+0.006 [+0.000, +0.011], Holm p 0.151).
4. Person level: no reliable difference for either model in either condition (ensemble -0.014 [-0.032, +0.004] in OBS, +0.001 [-0.013, +0.016] in OBS_R), so the pre-set rule for the app is not met and logistic regression stays the only live model.
5. Exploratory: simulated observed-webcam error from one person and one camera, repeat 0 only, ensemble examined earlier on other noise.

## Deviations, assumptions and caveats
- `ObsCond` (in `mco_lib.py`) subclasses the engine's `DelmasCond` to read the stored feature files; the engine's time-course and scan-path blocks are filled with zeros because the sample grids are not stored. Logistic regression, XGBoost and the ensemble read only the `base` block, and the exact reproduction shows the base features and folds are identical.
- The reproduction stage uses 3 repeats; XGBoost and the ensemble use repeat 0 only (as M2, which cut repeats for run time). With one repeat the intervals include no repeat-to-repeat variation of the XGBoost fits.
- OBS_R was run after Part A finished; its extra reproduction check against the Part A files passed.
- OBS and OBS_R share the participants, so their results are not independent; the Holm family treats the 4 rows jointly.
- Item AUC point differences of about 0.01 are small against the intervals; the item-level "better" for the ensemble in OBS_R is not matched at person level.
- All caveats of 2.2b and Part A apply: simulated frame rates and error, one person and one camera for the error parameters, Delmas viewers are other people.
- Cost is not measured again (it is in M2). Two parts of this task ran at the same time on the machine, plus one unrelated job of the coordinator.
- `tables_b.md` is the table text used in this card.

## Files
`PLAN.md`, `PLAN.sha256.txt`, `mco_lib.py`, `mco_run.py`, `mco_eval.py`, `mco_report.py`, `ck/` (fold records per tag, model, set), `repro_OBS.json`, `repro_OBS_R.json`, `results_item_person.csv/json`, `results_table.csv`, `results.json`, `table_observed.tex`, `tables_b.md`, logs `log_obs_lr.txt`, `log_obs_xgb.txt`, `log_obsr.txt`, `log_eval.txt`.
