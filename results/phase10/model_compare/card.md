# Task M2: same-fold comparison of XGBoost, LightGBM, CatBoost, an RBF-kernel SVM and a logistic regression + XGBoost ensemble with the frozen logistic regression

EXPLORATORY. Development data only (Delmas, all six group sets). The held-out dataset was never read; the camera was never opened.
Fills the red marker in paper_rec/main.tex, section "Model and decision" (paper_rec was not edited).
Plan: PLAN.md (SHA-256 `b5bb0e30...f6cf3` at 2026-11-07 00:01:47, written before any run) + Addendum A inside PLAN.md (`6399b232...b78b5`, 00:06:57, written after the timing pilot and before any model result; the current PLAN.md still has this hash) + ADDENDUM_ensemble.md (ensemble and final table; added later at the user's request; PLAN.md unchanged). Hashes: PLAN.sha256.txt.

## 1. Reproduction of the reference (done first, 3 repeats, all six sets, through the unchanged model_search engine)
| Condition | Stored | Reproduced | Max abs. score difference |
|---|---|---|---|
| FULL (clean 30 Hz, 13 features) pooled item AUC | 0.7854480364 | 0.7854480364 | 1.8e-15 |
| CAMERA draw 1 / 2 / 3 | 0.77324 / 0.77214 / 0.77574 | 0.77324 / 0.77214 / 0.77574 | 1.8e-15 each |
| CAMERA mean | 0.7737 | 0.77371 | |
(repro.json). The run would have stopped on a mismatch; it did not.
With ONE repeat (repeat 0, used for all comparisons below, see deviations) the logistic regression has item AUC 0.7824 (FULL) and 0.7709 (CAMERA); the 0.7854 / 0.7737 above are the 3-repeat values.

## 2. What ran
- Conditions: FULL = Delmas clean 30 Hz, 13 standardised features. CAMERA = Delmas 30 Hz, dwell from samples, no pupil, 10 features, measured webcam noise ("anch"), mean over the same 3 noise draws. Frozen folds (5 participant-disjoint x repeat 0).
- Models and grids exactly as in PLAN.md (XGBoost, LightGBM, CatBoost 8 settings each; SVM 9 settings), tuned only inside the training participants with the reference's inner splitter (StratifiedGroupKFold, 4 folds, seed = repeat), mean inner-fold item AUC, first maximum wins. Score = raw margin (trees) or decision function (SVM).
- Ensemble (ADDENDUM_ensemble.md): 50/50 average of the logistic regression and XGBoost scores; within each outer fold each model's test and inner-fold scores are z-scaled with the mean and SD of that model's own inner-fold training scores (no test person influences the scale); the two models keep the settings chosen inside the training people; no new training (mc_ens.py).
- Statistics as ms_eval.py: item AUC pooled per group set, mean over six sets (CAMERA: and over 3 draws); person AUC with the Delmas CIT statistic; EER; HTER, detection and realised false-alarm rate at the 7% threshold set on the training participants' inner scores; 95% paired participant-cluster bootstrap of the difference from the logistic regression (2000 resamples, seed 20261106); paired score swap (2000 swaps, seed 20261107); Holm over all 10 model x condition rows (4 models + ensemble, x 2 conditions) on the item AUC; person results get their own Holm over the 10 person rows and are reported separately.
- Verdict rule (fixed): better = interval above zero AND Holm p < .05; worse = interval below zero AND Holm p < .05; otherwise "no reliable difference". No winner by point estimate.

## 3. Environment and route
- CatBoost 1.2.10 was not installed. `pip install --dry-run` into `.venv` (Python 3.14.6) showed a cp314 wheel and only three NEW packages (catboost 1.2.10, graphviz 0.21, plotly 7.1.0), nothing changed or upgraded. To leave `.venv` untouched I installed them with `pip install --no-deps --only-binary=:all: --target results/phase10/model_compare/site_pkgs`, and `mc_lib.py` appends that folder to `sys.path`. A separate Python 3.12 venv was therefore not needed and no .npz exchange was used. `.venv` and `app/.venv` were not changed.
- Versions (same interpreter for everything): Python 3.14.6, numpy 2.3.5, scipy 1.17.1, scikit-learn 1.9.1, pandas 3.0.6, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10 (graphviz 0.21, plotly 7.1.0).
- One process, one thread: OMP/MKL/OPENBLAS = 1; xgboost n_jobs=1; lightgbm n_jobs=1; catboost thread_count=1, allow_writing_files=False.
- Run time: about 52 min until the session stopped (during W30_d2) and 34 min after the restart from checkpoints (about 86 min of computing in all; below the 2 h limit of the plan). Logs: log_run.txt, log_resume_console.txt.

## 4. Results (item level; repeat 0; difference from logistic regression with 95% paired participant-cluster bootstrap interval; Holm over 10 rows)
FULL (clean 30 Hz, 13 features)
| Model | Item AUC [95% CI] | Difference [95% CI] | Swap p | Holm p | Verdict | EER | HTER (7% thr.) | Detection / false alarm at 7% thr. |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.782 [0.742, 0.822] | reference | | | reference | 0.262 | 0.306 | 0.463 / 0.075 |
| XGBoost | 0.770 [0.734, 0.804] | -0.013 [-0.029, +0.005] | .159 | 1.00 | no reliable difference | 0.278 | 0.297 | 0.478 / 0.073 |
| LightGBM | 0.766 [0.732, 0.801] | -0.017 [-0.035, +0.004] | .124 | 1.00 | no reliable difference | 0.284 | 0.296 | 0.477 / 0.070 |
| CatBoost | 0.776 [0.739, 0.814] | -0.006 [-0.022, +0.012] | .525 | 1.00 | no reliable difference | 0.278 | 0.285 | 0.496 / 0.067 |
| RBF-kernel SVM | 0.782 [0.745, 0.818] | -0.000 [-0.015, +0.015] | .993 | 1.00 | no reliable difference | 0.262 | 0.287 | 0.500 / 0.074 |
| Ensemble LR + XGBoost (not blind) | 0.790 [0.751, 0.826] | +0.008 [-0.000, +0.015] | .079 | 0.71 | no reliable difference | 0.267 | 0.288 | 0.497 / 0.073 |

CAMERA (webcam-level noise, 10 features, mean of 3 draws)
| Model | Item AUC [95% CI] | Difference [95% CI] | Swap p | Holm p | Verdict | EER | HTER (7% thr.) | Detection / false alarm at 7% thr. |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.771 [0.729, 0.813] | reference | | | reference | 0.272 | 0.308 | 0.459 / 0.074 |
| XGBoost | 0.778 [0.743, 0.814] | +0.007 [-0.009, +0.025] | .377 | 1.00 | no reliable difference | 0.267 | 0.291 | 0.489 / 0.072 |
| LightGBM | 0.767 [0.734, 0.801] | -0.004 [-0.020, +0.014] | .692 | 1.00 | no reliable difference | 0.277 | 0.299 | 0.472 / 0.070 |
| CatBoost | 0.781 [0.743, 0.818] | +0.010 [-0.005, +0.026] | .213 | 1.00 | no reliable difference | 0.265 | 0.285 | 0.499 / 0.070 |
| RBF-kernel SVM | 0.775 [0.735, 0.815] | +0.004 [-0.004, +0.014] | .370 | 1.00 | no reliable difference | 0.274 | 0.288 | 0.499 / 0.074 |
| Ensemble LR + XGBoost (not blind) | 0.789 [0.751, 0.827] | +0.018 [+0.009, +0.027] | .0005 | 0.005 | better | 0.261 | 0.286 | 0.499 / 0.072 |

Person level (Delmas CIT statistic; its own Holm over 10 rows; descriptive companion to the item result)
| Condition | Model | Person AUC [95% CI] | Difference [95% CI] | Holm p | Verdict | EER | HTER (7% thr.) | Detection / false alarm |
|---|---|---|---|---|---|---|---|---|
| FULL | Logistic regression | 0.888 [0.828, 0.937] | reference | | reference | 0.167 | 0.172 | 0.689 / 0.033 |
| FULL | XGBoost | 0.873 [0.814, 0.924] | -0.016 [-0.056, +0.020] | 1.00 | no reliable difference | 0.167 | 0.189 | 0.667 / 0.044 |
| FULL | LightGBM | 0.853 [0.791, 0.907] | -0.036 [-0.082, +0.007] | 0.99 | no reliable difference | 0.172 | 0.194 | 0.644 / 0.033 |
| FULL | CatBoost | 0.873 [0.819, 0.922] | -0.016 [-0.052, +0.024] | 1.00 | no reliable difference | 0.150 | 0.194 | 0.667 / 0.056 |
| FULL | RBF-kernel SVM | 0.855 [0.787, 0.908] | -0.033 [-0.072, +0.002] | 0.58 | no reliable difference | 0.206 | 0.172 | 0.700 / 0.044 |
| FULL | Ensemble | 0.887 [0.824, 0.936] | -0.002 [-0.024, +0.019] | 1.00 | no reliable difference | 0.156 | 0.167 | 0.700 / 0.033 |
| CAMERA | Logistic regression | 0.874 [0.814, 0.926] | reference | | reference | 0.146 | 0.167 | 0.696 / 0.030 |
| CAMERA | XGBoost | 0.873 [0.818, 0.920] | -0.001 [-0.031, +0.033] | 1.00 | no reliable difference | 0.139 | 0.180 | 0.693 / 0.052 |
| CAMERA | LightGBM | 0.856 [0.801, 0.902] | -0.018 [-0.053, +0.016] | 1.00 | no reliable difference | 0.172 | 0.183 | 0.678 / 0.044 |
| CAMERA | CatBoost | 0.863 [0.808, 0.912] | -0.011 [-0.040, +0.020] | 1.00 | no reliable difference | 0.163 | 0.167 | 0.715 / 0.048 |
| CAMERA | RBF-kernel SVM | 0.856 [0.794, 0.907] | -0.018 [-0.037, -0.002] | 0.22 | no reliable difference (interval below zero, Holm p not < .05) | 0.194 | 0.152 | 0.726 / 0.030 |
| CAMERA | Ensemble | 0.876 [0.818, 0.926] | +0.002 [-0.013, +0.021] | 1.00 | no reliable difference | 0.139 | 0.156 | 0.719 / 0.030 |

Labelled extra (paired interval only, not tested, not in the Holm family): ensemble minus XGBoost alone, item AUC: FULL +0.020 [+0.006, +0.032]; CAMERA +0.011 [-0.001, +0.020]. Person AUC: FULL +0.014 [-0.009, +0.040]; CAMERA +0.003 [-0.019, +0.024].
The 7% threshold is set on the training participants; the realised false-alarm rates (0.067-0.075 at item level) agree with it.

## 5. Cost (laptop Intel Core i7-8550U, one thread; FULL features, 13; ensemble = logistic regression + XGBoost together)
Settings used for the cost: the most often chosen setting over the 30 outer folds (LR C=0.01; XGBoost 100 trees, depth 2, rate 0.03; LightGBM 100 trees, 4 leaves, 0.03; CatBoost 100 iterations, depth 4, 0.03; SVM C=1, gamma 0.01). One person has about 62 items (the three test people of fold 0, set 64_g1, have 61, 62 and 62 items).
| Model | Size (pickle) | Params / nodes | Score one person, CPU ms (wall ms) | Score 4 items of a display, CPU ms | Fit once, CPU ms (wall ms) | Peak memory growth, MiB | Core share, 4 items per 5 s display | Core share, whole person per display | Work per item |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 1,823 B | 53 | 0.71 (0.72) | 0.70 | 6.6 (6.5) | 1.2 | 0.014 % | 0.014 % | 52 arithmetic operations |
| XGBoost | 100,563 B | 698 nodes | 1.87 (1.87) | 1.77 | 156 (157) | 10.2 | 0.035 % | 0.037 % | 200 nodes visited (100 trees x 2.0) |
| LightGBM | 63,992 B | 700 nodes | 1.02 (1.02) | 0.85 | 26 (25) | 8.4 | 0.017 % | 0.020 % | 220 nodes visited (100 x 2.2) |
| CatBoost | 42,546 B | 2,000 numbers | 1.38 (1.30) | 1.36 | 250 (213) | 19.1 | 0.027 % | 0.028 % | 400 (100 trees x depth 4) |
| RBF-kernel SVM | 32,995 B | 3,498 numbers (248 SV) | 2.18 (2.13) | 0.91 | 10.9 (11.0) | 1.9 | 0.018 % | 0.044 % | 3,224 (248 SV x 13 features) |
| Ensemble | 101,847 B | 751 | 2.91 (2.93) | 2.85 | 152 (155) | 10.5 | 0.057 % | 0.058 % | 258 (52 + 200 + 6) |

How each number was computed (mc_cost.py; cost.csv, cost.json):
- Size = length of the Python pickle of the fitted pipeline (imputer, scaler, model); the ensemble is the pickle of the pair.
- CPU = `time.process_time`, wall = `time.perf_counter`. process_time has a 15.6 ms resolution on Windows, so each measurement is a batch of calls lasting at least 0.25 s (scoring) or 0.5 s (fit), divided by the number of calls (fit resolution about +-4 ms). Scoring: predict from precomputed features for all items of one person and, separately, for the 4 items of one display; 3 people x 6 rounds (the plan said 5 people; fold 0 of set 64_g1 has only 3 test people, deviation 9), models alternated within every round, median. Fit: one fit on the training people of fold 0, repeat 0, set 64_g1 (709 rows), 3 alternated rounds, median.
- Peak memory = fresh child process per model (fit + one scoring call): growth of the Windows peak working set after imports and data loading (baseline about 159 MiB for the Python process with the libraries imported); a lower bound for native allocations. Python-level tracemalloc peaks are in cost.csv.
- Core share = CPU time per display / 5 s display length. "4 items" = scoring the items of the one display just shown; "whole person" = rescoring all about 62 items once per display (conservative). Features are assumed precomputed; the cost of computing them is not included.
- Work per item: logistic regression = 13 subtractions + 13 divisions (standardise) + 13 multiplications + 13 additions = 52; SVM = support vectors x features (each term is about 3 raw operations); boosted trees = split nodes visited per item = trees x mean leaf depth, measured from the leaves the person's items reach (XGBoost, LightGBM; CatBoost oblivious trees: iterations x depth); ensemble = sum + 6 operations for the two z-scalings and the average (mixed units). With the 10 camera features the logistic regression needs 40 operations.
- Other load during the measurement (cost.json, "other_load", psutil over 3 s): before the measurement the machine was at about 3% CPU in total (VS Code, this assistant, a printer monitor; the one light statistics job running on the laptop was not among the processes using more than 0.5% of a core at that moment). After the measurement, background Windows maintenance (cleanmgr.exe about 45% of a core, TiWorker.exe 17%) was running, 26% of all cores. Scoring times were stable across the six rounds (spread under 6%); fit times vary between repeats of the measurement (a first run gave XGBoost 131 ms, CatBoost 184 ms, LightGBM 21 ms) and should be read as about +-30%.
- The cost measurement was run twice: the first run exposed that CatBoost's `predict` used all cores by default (CPU 10.8 ms vs wall 2.3 ms); after fixing it to `thread_count=1` the final numbers above come from the second run. Only the second run's numbers are reported; scores are identical (checked, mc_check_cat_threads.py).

## 6. Plain summary
- No tree model, no SVM and no ensemble member differs reliably from the logistic regression on new people: every difference of a single model lies between -0.017 and +0.010 item AUC, every interval includes zero, and every Holm-adjusted p is above .05 (item level, both conditions; person level too). XGBoost, LightGBM and CatBoost are numerically a little lower on the clean 30 Hz model and a little higher (XGBoost, CatBoost) or equal (LightGBM) on the camera model, but none of this is reliable. The SVM is equal on the full model (-0.000) and +0.004 on the camera model.
- The 50/50 ensemble of logistic regression and XGBoost has the highest AUC in both conditions. On the camera model the gain over logistic regression is +0.018 [+0.009, +0.027], Holm p = .005 (item level), which meets the pre-set rule for "better". On the full model it is +0.008 [-0.000, +0.015], Holm p = .71: no reliable difference. At person level the ensemble is equal to the logistic regression (-0.002 and +0.002). This result is not blind: the addendum was written after an initial look at the ensemble on two camera noise draws and the clean pass (repeat 0), so the interval and the Holm p are optimistic for the ensemble; it is exploratory, uses one repeat of the folds, and is a single positive row among ten. It should be confirmed on new data before it is claimed.
- Cost: all models score a person in under 3 ms and need under 0.06% of one CPU core in real time (one scoring per 5 s display), so processing power is not a reason to choose or reject any of them. The logistic regression is 18 to 56 times smaller (by pickle size) than the others (1.8 kB against 33-102 kB), needs the least memory (1.2 MiB growth against 2-19 MiB), and 52 operations per item against 200-3,224. The ensemble adds the XGBoost cost (about 100 kB, 150 ms fit).
- Decision: the logistic regression stays the frozen model of the preregistered held-out test. Nothing here changes that choice; the ensemble is reported as exploratory on development data only.

## 7. Deviations from the brief and from PLAN.md (all stated before the result they affect, except 5 and 6)
1. ONE repeat (repeat 0 of the frozen folds) instead of 3 for all models including the logistic regression, as decided by the run-time rule of PLAN.md (Addendum A; estimated 2.4 h for 2 repeats, 1.2 h for 1 repeat; the pilot estimate for 3 repeats was 3.6-6.2 h on the shared laptop). The 3-repeat reproduction of the reference was done in full. The 1-repeat AUCs (0.7824 FULL, 0.7709 CAMERA for the logistic regression) therefore differ from the paper's 3-repeat values. Checkpoints are per repeat, so repeats 1 and 2 can be added by resuming `mc_run.py all --repeats 1,2` and re-running `mc_ens.py`, `mc_eval.py`, `mc_cost.py`, `mc_report.py`.
2. Exact speed-up (Addendum A): XGBoost and LightGBM 100-tree settings are scored with the first 100 trees of the 300-tree model (identical scores, maximum difference 0.0, mc_check_prefix.py). CatBoost is not prefix-consistent and every setting was fitted directly.
3. Route for CatBoost: `--target` folder inside model_compare instead of installing into `.venv` (dry run showed no change to existing packages; the folder keeps `.venv` untouched). No Python 3.12 venv and no .npz exchange.
4. Ensemble and the 10-row Holm family were added by ADDENDUM_ensemble.md after an initial look; not blind.
5. CatBoost's `predict` was called without `thread_count=1` during the main run, so these (short) scoring calls could use several threads; fits and all other libraries used one thread. Scores are unaffected (identical, checked); `mc_lib.py` now passes `thread_count=1` (the edit was made after the run; it does not change any score).
6. The run stopped when the session ended (during W30_d2 after XGB 64_g1) and was restarted from the checkpoints; the unit in progress was recomputed from scratch (deterministic).
7. Cost measured with the laptop in the state described in section 5 (not a quiet machine); the cost measurement was run twice (CatBoost thread fix) and the second run is reported.
8. Peak memory is the working-set growth in a fresh process after imports, not an allocator-level peak.
9. Scoring times use the 3 test people of fold 0 of set 64_g1 (18 timings per model) instead of the 5 people of the plan.

## 8. Files (all in results/phase10/model_compare/)
PLAN.md, PLAN.sha256.txt, ADDENDUM_ensemble.md, card.md, results_table.csv (all metrics + cost columns for 12 rows), results.json, results_item_person.csv/.json (metrics before cost merge), table_models.tex, cost.csv, cost.json, repro.json, pilot_timing.json; code: mc_lib.py, mc_run.py, mc_ens.py, mc_eval.py, mc_cost.py, mc_report.py, mc_pilot.py, mc_prof.py, mc_check_prefix.py, mc_check_cat_threads.py; logs: log_run.txt, log_resume_console.txt, log_eval.txt, log_cost.txt; checkpoints ck/; CatBoost folder site_pkgs/; cost_data.npz (numbers of one training set, no images).

## 9. Paragraph for the paper (plain English, ready to paste)
We also asked whether a different classifier would do better than the logistic regression. On the same participant-disjoint folds and with the same tuning inside the training participants, we tried XGBoost, LightGBM, CatBoost and a support vector machine with a radial kernel. None of them was reliably better or worse than the logistic regression, either on the full model (clean 30 Hz recordings) or on the camera model (webcam-level noise): all differences in AUC were between $-0.017$ and $+0.010$, and every interval included zero. The logistic regression is also the smallest (1.8 kB against 33 to 102 kB) and needs the least work to score an item (52 operations); scoring a whole person took under 3 ms and under 0.06\% of one processor core in real time for every model, so the cost is not what decides the choice. A simple average of the logistic regression and XGBoost scores had the highest AUC and was reliably better than the logistic regression on the camera model ($+0.018$, 95\% interval $+0.009$ to $+0.027$), but not on the full model, and not at the level of the person. We looked at this average before fixing the test, so we treat it as a lead for future work and keep the logistic regression as the model of the held-out test. Differences were measured with one repeat of the cross-validation folds.
