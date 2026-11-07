# Task M2 plan: same-fold comparison of XGBoost, LightGBM, CatBoost and an RBF-kernel SVM with the frozen logistic regression

EXPLORATORY. Written before any full run. Development data only (Delmas, all six group sets). The held-out dataset (data/external/vdc2024, results/phase9/vdc_heldout) is never read.
Fills the red marker in paper_rec/main.tex, section "Model and decision". Source engine: results/phase10/model_search/ (ms_lib.py, ms_run.py, ms_eval.py, ms_extra.py), imported unchanged; nothing is written there.
Wording: "known item" / "unknown item".

## Conditions (same frozen folds as the stored analysis: 5 participant-disjoint folds x 3 repeats, results/phase2/delmas_raw/folds.json)
- FULL   = Delmas clean 30 Hz, the 13 standardised features (ALL_z without microsaccades). Reference stored item AUC 0.7854480364.
- CAMERA = Delmas 30 Hz, dwell from samples, no pupil, 10 features, measured webcam noise ("anch": 1.45 deg offset, jitter, 3% loss), three noise draws, AUC = mean over the 3 draws.
           Reference mean 0.7737 (draws 0.77324, 0.77214, 0.77574). 30 Hz is the paper's reference rate (not the 15 Hz app case).
Reference = the frozen L2 logistic regression through the same engine (C in {0.01, 0.1, 1, 10}). It is reproduced first; the task stops if it does not match.

## Models and grids (fixed here)
All inputs: the same feature matrix as the frozen model, median-imputed and standardised with training participants only (inside the model pipeline). Raw margin / decision function is the score.
All libraries one thread (xgboost n_jobs=1, lightgbm n_jobs=1, catboost thread_count=1, allow_writing_files=False, OMP/MKL/OPENBLAS=1). random_state = the outer repeat index (same seed rule as the reference).
- XGBoost (tree_method hist): n_estimators {100, 300} x max_depth {2, 3} x learning_rate {0.03, 0.1}; subsample 0.8, colsample_bytree 0.8, scale_pos_weight = n_unknown/n_known in the training rows. 8 settings.
- LightGBM: n_estimators {100, 300} x num_leaves {4, 8} x learning_rate {0.03, 0.1}; min_child_samples 20, class_weight "balanced", force_row_wise, deterministic. 8 settings.
- CatBoost: iterations {100, 300} x depth {2, 4} x learning_rate {0.03, 0.1}; auto_class_weights "Balanced". 8 settings.
- RBF-SVM (SVC): C {0.1, 1, 10} x gamma {"scale", 0.01, 0.1}; class_weight "balanced"; decision_function as score. 9 settings.
Tuning: only inside the training participants, the same inner splitter (StratifiedGroupKFold, 4 folds, shuffle, random_state = repeat) as the reference, by mean inner-fold item AUC, first maximum wins in the order of the grid as written above (first listed parameter varies slowest).

## Run-time rule (fixed before any result)
Each configuration is timed on one group set (pilot, not a result). If the estimated whole task is more than 2 hours, the number of outer repeats (of the 5-fold split; inner folds are not changed) is reduced, for ALL models including the logistic regression, to the largest of 2 or 1 whose estimate is at most 2 hours. If even one repeat exceeds 2 hours the task stops and is reported. The 3-repeat reproduction of the reference is always done.

## Metrics (as model_search/ms_eval.py)
Item AUC (pooled per group set, repeat-averaged out-of-fold scores, mean over the six sets, mean over the 3 draws for CAMERA); person AUC with the Delmas CIT statistic; EER; HTER, detection and realised false-alarm rate at the 7% threshold set on the training participants' inner out-of-fold scores.
Bootstrap: 95% paired participant-cluster interval of the difference from the logistic regression, 2000 resamples, seed 20261106 (participants resampled within each set, all models on the same resamples). Paired score swap: 2000 swaps, seed 20261107, normalised within-set ranks, two-sided. Holm over all 8 model x condition rows (primary metric = item AUC).
Decision: "better" only if the interval excludes zero on the positive side AND Holm p < .05; "worse" if the interval is below zero AND Holm p < .05; otherwise "no reliable difference". No winner by point estimate. Person AUC differences are secondary and descriptive (not in the family).

## Cost per model (logistic regression included), laptop Intel Core i7-8550U, one thread, measured while other jobs were running
- Model size: length of the Python pickle of the fitted model (the whole pipeline: imputer, scaler, model), for the most often chosen setting in the FULL condition; parameter count or nodes where meaningful.
- Scoring time: all rows of one person (a set-64_g1 person, features already computed), median over 5 people x 6 rounds, models alternated within every round; wall (perf_counter) and CPU (process_time) in ms.
- Fit time: one fit at the chosen setting on the training people of fold 0 / repeat 0 of set 64_g1, CPU and wall, median of 3 rounds, models alternated.
- Peak memory: sequential fresh subprocess per model (fit + score), Windows peak working set minus the baseline after imports (MiB).
- Share of one CPU core for real-time scoring: CPU time to score the items of one person once per display divided by the 5 s display length. Reported for (a) the four items of one display and (b) the person's whole set of items (conservative). 
- Work per item: logistic regression = arithmetic operations (standardise: p subtractions + p divisions; dot product: p multiplications + p additions); SVM = support vectors x features (mean over fitted models; each is a distance computation, about 3 operations); boosted trees = nodes visited = trees x mean leaf depth (measured on the person's items; CatBoost oblivious trees: iterations x depth).

## Addendum A (written after the timing pilot, before any model result; original SHA-256 of the text above in PLAN.sha256.txt line 1)
- Pilot (one outer fold of set 64_g1, all grid settings, one thread, laptop shared with a live webcam test; mc_pilot.py, pilot_timing.json): estimated whole task with 3 repeats about 6.2 h
  (first pilot) / 3.6 h (second pilot, after the speed-up below). Estimate for 2 repeats 2.4 h, for 1 repeat 1.2 h.
- Exact speed-up that does not change any score: for XGBoost and LightGBM the first 100 trees of the 300-tree model are the 100-tree model (fixed seed and learning rate);
  checked numerically (mc_check_prefix.py, maximum score difference 0.0). The 300-tree model is fitted once per (training rows, depth or leaves, learning rate) and the 100-tree setting scores with its first 100 trees.
  CatBoost is not prefix-consistent (difference > 1 in raw score), so every CatBoost setting is fitted directly.
- Decision by the run-time rule above: 2 repeats exceed 2 hours, so the outer repeats are reduced to ONE (repeat 0 of the frozen folds: the same 5 participant-disjoint folds) for all models including the logistic regression.
  Checkpoints are per repeat, so repeats 1 and 2 can be added later by resuming. The 3-repeat reproduction of the logistic regression (0.7854480364 FULL; mean 0.7737 CAMERA) is done in full first.
  Consequence: the primary AUCs of this task are 1-repeat values and differ slightly from the 3-repeat values of the paper's tables; the reference is the logistic regression on the same repeat.
