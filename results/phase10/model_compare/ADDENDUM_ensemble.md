# Task M2 addendum: final table and the logistic regression + XGBoost ensemble

EXPLORATORY. Added 2026-11-07 at the user's request. PLAN.md and its hash are unchanged; this file adds to them.
This addendum was written after an initial look at the ensemble on two camera noise draws and the clean pass (repeat 0), so the ensemble test is not blind. That must be stated wherever it is reported.

## Final table
One table, rows in this order, for both conditions (FULL and CAMERA):
logistic regression (reference), XGBoost, LightGBM, CatBoost, RBF-SVM, ensemble (logistic regression + XGBoost).
Each row against logistic regression: item AUC with interval, difference with paired interval, person AUC with interval, difference with paired interval, Holm p, verdict, EER, HTER.
Cost columns for every row, per the PLAN's cost section: scoring time, fit time, peak memory, model size, share of one core.

## Ensemble, fixed here
- Weights 50/50, no tuning.
- Scale: within each outer fold, each model's test scores and inner-fold scores are z-scaled with the mean and SD of that model's own inner-fold training scores. No test person influences the scale.
- The two models keep the settings already chosen for them inside the training people. No new training for accuracy.
- Same repeat (0), group sets, metrics, bootstrap and swap test as the PLAN; CAMERA is the mean over the three draws.

## Tests
- Holm family: the PLAN's rows (4 models x 2 conditions) plus the 2 ensemble rows, 10 in all.
- The ensemble is also compared with XGBoost alone (paired interval only, reported as a labelled extra).
- Verdict rule as in the PLAN: better = interval above zero and Holm p < .05; worse = interval below zero and Holm p < .05; otherwise no reliable difference. Item and person results are given separately.

## Cost of the ensemble
Measured in the same interleaved rounds as the other models. Scoring and fit time are logistic regression and XGBoost together; peak memory comes from a fresh process that fits and scores both; model size is the sum of both.

## Use
Logistic regression stays the frozen model of the preregistered held-out test. The ensemble is reported only as exploratory on development data.
