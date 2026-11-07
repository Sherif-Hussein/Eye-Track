# Card: held-out recognition analysis, Van der Cruyssen et al. (2024) data (Task 2.1, step 2)

Preregistered plan (OSF, registered 5 Nov 2026, updated 6 Nov 2026 before data access) as modified only by `../DEVIATIONS.md`.

Terms. "Known item" = the crime-related item; "unknown items" = the three other items in the same four-item display. Participants who KNOW the items: guilty (handled them) and informed innocent (saw pictures). Participants who do NOT know them: naive innocent. Higher score = the known item looks more like a recognised item.

## What was run
1. `s1_features.py`: cleaning (D6): the authors' 22 exclusions -> 133 participants (42 guilty / 45 informed / 46 naive); session suffixes stripped; practice rows removed with the response-file trial indices; participant 23 kept (18 valid displays, at least 8); participants 17, 31, 43 included. Displays (D3): the 32 four-item displays containing the known item, 4 items each (known item = label 1, three unknown items = label 0). 11 features (D2): dwell 1-1000 and 1000-5000 ms, five dwell bins (0-500, 500-1000, 1000-2000, 2000-3500, 3500-5000 ms, the same edges as our Delmas features, `src/edd/delmas_raw.BINS`; each is the mean of the 100 ms sample shares in bins 0-4, 5-9, 10-19, 20-34, 35-49), time to first look (start of the first 100 ms bin with a non-zero share, 5000 ms if none), fixation count, number of visits, first fixation on the item. No pupil features. `features.csv` (16,968 rows). Z-standardised within participant across all of the participant's items by `unified.zscore_within`.
2. `s2_authors_index.py`: the authors' equal-weight index, following `GuiltyNaive.R`.
3. `s3_folds.py`: `folds.csv` written before any fitting: 10 folds x 3 repeats, participant-disjoint, stratified by group, seeds 20261106 + repeat.
4. `s5_observed.py`: nested cross-validation with the project's own code (`unified.tune_fit`: L2 logistic regression, balanced weights, C from 9 log-spaced values 1e-3 to 10 by mean AUC over at most 5 inner participant-disjoint folds). The item model is trained only on the guilty participants of the training folds. Participant score = standardised difference between the known-item score and the unknown-item scores on the 32 displays (`unified._cit_z`, known-vs-unknown part), averaged over the 3 repeats (`scores.csv`). Threshold: set inside each training set to flag 7% of the training naive participants (`unified.thresholds`).
5. `s6_perm.py`: 1,000 refit label permutations each for H1 (guilty/naive) and H2 (guilty/informed); each refits the item model and threshold on all 30 folds. `s7_final.py`: p-values, bootstrap intervals (2,000 participant resamples, within group), H3, H5, Holm, estimation, exploratory analysis; output `results.json`.
6. One thread, one process throughout (`OMP/MKL/OPENBLAS_NUM_THREADS=1`).

## Reproduction check of the published AUC (H3 precondition)
Authors' index, guilty (n = 42) vs naive (n = 46): AUC 0.7122, DeLong 95% CI [0.603, 0.822]. Published .71 [.60, .82]. Reproduced (within +-0.01). `authors_index_check.json`.

## Results

| Hypothesis | Contrast (known vs not) | AUC | 95% CI (participant bootstrap) | p (one-sided unless stated) | Holm p | Decision |
|---|---|---|---|---|---|---|
| H1 (primary) | guilty (handled) vs naive | 0.687 | [0.573, 0.790] | 0.0030 | not applicable (primary) | Supported (p < .05 and CI lower bound > .50) |
| H2 | guilty vs informed | 0.654 | [0.531, 0.769] | 0.0100 | 0.030 | Supported after Holm |
| H3 | our pipeline minus authors' index, guilty vs naive (two-sided) | 0.687 vs 0.712; difference -0.025 | [-0.104, 0.052] | 0.510 | 0.510 | No evidence of a difference |
| H5 | informed (saw pictures) vs naive | 0.573 | [0.456, 0.690] | 0.117 | 0.234 | Not supported |

Notes. Permutation p = (1 + number of permuted AUCs >= observed) / (1 + 1000); permuted AUC mean 0.499 (SD 0.066) for H1, 0.499 (SD 0.064) for H2. H5: 1,000 permutations of the informed/naive labels over the fixed scores; the item model, trained on guilty participants, is unaffected by this permutation. H3: paired participant bootstrap (2,000) and a two-sided paired permutation test with 2,000 swaps of the two methods' scores within each participant; each method's scores were rank-transformed over the 88 participants before swapping. Holm over H2, H3, H5 (alpha .05). Our pipeline is nominally 0.025 AUC below the authors' index and the interval is wide, so equivalence is not shown either.

## Rates at the 7% threshold, and EER / HTER (estimation, no tests)

| Group | Flagged / n | Rate | Clopper-Pearson 95% CI |
|---|---|---|---|
| Naive (do not know): false-positive rate | 3 / 46 | 6.5% | [1.4%, 17.9%] |
| Informed (know, saw pictures): rate flagged as knowing | 1 / 45 | 2.2% | [0.1%, 11.8%] |
| Guilty (know, handled): detection rate | 7 / 42 | 16.7% | [7.0%, 31.4%] |

- Informed minus naive false-positive rate: -4.3 percentage points, bootstrap 95% CI [-13.0, +4.4].
- Equal error rate: guilty vs naive 0.330; guilty vs informed 0.425; informed vs naive 0.396.
- HTER at the 7% threshold (guilty vs naive) = (6.5% + 83.3%) / 2 = 0.449.
- Per-repeat rates (descriptive): guilty 21%, 26%, 19%; informed 4.4%, 2.2%, 2.2%; naive 8.7% in each repeat.
- With this threshold the detection rate is low: at about 5% false positives only about one in six handlers is flagged. Informed participants were flagged no more often than naive ones.

## Exploratory (D4, not in any family, no p-values)
Frozen pupil-free 30 Hz Delmas et al. model (`app/model/frozen_model.json`, variant sample_based, 10 features, 41 stored numbers) applied unchanged to the closest features (sign as stored: less dwell on the known item gives a higher score). All 10 inputs map by name to our features. Differences from the training features: dwell_1 window is 1-1000 ms (not 200-1000), bins and first look come from 100 ms occupancy time courses, and z-scoring is over display-level rows.
AUC guilty vs naive 0.681 [0.551, 0.790]; guilty vs informed 0.639 [0.517, 0.757]; informed vs naive 0.566 [0.449, 0.685]. Per-participant values: `scores_frozen_exploratory.csv`.

## Further deviations
1. Integer decisions at the 7% threshold. Each participant is tested once per repeat with a different fold threshold. For the rates above (and their Clopper-Pearson intervals) a participant is flagged when the mean over the 3 repeats of (statistic minus that repeat's fold threshold) is above 0, giving one integer decision per participant. Per-repeat rates are given as well.
2. The 7% rule flags at most 7% of the training naive participants: the threshold is the floor(0.07 m)-th largest training naive statistic and statistics strictly above it are flagged. With about 41 training naive participants this flags 2 of 41, about 4.9%.
3. Permutation runs use the stock `tune_fit` (scikit-learn) code with no solver change; about 5.5 h per 1,000 permutations per test.
4. Bootstrap intervals resample participants within each group (stratified). Bootstrap for H3 resamples the 88 participants jointly for both methods.
5. EER is computed on the repeat-averaged participant statistics (threshold-free); no threshold is needed for it.
6. Exploratory model: the frozen model's z-scoring is applied over display-level rows (128 per participant), not over item means across presentations as in the app, because items here are position-defined in each display.

## Summary (5 lines)
1. The low-cost pipeline separates people who handled the items from naive people with AUC 0.69 [0.57, 0.79], one-sided p = 0.003: the primary hypothesis H1 is supported.
2. It also separates handlers from people who only saw pictures (AUC 0.65, Holm p = 0.030), but not picture-viewers from naive people (AUC 0.57, Holm p = 0.23).
3. The authors' index reproduces their published AUC (0.71); ours is 0.69, and the difference is small and not significant (p = 0.51).
4. At a threshold allowing about 5-7% false positives, only 17% of handlers were detected (7 of 42); false positives were 6.5% for naive and 2.2% for informed participants; EER 0.33.
5. The frozen Delmas model gives a similar pattern (AUC 0.68 guilty vs naive, exploratory), so the finding is not specific to refitting.
