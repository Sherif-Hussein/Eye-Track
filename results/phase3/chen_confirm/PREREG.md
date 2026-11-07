# PREREG: chen_confirm (Phase 3 confirmatory test of I2b on Chen, Mathot & Akyurek 2026 data)

Written 2026-11-02, before any real-label outcome of this experiment was computed. This file is never edited after the run stage starts. Any deviation goes in `DEVIATIONS.md`. The SHA-256 of `folds.json` is recorded in `FREEZE.md` right after the folds are frozen (before the run stage).

What existed before this document: the Phase 2 screening results (`results/phase2/chen/`), which chose I2b by a pre-fixed rule on Exp 1 and Exp 2 separately with folds seeded 2026-2028. Only code smoke tests on label-shuffled data (outputs in a scratch folder, not in this project) were run for this experiment; no outcome of the real labels has been computed.

## 1. Hypothesis

H1 (one-sided): at a decision threshold calibrated to a false-positive rate (FPR) of at most 7% on unfamiliar-vs-unfamiliar data, I2b detects a larger share of the familiar-face participants than the paper's rate rule does, in the pooled sample of Exp 1 and Exp 2.

## 2. Data

- Pool Exp 1 (N = 29) and Exp 2 (N = 27): 56 participants, the same authors' exclusions as in screening (Exp 1 subjects 14, 15, 24, 30; Exp 2 subjects 8, 15, 22, 28). Participant unit id: Exp 1 `subject_nr`, Exp 2 `subject_nr + 1000`.
- Trials: familiar (T1 = 2, positive) and control 1 / control 2 (T1 = 3, 4; negative); target trials are dropped. Preprocessing exactly as `scripts/phase2_chen.py::preprocess` (default arguments: authors' blink reconstruction `p_dm`, 50-ms subtractive baseline, per-participant baseline |z| > 2 exclusion, 0-2500 ms, rate = diff + 25-sample Hanning). Traces come from `data/derived/chen_exp{1,2}_{traces.npz,meta.csv}` (built from `data/derived/chen_cache2/`); no re-extraction.
- Participant overlap check (done before the run, no outcome involved). The CSV logs contain `subject_nr`, `datetime`, `age`, `gender`, `handedness`, `parent_gender`, `controlImage1/2`, `familiarImage` and a constant `osf_id`; no field links a person across experiments. Findings: no identical datetime stamps (Exp 1 ran April-May 2023, Exp 2 June-October 2023); no pair of participants with the same two control images; the paper says both experiments recruited from University of Groningen first-year pools and does not say whether anyone took part in both or was excluded from the second. Demographic matching (same gender, handedness, parent gender, age differing by 0-1 year, Exp 2 not younger) gives 105 of 1023 cross-experiment pairs, and 25/33 Exp 1 and 22/31 Exp 2 participants have at least one such partner, so the key is uninformative: overlap can neither be confirmed nor excluded. Rule: (i) overlap is treated as unknown and reported as a limitation; (ii) folds are participant-disjoint within the pooled list, but a person who appears under two ids could sit in different folds; (iii) the sensitivity arm `I2b_perexp` (section 4) never trains on the other experiment, so it is immune to cross-experiment overlap; the primary result is described as robust to overlap only if `I2b_perexp` shows the same sign and a similar size (descriptive, section 8). Ids are never merged on demographic guesses.

## 3. Folds (frozen to `folds.json` before the run)

- 3 repeats x 10 participant-disjoint folds, seeds 3031, 3032, 3033 (not used in screening, which used 2026-2028), pooled list, stratified by experiment: within each experiment the participant list is permuted with `numpy.random.default_rng(seed)` and dealt round-robin into 10 folds (fold i gets items i, i+10, ...); fold i of Exp 1 is joined with fold 9-i of Exp 2 so fold sizes are balanced (5-6 participants each). Code: `edd.chen_confirm.make_folds`, written by `python -I scripts/phase3_chen_confirm.py folds`. The script never overwrites an existing `folds.json`.
- Each participant is in the test set exactly once per repeat.

## 4. Models (arms)

All use identical folds. Scores of a held-out participant come only from a model that did not see that participant.

- `rate_rule` (comparator): the paper's rate rule as in screening `B0_rate_peak`: peak sample = argmax |mean familiar - mean control| of the rate of pupil change over the pooled training trials of the same experiment; a one-dimensional logistic calibration (training trials) is used only for trial-level scores; the per-participant statistic is the rate value at the peak. Fit separately per experiment, as in screening (the paper analysed the experiments separately; the peak differs by experiment).
- `I2b` (primary model): exactly the screening model `I2b_bins_fusion`: 100-ms bin means over 0-2500 ms of baseline-corrected pupil size, rate, horizontal gaze toward the motion direction (dx), vertical gaze (dy) and missing-sample fraction (125 features, `phase2_chen.bin_features` with `FUSION` order), median imputation, standardisation, `SelectKBest(f_classif)` with k in {20, all}, L2 logistic regression with C in {0.003, 0.03, 0.3}, `class_weight="balanced"`, tuned by `GridSearchCV(scoring="roc_auc", cv=StratifiedGroupKFold(3, shuffle=True, random_state=0))` grouped by participant inside the training participants (the screening function `fit_lr`). No experiment indicator (screening used none). One model is fitted on the training participants of both experiments pooled (the pooled data is what is new); standardisation is therefore pooled.
- `I2b_noMiss` (pre-specified secondary): identical to I2b with the 25 missing-fraction features removed (100 features); `k` in {20, all}.
- `MissOnly` (exploratory): I2b pipeline on the 25 missing-fraction features only (k in {20, all}).
- `I2b_perexp` (sensitivity): the screening-style I2b fitted on the training participants of one experiment at a time (no cross-experiment training).

## 5. Primary outcome and test

Per participant, per repeat, the decision is `z_fam > tau`. `z_fam` is the one-sided Student t statistic (familiar vs both controls, held-out scores, as in `phase2_chen.person_tests`) converted to a normal z via the survival function. The threshold `tau` is calibrated on the training participants only: for every training participant, inner out-of-fold scores are produced with `StratifiedGroupKFold(5, shuffle=True, random_state=0)` grouped by participant, refitting the chosen pipeline (hyperparameters of the outer-training fit held fixed; for the rate rule the peak is re-picked inside each inner training part); for each training participant the control-1-vs-control-2 z is computed in both directions (z_c1, z_c2); the null set is the pooled z_c1 and z_c2 of the training participants of the same experiment as the test participant; `tau` is the smallest null value such that at most 7% of the null values lie strictly above it (the `floor(0.07 * m)`-th largest, 0-indexed, m = null size). The realised FPR on held-out control 1 vs control 2 is reported for every arm.

Primary outcome: person-level calibrated detection rate = mean over participants and over the 3 repeats of the 0/1 detection indicator (all 56 participants, equal weight), I2b minus rate_rule. The paired participant bootstrap (2000 resamples of the 56 participants with replacement, `numpy.random.default_rng(20261102)`, the same resamples for all arms) gives the 95% percentile CI. One-sided p = (1 + #{bootstrap deltas <= 0}) / (2001). Test at alpha = .05, a single test, no multiplicity correction. **H1 is confirmed if and only if p < .05.** A sign-flip randomisation test (20000 flips of participant-wise differences) is reported as a sensitivity check and does not change the decision.

## 6. Secondary outcomes and confound analyses (Holm family of 9)

Same bootstrap draws. One-sided "I2b better" tests, p = (1 + #{delta <= 0 in the better direction}) / 2001:

- S1 person AUC, S2 trial AUC, S3 trial EER, S4 trial HTER, S5 person EER, S6 person HTER, each I2b minus rate_rule. Person-level metrics follow screening (positives: each participant's familiar-vs-control z; negatives: the same participant's control-1-vs-2 z; threshold z = 1.645 for HTER); trial level uses `edd.metrics` definitions on the held-out trial scores (threshold 0 for HTER), averaged over repeats.
- Confound (a)/(c): C2 = detection at FPR <= 7%, `I2b_noMiss` minus rate_rule (one-sided). C3 = `I2b` minus `I2b_noMiss` (two-sided, bootstrap p = 2 x min tail).
- Confound (b): C1 = paired Wilcoxon test (two-sided) across the 56 participants of the within-participant difference (familiar minus control) in the mean missing-sample fraction over 0-2500 ms. Also reported (descriptive): per 100-ms bin, per experiment, trial number and block number (familiar vs control), bootstrap CI of the mean difference.
- Holm correction over the 9 tests S1-S6, C1, C2, C3. Reported with raw and Holm-adjusted p.
- Decision rule for confound (c): the I2b gain is said to **survive removal of the missing channel** if C2 has Holm p < .05. Also reported: delta(I2b-noMiss) / delta(I2b). If the primary test is not confirmed, C2 is still reported and interpreted descriptively.
- Interpretation of (b): a significant C1 with a missing-fraction channel that carries weight means the channel may reflect a blink or tracking difference between the trial types and its use is not evidence of recognition; absence of a difference is not proof of absence.

Exploratory, descriptive (CIs only, no p values, no decision): `MissOnly` and `I2b_perexp` minus rate_rule; I2b minus `I2b_noMiss` for detection, trial AUC and person AUC; channel share of absolute coefficients in a pooled I2b refit on all participants.

## 7. Metrics reported for every arm

Detection at FPR <= 7% and realised FPR; detection at alpha .05 and its FPR on control 1 vs 2; person AUC, EER, HTER; trial AUC, EER, HTER; all with 95% participant-bootstrap CIs; paired differences against rate_rule. Per experiment rows are descriptive only.

## 8. Stopping, reporting, engineering

- One run of the full design; no interim looks; no change of models, features, grid, folds, outcome or tests after the run starts. Everything is reported regardless of the outcome, including a null or negative result.
- Interpretation guide fixed in advance: confirmed = primary p < .05; "not confirmed" otherwise (the CI is then reported as evidence for or against a small effect, not as evidence of no effect). The `I2b_perexp` arm is described as consistent with the primary result if its delta over rate_rule has the same sign.
- Code bugs found during the run are fixed in new code only and logged in `DEVIATIONS.md`; debugging uses label-shuffled data. n_jobs = 2; no package changes; existing files are not modified.
- Checkpoints: one file per repeat in `checkpoints/`; `RESUME.md` is updated after every repeat.
- Files: `scripts/phase3_chen_confirm.py`, `src/edd/chen_confirm.py`; outputs only in `results/phase3/chen_confirm/` (PREREG.md, FREEZE.md, folds.json, checkpoints/, results.json, card.md, RESUME.md, logs).

## 9. Known features of the design, stated in advance

- The 7% threshold is a point estimate from about 50 training null values per experiment, so realised FPRs differ between arms and between experiments; the primary outcome compares detection at the nominal target, and the realised FPRs are reported next to it. If the realised FPR of I2b exceeds that of the rate rule by more than 3 percentage points, this is flagged in the report as a caveat, not as a change of the test.
- The control-1-vs-2 comparison is a surrogate for innocent people (each control is one fixed image); there is no innocent group.
- Screening selected I2b on the same participants, so the earlier p-values are optimistic. The present test uses new folds on the same 56 people: it is a confirmatory test of the fixed model and the fold-sampling procedure, not an independent sample.
