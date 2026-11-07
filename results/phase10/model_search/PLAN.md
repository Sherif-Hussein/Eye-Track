# Task M plan: can another model, ensemble or feature extraction reliably beat the fixed pipeline for recognition from gaze?

EXPLORATORY. Written before any full run of a candidate. Development datasets only (Delmas, Chen). The held-out dataset is not read.
Wording: "known item" / "unknown item".

## Prior results (not repeated)
SVM, XGBoost, LightGBM, MiniRocket, voting, stacking and cascade on saved predictions (results/ensemble_preview, not fully nested): no reliable gain
(docs/all_experiments.md, sections C2 and D1).

## Reference and reproduction (done or run first; the chain stops on a mismatch)
- Delmas 30 Hz clean, 13 standardised features (ALL_z without microsaccades), L2 logistic regression, C in {0.01, 0.1, 1, 10} by 4 participant-grouped
  inner folds, frozen folds (results/phase2/delmas_raw/folds.json). Stored image-level AUC 0.7854480364. Reproduced through the new engine: max abs score
  difference 1.8e-15, pooled AUC 0.7854480364 (repro_D30.json).
- Delmas webcam case (D W15): sample-based dwell, no pupil, 10 features, measured webcam noise ("anch": 1.45 deg offset, jitter, 3% loss), three noise draws,
  AUC = mean over draws. Reproduction at 30 Hz: draws 1-3 each match the stored out-of-fold scores (max diff 1.8e-15), pooled AUC 0.77324, 0.77214, 0.77574,
  mean 0.7737 (repro_W30_d*.json). Candidates are run at 15 Hz (the app's case).
- Chen 30 Hz, 75 features (25 x 100 ms bins of pupil, dx, dy, within-person z), unified pipeline (C grid of 9, inner folds min(5, n/3)), frozen folds.
  Stored trial AUC 0.6016. Repeat 0 reproduced bit-identically (scores and inner out-of-fold scores, difference 0.0); repeats 1-2 and the AUC are in the chain.

## Evaluation units, folds, tuning (identical for every candidate)
Delmas: image-level rows (participant, image) of the six group sets; 5 participant-disjoint folds x 3 repeats (frozen). Chen: trial rows, 10 folds x 3 repeats (frozen).
All tuning inside the training participants (same inner splitter and seed as the frozen model, shared by all candidates); mean inner-fold item AUC, first maximum wins.
Within-person z-scoring is label-free. Feature blocks that need fitting (PCA, scalers, imputers) are fitted on training participants only. No test person is used for
selection, scaling or thresholds. Labels are never part of features. CPU, one process, one thread.

## Candidates (conditions D30 = Delmas clean 30 Hz; W15 = Delmas no-pupil 15 Hz measured webcam noise; C30 = Chen 30 Hz)
Grids are fixed here.
- A1 M1_cl (D30, W15): conditional logit, softmax over the 4 items of each display, trained on displays with exactly one known item, same (trial-level, within-person z) base
  features, L2 with C in {0.01,0.1,1,10}; every trial item scored, image score = mean over the image's presentations. Not applicable to Chen (single-item trials).
- A2 M2_pw (D30, W15): pairwise logistic (known minus unknown within a display, 3 pairs per display, both signs, no intercept), same features and grid.
- B3 F3a_scan, F3b_scan+base (D30, W15): 12 scan-path features per item per trial from the samples (visit order, first, last, visits, returns, mean and max run length
  in ms, transitions out and in, transitions with the first and with the last visited item, out-transition entropy), mean over presentations, within-person z; LR.
- B4 F4a_pca, F4b_pca+base (D30, W15; Chen: F4a on the 75 bin features): per-image 100 ms occupancy time course (50 bins), PCA on training people only, k in {3,...,8},
  C in {0.01,0.1,1,10} (Chen: the 9-point grid), within-person z of the scores, LR.
- B5 F5a_c22, F5b_c22+base (D30, W15, C30): catch22 (aeon 1.6.0) per item per trial on the 50-bin occupancy series (Chen: on the 3 channel series at 30 Hz), mean over
  presentations, within-person z; LR.
- C6 C6_enet (all): elastic net (saga), Delmas C in {0.01,0.1,1,10} x l1 ratio {0.2,0.5,0.8}; Chen C in {0.001,0.01,0.1} x {0.2,0.8} (reduced because saga takes 76 s per fold at C=1).
- C7 C7_rf (all): random forest, 100 trees, depth in {2,4}, min leaf 5, balanced subsample.
- C8 C8_cnn (all, one try, no tuning): numpy 1-D CNN (torch is not installed): conv (8 filters, width 5) -> ReLU -> global mean and max -> linear; 65 parameters (Delmas input: 1 x 50
  occupancy), 145 (Chen input: 3 x 25 bins); Adam, 40 epochs, weight decay 1e-3. No inner folds, so no threshold metrics for it and it is not a member of the ensembles.
- D9 E9_vote (all): per outer fold, pick the member of the A-B pool (M1, M2, F3a, F3b, F4a, F4b, F5a, F5b; Chen: F4a, F5a, F5b) with the highest AUC on the training people's inner out-of-fold
  scores; soft vote = mean of within-person z-scores of the frozen model and that member.
- D10 E10_stack (all): level-2 L2 logistic regression (C in {0.01,0.1,1} by grouped inner folds) on the within-person z of the inner out-of-fold scores of the frozen model, the A-B pool, elastic net
  and the forest; applied to the test people's z-scores.
- E11a_trim20, E11b_median (D30, W15 only): person statistic of the Delmas CIT test (known-item score vs unknown-item scores, simulated unknown person from pseudo-probes) with a
  20% trimmed mean, or median / 1.4826 MAD, instead of mean / SD, on the frozen model's scores. Not run on Chen (its statistic is a one-sided t on hundreds of trials).
Frozen model F0 is run through the same engine for the inner out-of-fold scores needed by D9, D10.

## Metrics
Item AUC (primary); person AUC (Delmas CIT statistic; secondary, primary for E11 rows); EER; HTER, detection and realised false-alarm rate at the 7% threshold set on the training
participants' inner out-of-fold scores (item level and person level); 95% paired participant-cluster bootstrap intervals (2,000 resamples, seed 20261106; participants resampled
within each group set, all configurations on the same resamples, W15 = mean over the 3 draws); model size, ms per person, peak memory (ms_extra.py cost).

## Test and decision rule (fixed)
Paired score swap as results/phase8/R4_paired_swap: 2,000 swaps (seed 20261107), each participant swaps its scores between the candidate and the frozen model with probability 0.5 (a person in
two sets is swapped in both; W15: in all three draws), statistic = mean AUC difference; scores replaced by normalised within-set ranks (the AUC is unchanged) so that models with different
score scales can be exchanged; two-sided p. Holm over the whole family = all candidate x condition rows (primary metric above). A candidate WINS only if the bootstrap interval of its
difference from the frozen pipeline excludes zero on the positive side AND Holm-adjusted p < .05. A significantly lower candidate is reported as worse. The winner is never picked by the
best point estimate. If nothing wins, "no candidate beats the frozen pipeline" is the result. No candidate may be added after seeing full-run results unless labelled post hoc.

## Other reports
- Learning curve (25, 50, 75, 100% of the training participants, same folds): for the winner; if none, for the frozen pipeline and for the candidate with the highest point estimate, labelled descriptive.
- More displays per person (8, 16, 32, 64) for the frozen model: Delmas 64-trial sets (64_g1, 64_g2, 64_g3), model retrained at each n, displays drawn at random half with a known item (3 draws for 8, 16, 32; D30 and W15).
- Costs: per candidate parameters, ms per person to score from features, extra feature extraction ms per person, peak memory.

## Known deviations from the brief (declared in advance)
- CNN in numpy (no torch). Chen has no display structure, so M1, M2, B3, F4b and E11 do not apply there. Trial-level features are used for M1 and M2 (a display needs trial items); the frozen model
  uses presentation-averaged features. Elastic-net and forest grids are small to keep the run time feasible on one thread. The smoke tests used one repeat of one group set and are not results.
