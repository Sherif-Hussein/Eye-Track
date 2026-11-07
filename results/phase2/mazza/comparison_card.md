# Comparison card: RealEye webcam crime-scene dataset (track `mazza`)

**FINAL** (2026-11-02). Screening, all extras, 4-class, efficiency and permutation tests are complete. Tables are regenerated from `results.json`, `tables.md` and `fourclass_paired.json`.

## Summary for the authors (read this first)

1. **No reliable gain over the published method.** Under participant-disjoint nested CV, no idea beats the published logistic regression on any task. Every Holm-adjusted p is 1.000 on the three main tasks; on G2 vs G1 and G2 vs G4 every idea scores below the baseline. On the 4-class task, the best of 7 ideas (raw 90 features, LR) has accuracy +0.083 [0.000, +0.166] over the baseline, Holm p = 0.434.
2. **Selection after the fact is optimistic.** On G2 vs G3, the best of 18 configurations in hindsight (LightGBM, 90 features) scores AUC 0.750 (0.759 on repeat-averaged scores). When the configuration is chosen only on the other nine folds, the same procedure scores **0.673 ± 0.079** (95% CI [0.574, 0.808]), against an honest baseline of 0.679 and a published single run of 0.72. Do not quote 0.750 as an improvement. Its permutation p (0.032, 30 permutations) is for that one hindsight-picked configuration and does not account for choosing it from 18.
3. **Within-person z-scores for guilty vs innocent are promising but not proven.** Z-scoring each person's target-image fixation values against the same person's neutral images gives AUC 0.688 ± 0.009 versus 0.602 ± 0.014 for the baseline (0.673 vs 0.609 on two held-back fold seeds; permutation p = 0.031 with 31 permutations, the minimum possible; neutral-only control at chance, 0.484). The paired participant-bootstrap ΔAUC is **+0.072, 95% CI [-0.025, +0.169]** and Holm p = 1.000, so zero is inside the interval. The configuration was also the best of 18 on the same data. It deserves a confirmatory run; it is not a result.
4. **The deceptive vs honest signal is not specific to the crime scene.** An L2 logistic regression using only the six neutral room images (shown before the crime-related images) reaches AUC 0.659 [0.557, 0.765], as high as any model using the target images (best 0.655, z-scores). The negative control fails, so this task should not be presented as deception detection. The same neutral-only model reaches 0.664 [0.537, 0.782] on G2 vs G3, close to the full 36-feature baseline (0.679), so group differences in generic viewing behaviour (possibly the age, sex, education and glasses imbalances in the article's Table 1, or session effects) cannot be excluded for the published task either.
5. **The published numbers reproduce except the 4-class result.** G2 vs G3: ours 0.674 ± 0.033 over 10 seeds against 0.72 (inside our range 0.613 to 0.726). The 4-class 0.63 / 40.13% is not reproduced (ours 0.602 ± 0.015 and 35.4% ± 2.9, with 0.63 above every one of our 10 seeds).
6. Efficiency is trivial for all models (about 2.6 to 3.8 ms per participant, under 100 KB), so it is not a differentiator.

Dataset: Zenodo 10.5281/zenodo.7194229 (CC BY 4.0), deposited 13 Oct 2022. Local copy: `data/mazza/data.csv` (comma-separated ASCII, 157 rows × 92 columns) and `data/mazza/legend.docx`. The open-access article is saved at `data/mazza/paper/colasanti2023.pdf`.

Code: `scripts/phase2_mazza_repro.py` (audit and reproduction), `scripts/phase2_mazza.py` (folds, screening, extras), `scripts/phase2_mazza_configs.py` (pipelines), `scripts/phase2_mazza_tables.py` (tables), `src/edd/mazza.py` (features and runner). Numbers are in `results.json`, with reproduction in `reproduction.json`. The full 18-config tables for every task are in `tables.md`.

## 1. The published study

**Citation.** Colasanti, M., Melis, G., Monaro, M., Ricci, E., Bosco, F., Rossi, M., Biondi, S., Verrocchio, M. C., Di Domenico, A., Mazza, C., & Roma, P. (2024). Did You Commit a Crime There? Investigating the Visual Exploration Patterns of Guilty, Innocent, Honest, and Dishonest Subjects When Viewing a Complex Mock Crime Scene. *Journal of Nonverbal Behavior, 48*, 47–71. https://doi.org/10.1007/s10919-023-00438-5. Published online 31 Aug 2023.

**Corrections to the track brief.**
- The first author is Colasanti, not Mazza. The Mazza-first order is the Zenodo record's, and the Zenodo author list differs from the article's (no Verrocchio).
- The article is dated 2024 in print (online 2023), not 2022.
- **An ML baseline does exist.** The article itself reports logistic regression classifiers, so ours cannot be called the first ML work on this dataset. It can be called the first with nested CV, repeats, CIs, EER/HTER and efficiency.
- The crime area is the desk AOI, which is **left** on Targets 1, 3 and 4, central on Target 5, right on Target 6, and absent on Target 2. Central-third share is not the crime area. It rises in guilty groups because they look away from the desk.

**Later work.** No later ML study on this dataset or this paradigm by these groups was found (Semantic Scholar citation list of 6 papers, Zenodo, web searches). This is an absence from search, not proof. Citing paper titles were screened, not the full texts. Two nearby but different works are not comparable baselines: Foucher et al. 2025 (ETRA, EyeLink, other datasets) and Celniak et al. 2026 (another track).

| Item | Published |
|---|---|
| Task | Group discrimination in a mock theft; between-subjects, 4 groups |
| Participants | 160 recruited, 3 excluded for technical problems, 157 analysed; 56 male, 101 female; age 18–31 (M 22.27, SD 2.38); mostly students (89.8%) |
| Groups | Guilty Honest 40; Guilty Deceptive 38; Innocent Honest 40; Naive 39 |
| Sensor | RealEye webcam platform; HD webcam, 1080p at 60 fps, about 50 cm; gaze at least every 16 ms; stated platform accuracy about 113 px on average (Lewandowska, 2020) |
| Stimuli | Part 1: 6 neutral rooms; Part 2: recognition with 3 old and 3 new; Part 3: 6 target photos of the room. Each shown 10 s, in that fixed order |
| Features | Fixation count and summed duration per image, for the overall image and left/central/right thirds; Euclidean and time distance between fixations; gaze distance |
| Preprocessing | RealEye's own fixation detection and median noise filter. Otherwise not reported |
| Statistics | One-way ANOVA per feature, Tukey post hoc, Bonferroni-style alpha (.0125 whole image, .025 AOI), JASP 0.14 |
| Classifier | Logistic regression in WEKA 3.9, 36 predictors (count and duration for 6 targets × 3 AOIs), 10-fold CV. Single run; no tuning; no seed reported |
| Not reported | EER, HTER, MCC, sensitivity/specificity, CIs, permutation tests, repeated CV, efficiency, participant-disjoint statement (implicit, one row per person) |

Reported classification results (section "Classification Models", p. 63):

| Comparison | ROC AUC | Accuracy | Precision | Recall | F | Correct per group |
|---|---|---|---|---|---|---|
| 4-class | 0.63 | 40.13% | 0.39 | 0.40 | 0.40 | GH 10/40, GD 15/38, IH 19/40, N 19/39 |
| Guilty Deceptive vs Innocent Honest | 0.72 | 65.38% | 0.65 | 0.65 | 0.65 | GD 26/38, IH 25/40 |
| Guilty Deceptive vs Guilty Honest | 0.61 | 60.26% | 0.60 | 0.60 | 0.60 | GD 23/38, GH 24/40 |
| Guilty Deceptive vs Naive | 0.66 | 61.04% | 0.61 | 0.61 | 0.61 | GD 24/38, N 23/39 |

Guilty vs innocent (G1+G2 vs G3+G4) and deceptive vs honest (G2 vs rest) were **not reported**. Our numbers on those two tasks have no published counterpart.

Main statistical findings: Target 1 left AOI (desk), count F = 4.133, p = .008, η² = .075, and duration F = 3.327, p = .021, η² = .061; Target 1 central AOI, count F = 6.113, p < .001, η² = .107, and duration F = 7.200, p < .001, η² = .124. Target 4 left AOI count F = 5.268, p = .002, η² = .094. No whole-image effect. Effects fade over later images, which the authors attribute to fatigue or habituation. Their neutral images showed no overall group effect, but Neutral 2 central duration did differ between groups (F = 3.524, p = .017, η² = .065).

**Printed-value error in the article.** Table 3 gives Target 4 central duration for Guilty Honest as 399.50 (SD 1496.63). The data give 3909.50 (SD 1496.62), so the table value is a typo.

**Confounds reported in their Table 1.** Groups differ on age (p = .002), sex (p = .046), education (p = .021) and glasses during the task (p = .036). Naive stands out: 53.8% male and 61.5% high-school only. These variables are not in the deposited CSV, so we cannot adjust for them.

## 2. Reproduction under their protocol

Logistic regression (C = 1e8 as the stand-in for WEKA's ridge 1e-8) on the same 36 predictors, single stratified 10-fold CV with pooled predictions and weighted precision/recall/F as WEKA reports them, repeated over seeds 1–10. WEKA itself was not run.

The data checks out: group sizes match, and all 8 sampled group means match the tables, apart from rounding and the typo above. No missing values, no duplicate IDs.

| Comparison | Published AUC | Ours AUC, seed 1 | Ours AUC mean ± SD (range), seeds 1–10 | Published accuracy | Ours accuracy mean ± SD (range) |
|---|---|---|---|---|---|
| 4-class (weighted one-vs-rest AUC) | 0.63 | 0.602 | 0.602 ± 0.015 (0.563–0.622) | 40.13% | 35.4% ± 2.9 (31.2–38.9) |
| G2 vs G3 | 0.72 | 0.657 | 0.674 ± 0.033 (0.613–0.726) | 65.38% | 61.8% ± 3.3 (55.1–66.7) |
| G2 vs G1 | 0.61 | 0.670 | 0.664 ± 0.032 (0.605–0.723) | 60.26% | 62.8% ± 3.6 (56.4–67.9) |
| G2 vs G4 | 0.66 | 0.633 | 0.671 ± 0.029 (0.633–0.727) | 61.04% | 60.4% ± 2.4 (57.1–64.9) |

All three binary results fall inside the seed range. G2 vs G3, the paper's headline, sits near the top of its range: its 0.72 and 65.38% are reachable but about one SD above the seed mean, so a single-run number overstates this task somewhat. The 4-class values are not reproduced: 0.63 and 40.13% lie above the highest value from any of our 10 seeds. The likely causes are WEKA's own solver and its class-weighted AUC, and we cannot tell which without running WEKA. The ambiguity is that the paper reports no CV seed or fold assignment, so exact fold-level reproduction is impossible.

The WEKA seed-1 folds were not available, so "seed 1" here means scikit-learn's seed 1, not the same folds.

## 3. Honest baseline under our protocol

The published model (L2 logistic regression with C = 1e8 on the 36 target-AOI count and duration predictors; median imputation, in-fold 1st/99th percentile clipping and standardisation fitted on training participants only), evaluated with participant-disjoint stratified 10-fold CV, 3 repeats with different fold seeds (frozen in `folds.json`; two further seeds held back), one aggregate row per participant so the decision unit is the participant. Mean ± SD over the 3 repeats; threshold 0 on the signed score.

| task | auc | accuracy | balanced_accuracy | sensitivity | specificity | precision | f1 | mcc | eer | hter |
|---|---|---|---|---|---|---|---|---|---|---|
| guilty_vs_innocent | 0.602 ± 0.014 | 0.569 ± 0.008 | 0.569 ± 0.008 | 0.547 ± 0.016 | 0.591 ± 0.024 | 0.569 ± 0.010 | 0.558 ± 0.008 | 0.138 ± 0.016 | 0.425 ± 0.018 | 0.431 ± 0.008 |
| deceptive_vs_honest | 0.640 ± 0.012 | 0.645 ± 0.008 | 0.575 ± 0.009 | 0.439 ± 0.012 | 0.711 ± 0.008 | 0.327 ± 0.011 | 0.375 ± 0.011 | 0.137 ± 0.017 | 0.411 ± 0.009 | 0.425 ± 0.009 |
| g2_vs_g3 | 0.679 ± 0.010 | 0.615 ± 0.010 | 0.616 ± 0.011 | 0.649 ± 0.033 | 0.583 ± 0.012 | 0.597 ± 0.006 | 0.621 ± 0.018 | 0.233 ± 0.023 | 0.393 ± 0.016 | 0.384 ± 0.011 |
| g2_vs_g1 | 0.630 ± 0.045 | 0.556 ± 0.047 | 0.556 ± 0.047 | 0.570 ± 0.045 | 0.542 ± 0.051 | 0.542 ± 0.046 | 0.556 ± 0.045 | 0.112 ± 0.094 | 0.432 ± 0.044 | 0.444 ± 0.047 |
| g2_vs_g4 | 0.717 ± 0.020 | 0.654 ± 0.022 | 0.653 ± 0.022 | 0.579 ± 0.000 | 0.726 ± 0.044 | 0.675 ± 0.034 | 0.623 ± 0.015 | 0.310 ± 0.046 | 0.320 ± 0.022 | 0.347 ± 0.022 |

Notes. G2 vs G3: AUC 0.679 ± 0.010 on our folds, versus 0.72 published and 0.674 ± 0.033 in the section 2 reproduction (a different CV scheme: pooled predictions, 10 seeds). On the held-back fold seeds the same baseline gets 0.717 ± 0.017, so fold-seed luck alone moves G2 vs G3 by about 0.04. Repeat-averaged scores (used for bootstrap CIs) give 0.690 [0.567, 0.802]; do not mix them with the per-repeat means. The 4-class baseline is in section 4.

## 4. Ideas screened

All screening used identical folds (repeats 0 to 2): 18 configurations on guilty vs innocent, deceptive vs honest and G2 vs G3; 7 on G2 vs G1 and G2 vs G4; 8 on the 4-class task. Ideas: (1) other models on raw features (LR, SVM, random forest, LightGBM; 36 published vs 90 raw features), (2) within-person differential features against the neutral images (z-score, share difference, log ratio), (3) crime-area (desk AOI) features, (4) feature selection (ANOVA, L1, stability selection). Every result is kept, including failures. ΔAUC is the paired participant-bootstrap difference against the baseline on repeat-averaged scores; p is Holm-adjusted over the ideas tried on that task. The "(best)" mark is the hindsight best by mean AUC and is optimistic (see the selection-adjusted table below and summary point 2).

#### guilty_vs_innocent: G1+G2 (stole the exam) vs G3+G4

| config | idea | auc | accuracy | balanced_accuracy | sensitivity | specificity | precision | f1 | mcc | eer | hter | ΔAUC vs baseline [95% CI] | p (Holm) | acc@80% | acc@60% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_pub36_lr_unreg | honest baseline | 0.602 ± 0.014 | 0.569 ± 0.008 | 0.569 ± 0.008 | 0.547 ± 0.016 | 0.591 ± 0.024 | 0.569 ± 0.010 | 0.558 ± 0.008 | 0.138 ± 0.016 | 0.425 ± 0.018 | 0.431 ± 0.008 | — | — | 0.571 | 0.596 |
| i1_pub36_lr | 1 baseline models | 0.623 ± 0.005 | 0.586 ± 0.005 | 0.586 ± 0.005 | 0.577 ± 0.018 | 0.595 ± 0.021 | 0.585 ± 0.007 | 0.581 ± 0.008 | 0.172 ± 0.010 | 0.408 ± 0.010 | 0.414 ± 0.005 | +0.007 [-0.060, +0.080] | 1.000 | 0.603 | 0.617 |
| i1_raw90_lr | 1 baseline models | 0.597 ± 0.005 | 0.584 ± 0.006 | 0.584 ± 0.006 | 0.568 ± 0.012 | 0.599 ± 0.012 | 0.583 ± 0.006 | 0.576 ± 0.008 | 0.168 ± 0.012 | 0.418 ± 0.008 | 0.416 ± 0.006 | -0.011 [-0.108, +0.081] | 1.000 | 0.587 | 0.564 |
| i1_raw90_svm | 1 baseline models | 0.638 ± 0.026 | 0.599 ± 0.016 | 0.598 ± 0.016 | 0.551 ± 0.038 | 0.646 ± 0.027 | 0.606 ± 0.016 | 0.577 ± 0.024 | 0.198 ± 0.031 | 0.406 ± 0.011 | 0.402 ± 0.016 | +0.054 [-0.046, +0.150] | 1.000 | 0.643 | 0.670 |
| i1_raw90_rf | 1 baseline models | 0.640 ± 0.004 | 0.631 ± 0.014 | 0.630 ± 0.014 | 0.585 ± 0.032 | 0.675 ± 0.006 | 0.640 ± 0.010 | 0.611 ± 0.022 | 0.262 ± 0.027 | 0.395 ± 0.009 | 0.370 ± 0.014 | +0.044 [-0.045, +0.138] | 1.000 | 0.651 | 0.670 |
| i1_raw90_lgbm | 1 baseline models | 0.618 ± 0.023 | 0.586 ± 0.028 | 0.586 ± 0.028 | 0.556 ± 0.053 | 0.616 ± 0.012 | 0.587 ± 0.026 | 0.570 ± 0.040 | 0.172 ± 0.055 | 0.416 ± 0.026 | 0.414 ± 0.028 | +0.023 [-0.072, +0.119] | 1.000 | 0.587 | 0.638 |
| i2_diffshare_lr | 2 within-person differential | 0.658 ± 0.008 | 0.628 ± 0.003 | 0.629 ± 0.003 | 0.650 ± 0.006 | 0.608 ± 0.000 | 0.620 ± 0.002 | 0.635 ± 0.004 | 0.257 ± 0.006 | 0.378 ± 0.006 | 0.371 ± 0.003 | +0.046 [-0.067, +0.154] | 1.000 | 0.659 | 0.660 |
| i2_logratio_lr | 2 within-person differential | 0.674 ± 0.019 | 0.665 ± 0.025 | 0.665 ± 0.024 | 0.722 ± 0.012 | 0.608 ± 0.037 | 0.646 ± 0.025 | 0.682 ± 0.019 | 0.332 ± 0.048 | 0.352 ± 0.022 | 0.335 ± 0.024 | +0.072 [-0.038, +0.176] | 1.000 | 0.675 | 0.702 |
| i2_z_lr **(best)** | 2 within-person differential | 0.688 ± 0.009 | 0.641 ± 0.013 | 0.642 ± 0.013 | 0.709 ± 0.006 | 0.574 ± 0.022 | 0.622 ± 0.013 | 0.663 ± 0.010 | 0.286 ± 0.026 | 0.376 ± 0.014 | 0.358 ± 0.013 | +0.072 [-0.025, +0.169] | 1.000 | 0.667 | 0.691 |
| i2_diffshare_svm | 2 within-person differential | 0.685 ± 0.023 | 0.641 ± 0.011 | 0.641 ± 0.011 | 0.620 ± 0.016 | 0.662 ± 0.012 | 0.644 ± 0.011 | 0.632 ± 0.012 | 0.282 ± 0.022 | 0.361 ± 0.024 | 0.359 ± 0.011 | +0.099 [-0.012, +0.215] | 1.000 | 0.659 | 0.713 |
| i3_crime_lr | 3 crime-area features | 0.642 ± 0.027 | 0.597 ± 0.020 | 0.597 ± 0.020 | 0.607 ± 0.016 | 0.586 ± 0.026 | 0.592 ± 0.020 | 0.599 ± 0.018 | 0.193 ± 0.039 | 0.403 ± 0.026 | 0.403 ± 0.020 | +0.042 [-0.054, +0.137] | 1.000 | 0.619 | 0.670 |
| i3_crime_svm | 3 crime-area features | 0.669 ± 0.008 | 0.601 ± 0.017 | 0.601 ± 0.017 | 0.573 ± 0.042 | 0.629 ± 0.033 | 0.604 ± 0.017 | 0.587 ± 0.026 | 0.202 ± 0.033 | 0.393 ± 0.016 | 0.399 ± 0.017 | +0.083 [-0.015, +0.178] | 1.000 | 0.675 | 0.702 |
| i3_crime_lgbm | 3 crime-area features | 0.680 ± 0.025 | 0.643 ± 0.019 | 0.643 ± 0.019 | 0.628 ± 0.028 | 0.658 ± 0.018 | 0.645 ± 0.018 | 0.636 ± 0.022 | 0.287 ± 0.038 | 0.352 ± 0.011 | 0.357 ± 0.019 | +0.079 [-0.020, +0.181] | 1.000 | 0.651 | 0.702 |
| i3_crime_diffshare_lr | 3 crime-area features | 0.660 ± 0.022 | 0.620 ± 0.022 | 0.620 ± 0.022 | 0.641 ± 0.028 | 0.599 ± 0.026 | 0.612 ± 0.021 | 0.626 ± 0.023 | 0.240 ± 0.043 | 0.384 ± 0.023 | 0.380 ± 0.022 | +0.057 [-0.036, +0.150] | 1.000 | 0.635 | 0.670 |
| i4_all_lr_none | 4 feature selection | 0.663 ± 0.004 | 0.635 ± 0.017 | 0.635 ± 0.017 | 0.632 ± 0.022 | 0.637 ± 0.012 | 0.632 ± 0.016 | 0.632 ± 0.019 | 0.270 ± 0.033 | 0.365 ± 0.023 | 0.365 ± 0.017 | +0.060 [-0.033, +0.153] | 1.000 | 0.651 | 0.649 |
| i4_all_lr_anova | 4 feature selection | 0.649 ± 0.006 | 0.609 ± 0.008 | 0.609 ± 0.008 | 0.615 ± 0.010 | 0.603 ± 0.006 | 0.605 ± 0.007 | 0.610 ± 0.009 | 0.219 ± 0.016 | 0.389 ± 0.000 | 0.391 ± 0.008 | +0.051 [-0.047, +0.150] | 1.000 | 0.651 | 0.638 |
| i4_all_l1 | 4 feature selection | 0.602 ± 0.012 | 0.588 ± 0.024 | 0.588 ± 0.024 | 0.607 ± 0.006 | 0.570 ± 0.045 | 0.583 ± 0.026 | 0.594 ± 0.015 | 0.177 ± 0.048 | 0.425 ± 0.030 | 0.412 ± 0.024 | +0.005 [-0.087, +0.098] | 1.000 | 0.579 | 0.617 |
| i4_all_lr_stability | 4 feature selection | 0.633 ± 0.026 | 0.580 ± 0.010 | 0.580 ± 0.010 | 0.568 ± 0.026 | 0.591 ± 0.026 | 0.578 ± 0.012 | 0.573 ± 0.015 | 0.159 ± 0.021 | 0.414 ± 0.016 | 0.420 ± 0.010 | +0.031 [-0.075, +0.135] | 1.000 | 0.603 | 0.638 |

#### deceptive_vs_honest: G2 (lied) vs G1+G3+G4

| config | idea | auc | accuracy | balanced_accuracy | sensitivity | specificity | precision | f1 | mcc | eer | hter | ΔAUC vs baseline [95% CI] | p (Holm) | acc@80% | acc@60% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_pub36_lr_unreg | honest baseline | 0.640 ± 0.012 | 0.645 ± 0.008 | 0.575 ± 0.009 | 0.439 ± 0.012 | 0.711 ± 0.008 | 0.327 ± 0.011 | 0.375 ± 0.011 | 0.137 ± 0.017 | 0.411 ± 0.009 | 0.425 ± 0.009 | — | — | 0.675 | 0.766 |
| i1_pub36_lr | 1 baseline models | 0.595 ± 0.028 | 0.601 ± 0.003 | 0.549 ± 0.008 | 0.447 ± 0.021 | 0.650 ± 0.008 | 0.290 ± 0.006 | 0.352 ± 0.011 | 0.086 ± 0.013 | 0.452 ± 0.019 | 0.451 ± 0.008 | -0.025 [-0.062, +0.009] | 1.000 | 0.651 | 0.691 |
| i1_raw90_lr | 1 baseline models | 0.635 ± 0.013 | 0.641 ± 0.011 | 0.584 ± 0.020 | 0.474 ± 0.037 | 0.695 ± 0.004 | 0.331 ± 0.019 | 0.390 ± 0.025 | 0.151 ± 0.035 | 0.383 ± 0.030 | 0.416 ± 0.020 | +0.011 [-0.085, +0.114] | 1.000 | 0.722 | 0.734 |
| i1_raw90_svm | 1 baseline models | 0.581 ± 0.010 | 0.577 ± 0.042 | 0.533 ± 0.005 | 0.447 ± 0.094 | 0.619 ± 0.085 | 0.274 ± 0.005 | 0.335 ± 0.028 | 0.059 ± 0.007 | 0.445 ± 0.025 | 0.467 ± 0.005 | -0.036 [-0.154, +0.076] | 1.000 | 0.659 | 0.713 |
| i1_raw90_rf | 1 baseline models | 0.576 ± 0.029 | 0.679 ± 0.023 | 0.523 ± 0.038 | 0.219 ± 0.069 | 0.826 ± 0.014 | 0.284 ± 0.069 | 0.247 ± 0.070 | 0.049 ± 0.081 | 0.421 ± 0.016 | 0.477 ± 0.038 | -0.067 [-0.187, +0.047] | 1.000 | 0.714 | 0.777 |
| i1_raw90_lgbm | 1 baseline models | 0.579 ± 0.022 | 0.665 ± 0.011 | 0.525 ± 0.011 | 0.254 ± 0.012 | 0.796 ± 0.010 | 0.285 ± 0.020 | 0.269 ± 0.016 | 0.052 ± 0.024 | 0.434 ± 0.013 | 0.475 ± 0.011 | -0.056 [-0.180, +0.063] | 1.000 | 0.690 | 0.777 |
| i2_diffshare_lr | 2 within-person differential | 0.570 ± 0.043 | 0.535 ± 0.026 | 0.556 ± 0.047 | 0.596 ± 0.087 | 0.515 ± 0.008 | 0.281 ± 0.032 | 0.382 ± 0.047 | 0.096 ± 0.080 | 0.453 ± 0.041 | 0.444 ± 0.047 | -0.103 [-0.242, +0.031] | 1.000 | 0.556 | 0.585 |
| i2_logratio_lr | 2 within-person differential | 0.541 ± 0.030 | 0.531 ± 0.016 | 0.529 ± 0.025 | 0.526 ± 0.043 | 0.532 ± 0.008 | 0.264 ± 0.019 | 0.352 ± 0.026 | 0.050 ± 0.043 | 0.464 ± 0.030 | 0.471 ± 0.025 | -0.138 [-0.252, -0.023] | 0.408 | 0.540 | 0.564 |
| i2_z_lr **(best)** | 2 within-person differential | 0.655 ± 0.009 | 0.626 ± 0.022 | 0.592 ± 0.004 | 0.526 ± 0.037 | 0.658 ± 0.040 | 0.331 ± 0.012 | 0.405 ± 0.005 | 0.163 ± 0.008 | 0.382 ± 0.013 | 0.408 ± 0.004 | +0.029 [-0.065, +0.122] | 1.000 | 0.706 | 0.755 |
| i2_diffshare_svm | 2 within-person differential | 0.564 ± 0.012 | 0.397 ± 0.035 | 0.534 ± 0.016 | 0.798 ± 0.062 | 0.269 ± 0.062 | 0.259 ± 0.008 | 0.390 ± 0.013 | 0.067 ± 0.032 | 0.482 ± 0.013 | 0.466 ± 0.016 | -0.082 [-0.218, +0.055] | 1.000 | 0.333 | 0.330 |
| i3_crime_lr | 3 crime-area features | 0.620 ± 0.027 | 0.584 ± 0.018 | 0.567 ± 0.026 | 0.535 ± 0.054 | 0.599 ± 0.022 | 0.299 ± 0.021 | 0.383 ± 0.030 | 0.116 ± 0.045 | 0.422 ± 0.032 | 0.433 ± 0.026 | -0.019 [-0.133, +0.090] | 1.000 | 0.595 | 0.638 |
| i3_crime_svm | 3 crime-area features | 0.612 ± 0.004 | 0.446 ± 0.059 | 0.524 ± 0.027 | 0.675 ± 0.054 | 0.373 ± 0.091 | 0.257 ± 0.016 | 0.372 ± 0.016 | 0.041 ± 0.048 | 0.422 ± 0.037 | 0.476 ± 0.027 | -0.008 [-0.128, +0.103] | 1.000 | 0.508 | 0.543 |
| i3_crime_lgbm | 3 crime-area features | 0.648 ± 0.011 | 0.701 ± 0.016 | 0.588 ± 0.018 | 0.368 ± 0.021 | 0.807 ± 0.014 | 0.379 ± 0.030 | 0.374 ± 0.026 | 0.177 ± 0.037 | 0.396 ± 0.050 | 0.412 ± 0.018 | +0.017 [-0.098, +0.131] | 1.000 | 0.730 | 0.809 |
| i3_crime_diffshare_lr | 3 crime-area features | 0.563 ± 0.042 | 0.541 ± 0.019 | 0.527 ± 0.041 | 0.500 ± 0.086 | 0.555 ± 0.007 | 0.262 ± 0.032 | 0.344 ± 0.048 | 0.047 ± 0.071 | 0.479 ± 0.036 | 0.473 ± 0.041 | -0.079 [-0.194, +0.027] | 1.000 | 0.563 | 0.617 |
| i4_all_lr_none | 4 feature selection | 0.639 ± 0.018 | 0.648 ± 0.023 | 0.606 ± 0.016 | 0.526 ± 0.043 | 0.686 ± 0.039 | 0.350 ± 0.020 | 0.419 ± 0.019 | 0.190 ± 0.029 | 0.408 ± 0.036 | 0.394 ± 0.016 | -0.007 [-0.124, +0.113] | 1.000 | 0.690 | 0.702 |
| i4_all_lr_anova | 4 feature selection | 0.580 ± 0.027 | 0.601 ± 0.017 | 0.567 ± 0.026 | 0.500 ± 0.057 | 0.633 ± 0.021 | 0.303 ± 0.024 | 0.377 ± 0.033 | 0.116 ± 0.046 | 0.449 ± 0.016 | 0.433 ± 0.026 | -0.055 [-0.176, +0.058] | 1.000 | 0.635 | 0.628 |
| i4_all_l1 | 4 feature selection | 0.625 ± 0.036 | 0.626 ± 0.008 | 0.577 ± 0.023 | 0.482 ± 0.054 | 0.672 ± 0.012 | 0.319 ± 0.018 | 0.384 ± 0.030 | 0.137 ± 0.038 | 0.409 ± 0.009 | 0.423 ± 0.023 | -0.020 [-0.122, +0.085] | 1.000 | 0.690 | 0.681 |
| i4_all_lr_stability | 4 feature selection | 0.612 ± 0.027 | 0.594 ± 0.033 | 0.580 ± 0.047 | 0.553 ± 0.077 | 0.608 ± 0.024 | 0.310 ± 0.040 | 0.397 ± 0.052 | 0.139 ± 0.082 | 0.410 ± 0.035 | 0.420 ± 0.047 | -0.021 [-0.139, +0.094] | 1.000 | 0.627 | 0.713 |

#### g2_vs_g3: Guilty Deceptive vs Innocent Honest (both visited the room)

| config | idea | auc | accuracy | balanced_accuracy | sensitivity | specificity | precision | f1 | mcc | eer | hter | ΔAUC vs baseline [95% CI] | p (Holm) | acc@80% | acc@60% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_pub36_lr_unreg | honest baseline | 0.679 ± 0.010 | 0.615 ± 0.010 | 0.616 ± 0.011 | 0.649 ± 0.033 | 0.583 ± 0.012 | 0.597 ± 0.006 | 0.621 ± 0.018 | 0.233 ± 0.023 | 0.393 ± 0.016 | 0.384 ± 0.011 | — | — | 0.645 | 0.702 |
| i1_pub36_lr | 1 baseline models | 0.703 ± 0.016 | 0.684 ± 0.006 | 0.684 ± 0.006 | 0.684 ± 0.021 | 0.683 ± 0.012 | 0.672 ± 0.004 | 0.678 ± 0.011 | 0.368 ± 0.013 | 0.333 ± 0.018 | 0.316 ± 0.006 | -0.031 [-0.137, +0.080] | 1.000 | 0.661 | 0.681 |
| i1_raw90_lr | 1 baseline models | 0.743 ± 0.018 | 0.722 ± 0.006 | 0.722 ± 0.006 | 0.711 ± 0.037 | 0.733 ± 0.031 | 0.718 ± 0.014 | 0.713 ± 0.014 | 0.445 ± 0.012 | 0.286 ± 0.012 | 0.278 ± 0.006 | +0.085 [-0.022, +0.196] | 1.000 | 0.774 | 0.766 |
| i1_raw90_svm | 1 baseline models | 0.688 ± 0.018 | 0.645 ± 0.012 | 0.642 ± 0.013 | 0.526 ± 0.037 | 0.758 ± 0.012 | 0.674 ± 0.005 | 0.590 ± 0.026 | 0.293 ± 0.023 | 0.371 ± 0.021 | 0.358 ± 0.013 | +0.019 [-0.087, +0.127] | 1.000 | 0.645 | 0.681 |
| i1_raw90_rf | 1 baseline models | 0.728 ± 0.018 | 0.709 ± 0.024 | 0.709 ± 0.024 | 0.702 ± 0.033 | 0.717 ± 0.031 | 0.702 ± 0.026 | 0.702 ± 0.026 | 0.419 ± 0.048 | 0.295 ± 0.064 | 0.291 ± 0.024 | +0.048 [-0.064, +0.164] | 1.000 | 0.742 | 0.723 |
| i1_raw90_lgbm **(best)** | 1 baseline models | 0.750 ± 0.031 | 0.688 ± 0.022 | 0.688 ± 0.022 | 0.684 ± 0.037 | 0.692 ± 0.012 | 0.678 ± 0.017 | 0.681 ± 0.026 | 0.376 ± 0.044 | 0.312 ± 0.006 | 0.312 ± 0.022 | +0.068 [-0.055, +0.196] | 1.000 | 0.726 | 0.745 |
| i2_diffshare_lr | 2 within-person differential | 0.587 ± 0.031 | 0.585 ± 0.026 | 0.586 ± 0.026 | 0.605 ± 0.021 | 0.567 ± 0.031 | 0.571 ± 0.026 | 0.587 ± 0.024 | 0.172 ± 0.052 | 0.414 ± 0.024 | 0.414 ± 0.026 | -0.130 [-0.285, +0.039] | 1.000 | 0.565 | 0.596 |
| i2_logratio_lr | 2 within-person differential | 0.581 ± 0.042 | 0.585 ± 0.042 | 0.585 ± 0.043 | 0.579 ± 0.057 | 0.592 ± 0.031 | 0.573 ± 0.040 | 0.576 ± 0.048 | 0.171 ± 0.085 | 0.406 ± 0.053 | 0.415 ± 0.043 | -0.164 [-0.307, -0.013] | 0.646 | 0.516 | 0.553 |
| i2_z_lr | 2 within-person differential | 0.670 ± 0.026 | 0.632 ± 0.022 | 0.634 ± 0.021 | 0.684 ± 0.000 | 0.583 ± 0.042 | 0.610 ± 0.024 | 0.645 ± 0.013 | 0.269 ± 0.042 | 0.359 ± 0.038 | 0.366 ± 0.021 | -0.047 [-0.187, +0.093] | 1.000 | 0.629 | 0.681 |
| i2_diffshare_svm | 2 within-person differential | 0.577 ± 0.018 | 0.513 ± 0.046 | 0.506 ± 0.047 | 0.228 ± 0.108 | 0.783 ± 0.051 | 0.477 ± 0.094 | 0.302 ± 0.120 | 0.005 ± 0.108 | 0.449 ± 0.021 | 0.494 ± 0.047 | -0.126 [-0.287, +0.037] | 1.000 | 0.548 | 0.532 |
| i3_crime_lr | 3 crime-area features | 0.730 ± 0.011 | 0.688 ± 0.012 | 0.688 ± 0.012 | 0.693 ± 0.012 | 0.683 ± 0.012 | 0.675 ± 0.012 | 0.684 ± 0.012 | 0.376 ± 0.024 | 0.308 ± 0.010 | 0.312 ± 0.012 | +0.028 [-0.086, +0.153] | 1.000 | 0.677 | 0.745 |
| i3_crime_svm | 3 crime-area features | 0.714 ± 0.033 | 0.585 ± 0.016 | 0.577 ± 0.017 | 0.263 ± 0.077 | 0.892 ± 0.047 | 0.706 ± 0.034 | 0.375 ± 0.075 | 0.202 ± 0.027 | 0.355 ± 0.063 | 0.423 ± 0.017 | +0.036 [-0.094, +0.177] | 1.000 | 0.629 | 0.681 |
| i3_crime_lgbm | 3 crime-area features | 0.709 ± 0.015 | 0.624 ± 0.016 | 0.624 ± 0.016 | 0.614 ± 0.033 | 0.633 ± 0.031 | 0.614 ± 0.018 | 0.614 ± 0.019 | 0.248 ± 0.032 | 0.380 ± 0.012 | 0.376 ± 0.016 | +0.032 [-0.105, +0.167] | 1.000 | 0.677 | 0.745 |
| i3_crime_diffshare_lr | 3 crime-area features | 0.698 ± 0.041 | 0.658 ± 0.037 | 0.658 ± 0.037 | 0.649 ± 0.054 | 0.667 ± 0.024 | 0.648 ± 0.034 | 0.648 ± 0.044 | 0.316 ± 0.074 | 0.338 ± 0.042 | 0.342 ± 0.037 | +0.026 [-0.098, +0.161] | 1.000 | 0.677 | 0.723 |
| i4_all_lr_none | 4 feature selection | 0.724 ± 0.016 | 0.688 ± 0.006 | 0.688 ± 0.007 | 0.675 ± 0.045 | 0.700 ± 0.035 | 0.682 ± 0.012 | 0.678 ± 0.017 | 0.377 ± 0.013 | 0.303 ± 0.030 | 0.312 ± 0.007 | +0.045 [-0.056, +0.152] | 1.000 | 0.758 | 0.745 |
| i4_all_lr_anova | 4 feature selection | 0.728 ± 0.013 | 0.654 ± 0.028 | 0.654 ± 0.028 | 0.675 ± 0.045 | 0.633 ± 0.012 | 0.636 ± 0.022 | 0.655 ± 0.033 | 0.309 ± 0.057 | 0.355 ± 0.034 | 0.346 ± 0.028 | +0.035 [-0.077, +0.155] | 1.000 | 0.726 | 0.745 |
| i4_all_l1 | 4 feature selection | 0.729 ± 0.009 | 0.697 ± 0.012 | 0.697 ± 0.012 | 0.719 ± 0.012 | 0.675 ± 0.020 | 0.678 ± 0.014 | 0.698 ± 0.011 | 0.395 ± 0.024 | 0.320 ± 0.021 | 0.303 ± 0.012 | +0.057 [-0.068, +0.177] | 1.000 | 0.726 | 0.787 |
| i4_all_lr_stability | 4 feature selection | 0.700 ± 0.019 | 0.641 ± 0.010 | 0.641 ± 0.011 | 0.632 ± 0.021 | 0.650 ± 0.000 | 0.631 ± 0.008 | 0.631 ± 0.015 | 0.282 ± 0.021 | 0.368 ± 0.006 | 0.359 ± 0.011 | +0.028 [-0.094, +0.159] | 1.000 | 0.710 | 0.745 |

#### g2_vs_g1: Guilty Deceptive vs Guilty Honest (published pair)

| config | idea | auc | accuracy | balanced_accuracy | sensitivity | specificity | precision | f1 | mcc | eer | hter | ΔAUC vs baseline [95% CI] | p (Holm) | acc@80% | acc@60% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_pub36_lr_unreg **(best)** | honest baseline | 0.630 ± 0.045 | 0.556 ± 0.047 | 0.556 ± 0.047 | 0.570 ± 0.045 | 0.542 ± 0.051 | 0.542 ± 0.046 | 0.556 ± 0.045 | 0.112 ± 0.094 | 0.432 ± 0.044 | 0.444 ± 0.047 | — | — | 0.597 | 0.702 |
| i1_pub36_lr | 1 baseline models | 0.586 ± 0.017 | 0.564 ± 0.046 | 0.564 ± 0.045 | 0.553 ± 0.037 | 0.575 ± 0.054 | 0.554 ± 0.047 | 0.553 ± 0.042 | 0.128 ± 0.091 | 0.453 ± 0.053 | 0.436 ± 0.045 | -0.036 [-0.084, +0.012] | 0.632 | 0.597 | 0.660 |
| i1_raw90_lgbm | 1 baseline models | 0.446 ± 0.060 | 0.457 ± 0.042 | 0.458 ± 0.041 | 0.482 ± 0.025 | 0.433 ± 0.082 | 0.450 ± 0.037 | 0.465 ± 0.025 | -0.085 ± 0.084 | 0.517 ± 0.040 | 0.542 ± 0.041 | -0.203 [-0.347, -0.058] | 0.078 | 0.484 | 0.468 |
| i2_diffshare_lr | 2 within-person differential | 0.607 ± 0.022 | 0.632 ± 0.006 | 0.634 ± 0.006 | 0.675 ± 0.033 | 0.592 ± 0.031 | 0.611 ± 0.008 | 0.641 ± 0.013 | 0.268 ± 0.012 | 0.359 ± 0.027 | 0.366 ± 0.006 | -0.024 [-0.158, +0.111] | 0.782 | 0.694 | 0.617 |
| i3_crime_lr | 3 crime-area features | 0.567 ± 0.026 | 0.585 ± 0.042 | 0.585 ± 0.042 | 0.553 ± 0.037 | 0.617 ± 0.047 | 0.579 ± 0.046 | 0.565 ± 0.041 | 0.170 ± 0.085 | 0.419 ± 0.030 | 0.415 ± 0.042 | -0.091 [-0.230, +0.049] | 0.632 | 0.565 | 0.532 |
| i3_crime_diffshare_lr | 3 crime-area features | 0.554 ± 0.047 | 0.573 ± 0.026 | 0.572 ± 0.026 | 0.561 ± 0.025 | 0.583 ± 0.031 | 0.562 ± 0.028 | 0.561 ± 0.026 | 0.145 ± 0.053 | 0.436 ± 0.018 | 0.428 ± 0.026 | -0.109 [-0.243, +0.022] | 0.515 | 0.581 | 0.511 |
| i4_all_lr_anova | 4 feature selection | 0.564 ± 0.034 | 0.564 ± 0.028 | 0.564 ± 0.027 | 0.570 ± 0.012 | 0.558 ± 0.042 | 0.552 ± 0.029 | 0.561 ± 0.021 | 0.129 ± 0.055 | 0.436 ± 0.031 | 0.436 ± 0.027 | -0.058 [-0.189, +0.081] | 0.782 | 0.565 | 0.574 |

#### g2_vs_g4: Guilty Deceptive vs Naive (published pair)

| config | idea | auc | accuracy | balanced_accuracy | sensitivity | specificity | precision | f1 | mcc | eer | hter | ΔAUC vs baseline [95% CI] | p (Holm) | acc@80% | acc@60% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_pub36_lr_unreg **(best)** | honest baseline | 0.717 ± 0.020 | 0.654 ± 0.022 | 0.653 ± 0.022 | 0.579 ± 0.000 | 0.726 ± 0.044 | 0.675 ± 0.034 | 0.623 ± 0.015 | 0.310 ± 0.046 | 0.320 ± 0.022 | 0.347 ± 0.022 | — | — | 0.726 | 0.739 |
| i1_pub36_lr | 1 baseline models | 0.657 ± 0.017 | 0.628 ± 0.034 | 0.627 ± 0.034 | 0.605 ± 0.021 | 0.650 ± 0.053 | 0.629 ± 0.041 | 0.617 ± 0.029 | 0.255 ± 0.068 | 0.390 ± 0.011 | 0.373 ± 0.034 | -0.063 [-0.132, +0.003] | 0.372 | 0.677 | 0.717 |
| i1_raw90_lgbm | 1 baseline models | 0.647 ± 0.030 | 0.597 ± 0.018 | 0.597 ± 0.019 | 0.544 ± 0.069 | 0.650 ± 0.032 | 0.601 ± 0.011 | 0.569 ± 0.044 | 0.195 ± 0.036 | 0.416 ± 0.032 | 0.403 ± 0.019 | -0.096 [-0.237, +0.041] | 0.570 | 0.629 | 0.696 |
| i2_diffshare_lr | 2 within-person differential | 0.664 ± 0.021 | 0.623 ± 0.032 | 0.623 ± 0.032 | 0.605 ± 0.037 | 0.641 ± 0.036 | 0.622 ± 0.034 | 0.613 ± 0.034 | 0.247 ± 0.064 | 0.381 ± 0.033 | 0.377 ± 0.032 | -0.088 [-0.239, +0.068] | 0.570 | 0.661 | 0.630 |
| i3_crime_lr | 3 crime-area features | 0.612 ± 0.025 | 0.593 ± 0.027 | 0.593 ± 0.026 | 0.605 ± 0.021 | 0.581 ± 0.048 | 0.586 ± 0.028 | 0.595 ± 0.021 | 0.187 ± 0.053 | 0.394 ± 0.006 | 0.407 ± 0.026 | -0.117 [-0.259, +0.028] | 0.570 | 0.597 | 0.630 |
| i3_crime_diffshare_lr | 3 crime-area features | 0.649 ± 0.037 | 0.645 ± 0.048 | 0.645 ± 0.048 | 0.649 ± 0.087 | 0.641 ± 0.042 | 0.636 ± 0.043 | 0.641 ± 0.062 | 0.291 ± 0.096 | 0.364 ± 0.066 | 0.355 ± 0.048 | -0.094 [-0.238, +0.054] | 0.570 | 0.645 | 0.630 |
| i4_all_lr_anova | 4 feature selection | 0.623 ± 0.015 | 0.606 ± 0.006 | 0.606 ± 0.006 | 0.623 ± 0.012 | 0.590 ± 0.021 | 0.597 ± 0.008 | 0.609 ± 0.004 | 0.213 ± 0.012 | 0.398 ± 0.012 | 0.394 ± 0.006 | -0.123 [-0.270, +0.032] | 0.570 | 0.629 | 0.609 |

#### 4-class (G1/G2/G3/G4): published AUC 0.63, accuracy 40.13%

| config | macro_auc_ovr | weighted_auc_ovr | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | macro_eer_ovr | macro_hter_ovr_argmax | correct per group (repeat 0) |
|---|---|---|---|---|---|---|---|---|---|
| base_pub36_lr_unreg | 0.590 ± 0.003 | 0.590 ± 0.003 | 0.335 ± 0.011 | 0.336 ± 0.011 | 0.335 ± 0.011 | 0.335 ± 0.011 | 0.427 ± 0.006 | 0.443 ± 0.007 | {'1': 9, '2': 14, '3': 17, '4': 15} |
| i1_pub36_lr | 0.617 ± 0.014 | 0.617 ± 0.013 | 0.372 ± 0.023 | 0.371 ± 0.023 | 0.369 ± 0.023 | 0.370 ± 0.024 | 0.394 ± 0.019 | 0.419 ± 0.016 | {'1': 12, '2': 13, '3': 18, '4': 11} |
| i1_raw90_lr | 0.639 ± 0.008 | 0.639 ± 0.008 | 0.418 ± 0.006 | 0.418 ± 0.006 | 0.416 ± 0.006 | 0.415 ± 0.006 | 0.401 ± 0.013 | 0.388 ± 0.004 | {'1': 13, '2': 16, '3': 20, '4': 16} |
| i1_raw90_lgbm | 0.618 ± 0.016 | 0.618 ± 0.015 | 0.361 ± 0.011 | 0.360 ± 0.011 | 0.357 ± 0.010 | 0.358 ± 0.009 | 0.407 ± 0.009 | 0.426 ± 0.007 | {'1': 12, '2': 12, '3': 19, '4': 13} |
| i2_diffshare_lr | 0.599 ± 0.010 | 0.599 ± 0.010 | 0.361 ± 0.023 | 0.361 ± 0.023 | 0.361 ± 0.023 | 0.361 ± 0.023 | 0.418 ± 0.006 | 0.426 ± 0.016 | {'1': 15, '2': 13, '3': 14, '4': 19} |
| i3_crime_lr | 0.634 ± 0.007 | 0.633 ± 0.007 | 0.382 ± 0.014 | 0.382 ± 0.014 | 0.380 ± 0.014 | 0.380 ± 0.014 | 0.407 ± 0.009 | 0.412 ± 0.009 | {'1': 12, '2': 15, '3': 19, '4': 15} |
| i3_crime_diffshare_lr | 0.639 ± 0.001 | 0.639 ± 0.001 | 0.397 ± 0.027 | 0.397 ± 0.027 | 0.396 ± 0.027 | 0.395 ± 0.027 | 0.395 ± 0.007 | 0.402 ± 0.018 | {'1': 13, '2': 16, '3': 19, '4': 20} |
| i4_all_lr_anova | 0.609 ± 0.009 | 0.609 ± 0.008 | 0.352 ± 0.011 | 0.353 ± 0.011 | 0.344 ± 0.008 | 0.343 ± 0.008 | 0.420 ± 0.015 | 0.431 ± 0.007 | {'1': 6, '2': 13, '3': 23, '4': 15} |

Paired participant bootstrap of the 4-class configurations against the baseline (repeat-averaged probabilities, 1000 stratified resamples, Holm over 7 ideas on accuracy; `fourclass_paired.json`, `scripts/phase2_mazza_4class_paired.py`). The baseline has accuracy 0.363 and macro AUC 0.599 on repeat-averaged probabilities.

| config | accuracy (avg scores) | Δ accuracy [95% CI] | Δ macro AUC [95% CI] | Holm p (accuracy) |
|---|---|---|---|---|
| i1_pub36_lr | 0.395 | +0.032 [-0.032, +0.102] | +0.028 [-0.020, +0.073] | 1.000 |
| i1_raw90_lr | 0.446 | +0.083 [+0.000, +0.166] | +0.050 [-0.005, +0.110] | 0.434 |
| i1_raw90_lgbm | 0.389 | +0.025 [-0.064, +0.121] | +0.030 [-0.033, +0.096] | 1.000 |
| i2_diffshare_lr | 0.389 | +0.025 [-0.057, +0.121] | -0.001 [-0.063, +0.069] | 1.000 |
| i3_crime_lr | 0.395 | +0.032 [-0.051, +0.115] | +0.036 [-0.020, +0.090] | 1.000 |
| i3_crime_diffshare_lr | 0.408 | +0.045 [-0.038, +0.134] | +0.039 [-0.024, +0.100] | 1.000 |
| i4_all_lr_anova | 0.382 | +0.019 [-0.064, +0.108] | +0.019 [-0.043, +0.080] | 1.000 |

The raw-90-feature LR (accuracy 0.418 ± 0.006 per repeat, macro AUC 0.639) is the nominal best and its accuracy interval just touches 0, but after Holm p = 0.434. The published 40.13% is reachable by some of our configurations, but none is reliably better than the baseline, and the published LR itself did not reproduce (section 2).

#### Selection-adjusted best (config chosen per outer fold on the other 9 folds)

| task | candidates | hindsight best | AUC | accuracy | balanced acc | EER | HTER | AUC 95% CI (avg scores) | picks |
|---|---|---|---|---|---|---|---|---|---|
| guilty_vs_innocent | 18 | i2_z_lr | 0.654 ± 0.023 | 0.607 ± 0.022 | 0.607 ± 0.021 | 0.401 ± 0.010 | 0.393 ± 0.021 | [0.595, 0.759] | {'i2_z_lr': 12, 'i3_crime_lgbm': 9, 'i2_diffshare_svm': 6, 'i2_logratio_lr': 2, 'i1_raw90_svm': 1} |
| deceptive_vs_honest | 18 | i2_z_lr | 0.574 ± 0.028 | 0.609 ± 0.030 | 0.527 ± 0.012 | 0.444 ± 0.044 | 0.473 ± 0.012 | [0.477, 0.678] | {'i3_crime_lgbm': 9, 'i2_z_lr': 7, 'i4_all_l1': 5, 'base_pub36_lr_unreg': 4, 'i4_all_lr_none': 2, 'i1_raw90_lr': 2, 'i3_crime_lr': 1} |
| g2_vs_g3 | 18 | i1_raw90_lgbm | 0.673 ± 0.079 | 0.650 ± 0.048 | 0.649 ± 0.049 | 0.368 ± 0.042 | 0.351 ± 0.049 | [0.574, 0.808] | {'i1_raw90_lgbm': 15, 'i4_all_lr_anova': 4, 'i1_raw90_lr': 4, 'i3_crime_svm': 2, 'i4_all_lr_none': 2, 'i3_crime_diffshare_lr': 1, 'i1_raw90_rf': 1, 'i3_crime_lr': 1} |
| g2_vs_g1 | 7 | base_pub36_lr_unreg | 0.575 ± 0.099 | 0.551 ± 0.073 | 0.551 ± 0.074 | 0.440 ± 0.097 | 0.449 ± 0.074 | [0.443, 0.698] | {'base_pub36_lr_unreg': 16, 'i2_diffshare_lr': 10, 'i4_all_lr_anova': 4} |
| g2_vs_g4 | 7 | base_pub36_lr_unreg | 0.642 ± 0.082 | 0.602 ± 0.071 | 0.601 ± 0.071 | 0.364 ± 0.059 | 0.399 ± 0.071 | [0.554, 0.788] | {'base_pub36_lr_unreg': 24, 'i1_raw90_lgbm': 4, 'i2_diffshare_lr': 2} |

In each outer fold the configuration is chosen on the other nine folds (score divided by within-repeat SD), so the reported AUC is what a person without hindsight would get. This is the honest number to compare with the baseline and the published values: guilty vs innocent 0.654 (baseline 0.602), deceptive vs honest 0.574 (0.640), **G2 vs G3 0.673 (0.679; published 0.72)**, G2 vs G1 0.575 (0.630), G2 vs G4 0.642 (0.717). The selected pipeline is not better than the fixed baseline on any task except guilty vs innocent, where it is +0.052 and inside the noise.

## 5. Best candidates vs published vs honest baseline

Two candidates: **A, i2_z_lr** (guilty vs innocent; also best on deceptive vs honest) because it is the only consistent gain, and **B, i1_raw90_lgbm** (G2 vs G3), the hindsight best on the published headline pair.

#### Best vs baseline with participant-bootstrap 95% CI (repeat-averaged scores)

| task | config | AUC | accuracy | balanced acc | EER | HTER | MCC |
|---|---|---|---|---|---|---|---|
| guilty_vs_innocent | i2_z_lr | 0.677 [0.592, 0.761] | 0.643 [0.567, 0.713] | 0.644 [0.567, 0.714] | 0.376 [0.293, 0.446] | 0.356 [0.286, 0.433] | 0.290 [0.136, 0.433] |
| guilty_vs_innocent | base_pub36_lr_unreg | 0.605 [0.517, 0.691] | 0.567 [0.490, 0.643] | 0.567 [0.490, 0.643] | 0.439 [0.350, 0.510] | 0.433 [0.357, 0.510] | 0.134 [-0.021, 0.287] |
| deceptive_vs_honest | i2_z_lr | 0.677 [0.580, 0.770] | 0.662 [0.586, 0.732] | 0.616 [0.525, 0.707] | 0.374 [0.266, 0.430] | 0.384 [0.293, 0.475] | 0.208 [0.045, 0.367] |
| deceptive_vs_honest | base_pub36_lr_unreg | 0.648 [0.548, 0.743] | 0.637 [0.561, 0.707] | 0.572 [0.484, 0.660] | 0.442 [0.335, 0.515] | 0.428 [0.340, 0.516] | 0.131 [-0.029, 0.288] |
| g2_vs_g3 | i1_raw90_lgbm | 0.759 [0.647, 0.864] | 0.692 [0.590, 0.795] | 0.693 [0.591, 0.795] | 0.295 [0.192, 0.423] | 0.307 [0.205, 0.409] | 0.386 [0.182, 0.590] |
| g2_vs_g3 | base_pub36_lr_unreg | 0.690 [0.567, 0.802] | 0.641 [0.526, 0.744] | 0.642 [0.528, 0.744] | 0.411 [0.281, 0.500] | 0.358 [0.256, 0.472] | 0.285 [0.058, 0.488] |
| g2_vs_g1 | base_pub36_lr_unreg | 0.639 [0.509, 0.760] | 0.564 [0.449, 0.667] | 0.565 [0.451, 0.669] | 0.462 [0.308, 0.551] | 0.435 [0.331, 0.549] | 0.131 [-0.100, 0.344] |
| g2_vs_g4 | base_pub36_lr_unreg | 0.747 [0.628, 0.852] | 0.675 [0.571, 0.779] | 0.674 [0.571, 0.778] | 0.312 [0.195, 0.416] | 0.326 [0.222, 0.429] | 0.352 [0.144, 0.560] |

#### Permutation tests (labels shuffled between participants, full nested pipeline, repeat 0 folds)

| task | config | observed AUC | null mean | null 95th pct | p | n_perm |
|---|---|---|---|---|---|---|
| guilty_vs_innocent | i2_z_lr | 0.680 | 0.488 | 0.565 | 0.0312 | 31 |
| deceptive_vs_honest | i2_z_lr | 0.654 | 0.496 | 0.651 | 0.0645 | 30 |
| g2_vs_g3 | i1_raw90_lgbm | 0.784 | 0.477 | 0.630 | 0.0323 | 30 |

The permutation null shuffles labels between participants and reruns the full nested pipeline on the repeat-0 folds. Only 30 to 31 permutations were affordable on the shared CPU (about 6 to 12 minutes per task), so the smallest attainable p is about 0.031 and the p-values are coarse. They test the already chosen configuration and do not account for choosing it among 18.

| candidate | task | observed AUC (repeat 0) | null mean | null 95th pct | p | reading |
|---|---|---|---|---|---|---|
| A i2_z_lr | guilty vs innocent | 0.680 | 0.488 | 0.565 | 0.0312 (31 perms) | above chance; minimum possible p; the CI of the gain over baseline includes 0 |
| A i2_z_lr | deceptive vs honest | 0.654 | 0.496 | 0.651 | 0.0645 (30 perms) | not significant at .05; observed AUC equals the null 95th percentile |
| B i1_raw90_lgbm | G2 vs G3 | 0.784 | 0.477 | 0.630 | 0.0323 (30 perms) | above chance for the hindsight-picked config; says nothing about beating the baseline |

No permutation test was run for the baseline itself, for G2 vs G1 / G2 vs G4 (the baseline is the best there) or for the 4-class task.

**Deltas against the baseline (repeat-averaged participant AUC, paired participant bootstrap).** A on guilty vs innocent: +0.072 [-0.025, +0.169]. A on deceptive vs honest: +0.029 [-0.065, +0.122]. B on G2 vs G3: +0.068 [-0.055, +0.196]. Against the published 0.72 for G2 vs G3: B 0.750 (hindsight) and 0.673 (selected on other folds), baseline 0.679. No published value exists for guilty vs innocent or deceptive vs honest.

**Why A helps (mechanism, with caveats).** Each target-image count or duration is z-scored against the same person's values on the six neutral images, which removes person-level differences in how much people look at the left, centre or right third of any image. The model keeps the article's desk finding: top coefficients are Target 1 left count (-0.097; the desk, so guilty people look away from it) and Target 1 central duration (+0.097) and count (+0.095). Ablations on guilty vs innocent (L2 LR on one raw block): neutral images only 0.484 [0.401, 0.586], target overall-image features only 0.474, scanpath features only 0.523, all non-neutral raw features 0.597; so the signal lives in the target-image AOI values and the z-score step adds about 0.09 to them. The mechanism does not transfer: i2_z_lr on G2 vs G3 is 0.670 (ΔAUC -0.047 [-0.187, +0.093]); share difference and log ratio hurt G2 vs G3 (0.587 and 0.581) and log ratio hurts deceptive vs honest (0.541). Clipping is neutral (guilty vs innocent 0.693 without vs 0.688 with; deceptive vs honest 0.676 vs 0.655; G2 vs G3 0.734 vs 0.750; all within SD).

**Why B scores high (and why it is not a clean gain).** LightGBM on all 90 raw features (including the neutral images and scanpath features that the article does not use) reaches 0.750 in hindsight. Its top univariate features are Target 1 central duration (AUC 0.743) and count (0.708), the same desk-avoidance signal the LR uses, plus Neutral recognised 4 overall duration (0.690). Block ablation for G2 vs G3: neutral images only 0.664 [0.537, 0.782], scanpath only 0.716, everything except the neutral images 0.723 [0.594, 0.830]. Much of the G2 vs G3 signal is therefore available without the target images, which argues against a crime-specific reading. B's probabilities are poorly calibrated (raw ECE 0.139, Platt ECE 0.135).

**Held-back fold seeds (same participants, two other fold assignments).**

#### Held-back fold seeds (repeats 3-4, same participants)

| task | config | AUC screening | AUC held-back seeds | BA held-back | EER held-back |
|---|---|---|---|---|---|
| guilty_vs_innocent | i2_z_lr | 0.688 ± 0.009 | 0.673 ± 0.005 | 0.644 ± 0.000 | 0.376 ± 0.006 |
| guilty_vs_innocent | base_pub36_lr_unreg | 0.602 ± 0.014 | 0.609 ± 0.006 | 0.563 ± 0.010 | 0.420 ± 0.000 |
| deceptive_vs_honest | i2_z_lr | 0.655 ± 0.009 | 0.649 ± 0.018 | 0.587 ± 0.006 | 0.393 ± 0.036 |
| deceptive_vs_honest | base_pub36_lr_unreg | 0.640 ± 0.012 | 0.660 ± 0.006 | 0.583 ± 0.015 | 0.410 ± 0.015 |
| g2_vs_g3 | i1_raw90_lgbm | 0.750 ± 0.031 | 0.770 ± 0.016 | 0.717 ± 0.000 | 0.282 ± 0.013 |
| g2_vs_g3 | base_pub36_lr_unreg | 0.679 ± 0.010 | 0.717 ± 0.017 | 0.659 ± 0.006 | 0.327 ± 0.007 |

These control fold-assignment luck only. The participants are the same, so they do not remove selection bias: B rises from 0.750 to 0.770 and the baseline from 0.679 to 0.717 (gap 0.071 shrinks to 0.053). For deceptive vs honest the baseline (0.660) is slightly above i2_z_lr (0.649).

#### Block ablation: L2-LR on one raw block (neutral = negative control)

| task | block | AUC | AUC 95% CI | balanced acc | EER |
|---|---|---|---|---|---|
| guilty_vs_innocent | neutral | 0.484 ± 0.039 | [0.401, 0.586] | 0.476 ± 0.024 | 0.527 ± 0.026 |
| guilty_vs_innocent | target_overall | 0.474 ± 0.020 | [0.384, 0.564] | 0.477 ± 0.009 | 0.507 ± 0.018 |
| guilty_vs_innocent | scanpath | 0.523 ± 0.018 | [0.450, 0.626] | 0.554 ± 0.021 | 0.452 ± 0.016 |
| guilty_vs_innocent | no_neutral | 0.597 ± 0.016 | [0.481, 0.662] | 0.577 ± 0.016 | 0.403 ± 0.013 |
| deceptive_vs_honest | neutral | 0.659 ± 0.027 | [0.557, 0.765] | 0.650 ± 0.010 | 0.335 ± 0.015 |
| deceptive_vs_honest | target_overall | 0.534 ± 0.020 | [0.431, 0.635] | 0.506 ± 0.041 | 0.475 ± 0.026 |
| deceptive_vs_honest | scanpath | 0.560 ± 0.046 | [0.459, 0.660] | 0.563 ± 0.044 | 0.442 ± 0.035 |
| deceptive_vs_honest | no_neutral | 0.655 ± 0.028 | [0.587, 0.771] | 0.573 ± 0.018 | 0.415 ± 0.018 |
| g2_vs_g3 | neutral | 0.664 ± 0.012 | [0.537, 0.782] | 0.688 ± 0.022 | 0.307 ± 0.018 |
| g2_vs_g3 | target_overall | 0.554 ± 0.024 | [0.428, 0.686] | 0.556 ± 0.016 | 0.436 ± 0.018 |
| g2_vs_g3 | scanpath | 0.716 ± 0.030 | [0.615, 0.836] | 0.628 ± 0.022 | 0.380 ± 0.047 |
| g2_vs_g3 | no_neutral | 0.723 ± 0.005 | [0.594, 0.830] | 0.687 ± 0.016 | 0.312 ± 0.024 |

#### Outlier clipping ablation (best config without the in-fold 1st/99th percentile clipping)

| task | config | AUC with clip | AUC without clip | EER with | EER without |
|---|---|---|---|---|---|
| guilty_vs_innocent | i2_z_lr | 0.688 ± 0.009 | 0.693 ± 0.010 | 0.376 ± 0.014 | 0.382 ± 0.023 |
| deceptive_vs_honest | i2_z_lr | 0.655 ± 0.009 | 0.676 ± 0.048 | 0.382 ± 0.013 | 0.372 ± 0.037 |
| g2_vs_g3 | i1_raw90_lgbm | 0.750 ± 0.031 | 0.734 ± 0.018 | 0.312 ± 0.006 | 0.334 ± 0.018 |
| g2_vs_g1 | base_pub36_lr_unreg | 0.630 ± 0.045 | 0.662 ± 0.036 | 0.432 ± 0.044 | 0.406 ± 0.022 |
| g2_vs_g4 | base_pub36_lr_unreg | 0.717 ± 0.020 | 0.717 ± 0.008 | 0.320 ± 0.022 | 0.346 ± 0.012 |

#### Calibration and selective prediction (mean ± SD over repeats)

| task | config | Brier raw | ECE raw | Brier Platt | ECE Platt | acc@100% | acc@80% | acc@60% | BA@80% | BA@60% |
|---|---|---|---|---|---|---|---|---|---|---|
| guilty_vs_innocent | i2_z_lr | 0.230 ± 0.005 | 0.094 ± 0.015 | 0.226 ± 0.002 | 0.063 ± 0.003 | 0.641 ± 0.013 | 0.669 ± 0.010 | 0.702 ± 0.015 | 0.672 ± 0.009 | 0.704 ± 0.015 |
| guilty_vs_innocent | base_pub36_lr_unreg | 0.303 ± 0.009 | 0.233 ± 0.009 | 0.244 ± 0.002 | 0.038 ± 0.008 | 0.569 ± 0.008 | 0.582 ± 0.004 | 0.599 ± 0.013 | 0.582 ± 0.005 | 0.599 ± 0.013 |
| deceptive_vs_honest | i2_z_lr | 0.241 ± 0.006 | 0.205 ± 0.010 | 0.174 ± 0.004 | 0.039 ± 0.019 | 0.626 ± 0.022 | 0.669 ± 0.016 | 0.734 ± 0.023 | 0.614 ± 0.015 | 0.671 ± 0.021 |
| deceptive_vs_honest | base_pub36_lr_unreg | 0.276 ± 0.002 | 0.272 ± 0.006 | 0.178 ± 0.002 | 0.044 ± 0.015 | 0.645 ± 0.008 | 0.677 ± 0.004 | 0.723 ± 0.026 | 0.601 ± 0.006 | 0.650 ± 0.020 |
| g2_vs_g3 | i1_raw90_lgbm | 0.215 ± 0.019 | 0.139 ± 0.012 | 0.210 ± 0.005 | 0.135 ± 0.051 | 0.688 ± 0.022 | 0.737 ± 0.033 | 0.766 ± 0.030 | 0.737 ± 0.034 | 0.767 ± 0.031 |
| g2_vs_g3 | base_pub36_lr_unreg | 0.383 ± 0.011 | 0.383 ± 0.012 | 0.235 ± 0.001 | 0.063 ± 0.045 | 0.615 ± 0.010 | 0.618 ± 0.008 | 0.652 ± 0.036 | 0.620 ± 0.009 | 0.652 ± 0.038 |
| g2_vs_g1 | base_pub36_lr_unreg | 0.442 ± 0.048 | 0.444 ± 0.048 | 0.252 ± 0.003 | 0.057 ± 0.018 | 0.556 ± 0.047 | 0.597 ± 0.040 | 0.638 ± 0.035 | 0.598 ± 0.040 | 0.636 ± 0.034 |
| g2_vs_g4 | base_pub36_lr_unreg | 0.342 ± 0.024 | 0.348 ± 0.025 | 0.241 ± 0.004 | 0.084 ± 0.016 | 0.654 ± 0.022 | 0.688 ± 0.015 | 0.710 ± 0.020 | 0.686 ± 0.015 | 0.709 ± 0.021 |

Raw logistic-regression probabilities of the baseline are badly overconfident (ECE 0.23 to 0.44; unregularised C = 1e8 fit); Platt scaling fitted inside the training folds brings ECE to 0.04 to 0.08. Selective prediction (keep the most confident participants) raises accuracy, e.g. A on guilty vs innocent 0.641 at full coverage, 0.669 at 80% and 0.702 at 60%; the baseline goes 0.569, 0.582, 0.599. This is risk-coverage evaluation, not a deployed threshold.

**Caveats.** n = 157 (about 40 per group; 78 guilty vs 79 innocent); single-AUC bootstrap CIs are about ±0.09; the 3 fold repeats do not reduce participant sampling variance; one dataset and no external validation; features are RealEye's own aggregates; group differences in age, sex, education and glasses are confounded and cannot be adjusted for with the deposited CSV; the image order is fixed for all participants, so order effects cannot be separated from group effects; the permutation tests are small and do not include selection.

## 6. Efficiency

Single-threaded on this laptop (`edd.efficiency`): {'platform': 'Windows-11-10.0.26200-SP0', 'processor': 'Intel64 Family 6 Model 142 Stepping 10, GenuineIntel', 'python': '3.14.6', 'cpu_count': 8}. Pipelines are the final nested-search winners refit on all participants of the task. Latency is per participant (the decision unit), end-to-end through the scikit-learn pipeline (imputer, clipping, scaling, classifier) for one row, 500 runs. Throughput is a batch of all 157 participants. Peak memory is the extra allocation during one inference (tracemalloc). Model size is the pickled pipeline. "Preprocessing params" is the approximate count of stored imputer, scaler and clip bounds. The CPU was shared with other experiments during measurement, so p95 values are noisy.

| model | task | inputs | latency p50 / p95 ms | feature step p50 ms | throughput participants/s | peak mem KB | model size KB | classifier params | preprocessing params | nested fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline: published LR re-implementation | g2_vs_g3 | 36 | 2.65 / 5.39 | 0.76 | 19718 | 15.6 | 8.9 | 37 | 180 | 0.4 |
| A: i2_z_lr | guilty_vs_innocent | 36 | 2.70 / 3.25 | 0.80 | 21783 | 12.6 | 8.2 | 37 | 180 | 1.2 |
| A: i2_z_lr | deceptive_vs_honest | 36 | 2.64 / 3.03 | 0.75 | 22325 | 12.6 | 8.2 | 37 | 180 | 1.0 |
| B: i1_raw90_lgbm | g2_vs_g3 | 90 | 3.79 / 4.59 | 0.57 | 15066 | 13.5 | 91.6 | 822 | 450 | 2.4 |
| baseline | g2_vs_g1 | 36 | 3.03 / 6.36 | 0.98 | 13225 | 15.6 | 8.9 | 37 | 180 | 0.5 |
| baseline | g2_vs_g4 | 36 | 2.85 / 6.90 | 0.75 | 18769 | 15.6 | 8.9 | 37 | 180 | 0.4 |

Not applicable: the deposited CSV holds RealEye's per-image aggregates, not the raw gaze stream, so per-sample FPS, samples per second and the real-time factor against the 60 Hz sensor cannot be computed, and feature extraction from raw gaze is not measured. Latency is dominated by pipeline overhead (a 36-weight dot product alone takes microseconds). All candidates are far below any real-time constraint. The z-score candidate costs nothing extra at inference (same 37 classifier parameters as the baseline; its within-person transform is part of the feature step) with a model size of 8.2 KB vs 8.9 KB. The LightGBM candidate is about 10 times larger (91.6 KB, 822 stored tree nodes) and about 1 ms slower per participant.

## 7. Recommendation

- **Do not claim an improvement over Colasanti et al.** under a leakage-free protocol. Report the honest baseline (G2 vs G3 AUC 0.679 ± 0.010; selected pipeline 0.673) next to their 0.72 and note that 0.72 lies within our seed spread.
- **A full run is worth it only for the within-person z-score for guilty vs innocent**, as a confirmatory test fixed in advance: pre-specify i2_z_lr, more fold repeats, many more permutations (at least 1000, with a faster implementation) and, if possible, an independent sample or the other datasets in the benchmark. Current evidence: AUC 0.688 vs 0.602, held-back seeds 0.673 vs 0.609, permutation p 0.031 (the floor), but paired ΔAUC CI [-0.025, +0.169].
- **Do not present deceptive vs honest (or G2 vs G3) as crime-specific detection** until the neutral-image-only signal is explained. The strongest next check is whether it is a group or session confound (order, age, sex, education, glasses), which needs metadata missing from the deposited CSV that the article authors could supply.
- **Use this dataset as a low-signal external benchmark**, not for a state-of-the-art claim: webcam data, 157 people, accuracy about 0.60 to 0.70.
- Out of scope and not tried: raw gaze sequence models (data unavailable) and an exact WEKA reproduction.

## Files

`results.json` (all numbers), `extras.json`, `screen_summary.json`, `selection.json`, `reproduction.json`, `fourclass_paired.json`, `tables.md` (full tables), `folds.json`, `oof/` (out-of-fold scores of every configuration), `screen_log.txt`, `extras_log.txt`, `RESUME.md`. Scripts: `scripts/phase2_mazza.py`, `phase2_mazza_configs.py`, `phase2_mazza_repro.py`, `phase2_mazza_select.py`, `phase2_mazza_tables.py`, `phase2_mazza_4class_paired.py`, `phase2_mazza_card.py`, `src/edd/mazza.py`. Note: the `multiclass` timing in `extras_log.txt` (15165 s) is wall-clock time that far exceeds the compute and probably includes time the session was paused. MU3D is not used in this track.
