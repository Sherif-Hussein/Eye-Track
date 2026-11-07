# Comparison card: RealEye webcam crime-scene dataset (track `mazza`)

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

