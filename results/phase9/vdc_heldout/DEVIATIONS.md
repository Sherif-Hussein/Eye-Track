# Deviation note: held-out recognition test on the Van der Cruyssen et al. (2024) data

**Registration:** OSF Secondary Data Preregistration "Held-out test of a low-cost eye-movement CIT pipeline", associated project https://osf.io/2p4gm, registered 5 November 2026, 14:24; updated on 6 November 2026 before data access (Prior Knowledge corrected; H5 added; 15 Hz and 10 Hz descriptive analyses added).
**Data:** https://osf.io/p562n (Van der Cruyssen, Ben-Shakhar, Pertzov & Verschuere, 2024, https://doi.org/10.1037/mac0000140). Download started 6 November 2026, 02:22:16 UTC (`step1/ACCESS.md`, with SHA-256 hashes of every file).
**This note** was written on 6 November 2026 after the data files had been inspected (file contents, column meanings and participant counts per group) and **before any analysis that uses the group labels**. No AUC, model or comparison has been computed.

**Terms.** In the original study, "guilty" participants handled the crime-related items, "informed innocent" participants saw pictures of them, and "naive innocent" participants never saw them. In recognition terms: the first two groups **know** the items and the third does not. The registered hypotheses are unchanged; they are read as tests of whether a low-cost gaze measure detects recognition.

## What the deposit contains
Per-trial summaries, not raw gaze samples, and no pupil data: dwell time per display area in 1–1000 ms and 1000–5000 ms; fixation counts per area; the sequence of fixated areas (no onsets or durations); fixation durations on the single item; responses and reaction times; and, for the 32 trials whose four-item display contains a crime-related item, the share of samples on each area in 100 ms bins (0–5000 ms). Participants: 155 tested, 133 retained by the original authors (42 / 45 / 46), identical to the article.

## Deviations
- **D1 Data type.** The registration foresaw raw samples, or fixation reports as a fallback. Only the summaries above exist, so all features are computed from them.
- **D2 Features (11 of the 13).**
  - Dwell in 1–1000 ms (the authors' window; registered 200–1000 ms) and in 1000–5000 ms.
  - Dwell in the five finer time bins and the time to the first look, approximated from the 100 ms time course.
  - Number of fixations; number of visits from the fixation sequence (a visit is a run of consecutive fixations on the same area; a fixation outside all areas ends a visit); whether the first fixation landed on the item.
  - **Dropped:** the two pupil features, which are not deposited.
  - **No downsampling:** there are no samples to downsample.
- **D3 Displays used.** The time-course features exist only for the 32 displays that contain a crime-related item. The item model is therefore trained and scored on those displays: the crime-related item (label 1) against the three other items (label 0). The participant score is computed on the same displays, as registered.
- **D4 H4 cannot be run as registered.** The frozen 13-feature Delmas et al. model needs pupil features and sample-based definitions. H4 is replaced by an **exploratory** analysis: the frozen pupil-free 30 Hz Delmas et al. model, applied to the closest available features. It is **not** in the Holm family. The secondary family becomes **H2, H3, H5** (Holm, α = .05; H2 and H5 one-sided, H3 two-sided).
- **D5 Frame-rate analyses impossible.** The registered descriptive analyses at 15 Hz, 10 Hz and 1000 Hz cannot be produced without samples. The rate question is addressed on other datasets.
- **D6 Data cleaning** (fixed now, before analysis):
  - (a) Practice rows, which are labelled as trial 64 in the gaze files, are removed by keeping only the trial indices listed in each participant's response file.
  - (b) Session-name suffixes (`_G`, `_I`, `_N`, `_1`, `_2`, `_3`) reveal the group, so they are stripped before features and folds are built. Participant number is the only identifier.
  - (c) The original authors' 22 exclusions are applied, leaving 133.
  - (d) Participant 23 (44 of 64 gaze trials) is kept if they have at least 8 valid displays containing a crime-related item (the registered rule); otherwise excluded, and reported.
  - (e) Participants 17, 31 and 43 have no reaction times. They are included in H1, H2 and H5, which use no reaction times. For H3, the authors' index uses the measures available for them, as in the authors' scripts.
- **D7 H3 comparator.** The authors' equal-weight index (six measures, z-scored within participant, signed as in the article, averaged) is computed following the article's description. **Check before H3:** it must reproduce the published AUC of .71 (guilty vs naive) to within rounding. Any difference between the article and the deposited scripts is reported.
- **D8 Disclosure.** While inventorying the files, the text of the authors' deposited PDF "ExploratoryAnalysis_IndividualMeasurements" (per-measure AUCs, partly published in the article) was printed to the console. It was not used in any decision recorded here.

Everything else is as registered:
- 10 folds × 3 repeats, participant-disjoint and stratified by group;
- tuning and thresholds inside the training participants only;
- threshold set to flag 7% of the training participants who do not know the items;
- 1,000 refit label permutations for H1, H2 and H5; 2,000-swap paired test for H3;
- participant bootstrap intervals; Clopper–Pearson intervals for rates.
