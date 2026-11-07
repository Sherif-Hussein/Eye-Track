# Phase 4 card: the unified pipeline, one fixed recipe on every dataset

Pre-registered in `PREREG.md` (design approved by the senior analyst, amendments recorded). All numbers are out-of-fold on the tracks' frozen folds, from `results.json` and the `results_*.json` files; figure data are in `figdata/`. Deviations and post-hoc additions: `DEVIATIONS.md`. Everything is reported, whatever the outcome.

**Recipe.** 30 Hz where raw samples exist, label-free vectorised features, within-participant standardisation (as pre-registered per dataset), L2 logistic regression with balanced weights, C from one 9-point grid by participant-grouped inner folds, person decision at thresholds calibrated on training participants only.

## 1. Headline

| Claim | Result |
|---|---|
| Efficiency at matched accuracy (Delmas) | Same AUC as P4 SVM 1000 Hz (0.7813 vs 0.7811, delta +0.0002) but non-inferiority is **not established after Holm** (raw p = .038, Holm p = .19; 95% CI [-0.022, +0.020], lower bound just outside the -0.02 margin). 58x faster per participant (3.2 vs 183 ms), 154x smaller (719 B vs 111 kB) |
| Chen (30 Hz, unified recipe) | Detection at calibrated FPR <= 7% 57.7% vs rate rule 42.9% (delta +0.149, CI [+0.012, +0.280], raw p = .019, **Holm p = .114**, not significant in the 7-test family). I2b (confirmed, separately) 59.5% |
| Only Holm-significant test in the family | Preti Level A (ocular only): AUC 0.569 vs 0.380 for the paper's ocular baseline (Holm p = .045); the baseline is below chance, and the unified value is itself near chance |
| Celniak | The unified recipe fails: AUC 0.461 vs 0.608 honest baseline. The tracker's validation error alone gives 0.803 (see section 4) |
| Sampling rate on Chen (analysis A) | 60 and 30 Hz are non-inferior to 100 Hz for both models (all four Holm p = .002) and beat the rate rule at 100 Hz on trial AUC (all four Holm p = .002) |
| Adaptive threshold under webcam noise (analysis B) | Not non-inferior at any level, including clean (Holm p >= .24); degradation with noise is the same shape as with the oracle threshold |
| Cascade (Chen) | Detection 67.3% vs I2b 59.5% and non-inferior, with 38% of cases deferred, but **no cost saving**: mean cost 15.4 ms per decision vs 12.3 ms for I2b alone |

## 2. The seven pre-registered tests (Holm family)

Paired participant-cluster bootstrap, 2000 resamples, one-sided. Unified minus baseline.

| # | Test | Baseline | Unified | Delta [95% CI] | Raw p | Holm p |
|---|---|---|---|---|---|---|
| 1 | Delmas, image AUC, mean over 6 sets, non-inferiority margin 0.02 vs P4 SVM 1000 Hz | 0.7811 | 0.7813 | +0.0002 [-0.0216, +0.0203] | .038 | .192 |
| 2 | Chen, person detection at FPR <= 7% vs rate rule | 0.429 | 0.577 | +0.149 [+0.012, +0.280] | .019 | .114 |
| 3 | Celniak, participant AUC vs honest SVM | 0.608 | 0.461 | -0.147 [-0.307, +0.008] | .971 | 1.0 |
| 4 | Preti Level A, AUC vs `B_ocular` | 0.380 | 0.569 | +0.189 [+0.042, +0.342] | .0065 | **.045** |
| 5 | Preti Level B, AUC vs `B_fusion` | 0.928 | 0.964 | +0.036 [-0.005, +0.088] | .062 | .250 |
| 6 | Mazza G2 vs G3, AUC vs published LR | 0.690 | 0.601 | -0.090 [-0.233, +0.056] | .890 | 1.0 |
| 7 | Mazza guilty vs innocent, AUC vs published LR | 0.605 | 0.664 | +0.059 [-0.040, +0.161] | .117 | .351 |

Delmas sensitivity, group 3 counted once (set `32_g3` dropped, 5 sets): 0.7715 vs 0.7723, delta +0.0008 [-0.0247, +0.0245], raw p = .053.

**Scale caveat (post hoc, `DEVIATIONS.md` item 3).** The person-level tests average raw logits over repeats, and tuned C differs between folds, so the repeat-averaged AUC understates the unified models. Per-repeat mean AUC, unified vs baseline (3 shared repeats, descriptive): Celniak 0.466 vs 0.597; Preti A 0.552 vs 0.399 (delta +0.153 [+0.047, +0.272]); Preti B 0.951 vs 0.927 (+0.025 [-0.015, +0.063]); Mazza G2 vs G3 0.684 vs 0.679 (+0.006 [-0.106, +0.144]); Mazza guilty vs innocent 0.667 vs 0.602 (+0.065 [-0.043, +0.161]). The Mazza G2 vs G3 result changes from clearly negative to a tie; the other conclusions do not change.

## 3. Per-dataset metrics of the unified model (mean over repeats)

Row level = trial / image / slide; person level at the EER-calibrated threshold (HTER, accuracy) and at the FPR <= 7% threshold (detection, realised FPR). Full metric set (sensitivity, specificity, precision, F1, MCC, EER) is in `results.json` under `tasks`.

| Task | Features | Participants | Row AUC | Person AUC | Person EER | HTER at EER threshold | Accuracy | Detection at FPR<=7% (realised FPR) |
|---|---|---|---|---|---|---|---|---|
| Delmas 64 g1 control | 13 | 15 | 0.921 | 1.000 | 0.000 | 0.056 | 0.944 | 0.978 (0.089) |
| Delmas 64 g2 concealment | 13 | 15 | 0.884 | 1.000 | 0.000 | 0.044 | 0.956 | 0.956 (0.000) |
| Delmas 64 g3 simple CM | 13 | 15 | 0.838 | 0.933 | 0.033 | 0.067 | 0.933 | 0.867 (0.000) |
| Delmas 32 g3 | 13 | 15 | 0.817 | 0.926 | 0.111 | 0.111 | 0.889 | 0.844 (0.067) |
| Delmas 32 g4 enhanced CM | 13 | 15 | 0.561 | 0.668 | 0.356 | 0.367 | 0.633 | 0.267 (0.044) |
| Delmas 32 g5 CM + feedback | 13 | 15 | 0.619 | 0.653 | 0.389 | 0.422 | 0.578 | 0.467 (0.178) |
| Chen pooled | 75 | 56 | 0.601 (trial) | 0.854 | 0.232 | 0.247 | 0.758 | 0.577 (0.074) |
| Chen cascade stage 1 | 10 | 56 | 0.586 | 0.849 | 0.228 | 0.223 | 0.780 | 0.506 (0.092) |
| Celniak | 19 | 39 | 0.466 | 0.466 | 0.496 | 0.469 | 0.530 | 0.083 (0.088) |
| Preti A ocular | 16 | 57 | 0.552 | 0.552 | 0.462 | 0.475 | 0.526 | 0.173 (0.100) |
| Preti B ocular + RT | 36 | 57 | 0.951 | 0.951 | 0.083 | 0.094 | 0.906 | 0.901 (0.089) |
| Mazza G2 vs G3 | 36 | 78 | 0.671 | 0.671 | 0.349 | 0.353 | 0.646 | 0.274 (0.095) |
| Mazza guilty vs innocent | 36 | 157 | 0.662 | 0.662 | 0.377 | 0.367 | 0.633 | 0.285 (0.091) |

Delmas "person" = the CIT decision with simulated innocents (the negatives are the same person's unknown faces re-used as pseudo-probes). The two countermeasure groups with enhanced strategies stay near 0.57-0.62, as in Phase 2. Calibrated thresholds held the false-positive target roughly (realised FPR 0.07-0.10 on most tasks; 0.18 on Delmas 32 g5). ROC/DET with EER points are in `figdata/roc_det_eer.csv`. Tuned C is summarised per task in `results.json` (`C_chosen_counts`) and the validation curves in `figdata/validation_curve.csv`; learning curves (AUC at 25/50/75/100% of training participants) rise gradually for Delmas groups 1-3, Chen (0.575 to 0.601), Preti B (0.913 to 0.951) and Mazza (0.64 to 0.67) and stay flat near chance for Celniak and Preti A.

## 4. Celniak control and Preti sensitivities (descriptive)

- Celniak: the 19 probe-vs-irrelevant features give 0.461 (CI of the person AUC overlaps chance). The tracker's validation error alone gives 0.803; the 19 features plus validation error as a covariate give 0.795, no better than the validation error alone (control minus unified +0.334 [+0.095, +0.558]; unified minus validation error alone -0.342 [-0.584, -0.080]). The only usable signal in this dataset is recording quality, as found in Phase 2.
- Preti contrast-only (strict within-participant reference): Level A 0.622 (delta vs `B_ocular` +0.242 [+0.100, +0.392]), Level B 0.960 (+0.032 [-0.029, +0.115]). Dropping raw levels does not hurt Level B. Level A stays near chance on its own scale (0.57-0.62), so the "win" in test 4 reflects an ocular baseline below chance rather than an ocular signal.

## 5. Chen cascade (separate family, one test)

Stage 1 = unified recipe on 10 coarse features (pupil, dx at 500 ms). Deferral fixed on training participants: detect above the training-null 98th percentile, reject below the 10th percentile of training familiar statistics, defer the rest to the stored I2b decision.

| | Detection (FPR <= 7% target) | Realised FPR | Deferred |
|---|---|---|---|
| I2b alone | 0.595 | 0.065 | n/a |
| Cascade | 0.673 | 0.068 | 38% of cases (46% of familiar cases, 34% of control cases) |

Delta +0.077 [+0.012, +0.155]; non-inferiority (margin -0.05) p = .0005. **Cost:** stage 1 10.7 ms p50, I2b 12.3 ms, cascade mean 10.7 + 0.38 x 12.3 = 15.4 ms per decision, i.e. 25% more than I2b alone. The cascade's accuracy gain is real in this sample (no cost-benefit claim): stage 1 is not meaningfully cheaper than I2b because decimation and binning dominate the cost, not the classifier.

## 6. Analysis A: sampling rate on Chen (reference 100 Hz = the published analysis rate; claim: 30-60 Hz vs the published rate)

Trial AUC, Holm within each family of four.

| Model | 100 Hz | 60 Hz | 30 Hz | Low rate minus 100 Hz (CI) | A1 Holm p |
|---|---|---|---|---|---|
| Unified | 0.6016 | 0.6016 | 0.6016 | 60 Hz -0.0000 [-0.004, +0.004]; 30 Hz -0.0001 [-0.004, +0.004] | .002 / .002 |
| I2b | 0.6064 | 0.6044 | 0.6044 | 60 Hz -0.0020 [-0.006, +0.002]; 30 Hz -0.0020 [-0.006, +0.002] | .002 / .002 |
| Rate rule | 0.5683 | | | | |

A2: all four low-rate models beat the rate rule at 100 Hz on trial AUC, delta +0.033 to +0.036 (CIs about [+0.014, +0.056]), Holm p = .002. Secondary: person AUC 0.855 / 0.855 (unified 60 / 30 Hz), 0.880 / 0.877 (I2b); detection at FPR <= 7% 0.577 / 0.577 (unified), 0.619 / 0.613 (I2b), vs rate rule 0.429. The reference is 100 Hz because gaze exists only as 100 Hz epochs; I2b features at 100 Hz match the confirmatory matrix exactly (max diff 0). Pupil at 60/30 Hz comes from a trace already smoothed and down-sampled by the authors' pipeline, so this does not test 1000 Hz sensors.

## 7. Analysis B: data-driven adaptive fixation threshold (rule `ek`, Delmas, unified model)

Threshold = max(30, 6 x sqrt((SDx^2+SDy^2)/2)) deg/s, SDs median-based, per participant, no labels. Clean: 30 deg/s for everyone; under noise it averages 77 / 88 / 100 / 122 / 145 deg/s for L = 1 / 1.5 / 2 / 3 / 4 deg. AUC vs the clean P4 SVM 1000 Hz (0.7811), margin 0.02, Holm over six levels.

| Offset L (deg) | AUC | Delta | 95% CI | Raw p | Holm p | Non-inferior |
|---|---|---|---|---|---|---|
| 0 (clean) | 0.7813 | +0.0002 | [-0.022, +0.019] | .040 | .240 | no |
| 1 | 0.7650 | -0.0161 | [-0.035, +0.001] | .364 | 1.0 | no |
| 1.5 | 0.7651 | -0.0160 | [-0.036, +0.001] | .370 | 1.0 | no |
| 2 | 0.7604 | -0.0207 | [-0.041, -0.003] | .577 | 1.0 | no |
| 3 | 0.7536 | -0.0275 | [-0.048, -0.010] | .805 | 1.0 | no |
| 4 | 0.7373 | -0.0438 | [-0.065, -0.025] | .991 | 1.0 | no |

The adaptive rule recovers the clean level exactly (it equals the fixed threshold there) and recovers part of the noise loss (Phase 3 fixed 30 deg/s: -0.028 / -0.039 / -0.054 / -0.081 / -0.140 at L = 1 / 1.5 / 2 / 3 / 4), but non-inferiority is not shown at any level. At the clean level the test itself is borderline for the same reason as Delmas test 1 (CI lower bound -0.022). The `nh` rule was dropped (PREREG amendment: thresholds 140-220 deg/s at 30 Hz).

## 8. Efficiency (same session, single thread, interleaved rounds; p50 / p95 ms per decision unit)

| Dataset | Baseline | Unified | Latency | Throughput | Real-time factor | Peak memory | Model size, parameters |
|---|---|---|---|---|---|---|---|
| Delmas (one participant, 64 images, decimation excluded) | P4 SVM 1000 Hz, original code: 183 / 191 ms | LR 30 Hz fast: 3.2 / 3.3 ms (chunk 8: 9.6 / 11.0) | 58x faster (19x at chunk 8) | 5.5 to 317 participants/s | 1,921x to 111,735x | 612 kB to 961 kB (157 kB at chunk 8) | 110,953 B / 10,493 to 719 B / 53 |
| Chen (from 100 Hz epochs; 144 trials) | rate rule: 6.6 / 7.4 ms | unified 30 Hz LR 14.2 / 15.5; stage 1 10.7 / 12.4; I2b 12.3 / 13.5 | 2.2x slower than the rule | 152 to 70/s | 52,124x to 72,212x | 16 kB to 415 kB | 24 B / 2 to 2,703 B / 301 |
| Celniak (Recording10, 250 Hz, 101k samples) | paper SVM, original extractor: 912 / 948 ms | LR with fast extraction: 9.7 / 10.5 ms | 94x faster | 1.1 to 103/s | 445x to 41,866x | 85.5 MB to 4.4 MB | 126,285 B / 14,458 to 911 B / 77 |
| Preti (one participant) | `B_fusion` LR, original code: 36.5 / 38.6 ms | Level B fast: 0.36 / 0.41 ms | 101x faster | 27 to 2,795/s | n/a | 142 kB to 7 kB | 1,650 B / 37 to 1,455 B / 145 |
| Mazza (one participant) | published LR: 0.46 / 0.50 ms | z-features LR: 0.29 / 0.30 ms | 1.6x faster | 2,158 to 3,469/s | n/a | 11.9 kB to 5.2 kB | 953 B / 37 to 1,455 B / 145 |

Chen is timed from preprocessed epochs (raw EDF reading and blink reconstruction are shared by all methods and excluded). The Chen unified model is slower than the paper's single-sample rule, which is a one-number statistic. Stored parameters for LR = coefficients + intercept + 3 per feature. Full table: `figdata/efficiency.csv`.

## 9. Caveats

- The recipe is informed by Phase 2/3 results on the same people (PREREG section 0); the run is a fixed-recipe, fixed-fold, fully reported comparison, not an independent confirmation.
- Only one of seven Holm tests passes (Preti Level A, p = .045), against a below-chance baseline. The Delmas non-inferiority claim and the Chen detection gain both have raw p < .05 and Holm p > .05 in the seven-test family; the Chen gain is also confirmed in the separate Phase 3 test and in analysis A (trial AUC, Holm .002 within its own family).
- Repeat-averaging of logits across folds with different tuned C lowers the person-level unified AUCs (see section 2 caveat); Mazza G2 vs G3 changes from a loss to a tie under per-repeat AUC.
- Celniak is a recording-quality signal, not a concealed-information signal; the unified CIT features carry none (AUC 0.46).
- Chen at 30 Hz is simulated from 100 Hz epochs; Delmas from 1000 Hz. Neither adds webcam spatial error; the webcam-noise result (analysis B) shows the claim does not extend to webcam-grade spatial error.
- The cascade saves no compute with this implementation.
- Chen control-vs-control stands in for innocent people; Delmas innocents are simulated; the Chen participants overlap with the screening sample.
- Liar's Dice excluded.
