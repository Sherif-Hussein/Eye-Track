# Phase 6, Track C: eye-tracking validity (card)

Pre-registration `PREREG.md`; deviations `DEVIATIONS.md`; code `scripts/phase6_trackC_*.py`. Only the six-test sample-based family in C1 is a decision family; everything else is exploratory.

## C1. Sample-based dwell features under webcam noise

Reference: P4 SVM 1000 Hz, AUC 0.7811. Checks: level-0 I-VT reproduces 0.7854 (out-of-fold scores identical to the stored file, max difference 1.1e-12); the recomputed L = 1, draw 1 I-VT run matches the stored one (1.2e-12); the vectorised sample-based features match an independent loop implementation (max difference 2.3e-13); the noise code reproduces `apply_noise` exactly.

Full noise (offset + jitter + 10% loss), same draws as the webcam study. Delta = AUC minus 0.7811, paired participant bootstrap (2000). p = raw one-sided non-inferiority p at margin 0.02. Holm over the six primary tests in the next-to-last column.

| level (deg) | I-VT AUC | I-VT Delta [95% CI] | I-VT p | sample-based AUC | Delta [95% CI] | p raw | Holm p (family of 6) | sample minus I-VT [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7854 | +0.0044 [-0.0159, +0.0231] | 0.007 | 0.7723 | -0.0088 [-0.0277, +0.0097] | 0.124 | 0.567 | -0.0131 [-0.0222, -0.0023] |
| 1 | 0.7527 | -0.0284 [-0.0492, -0.0071] | 0.823 | 0.7736 | -0.0075 [-0.0267, +0.0108] | 0.113 | 0.567 | +0.0209 [+0.0072, +0.0350] |
| 1.5 | 0.7418 | -0.0393 [-0.0615, -0.0189] | 0.968 | 0.7750 | -0.0061 [-0.0248, +0.0124] | 0.080 | 0.480 | +0.0332 [+0.0157, +0.0527] |
| 2 | 0.7267 | -0.0544 [-0.0767, -0.0337] | 1.000 | 0.7709 | -0.0101 [-0.0285, +0.0082] | 0.175 | 0.567 | +0.0442 [+0.0258, +0.0644] |
| 3 | 0.7005 | -0.0806 [-0.1077, -0.0542] | 1.000 | 0.7655 | -0.0156 [-0.0344, +0.0037] | 0.377 | 0.755 | +0.0650 [+0.0429, +0.0865] |
| 4 | 0.6416 | -0.1395 [-0.1697, -0.1097] | 1.000 | 0.7442 | -0.0368 [-0.0587, -0.0166] | 0.947 | 0.947 | +0.1026 [+0.0800, +0.1257] |

Non-inferiority is not shown for the sample-based features at any level, including level 0 (Holm p 0.48 to 0.95). The point estimates are within 0.02 of the reference up to 3 deg (Delta -0.006 to -0.016), but the lower confidence limits (-0.025 to -0.034) cross the margin. The clean sample-based model is itself 0.013 below the I-VT model [-0.022, -0.002]. At 4 deg it is clearly below the reference (Delta -0.037, CI excludes zero). Against I-VT under the same noise the sample-based model is better by +0.021 (1 deg) to +0.103 (4 deg), all intervals excluding zero. Adding a run count of at least two in-area samples (`SB+seg`, closest event-free analogue of n_fix) gives the same picture (`c1_all_conditions.csv`).

Noise decomposition (AUC, mean of 3 draws; Delta, CI and p for every cell in `c1_all_conditions.csv`):

| level | offset only I-VT | offset only sample | jitter only I-VT | jitter only sample | loss only I-VT | loss only sample | all three I-VT | all three sample |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.7844 | 0.7728 | 0.7707 | 0.7702 | 0.7698 | 0.7730 | 0.7527 | 0.7736 |
| 1.5 | 0.7822 | 0.7705 | 0.7660 | 0.7695 | 0.7698 | 0.7730 | 0.7418 | 0.7750 |
| 2 | 0.7800 | 0.7684 | 0.7513 | 0.7709 | 0.7698 | 0.7730 | 0.7267 | 0.7709 |
| 3 | 0.7635 | 0.7556 | 0.7368 | 0.7735 | 0.7698 | 0.7730 | 0.7005 | 0.7655 |
| 4 | 0.7288 | 0.7212 | 0.7012 | 0.7754 | 0.7698 | 0.7730 | 0.6416 | 0.7442 |

Jitter is what breaks I-VT: jitter alone takes it to 0.701 at 4 deg (offset alone 0.729, loss alone 0.770), and the combined effect at 1 to 3 deg is larger than each part. The sample-based dwell is flat under jitter (0.770 to 0.775). An offset costs the sample-based features only from 3 deg (0.756; 0.721 at 4 deg), since the face areas are 11.5 x 8.6 deg. 10% sample loss costs both about 0.011 to 0.013 (0.7698 I-VT, 0.7730 sample-based).

Anchored to measured webcam precision (Kaduk et al. 2024: offset mean 1.45 deg, jitter SD 1.17 / 1.00 deg per axis from RMS sample-to-sample, 3% loss; exploratory): I-VT 0.7304 (Delta -0.0507 [-0.0732, -0.0289]), sample-based 0.7725 (Delta -0.0086 [-0.0267, +0.0090], raw p 0.116), sample-based plus segment count 0.7733. Same conclusion: at measured precision I-VT loses about 0.05, the sample-based dwell about 0.009, and non-inferiority is not shown.

## C2. Event-detection validity by sampling rate

| rate (Hz) | fixations / trial | on-face fixations / trial | mean duration (ms) | median (ms) | share of window in fixations |
|---|---|---|---|---|---|
| EyeLink parser | 16.37 | 16.03 | 256 | 219 | 0.812 |
| 1000 | 16.31 | 15.98 | 237 | 206 | 0.810 |
| 500 | 16.28 | 15.96 | 239 | 208 | 0.814 |
| 250 | 16.21 | 15.89 | 242 | 208 | 0.820 |
| 120 | 15.66 | 15.40 | 254 | 217 | 0.834 |
| 60 | 13.86 | 13.65 | 285 | 233 | 0.831 |
| 30 | 9.74 | 9.57 | 392 | 333 | 0.802 |

Agreement with the EyeLink parser (Pearson r). Counts = fixations with onset in 0-5 s, assigned to a face by the fixation centroid (I-VT) or mean gaze position (parser), per face. Dwell = share of 1-5 s spent in fixations on the face.

| rate (Hz) | count r, trial x face (n = 19,200) [95% CI] | count r, participant x image [95% CI] | count r, image level | I-VT / parser count | dwell r, participant x image | sample-based dwell r |
|---|---|---|---|---|---|---|
| 1000 | 0.906 [0.860, 0.942] | 0.932 [0.892, 0.959] | 0.995 | 1.00 | 0.971 | 0.983 |
| 500 | 0.917 [0.875, 0.948] | 0.939 [0.902, 0.964] | 0.996 | 1.00 | 0.979 | 0.983 |
| 250 | 0.938 [0.914, 0.956] | 0.956 [0.934, 0.971] | 0.997 | 0.99 | 0.990 | 0.983 |
| 120 | 0.959 [0.950, 0.966] | 0.973 [0.965, 0.978] | 0.998 | 0.96 | 0.997 | 0.983 |
| 60 | 0.862 [0.844, 0.879] | 0.884 [0.861, 0.905] | 0.988 | 0.85 | 0.994 | 0.983 |
| 30 | 0.676 [0.644, 0.704] | 0.682 [0.627, 0.734] | 0.948 | 0.60 | 0.989 | 0.982 |

**Verdict on 0.77 versus 0.68.** The paper should report 0.68 at 30 Hz (0.676 trial x face, 0.682 participant x image) and state the unit. The participant x image definition returns 0.97 at 120 Hz and 0.88 at 60 Hz, which are the other two figures of the earlier card, so those stand and 0.77 does not. None of the 16 variants tried (unit, mean or sum, Pearson or Spearman, 64- or 32-trial set, the published label rule, within-participant, trial level; `c2_provenance.csv`) gives 0.77 at 30 Hz. The nearest are group 1 alone (0.79), the published-table label rule (0.85) and the image-level mean (0.95). I could not trace where 0.77 came from, so it should be dropped. Note that at 1000 Hz the agreement is 0.91 to 0.93, below its 120 Hz value (0.96 to 0.97), because the unfiltered 1000 Hz I-VT splits fixations that the parser keeps whole.

At 30 Hz fixations per trial fall from 16.3 to 9.7 and mean duration rises from 237 to 392 ms; the I-VT finds 60% of the parser's per-face fixations. Dwell agreement stays at r = 0.99 (participant x image) at every rate, and sample-based dwell agrees with parser dwell at 0.98 at every rate. The 30 Hz result therefore holds for dwell features, not for fixation counts.

Smallest detectable saccade. Velocity span 2h/fs with h = max(1, round(0.004 fs)) samples; the 30 deg/s threshold needs at least this displacement between the two samples (noise-free bound). Empirical columns: share of transitions between consecutive parser fixations (n = 72,671) that I-VT keeps as two separate fixations, by amplitude.

| rate (Hz) | half-width (samples) | span (ms) | min displacement (deg) | min fixation enforced (ms) | resolved 0-0.5 deg | 0.5-1 | 1-2 | 2-4 | 4-8 | >= 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 4 | 8.0 | 0.24 | 60.0 | 85% | 94% | 94% | 91% | 74% | 92% |
| 500 | 2 | 8.0 | 0.24 | 60.0 | 85% | 94% | 94% | 91% | 74% | 92% |
| 250 | 1 | 8.0 | 0.24 | 60.0 | 85% | 95% | 95% | 92% | 75% | 93% |
| 120 | 1 | 16.7 | 0.50 | 66.7 | 63% | 93% | 95% | 93% | 74% | 94% |
| 60 | 1 | 33.3 | 1.00 | 66.7 | 3% | 31% | 92% | 93% | 75% | 94% |
| 30 | 1 | 66.7 | 2.00 | 66.7 | 1% | 1% | 11% | 86% | 66% | 93% |

The window is 8 ms only at 250 Hz and above; below that it is one sample either side (16.7, 33.3, 66.7 ms). At 30 Hz saccades under 2 deg cannot reach the threshold (1-2 deg: 11% resolved; 0.5-1 deg: 0.5%); at 60 Hz saccades under 1 deg cannot (0.5-1 deg: 31%). The 4-8 deg bin loses 25 to 34% at every rate (a missing sample at a fixation midpoint), so that loss is not a rate effect.

## C3. Data-quality table

| dataset | device | rate | eyes / head | % missing or invalid samples | % trials / participants excluded | calibration or validation error | class difference in quality |
|---|---|---|---|---|---|---|---|
| Delmas | EyeLink 1000 | 1000 Hz | right eye; chin and forehead rest, 57 cm | 1.03% of samples in the 5 s displays (participants 0 to 25.7%); by group G1 0.6%, G2 2.4%, G3 0.6%, G4 0.9%, G5 0.6% | image rows 5.7% removed by the authors (4,525 of 4,800); trials 0%; participants 0 of 75 | HV9 validation, mean error 0.46 deg (median 0.43; participant means 0.17 to 0.84); 26 of 192 validations labelled POOR; 1 failed validation excluded | G2 (concealment) loses most data (2.4% against 0.6 to 0.9%) |
| Celniak | Tobii Pro Fusion | 250 Hz (38 recordings), 120 Hz (1) | both eyes; remote, head free | 26.0% in the question slides with at least one eye invalid (1 to 73% per recording); both eyes invalid 21.5% | slides 0% by us; 19 of 39 recordings have a slide below 50% valid (147 of 819 slides); participants: not available | validation accuracy mean 0.69 deg (median 0.55, max 2.84); calibration accuracy median 0.12 (max 5.0); precision RMS median 0.26; 1 recording has no validation value | validation accuracy differs by class: AUC 0.82 [0.65, 0.95], medians 0.63 (guilty) vs 0.41; missing data does not (AUC 0.50) |
| Chen exp 1 | EyeLink 1000 | 1000 Hz | monocular; chin rest | 4.9% of pupil samples in the RSVP phase (participants 1.0 to 17.7%) | participants 4 of 33 (12%, authors' scripts); trials 3.8% (baseline rules) | first validation per participant (22 of 29 participants have one): mean 0.66 deg, median 0.56, range 0.26 to 1.64 | familiar minus control missing share +0.02 points [-0.31, +0.35] (no difference) |
| Chen exp 2 | EyeLink 1000 | 1000 Hz | monocular; chin rest | 2.9% of pupil samples in the RSVP phase (participants 0.1 to 6.6%) | participants 4 of 31 (13%, authors' scripts); trials 4.1% (baseline rules) | first validation per participant (22 of 27 participants have one): mean 0.72 deg, median 0.52, range 0.26 to 2.48 | familiar minus control missing share -0.03 points [-0.19, +0.12] (no difference) |
| Preti | not reported | not available | not available | not available (fixation-level data); 7.0% of fixation rows flagged as blink | 4.3% error trials; 0.27% without a pupil value; 3 of 60 participants removed by the authors | not available | blink share differs: liar median 0.059 vs 0.044, AUC 0.66 [0.52, 0.79] |
| Colasanti (RealEye) | webcam, 1080p at 60 fps, RealEye software | 60 Hz nominal (gaze at least every 16 ms); effective rate not available | not available; no head support (online) | not available (aggregate tables) | 3 of 160 participants (1.8%) excluded by the authors; trials not available | RealEye platform accuracy 113 px on average (90 to 156), quoted in the paper; per participant: not available | not available |

Liar's Dice is omitted (DEVIATIONS 5). Celniak's missing-data share is large (1 to 73% between recordings) but does not separate the classes; the validation error does. In Delmas the data loss separates group G2 from the others. Chen validation values come from the EDF messages and exist for 22 of the used participants in each experiment.

## Claims in the paper that these change

- "I-VT agrees with the EyeLink parser at r = 0.97, 0.88 and 0.77": 0.77 becomes 0.68, with the unit stated. Add dwell agreement r = 0.99 at every rate.
- "Fixations" at 30 Hz are dwell segments (392 ms mean, 60% of the parser count). State the velocity window per rate and the saccade floor (2 deg at 30 Hz, 1 deg at 60 Hz).
- Webcam claim: the loss comes from jitter interacting with I-VT. Sample-based dwell loses 0.009 at measured webcam precision against 0.051 for I-VT, but non-inferiority at margin 0.02 is not shown at any level, level 0 included. The axis of Fig. 5c needs relabelling and the decomposition added. Pupil features still need a pupil channel that webcams do not give.
- Table 1 and the reporting checklist: device, rate, eyes, head support, data loss and validation error are in the C3 table. Celniak's class difference in validation error (AUC 0.82) is a data-quality result.
