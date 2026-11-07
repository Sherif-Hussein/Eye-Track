# Comparison card for track `chen` (moving-RSVP familiar-face recognition)

> **Final (2026-11-02).** All stages are complete: Exp 1 and Exp 2 reproduction, screening of 9 ideas and 4 ablations on frozen participant-disjoint folds, best candidate with CIs and permutation test, cross-experiment transfer, efficiency. Numbers are in `results.json`; scripts are `scripts/phase2_chen*.py` and `src/edd/chen.py`; resume notes and pitfalls are in `RESUME.md`.

## 1. The published study

**Citation.** Chen, I., Mathôt, S. & Akyürek, E. G. (2026). Subtle gaze and pupil dynamics: Detecting recognition of familiar faces with moving serial visual presentation. *Journal of Cognition, 9*(1):20. DOI 10.5334/joc.492. Full text: PMC12962246.

**Data and code.** OSF: Exp 1 at osf.io/ske56, Exp 2 at osf.io/sk5zq. Shared: EDF files, OpenSesame CSV logs and the authors' Python analysis scripts (`eyelinkparser`, `datamatrix`, `time_series_test`). Familiar-face images are withheld.

| Item | Exp 1 | Exp 2 |
|---|---|---|
| Participants | 33 recruited; 4 excluded (subjects 14, 15, 24, 30) → **N = 29** | 31 recruited; 4 excluded (8, 15, 22, 28) → **N = 27** |
| Gaze instruction | Central fixation; follow only the target face with the eyes | Free viewing; keypress after the stream |
| Task | Concealed recognition inside a moving RSVP stream. Each trial has 13 grayscale faces, a new one every 300 ms (SOA), each on screen for 600 ms while moving 60 px left or right | Same |
| Critical face | Position 5–8, one of: target (opposite sex), **familiar** (the participant's own parent), **control 1**, **control 2**. Each control is one fixed unfamiliar face image per participant, sex-matched to the parent and repeated 48 times | Same |
| Trials | 192 (48 per type, randomised over 12 blocks) | 192 |
| Sensor | EyeLink 1000, monocular, 1000 Hz, downsampled to 100 Hz; 100 Hz monitor | Same |
| Preprocessing | `defaulttraceprocessor(blinkreconstruct=True, downsample=10, mode='advanced')`. Baseline is the mean of the 50 ms before critical-face onset. Trials are dropped if the baseline is undefined or its per-participant \|z\| > 2 (211 of 5568). Baseline correction is subtractive, over 0–2500 ms. Rate = `np.diff` of pupil, then a 25-sample Hanning smooth. Gaze dx = (x − 960) × motion direction | 215 of 5184 trials dropped |
| Group analysis | Sample-by-sample LME (`pupil ~ condition`, random slopes by subject), Benjamini–Hochberg FDR, clusters of 200 ms or more | Same |
| **Individual detection** | Leave-one-participant-out: the peak sample is argmax \|mean familiar − mean control\| over the other participants' pooled trials. Then a one-tailed Student t-test of familiar vs both controls at that sample, α = .05. No multiplicity correction; 4 measures are tested per person | Same |

**Reported metrics** (Results, Figs. 4/6 and 9/11; abstract):

| Measure | Exp 1 | Exp 2 |
|---|---|---|
| Pupil size, detected | 10/29 (34.5%) | 5/27 (18.5%) |
| **Rate of pupil change, detected** | **16/29 (55.2%)** | **9/27 (33.3%)** |
| Gaze 0–1000 ms | 4/29 | 0/27 |
| Gaze 1000–2500 ms | 3/29 | 4/27 |
| AUC, EER, HTER, specificity, false-positive rate, innocent group | not reported | not reported |

The authors list an innocent group and ROC/AUC analyses as the next step.

## 2. Reproduction under their protocol (Exp 1 and Exp 2 done)

**Data status.** All 64 EDFs (2.35 GB) are downloaded and verified against the OSF sizes. All 64 subjects are extracted with a port of the authors' exact `datamatrix` advanced blink reconstruction (`edd.chen.dm_blinkreconstruct`). Every participant has 192 experimental trials.

**Exp 1, the authors' pipeline and leave-one-out rule re-implemented from their shared scripts:**

| | Published | Ours, exact blink method | Ours, approximate gap interpolation |
|---|---|---|---|
| Trials excluded by the baseline rule | 211 / 5568 | 214 / 5568 | 212 / 5568 |
| Pupil size detected | 10/29 (34.5%) | **10/29 (34.5%)** | 9/29 |
| Rate of pupil change detected | 16/29 (55.2%) | **15/29 (51.7%)** | 13/29 |
| Gaze 0–1000 / 1000–2500 ms detected | 4 / 3 | **4 / 3** | 4 / 3 |
| Rate peak (leave-one-out median) | cluster 660–860 ms | 790 ms | 810 ms |

The reproduction matches the paper on pupil size and both gaze windows. The rate result is off by one participant. The remaining gap is plausibly due to the edfapi's sample-gap handling ("Inserting dummy samples") or the floating-point details of the `ttest_ind` threshold.

**What the paper did not measure: false positives (Exp 1, exact method).** Each participant's control 1 and control 2 are single fixed unfamiliar face images, so running the same rule on control 1 vs control 2 measures its false-positive rate on recognition-free data. "Detection at calibrated threshold" uses as threshold the 95th percentile of the other participants' control-vs-control statistics.

| Rule | Detection | FPR, control 1 vs 2 (per one-tailed test) | People flagged on control vs control | FPR, random split of control trials | Detection at calibrated threshold (FPR) | Person-level AUC / EER / HTER |
|---|---|---|---|---|---|---|
| Pupil size | 34.5% | 5.2% | 3/29 (p = .57) | 5.4% | 34.5% (6.9%) | 0.728 / 0.345 / 0.362 |
| **Rate (paper's headline)** | **51.7%** | **15.5%** | **9/29 (binomial p = .0016 vs 10%)** | 5.0% | **34.5% (6.9%)** | 0.750 / 0.293 / 0.362 |

**Interpretation.** The t-test itself is calibrated, since a random split of the same trials rejects at 5%. The rate rule's excess false positives therefore come from image identity: one specific face elicits its own pupil-rate transient, and a single-sample test cannot tell this apart from recognition. With the threshold set so that unfamiliar-vs-unfamiliar comparisons are flagged at about 5–7%, the rate rule detects 34.5% of familiar participants in Exp 1, not 51.7–55.2%. The Exp 2 check below did not reproduce the excess, and the Exp 1 calibrated figure has a wide CI (13.8-69.0%, section 5).


**Exp 2 (N = 27), same code, exact blink method (run 2026-11-02):**

| | Published | Ours, exact blink method | Ours, approximate gap interpolation |
|---|---|---|---|
| Trials excluded by the baseline rule | 215 / 5184 | 215 / 5184 (5 undefined + 210 outliers) | not recorded |
| Pupil size detected | 5/27 (18.5%) | **5/27 (18.5%)** | 4/27 |
| Rate of pupil change detected | 9/27 (33.3%) | **8/27 (29.6%)** | 5/27 |
| Gaze 0-1000 / 1000-2500 ms detected | 0 / 4 | **0 / 4** | not run |
| Rate peak (leave-one-out median) | cluster 1510-1730 ms | 1620 ms | |

Exp 2 reproduces pupil size and both gaze windows exactly and the rate rule to within one participant (8 vs 9), the same direction of deviation as in Exp 1 (15 vs 16).

**False positives in Exp 2 (exact method).**

| Rule | Detection | FPR, control 1 vs 2 (per one-tailed test) | People flagged on control vs control | FPR, random split of control trials | Detection at calibrated threshold (FPR) | Person-level AUC / EER / HTER |
|---|---|---|---|---|---|---|
| Pupil size | 18.5% | 3.7% | 2/27 (p = .77) | 5.3% | 22.2% (5.6%) | 0.722 / 0.370 / 0.426 |
| **Rate** | **29.6%** | **5.6%** | **3/27 (binomial p = .52)** | 5.2% | **29.6% (5.6%)** | 0.841 / 0.185 / 0.352 |

**Exp 1 versus Exp 2.** The excess control-vs-control false-positive rate of the rate rule seen in Exp 1 (15.5%, 9/29) is absent in Exp 2 (5.6%, 3/27). In Exp 2 the paper's rule is calibrated on control data and the calibrated detection equals the nominal detection (29.6%). The Exp 1 image-identity effect therefore did not replicate in Exp 2 (we have not tested why; the experiments differ in the gaze instruction and in the control faces drawn, and both use one fixed image per control, so this is not established as a property of the instruction). The Exp 1 calibrated detection of 34.5% versus the published 55.2% stands, but it rests on 9 false alarms in one experiment, so the paper should state it with that caveat.

## 3. Honest baseline under our protocol

Participant-disjoint outer folds (3 repeats x 10 folds per experiment, seeds 2026-2028, frozen in `folds.json`). The paper's rule is refit on training participants only: the peak sample is the argmax of |mean familiar - mean control| over the pooled training trials, and a 1-D logistic calibration (training trials only) turns the value at the peak into a signed score. Models are scored on held-out participants and each participant's held-out scores are tested with the paper's one-tailed Student t-test (familiar vs both controls, alpha .05). Rows `B0_rate_peak` (the paper's best measure) and `B0_size_peak` in the tables of section 4 are this baseline.

| | Exp 1 (N = 29) | Exp 2 (N = 27) |
|---|---|---|
| B0 rate: detected at alpha .05 | 52.9% ± 1.6 | 29.6% ± 0.0 |
| B0 rate: FPR on control 1 vs 2 | 15.5% | 5.6% |
| B0 rate: detection at calibrated threshold (FPR) | 34.5% (6.9%) | 29.6% (5.6%) |
| B0 rate: trial AUC / EER / HTER | 0.578 / 0.445 / 0.445 | 0.559 / 0.456 / 0.453 |
| B0 rate: person AUC / EER / HTER | 0.743 / 0.316 / 0.356 | 0.839 / 0.210 / 0.352 |
| B0 size: detected at alpha .05 | 33.3% ± 1.6 | 18.5% ± 0.0 |
| B0 size: trial AUC | 0.548 | 0.540 |

The honest baseline matches the reproduction (52.9% vs 51.7% in Exp 1, 29.6% vs 29.6% in Exp 2), so selecting the peak on training participants only costs nothing (the paper already used leave-one-out). What changes is the reading: single trials are barely separable (trial AUC 0.54-0.58), the rate rule's Exp 1 detection of about 52% comes with a 15.5% false-positive rate on unfamiliar faces, and the calibrated figure is 34.5%.

**Metric definitions.** *Trial level*: all held-out familiar (positive) vs control (negative) trials, threshold 0 on the model's signed score, `edd.metrics.binary_metrics`. *Detected*: per-participant one-tailed t-test of the model's scores, familiar vs both controls. *FPR ctrl 1 vs 2*: same test on control 1 vs control 2 trials (both directions averaged), which is the nearest thing to an innocent person this dataset has: each control is one fixed unfamiliar face, like the familiar face. *Calibrated threshold*: for each participant the cut-off is the 95th percentile of the other participants' control-1-vs-2 z statistics. *Person AUC/EER/HTER*: positives are each participant's familiar-vs-control z score and negatives their control-1-vs-2 z score. There is no real innocent group, so every "specificity" here is a surrogate.

## 4. Ideas screened (all results, none dropped)

Ideas (see `phase2_chen.py`): I2a 100-ms bins of pupil size and rate + SelectKBest + logistic regression; I2b I2a plus gaze dx, dy and missing-sample fraction; I2c mean pupil/rate in the paper's group-level windows (windows chosen on the full sample, so optimistic and excluded from best-candidate selection); I3a/I3b I2a/I2b with within-person z-scoring against the person's own control trials; I4a/I4b MiniRocket (1000 kernels, fit on the other experiment's participants) + ridge, without and with person-normalised features. A0-A3 are preprocessing ablations of I3b (A0 gap interpolation instead of the authors' blink reconstruction, A1 no blink reconstruction, A2 divisive baseline, A3 200-ms baseline). Hyper-parameters (C, k, ridge alpha) are tuned inside training participants with `StratifiedGroupKFold(3)`.


**exp1, trial level** (pooled out-of-fold trials, familiar vs control; mean ± SD over 3 repeats)

| Idea | Acc | BalAcc | Sens | Spec | MCC | AUC | EER | HTER |
|---|---|---|---|---|---|---|---|---|
| B0_size_peak | 0.528 ± 0.001 | 0.527 ± 0.001 | 0.522 ± 0.002 | 0.532 ± 0.001 | 0.051 ± 0.001 | 0.548 ± 0.001 | 0.474 ± 0.001 | 0.473 ± 0.001 |
| B0_rate_peak | 0.557 ± 0.002 | 0.555 ± 0.002 | 0.549 ± 0.001 | 0.561 ± 0.003 | 0.105 ± 0.003 | 0.578 ± 0.002 | 0.445 ± 0.002 | 0.445 ± 0.002 |
| I2a_bins_pupil | 0.577 ± 0.002 | 0.572 ± 0.004 | 0.556 ± 0.008 | 0.587 ± 0.001 | 0.135 ± 0.007 | 0.593 ± 0.003 | 0.427 ± 0.003 | 0.428 ± 0.004 |
| I2b_bins_fusion | 0.579 ± 0.004 | 0.574 ± 0.005 | 0.559 ± 0.009 | 0.590 ± 0.003 | 0.141 ± 0.009 | 0.594 ± 0.001 | 0.424 ± 0.002 | 0.426 ± 0.005 |
| I2c_paperwin_pupil | 0.572 ± 0.001 | 0.572 ± 0.001 | 0.569 ± 0.002 | 0.574 ± 0.003 | 0.135 ± 0.001 | 0.595 ± 0.001 | 0.429 ± 0.002 | 0.428 ± 0.001 |
| I3a_pnorm_pupil | 0.582 ± 0.002 | 0.574 ± 0.002 | 0.550 ± 0.003 | 0.598 ± 0.001 | 0.140 ± 0.004 | 0.599 ± 0.001 | 0.426 ± 0.001 | 0.426 ± 0.002 |
| I3b_pnorm_fusion | 0.590 ± 0.005 | 0.576 ± 0.004 | 0.533 ± 0.012 | 0.618 ± 0.012 | 0.144 ± 0.007 | 0.601 ± 0.002 | 0.430 ± 0.003 | 0.424 ± 0.004 |
| I4a_minirocket_fusion | 0.577 ± 0.002 | 0.553 ± 0.004 | 0.482 ± 0.010 | 0.624 ± 0.003 | 0.102 ± 0.007 | 0.575 ± 0.004 | 0.452 ± 0.005 | 0.447 ± 0.004 |
| I4b_minirocket_fusion_pnormfeat | 0.599 ± 0.003 | 0.562 ± 0.002 | 0.453 ± 0.003 | 0.672 ± 0.006 | 0.121 ± 0.005 | 0.583 ± 0.001 | 0.441 ± 0.003 | 0.438 ± 0.002 |
| A0_gapinterp | 0.593 ± 0.003 | 0.578 ± 0.002 | 0.531 ± 0.002 | 0.624 ± 0.004 | 0.148 ± 0.004 | 0.603 ± 0.001 | 0.426 ± 0.001 | 0.422 ± 0.002 |
| A1_noblinkrecon | 0.595 ± 0.002 | 0.578 ± 0.001 | 0.527 ± 0.002 | 0.629 ± 0.002 | 0.148 ± 0.003 | 0.603 ± 0.005 | 0.424 ± 0.005 | 0.422 ± 0.001 |
| A2_divisive | 0.591 ± 0.003 | 0.575 ± 0.005 | 0.527 ± 0.017 | 0.624 ± 0.009 | 0.144 ± 0.009 | 0.600 ± 0.003 | 0.429 ± 0.005 | 0.425 ± 0.005 |
| A3_base200 | 0.588 ± 0.001 | 0.574 ± 0.001 | 0.535 ± 0.002 | 0.614 ± 0.002 | 0.142 ± 0.002 | 0.601 ± 0.002 | 0.429 ± 0.001 | 0.426 ± 0.001 |

**exp1, per participant** (paper's one-tailed t-test on the model's scores; innocent surrogate = control 1 vs control 2; mean ± SD over 3 repeats)

| Idea | Detected at alpha .05 | FPR ctrl 1 vs 2 | Detection at calibrated threshold | FPR at calibrated threshold | Person AUC | Person EER | Person HTER |
|---|---|---|---|---|---|---|---|
| B0_size_peak | 0.333 ± 0.016 | 0.052 ± 0.000 | 0.345 ± 0.028 | 0.063 ± 0.008 | 0.721 ± 0.010 | 0.362 ± 0.014 | 0.368 ± 0.008 |
| B0_rate_peak | 0.529 ± 0.016 | 0.155 ± 0.000 | 0.345 ± 0.000 | 0.069 ± 0.000 | 0.743 ± 0.010 | 0.316 ± 0.022 | 0.356 ± 0.008 |
| I2a_bins_pupil | 0.517 ± 0.028 | 0.115 ± 0.016 | 0.471 ± 0.016 | 0.057 ± 0.008 | 0.792 ± 0.007 | 0.264 ± 0.016 | 0.299 ± 0.022 |
| I2b_bins_fusion | 0.471 ± 0.016 | 0.057 ± 0.016 | 0.483 ± 0.028 | 0.063 ± 0.008 | 0.803 ± 0.019 | 0.264 ± 0.022 | 0.310 ± 0.014 |
| I2c_paperwin_pupil | 0.586 ± 0.000 | 0.098 ± 0.008 | 0.471 ± 0.016 | 0.069 ± 0.000 | 0.818 ± 0.003 | 0.276 ± 0.000 | 0.276 ± 0.000 |
| I3a_pnorm_pupil | 0.517 ± 0.028 | 0.132 ± 0.016 | 0.437 ± 0.016 | 0.069 ± 0.000 | 0.813 ± 0.004 | 0.264 ± 0.022 | 0.310 ± 0.024 |
| I3b_pnorm_fusion | 0.506 ± 0.043 | 0.103 ± 0.014 | 0.460 ± 0.016 | 0.063 ± 0.008 | 0.808 ± 0.005 | 0.276 ± 0.014 | 0.316 ± 0.022 |
| I4a_minirocket_fusion | 0.575 ± 0.043 | 0.069 ± 0.000 | 0.494 ± 0.016 | 0.063 ± 0.008 | 0.798 ± 0.020 | 0.270 ± 0.016 | 0.270 ± 0.035 |
| I4b_minirocket_fusion_pnormfeat | 0.563 ± 0.016 | 0.075 ± 0.008 | 0.563 ± 0.016 | 0.057 ± 0.008 | 0.820 ± 0.015 | 0.276 ± 0.024 | 0.276 ± 0.024 |
| A0_gapinterp | 0.540 ± 0.033 | 0.075 ± 0.008 | 0.529 ± 0.016 | 0.069 ± 0.000 | 0.836 ± 0.014 | 0.247 ± 0.022 | 0.287 ± 0.022 |
| A1_noblinkrecon | 0.598 ± 0.016 | 0.069 ± 0.014 | 0.563 ± 0.033 | 0.057 ± 0.008 | 0.821 ± 0.012 | 0.276 ± 0.024 | 0.253 ± 0.008 |
| A2_divisive | 0.506 ± 0.043 | 0.080 ± 0.008 | 0.471 ± 0.016 | 0.063 ± 0.008 | 0.815 ± 0.006 | 0.264 ± 0.016 | 0.310 ± 0.024 |
| A3_base200 | 0.506 ± 0.016 | 0.092 ± 0.008 | 0.483 ± 0.028 | 0.063 ± 0.008 | 0.826 ± 0.014 | 0.264 ± 0.035 | 0.310 ± 0.000 |

**exp1, paired against B0_rate_peak** (repeat-averaged scores; Holm over the 11 non-baseline entries; p raw (Holm))

| Idea | Δ within-person AUC | Wilcoxon p | detections gained/lost | McNemar p |
|---|---|---|---|---|
| A0_gapinterp | +0.023 | 0.046 (0.501) | 2/2 | 1.000 (1.000) |
| A1_noblinkrecon | +0.022 | 0.056 (0.506) | 3/2 | 1.000 (1.000) |
| A2_divisive | +0.021 | 0.092 (0.586) | 2/3 | 1.000 (1.000) |
| A3_base200 | +0.023 | 0.062 (0.506) | 1/2 | 1.000 (1.000) |
| I2a_bins_pupil | +0.016 | 0.149 (0.630) | 0/1 | 1.000 (1.000) |
| I2b_bins_fusion | +0.018 | 0.084 (0.586) | 0/2 | 0.500 (1.000) |
| I2c_paperwin_pupil | +0.018 | 0.126 (0.630) | 1/0 | 1.000 (1.000) |
| I3a_pnorm_pupil | +0.018 | 0.126 (0.630) | 0/1 | 1.000 (1.000) |
| I3b_pnorm_fusion | +0.022 | 0.048 (0.501) | 2/3 | 1.000 (1.000) |
| I4a_minirocket_fusion | +0.000 | 0.932 (1.000) | 3/3 | 1.000 (1.000) |
| I4b_minirocket_fusion_pnormfeat | +0.002 | 0.767 (1.000) | 3/3 | 1.000 (1.000) |

**exp1, paired against B0_size_peak** (repeat-averaged scores; Holm over the 11 non-baseline entries; p raw (Holm))

| Idea | Δ within-person AUC | Wilcoxon p | detections gained/lost | McNemar p |
|---|---|---|---|---|
| A0_gapinterp | +0.054 | 0.001 (0.014) | 9/2 | 0.065 (0.458) |
| A1_noblinkrecon | +0.053 | 0.001 (0.012) | 10/2 | 0.039 (0.386) |
| A2_divisive | +0.052 | 0.002 (0.016) | 8/2 | 0.109 (0.656) |
| A3_base200 | +0.054 | 0.002 (0.015) | 9/3 | 0.146 (0.656) |
| I2a_bins_pupil | +0.047 | 0.013 (0.039) | 8/2 | 0.109 (0.656) |
| I2b_bins_fusion | +0.049 | 0.007 (0.030) | 8/3 | 0.227 (0.656) |
| I2c_paperwin_pupil | +0.049 | 0.005 (0.029) | 8/0 | 0.008 (0.086) |
| I3a_pnorm_pupil | +0.049 | 0.006 (0.029) | 8/2 | 0.109 (0.656) |
| I3b_pnorm_fusion | +0.054 | 0.002 (0.016) | 8/2 | 0.109 (0.656) |
| I4a_minirocket_fusion | +0.032 | 0.014 (0.039) | 8/1 | 0.039 (0.386) |
| I4b_minirocket_fusion_pnormfeat | +0.033 | 0.016 (0.039) | 8/1 | 0.039 (0.386) |

**exp2, trial level** (pooled out-of-fold trials, familiar vs control; mean ± SD over 3 repeats)

| Idea | Acc | BalAcc | Sens | Spec | MCC | AUC | EER | HTER |
|---|---|---|---|---|---|---|---|---|
| B0_size_peak | 0.531 ± 0.001 | 0.531 ± 0.000 | 0.531 ± 0.001 | 0.531 ± 0.002 | 0.058 ± 0.001 | 0.540 ± 0.000 | 0.469 ± 0.001 | 0.469 ± 0.000 |
| B0_rate_peak | 0.540 ± 0.002 | 0.547 ± 0.003 | 0.568 ± 0.006 | 0.526 ± 0.001 | 0.089 ± 0.005 | 0.559 ± 0.001 | 0.456 ± 0.003 | 0.453 ± 0.003 |
| I2a_bins_pupil | 0.557 ± 0.001 | 0.548 ± 0.002 | 0.524 ± 0.004 | 0.573 ± 0.001 | 0.092 ± 0.004 | 0.566 ± 0.002 | 0.451 ± 0.004 | 0.452 ± 0.002 |
| I2b_bins_fusion | 0.584 ± 0.001 | 0.574 ± 0.000 | 0.545 ± 0.003 | 0.604 ± 0.003 | 0.141 ± 0.001 | 0.604 ± 0.005 | 0.424 ± 0.002 | 0.426 ± 0.000 |
| I2c_paperwin_pupil | 0.544 ± 0.001 | 0.549 ± 0.001 | 0.563 ± 0.003 | 0.535 ± 0.002 | 0.092 ± 0.002 | 0.558 ± 0.001 | 0.451 ± 0.000 | 0.451 ± 0.001 |
| I3a_pnorm_pupil | 0.557 ± 0.001 | 0.546 ± 0.002 | 0.513 ± 0.007 | 0.580 ± 0.004 | 0.087 ± 0.004 | 0.566 ± 0.007 | 0.454 ± 0.003 | 0.454 ± 0.002 |
| I3b_pnorm_fusion | 0.594 ± 0.002 | 0.570 ± 0.003 | 0.498 ± 0.005 | 0.641 ± 0.001 | 0.134 ± 0.005 | 0.597 ± 0.002 | 0.432 ± 0.004 | 0.430 ± 0.003 |
| I4a_minirocket_fusion | 0.574 ± 0.005 | 0.553 ± 0.003 | 0.492 ± 0.011 | 0.615 ± 0.011 | 0.102 ± 0.006 | 0.569 ± 0.001 | 0.453 ± 0.002 | 0.447 ± 0.003 |
| I4b_minirocket_fusion_pnormfeat | 0.593 ± 0.002 | 0.555 ± 0.005 | 0.442 ± 0.016 | 0.668 ± 0.006 | 0.108 ± 0.010 | 0.569 ± 0.007 | 0.448 ± 0.003 | 0.445 ± 0.005 |
| A0_gapinterp | 0.603 ± 0.002 | 0.577 ± 0.002 | 0.497 ± 0.001 | 0.656 ± 0.003 | 0.148 ± 0.004 | 0.601 ± 0.001 | 0.426 ± 0.001 | 0.423 ± 0.002 |
| A1_noblinkrecon | 0.596 ± 0.002 | 0.572 ± 0.003 | 0.500 ± 0.006 | 0.644 ± 0.001 | 0.139 ± 0.005 | 0.597 ± 0.002 | 0.428 ± 0.002 | 0.428 ± 0.003 |
| A2_divisive | 0.593 ± 0.002 | 0.568 ± 0.003 | 0.492 ± 0.006 | 0.644 ± 0.001 | 0.130 ± 0.006 | 0.595 ± 0.002 | 0.433 ± 0.006 | 0.432 ± 0.003 |
| A3_base200 | 0.594 ± 0.004 | 0.573 ± 0.004 | 0.509 ± 0.004 | 0.637 ± 0.004 | 0.139 ± 0.007 | 0.598 ± 0.002 | 0.430 ± 0.000 | 0.427 ± 0.004 |

**exp2, per participant** (paper's one-tailed t-test on the model's scores; innocent surrogate = control 1 vs control 2; mean ± SD over 3 repeats)

| Idea | Detected at alpha .05 | FPR ctrl 1 vs 2 | Detection at calibrated threshold | FPR at calibrated threshold | Person AUC | Person EER | Person HTER |
|---|---|---|---|---|---|---|---|
| B0_size_peak | 0.185 ± 0.000 | 0.037 ± 0.000 | 0.222 ± 0.000 | 0.056 ± 0.000 | 0.722 ± 0.003 | 0.370 ± 0.000 | 0.426 ± 0.000 |
| B0_rate_peak | 0.296 ± 0.000 | 0.056 ± 0.000 | 0.296 ± 0.000 | 0.056 ± 0.000 | 0.839 ± 0.005 | 0.210 ± 0.017 | 0.352 ± 0.000 |
| I2a_bins_pupil | 0.481 ± 0.030 | 0.025 ± 0.009 | 0.568 ± 0.035 | 0.068 ± 0.009 | 0.784 ± 0.007 | 0.235 ± 0.031 | 0.272 ± 0.023 |
| I2b_bins_fusion | 0.617 ± 0.017 | 0.074 ± 0.015 | 0.519 ± 0.030 | 0.062 ± 0.009 | 0.884 ± 0.005 | 0.216 ± 0.017 | 0.228 ± 0.009 |
| I2c_paperwin_pupil | 0.309 ± 0.017 | 0.037 ± 0.000 | 0.333 ± 0.000 | 0.068 ± 0.009 | 0.809 ± 0.004 | 0.253 ± 0.023 | 0.346 ± 0.009 |
| I3a_pnorm_pupil | 0.469 ± 0.046 | 0.019 ± 0.015 | 0.531 ± 0.035 | 0.068 ± 0.009 | 0.778 ± 0.013 | 0.253 ± 0.023 | 0.278 ± 0.015 |
| I3b_pnorm_fusion | 0.593 ± 0.030 | 0.074 ± 0.000 | 0.543 ± 0.017 | 0.074 ± 0.000 | 0.861 ± 0.009 | 0.173 ± 0.017 | 0.241 ± 0.015 |
| I4a_minirocket_fusion | 0.494 ± 0.046 | 0.043 ± 0.009 | 0.531 ± 0.035 | 0.056 ± 0.000 | 0.793 ± 0.011 | 0.272 ± 0.023 | 0.259 ± 0.030 |
| I4b_minirocket_fusion_pnormfeat | 0.457 ± 0.035 | 0.043 ± 0.009 | 0.494 ± 0.017 | 0.068 ± 0.009 | 0.773 ± 0.014 | 0.272 ± 0.023 | 0.284 ± 0.023 |
| A0_gapinterp | 0.605 ± 0.035 | 0.062 ± 0.009 | 0.580 ± 0.035 | 0.068 ± 0.009 | 0.856 ± 0.002 | 0.185 ± 0.026 | 0.228 ± 0.023 |
| A1_noblinkrecon | 0.580 ± 0.017 | 0.068 ± 0.017 | 0.568 ± 0.035 | 0.056 ± 0.000 | 0.830 ± 0.006 | 0.210 ± 0.009 | 0.247 ± 0.009 |
| A2_divisive | 0.580 ± 0.035 | 0.068 ± 0.009 | 0.556 ± 0.000 | 0.062 ± 0.009 | 0.860 ± 0.007 | 0.198 ± 0.031 | 0.241 ± 0.015 |
| A3_base200 | 0.543 ± 0.017 | 0.068 ± 0.009 | 0.519 ± 0.030 | 0.068 ± 0.009 | 0.854 ± 0.008 | 0.179 ± 0.009 | 0.265 ± 0.009 |

**exp2, paired against B0_rate_peak** (repeat-averaged scores; Holm over the 11 non-baseline entries; p raw (Holm))

| Idea | Δ within-person AUC | Wilcoxon p | detections gained/lost | McNemar p |
|---|---|---|---|---|
| A0_gapinterp | +0.045 | 0.010 (0.096) | 9/1 | 0.021 (0.215) |
| A1_noblinkrecon | +0.039 | 0.020 (0.121) | 9/1 | 0.021 (0.215) |
| A2_divisive | +0.043 | 0.015 (0.121) | 8/1 | 0.039 (0.312) |
| A3_base200 | +0.043 | 0.015 (0.121) | 8/1 | 0.039 (0.312) |
| I2a_bins_pupil | +0.009 | 0.386 (1.000) | 6/2 | 0.289 (0.625) |
| I2b_bins_fusion | +0.044 | 0.008 (0.083) | 10/1 | 0.012 (0.129) |
| I2c_paperwin_pupil | -0.001 | 0.915 (1.000) | 1/1 | 1.000 (1.000) |
| I3a_pnorm_pupil | +0.007 | 0.562 (1.000) | 6/1 | 0.125 (0.625) |
| I3b_pnorm_fusion | +0.044 | 0.011 (0.101) | 8/1 | 0.039 (0.312) |
| I4a_minirocket_fusion | +0.011 | 0.386 (1.000) | 6/1 | 0.125 (0.625) |
| I4b_minirocket_fusion_pnormfeat | +0.009 | 0.578 (1.000) | 6/1 | 0.125 (0.625) |

**exp2, paired against B0_size_peak** (repeat-averaged scores; Holm over the 11 non-baseline entries; p raw (Holm))

| Idea | Δ within-person AUC | Wilcoxon p | detections gained/lost | McNemar p |
|---|---|---|---|---|
| A0_gapinterp | +0.067 | 0.001 (0.005) | 12/1 | 0.003 (0.034) |
| A1_noblinkrecon | +0.061 | 0.002 (0.009) | 12/1 | 0.003 (0.034) |
| A2_divisive | +0.065 | 0.001 (0.005) | 11/1 | 0.006 (0.051) |
| A3_base200 | +0.065 | 0.001 (0.006) | 11/1 | 0.006 (0.051) |
| I2a_bins_pupil | +0.031 | 0.069 (0.208) | 8/1 | 0.039 (0.078) |
| I2b_bins_fusion | +0.066 | 0.000 (0.004) | 13/1 | 0.002 (0.020) |
| I2c_paperwin_pupil | +0.021 | 0.073 (0.208) | 5/2 | 0.453 (0.453) |
| I3a_pnorm_pupil | +0.029 | 0.091 (0.208) | 8/0 | 0.008 (0.051) |
| I3b_pnorm_fusion | +0.066 | 0.000 (0.004) | 11/1 | 0.006 (0.051) |
| I4a_minirocket_fusion | +0.033 | 0.034 (0.136) | 8/0 | 0.008 (0.051) |
| I4b_minirocket_fusion_pnormfeat | +0.030 | 0.023 (0.115) | 8/0 | 0.008 (0.051) |

**Reading the screening tables.**

1. Trial-level AUC is 0.55-0.60 everywhere; the best entries (I2b, I3b, A0-A3) reach 0.60 in both experiments. They differ by 0.01 or less, so they should not be ranked against each other.
2. Against the size rule every model with a bin-based feature set is reliably better in Exp 1 (Holm-adjusted Wilcoxon p between .012 and .039) and I2b, I3b and A0-A3 in Exp 2 (p = .004-.006), but against the rate rule **no entry survives Holm correction** in either experiment (smallest adjusted p: .083 for I2b in Exp 2, .501 in Exp 1). Verdict: no reliable difference from the paper's rate rule, although the point estimates favour the fusion models, especially in Exp 2 (within-person AUC +0.044, 10 detections gained vs 1 lost, unadjusted McNemar p = .012, adjusted .129).
3. Detection at alpha .05 is not comparable between models because their false-positive rates differ. At the calibrated threshold (FPR 5-7%) the rate rule detects 34.5% in Exp 1 and 29.6% in Exp 2, the fusion ideas (I2b, I3b, I4a, I4b) 46-56% in Exp 1 and 49-54% in Exp 2.
4. MiniRocket (I4a, I4b) is not better than the binned logistic models and has a worse trial AUC in both experiments. Within-person normalisation (I3) did not help on its own (I3a vs I2a: +0.006 and -0.000 trial AUC; I3b vs I2b: +0.007 and -0.007).
5. Preprocessing ablations make little difference (trial AUC 0.595-0.603). Skipping blink reconstruction altogether (A1) is as good as the authors' method for the screened models, so the blink reconstruction matters for reproducing the paper's rule exactly (10/29 and 15/29 vs 9/29 and 13/29 with simple interpolation) but not for the models.
6. The screening Holm family has 11 comparisons per baseline and experiment; nothing here was tuned on outer-fold results. The ablations A0-A3 reuse I3b and were counted in the family.

**Best-candidate selection (fixed before any full-data result, recorded in RESUME.md).** Highest mean person-level AUC over both experiments among the ideas, excluding I2c. Result: `I2b_bins_fusion`, 0.844 (Exp 1 0.803, Exp 2 0.884), ahead of I3b (0.835), I3a (0.796), I4b (0.797), I4a (0.795), I2a (0.788). The ablations A0 (0.846), A3 (0.840) and A2 (0.837) are variants of I3b rather than ideas in the `idea_specs` catalogue, and the rule was applied to ideas. They differ from I2b by 0.002-0.007, which is within noise. I2b is also the entry with the lowest FPR among the fusion models in Exp 1.

## 5. Best candidate (`I2b_bins_fusion`) vs the honest baseline vs the published numbers

**Model.** 100-ms bin means (0-2500 ms) of baseline-corrected pupil size, rate of pupil change, horizontal gaze toward the motion direction (dx), vertical gaze (dy) and the fraction of missing samples (125 features); median imputation, standardisation, SelectKBest (k in {20, all}) and L2 logistic regression (C in {0.003, 0.03, 0.3}), all tuned on training participants only. The inner CV picked C = 0.003 and k = all 125 in the full-sample refit of both experiments, so the model is a heavily regularised average of standardised features. Each participant is scored from their own held-out trials only; no per-person statistics are used.

**How the point estimates were computed.** Scores are averaged over the 3 repeats of the frozen participant-disjoint 10-fold CV (each trial is tested once per repeat). 95% CIs are participant-cluster bootstraps (1000 resamples, `full_ci`); the CIs for detection counts are Clopper-Pearson. "Calibrated" uses the 95th percentile of the other participants' control-1-vs-2 statistics as threshold. The leave-one-participant-out run (LOSO, the paper's own design) is in `results.json` under `best`.

| | Exp 1 (N = 29) | | Exp 2 (N = 27) | |
|---|---|---|---|---|
| | Paper's rate rule (our re-implementation) | I2b | Paper's rate rule | I2b |
| Published detection | 16/29 = 55.2% | n/a | 9/27 = 33.3% | n/a |
| Detected at alpha .05, CV scores | 16/29 = 55.2% (CP 35.7-73.6) | 14/29 = 48.3% (CP 29.4-67.5) | 8/27 = 29.6% (CP 13.8-50.2) | 17/27 = 63.0% (CP 42.4-80.6) |
| Detected, LOSO design | 15/29 (51.7%) | 15/29 (51.7%) | 8/27 (29.6%) | 17/27 (63.0%) |
| FPR on control 1 vs 2 | 15.5% (6.9-24.1) | 5.2% (0.0-12.1) | 5.6% (0.0-11.1) | 7.4% (1.9-13.0) |
| Detection at calibrated threshold (FPR) | 34.5% (13.8-69.0) at 6.9% | 48.3% (24.1-69.0) at 5.2% | 29.6% (11.1-63.0) at 5.6% | 51.9% (29.6-85.2) at 7.4% |
| Trial AUC | 0.578 (0.541-0.609) | 0.596 (0.564-0.625) | 0.559 (0.533-0.584) | 0.605 (0.576-0.634) |
| Trial EER / HTER | 0.445 / 0.444 | 0.422 / 0.424 | 0.454 / 0.453 | 0.431 / 0.428 |
| Trial balanced accuracy | 0.556 | 0.576 | 0.547 | 0.572 |
| Person AUC (familiar vs control-1-vs-2 test) | 0.736 (0.597-0.855) | 0.811 (0.688-0.905) | 0.840 (0.716-0.951) | 0.881 (0.775-0.970) |
| Person EER / HTER | 0.293 / 0.345 | 0.259 / 0.293 | 0.222 / 0.352 | 0.204 / 0.222 |

The paper's rule in this table is the repeat-averaged `B0_rate_peak`. Section 4 reports the same rule as mean ± SD over single repeats (52.9% and 29.6% detected). Sensitivity and specificity at trial level for I2b are in the section 4 tables.

**Paired difference, I2b minus the paper's rate rule** (identical folds, same participants; participant bootstrap, 2000 resamples; `delta_ci.json`):

| | Exp 1 | Exp 2 |
|---|---|---|
| Trial AUC | +0.018 (-0.001, +0.039) | +0.046 (+0.019, +0.074) |
| Person AUC | +0.075 (-0.001, +0.155) | +0.041 (-0.097, +0.177) |
| Detected at alpha .05 (count) | -2 (-5, 0) | +9 (+3, +15) |
| Wilcoxon on within-person AUC, raw (Holm over 11) | .084 (.586) | .008 (.083) |
| McNemar on detection, raw (Holm over 11) | .500 (1.000) | .012 (.129) |

**Against the size rule** (the paper's pupil-size measure) I2b is better after Holm in both experiments (adjusted Wilcoxon p = .030 and .004; trial AUC +0.048 [0.017, 0.080] and +0.065 [0.037, 0.091]).

**Permutation test** (within-participant shuffle of the familiar/control/control labels, full nested pipeline rerun, repeat-0 folds, 30 permutations only, so the smallest attainable p is 1/31 = .032):

| | Observed AUC | Null AUC mean (95th pct) | p | Observed detection | Null detection mean (95th pct) | p |
|---|---|---|---|---|---|---|
| Exp 1 | 0.594 | 0.500 (0.519) | .032 | 44.8% | 4.1% (10.3%) | .032 |
| Exp 2 | 0.605 | 0.499 (0.513) | .032 | 63.0% | 5.1% (13.1%) | .032 |

No permutation reached the observed value in any of the four cases, so p sits at its floor. This shows the signal is real; it does not show that I2b beats the baseline. The null detection rate is near the nominal 5%, so the shuffled-label pipeline is calibrated.

**Why it helps (feature weights and one ablation).** Share of the summed absolute standardised coefficient by channel, refit on all participants (`coefficients_*.csv`):

| Channel | Exp 1 | Exp 2 |
|---|---|---|
| Rate of pupil change | 0.33 | 0.26 |
| Horizontal gaze (dx) | 0.26 | 0.31 |
| Missing-sample fraction | 0.18 | 0.23 |
| Vertical gaze (dy) | 0.13 | 0.11 |
| Pupil size | 0.10 | 0.09 |

The largest weights in Exp 1 are rate at 700-900 ms (positive, in line with the paper's rate cluster at 660-860 ms), rate at 1000-1200 ms (negative), and dx at 600-700 ms (positive, gaze toward the motion) and 1200-1400 ms (negative, gaze away). In Exp 2 the largest are dx at 1300-1400 ms (negative), rate at 1100-1200 ms (negative), dx at 600-700 ms (positive) and rate at 800-900 ms (positive). This is consistent with the paper's qualitative gaze finding. The one ablation that tests the mechanism is I2a (pupil and rate only) vs I2b (adding gaze and missing samples): trial AUC 0.593 to 0.594 in Exp 1 and 0.566 to 0.604 in Exp 2; person AUC 0.792 to 0.803 and 0.784 to 0.884. The gaze and missing-sample channels therefore help mainly in Exp 2 (free viewing) and hardly in Exp 1 (central fixation). The gain over the rate rule thus combines (i) pooling many time bins instead of one peak sample (I2a already gains +0.016 and +0.009 in within-person AUC over the rate rule, not significant) and (ii) the gaze and missing-sample channels. The missing-sample fraction (blinks and tracking loss) carries 18-23% of the weight; we have not tested whether it reflects recognition or a trial-order or image confound, and no per-channel ablation was run beyond I2a vs I2b.

**Cross-experiment transfer** (fit on all participants of one experiment with nested tuning, applied unchanged to all participants of the other; no refit. The calibrated threshold uses only the target experiment's own control-vs-control statistics):

| Train to test | Rule | Detected (alpha .05) | FPR ctrl 1 vs 2 | Calibrated detection (FPR) | Trial AUC | Trial EER / HTER | Person AUC |
|---|---|---|---|---|---|---|---|
| Exp 1 to Exp 2 | I2b | 14/27 = 51.9% | 3.7% | 59.3% (33.3-81.5) at 5.6% | 0.597 (0.564-0.630) | 0.430 / 0.429 | 0.833 (0.741-0.926) |
| | Rate rule (peak from Exp 1) | 9/27 = 33.3% | 14.8% (7.4-24.1) | 25.9% (3.7-44.4) at 7.4% | 0.534 (0.496-0.572) | 0.478 / 0.482 | 0.638 (0.505-0.776) |
| | Size rule | 5/27 = 18.5% | 3.7% | 18.5% (0.0-40.7) at 5.6% | 0.540 (0.526-0.554) | 0.469 / 0.471 | 0.706 (0.573-0.840) |
| Exp 2 to Exp 1 | I2b | 14/29 = 48.3% | 5.2% | 51.7% (27.6-89.7) at 5.2% | 0.606 (0.580-0.631) | 0.424 / 0.418 | 0.927 (0.837-0.993) |
| | Rate rule (peak from Exp 2) | 9/29 = 31.0% | 6.9% | 31.0% (6.9-55.2) at 6.9% | 0.546 (0.520-0.572) | 0.467 / 0.467 | 0.734 (0.617-0.844) |
| | Size rule | 9/29 = 31.0% | 5.2% | 34.5% (10.3-55.2) at 6.9% | 0.550 (0.528-0.571) | 0.470 / 0.469 | 0.722 (0.583-0.828) |

I2b keeps its within-experiment performance when transferred (trial AUC 0.597 and 0.606 against 0.596 and 0.605 within), whereas the single-peak rate rule loses most of its signal when its peak sample comes from the other experiment (trial AUC 0.534 and 0.546; the Exp 1 peak flags 14.8% of Exp 2 control-vs-control comparisons). The signal I2b uses is therefore not specific to one experiment's gaze instruction or sample. Caveat: whether any participant took part in both experiments was not verified (subject numbers are separate per experiment).

**Does it beat the honest baseline and the published number?** Under the protocol's winner rule: **no reliable difference from the paper's rate rule after Holm correction** in either experiment. The direction is consistent but the size differs. Exp 2 shows a clear raw difference (trial AUC +0.046 with a CI excluding 0; +9 participants detected; raw McNemar p = .012, Holm .129). Exp 1 shows a small AUC gain (+0.018, CI touching 0) and 2 fewer participants flagged at alpha .05, which is the cost of a 10-point lower false-positive rate (5.2% vs 15.5%). At a calibrated threshold (FPR 5-7% in all cells) the point estimates are 48.3% vs 34.5% (Exp 1) and 51.9% vs 29.6% (Exp 2), but the CIs overlap widely (N = 27-29, and a 5% FPR means 1-2 false alarms). Against the published numbers, 48.3% (Exp 1) is below the published 55.2% and 63.0% (Exp 2) is above the published 33.3%; the published figures come without an unfamiliar-face check, so a like-for-like comparison is only possible at a calibrated FPR. I2b was chosen on the same data on which it is evaluated (selection by person AUC among the screened ideas), so the raw p-values above are optimistic; the Holm correction covers the 11 screened alternatives, not the choice among them.

**Caveats.** (1) N = 29 and 27, one lab, one stimulus set (parents' faces). (2) No genuine innocent group: control 1 vs control 2 is a surrogate, where each control is one fixed unfamiliar face repeated 48 times, so every "specificity" and FPR here is a surrogate. (3) The SDs over repeats reflect fold-seed variance only, not sampling variance; use the bootstrap CIs. (4) The permutation test used 30 permutations. (5) Trial types, motion direction and onset times are known by design, so a field system would need the same stimulus log. (6) A per-participant decision uses 144 familiar-or-control trials, about 15 minutes of recording at 6.4 s per trial.

## 6. Efficiency

Hardware (`hardware_info()`): Windows 11 (10.0.26200), Intel64 Family 6 Model 142 Stepping 10 (8 logical CPUs), Python 3.14.6, single-threaded (`threadpool_limits(1)`), while other long jobs were running on the same machine, so absolute latencies are upper bounds. Timed from the raw 1 kHz EyeLink samples of one participant (`exp1/Face_16.edf`, held out when the models were fitted). A trial is the 13-face RSVP stream (6409 samples = 6.4 s on average). Per-trial cost includes blink reconstruction, 1 kHz to 100 Hz downsampling, epoching, baseline, rate smoothing, feature binning and scoring. Per-participant cost is 144 trials plus the one-tailed t-test. Source: `efficiency.json`, `scripts/phase2_chen_efficiency.py`.

| | Paper's rate rule (re-implementation) | I2b_bins_fusion |
|---|---|---|
| Latency per trial, p50 / p95 | 3.52 / 4.80 ms | 20.14 / 22.24 ms |
| Latency per participant (144 trials), p50 / p95 | 375.8 / 396.4 ms | 2366 / 2484 ms |
| Throughput | 277.9 trials/s = 1.78 M raw samples/s | 47.7 trials/s = 0.306 M raw samples/s |
| Real-time factor vs the 1000 Hz sensor (processing time / data time) | 0.00055 (1820x faster than real time) | 0.0031 (318x faster than real time) |
| Raw signal to 100 Hz epoch stage per trial, p50 / p95 | 3.53 / 3.80 ms | 4.67 / 5.07 ms |
| Model-only inference per trial (precomputed features), p50 / p95 | not applicable (one sample read-out) | 2.64 / 3.87 ms |
| Peak Python-tracked memory, one participant | 281 kB | 506 kB |
| Model size (pickle) | 703 bytes | 8495 bytes |
| Stored parameters | 3 (peak index, logistic slope and intercept; detection itself only needs the peak index) | 501 (125 weights + bias, 125 scaler means and 125 scales, 125 imputer medians) |

"FPS equivalent": the sensor delivers 1000 samples per second; the two detectors process raw samples at about 1.8 M and 0.31 M samples per second. The trial latency of I2b (20.1 ms) is more than the sum of the epoch stage and the model (about 7 ms) because it also includes the binning of the five channels into 125 features. Reading the EDF file with the edfapi costs 25.2 microseconds per sample (34.9 s for the 1,384,405 RSVP-phase samples of one participant file); this is an offline file-decoding cost and is excluded from the table. Memory is measured with `tracemalloc` (Python and NumPy allocations only, not the interpreter).

## 7. Recommendation

**Worth a full run: yes, as a benchmark and with a changed question, not as a claim that a model beats the paper.**

1. **Report false positives alongside detection.** The paper's headline rate figure comes without any unfamiliar-face check. In Exp 1 its rule flags 9/29 people on two unfamiliar faces (15.5%, binomial p = .0016) and at a calibrated FPR of about 7% it detects 34.5% instead of 55.2%. This did not replicate in Exp 2 (5.6%), so it is a property of that experiment's stimuli or sample rather than of the rule in general. It needs item-level analysis of the control faces before the paper states it.
2. **A multi-bin, multi-channel model (I2b) is the most promising screened idea.** Trial AUC is about 0.60 in both experiments, detection is 48-52% at 5-7% FPR (against 30-35% for the rate rule at the same FPR), performance is unchanged when trained on one experiment and tested on the other, and its cost is 6x higher per trial and 6x per participant than the paper's rule but still more than 300 times faster than real time. The gains over the paper's rule are not significant after Holm correction, so a full run should be powered for them: pool both experiments (56 participants) with leave-one-experiment-out and participant-disjoint CV, and preregister I2b.
3. **To try next (not run here):** per-channel ablations of the missing-sample and gaze channels to rule out blink-rate or oculomotor confounds; a genuine innocent group, or more unfamiliar faces per person, in place of the control-1-vs-2 surrogate; 1000 or more permutations; a check of whether any participant took part in both experiments.
4. **Not worth pursuing at this size:** MiniRocket (no better than the binned logistic models, trial AUC 0.57-0.58) and within-person normalisation (no gain).
