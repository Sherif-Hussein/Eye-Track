# Phase 7, Track D: sensing robustness without pupil, and face-area size (card)

Pre-registration `PREREG.md`; deviations `DEVIATIONS.md`; code `scripts/phase7_trackD_run.py` (features and CV, imports the Track C harness unchanged), `phase7_trackD_stats.py`, `phase7_trackD_card.py` (tables). Simulated from chin-rest EyeLink data decimated to 30 Hz (Delmas raw, 75 participants, frozen folds, pre-registered noise draws). Everything here is exploratory; there is no real webcam recording.

Reference: P4 SVM 1000 Hz, AUC 0.7811. AUC = image-level, pooled within each of the six sets, mean over sets; a level's AUC is the mean of 3 noise draws. Delta = AUC minus 0.7811, paired participant bootstrap (2000 resamples, seed 0). p = raw one-sided non-inferiority p at margin 0.02. Holm: over the 14 no-pupil configurations (D1 family) and within each route (7).

Harness checks (all passed): with full-size AOI and pupil, the new code path reproduces Track C's clean sample-based AUC 0.7723 (scores identical) and clean I-VT AUC 0.7854 (max score difference 1.1e-12); AOI patching at scale 1.0 is the identity; vectorised features at 75% and 50% match the unvectorised references (max difference 3.4e-13 sample-based, 2.3e-13 I-VT); the layout file's rectangles and px/deg equal the code's.

Feature sets: no pupil = I-VT 11 features (13 minus `pupil_face`, `pupil_trial`), sample-based 10 features (12 minus the same two). With pupil = the Track C models (13 and 12; stored scores reused).

## D1. No-pupil webcam simulation (full-size face areas)

| level (deg) | route | AUC [95% CI] | Delta vs 0.7811 [95% CI] | p raw | Holm p (14) | Holm p (route, 7) | Delta vs own clean |
|---|---|---|---|---|---|---|---|
| 0 (clean) | I-VT | 0.7863 [0.7437, 0.8241] | +0.0052 [-0.0142, +0.0232] | 0.006 | 0.084 | 0.042 | +0.0000 |
| 0 (clean) | sample-based | 0.7732 [0.7276, 0.8138] | -0.0079 [-0.0274, +0.0102] | 0.120 | 1.000 | 0.531 | +0.0000 |
| 1 | I-VT | 0.7539 [0.7188, 0.7870] | -0.0272 [-0.0473, -0.0086] | 0.806 | 1.000 | 1.000 | -0.0323 |
| 1 | sample-based | 0.7737 [0.7308, 0.8130] | -0.0073 [-0.0263, +0.0102] | 0.096 | 1.000 | 0.531 | +0.0006 |
| 1.5 | I-VT | 0.7431 [0.7101, 0.7743] | -0.0380 [-0.0577, -0.0197] | 0.974 | 1.000 | 1.000 | -0.0432 |
| 1.5 | sample-based | 0.7754 [0.7323, 0.8143] | -0.0057 [-0.0234, +0.0109] | 0.062 | 0.812 | 0.437 | +0.0022 |
| 2 | I-VT | 0.7274 [0.6937, 0.7592] | -0.0537 [-0.0753, -0.0354] | 1.000 | 1.000 | 1.000 | -0.0589 |
| 2 | sample-based | 0.7728 [0.7295, 0.8114] | -0.0083 [-0.0262, +0.0083] | 0.108 | 1.000 | 0.531 | -0.0004 |
| 3 | I-VT | 0.7049 [0.6797, 0.7303] | -0.0762 [-0.1014, -0.0509] | 1.000 | 1.000 | 1.000 | -0.0814 |
| 3 | sample-based | 0.7669 [0.7268, 0.8040] | -0.0142 [-0.0320, +0.0031] | 0.306 | 1.000 | 0.612 | -0.0063 |
| 4 | I-VT | 0.6466 [0.6240, 0.6694] | -0.1345 [-0.1633, -0.1052] | 1.000 | 1.000 | 1.000 | -0.1397 |
| 4 | sample-based | 0.7462 [0.7098, 0.7793] | -0.0349 [-0.0556, -0.0154] | 0.936 | 1.000 | 0.936 | -0.0270 |
| anchored | I-VT | 0.7344 [0.7008, 0.7645] | -0.0467 [-0.0677, -0.0264] | 0.995 | 1.000 | 1.000 | -0.0519 |
| anchored | sample-based | 0.7737 [0.7302, 0.8131] | -0.0074 [-0.0256, +0.0094] | 0.088 | 1.000 | 0.531 | +0.0005 |

Non-inferiority at margin 0.02 is not shown for any no-pupil configuration under Holm over 14 (smallest Holm p 0.084, I-VT clean). Within the I-VT route alone (7 tests) the clean no-pupil I-VT model is non-inferior (Holm p 0.042); it is the only one. The sample-based features have point estimates within 0.015 of the reference up to 3 deg and at the anchored level (-0.007), but the lower confidence limits (-0.023 to -0.032) cross the margin; at 4 deg the whole interval is below zero (-0.035 [-0.056, -0.015]). I-VT without pupil loses 0.03 to 0.13 from 1 to 4 deg and 0.047 at the anchored level (CI excludes zero from 1 deg). Sample-based minus I-VT (no pupil), paired: +0.020 (1 deg), +0.032 (1.5), +0.045 (2), +0.062 (3), +0.100 (4), +0.039 (anchored); every interval excludes zero (`d1_table.csv`).

### Pupil contribution (AUC with pupil minus AUC without, paired bootstrap)

| level (deg) | I-VT with | I-VT without | difference [95% CI] | sample-based with | sample-based without | difference [95% CI] |
|---|---|---|---|---|---|---|
| 0 (clean) | 0.7854 | 0.7863 | -0.0008 [-0.0065, +0.0044] | 0.7723 | 0.7732 | -0.0009 [-0.0073, +0.0052] |
| 1 | 0.7527 | 0.7539 | -0.0012 [-0.0061, +0.0035] | 0.7736 | 0.7737 | -0.0001 [-0.0050, +0.0042] |
| 1.5 | 0.7418 | 0.7431 | -0.0013 [-0.0070, +0.0038] | 0.7750 | 0.7754 | -0.0004 [-0.0052, +0.0037] |
| 2 | 0.7267 | 0.7274 | -0.0007 [-0.0058, +0.0043] | 0.7709 | 0.7728 | -0.0019 [-0.0066, +0.0025] |
| 3 | 0.7005 | 0.7049 | -0.0044 [-0.0101, +0.0013] | 0.7655 | 0.7669 | -0.0015 [-0.0061, +0.0028] |
| 4 | 0.6416 | 0.6466 | -0.0050 [-0.0120, +0.0018] | 0.7442 | 0.7462 | -0.0020 [-0.0071, +0.0027] |
| anchored | 0.7304 | 0.7344 | -0.0039 [-0.0093, +0.0015] | 0.7725 | 0.7737 | -0.0012 [-0.0063, +0.0037] |

Dropping the two pupil features does not lower AUC at any level: without pupil the AUC is 0.0001 to 0.005 higher, and every interval includes zero (the difference is with minus without, so negative means pupil did not help). In this dataset the two pupil features add no accuracy to the 30 Hz ALL_z model, so the pupil optimism the editor flagged does not change the webcam numbers here. This says nothing about other datasets (e.g. Chen, where pupil is the signal).

## D2. Face-area size sensitivity

Face areas, centres unchanged: full 300 x 400 px = 8.57 x 11.43 deg (width x height; the review's 11.5 x 8.6 deg is height x width); 75% = 225 x 300 px = 6.43 x 8.57 deg; 50% = 150 x 200 px = 4.29 x 5.72 deg (34.987 px/deg; layout file checked). Face centres are 640 px apart horizontally and 540 px vertically, i.e. 18.3 and 15.4 deg.

Delta full = AUC minus the full-size area of the same configuration (same route, pupil setting, condition, draws), paired bootstrap. Delta ref and p are descriptive (no decision family in D2).

| condition | route | pupil | area (deg) | AUC [95% CI] | Delta vs full size [95% CI] | Delta vs 0.7811 [95% CI] | p raw |
|---|---|---|---|---|---|---|---|
| clean 30 Hz | sample-based | with | 8.57 x 11.43 (100%) | 0.7723 [0.7292, 0.8122] | (reference) | -0.0088 [-0.0277, +0.0097] | 0.124 |
| clean 30 Hz | sample-based | with | 6.43 x 8.57 (75%) | 0.7676 [0.7224, 0.8077] | -0.0047 [-0.0093, -0.0005] | -0.0135 [-0.0337, +0.0059] | 0.284 |
| clean 30 Hz | sample-based | with | 4.29 x 5.72 (50%) | 0.7710 [0.7290, 0.8084] | -0.0013 [-0.0130, +0.0092] | -0.0101 [-0.0301, +0.0094] | 0.196 |
| clean 30 Hz | sample-based | no | 8.57 x 11.43 (100%) | 0.7732 [0.7276, 0.8138] | (reference) | -0.0079 [-0.0274, +0.0102] | 0.120 |
| clean 30 Hz | sample-based | no | 6.43 x 8.57 (75%) | 0.7681 [0.7207, 0.8102] | -0.0051 [-0.0096, -0.0003] | -0.0130 [-0.0338, +0.0070] | 0.268 |
| clean 30 Hz | sample-based | no | 4.29 x 5.72 (50%) | 0.7687 [0.7265, 0.8063] | -0.0045 [-0.0161, +0.0058] | -0.0124 [-0.0311, +0.0053] | 0.259 |
| clean 30 Hz | I-VT | with | 8.57 x 11.43 (100%) | 0.7854 [0.7432, 0.8231] | (reference) | +0.0044 [-0.0159, +0.0231] | 0.007 |
| clean 30 Hz | I-VT | with | 6.43 x 8.57 (75%) | 0.7831 [0.7400, 0.8206] | -0.0023 [-0.0064, +0.0020] | +0.0020 [-0.0180, +0.0218] | 0.013 |
| clean 30 Hz | I-VT | with | 4.29 x 5.72 (50%) | 0.7738 [0.7341, 0.8086] | -0.0117 [-0.0244, -0.0004] | -0.0073 [-0.0285, +0.0143] | 0.144 |
| clean 30 Hz | I-VT | no | 8.57 x 11.43 (100%) | 0.7863 [0.7437, 0.8241] | (reference) | +0.0052 [-0.0142, +0.0232] | 0.006 |
| clean 30 Hz | I-VT | no | 6.43 x 8.57 (75%) | 0.7840 [0.7405, 0.8222] | -0.0023 [-0.0061, +0.0018] | +0.0029 [-0.0170, +0.0217] | 0.010 |
| clean 30 Hz | I-VT | no | 4.29 x 5.72 (50%) | 0.7721 [0.7327, 0.8076] | -0.0142 [-0.0264, -0.0036] | -0.0090 [-0.0293, +0.0103] | 0.163 |
| anchored | sample-based | with | 8.57 x 11.43 (100%) | 0.7725 [0.7289, 0.8109] | (reference) | -0.0086 [-0.0267, +0.0090] | 0.116 |
| anchored | sample-based | with | 6.43 x 8.57 (75%) | 0.7652 [0.7230, 0.8034] | -0.0073 [-0.0135, -0.0016] | -0.0158 [-0.0346, +0.0029] | 0.380 |
| anchored | sample-based | with | 4.29 x 5.72 (50%) | 0.7524 [0.7161, 0.7878] | -0.0200 [-0.0328, -0.0079] | -0.0286 [-0.0488, -0.0079] | 0.819 |
| anchored | sample-based | no | 8.57 x 11.43 (100%) | 0.7737 [0.7302, 0.8131] | (reference) | -0.0074 [-0.0256, +0.0094] | 0.088 |
| anchored | sample-based | no | 6.43 x 8.57 (75%) | 0.7668 [0.7229, 0.8061] | -0.0069 [-0.0127, -0.0017] | -0.0143 [-0.0331, +0.0036] | 0.305 |
| anchored | sample-based | no | 4.29 x 5.72 (50%) | 0.7537 [0.7166, 0.7901] | -0.0200 [-0.0326, -0.0081] | -0.0274 [-0.0471, -0.0084] | 0.793 |
| anchored | I-VT | with | 8.57 x 11.43 (100%) | 0.7304 [0.6975, 0.7610] | (reference) | -0.0507 [-0.0732, -0.0289] | 0.999 |
| anchored | I-VT | with | 6.43 x 8.57 (75%) | 0.7232 [0.6913, 0.7534] | -0.0072 [-0.0131, -0.0019] | -0.0578 [-0.0813, -0.0361] | 1.000 |
| anchored | I-VT | with | 4.29 x 5.72 (50%) | 0.7123 [0.6814, 0.7414] | -0.0182 [-0.0290, -0.0078] | -0.0688 [-0.0942, -0.0446] | 1.000 |
| anchored | I-VT | no | 8.57 x 11.43 (100%) | 0.7344 [0.7008, 0.7645] | (reference) | -0.0467 [-0.0677, -0.0264] | 0.995 |
| anchored | I-VT | no | 6.43 x 8.57 (75%) | 0.7253 [0.6930, 0.7559] | -0.0090 [-0.0148, -0.0040] | -0.0557 [-0.0780, -0.0357] | 1.000 |
| anchored | I-VT | no | 4.29 x 5.72 (50%) | 0.7122 [0.6814, 0.7414] | -0.0221 [-0.0331, -0.0116] | -0.0689 [-0.0936, -0.0460] | 1.000 |

At the anchored level, shrinking the areas costs accuracy in both routes: about -0.007 to -0.009 at 75% and -0.018 to -0.022 at 50% (each interval excludes zero), with or without pupil. At 50% the anchored sample-based AUC is 0.754 (no pupil; Delta vs reference -0.027 [-0.047, -0.008]) and I-VT is 0.712. On clean 30 Hz data the effect is smaller and less certain: sample-based -0.005 at 75% and -0.005 at 50% (interval includes zero), I-VT -0.002 at 75% and -0.014 at 50% (interval excludes zero). The anchored offset (mean 1.45 deg) is large relative to the half-widths of the 50% area (2.1 deg horizontally, 2.9 deg vertically). Pupil makes no difference to the pattern. Not run: an offset-by-area grid (the review asks for the offset at which detection degrades as a function of area size; the plan asked only for clean and anchored).

## What this means for the paper

- The no-pupil gaze-only webcam variant gives the same numbers as the with-pupil variant (differences at most 0.005, intervals include zero). The webcam result does not depend on the EyeLink-quality pupil and can be stated for the gaze-only variant with no change in strength.
- The claim keeps its earlier strength, no more: sample-based dwell loses about 0.007 at the Kaduk-anchored level and 0.006 to 0.014 from 1 to 3 deg, but non-inferiority at margin 0.02 is not shown at any level (Holm over 14, p >= 0.08). I-VT without pupil is clearly below the reference under webcam noise (-0.027 at 1 deg, -0.047 anchored).
- With smaller face areas the claim weakens: at 75% the anchored sample-based loss vs the reference is -0.014 to -0.016 (CI includes zero), at 50% it is -0.027 (interval excludes zero). The robustness to offsets up to 3 deg is partly a property of the 8.6 x 11.4 deg areas and 15 to 18 deg spacing; smaller or closer item layouts, as in other CIT designs, were only tested down to 4.3 x 5.7 deg.
- Still simulated from chin-rest EyeLink data: no head movement, no timing jitter, no real webcam recording.
