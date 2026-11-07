"""Phase 7, Track D: build card.md tables from the finished CSVs (reads results only)."""
import json
from pathlib import Path

import pandas as pd

O = Path(__file__).resolve().parents[1] / "results" / "phase7" / "trackD"
d1 = pd.read_csv(O / "d1_table.csv")
pc = pd.read_csv(O / "d1_pupil_contribution.csv")
d2 = pd.read_csv(O / "d2_table.csv")
ck = json.loads((O / "checks.json").read_text())


def f(x, n=4, s=False):
    return f"{x:+.{n}f}" if s else f"{x:.{n}f}"


L = []
L.append("# Phase 7, Track D: sensing robustness without pupil, and face-area size (card)\n")
L.append("Pre-registration `PREREG.md`; deviations `DEVIATIONS.md`; code `scripts/phase7_trackD_run.py` (features and CV, imports the Track C harness unchanged), `phase7_trackD_stats.py`, `phase7_trackD_card.py` (tables). Simulated from chin-rest EyeLink data decimated to 30 Hz (Delmas raw, 75 participants, frozen folds, pre-registered noise draws). Everything here is exploratory; there is no real webcam recording.\n")
L.append("Reference: P4 SVM 1000 Hz, AUC 0.7811. AUC = image-level, pooled within each of the six sets, mean over sets; a level's AUC is the mean of 3 noise draws. Delta = AUC minus 0.7811, paired participant bootstrap (2000 resamples, seed 0). p = raw one-sided non-inferiority p at margin 0.02. Holm: over the 14 no-pupil configurations (D1 family) and within each route (7).\n")
L.append("Harness checks (all passed): with full-size AOI and pupil, the new code path reproduces Track C's clean sample-based AUC 0.7723 (scores identical) and clean I-VT AUC 0.7854 (max score difference 1.1e-12); AOI patching at scale 1.0 is the identity; vectorised features at 75% and 50% match the unvectorised references (max difference 3.4e-13 sample-based, 2.3e-13 I-VT); the layout file's rectangles and px/deg equal the code's.\n")
L.append("Feature sets: no pupil = I-VT 11 features (13 minus `pupil_face`, `pupil_trial`), sample-based 10 features (12 minus the same two). With pupil = the Track C models (13 and 12; stored scores reused).\n")
L.append("## D1. No-pupil webcam simulation (full-size face areas)\n")
L.append("| level (deg) | route | AUC [95% CI] | Delta vs 0.7811 [95% CI] | p raw | Holm p (14) | Holm p (route, 7) | Delta vs own clean |\n|---|---|---|---|---|---|---|---|")
for _, r in d1.iterrows():
    L.append(f"| {r.level_deg} | {'sample-based' if r.route == 'sb' else 'I-VT'} | {f(r.auc)} [{f(r.auc_lo)}, {f(r.auc_hi)}] | {f(r.delta_vs_ref, 4, True)} [{f(r.delta_lo, 4, True)}, {f(r.delta_hi, 4, True)}] | {r.p_noninferior_raw:.3f} | {r.p_holm_family14:.3f} | {r.p_holm_route7:.3f} | {f(r.delta_vs_own_clean, 4, True)} |")
L.append("\nNon-inferiority at margin 0.02 is not shown for any no-pupil configuration under Holm over 14 (smallest Holm p 0.084, I-VT clean). Within the I-VT route alone (7 tests) the clean no-pupil I-VT model is non-inferior (Holm p 0.042); it is the only one. The sample-based features have point estimates within 0.015 of the reference up to 3 deg and at the anchored level (-0.007), but the lower confidence limits (-0.023 to -0.032) cross the margin; at 4 deg the whole interval is below zero (-0.035 [-0.056, -0.015]). I-VT without pupil loses 0.03 to 0.13 from 1 to 4 deg and 0.047 at the anchored level (CI excludes zero from 1 deg). Sample-based minus I-VT (no pupil), paired: +0.020 (1 deg), +0.032 (1.5), +0.045 (2), +0.062 (3), +0.100 (4), +0.039 (anchored); every interval excludes zero (`d1_table.csv`).\n")
L.append("### Pupil contribution (AUC with pupil minus AUC without, paired bootstrap)\n")
L.append("| level (deg) | I-VT with | I-VT without | difference [95% CI] | sample-based with | sample-based without | difference [95% CI] |\n|---|---|---|---|---|---|---|")
a = pc[pc.scale == 1.0]
for lv in d1.level_deg.unique():
    i = a[(a.route == "ivt") & (a.cond == lv)].iloc[0]
    s = a[(a.route == "sb") & (a.cond == lv)].iloc[0]
    L.append(f"| {lv} | {f(i.auc_with_pupil)} | {f(i.auc_no_pupil)} | {f(i.with_minus_no, 4, True)} [{f(i.lo, 4, True)}, {f(i.hi, 4, True)}] | {f(s.auc_with_pupil)} | {f(s.auc_no_pupil)} | {f(s.with_minus_no, 4, True)} [{f(s.lo, 4, True)}, {f(s.hi, 4, True)}] |")
L.append("\nDropping the two pupil features does not lower AUC at any level: without pupil the AUC is 0.0001 to 0.005 higher, and every interval includes zero (the difference is with minus without, so negative means pupil did not help). In this dataset the two pupil features add no accuracy to the 30 Hz ALL_z model, so the pupil optimism the editor flagged does not change the webcam numbers here. This says nothing about other datasets (e.g. Chen, where pupil is the signal).\n")
L.append("## D2. Face-area size sensitivity\n")
g = ck["aoi_deg"]
L.append("Face areas, centres unchanged: full 300 x 400 px = %.2f x %.2f deg (width x height; the review's 11.5 x 8.6 deg is height x width); 75%% = 225 x 300 px = %.2f x %.2f deg; 50%% = 150 x 200 px = %.2f x %.2f deg (34.987 px/deg; layout file checked). Face centres are 640 px apart horizontally and 540 px vertically, i.e. 18.3 and 15.4 deg.\n" % (g["1.0"]["width_deg"], g["1.0"]["height_deg"], g["0.75"]["width_deg"], g["0.75"]["height_deg"], g["0.5"]["width_deg"], g["0.5"]["height_deg"]))
L.append("Delta full = AUC minus the full-size area of the same configuration (same route, pupil setting, condition, draws), paired bootstrap. Delta ref and p are descriptive (no decision family in D2).\n")
L.append("| condition | route | pupil | area (deg) | AUC [95% CI] | Delta vs full size [95% CI] | Delta vs 0.7811 [95% CI] | p raw |\n|---|---|---|---|---|---|---|---|")
for cond in ("clean", "anch"):
    for route in ("sb", "ivt"):
        for pup in ("wp", "np"):
            for _, r in d2[(d2.cond == cond) & (d2.route == route) & (d2.pupil == pup)].iterrows():
                full = r.scale == 1.0
                dfull = "(reference)" if full else f"{f(r.delta_vs_fullsize, 4, True)} [{f(r.delta_vs_fullsize_lo, 4, True)}, {f(r.delta_vs_fullsize_hi, 4, True)}]"
                L.append(f"| {'clean 30 Hz' if cond == 'clean' else 'anchored'} | {'sample-based' if route == 'sb' else 'I-VT'} | {'with' if pup == 'wp' else 'no'} | {r.aoi_deg} ({int(round(r.scale * 100))}%) | {f(r.auc)} [{f(r.auc_lo)}, {f(r.auc_hi)}] | {dfull} | {f(r.delta_vs_ref, 4, True)} [{f(r.delta_vs_ref_lo, 4, True)}, {f(r.delta_vs_ref_hi, 4, True)}] | {r.p_noninferior_raw_descriptive:.3f} |")
L.append("\nAt the anchored level, shrinking the areas costs accuracy in both routes: about -0.007 to -0.009 at 75% and -0.018 to -0.022 at 50% (each interval excludes zero), with or without pupil. At 50% the anchored sample-based AUC is 0.754 (no pupil; Delta vs reference -0.027 [-0.047, -0.008]) and I-VT is 0.712. On clean 30 Hz data the effect is smaller and less certain: sample-based -0.005 at 75% and -0.005 at 50% (interval includes zero), I-VT -0.002 at 75% and -0.014 at 50% (interval excludes zero). The anchored offset (mean 1.45 deg) is large relative to the half-widths of the 50% area (2.1 deg horizontally, 2.9 deg vertically). Pupil makes no difference to the pattern. Not run: an offset-by-area grid (the review asks for the offset at which detection degrades as a function of area size; the plan asked only for clean and anchored).\n")
L.append("## What this means for the paper\n")
L.append("- The no-pupil gaze-only webcam variant gives the same numbers as the with-pupil variant (differences at most 0.005, intervals include zero). The webcam result does not depend on the EyeLink-quality pupil and can be stated for the gaze-only variant with no change in strength.\n- The claim keeps its earlier strength, no more: sample-based dwell loses about 0.007 at the Kaduk-anchored level and 0.006 to 0.014 from 1 to 3 deg, but non-inferiority at margin 0.02 is not shown at any level (Holm over 14, p >= 0.08). I-VT without pupil is clearly below the reference under webcam noise (-0.027 at 1 deg, -0.047 anchored).\n- With smaller face areas the claim weakens: at 75% the anchored sample-based loss vs the reference is -0.014 to -0.016 (CI includes zero), at 50% it is -0.027 (interval excludes zero). The robustness to offsets up to 3 deg is partly a property of the 8.6 x 11.4 deg areas and 15 to 18 deg spacing; smaller or closer item layouts, as in other CIT designs, were only tested down to 4.3 x 5.7 deg.\n- Still simulated from chin-rest EyeLink data: no head movement, no timing jitter, no real webcam recording.\n")
(O / "card.md").write_text("\n".join(L), encoding="utf-8")
print("card written", len(L))
