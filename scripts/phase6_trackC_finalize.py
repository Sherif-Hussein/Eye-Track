"""Phase 6 Track C: consolidate results_c1/c2/c3.json into results.json and write card.md. Reads only finished outputs."""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results" / "phase6" / "trackC"
c1 = json.loads((OUT / "results_c1.json").read_text())
c2 = json.loads((OUT / "results_c2.json").read_text())
c3 = json.loads((OUT / "results_c3.json").read_text())
prov = pd.read_csv(OUT / "c2_provenance.csv")
cond = pd.DataFrame(c1["conditions"])


def row(v, c, L):
    return cond[(cond.variant == v) & (cond.cond == c) & (cond.level.astype(str) == str(L))].iloc[0]


R = {"track": "C", "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "prereg": "PREREG.md", "deviations": "DEVIATIONS.md",
     "C1": c1, "C2": {**c2, "provenance_of_r": prov.to_dict("records")}, "C3": c3,
     "completeness": {"C1_configurations": 133, "C1_checks_pass": True, "C2": "complete",
                      "C3": "complete (5 datasets; Liar's Dice omitted, see DEVIATIONS 5)"}}
(OUT / "results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")

L = []
L.append("# Phase 6, Track C: eye-tracking validity (card)\n")
L.append("Pre-registration `PREREG.md`; deviations `DEVIATIONS.md`; code `scripts/phase6_trackC_*.py`. Only the six-test sample-based family in C1 is a decision family; everything else is exploratory.\n")
L.append("## C1. Sample-based dwell features under webcam noise\n")
ck = c1["checks"]
L.append(f"Reference: P4 SVM 1000 Hz, AUC {c1['reference_auc']:.4f}. Checks: level-0 I-VT reproduces 0.7854 (out-of-fold scores identical to the stored file, max difference {ck['ivt_clean_L000_d0']['max_abs_score_diff']:.1e}); the recomputed L = 1, draw 1 I-VT run matches the stored one ({ck['ivt_all_L100_d1']['max_abs_score_diff']:.1e}); the vectorised sample-based features match an independent loop implementation (max difference {ck['sb_vectorised_vs_reference_max_abs_diff']:.1e}); the noise code reproduces `apply_noise` exactly.\n")
L.append("Full noise (offset + jitter + 10% loss), same draws as the webcam study. Delta = AUC minus 0.7811, paired participant bootstrap (2000). p = raw one-sided non-inferiority p at margin 0.02. Holm over the six primary tests in the next-to-last column.\n")
L.append("| level (deg) | I-VT AUC | I-VT Delta [95% CI] | I-VT p | sample-based AUC | Delta [95% CI] | p raw | Holm p (family of 6) | sample minus I-VT [95% CI] |")
L.append("|---|---|---|---|---|---|---|---|---|")
for lev in (0.0, 1.0, 1.5, 2.0, 3.0, 4.0):
    a = row("ivt", "clean" if lev == 0 else "all", lev)
    b = row("sb", "clean" if lev == 0 else "all", lev)
    L.append(f"| {lev:g} | {a.auc:.4f} | {a.delta:+.4f} [{a.delta_ci95[0]:+.4f}, {a.delta_ci95[1]:+.4f}] | {a.p_inferior_raw:.3f} | {b.auc:.4f} | {b.delta:+.4f} [{b.delta_ci95[0]:+.4f}, {b.delta_ci95[1]:+.4f}] | {b.p_inferior_raw:.3f} | {b.p_holm_primary_family:.3f} | {b.delta_vs_ivt_same_cond:+.4f} [{b.delta_vs_ivt_same_cond_ci95[0]:+.4f}, {b.delta_vs_ivt_same_cond_ci95[1]:+.4f}] |")
L.append("\nNon-inferiority is not shown for the sample-based features at any level, including level 0 (Holm p 0.48 to 0.95). The point estimates are within 0.02 of the reference up to 3 deg (Delta -0.006 to -0.016), but the lower confidence limits (-0.025 to -0.034) cross the margin. The clean sample-based model is itself 0.013 below the I-VT model [-0.022, -0.002]. At 4 deg it is clearly below the reference (Delta -0.037, CI excludes zero). Against I-VT under the same noise the sample-based model is better by +0.021 (1 deg) to +0.103 (4 deg), all intervals excluding zero. Adding a run count of at least two in-area samples (`SB+seg`, closest event-free analogue of n_fix) gives the same picture (`c1_all_conditions.csv`).\n")
L.append("Noise decomposition (AUC, mean of 3 draws; Delta, CI and p for every cell in `c1_all_conditions.csv`):\n")
L.append("| level | offset only I-VT | offset only sample | jitter only I-VT | jitter only sample | loss only I-VT | loss only sample | all three I-VT | all three sample |")
L.append("|---|---|---|---|---|---|---|---|---|")
for lev in (1.0, 1.5, 2.0, 3.0, 4.0):
    L.append(f"| {lev:g} | " + " | ".join(f"{row(v, c, lev).auc:.4f}" for c in ("off", "jit", "loss", "all") for v in ("ivt", "sb")) + " |")
L.append("\nJitter is what breaks I-VT: jitter alone takes it to 0.701 at 4 deg (offset alone 0.729, loss alone 0.770), and the combined effect at 1 to 3 deg is larger than each part. The sample-based dwell is flat under jitter (0.770 to 0.775). An offset costs the sample-based features only from 3 deg (0.756; 0.721 at 4 deg), since the face areas are 11.5 x 8.6 deg. 10% sample loss costs both about 0.011 to 0.013 (0.7698 I-VT, 0.7730 sample-based).\n")
a, b, c = row("ivt", "anch", "A"), row("sb", "anch", "A"), row("sbseg", "anch", "A")
L.append(f"Anchored to measured webcam precision (Kaduk et al. 2024: offset mean 1.45 deg, jitter SD 1.17 / 1.00 deg per axis from RMS sample-to-sample, 3% loss; exploratory): I-VT {a.auc:.4f} (Delta {a.delta:+.4f} [{a.delta_ci95[0]:+.4f}, {a.delta_ci95[1]:+.4f}]), sample-based {b.auc:.4f} (Delta {b.delta:+.4f} [{b.delta_ci95[0]:+.4f}, {b.delta_ci95[1]:+.4f}], raw p {b.p_inferior_raw:.3f}), sample-based plus segment count {c.auc:.4f}. Same conclusion: at measured precision I-VT loses about 0.05, the sample-based dwell about 0.009, and non-inferiority is not shown.\n")
L.append("## C2. Event-detection validity by sampling rate\n")
ev = pd.DataFrame(c2["event_stats"])
L.append("| rate (Hz) | fixations / trial | on-face fixations / trial | mean duration (ms) | median (ms) | share of window in fixations |")
L.append("|---|---|---|---|---|---|")
for _, r in ev.iterrows():
    L.append(f"| {r.rate_hz} | {r.fix_per_trial_all:.2f} | {r.fix_per_trial_on_face:.2f} | {r.mean_fix_dur_ms_pooled:.0f} | {r.median_fix_dur_ms:.0f} | {r.share_window_in_fixation:.3f} |")
agdf = pd.read_csv(OUT / "c2_parser_agreement.csv")
ci = c2["parser_agreement_bootstrap_ci95_counts"]
L.append("\nAgreement with the EyeLink parser (Pearson r). Counts = fixations with onset in 0-5 s, assigned to a face by the fixation centroid (I-VT) or mean gaze position (parser), per face. Dwell = share of 1-5 s spent in fixations on the face.\n")
L.append("| rate (Hz) | count r, trial x face (n = 19,200) [95% CI] | count r, participant x image [95% CI] | count r, image level | I-VT / parser count | dwell r, participant x image | sample-based dwell r |")
L.append("|---|---|---|---|---|---|---|")
for fs in (1000, 500, 250, 120, 60, 30):
    def g(m):
        return agdf[(agdf.metric == m) & (agdf.rate_hz == fs)].iloc[0]
    cc = g("count")
    t, p_ = ci[str(fs)]["trial_face"], ci[str(fs)]["participant_image"]
    L.append(f"| {fs} | {cc.trial_face:.3f} [{t[0]:.3f}, {t[1]:.3f}] | {cc.participant_image:.3f} [{p_[0]:.3f}, {p_[1]:.3f}] | {cc.image_level:.3f} | {cc.mean_ivt / cc.mean_parser:.2f} | {g('dwell_2').participant_image:.3f} | {g('sample_dwell_2_vs_parser_dwell_2').participant_image:.3f} |")
L.append("\n**Verdict on 0.77 versus 0.68.** The paper should report 0.68 at 30 Hz (0.676 trial x face, 0.682 participant x image) and state the unit. The participant x image definition returns 0.97 at 120 Hz and 0.88 at 60 Hz, which are the other two figures of the earlier card, so those stand and 0.77 does not. None of the 16 variants tried (unit, mean or sum, Pearson or Spearman, 64- or 32-trial set, the published label rule, within-participant, trial level; `c2_provenance.csv`) gives 0.77 at 30 Hz. The nearest are group 1 alone (0.79), the published-table label rule (0.85) and the image-level mean (0.95). I could not trace where 0.77 came from, so it should be dropped. Note that at 1000 Hz the agreement is 0.91 to 0.93, below its 120 Hz value (0.96 to 0.97), because the unfiltered 1000 Hz I-VT splits fixations that the parser keeps whole.\n")
L.append("At 30 Hz fixations per trial fall from 16.3 to 9.7 and mean duration rises from 237 to 392 ms; the I-VT finds 60% of the parser's per-face fixations. Dwell agreement stays at r = 0.99 (participant x image) at every rate, and sample-based dwell agrees with parser dwell at 0.98 at every rate. The 30 Hz result therefore holds for dwell features, not for fixation counts.\n")
s = pd.read_csv(OUT / "c2_saccade_detectability.csv")
L.append("Smallest detectable saccade. Velocity span 2h/fs with h = max(1, round(0.004 fs)) samples; the 30 deg/s threshold needs at least this displacement between the two samples (noise-free bound). Empirical columns: share of transitions between consecutive parser fixations (n = 72,671) that I-VT keeps as two separate fixations, by amplitude.\n")
L.append("| rate (Hz) | half-width (samples) | span (ms) | min displacement (deg) | min fixation enforced (ms) | resolved 0-0.5 deg | 0.5-1 | 1-2 | 2-4 | 4-8 | >= 8 |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in s.iterrows():
    L.append(f"| {int(r.rate_hz)} | {int(r.half_width_samples)} | {r.velocity_span_ms:.1f} | {r.min_displacement_deg_noise_free:.2f} | {r.min_fixation_ms_enforced:.1f} | " + " | ".join(f"{r[f'resolved_pct_{k}']:.0f}%" for k in ("0-0.5", "0.5-1", "1-2", "2-4", "4-8", ">=8")) + " |")
L.append("\nThe window is 8 ms only at 250 Hz and above; below that it is one sample either side (16.7, 33.3, 66.7 ms). At 30 Hz saccades under 2 deg cannot reach the threshold (1-2 deg: 11% resolved; 0.5-1 deg: 0.5%); at 60 Hz saccades under 1 deg cannot (0.5-1 deg: 31%). The 4-8 deg bin loses 25 to 34% at every rate (a missing sample at a fixation midpoint), so that loss is not a rate effect.\n")
L.append("## C3. Data-quality table\n")
d, ce, ch, pr_ = c3["delmas"], c3["celniak"], c3["chen"], c3["preti"]
v = d["calibration_validation_error_deg"]
L.append("| dataset | device | rate | eyes / head | % missing or invalid samples | % trials / participants excluded | calibration or validation error | class difference in quality |")
L.append("|---|---|---|---|---|---|---|---|")
L.append(f"| Delmas | EyeLink 1000 | 1000 Hz | right eye; chin and forehead rest, 57 cm | {d['pct_samples_missing']:.2f}% of samples in the 5 s displays (participants 0 to {d['pct_samples_missing_range_participants'][1]:.1f}%); by group {', '.join(f'G{k} {x:.1f}%' for k, x in d['pct_samples_missing_by_group'].items())} | image rows 5.7% removed by the authors (4,525 of 4,800); trials 0%; participants 0 of 75 | HV9 validation, mean error {v['mean_of_participant_means']:.2f} deg (median {v['median']:.2f}; participant means {v['range_participant_means'][0]:.2f} to {v['range_participant_means'][1]:.2f}); 26 of 192 validations labelled POOR; 1 failed validation excluded | G2 (concealment) loses most data (2.4% against 0.6 to 0.9%) |")
vc, cd = ce["calibration_validation_error_deg"], ce["class_difference"]
L.append(f"| Celniak | Tobii Pro Fusion | 250 Hz (38 recordings), 120 Hz (1) | both eyes; remote, head free | {ce['pct_samples_missing']:.1f}% in the question slides with at least one eye invalid ({ce['pct_samples_missing_range_participants'][0]:.0f} to {ce['pct_samples_missing_range_participants'][1]:.0f}% per recording); both eyes invalid 21.5% | slides 0% by us; 19 of 39 recordings have a slide below 50% valid (147 of 819 slides); participants: not available | validation accuracy mean {vc['validation_accuracy']['mean']:.2f} deg (median {vc['validation_accuracy']['median']:.2f}, max {vc['validation_accuracy']['max']:.2f}); calibration accuracy median {vc['calibration_accuracy']['median']:.2f} (max {vc['calibration_accuracy']['max']:.1f}); precision RMS median {vc['validation_precision_rms']['median']:.2f}; 1 recording has no validation value | validation accuracy differs by class: AUC {cd['validation_accuracy_deg']['auc']:.2f} [{cd['validation_accuracy_deg']['ci95'][0]:.2f}, {cd['validation_accuracy_deg']['ci95'][1]:.2f}], medians {cd['validation_accuracy_deg']['median_class1']:.2f} (guilty) vs {cd['validation_accuracy_deg']['median_class0']:.2f}; missing data does not (AUC {cd['pct_either_eye_invalid_slides']['auc']:.2f}) |")
for e, nm in (("exp1", "Chen exp 1"), ("exp2", "Chen exp 2")):
    x = ch[e]
    vv = x["validation"]
    md = x["missing_diff_familiar_minus_control_pct_points"]
    L.append(f"| {nm} | EyeLink 1000 | 1000 Hz | monocular; chin rest | {x['pct_samples_missing_pupil_rsvp_phase']:.1f}% of pupil samples in the RSVP phase (participants {x['pct_samples_missing_range_participants'][0]:.1f} to {x['pct_samples_missing_range_participants'][1]:.1f}%) | participants {len(x['n_participants_excluded_by_authors_scripts'])} of {x['n_edf_files']} ({x['pct_participants_excluded']:.0f}%, authors' scripts); trials {x['pct_trials_excluded_baseline_rules']:.1f}% (baseline rules) | first validation per participant ({vv['n_participants_with_validation']} of {x['n_participants_used']} participants have one): mean {vv['first_validation_per_participant']['mean']:.2f} deg, median {vv['first_validation_per_participant']['median']:.2f}, range {vv['first_validation_per_participant']['range'][0]:.2f} to {vv['first_validation_per_participant']['range'][1]:.2f} | familiar minus control missing share {md['mean']:+.2f} points [{md['ci95'][0]:+.2f}, {md['ci95'][1]:+.2f}] (no difference) |")
pc = pr_["class_difference"]["blink_flagged_fixation_share_liar_vs_truth"]
L.append(f"| Preti | not reported | not available | not available | not available (fixation-level data); 7.0% of fixation rows flagged as blink | 4.3% error trials; 0.27% without a pupil value; 3 of 60 participants removed by the authors | not available | blink share differs: liar median {pc['median_class1']:.3f} vs {pc['median_class0']:.3f}, AUC {pc['auc']:.2f} [{pc['ci95'][0]:.2f}, {pc['ci95'][1]:.2f}] |")
L.append("| Colasanti (RealEye) | webcam, 1080p at 60 fps, RealEye software | 60 Hz nominal (gaze at least every 16 ms); effective rate not available | not available; no head support (online) | not available (aggregate tables) | 3 of 160 participants (1.8%) excluded by the authors; trials not available | RealEye platform accuracy 113 px on average (90 to 156), quoted in the paper; per participant: not available | not available |")
L.append("\nLiar's Dice is omitted (DEVIATIONS 5). Celniak's missing-data share is large (1 to 73% between recordings) but does not separate the classes; the validation error does. In Delmas the data loss separates group G2 from the others. Chen validation values come from the EDF messages and exist for 22 of the used participants in each experiment.\n")
L.append("## Claims in the paper that these change\n")
L.append("- \"I-VT agrees with the EyeLink parser at r = 0.97, 0.88 and 0.77\": 0.77 becomes 0.68, with the unit stated. Add dwell agreement r = 0.99 at every rate.")
L.append("- \"Fixations\" at 30 Hz are dwell segments (392 ms mean, 60% of the parser count). State the velocity window per rate and the saccade floor (2 deg at 30 Hz, 1 deg at 60 Hz).")
L.append("- Webcam claim: the loss comes from jitter interacting with I-VT. Sample-based dwell loses 0.009 at measured webcam precision against 0.051 for I-VT, but non-inferiority at margin 0.02 is not shown at any level, level 0 included. The axis of Fig. 5c needs relabelling and the decomposition added. Pupil features still need a pupil channel that webcams do not give.")
L.append("- Table 1 and the reporting checklist: device, rate, eyes, head support, data loss and validation error are in the C3 table. Celniak's class difference in validation error (AUC 0.82) is a data-quality result.\n")
(OUT / "card.md").write_text("\n".join(L), encoding="utf-8")
print("ok")
