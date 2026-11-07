"""Task 2.2 (EXPLORATORY), final step: assemble rate_table.csv, results.json and card.md from the stage outputs
(rows_delmas.csv, rows_chen.csv, rows_webcam.csv, event_diag.csv, timing.csv, results_*.json, repro_*.json).
Reported rates: 1000/100 Hz reference (stored) and 30/20/15/10 Hz. (60 Hz rows that exist in rows_delmas.csv / rows_chen.csv are not reported.)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent
KEEP = {1000, 100, 30, 20, 15, 10}
LAB = {"P4_svm": "published 4-feature SVM", "ALLz_lr": "our 13 standardised features, L2 logistic regression"}

d = pd.read_csv(OUT / "rows_delmas.csv")
d["model"] = d.model.map(LAB)
c = pd.read_csv(OUT / "rows_chen.csv")
w = pd.read_csv(OUT / "rows_webcam.csv")
tab = pd.concat([d, c, w], ignore_index=True)
tab = tab[tab.rate_hz.isin(KEEP)].copy()
tab["ci_note"] = np.where(tab.dataset == "Delmas", "AUC CI: participant-cluster bootstrap, first 300 of 2000 resamples (webcam rows: mean over 3 draws); delta CI: all 2000 paired resamples",
                          "AUC and delta CI: 2000 paired participant-cluster resamples")
tab.loc[(tab.dataset == "Delmas") & tab.model.isin(LAB.values()), "ci_note"] = "AUC CI: first 300 of 2000 paired participant-cluster resamples (as the original curve); delta CI: all 2000"
tab["within_0.02_of_30Hz_point"] = tab.delta_vs_30hz >= -0.02
tab["lower_CI_above_-0.02"] = tab.delta_lo >= -0.02
tab["status"] = "EXPLORATORY (not preregistered)"
tab = tab.rename(columns={"auc_lo": "ci_lo", "auc_hi": "ci_hi", "delta_lo": "delta_ci_lo", "delta_hi": "delta_ci_hi"})
cols = ["dataset", "model", "condition", "rate_hz", "auc", "ci_lo", "ci_hi", "delta_vs_30hz", "delta_ci_lo", "delta_ci_hi",
        "within_0.02_of_30Hz_point", "lower_CI_above_-0.02", "sb_minus_ivt", "sb_minus_ivt_lo", "sb_minus_ivt_hi", "source", "ci_note", "status"]
tab = tab[cols]
order = {1000: 0, 100: 0, 30: 1, 20: 2, 15: 3, 10: 4}
tab["_o"] = tab.rate_hz.map(order)
tab = tab.sort_values(["dataset", "model", "condition", "_o"], kind="stable").drop(columns="_o")
tab.to_csv(OUT / "rate_table.csv", index=False)


def lowest_within(sub, ci=False):
    """Lowest tested rate such that every tested rate from 30 Hz down to it has delta >= -0.02 (point estimate, or lower CI limit)."""
    s = sub[sub.rate_hz.isin([30, 20, 15, 10])].sort_values("rate_hz", ascending=False)
    ok = None
    for _, r in s.iterrows():
        v = r.delta_ci_lo if ci else r.delta_vs_30hz
        if v >= -0.02:
            ok = int(r.rate_hz)
        else:
            break
    return ok


groups = {}
for (ds, m, cond), sub in tab.groupby(["dataset", "model", "condition"]):
    groups[(ds, m, cond)] = (lowest_within(sub), lowest_within(sub, ci=True))

ev = pd.read_csv(OUT / "event_diag.csv")
tm = pd.read_csv(OUT / "timing.csv")
timing_json = tm.to_dict("records")


def f(x, n=3):
    return f"{x:.{n}f}"


def md_table(df, cols, heads, fmt):
    rows = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(fmt[i](r[c_]) if c_ in r and pd.notna(r[c_]) else "" for i, c_ in enumerate(cols)) + " |")
    return "\n".join(rows)


def rate_fmt(x):
    return "1000 (stored)" if x == 1000 else ("100 (stored)" if x == 100 else f"{int(x)}")


def rate_block(sub):
    sub = sub.copy()
    sub["auc_ci"] = [f"{a:.4f} [{l:.4f}, {h:.4f}]" for a, l, h in zip(sub.auc, sub.ci_lo, sub.ci_hi)]
    sub["delta_ci"] = ["(reference)" if r == 30 else f"{a:+.4f} [{l:+.4f}, {h:+.4f}]" for r, a, l, h in zip(sub.rate_hz, sub.delta_vs_30hz, sub.delta_ci_lo, sub.delta_ci_hi)]
    sub["within"] = ["" if r == 30 else ("yes" if p else "NO") + (" (CI also)" if (p and q) else "") for r, p, q in zip(sub.rate_hz, sub["within_0.02_of_30Hz_point"], sub["lower_CI_above_-0.02"])]
    return md_table(sub, ["rate_hz", "auc_ci", "delta_ci", "within"], ["Rate (Hz)", "AUC [95% CI]", "Delta vs 30 Hz [95% CI]", "Within 0.02 of 30 Hz"],
                    [rate_fmt, str, str, str])


def sect(ds, m, cond):
    sub = tab[(tab.dataset == ds) & (tab.model == m) & (tab.condition == cond)]
    return rate_block(sub)


lines = []
A = lines.append
A("# Task 2.2: lowest usable sampling rate (card)\n")
A("**Status: EXPLORATORY. Not preregistered.** Plan: `docs/PLAN.md` Task 2.2. Output folder `results/phase9/low_rate/`. Scripts: `s1_extract.py` (EDF -> 30/20/15/10 Hz samples and features), `s2_delmas_curve.py`, `s3_chen_curve.py`, `s4_webcam.py`, `s5_events_timing.py`, `s6_assemble.py`, helper `lr_common.py`, chain `run_all.sh`; logs `log_*.txt`; per-stage outputs `rows_*.csv`, `results_*.json`, `repro_*.json`. One process, single thread throughout; no existing script or result was edited (existing code is imported unchanged; the source of each reused piece is stated at the top of each script).\n")
A("## What was run\n")
A("All settings other than the sampling rate are those of the originals: same frozen folds, seeds, tuning grids and nested cross-validation.\n")
A("1. **Delmas et al. (EyeLink, 1000 Hz recordings, 75 participants, six group sets).** Source: `scripts/phase2_delmas_raw.py` stage `rate`, drawn by `scripts/phase5_figures.py` into `paper/figures/data_delmas_rate_curve.csv`. New rates 20, 15 and 10 Hz made by anti-aliased polyphase resampling of the 1000 Hz samples (`edd.delmas_raw.degrade`), I-VT 30 deg/s with 60 ms minimum, the same per-face features. Two models: the published four features with an RBF-SVM (`P4_svm`), and our 13 standardised features (no microsaccades) with L2 logistic regression (`ALLz_lr`). Metric: image-level AUC, pooled within each of the six sets, mean over sets (repeat-averaged out-of-fold scores). The 1000 Hz row is the stored result.")
A("2. **Chen et al. (released 100 Hz epochs, trial-level AUC).** Source: `scripts/phase4_unified.py` stage `chen_rate` (`results/phase4/unified/results_chen_rate.json`). New rates 20, 15, 10 Hz by anti-aliased polyphase resampling of the 100 Hz epochs (`edd.unified.decimate_to_rate`). Two models: our pipeline (`unified`: 25 x 100 ms bins of pupil and gaze, within-participant z-scores, L2 logistic regression) and `I2b` (pupil bins + gaze, 125 features). The 100 Hz row is the stored result.")
A("3. **Delmas, measured-webcam-noise condition** (`scripts/phase6_trackC_c1.py` condition `anch`: 1.45 deg mean offset per participant, per-sample jitter SD 1.66/sqrt(2) deg (x) and 1.42/sqrt(2) deg (y), 3% sample loss; three noise draws), at 30, 20, 15 and 10 Hz, no pupil features: dwell computed directly from gaze samples on each face area (`sb`, 10 features) against fixed-threshold event detection (I-VT 30 deg/s, 60 ms, 11 features). A noise-free (clean) run of each route at each rate is the reference. Offsets are the same per participant at every rate; jitter and loss are drawn per sample at that rate.")
A("4. **Event-detection diagnostic** (Delmas, clean data) like Supplementary Table S10 (`scripts/phase6_trackC_c2.py` functions): fixations per trial, mean fixation duration, smallest saccade the velocity window can detect (30 deg/s x window), and Pearson r of per-item fixation count and per-item dwell with the tracker's own (EyeLink parser) values.")
A("5. **Processing time per participant** (Delmas, 64 trials x 5.5 s of 1000 Hz samples = 352 s of data) from the 1000 Hz samples to per-image scores, median of 3 rounds per participant after a warm-up, median over 5 participants, one thread.\n")
A("## Reproduction check (done before any new rate; all passed with zero difference)\n")
rd = json.loads((OUT / "repro_delmas.json").read_text())
rc = json.loads((OUT / "repro_chen.json").read_text())
rw = json.loads((OUT / "repro_webcam.json").read_text())
re_ = json.loads((OUT / "repro_events.json").read_text())
rf = json.loads((OUT / "repro_features_30hz.json").read_text())
A("| Quantity at 30 Hz | Stored | Re-run here | Max abs difference in out-of-fold scores |\n|---|---|---|---|")
A(f"| Delmas, 13 features + LR, AUC (repeat-averaged, pooled per set, mean over sets) | {f(rd['ALLz_lr']['repeat_averaged_pooled_auc_stored_curve'], 4)} | {f(rd['ALLz_lr']['repeat_averaged_pooled_auc_new'], 4)} | {rd['ALLz_lr']['max_abs_oof_score_diff']} |")
A(f"| Same model, mean of per-fold AUCs (screening log, \"0.781\") | {f(rd['ALLz_lr']['mean_of_fold_auc_stored'], 4)} | {f(rd['ALLz_lr']['mean_of_fold_auc_new'], 4)} | |")
A(f"| Delmas, published 4 features + SVM, AUC | {f(rd['P4_svm']['repeat_averaged_pooled_auc_stored_curve'], 4)} | {f(rd['P4_svm']['repeat_averaged_pooled_auc_new'], 4)} | {rd['P4_svm']['max_abs_oof_score_diff']} |")
A(f"| Chen, our pipeline, trial AUC | {f(rc['unified']['trial_auc_stored'], 4)} | {f(rc['unified']['trial_auc_new'], 4)} | {rc['unified']['max_abs_score_diff_vs_stored_checkpoint']} |")
A(f"| Chen, pupil bins + gaze (I2b), trial AUC | {f(rc['I2b']['trial_auc_stored'], 4)} | {f(rc['I2b']['trial_auc_new'], 4)} | {rc['I2b']['max_abs_score_diff_vs_stored_checkpoint']} |")
for key, lab in (("ivt_clean_30Hz", "Webcam study, I-VT, no pupil, clean"), ("sb_clean_30Hz", "Webcam study, sample dwell, no pupil, clean"),
                 ("ivt_anch_30Hz", "Webcam study, I-VT, no pupil, measured noise"), ("sb_anch_30Hz", "Webcam study, sample dwell, no pupil, measured noise")):
    A(f"| {lab}, AUC (Phase 7 Track D `d1_table.csv`) | {f(rw[key]['auc_stored_trackD'], 4)} | {f(rw[key]['auc_new'], 4)} | |")
A(f"| Event diagnostic: fixations per trial / mean fixation duration (Track C2) | {f(re_['30Hz_fix_per_trial_all']['stored'])} / {f(re_['30Hz_mean_fix_dur_ms']['stored'], 1)} ms | {f(re_['30Hz_fix_per_trial_all']['new'])} / {f(re_['30Hz_mean_fix_dur_ms']['new'], 1)} ms | |")
A(f"\nAlso identical: the 19,200 per-face feature rows at 30 Hz against `data/derived/delmas_raw_face_trials.csv.gz` (max difference {rf['max_abs_diff_trial_features_30Hz']:.1e}, floating-point only), the 30 Hz decimated samples against the earlier webcam cache, and the edge-based Chen feature builders used for 20/15/10 Hz against `unified.chen_features` / `chen_i2b_features` at 30 Hz (difference 0.0).\n")

A("## Result 1. Delmas, clean EyeLink data, image-level AUC\n")
A("AUC CI: participant-cluster bootstrap, first 300 of 2000 resamples (as the original curve). Delta CI: all 2000 resamples, paired (all rates evaluated on the same resamples).\n")
for m in LAB.values():
    A(f"**{m}**\n")
    A(sect("Delmas", m, "clean EyeLink (anti-aliased resampling)"))
    A("")
A("## Result 2. Chen, trial-level AUC\n")
A("2000 paired participant-cluster resamples (same seed and scheme as the stored analysis).\n")
for m in ("our pipeline (unified)", "pupil bins + gaze (I2b)"):
    A(f"**{m}**\n")
    A(sect("Chen", m, "released 100 Hz epochs, anti-aliased resampling"))
    A("")
A("Adaptations needed for rates that do not split 100 ms into whole samples (the originals only handle rates that do): bin means use sample edges round(k x rate / 10) (one or two samples per 100 ms bin at 15 Hz; identical to the original reshape at the other rates), and the I2b pupil-rate smoothing window (250 ms, `round(0.25 x rate)` samples) would be 2 samples at 10 Hz, which makes the Hanning weights 0/0, so a 3-sample window (300 ms) is used at 10 Hz.\n")

A("## Result 3. Delmas, webcam noise, dwell from samples versus I-VT, no pupil features\n")
A("Mean over 3 noise draws; AUC CI first 300 of 2000 paired resamples, delta CI all 2000 (resampling the mean over draws).\n")
for cond in ("clean, no added noise", "measured webcam values (1.45 deg offset, ~1.1 deg jitter, 3% loss), mean of 3 draws"):
    for m in tab[(tab.dataset == "Delmas") & (tab.condition == cond)].model.unique():
        A(f"**{m}; {cond}**\n")
        A(sect("Delmas", m, cond))
        A("")
sbm = tab[(tab.model.str.startswith("sb")) & tab.sb_minus_ivt.notna()].copy()
sbm["cond"] = np.where(sbm.condition.str.startswith("clean"), "clean", "measured noise")
sbm["diff"] = [f"{a:+.4f} [{l:+.4f}, {h:+.4f}]" for a, l, h in zip(sbm.sb_minus_ivt, sbm.sb_minus_ivt_lo, sbm.sb_minus_ivt_hi)]
sbm = sbm.sort_values(["cond", "rate_hz"], ascending=[True, False])
A("**Sample dwell minus I-VT at the same rate (paired)**\n")
A(md_table(sbm, ["cond", "rate_hz", "diff"], ["Condition", "Rate (Hz)", "AUC difference [95% CI]"], [str, lambda x: str(int(x)), str]))
A("")

A("## Result 4. Event-detection diagnostic (Delmas, clean data)\n")
A("Reference: the tracker's own parser at 1000 Hz. r = Pearson correlation with the parser's per-item value over participant x item cells (trial x face and image-level values are in `event_diag_agreement_long.csv`).\n")
e = ev.copy()
e["rate"] = e.rate_hz.astype(str)
cols = ["rate", "fix_per_trial_all", "mean_fix_dur_ms", "median_fix_dur_ms", "velocity_window_ms", "min_detectable_saccade_deg", "count_r_participant_x_image",
        "dwell_1_r_participant_x_image", "dwell_2_r_participant_x_image", "sample_dwell_1_vs_parser_r_participant_x_image", "sample_dwell_2_vs_parser_r_participant_x_image"]
heads = ["Rate (Hz)", "Fixations / trial", "Mean fix. duration (ms)", "Median (ms)", "Velocity window (ms)", "Smallest saccade detectable (deg)", "r fixation count", "r dwell 0.2-1 s",
         "r dwell 1-5 s", "r sample dwell 0.2-1 s", "r sample dwell 1-5 s"]
A(md_table(e, cols, heads, [str, lambda x: f(x, 2), lambda x: f(x, 0), lambda x: f(x, 0), lambda x: f(x, 0), lambda x: f(x, 0)] + [lambda x: f(x, 3)] * 5))
A("")
A("**Per-item fixation counts and dwell, last two columns:** \"sample dwell\" is the share of valid samples on the face area with no event detection; \"dwell\" columns use the I-VT fixations.\n")

A("## Result 5. Processing time per participant (Delmas, from 1000 Hz samples to per-image scores)\n")
A("`resample` = anti-aliased resampling from 1000 Hz (a simulation step: a real sensor delivers the low rate directly, so it is not a deployed cost); `features + score` = I-VT, features, aggregation, z-scoring and the logistic-regression score. Median over 5 participants of the median of 3 rounds; min-max of the 5 participants' totals in brackets. This laptop's timings are noisy (the min-max range is wide, and the 15 and 10 Hz resampling medians vary between runs), so only large differences should be read.\n")
t = tm.copy()
t["rate"] = t.rate_hz.map(lambda x: "1000 (no resampling)" if x == 1000 else str(int(x)))
t["tot"] = [f"{a:.0f} [{b:.0f}-{c_:.0f}]" for a, b, c_ in zip(t.total_ms, t.total_ms_min, t.total_ms_max)]
for impl in ("original loop code", "vectorised code"):
    A(f"**{impl}**\n")
    A(md_table(t[t.implementation == impl], ["rate", "resample_ms", "features_score_ms", "tot", "real_time_factor"],
               ["Rate (Hz)", "Resample (ms)", "Features + score (ms)", "Total (ms) [range]", "Real-time factor (x)"], [str, lambda x: f(x, 1), lambda x: f(x, 1), str, lambda x: f(x, 0)]))
    A("")

A("## Plain summary (5 lines)\n")
g = groups
def gl(key):
    return g[key]
dsvm, dlr = gl(("Delmas", LAB["P4_svm"], "clean EyeLink (anti-aliased resampling)")), gl(("Delmas", LAB["ALLz_lr"], "clean EyeLink (anti-aliased resampling)"))
cun, cib = gl(("Chen", "our pipeline (unified)", "released 100 Hz epochs, anti-aliased resampling")), gl(("Chen", "pupil bins + gaze (I2b)", "released 100 Hz epochs, anti-aliased resampling"))
wc = {k: v for k, v in g.items() if k[0] == "Delmas" and k[2].startswith("measured")}
wcl = {k: v for k, v in g.items() if k[0] == "Delmas" and k[2].startswith("clean, no")}
sb_a = [v for k, v in wc.items() if k[1].startswith("sb")][0]
iv_a = [v for k, v in wc.items() if k[1].startswith("I-VT")][0]
sb_c = [v for k, v in wcl.items() if k[1].startswith("sb")][0]
iv_c = [v for k, v in wcl.items() if k[1].startswith("I-VT")][0]
def hz(v):
    return "no tested rate" if v is None else f"{v} Hz"
def val(m, r, col="delta_vs_30hz", cond="clean EyeLink (anti-aliased resampling)", ds="Delmas"):
    return float(tab[(tab.dataset == ds) & (tab.model == m) & (tab.condition == cond) & (tab.rate_hz == r)][col].iloc[0])
A(f"1. Delmas, clean EyeLink: the published four-feature SVM stays within 0.02 AUC of 30 Hz down to {hz(dsvm[0])} (delta {val(LAB['P4_svm'], 10):+.3f}, CI lower limit {val(LAB['P4_svm'], 10, 'delta_ci_lo'):+.3f}, so 10 Hz is not shown within the margin; {hz(dsvm[1])} is, by the CI); our 13-feature LR stays within 0.02 down to {hz(dlr[0])} (delta {val(LAB['ALLz_lr'], 15):+.3f} at 15 Hz, CI lower limit {val(LAB['ALLz_lr'], 15, 'delta_ci_lo'):+.3f}; by the CI only {hz(dlr[1])}) and loses {abs(val(LAB['ALLz_lr'], 10)):.3f} at 10 Hz.")
A(f"2. Chen (100 Hz epochs, trial-level AUC): both models stay within 0.02 of 30 Hz at every rate tested, down to {hz(min(x for x in (cun[0], cib[0])))} for both models; the largest change is {max(abs(tab[(tab.dataset == 'Chen') & tab.rate_hz.isin([20, 15, 10])].delta_vs_30hz)):.4f} AUC and the lower CI limits are above -0.004 (Chen's pupil signal is slow; it does not depend on the rate in this range).")
A(f"3. Dwell computed from samples (no pupil) under the measured webcam noise stays within 0.02 of 30 Hz down to {hz(sb_a[0])} (CI-supported to {hz(sb_a[1])}; delta {val('sb: dwell from samples, no event detection (no pupil)', 10, cond='measured webcam values (1.45 deg offset, ~1.1 deg jitter, 3% loss), mean of 3 draws'):+.3f} at 10 Hz); it is not below the I-VT route at any rate under noise (+{val('sb: dwell from samples, no event detection (no pupil)', 30, 'sb_minus_ivt', cond='measured webcam values (1.45 deg offset, ~1.1 deg jitter, 3% loss), mean of 3 draws'):.3f} at 30 Hz, +{val('sb: dwell from samples, no event detection (no pupil)', 10, 'sb_minus_ivt', cond='measured webcam values (1.45 deg offset, ~1.1 deg jitter, 3% loss), mean of 3 draws'):.3f} at 10 Hz).")
A(f"4. Event detection (I-VT) breaks at 10 Hz on clean data (delta {val('I-VT event detection 30 deg/s, 60 ms (no pupil)', 10, cond='clean, no added noise'):+.3f}; within 0.02 down to {hz(iv_c[0])}), and the fixation count it produces already agrees poorly with the tracker's parser at 30 Hz (r {float(ev[ev.rate_hz.astype(str) == '30'].count_r_participant_x_image.iloc[0]):.2f}, falling to {float(ev[ev.rate_hz.astype(str) == '10'].count_r_participant_x_image.iloc[0]):.2f} at 10 Hz) while the smallest detectable saccade grows from 2 to 6 deg; under measured webcam noise I-VT stays within 0.02 of its own 30 Hz value (down to {hz(iv_a[0])}) only because its 30 Hz value is already low (it is higher at 20, 15, 10 Hz).")
A(f"5. Dwell per item agrees with the tracker's parser at r >= {float(ev[ev.rate_hz.astype(str) == '10'].dwell_2_r_participant_x_image.iloc[0]):.2f} even at 10 Hz (sample dwell r >= {float(ev[ev.rate_hz.astype(str) == '10'].sample_dwell_2_vs_parser_r_participant_x_image.iloc[0]):.2f}), so dwell-based recognition measures need far fewer samples than fixation counts; the cost of the whole pipeline from resampled samples is about 5 ms (vectorised code) or 0.1-0.16 s (original loop code) per participant for features and score at every rate here, so rate is limited by accuracy, not compute.\n")

A("## What this does and does not show\n")
A("- Exploratory only: no preregistration; the 0.02 margin is the one used in the project's earlier rate analyses and is applied here to the point estimate and, separately, to the lower CI limit (columns `within_0.02_of_30Hz_point` and `lower_CI_above_-0.02` in `rate_table.csv`). \"Lowest rate within 0.02\" is the lowest tested rate such that all tested rates from 30 Hz down to it are within the margin.")
A("- The low rates are simulated by anti-aliased resampling of high-rate recordings (Delmas 1000 Hz, Chen 100 Hz) from chin-rest research trackers; webcam noise is simulated, not recorded. A real 15 Hz webcam stream may differ in timing jitter, dropped frames and signal filtering; these are not simulated.")
A("- Differences between neighbouring rates are mostly inside the intervals (Delmas 75 participants, six group sets that include strong-countermeasure groups with AUC near 0.55); the 10 Hz losses of the 13-feature LR (-0.035) and of I-VT on clean data (-0.034) are the only ones whose point estimate is beyond the margin; their intervals exclude zero but still reach the margin edge (upper limits -0.019 and -0.018). The Delmas SVM curve is non-monotonic (20 Hz above 30 Hz), which is noise.")
A("- Under measured webcam noise the I-VT route is better at 20/15/10 Hz than at 30 Hz. A plausible reason (not tested here) is that per-sample jitter produces velocity of jitter/window, and the velocity window widens as the rate falls (67 ms at 30 Hz, 133 ms at 15 Hz), so fewer noise samples cross the 30 deg/s threshold. It is not a recommendation to lower the rate for I-VT; under noise the sample-dwell route is higher at every rate (on clean data I-VT is higher at 30 Hz and 20 Hz, sample dwell at 10 Hz).")
A("- Timing: the laptop's timings fluctuate between runs (a first run of the same script gave 78-106 ms vectorised totals across rates against 79-125 ms here); the resample column is an artefact of simulating a low-rate sensor from 1000 Hz data.\n")
A("## Files\n")
A("`rate_table.csv` (dataset, model, condition, rate, AUC, CI, delta vs 30 Hz with CI, within-margin flags, sample-minus-I-VT columns), `event_diag.csv` (+ `event_diag_agreement_long.csv`), `timing.csv` (+ `timing_per_participant.csv`), `results.json`, `repro_*.json`, the scripts above, and the per-stage checkpoints (`checkpoints_delmas/`, `checkpoints_chen/`, `checkpoints_webcam/`, `cache/`, `derived/`).")
(OUT / "card.md").write_text("\n".join(lines), encoding="utf-8")

res = {"status": "EXPLORATORY (not preregistered)", "reported_rates_hz": sorted(KEEP, reverse=True),
       "reproduction": {"delmas": rd, "chen": rc, "webcam": rw, "events": re_, "features_30hz": rf},
       "delmas": json.loads((OUT / "results_delmas.json").read_text()), "chen": json.loads((OUT / "results_chen.json").read_text()),
       "webcam": json.loads((OUT / "results_webcam.json").read_text()), "event_diag": ev.to_dict("records"), "timing": timing_json,
       "lowest_rate_within_0.02_of_30Hz": {"|".join(k): {"point_estimate": v[0], "lower_ci_limit": v[1]} for k, v in groups.items()}}
# 60 Hz entries that exist in the stage files are not reported
res["delmas"]["curve"] = {m: {r: v for r, v in cur.items() if r != "60"} for m, cur in res["delmas"]["curve"].items()}
res["chen"]["rates"] = {k: v for k, v in res["chen"]["rates"].items() if not k.endswith("@60")}
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print("written")
print("\n".join(lines[-22:]))
