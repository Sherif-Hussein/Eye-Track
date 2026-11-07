"""Assemble the FINAL comparison_card.md for track mazza from card_sections_1_2.md, tables.md and results.json.
Usage: python -I scripts/phase2_mazza_card.py results/phase2/mazza   (run phase2_mazza_tables.py first)"""
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
R = json.loads((out / "results.json").read_text())
S12 = (out / "card_sections_1_2.md").read_text(encoding="utf-8")
T = (out / "tables.md").read_text(encoding="utf-8")
M = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "eer", "hter"]


def sect(title_start):
    """Return the tables.md block starting at '#### <title_start>' up to the next '####'."""
    m = re.search(r"(#### " + re.escape(title_start) + r".*?)(?=\n#### |\Z)", T, re.S)
    return m.group(1).strip() if m else ""


rows = []
for t, d in R["screen"].items():
    b = d["base_pub36_lr_unreg"]["repeats"]
    rows.append(f"| {t} | " + " | ".join(f"{b[k]['mean']:.3f} ± {b[k]['sd']:.3f}" for k in M) + " |")
base_tab = "| task | " + " | ".join(M) + " |\n|" + "---|" * (len(M) + 1) + "\n" + "\n".join(rows)

screen_tabs = "\n\n".join(sect(x) for x in ["guilty_vs_innocent", "deceptive_vs_honest", "g2_vs_g3", "g2_vs_g1", "g2_vs_g4"])
fp = R["fourclass_paired"]
fp_rows = "\n".join(
    f"| {k} | {v['acc_avg_scores']:.3f} | {v['delta_acc']:+.3f} [{v['delta_acc_ci95'][0]:+.3f}, {v['delta_acc_ci95'][1]:+.3f}] | "
    f"{v['delta_macro_auc']:+.3f} [{v['delta_macro_auc_ci95'][0]:+.3f}, {v['delta_macro_auc_ci95'][1]:+.3f}] | {v['p_holm_acc']:.3f} |"
    for k, v in fp.items() if not k.startswith("_"))
four_tab = sect("4-class (G1/G2/G3/G4)").split("\n", 1)[1].strip()

HEAD = """# Comparison card: RealEye webcam crime-scene dataset (track `mazza`)

**FINAL** (2026-11-02). Screening, all extras, 4-class, efficiency and permutation tests are complete. Tables are regenerated from `results.json`, `tables.md` and `fourclass_paired.json`.

## Summary for the authors (read this first)

1. **No reliable gain over the published method.** Under participant-disjoint nested CV, no idea beats the published logistic regression on any task. Every Holm-adjusted p is 1.000 on the three main tasks; on G2 vs G1 and G2 vs G4 every idea scores below the baseline. On the 4-class task, the best of 7 ideas (raw 90 features, LR) has accuracy +0.083 [0.000, +0.166] over the baseline, Holm p = 0.434.
2. **Selection after the fact is optimistic.** On G2 vs G3, the best of 18 configurations in hindsight (LightGBM, 90 features) scores AUC 0.750 (0.759 on repeat-averaged scores). When the configuration is chosen only on the other nine folds, the same procedure scores **0.673 ± 0.079** (95% CI [0.574, 0.808]), against an honest baseline of 0.679 and a published single run of 0.72. Do not quote 0.750 as an improvement. Its permutation p (0.032, 30 permutations) is for that one hindsight-picked configuration and does not account for choosing it from 18.
3. **Within-person z-scores for guilty vs innocent are promising but not proven.** Z-scoring each person's target-image fixation values against the same person's neutral images gives AUC 0.688 ± 0.009 versus 0.602 ± 0.014 for the baseline (0.673 vs 0.609 on two held-back fold seeds; permutation p = 0.031 with 31 permutations, the minimum possible; neutral-only control at chance, 0.484). The paired participant-bootstrap ΔAUC is **+0.072, 95% CI [-0.025, +0.169]** and Holm p = 1.000, so zero is inside the interval. The configuration was also the best of 18 on the same data. It deserves a confirmatory run; it is not a result.
4. **The deceptive vs honest signal is not specific to the crime scene.** An L2 logistic regression using only the six neutral room images (shown before the crime-related images) reaches AUC 0.659 [0.557, 0.765], as high as any model using the target images (best 0.655, z-scores). The negative control fails, so this task should not be presented as deception detection. The same neutral-only model reaches 0.664 [0.537, 0.782] on G2 vs G3, close to the full 36-feature baseline (0.679), so group differences in generic viewing behaviour (possibly the age, sex, education and glasses imbalances in the article's Table 1, or session effects) cannot be excluded for the published task either.
5. **The published numbers reproduce except the 4-class result.** G2 vs G3: ours 0.674 ± 0.033 over 10 seeds against 0.72 (inside our range 0.613 to 0.726). The 4-class 0.63 / 40.13% is not reproduced (ours 0.602 ± 0.015 and 35.4% ± 2.9, with 0.63 above every one of our 10 seeds).
6. Efficiency is trivial for all models (about 2.6 to 3.8 ms per participant, under 100 KB), so it is not a differentiator.

"""

S3 = f"""
## 3. Honest baseline under our protocol

The published model (L2 logistic regression with C = 1e8 on the 36 target-AOI count and duration predictors; median imputation, in-fold 1st/99th percentile clipping and standardisation fitted on training participants only), evaluated with participant-disjoint stratified 10-fold CV, 3 repeats with different fold seeds (frozen in `folds.json`; two further seeds held back), one aggregate row per participant so the decision unit is the participant. Mean ± SD over the 3 repeats; threshold 0 on the signed score.

{base_tab}

Notes. G2 vs G3: AUC 0.679 ± 0.010 on our folds, versus 0.72 published and 0.674 ± 0.033 in the section 2 reproduction (a different CV scheme: pooled predictions, 10 seeds). On the held-back fold seeds the same baseline gets 0.717 ± 0.017, so fold-seed luck alone moves G2 vs G3 by about 0.04. Repeat-averaged scores (used for bootstrap CIs) give 0.690 [0.567, 0.802]; do not mix them with the per-repeat means. The 4-class baseline is in section 4.
"""

S4 = f"""
## 4. Ideas screened

All screening used identical folds (repeats 0 to 2): 18 configurations on guilty vs innocent, deceptive vs honest and G2 vs G3; 7 on G2 vs G1 and G2 vs G4; 8 on the 4-class task. Ideas: (1) other models on raw features (LR, SVM, random forest, LightGBM; 36 published vs 90 raw features), (2) within-person differential features against the neutral images (z-score, share difference, log ratio), (3) crime-area (desk AOI) features, (4) feature selection (ANOVA, L1, stability selection). Every result is kept, including failures. ΔAUC is the paired participant-bootstrap difference against the baseline on repeat-averaged scores; p is Holm-adjusted over the ideas tried on that task. The "(best)" mark is the hindsight best by mean AUC and is optimistic (see the selection-adjusted table below and summary point 2).

{screen_tabs}

#### 4-class (G1/G2/G3/G4): published AUC 0.63, accuracy 40.13%

{four_tab}

Paired participant bootstrap of the 4-class configurations against the baseline (repeat-averaged probabilities, 1000 stratified resamples, Holm over 7 ideas on accuracy; `fourclass_paired.json`, `scripts/phase2_mazza_4class_paired.py`). The baseline has accuracy {fp['_baseline']['acc_avg_scores']:.3f} and macro AUC {fp['_baseline']['macro_auc_avg_scores']:.3f} on repeat-averaged probabilities.

| config | accuracy (avg scores) | Δ accuracy [95% CI] | Δ macro AUC [95% CI] | Holm p (accuracy) |
|---|---|---|---|---|
{fp_rows}

The raw-90-feature LR (accuracy 0.418 ± 0.006 per repeat, macro AUC 0.639) is the nominal best and its accuracy interval just touches 0, but after Holm p = 0.434. The published 40.13% is reachable by some of our configurations, but none is reliably better than the baseline, and the published LR itself did not reproduce (section 2).

{sect("Selection-adjusted best")}

In each outer fold the configuration is chosen on the other nine folds (score divided by within-repeat SD), so the reported AUC is what a person without hindsight would get. This is the honest number to compare with the baseline and the published values: guilty vs innocent 0.654 (baseline 0.602), deceptive vs honest 0.574 (0.640), **G2 vs G3 0.673 (0.679; published 0.72)**, G2 vs G1 0.575 (0.630), G2 vs G4 0.642 (0.717). The selected pipeline is not better than the fixed baseline on any task except guilty vs innocent, where it is +0.052 and inside the noise.
"""

S5 = f"""
## 5. Best candidates vs published vs honest baseline

Two candidates: **A, i2_z_lr** (guilty vs innocent; also best on deceptive vs honest) because it is the only consistent gain, and **B, i1_raw90_lgbm** (G2 vs G3), the hindsight best on the published headline pair.

{sect("Best vs baseline with participant-bootstrap 95% CI")}

{sect("Permutation tests")}

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

{sect("Held-back fold seeds")}

These control fold-assignment luck only. The participants are the same, so they do not remove selection bias: B rises from 0.750 to 0.770 and the baseline from 0.679 to 0.717 (gap 0.071 shrinks to 0.053). For deceptive vs honest the baseline (0.660) is slightly above i2_z_lr (0.649).

{sect("Block ablation")}

{sect("Outlier clipping ablation")}

{sect("Calibration and selective prediction")}

Raw logistic-regression probabilities of the baseline are badly overconfident (ECE 0.23 to 0.44; unregularised C = 1e8 fit); Platt scaling fitted inside the training folds brings ECE to 0.04 to 0.08. Selective prediction (keep the most confident participants) raises accuracy, e.g. A on guilty vs innocent 0.641 at full coverage, 0.669 at 80% and 0.702 at 60%; the baseline goes 0.569, 0.582, 0.599. This is risk-coverage evaluation, not a deployed threshold.

**Caveats.** n = 157 (about 40 per group; 78 guilty vs 79 innocent); single-AUC bootstrap CIs are about ±0.09; the 3 fold repeats do not reduce participant sampling variance; one dataset and no external validation; features are RealEye's own aggregates; group differences in age, sex, education and glasses are confounded and cannot be adjusted for with the deposited CSV; the image order is fixed for all participants, so order effects cannot be separated from group effects; the permutation tests are small and do not include selection.
"""

e = R["efficiency"]


def effrow(label, v):
    lat, fe = v["latency_ms_per_participant_end_to_end"], v["latency_ms_feature_extraction_only"]
    return (f"| {label} | {v['task']} | {v['n_model_inputs']} | {lat['p50']:.2f} / {lat['p95']:.2f} | {fe['p50']:.2f} | "
            f"{v['throughput_participants_per_s_batch157']:.0f} | {v['peak_memory_bytes_inference_one'] / 1024:.1f} | "
            f"{v['model_size_bytes'] / 1024:.1f} | {v['stored_parameters_classifier']} | "
            f"{v['stored_parameters_preprocessing_approx']} | {v['fit_seconds_full_nested_search']:.1f} |")


eff_rows = "\n".join([effrow("baseline: published LR re-implementation", e["baseline"]),
                      effrow("A: i2_z_lr", e["best_guilty_vs_innocent"]),
                      effrow("A: i2_z_lr", e["best_deceptive_vs_honest"]),
                      effrow("B: i1_raw90_lgbm", e["best_g2_vs_g3"]),
                      effrow("baseline", e["best_g2_vs_g1"]), effrow("baseline", e["best_g2_vs_g4"])])
S6 = f"""
## 6. Efficiency

Single-threaded on this laptop (`edd.efficiency`): {e['hardware']}. Pipelines are the final nested-search winners refit on all participants of the task. Latency is per participant (the decision unit), end-to-end through the scikit-learn pipeline (imputer, clipping, scaling, classifier) for one row, 500 runs. Throughput is a batch of all 157 participants. Peak memory is the extra allocation during one inference (tracemalloc). Model size is the pickled pipeline. "Preprocessing params" is the approximate count of stored imputer, scaler and clip bounds. The CPU was shared with other experiments during measurement, so p95 values are noisy.

| model | task | inputs | latency p50 / p95 ms | feature step p50 ms | throughput participants/s | peak mem KB | model size KB | classifier params | preprocessing params | nested fit s |
|---|---|---|---|---|---|---|---|---|---|---|
{eff_rows}

Not applicable: the deposited CSV holds RealEye's per-image aggregates, not the raw gaze stream, so per-sample FPS, samples per second and the real-time factor against the 60 Hz sensor cannot be computed, and feature extraction from raw gaze is not measured. Latency is dominated by pipeline overhead (a 36-weight dot product alone takes microseconds). All candidates are far below any real-time constraint. The z-score candidate costs nothing extra at inference (same 37 classifier parameters as the baseline; its within-person transform is part of the feature step) with a model size of 8.2 KB vs 8.9 KB. The LightGBM candidate is about 10 times larger (91.6 KB, 822 stored tree nodes) and about 1 ms slower per participant.
"""

S7 = """
## 7. Recommendation

- **Do not claim an improvement over Colasanti et al.** under a leakage-free protocol. Report the honest baseline (G2 vs G3 AUC 0.679 ± 0.010; selected pipeline 0.673) next to their 0.72 and note that 0.72 lies within our seed spread.
- **A full run is worth it only for the within-person z-score for guilty vs innocent**, as a confirmatory test fixed in advance: pre-specify i2_z_lr, more fold repeats, many more permutations (at least 1000, with a faster implementation) and, if possible, an independent sample or the other datasets in the benchmark. Current evidence: AUC 0.688 vs 0.602, held-back seeds 0.673 vs 0.609, permutation p 0.031 (the floor), but paired ΔAUC CI [-0.025, +0.169].
- **Do not present deceptive vs honest (or G2 vs G3) as crime-specific detection** until the neutral-image-only signal is explained. The strongest next check is whether it is a group or session confound (order, age, sex, education, glasses), which needs metadata missing from the deposited CSV that the article authors could supply.
- **Use this dataset as a low-signal external benchmark**, not for a state-of-the-art claim: webcam data, 157 people, accuracy about 0.60 to 0.70.
- Out of scope and not tried: raw gaze sequence models (data unavailable) and an exact WEKA reproduction.

## Files

`results.json` (all numbers), `extras.json`, `screen_summary.json`, `selection.json`, `reproduction.json`, `fourclass_paired.json`, `tables.md` (full tables), `folds.json`, `oof/` (out-of-fold scores of every configuration), `screen_log.txt`, `extras_log.txt`, `RESUME.md`. Scripts: `scripts/phase2_mazza.py`, `phase2_mazza_configs.py`, `phase2_mazza_repro.py`, `phase2_mazza_select.py`, `phase2_mazza_tables.py`, `phase2_mazza_4class_paired.py`, `phase2_mazza_card.py`, `src/edd/mazza.py`. Note: the `multiclass` timing in `extras_log.txt` (15165 s) is wall-clock time that far exceeds the compute and probably includes time the session was paused. MU3D is not used in this track.
"""

body12 = S12.split("\n", 1)[1].lstrip("\n")
(out / "comparison_card.md").write_text(HEAD + body12.rstrip() + "\n" + S3 + S4 + S5 + S6 + S7, encoding="utf-8")
print("card written")
