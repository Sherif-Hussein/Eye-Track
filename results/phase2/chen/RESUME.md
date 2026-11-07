# Track `chen`: **COMPLETE** (2026-11-02). Nothing is running; nothing is left to resume.

All six stages are done and `comparison_card.md` is final (INTERIM removed). Headline: the paper's rate rule is reproduced (Exp 1 15/29 vs 16/29; Exp 2 8/27 vs 9/27), it has a 15.5% false-positive rate on unfamiliar faces in Exp 1 (5.6% in Exp 2); best candidate I2b_bins_fusion (chosen by the pre-fixed rule) detects 48.3% / 63.0% at 5.2% / 7.4% FPR, transfers across experiments, but its gain over the rate rule is not significant after Holm correction. The handover notes below are kept for the record; the resume commands at the bottom of the table are historical, do not rerun them (stages are idempotent and would only skip finished work, but `best` and `cross` would overwrite `results.json` keys).

## Handover notes (original)

The track brief is in the coordinator's original task. Paper: Chen, Mathôt & Akyürek (2026), J. Cognition 9(1):20. Follow `docs/phase2_protocol.md`, including its Checkpointing section.

## State at handover

Nothing of mine is running; all `phase2_chen` processes were stopped.

| Step | Status |
|---|---|
| Download | **Complete and verified.** 64 EDFs (exp1 33, exp2 31; 2.35 GB) match the OSF sizes in `data/chen/osf_manifest.json`. CSVs (behaviour logs) and the authors' analysis scripts are in `data/chen/exp{1,2}/{CSV,Analysis}/`. |
| Extraction | **Complete.** All 64 subjects are in `data/derived/chen_cache2/` (33 `exp1_*.npz`, 31 `exp2_*.npz`, no `.tmp` leftovers). Combined into `data/derived/chen_exp1_traces.npz` + `chen_exp1_meta.csv` (7536 trials) and `chen_exp2_*` (6720 trials). Every subject has 192 experimental trials. `data/derived/chen_cache/` is an obsolete older cache and can be deleted. |
| `folds.json` | **Frozen** (written 23:49 with all participants): 3 repeats x 10 participant-disjoint folds per experiment, seeds 2026–2028. Keep it; the script never overwrites it. |
| Reproduction (`repro` stage) | **Both experiments finished** (2026-11-02, re-run reproduced the Exp 1 handover numbers exactly). Saved in `results.json` under `reproduction_paper_protocol`. Exp 2: pupil size 5/27 (published 5), rate 8/27 (published 9), gaze 0 and 4 (published 0, 4). Exp 2 control-1-vs-2 FPR of the rate rule 5.6% (3/27 flagged, binomial p = .52): the Exp 1 image-identity excess did NOT replicate in Exp 2. Calibrated rate detection Exp 2: 29.6% at 5.6% FPR; person AUC 0.841 / EER 0.185 / HTER 0.352. |
| Screening | **Finished** (9 ideas + 4 ablations x 2 experiments; `ideas_table.csv`, `ideas/metrics_*.json`, `scores_*.csv.gz`, `results.json` keys `screening`, `paired_comparisons`). Note: exp2 I4b logged 14589 s wall time because the session was suspended during that run, not compute time. |
| Best candidate | **Chosen by the pre-fixed rule: `I2b_bins_fusion`** (mean person AUC 0.844 = (0.803 + 0.884)/2, highest among the 9 ideas after excluding I2c). The ablations A0-A3 are preprocessing variants of I3b, not ideas in `idea_specs`, and `best` cannot run them; A0_gapinterp (0.846), A3_base200 (0.840) are within 0.002-0.01 of it, i.e. no reliable difference. Decision taken before the `best` stage was run. |
| `best` + `cross` | **Finished** (perm=30) (`run_best.log`, `results.json` keys `best`, `cross_experiment`). I2b: Exp 1 14/29 detected (repeat-averaged scores), trial AUC 0.596, person AUC 0.811; Exp 2 17/27, 0.605, 0.881. Permutation (30 perms, min p = 1/31 = .032): p = .032 for AUC and detection in both. LOSO: Exp 1 15/29, Exp 2 17/27. Cross-experiment I2b: exp1->exp2 14/27, AUC 0.597, FPR 3.7%; exp2->exp1 14/29, AUC 0.606, FPR 5.2%. |
| Efficiency | **Finished** (`efficiency.json`, `run_eff.log`), I2b vs paper rule, timed on `exp1/Face_16.edf` from raw 1 kHz samples. |
| Paired CIs | `scripts/phase2_chen_delta.py` (best minus baseline, participant bootstrap) -> `delta_ci.json`, merged into `results.json` key `best_vs_baseline_paired`. |
| `best`, `cross`, efficiency, final card | Not started. |

### Exp 1 reproduction with the exact method (saved)

| | Published | Ours |
|---|---|---|
| Trials excluded | 211 | 214 |
| Pupil size detected | 10/29 | **10/29** |
| Rate detected | 16/29 | **15/29** |
| Gaze 0–1000 / 1000–2500 ms | 4 / 3 | **4 / 3** |

Control-1 vs control-2 false-positive rate of the rate rule:
- **15.5% per one-tailed test.** 9 of 29 people are flagged on an unfamiliar-vs-unfamiliar comparison; binomial p = .0016 vs 10%.
- **Random split of the same control trials: 5.0%.** The test is calibrated, so the excess comes from the specific face image (an item effect).

Detection at an empirically calibrated threshold (leave-one-person-out 95th percentile of the control-vs-control statistics):
- **Rate:** 34.5%, at 6.9% FPR. Person-level AUC 0.750, EER 0.293, HTER 0.362.
- **Pupil size:** detection 34.5% (10/29), FPR 5.2%, calibrated 34.5% at 6.9% FPR. Person AUC 0.728.

## Resume commands (project root, Git Bash; keep n_jobs ≤ 2, already set in the script)

```
# 1. reproduction + screening + paired comparisons (checkpointed per idea)
.venv/Scripts/python.exe -I scripts/phase2_chen.py repro screen compare > results/phase2/chen/run_screen.log 2>&1
# 2. choose the best idea (rule below), then CIs / LOSO / permutation / coefficients and cross-experiment transfer
.venv/Scripts/python.exe -I scripts/phase2_chen.py best best=<idea> perm=<n> cross > results/phase2/chen/run_best.log 2>&1
# 3. efficiency (writes efficiency.json)
.venv/Scripts/python.exe -I scripts/phase2_chen_efficiency.py <idea> > results/phase2/chen/run_eff.log 2>&1
# 4. finalise comparison_card.md (remove INTERIM)
```

**Best-candidate rule, fixed before any full-data result:** highest mean person-level AUC (`person_auc`, from the familiar test vs the control-1-vs-control-2 test) over both experiments. Exclude I2c, because its windows were picked on the full sample.

## Non-obvious decisions and pitfalls

1. **Blink reconstruction is a port.** `edd.chen.dm_blinkreconstruct` ports datamatrix 1.0.20 `blinkreconstruct(mode='advanced')`, the authors' exact call. The source was read as text from the wheel, nothing was installed, and the port is GPL-3.0-derived. Unlike simple interpolation, it leaves undetected gaps NaN. With an approximate interpolating method Exp 1 gave 9/29 and 13/29; the port gives 10/29 and 15/29. The approximate channel `p_rec` is kept as ablation `A0_gapinterp`.
2. **EDF reading.** `edd.chen.read_edf_fast` uses the edfapi DLL shipped inside `eyelinkio` and only converts samples inside the RSVP phase. The library costs ~30 µs per sample, so a full re-extraction takes ~40 min at 2 workers. It is cached; do not redo it.
3. **Epoching follows the authors' scripts.** T1onset = (T1pos − 1) × 30 samples from RSVP-phase start. Baseline = 5 samples (50 ms) before onset, subtractive. Trials are dropped for an undefined baseline or per-person baseline |z| > 2. Window 0–250 samples. Rate = Hanning-25 smooth of `np.diff`. Controls are pooled (T1 3+4). The peak is argmax |mean familiar − mean control| over pooled training trials. The test is a one-tailed Student t (p/2 if the mean difference is > 0, else p = 1).
4. **Each control is one fixed face image** per participant (checked in the CSV `controlImage1/2`), as is the familiar face. Control 1 vs control 2 is therefore an innocent surrogate with the same structure, and the false-positive rate must be read with this in mind.
5. **MiniRocket (I4a/b) is fit on the other experiment's participants.** Its kernels and bias quantiles are unsupervised but depend on the data, so fitting on independent people keeps them leak-free. Only the ridge head is fit per fold. This choice avoided a ~250 s per-fold refit.
6. **Within-person normalisation (I3a/b)** z-scores features against the person's own control trials. Trial types are known by design in a CIT; familiar trials are never used.
7. **Never run two `phase2_chen.py` processes at once.** Each reads `results.json` at start and rewrites it, so keys would clobber each other. Stages are idempotent; screening skips ideas that already have `ideas/metrics_*.json`.
8. **Runtime under shared CPU.** The `repro` stage took ~9 min for Exp 1 alone (the random-split FPR uses 500 splits x person x measure, plus a `p_rec` sensitivity rerun). Screening is 9 ideas + 4 ablations x 2 experiments x 30 folds; LR ideas are fast, MiniRocket needs a numba compile per worker. The `best` stage permutation costs about one full-idea run per permutation, so start with `perm=30` or fewer and raise it only if time allows.
9. **pandas 3:** `read_json` needs `io.StringIO`, which the scripts already use. Run scripts with `-I` from the project root; MSYS mangles leading-slash arguments in Git Bash, so set `MSYS_NO_PATHCONV=1` if you pass any.
10. **Efficiency script:** it times from raw 1 kHz EDF samples of `exp1/Face_16.edf` (that subject is held out when fitting). It imports `phase2_chen.py` via importlib and fits the chosen idea; only `kind == "lr"` ideas are supported.

## Files (final)

Extra scripts written by the finishing agent: `scripts/phase2_chen_tables.py` (renders `tables_screening.md`), `scripts/phase2_chen_delta.py` (paired bootstrap, `delta_ci.json`). `results.json` has final keys `reproduction_paper_protocol`, `screening`, `paired_comparisons`, `best`, `cross_experiment`, `best_vs_baseline_paired`, `efficiency`, `best_candidate_rule`, `status`.

Scripts:
- `scripts/phase2_chen_download.py`
- `scripts/phase2_chen_extract.py`
- `scripts/phase2_chen.py`
- `scripts/phase2_chen_efficiency.py`
- module `src/edd/chen.py`

Card status: sections 1-4 written (screening tables from `scripts/phase2_chen_tables.py` -> `tables_screening.md`); sections 5-7 pending the `best`, `cross` and efficiency stages. After `best`/`cross` finish: run efficiency (`phase2_chen_efficiency.py I2b_bins_fusion`), then write sections 5-7, remove INTERIM, mark this file COMPLETE.

Outputs so far:
- `folds.json`
- `results.json` (interim)
- `comparison_card.md` (INTERIM)
- logs `download_log.txt`, `extract_log*.txt`, `run_screen.log`
