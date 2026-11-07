# delmas_raw: STATUS COMPLETE (2026-11-02). Historical resume notes below.

All background jobs for this track are stopped and no python process is writing here. The analysis was stopped cleanly between configurations: the last saved configuration is `P4_svm_120`, and `P4_svm_60` was running when it was killed (it is not saved).

## Data (done)
- `data/delmas/raw/Data_raw.zip` is complete: **940,857,791 bytes**, zip CRC test passed. Extracted to `data/delmas/raw/data_raw/` (83 Sj_XX folders, of which 75 are analysed).
- Format: main-session samples exist **only as EDF** (`I<id>c2_<half>.edf`; Exp. 2 has `_1` = first 32 trials and `_2` = last 32). The c2 `.asc` files are events only. Per-trial face layouts (`idcible`, `distractors`, `*Position`) are in the Psychtoolbox files `FI<id>c2_<half>.mat`. `FI<id>c3_1.dat` is the post-test (image, famous yes=1 / no=2). `dataMiSes_*.mat/.csv` is only the authors' saccade table.
- Extraction is done: 75 of 75 participants, 64 trials each, 6 rates (1000/500/250/120/60/30 Hz). Outputs are `data/derived/delmas_raw_face_trials.csv.gz` (one row per participant x rate x trial x quadrant) and `delmas_raw_trial_summary.csv.gz` (identity probe), with a per-participant cache in `data/derived/delmas_raw_cache/p<id>.npz`. It took about 45 s per participant.

## Non-obvious decisions and pitfalls
- **EDF reader.** pyedfread has no wheel for Python 3.14 on Windows. We installed `eyelinkio==0.3.0` with `pip install --no-deps` (its only runtime dependency, numpy, was already present). It bundles SR Research `edfapi64.dll`. eyelinkio's own `read_edf` takes 40 to 80 s per file, so `edd.delmas_raw.read_edf_fast` calls edfapi directly through ctypes. It only fetches samples between `EVENT_FixationDot2` and onset + 5000 ms, because `edf_get_float_data` costs about 20 us per call. It was verified bit-identical to eyelinkio. Missing gaze is -32768 or 1e8, and is set to NaN.
- **Timing and AOIs.** Display onset is the message `EVENT_ParallelDisplay`. AOIs are the four 300 x 400 px picture rectangles; 34.99 px/deg.
- **I-VT.** Fixed parameters at every rate: 30 deg/s, a fixed-time differentiator (+-4 ms span, at least 1 sample) and a minimum fixation of 60 ms. Fixation counts correlate r = 0.97 with the EyeLink parser at 120 Hz and above, 0.88 at 60 Hz and 0.77 at 30 Hz. Degradation uses `scipy.signal.resample_poly` (anti-alias FIR) after interpolating gaps; an output sample is marked missing if any input in its window was missing.
- **The published counts use a label-dependent rule.** Famous faces are summed over all presentations; unknown faces only over presentations in trials without a famous face (r = 0.93 and 0.97; exact match for 80% and 91% of faces). The number of presentations counted alone gives AUC 0.943 (64-trial table). Our "P4" features are per-presentation means over all presentations, which is label-free and deployable.
- **The published dwell columns could not be reproduced** by any rule tried (own, all presentations; non-famous trials only; famous trials only; pooled over co-presented unknown faces). Every rule gives r about 0.3. Within-class SD is 0.018 published vs 0.05 to 0.06 rebuilt, and published dwell_2 alone gives AUC 0.998 vs 0.818 rebuilt. This is an open question for the authors; do not claim a cause.
- **Labels and inclusion.** The (participant, image) rows and the `face` label are taken from the authors' BookFIP tables; we did not re-derive their misidentification exclusions. The 32-trial set is execution-order trials 32 to 63.
- **Stats needs OOF scores.** Paired Holm tests, CIs and best-candidate selection need per-configuration OOF scores. These are now saved per configuration (`checkpoints/<tag>_oof.csv.gz`) and reloaded by `run_grid`, so the stats stage resumes from disk. Exception: the three grids seeded from run 1 (`P4_pubrule_svm_1000`, `counts_pubrule_svm_1000`, `counts_all_svm_1000`) have metrics and per-participant AUCs but no OOF file. They are not candidates in the stats stage, so this does not matter. Delete their JSON if CIs are wanted for them.
- **Compute budget.** The SVM grid is C {0.1, 1, 10} x gamma {0.01, 0.1, 1}, the same for every SVM run. The features stage runs SVM only for P4_z and ALL_z (LR for the rest). Each SVM configuration (6 group-sets x 15 outer fits) takes about 3 min with n_jobs = 2. The permutation test uses LR for both the real and shuffled labels (30 permutations).

## Done (image-level AUC / EER / HTER per set, mean over 3 repeats; saved in `checkpoints/`)
| Config | G1-64 | G2-64 | G3-64 | G3-32 | G4-32 | G5-32 | mean AUC | CIT AUC |
|---|---|---|---|---|---|---|---|---|
| Published table + SVM (honest baseline, `book_svm`) | 0.999/0.006/0.009 | 0.994/0.023/0.023 | 0.999/0.002/0.003 | 0.978/0.056/0.059 | 1.000/0.010/0.008 | 1.000/0.000/0.016 | 0.995 | 1.000 |
| Own detector, published label rule (`P4_pubrule_svm_1000`) | 0.930 | 0.921 | 0.876 | 0.774 | 0.585 | 0.599 | 0.781 | 0.895 |
| Counts only, published rule | 0.666 | 0.739 | 0.601 | 0.590 | 0.591 | 0.578 | 0.627 | 0.803 |
| Counts only, all presentations | 0.850 | 0.820 | 0.796 | 0.791 | 0.592 | 0.620 | 0.745 | 0.876 |
| **P4 label-free SVM 1000 Hz** | 0.916/0.168/0.162 | 0.888/0.199/0.200 | 0.838/0.179/0.194 | 0.825/0.246/0.246 | 0.572/0.459/0.437 | 0.618/0.419/0.419 | 0.776 | 0.882 |
| P4 SVM 500 Hz | 0.918 | 0.886 | 0.836 | 0.823 | 0.597 | 0.617 | 0.780 | 0.876 |
| P4 SVM 250 Hz | 0.918 | 0.890 | 0.837 | 0.823 | 0.579 | 0.621 | 0.778 | 0.883 |
| P4 SVM 120 Hz | 0.917 | 0.880 | 0.837 | 0.827 | 0.557 | 0.619 | 0.773 | 0.882 |

So far the rate curve is flat from 1000 to 120 Hz (mean AUC 0.773 to 0.780). The large drop comes from the label-free aggregation and the strong countermeasures (G4, G5), not from the sampling rate.

## Left (in script order; a rerun skips everything in `checkpoints/`)
1. rate: `P4_svm_60` (was running), `P4_svm_30`, then `ALLz_lr_{1000..30}` (LR, fast).
2. features: LR on P4+bins, NEW, pupil_only, firstlook_only, P4+NEW, ALL, P4_z, ALL_z; SVM on P4_z and ALL_z; P4_lr.
3. transfer: LR on P4 / P4_z / ALL_z at 1000 and 60 Hz; SVM on the published table.
4. models: LightGBM on P4 and ALL_z.
5. stats: paired Wilcoxon + Holm vs P4_svm_1000 and vs book_svm; cluster-bootstrap CIs; permutation test for the best candidate.
6. identity probe, then efficiency (loads Sj_04's EDF, about 30 s, then timing; the end-to-end figures exclude EDF decoding).
Estimated remaining time: about 30 to 40 min with n_jobs = 2.

## Resume commands (from the project root; run in the background, logging to the track folder)
    OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I scripts/phase2_delmas_raw.py data/derived data/delmas/BookFIP_64_trials.xlsx data/delmas/BookFIP_32_trials.xlsx results/phase2/delmas_raw data/delmas/raw/data_raw > results/phase2/delmas_raw/run_log.txt 2>&1
    .venv/Scripts/python.exe -I scripts/phase2_delmas_raw_tables.py results/phase2/delmas_raw/results.json results/phase2/delmas_raw/tables.md
Re-extraction is only needed if `data/derived/delmas_raw_face_trials.csv.gz` is missing:
    .venv/Scripts/python.exe -I scripts/phase2_delmas_raw_extract.py data/delmas/raw/data_raw data/delmas/BookFIP_64_trials.xlsx data/derived --n_jobs 2

## Files
- Scripts: `scripts/phase2_delmas_raw_extract.py`, `scripts/phase2_delmas_raw.py`, `scripts/phase2_delmas_raw_tables.py`; library `src/edd/delmas_raw.py`.
- `folds.json` (frozen, 5 folds x 3 repeats per group-set), `checkpoints/`, `progress.md` (one line per finished configuration), `run_log.txt`, `results.json` (rewritten after each stage; currently holds audit + baseline), `checkpoints/results_run1_partial.json` (first run).
- After the run, finish `comparison_card.md` sections 4 to 7 from `results.json` and `tables.md`, and remove the INTERIM mark.


## COMPLETE (2026-11-02)
All stages finished; nothing is running and nothing is left to resume.
- Main run (`scripts/phase2_delmas_raw.py`) finished: rate curve incl. 60 and 30 Hz, features, transfer, models, stats, identity probe and efficiency are all saved in `checkpoints/` and `results.json` ("done in 17760 s"; log `run_log.txt`; the first-session log is `run_log_part1.txt`).
- Non-inferiority (margin 0.02, Holm) with `scripts/phase2_delmas_raw_noninf.py`: `noninferiority.json`, `noninf_log.txt`; also stored in `results.json` under `noninferiority_margin_0.02_holm`.
- `tables.md` regenerated with `scripts/phase2_delmas_raw_tables.py`.
- `comparison_card.md` is final (INTERIM removed). The earlier interim card is kept as `checkpoints/comparison_card_interim_backup.md`.
- Headline numbers (mean image-level AUC over the six sets): published table + SVM 0.995; label-free P4 SVM 1000/500/250/120/60/30 Hz = 0.776/0.780/0.778/0.773/0.770/0.769; ALL_z LR = 0.789/0.791/0.789/0.789/0.790/0.781; best candidate by the stats rule P4 LR 0.779 (no reliable winner after Holm); LightGBM on P4 worse (Holm p = 0.018); new pupil/first-look/microsaccade features alone 0.51-0.53; permutation p = 0.032 (30 perms, the floor).
- To rerun anything: the main script skips everything in `checkpoints/` (delete a config's json to recompute it).
