# RESUME: track mazza -- COMPLETE (2026-11-02 ~05:10)

Status: ALL DONE. No process is running. Nothing left to run. The final `comparison_card.md` (no INTERIM) and `results.json` are written.

## Files
- `comparison_card.md` (final, built by `scripts/phase2_mazza_card.py` from `card_sections_1_2.md` + `tables.md` + `results.json`)
- `results.json` (all numbers incl. `fourclass_paired`), `extras.json`, `extras_cache.json`, `screen_summary.json`, `selection.json`, `reproduction.json`, `fourclass_paired.json`, `tables.md`, `folds.json`, `oof/`, logs.

## Headline results
- Permutation tests (repeat-0 folds, full nested pipeline, 30-31 perms, so coarse): guilty_vs_innocent i2_z_lr obs AUC 0.680, p = 0.031 (31); deceptive_vs_honest i2_z_lr obs 0.654, null p95 0.651, p = 0.0645 (30); g2_vs_g3 i1_raw90_lgbm obs 0.784, null mean 0.477, p = 0.032 (30). None accounts for choosing the config among 18.
- Baseline vs best (mean AUC over 3 repeats): guilty 0.602 vs 0.688 (delta +0.072 [-0.025, +0.169], Holm 1.0); deceptive 0.640 vs 0.655; g2_vs_g3 0.679 vs 0.750 hindsight (0.673 selected fairly); g2_vs_g1 baseline best 0.630; g2_vs_g4 baseline best 0.717.
- 4-class: baseline macro AUC 0.590 / acc 0.335; best raw90 LR 0.639 / 0.418 (delta acc +0.083 [0.000, +0.166], Holm 0.434). Published 0.63 / 40.13% not reproduced by the published LR (ours 0.602 / 35.4%).
- Deceptive vs honest neutral-only control 0.659 (not crime-specific); G2 vs G3 neutral-only 0.664.
- Efficiency: baseline 2.65 / 5.39 ms p50/p95 per participant, 19.7k participants/s, 15.6 KB peak, 8.9 KB model, 37 params; i2_z_lr 2.70 / 3.25 ms, 8.2 KB; lgbm 3.79 / 4.59 ms, 91.6 KB, 822 nodes.

## If anything must be regenerated (all steps skip finished work)
    export OMP_NUM_THREADS=1
    .venv/Scripts/python.exe -I -u scripts/phase2_mazza.py data/mazza/data.csv results/phase2/mazza extras --perm 200 >> results/phase2/mazza/extras_log.txt 2>&1   # rewrites extras.json/results.json from extras_cache.json (then re-add fourclass_paired to results.json by rerunning the next line and the python snippet in card step below)
    .venv/Scripts/python.exe -I scripts/phase2_mazza_4class_paired.py results/phase2/mazza
    .venv/Scripts/python.exe -I scripts/phase2_mazza_tables.py results/phase2/mazza
    .venv/Scripts/python.exe -I scripts/phase2_mazza_card.py results/phase2/mazza
(results.json gets key `fourclass_paired` copied from fourclass_paired.json; `extras` stage rewrites results.json without it, so add it again if rerun.)
Notes: extras_log.txt contains an old traceback (TerminatedWorkerError from a PC restart) and a 15165 s multiclass wall time (session pause); both harmless.
