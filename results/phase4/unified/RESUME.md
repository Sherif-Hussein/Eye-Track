# RESUME: Phase 4 unified pipeline run

STATUS (2026-11-02): COMPLETE. All stages ran on real labels after the approval recorded in GO_AHEAD.txt (extract, run, efficiency, stats, chen_rate, webcam_adaptive, curves, figdata, plus the post-hoc descriptive stage `posthoc`). Nothing is running, nothing is left.

Outputs: `card.md` (all results), `results.json` (everything, incl. analyses A and B, efficiency, post-hoc), `results_chen_rate.json`, `results_webcam_adaptive.json`, `results_posthoc_per_repeat_auc.json`, `figdata/*.csv` (+ README.md), `oof/` (out-of-fold rows and person tables), `checkpoints/`, `PREREG.md` (with Amendments), `DEVIATIONS.md`, logs `log_*.txt`.

Re-run (repo root, Git Bash, n_jobs <= 4; all stages skip finished checkpoints):
```
export OMP_NUM_THREADS=1
for st in extract run efficiency stats chen_rate webcam_adaptive curves figdata posthoc; do .venv/Scripts/python.exe -I scripts/phase4_unified.py $st > results/phase4/unified/log_$st.txt 2>&1; done
```
Note: `results.json` was extended after `stats` with the keys analysis_A_*, analysis_B_*, posthoc_*, extract_equivalence, efficiency (a re-run of `stats` overwrites it; merge them again from the results_*.json files).
