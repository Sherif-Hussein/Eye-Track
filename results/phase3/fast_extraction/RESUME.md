# RESUME: Phase 3 fast_extraction

STATUS: COMPLETE (2026-11-02). Nothing is running, nothing is left.

Outputs: `card.md` (before/after, equivalence, AUC replay), `results.json` (all numbers; `speedup_claimed` per pipeline), `checkpoints/` (one JSON per stage, round files for the benchmarks, per-model AUC files), run logs `run_log_*.txt`, `cache/` (decoded Delmas samples, ~0.3 GB, and the fast tables used for the AUC replay; safe to delete, regenerate with the decode and equiv stages).
Code: `src/edd/delmas_fast.py`, `src/edd/celniak_fast.py`, `scripts/phase3_fast_extraction.py`.

Headline: Delmas and Celniak features equal to the originals (max abs diff <= 2.7e-12, NaN patterns identical, allclose rtol 1e-9); out-of-fold AUC identical (P4 SVM 1000 Hz 0.776201, P4 LR 30 Hz 0.773964, ALL_z LR 30 Hz 0.781130; Celniak SVM 0.597368, global_gaze_lr 0.764912). Speed-ups see card.md.

Re-run from scratch (repo root, each stage skips if its checkpoint exists):
    OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I scripts/phase3_fast_extraction.py --stages delmas_decode,delmas_equiv,delmas_auc,delmas_bench,celniak_equiv,celniak_auc,celniak_bench
    OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I scripts/phase3_fast_extraction.py --stages report
Benchmarks were run with nothing else of ours running; a very-low-priority phase1_celniak.py job and other processes loaded the CPU, so times are upper bounds.
