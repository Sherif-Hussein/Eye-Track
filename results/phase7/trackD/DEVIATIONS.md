# Track D: deviations from PREREG.md

Logged after the PREREG was written (2026-11-02, 21:24).

1. Harness check: the first version of the check script compared the mean of the per-fold AUCs printed by `run_grid` (0.7811 and 0.7666) with 0.7854 and 0.7723, which are pooled image-level AUCs on repeat-averaged scores (the non-inferiority definition). The check was corrected to compute the pooled definition from the new out-of-fold scores. The scores themselves were already identical to the stored Track C files (max difference 1.1e-12 for I-VT, 0 for sample-based), so the real-label D1/D2 runs were started only after the corrected check passed. No plan change.
2. `phase7_trackD_stats.py` first crashed (argument order of the tag helper) before writing any output; fixed and rerun. No result affected.
3. The harness check, D1 and D2 ran exactly as pre-registered (all 70 configurations, 2 worker processes). The run was interrupted by a machine sleep (clock jump 22:06 to 05:06) and resumed from checkpoints without loss. No other deviation.
