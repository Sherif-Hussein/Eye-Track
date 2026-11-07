# Deviations from PREREG.md
1. Before the real-label run started (after PREREG.md and folds.json were frozen), the analysis script was extended to also output the ratio delta(I2b-noMiss)/delta(I2b), which PREREG section 6 already says is reported. No design element changed.
2. A first smoke test on label-shuffled data (scratchpad, before the freeze of folds.json) failed with a "0 samples" error in the calibration LogisticRegression; a rerun with the final code on the frozen folds did not reproduce it. The cause was not identified. The real run completed all 30 folds x 3 repeats without errors.
3. Wording note: PREREG section 8 allows sensitivity arm I2b_perexp to be called "consistent" if it has the same sign; it has delta +0.095 (CI -0.036, +0.220) on detection, so the sign agrees but the CI includes 0.
No other deviations.
