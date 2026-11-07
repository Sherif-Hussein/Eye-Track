# chen_confirm RESUME (2026-11-02 07:05)
Status: COMPLETE. run stage (3 repeats) and analyse stage finished. Nothing is running; nothing left to resume.
Outputs: PREREG.md (frozen), FREEZE.md, folds.json, checkpoints/repeat{0,1,2}.pkl, results.json, card.md, run.log, analyse.log, DEVIATIONS.md.
Headline: primary delta (I2b minus rate rule, detection at FPR<=7%) = +0.167, 95% CI +0.030 to +0.292, one-sided bootstrap p = 0.0105 -> confirmed per PREREG.
Re-run (only if needed): `.venv/Scripts/python.exe -I scripts/phase3_chen_confirm.py run` / `analyse` (run skips finished repeats).
