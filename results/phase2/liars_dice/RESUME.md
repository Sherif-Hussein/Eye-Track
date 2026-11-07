# RESUME — liars_dice track: COMPLETE (2026-11-02)

Nothing is running and nothing is left. `comparison_card.md` is the FINAL version (no INTERIM or PENDING), covering protocol sections 1–7.

## Done (on disk)
- Data in `data/liars_dice/osf_wbj83/` (hashes verified); manuscript in `data/liars_dice/preprint_q5ngh/`.
- Label audit (`data_audit.json`): released BidHonesty is inverted, so deceptive = released "Truth". Bid table `data/derived/liars_dice_bids.csv`; reproduction in `reproduction.json`; frozen folds in `folds.json`.
- All 39 CV arms x 3 repeats cached in `data/derived/liars_dice_oofcache/` (18 pvc, 21 pvp; spot-checked against `results.json`).
- `results.json` (complete): both experiments, 3 repeats, Holm comparisons, participant-cluster CIs, permutation test (50 within-participant shuffles, p = .0196 = the minimum for 50), efficiency, and 10 transfer cells.
- `followup.json` (ablation, importance, bid-time proxy, between-person level, Holm across "eyes add" tests, per-participant CIs): folded into card section 5.
- `comparison_card.md` rendered with `scripts/phase2_liars_dice_card.py --final` from the JSON files and `card_narrative.md`.

## Headline numbers
- PvC best eye-only `I3_lgbm_ocular_online_raw` AUC .617 [.578, .668], EER .422, HTER .419 vs honest baseline `I1_paper_ocular` .522 [.490, .554]: dAUC +.096 [+.035, +.163], Holm p < .001.
- PvP best eye-only `I3_lr_ocular_online` AUC .602 [.568, .635], EER .419, HTER .420 vs .550 [.517, .582]: dAUC +.052 [+.018, +.086], Holm p = .027.
- Per-participant Wilcoxon, Holm: p = .106 (PvC), .180 (PvP), so not reliable person by person.
- Context AUC .753 / .777; context + RT .789 / .804; fusion .793 / .809; eyes added to context + RT: +.005 / +.003 (not reliable).
- Transfer AUC .519–.598 (near chance to weak).

## Open issues for the user (not computational)
- Data licence unclear (private OSF node, no licence): use in the paper only with the authors' permission.
- Tell the authors about the label inversion.

## To regenerate the card (project root, Git Bash)
```
.venv/Scripts/python.exe -I scripts/phase2_liars_dice_card.py results/phase2/liars_dice --final
```
Full pipeline commands, if ever needed: `scripts/phase2_liars_dice.py` (`--raw-dir data/liars_dice/osf_wbj83/Data --n-perm 50`) and `scripts/phase2_liars_dice_followup.py` (arguments in the script docstring). The screening script skips cached arms.
