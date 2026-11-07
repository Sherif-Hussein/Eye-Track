# webcam_noise: COMPLETE (2026-11-02)

All 47 configurations, the stats stage and `card.md` are finished. Nothing is running and nothing is left to resume.

- Extraction (75 of 75 participants), clean check, level-0 OOF check: done and passed.
- Runs: 2 (level 0) + 5 levels x 3 draws x 3 variants = 47 checkpoints in `checkpoints/`.
- Stats: `results.json`, `curve_table.md/csv`, `metrics_table.md`, `stats_log.txt`.
- Headline: ALL_z LR 30 Hz AUC 0.7854 (clean) -> 0.7527 / 0.7418 / 0.7267 / 0.7005 / 0.6416 at 1 / 1.5 / 2 / 3 / 4 deg offset. Largest non-inferior level vs P4 SVM 1000 Hz = 0 deg (clean only). See `card.md`.
- To rerun a configuration, delete `checkpoints/<tag>.json` and `<tag>_oof.csv.gz`, then:
  `OMP_NUM_THREADS=1 .venv/Scripts/python.exe -W ignore -I scripts/phase3_webcam_noise.py --stages run,stats`
