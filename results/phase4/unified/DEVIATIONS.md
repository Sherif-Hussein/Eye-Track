# Deviations and additions (Phase 4 unified run)

No test, margin, feature list, grid, fold set or threshold in PREREG.md sections 1-12 was changed after the real run started.

1. **PREREG amendments (before any real-label outcome):** `nh` secondary threshold rule dropped by ruling of the senior analyst; the 100 Hz-as-published-rate sentence added to analysis A. See "Amendments" in PREREG.md.
2. **Code fixes made during label-shuffled smoke tests, before the real run** (not during it): `Series.repeat` attribute clash in the cascade and Chen-primary code, Delmas `set` column parsed as int when re-read from CSV, stage-1 cascade timed with only its two channels, deployed-model size excludes feature names. None involved real labels.
3. **Post-hoc descriptive addition after seeing the real results (stage `posthoc`, `results_posthoc_per_repeat_auc.json`):** the pre-registered person-level tests average raw logits over repeats. Tuned C, and therefore logit scale, differs between folds, which lowers the repeat-averaged AUC of the unified model relative to its per-repeat mean (Mazza G2 vs G3: 0.601 repeat-averaged vs 0.671 per-repeat mean). The per-repeat-mean comparison is reported next to the pre-registered one, for description only. No decision is based on it and the Holm family is unchanged. The stored baselines hold 3 repeats and Mazza's unified run 5, so this comparison uses the 3 shared repeats.
4. **Efficiency timings** were taken in one session with no other job of ours running, but system load was not otherwise controlled; ratios are the claim.
