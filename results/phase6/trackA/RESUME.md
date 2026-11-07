# RESUME, Phase 6 Track A: COMPLETE (2026-11-02)

All eight analyses (A1 to A8) finished and saved; nothing is running. Re-run any analysis from the project root with `.venv/Scripts/python.exe -I scripts/phase6_<name>.py` (each writes its own `checkpoint_A*.json` and merges its section into `results.json`; `--card-only` re-renders the card paragraph for A1, A2, A4, A5 from the stored section).

| Analysis | Script | Run time |
|---|---|---|
| A1 | `scripts/phase6_a1_chen_roc.py` | about 6 min |
| A2 | `scripts/phase6_a2_auc_definition.py` | about 6 min |
| A3 | `scripts/phase6_a3_realised_fpr.py` | under 1 min |
| A4 | `scripts/phase6_a4_delmas_cit.py` | about 4 min |
| A5 | `scripts/phase6_a5_delmas_contrasts.py` | about 10 min |
| A6 | `scripts/phase6_a6_power.py` | about 3 min |
| A7 | `scripts/phase6_a7_preti_rt.py` | about 1.5 min |
| A8 | `scripts/phase6_a8_corrected_cv.py` | under 1 min |

Shared helpers: `scripts/phase6_common.py`. Definitions: `PREREG.md`. Deviations: `DEVIATIONS.md`. Summary paragraphs: `card.md`.
