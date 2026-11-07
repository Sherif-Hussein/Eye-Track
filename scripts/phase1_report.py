"""Render Phase 1 summaries as a Markdown report.

Usage: python -I scripts/phase1_report.py <results/phase1> <out.md>
"""
import json
import sys
from pathlib import Path

CELNIAK_PUBLISHED = {"svm": "0.95 acc / 1.00 AUC (with selection); 0.90 / 0.90 (without)",
                     "rf": "0.64 acc / 0.62 AUC", "xgb": "0.74 acc / 0.75 AUC"}


def ms(d, k):
    return f"{d[k]['mean']:.3f} ± {d[k]['sd']:.3f}"


def ci(c, k):
    return f"[{c[k][0]:.2f}, {c[k][1]:.2f}]"


def celniak_section(r):
    lines = ["## Celniak et al. 2026 (39 participants, 819 slides, Tobii Pro Fusion 120 Hz)", "",
             "Participant-disjoint outer folds (1 guilty + 1 innocent held out), 5 repeats; hyperparameters and "
             "feature selection tuned in participant-grouped inner folds. Participant score = mean of its 21 slide scores.", "",
             "| Setting | Participant AUC | Participant acc | Bal. acc | Sens | Spec | F1 | MCC | EER | HTER | Trial AUC | 95% CI AUC (participant bootstrap) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in r.items():
        if not isinstance(v, dict) or "participant" not in v:
            continue
        p, t = v["participant"], v["trial"]
        lines.append(f"| {name} | {ms(p, 'auc')} | {ms(p, 'accuracy')} | {ms(p, 'balanced_accuracy')} | "
                     f"{p['sensitivity']['mean']:.2f} | {p['specificity']['mean']:.2f} | {p['f1']['mean']:.2f} | "
                     f"{p['mcc']['mean']:.2f} | {p['eer']['mean']:.3f} | {p['hter']['mean']:.3f} | {t['auc']['mean']:.3f} | "
                     f"{ci(v['participant_ci95'], 'auc')} |")
    if "permutation" in r.get("svm_nested", {}):
        pm = r["svm_nested"]["permutation"]
        lines += ["", f"Permutation test (honest SVM, {pm['n_perm']} label shuffles between participants): observed AUC "
                      f"{pm['observed_auc']:.3f}, null mean {pm['null_mean']:.3f}, null 95th pct {pm['null_p95']:.3f}, "
                      f"p = {pm['p_value']:.3f}."]
    lines += ["", "Published (Celniak et al. 2026, Table 4, tuning not described as nested): " +
              "; ".join(f"{k.upper()} {v}" for k, v in CELNIAK_PUBLISHED.items()) + "."]
    e = r["efficiency"]
    lines += ["", "### Deployment cost (final SVM, single CPU thread)", "",
              f"- Hardware: {e['hardware']['processor']}, {e['hardware']['platform']}",
              f"- Final hyperparameters: {e['final_params']}; selected features: {e['selected_features']}",
              f"- Model size: {e['model_size_bytes'] / 1024:.1f} KiB; stored parameters: {e['parameter_count']:,}",
              f"- Inference latency per participant (21 slides): p50 {e['inference_per_participant_21_slides_ms']['p50']:.2f} ms, "
              f"p95 {e['inference_per_participant_21_slides_ms']['p95']:.2f} ms",
              f"- Inference per slide: p50 {e['inference_per_slide_ms']['p50']:.2f} ms; throughput "
              f"{e['inference_throughput_slides_per_s']:,.0f} slides/s; peak memory {e['inference_peak_memory_bytes'] / 1024:.1f} KiB",
              f"- Feature extraction per recording (pandas reference implementation): p50 "
              f"{e['feature_extraction_per_recording_ms']['p50']:.0f} ms; {e['feature_extraction_samples_per_s']:,.0f} samples/s "
              f"= {e['realtime_factor_vs_120Hz']:.0f}x real time at 120 Hz; peak memory "
              f"{e['feature_extraction_peak_memory_bytes'] / 2**20:.1f} MiB"]
    return lines


def delmas_section(r):
    lines = ["## Delmas et al. 2024 (famous-faces CIT, EyeLink 1000 Hz, 15 participants per group)", "",
             "Leave-one-participant-out within each group, as in the paper. Per-participant metrics are means over "
             "held-out participants; EER/HTER/MCC are pooled over all held-out trials.", "",
             "| Set | Published AUC / BA | Reproduced AUC / BA | Honest AUC / BA | Honest EER | Honest HTER | Honest MCC | Random-split AUC | Chance-label AUC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, v in r.items():
        if k == "efficiency":
            continue
        p, h = v["paper"]["per_participant"], v["honest"]
        lines.append(f"| {k} | {v['published']['auc']:.2f} / {v['published']['ba']:.2f} | "
                     f"{p['auc']['mean']:.3f} / {p['ba']['mean']:.3f} | "
                     f"{h['per_participant']['auc']['mean']:.3f} / {h['per_participant']['ba']['mean']:.3f} | "
                     f"{h['pooled']['eer']:.3f} | {h['pooled']['hter']:.3f} | {h['pooled']['mcc']:.3f} | "
                     f"{v['leak_random_trial_split']['auc']:.3f} | {v['paper_chance_baseline']['auc']['mean']:.3f} |")
    e = r["efficiency"]
    lines += ["", f"Deployment cost (last fitted SVM, 4 features): {e['model_size_bytes'] / 1024:.1f} KiB, "
                  f"{e['parameter_count']:,} stored parameters, p50 {e['inference_per_participant_32_trials_ms']['p50']:.2f} ms "
                  f"per participant (32 trials), peak memory {e['inference_peak_memory_bytes'] / 1024:.1f} KiB."]
    return lines


def main(results_dir, out_md):
    results_dir = Path(results_dir)
    lines = ["# Phase 1 results: evaluation harness and re-baselines", ""]
    for name, fn in (("celniak", celniak_section), ("delmas", delmas_section)):
        f = results_dir / name / "summary.json"
        if f.exists():
            lines += fn(json.loads(f.read_text())) + [""]
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
