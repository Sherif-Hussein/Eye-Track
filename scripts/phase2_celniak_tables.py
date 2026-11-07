"""Phase 2 (celniak track): render the numeric tables of comparison_card.md from results.json.

Usage: python -I scripts/phase2_celniak_tables.py <results.json> <out.md>
"""
import json
import sys
from pathlib import Path

METRICS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "eer", "hter"]
HEAD = ["AUC", "Acc", "BalAcc", "Sens", "Spec", "Prec", "F1", "MCC", "EER", "HTER"]


def ms(p, k):
    return f"{p[k]['mean']:.3f} ± {p[k]['sd']:.3f}"


def screen_table(r):
    names = [n for n, v in r.items() if isinstance(v, dict) and "participant" in v]
    order = {"baseline": 0, "reference": 1, "idea": 2, "ablation": 3}
    names.sort(key=lambda n: (order[r[n]["kind"]], -r[n]["participant"]["auc"]["mean"]))
    out = ["| Setting | Kind | " + " | ".join(HEAD) + " | Order-stratified AUC | ΔAUC vs baseline [95% CI] | p (Holm) |",
           "|---|---|" + "---|" * (len(HEAD) + 3)]
    pv = r["paired_vs_baseline"]
    for n in names:
        p = r[n]["participant"]
        c = pv.get(n)
        d = f"{c['auc_diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]" if c else "n/a"
        h = f"{c['p_two_sided']:.3f} ({c['p_holm']:.3f})" if c else "n/a"
        s = r[n]["order_stratified_auc"]
        out.append(f"| {n} | {r[n]['kind']} | " + " | ".join(ms(p, k) for k in METRICS) +
                   f" | {s['mean']:.3f} ± {s['sd']:.3f} | {d} | {h} |")
    return out


def ci_table(r, names):
    out = ["| Setting | " + " | ".join(HEAD) + " |", "|---|" + "---|" * len(HEAD)]
    for n in names:
        a, c = r[n]["participant_avg_scores"], r[n]["participant_ci95"]
        out.append(f"| {n} | " + " | ".join(f"{a[k]:.3f} [{c[k][0]:.3f}, {c[k][1]:.3f}]" for k in METRICS) + " |")
    return out


def perm_lines(r, names):
    out = []
    for n in names:
        for key in ("permutation", "permutation_within_order"):
            if key in r[n]:
                x = r[n][key]
                out.append(f"- {n}, {key}: {x['n_perm']} shuffles, observed AUC {x['observed_auc']:.3f}, null mean "
                           f"{x['null_mean']:.3f}, null 95th percentile {x['null_p95']:.3f}, p = {x['p_value']:.3f}")
    return out


def eff_table(e):
    b, c = e["baseline_svm_paper"], e.get("best_candidate", {})
    lat = lambda d, k: f"{d[k]['p50']:.2f} / {d[k]['p95']:.2f}" if k in d else "n/a"
    rows = [("Inference latency per participant, p50 / p95 (ms)", lat(b, "latency_per_participant_ms"), lat(c, "latency_per_participant_ms")),
            ("Inference latency per slide, p50 / p95 (ms)", lat(b, "latency_per_slide_ms"), "n/a (decision needs the whole session)"),
            ("Throughput (decisions/s)", f"{b['throughput_slides_per_s']:,.0f} slides/s", f"{c.get('throughput_participants_per_s', 0):,.0f} participants/s"),
            ("Peak inference memory (KiB)", f"{b['peak_memory_inference_bytes'] / 1024:.1f}", f"{c.get('peak_memory_inference_bytes', 0) / 1024:.1f}"),
            ("Model size (KiB)", f"{b['model_size_bytes'] / 1024:.1f}", f"{c.get('model_size_bytes', 0) / 1024:.1f}"),
            ("Stored parameters", f"{b['parameter_count']:,}", f"{c.get('parameter_count', 0):,}"),
            ("Feature extraction per recording, p50 / p95 (ms)", lat(b, "feature_extraction_per_recording_ms"), lat(c, "feature_extraction_per_recording_ms")),
            ("Extraction samples/s", f"{b['feature_extraction_samples_per_s']:,.0f}", f"{c.get('feature_extraction_samples_per_s', 0):,.0f}"),
            ("Real-time factor at the sensor rate", f"{b['realtime_factor_vs_sensor']:.0f}x", f"{c.get('realtime_factor_vs_sensor', 0):.0f}x"),
            ("Extraction peak memory (MiB)", f"{b['feature_extraction_peak_memory_bytes'] / 2**20:.1f}", f"{c.get('feature_extraction_peak_memory_bytes', 0) / 2**20:.1f}"),
            ("Participant table build (ms per participant)", "n/a", f"{c.get('participant_table_build_ms_per_participant', 0):.1f}")]
    return ["| Measure | Published method (our SVM reimplementation) | Best candidate |", "|---|---|---|"] + \
        [f"| {a} | {x} | {y} |" for a, x, y in rows]


def main(path, out):
    r = json.loads(Path(path).read_text())
    best, probe = r["best_candidate"], r["best_probe_specific"]
    lines = ["## Screen", ""] + screen_table(r) + ["", "## CI", ""] + \
        ci_table(r, ["baseline_svm_paper", best, probe, "validation_accuracy_lr", "order_only"]) + \
        ["", "## Permutation", ""] + perm_lines(r, [best, probe]) + (["", "## Efficiency", ""] + eff_table(r["efficiency"]) if "efficiency" in r else [])
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:3])
