"""Markdown tables for the Preti comparison card, rendered from results.json / final.json / efficiency.json.

Usage: python -I scripts/phase2_preti_tables.py results/phase2/preti <output.md>
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
res = json.loads((OUT / "results.json").read_text())
fin = json.loads((OUT / "final.json").read_text()) if (OUT / "final.json").exists() else {}
eff = json.loads((OUT / "efficiency.json").read_text()) if (OUT / "efficiency.json").exists() else {}
L = []
MET = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "eer", "hter", "brier", "ece"]


def cell(m, k):
    return f"{m[k]['mean']:.3f} ± {m[k]['sd']:.3f}"


L += ["## Arms", "", "| Arm | Group | Modality | Feats | " + " | ".join(m.upper() for m in MET) + " | ΔAUC vs baseline [95% CI] | p boot | p Holm |",
      "|" + "---|" * (4 + len(MET) + 3)]
for n, a in res["arms"].items():
    t = res["paired_tests"].get(n)
    tx = f"{t['delta_auc']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] vs {t['vs']} | {t['p_boot']:.3f} | {t.get('p_holm', float('nan')):.3f}" if t else "n/a | n/a | n/a"
    feats = a.get("n_features", len(a.get("features", [])))
    L.append(f"| {n} | {a['group']} | {a['modality']} | {feats} | " + " | ".join(cell(a["participant"], m) for m in MET) + f" | {tx} |")

L += ["", "## Other paired tests", "", "| Comparison | ΔAUC [95% CI] | p boot |", "|---|---|---|"]
for k, t in res["paired_tests"].items():
    if "__minus__" in k:
        L.append(f"| {k.replace('__minus__', ' minus ')} | {t['delta_auc']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {t['p_boot']:.3f} |")

L += ["", "## Paper-style reproduction (10 x stratified 10-fold, default hyperparameters, non-nested)", "",
      "| Data \\| features | LR acc | SVM acc | RF acc | NB acc | Tree acc | kNN acc | Best mean acc | Best single seed |", "|---|---|---|---|---|---|---|---|---|"]
for k, v in res["reproduction"].items():
    if "LR" in v and isinstance(v["LR"], dict):
        L.append(f"| {k} | " + " | ".join(f"{v[m]['acc_mean']:.3f}" for m in ("LR", "SVM", "RF", "NB", "Tree", "kNN")) +
                 f" | {v['_up_to_acc_mean']:.3f} ({v['_best_model']}) | {v['_up_to_acc_any_seed']:.3f} |")
    else:
        L.append(f"| {k} (trial accuracy, leaky) | {v['LR']:.3f} | | {v['RF']:.3f} | | | | | |")

if fin:
    L += ["", "## Candidates (repeat-averaged scores, 95% CI)", "", "| Arm | AUC | Accuracy | EER | HTER | AUC drop s3 | AUC flip s3 |", "|---|---|---|---|---|---|---|"]
    for n, r in fin["by_arm"].items():
        m, c = r["avg_score_metrics"], r["ci95"]
        f = lambda k: f"{m[k]:.3f} [{c[k][0]:.3f}, {c[k][1]:.3f}]"  # noqa: E731, B023
        L.append(f"| {n} | {f('auc')} | {f('accuracy')} | {f('eer')} | {f('hter')} | {r['auc_drop_subject3']:.3f} | {r['auc_flip_subject3']:.3f} |")
    if "ocular_validity" in fin:
        v = fin["ocular_validity"]
        L += ["", "## RT dependence of participant scores", "", "| Arm | AUC (mean over repeats) | AUC (averaged scores) | ρ score vs mean log RT | ρ within liars | ρ within truth-tellers | AUC after regressing out mean log RT |", "|---|---|---|---|---|---|---|"]
        for n, r in v["arms"].items():
            L.append(f"| {n} | {r['participant']['auc']['mean']:.3f} ± {r['participant']['auc']['sd']:.3f} | {r['auc']:.3f} | {r['spearman_score_vs_meanlogRT']:+.3f} | {r['spearman_within_liars']:+.3f} | {r['spearman_within_truth']:+.3f} | {r['auc_after_regressing_out_meanlogRT']:.3f} |")
        L += ["", "Trial correlation with log RT: " + json.dumps(v["trial_corr_with_logRT"])]
    if "permutation" in fin:
        L += ["", "## Permutation", ""]
        for k, p in fin["permutation"].items():
            L.append(f"- {k} ({p['arm']}): observed {p['observed_auc_mean_over_repeats']:.3f}, null mean {p['null_mean']:.3f} ± {p['null_sd']:.3f}, null p95 {p['null_p95']:.3f}, p = {p['p_value']:.3f}, n = {p['n_perm']}")
    for k in ("no_model_RT_U", "no_model_RT_all", "corr_subject_mean_RT_vs_nfix", "group_means", "lr_weights"):
        if k in fin:
            L += ["", f"{k}: {json.dumps(fin[k])}"]
    if "univariate_auc" in fin:
        L += ["", "Top univariate AUC: " + json.dumps({k: round(v, 3) for k, v in list(fin["univariate_auc"].items())[:14]}),
              "Ocular univariate AUC: " + json.dumps({k: round(v, 3) for k, v in fin["univariate_auc"].items() if k.startswith(("pupil", "blink", "pc_"))})]
if eff:
    L += ["", "## Efficiency", "", json.dumps(eff["hardware"]), "",
          "| Method | Arm | Model | Feats | Params | Size (B) | Feature extraction p50/p95 (ms) | Inference p50/p95 (ms) | End-to-end p50/p95 (ms) | Trial features p50/p95 (ms) | Peak mem (B) | Participants/s | Fixation rows/s |",
          "|" + "---|" * 13]
    for k, r in eff["methods"].items():
        q = lambda n: f"{r[n]['p50']:.3f} / {r[n]['p95']:.3f}"  # noqa: E731, B023
        L.append(f"| {k} | {r['arm']} | {r['chosen']} | {r['n_features']} | {r['stored_parameters']} | {r['model_size_bytes']} | {q('feature_extraction_ms_per_participant')} | {q('inference_only_ms_per_participant')} | {q('end_to_end_ms_per_participant')} | {q('trial_feature_extraction_ms')} | {r['peak_memory_bytes_end_to_end']} | {r['throughput_participants_per_s']:.1f} | {r['throughput_fixation_rows_per_s']:.0f} |")
Path(sys.argv[2]).write_text("\n".join(L), encoding="utf-8")
print(len(L), "lines")
