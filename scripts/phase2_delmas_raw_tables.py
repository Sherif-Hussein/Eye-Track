"""Markdown tables for the delmas_raw comparison card, from results.json.

Usage: python -I scripts/phase2_delmas_raw_tables.py <results.json> <out.md>
"""
import json
import sys
from pathlib import Path

SETS = ["64_g1", "64_g2", "64_g3", "32_g3", "32_g4", "32_g5"]
NAMES = {"64_g1": "G1 control (64)", "64_g2": "G2 concealment (64)", "64_g3": "G3 simple CM (64)",
         "32_g3": "G3 simple CM (last 32)", "32_g4": "G4 enhanced CM (last 32)", "32_g5": "G5 CM+feedback (last 32)"}


def ms(d, k):
    return f"{d[k]['mean']:.3f} ± {d[k]['sd']:.3f}"


def row_all(name, res):
    """One line per configuration: mean over the six group-sets of each metric's repeat mean."""
    import numpy as np
    keys = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "auc", "eer", "hter"]
    img = {k: np.mean([res[s]["image"][k]["mean"] for s in SETS]) for k in keys}
    sd = {k: np.mean([res[s]["image"][k]["sd"] for s in SETS]) for k in keys}
    cm = {k: np.mean([res[s]["image"][k]["mean"] for s in SETS if s != "64_g1"]) for k in ("auc", "eer", "hter")}
    pp = np.mean([res[s]["per_participant_mean"]["auc"]["mean"] for s in SETS])
    cit = {k: np.mean([res[s]["participant_cit"][k]["mean"] for s in SETS]) for k in ("auc", "eer", "hter")}
    return (f"| {name} | " + " | ".join(f"{img[k]:.3f}±{sd[k]:.3f}" for k in keys)
            + f" | {cm['auc']:.3f} / {cm['eer']:.3f} / {cm['hter']:.3f} | {pp:.3f} | {cit['auc']:.3f} / {cit['eer']:.3f} / {cit['hter']:.3f} |")


def main():
    R = json.loads(Path(sys.argv[1]).read_text())
    L = []
    hdr = ("| Config | Acc | BA | Sens | Spec | Prec | F1 | MCC | AUC | EER | HTER | CM sets AUC/EER/HTER | "
           "per-participant AUC | Participant CIT AUC/EER/HTER |")
    sep = "|" + "---|" * 15
    # per-set table for the honest baseline and P4
    L.append("## Per group-set (image level, mean ± SD over 3 repeats)\n")
    L.append("| Config | " + " | ".join(NAMES[s] for s in SETS) + " |")
    L.append("|---|" + "---|" * len(SETS))
    tags = [("Published table + SVM (honest baseline)", R.get("baseline_book_svm")),
            ("Own detector, published label-dependent aggregation, SVM", R.get("pubrule_svm")),
            ("Counts only, published aggregation, SVM", R.get("pubrule_counts_only_svm")),
            ("Counts only, all presentations, SVM", R.get("counts_all_svm"))]
    for rate, v in R.get("rate_curve", {}).items():
        tags.append((f"P4 SVM {rate} Hz", v["svm"]))
        tags.append((f"ALL_z LR {rate} Hz", v["lr_ALL_z"]))
    for k, v in R.get("feature_sets", {}).items():
        tags.append((f"{k} 1000 Hz", v))
    for k, v in R.get("models", {}).items():
        tags.append((f"{k} 1000 Hz", v))
    for name, res in tags:
        if res:
            L.append(f"| {name} | " + " | ".join(
                f"AUC {res[s]['image']['auc']['mean']:.3f} / EER {res[s]['image']['eer']['mean']:.3f} / "
                f"HTER {res[s]['image']['hter']['mean']:.3f}" for s in SETS) + " |")
    L.append("\n## All metrics, mean over the six group-sets\n")
    L.append(hdr)
    L.append(sep)
    for name, res in tags:
        if res:
            L.append(row_all(name, res))
    L.append("\n## Transfer\n")
    L.append("| Config | Direction | Test AUC | BA | EER | HTER | MCC | per-group AUC | CIT AUC/HTER |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, v in R.get("transfer", {}).items():
        for d, e in v.items():
            a = e["all"]
            pg = ", ".join(f"G{g}: {x['image']['auc']['mean']:.3f}" for g, x in e["per_group"].items())
            L.append(f"| {k} | {d} | {ms(a['image'], 'auc')} | {a['image']['balanced_accuracy']['mean']:.3f} | "
                     f"{a['image']['eer']['mean']:.3f} | {a['image']['hter']['mean']:.3f} | {a['image']['mcc']['mean']:.3f} | {pg} | "
                     f"{a['participant_cit']['auc']['mean']:.3f} / {a['participant_cit']['hter']['mean']:.3f} |")
    Path(sys.argv[2]).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
