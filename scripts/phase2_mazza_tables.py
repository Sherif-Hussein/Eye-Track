"""Render markdown tables for the comparison card from results/phase2/mazza/results.json, or, while the
run is unfinished, from screen_summary.json + extras_cache.json + selection.json (sections whose numbers
do not exist yet are skipped).

Usage: python -I scripts/phase2_mazza_tables.py <results_dir>   (writes tables.md there)
"""
import json
import sys
from pathlib import Path

NL = chr(10)
M = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "eer", "hter"]


def f(x, d=3):
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


def ms(m):
    return f"{m['mean']:.3f} ± {m['sd']:.3f}"


def main():
    out = Path(sys.argv[1])
    r = load_any(out)
    L = []
    sections(r, L)
    (out / "tables.md").write_text(NL.join(L), encoding="utf-8")
    if "--interim-card" in sys.argv:
        import datetime
        head = (f"> **INTERIM** ({datetime.datetime.now():%Y-%m-%d %H:%M}). Sections 1-2 are final; the tables below are "
                "regenerated from the saved results after every finished step. "
                + (out / "interim_headline.md").read_text(encoding="utf-8").strip())
        body = (out / "card_sections_1_2.md").read_text(encoding="utf-8")
        card = NL.join([head, "", body, "", "## 3-7. Interim tables (auto-generated from saved results)", "", NL.join(L)])
        (out / "comparison_card.md").write_text(card, encoding="utf-8")
    print(f"wrote {out / 'tables.md'} ({len(L)} lines)")


def load_any(out):
    rd = lambda n: json.loads((out / n).read_text(encoding="utf-8")) if (out / n).exists() else None  # noqa: E731
    r = rd("results.json")
    if r is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from edd.mazza import TASKS
        r = {"screen": rd("screen_summary.json"), "tasks": {k: v[2] for k, v in TASKS.items()}}
        r["best"] = {t: max(rows, key=lambda k: rows[k]["repeats"]["auc"]["mean"]) for t, rows in r["screen"].items()}
        c = rd("extras_cache.json") or {}
        for k, v in c.items():
            parts = k.split("|")
            kind = parts[0]
            if kind == "ci":
                r.setdefault("ci95", {})[f"{parts[1]}__{parts[2]}"] = v
            elif kind == "calib":
                r.setdefault("calibration_selective", {}).setdefault(parts[1], {})[parts[2]] = v
            elif kind == "noclip":
                r.setdefault("clip_ablation", {})[parts[1]] = {"config": parts[2], "no_clip": v}
            elif kind == "fi":
                r.setdefault("feature_importance", {})[parts[1]] = {parts[2]: v}
            elif kind == "blocks":
                r.setdefault("block_ablation", {})[parts[1]] = v
            elif kind == "reserved":
                r.setdefault("reserved_repeats", {}).setdefault(parts[1], {})[parts[2]] = v
            elif kind == "perm":
                r.setdefault("permutation", {})[parts[1]] = v
            elif kind == "multiclass":
                r["multiclass"] = v
    if r.get("selection") is None and (out / "selection.json").exists():
        r["selection"] = rd("selection.json")
    return r


def sections(r, L):
    for task, rows in r["screen"].items():
        L.append(f"\n#### {task}: {r['tasks'][task]}\n")
        L.append("| config | idea | " + " | ".join(M) + " | ΔAUC vs baseline [95% CI] | p (Holm) | acc@80% | acc@60% |")
        L.append("|" + "---|" * (len(M) + 6))
        for name, v in rows.items():
            rep = v["repeats"]
            vb = v.get("vs_baseline")
            delta = (f"{vb['delta_auc']:+.3f} [{vb['ci95'][0]:+.3f}, {vb['ci95'][1]:+.3f}]" if vb else "—")
            p = f"{vb['p_holm']:.3f}" if vb and "p_holm" in vb else "—"
            sel = v["selective"]
            star = " **(best)**" if r["best"].get(task) == name else ""
            L.append(f"| {name}{star} | {v['idea']} | " + " | ".join(ms(rep[k]) for k in M) +
                     f" | {delta} | {p} | {f(sel['80']['accuracy'])} | {f(sel['60']['accuracy'])} |")
    L.append("\n#### Best vs baseline with participant-bootstrap 95% CI (repeat-averaged scores)\n")
    L.append("| task | config | AUC | accuracy | balanced acc | EER | HTER | MCC |")
    L.append("|---|---|---|---|---|---|---|---|")
    for task, best in r["best"].items():
        for name in dict.fromkeys([best, "base_pub36_lr_unreg"]):
            pt = r["screen"][task][name]["avg_scores"]
            ci = r.get("ci95", {}).get(f"{task}__{name}")
            if ci is None:
                continue
            L.append(f"| {task} | {name} | " + " | ".join(
                f"{pt[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k in ("auc", "accuracy", "balanced_accuracy", "eer", "hter", "mcc")) + " |")
    L.append("\n#### Permutation tests (labels shuffled between participants, full nested pipeline, repeat 0 folds)\n")
    L.append("| task | config | observed AUC | null mean | null 95th pct | p | n_perm |")
    L.append("|---|---|---|---|---|---|---|")
    for task, p in r.get("permutation", {}).items():
        L.append(f"| {task} | {r['best'][task]} | {p['observed_auc']:.3f} | {p['null_mean']:.3f} | {p['null_p95']:.3f} | "
                 f"{p['p_value']:.4f} | {p['n_perm']} |")
    L.append("\n#### Calibration and selective prediction (mean ± SD over repeats)\n")
    L.append("| task | config | Brier raw | ECE raw | Brier Platt | ECE Platt | acc@100% | acc@80% | acc@60% | BA@80% | BA@60% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for task, d in r.get("calibration_selective", {}).items():
        for name, v in d.items():
            rp, pp, s = v.get("raw_prob", {}), v["platt_prob"], v["sel_raw"]
            L.append(f"| {task} | {name} | {ms(rp['brier']) if rp else 'n/a'} | {ms(rp['ece']) if rp else 'n/a'} | "
                     f"{ms(pp['brier'])} | {ms(pp['ece'])} | {ms(s['100']['accuracy'])} | {ms(s['80']['accuracy'])} | "
                     f"{ms(s['60']['accuracy'])} | {ms(s['80']['balanced_accuracy'])} | {ms(s['60']['balanced_accuracy'])} |")
    L.append("\n#### Outlier clipping ablation (best config without the in-fold 1st/99th percentile clipping)\n")
    L.append("| task | config | AUC with clip | AUC without clip | EER with | EER without |")
    L.append("|---|---|---|---|---|---|")
    for task, v in r.get("clip_ablation", {}).items():
        w = r["screen"][task][v["config"]]["repeats"]
        L.append(f"| {task} | {v['config']} | {ms(w['auc'])} | {ms(v['no_clip']['auc'])} | {ms(w['eer'])} | {ms(v['no_clip']['eer'])} |")
    L.append("\n#### Block ablation: L2-LR on one raw block (neutral = negative control)\n")
    L.append("| task | block | AUC | AUC 95% CI | balanced acc | EER |")
    L.append("|---|---|---|---|---|---|")
    for task, blocks in r.get("block_ablation", {}).items():
        for blk, v in blocks.items():
            rp = v["repeats"]
            L.append(f"| {task} | {blk} | {ms(rp['auc'])} | [{v['ci95_auc'][0]:.3f}, {v['ci95_auc'][1]:.3f}] | "
                     f"{ms(rp['balanced_accuracy'])} | {ms(rp['eer'])} |")
    L.append("\n#### Held-back fold seeds (repeats 3-4, same participants)\n")
    L.append("| task | config | AUC screening | AUC held-back seeds | BA held-back | EER held-back |")
    L.append("|---|---|---|---|---|---|")
    for task, d in r.get("reserved_repeats", {}).items():
        for name, v in d.items():
            L.append(f"| {task} | {name} | {ms(r['screen'][task][name]['repeats']['auc'])} | {ms(v['auc'])} | "
                     f"{ms(v['balanced_accuracy'])} | {ms(v['eer'])} |")
    L.append("\n#### 4-class (G1/G2/G3/G4), mean ± SD over repeats\n")
    keys = ["macro_auc_ovr", "weighted_auc_ovr", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
            "macro_eer_ovr", "macro_hter_ovr_argmax"]
    L.append("| config | " + " | ".join(keys) + " | correct per group (repeat 0) |")
    L.append("|" + "---|" * (len(keys) + 2))
    for name, v in r.get("multiclass", {}).items():
        L.append(f"| {name} | " + " | ".join(ms(v["repeats"][k]) for k in keys) + f" | {v['correct_per_group_repeat0']} |")
    if r.get("selection"):
        L.append("\n#### Selection-adjusted best (config chosen per outer fold on the other 9 folds)\n")
        L.append("| task | candidates | hindsight best | AUC | accuracy | balanced acc | EER | HTER | AUC 95% CI (avg scores) | picks |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for task, v in r["selection"].items():
            rp = v["repeats"]
            L.append(f"| {task} | {v['n_candidates']} | {v['hindsight_best']} | {ms(rp['auc'])} | {ms(rp['accuracy'])} | "
                     f"{ms(rp['balanced_accuracy'])} | {ms(rp['eer'])} | {ms(rp['hter'])} | "
                     f"[{v['ci95']['auc'][0]:.3f}, {v['ci95']['auc'][1]:.3f}] | {v['pick_counts']} |")
    if "efficiency" not in r:
        return
    L.append("\n#### Efficiency (single-threaded, this laptop)\n")
    e = r["efficiency"]
    L.append(f"Hardware: {e['hardware']}\n")
    L.append("| model | task | inputs | latency p50/p95 ms (1 participant, end-to-end) | feature extraction p50 ms | "
             "throughput participants/s | peak mem KB | model size KB | classifier params | preprocessing params | nested fit s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in e.items():
        if k == "hardware":
            continue
        lat, fe = v["latency_ms_per_participant_end_to_end"], v["latency_ms_feature_extraction_only"]
        L.append(f"| {v['config']} | {v['task']} | {v['n_model_inputs']} | {lat['p50']:.3f} / {lat['p95']:.3f} | "
                 f"{fe['p50']:.3f} | {v['throughput_participants_per_s_batch157']:.0f} | "
                 f"{v['peak_memory_bytes_inference_one'] / 1024:.1f} | {v['model_size_bytes'] / 1024:.1f} | "
                 f"{v['stored_parameters_classifier']} | {v['stored_parameters_preprocessing_approx']} | "
                 f"{v['fit_seconds_full_nested_search']:.1f} |")


if __name__ == "__main__":
    main()
