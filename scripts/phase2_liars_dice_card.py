"""Phase 2 / liars_dice: render comparison_card.md from the saved JSON outputs (results.json or
results_partial.json, reproduction.json, data_audit.json, followup.json). Sections whose numbers do not exist
yet are marked PENDING and the card is marked INTERIM until everything is present.

Usage: python -I scripts/phase2_liars_dice_card.py <results_dir> [--final]
"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
FINAL = "--final" in sys.argv


def jl(name):
    p = D / name
    return json.loads(p.read_text()) if p.exists() else None


res = jl("results.json")
if res is None or res.get("INTERIM"):
    res = jl("results_partial.json") or {}
rep, audit, fu = jl("reproduction.json"), jl("data_audit.json"), jl("followup.json")
NARR = D / "card_narrative.md"
narr = {}
if NARR.exists():
    cur = None
    for line in NARR.read_text(encoding="utf-8").splitlines():
        if line.startswith("@@"):
            cur = line[2:].strip(); narr[cur] = []
        elif cur:
            narr[cur].append(line)
narr = {k: "\n".join(v).strip() for k, v in narr.items()}

f3 = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
L = []
complete = bool(res.get("transfer")) and fu is not None and all(e in res.get("best", {}) for e in ("pvc", "pvp"))
if not (FINAL and complete):
    L.append("# INTERIM — liars_dice comparison card\n")
    L.append(f"Status: {'all numbers present, narrative pending' if complete else 'sections without numbers are marked PENDING'}. See RESUME.md.\n")
else:
    L.append("# liars_dice comparison card (Phase 2 screening)\n")
L.append(narr.get("summary", ""))

# ---------------- 1
L.append("\n## 1. The published study\n")
L.append(narr.get("study", "PENDING"))

# ---------------- 2
L.append("\n## 2. Reproduction under their protocol (`reproduction.json`)\n")
L.append(narr.get("reproduction", ""))
if rep:
    L.append("\n| Exp | Measure | Ours lie / truth (SD) | Ours F, η²p | Paper lie / truth | Paper F, η²p |\n|---|---|---|---|---|---|")
    for exp, block in rep["bidder"].items():
        for col, r in block.items():
            P = r["published"]
            pl = "n.s. (no values)" if P["lie_M"] is None else f"{P['lie_M']} / {P['truth_M']}"
            pf = P["p"] if P["F"] is None else f"{P['F']}, {P['eta2p']}"
            L.append(f"| {exp} | {col} | {r['pos_M']:.2f} ({r['pos_SD']:.2f}) / {r['neg_M']:.2f} ({r['neg_SD']:.2f}) | "
                     f"F({r['df']})={r['F']:.1f}, {r['eta2p']:.2f} (p={r['p']:.3g}) | {pl} | {pf} |")
    L.append("\nChallenger (PvP), contest vs accept:\n\n| Measure | Ours contest / accept | Ours F, η²p | Paper contest / accept | Paper F, η²p |\n|---|---|---|---|---|")
    for col, r in rep["challenger_pvp"].items():
        P = r["published"]
        L.append(f"| {col} | {r['pos_M']:.2f} / {r['neg_M']:.2f} | {r['F']:.1f}, {r['eta2p']:.2f} | {P['contest_M']} / {P['accept_M']} | {P['F']}, {P['eta2p']} |")

# ---------------- 3-4
METRICS = ["auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "eer", "hter"]
L.append("\n## 3–4. Honest baseline and ideas screened (our protocol)\n")
L.append(narr.get("protocol", ""))
for exp in ("pvc", "pvp"):
    ideas = res.get("ideas", {}).get(exp)
    if not ideas:
        L.append(f"\n**{exp}: PENDING**\n"); continue
    L.append(f"\n**{exp.upper()}** — trial level, mean ± SD over 3 repeats (threshold 0). The honest baseline is `I1_paper_ocular`.\n")
    L.append("| Idea | Block | " + " | ".join(m.upper() if m in ("auc", "eer", "hter", "mcc") else m for m in METRICS) + " |")
    L.append("|---|---|" + "---|" * len(METRICS))
    for n, ev in ideas.items():
        t = ev["trial"]
        L.append(f"| {n} | {ev['block']} | " + " | ".join(f"{t[m]['mean']:.3f}±{t[m]['sd']:.3f}" for m in METRICS) + " |")
    L.append(f"\n{exp.upper()} per participant (scores averaged over repeats; detection = one-sided Mann-Whitney lie > truth at α=.05), plus AUC within bid-position strata:\n")
    L.append("| Idea | mean ppAUC | median | frac ppAUC>.5 | detected | mean pp BA | mean pp EER | mean pp HTER | AUC within bid-pos |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for n, ev in ideas.items():
        p = ev["per_participant"]
        L.append(f"| {n} | {p['auc_mean']:.3f} | {p['auc_median']:.3f} | {p['frac_auc_gt_0.5']:.2f} | {p['detected_alpha05']}/{p['n_participants']} | "
                 f"{p['bacc_mean']:.3f} | {p['eer_mean']:.3f} | {p['hter_mean']:.3f} | {res['within_bidpos_auc'][exp][n]['weighted']:.3f} |")
    cmp = res.get("comparisons", {}).get(exp)
    if cmp:
        L.append(f"\n{exp.upper()} paired comparisons on identical folds. ΔAUC is pooled trial AUC with a participant-cluster bootstrap CI, Holm across the ocular ideas; Wilcoxon is on per-participant AUCs (Holm in `followup.json`):\n")
        L.append("| Comparison | ΔAUC [95% CI] | p | Holm p | reliable | Δ mean ppAUC | Wilcoxon p |\n|---|---|---|---|---|---|---|")
        for n, v in cmp["ocular_vs_honest_baseline"].items():
            L.append(f"| {n} vs I1_paper_ocular | {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['p_two_sided']:.3f} | {v['holm_p']:.3f} | {'yes' if v['reliable'] else 'no'} | {v['per_participant_auc_diff_mean']:+.3f} | {v['wilcoxon_p']:.3f} |")
        for k, vv in cmp["eyes_add"].items():
            for n, v in vv.items():
                L.append(f"| {k}: {n} | {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['p_two_sided']:.3f} | (see §5) | – | {v['per_participant_auc_diff_mean']:+.3f} | {v['wilcoxon_p']:.3f} |")

tr = res.get("transfer")
L.append("\n**Idea 5: transfer between experiments** (train on all of one, test on all of the other; common ocular features; participant-cluster CI):\n")
if tr:
    L.append("| Direction / model | AUC [95% CI] | BA | EER | HTER | mean ppAUC | detected |\n|---|---|---|---|---|---|---|")
    for n, v in tr.items():
        t, c, p = v["trial"], v["trial_cluster_ci95"], v["per_participant"]
        L.append(f"| {n} | {t['auc']:.3f} [{c['auc'][0]:.3f}, {c['auc'][1]:.3f}] | {t['balanced_accuracy']:.3f} | {t['eer']:.3f} | {t['hter']:.3f} | {p['auc_mean']:.3f} | {p['detected_alpha05']}/{p['n_participants']} |")
else:
    L.append("PENDING")
L.append(narr.get("transfer", ""))

# ---------------- 5
L.append("\n## 5. Best ocular-only candidate vs published vs honest baseline\n")
for exp in ("pvc", "pvp"):
    b = res.get("best", {}).get(exp)
    if not b:
        L.append(f"**{exp}: PENDING**\n"); continue
    L.append(f"\n**{exp.upper()}** (metrics on repeat-averaged out-of-fold scores; CI = participant-cluster bootstrap, 2000 draws; edd row bootstrap in results.json)\n")
    L.append("| Model | AUC [CI] | BA [CI] | EER [CI] | HTER [CI] | MCC | mean ppAUC [CI] | Permutation p |\n|---|---|---|---|---|---|---|---|")
    for key in ("best_ocular_candidate", "honest_baseline"):
        x = b[key]; m = x["trial_metrics_repeat_avg_scores"]; c = x["trial_cluster_ci95"]
        perm = x.get("permutation_within_participant")
        ps = f"{perm['p_value']:.3f} (obs {perm['observed_trial_auc_repeat0_frozen']:.3f}, null {perm['null_mean']:.3f}, n={perm['n_perm']})" if perm else "–"
        L.append(f"| {x['name']} | {m['auc']:.3f} [{c['auc'][0]:.3f}, {c['auc'][1]:.3f}] | {m['balanced_accuracy']:.3f} [{c['balanced_accuracy'][0]:.3f}, {c['balanced_accuracy'][1]:.3f}] | "
                 f"{m['eer']:.3f} [{c['eer'][0]:.3f}, {c['eer'][1]:.3f}] | {m['hter']:.3f} [{c['hter'][0]:.3f}, {c['hter'][1]:.3f}] | {m['mcc']:.3f} | "
                 f"{x.get('per_participant_auc_mean_ci95') and ''}[{x['per_participant_auc_mean_ci95'][0]:.3f}, {x['per_participant_auc_mean_ci95'][1]:.3f}] | {ps} |")
    L.append("| Published | not reported (no classifier; effect sizes only, see §1) | | | | | | |")
    if fu and exp in fu:
        F = fu[exp]
        L.append(f"\nMechanism checks for `{F['best']}` (`followup.json`):\n")
        L.append("Feature-family ablation (same model family, nested tuning, same folds; Δ = pooled AUC vs the full candidate with a cluster-bootstrap CI):\n")
        L.append("| Variant | AUC (mean±SD) | mean ppAUC | ΔAUC vs full [CI] |\n|---|---|---|---|")
        for k, v in F["ablation"].items():
            dd = v["delta_vs_full"]
            L.append(f"| {k} | {v['auc']['mean']:.3f}±{v['auc']['sd']:.3f} | {v['pp_auc_mean']:.3f} | {dd['diff']:+.3f} [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}] |")
        imp = F["importance"]
        L.append(f"\nTop features ({imp['kind']}): " + ", ".join(f"{k} {v:+.3f}" for k, v in list(imp["top"].items())[:8]))
        L.append("\nIs it a bid-time proxy, and how much is between-person level?\n")
        L.append("| Model | Spearman(score, log bid time) | AUC within person-RT quartiles | pooled AUC | pooled AUC, person-centred scores | mean ppAUC |\n|---|---|---|---|---|---|")
        for n, v in F["proxy_base_rate"].items():
            L.append(f"| {n} | {v['spearman_score_vs_log_bid_time']:+.3f} | {v['auc_within_rt_quartiles_weighted']:.3f} | {v['pooled_auc']:.3f} | {v['pooled_auc_after_person_mean_centring_scores']:.3f} | {v['mean_per_participant_auc']:.3f} |")
        L.append(f"\nHolm across the 'eyes add' tests: pooled-AUC bootstrap {json.dumps({k: round(v, 4) for k, v in F['eyes_add_holm'].items()})}; per-participant Wilcoxon {json.dumps({k: round(v, 4) for k, v in F['eyes_add_wilcoxon_holm'].items()})}. "
                 f"Per-participant Wilcoxon vs baseline, Holm: {json.dumps({k: round(v, 4) for k, v in F['ocular_vs_baseline_wilcoxon_holm'].items()})}.")
        pc = F["per_participant_ci95"]
        L.append("\nPer-participant means with participant-bootstrap CIs: " + "; ".join(
            f"{n}: AUC {v['auc']['mean']:.3f} [{v['auc']['ci95'][0]:.3f}, {v['auc']['ci95'][1]:.3f}], BA {v['bacc']['mean']:.3f}, EER {v['eer']['mean']:.3f} [{v['eer']['ci95'][0]:.3f}, {v['eer']['ci95'][1]:.3f}], HTER {v['hter']['mean']:.3f} [{v['hter']['ci95'][0]:.3f}, {v['hter']['ci95'][1]:.3f}]"
            for n, v in pc.items()))
L.append("\n" + narr.get("why", "Mechanism narrative: PENDING"))

# ---------------- 6
L.append("\n## 6. Efficiency (this laptop, single-threaded CPU)\n")
hw = res.get("meta", {}).get("hardware")
if hw:
    L.append(f"Hardware: {hw['processor']}, {hw['cpu_count']} logical CPUs, {hw['platform']}, Python {hw['python']}. Unit of decision = one bid. The input is the bid's fixation and blink event list (no raw gaze samples are released), so sensor-rate FPS does not apply.\n")
eff = res.get("efficiency", {})
if eff:
    L.append("| Exp | Model | Latency/bid p50 / p95 (ms) | Latency/participant p50 / p95 (ms, n bids) | Throughput (bids/s) | Peak mem (KB) | Model size (B) | Params | Feature extraction/bid p50 / p95 (ms) |\n|---|---|---|---|---|---|---|---|---|")
    for exp, blk in eff.items():
        for n, e in blk.items():
            L.append(f"| {exp} | {n} | {e['latency_ms_per_bid']['p50']:.3f} / {e['latency_ms_per_bid']['p95']:.3f} | {e['latency_ms_per_participant']['p50']:.3f} / {e['latency_ms_per_participant']['p95']:.3f} ({e['bids_in_that_participant']}) | "
                     f"{e['throughput_bids_per_s']:,.0f} | {e['peak_memory_bytes_inference_participant'] / 1024:.1f} | {e['model_size_bytes']:,} | {e['parameter_count']:,} | {e['feature_extraction_ms_per_bid_from_events']['p50']:.3f} / {e['feature_extraction_ms_per_bid_from_events']['p95']:.3f} |")
else:
    L.append("PENDING")
L.append(narr.get("efficiency", ""))

# ---------------- 7
L.append("\n## 7. Recommendation\n")
L.append(narr.get("recommendation", "PENDING"))
L.append("\n## Files\n")
L.append(narr.get("files", ""))
(D / "comparison_card.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("card written", "FINAL" if (FINAL and complete) else "INTERIM", "complete" if complete else "incomplete")
