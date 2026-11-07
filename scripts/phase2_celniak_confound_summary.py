"""Paired bootstrap of every confound-control variant vs unadjusted global_gaze_lr and vs validation-error alone
(participant bootstrap on repeat-averaged scores, 2000 draws), and the markdown table used in comparison_card.md.
Adds `paired` to stage_confounds.json and writes confound_table.md. Usage: python -I scripts/phase2_celniak_confound_summary.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd  # noqa: E402

import phase2_celniak_screen as S  # noqa: E402

OUT = ROOT / "results" / "phase2" / "celniak"


def main():
    st = json.loads((OUT / "stage_confounds.json").read_text())
    ref = S.avg_scores(pd.read_csv(OUT / "trials_global_gaze_lr.csv"))
    val = S.avg_scores(pd.read_csv(OUT / "trials_validation_accuracy_lr.csv"))
    res = json.loads((OUT / "results.json").read_text())
    base_auc = res["global_gaze_lr"]["participant"]["auc"]["mean"]
    paired, rows = {}, []
    hdr = "| Variant | AUC (mean ± SD, 3 repeats) | Repeat-avg AUC [95% CI] | Acc | EER | HTER | Order-stratified AUC | Δ vs global_gaze_lr [95% CI], p | Share of above-chance AUC kept |\n|---|---|---|---|---|---|---|---|---|"
    for name, v in st["variants"].items():
        a = S.avg_scores(pd.read_csv(OUT / f"trials_conf_{name}.csv"))
        paired[name] = {"vs_global_gaze_lr": S.paired_bootstrap(a, ref), "vs_validation_accuracy_lr": S.paired_bootstrap(a, val)}
    st["paired"] = paired
    (OUT / "stage_confounds.json").write_text(json.dumps(st, indent=1, default=str))
    def line(label, p, avg, ci, strat, d):
        keep = (p["auc"]["mean"] - 0.5) / (base_auc - 0.5)
        dd = f"{d['auc_diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}], p={d['p_two_sided']:.2f}" if d else "n/a"
        return (f"| {label} | {p['auc']['mean']:.3f} ± {p['auc']['sd']:.3f} | {avg:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                f"{p['accuracy']['mean']:.3f} | {p['eer']['mean']:.3f} | {p['hter']['mean']:.3f} | {strat:.3f} | {dd} | {keep:.0%} |")
    out = [hdr]
    g = res["global_gaze_lr"]
    out.append(line("global_gaze_lr (unadjusted)", g["participant"], g["participant_avg_scores"]["auc"], g["participant_ci95"]["auc"], g["order_stratified_auc"]["mean"], None))
    v = res["validation_accuracy_lr"]
    out.append(line("validation error alone (reference)", v["participant"], v["participant_avg_scores"]["auc"], v["participant_ci95"]["auc"], v["order_stratified_auc"]["mean"], paired["gg_val_cov"]["vs_global_gaze_lr"] and S.paired_bootstrap(val, ref)))
    for name, ev in st["variants"].items():
        out.append(line(name, ev["participant"], ev["participant_avg_scores"]["auc"], ev["participant_ci95"]["auc"], ev["order_stratified_auc"]["mean"], paired[name]["vs_global_gaze_lr"]))
    (OUT / "confound_table.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(json.dumps({k: {kk: [round(x, 3) if isinstance(x, float) else x for x in (vv['auc_diff'], vv['p_two_sided'])] for kk, vv in d.items()} for k, d in paired.items()}))


if __name__ == "__main__":
    main()
