"""Phase 2 (celniak track), final stage: merge the takeover stage files into results.json.

Reads stage_perm_*.json, stage_confounds.json, stage_reproduction.json, stage_efficiency.json and writes them into
results.json under: <idea>.permutation(_within_order), confound_controls, reproduction_paper_protocol, efficiency.
A timestamped copy of the previous results.json is kept. Re-runnable.

Usage: python -I scripts/phase2_celniak_finalize.py
"""
import json
import shutil
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "phase2" / "celniak"


def main():
    rf = OUT / "results.json"
    r = json.loads(rf.read_text())
    shutil.copy(rf, OUT / f"results_before_finalize_{time.strftime('%H%M%S')}.json")
    for name, key in (("global_gaze_lr", "permutation_within_order"), ("orient_lr", "permutation")):
        f = OUT / f"stage_perm_{name}_{key}.json"
        if f.exists():
            s = json.loads(f.read_text())
            if s.get("done"):
                r[name][key] = s
    for fname, key in (("stage_confounds.json", "confound_controls"), ("stage_reproduction.json", "reproduction_paper_protocol"),
                       ("stage_efficiency.json", "efficiency"),
                       ("stage_reproduction_leak.json", "reproduction_random_slide_split_leaky")):
        if (OUT / fname).exists():
            r[key] = json.loads((OUT / fname).read_text())
    r["meta"]["status"] = "final"
    r["meta"]["finalized"] = time.strftime("%Y-%m-%d %H:%M")
    tmp = OUT / "results.json.tmp"
    tmp.write_text(json.dumps(r, indent=1, default=str))
    tmp.replace(rf)
    print("merged keys:", [k for k in ("confound_controls", "reproduction_paper_protocol", "efficiency") if k in r])


if __name__ == "__main__":
    main()
