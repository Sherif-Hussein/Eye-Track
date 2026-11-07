"""Phase 2 / liars_dice: build the bid-level table from the released Rda files and audit the labels.

Usage: python -I scripts/phase2_liars_dice_build.py <data_dir> <out_csv> <audit_json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edd.liars_dice import EXPS, KEYS, _read_rda, build_bids  # noqa: E402


def label_audit(data_dir):
    """Game rule: a contested bid is won by the bidder iff it is truthful. Cross-tabulate the released
    BidHonesty against the outcome of contested bids."""
    out = {}
    for exp, stem in EXPS.items():
        fx = _read_rda(f"{data_dir}/{stem}_fixations.Rda")
        b = fx[fx.Role == "Bidder"].drop_duplicates(KEYS)
        c = b[b.LastBid]
        tab = pd.crosstab(c.BidHonesty, c.Outcome)
        out[exp] = {"contested_bids_released_label_x_bidder_outcome": tab.to_dict(),
                    "released_Lie_won_frac": float((c[c.BidHonesty == "Lie"].Outcome == "Success").mean()),
                    "released_Truth_won_frac": float((c[c.BidHonesty == "Truth"].Outcome == "Success").mean())}
        bl = _read_rda(f"{data_dir}/{stem}_blinks.Rda")
        m = b[KEYS + ["BidTime"]].merge(bl.drop_duplicates(KEYS)[KEYS + ["BidTime"]], on=KEYS, suffixes=("_fix", "_blink"))
        d = (m.BidTime_fix - m.BidTime_blink)
        out[exp]["bidtime_fix_vs_blink"] = {"n_matched_bidder_bids": int(len(m)), "n_diff_gt_1ms": int((d.abs() > 1).sum()),
                                            "median_abs_diff_ms": float(d.abs()[d.abs() > 1].median()) if (d.abs() > 1).any() else 0.0,
                                            "examples": m[d.abs() > 1].head(5).to_dict("records")}
    return out


def main(data_dir, out_csv, audit_json):
    audit = label_audit(data_dir)
    df = build_bids(data_dir)
    b = df[df.Role == "Bidder"]
    audit["bids"] = {exp: {"bidder_bids": int(len(t)), "participants": int(t.pid.nunique()),
                           "deceptive_frac_after_recode": float(t.y.mean()),
                           "bids_without_fixations": int((t.n_fix == 0).sum()),
                           "deceptive_frac_by_bid_pos": t.groupby("Bid").y.mean().round(3).to_dict(),
                           "per_participant_n_bids": t.groupby("pid").size().describe().round(1).to_dict(),
                           "per_participant_deceptive_frac": t.groupby("pid").y.mean().describe().round(3).to_dict()}
                     for exp, t in b.groupby("exp")}
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    Path(audit_json).write_text(json.dumps(audit, indent=1, default=str))
    print(json.dumps(audit, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main(*sys.argv[1:4])
