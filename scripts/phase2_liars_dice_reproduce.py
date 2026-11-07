"""Phase 2 / liars_dice: reproduce the paper's statistics (Foucher & Huckauf, PsyArXiv 2026).

The paper fits lmer(Parameter ~ Honesty + (1|ParticipantID)) and reports F(1, N-1). Denominator df of N-1
imply the model was fitted on participant x honesty cell means, where the random-intercept model is
equivalent to a paired t-test (F = t^2, eta2p = F / (F + df)). We reproduce that, plus a trial-level
within-participant (demeaned) test as a second reading of the ambiguous description.

Usage: python -I scripts/phase2_liars_dice_reproduce.py <bids_csv> <out_json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Paper values: (lie M, lie SD, truth M, truth SD, F, df, p, eta2p); None = not reported.
PUBLISHED = {
    "pvc": {"bid_time": (3317, 2214, 2351, 1205, 17.9, 34, "<.001", .34),
            "fix_rate": (3.37, 0.51, 3.49, 0.45, 12.7, 34, ".001", .27),
            "fix_dur_mean": (262, 37, 256, 32, 4.1, 34, ".05", .11),
            "blink_rate": (None, None, None, None, None, None, "n.s.", None),
            "blink_dur_mean": (120, 35, 123, 37, 3.1, 34, ".09", .09)},
    "pvp": {"bid_time": (2109, 412, 1870, 366, 30.9, 41, "<.001", .43),
            "fix_rate": (2.11, 0.39, 2.21, 0.45, 7.2, 41, ".01", .15),
            "fix_dur_mean": (442, 108, 397, 90, 16.0, 41, "<.001", .28),
            "blink_rate": (0.75, 0.19, 0.79, 0.19, 4.3, 41, ".04", .09),
            "blink_dur_mean": (246, 32, 251, 33, 3.8, 41, ".06", .08),
            "rate_face": (0.76, 0.26, 0.88, 0.28, 17.0, 40, "<.001", .30),
            "dur_face": (None, None, None, None, None, None, "n.s.", None),
            "rate_dice": (None, None, None, None, None, None, "n.s.", None),
            "rate_out": (None, None, None, None, None, None, "n.s.", None),
            "dur_dice": (364, 118, 291, 91, 26.3, 41, "<.001", .39),
            "dur_out": (478, 195, 427, 190, 6.6, 41, ".01", .14)},
}
# Challenger (PvP): contest vs accept.
PUBLISHED_CHAL = {"bid_time": (2153, 586, 1898, 523, 18.8, 41, "<.001", .31),
                  "fix_rate": (1.93, 0.43, 2.16, 0.33, 17.6, 41, "<.001", .30),
                  "fix_dur_mean": (494, 145, 363, 103, 43.9, 41, "<.001", .52),
                  "blink_rate": (0.68, 0.23, 0.81, 0.18, 23.1, 41, "<.001", .36),
                  "blink_dur_mean": (244, 32, 257, 32, 6.7, 41, ".01", .14),
                  "rate_face": (1.34, 0.45, 0.90, 0.27, 90.2, 41, "<.001", .69)}


def paired(df, col, cond_col, pos, neg):
    m = df.groupby(["pid", cond_col])[col].mean().unstack().dropna(subset=[pos, neg])
    t, p = stats.ttest_rel(m[pos], m[neg])
    F, dfree = float(t ** 2), len(m) - 1
    return {"pos_M": float(m[pos].mean()), "pos_SD": float(m[pos].std()), "neg_M": float(m[neg].mean()),
            "neg_SD": float(m[neg].std()), "F": F, "df": dfree, "p": float(p), "eta2p": F / (F + dfree),
            "n_participants": len(m)}


def within_trial(df, col, cond):
    d = df[[col, "pid"]].assign(c=cond.astype(float)).dropna()
    yc = d[col] - d.groupby("pid")[col].transform("mean")
    xc = d.c - d.groupby("pid").c.transform("mean")
    beta = (xc * yc).sum() / (xc ** 2).sum()
    resid = yc - beta * xc
    dof = len(d) - d.pid.nunique() - 1
    se = np.sqrt((resid ** 2).sum() / dof / (xc ** 2).sum())
    return {"beta": float(beta), "t": float(beta / se), "dof": int(dof), "p": float(2 * stats.t.sf(abs(beta / se), dof))}


def main(bids_csv, out_json):
    df = pd.read_csv(bids_csv)
    df["H"] = np.where(df.y == 1, "Lie", "Truth")
    # Variant reading: rates averaged only over bids where the event occurred (blink file / AOI cells
    # only contain bids with >= 1 event), which is what a per-event table aggregated in R would give.
    df["blink_rate_cond"] = df.blink_rate.where(df.n_blink > 0)
    for a in ("face", "dice", "out"):
        df[f"rate_{a}_cond"] = df[f"rate_{a}"].where(df[f"rate_{a}"] > 0)
    for exp in PUBLISHED:
        PUBLISHED[exp]["blink_rate_cond"] = PUBLISHED[exp]["blink_rate"]
    PUBLISHED["pvp"]["rate_face_cond"] = PUBLISHED["pvp"]["rate_face"]
    PUBLISHED_CHAL["blink_rate_cond"] = PUBLISHED_CHAL["blink_rate"]
    PUBLISHED_CHAL["rate_face_cond"] = PUBLISHED_CHAL["rate_face"]
    out = {"bidder": {}, "challenger_pvp": {}}
    for exp, pub in PUBLISHED.items():
        b = df[(df.exp == exp) & (df.Role == "Bidder")]
        out["bidder"][exp] = {}
        for col, ref in pub.items():
            r = paired(b, col, "H", "Lie", "Truth")
            r["within_trial"] = within_trial(b, col, b.y)
            r["published"] = dict(zip(["lie_M", "lie_SD", "truth_M", "truth_SD", "F", "df", "p", "eta2p"], ref))
            # Same statistic with the released (un-recoded) labels, for the label audit.
            r["released_label_lie_M"] = r["neg_M"]
            out["bidder"][exp][col] = r
    c = df[(df.exp == "pvp") & (df.Role == "Challenger")]
    for col, ref in PUBLISHED_CHAL.items():
        r = paired(c, col, "ChalDecision", "Contest", "Accept")
        r["published"] = dict(zip(["contest_M", "contest_SD", "accept_M", "accept_SD", "F", "df", "p", "eta2p"], ref))
        out["challenger_pvp"][col] = r
    Path(out_json).write_text(json.dumps(out, indent=1))
    for k, v in out["bidder"].items():
        print("==", k)
        for col, r in v.items():
            P = r["published"]
            print(f"{col:15s} lie {r['pos_M']:8.2f}({r['pos_SD']:.2f}) truth {r['neg_M']:8.2f}({r['neg_SD']:.2f}) "
                  f"F({r['df']})={r['F']:.1f} p={r['p']:.3g} eta={r['eta2p']:.2f} | paper lie {P['lie_M']}({P['lie_SD']}) "
                  f"truth {P['truth_M']}({P['truth_SD']}) F={P['F']} p={P['p']} eta={P['eta2p']} | within-trial t={r['within_trial']['t']:.1f}")
    print("== challenger pvp")
    for col, r in out["challenger_pvp"].items():
        P = r["published"]
        print(f"{col:15s} contest {r['pos_M']:8.2f}({r['pos_SD']:.2f}) accept {r['neg_M']:8.2f}({r['neg_SD']:.2f}) F={r['F']:.1f} eta={r['eta2p']:.2f}"
              f" | paper contest {P['contest_M']} accept {P['accept_M']} F={P['F']} eta={P['eta2p']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
