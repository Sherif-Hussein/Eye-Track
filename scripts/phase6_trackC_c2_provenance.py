"""Phase 6, Track C, C2 (part 1): where do r = 0.77 (earlier figure) and r = 0.68 (reviewer) come from?
Label-free re-computation of the I-VT vs EyeLink-parser fixation-count agreement from the stored per-face table
(data/derived/delmas_raw_face_trials.csv.gz: `el_nfix` = parser count per face and trial at 1000 Hz, `n_fix` = our I-VT count
at each rate) under every plausible aggregation. Writes results/phase6/trackC/c2_provenance.csv.
Usage: .venv/Scripts/python.exe -I scripts/phase6_trackC_c2_provenance.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from edd.delmas import load_trials  # noqa: E402

OUT = ROOT / "results" / "phase6" / "trackC"
RATES = [1000, 500, 250, 120, 60, 30]


def pr(a, b, kind="pearson"):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[ok], np.asarray(b)[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(spearmanr(a, b)[0]) if kind == "spearman" else float(np.corrcoef(a, b)[0, 1])


def main():
    f = pd.read_csv(ROOT / "data" / "derived" / "delmas_raw_face_trials.csv.gz")
    el = f[f.rate == 1000][["participant", "trial", "quad", "image", "fam_trial", "el_nfix"]]
    rows = []

    def add(name, unit, fn):
        r = {"definition": name, "unit": unit}
        for rate in RATES:
            r[str(rate)] = fn(rate)
        rows.append(r)

    mine = {rate: f[f.rate == rate][["participant", "trial", "quad", "n_fix", "n_visit"]] for rate in RATES}
    J = {rate: el.merge(mine[rate], on=["participant", "trial", "quad"]) for rate in RATES}
    add("A. trial x face, Pearson (n=19200)", "participant, trial, face", lambda r: pr(J[r].el_nfix, J[r].n_fix))
    add("A2. trial x face, Spearman", "participant, trial, face", lambda r: pr(J[r].el_nfix, J[r].n_fix, "spearman"))

    def part_image(r, how):
        g = J[r].groupby(["participant", "image"])[["el_nfix", "n_fix"]]
        g = g.mean() if how == "mean" else g.sum()
        return g.reset_index()
    add("B. participant x image, mean over presentations", "participant, image", lambda r: pr(*part_image(r, "mean")[["el_nfix", "n_fix"]].T.values))
    add("B2. participant x image, Spearman", "participant, image", lambda r: pr(*part_image(r, "mean")[["el_nfix", "n_fix"]].T.values, "spearman"))

    def image_level(r):
        g = part_image(r, "mean").groupby("image")[["el_nfix", "n_fix"]].mean()
        return pr(g.el_nfix, g.n_fix)
    add("C. image level (mean over participants)", "image", image_level)

    def part_level(r):
        g = J[r].groupby("participant")[["el_nfix", "n_fix"]].mean()
        return pr(g.el_nfix, g.n_fix)
    add("D. participant level (mean over trials and faces)", "participant", part_level)

    def within(r):
        d = J[r].copy()
        d[["el_nfix", "n_fix"]] = d[["el_nfix", "n_fix"]] - d.groupby("participant")[["el_nfix", "n_fix"]].transform("mean")
        return pr(d.el_nfix, d.n_fix)
    add("E. trial x face, participant means removed", "participant, trial, face", within)

    def trial_total(r):
        g = J[r].groupby(["participant", "trial"])[["el_nfix", "n_fix"]].sum()
        return pr(g.el_nfix, g.n_fix)
    add("F. trial level, sum over the 4 faces", "participant, trial", trial_total)

    def nonzero(r):
        d = J[r]
        d = d[d.el_nfix > 0]
        return pr(d.el_nfix, d.n_fix)
    add("G. trial x face, cells with parser count > 0", "participant, trial, face", nonzero)

    # published-table style aggregation (label-dependent presentation rule: famous faces all presentations, unknown
    # faces only in trials without a famous face); counts summed over the counted presentations
    book = {"64": load_trials(ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"), "32": load_trials(ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx")}

    def published_rule(r, s, mean_ivt=False):
        d = J[r]
        d = d if s == "64" else d[d.trial >= 32]
        d = d[d.participant.isin(book[s].participant.unique())]
        pub = d[(d.image >= 57) | (d.fam_trial == 0)]
        g = pub.groupby(["participant", "image"]).agg(el=("el_nfix", "sum"), nf=("n_fix", "sum")).reset_index()
        return pr(g.el, g.nf)
    add("H. published-table rule (label-dependent presentations), sums, 64-trial set", "participant, image", lambda r: published_rule(r, "64"))
    add("H2. same, 32-trial set", "participant, image", lambda r: published_rule(r, "32"))

    def build_tables_style(r, s):
        """track `build_tables`: per-presentation MEAN of our n_fix over all presentations vs parser SUM over all presentations."""
        d = J[r]
        d = d if s == "64" else d[d.trial >= 32]
        d = d[d.participant.isin(book[s].participant.unique())]
        g = d.groupby(["participant", "image"]).agg(el_sum=("el_nfix", "sum"), nf_mean=("n_fix", "mean")).reset_index()
        return pr(g.el_sum, g.nf_mean)
    add("I. our mean per presentation vs parser SUM over presentations (scale mismatch), 64-trial set", "participant, image", lambda r: build_tables_style(r, "64"))
    add("I2. same, 32-trial set", "participant, image", lambda r: build_tables_style(r, "32"))

    def svs(r, s):
        d = J[r]
        d = d if s == "64" else d[d.trial >= 32]
        d = d[d.participant.isin(book[s].participant.unique())]
        g = d.groupby(["participant", "image"]).agg(el=("el_nfix", "mean"), nf=("n_fix", "mean")).reset_index()
        return pr(g.el, g.nf)
    add("J. participant x image mean, 64-trial rows only (as B)", "participant, image", lambda r: svs(r, "64"))
    add("J2. participant x image mean, last 32 trials", "participant, image", lambda r: svs(r, "32"))

    # visits against parser counts (an I-VT visit merges consecutive same-face fixations)
    add("K. trial x face, I-VT n_visit vs parser count", "participant, trial, face", lambda r: pr(J[r].el_nfix, J[r].n_visit))
    # rate-to-rate: I-VT at rate r vs I-VT at 1000 Hz
    base = mine[1000].rename(columns={"n_fix": "n1000"})
    add("L. trial x face, I-VT count at rate vs I-VT count at 1000 Hz", "participant, trial, face",
        lambda r: pr(*mine[r].merge(base, on=["participant", "trial", "quad"])[["n_fix", "n1000"]].T.values))
    # count ratio
    add("M. mean I-VT count / mean parser count", "ratio", lambda r: float(J[r].n_fix.mean() / J[r].el_nfix.mean()))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "c2_provenance.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 100)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
