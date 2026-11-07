"""Task M2 addendum: ensemble of logistic regression + XGBoost. EXPLORATORY. Development data only.
Source: ADDENDUM_ensemble.md (this folder); fold records written by mc_run.py (ck/<tag>/{LR,XGB}/<set>_r<repeat>.pkl). No model is re-trained.
Rule (fixed in the addendum): weights 50/50; within every outer fold each model's test scores and inner-fold (training) scores are z-scaled with the
mean and SD of THAT MODEL'S OWN inner-fold training scores (inner_oof); no test person influences the scale. Settings are those already chosen inside the training people.
NOT BLIND: the addendum was written after an initial look at the ensemble on two camera noise draws and the clean pass.
Usage: .venv/Scripts/python.exe -I results/phase10/model_compare/mc_ens.py [--repeats 0]
Writes ck/<tag>/ENS/<set>_r<repeat>.pkl in the same fold-record layout as the other models.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_lib as M  # noqa: E402

L = M.L
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402

CK = M.HERE / "ck"


def ens_records(rec_lr, rec_xgb):
    out = []
    for a, b in zip(rec_lr, rec_xgb):
        assert np.array_equal(a["tr"], b["tr"]) and np.array_equal(a["te"], b["te"])
        ma, sa = a["inner_oof"].mean(), a["inner_oof"].std() + 1e-12
        mb, sb = b["inner_oof"].mean(), b["inner_oof"].std() + 1e-12
        za_in, zb_in = (a["inner_oof"] - ma) / sa, (b["inner_oof"] - mb) / sb
        za_te, zb_te = (a["score"] - ma) / sa, (b["score"] - mb) / sb
        out.append({"tr": a["tr"], "te": a["te"], "score": 0.5 * za_te + 0.5 * zb_te, "inner_oof": 0.5 * za_in + 0.5 * zb_in,
                    "hp": (a["hp"], b["hp"]), "scale": {"LR": (float(ma), float(sa)), "XGB": (float(mb), float(sb))}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", default="0")
    a = ap.parse_args()
    for tag in sorted(p.name for p in CK.iterdir() if p.is_dir()):
        for sk in L.SETS:
            for r in [int(x) for x in a.repeats.split(",")]:
                dst = CK / tag / "ENS" / f"{sk}_r{r}.pkl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                lr = pickle.load(open(CK / tag / "LR" / f"{sk}_r{r}.pkl", "rb"))
                xg = pickle.load(open(CK / tag / "XGB" / f"{sk}_r{r}.pkl", "rb"))
                res = {r: ens_records(lr[r], xg[r])}
                tmp = Path(str(dst) + ".tmp")
                pickle.dump(res, open(tmp, "wb"))
                tmp.replace(dst)
        L.log("ensemble records written", tag)


if __name__ == "__main__":
    main()
