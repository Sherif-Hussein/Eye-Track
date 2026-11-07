"""Task 2.2b (EXPLORATORY), step 1: Delmas raw EDF -> 1000 Hz trial arrays (x, y, pupil), kept for the timing-based resampling.

Reuses UNCHANGED (imported from their files): scripts/phase6_trackC_c2.py load_halves (EDF -> 1000 Hz trials, the same cutting as
src/edd/delmas_raw.py load_session), as in results/phase9/low_rate/s1_extract.py. Output: cache1000/p<id>.npz (float32 x, y, p of shape
(64, 5500): 500 ms before to 5000 ms after display onset; images; group). Resumable.
Check (every participant): edd.delmas_raw.degrade(x, y, p, 30) of the arrays stored here equals the 30 Hz arrays of the earlier
cache (results/phase9/low_rate/cache/p<id>.npz, read only): max abs difference and NaN pattern -> repro_raw_reload.json.
"""
import importlib.util
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import lro_common as C  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

from edd import delmas_raw as dr  # noqa: E402
from edd.delmas import load_trials  # noqa: E402

spec = importlib.util.spec_from_file_location("trackC_c2", C.ROOT / "scripts" / "phase6_trackC_c2.py")
c2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c2)
BOOK64 = C.ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"


def main():
    C.CACHE1000.mkdir(exist_ok=True)
    book = load_trials(BOOK64)
    parts = {int(p): int(g) for p, g in book.groupby("participant").group.first().items()}
    rep = {}
    t0 = C.time.time()
    for i, (pid, g) in enumerate(parts.items()):
        path = C.CACHE1000 / f"p{pid:02d}.npz"
        if not path.exists():
            x, y, p, images, fix = c2.load_halves(pid)
            tmp = C.CACHE1000 / f"p{pid:02d}.tmp.npz"
            np.savez_compressed(tmp, x=x, y=y, p=p, images=images, group=g)
            tmp.replace(path)
            status = "extracted"
        else:
            status = "cached"
        z = np.load(path)
        old = np.load(C.LOWRATE / "cache" / f"p{pid:02d}.npz")
        mx, nan_same = 0.0, True
        for k in range(z["x"].shape[0]):
            a, b, c = dr.degrade(z["x"][k], z["y"][k], z["p"][k], 30)
            for u, v in ((a, old["x30"][k]), (b, old["y30"][k]), (c, old["p30"][k])):
                nan_same &= bool((np.isnan(u) == np.isnan(v)).all())
                if np.isfinite(u).any():
                    mx = max(mx, float(np.nanmax(np.abs(u - v))))
        rep[pid] = {"max_abs_diff_30hz": mx, "nan_pattern_identical": nan_same, "images_equal": bool((z["images"] == old["images"]).all())}
        C.log(f"[{i + 1}/{len(parts)}] participant {pid} {status} reload-check max diff {mx:.2e} nan_same {nan_same} ({C.time.time() - t0:.0f}s)")
    ok = all(v["max_abs_diff_30hz"] < 1e-9 and v["nan_pattern_identical"] and v["images_equal"] for v in rep.values())
    C.jdump(C.HERE / "repro_raw_reload.json", {"all_passed": ok, "n_participants": len(rep), "max_over_participants": max(v["max_abs_diff_30hz"] for v in rep.values()), "per_participant": rep})
    C.log("reload check passed" if ok else "RELOAD CHECK FAILED")
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
