"""Task 2.2c Part B evaluation (EXPLORATORY). Calls results/phase10/model_compare/mc_eval.py evaluate() unchanged with the redirected folder, the two
conditions OBS and OBS_R and the models LR, XGB, ENS (see mco_lib.py). 2000 bootstrap resamples (seed 20261106), 2000 swaps (seed 20261107), repeat 0.
Writes results_item_person.csv/json into this folder. Holm family = XGB and ENS x OBS and OBS_R = 4 rows (item and person level separately).
Usage: .venv/Scripts/python.exe -I -B results/phase10/model_compare_observed/mco_eval.py
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mco_lib as O  # noqa: E402

df = O.EV.evaluate([0], 2000, 2000)
assert len(df[df.model != "LR"]) == 4, len(df)
