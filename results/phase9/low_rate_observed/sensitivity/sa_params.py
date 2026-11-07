"""Task 2.2c Part A (EXPLORATORY): reduced-slow-component jitter parameters.
Rule (card docs/task_observed_sensitivity_and_models.md): keep tau_fast, tau_slow and the fast component's ABSOLUTE variance as fitted in
measured/jitter_model.json; rescale only the slow component's variance so that the process's total long-run SD equals the measured residual SD
(x 2.0950, y 1.5262 deg). If the fast variance alone exceeds that, stop.
The 2.2b code builds the process as sigma*sqrt(w) (fast) + sigma*sqrt(1-w) (slow), so the new (sigma, w) = (target SD, fast variance / target variance).
Writes jitter_params_reduced.json in this folder only.
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LRO = HERE.parent
jm = json.loads((LRO / "measured" / "jitter_model.json").read_text(encoding="utf-8"))
out = {"rule": "fast variance, tau_fast, tau_slow kept; slow variance rescaled so total SD = measured residual SD"}
for ax in ("x", "y"):
    p = jm[ax]["params"]
    target = jm[ax]["measured"]["sigma_deg"]
    fast_var = p["share_fast"] * p["sigma_deg"] ** 2
    slow_var_old = (1 - p["share_fast"]) * p["sigma_deg"] ** 2
    t_var = target ** 2
    if fast_var >= t_var:
        print("STOP: fast variance", fast_var, "exceeds target variance", t_var, "on axis", ax)
        sys.exit(2)
    slow_var_new = t_var - fast_var
    new = {"sigma_deg": target, "share_fast": fast_var / t_var, "tau_fast_ms": p["tau_fast_ms"], "tau_slow_ms": p["tau_slow_ms"]}
    out[ax] = {"params": new, "old_params": p, "fast_var": fast_var, "fast_sd": fast_var ** 0.5, "slow_var_old": slow_var_old, "slow_sd_old": slow_var_old ** 0.5,
               "slow_var_new": slow_var_new, "slow_sd_new": slow_var_new ** 0.5, "slow_variance_ratio_new_over_old": slow_var_new / slow_var_old,
               "total_sd_old": p["sigma_deg"], "total_sd_new": target}
    print(ax, json.dumps(out[ax], indent=1))
(HERE / "jitter_params_reduced.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
