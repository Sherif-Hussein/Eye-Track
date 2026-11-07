# Phase 7 Track E card

Definitions: `PREREG.md`; deviations: `DEVIATIONS.md`. No model was refitted. No charts.

## E1 (like-for-like timing)

- published P4 SVM, 1000 Hz: original code p50 198.4 ms (p95 519.9), vectorised p50 66.54 ms (p95 144.69)
- published P4 SVM, 30 Hz: original code p50 116.5 ms (p95 209.2), vectorised p50 15.71 ms (p95 33.72)
- our pipeline, 1000 Hz: original code p50 241.5 ms (p95 424.2), vectorised p50 93.00 ms (p95 131.19)
- our pipeline, 30 Hz: original code p50 99.7 ms (p95 112.6), vectorised p50 3.13 ms (p95 3.98)
- total speed-up (published SVM, 1000 Hz, original code -> ours, 30 Hz, vectorised): 63.3x [54.6, 92.7] (bootstrap over rounds)
- Shapley factor rate: 5.45x [5.22, 6.20]
- Shapley factor code: 7.47x [7.00, 8.38]
- Shapley factor model: 1.56x [1.44, 1.78]
- edge rate: 1000 -> 30 at code=orig, model=pub: 1.70x [1.46, 2.51]
- edge rate: 1000 -> 30 at code=orig, model=ours: 2.42x [2.25, 3.36]
- edge rate: 1000 -> 30 at code=vec, model=pub: 4.24x [3.28, 4.54]
- edge rate: 1000 -> 30 at code=vec, model=ours: 29.67x [28.40, 33.39]
- edge code: orig -> vec at rate=1000, model=pub: 2.98x [2.70, 4.32]
- edge code: orig -> vec at rate=1000, model=ours: 2.60x [2.31, 3.27]
- edge code: orig -> vec at rate=30, model=pub: 7.41x [6.07, 7.79]
- edge code: orig -> vec at rate=30, model=ours: 31.81x [29.64, 32.98]
- edge model: pub -> ours at rate=1000, code=orig: 0.82x [0.57, 1.12]
- edge model: pub -> ours at rate=1000, code=vec: 0.72x [0.63, 0.76]
- edge model: pub -> ours at rate=30, code=orig: 1.17x [1.07, 1.39]
- edge model: pub -> ours at rate=30, code=vec: 5.01x [4.59, 6.60]
- microsaccade cost: ours 14 features / 13 features at 1000 Hz, orig code: 1.47x [1.32, 1.98]
- microsaccade cost: ours 14 features / 13 features at 1000 Hz, vec code: 2.02x [1.88, 2.32]
- replicate run1 p50 ms: pub|1000|orig 208.8, pub|1000|vec 70.1, pub|30|orig 126.6, pub|30|vec 16.4, ours|1000|orig 250.7, ours|1000|vec 117.1, ours|30|orig 160.2, ours|30|vec 4.3, ours13|1000|orig 239.8, ours13|1000|vec 78.3
- replicate run2 p50 ms: pub|1000|orig 342.4, pub|1000|vec 111.3, pub|30|orig 238.2, pub|30|vec 31.7, ours|1000|orig 504.9, ours|1000|vec 154.9, ours|30|orig 218.4, ours|30|vec 6.6, ours13|1000|orig 355.7, ours13|1000|vec 97.9

## E2 (conservative threshold; unit-level Clopper-Pearson 95% CIs)

- delmas_64_g1 (primary): nominal detection 97.8% [78.2%, 100.0%], FPR 8.9% [0.2%, 31.9%]; conservative: infeasible in every fold (m = 12 training negatives; bound at k = 0 is 22.1%), flags nobody (detection 0, FPR 0)
- delmas_64_g2 (primary): nominal detection 95.6% [68.1%, 99.8%], FPR 0.0% [0.0%, 21.8%]; conservative: infeasible in every fold (m = 12 training negatives; bound at k = 0 is 22.1%), flags nobody (detection 0, FPR 0)
- delmas_64_g3 (primary): nominal detection 86.7% [59.5%, 98.3%], FPR 0.0% [0.0%, 21.8%]; conservative: infeasible in every fold (m = 12 training negatives; bound at k = 0 is 22.1%), flags nobody (detection 0, FPR 0)
- delmas_32_g3 (primary): nominal detection 84.4% [59.5%, 98.3%], FPR 6.7% [0.2%, 31.9%]; conservative: infeasible in every fold (m = 12 training negatives; bound at k = 0 is 22.1%), flags nobody (detection 0, FPR 0)
- delmas_32_g4 (primary): nominal detection 26.7% [7.8%, 55.1%], FPR 4.4% [0.2%, 31.9%]; conservative: infeasible in every fold (m = 12 training negatives; bound at k = 0 is 22.1%), flags nobody (detection 0, FPR 0)
- delmas_32_g5 (primary): nominal detection 46.7% [21.3%, 73.4%], FPR 17.8% [4.3%, 48.1%]; conservative: infeasible in every fold (m = 12 training negatives; bound at k = 0 is 22.1%), flags nobody (detection 0, FPR 0)
- chen_unified (primary): nominal detection 57.7% [43.2%, 70.3%], FPR 7.4% [2.0%, 17.3%]; conservative (60/60 folds feasible, k* = 0.0): detection 36.9% [24.9%, 51.5%], FPR 1.5% [0.0%, 9.6%]
- celniak_unified (primary): nominal detection 8.3% [1.2%, 31.7%], FPR 8.8% [1.3%, 33.1%]; conservative: infeasible in every fold (m = 18 training negatives; bound at k = 0 is 15.3%), flags nobody (detection 0, FPR 0)
- celniak_ctrl_valcov (control): nominal detection 33.3% [15.4%, 59.2%], FPR 10.5% [1.3%, 33.1%]; conservative: infeasible in every fold (m = 18 training negatives; bound at k = 0 is 15.3%), flags nobody (detection 0, FPR 0)
- celniak_ref_valonly (control): nominal detection 56.7% [31.5%, 76.9%], FPR 8.8% [1.3%, 33.1%]; conservative: infeasible in every fold (m = 18 training negatives; bound at k = 0 is 15.3%), flags nobody (detection 0, FPR 0)
- preti_A_ocular (primary): nominal detection 17.3% [6.3%, 38.1%], FPR 10.0% [2.1%, 26.5%]; conservative: infeasible in every fold (m = 29 training negatives; bound at k = 0 is 9.8%), flags nobody (detection 0, FPR 0)
- preti_B_ocular_rt (primary): nominal detection 90.1% [70.8%, 97.6%], FPR 8.9% [2.1%, 26.5%]; conservative: infeasible in every fold (m = 29 training negatives; bound at k = 0 is 9.8%), flags nobody (detection 0, FPR 0)
- preti_A_contrast_only (secondary): nominal detection 19.8% [6.3%, 38.1%], FPR 12.2% [3.8%, 30.7%]; conservative: infeasible in every fold (m = 29 training negatives; bound at k = 0 is 9.8%), flags nobody (detection 0, FPR 0)
- preti_B_contrast_only (secondary): nominal detection 82.7% [61.9%, 93.7%], FPR 10.0% [2.1%, 26.5%]; conservative: infeasible in every fold (m = 29 training negatives; bound at k = 0 is 9.8%), flags nobody (detection 0, FPR 0)
- mazza_g2_vs_g3 (primary): nominal detection 27.4% [13.4%, 43.1%], FPR 9.5% [2.8%, 23.7%]; conservative: infeasible in every fold (m = 36 training negatives; bound at k = 0 is 8.0%), flags nobody (detection 0, FPR 0)
- mazza_guilty_vs_innocent (primary): nominal detection 28.5% [18.6%, 39.5%], FPR 9.1% [3.6%, 17.4%]; conservative (50/50 folds feasible, k* = 1.0): detection 14.4% [7.3%, 23.8%], FPR 4.1% [0.8%, 10.7%]

## E3 (Chen corrected repeated-CV intervals)

- unified_minus_rate_rule on detection: mean per-fold delta +0.139, corrected 95% CI [-0.061, +0.339] (naive [+0.043, +0.235]; across-repeat [+0.108, +0.170])
- unified_minus_rate_rule on person_auc: mean per-fold delta +0.093, corrected 95% CI [+0.007, +0.179] (naive [+0.052, +0.134]; across-repeat [+0.067, +0.118])
- unified_minus_rate_rule on person_auc_c1only: mean per-fold delta +0.093, corrected 95% CI [-0.009, +0.195] (naive [+0.044, +0.142]; across-repeat [+0.034, +0.152])
- I2b_minus_rate_rule on detection: mean per-fold delta +0.160, corrected 95% CI [-0.038, +0.358] (naive [+0.065, +0.255]; across-repeat [+0.102, +0.218])
- I2b_minus_rate_rule on person_auc: mean per-fold delta +0.110, corrected 95% CI [+0.013, +0.207] (naive [+0.063, +0.157]; across-repeat [+0.073, +0.147])
- I2b_minus_rate_rule on person_auc_c1only: mean per-fold delta +0.121, corrected 95% CI [+0.012, +0.231] (naive [+0.069, +0.174]; across-repeat [+0.073, +0.170])
- unified_minus_I2b on detection: mean per-fold delta -0.021, corrected 95% CI [-0.151, +0.108] (naive [-0.083, +0.041]; across-repeat [-0.100, +0.058])
- unified_minus_I2b on person_auc: mean per-fold delta -0.017, corrected 95% CI [-0.059, +0.025] (naive [-0.037, +0.003]; across-repeat [-0.041, +0.007])
- unified_minus_I2b on person_auc_c1only: mean per-fold delta -0.028, corrected 95% CI [-0.079, +0.022] (naive [-0.053, -0.004]; across-repeat [-0.042, -0.015])
