# Task M (model and feature search): stopped by the user, partial result

Stopped on 2026-11-06 at about 07:55 at the user's request, during the first condition (D30 = Delmas, clean, 30 Hz). Exploratory. Plan fixed before the full runs: PLAN.md (hash in PLAN.sha256.txt).
Reproduction checks passed before any candidate run: D30 0.78545 (max score difference 1.8e-15), webcam case at 30 Hz 0.7737, Chen 30 Hz 0.6016.

Completed: all 12 single candidates on 5 of the 6 Delmas group sets (64_g1, 64_g2, 64_g3, 32_g3, 32_g4); 32_g5 incomplete. Not run: the webcam condition (W15), Chen (C30), the nested ensembles (E9, E10), the learning curve and the more-displays analysis.
Evaluation of the completed part (`partial_D30_5sets/`, by ms_eval.py as planned): no candidate beats the frozen pipeline (bootstrap interval includes zero for every candidate; Holm p 0.87-1.00 by paired swap). Largest point gains: pairwise ranking +0.006 [-0.003, +0.017], catch22 + base +0.005 [-0.005, +0.016], conditional logit +0.005 [-0.003, +0.016]. Features used alone (scan path, PCA, catch22) and the tiny CNN were lower by 0.011-0.017.
The AUC here (0.815 for the frozen model) is over 5 group sets and is not comparable with the 0.785 over all six.
