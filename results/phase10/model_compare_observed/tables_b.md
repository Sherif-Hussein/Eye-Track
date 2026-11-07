**Item level**

| Condition | Model | Item AUC [95% CI] | Change from LR [95% CI] | Swap p (raw) | Holm p | Verdict | EER | HTER (7% threshold) |
|---|---|---|---|---|---|---|---|---|
| OBS | Logistic regression (reference) | 0.7237 [0.6919, 0.7561] | reference | -- | -- | reference | 0.319 | 0.370 |
| OBS | XGBoost | 0.7120 [0.6834, 0.7415] | -0.0116 [-0.0225, +0.0004] | 0.0645 | 0.1514 | no reliable difference | 0.335 | 0.368 |
| OBS | Ensemble (LR + XGBoost) | 0.7294 [0.6990, 0.7605] | +0.0057 [+0.0003, +0.0109] | 0.0505 | 0.1514 | no reliable difference | 0.323 | 0.358 |
| OBS_R | Logistic regression (reference) | 0.7462 [0.7109, 0.7814] | reference | -- | -- | reference | 0.299 | 0.344 |
| OBS_R | XGBoost | 0.7352 [0.7052, 0.7662] | -0.0109 [-0.0231, +0.0025] | 0.1254 | 0.1514 | no reliable difference | 0.310 | 0.341 |
| OBS_R | Ensemble (LR + XGBoost) | 0.7578 [0.7235, 0.7912] | +0.0116 [+0.0059, +0.0175] | 0.0005 | 0.0020 | better | 0.295 | 0.338 |

**Person level**

| Condition | Model | Person AUC [95% CI] | Change from LR [95% CI] | Swap p (raw) | Holm p | Verdict | EER | HTER (7% threshold) |
|---|---|---|---|---|---|---|---|---|
| OBS | Logistic regression (reference) | 0.8385 [0.7899, 0.8812] | reference | -- | -- | reference | 0.183 | 0.241 |
| OBS | XGBoost | 0.8025 [0.7526, 0.8464] | -0.0360 [-0.0664, -0.0044] | 0.0220 | 0.0880 | no reliable difference | 0.241 | 0.257 |
| OBS | Ensemble (LR + XGBoost) | 0.8249 [0.7751, 0.8682] | -0.0136 [-0.0324, +0.0037] | 0.0790 | 0.2369 | no reliable difference | 0.217 | 0.250 |
| OBS_R | Logistic regression (reference) | 0.8607 [0.8099, 0.9057] | reference | -- | -- | reference | 0.176 | 0.213 |
| OBS_R | XGBoost | 0.8469 [0.7958, 0.8933] | -0.0138 [-0.0398, +0.0151] | 0.3068 | 0.6137 | no reliable difference | 0.196 | 0.226 |
| OBS_R | Ensemble (LR + XGBoost) | 0.8620 [0.8128, 0.9082] | +0.0012 [-0.0133, +0.0160] | 0.8586 | 0.8586 | no reliable difference | 0.183 | 0.202 |

**Extra (labelled): ensemble minus XGBoost, paired interval only**

| Condition | Item: ensemble minus XGBoost [95% CI] | Person: ensemble minus XGBoost [95% CI] |
|---|---|---|
| OBS | +0.0174 [+0.0085, +0.0250] | +0.0225 [+0.0030, +0.0417] |
| OBS_R | +0.0225 [+0.0122, +0.0318] | +0.0151 [-0.0047, +0.0328] |
