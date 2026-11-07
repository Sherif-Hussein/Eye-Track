# Phase 6 Track B card: refit-aware permutation tests (B1) and Celniak best-of-grid null (B2)

Definitions, formulas and seeds: `PREREG.md`. p = (1 + #null >= observed) / (1 + N), one-sided as pre-registered; null ranges are 2.5-97.5 percentiles.

## B1

| Test | N | Baseline obs [null 95%] p_single | Ours obs [null 95%] p_single | Delta obs [null 95%] | refit-aware p (Holm) | Phase 4 bootstrap p (Holm) |
|---|---|---|---|---|---|---|
| delmas_noninferiority | 100 | 0.781 [0.479, 0.527] p=0.010 (low 1.000) | 0.777 [0.472, 0.535] p=0.010 (low 1.000) | -0.004 [-0.031, +0.041] | 0.1584 (0.8182) | 0.0385 (0.1924) |
| chen_detection_fpr7 | 125 | 0.446 [0.018, 0.161] p=0.008 (low 1.000) | 0.607 [0.002, 0.179] p=0.008 (low 1.000) | +0.161 [-0.123, +0.123] | 0.0317 (0.2222) | 0.0190 (0.1139) |
| celniak_participant_auc | 100 | 0.624 [0.314, 0.688] p=0.119 (low 0.901) | 0.426 [0.168, 0.672] p=0.604 (low 0.406) | -0.197 [-0.379, +0.145] | 0.9109 (1.0000) | 0.9710 (1.0000) |
| preti_A_participant_auc | 175 | 0.367 [0.270, 0.662] p=0.852 (low 0.153) | 0.486 [0.327, 0.686] p=0.534 (low 0.472) | +0.120 [-0.185, +0.188] | 0.1364 (0.8182) | 0.0065 (0.0455) |
| preti_B_participant_auc | 175 | 0.912 [0.262, 0.685] p=0.006 (low 1.000) | 0.956 [0.285, 0.683] p=0.006 (low 1.000) | +0.043 [-0.217, +0.213] | 0.4318 (1.0000) | 0.0625 (0.2499) |
| mazza_g2_vs_g3_participant_auc | 100 | 0.690 [0.310, 0.702] p=0.069 (low 0.950) | 0.601 [0.350, 0.675] p=0.208 (low 0.812) | -0.089 [-0.212, +0.206] | 0.8218 (1.0000) | 0.8901 (1.0000) |
| mazza_guilty_vs_innocent_participant_auc | 100 | 0.605 [0.387, 0.595] p=0.030 (low 0.980) | 0.664 [0.387, 0.602] p=0.010 (low 1.000) | +0.059 [-0.096, +0.118] | 0.1782 (0.8182) | 0.1169 (0.3508) |

Chen (detection rate at calibrated FPR <= 7%; N = 125): person AUC baseline 0.770 (p 0.008), ours 0.859 (p 0.008); person-AUC delta p 0.119.

Delmas non-inferiority: shift p_NI = 0.1584; delta - 1.645 x SD_null = -0.0338 (margin -0.02).

## B2 (Celniak best-of-216-grid null)

N = 50 shuffles. Null best-of-grid accuracy: mean 0.643, 95% range [0.544, 0.738], max 0.744.
Null best-of-grid AUC: mean 0.645, 95% range [0.515, 0.779], max 0.792.
Real reproduction: acc 0.718, AUC 0.671 / 0.721. p_acc = 0.1176; p_auc(0.721) = 0.1569; p_auc(0.671) = 0.3922.

## Records used (D3 + D5)

| Task | on disk | pre-sleep | excluded post-sleep | excluded partial block | N used | p resolution 1/(N+1) |
|---|---|---|---|---|---|---|
| delmas | 126 | 124 | 2 | 24 | 100 | 0.0099 |
| chen | 133 | 130 | 3 | 5 | 125 | 0.0079 |
| celniak | 106 | 106 | 0 | 6 | 100 | 0.0099 |
| preti | 196 | 193 | 3 | 18 | 175 | 0.0057 |
| mazza_g2_vs_g3 | 107 | 103 | 4 | 3 | 100 | 0.0099 |
| mazza_guilty_vs_innocent | 100 | 100 | 0 | 0 | 100 | 0.0099 |
| b2 | 77 | 76 | 1 | 26 | 50 | 0.0196 |

B2 supplement (all 76 pre-sleep shuffles, descriptive): p_acc 0.1039, p_auc(0.721) 0.1429, p_auc(0.671) 0.3247; null acc mean 0.633, null AUC mean 0.629.
