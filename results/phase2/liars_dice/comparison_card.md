# liars_dice comparison card (Phase 2 screening)

**Read this first (all numbers are in the sections below; AUC, EER and HTER are trial-level on repeat-averaged out-of-fold scores unless stated).**

1. **The released labels are inverted.** `BidHonesty = "Lie"` marks bids the bidder won when contested (1057/1059 in PvC, 936/998 in PvP), so deceptive = released `"Truth"`. We recoded it (§2, `data_audit.json`). The authors should be told.
2. **The data licence is unclear.** The OSF node is private (view-only link), carries no licence, and the CC-BY 4.0 applies to the PDF only. **This dataset should be used in the paper only if the authors give permission. That is the user's decision, not ours.**
3. **Eye-only models beat the paper-effect classifier on pooled trial AUC, but the absolute level is weak.** Participant-disjoint 5-fold, 3 repeats, nested tuning, Holm across the ocular ideas, participant-cluster bootstrap CIs:
   - PvC (35 players, 5018 bids): `I3_lgbm_ocular_online_raw` AUC 0.617 [0.578, 0.668], EER 0.422, HTER 0.419, against `I1_paper_ocular` AUC 0.522 [0.490, 0.554]. ΔAUC +0.096 [+0.035, +0.163], Holm p < .001 (7 comparisons; the bootstrap p was 0 in 2000 draws).
   - PvP (42 players in 21 dyads, 4112 bids): `I3_lr_ocular_online` AUC 0.602 [0.568, 0.635], EER 0.419, HTER 0.420, against 0.550 [0.517, 0.582]. ΔAUC +0.052 [+0.018, +0.086], Holm p = .027 (10 comparisons).
   - Weaker evidence on a within-person measure: Wilcoxon on per-participant AUCs gives Holm p = .106 (PvC) and .180 (PvP), so the gain is not shown to hold person by person.
   - Label-permutation p = .020 in both (50 within-participant shuffles; this is the smallest p that 50 permutations can give).
4. **Game context dominates, and the eyes add little beyond context + bid time.** Context alone gives AUC 0.753 (PvC) and 0.777 (PvP); context + bid time 0.789 and 0.804; the best fusion 0.793 and 0.809. Adding the eyes to context + RT changes AUC by +0.005 [-0.004, +0.014] (PvC) and +0.003 [-0.006, +0.013] (PvP), not reliable. Adding the eyes to context alone gives +0.030 and +0.027 (Holm p ≤ .003), which is largely the bid-time information (below).
5. **Eye-only models are partly a bid-time proxy.** Bid time alone gives AUC 0.613 (PvC) and 0.595 (PvP), the same as the best eye-only model; ocular + RT versus RT is not reliable (+0.017 and +0.019).
6. **Cross-setup transfer is near chance to weak:** AUC 0.519–0.558 (PvC to PvP) and 0.547–0.598 (PvP to PvC); learned models do not transfer better than the fixed paper signs.
7. **Efficiency is not a constraint:** 3.7 ms (PvC LightGBM) and 4.1 ms (PvP logistic regression) per bid on one CPU thread, under 150 KB peak memory, model size 0.4 MB or less.

## 1. The published study

- **Citation:** Foucher, V., & Huckauf, A. (2026). *Liar's Dice: Eye Measures in a Social Deceptive Game.* PsyArXiv preprint v1, posted 2026-09-02. https://doi.org/10.31234/osf.io/q5ngh_v1 (OSF q5ngh; Ulm University; EU Horizon Europe grant 101072410). As of 2026-11-01 we found no peer-reviewed version. A matching TeaP 2026 conference abstract exists ("Liar's Dice: Eye Movements of Deceivers and Challengers").
- **Task:** Liar's Dice for two players with two dice each and a fixed ranking (doubles highest, then ordered pairs). The bidder rolls unseen and bids truthfully or deceptively, and the challenger accepts or contests. A contested bid wins for the bidder only if it is true; an accepted bid swaps roles and forces a higher bid (escalation). Deception is **spontaneous**, not instructed. Label unit = one bid.
- **Exp 1, Player vs Computer (PvC):** 36 recruited and 35 analysed (25 F/11 M overall, M=25.2 years). EyeLink 1000 at **1000 Hz** with chinrest, 110 cm, 2560×1440 screen. 3 sessions × 45 rounds. The computer follows a published rule-based strategy (Appendix B), and participants won 51.8% of rounds.
- **Exp 2, Player vs Player (PvP):** 42 women in 21 dyads of strangers (M=23.2 years). Pupil Labs Neon glasses at **200 Hz**, no calibration. 3 sessions × 31 rounds. Events (roll, bid, challenge) were hand-labelled from scene video and audio. AOIs: opponent face (Face Mapper), own dice (Marker Mapper), out.
- **Preprocessing:** EyeLink Data Viewer or Pupil Labs event classification. Fixations under 60 ms or longer than the bid were dropped, blinks outside 60–700 ms were dropped, and off-screen fixations were dropped (Exp 1). Bid time runs from dice display (Exp 1) or dice stop (Exp 2) to the click or the announced bid.
- **Features:** bid time; fixation rate and mean duration; blink rate and mean duration; in Exp 2 also per-AOI fixation rate and duration. Rates are count / bid duration.
- **Model and evaluation:** `lmer(Parameter ~ Honesty + (1|ParticipantID))` (and `~ Honesty*AOI`), Satterthwaite F, partial η², and Lenth post-hoc power. **No classifier, no held-out evaluation, and no accuracy, AUC, EER or HTER: not reported.**
- **Reported results (bidder, lie vs truth; M (SD); F, η²p):**

| Measure | Exp 1 lie / truth | Exp 1 F(1,34), η²p | Exp 2 lie / truth | Exp 2 F(1,41), η²p |
|---|---|---|---|---|
| Bid time (ms) | 3317 (2214) / 2351 (1205) | 17.9, .34 (p<.001) | 2109 (412) / 1870 (366) | 30.9, .43 (p<.001) |
| Fixation rate (/s) | 3.37 (.51) / 3.49 (.45) | 12.7, .27 (p=.001) | 2.11 (.39) / 2.21 (.45) | 7.2, .15 (p=.01) |
| Fixation duration (ms) | 262 (37) / 256 (32) | 4.1, .11 (p=.05) | 442 (108) / 397 (90) | 16.0, .28 (p<.001) |
| Blink rate (/s) | n.s., values not reported | – | .75 (.19) / .79 (.19) | 4.3, .09 (p=.04) |
| Blink duration (ms) | 120 (35) / 123 (37) | 3.1, .09 (p=.09) | 246 (32) / 251 (33) | 3.8, .08 (p=.06) |
| Face fixation rate (/s) | – | – | .76 (.26) / .88 (.28) | F(1,40)=17.0, .30 (p<.001) |
| Face fixation duration | – | – | n.s. | – |
| Dice / out fixation rate | – | – | n.s. | – |
| Dice fixation duration (ms) | – | – | 364 (118) / 291 (91) | 26.3, .39 (p<.001) |
| Out fixation duration (ms) | – | – | 478 (195) / 427 (190) | 6.6, .14 (p=.01) |

Exp 2 AOI main effects: fixation rate F(2,123)=13.9, η²p=.18; fixation duration F(2,82)=25.4, η²p=.38. Challenger, contest vs accept: time 2153 vs 1898 ms (F=18.8, .31); fixation rate 1.93 vs 2.16 (17.6, .30); fixation duration 494 vs 363 ms (43.9, .52); blink rate .68 vs .81 (23.1, .36); blink duration 244 vs 257 ms (6.7, .14); face fixation rate 1.34 vs 0.90 (90.2, .69). Post-hoc power was .68–.998 (Appendix C).

- **Data and licence: rights are unclear, so ask the authors.** The data are on OSF node `wbj83`. The node is **private** (`public: false`) and is reached through a view-only link, which the preprint lists as its data link. It has **no licence** (`node_license: null`). The CC-BY 4.0 licence applies to the preprint PDF only. The files are 4 Rda tables (fixations and blinks × PvC/PvP), Variables.pdf, and the PsychoPy task (`PlayerComputer.zip`; it asks users to cite the publication). **There is no analysis code.** The team must get the authors' permission before redistributing the data or publishing results on it. We also need to tell them about the label inversion below.

## 2. Reproduction under their protocol (`reproduction.json`)

- **The released labels are inverted.** In the Rda files, contested bids labelled `Lie` were won by the bidder in **1057/1059 (PvC)** and **936/998 (PvP)** cases, and contested `Truth` bids in 8/881 and 25/1019. Under the rules a contested bid is won only if it is true. With swapped labels the paper's descriptives are reproduced almost exactly (table below); with the released coding every effect has the opposite sign. Swapping also restores the escalation the paper describes: P(lie) by bid position is .20/.50/.67/.71 in PvC and .09/.49/.65/.69 in PvP. **We use deceptive = released `"Truth"`** (`data_audit.json`). The remaining 0.9% (PvC) and 4.3% (PvP) of contested bids disagree under either coding, which suggests label noise (PvP events were hand-coded).
- The df in the paper (F(1, N−1)) imply the lmer was fitted on participant × honesty cell means. With one observation per cell that is a paired t-test (F = t²), which is what we report. A trial-level within-person test gives the same signs with larger t (`reproduction.json`).
- **Blink rate and AOI rates** in the paper are averaged only over bids in which the event occurred: bids without blinks, or without face fixations, are absent from the event tables. With that reading, PvP blink rate is .75/.79 (exact) and face rate is .78/.89 (paper .76/.88). The `_cond` rows below use this reading. As true rates including zeros, the PvP blink effect disappears (F=0.7).
- Under the conditional reading, PvC blink rate is *significant* (F=11.1) while the paper reports n.s. The ambiguity is unresolved: the PvC blink file has its own BidTime, which differs from the fixation file in 1207/1886 bids (median +1.0 s). The fixation-file BidTime reproduces the paper's bid times, so we use it.
- Overall: the means for bid time, fixation rate and duration, and the AOI durations match within a few units, and the F values are within about ±2 of the paper's. Residual differences come from undocumented exclusions (e.g. PvC blink duration uses 31 of 35 participants).

| Exp | Measure | Ours lie / truth (SD) | Ours F, η²p | Paper lie / truth | Paper F, η²p |
|---|---|---|---|---|---|
| pvc | bid_time | 3290.80 (2167.90) / 2342.45 (1204.65) | F(34)=18.7, 0.35 (p=0.000128) | 3317 / 2351 | 17.9, 0.34 |
| pvc | fix_rate | 3.38 (0.48) / 3.48 (0.45) | F(34)=10.7, 0.24 (p=0.00251) | 3.37 / 3.49 | 12.7, 0.27 |
| pvc | fix_dur_mean | 263.30 (35.82) / 257.26 (31.09) | F(34)=4.0, 0.11 (p=0.0524) | 262 / 256 | 4.1, 0.11 |
| pvc | blink_rate | 0.23 (0.21) / 0.25 (0.20) | F(34)=1.7, 0.05 (p=0.206) | n.s. (no values) | n.s. |
| pvc | blink_dur_mean | 119.29 (33.96) / 122.48 (34.34) | F(30)=2.4, 0.07 (p=0.132) | 120 / 123 | 3.1, 0.09 |
| pvc | blink_rate_cond | 0.54 (0.22) / 0.64 (0.19) | F(30)=11.1, 0.27 (p=0.00231) | n.s. (no values) | n.s. |
| pvp | bid_time | 2091.84 (402.87) / 1865.70 (364.83) | F(41)=28.7, 0.41 (p=3.53e-06) | 2109 / 1870 | 30.9, 0.43 |
| pvp | fix_rate | 2.08 (0.40) / 2.21 (0.46) | F(41)=9.7, 0.19 (p=0.0033) | 2.11 / 2.21 | 7.2, 0.15 |
| pvp | fix_dur_mean | 443.30 (108.74) / 397.60 (89.30) | F(41)=15.4, 0.27 (p=0.000321) | 442 / 397 | 16.0, 0.28 |
| pvp | blink_rate | 0.60 (0.24) / 0.61 (0.25) | F(41)=0.7, 0.02 (p=0.423) | 0.75 / 0.79 | 4.3, 0.09 |
| pvp | blink_dur_mean | 246.31 (32.33) / 250.62 (33.25) | F(41)=2.9, 0.07 (p=0.095) | 246 / 251 | 3.8, 0.08 |
| pvp | rate_face | 0.41 (0.32) / 0.52 (0.37) | F(41)=12.5, 0.23 (p=0.00104) | 0.76 / 0.88 | 17.0, 0.3 |
| pvp | dur_face | 601.13 (241.12) / 555.99 (197.06) | F(40)=2.1, 0.05 (p=0.153) | n.s. (no values) | n.s. |
| pvp | rate_dice | 0.98 (0.46) / 1.00 (0.56) | F(41)=0.2, 0.01 (p=0.647) | n.s. (no values) | n.s. |
| pvp | rate_out | 0.69 (0.34) / 0.69 (0.33) | F(41)=0.0, 0.00 (p=0.831) | n.s. (no values) | n.s. |
| pvp | dur_dice | 370.95 (124.36) / 290.57 (90.63) | F(41)=28.1, 0.41 (p=4.3e-06) | 364 / 291 | 26.3, 0.39 |
| pvp | dur_out | 470.88 (195.48) / 421.08 (165.56) | F(41)=7.6, 0.16 (p=0.00885) | 478 / 427 | 6.6, 0.14 |
| pvp | blink_rate_cond | 0.75 (0.19) / 0.79 (0.19) | F(41)=6.1, 0.13 (p=0.0176) | 0.75 / 0.79 | 4.3, 0.09 |
| pvp | rate_face_cond | 0.78 (0.26) / 0.89 (0.28) | F(40)=15.8, 0.28 (p=0.000289) | 0.76 / 0.88 | 17.0, 0.3 |

Challenger (PvP), contest vs accept:

| Measure | Ours contest / accept | Ours F, η²p | Paper contest / accept | Paper F, η²p |
|---|---|---|---|---|
| bid_time | 2162.31 / 1900.34 | 18.1, 0.31 | 2153 / 1898 | 18.8, 0.31 |
| fix_rate | 1.92 / 2.16 | 17.3, 0.30 | 1.93 / 2.16 | 17.6, 0.3 |
| fix_dur_mean | 497.42 / 361.98 | 44.7, 0.52 | 494 / 363 | 43.9, 0.52 |
| blink_rate | 0.31 / 0.39 | 4.8, 0.10 | 0.68 / 0.81 | 23.1, 0.36 |
| blink_dur_mean | 244.21 / 255.87 | 5.8, 0.12 | 244 / 257 | 6.7, 0.14 |
| rate_face | 1.13 / 0.40 | 125.7, 0.75 | 1.34 / 0.9 | 90.2, 0.69 |
| blink_rate_cond | 0.69 / 0.80 | 19.0, 0.32 | 0.68 / 0.81 | 23.1, 0.36 |
| rate_face_cond | 1.34 / 0.91 | 88.9, 0.68 | 1.34 / 0.9 | 90.2, 0.69 |

## 3–4. Honest baseline and ideas screened (our protocol)

Protocol:
- Outer folds are participant-disjoint 5-fold, 3 repeats (seeds 2026–2028), frozen in `folds.json`. **PvP folds are dyad-disjoint**: both players of a dyad are always in the same fold. The dyads (1,2), (3,4), … were verified from bid/role alignment.
- Tuning is nested `StratifiedGroupKFold(3)` on the training participants only, with class-balanced models and threshold 0.
- Only bidder bids are modelled: 5018 bids from 35 PvC players, and 4112 bids from 42 PvP players. P(lie) is .40 (PvC) and .33 (PvP).
- Feature blocks:
  - **Ocular:** time-normalised fixation and blink features (rate, duration statistics, first-fixation duration, fixation-time fraction, blink presence), plus AOI rates, durations, dwell, face share and switch rate in PvP. Raw counts are excluded because they re-encode bid time.
  - **RT:** log bid time.
  - **Context:** information available when the bid is made: bid position, own bid index, who opened, session, round, previous-round outcome, running win rate. Post-hoc fields are excluded: LastBid, outcome, the current-round winner and the challenger's decision.
- **Online** = each feature minus the player's running mean over their *earlier* bids, shrunk toward the training-population mean with k pseudo-bids; label-free and causal. **TRANSDUCTIVE** = person-centred using all of that player's bids; reference only, not deployable.
- "AUC within bid-pos" = AUC computed inside bid-position strata (1, 2, ≥3), n-weighted. It shows signal not explained by game state.

**PVC** — trial level, mean ± SD over 3 repeats (threshold 0). The honest baseline is `I1_paper_ocular`.

| Idea | Block | AUC | balanced_accuracy | accuracy | sensitivity | specificity | precision | f1 | MCC | EER | HTER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I1_paper_ocular | ocular | 0.522±0.003 | 0.527±0.003 | 0.530±0.002 | 0.511±0.008 | 0.543±0.002 | 0.425±0.003 | 0.464±0.005 | 0.053±0.006 | 0.474±0.002 | 0.473±0.003 |
| I1_paper_ocular_rt | ocular+rt | 0.588±0.003 | 0.560±0.002 | 0.562±0.001 | 0.551±0.005 | 0.569±0.002 | 0.458±0.001 | 0.501±0.003 | 0.118±0.003 | 0.439±0.003 | 0.440±0.002 |
| I1_paper_ocular_online | ocular | 0.540±0.001 | 0.534±0.001 | 0.538±0.001 | 0.513±0.002 | 0.555±0.001 | 0.433±0.001 | 0.470±0.002 | 0.067±0.003 | 0.464±0.001 | 0.466±0.001 |
| I2_lr_ocular | ocular | 0.608±0.004 | 0.580±0.002 | 0.595±0.003 | 0.507±0.006 | 0.653±0.007 | 0.491±0.003 | 0.499±0.003 | 0.159±0.005 | 0.424±0.002 | 0.420±0.002 |
| I2_lgbm_ocular | ocular | 0.608±0.004 | 0.571±0.004 | 0.587±0.003 | 0.494±0.011 | 0.648±0.006 | 0.482±0.004 | 0.488±0.007 | 0.142±0.008 | 0.430±0.004 | 0.429±0.004 |
| I3_lr_ocular_online | ocular | 0.569±0.004 | 0.551±0.003 | 0.556±0.004 | 0.524±0.003 | 0.578±0.005 | 0.451±0.004 | 0.485±0.003 | 0.100±0.007 | 0.449±0.001 | 0.449±0.003 |
| I3_lr_ocular_online_raw | ocular | 0.598±0.006 | 0.574±0.009 | 0.587±0.006 | 0.512±0.026 | 0.637±0.007 | 0.482±0.008 | 0.496±0.016 | 0.147±0.018 | 0.429±0.006 | 0.426±0.009 |
| I3_lgbm_ocular_online_raw | ocular | 0.612±0.004 | 0.579±0.003 | 0.596±0.005 | 0.495±0.009 | 0.663±0.013 | 0.493±0.006 | 0.494±0.003 | 0.158±0.007 | 0.421±0.003 | 0.421±0.003 |
| I3_lr_ocular_offline_TRANSDUCTIVE | ocular | 0.609±0.004 | 0.585±0.009 | 0.602±0.007 | 0.500±0.020 | 0.670±0.008 | 0.500±0.009 | 0.500±0.014 | 0.169±0.017 | 0.419±0.004 | 0.415±0.009 |
| I3_lr_ocular_online_allroles | ocular | 0.605±0.004 | 0.576±0.006 | 0.591±0.006 | 0.503±0.009 | 0.650±0.005 | 0.487±0.007 | 0.495±0.008 | 0.152±0.012 | 0.426±0.001 | 0.424±0.006 |
| I4_lr_rt | rt | 0.613±0.002 | 0.588±0.005 | 0.606±0.004 | 0.500±0.013 | 0.676±0.009 | 0.505±0.006 | 0.503±0.008 | 0.177±0.009 | 0.418±0.003 | 0.412±0.005 |
| I4_lr_context | context | 0.750±0.001 | 0.690±0.001 | 0.684±0.001 | 0.719±0.005 | 0.661±0.005 | 0.584±0.002 | 0.644±0.001 | 0.372±0.002 | 0.310±0.001 | 0.310±0.001 |
| I4_lgbm_context | context | 0.753±0.004 | 0.701±0.006 | 0.685±0.005 | 0.777±0.011 | 0.624±0.002 | 0.577±0.004 | 0.663±0.007 | 0.394±0.012 | 0.311±0.004 | 0.299±0.006 |
| I4_lgbm_context_rt | context+rt | 0.789±0.005 | 0.719±0.003 | 0.717±0.003 | 0.731±0.003 | 0.708±0.003 | 0.623±0.003 | 0.673±0.003 | 0.430±0.006 | 0.280±0.003 | 0.281±0.003 |
| I4_lr_ocular_rt | ocular+rt | 0.630±0.001 | 0.600±0.004 | 0.613±0.002 | 0.537±0.014 | 0.664±0.008 | 0.513±0.003 | 0.525±0.007 | 0.199±0.007 | 0.407±0.005 | 0.400±0.004 |
| I4_lr_fusion | fusion | 0.782±0.004 | 0.712±0.001 | 0.717±0.000 | 0.689±0.009 | 0.736±0.007 | 0.633±0.003 | 0.660±0.003 | 0.419±0.002 | 0.283±0.002 | 0.288±0.001 |
| I4_lgbm_context_ocular | context+ocular | 0.780±0.004 | 0.710±0.005 | 0.710±0.006 | 0.707±0.008 | 0.713±0.012 | 0.619±0.008 | 0.660±0.005 | 0.412±0.010 | 0.290±0.006 | 0.290±0.005 |
| I4_lgbm_fusion | fusion | 0.793±0.005 | 0.723±0.008 | 0.726±0.007 | 0.708±0.013 | 0.737±0.010 | 0.640±0.009 | 0.672±0.009 | 0.439±0.015 | 0.277±0.007 | 0.277±0.008 |

PVC per participant (scores averaged over repeats; detection = one-sided Mann-Whitney lie > truth at α=.05), plus AUC within bid-position strata:

| Idea | mean ppAUC | median | frac ppAUC>.5 | detected | mean pp BA | mean pp EER | mean pp HTER | AUC within bid-pos |
|---|---|---|---|---|---|---|---|---|
| I1_paper_ocular | 0.550 | 0.539 | 0.80 | 9/35 | 0.532 | 0.465 | 0.468 | 0.496 |
| I1_paper_ocular_rt | 0.594 | 0.582 | 0.89 | 18/35 | 0.552 | 0.438 | 0.448 | 0.579 |
| I1_paper_ocular_online | 0.547 | 0.548 | 0.74 | 7/35 | 0.539 | 0.469 | 0.461 | 0.511 |
| I2_lr_ocular | 0.567 | 0.546 | 0.77 | 13/35 | 0.541 | 0.460 | 0.459 | 0.635 |
| I2_lgbm_ocular | 0.577 | 0.566 | 0.97 | 14/35 | 0.551 | 0.454 | 0.449 | 0.634 |
| I3_lr_ocular_online | 0.574 | 0.550 | 0.86 | 14/35 | 0.555 | 0.448 | 0.445 | 0.559 |
| I3_lr_ocular_online_raw | 0.567 | 0.557 | 0.86 | 13/35 | 0.536 | 0.467 | 0.464 | 0.622 |
| I3_lgbm_ocular_online_raw | 0.583 | 0.578 | 0.97 | 15/35 | 0.553 | 0.453 | 0.447 | 0.636 |
| I3_lr_ocular_offline_TRANSDUCTIVE | 0.573 | 0.563 | 0.89 | 13/35 | 0.527 | 0.461 | 0.473 | 0.633 |
| I3_lr_ocular_online_allroles | 0.569 | 0.553 | 0.86 | 14/35 | 0.531 | 0.474 | 0.469 | 0.624 |
| I4_lr_rt | 0.578 | 0.587 | 0.74 | 17/35 | 0.555 | 0.444 | 0.445 | 0.645 |
| I4_lr_context | 0.745 | 0.767 | 0.97 | 34/35 | 0.705 | 0.305 | 0.295 | 0.582 |
| I4_lgbm_context | 0.746 | 0.761 | 1.00 | 34/35 | 0.703 | 0.322 | 0.297 | 0.590 |
| I4_lgbm_context_rt | 0.779 | 0.807 | 0.97 | 34/35 | 0.698 | 0.285 | 0.302 | 0.684 |
| I4_lr_ocular_rt | 0.598 | 0.589 | 0.86 | 18/35 | 0.563 | 0.428 | 0.437 | 0.654 |
| I4_lr_fusion | 0.782 | 0.819 | 0.97 | 34/35 | 0.703 | 0.283 | 0.297 | 0.672 |
| I4_lgbm_context_ocular | 0.770 | 0.778 | 1.00 | 34/35 | 0.700 | 0.309 | 0.300 | 0.666 |
| I4_lgbm_fusion | 0.783 | 0.801 | 1.00 | 34/35 | 0.711 | 0.289 | 0.289 | 0.693 |

PVC paired comparisons on identical folds. ΔAUC is pooled trial AUC with a participant-cluster bootstrap CI, Holm across the ocular ideas; Wilcoxon is on per-participant AUCs (Holm in `followup.json`):

| Comparison | ΔAUC [95% CI] | p | Holm p | reliable | Δ mean ppAUC | Wilcoxon p |
|---|---|---|---|---|---|---|
| I1_paper_ocular_online vs I1_paper_ocular | +0.018 [-0.003, +0.041] | 0.121 | 0.121 | no | -0.003 | 0.252 |
| I2_lr_ocular vs I1_paper_ocular | +0.089 [+0.024, +0.162] | 0.003 | 0.015 | yes | +0.017 | 0.295 |
| I2_lgbm_ocular vs I1_paper_ocular | +0.090 [+0.026, +0.164] | 0.002 | 0.012 | yes | +0.028 | 0.067 |
| I3_lr_ocular_online vs I1_paper_ocular | +0.049 [+0.020, +0.077] | 0.003 | 0.015 | yes | +0.024 | 0.140 |
| I3_lr_ocular_online_raw vs I1_paper_ocular | +0.084 [+0.016, +0.152] | 0.010 | 0.021 | yes | +0.017 | 0.302 |
| I3_lgbm_ocular_online_raw vs I1_paper_ocular | +0.096 [+0.035, +0.163] | 0.000 | 0.000 | yes | +0.033 | 0.015 |
| I3_lr_ocular_online_allroles vs I1_paper_ocular | +0.088 [+0.019, +0.161] | 0.007 | 0.021 | yes | +0.019 | 0.342 |
| fusion_vs_context_rt: I4_lgbm_fusion | +0.005 [-0.004, +0.014] | 0.249 | (see §5) | – | +0.004 | 0.565 |
| context_ocular_vs_context: I4_lgbm_context_ocular | +0.030 [+0.013, +0.048] | 0.001 | (see §5) | – | +0.024 | 0.001 |
| ocular_rt_vs_rt: I4_lr_ocular_rt | +0.017 [-0.007, +0.043] | 0.142 | (see §5) | – | +0.020 | 0.016 |

**PVP** — trial level, mean ± SD over 3 repeats (threshold 0). The honest baseline is `I1_paper_ocular`.

| Idea | Block | AUC | balanced_accuracy | accuracy | sensitivity | specificity | precision | f1 | MCC | EER | HTER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I1_paper_ocular | ocular | 0.550±0.000 | 0.528±0.001 | 0.536±0.001 | 0.503±0.003 | 0.553±0.002 | 0.354±0.001 | 0.416±0.001 | 0.052±0.002 | 0.473±0.002 | 0.472±0.001 |
| I1_paper_ocular_rt | ocular+rt | 0.579±0.000 | 0.555±0.001 | 0.560±0.001 | 0.540±0.003 | 0.570±0.003 | 0.380±0.001 | 0.446±0.001 | 0.103±0.002 | 0.445±0.000 | 0.445±0.001 |
| I1_paper_ocular_aoi | ocular | 0.571±0.001 | 0.546±0.001 | 0.561±0.001 | 0.502±0.002 | 0.590±0.001 | 0.374±0.001 | 0.429±0.001 | 0.087±0.002 | 0.456±0.001 | 0.454±0.001 |
| I1_paper_ocular_online | ocular | 0.582±0.000 | 0.551±0.000 | 0.565±0.001 | 0.509±0.000 | 0.592±0.001 | 0.379±0.001 | 0.434±0.000 | 0.096±0.001 | 0.448±0.001 | 0.449±0.000 |
| I2_lr_ocular | ocular | 0.576±0.003 | 0.552±0.002 | 0.576±0.001 | 0.482±0.005 | 0.622±0.001 | 0.384±0.002 | 0.427±0.003 | 0.099±0.004 | 0.452±0.003 | 0.448±0.002 |
| I2_lgbm_ocular | ocular | 0.559±0.002 | 0.547±0.004 | 0.579±0.003 | 0.454±0.009 | 0.640±0.003 | 0.381±0.004 | 0.414±0.006 | 0.090±0.008 | 0.463±0.003 | 0.453±0.004 |
| I2_lr_ocular_common | ocular | 0.569±0.003 | 0.551±0.007 | 0.583±0.010 | 0.457±0.005 | 0.645±0.015 | 0.386±0.010 | 0.419±0.007 | 0.098±0.015 | 0.455±0.004 | 0.449±0.007 |
| I3_lr_ocular_online | ocular | 0.600±0.002 | 0.576±0.003 | 0.591±0.003 | 0.530±0.004 | 0.621±0.003 | 0.406±0.003 | 0.460±0.003 | 0.144±0.005 | 0.424±0.003 | 0.424±0.003 |
| I3_lr_ocular_online_raw | ocular | 0.598±0.003 | 0.574±0.004 | 0.595±0.002 | 0.514±0.009 | 0.634±0.002 | 0.407±0.003 | 0.455±0.005 | 0.142±0.007 | 0.427±0.002 | 0.426±0.004 |
| I3_lgbm_ocular_online_raw | ocular | 0.583±0.003 | 0.559±0.009 | 0.593±0.004 | 0.460±0.023 | 0.658±0.007 | 0.396±0.008 | 0.426±0.014 | 0.114±0.016 | 0.441±0.004 | 0.441±0.009 |
| I3_lr_ocular_offline_TRANSDUCTIVE | ocular | 0.596±0.005 | 0.572±0.006 | 0.593±0.006 | 0.512±0.007 | 0.633±0.008 | 0.405±0.007 | 0.452±0.007 | 0.138±0.012 | 0.431±0.005 | 0.428±0.006 |
| I3_lr_ocular_online_allroles | ocular | 0.582±0.004 | 0.561±0.003 | 0.577±0.003 | 0.516±0.009 | 0.606±0.006 | 0.390±0.003 | 0.445±0.005 | 0.116±0.006 | 0.441±0.003 | 0.439±0.003 |
| I3_lr_ocular_common_online_raw | ocular | 0.566±0.001 | 0.551±0.001 | 0.582±0.004 | 0.461±0.015 | 0.642±0.014 | 0.386±0.001 | 0.420±0.006 | 0.099±0.001 | 0.458±0.004 | 0.449±0.001 |
| I4_lr_rt | rt | 0.595±0.001 | 0.574±0.001 | 0.585±0.001 | 0.540±0.005 | 0.607±0.003 | 0.402±0.001 | 0.461±0.002 | 0.139±0.002 | 0.428±0.001 | 0.426±0.001 |
| I4_lr_context | context | 0.777±0.003 | 0.755±0.000 | 0.715±0.000 | 0.873±0.000 | 0.637±0.000 | 0.540±0.000 | 0.668±0.000 | 0.480±0.000 | 0.281±0.001 | 0.245±0.000 |
| I4_lgbm_context | context | 0.772±0.001 | 0.753±0.002 | 0.714±0.001 | 0.867±0.004 | 0.640±0.001 | 0.540±0.001 | 0.665±0.002 | 0.476±0.003 | 0.289±0.001 | 0.247±0.002 |
| I4_lgbm_context_rt | context+rt | 0.804±0.002 | 0.748±0.001 | 0.726±0.001 | 0.813±0.002 | 0.684±0.001 | 0.557±0.001 | 0.661±0.001 | 0.467±0.002 | 0.259±0.003 | 0.252±0.001 |
| I4_lr_ocular_rt | ocular+rt | 0.613±0.003 | 0.585±0.006 | 0.604±0.004 | 0.530±0.009 | 0.641±0.002 | 0.419±0.006 | 0.468±0.007 | 0.162±0.011 | 0.416±0.001 | 0.415±0.006 |
| I4_lr_fusion | fusion | 0.809±0.001 | 0.749±0.001 | 0.726±0.001 | 0.815±0.002 | 0.683±0.003 | 0.557±0.002 | 0.662±0.001 | 0.468±0.002 | 0.262±0.002 | 0.251±0.001 |
| I4_lgbm_context_ocular | context+ocular | 0.797±0.002 | 0.744±0.002 | 0.729±0.001 | 0.790±0.003 | 0.699±0.002 | 0.562±0.002 | 0.657±0.002 | 0.461±0.003 | 0.261±0.001 | 0.256±0.002 |
| I4_lgbm_fusion | fusion | 0.806±0.001 | 0.744±0.004 | 0.735±0.003 | 0.771±0.009 | 0.717±0.002 | 0.571±0.003 | 0.656±0.005 | 0.461±0.008 | 0.259±0.004 | 0.256±0.004 |

PVP per participant (scores averaged over repeats; detection = one-sided Mann-Whitney lie > truth at α=.05), plus AUC within bid-position strata:

| Idea | mean ppAUC | median | frac ppAUC>.5 | detected | mean pp BA | mean pp EER | mean pp HTER | AUC within bid-pos |
|---|---|---|---|---|---|---|---|---|
| I1_paper_ocular | 0.560 | 0.548 | 0.57 | 15/42 | 0.529 | 0.466 | 0.471 | 0.543 |
| I1_paper_ocular_rt | 0.592 | 0.579 | 0.74 | 19/42 | 0.558 | 0.431 | 0.442 | 0.578 |
| I1_paper_ocular_aoi | 0.581 | 0.560 | 0.69 | 18/42 | 0.547 | 0.448 | 0.453 | 0.567 |
| I1_paper_ocular_online | 0.583 | 0.565 | 0.74 | 18/42 | 0.553 | 0.444 | 0.447 | 0.572 |
| I2_lr_ocular | 0.595 | 0.614 | 0.79 | 23/42 | 0.556 | 0.433 | 0.444 | 0.576 |
| I2_lgbm_ocular | 0.567 | 0.568 | 0.74 | 16/42 | 0.542 | 0.460 | 0.458 | 0.557 |
| I2_lr_ocular_common | 0.578 | 0.563 | 0.79 | 16/42 | 0.546 | 0.441 | 0.454 | 0.580 |
| I3_lr_ocular_online | 0.601 | 0.604 | 0.79 | 21/42 | 0.579 | 0.420 | 0.421 | 0.594 |
| I3_lr_ocular_online_raw | 0.600 | 0.612 | 0.83 | 22/42 | 0.571 | 0.426 | 0.429 | 0.593 |
| I3_lgbm_ocular_online_raw | 0.586 | 0.603 | 0.76 | 21/42 | 0.568 | 0.435 | 0.432 | 0.585 |
| I3_lr_ocular_offline_TRANSDUCTIVE | 0.602 | 0.615 | 0.83 | 22/42 | 0.572 | 0.425 | 0.428 | 0.597 |
| I3_lr_ocular_online_allroles | 0.593 | 0.621 | 0.79 | 22/42 | 0.565 | 0.436 | 0.435 | 0.583 |
| I3_lr_ocular_common_online_raw | 0.576 | 0.558 | 0.76 | 18/42 | 0.550 | 0.446 | 0.450 | 0.569 |
| I4_lr_rt | 0.606 | 0.611 | 0.81 | 22/42 | 0.572 | 0.420 | 0.428 | 0.609 |
| I4_lr_context | 0.789 | 0.802 | 1.00 | 41/42 | 0.757 | 0.272 | 0.243 | 0.486 |
| I4_lgbm_context | 0.777 | 0.782 | 0.98 | 41/42 | 0.757 | 0.290 | 0.243 | 0.495 |
| I4_lgbm_context_rt | 0.808 | 0.822 | 1.00 | 40/42 | 0.752 | 0.258 | 0.248 | 0.599 |
| I4_lr_ocular_rt | 0.620 | 0.620 | 0.86 | 25/42 | 0.586 | 0.405 | 0.414 | 0.617 |
| I4_lr_fusion | 0.815 | 0.826 | 1.00 | 41/42 | 0.750 | 0.259 | 0.250 | 0.610 |
| I4_lgbm_context_ocular | 0.804 | 0.814 | 1.00 | 41/42 | 0.749 | 0.254 | 0.251 | 0.573 |
| I4_lgbm_fusion | 0.814 | 0.830 | 1.00 | 42/42 | 0.748 | 0.250 | 0.252 | 0.612 |

PVP paired comparisons on identical folds. ΔAUC is pooled trial AUC with a participant-cluster bootstrap CI, Holm across the ocular ideas; Wilcoxon is on per-participant AUCs (Holm in `followup.json`):

| Comparison | ΔAUC [95% CI] | p | Holm p | reliable | Δ mean ppAUC | Wilcoxon p |
|---|---|---|---|---|---|---|
| I1_paper_ocular_aoi vs I1_paper_ocular | +0.021 [+0.006, +0.036] | 0.003 | 0.027 | yes | +0.021 | 0.016 |
| I1_paper_ocular_online vs I1_paper_ocular | +0.032 [+0.013, +0.050] | 0.002 | 0.020 | yes | +0.023 | 0.004 |
| I2_lr_ocular vs I1_paper_ocular | +0.027 [-0.000, +0.055] | 0.051 | 0.204 | no | +0.035 | 0.029 |
| I2_lgbm_ocular vs I1_paper_ocular | +0.012 [-0.016, +0.039] | 0.369 | 0.387 | no | +0.006 | 0.620 |
| I2_lr_ocular_common vs I1_paper_ocular | +0.020 [-0.007, +0.045] | 0.129 | 0.387 | no | +0.017 | 0.248 |
| I3_lr_ocular_online vs I1_paper_ocular | +0.052 [+0.018, +0.086] | 0.003 | 0.027 | yes | +0.041 | 0.023 |
| I3_lr_ocular_online_raw vs I1_paper_ocular | +0.049 [+0.016, +0.082] | 0.003 | 0.027 | yes | +0.040 | 0.027 |
| I3_lgbm_ocular_online_raw vs I1_paper_ocular | +0.038 [+0.008, +0.068] | 0.015 | 0.090 | no | +0.026 | 0.141 |
| I3_lr_ocular_online_allroles vs I1_paper_ocular | +0.034 [+0.003, +0.065] | 0.029 | 0.145 | no | +0.033 | 0.055 |
| I3_lr_ocular_common_online_raw vs I1_paper_ocular | +0.017 [-0.007, +0.040] | 0.152 | 0.387 | no | +0.015 | 0.196 |
| fusion_vs_context_rt: I4_lgbm_fusion | +0.003 [-0.006, +0.013] | 0.505 | (see §5) | – | +0.006 | 0.442 |
| context_ocular_vs_context: I4_lgbm_context_ocular | +0.027 [+0.013, +0.041] | 0.000 | (see §5) | – | +0.027 | 0.003 |
| ocular_rt_vs_rt: I4_lr_ocular_rt | +0.019 [-0.004, +0.042] | 0.114 | (see §5) | – | +0.014 | 0.118 |

**Idea 5: transfer between experiments** (train on all of one, test on all of the other; common ocular features; participant-cluster CI):

| Direction / model | AUC [95% CI] | BA | EER | HTER | mean ppAUC | detected |
|---|---|---|---|---|---|---|
| pvc_to_pvp/paper_ocular | 0.547 [0.516, 0.576] | 0.527 | 0.468 | 0.473 | 0.554 | 14/42 |
| pvc_to_pvp/paper_ocular_online | 0.558 [0.526, 0.589] | 0.536 | 0.458 | 0.464 | 0.561 | 15/42 |
| pvc_to_pvp/lr_ocular_common | 0.525 [0.506, 0.545] | 0.514 | 0.481 | 0.486 | 0.531 | 5/42 |
| pvc_to_pvp/lr_ocular_common_online_raw | 0.519 [0.497, 0.539] | 0.505 | 0.486 | 0.495 | 0.532 | 7/42 |
| pvc_to_pvp/lr_ocular_common_online | 0.538 [0.518, 0.558] | 0.529 | 0.468 | 0.471 | 0.544 | 10/42 |
| pvp_to_pvc/paper_ocular | 0.571 [0.537, 0.605] | 0.544 | 0.450 | 0.456 | 0.560 | 11/35 |
| pvp_to_pvc/paper_ocular_online | 0.547 [0.525, 0.572] | 0.535 | 0.463 | 0.465 | 0.551 | 11/35 |
| pvp_to_pvc/lr_ocular_common | 0.598 [0.551, 0.652] | 0.555 | 0.443 | 0.445 | 0.557 | 12/35 |
| pvp_to_pvc/lr_ocular_common_online_raw | 0.568 [0.540, 0.596] | 0.539 | 0.452 | 0.461 | 0.567 | 13/35 |
| pvp_to_pvc/lr_ocular_common_online | 0.566 [0.542, 0.592] | 0.550 | 0.457 | 0.450 | 0.571 | 13/35 |
Reading: train on one experiment, test on the other, with the common ocular features only (PvP AOI features are not available in PvC). AUC is 0.519–0.558 for PvC to PvP (balanced accuracy 0.505–0.536) and 0.547–0.598 for PvP to PvC (balanced accuracy 0.535–0.555). This is near chance to weak. The best cell, `pvp_to_pvc/lr_ocular_common` (0.598 [0.551, 0.652]), has a CI that excludes 0.5 but is no better than the within-experiment ocular models (PvC 0.608–0.612). The fixed-sign paper classifier transfers about as well as the learned models (0.547 and 0.571), so training adds nothing across setups. The setups differ in sensor (EyeLink 1000 Hz, chinrest, screen vs Pupil Labs Neon 200 Hz, glasses), opponent (computer vs a human stranger), and sample (mixed vs women only). The transfer CIs are not corrected for multiple comparisons, and the detection counts (5–15 of 35 or 42 participants at α = .05, uncorrected) are low.

## 5. Best ocular-only candidate vs published vs honest baseline


**PVC** (metrics on repeat-averaged out-of-fold scores; CI = participant-cluster bootstrap, 2000 draws; edd row bootstrap in results.json)

| Model | AUC [CI] | BA [CI] | EER [CI] | HTER [CI] | MCC | mean ppAUC [CI] | Permutation p |
|---|---|---|---|---|---|---|---|
| I3_lgbm_ocular_online_raw | 0.617 [0.578, 0.668] | 0.581 [0.549, 0.622] | 0.422 [0.381, 0.449] | 0.419 [0.378, 0.451] | 0.162 | [0.563, 0.605] | 0.020 (obs 0.606, null 0.541, n=50) |
| I1_paper_ocular | 0.522 [0.490, 0.554] | 0.526 [0.503, 0.549] | 0.472 [0.450, 0.496] | 0.474 [0.451, 0.497] | 0.050 | [0.526, 0.575] | – |
| Published | not reported (no classifier; effect sizes only, see §1) | | | | | | |

Mechanism checks for `I3_lgbm_ocular_online_raw` (`followup.json`):

Feature-family ablation (same model family, nested tuning, same folds; Δ = pooled AUC vs the full candidate with a cluster-bootstrap CI):

| Variant | AUC (mean±SD) | mean ppAUC | ΔAUC vs full [CI] |
|---|---|---|---|
| drop_fixation_core | 0.594±0.010 | 0.569 | -0.018 [-0.033, -0.004] |
| only_fixation_core | 0.576±0.005 | 0.568 | -0.037 [-0.067, -0.010] |
| drop_fixation_length_dependent | 0.609±0.006 | 0.571 | -0.003 [-0.015, +0.008] |
| only_fixation_length_dependent | 0.579±0.004 | 0.566 | -0.034 [-0.072, -0.002] |
| drop_blink | 0.601±0.002 | 0.589 | -0.010 [-0.034, +0.010] |
| only_blink | 0.570±0.011 | 0.540 | -0.042 [-0.063, -0.022] |

Top features (lgbm_gain_share): dev_any_blink +0.108, fix_dur_max +0.067, dev_blink_rate +0.066, fix_rate +0.059, dev_fix_dur_sd +0.056, dev_fix_rate +0.056, fix_dur_mean +0.055, dev_fix_dur_max +0.053

Is it a bid-time proxy, and how much is between-person level?

| Model | Spearman(score, log bid time) | AUC within person-RT quartiles | pooled AUC | pooled AUC, person-centred scores | mean ppAUC |
|---|---|---|---|---|---|
| I3_lgbm_ocular_online_raw | +0.442 | 0.600 | 0.617 | 0.570 | 0.583 |
| I1_paper_ocular | -0.117 | 0.529 | 0.522 | 0.550 | 0.550 |
| I4_lr_rt | +0.975 | 0.593 | 0.614 | 0.576 | 0.578 |

Holm across the 'eyes add' tests: pooled-AUC bootstrap {"context_ocular_vs_context": 0.003, "ocular_rt_vs_rt": 0.284, "fusion_vs_context_rt": 0.284}; per-participant Wilcoxon {"context_ocular_vs_context": 0.0034, "ocular_rt_vs_rt": 0.0316, "fusion_vs_context_rt": 0.5654}. Per-participant Wilcoxon vs baseline, Holm: {"I3_lgbm_ocular_online_raw": 0.1056, "I2_lgbm_ocular": 0.4044, "I3_lr_ocular_online": 0.6983, "I1_paper_ocular_online": 1.0, "I2_lr_ocular": 1.0, "I3_lr_ocular_online_raw": 1.0, "I3_lr_ocular_online_allroles": 1.0}.

Per-participant means with participant-bootstrap CIs: I3_lgbm_ocular_online_raw: AUC 0.583 [0.563, 0.605], BA 0.553, EER 0.453 [0.438, 0.469], HTER 0.447 [0.426, 0.465]; I1_paper_ocular: AUC 0.550 [0.526, 0.575], BA 0.532, EER 0.465 [0.442, 0.487], HTER 0.468 [0.446, 0.488]

**PVP** (metrics on repeat-averaged out-of-fold scores; CI = participant-cluster bootstrap, 2000 draws; edd row bootstrap in results.json)

| Model | AUC [CI] | BA [CI] | EER [CI] | HTER [CI] | MCC | mean ppAUC [CI] | Permutation p |
|---|---|---|---|---|---|---|---|
| I3_lr_ocular_online | 0.602 [0.568, 0.635] | 0.580 [0.554, 0.606] | 0.419 [0.395, 0.446] | 0.420 [0.394, 0.446] | 0.152 | [0.566, 0.635] | 0.020 (obs 0.601, null 0.501, n=50) |
| I1_paper_ocular | 0.550 [0.517, 0.582] | 0.527 [0.502, 0.550] | 0.474 [0.446, 0.497] | 0.473 [0.450, 0.498] | 0.050 | [0.526, 0.592] | – |
| Published | not reported (no classifier; effect sizes only, see §1) | | | | | | |

Mechanism checks for `I3_lr_ocular_online` (`followup.json`):

Feature-family ablation (same model family, nested tuning, same folds; Δ = pooled AUC vs the full candidate with a cluster-bootstrap CI):

| Variant | AUC (mean±SD) | mean ppAUC | ΔAUC vs full [CI] |
|---|---|---|---|
| drop_fixation_core | 0.603±0.002 | 0.603 | +0.003 [-0.001, +0.007] |
| only_fixation_core | 0.561±0.004 | 0.563 | -0.041 [-0.067, -0.017] |
| drop_fixation_length_dependent | 0.599±0.002 | 0.600 | -0.002 [-0.007, +0.005] |
| only_fixation_length_dependent | 0.571±0.001 | 0.573 | -0.031 [-0.060, -0.003] |
| drop_blink | 0.598±0.001 | 0.600 | -0.002 [-0.008, +0.004] |
| only_blink | 0.526±0.004 | 0.535 | -0.075 [-0.110, -0.040] |
| drop_aoi | 0.575±0.004 | 0.578 | -0.027 [-0.053, -0.002] |
| only_aoi | 0.596±0.001 | 0.597 | -0.005 [-0.014, +0.005] |

Top features (lr_standardised_coef): dev_fix_dur_max +0.408, dev_dwell_face -0.266, dev_dur_dice +0.258, dev_fix_dur_mean -0.205, dev_dur_face +0.157, dev_fix_dur_sd -0.137, dev_dwell_dice +0.131, dev_share_face +0.116

Is it a bid-time proxy, and how much is between-person level?

| Model | Spearman(score, log bid time) | AUC within person-RT quartiles | pooled AUC | pooled AUC, person-centred scores | mean ppAUC |
|---|---|---|---|---|---|
| I3_lr_ocular_online | +0.266 | 0.572 | 0.602 | 0.600 | 0.601 |
| I1_paper_ocular | +0.158 | 0.535 | 0.550 | 0.557 | 0.560 |
| I4_lr_rt | +0.918 | 0.517 | 0.595 | 0.600 | 0.606 |

Holm across the 'eyes add' tests: pooled-AUC bootstrap {"context_ocular_vs_context": 0.0, "ocular_rt_vs_rt": 0.228, "fusion_vs_context_rt": 0.505}; per-participant Wilcoxon {"context_ocular_vs_context": 0.0076, "ocular_rt_vs_rt": 0.2368, "fusion_vs_context_rt": 0.4421}. Per-participant Wilcoxon vs baseline, Holm: {"I1_paper_ocular_online": 0.0395, "I1_paper_ocular_aoi": 0.1477, "I3_lr_ocular_online": 0.1802, "I3_lr_ocular_online_raw": 0.1869, "I2_lr_ocular": 0.1869, "I3_lr_ocular_online_allroles": 0.2757, "I3_lgbm_ocular_online_raw": 0.5652, "I3_lr_ocular_common_online_raw": 0.5868, "I2_lr_ocular_common": 0.5868, "I2_lgbm_ocular": 0.6204}.

Per-participant means with participant-bootstrap CIs: I3_lr_ocular_online: AUC 0.601 [0.566, 0.635], BA 0.579, EER 0.420 [0.394, 0.444], HTER 0.421 [0.394, 0.447]; I1_paper_ocular: AUC 0.560 [0.526, 0.592], BA 0.529, EER 0.466 [0.440, 0.492], HTER 0.471 [0.449, 0.494]

**Why the eye-only models help, and how far (mechanism, `followup.json`).**
- **Learned weights beat the fixed-sign classifier.** `I1_paper_ocular` z-scores four measures (fixation rate, fixation duration, blink rate, blink duration) with the paper's signs and equal weights, and fits nothing. In PvC it has AUC 0.496 inside bid-position strata, i.e. no signal beyond bid position; the learned ocular models that keep raw features reach 0.622–0.636 (`I3_lr_ocular_online`, which drops them, 0.559). The learned models use more features (maximum, SD and first-fixation duration, fixation-time fraction, blink presence) and, in the online variants, each feature's deviation from the player's own earlier bids.
- **No single feature family carries it in PvC.** Ablation on the frozen folds (ΔAUC versus the full model 0.612, cluster CI, unadjusted for the 6 tests): dropping core fixation features -0.018 [-0.033, -0.004]; dropping blink features -0.010 [-0.034, +0.010]; dropping length-dependent fixation features -0.003 [-0.015, +0.008]. Any one family alone gives 0.570–0.579. Top gain shares are blink presence (0.108), maximum fixation duration (0.067) and the deviation of blink rate (0.066).
- **In PvP the AOI features carry the signal.** Full model 0.600. Dropping the AOI features costs -0.027 [-0.053, -0.002]; AOI features alone give 0.596 (-0.005 [-0.014, +0.005], not different from the full model); global fixation or blink features alone give only 0.526–0.571. The largest coefficients are the deviations of maximum fixation duration (+0.41), face dwell (-0.27) and dice fixation duration (+0.26). This agrees with the paper's own AOI effects (dice fixation duration F = 26.3).
- **Part of it is bid time.** The PvC best score correlates with log bid time at Spearman 0.44 (PvP 0.27), and the bid-time-only model matches it (0.613 vs 0.612). Ocular + RT versus RT alone is +0.017 [-0.007, +0.043] (PvC, Holm p = .284) and +0.019 [-0.004, +0.042] (PvP, Holm p = .228). The eye score still separates lies inside within-person bid-time quartiles (AUC 0.600 PvC, 0.572 PvP, against 0.593 and 0.517 for the bid-time model), so it is not only bid time, but the data cannot show that the eyes add reliably to bid time on the pooled measure. On the per-participant measure ocular + RT versus RT is nominally reliable in PvC (Wilcoxon Holm p = .032) and not in PvP (.237).
- **Part of the PvC pooled AUC is between-person.** After subtracting each participant's mean score, PvC AUC falls from 0.617 to 0.570; PvP is unchanged (0.602 to 0.600). The within-person figure is the mean per-participant AUC: PvC 0.583 [0.563, 0.605] (baseline 0.550 [0.526, 0.575]), PvP 0.601 [0.566, 0.635] (baseline 0.560 [0.526, 0.592]); at α = .05 (uncorrected) 15/35 vs 9/35 and 21/42 vs 15/42 players are detected. One PvC player (pvc_28) lied in 98.8% of 169 bids.
- **Why context dominates.** Deception here is spontaneous and strategic: P(lie) rises with bid position (.20/.50/.67/.71 in PvC; .09/.49/.65/.69 in PvP), and context also carries session, round and previous-round outcome. In PvP the context model has AUC 0.486–0.495 inside bid-position strata, so its signal is bid position alone; in PvC it is 0.582–0.590. A context model predicts game strategy, not behaviour, and says nothing about deception detection from the eyes. This is the main reason the eye gains must be read against the context and bid-time rows.

**Caveats.**
- The best ocular model is the top of 7 (PvC) or 10 (PvP) ocular variants chosen on the same CV, so its absolute level is mildly optimistic; the top variants are within about 0.004 AUC of each other (PvC `I2_lr`, `I2_lgbm`, `I3_lgbm` 0.608–0.612; PvP `I3_lr_online` 0.600 vs `I3_lr_online_raw` 0.598). Holm covers the comparison against the baseline, not this selection.
- The permutation test shuffles labels **within** each participant (keeps each person's lie rate), with hyperparameters frozen at the all-data nested choice and fold repeat 0 only (observed AUC 0.606 PvC, 0.601 PvP; null means 0.541 and 0.501). It tests the bid-wise link between eyes and honesty. The protocol's between-participant shuffle was not run. With 50 permutations the smallest possible p is 1/51 = .0196.
- The PvP baseline `I1_paper_ocular` uses the four global measures only. The paper-effect variants with online normalisation (0.582) or AOI features (0.571) are stronger and were not tested directly against the best model; the Holm family compares every ocular idea with `I1_paper_ocular` only.
- Bootstrap CIs resample participants. In PvP the two players of a dyad share one game, so dyad-level clustering would give somewhat wider CIs.
- Labels: 0.9% (PvC) and 4.3% (PvP) of contested bids disagree under either coding, so some label noise remains. Bid time is taken from the fixation file (the PvC blink file has its own, different BidTime). Only the bidder's bids are modelled.
- No raw gaze samples are released, so the pipeline works on the fixation and blink event tables.

## 6. Efficiency (this laptop, single-threaded CPU)

Hardware: Intel64 Family 6 Model 142 Stepping 10, GenuineIntel, 8 logical CPUs, Windows-11-10.0.26200-SP0, Python 3.14.6. Unit of decision = one bid. The input is the bid's fixation and blink event list (no raw gaze samples are released), so sensor-rate FPS does not apply.

| Exp | Model | Latency/bid p50 / p95 (ms) | Latency/participant p50 / p95 (ms, n bids) | Throughput (bids/s) | Peak mem (KB) | Model size (B) | Params | Feature extraction/bid p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| pvc | I1_paper_ocular | 0.090 / 0.136 | 0.122 / 0.163 (155) | 9,961,486 | 15.7 | 467 | 13 | 0.225 / 0.268 |
| pvc | I3_lgbm_ocular_online_raw | 3.683 / 5.046 | 11.208 / 12.980 (155) | 20,064 | 117.2 | 356,379 | 11,410 | 0.226 / 0.250 |
| pvp | I1_paper_ocular | 0.111 / 0.208 | 0.144 / 0.195 (97) | 8,164,445 | 10.2 | 467 | 13 | 0.549 / 0.925 |
| pvp | I3_lr_ocular_online | 4.108 / 13.005 | 3.935 / 6.041 (97) | 392,050 | 143.4 | 3,268 | 116 | 0.524 / 0.651 |
Both best models run in a few milliseconds per bid on one CPU thread (LightGBM `n_jobs=1`), using under 150 KB at inference, and feature extraction from the event list costs 0.2–0.55 ms per bid (p50). The 3.7–4.1 ms single-bid latency is mostly scikit-learn pipeline call overhead: scoring a whole participant (155 or 97 bids) takes 11.2 ms (PvC LightGBM) and 3.9 ms (PvP logistic regression), and batch throughput is 20,064 and 392,050 bids/s. The paper-effect classifier takes about 0.1 ms and 467 bytes. A bid lasts about 2–3 s, so even the slowest model uses well under 1% of the available time. The sensor-rate FPS and real-time factor are not applicable (EyeLink 1000 Hz and Neon 200 Hz data are only released as event tables). Latency and memory were measured once, on a laptop that was also running other experiments, so treat them as indicative.

## 7. Recommendation

**Not worth a full run as a headline deception-detection result; keep it, if the authors permit, as a context-controlled benchmark and a cautionary case.**
- **Licence first.** Use this dataset in the paper only if the Foucher and Huckauf team give permission (the user's decision). Send them the label-inversion finding at the same time.
- **What the evidence supports.** From eye measures alone, a participant-disjoint classifier reaches AUC about 0.60–0.62 (EER 0.42), clearly above the paper's effects turned into a classifier (0.52–0.55) on pooled trial AUC, but this is weak, depends on bid time, and the per-participant evidence does not pass Holm. Game context and bid time give 0.79–0.81, and the eyes add no reliable gain over them (+0.003 to +0.005 AUC). Transfer between the two setups is near chance.
- **If it is included in the paper,** report the context and bid-time rows next to every eye-only result, and use AUC within bid-position strata as the primary measure. Do not present the context models as lie detection.
- **To try next, if worth the effort:** (1) 1000+ permutations, both within- and between-participant; (2) dyad-clustered CIs in PvP; (3) a direct test of the best model against the strongest paper-effect variants (online, AOI); (4) matching or stratifying on bid position and previous-round outcome before modelling; (5) raw gaze samples (if the authors can share them) for sensor-rate efficiency and for pupil or saccade features, which the event tables do not contain; (6) a pre-registered split so that model selection and evaluation do not share the same CV.

## Files

- `results.json`: all numbers. `folds.json`: frozen folds. `reproduction.json`, `data_audit.json`, `followup.json`. Run logs are `run_log*.txt`.
- Scripts:
  - `scripts/phase2_liars_dice_build.py`: bid table and label audit.
  - `scripts/phase2_liars_dice_reproduce.py`: paper statistics.
  - `scripts/phase2_liars_dice.py`: screening, comparisons, permutation, efficiency, transfer.
  - `scripts/phase2_liars_dice_followup.py`: ablation, importance, confound checks.
  - `scripts/phase2_liars_dice_card.py`: renders this card from the JSON outputs and `card_narrative.md`.
  - Library: `src/edd/liars_dice.py`.
- Out-of-fold scores: `data/derived/liars_dice_oofcache/` (per arm) and `data/derived/liars_dice_oof.csv.gz`.
