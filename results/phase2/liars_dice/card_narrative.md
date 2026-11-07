@@study
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

@@reproduction
- **The released labels are inverted.** In the Rda files, contested bids labelled `Lie` were won by the bidder in **1057/1059 (PvC)** and **936/998 (PvP)** cases, and contested `Truth` bids in 8/881 and 25/1019. Under the rules a contested bid is won only if it is true. With swapped labels the paper's descriptives are reproduced almost exactly (table below); with the released coding every effect has the opposite sign. Swapping also restores the escalation the paper describes: P(lie) by bid position is .20/.50/.67/.71 in PvC and .09/.49/.65/.69 in PvP. **We use deceptive = released `"Truth"`** (`data_audit.json`). The remaining 0.9% (PvC) and 4.3% (PvP) of contested bids disagree under either coding, which suggests label noise (PvP events were hand-coded).
- The df in the paper (F(1, N−1)) imply the lmer was fitted on participant × honesty cell means. With one observation per cell that is a paired t-test (F = t²), which is what we report. A trial-level within-person test gives the same signs with larger t (`reproduction.json`).
- **Blink rate and AOI rates** in the paper are averaged only over bids in which the event occurred: bids without blinks, or without face fixations, are absent from the event tables. With that reading, PvP blink rate is .75/.79 (exact) and face rate is .78/.89 (paper .76/.88). The `_cond` rows below use this reading. As true rates including zeros, the PvP blink effect disappears (F=0.7).
- Under the conditional reading, PvC blink rate is *significant* (F=11.1) while the paper reports n.s. The ambiguity is unresolved: the PvC blink file has its own BidTime, which differs from the fixation file in 1207/1886 bids (median +1.0 s). The fixation-file BidTime reproduces the paper's bid times, so we use it.
- Overall: the means for bid time, fixation rate and duration, and the AOI durations match within a few units, and the F values are within about ±2 of the paper's. Residual differences come from undocumented exclusions (e.g. PvC blink duration uses 31 of 35 participants).

@@protocol
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

@@files
- `results.json`: all numbers. `folds.json`: frozen folds. `reproduction.json`, `data_audit.json`, `followup.json`. Run logs are `run_log*.txt`.
- Scripts:
  - `scripts/phase2_liars_dice_build.py`: bid table and label audit.
  - `scripts/phase2_liars_dice_reproduce.py`: paper statistics.
  - `scripts/phase2_liars_dice.py`: screening, comparisons, permutation, efficiency, transfer.
  - `scripts/phase2_liars_dice_followup.py`: ablation, importance, confound checks.
  - `scripts/phase2_liars_dice_card.py`: renders this card from the JSON outputs and `card_narrative.md`.
  - Library: `src/edd/liars_dice.py`.
- Out-of-fold scores: `data/derived/liars_dice_oofcache/` (per arm) and `data/derived/liars_dice_oof.csv.gz`.

@@summary
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

@@transfer
Reading: train on one experiment, test on the other, with the common ocular features only (PvP AOI features are not available in PvC). AUC is 0.519–0.558 for PvC to PvP (balanced accuracy 0.505–0.536) and 0.547–0.598 for PvP to PvC (balanced accuracy 0.535–0.555). This is near chance to weak. The best cell, `pvp_to_pvc/lr_ocular_common` (0.598 [0.551, 0.652]), has a CI that excludes 0.5 but is no better than the within-experiment ocular models (PvC 0.608–0.612). The fixed-sign paper classifier transfers about as well as the learned models (0.547 and 0.571), so training adds nothing across setups. The setups differ in sensor (EyeLink 1000 Hz, chinrest, screen vs Pupil Labs Neon 200 Hz, glasses), opponent (computer vs a human stranger), and sample (mixed vs women only). The transfer CIs are not corrected for multiple comparisons, and the detection counts (5–15 of 35 or 42 participants at α = .05, uncorrected) are low.

@@why
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

@@efficiency
Both best models run in a few milliseconds per bid on one CPU thread (LightGBM `n_jobs=1`), using under 150 KB at inference, and feature extraction from the event list costs 0.2–0.55 ms per bid (p50). The 3.7–4.1 ms single-bid latency is mostly scikit-learn pipeline call overhead: scoring a whole participant (155 or 97 bids) takes 11.2 ms (PvC LightGBM) and 3.9 ms (PvP logistic regression), and batch throughput is 20,064 and 392,050 bids/s. The paper-effect classifier takes about 0.1 ms and 467 bytes. A bid lasts about 2–3 s, so even the slowest model uses well under 1% of the available time. The sensor-rate FPS and real-time factor are not applicable (EyeLink 1000 Hz and Neon 200 Hz data are only released as event tables). Latency and memory were measured once, on a laptop that was also running other experiments, so treat them as indicative.

@@recommendation
**Not worth a full run as a headline deception-detection result; keep it, if the authors permit, as a context-controlled benchmark and a cautionary case.**
- **Licence first.** Use this dataset in the paper only if the Foucher and Huckauf team give permission (the user's decision). Send them the label-inversion finding at the same time.
- **What the evidence supports.** From eye measures alone, a participant-disjoint classifier reaches AUC about 0.60–0.62 (EER 0.42), clearly above the paper's effects turned into a classifier (0.52–0.55) on pooled trial AUC, but this is weak, depends on bid time, and the per-participant evidence does not pass Holm. Game context and bid time give 0.79–0.81, and the eyes add no reliable gain over them (+0.003 to +0.005 AUC). Transfer between the two setups is near chance.
- **If it is included in the paper,** report the context and bid-time rows next to every eye-only result, and use AUC within bid-position strata as the primary measure. Do not present the context models as lie detection.
- **To try next, if worth the effort:** (1) 1000+ permutations, both within- and between-participant; (2) dyad-clustered CIs in PvP; (3) a direct test of the best model against the strongest paper-effect variants (online, AOI); (4) matching or stratifying on bid position and previous-round outcome before modelling; (5) raw gaze samples (if the authors can share them) for sensor-rate efficiency and for pupil or saccade features, which the event tables do not contain; (6) a pre-registered split so that model selection and evaluation do not share the same CV.
