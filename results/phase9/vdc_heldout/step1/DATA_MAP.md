# DATA_MAP: Van der Cruyssen, Ben-Shakhar, Pertzov & Verschuere (2024) data, OSF p562n

Task 2.1, step 1 (download and inspection only). No group comparison, AUC, model or any statistic that involves
condition labels was computed. Only descriptive counts, value ranges and structure checks were produced.
Download time: 2026-11-06 02:22:16 UTC (start) to 02:23 UTC (end). Raw files: `data/external/vdc2024/raw/p562n/osfstorage/`
(outside any git-tracked folder; this project folder is not a git repository). Details, sizes and SHA-256 hashes: `ACCESS.md`.
Licence shown on the project: none stated.

## 0. Verdict in one paragraph

The deposited data are **aggregates (path c), plus two partial sources**: (i) the order of fixations by interest area
(no onsets, durations or coordinates) for the four-item display, and (ii) a 100 ms-bin time course of the share of eye-tracker
samples in each area for the 32 trials that contain the crime-related item. There are **no raw gaze samples and no pupil data**,
and there is **no complete fixation report** (the single display has fixation durations, the four-item display does not).
Consequences: the 30, 15 and 10 Hz versions (anti-aliased downsampling) and the 1000 Hz sample-based version are **not
possible**; the registered fallback "(b) features from fixation reports" is only partly available. Our 13 Delmas-style item
features: 5 computable as defined (dwell_1, dwell_2, n_fix, n_visit, first_fix), 5 computable only approximately
(dwell_b1 to dwell_b5, from the 100 ms time course, critical-present trials only), 1 approximately (lat_first, 100 ms
resolution), and 2 not at all (pupil_face, pupil_trial). The authors' six measures for their equal-weight index can all be computed.
Opus must decide how the registered plan is applied to this; the deviations must be reported.

## 1. File inventory (Data folder; all are .xlsx, one sheet each)

| File | Sheet | Rows x cols | Unit of one row | What it is |
|---|---|---|---|---|
| Logboek.xlsx | Blad1 | 155 x 5 | participant | Participant log: ID, Condition label, date, time, exclusion flag |
| RT.xlsx | RT | 8,368 x 24 | single-display trial | Reaction time and key response, single display (AGGREGATE per trial) |
| FirstPhaseParallel_DwellTime.xlsx | same | 38,552 x 14 | trial x area (4 per trial) | Dwell time per area, first phase, four-item display (AGGREGATE) |
| SecondPhaseParallel_DwellTime.xlsx | same | 42,048 x 13 | trial x area | Dwell time per area, second phase (AGGREGATE) |
| Parallel_FixationCount.xlsx | same | 42,048 x 13 | trial x area | Number of fixations per area (AGGREGATE) |
| Parallel_Visits.xlsx | same | 188,886 x 12 | one fixation, in order | Sequence of fixations with the area each fell in (FIXATION ORDER ONLY: no time, duration, x, y) |
| Single_FixationDuration.xlsx | same | 57,708 x 13 | one fixation | Single display: duration of each fixation and whether in the single area ("IA_single") or outside (".") |
| TimeCourse.xlsx | TimeCourseFinal | 231,800 x 24 | trial x 100 ms bin | Share of samples in each area per 100 ms bin, 50 bins (0 to 5,000 ms), 32 trials per participant |

Other files in the project (not data): `Analytic scripts/` (GuiltyNaive.R, GuiltyInformed.R, InformedNaive.R, TimecoursFigure.R: the
authors' own pipeline, very informative), `Materials/` (InstructionsMockCrime.pdf; MockCrimeStudy_deploy.exe, downloaded
but never executed), `Ethics/EthicalApproval.pdf`, `Power analyse ROC.JPG`, and `Additional exploratory analyses/ExploratoryAnalysis_IndividualMeasurements.pdf`.
Note for the preregistration record: this last PDF is a short table of the authors' per-measure AUCs (published results of the
authors, not ours). It was opened while extracting text for the inventory and appeared in the console log; nothing from it was
used, copied or computed on. Opus should know this exists and decide whether it matters.

The 5 example rows of every file, the column list with data types, unique counts and missing counts are in
`inventory_part1.txt` and `inventory_part2.txt` (script `inspect_vdc.py`). Structure checks: `counts_output.txt`,
`design_output.txt`, `trials_output.txt`, `practice_output.txt`.

Reading note: Logboek, RT, Parallel_FixationCount, Parallel_Visits, Single_FixationDuration, SecondPhase and TimeCourse are
"strict" Open XML files that openpyxl lists as having no sheets; `inspect_vdc.py` converts the XML namespaces in memory (raw files untouched).

## 2. Data dictionary

### 2.1 Common columns

| Column | Meaning | Mapping to our variables |
|---|---|---|
| `Participants` (FirstPhase), `RECORDING_SESSION_LABEL` (all others except RT), `Session_Name` (RT) | Participant. In `RECORDING_SESSION_LABEL` it is a text label: a number plus an optional suffix (`_G`, `_I`, `_N`, `_n`, `_1`, `_2`, `_3`, or none), e.g. `100_I`. The numeric part is the participant ID (1 to 155) | participant id = the leading digits. Suffixes agree with the Logboek condition (G=guilty; I=informed; N/n=naive; _1=guilty; _2=informed; _3=naive; blank = any), so session labels leak the condition and must not be used as a feature |
| `TRIAL_INDEX` | Presentation order in the recording (includes practice rows) | trial order; trials of a participant are NOT stored in presentation order in the files (sorted by `trial`) |
| `trial` | Stimulus (trial) number 1 to 64 in the authors' trial book, not presentation order. Practice rows are also labelled 64 (see 3.3). `UNDEFINED` occurs in 4 rows of participant 74 (excluded) | trial id |
| `topleft`, `topright`, `bottomleft`, `bottomright` | Image file shown in each quadrant of the four-item display | item identity by position |
| `single` | Image file shown in the single display | single-display item |
| `IA_ID`, `IA_LABEL` | Area 1 TopLeft, 2 TopRight, 3 BottomLeft, 4 BottomRight | position |
| `criticalplace` | Position (1 to 4) of the crime-related image in the four-item display; 0 = no crime-related image in that display | item type by position (crime-related vs foil). Verified: the image at that position is always the file without a trailing digit |
| `fourorcenter` | Trial type (my reading, verified against the images in all 8,320 RT rows): 0 = single display shows a foil, four-item display has no crime-related image (24 trials per participant); 1 = single foil, four-item display contains the crime-related image (24); 2 = single display shows the crime-related image, four-item display has none (8); 3 = single crime-related and four-item contains it (8). The authors treat 2 and 3 as "crime-related" on the single display (RT, fixation duration) | display-type/trial-type code |
| `Exclusions` (FirstPhase only) | Constant 0, carries no information | none |

Item identity: eight categories (Face, Backpack, Gloves, Phone, Purse, Safe, Scarf, Wallet), 8 trials each = 64 trials. In each
category the crime-related image is the file with no number (`Face.jpg`); foils are numbered (`Face1.jpg` ...). The four-item
display is shown for 5 s (bins 0 to 49 x 100 ms). In 32 trials it contains the crime-related image at each of the 4 positions
(8 trials per position), in 32 trials it contains only foils (criticalplace = 0).

### 2.2 File-specific

| File.column | Meaning | Mapping |
|---|---|---|
| FirstPhase `IA_DWELL_TIME` | Milliseconds in the area during the first phase of the four-item display (observed 0 to 952; authors: 1 to 1,000 ms) | our `dwell_1` (we use 200 to 1,000 ms of fixation samples; the VdC window and fixation definition are the authors') |
| SecondPhase `IA_DWELL_TIME` | Milliseconds in the area, second phase (observed 0 to 4,000; authors: 1,000 to 5,000 ms) | our `dwell_2` (same 1,000 to 5,000 window) |
| FixationCount `IA_FIXATION_COUNT` | Fixations in the area (observed 0 to 17) | our `n_fix` |
| Visits `CURRENT_FIX_INTEREST_AREA_LABEL`/`_ID` | Area of each fixation in order; `.` = fixation outside the four areas | our `n_visit` (runs of consecutive same-area fixations) and `first_fix` (first fixation in an area) |
| Single `CURRENT_FIX_DURATION` | Fixation duration in ms (1 to 2,000) for each fixation on the single display; area label `IA_single` or `.` | authors' mean fixation duration (single display). Not one of our 13 features |
| Single `CURRENT_FIX_INTEREST_AREA_LABEL` | `IA_single` or `.` | the authors drop `.` |
| RT `RT_press` | Reaction time in ms on the single display (324 to 1,975); 641 of 8,368 rows empty (no response) | authors' RT measure. Not one of our 13 |
| RT `Key_response`, `correctresponse`, `correct`, `correcttrial` | Key pressed (p/q; `G` appears in 593 rows, probably the practice/training key), correct answer, correctness. `correcttrial` is 1 in half of trials (4,160 of 8,320) | response accuracy |
| RT `Trial_Index` | Presentation index; Equals TRIAL_INDEX in the eye files (checked for the 130 retained participants who have RT) | key for removing practice rows |
| RT `Trial_Recycled`, `numOfCorrect`, `*_training`, `TotallCorrect`, `filter_$`, `Critical_items*` | Training/bookkeeping and SPSS filter variables | none |
| TimeCourse `BIN_INDEX` | 100 ms bin, 0 to 49 (0 to 5,000 ms after display onset; bin width from the authors' TimecoursFigure.R) | time axis |
| TimeCourse `AVERAGE_IA_k_SAMPLE_COUNT_%` | Proportion (0 to 1, steps of 0.01) of the samples in the bin that fell in area k. The 0.01 steps suggest 100 samples per 100 ms, i.e. 1000 Hz recording [inferred, unverified] | per-area gaze occupancy at 10 Hz bin resolution |
| TimeCourse `other` | Proportion outside all four areas | background |
| TimeCourse `crimerelated`, `control` | Proportion on the crime-related area; for `control` the mean over the 3 foil areas (observed: bin 1 control = (0+0+0.61)/3 = 0.203) | critical vs foil occupancy |
| TimeCourse rows | Only the 32 four-display trials that contain the crime-related image (fourorcenter 1 or 3; criticalplace 1 to 4) | |

### 2.3 Condition and exclusion coding (Logboek.xlsx)

| Column | Meaning |
|---|---|
| first (blank header) | participant ID 1 to 155 |
| `Condition` | `Guilty`, `Informed` (informed innocent), `Naive` (naive innocent) |
| `Date`, `Time` | session date and time (5 missing each) |
| `Exclusions after other criteria` | 1 = excluded (22 participants), 0 = kept |

The authors' R scripts keep `Exclusions after other criteria != 1`, and also hard-code the same 22 IDs
(13, 14, 15, 20, 24, 35, 71, 74, 77, 88, 95, 104, 110, 114, 124, 127, 135, 138, 142, 148, 149, 150) in the data-file filters.
The two lists are identical. Their `TimecoursFigure.R` lists 42 guilty, 45 informed and 46 naive IDs; these agree 100 percent with
the Logboek labels for all 133 retained participants. The Logboek has no other flag (e.g. no reason for exclusion).

## 3. Counts (descriptive)

### 3.1 Participants per condition

| | Guilty | Informed | Naive | Total |
|---|---|---|---|---|
| Logboek, all participants | 53 | 50 | 52 | 155 |
| Logboek, flag = 0 (retained) | 42 | 45 | 46 | **133** (matches the expected 42/45/46) |

### 3.2 Which participants are in which file

| File | Distinct participants | Excluded participants present |
|---|---|---|
| Logboek | 155 | 22 flagged |
| FirstPhaseParallel_DwellTime | 134 | only #24 |
| SecondPhaseParallel_DwellTime, Parallel_FixationCount, Parallel_Visits, Single_FixationDuration, TimeCourse | 146 | 13 of the 22 present: 13, 15, 24, 71, 74, 77, 88, 124, 135, 142, 148, 149, 150 (absent: 14, 20, 35, 95, 104, 110, 114, 127, 138) |
| RT | 133 | none (only the retained participants) |

All 133 retained participants appear in every file. Excluded participants are NOT flagged inside the eye-tracking files (the
FirstPhase `Exclusions` column is all 0); the only flag is in the Logboek, so the analysis set must be defined by the Logboek.
Counts per condition after applying the Logboek flag are identical in every file (42/45/46).

### 3.3 Trials per participant

- Expected: 64. Rows per participant are larger because of practice rows: TRIAL_INDEX count per participant is 69, 74, 79 or 84
  = 64 + 5, 10, 15 or 20 practice rows (a practice trial repeated in blocks of 5). Practice rows are labelled `trial` = 64
  and have the same stimulus configuration as the real trial 64 (Purse set, fourorcenter 2, criticalplace 0), so they cannot
  be told apart by stimulus. They can be removed by keeping only the TRIAL_INDEX values listed in RT.xlsx for the same
  participant (verified: this leaves exactly 64 trials x 4 areas for 129 of the 133 retained participants, in all five trial files).
  Equivalent rule: of the rows labelled 64, keep only the one with the highest TRIAL_INDEX.
  The authors' own script drops all `trial == 64` rows from the single-display fixation-duration data (this removes the real
  trial 64 as well; no other file is trimmed this way). Note this for the replication of the authors' index.
- **Exceptions among the 133 retained:**
  - **Participant 23 (Naive, kept by the authors)** has eye-tracking data for only 44 of the 64 trials (and 18 of the 32 time-course
    trials), but RT for 64 trials (8 empty). So the "64 trials per participant" statement does not hold for this participant.
  - **Participants 17 (Naive), 31 (Guilty) and 43 (Informed)** have no RT data (RT rows exist but are empty), so RT-based measures
    are missing for them (the authors add them as NA); their eye-tracking data are complete (64 trials). For them the RT-based rule
    for removing practice rows is not available; use the "highest TRIAL_INDEX among trial = 64" rule.
- TimeCourse: 32 trials per participant (the crime-related-present trials) x 50 bins = 1,600 rows for 144 of 146 participants
  (pid 23 has 18 trials; one excluded participant has 10).
- RT: 64 rows per participant for 130 participants; empty responses per participant range 0 to 17.
- Fixation rows per participant: see `trials_output.txt` (counts there use the RT-based practice filter and show 0 for
  participants 17, 31, 43 only because they have no RT Trial_Index to filter by).

## 4. Feasibility of the registered analyses

### 4.1 Which path applies

**Path (c), aggregates**, with the partial additions described in section 0. Not (a): no x, y, pupil per sample anywhere.
Not (b) in the registered sense: the only per-fixation records are the ordered area labels for the four-item display (no onsets,
durations, coordinates) and fixation durations for the single display (no area beyond in/out). The 30, 15, 10 Hz versions and
the 1000 Hz sample-based version are **not possible**. The only time-resolved information is the 100 ms-bin share of samples
per area (equivalent to a 10 Hz occupancy series, but computed by the authors from the original recording; it is not a
downsampling of raw samples).

### 4.2 Our 13 item features (Delmas-style, 30 Hz set; definitions from `src/edd/delmas_raw.py`)

Unit in our pipeline: per quadrant per trial, then averaged per item and z-scored per participant. The VdC data have the same
structure (4 areas per trial), but only in the 32 crime-related-present trials is there one crime-related and three foil images.

| # | Feature | Available? | Source and caveat |
|---|---|---|---|
| 1 | dwell_1 (200 to 1,000 ms) | Yes, with a different window | FirstPhase dwell (authors' first phase, 1 to 1,000 ms; fixation-based dwell). Window start differs (1 vs 200 ms) |
| 2 | dwell_2 (1,000 to 5,000 ms) | Yes | SecondPhase dwell |
| 3 | n_fix | Yes | FixationCount (window presumably 0 to 5,000 ms; to confirm) |
| 4 | n_visit | Yes | Parallel_Visits: collapse consecutive same-area fixations. `.` (outside) fixations should break runs as in our code; the authors' R code treats them differently (see open questions) |
| 5 | dwell_b1 to dwell_b5 (0 to 500, 500 to 1000, 1000 to 2000, 2000 to 3500, 3500 to 5000 ms) | Approximately | Sum of the 100 ms-bin sample shares from TimeCourse (all bin edges fall on 100 ms boundaries). Sample-based occupancy, not fixation-restricted; only the 32 crime-related-present trials (the 32 foil-only trials have no bins). Not available for foil-only trials |
| 6 | first_fix | Yes | First row of a trial in Parallel_Visits with an area label (not `.`); requires the first rows to start at display onset (to confirm) |
| 7 | lat_first | Approximately | No onsets in Visits. From TimeCourse: first 100 ms bin with a non-zero share in the area (100 ms resolution, only 32 trials). Our value is the first fixation onset in ms |
| 8 | pupil_face | **No** | No pupil data in any file |
| 9 | pupil_trial | **No** | No pupil data |
| (14) | msacc_rate | No (and not in the 13) | Needs at least 500 Hz samples |

So: 5 features as defined (dwell_1 with a window difference, dwell_2, n_fix, n_visit, first_fix), 6 approximately
(dwell_b1 to b5 and lat_first, from the 100 ms time course, 32 trials only), 2 impossible (both pupil features).

### 4.3 The authors' six measures (equal-weight index)

All six can be computed from the data exactly as the authors do (their R scripts are deposited and give the exact recipe):
number of visits (Parallel_Visits), fixation count (Parallel_FixationCount), dwell 1 to 1,000 ms (FirstPhase), dwell 1,000 to 5,000 ms
(SecondPhase), mean fixation duration on the single display (Single_FixationDuration), and RT (RT.xlsx; missing for participants 17, 31, 43).
Authors' recipe (from GuiltyNaive.R): per participant, z-score each measure over that participant's trials (dwell, count and visit
measures over the trials with a crime-related image in the four-item display, `criticalplace != 0`; RT over all 64 trials; fixation
duration over fixations), average the z over the crime-related cases, then average the six z-scores with equal weights
(row-wise mean, `na.rm = TRUE`). Features of their code that matter for a replication: the z-scores of RT, second-phase dwell,
fixation count and visits are multiplied by -1, whereas those of first-phase dwell and fixation duration are not; the foil visit
count is divided by 3; the single-display fixation file drops `trial == 64`.

## 5. Open questions (for Opus)

1. **Registered path.** The registration names (a) raw samples or (b) fixation reports. What exists is (c) plus partial sources. Which parts of
   the registered plan can still be run exactly (H1 to H5 with which feature set), and how are the deviations worded? Does
   the registration allow a feature set restricted to those computable here? Reading the registration text is needed.
2. **No downsampling, no 1000 Hz sample version.** The OSF update added 15/10 Hz descriptive versions; these cannot be produced from these data. Only the
   TimeCourse (100 ms bins, 10 Hz-equivalent occupancy) is a related time-resolved view.
3. **Pupil features** (2 of 13) are impossible. The 13-feature model cannot be run unchanged and the frozen Delmas 30 Hz model (H4) cannot be applied unchanged
   (it needs all 13 inputs and our sample-based definitions). How should H4 be handled (e.g. a pupil-free refit as a labelled deviation)?
4. **Dwell windows.** Authors' phases are 1 to 1,000 and 1,000 to 5,000 ms and fixation-based; ours are 200 to 1,000 and 1,000 to 5,000. The first-phase dwell maximum is 952 ms.
5. **Per-item structure.** In 32 trials each participant sees the crime-related image among 3 foils (positions balanced, 8 per position); in the other 32 only foils. Which of these are the "items" for our
   item-level features and z-scoring? The authors compute z-scores over all trials (first-phase etc. restricted to criticalplace != 0 trials).
6. **TimeCourse** covers only the 32 crime-related-present trials and is based on samples, so dwell_b1 to b5 and lat_first are not identical in kind to our fixation-restricted features.
7. **Visit rule.** The authors' R code sets `.` rows to NA before collapsing duplicates; the collapse test ignores NA (`all(..., na.rm=TRUE)`), which makes `.` rows behave differently from our rule (background fixation breaks a run). Which rule do we follow for n_visit?
8. **Start of the Visits sequence.** Is the first fixation of each trial after display onset or the preceding fixation cross (`.` first rows)? Affects first_fix.
9. **Practice rows.** Adopted rule: keep the real trial 64 (highest TRIAL_INDEX among rows labelled 64). The authors drop all 64s in the fixation-duration analysis only.
10. **Participant 23** has 44 of 64 eye-tracking trials (kept by the authors, Naive). Include (with fewer trials) or follow the exact "64 trials" rule? Participants 17, 31, 43 have no RT.
11. **Condition leakage in labels.** Session labels carry condition suffixes (`_G`, `_I`, `_N`, `_1`, `_2`, `_3`); these must be stripped before any feature or fold construction. The authors' lists and the Logboek agree.
12. **Excluded participants** are present (without a flag) in 6 of the 7 data files; the Logboek flag must be applied. The Logboek has no exclusion reasons.
13. **Sampling rate** of the original recording is not stated in the data (1000 Hz inferred from 0.01 steps in the time course; to verify in the published paper).
14. **Licence.** None stated on the project; the project page carries no CC licence. Use is for research; we do not redistribute the files.
15. **Authors' exploratory PDF** with per-measure AUCs (see section 1): disclosed above.
16. The registration/OSF project date_modified shows 2022-12-05 for the p562n node; file modification dates are in `ACCESS.md`. The author update mentioned in the task is not visible as a node-level change in the API; check whether the registration text, not the data project, was updated.

## 6. Files written (this step)

`ACCESS.md`, `DATA_MAP.md`, `download_vdc.py`, `download_manifest.json`, `osf_listing.json`, `inspect_vdc.py`, `counts_vdc.py`, `design_vdc.py`, `trials_vdc.py`,
`practice_vdc.py`, and their outputs `inventory_part1.txt`, `inventory_part2.txt`, `counts_output.txt`, `design_output.txt`, `trials_output.txt`, `practice_output.txt`
(all in `results/phase9/vdc_heldout/step1/`).
