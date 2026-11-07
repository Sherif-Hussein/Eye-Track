# Task 2.3: eye-movement figures for Paper A (card)

Plan: `docs/PLAN.md` Task 2.3. Script: `make_figures.py` (this folder; one run, one thread, about 25 s). Only numbers are used; no camera images are read or saved. Wording: recognition terms only ("known item", "unknown item"). Style: set up as in `scripts/phase5_figures.py` (Arial 8 pt, same palette, same rcParams, 5.5 in width, PDF with embedded fonts plus PNG at 300 dpi).

Files per figure: `<name>.pdf`, `<name>.png`, and the CSV (or CSVs) it is drawn from, plus a small check/selection JSON.

## (a) fig_trajectory: one trial at 1000 Hz and 30 Hz

**Source files.** Delmas et al. raw EyeLink files `data/delmas/raw/data_raw/Sj_XX/` read by `scripts/phase6_trackC_c2.py: load_halves` (same trial cutting as `src/edd/delmas_raw.py`); 30 Hz path made by `edd.delmas_raw.degrade(x, y, p, 30)` (anti-aliased polyphase FIR, as in `results/phase9/low_rate/s1_extract.py`); layout from `delmas_raw.QUAD_RECTS` (four 300 x 400 px items at 34.99 px/deg, screen centre (960, 540)). The selection scanned the 1000 Hz arrays from an earlier scratch copy of `load_halves` output (`.../scratchpad/trackC/c2_raw/p<id>.npz`); the chosen participant was then re-read from the EDF files and the arrays compared: identical (`cache_equals_raw_edf = true`).

**Selection rule (fixed before any plot was drawn; `select_trial()`).**
1. Population: the 75 analysed participants (the same 75 as Task 2.2), all their trials.
2. Eligible trial: the four-item display contains a known item (image id 57-64, the eight famous faces, as in `s1_extract.py`; this matches the authors' `face = 1` flag except for 12 of 4,800 book rows, which are not used here) and at least 90% of the 5,000 display samples (0-5 s after onset) have valid x and y.
3. Trial dwell = number of 1000 Hz samples (ms) in the 0-5 s display window that are valid and fall inside any of the four item rectangles.
4. Participant total dwell = mean of trial dwell over that participant's eligible trials (mean, not sum, so participants with fewer eligible trials are not penalised; this is my reading of "total dwell"). Participants sorted by this value; the median participant is rank (n+1)/2 = 38 of 75.
5. That participant's eligible trials sorted by trial dwell; the median trial is rank (m+1)/2 rounded down for even m, here rank 16 of 32.

**Chosen.** Participant 85 (code, no name), trial 15 of the session (1-based; index 14), known item in the lower-right slot (image 64), unknown items 37, 39, 48.

**Key numbers.**
- 2,400 trials with a known item, 2,355 eligible (at least 90% valid); all 75 participants have eligible trials.
- Median participant: mean trial dwell 4,588.6 ms (rank 38 of 75); the participant's 32 eligible trials range 4,227-4,711 ms; chosen trial dwell 4,598 ms, valid share 100%.
- 1000 Hz: 5,000 samples, all valid; dwell on items 4,598 ms (per slot upper-left, upper-right, lower-left, lower-right known: 1,112, 1,473, 1,119, 894 ms).
- 30 Hz: 150 samples in 0-5 s, all valid; dwell on items 4,600 ms (1,100, 1,500, 1,100, 900 ms). The dwell differs by 2 ms (0.04%) between the two rates for this trial.
- Gaze stays within -11.0 to 11.3 deg horizontally and -7.9 to 9.5 deg vertically (item areas are 8.6 x 11.4 deg).

**CSVs.** `fig_trajectory.csv` (rate_hz, sample, t_ms, x_px, y_px, x_deg, y_deg, valid; 5,000 + 150 rows), `fig_trajectory_items.csv` (the four item rectangles in px and deg, image id, known/unknown), `fig_trajectory_selection.json` (all selection numbers).

**Description (for the paper).** The figure shows the gaze path of one 5-second trial over the four-item display, drawn at 1000 Hz (left) and after anti-aliased downsampling to 30 Hz (right); axes are degrees of visual angle from the screen centre, the item areas are drawn as boxes (the known item in orange, the three unknown items in grey), and path colour darkens with time from 0 to 5 s. The trial is the median-dwell trial of the median-dwell participant among trials that contained a known item and had at least 90% valid samples (selected by a fixed rule before plotting). The viewer starts near the centre, moves to the upper-left item and then visits the other items in turn, spending 4.60 s of the 5 s on the items; the 30 Hz path (150 samples) follows the same route as the 1000 Hz path (5,000 samples) and gives the same dwell to within 2 ms, but shows only coarse samples during the fast movements between items.

## (b) fig_accuracy_targets: webcam gaze against the five validation targets

**Source.** Recording R4 of Paper A Fig. 2d (Table S17): session folder `app/data/test06_faces_I_emeet_glasses/` (`frames.csv`, `events.csv`, `meta.json`; numeric landmark-derived columns only; the `report/` folder and any video were not touched). Found by reading `scripts/phase5_figures.py` (panel d reads `results/app_pilot/paper_summary/fig3d_samples.csv`, `fig3d_targets.csv`), which are written by `results/app_pilot/paper_summary/export_fig3de.py` from that session. I re-derived the same numbers from the session logs with the same rule (first validation of the session, first 0.7 s of each target not scored, live gaze while a face was found; degrees from the screen centre at the session's 138.9 px/deg) and compared with the stored CSVs.

**Reproduction check (done before drawing).** 198 samples (40, 40, 40, 38, 40 per target); max absolute difference to the stored sample and target tables 5e-5 deg (rounding to four decimals). Mean of the five offsets: 1.3343 deg (new 1.33429, stored table 1.33430, `emeet_sessions.csv` accuracy_deg 1.33438), the number the existing panel prints ("mean 1.33 deg"). Reproduced.

**Key numbers.** Offsets (median gaze to target): lower-left 0.58, centre 1.28, lower-right 1.82, upper-right 1.94, upper-left 1.05 deg (target index 0-4 in `fig_accuracy_targets_targets.csv`); mean 1.33 deg. Sample scatter around each target's median: root mean square 1.30-1.74 deg. Screen 3840 x 2160 px = 27.6 x 15.6 deg at this recording's calibration scale (viewing distance 60.9 cm).

**CSVs.** `fig_accuracy_targets_samples.csv` (target, x_deg, y_deg), `fig_accuracy_targets_targets.csv` (target position, median gaze, offset, n samples), `fig_accuracy_targets_check.json`.

**Description (for the paper).** The figure shows, in degrees of visual angle from the screen centre, the gaze estimated from a laptop webcam at 30 Hz while one person looked at five known on-screen validation targets (orange crosses; 198 samples in blue, 38-40 per target). For each target the median gaze is a black dot joined to the target by a line, labelled with the distance between them; these offsets are 0.6 to 1.9 deg and average 1.33 deg. The samples form a cloud about 1.3 to 1.7 deg (root mean square) around each median, so both a constant offset and sample-to-sample noise contribute to the webcam error.

## (c) fig_drift: offset over the session

**Source.** The same session (`trials.csv`, `practice.csv`, `events.csv`; the logged `drift_deg` = distance between gaze and the fixation cross before each trial, as the app computes it) and `app/config.py` (`drift_recheck_deg = 4.0`, `auto_recheck_trials = 2`; no `config.json` override; the session `meta.json` also records 4.0). The numbers for the 18 scored trials equal `fig3e_drift.csv` (max difference 5e-5 deg), the table behind Fig. 2e. Logs for this existed; nothing was substituted. Per-trial offsets only: the app logs no continuous offset trace outside these fixation checks.

**What is drawn.** Offset at the fixation cross for the two practice trials (open markers, P1, P2) and the 18 scored trials (filled), in the order of the session; median of the 18 scored trials (3.04 deg); the re-check limit (4 deg); the automatic re-check and recalibration after P2 (event `auto_recheck`, 110.2 s, practice offsets 5.9 and 6.0 deg).

**Key numbers.** Practice offsets 5.91 and 5.99 deg (both above the 4 deg limit, two in a row, so the app started a re-check and recalibrated). After that: scored-trial median 3.04 deg (median of all 20 trials 3.22), range 0.33-4.98, 90th percentile 4.11. Two scored trials exceed 4 deg (trial 7: 4.37; trial 12: 4.98), not in a row, so no further re-check; trials 13 and 14 are 3.99 and 3.97. Session span 91.5-287.1 s (about 3.3 min).

**CSV.** `fig_drift.csv` (kind, trial_in_kind, time_s, drift_deg, calibration_segment, fixation_ok, order, above_recheck_limit), `fig_drift_check.json`.

**Description (for the paper).** The figure shows how far the estimated gaze was from the fixation cross at the start of each trial during one 20-trial webcam session (offset in degrees, trials in order). The two practice trials, run on the first calibration, had offsets of about 6 deg, above the 4 deg re-check limit, which triggered the automatic re-check and a new calibration. After it the offset stayed between 0.3 and 5.0 deg with a median of 3.0 deg and crossed the limit on two of the 18 scored trials, not in succession, so no further re-check was needed.

## Deviations and notes
- The 1000 Hz arrays for the selection came from a scratch copy of the raw cut-outs made in an earlier session; the chosen participant was verified against the EDF files (identical). The rest were not re-read from the EDF.
- "Total dwell" for ranking participants is the mean over their eligible trials (interpretation, stated above).
- The existing Fig. 2d code draws the screen outline as 28.6 x 16.1 deg (134.3 px/deg, the value of another recording, test05); this session's own scale is 138.9 px/deg (27.6 x 15.6 deg). Panel d's offsets and samples are not affected (they use 138.9). My figure does not draw the screen outline. Worth fixing if the old panel is kept.
- (c) includes the two practice trials, which the old Fig. 2e left out; the scored trials are identical to it.
- Colour bar in (a) is a second axes object (colour key for time); every plotting panel is a single axis.
