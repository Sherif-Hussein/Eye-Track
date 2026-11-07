# Recognition from looking time at low frame rates: analysis code

Analysis code, plans, cards and summary tables for the paper. No dataset, feature file or camera recording is included; the code rebuilds what it needs from the original deposits.

## Datasets (download from the public sources)
- Delmas et al.: https://osf.io/vygfz (CC BY-NC-ND 4.0)
- Chen et al.: https://osf.io/ske56 and https://osf.io/sk5zq (no licence stated)
- Celniak et al.: https://doi.org/10.5281/zenodo.18525952 (CC BY 4.0)
- Colasanti et al.: https://doi.org/10.5281/zenodo.7194229 (CC BY 4.0)
- Preti and Bolognini: https://doi.org/10.5281/zenodo.21451382 (CC BY 4.0)
- Van der Cruyssen et al.: https://osf.io/p562n (no licence stated)

Place them under `data/`. Some licences forbid derivatives or state none, so nothing derived from the data is redistributed beyond summary results.

## Layout
- `src/edd/`: shared analysis code.
- `scripts/`: analysis scripts by project phase.
- `results/`: for each analysis, its plan, card, scripts and summary tables.
- `paper_tables/`: the tables and figure code of the paper.
- `app/features.py`, `app/protocol.py`: the two files the frame-rate simulation uses.

## Notes
Paths are relative to the repository root. The application itself is not in this repository. The MIT licence covers this code only: it grants no rights to the datasets and no rights under any patent.
