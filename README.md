# STEAD Metadata Project

This project will study physical relationships in the metadata supplied with the
STanford EArthquake Dataset (STEAD). The current data file is
`data/raw/merge.csv.zip`.

## Current scope

The supplied archive contains a metadata CSV, not the waveform HDF5 file. It
supports analyses of event distance, P/S arrival picks, magnitude, depth, and
signal-to-noise ratio (SNR). It does not support FFT or waveform-based
earthquake/noise classification. The proposed questions and analysis plan are in
[`docs/research_questions.md`](docs/research_questions.md).

## Dataset snapshot

The archive has 1,265,657 metadata rows: 1,030,231 local-earthquake traces and
235,426 noise traces. The noise rows do not have earthquake magnitude, distance,
arrival-pick, or SNR values. Among earthquake rows, 724,824 use the local
magnitude scale (`ml`). The analysis applies additional complete-case filters;
its final sample counts are written to `results/tables/dataset_summary.csv`.

## Reproducing the analysis

The requested scientific packages and Snakemake are pinned in
[`requirements.txt`](requirements.txt). From the repository root, install them
in a Python 3.11 environment with `python -m venv .venv` and
`.venv/bin/python -m pip install -r requirements.txt`, then run
`.venv/bin/snakemake --cores 1`. The workspace points Pylance at this `.venv`.
The workflow writes tables to `results/tables/` and figures to
`results/figures/`.
If the user's cache directory is read-only, set `XDG_CACHE_HOME` to a writable
location before running Snakemake (for example, `XDG_CACHE_HOME=/tmp/snakemake-cache`).

## Current analysis snapshot

- RQ1: the depth-aware model achieved held-out MAE 4.871 km, RMSE 9.663 km, and
  R² 0.9606. The linear timing-only model achieved 5.823 km, 11.281 km, and
  0.9464 respectively.
- RQ2: after event-level aggregation and adjustment for distance and depth, the
  fitted ML coefficient was +8.659 dB per magnitude unit (95% CI 8.613 to
  8.705; model-based p < 0.001). The event-level model R² was 0.349.

The analysis was rerun with the requested NumPy, SciPy, pandas, and Matplotlib
pins. Actual versions are recorded in `results/tables/software_versions.csv`.

## Source

The project page supplied by the team is
[Stanford Earthquake Dataset (STEAD) on Kaggle](https://www.kaggle.com/datasets/isevilla/stanford-earthquake-dataset-stead?resource=download).
The STEAD dataset and its data-access notes are maintained by the original
authors at [smousavi05/STEAD](https://github.com/smousavi05/STEAD). The dataset
paper is Mousavi et al. (2019), “STanford EArthquake Dataset (STEAD): A Global
Data Set of Seismic Signals for AI,” *IEEE Access*,
[doi:10.1109/ACCESS.2019.2947848](https://doi.org/10.1109/ACCESS.2019.2947848).

## Status

The research questions, analysis code, initial results, figures, and paper
draft are in place. The public GitHub URL, team author/contribution details,
pytest tests, and any further revisions remain to be completed.
# CSPC_FinalProject
