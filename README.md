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
magnitude scale (`ml`). These counts were obtained by streaming the supplied CSV
archive and should be reproduced by the project pipeline before they are used in
the final paper.

## Reproducibility plan

The final repository should include a pinned environment, a Snakemake pipeline
from the raw metadata archive to every analysis table and figure, pytest tests,
and documentation for reproducing the results. The team should implement and
review the analysis code and write the paper in its own words, following the
course handout and the LaTeX template supplied on Moodle.

## Source

The STEAD dataset and its data-access notes are maintained by the original
authors at [smousavi05/STEAD](https://github.com/smousavi05/STEAD). The dataset
paper is Mousavi et al. (2019), “STanford EArthquake Dataset (STEAD): A Global
Data Set of Seismic Signals for AI,” *IEEE Access*,
[doi:10.1109/ACCESS.2019.2947848](https://doi.org/10.1109/ACCESS.2019.2947848).

## Status

This is a planning scaffold. The research questions need team agreement, and
analysis scripts, figures, tests, and the paper still need to be authored and
reviewed by the team.
# CSPC_FinalProject
