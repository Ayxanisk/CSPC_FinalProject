# Data

`raw/merge.csv.zip` is the archive supplied for this project. It contains
`merge.csv` with STEAD metadata only. The waveform arrays are not included in
this archive; the official STEAD repository distributes waveform HDF5 files
separately.

Keep the original archive unchanged. The eventual Snakemake workflow should read
the CSV directly from the ZIP or extract it into a generated, ignored directory.
Do not describe this metadata archive as the complete waveform dataset.

Source: [official STEAD repository](https://github.com/smousavi05/STEAD).
