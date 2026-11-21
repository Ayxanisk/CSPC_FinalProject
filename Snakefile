import sys

PYTHON = sys.executable


rule all:
    input:
        "results/tables/dataset_summary.csv",
        "results/tables/rq1_model_metrics.csv",
        "results/tables/rq1_model_parameters.csv",
        "results/tables/rq2_snr_model.csv",
        "results/tables/rq2_fit_metrics.csv",
        "results/tables/software_versions.csv",
        "results/figures/rq1_distance_diagnostics.png",
        "results/figures/rq2_snr_magnitude.png",


rule analyze_stead_metadata:
    input:
        archive="data/raw/merge.csv.zip",
        script="src/analysis.py",
    output:
        "results/tables/dataset_summary.csv",
        "results/tables/rq1_model_metrics.csv",
        "results/tables/rq1_model_parameters.csv",
        "results/tables/rq2_snr_model.csv",
        "results/tables/rq2_fit_metrics.csv",
        "results/tables/software_versions.csv",
        "results/figures/rq1_distance_diagnostics.png",
        "results/figures/rq2_snr_magnitude.png",
    shell:
        "{PYTHON} {input.script} --archive {input.archive}"
