"""Reproducible metadata analyses for the STEAD capstone project.

The supplied archive contains a CSV of metadata, not waveform arrays. This
script therefore analyzes arrival-pick metadata and SNR labels; it does not run
FFT or waveform classification.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

# Matplotlib's default user config directory is read-only in some managed
# environments. Keep its font/cache files in the project workspace instead.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import curve_fit


DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "merge.csv.zip"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
SAMPLE_RATE_HZ = 100.0
RANDOM_SEED = 20261121
TEST_FRACTION = 0.20
PLOT_SAMPLE_SIZE = 8_000

REQUIRED_COLUMNS = [
    "trace_category",
    "source_id",
    "source_magnitude",
    "source_magnitude_type",
    "source_distance_km",
    "source_depth_km",
    "snr_db",
    "p_arrival_sample",
    "s_arrival_sample",
]


def load_metadata(archive_path: Path = DEFAULT_ARCHIVE) -> pd.DataFrame:
    """Read the metadata CSV from the supplied ZIP archive."""
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Metadata archive not found: {archive_path}")

    metadata = pd.read_csv(
        archive_path,
        compression="zip" if archive_path.suffix.lower() == ".zip" else "infer",
        usecols=lambda column: column in REQUIRED_COLUMNS,
        low_memory=False,
    )
    missing = sorted(set(REQUIRED_COLUMNS) - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    metadata["source_id"] = metadata["source_id"].astype("string").str.strip()
    metadata["trace_category"] = metadata["trace_category"].astype("string").str.strip()
    metadata["source_magnitude_type"] = (
        metadata["source_magnitude_type"].astype("string").str.strip().str.lower()
    )
    return metadata


def _numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy with the requested columns converted to numeric values."""
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def prepare_distance_data(metadata: pd.DataFrame) -> pd.DataFrame:
    """Prepare valid earthquake picks for RQ1."""
    numeric = [
        "source_distance_km",
        "source_depth_km",
        "p_arrival_sample",
        "s_arrival_sample",
    ]
    frame = _numeric_columns(metadata, numeric)
    valid_id = frame["source_id"].notna() & frame["source_id"].ne("")
    values = frame[numeric].to_numpy(dtype=float)
    finite_values = np.isfinite(values).all(axis=1)

    mask = (
        frame["trace_category"].eq("earthquake_local").to_numpy()
        & valid_id.to_numpy()
        & finite_values
        & frame["source_distance_km"].ge(0).to_numpy()
        & frame["p_arrival_sample"].ge(0).to_numpy()
        & frame["s_arrival_sample"].le(6_000).to_numpy()
        & frame["s_arrival_sample"].gt(frame["p_arrival_sample"]).to_numpy()
    )
    selected = frame.loc[
        mask,
        [
            "source_id",
            "source_distance_km",
            "source_depth_km",
            "p_arrival_sample",
            "s_arrival_sample",
        ],
    ].copy()
    selected["sp_time_s"] = (
        selected["s_arrival_sample"] - selected["p_arrival_sample"]
    ) / SAMPLE_RATE_HZ
    return selected.reset_index(drop=True)


def linear_distance_model(sp_time_s: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Predict source distance from S-P time."""
    return intercept + slope * sp_time_s


def quadratic_distance_model(
    sp_time_s: np.ndarray,
    intercept: float,
    linear_slope: float,
    quadratic_slope: float,
) -> np.ndarray:
    """Allow curvature in the S-P time-to-distance relation."""
    return intercept + linear_slope * sp_time_s + quadratic_slope * sp_time_s**2


def depth_distance_model(
    inputs: tuple[np.ndarray, np.ndarray],
    intercept: float,
    sp_slope: float,
    depth_slope: float,
) -> np.ndarray:
    """Predict source distance from S-P time and source depth."""
    sp_time_s, depth_km = inputs
    return intercept + sp_slope * sp_time_s + depth_slope * depth_km


def _regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    r_squared = 1.0 - np.sum(residual**2) / denominator if denominator else np.nan
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r_squared": float(r_squared),
    }


def _group_train_test_split(
    data: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by earthquake ID so traces from one event cannot cross the split."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    groups = data["source_id"].drop_duplicates().to_numpy()
    if len(groups) < 2:
        raise ValueError("At least two distinct source_id values are needed")

    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(groups)
    n_test = min(max(1, int(round(len(shuffled) * test_fraction))), len(shuffled) - 1)
    test_groups = shuffled[:n_test]
    test_mask = data["source_id"].isin(test_groups)
    return data.loc[~test_mask].copy(), data.loc[test_mask].copy()


def analyse_distance(
    distance_data: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Fit linear, quadratic, and depth-aware RQ1 models."""
    train, test = _group_train_test_split(distance_data, test_fraction, random_seed)
    x_train = train["sp_time_s"].to_numpy(dtype=float)
    z_train = train["source_depth_km"].to_numpy(dtype=float)
    y_train = train["source_distance_km"].to_numpy(dtype=float)
    x_test = test["sp_time_s"].to_numpy(dtype=float)
    z_test = test["source_depth_km"].to_numpy(dtype=float)
    y_test = test["source_distance_km"].to_numpy(dtype=float)

    linear_parameters, _ = curve_fit(
        linear_distance_model,
        x_train,
        y_train,
        p0=(0.0, 10.0),
        maxfev=10_000,
    )
    quadratic_parameters, _ = curve_fit(
        quadratic_distance_model,
        x_train,
        y_train,
        p0=(float(linear_parameters[0]), float(linear_parameters[1]), 0.0),
        maxfev=10_000,
    )
    depth_parameters, _ = curve_fit(
        depth_distance_model,
        (x_train, z_train),
        y_train,
        p0=(float(linear_parameters[0]), float(linear_parameters[1]), 0.0),
        maxfev=10_000,
    )

    predictions = test.copy()
    predictions["predicted_baseline_km"] = linear_distance_model(x_test, *linear_parameters)
    predictions["predicted_quadratic_km"] = quadratic_distance_model(
        x_test, *quadratic_parameters
    )
    predictions["predicted_depth_aware_km"] = depth_distance_model(
        (x_test, z_test), *depth_parameters
    )

    metric_rows = []
    for model_name, prediction_column in [
        ("baseline_sp_only", "predicted_baseline_km"),
        ("quadratic_sp_only", "predicted_quadratic_km"),
        ("depth_aware", "predicted_depth_aware_km"),
    ]:
        metrics = _regression_metrics(y_test, predictions[prediction_column].to_numpy())
        metric_rows.append(
            {
                "model": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_events": train["source_id"].nunique(),
                "test_events": test["source_id"].nunique(),
                "mae_km": metrics["mae"],
                "rmse_km": metrics["rmse"],
                "r_squared": metrics["r_squared"],
            }
        )

    parameter_rows = [
        {
            "model": "baseline_sp_only",
            "intercept": linear_parameters[0],
            "sp_slope": linear_parameters[1],
            "quadratic_sp_slope": np.nan,
            "depth_slope": np.nan,
        },
        {
            "model": "quadratic_sp_only",
            "intercept": quadratic_parameters[0],
            "sp_slope": quadratic_parameters[1],
            "quadratic_sp_slope": quadratic_parameters[2],
            "depth_slope": np.nan,
        },
        {
            "model": "depth_aware",
            "intercept": depth_parameters[0],
            "sp_slope": depth_parameters[1],
            "quadratic_sp_slope": np.nan,
            "depth_slope": depth_parameters[2],
        },
    ]

    return {
        "metrics": pd.DataFrame(metric_rows),
        "parameters": pd.DataFrame(parameter_rows),
        "test_data": predictions,
        "median_depth_km": float(train["source_depth_km"].median()),
        "linear_parameters": linear_parameters,
        "quadratic_parameters": quadratic_parameters,
        "depth_parameters": depth_parameters,
    }


def _median_component_snr(value: object) -> float:
    """Parse a STEAD three-component SNR string and return its finite median."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    values = np.fromstring(text.strip("[]()").replace(",", " "), sep=" ")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def prepare_snr_data(metadata: pd.DataFrame) -> pd.DataFrame:
    """Prepare ML-magnitude earthquake traces for RQ2."""
    numeric = ["source_magnitude", "source_distance_km", "source_depth_km"]
    frame = _numeric_columns(metadata, numeric)
    frame["snr_trace_median_db"] = frame["snr_db"].map(_median_component_snr)
    valid_id = frame["source_id"].notna() & frame["source_id"].ne("")
    numeric_values = frame[numeric + ["snr_trace_median_db"]].to_numpy(dtype=float)
    finite_values = np.isfinite(numeric_values).all(axis=1)

    mask = (
        frame["trace_category"].eq("earthquake_local").to_numpy()
        & frame["source_magnitude_type"].eq("ml").to_numpy()
        & valid_id.to_numpy()
        & finite_values
        & frame["source_distance_km"].ge(0).to_numpy()
    )
    return frame.loc[
        mask,
        [
            "source_id",
            "source_magnitude",
            "source_distance_km",
            "source_depth_km",
            "snr_trace_median_db",
        ],
    ].reset_index(drop=True)


def snr_event_model(
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    intercept: float,
    magnitude_slope: float,
    log_distance_slope: float,
    depth_slope: float,
) -> np.ndarray:
    """Event-level SNR model with distance and depth covariates."""
    magnitude, log_distance, depth = inputs
    return (
        intercept
        + magnitude_slope * magnitude
        + log_distance_slope * log_distance
        + depth_slope * depth
    )


def analyse_snr(snr_data: pd.DataFrame) -> dict:
    """Fit an event-level SNR model and estimate the adjusted magnitude effect.

    Traces are first aggregated to one row per source_id. This prevents events
    recorded at many stations from being counted as independent earthquakes.
    """
    if snr_data.empty:
        raise ValueError("No valid ML-magnitude traces with SNR were found")

    events = (
        snr_data.groupby("source_id", as_index=False)
        .agg(
            source_magnitude=("source_magnitude", "median"),
            source_distance_km=("source_distance_km", "median"),
            source_depth_km=("source_depth_km", "median"),
            snr_event_median_db=("snr_trace_median_db", "median"),
            trace_count=("snr_trace_median_db", "size"),
        )
    )
    events["log_distance"] = np.log1p(events["source_distance_km"])
    inputs = (
        events["source_magnitude"].to_numpy(dtype=float),
        events["log_distance"].to_numpy(dtype=float),
        events["source_depth_km"].to_numpy(dtype=float),
    )
    observed = events["snr_event_median_db"].to_numpy(dtype=float)
    initial = (float(np.median(observed)), 0.0, 0.0, 0.0)
    parameters, covariance = curve_fit(
        snr_event_model,
        inputs,
        observed,
        p0=initial,
        maxfev=20_000,
    )

    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    degrees_freedom = max(len(events) - len(parameters), 1)
    t_statistics = np.divide(
        parameters,
        standard_errors,
        out=np.full_like(parameters, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2.0 * stats.t.sf(np.abs(t_statistics), df=degrees_freedom)
    p_value_reports = [
        "<0.001" if (not np.isfinite(p_value) or p_value < 0.001) else f"{p_value:.3e}"
        for p_value in p_values
    ]
    critical_t = stats.t.ppf(0.975, df=degrees_freedom)
    lower = parameters - critical_t * standard_errors
    upper = parameters + critical_t * standard_errors

    parameter_names = ["intercept", "magnitude_per_ml", "log1p_distance", "depth_km"]
    model_table = pd.DataFrame(
        {
            "parameter": parameter_names,
            "estimate": parameters,
            "standard_error": standard_errors,
            "t_statistic": t_statistics,
            "ci95_low": lower,
            "ci95_high": upper,
            "p_value": p_value_reports,
            "events": len(events),
        }
    )
    events["predicted_snr_db"] = snr_event_model(inputs, *parameters)
    event_fit = _regression_metrics(observed, events["predicted_snr_db"].to_numpy())
    fit_metrics = pd.DataFrame(
        [
            {
                "sample": "event_level_in_sample_fit",
                "events": len(events),
                "mae_db": event_fit["mae"],
                "rmse_db": event_fit["rmse"],
                "r_squared": event_fit["r_squared"],
            }
        ]
    )

    return {
        "events": events,
        "model_table": model_table,
        "fit_metrics": fit_metrics,
        "parameters": parameters,
        "covariance": covariance,
    }


def make_figures(distance_results: dict, snr_results: dict, output_dir: Path = FIGURES_DIR) -> None:
    """Save diagnostic and relationship figures for the two research questions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # RQ1: held-out predictions and residuals. Sampling only affects the plot,
    # not the fit or metrics.
    test = distance_results["test_data"]
    plot_data = test.sample(
        n=min(PLOT_SAMPLE_SIZE, len(test)), random_state=RANDOM_SEED
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        plot_data["source_distance_km"],
        plot_data["predicted_quadratic_km"],
        s=8,
        alpha=0.22,
        label="Quadratic S-P",
    )
    axes[0].scatter(
        plot_data["source_distance_km"],
        plot_data["predicted_baseline_km"],
        s=8,
        alpha=0.22,
        label="S-P only",
    )
    axes[0].scatter(
        plot_data["source_distance_km"],
        plot_data["predicted_depth_aware_km"],
        s=8,
        alpha=0.22,
        label="S-P + depth",
    )
    maximum = float(
        np.nanmax(
            np.concatenate(
                [
                    plot_data["source_distance_km"].to_numpy(),
                    plot_data["predicted_baseline_km"].to_numpy(),
                    plot_data["predicted_quadratic_km"].to_numpy(),
                    plot_data["predicted_depth_aware_km"].to_numpy(),
                ]
            )
        )
    )
    axes[0].plot([0, maximum], [0, maximum], color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Observed source distance (km)", ylabel="Predicted distance (km)")
    axes[0].set_title("Held-out predictions")
    axes[0].legend(frameon=False)

    residual = plot_data["source_distance_km"] - plot_data["predicted_depth_aware_km"]
    axes[1].scatter(plot_data["source_distance_km"], residual, s=8, alpha=0.25)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Observed source distance (km)",
        ylabel="Residual (observed - predicted, km)",
    )
    axes[1].set_title("Depth-aware model residuals")
    figure.tight_layout()
    figure.savefig(output_dir / "rq1_distance_diagnostics.png", dpi=180)
    plt.close(figure)

    # RQ2: one plotted point per earthquake event; the adjusted line holds
    # distance and depth at their event-level medians.
    events = snr_results["events"]
    plot_events = events.sample(n=min(PLOT_SAMPLE_SIZE, len(events)), random_state=RANDOM_SEED)
    parameters = snr_results["parameters"]
    magnitude_line = np.linspace(
        float(events["source_magnitude"].min()),
        float(events["source_magnitude"].max()),
        200,
    )
    log_distance_median = float(events["log_distance"].median())
    depth_median = float(events["source_depth_km"].median())
    adjusted_line = snr_event_model(
        (
            magnitude_line,
            np.full_like(magnitude_line, log_distance_median),
            np.full_like(magnitude_line, depth_median),
        ),
        *parameters,
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        plot_events["source_magnitude"],
        plot_events["snr_event_median_db"],
        s=10,
        alpha=0.25,
        label="Earthquake events",
    )
    axis.plot(
        magnitude_line,
        adjusted_line,
        color="darkred",
        linewidth=2,
        label="Adjusted fit at median distance/depth",
    )
    axis.set(
        xlabel="Local magnitude, ML",
        ylabel="Event median SNR (dB)",
        title="SNR and magnitude, adjusted for distance and depth",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "rq2_snr_magnitude.png", dpi=180)
    plt.close(figure)


def build_data_summary(
    metadata: pd.DataFrame,
    distance_data: pd.DataFrame,
    snr_data: pd.DataFrame,
    snr_results: dict,
) -> pd.DataFrame:
    """Create a compact table documenting the analyzed sample sizes."""
    category_counts = metadata["trace_category"].value_counts(dropna=False)
    rows = [
        {"quantity": "all_metadata_rows", "count": len(metadata)},
        {"quantity": "earthquake_local_rows", "count": int(category_counts.get("earthquake_local", 0))},
        {"quantity": "noise_rows", "count": int(category_counts.get("noise", 0))},
        {"quantity": "RQ1_valid_trace_rows", "count": len(distance_data)},
        {"quantity": "RQ1_unique_source_ids", "count": int(distance_data["source_id"].nunique())},
        {"quantity": "RQ2_valid_ml_trace_rows", "count": len(snr_data)},
        {"quantity": "RQ2_unique_events", "count": len(snr_results["events"])},
    ]
    return pd.DataFrame(rows)


def run_analysis(
    archive_path: Path = DEFAULT_ARCHIVE,
    tables_dir: Path = TABLES_DIR,
    figures_dir: Path = FIGURES_DIR,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Run both analyses and write CSV tables and figures."""
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(archive_path)
    distance_data = prepare_distance_data(metadata)
    snr_data = prepare_snr_data(metadata)
    if distance_data.empty:
        raise ValueError("No valid earthquake rows were found for RQ1")

    distance_results = analyse_distance(distance_data, random_seed=random_seed)
    snr_results = analyse_snr(snr_data)
    summary = build_data_summary(metadata, distance_data, snr_data, snr_results)

    summary.to_csv(tables_dir / "dataset_summary.csv", index=False)
    distance_results["metrics"].to_csv(tables_dir / "rq1_model_metrics.csv", index=False)
    distance_results["parameters"].to_csv(tables_dir / "rq1_model_parameters.csv", index=False)
    snr_results["model_table"].to_csv(tables_dir / "rq2_snr_model.csv", index=False)
    snr_results["fit_metrics"].to_csv(tables_dir / "rq2_fit_metrics.csv", index=False)
    software_versions = pd.DataFrame(
        [
            {"package": "python", "version": platform.python_version()},
            {"package": "numpy", "version": np.__version__},
            {"package": "scipy", "version": scipy.__version__},
            {"package": "pandas", "version": pd.__version__},
            {"package": "matplotlib", "version": matplotlib.__version__},
        ]
    )
    software_versions.to_csv(tables_dir / "software_versions.csv", index=False)
    make_figures(distance_results, snr_results, figures_dir)

    print("\nDataset and analysis sample sizes")
    print(summary.to_string(index=False))
    print("\nRQ1 held-out model metrics (split by source_id)")
    print(distance_results["metrics"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nRQ2 event-level regression (one row per source_id)")
    print(
        snr_results["model_table"].to_string(
            index=False, float_format=lambda x: f"{x:.6g}"
        )
    )
    print("\nRQ2 event-level in-sample fit metrics (descriptive, not a test-set score)")
    print(snr_results["fit_metrics"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nRuntime package versions")
    print(software_versions.to_string(index=False))
    print(f"\nTables written to: {tables_dir}")
    print(f"Figures written to: {figures_dir}")

    return {
        "summary": summary,
        "distance": distance_results,
        "snr": snr_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Path to the STEAD metadata CSV or ZIP archive",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed for the RQ1 group split"
    )
    args = parser.parse_args()
    run_analysis(archive_path=args.archive, random_seed=args.seed)


if __name__ == "__main__":
    main()
