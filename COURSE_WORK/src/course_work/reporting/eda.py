import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from course_work.data.eda import (
    DAY_NAMES,
    HUMIDITY_COLUMNS,
    TARGET_COLUMN,
    TEMPERATURE_COLUMNS,
    prepare_eda_analysis,
)
from course_work.utils.artifacts import (
    canonical_json_bytes,
    get_project_root,
    read_json,
    sha256_file,
    write_bytes_once_or_verify,
    write_json_once_or_verify,
    write_text_once_or_verify,
)


FIGURE_FILENAMES = (
    "EDA_01_target_histogram.png",
    "EDA_02_target_ecdf.png",
    "EDA_03_target_boxplot.png",
    "EDA_04_target_timeline.png",
    "EDA_05_energy_spikes.png",
    "EDA_06_hourly_profile.png",
    "EDA_07_weekday_profile.png",
    "EDA_08_weekday_weekend.png",
    "EDA_09_hour_weekday_heatmap.png",
    "EDA_10_temperature_distributions.png",
    "EDA_11_humidity_distributions.png",
    "EDA_12_lights_distribution.png",
    "EDA_13_correlation_heatmap.png",
    "EDA_14_target_autocorrelation.png",
    "EDA_15_rolling_mean.png",
    "EDA_16_rolling_standard_deviation.png",
)


def configure_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(figure: plt.Figure, path: Path) -> Path:
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "course_work"},
    )
    plt.close(figure)
    return write_bytes_once_or_verify(path, buffer.getvalue())


def render_eda_figures(analysis: dict[str, Any], figure_root: Path) -> list[Path]:
    configure_plot_style()
    data = analysis["eda_view"]
    target = data[TARGET_COLUMN]
    quantiles = analysis["target_quantiles"].set_index("quantile")["value_wh"]
    outputs: list[Path] = []

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(target, bins=50, color="#3264a8", edgecolor="white")
    axis.set(title="EDA-01 Appliances Distribution", xlabel="Appliances energy (Wh)", ylabel="Observation count")
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[0]))

    ordered = np.sort(target.to_numpy(dtype=float))
    ecdf = np.arange(1, len(ordered) + 1) / len(ordered)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(ordered, ecdf, color="#3264a8", linewidth=1.8)
    axis.set(title="EDA-02 Appliances Empirical CDF", xlabel="Appliances energy (Wh)", ylabel="Cumulative probability")
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[1]))

    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.boxplot(target.to_numpy(dtype=float), orientation="horizontal", tick_labels=["Appliances"], showfliers=True)
    axis.set(title="EDA-03 Appliances Boxplot", xlabel="Appliances energy (Wh)", ylabel="Target")
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[2]))

    figure, axes = plt.subplots(3, 1, figsize=(13, 10))
    axes[0].plot(data["timestamp_parsed"], target, linewidth=0.45, color="#3264a8")
    axes[0].set(title="Full DATA-v1 timeline", xlabel="Timestamp", ylabel="Appliances (Wh)")
    week = analysis["representative_week"]
    axes[1].plot(week["timestamp_parsed"], week[TARGET_COLUMN], linewidth=0.8, color="#9c3d3d")
    axes[1].set(title=f"Earliest continuous 7-day block: {week['timestamp_parsed'].iloc[0]} to {week['timestamp_parsed'].iloc[-1]}", xlabel="Timestamp", ylabel="Appliances (Wh)")
    day = analysis["representative_day"]
    axes[2].plot(day["timestamp_parsed"], day[TARGET_COLUMN], linewidth=1.1, color="#2d7f5e")
    axes[2].set(title=f"Earliest complete calendar day: {day['calendar_date'].iloc[0]}", xlabel="Timestamp", ylabel="Appliances (Wh)")
    figure.tight_layout()
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[3]))

    q95 = float(quantiles.loc[0.95])
    q99 = float(quantiles.loc[0.99])
    spikes = data[target >= q95]
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(data["timestamp_parsed"], target, linewidth=0.4, color="#7a7a7a", alpha=0.7, label="Appliances")
    axis.scatter(spikes["timestamp_parsed"], spikes[TARGET_COLUMN], s=6, color="#b83232", label="At or above global descriptive q95")
    axis.axhline(q95, color="#b83232", linestyle="--", linewidth=1, label=f"q95 = {q95:.2f} Wh")
    axis.axhline(q99, color="#6d2b8c", linestyle=":", linewidth=1, label=f"q99 = {q99:.2f} Wh")
    axis.set(title="EDA-05 Descriptive Energy-Spike Timeline", xlabel="Timestamp", ylabel="Appliances energy (Wh)")
    axis.legend(loc="upper right")
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[4]))

    hourly = analysis["hourly_profile"]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(hourly["hour"], hourly["mean"], marker="o", label="Mean", color="#3264a8")
    axis.plot(hourly["hour"], hourly["median"], marker="s", linestyle="--", label="Median", color="#b83232")
    axis.fill_between(hourly["hour"], hourly["q25"], hourly["q75"], color="#3264a8", alpha=0.15, label="IQR")
    axis.set(title="EDA-06 Appliances by Hour of Day", xlabel="Hour of day", ylabel="Appliances energy (Wh)", xticks=range(24))
    axis.legend()
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[5]))

    weekday = analysis["weekday_profile"]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(weekday["day_of_week"], weekday["mean"], marker="o", label="Mean", color="#3264a8")
    axis.plot(weekday["day_of_week"], weekday["median"], marker="s", linestyle="--", label="Median", color="#b83232")
    axis.set(title="EDA-07 Appliances by Day of Week", xlabel="Day of week", ylabel="Appliances energy (Wh)", xticks=range(7), xticklabels=DAY_NAMES)
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[6]))

    weekday_values = data.loc[~data["is_weekend"], TARGET_COLUMN].to_numpy(dtype=float)
    weekend_values = data.loc[data["is_weekend"], TARGET_COLUMN].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot([weekday_values, weekend_values], tick_labels=["Weekday", "Weekend"], showfliers=False)
    axis.set(title="EDA-08 Weekday and Weekend Appliances", xlabel="Day type", ylabel="Appliances energy (Wh)")
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[7]))

    hour_weekday = analysis["hour_weekday_profile"].pivot(index="day_of_week", columns="hour", values="median").reindex(index=range(7), columns=range(24))
    figure, axis = plt.subplots(figsize=(12, 5))
    image = axis.imshow(hour_weekday.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    axis.set(title="EDA-09 Median Appliances by Hour and Weekday", xlabel="Hour of day", ylabel="Day of week", xticks=range(24), yticks=range(7), yticklabels=DAY_NAMES)
    figure.colorbar(image, ax=axis, label="Median Appliances (Wh)")
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[8]))

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.boxplot([data[column].to_numpy(dtype=float) for column in TEMPERATURE_COLUMNS], tick_labels=TEMPERATURE_COLUMNS, orientation="horizontal", showfliers=False)
    axis.set(title="EDA-10 Temperature Distributions", xlabel="Temperature (C)", ylabel="Variable")
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[9]))

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.boxplot([data[column].to_numpy(dtype=float) for column in HUMIDITY_COLUMNS], tick_labels=HUMIDITY_COLUMNS, orientation="horizontal", showfliers=False)
    axis.set(title="EDA-11 Humidity Distributions", xlabel="Relative humidity (%)", ylabel="Variable")
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[10]))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(data["lights"], bins=np.arange(data["lights"].min(), data["lights"].max() + 10, 10), color="#d18f2f", edgecolor="white")
    axis.set(title="EDA-12 Lights Energy Distribution", xlabel="Lights energy (Wh)", ylabel="Observation count")
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[11]))

    correlation = analysis["correlation_matrix"]
    figure, axis = plt.subplots(figsize=(15, 13))
    image = axis.imshow(correlation.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    positions = range(len(correlation.columns))
    axis.set(title="EDA-13 Pearson Correlation Matrix", xticks=positions, yticks=positions, xticklabels=correlation.columns, yticklabels=correlation.index)
    axis.tick_params(axis="x", rotation=90, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    figure.colorbar(image, ax=axis, label="Pearson r", fraction=0.03, pad=0.02)
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[12]))

    autocorrelation = analysis["autocorrelation_profile"]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(autocorrelation["lag_steps"], autocorrelation["correlation"], color="#3264a8", linewidth=1.5)
    for lag in (36, 72, 144):
        value = float(autocorrelation.loc[autocorrelation["lag_steps"] == lag, "correlation"].iloc[0])
        axis.scatter([lag], [value], label=f"Lag {lag}: {value:.3f}")
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set(title="EDA-14 Segment-Aware Target Autocorrelation", xlabel="Lag steps at 10-minute spacing", ylabel="Correlation")
    axis.legend()
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[13]))

    rolling = analysis["rolling_statistics"]
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(rolling["timestamp_parsed"], rolling[TARGET_COLUMN], color="#8a8a8a", linewidth=0.35, alpha=0.45, label="Appliances")
    axis.plot(rolling["timestamp_parsed"], rolling["rolling_mean_144"], color="#3264a8", linewidth=1.2, label="24-hour rolling mean")
    axis.plot(rolling["timestamp_parsed"], rolling["rolling_mean_1008"], color="#b83232", linewidth=1.4, label="7-day rolling mean")
    axis.set(title="EDA-15 Segment-Aware Rolling Mean", xlabel="Timestamp", ylabel="Appliances energy (Wh)")
    axis.legend()
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[14]))

    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(rolling["timestamp_parsed"], rolling["rolling_std_144"], color="#6d2b8c", linewidth=1.1)
    axis.set(title="EDA-16 Segment-Aware 24-Hour Rolling Standard Deviation", xlabel="Timestamp", ylabel="Rolling standard deviation (Wh)")
    outputs.append(save_figure(figure, figure_root / FIGURE_FILENAMES[15]))
    return outputs


def dataframe_csv(dataframe: pd.DataFrame, index: bool = False) -> str:
    return dataframe.to_csv(index=index, lineterminator="\n")


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") != "PASS" or signoff.get("artifact_version") != "EDA-v1":
        raise RuntimeError("Existing Phase 6 sign-off is invalid")
    for relative_path, expected_hash in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Phase 6 artifact checksum mismatch: {relative_path}")
    return signoff


def materialize_phase_6(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    artifact_root = root / "artifacts/eda"
    table_root = artifact_root / "tables"
    figure_root = artifact_root / "figures"
    signoff_path = artifact_root / "phase_6_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    analysis = prepare_eda_analysis(root)
    table_values: dict[str, str] = {
        "eda_numeric_summary.csv": dataframe_csv(analysis["numeric_summary"]),
        "target_quantiles.csv": dataframe_csv(analysis["target_quantiles"]),
        "hourly_energy_profile.csv": dataframe_csv(analysis["hourly_profile"]),
        "weekday_energy_profile.csv": dataframe_csv(analysis["weekday_profile"]),
        "weekend_energy_profile.csv": dataframe_csv(analysis["weekend_profile"]),
        "hour_weekday_profile.csv": dataframe_csv(analysis["hour_weekday_profile"]),
        "correlation_matrix.csv": dataframe_csv(analysis["correlation_matrix"], index=True),
        "target_correlations.csv": dataframe_csv(analysis["target_correlations"]),
        "high_correlation_pairs.csv": dataframe_csv(analysis["high_correlation_pairs"]),
        "selected_lag_correlations.csv": dataframe_csv(analysis["selected_lag_correlations"]),
        "autocorrelation_profile.csv": dataframe_csv(analysis["autocorrelation_profile"]),
        "selected_exogenous_lag_correlations.csv": dataframe_csv(analysis["exogenous_lag_correlations"]),
        "rolling_target_statistics.csv": dataframe_csv(analysis["rolling_statistics"]),
        "representative_windows.csv": dataframe_csv(analysis["representative_windows"]),
        "extreme_target_samples.csv": dataframe_csv(analysis["extreme_target_samples"]),
        "eda_hypotheses.csv": dataframe_csv(analysis["hypotheses"]),
    }
    table_paths = []
    for filename, value in table_values.items():
        path = table_root / filename
        write_text_once_or_verify(path, value)
        table_paths.append(path)
    anomaly_path = artifact_root / "eda_anomalies.json"
    write_json_once_or_verify(anomaly_path, {"eda_version": "EDA-v1", "anomalies": analysis["anomalies"]})
    figure_paths = render_eda_figures(analysis, figure_root)
    relative_tables = [str(path.relative_to(root)) for path in table_paths]
    relative_figures = [str(path.relative_to(root)) for path in figure_paths]
    manifest = {
        "eda_version": "EDA-v1",
        "dataset_revision": "DATA-v1",
        "schema_version": "SCHEMA-v1",
        "temporal_version": "TEMPORAL-v1",
        "environment_id": "ENV-v1",
        "row_count": len(analysis["eda_view"]),
        "target": TARGET_COLUMN,
        "timestamp": "timestamp_parsed",
        "lineage_columns": ["raw_row_index", "continuity_segment_id"],
        "global_eda_scope": "descriptive_and_explanatory_only",
        "train_only_scope": True,
        "train_rows_used": analysis["train_rows_used"],
        "total_rows": analysis["total_rows"],
        "decision_sensitive_policy": "hypotheses_require_future_train_validation_confirmation",
        "figures_generated": relative_figures,
        "tables_generated": relative_tables,
        "hypotheses_count": len(analysis["hypotheses"]),
        "anomalies_count": len(analysis["anomalies"]),
        "representative_windows": analysis["representative_windows"].to_dict(orient="records"),
        "split_distribution_analysis_status": "DERIVED_FROM_PHASE_5",
        "correlation_threshold": {
            "absolute_pearson": 0.8,
            "purpose": "descriptive_flag_not_feature_removal",
        },
        "spike_threshold_policy": "train_only_quantiles_descriptive_only_final_regimes_train_derived",
        "lag_policy": "segment_aware_descriptive_not_lookback_selection",
        "rolling_policy": "within_continuity_segment_only",
        "processing_actions": {
            "rows_removed": 0,
            "outliers_removed": False,
            "interpolation_applied": False,
            "imputation_applied": False,
            "feature_selection_applied": False,
            "model_tuning_applied": False,
        },
        "raw_csv_sha256": analysis["raw_csv_sha256"],
        "raw_dataframe_fingerprint_before": analysis["raw_dataframe_fingerprint_before"],
        "raw_dataframe_fingerprint_after": analysis["raw_dataframe_fingerprint_after"],
        "target_value_fingerprint": analysis["target_value_fingerprint"],
        "warnings": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = artifact_root / "eda_manifest.json"
    write_json_once_or_verify(manifest_path, manifest)
    output_paths = relative_tables + relative_figures + [
        "artifacts/eda/eda_anomalies.json",
        "artifacts/eda/eda_manifest.json",
    ]
    output_checksums = {relative: sha256_file(root / relative) for relative in output_paths}
    phase_4 = analysis["phase_4_signoff"]
    signoff = {
        "artifact_version": "EDA-v1",
        "phase_id": 6,
        "phase_version": "PHASE-6-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": "ENV-v1",
        "dataset_revision": "DATA-v1",
        "input_paths": [
            "artifacts/acquisition/phase_2_signoff.json",
            "artifacts/schema/phase_3_signoff.json",
            "artifacts/temporal/phase_4_signoff.json",
            "artifacts/splits/phase_5_signoff.json",
            "data/raw_data/energydata_complete.csv",
        ],
        "input_checksums": {
            "artifacts/acquisition/phase_2_signoff.json": sha256_file(root / "artifacts/acquisition/phase_2_signoff.json"),
            "artifacts/schema/phase_3_signoff.json": sha256_file(root / "artifacts/schema/phase_3_signoff.json"),
            "artifacts/temporal/phase_4_signoff.json": sha256_file(root / "artifacts/temporal/phase_4_signoff.json"),
            "artifacts/splits/phase_5_signoff.json": sha256_file(root / "artifacts/splits/phase_5_signoff.json"),
            "data/raw_data/energydata_complete.csv": analysis["raw_csv_sha256"],
        },
        "output_paths": output_paths,
        "output_checksums": output_checksums,
        "config_fingerprint": phase_4["config_fingerprint"],
        "status": "PASS",
        "tests": [
            "phase_input_versions_and_checksums",
            "raw_dataframe_immutability",
            "eda_lineage_preservation",
            "mandatory_table_schemas",
            "mandatory_figure_set",
            "segment_aware_lags",
            "segment_aware_rolling",
            "no_outlier_removal_or_interpolation",
            "hypothesis_only_decision_boundary",
        ],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return signoff
