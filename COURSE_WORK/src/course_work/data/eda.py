import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from course_work.data.schema import dataframe_fingerprint, load_raw_csv
from course_work.data.splitting import load_validated_split_membership, materialize_phase_5
from course_work.data.temporal import load_validated_temporal_view, materialize_phase_4
from course_work.utils.artifacts import get_project_root, read_json, sha256_file


TARGET_COLUMN = "Appliances"
SELECTED_LAGS = (1, 6, 36, 72, 144, 1008)
CROSS_CORRELATION_LAGS = (1, 6, 36, 72, 144)
CROSS_CORRELATION_FEATURES = ("lights", "T1", "RH_1", "T_out")
TEMPERATURE_COLUMNS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T_out")
HUMIDITY_COLUMNS = ("RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9", "RH_out")
WEATHER_COLUMNS = ("T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint")
DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def build_eda_view(temporal_view: pd.DataFrame) -> pd.DataFrame:
    required = {"raw_row_index", "timestamp_parsed", "continuity_segment_id", TARGET_COLUMN}
    missing = sorted(required.difference(temporal_view.columns))
    if missing:
        raise ValueError(f"Temporal view is missing required lineage columns: {missing}")
    view = temporal_view.copy(deep=True)
    view["hour"] = view["timestamp_parsed"].dt.hour
    view["day_of_week"] = view["timestamp_parsed"].dt.dayofweek
    view["day_name"] = pd.Categorical(
        view["day_of_week"].map(dict(enumerate(DAY_NAMES))),
        categories=list(DAY_NAMES),
        ordered=True,
    )
    view["is_weekend"] = view["day_of_week"].ge(5)
    view["calendar_date"] = view["timestamp_parsed"].dt.strftime("%Y-%m-%d")
    view["month"] = view["timestamp_parsed"].dt.month
    view["week"] = view["timestamp_parsed"].dt.isocalendar().week.astype(int)
    return view


def numeric_summary(dataframe: pd.DataFrame, schema_manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in schema_manifest["ordered_columns"]:
        if variable == schema_manifest["timestamp_column"]:
            continue
        series = pd.to_numeric(dataframe[variable], errors="raise")
        quantiles = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        rows.append({
            "variable": variable,
            "role": schema_manifest["role_mapping"][variable],
            "group": schema_manifest["feature_group_mapping"][variable],
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "q01": float(quantiles.loc[0.01]),
            "q05": float(quantiles.loc[0.05]),
            "q25": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.5]),
            "q75": float(quantiles.loc[0.75]),
            "q90": float(quantiles.loc[0.9]),
            "q95": float(quantiles.loc[0.95]),
            "q99": float(quantiles.loc[0.99]),
            "max": float(series.max()),
            "skewness": float(series.skew()),
        })
    return pd.DataFrame(rows)


def target_quantiles(dataframe: pd.DataFrame) -> pd.DataFrame:
    probabilities = [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    values = dataframe[TARGET_COLUMN].quantile(probabilities)
    return pd.DataFrame({"quantile": probabilities, "value_wh": [float(values.loc[value]) for value in probabilities]})


def hourly_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    profile = dataframe.groupby("hour", observed=True)[TARGET_COLUMN].agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    )
    return profile.reset_index()


def weekday_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    profile = dataframe.groupby(["day_of_week", "day_name"], observed=True)[TARGET_COLUMN].agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    )
    return profile.reset_index().sort_values("day_of_week")


def weekend_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    profile = dataframe.groupby("is_weekend", observed=True)[TARGET_COLUMN].agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    )
    output = profile.reset_index()
    output.insert(1, "day_type", output["is_weekend"].map({False: "Weekday", True: "Weekend"}))
    return output


def hour_weekday_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    profile = dataframe.groupby(["day_of_week", "day_name", "hour"], observed=True)[TARGET_COLUMN].agg(
        count="count",
        mean="mean",
        median="median",
    )
    return profile.reset_index().sort_values(["day_of_week", "hour"])


def correlation_outputs(dataframe: pd.DataFrame, schema_manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_columns = [column for column in schema_manifest["ordered_columns"] if column != schema_manifest["timestamp_column"]]
    numeric = dataframe[numeric_columns]
    pearson = numeric.corr(method="pearson")
    spearman_target = numeric.corr(method="spearman")[TARGET_COLUMN]
    target_rows = [
        {
            "feature": feature,
            "pearson_r": float(pearson.loc[feature, TARGET_COLUMN]),
            "absolute_pearson_r": float(abs(pearson.loc[feature, TARGET_COLUMN])),
            "spearman_rho": float(spearman_target.loc[feature]),
        }
        for feature in numeric_columns
        if feature != TARGET_COLUMN
    ]
    target = pd.DataFrame(target_rows).sort_values("absolute_pearson_r", ascending=False).reset_index(drop=True)
    feature_columns = [column for column in numeric_columns if column != TARGET_COLUMN]
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(feature_columns):
        for right in feature_columns[left_index + 1 :]:
            correlation = float(pearson.loc[left, right])
            if abs(correlation) >= 0.8:
                pairs.append({
                    "feature_1": left,
                    "feature_2": right,
                    "pearson_r": correlation,
                    "absolute_pearson_r": abs(correlation),
                    "threshold": 0.8,
                    "threshold_purpose": "descriptive_flag_not_feature_removal",
                })
    high_pairs = pd.DataFrame(
        pairs,
        columns=["feature_1", "feature_2", "pearson_r", "absolute_pearson_r", "threshold", "threshold_purpose"],
    )
    if not high_pairs.empty:
        high_pairs = high_pairs.sort_values("absolute_pearson_r", ascending=False).reset_index(drop=True)
    return pearson, target, high_pairs


def segment_aware_lag_correlation(dataframe: pd.DataFrame, value_column: str, lag: int) -> tuple[float, int]:
    if lag < 1:
        raise ValueError("lag must be positive")
    current_parts: list[np.ndarray] = []
    lagged_parts: list[np.ndarray] = []
    for _, segment in dataframe.groupby("continuity_segment_id", sort=False):
        if len(segment) <= lag:
            continue
        timestamps = segment["timestamp_parsed"].reset_index(drop=True)
        valid_spacing = timestamps.iloc[lag:].reset_index(drop=True).sub(timestamps.iloc[:-lag].reset_index(drop=True)).dt.total_seconds().div(60).eq(lag * 10)
        current = segment[value_column].iloc[lag:].reset_index(drop=True)[valid_spacing].to_numpy(dtype=float)
        lagged = segment[value_column].iloc[:-lag].reset_index(drop=True)[valid_spacing].to_numpy(dtype=float)
        current_parts.append(current)
        lagged_parts.append(lagged)
    if not current_parts:
        return float("nan"), 0
    current_values = np.concatenate(current_parts)
    lagged_values = np.concatenate(lagged_parts)
    if len(current_values) < 2:
        return float("nan"), len(current_values)
    return float(np.corrcoef(current_values, lagged_values)[0, 1]), len(current_values)


def segment_aware_cross_correlation(dataframe: pd.DataFrame, feature: str, lag: int) -> tuple[float, int]:
    target_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    for _, segment in dataframe.groupby("continuity_segment_id", sort=False):
        if len(segment) <= lag:
            continue
        timestamps = segment["timestamp_parsed"].reset_index(drop=True)
        valid_spacing = timestamps.iloc[lag:].reset_index(drop=True).sub(timestamps.iloc[:-lag].reset_index(drop=True)).dt.total_seconds().div(60).eq(lag * 10)
        target_values = segment[TARGET_COLUMN].iloc[lag:].reset_index(drop=True)[valid_spacing].to_numpy(dtype=float)
        feature_values = segment[feature].iloc[:-lag].reset_index(drop=True)[valid_spacing].to_numpy(dtype=float)
        target_parts.append(target_values)
        feature_parts.append(feature_values)
    if not target_parts:
        return float("nan"), 0
    target_values = np.concatenate(target_parts)
    feature_values = np.concatenate(feature_parts)
    if len(target_values) < 2:
        return float("nan"), len(target_values)
    return float(np.corrcoef(target_values, feature_values)[0, 1]), len(target_values)


def selected_lag_correlations(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lag in SELECTED_LAGS:
        correlation, pairs = segment_aware_lag_correlation(dataframe, TARGET_COLUMN, lag)
        rows.append({
            "lag_steps": lag,
            "lag_minutes": lag * 10,
            "lag_label": {1: "10 min", 6: "1 h", 36: "6 h", 72: "12 h", 144: "24 h", 1008: "7 d"}[lag],
            "correlation": correlation,
            "valid_pair_count": pairs,
            "decision_status": "DESCRIPTIVE_NOT_LOOKBACK_SELECTION",
        })
    return pd.DataFrame(rows)


def autocorrelation_profile(dataframe: pd.DataFrame, maximum_lag: int = 144) -> pd.DataFrame:
    rows = []
    for lag in range(1, maximum_lag + 1):
        correlation, pairs = segment_aware_lag_correlation(dataframe, TARGET_COLUMN, lag)
        rows.append({"lag_steps": lag, "lag_minutes": lag * 10, "correlation": correlation, "valid_pair_count": pairs})
    return pd.DataFrame(rows)


def exogenous_lag_correlations(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in CROSS_CORRELATION_FEATURES:
        for lag in CROSS_CORRELATION_LAGS:
            correlation, pairs = segment_aware_cross_correlation(dataframe, feature, lag)
            rows.append({
                "feature": feature,
                "lag_steps": lag,
                "lag_minutes": lag * 10,
                "correlation_with_future_target": correlation,
                "valid_pair_count": pairs,
                "decision_status": "HYPOTHESIS_ONLY",
            })
    return pd.DataFrame(rows)


def rolling_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe[["raw_row_index", "timestamp_parsed", "continuity_segment_id", TARGET_COLUMN]].copy(deep=True)
    grouped = dataframe.groupby("continuity_segment_id", sort=False)[TARGET_COLUMN]
    output["rolling_mean_36"] = grouped.transform(lambda values: values.rolling(36, min_periods=36).mean())
    output["rolling_mean_144"] = grouped.transform(lambda values: values.rolling(144, min_periods=144).mean())
    output["rolling_mean_1008"] = grouped.transform(lambda values: values.rolling(1008, min_periods=1008).mean())
    output["rolling_std_144"] = grouped.transform(lambda values: values.rolling(144, min_periods=144).std())
    return output


def representative_windows(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete_days = dataframe.groupby("calendar_date", sort=True).filter(lambda values: len(values) == 144)
    if complete_days.empty:
        raise RuntimeError("No complete deterministic 24-hour window exists")
    first_day = complete_days["calendar_date"].iloc[0]
    day_window = complete_days[complete_days["calendar_date"] == first_day].copy(deep=True)
    week_window = None
    for _, segment in dataframe.groupby("continuity_segment_id", sort=False):
        if len(segment) >= 1008:
            week_window = segment.iloc[:1008].copy(deep=True)
            break
    if week_window is None:
        raise RuntimeError("No complete deterministic 7-day window exists")
    metadata = pd.DataFrame([
        {
            "window_id": "REPRESENTATIVE_24H",
            "selection_rule": "earliest_complete_calendar_day",
            "start_timestamp": day_window["timestamp_parsed"].iloc[0].isoformat(sep=" "),
            "end_timestamp": day_window["timestamp_parsed"].iloc[-1].isoformat(sep=" "),
            "row_count": len(day_window),
        },
        {
            "window_id": "REPRESENTATIVE_7D",
            "selection_rule": "earliest_complete_continuous_1008_row_block",
            "start_timestamp": week_window["timestamp_parsed"].iloc[0].isoformat(sep=" "),
            "end_timestamp": week_window["timestamp_parsed"].iloc[-1].isoformat(sep=" "),
            "row_count": len(week_window),
        },
    ])
    return day_window, week_window, metadata


def extreme_target_samples(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = ["raw_row_index", "timestamp_parsed", "continuity_segment_id", "Appliances", "lights", "T1", "RH_1", "T_out", "RH_out", "Windspeed"]
    output = dataframe.nlargest(20, TARGET_COLUMN)[columns].copy(deep=True)
    output = output.rename(columns={"timestamp_parsed": "timestamp"})
    output["descriptive_only"] = True
    return output


def hypothesis_registry(analysis: dict[str, Any]) -> pd.DataFrame:
    target_row = analysis["numeric_summary"].set_index("variable").loc[TARGET_COLUMN]
    selected = analysis["selected_lag_correlations"].set_index("lag_steps")
    high_pair_count = len(analysis["high_correlation_pairs"])
    hourly = analysis["hourly_profile"]
    weekend = analysis["weekend_profile"].set_index("day_type")
    lights_correlation = analysis["target_correlations"].set_index("feature").loc["lights", "pearson_r"]
    rv_correlations = analysis["target_correlations"].set_index("feature").loc[["rv1", "rv2"], "pearson_r"].abs().max()
    rolling = analysis["rolling_statistics"]
    rolling_first = float(rolling["rolling_mean_1008"].dropna().iloc[0])
    rolling_last = float(rolling["rolling_mean_1008"].dropna().iloc[-1])
    rows = [
        ("H-EDA-001", "target_distribution", f"Target is right-skewed with skewness {target_row['skewness']:.6f}.", "DESCRIPTIVE_ONLY", "eda_numeric_summary.csv", "Compare loss sensitivity without deleting spikes.", "Phase 37", "UNTESTED", "Global descriptor only"),
        ("H-EDA-002", "target_spikes", f"The global 95th percentile is {target_row['q95']:.6f} Wh.", "DESCRIPTIVE_ONLY", "target_quantiles.csv;EDA_05_energy_spikes.png", "High-consumption observations may contribute strongly to squared error.", "Phase 37;Phase 50", "UNTESTED", "Final regimes must be Train-derived"),
        ("H-EDA-003", "autocorrelation", f"Lag-1 target correlation is {selected.loc[1, 'correlation']:.6f}.", "HYPOTHESIS_FOR_VALIDATION", "selected_lag_correlations.csv", "Past target may add predictive signal.", "Phase 23", "UNTESTED", "FS0, FS1 and FS2 remain required"),
        ("H-EDA-004", "autocorrelation", f"Lag-144 target correlation is {selected.loc[144, 'correlation']:.6f}.", "HYPOTHESIS_FOR_VALIDATION", "selected_lag_correlations.csv;EDA_14_target_autocorrelation.png", "Daily dependence may inform comparison of registered lookbacks.", "Phase 26", "UNTESTED", "Does not select L144"),
        ("H-EDA-005", "sensor_redundancy", f"There are {high_pair_count} feature pairs with absolute Pearson correlation at least 0.8.", "HYPOTHESIS_FOR_VALIDATION", "high_correlation_pairs.csv;EDA_13_correlation_heatmap.png", "Sensor redundancy should be studied without automatic removal.", "Phase 23", "UNTESTED", "Threshold is descriptive"),
        ("H-EDA-006", "daily_pattern", f"Hourly means span {hourly['mean'].min():.6f} to {hourly['mean'].max():.6f} Wh.", "HYPOTHESIS_FOR_VALIDATION", "hourly_energy_profile.csv;EDA_06_hourly_profile.png", "Calendar features may help forecasting.", "Phase 24", "UNTESTED", "TF0 and TF1 remain required"),
        ("H-EDA-007", "weekly_pattern", f"Weekday and weekend means are {weekend.loc['Weekday', 'mean']:.6f} and {weekend.loc['Weekend', 'mean']:.6f} Wh.", "HYPOTHESIS_FOR_VALIDATION", "weekend_energy_profile.csv;EDA_08_weekday_weekend.png", "Weekend context may capture schedule differences.", "Phase 24", "UNTESTED", "Descriptive full-data evidence"),
        ("H-EDA-008", "temporal_drift", f"First and last available 7-day rolling means are {rolling_first:.6f} and {rolling_last:.6f} Wh.", "HYPOTHESIS_FOR_VALIDATION", "rolling_target_statistics.csv;EDA_15_rolling_mean.png", "Temporal moments may differ across future chronological splits.", "Phase 8;Phase 40", "UNTESTED", "Does not activate RevIN"),
        ("H-EDA-009", "lighting", f"Instantaneous lights-target Pearson correlation is {lights_correlation:.6f}.", "HYPOTHESIS_FOR_VALIDATION", "target_correlations.csv", "Lighting may provide useful exogenous context.", "Phase 23", "UNTESTED", "Pairwise association is not predictive importance"),
        ("H-EDA-010", "random_controls", f"Maximum absolute instantaneous rv1/rv2 target correlation is {rv_correlations:.6f}.", "HYPOTHESIS_FOR_VALIDATION", "target_correlations.csv", "Random controls provide an overfitting sensitivity check.", "Phase 23", "UNTESTED", "FS2 remains required"),
    ]
    columns = ["hypothesis_id", "category", "statement", "evidence_type", "figure_or_table", "modeling_implication", "future_phase", "decision_status", "notes"]
    return pd.DataFrame(rows, columns=columns)


def anomaly_registry(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    target_row = analysis["numeric_summary"].set_index("variable").loc[TARGET_COLUMN]
    spike_count = int((analysis["eda_view"][TARGET_COLUMN] >= target_row["q95"]).sum())
    zero_lights = int(analysis["eda_view"]["lights"].eq(0).sum())
    return [
        {
            "id": "EA-001",
            "category": "TARGET_SPIKE",
            "timestamp_or_variable": TARGET_COLUMN,
            "observation": f"{spike_count} observations are at or above the global descriptive 95th percentile",
            "possible_interpretation": "High-consumption events are part of the observed target distribution",
            "requires_action": False,
            "future_phase": "Phase 37;Phase 50",
            "status": "HYPOTHESIS_ONLY",
        },
        {
            "id": "EA-002",
            "category": "HIGH_REDUNDANCY",
            "timestamp_or_variable": "numeric_features",
            "observation": f"{len(analysis['high_correlation_pairs'])} feature pairs meet the descriptive 0.8 threshold",
            "possible_interpretation": "Multiple sensors may encode overlapping environmental conditions",
            "requires_action": False,
            "future_phase": "Phase 23",
            "status": "HYPOTHESIS_ONLY",
        },
        {
            "id": "EA-003",
            "category": "ZERO_INFLATION",
            "timestamp_or_variable": "lights",
            "observation": f"{zero_lights} observations have zero lights energy use",
            "possible_interpretation": "Zero is a valid observed energy state and not missingness",
            "requires_action": False,
            "future_phase": "Phase 6;Phase 23",
            "status": "DESCRIPTIVE_ONLY",
        },
    ]


def compare_temporal_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    required = {"Train", "Validation", "Test"}
    if set(splits) != required:
        raise ValueError("splits must contain Train, Validation and Test")
    rows = []
    for split_name in ("Train", "Validation", "Test"):
        frame = splits[split_name]
        values = frame[TARGET_COLUMN]
        rows.append({
            "split": split_name,
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q05": float(values.quantile(0.05)),
            "median": float(values.median()),
            "q95": float(values.quantile(0.95)),
        })
    return pd.DataFrame(rows)


def prepare_eda_analysis(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    phase_4 = materialize_phase_4(root)
    if phase_4.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
        raise RuntimeError("TEMPORAL-v1 does not permit EDA")
    phase_5 = materialize_phase_5(root)
    if phase_5.get("status") != "PASS":
        raise RuntimeError("SPLIT-v1 does not permit EDA")
    raw_path = root / "data/raw_data/energydata_complete.csv"
    raw_hash_before = sha256_file(raw_path)
    raw = load_raw_csv(raw_path)
    raw_dataframe_before = dataframe_fingerprint(raw)
    temporal_view = load_validated_temporal_view(root)
    split_membership = load_validated_split_membership(root)
    train_mask = split_membership["split_id"].eq("TRAIN")
    if int(train_mask.sum()) != phase_5["train_rows"]:
        raise RuntimeError("TRAIN mask row count does not match SPLIT-v1 train_rows")
    train_temporal_view = temporal_view.loc[train_mask.reset_index(drop=True)].reset_index(drop=True)
    eda_view = build_eda_view(train_temporal_view)
    schema_manifest = read_json(root / "artifacts/schema/schema_manifest.json")
    summary = numeric_summary(eda_view, schema_manifest)
    quantiles = target_quantiles(eda_view)
    hourly = hourly_profile(eda_view)
    weekday = weekday_profile(eda_view)
    weekend = weekend_profile(eda_view)
    hour_weekday = hour_weekday_profile(eda_view)
    correlation, target_correlation, high_pairs = correlation_outputs(eda_view, schema_manifest)
    selected_lags = selected_lag_correlations(eda_view)
    autocorrelation = autocorrelation_profile(eda_view)
    exogenous = exogenous_lag_correlations(eda_view)
    rolling = rolling_statistics(eda_view)
    day_window, week_window, window_metadata = representative_windows(eda_view)
    extremes = extreme_target_samples(eda_view)
    analysis: dict[str, Any] = {
        "project_root": root,
        "phase_4_signoff": phase_4,
        "phase_5_signoff": phase_5,
        "raw_csv_sha256": raw_hash_before,
        "raw_dataframe_fingerprint_before": raw_dataframe_before,
        "eda_view": eda_view,
        "train_only": True,
        "train_rows_used": int(train_mask.sum()),
        "total_rows": len(temporal_view),
        "numeric_summary": summary,
        "target_quantiles": quantiles,
        "hourly_profile": hourly,
        "weekday_profile": weekday,
        "weekend_profile": weekend,
        "hour_weekday_profile": hour_weekday,
        "correlation_matrix": correlation,
        "target_correlations": target_correlation,
        "high_correlation_pairs": high_pairs,
        "selected_lag_correlations": selected_lags,
        "autocorrelation_profile": autocorrelation,
        "exogenous_lag_correlations": exogenous,
        "rolling_statistics": rolling,
        "representative_day": day_window,
        "representative_week": week_window,
        "representative_windows": window_metadata,
        "extreme_target_samples": extremes,
    }
    analysis["hypotheses"] = hypothesis_registry(analysis)
    analysis["anomalies"] = anomaly_registry(analysis)
    raw_dataframe_after = dataframe_fingerprint(raw)
    if raw_dataframe_after != raw_dataframe_before:
        raise RuntimeError("Raw DataFrame mutated during EDA")
    if sha256_file(raw_path) != raw_hash_before:
        raise RuntimeError("Raw CSV changed during EDA")
    if len(eda_view) != int(train_mask.sum()) or not eda_view["raw_row_index"].is_unique:
        raise RuntimeError("EDA TRAIN-only lineage is incomplete")
    if len(eda_view) > len(raw):
        raise RuntimeError("EDA TRAIN-only view exceeds raw dataset")
    target_digest = hashlib.sha256(eda_view[TARGET_COLUMN].to_numpy().tobytes()).hexdigest()
    analysis["raw_dataframe_fingerprint_after"] = raw_dataframe_after
    analysis["target_value_fingerprint"] = target_digest
    return analysis
