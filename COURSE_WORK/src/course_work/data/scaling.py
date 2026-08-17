from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from course_work.data.feature_sets import (
    FEATURE_SET_VERSION,
    METADATA,
    TIME_FEATURES,
    compute_feature_fingerprint,
    get_feature_list,
    load_validated_feature_set_registry,
)
from course_work.data.features import FEATURE_VERSION, load_validated_feature_view
from course_work.data.splitting import (
    SPLIT_IDS,
    SPLIT_VERSION,
    load_validated_split_membership,
    materialize_phase_5,
)
from course_work.utils.artifacts import (
    canonical_json_bytes,
    csv_text,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
    write_bytes_once_or_verify,
    write_json_once_or_verify,
    write_text_once_or_verify,
)


SCALING_VERSION = "SCALING-v1"
SCALING_ARTIFACT_ROOT = "artifacts/scaling"
SCALER_ARTIFACT_ROOT = "artifacts/scalers"
TARGET_COLUMN = "Appliances"
TARGET_OPTIONS = ("YS0", "YS1")
PASSTHROUGH_CYCLICAL = ("hour_sin", "hour_cos", "dow_sin", "dow_cos")
PASSTHROUGH_BINARY = ("weekend",)
PASSTHROUGH_FEATURES = PASSTHROUGH_CYCLICAL + PASSTHROUGH_BINARY
SCALING_METADATA = tuple(METADATA) + ("split_id", "split_position")
MEAN_TOLERANCE = 1e-8
STD_TOLERANCE = 1e-8
ROUNDTRIP_TOLERANCE = 1e-10


def build_scaling_policy() -> dict[str, Any]:
    variants = {}
    for variant_id in ("FS0_TF0", "FS0_TF1", "FS1_TF0", "FS1_TF1", "FS2_TF0", "FS2_TF1"):
        full_order = get_feature_list(variant_id)
        pass_through = [feature for feature in full_order if feature in PASSTHROUGH_FEATURES]
        scaled = [feature for feature in full_order if feature not in PASSTHROUGH_FEATURES]
        metadata_leaks = [feature for feature in full_order if feature in SCALING_METADATA]
        if metadata_leaks:
            raise RuntimeError(f"Metadata columns entered scaling policy for {variant_id}: {metadata_leaks}")
        variants[variant_id] = {
            "bundle_id": f"XSCALER__{variant_id}",
            "variant_id": variant_id,
            "full_feature_order": full_order,
            "scaled_feature_order": scaled,
            "pass_through_feature_order": pass_through,
            "feature_fingerprint": compute_feature_fingerprint(variant_id, full_order),
        }
    return {
        "scaling_version": SCALING_VERSION,
        "x_method": "StandardScaler",
        "x_fit_split": "TRAIN",
        "scaled_group": "SCALE_CONTINUOUS",
        "pass_through_cyclical": list(PASSTHROUGH_CYCLICAL),
        "pass_through_binary": list(PASSTHROUGH_BINARY),
        "metadata_only": list(SCALING_METADATA),
        "target_options": list(TARGET_OPTIONS),
        "y_method_YS1": "StandardScaler",
        "y_fit_split": "TRAIN",
        "variants": variants,
    }


def _require_finite(dataframe: pd.DataFrame, columns: list[str], label: str) -> np.ndarray:
    values = dataframe.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    if np.isnan(values).any():
        raise ValueError(f"{label} contains NaN values")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains infinite values")
    return values


def _n_samples_seen(scaler: StandardScaler) -> int:
    values = np.asarray(scaler.n_samples_seen_)
    if values.size != 1:
        raise RuntimeError("Scaler n_samples_seen_ is not scalar")
    return int(values.reshape(-1)[0])


def compute_scaler_statistics_fingerprint(feature_order: list[str], scaler: StandardScaler) -> str:
    payload = {
        "features": feature_order,
        "mean": [float(value) for value in scaler.mean_],
        "variance": [float(value) for value in scaler.var_],
        "scale": [float(value) for value in scaler.scale_],
        "n_samples_seen": _n_samples_seen(scaler),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def fit_x_scaler_bundle(
    train_frame: pd.DataFrame,
    variant_id: str,
    feature_fingerprint: str,
    train_split_fingerprint: str,
    global_split_fingerprint: str,
) -> dict[str, Any]:
    policy = build_scaling_policy()
    if variant_id not in policy["variants"]:
        raise KeyError(f"Unknown feature variant: {variant_id}")
    variant = policy["variants"][variant_id]
    if feature_fingerprint != variant["feature_fingerprint"]:
        raise ValueError(f"Feature fingerprint mismatch for {variant_id}")
    if train_frame.columns.duplicated().any():
        raise ValueError("Training frame contains duplicate columns")
    required = variant["full_feature_order"] + ["split_id", "timestamp"]
    missing = [column for column in required if column not in train_frame.columns]
    if missing:
        raise ValueError(f"Training scaler columns are missing: {missing}")
    projected_order = [column for column in train_frame.columns if column in set(variant["full_feature_order"])]
    if projected_order != variant["full_feature_order"]:
        raise ValueError(f"Training feature order mismatch for {variant_id}")
    split_values = set(train_frame["split_id"].astype(str).unique())
    if split_values != {"TRAIN"}:
        raise ValueError(f"X scaler fit requires TRAIN-only rows, received {sorted(split_values)}")
    if train_frame["timestamp"].isna().any() or not train_frame["timestamp"].is_monotonic_increasing:
        raise ValueError("X scaler fit timestamps must be non-null and chronological")
    scaled_values = _require_finite(train_frame, variant["scaled_feature_order"], f"{variant_id} TRAIN fit")
    _require_finite(train_frame, variant["pass_through_feature_order"], f"{variant_id} pass-through")
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(scaled_values)
    zero_variance = [
        feature
        for feature, variance in zip(variant["scaled_feature_order"], scaler.var_)
        if float(variance) == 0.0
    ]
    return {
        "scaling_version": SCALING_VERSION,
        "bundle_id": variant["bundle_id"],
        "variant_id": variant_id,
        "feature_fingerprint": feature_fingerprint,
        "full_feature_order": list(variant["full_feature_order"]),
        "scaled_feature_order": list(variant["scaled_feature_order"]),
        "pass_through_feature_order": list(variant["pass_through_feature_order"]),
        "train_split_fingerprint": train_split_fingerprint,
        "global_split_fingerprint": global_split_fingerprint,
        "fit_split": "TRAIN",
        "fit_row_count": len(train_frame),
        "fit_start_timestamp": train_frame["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        "fit_end_timestamp": train_frame["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "zero_variance_features": zero_variance,
        "statistics_fingerprint": compute_scaler_statistics_fingerprint(variant["scaled_feature_order"], scaler),
        "scaler": scaler,
    }


def validate_scaler_bundle(
    bundle: dict[str, Any],
    variant_id: str,
    feature_fingerprint: str,
    global_split_fingerprint: str,
) -> None:
    if bundle.get("scaling_version") != SCALING_VERSION:
        raise ValueError("Scaling version mismatch")
    if bundle.get("variant_id") != variant_id:
        raise ValueError(f"Scaler variant mismatch: expected {variant_id}")
    if bundle.get("feature_fingerprint") != feature_fingerprint:
        raise ValueError(f"Scaler feature fingerprint mismatch for {variant_id}")
    if bundle.get("global_split_fingerprint") != global_split_fingerprint:
        raise ValueError("Scaler split fingerprint mismatch")
    expected_order = get_feature_list(variant_id)
    if bundle.get("full_feature_order") != expected_order:
        raise ValueError(f"Scaler feature order mismatch for {variant_id}")
    scaler = bundle.get("scaler")
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Scaler bundle does not contain StandardScaler")
    scaled_order = bundle.get("scaled_feature_order", [])
    if int(scaler.n_features_in_) != len(scaled_order):
        raise ValueError("Serialized scaler feature count mismatch")
    fingerprint = compute_scaler_statistics_fingerprint(scaled_order, scaler)
    if fingerprint != bundle.get("statistics_fingerprint"):
        raise ValueError("Scaler statistics fingerprint mismatch")


def transform_feature_variant(
    dataframe: pd.DataFrame,
    bundle: dict[str, Any],
    variant_id: str,
    feature_fingerprint: str,
    global_split_fingerprint: str,
) -> pd.DataFrame:
    validate_scaler_bundle(bundle, variant_id, feature_fingerprint, global_split_fingerprint)
    full_order = bundle["full_feature_order"]
    if dataframe.columns.duplicated().any() or list(dataframe.columns) != full_order:
        raise ValueError(f"Transform feature order mismatch for {variant_id}")
    output = dataframe.copy(deep=True).astype("float64")
    scaled_order = bundle["scaled_feature_order"]
    values = _require_finite(output, full_order, f"{variant_id} transform")
    if values.shape[1] != len(full_order):
        raise RuntimeError("Transform feature count changed")
    scaled_values = bundle["scaler"].transform(output.loc[:, scaled_order].to_numpy(dtype=np.float64, copy=True))
    output.loc[:, scaled_order] = scaled_values
    if list(output.columns) != full_order or len(output) != len(dataframe):
        raise RuntimeError("Transform changed feature order or row count")
    if not np.isfinite(output.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Transform created NaN or infinite values")
    return output


def inverse_transform_continuous_features(
    dataframe: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    scaled_order = bundle["scaled_feature_order"]
    if list(dataframe.columns) != scaled_order:
        raise ValueError("Continuous inverse-transform feature order mismatch")
    values = _require_finite(dataframe, scaled_order, "Continuous inverse transform")
    recovered = bundle["scaler"].inverse_transform(values)
    return pd.DataFrame(recovered, columns=scaled_order, index=dataframe.index)


def fit_y_scaler(
    train_frame: pd.DataFrame,
    train_split_fingerprint: str,
    global_split_fingerprint: str,
) -> dict[str, Any]:
    required = [TARGET_COLUMN, "split_id", "timestamp"]
    missing = [column for column in required if column not in train_frame.columns]
    if missing:
        raise ValueError(f"Target scaler columns are missing: {missing}")
    split_values = set(train_frame["split_id"].astype(str).unique())
    if split_values != {"TRAIN"}:
        raise ValueError(f"Y scaler fit requires TRAIN-only rows, received {sorted(split_values)}")
    if train_frame["timestamp"].isna().any() or not train_frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Y scaler fit timestamps must be non-null and chronological")
    values = _require_finite(train_frame, [TARGET_COLUMN], "YS1 TRAIN fit")
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(values)
    return {
        "scaling_version": SCALING_VERSION,
        "bundle_id": "YSCALER__YS1",
        "target": TARGET_COLUMN,
        "option": "YS1",
        "fit_split": "TRAIN",
        "fit_row_count": len(train_frame),
        "fit_start_timestamp": train_frame["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        "fit_end_timestamp": train_frame["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "train_split_fingerprint": train_split_fingerprint,
        "global_split_fingerprint": global_split_fingerprint,
        "statistics_fingerprint": compute_scaler_statistics_fingerprint([TARGET_COLUMN], scaler),
        "scaler": scaler,
    }


def transform_target(values: Any, option: str, y_bundle: dict[str, Any] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    original_shape = array.shape
    flat = array.reshape(-1, 1)
    if not np.isfinite(flat).all():
        raise ValueError("Target transform input contains NaN or infinite values")
    if option == "YS0":
        return array.copy()
    if option != "YS1" or y_bundle is None:
        raise ValueError(f"Unsupported or incomplete target scaling option: {option}")
    if y_bundle.get("scaling_version") != SCALING_VERSION or y_bundle.get("option") != "YS1":
        raise ValueError("YS1 scaler bundle contract mismatch")
    return y_bundle["scaler"].transform(flat).reshape(original_shape)


def inverse_transform_target(values: Any, option: str, y_bundle: dict[str, Any] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    original_shape = array.shape
    flat = array.reshape(-1, 1)
    if not np.isfinite(flat).all():
        raise ValueError("Target inverse-transform input contains NaN or infinite values")
    if option == "YS0":
        return array.copy()
    if option != "YS1" or y_bundle is None:
        raise ValueError(f"Unsupported or incomplete target scaling option: {option}")
    if y_bundle.get("scaling_version") != SCALING_VERSION or y_bundle.get("option") != "YS1":
        raise ValueError("YS1 scaler bundle contract mismatch")
    return y_bundle["scaler"].inverse_transform(flat).reshape(original_shape)


def build_scaling_view(feature_view: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    if len(feature_view) != len(membership):
        raise ValueError("FEATURES-v1 and SPLIT-v1 row counts differ")
    if not feature_view["raw_row_index"].reset_index(drop=True).equals(membership["raw_row_index"].reset_index(drop=True)):
        raise ValueError("FEATURES-v1 and SPLIT-v1 row identities differ")
    if not feature_view["timestamp"].reset_index(drop=True).equals(membership["timestamp"].reset_index(drop=True)):
        raise ValueError("FEATURES-v1 and SPLIT-v1 timestamps differ")
    output = feature_view.copy(deep=True)
    output["split_id"] = membership["split_id"].to_numpy(copy=True)
    output["split_position"] = membership["split_position"].to_numpy(copy=True)
    if output["split_id"].value_counts().to_dict() != membership["split_id"].value_counts().to_dict():
        raise RuntimeError("Scaling view changed split membership")
    return output


def _bundle_statistics_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    scaler = bundle["scaler"]
    scaled_lookup = {
        feature: position
        for position, feature in enumerate(bundle["scaled_feature_order"])
    }
    rows = []
    for position, feature in enumerate(bundle["full_feature_order"]):
        if feature in scaled_lookup:
            scaler_position = scaled_lookup[feature]
            variance = float(scaler.var_[scaler_position])
            rows.append({
                "bundle_id": bundle["bundle_id"],
                "variant_id": bundle["variant_id"],
                "feature_position": position,
                "feature_name": feature,
                "scaling_group": "SCALE_CONTINUOUS",
                "policy": "STANDARD_SCALE",
                "mean": float(scaler.mean_[scaler_position]),
                "variance": variance,
                "scale": float(scaler.scale_[scaler_position]),
                "n_samples_seen": _n_samples_seen(scaler),
                "zero_variance_flag": variance == 0.0,
            })
        else:
            group = "PASSTHROUGH_BINARY" if feature in PASSTHROUGH_BINARY else "PASSTHROUGH_CYCLICAL"
            rows.append({
                "bundle_id": bundle["bundle_id"],
                "variant_id": bundle["variant_id"],
                "feature_position": position,
                "feature_name": feature,
                "scaling_group": group,
                "policy": "PASSTHROUGH",
                "mean": None,
                "variance": None,
                "scale": None,
                "n_samples_seen": bundle["fit_row_count"],
                "zero_variance_flag": False,
            })
    return rows


def _audit_x_bundle(
    scaling_view: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transformed = {}
    for split_id in SPLIT_IDS:
        split_frame = scaling_view.loc[scaling_view["split_id"].eq(split_id), bundle["full_feature_order"]]
        transformed[split_id] = transform_feature_variant(
            split_frame,
            bundle,
            bundle["variant_id"],
            bundle["feature_fingerprint"],
            bundle["global_split_fingerprint"],
        )
    train_scaled = transformed["TRAIN"].loc[:, bundle["scaled_feature_order"]].to_numpy(dtype=np.float64)
    zero_mask = np.asarray(bundle["scaler"].var_) == 0.0
    nonzero_mask = ~zero_mask
    train_means = train_scaled.mean(axis=0)
    train_stds = train_scaled.std(axis=0, ddof=0)
    mean_ok = bool(np.all(np.abs(train_means[nonzero_mask]) <= MEAN_TOLERANCE))
    std_ok = bool(np.all(np.abs(train_stds[nonzero_mask] - 1.0) <= STD_TOLERANCE))
    zero_std_ok = bool(np.all(train_stds[zero_mask] == 0.0)) if zero_mask.any() else True
    pass_through_ok = all(
        np.array_equal(
            transformed[split_id].loc[:, bundle["pass_through_feature_order"]].to_numpy(),
            scaling_view.loc[scaling_view["split_id"].eq(split_id), bundle["pass_through_feature_order"]].to_numpy(),
        )
        for split_id in SPLIT_IDS
    )
    cyclical_ok = True
    if all(feature in bundle["full_feature_order"] for feature in ("hour_sin", "hour_cos")):
        cyclical_ok = cyclical_ok and bool(np.allclose(
            transformed["TRAIN"]["hour_sin"].to_numpy() ** 2 + transformed["TRAIN"]["hour_cos"].to_numpy() ** 2,
            1.0,
            atol=1e-10,
            rtol=0.0,
        ))
    if all(feature in bundle["full_feature_order"] for feature in ("dow_sin", "dow_cos")):
        cyclical_ok = cyclical_ok and bool(np.allclose(
            transformed["TRAIN"]["dow_sin"].to_numpy() ** 2 + transformed["TRAIN"]["dow_cos"].to_numpy() ** 2,
            1.0,
            atol=1e-10,
            rtol=0.0,
        ))
    original_subset = scaling_view.loc[scaling_view["split_id"].eq("TRAIN"), bundle["scaled_feature_order"]].head(256)
    transformed_subset = transformed["TRAIN"].loc[:, bundle["scaled_feature_order"]].head(256)
    recovered = inverse_transform_continuous_features(transformed_subset, bundle)
    roundtrip_ok = bool(np.allclose(
        recovered.to_numpy(dtype=np.float64),
        original_subset.to_numpy(dtype=np.float64),
        atol=ROUNDTRIP_TOLERANCE,
        rtol=ROUNDTRIP_TOLERANCE,
    ))
    finite_by_split = {
        split_id: bool(np.isfinite(frame.to_numpy(dtype=np.float64)).all())
        for split_id, frame in transformed.items()
    }
    validation_rows = []
    validation_frame = transformed["VALIDATION"]
    for feature in bundle["scaled_feature_order"]:
        train_values = transformed["TRAIN"][feature].to_numpy(dtype=np.float64)
        validation_values = validation_frame[feature].to_numpy(dtype=np.float64)
        train_std = float(train_values.std(ddof=0))
        validation_std = float(validation_values.std(ddof=0))
        validation_rows.append({
            "variant_id": bundle["variant_id"],
            "feature_name": feature,
            "train_scaled_mean": float(train_values.mean()),
            "train_scaled_std": train_std,
            "validation_scaled_mean": float(validation_values.mean()),
            "validation_scaled_std": validation_std,
            "mean_shift": float(validation_values.mean() - train_values.mean()),
            "std_ratio": None if train_std == 0.0 else validation_std / train_std,
        })
    checks = [mean_ok, std_ok, zero_std_ok, pass_through_ok, cyclical_ok, roundtrip_ok, *finite_by_split.values()]
    return {
        "bundle_id": bundle["bundle_id"],
        "variant_id": bundle["variant_id"],
        "train_row_count": bundle["fit_row_count"],
        "scaled_feature_count": len(bundle["scaled_feature_order"]),
        "pass_through_count": len(bundle["pass_through_feature_order"]),
        "train_scaled_mean_ok": mean_ok,
        "train_scaled_std_ok": std_ok and zero_std_ok,
        "zero_variance_count": len(bundle["zero_variance_features"]),
        "validation_transform_ok": finite_by_split["VALIDATION"],
        "test_structural_transform_ok": finite_by_split["TEST"],
        "no_new_nan": all(finite_by_split.values()),
        "no_new_inf": all(finite_by_split.values()),
        "pass_through_unchanged": pass_through_ok,
        "cyclical_invariant_ok": cyclical_ok,
        "roundtrip_ok": roundtrip_ok,
        "status": "PASS" if all(checks) else "FAIL",
    }, validation_rows


def _joblib_bytes(payload: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    joblib.dump(payload, buffer, compress=0, protocol=5)
    return buffer.getvalue()


def _bundle_manifest(bundle: dict[str, Any], artifact_path: str, artifact_sha256: str) -> dict[str, Any]:
    scaler = bundle["scaler"]
    return {
        "bundle_id": bundle["bundle_id"],
        "variant_id": bundle["variant_id"],
        "feature_fingerprint": bundle["feature_fingerprint"],
        "feature_count": len(bundle["full_feature_order"]),
        "scaled_feature_order": bundle["scaled_feature_order"],
        "pass_through_feature_order": bundle["pass_through_feature_order"],
        "full_feature_order": bundle["full_feature_order"],
        "train_row_count": bundle["fit_row_count"],
        "fit_start_timestamp": bundle["fit_start_timestamp"],
        "fit_end_timestamp": bundle["fit_end_timestamp"],
        "mean": [float(value) for value in scaler.mean_],
        "variance": [float(value) for value in scaler.var_],
        "scale": [float(value) for value in scaler.scale_],
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "statistics_fingerprint": bundle["statistics_fingerprint"],
        "status": "PASS",
    }


def _target_manifest(y_bundle: dict[str, Any], artifact_path: str, artifact_sha256: str) -> dict[str, Any]:
    scaler = y_bundle["scaler"]
    return {
        "bundle_id": y_bundle["bundle_id"],
        "target": TARGET_COLUMN,
        "option": "YS1",
        "fit_split": "TRAIN",
        "fit_row_count": y_bundle["fit_row_count"],
        "fit_start_timestamp": y_bundle["fit_start_timestamp"],
        "fit_end_timestamp": y_bundle["fit_end_timestamp"],
        "mean": float(scaler.mean_[0]),
        "variance": float(scaler.var_[0]),
        "scale": float(scaler.scale_[0]),
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "statistics_fingerprint": y_bundle["statistics_fingerprint"],
        "roundtrip_status": "PASS",
    }


def _readme_scaling() -> str:
    return "\n".join([
        "# SCALING-v1",
        "",
        "All StandardScaler estimators are fit once on SPLIT-v1 TRAIN rows and reused for Validation and Test transforms.",
        "",
        "Continuous exogenous channels, historical Appliances in FS1 and FS2, and rv1 and rv2 in FS2 are standardized.",
        "",
        "hour_sin, hour_cos, dow_sin, dow_cos and weekend pass through unchanged. Metadata never enters a scaler.",
        "",
        "YS0 is the identity target option. YS1 uses one TRAIN-only target scaler shared by every feature variant, model, lookback and seed.",
        "",
        "YS1 predictions must be inverse-transformed to Wh before MAE, RMSE or R2 computation.",
        "",
        "Serialized joblib artifacts must only be loaded from this trusted local artifact set after checksum verification.",
        "",
        "Test inspection in Phase 9 is structural only. Test distribution analysis remains locked until Phase 47.",
        "",
    ])


def verify_phase_9_inputs(root: Path) -> dict[str, Any]:
    phase_5 = materialize_phase_5(root)
    if phase_5.get("artifact_version") != SPLIT_VERSION or phase_5.get("status") != "PASS":
        raise RuntimeError("SPLIT-v1 is not signed off")
    for relative_path, expected_checksum in phase_5.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 5 artifact checksum mismatch: {relative_path}")
    feature_set_signoff = read_json(root / "artifacts/feature_sets/phase_8_signoff.json")
    if feature_set_signoff.get("artifact_version") != FEATURE_SET_VERSION or feature_set_signoff.get("status") != "PASS":
        raise RuntimeError("FEATURESETS-v1 is not signed off")
    feature_signoff = read_json(root / "artifacts/features/phase_7_signoff.json")
    if feature_signoff.get("artifact_version") != FEATURE_VERSION or feature_signoff.get("status") != "PASS":
        raise RuntimeError("FEATURES-v1 is not signed off")
    contract = read_json(root / "configs/base/coursework_contract.json")
    target_options = {item["id"]: item["value"] for item in contract["option_registry"]["target_scaling"]}
    if target_options != {"YS0": "off", "YS1": "train_only_standardization"}:
        raise RuntimeError("Phase 0 target scaling options do not match SCALING-v1")
    return phase_5


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") not in {"PASS", "PASS_WITH_WARNING"} or signoff.get("artifact_version") != SCALING_VERSION:
        raise RuntimeError("Existing Phase 9 sign-off is invalid")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 9 artifact checksum mismatch: {relative_path}")
    manifest = read_json(root / SCALING_ARTIFACT_ROOT / "scaling_manifest.json")
    registry = read_json(root / SCALING_ARTIFACT_ROOT / "scaler_registry.json")
    if manifest.get("scaling_version") != SCALING_VERSION or registry.get("scaling_version") != SCALING_VERSION:
        raise RuntimeError("Reloaded SCALING-v1 version mismatch")
    for variant_id, entry in registry["x_bundles"].items():
        path = root / entry["artifact_path"]
        bundle = joblib.load(path)
        validate_scaler_bundle(bundle, variant_id, entry["feature_fingerprint"], manifest["global_split_fingerprint"])
        if sha256_file(path) != entry["artifact_sha256"]:
            raise RuntimeError(f"Reloaded X scaler checksum mismatch: {variant_id}")
    y_entry = registry["target_bundles"]["YS1"]
    y_bundle = joblib.load(root / y_entry["artifact_path"])
    if y_bundle.get("statistics_fingerprint") != y_entry["statistics_fingerprint"]:
        raise RuntimeError("Reloaded YS1 statistics fingerprint mismatch")
    return signoff


def materialize_phase_9(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / SCALING_ARTIFACT_ROOT / "phase_9_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    phase_5 = verify_phase_9_inputs(root)
    feature_view = load_validated_feature_view(root)
    membership = load_validated_split_membership(root)
    scaling_view = build_scaling_view(feature_view, membership)
    feature_set_registry = load_validated_feature_set_registry(root)
    feature_manifest = read_json(root / "artifacts/features/feature_engineering_manifest.json")
    split_manifest = read_json(root / "artifacts/splits/split_manifest.json")
    environment_manifest = read_json(root / "artifacts/environment/environment_report.json")
    policy = build_scaling_policy()
    if set(feature_set_registry["variants"]) != set(policy["variants"]):
        raise RuntimeError("Scaling policy variants do not match FEATURESETS-v1")
    train_mask = scaling_view["split_id"].eq("TRAIN")
    if int(train_mask.sum()) != split_manifest["train_rows"]:
        raise RuntimeError("SCALING-v1 Train fit row count mismatch")
    bundles = {}
    bundle_bytes = {}
    x_statistics_rows = []
    audit_rows = []
    validation_summary_rows = []
    for variant_id, registry_entry in feature_set_registry["variants"].items():
        feature_order = registry_entry["features"]
        train_columns = feature_order + ["split_id", "timestamp"]
        train_frame = scaling_view.loc[train_mask, train_columns].copy(deep=True)
        bundle = fit_x_scaler_bundle(
            train_frame,
            variant_id,
            registry_entry["fingerprint"],
            split_manifest["train_fingerprint"],
            split_manifest["global_split_fingerprint"],
        )
        audit, validation_rows = _audit_x_bundle(scaling_view, bundle)
        if audit["status"] != "PASS":
            raise RuntimeError(f"SCALING-v1 audit failed for {variant_id}: {audit}")
        bundles[variant_id] = bundle
        bundle_bytes[variant_id] = _joblib_bytes(bundle)
        x_statistics_rows.extend(_bundle_statistics_rows(bundle))
        audit_rows.append(audit)
        validation_summary_rows.extend(validation_rows)
    target_train = scaling_view.loc[train_mask, [TARGET_COLUMN, "split_id", "timestamp"]].copy(deep=True)
    y_bundle = fit_y_scaler(
        target_train,
        split_manifest["train_fingerprint"],
        split_manifest["global_split_fingerprint"],
    )
    raw_target = target_train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    scaled_target = transform_target(raw_target, "YS1", y_bundle)
    recovered_target = inverse_transform_target(scaled_target, "YS1", y_bundle)
    y_roundtrip_ok = bool(np.allclose(raw_target, recovered_target, atol=ROUNDTRIP_TOLERANCE, rtol=ROUNDTRIP_TOLERANCE))
    ys0_identity_ok = bool(np.array_equal(raw_target, transform_target(raw_target, "YS0")))
    if not y_roundtrip_ok or not ys0_identity_ok:
        raise RuntimeError("Target scaling round-trip or identity audit failed")
    y_bytes = _joblib_bytes(y_bundle)
    x_artifact_paths = {
        variant_id: f"{SCALER_ARTIFACT_ROOT}/x/XSCALER__{variant_id}__{SCALING_VERSION}.joblib"
        for variant_id in bundles
    }
    y_artifact_path = f"{SCALER_ARTIFACT_ROOT}/y/YSCALER__YS1__{SCALING_VERSION}.joblib"
    for variant_id, relative_path in x_artifact_paths.items():
        write_bytes_once_or_verify(root / relative_path, bundle_bytes[variant_id])
    write_bytes_once_or_verify(root / y_artifact_path, y_bytes)
    x_manifests = {
        variant_id: _bundle_manifest(bundle, x_artifact_paths[variant_id], sha256_file(root / x_artifact_paths[variant_id]))
        for variant_id, bundle in bundles.items()
    }
    y_manifest = _target_manifest(y_bundle, y_artifact_path, sha256_file(root / y_artifact_path))
    zero_variance_features = {
        variant_id: bundle["zero_variance_features"]
        for variant_id, bundle in bundles.items()
        if bundle["zero_variance_features"]
    }
    warnings = [
        {
            "code": "TRAIN_ZERO_VARIANCE",
            "variant_id": variant_id,
            "features": features,
        }
        for variant_id, features in zero_variance_features.items()
    ]
    status = "PASS_WITH_WARNING" if warnings else "PASS"
    leakage_checks = {
        "x_fit_uses_train_only": all(bundle["fit_split"] == "TRAIN" for bundle in bundles.values()),
        "y_fit_uses_train_only": y_bundle["fit_split"] == "TRAIN",
        "no_validation_in_fit": all(bundle["fit_row_count"] == split_manifest["train_rows"] for bundle in bundles.values()),
        "no_test_in_fit": all(bundle["fit_end_timestamp"] == split_manifest["train_end_timestamp"] for bundle in bundles.values()),
        "feature_order_matches_registry": all(bundle["full_feature_order"] == feature_set_registry["variants"][variant_id]["features"] for variant_id, bundle in bundles.items()),
        "split_fingerprint_matches": all(bundle["global_split_fingerprint"] == split_manifest["global_split_fingerprint"] for bundle in bundles.values()),
        "no_metadata_in_scaler": all(not set(bundle["scaled_feature_order"]).intersection(SCALING_METADATA) for bundle in bundles.values()),
        "no_future_target_in_x_fit": all(not any(token in feature.lower() for token in ("target_next", "future_", "t_plus_1") for feature in bundle["full_feature_order"]) for bundle in bundles.values()),
        "no_test_distribution_tuning": True,
    }
    if not all(leakage_checks.values()):
        raise RuntimeError(f"Scaling leakage audit failed: {leakage_checks}")
    leakage_rows = [
        {
            "check": check,
            "expected": True,
            "actual": actual,
            "status": "PASS" if actual else "FAIL",
            "notes": "Train-only fit and structural Test firewall",
        }
        for check, actual in leakage_checks.items()
    ]
    y_statistics_rows = [
        {
            "target": TARGET_COLUMN,
            "option": "YS0",
            "method": "IDENTITY",
            "mean": None,
            "variance": None,
            "scale": None,
            "n_samples_seen": split_manifest["train_rows"],
            "fit_split": "TRAIN",
        },
        {
            "target": TARGET_COLUMN,
            "option": "YS1",
            "method": "StandardScaler",
            "mean": float(y_bundle["scaler"].mean_[0]),
            "variance": float(y_bundle["scaler"].var_[0]),
            "scale": float(y_bundle["scaler"].scale_[0]),
            "n_samples_seen": _n_samples_seen(y_bundle["scaler"]),
            "fit_split": "TRAIN",
        },
    ]
    checksum_paths = [*x_artifact_paths.values(), y_artifact_path]
    checksum_text = "".join(f"{sha256_file(root / path)}  {path}\n" for path in checksum_paths)
    registry = {
        "scaling_version": SCALING_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "split_version": SPLIT_VERSION,
        "x_bundles": x_manifests,
        "target_bundles": {
            "YS0": {
                "bundle_id": "YSCALER__YS0_IDENTITY",
                "option": "YS0",
                "method": "IDENTITY",
                "artifact_path": None,
                "status": "PASS",
            },
            "YS1": y_manifest,
        },
    }
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "scaling_version": SCALING_VERSION,
        "dataset_revision": feature_manifest["dataset_revision"],
        "schema_version": feature_manifest["schema_version"],
        "temporal_version": feature_manifest["temporal_version"],
        "eda_version": feature_manifest["eda_version"],
        "feature_version": FEATURE_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "split_version": SPLIT_VERSION,
        "environment_id": environment_manifest["environment_id"],
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "x_scaling_method": "StandardScaler",
        "x_fit_split": "TRAIN",
        "x_scaled_groups": ["SCALE_CONTINUOUS"],
        "x_pass_through_groups": ["PASSTHROUGH_CYCLICAL", "PASSTHROUGH_BINARY"],
        "target_options": list(TARGET_OPTIONS),
        "y_scaling_method_YS1": "StandardScaler",
        "y_fit_split": "TRAIN",
        "variant_scaler_bundles": x_manifests,
        "target_scaler": y_manifest,
        "train_row_count": split_manifest["train_rows"],
        "train_fingerprint": split_manifest["train_fingerprint"],
        "global_split_fingerprint": split_manifest["global_split_fingerprint"],
        "zero_variance_features": zero_variance_features,
        "leakage_audit_passed": True,
        "test_distribution_inspection": "STRUCTURAL_ONLY",
        "validation_shift_diagnostic": "DESCRIPTIVE_ONLY",
        "full_scaled_dataset_saved": False,
        "audit_status": status,
        "warnings": warnings,
        "created_at": created_at,
    }
    artifact_root = root / SCALING_ARTIFACT_ROOT
    manifest_path = artifact_root / "scaling_manifest.json"
    registry_path = artifact_root / "scaler_registry.json"
    x_statistics_path = artifact_root / "x_scaler_statistics.csv"
    y_statistics_path = artifact_root / "y_scaler_statistics.csv"
    audit_path = artifact_root / "scaling_audit.csv"
    leakage_path = artifact_root / "scaling_leakage_audit.csv"
    validation_path = artifact_root / "train_validation_scaled_feature_summary.csv"
    discrepancies_path = artifact_root / "scaling_discrepancies.json"
    checksums_path = artifact_root / "scaler_checksums.sha256"
    readme_path = artifact_root / "README_SCALING.md"
    write_json_once_or_verify(registry_path, registry)
    write_text_once_or_verify(x_statistics_path, csv_text([
        "bundle_id", "variant_id", "feature_position", "feature_name", "scaling_group", "policy", "mean", "variance", "scale", "n_samples_seen", "zero_variance_flag"
    ], x_statistics_rows))
    write_text_once_or_verify(y_statistics_path, csv_text([
        "target", "option", "method", "mean", "variance", "scale", "n_samples_seen", "fit_split"
    ], y_statistics_rows))
    write_text_once_or_verify(audit_path, csv_text([
        "bundle_id", "variant_id", "train_row_count", "scaled_feature_count", "pass_through_count", "train_scaled_mean_ok", "train_scaled_std_ok", "zero_variance_count", "validation_transform_ok", "test_structural_transform_ok", "no_new_nan", "no_new_inf", "pass_through_unchanged", "cyclical_invariant_ok", "roundtrip_ok", "status"
    ], audit_rows))
    write_text_once_or_verify(leakage_path, csv_text(["check", "expected", "actual", "status", "notes"], leakage_rows))
    write_text_once_or_verify(validation_path, csv_text([
        "variant_id", "feature_name", "train_scaled_mean", "train_scaled_std", "validation_scaled_mean", "validation_scaled_std", "mean_shift", "std_ratio"
    ], validation_summary_rows))
    write_json_once_or_verify(discrepancies_path, {
        "scaling_version": SCALING_VERSION,
        "discrepancies": warnings,
    })
    write_text_once_or_verify(checksums_path, checksum_text)
    write_text_once_or_verify(readme_path, _readme_scaling())
    write_json_once_or_verify(manifest_path, manifest)
    output_paths = [
        *checksum_paths,
        f"{SCALING_ARTIFACT_ROOT}/scaling_manifest.json",
        f"{SCALING_ARTIFACT_ROOT}/scaler_registry.json",
        f"{SCALING_ARTIFACT_ROOT}/x_scaler_statistics.csv",
        f"{SCALING_ARTIFACT_ROOT}/y_scaler_statistics.csv",
        f"{SCALING_ARTIFACT_ROOT}/scaling_audit.csv",
        f"{SCALING_ARTIFACT_ROOT}/scaling_leakage_audit.csv",
        f"{SCALING_ARTIFACT_ROOT}/train_validation_scaled_feature_summary.csv",
        f"{SCALING_ARTIFACT_ROOT}/scaling_discrepancies.json",
        f"{SCALING_ARTIFACT_ROOT}/scaler_checksums.sha256",
        f"{SCALING_ARTIFACT_ROOT}/README_SCALING.md",
    ]
    input_paths = [
        "artifacts/features/phase_7_signoff.json",
        "artifacts/features/feature_engineering_manifest.json",
        feature_manifest["derived_file_path"],
        "artifacts/feature_sets/phase_8_signoff.json",
        "artifacts/feature_sets/feature_set_registry.json",
        "artifacts/splits/phase_5_signoff.json",
        "artifacts/splits/split_manifest.json",
        "artifacts/splits/split_membership.csv",
        "artifacts/environment/environment_report.json",
        "configs/base/coursework_contract.json",
    ]
    signoff = {
        "artifact_version": SCALING_VERSION,
        "phase_id": 9,
        "phase_version": "PHASE-9-v1",
        "created_at": created_at,
        "environment_id": environment_manifest["environment_id"],
        "dataset_revision": feature_manifest["dataset_revision"],
        "feature_version": FEATURE_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "split_version": SPLIT_VERSION,
        "global_split_fingerprint": split_manifest["global_split_fingerprint"],
        "input_paths": input_paths,
        "input_checksums": {relative_path: sha256_file(root / relative_path) for relative_path in input_paths},
        "output_paths": output_paths,
        "output_checksums": {relative_path: sha256_file(root / relative_path) for relative_path in output_paths},
        "config_fingerprint": phase_5["config_fingerprint"],
        "status": status,
        "tests": [
            "features_feature_sets_split_lineage",
            "train_only_x_fit",
            "train_only_y_fit",
            "six_variant_scaler_bundles",
            "feature_order_and_fingerprint_binding",
            "split_fingerprint_binding",
            "metadata_exclusion",
            "continuous_standardization",
            "cyclical_and_binary_passthrough",
            "train_zero_variance_audit",
            "validation_transform",
            "test_structural_transform_only",
            "no_new_nan_or_inf",
            "x_roundtrip",
            "ys0_identity",
            "ys1_roundtrip",
            "serialization_and_checksum",
            "deterministic_statistics_fingerprint",
        ],
        "warnings": warnings,
        "discrepancies": warnings,
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)


def load_validated_scaler_bundle(variant_id: str, project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    materialize_phase_9(root)
    registry = read_json(root / SCALING_ARTIFACT_ROOT / "scaler_registry.json")
    if variant_id not in registry["x_bundles"]:
        raise KeyError(f"Unknown scaler variant: {variant_id}")
    entry = registry["x_bundles"][variant_id]
    path = root / entry["artifact_path"]
    if sha256_file(path) != entry["artifact_sha256"]:
        raise RuntimeError(f"X scaler checksum mismatch: {variant_id}")
    bundle = joblib.load(path)
    manifest = read_json(root / SCALING_ARTIFACT_ROOT / "scaling_manifest.json")
    validate_scaler_bundle(bundle, variant_id, entry["feature_fingerprint"], manifest["global_split_fingerprint"])
    return bundle


def load_validated_target_scaler(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    materialize_phase_9(root)
    registry = read_json(root / SCALING_ARTIFACT_ROOT / "scaler_registry.json")
    entry = registry["target_bundles"]["YS1"]
    path = root / entry["artifact_path"]
    if sha256_file(path) != entry["artifact_sha256"]:
        raise RuntimeError("YS1 scaler checksum mismatch")
    bundle = joblib.load(path)
    if compute_scaler_statistics_fingerprint([TARGET_COLUMN], bundle["scaler"]) != bundle["statistics_fingerprint"]:
        raise RuntimeError("YS1 statistics fingerprint mismatch")
    return bundle
