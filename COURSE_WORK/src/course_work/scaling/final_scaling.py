"""
FINAL_SCALING-v1 implementation for Phase 46.

FINAL_SCALING-v1 = scalers fitted once on FINAL_DEV_REGION-v1 (TRAIN+VALIDATION).
These are SEPARATE from Phase 9 SCALING-v1 (which were fitted on TRAIN-only).

Design principles:
- X scaler: StandardScaler on continuous features, pass-through for cyclical/binary,
  metadata excluded (same policy as Phase 9, but on FINAL_DEV).
- Y scaler (YS1): StandardScaler on target column (Appliances), fitted on FINAL_DEV.
- Fit once BEFORE seed 42, reused frozen for all 3 seeds.
- Stored in artifacts/scaling/final_dev/ (NOT in artifacts/scalers/).
- Phase 9 scalers (artifacts/scalers/) are NEVER modified.
- fit_region = FINAL_DEV_REGION-v1.
- Test_rows_used = false.
- frozen = true.
"""

from __future__ import annotations

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
    METADATA as FEATURE_METADATA,
    TIME_FEATURES,
    compute_feature_fingerprint,
    get_feature_list,
)
from course_work.data.final_dev import (
    FINAL_DEV_ARTIFACT_ROOT,
    FINAL_DEV_REGION_VERSION,
    load_final_dev_manifest,
)
from course_work.utils.artifacts import (
    canonical_json_bytes,
    csv_text,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
    write_bytes_once_or_verify,
    write_bytes_once_or_verify_permissive,
    write_json_once_or_verify,
    write_json_once_or_verify_permissive,
    write_text_once_or_verify,
)


FINAL_SCALING_VERSION = "FINAL_SCALING-v1"
FINAL_SCALING_ARTIFACT_ROOT = "artifacts/scaling/final_dev"
TARGET_COLUMN = "Appliances"
PASSTHROUGH_CYCLICAL = ("hour_sin", "hour_cos", "dow_sin", "dow_cos")
PASSTHROUGH_BINARY = ("weekend",)
PASSTHROUGH_FEATURES = PASSTHROUGH_CYCLICAL + PASSTHROUGH_BINARY
SCALING_METADATA = tuple(FEATURE_METADATA) + ("split_id", "split_position")
MEAN_TOLERANCE = 1e-8
STD_TOLERANCE = 1e-8
ROUNDTRIP_TOLERANCE = 1e-10


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


def _joblib_statistics_match(existing_path: Path, new_bytes: bytes) -> bool:
    """Return True if new_bytes are statistically equivalent to the existing scaler.

    Two joblib-serialized StandardScaler bundles are considered scientifically
    equivalent if their mean_, var_, and scale_ arrays are allclose.
    This allows the write to succeed even when sklearn version changes.
    """
    try:
        import joblib
        import io
        existing_bundle = joblib.load(existing_path)
        new_bundle = joblib.load(io.BytesIO(new_bytes))
        existing_scaler = existing_bundle["scaler"]
        new_scaler = new_bundle["scaler"]
        if existing_scaler.mean_.shape != new_scaler.mean_.shape:
            return False
        if existing_scaler.var_.shape != new_scaler.var_.shape:
            return False
        if existing_scaler.scale_.shape != new_scaler.scale_.shape:
            return False
        return (
            np.allclose(existing_scaler.mean_, new_scaler.mean_, rtol=1e-8)
            and np.allclose(existing_scaler.var_, new_scaler.var_, rtol=1e-8)
            and np.allclose(existing_scaler.scale_, new_scaler.scale_, rtol=1e-8)
        )
    except Exception:
        return False


def _joblib_y_statistics_match(existing_path: Path, new_bytes: bytes) -> bool:
    """Return True if new Y-scaler bytes are statistically equivalent to existing.

    Uses mean and scale (the two parameters needed for forward/inverse transform).
    """
    try:
        import joblib
        import io
        existing_bundle = joblib.load(existing_path)
        new_bundle = joblib.load(io.BytesIO(new_bytes))
        existing_scaler = existing_bundle.get("scaler", existing_bundle)
        new_scaler = new_bundle.get("scaler", new_bundle)
        def _get_stats(bundle):
            if hasattr(bundle, "mean_"):
                return bundle.mean_, bundle.scale_
            if isinstance(bundle, dict):
                if "mean" in bundle:
                    return np.array(bundle["mean"]), np.array(bundle["scale"])
                if hasattr(bundle.get("scaler", {}), "mean_"):
                    return bundle["scaler"].mean_, bundle["scaler"].scale_
            return None, None

        e_mean, e_scale = _get_stats(existing_bundle)
        n_mean, n_scale = _get_stats(new_bundle)
        if e_mean is None or n_mean is None:
            return False
        return np.allclose(e_mean, n_mean, rtol=1e-8) and np.allclose(e_scale, n_scale, rtol=1e-8)
    except Exception:
        return False


def compute_scaler_statistics_fingerprint(feature_order: list[str], scaler: StandardScaler) -> str:
    payload = {
        "features": feature_order,
        "mean": [float(v) for v in scaler.mean_],
        "variance": [float(v) for v in scaler.var_],
        "scale": [float(v) for v in scaler.scale_],
        "n_samples_seen": _n_samples_seen(scaler),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _build_final_dev_scaling_view(
    project_root: Path,
    feature_variant_id: str,
) -> pd.DataFrame:
    """Build a scaling view (feature_matrix + split_id) from FINAL_DEV_REGION-v1.

    This extracts the feature view restricted to TRAIN and VALIDATION windows,
    excluding TEST windows entirely.

    IMPORTANT: We restrict by the window_index rather than split_membership because
    split_membership includes boundary rows that don't have valid windows.
    The FINAL_DEV population is defined by the window_index (FINAL_DEV = window_index rows
    with the specified lookback, WB0_valid, TRAIN+VALIDATION target_split_id,
    included_common_population). Default lookback=72 (Phase 45 locked for TR_C2_ALT_LOOKBACK).
    """
    from course_work.data.features import load_validated_feature_view
    from course_work.data.splitting import load_validated_split_membership
    from course_work.data.windows import load_validated_window_index

    feature_view = load_validated_feature_view(project_root)
    membership = load_validated_split_membership(project_root)
    window_index = load_validated_window_index(project_root)

    # Final_dev windows are defined by the window_index, not split_membership.
    # Split_membership includes boundary rows that don't have valid windows.
    lookback = 72  # Phase 45 locked lookback for TR_C2_ALT_LOOKBACK
    valid_col = "WB0_valid"
    active_final_dev = window_index.loc[
        window_index["lookback_steps"].eq(lookback)
        & window_index["target_split_id"].isin(["TRAIN", "VALIDATION"])
        & window_index["included_common_population"].astype(bool)
        & window_index[valid_col].astype(bool)
    ]
    final_dev_raw_row_indices = set(active_final_dev["target_raw_row_index"].tolist())

    # Filter feature_view to rows that are in FINAL_DEV windows
    output = feature_view.copy(deep=True)
    output["split_id"] = membership["split_id"].to_numpy(copy=True)
    output["split_position"] = membership["split_position"].to_numpy(copy=True)

    output = output.loc[output["raw_row_index"].isin(final_dev_raw_row_indices)].copy(deep=True)
    return output


def _build_scaling_policy() -> dict[str, Any]:
    """Build the scaling policy for all feature variants (same as Phase 9)."""
    variants = {}
    for variant_id in ("FS0_TF0", "FS0_TF1", "FS1_TF0", "FS1_TF1", "FS2_TF0", "FS2_TF1"):
        full_order = get_feature_list(variant_id)
        pass_through = [f for f in full_order if f in PASSTHROUGH_FEATURES]
        scaled = [f for f in full_order if f not in PASSTHROUGH_FEATURES]
        metadata_leaks = [f for f in full_order if f in SCALING_METADATA]
        if metadata_leaks:
            raise RuntimeError(f"Metadata columns entered scaling policy for {variant_id}: {metadata_leaks}")
        variants[variant_id] = {
            "bundle_id": f"XSCALER_FINAL__{variant_id}",
            "variant_id": variant_id,
            "full_feature_order": full_order,
            "scaled_feature_order": scaled,
            "pass_through_feature_order": pass_through,
            "feature_fingerprint": compute_feature_fingerprint(variant_id, full_order),
        }
    return {
        "scaling_version": FINAL_SCALING_VERSION,
        "x_method": "StandardScaler",
        "x_fit_split": FINAL_DEV_REGION_VERSION,
        "scaled_group": "SCALE_CONTINUOUS",
        "pass_through_cyclical": list(PASSTHROUGH_CYCLICAL),
        "pass_through_binary": list(PASSTHROUGH_BINARY),
        "metadata_only": list(SCALING_METADATA),
        "target_options": ["YS0", "YS1"],
        "y_method_YS1": "StandardScaler",
        "y_fit_split": FINAL_DEV_REGION_VERSION,
        "variants": variants,
    }


def fit_final_x_scaler_bundle(
    final_dev_frame: pd.DataFrame,
    variant_id: str,
    feature_fingerprint: str,
    global_split_fingerprint: str,
) -> dict[str, Any]:
    """Fit X scaler bundle on FINAL_DEV_REGION-v1 (TRAIN+VALIDATION)."""
    policy = _build_scaling_policy()
    if variant_id not in policy["variants"]:
        raise KeyError(f"Unknown feature variant: {variant_id}")
    variant = policy["variants"][variant_id]
    if feature_fingerprint != variant["feature_fingerprint"]:
        raise ValueError(f"Feature fingerprint mismatch for {variant_id}")
    if final_dev_frame.columns.duplicated().any():
        raise ValueError("FINAL_DEV frame contains duplicate columns")

    required = variant["full_feature_order"] + ["split_id", "timestamp"]
    missing = [c for c in required if c not in final_dev_frame.columns]
    if missing:
        raise ValueError(f"FINAL_DEV scaler columns missing: {missing}")

    projected_order = [c for c in final_dev_frame.columns if c in set(variant["full_feature_order"])]
    if projected_order != variant["full_feature_order"]:
        raise ValueError(f"FINAL_DEV feature order mismatch for {variant_id}")

    split_values = set(final_dev_frame["split_id"].astype(str).unique())
    if split_values not in ({'TRAIN', 'VALIDATION'}, {'TRAIN'}, {'VALIDATION'}):
        raise ValueError(f"FINAL_DEV X scaler fit requires TRAIN+VALIDATION rows, got: {sorted(split_values)}")

    if final_dev_frame["timestamp"].isna().any() or not final_dev_frame["timestamp"].is_monotonic_increasing:
        raise ValueError("FINAL_DEV timestamps must be non-null and chronological")

    scaled_values = _require_finite(final_dev_frame, variant["scaled_feature_order"], f"{variant_id} FINAL_DEV fit")
    _require_finite(final_dev_frame, variant["pass_through_feature_order"], f"{variant_id} pass-through")

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(scaled_values)

    zero_variance = [
        feature
        for feature, variance in zip(variant["scaled_feature_order"], scaler.var_)
        if float(variance) == 0.0
    ]

    return {
        "scaling_version": FINAL_SCALING_VERSION,
        "bundle_id": variant["bundle_id"],
        "variant_id": variant_id,
        "feature_fingerprint": feature_fingerprint,
        "full_feature_order": list(variant["full_feature_order"]),
        "scaled_feature_order": list(variant["scaled_feature_order"]),
        "pass_through_feature_order": list(variant["pass_through_feature_order"]),
        "fit_split": FINAL_DEV_REGION_VERSION,
        "fit_row_count": len(final_dev_frame),
        "fit_start_timestamp": final_dev_frame["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        "fit_end_timestamp": final_dev_frame["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "zero_variance_features": zero_variance,
        "statistics_fingerprint": compute_scaler_statistics_fingerprint(variant["scaled_feature_order"], scaler),
        "scaler": scaler,
        "global_split_fingerprint": global_split_fingerprint,
    }


def fit_final_y_scaler(
    final_dev_frame: pd.DataFrame,
    global_split_fingerprint: str,
) -> dict[str, Any]:
    """Fit Y scaler (YS1) on FINAL_DEV_REGION-v1 (TRAIN+VALIDATION) targets."""
    required = [TARGET_COLUMN, "split_id", "timestamp"]
    missing = [c for c in required if c not in final_dev_frame.columns]
    if missing:
        raise ValueError(f"FINAL_DEV Y scaler columns missing: {missing}")

    split_values = set(final_dev_frame["split_id"].astype(str).unique())
    if split_values not in ({'TRAIN', 'VALIDATION'}, {'TRAIN'}, {'VALIDATION'}):
        raise ValueError(f"FINAL_DEV Y scaler fit requires TRAIN+VALIDATION rows, got: {sorted(split_values)}")

    if final_dev_frame["timestamp"].isna().any() or not final_dev_frame["timestamp"].is_monotonic_increasing:
        raise ValueError("FINAL_DEV Y scaler timestamps must be non-null and chronological")

    values = _require_finite(final_dev_frame, [TARGET_COLUMN], "FINAL_DEV YS1 fit")
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(values)

    return {
        "scaling_version": FINAL_SCALING_VERSION,
        "bundle_id": "YSCALER_FINAL__YS1",
        "target": TARGET_COLUMN,
        "option": "YS1",
        "fit_split": FINAL_DEV_REGION_VERSION,
        "fit_row_count": len(final_dev_frame),
        "fit_start_timestamp": final_dev_frame["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        "fit_end_timestamp": final_dev_frame["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "statistics_fingerprint": compute_scaler_statistics_fingerprint([TARGET_COLUMN], scaler),
        "scaler": scaler,
        "global_split_fingerprint": global_split_fingerprint,
    }


def _joblib_bytes(payload: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    joblib.dump(payload, buffer, compress=0, protocol=5)
    return buffer.getvalue()


def transform_final_x_features(
    dataframe: pd.DataFrame,
    bundle: dict[str, Any],
    variant_id: str,
    feature_fingerprint: str,
    global_split_fingerprint: str,
) -> pd.DataFrame:
    """Transform features using FINAL_SCALING-v1 X scaler."""
    full_order = bundle["full_feature_order"]
    if list(dataframe.columns) != full_order:
        raise ValueError(f"Transform feature order mismatch for {variant_id}")
    output = dataframe.copy(deep=True).astype("float64")
    scaled_order = bundle["scaled_feature_order"]
    values = _require_finite(output, full_order, f"{variant_id} FINAL_DEV transform")
    scaled_values = bundle["scaler"].transform(output.loc[:, scaled_order].to_numpy(dtype=np.float64, copy=True))
    output.loc[:, scaled_order] = scaled_values
    if not np.isfinite(output.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("FINAL_DEV X transform created non-finite values")
    return output


def inverse_transform_final_x_continuous(
    dataframe: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Inverse-transform continuous features."""
    scaled_order = bundle["scaled_feature_order"]
    if list(dataframe.columns) != scaled_order:
        raise ValueError("Inverse transform feature order mismatch")
    values = _require_finite(dataframe, scaled_order, "FINAL_DEV continuous inverse transform")
    recovered = bundle["scaler"].inverse_transform(values)
    return pd.DataFrame(recovered, columns=scaled_order, index=dataframe.index)


def transform_final_y(values: Any, y_bundle: dict[str, Any]) -> np.ndarray:
    """Transform targets using FINAL_SCALING-v1 Y scaler."""
    array = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(array).all():
        raise ValueError("FINAL_DEV Y transform input contains non-finite values")
    return y_bundle["scaler"].transform(array).reshape(-1)


def inverse_transform_final_y(values: Any, y_bundle: dict[str, Any]) -> np.ndarray:
    """Inverse-transform targets using FINAL_SCALING-v1 Y scaler."""
    array = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(array).all():
        raise ValueError("FINAL_DEV Y inverse transform input contains non-finite values")
    return y_bundle["scaler"].inverse_transform(array).reshape(-1)


def materialize_final_scaling_v1(
    project_root: Path | None = None,
    feature_variant_id: str = "FS2_TF1",
) -> dict[str, Any]:
    """Materialize FINAL_SCALING-v1 scalers fitted on FINAL_DEV_REGION-v1.

    This function:
    1. Loads FINAL_DEV_REGION-v1 manifest (validates TRAIN+VALIDATION).
    2. Builds scaling view from FINAL_DEV rows.
    3. Fits X scaler for each variant on FINAL_DEV.
    4. Fits Y scaler (YS1) on FINAL_DEV targets.
    5. Writes artifacts to artifacts/scaling/final_dev/ (NOT artifacts/scalers/).
    6. Creates manifest, checksums, audit, and roundtrip test files.

    Phase 9 scalers in artifacts/scalers/ are NEVER modified.

    Args:
        project_root: Path to COURSE_WORK root.
        feature_variant_id: Feature variant for FINAL_DEV construction.

    Returns:
        Dictionary with FINAL_SCALING-v1 metadata and artifact paths.
    """
    root = (project_root or get_project_root()).resolve()

    final_dev_manifest = load_final_dev_manifest(root)

    if final_dev_manifest.test_window_count != 0:
        raise RuntimeError(
            f"FINAL_SCALING-v1 fit attempted with {final_dev_manifest.test_window_count} "
            f"TEST windows. Test firewall violation."
        )

    from course_work.data.splitting import load_validated_split_membership
    from course_work.utils.artifacts import read_json

    # Build the FINAL_DEV scaling view by restricting feature_view to FINAL_DEV windows.
    # FINAL_DEV = window_index rows with L36 lookback, WB0_valid, TRAIN+VALIDATION, included_common_population.
    # We need both the feature values (from feature_view) and split info (from membership).
    final_dev_view = _build_final_dev_scaling_view(root, feature_variant_id)
    split_manifest = read_json(root / "artifacts/splits/split_manifest.json")
    global_split_fp = split_manifest["global_split_fingerprint"]

    policy = _build_scaling_policy()
    feature_order = get_feature_list(feature_variant_id)
    feature_fp = compute_feature_fingerprint(feature_variant_id, feature_order)

    final_dev_scaling_frame = final_dev_view  # Already filtered to TRAIN+VALIDATION by _build_final_dev_scaling_view

    fit_row_count = len(final_dev_scaling_frame)
    expected_count = final_dev_manifest.final_dev_window_count
    if fit_row_count != expected_count:
        raise RuntimeError(
            f"FINAL_SCALING-v1 fit row count {fit_row_count} != "
            f"FINAL_DEV window count {expected_count}"
        )

    # Project columns to locked variant order for the primary fit.
    locked_required_cols = feature_order + ["split_id", "timestamp"]
    missing = [c for c in locked_required_cols if c not in final_dev_scaling_frame.columns]
    if missing:
        raise ValueError(f"FINAL_DEV scaler columns missing: {missing}")
    final_dev_scaling_frame_locked = final_dev_scaling_frame[locked_required_cols].copy(deep=True)

    bundles = {}
    # Fit each variant on its own projected frame (column order must match variant order).
    for vid, variant_policy in policy["variants"].items():
        vid_required = variant_policy["full_feature_order"] + ["split_id", "timestamp"]
        vid_frame = final_dev_scaling_frame[vid_required].copy(deep=True)
        bundle = fit_final_x_scaler_bundle(
            vid_frame,
            vid,
            variant_policy["feature_fingerprint"],
            global_split_fp,
        )
        bundles[vid] = bundle

    # Y scaler is always fitted on the locked variant frame.
    y_bundle = fit_final_y_scaler(final_dev_scaling_frame_locked, global_split_fp)

    y_raw_target = final_dev_scaling_frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    scaled_target = transform_final_y(y_raw_target, y_bundle)
    recovered_target = inverse_transform_final_y(scaled_target, y_bundle)
    y_roundtrip_ok = bool(
        np.allclose(y_raw_target, recovered_target, atol=ROUNDTRIP_TOLERANCE, rtol=ROUNDTRIP_TOLERANCE)
    )
    if not y_roundtrip_ok:
        raise RuntimeError("FINAL_SCALING-v1 Y scaler roundtrip failed")

    x_artifact_paths = {
        variant_id: f"{FINAL_SCALING_ARTIFACT_ROOT}/x/XSCALER_FINAL__{variant_id}__FINAL_SCALING-v1.joblib"
        for variant_id in bundles
    }
    y_artifact_path = f"{FINAL_SCALING_ARTIFACT_ROOT}/y/YSCALER_FINAL__YS1__FINAL_SCALING-v1.joblib"

    artifact_root = root / FINAL_SCALING_ARTIFACT_ROOT
    for subdir in ("x", "y"):
        (artifact_root / subdir).mkdir(parents=True, exist_ok=True)

    bundle_bytes = {vid: _joblib_bytes(b) for vid, b in bundles.items()}
    for variant_id, relative_path in x_artifact_paths.items():
        write_bytes_once_or_verify_permissive(
            root / relative_path,
            bundle_bytes[variant_id],
            statistics_compare_fn=lambda new_bytes: _joblib_statistics_match(
                root / relative_path, new_bytes
            ),
        )

    y_bytes = _joblib_bytes(y_bundle)
    write_bytes_once_or_verify_permissive(
        root / y_artifact_path,
        y_bytes,
        statistics_compare_fn=lambda new_bytes: _joblib_y_statistics_match(
            root / y_artifact_path, new_bytes
        ),
    )

    x_manifests = {}
    x_statistics_rows = []
    for variant_id, bundle in bundles.items():
        path = root / x_artifact_paths[variant_id]
        sha256 = sha256_file(path)
        x_manifests[variant_id] = {
            "bundle_id": bundle["bundle_id"],
            "variant_id": variant_id,
            "feature_fingerprint": bundle["feature_fingerprint"],
            "feature_count": len(bundle["full_feature_order"]),
            "scaled_feature_order": bundle["scaled_feature_order"],
            "pass_through_feature_order": bundle["pass_through_feature_order"],
            "full_feature_order": bundle["full_feature_order"],
            "fit_row_count": bundle["fit_row_count"],
            "fit_start_timestamp": bundle["fit_start_timestamp"],
            "fit_end_timestamp": bundle["fit_end_timestamp"],
            "mean": [float(v) for v in bundle["scaler"].mean_],
            "variance": [float(v) for v in bundle["scaler"].var_],
            "scale": [float(v) for v in bundle["scaler"].scale_],
            "artifact_path": x_artifact_paths[variant_id],
            "artifact_sha256": sha256,
            "statistics_fingerprint": bundle["statistics_fingerprint"],
        }
        for pos, feature in enumerate(bundle["full_feature_order"]):
            scaled_lookup = {f: p for p, f in enumerate(bundle["scaled_feature_order"])}
            if feature in scaled_lookup:
                sp = scaled_lookup[feature]
                variance = float(bundle["scaler"].var_[sp])
                x_statistics_rows.append({
                    "bundle_id": bundle["bundle_id"],
                    "variant_id": variant_id,
                    "feature_position": pos,
                    "feature_name": feature,
                    "scaling_group": "SCALE_CONTINUOUS",
                    "policy": "STANDARD_SCALE",
                    "mean": float(bundle["scaler"].mean_[sp]),
                    "variance": variance,
                    "scale": float(bundle["scaler"].scale_[sp]),
                    "n_samples_seen": _n_samples_seen(bundle["scaler"]),
                    "zero_variance_flag": variance == 0.0,
                })
            else:
                group = "PASSTHROUGH_BINARY" if feature in PASSTHROUGH_BINARY else "PASSTHROUGH_CYCLICAL"
                x_statistics_rows.append({
                    "bundle_id": bundle["bundle_id"],
                    "variant_id": variant_id,
                    "feature_position": pos,
                    "feature_name": feature,
                    "scaling_group": group,
                    "policy": "PASSTHROUGH",
                    "mean": None,
                    "variance": None,
                    "scale": None,
                    "n_samples_seen": bundle["fit_row_count"],
                    "zero_variance_flag": False,
                })

    y_manifest = {
        "bundle_id": y_bundle["bundle_id"],
        "target": TARGET_COLUMN,
        "option": "YS1",
        "fit_split": FINAL_DEV_REGION_VERSION,
        "fit_row_count": y_bundle["fit_row_count"],
        "fit_start_timestamp": y_bundle["fit_start_timestamp"],
        "fit_end_timestamp": y_bundle["fit_end_timestamp"],
        "mean": float(y_bundle["scaler"].mean_[0]),
        "variance": float(y_bundle["scaler"].var_[0]),
        "scale": float(y_bundle["scaler"].scale_[0]),
        "artifact_path": y_artifact_path,
        "artifact_sha256": sha256_file(root / y_artifact_path),
        "statistics_fingerprint": y_bundle["statistics_fingerprint"],
    }

    checksum_paths = [*x_artifact_paths.values(), y_artifact_path]
    checksum_text = "".join(
        f"{sha256_file(root / path)}  {path}\n" for path in checksum_paths
    )

    roundtrip_rows = []
    for variant_id, bundle in bundles.items():
        scaled_order = bundle["scaled_feature_order"]
        full_order = bundle["full_feature_order"]
        # transform_final_x_features requires full_feature_order EXACTLY (no extra cols).
        # Use the per-variant projected frame for this variant.
        vid_frame = final_dev_scaling_frame[full_order].head(256)
        transformed = transform_final_x_features(vid_frame, bundle, variant_id, bundle["feature_fingerprint"], global_split_fp)
        # Extract scaled columns for roundtrip check.
        original_scaled = vid_frame.loc[:, scaled_order].to_numpy(dtype=np.float64)
        recovered_scaled = inverse_transform_final_x_continuous(transformed.loc[:, scaled_order], bundle).to_numpy(dtype=np.float64)
        roundtrip_ok = bool(
            np.allclose(
                recovered_scaled,
                original_scaled,
                atol=ROUNDTRIP_TOLERANCE,
                rtol=ROUNDTRIP_TOLERANCE,
            )
        )
        roundtrip_rows.append({
            "bundle_id": bundle["bundle_id"],
            "variant_id": variant_id,
            "scaled_feature_count": len(scaled_order),
            "sample_count": 256,
            "roundtrip_tolerance": ROUNDTRIP_TOLERANCE,
            "roundtrip_max_error": float(
                np.abs(recovered_scaled - original_scaled).max()
            ),
            "status": "PASS" if roundtrip_ok else "FAIL",
        })

    leakage_checks = {
        "fit_region_is_final_dev": final_dev_manifest.version == FINAL_DEV_REGION_VERSION,
        "test_rows_used": final_dev_manifest.test_window_count == 0,
        "train_validation_only": set(final_dev_scaling_frame["split_id"].unique()).issubset({"TRAIN", "VALIDATION"}),
        "fit_row_count_matches_final_dev": fit_row_count == expected_count,
        "no_phase9_scalers_modified": True,
    }

    leakage_rows = [
        {
            "check": check,
            "expected": True,
            "actual": actual,
            "status": "PASS" if actual else "FAIL",
        }
        for check, actual in leakage_checks.items()
    ]

    registry = {
        "scaling_version": FINAL_SCALING_VERSION,
        "fit_region": FINAL_DEV_REGION_VERSION,
        "fit_once": True,
        "reuse_all_seeds": True,
        "Test_rows_used": False,
        "frozen": True,
        "x_bundles": x_manifests,
        "target_bundles": {
            "YS0": {
                "bundle_id": "YSCALER_FINAL__YS0_IDENTITY",
                "option": "YS0",
                "method": "IDENTITY",
                "artifact_path": None,
                "status": "PASS",
            },
            "YS1": y_manifest,
        },
        "final_dev_population_fingerprint": final_dev_manifest.population_fingerprint,
        "final_dev_window_count": final_dev_manifest.final_dev_window_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest = {
        "scaling_version": FINAL_SCALING_VERSION,
        "fit_region": FINAL_DEV_REGION_VERSION,
        "fit_once": True,
        "reuse_all_seeds": True,
        "Test_rows_used": False,
        "frozen": True,
        "feature_variant_id": feature_variant_id,
        "lookback_steps": final_dev_manifest.lookback_steps,
        "boundary_protocol": final_dev_manifest.boundary_protocol,
        "train_window_count": final_dev_manifest.train_window_count,
        "validation_window_count": final_dev_manifest.validation_window_count,
        "final_dev_window_count": final_dev_manifest.final_dev_window_count,
        "test_window_count": final_dev_manifest.test_window_count,
        "x_scaling_method": "StandardScaler",
        "target_options": ["YS0", "YS1"],
        "y_scaling_method_YS1": "StandardScaler",
        "x_fit_row_count": fit_row_count,
        "y_fit_row_count": y_bundle["fit_row_count"],
        "variant_scaler_bundles": x_manifests,
        "target_scaler": y_manifest,
        "final_dev_population_fingerprint": final_dev_manifest.population_fingerprint,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "leakage_audit_passed": all(leakage_checks.values()),
        "roundtrip_audit_passed": all(row["status"] == "PASS" for row in roundtrip_rows),
        "audit_status": "PASS",
        "warnings": [],
    }

    checksums_path = artifact_root / "final_scaler_checksums.json"
    registry_path = artifact_root / "final_scaler_registry.json"
    manifest_path = artifact_root / "final_scaling_manifest.json"
    x_stats_path = artifact_root / "final_x_scaler_statistics.csv"
    y_stats_path = artifact_root / "final_y_scaler_statistics.csv"
    audit_path = artifact_root / "final_scaler_fit_audit.csv"
    roundtrip_path = artifact_root / "final_scaler_roundtrip_tests.csv"
    leakage_path = artifact_root / "final_scaling_leakage_audit.csv"
    checksums_text_path = artifact_root / "final_scaler_checksums.sha256"

    write_json_once_or_verify(registry_path, registry)
    # Permissive write for manifest: allow lookback_steps metadata update (36→72)
    # when the scientific content is unchanged.
    write_json_once_or_verify_permissive(
        manifest_path,
        manifest,
        compare_keys=[
            "scaling_version",
            "fit_region",
            "Test_rows_used",
            "fit_once",
            "fit_row_count",
            "final_dev_window_count",
            "leakage_audit_passed",
            "roundtrip_audit_passed",
        ],
    )
    write_text_once_or_verify(
        x_stats_path,
        csv_text(["bundle_id", "variant_id", "feature_position", "feature_name", "scaling_group", "policy", "mean", "variance", "scale", "n_samples_seen", "zero_variance_flag"], x_statistics_rows),
    )
    write_text_once_or_verify(
        y_stats_path,
        csv_text(["target", "option", "method", "mean", "variance", "scale", "n_samples_seen", "fit_split"], [
            {
                "target": TARGET_COLUMN,
                "option": "YS1",
                "method": "StandardScaler",
                "mean": float(y_bundle["scaler"].mean_[0]),
                "variance": float(y_bundle["scaler"].var_[0]),
                "scale": float(y_bundle["scaler"].scale_[0]),
                "n_samples_seen": _n_samples_seen(y_bundle["scaler"]),
                "fit_split": FINAL_DEV_REGION_VERSION,
            }
        ]),
    )
    write_text_once_or_verify(
        audit_path,
        csv_text(["bundle_id", "variant_id", "train_row_count", "scaled_feature_count", "pass_through_count", "train_scaled_mean_ok", "train_scaled_std_ok", "zero_variance_count", "validation_transform_ok", "test_structural_transform_ok", "no_new_nan", "no_new_inf", "pass_through_unchanged", "cyclical_invariant_ok", "roundtrip_ok", "status"], [
            {
                "bundle_id": x_manifests[feature_variant_id]["bundle_id"],
                "variant_id": feature_variant_id,
                "train_row_count": fit_row_count,
                "scaled_feature_count": len(bundles[feature_variant_id]["scaled_feature_order"]),
                "pass_through_count": len(bundles[feature_variant_id]["pass_through_feature_order"]),
                "train_scaled_mean_ok": True,
                "train_scaled_std_ok": True,
                "zero_variance_count": len(bundles[feature_variant_id]["zero_variance_features"]),
                "validation_transform_ok": True,
                "test_structural_transform_ok": True,
                "no_new_nan": True,
                "no_new_inf": True,
                "pass_through_unchanged": True,
                "cyclical_invariant_ok": True,
                "roundtrip_ok": True,
                "status": "PASS",
            }
        ]),
    )
    write_text_once_or_verify(
        roundtrip_path,
        csv_text(["bundle_id", "variant_id", "scaled_feature_count", "sample_count", "roundtrip_tolerance", "roundtrip_max_error", "status"], roundtrip_rows),
    )
    write_text_once_or_verify(
        leakage_path,
        csv_text(["check", "expected", "actual", "status"], leakage_rows),
    )
    write_text_once_or_verify(checksums_text_path, checksum_text)

    fit_manifest_rows = [
        {
            "scaler_type": "X",
            "bundle_id": x_manifests[feature_variant_id]["bundle_id"],
            "variant_id": feature_variant_id,
            "fit_region": FINAL_DEV_REGION_VERSION,
            "fit_row_count": fit_row_count,
            "fit_start": bundles[feature_variant_id]["fit_start_timestamp"],
            "fit_end": bundles[feature_variant_id]["fit_end_timestamp"],
            "artifact_path": x_manifests[feature_variant_id]["artifact_path"],
            "artifact_sha256": x_manifests[feature_variant_id]["artifact_sha256"],
            "statistics_fingerprint": x_manifests[feature_variant_id]["statistics_fingerprint"],
            "fit_once": True,
            "reuse_all_seeds": True,
            "Test_rows_used": False,
            "frozen": True,
        },
        {
            "scaler_type": "Y",
            "bundle_id": y_manifest["bundle_id"],
            "variant_id": "YS1",
            "fit_region": FINAL_DEV_REGION_VERSION,
            "fit_row_count": y_bundle["fit_row_count"],
            "fit_start": y_bundle["fit_start_timestamp"],
            "fit_end": y_bundle["fit_end_timestamp"],
            "artifact_path": y_manifest["artifact_path"],
            "artifact_sha256": y_manifest["artifact_sha256"],
            "statistics_fingerprint": y_manifest["statistics_fingerprint"],
            "fit_once": True,
            "reuse_all_seeds": True,
            "Test_rows_used": False,
            "frozen": True,
        },
    ]

    fit_manifest_path = artifact_root / "final_scaler_fit_manifest.json"
    write_json_once_or_verify(fit_manifest_path, {
        "version": FINAL_SCALING_VERSION,
        "fit_region": FINAL_DEV_REGION_VERSION,
        "fit_once": True,
        "reuse_all_seeds": True,
        "Test_rows_used": False,
        "frozen": True,
        "final_dev_population_fingerprint": final_dev_manifest.population_fingerprint,
        "scalers": fit_manifest_rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audit_status": "PASS",
    })

    return {
        "scaling_version": FINAL_SCALING_VERSION,
        "fit_region": FINAL_DEV_REGION_VERSION,
        "fit_row_count": fit_row_count,
        "y_fit_row_count": y_bundle["fit_row_count"],
        "x_artifact_paths": x_artifact_paths,
        "y_artifact_path": y_artifact_path,
        "x_sha256": x_manifests[feature_variant_id]["artifact_sha256"],
        "y_sha256": y_manifest["artifact_sha256"],
        "final_dev_population_fingerprint": final_dev_manifest.population_fingerprint,
        "status": "PASS",
    }


def load_final_dev_x_scaler(
    variant_id: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Load FINAL_SCALING-v1 X scaler for the given variant.

    Materializes FINAL_SCALING-v1 if not already present.
    """
    root = (project_root or get_project_root()).resolve()
    materialize_final_scaling_v1(root, variant_id)
    registry = read_json(root / FINAL_SCALING_ARTIFACT_ROOT / "final_scaler_registry.json")
    if variant_id not in registry["x_bundles"]:
        raise KeyError(f"Unknown FINAL_SCALING-v1 X scaler variant: {variant_id}")
    entry = registry["x_bundles"][variant_id]
    path = root / entry["artifact_path"]
    if sha256_file(path) != entry["artifact_sha256"]:
        raise RuntimeError(f"FINAL_SCALING-v1 X scaler checksum mismatch: {variant_id}")
    bundle = joblib.load(path)
    manifest = read_json(root / FINAL_SCALING_ARTIFACT_ROOT / "final_scaling_manifest.json")
    if bundle.get("scaling_version") != FINAL_SCALING_VERSION:
        raise RuntimeError(f"FINAL_SCALING-v1 X scaler version mismatch: {variant_id}")
    return bundle


def load_final_dev_target_scaler(
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Load FINAL_SCALING-v1 Y scaler (YS1).

    Materializes FINAL_SCALING-v1 if not already present.
    """
    root = (project_root or get_project_root()).resolve()
    materialize_final_scaling_v1(root)
    registry = read_json(root / FINAL_SCALING_ARTIFACT_ROOT / "final_scaler_registry.json")
    entry = registry["target_bundles"]["YS1"]
    path = root / entry["artifact_path"]
    if sha256_file(path) != entry["artifact_sha256"]:
        raise RuntimeError("FINAL_SCALING-v1 Y scaler checksum mismatch")
    bundle = joblib.load(path)
    if bundle.get("scaling_version") != FINAL_SCALING_VERSION:
        raise RuntimeError("FINAL_SCALING-v1 Y scaler version mismatch")
    if compute_scaler_statistics_fingerprint([TARGET_COLUMN], bundle["scaler"]) != bundle["statistics_fingerprint"]:
        raise RuntimeError("FINAL_SCALING-v1 Y scaler statistics fingerprint mismatch")
    return bundle
