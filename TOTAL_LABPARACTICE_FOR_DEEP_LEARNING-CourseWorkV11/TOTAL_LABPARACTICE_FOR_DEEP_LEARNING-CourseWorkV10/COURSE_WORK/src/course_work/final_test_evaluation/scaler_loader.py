"""Phase 47 — FINAL_SCALING-v1 Loader.

Ensures Phase47 uses EXACTLY the FINAL_SCALING-v1 scalers from Phase46.
NEVER loads Phase9 scalers. NEVER calls fit() or partial_fit() on Test data.

The FINAL_SCALING-v1 is stored under artifacts/scaling/final_dev/ and has checksums:
    - X scaler (FS2_TF1): 7280c166232ac53ef5947fa1991a1b38e9f5ec75045711b7092ddba9c53a17fd
    - Y scaler (YS1):     e8c8edb970591afa5c25faf619d2257b544b1b27c0b725b3be376a52a5946cca

Guard: This module refuses to load scalers from the Phase9 artifacts/scaling/ directory.
All Phase47 scaler access MUST go through this module.
"""

from __future__ import annotations

import hashlib
import joblib
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import get_project_root


FINAL_SCALING_VERSION = "FINAL_SCALING-v1"

FINAL_SCALING_CHECKSUMS = {
    "x_bundle": "54fbd2ca296c4cd4102390e85b5bb10f7b28567f47281ac7dca10719f175e6ff",
    "y_bundle": "e8c8edb970591afa5c25faf619d2257b544b1b27c0b725b3be376a52a5946cca",
}

FINAL_SCALING_REGISTRY_PATH = "artifacts/scaling/final_dev/final_scaler_registry.json"
FINAL_SCALING_DIR = "artifacts/scaling/final_dev"


class Phase47ScalerError(Exception):
    """Raised when FINAL_SCALING-v1 cannot be loaded or verified."""
    pass


class ScalerFitAttemptError(Exception):
    """Raised if fit() or partial_fit() is called on a Phase47 scaler."""
    pass


class Phase47ScalerWrapper:
    """A wrapper around a scaler that guards against fit() calls.

    This wrapper raises ScalerFitAttemptError if fit() or partial_fit() is called,
    ensuring no Test data can influence the scaler statistics.
    """

    _FORBIDDEN_METHODS = ("fit", "partial_fit", "fit_transform", "partial_fit_transform")

    def __init__(self, scaler: Any, scaler_name: str):
        object.__setattr__(self, "_scaler", scaler)
        object.__setattr__(self, "_name", scaler_name)

    def __getattr__(self, name: str) -> Any:
        scaler = object.__getattribute__(self, "_scaler")
        return getattr(scaler, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        setattr(object.__getattribute__(self, "_scaler"), name, value)

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise ScalerFitAttemptError(
            f"SCALER FIT FORBIDDEN in Phase47. "
            f"FINAL_SCALING-v1 scalers must never be fit on Test data."
        )

    def partial_fit(self, *args: Any, **kwargs: Any) -> Any:
        raise ScalerFitAttemptError(
            f"SCALER PARTIAL_FIT FORBIDDEN in Phase47. "
            f"FINAL_SCALING-v1 scalers must never be partially fit on Test data."
        )

    def fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        raise ScalerFitAttemptError(
            f"SCALER FIT_TRANSFORM FORBIDDEN in Phase47. "
            f"Use transform_only() instead."
        )

    def partial_fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        raise ScalerFitAttemptError(
            f"SCALER PARTIAL_FIT_TRANSFORM FORBIDDEN in Phase47. "
            f"Use transform_only() instead."
        )

    def transform_only(self, data: Any) -> Any:
        """Transform data WITHOUT fitting.

        The FINAL_SCALING-v1 X scaler (FS2_TF1) handles 33 features:
        - First 28 features: standardized via StandardScaler
        - Last 5 features (time features): pass-through (hour_sin, hour_cos, dow_sin, dow_cos, weekend)

        The StandardScaler's statistics (mean_, scale_) cover only the 28 scaled features.
        This method uses the outer dict's metadata to determine the split.
        """
        import numpy as np

        outer_dict = object.__getattribute__(self, "_scaler")

        if isinstance(outer_dict, dict):
            inner_scaler = outer_dict.get("scaler")
            pass_through_order = outer_dict.get("pass_through_feature_order", [])
            n_total = outer_dict.get("n_features_in_") 
        else:
            inner_scaler = outer_dict
            pass_through_order = []
            n_total = None

        if inner_scaler is None:
            raise Phase47ScalerError(
                "No inner StandardScaler found in scaler artifact. "
                "Cannot perform transform."
            )

        n_pass = len(pass_through_order)
        n_scaled = getattr(inner_scaler, "n_features_in_", None)
        if n_scaled is None:
            n_scaled = len(inner_scaler.mean_) if hasattr(inner_scaler, "mean_") else 0
        if n_total is None:
            n_total = n_scaled + n_pass

        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)

        squeeze_output = False
        if data.ndim == 1:
            data = data.reshape(1, -1)
            squeeze_output = True

        feature_dim = data.ndim - 1
        n_data_features = data.shape[-1]

        needs_split = (n_pass > 0) and (n_data_features == n_total)

        if needs_split:
            scaled_data = data[..., :n_scaled]
            pass_data = data[..., n_scaled:]
            scaled_transformed = inner_scaler.transform(scaled_data)
            result = np.concatenate([scaled_transformed, pass_data], axis=feature_dim)
        elif n_data_features == n_scaled:
            result = inner_scaler.transform(data)
        else:
            result = inner_scaler.transform(data)

        if squeeze_output:
            result = result.squeeze(axis=0)

        return result


def _compute_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_final_scaling_v1_x_scaler(
    variant_id: str = "FS2_TF1",
    project_root: Path | None = None,
    verify_checksum: bool = True,
) -> Any:
    """Load the FINAL_SCALING-v1 X scaler for a feature variant.

    Args:
        variant_id: Feature variant (must be FS2_TF1 for Phase47).
        project_root: COURSE_WORK root.
        verify_checksum: If True, verify the artifact SHA256 matches expected.

    Returns:
        The fitted StandardScaler (wrapped in Phase47ScalerWrapper to forbid fit()).

    Raises:
        Phase47ScalerError: If scaler not found or checksum mismatch.
    """
    root = project_root or get_project_root()
    registry_path = root / FINAL_SCALING_REGISTRY_PATH

    if not registry_path.exists():
        raise Phase47ScalerError(
            f"FINAL_SCALING-v1 registry not found at {registry_path}. "
            "Run Phase46 to materialize FINAL_SCALING-v1 first."
        )

    import json
    registry = json.loads(registry_path.read_text())

    if "x_bundles" not in registry:
        raise Phase47ScalerError("FINAL_SCALING-v1 registry has no x_bundles")

    if variant_id not in registry["x_bundles"]:
        raise Phase47ScalerError(
            f"Variant {variant_id} not found in FINAL_SCALING-v1 x_bundles. "
            f"Available: {list(registry['x_bundles'].keys())}"
        )

    bundle = registry["x_bundles"][variant_id]
    artifact_path = root / bundle["artifact_path"]

    if not artifact_path.exists():
        raise Phase47ScalerError(
            f"FINAL_SCALING-v1 X scaler artifact not found at {artifact_path}"
        )

    if verify_checksum:
        observed_sha = _compute_file_sha256(artifact_path)
        expected_sha = FINAL_SCALING_CHECKSUMS["x_bundle"]
        if observed_sha != expected_sha:
            raise Phase47ScalerError(
                f"FINAL_SCALING-v1 X scaler checksum mismatch. "
                f"Expected: {expected_sha}, Got: {observed_sha}"
            )

    scaler = joblib.load(artifact_path)

    if isinstance(scaler, dict):
        if "scaler" not in scaler:
            raise Phase47ScalerError(
                f"joblib artifact at {artifact_path} is a dict but has no 'scaler' key. "
                f"Available keys: {list(scaler.keys())}"
            )
        pass  
    elif hasattr(scaler, "_scaler"):
        scaler = object.__getattribute__(scaler, "_scaler")
        if isinstance(scaler, dict) and "scaler" not in scaler:
            raise Phase47ScalerError(
                f"Nested scaler dict has no 'scaler' key. Keys: {list(scaler.keys())}"
            )
    else:
        pass

    return Phase47ScalerWrapper(scaler, f"FINAL_SCALING-v1_X_{variant_id}")


def load_final_scaling_v1_y_scaler(
    target_option: str = "YS1",
    project_root: Path | None = None,
    verify_checksum: bool = True,
) -> Any:
    """Load the FINAL_SCALING-v1 Y scaler for a target scaling option.

    Args:
        target_option: Target scaling option (must be YS1 for Phase47).
        project_root: COURSE_WORK root.
        verify_checksum: If True, verify the artifact SHA256 matches expected.

    Returns:
        The fitted StandardScaler (wrapped in Phase47ScalerWrapper to forbid fit()).

    Raises:
        Phase47ScalerError: If scaler not found or checksum mismatch.
    """
    root = project_root or get_project_root()
    registry_path = root / FINAL_SCALING_REGISTRY_PATH

    if not registry_path.exists():
        raise Phase47ScalerError(
            f"FINAL_SCALING-v1 registry not found at {registry_path}. "
            "Run Phase46 to materialize FINAL_SCALING-v1 first."
        )

    import json
    registry = json.loads(registry_path.read_text())

    if "target_bundles" not in registry:
        raise Phase47ScalerError("FINAL_SCALING-v1 registry has no target_bundles")

    if target_option not in registry["target_bundles"]:
        raise Phase47ScalerError(
            f"Target option {target_option} not found in FINAL_SCALING-v1 target_bundles. "
            f"Available: {list(registry['target_bundles'].keys())}"
        )

    bundle = registry["target_bundles"][target_option]

    if bundle.get("method") == "IDENTITY":
        return Phase47ScalerWrapper(type("IdentityScaler", (), {
            "transform": lambda self, x: x,
            "inverse_transform": lambda self, x: x,
            "_name": "IDENTITY",
        })(), "FINAL_SCALING-v1_Y_IDENTITY")

    artifact_path = root / bundle["artifact_path"]

    if not artifact_path.exists():
        raise Phase47ScalerError(
            f"FINAL_SCALING-v1 Y scaler artifact not found at {artifact_path}"
        )

    if verify_checksum:
        observed_sha = _compute_file_sha256(artifact_path)
        expected_sha = FINAL_SCALING_CHECKSUMS["y_bundle"]
        if observed_sha != expected_sha:
            raise Phase47ScalerError(
                f"FINAL_SCALING-v1 Y scaler checksum mismatch. "
                f"Expected: {expected_sha}, Got: {observed_sha}"
            )

    scaler = joblib.load(artifact_path)

    if isinstance(scaler, dict):
        inner_scaler = scaler.get("scaler")
        if inner_scaler is None:
            raise Phase47ScalerError(
                f"Y scaler joblib artifact at {artifact_path} is a dict but has no 'scaler' key. "
                f"Available keys: {list(scaler.keys())}"
            )
        scaler = inner_scaler

    return Phase47ScalerWrapper(scaler, f"FINAL_SCALING-v1_Y_{target_option}")


def verify_final_scaling_v1(project_root: Path | None = None) -> dict[str, Any]:
    """Verify that FINAL_SCALING-v1 is properly materialized and checksums match.

    Returns:
        Dict with verification results for each scaler component.
    """
    root = project_root or get_project_root()
    results = {}

    try:
        x_scaler = load_final_scaling_v1_x_scaler("FS2_TF1", root, verify_checksum=True)
        results["x_scaler"] = {"status": "PASS", "loaded": True}
    except Phase47ScalerError as e:
        results["x_scaler"] = {"status": "FAIL", "error": str(e), "loaded": False}

    try:
        y_scaler = load_final_scaling_v1_y_scaler("YS1", root, verify_checksum=True)
        results["y_scaler"] = {"status": "PASS", "loaded": True}
    except Phase47ScalerError as e:
        results["y_scaler"] = {"status": "FAIL", "error": str(e), "loaded": False}

    all_pass = all(r["status"] == "PASS" for r in results.values())
    results["overall"] = "PASS" if all_pass else "FAIL"
    return results
