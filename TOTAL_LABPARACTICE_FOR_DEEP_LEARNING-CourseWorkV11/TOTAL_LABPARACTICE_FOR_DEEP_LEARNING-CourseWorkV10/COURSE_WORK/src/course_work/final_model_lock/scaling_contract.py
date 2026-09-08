"""FINAL_SCALING-v1 contract for Phase46.

Plan §29:
- Scalers fit ONCE on FINAL_DEV (TRAIN+VALIDATION).
- Same scalers reused across all 3 seeds.
- No Test rows in scaler fit.
- Time/binary features pass through (no scaling).
- RevIN bridge rules defined if active (RevIN is OFF in locked config).

Plan §114:
- Phase45 must NOT silently invent scaler checksums.
- If final-fit scalers are intentionally NOT materialized in Phase45,
  record x_scaler_bundle_checksum = REQUIRED_AT_PHASE46.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScalingContract:
    version: str
    fit_region: str
    X_scaler_semantics: str
    Y_scaler_semantics: str
    x_scaler_bundle_id: str
    y_scaler_bundle_id: str
    x_scaler_bundle_checksum: str  
    y_scaler_bundle_checksum: str
    fit_once: bool
    reuse_all_seeds: bool
    time_features_passthrough: bool
    binary_passthrough: bool
    revin_enabled: bool
    revin_bridge_rule: str
    Test_rows_used: bool
    status: str


def build_scaling_contract(
    locked_config: dict[str, Any],
    *,
    materialize_final_fit: bool = False,
) -> ScalingContract:
    """Construct the FINAL_SCALING-v1 contract.

    If ``materialize_final_fit`` is True, the caller is responsible for actually
    fitting the scalers and computing their checksums (no-train: still fits
    scaler objects, not the model). If False, we record
    ``REQUIRED_AT_PHASE46`` placeholders per plan §114.
    """
    scaling_v = locked_config.get("lineage", {}).get("scaling_version", "SCALING-v1")
    x_bundle_id = locked_config.get("lineage", {}).get("scaler_bundle_id", "XSCALER__FS2_TF1")
    y_bundle_id = locked_config.get("lineage", {}).get("target_scaler_bundle_id", "YSCALER__YS1")
    y_option = locked_config.get("data", {}).get("target_scaling_option", "YS1")
    revin_enabled = bool(
        locked_config.get("training", {}).get("revin_enabled", False)
    )

    if materialize_final_fit:
        raise NotImplementedError(
            "materialize_final_fit=True requires fit_final_dev_y_scaler() "
            "to be called explicitly with FINAL_DEV population; do not invent "
            "checksums inside the contract."
        )

    return ScalingContract(
        version="FINAL_SCALING-v1",
        fit_region="FINAL_DEV_REGION-v1",
        X_scaler_semantics=scaling_v,
        Y_scaler_semantics=y_option,
        x_scaler_bundle_id=x_bundle_id,
        y_scaler_bundle_id=y_bundle_id,
        x_scaler_bundle_checksum="REQUIRED_AT_PHASE46",
        y_scaler_bundle_checksum="REQUIRED_AT_PHASE46",
        fit_once=True,
        reuse_all_seeds=True,
        time_features_passthrough=True,
        binary_passthrough=True,
        revin_enabled=revin_enabled,
        revin_bridge_rule=(
            "fold_independent_global_scaler_bridge"
            if revin_enabled
            else "not_applicable_revin_disabled"
        ),
        Test_rows_used=False,
        status="PASS",
    )


def scaling_contract_to_dict(contract: ScalingContract) -> dict[str, Any]:
    return {
        "version": contract.version,
        "fit_region": contract.fit_region,
        "X_scaler_semantics": contract.X_scaler_semantics,
        "Y_scaler_semantics": contract.Y_scaler_semantics,
        "x_scaler_bundle_id": contract.x_scaler_bundle_id,
        "y_scaler_bundle_id": contract.y_scaler_bundle_id,
        "x_scaler_bundle_checksum": contract.x_scaler_bundle_checksum,
        "y_scaler_bundle_checksum": contract.y_scaler_bundle_checksum,
        "fit_once": contract.fit_once,
        "reuse_all_seeds": contract.reuse_all_seeds,
        "time_features_passthrough": contract.time_features_passthrough,
        "binary_passthrough": contract.binary_passthrough,
        "RevIN_fold_independent_global_scaler_bridge_if_active": (
            contract.revin_enabled
        ),
        "revin_bridge_rule": contract.revin_bridge_rule,
        "Test_rows_used": contract.Test_rows_used,
        "expected_checksum_fields": [
            "x_scaler_bundle_checksum",
            "y_scaler_bundle_checksum",
        ],
        "checksum_resolution_phase": "PHASE46",
        "status": contract.status,
    }


def fit_final_dev_y_scaler(*args: Any, **kwargs: Any) -> None:
    """Optional helper. Currently disabled — Phase45 must NOT invent checksums.

    See plan §114. Callers who want to materialize final-fit scalers should
    load the FINAL_DEV population explicitly and fit using existing helpers
    from ``course_work.scaling``. Phase45 orchestrator does NOT invoke this.
    """
    raise NotImplementedError(
        "Phase45 does not materialize final-fit scaler checksums. "
        "If you need them, fit them in Phase46 preflight and re-emit "
        "final_scaling_contract.json with the real SHA256."
    )


__all__ = [
    "ScalingContract",
    "build_scaling_contract",
    "scaling_contract_to_dict",
    "fit_final_dev_y_scaler",
]
