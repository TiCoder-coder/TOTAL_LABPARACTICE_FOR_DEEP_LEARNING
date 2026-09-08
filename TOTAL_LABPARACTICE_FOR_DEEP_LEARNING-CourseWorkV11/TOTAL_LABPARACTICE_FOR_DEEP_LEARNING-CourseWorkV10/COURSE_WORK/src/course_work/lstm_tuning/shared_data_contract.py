"""Canonical Phase 42 -> Phase 43 shared data contract resolution.

Resolves the hard fields required by the Phase 43 plan from the Phase 42/43
handoff and from the upstream Phase 11 DataLoader manifest. Does NOT
hardcode conflicting values.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from course_work.experiments.registry import load_upstream_context
from course_work.utils.artifacts import read_json


HANDOFF_RELATIVE = Path("artifacts/candidate_synthesis/phase43_lstm_tuning_handoff.json")
PHASE_42_SIGNOFF_RELATIVE = Path("artifacts/candidate_synthesis/phase_42_signoff.json")

REQUIRED_FIELDS = (
    "forecast_horizon",
    "feature_variant_id",
    "target_scaling_id",
    "lookback_steps",
    "boundary_protocol",
    "population_fingerprint",
    "train_target_ids_fingerprint",
    "validation_target_ids_fingerprint",
    "feature_fingerprint",
    "x_scaler_checksum",
    "target_scaler_checksum",
    "metric_version",
    "test_locked",
)


@dataclass(frozen=True)
class SharedDataContract:
    forecast_task: str
    forecast_horizon: int
    target: str
    feature_variant_id: str
    feature_names: list[str]
    feature_fingerprint: str
    feature_count: int
    target_scaling_id: str
    target_scaler_bundle_id: str
    target_scaler_checksum: str
    lookback_id: str
    lookback_steps: int
    boundary_protocol: str
    window_population_version: str
    population_fingerprint: str
    train_target_ids_fingerprint: str
    validation_target_ids_fingerprint: str
    train_sample_count: int
    validation_sample_count: int
    test_sample_count: int
    test_locked: bool
    x_scaler_bundle_id: str
    x_scaler_checksum: str
    batch_size: int
    split_version: str
    metric_version: str
    handoff_source: str
    handoff_phase42_shortlist_fingerprint: str
    transformer_primary_candidate_id: str
    transformer_primary_run_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SharedDataContractError(RuntimeError):
    pass


def _resolve_target_scaler(scalers: dict[str, Any], target_option: str) -> tuple[str, str]:
    target_bundles = scalers.get("target_bundles", {})
    if target_option not in target_bundles:
        raise SharedDataContractError(f"target_scaling '{target_option}' not in scaler registry")
    bundle = target_bundles[target_option]
    return bundle.get("bundle_id", "YSCALER__" + target_option), bundle.get("artifact_sha256") or "YS0_IDENTITY"


def _resolve_x_scaler(scalers: dict[str, Any], variant_id: str) -> tuple[str, str]:
    x_bundles = scalers.get("x_bundles", {})
    if variant_id not in x_bundles:
        raise SharedDataContractError(f"feature_variant '{variant_id}' not in scaler registry")
    bundle = x_bundles[variant_id]
    return bundle.get("bundle_id"), bundle.get("artifact_sha256")


def _resolve_feature_fingerprint(feature_sets: dict[str, Any], variant_id: str) -> tuple[str, list[str]]:
    fingerprints = feature_sets.get("variant_fingerprints", {})
    components = feature_sets.get("variant_components", {})
    feature_counts = feature_sets.get("variant_feature_counts", {})
    if variant_id not in fingerprints:
        raise SharedDataContractError(f"feature_variant '{variant_id}' not in feature_set registry")
    if variant_id not in feature_counts:
        raise SharedDataContractError(f"feature_count missing for '{variant_id}'")
    components_list = components.get(variant_id, [])
    feature_names: list[str] = []
    for component in components_list:
        if isinstance(component, dict):
            feature_names.extend(component.get("feature_names", []))
        elif isinstance(component, str):
            feature_names.append(component)
    return fingerprints[variant_id], feature_names


def resolve_phase43_handoff(project_root: Path) -> dict[str, Any]:
    handoff_path = project_root / HANDOFF_RELATIVE
    if not handoff_path.exists():
        raise SharedDataContractError(f"Phase 43 handoff missing: {handoff_path}")
    return read_json(handoff_path)


def validate_phase42_signoff(project_root: Path) -> dict[str, Any]:
    signoff_path = project_root / PHASE_42_SIGNOFF_RELATIVE
    if not signoff_path.exists():
        raise SharedDataContractError(f"Phase 42 signoff missing: {signoff_path}")
    signoff = read_json(signoff_path)
    status = signoff.get("overall_status") or signoff.get("status")
    if status not in {"PASS", "PASS_WITH_WARNING"}:
        raise SharedDataContractError(f"Phase 42 signoff status not PASS-like: {status!r}")
    if not signoff.get("ready_for_phase43", False) and signoff.get("ready_for_phase43") is not None:
        raise SharedDataContractError("Phase 42 signoff does not mark ready_for_phase43")
    return signoff


def resolve_shared_data_contract(
    project_root: Path,
    handoff: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> SharedDataContract:
    if handoff is None:
        handoff = resolve_phase43_handoff(project_root)
    if context is None:
        context = load_upstream_context(project_root)

    variant_id = handoff["selected_feature_variant"]
    target_option = handoff["selected_target_scaling"]
    lookback = int(handoff["selected_lookback"])
    boundary_protocol = handoff["boundary_protocol"]
    window_population_version = handoff["window_population_policy"]

    feature_sets = context["feature_sets"]
    scalers = context["scalers"]
    windows = context["window_fingerprints"]
    dataloaders = context["dataloaders"]
    metrics = context["metrics"]

    feature_fingerprint, feature_names = _resolve_feature_fingerprint(feature_sets, variant_id)
    x_bundle_id, x_checksum = _resolve_x_scaler(scalers, variant_id)
    y_bundle_id, y_checksum = _resolve_target_scaler(scalers, target_option)

    feature_count = feature_sets["variant_feature_counts"][variant_id]

    baseline_batch = dataloaders["baseline_batch_size"]
    baseline_loader_fingerprints = dataloaders["baseline_loader_fingerprints"]
    val_loader_fp = baseline_loader_fingerprints.get("VALIDATION")
    train_loader_fp = baseline_loader_fingerprints.get("TRAIN")

    lookback_window_fp = windows["window_index_fingerprints"].get(f"L{lookback}_H01_{boundary_protocol}")
    if lookback_window_fp is None:
        lookback_window_fp = windows["window_index_fingerprints"].get("L144_H01_WB0")

    return SharedDataContract(
        forecast_task=str(handoff.get("forecast_task", "UCI Appliances Energy Prediction, Sequence-to-One")),
        forecast_horizon=int(handoff.get("horizon", 1)),
        target="Appliances",
        feature_variant_id=variant_id,
        feature_names=feature_names,
        feature_fingerprint=feature_fingerprint,
        feature_count=int(feature_count),
        target_scaling_id=target_option,
        target_scaler_bundle_id=y_bundle_id,
        target_scaler_checksum=y_checksum,
        lookback_id=f"L{lookback}_H01_{boundary_protocol}",
        lookback_steps=lookback,
        boundary_protocol=boundary_protocol,
        window_population_version=window_population_version,
        population_fingerprint=str(context["lineage"]["population_fingerprint"]),
        train_target_ids_fingerprint=train_loader_fp or "",
        validation_target_ids_fingerprint=val_loader_fp or "",
        train_sample_count=int(dataloaders["train_sample_count"]),
        validation_sample_count=int(dataloaders["validation_sample_count"]),
        test_sample_count=int(dataloaders["test_sample_count"]),
        test_locked=bool(handoff.get("test_locked", False)),
        x_scaler_bundle_id=x_bundle_id,
        x_scaler_checksum=x_checksum,
        batch_size=int(baseline_batch),
        split_version=str(context["lineage"]["split_version"]),
        metric_version=str(context["lineage"]["metric_version"]),
        handoff_source=str(HANDOFF_RELATIVE),
        handoff_phase42_shortlist_fingerprint=str(handoff.get("phase42_shortlist_fingerprint", "")),
        transformer_primary_candidate_id=str(handoff.get("transformer_primary_candidate_id", "")),
        transformer_primary_run_id=str(handoff.get("transformer_primary_run_id", "")),
    )


def assert_test_locked(contract: SharedDataContract) -> None:
    if not contract.test_locked:
        raise SharedDataContractError("Test lock violated: contract.test_locked is False")
    if contract.test_sample_count <= 0:
        raise SharedDataContractError(
            f"Test sample_count invalid: {contract.test_sample_count} (Test firewall broken)"
        )


def assert_required_fields(contract: SharedDataContract) -> None:
    missing = [field for field in REQUIRED_FIELDS if getattr(contract, field, None) in (None, "", 0)]
    if missing:
        raise SharedDataContractError(f"Shared data contract missing required fields: {missing}")
