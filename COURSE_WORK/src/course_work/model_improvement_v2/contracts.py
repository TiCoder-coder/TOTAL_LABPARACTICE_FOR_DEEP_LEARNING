"""Immutable foundation contracts for MODEL_IMPROVEMENT-v2.

This module is deliberately free of model, training, inference and artifact-write
side effects.  It owns only V2 governance constants and validation helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = "MODEL_IMPROVEMENT_V2_CONTRACT-v1"
TRACK_ID = "MODEL_IMPROVEMENT-v2"

V2_SOURCE_ROOT = PurePosixPath("COURSE_WORK/src/course_work/model_improvement_v2")
V2_ARTIFACT_ROOT = PurePosixPath("COURSE_WORK/artifacts/model_improvement_v2")
V1_ARTIFACT_ROOTS = (
    PurePosixPath("COURSE_WORK/artifacts/rolling_origin"),
    PurePosixPath("COURSE_WORK/artifacts/final_model_lock"),
    PurePosixPath("COURSE_WORK/artifacts/three_seed_final_runs"),
    PurePosixPath("COURSE_WORK/artifacts/final_test"),
)

TRAINING_AUTHORIZED = False
TEST_ACCESS_AUTHORIZED = False
V1_MUTATION_FORBIDDEN = True
TEST_SELECTION_FORBIDDEN = True

ALLOWED_WAVE1_EXPERIMENT_IDS = frozenset(f"E{index:02d}" for index in range(10))


@dataclass(frozen=True)
class DevelopmentBaseline:
    """Human-locked V1-equivalent development baseline for all V2 promotion."""

    candidate_id: str
    config_fingerprint: str
    feature_set: str
    lookback: int
    boundary_protocol: str
    rolling_origin_protocol: str
    folds: tuple[str, ...]
    fold_count: int
    fold_local_scaling: bool
    baseline_pooled_rmse_wh: float
    optimizer: str
    learning_rate: float
    loss: str
    phase44_status: str
    phase44_test_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CANONICAL_DEVELOPMENT_BASELINE = DevelopmentBaseline(
    candidate_id="TR_C2_ALT_LOOKBACK",
    config_fingerprint=(
        "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"
    ),
    feature_set="FS2_TF1",
    lookback=72,
    boundary_protocol="WB0_CONTEXT_CARRY_OVER",
    rolling_origin_protocol="RO3_EXPANDING_PRETEST-v1",
    folds=("RO1", "RO2", "RO3"),
    fold_count=3,
    fold_local_scaling=True,
    baseline_pooled_rmse_wh=59.85979463604273,
    optimizer="AdamW",
    learning_rate=3e-4,
    loss="MSE",
    phase44_status="PASS",
    phase44_test_status="NOT_ACCESSED",
)

# Each tuple is one scientific factor even when represented by several nested
# config fields. E00/E01 intentionally permit no primary config change.
PRIMARY_CHANGE_PREFIXES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "E00": (),
        "E01": (),
        "E02": (
            "data.feature_set",
            "data.feature_variant_id",
            "data.feature_count",
            "model.input_size",
            "lineage.feature_fingerprint",
            "lineage.scaler_bundle_id",
            "lineage.scaler_bundle_checksum",
        ),
        "E03": (
            "model.prediction_formulation",
        ),
        "E04": ("model.prediction_head",),
        "E05": ("model.residual_gate",),
        "E06": ("training.learning_rate",),
        "E07": ("training.scheduler_name", "training.scheduler_config"),
        "E08": (
            "training.optimizer",
            "training.optimizer_name",
            "training.optimizer_config",
            "training.learning_rate",
        ),
        "E09": (
            "training.optimizer",
            "training.optimizer_name",
            "training.optimizer_config",
            "training.learning_rate",
            "training.weight_decay",
        ),
    }
)


def assert_allowed_experiment_id(experiment_id: str) -> None:
    """Reject IDs outside the human-approved Wave 1 range E00--E09."""

    if experiment_id not in ALLOWED_WAVE1_EXPERIMENT_IDS:
        raise ValueError(f"Experiment is not approved for Wave 1: {experiment_id!r}")


def assert_no_test_access(*, split_id: str | None = None, test_access: bool = False) -> None:
    """Enforce the V2 Test firewall at a config or split boundary."""

    if test_access:
        raise PermissionError("MODEL_IMPROVEMENT-v2 Test access is not authorized")
    if split_id is not None and "TEST" in split_id.upper():
        raise PermissionError(f"Test split is forbidden for V2 selection: {split_id!r}")


def assert_v2_artifact_path(path: str | PurePosixPath) -> None:
    """Reject writes outside the isolated V2 artifact namespace."""

    candidate = PurePosixPath(path)
    if candidate == V2_ARTIFACT_ROOT:
        return
    if V2_ARTIFACT_ROOT not in candidate.parents:
        raise PermissionError(f"Artifact write is outside the V2 namespace: {candidate}")
    if any(root == candidate or root in candidate.parents for root in V1_ARTIFACT_ROOTS):
        raise PermissionError(f"V1 artifact mutation is forbidden: {candidate}")


def _flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_config(value, field))
        else:
            flattened[field] = value
    return flattened


def assert_one_primary_change(
    experiment_id: str,
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate that a candidate changes only its approved scientific factor.

    The function returns changed leaf paths for audit recording. E00/E01 require
    exact config equality; E02--E09 require at least one changed leaf under one
    approved factor group and reject every out-of-factor difference.
    """

    assert_allowed_experiment_id(experiment_id)
    baseline = _flatten_config(baseline_config)
    candidate = _flatten_config(candidate_config)
    fields = sorted(set(baseline) | set(candidate))
    changed = tuple(field for field in fields if baseline.get(field) != candidate.get(field))
    allowed = PRIMARY_CHANGE_PREFIXES[experiment_id]

    if not allowed:
        if changed:
            raise ValueError(f"{experiment_id} permits no primary config change: {changed}")
        return changed
    if not changed:
        raise ValueError(f"{experiment_id} requires its approved primary change")

    forbidden = tuple(
        field
        for field in changed
        if not any(field == prefix or field.startswith(f"{prefix}.") for prefix in allowed)
    )
    if forbidden:
        raise ValueError(f"{experiment_id} changes fields outside its primary factor: {forbidden}")
    return changed
