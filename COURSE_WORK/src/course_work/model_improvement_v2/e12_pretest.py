"""Causal TRAIN+VALIDATION-only daily-lag adapter for V2 E12."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from course_work.data.feature_sets import get_feature_list
from course_work.model_improvement_v2.pretest_adapter import (
    BOUNDARY_PROTOCOL,
    HORIZON,
    LOOKBACK,
    CanonicalFoldEvidence,
    PretestLoadAudit,
    V2PretestWindowDataset,
    _build_window_records,
    _read_pretest_prefix,
    _target_id,
    _target_position,
    _verify_phase9_scaler_provenance,
    load_phase44_fold_evidence,
)
from course_work.rolling_origin.folds import build_rolling_folds
from course_work.rolling_origin.populations import compute_population_fingerprint

CONTROL_VARIANT = "FS2_TF1"
CHALLENGER_VARIANT = "FS2_TF1_LAG144"
LAG_FEATURE = "Appliances_lag_144"
LAG_STEPS = 144
LAG_DEFINITION = "y[s-144]"
FIRST_ELIGIBLE_TARGET_POSITION = LAG_STEPS + LOOKBACK


@dataclass(frozen=True)
class E12PopulationAudit:
    control: PretestLoadAudit
    challenger: PretestLoadAudit
    lag_feature: str
    lag_steps: int
    lag_definition: str
    lag_uses_current_or_future_target: bool
    imputation: str
    common_ordered_target_ids: bool
    original_phase44_target_count: int
    common_target_count: int
    lost_target_count: int
    lost_target_ids: tuple[str, ...]
    lost_target_fingerprint: str
    first_eligible_target_id: str
    common_population_fingerprint: str
    no_future_target_access: bool
    off_by_one_alignment_verified: bool

    def to_dict(self) -> dict:
        return asdict(self)


def challenger_feature_order() -> tuple[str, ...]:
    return (*get_feature_list(CONTROL_VARIANT), LAG_FEATURE)


def derive_past_only_lag144_feature(targets: Sequence[float]) -> np.ndarray:
    """Return a column where row s contains exactly y[s-144]."""

    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    lag = np.full((len(y), 1), np.nan, dtype=np.float64)
    if len(y) > LAG_STEPS:
        lag[LAG_STEPS:, 0] = y[:-LAG_STEPS]
    return lag


def _filtered_fold_evidence(
    phase44: CanonicalFoldEvidence,
    eligible_ids: tuple[str, ...],
) -> CanonicalFoldEvidence:
    eligible = set(eligible_ids)
    train_ids = tuple(value for value in phase44.train_ids if value in eligible)
    validation_ids = tuple(value for value in phase44.validation_ids if value in eligible)
    folds = tuple(build_rolling_folds(train_ids, validation_ids, k=3))
    population_fields = (
        "inner_train_ids",
        "inner_val_ids",
        "outer_train_ids",
        "outer_eval_ids",
    )
    status: dict[str, str] = {}
    for original, rebuilt in zip(phase44.folds, folds):
        for field in population_fields:
            expected = tuple(value for value in getattr(original, field) if value in eligible)
            if getattr(rebuilt, field) != expected:
                raise RuntimeError(
                    f"E12 {rebuilt.fold_id} {field} is not the ordered Phase44 intersection"
                )
        status[str(rebuilt.fold_id)] = "PASS"
    return CanonicalFoldEvidence(train_ids, validation_ids, folds, status)


def build_e12_pretest_datasets(
    project_root: Path,
) -> tuple[dict[str, V2PretestWindowDataset], CanonicalFoldEvidence, E12PopulationAudit]:
    """Build the E12 population-matched control and lag-144 projection."""

    root = Path(project_root).resolve()
    base_order = tuple(get_feature_list(CONTROL_VARIANT))
    extended_order = challenger_feature_order()
    _verify_phase9_scaler_provenance(root, CONTROL_VARIANT)
    phase44 = load_phase44_fold_evidence(root)
    frame, source_kind, split = _read_pretest_prefix(root, base_order)

    phase44_ids = phase44.train_ids + phase44.validation_ids
    eligible_ids = tuple(
        value
        for value in phase44_ids
        if _target_position(value) - HORIZON - LOOKBACK + 1 >= LAG_STEPS
    )
    expected_ids = tuple(
        _target_id(position)
        for position in range(FIRST_ELIGIBLE_TARGET_POSITION, len(frame))
    )
    if eligible_ids != expected_ids:
        raise RuntimeError("E12 lag-144 eligibility differs from the ordered Phase44 intersection")
    lost_ids = tuple(value for value in phase44_ids if value not in set(eligible_ids))
    evidence = _filtered_fold_evidence(phase44, eligible_ids)
    if evidence.train_ids + evidence.validation_ids != eligible_ids:
        raise RuntimeError("E12 filtered train/validation population changed order")
    if any(_target_position(value) >= len(frame) for value in eligible_ids):
        raise PermissionError("E12 hard Test firewall rejected target population")

    records = _build_window_records(
        frame,
        eligible_ids,
        int(split["train_rows"]),
        split["test_start_timestamp"],
    )
    if records["target_id"].tolist() != list(eligible_ids):
        raise RuntimeError("E12 ordered common target population changed")
    used_start = int(records["timeline_input_start"].min())
    if used_start != LAG_STEPS:
        raise RuntimeError("E12 lag-144 window alignment is off by one")

    targets = frame["Appliances"].to_numpy(dtype=np.float64, copy=True)
    lag = derive_past_only_lag144_feature(targets)
    if not np.array_equal(lag[LAG_STEPS:, 0], targets[:-LAG_STEPS]):
        raise RuntimeError("E12 lag-144 causal identity failed")
    base = frame.loc[:, base_order].to_numpy(dtype=np.float32, copy=True)
    extended = np.concatenate((base, lag.astype(np.float32)), axis=1)
    if not np.isfinite(extended[used_start:]).all():
        raise RuntimeError("E12 used feature population contains undefined lag values")

    datasets = {
        CONTROL_VARIANT: V2PretestWindowDataset(
            base,
            targets,
            records,
            base_order,
            experiment_id="E12",
            feature_variant_id=CONTROL_VARIANT,
        ),
        CHALLENGER_VARIANT: V2PretestWindowDataset(
            extended,
            targets,
            records,
            extended_order,
            experiment_id="E12",
            feature_variant_id=CHALLENGER_VARIANT,
        ),
    }
    common = dict(
        source_kind=source_kind,
        rows_loaded=len(frame),
        train_rows_loaded=int(split["train_rows"]),
        validation_rows_loaded=int(split["validation_rows"]),
        test_rows_read=0,
        test_target_ids_seen=0,
        first_timestamp=str(frame["timestamp"].iloc[0]),
        last_timestamp=str(frame["timestamp"].iloc[-1]),
        test_start_timestamp=str(split["test_start_timestamp"]),
        feature_order_status="PASS",
        population_status="PASS",
        fold_fingerprint_status=evidence.fingerprint_status,
        phase9_scaler_provenance_status="PASS_NOT_USED_FOR_TRAINING",
    )
    audit = E12PopulationAudit(
        control=PretestLoadAudit(**common),
        challenger=PretestLoadAudit(**common),
        lag_feature=LAG_FEATURE,
        lag_steps=LAG_STEPS,
        lag_definition=LAG_DEFINITION,
        lag_uses_current_or_future_target=False,
        imputation="NONE",
        common_ordered_target_ids=True,
        original_phase44_target_count=len(phase44_ids),
        common_target_count=len(eligible_ids),
        lost_target_count=len(lost_ids),
        lost_target_ids=lost_ids,
        lost_target_fingerprint=compute_population_fingerprint(lost_ids),
        first_eligible_target_id=eligible_ids[0],
        common_population_fingerprint=compute_population_fingerprint(eligible_ids),
        no_future_target_access=True,
        off_by_one_alignment_verified=True,
    )
    return datasets, evidence, audit
