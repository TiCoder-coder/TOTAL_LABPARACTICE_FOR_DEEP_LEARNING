"""Causal TRAIN+VALIDATION-only rolling-target adapter for V2 E11."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from course_work.data.feature_sets import get_feature_list
from course_work.model_improvement_v2.pretest_adapter import (
    COMMON_POPULATION_ANCHOR,
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

CONTROL_VARIANT = "FS2_TF1"
CHALLENGER_VARIANT = "FS2_TF1_ROLL7"
ROLLING_FEATURES = (
    "appliances_roll_mean_3",
    "appliances_roll_mean_6",
    "appliances_roll_mean_12",
    "appliances_roll_std_6",
    "appliances_roll_std_12",
    "appliances_roll_max_12",
    "appliances_roll_min_12",
)
ROLLING_DEFINITIONS = {
    "appliances_roll_mean_3": "mean(y[s-2:s+1])",
    "appliances_roll_mean_6": "mean(y[s-5:s+1])",
    "appliances_roll_mean_12": "mean(y[s-11:s+1])",
    "appliances_roll_std_6": "std(y[s-5:s+1], ddof=0)",
    "appliances_roll_std_12": "std(y[s-11:s+1], ddof=0)",
    "appliances_roll_max_12": "max(y[s-11:s+1])",
    "appliances_roll_min_12": "min(y[s-11:s+1])",
}


@dataclass(frozen=True)
class E11PopulationAudit:
    control: PretestLoadAudit
    challenger: PretestLoadAudit
    rolling_definitions: dict[str, str]
    trailing_includes_current_observation: bool
    centered_rolling: bool
    ddof: int
    partial_windows: bool
    imputation: str
    common_ordered_target_ids: bool
    common_target_count: int
    common_population_fingerprint: str
    no_future_target_access: bool

    def to_dict(self) -> dict:
        return asdict(self)


def challenger_feature_order() -> tuple[str, ...]:
    return (*get_feature_list(CONTROL_VARIANT), *ROLLING_FEATURES)


def derive_past_only_rolling_features(targets: Sequence[float]) -> np.ndarray:
    """Compute seven trailing features using y[s] and earlier only."""
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    result = np.full((len(y), len(ROLLING_FEATURES)), np.nan, dtype=np.float64)

    def windows(width: int) -> np.ndarray:
        if len(y) < width:
            return np.empty((0, width), dtype=np.float64)
        return np.lib.stride_tricks.sliding_window_view(y, width)

    w3, w6, w12 = windows(3), windows(6), windows(12)
    if len(w3):
        result[2:, 0] = w3.mean(axis=1)
    if len(w6):
        result[5:, 1] = w6.mean(axis=1)
        result[5:, 3] = w6.std(axis=1, ddof=0)
    if len(w12):
        result[11:, 2] = w12.mean(axis=1)
        result[11:, 4] = w12.std(axis=1, ddof=0)
        result[11:, 5] = w12.max(axis=1)
        result[11:, 6] = w12.min(axis=1)
    return result


def build_e11_pretest_datasets(
    project_root: Path,
) -> tuple[dict[str, V2PretestWindowDataset], CanonicalFoldEvidence, E11PopulationAudit]:
    from course_work.rolling_origin.populations import compute_population_fingerprint

    root = Path(project_root).resolve()
    base_order = tuple(get_feature_list(CONTROL_VARIANT))
    extended_order = challenger_feature_order()
    _verify_phase9_scaler_provenance(root, CONTROL_VARIANT)
    evidence = load_phase44_fold_evidence(root)
    frame, source_kind, split = _read_pretest_prefix(root, base_order)

    target_ids = evidence.train_ids + evidence.validation_ids
    eligible = tuple(
        _target_id(position)
        for position in range(COMMON_POPULATION_ANCHOR, len(frame))
        if position - 1 - 72 + 1 >= 11
    )
    if target_ids != eligible:
        raise RuntimeError("E11 population differs from the Phase44 common intersection")
    if any(_target_position(value) >= len(frame) for value in target_ids):
        raise PermissionError("E11 hard Test firewall rejected target population")

    records = _build_window_records(
        frame, target_ids, int(split["train_rows"]), split["test_start_timestamp"]
    )
    if records["target_id"].tolist() != list(target_ids):
        raise RuntimeError("E11 ordered common target population changed")
    if int(records["timeline_input_start"].min()) < 11:
        raise RuntimeError("E11 common population lacks eleven causal history rows")

    targets = frame["Appliances"].to_numpy(dtype=np.float64, copy=True)
    rolling = derive_past_only_rolling_features(targets)
    base = frame.loc[:, base_order].to_numpy(dtype=np.float32, copy=True)
    extended = np.concatenate((base, rolling.astype(np.float32)), axis=1)
    used_start = int(records["timeline_input_start"].min())
    if not np.isfinite(extended[used_start:]).all():
        raise RuntimeError("E11 used feature population contains partial/non-finite windows")

    datasets = {
        CONTROL_VARIANT: V2PretestWindowDataset(
            base, targets, records, base_order,
            experiment_id="E11", feature_variant_id=CONTROL_VARIANT,
        ),
        CHALLENGER_VARIANT: V2PretestWindowDataset(
            extended, targets, records, extended_order,
            experiment_id="E11", feature_variant_id=CHALLENGER_VARIANT,
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
    audit = E11PopulationAudit(
        control=PretestLoadAudit(**common),
        challenger=PretestLoadAudit(**common),
        rolling_definitions=dict(ROLLING_DEFINITIONS),
        trailing_includes_current_observation=True,
        centered_rolling=False,
        ddof=0,
        partial_windows=False,
        imputation="NONE",
        common_ordered_target_ids=True,
        common_target_count=len(target_ids),
        common_population_fingerprint=compute_population_fingerprint(target_ids),
        no_future_target_access=True,
    )
    return datasets, evidence, audit
