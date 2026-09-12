"""Causal TRAIN+VALIDATION-only feature adapter for V2 E10."""
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
CHALLENGER_VARIANT = "FS2_TF1_DELTA3"
DELTA_FEATURES = (
    "appliances_delta_1",
    "appliances_abs_delta_1",
    "appliances_delta_2",
)


@dataclass(frozen=True)
class E10PopulationAudit:
    control: PretestLoadAudit
    challenger: PretestLoadAudit
    delta_definitions: dict[str, str]
    common_ordered_target_ids: bool
    common_target_count: int
    no_future_target_access: bool

    def to_dict(self) -> dict:
        return asdict(self)


def challenger_feature_order() -> tuple[str, ...]:
    return (*get_feature_list(CONTROL_VARIANT), *DELTA_FEATURES)


def derive_causal_target_deltas(targets: Sequence[float]) -> np.ndarray:
    """Return delta1, abs(delta1), delta2; unavailable history stays NaN."""
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    result = np.full((len(y), 3), np.nan, dtype=np.float64)
    if len(y) > 1:
        result[1:, 0] = y[1:] - y[:-1]
        result[1:, 1] = np.abs(result[1:, 0])
    if len(y) > 2:
        result[2:, 2] = y[2:] - y[:-2]
    return result


def build_e10_pretest_datasets(
    project_root: Path,
) -> tuple[dict[str, V2PretestWindowDataset], CanonicalFoldEvidence, E10PopulationAudit]:
    root = Path(project_root).resolve()
    base_order = tuple(get_feature_list(CONTROL_VARIANT))
    extended_order = challenger_feature_order()
    _verify_phase9_scaler_provenance(root, CONTROL_VARIANT)
    evidence = load_phase44_fold_evidence(root)
    frame, source_kind, split = _read_pretest_prefix(root, base_order)
    target_ids = evidence.train_ids + evidence.validation_ids
    expected = tuple(_target_id(i) for i in range(COMMON_POPULATION_ANCHOR, len(frame)))
    if target_ids != expected:
        raise RuntimeError("E10 population differs from canonical Phase44 evidence")
    if any(_target_position(value) >= len(frame) for value in target_ids):
        raise PermissionError("E10 hard Test firewall rejected target population")

    records = _build_window_records(
        frame, target_ids, int(split["train_rows"]), split["test_start_timestamp"]
    )
    if records["target_id"].tolist() != list(target_ids):
        raise RuntimeError("E10 ordered common target population changed")
    if int(records["timeline_input_start"].min()) < 2:
        raise RuntimeError("E10 common population lacks two causal history rows")

    targets = frame["Appliances"].to_numpy(dtype=np.float64, copy=True)
    deltas = derive_causal_target_deltas(targets)
    base = frame.loc[:, base_order].to_numpy(dtype=np.float32, copy=True)
    extended = np.concatenate((base, deltas.astype(np.float32)), axis=1)
    used_start = int(records["timeline_input_start"].min())
    if not np.isfinite(extended[used_start:]).all():
        raise RuntimeError("E10 used feature population contains non-finite values")

    datasets = {
        CONTROL_VARIANT: V2PretestWindowDataset(
            base, targets, records, base_order,
            experiment_id="E10", feature_variant_id=CONTROL_VARIANT,
        ),
        CHALLENGER_VARIANT: V2PretestWindowDataset(
            extended, targets, records, extended_order,
            experiment_id="E10", feature_variant_id=CHALLENGER_VARIANT,
        ),
    }
    common_audit = dict(
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
    audit = E10PopulationAudit(
        control=PretestLoadAudit(**common_audit),
        challenger=PretestLoadAudit(**common_audit),
        delta_definitions={
            "appliances_delta_1": "y[s] - y[s-1]",
            "appliances_abs_delta_1": "abs(y[s] - y[s-1])",
            "appliances_delta_2": "y[s] - y[s-2]",
        },
        common_ordered_target_ids=True,
        common_target_count=len(target_ids),
        no_future_target_access=True,
    )
    return datasets, evidence, audit
