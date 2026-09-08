"""Build the FINAL_DEV_REGION-v1 population from canonical splits.

FINAL_DEV_REGION-v1 = TRAIN + VALIDATION (excluding TEST).

We derive the target timestamps from ``artifacts/splits/split_membership.csv``.
We compute a deterministic target-ID fingerprint and assert:

    last FINAL_DEV target timestamp < first TEST target timestamp
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FinalDevPopulation:
    region_id: str
    target_count: int
    first_target_timestamp: str
    last_target_timestamp: str
    first_test_timestamp: str
    target_ids: list[str]
    target_ids_fingerprint: str
    lookback_steps: int
    horizon_steps: int
    boundary_protocol: str
    split_version: str
    test_target_values_accessed: bool


def build_final_dev_population(
    project_root: Path,
    lookback_steps: int,
    horizon_steps: int = 1,
) -> FinalDevPopulation:
    """Build the FINAL_DEV population from canonical splits."""
    membership_path = project_root / "artifacts" / "splits" / "split_membership.csv"
    boundaries_path = project_root / "artifacts" / "splits" / "split_boundaries.csv"

    if not membership_path.exists():
        raise FileNotFoundError(f"split_membership.csv not found at {membership_path}")
    if not boundaries_path.exists():
        raise FileNotFoundError(f"split_boundaries.csv not found at {boundaries_path}")

    first_test_ts: str | None = None
    train_last_ts: str | None = None
    boundary_split_version = "SPLIT-v1"
    with boundaries_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("split_id", "")
            if sid == "TEST":
                first_test_ts = row.get("start_timestamp")
            elif sid == "VALIDATION":
                train_last_ts = row.get("end_timestamp")

    if first_test_ts is None:
        raise RuntimeError("TEST split boundary missing in split_boundaries.csv")

    target_ids: list[str] = []
    first_ts: str | None = None
    last_ts: str | None = None
    with membership_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("split_id", "")
            if sid in ("TRAIN", "VALIDATION"):
                ts = row.get("timestamp", "")
                row_idx = row.get("raw_row_index", "")
                target_ids.append(f"{sid}:{row_idx}:{ts}")
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

    if not target_ids:
        raise RuntimeError("FINAL_DEV population is empty")

    if not (last_ts < first_test_ts):
        raise RuntimeError(
            f"FINAL_DEV last timestamp {last_ts!r} not < first TEST timestamp {first_test_ts!r}"
        )

    canon = "\n".join(sorted(target_ids)).encode("utf-8")
    fingerprint = hashlib.sha256(canon).hexdigest()

    return FinalDevPopulation(
        region_id="FINAL_DEV_REGION-v1",
        target_count=len(target_ids),
        first_target_timestamp=first_ts,
        last_target_timestamp=last_ts,
        first_test_timestamp=first_test_ts,
        target_ids=target_ids,
        target_ids_fingerprint=fingerprint,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        split_version=boundary_split_version,
        test_target_values_accessed=False,
    )


def build_final_dev_contract(pop: FinalDevPopulation, lineage: dict[str, Any]) -> dict[str, Any]:
    """Build the ``final_data_region_contract.json`` payload."""
    return {
        "region_id": pop.region_id,
        "included_splits": ["TRAIN", "VALIDATION"],
        "excluded_splits": ["TEST"],
        "split_version": pop.split_version,
        "first_allowed_timestamp": pop.first_target_timestamp,
        "last_allowed_training_target_timestamp": pop.last_target_timestamp,
        "first_test_target_timestamp_metadata": pop.first_test_timestamp,
        "last_target_timestamp_before_test": pop.last_target_timestamp,
        "hard_assertion": "last_target_timestamp < first_test_target_timestamp",
        "target_population_rule": "WB0_CONTEXT_CARRY_OVER",
        "WB0": True,
        "WB1": False,
        "continuity": "strict",
        "lookback": pop.lookback_steps,
        "horizon": pop.horizon_steps,
        "boundary_protocol": pop.boundary_protocol,
        "target_count": pop.target_count,
        "target_ids_fingerprint": pop.target_ids_fingerprint,
        "test_target_values_accessed": pop.test_target_values_accessed,
        "WINDOWPOP_v1_policy": "preserved",
        "no_candidate_specific_expansion": True,
        "no_test_expansion": True,
        "lineage": lineage,
        "status": "PASS",
    }


__all__ = ["FinalDevPopulation", "build_final_dev_population", "build_final_dev_contract"]
