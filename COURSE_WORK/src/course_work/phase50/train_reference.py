"""Phase 50 — REGIME_REFERENCE_TRAIN-v1 construction.

Definition:
    REGIME_REFERENCE_TRAIN-v1
      =  original Train target_ids
       ∩  WINDOWPOP-v1 target_ids
       ∩  continuity-valid target_ids
       ∩  temporally valid target_ids
       ∩  raw Appliances Wh (not standardized, not RevIN)

Excludes:
    Validation, Test, Train+Validation final-refit, YS1, RevIN coords.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

from . import contract
from . import sources


def _deterministic_sha256_of_strings(values: list[str]) -> str:
    """Deterministic sha256 over sorted UTF-8 strings, LF-separated."""
    s = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def build_regime_reference_train(
    project_root: Path | None = None,
) -> dict:
    """Build REGIME_REFERENCE_TRAIN-v1 from canonical sources.

    Returns dict with:
        target_ids: list[str]
        target_ids_sha256: str
        n: int
        raw_appliances: dict[target_id, float]
        previous_target_id_by_target_id: dict[target_id, str | None]
        continuity_segment_id_by_target_id: dict[target_id, str]
        raw_row_index_by_target_id: dict[target_id, int]
        target_timestamp_by_target_id: dict[target_id, str]
    """
    root = project_root if project_root is not None else sources.project_root()

    # 1. WINDOWPOP common target population (filtered to TRAIN)
    ctp, _ = sources.load_windowpop_common_target(root)
    ctp_by_id = {r["target_sample_id"]: r for r in ctp}
    train_rows = [r for r in ctp if r["target_split_id"] == "TRAIN"]

    # 2. Raw Appliances Wh
    raw, _ = sources.load_raw_appliances(root)
    # raw[0] has date=2016-01-11 17:00:00; target_sample_id 'TGT_00000144' (timeline=144) maps to raw row index 143
    # Mapping: raw_row_index = timeline_target − 1
    raw_by_idx: dict[int, dict[str, str]] = {i: r for i, r in enumerate(raw)}

    # 3. Split membership (chronological continuity_segment_id per raw_row_index)
    split = sources.load_split_membership(root)
    split_by_raw_idx: dict[int, dict[str, str]] = {
        int(r["raw_row_index"]): r for r in split
    }

    target_ids: list[str] = []
    raw_appliances: dict[str, float] = {}
    continuity_by_id: dict[str, str] = {}
    ts_by_id: dict[str, str] = {}
    raw_idx_by_id: dict[str, int] = {}

    for r in train_rows:
        tid = r["target_sample_id"]
        timeline = int(r["timeline_target"])  # 1-based
        raw_idx = timeline - 1  # 0-based row in raw CSV
        cont = r["continuity_segment_id"]
        ts = r["target_timestamp"]
        if raw_idx not in raw_by_idx:
            continue
        appl = raw_by_idx[raw_idx]["Appliances"]
        target_ids.append(tid)
        raw_appliances[tid] = float(appl)
        continuity_by_id[tid] = cont
        ts_by_id[tid] = ts
        raw_idx_by_id[tid] = raw_idx

    target_ids_sorted = sorted(target_ids)
    target_ids_sha256 = _deterministic_sha256_of_strings(target_ids_sorted)

    # 4. Predecessor map (gap-safe, exact 10-min, same continuity_segment)
    prev_by_id: dict[str, str | None] = {}
    sorted_train = sorted(
        target_ids, key=lambda t: (raw_idx_by_id[t], t)
    )
    for i, tid in enumerate(sorted_train):
        if i == 0:
            prev_by_id[tid] = None
            continue
        prev_tid = sorted_train[i - 1]
        # Continuity + 10-min check
        same_seg = continuity_by_id[prev_tid] == continuity_by_id[tid]
        ts_diff_min = (
            _timestamp_diff_minutes(ts_by_id[prev_tid], ts_by_id[tid])
            if same_seg
            else None
        )
        if same_seg and ts_diff_min == contract.CADENCE_MINUTES:
            prev_by_id[tid] = prev_tid
        else:
            prev_by_id[tid] = None  # unclassified (gap or split crossing)

    return {
        "name": "REGIME_REFERENCE_TRAIN-v1",
        "n": len(target_ids_sorted),
        "target_ids": target_ids_sorted,
        "target_ids_sha256": target_ids_sha256,
        "raw_appliances": raw_appliances,
        "previous_target_id_by_target_id": prev_by_id,
        "continuity_segment_id_by_target_id": continuity_by_id,
        "raw_row_index_by_target_id": raw_idx_by_id,
        "target_timestamp_by_target_id": ts_by_id,
        "provenance": {
            "source_population": "artifacts/windows/common_target_population.csv",
            "filter": "target_split_id == TRAIN (raw Appliances Wh)",
            "raw_data_source": "data/raw_data/energydata_complete.csv",
            "split_version": "SPLIT-v1",
            "windowpop_version": "WINDOWPOP-v1",
            "temporal_version": "TEMPORAL-v1",
            "excluded_populations": [
                "VALIDATION",
                "TEST",
                "Train+Validation final-refit population",
                "YS1 standardized y_model",
                "RevIN normalized coordinates",
            ],
        },
    }


def _timestamp_diff_minutes(ts_a: str, ts_b: str) -> int | None:
    """Return (ts_b − ts_a) in minutes if both parse as YYYY-MM-DD HH:MM:SS."""
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        a = datetime.strptime(ts_a, fmt)
        b = datetime.strptime(ts_b, fmt)
    except Exception:
        return None
    return int((b - a).total_seconds() // 60)


def compute_train_delta_y(
    ref: dict, target_id: str
) -> tuple[float, str]:
    """Return (delta_y_wh, predecessor_status)."""
    prev = ref["previous_target_id_by_target_id"].get(target_id)
    if prev is None:
        return (float("nan"), "CHANGE_UNCLASSIFIED")
    y_t = ref["raw_appliances"][target_id]
    y_prev = ref["raw_appliances"][prev]
    return (y_t - y_prev, "CLASSIFIED")
