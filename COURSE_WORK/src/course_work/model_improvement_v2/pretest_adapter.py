"""Strict pre-Test L72/WB0 dataset adapter for V2 rolling experiments.

This module intentionally bypasses historical Phase 9/10 materializers. It
parses only the frozen TRAIN+VALIDATION prefix and binds its target population
to the active Phase 44 rolling-origin fold evidence. E01 remains locked to
FS2_TF1; E02 uses the same population with the canonical FS1_TF1 projection.
"""

from __future__ import annotations

import csv
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Sequence, TextIO

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from course_work.data.feature_sets import get_feature_list
from course_work.rolling_origin.folds import FoldDefinition, build_rolling_folds
from course_work.utils.artifacts import read_json, sha256_file


EXPERIMENT_ID = "E01"
CANDIDATE_ID = "TR_C2_ALT_LOOKBACK"
FEATURE_VARIANT_ID = "FS2_TF1"
LOOKBACK = 72
HORIZON = 1
BOUNDARY_PROTOCOL = "WB0_CONTEXT_CARRY_OVER"
FOLD_PROTOCOL = "RO3_EXPANDING_PRETEST-v1"
COMMON_POPULATION_ANCHOR = 144

FEATURE_SOURCE_RELATIVE = Path(
    "data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv"
)
RELOCATED_FEATURE_SOURCE_RELATIVE = Path(
    "link/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv"
)
ARCHIVE_MEMBER = (
    "COURSE_WORK/link/interim/uci_appliances_energy_prediction/"
    "energydata_feature_engineered_v1.csv"
)


@dataclass(frozen=True)
class PretestLoadAudit:
    source_kind: str
    rows_loaded: int
    train_rows_loaded: int
    validation_rows_loaded: int
    test_rows_read: int
    test_target_ids_seen: int
    first_timestamp: str
    last_timestamp: str
    test_start_timestamp: str
    feature_order_status: str
    population_status: str
    fold_fingerprint_status: dict[str, str]
    phase9_scaler_provenance_status: str


@dataclass(frozen=True)
class CanonicalFoldEvidence:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    folds: tuple[FoldDefinition, ...]
    fingerprint_status: dict[str, str]


def _target_id(position: int) -> str:
    return f"TGT_{position:08d}"


def _target_position(target_id: str) -> int:
    value = str(target_id)
    if not value.startswith("TGT_") or not value[4:].isdigit():
        raise ValueError(f"Invalid canonical target ID: {target_id!r}")
    return int(value[4:])


def load_phase44_fold_evidence(root: Path) -> CanonicalFoldEvidence:
    path = root / "artifacts/rolling_origin/rolling_origin_fold_manifest.json"
    document = read_json(path)
    if document.get("fold_protocol") != FOLD_PROTOCOL:
        raise RuntimeError("Phase44 fold protocol mismatch")
    if document.get("boundary_protocol") != BOUNDARY_PROTOCOL:
        raise RuntimeError("Phase44 boundary protocol mismatch")
    if document.get("test_region_used") is not False:
        raise RuntimeError("Phase44 evidence does not prove a pre-Test population")
    rows = document.get("folds", [])
    if [row.get("fold_id") for row in rows] != ["RO1", "RO2", "RO3"]:
        raise RuntimeError("Phase44 evidence must contain ordered RO1/RO2/RO3")

    train_ids = tuple(str(value) for value in rows[0]["outer_train_ids"])
    validation_ids = tuple(
        str(value) for row in rows for value in row["outer_eval_ids"]
    )
    rebuilt = tuple(build_rolling_folds(train_ids, validation_ids, k=3))
    status: dict[str, str] = {}
    population_fields = (
        "inner_train_ids",
        "inner_val_ids",
        "outer_train_ids",
        "outer_eval_ids",
    )
    fingerprint_fields = (
        "inner_train_fingerprint",
        "inner_val_fingerprint",
        "outer_train_fingerprint",
        "outer_eval_fingerprint",
        "fold_population_fingerprint",
    )
    for evidence_row, fold in zip(rows, rebuilt):
        fold_id = str(fold.fold_id)
        for field in population_fields:
            if list(getattr(fold, field)) != [str(value) for value in evidence_row[field]]:
                raise RuntimeError(f"{fold_id} ordered {field} differs from Phase44")
        for field in fingerprint_fields:
            if getattr(fold, field) != evidence_row[field]:
                raise RuntimeError(f"{fold_id} {field} differs from Phase44")
        status[fold_id] = "PASS"
    return CanonicalFoldEvidence(train_ids, validation_ids, rebuilt, status)


def _verify_phase9_scaler_provenance(root: Path, feature_variant_id: str) -> None:
    """Verify the frozen variant/YS1 files for provenance; never fit or use them."""

    signoff = read_json(root / "artifacts/scaling/phase_9_signoff.json")
    registry = read_json(root / "artifacts/scaling/scaler_registry.json")
    entries = (
        registry["x_bundles"][feature_variant_id],
        registry["target_bundles"]["YS1"],
    )
    frozen_checksums = signoff.get("output_checksums", {})
    for entry in entries:
        relative = entry["artifact_path"]
        expected = frozen_checksums.get(relative)
        if expected is None or expected != entry.get("artifact_sha256"):
            raise RuntimeError(f"Frozen Phase9 provenance mismatch: {relative}")
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen Phase9 scaler checksum mismatch: {relative}")


@contextmanager
def _open_feature_source(root: Path) -> Iterator[tuple[TextIO, str]]:
    for relative in (FEATURE_SOURCE_RELATIVE, RELOCATED_FEATURE_SOURCE_RELATIVE):
        path = root / relative
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as stream:
                yield stream, relative.as_posix()
            return

    archive_path = root.parent / "COURSE_WORK_FINAL.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(
            "No canonical feature CSV or read-only COURSE_WORK_FINAL.zip fallback"
        )
    with zipfile.ZipFile(archive_path) as archive:
        try:
            binary = archive.open(ARCHIVE_MEMBER, "r")
        except KeyError as exc:
            raise FileNotFoundError(f"Archive member missing: {ARCHIVE_MEMBER}") from exc
        with binary, TextIOWrapper(binary, encoding="utf-8", newline="") as stream:
            yield stream, f"{archive_path.name}!/{ARCHIVE_MEMBER}"


def _read_pretest_prefix(
    root: Path,
    feature_order: Sequence[str],
) -> tuple[pd.DataFrame, str, dict]:
    split_manifest = read_json(root / "artifacts/splits/split_manifest.json")
    train_count = int(split_manifest["train_rows"])
    validation_count = int(split_manifest["validation_rows"])
    pretest_count = train_count + validation_count
    selected = [
        "raw_row_index",
        "timestamp",
        "continuity_segment_id",
        *feature_order,
    ]
    # Appliances is both the raw target and an FS2 historical input. De-dup it.
    selected = list(dict.fromkeys(selected))

    records: list[dict[str, str]] = []
    with _open_feature_source(root) as (stream, source_kind):
        reader = csv.DictReader(stream)
        missing = set(selected) - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Feature source columns missing: {sorted(missing)}")
        for expected_position in range(pretest_count):
            try:
                row = next(reader)
            except StopIteration as exc:
                raise RuntimeError("Feature source ended before frozen Test boundary") from exc
            if int(row["raw_row_index"]) != expected_position:
                raise RuntimeError("Feature source raw-row order mismatch")
            records.append({column: row[column] for column in selected})

    frame = pd.DataFrame.from_records(records, columns=selected)
    frame["raw_row_index"] = frame["raw_row_index"].astype(np.int64)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["continuity_segment_id"] = frame["continuity_segment_id"].astype(str)
    for column in feature_order:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.loc[:, feature_order].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Pre-Test feature/target prefix contains non-finite values")

    test_start = pd.Timestamp(split_manifest["test_start_timestamp"])
    expected_last = pd.Timestamp(split_manifest["validation_end_timestamp"])
    if len(frame) != pretest_count or frame["timestamp"].iloc[-1] != expected_last:
        raise RuntimeError("Loaded prefix does not end at frozen Validation boundary")
    if bool(frame["timestamp"].ge(test_start).any()):
        raise RuntimeError("Hard Test firewall: row at/after Test boundary was loaded")
    if not frame["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Pre-Test source is not chronological")
    return frame, source_kind, split_manifest


def _build_window_records(
    frame: pd.DataFrame,
    target_ids: Sequence[str],
    train_count: int,
    test_start_timestamp: str,
) -> pd.DataFrame:
    rows = []
    test_start = pd.Timestamp(test_start_timestamp)
    for canonical_index, target_id in enumerate(target_ids):
        target = _target_position(target_id)
        input_start = target - HORIZON - LOOKBACK + 1
        input_end = target - HORIZON
        if input_start < 0 or target >= len(frame):
            raise RuntimeError(f"E01 window bounds invalid for {target_id}")
        input_frame = frame.iloc[input_start : input_end + 1]
        target_row = frame.iloc[target]
        if len(input_frame) != LOOKBACK:
            raise RuntimeError(f"E01 lookback mismatch for {target_id}")
        if target_row["timestamp"] >= test_start or bool(input_frame["timestamp"].ge(test_start).any()):
            raise RuntimeError(f"Hard Test firewall reached by {target_id}")
        if input_frame["continuity_segment_id"].nunique() != 1 or str(
            input_frame["continuity_segment_id"].iloc[0]
        ) != str(target_row["continuity_segment_id"]):
            raise RuntimeError(f"Discontinuous E01 window for {target_id}")
        target_split = "TRAIN" if target < train_count else "VALIDATION"
        input_start_split = "TRAIN" if input_start < train_count else "VALIDATION"
        input_end_split = "TRAIN" if input_end < train_count else "VALIDATION"
        rows.append(
            {
                "canonical_sample_idx": canonical_index,
                "window_id": f"WIN_L072_H01_{target_id}",
                "target_sample_id": target_id,
                "target_id": target_id,
                "lookback_steps": LOOKBACK,
                "horizon_steps": HORIZON,
                "timeline_input_start": input_start,
                "timeline_input_end": input_end,
                "timeline_target": target,
                "input_start_timestamp": str(input_frame["timestamp"].iloc[0]),
                "input_end_timestamp": str(input_frame["timestamp"].iloc[-1]),
                "target_timestamp": str(target_row["timestamp"]),
                "input_start_raw_row_index": int(frame["raw_row_index"].iloc[input_start]),
                "input_end_raw_row_index": int(frame["raw_row_index"].iloc[input_end]),
                "target_raw_row_index": int(target_row["raw_row_index"]),
                "continuity_segment_id": str(target_row["continuity_segment_id"]),
                "target_split_id": target_split,
                "input_start_split_id": input_start_split,
                "input_end_split_id": input_end_split,
                "crosses_split_boundary": input_start_split != target_split,
                "WB0_valid": True,
                "WB1_valid": input_start_split == target_split,
                "included_common_population": True,
            }
        )
    return pd.DataFrame(rows)


class V2PretestWindowDataset(Dataset):
    """Raw pre-Test features with fold-local scaling applied by V2 loaders."""

    def __init__(
        self,
        feature_matrix: np.ndarray,
        target_values: np.ndarray,
        window_records: pd.DataFrame,
        feature_order: Sequence[str],
        *,
        experiment_id: str = EXPERIMENT_ID,
        feature_variant_id: str = FEATURE_VARIANT_ID,
    ) -> None:
        self._feature_matrix = np.ascontiguousarray(feature_matrix, dtype=np.float32)
        self._target_values = np.ascontiguousarray(target_values, dtype=np.float64)
        self.window_records = window_records.reset_index(drop=True).copy(deep=True)
        self._records = self.window_records
        self.sample_indices = self.window_records["canonical_sample_idx"].to_numpy(
            dtype=np.int64, copy=True
        )
        self.feature_order = tuple(feature_order)
        self.experiment_id = experiment_id
        self.feature_variant_id = feature_variant_id
        self.config = SimpleNamespace(
            lookback=LOOKBACK,
            feature_count=len(feature_order),
            target_option="YS1",
            target_access_mode="VALIDATION",
            boundary_protocol=BOUNDARY_PROTOCOL,
        )
        if self._feature_matrix.shape[1] != len(self.feature_order):
            raise RuntimeError("V2 feature matrix width differs from canonical feature order")

    def __len__(self) -> int:
        return len(self.window_records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.window_records.iloc[int(index)]
        start = int(record["timeline_input_start"])
        end = int(record["timeline_input_end"])
        target = int(record["timeline_target"])
        x = torch.from_numpy(self._feature_matrix[start : end + 1].copy())
        y_raw = torch.tensor([self._target_values[target]], dtype=torch.float32)
        # Explicit past-only persistence anchor: input_end is t and target is
        # t+1 for horizon=1. Never derive this value from scaled X.
        context_index = int(record["timeline_input_end"])
        if context_index != target - HORIZON:
            raise RuntimeError("V2 persistence anchor is not the last observable target")
        y_context_raw = torch.tensor(
            [self._target_values[context_index]], dtype=torch.float32
        )
        if x.shape != (LOOKBACK, len(self.feature_order)):
            raise RuntimeError(f"{self.experiment_id} sequence shape mismatch")
        return {
            "x": x,
            "y_model": y_raw.clone(),
            "y_raw_wh": y_raw,
            "y_context_raw_wh": y_context_raw,
            "sample_idx": torch.tensor(
                int(record["canonical_sample_idx"]), dtype=torch.int64
            ),
        }


def build_v2_pretest_dataset(
    project_root: Path,
    feature_order: Sequence[str],
    *,
    experiment_id: str,
    feature_variant_id: str,
) -> tuple[V2PretestWindowDataset, CanonicalFoldEvidence, PretestLoadAudit]:
    """Build one approved V2 L72/WB0 feature projection without reading Test."""

    approved = {
        "E01": "FS2_TF1",
        "E02": "FS1_TF1",
        "E03": "FS2_TF1",
        "E04": "FS2_TF1",
        "E05": "FS2_TF1",
        "E06": "FS2_TF1",
        "E07": "FS2_TF1",
        "E08": "FS2_TF1",
        "E09": "FS2_TF1",
    }
    if approved.get(experiment_id) != feature_variant_id:
        raise RuntimeError(
            f"Unsupported V2 pre-Test projection: {experiment_id}/{feature_variant_id}"
        )
    root = Path(project_root).resolve()
    canonical_features = tuple(get_feature_list(feature_variant_id))
    if tuple(feature_order) != canonical_features:
        raise RuntimeError(
            f"{experiment_id} requires canonical ordered {feature_variant_id} features"
        )
    _verify_phase9_scaler_provenance(root, feature_variant_id)
    evidence = load_phase44_fold_evidence(root)
    frame, source_kind, split_manifest = _read_pretest_prefix(root, feature_order)
    target_ids = evidence.train_ids + evidence.validation_ids
    expected_ids = tuple(
        _target_id(position)
        for position in range(COMMON_POPULATION_ANCHOR, len(frame))
    )
    if target_ids != expected_ids:
        raise RuntimeError(
            f"{experiment_id} target population differs from canonical Phase44 evidence"
        )
    if any(_target_position(value) >= len(frame) for value in target_ids):
        raise RuntimeError("Hard Test firewall: canonical population contains Test IDs")

    records = _build_window_records(
        frame,
        target_ids,
        int(split_manifest["train_rows"]),
        split_manifest["test_start_timestamp"],
    )
    if records["target_id"].tolist() != list(target_ids):
        raise RuntimeError(
            f"{experiment_id} ordered target IDs changed during window construction"
        )
    feature_matrix = frame.loc[:, feature_order].to_numpy(dtype=np.float32, copy=True)
    targets = frame["Appliances"].to_numpy(dtype=np.float64, copy=True)
    dataset = V2PretestWindowDataset(
        feature_matrix,
        targets,
        records,
        feature_order,
        experiment_id=experiment_id,
        feature_variant_id=feature_variant_id,
    )
    audit = PretestLoadAudit(
        source_kind=source_kind,
        rows_loaded=len(frame),
        train_rows_loaded=int(split_manifest["train_rows"]),
        validation_rows_loaded=int(split_manifest["validation_rows"]),
        test_rows_read=0,
        test_target_ids_seen=0,
        first_timestamp=str(frame["timestamp"].iloc[0]),
        last_timestamp=str(frame["timestamp"].iloc[-1]),
        test_start_timestamp=str(split_manifest["test_start_timestamp"]),
        feature_order_status="PASS",
        population_status="PASS",
        fold_fingerprint_status=evidence.fingerprint_status,
        phase9_scaler_provenance_status="PASS_NOT_USED_FOR_TRAINING",
    )
    return dataset, evidence, audit


def build_e01_pretest_dataset(
    project_root: Path,
    feature_order: Sequence[str],
) -> tuple[V2PretestWindowDataset, CanonicalFoldEvidence, PretestLoadAudit]:
    """Backward-compatible E01 wrapper with its exact FS2_TF1 contract."""

    return build_v2_pretest_dataset(
        project_root,
        feature_order,
        experiment_id="E01",
        feature_variant_id="FS2_TF1",
    )
