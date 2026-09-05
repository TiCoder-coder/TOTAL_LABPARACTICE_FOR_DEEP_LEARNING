"""Phase 48 — load and verify frozen Phase 47 prediction bundles.

Phase 48 inputs.py NEVER mutates the source files. It only reads them,
verifies SHA-256 / row count / target_ids / timestamps / y_true, and
exposes an immutable in-memory bundle object.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from course_work.phase48.contract import (
    LOCKED_TRANSFORMER_SEEDS,
    OUTPUT_DIR,
    PERSISTENCE_LABEL,
)


@dataclass(frozen=True)
class PredictionBundle:
    model_id: str
    seed: str
    path: Path
    sha256: str
    n_rows: int
    target_ids: tuple[str, ...]
    target_timestamps: tuple[str, ...]
    y_true_wh: tuple[float, ...]
    y_pred_wh: tuple[float, ...]
    source_population_sha256: Optional[str] = field(default=None)


@dataclass(frozen=True)
class Phase48Inputs:
    transformer_bundles: dict[int, PredictionBundle]
    persistence_bundle: PredictionBundle
    population_sha256: str
    n_test: int
    first_target_id: str
    last_target_id: str
    first_target_timestamp: str
    last_target_timestamp: str
    final_lock_sha256: str
    phase47_signoff_sha256: str
    source_verification: tuple[dict, ...]
    checksum_registry_sha256: str


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_ts(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _load_bundle_csv(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    target_ids: list[str] = []
    target_timestamps: list[str] = []
    y_true_wh: list[float] = []
    y_pred_wh: list[float] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            target_ids.append(row["target_id"])
            target_timestamps.append(row["target_timestamp"])
            y_true_wh.append(float(row["y_true_wh"]))
            y_pred_wh.append(float(row["y_pred_wh"]))
    return (
        tuple(target_ids),
        tuple(target_timestamps),
        tuple(y_true_wh),
        tuple(y_pred_wh),
    )


def _bundle_chronological(timestamps: tuple[str, ...]) -> bool:
    parsed = [_parse_ts(t) for t in timestamps]
    if any(p is None for p in parsed):
        return False
    for a, b in zip(parsed, parsed[1:]):
        if a >= b:
            return False
    return True


def _bundle_all_finite(ys: tuple[float, ...]) -> bool:
    return all(math.isfinite(y) for y in ys)


def _bundle_unique(target_ids: tuple[str, ...]) -> bool:
    return len(set(target_ids)) == len(target_ids)


def load_phase48_inputs(
    final_test_dir: Path = Path("artifacts/final_test"),
    predictions_dir: Optional[Path] = None,
) -> Phase48Inputs:
    """Load frozen Phase 47 bundles + verify checksums.

    Inputs are READ-ONLY. No source file is modified.
    """
    if predictions_dir is None:
        predictions_dir = final_test_dir / "predictions"

    checksum_registry = json.loads((final_test_dir / "prediction_checksums.json").read_text())
    population_manifest = json.loads((final_test_dir / "final_test_population_manifest.json").read_text())
    phase47_signoff = json.loads((final_test_dir / "phase_47_signoff.json").read_text())
    phase47_signoff_sha = _sha256_file(final_test_dir / "phase_47_signoff.json")

    pop_sha = population_manifest["target_ids_sha256"]
    n_test = population_manifest["target_count"]
    first_id = population_manifest["first_target_id"]
    last_id = population_manifest["last_target_id"]
    first_ts = population_manifest["first_target_timestamp"]
    last_ts = population_manifest["last_target_timestamp"]

    transformer_bundles: dict[int, PredictionBundle] = {}
    source_verification_rows: list[dict] = []

    for seed in LOCKED_TRANSFORMER_SEEDS:
        rel = checksum_registry["predictions"][f"seed_{seed}"]["path"]
        path = Path(rel)
        observed_sha = _sha256_file(path)
        expected_sha = checksum_registry["predictions"][f"seed_{seed}"]["sha256"]
        observed_rows = sum(1 for _ in path.open()) - 1
        expected_rows = checksum_registry["predictions"][f"seed_{seed}"]["rows"]
        target_ids, target_timestamps, y_true, y_pred = _load_bundle_csv(path)
        bundle = PredictionBundle(
            model_id=f"TRANSFORMER_SEED{seed}",
            seed=str(seed),
            path=path,
            sha256=observed_sha,
            n_rows=len(target_ids),
            target_ids=target_ids,
            target_timestamps=target_timestamps,
            y_true_wh=y_true,
            y_pred_wh=y_pred,
            source_population_sha256=pop_sha,
        )
        transformer_bundles[seed] = bundle
        source_verification_rows.append({
            "source_id": f"TRANSFORMER_SEED{seed}",
            "file": str(path),
            "expected_sha256": expected_sha,
            "observed_sha256": observed_sha,
            "row_count": observed_rows,
            "expected_row_count": expected_rows,
            "population_sha256": pop_sha,
            "frozen": True,
            "checksum_match": observed_sha == expected_sha,
            "row_count_match": observed_rows == expected_rows,
            "all_target_ids_unique": _bundle_unique(target_ids),
            "all_predictions_finite": _bundle_all_finite(y_pred),
            "all_y_true_finite": _bundle_all_finite(y_true),
            "chronological": _bundle_chronological(target_timestamps),
            "status": "PASS" if (observed_sha == expected_sha and observed_rows == expected_rows and _bundle_unique(target_ids) and _bundle_all_finite(y_pred) and _bundle_chronological(target_timestamps)) else "FAIL",
        })

    pers_entry = checksum_registry["predictions"]["persistence"]
    pers_path = Path(pers_entry["path"])
    pers_sha = _sha256_file(pers_path)
    pers_target_ids, pers_target_timestamps, pers_y_true, pers_y_pred = _load_bundle_csv(pers_path)
    persistence_bundle = PredictionBundle(
        model_id=PERSISTENCE_LABEL,
        seed=PERSISTENCE_LABEL,
        path=pers_path,
        sha256=pers_sha,
        n_rows=len(pers_target_ids),
        target_ids=pers_target_ids,
        target_timestamps=pers_target_timestamps,
        y_true_wh=pers_y_true,
        y_pred_wh=pers_y_pred,
        source_population_sha256=pop_sha,
    )
    pers_expected_sha = pers_entry["sha256"]
    pers_expected_rows = pers_entry["rows"]
    source_verification_rows.append({
        "source_id": PERSISTENCE_LABEL,
        "file": str(pers_path),
        "expected_sha256": pers_expected_sha,
        "observed_sha256": pers_sha,
        "row_count": len(pers_target_ids),
        "expected_row_count": pers_expected_rows,
        "population_sha256": pop_sha,
        "frozen": True,
        "checksum_match": pers_sha == pers_expected_sha,
        "row_count_match": len(pers_target_ids) == pers_expected_rows,
        "all_target_ids_unique": _bundle_unique(pers_target_ids),
        "all_predictions_finite": _bundle_all_finite(pers_y_pred),
        "all_y_true_finite": _bundle_all_finite(pers_y_true),
        "chronological": _bundle_chronological(pers_target_timestamps),
        "status": "PASS" if (pers_sha == pers_expected_sha and len(pers_target_ids) == pers_expected_rows and _bundle_unique(pers_target_ids) and _bundle_all_finite(pers_y_pred) and _bundle_chronological(pers_target_timestamps)) else "FAIL",
    })

    return Phase48Inputs(
        transformer_bundles=transformer_bundles,
        persistence_bundle=persistence_bundle,
        population_sha256=pop_sha,
        n_test=n_test,
        first_target_id=first_id,
        last_target_id=last_id,
        first_target_timestamp=first_ts,
        last_target_timestamp=last_ts,
        final_lock_sha256=phase47_signoff["final_lock_sha256"],
        phase47_signoff_sha256=phase47_signoff_sha,
        source_verification=tuple(source_verification_rows),
        checksum_registry_sha256=_sha256_file(final_test_dir / "prediction_checksums.json"),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
