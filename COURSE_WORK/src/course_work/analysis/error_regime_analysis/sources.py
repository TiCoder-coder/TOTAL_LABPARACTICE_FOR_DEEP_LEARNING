"""Phase 50 — read-only sources loader (frozen Phase 47/48/49 artifacts)."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import get_project_root


# Canonical artifact paths
PHASE47_SIGNOFF_REL = "artifacts/final_test/phase_47_signoff.json"
PHASE47_POPULATION_MANIFEST_REL = "artifacts/final_test/final_test_population_manifest.json"
PHASE47_PREDICTION_CHECKSUMS_REL = "artifacts/final_test/prediction_checksums.json"
PHASE47_PREDICTIONS_DIR_REL = "artifacts/final_test/predictions"
PHASE47_PERSISTENCE_PRED_REL = (
    "artifacts/final_test/predictions/final_test_predictions_persistence.csv"
)
PHASE47_SUMMARY_REL = "artifacts/final_test/final_test_summary.json"
PHASE47_LSTM_ELIGIBILITY_REL = "artifacts/final_test/final_test_lstm_eligibility.json"

PHASE48_SIGNOFF_REL = "artifacts/prediction_analysis/phase_48_signoff.json"
PHASE48_SEED_SPREAD_REL = (
    "artifacts/prediction_analysis/prediction_seed_spread.csv"
)
PHASE48_TOP_DISAGREEMENT_REL = (
    "artifacts/prediction_analysis/prediction_top_seed_disagreement.csv"
)

PHASE49_SIGNOFF_REL = "artifacts/residual_analysis/phase_49_signoff.json"
PHASE49_RESIDUAL_LONG_REL = "artifacts/residual_analysis/residual_long_table.csv"
PHASE49_RESIDUAL_WIDE_REL = "artifacts/residual_analysis/residual_wide_table.csv"
PHASE49_SIGN_CONSENSUS_REL = (
    "artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv"
)

SPLIT_MANIFEST_REL = "artifacts/splits/split_manifest.json"
SPLIT_MEMBERSHIP_REL = "artifacts/splits/split_membership.csv"
WINDOWPOP_MANIFEST_REL = "artifacts/windows/window_manifest.json"
WINDOWPOP_COMMON_TARGET_REL = "artifacts/windows/common_target_population.csv"
WINDOWPOP_FINGERPRINTS_REL = "artifacts/windows/window_fingerprints.json"
TEMPORAL_MANIFEST_REL = "artifacts/temporal/temporal_manifest.json"

RAW_APPLIANCES_REL = "data/raw_data/energydata_complete.csv"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def project_root() -> Path:
    return get_project_root()


def load_phase47_signoff(root_or_none: Path | None = None) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / PHASE47_SIGNOFF_REL)


def load_phase47_population(root_or_none: Path | None = None) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / PHASE47_POPULATION_MANIFEST_REL)


def load_phase47_prediction_checksums(
    root_or_none: Path | None = None,
) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / PHASE47_PREDICTION_CHECKSUMS_REL)


def load_phase47_summary(root_or_none: Path | None = None) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / PHASE47_SUMMARY_REL)


def load_phase47_lstm_eligibility(root_or_none: Path | None = None) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / PHASE47_LSTM_ELIGIBILITY_REL)


def load_phase47_persistence_predictions(
    root_or_none: Path | None = None,
) -> tuple[list[dict[str, str]], str]:
    root = root_or_none if root_or_none is not None else project_root()
    p = root / PHASE47_PERSISTENCE_PRED_REL
    rows = _read_csv(p)
    return rows, _sha256(p)


def load_phase48_signoff(root_or_none: Path | None = None) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / PHASE48_SIGNOFF_REL)


def load_phase48_seed_spread(
    root_or_none: Path | None = None,
) -> tuple[list[dict[str, str]], str]:
    root = root_or_none if root_or_none is not None else project_root()
    p = root / PHASE48_SEED_SPREAD_REL
    rows = _read_csv(p)
    return rows, _sha256(p)


def load_phase49_signoff(root_or_none: Path | None = None) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / PHASE49_SIGNOFF_REL)


def load_phase49_residual_long(
    root_or_none: Path | None = None,
) -> tuple[list[dict[str, str]], str]:
    root = root_or_none if root_or_none is not None else project_root()
    p = root / PHASE49_RESIDUAL_LONG_REL
    rows = _read_csv(p)
    return rows, _sha256(p)


def load_phase49_residual_wide(
    root_or_none: Path | None = None,
) -> tuple[list[dict[str, str]], str]:
    root = root_or_none if root_or_none is not None else project_root()
    p = root / PHASE49_RESIDUAL_WIDE_REL
    rows = _read_csv(p)
    return rows, _sha256(p)


def load_phase49_sign_consensus(
    root_or_none: Path | None = None,
) -> tuple[list[dict[str, str]], str]:
    root = root_or_none if root_or_none is not None else project_root()
    p = root / PHASE49_SIGN_CONSENSUS_REL
    rows = _read_csv(p)
    return rows, _sha256(p)


def load_windowpop_manifest(root_or_none: Path | None = None) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / WINDOWPOP_MANIFEST_REL)


def load_windowpop_common_target(
    root_or_none: Path | None = None,
) -> tuple[list[dict[str, str]], str]:
    root = root_or_none if root_or_none is not None else project_root()
    p = root / WINDOWPOP_COMMON_TARGET_REL
    rows = _read_csv(p)
    return rows, _sha256(p)


def load_windowpop_fingerprints(root_or_none: Path | None = None) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / WINDOWPOP_FINGERPRINTS_REL)


def load_split_manifest(root_or_none: Path | None = None) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / SPLIT_MANIFEST_REL)


def load_split_membership(root_or_none: Path | None = None) -> list[dict[str, str]]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_csv(root / SPLIT_MEMBERSHIP_REL)


def load_temporal_manifest(root_or_none: Path | None = None) -> dict[str, Any]:
    root = root_or_none if root_or_none is not None else project_root()
    return _read_json(root / TEMPORAL_MANIFEST_REL)


def load_raw_appliances(
    root_or_none: Path | None = None,
) -> tuple[list[dict[str, str]], str]:
    root = root_or_none if root_or_none is not None else project_root()
    p = root / RAW_APPLIANCES_REL
    rows = _read_csv(p)
    return rows, _sha256(p)
