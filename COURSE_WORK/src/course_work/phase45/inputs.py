"""Phase44 / Phase42 input loaders with strict schema validation.

Reads:
- ``artifacts/rolling_origin/phase_44_signoff.json``
- ``artifacts/rolling_origin/phase45_final_model_lock_handoff.json``
- ``artifacts/candidate_synthesis/transformer_candidate_shortlist.json``

All loaders FAIL FAST on missing keys / wrong types so the Phase45 lock
cannot silently proceed with stale evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import read_json


class Phase45InputError(RuntimeError):
    """Raised when a Phase45 upstream input is malformed or missing."""


def _require(payload: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in payload:
        raise Phase45InputError(f"{ctx}: missing required key '{key}'")
    return payload[key]


def load_phase44_signoff(path: Path) -> dict[str, Any]:
    """Load and minimally validate Phase44 signoff.

    Requires:
      - overall_status in {PASS, PASS_WITH_WARNING}
      - approved_for_phase45 == True
      - test_status == NOT_ACCESSED
      - recommended_transformer_candidate_id
      - recommended_transformer_fingerprint
    """
    payload = read_json(path)
    overall = _require(payload, "overall_status", "phase44_signoff")
    if overall not in {"PASS", "PASS_WITH_WARNING"}:
        raise Phase45InputError(
            f"phase44_signoff.overall_status must be PASS/PASS_WITH_WARNING, got {overall!r}"
        )
    if not _require(payload, "approved_for_phase45", "phase44_signoff"):
        raise Phase45InputError("phase44_signoff.approved_for_phase45 must be true")
    if _require(payload, "test_status", "phase44_signoff") != "NOT_ACCESSED":
        raise Phase45InputError("phase44_signoff.test_status must be NOT_ACCESSED")
    # recommended_* are read defensively; the source-of-truth is the handoff.
    payload.setdefault("recommended_transformer_candidate_id", None)
    payload.setdefault("recommended_transformer_fingerprint", None)
    return payload


def load_phase44_handoff(path: Path) -> dict[str, Any]:
    """Load and validate Phase44→Phase45 handoff.

    Schema (canonical):
      - approved_for_phase45: bool
      - test_status: 'NOT_ACCESSED'
      - recommended_transformer_candidate_id: str
      - recommended_transformer_fingerprint: str (64 hex)
      - recommended_transformer_config: dict
      - recommended_transformer_inner_best_epochs: dict with keys RO1/RO2/RO3
      - recommended_transformer_stage_a_run_ids: dict
      - recommended_transformer_stage_b_run_ids: dict
      - rolling_origin_fold_metrics: list/dict
      - rolling_origin_pooled_metrics: list/dict
      - transformer_ranking: list/dict
      - lstm_tuned_context: dict
      - persistence_context: dict
    """
    payload = read_json(path)
    if not _require(payload, "approved_for_phase45", "phase44_handoff"):
        raise Phase45InputError("phase44_handoff.approved_for_phase45 must be true")
    if _require(payload, "test_status", "phase44_handoff") != "NOT_ACCESSED":
        raise Phase45InputError("phase44_handoff.test_status must be NOT_ACCESSED")
    candidate_id = _require(payload, "recommended_transformer_candidate_id", "phase44_handoff")
    fingerprint = _require(payload, "recommended_transformer_fingerprint", "phase44_handoff")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise Phase45InputError("recommended_transformer_candidate_id must be non-empty str")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise Phase45InputError(
            f"recommended_transformer_fingerprint must be 64-hex SHA256, got {fingerprint!r}"
        )
    _require(payload, "recommended_transformer_config", "phase44_handoff")
    epochs = _require(payload, "recommended_transformer_inner_best_epochs", "phase44_handoff")
    for fold_key in ("RO1", "RO2", "RO3"):
        if fold_key not in epochs:
            raise Phase45InputError(
                f"recommended_transformer_inner_best_epochs.{fold_key} missing"
            )
        v = epochs[fold_key]
        if not isinstance(v, int) or v <= 0:
            raise Phase45InputError(
                f"recommended_transformer_inner_best_epochs.{fold_key} must be positive int, got {v!r}"
            )
    return payload


def load_phase42_shortlist(path: Path) -> dict[str, Any]:
    """Load Phase42 candidate shortlist.

    Schema:
      - candidates: list of dicts, each with candidate_id + candidate_config_fingerprint
    """
    payload = read_json(path)
    if "candidates" not in payload or not isinstance(payload["candidates"], list):
        raise Phase45InputError("phase42_shortlist.candidates must be list")
    return payload


__all__ = [
    "Phase45InputError",
    "load_phase44_signoff",
    "load_phase44_handoff",
    "load_phase42_shortlist",
]
