"""Deterministic fingerprint generation for Phase45.

Four fingerprints:
- FINAL_MODEL_CONFIG_SHA256  : hash of canonical locked scientific config.
- FINAL_TRAINING_RECIPE_SHA256: hash of canonical Phase46 recipe.
- FINAL_LINEAGE_SHA256       : hash of canonical lineage payload.
- FINAL_MODEL_LOCK_SHA256    : SHA256(config_sha + recipe_sha + lineage_sha).

Inputs MUST NOT include timestamps or wall-clock fields. Same input →
same hash.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from course_work.utils.artifacts import canonical_json_bytes


def _sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def config_fingerprint(locked_config: dict[str, Any]) -> str:
    """Hash over the canonical locked scientific config (model + data + training + lineage)."""
    return _sha256(locked_config)


def recipe_fingerprint(recipe_dict: dict[str, Any]) -> str:
    return _sha256(recipe_dict)


def lineage_fingerprint(lineage_payload: dict[str, Any]) -> str:
    return _sha256(lineage_payload)


def lock_fingerprint(
    config_sha: str,
    recipe_sha: str,
    lineage_sha: str,
) -> str:
    """SHA256 over the deterministic concatenation of the three sub-hashes."""
    combined = f"{config_sha}\n{recipe_sha}\n{lineage_sha}\n"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


__all__ = [
    "config_fingerprint",
    "recipe_fingerprint",
    "lineage_fingerprint",
    "lock_fingerprint",
]
