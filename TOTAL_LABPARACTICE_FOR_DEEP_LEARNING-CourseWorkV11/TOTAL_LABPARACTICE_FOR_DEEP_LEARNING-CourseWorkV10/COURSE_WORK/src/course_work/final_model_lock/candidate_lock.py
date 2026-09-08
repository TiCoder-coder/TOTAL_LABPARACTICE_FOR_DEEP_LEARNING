"""Lock the recommended Transformer candidate.

Final model family MUST remain TRANSFORMER_ENCODER. LSTM and Persistence are
baseline context only.

Returns a deterministic ``LockedCandidate`` dataclass containing the candidate
id, its Phase42-derived config, its fingerprint, and provenance info.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .inputs import load_phase42_shortlist


@dataclass(frozen=True)
class LockedCandidate:
    candidate_id: str
    config_fingerprint: str  # 64-hex
    config: dict[str, Any]
    shortlist_position: int
    changed_factor: str
    changed_to: str
    model_family: str
    lookback_steps: int
    feature_variant_id: str
    target_scaling_option: str
    source_phase: int
    evidence_class: str


class CandidateLockError(RuntimeError):
    """Raised when the recommended candidate cannot be locked."""


def lock_candidate(
    handoff: dict[str, Any],
    shortlist_payload: dict[str, Any] | None = None,
    shortlist_path: str | None = None,
) -> LockedCandidate:
    """Lock the candidate declared by ``handoff['recommended_transformer_*']``.

    Verifies:
      - The candidate exists in the Phase42 shortlist.
      - The Phase42 fingerprint equals the handoff fingerprint.
      - The candidate's ``model.model_family`` is TRANSFORMER_ENCODER.
    """
    if shortlist_payload is None:
        if shortlist_path is None:
            raise CandidateLockError("Must provide shortlist_payload or shortlist_path")
        shortlist_payload = load_phase42_shortlist(__import__("pathlib").Path(shortlist_path))

    candidate_id: str = handoff["recommended_transformer_candidate_id"]
    handoff_fp: str = handoff["recommended_transformer_fingerprint"]

    found: dict[str, Any] | None = None
    for c in shortlist_payload.get("candidates", []):
        if c.get("candidate_id") == candidate_id:
            found = c
            break
    if found is None:
        raise CandidateLockError(
            f"Recommended candidate {candidate_id!r} missing from Phase42 shortlist"
        )

    shortlist_fp = found.get("candidate_config_fingerprint")
    if shortlist_fp != handoff_fp:
        raise CandidateLockError(
            "Phase42 shortlist fingerprint != Phase44 handoff fingerprint for "
            f"{candidate_id}: {shortlist_fp!r} vs {handoff_fp!r}"
        )

    config: dict[str, Any] = found.get("config", {})

    model_section = config.get("model", {})
    family = model_section.get("model_family")
    if family != "TRANSFORMER_ENCODER":
        raise CandidateLockError(
            f"Recommended candidate {candidate_id!r} is not TRANSFORMER_ENCODER "
            f"(got {family!r})"
        )

    data_section = config.get("data", {})
    return LockedCandidate(
        candidate_id=candidate_id,
        config_fingerprint=handoff_fp,
        config=config,
        shortlist_position=int(found.get("shortlist_position", -1)),
        changed_factor=str(found.get("changed_factor", "")),
        changed_to=str(found.get("changed_to", "")),
        model_family=family,
        lookback_steps=int(data_section.get("lookback_steps", 0)),
        feature_variant_id=str(data_section.get("feature_variant_id", "")),
        target_scaling_option=str(data_section.get("target_scaling_option", "")),
        source_phase=int(found.get("source_phase", 42)),
        evidence_class=str(found.get("evidence_class", "")),
    )


__all__ = ["LockedCandidate", "CandidateLockError", "lock_candidate"]
