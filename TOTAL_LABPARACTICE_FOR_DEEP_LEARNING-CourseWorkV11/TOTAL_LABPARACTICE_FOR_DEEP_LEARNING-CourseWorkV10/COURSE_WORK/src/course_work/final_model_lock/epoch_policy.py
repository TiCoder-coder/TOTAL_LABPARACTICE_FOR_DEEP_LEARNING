"""Epoch policy for Phase45: median of 3 inner best epochs.

Forbidden fallbacks (plan §194):
- fallback to 50
- mean / rounded mean
- LSTM epoch
- outer-eval epoch
- original Validation best epoch
- seed-specific epoch
- manual override
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class FinalEpochDecision:
    RO1: int
    RO2: int
    RO3: int
    sorted_epochs: tuple[int, int, int]
    FINAL_REFIT_EPOCHS: int
    source_run_id_RO1: str | None
    source_run_id_RO2: str | None
    source_run_id_RO3: str | None
    candidate_max_epochs: int
    aggregation_rule: str
    within_cap: bool
    status: str 


class EpochPolicyError(RuntimeError):
    """Raised when epoch policy is violated."""


def derive_final_epoch(
    handoff: dict[str, Any],
    locked_candidate_max_epochs: int,
) -> FinalEpochDecision:
    """Compute FINAL_REFIT_EPOCHS as median(RO1, RO2, RO3) of the recommended Transformer's three Stage-A inner best epochs.

    Hard-fails on:
      - any missing fold epoch
      - any epoch <= 0
      - any epoch > candidate.max_epochs
      - any forbidden fallback path
    """
    epochs_raw: dict[str, Any] = handoff.get(
        "recommended_transformer_inner_best_epochs", {}
    )
    for k in ("RO1", "RO2", "RO3"):
        if k not in epochs_raw:
            raise EpochPolicyError(f"Missing inner best epoch for {k}")
        v = epochs_raw[k]
        if not isinstance(v, int) or v <= 0:
            raise EpochPolicyError(
                f"{k} inner best epoch must be a positive int, got {v!r}"
            )
    ro1, ro2, ro3 = int(epochs_raw["RO1"]), int(epochs_raw["RO2"]), int(epochs_raw["RO3"])

    for label, ep in (("RO1", ro1), ("RO2", ro2), ("RO3", ro3)):
        if ep > locked_candidate_max_epochs:
            raise EpochPolicyError(
                f"{label} epoch {ep} exceeds candidate max_epochs {locked_candidate_max_epochs}"
            )

    sorted_epochs: tuple[int, int, int] = tuple(sorted((ro1, ro2, ro3)))
    final = int(median([ro1, ro2, ro3]))

    stage_a_ids = handoff.get("recommended_transformer_stage_a_run_ids") or {}
    ro1_run = stage_a_ids.get("RO1")
    ro2_run = stage_a_ids.get("RO2")
    ro3_run = stage_a_ids.get("RO3")

    return FinalEpochDecision(
        RO1=ro1,
        RO2=ro2,
        RO3=ro3,
        sorted_epochs=sorted_epochs,
        FINAL_REFIT_EPOCHS=final,
        source_run_id_RO1=ro1_run,
        source_run_id_RO2=ro2_run,
        source_run_id_RO3=ro3_run,
        candidate_max_epochs=locked_candidate_max_epochs,
        aggregation_rule="MEDIAN_RO_INNER_BEST_EPOCHS-v1",
        within_cap=final <= locked_candidate_max_epochs,
        status="PASS",
    )


def build_epoch_source_audit(
    decision: FinalEpochDecision,
) -> list[dict[str, Any]]:
    """Per-row CSV body for ``final_epoch_source_audit.csv``."""
    return [
        {"fold_id": "RO1",
         "candidate_id": "recommended_transformer",
         "best_epoch_inner": decision.RO1,
         "source_stage_a_run_id": decision.source_run_id_RO1 or "",
         "max_epochs_cap": decision.candidate_max_epochs,
         "within_cap": decision.RO1 <= decision.candidate_max_epochs},
        {"fold_id": "RO2",
         "candidate_id": "recommended_transformer",
         "best_epoch_inner": decision.RO2,
         "source_stage_a_run_id": decision.source_run_id_RO2 or "",
         "max_epochs_cap": decision.candidate_max_epochs,
         "within_cap": decision.RO2 <= decision.candidate_max_epochs},
        {"fold_id": "RO3",
         "candidate_id": "recommended_transformer",
         "best_epoch_inner": decision.RO3,
         "source_stage_a_run_id": decision.source_run_id_RO3 or "",
         "max_epochs_cap": decision.candidate_max_epochs,
         "within_cap": decision.RO3 <= decision.candidate_max_epochs},
    ]


def build_epoch_policy_contract(decision: FinalEpochDecision) -> dict[str, Any]:
    """Build ``final_epoch_policy.json`` payload."""
    return {
        "policy_id": decision.aggregation_rule,
        "source_phase": 44,
        "source_candidate_id": "recommended_transformer",
        "RO1_inner_best_epoch": decision.RO1,
        "RO2_inner_best_epoch": decision.RO2,
        "RO3_inner_best_epoch": decision.RO3,
        "sorted_epochs": list(decision.sorted_epochs),
        "final_refit_epochs": decision.FINAL_REFIT_EPOCHS,
        "candidate_max_epochs": decision.candidate_max_epochs,
        "within_cap": decision.within_cap,
        "no_test_dependency": True,
        "no_seed_dependency": True,
        "no_validation_dependency": True,
        "no_manual_override": True,
        "aggregation_rule": decision.aggregation_rule,
        "forbidden_fallbacks_rejected": [
            "fallback_to_50",
            "mean",
            "rounded_mean",
            "lstm_epoch",
            "outer_eval_epoch",
            "original_validation_best_epoch",
            "phase21_epoch",
            "seed_specific_epoch",
            "manual_override",
        ],
        "status": decision.status,
    }


__all__ = [
    "FinalEpochDecision",
    "EpochPolicyError",
    "derive_final_epoch",
    "build_epoch_source_audit",
    "build_epoch_policy_contract",
]
