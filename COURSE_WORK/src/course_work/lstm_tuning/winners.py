"""Phase 43 winner selection semantics.

Every scientific candidate:
    * full Validation evaluation each epoch (handled by training engine)
    * best selected by full-precision Validation RMSE Wh
    * MAE / R^2 secondary
    * exact tie parsimony per Phase 43 plan section 19
    * failed candidate retained (not omitted)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


TIER_ORDER = ("rmse_wh", "mae_wh", "r2")


@dataclass(frozen=True)
class StageWinner:
    option: str
    run_id: str | None
    value: Any
    validation_rmse_wh: float
    validation_mae_wh: float
    validation_r2: float
    best_epoch: int | None
    epochs_completed: int | None
    stop_reason: str | None
    trainable_parameters: int | None
    exact_tie: bool = False
    tie_rule_applied: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_metrics(candidate_record: dict[str, Any]) -> tuple[float, float, float]:
    metrics = candidate_record.get("metrics") or {}
    rmse = metrics.get("rmse_wh")
    mae = metrics.get("mae_wh")
    r2 = metrics.get("r2")
    if rmse is None:
        rmse = candidate_record.get("best_validation_rmse_wh")
    if rmse is None or mae is None or r2 is None:
        raise ValueError(f"Candidate {candidate_record.get('run_id')} missing required metrics")
    return float(rmse), float(mae), float(r2)


def _tie_rule_for(stage: str) -> str:
    return {
        "LT1": "smaller_hidden_size",
        "LT2": "fewer_layers",
        "LT3": "lower_dropout",
        "LT4": "lower_learning_rate",
        "LT5": "lower_weight_decay",
    }.get(stage, "no_tie_rule")


def _parsimony_sort_key(stage: str, candidate_value: Any) -> tuple[int, float]:
    try:
        return (0, float(candidate_value))
    except (TypeError, ValueError):
        return (0, 0.0)


def select_stage_winner(
    stage: str,
    candidates: list[dict[str, Any]],
) -> StageWinner:
    """Select the stage winner from completed candidate records.

    Each candidate record must have at minimum: option, value, run_id,
    metrics.rmse_wh, metrics.mae_wh, metrics.r2.
    """
    if not candidates:
        raise ValueError(f"No candidates provided for stage {stage}")

    completed = [c for c in candidates if c.get("run_id") is not None and c.get("metrics")]
    if not completed:
        raise ValueError(f"Stage {stage}: no completed candidates with metrics")

    best = None
    for cand in completed:
        rmse, mae, r2 = _candidate_metrics(cand)
        sort_key = (rmse, mae, -r2)
        if best is None or sort_key < best["sort_key"]:
            best = {
                "candidate": cand,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "sort_key": sort_key,
            }

    tied = [
        c
        for c in completed
        if (c.get("metrics", {}).get("rmse_wh") or c.get("best_validation_rmse_wh")) == best["rmse"]
    ]
    exact_tie = len(tied) > 1
    chosen = best["candidate"]
    if exact_tie:
        tied.sort(key=lambda c: _parsimony_sort_key(stage, c.get("value")))
        chosen = tied[0]

    rmse, mae, r2 = _candidate_metrics(chosen)
    return StageWinner(
        option=chosen["option"],
        run_id=chosen.get("run_id"),
        value=chosen.get("value"),
        validation_rmse_wh=rmse,
        validation_mae_wh=mae,
        validation_r2=r2,
        best_epoch=chosen.get("best_epoch"),
        epochs_completed=chosen.get("epochs_completed"),
        stop_reason=chosen.get("stop_reason"),
        trainable_parameters=chosen.get("trainable_parameters"),
        exact_tie=exact_tie,
        tie_rule_applied=_tie_rule_for(stage) if exact_tie else None,
    )


def assert_tuned_not_worse_than_reference(
    tuned_rmse_wh: float,
    reference_rmse_wh: float,
) -> None:
    if tuned_rmse_wh > reference_rmse_wh + 1e-12:
        raise ValueError(
            f"Tuned RMSE {tuned_rmse_wh:.6f} worse than reference RMSE {reference_rmse_wh:.6f}"
        )
