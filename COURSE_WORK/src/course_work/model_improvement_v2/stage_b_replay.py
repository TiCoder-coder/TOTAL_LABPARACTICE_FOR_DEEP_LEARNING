"""V2 E07-D — Stage-A exact LR trace replay for Stage-B refit.

Stage B must NOT recompute scheduler behavior. Instead, for each
candidate/fold:

  1. Load the completed Stage-A training_history.csv
  2. Read selected best_epoch_inner
  3. Extract ordered lr_used_for_epoch list for epochs 1..best_epoch_inner
  4. In Stage B, BEFORE each epoch k:
       optimizer.param_groups[i].lr = trace[k - 1]  for all i
  5. Train that epoch

Therefore Stage-B LR sequence MUST be EXACTLY equal to Stage-A LR sequence
through best_epoch_inner.

For REDUCE_ON_PLATEAU this is mandatory because Stage B has no validation
metrics — Plateu.step(metric) cannot be called.

For OFF (no scheduler), the trace is constant 3e-4; Stage-B replay is a
no-op but the metadata is still recorded.

Scope: E07-D only. The actual orchestration (loading Stage-A history,
building the trace, passing it to RefitEngine) lives in the E07 runner
(sub-part E07-E or later).

This module provides:
  - validate_lr_trace(trace, best_epoch_inner)
  - set_optimizer_lr(optimizer, lr_value)
  - extract_lr_trace_from_history(history_dict)
  - LR_POLICY_STAGE_A_REPLAY constant

Stage-B uses the canonical schema `STAGE_A_EXACT_TRACE_REPLAY`.
OFF / non-replay paths use `STAGE_B_CONSTANT_LR_FROM_CONFIG` (the existing
default), so backward-compat is preserved.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.optim import Optimizer

__all__ = [
    "validate_lr_trace",
    "set_optimizer_lr",
    "extract_lr_trace_from_history",
    "LR_POLICY_STAGE_A_REPLAY",
    "LR_POLICY_STAGE_B_CONSTANT",
]

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
LR_POLICY_STAGE_A_REPLAY = "STAGE_A_EXACT_TRACE_REPLAY"
LR_POLICY_STAGE_B_CONSTANT = "STAGE_B_CONSTANT_LR_FROM_CONFIG"


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate_lr_trace(
    trace: list[float] | tuple[float, ...] | None,
    best_epoch_inner: int,
) -> list[float]:
    """Validate the Stage-A LR trace against best_epoch_inner.

    Parameters
    ----------
    trace : list[float] | tuple[float, ...] | None
        Ordered list of lr_used_for_epoch values from Stage A
        (index 0 = epoch 1, index 1 = epoch 2, ...).
        None is treated as empty (replay disabled).
    best_epoch_inner : int
        Selected best_epoch_inner from Stage A.

    Returns
    -------
    list[float]
        The validated trace (as a list).

    Raises
    ------
    ValueError
        If trace is None/empty, or len(trace) < best_epoch_inner,
        or any element is non-finite or negative.
    """
    if best_epoch_inner < 1:
        raise ValueError(
            f"best_epoch_inner must be >= 1 (got {best_epoch_inner})"
        )
    if trace is None:
        raise ValueError(
            f"Stage-B LR replay requires a non-empty trace, got None "
            f"(best_epoch_inner={best_epoch_inner})"
        )
    trace_list = list(trace)
    if len(trace_list) == 0:
        raise ValueError(
            f"Stage-B LR replay requires a non-empty trace, got empty list "
            f"(best_epoch_inner={best_epoch_inner})"
        )
    if len(trace_list) < best_epoch_inner:
        raise ValueError(
            f"Stage-B LR trace too short: len(trace)={len(trace_list)} < "
            f"best_epoch_inner={best_epoch_inner}"
        )
    for idx, lr in enumerate(trace_list):
        if not isinstance(lr, (int, float)):
            raise ValueError(
                f"Stage-B LR trace[{idx}] is not a number: {lr!r}"
            )
        lr_f = float(lr)
        if not math.isfinite(lr_f):
            raise ValueError(
                f"Stage-B LR trace[{idx}] is not finite: {lr_f!r}"
            )
        if lr_f < 0:
            raise ValueError(
                f"Stage-B LR trace[{idx}] is negative: {lr_f!r}"
            )
    return trace_list


def set_optimizer_lr(optimizer: "Optimizer", lr_value: float) -> None:
    """Set the LR for every parameter group of the optimizer.

    This is the explicit Stage-B LR replay action:
    optimizer.param_groups[i].lr = lr_value for all i.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr_value)


# ----------------------------------------------------------------------
# Trace extraction
# ----------------------------------------------------------------------
def extract_lr_trace_from_history(history: dict[str, Any]) -> list[float]:
    """Extract lr_used_for_epoch column from a Stage-A training_history dict.

    Parameters
    ----------
    history : dict
        A dict shaped like a training_history.csv row list, or a
        pandas DataFrame converted to dict(orient="records").
        Each row must contain `epoch` and `lr_used_for_epoch`.

    Returns
    -------
    list[float]
        Ordered list, index 0 = epoch 1, index 1 = epoch 2, ...
        Length is min(epochs present in history).
    """
    if isinstance(history, dict):
        # Accept a {column: list} shaped dict (pd.DataFrame.to_dict())
        if "lr_used_for_epoch" in history and "epoch" in history:
            epochs = list(history["epoch"])
            lrs = list(history["lr_used_for_epoch"])
            # Sort by epoch to be safe
            paired = sorted(zip(epochs, lrs), key=lambda x: int(x[0]))
            return [float(lr) for _, lr in paired]
        # Accept a list-of-rows dict
        if "records" in history and isinstance(history["records"], list):
            rows = history["records"]
            sorted_rows = sorted(rows, key=lambda r: int(r["epoch"]))
            return [float(r["lr_used_for_epoch"]) for r in sorted_rows]
    raise ValueError(
        f"Cannot extract lr trace from history of type {type(history).__name__}"
    )
