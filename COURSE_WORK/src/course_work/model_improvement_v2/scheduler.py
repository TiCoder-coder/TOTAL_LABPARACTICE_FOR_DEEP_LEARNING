"""V2 AdamW learning-rate scheduler construction and step contract.

Scope: E07-B scheduler ablation (OFF / COSINE / REDUCE_ON_PLATEAU).
Stage-A only. No Stage-B LR replay in this sub-part.

Step contracts (E07 config snapshot):
  COSINE:        scheduler.step() after each completed epoch (no metric).
  REDUCE_ON_PLATEAU: scheduler.step(val_rmse_wh) after validation.

Both schedulers use the same initial lr=3e-4 as constant-LR E01/E06.
Epoch 1 trains at lr=3e-4 (scheduler is fresh, not stepped before training).

V1 / non-scheduler configs: unchanged.
"""
from __future__ import annotations

import torch.optim.lr_scheduler
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

__all__ = [
    "build_scheduler",
    "SUPPORTED_SCHEDULERS",
    "SCHEDULER_CONTRACTS",
]

# ----------------------------------------------------------------------
# Scheduler registry
# ----------------------------------------------------------------------
SUPPORTED_SCHEDULERS = frozenset({"OFF", "COSINE", "REDUCE_ON_PLATEAU"})


def _build_cosine(
    optimizer: "Optimizer",
    scheduler_config: dict[str, Any],
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Build CosineAnnealingLR.

    Config keys:
      T_max: int  — number of epochs (50 for E07 canonical).
      eta_min: float — minimum LR (1e-5 for E07 canonical).
      last_epoch: int — -1 means start fresh (no warm restart).

    Step contract:
      - scheduler.step() is called ONCE after each completed training epoch.
      - NO metric argument.
      - Epoch 1: optimizer starts at lr=3e-4; scheduler stepped after epoch 1
        completes → lr becomes cos(π/50) * initial.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_config.get("T_max", 50)),
        eta_min=float(scheduler_config.get("eta_min", 1e-5)),
        last_epoch=int(scheduler_config.get("last_epoch", -1)),
    )


def _build_reduce_on_plateau(
    optimizer: "Optimizer",
    scheduler_config: dict[str, Any],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build ReduceLROnPlateau.

    Config keys:
      mode: str — "min" (default).
      factor: float — LR multiplier on plateau (0.5 for E07 canonical).
      patience: int — epochs with no improvement before reducing (3 for E07).
      threshold: float — minimum change to qualify as improvement (0.0).
      threshold_mode: str — "abs" or "rel" ("abs" for E07).
      cooldown: int — epochs to wait before resuming reduction (0 for E07).
      min_lr: float — LR floor (1e-5 for E07 canonical).
      eps: float — decay applied to LR (1e-8 for E07 canonical).

    Step contract:
      - scheduler.step(val_rmse_wh) is called ONCE after each completed
        epoch and its validation RMSE is computed.
      - Epoch 1: optimizer starts at lr=3e-4; scheduler stepped after
        epoch 1 validation with val_rmse_wh.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_config.get("mode", "min")),
        factor=float(scheduler_config.get("factor", 0.5)),
        patience=int(scheduler_config.get("patience", 3)),
        threshold=float(scheduler_config.get("threshold", 0.0)),
        threshold_mode=str(scheduler_config.get("threshold_mode", "abs")),
        cooldown=int(scheduler_config.get("cooldown", 0)),
        min_lr=float(scheduler_config.get("min_lr", 1e-5)),
        eps=float(scheduler_config.get("eps", 1e-8)),
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
SCHEDULER_BUILDERS = {
    "COSINE": _build_cosine,
    "REDUCE_ON_PLATEAU": _build_reduce_on_plateau,
}


def build_scheduler(
    optimizer: "Optimizer",
    scheduler_name: str | None,
    scheduler_config: dict[str, Any] | None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build a PyTorch LR scheduler, or return None for OFF / None.

    Parameters
    ----------
    optimizer : Optimizer
        The freshly-constructed AdamW optimizer (not yet used).
    scheduler_name : str or None
        One of "OFF", "COSINE", "REDUCE_ON_PLATEAU", or None.
        "OFF" and None both produce None (no scheduler).
    scheduler_config : dict or None
        Scheduler-specific configuration dict. Required for COSINE
        and REDUCE_ON_PLATEAU; ignored for OFF.

    Returns
    -------
    LRScheduler or None
        A fresh scheduler instance (not stepped yet), or None for OFF.

    Raises
    ------
    ValueError
        If scheduler_name is unknown or config is missing for non-OFF.
    """
    if scheduler_name is None or scheduler_name == "OFF":
        return None

    if scheduler_name not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler_name: {scheduler_name!r}. "
            f"Supported: {sorted(SUPPORTED_SCHEDULERS)}"
        )

    if scheduler_config is None:
        raise ValueError(
            f"scheduler_config is required for {scheduler_name}"
        )

    builder = SCHEDULER_BUILDERS[scheduler_name]
    return builder(optimizer, scheduler_config)


# ----------------------------------------------------------------------
# Step contracts (documentation for callers)
# ----------------------------------------------------------------------
# COSINE:
#   for epoch in range(1, max_epochs + 1):
#       train()  # optimizer lr is already set
#       validate()
#       scheduler.step()           # ← no metric argument
#
# REDUCE_ON_PLATEAU:
#   for epoch in range(1, max_epochs + 1):
#       train()  # optimizer lr is already set
#       val_rmse = validate()
#       scheduler.step(val_rmse)  # ← metric argument (validation RMSE)
#
# OFF: no scheduler step. Optimizer lr stays constant at 3e-4.


SCHEDULER_CONTRACTS = {
    "OFF": {
        "step_contract": "NONE",
        "metric_argument": None,
        "epoch1_lr": "initial_lr",
    },
    "COSINE": {
        "step_contract": "ONCE_PER_COMPLETED_EPOCH",
        "metric_argument": None,
        "epoch1_lr": "initial_lr",
    },
    "REDUCE_ON_PLATEAU": {
        "step_contract": "AFTER_EACH_INNER_VALIDATION",
        "metric_argument": "INNER_VALIDATION_RMSE_WH",
        "epoch1_lr": "initial_lr",
    },
}
