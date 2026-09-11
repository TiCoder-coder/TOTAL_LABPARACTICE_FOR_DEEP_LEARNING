"""Strict optimizer construction for MODEL_IMPROVEMENT-v2 experiments.

The default is deliberately AdamW so legacy configs that predate an explicit
``optimizer_name`` retain their historical construction semantics.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def build_optimizer(
    model_parameters: Iterable[torch.nn.Parameter],
    optimizer_name: str | None,
    learning_rate: float,
    weight_decay: float,
    optimizer_config: Mapping[str, Any] | None = None,
) -> torch.optim.Optimizer:
    """Build AdamW or the strictly bounded E08/E09 SGD optimizers."""

    name = "AdamW" if optimizer_name in {None, ""} else str(optimizer_name)
    config = dict(optimizer_config or {})
    if name == "AdamW":
        # Historical code ignored optimizer-specific extras and always built
        # AdamW from lr + weight_decay only; retain that behavior exactly.
        return torch.optim.AdamW(
            model_parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    if name == "SGD":
        if set(config) != {"momentum", "nesterov"}:
            raise ValueError("SGD optimizer_config requires exactly momentum and nesterov")
        momentum = float(config["momentum"])
        if momentum not in {0.0, 0.9} or config["nesterov"] is not False:
            raise ValueError(
                "V2 SGD requires momentum in {0.0, 0.9} and nesterov=False"
            )
        return torch.optim.SGD(
            model_parameters,
            lr=float(learning_rate),
            momentum=momentum,
            weight_decay=float(weight_decay),
            nesterov=False,
        )
    raise ValueError(f"Unsupported optimizer_name: {name!r}")
