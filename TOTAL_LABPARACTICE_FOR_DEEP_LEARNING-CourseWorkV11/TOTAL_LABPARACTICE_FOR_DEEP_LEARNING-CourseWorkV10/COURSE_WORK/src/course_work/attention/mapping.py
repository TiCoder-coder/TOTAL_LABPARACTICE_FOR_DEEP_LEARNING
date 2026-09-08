from pathlib import Path
from typing import Any

import numpy as np
import torch


def build_position_mapping(lookback: int) -> list[dict[str, Any]]:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    rows: list[dict[str, Any]] = []
    for position in range(lookback):
        lag_steps = lookback - 1 - position
        rows.append(
            {
                "position_index": position,
                "lag_steps": lag_steps,
                "lag_minutes": lag_steps * 10,
                "is_last_step": position == lookback - 1,
            }
        )
    return rows


def map_attention_to_lags(attention: torch.Tensor, lookback: int) -> np.ndarray:
    if attention.ndim != 4:
        raise ValueError("attention must have shape [B, H, L, L]")
    batch_size, num_heads, query_len, key_len = attention.shape
    if query_len != lookback or key_len != lookback:
        raise ValueError("attention sequence length mismatch")
    last_query = attention[:, :, -1, :]
    return last_query.detach().cpu().numpy()


def aggregate_head_attention(attention: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if attention.ndim != 4:
        raise ValueError("attention must have shape [B, H, L, L]")
    if mode == "mean":
        return attention.mean(dim=1)
    if mode == "sum":
        return attention.sum(dim=1)
    raise ValueError(f"Unsupported aggregation mode: {mode}")
