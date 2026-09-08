"""Phase 44 — internal utilities.

Lightweight helpers used across the orchestrator and stage modules.
"""
from __future__ import annotations

from typing import List

import torch


def collate_batch_to_tensors(batch_list: List[dict]) -> dict:
    """Stack a list of per-sample dicts into a single batch.

    For each key:
      - if every value is a torch.Tensor with the same shape, stack on dim=0
      - if values are scalars (e.g. target_id), convert to a tensor
      - if values are lists of length 1, convert to a tensor
    """
    if not batch_list:
        return {}
    keys = batch_list[0].keys()
    result = {}
    for key in keys:
        values = [b[key] for b in batch_list]
        first = values[0]
        if isinstance(first, torch.Tensor):
            if all(v.shape == first.shape for v in values):
                result[key] = torch.stack(values, dim=0)
            else:
                result[key] = [v.tolist() if hasattr(v, "tolist") else v for v in values]
        else:
            try:
                result[key] = torch.as_tensor(values)
            except (TypeError, ValueError):
                result[key] = values
    return result
