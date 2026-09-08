"""Phase 52 — derived summaries: last-query, full-matrix streaming, recent-mass, top-source.

All summaries are EXTRACTION DIAGNOSTICS, NOT scientific interpretation.
No causal / importance / head-quality claims.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .sources import (
    EPS_H,
    LOOKBACK,
    NUM_HEADS,
    NUM_LAYERS,
    RECENT_WINDOWS_STEPS,
    TOP_K_SOURCES,
)


def _entropy(p: np.ndarray) -> float:
    """Negative entropy of a probability vector (base e)."""
    p_safe = np.clip(p, EPS_H, 1.0)
    return float(-(p_safe * np.log(p_safe)).sum())


def last_query_summary(
    seed: int,
    target_ids: tuple[str, ...],
    last_query: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-target × per-layer × per-head last-query summary.

    last_query shape: [N, L, H, L]
      last_query[n, layer, head, s] = A[n, layer, head, L-1, s]
    """
    N, _n_layers, H, LOOK = last_query.shape
    rows: list[dict[str, Any]] = []
    log_S = math.log(LOOK)
    for n in range(N):
        target_id = target_ids[n]
        for layer in range(NUM_LAYERS):
            for head in range(H):
                a = last_query[n, layer, head, :]  
                lag_steps = LOOK - np.arange(LOOK) 
                lag_minutes = 10 * lag_steps
                ent = _entropy(a)
                norm_ent = ent / log_S if log_S > 0 else 0.0
                exp_lag_steps = float((a * lag_steps).sum())
                exp_lag_minutes = 10.0 * exp_lag_steps

                max_w = float(a.max())
                tied_positions = np.flatnonzero(np.isclose(a, max_w))
                top1_position = int(tied_positions.max())
                top1_lag_steps = LOOK - top1_position
                top1_lag_minutes = 10 * top1_lag_steps
                top1_weight = max_w
                top1_tie_count = int(tied_positions.size)

                top_idx = np.argsort(-a)[: min(TOP_K_SOURCES, LOOK)]
                top5_mass = float(a[top_idx].sum())

                row = {
                    "seed": seed,
                    "target_id": target_id,
                    "layer_idx0": layer,
                    "head_idx0": head,
                    "entropy": ent,
                    "normalized_entropy": norm_ent,
                    "expected_lag_steps": exp_lag_steps,
                    "expected_lag_minutes": exp_lag_minutes,
                    "top1_source_position_idx0": top1_position,
                    "top1_lag_steps": top1_lag_steps,
                    "top1_lag_minutes": top1_lag_minutes,
                    "top1_weight": top1_weight,
                    "top1_tie_count": top1_tie_count,
                    "top5_mass": top5_mass,
                }

                for name, steps in RECENT_WINDOWS_STEPS.items():
                    eff_steps = min(steps, LOOK)
                    truncated = eff_steps < steps
                    row[f"{name}_effective_steps"] = int(eff_steps)
                    row[f"{name}_coverage_truncated"] = bool(truncated)
                    pos_lo = LOOK - eff_steps
                    row[f"{name}_mass"] = float(a[pos_lo:].sum())

                rows.append(row)
    return rows


def full_matrix_summary(
    seed: int,
    target_ids: tuple[str, ...],
    attention_layers_batch: list[np.ndarray],
) -> list[dict[str, Any]]:
    """Streaming per-target × per-layer × per-head whole-matrix metrics.

    attention_layers_batch[layer] shape: [B, H, L, L]
    """
    rows: list[dict[str, Any]] = []
    B = len(target_ids)
    L = attention_layers_batch[0].shape[-1]
    for n in range(B):
        target_id = target_ids[n]
        for layer_idx0, layer_attn in enumerate(attention_layers_batch):
            for head_idx0 in range(layer_attn.shape[1]):
                A = layer_attn[n, head_idx0, :, :]  
                row_sums = A.sum(axis=-1)  
                if not np.allclose(row_sums, 1.0, atol=1e-5):
                    norm = A / np.clip(row_sums[:, None], 1e-12, None)
                else:
                    norm = A
                ent = (-np.where(norm > 0, norm * np.log(np.clip(norm, EPS_H, None)), 0.0)).sum(axis=-1)
                mean_ent = float(ent.mean())
                mean_norm_ent = mean_ent / math.log(L) if L > 1 else 0.0

                diag = np.diag(A)
                self_mass = float(diag.mean())

                q_idx = np.arange(L)[:, None]
                s_idx = np.arange(L)[None, :]
                dist = np.abs(q_idx - s_idx)
                mean_abs_dist = float((A * dist).sum() / A.sum() if A.sum() > 0 else 0.0)

                triu_mask = np.triu(np.ones((L, L), dtype=bool), k=0)
                tril_mask = np.tril(np.ones((L, L), dtype=bool), k=-1)
                bwsm = float(A[triu_mask].mean()) if L > 0 else 0.0
                fwim = float(A[tril_mask].mean()) if L > 0 else 0.0

                rows.append({
                    "seed": seed,
                    "target_id": target_id,
                    "layer_idx0": layer_idx0,
                    "head_idx0": head_idx0,
                    "mean_query_entropy": mean_ent,
                    "mean_normalized_query_entropy": mean_norm_ent,
                    "mean_self_attention_weight": self_mass,
                    "mean_absolute_query_source_distance_steps": mean_abs_dist,
                    "backward_or_same_mass": bwsm,
                    "forward_within_input_mass": fwim,
                })
    return rows


def top_source_summary(
    seed: int,
    target_ids: tuple[str, ...],
    last_query: np.ndarray,
    top_k: int = TOP_K_SOURCES,
) -> list[dict[str, Any]]:
    """Top-K source positions by last-query weight (per-target × per-layer × per-head).

    last_query shape: [N, L, H, L]
    """
    N, L, H, _ = last_query.shape
    rows: list[dict[str, Any]] = []
    for n in range(N):
        target_id = target_ids[n]
        for layer in range(L):
            for head in range(H):
                a = last_query[n, layer, head, :]
                top_idx = np.argsort(-a)[: min(top_k, L)]
                for rank, src_pos in enumerate(top_idx):
                    src_pos = int(src_pos)
                    weight = float(a[src_pos])
                    lag_steps = L - src_pos
                    lag_minutes = 10 * lag_steps
                    rows.append({
                        "seed": seed,
                        "target_id": target_id,
                        "layer_idx0": layer,
                        "head_idx0": head,
                        "rank": rank + 1,
                        "source_position_idx0": src_pos,
                        "lag_steps_from_target": lag_steps,
                        "lag_minutes_from_target": lag_minutes,
                        "weight": weight,
                    })
    return rows


def consistency_check(
    seed: int,
    dense_attention: np.ndarray,
    last_query: np.ndarray,
    target_ids_dense: tuple[str, ...],
    target_ids_all: tuple[str, ...],
) -> list[dict[str, Any]]:
    """For each dense case, compare dense[..., L-1, :] vs last_query row.

    dense_attention: [K, L, H, L, L]
    last_query: [N, L, H, L]
    """
    id_to_all_idx = {tid: i for i, tid in enumerate(target_ids_all)}
    rows: list[dict[str, Any]] = []
    for k_pos in range(dense_attention.shape[0]):
        tid = target_ids_dense[k_pos]
        if tid not in id_to_all_idx:
            rows.append({
                "seed": seed,
                "target_id": tid,
                "status": "FAIL",
                "allclose": False,
                "max_abs_difference": float("nan"),
                "layer_idx0": None,
                "head_idx0": None,
                "note": "missing_in_all_test_last_query",
            })
            continue
        all_idx = id_to_all_idx[tid]
        for layer_idx0 in range(dense_attention.shape[1]):
            for head_idx0 in range(dense_attention.shape[2]):
                dense_row = dense_attention[k_pos, layer_idx0, head_idx0, -1, :] 
                lq_row = last_query[all_idx, layer_idx0, head_idx0, :] 
                max_abs = float(np.abs(dense_row - lq_row).max())
                allclose = bool(np.allclose(dense_row, lq_row, atol=1e-5, rtol=1e-5))
                rows.append({
                    "seed": seed,
                    "target_id": tid,
                    "status": "PASS" if allclose else "FAIL",
                    "allclose": allclose,
                    "max_abs_difference": max_abs,
                    "layer_idx0": layer_idx0,
                    "head_idx0": head_idx0,
                    "note": "",
                })
    return rows
