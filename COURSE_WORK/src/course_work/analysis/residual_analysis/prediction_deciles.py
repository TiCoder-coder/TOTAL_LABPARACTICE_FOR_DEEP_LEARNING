from __future__ import annotations

from typing import Any

import numpy as np


PHASE49_DECILE_COUNT = 10


def build_prediction_decile_edges(
    y_pred: np.ndarray,
    decile_count: int = PHASE49_DECILE_COUNT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build decile edges from y_pred via np.quantile at linspace(0, 1, decile_count + 1).

    Returns (edges, meta). meta documents duplicate-edge conditions.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, decile_count + 1)
    edges = np.quantile(y_pred, quantiles)
    unique_edges = np.unique(edges)
    duplicate_edges_present = bool(unique_edges.size < edges.size)
    meta = {
        "decile_count": decile_count,
        "n_unique_edges": int(unique_edges.size),
        "n_edges_requested": int(edges.size),
        "duplicate_edges_present": duplicate_edges_present,
        "deterministic_quantile_method": "np.quantile(y_pred, np.linspace(0,1,11))",
        "phase50_regime_use": False,
        "use": "PREDICTION-DECILE DESCRIPTIVE DIAGNOSTICS ONLY",
    }
    return edges, meta


def assign_deciles(
    y_pred: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Assign each row to a decile index 1..10.

    Uses np.searchsorted with 'right' on edges[1:-1] as breakpoints, ensuring
    deterministic assignment that handles duplicate edges explicitly.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    inner_edges = edges[1:-1]
    assignments = np.searchsorted(inner_edges, y_pred, side="right") + 1
    return np.clip(assignments, 1, len(edges) - 1)


def compute_decile_diagnostics(
    residuals: np.ndarray,
    signs: list[str],
    y_pred: np.ndarray,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute per-decile diagnostics.

    Returns (rows, meta).
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    edges, meta = build_prediction_decile_edges(y_pred)
    assignments = assign_deciles(y_pred, edges)

    rows: list[dict[str, Any]] = []
    counts_per_decile: dict[int, int] = {}
    for d in range(1, PHASE49_DECILE_COUNT + 1):
        mask = assignments == d
        n_d = int(mask.sum())
        counts_per_decile[d] = n_d
        if n_d == 0:
            rows.append(
                {
                    "seed": seed,
                    "decile": d,
                    "N": 0,
                    "pred_min": float("nan"),
                    "pred_max": float("nan"),
                    "mean_prediction": float("nan"),
                    "mean_residual": float("nan"),
                    "median_residual": float("nan"),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "underprediction_fraction": float("nan"),
                    "overprediction_fraction": float("nan"),
                    "exact_fraction": float("nan"),
                    "edge_left": float(edges[d - 1]),
                    "edge_right": float(edges[d]),
                    "status": "EMPTY_DECILE",
                }
            )
            continue
        r_in = residuals[mask]
        p_in = y_pred[mask]
        s_in = [signs[i] for i in range(len(signs)) if mask[i]]
        n_under = sum(1 for s in s_in if s == "UNDERPREDICTION")
        n_over = sum(1 for s in s_in if s == "OVERPREDICTION")
        n_exact = sum(1 for s in s_in if s == "EXACT")
        rows.append(
            {
                "seed": seed,
                "decile": d,
                "N": n_d,
                "pred_min": float(np.min(p_in)),
                "pred_max": float(np.max(p_in)),
                "mean_prediction": float(np.mean(p_in)),
                "mean_residual": float(np.mean(r_in)),
                "median_residual": float(np.median(r_in)),
                "mae": float(np.mean(np.abs(r_in))),
                "rmse": float(np.sqrt(np.mean(r_in * r_in))),
                "underprediction_fraction": n_under / n_d,
                "overprediction_fraction": n_over / n_d,
                "exact_fraction": n_exact / n_d,
                "edge_left": float(edges[d - 1]),
                "edge_right": float(edges[d]),
                "status": "PASS",
            }
        )
    meta["counts_per_decile"] = counts_per_decile
    meta["total_rows_assigned"] = int(sum(counts_per_decile.values()))
    return rows, meta
