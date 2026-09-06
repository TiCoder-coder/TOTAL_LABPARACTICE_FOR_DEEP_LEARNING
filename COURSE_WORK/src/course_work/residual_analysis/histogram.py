from __future__ import annotations

from typing import Any

import numpy as np


PHASE49_HISTOGRAM_BINS = 50


def compute_common_bin_edges(
    residuals_by_seed: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute 50-common-bin edges shared across all seeds.

    Range = [min(all_seeds), max(all_seeds)].
    Edges = np.linspace(min, max, 51) -> 50 bins.
    """
    all_residuals = np.concatenate(
        [residuals_by_seed[s] for s in sorted(residuals_by_seed.keys())],
        axis=0,
    )
    r_min = float(np.min(all_residuals))
    r_max = float(np.max(all_residuals))
    edges = np.linspace(r_min, r_max, PHASE49_HISTOGRAM_BINS + 1, dtype=np.float64)
    return edges


def compute_histogram_rows(
    residuals_by_seed: dict[str, np.ndarray],
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    n_bins = edges.size - 1
    if n_bins != PHASE49_HISTOGRAM_BINS:
        raise ValueError(
            f"Phase49-C histogram expects {PHASE49_HISTOGRAM_BINS} bins; got {n_bins}"
        )

    rows: list[dict[str, Any]] = []
    for seed in sorted(residuals_by_seed.keys()):
        residuals = residuals_by_seed[seed]
        counts, _ = np.histogram(residuals, bins=edges)
        total = int(np.sum(counts))
        if total != residuals.size:
            raise ValueError(
                f"Histogram counts sum to {total} but seed {seed} has {residuals.size} residuals"
            )
        for bin_index in range(n_bins):
            left = float(edges[bin_index])
            right = float(edges[bin_index + 1])
            center = 0.5 * (left + right)
            count = int(counts[bin_index])
            fraction = float(count) / float(total)
            rows.append(
                {
                    "bin_index": bin_index,
                    "bin_left": left,
                    "bin_right": right,
                    "bin_center": center,
                    "seed": seed,
                    "count": count,
                    "fraction": fraction,
                }
            )
    return rows


def histogram_invariants(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Histogram rows empty")
    bin_indices = {r["bin_index"] for r in rows}
    edges_per_bin: dict[int, tuple[float, float]] = {}
    for r in rows:
        i = r["bin_index"]
        edge_pair = (r["bin_left"], r["bin_right"])
        if i in edges_per_bin:
            prev = edges_per_bin[i]
            if prev != edge_pair:
                return {
                    "shared_bin_edges_across_seeds": False,
                    "expected_bin_count": PHASE49_HISTOGRAM_BINS,
                    "actual_bin_count": len(bin_indices),
                }
        else:
            edges_per_bin[i] = edge_pair

    seeds = sorted({r["seed"] for r in rows})
    per_seed_total: dict[str, int] = {}
    per_seed_fraction_total: dict[str, float] = {}
    for s in seeds:
        s_rows = [r for r in rows if r["seed"] == s]
        per_seed_total[s] = sum(r["count"] for r in s_rows)
        per_seed_fraction_total[s] = sum(r["fraction"] for r in s_rows)

    return {
        "shared_bin_edges_across_seeds": True,
        "expected_bin_count": PHASE49_HISTOGRAM_BINS,
        "actual_bin_count": len(bin_indices),
        "n_seeds": len(seeds),
        "per_seed_count_sum": per_seed_total,
        "per_seed_fraction_sum": per_seed_fraction_total,
        "all_seeds_count_to_2961": all(v == 2961 for v in per_seed_total.values()),
        "all_seeds_fraction_to_1": all(
            abs(v - 1.0) < 1e-12 for v in per_seed_fraction_total.values()
        ),
        "n_rows": len(rows),
        "n_rows_expected": PHASE49_HISTOGRAM_BINS * len(seeds),
    }


HISTOGRAM_FIELDS: list[str] = [
    "bin_index",
    "bin_left",
    "bin_right",
    "bin_center",
    "seed",
    "count",
    "fraction",
]
