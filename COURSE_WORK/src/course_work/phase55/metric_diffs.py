"""Phase 55 - pairwise behavioral metric differences + paired same-target diffs.

Phase 55 §23: Delta R_w = Median(R_{a,w}) - Median(R_{b,w})
Phase 55 §43-§44: paired per-target metric differences
Phase 55 §112-§115: schemas
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sources import FrozenSources55, NUM_HEADS, NUM_LAYERS, SEEDS


# ---------------------------------------------------------------------------
# Median-based metric differences per head pair
# ---------------------------------------------------------------------------

def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


@dataclass
class PairMetricDifference:
    seed: int
    layer_idx0: int
    head_a_idx0: int
    head_b_idx0: int
    metric: str
    value_a: float
    value_b: float
    delta_a_minus_b: float
    abs_delta: float
    status: str


def _get_metric_from_head_summary(
    summary_rows: list[dict[str, str]],
    seed: int,
    layer_idx0: int,
    head_idx0: int,
    metric: str,
) -> float:
    """Look up head-summary median metric. Schema is long: (seed, layer_idx0, head_idx0, metric, ...)."""
    for r in summary_rows:
        try:
            if (int(r["seed"]) == seed
                    and int(r["layer_idx0"]) == layer_idx0
                    and int(r["head_idx0"]) == head_idx0
                    and r.get("metric") == metric):
                val = r.get("median", "")
                if val == "" or val is None:
                    return float("nan")
                return float(val)
        except Exception:
            continue
    return float("nan")


def build_pair_metric_differences(sources: FrozenSources55) -> list[PairMetricDifference]:
    """For each (seed, layer, head_a, head_b), compute signed + abs delta per metric.

    Metric column name in the head-summary CSV (Phase 54 output):
      - median_normalized_entropy
      - median_expected_lag_minutes
      - median_recent_1h_mass
      - median_recent_6h_mass
      - median_recent_12h_mass (if supported)
      - median_recent_24h_mass (if supported)
      - median_lag80_minutes
      - median_top5_mass
    """
    rows: list[PairMetricDifference] = []
    metric_columns = [
        "normalized_entropy",
        "expected_lag_minutes",
        "recent_1h_mass",
        "recent_6h_mass",
        "recent_12h_mass",
        "recent_24h_mass",
        "lag80_steps",
        "top5_mass",
    ]
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            for ha in range(NUM_HEADS):
                for hb in range(ha + 1, NUM_HEADS):
                    for metric in metric_columns:
                        va = _get_metric_from_head_summary(
                            sources.metric_summary_by_head, seed, layer, ha, metric,
                        )
                        vb = _get_metric_from_head_summary(
                            sources.metric_summary_by_head, seed, layer, hb, metric,
                        )
                        if np.isnan(va) or np.isnan(vb):
                            status = "N/A"
                            delta = float("nan")
                            abs_d = float("nan")
                        else:
                            status = "OK"
                            delta = va - vb
                            abs_d = abs(delta)
                        rows.append(PairMetricDifference(
                            seed=seed,
                            layer_idx0=layer,
                            head_a_idx0=ha,
                            head_b_idx0=hb,
                            metric=metric,
                            value_a=va,
                            value_b=vb,
                            delta_a_minus_b=delta,
                            abs_delta=abs_d,
                            status=status,
                        ))
    return rows


# ---------------------------------------------------------------------------
# Paired same-target differences
# ---------------------------------------------------------------------------

@dataclass
class PairedDifferenceSummary:
    seed: int
    layer_idx0: int
    head_a_idx0: int
    head_b_idx0: int
    metric: str
    N: int
    mean_difference: float
    median_difference: float
    sample_sd_difference: float
    p05: float
    p25: float
    p75: float
    p95: float
    fraction_positive: float
    fraction_zero: float
    fraction_negative: float
    status: str


def _target_id_key(raw: str) -> int:
    """Stable integer hash of a target id string (e.g. 'TGT_00016774').

    Required by Phase 55 §96 — paired per-target metric differences need a
    deterministic numeric key per target. TGT_xxxxx strings cannot be
    parsed as int() directly; we hash them to int.
    """
    s = str(raw).strip()
    try:
        return int(s)
    except ValueError:
        # Use a 31-bit deterministic hash for non-numeric target ids
        import hashlib
        h = hashlib.md5(s.encode("utf-8")).digest()
        # First 4 bytes → unsigned int
        return int.from_bytes(h[:4], "big", signed=False)


def _pivot_metric_long(
    metrics_long: list[dict[str, str]],
) -> dict[tuple[int, int, int, int, str], float]:
    """Pivot last_query_metrics_long: (seed, target_id, layer_idx0, head_idx0, metric) -> value."""
    out: dict[tuple[int, int, int, int, str], float] = {}
    for r in metrics_long:
        try:
            seed = int(r["seed"])
            tid = _target_id_key(r["target_id"])
            layer = int(r["layer_idx0"])
            head = int(r["head_idx0"])
        except Exception:
            continue
        for metric in (
            "normalized_entropy",
            "expected_lag_minutes",
            "recent_1h_mass",
            "lag80_minutes",
        ):
            if metric in r and r[metric] not in ("", "nan", "NaN", "None"):
                try:
                    out[(seed, tid, layer, head, metric)] = float(r[metric])
                except Exception:
                    continue
    return out


def build_paired_difference_summaries(
    sources: FrozenSources55,
) -> list[PairedDifferenceSummary]:
    """For each (seed, layer, head_a, head_b, metric): N, mean, median, sd, p05/25/75/95, frac+/-/0."""
    pivoted = _pivot_metric_long(sources.last_query_metrics_long)
    rows: list[PairedDifferenceSummary] = []
    metrics = (
        "normalized_entropy",
        "expected_lag_minutes",
        "recent_1h_mass",
        "lag80_minutes",
    )
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            for ha in range(NUM_HEADS):
                for hb in range(ha + 1, NUM_HEADS):
                    for metric in metrics:
                        # Build per-target difference for same target IDs
                        # Collect all target_ids that have both head values for this seed/layer/metric
                        d_a: dict[int, float] = {}
                        d_b: dict[int, float] = {}
                        for (s, tid, lay, h, m), v in pivoted.items():
                            if s == seed and lay == layer and m == metric:
                                if h == ha:
                                    d_a[tid] = v
                                elif h == hb:
                                    d_b[tid] = v
                        ids_a = set(d_a.keys())
                        ids_b = set(d_b.keys())
                        common = sorted(ids_a & ids_b)
                        if not common:
                            rows.append(PairedDifferenceSummary(
                                seed=seed,
                                layer_idx0=layer,
                                head_a_idx0=ha,
                                head_b_idx0=hb,
                                metric=metric,
                                N=0,
                                mean_difference=float("nan"),
                                median_difference=float("nan"),
                                sample_sd_difference=float("nan"),
                                p05=float("nan"),
                                p25=float("nan"),
                                p75=float("nan"),
                                p95=float("nan"),
                                fraction_positive=float("nan"),
                                fraction_zero=float("nan"),
                                fraction_negative=float("nan"),
                                status="NO_COMMON_TARGETS",
                            ))
                            continue
                        diff = np.array([d_a[t] - d_b[t] for t in common], dtype=np.float64)
                        n = int(diff.size)
                        mean_d = float(diff.mean())
                        median_d = float(np.median(diff))
                        sd_d = float(diff.std(ddof=1)) if n > 1 else 0.0
                        p05 = _quantile(diff, 0.05)
                        p25 = _quantile(diff, 0.25)
                        p75 = _quantile(diff, 0.75)
                        p95 = _quantile(diff, 0.95)
                        # Use tolerance for zero
                        zero_tol = 1e-9
                        n_pos = int(np.sum(diff > zero_tol))
                        n_neg = int(np.sum(diff < -zero_tol))
                        n_zero = int(n - n_pos - n_neg)
                        frac_pos = n_pos / n if n else 0.0
                        frac_zero = n_zero / n if n else 0.0
                        frac_neg = n_neg / n if n else 0.0
                        rows.append(PairedDifferenceSummary(
                            seed=seed,
                            layer_idx0=layer,
                            head_a_idx0=ha,
                            head_b_idx0=hb,
                            metric=metric,
                            N=n,
                            mean_difference=mean_d,
                            median_difference=median_d,
                            sample_sd_difference=sd_d,
                            p05=p05,
                            p25=p25,
                            p75=p75,
                            p95=p95,
                            fraction_positive=frac_pos,
                            fraction_zero=frac_zero,
                            fraction_negative=frac_neg,
                            status="OK",
                        ))
    return rows


# ---------------------------------------------------------------------------
# Top1 distribution distance (TVD + JSD)
# ---------------------------------------------------------------------------

def _pivot_top1_frequency(
    top1_rows: list[dict[str, str]],
) -> dict[tuple[int, int, int], dict[int, float]]:
    """Pivot last_query_top1_lag_frequency: (seed, layer, head) -> {lag_min: fraction}.

    Schema: seed,layer_idx0,head_idx0,lag_steps,lag_minutes,count,fraction,total_targets,status
    """
    out: dict[tuple[int, int, int], dict[int, float]] = {}
    for r in top1_rows:
        try:
            seed = int(r["seed"])
            layer = int(r["layer_idx0"])
            head = int(r["head_idx0"])
            lag = int(r["lag_minutes"])
            freq = float(r["fraction"])
        except Exception:
            continue
        out.setdefault((seed, layer, head), {})[lag] = freq
    return out


def build_top1_distribution_distance(sources: FrozenSources55) -> list[dict[str, Any]]:
    """For each (seed, layer, head_a, head_b): TVD + JSD on top1 lag distribution."""
    pivoted = _pivot_top1_frequency(sources.top1_lag_frequency)
    rows: list[dict[str, Any]] = []
    # Determine global support (union of lag_min keys per (seed, layer))
    # The canonical support is LAG_MINUTES; we map lag_step -> lag_min = (L - step) * 10
    # Top1 frequency uses lag_min directly.
    from .profile_metrics import top1_tvd, jsd_natural_log

    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            for ha in range(NUM_HEADS):
                for hb in range(ha + 1, NUM_HEADS):
                    d_a = pivoted.get((seed, layer, ha), {})
                    d_b = pivoted.get((seed, layer, hb), {})
                    support = sorted(set(d_a.keys()) | set(d_b.keys()))
                    if not support:
                        rows.append({
                            "seed": seed,
                            "layer_idx0": layer,
                            "head_a": ha,
                            "head_b": hb,
                            "tvd": float("nan"),
                            "jsd": float("nan"),
                            "lag_support_count": 0,
                            "status": "MISSING",
                        })
                        continue
                    a_vec = np.array([d_a.get(k, 0.0) for k in support], dtype=np.float64)
                    b_vec = np.array([d_b.get(k, 0.0) for k in support], dtype=np.float64)
                    tvd = top1_tvd(a_vec, b_vec)
                    jsd = jsd_natural_log(a_vec, b_vec)
                    rows.append({
                        "seed": seed,
                        "layer_idx0": layer,
                        "head_a": ha,
                        "head_b": hb,
                        "tvd": tvd,
                        "jsd": jsd,
                        "lag_support_count": len(support),
                        "status": "OK",
                    })
    return rows


# ---------------------------------------------------------------------------
# Wasserstein distance table (for separate audit; main copy is in pair table)
# ---------------------------------------------------------------------------

def build_wasserstein_table(
    sources: FrozenSources55,
    profile_metrics_rows: list[Any],
) -> list[dict[str, Any]]:
    """Reuse profile_metrics rows; subset Wasserstein per pair."""
    out: list[dict[str, Any]] = []
    for pm in profile_metrics_rows:
        out.append({
            "seed": pm.seed,
            "layer_idx0": pm.layer_idx0,
            "head_a": pm.head_a_idx0,
            "head_b": pm.head_b_idx0,
            "wasserstein_minutes": pm.wasserstein_minutes,
            "lag_min_minutes": int(10),
            "lag_max_minutes": int(720),
            "profile_sum_a": 1.0,
            "profile_sum_b": 1.0,
            "status": "OK",
        })
    return out
