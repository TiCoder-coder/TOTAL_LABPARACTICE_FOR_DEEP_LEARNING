"""Phase 55 - analysis contract (frozen before any numerical result).

Per canonical Phase 55 §107 and §174: contract is locked before any
pairwise metric is computed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .sources import (
    SEEDS,
    NUM_LAYERS,
    NUM_HEADS,
    N_TEST,
    LOOKBACK,
    HEAD_ORDER,
    EXPECTED_PAIR_COUNT_PER_LAYER,
    PAIR_DELTA_CONVENTION,
    JS_METRIC_BOUNDS,
    COSINE_BOUNDS,
    WASSERSTEIN_BOUNDS_MINUTES,
    PROFILE_SUM_TOL,
    TOP1_SUM_TOL,
)


@dataclass
class AnalysisContract55:
    """Frozen Phase 55 head-comparison analysis contract."""

    # Scope
    phase: int = 55
    phase_name: str = "Head Comparison Analysis"
    version: str = "HEAD_COMPARISON-v2"
    primary_comparison_scope: str = "WITHIN_SEED_WITHIN_LAYER_ACROSS_HEADS"
    head_order: str = HEAD_ORDER
    pair_count_per_layer: int = EXPECTED_PAIR_COUNT_PER_LAYER
    total_pairs: int = EXPECTED_PAIR_COUNT_PER_LAYER * len(SEEDS) * NUM_LAYERS

    # Population
    seeds: tuple[int, ...] = SEEDS
    n_layers: int = NUM_LAYERS
    n_heads: int = NUM_HEADS
    lookback: int = LOOKBACK
    n_test: int = N_TEST

    # Pairwise profile metrics (frozen before computation)
    pairwise_profile_metrics: tuple[str, ...] = (
        "pearson", "spearman", "cosine", "jsd", "l1", "l2", "wasserstein_minutes",
    )

    # Pairwise behavioral differences
    pairwise_metric_differences: tuple[str, ...] = (
        "median_normalized_entropy",
        "median_expected_lag_minutes",
        "median_recent_1h_mass",
        "median_recent_6h_mass",
        "median_recent_12h_mass",
        "median_recent_24h_mass",
        "median_lag80_minutes",
        "median_top5_mass",
    )

    # Paired same-target differences
    paired_metrics: tuple[str, ...] = (
        "normalized_entropy",
        "expected_lag_minutes",
        "recent_1h_mass",
        "lag80_minutes",
    )

    # Top1 distribution
    top1_distribution_metrics: tuple[str, ...] = ("tvd", "jsd")
    top1_tie_rule: str = "NEWEST_SOURCE"

    # Aggregations
    center_metric: str = "median"  # canonical Phase 55 §69
    pair_delta_convention: str = PAIR_DELTA_CONVENTION  # Delta = A - B

    # Bound guards
    js_metric_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in JS_METRIC_BOUNDS.items()}
    )
    cosine_bounds: tuple[float, float] = COSINE_BOUNDS
    wasserstein_bounds_minutes: tuple[float, float] = WASSERSTEIN_BOUNDS_MINUTES

    # Tolerances
    profile_sum_tol: float = PROFILE_SUM_TOL
    top1_sum_tol: float = TOP1_SUM_TOL

    # Forbidden actions (HARD)
    forbidden_actions: tuple[str, ...] = (
        "training",
        "fine_tune",
        "optimizer_step",
        "backward",
        "model_train",
        "scaler_fit",
        "new_test_inference",
        "new_attention_extraction",
        "checkpoint_loading",
        "model_forward",
        "return_attention",
        "extract_attention",
        "materialize_phase52",
        "best_seed_selection",
        "ensemble",
        "best_head_selection",
        "head_pruning",
        "head_ablation",
        "head_ranking",
        "head_clustering_core",
        "unsupervised_clustering",
        "kmeans",
        "spectral_clustering",
        "hierarchical_clustering",
        "weighted_head_diversity_score",
        "post_hoc_redundancy_threshold",
        "cross_seed_head_matching",
        "cross_seed_semantic_alignment_assumed",
        "cross_layer_same_index_identity_assumed",
        "error_conditioning",
        "regime_conditioning",
        "worst_case_statistical_comparison",
        "iid_significance_test_on_dependent_targets",
        "feature_importance_claim",
        "causal_claim",
        "phase56_implementation",
        "phase57_implementation",
        "png_numeric_read",
        "notebook_modification",
    )

    # Source pointers
    source_phase54_version: str = "LAST_QUERY_ATTENTION-v2"
    source_phase52_version: str = "ATTENTION_EXTRACTION-v1"

    def freeze(self, p54_signoff_sha: str, handoff_p55_sha: str) -> "AnalysisContract55":
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "phase_name": self.phase_name,
            "version": self.version,
            "primary_comparison_scope": self.primary_comparison_scope,
            "head_order": self.head_order,
            "pair_count_per_layer": self.pair_count_per_layer,
            "total_pairs": self.total_pairs,
            "seeds": list(self.seeds),
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "lookback": self.lookback,
            "n_test": self.n_test,
            "pairwise_profile_metrics": list(self.pairwise_profile_metrics),
            "pairwise_metric_differences": list(self.pairwise_metric_differences),
            "paired_metrics": list(self.paired_metrics),
            "top1_distribution_metrics": list(self.top1_distribution_metrics),
            "top1_tie_rule": self.top1_tie_rule,
            "center_metric": self.center_metric,
            "pair_delta_convention": self.pair_delta_convention,
            "js_metric_bounds": {k: list(v) for k, v in self.js_metric_bounds.items()},
            "cosine_bounds": list(self.cosine_bounds),
            "wasserstein_bounds_minutes": list(self.wasserstein_bounds_minutes),
            "profile_sum_tol": self.profile_sum_tol,
            "top1_sum_tol": self.top1_sum_tol,
            "forbidden_actions": list(self.forbidden_actions),
            "source_phase54_version": self.source_phase54_version,
            "source_phase52_version": self.source_phase52_version,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


def freeze_contract(p54_signoff_sha: str = "", handoff_p55_sha: str = "") -> AnalysisContract55:
    """Freeze the Phase 55 head-comparison analysis contract."""
    contract = AnalysisContract55()
    contract.freeze(p54_signoff_sha, handoff_p55_sha)
    return contract


def contract_to_dict(contract: AnalysisContract55, p54_signoff_sha: str, handoff_p55_sha: str) -> dict[str, Any]:
    d = contract.to_dict()
    d["p54_signoff_sha256"] = p54_signoff_sha
    d["handoff_p55_sha256"] = handoff_p55_sha
    # Compute contract SHA256 over canonical serialization
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    d["contract_sha256"] = hashlib.sha256(blob).hexdigest()
    return d
