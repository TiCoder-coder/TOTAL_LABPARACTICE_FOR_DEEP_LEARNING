"""Phase 54 — analysis contract freeze.

Defines `AnalysisContract` dataclass and `build_analysis_contract()` which
encodes all frozen analysis parameters (formula, epsilon, windows, etc.)
and writes `last_query_attention_contract.json`.

Contract must be frozen BEFORE any numerical computation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .sources import (
    CADENCE_MINUTES,
    COVERAGE_LEVELS,
    EPSILON_H,
    EPSILON_NEG,
    LAST_QUERY_DEFINITION,
    LOOKBACK,
    N_TEST,
    NON_OVERLAP_BIN_LABELS,
    NON_OVERLAP_BINS,
    NUM_HEADS,
    NUM_LAYERS,
    PROB_MAX_TOL,
    RAW_AXIS_ORDER,
    RECENT_WINDOWS_HOURS,
    RECENT_WINDOWS_STEPS,
    SEEDS,
    SUM_TOL,
    FrozenSources54,
)
from .writers import write_json_atomic


@dataclass
class AnalysisContract:
    version: str
    phase: int
    phase_name: str
    raw_source: str
    last_query_definition: str
    raw_axis_order: list[str]
    n_test: int
    lookback_steps: int
    num_layers: int
    num_heads: int
    seeds: list[int]
    pooling: str
    last_query_directly_corresponds_to_pooled_token: bool
    lag_steps_formula: str
    lag_minutes_formula: str
    cadence_minutes: int
    entropy_epsilon: float
    entropy_formula: str
    normalized_entropy_formula: str
    effective_source_count_formula: str
    expected_lag_steps_formula: str
    expected_lag_minutes_formula: str
    lag_sd_steps_formula: str
    top1_tie_rule: str
    top5_k: int
    recent_windows_steps: dict[str, int]
    recent_windows_hours: dict[str, int]
    recent_truncation_rule: str
    coverage_levels: list[float]
    coverage_radius_definition: str
    non_overlap_bins: list[tuple[int, int]]
    non_overlap_bin_labels: list[str]
    sum_tolerance: float
    prob_max_tol: float
    neg_tolerance: float
    metric_summary_quantiles: list[float]
    profile_quantiles: list[float]
    sample_sd_ddof: int
    head_order: str
    report_case_selection_rule: str
    forbidden_actions: list[str]
    figure_rules: list[str]
    created_at_utc: str = ""
    contract_sha256: str = ""


def build_analysis_contract(src: FrozenSources54) -> AnalysisContract:
    """Build the AnalysisContract from frozen sources."""
    contract = AnalysisContract(
        version="LAST_QUERY_ATTENTION-v1",
        phase=54,
        phase_name="Last-Query Attention Analysis",
        raw_source="artifacts/attention_extraction/raw/last_query_attention_seed{42,123,2026}.npz",
        last_query_definition=LAST_QUERY_DEFINITION,
        raw_axis_order=list(RAW_AXIS_ORDER),
        n_test=N_TEST,
        lookback_steps=LOOKBACK,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        seeds=list(SEEDS),
        pooling=src.pooling or "LAST_STEP",
        last_query_directly_corresponds_to_pooled_token=bool(
            src.last_query_directly_corresponds_to_pooled_token
        ),
        lag_steps_formula="H + (L - 1 - p), H = 1",
        lag_minutes_formula=f"{CADENCE_MINUTES} * lag_steps_p",
        cadence_minutes=CADENCE_MINUTES,
        entropy_epsilon=EPSILON_H,
        entropy_formula="H = -sum_p a_p * log(a_p + epsilon_H)",
        normalized_entropy_formula="H_norm = H / log(L)",
        effective_source_count_formula="N_eff = exp(H)",
        expected_lag_steps_formula="E[LagSteps] = sum_p a_p * lag_steps_p",
        expected_lag_minutes_formula=f"E[LagMinutes] = {CADENCE_MINUTES} * E[LagSteps]",
        lag_sd_steps_formula="SD(Lag) = sqrt(sum_p a_p * (lag_steps_p - E[LagSteps])^2)",
        top1_tie_rule="NEWEST_SOURCE",
        top5_k=5,
        recent_windows_steps=dict(RECENT_WINDOWS_STEPS),
        recent_windows_hours=dict(RECENT_WINDOWS_HOURS),
        recent_truncation_rule="effective_steps = min(requested_steps, L)",
        coverage_levels=list(COVERAGE_LEVELS),
        coverage_radius_definition=(
            "LagX = min{k: cumulative_recency_mass(k) >= X}; "
            "recency order = newest->oldest (position L-1 -> position 0)"
        ),
        non_overlap_bins=[tuple(b) for b in NON_OVERLAP_BINS],
        non_overlap_bin_labels=list(NON_OVERLAP_BIN_LABELS),
        sum_tolerance=SUM_TOL,
        prob_max_tol=PROB_MAX_TOL,
        neg_tolerance=EPSILON_NEG,
        metric_summary_quantiles=[0.05, 0.25, 0.50, 0.75, 0.95],
        profile_quantiles=[0.05, 0.25, 0.50, 0.75, 0.95],
        sample_sd_ddof=1,
        head_order="ARCHITECTURAL_INDEX",
        report_case_selection_rule="PHASE51_W2_SHARED_RANK_1_TO_5",
        forbidden_actions=[
            "new_attention_extraction",
            "new_test_inference",
            "checkpoint_loading",
            "model_forward",
            "training",
            "optimizer",
            "backward",
            "scaler_fit",
            "best_seed_selection",
            "ensemble",
            "best_head_selection",
            "head_ranking",
            "head_clustering",
            "head_ablation",
            "cross_seed_head_matching",
            "error_conditioned_analysis",
            "regime_conditioned_analysis",
            "feature_importance_claim",
            "causal_attribution_claim",
            "png_numeric_read",
            "phase55_implementation",
            "phase56_implementation",
            "phase57_implementation",
        ],
        figure_rules=[
            "no_smoothing",
            "no_interpolation",
            "no_renormalization",
            "common_y_axis_within_seed_layer",
            "x_axis_lag_minutes_recency_order",
            "no_head_sort_by_metric",
            "no_per_head_max_normalization",
        ],
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        contract_sha256="",
    )
    # Compute contract SHA over the canonical payload (excluding the sha field itself)
    payload = asdict(contract)
    payload.pop("contract_sha256", None)
    payload_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    contract.contract_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    return contract


def write_analysis_contract(contract: AnalysisContract, out_dir: Path) -> Path:
    """Write contract JSON to out_dir and return the path."""
    fp = out_dir / "last_query_attention_contract.json"
    payload = asdict(contract)
    write_json_atomic(fp, payload)
    return fp
