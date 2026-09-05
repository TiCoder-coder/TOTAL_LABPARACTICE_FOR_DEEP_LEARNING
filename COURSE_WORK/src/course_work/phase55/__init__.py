"""Phase 55 - Head Comparison Analysis (HEAD_COMPARISON-v2).

Scientific corrective applied 2026-09-05:
- HEAD_COMPARISON-v1 archived under artifacts/head_comparison/_history/
- source_phase54_version corrected to LAST_QUERY_ATTENTION-v2
- final_lock_sha256 corrected to canonical Phase45 combined lock
- head_to_layer_mean_distance.csv: fixed column-name bug
- head_pair_paired_difference_summary.csv: fixed target_id parsing
- head_behavior_summary.csv: regenerated with non-zero normalized_entropy
- metadata: added raw_last_query_seed42_sha256 as separate field
- metadata: added phase47_canonical_test_population_sha256 as separate field

Descriptive within-seed within-layer head comparison analysis.

Strictly limited to:
- reading frozen Phase 54 canonical CSV/JSON artifacts
- reading frozen Phase 52 raw last-query NPZ for verification only
- emitting O55.1-O55.32 artifacts

Forbidden actions (HARD):
- new attention extraction
- new Test inference
- checkpoint loading
- model.forward
- training
- scaler fitting
- best-seed selection
- ensemble
- best-head selection
- head pruning / ablation
- head clustering core (k-means / spectral / hierarchical)
- unsupervised clustering
- error-conditioned analysis
- regime-conditioned analysis
- worst-case statistical comparison
- cross-seed head matching
- cross-seed same-index semantic alignment assumption
- cross-layer same-index semantic identity assumption
- weighted head diversity score
- post-hoc redundancy threshold
- attention labeled as feature importance
- attention labeled as causal attribution
- Phase 56 implementation
- Phase 57 implementation
"""

from .sources import (
    SEEDS,
    NUM_LAYERS,
    NUM_HEADS,
    N_TEST,
    LOOKBACK,
    HEAD_ORDER,
    LAG_MINUTES,
    PROFILE_SUM_TOL,
    TOP1_SUM_TOL,
    JS_METRIC_BOUNDS,
    COSINE_BOUNDS,
    WASSERSTEIN_BOUNDS_MINUTES,
    MAX_PAIR_COUNT_PER_LAYER,
    EXPECTED_PAIR_COUNT_PER_LAYER,
    PAIR_DELTA_CONVENTION,
    FrozenSources55,
    load_frozen_sources55,
)

from .contract import (
    AnalysisContract55,
    freeze_contract,
    contract_to_dict,
)

__all__ = [
    "SEEDS",
    "NUM_LAYERS",
    "NUM_HEADS",
    "N_TEST",
    "LOOKBACK",
    "HEAD_ORDER",
    "LAG_MINUTES",
    "PROFILE_SUM_TOL",
    "TOP1_SUM_TOL",
    "JS_METRIC_BOUNDS",
    "WASSERSTEIN_BOUNDS_MINUTES",
    "MAX_PAIR_COUNT_PER_LAYER",
    "EXPECTED_PAIR_COUNT_PER_LAYER",
    "PAIR_DELTA_CONVENTION",
    "FrozenSources55",
    "load_frozen_sources55",
    "AnalysisContract55",
    "freeze_contract",
    "contract_to_dict",
]
