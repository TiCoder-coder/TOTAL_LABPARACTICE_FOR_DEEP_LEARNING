"""
Analysis phases container.

This package contains all post-training analysis phases:
- prediction: Prediction analysis (Phase 48)
- residual: Residual analysis (Phase 49)
- error_regime: Error by regime analysis (Phase 50)
- worst_error: Worst error analysis (Phase 51)
- attention_extraction: Attention extraction (Phase 52)
- attention_heatmaps: Attention heatmaps (Phase 53)
- last_query_attention: Last query attention (Phase 54)
- head_comparison: Head comparison (Phase 55)
- error_conditioned_attention: Error-conditioned attention (Phase 56)
- seed_stability_attention: Seed stability attention (Phase 57)
- final_tables: Final tables (Phase 58)
- final_conclusions: Final conclusions (Phase 59)

Each phase is read-only over frozen upstream artifacts.
"""

__all__ = [
    # Analysis phases
    "prediction_analysis",
    "residual_analysis",
    "error_regime_analysis",
    "worst_error_analysis",
    "attention_extraction",
    "attention_heatmaps",
    "last_query_attention",
    "head_comparison",
    "error_conditioned_attention",
    "seed_stability_attention",
    "final_tables",
    "final_conclusions",
]
