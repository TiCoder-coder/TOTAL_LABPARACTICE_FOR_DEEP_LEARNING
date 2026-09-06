# -*- coding: utf-8 -*-
"""Phase 59 — frozen constants and inventories.

All Phase 59 scientific wording is anchored on these constants. No
machine-readable number in the final conclusion may deviate from the
frozen upstream values.
"""

from __future__ import annotations

PHASE_ID = "PHASE_59_FINAL_CONCLUSIONS"
PHASE_NAME = "Final Conclusions"
VERSION = "FINAL_CONCLUSIONS-v2"
SOURCE_PHASE = 58

# Authoritative upstream lineage (post-v2 corrective chain)
SOURCE_VERSION = "FINAL_TABLES-v2"
SOURCE_PHASE54_VERSION = "LAST_QUERY_ATTENTION-v2"
SOURCE_PHASE55_VERSION = "HEAD_COMPARISON-v2"
SOURCE_PHASE56_VERSION = "ERROR_CONDITIONED_ATTENTION-v2"
SOURCE_PHASE57_VERSION = "SEED_STABILITY_ATTENTION-v2"

# Canonical identities (stored separately; do not conflate)
CANONICAL_FINAL_LOCK_SHA256 = "81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec"
CANONICAL_CONFIG_FINGERPRINT_SHA256 = "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"
CANONICAL_TEST_POPULATION_SHA256 = "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"

# Corrective lineage
CORRECTIVE = True
CORRECTIVE_FROM_VERSION = "FINAL_CONCLUSIONS-v1"
CORRECTIVE_AT_UTC = "2026-09-05T14:40:50+00:00"

# Seeds (frozen, single source of truth)
OFFICIAL_SEEDS = [42, 123, 2026]
SEEDS_TUPLE = tuple(OFFICIAL_SEEDS)
SEED_LABEL = "42 / 123 / 2026"
DDOF = 1  # sample SD

# Evidence classes (preserved exactly)
EVIDENCE_DEVELOPMENT = "DEVELOPMENT_EVIDENCE"
EVIDENCE_HELD_OUT_TEST = "HELD_OUT_TEST_EVIDENCE"
EVIDENCE_POST_TEST_DIAGNOSTIC = "POST_TEST_DIAGNOSTIC_EVIDENCE"
EVIDENCE_FROZEN_CONFIG = "FROZEN_CONFIG"
EVIDENCE_CLASSES = (
    EVIDENCE_DEVELOPMENT,
    EVIDENCE_HELD_OUT_TEST,
    EVIDENCE_POST_TEST_DIAGNOSTIC,
)

# Claim levels (LEVEL_4 is unsupported)
LEVEL_0 = "LEVEL_0_DESCRIPTIVE_ONLY"
LEVEL_1 = "LEVEL_1_OBSERVED_ASSOCIATION"
LEVEL_2 = "LEVEL_2_ROBUST_DESCRIPTIVE_PATTERN"
LEVEL_3 = "LEVEL_3_FINAL_HELD_OUT_RESULT"
LEVEL_4 = "LEVEL_4_CAUSAL_UNIVERSAL_EXTERNAL_GENERALIZATION"

CLAIM_LEVELS = (LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4)
SUPPORTED_CLAIM_LEVELS = (LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3)
LEVEL_4_STATUS = "NOT_SUPPORTED"

# Outcome statuses
SUPPORTED = "SUPPORTED"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
MIXED = "MIXED"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_APPLICABLE = "NOT_APPLICABLE"
OUTCOME_STATUSES = (SUPPORTED, PARTIALLY_SUPPORTED, MIXED, INCONCLUSIVE, NOT_APPLICABLE)

# Residual convention (frozen)
RESIDUAL_CONVENTION = (
    "residual = y_true - y_pred; "
    "positive residual = UNDERPREDICTION; "
    "negative residual = OVERPREDICTION"
)

# Attention interpretation (frozen)
ATTENTION_INTERPRETATION = (
    "temporal token allocation, NOT raw-feature importance, NOT causal attribution"
)

# HIGH/LOW cohort interpretation (frozen)
HIGH_LOW_INTERPRETATION = (
    "HIGH_ERROR / LOW_ERROR are Test-relative diagnostic cohorts, "
    "NOT deployment regimes, NOT used for retuning"
)

# Cross-seed head semantic caveat (frozen)
SAME_INDEX_HEAD_CAVEAT = (
    "Same numeric head indices across seeds were NOT assumed semantically equivalent; "
    "Phase 57 used canonical JSD matching within each layer"
)

# Forbidden phrases (must be reviewed when they appear in polished prose)
FORBIDDEN_PHRASES = (
    "prove", "proves", "proved",
    "cause", "causes", "caused", "causal",
    "guarantee", "guarantees",
    "universal", "universally",
    "always",
    "optimal", "optimally",
    "best model", "best seed", "best head",
    "statistically significant",
    "feature importance",
    "deployment-ready", "deployment ready",
    "generalizes", "generalize",
    "explain", "explains", "explained",
)
# Review-only (allowed but must be contextual)
REVIEW_PHRASES = (
    "significant", "best", "general",
)

# Mandatory canonical limitations (L1..L6)
LIMITATION_CATEGORIES = (
    "L1_DATASET",
    "L2_FORECASTING_DESIGN",
    "L3_MODEL_SELECTION_EVALUATION",
    "L4_STATISTICAL",
    "L5_ATTENTION_INTERPRETABILITY",
    "L6_DEPLOYMENT_GENERALIZATION",
)

# Future-work categories (canonical FW1..FW10)
FUTURE_WORK_CATEGORIES = (
    "FW1_MULTI_STEP_FORECASTING",
    "FW2_EXTERNAL_MULTI_HOUSE_VALIDATION",
    "FW3_ADDITIONAL_MODEL_BASELINES",
    "FW4_MORE_SEEDS_UNCERTAINTY",
    "FW5_ATTENTION_ABLATION_FUNCTIONAL_ATTRIBUTION",
    "FW6_FEATURE_LEVEL_ATTRIBUTION",
    "FW7_TEMPORAL_BLOCK_AWARE_STATISTICAL_INFERENCE",
    "FW8_ONLINE_ROLLING_DEPLOYMENT",
    "FW9_UNCERTAINTY_AWARE_FORECASTING",
    "FW10_EFFICIENCY_LATENCY",
)

# Coursework objectives that must close
COURSEWORK_OBJECTIVES = (
    "MULTIVARIATE_TIME_SERIES_REGRESSION",
    "UCI_APPLIANCES_DATASET",
    "TRANSFORMER_ENCODER",
    "LSTM_BASELINE",
    "PERSISTENCE_BASELINE",
    "REGRESSION_METRICS_MAE_RMSE_R2",
    "ATTENTION_MAPS_INTERPRETATION",
    "ERROR_ANALYSIS",
    "SEED_ROBUSTNESS",
    "REPRODUCIBILITY",
)

# Frozen evidence-class label dictionary (Phase 58 FT10 compatible)
EVIDENCE_LABELS = {
    "development": EVIDENCE_DEVELOPMENT,
    "held_out_test": EVIDENCE_HELD_OUT_TEST,
    "post_test_diagnostic": EVIDENCE_POST_TEST_DIAGNOSTIC,
}

# Mandatory canonical conclusion-sentence template sections
CONCLUSION_SECTIONS = (
    "objective_recap",
    "final_test_performance",
    "baseline_comparison",
    "temporal_robustness",
    "error_behavior",
    "temporal_attention",
    "head_diversity",
    "error_conditioned_attention",
    "seed_stability_attention",
    "overall_coursework_goal",
    "limitations",
    "future_work",
    "closing_statement",
)

# Required preflight checks (12)
PREFLIGHT_CHECKS = (
    "PHASE58_APPROVED",
    "PHASE59_READY",
    "FT01_FT10_AVAILABLE",
    "SOURCE_LEDGER_AVAILABLE",
    "CLAIM_TRACEABILITY_AVAILABLE",
    "UPSTREAM_WARNINGS_AVAILABLE",
    "FINAL_LOCK_SHA_AVAILABLE",
    "TEST_POPULATION_SHA_AVAILABLE",
    "SEED_LIST_EXACTLY_42_123_2026",
    "NO_MISSING_CRITICAL_TABLE",
    "CONTRACT_FROZEN_BEFORE_DRAFTING",
    "FINDINGS_LEDGER_AVAILABLE",
)

# Acceptance checklist ids (canonical)
ACCEPTANCE_CHECKLISTS = (
    "RQ_CLOSURE",
    "PERFORMANCE_SCOPE",
    "ROBUSTNESS_DEVELOPMENT_LABEL",
    "ERROR_SCOPE",
    "ATTENTION_SCOPE",
    "ERROR_ATTENTION_SCOPE",
    "SEED_STABILITY_SCOPE",
    "LIMITATION_COVERAGE",
    "FUTURE_WORK_FUTURE",
    "LANGUAGE_FORBIDDEN",
    "NUMERIC_TRACEABILITY",
    "SENTENCE_LEVEL_AUDIT",
    "FINAL_PROJECT_CLOSURE",
)

# Mapping forbidden actions for discrepancy taxonomy
DISCREPANCY_TAXONOMY = (
    "PHASE58_NOT_APPROVED",
    "PHASE59_HANDOFF_NOT_READY",
    "FINAL_TABLE_MISSING",
    "CLAIM_TRACEABILITY_MISSING",
    "SOURCE_LEDGER_MISSING",
    "FINAL_LOCK_MISMATCH",
    "TEST_POPULATION_MISMATCH",
    "SEED_SET_MISMATCH",
    "UNTRACEABLE_CLAIM",
    "UNTRACEABLE_NUMBER",
    "WRONG_EVIDENCE_CLASS",
    "DEVELOPMENT_RESULT_MISLABELED_TEST",
    "TEST_RESULT_OVERGENERALIZED",
    "THREE_SEED_SUMMARY_MISLABELED_ENSEMBLE",
    "BEST_SEED_SELECTED",
    "BEST_HEAD_SELECTED",
    "OPTIMAL_HYPERPARAMETER_CLAIM",
    "UNSUPPORTED_TRANSFORMER_SUPERIORITY",
    "PERSISTENCE_RESULT_OMITTED",
    "LSTM_RESULT_OMITTED",
    "MIXED_RESULT_SPUN_AS_POSITIVE",
    "R2_MISINTERPRETED",
    "RESIDUAL_SIGN_DRIFT",
    "REGIME_CAUSAL_CLAIM",
    "ERROR_COHORT_MISLABELED_DEPLOYMENT_REGIME",
    "WORST_CASE_USED_TO_DISCARD_DATA",
    "ATTENTION_MISLABELED_FEATURE_IMPORTANCE",
    "ATTENTION_CAUSAL_CLAIM",
    "ATTENTION_FULL_EXPLANATION_CLAIM",
    "SAME_INDEX_HEAD_SEMANTIC_CLAIM",
    "MATCHING_AMBIGUITY_HIDDEN",
    "SEED_VARIABILITY_HIDDEN",
    "STATISTICAL_SIGNIFICANCE_CLAIM_WITHOUT_TEST",
    "CONFIDENCE_INTERVAL_CLAIM_WITHOUT_SOURCE",
    "EXTERNAL_GENERALIZATION_CLAIM",
    "MULTI_HOUSE_GENERALIZATION_CLAIM",
    "DEPLOYMENT_READY_CLAIM",
    "ENERGY_SAVING_IMPACT_CLAIM",
    "NEW_ANALYSIS_PERFORMED",
    "NEW_METRIC_ADDED",
    "NEW_STATISTICAL_TEST_ADDED",
    "POST_TEST_RETUNING",
    "NEW_TEST_INFERENCE",
    "NEW_ATTENTION_EXTRACTION",
    "FUTURE_WORK_WRITTEN_AS_COMPLETED",
    "LIMITATION_OMITTED",
    "UPSTREAM_WARNING_OMITTED",
    "NUMERIC_ROUNDING_MISMATCH",
    "OTHER",
)

# Display rounding (mirroring Phase 58)
DISPLAY_PRECISION = {
    "Wh": 2,
    "R2": 3,
    "dimensionless": 3,
    "minutes": 1,
    "percent": 1,
    "count": 0,
}

# Frozen research-question list (RQ1..RQ10)
RESEARCH_QUESTIONS = (
    ("RQ1", "Final Transformer dự báo Appliances energy tốt đến mức nào trên held-out chronological Test?",
     "FT02", EVIDENCE_HELD_OUT_TEST),
    ("RQ2", "Final Transformer so với Persistence và tuned LSTM như thế nào?",
     "FT02", EVIDENCE_HELD_OUT_TEST),
    ("RQ3", "Performance có robust theo temporal rolling-origin evaluation không?",
     "FT03", EVIDENCE_DEVELOPMENT),
    ("RQ4", "Forecast error tập trung ở những kiểu tình huống/regime nào?",
     "FT05", EVIDENCE_POST_TEST_DIAGNOSTIC),
    ("RQ5", "Last-query attention tập trung vào các temporal lags nào?",
     "FT06", EVIDENCE_POST_TEST_DIAGNOSTIC),
    ("RQ6", "Các attention heads có học temporal allocation behavior khác nhau không?",
     "FT07", EVIDENCE_POST_TEST_DIAGNOSTIC),
    ("RQ7", "Attention behavior có co-vary với realized forecast error không?",
     "FT08", EVIDENCE_POST_TEST_DIAGNOSTIC),
    ("RQ8", "Các attention findings có ổn định qua ba final seeds không?",
     "FT09", EVIDENCE_POST_TEST_DIAGNOSTIC),
    ("RQ9", "Những điều gì có thể và không thể kết luận từ attention analysis?",
     "FT10", EVIDENCE_POST_TEST_DIAGNOSTIC),
    ("RQ10", "Những limitation nào giới hạn external/general scientific interpretation?",
     "FA12", EVIDENCE_POST_TEST_DIAGNOSTIC),
)

# RQ categorization topics for outcome matrix
OUTCOME_TOPICS = (
    "Final Test performance",
    "Transformer vs Persistence",
    "Transformer vs LSTM",
    "Temporal robustness",
    "Error concentration",
    "Regime difficulty",
    "Recent-history attention",
    "Long-history attention",
    "Head diversity",
    "Error-attention association",
    "Attention seed stability",
    "Head semantic stability",
)
