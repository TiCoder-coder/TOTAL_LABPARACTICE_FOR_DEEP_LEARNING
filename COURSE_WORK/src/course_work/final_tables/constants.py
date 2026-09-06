# -*- coding: utf-8 -*-
"""Phase 58 - constants and frozen inventories.

Per `docs/plan-doc/plan_detail_for_each_phase/Phase_58_Final_tables.md`.

Frozen before any rendering:

* Main table inventory: FT01..FT10.
* Appendix table inventory: FA01..FA12.
* Model-order contract and seed-order contract.
* Display precision per unit family.
* Source-of-truth precedence (canonical phase 58 plan §7).
"""

from __future__ import annotations

PHASE_ID = 58
PHASE_NAME = "Final Tables"
VERSION = "FINAL_TABLES-v2"
PREVIOUS_VERSION = "FINAL_TABLES-v1"
PREVIOUS_VERSION_ARCHIVED_PATH = (
    "artifacts/final_tables/_history/FINAL_TABLES-v1_ARCHIVED_20260905T135340396+0000"
)
CORRECTIVE = True
CORRECTIVE_AT_UTC = "2026-09-05T13:53:40Z"

# Canonical identity SHAs (separated, NOT conflated)
CANONICAL_FINAL_LOCK_SHA256 = (
    "81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec"  # combined Phase45 lock
)
CANONICAL_CONFIG_FINGERPRINT_SHA256 = (
    "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"  # config fingerprint only
)
CANONICAL_PHASE47_TEST_POPULATION_SHA256 = (
    "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
)

# Upstream authoritative versions
SOURCE_PHASE54_VERSION = "LAST_QUERY_ATTENTION-v2"
SOURCE_PHASE55_VERSION = "HEAD_COMPARISON-v2"
SOURCE_PHASE56_VERSION = "ERROR_CONDITIONED_ATTENTION-v2"
SOURCE_PHASE57_VERSION = "SEED_STABILITY_ATTENTION-v2"

OFFICIAL_SEEDS = [42, 123, 2026]

MAIN_TABLE_IDS = [
    "FT01", "FT02", "FT03", "FT04", "FT05",
    "FT06", "FT07", "FT08", "FT09", "FT10",
]
APPENDIX_TABLE_IDS = [
    "FA01", "FA02", "FA03", "FA04", "FA05", "FA06",
    "FA07", "FA08", "FA09", "FA10", "FA11", "FA12",
]
ALL_TABLE_IDS = MAIN_TABLE_IDS + APPENDIX_TABLE_IDS

MODEL_ORDER = [
    "Persistence Baseline",
    "Tuned LSTM Baseline",
    "Final Transformer \u2014 Seed 42",
    "Final Transformer \u2014 Seed 123",
    "Final Transformer \u2014 Seed 2026",
    "Final Transformer \u2014 Three-Seed Summary",
]

# Internal model-name mapping (canonical Phase 47/45 model IDs)
INTERNAL_MODEL_MAP = {
    "TRANSFORMER_SEED42": ("Final Transformer \u2014 Seed 42", 42),
    "TRANSFORMER_SEED123": ("Final Transformer \u2014 Seed 123", 123),
    "TRANSFORMER_SEED2026": ("Final Transformer \u2014 Seed 2026", 2026),
    "PERSISTENCE_LAST_VALUE": ("Persistence Baseline", None),
    "LSTM_TUNED_WINNER": ("Tuned LSTM Baseline", None),
    "TRANSFORMER_THREE_SEED_SUMMARY": ("Final Transformer \u2014 Three-Seed Summary", None),
}

# Display precision (per Phase 58 §30)
DISPLAY_PRECISION = {
    "Wh": 2,            # MAE / RMSE / residual / absolute_error
    "r2": 3,            # R^2
    "dimensionless": 3, # JSD / cosine / Spearman / Cliff's delta
    "minutes": 1,       # Wasserstein-1 / expected_lag
    "percent": 1,       # cohort share
    "count": 0,         # integer
}

MEAN_SD_DDOF = 1  # three-seed sample SD uses ddof=1
MISSING_LABEL = "N/A"
NEGATIVE_ZERO_FIX = True
NO_BEST_VALUE_HIGHLIGHT = True
NO_ENSEMBLE = True

# Evidence-class labels
EVIDENCE_FROZEN_CONFIG = "METHOD / LOCKED_CONFIG"
EVIDENCE_HELD_OUT_TEST = "HELD_OUT_TEST_EVIDENCE"
EVIDENCE_POST_TEST_DIAGNOSTIC = "POST_TEST_DIAGNOSTIC_EVIDENCE"
EVIDENCE_DEVELOPMENT = "DEVELOPMENT_EVIDENCE"
EVIDENCE_EVIDENCE_LIMITATION = "EVIDENCE_AND_LIMITATION"

# Frozen source-authority precedence
SOURCE_AUTHORITY_PRECEDENCE = [
    ("FT01", "Phase 45 final model lock"),
    ("FT02", "Phase 47 final Test metrics"),
    ("FT03", "Phase 44 rolling-origin robustness"),
    ("FT04", "Phase 48 prediction spread + Phase 49 residuals"),
    ("FT05", "Phase 50 error regimes + Phase 51 worst cases"),
    ("FT06", "Phase 54 last-query attention"),
    ("FT07", "Phase 55 head comparison diversity"),
    ("FT08", "Phase 56 layer head-mean error-conditioned attention"),
    ("FT09", "Phase 57 layer head-mean stability + matching"),
    ("FT10", "upstream-supported claim trace"),
]

# Frozen figure inventory (Phase 53/54/55/56/57 PNG references).
FIGURE_INVENTORY = [
    {"figure_id": "FF01", "source_phase": 48,
     "source_artifact": "artifacts/prediction_analysis/figures/"},
    {"figure_id": "FF02", "source_phase": 49,
     "source_artifact": "artifacts/residual_analysis/figures/"},
    {"figure_id": "FF03", "source_phase": 50,
     "source_artifact": "artifacts/error_by_regime/figures/"},
    {"figure_id": "FF04", "source_phase": 53,
     "source_artifact": "artifacts/attention_heatmaps/figures/"},
    {"figure_id": "FF05", "source_phase": 54,
     "source_artifact": "artifacts/last_query_attention/figures/"},
    {"figure_id": "FF06", "source_phase": 55,
     "source_artifact": "artifacts/head_comparison/figures/"},
    {"figure_id": "FF07", "source_phase": 56,
     "source_artifact": "artifacts/error_conditioned_attention/figures/"},
    {"figure_id": "FF08", "source_phase": 57,
     "source_artifact": "artifacts/seed_stability_attention/figures/"},
]
