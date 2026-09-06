"""Phase 48 — constants and frozen contracts."""
from __future__ import annotations

from pathlib import Path

PHASE_NUM = 48
PHASE_NAME = "Prediction Analysis"
OUTPUT_VERSION = "PREDICTION_ANALYSIS-v1"

LOCKED_TRANSFORMER_SEEDS = [42, 123, 2026]
PERSISTENCE_LABEL = "PERSISTENCE"

LOCKED_FINAL_TEST_VERSION = "FINAL_TEST_EVAL-v1"
LOCKED_POPULATION_ID = "FINAL_TEST_POP-v1"

LAG_RANGE = list(range(-6, 7))
LAG_SIGN_CONVENTION = (
    "lag_k_positive_means_prediction_compared_to_truth_shifted_k_future_steps"
)

ACF_REGISTERED_LAGS = [1, 6, 12, 36, 72, 144]

PEAK_TIMING_WINDOW_STEPS = 1

ROLLING_WINDOW = 144
ROLLING_WINDOW_LABEL = "24h_at_10min_cadence"

TOP_DISAGREEMENT_K = 20

SEED_STD_DDOF = 1

OUTPUT_DIR = Path("artifacts/prediction_analysis")
FIGURES_DIR_NAME = "figures"
