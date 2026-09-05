"""Phase 50 — frozen contract values and invariant assertions."""
from __future__ import annotations
from typing import Any

# Frozen upstream values (carry forward from Phase 47 / 48 / 49)
SEEDS: tuple[str, ...] = ("42", "123", "2026")
N_TEST: int = 2961
TEST_POPULATION_FINGERPRINT: str = (
    "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
)
RESIDUAL_CONVENTION: str = "y_true - y_pred"
POSITIVE_RESIDUAL_SEMANTICS: str = "UNDERPREDICTION"
NEGATIVE_RESIDUAL_SEMANTICS: str = "OVERPREDICTION"
ZERO_POLICY: str = "EXACT_ZERO"

# Frozen quantile definition
QUANTILE_LIBRARY: str = "numpy"
QUANTILE_FUNCTION: str = "numpy.quantile"
QUANTILE_METHOD: str = "linear"
QUANTILE_DTYPE: str = "float64"

# Frozen temporal contract
CADENCE_MINUTES: int = 10
WB0_TEST_PREDECESSOR_AUTHORIZED: bool = True
WB0_TEST_FIRST_PREVIOUS_TARGET_ID: str = "TGT_00016773"  # last VALIDATION target

# Regime families (six, frozen)
REGIME_FAMILIES: tuple[str, ...] = (
    "R1_TARGET_LEVEL",
    "R2_EXTREME_HIGH",
    "R3_CHANGE_MAGNITUDE",
    "R4_CHANGE_DIRECTION",
    "R5_TIME_OF_DAY",
    "R6_DAY_TYPE",
)

# Frozen regime labels per family
REGIME_LABELS: dict[str, tuple[str, ...]] = {
    "R1_TARGET_LEVEL": ("TL_LOW", "TL_MID", "TL_HIGH"),
    "R2_EXTREME_HIGH": ("EXTREME_HIGH", "NON_EXTREME"),
    "R3_CHANGE_MAGNITUDE": ("CHANGE_NORMAL", "CHANGE_RAPID", "CHANGE_UNCLASSIFIED"),
    "R4_CHANGE_DIRECTION": ("DIR_UP", "DIR_FLAT", "DIR_DOWN", "DIR_UNCLASSIFIED"),
    "R5_TIME_OF_DAY": ("TOD_NIGHT", "TOD_MORNING", "TOD_AFTERNOON", "TOD_EVENING"),
    "R6_DAY_TYPE": ("DAY_WEEKDAY", "DAY_WEEKEND"),
}

# Time-of-day blocks (clock)
TOD_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("TOD_NIGHT", 0, 6),       # 00:00-05:59
    ("TOD_MORNING", 6, 12),    # 06:00-11:59
    ("TOD_AFTERNOON", 12, 18),  # 12:00-17:59
    ("TOD_EVENING", 18, 24),   # 18:00-23:59
)

# Phase 50 contract resolutions (Human-approved 2026-09-04)
RMSE_LIFT_WH: str = "regime_rmse - global_seed_rmse"           # signed Wh
RMSE_LIFT_RATIO: str = "regime_rmse / global_seed_rmse - 1"   # dimensionless
RMSE_LIFT_PCT: str = "100 * rmse_lift_ratio"                  # percentage
R2_NOT_DEFINED_LABEL: str = "NOT_DEFINED"

# Cross-seed aggregation
CROSS_SEED_DDOF: int = 1


# Invariant assertions
def assert_train_only(y_values_source: str) -> None:
    if y_values_source != "REGIME_REFERENCE_TRAIN-v1":
        raise AssertionError(
            f"Phase 50 forbids thresholds from {y_values_source!r}; "
            "must use REGIME_REFERENCE_TRAIN-v1 (Train only)."
        )


def assert_no_test_threshold(threshold_source: str) -> None:
    if "test" in threshold_source.lower():
        raise AssertionError(
            f"Phase 50 forbids Test-derived thresholds; got source={threshold_source!r}."
        )


def assert_no_best_seed(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase 50 forbids best-seed selection.")


def assert_no_ensemble(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase 50 forbids ensemble promotion.")


def assert_no_3n_iid_pooling(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase 50 forbids 3N iid pooling of residuals.")


def assert_no_cartesian_mining(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase 50 forbids Cartesian regime mining.")


def assert_no_worst_error_ranking(x: Any) -> None:
    if x is True:
        raise AssertionError(
            "Phase 50 forbids worst-error ranking; deferred to Phase 51."
        )


def assert_no_attention_analysis(x: Any) -> None:
    if x is True:
        raise AssertionError(
            "Phase 50 forbids attention extraction/visualization; deferred to Phase 52+."
        )


def assert_decile_not_phase50_regime(x: Any) -> None:
    """Ensure Phase 49 prediction deciles are NOT re-used as Phase 50 regimes."""
    if x is True:
        raise AssertionError(
            "Phase 49 prediction deciles are diagnostic-only and must NOT be reused as Phase 50 regimes."
        )


def assert_residual_sign_semantics(
    positive_label: str, negative_label: str, zero_label: str
) -> None:
    if positive_label != POSITIVE_RESIDUAL_SEMANTICS:
        raise AssertionError(
            f"positive residual must be {POSITIVE_RESIDUAL_SEMANTICS}, got {positive_label!r}"
        )
    if negative_label != NEGATIVE_RESIDUAL_SEMANTICS:
        raise AssertionError(
            f"negative residual must be {NEGATIVE_RESIDUAL_SEMANTICS}, got {negative_label!r}"
        )
    if zero_label != ZERO_POLICY:
        raise AssertionError(
            f"zero residual must be {ZERO_POLICY}, got {zero_label!r}"
        )


def contract_n_test() -> int:
    return N_TEST


def contract_seed_list() -> list[str]:
    return list(SEEDS)


def contract_test_population_sha256() -> str:
    return TEST_POPULATION_FINGERPRINT


def contract_residual_convention() -> str:
    return RESIDUAL_CONVENTION
