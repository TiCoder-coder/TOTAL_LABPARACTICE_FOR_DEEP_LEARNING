"""Phase 51 — frozen selection-contract constants and invariant assertions.

This module is the authoritative source of truth for all Phase 51 governance constants.
It is read-only at runtime — no computation, no inference, no ranking.
"""
from __future__ import annotations

from typing import Any

# ── Upstream frozen values (inherited from Phase 47/48/49/50) ─────────────────
SEEDS: tuple[str, ...] = ("42", "123", "2026")
N_TEST: int = 2961
TEST_POPULATION_FINGERPRINT: str = (
    "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
)

RESIDUAL_CONVENTION: str = "y_true - y_pred"
POSITIVE_RESIDUAL_SEMANTICS: str = "UNDERPREDICTION"
NEGATIVE_RESIDUAL_SEMANTICS: str = "OVERPREDICTION"
ZERO_POLICY: str = "EXACT_ZERO"

# ── Temporal contract ───────────────────────────────────────────────────────────
CADENCE_MINUTES: int = 10          # inherited from Phase 49 / Phase 50
LOCAL_CONTEXT_RADIUS: int = 6       # target-centric ±6 steps

# ── Selection contract (HUMAN-APPROVED 2026-09-04, v1.13) ─────────────────────
K_ABS_PER_SEED: int = 20          # W1: top 20 per seed
K_SHARED: int = 20                # W2: top 20 by mean(|e|)
K_UNDER_PER_SEED: int = 10        # W3: top 10 per seed, residual > 0
K_OVER_PER_SEED: int = 10          # W4: top 10 per seed, residual < 0
K_SHARED_SIGNED: int = 10         # W3_SH / W4_SH: shared all-under / all-over

PRIMARY_RANKING_METRIC: str = "absolute_error_wh"
PRIMARY_RANKING_DIRECTION: str = "DESC"
TIE_BREAK: str = "target_id ASC"
RANDOM_TIEBREAK_FORBIDDEN: bool = True
MANUAL_PRESELECTION_FORBIDDEN: bool = True
CHERRY_PICKING_FORBIDDEN: bool = True

# ── Ranking families ────────────────────────────────────────────────────────────
RANKING_FAMILIES: dict[str, dict[str, Any]] = {
    "W1": {
        "name": "PER_SEED_WORST",
        "description": "Top 20 absolute_error_wh per seed separately (seed 42 / 123 / 2026)",
        "k": K_ABS_PER_SEED,
        "filter": None,
        "group_by": "seed",
    },
    "W2": {
        "name": "SHARED_WORST",
        "description": (
            "Top 20 by mean(|e42|+|e123|+|e2026|)/3 across all Test targets"
        ),
        "k": K_SHARED,
        "filter": None,
        "group_by": None,
    },
    "W3": {
        "name": "UNDERPREDICTION_WORST",
        "description": "Top 10 absolute_error_wh per seed, residual > 0 only",
        "k": K_UNDER_PER_SEED,
        "filter": "residual > 0",
        "group_by": "seed",
    },
    "W4": {
        "name": "OVERPREDICTION_WORST",
        "description": "Top 10 absolute_error_wh per seed, residual < 0 only",
        "k": K_OVER_PER_SEED,
        "filter": "residual < 0",
        "group_by": "seed",
    },
    "W3_SH": {
        "name": "SHARED_ALL_UNDER",
        "description": "Top 10 with residual > 0 for ALL 3 seeds",
        "k": K_SHARED_SIGNED,
        "filter": "residual > 0 for ALL seeds",
        "group_by": None,
    },
    "W4_SH": {
        "name": "SHARED_ALL_OVER",
        "description": "Top 10 with residual < 0 for ALL 3 seeds",
        "k": K_SHARED_SIGNED,
        "filter": "residual < 0 for ALL seeds",
        "group_by": None,
    },
}

# ── Phase 50 regime contract (inherited, NOT recomputed) ──────────────────────
REGIME_FAMILIES: tuple[str, ...] = (
    "R1_TARGET_LEVEL",
    "R2_EXTREME_HIGH",
    "R3_CHANGE_MAGNITUDE",
    "R4_CHANGE_DIRECTION",
    "R5_TIME_OF_DAY",
    "R6_DAY_TYPE",
)

REGIME_LABELS: dict[str, tuple[str, ...]] = {
    "R1_TARGET_LEVEL": ("TL_LOW", "TL_MID", "TL_HIGH"),
    "R2_EXTREME_HIGH": ("EXTREME_HIGH", "NON_EXTREME"),
    "R3_CHANGE_MAGNITUDE": ("CHANGE_NORMAL", "CHANGE_RAPID", "CHANGE_UNCLASSIFIED"),
    "R4_CHANGE_DIRECTION": ("DIR_UP", "DIR_FLAT", "DIR_DOWN", "DIR_UNCLASSIFIED"),
    "R5_TIME_OF_DAY": ("TOD_NIGHT", "TOD_MORNING", "TOD_AFTERNOON", "TOD_EVENING"),
    "R6_DAY_TYPE": ("DAY_WEEKDAY", "DAY_WEEKEND"),
}

# Frozen Phase 50 thresholds (TRAIN-derived only, via v1.12)
PHASE50_THRESHOLDS: dict[str, float] = {
    "Q25_y": 50.0,
    "Q75_y": 100.0,
    "Q90_y": 210.0,
    "Q90_abs_delta": 80.0,
}

# Frozen Phase 50 SHA256 values
PHASE50_TEST_ASSIGNMENT_SHA256: str = (
    "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac"
)
PHASE50_TRAIN_THRESHOLD_SHA256: str = (
    "2fe9ad4f873e3b3e42013fe3b2d630e377e4e568120769bd2234a76d6974b109"
)

# LSTM eligibility (NOT_ELIGIBLE_CONFIG_MISMATCH, frozen from Phase 47)
LSTM_ELIGIBILITY: str = "NOT_ELIGIBLE_CONFIG_MISMATCH"
LSTM_NOT_ELIGIBLE_REASON: str = (
    "LSTM uses lookback=None, but final Transformer uses lookback=72"
)

# Cross-seed aggregation
CROSS_SEED_DDOF: int = 1
ALL_THREE_SEEDS_RETAINED: bool = True
SEED_POOLING_AS_3N_IID: bool = False
BEST_SEED_SELECTION: bool = False
ENSEMBLE: bool = False
PERSISTENCE_CONTEXT_ONLY: bool = True


# ── Forbidden-in-Phase-51-B invariants ────────────────────────────────────────

def assert_no_worst_error_ranking_in_phase51b(x: Any) -> None:
    """Raise if any Phase 51-B step attempts to rank errors."""
    if x is True:
        raise AssertionError(
            "Phase 51-B forbids worst-error ranking; deferred to Phase 51-C."
        )


def assert_no_training(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase 51 forbids training.")


def assert_no_inference(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase 51 forbids new Test inference.")


def assert_no_checkpoint_loading(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase 51 forbids checkpoint loading.")


def assert_no_best_seed(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase 51 forbids best-seed selection.")


def assert_no_ensemble(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase 51 forbids ensemble construction.")


def assert_no_prediction_correction(x: Any) -> None:
    if x is True:
        raise AssertionError(
            "Phase 51 forbids prediction correction / recalibration."
        )


def assert_phase52_not_started(x: Any) -> None:
    if x is True:
        raise AssertionError(
            "Phase 52 is not authorized; attention analysis deferred to Phase 52+."
        )


def assert_residual_sign_semantics(
    positive_label: str, negative_label: str, zero_label: str
) -> None:
    if positive_label != POSITIVE_RESIDUAL_SEMANTICS:
        raise AssertionError(
            f"positive residual must be {POSITIVE_RESIDUAL_SEMANTICS}, "
            f"got {positive_label!r}"
        )
    if negative_label != NEGATIVE_RESIDUAL_SEMANTICS:
        raise AssertionError(
            f"negative residual must be {NEGATIVE_RESIDUAL_SEMANTICS}, "
            f"got {negative_label!r}"
        )
    if zero_label != ZERO_POLICY:
        raise AssertionError(
            f"zero residual must be {ZERO_POLICY}, got {zero_label!r}"
        )


def assert_no_phase50_regime_recomputation(x: Any) -> None:
    if x is True:
        raise AssertionError(
            "Phase 51 forbids recomputing Phase 50 regime thresholds or labels."
        )


# ── Callable contract accessors ────────────────────────────────────────────────

def contract_n_test() -> int:
    return N_TEST


def contract_seed_list() -> list[str]:
    return list(SEEDS)


def contract_test_population_sha256() -> str:
    return TEST_POPULATION_FINGERPRINT


def contract_residual_convention() -> str:
    return RESIDUAL_CONVENTION


def contract_phase50_assignment_sha256() -> str:
    return PHASE50_TEST_ASSIGNMENT_SHA256


def contract_phase50_threshold_sha256() -> str:
    return PHASE50_TRAIN_THRESHOLD_SHA256
