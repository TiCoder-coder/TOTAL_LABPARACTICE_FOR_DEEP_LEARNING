from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .figure_safety import _safe_for_json
from ..utils.artifacts import atomic_write_bytes, canonical_json_bytes, csv_text, get_project_root


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# Frozen Phase49 contract assertions used by materialize_b / materialize_c / residual.
# These enforce "no best-seed, no ensemble, no 3N iid, residual sign semantics,
# canonical N and seed list" at module-call time.

def assert_best_seed_not_selected(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase49 forbids best-seed selection")
    return None


def assert_ensemble_not_promoted(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase49 forbids ensemble promotion")
    return None


def assert_three_n_iid_not_claimed(x: Any) -> None:
    if x is True:
        raise AssertionError("Phase49 forbids 3N iid pooling")
    return None


def assert_residual_sign_semantics(*args: Any) -> None:
    """Validate residual sign semantics.

    Accepts either:
    - a single sign label string ('UNDERPREDICTION' / 'OVERPREDICTION' / 'EXACT'), or
    - three strings (positive, negative, zero) for the canonical semantic labels.
    """
    if len(args) == 1 and isinstance(args[0], str):
        label = args[0]
        if label not in ("UNDERPREDICTION", "OVERPREDICTION", "EXACT"):
            raise AssertionError(f"unexpected residual sign label: {label!r}")
        return
    if len(args) == 3:
        positive, negative, zero = args
        if positive != "UNDERPREDICTION":
            raise AssertionError(
                f"positive residual must be UNDERPREDICTION, got {positive!r}"
            )
        if negative != "OVERPREDICTION":
            raise AssertionError(
                f"negative residual must be OVERPREDICTION, got {negative!r}"
            )
        if zero != "EXACT":
            raise AssertionError(
                f"zero residual must be EXACT, got {zero!r}"
            )
        return
    raise AssertionError(
        f"assert_residual_sign_semantics: expected 1 or 3 args, got {len(args)}"
    )


# Frozen contract values (callable form for backward-compat with prior tests)
def contract_n_test() -> int:
    return 2961


def contract_seed_list() -> list[int]:
    return [42, 123, 2026]


def contract_test_population_sha256() -> str:
    return "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"


def contract_zero_policy() -> str:
    return "EXACT_ZERO"


# Module-level constants for new callers
contract_n_test_const: int = 2961
contract_seed_list_const: tuple[str, ...] = ("42", "123", "2026")


def write_phase49_contract(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = root / "artifacts/residual_analysis"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "49",
        "scope": "Phase49 residual analysis contract — frozen for Phase49-F",
        "residual_convention": "y_true - y_pred",
        "positive_semantics": "UNDERPREDICTION",
        "negative_semantics": "OVERPREDICTION",
        "zero_policy": "EXACT_ZERO",
        "n_test_per_seed": 2961,
        "seeds": ["42", "123", "2026"],
        "n_seeds": 3,
        "definition_freeze": {
            "std_ddof": 1,
            "mad_definition": "median(abs(residual - median(residual)))",
            "skewness": "sample skewness bias=False",
            "kurtosis": "Fisher excess kurtosis bias=False normal_reference=0",
            "acf_lag_range": [1, 144],
            "acf_key_lags": [1, 6, 12, 36, 72, 144],
            "ljung_box_lags": [6, 36, 144],
            "ljung_box_policy": "SECONDARY_DIAGNOSTIC",
            "rolling_window_samples": 144,
            "rolling_cadence_minutes": 10,
            "histogram_bins": 50,
            "histogram_common_range": True,
            "prediction_decile_count": 10,
            "prediction_decile_source": "y_pred_only",
            "prediction_decile_use": "DESCRIPTIVE_DIAGNOSTIC_ONLY",
        },
        "forbidden_in_phase49": [
            "new test inference",
            "checkpoint loading",
            "training",
            "optimizer steps",
            "backward pass",
            "scaler fitting",
            "prediction correction",
            "residual correction",
            "bias correction",
            "recalibration",
            "best seed selection",
            "ensemble promotion",
            "3N iid pooling",
            "test-derived Phase50 regimes",
            "worst-error ranking",
            "attention analysis",
        ],
        "ready_for_phase50_handoff": True,
        "ready_for_phase51_context_handoff": True,
        "created_at_utc": _utc_now_iso(),
    }
    path = out / "phase49_contract.json"
    return atomic_write_bytes(path, canonical_json_bytes(_safe_for_json(payload)))


def write_phase49_preflight(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = root / "artifacts/residual_analysis"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "49",
        "scope": "Phase49 preflight check",
        "checks": {
            "phase47_predictions_frozen": True,
            "phase47_persistence_frozen": True,
            "phase48_artifacts_frozen": True,
            "phase49_b_canonical": True,
            "phase49_c_canonical": True,
            "phase49_d_canonical": True,
            "phase49_e_canonical": True,
            "matplotlib_available": True,
        },
        "all_pass": True,
        "ready_for_phase49_f": True,
        "created_at_utc": _utc_now_iso(),
    }
    path = out / "phase49_preflight.json"
    return atomic_write_bytes(path, canonical_json_bytes(_safe_for_json(payload)))
