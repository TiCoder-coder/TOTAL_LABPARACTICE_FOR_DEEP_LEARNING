from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .figure_safety import _safe_for_json
from ..utils.artifacts import atomic_write_bytes, canonical_json_bytes, get_project_root


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def build_phase50_handoff_payload(
    phase49_e_manifest: dict[str, Any],
    phase49_findings: dict[str, Any],
    seed_mae_rmse: dict[str, dict[str, float]],
    persistence_metrics: dict[str, Any],
    o49_completeness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completeness_ok = True
    if o49_completeness is not None:
        completeness_ok = o49_completeness.get("all_complete", False)
    return {
        "phase": "50",
        "scope": "Phase50 handoff (consumed by Phase50 implementation)",
        "residual_convention": "y_true - y_pred",
        "phase49_residual_findings_are_descriptive": True,
        "phase50_target_regime_policy": "TRAIN_DERIVED_ONLY",
        "phase50_threshold_policy": "TRAIN_DERIVED_ONLY",
        "phase49_prediction_deciles_are_NOT_phase50_regimes": True,
        "no_test_derived_thresholds_for_phase50": True,
        "phase49_passed_gates": (
            completeness_ok
            and seed_mae_rmse.get("42", {}).get("n") == 2961
            and seed_mae_rmse.get("123", {}).get("n") == 2961
            and seed_mae_rmse.get("2026", {}).get("n") == 2961
        ),
        "phase49_seed_performance": seed_mae_rmse,
        "phase49_persistence_baseline": persistence_metrics,
        "phase49_findings_summary": {
            "n_findings": phase49_findings.get("n_findings"),
            "policy_notes": phase49_findings.get("policy_notes"),
        },
        "forbidden_in_phase50": [
            "creating test-derived target regimes",
            "reusing phase49 prediction deciles as phase50 regimes",
            "applying residual correction based on phase49 findings alone",
            "selecting a best seed based on phase49 metrics alone",
            "promoting an ensemble metric as the canonical model",
        ],
        "required_in_phase50": [
            "derive target regimes exclusively from TRAIN residuals",
            "preserve phase49 residual convention (y_true - y_pred)",
            "preserve zero policy (EXACT_ZERO, no epsilon)",
            "report seed-level metrics; no best-seed selection",
            "no 3N iid pooling",
        ],
        "ready_for_phase50": True,
        "created_at_utc": _utc_now_iso(),
    }


def build_phase51_context_payload(
    phase49_e_manifest: dict[str, Any],
    o49_completeness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completeness_ok = True
    if o49_completeness is not None:
        completeness_ok = o49_completeness.get("all_complete", False)
    return {
        "phase": "51",
        "scope": "Phase51 context handoff (consumed by Phase51 implementation)",
        "phase49_performed_no_worst_error_ranking": True,
        "phase49_emitted_no_top_k_worst_rows": True,
        "phase51_first_phase_allowed_to_rank_worst_errors": True,
        "residual_convention": "y_true - y_pred",
        "phase49_o49_complete": completeness_ok,
        "phase49_seed_list": ["42", "123", "2026"],
        "phase49_n_test": 2961,
        "phase49_signoff_required_for_phase51_start": True,
        "forbidden_in_phase49": [
            "Top-K worst-error ranking",
            "ranking individual targets by absolute error",
            "selecting a worst window",
            "applying residual correction based on worst cases",
        ],
        "required_in_phase51": [
            "perform worst-error ranking for the first time",
            "use absolute_error_wh from phase49_residual_long_table.csv",
            "preserve seed isolation (no cross-seed worst pooling)",
            "preserve residual convention (y_true - y_pred)",
            "report top-K with explicit k and selection rule",
        ],
        "ready_for_phase51": completeness_ok,
        "created_at_utc": _utc_now_iso(),
    }


def write_phase50_handoff(payload: dict[str, Any], project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = root / "artifacts/residual_analysis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase50_handoff.json"
    return atomic_write_bytes(path, canonical_json_bytes(_safe_for_json(payload)))


def write_phase51_context(payload: dict[str, Any], project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = root / "artifacts/residual_analysis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase51_context_handoff.json"
    return atomic_write_bytes(path, canonical_json_bytes(_safe_for_json(payload)))
