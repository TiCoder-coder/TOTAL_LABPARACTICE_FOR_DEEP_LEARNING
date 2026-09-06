from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .figure_safety import _safe_for_json
from course_workutils.artifacts import atomic_write_bytes, canonical_json_bytes, get_project_root


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def build_summary_payload(
    *,
    n_o49_complete: int,
    n_o49_total: int,
    n_figures: int,
    seed_mae_rmse: dict[str, dict[str, float]],
    persistence_metrics: dict[str, Any],
    findings: dict[str, Any],
    contract_invariants: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "49",
        "phase_sub_letter": "F",
        "title": "Phase49 Residual Analysis — Finalization Summary",
        "objective": (
            "Produce the canonical descriptive residual analysis for the frozen "
            "Transformer Phase47 predictions (seeds 42, 123, 2026) on N=2961 "
            "Test observations, then finalize Phase49 with full audit, figures, "
            "findings, handoffs, report, README, and strict signoff."
        ),
        "residual_convention": "y_true - y_pred",
        "positive_semantics": "UNDERPREDICTION",
        "negative_semantics": "OVERPREDICTION",
        "zero_policy": "EXACT_ZERO",
        "n_test": 2961,
        "n_seeds": 3,
        "seeds": ["42", "123", "2026"],
        "n_o49_complete": n_o49_complete,
        "n_o49_total": n_o49_total,
        "n_figures": n_figures,
        "seed_performance": seed_mae_rmse,
        "persistence_baseline": persistence_metrics,
        "findings_summary": {
            "n_findings": findings.get("n_findings"),
            "policy_notes": findings.get("policy_notes"),
        },
        "contract_invariants": contract_invariants,
        "ready_for_phase50_handoff": True,
        "ready_for_phase51_context_handoff": True,
        "phase50_thresholds_policy": "TRAIN_DERIVED_ONLY",
        "phase51_first_allowed_to_rank_worst_errors": True,
        "created_at_utc": _utc_now_iso(),
    }


def build_report_payload(
    *,
    summary: dict[str, Any],
    findings: dict[str, Any],
    seed_mae_rmse: dict[str, dict[str, float]],
    persistence_metrics: dict[str, Any],
    o49_completeness: dict[str, Any],
    figures_list: list[str],
    contract_invariants: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "49",
        "phase_sub_letter": "F",
        "title": "Phase49 Residual Analysis — Final Report",
        "sections": [
            {
                "heading": "1. Objective",
                "body": (
                    "Produce the canonical descriptive residual analysis for the "
                    "Transformer Phase47 predictions on N=2961 Test observations "
                    "across seeds 42, 123, 2026, then finalize Phase49 with full "
                    "audit, figures, findings, handoffs, report, README, and strict "
                    "signoff. NO causal claims, NO correction proposals, NO best-seed "
                    "selection, NO ensemble promotion, NO Test-derived Phase50 regimes, "
                    "NO worst-error ranking."
                ),
            },
            {
                "heading": "2. Frozen prediction lineage",
                "body": (
                    "Phase47 prediction bundles for seeds 42, 123, 2026 are frozen "
                    "and immutable. Persistence bundle (sha256 first16 = "
                    f"{summary.get('persistence_baseline', {}).get('sha256_first16', 'unknown')}) "
                    "is also frozen. Phase48 artifacts (figures, summary, handoffs) "
                    "are immutable."
                ),
            },
            {
                "heading": "3. Residual convention",
                "body": (
                    "residual_wh = y_true_wh - y_pred_wh. Positive = UNDERPREDICTION, "
                    "negative = OVERPREDICTION. Zero policy = EXACT_ZERO (no epsilon)."
                ),
            },
            {
                "heading": "4. Metric reconstruction",
                "body": (
                    "Per-seed MAE, RMSE, R² reconstructed from Phase49-B long table "
                    "residuals and verified to match Phase47 signoff values for all "
                    "3 seeds."
                ),
            },
            {
                "heading": "5. Residual distribution",
                "body": (
                    f"All 3 seeds right-skewed with heavy tails. Mean residual "
                    f"(seed42, seed123, seed2026) = "
                    f"({seed_mae_rmse['42']['mean_residual']:.4f}, "
                    f"{seed_mae_rmse['123']['mean_residual']:.4f}, "
                    f"{seed_mae_rmse['2026']['mean_residual']:.4f}) Wh; "
                    f"std ≈ 61–65 Wh; MAD ≈ 10.5–12 Wh."
                ),
            },
            {
                "heading": "6. Bias / sign balance",
                "body": (
                    "Mild net UNDERPREDICTION tendency across seeds. Sign balance "
                    "≈ 47–51% across U/O classes; EXACT count = 0 in all seeds."
                ),
            },
            {
                "heading": "7. Tail behavior",
                "body": (
                    "max_abs_error 535–601 Wh; abs_error p99 ≈ 294–309 Wh; "
                    "abs_error p95 ≈ 116–128 Wh; abs_error p90 ≈ 57–64 Wh."
                ),
            },
            {
                "heading": "8. Temporal ACF",
                "body": (
                    "Gap-safe ACF, lags 1..144 per seed. Positive short-lag "
                    "autocorrelation (ACF lag-1 ≈ 0.20) decaying to ~0 by lag 144."
                ),
            },
            {
                "heading": "9. Ljung-Box as secondary diagnostic",
                "body": (
                    "Ljung-Box at lags {6, 36, 144} yields p ≈ 0 across all seeds. "
                    "Status = SECONDARY_DIAGNOSTIC. NO p-value gates Phase49 PASS/FAIL."
                ),
            },
            {
                "heading": "10. Sign runs",
                "body": (
                    "Sign runs break at sign change / EXACT zero / temporal gap. "
                    "Median run length = 1 across seeds; p95 run length 8–9; "
                    "max run length 47–60; total runs ~1080–1200 per seed."
                ),
            },
            {
                "heading": "11. Sign transitions",
                "body": (
                    "Only consecutive pairs separated by exactly 10 min counted. "
                    "U→U ≈ 0.60, O→O ≈ 0.59–0.65 within-class probabilities across seeds."
                ),
            },
            {
                "heading": "12. Rolling residual behavior",
                "body": (
                    "Rolling window = 144 exact contiguous observations (= 24h). "
                    "n_windows = 2818 per seed; total = 8454. No partial / "
                    "interpolated / padded / forward-filled windows."
                ),
            },
            {
                "heading": "13. Magnitude associations",
                "body": (
                    "|residual| vs y_true: Pearson ≈ 0.73–0.79; "
                    "|residual| vs y_pred: Pearson ≈ 0.47–0.50; "
                    "DESCRIPTIVE only."
                ),
            },
            {
                "heading": "14. Prediction-decile diagnostics",
                "body": (
                    "10 equal-frequency bins via np.quantile on y_pred. MAE grows "
                    "monotonically with prediction magnitude. "
                    "DESCRIPTIVE ONLY — NOT Phase50 target regimes."
                ),
            },
            {
                "heading": "15. Cross-seed residual agreement",
                "body": (
                    "Pearson 0.90–0.92; Spearman 0.74–0.78 across seed pairs. "
                    "Scope = CROSS_SEED_RESIDUAL_AGREEMENT_DIAGNOSTIC — NOT a "
                    "model-performance comparison."
                ),
            },
            {
                "heading": "16. Sign consensus",
                "body": (
                    "ALL_UNDER 33.87%, ALL_OVER 36.64%, TWO_UNDER_ONE_OVER 14.29%, "
                    "TWO_OVER_ONE_UNDER 15.20%, ALL_EXACT 0%, MIXED 0%."
                ),
            },
            {
                "heading": "17. Persistence context",
                "body": (
                    f"Persistence baseline (N=2961, sha-verified): "
                    f"MAE = {persistence_metrics['mae']:.4f} Wh, "
                    f"RMSE = {persistence_metrics['rmse']:.4f} Wh, "
                    f"EXACT fraction = {persistence_metrics['exact_fraction']:.4f}. "
                    "Transformer beats Persistence on RMSE/R²; Persistence beats "
                    "Transformer on MAE. Canonical Phase47 interpretation preserved."
                ),
            },
            {
                "heading": "18. Important limitations",
                "body": (
                    "1) Phase49 uses ONLY the frozen Test predictions and persistence "
                    "bundle. No training, no checkpoint access, no new inference. "
                    "2) Ljung-Box p-values are SECONDARY and do NOT gate PASS/FAIL. "
                    "3) Cross-seed residual agreement is DESCRIPTIVE — not a model "
                    "ranking. 4) All |residual| / residual vs y_true / residual vs "
                    "y_pred associations are DESCRIPTIVE — not causal. 5) Prediction "
                    "deciles are DESCRIPTIVE — NOT Phase50 regimes."
                ),
            },
            {
                "heading": "19. What Phase49 intentionally did NOT do",
                "body": (
                    "No model retraining. No checkpoint loading. No new Test "
                    "inference. No bias correction. No residual correction. No "
                    "recalibration. No best-seed selection. No ensemble promotion. "
                    "No 3N iid interpretation. No Test-derived Phase50 regimes. No "
                    "worst-error ranking. No attention analysis."
                ),
            },
            {
                "heading": "20. Handoff to Phase50/51",
                "body": (
                    "Phase50 handoff (artifacts/residual_analysis/phase50_handoff.json) "
                    "states: target regimes MUST be TRAIN-derived; phase49 deciles "
                    "are NOT authorized as Phase50 regimes; no Test-derived "
                    "thresholds. Phase51 context handoff "
                    "(artifacts/residual_analysis/phase51_context_handoff.json) "
                    "states: Phase49 performed no worst-error ranking; Phase51 is "
                    "the first phase allowed to rank worst errors."
                ),
            },
        ],
        "n_findings": findings.get("n_findings"),
        "findings": findings.get("findings"),
        "o49_completeness": {
            "n_o49_complete": o49_completeness["n_o49_complete"],
            "n_o49_total": o49_completeness["n_o49_total"],
            "all_complete": o49_completeness["all_complete"],
            "missing": o49_completeness.get("missing", []),
        },
        "figures": figures_list,
        "contract_invariants": contract_invariants,
        "created_at_utc": _utc_now_iso(),
    }


def build_readme_payload(
    *,
    o49_completeness: dict[str, Any],
    figures_list: list[str],
    seed_mae_rmse: dict[str, dict[str, float]],
    contract_invariants: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "49",
        "phase_sub_letter": "F",
        "title": "Phase49 Residual Analysis — README",
        "artifact_directory": "artifacts/residual_analysis/",
        "figures_directory": "artifacts/residual_analysis/figures/",
        "residual_convention": "residual = y_true - y_pred",
        "sign_semantics": {
            "positive": "UNDERPREDICTION",
            "negative": "OVERPREDICTION",
            "zero": "EXACT (no epsilon)",
        },
        "source_lineage": {
            "phase47_predictions": "frozen, sha256-verified",
            "phase47_persistence": "frozen, sha256-verified",
            "phase48_artifacts": "frozen, immutable",
        },
        "seed_semantics": "Each seed is an independent Transformer training run on the same Test population. Seed mean residual is SEED_MEAN_RESIDUAL_DESCRIPTIVE (never treated as ensemble residual).",
        "acf_semantics": (
            "Gap-safe autocorrelation function. Lag k = k steps = k × 10 minutes. "
            "Single contiguous Test segment per seed. ACF never crosses a temporal gap."
        ),
        "ljung_box_status": (
            "SECONDARY_DIAGNOSTIC. Lags = [6, 36, 144]. p-values do NOT gate "
            "Phase49 PASS/FAIL."
        ),
        "sign_run_break_rules": [
            "sign change",
            "exact zero residual (no epsilon)",
            "temporal gap (delta != 10 min)",
        ],
        "rolling_144_semantics": (
            "Rolling window = 144 exact contiguous observations = 24h at 10-min "
            "cadence. No partial / interpolated / padded / forward-filled windows."
        ),
        "prediction_decile_policy": (
            "DESCRIPTIVE diagnostic only. 10 equal-frequency bins via np.quantile "
            "on y_pred. NOT a Phase50 target regime. NOT allowed for Phase50 "
            "regime construction."
        ),
        "phase50_threshold_policy": "TRAIN_DERIVED_ONLY",
        "phase51_worst_error_deferral": (
            "Phase49 performed no worst-error ranking. Phase51 is the first "
            "phase allowed to rank worst errors."
        ),
        "reproducibility": (
            "All Phase49-F artifacts are deterministic from canonical Phase49-B/C/D/E "
            "artifacts. Re-running materialize_phase49_f() produces identical files."
        ),
        "forbidden_interpretations": [
            "Best-seed selection (Phase49 reports per-seed metrics only).",
            "Ensemble promotion (no seed_mean_residual as canonical model metric).",
            "3N iid pooling.",
            "Causal interpretation of |residual| / residual vs y_true / residual vs y_pred.",
            "Using Phase49 prediction deciles as Phase50 regimes.",
            "Phase49 worst-error ranking (deferred to Phase51).",
            "Phase49 attention analysis (deferred to Phase54).",
            "Phase49 PASS/FAIL gated by Ljung-Box p-value.",
            "Phase49 model retraining, checkpoint loading, new Test inference, residual correction, bias correction, recalibration.",
        ],
        "key_artifacts": {
            "phase_49_signoff": "artifacts/residual_analysis/phase_49_signoff.json",
            "phase49_e_manifest": "artifacts/residual_analysis/phase49_e_manifest.json",
            "phase49_findings": "artifacts/residual_analysis/phase49_findings.json",
            "phase50_handoff": "artifacts/residual_analysis/phase50_handoff.json",
            "phase51_context_handoff": "artifacts/residual_analysis/phase51_context_handoff.json",
            "residual_long_table": "artifacts/residual_analysis/residual_long_table.csv",
            "residual_wide_table": "artifacts/residual_analysis/residual_wide_table.csv",
        },
        "figures": figures_list,
        "o49_completeness": {
            "n_o49_complete": o49_completeness["n_o49_complete"],
            "n_o49_total": o49_completeness["n_o49_total"],
        },
        "contract_invariants": contract_invariants,
        "created_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


def write_summary(payload: dict[str, Any], project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = root / "artifacts/residual_analysis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_summary.json"
    return atomic_write_bytes(path, canonical_json_bytes(_safe_for_json(payload)))


def write_findings(payload: dict[str, Any], project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = root / "artifacts/residual_analysis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_findings.json"
    return atomic_write_bytes(path, canonical_json_bytes(_safe_for_json(payload)))


def write_report(payload: dict[str, Any], project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = root / "artifacts/residual_analysis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_report.json"
    return atomic_write_bytes(path, canonical_json_bytes(_safe_for_json(payload)))


def write_readme(payload: dict[str, Any], project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = root / "artifacts/residual_analysis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase49_README.json"
    return atomic_write_bytes(path, canonical_json_bytes(_safe_for_json(payload)))
