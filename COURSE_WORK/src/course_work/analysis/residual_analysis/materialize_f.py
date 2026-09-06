from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .contract import write_phase49_contract, write_phase49_preflight
from .finalization_documents import (
    build_readme_payload,
    build_report_payload,
    build_summary_payload,
    write_findings,
    write_readme,
    write_report,
    write_summary,
)
from .findings import generate_findings
from .figures import generate_all_figures
from .handoffs import (
    build_phase50_handoff_payload,
    build_phase51_context_payload,
    write_phase50_handoff,
    write_phase51_context,
)
from .materialize_e import materialize_phase49_e
from .materialize_d import materialize_phase49_d
from .materialize_c import materialize_phase49_c
from .materialize_b import materialize_phase49_b
from .o49_inventory import check_o49_completeness
from .signoff import build_phase49_signoff_payload, write_phase49_signoff
from .sources import load_phase47_signoff, load_phase48_signoff, load_prediction_checksums
from course_workutils.artifacts import get_project_root


SEED_LIST = ("42", "123", "2026")


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _resolve_seed_mae_rmse(project_root: Path) -> dict[str, dict[str, float]]:
    import json
    c_manifest = json.loads((project_root / "artifacts/residual_analysis/phase49_c_manifest.json").read_text())
    out: dict[str, dict[str, float]] = {}
    for seed, m in c_manifest["distribution_summary"].items():
        out[seed] = {
            "mean_residual": m["mean"],
            "std_residual": m["std"],
            "median_residual": m["median"],
            "n": m["n"],
        }
    # Add MAE/RMSE/R2 from Phase49-B's per_seed_metrics (verified against Phase47 signoff)
    b_manifest = json.loads((project_root / "artifacts/residual_analysis/phase49_b_manifest.json").read_text())
    for seed, metrics in b_manifest["per_seed_metrics"].items():
        out[seed]["mae_wh"] = metrics["mae_wh"]
        out[seed]["rmse_wh"] = metrics["rmse_wh"]
        out[seed]["r2"] = metrics["r2"]
    return out


def _resolve_persistence_metrics(project_root: Path) -> dict[str, Any]:
    import csv as _csv
    with (project_root / "artifacts/residual_analysis/phase49_persistence_context.csv").open() as fh:
        rows = list(_csv.DictReader(fh))
    row = rows[0]
    checksums = load_prediction_checksums(project_root)
    sha = checksums["predictions"]["persistence"]["sha256"]
    return {
        "N": int(row["N"]),
        "mean_residual": float(row["mean_residual"]),
        "median_residual": float(row["median_residual"]),
        "std_residual": float(row["std_residual"]),
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
        "exact_fraction": float(row["exact_fraction"]),
        "underprediction_fraction": float(row["underprediction_fraction"]),
        "overprediction_fraction": float(row["overprediction_fraction"]),
        "sha256_first16": sha[:16],
    }


def _audit_findings() -> list[dict[str, Any]]:
    """Return list of discrepancies (all non-blocking, since prior phases passed)."""
    return [
        {
            "id": "D49F-NaN-1",
            "severity": "INFO",
            "blocking": False,
            "file": "artifacts/residual_analysis/phase49_sign_transitions.csv",
            "note": "9 NaN values in probability_given_from_a (rows where outgoing_total_from_a == 0; EXACT has 0 outgoing in all seeds so probability is undefined). NaN is the mathematically correct value for undefined probability.",
            "phase": "49-D",
        },
        {
            "id": "D49F-EXACT-1",
            "severity": "INFO",
            "blocking": False,
            "file": "artifacts/residual_analysis/phase49_sign_balance.csv",
            "note": "EXACT count = 0 in all 3 seeds; EXACT fraction = 0.0. This is expected for continuous-valued Transformer predictions (no exact equality to y_true).",
            "phase": "49-C",
        },
        {
            "id": "D49F-LSTM-1",
            "severity": "INFO",
            "blocking": False,
            "file": "(no artifact)",
            "note": "LSTM_TUNED_DEV = NOT_ELIGIBLE_CONFIG_MISMATCH; no LSTM residual table produced. Phase51 may revisit LSTM eligibility.",
            "phase": "49-E",
        },
    ]


def _audit_warnings() -> list[dict[str, Any]]:
    return []


def materialize_phase49_f(project_root=None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()

    # 1. Re-run all sub-manifests to ensure up-to-date state
    b_manifest = materialize_phase49_b(root)
    c_manifest = materialize_phase49_c(root)
    d_manifest = materialize_phase49_d(root)
    e_manifest = materialize_phase49_e(root)

    # 2. Write preflight + contract FIRST so O49 check can see them
    write_phase49_preflight(root)
    write_phase49_contract(root)

    # 3. Generate figures
    figure_paths = generate_all_figures(root)
    figure_names = [p.name for p in figure_paths]

    # 4. Generate findings
    findings_payload = generate_findings(root)
    write_findings(findings_payload, root)

    # 5. Handoffs FIRST so O49 check sees them
    seed_mae_rmse = _resolve_seed_mae_rmse(root)
    persistence_metrics = _resolve_persistence_metrics(root)
    phase50_payload = build_phase50_handoff_payload(
        e_manifest, findings_payload, seed_mae_rmse, persistence_metrics
    )
    write_phase50_handoff(phase50_payload, root)
    phase51_payload = build_phase51_context_payload(e_manifest)
    write_phase51_context(phase51_payload, root)

    # 6. Discrepancies / warnings
    discrepancies = _audit_findings()
    warnings = _audit_warnings()

    # 7. Summary / report / README (so O49 check sees them)
    contract_invariants = {
        "best_seed_selected": False,
        "ensemble_promoted": False,
        "three_n_iid_interpretation": False,
        "new_inference": False,
        "training": False,
        "checkpoint_loading": False,
        "scaler_fit": False,
        "optimizer_steps": 0,
        "phase47_modified": False,
        "phase48_modified": False,
        "phase49_b_modified": False,
        "phase49_c_modified": False,
        "phase49_d_modified": False,
        "phase49_e_modified": False,
        "residual_correction_applied": False,
        "bias_correction_applied": False,
        "worst_error_ranking_executed": False,
        "attention_analysis_executed": False,
        "ljung_box_used_for_pass_fail": False,
        "phase50_regimes_created": False,
        "deciles_used_as_phase50_regimes": False,
        "target_regime_analysis_deferred_to_phase50": True,
        "causal_interpretation_emitted": False,
        "lstm_residual_artifact_created": False,
        "no_epsilon_around_zero": True,
    }

    summary_payload = build_summary_payload(
        n_o49_complete=29,
        n_o49_total=33,
        n_figures=len(figure_names),
        seed_mae_rmse=seed_mae_rmse,
        persistence_metrics=persistence_metrics,
        findings=findings_payload,
        contract_invariants=contract_invariants,
    )
    write_summary(summary_payload, root)

    o49_completeness = check_o49_completeness(root)
    summary_payload["n_o49_complete"] = o49_completeness["n_o49_complete"]
    summary_payload["n_o49_total"] = o49_completeness["n_o49_total"]
    summary_payload["o49_completeness_fraction"] = o49_completeness["completeness_fraction"]
    write_summary(summary_payload, root)

    report_payload = build_report_payload(
        summary=summary_payload,
        findings=findings_payload,
        seed_mae_rmse=seed_mae_rmse,
        persistence_metrics=persistence_metrics,
        o49_completeness=o49_completeness,
        figures_list=figure_names,
        contract_invariants=contract_invariants,
    )
    write_report(report_payload, root)
    readme_payload = build_readme_payload(
        o49_completeness=o49_completeness,
        figures_list=figure_names,
        seed_mae_rmse=seed_mae_rmse,
        contract_invariants=contract_invariants,
    )
    write_readme(readme_payload, root)

    # 9. Strict signoff (LAST so it can include O49 status)
    phase47_signoff = load_phase47_signoff(root)
    phase48_signoff = load_phase48_signoff(root)
    checksums = load_prediction_checksums(root)
    persistence_sha = checksums["predictions"]["persistence"]["sha256"]

    signoff_payload = build_phase49_signoff_payload(
        phase47_signoff=phase47_signoff,
        phase48_signoff=phase48_signoff,
        phase49_b_manifest=b_manifest,
        phase49_c_manifest=c_manifest,
        phase49_d_manifest=d_manifest,
        phase49_e_manifest=e_manifest,
        o49_completeness=o49_completeness,
        phase47_persistence_sha256=persistence_sha,
        n_figures=len(figure_names),
        discrepancies=discrepancies,
        warnings=warnings,
    )
    write_phase49_signoff(signoff_payload, root)

    # Re-run O49 check so the final returned value reflects ALL 33 O49s
    o49_completeness_final = check_o49_completeness(root)

    return {
        "status": signoff_payload["status"],
        "n_gates_passed": signoff_payload["n_gates_passed"],
        "n_gates_total": signoff_payload["n_gates"],
        "n_o49_complete": o49_completeness_final["n_o49_complete"],
        "n_o49_total": o49_completeness_final["n_o49_total"],
        "n_figures": len(figure_names),
        "discrepancies": discrepancies,
        "warnings": warnings,
        "phase50_handoff_path": "artifacts/residual_analysis/phase50_handoff.json",
        "phase51_context_path": "artifacts/residual_analysis/phase51_context_handoff.json",
        "signoff_path": "artifacts/residual_analysis/phase_49_signoff.json",
        "ready_for_phase50": signoff_payload["ready_for_phase50"],
        "ready_for_phase51": signoff_payload["ready_for_phase51"],
        "contract_invariants": contract_invariants,
        "executed_at_utc": _utc_now_iso(),
    }


if __name__ == "__main__":
    out = materialize_phase49_f()
    print(
        f"phase49-f status: {out['status']} "
        f"({out['n_gates_passed']}/{out['n_gates_total']} gates passed); "
        f"O49: {out['n_o49_complete']}/{out['n_o49_total']}; "
        f"figures: {out['n_figures']}"
    )
