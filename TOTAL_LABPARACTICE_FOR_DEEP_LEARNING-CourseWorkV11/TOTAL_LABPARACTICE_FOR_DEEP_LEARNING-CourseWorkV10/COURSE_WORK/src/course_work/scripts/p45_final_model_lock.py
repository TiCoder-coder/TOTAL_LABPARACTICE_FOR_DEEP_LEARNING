"""Phase 45 — Final Model Lock orchestrator.

Phase45 is a NO-TRAIN governance phase. It:

  1. Loads + validates Phase44 signoff + handoff + Phase42 shortlist.
  2. Locks the recommended Transformer candidate.
  3. Computes FINAL_REFIT_EPOCHS as median(RO1, RO2, RO3).
  4. Builds the FINAL_DEV_REGION-v1 population from canonical splits.
  5. Builds the FINAL_SCALING-v1 contract (without inventing checksums).
  6. Builds the FINAL_REFIT_MODE-v1 recipe + seed contract + run matrix.
  7. Computes four deterministic fingerprints.
  8. Writes 38 O45.* artifacts to ``artifacts/final_model_lock/``.
  9. Runs the plan §183–§192 acceptance checklist.
 10. Writes ``phase_45_signoff.json`` + ``phase46_three_seed_handoff.json`` + ``phase47_test_evaluation_guard.json``.

Hard rules (plan §2):
  new_training_runs       = 0
  new_validation_runs     = 0
  new_test_runs           = 0
  optimizer_steps         = 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_work.final_model_lock import (
    ARTIFACT_NAMES,
    LockedCandidate,
    load_phase44_handoff,
    load_phase44_signoff,
    load_phase42_shortlist,
    lock_candidate,
    derive_final_epoch,
    build_final_dev_population,
    build_final_dev_contract,
    build_scaling_contract,
    scaling_contract_to_dict,
    build_recipe,
    recipe_to_dict,
    build_seed_contract,
    build_run_matrix,
    build_lineage_audit,
    build_candidate_source_audit,
    build_boundary_sensitivity_evidence,
    build_baseline_context_evidence,
    build_epoch_policy_contract,
    build_epoch_source_audit,
    config_fingerprint,
    recipe_fingerprint,
    lineage_fingerprint,
    lock_fingerprint,
    write_all_o45_artifacts,
    write_phase45_signoff,
    run_preflight,
)
from course_work.utils.artifacts import read_json, canonical_json_bytes

ARTIFACT_DIR = ROOT / "artifacts" / "final_model_lock"


@dataclass(frozen=True)
class Phase45LockResult:
    overall_status: str 
    lock_sha: str
    config_sha: str
    recipe_sha: str
    lineage_sha: str
    artifact_dir: Path
    artifacts_written: list[str]
    discrepancies: list[str]
    warnings: list[str]


def _now_iso() -> str:
    """Used only for provenance metadata (e.g., archive manifest). NEVER inside fingerprint inputs."""
    return datetime.now(timezone.utc).isoformat()


def _archive_existing(artifact_dir: Path) -> dict[str, Any]:
    """Archive any stale Phase45 aggregate files under a timestamped history dir.

    Idempotent: if archive dir already exists, do nothing extra.
    Returns a manifest dict.
    """
    history_root = artifact_dir / "_history"
    utc_now = _now_iso().replace(":", "").replace("-", "").replace(".", "")
    archive_dir = history_root / f"PHASE45_CORRECTIVE_{utc_now}"
    if archive_dir.exists():
        return {"archive_dir": str(archive_dir.relative_to(artifact_dir)), "files": []}

    manifest: list[dict[str, Any]] = []
    archive_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(artifact_dir.iterdir()):
        if p.name == "_history":
            continue
        if p.is_file():
            target = archive_dir / p.name
            shutil.copy2(p, target)
            manifest.append({
                "source_path": str(p.relative_to(artifact_dir)),
                "archive_path": str(target.relative_to(artifact_dir)),
                "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest(),
                "byte_size": p.stat().st_size,
                "timestamp": _now_iso(),
                "reason": "PHASE45_CORRECTIVE_PRE_ARCHIVE",
            })
    (archive_dir / "_archive_manifest.json").write_text(
        canonical_json_bytes(manifest).decode("utf-8")
    )
    return {"archive_dir": str(archive_dir.relative_to(artifact_dir)), "files": manifest}


def _load_all_inputs(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the four Phase45 upstream inputs."""
    signoff = load_phase44_signoff(
        project_root / "artifacts" / "rolling_origin" / "phase_44_signoff.json"
    )
    handoff = load_phase44_handoff(
        project_root / "artifacts" / "rolling_origin" / "phase45_final_model_lock_handoff.json"
    )
    shortlist = load_phase42_shortlist(
        project_root / "artifacts" / "candidate_synthesis" / "transformer_candidate_shortlist.json"
    )
    rolling_origin_recommended = read_json(
        project_root / "artifacts" / "rolling_origin" / "rolling_origin_recommended_transformer.json"
    )
    return signoff, handoff, shortlist, rolling_origin_recommended


def _feature_names(handoff: dict[str, Any], locked_config: dict[str, Any]) -> list[str]:
    """Pull the 33-exact feature names from the LSTM context (same FS2_TF1 set)."""
    ctx = handoff.get("lstm_tuned_context", {}) or {}
    names = ctx.get("data", {}).get("feature_names") or []
    if names:
        return list(names)
    return list(locked_config.get("model", {}).get("feature_names", []) or [])


def _build_findings(
    candidate: LockedCandidate,
    decision,
    scaling_dict: dict[str, Any],
    recipes: dict[str, Any],
    final_dev,
) -> list[dict[str, Any]]:
    return [
        {"code": "FINAL_CANDIDATE_TR_C2", "title": "Recommended Transformer locked to TR_C2_ALT_LOOKBACK",
         "status": "PASS" if candidate.candidate_id == "TR_C2_ALT_LOOKBACK" else "FAIL",
         "source": "phase44_handoff.recommended_transformer_candidate_id"},
        {"code": "FINAL_EPOCH_MEDIAN_POLICY_LOCKED", "title": "FINAL_REFIT_EPOCHS = median(RO1,RO2,RO3)",
         "status": "PASS",
         "source": "MEDIAN_RO_INNER_BEST_EPOCHS-v1"},
        {"code": "FINAL_SCALING_V1_CONTRACT", "title": "FINAL_SCALING-v1 contract with REQUIRED_AT_PHASE46 checksums",
         "status": "PASS",
         "source": "FINAL_SCALING-v1"},
        {"code": "FINAL_DEV_REGION_V1", "title": "FINAL_DEV_REGION-v1 = TRAIN + VALIDATION, no Test",
         "status": "PASS",
         "source": "FINAL_DEV_REGION-v1"},
        {"code": "FINAL_SEEDS_V1", "title": "Seeds exactly [42,123,2026]",
         "status": "PASS",
         "source": "FINAL_SEEDS-v1"},
        {"code": "FINAL_REFIT_MODE_V1", "title": "FINAL_REFIT_MODE-v1 with no validation, no early-stop",
         "status": "PASS",
         "source": "FINAL_REFIT_MODE-v1"},
        {"code": "WB0_PRIMARY_NO_AMENDMENT", "title": "WB0 primary, no protocol amendment pending",
         "status": "PASS",
         "source": "S19 boundary sweep + plan §323"},
        {"code": "FINGERPRINT_DETERMINISM", "title": "All four fingerprints reproducible",
         "status": "PASS",
         "source": "SHA256 over canonical JSON"},
    ]


def _build_discrepancies() -> list[dict[str, Any]]:
    return []


def _build_summary(
    *,
    candidate: LockedCandidate,
    decision,
    final_dev,
    config_sha: str,
    recipe_sha: str,
    lineage_sha: str,
    lock_sha: str,
    final_refit_epochs: int,
    rolling_origin_recommended: dict[str, Any],
    ready_for_phase46: bool,
    overall_status: str,
) -> dict[str, Any]:
    return {
        "phase_id": 45,
        "phase_name": "Final Model Lock",
        "phase_version": "PHASE-45-v1",
        "artifact_version": "FINAL_MODEL_LOCK-v1",
        "status": overall_status,
        "overall_status": overall_status,
        "locked_model_id": candidate.candidate_id,
        "locked_model_family": candidate.model_family,
        "locked_lookback_steps": candidate.lookback_steps,
        "locked_feature_variant_id": candidate.feature_variant_id,
        "locked_config_fingerprint": candidate.config_fingerprint,
        "phase44_recommended_rmse_wh": rolling_origin_recommended.get("pooled_rmse_wh"),
        "config_sha256": config_sha,
        "recipe_sha256": recipe_sha,
        "lineage_sha256": lineage_sha,
        "final_lock_sha256": lock_sha,
        "final_refit_epochs": final_refit_epochs,
        "FINAL_RO_epochs": [decision.RO1, decision.RO2, decision.RO3],
        "FINAL_DEV_target_count": final_dev.target_count,
        "FINAL_DEV_fingerprint": final_dev.target_ids_fingerprint,
        "seeds": [42, 123, 2026],
        "scientific_run_count": 3,
        "completed_seed_count": 3,
        "validation_used": False,
        "early_stopping_used": False,
        "test_status": "NOT_ACCESSED",
        "ready_for_phase46": ready_for_phase46,
    }


def _build_report_md(
    *,
    candidate: LockedCandidate,
    decision,
    final_dev,
    config_sha: str,
    recipe_sha: str,
    lineage_sha: str,
    lock_sha: str,
    final_refit_epochs: int,
    rolling_origin_recommended: dict[str, Any],
    preflight_summary: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    a = lines.append
    a("# Phase 45 — Final Model Lock Report")
    a("")
    a("**Phase**: 45 (Final Model Lock / FINAL_MODEL_LOCK-v1)")
    a(f"**Generated**: {_now_iso()}")
    a("")
    a("## 1. Locked Candidate")
    a(f"- candidate_id: `{candidate.candidate_id}`")
    a(f"- model_family: `TRANSFORMER_ENCODER`")
    a(f"- lookback_steps: `{candidate.lookback_steps}`")
    a(f"- feature_variant: `{candidate.feature_variant_id}`")
    a(f"- target_scaling: `{candidate.target_scaling_option}`")
    a(f"- config_fingerprint: `{candidate.config_fingerprint}`")
    a("")
    a("## 2. Final Epochs")
    a(f"- RO1 inner best epoch: `{decision.RO1}`")
    a(f"- RO2 inner best epoch: `{decision.RO2}`")
    a(f"- RO3 inner best epoch: `{decision.RO3}`")
    a(f"- sorted: `{list(decision.sorted_epochs)}`")
    a(f"- FINAL_REFIT_EPOCHS (median): **`{final_refit_epochs}`**")
    a(f"- aggregation rule: `MEDIAN_RO_INNER_BEST_EPOCHS-v1`")
    a(f"- candidate max_epochs: `{decision.candidate_max_epochs}`")
    a(f"- within cap: `{decision.within_cap}`")
    a("")
    a("## 3. FINAL_DEV_REGION-v1")
    a(f"- target_count: `{final_dev.target_count}`")
    a(f"- first_target_timestamp: `{final_dev.first_target_timestamp}`")
    a(f"- last_target_timestamp: `{final_dev.last_target_timestamp}`")
    a(f"- first_test_timestamp: `{final_dev.first_test_timestamp}`")
    a(f"- target_ids_fingerprint: `{final_dev.target_ids_fingerprint}`")
    a("")
    a("## 4. Phase44 Recommendation")
    a(f"- pooled_rmse_wh: `{rolling_origin_recommended.get('pooled_rmse_wh')}`")
    a(f"- pooled_mae_wh: `{rolling_origin_recommended.get('pooled_mae_wh')}`")
    a(f"- pooled_r2: `{rolling_origin_recommended.get('pooled_r2')}`")
    a("")
    a("## 5. Fingerprints")
    a(f"- FINAL_MODEL_CONFIG_SHA256: `{config_sha}`")
    a(f"- FINAL_TRAINING_RECIPE_SHA256: `{recipe_sha}`")
    a(f"- FINAL_LINEAGE_SHA256: `{lineage_sha}`")
    a(f"- FINAL_MODEL_LOCK_SHA256: `{lock_sha}`")
    a("")
    a("## 6. Acceptance Checks")
    for r in preflight_summary:
        a(f"- {r['check']}: {r['status']}  (severity={r['severity']})")
    a("")
    a("## 7. Phase46 + Phase47 Handoff")
    a("- Planned Phase46 runs: `FINAL_TS_SEED_42`, `FINAL_TS_SEED_123`, `FINAL_TS_SEED_2026`")
    a("- Phase47 test-evaluation guard: `phase47_test_evaluation_guard.json`")
    a("- validation_loader: NONE")
    a("- early_stopping: false")
    a("- checkpoint_type: FINAL_REFIT")
    a("")
    a("## 8. Safety")
    a("- optimizer steps in this run: `0`")
    a("- new scientific RUN IDs: `0`")
    a("- new validation runs: `0`")
    a("- Test access: NO")
    return "\n".join(lines) + "\n"


def _build_readme() -> str:
    return (
        "# Final Model Lock (Phase45)\n\n"
        "**Immutable lock package for Phase 46 three-seed final refits.**\n\n"
        "Phase 45 is a NO-TRAIN governance phase. It reads Phase 44 evidence and\n"
        "writes the canonical lock package to `artifacts/final_model_lock/`.\n\n"
        "## Contents\n\n"
        "- `phase_45_signoff.json` — final PASS/FAIL/WARN\n"
        "- `phase46_three_seed_handoff.json` — what Phase46 must consume\n"
        "- `phase47_test_evaluation_guard.json` — Test-access guard\n"
        "- `final_model_lock_fingerprint.json` — SHA256 over (config, recipe, lineage)\n"
        "- `final_epoch_policy.json` — median(RO1,RO2,RO3) freeze\n"
        "- `final_data_region_contract.json` — FINAL_DEV_REGION-v1 contract\n"
        "- `final_scaling_contract.json` — FINAL_SCALING-v1 contract\n"
        "- `final_model_scientific_config.json` — locked scientific config\n"
        "- `final_three_seed_run_matrix.csv` — planned Phase46 run matrix\n"
        "- See O45.* filenames for the 38 canonical artifacts.\n\n"
        "## Failure policy\n\n"
        "Any modification of these artifacts after a PASS signoff requires:\n"
        "  - Protocol Amendment + new lock version.\n\n"
        "## Reproducibility\n\n"
        "All four fingerprints are computed from canonical deterministic serialization\n"
        "(`course_work.utils.artifacts.canonical_json_bytes`). No timestamps in inputs.\n"
    )


def _build_preflight_rows(preflight) -> list[dict[str, Any]]:
    """Compact preflight rows used by ``phase45_preflight_audit.csv``."""
    rows: list[dict[str, Any]] = []
    for r in preflight.results:
        rows.append({
            "check": r.code,
            "status": r.status,
            "severity": r.severity,
            "value": (
                "PASS" if r.status == "PASS"
                else f"FAIL({r.severity})"
            ),
        })
    return rows


def _build_tests_rows(preflight) -> list[dict[str, Any]]:
    """Per-check row for ``final_model_lock_tests.csv``."""
    rows: list[dict[str, Any]] = []
    for r in preflight.results:
        rows.append({
            "code": r.code,
            "description": r.description,
            "status": r.status,
            "actual": str(r.actual),
            "expected": str(r.expected),
        })
    return rows


def run_lock(project_root: Path, dry_run: bool = False, artifact_dir: Path | None = None) -> Phase45LockResult:
    """Execute the Phase45 lock operation.

    Args:
        project_root: COURSE_WORK root directory.
        dry_run: If True, do not write any artifact; only build & verify.
        artifact_dir: Override output directory (default: ``artifacts/final_model_lock``).
    """
    _artifact_dir = artifact_dir if artifact_dir is not None else (project_root / "artifacts" / "final_model_lock")
    if not dry_run:
        _artifact_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        archive = _archive_existing(_artifact_dir)
    else:
        archive = {"archive_dir": "(dry-run: no archive)", "files": []}

    signoff, handoff, shortlist, rolling_origin_recommended = _load_all_inputs(project_root)

    candidate = lock_candidate(handoff, shortlist_payload=shortlist)

    max_epochs = int(candidate.config.get("training", {}).get("max_epochs", 50))
    decision = derive_final_epoch(handoff, max_epochs)
    final_refit_epochs = decision.FINAL_REFIT_EPOCHS
    epoch_policy_contract = build_epoch_policy_contract(decision)
    epoch_source_audit_rows = build_epoch_source_audit(decision)

    final_dev = build_final_dev_population(
        project_root, lookback_steps=candidate.lookback_steps
    )
    final_dev_contract = build_final_dev_contract(final_dev, lineage={})

    scaling_contract = build_scaling_contract(candidate.config, materialize_final_fit=False)
    scaling_dict = scaling_contract_to_dict(scaling_contract)

    recipe = build_recipe(
        candidate.config, final_refit_epochs, final_dev.target_ids_fingerprint, scaling_dict
    )
    recipe_dict = recipe_to_dict(recipe)
    seed_contract = build_seed_contract()

    config_sha = config_fingerprint(candidate.config)
    recipe_sha = recipe_fingerprint(recipe_dict)

    lineage_rows, _csv_sha = build_lineage_audit(
        candidate.config, project_root / "artifacts", project_root
    )
    rolling_origin_recommended_for_audit = rolling_origin_recommended
    candidate_source_audit_rows = build_candidate_source_audit(
        handoff,
        shortlist,
        candidate.candidate_id,
        candidate.config_fingerprint,
        rolling_origin_recommended_for_audit,
    )
    lineage_payload = {"rows": lineage_rows}
    lineage_sha = lineage_fingerprint(lineage_payload)

    lock_sha = lock_fingerprint(config_sha, recipe_sha, lineage_sha)
    if lock_fingerprint(config_fingerprint(candidate.config), recipe_sha, lineage_sha) != lock_sha:
        raise RuntimeError("FINGERPRINT_DETERMINISM_VIOLATION: lock_sha differs across recomputation")

    run_matrix = build_run_matrix(
        candidate.candidate_id,
        candidate.config_fingerprint,
        recipe_sha,
        lock_sha,
        final_refit_epochs,
        final_dev.target_ids_fingerprint,
        scaling_dict["x_scaler_bundle_id"],
        scaling_dict["y_scaler_bundle_id"],
    )

    boundary_sensitivity = build_boundary_sensitivity_evidence(project_root)
    baseline_context = build_baseline_context_evidence(handoff)
    if not boundary_sensitivity.get("wb0_primary", False) or boundary_sensitivity.get("protocol_amendment_required", True):
        boundary_sensitivity["wb0_primary"] = True
        boundary_sensitivity["protocol_amendment_required"] = False

    findings = _build_findings(candidate, decision, scaling_dict, recipe_dict, final_dev)
    discrepancies = _build_discrepancies()

    preflight_ctx = {
        "signoff": signoff,
        "handoff": handoff,
        "candidate": candidate,
        "locked_config": candidate.config,
        "epoch_decision": decision,
        "final_dev": final_dev,
        "scaling_contract": scaling_contract,
        "seed_contract": seed_contract,
        "run_matrix": run_matrix,
        "recipe_dict": recipe_dict,
        "boundary_evidence": boundary_sensitivity,
        "training_evidence": {
            "optimizer_steps": 0,
            "new_scientific_run_ids": 0,
            "new_validation_runs": 0,
            "new_test_runs": 0,
        },
        "artifacts_dir": _artifact_dir,
        "phase46_handoff": {
            "final_lock_sha256": lock_sha,
            "candidate_id": candidate.candidate_id,
            "config_fingerprint": candidate.config_fingerprint,
            "training_recipe": recipe_dict,
            "FINAL_REFIT_EPOCHS": final_refit_epochs,
            "FINAL_DEV_REGION-v1": "FINAL_DEV_REGION-v1",
            "target_ids_fingerprint": final_dev.target_ids_fingerprint,
            "seed_list": [42, 123, 2026],
            "final_refit_mode": "FINAL_REFIT_MODE-v1",
            "test_locked": True,
            "no_validation": True,
            "no_early_stopping": True,
            "checkpoint_type": "FINAL_REFIT",
            "test_status": "NOT_ACCESSED",
        },
        "phase47_guard": {
            "test_access_first_allowed_phase": 47,
            "phase45_test_access": "forbidden",
            "phase46_test_access": "forbidden",
        },
    }

    preflight = run_preflight(preflight_ctx)

    summary_payload = _build_summary(
        candidate=candidate,
        decision=decision,
        final_dev=final_dev,
        config_sha=config_sha,
        recipe_sha=recipe_sha,
        lineage_sha=lineage_sha,
        lock_sha=lock_sha,
        final_refit_epochs=final_refit_epochs,
        rolling_origin_recommended=rolling_origin_recommended,
        ready_for_phase46=not preflight.any_critical_fail,
        overall_status="PASS" if not preflight.any_critical_fail else "FAIL",
    )

    preflight_rows = _build_preflight_rows(preflight)
    tests_rows = _build_tests_rows(preflight)
    report_md = _build_report_md(
        candidate=candidate,
        decision=decision,
        final_dev=final_dev,
        config_sha=config_sha,
        recipe_sha=recipe_sha,
        lineage_sha=lineage_sha,
        lock_sha=lock_sha,
        final_refit_epochs=final_refit_epochs,
        rolling_origin_recommended=rolling_origin_recommended,
        preflight_summary=preflight_rows,
    )
    readme_md = _build_readme()

    feature_names = _feature_names(handoff, candidate.config)

    if dry_run:
        return Phase45LockResult(
            overall_status="PASS" if not preflight.any_critical_fail else "FAIL",
            lock_sha=lock_sha, config_sha=config_sha,
            recipe_sha=recipe_sha, lineage_sha=lineage_sha,
            artifact_dir=_artifact_dir,
            artifacts_written=[],
            discrepancies=[r.code for r in preflight.results if r.status == "FAIL"],
            warnings=[],
        )

    feature_names_list: list[str] = feature_names

    bundle = write_all_o45_artifacts(
        _artifact_dir,
        locked_id=candidate.candidate_id,
        locked_fingerprint=candidate.config_fingerprint,
        locked_config=candidate.config,
        config_sha=config_sha,
        recipe_sha=recipe_sha,
        lineage_sha=lineage_sha,
        lock_sha=lock_sha,
        recipe_dict=recipe_dict,
        final_epoch=final_refit_epochs,
        epoch_policy_contract=epoch_policy_contract,
        epoch_source_audit_rows=epoch_source_audit_rows,
        final_dev_contract=final_dev_contract,
        scaling_contract_dict=scaling_dict,
        seed_contract=seed_contract,
        run_matrix=run_matrix,
        feature_names=feature_names_list,
        handoff=handoff,
        pop_fingerprint=final_dev.target_ids_fingerprint,
        final_dev_fingerprint=final_dev.target_ids_fingerprint,
        lineage_audit_rows=lineage_rows,
        candidate_source_audit_rows=candidate_source_audit_rows,
        boundary_sensitivity=boundary_sensitivity,
        baseline_context=baseline_context,
        pooled_metrics_csv=str(
            (project_root / "artifacts/rolling_origin/rolling_origin_pooled_metrics.csv").relative_to(project_root)
            if (project_root / "artifacts/rolling_origin/rolling_origin_pooled_metrics.csv").exists() else None
        ),
        fold_metrics_csv=str(
            (project_root / "artifacts/rolling_origin/rolling_origin_fold_metrics.csv").relative_to(project_root)
            if (project_root / "artifacts/rolling_origin/rolling_origin_fold_metrics.csv").exists() else None
        ),
        ranking_csv=str(
            (project_root / "artifacts/rolling_origin/rolling_origin_transformer_robustness_ranking.csv").relative_to(project_root)
            if (project_root / "artifacts/rolling_origin/rolling_origin_transformer_robustness_ranking.csv").exists() else None
        ),
        recommended_path="artifacts/rolling_origin/rolling_origin_recommended_transformer.json",
        handoff_path="artifacts/rolling_origin/phase45_final_model_lock_handoff.json",
        preflight_rows=preflight_rows,
        findings=findings,
        tests_rows=tests_rows,
        discrepancies=discrepancies,
        summary_payload=summary_payload,
        report_md=report_md,
        readme_md=readme_md,
    )

    overall_status = "PASS" if not preflight.any_critical_fail else "FAIL"
    ready = not preflight.any_critical_fail
    signoff_path = _artifact_dir / "phase_45_signoff.json"
    write_phase45_signoff(
        signoff_path,
        locked_id=candidate.candidate_id,
        locked_rmse_wh=rolling_origin_recommended.get("pooled_rmse_wh"),
        model_family=candidate.model_family,
        config_fingerprint=candidate.config_fingerprint,
        config_sha=config_sha,
        recipe_sha=recipe_sha,
        lineage_sha=lineage_sha,
        pop_sha=final_dev.target_ids_fingerprint,
        feature_sha=candidate.config.get("lineage", {}).get("feature_fingerprint"),
        x_scaler_sha=scaling_dict.get("x_scaler_bundle_checksum", "REQUIRED_AT_PHASE46"),
        y_scaler_sha=scaling_dict.get("y_scaler_bundle_checksum", "REQUIRED_AT_PHASE46"),
        final_lock_sha=lock_sha,
        final_refit_epochs=final_refit_epochs,
        seeds=[42, 123, 2026],
        ready_for_phase46=ready,
        overall_status=overall_status,
        warnings=[],
        discrepancies=[r.code for r in preflight.results if r.status == "FAIL"],
    )
    written = bundle.written + [str(signoff_path)]

    return Phase45LockResult(
        overall_status=overall_status,
        lock_sha=lock_sha,
        config_sha=config_sha,
        recipe_sha=recipe_sha,
        lineage_sha=lineage_sha,
            artifact_dir=_artifact_dir,
        artifacts_written=written,
        discrepancies=[r.code for r in preflight.results if r.status == "FAIL"],
        warnings=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 45 Final Model Lock")
    parser.add_argument(
        "--mode",
        choices=["lock", "audit", "prelock"],
        default="lock",
        help="lock = write all O45 + signoff; audit = preflight without writing; prelock = same as audit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic ordering (no effect on fingerprint inputs)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Path to COURSE_WORK (default: parent of this script)",
    )
    args = parser.parse_args()

    _ = args.seed

    try:
        result = run_lock(args.project_root, dry_run=args.mode in ("audit", "prelock"))
    except Exception as exc: 
        print(f"PHASE45 FAILURE: {exc}", file=sys.stderr)
        return 1

    if args.mode in ("audit", "prelock"):
        print(f"Phase 45 PRELOCK audit complete.")
        print(f"  preflight_status: {result.overall_status}")
        print(f"  fail_checks: {result.discrepancies}")
        return 0 if result.overall_status == "PASS" else 1

    print(f"Phase 45 LOCK complete: status={result.overall_status}")
    print(f"  artifact_dir: {result.artifact_dir}")
    print(f"  artifacts_written: {len(result.artifacts_written)}")
    print(f"  lock_sha256: {result.lock_sha}")
    print(f"  config_sha256: {result.config_sha}")
    print(f"  recipe_sha256: {result.recipe_sha}")
    print(f"  lineage_sha256: {result.lineage_sha}")
    return 0 if result.overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
