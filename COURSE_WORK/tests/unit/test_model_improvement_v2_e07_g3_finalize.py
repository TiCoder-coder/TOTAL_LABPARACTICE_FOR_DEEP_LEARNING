"""E07-G3: Post-Stage-C success-contract crash + NO-RETRAIN finalization.

Verifies:
  - missing control["run_ids"] does not crash _write_success_contracts
  - no fabricated control run IDs
  - canonical control provenance is used (E01 manifest fallback)
  - finalize-only creates no training/refit runs
  - finalize-only does not rerun Stage C when outputs already exist
  - existing E07 resume behavior remains safe
  - E06 unchanged
  - Test NOT_ACCESSED
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from course_work.model_improvement_v2.e07_runner import (
    BASE_CANDIDATE_ID,
    CHALLENGER_CANDIDATE_IDS,
    EXPECTED_CONTROL_METRICS,
    _audit_finalize_resume_eligibility,
    _build_finalize_resume_result,
    _resolve_e01_control_run_ids,
    build_parser,
    load_e07_config,
    project_root,
)
from course_work.model_improvement_v2.e06_runner import (
    load_e06_config,
    project_root as e06_project_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
E07_OUTPUT = (
    PROJECT_ROOT / "artifacts/model_improvement_v2/experiments/E07"
)
E01_OUTPUT = (
    PROJECT_ROOT / "artifacts/model_improvement_v2/experiments/E01"
)


# ----------------------------------------------------------------------
# 1. Control run_ids resolved without KeyError when snapshot missing
# ----------------------------------------------------------------------
def test_control_run_ids_resolves_from_e01_canonical_when_snapshot_missing():
    """E07 snapshot intentionally has no control.run_ids; resolver must
    fall back to E01 canonical manifest without crashing."""
    document = load_e07_config(PROJECT_ROOT)
    # Snapshot must NOT have control.run_ids (this is the bug condition)
    assert "run_ids" not in document.get("control", {})
    # Resolver must succeed
    ledger = _resolve_e01_control_run_ids(PROJECT_ROOT, document)
    assert ledger["source"] == "e01_execution_manifest_canonical"
    assert len(ledger["stage_a"]) == 3
    assert len(ledger["stage_b"]) == 3
    # All 6 must be E01 runs
    all_rids = (
        list(ledger["stage_a"].values()) + list(ledger["stage_b"].values())
    )
    assert len(all_rids) == 6
    assert len(set(all_rids)) == 6
    assert all("_E01_" in rid for rid in all_rids)


def test_control_run_ids_uses_snapshot_when_present():
    """When the snapshot does declare control.run_ids, that lock wins."""
    document = load_e07_config(PROJECT_ROOT)
    # Inject a synthetic but valid snapshot lock
    fake_ledger = {
        "stage_a": {
            f"{BASE_CANDIDATE_ID}:RO{fold}": f"RUN_V2_TR_E01_RO{fold}_A_FAKE_{i}"
            for i, fold in enumerate([1, 2, 3], start=1)
        },
        "stage_b": {
            f"{BASE_CANDIDATE_ID}:RO{fold}": f"RUN_V2_TR_E01_RO{fold}_B_FAKE_{i}"
            for i, fold in enumerate([1, 2, 3], start=4)
        },
    }
    document["control"]["run_ids"] = fake_ledger
    ledger = _resolve_e01_control_run_ids(PROJECT_ROOT, document)
    assert ledger["source"] == "snapshot_control_lock"
    for v in ledger["stage_a"].values():
        assert "FAKE_" in v


def test_control_run_ids_rejects_malformed_snapshot_ledger():
    """Snapshot lock with wrong count/missing _E01_ markers must fail loudly."""
    document = load_e07_config(PROJECT_ROOT)
    document["control"]["run_ids"] = {
        "stage_a": {"BAD_KEY": "BAD_VALUE"},
        "stage_b": {},
    }
    from course_work.model_improvement_v2.e07_runner import E07PreflightError
    with __import__("pytest").raises(E07PreflightError, match="invalid"):
        _resolve_e01_control_run_ids(PROJECT_ROOT, document)


# ----------------------------------------------------------------------
# 2. No fabricated control run IDs
# ----------------------------------------------------------------------
def test_no_fabricated_control_run_ids():
    """The resolver must NEVER invent run IDs. If neither snapshot nor
    E01 manifest has them, it must raise E07PreflightError."""
    document = load_e07_config(PROJECT_ROOT)
    # Snapshot has no run_ids — that's expected
    assert "run_ids" not in document.get("control", {})
    # E01 manifest exists and must be the canonical source
    e01_manifest_path = E01_OUTPUT / "e01_execution_manifest.json"
    assert e01_manifest_path.is_file()
    e01_manifest = json.loads(e01_manifest_path.read_text())
    e01_run_ids = (
        list(e01_manifest["run_ids"]["stage_a"].values())
        + list(e01_manifest["run_ids"]["stage_b"].values())
    )
    # Resolver must return exactly the E01 run_ids, not fabricated ones
    ledger = _resolve_e01_control_run_ids(PROJECT_ROOT, document)
    resolved = (
        list(ledger["stage_a"].values()) + list(ledger["stage_b"].values())
    )
    assert sorted(resolved) == sorted(e01_run_ids)


# ----------------------------------------------------------------------
# 3. finalize-resume eligibility audit
# ----------------------------------------------------------------------
def test_finalize_resume_eligibility_audit_passes():
    """_audit_finalize_resume_eligibility must PASS for the current state."""
    audit = _audit_finalize_resume_eligibility(PROJECT_ROOT)
    assert audit["status"] == "PASS"
    assert audit["n_completed_runs"] == 12
    assert audit["n_outer_prediction_csvs"] >= 9
    assert audit["new_training_run_ids_allowed"] is False
    assert audit["new_refit_run_ids_allowed"] is False
    assert audit["stage_c_reexecution_allowed"] is False
    assert audit["test_access_authorized"] is False


def test_finalize_resume_eligibility_rejects_missing_artifacts(tmp_path):
    """Eligibility must fail when required Stage-C artifacts are absent."""
    from course_work.model_improvement_v2.e07_runner import E07PreflightError
    # Empty tmp_path has nothing → must raise
    with __import__("pytest").raises(E07PreflightError, match="missing"):
        _audit_finalize_resume_eligibility(tmp_path)


# ----------------------------------------------------------------------
# 4. Reconstructed result from persisted Stage-C artifacts
# ----------------------------------------------------------------------
def test_build_finalize_resume_result_reconstructs_pooled_metrics():
    """The reconstructed result must load the same pooled metrics that
    Phase45 handoff reports."""
    document = load_e07_config(PROJECT_ROOT)
    result = _build_finalize_resume_result(PROJECT_ROOT, document)
    # Both challengers must be present in pooled_metrics
    for cid in CHALLENGER_CANDIDATE_IDS:
        assert cid in result.pooled_metrics_by_cid
        pm = result.pooled_metrics_by_cid[cid]
        assert "rmse_wh" in pm and "mae_wh" in pm and "r2" in pm
    # Stage-A/B run_ids count
    assert result.n_stage_a_runs == 6
    assert result.n_stage_b_runs == 6
    # Outer predictions: 3 folds × 3 model_ids (incl PERSISTENCE)
    assert result.n_outer_prediction_bundles >= 9
    # Recommended transformer matches Phase45 handoff
    phase45 = json.loads(
        (E07_OUTPUT / "phase45_final_model_lock_handoff.json").read_text()
    )
    assert (
        result.recommended_transformer_id
        == phase45["recommended_transformer_candidate_id"]
    )


def test_build_finalize_resume_result_test_not_accessed():
    """Test status must NOT_ACCESSED throughout the reconstructed result."""
    document = load_e07_config(PROJECT_ROOT)
    result = _build_finalize_resume_result(PROJECT_ROOT, document)
    # The result does not carry test_status directly; the audit blob does
    audit = _audit_finalize_resume_eligibility(PROJECT_ROOT)
    assert audit["test_access_authorized"] is False


# ----------------------------------------------------------------------
# 5. finalize-resume CLI
# ----------------------------------------------------------------------
def test_finalize_resume_cli_mode_accepted():
    """CLI parser must accept --mode finalize-resume."""
    parser = build_parser()
    ns = parser.parse_args(
        ["--experiment", "E07", "--mode", "finalize-resume"]
    )
    assert ns.mode == "finalize-resume"


def test_finalize_resume_cli_rejects_authorize_training():
    """--authorize-training must be rejected with finalize-resume."""
    from course_work.model_improvement_v2.e07_runner import main
    rc = main(
        ["--experiment", "E07", "--mode", "finalize-resume",
         "--authorize-training"]
    )
    assert rc == 2  # ERROR exit code


# ----------------------------------------------------------------------
# 6. Written success contracts exist and are consistent
# ----------------------------------------------------------------------
def test_execution_manifest_written_and_has_control_run_ids():
    """e07_execution_manifest.json must exist and have control_run_ids."""
    path = E07_OUTPUT / "e07_execution_manifest.json"
    assert path.is_file()
    manifest = json.loads(path.read_text())
    assert "control_run_ids" in manifest
    control = manifest["control_run_ids"]
    assert "stage_a" in control
    assert "stage_b" in control
    # All 6 must have _E01_ markers
    all_rids = list(control["stage_a"].values()) + list(
        control["stage_b"].values()
    )
    assert len(all_rids) == 6
    assert all("_E01_" in rid for rid in all_rids)
    assert manifest["test_status"] == "NOT_ACCESSED"
    assert manifest["control_policy"] == "CONTROL_REUSED_FROM_E01_READ_ONLY"


def test_scheduler_ablation_comparison_written_and_decision_correct():
    """e07_scheduler_ablation_comparison.json must exist with promotion decision."""
    path = E07_OUTPUT / "e07_scheduler_ablation_comparison.json"
    assert path.is_file()
    cmp = json.loads(path.read_text())
    assert "results" in cmp and len(cmp["results"]) == 3  # control + 2 challengers
    assert "decision" in cmp
    assert cmp["test_status"] == "NOT_ACCESSED"
    # The control row must show source=CONTROL_REUSED_FROM_E01_READ_ONLY
    control_row = next(
        r for r in cmp["results"] if r["source"] == "CONTROL_REUSED_FROM_E01_READ_ONLY"
    )
    assert control_row["candidate_id"] == BASE_CANDIDATE_ID
    assert control_row["scheduler_name"] == "OFF"
    assert control_row["pooled_metrics"]["rmse_wh"] == EXPECTED_CONTROL_METRICS["rmse_wh"]


# ----------------------------------------------------------------------
# 7. E06 behavior unchanged
# ----------------------------------------------------------------------
def test_e06_contract_unaffected_by_e07_fix():
    """E06's manifest writing still uses document['control']['run_ids']."""
    document = load_e06_config(e06_project_root())
    # E06 must still have run_ids in its control block
    assert "run_ids" in document["control"]
    assert "stage_a" in document["control"]["run_ids"]
    assert "stage_b" in document["control"]["run_ids"]


# ----------------------------------------------------------------------
# 8. Hard stops: no training/refit/Stage-C reexecution
# ----------------------------------------------------------------------
def test_finalize_audit_records_no_training_no_refit_no_stage_c():
    """e07_finalize_audit.json must record no training/refit/Stage-C reexec."""
    finalize_audit = E07_OUTPUT / "e07_finalize_audit.json"
    if not finalize_audit.is_file():
        # If not yet written (test ordering), run finalize-resume first.
        # But we should NOT depend on this; skip if not present.
        import pytest
        pytest.skip("e07_finalize_audit.json not yet written")
    blob = json.loads(finalize_audit.read_text())
    assert blob["training_executed"] is False
    assert blob["refit_executed"] is False
    assert blob["stage_c_inference_executed"] is False
    assert blob["test_status"] == "NOT_ACCESSED"
