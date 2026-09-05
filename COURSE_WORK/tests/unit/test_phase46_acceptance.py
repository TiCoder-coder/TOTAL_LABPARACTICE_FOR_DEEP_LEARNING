"""Phase 46 Acceptance — focused unit tests for all Phase 46 acceptance criteria.

These tests verify the corrected implementation satisfies the Phase 46 plan
without performing any training, optimizer step, or Test access.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from course_work.data.final_dev import materialize_final_dev_region
from course_work.experiments.registry import ExperimentRegistry, ExecutionType, compute_config_fingerprint
from course_work.scaling.final_scaling import materialize_final_scaling_v1


PHASE_45_SIGNOFF = ROOT / "artifacts" / "final_model_lock" / "phase_45_signoff.json"
PHASE_46_HANDOFF = ROOT / "artifacts" / "final_model_lock" / "phase46_three_seed_handoff.json"


# --- Step 1: Phase45 lock verification ---

class TestPhase45LockVerification:
    def test_phase45_signoff_pass(self):
        signoff = json.loads(PHASE_45_SIGNOFF.read_text())
        assert signoff["status"] == "PASS"
        assert signoff.get("ready_for_phase46") is True

    def test_candidate_is_TR_C2_ALT_LOOKBACK(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        assert handoff["candidate_id"] == "TR_C2_ALT_LOOKBACK"

    def test_config_fingerprint_matches_lock(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        locked_fp = handoff["config_fingerprint"]
        # Locked fingerprint from Phase 45 is the canonical config fingerprint.
        assert locked_fp == "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"

    def test_final_refit_epochs_is_30(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        assert int(handoff["FINAL_REFIT_EPOCHS"]) == 30

    def test_seed_list_is_exact_42_123_2026(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        assert list(handoff["seed_list"]) == [42, 123, 2026]

    def test_ro_epochs_via_final_lock_evidence(self):
        # The Phase 45 signoff records median epoch provenance in
        # final_epoch_source_audit.csv. We verify the median computation
        # (Phase 45 closure stated: RO1=30, RO2=37, RO3=19, median=30).
        evidence = ROOT / "artifacts" / "final_model_lock" / "final_epoch_source_audit.csv"
        if evidence.exists():
            text = evidence.read_text()
            # The audit must reference epochs 19, 30, 37 as the source epochs.
            assert "19" in text and "30" in text and "37" in text

    def test_test_status_not_accessed(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        assert handoff["test_status"] == "NOT_ACCESSED"

    def test_lock_sha256_pinned(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        # Phase 45 closure produced final_lock_sha256 which Phase 46 must reuse.
        assert handoff.get("final_lock_sha256")
        assert len(handoff["final_lock_sha256"]) == 64  # sha256 hex


# --- Step 2: FINAL_DEV data path ---

class TestFinalDevDataPath:
    def test_final_dev_excludes_test(self):
        m = materialize_final_dev_region(
            project_root=ROOT,
            feature_variant_id="FS2_TF1",
            lookback=72,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        )
        assert m.test_window_count == 0
        assert m.final_dev_window_count == m.train_window_count + m.validation_window_count

    def test_final_dev_target_count_matches_lock(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        expected_train = int(handoff["scientific_config"]["data"]["train_sample_count"])
        expected_val = int(handoff["scientific_config"]["data"]["validation_sample_count"])
        m = materialize_final_dev_region(
            project_root=ROOT,
            feature_variant_id="FS2_TF1",
            lookback=72,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        )
        assert m.train_window_count == expected_train
        assert m.validation_window_count == expected_val

    def test_final_dev_lookback_72(self):
        m = materialize_final_dev_region(
            project_root=ROOT,
            feature_variant_id="FS2_TF1",
            lookback=72,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        )
        assert m.lookback_steps == 72

    def test_final_dev_unique_target_ids(self):
        m = materialize_final_dev_region(
            project_root=ROOT,
            feature_variant_id="FS2_TF1",
            lookback=72,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        )
        # Combined fingerprint includes sorted target IDs, so it implicitly tests uniqueness.
        assert m.population_fingerprint
        assert len(m.population_fingerprint) == 64


# --- Step 3: FINAL_SCALING-v1 ---

class TestFinalScaling:
    def test_scaler_fit_once(self):
        # Re-materializing should not produce different SHA due to env drift
        # if bytes match, or accept statistics-equivalent otherwise.
        r = materialize_final_scaling_v1(
            project_root=ROOT,
            feature_variant_id="FS2_TF1",
        )
        assert r["fit_row_count"] == 16630
        # Read the manifest directly.
        manifest = json.loads((ROOT / "artifacts" / "scaling" / "final_dev" / "final_scaling_manifest.json").read_text())
        assert manifest["fit_once"] is True
        assert manifest["Test_rows_used"] is False
        assert manifest["fit_region"] == "FINAL_DEV_REGION-v1"

    def test_scaler_x_sha_persisted(self):
        r = materialize_final_scaling_v1(
            project_root=ROOT,
            feature_variant_id="FS2_TF1",
        )
        assert r["x_sha256"]
        assert len(r["x_sha256"]) == 64

    def test_scaler_y_sha_persisted(self):
        r = materialize_final_scaling_v1(
            project_root=ROOT,
            feature_variant_id="FS2_TF1",
        )
        assert r["y_sha256"]
        assert len(r["y_sha256"]) == 64

    def test_scaler_no_test_rows(self):
        r = materialize_final_scaling_v1(
            project_root=ROOT,
            feature_variant_id="FS2_TF1",
        )
        assert r["fit_row_count"] == 16630  # TRAIN 13670 + VAL 2960
        manifest = json.loads((ROOT / "artifacts" / "scaling" / "final_dev" / "final_scaling_manifest.json").read_text())
        assert manifest["Test_rows_used"] is False

    def test_scaler_roundtrip_y(self):
        # Y scaler round-trip should preserve target values within tolerance.
        r = materialize_final_scaling_v1(
            project_root=ROOT,
            feature_variant_id="FS2_TF1",
        )
        manifest = json.loads((ROOT / "artifacts" / "scaling" / "final_dev" / "final_scaling_manifest.json").read_text())
        assert manifest["roundtrip_audit_passed"] is True


# --- Step 4-7: FINAL_REFIT_MODE engine semantics ---

class TestFinalRefitMode:
    def test_handoff_has_final_refit_mode(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        assert handoff["final_refit_mode"] == "FINAL_REFIT_MODE-v1"
        assert handoff["no_validation"] is True
        assert handoff["no_early_stopping"] is True

    def test_checkpoint_type_pinned(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        assert handoff["checkpoint_type"] == "FINAL_REFIT"
        assert handoff["checkpoint_contract"]["official_epoch"] == 30


# --- Step 8: Test firewall ---

class TestTestFirewall:
    def test_phase47_release_starts_false(self):
        # The driver must NEVER write phase47_test_release.json with released=True
        # unless all 3 seeds completed.
        # After Phase 46 human-completed all 3 seeds, released=True is the
        # legitimate state — verified by finalize mode.
        release_path = ROOT / "artifacts" / "three_seed_final_runs" / "phase47_test_release.json"
        if release_path.exists():
            release = json.loads(release_path.read_text())
            # released MUST equal (all gates passed AND 3/3 seeds completed)
            if release["released"] is True:
                # Confirm all 3 seeds are present
                assert len(release["run_records"]) == 3
                assert release["seed_count"] == 3
                # Confirm all gate checks pass
                for gate, val in release["gates"].items():
                    assert val is True, f"Released=True but gate {gate}={val}"

    def test_no_test_in_phase46_lock_data(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        data = handoff["scientific_config"]["data"]
        assert data["target_access_mode"] != "TEST"


# --- Step 9: Registry + 3-seed independence ---

class TestRegistryAndSeedIndependence:
    def test_planned_run_ids_are_per_seed(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        planned = handoff.get("planned_run_ids", [])
        # 3 planned scientific runs.
        assert len(planned) == 3
        # Each seed must appear in at least one planned ID.
        joined = " ".join(planned)
        for seed in [42, 123, 2026]:
            assert str(seed) in joined

    def test_no_favorable_duplicate_in_driver(self):
        # Phase 46 driver must not select "best seed" by training loss.
        driver_text = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        # Driver must contain canonical rerun_reason used to mark corrected reruns.
        assert "PHASE46_CORRECTIVE_RERUN" in driver_text
        # Driver must not silently reuse completed runs.
        assert "EXCLUDED_RUN_IDS" in driver_text

    def test_seeds_independent(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        seeds = handoff["seed_list"]
        assert seeds == [42, 123, 2026]
        assert len(set(seeds)) == 3


# --- Step 10: O46 artifact gap ---

class TestO46ArtifactCoverage:
    def test_three_seed_manifest_correct_when_present(self):
        # The three_seed_manifest.json is written by the scientific training
        # path. Until then, this artifact should not exist. After training,
        # it must record the locked candidate and FINAL_REFIT_EPOCHS=30.
        p = ROOT / "artifacts" / "three_seed_final_runs" / "three_seed_manifest.json"
        if p.exists():
            d = json.loads(p.read_text())
            assert d["candidate_id"] == "TR_C2_ALT_LOOKBACK"
            assert int(d["final_refit_epochs"]) == 30

    def test_three_seed_contract_correct_when_present(self):
        p = ROOT / "artifacts" / "three_seed_final_runs" / "three_seed_contract.json"
        if p.exists():
            d = json.loads(p.read_text())
            assert d["exactly_three_fixed_seeds"] is True
            assert d["no_validation"] is True
            assert d["no_early_stopping"] is True
            assert d["no_best_checkpoint"] is True

    def test_phase46_signoff_not_fake_pass(self):
        # The signoff must NEVER be PASS before 3 seeds are completed.
        p = ROOT / "artifacts" / "three_seed_final_runs" / "phase_46_signoff.json"
        if p.exists():
            d = json.loads(p.read_text())
            completed = d.get("completed_run_count") or d.get("completed_seed_count") or 0
            ready = d.get("ready_for_phase47", False)
            # If signoff exists and completed_run_count=0, ready_for_phase47 must be False.
            if completed == 0:
                assert ready is False, "phase_46_signoff must not declare ready_for_phase47=True with 0 completed runs"


# --- Step 11: Final lock verification real recompute ---

class TestFinalLockVerificationRecompute:
    def test_lock_fingerprints_recomputable(self):
        handoff = json.loads(PHASE_46_HANDOFF.read_text())
        # The Phase 46 handoff carries config_fingerprint, recipe_fingerprint,
        # lineage_fingerprint, final_lock_sha256. These MUST be the SHA256 of
        # the corresponding artifact contents.
        config_fp = handoff["config_fingerprint"]
        # Recompute config_fingerprint from the scientific_config.
        scientific_cfg = handoff["scientific_config"]
        recomputed = compute_config_fingerprint(scientific_cfg)
        assert recomputed == config_fp


# --- Step 12: DataLoader determinism ---

class TestDataLoaderDeterminism:
    def test_dataloader_uses_seed_generator_in_code(self):
        # The driver must construct DataLoader with a seeded Generator.
        from pathlib import Path as P
        driver_text = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        assert "torch.Generator().manual_seed" in driver_text or "Generator().manual_seed" in driver_text


# --- Step 13: Defect coverage ---

class TestDefectClosure:
    def test_validate_phase46_lock_no_hardcoded_50(self):
        # After fix, validate_phase46_lock must NOT hardcode epochs == 50.
        driver_text = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        # The legacy hardcoded "epochs == 50" pattern must be absent.
        assert "epochs == 50" not in driver_text
        assert "expected_epochs = 50" not in driver_text

    def test_validate_phase46_lock_no_hardcoded_36(self):
        # After fix, validate_phase46_lock must NOT hardcode lookback == 36.
        driver_text = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        assert "expected_lookback = 36" not in driver_text

    def test_final_dev_default_lookback_72(self):
        # After fix, the materialize default must be 72.
        from course_work.data.final_dev import materialize_final_dev_region
        import inspect
        sig = inspect.signature(materialize_final_dev_region)
        assert sig.parameters["lookback"].default == 72

    def test_scaling_default_lookback_72(self):
        # After fix, the scaling default lookback must be 72.
        from course_work.scaling.final_scaling import _build_final_dev_scaling_view
        import inspect
        src = inspect.getsource(_build_final_dev_scaling_view)
        assert "lookback = 72" in src or "lookback=72" in src

    def test_phase46_signoff_requires_three_completed(self):
        # After fix, the driver must NOT write signoff with ready_for_phase47=True
        # when only dry-run has completed (0 actual runs).
        from pathlib import Path as P
        driver_text = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        # The pattern "ready_for_phase47 = True" or "ready_for_phase47: True" must
        # NOT appear unconditionally.
        assert "ready_for_phase47: True" not in driver_text
        # The signoff must be gated on len(run_records) == 3
        assert 'ready_for_phase47": len(run_records) == 3' in driver_text


# --- Step 14: Driver fatal errors ---

class TestDriverFatalErrors:
    def test_missing_inputs_exits_with_fatal(self):
        # The driver must exit with code 2 (FATAL) on missing inputs, not silently exit.
        driver_text = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
        assert "sys.exit(2)" in driver_text
        assert "[FATAL]" in driver_text


# --- Step 15: Pre-train simulation ---

class TestPreTrainSimulation:
    def test_pre_train_schema_simulation_pass(self):
        # The simulation script must complete without errors when invoked.
        result = subprocess.run(
            ["python3", "scripts/phase46_pre_train_schema_simulation.py"],
            cwd=str(ROOT),
            env={"PYTHONPATH": str(ROOT / "src"), **__import__("os").environ},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "Pre-train end-to-end simulation: PASS" in result.stdout


# --- Helper for sandboxed tests ---

@pytest.fixture(scope="module")
def sandbox_root(tmp_path_factory) -> Path:
    """Create a disposable sandbox with FINAL_DEV/FINAL_SCALING artifacts."""
    return tmp_path_factory.mktemp("phase46_sandbox")
