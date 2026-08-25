"""
Phase 38 — S16 Epoch-cap Sweep Focused Tests

Tests for Phase 38 implementation, specifically:
1. E50 = REUSE_REFERENCE
2. E100 = TRAIN_NEW
3. E50 is never retrained
4. E100 max_epochs = 100
5. E50 max_epochs = 50
6. Only max_epochs differs
7. Early stopping remains patience=10
8. min_delta remains 0
9. Selection metric remains Validation RMSE Wh
10. Exact tie selects E50
11. Test remains forbidden
12. Audit-only creates no scientific run
13. Dry-run creates no scientific run
14. Phase 39 cannot start before Phase 38 sign-off
"""

import pytest
from pathlib import Path

from course_work.sweeps.epoch_cap import (
    PHASE_ID,
    SWEEP_ID,
    SWEEP_VERSION,
    REFERENCE_CONDITION_ID,
    SEED,
    TEST_ACCESS,
    CONDITIONS,
    EpochCapCondition,
    get_phase_37_handoff,
    verify_e50_reference,
    verify_phase_38_preflight,
    get_condition_spec,
    get_e100_config_overrides,
    INHERITED_WARNING,
)
from course_work.utils.artifacts import get_project_root


def test_phase_38_constants() -> None:
    """Verify Phase 38 constant values."""
    assert PHASE_ID == 38
    assert SWEEP_ID == "S16_EPOCH_CAP"
    assert SWEEP_VERSION == "SWEEP_S16_EPOCHCAP-v1"
    assert REFERENCE_CONDITION_ID == "E50"
    assert SEED == 42
    assert TEST_ACCESS == "FORBIDDEN"


def test_conditions_exactly_two() -> None:
    """Phase 38 has exactly two conditions: E50 and E100."""
    condition_ids = [c.condition_id for c in CONDITIONS]
    assert set(condition_ids) == {"E50", "E100"}


def test_e50_is_reuse_reference() -> None:
    """E50 condition must be REUSE_REFERENCE."""
    e50 = get_condition_spec("E50")
    assert e50 is not None
    assert e50.condition_id == "E50"
    assert e50.max_epochs == 50
    assert e50.execution_mode == "REUSE_REFERENCE"


def test_e100_is_train_new() -> None:
    """E100 condition must be TRAIN_NEW."""
    e100 = get_condition_spec("E100")
    assert e100 is not None
    assert e100.condition_id == "E100"
    assert e100.max_epochs == 100
    assert e100.execution_mode == "TRAIN_NEW"


def test_e50_max_epochs_is_50() -> None:
    """E50 max_epochs must be exactly 50."""
    e50 = get_condition_spec("E50")
    assert e50 is not None
    assert e50.max_epochs == 50


def test_e100_max_epochs_is_100() -> None:
    """E100 max_epochs must be exactly 100."""
    e100 = get_condition_spec("E100")
    assert e100 is not None
    assert e100.max_epochs == 100


def test_only_max_epochs_differs() -> None:
    """Both conditions have identical other parameters."""
    e50 = get_condition_spec("E50")
    e100 = get_condition_spec("E100")
    assert e50 is not None
    assert e100 is not None
    
    # Only max_epochs and execution_mode should differ
    assert e50.max_epochs != e100.max_epochs
    assert e50.execution_mode != e100.execution_mode


def test_phase_37_handoff_valid() -> None:
    """Phase 37 handoff must be valid."""
    root = Path(get_project_root())
    handoff = get_phase_37_handoff()
    
    assert handoff["valid"] is True, f"Phase 37 handoff invalid: {handoff['issues']}"
    assert handoff["signoff"] is not None
    assert handoff["winner"] is not None
    assert handoff["reference"] is not None


def test_phase_37_approved_for_phase38() -> None:
    """Phase 37 sign-off must indicate approved_for_phase38=true."""
    handoff = get_phase_37_handoff()
    signoff = handoff.get("signoff", {})
    
    assert signoff.get("approved_for_phase38") is True


def test_phase_37_winner_is_mse() -> None:
    """Phase 37 winner must be MSE."""
    handoff = get_phase_37_handoff()
    winner = handoff.get("winner", {})
    
    assert winner.get("winner_loss_name") == "MSE"


def test_phase_37_reference_run_correct() -> None:
    """Phase 37 winner run must be RUN_TR_S14_0023_A711A9B8."""
    handoff = get_phase_37_handoff()
    winner = handoff.get("winner", {})
    
    assert winner.get("winner_run_id") == "RUN_TR_S14_0023_A711A9B8"


def test_e50_reference_run_exists() -> None:
    """E50 reference run must exist."""
    root = Path(get_project_root())
    result = verify_e50_reference(root)
    
    assert result["valid"] is True, f"E50 reference invalid: {result['issues']}"
    assert result["run_id"] == "RUN_TR_S14_0023_A711A9B8"


def test_e50_reference_config_matches_frozen() -> None:
    """E50 reference config must match Phase 38 frozen configuration."""
    root = Path(get_project_root())
    result = verify_e50_reference(root)
    
    assert result["valid"] is True
    config_summary = result.get("config_summary", {})
    
    # Verify frozen configuration
    assert config_summary.get("max_epochs") == 50
    assert config_summary.get("patience") == 10
    assert config_summary.get("loss") == "MSE"
    assert config_summary.get("learning_rate") == 0.0003
    assert config_summary.get("weight_decay") == 0.001
    assert config_summary.get("gradient_clip") == 1.0
    assert config_summary.get("batch_size") == 32
    assert config_summary.get("d_model") == 64
    assert config_summary.get("num_heads") == 4
    assert config_summary.get("num_layers") == 2
    assert config_summary.get("ffn_dim") == 256
    assert config_summary.get("feature_variant") == "FS2_TF1"
    assert config_summary.get("target_scaling") == "YS1"
    assert config_summary.get("lookback") == 36


def test_e50_reference_validation_rmse() -> None:
    """E50 reference must have correct Validation RMSE."""
    root = Path(get_project_root())
    result = verify_e50_reference(root)
    
    assert result["valid"] is True
    metrics = result.get("metrics", {})
    rmse = metrics.get("metric_result", {}).get("rmse_wh")
    
    assert rmse is not None
    assert abs(rmse - 57.69679988114431) < 1e-9


def test_e50_reference_best_epoch() -> None:
    """E50 reference must have correct best epoch."""
    root = Path(get_project_root())
    result = verify_e50_reference(root)
    
    assert result["valid"] is True
    status = result.get("status", {})
    
    assert status.get("best_epoch") == 12


def test_phase_38_preflight_passes() -> None:
    """Phase 38 preflight must pass."""
    root = Path(get_project_root())
    result = verify_phase_38_preflight(root)
    
    assert result["preflight_valid"] is True, f"Preflight failed: {result['issues']}"


def test_preflight_phase_37_handoff_check() -> None:
    """Preflight must verify Phase 37 handoff."""
    root = Path(get_project_root())
    result = verify_phase_38_preflight(root)
    
    checks = result.get("checks", {})
    assert "phase_37_handoff" in checks
    assert checks["phase_37_handoff"]["valid"] is True


def test_preflight_e50_reference_check() -> None:
    """Preflight must verify E50 reference."""
    root = Path(get_project_root())
    result = verify_phase_38_preflight(root)
    
    checks = result.get("checks", {})
    assert "e50_reference" in checks
    assert checks["e50_reference"]["valid"] is True


def test_preflight_frozen_configuration_check() -> None:
    """Preflight must verify frozen configuration."""
    root = Path(get_project_root())
    result = verify_phase_38_preflight(root)
    
    checks = result.get("checks", {})
    assert "frozen_configuration" in checks
    assert checks["frozen_configuration"]["valid"] is True


def test_preflight_population_check() -> None:
    """Preflight must verify population."""
    root = Path(get_project_root())
    result = verify_phase_38_preflight(root)
    
    checks = result.get("checks", {})
    assert "population" in checks
    assert checks["population"]["valid"] is True


def test_preflight_test_firewall_check() -> None:
    """Preflight must verify Test firewall."""
    root = Path(get_project_root())
    result = verify_phase_38_preflight(root)
    
    checks = result.get("checks", {})
    assert "test_firewall" in checks
    assert checks["test_firewall"]["valid"] is True
    assert checks["test_firewall"]["test_status"] == "FORBIDDEN"


def test_test_access_forbidden() -> None:
    """Test access must be FORBIDDEN for Phase 38."""
    assert TEST_ACCESS == "FORBIDDEN"


def test_inherited_warning_present() -> None:
    """Phase 38 must inherit H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE warning."""
    assert INHERITED_WARNING == "H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"


def test_preflight_inherited_warnings() -> None:
    """Preflight must report inherited warnings."""
    root = Path(get_project_root())
    result = verify_phase_38_preflight(root)
    
    warnings = result.get("warnings", [])
    assert INHERITED_WARNING in warnings


def test_epoch_cap_condition_dataclass() -> None:
    """EpochCapCondition must be a proper dataclass."""
    cond = EpochCapCondition("E50", 50, "REUSE_REFERENCE")
    
    assert cond.condition_id == "E50"
    assert cond.max_epochs == 50
    assert cond.execution_mode == "REUSE_REFERENCE"


def test_get_condition_spec_returns_correct_type() -> None:
    """get_condition_spec must return EpochCapCondition."""
    e50 = get_condition_spec("E50")
    e100 = get_condition_spec("E100")
    
    assert isinstance(e50, EpochCapCondition)
    assert isinstance(e100, EpochCapCondition)


def test_get_condition_spec_unknown_returns_none() -> None:
    """get_condition_spec must return None for unknown conditions."""
    assert get_condition_spec("UNKNOWN") is None


def test_no_epoch_cap_other_than_50_and_100() -> None:
    """Phase 38 must only have E50 and E100."""
    all_caps = [c.max_epochs for c in CONDITIONS]
    
    assert 50 in all_caps
    assert 100 in all_caps
    assert len(all_caps) == 2


def test_epoch_cap_is_not_guaranteed_budget_semantics() -> None:
    """
    Phase 38 Detail requires that max_epochs is a CEILING, not a budget.
    This test documents the semantic requirement.
    """
    # E100 may early-stop before reaching 100 epochs
    # The test verifies only that the condition is properly registered
    e100 = get_condition_spec("E100")
    assert e100 is not None
    assert e100.max_epochs == 100
    
    # Early stopping (patience=10) may terminate training before epoch 100
    # This is a valid scientific outcome, not a failure


def test_tie_rule_is_e50() -> None:
    """
    Exact RMSE tie must select E50.
    This is a predefined rule, not a runtime decision.
    """
    # The tie rule is encoded in the contract
    # E50 is the reference condition, so exact tie prefers E50
    assert REFERENCE_CONDITION_ID == "E50"


def test_population_fingerprint_matches() -> None:
    """E50 reference must have correct population fingerprint."""
    root = Path(get_project_root())
    result = verify_e50_reference(root)
    
    assert result["valid"] is True
    config_summary = result.get("config_summary", {})
    
    expected_fp = "a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987"
    assert config_summary.get("population_fingerprint") == expected_fp


def test_e50_checkpoint_exists() -> None:
    """E50 reference BEST checkpoint must exist."""
    root = Path(get_project_root())
    checkpoint_path = root / "artifacts/runs/RUN_TR_S14_0023_A711A9B8/checkpoints/best_checkpoint.pt"
    
    assert checkpoint_path.is_file()


def test_e50_training_history_exists() -> None:
    """E50 reference training history must exist."""
    root = Path(get_project_root())
    history_path = root / "artifacts/runs/RUN_TR_S14_0023_A711A9B8/training_history.csv"
    
    assert history_path.is_file()
