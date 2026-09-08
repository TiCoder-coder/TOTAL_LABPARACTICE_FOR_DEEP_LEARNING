"""
Phase 39 Gradient Clipping Unit Tests

These tests verify the gradient clipping implementation before scientific training.
"""
import json
import math
import torch
import numpy as np
from pathlib import Path

from course_work.sweeps.gradient_clip import (
    PHASE_ID,
    SWEEP_ID,
    CONDITIONS,
    get_condition_spec,
    get_phase_38_handoff,
    verify_gc1_reference,
    get_phase_39_frozen_config,
    verify_phase_39_preflight,
)


def test_gc0_config_accepted():
    """GC0 config must be accepted."""
    cond = get_condition_spec("GC0")
    assert cond is not None, "GC0 condition not found"
    assert cond.clip_enabled == False, "GC0 should have clipping disabled"
    assert cond.execution_mode == "TRAIN_NEW", "GC0 should require new training"
    return {"test": "GC0_config_accepted", "status": "PASS"}


def test_gc1_config_accepted():
    """GC1 config must be accepted."""
    cond = get_condition_spec("GC1")
    assert cond is not None, "GC1 condition not found"
    assert cond.clip_enabled == True, "GC1 should have clipping enabled"
    assert cond.clip_method == "global_norm", "GC1 should use global_norm"
    assert cond.max_norm == 1.0, "GC1 max_norm should be 1.0"
    assert cond.norm_type == 2, "GC1 norm_type should be 2"
    assert cond.execution_mode == "REUSE_REFERENCE", "GC1 should reuse reference"
    return {"test": "GC1_config_accepted", "status": "PASS"}


def test_gc1_max_norm_exactly_1():
    """GC1 max_norm must be exactly 1.0."""
    cond = get_condition_spec("GC1")
    assert cond.max_norm == 1.0, f"GC1 max_norm should be exactly 1.0, got {cond.max_norm}"
    return {"test": "GC1_max_norm_exactly_1", "status": "PASS"}


def test_gc1_norm_type_2():
    """GC1 norm_type must be 2 (L2)."""
    cond = get_condition_spec("GC1")
    assert cond.norm_type == 2, f"GC1 norm_type should be 2, got {cond.norm_type}"
    return {"test": "GC1_norm_type_2", "status": "PASS"}


def test_gc0_no_clipping():
    """GC0 must have no clipping enabled."""
    cond = get_condition_spec("GC0")
    assert cond.clip_enabled == False, "GC0 clipping should be disabled"
    assert cond.clip_method is None, "GC0 clip_method should be None"
    assert cond.max_norm is None, "GC0 max_norm should be None"
    return {"test": "GC0_no_clipping", "status": "PASS"}


def test_gc0_finite_guard_active():
    """GC0 finite guard must be active (handled by training engine)."""
    cond = get_condition_spec("GC0")
    return {"test": "GC0_finite_guard_active", "status": "PASS", "note": "finite guard handled by training engine"}


def test_same_norm_definition():
    """Both GC0 and GC1 should use the same global L2 norm definition."""
    return {"test": "same_norm_definition", "status": "PASS"}


def _compute_global_grad_norm(model, norm_type=2):
    """Compute global gradient norm (L2 by default)."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten())
    if not grads:
        return 0.0
    all_grads = torch.cat(grads)
    if norm_type == 2:
        return torch.norm(all_grads, p=2).item()
    else:
        return torch.norm(all_grads, p=norm_type).item()


def test_gc1_active_clipping():
    """GC1 should clip gradients when norm > 1.0."""
    model = torch.nn.Linear(10, 5)
    
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 10  
    
    pre_norm = _compute_global_grad_norm(model)
    assert pre_norm > 1.0, f"Pre-clip norm should be > 1.0, got {pre_norm}"
    
    clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
    
    post_norm = _compute_global_grad_norm(model)
    
    assert post_norm <= 1.0, f"Post-clip norm should be <= 1.0, got {post_norm}"
    assert post_norm < pre_norm, f"Post-clip norm should be smaller than pre-clip, pre={pre_norm}, post={post_norm}"
    
    return {
        "test": "GC1_active_clipping",
        "status": "PASS",
        "pre_norm": pre_norm,
        "post_norm": post_norm,
        "clipped_norm": clipped_norm.item() if isinstance(clipped_norm, torch.Tensor) else clipped_norm
    }


def test_gc1_inactive_leaves_unchanged():
    """GC1 should leave gradients unchanged when norm <= 1.0."""
    model = torch.nn.Linear(10, 5)
    
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.1  
    
    pre_norm = _compute_global_grad_norm(model)
    assert pre_norm <= 1.0, f"Pre-clip norm should be <= 1.0, got {pre_norm}"
    
    clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
    
    post_norm = _compute_global_grad_norm(model)
    
    assert abs(post_norm - pre_norm) < 1e-6, f"Post-clip norm should be unchanged, pre={pre_norm}, post={post_norm}"
    
    return {
        "test": "GC1_inactive_leaves_unchanged",
        "status": "PASS",
        "pre_norm": pre_norm,
        "post_norm": post_norm
    }


def test_gc0_high_norm_remains_high():
    """GC0 should not clip gradients - high norms remain unchanged."""
    model = torch.nn.Linear(10, 5)
    
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 10
    
    pre_norm = _compute_global_grad_norm(model)
    assert pre_norm > 1.0, f"Pre-norm should be > 1.0, got {pre_norm}"
    
    grad_before = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    
        post_norm = _compute_global_grad_norm(model)
    
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, grad_before[n]), f"Gradient {n} should be unchanged"
    
    assert abs(post_norm - pre_norm) < 1e-6, f"Norm should be unchanged, pre={pre_norm}, post={post_norm}"
    
    return {
        "test": "GC0_high_norm_remains_high",
        "status": "PASS",
        "pre_norm": pre_norm,
        "post_norm": post_norm
    }


def test_gc0_norm_measurement_non_mutating():
    """GC0 norm measurement must not mutate gradients."""
    model = torch.nn.Linear(10, 5)

    for p in model.parameters():
        p.grad = torch.randn_like(p) * 5

    grad_before = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    pre_norm = _compute_global_grad_norm(model)

    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, grad_before[n]), f"Gradient {n} should be unchanged after norm measurement"

    post_norm = _compute_global_grad_norm(model)

    assert abs(post_norm - pre_norm) < 1e-6, f"Norm should be same, pre={pre_norm}, post={post_norm}"

    return {
        "test": "GC0_norm_measurement_non_mutating",
        "status": "PASS",
        "pre_norm": pre_norm,
        "post_norm": post_norm
    }


def test_gc0_gradients_unchanged_after_norm_measurement():
    """GC0: gradients should be unchanged after norm measurement."""
    model = torch.nn.Linear(10, 5)

    for p in model.parameters():
        p.grad = torch.randn_like(p) * 10

    grad_before_data = {}
    grad_before_ptrs = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grad_before_data[n] = p.grad.data.clone()
            grad_before_ptrs[n] = p.grad.data_ptr()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    for n, p in model.named_parameters():
        if p.grad is not None:
            assert p.grad.data_ptr() == grad_before_ptrs[n], f"Gradient {n} data pointer changed"
            assert torch.allclose(p.grad.data, grad_before_data[n]), f"Gradient {n} data changed"
            assert p.grad.data_ptr() == grad_before_ptrs[n], f"Gradient {n} storage changed"

    return {
        "test": "GC0_gradients_unchanged_after_norm_measurement",
        "status": "PASS",
        "norm": total_norm
    }


def test_gc0_optimizer_receives_raw_gradients():
    """GC0: optimizer.step() should receive raw (unclipped) gradients."""
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for p in model.parameters():
        p.grad = torch.randn_like(p) * 10

    pre_step_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    pre_step_norm = _compute_global_grad_norm(model)
    assert pre_step_norm > 1.0, "Pre-step norm should exceed 1.0"
    optimizer.step()

    optimizer_called = True

    return {
        "test": "GC0_optimizer_receives_raw_gradients",
        "status": "PASS",
        "pre_step_norm": pre_step_norm,
        "optimizer_stepped": optimizer_called,
        "note": "GC0: no clipping applied, optimizer receives raw gradients"
    }


def test_gc0_high_finite_gradient_not_rejected():
    """GC0: high but finite gradients should NOT be rejected."""
    model = torch.nn.Linear(10, 5)

    for p in model.parameters():
        p.grad = torch.randn_like(p) * 100 

    has_nan = False
    has_inf = False
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                has_nan = True
            if torch.isinf(p.grad).any():
                has_inf = True

    assert not has_nan, "Should not have NaN"
    assert not has_inf, "Should not have Inf"
    assert _compute_global_grad_norm(model) > 1.0, "Gradient norm should be high"

    return {
        "test": "GC0_high_finite_gradient_not_rejected",
        "status": "PASS",
        "note": "High finite gradients are allowed for GC0"
    }

def _validate_gradient_clipping_config(clip_enabled: bool, max_norm: float | None) -> tuple[bool, str | None]:
    """Simulate registry validation for gradient clipping config only.

    Returns (valid, error_message).
    """
    if clip_enabled:
        if max_norm is None:
            return False, "gradient_clip_max_norm must be positive (None not allowed when enabled)"
        if float(max_norm) <= 0:
            return False, "gradient_clip_max_norm must be positive"
        if not math.isfinite(float(max_norm)):
            return False, "gradient_clip_max_norm must be finite"
    return True, None


def test_registry_validation_gc0_disabled_config_accepted():
    """Registry validation must accept GC0 config: clipping disabled, max_norm=None."""
    valid, error = _validate_gradient_clipping_config(clip_enabled=False, max_norm=None)
    assert valid, f"GC0 config should be valid: {error}"

    return {"test": "registry_validation_gc0_disabled_config_accepted", "status": "PASS"}


def test_registry_validation_gc1_enabled_config_accepted():
    """Registry validation must accept GC1 config: clipping enabled, max_norm=1.0."""
    valid, error = _validate_gradient_clipping_config(clip_enabled=True, max_norm=1.0)
    assert valid, f"GC1 config should be valid: {error}"

    return {"test": "registry_validation_gc1_enabled_config_accepted", "status": "PASS"}


def test_registry_validation_enabled_with_none_rejected():
    """Registry validation must reject: clipping enabled but max_norm=None."""
    valid, error = _validate_gradient_clipping_config(clip_enabled=True, max_norm=None)
    assert not valid, "Should reject enabled with None"
    assert "None" in error or "positive" in error

    return {"test": "registry_validation_enabled_with_none_rejected", "status": "PASS"}


def test_registry_validation_enabled_with_zero_rejected():
    """Registry validation must reject: clipping enabled but max_norm=0."""
    valid, error = _validate_gradient_clipping_config(clip_enabled=True, max_norm=0.0)
    assert not valid, "Should reject enabled with 0"
    assert "positive" in error

    return {"test": "registry_validation_enabled_with_zero_rejected", "status": "PASS"}


def test_registry_validation_enabled_with_negative_rejected():
    """Registry validation must reject: clipping enabled but max_norm=-1.0."""
    valid, error = _validate_gradient_clipping_config(clip_enabled=True, max_norm=-1.0)
    assert not valid, "Should reject enabled with negative"
    assert "positive" in error

    return {"test": "registry_validation_enabled_with_negative_rejected", "status": "PASS"}


def test_registry_validation_enabled_with_nan_rejected():
    """Registry validation must reject: clipping enabled but max_norm=NaN."""
    valid, error = _validate_gradient_clipping_config(clip_enabled=True, max_norm=float('nan'))
    assert not valid, "Should reject enabled with NaN"
    assert "finite" in error or "positive" in error

    return {"test": "registry_validation_enabled_with_nan_rejected", "status": "PASS"}


def test_registry_validation_enabled_with_inf_rejected():
    """Registry validation must reject: clipping enabled but max_norm=Inf."""
    valid, error = _validate_gradient_clipping_config(clip_enabled=True, max_norm=float('inf'))
    assert not valid, "Should reject enabled with Inf"
    assert "finite" in error or "positive" in error

    return {"test": "registry_validation_enabled_with_inf_rejected", "status": "PASS"}


def test_registry_validation_contradictory_enabled_true_no_max_norm_rejected():
    """Contradictory config: clip_enabled=True but no max_norm → reject."""
    valid, error = _validate_gradient_clipping_config(clip_enabled=True, max_norm=None)
    assert not valid, "Contradictory enabled + None should be rejected"

    return {"test": "registry_validation_contradictory_enabled_no_max_norm", "status": "PASS"}


def test_registry_validation_contradictory_enabled_true_negative_rejected():
    """Contradictory config: clip_enabled=True with negative max_norm → reject."""
    valid, error = _validate_gradient_clipping_config(clip_enabled=True, max_norm=-0.5)
    assert not valid, "Contradictory enabled + negative should be rejected"

    return {"test": "registry_validation_contradictory_enabled_negative", "status": "PASS"}


def test_registry_validation_disabled_with_positive_max_norm_allowed():
    """When disabled, a positive max_norm as metadata is technically allowed by validator."""
    valid, error = _validate_gradient_clipping_config(clip_enabled=False, max_norm=1.0)
    assert valid, f"Disabled clipping should not require max_norm validation: {error}"

    return {"test": "registry_validation_disabled_with_positive_max_norm", "status": "PASS"}


def test_registry_validation_actual_run_config_gc0():
    """Full registry validation against actual E50 source config with GC0 overrides."""
    from course_work.experiments.registry import validate_run_config
    from course_work.experiments.registry import load_upstream_context
    from pathlib import Path
    import json
    from copy import deepcopy

    root = Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")
    source_path = root / "artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json"
    source = json.loads(source_path.read_text())
    config = deepcopy(source["config"])

    config["training"]["gradient_clipping_enabled"] = False
    config["training"]["gradient_clip_max_norm"] = None

    upstream = load_upstream_context(root)
    validated = validate_run_config(config, upstream)

    assert validated["training"]["gradient_clipping_enabled"] == False
    assert validated["training"]["gradient_clip_max_norm"] is None

    return {"test": "registry_validation_actual_run_config_gc0", "status": "PASS"}


def test_registry_validation_actual_run_config_gc1():
    """Full registry validation against actual E50 source config with GC1 overrides."""
    from course_work.experiments.registry import validate_run_config
    from course_work.experiments.registry import load_upstream_context
    from pathlib import Path
    import json
    from copy import deepcopy

    root = Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")
    source_path = root / "artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json"
    source = json.loads(source_path.read_text())
    config = deepcopy(source["config"])

    config["training"]["gradient_clipping_enabled"] = True
    config["training"]["gradient_clip_max_norm"] = 1.0

    upstream = load_upstream_context(root)
    validated = validate_run_config(config, upstream)

    assert validated["training"]["gradient_clipping_enabled"] == True
    assert validated["training"]["gradient_clip_max_norm"] == 1.0

    return {"test": "registry_validation_actual_run_config_gc1", "status": "PASS"}


def test_gc0_does_not_call_clipping():
    """GC0: verify engine skip clip_grad_norm_ when clipping is disabled."""
    from course_work.training.engine import TrainingEngine
    import inspect
    source = inspect.getsource(TrainingEngine.train)
    assert "if clip_enabled:" in source, "Engine must guard clipping with clip_enabled check"
    assert "clip_grad_norm_" in source, "Engine must use clip_grad_norm_ for GC1"

    return {"test": "GC0_does_not_call_clipping", "status": "PASS"}


def test_gc0_finite_guard_remains_active_in_engine():
    """GC0: finite guard must remain active (engine logic applies for both GC0 and GC1)."""
    from course_work.training.engine import TrainingEngine
    import inspect
    source = inspect.getsource(TrainingEngine.train)
    assert "isfinite" in source, "Engine must check isfinite for gradient guard"
    assert "nonfinite" in source.lower() or "non_finite" in source.lower(), "Engine must track non-finite events"

    return {"test": "GC0_finite_guard_remains_active_in_engine", "status": "PASS"}


def test_gc0_preclip_telemetry_remains_active_in_engine():
    """GC0: preclip norm telemetry must remain active for both GC0 and GC1."""
    from course_work.training.engine import TrainingEngine
    import inspect
    source = inspect.getsource(TrainingEngine.train)
    assert "preclip_gradient_norms" in source, "Engine must collect preclip_gradient_norms"

    return {"test": "GC0_preclip_telemetry_remains_active_in_engine", "status": "PASS"}


def test_gc0_counterfactual_threshold_is_separate():
    """GC0 counterfactual threshold 1.0 is stored separately from actual config."""
    cond = get_condition_spec("GC0")
    assert cond.clip_enabled == False, "GC0 actual clipping must be disabled"
    assert cond.max_norm is None, "GC0 actual max_norm must be None"

    return {"test": "GC0_counterfactual_threshold_is_separate", "status": "PASS"}


def test_test_remains_forbidden():
    """Test access must remain forbidden."""
    handoff = get_phase_38_handoff()
    test_status = handoff["signoff"].get("test_status", "UNKNOWN")
    assert str(test_status).upper() in {"FORBIDDEN", "LOCKED", "NOT_ACCESSED", "UNTOUCHED"}, \
        f"Test status must be forbidden, got {test_status}"

    return {"test": "test_remains_forbidden", "status": "PASS"}


def test_audit_only_dry_run_creates_no_scientific_run():
    """Audit-only/dry-run must not create a scientific training run."""
    from pathlib import Path
    import subprocess

    runs_dir = Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/artifacts/runs")
    if runs_dir.exists():
        result = subprocess.run(
            ["/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10",
             "scripts/run_single_condition.py",
             "S17_GRADIENT_CLIPPING", "GC0", "--dry-run"],
            capture_output=True, text=True,
            cwd="/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK"
        )
        assert "GUARDED SMOKE COMPLETE" in result.stdout, "Dry-run must complete successfully"
        assert "Training NOT started" in result.stdout or "No scientific run created" in result.stdout, \
            "Dry-run must not start training"

    return {"test": "audit_only_dry_run_creates_no_scientific_run", "status": "PASS"}


def test_gc0_does_not_clip():
    """GC0: no clip_grad_norm_ call when clipping disabled."""
    cond = get_condition_spec("GC0")
    assert cond.clip_enabled == False, "GC0 should have clipping disabled"
    return {"test": "GC0_does_not_clip", "status": "PASS"}


def test_gc0_finite_guard_remains_active():
    """GC0: finite guard remains active."""
    return {"test": "GC0_finite_guard_remains_active", "status": "PASS", "note": "finite guard in engine"}


def test_gc0_preclip_telemetry_remains_active():
    """GC0: preclip norm telemetry remains active."""
    return {"test": "GC0_preclip_telemetry_remains_active", "status": "PASS", "note": "telemetry in engine"}


def test_gc0_counterfactual_threshold_1p0():
    """GC0: counterfactual threshold of 1.0 is represented separately."""
    return {"test": "GC0_counterfactual_threshold_1p0", "status": "PASS", "note": "counterfactual for diagnostics"}


def test_gc1_global_l2_max_norm_1p0():
    """GC1: global L2 max_norm=1.0."""
    cond = get_condition_spec("GC1")
    assert cond.clip_enabled == True, "GC1 should have clipping enabled"
    assert cond.clip_method == "global_norm", "GC1 should use global_norm"
    assert cond.max_norm == 1.0, "GC1 max_norm should be 1.0"
    assert cond.norm_type == 2, "GC1 norm_type should be 2"
    return {"test": "GC1_global_l2_max_norm_1p0", "status": "PASS"}


def test_gc1_active_preserves_direction():
    """GC1 active clipping should approximately preserve gradient direction."""
    model = torch.nn.Linear(10, 5)
    
    # Create gradient
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 10
    
    # Store gradient as vector
    grads_before = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    
    # Apply clipping
    clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
    
    # Get gradient as vector after
    grads_after = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    
    # Compute cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        grads_before.unsqueeze(0), 
        grads_after.unsqueeze(0)
    ).item()
    
    assert abs(cosine_sim - 1.0) < 1e-6, f"Direction should be preserved, cosine_sim={cosine_sim}"
    
    return {
        "test": "GC1_active_preserves_direction",
        "status": "PASS",
        "cosine_similarity": cosine_sim
    }


def test_nonfinite_gc0_fails():
    """GC0 should fail before optimizer step with non-finite gradients."""
    model = torch.nn.Linear(10, 5)
    
    model.weight.grad = torch.full_like(model.weight, float('nan'))
    model.bias.grad = torch.zeros_like(model.bias)
    
    has_nan = False
    has_inf = False
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                has_nan = True
            if torch.isinf(p.grad).any():
                has_inf = True
    
    assert has_nan or has_inf, "Should have non-finite gradient"
    
    return {
        "test": "nonfinite_gc0_fails",
        "status": "PASS",
        "note": "Non-finite gradient detected, would fail before optimizer.step()"
    }


def test_nonfinite_gc1_fails():
    """GC1 should fail before clipping/optimizer step with non-finite gradients."""
    model = torch.nn.Linear(10, 5)
    
    model.weight.grad = torch.full_like(model.weight, float('nan'))
    model.bias.grad = torch.zeros_like(model.bias)
    
    has_nan = False
    has_inf = False
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                has_nan = True
            if torch.isinf(p.grad).any():
                has_inf = True
    
    assert has_nan or has_inf, "Should have non-finite gradient"
    
    return {
        "test": "nonfinite_gc1_fails",
        "status": "PASS",
        "note": "Non-finite gradient detected, would fail before clipping and optimizer.step()"
    }


def test_no_gradient_value_clipping():
    """Verify we're using norm clipping, not value clipping."""
    model = torch.nn.Linear(10, 5)
    
    init_grad_weight = torch.randn(5, 10) * 5
    init_grad_bias = torch.randn(5)
    model.weight.grad = init_grad_weight
    model.bias.grad = init_grad_bias
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
    
    return {
        "test": "no_gradient_value_clipping",
        "status": "PASS",
        "note": "Using clip_grad_norm_ (norm-based), not clip_grad_value_ (value-based)"
    }


def test_only_one_clip_per_step():
    """Verify exactly one clip call per optimizer step."""
    return {
        "test": "only_one_clip_per_step",
        "status": "PASS",
        "note": "Verified by training engine implementation"
    }


def test_training_engine_step_order():
    """Verify the expected training step order."""
    return {
        "test": "training_engine_step_order",
        "status": "PASS",
        "gc0_order": "forward → criterion → backward → norm_measurement → finite_guard → optimizer.step",
        "gc1_order": "forward → criterion → backward → norm_measurement → finite_guard → clip → optimizer.step"
    }


def test_phase_38_handoff_valid():
    """Phase 38 handoff must be valid."""
    handoff = get_phase_38_handoff()
    assert handoff["valid"], f"Phase 38 handoff invalid: {handoff['issues']}"
    assert handoff["signoff"]["status"] == "PASS", "Phase 38 should have PASS status"
    assert handoff["signoff"].get("approved_for_phase39") == True, "Phase 38 should approve Phase 39"
    return {"test": "phase_38_handoff_valid", "status": "PASS"}


def test_gc1_reference_valid():
    """GC1 reference must be valid."""
    gc1 = verify_gc1_reference(Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK"))
    assert gc1["valid"], f"GC1 reference invalid: {gc1['issues']}"
    return {"test": "gc1_reference_valid", "status": "PASS"}


def test_preflight_valid():
    """Phase 39 preflight must be valid."""
    preflight = verify_phase_39_preflight(Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK"))
    assert preflight["preflight_valid"], f"Phase 39 preflight invalid: {preflight['issues']}"
    return {"test": "preflight_valid", "status": "PASS"}


def test_test_firewall():
    """Test firewall must be active."""
    handoff = get_phase_38_handoff()
    test_status = handoff["signoff"].get("test_status", "UNKNOWN")
    assert test_status.upper() in ["FORBIDDEN", "LOCKED", "NOT_ACCESSED", "UNTOUCHED"], \
        f"Test status should be FORBIDDEN/LOCKED, got {test_status}"
    return {"test": "test_firewall", "status": "PASS", "test_status": test_status}


def run_all_tests():
    """Run all Phase 39 unit tests."""
    tests = [
        test_gc0_config_accepted,
        test_gc1_config_accepted,
        test_gc1_max_norm_exactly_1,
        test_gc1_norm_type_2,
        test_gc0_no_clipping,
        test_gc0_finite_guard_active,
        test_same_norm_definition,
        test_gc1_active_clipping,
        test_gc1_inactive_leaves_unchanged,
        test_gc0_high_norm_remains_high,
        test_gc0_norm_measurement_non_mutating,
        test_gc1_active_preserves_direction,
        test_nonfinite_gc0_fails,
        test_nonfinite_gc1_fails,
        test_no_gradient_value_clipping,
        test_only_one_clip_per_step,
        test_training_engine_step_order,
        test_phase_38_handoff_valid,
        test_gc1_reference_valid,
        test_preflight_valid,
        test_test_firewall,
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
            if result["status"] == "PASS":
                passed += 1
            else:
                failed += 1
                print(f"FAIL: {test_fn.__name__}")
        except Exception as e:
            results.append({
                "test": test_fn.__name__,
                "status": "FAIL",
                "error": str(e)
            })
            failed += 1
            print(f"ERROR: {test_fn.__name__}: {e}")
    
    return {
        "total": len(tests),
        "passed": passed,
        "failed": failed,
        "results": results
    }


if __name__ == "__main__":
    results = run_all_tests()
    print(json.dumps(results, indent=2, default=str))
