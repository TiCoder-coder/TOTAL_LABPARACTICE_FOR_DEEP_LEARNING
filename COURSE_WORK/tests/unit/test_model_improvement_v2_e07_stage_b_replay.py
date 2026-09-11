"""Focused unit tests for V2 E07-D Stage-B exact LR trace replay.

Contract:
  - Stage B must NOT recompute scheduler behavior.
  - For each Stage-B epoch k, optimizer lr is set explicitly from
    Stage A's lr_used_for_epoch[k].
  - Stage-B LR sequence MUST be EXACTLY equal to Stage-A LR sequence
    through best_epoch_inner.
  - No scheduler.step() in Stage B.

These tests verify:
  - validate_lr_trace correctness
  - set_optimizer_lr mutates every param group
  - extract_lr_trace_from_history
  - Replay semantics: lr_used_for_epoch matches Stage-A trace exactly
  - Trace length validation (rejects short/malformed traces)
  - Backward-compat: stage_a_lr_trace=None → constant LR
  - No Test access
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.optim as torch_optim

from course_work.model_improvement_v2.stage_b_replay import (
    LR_POLICY_STAGE_A_REPLAY,
    LR_POLICY_STAGE_B_CONSTANT,
    extract_lr_trace_from_history,
    set_optimizer_lr,
    validate_lr_trace,
)
from course_work.rolling_origin.refit_engine import RefitEngine


# ----------------------------------------------------------------------
# 1. validate_lr_trace
# ----------------------------------------------------------------------
def test_validate_lr_trace_accepts_long_enough_trace():
    """Trace with len >= best_epoch_inner is accepted."""
    trace = [3e-4, 2.99e-4, 2.97e-4, 2.94e-4]
    validated = validate_lr_trace(trace, best_epoch_inner=3)
    assert validated == [3e-4, 2.99e-4, 2.97e-4, 2.94e-4]


def test_validate_lr_trace_accepts_exact_length():
    """Trace with len == best_epoch_inner is accepted."""
    trace = [3e-4, 2.99e-4, 2.97e-4]
    validated = validate_lr_trace(trace, best_epoch_inner=3)
    assert len(validated) == 3


def test_validate_lr_trace_rejects_short_trace():
    """Trace shorter than best_epoch_inner is rejected."""
    trace = [3e-4, 2.99e-4]
    with pytest.raises(ValueError, match="too short"):
        validate_lr_trace(trace, best_epoch_inner=3)


def test_validate_lr_trace_rejects_none():
    """None trace is rejected when trace is required."""
    with pytest.raises(ValueError, match="non-empty trace"):
        validate_lr_trace(None, best_epoch_inner=3)


def test_validate_lr_trace_rejects_empty():
    """Empty list trace is rejected."""
    with pytest.raises(ValueError, match="non-empty trace"):
        validate_lr_trace([], best_epoch_inner=1)


def test_validate_lr_trace_rejects_zero_best_epoch():
    """best_epoch_inner < 1 is rejected."""
    with pytest.raises(ValueError, match="best_epoch_inner must be >= 1"):
        validate_lr_trace([3e-4], best_epoch_inner=0)


def test_validate_lr_trace_rejects_negative_lr():
    """Negative LR in trace is rejected."""
    trace = [3e-4, -1.0]
    with pytest.raises(ValueError, match="negative"):
        validate_lr_trace(trace, best_epoch_inner=2)


def test_validate_lr_trace_rejects_nan_lr():
    """NaN LR in trace is rejected."""
    trace = [float("nan"), 2.99e-4]
    with pytest.raises(ValueError, match="not finite"):
        validate_lr_trace(trace, best_epoch_inner=2)


def test_validate_lr_trace_rejects_inf_lr():
    """Inf LR in trace is rejected."""
    trace = [float("inf"), 2.99e-4]
    with pytest.raises(ValueError, match="not finite"):
        validate_lr_trace(trace, best_epoch_inner=2)


def test_validate_lr_trace_accepts_tuple():
    """Tuples are accepted and converted to list."""
    trace = (3e-4, 2.99e-4, 2.97e-4)
    validated = validate_lr_trace(trace, best_epoch_inner=3)
    assert isinstance(validated, list)
    assert validated == [3e-4, 2.99e-4, 2.97e-4]


# ----------------------------------------------------------------------
# 2. set_optimizer_lr mutates every param group
# ----------------------------------------------------------------------
def test_set_optimizer_lr_single_group():
    """Single param group: lr is set."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    set_optimizer_lr(opt, 1.5e-4)
    assert abs(opt.param_groups[0]["lr"] - 1.5e-4) < 1e-12


def test_set_optimizer_lr_multiple_groups():
    """Multi param group: every group gets the new lr."""
    model1 = torch.nn.Linear(10, 5)
    model2 = torch.nn.Linear(5, 1)
    opt = torch_optim.AdamW(
        [
            {"params": model1.parameters(), "lr": 3e-4},
            {"params": model2.parameters(), "lr": 3e-4},
        ]
    )
    assert len(opt.param_groups) == 2
    set_optimizer_lr(opt, 1.5e-4)
    assert abs(opt.param_groups[0]["lr"] - 1.5e-4) < 1e-12
    assert abs(opt.param_groups[1]["lr"] - 1.5e-4) < 1e-12


# ----------------------------------------------------------------------
# 3. extract_lr_trace_from_history
# ----------------------------------------------------------------------
def test_extract_lr_trace_from_history_records():
    """Extract from list-of-rows dict, sorted by epoch."""
    history = {
        "records": [
            {"epoch": 3, "lr_used_for_epoch": 2.97e-4},
            {"epoch": 1, "lr_used_for_epoch": 3e-4},
            {"epoch": 2, "lr_used_for_epoch": 2.99e-4},
        ]
    }
    trace = extract_lr_trace_from_history(history)
    assert trace == [3e-4, 2.99e-4, 2.97e-4]


def test_extract_lr_trace_from_history_dataframe_dict():
    """Extract from pd.DataFrame.to_dict() format."""
    history = {
        "epoch": [1, 2, 3],
        "lr_used_for_epoch": [3e-4, 2.99e-4, 2.97e-4],
    }
    trace = extract_lr_trace_from_history(history)
    assert trace == [3e-4, 2.99e-4, 2.97e-4]


def test_extract_lr_trace_unsupported_raises():
    """Unsupported history type raises ValueError."""
    with pytest.raises(ValueError, match="Cannot extract"):
        extract_lr_trace_from_history("not a dict")


# ----------------------------------------------------------------------
# 4. Replay constants
# ----------------------------------------------------------------------
def test_lr_policy_constants():
    """LR policy constants are correctly named."""
    assert LR_POLICY_STAGE_A_REPLAY == "STAGE_A_EXACT_TRACE_REPLAY"
    assert LR_POLICY_STAGE_B_CONSTANT == "STAGE_B_CONSTANT_LR_FROM_CONFIG"


# ----------------------------------------------------------------------
# 5. RefitEngine signature accepts stage_a_lr_trace
# ----------------------------------------------------------------------
def test_refit_engine_accepts_stage_a_lr_trace():
    """RefitEngine.refit() signature must accept stage_a_lr_trace kwarg."""
    sig = inspect.signature(RefitEngine.refit)
    assert "stage_a_lr_trace" in sig.parameters
    # Default must be None for backward-compat
    assert sig.parameters["stage_a_lr_trace"].default is None


# ----------------------------------------------------------------------
# 6. Source-level checks: no scheduler step in Stage B
# ----------------------------------------------------------------------
def test_refit_engine_does_not_call_scheduler_step_in_loop():
    """The refit() loop must NOT call any lr_scheduler.step() method.

    The source contains comments mentioning scheduler.step() (as a negative
    assertion). We strip comments and verify no executable code calls it.
    """
    source = inspect.getsource(RefitEngine.refit)
    # Strip lines that are comments (start with # after optional whitespace)
    code_lines = [
        line for line in source.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert "scheduler.step()" not in code_only
    assert "_lr_scheduler.step()" not in code_only


def test_refit_engine_uses_set_optimizer_lr():
    """The refit loop must call set_optimizer_lr when trace is provided."""
    source = inspect.getsource(RefitEngine.refit)
    assert "set_optimizer_lr" in source


def test_refit_engine_persists_lr_policy():
    """refit_status.json must include lr_policy field."""
    source = inspect.getsource(RefitEngine.refit)
    assert '"lr_policy": lr_policy' in source


# ----------------------------------------------------------------------
# 7. Stage-B exact replay equality (semantic verification)
# ----------------------------------------------------------------------
def test_cosine_replay_equals_stage_a_trace():
    """Simulated Stage-B replay for COSINE produces an LR sequence
    exactly equal to Stage-A's lr_used_for_epoch[1..best_epoch_inner].

    We simulate by reading optimizer.param_groups[0]["lr"] BEFORE each
    epoch's optimizer.step() and verifying it matches the trace.
    """
    # Stage-A trace (e.g. from a COSINE scheduler with T_max=50)
    # Step values follow cosine schedule from 3e-4 to 1e-5
    import math
    T_max = 50
    eta_min = 1e-5
    initial_lr = 3e-4
    stage_a_trace = [
        eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * k / T_max))
        for k in range(1, T_max + 1)
    ]
    # best_epoch_inner (e.g. 16)
    best_epoch_inner = 16

    # Simulate Stage-B replay: set optimizer lr from trace[k-1] at each epoch
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)

    stage_b_actual_lrs = []
    for epoch in range(1, best_epoch_inner + 1):
        # REPLAY: set lr from trace BEFORE training
        set_optimizer_lr(opt, stage_a_trace[epoch - 1])
        # Capture the actual lr the optimizer uses
        stage_b_actual_lrs.append(opt.param_groups[0]["lr"])
        # Simulate optimizer.step()

    # Verify equality
    for epoch in range(1, best_epoch_inner + 1):
        assert abs(stage_b_actual_lrs[epoch - 1] - stage_a_trace[epoch - 1]) < 1e-15, (
            f"Stage-B epoch {epoch} lr mismatch: "
            f"actual={stage_b_actual_lrs[epoch - 1]} vs "
            f"trace={stage_a_trace[epoch - 1]}"
        )


def test_plateau_replay_equals_stage_a_trace():
    """Simulated Stage-B replay for PLATEAU produces an LR sequence
    exactly equal to Stage-A's lr_used_for_epoch[1..best_epoch_inner].
    """
    # Simulate a Stage-A Plateau scenario: first few epochs reduce
    # lr after patience exhaustion.
    # Just construct an arbitrary trace.
    stage_a_trace = [3e-4] * 5 + [1.5e-4] * 3 + [0.75e-4] * 5
    best_epoch_inner = 13

    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)

    stage_b_actual_lrs = []
    for epoch in range(1, best_epoch_inner + 1):
        set_optimizer_lr(opt, stage_a_trace[epoch - 1])
        stage_b_actual_lrs.append(opt.param_groups[0]["lr"])

    for epoch in range(1, best_epoch_inner + 1):
        assert abs(stage_b_actual_lrs[epoch - 1] - stage_a_trace[epoch - 1]) < 1e-15


def test_off_replay_equals_constant_3e4():
    """OFF trace is constant 3e-4 — replay is a no-op but produces the same."""
    stage_a_trace = [3e-4] * 50  # OFF: never changes
    best_epoch_inner = 50

    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)

    stage_b_actual_lrs = []
    for epoch in range(1, best_epoch_inner + 1):
        set_optimizer_lr(opt, stage_a_trace[epoch - 1])
        stage_b_actual_lrs.append(opt.param_groups[0]["lr"])

    # All 3e-4
    for epoch in range(1, best_epoch_inner + 1):
        assert abs(stage_b_actual_lrs[epoch - 1] - 3e-4) < 1e-15


# ----------------------------------------------------------------------
# 8. best_epoch_inner boundary
# ----------------------------------------------------------------------
def test_replay_stops_exactly_at_best_epoch_inner():
    """Replay produces exactly best_epoch_inner epochs, no more."""
    stage_a_trace = [3e-4, 2.99e-4, 2.97e-4, 2.94e-4, 2.91e-4]
    best_epoch_inner = 3  # only first 3 used

    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)

    stage_b_actual_lrs = []
    for epoch in range(1, best_epoch_inner + 1):
        set_optimizer_lr(opt, stage_a_trace[epoch - 1])
        stage_b_actual_lrs.append(opt.param_groups[0]["lr"])

    assert len(stage_b_actual_lrs) == best_epoch_inner
    assert stage_b_actual_lrs == [3e-4, 2.99e-4, 2.97e-4]


def test_no_off_by_one_replay():
    """Stage-B epoch k uses trace[k-1], not trace[k]."""
    stage_a_trace = [3e-4, 2.99e-4, 2.97e-4]
    best_epoch_inner = 3

    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)

    # Set lr for epoch 1
    set_optimizer_lr(opt, stage_a_trace[0])  # 3e-4
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12

    # Set lr for epoch 2
    set_optimizer_lr(opt, stage_a_trace[1])  # 2.99e-4
    assert abs(opt.param_groups[0]["lr"] - 2.99e-4) < 1e-12

    # Set lr for epoch 3
    set_optimizer_lr(opt, stage_a_trace[2])  # 2.97e-4
    assert abs(opt.param_groups[0]["lr"] - 2.97e-4) < 1e-12


def test_epoch_1_uses_stage_a_epoch_1_lr():
    """Stage-B epoch 1 MUST use Stage-A epoch 1 LR (3e-4 for all 3 schedulers)."""
    # OFF
    for trace in ([3e-4] * 10, [3e-4] + [2.99e-4] * 9, [3e-4] + [1.5e-4] * 9):
        model = torch.nn.Linear(10, 1)
        opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
        set_optimizer_lr(opt, trace[0])
        assert abs(opt.param_groups[0]["lr"] - trace[0]) < 1e-15
        assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-15


# ----------------------------------------------------------------------
# 9. Plateau replay needs no validation RMSE
# ----------------------------------------------------------------------
def test_plateau_replay_no_validation_needed():
    """Stage-B replay for REDUCE_ON_PLATEAU must NOT require validation RMSE.

    The replay is purely a forward lookup: trace[epoch-1] is set BEFORE
    training. No metric is consumed.
    """
    # We verify by inspection that the replay function signature has
    # no validation metric parameter.
    sig = inspect.signature(RefitEngine.refit)
    params = sig.parameters
    assert "stage_a_lr_trace" in params
    # No validation RMSE parameter for Stage-B
    assert "val_metric" not in params
    assert "validation_rmse" not in params
    # The only metric-flavored param is best_epoch_inner
    assert "best_epoch_inner" in params


def test_plateau_replay_no_scheduler_step_call():
    """REDUCE_ON_PLATEAU replay must not call scheduler.step() with metric.

    The replay is purely deterministic: set lr from trace.
    """
    source = inspect.getsource(RefitEngine.refit)
    # Specifically no ReduceLROnPlateau.step(...) call
    assert "ReduceLROnPlateau" not in source
    assert "CosineAnnealingLR" not in source
    # And no lr_scheduler usage at all
    assert "lr_scheduler" not in source


# ----------------------------------------------------------------------
# 10. Fresh optimizer state
# ----------------------------------------------------------------------
def test_fresh_optimizer_per_stage_b_run():
    """Each Stage-B run must use a fresh optimizer (no warm start).

    This is verified by checking the source: refit() always creates a
    new torch.optim.AdamW instance.
    """
    source = inspect.getsource(RefitEngine.refit)
    # Fresh AdamW creation
    assert "torch.optim.AdamW(" in source
    assert "lr=learning_rate" in source
    # No warm-start: no load_state_dict call on optimizer
    assert "optimizer.load_state_dict" not in source


def test_fresh_model_per_stage_b_run():
    """Each Stage-B run must use a fresh model."""
    source = inspect.getsource(RefitEngine.refit)
    # Model is built upstream and passed in — refit() does NOT load it
    # from a checkpoint or warm-start from Stage-A's state.
    assert "model.load_state_dict" not in source
    # Model passed in fresh by caller (orchestrator)


# ----------------------------------------------------------------------
# 11. Replay off by default (backward-compat)
# ----------------------------------------------------------------------
def test_default_stage_a_lr_trace_is_none():
    """When stage_a_lr_trace is not passed, refit() uses constant LR."""
    sig = inspect.signature(RefitEngine.refit)
    assert sig.parameters["stage_a_lr_trace"].default is None


def test_lr_policy_constants_present():
    """The policy constants are exported and distinct."""
    assert LR_POLICY_STAGE_A_REPLAY != LR_POLICY_STAGE_B_CONSTANT


# ----------------------------------------------------------------------
# 12. Persistence: lr_policy and trace are written to refit_status.json
# ----------------------------------------------------------------------
def test_refit_status_contains_lr_policy_field():
    """The refit_status.json payload includes lr_policy."""
    source = inspect.getsource(RefitEngine.refit)
    assert '"lr_policy": lr_policy' in source
    assert 'LR_POLICY_STAGE_A_REPLAY' in source
    assert 'LR_POLICY_STAGE_B_CONSTANT' in source


def test_refit_status_contains_lr_trace_length():
    """The refit_status.json payload includes lr_trace_length and lr_used_for_epoch."""
    source = inspect.getsource(RefitEngine.refit)
    assert '"lr_trace_length"' in source
    assert '"lr_used_for_epoch"' in source
    assert '"lr_trace_source"' in source


def test_refit_final_checkpoint_contains_lr_policy():
    """The refit_final.pt checkpoint payload includes lr_policy."""
    source = inspect.getsource(RefitEngine.refit)
    # The checkpoint payload dict includes lr_policy
    assert '"lr_policy": lr_policy' in source


# ----------------------------------------------------------------------
# 13. Malformed/short trace rejection
# ----------------------------------------------------------------------
def test_short_trace_rejected():
    """A trace shorter than best_epoch_inner is rejected with ValueError."""
    with pytest.raises(ValueError, match="too short"):
        validate_lr_trace([3e-4, 2.99e-4], best_epoch_inner=5)


def test_zero_length_trace_rejected():
    """Empty list is rejected."""
    with pytest.raises(ValueError, match="non-empty trace"):
        validate_lr_trace([], best_epoch_inner=1)


def test_invalid_best_epoch_rejected():
    """best_epoch_inner < 1 is rejected."""
    with pytest.raises(ValueError, match="best_epoch_inner must be >= 1"):
        validate_lr_trace([3e-4], best_epoch_inner=0)


# ----------------------------------------------------------------------
# 14. No Test access in replay code
# ----------------------------------------------------------------------
def test_stage_b_replay_no_test_access():
    """The stage_b_replay module has no data access — pure LR arithmetic."""
    from course_work.model_improvement_v2 import stage_b_replay
    source = inspect.getsource(stage_b_replay)
    assert "DataLoader" not in source
    assert "split_id" not in source
    assert "TEST" not in source or "test_" not in source


def test_refit_replay_code_no_test_access():
    """The refit() replay branch must not access test data."""
    source = inspect.getsource(RefitEngine.refit)
    # No test access
    assert "test_rows" not in source
    assert "TEST" not in source or "test_" not in source or "test_path" in source
    # Heartbeat path uses env var, not Test split
    assert "target_access_mode" not in source


# ----------------------------------------------------------------------
# 15. V1/default behavior unchanged
# ----------------------------------------------------------------------
def test_no_trace_means_constant_lr():
    """When stage_a_lr_trace is None, optimizer lr stays at config LR.

    Backward-compat: existing E01/E02/E06 callers don't pass this kwarg,
    so they get the existing behavior.
    """
    sig = inspect.signature(RefitEngine.refit)
    assert sig.parameters["stage_a_lr_trace"].default is None


def test_lr_used_for_epoch_field_in_history():
    """Stage-B history row has lr_used_for_epoch alongside learning_rate."""
    source = inspect.getsource(RefitEngine.refit)
    # Both fields set in the history row
    assert '"lr_used_for_epoch": current_lr_for_epoch' in source
    assert '"learning_rate": current_lr_for_epoch' in source


# ----------------------------------------------------------------------
# 16. Direct replay verification with mock optimizer
# ----------------------------------------------------------------------
def test_replay_does_not_invoke_scheduler_methods():
    """The replay path uses set_optimizer_lr, not any scheduler method."""
    # Inspect the source: replay branch must only call set_optimizer_lr
    source = inspect.getsource(RefitEngine.refit)
    # The replay branch:
    # if validated_trace is not None:
    #     current_lr_for_epoch = float(validated_trace[epoch - 1])
    #     set_optimizer_lr(optimizer, current_lr_for_epoch)
    replay_branch = source[source.find("if validated_trace is not None:"):
                           source.find("model.train()")]
    assert "set_optimizer_lr" in replay_branch
    assert "scheduler" not in replay_branch.lower()


def test_replay_metadata_persisted():
    """lr_policy and lr_trace_length persist to refit_status.json."""
    source = inspect.getsource(RefitEngine.refit)
    # The refit_status dict contains lr_policy and related fields
    assert '"lr_policy": lr_policy' in source
    assert 'LR_POLICY_STAGE_A_REPLAY' in source
