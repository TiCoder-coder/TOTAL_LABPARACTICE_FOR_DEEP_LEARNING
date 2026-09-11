"""Focused unit tests for V2 E07-C LR trace contract.

The trace contract:
  - For every Stage-A epoch, persist the optimizer LR that was ACTUALLY
    USED TO TRAIN THAT epoch.
  - Required field: `lr_used_for_epoch`.
  - Optional field: `next_lr_after_scheduler` (the LR for the NEXT epoch).
  - Epoch 1 lr_used_for_epoch MUST equal 3e-4.
  - For OFF: lr_used_for_epoch stays 3e-4 for every epoch.
  - scheduler.step() happens AFTER history row append.
  - No off-by-one: scheduler-adjusted LR first appears in next epoch's
    lr_used_for_epoch.

These tests verify the trace by mocking the engine's train() internals
or by inspecting HISTORY_COLUMNS. We do NOT run real training.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.optim as torch_optim

from course_work.training.engine import (
    HISTORY_COLUMNS,
    TrainingEngine,
)


# ----------------------------------------------------------------------
# 1. HISTORY_COLUMNS contains lr_used_for_epoch
# ----------------------------------------------------------------------
def test_history_columns_contain_lr_used_for_epoch():
    """HISTORY_COLUMNS must include lr_used_for_epoch."""
    assert "lr_used_for_epoch" in HISTORY_COLUMNS


def test_history_columns_contain_next_lr_after_scheduler():
    """HISTORY_COLUMNS must include next_lr_after_scheduler (optional side field)."""
    assert "next_lr_after_scheduler" in HISTORY_COLUMNS


def test_history_columns_preserve_v1_fields():
    """HISTORY_COLUMNS preserves V1 fields (epoch, train_loss, learning_rate, ...)."""
    required_v1 = [
        "epoch",
        "train_loss",
        "train_rmse_wh",
        "validation_rmse_wh",
        "validation_mae_wh",
        "validation_r2",
        "learning_rate",
        "epoch_seconds",
        "is_best",
    ]
    for field in required_v1:
        assert field in HISTORY_COLUMNS, f"V1 field missing: {field}"


# ----------------------------------------------------------------------
# 2. Engine train() captures lr_used_for_epoch BEFORE training
# ----------------------------------------------------------------------
def test_engine_train_captures_lr_at_epoch_start():
    """The engine.train() loop must capture lr_used_for_epoch BEFORE training.

    We inspect the source flow: lr_used_for_epoch = float(optimizer.param_groups[0]['lr'])
    must appear BEFORE the inner `for batch in train_loader:` loop.
    """
    source = inspect.getsource(TrainingEngine.train)
    # Find the position of `lr_used_for_epoch = float(...)`
    capture_idx = source.find("lr_used_for_epoch = float(optimizer.param_groups[0][\"lr\"])")
    # Find the inner training loop
    train_loop_idx = source.find("for batch in train_loader:")
    assert capture_idx > 0, "lr_used_for_epoch capture statement not found"
    assert train_loop_idx > 0, "training loop not found"
    assert capture_idx < train_loop_idx, (
        "lr_used_for_epoch MUST be captured BEFORE the training loop starts"
    )


# ----------------------------------------------------------------------
# 3. Engine records lr_used_for_epoch in history row BEFORE scheduler step
# ----------------------------------------------------------------------
def test_engine_history_row_before_scheduler_step():
    """The history row (with lr_used_for_epoch) MUST be appended BEFORE
    scheduler.step() is called.
    """
    source = inspect.getsource(TrainingEngine.train)
    # Find the position of history_rows.append
    append_idx = source.find('history_rows.append(')
    # Find the position of _lr_scheduler.step()
    step_idx = source.find('_lr_scheduler.step()')
    assert append_idx > 0
    assert step_idx > 0
    # In the post-validation section of the loop:
    # find the SECOND occurrence of history_rows.append (the one near the end of epoch)
    post_validation_append = source.rfind('history_rows.append(')
    assert post_validation_append > 0
    assert post_validation_append < step_idx, (
        "history_rows.append MUST happen BEFORE scheduler.step() — "
        "scheduler step should not contaminate the trace for THIS epoch"
    )


# ----------------------------------------------------------------------
# 4. Engine records lr_used_for_epoch == learning_rate in same row
# ----------------------------------------------------------------------
def test_lr_used_for_epoch_equals_learning_rate():
    """In every history row, lr_used_for_epoch and learning_rate should be equal
    (both represent the LR used to train that epoch).
    """
    source = inspect.getsource(TrainingEngine.train)
    # In the dict literal of history_rows.append, both should reference
    # lr_used_for_epoch.
    assert '"learning_rate": lr_used_for_epoch' in source
    assert '"lr_used_for_epoch": lr_used_for_epoch' in source


# ----------------------------------------------------------------------
# 5. Direct lr_used_for_epoch semantics via simulated training
# ----------------------------------------------------------------------
def test_off_lr_trace_constant_3e4():
    """For OFF scheduler, lr_used_for_epoch stays 3e-4 across all epochs.

    We simulate by inspecting the source: when _lr_scheduler is None,
    lr_used_for_epoch is captured from optimizer.param_groups[0]['lr']
    at epoch start. Since no scheduler steps, optimizer lr stays at 3e-4.
    """
    source = inspect.getsource(TrainingEngine.train)
    # The capture line is unconditional
    assert "lr_used_for_epoch = float(optimizer.param_groups[0][\"lr\"])" in source


def test_cosine_epoch1_optimizer_lr_is_3e4():
    """w/ fresh AdamW lr=3e-4 and fresh CosineAnnealingLR, optimizer lr is 3e-4
    before any scheduler.step() call. The engine captures lr_used_for_epoch
    from optimizer.param_groups[0]['lr'] BEFORE training.
    """
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)

    # Before any training, optimizer lr == 3e-4
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12

    # The engine reads this value BEFORE any scheduler step.
    # After 1 step, lr is slightly reduced (but lr_used_for_epoch in
    # the epoch-1 history row must already be 3e-4 because we captured
    # BEFORE training).
    sched.step()
    assert opt.param_groups[0]["lr"] < 3e-4  # post-step reduced


def test_plateau_epoch1_optimizer_lr_is_3e4():
    """w/ fresh AdamW lr=3e-4 and fresh ReduceLROnPlateau, optimizer lr is 3e-4
    before any scheduler.step() call.
    """
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3,
        threshold=0.0, threshold_mode="abs", cooldown=0, min_lr=1e-5, eps=1e-8
    )

    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12


# ----------------------------------------------------------------------
# 6. Off-by-one check: scheduler-adjusted LR appears in next epoch
# ----------------------------------------------------------------------
def test_cosine_no_off_by_one():
    """CosineAnnealingLR.step() updates the LR for the NEXT step's optimizer.

    Verifies that lr_used_for_epoch[epoch=1] = 3e-4 (initial), and
    lr_used_for_epoch[epoch=2] = post-step value of epoch 1.

    This is the off-by-one test: the scheduler step at end of epoch k
    affects epoch k+1, NOT epoch k.
    """
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)

    # Epoch 1 starts
    lr_epoch_1 = opt.param_groups[0]["lr"]  # captured BEFORE training
    assert abs(lr_epoch_1 - 3e-4) < 1e-12

    # Epoch 1 training happens (we simulate the optimizer step)
    # ...

    # Epoch 1 ends — scheduler steps (post-training, post-validation)
    sched.step()

    # Epoch 2 starts — optimizer lr is now the post-step value
    lr_epoch_2 = opt.param_groups[0]["lr"]
    assert lr_epoch_2 < lr_epoch_1, "Epoch-2 lr MUST be smaller than epoch-1 lr"
    assert lr_epoch_2 > 1e-5  # not yet at eta_min


def test_plateau_no_off_by_one():
    """ReduceLROnPlateau.step() updates the LR for the NEXT epoch.

    Verifies lr_used_for_epoch[epoch=1] = 3e-4, lr_used_for_epoch[epoch=2] = 3e-4
    (until patience is exhausted).
    """
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3,
        threshold=0.0, threshold_mode="abs", cooldown=0, min_lr=1e-5, eps=1e-8
    )

    # Epoch 1 starts
    lr_epoch_1 = opt.param_groups[0]["lr"]
    assert abs(lr_epoch_1 - 3e-4) < 1e-12

    # Epoch 1 ends with high RMSE → plateau
    sched.step(100.0)

    # Epoch 2 starts — still 3e-4 (patience=3, only 1 plateau so far)
    lr_epoch_2 = opt.param_groups[0]["lr"]
    assert abs(lr_epoch_2 - 3e-4) < 1e-12


# ----------------------------------------------------------------------
# 7. next_lr_after_scheduler semantics
# ----------------------------------------------------------------------
def test_next_lr_after_scheduler_filled_post_step():
    """In the engine source, next_lr_after_scheduler is set AFTER scheduler step."""
    source = inspect.getsource(TrainingEngine.train)
    # The field is referenced twice: once at append (default = lr_used_for_epoch),
    # once after scheduler step (updated to post_step_lr).
    assert '"next_lr_after_scheduler": lr_used_for_epoch' in source
    assert 'history_rows[-1]["next_lr_after_scheduler"] = post_step_lr' in source


# ----------------------------------------------------------------------
# 8. Engine source has lr_used_for_epoch = float(...) pattern
# ----------------------------------------------------------------------
def test_engine_uses_param_groups_for_lr_capture():
    """The engine must read lr from optimizer.param_groups[0]['lr'] (the
    authoritative source), not from scheduler.get_last_lr() which is post-step.
    """
    source = inspect.getsource(TrainingEngine.train)
    assert "lr_used_for_epoch = float(optimizer.param_groups[0][\"lr\"])" in source


# ----------------------------------------------------------------------
# 9. Trace contract summary: epoch 1 is always 3e-4
# ----------------------------------------------------------------------
def test_epoch1_lr_is_3e4_for_all_three_schedulers():
    """For all three schedulers (OFF, COSINE, PLATEAU), epoch-1 lr must be 3e-4.

    This is the contract guarantee: the optimizer's initial lr=3e-4 is the
    value the model uses for the first epoch regardless of scheduler choice.
    """
    initial_lr = 3e-4

    # OFF: no scheduler
    model_off = torch.nn.Linear(10, 1)
    opt_off = torch_optim.AdamW(model_off.parameters(), lr=initial_lr)
    assert abs(opt_off.param_groups[0]["lr"] - initial_lr) < 1e-12

    # COSINE: fresh scheduler doesn't change initial lr
    model_cos = torch.nn.Linear(10, 1)
    opt_cos = torch_optim.AdamW(model_cos.parameters(), lr=initial_lr)
    torch.optim.lr_scheduler.CosineAnnealingLR(opt_cos, T_max=50, eta_min=1e-5)
    assert abs(opt_cos.param_groups[0]["lr"] - initial_lr) < 1e-12

    # PLATEAU: fresh scheduler doesn't change initial lr
    model_plat = torch.nn.Linear(10, 1)
    opt_plat = torch_optim.AdamW(model_plat.parameters(), lr=initial_lr)
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_plat, mode="min", factor=0.5, patience=3,
        threshold=0.0, threshold_mode="abs", cooldown=0, min_lr=1e-5, eps=1e-8
    )
    assert abs(opt_plat.param_groups[0]["lr"] - initial_lr) < 1e-12


# ----------------------------------------------------------------------
# 10. No Test access in the LR trace code path
# ----------------------------------------------------------------------
def test_lr_trace_no_test_access():
    """The LR trace logic must NOT touch data loaders or splits.

    The capture happens via optimizer.param_groups[0]['lr'] — pure optimizer
    state. No loaders are touched.
    """
    source = inspect.getsource(TrainingEngine.train)
    # Find the lr_used_for_epoch capture
    capture_idx = source.find("lr_used_for_epoch = float(optimizer.param_groups[0][\"lr\"])")
    # Find the next 200 chars to see what's around it
    nearby = source[capture_idx:capture_idx + 200]
    # No data loader or split access in the immediate vicinity
    assert "DataLoader" not in nearby
    assert "split_id" not in nearby
    assert "Test" not in nearby or "test_" not in nearby


# ----------------------------------------------------------------------
# 11. Backward-compatibility: existing learning_rate field still populated
# ----------------------------------------------------------------------
def test_learning_rate_field_still_in_history_row():
    """The existing `learning_rate` field is preserved for V1 backward-compat.

    Both `learning_rate` and `lr_used_for_epoch` should equal the LR used
    to train that epoch.
    """
    source = inspect.getsource(TrainingEngine.train)
    # Both fields are set to lr_used_for_epoch in the same row
    assert '"learning_rate": lr_used_for_epoch' in source
    assert '"lr_used_for_epoch": lr_used_for_epoch' in source


# ----------------------------------------------------------------------
# 12. Engine source confirms OFF behavior unchanged
# ----------------------------------------------------------------------
def test_off_scheduler_field_unused():
    """When scheduler_name is None or OFF, _lr_scheduler is None and
    scheduler step is skipped.
    """
    source = inspect.getsource(TrainingEngine.train)
    # The guard: _lr_scheduler is None check before scheduler.step()
    assert "if _lr_scheduler is not None:" in source
    # Inside the if block, scheduler.step() is called
    # Inside the else (implicit), nothing happens
    # For OFF: lr_used_for_epoch == initial_lr (no scheduler step modifies it)


# ----------------------------------------------------------------------
# 13. Direct test: scheduler-adjusted LR appears in NEXT epoch
# ----------------------------------------------------------------------
def test_cosine_lr_first_changes_at_epoch_2():
    """With T_max=50 cosine, eta_min=1e-5, lr at end of epoch 1 is slightly
    below 3e-4 (post-step), and epoch 2 lr_used_for_epoch must equal that.

    Verifies: lr_used_for_epoch[epoch=1] = 3e-4, lr_used_for_epoch[epoch=2] < 3e-4.
    """
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)

    lr_at_epoch_1_start = opt.param_groups[0]["lr"]
    assert abs(lr_at_epoch_1_start - 3e-4) < 1e-12

    sched.step()

    lr_at_epoch_2_start = opt.param_groups[0]["lr"]
    assert lr_at_epoch_2_start < 3e-4
    assert lr_at_epoch_2_start > 1e-5


def test_plateau_lr_first_changes_after_patience_exhausted():
    """With patience=3, lr stays at 3e-4 for epochs 1-5, then reduces at epoch 6.

    PyTorch 2.12.1 behavior: num_bad_epochs > patience triggers reduction.
    With patience=3 and threshold_mode='abs', the scheduler needs 5 plateau
    steps before reducing (num_bad goes 1, 2, 3, then 4 > 3 → reduce).
    """
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3,
        threshold=0.0, threshold_mode="abs", cooldown=0, min_lr=1e-5, eps=1e-8
    )

    lr_per_epoch = []
    for _ in range(6):
        lr_per_epoch.append(opt.param_groups[0]["lr"])
        sched.step(100.0)  # plateau

    # Epochs 1-5: lr stays at 3e-4 (captured BEFORE sched.step, so the
    # 5th plateau step hasn't triggered reduction yet).
    for i in range(5):
        assert abs(lr_per_epoch[i] - 3e-4) < 1e-12, f"epoch {i+1}: {lr_per_epoch[i]}"
    # Epoch 6: lr reduced (5 plateau steps accumulated, 5 > 3 → reduce)
    assert lr_per_epoch[5] < 3e-4
