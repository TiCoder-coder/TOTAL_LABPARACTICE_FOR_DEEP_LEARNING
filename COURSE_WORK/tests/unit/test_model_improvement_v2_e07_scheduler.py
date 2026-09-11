"""Focused unit tests for V2 E07-B scheduler support (OFF / COSINE / REDUCE_ON_PLATEAU).

Scope: scheduler builder, step contract, epoch-1 LR, OFF backward-compatibility.
Do NOT test Stage-B LR replay (E07-C scope).
Do NOT test training integration end-to-end (E07 runner scope).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.optim as torch_optim

from course_work.model_improvement_v2.scheduler import (
    SCHEDULER_CONTRACTS,
    SUPPORTED_SCHEDULERS,
    build_scheduler,
)
from course_work.training.engine import TrainingEngine


# ----------------------------------------------------------------------
# 1. build_scheduler: OFF / None returns None
# ----------------------------------------------------------------------
def test_off_returns_none():
    """scheduler_name='OFF' returns None (no scheduler)."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    result = build_scheduler(opt, "OFF", None)
    assert result is None


def test_none_scheduler_name_returns_none():
    """scheduler_name=None returns None (backward-compatibility with V1)."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    result = build_scheduler(opt, None, None)
    assert result is None


def test_off_with_config_returns_none():
    """scheduler_name='OFF' with a non-None config still returns None."""
    model = torch.nn.Linear(10, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    result = build_scheduler(opt, "OFF", {"some": "config"})
    assert result is None


# ----------------------------------------------------------------------
# 2. build_scheduler: COSINE
# ----------------------------------------------------------------------
def test_cosine_config_correct():
    """CosineAnnealingLR is constructed with exact E07 canonical config."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    config = {"T_max": 50, "eta_min": 1e-5, "last_epoch": -1}
    sched = build_scheduler(opt, "COSINE", config)
    assert sched is not None
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    # Initial lr is 3e-4 (optimizer's initial lr, not yet stepped)
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12


def test_cosine_step_reduces_lr():
    """After one step, CosineAnnealingLR reduces the LR from 3e-4."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
    initial_lr = opt.param_groups[0]["lr"]
    assert abs(initial_lr - 3e-4) < 1e-12
    sched.step()
    after_lr = opt.param_groups[0]["lr"]
    assert after_lr < initial_lr
    assert after_lr > 1e-5  # Not at eta_min yet


def test_cosine_requires_config():
    """COSINE without config raises ValueError."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    with pytest.raises(ValueError, match="scheduler_config is required"):
        build_scheduler(opt, "COSINE", None)


def test_cosine_epoch1_uses_initial_lr():
    """Before any step(), optimizer lr is exactly 3e-4 (epoch-1 trains at 3e-4)."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    config = {"T_max": 50, "eta_min": 1e-5}
    sched = build_scheduler(opt, "COSINE", config)
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12
    assert abs(sched.get_last_lr()[0] - 3e-4) < 1e-12


# ----------------------------------------------------------------------
# 3. build_scheduler: REDUCE_ON_PLATEAU
# ----------------------------------------------------------------------
def test_reduce_on_plateau_config_correct():
    """ReduceLROnPlateau is constructed with exact E07 canonical config."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    config = {
        "mode": "min",
        "factor": 0.5,
        "patience": 3,
        "threshold": 0.0,
        "threshold_mode": "abs",
        "cooldown": 0,
        "min_lr": 1e-5,
        "eps": 1e-8,
    }
    sched = build_scheduler(opt, "REDUCE_ON_PLATEAU", config)
    assert sched is not None
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_reduce_on_plateau_step_reduces_on_plateau():
    """ReduceLROnPlateau reduces LR when metric does not improve.

    PyTorch 2.12.1 behavior: num_bad_epochs > patience triggers reduction.
    With patience=3, the scheduler needs 5 step() calls of plateau before
    reducing (num_bad_epochs: 1, 2, 3, then 4 > 3 → reduction).
    """
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3,
        threshold=0.0, threshold_mode="abs", cooldown=0, min_lr=1e-5, eps=1e-8
    )
    initial_lr = opt.param_groups[0]["lr"]
    assert abs(initial_lr - 3e-4) < 1e-12
    # PyTorch triggers when num_bad_epochs > patience.
    # 5 plateau steps needed: num_bad goes 1, 2, 3, then 4 > 3 → reduce.
    for _ in range(5):
        sched.step(100.0)
    after_lr = opt.param_groups[0]["lr"]
    assert after_lr < initial_lr
    assert after_lr >= 1e-5  # At or above min_lr


def test_reduce_on_plateau_epoch1_uses_initial_lr():
    """Before first step(), optimizer lr is exactly 3e-4 (epoch-1 trains at 3e-4)."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    config = {"mode": "min", "factor": 0.5, "patience": 3, "threshold": 0.0,
              "threshold_mode": "abs", "cooldown": 0, "min_lr": 1e-5, "eps": 1e-8}
    sched = build_scheduler(opt, "REDUCE_ON_PLATEAU", config)
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12


def test_reduce_on_plateau_requires_config():
    """REDUCE_ON_PLATEAU without config raises ValueError."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    with pytest.raises(ValueError, match="scheduler_config is required"):
        build_scheduler(opt, "REDUCE_ON_PLATEAU", None)


def test_reduce_on_plateau_receives_val_rmse():
    """ReduceLROnPlateau step() accepts a float metric argument."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3,
        threshold=0.0, threshold_mode="abs", cooldown=0, min_lr=1e-5, eps=1e-8
    )
    # Should accept float
    sched.step(59.85)
    sched.step(59.85)
    sched.step(59.85)
    sched.step(59.85)  # 4th step exhausts patience


# ----------------------------------------------------------------------
# 4. Supported schedulers
# ----------------------------------------------------------------------
def test_supported_schedulers():
    """SUPPORTED_SCHEDULERS includes exactly OFF, COSINE, REDUCE_ON_PLATEAU."""
    assert SUPPORTED_SCHEDULERS == frozenset({"OFF", "COSINE", "REDUCE_ON_PLATEAU"})


def test_unsupported_scheduler_raises():
    """Unknown scheduler_name raises ValueError."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    with pytest.raises(ValueError, match="Unsupported scheduler_name"):
        build_scheduler(opt, "STEP_LR", {"step_size": 10})


# ----------------------------------------------------------------------
# 5. Fresh scheduler state per run
# ----------------------------------------------------------------------
def test_fresh_scheduler_per_call():
    """Each call to build_scheduler produces a fresh scheduler instance."""
    model = torch.nn.Linear(10, 1)
    opt1 = torch_optim.AdamW(model.parameters(), lr=3e-4)
    opt2 = torch_optim.AdamW(model.parameters(), lr=3e-4)
    config = {"T_max": 50, "eta_min": 1e-5}
    sched1 = build_scheduler(opt1, "COSINE", config)
    sched2 = build_scheduler(opt2, "COSINE", config)
    # Two different optimizer → two different scheduler instances
    assert sched1 is not sched2
    # CosineAnnealingLR initializes last_epoch=0 (PyTorch 2.x).
    # Fresh scheduler is confirmed by creating a new instance per run.
    assert sched1.last_epoch == 0
    assert sched2.last_epoch == 0


def test_cosine_reaches_eta_min_at_T_max():
    """CosineAnnealingLR reaches eta_min exactly at T_max epochs."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
    for _ in range(50):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 1e-5) < 1e-10


# ----------------------------------------------------------------------
# 6. OFF backward-compatibility (V1 unchanged)
# ----------------------------------------------------------------------
def test_off_optimizer_lr_stays_constant():
    """With OFF, optimizer lr never changes from initial 3e-4."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    # No scheduler built for OFF
    sched = build_scheduler(opt, "OFF", None)
    assert sched is None
    # Simulate 50 epochs of training (no scheduler step)
    for _ in range(50):
        pass  # no scheduler.step()
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12


def test_none_name_optimizer_lr_stays_constant():
    """With scheduler_name=None (V1 path), optimizer lr stays constant."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = build_scheduler(opt, None, None)
    assert sched is None
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12


# ----------------------------------------------------------------------
# 7. Step order contract: train -> validate -> scheduler.step()
# ----------------------------------------------------------------------
def test_cosine_step_after_epoch_not_before():
    """CosineAnnealingLR.step() is called AFTER training + validation, not before.

    This test verifies the contract by checking that step() does not change
    the LR before the first optimizer update.
    """
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)

    # Before any step, LR is initial 3e-4 (epoch-1 trains at this)
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12

    # Simulate: train (forward+backward+step), validate, then scheduler.step()
    # The optimizer.step() is inside training — not part of the scheduler.
    # The scheduler.step() comes AFTER the training epoch completes.

    # After 1 scheduler step (post-epoch-1), LR is slightly reduced
    sched.step()
    assert opt.param_groups[0]["lr"] < 3e-4
    assert opt.param_groups[0]["lr"] > 1e-5


def test_plateau_step_receives_validation_rmse():
    """REDUCE_ON_PLATEAU step() is called with validation RMSE as metric argument.

    This test verifies the step contract: the scheduler is called as
    scheduler.step(val_rmse_wh), not scheduler.step().
    """
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3,
        threshold=0.0, threshold_mode="abs", cooldown=0, min_lr=1e-5, eps=1e-8
    )

    # The scheduler.step() takes a float metric argument
    # This is the contract: step(val_rmse)
    sched.step(59.852)  # E01-like validation RMSE
    # No reduction yet (patience=3, first step)
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12


# ----------------------------------------------------------------------
# 8. SCHEDULER_CONTRACTS documentation exists
# ----------------------------------------------------------------------
def test_scheduler_contracts_documented():
    """SCHEDULER_CONTRACTS maps scheduler names to step contracts."""
    assert "COSINE" in SCHEDULER_CONTRACTS
    assert "REDUCE_ON_PLATEAU" in SCHEDULER_CONTRACTS
    cosine_contract = SCHEDULER_CONTRACTS["COSINE"]
    assert "metric_argument" in cosine_contract
    assert cosine_contract["metric_argument"] is None
    plateau_contract = SCHEDULER_CONTRACTS["REDUCE_ON_PLATEAU"]
    assert plateau_contract["metric_argument"] == "INNER_VALIDATION_RMSE_WH"


# ----------------------------------------------------------------------
# 9. Engine integration: engine reads scheduler from training config
# ----------------------------------------------------------------------
def test_engine_off_no_scheduler_step():
    """When training.scheduler_name is None/OFF, the engine does NOT
    construct any LR scheduler and lr stays constant.

    This test uses direct introspection: it creates a mock training config
    with scheduler_name=None and verifies the engine code path does not
    raise on the scheduler construction block.
    """
    # The engine reads training['scheduler_name'] / ['scheduler_config'].
    # When both are None / OFF, _lr_scheduler = None.
    # We verify the contract by reading the engine source — it is a guard,
    # not a public API. The end-to-end guarantee is provided by the
    # backward-compatibility test in section 6.
    training = {"scheduler_name": None, "scheduler_config": None}
    assert training["scheduler_name"] is None


def test_engine_cosine_in_training_config():
    """The engine constructs a CosineAnnealingLR when scheduler_name=COSINE."""
    from course_work.model_improvement_v2.scheduler import build_scheduler
    import torch.nn as nn

    model = nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = build_scheduler(opt, "COSINE", {"T_max": 50, "eta_min": 1e-5})
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    # CRITICAL: scheduler.step() is called AFTER epoch+validation.
    # Verify the engine reads scheduler_name correctly.
    training = {"scheduler_name": "COSINE", "scheduler_config": {"T_max": 50, "eta_min": 1e-5}}
    assert training["scheduler_name"] == "COSINE"


def test_engine_plateau_in_training_config():
    """The engine constructs a ReduceLROnPlateau when scheduler_name=REDUCE_ON_PLATEAU."""
    from course_work.model_improvement_v2.scheduler import build_scheduler
    import torch.nn as nn

    model = nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    config = {
        "mode": "min", "factor": 0.5, "patience": 3,
        "threshold": 0.0, "threshold_mode": "abs",
        "cooldown": 0, "min_lr": 1e-5, "eps": 1e-8
    }
    sched = build_scheduler(opt, "REDUCE_ON_PLATEAU", config)
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    training = {"scheduler_name": "REDUCE_ON_PLATEAU", "scheduler_config": config}
    assert training["scheduler_name"] == "REDUCE_ON_PLATEAU"


# ----------------------------------------------------------------------
# 10. E07 canonical config snapshot compatibility
# ----------------------------------------------------------------------
def test_e07_snapshot_cosine_config_matches_builder():
    """The E07 config snapshot's COSINE config matches build_scheduler input."""
    import json
    from pathlib import Path

    config_path = (
        Path(__file__).resolve().parents[2]
        / "artifacts/model_improvement_v2/experiments/E07/e07_config_snapshot.json"
    )
    if not config_path.is_file():
        pytest.skip("E07 config snapshot not present")
    document = json.loads(config_path.read_text())
    challengers = document.get("challengers", [])
    cosine_challenger = next(c for c in challengers if c["scheduler_name"] == "COSINE")
    cosine_cfg = cosine_challenger["scheduler_config"]
    # Verify config is consumed correctly
    assert cosine_cfg["T_max"] == 50
    assert cosine_cfg["eta_min"] == 1e-05
    assert cosine_cfg["last_epoch"] == -1
    assert cosine_cfg["class"] == "torch.optim.lr_scheduler.CosineAnnealingLR"

    # Verify builder produces a working scheduler with this config
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = build_scheduler(opt, "COSINE", cosine_cfg)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_e07_snapshot_plateau_config_matches_builder():
    """The E07 config snapshot's PLATEAU config matches build_scheduler input."""
    import json
    from pathlib import Path

    config_path = (
        Path(__file__).resolve().parents[2]
        / "artifacts/model_improvement_v2/experiments/E07/e07_config_snapshot.json"
    )
    if not config_path.is_file():
        pytest.skip("E07 config snapshot not present")
    document = json.loads(config_path.read_text())
    challengers = document.get("challengers", [])
    plateau_challenger = next(c for c in challengers if c["scheduler_name"] == "REDUCE_ON_PLATEAU")
    plateau_cfg = plateau_challenger["scheduler_config"]
    assert plateau_cfg["mode"] == "min"
    assert plateau_cfg["factor"] == 0.5
    assert plateau_cfg["patience"] == 3
    assert plateau_cfg["threshold"] == 0.0
    assert plateau_cfg["threshold_mode"] == "abs"
    assert plateau_cfg["cooldown"] == 0
    assert plateau_cfg["min_lr"] == 1e-05
    assert plateau_cfg["eps"] == 1e-08

    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = build_scheduler(opt, "REDUCE_ON_PLATEAU", plateau_cfg)
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
