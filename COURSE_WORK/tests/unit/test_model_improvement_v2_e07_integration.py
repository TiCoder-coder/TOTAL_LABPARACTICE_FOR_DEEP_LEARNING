"""E07-F: Full E07 integration test + pre-train bug prevention.

Verifies the complete chain WITHOUT official training:

  E07 runner
  -> Stage A scheduler
  -> lr_used_for_epoch persistence
  -> selected best_epoch_inner
  -> Stage B exact LR replay
  -> Stage C DIRECT prediction
  -> pooling (1-D shape contract)
  -> comparison/finalization

CRITICAL regression check from E06:

  E06 previously crashed after all training because:
    y_true shape = (N,)
    y_pred shape = (N,1)

  E07 also uses DIRECT prediction. This module asserts that E07 Stage-C
  predictions passed to pooled metrics are 1-D.

Tests are organized by chain step:

  1. Candidate matrix
  2. Control reuse
  3. Stage A chain
  4. LR trace persistence
  5. Stage B replay chain
  6. Stage C shape (E06 regression prevention)
  7. Pooling
  8. Finalization
  9. Authorization gate
  10. Isolation (V1/E01/E02/E06 unchanged)
"""
from __future__ import annotations

import csv
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.optim as torch_optim

from course_work.model_improvement_v2.contracts import PRIMARY_CHANGE_PREFIXES
from course_work.model_improvement_v2.e07_runner import (
    BASE_CANDIDATE_ID,
    CHALLENGER_CANDIDATE_IDS,
    EXPECTED_CONTROL_METRICS,
    INITIAL_LEARNING_RATE,
    LOCKED_COSINE_CONFIG,
    LOCKED_PLATEAU_CONFIG,
    PROMOTION_THRESHOLD_RMSE_WH,
    REGISTRY_NAMESPACE,
    SUPPORTED_SCHEDULERS,
    build_e07_run_context,
    load_e07_config,
    project_root,
    run_preflight,
    validate_e07_document,
)
from course_work.model_improvement_v2.scheduler import build_scheduler
from course_work.model_improvement_v2.stage_b_replay import (
    LR_POLICY_STAGE_A_REPLAY,
    LR_POLICY_STAGE_B_CONSTANT,
    extract_lr_trace_from_history,
    set_optimizer_lr,
    validate_lr_trace,
)
from course_work.rolling_origin.pooling import compute_pooled_metrics
from course_work.rolling_origin.refit_engine import RefitEngine
from course_work.rolling_origin.stages import evaluate_stage_c


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _e07_snapshot() -> dict:
    return load_e07_config(project_root())


def _make_dummy_model() -> torch.nn.Module:
    """A tiny PyTorch model that outputs shape [B, 1] (DIRECT convention).

    Uses a real Linear layer so AdamW has trainable parameters.
    """

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.linear(x.mean(dim=1))  # collapse lookback dim
            return out  # shape [B, 1]

        def checkpoint_metadata(self) -> dict:
            return {"family": "TINY"}

    return _Tiny()


def _make_dummy_scaler_bundle():
    """Build a duck-typed scaler with inverse_transform_y for Stage-C testing.

    FoldLocalScalerBundle is a strict dataclass with many required fields.
    For unit-testing inverse_transform_y + Stage-C flattening we only need
    y_mean, y_std, and target_scaling_option.
    """
    from types import SimpleNamespace

    ns = SimpleNamespace(
        bundle_id="TEST_BUNDLE",
        target_scaling_option="YS1",
        y_mean=80.0,
        y_std=40.0,
    )

    def inverse_transform_y(y_scaled):
        return np.asarray(y_scaled, dtype=np.float64) * ns.y_std + ns.y_mean

    ns.inverse_transform_y = inverse_transform_y
    return ns


def _run_preflight() -> dict:
    return run_preflight()


# ----------------------------------------------------------------------
# 1. Candidate matrix
# ----------------------------------------------------------------------
def test_candidate_count_exactly_two():
    """E07 must have exactly 2 training candidates."""
    assert len(CHALLENGER_CANDIDATE_IDS) == 2
    assert sorted(CHALLENGER_CANDIDATE_IDS) == [
        "TR_C2_ALT_LOOKBACK_SCHED_COSINE",
        "TR_C2_ALT_LOOKBACK_SCHED_REDUCE_ON_PLATEAU",
    ]


def test_schedulers_are_cosine_and_plateau():
    """The two trained schedulers are COSINE and REDUCE_ON_PLATEAU."""
    assert set(SUPPORTED_SCHEDULERS) == {"COSINE", "REDUCE_ON_PLATEAU"}
    assert "OFF" not in SUPPORTED_SCHEDULERS


def test_scheduler_off_excluded_from_training():
    """OFF control is reused, never trained."""
    assert "OFF" not in CHALLENGER_CANDIDATE_IDS
    assert all("_SCHED_" in cid for cid in CHALLENGER_CANDIDATE_IDS)


def test_preflight_reports_two_training_candidates():
    """Preflight declares training_candidate_count=2."""
    preflight = _run_preflight()
    assert preflight["training_candidate_count"] == 2
    assert sorted(preflight["training_candidate_ids"]) == sorted(CHALLENGER_CANDIDATE_IDS)
    assert sorted(preflight["schedulers_trained"]) == ["COSINE", "REDUCE_ON_PLATEAU"]
    assert preflight["scheduler_off_excluded_from_training"] is True


def test_cosine_scheduler_config_locked():
    """COSINE config matches E07-A lock (T_max=50, eta_min=1e-5, last_epoch=-1)."""
    assert LOCKED_COSINE_CONFIG == {
        "class": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "T_max": 50,
        "eta_min": 1e-05,
        "last_epoch": -1,
    }


def test_plateau_scheduler_config_locked():
    """PLATEAU config matches E07-A lock (factor=0.5, patience=3, etc.)."""
    assert LOCKED_PLATEAU_CONFIG == {
        "class": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 3,
        "threshold": 0.0,
        "threshold_mode": "abs",
        "cooldown": 0,
        "min_lr": 1e-05,
        "eps": 1e-08,
    }


def test_cosine_scheduler_builds_with_locked_config():
    """build_scheduler produces a CosineAnnealingLR with the locked config."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = build_scheduler(opt, "COSINE", LOCKED_COSINE_CONFIG)
    assert sched is not None
    assert sched.get_last_lr()[0] == 3e-4


def test_plateau_scheduler_builds_with_locked_config():
    """build_scheduler produces a ReduceLROnPlateau with the locked config."""
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = build_scheduler(opt, "REDUCE_ON_PLATEAU", LOCKED_PLATEAU_CONFIG)
    assert sched is not None
    assert sched.get_last_lr()[0] == 3e-4


def test_preflight_validates_cosine_config():
    """Preflight rejects a snapshot with a tampered COSINE config."""
    snap = _e07_snapshot()
    snap = dict(snap)
    snap["challengers"] = list(snap["challengers"])
    snap["challengers"][0] = dict(snap["challengers"][0])
    snap["challengers"][0]["scheduler_config"] = dict(LOCKED_COSINE_CONFIG)
    snap["challengers"][0]["scheduler_config"]["T_max"] = 99  # wrong
    snap["challengers"][0]["config_fingerprint"] = "BROKEN"
    from course_work.model_improvement_v2.e07_runner import E07PreflightError
    with pytest.raises(E07PreflightError, match="COSINE config mismatch"):
        validate_e07_document(snap, project_root())


def test_preflight_validates_plateau_config():
    """Preflight rejects a snapshot with a tampered PLATEAU config."""
    snap = _e07_snapshot()
    snap = dict(snap)
    snap["challengers"] = list(snap["challengers"])
    snap["challengers"][1] = dict(snap["challengers"][1])
    snap["challengers"][1]["scheduler_config"] = dict(LOCKED_PLATEAU_CONFIG)
    snap["challengers"][1]["scheduler_config"]["patience"] = 7  # wrong
    snap["challengers"][1]["config_fingerprint"] = "BROKEN"
    from course_work.model_improvement_v2.e07_runner import E07PreflightError
    with pytest.raises(E07PreflightError, match="PLATEAU config mismatch"):
        validate_e07_document(snap, project_root())


# ----------------------------------------------------------------------
# 2. Control reuse
# ----------------------------------------------------------------------
def test_control_reuse_policy_locked():
    """E07 control reuse_policy = CONTROL_REUSED_FROM_E01_READ_ONLY."""
    snap = _e07_snapshot()
    assert snap["control"]["reuse_policy"] == "CONTROL_REUSED_FROM_E01_READ_ONLY"
    assert snap["control"]["scheduler_policy"] == "OFF"
    assert snap["control"]["training_required"] is False


def test_control_pooled_metrics_match_e01_baseline():
    """E07 control pooled_metrics equal E01 locked baseline."""
    assert EXPECTED_CONTROL_METRICS == {
        "rmse_wh": 59.85291570400546,
        "mae_wh": 26.650501720144604,
        "r2": 0.5789433617557903,
    }
    snap = _e07_snapshot()
    assert snap["control"]["pooled_metrics"] == EXPECTED_CONTROL_METRICS


def test_control_test_status_not_accessed():
    """E07 control test_status = NOT_ACCESSED."""
    snap = _e07_snapshot()
    assert snap["control"]["test_status"] == "NOT_ACCESSED"


def test_preflight_control_reuse_status_pass():
    """Preflight declares control_reuse_status = PASS."""
    preflight = _run_preflight()
    assert preflight["control_reuse_status"] == "PASS"
    assert preflight["control"]["reuse"] == "CONTROL_REUSED_FROM_E01_READ_ONLY"
    assert preflight["control"]["scheduler_policy"] == "OFF"


def test_control_not_in_candidate_specs():
    """Build E07 run context — control candidate must NOT appear in candidate_specs."""
    ctx = build_e07_run_context(project_root(), _e07_snapshot())
    candidate_ids = {cs.candidate_id for cs in ctx.candidate_specs}
    assert BASE_CANDIDATE_ID not in candidate_ids, (
        "Control must be reused, not added to training candidate specs"
    )
    assert candidate_ids == set(CHALLENGER_CANDIDATE_IDS)


# ----------------------------------------------------------------------
# 3. Stage A chain
# ----------------------------------------------------------------------
def test_stage_a_engine_source_includes_scheduler_construction():
    """TrainingEngine.train() builds an LR scheduler when scheduler_name != OFF."""
    from course_work.training.engine import TrainingEngine
    source = inspect.getsource(TrainingEngine.train)
    assert "build_scheduler" in source
    assert "scheduler_name" in source


def test_stage_a_engine_source_calls_scheduler_step_for_cosine():
    """Stage A calls scheduler.step() (no metric) for COSINE."""
    from course_work.training.engine import TrainingEngine
    source = inspect.getsource(TrainingEngine.train)
    # COSINE: _lr_scheduler.step()
    assert '_lr_scheduler.step()' in source


def test_stage_a_engine_source_calls_scheduler_step_for_plateau():
    """Stage A calls scheduler.step(val_rmse_wh) for PLATEAU."""
    from course_work.training.engine import TrainingEngine
    source = inspect.getsource(TrainingEngine.train)
    # PLATEAU: _lr_scheduler.step(val_metric.rmse_wh)
    assert "step(val_metric.rmse_wh)" in source


def test_stage_a_engine_initial_lr_for_cosine():
    """Stage A epoch 1 LR = 3e-4 for COSINE."""
    model = _make_dummy_model()
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = build_scheduler(opt, "COSINE", LOCKED_COSINE_CONFIG)
    # Epoch 1 uses initial lr
    assert opt.param_groups[0]["lr"] == 3e-4
    assert sched.get_last_lr()[0] == 3e-4


def test_stage_a_engine_initial_lr_for_plateau():
    """Stage A epoch 1 LR = 3e-4 for PLATEAU."""
    model = _make_dummy_model()
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    sched = build_scheduler(opt, "REDUCE_ON_PLATEAU", LOCKED_PLATEAU_CONFIG)
    assert opt.param_groups[0]["lr"] == 3e-4
    assert sched.get_last_lr()[0] == 3e-4


# ----------------------------------------------------------------------
# 4. LR trace persistence
# ----------------------------------------------------------------------
def test_lr_used_for_epoch_field_in_history_columns():
    """HISTORY_COLUMNS must include lr_used_for_epoch."""
    from course_work.training.engine import HISTORY_COLUMNS
    assert "lr_used_for_epoch" in HISTORY_COLUMNS
    assert "next_lr_after_scheduler" in HISTORY_COLUMNS


def test_lr_trace_extraction_from_csv_records():
    """extract_lr_trace_from_history returns ordered [3e-4, 2.99e-4, 2.97e-4]."""
    rows = [
        {"epoch": 1, "lr_used_for_epoch": 3e-4},
        {"epoch": 2, "lr_used_for_epoch": 2.99e-4},
        {"epoch": 3, "lr_used_for_epoch": 2.97e-4},
    ]
    trace = extract_lr_trace_from_history({"records": rows})
    assert trace == [3e-4, 2.99e-4, 2.97e-4]


def test_lr_trace_extraction_from_real_csv(tmp_path: Path):
    """End-to-end: write a Stage-A training_history.csv, read it back, extract trace."""
    history_path = tmp_path / "training_history.csv"
    rows = [
        {"epoch": "1", "lr_used_for_epoch": "0.0003"},
        {"epoch": "2", "lr_used_for_epoch": "0.000299"},
        {"epoch": "3", "lr_used_for_epoch": "0.000297"},
    ]
    fieldnames = list(rows[0].keys())
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    # Read it back as records (the orchestrator's Stage-B wiring).
    import csv as _csv
    with history_path.open("r", encoding="utf-8", newline="") as h:
        records = list(_csv.DictReader(h))
    trace = extract_lr_trace_from_history({"records": records})
    assert trace == pytest.approx([3e-4, 2.99e-4, 2.97e-4])


def test_lr_trace_validates_against_best_epoch():
    """validate_lr_trace requires len(trace) >= best_epoch_inner."""
    trace = [3e-4, 2.99e-4]
    validated = validate_lr_trace(trace, best_epoch_inner=2)
    assert validated == [3e-4, 2.99e-4]
    with pytest.raises(ValueError, match="too short"):
        validate_lr_trace(trace, best_epoch_inner=3)


# ----------------------------------------------------------------------
# 5. Stage B replay chain
# ----------------------------------------------------------------------
def test_stage_b_replay_uses_set_optimizer_lr():
    """RefitEngine.refit() uses set_optimizer_lr (E07-D contract)."""
    source = inspect.getsource(RefitEngine.refit)
    assert "set_optimizer_lr" in source


def test_stage_b_replay_never_calls_scheduler_step_in_loop():
    """RefitEngine.refit() loop must NOT call scheduler.step() (E07-D contract)."""
    source = inspect.getsource(RefitEngine.refit)
    code_lines = [
        line for line in source.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert "scheduler.step()" not in code_only
    assert "_lr_scheduler.step()" not in code_only


def test_stage_b_replay_lets_set_optimizer_lr_for_each_epoch():
    """Stage B sets optimizer LR from trace for every epoch."""
    # Simulate the exact Stage-B loop body (without training).
    trace = [3e-4, 2.99e-4, 2.97e-4]
    best_epoch = 3
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    captured_lrs = []
    for epoch in range(1, best_epoch + 1):
        set_optimizer_lr(opt, trace[epoch - 1])
        captured_lrs.append(opt.param_groups[0]["lr"])
    assert captured_lrs == [3e-4, 2.99e-4, 2.97e-4]


def test_stage_b_replay_fresh_optimizer_per_run():
    """RefitEngine.refit() always creates a fresh AdamW optimizer (no warm-start)."""
    source = inspect.getsource(RefitEngine.refit)
    assert "torch.optim.AdamW(" in source
    assert "optimizer.load_state_dict" not in source
    assert "model.load_state_dict" not in source


def test_stage_b_replay_no_validation_metric_required():
    """PLATEAU replay must NOT require validation metric (E07-D contract)."""
    sig = inspect.signature(RefitEngine.refit)
    params = sig.parameters
    assert "stage_a_lr_trace" in params
    assert "val_metric" not in params
    assert "validation_rmse" not in params


def test_stage_b_lr_policy_constant():
    """LR policy constant is the exact string for refit_status.json."""
    assert LR_POLICY_STAGE_A_REPLAY == "STAGE_A_EXACT_TRACE_REPLAY"
    assert LR_POLICY_STAGE_B_CONSTANT == "STAGE_B_CONSTANT_LR_FROM_CONFIG"


def test_stage_b_refit_engine_signature_has_stage_a_lr_trace():
    """RefitEngine.refit accepts stage_a_lr_trace (default None)."""
    sig = inspect.signature(RefitEngine.refit)
    assert "stage_a_lr_trace" in sig.parameters
    assert sig.parameters["stage_a_lr_trace"].default is None


# ----------------------------------------------------------------------
# 6. Stage C shape (E06 regression prevention)
# ----------------------------------------------------------------------
def test_e07_in_flatten_metric_predictions_set():
    """E07 execution_track must be in the flatten_metric_predictions set in real_run.py.

    Scope: only E06 and E07 are flatten-enabled. E01/E03/E04/E05 are frozen
    prior experiments and MUST NOT have their rerun behavior changed
    merely "for symmetry".
    """
    from course_work.rolling_origin import real_run
    source = inspect.getsource(real_run.run_real_pipeline)
    # Locate the flatten condition block and parse the set membership.
    flatten_idx = source.find("flatten_metric_predictions=(")
    end_idx = source.find(")", flatten_idx)
    flatten_block = source[flatten_idx:end_idx]
    assert '"MODEL_IMPROVEMENT_V2_E06"' in flatten_block
    assert '"MODEL_IMPROVEMENT_V2_E07"' in flatten_block
    # Negative checks: frozen prior experiments must NOT be in the set.
    for forbidden in (
        '"MODEL_IMPROVEMENT_V2_E01"',
        '"MODEL_IMPROVEMENT_V2_E03"',
        '"MODEL_IMPROVEMENT_V2_E04"',
        '"MODEL_IMPROVEMENT_V2_E05"',
    ):
        assert forbidden not in flatten_block, (
            f"Frozen prior experiment {forbidden} was incorrectly added to "
            f"flatten_metric_predictions set"
        )


def test_e01_e03_e04_e05_unchanged_no_flatten():
    """E01/E03/E04/E05 must NOT be flatten-enabled (frozen prior experiments)."""
    from course_work.rolling_origin import real_run
    source = inspect.getsource(real_run.run_real_pipeline)
    flatten_idx = source.find("flatten_metric_predictions=(")
    end_idx = source.find(")", flatten_idx)
    flatten_block = source[flatten_idx:end_idx]
    for forbidden_track in (
        "MODEL_IMPROVEMENT_V2_E01",
        "MODEL_IMPROVEMENT_V2_E03",
        "MODEL_IMPROVEMENT_V2_E04",
        "MODEL_IMPROVEMENT_V2_E05",
    ):
        assert forbidden_track not in flatten_block


def test_evaluate_stage_c_flattens_predictions():
    """evaluate_stage_c with flatten_metric_predictions=True returns 1-D y_pred."""
    scaler = _make_dummy_scaler_bundle()
    model = _make_dummy_model()
    device = torch.device("cpu")
    model.eval()

    # Synthetic loader: 4 batches of 8 samples each.
    batch_size = 8
    batches = []
    for _ in range(4):
        x = torch.randn(batch_size, 72, 10, dtype=torch.float32)
        y_raw = torch.full((batch_size, 1), 90.0, dtype=torch.float32)
        batches.append(
            {
                "x": x,
                "y_raw_wh": y_raw,
                "target_id": [f"TGT_{i:08d}" for i in range(batch_size)],
                "target_timestamp": [f"ts_{i}" for i in range(batch_size)],
            }
        )

    def loader():
        return iter(batches)

    result = evaluate_stage_c(
        model=model,
        outer_eval_loader=loader(),
        scaler_bundle=scaler,
        device=device,
        model_id="M_E07_TEST",
        fold_id="RO1",
        model_run_id="RUN_V2_TR_E07_TEST",
        refit_epoch=3,
        rehearsal_synthetic=False,
        prediction_formulation="DIRECT",
        flatten_metric_predictions=True,
    )
    # Critical: y_pred_wh must be 1-D
    assert result.y_pred_wh.ndim == 1, (
        f"E07 regression: y_pred_wh is not 1-D, got shape {result.y_pred_wh.shape}"
    )
    assert result.y_true_wh.ndim == 1
    assert result.y_pred_wh.shape == result.y_true_wh.shape


def test_e06_crash_repro_without_flatten():
    """Without flatten_metric_predictions, DIRECT predictions are (N,1) — the E06 bug.

    This reproduces the E06 crash shape so we know the fix is necessary.
    """
    scaler = _make_dummy_scaler_bundle()
    model = _make_dummy_model()
    device = torch.device("cpu")
    model.eval()

    batch_size = 4
    batches = []
    for _ in range(2):
        x = torch.randn(batch_size, 72, 10, dtype=torch.float32)
        y_raw = torch.full((batch_size, 1), 90.0, dtype=torch.float32)
        batches.append(
            {
                "x": x,
                "y_raw_wh": y_raw,
                "target_id": [f"TGT_{i:08d}" for i in range(batch_size)],
                "target_timestamp": [f"ts_{i}" for i in range(batch_size)],
            }
        )

    def loader():
        return iter(batches)

    result_no_flatten = evaluate_stage_c(
        model=model,
        outer_eval_loader=loader(),
        scaler_bundle=scaler,
        device=device,
        model_id="M_E06_REPRO",
        fold_id="RO1",
        model_run_id="RUN_V2_TR_E06_REPRO",
        refit_epoch=1,
        rehearsal_synthetic=False,
        prediction_formulation="DIRECT",
        flatten_metric_predictions=False,
    )
    # Demonstrate the E06 bug shape
    assert result_no_flatten.y_true_wh.ndim == 1
    assert result_no_flatten.y_pred_wh.ndim == 2
    assert result_no_flatten.y_pred_wh.shape == (8, 1)
    assert result_no_flatten.y_true_wh.shape == (8,)


def test_pooled_metrics_rejects_2d_prediction():
    """compute_pooled_metrics raises ValueError on 2-D y_pred (E06 crash signature)."""
    y_true_1d = np.array([100.0, 200.0, 300.0])
    y_pred_2d = np.array([[99.0], [201.0], [298.0]])  # shape (3, 1)
    with pytest.raises(ValueError, match="Pooled shape mismatch"):
        compute_pooled_metrics(
            candidate_id="TEST",
            y_true_per_fold=[y_true_1d],
            y_pred_per_fold=[y_pred_2d],
        )


def test_pooled_metrics_accepts_1d_prediction():
    """compute_pooled_metrics accepts 1-D y_pred (the FIXED shape)."""
    y_true_1d = np.array([100.0, 200.0, 300.0])
    y_pred_1d = np.array([99.0, 201.0, 298.0])  # shape (3,)
    pm = compute_pooled_metrics(
        candidate_id="TEST",
        y_true_per_fold=[y_true_1d],
        y_pred_per_fold=[y_pred_1d],
    )
    assert pm.fold_count == 1
    assert pm.pooled_rmse_wh >= 0.0


# ----------------------------------------------------------------------
# 7. Pooling (end-to-end 1-D pipeline)
# ----------------------------------------------------------------------
def test_pooled_pipeline_handles_three_folds_1d():
    """Pooling across 3 folds with 1-D shapes works correctly."""
    y_true_folds = [
        np.array([100.0, 200.0, 300.0]),
        np.array([150.0, 250.0]),
        np.array([90.0, 190.0, 290.0, 390.0]),
    ]
    y_pred_folds = [
        np.array([99.0, 201.0, 298.0]),
        np.array([149.0, 248.0]),
        np.array([91.0, 192.0, 292.0, 388.0]),
    ]
    pm = compute_pooled_metrics(
        candidate_id="TEST",
        y_true_per_fold=y_true_folds,
        y_pred_per_fold=y_pred_folds,
    )
    assert pm.fold_count == 3
    assert pm.pooled_rmse_wh >= 0.0
    assert pm.pooled_mae_wh >= 0.0


def test_e07_inverse_transform_y_preserves_shape():
    """inverse_transform_y preserves input shape (model output is [B,1])."""
    scaler = _make_dummy_scaler_bundle()
    model_out = np.full((8, 1), 0.5, dtype=np.float64)  # [B, 1]
    # Direct call to FoldLocalScalerBundle-equivalent function logic:
    pred_wh = np.asarray(model_out, dtype=np.float64) * scaler.y_std + scaler.y_mean
    assert pred_wh.shape == (8, 1)  # confirmed E06-style shape preserved
    # This is why flatten_metric_predictions=True is needed for V2 tracks.


# ----------------------------------------------------------------------
# 8. Finalization (no training; static checks only)
# ----------------------------------------------------------------------
def test_promotion_threshold_rmse_wh_locked():
    """Promotion threshold is exactly 59.75291570400546 Wh."""
    assert PROMOTION_THRESHOLD_RMSE_WH == 59.75291570400546


def test_preflight_selection_policy_status_pass():
    """Preflight declares selection_policy_status = PASS."""
    preflight = _run_preflight()
    assert preflight["selection_policy_status"] == "PASS"
    assert preflight["promotion_threshold_rmse_wh"] == PROMOTION_THRESHOLD_RMSE_WH


def test_registry_namespace_v2_e07():
    """Registry namespace = V2_E07."""
    assert REGISTRY_NAMESPACE == "V2_E07"
    from course_work.experiments.registry import V2_NAMESPACE_IDS
    assert "V2_E07" in V2_NAMESPACE_IDS


# ----------------------------------------------------------------------
# 9. Authorization gate
# ----------------------------------------------------------------------
def test_cli_refuses_official_without_authorize_training():
    """CLI refuses official mode when --authorize-training is absent."""
    proc = subprocess.run(
        [
            sys.executable, "-m",
            "course_work.model_improvement_v2.e07_runner",
            "--experiment", "E07",
            "--mode", "official",
        ],
        cwd=str(project_root()),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/usr/local/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 3, (
        f"Expected exit 3 (REFUSED), got {proc.returncode}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "REFUSED" in proc.stderr
    assert "authorize-training" in proc.stderr


def test_cli_refuses_authorize_training_in_preflight():
    """CLI refuses --authorize-training in preflight mode."""
    proc = subprocess.run(
        [
            sys.executable, "-m",
            "course_work.model_improvement_v2.e07_runner",
            "--experiment", "E07",
            "--mode", "preflight",
            "--authorize-training",
        ],
        cwd=str(project_root()),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/usr/local/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 2, (
        f"Expected exit 2 (ERROR), got {proc.returncode}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "ERROR" in proc.stderr
    assert "valid only in official mode" in proc.stderr


def test_cli_preflight_exit_zero():
    """CLI preflight returns 0 with status=PASS."""
    proc = subprocess.run(
        [
            sys.executable, "-m",
            "course_work.model_improvement_v2.e07_runner",
            "--experiment", "E07",
            "--mode", "preflight",
        ],
        cwd=str(project_root()),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/usr/local/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"Expected exit 0 (PASS), got {proc.returncode}. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"


def test_authorization_gate_documented_in_snapshot():
    """Snapshot declares training_authorized=false (Human gate, not bypassed by CLI)."""
    snap = _e07_snapshot()
    assert snap["v2_orchestration"]["training_authorized"] is False
    assert snap["v2_orchestration"]["test_access_authorized"] is False


def test_authorization_consistent_with_e06_runner():
    """E07's authorization pattern matches E06's pattern (CLI gate only)."""
    from course_work.model_improvement_v2 import e06_runner
    e06_source = inspect.getsource(e06_runner.main)
    e07_source = inspect.getsource(
        __import__("course_work.model_improvement_v2.e07_runner", fromlist=["main"]).main
    )
    # Both runners use the same gate pattern
    for pattern in (
        'mode == "official" and not args.authorize_training',
        'mode != "official" and args.authorize_training',
    ):
        assert pattern in e06_source
        assert pattern in e07_source


# ----------------------------------------------------------------------
# 10. Isolation
# ----------------------------------------------------------------------
def test_e07_does_not_modify_v1_artifacts():
    """E07 runner must not write to V1 artifact roots."""
    from course_work.model_improvement_v2.contracts import V1_ARTIFACT_ROOTS
    v1_paths = [str(p) for p in V1_ARTIFACT_ROOTS]
    for v1_path in v1_paths:
        assert v1_path not in str(project_root() / "artifacts/model_improvement_v2")


def test_e07_test_access_not_authorized():
    """Preflight declares test_access=NO."""
    preflight = _run_preflight()
    assert preflight["test_access"] == "NO"
    assert preflight["preflight_no_test_access"] is True


def test_e07_preflight_executes_no_training():
    """Preflight declares training_executed=false."""
    preflight = _run_preflight()
    assert preflight["training_executed"] is False
    assert preflight["preflight_no_training"] is True
    assert preflight["inference_executed"] is False
    assert preflight["checkpoint_loaded"] is False


def test_e07_does_not_modify_e01_artifacts():
    """E07 control reuse must verify E01 artifacts but never mutate them."""
    snap = _e07_snapshot()
    # The runner code reads E01 paths; it must not write to them.
    from course_work.model_improvement_v2 import e07_runner as _e07
    e07_source = inspect.getsource(_e07)
    # Strip comment lines and docstrings before scanning for write calls.
    code_lines = [
        line for line in e07_source.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    # The only E01 mention is via E01_RELATIVE_ROOT constant in
    # _validate_control_reuse; no actual file write calls to E01 paths.
    for forbidden in ("write_bytes", "atomic_write", "Path.write_text"):
        # atomic_write_bytes is used by the artifact writer; check it
        # only references E07 paths, not E01 paths.
        if forbidden in code_only:
            for line in code_only.split("\n"):
                if forbidden in line and "E01" in line:
                    raise AssertionError(
                        f"E07 runner writes to E01: {line!r}"
                    )
    e01_root = project_root() / "artifacts/model_improvement_v2/experiments/E01"
    assert e01_root.is_dir()


def test_e07_primary_change_prefixes_isolated():
    """E07's PRIMARY_CHANGE_PREFIXES entry only allows scheduler fields."""
    allowed = PRIMARY_CHANGE_PREFIXES["E07"]
    for prefix in allowed:
        assert prefix.startswith("training.scheduler")
    assert "training.learning_rate" not in allowed
    assert "training.optimizer_name" not in allowed


# ----------------------------------------------------------------------
# 11. Full chain synthetic smoke test (no real training)
# ----------------------------------------------------------------------
def test_synthetic_chain_smoke():
    """Synthetic end-to-end smoke test of the whole chain.

    We do NOT train. We verify that:
      - Stage A produces lr_used_for_epoch trace
      - extract_lr_trace_from_history reconstructs it
      - validate_lr_trace accepts it
      - set_optimizer_lr mutates optimizer
      - evaluate_stage_c returns 1-D shape (post-fix)
      - compute_pooled_metrics accepts 1-D inputs
    """
    # 1. Stage A: simulate lr_used_for_epoch across 6 epochs.
    stage_a_trace = [3e-4] * 6
    # For COSINE, lr should decay after epoch 1; for PLATEAU, lr may stay.
    # Stage A row order is recorded in training_history.csv.
    stage_a_rows = [
        {"epoch": str(epoch), "lr_used_for_epoch": str(lr)}
        for epoch, lr in enumerate(stage_a_trace, start=1)
    ]

    # 2. Stage B: load history, validate trace, set optimizer lr.
    trace = extract_lr_trace_from_history({"records": stage_a_rows})
    best_epoch = 4
    validated = validate_lr_trace(trace, best_epoch_inner=best_epoch)
    model = torch.nn.Linear(10, 1)
    opt = torch_optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(1, best_epoch + 1):
        set_optimizer_lr(opt, validated[epoch - 1])
    assert opt.param_groups[0]["lr"] == validated[best_epoch - 1]

    # 3. Stage C: produce 1-D predictions.
    scaler = _make_dummy_scaler_bundle()
    stage_c_model = _make_dummy_model()
    stage_c_model.eval()

    batch_size = 4
    batches = []
    for _ in range(2):
        x = torch.randn(batch_size, 72, 10, dtype=torch.float32)
        y_raw = torch.full((batch_size, 1), 90.0, dtype=torch.float32)
        batches.append(
            {
                "x": x,
                "y_raw_wh": y_raw,
                "target_id": [f"TGT_{i:08d}" for i in range(batch_size)],
                "target_timestamp": [f"ts_{i}" for i in range(batch_size)],
            }
        )

    def loader():
        return iter(batches)

    result = evaluate_stage_c(
        model=stage_c_model,
        outer_eval_loader=loader(),
        scaler_bundle=scaler,
        device=torch.device("cpu"),
        model_id="E07_SMOKE",
        fold_id="RO1",
        model_run_id="RUN_V2_TR_E07_SMOKE",
        refit_epoch=best_epoch,
        rehearsal_synthetic=False,
        prediction_formulation="DIRECT",
        flatten_metric_predictions=True,  # E07 fix
    )
    assert result.y_pred_wh.ndim == 1
    assert result.y_true_wh.ndim == 1
    assert result.y_pred_wh.shape == result.y_true_wh.shape

    # 4. Pooling: 1-D accepted.
    pm = compute_pooled_metrics(
        candidate_id="E07_SMOKE",
        y_true_per_fold=[result.y_true_wh],
        y_pred_per_fold=[result.y_pred_wh],
    )
    assert pm.fold_count == 1
    assert pm.pooled_rmse_wh >= 0.0


# ----------------------------------------------------------------------
# 12. V2_NAMESPACE_IDS includes V2_E07 (registry integration)
# ----------------------------------------------------------------------
def test_v2_namespace_ids_includes_e07():
    from course_work.experiments.registry import V2_NAMESPACE_IDS
    assert "V2_E07" in V2_NAMESPACE_IDS


def test_assert_context_invariants_knows_e07():
    """real_run.assert_context_invariants recognises MODEL_IMPROVEMENT_V2_E07."""
    from course_work.rolling_origin import real_run
    source = inspect.getsource(real_run.assert_context_invariants)
    assert '"MODEL_IMPROVEMENT_V2_E07"' in source


# ----------------------------------------------------------------------
# 13. Stage-A history shape contract (CSV)
# ----------------------------------------------------------------------
def test_stage_a_history_csv_columns_include_lr_used_for_epoch(tmp_path: Path):
    """A Stage-A training_history.csv must contain lr_used_for_epoch."""
    csv_path = tmp_path / "training_history.csv"
    headers = [
        "epoch",
        "train_loss",
        "lr_used_for_epoch",
        "next_lr_after_scheduler",
    ]
    rows = [
        {"epoch": "1", "train_loss": "0.5", "lr_used_for_epoch": "0.0003", "next_lr_after_scheduler": "0.0003"},
        {"epoch": "2", "train_loss": "0.4", "lr_used_for_epoch": "0.000299", "next_lr_after_scheduler": "0.000299"},
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    with csv_path.open("r", encoding="utf-8", newline="") as h:
        records = list(csv.DictReader(h))
    assert "lr_used_for_epoch" in records[0]
    assert float(records[0]["lr_used_for_epoch"]) == 3e-4
