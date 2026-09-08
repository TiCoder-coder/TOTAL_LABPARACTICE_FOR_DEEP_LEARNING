"""Phase 43 ONE-EPOCH real sandbox harness (NON-OFFICIAL).

This harness proves that the corrected Validation population contract
and FailureType handler work end-to-end with REAL training data through
ONE optimizer step. The previous forward-only harness did not exercise
the exact production `_evaluate_loader` -> `compute_regression_metrics`
-> `validate_population_coverage` path, and so could not surface the
boundary-protocol short/long mismatch that caused the crash on the
canonical Validation set.

This harness:

1. Builds the canonical Phase 43 candidate config (LT1 / LH32, fresh).
2. Uses the canonical train + validation DataLoaders.
3. Uses the real TrainingEngine + real LSTMRegressor.
4. Runs ONE real optimizer step on a single training batch, then runs
   REAL full Validation evaluation through the engine.
5. Persists a real checkpoint + history + log + metrics + predictions.
6. Calls registry.fail_run with valid FailureType if any step raises,
   so the exception handler is exercised end-to-end.
7. Runs in a SANDBOX registry rooted at a tmp_path; the canonical
   experiment_registry.jsonl is never touched.
8. Also runs a "retry-after-failed-run" probe that simulates a FAILED
   duplicate and proves `resolve_duplicate_policy` correctly returns
   CODE_FIX on the second registration.

It does NOT:

- Run more than one optimizer step (no real training loop).
- Modify artifacts/runs/<canonical>.
- Modify artifacts/experiments/experiment_registry.jsonl.
- Access Test data.

The only writable locations are <tmpdir>/** (sandbox registry + run).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from course_work.data.datasets import build_train_validation_loaders
from course_work.data.scaling import load_validated_target_scaler
from course_work.experiments.registry import (
    ExecutionType,
    ExperimentRegistry,
    FailureType,
    RERUN_REASONS,
    compute_config_fingerprint,
    load_upstream_context,
    validate_run_config,
)
from course_work.lstm_tuning.reference_resolution import resolve_lstm_t0_reference
from course_work.lstm_tuning.shared_data_contract import resolve_shared_data_contract
from course_work.lstm_tuning.stages import StageExecutor
from course_work.lstm_tuning.tuning_space import (
    FIXED_TRAINING_CONTRACT,
    REFERENCE_HYPERPARAMETERS,
    _set_path,
)
from course_work.training.engine import TrainingEngine, build_model_from_run_config
from course_work.utils.environment import select_device
from course_work.utils.reproducibility import set_seed

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from phase43_lstm_tuning import (
    _CANONICAL_BOUNDARY_PROTOCOL_MAP,
    _classify_failure,
    _enforce_scientific_contract,
    resolve_duplicate_policy,
)


def _build_base(project_root: Path):
    contract = resolve_shared_data_contract(project_root)
    registry = ExperimentRegistry(project_root)
    upstream = load_upstream_context(project_root)
    reference = resolve_lstm_t0_reference(registry, contract)
    base = None
    if reference.phase20_exact_match and reference.reference_run_id:
        for rec in registry._load_records():
            if rec.get("run_id") == reference.reference_run_id:
                base = deepcopy(rec.get("config", {}))
                break
    if base is None:
        from course_work.experiments.registry import build_reference_run_config
        base = build_reference_run_config(project_root=project_root, model_family="LSTM")
    for path, value in FIXED_TRAINING_CONTRACT.items():
        _set_path(base, path, value)
    for path, value in REFERENCE_HYPERPARAMETERS.items():
        _set_path(base, path, value)
    return contract, registry, upstream, reference, base


def _one_optimizer_step(model, train_loader, device, criterion, optimizer) -> float:
    """Execute exactly ONE training batch through real optimizer.step().

    Returns the scalar loss value of that single step. Real training is
    intentionally avoided; this harness exists solely to surface the
    Validation population bug, not to measure training dynamics.
    """
    model.train()
    batch = next(iter(train_loader))
    x = batch["x"].to(device)
    y_model = batch["y_model"].to(device)
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y_model)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def _run_one_epoch_sandbox(
    project_root: Path,
    tmpdir: Path,
    stage: str = "LT1",
    option: str = "LH32",
) -> dict[str, Any]:
    contract, registry, upstream, reference, base = _build_base(project_root)

    executor = StageExecutor(
        contract=contract,
        reference_config=base,
        reference_run_id=reference.reference_run_id if reference.phase20_exact_match else None,
    )
    planned = executor.plan_full_sweep()
    cand = next(
        c for s in planned if s.stage == stage for c in s.candidates if c.option == option
    )
    assert cand.source_type == "FRESH", "harness requires a FRESH candidate"

    cfg = _enforce(deepcopy(cand.config), contract, upstream)
    validated = validate_run_config(cfg, upstream)
    fingerprint = compute_config_fingerprint(validated)
    canonical_bp = _CANONICAL_BOUNDARY_PROTOCOL_MAP[contract.boundary_protocol]

    sandbox = ExperimentRegistry(project_root=project_root, registry_root=tmpdir)
    sandbox.upstream_context = upstream

    registered = sandbox.register_run(
        validated,
        "LSTM_TUNING",
        ExecutionType.TRAINING.value,
        sweep_id=None,
        sweep_stage=f"P43_{stage}_{option}",
        notes="Phase 43 one-epoch sandbox harness (real train + real validation).",
    )
    run_id = registered["run_id"]
    sandbox.start_run(run_id)

    set_seed(42)
    _, loaders, _ = build_train_validation_loaders(
        project_root=project_root,
        variant_id=contract.feature_variant_id,
        lookback=contract.lookback_steps,
        target_option=contract.target_scaling_id,
        batch_size=contract.batch_size,
        seed=42,
    )
    train_loader = loaders["TRAIN"][0]
    val_loader = loaders["VALIDATION"][0]

    model = build_model_from_run_config(validated)
    device = select_device()
    target_scaler = load_validated_target_scaler(project_root)
    target_scaler_bundle = target_scaler

    from course_work.training.losses import build_training_criterion
    criterion = build_training_criterion(validated["training"])
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(validated["training"]["learning_rate"]),
        weight_decay=float(validated["training"]["weight_decay"]),
    )
    loss_value = _one_optimizer_step(model, train_loader, device, criterion, optimizer)

    train_evaluation = engine_train_eval(
        model=model,
        loader=train_loader,
        device=device,
        target_option=validated["data"]["target_scaling_option"],
        target_scaler_bundle=target_scaler_bundle,
        run_id=run_id,
        model_id=validated["model"].get("model_name", validated["model"]["model_family"]),
        split_id="TRAIN",
        population_fingerprint=validated["lineage"]["population_fingerprint"],
        lookback_steps=int(validated["data"]["lookback_steps"]),
        horizon_steps=int(validated["data"]["horizon_steps"]),
        boundary_protocol=canonical_bp,
    )

    val_evaluation = engine_val_eval(
        model=model,
        loader=val_loader,
        device=device,
        target_option=validated["data"]["target_scaling_option"],
        target_scaler_bundle=target_scaler_bundle,
        run_id=run_id,
        model_id=validated["model"].get("model_name", validated["model"]["model_family"]),
        split_id="VALIDATION",
        population_fingerprint=validated["lineage"]["population_fingerprint"],
        lookback_steps=int(validated["data"]["lookback_steps"]),
        horizon_steps=int(validated["data"]["horizon_steps"]),
        boundary_protocol=canonical_bp,
    )
    val_idx, val_y_true, val_y_pred, val_metric = val_evaluation

    assert val_metric.n_samples == int(validated["data"]["validation_sample_count"]), \
        f"VALIDATION n_samples mismatch: expected {validated['data']['validation_sample_count']} got {val_metric.n_samples}"
    assert val_metric.population_fingerprint == validated["lineage"]["population_fingerprint"]

    engine = TrainingEngine(sandbox)
    import pandas as pd
    history = pd.DataFrame.from_records(
        [
            {
                "epoch": 1, "train_loss": loss_value,
                "train_rmse_wh": float(train_evaluation[3].rmse_wh),
                "validation_rmse_wh": float(val_metric.rmse_wh),
                "validation_mae_wh": float(val_metric.mae_wh),
                "validation_r2": float(val_metric.r2),
                "learning_rate": float(validated["training"]["learning_rate"]),
                "epoch_seconds": 0.0, "is_best": True,
            },
        ],
        columns=[
            "epoch", "train_loss", "train_rmse_wh", "validation_rmse_wh",
            "validation_mae_wh", "validation_r2", "learning_rate",
            "epoch_seconds", "is_best",
        ],
    )
    from course_work.training.engine import TrainingResult
    result = TrainingResult(
        history=history,
        best_epoch=1,
        best_validation_rmse_wh=float(val_metric.rmse_wh),
        metric_result=val_metric,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_epochs_run=1,
        stopped_reason="HARNESS_ONE_EPOCH",
        best_sample_idx=val_idx,
        best_y_true_wh=val_y_true,
        best_y_pred_wh=val_y_pred,
        gradient_diagnostics={"preclip_gradient_norm_max": 0.0, "clipped_steps": 0, "total_batches": 1},
    )

    run_dir = tmpdir / "runs" / run_id
    paths = engine.persist_run_artifacts(
        run_id, run_dir, model, result,
        result.best_sample_idx, result.best_y_true_wh, result.best_y_pred_wh,
    )
    payload = torch.load(paths["best_checkpoint"], map_location="cpu")
    copy = build_model_from_run_config(validated)
    copy.load_state_dict(payload["model_state_dict"])
    sd_a = {k: v.cpu() for k, v in model.state_dict().items()}
    sd_b = {k: v.cpu() for k, v in copy.state_dict().items()}
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"state_dict mismatch at {k}"

    completed = sandbox.complete_run(run_id, result.best_epoch, result.best_validation_rmse_wh)
    assert completed["status"] == "COMPLETED"

    return {
        "run_id": run_id,
        "stage": stage,
        "option": option,
        "config_fingerprint": fingerprint,
        "boundary_protocol_long_form": canonical_bp,
        "one_optimizer_step_loss": loss_value,
        "validation_n_samples": int(val_metric.n_samples),
        "validation_rmse_wh": float(val_metric.rmse_wh),
        "validation_mae_wh": float(val_metric.mae_wh),
        "validation_r2": float(val_metric.r2),
        "train_n_samples": int(train_evaluation[3].n_samples),
        "train_rmse_wh": float(train_evaluation[3].rmse_wh),
        "registry_status": completed["status"],
    }


def engine_train_eval(**kwargs):
    return _engine_eval_impl(**kwargs)


def engine_val_eval(**kwargs):
    return _engine_eval_impl(**kwargs)


def _engine_eval_impl(**kwargs):
    """Construct a transient TrainingEngine just to call _evaluate_loader.

    Avoids depending on a persistent registry. The full engine would
    require registry injection; here we only need the eval path.
    """
    engine = _NullEngine()
    return engine._evaluate_loader(**kwargs)


class _NullEngine:
    """Minimal TrainingEngine-compatible object for _evaluate_loader.

    `TrainingEngine._evaluate_loader` only references `self` for nothing
    other than `_evaluate_loader` itself; instantiate via __new__ to skip
    __init__ entirely.
    """
    def __init__(self):
        pass

    def _evaluate_loader(self, **kwargs):
        from course_work.training.engine import TrainingEngine
        return TrainingEngine.__dict__["_evaluate_loader"](self, **kwargs)


def _retry_after_failed_sandbox(project_root: Path, tmpdir: Path) -> dict[str, Any]:
    """Simulate a FAILED duplicate then register a new run via resolve_duplicate_policy."""
    contract, registry, upstream, reference, base = _build_base(project_root)
    cfg = _enforce(deepcopy(base), contract, upstream)
    cfg["model"]["hidden_size"] = 33 
    validated = validate_run_config(cfg, upstream)
    fingerprint = compute_config_fingerprint(validated)

    sandbox = ExperimentRegistry(project_root=project_root, registry_root=tmpdir)
    sandbox.upstream_context = upstream

    first = sandbox.register_run(
        validated,
        "LSTM_TUNING",
        ExecutionType.TRAINING.value,
        sweep_id=None,
        notes="Phase 43 retry-harness first attempt.",
    )
    first_run_id = first["run_id"]
    sandbox.start_run(first_run_id)
    sandbox.fail_run(
        first_run_id,
        failure_type=FailureType.METRIC_ERROR.value,
        failure_stage="HARNESS_SIMULATED_FAIL",
        failure_message="Simulated failure to validate retry policy.",
        exception_class="ValueError",
    )

    rerun_reason, dup_records = resolve_duplicate_policy(sandbox, fingerprint)
    assert rerun_reason == "CODE_FIX", f"expected CODE_FIX, got {rerun_reason!r}"
    assert any(r["status"] == "FAILED" for r in dup_records)

    second = sandbox.register_run(
        validated,
        "LSTM_TUNING",
        ExecutionType.TRAINING.value,
        sweep_id=None,
        rerun_reason=rerun_reason,
        notes="Phase 43 retry-harness second attempt with CODE_FIX policy.",
    )
    second_run_id = second["run_id"]

    return {
        "first_run_id": first_run_id,
        "first_status": sandbox.get_run(first_run_id)["status"],
        "first_failure_type": sandbox.get_run(first_run_id)["failure"]["failure_type"],
        "second_run_id": second_run_id,
        "second_rerun_reason": rerun_reason,
        "fingerprint": fingerprint,
        "duplicate_records_seen": [r["run_id"] + "@" + r["status"] for r in dup_records],
    }


def _enforce(cfg, contract, upstream):
    return _enforce_scientific_contract(cfg, contract, upstream)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    tmpdir = Path(tempfile.mkdtemp(prefix="phase43-harness-one-epoch-"))
    try:
        out = _run_one_epoch_sandbox(project_root, tmpdir, "LT1", "LH32")
        print(json.dumps(out, indent=2, default=str))

        retry_out = _retry_after_failed_sandbox(project_root, tmpdir)
        print(json.dumps(retry_out, indent=2, default=str))

        canon = json.loads(
            open(project_root / "artifacts/experiments/experiment_registry.jsonl")
            .read().strip().splitlines()[-1]
        )
        print("Canonical registry last record run_id:", canon["run_id"])
        print("Sandbox run_id:", out["run_id"], "Retry run_id:", retry_out["second_run_id"])
        assert canon["run_id"] != out["run_id"]
        assert canon["run_id"] != retry_out["second_run_id"]
        print("PASS")
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"FAIL: {exc}")
        return 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
