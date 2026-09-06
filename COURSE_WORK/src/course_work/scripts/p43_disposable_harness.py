"""Phase 43 end-to-end path harness (NON-OFFICIAL, sandboxed).

This harness exists to prove that the full Phase 43 scientific execution
path is wired correctly BEFORE the human attempts official training.

It exercises:

- Real LSTM model construction from a Phase 43 candidate config.
- Real TRAIN/VALIDATION DataLoader construction on the canonical dataset.
- Real scalers (read-only).
- Real forward pass + Real _evaluate_loader() on the canonical Validation
  set with the corrected model (no backward, no optimizer step, no
  training loop).
- Real registry.register_run → register_metric → register_artifact →
  complete_run on a SANDBOX ExperimentRegistry rooted in tmp_path.
  The canonical experiment_registry.jsonl is NEVER touched.
- Real persist_run_artifacts schema: writes a checkpoint .pt matching the
  engine's payload shape, training_history.csv, training.log,
  metrics/best_validation_metrics.json and predictions CSV.
- Strict checkpoint reload: torch.load on the same payload and verify
  the resulting state_dict matches model.state_dict().

It does NOT:

- Run an actual training loop (no optimizer.step).
- Modify any artifact under artifacts/runs/<canonical>.
- Modify artifacts/experiments/experiment_registry.jsonl.
- Modify any file under artifacts/scalers, artifacts/windows, etc.
- Access Test data.

The only writable filesystem locations are:
- <tmp_path>/experiment_registry.jsonl (sandbox registry)
- <tmp_path>/runs/RUN_LS_LST_9999_*/...  (sandbox runs)
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import numpy as np

from course_work.data.datasets import build_train_validation_loaders
from course_work.data.scaling import load_validated_target_scaler
from course_work.experiments.registry import (
    ExperimentRegistry,
    ExecutionType,
    RERUN_REASONS,
    compute_config_fingerprint,
    validate_run_config,
    load_upstream_context,
)
from course_work.lstm_tuning.shared_data_contract import resolve_shared_data_contract
from course_work.lstm_tuning.reference_resolution import resolve_lstm_t0_reference
from course_work.lstm_tuning.stages import StageExecutor
from course_work.lstm_tuning.tuning_space import (
    FIXED_TRAINING_CONTRACT,
    REFERENCE_HYPERPARAMETERS,
    _set_path,
)
from course_work.training.engine import (
    TrainingEngine,
    TrainingResult,
    build_model_from_run_config,
)
from course_work.utils.environment import select_device
from course_work.utils.reproducibility import set_seed


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


def _enforce(cfg, contract, upstream):
    from scripts.phase43_lstm_tuning import _enforce_scientific_contract
    cfg = deepcopy(cfg)
    return _enforce_scientific_contract(cfg, contract, upstream)


def _run_one_candidate_in_sandbox(
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

    cfg = _enforce(cand.config, contract, upstream)
    validated = validate_run_config(cfg, upstream)
    fingerprint = compute_config_fingerprint(validated)

    # === SANDBOX REGISTRY ===
    sandbox = ExperimentRegistry(project_root=project_root, registry_root=tmpdir)
    sandbox.upstream_context = upstream

    registered = sandbox.register_run(
        validated,
        "LSTM_TUNING",
        ExecutionType.TRAINING.value,
        sweep_id=None,
        sweep_stage=f"P43_{stage}_{option}",
        notes="Phase 43 corrective end-to-end harness (sandbox).",
    )
    run_id = registered["run_id"]
    sandbox.start_run(run_id)

    # === REAL DATA ===
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
    assert train_loader.batch_size == contract.batch_size
    assert val_loader.batch_size == contract.batch_size

    # === REAL MODEL ===
    model = build_model_from_run_config(validated)
    # First LSTM layer weight_ih has shape [4*hidden_size, input_size]
    first_param = next(iter(model.parameters()))
    assert first_param.shape[1] == validated["model"]["input_size"], \
        f"input_size mismatch: weight_ih shape={tuple(first_param.shape)} input_size={validated['model']['input_size']}"
    assert first_param.shape[0] == 4 * validated["model"]["hidden_size"], \
        f"hidden_size mismatch: weight_ih first dim={first_param.shape[0]} expected={4 * validated['model']['hidden_size']}"

    device = select_device()
    target_scaler = load_validated_target_scaler(project_root)
    engine = TrainingEngine(sandbox)
    model = model.to(device)

    # === REAL _evaluate_loader (no training loop, no optimizer.step) ===
    evaluation = engine._evaluate_loader(
        model=model,
        loader=val_loader,
        device=device,
        target_option=validated["data"]["target_scaling_option"],
        target_scaler_bundle=target_scaler,
        run_id=run_id,
        model_id=validated["model"].get("model_name", validated["model"]["model_family"]),
        split_id="VALIDATION",
        population_fingerprint=validated["lineage"]["population_fingerprint"],
        lookback_steps=int(validated["data"]["lookback_steps"]),
        horizon_steps=int(validated["data"]["horizon_steps"]),
        boundary_protocol=str(validated["data"]["boundary_protocol"]),
    )
    sample_idx, y_true, y_pred, metric = evaluation
    assert metric.n_samples == int(validated["data"]["validation_sample_count"]), \
        f"VALIDATION n_samples mismatch: expected {validated['data']['validation_sample_count']} got {metric.n_samples}"
    assert metric.population_fingerprint == validated["lineage"]["population_fingerprint"]

    # === SYNTHESIZE TrainingResult (0 epochs trained, but evaluate produced a metric) ===
    history_columns = [
        "epoch", "train_loss", "train_rmse_wh", "validation_rmse_wh",
        "validation_mae_wh", "validation_r2", "learning_rate",
        "epoch_seconds", "is_best",
    ]
    history_records = [
        {
            "epoch": 0, "train_loss": None, "train_rmse_wh": None,
            "validation_rmse_wh": float(metric.rmse_wh),
            "validation_mae_wh": float(metric.mae_wh),
            "validation_r2": float(metric.r2),
            "learning_rate": float(validated["training"]["learning_rate"]),
            "epoch_seconds": 0.0, "is_best": True,
        },
    ]
    import pandas as pd
    history = pd.DataFrame.from_records(history_records, columns=history_columns)
    result = TrainingResult(
        history=history,
        best_epoch=1,  # best_epoch must be a positive int per registry.complete_run
        best_validation_rmse_wh=float(metric.rmse_wh),
        metric_result=metric,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_epochs_run=0,
        stopped_reason="HARNESS_NO_TRAINING",
        best_sample_idx=sample_idx,
        best_y_true_wh=y_true,
        best_y_pred_wh=y_pred,
        gradient_diagnostics={"preclip_gradient_norm_max": 0.0, "clipped_steps": 0, "total_batches": 0},
    )

    # === REAL persist_run_artifacts schema ===
    run_dir = tmpdir / "runs" / run_id
    paths = engine.persist_run_artifacts(
        run_id, run_dir, model, result,
        result.best_sample_idx, result.best_y_true_wh, result.best_y_pred_wh,
    )

    # Verify the persisted checkpoint strict-loads and matches the in-memory state dict.
    payload = torch.load(paths["best_checkpoint"], map_location="cpu")
    assert "model_state_dict" in payload
    assert "best_epoch" in payload
    assert "best_validation_rmse_wh" in payload
    assert "criterion_config" in payload
    # Round-trip
    copy = build_model_from_run_config(validated)
    copy.load_state_dict(payload["model_state_dict"])
    sd_a = {k: v.cpu() for k, v in model.state_dict().items()}
    sd_b = {k: v.cpu() for k, v in copy.state_dict().items()}
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"state_dict mismatch at {k}"
    # Round-trip via the same shape key set
    assert set(sd_a.keys()) == set(sd_b.keys())

    # === METRICS: engine.persist_run_artifacts already registered the three
    # VALIDATION metric rows (mae_wh, rmse_wh, r2). Verify they're present.
    rec = sandbox.get_run(run_id)
    persisted = [
        (m["split_id"], m["metric_name"], m["epoch_or_checkpoint"])
        for m in rec["metrics"]
    ]
    assert ("VALIDATION", "mae_wh", f"epoch_{result.best_epoch}") in persisted
    assert ("VALIDATION", "rmse_wh", f"epoch_{result.best_epoch}") in persisted
    assert ("VALIDATION", "r2", f"epoch_{result.best_epoch}") in persisted

    # === COMPLETE ===
    completed = sandbox.complete_run(run_id, result.best_epoch, result.best_validation_rmse_wh)
    assert completed["status"] == "COMPLETED"

    # Persisted file checks
    assert paths["history"].exists() and paths["history"].read_bytes().startswith(b"epoch")
    assert paths["log"].exists()
    assert paths["metrics"].exists() and json.loads(paths["metrics"].read_text())
    assert paths["predictions"].exists()
    assert paths["best_checkpoint"].exists()
    assert paths["last_checkpoint"].exists()

    # Training log content
    log_text = paths["log"].read_text()
    assert "best_epoch=1" in log_text
    assert "stopped_reason=HARNESS_NO_TRAINING" in log_text

    # Sample count, batch size, lookback invariants
    expected_batch_size = int(validated["training"]["batch_size"])
    actual_batch_size = train_loader.batch_size
    assert actual_batch_size == expected_batch_size
    assert int(validated["data"]["lookback_steps"]) == 36
    assert int(validated["data"]["feature_count"]) == 33
    assert validated["data"]["feature_variant_id"] == "FS2_TF1"
    assert validated["data"]["target_scaling_option"] == "YS1"
    assert validated["data"]["test_locked"] is True
    assert validated["data"]["test_access_enabled"] is False
    assert validated["data"]["target_access_mode"] == "VALIDATION"
    assert validated["data"]["boundary_protocol"] == "WB0_CONTEXT_CARRY_OVER"

    return {
        "run_id": run_id,
        "config_fingerprint": fingerprint,
        "stage": stage,
        "option": option,
        "hidden_size": int(validated["model"]["hidden_size"]),
        "num_layers": int(validated["model"]["num_layers"]),
        "dropout": float(validated["model"]["dropout"]),
        "learning_rate": float(validated["training"]["learning_rate"]),
        "weight_decay": float(validated["training"]["weight_decay"]),
        "lookback_steps": int(validated["data"]["lookback_steps"]),
        "feature_count": int(validated["data"]["feature_count"]),
        "train_sample_count": int(validated["data"]["train_sample_count"]),
        "validation_sample_count": int(validated["data"]["validation_sample_count"]),
        "validation_n_samples_evaluated": int(metric.n_samples),
        "validation_population_fingerprint_match": metric.population_fingerprint == validated["lineage"]["population_fingerprint"],
        "validation_rmse_wh": float(metric.rmse_wh),
        "validation_mae_wh": float(metric.mae_wh),
        "validation_r2": float(metric.r2),
        "checkpoint_paths": {k: str(v) for k, v in paths.items()},
        "registry_status": completed["status"],
        "metrics_persisted": len(completed["metrics"]),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    tmpdir = Path(tempfile.mkdtemp(prefix="phase43-harness-"))
    try:
        out = _run_one_candidate_in_sandbox(project_root, tmpdir, "LT1", "LH32")
        print(json.dumps(out, indent=2, default=str))
        # Critical: canonical registry must remain untouched
        canon = json.loads(open(project_root / "artifacts/experiments/experiment_registry.jsonl").read().strip().splitlines()[-1])
        print("Canonical registry last record run_id:", canon["run_id"])
        print("Sandbox run_id:", out["run_id"])
        assert canon["run_id"] != out["run_id"], "canonical registry must NOT contain sandbox runs"
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
