import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from course_work.data.scaling import inverse_transform_target
from course_work.evaluation.metrics import (
    EvaluationMode,
    compute_regression_metrics,
)
from course_work.experiments.registry import ArtifactType, ExperimentRegistry
from course_work.models.lstm_regressor import LSTMRegressor
from course_work.models.transformer_regressor import TransformerRegressor
from course_work.training.losses import (
    build_training_criterion,
    criterion_config,
    validate_criterion_inputs,
)
from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes, csv_text


HISTORY_COLUMNS = [
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


@dataclass(frozen=True)
class TrainingResult:
    history: pd.DataFrame
    best_epoch: int
    best_validation_rmse_wh: float
    metric_result: Any
    trainable_parameters: int
    total_epochs_run: int
    stopped_reason: str
    best_sample_idx: np.ndarray
    best_y_true_wh: np.ndarray
    best_y_pred_wh: np.ndarray
    gradient_diagnostics: dict[str, Any]


def build_model_from_run_config(config: dict[str, Any]) -> nn.Module:
    model_config = config["model"]
    family = model_config["model_family"]
    if family == "LSTM":
        return LSTMRegressor(model_config)
    if family == "TRANSFORMER_ENCODER":
        return TransformerRegressor(model_config)
    raise ValueError(f"Unsupported model family for training: {family}")


def _inverse_predictions_to_wh(
    predictions: np.ndarray,
    target_option: str,
    target_scaler_bundle: dict[str, Any] | None,
) -> np.ndarray:
    flat = predictions.reshape(-1, 1)
    if target_option == "YS0":
        return flat.reshape(-1)
    if target_scaler_bundle is None:
        raise ValueError("YS1 evaluation requires target scaler bundle")
    return inverse_transform_target(flat, "YS1", target_scaler_bundle).reshape(-1)


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "MIN") -> None:
        if patience <= 0:
            raise ValueError("patience must be positive")
        if mode != "MIN":
            raise ValueError("Only MIN mode is supported")
        self.patience = patience
        self.mode = mode
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.wait_count = 0
        self.stopped_early = False

    def update(self, epoch: int, metric_value: float) -> bool:
        if not math.isfinite(metric_value):
            raise ValueError("metric_value must be finite")
        improved = self.best_value is None or metric_value < self.best_value
        if improved:
            self.best_value = metric_value
            self.best_epoch = epoch
            self.wait_count = 0
            return True
        self.wait_count += 1
        if self.wait_count >= self.patience:
            self.stopped_early = True
        return False

    def should_stop(self) -> bool:
        return self.stopped_early


class TrainingEngine:
    def __init__(self, registry: ExperimentRegistry, clock: Callable[[], str] | None = None) -> None:
        self.registry = registry
        self.clock = clock or registry.clock

    def _evaluate_loader(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        target_option: str,
        target_scaler_bundle: dict[str, Any] | None,
        run_id: str,
        model_id: str,
        split_id: str,
        population_fingerprint: str,
        lookback_steps: int,
        horizon_steps: int,
        boundary_protocol: str = "WB0_CONTEXT_CARRY_OVER",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        model.eval()
        sample_indices: list[int] = []
        y_true_wh: list[float] = []
        y_pred_wh: list[float] = []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                y_model = batch["y_model"].to(device)
                y_raw = batch["y_raw_wh"].to(device)
                sample_idx = batch["sample_idx"].cpu().numpy()
                predictions = model(x)
                if predictions.shape != y_model.shape:
                    raise RuntimeError("Prediction and target shape mismatch")
                pred_wh = _inverse_predictions_to_wh(predictions.cpu().numpy(), target_option, target_scaler_bundle)
                true_wh = y_raw.cpu().numpy().reshape(-1)
                sample_indices.extend(sample_idx.tolist())
                y_true_wh.extend(true_wh.tolist())
                y_pred_wh.extend(pred_wh.tolist())
        evaluation_mode = EvaluationMode.TRAIN_DIAGNOSTIC.value if split_id == "TRAIN" else EvaluationMode.VALIDATION.value
        metric_result = compute_regression_metrics(
            np.asarray(y_true_wh),
            np.asarray(y_pred_wh),
            np.asarray(sample_indices, dtype=np.int64),
            split_id,
            evaluation_mode,
            population_fingerprint,
            run_id,
            model_id,
            lookback_steps=lookback_steps,
            horizon_steps=horizon_steps,
            target_scaling_option=target_option,
            boundary_protocol=boundary_protocol,
        )
        return (
            np.asarray(sample_indices, dtype=np.int64),
            np.asarray(y_true_wh, dtype=np.float64),
            np.asarray(y_pred_wh, dtype=np.float64),
            metric_result,
        )

    def train(
        self,
        run_id: str,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        target_scaler_bundle: dict[str, Any] | None,
        population_fingerprint: str,
        boundary_protocol: str = "WB0_CONTEXT_CARRY_OVER",
        evaluate_validation: bool = True,
    ) -> TrainingResult:
        record = self.registry.get_run(run_id)
        config = record["config"]
        training = config["training"]
        data = config["data"]
        model_id = config["model"].get("model_name", config["model"]["model_family"])
        target_option = data["target_scaling_option"]
        max_epochs = int(training["max_epochs"])
        learning_rate = float(training["learning_rate"])
        weight_decay = float(training["weight_decay"])
        clip_enabled = bool(training["gradient_clipping_enabled"])
        clip_norm = float(training["gradient_clip_max_norm"]) if training["gradient_clip_max_norm"] is not None else 0.0
        patience = int(training["early_stopping_patience"])
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = build_training_criterion(training)
        early_stop = EarlyStopping(patience=patience, mode="MIN")
        history_rows: list[dict[str, Any]] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_metric_result = None
        best_sample_idx = None
        best_y_true = None
        best_y_pred = None
        stopped_reason = "MAX_EPOCHS"
        train_start = time.time()
        preclip_gradient_norms: list[float] = []
        clipped_batches = 0
        total_gradient_batches = 0
        nonfinite_gradient_events = 0

        # Heartbeat path: external watchdog can poll this to detect hangs
        heartbeat_path = Path(os.environ.get("SWEEP_HEARTBEAT_PATH", "/tmp/sweep_heartbeat.txt"))

        def _write_heartbeat(epoch: int, stage: str) -> None:
            try:
                heartbeat_path.write_text(
                    f"epoch={epoch}\nstage={stage}\ntimestamp={time.time()}\nrun_id={run_id}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass  # Heartbeat is best-effort

        _write_heartbeat(0, "training_started")

        print(f"[TRAIN] Starting {max_epochs} epochs (run_id={run_id})", flush=True)
        print(f"[TRAIN] Heartbeat file: {heartbeat_path}", flush=True)

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            sample_count = 0
            batch_count = 0
            total_batches = len(train_loader)
            epoch_start = time.time()

            _write_heartbeat(epoch, "epoch_started")

            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y_model"].to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(x)
                validate_criterion_inputs(predictions, y)
                loss = loss_fn(predictions, y)
                loss.backward()

                # Phase 39: Non-finite gradient guard
                # Check BEFORE any clipping for both GC0 and GC1
                has_nonfinite = False
                for p in model.parameters():
                    if p.grad is not None:
                        if not p.grad.isfinite().all():
                            has_nonfinite = True
                            break
                if has_nonfinite:
                    nonfinite_gradient_events += 1
                    raise ValueError(
                        f"Non-finite gradient detected at epoch {epoch}, batch {batch_count}. "
                        "Training terminated before optimizer step."
                    )

                # Compute pre-clip gradient norm (L2, non-mutating)
                # This is done BEFORE any clipping for both GC0 and GC1
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)  # L2 norm per parameter
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                norm_value = float(total_norm)
                preclip_gradient_norms.append(norm_value)
                total_gradient_batches += 1

                if clip_enabled:
                    # GC1: clip gradients if norm exceeds threshold
                    clipped_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        clip_norm,
                        error_if_nonfinite=False,  # Already checked above
                    )
                    if norm_value > clip_norm:
                        clipped_batches += 1

                optimizer.step()
                batch_size = x.shape[0]
                epoch_loss += float(loss.item()) * batch_size
                sample_count += batch_size
                batch_count += 1

                # Progress logging every 50 batches
                if batch_count % 50 == 0 or batch_count == total_batches:
                    print(
                        f"  [TRAIN] epoch {epoch}/{max_epochs} - batch {batch_count}/{total_batches} - loss={loss.item():.4f}",
                        flush=True,
                    )

            train_loss = epoch_loss / max(sample_count, 1)
            print(f"  [TRAIN] epoch {epoch} - train_loss={train_loss:.4f} - evaluating...", flush=True)

            _write_heartbeat(epoch, "eval_train_started")
            _, train_true, train_pred, train_metric = self._evaluate_loader(
                model,
                train_loader,
                device,
                target_option,
                target_scaler_bundle,
                run_id,
                model_id,
                "TRAIN",
                population_fingerprint,
                data["lookback_steps"],
                data["horizon_steps"],
                boundary_protocol,
            )
            if evaluate_validation:
                _write_heartbeat(epoch, "eval_val_started")
                sample_idx, y_true, y_pred, val_metric = self._evaluate_loader(
                    model,
                    validation_loader,
                    device,
                    target_option,
                    target_scaler_bundle,
                    run_id,
                    model_id,
                    "VALIDATION",
                    population_fingerprint,
                    data["lookback_steps"],
                    data["horizon_steps"],
                    boundary_protocol,
                )
            else:
                sample_idx = np.array([], dtype=np.int64)
                y_true = np.array([], dtype=np.float64)
                y_pred = np.array([], dtype=np.float64)
                val_metric = train_metric
            _write_heartbeat(epoch, "epoch_completed")

            improved = (
                early_stop.update(epoch, val_metric.rmse_wh)
                if evaluate_validation
                else epoch == max_epochs
            )
            if improved:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_metric_result = val_metric
                best_sample_idx = sample_idx
                best_y_true = y_true
                best_y_pred = y_pred

            epoch_seconds = time.time() - epoch_start
            elapsed_total = time.time() - train_start

            print(
                f"  [TRAIN] epoch {epoch}/{max_epochs} - "
                f"train_loss={train_loss:.4f} - "
                f"train_rmse={train_metric.rmse_wh:.4f} - "
                f"val_rmse={val_metric.rmse_wh:.4f} - "
                f"val_mae={val_metric.mae_wh:.4f} - "
                f"val_r2={val_metric.r2:.4f} - "
                f"{'BEST' if improved else ''} - "
                f"{epoch_seconds:.1f}s - "
                f"total={elapsed_total:.1f}s",
                flush=True,
            )

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_rmse_wh": train_metric.rmse_wh,
                    "validation_rmse_wh": val_metric.rmse_wh,
                    "validation_mae_wh": val_metric.mae_wh,
                    "validation_r2": val_metric.r2,
                    "learning_rate": learning_rate,
                    "epoch_seconds": epoch_seconds,
                    "is_best": improved,
                }
            )
            if evaluate_validation and early_stop.should_stop():
                stopped_reason = "EARLY_STOPPING"
                break
        if best_state is None or best_metric_result is None:
            raise RuntimeError("Training did not produce a best checkpoint")
        model.load_state_dict(best_state)
        history = pd.DataFrame(history_rows, columns=HISTORY_COLUMNS)
        return TrainingResult(
            history=history,
            best_epoch=int(early_stop.best_epoch or 1),
            best_validation_rmse_wh=float(best_metric_result.rmse_wh),
            metric_result=best_metric_result,
            trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            total_epochs_run=len(history_rows),
            stopped_reason=stopped_reason,
            best_sample_idx=best_sample_idx if best_sample_idx is not None else np.array([], dtype=np.int64),
            best_y_true_wh=best_y_true if best_y_true is not None else np.array([], dtype=np.float64),
            best_y_pred_wh=best_y_pred if best_y_pred is not None else np.array([], dtype=np.float64),
            gradient_diagnostics={
                "mean_preclip_global_grad_norm": (
                    float(np.mean(preclip_gradient_norms)) if preclip_gradient_norms else None
                ),
                "max_preclip_global_grad_norm": (
                    float(np.max(preclip_gradient_norms)) if preclip_gradient_norms else None
                ),
                "clipped_batches": clipped_batches,
                "total_batches": total_gradient_batches,
                "clipping_fraction": (
                    clipped_batches / total_gradient_batches if total_gradient_batches else 0.0
                ),
                "nonfinite_grad_events": nonfinite_gradient_events,
                "clip_max_norm": clip_norm if clip_enabled else None,
                "clip_order": "ZERO_GRAD_FORWARD_CRITERION_BACKWARD_CLIP_OPTIMIZER_STEP",
            },
        )

    def persist_run_artifacts(
        self,
        run_id: str,
        run_directory: Path,
        model: nn.Module,
        result: TrainingResult,
        sample_idx: np.ndarray,
        y_true_wh: np.ndarray,
        y_pred_wh: np.ndarray,
    ) -> dict[str, Path]:
        run_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = run_directory / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_dir / "best_checkpoint.pt"
        last_path = checkpoint_dir / "last_checkpoint.pt"
        payload = {
            "model_state_dict": model.state_dict(),
            "checkpoint_metadata": model.checkpoint_metadata() if hasattr(model, "checkpoint_metadata") else {},
            "best_epoch": result.best_epoch,
            "best_validation_rmse_wh": result.best_validation_rmse_wh,
            "criterion_config": criterion_config(self.registry.get_run(run_id)["config"]["training"]),
        }
        torch.save(payload, best_path)
        torch.save(payload, last_path)
        history_path = run_directory / "training_history.csv"
        atomic_write_bytes(history_path, csv_text(HISTORY_COLUMNS, result.history.to_dict(orient="records")).encode("utf-8"))
        log_path = run_directory / "training.log"
        log_path.write_text(f"best_epoch={result.best_epoch}\nstopped_reason={result.stopped_reason}\n", encoding="utf-8")
        metrics_dir = run_directory / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "best_validation_metrics.json"
        atomic_write_bytes(
            metrics_path,
            canonical_json_bytes(
                {
                    "metric_result": asdict(result.metric_result),
                    "criterion_config": criterion_config(
                        self.registry.get_run(run_id)["config"]["training"]
                    ),
                    "gradient_diagnostics": result.gradient_diagnostics,
                }
            ),
        )
        predictions_dir = run_directory / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / "best_validation_predictions.csv"
        prediction_rows = [
            {
                "run_id": run_id,
                "sample_idx": int(sample_idx[index]),
                "y_true_wh": float(y_true_wh[index]),
                "y_pred_wh": float(y_pred_wh[index]),
                "residual_wh": float(y_true_wh[index] - y_pred_wh[index]),
            }
            for index in range(len(sample_idx))
        ]
        atomic_write_bytes(
            predictions_path,
            csv_text(["run_id", "sample_idx", "y_true_wh", "y_pred_wh", "residual_wh"], prediction_rows).encode("utf-8"),
        )
        paths = {
            "history": history_path,
            "log": log_path,
            "best_checkpoint": best_path,
            "last_checkpoint": last_path,
            "metrics": metrics_path,
            "predictions": predictions_path,
        }
        for artifact_type, path in (
            (ArtifactType.TRAIN_LOG.value, log_path),
            (ArtifactType.BEST_CHECKPOINT.value, best_path),
            (ArtifactType.METRICS.value, metrics_path),
            (ArtifactType.PREDICTIONS.value, predictions_path),
        ):
            self.registry.register_artifact(run_id, artifact_type, path, required=artifact_type != ArtifactType.PREDICTIONS.value)
        metric = result.metric_result
        for metric_name, metric_value in (
            ("mae_wh", metric.mae_wh),
            ("rmse_wh", metric.rmse_wh),
            ("r2", metric.r2),
        ):
            self.registry.register_metric(
                run_id,
                "VALIDATION",
                metric_name,
                metric_value,
                metric_unit_for(metric_name),
                metric.n_samples,
                metric.population_fingerprint,
                f"epoch_{result.best_epoch}",
                status=metric.status,
            )
        return paths


def metric_unit_for(metric_name: str) -> str:
    if metric_name == "r2":
        return "dimensionless"
    return "Wh"


class TrainingTimeoutError(Exception):
    """Raised when training exceeds the configured wall-clock timeout."""


def _timeout_handler(signum, frame):
    raise TrainingTimeoutError("Training exceeded wall-clock timeout")


def install_training_timeout(seconds: int) -> None:
    """Install a SIGALRM handler that raises TrainingTimeoutError after `seconds`.

    Use `disable_training_timeout()` to cancel. Only works on Unix/macOS main thread.
    """
    if seconds <= 0:
        return
    seconds = int(seconds)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)


def disable_training_timeout() -> None:
    signal.alarm(0)
