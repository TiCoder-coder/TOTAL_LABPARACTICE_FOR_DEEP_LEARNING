"""Phase 44 — Stage A / Stage C wrappers.

Stage A wraps `TrainingEngine.train` with:
  - fold-specific inner_train / inner_val loaders
  - fold-local scaler bundle
  - candidate's frozen training config (no max_epochs/patience override)
  - early stopping with candidate patience / min_delta / monitor = inner RMSE
  - best_epoch_inner captured

Stage C wraps outer evaluation with:
  - fold-local scaler bundle (Stage B bundle)
  - model.eval()
  - torch.inference_mode()
  - per-target prediction bundle
  - same y_true across candidates (asserted)
  - no scaler fit, no backward
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from course_work.rolling_origin.scaling import FoldLocalScalerBundle
from course_work.utils.reproducibility import set_seed
from course_work.experiments.registry import RunStatus


@dataclass
class StageAResult:
    run_id: str
    best_epoch_inner: int
    best_inner_rmse_wh: float
    best_inner_mae_wh: float
    best_inner_r2: float
    history: pd.DataFrame
    population_fingerprint: str
    scaler_bundle: FoldLocalScalerBundle
    sample_count_inner_val: int


@dataclass
class StageCResult:
    model_id: str
    fold_id: str
    target_ids: tuple[int, ...]
    target_timestamps: tuple[str, ...]
    y_true_wh: np.ndarray
    y_pred_wh: np.ndarray
    residuals_wh: np.ndarray
    model_run_id: str
    refit_epoch: int

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fold_id": [self.fold_id] * len(self.target_ids),
                "target_id": list(self.target_ids),
                "target_timestamp": list(self.target_timestamps),
                "y_true_wh": self.y_true_wh.tolist(),
                "y_pred_wh": self.y_pred_wh.tolist(),
                "residual_wh": self.residuals_wh.tolist(),
                "model_run_id": [self.model_run_id] * len(self.target_ids),
                "refit_epoch": [self.refit_epoch] * len(self.target_ids),
            }
        )


def load_fold_subset_loader(
    *,
    base_dataset,
    target_ids: list[int],
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    drop_last: bool = False,
) -> tuple[DataLoader, list[int]]:
    """Build a DataLoader over `base_dataset` restricted to target_ids.

    Returns (loader, kept_indices_into_base_dataset).

    The base_dataset must expose `window_records` with a `target_id` column.
    """
    window_records = getattr(base_dataset, "window_records", None)
    if window_records is None or "target_id" not in window_records.columns:
        raise ValueError(
            "load_fold_subset_loader: base_dataset must have .window_records with target_id column"
        )

    def _norm_key(t):
        """Normalize target_id to int if possible, else string."""
        s = str(t)
        if s.startswith("TGT_") or s.startswith("tgt_"):
            try:
                return int(s.split("_", 1)[1])
            except (ValueError, IndexError):
                return s
        try:
            return int(s)
        except (ValueError, TypeError):
            return s

    target_id_to_pos = {
        _norm_key(tid): i
        for i, tid in enumerate(window_records["target_id"].tolist())
    }
    indices: list[int] = [
        target_id_to_pos[_norm_key(t)] for t in target_ids
        if _norm_key(t) in target_id_to_pos
    ]
    subset = Subset(base_dataset, indices)

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last,
        generator=g if shuffle else None,
    )
    return loader, indices


def load_fold_subset_loader_with_y_rescale(
    *,
    base_dataset,
    target_ids: list[int],
    batch_size: int,
    shuffle: bool,
    fold_stage_target_scaler,
    seed: int = 42,
    drop_last: bool = False,
) -> tuple[DataLoader, list[int]]:
    """Variant of `load_fold_subset_loader` that re-scales batch["y_model"]
    using a fold-local Y scaler.

    Phase 44 plan requires fold-local Y statistics; the canonical
    SequenceWindowDataset bakes the global Phase9 YS1 scaler into
    DatasetConfig. This loader rewrites y_model in each batch through the
    closed-form affine correction (raw Wh → fold-local Y scaled space).

    `y_raw_wh`, `x`, `sample_idx`, `target_id` are left UNTOUCHED.

    For YS0 (identity), this is exactly equivalent to `load_fold_subset_loader`.
    """
    from course_work.rolling_origin.utils import collate_batch_to_tensors
    import torch as _torch

    window_records = getattr(base_dataset, "window_records", None)
    if window_records is None or "target_id" not in window_records.columns:
        raise ValueError(
            "load_fold_subset_loader_with_y_rescale: base_dataset must have "
            ".window_records with target_id column"
        )

    def _norm_key(t):
        s = str(t)
        if s.startswith("TGT_") or s.startswith("tgt_"):
            try:
                return int(s.split("_", 1)[1])
            except (ValueError, IndexError):
                return s
        try:
            return int(s)
        except (ValueError, TypeError):
            return s

    target_id_to_pos = {
        _norm_key(tid): i
        for i, tid in enumerate(window_records["target_id"].tolist())
    }
    indices: list[int] = [
        target_id_to_pos[_norm_key(t)] for t in target_ids
        if _norm_key(t) in target_id_to_pos
    ]
    subset = Subset(base_dataset, indices)

    resolved_target_ids = [
        window_records["target_id"].iloc[i] for i in indices
    ]
    ts_col = "target_timestamp"
    resolved_target_ts = [
        window_records[ts_col].iloc[i] if ts_col in window_records.columns else ""
        for i in indices
    ]

    fold_y_mean = fold_stage_target_scaler.y_mean
    fold_y_std = fold_stage_target_scaler.y_std

    state = {"cursor": 0}

    def _collate(batch_list):
        batch = collate_batch_to_tensors(batch_list)
        start = state["cursor"]
        end = start + len(batch_list)
        batch["target_id"] = [str(t) for t in resolved_target_ids[start:end]]
        batch["target_timestamp"] = [str(t) for t in resolved_target_ts[start:end]]
        state["cursor"] = end
        if fold_y_mean is None or fold_y_std is None:
            return batch
        if "y_model" not in batch or "y_raw_wh" not in batch:
            return batch
        y_raw = batch["y_raw_wh"]
        if not _torch.is_tensor(y_raw):
            y_raw = _torch.as_tensor(y_raw, dtype=_torch.float32)
        y_raw = y_raw.float().reshape(-1)
        y_model_corrected = (y_raw - float(fold_y_mean)) / float(fold_y_std)
        batch["y_model"] = y_model_corrected.reshape(-1, 1).float()
        return batch

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last,
        generator=g if shuffle else None,
        collate_fn=_collate,
    )
    return loader, indices


def train_stage_a(
    *,
    engine,
    registry,
    refit_engine_unused,  
    run_id: str,
    inner_train_loader: DataLoader,
    inner_val_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    scaler_bundle: FoldLocalScalerBundle,
    fold_population_fingerprint: str,
    seed: int = 42,
    rehearsal_synthetic: bool = False,
    inner_train_expected_sample_idx=None,
    inner_val_expected_sample_idx=None,
    inner_train_population_fingerprint: str | None = None,
    inner_val_population_fingerprint: str | None = None,
) -> StageAResult:
    """Train Stage A: inner validation epoch selection with fold-local scalers.

    The model's training config (read from registry) is used at face value:
    max_epochs, patience, min_delta are the candidate's frozen training
    hyperparameters. NO fast-mode overrides.

    When `rehearsal_synthetic=True`, this returns a deterministic
    synthetic StageAResult WITHOUT calling engine.train(). This is used
    exclusively for disposable rehearsals; the orchestrator's
    assert_context_invariants guarantees it is forbidden in official mode.
    """
    if rehearsal_synthetic:
        set_seed(seed)

        import hashlib
        digest = hashlib.sha256(scaler_bundle.bundle_id.encode("utf-8")).hexdigest()
        best_epoch = (int(digest[:8], 16) % 30) + 1 
        history = pd.DataFrame({
            "epoch": list(range(1, best_epoch + 1)),
            "train_rmse_wh": [100.0 - i for i in range(best_epoch)],
            "validation_rmse_wh": [110.0 - i for i in range(best_epoch)],
        })
        try:
            registry._transition(run_id, RunStatus.RUNNING.value)
        except Exception:
            pass
        try:
            registry._transition(
                run_id, RunStatus.COMPLETED.value,
                update={"best_epoch": best_epoch,
                        "best_validation_rmse_wh": 80.0,
                        "synthetic": True},
            )
        except Exception:
            pass
        return StageAResult(
            run_id=run_id,
            best_epoch_inner=best_epoch,
            best_inner_rmse_wh=80.0,
            best_inner_mae_wh=60.0,
            best_inner_r2=0.5,
            history=history,
            population_fingerprint=fold_population_fingerprint,
            scaler_bundle=scaler_bundle,
            sample_count_inner_val=10,
        )

    set_seed(seed)
    scaler_dict = scaler_bundle.as_dict()

    from course_work.evaluation.metrics import MetricPopulationContext
    import numpy as np

    train_pop_fp = inner_train_population_fingerprint or fold_population_fingerprint
    val_pop_fp = inner_val_population_fingerprint or fold_population_fingerprint
    train_pop_ctx = None
    val_pop_ctx = None
    if inner_train_expected_sample_idx is not None:
        train_pop_ctx = MetricPopulationContext(
            split_id="TRAIN",
            expected_sample_idx=np.asarray(inner_train_expected_sample_idx, dtype=np.int64),
            population_fingerprint=train_pop_fp,
        )
    if inner_val_expected_sample_idx is not None:
        val_pop_ctx = MetricPopulationContext(
            split_id="VALIDATION",
            expected_sample_idx=np.asarray(inner_val_expected_sample_idx, dtype=np.int64),
            population_fingerprint=val_pop_fp,
        )

    result = engine.train(
        run_id=run_id,
        train_loader=inner_train_loader,
        validation_loader=inner_val_loader,
        model=model,
        device=device,
        target_scaler_bundle=scaler_dict,
        population_fingerprint=train_pop_fp,
        boundary_protocol=scaler_bundle.boundary_protocol,
        evaluate_validation=True,
        final_refit_mode=False,
        train_population_context=train_pop_ctx,
        validation_population_context=val_pop_ctx,
        validation_population_fingerprint=val_pop_fp,
    )

    best_epoch = int(result.best_epoch)
    if best_epoch < 1:
        raise RuntimeError(f"Stage A returned best_epoch={best_epoch} for {run_id}")

    sample_count_inner_val = int(len(result.best_sample_idx))

    inner_rmse = float(result.best_validation_rmse_wh)
    inner_mae = float(result.metric_result.mae_wh)
    inner_r2 = float(result.metric_result.r2)

    return StageAResult(
        run_id=run_id,
        best_epoch_inner=best_epoch,
        best_inner_rmse_wh=inner_rmse,
        best_inner_mae_wh=inner_mae,
        best_inner_r2=inner_r2,
        history=result.history,
        population_fingerprint=fold_population_fingerprint,
        scaler_bundle=scaler_bundle,
        sample_count_inner_val=sample_count_inner_val,
    )


@torch.inference_mode()
def evaluate_stage_c(
    *,
    model: torch.nn.Module,
    outer_eval_loader: DataLoader,
    scaler_bundle: FoldLocalScalerBundle,
    device: torch.device,
    model_id: str,
    fold_id: str,
    model_run_id: str,
    refit_epoch: int,
    rehearsal_synthetic: bool = False,
) -> StageCResult:
    """Evaluate Stage C on outer block, returning per-target predictions.

    When `rehearsal_synthetic=True`, the model is NOT invoked; instead a
    deterministic synthetic prediction set is returned that is consistent
    with the outer_eval_loader's iteration order. Used only for disposable
    rehearsals; blocked by assert_context_invariants in official mode.
    """
    model.eval()
    target_ids: list[int] = []
    target_timestamps: list[str] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    if rehearsal_synthetic:
        outer_count = 0
        for batch in outer_eval_loader:
            btid = batch.get("target_id", None)
            bts = batch.get("target_timestamp", None)
            x = batch["x"]
            ys = batch.get("y_raw_wh", batch.get("y", torch.zeros(x.shape[0])))
            if btid is not None:
                target_ids.extend([str(t) for t in btid])
            else:
                target_ids.extend([f"unknown_{i}" for i in range(x.shape[0])])
            if bts is not None:
                target_timestamps.extend([str(t) for t in bts])
            else:
                target_timestamps.extend([f"ts_{i}" for i in range(x.shape[0])])
            y_true.append(ys.numpy().flatten() if hasattr(ys, "numpy") else np.asarray(ys).flatten())
            outer_count += x.shape[0]
        if outer_count == 0:
            return StageCResult(
                model_id=model_id, fold_id=fold_id,
                target_ids=tuple(), target_timestamps=tuple(),
                y_true_wh=np.zeros(0), y_pred_wh=np.zeros(0),
                residuals_wh=np.zeros(0),
                model_run_id=model_run_id, refit_epoch=refit_epoch,
            )
        import hashlib
        digest = hashlib.sha256(f"{model_id}|{fold_id}|{refit_epoch}".encode("utf-8")).hexdigest()
        rng = np.random.default_rng(int(digest[:8], 16))
        y_true_arr = np.concatenate(y_true).astype(np.float64)
        noise = rng.normal(0.0, 5.0, size=y_true_arr.shape)
        y_pred_arr = y_true_arr + noise
        return StageCResult(
            model_id=model_id, fold_id=fold_id,
            target_ids=tuple(target_ids),
            target_timestamps=tuple(target_timestamps),
            y_true_wh=y_true_arr,
            y_pred_wh=y_pred_arr,
            residuals_wh=(y_true_arr - y_pred_arr),
            model_run_id=model_run_id, refit_epoch=refit_epoch,
        )

    for batch in outer_eval_loader:
        x = batch["x"].to(device)
        y_raw = batch["y_raw_wh"]
        batch_target_ids = batch.get("target_id", None)
        batch_target_ts = batch.get("target_timestamp", None)
        if batch_target_ids is None:
            batch_target_ids = [-1] * x.shape[0]
        if batch_target_ts is None:
            batch_target_ts = [""] * x.shape[0]
        out = model(x)
        pred_wh = scaler_bundle.inverse_transform_y(out.cpu().numpy())
        y_pred.append(pred_wh.flatten())
        y_true.append(y_raw.numpy().flatten() if hasattr(y_raw, "numpy") else np.asarray(y_raw).flatten())
        target_ids.extend([str(t) for t in batch_target_ids])
        target_timestamps.extend([str(t) for t in batch_target_ts])

    y_true_arr = np.concatenate(y_true) if y_true else np.zeros(0, dtype=np.float64)
    y_pred_arr = np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=np.float64)
    residuals = y_true_arr - y_pred_arr

    return StageCResult(
        model_id=model_id,
        fold_id=fold_id,
        target_ids=tuple(target_ids),
        target_timestamps=tuple(target_timestamps),
        y_true_wh=y_true_arr,
        y_pred_wh=y_pred_arr,
        residuals_wh=residuals,
        model_run_id=model_run_id,
        refit_epoch=int(refit_epoch),
    )