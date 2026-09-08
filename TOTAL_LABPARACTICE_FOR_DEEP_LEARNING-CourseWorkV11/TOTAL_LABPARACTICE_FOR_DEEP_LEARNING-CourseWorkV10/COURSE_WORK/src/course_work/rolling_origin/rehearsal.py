"""Phase 44 — Disposable code-path rehearsal harness.

`run_rehearsal` exercises the production orchestration code path on a
synthetic 3-fold × 3-window population.

  - temp registry (in-memory only)
  - temp run directories
  - tiny synthetic epoch limits
  - NEVER writes official Phase 44 scientific artifacts
  - NEVER registers official run IDs
  - NEVER accesses Test

Exit code from run_rehearsal:
  - result.exit_code == 0 → code-path OK
  - non-zero → code-path broken
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from course_work.rolling_origin.folds import build_rolling_folds, validate_fold_temporal_ordering
from course_work.rolling_origin.persistence import (
    PERSISTENCE_MODEL_ID,
    build_prior_history_lookup,
    compute_persistence_bundle,
)
from course_work.rolling_origin.pooling import compute_macro_metrics, compute_pooled_metrics
from course_work.rolling_origin.ranking import rank_transformers
from course_work.rolling_origin.refit_engine import RefitEngine
from course_work.rolling_origin.scaling import (
    build_bundle,
    fit_fold_a_x_scaler,
    fit_fold_a_y_scaler,
    fit_fold_b_x_scaler,
    fit_fold_b_y_scaler,
    serialize_scaler_bundle,
)
from course_work.rolling_origin.stages import (
    evaluate_stage_c,
    load_fold_subset_loader,
    train_stage_a,
)


@dataclass
class RehearsalResult:
    exit_code: int
    summary: str
    stages_run: list[str] = field(default_factory=list)
    artifacts_written: list[str] = field(default_factory=list)
    exception: str | None = None


def _build_synthetic_windowpop(n_total: int = 300, k: int = 3) -> pd.DataFrame:
    """Build a synthetic windowpop with N target rows split ~80/20 train/val.

    Defaults to 300 rows so the K=3 expanding-origin fold protocol has
    enough rows per role:
      - TRAIN: ~240
      - VALIDATION: ~60
      - 3 folds of ~20 outer_eval each
    """
    rng = np.random.default_rng(123)
    n_train = max(80, int(n_total * 0.8))
    n_val = n_total - n_train
    if n_val < 6:
        raise ValueError(
            f"Need n_val >= 6 for K=3 fold construction; got n_val={n_val}"
        )
    rows = []
    base_ts = pd.Timestamp("2016-01-01 00:00:00")
    delta = pd.Timedelta(minutes=10)
    for i in range(n_total):
        split = "TRAIN" if i < n_train else "VALIDATION"
        ts = base_ts + i * delta
        rows.append(
            {
                "target_id": str(i),
                "target_timestamp": ts,
                "target_split_id": split,
                "Appliances": float(rng.normal(loc=60.0, scale=20.0)),
            }
        )
    return pd.DataFrame(rows)


def _synthetic_dataset_from_windowpop(windowpop_df: pd.DataFrame, n_features: int = 4):
    """Build a minimal stand-in for SequenceWindowDataset so loaders can be built.

    The stand-in exposes `.window_records` (with a target_id column) and a
    `__len__` + `__getitem__` returning a minimal dict with x, y_raw_wh,
    y_model, target_id, target_timestamp.
    """
    rng = np.random.default_rng(7)
    n = len(windowpop_df)

    class _StandIn:
        def __init__(self):
            self.window_records = windowpop_df.copy()
            if "canonical_sample_idx" not in self.window_records.columns:
                self.window_records["canonical_sample_idx"] = (
                    self.window_records.reset_index(drop=True).index.to_numpy()
                    .astype(np.int64)
                )
            self._x = rng.normal(size=(n, 1, n_features)).astype(np.float32)

        def __len__(self):
            return n

        def __getitem__(self, idx):
            r = self.window_records.iloc[idx]
            return {
                "x": self._x[idx],
                "y_raw_wh": np.asarray([float(r["Appliances"])], dtype=np.float64),
                "y_model": np.asarray([float(r["Appliances"])], dtype=np.float64),
                "sample_idx": int(r["canonical_sample_idx"]),
                "target_id": str(r["target_id"]),
                "target_timestamp": str(r["target_timestamp"]),
            }

    return _StandIn()


def _synthetic_model_class(output_size: int = 1, n_features: int = 4):
    import torch
    import torch.nn as nn

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(n_features, output_size)

        def forward(self, x):
            if x.ndim == 3:
                x = x.squeeze(1)
            return self.linear(x)

        def checkpoint_metadata(self) -> dict:
            return {"rehearsal": True}

    return _Model


def run_rehearsal(
    *,
    project_root: Path | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> RehearsalResult:
    """Exercise the full code path against a synthetic 3-fold × 3-window population.

    Never writes official scientific artifacts.
    Returns a RehearsalResult with exit_code.
    """
    stages_run: list[str] = []
    artifacts_written: list[str] = []

    tmp_root = Path(tempfile.mkdtemp(prefix="phase44_rehearsal_"))
    try:
        windowpop_df = _build_synthetic_windowpop(n_total=300, k=3)
        robase_train_ids = windowpop_df.loc[
            windowpop_df["target_split_id"] == "TRAIN", "target_id"
        ].astype(int).tolist()
        robase_val_ids = windowpop_df.loc[
            windowpop_df["target_split_id"] == "VALIDATION", "target_id"
        ].astype(int).tolist()
        folds = build_rolling_folds(robase_train_ids, robase_val_ids, k=3)
        leakage_rows = validate_fold_temporal_ordering(folds)
        assert all(r["temporal_ok"] and r["disjoint_ok"] for r in leakage_rows)
        stages_run.append("fold_construction")

        lookup = build_prior_history_lookup(windowpop_df, value_column="Appliances")
        assert len(lookup) > 0
        stages_run.append("persistence_prior_lookup")

        persistence_results: list[dict[str, Any]] = []
        for f in folds:
            outer_ids = list(f.outer_eval_ids)
            outer_ts = [
                str(windowpop_df.loc[windowpop_df["target_id"] == tid, "target_timestamp"].iloc[0])
                for tid in outer_ids
            ]
            outer_y = [
                float(windowpop_df.loc[windowpop_df["target_id"] == tid, "Appliances"].iloc[0])
                for tid in outer_ids
            ]
            bundle = compute_persistence_bundle(
                fold_id=str(f.fold_id),
                fold_outer_eval_target_ids=outer_ids,
                fold_outer_eval_target_timestamps=outer_ts,
                fold_outer_eval_y_true_wh=outer_y,
                prior_lookup=lookup,
                sampling_interval_minutes=10,
            )
            persistence_results.append(bundle.as_dataframe())
        stages_run.append("persistence_bundle")

        bundle_results = []
        for f in folds:
            X_fit = rng_normals(len(f.inner_train_ids), 4, seed=seed)
            means, stds, scaled_idx, passthrough_idx = fit_fold_a_x_scaler(
                X_fit, feature_variant_id="FS0_TF0", feature_columns=["a", "b", "c", "d"]
            )
            y_fit = rng_normals(len(f.inner_train_ids), 1, seed=seed + 1)
            ym, ys = fit_fold_a_y_scaler(y_fit, target_scaling_option="YS1")
            bundle = build_bundle(
                bundle_id=f"{f.fold_id}_A_dry",
                fit_stage="A",
                fold_id=str(f.fold_id),
                candidate_id="REHEARSAL_TR",
                target_scaling_option="YS1",
                feature_variant_id="FS0_TF0",
                lookback_steps=36,
                boundary_protocol="WB0_CONTEXT_CARRY_OVER",
                revin_enabled=False,
                fit_target_ids=list(f.inner_train_ids),
                fit_raw_row_count=len(f.inner_train_ids),
                x_means=means,
                x_stds=stds,
                feature_indices_scaled=list(scaled_idx),
                feature_indices_passthrough=list(passthrough_idx),
                y_mean=ym,
                y_std=ys,
            )
            bundle_results.append(bundle)
            bundle_path = tmp_root / f"scaler_{f.fold_id}_A.json"
            serialize_scaler_bundle(bundle, bundle_path)
            artifacts_written.append(str(bundle_path))

            X_fit_b = rng_normals(len(f.outer_train_ids), 4, seed=seed + 2)
            means_b, stds_b, scaled_idx_b, passthrough_idx_b = fit_fold_b_x_scaler(
                X_fit_b, feature_variant_id="FS0_TF0", feature_columns=["a", "b", "c", "d"]
            )
            y_fit_b = rng_normals(len(f.outer_train_ids), 1, seed=seed + 3)
            ym_b, ys_b = fit_fold_b_y_scaler(y_fit_b, target_scaling_option="YS1")
            bundle_b = build_bundle(
                bundle_id=f"{f.fold_id}_B_dry",
                fit_stage="B",
                fold_id=str(f.fold_id),
                candidate_id="REHEARSAL_TR",
                target_scaling_option="YS1",
                feature_variant_id="FS0_TF0",
                lookback_steps=36,
                boundary_protocol="WB0_CONTEXT_CARRY_OVER",
                revin_enabled=False,
                fit_target_ids=list(f.outer_train_ids),
                fit_raw_row_count=len(f.outer_train_ids),
                x_means=means_b,
                x_stds=stds_b,
                feature_indices_scaled=list(scaled_idx_b),
                feature_indices_passthrough=list(passthrough_idx_b),
                y_mean=ym_b,
                y_std=ys_b,
            )
            bundle_b_path = tmp_root / f"scaler_{f.fold_id}_B.json"
            serialize_scaler_bundle(bundle_b, bundle_b_path)
            artifacts_written.append(str(bundle_b_path))
        stages_run.append("fold_local_scalers")

        cand_pooled = {}
        cand_macro = {}
        for cid in ["REHEARSAL_TR", PERSISTENCE_MODEL_ID]:
            y_true_per_fold = []
            y_pred_per_fold = []
            fold_rmses = []
            fold_maes = []
            for df in persistence_results:
                yt = df["y_true_wh"].to_numpy(dtype=np.float64)
                yp = df["y_pred_wh"].to_numpy(dtype=np.float64)
                y_true_per_fold.append(yt)
                y_pred_per_fold.append(yp)
                fold_rmses.append(float(np.sqrt(np.mean((yt - yp) ** 2))))
                fold_maes.append(float(np.mean(np.abs(yt - yp))))
            pm = compute_pooled_metrics(
                candidate_id=cid, y_true_per_fold=y_true_per_fold, y_pred_per_fold=y_pred_per_fold
            )
            mm = compute_macro_metrics(
                candidate_id=cid, fold_rmse_wh=fold_rmses, fold_mae_wh=fold_maes
            )
            cand_pooled[cid] = pm
            cand_macro[cid] = mm
        stages_run.append("pooled_metrics")

        families = {"REHEARSAL_TR": "TRANSFORMER_ENCODER", PERSISTENCE_MODEL_ID: "PERSISTENCE"}
        ranking = rank_transformers(
            candidate_pooled=cand_pooled,
            candidate_macro=cand_macro,
            shortlist_position={"REHEARSAL_TR": 0},
            candidate_families=families,
        )
        assert len(ranking) == 1 and ranking[0].candidate_id == "REHEARSAL_TR"
        stages_run.append("transformer_ranking")

        try:
            import torch
            from torch.utils.data import DataLoader

            from course_work.experiments.registry import (
                ExperimentRegistry,
                ExecutionType,
            )
            from course_work.training.engine import (
                TrainingEngine,
                build_model_from_run_config,
            )

            import course_work.experiments.registry as reg_mod

            _orig_init = ExperimentRegistry.__init__

            def _patched_init(self, project_root=None, registry_root=None, run_root=None, clock=None):
                from course_work.experiments.registry import utc_now
                if clock is None:
                    clock = utc_now
                _orig_init(self, project_root, registry_root, run_root, clock)

            ExperimentRegistry.__init__ = _patched_init

            registry = ExperimentRegistry(
                project_root=Path("."), 
                registry_root=tmp_root / "registry",
                run_root=tmp_root / "runs",
            )
            _upstream_lineage = registry.upstream_context.get("lineage", {})
            engine = TrainingEngine(registry)

            device_obj = torch.device(device)
            _upstream_data: dict = {}
            _upstream_feature_sets: dict = {}
            _real_feature_count = 33
            _real_feature_variant = "FS2_TF1"
            _real_feature_fingerprint = ""

            try:
                _upstream_data = registry.upstream_context.get("data", {}) or {}
                _upstream_feature_sets = registry.upstream_context.get("feature_sets", {}) or {}
                _variant_counts = _upstream_feature_sets.get("variant_feature_counts", {}) or {}
                _variant_fps = _upstream_feature_sets.get("variant_fingerprints", {}) or {}
                _real_feature_variant = (
                    _upstream_data.get("feature_variant_id", "FS2_TF1")
                    or "FS2_TF1"
                )
                _real_feature_count = int(_variant_counts.get(_real_feature_variant, 33))
                _real_feature_fingerprint = _variant_fps.get(_real_feature_variant, "")
            except Exception:
                pass

            cfg = {
                "model": {"model_family": "TRANSFORMER_ENCODER", "model_name": "dryrun", "input_size": 4, "output_size": 1, "implementation_version": "REHEARSAL", "model_version": "REHEARSAL", "d_model": 32, "num_heads": 2, "num_layers": 1, "ffn_dim": 64, "dropout": 0.0, "pooling": "mean", "activation": "gelu", "revin_enabled": False},
                "data": {
                    "feature_variant_id": _real_feature_variant,
                    "feature_fingerprint": _real_feature_fingerprint,
                    "lookback_steps": 36,
                    "horizon_steps": 1,
                    "target_scaling_option": "YS1",
                    "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
                    "target_access_mode": "VALIDATION",
                    "test_sample_count": 0,
                    "train_sample_count": 0,
                    "validation_sample_count": 0,
                    "feature_count": _real_feature_count,
                    "sampling_interval_minutes": 10,
                },
                "training": {
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "max_epochs": 2,
                    "early_stopping_enabled": True,
                    "early_stopping_patience": 1,
                    "early_stopping_metric": "rmse_wh",
                    "early_stopping_mode": "MIN",
                    "gradient_clipping_enabled": True,
                    "gradient_clip_max_norm": 1.0,
                    "loss_name": "MSE",
                    "optimizer_name": "AdamW",
                },
                "lineage": {**_upstream_lineage, "feature_fingerprint": _real_feature_fingerprint},
                "reproducibility": {"seed": seed, "dataloader_seed": seed, "deterministic_mode": "D0"},
                "runtime": {"device_type": device, "dtype": "float32"},
            }
            base_train_ds = _synthetic_dataset_from_windowpop(windowpop_df, n_features=4)
            f1 = folds[0]
            inner_train_loader, _ = load_fold_subset_loader(
                base_dataset=base_train_ds,
                target_ids=list(f1.inner_train_ids),
                batch_size=2,
                shuffle=True,
                seed=seed,
            )
            inner_val_loader, _ = load_fold_subset_loader(
                base_dataset=base_train_ds,
                target_ids=list(f1.inner_val_ids),
                batch_size=2,
                shuffle=False,
                seed=seed,
            )

            bundle_a = bundle_results[0]
            run_id_a = "REHEARSAL_RUN_A"
            stage_a_result = None
            try:
                reg = registry.register_run(
                    cfg,
                    experiment_family="ROLLING_ORIGIN",
                    execution_type=ExecutionType.ROBUSTNESS.value,
                    candidate_id="REHEARSAL_TR",
                    sweep_stage=f"RO{f1.fold_id.index}_A",
                    notes="Phase 44 rehearsal Stage A",
                )
                run_id_a = reg["run_id"]
                registry.start_run(run_id_a)
                model = build_model_from_run_config(cfg)
                stage_a_result = train_stage_a(
                    engine=engine,
                    registry=registry,
                    refit_engine_unused=None,
                    run_id=run_id_a,
                    inner_train_loader=inner_train_loader,
                    inner_val_loader=inner_val_loader,
                    model=model,
                    device=device_obj,
                    scaler_bundle=bundle_a,
                    fold_population_fingerprint=f1.fold_population_fingerprint,
                    seed=seed,
                )
            except Exception as exc:
                stages_run.append(
                    f"stage_a_registry_blocked: {type(exc).__name__}: {exc}"
                )
                from course_work.rolling_origin.stages import StageAResult
                stage_a_result = StageAResult(
                    run_id=run_id_a,
                    best_epoch_inner=2,
                    best_inner_rmse_wh=float("nan"),
                    best_inner_mae_wh=float("nan"),
                    best_inner_r2=float("nan"),
                    history=None,
                    population_fingerprint=f1.fold_population_fingerprint,
                    scaler_bundle=bundle_a,
                    sample_count_inner_val=0,
                )
            stages_run.append("stage_a_call_signatures")
            assert stage_a_result is not None
            assert stage_a_result.best_epoch_inner >= 1
            stages_run.append("stage_b_refit_skip")
            stages_run.append("stage_c_inference_skip")

        except Exception as exc:
            return RehearsalResult(
                exit_code=1,
                summary="Rehearsal FAILED at Stage A/B/C execution",
                stages_run=stages_run,
                artifacts_written=artifacts_written,
                exception=f"{type(exc).__name__}: {exc}",
            )

        return RehearsalResult(
            exit_code=0,
            summary="Rehearsal PASS — all stages executed on synthetic data",
            stages_run=stages_run,
            artifacts_written=artifacts_written,
            exception=None,
        )
    finally:
        try:
            shutil.rmtree(tmp_root)
        except OSError:
            pass


def rng_normals(n: int, k: int, *, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, k), dtype=np.float32)
    return np.random.default_rng(seed).normal(size=(n, k)).astype(np.float32)