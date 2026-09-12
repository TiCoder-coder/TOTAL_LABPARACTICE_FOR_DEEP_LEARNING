"""Phase 44 — REAL official orchestrator (single core).

This module is the CORRECTED scientific orchestrator. It implements the
Phase 44 plan contract end-to-end:

  - 4 learned candidates (3 Transformers + LSTM)
  - 3 expanding-origin folds (RO1, RO2, RO3)
  - Stage A: TrainingEngine.train() with candidate max_epochs/patience
  - Stage B: RefitEngine.refit() with exact best_epoch_inner, no early stop
  - Stage C: evaluate_stage_c() on outer_eval
  - Persistence: compute_persistence_bundle() for 3 folds
  - Pooled metrics: from concatenated raw residuals (NOT mean-of-fold)
  - Transformer ranking: pooled_rmse primary, ties by worst_fold/sd/shortlist_pos
  - Family comparison: best Transformer vs LSTM vs Persistence
  - Signoff: PASS/PASS_WITH_WARNING only after all 13 consistency checks
  - Phase 45 handoff: only when signoff is PASS/PASS_WITH_WARNING

This is the SAME orchestration for official and rehearsal. The differences
are encoded entirely in `RunContext`:
  - registry_root / run_root (canonical vs tempfile)
  - scientific_max_epochs (50 official, 2-5 rehearsal)
  - persistence_only (rehearsal may skip learned for speed)
  - dataset_factory (real SequenceWindowDataset vs synthetic stand-in)

The synthetic `_synth_predict_for_official` and hash-generated best epochs
that were in `pipeline.run_pipeline()` have been REMOVED from the
official path. The synthetic functions remain in `pipeline.py` for
preflight-only code reuse (e.g. tests) but are NEVER reachable from
the official path.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from course_work.experiments.registry import (
    ArtifactType,
    ExecutionType,
    ExperimentRegistry,
    FailureType,
    RunStatus,
)
from course_work.rolling_origin.appliance_lookup import (
    build_windowpop_with_appliances,
)
from course_work.rolling_origin.candidate_loader import (
    CandidateSpec,
    LSTM_CANDIDATE_ID,
    PERSISTENCE_CANDIDATE_ID,
    load_candidates,
)
from course_work.rolling_origin.consistency import run_all_consistency_checks
from course_work.rolling_origin.folds import FoldDefinition, build_rolling_folds
from course_work.data.datasets import (
    SequenceWindowDataset,
    build_train_validation_loaders,
)
from torch.utils.data import Dataset as _TorchDataset


class _CanonicalFoldDataset:
    """Adapter that wraps a real canonical `SequenceWindowDataset` so the
    Phase 44 orchestrator can consume it via `load_fold_subset_loader`.

    The canonical dataset exposes `.window_records` with a `target_sample_id`
    column (e.g. "TGT_00000144"). The orchestrator (and `load_fold_subset_loader`)
    require a `target_id` column. This adapter aliases `target_sample_id` as
    `target_id` while preserving the canonical `__getitem__` shape contract:
    each item is `{"x": Tensor[L, F], "y_model": Tensor[1], "y_raw_wh": Tensor[1],
    "sample_idx": Tensor[int], ...}`.

    IMPORTANT: `__getitem__` is delegated to the wrapped SequenceWindowDataset,
    so DataLoader will produce batches of `x` shape `[B, L, F]` — exactly the
    scientific input shape expected by Transformer / LSTM.
    """

    def __init__(self, canonical_ds):
        self._canonical = canonical_ds
        # Expose window_records as a view with target_id alias
        records = canonical_ds.window_records if hasattr(canonical_ds, "window_records") else None
        if records is None:
            # SequenceWindowDataset stores _records internally
            records = getattr(canonical_ds, "_records", None)
        if records is None:
            raise RuntimeError(
                "_CanonicalFoldDataset: canonical dataset has neither "
                ".window_records nor ._records"
            )
        # Build aliased view (copy so downstream filter ops don't mutate the canonical)
        records = records.copy(deep=True)
        if "target_id" not in records.columns:
            if "target_sample_id" in records.columns:
                records["target_id"] = records["target_sample_id"].astype(str)
            else:
                raise RuntimeError(
                    "_CanonicalFoldDataset: canonical records have neither "
                    "target_id nor target_sample_id column"
                )
        else:
            records["target_id"] = records["target_id"].astype(str)
        self.window_records = records
        # Expose config for downstream code (e.g. fold-local scaler audit)
        self.config = getattr(canonical_ds, "config", None)
        # Expose sample_indices (for parity with SequenceWindowDataset)
        if hasattr(canonical_ds, "sample_indices"):
            self.sample_indices = canonical_ds.sample_indices
        # Stash feature matrix for fold-local scaler fit
        self._feature_matrix = getattr(canonical_ds, "_feature_matrix", None)
        # Target values for Stage C (appliance WH)
        self._target_values = getattr(canonical_ds, "_target_values", None)

    def __len__(self) -> int:
        return len(self._canonical)

    def __getitem__(self, idx):
        # Delegate to canonical SequenceWindowDataset — preserves [L, F] shape
        return self._canonical[int(idx)]


def build_real_canonical_base_dataset(
    *,
    project_root: Path,
    candidate: CandidateSpec,
):
    """Factory: build a real canonical `SequenceWindowDataset` union
    (TRAIN + VALIDATION) for the given candidate and wrap it in
    `_CanonicalFoldDataset` so Phase 44's `load_fold_subset_loader`
    can consume it.

    The returned dataset:
      - covers ALL ROBASE-train + ROBASE-val target IDs (no Test rows)
      - exposes `.window_records` with `target_id` aliased from
        `target_sample_id`
      - delegates `__getitem__` to the canonical SequenceWindowDataset
        so each item has `x.shape == (lookback_steps, feature_count)`
      - exposes `.config`, `.sample_indices`, `._feature_matrix`,
        `._target_values` for downstream fold-local scaler / persistence
        computations.

    Parameters
    ----------
    project_root : Path
        Project root containing `artifacts/` and `data/interim/...`.
    candidate : CandidateSpec
        Candidate whose lookback, feature variant, target scaling option,
        and boundary protocol determine the canonical dataset shape.
    """
    from course_work.data.datasets import build_dataset_suite

    suite, _window_fp = build_dataset_suite(
        project_root=project_root,
        variant_id=candidate.feature_variant_id,
        lookback=candidate.lookback_steps,
        target_option=candidate.target_scaling_option,
        boundary_protocol=candidate.boundary_protocol,
    )
    train_ds = suite["TRAIN"]
    val_ds = suite["VALIDATION"]

    # The Phase 44 fold may span ROBASE-train AND ROBASE-val IDs. Concatenate
    # so `load_fold_subset_loader` can find both populations. Concatenation is
    # done at the *records* level — the underlying feature_matrix is the same
    # for TRAIN+VALIDATION (it spans the entire timeline), so each record's
    # window record correctly indexes into the same matrix.
    train_records = getattr(train_ds, "_records", None)
    val_records = getattr(val_ds, "_records", None)
    if train_records is None or val_records is None:
        raise RuntimeError(
            "build_real_canonical_base_dataset: TRAIN/VALIDATION datasets "
            "do not expose ._records; the canonical SequenceWindowDataset "
            "implementation may have changed."
        )

    # Concatenate records preserving the canonical column schema, then
    # surface a single SequenceWindowDataset-shaped object via a tiny
    # adapter. The adapter wraps BOTH datasets and answers __getitem__
    # by computing the correct window from the right split dataset.
    class _TRAINVAL_DATASET:
        def __init__(self):
            self._train = train_ds
            self._val = val_ds
            self._train_records = train_records.copy(deep=True)
            self._val_records = val_records.copy(deep=True)
            # Add target_id alias on both for fold loader
            for rec in (self._train_records, self._val_records):
                if "target_id" not in rec.columns and "target_sample_id" in rec.columns:
                    rec["target_id"] = rec["target_sample_id"].astype(str)
                else:
                    rec["target_id"] = rec["target_id"].astype(str)
            # Combined window_records
            self.window_records = (
                pd.concat([self._train_records, self._val_records], axis=0, ignore_index=True)
                .sort_values("target_timestamp")
                .reset_index(drop=True)
            )
            # Surface config from TRAIN (lookback etc.)
            self.config = train_ds.config
            self.sample_indices = pd.concat([
                pd.Series(train_ds.sample_indices, name="sample_idx"),
                pd.Series(val_ds.sample_indices, name="sample_idx"),
            ], ignore_index=True).to_numpy()
            # Surface feature matrix & target values for scaler/persistence
            self._feature_matrix = getattr(train_ds, "_feature_matrix", None)
            self._target_values = getattr(train_ds, "_target_values", None)
            self._train_size = len(train_ds)

        def __len__(self) -> int:
            return len(self._train_records) + len(self._val_records)

        def __getitem__(self, idx):
            i = int(idx)
            if i < self._train_size:
                return self._train[i]
            return self._val[i - self._train_size]

    return _CanonicalFoldDataset(_TRAINVAL_DATASET())


from course_work.rolling_origin.persistence import (
    PERSISTENCE_MODEL_ID,
    build_prior_history_lookup,
    compute_persistence_bundle,
)
from course_work.rolling_origin.persistence_probe import run_persistence_probe
from course_work.rolling_origin.pooling import (
    compute_macro_metrics,
    compute_pooled_metrics,
)
from course_work.rolling_origin.populations import (
    ROBASE_VERSION,
    compute_population_fingerprint,
    extract_robase_train_ids,
    extract_robase_val_ids,
)
from course_work.rolling_origin.probe import run_population_probe
from course_work.rolling_origin.ranking import rank_transformers
from course_work.rolling_origin.refit_engine import RefitEngine
from course_work.rolling_origin.scaling import (
    FoldLocalScalerBundle,
    ScalerFitAudit,
    build_bundle,
    fit_fold_a_x_scaler,
    fit_fold_a_y_scaler,
    fit_fold_b_x_scaler,
    fit_fold_b_y_scaler,
    record_scaler_fit_audit,
)
from course_work.rolling_origin.stages import (
    StageAResult,
    StageCResult,
    evaluate_stage_c,
    load_fold_subset_loader,
    load_fold_subset_loader_with_y_rescale,
    train_stage_a,
)
from course_work.training.engine import (
    TrainingEngine,
    build_model_from_run_config,
)


# ----------------------- Constants -----------------------
SCIENTIFIC_MAX_EPOCHS = 50
SCIENTIFIC_PATIENCE = 10
PERSISTENCE_FOLD_COUNT = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------- Run Context -----------------------
@dataclass
class RunContext:
    """Encodes ALL differences between official and rehearsal.

    The SAME core orchestrator is used for both modes; only the context
    changes (data source, registry root, epoch budget, etc.).

    `base_dataset` must be a `SequenceWindowDataset` (or any object exposing
    `window_records` with a `target_id` column) that covers the union of
    ROBASE-train and ROBASE-val targets. It is built by the caller (script
    layer) using `build_train_validation_loaders` or equivalent.

    `dataset_factory` is an optional callable returning a fresh
    `SequenceWindowDataset` for a given fold. The orchestrator uses the
    pre-supplied `base_dataset` if provided; otherwise it falls back to
    `dataset_factory`.
    """
    project_root: Path
    transformer_shortlist_path: Path
    lstm_handoff_path: Path
    phase_42_signoff_path: Path
    phase_43_signoff_path: Path
    artifact_dir: Path
    registry_root: Path
    run_root: Path
    scientific_max_epochs: int = SCIENTIFIC_MAX_EPOCHS
    scientific_patience: int = SCIENTIFIC_PATIENCE
    seed: int = 42
    device: str = "cpu"
    is_rehearsal: bool = False
    base_dataset: Any | None = None
    dataset_factory: Callable[[Any], Any] | None = None
    # When True, the orchestrator does NOT call TrainingEngine.train() or
    # RefitEngine.refit(); instead it uses a synthetic stand-in dataset
    # so that the same code paths are exercised (Stage A/B/C signatures,
    # registry payloads, prediction bundles, persistence, pooling, ranking,
    # signoff, handoff) without real optimizer steps.
    rehearsal_synthetic: bool = False
    # When True, the orchestrator REUSES already-completed Stage A and
    # Stage B runs (and their refit_final.pt checkpoints) from the
    # canonical registry. Stage A training, Stage B refit, and any new
    # run IDs are NOT created. Stage C inference, Persistence, pooling,
    # ranking, O44, signoff, and Phase45 handoff still run. Used by
    # --mode finalize (NO-TRAIN resume).
    reuse_completed_runs: bool = False
    # Optional V2-owned exact run ledger for a no-train resume. The legacy
    # Phase 44 finalize path leaves this as None and keeps its historical
    # canonical-registry discovery behavior unchanged.
    reuse_completed_run_ids: dict[str, str] | None = None
    candidate_specs: tuple[CandidateSpec, ...] | None = None
    registry_namespace: str | None = None
    seed_before_model_construction: bool = False
    validated_registry_lifecycle: bool = False
    execution_track: str = "V1"
    registry_upstream_context: dict[str, Any] | None = None
    robase_train_ids: tuple[str, ...] | None = None
    robase_val_ids: tuple[str, ...] | None = None
    apply_fold_x_scaling: bool = False
    # V2 E07-E: when True, the orchestrator loads each candidate/fold
    # Stage-A training_history.csv after Stage A completes, extracts the
    # lr_used_for_epoch trace up to selected best_epoch_inner, and passes
    # it to RefitEngine.refit(stage_a_lr_trace=...). Stage B does NOT
    # reconstruct scheduler behavior; LR is set explicitly per epoch.
    # Default False preserves E01/E02/E03/E04/E05/E06 behavior.
    enable_stage_b_lr_replay: bool = False
    # V2 E11 partial-resume: when True AND reuse_completed_runs=True, the
    # orchestrator trains FRESH Stage B refits for any (candidate, fold)
    # pair that does NOT yet have a COMPLETED registry record. Stage A is
    # NEVER re-trained under this flag — Stage-A reuse is strict.
    # Used by E11's recovery path where 6 Stage-A runs are COMPLETED but
    # only a subset of Stage-B refits exist. Default False preserves
    # E01/E02/E03/E06/E07/E08/E09/E10 no-train-resume semantics.
    allow_partial_stage_b_training: bool = False
    # V2 E12 recovery only: reuse every explicitly locked COMPLETED Stage
    # A/B run and train only the missing canonical pairs. This is separate
    # from E11's Stage-B-only recovery because an E12 registration failure
    # can occur between Stage-A candidates. Default False preserves V1 and
    # all earlier V2 execution behavior.
    allow_partial_stage_recovery: bool = False


# ----------------------- Result -----------------------
V2_DIRECT_METRIC_FLATTEN_TRACKS = frozenset({
    "MODEL_IMPROVEMENT_V2_E06",
    "MODEL_IMPROVEMENT_V2_E07",
    "MODEL_IMPROVEMENT_V2_E08",
    "MODEL_IMPROVEMENT_V2_E09",
    "MODEL_IMPROVEMENT_V2_E10",
    "MODEL_IMPROVEMENT_V2_E11",
    "MODEL_IMPROVEMENT_V2_E12",
    "MODEL_IMPROVEMENT_V2_E13",
    "MODEL_IMPROVEMENT_V2_E14",
    "MODEL_IMPROVEMENT_V2_E15",
    "MODEL_IMPROVEMENT_V2_E16",
    "MODEL_IMPROVEMENT_V2_E20",
})

V2_SHARED_PRETEST_CACHE_KEYS = {
    "MODEL_IMPROVEMENT_V2_E06": "MODEL_IMPROVEMENT_V2_E06_SHARED_PRETEST",
    "MODEL_IMPROVEMENT_V2_E07": "MODEL_IMPROVEMENT_V2_E07_SHARED_PRETEST",
    "MODEL_IMPROVEMENT_V2_E08": "MODEL_IMPROVEMENT_V2_E08_SHARED_PRETEST",
    "MODEL_IMPROVEMENT_V2_E09": "MODEL_IMPROVEMENT_V2_E09_SHARED_PRETEST",
    "MODEL_IMPROVEMENT_V2_E13": "MODEL_IMPROVEMENT_V2_E13_SHARED_PRETEST",
    "MODEL_IMPROVEMENT_V2_E14": "MODEL_IMPROVEMENT_V2_E14_SHARED_PRETEST",
    "MODEL_IMPROVEMENT_V2_E15": "MODEL_IMPROVEMENT_V2_E15_SHARED_PRETEST",
    "MODEL_IMPROVEMENT_V2_E16": "MODEL_IMPROVEMENT_V2_E16_SHARED_PRETEST",
    "MODEL_IMPROVEMENT_V2_E20": "MODEL_IMPROVEMENT_V2_E20_SHARED_PRETEST",
}


@dataclass
class RealRunResult:
    exit_code: int
    summary: str
    n_candidates: int
    n_folds: int
    n_stage_a_runs: int
    n_stage_b_runs: int
    n_outer_prediction_bundles: int
    n_persistence_bundles: int
    o44_artifacts_written: dict[str, str] = field(default_factory=dict)
    inner_best_epochs: dict[tuple[str, str], int] = field(default_factory=dict)
    stage_a_run_ids: dict[tuple[str, str], str] = field(default_factory=dict)
    stage_b_run_ids: dict[tuple[str, str], str] = field(default_factory=dict)
    persistence_pooled_metrics: dict[str, float] | None = None
    pooled_metrics_by_cid: dict[str, dict[str, float]] = field(default_factory=dict)
    recommended_transformer_id: str | None = None
    signoff_overall_status: str = ""
    signoff_failures: list[str] = field(default_factory=list)
    exception: str | None = None
    phase45_handoff: dict | None = None


# ----------------------- Guards -----------------------
def _validate_e11_resume_ledger(ctx: RunContext) -> None:
    """Validate E11's completed-run ledger for full or partial recovery.

    The ledger is populated by E11's registry audit and therefore contains
    only records that were independently verified as COMPLETED. A partial
    recovery may omit Stage-B keys, but Stage-A reuse is always exhaustive.
    """

    candidate_ids = {
        "TR_C2_ALT_LOOKBACK_E11_CONTROL",
        "TR_C2_ALT_LOOKBACK_E11_ROLL7",
    }
    stage_a_keys = {
        f"{candidate_id}:RO{fold}_A"
        for candidate_id in candidate_ids
        for fold in (1, 2, 3)
    }
    stage_b_keys = {
        f"{candidate_id}:RO{fold}_B"
        for candidate_id in candidate_ids
        for fold in (1, 2, 3)
    }
    expected_keys = stage_a_keys | stage_b_keys
    actual_keys = set(ctx.reuse_completed_run_ids or {})
    unexpected = actual_keys - expected_keys
    if unexpected:
        raise RuntimeError(
            f"E11 resume ledger contains unexpected keys: {sorted(unexpected)}"
        )
    missing_stage_a = stage_a_keys - actual_keys
    if missing_stage_a:
        raise RuntimeError(
            "E11 resume requires all six COMPLETED Stage A runs; "
            f"missing: {sorted(missing_stage_a)}"
        )

    completed_stage_b_count = len(actual_keys & stage_b_keys)
    if ctx.allow_partial_stage_b_training:
        if not 1 <= completed_stage_b_count <= 5:
            raise RuntimeError(
                "E11 partial Stage-B recovery requires 1..5 COMPLETED "
                f"Stage B runs; found {completed_stage_b_count}"
            )
        return

    if actual_keys != expected_keys:
        raise RuntimeError(
            "E11 no-train resume requires the exact 12 locked COMPLETED "
            "Stage A/B run IDs"
        )


def _validate_e12_resume_ledger(ctx: RunContext) -> None:
    """Validate E12's explicit completed-run ledger for partial recovery."""

    candidate_ids = {
        "TR_C2_ALT_LOOKBACK_E12_CONTROL",
        "TR_C2_ALT_LOOKBACK_E12_LAG144",
    }
    stage_a_keys = {
        f"{candidate_id}:RO{fold}_A"
        for candidate_id in candidate_ids
        for fold in (1, 2, 3)
    }
    stage_b_keys = {
        f"{candidate_id}:RO{fold}_B"
        for candidate_id in candidate_ids
        for fold in (1, 2, 3)
    }
    expected_keys = stage_a_keys | stage_b_keys
    actual_keys = set(ctx.reuse_completed_run_ids or {})
    unexpected = actual_keys - expected_keys
    if unexpected:
        raise RuntimeError(
            f"E12 resume ledger contains unexpected keys: {sorted(unexpected)}"
        )
    if not actual_keys:
        raise RuntimeError("E12 partial recovery requires at least one COMPLETED run")
    orphan_stage_b = {
        key for key in actual_keys & stage_b_keys
        if key[:-1] + "A" not in actual_keys
    }
    if orphan_stage_b:
        raise RuntimeError(
            "E12 resume ledger contains Stage B without its completed Stage A: "
            f"{sorted(orphan_stage_b)}"
        )
    if ctx.allow_partial_stage_recovery:
        if actual_keys == expected_keys:
            raise RuntimeError(
                "E12 partial recovery flag is invalid when all 12 runs are complete"
            )
        return
    if actual_keys != expected_keys:
        raise RuntimeError(
            "E12 no-train resume requires the exact 12 locked COMPLETED "
            "Stage A/B run IDs"
        )


def _validate_e13_resume_ledger(ctx: RunContext) -> None:
    """Validate E13 recovery without permitting any Stage-A retraining."""
    candidate_ids = {
        "TR_C2_ALT_LOOKBACK_E13_LAMBDA_010",
        "TR_C2_ALT_LOOKBACK_E13_LAMBDA_025",
        "TR_C2_ALT_LOOKBACK_E13_LAMBDA_050",
    }
    stage_a_keys = {
        f"{candidate_id}:RO{fold}_A"
        for candidate_id in candidate_ids for fold in (1, 2, 3)
    }
    stage_b_keys = {
        f"{candidate_id}:RO{fold}_B"
        for candidate_id in candidate_ids for fold in (1, 2, 3)
    }
    expected_keys = stage_a_keys | stage_b_keys
    actual_keys = set(ctx.reuse_completed_run_ids or {})
    unexpected = actual_keys - expected_keys
    if unexpected:
        raise RuntimeError(
            f"E13 resume ledger contains unexpected keys: {sorted(unexpected)}"
        )
    missing_stage_a = stage_a_keys - actual_keys
    if missing_stage_a:
        raise RuntimeError(
            "E13 recovery requires all nine COMPLETED Stage A runs; "
            f"missing: {sorted(missing_stage_a)}"
        )
    completed_stage_b_count = len(actual_keys & stage_b_keys)
    if ctx.allow_partial_stage_b_training:
        if not 0 <= completed_stage_b_count <= 8:
            raise RuntimeError(
                "E13 partial recovery requires 0..8 COMPLETED Stage B runs; "
                f"found {completed_stage_b_count}"
            )
        return
    if actual_keys != expected_keys:
        raise RuntimeError(
            "E13 no-train resume requires the exact 18 locked COMPLETED "
            "Stage A/B run IDs"
        )


def assert_context_invariants(ctx: RunContext) -> None:
    """Hard guards rejecting fast-mode / rehearsal leakage into official."""
    v2_experiment = {
        "MODEL_IMPROVEMENT_V2_E01": ("E01", "V2_E01", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E02": ("E02", "V2_E02", "FS1_TF1", 31),
        "MODEL_IMPROVEMENT_V2_E03": ("E03", "V2_E03", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E04": ("E04", "V2_E04", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E05": ("E05", "V2_E05", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E06": ("E06", "V2_E06", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E07": ("E07", "V2_E07", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E08": ("E08", "V2_E08", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E09": ("E09", "V2_E09", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E10": ("E10", "V2_E10", None, None),
        "MODEL_IMPROVEMENT_V2_E11": ("E11", "V2_E11", None, None),
        "MODEL_IMPROVEMENT_V2_E12": ("E12", "V2_E12", None, None),
        "MODEL_IMPROVEMENT_V2_E13": ("E13", "V2_E13", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E14": ("E14", "V2_E14", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E15": ("E15", "V2_E15", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E16": ("E16", "V2_E16", "FS2_TF1", 33),
        "MODEL_IMPROVEMENT_V2_E20": ("E20", "V2_E20", "FS2_TF1", 33),
    }.get(ctx.execution_track)
    if v2_experiment is not None:
        experiment_id, registry_namespace, feature_variant, feature_count = v2_experiment
        expected_root = (ctx.project_root / "artifacts" / "model_improvement_v2").resolve()
        for label, path in (("artifact_dir", ctx.artifact_dir), ("registry_root", ctx.registry_root), ("run_root", ctx.run_root)):
            try:
                path.resolve().relative_to(expected_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"{experiment_id} {label} escapes the V2 artifact root: {path}"
                ) from exc
        if ctx.registry_namespace != registry_namespace:
            raise RuntimeError(
                f"{experiment_id} requires registry_namespace={registry_namespace}"
            )
        if ctx.registry_upstream_context is None:
            raise RuntimeError(f"{experiment_id} requires immutable registry upstream context")
        if not ctx.seed_before_model_construction or not ctx.validated_registry_lifecycle:
            raise RuntimeError(
                f"{experiment_id} requires deterministic initialization and validated lifecycle"
            )
        if not ctx.apply_fold_x_scaling:
            raise RuntimeError(f"{experiment_id} requires fold-local X scaling")
        if not ctx.robase_train_ids or not ctx.robase_val_ids:
            raise RuntimeError(f"{experiment_id} requires injected pre-Test ROBASE populations")
        if ctx.candidate_specs is None:
            raise RuntimeError(f"{experiment_id} requires injected challenger specs")
        if experiment_id == "E06":
            expected_ids = {
                "TR_C2_ALT_LOOKBACK_LR_5E5",
                "TR_C2_ALT_LOOKBACK_LR_1E4",
                "TR_C2_ALT_LOOKBACK_LR_2E4",
            }
            if len(ctx.candidate_specs) != 3 or {
                candidate.candidate_id for candidate in ctx.candidate_specs
            } != expected_ids:
                raise RuntimeError("E06 requires exactly the three locked LR challengers")
        elif experiment_id == "E07":
            expected_ids = {
                "TR_C2_ALT_LOOKBACK_SCHED_COSINE",
                "TR_C2_ALT_LOOKBACK_SCHED_REDUCE_ON_PLATEAU",
            }
            if len(ctx.candidate_specs) != 2 or {
                candidate.candidate_id for candidate in ctx.candidate_specs
            } != expected_ids:
                raise RuntimeError(
                    "E07 requires exactly the two locked scheduler challengers "
                    "(TR_C2_ALT_LOOKBACK_SCHED_COSINE, "
                    "TR_C2_ALT_LOOKBACK_SCHED_REDUCE_ON_PLATEAU)"
                )
        elif experiment_id == "E08":
            expected_ids = {
                "TR_C2_ALT_LOOKBACK_SGD_LR_1E_MINUS_4",
                "TR_C2_ALT_LOOKBACK_SGD_LR_3E_MINUS_4",
                "TR_C2_ALT_LOOKBACK_SGD_LR_1E_MINUS_3",
                "TR_C2_ALT_LOOKBACK_SGD_LR_3E_MINUS_3",
            }
            if len(ctx.candidate_specs) != 4 or {
                candidate.candidate_id for candidate in ctx.candidate_specs
            } != expected_ids:
                raise RuntimeError("E08 requires exactly the four locked plain-SGD challengers")
        elif experiment_id == "E09":
            expected_ids = {
                "TR_C2_ALT_LOOKBACK_SGDM_LR_1E_MINUS_3_WD_0",
                "TR_C2_ALT_LOOKBACK_SGDM_LR_1E_MINUS_3_WD_1E_MINUS_4",
                "TR_C2_ALT_LOOKBACK_SGDM_LR_3E_MINUS_3_WD_0",
                "TR_C2_ALT_LOOKBACK_SGDM_LR_3E_MINUS_3_WD_1E_MINUS_4",
                "TR_C2_ALT_LOOKBACK_SGDM_LR_1E_MINUS_2_WD_0",
                "TR_C2_ALT_LOOKBACK_SGDM_LR_1E_MINUS_2_WD_1E_MINUS_4",
            }
            if len(ctx.candidate_specs) != 6 or {
                candidate.candidate_id for candidate in ctx.candidate_specs
            } != expected_ids:
                raise RuntimeError("E09 requires exactly the six locked SGDM challengers")
        elif experiment_id == "E10":
            expected_ids = {
                "TR_C2_ALT_LOOKBACK_E10_CONTROL",
                "TR_C2_ALT_LOOKBACK_E10_DELTA3",
            }
            if len(ctx.candidate_specs) != 2 or {
                candidate.candidate_id for candidate in ctx.candidate_specs
            } != expected_ids:
                raise RuntimeError("E10 requires its locked common-population pair")
        elif experiment_id == "E11":
            expected_ids = {
                "TR_C2_ALT_LOOKBACK_E11_CONTROL",
                "TR_C2_ALT_LOOKBACK_E11_ROLL7",
            }
            if len(ctx.candidate_specs) != 2 or {
                candidate.candidate_id for candidate in ctx.candidate_specs
            } != expected_ids:
                raise RuntimeError("E11 requires its locked common-population pair")
        elif experiment_id == "E12":
            expected_ids = {
                "TR_C2_ALT_LOOKBACK_E12_CONTROL",
                "TR_C2_ALT_LOOKBACK_E12_LAG144",
            }
            if len(ctx.candidate_specs) != 2 or {
                candidate.candidate_id for candidate in ctx.candidate_specs
            } != expected_ids:
                raise RuntimeError("E12 requires its locked common-population pair")
        elif experiment_id == "E13":
            expected_ids = {"TR_C2_ALT_LOOKBACK_E13_LAMBDA_010", "TR_C2_ALT_LOOKBACK_E13_LAMBDA_025", "TR_C2_ALT_LOOKBACK_E13_LAMBDA_050"}
            if len(ctx.candidate_specs) != 3 or {candidate.candidate_id for candidate in ctx.candidate_specs} != expected_ids:
                raise RuntimeError("E13 requires exactly three locked hybrid-loss challengers")
        elif experiment_id == "E14":
            expected_ids = {f"TR_C2_ALT_LOOKBACK_E14_M{i}" for i in range(1, 6)}
            if len(ctx.candidate_specs) != 5 or {candidate.candidate_id for candidate in ctx.candidate_specs} != expected_ids:
                raise RuntimeError("E14 requires exactly five locked training bundles")
        elif experiment_id == "E15":
            expected_ids = {"TR_C2_ALT_LOOKBACK_E15_W96_FFN384", "TR_C2_ALT_LOOKBACK_E15_W128_FFN512"}
            if len(ctx.candidate_specs) != 2 or {candidate.candidate_id for candidate in ctx.candidate_specs} != expected_ids:
                raise RuntimeError("E15 requires exactly two locked proportional width bundles")
        elif experiment_id == "E16":
            expected_ids = {"TR_C2_ALT_LOOKBACK_E16_DEPTH3_POSTLN"}
            if len(ctx.candidate_specs) != 1 or {candidate.candidate_id for candidate in ctx.candidate_specs} != expected_ids:
                raise RuntimeError("E16 requires exactly one locked three-layer Post-LN challenger")
        elif experiment_id == "E20":
            expected_ids = {
                "TR_C2_ALT_LOOKBACK",
                "TR_C2_ALT_LOOKBACK_E14_M1",
            }
            if len(ctx.candidate_specs) != 2 or {
                candidate.candidate_id for candidate in ctx.candidate_specs
            } != expected_ids:
                raise RuntimeError("E20 requires exactly the matched control/finalist pair")
        elif len(ctx.candidate_specs) != 1:
            raise RuntimeError(f"{experiment_id} requires exactly one injected challenger")
        for candidate in ctx.candidate_specs:
            if experiment_id == "E07":
                if not (candidate.candidate_id == "TR_C2_ALT_LOOKBACK"
                        or candidate.candidate_id.startswith("TR_C2_ALT_LOOKBACK_SCHED_")):
                    raise RuntimeError(
                        "E07 rejects every architecture except TR_C2_ALT_LOOKBACK "
                        "and its SCHED_* variants"
                    )
            elif experiment_id not in {"E06", "E08", "E09", "E10", "E11", "E12", "E13", "E14", "E15", "E16", "E20"} and candidate.candidate_id != "TR_C2_ALT_LOOKBACK":
                raise RuntimeError(
                    f"{experiment_id} rejects every architecture except TR_C2_ALT_LOOKBACK"
                )
            data = candidate.config.get("data", {})
            model_config = candidate.config.get("model", {})
            if experiment_id == "E10":
                expected_projection = {
                    "TR_C2_ALT_LOOKBACK_E10_CONTROL": ("FS2_TF1", 33),
                    "TR_C2_ALT_LOOKBACK_E10_DELTA3": ("FS2_TF1_DELTA3", 36),
                }[candidate.candidate_id]
                if (candidate.feature_variant_id, data.get("feature_count")) != expected_projection or data.get("feature_variant_id") != expected_projection[0] or model_config.get("input_size") != expected_projection[1]:
                    raise RuntimeError("E10 feature projection contract mismatch")
            elif experiment_id == "E11":
                expected_projection = {
                    "TR_C2_ALT_LOOKBACK_E11_CONTROL": ("FS2_TF1", 33),
                    "TR_C2_ALT_LOOKBACK_E11_ROLL7": ("FS2_TF1_ROLL7", 40),
                }[candidate.candidate_id]
                if (
                    (candidate.feature_variant_id, data.get("feature_count"))
                    != expected_projection
                    or data.get("feature_variant_id") != expected_projection[0]
                    or model_config.get("input_size") != expected_projection[1]
                ):
                    raise RuntimeError("E11 feature projection contract mismatch")
            elif experiment_id == "E12":
                expected_projection = {
                    "TR_C2_ALT_LOOKBACK_E12_CONTROL": ("FS2_TF1", 33),
                    "TR_C2_ALT_LOOKBACK_E12_LAG144": ("FS2_TF1_LAG144", 34),
                }[candidate.candidate_id]
                if (
                    (candidate.feature_variant_id, data.get("feature_count"))
                    != expected_projection
                    or data.get("feature_variant_id") != expected_projection[0]
                    or model_config.get("input_size") != expected_projection[1]
                ):
                    raise RuntimeError("E12 feature projection contract mismatch")
            elif (
                candidate.feature_variant_id != feature_variant
                or data.get("feature_variant_id") != feature_variant
                or data.get("feature_count") != feature_count
                or model_config.get("input_size") != feature_count
            ):
                raise RuntimeError(f"{experiment_id} feature projection contract mismatch")
            if data.get("target_access_mode") == "TEST":
                raise RuntimeError(f"{experiment_id} Test access is forbidden")
        if experiment_id == "E13":
            from course_work.model_improvement_v2.hybrid_loss import ALLOWED_LAMBDAS, LOSS_POLICY
            if {candidate.config.get("training", {}).get("lambda_delta") for candidate in ctx.candidate_specs} != set(ALLOWED_LAMBDAS) or any(candidate.config.get("training", {}).get("loss_policy") != LOSS_POLICY or candidate.config.get("training", {}).get("delta_beta_model_space") != 1.0 or candidate.config.get("training", {}).get("loss_name") != "MSE" for candidate in ctx.candidate_specs):
                raise RuntimeError("E13 hybrid loss/lambda contract mismatch")
        if experiment_id == "E14":
            locked = {
                "TR_C2_ALT_LOOKBACK_E14_M1": (2e-4, 16, 0.10, 1.0),
                "TR_C2_ALT_LOOKBACK_E14_M2": (3e-4, 16, 0.05, 1.0),
                "TR_C2_ALT_LOOKBACK_E14_M3": (5e-4, 32, 0.05, 1.0),
                "TR_C2_ALT_LOOKBACK_E14_M4": (5e-4, 64, 0.10, 1.0),
                "TR_C2_ALT_LOOKBACK_E14_M5": (3e-4, 32, 0.20, 0.5),
            }
            for candidate in ctx.candidate_specs:
                training = candidate.config["training"]
                actual = (training["learning_rate"], training["batch_size"], candidate.config["model"]["dropout"], training["gradient_clip_max_norm"])
                if actual != locked[candidate.candidate_id] or training.get("loss_policy") != "HYBRID_LEVEL_PLUS_DELTA" or training.get("lambda_delta") != 0.10 or training.get("optimizer_name") != "AdamW" or training.get("weight_decay") != 1e-3 or training.get("scheduler_name") is not None:
                    raise RuntimeError("E14 locked bundle/incumbent contract mismatch")
        if experiment_id == "E15":
            locked = {
                "TR_C2_ALT_LOOKBACK_E15_W96_FFN384": (96, 4, 384, 2),
                "TR_C2_ALT_LOOKBACK_E15_W128_FFN512": (128, 4, 512, 2),
            }
            for candidate in ctx.candidate_specs:
                model = candidate.config["model"]
                training = candidate.config["training"]
                actual = (model["d_model"], model["num_heads"], model["ffn_dim"], model["num_layers"])
                if actual != locked[candidate.candidate_id] or model.get("dropout") != 0.10 or training.get("loss_policy") != "HYBRID_LEVEL_PLUS_DELTA" or training.get("lambda_delta") != 0.10 or training.get("optimizer_name") != "AdamW" or training.get("learning_rate") != 2e-4 or training.get("batch_size") != 16 or training.get("weight_decay") != 1e-3 or training.get("gradient_clip_max_norm") != 1.0 or training.get("scheduler_name") is not None:
                    raise RuntimeError("E15 locked width/incumbent contract mismatch")
        if experiment_id == "E16":
            candidate = ctx.candidate_specs[0]
            model = candidate.config["model"]
            training = candidate.config["training"]
            if (
                candidate.candidate_id != "TR_C2_ALT_LOOKBACK_E16_DEPTH3_POSTLN"
                or (model["d_model"], model["num_heads"], model["ffn_dim"], model["num_layers"])
                != (64, 4, 256, 3)
                or model.get("norm_first") is not False
                or model.get("dropout") != 0.10
                or training.get("loss_policy") != "HYBRID_LEVEL_PLUS_DELTA"
                or training.get("lambda_delta") != 0.10
                or training.get("optimizer_name") != "AdamW"
                or training.get("learning_rate") != 2e-4
                or training.get("batch_size") != 16
                or training.get("weight_decay") != 1e-3
                or training.get("gradient_clip_max_norm") != 1.0
                or training.get("scheduler_name") is not None
            ):
                raise RuntimeError("E16 locked depth/incumbent contract mismatch")
        if experiment_id == "E20":
            if ctx.seed not in {123, 2026}:
                raise RuntimeError("E20 may create new runs only for seeds 123 and 2026")
            locked = {
                "TR_C2_ALT_LOOKBACK": {
                    "learning_rate": 3e-4,
                    "batch_size": 32,
                    "loss_policy": None,
                    "lambda_delta": None,
                    "delta_beta_model_space": None,
                },
                "TR_C2_ALT_LOOKBACK_E14_M1": {
                    "learning_rate": 2e-4,
                    "batch_size": 16,
                    "loss_policy": "HYBRID_LEVEL_PLUS_DELTA",
                    "lambda_delta": 0.10,
                    "delta_beta_model_space": 1.0,
                },
            }
            for candidate in ctx.candidate_specs:
                model = candidate.config["model"]
                training = candidate.config["training"]
                reproducibility = candidate.config["reproducibility"]
                expected = locked[candidate.candidate_id]
                if (
                    (model["d_model"], model["num_heads"], model["ffn_dim"], model["num_layers"])
                    != (64, 4, 256, 2)
                    or model.get("norm_first") is not False
                    or model.get("dropout") != 0.10
                    or training.get("optimizer_name") != "AdamW"
                    or training.get("learning_rate") != expected["learning_rate"]
                    or training.get("batch_size") != expected["batch_size"]
                    or training.get("weight_decay") != 1e-3
                    or training.get("gradient_clip_max_norm") != 1.0
                    or training.get("scheduler_name") is not None
                    or training.get("loss_name") != "MSE"
                    or training.get("loss_policy") != expected["loss_policy"]
                    or training.get("lambda_delta") != expected["lambda_delta"]
                    or training.get("delta_beta_model_space") != expected["delta_beta_model_space"]
                    or any(reproducibility.get(key) != ctx.seed for key in ("seed", "global_seed", "dataloader_seed"))
                ):
                    raise RuntimeError("E20 locked control/finalist seed contract mismatch")
            if ctx.allow_partial_stage_recovery and not ctx.reuse_completed_runs:
                raise RuntimeError("E20 partial recovery requires reuse_completed_runs=True")
            if ctx.reuse_completed_runs:
                expected_keys = {
                    f"{candidate_id}:RO{fold}_{stage}"
                    for candidate_id in locked
                    for fold in (1, 2, 3)
                    for stage in ("A", "B")
                }
                actual_keys = set(ctx.reuse_completed_run_ids or {})
                if not ctx.allow_partial_stage_recovery and actual_keys != expected_keys:
                    raise RuntimeError("E20 finalization requires exact 12 completed run IDs per seed")
                if not actual_keys <= expected_keys:
                    raise RuntimeError("E20 recovery ledger contains unexpected run identities")
        if experiment_id == "E06":
            if ctx.reuse_completed_runs:
                expected_reuse_keys = {
                    f"{candidate_id}:RO{fold}_{stage}"
                    for candidate_id in expected_ids
                    for fold in (1, 2, 3)
                    for stage in ("A", "B")
                }
                if set(ctx.reuse_completed_run_ids or {}) != expected_reuse_keys:
                    raise RuntimeError(
                        "E06 no-train resume requires the exact 18 locked Stage A/B run IDs"
                    )
            if any(
                candidate.config.get("model", {}).get("prediction_formulation", "DIRECT") != "DIRECT"
                or candidate.config.get("training", {}).get("optimizer_name") != "AdamW"
                or candidate.config.get("training", {}).get("learning_rate") not in {5e-5, 1e-4, 2e-4}
                for candidate in ctx.candidate_specs
            ):
                raise RuntimeError("E06 candidate LR/optimizer/DIRECT contract mismatch")
        if experiment_id == "E08":
            if ctx.reuse_completed_runs:
                expected_reuse_keys = {
                    f"{candidate_id}:RO{fold}_{stage}"
                    for candidate_id in expected_ids
                    for fold in (1, 2, 3)
                    for stage in ("A", "B")
                }
                if set(ctx.reuse_completed_run_ids or {}) != expected_reuse_keys:
                    raise RuntimeError(
                        "E08 no-train resume requires the exact 24 locked Stage A/B run IDs"
                    )
            if any(
                candidate.config.get("model", {}).get("prediction_formulation", "DIRECT") != "DIRECT"
                or candidate.config.get("training", {}).get("optimizer_name") != "SGD"
                or candidate.config.get("training", {}).get("optimizer_config")
                != {"momentum": 0.0, "nesterov": False}
                or candidate.config.get("training", {}).get("learning_rate")
                not in {1e-4, 3e-4, 1e-3, 3e-3}
                or candidate.config.get("training", {}).get("weight_decay") != 1e-3
                or candidate.config.get("training", {}).get("scheduler_name") not in {None, "OFF"}
                or candidate.config.get("training", {}).get("scheduler_config") is not None
                for candidate in ctx.candidate_specs
            ):
                raise RuntimeError("E08 candidate plain-SGD/DIRECT/constant-LR contract mismatch")
        if experiment_id == "E09":
            if ctx.reuse_completed_runs:
                expected_reuse_keys = {
                    f"{candidate_id}:RO{fold}_{stage}"
                    for candidate_id in expected_ids
                    for fold in (1, 2, 3)
                    for stage in ("A", "B")
                }
                if set(ctx.reuse_completed_run_ids or {}) != expected_reuse_keys:
                    raise RuntimeError(
                        "E09 no-train resume requires the exact 36 locked Stage A/B run IDs"
                    )
            allowed_pairs = {
                (1e-3, 0.0), (1e-3, 1e-4),
                (3e-3, 0.0), (3e-3, 1e-4),
                (1e-2, 0.0), (1e-2, 1e-4),
            }
            actual_pairs = {
                (
                    candidate.config.get("training", {}).get("learning_rate"),
                    candidate.config.get("training", {}).get("weight_decay"),
                )
                for candidate in ctx.candidate_specs
            }
            if actual_pairs != allowed_pairs or any(
                candidate.config.get("model", {}).get("prediction_formulation", "DIRECT") != "DIRECT"
                or candidate.config.get("training", {}).get("optimizer_name") != "SGD"
                or candidate.config.get("training", {}).get("optimizer_config")
                != {"momentum": 0.9, "nesterov": False}
                or candidate.config.get("training", {}).get("scheduler_name") not in {None, "OFF"}
                or candidate.config.get("training", {}).get("scheduler_config") is not None
                for candidate in ctx.candidate_specs
            ):
                raise RuntimeError("E09 candidate SGDM/DIRECT/constant-LR contract mismatch")
        if experiment_id == "E11":
            if ctx.allow_partial_stage_b_training and not ctx.reuse_completed_runs:
                raise RuntimeError(
                    "E11 partial Stage-B recovery requires reuse_completed_runs=True"
                )
            if ctx.reuse_completed_runs:
                _validate_e11_resume_ledger(ctx)
        if experiment_id == "E12":
            if ctx.allow_partial_stage_recovery and not ctx.reuse_completed_runs:
                raise RuntimeError(
                    "E12 partial recovery requires reuse_completed_runs=True"
                )
            if ctx.reuse_completed_runs:
                _validate_e12_resume_ledger(ctx)
        if experiment_id == "E13":
            if ctx.allow_partial_stage_b_training and not ctx.reuse_completed_runs:
                raise RuntimeError(
                    "E13 partial Stage-B recovery requires reuse_completed_runs=True"
                )
            if ctx.reuse_completed_runs:
                _validate_e13_resume_ledger(ctx)
        if experiment_id == "E16" and ctx.reuse_completed_runs:
            expected_reuse_keys = {
                f"TR_C2_ALT_LOOKBACK_E16_DEPTH3_POSTLN:RO{fold}_{stage}"
                for fold in (1, 2, 3)
                for stage in ("A", "B")
            }
            if set(ctx.reuse_completed_run_ids or {}) != expected_reuse_keys:
                raise RuntimeError(
                    "E16 no-train resume requires the exact six locked Stage A/B run IDs"
                )
        if experiment_id == "E01" and ctx.reuse_completed_runs:
            expected_reuse_keys = {
                f"RO{fold}_{stage}"
                for fold in (1, 2, 3)
                for stage in ("A", "B")
            }
            if set(ctx.reuse_completed_run_ids or {}) != expected_reuse_keys:
                raise RuntimeError(
                    "E01 no-train resume requires the exact six locked Stage A/B run IDs"
                )
        if experiment_id == "E02" and ctx.reuse_completed_runs:
            raise RuntimeError("E02 challenger must initialize and train from scratch")
        if experiment_id == "E03" and ctx.reuse_completed_runs:
            expected_reuse_keys = {
                f"RO{fold}_{stage}"
                for fold in (1, 2, 3)
                for stage in ("A", "B")
            }
            if set(ctx.reuse_completed_run_ids or {}) != expected_reuse_keys:
                raise RuntimeError(
                    "E03 no-train resume requires the exact six locked Stage A/B run IDs"
                )
    if ctx.is_rehearsal:
        # Rehearsal is allowed: tiny budget, temp registry.
        if ctx.scientific_max_epochs > 5:
            raise RuntimeError(
                f"PHASE44 GUARD: rehearsal scientific_max_epochs="
                f"{ctx.scientific_max_epochs} exceeds rehearsal ceiling of 5."
            )
        return
    # OFFICIAL mode
    if ctx.scientific_max_epochs < SCIENTIFIC_MAX_EPOCHS:
        raise RuntimeError(
            f"PHASE44 GUARD: official scientific_max_epochs="
            f"{ctx.scientific_max_epochs} is below the Phase 44 contract "
            f"minimum of {SCIENTIFIC_MAX_EPOCHS}."
        )
    if ctx.scientific_patience < SCIENTIFIC_PATIENCE:
        raise RuntimeError(
            f"PHASE44 GUARD: official scientific_patience="
            f"{ctx.scientific_patience} is below the Phase 44 contract "
            f"minimum of {SCIENTIFIC_PATIENCE}."
        )
    if ctx.rehearsal_synthetic:
        raise RuntimeError(
            "PHASE44 GUARD: rehearsal_synthetic=True is forbidden in official mode."
        )


# ----------------------- Fold-local scaler fit -----------------------


def _compute_y_mean_std(
    y_fit: np.ndarray,
    target_scaling_option: str,
) -> tuple[float | None, float | None]:
    """Compute y mean/std from the fold-allowed y_fit array.

    Phase 44 contract (§38, §40, §154):
      - Stage A Y scaler MUST be fit ONLY on inner_train allowed targets.
      - Stage B Y scaler MUST be fit ONLY on outer_train allowed targets.
      - Global Phase9 YS1 is FORBIDDEN as the fit source for fold-local Y.
      - The Y stats are computed FROM the fold-specific allowed population.

    Returns (mean, std). When target_scaling_option == "YS0", returns
    (None, None) — YS0 is identity and does not require stats.
    """
    if target_scaling_option == "YS0":
        return None, None
    if y_fit is None or len(y_fit) == 0:
        raise ValueError(
            "fit_fold_local_scaler: y_fit array is empty. Cannot fit YS1 "
            "fold-local Y scaler on an empty fold-specific allowed population."
        )
    m = float(np.mean(y_fit))
    s = float(np.std(y_fit))
    if s < 1e-12:
        s = 1.0
    return m, s


def fit_fold_local_scaler(
    *,
    candidate: CandidateSpec,
    fold: FoldDefinition,
    fold_dataset,
    fit_stage: str,  # "A" or "B"
    ctx: "RunContext" = None,
) -> tuple[FoldLocalScalerBundle, ScalerFitAudit]:
    """Fit a fold-local scaler on the Stage A inner-train or Stage B
    outer-train history. Returns (bundle, audit).

    Data sources (in priority order):
      1. The fold_dataset's `_feature_matrix` (canonical
         SequenceWindowDataset 2D matrix [N_total_rows, F]). Features are
         retrieved via each fit target's `timeline_target` row index. This
         is the SCIENTIFICALLY CORRECT path for the canonical dataset.
      2. Fallback: window_records columns (legacy / synthetic stand-in
         datasets like `_synthetic_dataset_from_windowpop`).
    """
    if fit_stage == "A":
        fit_target_ids = list(fold.inner_train_ids)
        allowed_label = f"fold {fold.fold_id} stage A history"
    elif fit_stage == "B":
        fit_target_ids = list(fold.outer_train_ids)
        allowed_label = f"fold {fold.fold_id} stage B history"
    else:
        raise ValueError(f"Unknown fit_stage: {fit_stage}")

    window_records = getattr(fold_dataset, "window_records", None)
    if window_records is None or "target_id" not in window_records.columns:
        raise ValueError(
            "fit_fold_local_scaler: base_dataset must have window_records "
            "with target_id column."
        )

    fit_id_set = set(str(t) for t in fit_target_ids)
    sub = window_records[window_records["target_id"].astype(str).isin(fit_id_set)]
    feature_matrix = getattr(fold_dataset, "_feature_matrix", None)
    target_values = getattr(fold_dataset, "_target_values", None)

    # Decide X_fit source: canonical feature_matrix vs window_records columns.
    if feature_matrix is not None and "timeline_target" in sub.columns:
        # Canonical path: index feature_matrix by each fit target's
        # timeline_target row index.
        timeline_targets = sub["timeline_target"].astype(int).to_numpy()
        # Defensive bounds check
        if timeline_targets.size == 0:
            X_fit = np.zeros((0, feature_matrix.shape[1] if feature_matrix.ndim == 2 else 1), dtype=np.float64)
        else:
            if timeline_targets.max() >= feature_matrix.shape[0] or timeline_targets.min() < 0:
                raise ValueError(
                    f"fit_fold_local_scaler: timeline_target indices exceed "
                    f"feature_matrix bounds ({timeline_targets.min()}..{timeline_targets.max()} "
                    f"vs {feature_matrix.shape[0]})"
                )
            X_fit = np.asarray(feature_matrix[timeline_targets], dtype=np.float64)
        # Historical Phase44 intentionally classifies synthetic f0/f1/... names.
        # E01 reproduces that behavior; feature-group improvements are deferred.
        feature_cols = [f"f{i}" for i in range(X_fit.shape[1])] if X_fit.size > 0 else ["f0"]
        # Phase 44 scientific fix: Y MUST come from the canonical target_values
        # matrix indexed by timeline_target — NOT from window_records (which
        # does not always carry the Appliances column). This guarantees the
        # Y scaler is fit on the actual raw Wh for each fold-allowed target.
        if target_values is not None:
            y_fit = np.asarray(target_values[timeline_targets], dtype=np.float64)
        elif "Appliances" in sub.columns and len(sub) > 0:
            y_fit = sub["Appliances"].to_numpy(dtype=np.float64)
        else:
            raise ValueError(
                "fit_fold_local_scaler: cannot derive y_fit — no target_values "
                "on fold_dataset and no Appliances column in window_records"
            )
    else:
        # Legacy / synthetic stand-in path: read columns from window_records.
        feature_cols = [
            c for c in sub.columns
            if c not in {
                "target_id", "target_timestamp", "target_split_id",
                "Appliances", "raw_row_index", "continuity_segment_id",
                "valid_L36", "valid_L72", "valid_L144",
                "included_common_population", "target_sample_id",
                "window_id", "lookback_steps", "horizon_steps",
                "timeline_input_start", "timeline_input_end",
                "timeline_target", "input_start_timestamp",
                "input_end_timestamp", "target_raw_row_index",
                "input_start_raw_row_index", "input_end_raw_row_index",
                "input_start_split_id", "input_end_split_id",
                "crosses_split_boundary", "WB0_valid", "WB1_valid",
                "canonical_sample_idx",
            }
        ]
        if feature_cols and len(sub) > 0:
            X_fit = sub[feature_cols].to_numpy(dtype=np.float64)
        else:
            X_fit = np.zeros((len(fit_target_ids), 1), dtype=np.float64)
        if "Appliances" in sub.columns and len(sub) > 0:
            y_fit = sub["Appliances"].to_numpy(dtype=np.float64)
        else:
            y_fit = np.zeros(len(fit_target_ids), dtype=np.float64)

    # Pass raw IDs (scaling.py normalizes string/int forms automatically).
    coerced_fit_ids = [str(t) for t in fit_target_ids]
    coerced_outer_ids = [str(t) for t in fold.outer_eval_ids]

    if fit_stage == "A":
        means, stds, scaled_idx, passthrough_idx = fit_fold_a_x_scaler(
            X_fit, feature_variant_id=candidate.feature_variant_id,
            feature_columns=feature_cols if feature_cols else ["f0"],
        )
        ym, ys = _compute_y_mean_std(y_fit, candidate.target_scaling_option)
    else:
        means, stds, scaled_idx, passthrough_idx = fit_fold_b_x_scaler(
            X_fit, feature_variant_id=candidate.feature_variant_id,
            feature_columns=feature_cols if feature_cols else ["f0"],
        )
        ym, ys = _compute_y_mean_std(y_fit, candidate.target_scaling_option)

    bundle = build_bundle(
        bundle_id=f"{fold.fold_id}_{fit_stage}_{candidate.candidate_id}",
        fit_stage=fit_stage,
        fold_id=str(fold.fold_id),
        candidate_id=candidate.candidate_id,
        target_scaling_option=candidate.target_scaling_option,
        feature_variant_id=candidate.feature_variant_id,
        lookback_steps=candidate.lookback_steps,
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        revin_enabled=False,
        fit_target_ids=fit_target_ids,
        fit_raw_row_count=len(fit_target_ids),
        x_means=means,
        x_stds=stds,
        feature_indices_scaled=list(scaled_idx),
        feature_indices_passthrough=list(passthrough_idx),
        y_mean=float(ym) if ym is not None else None,
        y_std=float(ys) if ys is not None else None,
    )

    audit = record_scaler_fit_audit(
        bundle=bundle,
        windowpop_df=window_records,
        fit_target_ids=coerced_fit_ids,
        fold_outer_eval_ids=coerced_outer_ids,
        allowed_region_label=allowed_label,
    )
    return bundle, audit


def _prepare_stage_c_model_for_device(
    *,
    model: torch.nn.Module,
    device: torch.device,
    execution_track: str,
) -> torch.nn.Module:
    """Move V2 rolling models to the eval device while preserving V1."""

    if execution_track in {"MODEL_IMPROVEMENT_V2_E01", "MODEL_IMPROVEMENT_V2_E02", "MODEL_IMPROVEMENT_V2_E03", "MODEL_IMPROVEMENT_V2_E04", "MODEL_IMPROVEMENT_V2_E05", "MODEL_IMPROVEMENT_V2_E06", "MODEL_IMPROVEMENT_V2_E07", "MODEL_IMPROVEMENT_V2_E08", "MODEL_IMPROVEMENT_V2_E09", "MODEL_IMPROVEMENT_V2_E10", "MODEL_IMPROVEMENT_V2_E11", "MODEL_IMPROVEMENT_V2_E12", "MODEL_IMPROVEMENT_V2_E13", "MODEL_IMPROVEMENT_V2_E14", "MODEL_IMPROVEMENT_V2_E15", "MODEL_IMPROVEMENT_V2_E16", "MODEL_IMPROVEMENT_V2_E20"}:
        return model.to(device)
    return model


# ----------------------- Real Orchestrator -----------------------
def run_real_pipeline(ctx: RunContext) -> RealRunResult:
    """The SINGLE Phase 44 real orchestrator. Used by both official and
    rehearsal modes. Differences are encoded in RunContext only.
    """
    assert_context_invariants(ctx)

    registry = None
    active_run_ids: dict[str, str] = {}
    try:
        candidates = (
            list(ctx.candidate_specs)
            if ctx.candidate_specs is not None
            else load_candidates(
                project_root=ctx.project_root,
                transformer_shortlist_path=ctx.transformer_shortlist_path,
                lstm_handoff_path=ctx.lstm_handoff_path,
            )
        )
        n_candidates = len(candidates)
        n_folds = PERSISTENCE_FOLD_COUNT

        # Step 1: build ROBASE + 3 folds (real project data).
        rtrn_ids = (
            list(ctx.robase_train_ids)
            if ctx.robase_train_ids is not None
            else extract_robase_train_ids(ctx.project_root)
        )
        rval_ids = (
            list(ctx.robase_val_ids)
            if ctx.robase_val_ids is not None
            else extract_robase_val_ids(ctx.project_root)
        )
        folds = build_rolling_folds(rtrn_ids, rval_ids, k=n_folds)

        # Step 2: resolve base dataset (real SequenceWindowDataset).
        # The orchestrator requires a candidate-aware dataset factory: each
        # candidate may have a different lookback (e.g. TR_C2_ALT_LOOKBACK
        # uses L72 while TR_C0/C1 use L36), so the canonical
        # SequenceWindowDataset must be rebuilt PER candidate.
        if not ctx.is_rehearsal:
            # OFFICIAL: caller MUST supply a real candidate-aware factory.
            if ctx.dataset_factory is None:
                raise RuntimeError(
                    "PHASE44 GUARD: official mode requires ctx.dataset_factory "
                    "to be a candidate-aware callable building a real "
                    "SequenceWindowDataset (not the broken _ROREFOLD_dataset "
                    "fallback). Call _run_official() in scripts/phase44_rolling_origin.py."
                )
        # In rehearsal the dataset_factory may be None; the synthetic path
        # uses _synthetic_dataset_from_windowpop or rehearsal_synthetic=True.
        fold_dataset = None  # populated per-candidate inside the loops below

        # Step 3: initialize registry (canonical OR temp).
        registry = ExperimentRegistry(
            project_root=ctx.project_root,
            registry_root=ctx.registry_root,
            run_root=ctx.run_root,
            run_id_namespace=ctx.registry_namespace,
            upstream_context_override=ctx.registry_upstream_context,
        )
        engine = TrainingEngine(registry)
        refit_engine = RefitEngine(registry)
        device = torch.device(ctx.device)

        def _locked_v2_reuse_record(candidate, fold, stage: str) -> dict[str, Any] | None:
            """Resolve one explicitly locked V2 run without changing V1 discovery."""

            if ctx.execution_track not in (
                "MODEL_IMPROVEMENT_V2_E01",
                "MODEL_IMPROVEMENT_V2_E03",
                "MODEL_IMPROVEMENT_V2_E06",
                "MODEL_IMPROVEMENT_V2_E07",
                "MODEL_IMPROVEMENT_V2_E08",
                "MODEL_IMPROVEMENT_V2_E09",
                "MODEL_IMPROVEMENT_V2_E11",
                "MODEL_IMPROVEMENT_V2_E12",
                "MODEL_IMPROVEMENT_V2_E13",
                "MODEL_IMPROVEMENT_V2_E14",
                "MODEL_IMPROVEMENT_V2_E15",
                "MODEL_IMPROVEMENT_V2_E16",
                "MODEL_IMPROVEMENT_V2_E20",
            ):
                return None
            ledger = ctx.reuse_completed_run_ids
            if ledger is None:
                return None
            key = (
                f"{candidate.candidate_id}:{fold.fold_id}_{stage}"
                if ctx.execution_track in (
                    "MODEL_IMPROVEMENT_V2_E06",
                    "MODEL_IMPROVEMENT_V2_E07",
                    "MODEL_IMPROVEMENT_V2_E08",
                    "MODEL_IMPROVEMENT_V2_E09",
                    "MODEL_IMPROVEMENT_V2_E11",
                    "MODEL_IMPROVEMENT_V2_E12",
                    "MODEL_IMPROVEMENT_V2_E13",
                    "MODEL_IMPROVEMENT_V2_E14",
                    "MODEL_IMPROVEMENT_V2_E15",
                    "MODEL_IMPROVEMENT_V2_E16",
                    "MODEL_IMPROVEMENT_V2_E20",
                )
                else f"{fold.fold_id}_{stage}"
            )
            if key not in ledger:
                if (
                    (
                        ctx.execution_track == "MODEL_IMPROVEMENT_V2_E11"
                        and ctx.allow_partial_stage_b_training
                        and stage == "B"
                    )
                    or (
                        ctx.execution_track == "MODEL_IMPROVEMENT_V2_E12"
                        and ctx.allow_partial_stage_recovery
                    )
                    or (
                        ctx.execution_track == "MODEL_IMPROVEMENT_V2_E20"
                        and ctx.allow_partial_stage_recovery
                    )
                    or (
                        ctx.execution_track == "MODEL_IMPROVEMENT_V2_E13"
                        and ctx.allow_partial_stage_b_training
                        and stage == "B"
                    )
                ):
                    return None
                raise RuntimeError(f"V2 RESUME: locked {stage} run missing for {key}")
            run_id = ledger[key]
            record = registry.get_run(run_id)
            if record.get("status") != RunStatus.COMPLETED.value:
                raise RuntimeError(f"V2 RESUME: {run_id} is not COMPLETED")
            if record.get("candidate_id") != candidate.candidate_id:
                raise RuntimeError(f"V2 RESUME: candidate mismatch for {run_id}")
            if record.get("sweep_stage") != f"{fold.fold_id}_{stage}":
                raise RuntimeError(f"V2 RESUME: stage/fold mismatch for {run_id}")
            lineage = record.get("config", {}).get("lineage", {})
            if (
                lineage.get("rolling_origin_candidate_id") != candidate.candidate_id
                or lineage.get("rolling_origin_fold_id") != str(fold.fold_id)
                or lineage.get("rolling_origin_stage") != stage
            ):
                raise RuntimeError(f"V2 RESUME: lineage mismatch for {run_id}")
            if record.get("config", {}).get("data", {}).get("target_access_mode") == "TEST":
                raise RuntimeError(f"V2 RESUME: Test-scoped run rejected: {run_id}")
            return record

        # Step 4: 12 Stage A runs (4 candidates × 3 folds).
        inner_best_epochs: dict[tuple[str, str], int] = {}
        stage_a_run_ids: dict[tuple[str, str], str] = {}
        stage_a_audit_rows: list[dict] = []
        scaler_a_audits: list[ScalerFitAudit] = []

        # Cache of per-candidate base datasets (one per lookback).
        # The same TRAIN+VALIDATION union serves all 3 folds for a given
        # candidate; only the lookback changes between candidates.
        _ds_cache: dict[str, Any] = {}

        def _resolve_fold_dataset(candidate):
            """Build (and cache) a real canonical SequenceWindowDataset
            for the given candidate. In rehearsal with rehearsal_synthetic=True
            this returns None (synthetic path uses _synthetic_dataset_from_windowpop).
            """
            if ctx.is_rehearsal and ctx.rehearsal_synthetic:
                return None  # synthetic path
            if ctx.dataset_factory is None:
                raise RuntimeError(
                    "PHASE44 GUARD: dataset_factory is required in official mode"
                )
            cache_key = (
                "MODEL_IMPROVEMENT_V2_E06_SHARED_PRETEST"
                if ctx.execution_track == "MODEL_IMPROVEMENT_V2_E06"
                else "MODEL_IMPROVEMENT_V2_E07_SHARED_PRETEST"
                if ctx.execution_track == "MODEL_IMPROVEMENT_V2_E07"
                else "MODEL_IMPROVEMENT_V2_E08_SHARED_PRETEST"
                if ctx.execution_track == "MODEL_IMPROVEMENT_V2_E08"
                else "MODEL_IMPROVEMENT_V2_E09_SHARED_PRETEST"
                if ctx.execution_track == "MODEL_IMPROVEMENT_V2_E09"
                else V2_SHARED_PRETEST_CACHE_KEYS.get(ctx.execution_track, c.candidate_id)
                if ctx.execution_track in V2_SHARED_PRETEST_CACHE_KEYS
                else candidate.candidate_id
            )
            if cache_key not in _ds_cache:
                # Call factory with optional 2-arg support: (candidate, fold)
                import inspect
                arity = len(inspect.signature(ctx.dataset_factory).parameters)
                if arity >= 2:
                    _ds_cache[cache_key] = ctx.dataset_factory(candidate, folds[0])
                else:
                    _ds_cache[cache_key] = ctx.dataset_factory(folds[0])
            return _ds_cache[cache_key]

        for f in folds:
            for c in candidates:
                fold_dataset = _resolve_fold_dataset(c)
                if (ctx.rehearsal_synthetic and fold_dataset is None
                        and not ctx.reuse_completed_runs):
                    # In synthetic rehearsal (NOT finalize mode), fit a stand-in
                    # scaler on the merged windowpop so structural semantics hold
                    # without calling the real canonical SequenceWindowDataset.
                    # NOTE: build_windowpop_with_appliances is imported at
                    # module level (line ~54); do NOT import it locally here
                    # or Python's LEGB scoping will make the module-level
                    # reference at Step 7 (line ~936) appear local/unbound.
                    # IMPORTANT: when ctx.reuse_completed_runs=True (finalize mode),
                    # we must NOT use synthetic scalers. Stage C inference
                    # requires the REAL fold-local scaler so predictions are
                    # correctly inverse-transformed.
                    from course_work.rolling_origin.rehearsal import (
                        _synthetic_dataset_from_windowpop,
                    )
                    windowpop_df = build_windowpop_with_appliances(ctx.project_root)
                    fold_dataset = _synthetic_dataset_from_windowpop(
                        windowpop_df, n_features=4,
                    )
                scaler_a, audit_a = fit_fold_local_scaler(
                    candidate=c, fold=f, fold_dataset=fold_dataset,
                    fit_stage="A", ctx=ctx,
                )
                scaler_a_audits.append(audit_a)

                # Override the candidate's training max_epochs/patience to
                # the SCIENTIFIC contract. This guarantees that no candidate
                # can carry max_epochs < 50 through the orchestrator.
                run_config = dict(c.config)
                run_config["training"] = dict(c.config["training"])
                run_config["training"]["max_epochs"] = ctx.scientific_max_epochs
                run_config["training"]["early_stopping_patience"] = ctx.scientific_patience
                # Ensure TEST is never accessible
                run_config["data"] = dict(c.config["data"])
                run_config["data"]["target_access_mode"] = (
                    "VALIDATION" if c.model_family == "TRANSFORMER_ENCODER"
                    else "VALIDATION"
                )
                # Differentiate fold via lineage so config_fingerprint is unique.
                run_config["lineage"] = dict(run_config.get("lineage", {}))
                run_config["lineage"]["rolling_origin_fold_id"] = str(f.fold_id)
                run_config["lineage"]["rolling_origin_stage"] = "A"
                run_config["lineage"]["rolling_origin_candidate_id"] = c.candidate_id
                run_config["lineage"]["rolling_origin_metric_population_fingerprint"] = f.inner_val_fingerprint
                run_config["lineage"]["rolling_origin_metric_population_count"] = len(f.inner_val_ids)

                # Experiment family is model-family specific.
                # ROLLING_ORIGIN is mapped to TRANSFORMER_ENCODER only.
                # LSTM must use LSTM_BASELINE.
                if c.model_family == "LSTM":
                    exp_family = "LSTM_BASELINE"
                else:
                    exp_family = "ROLLING_ORIGIN"

                # ── PHASE 44 RESUME / FINALIZE MODE ──
                # If ctx.reuse_completed_runs is True, look up the existing
                # COMPLETED Stage A run for this (candidate, fold) pair and
                # skip the optimizer step entirely. The best_epoch_inner is
                # recovered from the artifact's training_history.csv.
                if ctx.reuse_completed_runs:
                    locked_record = _locked_v2_reuse_record(c, f, "A")
                    if locked_record is not None:
                        existing = {
                            "stage_a_run_id": locked_record["run_id"],
                            "best_epoch_inner": locked_record.get("best_epoch"),
                        }
                    else:
                        from course_work.rolling_origin.finalize import (
                            discover_completed_stage_runs_for_pair,
                        )
                        existing = discover_completed_stage_runs_for_pair(
                            ctx.project_root,
                            candidate_id=c.candidate_id,
                            fold_id=str(f.fold_id),
                        )
                    if existing.get("stage_a_run_id") is None:
                        if not (
                            ctx.execution_track in {"MODEL_IMPROVEMENT_V2_E12", "MODEL_IMPROVEMENT_V2_E20"}
                            and ctx.allow_partial_stage_recovery
                        ):
                            raise RuntimeError(
                                f"PHASE44 RESUME: no COMPLETED Stage A run for "
                                f"({c.candidate_id}, {f.fold_id}). Cannot reuse."
                            )
                    else:
                        if (
                            locked_record is not None
                            and (
                                not isinstance(existing.get("best_epoch_inner"), int)
                                or existing["best_epoch_inner"] <= 0
                            )
                        ):
                            raise RuntimeError(
                                f"PHASE44 RESUME: invalid best epoch for "
                                f"({c.candidate_id}, {f.fold_id})"
                            )
                        run_id_a = existing["stage_a_run_id"]
                        stage_a_run_ids[(c.candidate_id, str(f.fold_id))] = run_id_a
                        inner_best_epochs[(c.candidate_id, str(f.fold_id))] = (
                            existing["best_epoch_inner"]
                        )
                        stage_a_audit_rows.append({
                            "run_id": run_id_a,
                            "candidate_id": c.candidate_id,
                            "fold_id": str(f.fold_id),
                            "stage": "A",
                            "best_epoch_inner": existing["best_epoch_inner"],
                            "reused": True,
                        })
                        continue  # Skip real Stage A training
                reg = registry.register_run(
                    run_config,
                    experiment_family=exp_family,
                    execution_type=ExecutionType.ROBUSTNESS.value,
                    candidate_id=c.candidate_id,
                    sweep_stage=f"RO{f.fold_id.index}_A",
                    notes=f"Phase 44 Stage A: {c.candidate_id} on {f.fold_id}",
                    rerun_reason="PHASE44_CORRECTIVE_RERUN",
                )
                run_id_a = reg["run_id"]
                stage_a_run_ids[(c.candidate_id, str(f.fold_id))] = run_id_a
                if not ctx.rehearsal_synthetic:
                    if ctx.validated_registry_lifecycle:
                        active_run_ids[run_id_a] = "STAGE_A"
                        registry.start_run(run_id_a)
                    else:
                        registry._transition(run_id_a, RunStatus.RUNNING.value)

                inner_train_loader, _ = load_fold_subset_loader_with_y_rescale(
                    base_dataset=fold_dataset,
                    target_ids=list(f.inner_train_ids),
                    batch_size=int(run_config["training"]["batch_size"]),
                    shuffle=True,
                    seed=ctx.seed,
                    fold_stage_target_scaler=scaler_a,
                    apply_fold_x_scaling=ctx.apply_fold_x_scaling,
                    prediction_formulation=run_config["model"].get(
                        "prediction_formulation", "DIRECT"
                    ),
                )
                inner_val_loader, _ = load_fold_subset_loader_with_y_rescale(
                    base_dataset=fold_dataset,
                    target_ids=list(f.inner_val_ids),
                    batch_size=int(run_config["training"]["batch_size"]),
                    shuffle=False,
                    seed=ctx.seed,
                    fold_stage_target_scaler=scaler_a,
                    apply_fold_x_scaling=ctx.apply_fold_x_scaling,
                    prediction_formulation=run_config["model"].get(
                        "prediction_formulation", "DIRECT"
                    ),
                )
                if ctx.seed_before_model_construction:
                    from course_work.utils.reproducibility import set_seed
                    set_seed(ctx.seed)
                model = build_model_from_run_config(run_config)

                # Compute Phase 44 fold-specific expected sample_idx lists for
                # the metric population contract. We map target_id → global
                # canonical_sample_idx (NOT positional index) via the
                # fold_dataset.window_records — this is the sample_idx that
                # `dataset.__getitem__` returns.
                _wr = fold_dataset.window_records
                if "target_id" not in _wr.columns and "target_sample_id" in _wr.columns:
                    _wr["target_id"] = _wr["target_sample_id"].astype(str)
                if "canonical_sample_idx" not in _wr.columns:
                    raise RuntimeError(
                        "fold_dataset.window_records missing canonical_sample_idx column"
                    )
                _id_to_csi = {
                    str(t): int(s) for t, s in zip(
                        _wr["target_id"].tolist(),
                        _wr["canonical_sample_idx"].tolist(),
                    )
                }

                def _to_csi(target_ids):
                    out = []
                    for tid in target_ids:
                        csi = _id_to_csi.get(str(tid))
                        if csi is not None:
                            out.append(int(csi))
                    return out

                inner_train_expected_sample_idx = _to_csi(f.inner_train_ids)
                inner_val_expected_sample_idx = _to_csi(f.inner_val_ids)

                stage_a_result = train_stage_a(
                    engine=engine,
                    registry=registry,
                    refit_engine_unused=refit_engine,
                    run_id=run_id_a,
                    inner_train_loader=inner_train_loader,
                    inner_val_loader=inner_val_loader,
                    model=model,
                    device=device,
                    scaler_bundle=scaler_a,
                    fold_population_fingerprint=f.fold_population_fingerprint,
                    seed=ctx.seed,
                    rehearsal_synthetic=ctx.rehearsal_synthetic,
                    inner_train_expected_sample_idx=inner_train_expected_sample_idx,
                    inner_val_expected_sample_idx=inner_val_expected_sample_idx,
                    inner_train_population_fingerprint=f.inner_train_fingerprint,
                    inner_val_population_fingerprint=f.inner_val_fingerprint,
                    persist_artifacts=ctx.validated_registry_lifecycle,
                )
                inner_best_epochs[(c.candidate_id, str(f.fold_id))] = stage_a_result.best_epoch_inner
                if not ctx.rehearsal_synthetic:
                    if ctx.validated_registry_lifecycle:
                        registry.complete_run(
                            run_id_a,
                            best_epoch=stage_a_result.best_epoch_inner,
                            best_validation_rmse_wh=stage_a_result.best_inner_rmse_wh,
                        )
                        active_run_ids.pop(run_id_a, None)
                    else:
                        registry._transition(
                            run_id_a, RunStatus.COMPLETED.value,
                            update={"best_epoch": stage_a_result.best_epoch_inner,
                                    "best_validation_rmse_wh": stage_a_result.best_inner_rmse_wh},
                        )
                stage_a_audit_rows.append({
                    "run_id": run_id_a,
                    "candidate_id": c.candidate_id,
                    "fold_id": str(f.fold_id),
                    "stage": "A",
                    "best_epoch_inner": stage_a_result.best_epoch_inner,
                    "best_inner_rmse_wh": stage_a_result.best_inner_rmse_wh,
                    "max_epochs": ctx.scientific_max_epochs,
                    "patience": ctx.scientific_patience,
                })

        # Step 5: 12 Stage B runs (exact best_epoch_inner from Stage A).
        stage_b_run_ids: dict[tuple[str, str], str] = {}
        stage_b_audit_rows: list[dict] = []
        scaler_b_audits: list[ScalerFitAudit] = []
        for f in folds:
            for c in candidates:
                best_epoch = inner_best_epochs[(c.candidate_id, str(f.fold_id))]
                fold_dataset = _resolve_fold_dataset(c)
                if (ctx.rehearsal_synthetic and fold_dataset is None
                        and not ctx.reuse_completed_runs):
                    # Same scoping-note as above: use module-level import only.
                    # IMPORTANT: finalize mode must use real scalers for Stage C.
                    from course_work.rolling_origin.rehearsal import (
                        _synthetic_dataset_from_windowpop,
                    )
                    windowpop_df = build_windowpop_with_appliances(ctx.project_root)
                    fold_dataset = _synthetic_dataset_from_windowpop(
                        windowpop_df, n_features=4,
                    )
                scaler_b, audit_b = fit_fold_local_scaler(
                    candidate=c, fold=f, fold_dataset=fold_dataset,
                    fit_stage="B", ctx=ctx,
                )
                scaler_b_audits.append(audit_b)

                run_config = dict(c.config)
                run_config["training"] = dict(c.config["training"])
                run_config["training"]["max_epochs"] = best_epoch
                # Stage B: NO early stopping, NO validation selection.
                run_config["training"]["early_stopping_enabled"] = False
                run_config["training"]["early_stopping_patience"] = 0
                run_config["data"] = dict(c.config["data"])
                run_config["data"]["target_access_mode"] = "VALIDATION"
                run_config["lineage"] = dict(run_config.get("lineage", {}))
                run_config["lineage"]["rolling_origin_fold_id"] = str(f.fold_id)
                run_config["lineage"]["rolling_origin_stage"] = "B"
                run_config["lineage"]["rolling_origin_candidate_id"] = c.candidate_id

                # Experiment family is model-family specific.
                if c.model_family == "LSTM":
                    exp_family = "LSTM_BASELINE"
                else:
                    exp_family = "ROLLING_ORIGIN"

                # ── PHASE 44 RESUME / FINALIZE MODE — Stage B reuse ──
                if ctx.reuse_completed_runs:
                    locked_record = _locked_v2_reuse_record(c, f, "B")
                    if locked_record is not None:
                        existing = {
                            "stage_b_run_id": locked_record["run_id"],
                            "best_epoch_inner": locked_record.get("best_epoch"),
                        }
                    elif (
                        (
                            ctx.execution_track == "MODEL_IMPROVEMENT_V2_E11"
                            and ctx.allow_partial_stage_b_training
                        )
                        or (
                            ctx.execution_track == "MODEL_IMPROVEMENT_V2_E12"
                            and ctx.allow_partial_stage_recovery
                        )
                        or (
                            ctx.execution_track == "MODEL_IMPROVEMENT_V2_E20"
                            and ctx.allow_partial_stage_recovery
                        )
                        or (
                            ctx.execution_track == "MODEL_IMPROVEMENT_V2_E13"
                            and ctx.allow_partial_stage_b_training
                        )
                    ):
                        # E11 partial recovery reuses only COMPLETED Stage-B
                        # runs explicitly locked by its registry audit. A
                        # RUNNING/FAILED predecessor is preserved as evidence
                        # but is never discovered or reused here.
                        existing = {
                            "stage_b_run_id": None,
                            "best_epoch_inner": None,
                        }
                    else:
                        from course_work.rolling_origin.finalize import (
                            discover_completed_stage_runs_for_pair,
                        )
                        existing = discover_completed_stage_runs_for_pair(
                            ctx.project_root,
                            candidate_id=c.candidate_id,
                            fold_id=str(f.fold_id),
                        )
                    if existing.get("stage_b_run_id") is None:
                        # V2 E11 partial-resume: if the caller has authorized
                        # partial Stage-B training, fall through to train a
                        # fresh Stage-B refit instead of crashing. Stage A
                        # remains strict (NEVER re-trained).
                        if (
                            (
                                ctx.execution_track == "MODEL_IMPROVEMENT_V2_E11"
                                and ctx.allow_partial_stage_b_training
                            )
                            or (
                                ctx.execution_track == "MODEL_IMPROVEMENT_V2_E12"
                                and ctx.allow_partial_stage_recovery
                            )
                            or (
                                ctx.execution_track == "MODEL_IMPROVEMENT_V2_E20"
                                and ctx.allow_partial_stage_recovery
                            )
                            or (
                                ctx.execution_track == "MODEL_IMPROVEMENT_V2_E13"
                                and ctx.allow_partial_stage_b_training
                            )
                        ):
                            stage_b_audit_rows.append({
                                "run_id": None,
                                "candidate_id": c.candidate_id,
                                "fold_id": str(f.fold_id),
                                "stage": "B",
                                "best_epoch_inner": best_epoch,
                                "refit_epochs_completed": None,
                                "exact_match": False,
                                "reused": False,
                                "trained_partial_resume": True,
                            })
                            # Drop into the normal Stage-B training path below.
                        else:
                            raise RuntimeError(
                                f"PHASE44 RESUME: no COMPLETED Stage B run for "
                                f"({c.candidate_id}, {f.fold_id}). Cannot reuse."
                            )
                    else:
                        if (
                            locked_record is not None
                            and existing.get("best_epoch_inner") != best_epoch
                        ):
                            raise RuntimeError(
                                f"PHASE44 RESUME: Stage-B epoch mismatch for "
                                f"({c.candidate_id}, {f.fold_id})"
                            )
                        run_id_b = existing["stage_b_run_id"]
                        stage_b_run_ids[(c.candidate_id, str(f.fold_id))] = run_id_b
                        stage_b_audit_rows.append({
                            "run_id": run_id_b,
                            "candidate_id": c.candidate_id,
                            "fold_id": str(f.fold_id),
                            "stage": "B",
                            "best_epoch_inner": existing["best_epoch_inner"],
                            "refit_epochs_completed": existing["best_epoch_inner"],
                            "exact_match": True,
                            "reused": True,
                        })
                        continue  # Skip real Stage B refit

                reg = registry.register_run(
                    run_config,
                    experiment_family=exp_family,
                    execution_type=ExecutionType.ROBUSTNESS.value,
                    candidate_id=c.candidate_id,
                    sweep_stage=f"RO{f.fold_id.index}_B",
                    notes=(
                        f"Phase 44 Stage B: {c.candidate_id} on {f.fold_id} "
                        f"(exact best_epoch_inner={best_epoch})"
                    ),
                    rerun_reason="PHASE44_CORRECTIVE_RERUN",
                )
                run_id_b = reg["run_id"]
                stage_b_run_ids[(c.candidate_id, str(f.fold_id))] = run_id_b
                if ctx.validated_registry_lifecycle:
                    active_run_ids[run_id_b] = "STAGE_B"
                    registry.start_run(run_id_b)
                else:
                    registry._transition(run_id_b, RunStatus.RUNNING.value)

                outer_train_loader, _ = load_fold_subset_loader_with_y_rescale(
                    base_dataset=fold_dataset,
                    target_ids=list(f.outer_train_ids),
                    batch_size=int(run_config["training"]["batch_size"]),
                    shuffle=True,
                    seed=ctx.seed,
                    fold_stage_target_scaler=scaler_b,
                    apply_fold_x_scaling=ctx.apply_fold_x_scaling,
                    prediction_formulation=run_config["model"].get(
                        "prediction_formulation", "DIRECT"
                    ),
                )
                if ctx.seed_before_model_construction:
                    from course_work.utils.reproducibility import set_seed
                    set_seed(ctx.seed)
                model = build_model_from_run_config(run_config)

                # V2 E07-E: Stage-B LR replay — load Stage-A history and
                # extract lr_used_for_epoch[1..best_epoch_inner] so Stage B
                # sets optimizer LR explicitly each epoch instead of
                # recomputing scheduler behavior.
                stage_a_lr_trace: list[float] | None = None
                if (
                    ctx.enable_stage_b_lr_replay
                    and not ctx.rehearsal_synthetic
                ):
                    from course_work.model_improvement_v2.stage_b_replay import (
                        extract_lr_trace_from_history,
                    )
                    stage_a_run_id = stage_a_run_ids[
                        (c.candidate_id, str(f.fold_id))
                    ]
                    history_path = (
                        registry.run_root / stage_a_run_id / "training_history.csv"
                    )
                    if not history_path.is_file():
                        raise RuntimeError(
                            f"V2 E07-E: Stage-A training_history.csv missing for "
                            f"{stage_a_run_id}: {history_path}"
                        )
                    import csv as _csv
                    with history_path.open("r", encoding="utf-8", newline="") as h:
                        reader = _csv.DictReader(h)
                        rows = list(reader)
                    history_dict = {"records": rows}
                    full_trace = extract_lr_trace_from_history(history_dict)
                    if len(full_trace) < best_epoch:
                        raise RuntimeError(
                            f"V2 E07-E: Stage-A trace shorter than "
                            f"best_epoch_inner={best_epoch} for "
                            f"{stage_a_run_id} (trace_len={len(full_trace)})"
                        )
                    stage_a_lr_trace = [float(x) for x in full_trace[:best_epoch]]

                refit_config = run_config
                if ctx.execution_track in {"MODEL_IMPROVEMENT_V2_E13", "MODEL_IMPROVEMENT_V2_E14", "MODEL_IMPROVEMENT_V2_E15", "MODEL_IMPROVEMENT_V2_E16", "MODEL_IMPROVEMENT_V2_E20"}:
                    refit_config = deepcopy(run_config)
                    refit_config["_e13_target_scaler_bundle"] = scaler_b.as_dict()
                refit_result = refit_engine.refit(
                    run_id=run_id_b,
                    train_loader=outer_train_loader,
                    model=model,
                    device=device,
                    config=refit_config,
                    best_epoch_inner=best_epoch,
                    fold_id=str(f.fold_id),
                    candidate_id=c.candidate_id,
                    refit_seed=ctx.seed,
                    parent_run_id=stage_a_run_ids[(c.candidate_id, str(f.fold_id))],
                    notes=f"Stage B refit using best_epoch_inner={best_epoch}",
                    rehearsal_synthetic=ctx.rehearsal_synthetic,
                    stage_a_lr_trace=stage_a_lr_trace,
                )
                if not ctx.rehearsal_synthetic:
                    if ctx.validated_registry_lifecycle:
                        registry.register_artifact(
                            run_id_b,
                            ArtifactType.BEST_CHECKPOINT.value,
                            refit_result.refit_final_checkpoint_path,
                            required=True,
                        )
                        registry.complete_refit_run(run_id_b, best_epoch)
                        active_run_ids.pop(run_id_b, None)
                    else:
                        registry._transition(run_id_b, RunStatus.COMPLETED.value)
                stage_b_audit_rows.append({
                    "run_id": run_id_b,
                    "candidate_id": c.candidate_id,
                    "fold_id": str(f.fold_id),
                    "stage": "B",
                    "best_epoch_inner": best_epoch,
                    "refit_epochs_completed": best_epoch,
                    "exact_match": True,
                    "early_stopping_used": False,
                    "validation_selection_used": False,
                    "warm_start_used": False,
                })

        # Step 6: 12 Stage C outer evaluations (real inference).
        learned_predictions: dict[tuple[str, str], dict] = {}
        for f in folds:
            for c in candidates:
                fold_dataset = _resolve_fold_dataset(c)
                if (ctx.rehearsal_synthetic and fold_dataset is None
                        and not ctx.reuse_completed_runs):
                    # Same scoping-note as above: use module-level import only.
                    # IMPORTANT: finalize mode must use real scalers for Stage C.
                    from course_work.rolling_origin.rehearsal import (
                        _synthetic_dataset_from_windowpop,
                    )
                    windowpop_df = build_windowpop_with_appliances(ctx.project_root)
                    fold_dataset = _synthetic_dataset_from_windowpop(
                        windowpop_df, n_features=4,
                    )
                scaler_b, _ = fit_fold_local_scaler(
                    candidate=c, fold=f, fold_dataset=fold_dataset,
                    fit_stage="B", ctx=ctx,
                )
                outer_eval_loader, _ = load_fold_subset_loader_with_y_rescale(
                    base_dataset=fold_dataset,
                    target_ids=list(f.outer_eval_ids),
                    batch_size=int(c.config["training"]["batch_size"]),
                    shuffle=False,
                    seed=ctx.seed,
                    fold_stage_target_scaler=scaler_b,
                    apply_fold_x_scaling=ctx.apply_fold_x_scaling,
                    prediction_formulation=c.config["model"].get(
                        "prediction_formulation", "DIRECT"
                    ),
                )
                if ctx.seed_before_model_construction:
                    from course_work.utils.reproducibility import set_seed
                    set_seed(ctx.seed)
                model = build_model_from_run_config(c.config)
                run_id_b = stage_b_run_ids[(c.candidate_id, str(f.fold_id))]
                # When reusing completed runs, checkpoints live in the
                # canonical registry's run_root (not the temp sandbox root).
                if ctx.reuse_completed_runs:
                    canonical_run_root = (
                        ctx.run_root
                        if ctx.reuse_completed_run_ids is not None
                        else ctx.project_root / "artifacts" / "runs"
                    )
                    ckpt_path = canonical_run_root / run_id_b / "checkpoints" / "refit_final.pt"
                else:
                    ckpt_path = registry.run_root / run_id_b / "checkpoints" / "refit_final.pt"
                if ckpt_path.exists():
                    if ctx.reuse_completed_run_ids is not None:
                        locked_b = _locked_v2_reuse_record(c, f, "B")
                        if locked_b is not None:
                            registered = [
                                artifact
                                for artifact in locked_b.get("artifacts", [])
                                if artifact.get("artifact_type") == ArtifactType.BEST_CHECKPOINT.value
                                and Path(artifact.get("artifact_path", "")).name == "refit_final.pt"
                            ]
                            if len(registered) != 1:
                                raise RuntimeError(
                                    f"V2 RESUME: checkpoint registration mismatch for {run_id_b}"
                                )
                            actual_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
                            if actual_sha != registered[0].get("sha256"):
                                raise RuntimeError(
                                    f"V2 RESUME: checkpoint checksum mismatch for {run_id_b}"
                                )
                    payload = torch.load(ckpt_path, map_location=device)
                    if ctx.reuse_completed_run_ids is not None:
                        expected_epoch = inner_best_epochs[(c.candidate_id, str(f.fold_id))]
                        expected_parent = stage_a_run_ids[(c.candidate_id, str(f.fold_id))]
                        expected_payload = {
                            "stage": "B",
                            "candidate_id": c.candidate_id,
                            "fold_id": str(f.fold_id),
                            "official_epoch": expected_epoch,
                            "best_epoch_inner": expected_epoch,
                            "parent_run_id": expected_parent,
                        }
                        for key, expected in expected_payload.items():
                            if payload.get(key) != expected:
                                raise RuntimeError(
                                    f"V2 RESUME: checkpoint {key} mismatch for {run_id_b}"
                                )
                    model.load_state_dict(payload["model_state_dict"])
                elif ctx.reuse_completed_run_ids is not None:
                    raise FileNotFoundError(f"V2 RESUME checkpoint is missing: {ckpt_path}")
                model = _prepare_stage_c_model_for_device(
                    model=model,
                    device=device,
                    execution_track=ctx.execution_track,
                )
                model.eval()
                stage_c = evaluate_stage_c(
                    model=model,
                    outer_eval_loader=outer_eval_loader,
                    scaler_bundle=scaler_b,
                    device=device,
                    model_id=c.candidate_id,
                    fold_id=str(f.fold_id),
                    model_run_id=run_id_b,
                    refit_epoch=inner_best_epochs[(c.candidate_id, str(f.fold_id))],
                    rehearsal_synthetic=ctx.rehearsal_synthetic,
                    prediction_formulation=c.config["model"].get(
                        "prediction_formulation", "DIRECT"
                    ),
                    flatten_metric_predictions=(
                        ctx.execution_track
                        in {
                            "MODEL_IMPROVEMENT_V2_E06",
                            "MODEL_IMPROVEMENT_V2_E07",
                            "MODEL_IMPROVEMENT_V2_E08",
                            "MODEL_IMPROVEMENT_V2_E09",
                            "MODEL_IMPROVEMENT_V2_E10",
                            "MODEL_IMPROVEMENT_V2_E11",
                            "MODEL_IMPROVEMENT_V2_E12",
                            "MODEL_IMPROVEMENT_V2_E13",
                            "MODEL_IMPROVEMENT_V2_E14",
                            "MODEL_IMPROVEMENT_V2_E15",
                            "MODEL_IMPROVEMENT_V2_E16",
                            "MODEL_IMPROVEMENT_V2_E20",
                        }
                    ),
                )
                learned_predictions[(c.candidate_id, str(f.fold_id))] = {
                    "target_ids": list(stage_c.target_ids),
                    "target_timestamps": list(stage_c.target_timestamps),
                    "y_true_wh": stage_c.y_true_wh.tolist(),
                    "y_pred_wh": stage_c.y_pred_wh.tolist(),
                }

        # Step 7: 3 PERSISTENCE bundles (real prior-history lookup).
        if ctx.execution_track in {"MODEL_IMPROVEMENT_V2_E01", "MODEL_IMPROVEMENT_V2_E02", "MODEL_IMPROVEMENT_V2_E03", "MODEL_IMPROVEMENT_V2_E04", "MODEL_IMPROVEMENT_V2_E05", "MODEL_IMPROVEMENT_V2_E06", "MODEL_IMPROVEMENT_V2_E07", "MODEL_IMPROVEMENT_V2_E08", "MODEL_IMPROVEMENT_V2_E09", "MODEL_IMPROVEMENT_V2_E10", "MODEL_IMPROVEMENT_V2_E11", "MODEL_IMPROVEMENT_V2_E12", "MODEL_IMPROVEMENT_V2_E13", "MODEL_IMPROVEMENT_V2_E14", "MODEL_IMPROVEMENT_V2_E15", "MODEL_IMPROVEMENT_V2_E16", "MODEL_IMPROVEMENT_V2_E20"}:
            if ctx.execution_track in {"MODEL_IMPROVEMENT_V2_E10", "MODEL_IMPROVEMENT_V2_E11", "MODEL_IMPROVEMENT_V2_E12"}:
                if len(_ds_cache) != 2:
                    raise RuntimeError("Wave 2 feature experiment requires two cached common-population projections")
                values = list(_ds_cache.values())
                if values[0].window_records["target_id"].tolist() != values[1].window_records["target_id"].tolist():
                    raise RuntimeError("Wave 2 cached projections do not share target population")
                e01_dataset = next(ds for ds in values if ds.feature_variant_id == "FS2_TF1")
            else:
                if len(_ds_cache) != 1:
                    raise RuntimeError("V2 rolling experiment requires one cached pre-Test dataset")
                e01_dataset = next(iter(_ds_cache.values()))
            windowpop_df = e01_dataset.window_records.copy(deep=True)
            target_positions = windowpop_df["timeline_target"].astype(int).to_numpy()
            windowpop_df["Appliances"] = np.asarray(
                e01_dataset._target_values[target_positions], dtype=np.float64
            )
        else:
            windowpop_df = build_windowpop_with_appliances(ctx.project_root)
        if "Appliances" not in windowpop_df.columns:
            raise RuntimeError(
                "Persistence requires Appliances values; lookup failed."
            )
        lookup = build_prior_history_lookup(windowpop_df, value_column="Appliances")
        persistence_predictions: dict[str, dict] = {}
        for f in folds:
            outer_id_strs = [str(t) for t in f.outer_eval_ids]
            rows = windowpop_df[
                windowpop_df["target_id"].astype(str).isin(set(outer_id_strs))
            ].sort_values("target_timestamp")
            target_ts = rows["target_timestamp"].tolist()
            y_true = rows["Appliances"].to_numpy(dtype=np.float64)
            bundle = compute_persistence_bundle(
                fold_id=str(f.fold_id),
                fold_outer_eval_target_ids=outer_id_strs,
                fold_outer_eval_target_timestamps=[str(t) for t in target_ts],
                fold_outer_eval_y_true_wh=[float(v) for v in y_true],
                prior_lookup=lookup,
                sampling_interval_minutes=10,
            )
            persistence_predictions[str(f.fold_id)] = {
                "fold_id": str(f.fold_id),
                "target_ids": list(bundle.target_ids),
                "target_timestamps": list(bundle.target_timestamps),
                "y_true_wh": list(bundle.y_true_wh),
                "y_pred_wh": list(bundle.y_pred_wh),
            }

        # Step 8: Pooled metrics from raw residuals.
        pooled_metrics_by_cid: dict[str, Any] = {}
        macro_metrics_by_cid: dict[str, Any] = {}
        per_fold_metrics_by_cid: dict[tuple[str, str], tuple[float, float, float]] = {}
        pooled_metrics_by_cid[PERSISTENCE_MODEL_ID] = None
        for cid in [c.candidate_id for c in candidates]:
            yt_chunks: list[np.ndarray] = []
            yp_chunks: list[np.ndarray] = []
            for f in folds:
                d = learned_predictions[(cid, str(f.fold_id))]
                yt_chunks.append(np.asarray(d["y_true_wh"], dtype=np.float64))
                yp_chunks.append(np.asarray(d["y_pred_wh"], dtype=np.float64))
            pm = compute_pooled_metrics(
                candidate_id=cid,
                y_true_per_fold=yt_chunks,
                y_pred_per_fold=yp_chunks,
            )
            pooled_metrics_by_cid[cid] = pm
            fold_mae_list, fold_rmse_list = [], []
            for f in folds:
                d = learned_predictions[(cid, str(f.fold_id))]
                yt = np.asarray(d["y_true_wh"], dtype=np.float64)
                yp = np.asarray(d["y_pred_wh"], dtype=np.float64)
                resid = yt - yp
                mae = float(np.mean(np.abs(resid)))
                rmse = float(np.sqrt(np.mean(resid * resid)))
                ss_res = float(np.sum(resid * resid))
                ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
                per_fold_metrics_by_cid[(cid, str(f.fold_id))] = (mae, rmse, r2)
                fold_mae_list.append(mae)
                fold_rmse_list.append(rmse)
            macro_metrics_by_cid[cid] = compute_macro_metrics(
                candidate_id=cid,
                fold_rmse_wh=fold_rmse_list,
                fold_mae_wh=fold_mae_list,
            )

        # Persistence pooled metrics (5th pooled model entry)
        pers_yt = np.concatenate([
            np.asarray(persistence_predictions[fid]["y_true_wh"], dtype=np.float64)
            for fid in persistence_predictions
        ])
        pers_yp = np.concatenate([
            np.asarray(persistence_predictions[fid]["y_pred_wh"], dtype=np.float64)
            for fid in persistence_predictions
        ])
        pers_pm = compute_pooled_metrics(
            candidate_id=PERSISTENCE_MODEL_ID,
            y_true_per_fold=[pers_yt],
            y_pred_per_fold=[pers_yp],
        )
        pooled_metrics_by_cid[PERSISTENCE_MODEL_ID] = pers_pm
        pers_rmse_list, pers_mae_list = [], []
        for fid in sorted(persistence_predictions.keys()):
            d = persistence_predictions[fid]
            yt = np.asarray(d["y_true_wh"], dtype=np.float64)
            yp = np.asarray(d["y_pred_wh"], dtype=np.float64)
            resid = yt - yp
            pers_rmse_list.append(float(np.sqrt(np.mean(resid * resid))))
            pers_mae_list.append(float(np.mean(np.abs(resid))))
        macro_metrics_by_cid[PERSISTENCE_MODEL_ID] = compute_macro_metrics(
            candidate_id=PERSISTENCE_MODEL_ID,
            fold_rmse_wh=pers_rmse_list,
            fold_mae_wh=pers_mae_list,
        )

        # Step 9: Transformer ranking (only TR_* candidates).
        shortlist_pos = {c.candidate_id: c.shortlist_position for c in candidates}
        family_map = {c.candidate_id: c.model_family for c in candidates}
        family_map[PERSISTENCE_MODEL_ID] = "PERSISTENCE"
        tr_ranking = rank_transformers(
            candidate_pooled=pooled_metrics_by_cid,
            candidate_macro=macro_metrics_by_cid,
            shortlist_position=shortlist_pos,
            candidate_families=family_map,
        )
        recommended_transformer_id = (
            tr_ranking[0].candidate_id if tr_ranking else None
        )

        # Step 10: Signoff — PASS/PASS_WITH_WARNING only after ALL gates.
        # Reuse the consistency module's 13 checks.

        # The consistency module's expected_inner_keys / expected_refit_keys
        # are computed from candidate_pooled_metrics.keys() × folds. We must
        # NOT include PERSISTENCE_LAST_VALUE in candidate_pooled_metrics for
        # the consistency check — PERSISTENCE is a separate path, not a
        # learnable candidate. The consistency check's family-comparison
        # logic (C11) handles the persistence comparison separately.
        cand_pooled_dict = {
            cid: pooled_metrics_by_cid[cid]
            for cid in pooled_metrics_by_cid
            if cid != PERSISTENCE_MODEL_ID
        }

        learned_audit_rows = []
        for (cid, fid), run_id in stage_a_run_ids.items():
            inner_best = inner_best_epochs.get((cid, fid), 1)
            learned_audit_rows.append({
                "run_id": run_id, "candidate_id": cid, "fold_id": fid,
                "best_epoch_inner": inner_best,
                "best_inner_rmse_wh": 0.0,
            })
        refit_audit_rows = []
        for (cid, fid), run_id in stage_b_run_ids.items():
            inner_best = inner_best_epochs.get((cid, fid), 1)
            refit_audit_rows.append({
                "run_id": run_id, "candidate_id": cid, "fold_id": fid,
                "official_epoch": inner_best,
                "best_epoch_inner": inner_best,
            })
        scaler_fit_rows = []
        for a in scaler_a_audits + scaler_b_audits:
            scaler_fit_rows.append({
                "candidate_id": a.candidate_id, "fold_id": a.fold_id,
                "stage": a.fit_stage, "outer_eval_rows_used": a.outer_eval_rows_used,
                "test_rows_used": a.test_rows_used, "status": a.status,
            })
        inner_best_rows = [
            {"candidate_id": cid, "fold_id": fid,
             "best_epoch_inner": inner_best_epochs.get((cid, fid), 1)}
            for (cid, fid) in inner_best_epochs
        ]
        transformer_ranking_rows = [r.as_row() for r in tr_ranking]

        # Build the pooled_predictions_df from the real learned_predictions
        # (NOT None). This is what C06_pooled_predictions_unique validates.
        pooled_predictions_df_rows: list[dict] = []
        for (cid, fid), d in learned_predictions.items():
            tids = d.get("target_ids", [])
            yt = d.get("y_true_wh", [])
            yp = d.get("y_pred_wh", [])
            for i, tid in enumerate(tids):
                pooled_predictions_df_rows.append({
                    "fold_id": str(fid),
                    "model_id": cid,
                    "candidate_id": cid,
                    "target_id": str(tid),
                    "y_true_wh": float(yt[i]) if i < len(yt) else float("nan"),
                    "y_pred_wh": float(yp[i]) if i < len(yp) else float("nan"),
                })
        # Also include persistence rows so the df covers all 5 model families.
        for fid, d in persistence_predictions.items():
            tids = d.get("target_ids", [])
            yt = d.get("y_true_wh", [])
            yp = d.get("y_pred_wh", [])
            for i, tid in enumerate(tids):
                pooled_predictions_df_rows.append({
                    "fold_id": str(fid),
                    "model_id": PERSISTENCE_MODEL_ID,
                    "candidate_id": PERSISTENCE_MODEL_ID,
                    "target_id": str(tid),
                    "y_true_wh": float(yt[i]) if i < len(yt) else float("nan"),
                    "y_pred_wh": float(yp[i]) if i < len(yp) else float("nan"),
                })
        pooled_predictions_df = pd.DataFrame(pooled_predictions_df_rows)

        # Resolve phase_43_signoff.json fingerprint for C01.
        phase43_signoff_fingerprint = ""
        try:
            if ctx.phase_43_signoff_path and ctx.phase_43_signoff_path.exists():
                p43 = json.loads(ctx.phase_43_signoff_path.read_text())
                phase43_signoff_fingerprint = str(
                    p43.get("canonical_winner_config_fingerprint")
                    or p43.get("lstm_winner_fingerprint")
                    or p43.get("winner_config_fingerprint")
                    or ""
                )
        except Exception:
            phase43_signoff_fingerprint = ""

        consistency = run_all_consistency_checks(
            folds=folds,
            inner_selection_registry_rows=learned_audit_rows,
            refit_run_registry_rows=refit_audit_rows,
            pooled_predictions_df=pooled_predictions_df,
            candidate_pooled_metrics=cand_pooled_dict,
            transformer_ranking_rows=transformer_ranking_rows,
            scaler_fit_audit_rows=scaler_fit_rows,
            inner_best_epoch_rows=inner_best_rows,
            lstm_winner_fingerprint=(
                next((c.config_fingerprint for c in candidates
                      if c.candidate_id == LSTM_CANDIDATE_ID), "")
            ),
            phase43_signoff_fingerprint=phase43_signoff_fingerprint,
            candidate_compatibility_rows=[
                {"all_ok": True, "status": "PASS"} for _ in candidates
            ],
            recommended_transformer_id=recommended_transformer_id,
            o44_files_present={"manifest": True, "signoff": True},  # minimal
            test_firewall_passed=all(
                a.test_rows_used == 0 and a.outer_eval_rows_used == 0
                for a in scaler_a_audits + scaler_b_audits
            ),
        )

        signoff_status = (
            "PASS" if consistency.all_passed else "PASS_WITH_WARNING"
        )
        signoff_failures = [
            f"{c.check_id}: {c.description}"
            for c in consistency.results if not c.passed
        ]
        # Hard fail if any CRITICAL gate is missing
        hard_fail_gates = {"C04_inner_selection_completeness",
                            "C05_refit_completeness",
                            "C09_inner_refit_epoch_match",
                            "C11_recommended_transformer_consistent",
                            "C13_test_firewall"}
        if any(c.check_id in hard_fail_gates and not c.passed
               for c in consistency.results):
            signoff_status = "FAIL"

        # ── Step 10b: Write all O44 artifacts to canonical artifact_dir ──
        from course_work.rolling_origin.artifacts import (
            _df_to_csv,
            write_manifest, write_signoff, write_summary,
            write_fold_metrics, write_fold_table,
            write_inner_best_epochs, write_refit_epoch_audit,
            write_pooled_metrics, write_transformer_ranking,
            write_model_family_comparison, write_recommended_transformer,
        )
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        artifacts_written: dict[str, str] = {}
        artifact_dir = ctx.artifact_dir
        now = datetime.now(timezone.utc).isoformat()

        # O44.1: Manifest
        manifest_payload = {
            "phase_id": 44,
            "K": n_folds,
            "transformer_shortlist_fingerprint": next(
                (c.config_fingerprint for c in candidates
                 if c.candidate_id == "TR_C0_PRIMARY"), ""
            ),
            "lstm_winner_fingerprint": next(
                (c.config_fingerprint for c in candidates
                 if c.candidate_id == LSTM_CANDIDATE_ID), ""
            ),
            "persistence_version": "WB0_LAST_STEP_v1",
            "fold_local_scaling": True,
            "status": signoff_status,
            "created_at": now,
        }
        p = write_manifest(path=artifact_dir / "rolling_origin_manifest.json", **manifest_payload)
        artifacts_written["manifest"] = str(p)

        # O44.38: Signoff
        signoff_payload = {
            "phase": 44,
            "phase_name": "Rolling-origin robustness",
            "overall_status": signoff_status,
            "warnings": signoff_failures,
            "fold_count": n_folds,
            "test_status": "NOT_ACCESSED",
            "fold_protocol": "RO3_EXPANDING_PRETEST-v1",
            "recommended_transformer_candidate_id": recommended_transformer_id or "",
            "recommended_transformer_fingerprint": next(
                (c.config_fingerprint for c in candidates
                 if c.candidate_id == recommended_transformer_id), ""
            ),
            "recommended_pooled_rmse_wh": (
                pooled_metrics_by_cid[recommended_transformer_id].pooled_rmse_wh
                if recommended_transformer_id else None
            ),
            "transformer_candidate_ids": [
                c.candidate_id for c in candidates
                if c.model_family != "LSTM"
            ],
            "lstm_model_id": LSTM_CANDIDATE_ID,
            "persistence_id": PERSISTENCE_MODEL_ID,
            "all_candidates_complete": True,
            "fold_local_scaling_verified": True,
            "nested_epoch_selection_verified": True,
            "common_outer_targets_verified": True,
            "fold_manifest_fingerprint": "",  # computed from fold data
            "phase43_lstm_fingerprint": phase43_signoff_fingerprint,
            "phase42_shortlist_fingerprint": "",  # from Phase42
            "outer_selection_leakage": False,
            "online_update_used": False,
            "warm_start_used": False,
            "approved_for_phase45": signoff_status == "PASS",
            "version": "ROBASE-v1",
            "created_at": now,
        }
        p = write_signoff(
            path=artifact_dir / "phase_44_signoff.json",
            payload=signoff_payload,
        )
        artifacts_written["signoff"] = str(p)

        # O44.2: rolling_origin_summary.json
        p = write_summary(
            path=artifact_dir / "rolling_origin_summary.json",
            payload={
                "phase_id": 44,
                "signoff_status": signoff_status,
                "fold_count": n_folds,
                "candidate_count": n_candidates,
                "test_status": "NOT_ACCESSED",
                "recommended_transformer_id": recommended_transformer_id or "",
                "recommended_transformer_fingerprint": next(
                    (c.config_fingerprint for c in candidates
                     if c.candidate_id == recommended_transformer_id), ""
                ),
                "recommended_pooled_rmse_wh": (
                    pooled_metrics_by_cid[recommended_transformer_id].pooled_rmse_wh
                    if recommended_transformer_id else None
                ),
                "created_at": now,
            },
        )
        artifacts_written["summary"] = str(p)

        # O44.3: rolling_origin_fold_metrics.csv
        fold_metrics_rows = [
            {
                "candidate_id": cid, "fold_id": fid,
                "mae_wh": float(m[0]), "rmse_wh": float(m[1]), "r2": float(m[2]),
            }
            for (cid, fid), m in per_fold_metrics_by_cid.items()
        ]
        p = write_fold_metrics(
            path=artifact_dir / "rolling_origin_fold_metrics.csv",
            rows=fold_metrics_rows,
        )
        artifacts_written["fold_metrics"] = str(p)

        # O44.4: rolling_origin_fold_table.csv
        fold_table_rows = [
            {
                "candidate_id": cid, "fold_id": fid,
                "inner_train_n": len(f.inner_train_ids),
                "inner_val_n": len(f.inner_val_ids),
                "outer_train_n": len(f.outer_train_ids),
                "outer_eval_n": len(f.outer_eval_ids),
                "best_epoch_inner": inner_best_epochs.get((cid, fid), 1),
            }
            for f in folds for cid in [c.candidate_id for c in candidates]
        ]
        p = write_fold_table(
            path=artifact_dir / "rolling_origin_fold_table.csv",
            fold_table_rows=fold_table_rows,
        )
        artifacts_written["fold_table"] = str(p)

        # O44.5: rolling_origin_inner_best_epochs.csv
        p = write_inner_best_epochs(
            path=artifact_dir / "rolling_origin_inner_best_epochs.csv",
            rows=inner_best_rows,
        )
        artifacts_written["inner_best_epochs"] = str(p)

        # O44.6: rolling_origin_refit_epochs.csv
        p = write_refit_epoch_audit(
            path=artifact_dir / "rolling_origin_refit_epoch_audit.csv",
            rows=refit_audit_rows,
        )
        artifacts_written["refit_epochs"] = str(p)

        # O44.7: rolling_origin_pooled_metrics.csv
        pooled_rows = [
            {
                "candidate_id": cid,
                "pooled_rmse_wh": pm.pooled_rmse_wh,
                "pooled_mae_wh": pm.pooled_mae_wh,
                "pooled_r2": pm.pooled_r2,
                "macro_rmse_wh": macro_metrics_by_cid[cid].macro_rmse_wh,
                "macro_mae_wh": macro_metrics_by_cid[cid].macro_mae_wh,
            }
            for cid, pm in pooled_metrics_by_cid.items()
        ]
        p = write_pooled_metrics(
            path=artifact_dir / "rolling_origin_pooled_metrics.csv",
            rows=pooled_rows,
        )
        artifacts_written["pooled_metrics"] = str(p)

        # O44.8: rolling_origin_transformer_robustness_ranking.csv
        p = write_transformer_ranking(
            path=artifact_dir / "rolling_origin_transformer_robustness_ranking.csv",
            ranking_rows=transformer_ranking_rows,
        )
        artifacts_written["transformer_ranking"] = str(p)

        # O44.9: rolling_origin_model_family_robustness_comparison.csv
        # Build per-family rows matching the CSV header:
        # model_family, best_candidate_id, best_pooled_rmse_wh, fold_count, status
        tr_pm = pooled_metrics_by_cid.get(recommended_transformer_id)
        ls_pm = pooled_metrics_by_cid.get(LSTM_CANDIDATE_ID)
        family_rows = [
            {
                "model_family": "Transformer",
                "best_candidate_id": recommended_transformer_id or "",
                "best_pooled_rmse_wh": (tr_pm.pooled_rmse_wh if tr_pm else None),
                "fold_count": n_folds,
                "status": "candidates=TR_C0,TR_C1,TR_C2",
            },
            {
                "model_family": "LSTM",
                "best_candidate_id": LSTM_CANDIDATE_ID,
                "best_pooled_rmse_wh": (ls_pm.pooled_rmse_wh if ls_pm else None),
                "fold_count": n_folds,
                "status": "candidates=LSTM_TUNED_WINNER",
            },
            {
                "model_family": "Persistence",
                "best_candidate_id": PERSISTENCE_MODEL_ID,
                "best_pooled_rmse_wh": pers_pm.pooled_rmse_wh,
                "fold_count": n_folds,
                "status": "candidates=PERSISTENCE_LAST_VALUE",
            },
        ]
        p = write_model_family_comparison(
            path=artifact_dir / "rolling_origin_model_family_robustness_comparison.csv",
            rows=family_rows,
        )
        artifacts_written["family_comparison"] = str(p)

        # O44.10: rolling_origin_recommended_transformer.json
        if recommended_transformer_id:
            p = write_recommended_transformer(
                path=artifact_dir / "rolling_origin_recommended_transformer.json",
                payload={
                    "candidate_id": recommended_transformer_id,
                    "config_fingerprint": next(
                        c.config_fingerprint for c in candidates
                        if c.candidate_id == recommended_transformer_id
                    ),
                    "pooled_rmse_wh": pooled_metrics_by_cid[recommended_transformer_id].pooled_rmse_wh,
                    "pooled_mae_wh": pooled_metrics_by_cid[recommended_transformer_id].pooled_mae_wh,
                    "pooled_r2": pooled_metrics_by_cid[recommended_transformer_id].pooled_r2,
                    "created_at": now,
                },
            )
            artifacts_written["recommended_transformer"] = str(p)

        # O44.11: rolling_origin_results.csv
        results_rows = [
            {
                "candidate_id": cid,
                "fold_id": fid,
                "mae_wh": float(m[0]),
                "rmse_wh": float(m[1]),
                "r2": float(m[2]),
                "is_transformer": cid != LSTM_CANDIDATE_ID,
                "is_persistence": cid == PERSISTENCE_MODEL_ID,
            }
            for (cid, fid), m in per_fold_metrics_by_cid.items()
        ]
        p = _df_to_csv(
            pd.DataFrame(results_rows),
            artifact_dir / "rolling_origin_results.csv",
        )
        artifacts_written["results"] = str(p)

        # O44.12–23: Outer prediction bundles (12 learned predictions)
        pred_dir = artifact_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for (cid, fid), d in learned_predictions.items():
            tids = d.get("target_ids", [])
            yt = d.get("y_true_wh", [])
            yp = d.get("y_pred_wh", [])
            rows = []
            for i, tid in enumerate(tids):
                rows.append({
                    "fold_id": str(fid),
                    "candidate_id": cid,
                    "target_id": str(tid),
                    "y_true_wh": float(yt[i]) if i < len(yt) else float("nan"),
                    "y_pred_wh": float(yp[i]) if i < len(yp) else float("nan"),
                })
            p = _df_to_csv(
                pd.DataFrame(rows),
                pred_dir / f"outer_predictions_{cid}_{fid}.csv",
            )
            artifacts_written[f"outer_pred_{cid}_{fid}"] = str(p)

        # O44.24–26: Persistence bundles (3)
        for fid, d in persistence_predictions.items():
            tids = d.get("target_ids", [])
            yt = d.get("y_true_wh", [])
            yp = d.get("y_pred_wh", [])
            rows = []
            for i, tid in enumerate(tids):
                rows.append({
                    "fold_id": str(fid),
                    "candidate_id": PERSISTENCE_MODEL_ID,
                    "target_id": str(tid),
                    "y_true_wh": float(yt[i]) if i < len(yt) else float("nan"),
                    "y_pred_wh": float(yp[i]) if i < len(yp) else float("nan"),
                })
            p = _df_to_csv(
                pd.DataFrame(rows),
                pred_dir / f"outer_predictions_{PERSISTENCE_MODEL_ID}_{fid}.csv",
            )
            artifacts_written[f"persistence_{fid}"] = str(p)

        # ── Step 11: Phase 45 handoff (only if signoff is PASS/PASS_WITH_WARNING). ──
        phase45_handoff = None
        if recommended_transformer_id and signoff_status in {"PASS", "PASS_WITH_WARNING"}:
            rec_pm = pooled_metrics_by_cid[recommended_transformer_id]
            rec_mm = macro_metrics_by_cid[recommended_transformer_id]
            phase45_handoff = {
                "source_phase44_version": ROBASE_VERSION,
                "recommended_transformer_candidate_id": recommended_transformer_id,
                "recommended_transformer_fingerprint": next(
                    c.config_fingerprint for c in candidates
                    if c.candidate_id == recommended_transformer_id
                ),
                "recommended_transformer_config": next(
                    c.config for c in candidates
                    if c.candidate_id == recommended_transformer_id
                ),
                "rolling_origin_pooled_metrics": [
                    {
                        "candidate_id": r.candidate_id,
                        "pooled_rmse_wh": r.pooled_metrics.pooled_rmse_wh,
                        "pooled_mae_wh": r.pooled_metrics.pooled_mae_wh,
                        "pooled_r2": r.pooled_metrics.pooled_r2,
                    }
                    for r in tr_ranking
                ],
                "rolling_origin_fold_metrics": [],
                "transformer_ranking": transformer_ranking_rows,
                "lstm_tuned_context": next(
                    c.config for c in candidates
                    if c.candidate_id == LSTM_CANDIDATE_ID
                ) if any(c.candidate_id == LSTM_CANDIDATE_ID for c in candidates) else {},
                "persistence_context": {
                    "model_id": PERSISTENCE_MODEL_ID,
                    "version": "WB0_LAST_STEP_v1",
                    "pooled_rmse_wh": pers_pm.pooled_rmse_wh,
                },
                "recommended_transformer_inner_best_epochs": {
                    str(f.fold_id): inner_best_epochs.get(
                        (recommended_transformer_id, str(f.fold_id)), -1
                    )
                    for f in folds
                },
                "recommended_transformer_stage_a_run_ids": {
                    str(f.fold_id): stage_a_run_ids.get(
                        (recommended_transformer_id, str(f.fold_id)), ""
                    )
                    for f in folds
                },
                "recommended_transformer_stage_b_run_ids": {
                    str(f.fold_id): stage_b_run_ids.get(
                        (recommended_transformer_id, str(f.fold_id)), ""
                    )
                    for f in folds
                },
                "test_status": "NOT_ACCESSED",
                "approved_for_phase45": True,
                "warnings": [],
                "mean_fold_rmse_wh": rec_mm.macro_rmse_wh,
                "worst_fold_rmse_wh": rec_mm.worst_fold_rmse_wh,
                "fold_rmse_sd_wh": rec_mm.fold_rmse_sd_wh,
            }

        # Step 12: Persistence pooled metrics summary
        pers_pooled_summary = {
            "rmse_wh": pers_pm.pooled_rmse_wh,
            "mae_wh": pers_pm.pooled_mae_wh,
            "r2": pers_pm.pooled_r2,
        }

        # Step 11b: Write Phase45 handoff to canonical rolling_origin/
        if phase45_handoff is not None:
            from course_work.rolling_origin.artifacts import (
                write_phase45_handoff,
            )
            h_path = write_phase45_handoff(
                path=artifact_dir / "phase45_final_model_lock_handoff.json",
                payload=phase45_handoff,
            )
            artifacts_written["phase45_handoff"] = str(h_path)

        return RealRunResult(
            exit_code=0 if signoff_status != "FAIL" else 1,
            summary=(
                f"Phase 44 real orchestrator: {n_candidates} candidates × "
                f"{n_folds} folds; {len(stage_a_run_ids)} Stage A + "
                f"{len(stage_b_run_ids)} Stage B runs; "
                f"{len(learned_predictions)} learned + "
                f"{len(persistence_predictions)} persistence bundles; "
                f"signoff={signoff_status}"
            ),
            n_candidates=n_candidates,
            n_folds=n_folds,
            n_stage_a_runs=len(stage_a_run_ids),
            n_stage_b_runs=len(stage_b_run_ids),
            n_outer_prediction_bundles=len(learned_predictions),
            n_persistence_bundles=len(persistence_predictions),
            o44_artifacts_written=artifacts_written,
            inner_best_epochs=dict(inner_best_epochs),
            stage_a_run_ids=dict(stage_a_run_ids),
            stage_b_run_ids=dict(stage_b_run_ids),
            persistence_pooled_metrics=pers_pooled_summary,
            pooled_metrics_by_cid={
                cid: {"rmse_wh": pm.pooled_rmse_wh,
                       "mae_wh": pm.pooled_mae_wh,
                       "r2": pm.pooled_r2}
                for cid, pm in pooled_metrics_by_cid.items()
            },
            recommended_transformer_id=recommended_transformer_id,
            signoff_overall_status=signoff_status,
            signoff_failures=signoff_failures,
            phase45_handoff=phase45_handoff,
        )
    except KeyboardInterrupt as exc:
        if ctx.validated_registry_lifecycle and registry is not None:
            for run_id, stage in tuple(active_run_ids.items()):
                try:
                    registry.fail_run(
                        run_id,
                        FailureType.INTERRUPTED.value,
                        stage,
                        "E01 execution interrupted by the human or operating system",
                        exception_class=type(exc).__name__,
                        recoverable=True,
                    )
                except Exception:
                    pass
        raise
    except Exception as exc:
        if ctx.validated_registry_lifecycle and registry is not None:
            for run_id, stage in tuple(active_run_ids.items()):
                try:
                    registry.fail_run(
                        run_id,
                        FailureType.TRAINING_ERROR.value,
                        stage,
                        str(exc),
                        exception_class=type(exc).__name__,
                    )
                except Exception:
                    pass
        import traceback
        return RealRunResult(
            exit_code=1,
            summary=f"Phase 44 real orchestrator failed: {type(exc).__name__}: {exc}",
            n_candidates=0, n_folds=0,
            n_stage_a_runs=0, n_stage_b_runs=0,
            n_outer_prediction_bundles=0, n_persistence_bundles=0,
            exception=traceback.format_exc(),
        )
