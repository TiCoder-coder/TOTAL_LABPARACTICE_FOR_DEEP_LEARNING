"""Persistence-only finalization helper for Seed 42 corrective Phase 46.

This module provides a deterministic, non-training pathway to finalize the
already-completed corrective Phase 46 Seed 42 training run
(RUN_TR_FSD_0183_C2F24D58) as the official FINAL_REFIT checkpoint, without
re-running any training or inference. It is a pure persistence-side fix for
the variable-name bug fixed in save_seed_checkpoint() (Part 2G-I).

SCOPE (read-only inputs, write-only outputs):
- READ: artifacts/runs/RUN_TR_FSD_0183_C2F24D58/{config.json, status.json,
       training_history.csv, training.log, checkpoints/best_checkpoint.pt,
       metrics/best_validation_metrics.json}.
- READ: artifacts/final_model_lock/{phase_45_signoff.json,
       phase46_three_seed_handoff.json, final_model_scientific_config.json,
       final_scaling_contract.json}.
- READ: artifacts/final_dev_region/final_dev_region_manifest.json.
- READ: artifacts/three_seed_final_runs/official_checkpoints/seed_42/
       seed_42_FINAL_REFIT_metadata.json (legacy, will be ARCHIVED).
- WRITE: artifacts/three_seed_final_runs/official_checkpoints/seed_42/
        seed_42_FINAL_REFIT.pt (FINAL_REFIT envelope wrapping
        best_checkpoint.pt's model_state_dict).
- WRITE: artifacts/three_seed_final_runs/official_checkpoints/seed_42/
        seed_42_FINAL_REFIT_metadata.json (canonical FINAL_REFIT metadata).
- WRITE: artifacts/three_seed_final_runs/official_checkpoints/seed_42/
        _history/SEED42_PERSISTENCE_RECOVERY_<UTC>/seed_42_FINAL_REFIT_metadata.json
        (archived legacy metadata, sha-verified).

The helper is invoked ONLY when all of the following hold:
  1. RUN_TR_FSD_0183_C2F24D58 evidence exists on disk and is COMPLETED.
  2. The Phase 45 lock and handoff pass all identity gates (seed=42,
     candidate=TR_C2_ALT_LOOKBACK, lookback=72, FINAL_REFIT_EPOCHS=30,
     FINAL_DEV population, X/Y FINAL_SCALING-v1 SHAs).
  3. The model's best_epoch == 30 == training_cfg.max_epochs.
  4. The model's model_state_dict is loadable.
  5. Deterministic-mode metadata is recorded
     (cudnn_deterministic=True, torch_deterministic_algorithms=True,
     deterministic_mode=D0).

INVARIANTS:
- NEVER call training engine, optimizer.step, model.train(), or .backward().
- NEVER fit scalers.
- NEVER touch Test. test_access_authorized must be False.
- NEVER overwrite existing FINAL_REFIT artifacts (idempotent).
- Archive any pre-existing legacy metadata at
  artifacts/three_seed_final_runs/official_checkpoints/seed_42/_history/SEED42_PERSISTENCE_RECOVERY_<UTC>/
  before writing the canonical metadata, with SHA256 verification.
- Model state at epoch 30 is deterministic given (seed, locked config,
  scalers, FINAL_DEV population), so persistence-only finalization is
  scientifically equivalent to retraining under the Phase 45 lock.

GOVERNANCE:
- Source checkpoint is classified RECOVERY_ONLY per
  docs/plan/plan_detail_for_each_phase/Phase_46_Three-seed_final_runs.md §63.
  This helper promotes the recovery-only checkpoint to FINAL_REFIT by
  attaching the canonical provenance envelope.
- The source run_id RUN_TR_FSD_0183_C2F24D58 is preserved in the FINAL_REFIT
  metadata's recovery_metadata block. The corrected plan §2.3 requires
  new run_ids from RUN_TR_FSD_0256+; RUN 0183 has sequence 183, predating
  the canonical plan. The recovery_metadata block records this lineage
  deviation explicitly so that downstream auditors can see the recovery
  is a persistence-only finalization of an interrupted pre-canonical
  training run, not a new RUN_TR_FSD_0256+ training pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import os as _os 

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parents[1])) 
from course_work.utils.artifacts import get_project_root as _gpr 

ROOT = _gpr()
del _gpr, _os

ARTIFACT_DIR = ROOT / "artifacts" / "three_seed_final_runs"
CHECKPOINT_DIR = ARTIFACT_DIR / "official_checkpoints"
RUNS_ROOT = ROOT / "artifacts" / "runs"
PHASE_45_SIGNOFF = ROOT / "artifacts" / "final_model_lock" / "phase_45_signoff.json"
PHASE_46_HANDOFF = ROOT / "artifacts" / "final_model_lock" / "phase46_three_seed_handoff.json"
PHASE_46_SCIENTIFIC_CONFIG = (
    ROOT / "artifacts" / "final_model_lock" / "final_model_scientific_config.json"
)
FINAL_SCALING_CONTRACT = (
    ROOT / "artifacts" / "final_model_lock" / "final_scaling_contract.json"
)
FINAL_DEV_MANIFEST = ROOT / "artifacts" / "final_dev_region" / "final_dev_region_manifest.json"

LOCKED_CANDIDATE_ID = "TR_C2_ALT_LOOKBACK"
LOCKED_LOOKBACK_STEPS = 72
LOCKED_FINAL_REFIT_EPOCHS = 30
LOCKED_SEEDS = (42, 123, 2026)
LOCKED_SEED = 42
LOCKED_CONFIG_FINGERPRINT = "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"
LOCKED_FINAL_LOCK_SHA256 = "81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec"
LOCKED_RECIPE_SHA256 = "857dbaf7903792cdba3e126a50d8919992f02e0fff838cba86180026c9e0220c"
LOCKED_FEATURE_SHA256 = "fc9c428964285d1ad74e97f3ae2c18e7efeb3e61d61f7acd0b54039bc2ca6dee"
LOCKED_LINEAGE_SHA256 = "9ba532539d548156e1d5d933abe8c1463617cdc790741e48d5249b452995d9fe"

SOURCE_RUN_ID = "RUN_TR_FSD_0183_C2F24D58"

class Seed42RecoveryContractViolation(RuntimeError):
    """Raised when the Seed 42 recovery contract is violated."""


class Seed42RecoveryTrainingAttempt(RuntimeError):
    """Raised if the recovery helper detects any training attempt."""

def _training_attempt_guard() -> None:
    """Install a guard that raises on any optimizer.step() call.

    Defensive sentinel — recovery helper must NEVER execute any optimizer
    step. If something does, we fail loud and clear.
    """
    import torch.optim as _opt

    def _make_guard_step():
        def _guard_step(self, *args, **kwargs):
            raise Seed42RecoveryTrainingAttempt(
                "TRAINING ATTEMPT DETECTED DURING SEED 42 RECOVERY — ABORT. "
                "Recovery is persistence-only; optimizer.step() must NEVER be called."
            )

        return _guard_step

    _opt.Optimizer.step = _make_guard_step()
    for klass in _opt.Optimizer.__subclasses__():
        klass.step = _make_guard_step()
        for subklass in klass.__subclasses__():
            subklass.step = _make_guard_step()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_load_checkpoint_torch(path: Path) -> dict[str, Any]:
    """Load a torch checkpoint without invoking the training engine."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)

def _verify_locked_identities(
    p45_signoff: dict[str, Any],
    p46_handoff: dict[str, Any],
    final_dev_manifest: dict[str, Any],
) -> None:
    """Verify Phase 45/46 locked identities used as recovery gates."""
    if p45_signoff.get("status") != "PASS":
        raise Seed42RecoveryContractViolation(
            f"Phase 45 signoff status={p45_signoff.get('status')!r} != PASS"
        )
    if not p46_handoff.get("ready_for_phase46"):
        raise Seed42RecoveryContractViolation(
            "Phase 46 handoff ready_for_phase46 != True"
        )
    if int(p46_handoff.get("FINAL_REFIT_EPOCHS", 0)) != LOCKED_FINAL_REFIT_EPOCHS:
        raise Seed42RecoveryContractViolation(
            f"Phase 46 handoff FINAL_REFIT_EPOCHS != {LOCKED_FINAL_REFIT_EPOCHS}"
        )
    handoff_seeds = p46_handoff.get("seeds") or p45_signoff.get("seed_list")
    if list(handoff_seeds) != list(LOCKED_SEEDS):
        raise Seed42RecoveryContractViolation(
            f"Phase 46 handoff / Phase 45 seed_list != {list(LOCKED_SEEDS)} "
            f"(got {list(handoff_seeds)})"
        )
    if p45_signoff.get("locked_model_id") != LOCKED_CANDIDATE_ID:
        raise Seed42RecoveryContractViolation(
            f"Phase 45 locked_model_id != {LOCKED_CANDIDATE_ID}"
        )
    if p45_signoff.get("config_fingerprint") != LOCKED_CONFIG_FINGERPRINT:
        raise Seed42RecoveryContractViolation(
            f"Phase 45 config_fingerprint != {LOCKED_CONFIG_FINGERPRINT}"
        )
    if p45_signoff.get("final_lock_sha256") != LOCKED_FINAL_LOCK_SHA256:
        raise Seed42RecoveryContractViolation(
            f"Phase 45 final_lock_sha256 != {LOCKED_FINAL_LOCK_SHA256}"
        )
    if final_dev_manifest.get("lookback_steps") != LOCKED_LOOKBACK_STEPS:
        raise Seed42RecoveryContractViolation(
            f"FINAL_DEV lookback_steps != {LOCKED_LOOKBACK_STEPS}"
        )
    if int(final_dev_manifest.get("final_dev_window_count", 0)) != 16630:
        raise Seed42RecoveryContractViolation(
            f"FINAL_DEV window_count={final_dev_manifest.get('final_dev_window_count')} != 16630"
        )
    if int(final_dev_manifest.get("test_window_count", -1)) != 0:
        raise Seed42RecoveryContractViolation(
            f"FINAL_DEV test_window_count={final_dev_manifest.get('test_window_count')} != 0"
        )

def _archive_legacy_metadata(
    legacy_meta_path: Path,
    seed42_dir: Path,
    timestamp: str,
) -> dict[str, Any]:
    """SHA-verify the legacy metadata and archive it under _history/.

    Returns the archive manifest entry.
    """
    if not legacy_meta_path.exists():
        return {"archived": False, "reason": "no legacy metadata"}

    pre_sha = _sha256_file(legacy_meta_path)
    archive_dir = seed42_dir / "_history" / f"SEED42_PERSISTENCE_RECOVERY_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / legacy_meta_path.name
    archive_path.write_bytes(legacy_meta_path.read_bytes())

    post_sha = _sha256_file(archive_path)
    if post_sha != pre_sha:
        archive_path.unlink()
        archive_dir.rmdir()
        raise Seed42RecoveryContractViolation(
            f"Legacy metadata archive SHA mismatch: pre={pre_sha} post={post_sha}"
        )

    def _rel_or_abs(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    return {
        "archived": True,
        "source_path": _rel_or_abs(legacy_meta_path),
        "archive_path": _rel_or_abs(archive_path),
        "source_sha256": pre_sha,
        "archive_sha256": post_sha,
        "archived_at": _now_iso(),
        "reason": (
            "SEED42_PERSISTENCE_RECOVERY: legacy metadata from RUN_TR_FSD_0153 "
            "(historical invalidated) replaced by canonical FINAL_REFIT metadata "
            "for RUN_TR_FSD_0183_C2F24D58 (corrected persistence-only finalization)."
        ),
    }


def _verify_run_0183_evidence() -> dict[str, Any]:
    """Read and validate every artifact under artifacts/runs/RUN_TR_FSD_0183_C2F24D58/."""
    run_root = RUNS_ROOT / SOURCE_RUN_ID
    if not run_root.exists():
        raise Seed42RecoveryContractViolation(
            f"Source run directory missing: {run_root}"
        )

    config_path = run_root / "config.json"
    status_path = run_root / "status.json"
    history_path = run_root / "training_history.csv"
    metrics_path = run_root / "metrics" / "best_validation_metrics.json"
    best_ckpt_path = run_root / "checkpoints" / "best_checkpoint.pt"

    for p in (config_path, status_path, history_path, metrics_path, best_ckpt_path):
        if not p.exists():
            raise Seed42RecoveryContractViolation(
                f"Source artifact missing: {p}"
            )

    config_data = _read_json(config_path)
    if config_data.get("run_id") != SOURCE_RUN_ID:
        raise Seed42RecoveryContractViolation(
            f"config.json run_id={config_data.get('run_id')!r} != {SOURCE_RUN_ID}"
        )

    config = config_data["config"]
    training = config["training"]
    data = config["data"]
    model_cfg = config["model"]
    reproducibility = config["reproducibility"]
    lineage = config["lineage"]

    if int(training["seed"]) != LOCKED_SEED:
        raise Seed42RecoveryContractViolation(
            f"training.seed={training['seed']} != {LOCKED_SEED}"
        )
    if data["lookback_steps"] != LOCKED_LOOKBACK_STEPS:
        raise Seed42RecoveryContractViolation(
            f"data.lookback_steps={data['lookback_steps']} != {LOCKED_LOOKBACK_STEPS}"
        )
    if int(training["max_epochs"]) != LOCKED_FINAL_REFIT_EPOCHS:
        raise Seed42RecoveryContractViolation(
            f"training.max_epochs={training['max_epochs']} != {LOCKED_FINAL_REFIT_EPOCHS}"
        )
    if not bool(training["final_refit_mode"]):
        raise Seed42RecoveryContractViolation(
            "training.final_refit_mode is not True"
        )
    if bool(training.get("early_stopping_enabled", True)):
        raise Seed42RecoveryContractViolation(
            "training.early_stopping_enabled is not False"
        )
    if config_data.get("test_access_authorized", False):
        raise Seed42RecoveryContractViolation(
            "config.test_access_authorized=True — recovery is forbidden when Test was accessed"
        )

    deterministic_mode = reproducibility.get("deterministic_mode", "")
    if not bool(reproducibility.get("cudnn_deterministic", False)):
        raise Seed42RecoveryContractViolation(
            "reproducibility.cudnn_deterministic != True — recovery equivalence uncertain"
        )
    if not bool(reproducibility.get("torch_deterministic_algorithms", False)):
        raise Seed42RecoveryContractViolation(
            "reproducibility.torch_deterministic_algorithms != True"
        )
    if deterministic_mode != "D0":
        raise Seed42RecoveryContractViolation(
            f"reproducibility.deterministic_mode={deterministic_mode!r} != 'D0'"
        )

    status = _read_json(status_path)
    if status.get("status") != "COMPLETED":
        raise Seed42RecoveryContractViolation(
            f"status.status={status.get('status')!r} != COMPLETED"
        )
    if int(status.get("best_epoch", -1)) != LOCKED_FINAL_REFIT_EPOCHS:
        raise Seed42RecoveryContractViolation(
            f"status.best_epoch={status.get('best_epoch')} != {LOCKED_FINAL_REFIT_EPOCHS}"
        )

    history_lines = history_path.read_text().strip().split("\n")
    header = history_lines[0].split(",")
    epoch_idx = header.index("epoch")
    is_best_idx = header.index("is_best")
    train_rmse_idx = header.index("train_rmse_wh")
    val_rmse_idx = header.index("validation_rmse_wh")
    last_row = history_lines[-1].split(",")
    if int(last_row[epoch_idx]) != LOCKED_FINAL_REFIT_EPOCHS:
        raise Seed42RecoveryContractViolation(
            f"training_history last epoch={last_row[epoch_idx]} != {LOCKED_FINAL_REFIT_EPOCHS}"
        )
    if last_row[is_best_idx] != "True":
        raise Seed42RecoveryContractViolation(
            f"training_history last row is_best={last_row[is_best_idx]} != True"
        )
    for line in history_lines[1:]:
        cols = line.split(",")
        if cols[train_rmse_idx] != cols[val_rmse_idx]:
            raise Seed42RecoveryContractViolation(
                f"FINAL_REFIT semantics violated at epoch {cols[epoch_idx]}: "
                f"train_rmse_wh={cols[train_rmse_idx]} != validation_rmse_wh={cols[val_rmse_idx]}"
            )

    ckpt = _safe_load_checkpoint_torch(best_ckpt_path)
    if int(ckpt.get("best_epoch", -1)) != LOCKED_FINAL_REFIT_EPOCHS:
        raise Seed42RecoveryContractViolation(
            f"best_checkpoint.pt best_epoch={ckpt.get('best_epoch')} != {LOCKED_FINAL_REFIT_EPOCHS}"
        )
    state_dict = ckpt.get("model_state_dict", {})
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise Seed42RecoveryContractViolation(
            "best_checkpoint.pt has empty or missing model_state_dict"
        )

    if lineage.get("final_scaling_x_sha256") != _sha256_final_scaling_x():
        raise Seed42RecoveryContractViolation(
            "lineage.final_scaling_x_sha256 does not match on-disk FINAL_SCALING-v1 X"
        )
    if lineage.get("final_scaling_y_sha256") != _sha256_final_scaling_y():
        raise Seed42RecoveryContractViolation(
            "lineage.final_scaling_y_sha256 does not match on-disk FINAL_SCALING-v1 Y"
        )

    return {
        "config": config,
        "config_data": config_data,
        "status": status,
        "history": history_lines,
        "metrics": _read_json(metrics_path),
        "best_ckpt_path": best_ckpt_path,
        "best_ckpt_payload": ckpt,
        "model_state_sha256": _sha256_file(best_ckpt_path),
        "source_run_id": SOURCE_RUN_ID,
    }


def _sha256_final_scaling_x() -> str:
    p = ROOT / "artifacts" / "scaling" / "final_dev" / "x" / "XSCALER_FINAL__FS2_TF1__FINAL_SCALING-v1.joblib"
    return _sha256_file(p)


def _sha256_final_scaling_y() -> str:
    p = ROOT / "artifacts" / "scaling" / "final_dev" / "y" / "YSCALER_FINAL__YS1__FINAL_SCALING-v1.joblib"
    return _sha256_file(p)


def _build_final_refit_envelope(
    evidence: dict[str, Any],
    final_dev_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build the FINAL_REFIT metadata envelope per Phase 45 lock + plan §2.4."""
    config = evidence["config"]
    status = evidence["status"]
    metrics = evidence["metrics"]
    training = config["training"]
    model_cfg = config["model"]
    model_state_sha = evidence["model_state_sha256"]
    source_ckpt_path = evidence["best_ckpt_path"]

    metric_result = metrics.get("metric_result", {})

    persisted_payload = dict(evidence["best_ckpt_payload"])
    persisted_payload["official_epoch"] = LOCKED_FINAL_REFIT_EPOCHS
    persisted_payload["FINAL_REFIT_EPOCHS"] = LOCKED_FINAL_REFIT_EPOCHS
    persisted_payload["seed"] = LOCKED_SEED
    persisted_payload["run_id"] = SOURCE_RUN_ID
    persisted_payload["checkpoint_type"] = "FINAL_REFIT"
    persisted_payload["final_lock_sha256"] = LOCKED_FINAL_LOCK_SHA256
    persisted_payload["config_fingerprint"] = LOCKED_CONFIG_FINGERPRINT
    persisted_payload["config_sha256"] = LOCKED_CONFIG_FINGERPRINT
    persisted_payload["recipe_sha256"] = LOCKED_RECIPE_SHA256
    persisted_payload["lineage_sha256"] = LOCKED_LINEAGE_SHA256
    persisted_payload["feature_sha256"] = LOCKED_FEATURE_SHA256
    persisted_payload["population_fingerprint"] = final_dev_manifest["population_fingerprint"]
    persisted_payload["final_dev_population_fingerprint"] = final_dev_manifest["population_fingerprint"]
    persisted_payload["x_scaler_sha256"] = _sha256_final_scaling_x()
    persisted_payload["y_scaler_sha256"] = _sha256_final_scaling_y()
    persisted_payload["model_config"] = {
        "d_model": model_cfg.get("d_model"),
        "num_heads": model_cfg.get("num_heads"),
        "num_layers": model_cfg.get("num_layers"),
        "ffn_dim": model_cfg.get("ffn_dim"),
        "dropout": model_cfg.get("dropout"),
        "pooling": model_cfg.get("pooling"),
    }
    persisted_payload["optimizer_config"] = {
        "optimizer_name": training.get("optimizer_name", "AdamW"),
        "learning_rate": training.get("learning_rate"),
        "weight_decay": training.get("weight_decay"),
    }
    persisted_payload["loss_config"] = {
        "loss_name": training.get("loss_name", "MSE"),
    }
    persisted_payload["gradient_clipping"] = {
        "enabled": training.get("gradient_clipping_enabled", True),
        "max_norm": training.get("gradient_clip_max_norm"),
    }
    persisted_payload["RevIN"] = {
        "enabled": config.get("model", {}).get("use_revin", False),
    }
    persisted_payload["training_history"] = [
        dict(zip(evidence["history"][0].split(","), line.split(",")))
        for line in evidence["history"][1:]
    ]
    persisted_payload["metric_result"] = metric_result
    persisted_payload["recovery_metadata"] = {
        "persistence_only_finalization": True,
        "source_run_id": SOURCE_RUN_ID,
        "source_checkpoint_path": str(source_ckpt_path.relative_to(ROOT)),
        "source_checkpoint_sha256": model_state_sha,
        "crash_reason": (
            "NameError: checkpoint_metadata is not defined in "
            "save_seed_checkpoint() (variable-name bug fixed in Part 2G-I)."
        ),
        "crash_fix_commit_intent": (
            "p46_three_seed_runs.py lines 1008-1010 renamed "
            "`checkpoint_metadata[...]` -> `payload[...]` so torch.save "
            "and write_json succeed."
        ),
        "equivalence_rationale": (
            "Source run was trained with deterministic_mode=D0, "
            "cudnn_deterministic=True, torch_deterministic_algorithms=True "
            "under the locked Phase 45 scientific contract. The model "
            "state at epoch=30 (=max_epochs=30, FINAL_REFIT) is "
            "deterministic given (seed, locked config, scalers, FINAL_DEV "
            "population). Persistence-only finalization wraps the existing "
            "model_state_dict in the canonical FINAL_REFIT provenance "
            "envelope without altering model parameters. Scientific "
            "identity is preserved."
        ),
        "canonical_plan_run_id_rule_deviation": (
            "Canonical Phase 46 plan §2.3 requires new run_ids from "
            "RUN_TR_FSD_0256+ prefix. RUN 0183 has sequence 183, predating "
            "the canonical plan. The corrected Phase 46 runner was "
            "executed before the canonical plan was finalized. This "
            "recovery preserves the source run_id (RUN_TR_FSD_0183_C2F24D58) "
            "explicitly in recovery_metadata so downstream auditors can see "
            "the lineage deviation."
        ),
        "finalized_at": _now_iso(),
    }

    metadata_sidecar = {
        "seed": LOCKED_SEED,
        "run_id": SOURCE_RUN_ID,
        "phase": 46,
        "checkpoint_type": "FINAL_REFIT",
        "official_epoch": LOCKED_FINAL_REFIT_EPOCHS,
        "FINAL_REFIT_EPOCHS": LOCKED_FINAL_REFIT_EPOCHS,
        "checkpoint_path": (
            f"artifacts/three_seed_final_runs/official_checkpoints/seed_{LOCKED_SEED}/"
            f"seed_{LOCKED_SEED}_FINAL_REFIT.pt"
        ),
        "model_state_sha256": model_state_sha,
        "final_lock_sha256": LOCKED_FINAL_LOCK_SHA256,
        "config_fingerprint": LOCKED_CONFIG_FINGERPRINT,
        "config_sha256": LOCKED_CONFIG_FINGERPRINT,
        "recipe_sha256": LOCKED_RECIPE_SHA256,
        "lineage_sha256": LOCKED_LINEAGE_SHA256,
        "feature_sha256": LOCKED_FEATURE_SHA256,
        "population_fingerprint": final_dev_manifest["population_fingerprint"],
        "final_dev_population_fingerprint": final_dev_manifest["population_fingerprint"],
        "x_scaler_sha256": _sha256_final_scaling_x(),
        "y_scaler_sha256": _sha256_final_scaling_y(),
        "rmse_wh": float(status["best_validation_rmse_wh"]),
        "test_metrics_computed": False,
        "created_at": _now_iso(),
        "recovery_metadata": persisted_payload["recovery_metadata"],
    }

    return {
        "persisted_payload": persisted_payload,
        "metadata_sidecar": metadata_sidecar,
        "final_lock_sha256": LOCKED_FINAL_LOCK_SHA256,
        "candidate_id": LOCKED_CANDIDATE_ID,
    }

def finalize_seed42(
    *,
    force: bool = False,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Finalize Seed 42 from RUN_TR_FSD_0183_C2F24D58 as FINAL_REFIT.

    Args:
        force: If True, overwrite an existing FINAL_REFIT .pt/.json. Default
            False (idempotent; refuses to overwrite).
        timestamp: Override the UTC timestamp used in the archive directory
            name. Default: now (UTC, %Y%m%dT%H%M%SZ).

    Returns:
        The finalization report dict (suitable for JSON serialization).
    """
    _training_attempt_guard()

    p45_signoff = _read_json(PHASE_45_SIGNOFF)
    p46_handoff = _read_json(PHASE_46_HANDOFF)
    final_dev_manifest = _read_json(FINAL_DEV_MANIFEST)
    x_sha = _sha256_final_scaling_x()
    y_sha = _sha256_final_scaling_y()

    _verify_locked_identities(
        p45_signoff,
        p46_handoff,
        final_dev_manifest,
    )

    evidence = _verify_run_0183_evidence()

    seed42_dir = CHECKPOINT_DIR / f"seed_{LOCKED_SEED}"
    seed42_dir.mkdir(parents=True, exist_ok=True)
    legacy_meta_path = seed42_dir / f"seed_{LOCKED_SEED}_FINAL_REFIT_metadata.json"
    final_pt_path = seed42_dir / f"seed_{LOCKED_SEED}_FINAL_REFIT.pt"

    if final_pt_path.exists() and not force:
        raise Seed42RecoveryContractViolation(
            f"FINAL_REFIT .pt already exists at {final_pt_path}. "
            "Refusing to overwrite (use force=True if intentional)."
        )

    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_entry = _archive_legacy_metadata(legacy_meta_path, seed42_dir, ts)

    envelope = _build_final_refit_envelope(evidence, final_dev_manifest)
    persisted_payload = envelope["persisted_payload"]
    metadata_sidecar = envelope["metadata_sidecar"]

    import torch
    tmp_pt = final_pt_path.with_suffix(final_pt_path.suffix + ".tmp")
    torch.save(persisted_payload, tmp_pt)
    tmp_pt.replace(final_pt_path)

    actual_sha = _sha256_file(final_pt_path)
    if actual_sha != metadata_sidecar["model_state_sha256"]:
        final_pt_path.unlink()
        raise Seed42RecoveryContractViolation(
            f"Persisted .pt SHA {actual_sha} != metadata {metadata_sidecar['model_state_sha256']}"
        )

    tmp_meta = legacy_meta_path.with_suffix(legacy_meta_path.suffix + ".tmp")
    tmp_meta.write_text(json.dumps(metadata_sidecar, indent=2))
    tmp_meta.replace(legacy_meta_path)

    return {
        "finalized": True,
        "seed": LOCKED_SEED,
        "source_run_id": SOURCE_RUN_ID,
        "final_pt_path": str(final_pt_path.relative_to(ROOT)),
        "final_metadata_path": str(legacy_meta_path.relative_to(ROOT)),
        "model_state_sha256": metadata_sidecar["model_state_sha256"],
        "official_epoch": LOCKED_FINAL_REFIT_EPOCHS,
        "final_lock_sha256": LOCKED_FINAL_LOCK_SHA256,
        "config_fingerprint": LOCKED_CONFIG_FINGERPRINT,
        "candidate_id": LOCKED_CANDIDATE_ID,
        "x_scaler_sha256": x_sha,
        "y_scaler_sha256": y_sha,
        "final_dev_population_fingerprint": final_dev_manifest["population_fingerprint"],
        "rmse_wh": float(metadata_sidecar["rmse_wh"]),
        "archive_entry": archive_entry,
        "force_used": bool(force),
        "finalized_at": _now_iso(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistence-only finalization of Seed 42 from RUN 0183 (no training).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing FINAL_REFIT artifacts (default: refuse to overwrite).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify all gates and print the planned writes WITHOUT actually writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.dry_run:
            _training_attempt_guard()
            p45_signoff = _read_json(PHASE_45_SIGNOFF)
            p46_handoff = _read_json(PHASE_46_HANDOFF)
            final_dev_manifest = _read_json(FINAL_DEV_MANIFEST)
            x_sha = _sha256_final_scaling_x()
            y_sha = _sha256_final_scaling_y()
            _verify_locked_identities(
                p45_signoff, p46_handoff, final_dev_manifest,
            )
            evidence = _verify_run_0183_evidence()
            seed42_dir = CHECKPOINT_DIR / f"seed_{LOCKED_SEED}"
            final_pt_path = seed42_dir / f"seed_{LOCKED_SEED}_FINAL_REFIT.pt"
            legacy_meta_path = seed42_dir / f"seed_{LOCKED_SEED}_FINAL_REFIT_metadata.json"

            print("=" * 70)
            print("PHASE 46 — SEED 42 PERSISTENCE-ONLY FINALIZATION (DRY-RUN)")
            print("=" * 70)
            print(f"  source_run_id                   = {SOURCE_RUN_ID}")
            print(f"  candidate_id                    = {LOCKED_CANDIDATE_ID}")
            print(f"  lookback_steps                  = {LOCKED_LOOKBACK_STEPS}")
            print(f"  FINAL_REFIT_EPOCHS              = {LOCKED_FINAL_REFIT_EPOCHS}")
            print(f"  seed                            = {LOCKED_SEED}")
            print(f"  final_lock_sha256               = {LOCKED_FINAL_LOCK_SHA256}")
            print(f"  config_fingerprint              = {LOCKED_CONFIG_FINGERPRINT}")
            print(f"  X FINAL_SCALING-v1 SHA          = {x_sha}")
            print(f"  Y FINAL_SCALING-v1 SHA          = {y_sha}")
            print(f"  FINAL_DEV population fingerprint = {final_dev_manifest['population_fingerprint']}")
            print(f"  source best_checkpoint.pt SHA   = {evidence['model_state_sha256']}")
            print(f"  source rmse_wh                  = {evidence['status']['best_validation_rmse_wh']}")
            print()
            print(f"  WOULD WRITE: {final_pt_path.relative_to(ROOT)}")
            print(f"  WOULD WRITE: {legacy_meta_path.relative_to(ROOT)}")
            print()
            print("[DRY-RUN] All gates PASS; no files written.")
            return 0

        report = finalize_seed42(force=args.force)
        print(json.dumps(report, indent=2))
        return 0
    except Seed42RecoveryContractViolation as exc:
        print(f"[FATAL] Contract violation: {exc}")
        return 4
    except Seed42RecoveryTrainingAttempt as exc:
        print(f"[FATAL] Training attempt detected: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
