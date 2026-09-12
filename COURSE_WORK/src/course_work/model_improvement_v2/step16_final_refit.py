"""Step 16: V2 final-refit lock and Human-gated three-seed execution.

``preflight`` is read-only. ``official`` and ``resume-partial`` are training
modes and refuse to run without an explicit Human authorization flag. No mode
loads or evaluates Test data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from course_work.data.feature_sets import get_feature_list
from course_work.experiments.registry import compute_config_fingerprint
from course_work.final_model_lock.epoch_policy import derive_final_epoch
from course_work.model_improvement_v2.e14_runner import bundle_config as e14_bundle_config
from course_work.model_improvement_v2.hybrid_loss import compute_hybrid_loss
from course_work.model_improvement_v2.optimizer import build_optimizer
from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset
from course_work.rolling_origin.populations import compute_population_fingerprint
from course_work.rolling_origin.scaling import FoldLocalScalerBundle, build_bundle
from course_work.training.engine import build_model_from_run_config
from course_work.utils.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from course_work.utils.environment import select_device
from course_work.utils.reproducibility import build_torch_generator, seed_worker, set_seed


EXPERIMENT_ID = "STEP16"
FINALIST_ID = "TR_C2_ALT_LOOKBACK_E14_M1"
FINAL_POLICY_ID = "MEAN_E14_M1_SEEDS_42_123_2026"
FINAL_SEEDS = (42, 123, 2026)
ENSEMBLE_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
FEATURE_VARIANT_ID = "FS2_TF1"
FEATURE_COUNT = 33
FINAL_REFIT_EPOCHS = 16
EPOCH_POLICY_ID = "MEDIAN_RO_INNER_BEST_EPOCHS-v1"
FINAL_DEV_VERSION = "V2_FINAL_DEV_REGION-v1"
SCALING_VERSION = "V2_FINAL_SCALING-v1"
DATALOADER_VERSION = "V2_FINAL_REFIT_DATALOADER-v1"
DATALOADER_FINGERPRINT = "ab0acbeb791ad4ad83c4cc362397f70b02f7a42efff36b88d6930ed2867cb604"
CONFIG_PATH = Path("artifacts/model_improvement_v2/final_model_lock/step16_final_refit_config_snapshot.json")
PLAN_PATH = Path("docs/plan/plan_before_process/model_improvement_v2_step16_final_refit_lock_pre_process_plan.md")
ARTIFACT_ROOT = Path("artifacts/model_improvement_v2/final_model_lock")
E14_ROOT = Path("artifacts/model_improvement_v2/experiments/E14")
E20_ROOT = Path("artifacts/model_improvement_v2/experiments/E20")
STAGE_A_SOURCE_RUNS = {
    "RO1": "RUN_V2_TR_E14_RO1_A_0001_B5DD2C2E",
    "RO2": "RUN_V2_TR_E14_RO2_A_0006_EC2A3FCE",
    "RO3": "RUN_V2_TR_E14_RO3_A_0011_DFA1252E",
}
PASSTHROUGH_FEATURES = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend")


class Step16PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Step16PreflightError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Step16PreflightError(f"Expected JSON object: {path}")
    return value


def _records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise Step16PreflightError(f"Missing registry: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {str(row["run_id"]): row for row in rows}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    temporary.replace(path)


def _write_once_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
    expected = canonical_json_bytes(dict(payload))
    if path.exists():
        if path.read_bytes() != expected:
            raise Step16PreflightError(f"Refusing to overwrite drifted lock artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(expected)
    temporary.replace(path)


def _decision_audit(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    e14 = _read(root / E14_ROOT / "e14_human_decision.json")
    e15 = _read(root / "artifacts/model_improvement_v2/experiments/E15/e15_human_decision.json")
    e16 = _read(root / "artifacts/model_improvement_v2/experiments/E16/e16_human_decision.json")
    e20 = _read(root / E20_ROOT / "e20_matched_seed_comparison.json")
    governance = document.get("human_governance", {})
    if (
        e14.get("decision") != "PROMOTE_M1"
        or e14.get("accepted_incumbent") != FINALIST_ID
        or e15.get("decision") != "REJECT_E15"
        or e16.get("decision") != "REJECT_E16"
        or e16.get("e17_eligibility") != "DEFERRED"
        or e20.get("test_status") != "NOT_ACCESSED"
        or governance != {
            "e20_finalist_confirmation": "REJECT",
            "step14a_seed_ensemble": "SELECT_FINALIST_ENSEMBLE",
            "step14b_persistence_blend": "KEEP_NEURAL_ENSEMBLE",
            "test_status": "NOT_ACCESSED",
        }
    ):
        raise Step16PreflightError("E14-Step14B Human/scientific decision chain mismatch")
    return {"status": "PASS", "active_policy": FINAL_POLICY_ID, "test_status": "NOT_ACCESSED"}


def _epoch_source_audit(root: Path) -> dict[str, Any]:
    registry = _records(root / E14_ROOT / "registry/experiment_registry.jsonl")
    epochs: dict[str, int] = {}
    checkpoint_sha: dict[str, str] = {}
    for fold, run_id in STAGE_A_SOURCE_RUNS.items():
        record = registry.get(run_id)
        if record is None or record.get("status") != "COMPLETED":
            raise Step16PreflightError(f"Missing completed E14-M1 Stage-A evidence: {fold}")
        if record.get("candidate_id") != FINALIST_ID or record.get("sweep_stage") != f"{fold}_A":
            raise Step16PreflightError(f"E14-M1 Stage-A identity mismatch: {fold}")
        epoch = record.get("best_epoch")
        if not isinstance(epoch, int) or epoch <= 0:
            raise Step16PreflightError(f"Invalid E14-M1 best epoch: {fold}")
        checkpoint = root / E14_ROOT / "runs" / run_id / "checkpoints/best_checkpoint.pt"
        artifacts = [item for item in record.get("artifacts", []) if Path(str(item.get("artifact_path", ""))).name == "best_checkpoint.pt"]
        if not checkpoint.is_file() or len(artifacts) != 1 or _sha(checkpoint) != artifacts[0].get("sha256"):
            raise Step16PreflightError(f"E14-M1 Stage-A checkpoint lineage mismatch: {fold}")
        epochs[fold] = epoch
        checkpoint_sha[fold] = _sha(checkpoint)
    decision = derive_final_epoch(
        {"recommended_transformer_inner_best_epochs": epochs, "recommended_transformer_stage_a_run_ids": STAGE_A_SOURCE_RUNS},
        locked_candidate_max_epochs=50,
    )
    if decision.FINAL_REFIT_EPOCHS != FINAL_REFIT_EPOCHS:
        raise Step16PreflightError("Final-refit epoch policy drift")
    return {
        "status": "PASS",
        "policy": decision.aggregation_rule,
        "fold_best_epochs": epochs,
        "sorted_epochs": list(decision.sorted_epochs),
        "final_refit_epochs": decision.FINAL_REFIT_EPOCHS,
        "same_epoch_all_seeds": True,
        "source_run_ids": dict(STAGE_A_SOURCE_RUNS),
        "source_checkpoint_sha256": checkpoint_sha,
        "test_dependency": False,
    }


@dataclass(frozen=True)
class FinalScalerState:
    bundle: FoldLocalScalerBundle
    x_payload: dict[str, Any]
    y_payload: dict[str, Any]
    x_sha256: str
    y_sha256: str
    bundle_sha256: str
    population_fingerprint: str
    target_count: int


def fit_final_dev_scaler(dataset: Any) -> FinalScalerState:
    records = dataset.window_records
    target_ids = records["target_id"].astype(str).tolist()
    timeline_targets = records["timeline_target"].astype(int).to_numpy()
    if len(target_ids) != 16630 or len(set(target_ids)) != len(target_ids):
        raise Step16PreflightError("FINAL_DEV target population/count mismatch")
    features = list(get_feature_list(FEATURE_VARIANT_ID))
    scaled = [index for index, name in enumerate(features) if name not in PASSTHROUGH_FEATURES]
    passthrough = [index for index, name in enumerate(features) if name in PASSTHROUGH_FEATURES]
    if [features[index] for index in passthrough] != list(PASSTHROUGH_FEATURES):
        raise Step16PreflightError("FINAL_SCALING-v1 passthrough feature order drift")
    x_fit = np.asarray(dataset._feature_matrix[timeline_targets], dtype=np.float64)
    y_fit = np.asarray(dataset._target_values[timeline_targets], dtype=np.float64)
    scaled_values = x_fit[:, scaled]
    means = scaled_values.mean(axis=0)
    stds = np.where(scaled_values.std(axis=0) > 1e-12, scaled_values.std(axis=0), 1.0)
    y_mean = float(y_fit.mean())
    y_std = float(y_fit.std()) if float(y_fit.std()) > 1e-12 else 1.0
    population_fp = compute_population_fingerprint(target_ids)
    bundle = build_bundle(
        bundle_id="V2_STEP16_FINAL_DEV_SCALER__FS2_TF1__YS1",
        fit_stage="FINAL_REFIT",
        fold_id="FINAL_DEV",
        candidate_id=FINALIST_ID,
        target_scaling_option="YS1",
        feature_variant_id=FEATURE_VARIANT_ID,
        lookback_steps=72,
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        revin_enabled=False,
        fit_target_ids=target_ids,
        fit_raw_row_count=len(target_ids),
        x_means=means,
        x_stds=stds,
        feature_indices_scaled=scaled,
        feature_indices_passthrough=passthrough,
        y_mean=y_mean,
        y_std=y_std,
    )
    x_payload = {
        "schema": "MODEL_IMPROVEMENT_V2_FINAL_X_SCALER-v1",
        "scaling_version": SCALING_VERSION,
        "fit_region": FINAL_DEV_VERSION,
        "feature_variant_id": FEATURE_VARIANT_ID,
        "feature_order": features,
        "fit_target_count": len(target_ids),
        "fit_target_ids_fingerprint": population_fp,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "scaled_indices": scaled,
        "passthrough_indices": passthrough,
        "test_rows_used": 0,
    }
    y_payload = {
        "schema": "MODEL_IMPROVEMENT_V2_FINAL_Y_SCALER-v1",
        "scaling_version": SCALING_VERSION,
        "fit_region": FINAL_DEV_VERSION,
        "target": "Appliances",
        "target_scaling_option": "YS1",
        "fit_target_count": len(target_ids),
        "fit_target_ids_fingerprint": population_fp,
        "mean": y_mean,
        "std": y_std,
        "test_rows_used": 0,
    }
    return FinalScalerState(
        bundle=bundle,
        x_payload=x_payload,
        y_payload=y_payload,
        x_sha256=sha256_bytes(canonical_json_bytes(x_payload)),
        y_sha256=sha256_bytes(canonical_json_bytes(y_payload)),
        bundle_sha256=bundle.checksum(),
        population_fingerprint=population_fp,
        target_count=len(target_ids),
    )


def final_refit_config(root: Path, seed: int, scaler: FinalScalerState) -> dict[str, Any]:
    if seed not in FINAL_SEEDS:
        raise Step16PreflightError(f"Unregistered final seed: {seed}")
    config = deepcopy(e14_bundle_config(root, "M1"))
    config["data"].update(target_access_mode="FINAL_DEV", train_sample_count=scaler.target_count, validation_sample_count=0, final_dev_sample_count=scaler.target_count)
    config["training"].update(max_epochs=FINAL_REFIT_EPOCHS, early_stopping_enabled=False, early_stopping_patience=0, final_refit_mode=True, validation_used=False)
    for key in ("seed", "global_seed", "dataloader_seed"):
        config["reproducibility"][key] = seed
    config["lineage"].update(
        population_version=FINAL_DEV_VERSION,
        population_fingerprint=scaler.population_fingerprint,
        final_dev_population_fingerprint=scaler.population_fingerprint,
        scaling_version=SCALING_VERSION,
        scaler_bundle_id="V2_STEP16_FINAL_DEV_X__FS2_TF1",
        scaler_bundle_checksum=scaler.x_sha256,
        target_scaler_bundle_id="V2_STEP16_FINAL_DEV_Y__YS1",
        target_scaler_checksum=scaler.y_sha256,
        final_scaler_bundle_checksum=scaler.bundle_sha256,
        final_refit_epoch_policy=EPOCH_POLICY_ID,
        final_refit_source_candidate_id=FINALIST_ID,
        dataloader_version=DATALOADER_VERSION,
        dataloader_fingerprint=DATALOADER_FINGERPRINT,
    )
    return config


def _source_checkpoint_audit(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    expected = document.get("ensemble_source_checkpoints", {})
    if set(expected) != {str(seed) for seed in FINAL_SEEDS}:
        raise Step16PreflightError("Step14-A source checkpoint set mismatch")
    for seed_text, folds in expected.items():
        if set(folds) != {"RO1", "RO2", "RO3"}:
            raise Step16PreflightError(f"Seed {seed_text} fold source set mismatch")
        for fold, source in folds.items():
            path = root / str(source["checkpoint_path"])
            if not path.is_file() or sha256_file(path) != source.get("checkpoint_sha256"):
                raise Step16PreflightError(f"Locked source checksum mismatch: seed={seed_text}/{fold}")
    return {"status": "PASS", "source_count": 9, "reuse_as_final_models": False, "purpose": "DEVELOPMENT_POLICY_AND_EPOCH_EVIDENCE_ONLY"}


def _assert_scientific_config(config: Mapping[str, Any], seed: int) -> None:
    data, model, training, reproducibility = (config[name] for name in ("data", "model", "training", "reproducibility"))
    expected_model = {"d_model": 64, "num_heads": 4, "ffn_dim": 256, "num_layers": 2, "norm_first": False, "dropout": 0.1, "input_size": 33}
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise Step16PreflightError("E14-M1 model configuration drift")
    if any((data.get("feature_variant_id") != FEATURE_VARIANT_ID, data.get("feature_count") != 33, data.get("lookback_steps") != 72, data.get("horizon_steps") != 1, data.get("boundary_protocol") != "WB0_CONTEXT_CARRY_OVER")):
        raise Step16PreflightError("E14-M1 data configuration drift")
    expected_training = {"optimizer_name": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-3, "batch_size": 16, "gradient_clip_max_norm": 1.0, "scheduler_name": None, "loss_policy": "HYBRID_LEVEL_PLUS_DELTA", "lambda_delta": 0.1, "delta_beta_model_space": 1.0, "max_epochs": 16, "early_stopping_enabled": False, "validation_used": False}
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise Step16PreflightError("Final-refit training configuration drift")
    if any(reproducibility.get(key) != seed for key in ("seed", "global_seed", "dataloader_seed")):
        raise Step16PreflightError("Per-run seed lock drift")


def validate_contract(document: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if document.get("schema") != "MODEL_IMPROVEMENT_V2_STEP16_FINAL_REFIT_CONTRACT-v1":
        raise Step16PreflightError("Wrong Step16 contract schema")
    if document.get("source_plan_sha256") != sha256_file(root / PLAN_PATH):
        raise Step16PreflightError("Step16 source plan checksum drift")
    expected_policy = {"policy_id": FINAL_POLICY_ID, "candidate_id": FINALIST_ID, "seeds": list(FINAL_SEEDS), "weights": list(ENSEMBLE_WEIGHTS)}
    if document.get("final_prediction_policy") != expected_policy:
        raise Step16PreflightError("Final prediction policy drift")
    if document.get("final_refit_epoch_policy", {}).get("epochs_by_seed") != {str(seed): FINAL_REFIT_EPOCHS for seed in FINAL_SEEDS}:
        raise Step16PreflightError("Final-refit seed epoch lock mismatch")
    if document.get("training_authorized") is not False or document.get("test_status") != "NOT_ACCESSED":
        raise Step16PreflightError("Training/Test governance lock mismatch")
    decision = _decision_audit(root, document)
    epoch_audit = _epoch_source_audit(root)
    checkpoint_audit = _source_checkpoint_audit(root, document)
    feature_order = tuple(get_feature_list(FEATURE_VARIANT_ID))
    if len(feature_order) != FEATURE_COUNT:
        raise Step16PreflightError("FS2_TF1 feature count drift")
    dataset, _folds, load_audit = build_v2_pretest_dataset(root, feature_order, experiment_id=EXPERIMENT_ID, feature_variant_id=FEATURE_VARIANT_ID)
    if load_audit.test_rows_read or load_audit.test_target_ids_seen:
        raise PermissionError("Step16 Test firewall breached")
    scaler = fit_final_dev_scaler(dataset)
    actual_scaler = {"x_scaler_sha256": scaler.x_sha256, "y_scaler_sha256": scaler.y_sha256, "bundle_sha256": scaler.bundle_sha256}
    if any(document.get("final_scaler_lock", {}).get(key) != value for key, value in actual_scaler.items()):
        raise Step16PreflightError("Deterministic FINAL_DEV scaler checksum drift")
    fingerprints = {}
    for seed in FINAL_SEEDS:
        config = final_refit_config(root, seed, scaler)
        _assert_scientific_config(config, seed)
        fingerprints[str(seed)] = compute_config_fingerprint(config)
    if document.get("final_refit_config_fingerprints") != fingerprints:
        raise Step16PreflightError("Final-refit config fingerprint drift")
    if document.get("expected_new_runs") != 3:
        raise Step16PreflightError("Step16 must create exactly three final-refit runs")
    ledger = audit_run_ledger(root, scaler)
    return {
        "experiment": EXPERIMENT_ID,
        "status": "PASS",
        "decision_chain": decision,
        "candidate_id": FINALIST_ID,
        "seeds": list(FINAL_SEEDS),
        "weights": list(ENSEMBLE_WEIGHTS),
        "epoch_policy": epoch_audit,
        "epochs_by_seed": {str(seed): FINAL_REFIT_EPOCHS for seed in FINAL_SEEDS},
        "data_population": {"status": "PASS", "region": FINAL_DEV_VERSION, "included_splits": ["TRAIN", "VALIDATION"], "excluded_splits": ["TEST"], "target_count": len(dataset), "target_ids_fingerprint": scaler.population_fingerprint, "first_target_id": str(dataset.window_records["target_id"].iloc[0]), "last_target_id": str(dataset.window_records["target_id"].iloc[-1])},
        "scaler_policy": {"status": "PASS", "version": SCALING_VERSION, "fit_region": FINAL_DEV_VERSION, "fit_once_shared_across_seeds": True, "fit_target_count": scaler.target_count, "x_scaled_feature_count": len(scaler.bundle.feature_indices_scaled), "x_passthrough_feature_count": len(scaler.bundle.feature_indices_passthrough), "passthrough_features": list(PASSTHROUGH_FEATURES), **actual_scaler},
        "source_checkpoint_audit": checkpoint_audit,
        "final_refit_config_fingerprints": fingerprints,
        "checkpoint_sha256_policy": "LOCK_AFTER_EACH_SUCCESSFUL_FINAL_REFIT",
        "fold_specific_stage_b_checkpoints_reused_as_final_models": False,
        "fresh_model_each_seed": True,
        "fresh_optimizer_each_seed": True,
        "scheduler": "OFF",
        "validation_used": False,
        "early_stopping": False,
        "recovery_contract": "VERIFY_COMPLETED_AND_TRAIN_ONLY_MISSING_SEEDS",
        "run_ledger": {
            "status": ledger["status"],
            "completed_seeds": sorted(ledger["completed"]),
            "missing_seeds": ledger["missing_seeds"],
            "failed_evidence_preserved": ledger["failed_preserved"],
            "running": ledger["running"],
        },
        "expected_new_runs": 3,
        "test_rows_read": 0,
        "test_target_ids_seen": 0,
        "test_status": "NOT_ACCESSED",
        "training_executed": False,
        "training_authorized": False,
    }


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    return validate_contract(_read(root / CONFIG_PATH), root)


class _ScaledFinalDevDataset(Dataset):
    def __init__(self, base: Any, scaler: FinalScalerState) -> None:
        self.base = base
        self.scaler = scaler
        self.feature_matrix = scaler.bundle.transform_x(base._feature_matrix)
        self.target_values = np.asarray(base._target_values, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.base.window_records.iloc[int(index)]
        start, end, target = (int(record[name]) for name in ("timeline_input_start", "timeline_input_end", "timeline_target"))
        if end != target - 1:
            raise RuntimeError("Step16 context is not strictly past-only")
        y_raw = float(self.target_values[target])
        y_model = float(self.scaler.bundle.transform_y(np.asarray([y_raw]))[0])
        return {"x": torch.from_numpy(self.feature_matrix[start : end + 1].copy()), "y_model": torch.tensor([y_model], dtype=torch.float32), "y_raw_wh": torch.tensor([y_raw], dtype=torch.float32), "y_context_raw_wh": torch.tensor([self.target_values[end]], dtype=torch.float32), "sample_idx": torch.tensor(int(record["canonical_sample_idx"]), dtype=torch.int64)}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _run_directories(root: Path) -> list[Path]:
    run_root = root / ARTIFACT_ROOT / "runs"
    return sorted(path for path in run_root.glob("RUN_V2_TR_STEP16_SEED*_FINAL_A*") if path.is_dir()) if run_root.exists() else []


def audit_run_ledger(root: Path, scaler: FinalScalerState) -> dict[str, Any]:
    completed: dict[int, dict[str, Any]] = {}
    failed: list[str] = []
    running: list[str] = []
    for run_dir in _run_directories(root):
        status = _read(run_dir / "status.json")
        seed = int(status.get("seed", -1))
        if seed not in FINAL_SEEDS:
            raise Step16PreflightError(f"Unexpected Step16 seed record: {run_dir.name}")
        state = status.get("status")
        if state == "COMPLETED":
            if seed in completed:
                raise Step16PreflightError(f"Duplicate completed Step16 seed: {seed}")
            config_doc = _read(run_dir / "config.json")
            expected_fp = compute_config_fingerprint(final_refit_config(root, seed, scaler))
            checkpoint = run_dir / "checkpoints/final_refit.pt"
            history = run_dir / "training_history.csv"
            rows = list(csv.DictReader(history.open())) if history.is_file() else []
            if config_doc.get("config_fingerprint") != expected_fp or status.get("config_fingerprint") != expected_fp or status.get("official_epoch") != FINAL_REFIT_EPOCHS or not checkpoint.is_file() or status.get("checkpoint_sha256") != sha256_file(checkpoint) or status.get("x_scaler_sha256") != scaler.x_sha256 or status.get("y_scaler_sha256") != scaler.y_sha256 or len(rows) != FINAL_REFIT_EPOCHS or status.get("test_status") != "NOT_ACCESSED":
                raise Step16PreflightError(f"Completed Step16 evidence failed reuse audit: {run_dir.name}")
            completed[seed] = status
        elif state == "FAILED":
            failed.append(run_dir.name)
        elif state == "RUNNING":
            running.append(run_dir.name)
        else:
            raise Step16PreflightError(f"Unsupported Step16 lifecycle state: {state}")
    if running:
        raise Step16PreflightError(f"RUNNING Step16 records require interruption audit: {running}")
    return {"status": "PASS", "completed": completed, "failed_preserved": failed, "running": running, "missing_seeds": [seed for seed in FINAL_SEEDS if seed not in completed]}


def _next_run_id(root: Path, seed: int, fingerprint: str) -> str:
    attempts = [path for path in _run_directories(root) if f"SEED{seed}_" in path.name]
    return f"RUN_V2_TR_STEP16_SEED{seed}_FINAL_A{len(attempts) + 1:02d}_{fingerprint[:8].upper()}"


def _train_seed(root: Path, base_dataset: Any, scaler: FinalScalerState, seed: int) -> dict[str, Any]:
    config = final_refit_config(root, seed, scaler)
    fingerprint = compute_config_fingerprint(config)
    run_id = _next_run_id(root, seed, fingerprint)
    run_dir = root / ARTIFACT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "config.json", {"run_id": run_id, "config_fingerprint": fingerprint, "config": config})
    _atomic_json(run_dir / "status.json", {"run_id": run_id, "seed": seed, "status": "RUNNING", "created_at_unix": time.time(), "test_status": "NOT_ACCESSED"})
    try:
        set_seed(seed)
        model = build_model_from_run_config(config)
        device = select_device()
        if str(device) != str(config["runtime"]["device_type"]):
            raise Step16PreflightError(f"Runtime device drift: selected={device} locked={config['runtime']['device_type']}")
        model = model.to(device)
        optimizer = build_optimizer(model.parameters(), "AdamW", 2e-4, 1e-3, None)
        loader = DataLoader(_ScaledFinalDevDataset(base_dataset, scaler), batch_size=16, shuffle=True, num_workers=0, pin_memory=False, generator=build_torch_generator(seed), worker_init_fn=seed_worker)
        history: list[dict[str, Any]] = []
        log_lines = [f"run_id={run_id}", f"seed={seed}", f"device={device}"]
        target_bundle = scaler.bundle.as_dict()
        for epoch in range(1, FINAL_REFIT_EPOCHS + 1):
            model.train()
            totals = {name: 0.0 for name in ("level_mse", "delta_smooth_l1", "weighted_delta_loss", "total_loss")}
            samples = 0
            started = time.time()
            for batch in loader:
                x, y = batch["x"].to(device), batch["y_model"].to(device)
                optimizer.zero_grad(set_to_none=True)
                components = compute_hybrid_loss(model(x), y, batch["y_context_raw_wh"], target_bundle, config["training"])
                components.total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                count = int(x.shape[0])
                samples += count
                for name in totals:
                    totals[name] += float(getattr(components, name).detach().item()) * count
            row = {"epoch": epoch, **{name: value / samples for name, value in totals.items()}, "learning_rate": 2e-4, "epoch_seconds": time.time() - started}
            history.append(row)
            line = f"epoch={epoch}/{FINAL_REFIT_EPOCHS} total_loss={row['total_loss']:.8f}"
            log_lines.append(line)
            print(f"[STEP16 seed={seed}] {line}", flush=True)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / "final_refit.pt"
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save({"schema": "MODEL_IMPROVEMENT_V2_STEP16_FINAL_CHECKPOINT-v1", "run_id": run_id, "seed": seed, "official_epoch": FINAL_REFIT_EPOCHS, "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "config": config, "config_fingerprint": fingerprint, "x_scaler_sha256": scaler.x_sha256, "y_scaler_sha256": scaler.y_sha256, "population_fingerprint": scaler.population_fingerprint, "test_status": "NOT_ACCESSED"}, temporary)
        temporary.replace(checkpoint)
        fields = ["epoch", "level_mse", "delta_smooth_l1", "weighted_delta_loss", "total_loss", "learning_rate", "epoch_seconds"]
        _write_csv(run_dir / "training_history.csv", fields, history)
        (run_dir / "training.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        result = {"run_id": run_id, "seed": seed, "status": "COMPLETED", "official_epoch": FINAL_REFIT_EPOCHS, "config_fingerprint": fingerprint, "checkpoint_path": str(checkpoint.relative_to(root)), "checkpoint_sha256": sha256_file(checkpoint), "x_scaler_sha256": scaler.x_sha256, "y_scaler_sha256": scaler.y_sha256, "test_status": "NOT_ACCESSED"}
        _atomic_json(run_dir / "status.json", result)
        return result
    except BaseException as exc:
        _atomic_json(run_dir / "status.json", {"run_id": run_id, "seed": seed, "status": "FAILED", "error": repr(exc), "test_status": "NOT_ACCESSED"})
        raise


def run_training(mode: str, root: Path | None = None) -> int:
    root = root or project_root()
    preflight = run_preflight(root)
    dataset, _folds, load_audit = build_v2_pretest_dataset(root, tuple(get_feature_list(FEATURE_VARIANT_ID)), experiment_id=EXPERIMENT_ID, feature_variant_id=FEATURE_VARIANT_ID)
    if load_audit.test_rows_read or load_audit.test_target_ids_seen:
        raise PermissionError("Step16 Test firewall breached")
    scaler = fit_final_dev_scaler(dataset)
    ledger = audit_run_ledger(root, scaler)
    if mode == "official" and (ledger["completed"] or ledger["failed_preserved"]):
        raise Step16PreflightError("official requires an empty Step16 namespace; use resume-partial after audit")
    output = root / ARTIFACT_ROOT
    _write_once_or_verify(output / "scalers/final_x_scaler.json", scaler.x_payload)
    _write_once_or_verify(output / "scalers/final_y_scaler.json", scaler.y_payload)
    for seed in ledger["missing_seeds"]:
        _train_seed(root, dataset, scaler, seed)
    final_ledger = audit_run_ledger(root, scaler)
    if final_ledger["missing_seeds"]:
        raise Step16PreflightError(f"Final-refit remains incomplete: {final_ledger['missing_seeds']}")
    runs = [final_ledger["completed"][seed] for seed in FINAL_SEEDS]
    manifest = {"schema": "MODEL_IMPROVEMENT_V2_STEP16_FINAL_REFIT_MANIFEST-v1", "preflight_contract_sha256": sha256_file(root / CONFIG_PATH), "runs": runs, "ensemble_policy": {"seeds": list(FINAL_SEEDS), "weights": list(ENSEMBLE_WEIGHTS)}, "all_checkpoint_sha256_locked": True, "failed_evidence_preserved": final_ledger["failed_preserved"], "test_status": "NOT_ACCESSED"}
    _write_once_or_verify(output / "step16_final_refit_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=[EXPERIMENT_ID], required=True)
    parser.add_argument("--mode", choices=["preflight", "official", "resume-partial"], default="preflight")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "preflight" and args.authorize_training:
        print("REFUSED: preflight must not carry --authorize-training", file=sys.stderr)
        return 2
    if args.mode in {"official", "resume-partial"} and not args.authorize_training:
        print("REFUSED: Step16 training requires explicit --authorize-training", file=sys.stderr)
        return 3
    try:
        if args.mode in {"official", "resume-partial"}:
            return run_training(args.mode)
        print(json.dumps(run_preflight(), indent=2, sort_keys=True))
        return 0
    except (Step16PreflightError, PermissionError, ValueError) as exc:
        print(f"STEP16 {args.mode.upper()} FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
