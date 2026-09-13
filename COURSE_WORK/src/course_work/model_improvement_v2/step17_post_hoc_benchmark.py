"""Human-gated Step 17 post-hoc benchmark for the locked V2 ensemble.

Preflight is strictly metadata/checksum-only: it does not open Test sources,
load checkpoint payloads, run inference, or write scientific results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from course_work.data.feature_sets import get_feature_list
from course_work.evaluation.metrics import compute_mape_pct
from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.pretest_adapter import _open_feature_source
from course_work.training.engine import build_model_from_run_config
from course_work.utils.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from course_work.utils.environment import select_device


EXPERIMENT_ID = "STEP17"
BENCHMARK_LABEL = "POST_HOC_V2_BENCHMARK"
FINAL_LOCK_PATH = Path("artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json")
CONFIG_PATH = Path("artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_benchmark_config_snapshot.json")
OUTPUT_ROOT = Path("artifacts/model_improvement_v2/post_hoc_v2_benchmark")
SPLIT_MANIFEST_PATH = Path("artifacts/splits/split_manifest.json")
FINAL_SEEDS = (42, 123, 2026)
ENSEMBLE_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
FEATURE_VARIANT_ID = "FS2_TF1"
LOOKBACK = 72
EXPECTED_TEST_COUNT = 2961
EXPECTED_TEST_POPULATION_FINGERPRINT = "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"


class Step17Error(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Step17Error(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Step17Error(f"Expected JSON object: {path}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(value)))
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise Step17Error(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _lock_fingerprint(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("lock_fingerprint", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _assert_locked_scientific_config(config: Mapping[str, Any], seed: int) -> None:
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    reproducibility = config.get("reproducibility", {})
    expected_model = {
        "model_family": "TRANSFORMER_ENCODER",
        "input_size": 33,
        "d_model": 64,
        "num_heads": 4,
        "ffn_dim": 256,
        "num_layers": 2,
        "norm_first": False,
        "dropout": 0.1,
    }
    expected_training = {
        "optimizer_name": "AdamW",
        "learning_rate": 2e-4,
        "weight_decay": 1e-3,
        "batch_size": 16,
        "scheduler_name": None,
        "loss_policy": "HYBRID_LEVEL_PLUS_DELTA",
        "lambda_delta": 0.1,
        "delta_beta_model_space": 1.0,
        "max_epochs": 16,
        "validation_used": False,
        "early_stopping_enabled": False,
    }
    expected_data = {
        "feature_variant_id": FEATURE_VARIANT_ID,
        "feature_count": 33,
        "lookback_steps": LOOKBACK,
        "horizon_steps": 1,
        "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
    }
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise Step17Error(f"Locked model config drift for seed {seed}")
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise Step17Error(f"Locked training config drift for seed {seed}")
    if any(data.get(key) != value for key, value in expected_data.items()):
        raise Step17Error(f"Locked data config drift for seed {seed}")
    if any(reproducibility.get(key) != seed for key in ("seed", "global_seed", "dataloader_seed")):
        raise Step17Error(f"Locked seed config drift for seed {seed}")


def validate_preflight(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "MODEL_IMPROVEMENT_V2_STEP17_POST_HOC_BENCHMARK_CONTRACT-v1":
        raise Step17Error("Wrong Step 17 contract schema")
    if contract.get("benchmark_label") != BENCHMARK_LABEL:
        raise Step17Error("Mandatory post-hoc benchmark label drift")
    if contract.get("test_access_authorized") is not False or contract.get("benchmark_authorized") is not False:
        raise Step17Error("Preparation snapshot must not authorize benchmark/Test access")
    if contract.get("training_authorized") is not False:
        raise Step17Error("Step 17 must never authorize training")

    lock_spec = contract.get("final_lock", {})
    lock_path = root / str(lock_spec.get("path"))
    if not lock_path.is_file() or sha256_file(lock_path) != lock_spec.get("file_sha256"):
        raise Step17Error("V2 final-lock file checksum mismatch")
    lock = _read_json(lock_path)
    if (
        lock.get("lock_status") != "LOCKED"
        or lock.get("step17_ready") is not True
        or lock.get("test_status") != "NOT_ACCESSED"
        or _lock_fingerprint(lock) != lock_spec.get("lock_fingerprint")
        or lock.get("lock_fingerprint") != lock_spec.get("lock_fingerprint")
    ):
        raise Step17Error("V2 final-lock state/fingerprint mismatch")

    expected_policy = {
        "policy_id": "MEAN_E14_M1_SEEDS_42_123_2026",
        "candidate_id": "TR_C2_ALT_LOOKBACK_E14_M1",
        "seeds": list(FINAL_SEEDS),
        "weights": list(ENSEMBLE_WEIGHTS),
    }
    locked_policy = lock.get("final_prediction_policy", {})
    configured_policy = contract.get("prediction_policy", {})
    if any(locked_policy.get(key) != value for key, value in expected_policy.items()):
        raise Step17Error("Final prediction policy differs from V2 lock")
    if any(configured_policy.get(key) != value for key, value in expected_policy.items()):
        raise Step17Error("Step 17 prediction policy drift")
    if configured_policy.get("best_seed_selection") != "FORBIDDEN":
        raise Step17Error("Best-seed selection must remain forbidden")

    feature_order = get_feature_list(FEATURE_VARIANT_ID)
    if len(feature_order) != 33:
        raise Step17Error("Canonical FS2_TF1 feature count drift")
    scaler_spec = contract.get("scaler_contract", {})
    scaler_docs: dict[str, dict[str, Any]] = {}
    for kind in ("x", "y"):
        path = root / str(scaler_spec.get(f"{kind}_path"))
        expected_sha = scaler_spec.get(f"{kind}_sha256")
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise Step17Error(f"Locked {kind.upper()} scaler checksum mismatch")
        scaler_docs[kind] = _read_json(path)
    if scaler_docs["x"].get("feature_order") != feature_order:
        raise Step17Error("Locked X scaler feature order drift")
    if scaler_docs["x"].get("test_rows_used") != 0 or scaler_docs["y"].get("test_rows_used") != 0:
        raise Step17Error("Final scaler lineage includes Test")
    if scaler_spec.get("mode") != "TRANSFORM_ONLY" or scaler_spec.get("fit_allowed") is not False:
        raise Step17Error("Step 17 scaler must be transform-only")

    locked_runs = lock.get("final_runs", [])
    if len(locked_runs) != 3 or [row.get("seed") for row in locked_runs] != list(FINAL_SEEDS):
        raise Step17Error("Final checkpoint set is not the exact ordered three-seed lock")
    checkpoints: list[dict[str, Any]] = []
    for row in locked_runs:
        seed = int(row["seed"])
        checkpoint = root / str(row.get("checkpoint_path"))
        if not checkpoint.is_file() or sha256_file(checkpoint) != row.get("checkpoint_sha256"):
            raise Step17Error(f"Final checkpoint checksum mismatch for seed {seed}")
        run_dir = checkpoint.parents[1]
        config_wrapper = _read_json(run_dir / "config.json")
        status = _read_json(run_dir / "status.json")
        config = config_wrapper.get("config")
        if not isinstance(config, dict):
            raise Step17Error(f"Missing locked run config for seed {seed}")
        fingerprint = compute_config_fingerprint(config)
        if (
            fingerprint != row.get("config_fingerprint")
            or config_wrapper.get("config_fingerprint") != fingerprint
            or status.get("status") != "COMPLETED"
            or status.get("checkpoint_sha256") != row.get("checkpoint_sha256")
            or status.get("official_epoch") != 16
            or row.get("official_epoch") != 16
            or row.get("test_status") != "NOT_ACCESSED"
        ):
            raise Step17Error(f"Final run provenance mismatch for seed {seed}")
        _assert_locked_scientific_config(config, seed)
        checkpoints.append({
            "seed": seed,
            "run_id": row["run_id"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "config_fingerprint": fingerprint,
            "status": "PASS",
        })

    output = contract.get("output_contract", {})
    if output.get("root") != OUTPUT_ROOT.as_posix():
        raise Step17Error("Step 17 output namespace drift")
    data = contract.get("data_contract", {})
    if (
        data.get("expected_target_count") != EXPECTED_TEST_COUNT
        or data.get("expected_first_target_id") != "TGT_00016774"
        or data.get("expected_last_target_id") != "TGT_00019734"
        or data.get("expected_population_fingerprint") != EXPECTED_TEST_POPULATION_FINGERPRINT
        or data.get("load_policy") != "ONE_SOURCE_LOAD_REUSED_IN_MEMORY"
        or data.get("shuffle") is not False
        or data.get("drop_last") is not False
    ):
        raise Step17Error("Step 17 Test population/loader contract drift")

    return {
        "experiment": EXPERIMENT_ID,
        "status": "PASS",
        "benchmark_label": BENCHMARK_LABEL,
        "old_test_interpretation": contract["old_test_interpretation"],
        "final_lock": {"status": "PASS", "path": str(lock_spec["path"]), "sha256": lock_spec["file_sha256"]},
        "checkpoints": {"status": "PASS", "count": 3, "records": checkpoints, "payloads_loaded": 0},
        "scalers": {"status": "PASS", "mode": "TRANSFORM_ONLY", "x_sha256": scaler_spec["x_sha256"], "y_sha256": scaler_spec["y_sha256"]},
        "prediction_policy": expected_policy,
        "feature_order": feature_order,
        "test_access_guard": {"status": "PASS", "test_files_opened": 0, "test_rows_read": 0, "test_target_ids_seen": 0},
        "training_executed": False,
        "inference_executed": False,
        "benchmark_authorized": False,
        "test_access_authorized": False,
    }


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    return validate_preflight(root, _read_json(root / CONFIG_PATH))


@dataclass(frozen=True)
class TestPopulation:
    target_ids: tuple[str, ...]
    timestamps: tuple[str, ...]
    feature_matrix_scaled: np.ndarray
    y_true_wh: np.ndarray
    persistence_wh: np.ndarray
    population_fingerprint: str
    source_kind: str


class TestAccessGuard:
    """One-shot loader guard; benchmark code receives no second Test loader."""

    def __init__(self, loader: Callable[[Path, Sequence[str], Mapping[str, Any]], TestPopulation]) -> None:
        self._loader = loader
        self.access_count = 0

    def load_once(self, root: Path, features: Sequence[str], contract: Mapping[str, Any]) -> TestPopulation:
        if self.access_count:
            raise PermissionError("Step 17 Test source may be loaded exactly once")
        self.access_count += 1
        return self._loader(root, features, contract)


def _population_fingerprint(target_ids: Sequence[str]) -> str:
    blob = json.dumps(sorted(str(value) for value in target_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_test_source_once(root: Path, feature_order: Sequence[str], contract: Mapping[str, Any]) -> TestPopulation:
    """Authorized path only: one CSV/zip-member open loads the complete timeline once."""
    split = _read_json(root / SPLIT_MANIFEST_PATH)
    total_rows = int(split["total_rows"])
    first_test = int(split["train_rows"]) + int(split["validation_rows"])
    selected = list(dict.fromkeys(["raw_row_index", "timestamp", "continuity_segment_id", *feature_order]))
    with _open_feature_source(root) as (stream, source_kind):
        frame = pd.read_csv(stream, usecols=selected)
    if len(frame) != total_rows:
        raise Step17Error("Canonical feature source row count drift")
    frame = frame.loc[:, selected]
    frame["raw_row_index"] = pd.to_numeric(frame["raw_row_index"], errors="raise").astype(np.int64)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["continuity_segment_id"] = frame["continuity_segment_id"].astype(str)
    for feature in feature_order:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")
    if frame["raw_row_index"].tolist() != list(range(total_rows)):
        raise Step17Error("Canonical feature-source raw row order drift")
    if not frame["timestamp"].is_monotonic_increasing:
        raise Step17Error("Canonical feature source is not chronological")
    test_start = pd.Timestamp(split["test_start_timestamp"])
    if frame["timestamp"].iloc[first_test] != test_start or bool(frame["timestamp"].iloc[:first_test].ge(test_start).any()):
        raise Step17Error("Frozen Test boundary mismatch")

    target_positions = np.arange(first_test, total_rows, dtype=np.int64)
    target_ids = tuple(f"TGT_{position:08d}" for position in target_positions)
    if len(target_ids) != EXPECTED_TEST_COUNT or target_ids[0] != "TGT_00016774" or target_ids[-1] != "TGT_00019734":
        raise Step17Error("Step 17 Test target population drift")
    timestamps = tuple(frame["timestamp"].iloc[target_positions].astype(str).tolist())
    segments = frame["continuity_segment_id"].to_numpy()
    for target in target_positions:
        start, end = int(target - LOOKBACK), int(target - 1)
        if start < 0 or end != target - 1 or len(set(segments[start : target + 1])) != 1:
            raise Step17Error(f"Invalid WB0 causal window at target {target}")
        deltas = frame["timestamp"].iloc[start : target + 1].diff().dropna()
        if not bool(deltas.eq(pd.Timedelta(minutes=10)).all()):
            raise Step17Error(f"Non-contiguous WB0 causal window at target {target}")

    x_scaler = _read_json(root / str(contract["scaler_contract"]["x_path"]))
    raw = frame.loc[:, feature_order].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(raw).all():
        raise Step17Error("Feature source contains non-finite values")
    scaled = raw.copy()
    indices = np.asarray(x_scaler["scaled_indices"], dtype=np.int64)
    means = np.asarray(x_scaler["means"], dtype=np.float64)
    stds = np.asarray(x_scaler["stds"], dtype=np.float64)
    if len(indices) != len(means) or len(means) != len(stds) or bool(np.any(stds <= 0)):
        raise Step17Error("Locked X scaler structure is invalid")
    scaled[:, indices] = (scaled[:, indices] - means) / stds
    target_values = frame["Appliances"].to_numpy(dtype=np.float64, copy=True)
    y_true = target_values[target_positions]
    persistence = target_values[target_positions - 1]
    fingerprint = _population_fingerprint(target_ids)
    if fingerprint != EXPECTED_TEST_POPULATION_FINGERPRINT:
        raise Step17Error("Step 17 ordered Test population fingerprint drift")
    return TestPopulation(target_ids, timestamps, scaled.astype(np.float32), y_true, persistence, fingerprint, source_kind)


class _WindowDataset(Dataset):
    def __init__(self, population: TestPopulation) -> None:
        self.population = population
        self.first_target = int(population.target_ids[0][4:])

    def __len__(self) -> int:
        return len(self.population.target_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        target = self.first_target + int(index)
        return torch.from_numpy(self.population.feature_matrix_scaled[target - LOOKBACK : target].copy())


def _load_locked_models(root: Path, lock: Mapping[str, Any]) -> list[tuple[dict[str, Any], torch.nn.Module]]:
    loaded: list[tuple[dict[str, Any], torch.nn.Module]] = []
    for record in lock["final_runs"]:
        path = root / record["checkpoint_path"]
        if sha256_file(path) != record["checkpoint_sha256"]:
            raise Step17Error(f"Checkpoint changed after preflight: seed {record['seed']}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema") != "MODEL_IMPROVEMENT_V2_STEP16_FINAL_CHECKPOINT-v1"
            or payload.get("run_id") != record["run_id"]
            or payload.get("seed") != record["seed"]
            or payload.get("official_epoch") != 16
            or payload.get("config_fingerprint") != record["config_fingerprint"]
            or payload.get("x_scaler_sha256") != record["x_scaler_sha256"]
            or payload.get("y_scaler_sha256") != record["y_scaler_sha256"]
            or payload.get("test_status") != "NOT_ACCESSED"
        ):
            raise Step17Error(f"Checkpoint payload provenance mismatch: seed {record['seed']}")
        config = payload.get("config")
        if not isinstance(config, dict) or compute_config_fingerprint(config) != record["config_fingerprint"]:
            raise Step17Error(f"Checkpoint embedded config mismatch: seed {record['seed']}")
        _assert_locked_scientific_config(config, int(record["seed"]))
        model = build_model_from_run_config(config)
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise Step17Error(f"Strict state-dict mismatch: seed {record['seed']}")
        model.eval()
        loaded.append((dict(record), model))
    return loaded


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape or not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise Step17Error("Metric population shape/finite check failed")
    error = true - pred
    denominator = float(np.square(true - true.mean()).sum())
    mape = compute_mape_pct(true, pred)
    return {
        "sample_count": int(len(true)),
        "rmse_wh": float(np.sqrt(np.mean(np.square(error)))),
        "mae_wh": float(np.mean(np.abs(error))),
        "mape_pct": float(mape.mape_pct),
        "mape_status": mape.mape_status,
        "r2": float(1.0 - np.square(error).sum() / denominator),
    }


def _predict(model: torch.nn.Module, population: TestPopulation, batch_size: int = 16) -> tuple[np.ndarray, float, int]:
    device = select_device()
    model = model.to(device)
    loader = DataLoader(_WindowDataset(population), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for x in loader:
            outputs.append(model(x.to(device)).detach().cpu().numpy().reshape(-1))
    elapsed = time.perf_counter() - started
    prediction_model = np.concatenate(outputs).astype(np.float64)
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    return prediction_model, elapsed, parameter_count


def _prediction_rows(population: TestPopulation, prediction: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "target_id": target_id,
            "target_timestamp": timestamp,
            "y_true_wh": float(true),
            "y_pred_wh": float(pred),
            "residual_wh": float(true - pred),
        }
        for target_id, timestamp, true, pred in zip(population.target_ids, population.timestamps, population.y_true_wh, prediction)
    ]


def run_benchmark(root: Path | None = None) -> int:
    root = root or project_root()
    contract = _read_json(root / CONFIG_PATH)
    preflight = validate_preflight(root, contract)
    access_event = root / OUTPUT_ROOT / contract["output_contract"]["access_event"]
    summary_path = root / OUTPUT_ROOT / contract["output_contract"]["summary"]
    if access_event.exists() or summary_path.exists():
        raise Step17Error("Step 17 Test was already opened or benchmark output exists; read-only recovery audit required")

    lock = _read_json(root / FINAL_LOCK_PATH)
    models = _load_locked_models(root, lock)
    _atomic_json(access_event, {
        "schema": "MODEL_IMPROVEMENT_V2_STEP17_TEST_ACCESS_EVENT-v1",
        "benchmark_label": BENCHMARK_LABEL,
        "authorization": "HUMAN_CLI_DOUBLE_ACKNOWLEDGEMENT",
        "old_test_unseen_claim": False,
        "created_at": _utc_now(),
    })
    guard = TestAccessGuard(_load_test_source_once)
    population = guard.load_once(root, preflight["feature_order"], contract)
    if guard.access_count != 1:
        raise Step17Error("Test source access count differs from exactly one")

    y_scaler = _read_json(root / contract["scaler_contract"]["y_path"])
    y_mean, y_std = float(y_scaler["mean"]), float(y_scaler["std"])
    seed_predictions: dict[int, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    prediction_checksums: dict[str, str] = {}
    for record, model in models:
        seed = int(record["seed"])
        prediction_model, runtime, parameter_count = _predict(model, population, batch_size=16)
        prediction = prediction_model * y_std + y_mean
        rows = _prediction_rows(population, prediction)
        path = root / OUTPUT_ROOT / "predictions" / f"seed_{seed}.csv"
        _write_csv(path, rows)
        prediction_checksums[str(path.relative_to(root))] = sha256_file(path)
        seed_predictions[seed] = prediction
        metric_rows.append({
            "model_id": f"V2_FINAL_SEED_{seed}",
            "seed": seed,
            "run_id": record["run_id"],
            **_metrics(population.y_true_wh, prediction),
            "parameter_count": parameter_count,
            "inference_runtime_seconds": runtime,
            "population_fingerprint": population.population_fingerprint,
        })

    ensemble = sum(weight * seed_predictions[seed] for weight, seed in zip(ENSEMBLE_WEIGHTS, FINAL_SEEDS))
    persistence = population.persistence_wh
    for model_id, prediction in (("V2_FINAL_EQUAL_WEIGHT_ENSEMBLE", ensemble), ("PERSISTENCE_LAST_VALUE", persistence)):
        rows = _prediction_rows(population, prediction)
        filename = "ensemble.csv" if model_id.startswith("V2_") else "persistence.csv"
        path = root / OUTPUT_ROOT / "predictions" / filename
        _write_csv(path, rows)
        prediction_checksums[str(path.relative_to(root))] = sha256_file(path)
        metric_rows.append({
            "model_id": model_id,
            "seed": None,
            "run_id": None,
            **_metrics(population.y_true_wh, prediction),
            "parameter_count": None,
            "inference_runtime_seconds": None,
            "population_fingerprint": population.population_fingerprint,
        })

    metrics_payload = {
        "schema": "MODEL_IMPROVEMENT_V2_STEP17_METRICS-v1",
        "benchmark_label": BENCHMARK_LABEL,
        "population_fingerprint": population.population_fingerprint,
        "metrics": metric_rows,
        "best_seed_selected": False,
        "ensemble_weights": list(ENSEMBLE_WEIGHTS),
    }
    metrics_path = root / OUTPUT_ROOT / contract["output_contract"]["metrics"]
    _atomic_json(metrics_path, metrics_payload)

    historical_spec = contract["evaluation_contract"]["v1_historical_reference"]
    historical_path = root / historical_spec["path"]
    historical = _read_json(historical_path)
    summary = {
        "schema": "MODEL_IMPROVEMENT_V2_STEP17_POST_HOC_BENCHMARK-v1",
        "benchmark_label": BENCHMARK_LABEL,
        "old_test_interpretation": contract["old_test_interpretation"],
        "final_policy": lock["final_prediction_policy"],
        "test_population": {
            "sample_count": len(population.target_ids),
            "first_target_id": population.target_ids[0],
            "last_target_id": population.target_ids[-1],
            "fingerprint": population.population_fingerprint,
            "source_load_count": guard.access_count,
        },
        "metrics": metric_rows,
        "v1_historical_reference": {
            "label": historical_spec["label"],
            "path": historical_spec["path"],
            "sha256": sha256_file(historical_path),
            "selection_use": "FORBIDDEN",
            "summary": historical,
        },
        "training_executed": False,
        "scaler_fit_executed": False,
        "best_seed_selected": False,
        "post_test_retuning": False,
        "test_status": "ACCESSED_ONCE_FOR_POST_HOC_V2_BENCHMARK",
        "created_at": _utc_now(),
    }
    _atomic_json(summary_path, summary)
    manifest = {
        "schema": "MODEL_IMPROVEMENT_V2_STEP17_MANIFEST-v1",
        "benchmark_label": BENCHMARK_LABEL,
        "final_lock_path": FINAL_LOCK_PATH.as_posix(),
        "final_lock_sha256": sha256_file(root / FINAL_LOCK_PATH),
        "checkpoint_sha256": {str(row["seed"]): row["checkpoint_sha256"] for row in lock["final_runs"]},
        "x_scaler_sha256": contract["scaler_contract"]["x_sha256"],
        "y_scaler_sha256": contract["scaler_contract"]["y_sha256"],
        "prediction_sha256": prediction_checksums,
        "metrics_sha256": sha256_file(metrics_path),
        "summary_sha256": sha256_file(summary_path),
        "test_source_load_count": guard.access_count,
        "status": "PASS",
    }
    manifest_path = root / OUTPUT_ROOT / contract["output_contract"]["manifest"]
    _atomic_json(manifest_path, manifest)
    _atomic_json(root / OUTPUT_ROOT / contract["output_contract"]["signoff"], {
        "schema": "MODEL_IMPROVEMENT_V2_STEP17_SIGNOFF-v1",
        "benchmark_label": BENCHMARK_LABEL,
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "human_interpretation_required": True,
        "unbiased_unseen_test_claim": False,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 17 locked post-hoc V2 benchmark")
    parser.add_argument("--experiment", required=True, choices=[EXPERIMENT_ID])
    parser.add_argument("--mode", required=True, choices=["preflight", "benchmark"])
    parser.add_argument("--authorize-post-hoc-benchmark", action="store_true")
    parser.add_argument("--acknowledge-old-test-not-unseen", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "preflight":
        if args.authorize_post_hoc_benchmark or args.acknowledge_old_test_not_unseen:
            print("ERROR: authorization flags are forbidden in preflight", flush=True)
            return 2
        try:
            print(json.dumps(run_preflight(), indent=2, sort_keys=True))
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            return 1
    if not (args.authorize_post_hoc_benchmark and args.acknowledge_old_test_not_unseen):
        print("REFUSED: benchmark requires explicit Human authorization and old-Test acknowledgement", flush=True)
        return 3
    try:
        return run_benchmark()
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
