import inspect
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from course_work.data.feature_sets import FEATURE_SET_VERSION, load_validated_feature_set_registry
from course_work.data.features import load_validated_feature_view
from course_work.data.scaling import SCALING_VERSION, load_validated_target_scaler
from course_work.data.splitting import SPLIT_VERSION
from course_work.data.windows import (
    POPULATION_VERSION,
    PRIMARY_BOUNDARY_PROTOCOL,
    WINDOW_VERSION,
    compute_population_fingerprint,
    compute_window_fingerprint,
    load_validated_common_population,
    load_validated_window_index,
    materialize_phase_10,
    materialize_window,
    transform_feature_timeline,
)
from course_work.utils.artifacts import (
    canonical_json_bytes,
    csv_text,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)
from course_work.utils.reproducibility import DEVELOPMENT_SEED, build_torch_generator, seed_worker


DATALOADER_VERSION = "DATALOADERS-v1"
DATALOADER_CONFIG_VERSION = "DLCFG-v1"
DATALOADER_ARTIFACT_ROOT = "artifacts/dataloaders"
BASELINE_VARIANT = "FS1_TF1"
BASELINE_LOOKBACK = 144
BASELINE_TARGET_OPTION = "YS1"
BASELINE_BATCH_SIZE = 64
SUPPORTED_BATCH_SIZES = (32, 64)
BASELINE_NUM_WORKERS = 0
SPLIT_IDS = ("TRAIN", "VALIDATION", "TEST")
SPLIT_SEED_OFFSETS = {"TRAIN": 0, "VALIDATION": 1, "TEST": 2}
PHASE_47_AUTHORIZATION = "PHASE_47_FINAL_EVALUATION"
DATASET_REGISTRY_COLUMNS = [
    "dataset_config_id",
    "split_id",
    "variant_id",
    "lookback",
    "horizon",
    "target_scaling",
    "target_access_mode",
    "sample_count",
    "feature_count",
    "window_fingerprint",
    "population_fingerprint",
    "feature_fingerprint",
    "scaler_bundle_id",
    "dataset_fingerprint",
    "status",
]
LOADER_REGISTRY_COLUMNS = [
    "loader_config_id",
    "dataset_config_id",
    "split_id",
    "batch_size",
    "shuffle",
    "drop_last",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "in_order",
    "generator_seed",
    "loader_fingerprint",
    "expected_batches",
    "status",
]
COVERAGE_COLUMNS = [
    "loader_config_id",
    "split_id",
    "dataset_size",
    "observed_samples",
    "unique_samples",
    "duplicate_samples",
    "missing_samples",
    "coverage_ratio",
    "status",
]
BATCH_AUDIT_COLUMNS = [
    "loader_config_id",
    "split_id",
    "batch_index",
    "observed_batch_size",
    "x_shape",
    "y_model_shape",
    "y_raw_shape",
    "sample_idx_shape",
    "x_dtype",
    "y_dtype",
    "finite_status",
    "status",
]


class TargetAccessMode(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST_LOCKED = "TEST_LOCKED"
    TEST_EVALUATION = "TEST_EVALUATION"


@dataclass(frozen=True)
class DatasetConfig:
    dataset_config_id: str
    split_id: str
    variant_id: str
    lookback: int
    horizon: int
    target_option: str
    target_access_mode: str
    feature_count: int
    feature_fingerprint: str
    scaler_bundle_id: str
    scaler_checksum: str
    window_fingerprint: str
    population_fingerprint: str
    boundary_protocol: str
    dataset_fingerprint: str


@dataclass(frozen=True)
class LoaderConfig:
    loader_config_id: str
    dataset_config_id: str
    split_id: str
    batch_size: int
    shuffle: bool
    drop_last: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    timeout: float
    in_order: bool
    in_order_supported: bool
    generator_seed: int
    worker_init_policy: str
    expected_batches: int
    loader_fingerprint: str


class SequenceWindowDataset(Dataset):
    def __init__(
        self,
        feature_matrix: np.ndarray,
        window_records: pd.DataFrame,
        config: DatasetConfig,
        target_values: np.ndarray | None = None,
        target_scaler: dict[str, Any] | None = None,
        audit_mode: bool = False,
    ) -> None:
        matrix = np.ascontiguousarray(feature_matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != config.feature_count or len(matrix) == 0:
            raise ValueError("Feature matrix does not satisfy DatasetConfig")
        if not np.isfinite(matrix).all():
            raise ValueError("Feature matrix contains non-finite values")
        if window_records.empty:
            raise ValueError("Window records must not be empty")
        if not window_records["target_split_id"].astype(str).eq(config.split_id).all():
            raise ValueError("Window records contain a split outside DatasetConfig")
        if not window_records["lookback_steps"].eq(config.lookback).all():
            raise ValueError("Window records contain a lookback outside DatasetConfig")
        if not window_records["target_timestamp"].is_monotonic_increasing:
            raise ValueError("Window records are not chronological")
        access_mode = TargetAccessMode(config.target_access_mode)
        expected_modes = {
            "TRAIN": TargetAccessMode.TRAIN,
            "VALIDATION": TargetAccessMode.VALIDATION,
            "TEST": TargetAccessMode.TEST_LOCKED,
        }
        if access_mode == TargetAccessMode.TEST_EVALUATION:
            if config.split_id != "TEST" or target_values is None:
                raise ValueError("TEST_EVALUATION requires explicit Test targets")
        elif access_mode != expected_modes[config.split_id]:
            raise ValueError("Target access mode does not match split policy")
        target_array = None
        if access_mode != TargetAccessMode.TEST_LOCKED:
            if target_values is None:
                raise ValueError("Target-enabled Dataset requires target values")
            target_array = np.ascontiguousarray(np.asarray(target_values, dtype=np.float64).reshape(-1))
            if len(target_array) != len(matrix) or not np.isfinite(target_array).all():
                raise ValueError("Target timeline does not match feature timeline")
            target_array.setflags(write=False)
        if config.target_option == "YS1" and access_mode != TargetAccessMode.TEST_LOCKED and target_scaler is None:
            raise ValueError("YS1 Dataset requires the frozen target scaler")
        if config.target_option not in {"YS0", "YS1"}:
            raise ValueError("Unsupported target option")
        records = window_records.copy(deep=True)
        if "canonical_sample_idx" not in records.columns:
            records.insert(0, "canonical_sample_idx", records.index.to_numpy(dtype=np.int64, copy=True))
        records = records.reset_index(drop=True)
        sample_indices = records["canonical_sample_idx"].to_numpy(dtype=np.int64, copy=True)
        if len(np.unique(sample_indices)) != len(sample_indices):
            raise ValueError("Dataset sample indices are not unique")
        matrix.setflags(write=False)
        sample_indices.setflags(write=False)
        self._feature_matrix = matrix
        self._records = records
        self._sample_indices = sample_indices
        self._target_values = target_array
        self._target_scaler = target_scaler
        self.config = config
        self.audit_mode = bool(audit_mode)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Dataset index must be an integer")
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("Dataset index is outside the registered population")
        record = self._records.iloc[position]
        locked = self.config.target_access_mode == TargetAccessMode.TEST_LOCKED.value
        sample = materialize_window(
            self._feature_matrix,
            record,
            None if locked else self._target_values,
            self.config.target_option,
            None if self.config.target_option == "YS0" else self._target_scaler,
            allow_test_targets=self.config.target_access_mode == TargetAccessMode.TEST_EVALUATION.value,
        )
        x = torch.tensor(sample["X_seq"], dtype=torch.float32)
        sample_idx = torch.tensor(int(self._sample_indices[position]), dtype=torch.int64)
        if x.dtype != torch.float32 or x.shape != (self.config.lookback, self.config.feature_count):
            raise RuntimeError("Dataset item X violates shape or dtype contract")
        if self.audit_mode and not bool(torch.isfinite(x).all().item()):
            raise RuntimeError("Dataset item X contains non-finite values")
        output = {"x": x, "sample_idx": sample_idx}
        if locked:
            return output
        y_model = torch.from_numpy(sample["y_model"])
        y_raw_wh = torch.from_numpy(sample["y_raw_wh"])
        if y_model.dtype != torch.float32 or y_raw_wh.dtype != torch.float32:
            raise RuntimeError("Dataset target dtype contract failed")
        if y_model.shape != (1,) or y_raw_wh.shape != (1,):
            raise RuntimeError("Dataset target shape contract failed")
        if self.audit_mode and (not bool(torch.isfinite(y_model).all().item()) or not bool(torch.isfinite(y_raw_wh).all().item())):
            raise RuntimeError("Dataset target contains non-finite values")
        output["y_model"] = y_model
        output["y_raw_wh"] = y_raw_wh
        return output

    @property
    def sample_indices(self) -> np.ndarray:
        return self._sample_indices.copy()

    def window_record(self, index: int) -> pd.Series:
        return self._records.iloc[index].copy(deep=True)


def _fingerprint(value: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _dataset_config(
    split_id: str,
    variant_id: str,
    lookback: int,
    target_option: str,
    target_access_mode: TargetAccessMode,
    feature_entry: dict[str, Any],
    scaler_entry: dict[str, Any],
    window_fingerprint: str,
    population_fingerprint: str,
) -> DatasetConfig:
    runtime_payload = {
        "split_id": split_id,
        "variant_id": variant_id,
        "lookback": lookback,
        "horizon": 1,
        "target_option": target_option,
        "target_access_mode": target_access_mode.value,
        "feature_count": feature_entry["feature_count"],
        "feature_fingerprint": feature_entry["fingerprint"],
        "scaler_bundle_id": scaler_entry["bundle_id"],
        "scaler_checksum": scaler_entry["artifact_sha256"],
        "window_fingerprint": window_fingerprint,
        "population_fingerprint": population_fingerprint,
        "boundary_protocol": PRIMARY_BOUNDARY_PROTOCOL,
    }
    fingerprint_payload = {
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "scaling_version": SCALING_VERSION,
        "split_version": SPLIT_VERSION,
        **runtime_payload,
    }
    dataset_fingerprint = _fingerprint(fingerprint_payload)
    dataset_config_id = f'DS_{split_id}__{variant_id}__L{lookback}__{target_option}__{target_access_mode.value}'
    return DatasetConfig(dataset_config_id=dataset_config_id, dataset_fingerprint=dataset_fingerprint, **runtime_payload)


def build_dataset_suite(
    project_root: Path | None = None,
    variant_id: str = BASELINE_VARIANT,
    lookback: int = BASELINE_LOOKBACK,
    target_option: str = BASELINE_TARGET_OPTION,
    audit_mode: bool = False,
) -> dict[str, SequenceWindowDataset]:
    root = (project_root or get_project_root()).resolve()
    materialize_phase_10(root)
    feature_view = load_validated_feature_view(root)
    feature_registry = load_validated_feature_set_registry(root)
    if variant_id not in feature_registry["variants"]:
        raise KeyError(f"Unknown feature variant: {variant_id}")
    feature_entry = feature_registry["variants"][variant_id]
    split_manifest = read_json(root / "artifacts/splits/split_manifest.json")
    scaler_registry = read_json(root / "artifacts/scaling/scaler_registry.json")
    scaler_entry = scaler_registry["x_bundles"][variant_id]
    window_fingerprints = read_json(root / "artifacts/windows/window_fingerprints.json")
    window_key = f"L{lookback:03d}_H01_WB0"
    if window_key not in window_fingerprints["window_index_fingerprints"]:
        raise ValueError(f"Unsupported lookback: {lookback}")
    window_index = load_validated_window_index(root)
    population = load_validated_common_population(root)
    active = window_index.loc[
        window_index["lookback_steps"].eq(lookback)
        & window_index["included_common_population"].astype(bool)
        & window_index["WB0_valid"].astype(bool)
    ].copy(deep=True)
    if compute_window_fingerprint(active) != window_fingerprints["window_index_fingerprints"][window_key]:
        raise RuntimeError("Active Dataset window fingerprint mismatch")
    population_fingerprint = compute_population_fingerprint(population)
    if population_fingerprint != window_fingerprints["common_population_fingerprint"]:
        raise RuntimeError("Active Dataset population fingerprint mismatch")
    if set(active["target_sample_id"]) != set(population["target_sample_id"]):
        raise RuntimeError("Active Dataset targets differ from WINDOWPOP-v1")
    matrix = transform_feature_timeline(
        feature_view,
        variant_id,
        feature_registry,
        split_manifest["global_split_fingerprint"],
        root,
    )
    target_values = feature_view["Appliances"].to_numpy(dtype=np.float64, copy=True)
    target_scaler = load_validated_target_scaler(root) if target_option == "YS1" else None
    modes = {
        "TRAIN": TargetAccessMode.TRAIN,
        "VALIDATION": TargetAccessMode.VALIDATION,
        "TEST": TargetAccessMode.TEST_LOCKED,
    }
    datasets = {}
    for split_id in SPLIT_IDS:
        records = active.loc[active["target_split_id"].eq(split_id)].copy(deep=True)
        config = _dataset_config(
            split_id,
            variant_id,
            lookback,
            target_option,
            modes[split_id],
            feature_entry,
            scaler_entry,
            window_fingerprints["window_index_fingerprints"][window_key],
            population_fingerprint,
        )
        datasets[split_id] = SequenceWindowDataset(
            matrix,
            records,
            config,
            None if split_id == "TEST" else target_values,
            target_scaler,
            audit_mode,
        )
    expected_counts = population["target_split_id"].value_counts().to_dict()
    if {split_id: len(dataset) for split_id, dataset in datasets.items()} != {
        split_id: int(expected_counts[split_id]) for split_id in SPLIT_IDS
    }:
        raise RuntimeError("Dataset suite sample counts differ from WINDOWPOP-v1")
    return datasets


def build_test_evaluation_dataset(
    authorization: str,
    project_root: Path | None = None,
    variant_id: str = BASELINE_VARIANT,
    lookback: int = BASELINE_LOOKBACK,
    target_option: str = BASELINE_TARGET_OPTION,
    audit_mode: bool = False,
) -> SequenceWindowDataset:
    if authorization != PHASE_47_AUTHORIZATION:
        raise PermissionError("Test evaluation targets remain locked until Phase 47")
    root = (project_root or get_project_root()).resolve()
    datasets = build_dataset_suite(root, variant_id, lookback, target_option, audit_mode)
    locked = datasets["TEST"]
    feature_view = load_validated_feature_view(root)
    target_values = feature_view["Appliances"].to_numpy(dtype=np.float64, copy=True)
    target_scaler = load_validated_target_scaler(root) if target_option == "YS1" else None
    feature_registry = load_validated_feature_set_registry(root)
    scaler_registry = read_json(root / "artifacts/scaling/scaler_registry.json")
    config = _dataset_config(
        "TEST",
        variant_id,
        lookback,
        target_option,
        TargetAccessMode.TEST_EVALUATION,
        feature_registry["variants"][variant_id],
        scaler_registry["x_bundles"][variant_id],
        locked.config.window_fingerprint,
        locked.config.population_fingerprint,
    )
    return SequenceWindowDataset(
        locked._feature_matrix,
        locked._records,
        config,
        target_values,
        target_scaler,
        audit_mode,
    )


def build_split_generator(seed: int, split_id: str) -> torch.Generator:
    if split_id not in SPLIT_SEED_OFFSETS:
        raise ValueError(f"Unsupported split: {split_id}")
    return build_torch_generator(seed + SPLIT_SEED_OFFSETS[split_id])


def _pin_memory(device_type: str) -> bool:
    normalized = str(device_type).split(":", 1)[0].lower()
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device type: {device_type}")
    return normalized == "cuda"


def build_dataloader(
    dataset: SequenceWindowDataset,
    batch_size: int = BASELINE_BATCH_SIZE,
    seed: int = DEVELOPMENT_SEED,
    num_workers: int = BASELINE_NUM_WORKERS,
    device_type: str = "cpu",
) -> tuple[DataLoader, LoaderConfig, torch.Generator]:
    if not isinstance(dataset, SequenceWindowDataset):
        raise TypeError("build_dataloader requires SequenceWindowDataset")
    if len(dataset) == 0:
        raise ValueError("Cannot build a DataLoader for an empty Dataset")
    if batch_size not in SUPPORTED_BATCH_SIZES:
        raise ValueError(f"batch_size must be one of {SUPPORTED_BATCH_SIZES}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    split_id = dataset.config.split_id
    shuffle = split_id == "TRAIN"
    drop_last = False
    pin_memory = _pin_memory(device_type)
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None
    generator_seed = seed + SPLIT_SEED_OFFSETS[split_id]
    generator = build_split_generator(seed, split_id)
    expected_batches = math.ceil(len(dataset) / batch_size)
    in_order_supported = "in_order" in inspect.signature(DataLoader).parameters
    payload = {
        "dataloader_config_version": DATALOADER_CONFIG_VERSION,
        "dataset_fingerprint": dataset.config.dataset_fingerprint,
        "split_id": split_id,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "timeout": 0.0,
        "in_order": True,
        "in_order_supported": in_order_supported,
        "generator_seed": generator_seed,
        "worker_init_policy": "torch_initial_seed_mod_2_32_numpy_python",
    }
    loader_fingerprint = _fingerprint(payload)
    loader_config_id = f'DL_{split_id}__{dataset.config.variant_id}__L{dataset.config.lookback}__{dataset.config.target_option}__B{batch_size}__SEED{seed}'
    config = LoaderConfig(
        loader_config_id=loader_config_id,
        dataset_config_id=dataset.config.dataset_config_id,
        split_id=split_id,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        timeout=0.0,
        in_order=True,
        in_order_supported=in_order_supported,
        generator_seed=generator_seed,
        worker_init_policy="torch_initial_seed_mod_2_32_numpy_python",
        expected_batches=expected_batches,
        loader_fingerprint=loader_fingerprint,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "timeout": 0.0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    if in_order_supported:
        loader_kwargs["in_order"] = True
    return DataLoader(**loader_kwargs), config, generator


def build_train_validation_loaders(
    project_root: Path | None = None,
    variant_id: str = BASELINE_VARIANT,
    lookback: int = BASELINE_LOOKBACK,
    target_option: str = BASELINE_TARGET_OPTION,
    batch_size: int = BASELINE_BATCH_SIZE,
    seed: int = DEVELOPMENT_SEED,
    num_workers: int = BASELINE_NUM_WORKERS,
    device_type: str = "cpu",
) -> dict[str, tuple[DataLoader, LoaderConfig, torch.Generator]]:
    datasets = build_dataset_suite(project_root, variant_id, lookback, target_option)
    return {
        split_id: build_dataloader(datasets[split_id], batch_size, seed, num_workers, device_type)
        for split_id in ("TRAIN", "VALIDATION")
    }


def build_test_locked_loader(
    project_root: Path | None = None,
    variant_id: str = BASELINE_VARIANT,
    lookback: int = BASELINE_LOOKBACK,
    target_option: str = BASELINE_TARGET_OPTION,
    batch_size: int = BASELINE_BATCH_SIZE,
    seed: int = DEVELOPMENT_SEED,
    num_workers: int = BASELINE_NUM_WORKERS,
    device_type: str = "cpu",
) -> tuple[DataLoader, LoaderConfig, torch.Generator]:
    dataset = build_dataset_suite(project_root, variant_id, lookback, target_option)["TEST"]
    return build_dataloader(dataset, batch_size, seed, num_workers, device_type)


def build_test_evaluation_loader(
    authorization: str,
    project_root: Path | None = None,
    variant_id: str = BASELINE_VARIANT,
    lookback: int = BASELINE_LOOKBACK,
    target_option: str = BASELINE_TARGET_OPTION,
    batch_size: int = BASELINE_BATCH_SIZE,
    seed: int = DEVELOPMENT_SEED,
    num_workers: int = BASELINE_NUM_WORKERS,
    device_type: str = "cpu",
) -> tuple[DataLoader, LoaderConfig, torch.Generator]:
    dataset = build_test_evaluation_dataset(authorization, project_root, variant_id, lookback, target_option)
    return build_dataloader(dataset, batch_size, seed, num_workers, device_type)


def _hash_order(values: list[int]) -> str:
    return sha256_bytes(canonical_json_bytes(values))


def _audit_loader(
    loader: DataLoader,
    config: LoaderConfig,
    dataset: SequenceWindowDataset,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    expected_indices = set(int(value) for value in dataset.sample_indices)
    observed = []
    batch_rows = []
    selected_batches = {0, config.expected_batches // 2, config.expected_batches - 1}
    observed_batches = 0
    for batch_index, batch in enumerate(loader):
        observed_batches += 1
        indices = [int(value) for value in batch["sample_idx"].tolist()]
        observed.extend(indices)
        keys_valid = set(batch) == ({"x", "sample_idx"} if config.split_id == "TEST" else {"x", "y_model", "y_raw_wh", "sample_idx"})
        x = batch["x"]
        y_model = batch.get("y_model")
        y_raw = batch.get("y_raw_wh")
        shape_valid = x.ndim == 3 and tuple(x.shape[1:]) == (dataset.config.lookback, dataset.config.feature_count)
        dtype_valid = x.dtype == torch.float32 and batch["sample_idx"].dtype == torch.int64
        finite = bool(torch.isfinite(x).all().item())
        if y_model is not None and y_raw is not None:
            shape_valid = shape_valid and y_model.shape == (len(indices), 1) and y_raw.shape == (len(indices), 1)
            dtype_valid = dtype_valid and y_model.dtype == torch.float32 and y_raw.dtype == torch.float32
            finite = finite and bool(torch.isfinite(y_model).all().item()) and bool(torch.isfinite(y_raw).all().item())
        batch_status = keys_valid and shape_valid and dtype_valid and finite and 1 <= len(indices) <= config.batch_size
        if batch_index in selected_batches:
            batch_rows.append({
                "loader_config_id": config.loader_config_id,
                "split_id": config.split_id,
                "batch_index": batch_index,
                "observed_batch_size": len(indices),
                "x_shape": str(list(x.shape)),
                "y_model_shape": "LOCKED" if y_model is None else str(list(y_model.shape)),
                "y_raw_shape": "LOCKED" if y_raw is None else str(list(y_raw.shape)),
                "sample_idx_shape": str(list(batch["sample_idx"].shape)),
                "x_dtype": str(x.dtype),
                "y_dtype": "LOCKED" if y_model is None else str(y_model.dtype),
                "finite_status": "PASS" if finite else "FAIL",
                "status": "PASS" if batch_status else "FAIL",
            })
        if not batch_status:
            raise RuntimeError(f"Batch contract failed for {config.loader_config_id} at batch {batch_index}")
    observed_set = set(observed)
    duplicate_count = len(observed) - len(observed_set)
    missing_count = len(expected_indices - observed_set)
    coverage_valid = (
        observed_batches == config.expected_batches
        and len(observed) == len(dataset)
        and observed_set == expected_indices
        and duplicate_count == 0
        and missing_count == 0
    )
    coverage = {
        "loader_config_id": config.loader_config_id,
        "split_id": config.split_id,
        "dataset_size": len(dataset),
        "observed_samples": len(observed),
        "unique_samples": len(observed_set),
        "duplicate_samples": duplicate_count,
        "missing_samples": missing_count,
        "coverage_ratio": len(observed_set) / len(dataset),
        "status": "PASS" if coverage_valid else "FAIL",
    }
    if not coverage_valid:
        raise RuntimeError(f"Sample coverage failed for {config.loader_config_id}")
    return coverage, batch_rows, observed


def _dataset_registry_rows(datasets: dict[str, SequenceWindowDataset]) -> list[dict[str, Any]]:
    rows = []
    for split_id in SPLIT_IDS:
        dataset = datasets[split_id]
        config = dataset.config
        rows.append({
            "dataset_config_id": config.dataset_config_id,
            "split_id": split_id,
            "variant_id": config.variant_id,
            "lookback": config.lookback,
            "horizon": config.horizon,
            "target_scaling": config.target_option,
            "target_access_mode": config.target_access_mode,
            "sample_count": len(dataset),
            "feature_count": config.feature_count,
            "window_fingerprint": config.window_fingerprint,
            "population_fingerprint": config.population_fingerprint,
            "feature_fingerprint": config.feature_fingerprint,
            "scaler_bundle_id": config.scaler_bundle_id,
            "dataset_fingerprint": config.dataset_fingerprint,
            "status": "PASS",
        })
    return rows


def _loader_registry_row(config: LoaderConfig) -> dict[str, Any]:
    values = asdict(config)
    return {
        column: "PASS" if column == "status" else values[column]
        for column in LOADER_REGISTRY_COLUMNS
    }


def _readme_dataloaders() -> str:
    return "\n".join([
        "# DATALOADERS-v1",
        "",
        "SequenceWindowDataset is a map-style Dataset that slices registered WINDOWS-v1 records lazily from a read-only float32 feature timeline.",
        "",
        "TRAIN and VALIDATION return x, y_model, y_raw_wh and sample_idx. TEST_LOCKED returns only x and sample_idx until the explicit Phase 47 evaluation gate is authorized.",
        "",
        "Batches use the batch-first layout B by L by F. Batch sizes 32 and 64 are supported, with 64 as the baseline.",
        "",
        "TRAIN shuffles window order with a split-specific torch.Generator. VALIDATION and TEST preserve chronological order. Every loader uses drop_last false.",
        "",
        "The correctness baseline uses num_workers 0. The top-level worker initializer seeds NumPy and Python from torch.initial_seed for controlled optional multi-worker execution.",
        "",
        "CUDA enables pin_memory. CPU and MPS keep pin_memory disabled. Dataset outputs remain CPU tensors and device transfer belongs to the training engine.",
        "",
        "Audit loaders are disposable. Every training run must create fresh loaders so audit iteration never consumes the production generator state.",
        "",
    ])


def verify_phase_11_inputs(root: Path) -> dict[str, Any]:
    phase_10 = materialize_phase_10(root)
    if phase_10.get("artifact_version") != WINDOW_VERSION or phase_10.get("status") != "PASS":
        raise RuntimeError("WINDOWS-v1 is not signed off")
    for relative_path, expected_checksum in phase_10.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 11 upstream checksum mismatch: {relative_path}")
    manifest = read_json(root / "artifacts/windows/window_manifest.json")
    if manifest.get("population_version") != POPULATION_VERSION or manifest.get("scaling_version") != SCALING_VERSION:
        raise RuntimeError("Phase 11 upstream version mismatch")
    if manifest.get("full_3d_windows_saved") or manifest.get("test_target_values_exported"):
        raise RuntimeError("Phase 11 upstream storage or Test firewall contract failed")
    return phase_10


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("artifact_version") != DATALOADER_VERSION or signoff.get("status") != "PASS":
        raise RuntimeError("Existing Phase 11 sign-off is invalid")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 11 artifact checksum mismatch: {relative_path}")
    manifest = read_json(root / DATALOADER_ARTIFACT_ROOT / "dataloader_manifest.json")
    if manifest.get("dataloader_version") != DATALOADER_VERSION or manifest.get("audit_status") != "PASS":
        raise RuntimeError("Reloaded DataLoader manifest is invalid")
    registry = pd.read_csv(root / DATALOADER_ARTIFACT_ROOT / "dataset_registry.csv")
    loader_registry = pd.read_csv(root / DATALOADER_ARTIFACT_ROOT / "dataloader_registry.csv")
    if list(registry.columns) != DATASET_REGISTRY_COLUMNS or not registry["status"].eq("PASS").all():
        raise RuntimeError("Reloaded Dataset registry is invalid")
    if list(loader_registry.columns) != LOADER_REGISTRY_COLUMNS or not loader_registry["status"].eq("PASS").all():
        raise RuntimeError("Reloaded DataLoader registry is invalid")
    return signoff


def materialize_phase_11(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / DATALOADER_ARTIFACT_ROOT / "phase_11_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    phase_10 = verify_phase_11_inputs(root)
    datasets = build_dataset_suite(root, audit_mode=True)
    feature_view = load_validated_feature_view(root)
    target_values = feature_view["Appliances"].to_numpy(dtype=np.float64, copy=True)
    target_scaler = load_validated_target_scaler(root)
    probe_checks = []
    for split_id in SPLIT_IDS:
        dataset = datasets[split_id]
        for position in (0, len(dataset) - 1):
            item = dataset[position]
            record = dataset.window_record(position)
            direct = materialize_window(
                dataset._feature_matrix,
                record,
                None if split_id == "TEST" else target_values,
                BASELINE_TARGET_OPTION,
                target_scaler,
                allow_test_targets=False,
            )
            probe_checks.append(bool(torch.equal(item["x"], torch.tensor(direct["X_seq"], dtype=torch.float32))))
            probe_checks.append(int(item["sample_idx"].item()) == int(record["canonical_sample_idx"]))
            if split_id == "TEST":
                probe_checks.append(set(item) == {"x", "sample_idx"})
            else:
                probe_checks.extend([
                    bool(torch.equal(item["y_model"], torch.from_numpy(direct["y_model"]))),
                    bool(torch.equal(item["y_raw_wh"], torch.from_numpy(direct["y_raw_wh"]))),
                ])
    if not all(probe_checks):
        raise RuntimeError("Dataset and Phase 10 materializer probe mismatch")
    environment = read_json(root / "artifacts/environment/environment_report.json")
    selected_device = environment["selected_device"]
    loader_registry_rows = []
    coverage_rows = []
    batch_rows = []
    observed_orders = {}
    for batch_size in SUPPORTED_BATCH_SIZES:
        for split_id in SPLIT_IDS:
            loader, config, _ = build_dataloader(
                datasets[split_id],
                batch_size,
                DEVELOPMENT_SEED,
                BASELINE_NUM_WORKERS,
                selected_device,
            )
            coverage, batches, order = _audit_loader(loader, config, datasets[split_id])
            loader_registry_rows.append(_loader_registry_row(config))
            coverage_rows.append(coverage)
            batch_rows.extend(batches)
            observed_orders[(batch_size, split_id)] = order
    validation_order = observed_orders[(BASELINE_BATCH_SIZE, "VALIDATION")]
    test_order = observed_orders[(BASELINE_BATCH_SIZE, "TEST")]
    sequential_rows = []
    for split_id, observed in (("VALIDATION", validation_order), ("TEST", test_order)):
        expected = [int(value) for value in datasets[split_id].sample_indices]
        timestamps = pd.to_datetime(datasets[split_id]._records["target_timestamp"])
        valid = observed == expected and timestamps.is_monotonic_increasing
        sequential_rows.append({
            "split": split_id,
            "sample_count": len(observed),
            "timestamp_monotonic": bool(timestamps.is_monotonic_increasing),
            "first_sample_idx": observed[0],
            "last_sample_idx": observed[-1],
            "order_fingerprint": _hash_order(observed),
            "status": "PASS" if valid else "FAIL",
        })
    if any(row["status"] != "PASS" for row in sequential_rows):
        raise RuntimeError("Sequential DataLoader order audit failed")
    same_loader_1, same_config_1, _ = build_dataloader(datasets["TRAIN"], BASELINE_BATCH_SIZE, DEVELOPMENT_SEED, 0, selected_device)
    same_loader_2, same_config_2, _ = build_dataloader(datasets["TRAIN"], BASELINE_BATCH_SIZE, DEVELOPMENT_SEED, 0, selected_device)
    different_loader, different_config, _ = build_dataloader(datasets["TRAIN"], BASELINE_BATCH_SIZE, DEVELOPMENT_SEED + 1, 0, selected_device)
    same_order_1 = [int(datasets["TRAIN"].sample_indices[index]) for index in same_loader_1.sampler]
    same_order_2 = [int(datasets["TRAIN"].sample_indices[index]) for index in same_loader_2.sampler]
    epoch_2_order = [int(datasets["TRAIN"].sample_indices[index]) for index in same_loader_1.sampler]
    different_order = [int(datasets["TRAIN"].sample_indices[index]) for index in different_loader.sampler]
    same_match = same_order_1 == same_order_2
    epoch_changes = same_order_1 != epoch_2_order
    different_changes = same_order_1 != different_order
    if not same_match or not epoch_changes or not different_changes:
        raise RuntimeError("Train shuffle reproducibility audit failed")
    shuffle_rows = [{
        "seed": DEVELOPMENT_SEED,
        "loader_config": same_config_1.loader_config_id,
        "same_seed_run_1_order_fingerprint": _hash_order(same_order_1),
        "same_seed_run_2_order_fingerprint": _hash_order(same_order_2),
        "same_seed_match": same_match,
        "epoch1_order_fingerprint": _hash_order(same_order_1),
        "epoch2_order_fingerprint": _hash_order(epoch_2_order),
        "different_seed": DEVELOPMENT_SEED + 1,
        "different_seed_order_fingerprint": _hash_order(different_order),
        "different_seed_loader_fingerprint": different_config.loader_fingerprint,
        "status": "PASS",
    }]
    test_batch = next(iter(build_dataloader(datasets["TEST"], BASELINE_BATCH_SIZE, DEVELOPMENT_SEED, 0, selected_device)[0]))
    authorization_guard = False
    try:
        build_test_evaluation_dataset("DENIED", root)
    except PermissionError:
        authorization_guard = True
    firewall_checks = {
        "test_locked_dataset_has_no_target": set(datasets["TEST"][0]) == {"x", "sample_idx"},
        "test_locked_loader_has_no_target": set(test_batch) == {"x", "sample_idx"},
        "test_sample_count_matches": len(datasets["TEST"]) == 2961,
        "test_order_chronological": test_order == [int(value) for value in datasets["TEST"].sample_indices],
        "same_population_fingerprint": datasets["TEST"].config.population_fingerprint == datasets["TRAIN"].config.population_fingerprint,
        "evaluation_factory_requires_explicit_mode": authorization_guard,
    }
    firewall_rows = [
        {"check": check, "expected": True, "actual": actual, "status": "PASS" if actual else "FAIL"}
        for check, actual in firewall_checks.items()
    ]
    if not all(firewall_checks.values()):
        raise RuntimeError("Test target firewall audit failed")
    worker_rows = [{
        "num_workers": 0,
        "worker_init_function": "course_work.utils.reproducibility.seed_worker",
        "generator_used": True,
        "persistent_workers": False,
        "prefetch_factor": None,
        "spawn_compatible": inspect.isfunction(seed_worker) and seed_worker.__name__ == "seed_worker",
        "status": "PASS",
        "notes": "correctness baseline",
    }]
    device_type = selected_device.split(":", 1)[0]
    pin_memory = _pin_memory(device_type)
    device_policy = {
        "selected_device": selected_device,
        "actual_pin_memory": pin_memory,
        "cpu": {"pin_memory": False, "non_blocking_transfer": False},
        "mps": {"pin_memory": False, "non_blocking_transfer": False},
        "cuda": {"pin_memory": True, "non_blocking_transfer": True},
        "dataset_output_device": "cpu",
        "device_transfer_owner": "training_engine",
        "status": "PASS",
    }
    baseline_batch = next(iter(build_dataloader(datasets["TRAIN"], BASELINE_BATCH_SIZE, DEVELOPMENT_SEED, 0, selected_device)[0]))
    transferred = baseline_batch["x"].to(torch.device(selected_device))
    if transferred.device.type != device_type or not bool(torch.isfinite(transferred).all().item()):
        raise RuntimeError("Device transfer smoke test failed")
    dataset_rows = _dataset_registry_rows(datasets)
    manifest = {
        "dataloader_version": DATALOADER_VERSION,
        "dataloader_config_version": DATALOADER_CONFIG_VERSION,
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "scaling_version": SCALING_VERSION,
        "split_version": SPLIT_VERSION,
        "environment_id": environment["environment_id"],
        "torch_version": torch.__version__,
        "dataset_class": "SequenceWindowDataset",
        "dataset_type": "map_style",
        "lazy_materialization": True,
        "baseline_variant": BASELINE_VARIANT,
        "baseline_lookback": BASELINE_LOOKBACK,
        "baseline_target_option": BASELINE_TARGET_OPTION,
        "baseline_batch_size": BASELINE_BATCH_SIZE,
        "supported_batch_sizes": list(SUPPORTED_BATCH_SIZES),
        "train_shuffle": True,
        "validation_shuffle": False,
        "test_shuffle": False,
        "drop_last": False,
        "baseline_num_workers": BASELINE_NUM_WORKERS,
        "worker_seed_policy": "torch_initial_seed_mod_2_32_numpy_python",
        "generator_seed_policy": {"TRAIN": "seed+0", "VALIDATION": "seed+1", "TEST": "seed+2"},
        "pin_memory_policy": {"cuda": True, "mps": False, "cpu": False},
        "actual_device": selected_device,
        "actual_pin_memory": pin_memory,
        "persistent_workers_policy": "false_for_num_workers_0_true_for_positive_workers",
        "prefetch_policy": "unset_for_num_workers_0_else_2",
        "in_order_policy": "true_if_supported_else_framework_default_ordered",
        "in_order_supported": "in_order" in inspect.signature(DataLoader).parameters,
        "test_target_access_policy": "TEST_LOCKED_UNTIL_EXPLICIT_PHASE_47_AUTHORIZATION",
        "train_sample_count": len(datasets["TRAIN"]),
        "validation_sample_count": len(datasets["VALIDATION"]),
        "test_sample_count": len(datasets["TEST"]),
        "total_sample_count": sum(len(dataset) for dataset in datasets.values()),
        "feature_count": datasets["TRAIN"].config.feature_count,
        "dataset_fingerprints": {split_id: datasets[split_id].config.dataset_fingerprint for split_id in SPLIT_IDS},
        "baseline_loader_fingerprints": {
            row["split_id"]: row["loader_fingerprint"]
            for row in loader_registry_rows
            if row["batch_size"] == BASELINE_BATCH_SIZE
        },
        "sample_coverage_passed": all(row["status"] == "PASS" for row in coverage_rows),
        "shuffle_reproducibility_passed": True,
        "sequential_order_passed": True,
        "batch_shape_audit_passed": all(row["status"] == "PASS" for row in batch_rows),
        "test_firewall_passed": True,
        "audit_status": "PASS",
        "warnings": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact_root = root / DATALOADER_ARTIFACT_ROOT
    artifact_payloads = {
        "dataloader_manifest.json": canonical_json_bytes(manifest),
        "dataset_registry.csv": csv_text(DATASET_REGISTRY_COLUMNS, dataset_rows).encode("utf-8"),
        "dataloader_registry.csv": csv_text(LOADER_REGISTRY_COLUMNS, loader_registry_rows).encode("utf-8"),
        "dataloader_sample_coverage.csv": csv_text(COVERAGE_COLUMNS, coverage_rows).encode("utf-8"),
        "dataloader_batch_audit.csv": csv_text(BATCH_AUDIT_COLUMNS, batch_rows).encode("utf-8"),
        "shuffle_reproducibility_audit.csv": csv_text(list(shuffle_rows[0]), shuffle_rows).encode("utf-8"),
        "sequential_order_audit.csv": csv_text(list(sequential_rows[0]), sequential_rows).encode("utf-8"),
        "worker_configuration_audit.csv": csv_text(list(worker_rows[0]), worker_rows).encode("utf-8"),
        "dataloader_test_firewall_audit.csv": csv_text(list(firewall_rows[0]), firewall_rows).encode("utf-8"),
        "device_transfer_policy.json": canonical_json_bytes(device_policy),
        "dataloader_discrepancies.json": canonical_json_bytes({"dataloader_version": DATALOADER_VERSION, "discrepancies": []}),
        "README_DATALOADERS.md": _readme_dataloaders().encode("utf-8"),
    }
    for filename, content in artifact_payloads.items():
        write_text_once_or_verify(artifact_root / filename, content.decode("utf-8"))
    output_paths = [f"{DATALOADER_ARTIFACT_ROOT}/{filename}" for filename in artifact_payloads]
    input_paths = [
        "configs/base/coursework_contract.json",
        "artifacts/environment/environment_report.json",
        "artifacts/feature_sets/feature_set_registry.json",
        "artifacts/splits/split_manifest.json",
        "artifacts/scaling/scaling_manifest.json",
        "artifacts/scaling/scaler_registry.json",
        "artifacts/scaling/phase_9_signoff.json",
        "artifacts/windows/window_manifest.json",
        "artifacts/windows/window_fingerprints.json",
        "artifacts/windows/window_index.csv",
        "artifacts/windows/common_target_population.csv",
        "artifacts/windows/phase_10_signoff.json",
    ]
    signoff = {
        "phase_id": 11,
        "phase_version": "PHASE-11-v1",
        "artifact_version": DATALOADER_VERSION,
        "dataset_revision": phase_10["dataset_revision"],
        "environment_id": environment["environment_id"],
        "config_fingerprint": phase_10["config_fingerprint"],
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "scaling_version": SCALING_VERSION,
        "split_version": SPLIT_VERSION,
        "input_paths": input_paths,
        "input_checksums": {path: sha256_file(root / path) for path in input_paths},
        "output_paths": output_paths,
        "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        "status": "PASS",
        "created_at": manifest["created_at"],
        "tests": [
            "upstream_versions_and_checksums",
            "map_style_lazy_dataset",
            "dataset_counts_match_WINDOWPOP_v1",
            "dataset_determinism",
            "dataset_equals_phase_10_materializer",
            "float32_item_and_batch_contract",
            "B32_and_B64_support",
            "drop_last_false_full_coverage",
            "train_shuffle_only",
            "same_seed_fresh_loader_reproducibility",
            "consecutive_epoch_reshuffle",
            "separate_split_generators",
            "validation_test_chronology",
            "worker_seed_policy",
            "device_transfer_policy",
            "test_target_firewall",
            "dataset_and_loader_fingerprints",
        ],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)
