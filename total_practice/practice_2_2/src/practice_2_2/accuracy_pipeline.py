from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .r7_training_correctness import (
    ExactCrossEntropyAccumulator,
    cross_entropy_batch_terms,
    set_frozen_batchnorm_eval,
    state_dict_sha256,
)


CLASS_NAMES = (
    "body_wash",
    "face_mask",
    "facial_cleanser",
    "lipstick",
    "moisturizer",
    "perfume",
    "serum",
    "shampoo",
    "sunscreen",
    "toner",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVELOPMENT_SPLITS = ("Train", "Validation")
SUPPORTED_ARCHITECTURES = ("resnet18", "efficientnet_b0", "densenet121")
REQUIRED_MANIFEST_COLUMNS = (
    "asset_id",
    "relative_path",
    "class_name",
    "component_id",
    "split",
    "raw_sha256",
    "decoded_pixel_sha256",
    "is_generated",
    "use_for_model",
)


@dataclass(frozen=True)
class AccuracyConfig:
    architecture: str = "resnet18"
    num_classes: int = 10
    image_size: int = 224
    dropout: float = 0.35
    batch_size: int = 32
    num_workers: int = 0
    warmup_epochs: int = 4
    finetune_epochs: int = 18
    head_learning_rate: float = 5e-4
    backbone_learning_rate: float = 2e-5
    weight_decay: float = 2e-4
    label_smoothing: float = 0.05
    patience: int = 5
    min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0
    pretrained: bool = True

    def __post_init__(self) -> None:
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        if self.num_classes != len(CLASS_NAMES):
            raise ValueError("AccuracyConfig requires the ten-class label contract")
        if self.image_size <= 0 or self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("Image and loader dimensions are invalid")
        if self.warmup_epochs <= 0 or self.finetune_epochs <= 0:
            raise ValueError("Both training stages require at least one epoch")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if min(self.head_learning_rate, self.backbone_learning_rate) <= 0:
            raise ValueError("Learning rates must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("Optimization settings are invalid")
        if self.patience <= 0 or self.min_delta < 0:
            raise ValueError("Stopping settings are invalid")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _boolean_series(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    values = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if values.isna().any():
        raise ValueError(f"Manifest column contains invalid booleans: {column}")
    return values.astype(bool)


def _safe_image_path(dataset_root: Path, relative_path: str) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    path = (root / str(relative_path)).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Image path escapes dataset root: {relative_path}")
    return path


def inspect_manifest_authorization(
    manifest_path: Path,
    dataset_root: Path,
    class_names: Sequence[str] = CLASS_NAMES,
    check_files: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    blocked_reasons: list[str] = []
    if not path.is_file():
        return {
            "authorized": False,
            "blocked_reasons": ["MANIFEST_MISSING"],
            "manifest_path": str(path),
            "dataset_root": str(root),
        }
    frame = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(frame.columns))
    if missing_columns:
        return {
            "authorized": False,
            "blocked_reasons": ["MANIFEST_COLUMNS_MISSING"],
            "missing_columns": missing_columns,
            "manifest_path": str(path),
            "dataset_root": str(root),
        }
    development = frame.loc[frame["split"].isin(DEVELOPMENT_SPLITS)].copy()
    if development.empty:
        blocked_reasons.append("DEVELOPMENT_ROWS_MISSING")
    use_for_model = _boolean_series(development["use_for_model"], "use_for_model")
    generated = _boolean_series(development["is_generated"], "is_generated")
    unauthorized_count = int((~use_for_model).sum())
    generated_count = int(generated.sum())
    if unauthorized_count:
        blocked_reasons.append("DEVELOPMENT_ROWS_NOT_AUTHORIZED")
    if generated_count:
        blocked_reasons.append("GENERATED_ROWS_PRESENT")
    observed_classes = set(development["class_name"].astype(str))
    expected_classes = set(class_names)
    missing_classes = sorted(expected_classes - observed_classes)
    unknown_classes = sorted(observed_classes - expected_classes)
    if missing_classes:
        blocked_reasons.append("DEVELOPMENT_CLASSES_MISSING")
    if unknown_classes:
        blocked_reasons.append("UNKNOWN_CLASSES_PRESENT")
    split_counts = {
        split: int((development["split"] == split).sum())
        for split in DEVELOPMENT_SPLITS
    }
    class_split_counts = {
        split: {
            class_name: int(
                (
                    (development["split"] == split)
                    & (development["class_name"] == class_name)
                ).sum()
            )
            for class_name in class_names
        }
        for split in DEVELOPMENT_SPLITS
    }
    if any(count == 0 for counts in class_split_counts.values() for count in counts.values()):
        blocked_reasons.append("CLASS_SPLIT_SUPPORT_MISSING")
    duplicate_asset_count = int(development["asset_id"].duplicated().sum())
    duplicate_path_count = int(development["relative_path"].duplicated().sum())
    if duplicate_asset_count:
        blocked_reasons.append("DUPLICATE_ASSET_IDS")
    if duplicate_path_count:
        blocked_reasons.append("DUPLICATE_RELATIVE_PATHS")
    train = development.loc[development["split"] == "Train"]
    validation = development.loc[development["split"] == "Validation"]
    overlap_counts = {
        column: len(set(train[column].astype(str)) & set(validation[column].astype(str)))
        for column in ("component_id", "raw_sha256", "decoded_pixel_sha256")
    }
    if any(overlap_counts.values()):
        blocked_reasons.append("TRAIN_VALIDATION_LEAKAGE")
    invalid_paths: list[str] = []
    missing_files: list[str] = []
    if check_files:
        for relative_path in development["relative_path"].astype(str):
            try:
                image_path = _safe_image_path(root, relative_path)
            except ValueError:
                invalid_paths.append(relative_path)
                continue
            if not image_path.is_file():
                missing_files.append(relative_path)
    if invalid_paths:
        blocked_reasons.append("INVALID_IMAGE_PATHS")
    if missing_files:
        blocked_reasons.append("DEVELOPMENT_FILES_MISSING")
    return {
        "authorized": not blocked_reasons,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "manifest_path": str(path),
        "dataset_root": str(root),
        "development_row_count": int(len(development)),
        "split_counts": split_counts,
        "class_split_counts": class_split_counts,
        "unauthorized_row_count": unauthorized_count,
        "generated_row_count": generated_count,
        "duplicate_asset_count": duplicate_asset_count,
        "duplicate_path_count": duplicate_path_count,
        "train_validation_overlap_counts": overlap_counts,
        "missing_classes": missing_classes,
        "unknown_classes": unknown_classes,
        "invalid_path_count": len(invalid_paths),
        "missing_file_count": len(missing_files),
    }


def load_authorized_development_manifest(
    manifest_path: Path,
    dataset_root: Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> pd.DataFrame:
    report = inspect_manifest_authorization(
        manifest_path,
        dataset_root,
        class_names=class_names,
        check_files=True,
    )
    if not report["authorized"]:
        reasons = ", ".join(report["blocked_reasons"])
        raise RuntimeError(f"Development manifest is not authorized: {reasons}")
    frame = pd.read_csv(Path(manifest_path).expanduser().resolve())
    development = frame.loc[frame["split"].isin(DEVELOPMENT_SPLITS)].copy()
    development["class_name"] = pd.Categorical(
        development["class_name"], categories=list(class_names), ordered=True
    )
    return development.sort_values(
        ["split", "class_name", "asset_id"], kind="stable"
    ).reset_index(drop=True)


def validate_authorized_development_frame(
    frame: pd.DataFrame,
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    missing_columns = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Development frame columns are missing: {missing_columns}")
    observed_splits = set(frame["split"].astype(str))
    if observed_splits != set(DEVELOPMENT_SPLITS):
        raise ValueError("Development frame must contain only Train and Validation")
    if not _boolean_series(frame["use_for_model"], "use_for_model").all():
        raise RuntimeError("Development frame contains unauthorized rows")
    if _boolean_series(frame["is_generated"], "is_generated").any():
        raise RuntimeError("Development frame contains generated rows")
    observed_classes = set(frame["class_name"].astype(str))
    if observed_classes != set(class_names):
        raise ValueError("Development frame does not match the class contract")
    if frame["asset_id"].duplicated().any() or frame["relative_path"].duplicated().any():
        raise ValueError("Development frame contains duplicate identities")
    train = frame.loc[frame["split"] == "Train"]
    validation = frame.loc[frame["split"] == "Validation"]
    for split_frame in (train, validation):
        if any((split_frame["class_name"] == class_name).sum() == 0 for class_name in class_names):
            raise ValueError("Development frame has incomplete class support")
    for column in ("component_id", "raw_sha256", "decoded_pixel_sha256"):
        if set(train[column].astype(str)) & set(validation[column].astype(str)):
            raise RuntimeError("Development frame contains Train/Validation leakage")


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_root: Path,
        split: str,
        transform: Any,
        class_names: Sequence[str] = CLASS_NAMES,
    ) -> None:
        if split not in DEVELOPMENT_SPLITS:
            raise ValueError(f"Unsupported development split: {split}")
        selected = frame.loc[frame["split"] == split].copy().reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"Manifest contains no rows for split: {split}")
        self.frame = selected
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.class_to_index = {
            class_name: index for index, class_name in enumerate(class_names)
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        image_path = _safe_image_path(self.dataset_root, str(row["relative_path"]))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        target = self.class_to_index[str(row["class_name"])]
        return tensor, target, str(row["asset_id"])


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomAffine(
                degrees=4,
                translate=(0.02, 0.02),
                scale=(0.96, 1.04),
                fill=255,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.02,
                    )
                ],
                p=0.6,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_evaluation_transform(
    image_size: int = 224,
    scale: float = 1.0,
) -> transforms.Compose:
    if not 0.0 < scale <= 1.0:
        raise ValueError("Evaluation scale must be in (0, 1]")
    operations: list[Any] = []
    if scale == 1.0:
        operations.append(
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
        )
    else:
        inner_size = max(1, int(round(image_size * scale)))
        difference = image_size - inner_size
        left = difference // 2
        top = difference // 2
        right = difference - left
        bottom = difference - top
        operations.extend(
            [
                transforms.Resize(
                    (inner_size, inner_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.Pad((left, top, right, bottom), fill=255),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def build_loader(
    frame: pd.DataFrame,
    dataset_root: Path,
    split: str,
    transform: Any,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    class_names: Sequence[str] = CLASS_NAMES,
) -> DataLoader:
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError(f"Unsupported development split: {split}")
    dataset = ManifestImageDataset(
        frame,
        dataset_root,
        split,
        transform,
        class_names=class_names,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "Train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def build_model(config: AccuracyConfig) -> nn.Module:
    if config.architecture == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None
        model = models.resnet18(weights=weights)
        features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(features, config.num_classes)
        )
        return model
    if config.architecture == "efficientnet_b0":
        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if config.pretrained
            else None
        )
        model = models.efficientnet_b0(weights=weights)
        features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(features, config.num_classes)
        )
        return model
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if config.pretrained else None
    model = models.densenet121(weights=weights)
    features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(config.dropout), nn.Linear(features, config.num_classes)
    )
    return model


def _head_module(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "resnet18":
        return model.fc
    return model.classifier


def _final_block_module(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "resnet18":
        return model.layer4
    if architecture == "efficientnet_b0":
        return model.features[-2:]
    return nn.ModuleList([model.features.denseblock4, model.features.norm5])


def configure_training_stage(
    model: nn.Module,
    architecture: str,
    stage: str,
) -> None:
    if stage not in ("head_warmup", "final_block_finetune"):
        raise ValueError(f"Unsupported training stage: {stage}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in _head_module(model, architecture).parameters():
        parameter.requires_grad = True
    if stage == "final_block_finetune":
        for parameter in _final_block_module(model, architecture).parameters():
            parameter.requires_grad = True


def build_optimizer(
    model: nn.Module,
    config: AccuracyConfig,
) -> torch.optim.AdamW:
    head = _head_module(model, config.architecture)
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    groups: list[dict[str, Any]] = []
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": config.backbone_learning_rate,
                "group_name": "backbone",
            }
        )
    groups.append(
        {
            "params": head_parameters,
            "lr": config.head_learning_rate,
            "group_name": "head",
        }
    )
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    return {
        str(group.get("group_name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def metrics_from_confusion(
    confusion_matrix: torch.Tensor,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    matrix = confusion_matrix.to(torch.float64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix shape does not match the label contract")
    total = matrix.sum()
    if total <= 0 or torch.any(matrix.sum(dim=1) <= 0):
        raise ValueError("Every class requires positive support")
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    recall = true_positive / support
    precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
    f1 = torch.where(
        precision + recall > 0,
        2.0 * precision * recall / (precision + recall),
        0.0,
    )
    per_class = {
        class_name: {
            "support": int(support[index].item()),
            "precision": float(precision[index].item()),
            "recall": float(recall[index].item()),
            "f1": float(f1[index].item()),
        }
        for index, class_name in enumerate(class_names)
    }
    return {
        "sample_count": int(total.item()),
        "accuracy": float(true_positive.sum().item() / total.item()),
        "macro_f1": float(f1.mean().item()),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix.tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    split: str,
    device: torch.device,
    label_smoothing: float,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 1.0,
) -> dict[str, Any]:
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError(f"Unsupported development split: {split}")
    training = split == "Train"
    if training and optimizer is None:
        raise ValueError("Train epoch requires an optimizer")
    if not training and optimizer is not None:
        raise ValueError("Validation epoch must not receive an optimizer")
    if training:
        model.train()
        set_frozen_batchnorm_eval(model)
    else:
        model.eval()
    accumulator = ExactCrossEntropyAccumulator(split)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets, _ in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            terms = cross_entropy_batch_terms(
                logits,
                targets,
                label_smoothing=label_smoothing,
            )
            if training:
                terms.optimization_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    gradient_clip_norm,
                )
                optimizer.step()
            accumulator.update(terms)
            predictions = logits.argmax(dim=1)
            flat = (
                targets.detach().cpu().to(torch.int64) * num_classes
                + predictions.detach().cpu().to(torch.int64)
            )
            confusion += torch.bincount(
                flat, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
    metrics = metrics_from_confusion(confusion)
    metrics.update(
        {
            "split": split,
            "loss": accumulator.mean,
            "loss_numerator": accumulator.numerator,
            "loss_denominator": accumulator.denominator,
            "batch_count": accumulator.batch_count,
        }
    )
    return metrics


def _checkpoint_is_better(
    candidate: Mapping[str, Any],
    best: Mapping[str, Any] | None,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    macro_f1 = float(candidate["macro_f1"])
    best_macro_f1 = float(best["macro_f1"])
    if macro_f1 > best_macro_f1 + min_delta:
        return True
    if abs(macro_f1 - best_macro_f1) <= min_delta:
        accuracy = float(candidate["accuracy"])
        best_accuracy = float(best["accuracy"])
        if accuracy > best_accuracy + min_delta:
            return True
        if abs(accuracy - best_accuracy) <= min_delta:
            return float(candidate["loss"]) < float(best["loss"])
    return False


def save_accuracy_checkpoint(
    path: Path,
    model: nn.Module,
    config: AccuracyConfig,
    class_names: Sequence[str],
    seed: int,
    stage: str,
    epoch: int,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    payload = {
        "schema_version": 1,
        "architecture": config.architecture,
        "constructor_weights": None,
        "config": asdict(config),
        "class_names": list(class_names),
        "seed": int(seed),
        "stage": stage,
        "epoch": int(epoch),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "model_state_dict": model_state,
        "model_state_sha256": state_dict_sha256(model_state),
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def load_accuracy_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("Unsupported accuracy checkpoint schema")
    if checkpoint.get("constructor_weights") is not None:
        raise RuntimeError("Checkpoint reload must not request pretrained weights")
    config_values = dict(checkpoint["config"])
    config_values["pretrained"] = False
    config = AccuracyConfig(**config_values)
    if checkpoint.get("architecture") != config.architecture:
        raise RuntimeError("Checkpoint architecture metadata mismatch")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if state_dict_sha256(model.state_dict()) != checkpoint.get("model_state_sha256"):
        raise RuntimeError("Checkpoint state hash mismatch")
    model.to(map_location)
    model.eval()
    return model, checkpoint


def _write_json(path: Path, value: Any) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    return output_path


def _run_stage(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: AccuracyConfig,
    device: torch.device,
    seed: int,
    class_names: Sequence[str],
    stage: str,
    epoch_count: int,
    epoch_offset: int,
    checkpoint_path: Path,
    stop_early: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configure_training_stage(model, config.architecture, stage)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epoch_count),
        eta_min=min(config.backbone_learning_rate, config.head_learning_rate) * 0.05,
    )
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    bad_epochs = 0
    for stage_epoch in range(1, epoch_count + 1):
        epoch = epoch_offset + stage_epoch
        learning_rates = optimizer_learning_rates(optimizer)
        train_metrics = run_epoch(
            model,
            train_loader,
            "Train",
            device,
            config.label_smoothing,
            config.num_classes,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            "Validation",
            device,
            config.label_smoothing,
            config.num_classes,
        )
        record = {
            "epoch": epoch,
            "stage_epoch": stage_epoch,
            "stage": stage,
            "learning_rates": learning_rates,
            "train": train_metrics,
            "validation": validation_metrics,
            "generalization_gap": train_metrics["accuracy"]
            - validation_metrics["accuracy"],
        }
        history.append(record)
        if _checkpoint_is_better(validation_metrics, best, config.min_delta):
            best = copy.deepcopy(validation_metrics)
            best["epoch"] = epoch
            best["stage"] = stage
            best["train_metrics"] = copy.deepcopy(train_metrics)
            save_accuracy_checkpoint(
                checkpoint_path,
                model,
                config,
                class_names,
                seed,
                stage,
                epoch,
                train_metrics,
                validation_metrics,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
        scheduler.step()
        if stop_early and bad_epochs >= config.patience:
            break
    if best is None:
        raise RuntimeError(f"Training stage produced no checkpoint: {stage}")
    return history, best


def train_single_seed(
    development_manifest: pd.DataFrame,
    dataset_root: Path,
    output_dir: Path,
    config: AccuracyConfig,
    seed: int,
    device: torch.device | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    validate_authorized_development_frame(development_manifest, class_names)
    set_reproducibility(seed)
    selected_device = device or select_device()
    train_loader = build_loader(
        development_manifest,
        dataset_root,
        "Train",
        build_train_transform(config.image_size),
        config.batch_size,
        seed,
        config.num_workers,
        class_names,
    )
    validation_loader = build_loader(
        development_manifest,
        dataset_root,
        "Validation",
        build_evaluation_transform(config.image_size),
        config.batch_size,
        seed,
        config.num_workers,
        class_names,
    )
    model = build_model(config).to(selected_device)
    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    warmup_path = run_dir / "best_warmup.pt"
    final_path = run_dir / "best_validation.pt"
    warmup_history, warmup_best = _run_stage(
        model,
        train_loader,
        validation_loader,
        config,
        selected_device,
        seed,
        class_names,
        "head_warmup",
        config.warmup_epochs,
        0,
        warmup_path,
        False,
    )
    warmup_checkpoint = torch.load(
        warmup_path, map_location=selected_device, weights_only=False
    )
    model.load_state_dict(warmup_checkpoint["model_state_dict"], strict=True)
    finetune_history, finetune_best = _run_stage(
        model,
        train_loader,
        validation_loader,
        config,
        selected_device,
        seed,
        class_names,
        "final_block_finetune",
        config.finetune_epochs,
        config.warmup_epochs,
        final_path,
        True,
    )
    reloaded_model, final_checkpoint = load_accuracy_checkpoint(
        final_path, map_location=selected_device
    )
    reloaded_validation = run_epoch(
        reloaded_model,
        validation_loader,
        "Validation",
        selected_device,
        config.label_smoothing,
        config.num_classes,
    )
    selected_validation = final_checkpoint["validation_metrics"]
    for metric in ("loss", "accuracy", "macro_f1"):
        if abs(float(reloaded_validation[metric]) - float(selected_validation[metric])) > 1e-10:
            raise RuntimeError(f"Reloaded Validation metric mismatch: {metric}")
    history = warmup_history + finetune_history
    result = {
        "architecture": config.architecture,
        "seed": seed,
        "device": str(selected_device),
        "checkpoint_path": str(final_path),
        "checkpoint_state_sha256": final_checkpoint["model_state_sha256"],
        "warmup_best": warmup_best,
        "finetune_best": finetune_best,
        "selected_validation": reloaded_validation,
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
    _write_json(run_dir / "history.json", history)
    _write_json(run_dir / "validation_result.json", result)
    return result


def configuration_id(config: AccuracyConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    smoothing = str(config.label_smoothing).replace(".", "p")
    return f"{config.architecture}_ls{smoothing}_{digest}"


def train_repeated_seeds(
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
    configurations: Sequence[AccuracyConfig],
    seeds: Sequence[int] = (42, 123, 2026),
    device: torch.device | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    if not configurations:
        raise ValueError("At least one configuration is required")
    configuration_ids = [configuration_id(config) for config in configurations]
    if len(set(configuration_ids)) != len(configuration_ids):
        raise ValueError("Configurations must be unique")
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("Repeated-seed evaluation requires unique seeds")
    development = load_authorized_development_manifest(
        manifest_path, dataset_root, class_names
    )
    selected_device = device or select_device()
    results = []
    for config in configurations:
        run_id = configuration_id(config)
        for seed in seeds:
            results.append(
                train_single_seed(
                    development,
                    dataset_root,
                    Path(output_root) / run_id / f"seed_{seed}",
                    config,
                    int(seed),
                    selected_device,
                    class_names,
                )
            )
    summaries = []
    for config in configurations:
        run_id = configuration_id(config)
        rows = [
            result
            for result in results
            if result["architecture"] == config.architecture
            and Path(result["checkpoint_path"]).parents[1].name == run_id
        ]
        macro_f1_values = [row["selected_validation"]["macro_f1"] for row in rows]
        accuracy_values = [row["selected_validation"]["accuracy"] for row in rows]
        summaries.append(
            {
                "configuration_id": run_id,
                "config": asdict(config),
                "seed_count": len(rows),
                "mean_validation_macro_f1": float(np.mean(macro_f1_values)),
                "std_validation_macro_f1": float(np.std(macro_f1_values, ddof=1)),
                "mean_validation_accuracy": float(np.mean(accuracy_values)),
                "std_validation_accuracy": float(np.std(accuracy_values, ddof=1)),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["mean_validation_macro_f1"],
            row["mean_validation_accuracy"],
        ),
        reverse=True,
    )
    output = {
        "selection_split": "Validation",
        "seeds": list(seeds),
        "device": str(selected_device),
        "summaries": summaries,
        "results": results,
        "selected_configuration_id": summaries[0]["configuration_id"],
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
    _write_json(Path(output_root) / "repeated_seed_summary.json", output)
    return output


def _predict_validation_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    model.eval()
    probabilities = []
    targets = []
    asset_ids: list[str] = []
    with torch.inference_mode():
        for inputs, batch_targets, batch_asset_ids in loader:
            logits = model(inputs.to(device, non_blocking=device.type == "cuda"))
            probabilities.append(torch.softmax(logits, dim=1).cpu())
            targets.append(batch_targets.to(torch.int64).cpu())
            asset_ids.extend(str(asset_id) for asset_id in batch_asset_ids)
    return torch.cat(probabilities), torch.cat(targets), tuple(asset_ids)


def _probability_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str],
) -> dict[str, Any]:
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.shape[0]:
        raise ValueError("Probability and target shapes are incompatible")
    probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    predictions = probabilities.argmax(dim=1)
    class_count = probabilities.shape[1]
    flat = targets * class_count + predictions
    confusion = torch.bincount(
        flat, minlength=class_count * class_count
    ).reshape(class_count, class_count)
    metrics = metrics_from_confusion(confusion, class_names)
    metrics["loss"] = float(
        -probabilities[torch.arange(targets.numel()), targets].log().mean().item()
    )
    return metrics


def select_validation_inference_strategy(
    checkpoint_paths: Sequence[Path],
    manifest_path: Path,
    dataset_root: Path,
    output_path: Path | None = None,
    tta_scales: Sequence[float] = (1.0, 0.95, 0.9),
    maximum_ensemble_members: int = 3,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    paths = tuple(Path(path).expanduser().resolve() for path in checkpoint_paths)
    if not paths:
        raise ValueError("At least one checkpoint is required")
    if len(set(paths)) != len(paths):
        raise ValueError("Checkpoint paths must be unique")
    if maximum_ensemble_members <= 0:
        raise ValueError("maximum_ensemble_members must be positive")
    scales = tuple(float(scale) for scale in tta_scales)
    if not scales or scales[0] != 1.0 or any(not 0.0 < scale <= 1.0 for scale in scales):
        raise ValueError("TTA scales must start at 1.0 and remain in (0, 1]")
    if len(set(scales)) != len(scales):
        raise ValueError("TTA scales must be unique")
    development = load_authorized_development_manifest(
        manifest_path, dataset_root, class_names
    )
    selected_device = device or select_device()
    prediction_bank: dict[tuple[str, float], torch.Tensor] = {}
    reference_targets: torch.Tensor | None = None
    reference_asset_ids: tuple[str, ...] | None = None
    checkpoint_metadata = {}
    for checkpoint_path in paths:
        model, checkpoint = load_accuracy_checkpoint(
            checkpoint_path, map_location=selected_device
        )
        if tuple(checkpoint["class_names"]) != tuple(class_names):
            raise RuntimeError("Checkpoint label order does not match the class contract")
        checkpoint_metadata[str(checkpoint_path)] = {
            "architecture": checkpoint["architecture"],
            "seed": checkpoint["seed"],
            "model_state_sha256": checkpoint["model_state_sha256"],
        }
        image_size = int(checkpoint["config"]["image_size"])
        for scale in scales:
            loader = build_loader(
                development,
                dataset_root,
                "Validation",
                build_evaluation_transform(image_size, scale),
                batch_size,
                0,
                num_workers,
                class_names,
            )
            probabilities, targets, asset_ids = _predict_validation_probabilities(
                model, loader, selected_device
            )
            if reference_targets is None:
                reference_targets = targets
                reference_asset_ids = asset_ids
            elif not torch.equal(reference_targets, targets) or reference_asset_ids != asset_ids:
                raise RuntimeError("Validation prediction order changed across views")
            prediction_bank[(str(checkpoint_path), scale)] = probabilities
    if reference_targets is None:
        raise RuntimeError("Validation prediction bank is empty")
    tta_profiles = ((1.0,), scales) if len(scales) > 1 else (scales,)
    candidates = []
    maximum_size = min(maximum_ensemble_members, len(paths))
    for member_count in range(1, maximum_size + 1):
        for members in itertools.combinations(paths, member_count):
            for profile in tta_profiles:
                tensors = [
                    prediction_bank[(str(member), scale)]
                    for member in members
                    for scale in profile
                ]
                averaged = torch.stack(tensors).mean(dim=0)
                metrics = _probability_metrics(
                    averaged, reference_targets, class_names
                )
                candidates.append(
                    {
                        "checkpoint_paths": [str(member) for member in members],
                        "member_count": member_count,
                        "tta_scales": list(profile),
                        "view_count": len(tensors),
                        "validation_metrics": metrics,
                    }
                )
    candidates.sort(
        key=lambda row: (
            row["validation_metrics"]["macro_f1"],
            row["validation_metrics"]["accuracy"],
            -row["validation_metrics"]["loss"],
            -row["view_count"],
        ),
        reverse=True,
    )
    result = {
        "selection_split": "Validation",
        "selection_primary": "validation_macro_f1",
        "selection_secondary": "validation_accuracy",
        "selected": candidates[0],
        "candidates": candidates,
        "checkpoint_metadata": checkpoint_metadata,
        "validation_asset_count": int(reference_targets.numel()),
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
    if output_path is not None:
        _write_json(output_path, result)
    return result
