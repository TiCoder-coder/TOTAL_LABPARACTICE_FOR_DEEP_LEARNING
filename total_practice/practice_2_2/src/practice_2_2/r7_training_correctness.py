from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import models

from .paths import get_practice_2_2_root


POLICY_RELATIVE_PATH = Path("configs/r7_training_correctness_policy.json")
LOSS_CONTRACT = "exact_global_cross_entropy_mean"
METRIC_SPLITS = ("Train", "Validation", "Test")


def load_r7_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R7 training-correctness policy schema")
    if policy.get("seed") != 42:
        raise RuntimeError("R7 must use seed 42")
    if policy.get("predecessor_phase") != "R6":
        raise RuntimeError("R7 predecessor must be R6")
    if policy.get("predecessor_gate_required") is not True:
        raise RuntimeError("R7 must require the R6 gate")
    if policy.get("loss_contract") != LOSS_CONTRACT:
        raise RuntimeError("R7 loss contract is invalid")
    if tuple(policy.get("supported_metric_splits", ())) != METRIC_SPLITS:
        raise RuntimeError("R7 metric split contract is invalid")
    baseline = policy.get("baseline_loss", {})
    if baseline.get("name") != "cross_entropy":
        raise RuntimeError("R7 baseline must use CrossEntropy")
    if baseline.get("class_weights") is not None:
        raise RuntimeError("R7 baseline CrossEntropy must be unweighted")
    if baseline.get("label_smoothing") != 0.0:
        raise RuntimeError("R7 baseline must disable label smoothing")
    if policy.get("class_weighting_requires_separate_experiment") is not True:
        raise RuntimeError("R7 class weighting must be a separate experiment")
    if policy.get("label_smoothing_requires_separate_experiment") is not True:
        raise RuntimeError("R7 label smoothing must be a separate experiment")
    checkpoint = policy.get("checkpoint", {})
    if checkpoint.get("architecture") != "resnet18":
        raise RuntimeError("R7 checkpoint architecture must be ResNet18")
    if checkpoint.get("constructor_weights") is not None:
        raise RuntimeError("R7 checkpoint reload must construct with weights=None")
    if checkpoint.get("strict_state_dict_loading") is not True:
        raise RuntimeError("R7 checkpoint loading must be strict")
    if policy.get("frozen_batchnorm_statistics") != "evaluation_mode":
        raise RuntimeError("R7 frozen BatchNorm policy is invalid")
    if policy.get("real_training_allowed") is not False:
        raise RuntimeError("R7 must not perform real training")
    if policy.get("validation_content_access_allowed") is not False:
        raise RuntimeError("R7 must prohibit Validation content access")
    if policy.get("test_content_access_allowed") is not False:
        raise RuntimeError("R7 must prohibit Test content access")
    if policy.get("test_loader_construction_allowed") is not False:
        raise RuntimeError("R7 must prohibit Test loader construction")
    return policy


@dataclass(frozen=True)
class CrossEntropyBatchTerms:
    optimization_loss: torch.Tensor
    numerator: float
    denominator: float
    sample_count: int


def cross_entropy_batch_terms(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> CrossEntropyBatchTerms:
    if logits.ndim != 2:
        raise ValueError("CrossEntropy logits must have shape [N, C]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("CrossEntropy targets must have shape [N]")
    if targets.numel() == 0:
        raise ValueError("CrossEntropy batch must not be empty")
    if targets.dtype != torch.long:
        raise ValueError("CrossEntropy targets must use torch.long")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if class_weights is not None:
        if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
            raise ValueError("class_weights must have one value per class")
        if class_weights.device != logits.device:
            raise ValueError("class_weights and logits must share a device")
        if not torch.isfinite(class_weights).all() or torch.any(class_weights < 0):
            raise ValueError("class_weights must be finite and non-negative")
    losses = functional.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    optimization_numerator = losses.sum()
    if class_weights is None:
        denominator_tensor = torch.tensor(
            targets.numel(), dtype=logits.dtype, device=logits.device
        )
        metric_denominator = float(targets.numel())
    else:
        denominator_tensor = class_weights[targets].sum()
        metric_denominator = float(
            class_weights[targets].detach().to(torch.float64).sum().item()
        )
    metric_numerator = float(
        losses.detach().to(torch.float64).sum().item()
    )
    if not torch.isfinite(optimization_numerator):
        raise RuntimeError("CrossEntropy numerator is not finite")
    if not torch.isfinite(denominator_tensor) or denominator_tensor.item() <= 0:
        raise RuntimeError("CrossEntropy denominator must be finite and positive")
    optimization_loss = optimization_numerator / denominator_tensor
    return CrossEntropyBatchTerms(
        optimization_loss=optimization_loss,
        numerator=metric_numerator,
        denominator=metric_denominator,
        sample_count=int(targets.numel()),
    )


class ExactCrossEntropyAccumulator:
    def __init__(self, split: str):
        if split not in METRIC_SPLITS:
            raise ValueError(f"Unsupported metric split: {split}")
        self.split = split
        self.numerator = 0.0
        self.denominator = 0.0
        self.sample_count = 0
        self.batch_count = 0

    def update(self, terms: CrossEntropyBatchTerms) -> None:
        if terms.denominator <= 0 or terms.sample_count <= 0:
            raise ValueError("CrossEntropy batch terms are invalid")
        self.numerator += terms.numerator
        self.denominator += terms.denominator
        self.sample_count += terms.sample_count
        self.batch_count += 1

    @property
    def mean(self) -> float:
        if self.denominator <= 0:
            raise RuntimeError("CrossEntropy accumulator is empty")
        return self.numerator / self.denominator

    def metric(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "contract": LOSS_CONTRACT,
            "value": self.mean,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "sample_count": self.sample_count,
            "batch_count": self.batch_count,
        }


@dataclass
class EarlyStoppingState:
    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    bad_epochs: int = 0
    should_stop: bool = False
    last_metric: float | None = None

    def update(self, value: float) -> bool:
        self.last_metric = float(value)
        if value < self.best - self.min_delta:
            self.best = float(value)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.should_stop = self.bad_epochs >= self.patience
        return self.should_stop


def route_corrected_validation_loss(
    validation_metric: Mapping[str, Any],
    scheduler: Any,
    early_stopping: EarlyStoppingState,
) -> dict[str, Any]:
    if validation_metric.get("split") != "Validation":
        raise ValueError("Training controls require a Validation metric")
    if validation_metric.get("contract") != LOSS_CONTRACT:
        raise ValueError("Training controls require corrected Validation loss")
    value = float(validation_metric["value"])
    scheduler.step(value)
    early_stopping.update(value)
    return {
        "metric_split": "Validation",
        "metric_contract": LOSS_CONTRACT,
        "scheduler_metric": value,
        "early_stopping_metric": early_stopping.last_metric,
        "early_stopping_best": early_stopping.best,
        "early_stopping_bad_epochs": early_stopping.bad_epochs,
        "early_stopping_should_stop": early_stopping.should_stop,
    }


def build_resnet18_classifier(
    num_classes: int,
    dropout: float = 0.2,
) -> nn.Module:
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    model = models.resnet18(weights=None)
    features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
    return model


def configure_trainable_parameters(model: nn.Module, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "head_only":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif mode == "partial_finetune":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unsupported training mode: {mode}")


def set_frozen_batchnorm_eval(model: nn.Module) -> int:
    frozen = 0
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        parameters = tuple(module.parameters(recurse=False))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            module.eval()
            frozen += 1
    return frozen


def build_named_adamw(
    model: nn.Module,
    head_learning_rate: float,
    backbone_learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    if min(head_learning_rate, backbone_learning_rate) <= 0:
        raise ValueError("Learning rates must be positive")
    head_parameters = [
        parameter for parameter in model.fc.parameters() if parameter.requires_grad
    ]
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    groups = []
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_learning_rate,
                "group_name": "backbone",
            }
        )
    if head_parameters:
        groups.append(
            {
                "params": head_parameters,
                "lr": head_learning_rate,
                "group_name": "head",
            }
        )
    if not groups:
        raise ValueError("Model contains no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    result = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", f"group_{index}"))
        if name in result:
            raise RuntimeError(f"Duplicate optimizer group name: {name}")
        result[name] = float(group["lr"])
    return result


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def save_complete_checkpoint(
    path: Path,
    model: nn.Module,
    num_classes: int,
    dropout: float,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "architecture": "resnet18",
        "constructor_weights": None,
        "num_classes": int(num_classes),
        "dropout": float(dropout),
        "model_state_dict": model.state_dict(),
        "model_state_sha256": state_dict_sha256(model.state_dict()),
        "metadata": dict(metadata or {}),
    }
    torch.save(checkpoint, path)
    return path


def load_complete_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location=map_location,
        weights_only=False,
    )
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("Unsupported R7 checkpoint schema")
    if checkpoint.get("architecture") != "resnet18":
        raise RuntimeError("R7 checkpoint architecture mismatch")
    if checkpoint.get("constructor_weights") is not None:
        raise RuntimeError("R7 checkpoint was not declared offline")
    model = build_resnet18_classifier(
        num_classes=int(checkpoint["num_classes"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    live_hash = state_dict_sha256(model.state_dict())
    if live_hash != checkpoint.get("model_state_sha256"):
        raise RuntimeError("R7 checkpoint state hash mismatch")
    model.to(map_location)
    model.eval()
    return model, checkpoint


def build_r7_report(
    r6_report: Mapping[str, Any],
    checks: Mapping[str, bool],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r7_policy())
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    blocked_reasons = []
    if not r6_report.get("gate_passed"):
        blocked_reasons.append("R6_GATE_NOT_PASSED")
    if r6_report.get("selected_recipe_id") is None:
        blocked_reasons.append("TRANSFORM_RECIPE_NOT_SELECTED")
    if "TRAIN_MANIFEST_NOT_AUTHORIZED" in r6_report.get("blocked_reasons", []):
        blocked_reasons.append("TRAIN_MANIFEST_NOT_AUTHORIZED")
    if failed_checks:
        blocked_reasons.append("R7_CORRECTNESS_CHECKS_FAILED")
    return {
        "schema_version": 1,
        "phase": "R7",
        "lineage": policy["policy_version"],
        "status": "passed" if not blocked_reasons else "blocked",
        "gate_passed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "correctness_exit_checks_passed": not failed_checks,
        "failed_correctness_checks": failed_checks,
        "loss_contract": policy["loss_contract"],
        "baseline_loss": policy["baseline_loss"],
        "class_weighting_is_separate_experiment": True,
        "label_smoothing_is_separate_experiment": True,
        "checkpoint_constructor_weights": None,
        "strict_checkpoint_loading": True,
        "head_and_backbone_learning_rates_logged_separately": checks.get(
            "named_learning_rates", False
        ),
        "frozen_batchnorm_verified": checks.get("frozen_batchnorm", False),
        "scheduler_consumed_corrected_validation_loss": checks.get(
            "scheduler_metric_routing", False
        ),
        "early_stopping_consumed_corrected_validation_loss": checks.get(
            "early_stopping_metric_routing", False
        ),
        "offline_reload_predictions_identical": checks.get(
            "offline_checkpoint_reload", False
        ),
        "test_loader_imported": False,
        "test_loader_constructed": False,
        "real_training_performed": False,
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "validation_evaluated": False,
        "test_evaluated": False,
        "source_images_mutated": False,
        "canonical_notebook_mutated": False,
        "successor_phase_executed": False,
    }


def write_json(value: Any, output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    root = get_practice_2_2_root().resolve()
    if root not in output_path.parents:
        raise RuntimeError("R7 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return output_path
