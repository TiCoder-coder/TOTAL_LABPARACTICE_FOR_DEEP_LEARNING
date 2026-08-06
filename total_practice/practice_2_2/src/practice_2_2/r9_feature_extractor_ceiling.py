from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torchvision import models

from .paths import get_practice_2_2_root
from .r7_training_correctness import set_frozen_batchnorm_eval, state_dict_sha256
from .resources import file_sha256


POLICY_RELATIVE_PATH = Path("configs/r9_feature_extractor_ceiling_policy.json")
RESULT_FIELDS = {
    "experiment_id",
    "seed",
    "initial_state_sha256",
    "train_accuracy",
    "validation_accuracy",
    "validation_macro_f1",
    "best_epoch",
}


def load_r9_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R9 feature-ceiling policy schema")
    if policy.get("predecessor_phase") != "R8":
        raise RuntimeError("R9 predecessor must be R8")
    if policy.get("predecessor_gate_required") is not True:
        raise RuntimeError("R9 must require the R8 gate")
    if policy.get("baseline_source") != "authorized_r8_candidate":
        raise RuntimeError("R9 baseline must come from R8")
    if policy.get("experiment_seeds") != [42, 123, 2026]:
        raise RuntimeError("R9 repeated-seed contract is invalid")
    if policy.get("higher_resolution_experiment_enabled") is not False:
        raise RuntimeError("R9 higher resolution must remain a separate experiment")
    if policy.get("ocr_or_image_text_features_enabled") is not False:
        raise RuntimeError("R9 must remain image-only")
    if policy.get("test_evidence_allowed") is not False:
        raise RuntimeError("R9 must prohibit Test evidence")
    if policy.get("test_content_access_allowed") is not False:
        raise RuntimeError("R9 must prohibit Test content access")
    if policy.get("test_loader_construction_allowed") is not False:
        raise RuntimeError("R9 must prohibit Test loader construction")
    experiments = policy.get("experiments", {})
    if tuple(experiments) != ("A0", "A1"):
        raise RuntimeError("R9 must define A0 and A1 in order")
    if experiments["A0"]["architecture"] != "resnet18":
        raise RuntimeError("R9 A0 must be the ResNet18 baseline")
    if experiments["A1"]["architecture"] != "efficientnet_b0":
        raise RuntimeError("R9 A1 must be EfficientNet-B0")
    parent = experiments[experiments["A1"]["parent"]]
    changed = [
        field
        for field in policy["controlled_experiment_fields"]
        if experiments["A1"][field] != parent[field]
    ]
    if changed != [experiments["A1"]["single_change"]]:
        raise RuntimeError("R9 A1 must change architecture only")
    if experiments["A0"]["image_size"] != experiments["A1"]["image_size"]:
        raise RuntimeError("R9 architecture comparison must share image size")
    if policy.get("same_split_required") is not True:
        raise RuntimeError("R9 experiments must share the split")
    if policy.get("same_transform_required") is not True:
        raise RuntimeError("R9 experiments must share the transform")
    if policy.get("same_seeds_required") is not True:
        raise RuntimeError("R9 experiments must share seeds")
    if policy.get("same_staged_protocol_required") is not True:
        raise RuntimeError("R9 experiments must share the staged protocol")
    return policy


def create_backbone(architecture: str) -> nn.Module:
    if architecture == "resnet18":
        return models.resnet18(weights=None)
    if architecture == "efficientnet_b0":
        return models.efficientnet_b0(weights=None)
    raise ValueError(f"Unsupported R9 architecture: {architecture}")


def load_authorized_backbone_state(
    architecture: str,
    checkpoint_path: Path,
    expected_file_sha256: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"R9 pretrained checkpoint is missing: {path}")
    live_hash = file_sha256(path)
    if live_hash != expected_file_sha256:
        raise RuntimeError("R9 pretrained checkpoint SHA-256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise RuntimeError("R9 pretrained checkpoint must contain a state dict")
    model = create_backbone(architecture)
    model.load_state_dict(state, strict=True)
    return dict(state), {
        "architecture": architecture,
        "checkpoint_file_sha256": live_hash,
        "state_dict_sha256": state_dict_sha256(state),
        "offline_loaded": True,
    }


def build_classifier_model(
    architecture: str,
    backbone_state: Mapping[str, torch.Tensor],
    num_classes: int,
    dropout: float,
    head_seed: int,
) -> nn.Module:
    model = create_backbone(architecture)
    model.load_state_dict(backbone_state, strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(head_seed)
        if architecture == "resnet18":
            features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(features, num_classes)
            )
        else:
            features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(features, num_classes)
            )
    return model


def classifier_module(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "resnet18":
        return model.fc
    if architecture == "efficientnet_b0":
        return model.classifier
    raise ValueError(f"Unsupported R9 architecture: {architecture}")


def final_backbone_modules(model: nn.Module, architecture: str) -> tuple[nn.Module, ...]:
    if architecture == "resnet18":
        return (model.layer4,)
    if architecture == "efficientnet_b0":
        return (model.features[7], model.features[8])
    raise ValueError(f"Unsupported R9 architecture: {architecture}")


def configure_architecture_stage(
    model: nn.Module,
    architecture: str,
    stage: str,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier_module(model, architecture).parameters():
        parameter.requires_grad = True
    if stage == "head_warmup":
        return
    if stage != "final_block_finetune":
        raise ValueError(f"Unsupported R9 training stage: {stage}")
    for module in final_backbone_modules(model, architecture):
        for parameter in module.parameters():
            parameter.requires_grad = True


def trainable_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def build_architecture_optimizer(
    model: nn.Module,
    architecture: str,
    policy: Mapping[str, Any],
) -> torch.optim.AdamW:
    classifier_parameters = [
        parameter
        for parameter in classifier_module(model, architecture).parameters()
        if parameter.requires_grad
    ]
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in classifier_ids
    ]
    config = policy["optimizer"]
    groups = []
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": float(config["backbone_learning_rate"]),
                "group_name": "backbone",
            }
        )
    groups.append(
        {
            "params": classifier_parameters,
            "lr": float(config["head_learning_rate"]),
            "group_name": "head",
        }
    )
    return torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))


def optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group["group_name"]): float(group["lr"])
        for group in optimizer.param_groups
    }


def prepare_training_mode(model: nn.Module) -> dict[str, Any]:
    model.train()
    frozen_batchnorm_count = set_frozen_batchnorm_eval(model)
    return {
        "frozen_batchnorm_count": frozen_batchnorm_count,
        "trainable_parameter_names": trainable_parameter_names(model),
    }


def save_architecture_checkpoint(
    path: Path,
    model: nn.Module,
    architecture: str,
    num_classes: int,
    dropout: float,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "phase": "R9",
        "architecture": architecture,
        "constructor_weights": None,
        "num_classes": int(num_classes),
        "dropout": float(dropout),
        "model_state_dict": model.state_dict(),
        "model_state_sha256": state_dict_sha256(model.state_dict()),
    }
    torch.save(checkpoint, path)
    return path


def load_architecture_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location=map_location,
        weights_only=False,
    )
    if checkpoint.get("schema_version") != 1 or checkpoint.get("phase") != "R9":
        raise RuntimeError("Unsupported R9 checkpoint schema")
    if checkpoint.get("constructor_weights") is not None:
        raise RuntimeError("R9 checkpoint reconstruction must be offline")
    architecture = str(checkpoint["architecture"])
    model = create_backbone(architecture)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        if architecture == "resnet18":
            features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(float(checkpoint["dropout"])),
                nn.Linear(features, int(checkpoint["num_classes"])),
            )
        else:
            features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(float(checkpoint["dropout"])),
                nn.Linear(features, int(checkpoint["num_classes"])),
            )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    live_hash = state_dict_sha256(model.state_dict())
    if live_hash != checkpoint.get("model_state_sha256"):
        raise RuntimeError("R9 checkpoint state hash mismatch")
    model.to(map_location)
    model.eval()
    return model, checkpoint


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    return 0


def profile_architecture(
    model: nn.Module,
    architecture: str,
    image_size: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    model = model.cpu().eval()
    torch.manual_seed(9009)
    inputs = torch.randn(1, 3, image_size, image_size)
    activation_bytes = 0
    handles = []

    def collect_activation_bytes(module: nn.Module, values: Any, output: Any) -> None:
        nonlocal activation_bytes
        activation_bytes += _tensor_bytes(output)

    for module in model.modules():
        if not tuple(module.children()):
            handles.append(module.register_forward_hook(collect_activation_bytes))
    with torch.inference_mode():
        output = model(inputs)
    for handle in handles:
        handle.remove()
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(inputs)
        latencies = []
        for _ in range(measured_iterations):
            started = time.perf_counter()
            model(inputs)
            latencies.append((time.perf_counter() - started) * 1000.0)
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    return {
        "architecture": architecture,
        "device": "cpu",
        "input_shape": list(inputs.shape),
        "output_shape": list(output.shape),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_memory_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
        "buffer_memory_bytes": sum(
            buffer.numel() * buffer.element_size() for buffer in buffers
        ),
        "forward_activation_bytes_sum": activation_bytes,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_std": statistics.pstdev(latencies),
        "performance_metric_source": "synthetic_input_only",
    }


def _validate_result_rows(
    results: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> None:
    expected_pairs = {
        (experiment_id, seed)
        for experiment_id in policy["experiments"]
        for seed in policy["experiment_seeds"]
    }
    observed_pairs = set()
    hashes: dict[str, set[str]] = {
        experiment_id: set() for experiment_id in policy["experiments"]
    }
    for row in results:
        if set(row) != RESULT_FIELDS:
            raise ValueError("R9 result contains missing or unauthorized fields")
        experiment_id = str(row["experiment_id"])
        pair = (experiment_id, int(row["seed"]))
        if pair in observed_pairs:
            raise ValueError("R9 result contains a duplicate architecture seed")
        observed_pairs.add(pair)
        if experiment_id not in hashes:
            raise ValueError("R9 result contains an unknown experiment")
        hashes[experiment_id].add(str(row["initial_state_sha256"]))
        for name in ("train_accuracy", "validation_accuracy", "validation_macro_f1"):
            if not 0.0 <= float(row[name]) <= 1.0:
                raise ValueError(f"R9 result metric is invalid: {name}")
    if observed_pairs != expected_pairs:
        raise ValueError("R9 results do not cover every architecture seed")
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("R9 architecture seeds do not share initialization")


def summarize_architecture_results(
    results: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = dict(policy or load_r9_policy())
    _validate_result_rows(results, policy)
    summaries = []
    for experiment_id in policy["experiments"]:
        rows = [row for row in results if row["experiment_id"] == experiment_id]
        macro_f1 = [float(row["validation_macro_f1"]) for row in rows]
        validation_accuracy = [float(row["validation_accuracy"]) for row in rows]
        train_accuracy = [float(row["train_accuracy"]) for row in rows]
        gaps = [
            train - validation
            for train, validation in zip(train_accuracy, validation_accuracy)
        ]
        summaries.append(
            {
                "experiment_id": experiment_id,
                "architecture": policy["experiments"][experiment_id]["architecture"],
                "seed_count": len(rows),
                "seeds": sorted(int(row["seed"]) for row in rows),
                "initial_state_sha256": str(rows[0]["initial_state_sha256"]),
                "mean_train_accuracy": statistics.mean(train_accuracy),
                "mean_validation_accuracy": statistics.mean(validation_accuracy),
                "std_validation_accuracy": statistics.stdev(validation_accuracy),
                "mean_validation_macro_f1": statistics.mean(macro_f1),
                "std_validation_macro_f1": statistics.stdev(macro_f1),
                "mean_generalization_gap": statistics.mean(gaps),
                "test_evidence_used": False,
            }
        )
    return summaries


def select_architecture_from_validation(
    summaries: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r9_policy())
    by_id = {str(row["experiment_id"]): row for row in summaries}
    if set(by_id) != {"A0", "A1"}:
        raise ValueError("R9 architecture summary coverage is incomplete")
    if any(row.get("test_evidence_used") is not False for row in summaries):
        raise ValueError("R9 architecture selection cannot use Test evidence")
    baseline = by_id["A0"]
    comparator = by_id["A1"]
    selection = policy["selection"]
    macro_improvement = (
        float(comparator["mean_validation_macro_f1"])
        - float(baseline["mean_validation_macro_f1"])
    )
    accuracy_change = (
        float(comparator["mean_validation_accuracy"])
        - float(baseline["mean_validation_accuracy"])
    )
    gap_reduction = (
        float(baseline["mean_generalization_gap"])
        - float(comparator["mean_generalization_gap"])
    )
    train_only_improvement = (
        float(comparator["mean_train_accuracy"])
        > float(baseline["mean_train_accuracy"])
        and macro_improvement < selection["minimum_macro_f1_improvement"]
    )
    eligible = (
        macro_improvement >= selection["minimum_macro_f1_improvement"]
        and accuracy_change >= -selection["maximum_validation_accuracy_decrease"]
        and gap_reduction >= selection["minimum_generalization_gap_reduction"]
        and not train_only_improvement
    )
    return {
        "selected_experiment_id": "A1" if eligible else "A0",
        "comparator_advances": eligible,
        "macro_f1_improvement": macro_improvement,
        "validation_accuracy_change": accuracy_change,
        "generalization_gap_reduction": gap_reduction,
        "train_only_improvement": train_only_improvement,
        "selection_source": "Validation_only",
        "test_evidence_used": False,
    }


def build_r9_report(
    r8_report: Mapping[str, Any],
    r6_report: Mapping[str, Any],
    checks: Mapping[str, bool],
    baseline_pretrained_available: bool,
    comparator_pretrained_available: bool,
    comparator_hash_frozen: bool,
    repeated_seed_evidence_available: bool,
    authorized_architecture_id: str | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r9_policy())
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    blocked_reasons = []
    if not r8_report.get("gate_passed"):
        blocked_reasons.append("R8_GATE_NOT_PASSED")
    if r8_report.get("authorized_candidate_experiment_id") is None:
        blocked_reasons.append("R8_CANDIDATE_NOT_AUTHORIZED")
    if r6_report.get("selected_recipe_id") is None:
        blocked_reasons.append("TRANSFORM_RECIPE_NOT_SELECTED")
    if "TRAIN_MANIFEST_NOT_AUTHORIZED" in r6_report.get("blocked_reasons", []):
        blocked_reasons.append("TRAIN_MANIFEST_NOT_AUTHORIZED")
    if not baseline_pretrained_available:
        blocked_reasons.append("BASELINE_PRETRAINED_STATE_UNAVAILABLE")
    if not comparator_pretrained_available:
        blocked_reasons.append("COMPARATOR_PRETRAINED_STATE_UNAVAILABLE")
    if not comparator_hash_frozen:
        blocked_reasons.append("COMPARATOR_PRETRAINED_HASH_NOT_FROZEN")
    if not repeated_seed_evidence_available:
        blocked_reasons.append("REPEATED_SEED_VALIDATION_EVIDENCE_UNAVAILABLE")
    if failed_checks:
        blocked_reasons.append("R9_PROTOCOL_CHECKS_FAILED")
    if authorized_architecture_id is None:
        blocked_reasons.append("NO_R9_ARCHITECTURE_AUTHORIZED")
    return {
        "schema_version": 1,
        "phase": "R9",
        "lineage": policy["policy_version"],
        "status": "passed" if not blocked_reasons else "blocked",
        "gate_passed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "protocol_exit_checks_passed": not failed_checks,
        "failed_protocol_checks": failed_checks,
        "experiments": list(policy["experiments"]),
        "architectures": [
            policy["experiments"][item]["architecture"]
            for item in policy["experiments"]
        ],
        "experiment_seeds": policy["experiment_seeds"],
        "image_size": policy["image_size"],
        "higher_resolution_experiment_executed": False,
        "ocr_or_image_text_features_executed": False,
        "architecture_profiles_reported": checks.get("architecture_profiles", False),
        "offline_checkpoint_reload_verified": checks.get(
            "offline_checkpoint_reload", False
        ),
        "staged_protocol_compatibility_verified": checks.get(
            "staged_protocol_compatibility", False
        ),
        "repeated_seed_selection_logic_verified": checks.get(
            "repeated_seed_selection", False
        ),
        "baseline_pretrained_state_available": baseline_pretrained_available,
        "comparator_pretrained_state_available": comparator_pretrained_available,
        "comparator_pretrained_hash_frozen": comparator_hash_frozen,
        "repeated_seed_validation_evidence_available": repeated_seed_evidence_available,
        "authorized_architecture_experiment_id": authorized_architecture_id,
        "selection_source": "Validation_only",
        "test_evidence_used": False,
        "test_loader_imported": False,
        "test_loader_constructed": False,
        "real_training_performed": False,
        "synthetic_protocol_verification_only": True,
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
        raise RuntimeError("R9 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return output_path
