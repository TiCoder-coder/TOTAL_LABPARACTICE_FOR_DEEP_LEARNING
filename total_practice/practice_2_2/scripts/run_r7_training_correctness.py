from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as functional

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.r7_training_correctness import (
    LOSS_CONTRACT,
    EarlyStoppingState,
    ExactCrossEntropyAccumulator,
    build_named_adamw,
    build_r7_report,
    build_resnet18_classifier,
    configure_trainable_parameters,
    cross_entropy_batch_terms,
    load_complete_checkpoint,
    load_r7_policy,
    optimizer_learning_rates,
    route_corrected_validation_loss,
    save_complete_checkpoint,
    set_frozen_batchnorm_eval,
    state_dict_sha256,
    tensor_sha256,
    write_json,
)
from practice_2_2.resources import file_sha256


def aggregate_partitions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    partitions: list[int],
    split: str,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> dict:
    if sum(partitions) != targets.numel():
        raise ValueError("Partition sizes must cover every sample")
    accumulator = ExactCrossEntropyAccumulator(split)
    offset = 0
    for size in partitions:
        selected = slice(offset, offset + size)
        accumulator.update(
            cross_entropy_batch_terms(
                logits[selected],
                targets[selected],
                class_weights=class_weights,
                label_smoothing=label_smoothing,
            )
        )
        offset += size
    return accumulator.metric()


def verify_losses() -> tuple[dict, dict[str, bool]]:
    torch.manual_seed(42)
    logits = torch.randn(17, 5, dtype=torch.float64)
    targets = torch.tensor(
        [0, 1, 2, 3, 4, 0, 1, 1, 2, 3, 4, 4, 4, 0, 2, 3, 1],
        dtype=torch.long,
    )
    weights = torch.tensor([0.7, 1.2, 1.8, 0.9, 2.4], dtype=torch.float64)
    cases = {
        "unweighted_baseline": {
            "weights": None,
            "label_smoothing": 0.0,
        },
        "weighted_separate_experiment": {
            "weights": weights,
            "label_smoothing": 0.0,
        },
        "smoothed_separate_experiment": {
            "weights": None,
            "label_smoothing": 0.1,
        },
        "weighted_smoothed_separate_experiment": {
            "weights": weights,
            "label_smoothing": 0.1,
        },
    }
    patterns = [[17], [1, 16], [3, 5, 2, 7], [4, 4, 4, 4, 1]]
    records = []
    checks = {}
    for name, case in cases.items():
        reference = functional.cross_entropy(
            logits,
            targets,
            weight=case["weights"],
            label_smoothing=case["label_smoothing"],
            reduction="mean",
        ).item()
        values = []
        for partitions in patterns:
            metric = aggregate_partitions(
                logits,
                targets,
                partitions,
                "Train",
                class_weights=case["weights"],
                label_smoothing=case["label_smoothing"],
            )
            values.append(metric["value"])
        maximum_reference_error = max(abs(value - reference) for value in values)
        maximum_partition_difference = max(values) - min(values)
        passed = maximum_reference_error < 1e-12 and maximum_partition_difference < 1e-12
        checks[name] = passed
        records.append(
            {
                "case": name,
                "reference_mean": reference,
                "partition_patterns": patterns,
                "partition_means": values,
                "maximum_reference_error": maximum_reference_error,
                "maximum_partition_difference": maximum_partition_difference,
                "passed": passed,
            }
        )
    report = {
        "schema_version": 1,
        "loss_contract": LOSS_CONTRACT,
        "sample_count": targets.numel(),
        "cases": records,
    }
    return report, {
        "unweighted_loss": checks["unweighted_baseline"],
        "weighted_loss": checks["weighted_separate_experiment"],
        "label_smoothing_loss": checks["smoothed_separate_experiment"],
        "weighted_label_smoothing_loss": checks[
            "weighted_smoothed_separate_experiment"
        ],
        "batch_partition_invariance": all(checks.values()),
    }


def verify_validation_controls() -> tuple[dict, dict[str, bool]]:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=0
    )
    early_stopping = EarlyStoppingState(patience=2)
    values = [1.2, 1.0, 1.1]
    routing = []
    for value in values:
        routing.append(
            route_corrected_validation_loss(
                {
                    "split": "Validation",
                    "contract": LOSS_CONTRACT,
                    "value": value,
                },
                scheduler,
                early_stopping,
            )
        )
    scheduler_values = [record["scheduler_metric"] for record in routing]
    early_stopping_values = [
        record["early_stopping_metric"] for record in routing
    ]
    report = {
        "schema_version": 1,
        "metric_split": "Validation",
        "metric_contract": LOSS_CONTRACT,
        "corrected_validation_losses": values,
        "scheduler_metrics": scheduler_values,
        "early_stopping_metrics": early_stopping_values,
        "scheduler_best": float(scheduler.best),
        "early_stopping_best": early_stopping.best,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    return report, {
        "scheduler_metric_routing": scheduler_values == values,
        "early_stopping_metric_routing": early_stopping_values == values,
    }


def verify_model_and_checkpoint(policy: dict) -> tuple[dict, dict, dict[str, bool]]:
    torch.manual_seed(policy["seed"])
    model = build_resnet18_classifier(num_classes=10, dropout=0.2)
    configure_trainable_parameters(model, "partial_finetune")
    model.train()
    frozen_batchnorm_count = set_frozen_batchnorm_eval(model)
    frozen_batchnorm_modules = []
    trainable_batchnorm_modules = []
    frozen_batchnorm_correct = True
    trainable_batchnorm_correct = True
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        parameters = tuple(module.parameters(recurse=False))
        is_trainable = any(parameter.requires_grad for parameter in parameters)
        if is_trainable:
            trainable_batchnorm_modules.append(name)
            trainable_batchnorm_correct &= module.training
        else:
            frozen_batchnorm_modules.append(name)
            frozen_batchnorm_correct &= not module.training
    optimizer = build_named_adamw(
        model,
        head_learning_rate=policy["optimizer"]["head_learning_rate"],
        backbone_learning_rate=policy["optimizer"]["backbone_learning_rate"],
        weight_decay=policy["optimizer"]["weight_decay"],
    )
    learning_rates = optimizer_learning_rates(optimizer)
    grouped_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    model.eval()
    torch.manual_seed(policy["seed"] + 1)
    inputs = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        original_output = model(inputs)
    original_state_hash = state_dict_sha256(model.state_dict())
    with tempfile.TemporaryDirectory(prefix="r7_offline_reload_") as directory:
        checkpoint_path = Path(directory) / "complete_checkpoint.pt"
        save_complete_checkpoint(
            checkpoint_path,
            model,
            num_classes=10,
            dropout=0.2,
            metadata={"phase": "R7", "training_performed": False},
        )
        with patch(
            "torch.hub.load_state_dict_from_url",
            side_effect=RuntimeError("network access forbidden"),
        ), patch(
            "torchvision.models._api.load_state_dict_from_url",
            side_effect=RuntimeError("network access forbidden"),
        ):
            loaded, checkpoint = load_complete_checkpoint(checkpoint_path)
        with torch.inference_mode():
            loaded_output = loaded(inputs)
    loaded_state_hash = state_dict_sha256(loaded.state_dict())
    predictions_identical = torch.equal(original_output, loaded_output)
    maximum_logit_difference = float((original_output - loaded_output).abs().max())
    model_report = {
        "schema_version": 1,
        "architecture": "resnet18",
        "constructor_weights": None,
        "training_mode_verified": "partial_finetune",
        "frozen_batchnorm_count": frozen_batchnorm_count,
        "frozen_batchnorm_modules": frozen_batchnorm_modules,
        "trainable_batchnorm_modules": trainable_batchnorm_modules,
        "frozen_batchnorm_modules_in_eval": frozen_batchnorm_correct,
        "trainable_batchnorm_modules_in_train": trainable_batchnorm_correct,
        "learning_rates": learning_rates,
        "optimizer_parameter_coverage_exact": grouped_parameter_ids
        == trainable_parameter_ids,
    }
    checkpoint_report = {
        "schema_version": 1,
        "architecture": checkpoint["architecture"],
        "constructor_weights": checkpoint["constructor_weights"],
        "strict_state_dict_loading": True,
        "network_access_guarded": True,
        "model_state_sha256_before_save": original_state_hash,
        "model_state_sha256_after_load": loaded_state_hash,
        "prediction_tensor_sha256_before_save": tensor_sha256(original_output),
        "prediction_tensor_sha256_after_load": tensor_sha256(loaded_output),
        "predictions_identical": predictions_identical,
        "maximum_logit_difference": maximum_logit_difference,
        "checkpoint_persisted_after_verification": False,
    }
    checks = {
        "frozen_batchnorm": bool(
            frozen_batchnorm_count
            and frozen_batchnorm_correct
            and trainable_batchnorm_correct
        ),
        "named_learning_rates": learning_rates
        == {
            "backbone": policy["optimizer"]["backbone_learning_rate"],
            "head": policy["optimizer"]["head_learning_rate"],
        },
        "optimizer_parameter_coverage": grouped_parameter_ids
        == trainable_parameter_ids,
        "offline_checkpoint_reload": bool(
            predictions_identical
            and maximum_logit_difference == 0.0
            and original_state_hash == loaded_state_hash
        ),
    }
    return model_report, checkpoint_report, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r6-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_r7_policy()
    r6_report = json.loads(args.r6_verification.read_text())
    root = get_practice_2_2_root().resolve()
    notebook_path = root / "notebooks/04_canonical_report.ipynb"
    notebook_hash_before = file_sha256(notebook_path)
    loss_report, loss_checks = verify_losses()
    control_report, control_checks = verify_validation_controls()
    model_report, checkpoint_report, model_checks = verify_model_and_checkpoint(policy)
    checks = {**loss_checks, **control_checks, **model_checks}
    checks["canonical_notebook_unchanged"] = (
        file_sha256(notebook_path) == notebook_hash_before
    )
    report = build_r7_report(r6_report, checks, policy)
    report["correctness_checks"] = checks
    report["canonical_notebook_sha256"] = notebook_hash_before
    report["synthetic_tensor_verification_only"] = True
    artifact_paths = [
        write_json(policy, args.output_dir / "policy_snapshot.json"),
        write_json(loss_report, args.output_dir / "loss_verification.json"),
        write_json(
            control_report, args.output_dir / "validation_control_verification.json"
        ),
        write_json(model_report, args.output_dir / "model_verification.json"),
        write_json(
            checkpoint_report,
            args.output_dir / "checkpoint_reload_verification.json",
        ),
        write_json(report, args.output_dir / "training_correctness_report.json"),
        write_json(report, args.report_output),
    ]
    artifact_manifest = {
        "schema_version": 1,
        "lineage": policy["policy_version"],
        "synthetic_tensor_verification_only": True,
        "real_training_performed": False,
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "test_loader_constructed": False,
        "source_images_mutated": False,
        "canonical_notebook_mutated": False,
        "canonical_notebook_sha256": notebook_hash_before,
        "artifacts": [
            {
                "path": path.resolve().relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(artifact_manifest, args.output_dir / "artifact_manifest.json")
    output = {
        "status": report["status"],
        "gate_passed": report["gate_passed"],
        "correctness_exit_checks_passed": report[
            "correctness_exit_checks_passed"
        ],
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(output, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
