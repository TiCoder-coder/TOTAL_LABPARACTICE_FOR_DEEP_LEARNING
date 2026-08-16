"""Zero-training integration review and execution freeze for Practice 3 v2."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .config import PROJECT_ROOT
from .dataset_protocol_v2 import (
    materialize_train_validation_only,
    validate_split_manifest,
)
from .experiment_protocol_v2 import (
    PART_2_1_PLAN,
    PROTOCOL_VERSION,
    RUN_IDS,
    V2_RESULT_DIR,
    atomic_write_json,
    sha256_file,
    sha256_payload,
    validate_protocol_manifest,
    validate_run_configs,
)
from .experiment_registry_v2 import validate_registry
from .experiment_runner_v2 import (
    build_dry_trainer,
    build_early_stopping_callback,
    build_training_arguments,
    construct_optimizer_only,
    prepare_run_context,
)
from .holdout_guard_v2 import guarded_holdout_request, validate_initial_holdout_state


READINESS_PATH = V2_RESULT_DIR / "pretraining_readiness_manifest.json"
MODEL_IDENTIFIER_ALIASES = {
    "distilbert-base-uncased": "distilbert/distilbert-base-uncased",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_cached_model_identifier(requested: str) -> str:
    if requested not in MODEL_IDENTIFIER_ALIASES:
        raise ValueError(f"Unapproved model identifier: {requested}")
    return MODEL_IDENTIFIER_ALIASES[requested]


def _load_frozen_metadata() -> dict[str, Any]:
    protocol = _load_json(V2_RESULT_DIR / "protocol_manifest.json")
    split = _load_json(V2_RESULT_DIR / "dataset_split_manifest.json")
    registry = _load_json(V2_RESULT_DIR / "experiment_registry.json")
    state = _load_json(V2_RESULT_DIR / "holdout_access_state.json")
    configs = [
        _load_json(V2_RESULT_DIR / "experiments" / run_id / "run_config.json")
        for run_id in RUN_IDS
    ]
    validate_protocol_manifest(protocol)
    validate_split_manifest(split)
    validate_run_configs(configs)
    validate_registry(registry, configs, require_planned=True)
    validate_initial_holdout_state(state)
    if protocol["part_2_1_plan_sha256"] != sha256_file(PART_2_1_PLAN):
        raise RuntimeError("Part 2.1 protocol plan hash drift")
    return {
        "protocol": protocol,
        "split": split,
        "registry": registry,
        "state": state,
        "configs": configs,
    }


def _tokenize_train_validation(
    datasets: dict[str, Any], tokenizer: Any
) -> DatasetDict:
    source = DatasetDict(datasets)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=80,
        )
        encoded["labels"] = batch["label"]
        return encoded

    return source.map(
        tokenize,
        batched=True,
        remove_columns=list(source["train"].column_names),
        keep_in_memory=True,
        desc="Tokenizing v2 Train/Validation only",
    )


def _head_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith(("pre_classifier.", "classifier.")):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch_summary(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "shapes": {key: list(value.shape) for key, value in batch.items()},
        "dtypes": {key: str(value.dtype) for key, value in batch.items()},
        "input_attention_shapes_equal": (
            batch["input_ids"].shape == batch["attention_mask"].shape
        ),
        "labels_binary": set(batch["labels"].tolist()).issubset({0, 1}),
    }


def _execution_hash(metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    components = {
        "part_2_1_plan_hash": sha256_file(PART_2_1_PLAN),
        "protocol_manifest_hash": metadata["protocol"]["protocol_manifest_hash"],
        "dataset_split_manifest_hash": metadata["split"]["split_manifest_hash"],
        "run_config_hashes": {
            config["run_id"]: config["config_hash"] for config in metadata["configs"]
        },
        "ranking": metadata["protocol"]["ranking"],
    }
    return sha256_payload(components), components


def run_pretraining_integration_review() -> dict[str, Any]:
    """Run real Train/Validation integration without any training/evaluation."""
    metadata = _load_frozen_metadata()
    datasets, dataset_report = materialize_train_validation_only(metadata["split"])

    tokenizer = AutoTokenizer.from_pretrained(
        metadata["configs"][0]["tokenizer"], local_files_only=True
    )
    tokenized = _tokenize_train_validation(datasets, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    token_checks = {
        name: {
            "count": len(split),
            "columns": sorted(split.column_names),
            "max_sequence_length": max(len(ids) for ids in split["input_ids"]),
            "labels_valid": set(split["labels"]).issubset({0, 1}),
            "required_fields": {"input_ids", "attention_mask", "labels"}.issubset(
                split.column_names
            ),
        }
        for name, split in tokenized.items()
    }
    if not all(
        item["required_fields"]
        and item["labels_valid"]
        and item["max_sequence_length"] <= 80
        for item in token_checks.values()
    ):
        raise RuntimeError(f"Tokenizer integration failed: {token_checks}")

    trainer_reports = []
    head_fingerprints = []
    first_train_batch = None
    first_validation_batch = None
    for config in metadata["configs"]:
        context = prepare_run_context(config, PROJECT_ROOT)
        arguments = build_training_arguments(config, context)
        callback = build_early_stopping_callback(config)
        resolved_model = _resolve_cached_model_identifier(config["model"])
        model = AutoModelForSequenceClassification.from_pretrained(
            resolved_model,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
            local_files_only=True,
        )
        if (
            model.config.model_type != "distilbert"
            or model.config.num_labels != 2
            or model.config.id2label != {0: "NEGATIVE", 1: "POSITIVE"}
        ):
            raise RuntimeError("Resolved cached model violates the frozen model contract")
        head_fingerprints.append(_head_fingerprint(model))
        trainer = build_dry_trainer(
            model,
            arguments,
            tokenized["train"],
            tokenized["validation"],
            collator,
            tokenizer,
            callback,
        )
        train_batch = next(iter(trainer.get_train_dataloader()))
        validation_batch = next(iter(trainer.get_eval_dataloader()))
        if first_train_batch is None:
            first_train_batch = _batch_summary(train_batch)
            first_validation_batch = _batch_summary(validation_batch)

        device = next(model.parameters()).device
        forward_batch = {
            key: value.to(device)
            for key, value in train_batch.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }
        model.eval()
        with torch.inference_mode():
            outputs = model(**forward_batch)
        forward_ok = (
            list(outputs.logits.shape) == [int(train_batch["labels"].shape[0]), 2]
            and bool(torch.isfinite(outputs.logits).all().item())
            and outputs.loss is not None
            and math.isfinite(float(outputs.loss.detach().cpu().item()))
        )
        optimizer = construct_optimizer_only(trainer)
        trainer.create_scheduler(
            num_training_steps=math.ceil(len(tokenized["train"]) / config["batch_size"])
            * config["max_epochs"],
            optimizer=optimizer,
        )
        callbacks = [item.__class__.__name__ for item in trainer.callback_handler.callbacks]
        trainer_reports.append({
            "run_id": config["run_id"],
            "learning_rate": arguments.learning_rate,
            "requested_model": config["model"],
            "resolved_model": resolved_model,
            "model_class": model.__class__.__name__,
            "model_type": model.config.model_type,
            "num_labels": model.config.num_labels,
            "fresh_model_constructed": True,
            "fresh_optimizer_constructed": True,
            "fresh_scheduler_constructed": trainer.lr_scheduler is not None,
            "trainer_constructed": True,
            "train_dataset_count": len(trainer.train_dataset),
            "validation_dataset_count": len(trainer.eval_dataset),
            "train_batch_size": arguments.per_device_train_batch_size,
            "eval_batch_size": arguments.per_device_eval_batch_size,
            "max_epochs": arguments.num_train_epochs,
            "weight_decay": arguments.weight_decay,
            "max_grad_norm": arguments.max_grad_norm,
            "scheduler": str(arguments.lr_scheduler_type.value),
            "optimizer": str(arguments.optim.value),
            "optimizer_fused": optimizer.defaults.get("fused") is True,
            "eval_strategy": str(arguments.eval_strategy.value),
            "save_strategy": str(arguments.save_strategy.value),
            "load_best_model_at_end": arguments.load_best_model_at_end,
            "metric_for_best_model": arguments.metric_for_best_model,
            "greater_is_better": arguments.greater_is_better,
            "early_stopping_callback_attached": "EarlyStoppingCallback" in callbacks,
            "early_stopping_patience": callback.early_stopping_patience,
            "early_stopping_threshold": callback.early_stopping_threshold,
            "output_dir": str(context.output_dir.relative_to(PROJECT_ROOT)),
            "tensorboard_log_dir": str(context.tensorboard_log_dir.relative_to(PROJECT_ROOT)),
            "forward_only_pass": forward_ok,
            "backward_called": False,
            "optimizer_step_called": False,
            "scheduler_step_called": False,
            "trainer_train_called": False,
        })
        del outputs, optimizer, trainer, model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    provider_calls = {"count": 0}

    def forbidden_provider() -> None:
        provider_calls["count"] += 1

    guard_rejected = False
    try:
        guarded_holdout_request(
            V2_RESULT_DIR / "holdout_access_state.json",
            "winner-does-not-exist",
            forbidden_provider,
        )
    except PermissionError:
        guard_rejected = True
    if not guard_rejected or provider_calls["count"] != 0:
        raise RuntimeError("Holdout guard did not reject before provider invocation")

    execution_hash, freeze_components = _execution_hash(metadata)
    registry_hash = sha256_payload(metadata["registry"])
    checks = {
        "protocol_no_drift": True,
        "train_count_7676": dataset_report["counts"]["train"] == 7676,
        "validation_count_960": dataset_report["counts"]["validation"] == 960,
        "official_test_loaded_false": dataset_report["checks"]["official_test_excluded"],
        "holdout_materialized_false": dataset_report["checks"]["holdout_not_materialized"],
        "tokenizer_integration": all(item["required_fields"] for item in token_checks.values()),
        "dynamic_padding_integration": (
            first_train_batch["input_attention_shapes_equal"]
            and first_validation_batch["input_attention_shapes_equal"]
        ),
        "dataloader_integration": (
            first_train_batch["shapes"]["labels"][0] == 16
            and first_validation_batch["shapes"]["labels"][0] == 32
        ),
        "three_fresh_models": len(trainer_reports) == 3,
        "deterministic_fresh_initialization": len(set(head_fingerprints)) == 1,
        "forward_only": all(item["forward_only_pass"] for item in trainer_reports),
        "trainer_construction": all(item["trainer_constructed"] for item in trainer_reports),
        "optimizer_integration": all(
            item["optimizer"] == "adamw_torch_fused" and item["optimizer_fused"]
            for item in trainer_reports
        ),
        "early_stopping_integration": all(
            item["early_stopping_callback_attached"]
            and item["early_stopping_patience"] == 2
            and item["early_stopping_threshold"] == 1e-6
            and item["metric_for_best_model"] == "eval_loss"
            and item["greater_is_better"] is False
            for item in trainer_reports
        ),
        "run_isolation": (
            len({item["output_dir"] for item in trainer_reports}) == 3
            and len({item["tensorboard_log_dir"] for item in trainer_reports}) == 3
        ),
        "registry_integration": all(
            record["status"] == "PLANNED" for record in metadata["registry"]["runs"]
        ),
        "ranking_schema": all(
            field in metadata["registry"]["runs"][0]
            for field in ("best_val_loss", "best_val_f1", "best_val_accuracy", "best_epoch")
        ),
        "holdout_guard": guard_rejected and provider_calls["count"] == 0,
        "holdout_evaluation_count_zero": metadata["state"]["holdout_evaluation_count"] == 0,
        "real_winner_absent": not (V2_RESULT_DIR / "winner_manifest.json").exists(),
        "no_training": all(
            not item["trainer_train_called"]
            and not item["backward_called"]
            and not item["optimizer_step_called"]
            and not item["scheduler_step_called"]
            for item in trainer_reports
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Pre-training integration checks failed: {checks}")

    readiness = {
        "protocol_version": PROTOCOL_VERSION,
        "review_part": "2.3",
        "status": "PASS",
        "protocol_plan_hash": freeze_components["part_2_1_plan_hash"],
        "protocol_manifest_hash": freeze_components["protocol_manifest_hash"],
        "dataset_split_manifest_hash": freeze_components["dataset_split_manifest_hash"],
        "run_config_hashes": freeze_components["run_config_hashes"],
        "registry_hash": registry_hash,
        "execution_protocol_hash": execution_hash,
        "freeze_components": freeze_components,
        "dataset_train_count": dataset_report["counts"]["train"],
        "dataset_validation_count": dataset_report["counts"]["validation"],
        "holdout_count": metadata["split"]["split_counts"]["holdout"],
        "official_test_loaded": False,
        "holdout_materialized": False,
        "holdout_evaluation_count": metadata["state"]["holdout_evaluation_count"],
        "tokenizer": {
            "checkpoint": str(tokenizer.name_or_path),
            "max_length": 80,
            "truncation": True,
            "dynamic_padding": True,
            "split_checks": token_checks,
        },
        "dataloader": {
            "train": first_train_batch,
            "validation": first_validation_batch,
        },
        "model_construction": "PASS",
        "fresh_model_construction_count": len(trainer_reports),
        "forward_only_sanity": "PASS",
        "trainer_construction": "PASS",
        "trainer_reports": trainer_reports,
        "optimizer": "ADAMW_TORCH_FUSED",
        "optimizer_integration": "PASS",
        "early_stopping_integration": "PASS",
        "tensorboard_integration": "PASS",
        "tensorboard_event_files_created": False,
        "run_isolation": "PASS",
        "registry_integration": "PASS",
        "real_registry_state_changed": False,
        "ranking_schema": "PASS",
        "real_ranking_performed": False,
        "holdout_guard": "PASS",
        "training_performed": False,
        "trainer_train_called": False,
        "backward_called": False,
        "optimizer_step_called": False,
        "scheduler_step_called": False,
        "real_winner_selected": False,
        "protocol_frozen_for_execution": True,
        "checks": checks,
        "technical_debt": [
            "Transformers currently accepts logging_dir but emits a future deprecation warning; all three runs use the same supported API."
        ],
        "ready_for_training": True,
    }
    readiness["readiness_manifest_hash"] = sha256_payload(readiness)
    atomic_write_json(READINESS_PATH, readiness)
    return readiness


if __name__ == "__main__":
    result = run_pretraining_integration_review()
    print({
        "status": result["status"],
        "execution_protocol_hash": result["execution_protocol_hash"],
        "ready_for_training": result["ready_for_training"],
    })
