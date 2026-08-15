"""Phase 9: guarded debug run and full DistilBERT fine-tuning."""

import gc
import json
import math
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from transformers import Trainer

from .config import RESULT_DIR
from .phase_01_environment import get_device, set_seed
from .phase_07_model_construction import build_model
from .phase_08_metrics_training_configuration import (
    compute_metrics,
    create_training_arguments,
    select_best_checkpoint,
)


PHASE_09_DIR = RESULT_DIR / "phase_09_training"
FULL_CHECKPOINT_DIR = PHASE_09_DIR / "checkpoints"
DEBUG_CHECKPOINT_DIR = PHASE_09_DIR / "debug_checkpoints"
MANIFEST_PATH = PHASE_09_DIR / "phase_09_training_manifest.json"
REQUIRED_ARTIFACTS = (
    "debug_verification.json",
    "training_configuration.json",
    "training_history.json",
    "validation_metrics_by_epoch.json",
    "checkpoint_records.json",
    "selected_checkpoint.json",
    "trainer_state.json",
    "phase_09_training_manifest.json",
)


def _save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    readback = json.loads(path.read_text(encoding="utf-8"))
    if readback != json.loads(json.dumps(payload, default=str)):
        raise RuntimeError(f"JSON read-back failed: {path}")
    return path


def _validate_split(dataset: Any, expected_count: int, split_name: str) -> None:
    if len(dataset) != expected_count:
        raise ValueError(f"{split_name} count {len(dataset)} != {expected_count}")
    columns = set(dataset.column_names)
    required = {"input_ids", "attention_mask", "labels"}
    if not required.issubset(columns):
        raise ValueError(f"{split_name} missing fields: {required - columns}")
    if not set(dataset["labels"]).issubset({0, 1}):
        raise ValueError(f"{split_name} labels are not a subset of {{0,1}}")


def _trainer(
    model: Any,
    args: Any,
    train_dataset: Any,
    validation_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
) -> Trainer:
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )


def _finite_validation_record(record: Dict[str, Any]) -> bool:
    keys = ("eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1")
    return all(key in record and math.isfinite(float(record[key])) for key in keys)


def run_debug(
    model: Any,
    train_dataset: Any,
    validation_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
) -> Dict[str, Any]:
    """Run the mandatory deterministic 1000/200 one-epoch pipeline check."""
    debug_train = train_dataset.select(range(1000))
    debug_validation = validation_dataset.select(range(200))
    _validate_split(debug_train, 1000, "debug_train")
    _validate_split(debug_validation, 200, "debug_validation")
    args = create_training_arguments(DEBUG_CHECKPOINT_DIR)
    args.num_train_epochs = 1.0
    trainer = _trainer(model, args, debug_train, debug_validation, data_collator, tokenizer)
    result = trainer.train()
    validation_records = [r for r in trainer.state.log_history if "eval_loss" in r]
    checkpoints = sorted(DEBUG_CHECKPOINT_DIR.glob("checkpoint-*"))
    training_loss = float(result.training_loss)
    all_pass = (
        trainer.state.global_step > 0
        and math.isfinite(training_loss)
        and len(validation_records) == 1
        and _finite_validation_record(validation_records[0])
        and len(checkpoints) >= 1
    )
    payload = {
        "status": "PASS" if all_pass else "FAIL",
        "run_role": "debug_only_not_final",
        "train_samples": len(debug_train),
        "validation_samples": len(debug_validation),
        "epochs": 1.0,
        "global_step": trainer.state.global_step,
        "training_loss": training_loss,
        "validation_record": validation_records[0] if validation_records else None,
        "checkpoint_paths": [str(path.resolve()) for path in checkpoints],
        "optimizer_updates_occurred": trainer.state.global_step > 0,
        "test_accessed": False,
        "final_experiment": False,
    }
    _save_json(PHASE_09_DIR / "debug_verification.json", payload)
    if not all_pass:
        raise RuntimeError(f"Phase 9 debug verification failed: {payload}")
    del trainer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return payload


def _checkpoint_records(log_history: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    records = []
    for record in log_history:
        if "eval_loss" not in record:
            continue
        step = int(record["step"])
        checkpoint = FULL_CHECKPOINT_DIR / f"checkpoint-{step}"
        candidate = {
            "checkpoint": str(checkpoint.resolve()),
            "epoch": float(record["epoch"]),
            "step": step,
            "eval_loss": float(record["eval_loss"]),
            "eval_f1": float(record["eval_f1"]),
        }
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Missing epoch checkpoint: {checkpoint}")
        if not any((checkpoint / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
            raise FileNotFoundError(f"Missing model weights in {checkpoint}")
        records.append(candidate)
    return records


def _effective_configuration(args: Any) -> Dict[str, Any]:
    value = lambda item: str(getattr(item, "value", item))
    return {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "eval_strategy": value(args.eval_strategy),
        "logging_strategy": value(args.logging_strategy),
        "save_strategy": value(args.save_strategy),
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": args.greater_is_better,
        "report_to": args.report_to,
    }


def verify_selected_checkpoint(
    model: Any, validation_dataset: Any, data_collator: Any
) -> Dict[str, Any]:
    """Verify the loaded policy checkpoint with a two-sample Validation batch."""
    samples = [validation_dataset[index] for index in range(2)]
    batch = data_collator(samples)
    device = torch.device(get_device())
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**{key: value.to(device) for key, value in batch.items()})
    logits_shape = list(outputs.logits.shape)
    logits_finite = bool(torch.isfinite(outputs.logits).all().item())
    loss_finite = bool(outputs.loss is not None and torch.isfinite(outputs.loss).item())
    all_pass = logits_shape == [2, 2] and logits_finite and loss_finite
    return {
        "status": "PASS" if all_pass else "FAIL",
        "source": "validation",
        "samples": 2,
        "logits_shape": logits_shape,
        "logits_finite": logits_finite,
        "loss_finite": loss_finite,
        "test_accessed": False,
    }


def run_full(
    train_dataset: Any,
    validation_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
    debug_verification: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Run one fresh three-epoch full baseline after debug PASS."""
    if debug_verification.get("status") != "PASS":
        raise RuntimeError("Full training is forbidden before debug PASS")
    _validate_split(train_dataset, 8530, "train")
    _validate_split(validation_dataset, 1066, "validation")
    set_seed(42)
    model = build_model()
    args = create_training_arguments(FULL_CHECKPOINT_DIR)
    configuration = {
        **_effective_configuration(args),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "model_fresh_after_debug": True,
        "test_accessed": False,
    }
    _save_json(PHASE_09_DIR / "training_configuration.json", configuration)

    started_at = datetime.now().isoformat()
    trainer = _trainer(model, args, train_dataset, validation_dataset, data_collator, tokenizer)
    train_result = trainer.train()
    ended_at = datetime.now().isoformat()
    history = [dict(record) for record in trainer.state.log_history]
    validation_history = [record for record in history if "eval_loss" in record]
    if len(validation_history) != 3 or not all(map(_finite_validation_record, validation_history)):
        raise RuntimeError("Expected three finite epoch Validation records")
    records = _checkpoint_records(history)
    if len(records) != 3:
        raise RuntimeError(f"Expected three checkpoint records, received {len(records)}")
    selected = select_best_checkpoint(records)
    trainer_choice = str(Path(trainer.state.best_model_checkpoint).resolve())
    policy_choice = selected["checkpoint"]
    choices_differ = trainer_choice != policy_choice

    # Authoritative downstream model always comes from the explicit policy path.
    selected_model = build_model(checkpoint=policy_choice)
    selected_model.eval()
    selected_sanity = verify_selected_checkpoint(
        selected_model, validation_dataset, data_collator
    )
    if selected_sanity["status"] != "PASS":
        raise RuntimeError(f"Selected checkpoint sanity failed: {selected_sanity}")
    selected_record = {
        **selected,
        "primary_criterion": "minimum eval_loss",
        "tie_breaker": "maximum eval_f1",
        "remaining_tie": "earlier epoch/checkpoint",
        "trainer_best_model_checkpoint": trainer_choice,
        "trainer_choice_differs_from_policy": choices_differ,
        "policy_checkpoint_explicitly_loaded": True,
        "selected_forward_sanity": selected_sanity,
        "test_used_for_selection": False,
    }

    trainer.state.save_to_json(str(PHASE_09_DIR / "trainer_state.json"))
    _save_json(PHASE_09_DIR / "training_history.json", history)
    _save_json(PHASE_09_DIR / "validation_metrics_by_epoch.json", validation_history)
    _save_json(PHASE_09_DIR / "checkpoint_records.json", records)
    _save_json(PHASE_09_DIR / "selected_checkpoint.json", selected_record)

    artifact_paths = {name: str((PHASE_09_DIR / name).resolve()) for name in REQUIRED_ARTIFACTS}
    completed_epochs = float(trainer.state.epoch or 0.0)
    manifest = {
        "status": "PASS",
        "run_mode": "fresh_full_baseline",
        "started_at": started_at,
        "ended_at": ended_at,
        "python_version": platform.python_version(),
        "transformers_version": version("transformers"),
        "torch_version": torch.__version__,
        "device": get_device(),
        "seed": 42,
        "debug_passed_before_full": True,
        "fresh_model_after_debug": True,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "completed_epochs": completed_epochs,
        "global_step": trainer.state.global_step,
        "training_loss": float(train_result.training_loss),
        "validation_record_count": len(validation_history),
        "checkpoint_record_count": len(records),
        "selected_checkpoint": policy_choice,
        "selected_checkpoint_loaded": True,
        "artifact_paths": artifact_paths,
        "trainer_created": True,
        "training_performed": True,
        "test_accessed": False,
        "phase_10_started": False,
    }
    _save_json(MANIFEST_PATH, manifest)
    return selected_model, manifest


def _valid_existing_run() -> Dict[str, Any] | None:
    if not MANIFEST_PATH.exists():
        return None
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("test_accessed") is not False:
        return None
    if not all((PHASE_09_DIR / name).is_file() for name in REQUIRED_ARTIFACTS):
        return None
    selected = json.loads((PHASE_09_DIR / "selected_checkpoint.json").read_text(encoding="utf-8"))
    if not Path(selected["checkpoint"]).is_dir():
        return None
    if selected.get("selected_forward_sanity", {}).get("status") != "PASS":
        return None
    if manifest.get("train_samples") != 8530 or manifest.get("validation_samples") != 1066:
        return None
    if manifest.get("validation_record_count") != 3 or manifest.get("checkpoint_record_count") != 3:
        return None
    return manifest


def run_or_load_phase_09(
    debug_model: Any,
    train_dataset: Any,
    validation_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
    force_train: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Reuse a verified run by default; otherwise execute debug then full once."""
    existing = None if force_train else _valid_existing_run()
    if existing is not None:
        selected = json.loads((PHASE_09_DIR / "selected_checkpoint.json").read_text(encoding="utf-8"))
        loaded_model = build_model(checkpoint=selected["checkpoint"])
        report = {**existing, "guard_action": "loaded_verified_artifacts_no_retraining"}
        return loaded_model, report
    if PHASE_09_DIR.exists() and any(PHASE_09_DIR.iterdir()):
        raise RuntimeError(
            "Phase 9 contains partial/unverified artifacts; refusing to overwrite or resume silently"
        )
    debug = run_debug(debug_model, train_dataset, validation_dataset, data_collator, tokenizer)
    selected_model, manifest = run_full(
        train_dataset, validation_dataset, data_collator, tokenizer, debug
    )
    return selected_model, {**manifest, "guard_action": "executed_debug_and_fresh_full_training"}


def load_phase_09_report() -> Dict[str, Any]:
    """Load presentation-ready Phase 9 artifacts without training."""
    manifest = _valid_existing_run()
    if manifest is None:
        raise RuntimeError("No complete verified Phase 9 run is available")
    return {
        "manifest": manifest,
        "debug": json.loads((PHASE_09_DIR / "debug_verification.json").read_text()),
        "configuration": json.loads((PHASE_09_DIR / "training_configuration.json").read_text()),
        "validation_metrics": json.loads((PHASE_09_DIR / "validation_metrics_by_epoch.json").read_text()),
        "checkpoint_records": json.loads((PHASE_09_DIR / "checkpoint_records.json").read_text()),
        "selected_checkpoint": json.loads((PHASE_09_DIR / "selected_checkpoint.json").read_text()),
    }
