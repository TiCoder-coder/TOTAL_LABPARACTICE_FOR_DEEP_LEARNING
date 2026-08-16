"""Single-run CLI for the tokenizer-corrected Practice 3 v2.3 protocol."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import json
import hashlib

def sha256_payload(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

import numpy as np
import torch
from filelock import FileLock
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    set_seed,
)

from .config import PROJECT_ROOT, REPO_ROOT
from .dataset_protocol_v2 import materialize_train_validation_only, validate_split_manifest
from .experiment_protocol_v2 import atomic_write_json
from .experiment_runner_v2 import collect_training_history, select_best_epoch, checkpoint_fingerprint
from .phase_08_metrics_training_configuration import compute_metrics
from .tokenizer_correction_v2_1 import (
    MODEL_IDENTIFIER,
    TOKENIZER_IDENTIFIER,
    load_json,
    tokenize_splits,
    validate_tokenizer_integrity,
)

PROTOCOL_VERSION = "practice_3_v2.3"
RUN_IDS = ["E1_lr_1e-5", "E2_lr_2e-5", "E3_lr_3e-5"]
STAGE_B_RUN_IDS = [
    "E4_weight_decay_0.05",
    "E5_classifier_dropout_0.20",    # preserved as invalid-experiment evidence
    "E5b_classifier_dropout_0.40",   # corrected dropout experiment
    "E6_staged_finetune",            # FAILED — MPS SDPA dropout not supported
    "E6b_staged_finetune",           # FAILED — optimizer parameter group count mismatch with scheduler
    "E6c_staged_finetune",           # corrected staged optimizer/scheduler logic
]
V22_RESULT_DIR = PROJECT_ROOT / "docs" / "result" / "practice_3_v2_3"
V22_RUNS_DIR = PROJECT_ROOT / "runs" / "practice_3_v2_3"

RANKING = {
    "primary": "validation_loss",
    "secondary": "validation_f1",
    "tertiary": "validation_accuracy",
    "fourth": "earlier_best_epoch",
    "final_deterministic_fallback": "lexical_experiment_id",
    "tolerance": 1e-6,
}



from transformers import TrainingArguments
import os

class StagedFinetuningCallback(TrainerCallback):
    def __init__(self, model, optimizer, learning_rate: float, weight_decay: float):
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.current_stage = 1
        
        # Apply initial Stage 1 freeze here (AFTER optimizer is created)
        for name, param in self.model.named_parameters():
            if name.startswith("distilbert.embeddings") or any(name.startswith(f"distilbert.transformer.layer.{i}.") for i in range(4)):
                param.requires_grad = False
            else:
                param.requires_grad = True
        
    def _count_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        return trainable, frozen

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Transition to Stage 2 at exactly epoch 2.0
        if state.epoch == 2.0 and self.current_stage == 1:
            self.current_stage = 2
            
            # Unfreeze all model parameters (NO add_param_group needed since all are already in optimizer)
            for param in self.model.parameters():
                param.requires_grad = True
                
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            t, f = self._count_params()
            logs["fine_tuning_stage"] = self.current_stage
            logs["trainable_params"] = t
            logs["frozen_params"] = f
            logs["trainable_layers"] = "all" if self.current_stage == 2 else "layers_4_5_and_classifier"

def training_arguments(config: dict[str, Any]) -> TrainingArguments:
    exp_id = config["experiment_id"]
    output_dir = V22_RUNS_DIR / exp_id / "checkpoints"
    logging_dir = REPO_ROOT / config["tensorboard_log_dir"]
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get("max_epochs", 15),
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("eval_batch_size", 32),
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("gradient_clipping", 1.0),
        warmup_steps=config.get("warmup_steps", 720),
        label_smoothing_factor=config.get("label_smoothing_factor", 0.1),
        # Hardcoded fixed parameters (ignore JSON values as they have NO_EFFECT)
        optim="adamw_torch",
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
    )

def validate_configs(configs: list[dict[str, Any]], registry: dict[str, Any]) -> None:
    for cfg in configs:
        if cfg["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError(f"Config protocol mismatch: {cfg['protocol_version']}")
        # Recompute hash
        supplied_hash = cfg.get("config_hash")
        body = {k: v for k, v in cfg.items() if k != "config_hash"}
        if supplied_hash != sha256_payload(body):
            raise ValueError(f"Config hash mismatch for {cfg['experiment_id']}")
        if cfg.get("label_smoothing_factor") != 0.1:
            raise ValueError("label_smoothing_factor must be 0.1")
        if cfg.get("max_epochs") != 15:
            raise ValueError("max_epochs must be 15")
        if cfg.get("warmup_steps") != 720:
            raise ValueError("warmup_steps must be 720")
        if cfg.get("model") != "distilbert/distilbert-base-uncased" or cfg.get("tokenizer") != "distilbert-base-uncased":
            raise ValueError("Model/tokenizer identifiers are strictly DistilBERT")
        # match registry hash
        registry_experiments = {e["experiment_id"]: e for e in registry["experiments"]}
        if registry_experiments[cfg["experiment_id"]]["config_hash"] != supplied_hash:
            raise ValueError("Registry config_hash mismatch")

    # E1/E2/E3 differ only by LR
    stage_a = [c for c in configs if c["stage"] == "A"]
    if len(stage_a) == 3:
        # Check LR
        lrs = {c["learning_rate"] for c in stage_a}
        if lrs != {1e-5, 2e-5, 3e-5}:
            raise ValueError("LRs must be 1e-5, 2e-5, 3e-5")
        base_body = {k: v for k, v in stage_a[0].items() if k not in {"learning_rate", "config_hash", "experiment_id", "tensorboard_log_dir"}}
        for c in stage_a[1:]:
            c_body = {k: v for k, v in c.items() if k not in {"learning_rate", "config_hash", "experiment_id", "tensorboard_log_dir"}}
            if c_body != base_body:
                raise ValueError("Stage A configs differ by more than learning_rate")


def validate_registry(registry: dict[str, Any], configs: list[dict[str, Any]]) -> None:
    if registry["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"Registry protocol mismatch: {registry['protocol_version']}")
    # 6 original runs + E5b correction = 7 expected; permit 6 or 7 during transition
    if len(registry["experiments"]) < 6:
        raise ValueError("Registry must contain at least 6 runs")


REGISTRY_PATH = V22_RESULT_DIR / "experiment_registry.json"


def load_and_verify_metadata(run_id: str) -> dict[str, Any]:
    registry = load_json(REGISTRY_PATH)
    all_known_ids = RUN_IDS + STAGE_B_RUN_IDS
    if run_id not in all_known_ids:
        raise ValueError(f"Unknown v2.3 run ID: {run_id}")
    
    # Stage B specific checks
    if run_id in STAGE_B_RUN_IDS:
        if not registry.get("winner_selected") or registry.get("stage_a_status") != "COMPLETED":
            raise PermissionError(f"Stage B run {run_id} is blocked pending Stage A completion and winner selection.")
            
    authorization = load_json(V22_RESULT_DIR / "training_authorization.json")
    auth_hash = authorization.get("authorization_hash")
    if auth_hash != sha256_payload({k: v for k, v in authorization.items() if k != "authorization_hash"}):
        raise PermissionError("v2.3 authorization self-hash mismatch")
    if authorization.get("training_authorized") is not True:
        raise PermissionError("v2.3 training is not authorized")
        
    auth_order = authorization.get("execution_order", {}).get(run_id)
    if auth_order not in ["FIRST", "AUTHORIZED"]:
        records = {item["experiment_id"]: item for item in registry["experiments"]}
        if run_id in RUN_IDS:
            index = RUN_IDS.index(run_id)
            if any(records[item]["status"] != "COMPLETED" for item in RUN_IDS[:index]):
                raise RuntimeError("v2.3 execution order violation")
        elif run_id in STAGE_B_RUN_IDS:
            index = STAGE_B_RUN_IDS.index(run_id)
            if any(records[item]["status"] != "COMPLETED" for item in STAGE_B_RUN_IDS[:index]):
                raise RuntimeError("v2.3 Stage B execution order violation")
                
    protocol = load_json(V22_RESULT_DIR / "protocol_manifest.json")
    supplied_protocol_hash = protocol.get("protocol_manifest_hash")
    if supplied_protocol_hash != sha256_payload({k: v for k, v in protocol.items() if k != "protocol_manifest_hash"}):
        raise RuntimeError("v2.3 protocol hash mismatch")
        
    V21_RESULT_DIR = PROJECT_ROOT / "docs" / "result" / "practice_3_v2_1"
    split = load_json(V21_RESULT_DIR / "dataset_split_manifest.json")
    validate_split_manifest(split)
    
    tokenizer_report = load_json(V22_RESULT_DIR / "tokenizer_validation.json")
    supplied_tokenizer_hash = tokenizer_report.get("artifact_hash")
    if supplied_tokenizer_hash != sha256_payload({k: v for k, v in tokenizer_report.items() if k != "artifact_hash"}):
        raise RuntimeError("Tokenizer evidence hash mismatch")
    if tokenizer_report.get("tokenizer_integrity") != "PASS" or tokenizer_report.get("vocab_size") != 30522:
        raise PermissionError("Tokenizer evidence does not authorize training")
        
    all_configs = [load_json(V22_RESULT_DIR / "experiments" / item / "experiment_config.json") for item in all_known_ids]
    validate_configs(all_configs, registry)
    validate_registry(registry, all_configs)
    
    holdout = load_json(V22_RESULT_DIR / "holdout_access_state.json")
    if not (holdout.get("holdout_access_allowed") is False
            and holdout.get("holdout_evaluation_count") == 0
            and holdout.get("holdout_evaluated") is False):
        raise PermissionError("v2.3 Holdout is not SEALED")
        
    components = {
        "protocol_manifest_hash": protocol["protocol_manifest_hash"],
        "dataset_split_manifest_hash": split["split_manifest_hash"],
        "tokenizer_validation_hash": tokenizer_report["artifact_hash"],
        "experiment_config_hashes": {item["experiment_id"]: item["config_hash"] for item in all_configs},
        "ranking": RANKING,
    }
    if registry.get("winner_selected"):
        selection_report = load_json(V22_RESULT_DIR / "stage_a_selection_report.json")
        components["stage_a_selection_hash"] = selection_report.get("selection_hash")
        
    execution_hash = sha256_payload(components)
    if execution_hash != authorization.get("execution_protocol_hash"):
        raise RuntimeError("v2.3 execution protocol hash mismatch")
        
    records = {item["experiment_id"]: item for item in registry["experiments"]}
    if records[run_id]["status"] != "PLANNED":
        raise RuntimeError(f"Run must be PLANNED, got {records[run_id]['status']}")
        
    run_dir = V22_RESULT_DIR / "experiments" / run_id
    conflicts = [path for path in run_dir.rglob("*") if path.is_file() and path.name != "experiment_config.json"]
    run_output_dir = V22_RUNS_DIR / run_id
    stale_outputs = [path for path in run_output_dir.rglob("*") if path.is_file()]
    if conflicts or stale_outputs:
        raise RuntimeError("Conflicting v2.3 output artifacts block training")
        
    target_config = next(c for c in all_configs if c["experiment_id"] == run_id)
        
    return {"authorization": authorization, "protocol": protocol, "split": split,
            "configs": all_configs, "registry": registry, "config": target_config,
            "execution_protocol_hash": execution_hash}


def persist_transition(configs: list[dict[str, Any]], run_id: str, expected: str,
                       new_status: str, fields: dict[str, Any] | None = None) -> dict[str, Any]:
    with FileLock(str(REGISTRY_PATH) + ".lock"):
        registry = load_json(REGISTRY_PATH)
        all_configs = [load_json(V22_RESULT_DIR / "experiments" / item / "experiment_config.json") for item in RUN_IDS + STAGE_B_RUN_IDS]
        validate_registry(registry, all_configs)
        records = [item for item in registry["experiments"] if item["experiment_id"] == run_id]
        if len(records) != 1 or records[0]["status"] != expected:
            raise RuntimeError("Concurrent v2.2 registry state change")
        allowed = {"PLANNED": {"RUNNING"}, "RUNNING": {"COMPLETED", "FAILED"}}
        if new_status not in allowed.get(expected, set()):
            raise ValueError("Invalid v2.2 lifecycle transition")
        records[0]["status"] = new_status
        if fields:
            if any("test" in key.lower() or "holdout" in key.lower() for key in fields):
                raise ValueError("Evaluation-only registry fields are prohibited")
            records[0].update(fields)
        registry["real_experiments_executed"] = True
        registry["status"] = "COMPLETED" if all(item["status"] == "COMPLETED" for item in registry["experiments"]) else "IN_PROGRESS"
        atomic_write_json(REGISTRY_PATH, registry, overwrite=True)
        return registry


def run_experiment(run_id: str) -> dict[str, Any]:
    metadata = load_and_verify_metadata(run_id)
    config, configs = metadata["config"], metadata["configs"]
    datasets, report = materialize_train_validation_only(metadata["split"])
    if report["counts"] != {"train": 7676, "validation": 960}:
        raise RuntimeError("v2.2 Train/Validation count mismatch")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_IDENTIFIER, local_files_only=True)
    validate_tokenizer_integrity(tokenizer, list(datasets["train"].select(range(128))["text"]))
    tokenized = tokenize_splits(datasets, tokenizer)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); set_seed(42)
    # Apply seq_classif_dropout from config if explicitly specified (e.g., E5b).
    # For baseline runs that do not set this key it defaults to the pretrained value (0.20).
    model_kwargs: dict[str, Any] = {
        "num_labels": 2,
        "id2label": {0: "NEGATIVE", 1: "POSITIVE"},
        "label2id": {"NEGATIVE": 0, "POSITIVE": 1},
        "local_files_only": True,
    }
    if "seq_classif_dropout" in config:
        model_kwargs["seq_classif_dropout"] = float(config["seq_classif_dropout"])
    # Pass attn_implementation from config (e.g. "eager" for MPS-safe staged fine-tuning).
    # This is an execution-backend field, not an experimental variable.
    if "attn_implementation" in config:
        model_kwargs["attn_implementation"] = config["attn_implementation"]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_IDENTIFIER, **model_kwargs)
    # Runtime sanity check: actual model dropout must match config
    actual_dropout = model.config.seq_classif_dropout
    expected_dropout = float(config.get("seq_classif_dropout", actual_dropout))
    if abs(actual_dropout - expected_dropout) > 1e-9:
        raise RuntimeError(
            f"seq_classif_dropout mismatch: config={expected_dropout}, "
            f"model.config={actual_dropout}. Runner did not apply the config correctly."
        )

    if config.get("fine_tuning_strategy") == "staged":
        # Ensure all parameters start with requires_grad=True so Trainer puts them all in the optimizer
        for param in model.parameters():
            param.requires_grad = True

    arguments = training_arguments(config)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=1e-6)]
    trainer = Trainer(model=model, args=arguments, train_dataset=tokenized["train"],
                      eval_dataset=tokenized["validation"],
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      processing_class=tokenizer, compute_metrics=compute_metrics,
                      callbacks=callbacks)
    trainer.create_optimizer()
    trainer.create_scheduler(math.ceil(7676 / 16) * 10, optimizer=trainer.optimizer)

    if config.get("fine_tuning_strategy") == "staged":
        staged_cb = StagedFinetuningCallback(
            model=model, 
            optimizer=trainer.optimizer, 
            learning_rate=config["learning_rate"], 
            weight_decay=config.get("weight_decay", 0.01)
        )
        trainer.add_callback(staged_cb)

    # Final read-only barrier immediately before the user-triggered real run starts.
    load_and_verify_metadata(run_id)
    persist_transition(configs, run_id, "PLANNED", "RUNNING")
    started = time.monotonic()
    try:
        train_result = trainer.train()
        runtime = float(train_result.metrics.get("train_runtime", time.monotonic() - started))
        history = collect_training_history(trainer.state.log_history)
        best = select_best_epoch(history)
        checkpoint = Path(arguments.output_dir) / f"checkpoint-{int(best['global_step'])}"
        fingerprint = checkpoint_fingerprint(checkpoint)
        run_dir = V22_RESULT_DIR / "experiments" / run_id
        history_payload = {"protocol_version": PROTOCOL_VERSION, "run_id": run_id,
                           "records": history, "holdout_accessed": False,
                           "official_test_loaded": False}
        atomic_write_json(run_dir / "training_history.json", history_payload)
        summary = {
            "protocol_version": PROTOCOL_VERSION, "run_id": run_id,
            "learning_rate": config["learning_rate"], "status": "COMPLETED",
            "stopped_epoch": history[-1]["epoch"], "best_epoch": best["epoch"],
            "best_val_loss": best["eval_loss"], "best_val_accuracy": best["eval_accuracy"],
            "best_val_precision": best["eval_precision"], "best_val_recall": best["eval_recall"],
            "best_val_f1": best["eval_f1"],
            "best_checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)), **fingerprint,
            "runtime_seconds": runtime, "config_hash": config["config_hash"],
            "split_fingerprint": config["split_fingerprint"],
            "execution_protocol_hash": metadata["execution_protocol_hash"],
            "tensorboard_log_dir": config["tensorboard_log_dir"],
            "holdout_accessed": False, "official_test_loaded": False,
        }
        atomic_write_json(run_dir / "run_summary.json", summary)
        fields = {key: summary[key] for key in ("stopped_epoch", "best_epoch", "best_val_loss",
                  "best_val_accuracy", "best_val_precision", "best_val_recall", "best_val_f1",
                  "best_checkpoint", "runtime_seconds")}
        fields["checkpoint_hash"] = summary["checkpoint_hash"]
        persist_transition(configs, run_id, "RUNNING", "COMPLETED", fields)
        return summary
    except BaseException as error:
        try:
            persist_transition(configs, run_id, "RUNNING", "FAILED", {
                "error_type": type(error).__name__, "error_message": str(error)[:2000],
                "runtime_seconds": time.monotonic() - started})
        except Exception as transition_error:
            error.add_note(f"FAILED registry transition also failed: {transition_error}")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one authorized Practice 3 v2.3 experiment")
    parser.add_argument("--run-id", required=True, choices=RUN_IDS + STAGE_B_RUN_IDS)
    return parser



def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_experiment(args.run_id), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
