from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torchvision import models

from .paths import get_practice_2_2_root
from .r7_training_correctness import (
    LOSS_CONTRACT,
    CrossEntropyBatchTerms,
    EarlyStoppingState,
    build_named_adamw,
    configure_trainable_parameters,
    cross_entropy_batch_terms,
    load_complete_checkpoint,
    route_corrected_validation_loss,
    save_complete_checkpoint,
    set_frozen_batchnorm_eval,
    state_dict_sha256,
)
from .resources import file_sha256


POLICY_RELATIVE_PATH = Path("configs/r8_staged_transfer_policy.json")
EXPERIMENT_FIELDS = ("warmup_epochs", "dropout", "label_smoothing")
RESULT_FIELDS = {
    "experiment_id",
    "seed",
    "initial_state_sha256",
    "train_accuracy",
    "validation_accuracy",
    "validation_macro_f1",
    "best_epoch",
}


def load_r8_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R8 staged-transfer policy schema")
    if policy.get("predecessor_phase") != "R7":
        raise RuntimeError("R8 predecessor must be R7")
    if policy.get("predecessor_gate_required") is not True:
        raise RuntimeError("R8 must require the R7 gate")
    if policy.get("architecture") != "resnet18":
        raise RuntimeError("R8 architecture must be ResNet18")
    if policy.get("pretrained_weights") != "IMAGENET1K_V1":
        raise RuntimeError("R8 pretrained authority is invalid")
    if policy.get("experiment_seeds") != [42, 123, 2026]:
        raise RuntimeError("R8 repeated-seed contract is invalid")
    minimum, maximum = policy.get("warmup_epoch_range", [None, None])
    warmup_epochs = policy.get("staged_warmup_epochs")
    if not minimum <= warmup_epochs <= maximum:
        raise RuntimeError("R8 warmup duration must be between three and five epochs")
    if policy.get("fine_tune_block") != "layer4":
        raise RuntimeError("R8 may fine-tune layer4 only")
    optimizer = policy.get("optimizer", {})
    if not 3e-4 <= optimizer.get("head_learning_rate", 0.0) <= 1e-3:
        raise RuntimeError("R8 head learning rate is outside the approved range")
    if not 1e-5 <= optimizer.get("backbone_learning_rate", 0.0) <= 3e-5:
        raise RuntimeError("R8 backbone learning rate is outside the approved range")
    if policy.get("test_evidence_allowed") is not False:
        raise RuntimeError("R8 must prohibit Test evidence")
    if policy.get("test_content_access_allowed") is not False:
        raise RuntimeError("R8 must prohibit Test content access")
    if policy.get("test_loader_construction_allowed") is not False:
        raise RuntimeError("R8 must prohibit Test loader construction")
    experiments = policy.get("experiments", {})
    if tuple(experiments) != ("M0", "M1", "M2", "M3"):
        raise RuntimeError("R8 must define M0 through M3 in order")
    for experiment_id in ("M1", "M2", "M3"):
        experiment = experiments[experiment_id]
        parent = experiments[experiment["parent"]]
        changed = [
            field
            for field in EXPERIMENT_FIELDS
            if experiment[field] != parent[field]
        ]
        if changed != [experiment["single_change"]]:
            raise RuntimeError(f"R8 experiment is not controlled: {experiment_id}")
    if experiments["M0"]["warmup_epochs"] != 0:
        raise RuntimeError("R8 M0 must fine-tune layer4 immediately")
    if any(experiments[item]["warmup_epochs"] != warmup_epochs for item in ("M1", "M2", "M3")):
        raise RuntimeError("R8 staged experiments must share the warmup duration")
    return policy


def load_authorized_pretrained_state(
    checkpoint_path: Path,
    expected_file_sha256: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"R8 pretrained checkpoint is missing: {path}")
    live_hash = file_sha256(path)
    if live_hash != expected_file_sha256:
        raise RuntimeError("R8 pretrained checkpoint SHA-256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise RuntimeError("R8 pretrained checkpoint must contain a state dict")
    model = models.resnet18(weights=None)
    model.load_state_dict(state, strict=True)
    return dict(state), {
        "weights": "IMAGENET1K_V1",
        "checkpoint_file_sha256": live_hash,
        "state_dict_sha256": state_dict_sha256(state),
        "offline_loaded": True,
    }


def build_experiment_model(
    pretrained_state: Mapping[str, torch.Tensor],
    dropout: float,
    num_classes: int,
    head_seed: int,
) -> nn.Module:
    model = models.resnet18(weights=None)
    model.load_state_dict(pretrained_state, strict=True)
    features = model.fc.in_features
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(head_seed)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
    return model


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def trainable_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def build_stage_optimizer(
    model: nn.Module,
    policy: Mapping[str, Any],
) -> torch.optim.AdamW:
    optimizer = policy["optimizer"]
    return build_named_adamw(
        model,
        head_learning_rate=float(optimizer["head_learning_rate"]),
        backbone_learning_rate=float(optimizer["backbone_learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
    )


def build_stage_controls(
    optimizer: torch.optim.Optimizer,
    policy: Mapping[str, Any],
) -> tuple[Any, EarlyStoppingState]:
    scheduler_config = policy["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config["mode"],
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
    )
    stopping = policy["early_stopping"]
    early_stopping = EarlyStoppingState(
        patience=int(stopping["patience"]),
        min_delta=float(stopping["min_delta"]),
    )
    return scheduler, early_stopping


def validate_validation_metrics(metrics: Mapping[str, Any]) -> None:
    required = {
        "split",
        "contract",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
    }
    if set(metrics) != required:
        raise ValueError("R8 warmup metrics do not match the Validation contract")
    if metrics["split"] != "Validation" or metrics["contract"] != LOSS_CONTRACT:
        raise ValueError("R8 requires corrected Validation metrics")
    values = [
        float(metrics["validation_loss"]),
        float(metrics["validation_accuracy"]),
        float(metrics["validation_macro_f1"]),
    ]
    if not all(torch.isfinite(torch.tensor(value)) for value in values):
        raise ValueError("R8 Validation metrics must be finite")


class StagedTransferSession:
    def __init__(
        self,
        experiment_id: str,
        pretrained_state: Mapping[str, torch.Tensor],
        checkpoint_path: Path,
        policy: Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = dict(policy or load_r8_policy())
        if experiment_id not in self.policy["experiments"]:
            raise ValueError(f"Unknown R8 experiment: {experiment_id}")
        self.experiment_id = experiment_id
        self.experiment = dict(self.policy["experiments"][experiment_id])
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.model = build_experiment_model(
            pretrained_state,
            dropout=float(self.experiment["dropout"]),
            num_classes=int(self.policy["num_classes"]),
            head_seed=int(self.policy["head_initialization_seed"]),
        )
        self.initial_state_sha256 = state_dict_sha256(self.model.state_dict())
        self.warmup_epochs_completed = 0
        self.best_warmup_epoch = None
        self.best_warmup_macro_f1 = float("-inf")
        self.best_warmup_validation_loss = float("inf")
        self.best_warmup_state_sha256 = None
        self.validation_control_records = []
        if self.experiment["warmup_epochs"] == 0:
            self.stage = "layer4_finetune"
            configure_trainable_parameters(self.model, "partial_finetune")
        else:
            self.stage = "head_warmup"
            configure_trainable_parameters(self.model, "head_only")
        self.optimizer = build_stage_optimizer(self.model, self.policy)
        self.scheduler, self.early_stopping = build_stage_controls(
            self.optimizer, self.policy
        )

    def prepare_training_mode(self) -> dict[str, Any]:
        self.model.train()
        frozen_batchnorm_count = set_frozen_batchnorm_eval(self.model)
        return {
            "stage": self.stage,
            "frozen_batchnorm_count": frozen_batchnorm_count,
            "trainable_parameter_names": trainable_parameter_names(self.model),
        }

    def compute_loss_terms(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> CrossEntropyBatchTerms:
        return cross_entropy_batch_terms(
            logits,
            targets,
            class_weights=None,
            label_smoothing=float(self.experiment["label_smoothing"]),
        )

    def record_warmup_epoch(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        if self.stage != "head_warmup":
            raise RuntimeError("R8 session is not in head warmup")
        if self.warmup_epochs_completed >= self.experiment["warmup_epochs"]:
            raise RuntimeError("R8 warmup duration is already complete")
        validate_validation_metrics(metrics)
        self.warmup_epochs_completed += 1
        control = route_corrected_validation_loss(
            {
                "split": "Validation",
                "contract": LOSS_CONTRACT,
                "value": float(metrics["validation_loss"]),
            },
            self.scheduler,
            self.early_stopping,
        )
        macro_f1 = float(metrics["validation_macro_f1"])
        validation_loss = float(metrics["validation_loss"])
        is_best = macro_f1 > self.best_warmup_macro_f1 or (
            macro_f1 == self.best_warmup_macro_f1
            and validation_loss < self.best_warmup_validation_loss
        )
        if is_best:
            self.best_warmup_epoch = self.warmup_epochs_completed
            self.best_warmup_macro_f1 = macro_f1
            self.best_warmup_validation_loss = validation_loss
            save_complete_checkpoint(
                self.checkpoint_path,
                self.model,
                num_classes=int(self.policy["num_classes"]),
                dropout=float(self.experiment["dropout"]),
                metadata={
                    "phase": "R8",
                    "experiment_id": self.experiment_id,
                    "stage": "head_warmup",
                    "warmup_epoch": self.best_warmup_epoch,
                    "validation_macro_f1": self.best_warmup_macro_f1,
                    "validation_loss": self.best_warmup_validation_loss,
                    "test_evidence_used": False,
                },
            )
            self.best_warmup_state_sha256 = state_dict_sha256(
                self.model.state_dict()
            )
        record = {
            "epoch": self.warmup_epochs_completed,
            "is_best": is_best,
            **control,
        }
        self.validation_control_records.append(record)
        return record

    def transition_to_layer4(self) -> dict[str, Any]:
        if self.stage != "head_warmup":
            raise RuntimeError("R8 session cannot transition from its current stage")
        if self.warmup_epochs_completed != self.experiment["warmup_epochs"]:
            raise RuntimeError("R8 warmup must finish before layer4 unfreeze")
        if self.best_warmup_epoch is None or not self.checkpoint_path.is_file():
            raise RuntimeError("R8 best warmup checkpoint is unavailable")
        restored, checkpoint = load_complete_checkpoint(self.checkpoint_path)
        metadata = checkpoint.get("metadata", {})
        if metadata.get("experiment_id") != self.experiment_id:
            raise RuntimeError("R8 warmup checkpoint experiment mismatch")
        restored_hash = state_dict_sha256(restored.state_dict())
        if restored_hash != self.best_warmup_state_sha256:
            raise RuntimeError("R8 did not restore the best warmup state")
        self.model = restored
        configure_trainable_parameters(self.model, "partial_finetune")
        self.stage = "layer4_finetune"
        self.optimizer = build_stage_optimizer(self.model, self.policy)
        self.scheduler, self.early_stopping = build_stage_controls(
            self.optimizer, self.policy
        )
        return {
            "restored_warmup_epoch": self.best_warmup_epoch,
            "restored_state_sha256": restored_hash,
            "stage": self.stage,
            "trainable_parameter_names": trainable_parameter_names(self.model),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "warmup_epochs_planned": self.experiment["warmup_epochs"],
            "warmup_epochs_completed": self.warmup_epochs_completed,
            "best_warmup_epoch": self.best_warmup_epoch,
            "initial_state_sha256": self.initial_state_sha256,
            "best_warmup_state_sha256": self.best_warmup_state_sha256,
            "current_state_sha256": state_dict_sha256(self.model.state_dict()),
            "trainable_parameter_names": trainable_parameter_names(self.model),
            "test_evidence_used": False,
        }


def _validate_result_rows(
    results: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> None:
    if not results:
        raise ValueError("R8 repeated-seed results are empty")
    expected_pairs = {
        (experiment_id, seed)
        for experiment_id in policy["experiments"]
        for seed in policy["experiment_seeds"]
    }
    observed_pairs = set()
    initial_hashes = set()
    for row in results:
        if set(row) != RESULT_FIELDS:
            raise ValueError("R8 result row contains missing or unauthorized fields")
        pair = (str(row["experiment_id"]), int(row["seed"]))
        if pair in observed_pairs:
            raise ValueError("R8 result contains a duplicate experiment seed")
        observed_pairs.add(pair)
        initial_hashes.add(str(row["initial_state_sha256"]))
        for name in ("train_accuracy", "validation_accuracy", "validation_macro_f1"):
            value = float(row[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"R8 result metric is invalid: {name}")
    if observed_pairs != expected_pairs:
        raise ValueError("R8 results do not cover every declared experiment seed")
    if len(initial_hashes) != 1:
        raise ValueError("R8 experiments did not share one initial tensor hash")


def summarize_repeated_seed_results(
    results: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = dict(policy or load_r8_policy())
    _validate_result_rows(results, policy)
    summaries = []
    for experiment_id in policy["experiments"]:
        rows = [row for row in results if row["experiment_id"] == experiment_id]
        macro_f1 = [float(row["validation_macro_f1"]) for row in rows]
        validation_accuracy = [float(row["validation_accuracy"]) for row in rows]
        gaps = [
            float(row["train_accuracy"]) - float(row["validation_accuracy"])
            for row in rows
        ]
        summaries.append(
            {
                "experiment_id": experiment_id,
                "seed_count": len(rows),
                "seeds": sorted(int(row["seed"]) for row in rows),
                "initial_state_sha256": str(rows[0]["initial_state_sha256"]),
                "mean_validation_macro_f1": statistics.mean(macro_f1),
                "std_validation_macro_f1": statistics.stdev(macro_f1),
                "mean_validation_accuracy": statistics.mean(validation_accuracy),
                "std_validation_accuracy": statistics.stdev(validation_accuracy),
                "mean_generalization_gap": statistics.mean(gaps),
                "test_evidence_used": False,
            }
        )
    return summaries


def select_candidate_from_validation(
    summaries: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r8_policy())
    by_id = {str(row["experiment_id"]): row for row in summaries}
    if set(by_id) != set(policy["experiments"]):
        raise ValueError("R8 summary coverage is incomplete")
    if any(row.get("test_evidence_used") is not False for row in summaries):
        raise ValueError("R8 selection cannot use Test evidence")
    baseline = by_id["M0"]
    selection = policy["experiment_selection"]
    candidates = []
    comparisons = []
    for experiment_id in ("M1", "M2", "M3"):
        row = by_id[experiment_id]
        macro_improvement = (
            float(row["mean_validation_macro_f1"])
            - float(baseline["mean_validation_macro_f1"])
        )
        accuracy_change = (
            float(row["mean_validation_accuracy"])
            - float(baseline["mean_validation_accuracy"])
        )
        gap_reduction = (
            float(baseline["mean_generalization_gap"])
            - float(row["mean_generalization_gap"])
        )
        eligible = (
            macro_improvement >= selection["minimum_macro_f1_improvement"]
            and accuracy_change >= -selection["maximum_validation_accuracy_decrease"]
            and gap_reduction >= selection["minimum_generalization_gap_reduction"]
        )
        comparison = {
            "experiment_id": experiment_id,
            "macro_f1_improvement": macro_improvement,
            "validation_accuracy_change": accuracy_change,
            "generalization_gap_reduction": gap_reduction,
            "eligible": eligible,
            "test_evidence_used": False,
        }
        comparisons.append(comparison)
        if eligible:
            candidates.append(row)
    selected = None
    if candidates:
        selected = max(
            candidates,
            key=lambda row: (
                float(row["mean_validation_macro_f1"]),
                float(row["mean_validation_accuracy"]),
                -float(row["mean_generalization_gap"]),
            ),
        )["experiment_id"]
    return {
        "selected_experiment_id": selected,
        "comparisons": comparisons,
        "selection_source": "Validation_only",
        "test_evidence_used": False,
    }


def build_r8_report(
    r7_report: Mapping[str, Any],
    r6_report: Mapping[str, Any],
    checks: Mapping[str, bool],
    pretrained_state_available: bool,
    repeated_seed_evidence_available: bool,
    authorized_candidate_id: str | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r8_policy())
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    blocked_reasons = []
    if not r7_report.get("gate_passed"):
        blocked_reasons.append("R7_GATE_NOT_PASSED")
    if r6_report.get("selected_recipe_id") is None:
        blocked_reasons.append("TRANSFORM_RECIPE_NOT_SELECTED")
    if "TRAIN_MANIFEST_NOT_AUTHORIZED" in r6_report.get("blocked_reasons", []):
        blocked_reasons.append("TRAIN_MANIFEST_NOT_AUTHORIZED")
    if not pretrained_state_available:
        blocked_reasons.append("PRETRAINED_INITIAL_STATE_UNAVAILABLE")
    if not repeated_seed_evidence_available:
        blocked_reasons.append("REPEATED_SEED_VALIDATION_EVIDENCE_UNAVAILABLE")
    if failed_checks:
        blocked_reasons.append("R8_PROTOCOL_CHECKS_FAILED")
    if authorized_candidate_id is None:
        blocked_reasons.append("NO_R8_CANDIDATE_AUTHORIZED")
    return {
        "schema_version": 1,
        "phase": "R8",
        "lineage": policy["policy_version"],
        "status": "passed" if not blocked_reasons else "blocked",
        "gate_passed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "protocol_exit_checks_passed": not failed_checks,
        "failed_protocol_checks": failed_checks,
        "experiments": list(policy["experiments"]),
        "experiment_seeds": policy["experiment_seeds"],
        "shared_initial_state_required": True,
        "shared_initial_state_verified_with_fixture": checks.get(
            "shared_initial_state", False
        ),
        "authorized_pretrained_state_available": pretrained_state_available,
        "selected_transform_recipe_id": r6_report.get("selected_recipe_id"),
        "warmup_checkpoint_restore_verified": checks.get(
            "warmup_checkpoint_restore", False
        ),
        "layer4_only_unfreeze_verified": checks.get("layer4_only_unfreeze", False),
        "learning_rate_ranges_verified": checks.get("learning_rate_ranges", False),
        "corrected_validation_controls_verified": checks.get(
            "corrected_validation_controls", False
        ),
        "repeated_seed_selection_logic_verified": checks.get(
            "repeated_seed_selection", False
        ),
        "repeated_seed_validation_evidence_available": repeated_seed_evidence_available,
        "authorized_candidate_experiment_id": authorized_candidate_id,
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
        raise RuntimeError("R8 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return output_path
