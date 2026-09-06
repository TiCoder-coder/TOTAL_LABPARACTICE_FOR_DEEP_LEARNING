"""FINAL_REFIT_MODE-v1 recipe + seed contract + run matrix.

Plan §40-43, §84-93:
- FINAL_SEEDS-v1 = [42, 123, 2026]
- Exactly 3 planned Phase46 scientific runs.
- Same config / data / scalers / epochs across all seeds.
- Fresh model + optimizer each seed. No warm-start. No validation.
- No early stopping. No BEST semantics.
- Checkpoint type = FINAL_REFIT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FINAL_SEEDS_V1: list[int] = [42, 123, 2026]


@dataclass(frozen=True)
class FinalRefitRecipe:
    model_config_fingerprint: str
    final_data_region: str
    final_population_fingerprint: str
    final_scaling_contract: dict[str, Any]
    batch: int
    optimizer: str
    LR: float
    WD: float
    loss: str
    gradient_clipping: float | None
    revin: bool
    epochs: int
    early_stopping: bool
    validation_loader: None
    scheduler: None
    warmup: None
    accumulation: int
    precision: str
    shuffle: bool
    drop_last: bool
    worker_policy: str
    seed_list: list[int]
    checkpoint_type: str
    attention_retention_during_training: bool
    final_refit_mode: str


def build_recipe(
    locked_config: dict[str, Any],
    final_epoch: int,
    final_dev_fingerprint: str,
    scaling_contract_dict: dict[str, Any],
) -> FinalRefitRecipe:
    """Build the deterministic Phase46 recipe."""
    training = locked_config.get("training", {})
    repro = locked_config.get("reproducibility", {})
    return FinalRefitRecipe(
        model_config_fingerprint=locked_config.get("candidate_config_fingerprint")
        or locked_config.get("config_fingerprint")
        or "",
        final_data_region="FINAL_DEV_REGION-v1",
        final_population_fingerprint=final_dev_fingerprint,
        final_scaling_contract=scaling_contract_dict,
        batch=int(training.get("batch_size", 32)),
        optimizer=str(training.get("optimizer_name", "AdamW")),
        LR=float(training.get("learning_rate", 0.0003)),
        WD=float(training.get("weight_decay", 0.001)),
        loss=str(training.get("loss_name", "MSE")),
        gradient_clipping=(
            float(training["gradient_clip_max_norm"])
            if training.get("gradient_clipping_enabled", False)
            and training.get("gradient_clip_max_norm") is not None
            else None
        ),
        revin=bool(training.get("revin_enabled", False)),
        epochs=final_epoch,
        early_stopping=False,
        validation_loader=None,
        scheduler=None,
        warmup=None,
        accumulation=1,
        precision="fp32",
        shuffle=True,
        drop_last=False,
        worker_policy=str(repro.get("worker_seed_policy", "torch_initial_seed_mod_2_32_numpy_python")),
        seed_list=list(FINAL_SEEDS_V1),
        checkpoint_type="FINAL_REFIT",
        attention_retention_during_training=False,
        final_refit_mode="FINAL_REFIT_MODE-v1",
    )


def recipe_to_dict(r: FinalRefitRecipe) -> dict[str, Any]:
    return {
        "model_config_fingerprint": r.model_config_fingerprint,
        "final_data_region": r.final_data_region,
        "final_population_fingerprint": r.final_population_fingerprint,
        "final_scaling_contract": r.final_scaling_contract,
        "batch": r.batch,
        "optimizer": r.optimizer,
        "LR": r.LR,
        "WD": r.WD,
        "loss": r.loss,
        "gradient_clipping": r.gradient_clipping,
        "RevIN": r.revin,
        "epochs": r.epochs,
        "early_stopping": r.early_stopping,
        "validation_loader": r.validation_loader,
        "scheduler": r.scheduler,
        "warmup": r.warmup,
        "accumulation": r.accumulation,
        "precision": r.precision,
        "shuffle": r.shuffle,
        "drop_last": r.drop_last,
        "worker_policy": r.worker_policy,
        "seed_list": r.seed_list,
        "checkpoint_type": r.checkpoint_type,
        "attention_retention_during_training": r.attention_retention_during_training,
        "final_refit_mode": r.final_refit_mode,
    }


def build_seed_contract() -> dict[str, Any]:
    return {
        "version": "FINAL_SEEDS-v1",
        "seeds": list(FINAL_SEEDS_V1),
        "run_order": list(FINAL_SEEDS_V1),
        "all_seeds_required": True,
        "seed_replacement_forbidden": True,
        "seed_order_dependence_forbidden": True,
        "same_config_all_seeds": True,
        "same_data_all_seeds": True,
        "same_scalers_all_seeds": True,
        "same_epochs_all_seeds": True,
        "same_optimizer_loss_clipping_all_seeds": True,
        "same_environment_all_seeds": True,
        "fresh_model_each_seed": True,
        "fresh_optimizer_each_seed": True,
        "warm_start_forbidden": True,
        "optimizer_state_reuse_forbidden": True,
        "validation_forbidden": True,
        "early_stopping_forbidden": True,
        "no_score_based_rerun": True,
    }


def build_run_matrix(
    locked_candidate_id: str,
    locked_fingerprint: str,
    recipe_sha: str,
    lock_sha: str,
    final_epoch: int,
    final_dev_fingerprint: str,
    x_scaler_bundle_id: str,
    y_scaler_bundle_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in FINAL_SEEDS_V1:
        rows.append({
            "logical_run_id": f"FINAL_TS_SEED_{s}",
            "seed": s,
            "candidate_id": locked_candidate_id,
            "config_fingerprint": locked_fingerprint,
            "training_recipe_fingerprint": recipe_sha,
            "lock_fingerprint": lock_sha,
            "final_refit_epochs": final_epoch,
            "data_region_id": "FINAL_DEV_REGION-v1",
            "population_fingerprint": final_dev_fingerprint,
            "x_scaler_bundle_id": x_scaler_bundle_id,
            "y_scaler_bundle_id": y_scaler_bundle_id,
            "checkpoint_type": "FINAL_REFIT",
            "status": "PLANNED",
        })
    return rows


__all__ = [
    "FINAL_SEEDS_V1",
    "FinalRefitRecipe",
    "build_recipe",
    "recipe_to_dict",
    "build_seed_contract",
    "build_run_matrix",
]
