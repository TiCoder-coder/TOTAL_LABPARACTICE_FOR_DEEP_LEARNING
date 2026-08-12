"""Experiment runner for Practice 2."""

import os
import uuid
import logging
from typing import Dict, Any, Optional

import json
import time
from datetime import datetime
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from configs import (
    EXPERIMENT_RESULTS_CSV,
    CONFIG,
    EXPERIMENTS,
    CLASS_NAMES,
    OUTPUT_DIR,
    REPORTS_DIR,
)
from .data import load_datasets
from .evaluate import evaluate as eval_full, compute_classification_metrics
from .logger import setup_logger, TensorBoardLogger
from .model import build_model, sanity_check_model
from .save_load import load_model_from_checkpoint
from .train import train_model
from .utils import get_device, setup_reproducibility

logger = setup_logger(__name__)

CONTROLLED_EXPERIMENT_IDS = (
    "E1_resnet18_head",
    "E2_resnet18_partial",
)
CONTROLLED_FIELDS = tuple(
    sorted(
        set(EXPERIMENTS[CONTROLLED_EXPERIMENT_IDS[0]])
        .union(EXPERIMENTS[CONTROLLED_EXPERIMENT_IDS[1]])
        .difference({"training_mode"})
    )
)


def _make_selection_dataloaders(train_subset, val_subset, config):
    """Create only Train and Validation loaders for experiment selection."""
    num_workers = config["num_workers"]
    persistent = num_workers > 0
    pin_memory = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader


def validate_controlled_experiment_configs() -> None:
    """Require E1/E2 to differ only in the fine-tuning strategy."""
    e1 = EXPERIMENTS[CONTROLLED_EXPERIMENT_IDS[0]]
    e2 = EXPERIMENTS[CONTROLLED_EXPERIMENT_IDS[1]]
    mismatches = [
        field for field in CONTROLLED_FIELDS
        if e1.get(field) != e2.get(field)
    ]
    if mismatches:
        raise ValueError(
            "E1/E2 controlled configuration mismatch: "
            + ", ".join(mismatches)
        )
    if e1["training_mode"] == e2["training_mode"]:
        raise ValueError("E1/E2 must use different fine-tuning strategies.")
    differing_fields = {
        field
        for field in set(e1).union(e2)
        if e1.get(field) != e2.get(field)
    }
    if differing_fields != {"training_mode"}:
        raise ValueError(
            "E1/E2 may differ only by training_mode; found: "
            + ", ".join(sorted(differing_fields))
        )


def select_experiment_by_validation(results):
    """Select the best experiment using Validation accuracy only."""
    if not results:
        raise ValueError("At least one Validation result is required.")
    for result in results:
        metadata = result.get("metadata", {})
        if metadata.get("test_data_used") is True:
            raise ValueError("Test data must not be used during model selection.")
        forbidden = [
            key for key in metadata
            if key.lower().startswith("test") and key != "test_data_used"
        ]
        if forbidden:
            raise ValueError(
                "Test metrics are forbidden during experiment selection: "
                + ", ".join(forbidden)
            )
    return max(results, key=lambda result: result["metadata"]["accuracy"])


def run_experiment(
    experiment_id: str, 
    use_quick_run: bool = False,
    resume_from: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a full experiment based on its configuration."""
    
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Experiment {experiment_id} not found in config.")
        
    exp_config = EXPERIMENTS[experiment_id].copy()
    
    # Merge global config with experiment-specific config
    run_config = CONFIG.copy()
    run_config.update(exp_config)
    if config_overrides:
        run_config.update(config_overrides)
    
    # Generate run ID
    run_id = f"{experiment_id}_{uuid.uuid4().hex[:8]}"
    if use_quick_run:
        logger.info("QUICK RUN MODE ON: Setting epochs=1, small batch size.")
        run_config["epochs"] = 1
        run_id = (
            f"{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:6]}"
        )
    run_config["run_id"] = run_id
    run_dir = Path(CONFIG["runs_dir"]) / run_id
    os.makedirs(run_dir, exist_ok=True)
    run_config["output_dir"] = str(run_dir)
    
    # Setup File Logger for this run
    log_file = os.path.join(run_dir, f"{run_id}.log")
    run_logger = setup_logger(__name__, log_file=log_file)
    
    # Setup TensorBoard Logger
    tb_logger = TensorBoardLogger(log_dir=str(run_dir))
    
    run_logger.info(f"Starting experiment: {experiment_id} - Run ID: {run_id}")
    run_logger.info(f"Configuration: {run_config}")
    
    # Set seed
    setup_reproducibility(run_config["seed"])
    device = get_device()
    run_logger.info(f"Using device: {device}")
    
    # Load Data
    run_logger.info("Loading Datasets...")
    train_subset, val_subset, _test_dataset = load_datasets(
        data_dir=run_config["data_dir"],
        val_ratio=1.0 - run_config["train_split_ratio"],
        seed=run_config["seed"]
    )
    train_loader, val_loader = _make_selection_dataloaders(
        train_subset,
        val_subset,
        run_config,
    )
    
    # Build Model
    run_logger.info("Initializing Model...")
    model = build_model(
        model_name=run_config["model_name"],
        training_mode=run_config["training_mode"],
        num_classes=len(CLASS_NAMES),
        dropout=run_config.get("dropout", 0.0),
        hidden_layers=run_config.get("hidden_layers", []),
    )
    
    # Log Model Graph
    try:
        dataiter = iter(train_loader)
        images, _ = next(dataiter)
        tb_logger.log_model_graph(model.to(device), images.to(device))
    except Exception as e:
        run_logger.warning(f"Could not log model graph: {e}")
    
    # Sanity Check
    run_logger.info("Running Model Sanity Check...")
    sanity_res = sanity_check_model(model, device)
    run_logger.info(f"Sanity Check Passed. Model Output Shape: {sanity_res['forward_shape']}")
    
    # Train
    run_logger.info("Starting Training...")
    train_start = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=run_config,
        device=device,
        resume_from=resume_from,
        tb_logger=tb_logger,
        run_logger=run_logger
    )
    training_time = time.time() - train_start
    
    # Evaluate Best Model
    run_logger.info("Evaluating Best Model on Validation Set...")
    best_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"Best checkpoint was not created for {experiment_id}: {best_path}"
        )
    model = load_model_from_checkpoint(best_path, device=device)
        
    val_start = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    val_res = eval_full(model, val_loader, criterion, device)
    val_time = time.time() - val_start

    manifest_path = Path(run_dir) / "checkpoint_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        verification = {
            "status": "PASS",
            "validation_loss": float(val_res["loss"]),
            "validation_accuracy": float(val_res["accuracy"]),
            "verified_at": datetime.now().isoformat(),
        }
        manifest["selected_checkpoint_reload_verification"] = verification
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    
    metrics = compute_classification_metrics(val_res["predictions"], val_res["labels"], len(CLASS_NAMES), CLASS_NAMES)

    if len(history["val_loss"]) < 2 and not use_quick_run:
        raise RuntimeError(
            "Controlled experiments must contain at least two training epochs."
        )
    best_epoch = int(history["best_epoch"])
    best_epoch_index = best_epoch - 1
    
    # Save results to CSV
    result = {
        "exp_id": experiment_id,
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": run_config["model_name"],
        "seed": run_config["seed"],
        "accuracy": val_res["accuracy"],
        "precision": metrics["macro_avg"]["precision"],
        "recall": metrics["macro_avg"]["recall"],
        "f1_score": metrics["macro_avg"]["f1-score"],
        "loss": val_res["loss"],
        "training_time": training_time,
        "validation_time": val_time,
        "learning_rate": run_config["learning_rate"],
        "batch_size": run_config["batch_size"],
        "optimizer": run_config["optimizer"],
        "scheduler": run_config.get("scheduler", "None"),
        "freeze_strategy": run_config["training_mode"],
        "epochs_trained": len(history["val_loss"]),
        "best_epoch": best_epoch,
        "best_val_acc": max(history["val_acc"]) / 100.0,
        "best_val_loss": history["val_loss"][best_epoch_index],
        "early_stopping_triggered": bool(history["stopped_early"]),
        "selection_source": "validation_only",
        "test_data_used": False,
        "checkpoint_path": str(Path(best_path).resolve()),
    }
    _append_to_csv(result, EXPERIMENT_RESULTS_CSV)
    
    # Save full JSON summary
    summary = {
        "metadata": result,
        "history": history,
        "metrics": metrics,
        "config": run_config
    }
    summary_path = os.path.join(run_dir, f"{run_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    tb_logger.close()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return summary


def run_controlled_experiments(
    output_dir: str = str(OUTPUT_DIR),
    reports_dir: str = str(REPORTS_DIR),
) -> Dict[str, Any]:
    """Run E1/E2, select by Validation only, and save presentation artifacts."""
    validate_controlled_experiment_configs()
    results = [
        run_experiment(experiment_id, use_quick_run=False)
        for experiment_id in CONTROLLED_EXPERIMENT_IDS
    ]
    selected = select_experiment_by_validation(results)

    output_path = Path(output_dir)
    reports_path = Path(reports_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for result in results:
        metadata = result["metadata"]
        comparison_rows.append({
            "experiment": metadata["exp_id"],
            "strategy": metadata["freeze_strategy"],
            "epochs_trained": metadata["epochs_trained"],
            "best_epoch": metadata["best_epoch"],
            "best_val_accuracy": metadata["accuracy"],
            "best_val_loss": metadata["best_val_loss"],
            "macro_f1": metadata["f1_score"],
            "training_time": metadata["training_time"],
        })
    comparison_path = output_path / "controlled_experiment_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)

    histories = {
        result["metadata"]["exp_id"]: result["history"]
        for result in results
    }
    history_path = output_path / "controlled_training_history.json"
    history_path.write_text(json.dumps(histories, indent=2))

    selection_artifact = {
        "selection_metric": "val_accuracy",
        "selection_source": "validation_only",
        "selected_experiment": selected["metadata"]["exp_id"],
        "selected_checkpoint": selected["metadata"]["checkpoint_path"],
        "best_epoch": selected["metadata"]["best_epoch"],
        "best_val_accuracy": selected["metadata"]["accuracy"],
        "best_val_loss": selected["metadata"]["best_val_loss"],
        "epochs_trained": selected["metadata"]["epochs_trained"],
        "early_stopping_triggered": selected["metadata"]["early_stopping_triggered"],
        "test_data_used": False,
        "experiments": [row["experiment"] for row in comparison_rows],
    }
    selection_path = output_path / "controlled_experiment_selection.json"
    selection_path.write_text(json.dumps(selection_artifact, indent=2))

    from .visualize import plot_learning_rate, plot_training_curves
    selected_history = selected["history"]
    plot_training_curves(
        selected_history,
        save_path=str(reports_path / "controlled_training_curves.png"),
    )
    plot_learning_rate(
        selected_history,
        save_path=str(reports_path / "controlled_learning_rate.png"),
    )

    return {
        "selection": selection_artifact,
        "results": results,
        "comparison_path": str(comparison_path),
        "history_path": str(history_path),
        "selection_path": str(selection_path),
    }


def _append_to_csv(result: Dict, csv_path: str) -> None:
    """Append a single result to the CSV file ensuring no duplicate exp_ids."""
    path = Path(csv_path)
    new_df = pd.DataFrame([result])
    
    if path.exists():
        try:
            existing_df = pd.read_csv(path)
            if not existing_df.empty and "exp_id" in existing_df.columns:
                existing_df = existing_df[existing_df["exp_id"] != result["exp_id"]]
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                existing_df = new_df
        except Exception:
            existing_df = new_df
    else:
        existing_df = new_df
        
    existing_df.to_csv(path, index=False)
