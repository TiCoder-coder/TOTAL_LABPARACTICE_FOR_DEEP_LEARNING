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

from configs import EXPERIMENT_RESULTS_CSV, CONFIG, EXPERIMENTS, CLASS_NAMES
from .data import make_dataloaders, load_datasets
from .evaluate import evaluate as eval_full, compute_classification_metrics
from .logger import setup_logger, TensorBoardLogger
from .model import build_model, sanity_check_model
from .save_load import load_model_from_checkpoint
from .train import train_model
from .utils import get_device, setup_reproducibility

logger = setup_logger(__name__)


def run_experiment(
    experiment_id: str, 
    use_quick_run: bool = False,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """Run a full experiment based on its configuration."""
    
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Experiment {experiment_id} not found in config.")
        
    exp_config = EXPERIMENTS[experiment_id].copy()
    
    # Merge global config with experiment-specific config
    run_config = CONFIG.copy()
    run_config.update(exp_config)
    
    # Generate run ID
    run_id = f"{experiment_id}_{uuid.uuid4().hex[:8]}"
    run_config["run_id"] = run_id
    
    if use_quick_run:
        logger.info("QUICK RUN MODE ON: Setting epochs=1, small batch size.")
        run_config["epochs"] = 1
        run_id = f"{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    train_subset, val_subset, test_dataset = load_datasets(
        data_dir=run_config["data_dir"],
        val_ratio=1.0 - run_config["train_split_ratio"],
        seed=run_config["seed"]
    )
    
    train_loader, val_loader, test_loader = make_dataloaders(
        train_subset, val_subset, test_dataset,
        batch_size=run_config["batch_size"],
        num_workers=run_config["num_workers"]
    )
    
    # Build Model
    run_logger.info("Initializing Model...")
    model = build_model(
        model_name=run_config["model_name"],
        training_mode=run_config["training_mode"],
        num_classes=len(CLASS_NAMES)
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
    if os.path.exists(best_path):
        model = load_model_from_checkpoint(best_path, device=device)
        
    val_start = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    val_res = eval_full(model, val_loader, criterion, device)
    val_time = time.time() - val_start
    
    metrics = compute_classification_metrics(val_res["predictions"], val_res["labels"], len(CLASS_NAMES), CLASS_NAMES)
    
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
        "best_val_acc": max(history["val_acc"])  # keep for backwards compatibility if needed
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
