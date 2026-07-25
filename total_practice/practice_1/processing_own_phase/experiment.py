import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from .train import run_training


def run_experiments(
    configs: Sequence[Dict],
    train_subset,
    augmented_train_subset,
    validation_loader: DataLoader,
    device: torch.device,
    runs_dir: Path,
    output_dir: Path,
    verbose: bool = True,
) -> Tuple[list, Dict, Path]:
    run_root = runs_dir / time.strftime("%Y%m%d-%H%M%S")
    experiment_output_dir = output_dir / "experiments"
    run_root.mkdir(parents=True, exist_ok=True)
    experiment_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    results = []

    for config in configs:
        experiment_id = config["experiment_id"]
        training_dataset = (
            augmented_train_subset
            if config["use_augmentation"]
            else train_subset
        )
        if verbose:
            print(f"Running {experiment_id}")
        result = run_training(
            config,
            training_dataset,
            validation_loader,
            device,
            log_directory=run_root / experiment_id,
            verbose=verbose,
        )
        results.append(result)
        torch.save(
            {
                "model_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value
                    in result["model"].state_dict().items()
                },
                "config": result["config"],
                "best_epoch": result["best_epoch"],
                "best_validation_accuracy": (
                    result["best_validation_accuracy"]
                ),
                "best_validation_loss": (
                    result["best_validation_loss"]
                ),
                "history": result["history"],
            },
            experiment_output_dir
            / f"{experiment_id}.pth",
        )

    selected_result = max(
        results,
        key=lambda result: (
            result["best_validation_accuracy"],
            -result["best_validation_loss"],
        ),
    )
    return results, selected_result, run_root
