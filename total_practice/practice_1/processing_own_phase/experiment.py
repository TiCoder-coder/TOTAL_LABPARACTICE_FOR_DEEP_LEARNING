"""Experiment runner for controlled hyperparameter experiments."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import EXPERIMENT_RESULTS_CSV, INPUT_DIM, NUM_CLASSES
from .evaluate import evaluate
from .model import FashionMLP, build_model
from .train import fit
from .utils import count_parameters, format_time


def build_optimizer(model: FashionMLP, name: str, lr: float, **extra) -> torch.optim.Optimizer:
    """Build an optimizer from a name string."""
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=extra.get("momentum", 0.9),
            weight_decay=extra.get("weight_decay", 0.0),
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=extra.get("weight_decay", 0.0),
        )
    raise ValueError(f"Unknown optimizer: {name}")


class ExperimentRunner:
    """Run controlled experiments with checkpoints and CSV logging."""

    def __init__(
        self,
        device: torch.device,
        csv_path: str = str(EXPERIMENT_RESULTS_CSV),
        output_dir: Optional[str] = None,
    ):
        self.device = device
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir) if output_dir else None
        self.results: List[Dict] = []

    def run(
        self,
        exp_id: str,
        description: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        hidden_dims: List[int],
        dropout: float = 0.0,
        optimizer_name: str = "SGD",
        learning_rate: float = 0.01,
        batch_size: int = 64,
        epochs: int = 10,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        save_checkpoint: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """Run a single experiment and return its results."""
        if verbose:
            print("=" * 70)
            print(f"EXPERIMENT: {exp_id}")
            print(f"Description: {description}")
            print(
                f"Architecture: {hidden_dims} | Dropout: {dropout} | "
                f"Optimizer: {optimizer_name} | LR: {learning_rate} | "
                f"Batch: {batch_size} | Epochs: {epochs}"
            )
            print("=" * 70)

        torch.manual_seed(42)
        model = build_model(hidden_dims, dropout).to(self.device)
        n_params = count_parameters(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(
            model, optimizer_name, learning_rate,
            momentum=momentum, weight_decay=weight_decay,
        )

        best_state = None
        best_val_acc = -1.0

        def save_best_callback(m, epoch, val_metrics):
            nonlocal best_state, best_val_acc
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

        fit_result = fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            self.device,
            epochs=epochs,
            save_best_callback=save_best_callback,
            verbose=verbose,
        )

        if save_checkpoint and best_state is not None and self.output_dir is not None:
            ckpt_path = self.output_dir / f"{exp_id}.pt"
            torch.save({
                "model_state_dict": best_state,
                "exp_id": exp_id,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "best_epoch": fit_result["best_epoch"],
                "best_val_acc": fit_result["best_val_acc"],
            }, ckpt_path)
            if verbose:
                print(f"  Saved checkpoint: {ckpt_path}")

        result = {
            "exp_id": exp_id,
            "description": description,
            "hidden_dims": json.dumps(hidden_dims),
            "dropout": dropout,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "n_parameters": n_params,
            "best_epoch": fit_result["best_epoch"],
            "best_val_loss": fit_result["best_val_loss"],
            "best_val_acc": fit_result["best_val_acc"],
            "total_time_seconds": fit_result["total_time"],
            "history": fit_result["history"],
        }
        self.results.append(result)
        self._append_to_csv(result)

        if verbose:
            print(
                f"  Best Val Acc: {fit_result['best_val_acc']:.4f} "
                f"@ Epoch {fit_result['best_epoch']} | "
                f"Params: {n_params:,} | Time: {format_time(fit_result['total_time'])}"
            )
        return result

    def _append_to_csv(self, result: Dict) -> None:
        """Append a single result to the CSV file."""
        row = {k: v for k, v in result.items() if k != "history"}
        df = pd.DataFrame([row])
        if self.csv_path.exists():
            df.to_csv(self.csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.csv_path, index=False)

    def save_all(self) -> None:
        """Save all results to a single CSV."""
        if not self.results:
            return
        rows = [{k: v for k, v in r.items() if k != "history"} for r in self.results]
        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)
