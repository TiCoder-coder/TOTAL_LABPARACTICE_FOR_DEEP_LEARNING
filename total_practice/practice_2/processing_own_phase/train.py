"""Training pipeline module with robust production features."""

import os
import time
import logging
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import format_time
from configs import CONFIG

# Configure logger
logger = logging.getLogger(__name__)


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Factory function for optimizers."""
    opt_name = config.get("optimizer", "Adam").lower()
    lr = config.get("learning_rate", 0.001)
    wd = config.get("weight_decay", 1e-4)
    
    # Filter only trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    if opt_name == "adam":
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    """Factory function for learning rate schedulers."""
    sched_name = config.get("scheduler", "ReduceLROnPlateau").lower()
    
    if sched_name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
    elif sched_name == "steplr":
        step_size = config.get("scheduler_step_size", 5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    elif sched_name == "cosineannealinglr":
        t_max = config.get("epochs", 10)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    else:
        logger.warning(f"Unsupported scheduler: {sched_name}. Defaulting to None.")
        return None


def get_criterion() -> nn.Module:
    """Factory function for loss."""
    return nn.CrossEntropyLoss()


class CheckpointManager:
    """Manages saving and loading model checkpoints."""
    def __init__(self, output_dir: str, logger_instance: Optional[logging.Logger] = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.latest_path = os.path.join(output_dir, "latest.pt")
        self.best_path = os.path.join(output_dir, "best.pt")
        self.logger = logger_instance or logger

    def save_checkpoint(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: Optional[Any], 
        epoch: int, 
        best_metric: float, 
        config: Dict[str, Any],
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        is_best: bool = False
    ):
        """Save a checkpoint containing all necessary states to resume."""
        metric_name = config.get("best_model_metric", "val_acc")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_metric_name": metric_name,
            "model_name": config.get("model_name", getattr(model, "model_name", "resnet18")),
            "training_mode": config.get("training_mode", "head_only"),
            "num_classes": config.get("num_classes", 10),
            "config": config,
        }
        if val_loss is not None:
            checkpoint["val_loss"] = val_loss
        if val_acc is not None:
            checkpoint["val_acc"] = val_acc
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save latest
        torch.save(checkpoint, self.latest_path)
        self.logger.debug(f"Saved latest checkpoint at epoch {epoch}")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.best_path)
            self.logger.info(
                f"Saved new best model at epoch {epoch} with "
                f"{metric_name}={best_metric:.4f}"
            )

    def load_checkpoint(
        self, 
        path: str, 
        model: nn.Module, 
        optimizer: Optional[torch.optim.Optimizer] = None, 
        scheduler: Optional[Any] = None
    ) -> Tuple[int, float]:
        """Load a checkpoint and restore states."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")

        self.logger.info(f"Loading checkpoint from {path}")
        # Map location to CPU to avoid CUDA OOM on loading
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        epoch = checkpoint.get("epoch", 0)
        best_metric = checkpoint.get("best_metric", float("inf"))
        
        metric_name = checkpoint.get("best_metric_name", "best_metric")
        self.logger.info(
            f"Loaded checkpoint at epoch {epoch} with "
            f"{metric_name}={best_metric:.4f}"
        )
        return epoch, best_metric


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Forward
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            
        else:
            # Standard Precision Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            
            # Gradient Clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            # Optimizer Step
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100.0
    
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100.0
    
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    resume_from: Optional[str] = None,
    tb_logger: Optional[Any] = None,
    run_logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Full training loop orchestrator."""
    active_logger = run_logger or logger
    epochs = config.get("epochs", 10)
    patience = config.get("early_stopping_patience", 3)
    grad_clip = config.get("grad_clip", 1.0)
    best_metric_name = config.get("best_model_metric", "val_acc")
    if best_metric_name not in {"val_acc", "accuracy", "val_loss"}:
        raise ValueError(
            "best_model_metric must be one of: val_acc, accuracy, val_loss"
        )
    
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    criterion = get_criterion()
    
    checkpoint_manager = CheckpointManager(
        config.get("output_dir", "outputs/models"), 
        logger_instance=active_logger
    )
    
    # Initialize AMP Scaler if CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    start_epoch = 0
    best_metric = (
        float("inf")
        if best_metric_name == "val_loss"
        else float("-inf")
    )
    epochs_no_improve = 0
    
    # Resume
    if resume_from:
        try:
            start_epoch, best_metric = checkpoint_manager.load_checkpoint(
                resume_from, model, optimizer, scheduler
            )
            active_logger.info(f"Resumed from epoch {start_epoch}")
        except FileNotFoundError as e:
            active_logger.warning(f"{e}. Starting from scratch.")

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": [],
    }
    best_epoch = 0
    stopped_early = False
    
    active_logger.info("Starting training loop...")
    
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip, scaler
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - start_time
        
        # Update history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)
        
        # Log to TensorBoard
        if tb_logger:
            tb_logger.log_scalars({
                "Loss/Train": train_loss,
                "Loss/Validation": val_loss,
                "Accuracy/Train": train_acc,
                "Accuracy/Validation": val_acc
            }, epoch + 1)
            tb_logger.log_learning_rate(current_lr, epoch + 1)
            tb_logger.log_histograms(model, epoch + 1)
        
        active_logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {format_time(epoch_time)} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%"
        )
        
        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        # Checkpointing and Early Stopping
        current_metric = (
            val_loss
            if best_metric_name == "val_loss"
            else val_acc
        )
        is_best = (
            current_metric < best_metric
            if best_metric_name == "val_loss"
            else current_metric > best_metric
        )
        if is_best:
            best_metric = current_metric
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        checkpoint_manager.save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            best_metric,
            config,
            val_loss=val_loss,
            val_acc=val_acc,
            is_best=is_best,
        )
        
        if patience > 0 and epochs_no_improve >= patience:
            active_logger.info(f"Early stopping triggered after {epoch + 1} epochs!")
            stopped_early = True
            break
            
    history["best_epoch"] = best_epoch
    history["best_metric"] = best_metric
    history["best_metric_name"] = best_metric_name
    history["stopped_early"] = stopped_early
    history["epochs_planned"] = epochs
    active_logger.info("Training complete.")
    return history
