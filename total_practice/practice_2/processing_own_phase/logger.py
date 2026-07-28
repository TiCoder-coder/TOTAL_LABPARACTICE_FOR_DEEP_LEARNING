"""Logging and TensorBoard utilities for Practice 2."""

import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with console and optional file handler (with rotation)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger is already set up
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler with Rotation (max 5MB, keep 3 backups)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent logs from propagating to root logger and duplicating
    logger.propagate = False
    return logger


class TensorBoardLogger:
    """Wrapper for TensorBoard SummaryWriter with custom methods."""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, tag_dict: Dict[str, float], global_step: int):
        """Log multiple scalars (e.g. loss, accuracy)."""
        for tag, value in tag_dict.items():
            self.writer.add_scalar(tag, value, global_step)

    def log_learning_rate(self, lr: float, global_step: int):
        self.writer.add_scalar("Train/LearningRate", lr, global_step)

    def log_model_graph(self, model: nn.Module, dummy_input: torch.Tensor):
        """Log the model computation graph. Catches exceptions if tracing fails."""
        try:
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not log model graph to TensorBoard: {e}")

    def log_histograms(self, model: nn.Module, global_step: int):
        """Log histograms of model parameters and gradients."""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f"{name}/weights", param.data, global_step)
                self.writer.add_histogram(f"{name}/gradients", param.grad, global_step)
            elif param.requires_grad:
                self.writer.add_histogram(f"{name}/weights", param.data, global_step)

    def log_images(self, tag: str, images: torch.Tensor, global_step: int):
        """Log a grid of images. Images should be in [B, C, H, W] format."""
        import torchvision
        grid = torchvision.utils.make_grid(images, normalize=True)
        self.writer.add_image(tag, grid, global_step)

    def close(self):
        """Close the writer."""
        self.writer.close()
