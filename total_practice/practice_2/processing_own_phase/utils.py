"""Utility functions for Practice 2."""

import os
import random
from typing import Optional

import numpy as np
import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """Get the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_reproducibility(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string (MM:SS)."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a PyTorch model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_environment_summary(device: torch.device):
    """Print a quick summary of the environment."""
    print(f"Device: {device}")
    print(f"PyTorch Version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA CUDNN: {torch.backends.cudnn.version()}")
    elif device.type == "mps":
        print("Using Apple Metal Performance Shaders (MPS).")
