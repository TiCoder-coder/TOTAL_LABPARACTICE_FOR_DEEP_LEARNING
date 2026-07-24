"""Helper utilities for reproducibility, device selection, and logging."""

import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """Return the best available torch device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_reproducibility(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into Mm Ss string."""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}m {secs:.1f}s"


def print_environment_summary(device: torch.device) -> None:
    """Print a small environment summary banner."""
    print("=" * 60)
    print("ENVIRONMENT SUMMARY")
    print("=" * 60)
    print(f"Python version     : {sys.version.split()[0]}")
    print(f"PyTorch version    : {torch.__version__}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    print(f"MPS available      : {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"Using device       : {device}")
    print("=" * 60)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
