"""Configuration module for Practice 2."""

from .core_config import (
    PROJECT_ROOT,
    DATA_DIR,
    OUTPUT_DIR,
    RUNS_DIR,
    REPORTS_DIR,
    CHECKPOINTS_DIR,
    MODEL_SAVE_PATH,
    EXPERIMENT_RESULTS_CSV,
    CLASS_NAMES,
    NUM_CLASSES,
    INPUT_DIM,
)
from .training_config import CONFIG
from .experiment_config import EXPERIMENTS

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "OUTPUT_DIR",
    "RUNS_DIR",
    "REPORTS_DIR",
    "CHECKPOINTS_DIR",
    "MODEL_SAVE_PATH",
    "EXPERIMENT_RESULTS_CSV",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "INPUT_DIM",
    "CONFIG",
    "EXPERIMENTS",
]
