"""Core configuration containing paths and dataset specifications."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODEL_SAVE_PATH = CHECKPOINTS_DIR / "best_model.pth"
EXPERIMENT_RESULTS_CSV = OUTPUT_DIR / "experiment_results.csv"

# Ensure directories exist
for d in [DATA_DIR, OUTPUT_DIR, RUNS_DIR, REPORTS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset specifications (CIFAR-10)
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

NUM_CLASSES = len(CLASS_NAMES)
INPUT_DIM = 224 * 224 * 3  # After resizing for ImageNet
