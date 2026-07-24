"""Central configuration for FashionMNIST classification experiments."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_SAVE_PATH = OUTPUT_DIR / "best_model.pth"
EXPERIMENT_RESULTS_CSV = PROJECT_ROOT / "save_log_agent_process_each_phase" / "experiment_results.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "save_log_agent_process_each_phase").mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

NUM_CLASSES = len(CLASS_NAMES)
INPUT_DIM = 28 * 28

CONFIG = {
    "seed": 42,
    "batch_size": 64,
    "num_workers": 0,
    "train_split_ratio": 0.9,
    "hidden_dims": [128],
    "dropout": 0.0,
    "epochs": 10,
    "learning_rate": 0.01,
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 0.0,
    "data_dir": str(DATA_DIR),
    "output_dir": str(OUTPUT_DIR),
    "model_save_path": str(MODEL_SAVE_PATH),
    "experiment_results_csv": str(EXPERIMENT_RESULTS_CSV),
}

EXPERIMENTS = {
    "E0_baseline": {
        "hidden_dims": [128],
        "dropout": 0.0,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "batch_size": 64,
        "epochs": 10,
        "description": "Baseline: 1 hidden layer, SGD, lr=0.01",
    },
    "E1_lr_low": {
        "hidden_dims": [128],
        "dropout": 0.0,
        "optimizer": "SGD",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 10,
        "description": "Low learning rate experiment",
    },
    "E2_lr_high": {
        "hidden_dims": [128],
        "dropout": 0.0,
        "optimizer": "SGD",
        "learning_rate": 0.1,
        "batch_size": 64,
        "epochs": 10,
        "description": "High learning rate experiment",
    },
    "E3_deeper": {
        "hidden_dims": [256, 128],
        "dropout": 0.0,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "batch_size": 64,
        "epochs": 10,
        "description": "Deeper architecture: 2 hidden layers",
    },
    "E4_dropout": {
        "hidden_dims": [256, 128],
        "dropout": 0.2,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "batch_size": 64,
        "epochs": 10,
        "description": "Regularization with dropout=0.2",
    },
    "E5_adam": {
        "hidden_dims": [256, 128],
        "dropout": 0.2,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 10,
        "description": "Adam optimizer comparison",
    },
}
