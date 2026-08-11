from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = PROJECT_ROOT / "runs"
EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "experiments"
MODEL_SAVE_PATH = OUTPUT_DIR / "fashion_mnist_model.pth"

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
    "validation_ratio": 0.10,
    "batch_size": 64,
    "num_workers": 0,
    "hidden_dims": (128,),
    "dropout": 0.0,
    "num_classes": NUM_CLASSES,
    "input_dim": INPUT_DIM,
    "epochs": 10,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "use_augmentation": False,
}

EXPERIMENTS = [
    {
        **CONFIG,
        "experiment_id": "E0_baseline",
        "hidden_dims": (128,),
        "dropout": 0.0,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "use_augmentation": False,
    },
    {
        **CONFIG,
        "experiment_id": "E1_deeper",
        "hidden_dims": (256, 128),
        "dropout": 0.0,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "use_augmentation": False,
    },
    {
        **CONFIG,
        "experiment_id": "E2_dropout",
        "hidden_dims": (256, 128),
        "dropout": 0.2,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "use_augmentation": False,
    },
    {
        **CONFIG,
        "experiment_id": "E3_sgd",
        "hidden_dims": (256, 128),
        "dropout": 0.0,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "use_augmentation": False,
    },
    {
        **CONFIG,
        "experiment_id": "E4_augmentation",
        "hidden_dims": (256, 128),
        "dropout": 0.0,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "use_augmentation": True,
    },
]
