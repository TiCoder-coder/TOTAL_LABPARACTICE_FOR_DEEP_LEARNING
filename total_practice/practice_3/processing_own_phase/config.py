from pathlib import Path


def find_repo_root(start_path: Path) -> Path:
    """Do nguoc thu muc cha toi khi gap thu muc chua .git, tra ve duong dan repo root."""
    for candidate in [start_path, *start_path.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(
        "Khong tim thay repo root (khong co thu muc .git o bat ky parent nao). "
        "Chay script nay tu ben trong repo da clone."
    )


PROJECT_ROOT = find_repo_root(Path(__file__).resolve())

DATASET_NAME = "rotten_tomatoes"
BASELINE_MODEL_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
BASE_MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_LABELS = 2
LABEL_NAMES = {0: "negative", 1: "positive"}
RANDOM_SEED = 42

# GIA TRI TAM, chua co can cu tu du lieu that. Se duoc cap nhat lai sau khi
# chay phase_05_dataset_eda.recommend_max_length() va xem ket qua P95/P99/max
# tinh tren token length that (khong phai word count). Xem plan.md muc 18:
# "Do not assume max_length without examining the dataset."
MAX_TOKEN_LENGTH = 128

RESULT_DIR = PROJECT_ROOT / "docs" / "result"

REQUIRED_PACKAGE_VERSIONS = {
    "transformers": "5.14.1",
    "datasets": "5.0.1",
    "evaluate": "0.4.6",
    "accelerate": "1.14.0",
    "torch": "2.13.0+cpu",
    "numpy": "2.4.6",
    "matplotlib": "3.11.1",
    "pandas": "3.0.5",
    "scikit-learn": "1.9.0",
}

PACKAGE_IMPORT_NAME_OVERRIDES = {"scikit-learn": "sklearn"}