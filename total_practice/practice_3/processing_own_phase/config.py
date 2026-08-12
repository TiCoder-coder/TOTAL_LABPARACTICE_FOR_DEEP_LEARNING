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


REPO_ROOT = find_repo_root(Path(__file__).resolve())
# Practice 3 root: total_practice/practice_3/.  Result and workflow paths
# must remain scoped to this practice instead of the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"
BASELINE_MODEL_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
BASE_MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_LABELS = 2
LABEL_NAMES = {0: "negative", 1: "positive"}
RANDOM_SEED = 42

# Phase 5 executed EDA (2026-08-12): train token lengths P95=47, P99=56,
# maximum=78 (including special tokens). 80 is the smallest convenient rounded
# value above the observed maximum, so it preserves every observed train sample.
MAX_TOKEN_LENGTH = 80

RESULT_DIR = PROJECT_ROOT / "docs" / "result"

REQUIRED_PACKAGE_VERSIONS = {
    "transformers": "5.14.1",
    "datasets": "5.0.1",
    "evaluate": "0.4.6",
    "accelerate": "1.14.0",
    "torch": "2.13.0",
    "numpy": "2.4.6",
    "matplotlib": "3.11.1",
    "pandas": "3.0.5",
    "scikit-learn": "1.9.0",
}

PACKAGE_IMPORT_NAME_OVERRIDES = {"scikit-learn": "sklearn"}
