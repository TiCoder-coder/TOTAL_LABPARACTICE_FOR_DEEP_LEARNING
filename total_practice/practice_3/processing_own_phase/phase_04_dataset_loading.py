import json
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset, DatasetDict

from .config import DATASET_NAME, RESULT_DIR


EXPECTED_SPLIT_SIZES = {"train": 8530, "validation": 1066, "test": 1066}


def load_rotten_tomatoes() -> DatasetDict:
    """
    Load Rotten Tomatoes dataset from Hugging Face.
    Returns a DatasetDict with train, validation, and test splits.
    """
    return load_dataset(DATASET_NAME)


def verify_split_sizes(dataset: DatasetDict) -> Dict[str, int]:
    """
    Verify that each split has the expected number of samples (8530/1066/1066).
    Raises ValueError if any split size mismatches.
    Returns dict split_name -> actual_size if all match.
    """
    actual_sizes = {}
    mismatches = []

    for split_name, expected_size in EXPECTED_SPLIT_SIZES.items():
        if split_name not in dataset:
            mismatches.append(f"{split_name}: split does not exist")
            continue
        actual_size = len(dataset[split_name])
        actual_sizes[split_name] = actual_size
        if actual_size != expected_size:
            mismatches.append(
                f"{split_name}: has {actual_size} samples, expected {expected_size}"
            )

    if mismatches:
        raise ValueError(
            "Split size mismatch: " + "; ".join(mismatches)
        )

    return actual_sizes


def verify_dataset_contract(dataset: DatasetDict) -> Dict[str, Any]:
    """Verify required splits, sizes, schema and every label in every split."""
    required_splits = set(EXPECTED_SPLIT_SIZES)
    actual_splits = set(dataset.keys())
    splits_exist = required_splits.issubset(actual_splits)
    if not splits_exist:
        raise AssertionError(f"Missing splits: {sorted(required_splits - actual_splits)}")

    split_sizes = verify_split_sizes(dataset)
    schema = {}
    label_sets = {}
    for split_name in EXPECTED_SPLIT_SIZES:
        columns = set(dataset[split_name].column_names)
        schema[split_name] = {
            "columns": sorted(columns),
            "has_text": "text" in columns,
            "has_label": "label" in columns,
        }
        if not (schema[split_name]["has_text"] and schema[split_name]["has_label"]):
            raise AssertionError(f"Invalid schema for {split_name}: {sorted(columns)}")
        labels = set(dataset[split_name]["label"])
        label_sets[split_name] = sorted(labels)
        if not labels.issubset({0, 1}):
            raise AssertionError(f"Invalid labels in {split_name}: {sorted(labels)}")

    return {
        "required_splits": sorted(required_splits),
        "actual_splits": sorted(actual_splits),
        "splits_exist": splits_exist,
        "split_sizes": split_sizes,
        "schema": schema,
        "label_sets": label_sets,
        "all_pass": True,
    }


def save_dataset_summary(summary: Dict[str, Any], path: Path | None = None) -> Path:
    """Save an executed Phase 4 verification summary as JSON."""
    output_path = path or RESULT_DIR / "phase_04_dataset_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def get_dataset_stats(dataset: DatasetDict) -> Dict[str, Any]:
    """
    Get statistics for the dataset.
    Returns a dict with sample counts per split and label distribution.
    """
    stats = {}

    for split_name, split_data in dataset.items():
        n_samples = len(split_data)
        labels = split_data["label"]

        label_set = set(labels)
        if not label_set.issubset({0, 1}):
            raise ValueError(f"Invalid labels in {split_name}: {sorted(label_set)}")
        n_positive = sum(1 for label in labels if label == 1)
        n_negative = sum(1 for label in labels if label == 0)

        stats[split_name] = {
            "total": n_samples,
            "positive": n_positive,
            "negative": n_negative,
            "positive_pct": n_positive / n_samples * 100 if n_samples > 0 else 0,
            "negative_pct": n_negative / n_samples * 100 if n_samples > 0 else 0,
        }

    return stats


def check_null_values(dataset: DatasetDict) -> Dict[str, Dict[str, int]]:
    """
    Check for null values in the dataset.
    Returns a dict with split name and count of null values per column.
    """
    null_report = {}

    for split_name, split_data in dataset.items():
        null_counts = {}

        for column in split_data.column_names:
            null_count = 0
            for item in split_data[column]:
                if item is None or (isinstance(item, str) and item.strip() == ""):
                    null_count += 1
            null_counts[column] = null_count

        null_report[split_name] = null_counts

    return null_report


def print_sample_rows(dataset: DatasetDict, split: str = "train", n: int = 3) -> None:
    """
    Print the first n rows of a specified split.
    """
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found. Available: {list(dataset.keys())}")

    print(f"\n--- First {n} rows of '{split}' split ---")
    for i in range(min(n, len(dataset[split]))):
        row = dataset[split][i]
        label_name = "POSITIVE" if row["label"] == 1 else "NEGATIVE"
        text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
        print(f"\nRow {i+1}:")
        print(f"  Text: {text_preview}")
        print(f"  Label: {row['label']} ({label_name})")


if __name__ == "__main__":
    print("Loading Rotten Tomatoes dataset...")
    dataset = load_rotten_tomatoes()

    print("\n--- Dataset Structure ---")
    print(f"Splits: {list(dataset.keys())}")
    print(f"Train: {len(dataset['train'])} samples")
    print(f"Validation: {len(dataset['validation'])} samples")
    print(f"Test: {len(dataset['test'])} samples")

    print("\n--- Verify Split Sizes ---")
    actual_sizes = verify_split_sizes(dataset)
    print(f"OK - split sizes match expected: {actual_sizes}")

    print("\n--- Dataset Statistics ---")
    stats = get_dataset_stats(dataset)
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.capitalize()}:")
        print(f"  Total: {split_stats['total']}")
        print(f"  Positive: {split_stats['positive']} ({split_stats['positive_pct']:.1f}%)")
        print(f"  Negative: {split_stats['negative']} ({split_stats['negative_pct']:.1f}%)")

    print("\n--- Null Values ---")
    null_report = check_null_values(dataset)
    for split_name, null_counts in null_report.items():
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {split_name}.{col}: {count} null values")
            else:
                print(f"  {split_name}.{col}: No null values")

    print_sample_rows(dataset, "train", 3)
    print_sample_rows(dataset, "validation", 2)
    print_sample_rows(dataset, "test", 2)
