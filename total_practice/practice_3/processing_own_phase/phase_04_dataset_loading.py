from typing import Dict, Any
from datasets import load_dataset, DatasetDict


EXPECTED_SPLIT_SIZES = {"train": 8530, "validation": 1066, "test": 1066}


def load_rotten_tomatoes() -> DatasetDict:
    """
    Load Rotten Tomatoes dataset from Hugging Face.
    Returns a DatasetDict with train, validation, and test splits.
    """
    return load_dataset("rotten_tomatoes")


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


def get_dataset_stats(dataset: DatasetDict) -> Dict[str, Any]:
    """
    Get statistics for the dataset.
    Returns a dict with sample counts per split and label distribution.
    """
    stats = {}

    for split_name, split_data in dataset.items():
        n_samples = len(split_data)
        labels = split_data["label"]

        n_positive = sum(1 for l in labels if l == 1)
        n_negative = len(labels) - n_positive

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