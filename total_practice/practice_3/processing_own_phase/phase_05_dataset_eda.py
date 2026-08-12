from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datasets import DatasetDict
from transformers import AutoTokenizer

from config import BASE_MODEL_CHECKPOINT


def check_schema(dataset: DatasetDict) -> Dict[str, bool]:
    """Check if the dataset has the two required columns: text and label."""
    required_columns = {"text", "label"}
    report = {}
    for split_name, split_data in dataset.items():
        actual_columns = set(split_data.column_names)
        report[split_name] = required_columns.issubset(actual_columns)
    return report


def compute_text_lengths(dataset: DatasetDict, tokenizer, split: str = "train") -> Dict[str, Dict[str, float]]:
    """Compute text lengths in three ways: character, word, and token (using the actual tokenizer)."""
    texts = dataset[split]["text"]

    char_lengths = [len(text) for text in texts]
    word_lengths = [len(text.split()) for text in texts]
    token_lengths = [
        len(tokenizer(text, truncation=False)["input_ids"]) for text in texts
    ]

    def _stats(values):
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }

    return {
        "character": _stats(char_lengths),
        "word": _stats(word_lengths),
        "token": _stats(token_lengths),
    }


def recommend_max_length(dataset: DatasetDict, tokenizer, split: str = "train") -> Dict[str, float]:
    """
    Compute P95/P99/max of actual token lengths (without padding) to choose max_length
    based on real data rather than guessing.
    """
    texts = dataset[split]["text"]
    token_lengths = [
        len(tokenizer(text, truncation=False)["input_ids"]) for text in texts
    ]
    return {
        "p95": float(np.percentile(token_lengths, 95)),
        "p99": float(np.percentile(token_lengths, 99)),
        "max": float(np.max(token_lengths)),
    }


def check_duplicates(dataset: DatasetDict) -> Dict[str, int]:
    """Check the number of duplicate texts (exact duplicates) in each split."""
    report = {}
    for split_name, split_data in dataset.items():
        texts = split_data["text"]
        report[split_name] = len(texts) - len(set(texts))
    return report


def plot_token_length_distribution(
    dataset: DatasetDict,
    tokenizer,
    split: str = "train",
    max_length_reference: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a histogram of TOKEN lengths (not words) for a given split.
    max_length_reference: actual value (e.g., P99 from recommend_max_length)
    to draw a reference line instead of hard-coding 128.
    """
    if save_path is None:
        save_path = "docs/result/phase_05_token_length_distribution.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    texts = dataset[split]["text"]
    token_lengths = [
        len(tokenizer(text, truncation=False)["input_ids"]) for text in texts
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, edgecolor="black", alpha=0.7)
    if max_length_reference is not None:
        plt.axvline(
            x=max_length_reference,
            color="red",
            linestyle="--",
            label=f"recommended max_length = {max_length_reference:.0f}",
        )
        plt.legend()
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title(f"Token Length Distribution - {split} split")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Histogram saved to: {save_path}")


def get_label_balance(dataset: DatasetDict) -> Dict[str, Dict[str, Any]]:
    """
    Get label balance (positive/negative counts and percentages) for each split.
    """
    balance = {}

    for split_name, split_data in dataset.items():
        labels = split_data["label"]
        total = len(labels)
        positive = sum(1 for l in labels if l == 1)
        negative = total - positive

        balance[split_name] = {
            "total": total,
            "positive": positive,
            "negative": negative,
            "positive_pct": positive / total * 100 if total > 0 else 0,
            "negative_pct": negative / total * 100 if total > 0 else 0,
        }

    return balance


if __name__ == "__main__":
    from phase_04_dataset_loading import load_rotten_tomatoes

    dataset = load_rotten_tomatoes()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT)

    print("=== Schema Check ===")
    schema_report = check_schema(dataset)
    for split, ok in schema_report.items():
        print(f"  {split}: {'OK' if ok else 'MISSING COLUMNS'}")

    print("\n=== Text Length Statistics (Train split) ===")
    length_stats = compute_text_lengths(dataset, tokenizer, "train")
    for length_type, stats in length_stats.items():
        print(f"  [{length_type}]")
        for key, value in stats.items():
            print(f"    {key}: {value:.2f}")

    print("\n=== Recommended max_length (based on real token distribution) ===")
    recommendation = recommend_max_length(dataset, tokenizer, "train")
    for key, value in recommendation.items():
        print(f"  {key}: {value:.2f}")

    print("\n=== Duplicate Check ===")
    duplicate_report = check_duplicates(dataset)
    for split, count in duplicate_report.items():
        print(f"  {split}: {count} duplicate(s)")

    print("\n=== Label Balance ===")
    balance = get_label_balance(dataset)
    for split, data in balance.items():
        print(f"  {split}:")
        print(f"    Total: {data['total']}")
        print(f"    Positive: {data['positive']} ({data['positive_pct']:.1f}%)")
        print(f"    Negative: {data['negative']} ({data['negative_pct']:.1f}%)")

    plot_token_length_distribution(
        dataset, tokenizer, "train", max_length_reference=recommendation["p99"]
    )