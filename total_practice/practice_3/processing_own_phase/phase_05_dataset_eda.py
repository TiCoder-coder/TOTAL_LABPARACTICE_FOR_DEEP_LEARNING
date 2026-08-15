import json
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datasets import DatasetDict
from transformers import AutoTokenizer

from .config import BASE_MODEL_CHECKPOINT, RESULT_DIR


def check_schema(dataset: DatasetDict) -> Dict[str, bool]:
    """Check if the dataset has the two required columns: text and label."""
    required_columns = {"text", "label"}
    report = {}
    for split_name, split_data in dataset.items():
        actual_columns = set(split_data.column_names)
        report[split_name] = required_columns.issubset(actual_columns)
    return report


def check_text_quality(dataset: DatasetDict) -> Dict[str, Dict[str, int]]:
    """Count null, empty and whitespace-only text separately for every split."""
    report = {}
    for split_name, split_data in dataset.items():
        texts = split_data["text"]
        report[split_name] = {
            "null_text": sum(text is None for text in texts),
            "empty_string": sum(text == "" for text in texts if text is not None),
            "whitespace_only": sum(
                isinstance(text, str) and text != "" and text.strip() == ""
                for text in texts
            ),
        }
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


def compute_all_split_text_lengths(
    dataset: DatasetDict, tokenizer
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute character, word and tokenizer-length statistics for every split."""
    return {
        split_name: compute_text_lengths(dataset, tokenizer, split_name)
        for split_name in ("train", "validation", "test")
    }


def _sample_length_records(dataset: DatasetDict, tokenizer, split: str) -> list:
    """Return deterministic, index-addressable length records for one split."""
    records = []
    for index, sample in enumerate(dataset[split]):
        text = sample["text"]
        records.append({
            "split": split,
            "index": index,
            "label": int(sample["label"]),
            "label_name": "Positive" if sample["label"] == 1 else "Negative",
            "character_count": len(text),
            "word_count": len(text.split()),
            "token_count": len(tokenizer(text, truncation=False)["input_ids"]),
            "text": text,
        })
    return records


def get_representative_samples(
    dataset: DatasetDict,
    tokenizer,
    split: str = "train",
    samples_per_label: int = 3,
) -> list:
    """Select the first N indexed samples per label; no quality-based cherry-picking."""
    if samples_per_label <= 0:
        raise ValueError("samples_per_label must be positive")
    records = _sample_length_records(dataset, tokenizer, split)
    selected = []
    for label in (0, 1):
        selected.extend([
            record for record in records if record["label"] == label
        ][:samples_per_label])
    return sorted(selected, key=lambda record: (record["label"], record["index"]))


def get_extreme_samples(
    dataset: DatasetDict,
    tokenizer,
    split: str = "train",
    samples_per_extreme: int = 5,
) -> Dict[str, list]:
    """Return shortest/longest reviews by token count with stable index ties."""
    if samples_per_extreme <= 0:
        raise ValueError("samples_per_extreme must be positive")
    records = _sample_length_records(dataset, tokenizer, split)
    shortest = sorted(records, key=lambda row: (row["token_count"], row["index"]))[
        :samples_per_extreme
    ]
    longest = sorted(
        records, key=lambda row: (-row["token_count"], row["index"])
    )[:samples_per_extreme]
    return {"shortest": shortest, "longest": longest}


def compute_truncation_impact(
    dataset: DatasetDict, tokenizer, max_length: int
) -> Dict[str, Dict[str, float]]:
    """Count sequences that would be truncated at the configured maximum."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    report = {}
    for split_name in ("train", "validation", "test"):
        lengths = [
            len(tokenizer(text, truncation=False)["input_ids"])
            for text in dataset[split_name]["text"]
        ]
        truncated = sum(length > max_length for length in lengths)
        report[split_name] = {
            "samples": len(lengths),
            "samples_over_max_length": truncated,
            "percentage_over_max_length": 100.0 * truncated / len(lengths),
            "observed_max": int(max(lengths)),
        }
    return report


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


def check_cross_split_overlap(dataset: DatasetDict) -> Dict[str, int]:
    """Count exact text overlap for every pair of official splits."""
    text_sets = {name: set(split["text"]) for name, split in dataset.items()}
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    return {
        f"{left}<->{right}": len(text_sets[left] & text_sets[right])
        for left, right in pairs
    }


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
        save_path = RESULT_DIR / "phase_05_token_length_distribution.png"

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


def plot_label_distribution(
    dataset: DatasetDict,
    save_path: Optional[str] = None,
) -> Path:
    """Save per-split Negative/Positive counts as a grouped bar chart."""
    output_path = Path(save_path or RESULT_DIR / "phase_05_label_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    balance = get_label_balance(dataset)
    splits = ["train", "validation", "test"]
    negative = [balance[split]["negative"] for split in splits]
    positive = [balance[split]["positive"] for split in splits]
    x = np.arange(len(splits))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(x - width / 2, negative, width, label="Negative (0)", color="#d95f5f")
    axis.bar(x + width / 2, positive, width, label="Positive (1)", color="#4c9f70")
    axis.set_xticks(x, [name.title() for name in splits])
    axis.set(xlabel="Official split", ylabel="Sample count", title="Phase 5 — Label Distribution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_length_distribution(
    dataset: DatasetDict,
    measure: str,
    save_path: Path,
) -> Path:
    """Plot comparable per-split character or word distributions."""
    if measure not in {"character", "word"}:
        raise ValueError("measure must be 'character' or 'word'")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    for split_name in ("train", "validation", "test"):
        texts = dataset[split_name]["text"]
        values = (
            [len(text) for text in texts]
            if measure == "character"
            else [len(text.split()) for text in texts]
        )
        axis.hist(values, bins=45, density=True, histtype="step", linewidth=1.8,
                  label=split_name.title())
    label = "Characters" if measure == "character" else "Words"
    axis.set(xlabel=f"{label} per review", ylabel="Density",
             title=f"Phase 5 — {label} Length Distribution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return save_path


def plot_character_length_distribution(
    dataset: DatasetDict, save_path: Optional[str] = None
) -> Path:
    return _plot_length_distribution(
        dataset,
        "character",
        Path(save_path or RESULT_DIR / "phase_05_character_length_distribution.png"),
    )


def plot_word_length_distribution(
    dataset: DatasetDict, save_path: Optional[str] = None
) -> Path:
    return _plot_length_distribution(
        dataset,
        "word",
        Path(save_path or RESULT_DIR / "phase_05_word_length_distribution.png"),
    )


def create_eda_figures(
    dataset: DatasetDict,
    tokenizer,
    max_length: int,
) -> Dict[str, str]:
    """Create the concise Phase 5 figure set and return artifact paths."""
    paths = {
        "label_distribution": plot_label_distribution(dataset),
        "character_length_distribution": plot_character_length_distribution(dataset),
        "word_length_distribution": plot_word_length_distribution(dataset),
        "token_length_distribution": RESULT_DIR / "phase_05_token_length_distribution.png",
    }
    plot_token_length_distribution(
        dataset,
        tokenizer,
        split="train",
        max_length_reference=max_length,
        save_path=str(paths["token_length_distribution"]),
    )
    return {name: str(path) for name, path in paths.items()}


def get_label_balance(dataset: DatasetDict) -> Dict[str, Dict[str, Any]]:
    """
    Get label balance (positive/negative counts and percentages) for each split.
    """
    balance = {}

    for split_name, split_data in dataset.items():
        labels = split_data["label"]
        total = len(labels)
        label_set = set(labels)
        if not label_set.issubset({0, 1}):
            raise ValueError(f"Invalid labels in {split_name}: {sorted(label_set)}")
        positive = sum(1 for label in labels if label == 1)
        negative = sum(1 for label in labels if label == 0)

        balance[split_name] = {
            "total": total,
            "positive": positive,
            "negative": negative,
            "positive_pct": positive / total * 100 if total > 0 else 0,
            "negative_pct": negative / total * 100 if total > 0 else 0,
        }

    return balance


def build_eda_summary(
    dataset: DatasetDict, tokenizer, max_length: Optional[int] = None
) -> Dict[str, Any]:
    """Run the complete task-focused EDA required before preprocessing."""
    schema = check_schema(dataset)
    text_quality = check_text_quality(dataset)
    within_split_duplicates = check_duplicates(dataset)
    cross_split_overlap = check_cross_split_overlap(dataset)
    label_distribution = get_label_balance(dataset)
    all_split_lengths = compute_all_split_text_lengths(dataset, tokenizer)
    train_lengths = all_split_lengths["train"]
    token_recommendation = recommend_max_length(dataset, tokenizer, "train")
    representative_samples = get_representative_samples(dataset, tokenizer)
    extreme_samples = get_extreme_samples(dataset, tokenizer)
    truncation_impact = (
        compute_truncation_impact(dataset, tokenizer, max_length)
        if max_length is not None
        else None
    )
    all_pass = (
        all(schema.values())
        and all(sum(counts.values()) == 0 for counts in text_quality.values())
        and all(set(split["label"]).issubset({0, 1}) for split in dataset.values())
    )
    return {
        "schema": schema,
        "text_quality": text_quality,
        "label_distribution": label_distribution,
        "within_split_duplicates": within_split_duplicates,
        "cross_split_overlap": cross_split_overlap,
        "train_length_statistics": train_lengths,
        "all_split_length_statistics": all_split_lengths,
        "token_length_recommendation": token_recommendation,
        "representative_samples": representative_samples,
        "extreme_samples": extreme_samples,
        "truncation_impact": truncation_impact,
        "all_pass": all_pass,
    }


def save_eda_summary(summary: Dict[str, Any], path: Path | None = None) -> Path:
    """Save executed EDA evidence as JSON."""
    output_path = path or RESULT_DIR / "phase_05_eda_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def save_token_length_statistics(
    statistics: Dict[str, Any], path: Path | None = None
) -> Path:
    """Save executed token-length evidence as a focused JSON artifact."""
    output_path = path or RESULT_DIR / "phase_05_token_length_statistics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    return output_path


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
