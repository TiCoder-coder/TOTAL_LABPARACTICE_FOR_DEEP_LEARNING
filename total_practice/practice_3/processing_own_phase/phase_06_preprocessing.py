import json
import re
from pathlib import Path
from typing import Dict, Any, List
from datasets import DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding

from .config import BASE_MODEL_CHECKPOINT, MAX_TOKEN_LENGTH, RESULT_DIR


def get_tokenizer(model_name: str = BASE_MODEL_CHECKPOINT) -> AutoTokenizer:
    """
    Load tokenizer from the specified pretrained model.
    Default is distilbert-base-uncased (config.BASE_MODEL_CHECKPOINT).
    """
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_TOKEN_LENGTH
) -> DatasetDict:
    """
    Tokenize the dataset with truncation but without padding.
    Returns a DatasetDict with tokenized splits.
    """
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        result["labels"] = examples["label"]
        return result

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "label"],
    )
    return tokenized_dataset


def get_data_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    """
    Create DataCollatorWithPadding to perform dynamic padding.
    This is a required component for the Trainer in Phase 8.
    """
    return DataCollatorWithPadding(tokenizer=tokenizer)


def check_tokenized_sample(
    tokenized_dataset: DatasetDict,
    split: str = "train",
    idx: int = 0
) -> Dict[str, Any]:
    """
    Check a tokenized sample to ensure it has the required structure:
    - input_ids exists
    - attention_mask exists
    - labels exists and is in {0, 1}
    - len(input_ids) == len(attention_mask)
    """
    sample = tokenized_dataset[split][idx]

    has_input_ids = "input_ids" in sample
    has_attention_mask = "attention_mask" in sample
    has_labels = "labels" in sample
    label_valid = has_labels and sample["labels"] in (0, 1)
    lengths_match = (
        has_input_ids
        and has_attention_mask
        and len(sample["input_ids"]) == len(sample["attention_mask"])
    )

    return {
        "has_input_ids": has_input_ids,
        "has_attention_mask": has_attention_mask,
        "has_labels": has_labels,
        "label_valid": label_valid,
        "lengths_match": lengths_match,
        "all_pass": all(
            [has_input_ids, has_attention_mask, has_labels, label_valid, lengths_match]
        ),
    }


def decode_sanity_check(
    tokenized_dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    original_dataset: DatasetDict,
    split: str = "train",
    n_samples: int = 3,
) -> List[Dict[str, Any]]:
    """
    Decode tokenized samples back to text and compare with original.
    Ensures preprocessing does not lose meaning.
    """
    results = []
    for i in range(min(n_samples, len(tokenized_dataset[split]))):
        original_text = original_dataset[split][i]["text"]
        input_ids = tokenized_dataset[split][i]["input_ids"]
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Normalize punctuation boundaries before comparing. DistilBERT decode
        # may insert/remove spaces around punctuation without changing words.
        original_words = set(re.findall(r"\b\w+\b", original_text.lower()))
        decoded_words = set(re.findall(r"\b\w+\b", decoded_text.lower()))
        overlap_ratio = (
            len(original_words & decoded_words) / len(original_words)
            if original_words
            else 0.0
        )

        results.append({
            "index": i,
            "original_text": original_text,
            "decoded_text": decoded_text,
            "word_overlap_ratio": round(overlap_ratio, 2),
            "pass": overlap_ratio >= 0.8,
        })
    return results


def verify_preprocessing(
    original_dataset: DatasetDict,
    tokenized_dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_TOKEN_LENGTH,
) -> Dict[str, Any]:
    """Verify the complete Phase 6 dataset contract across all splits."""
    split_reports = {}
    for split_name in ("train", "validation", "test"):
        original_count = len(original_dataset[split_name])
        processed_count = len(tokenized_dataset[split_name])
        columns = set(tokenized_dataset[split_name].column_names)
        labels = set(tokenized_dataset[split_name]["labels"])
        lengths = [len(ids) for ids in tokenized_dataset[split_name]["input_ids"]]
        masks = tokenized_dataset[split_name]["attention_mask"]
        ids = tokenized_dataset[split_name]["input_ids"]
        report = {
            "original_count": original_count,
            "processed_count": processed_count,
            "sample_count_preserved": original_count == processed_count,
            "has_input_ids": "input_ids" in columns,
            "has_attention_mask": "attention_mask" in columns,
            "has_labels": "labels" in columns,
            "label_set": sorted(labels),
            "labels_valid": labels.issubset({0, 1}),
            "max_observed_sequence_length": max(lengths),
            "within_max_length": max(lengths) <= max_length,
            "all_id_mask_lengths_match": all(len(a) == len(b) for a, b in zip(ids, masks)),
        }
        report["all_pass"] = all(
            report[key]
            for key in (
                "sample_count_preserved", "has_input_ids", "has_attention_mask",
                "has_labels", "labels_valid", "within_max_length",
                "all_id_mask_lengths_match",
            )
        )
        split_reports[split_name] = report

    sample_checks = {
        split: check_tokenized_sample(tokenized_dataset, split, 0)
        for split in ("train", "validation", "test")
    }
    decode_checks = decode_sanity_check(
        tokenized_dataset, tokenizer, original_dataset, "train", n_samples=3
    )
    all_pass = (
        all(report["all_pass"] for report in split_reports.values())
        and all(check["all_pass"] for check in sample_checks.values())
        and all(check["pass"] for check in decode_checks)
    )
    return {
        "max_length": max_length,
        "splits": split_reports,
        "sample_checks": sample_checks,
        "decode_checks": decode_checks,
        "all_pass": all_pass,
    }


def verify_dynamic_padding(tokenized_dataset: DatasetDict, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Assert real padding behavior using samples with different sequence lengths."""
    lengths = [len(ids) for ids in tokenized_dataset["train"]["input_ids"]]
    shortest_index = min(range(len(lengths)), key=lengths.__getitem__)
    longest_index = max(range(len(lengths)), key=lengths.__getitem__)
    middle_index = next(
        index for index, length in enumerate(lengths)
        if lengths[shortest_index] < length < lengths[longest_index]
    )
    indices = [shortest_index, middle_index, longest_index]
    original_lengths = [lengths[index] for index in indices]
    samples = [tokenized_dataset["train"][index] for index in indices]
    batch = get_data_collator(tokenizer)(samples)
    batch_length = int(batch["input_ids"].shape[1])
    padding_counts = [
        int((batch["attention_mask"][row] == 0).sum().item())
        for row in range(len(indices))
    ]
    expected_padding = [batch_length - length for length in original_lengths]
    shape_compatible = batch["input_ids"].shape == batch["attention_mask"].shape
    shorter_sequences_padded = all(
        padding_counts[i] > 0 for i, length in enumerate(original_lengths)
        if length < batch_length
    )
    mask_matches_padding = padding_counts == expected_padding
    all_pass = shape_compatible and shorter_sequences_padded and mask_matches_padding
    assert all_pass, "Dynamic-padding verification failed"
    return {
        "sample_indices": indices,
        "original_lengths": original_lengths,
        "batch_input_ids_shape": list(batch["input_ids"].shape),
        "batch_attention_mask_shape": list(batch["attention_mask"].shape),
        "padding_counts_from_attention_mask": padding_counts,
        "expected_padding_counts": expected_padding,
        "shape_compatible": shape_compatible,
        "shorter_sequences_padded": shorter_sequences_padded,
        "attention_mask_matches_padding": mask_matches_padding,
        "all_pass": all_pass,
    }


def save_preprocessing_summary(summary: Dict[str, Any], path: Path | None = None) -> Path:
    """Save executed Phase 6 verification evidence as JSON."""
    output_path = path or RESULT_DIR / "phase_06_preprocessing_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def show_tokenized_sample(
    tokenized_dataset: DatasetDict,
    split: str = "train",
    idx: int = 0
) -> None:
    """
    Print a sample from a tokenized dataset split for inspection.
    """
    if split not in tokenized_dataset:
        print(f"Split '{split}' not found. Available: {list(tokenized_dataset.keys())}")
        return

    sample = tokenized_dataset[split][idx]
    print(f"Sample from '{split}' split (index {idx}):")
    print(f"  input_ids length: {len(sample['input_ids'])}")
    print(f"  input_ids (first 20): {sample['input_ids'][:20]}")
    print(f"  attention_mask length: {len(sample['attention_mask'])}")
    print(f"  attention_mask (first 20): {sample['attention_mask'][:20]}")
    print(f"  labels: {sample.get('labels', 'MISSING')}")
    print(f"  Contains padding? {any(mask == 0 for mask in sample['attention_mask'])}")


if __name__ == "__main__":
    from phase_04_dataset_loading import load_rotten_tomatoes

    dataset = load_rotten_tomatoes()
    tokenizer = get_tokenizer()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    show_tokenized_sample(tokenized_dataset, "train", 0)

    print("\nTokenized dataset splits:")
    for split_name, split_data in tokenized_dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")

    print("\nColumns in tokenized dataset:")
    print(f"  {list(tokenized_dataset['train'].column_names)}")

    print("\n=== Sanity Check: Tokenized Sample Structure ===")
    check_result = check_tokenized_sample(tokenized_dataset, "train", 0)
    for key, value in check_result.items():
        print(f"  {key}: {value}")
    assert check_result["all_pass"], "Sanity check FAILED"

    print("\n=== Decode Sanity Check ===")
    decode_results = decode_sanity_check(tokenized_dataset, tokenizer, dataset, "train", n_samples=3)
    for result in decode_results:
        print(f"\n  [{result['index']}] Original: {result['original_text'][:80]}")
        print(f"      Decoded : {result['decoded_text'][:80]}")
        print(f"      Word overlap ratio: {result['word_overlap_ratio']}")

    print("\n=== Data Collator (Dynamic Padding) ===")
    collator = get_data_collator(tokenizer)
    batch_sample = [tokenized_dataset["train"][i] for i in range(4)]
    batch = collator(batch_sample)
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  Batch labels: {batch['labels'].tolist()}")
