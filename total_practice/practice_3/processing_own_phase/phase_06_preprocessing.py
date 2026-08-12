from typing import Dict, Any, List
from datasets import DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import BASE_MODEL_CHECKPOINT, MAX_TOKEN_LENGTH


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

        original_words = set(original_text.lower().split())
        decoded_words = set(decoded_text.lower().split())
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
        })
    return results


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