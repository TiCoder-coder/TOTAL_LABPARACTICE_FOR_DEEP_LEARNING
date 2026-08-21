"""Correct DistilBERT model/tokenizer separation and tokenizer safeguards.

The model cache identifier is namespaced, while the tokenizer must use the
canonical identifier that contains the complete 30,522-token vocabulary.
This module performs no training and writes no artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from datasets import DatasetDict


MODEL_IDENTIFIER = "distilbert/distilbert-base-uncased"
TOKENIZER_IDENTIFIER = "distilbert-base-uncased"
EXPECTED_VOCAB_SIZE = 30_522
MAX_LENGTH = 80
KNOWN_WORDS = ("this", "movie", "good", "bad", "positive", "negative")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one required JSON object without mutating it."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {source}")
    return value


def validate_tokenizer_integrity(tokenizer: Any, sample_texts: list[str]) -> dict[str, Any]:
    """Fail closed on the historical five-token/all-UNK tokenizer failure."""
    if not sample_texts or any(not isinstance(text, str) or not text.strip() for text in sample_texts):
        raise ValueError("Tokenizer validation requires non-empty text samples")
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    special_names = (
        "pad_token_id", "unk_token_id", "cls_token_id", "sep_token_id", "mask_token_id"
    )
    special_ids = {name: getattr(tokenizer, name, None) for name in special_names}
    special_tokens_present = all(isinstance(value, int) for value in special_ids.values())
    special_ids_in_range = special_tokens_present and all(0 <= value < vocab_size for value in special_ids.values())
    unk_id = special_ids["unk_token_id"]

    known_checks: dict[str, Any] = {}
    known_words_ok = True
    for word in KNOWN_WORDS:
        tokens = tokenizer.tokenize(word)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        all_unknown = not ids or all(item == unk_id for item in ids)
        known_words_ok = known_words_ok and not all_unknown
        known_checks[word] = {"tokens": tokens, "ids": ids, "all_unknown": all_unknown}

    encoded = tokenizer(sample_texts, truncation=True, padding=False, max_length=MAX_LENGTH)
    sequences = encoded.get("input_ids", []) if isinstance(encoded, Mapping) else []
    lexical_ids = [
        token_id for sequence in sequences for token_id in sequence
        if token_id not in set(getattr(tokenizer, "all_special_ids", []))
    ]
    all_ids = [token_id for sequence in sequences for token_id in sequence]
    unk_count = sum(token_id == unk_id for token_id in lexical_ids)
    unk_ratio = unk_count / len(lexical_ids) if lexical_ids else 1.0
    unique_encodings = len({tuple(sequence) for sequence in sequences})
    checks = {
        "vocab_size_exact": vocab_size == EXPECTED_VOCAB_SIZE,
        "known_words_not_unknown": known_words_ok,
        "special_tokens_present": special_tokens_present,
        "special_ids_in_range": special_ids_in_range,
        "all_ids_in_range": bool(all_ids) and all(isinstance(item, int) and 0 <= item < vocab_size for item in all_ids),
        "unk_ratio_acceptable": unk_ratio <= 0.05,
        "distinct_encodings": unique_encodings > 1,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Tokenizer integrity failed: {', '.join(failed)}")
    return {
        "tokenizer_identifier": TOKENIZER_IDENTIFIER,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": vocab_size,
        "special_token_ids": special_ids,
        "known_token_checks": known_checks,
        "sample_count": len(sample_texts),
        "unk_count": unk_count,
        "lexical_token_count": len(lexical_ids),
        "unk_ratio": unk_ratio,
        "unique_text_count": len(set(sample_texts)),
        "unique_encoded_sequence_count": unique_encodings,
        "checks": checks,
        "tokenizer_integrity": "PASS",
        "raw_sample_text_stored": False,
    }


def tokenize_splits(datasets: Mapping[str, Any], tokenizer: Any) -> DatasetDict:
    """Tokenize only supplied Train/Validation splits using the locked schema."""
    if set(datasets) != {"train", "validation"}:
        raise ValueError("Expected exactly Train and Validation splits")
    source = DatasetDict(dict(datasets))
    for name in ("train", "validation"):
        columns = set(source[name].column_names)
        if not {"text", "label"}.issubset(columns):
            raise ValueError(f"{name} split must contain text and label columns")

    def encode(batch: dict[str, Any]) -> dict[str, Any]:
        result = tokenizer(
            batch["text"], truncation=True, padding=False, max_length=MAX_LENGTH
        )
        result["labels"] = batch["label"]
        return result

    return source.map(
        encode,
        batched=True,
        remove_columns=list(source["train"].column_names),
        keep_in_memory=True,
        desc="Tokenizing v2.3 Train/Validation only",
    )


__all__ = (
    "MODEL_IDENTIFIER",
    "TOKENIZER_IDENTIFIER",
    "load_json",
    "tokenize_splits",
    "validate_tokenizer_integrity",
)
