from typing import Dict, Any, List
from transformers import AutoTokenizer

from .config import BASELINE_MODEL_CHECKPOINT, BASE_MODEL_CHECKPOINT


def tokenize_sentence_to_table(
    sentence: str,
    checkpoint: str = BASELINE_MODEL_CHECKPOINT,
) -> List[Dict[str, Any]]:
    """
    Tokenize a sentence and return as a list of dicts (each dict = one row of a readable table),
    containing: Position, Token, Token ID, Special token?.

    Required by plan.md section 15: "At least one tokenized sentence should be presented in a readable table."
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoding = tokenizer(sentence, return_tensors=None, truncation=True)

    input_ids = encoding["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special_ids = set(tokenizer.all_special_ids)

    table = []
    for position, (token, token_id) in enumerate(zip(tokens, input_ids)):
        table.append({
            "position": position,
            "token": token,
            "token_id": token_id,
            "is_special": token_id in special_ids,
        })
    return table


def print_tokenized_table(table: List[Dict[str, Any]]) -> None:
    """Print the tokenized table in a formatted, aligned manner."""
    header = f"{'Position':<10}{'Token':<20}{'Token ID':<12}{'Special?':<10}"
    print(header)
    print("-" * len(header))
    for row in table:
        special_flag = "Yes" if row["is_special"] else "No"
        print(f"{row['position']:<10}{row['token']:<20}{row['token_id']:<12}{special_flag:<10}")


def decode_sanity_check(
    sentence: str,
    checkpoint: str = BASELINE_MODEL_CHECKPOINT,
) -> Dict[str, Any]:
    """
    Required sanity check per plan.md section 15:
    "tokenizer.decode(input_ids). The decoded output should remain semantically consistent with the original input."

    Tokenizes -> decodes -> compares with original, returns word_overlap_ratio and semantically_consistent flag.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoding = tokenizer(sentence, return_tensors=None, truncation=True)
    input_ids = encoding["input_ids"]

    decoded_sentence = tokenizer.decode(input_ids, skip_special_tokens=True)

    original_words = set(sentence.lower().split())
    decoded_words = set(decoded_sentence.lower().split())
    overlap_ratio = (
        len(original_words & decoded_words) / len(original_words)
        if original_words
        else 0.0
    )

    return {
        "original_sentence": sentence,
        "decoded_sentence": decoded_sentence,
        "word_overlap_ratio": round(overlap_ratio, 2),
        "semantically_consistent": overlap_ratio >= 0.8,
    }


def compare_tokenizers(sentence: str) -> Dict[str, Any]:
    """
    Compare tokenizers from the fine-tuned model and the base model.
    Returns a dict containing tokenization results from both tokenizers
    and a boolean indicating whether they are identical.

    Demonstrates that fine-tuning only updates model weights, not the tokenizer.
    """
    tokenizer_finetuned = AutoTokenizer.from_pretrained(BASELINE_MODEL_CHECKPOINT)
    tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT)

    encoding_finetuned = tokenizer_finetuned(
        sentence, return_tensors=None, truncation=True, padding=True
    )
    encoding_base = tokenizer_base(
        sentence, return_tensors=None, truncation=True, padding=True
    )

    tokens_finetuned = tokenizer_finetuned.convert_ids_to_tokens(
        encoding_finetuned["input_ids"]
    )
    tokens_base = tokenizer_base.convert_ids_to_tokens(
        encoding_base["input_ids"]
    )

    are_identical = (
        tokenizer_finetuned.vocab_size == tokenizer_base.vocab_size
        and encoding_finetuned["input_ids"] == encoding_base["input_ids"]
        and encoding_finetuned["attention_mask"] == encoding_base["attention_mask"]
        and tokens_finetuned == tokens_base
    )

    return {
        "sentence": sentence,
        "are_identical": are_identical,
        "finetuned": {
            "vocab_size": tokenizer_finetuned.vocab_size,
            "input_ids": encoding_finetuned["input_ids"],
            "attention_mask": encoding_finetuned["attention_mask"],
            "tokens": tokens_finetuned,
        },
        "base": {
            "vocab_size": tokenizer_base.vocab_size,
            "input_ids": encoding_base["input_ids"],
            "attention_mask": encoding_base["attention_mask"],
            "tokens": tokens_base,
        }
    }


if __name__ == "__main__":
    test_sentence = (
        "I absolutely loved this movie! The performances were outstanding."
    )

    print("=" * 70)
    print("PHASE 3 - TOKENIZATION INVESTIGATION")
    print("=" * 70)

    print(f"\nSentence: {test_sentence}\n")

    print("--- Readable Token Table ---")
    table = tokenize_sentence_to_table(test_sentence)
    print_tokenized_table(table)

    print("\n--- Decode Sanity Check ---")
    decode_result = decode_sanity_check(test_sentence)
    print(f"  Original : {decode_result['original_sentence']}")
    print(f"  Decoded  : {decode_result['decoded_sentence']}")
    print(f"  Word overlap ratio: {decode_result['word_overlap_ratio']}")
    print(f"  Semantically consistent: {decode_result['semantically_consistent']}")
    assert decode_result["semantically_consistent"], (
        "Decode sanity check FAILED - decoded sentence lost meaning"
    )

    print("\n--- Tokenizer Comparison ---")
    result = compare_tokenizers(test_sentence)
    print(f"Are tokenizers identical: {result['are_identical']}")
    print(f"  Fine-tuned vocab_size: {result['finetuned']['vocab_size']}")
    print(f"  Base vocab_size:       {result['base']['vocab_size']}")

    if result['are_identical']:
        print("\nCONCLUSION: Tokenizers are identical.")
        print("  -> Fine-tuning only updates model weights, not the tokenizer.")
    else:
        print("\nCONCLUSION: Tokenizers are different (unexpected).")
    print("=" * 70)
