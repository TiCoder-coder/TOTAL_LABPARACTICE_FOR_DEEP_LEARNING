# Phase 3 — Tokenization Investigation Log

**Original implementation/fix date:** 2026-08-12  
**Separated from combined Priority 1 log:** 2026-08-14  
**Status:** PASS

## Objective

Inspect how the DistilBERT tokenizer transforms raw text into model inputs:

- readable token table;
- token IDs;
- attention mask;
- special-token identification;
- decode sanity check;
- comparison of fine-tuned and generic-base tokenizers.

## Implementation

Core module:

`processing_own_phase/phase_03_tokenization.py`

The Phase 3 syntax error in the original notebook sentence variable was corrected during Priority 1. The notebook now calls the module functions instead of duplicating tokenization logic:

- `tokenize_sentence_to_table()`;
- `print_tokenized_table()`;
- `decode_sanity_check()`;
- `compare_tokenizers()`.

## Investigated sentence

```text
I absolutely loved this movie! The performances were outstanding.
```

## Latest executed evidence

- Token count including special tokens: 13
- First token/ID: `[CLS]` / 101
- Final token/ID: `[SEP]` / 102
- Token IDs and attention mask lengths: 13 / 13
- Attention-mask values: all 1 because the single sentence requires no padding
- Decoded text: `i absolutely loved this movie! the performances were outstanding.`
- Decode word-overlap ratio: 1.0
- Decode sanity check: PASS

Tokenizer comparison:

| Check | Fine-tuned tokenizer | Base tokenizer |
|---|---:|---:|
| Vocabulary size | 30,522 | 30,522 |
| Token IDs | Identical | Identical |
| Attention mask | Identical | Identical |
| Tokens | Identical | Identical |

`are_identical=True`

This evidence supports the limited conclusion that the compared fine-tuned and base checkpoints use the same tokenizer vocabulary/encoding for the investigated sentence. Fine-tuning changed model weights, not this tokenizer contract.

## Notebook verification

- Phase 3 code-cell execution count in the latest Run All: 4
- Token table displayed: PASS
- Token IDs displayed: PASS
- Attention mask displayed: PASS
- Decode assertion: PASS
- Tokenizer comparison: PASS
- Syntax/import/runtime error: none

## Isolation

Phase 3 performs tokenizer investigation only. It does not train a model, load the Rotten Tomatoes dataset, access Test, select a checkpoint or create evaluation metrics.

Phase 3 evidence is stored in the executed notebook output rather than a separate result JSON.

## Conclusion

`PHASE 3: PASS — TOKENIZATION INVESTIGATION VERIFIED`
