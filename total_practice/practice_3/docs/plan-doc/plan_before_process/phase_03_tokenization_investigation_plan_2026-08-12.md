# Phase 3 Tokenization Investigation Plan

**Date:** 2026-08-11
**Phase:** 3/15  
**File path:** `docs/plan-doc/plan_before_process/phase_03_tokenization_investigation_plan_2026-08-12.md`  
**Dependencies:** Phase 2 (Pretrained Sentiment Inference)

---

## 1. Objective

Compare the tokenizer from the fine-tuned model (Exercise 1) with the tokenizer from the base model (Exercise 2) to demonstrate that:

- Tokenizers are **identical**.
- Fine-tuning only updates model weights, not the tokenizer.
- The base tokenizer can be used consistently for all phases.

---

## 2. Input

| Source           | Content                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------- |
| Fine-tuned Model | `distilbert-base-uncased-finetuned-sst-2-english`                                       |
| Base Model       | `distilbert-base-uncased`                                                               |
| Sample Sentence  | `"I absolutely loved this movie! The performances were outstanding."` (same as Phase 2) |

---

## 3. Processing

### 3.1 Load Tokenizers

- Load tokenizer from fine-tuned checkpoint.
- Load tokenizer from base checkpoint.
- Both are `DistilBertTokenizerFast`.

### 3.2 Tokenize and Compare

- Tokenize the same sentence with both tokenizers.
- Compare:
  - `vocab_size` (should be 30522)
  - `input_ids` (should be identical)
  - `tokens` (subword list, should be identical)
  - `attention_mask` (should be identical)

### 3.3 Conclusion

- Document that tokenizers are identical.
- Confirm that the base tokenizer (`distilbert-base-uncased`) will be used for Phases 4–6.

---

## 4. Module Functions (4 functions)

| Function                                                                                               | Purpose                                                                           |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `tokenize_sentence_to_table(sentence: str, checkpoint: str = BASELINE_MODEL_CHECKPOINT) -> List[Dict]` | Tokenize sentence and return as a readable table.                                 |
| `print_tokenized_table(table: List[Dict]) -> None`                                                     | Print the token table in aligned format.                                          |
| `decode_sanity_check(sentence: str, checkpoint: str = BASELINE_MODEL_CHECKPOINT) -> Dict`              | Tokenize → decode → compare with original. Return `semantically_consistent` flag. |
| `compare_tokenizers(sentence: str) -> Dict`                                                            | Compare fine-tuned and base tokenizers. Return `are_identical` boolean.           |

---

## 5. Expected Output

- `phase_03_tokenization.py` with 4 public functions.
- Notebook cells:
  - Call `tokenize_sentence_to_table()` to display a readable token table.
  - Call `decode_sanity_check()` to verify decoding preserves meaning.
  - Call `compare_tokenizers()` to show tokenizers are identical.
- Conclusion printed clearly.

---

## 6. Files Affected

| File                                                                                       | Action            |
| ------------------------------------------------------------------------------------------ | ----------------- |
| `docs/plan-doc/plan_before_process/phase_03_tokenization_investigation_plan_2026-08-12.md` | Create            |
| `processing_own_phase/phase_03_tokenization.py`                                            | Create            |
| `notebook_practice_3/practice_3.ipynb`                                                     | Add Phase 3 cells |

---

## 7. Validation / Sanity Checks

- [ ] `compare_tokenizers()` returns `are_identical = True`.
- [ ] `vocab_size` for both tokenizers = 30522.
- [ ] `input_ids`, `tokens`, `attention_mask` are identical.
- [ ] `decode_sanity_check()` returns `semantically_consistent = True`.
- [ ] No side effects when importing the module.

---

## 8. Completion Criteria

Phase 3 is complete when:

1. `phase_03_tokenization.py` runs correctly with all 4 functions.
2. Notebook demonstrates that tokenizers are identical.
3. Conclusion is clearly documented.

---

## 9. Decision Log

| Decision                     | Details                                               | Reason                             |
| ---------------------------- | ----------------------------------------------------- | ---------------------------------- |
| Compare multiple attributes  | `vocab_size`, `input_ids`, `tokens`, `attention_mask` | Comprehensive proof of identity    |
| Use same sentence as Phase 2 | Consistency across phases                             | Easier to follow for readers       |
| Add decode_sanity_check      | Required by plan.md section 15                        | Ensures decoding preserves meaning |

---

## 10. Next Step

After Phase 3 plan is approved → implement `phase_03_tokenization.py` and test in notebook.

---

**End of Phase 3 Plan**
