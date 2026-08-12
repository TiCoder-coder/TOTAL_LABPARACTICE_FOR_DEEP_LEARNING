# Phase 6 Tokenizer and Preprocessing Plan

**Date:** 2026-08-12  
**Phase:** 6/15  
**File path:** `docs/plan-doc/plan_before_process/phase_06_tokenizer_preprocessing_plan_2026-08-12.md`  
**Dependencies:** Phase 5 (EDA and Sanity Checks)

---

## 1. Objective

Prepare the dataset for fine-tuning by:

- Loading the base tokenizer (`distilbert-base-uncased`).
- Tokenizing the entire dataset (train, validation, test) with:
  - `truncation=True`
  - `max_length=128` (or from config)
  - `padding=False` (dynamic padding will be handled by `DataCollatorWithPadding` in Phase 8)
- Creating the `DataCollatorWithPadding` for dynamic padding.
- Performing sanity checks on tokenized samples.
- Returning the tokenized `DatasetDict` for the next phase.

---

## 2. Input

| Source     | Content                                                    |
| ---------- | ---------------------------------------------------------- |
| Dataset    | DatasetDict from Phase 4 (raw)                             |
| Tokenizer  | `AutoTokenizer.from_pretrained("distilbert-base-uncased")` |
| max_length | `MAX_TOKEN_LENGTH` from `config.py` (128)                  |
| Libraries  | transformers, datasets                                     |

---

## 3. Processing

### 3.1 Load Tokenizer

- Use `AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT)` from config.
- Base tokenizer (not fine-tuned).

### 3.2 Tokenize Dataset

- `tokenize_function`: tokenizes `text` with `truncation=True`, `padding=False`, `max_length=MAX_TOKEN_LENGTH`.
- Adds `labels` from the original dataset.
- Removes `text` and `label` columns (to save memory, `labels` is kept).
- Uses `dataset.map()` with `batched=True`.

### 3.3 Create DataCollator

- `DataCollatorWithPadding(tokenizer=tokenizer)` for dynamic padding in Phase 8.
- This ensures each batch is padded to the length of the longest sequence in that batch.

### 3.4 Sanity Checks

- `check_tokenized_sample()`: Verifies `input_ids`, `attention_mask`, `labels` exist and are valid.
- `decode_sanity_check()`: Tokenizes → decodes → compares with original text to ensure meaning is preserved.

### 3.5 Sample Inspection

- Print one tokenized sample to verify structure.

---

## 4. Module Functions (6 functions)

| Function                                                                                                                                                               | Purpose                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `get_tokenizer(model_name: str = BASE_MODEL_CHECKPOINT) -> AutoTokenizer`                                                                                              | Load and return base tokenizer.                          |
| `tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int = MAX_TOKEN_LENGTH) -> DatasetDict`                                                  | Tokenize dataset with truncation, no padding.            |
| `get_data_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding`                                                                                               | Create DataCollatorWithPadding for dynamic padding.      |
| `check_tokenized_sample(tokenized_dataset: DatasetDict, split: str = "train", idx: int = 0) -> Dict[str, Any]`                                                         | Check tokenized sample structure.                        |
| `decode_sanity_check(tokenized_dataset: DatasetDict, tokenizer: AutoTokenizer, original_dataset: DatasetDict, split: str = "train", n_samples: int = 3) -> List[Dict]` | Decode tokenized samples and compare with original text. |
| `show_tokenized_sample(tokenized_dataset: DatasetDict, split: str = "train", idx: int = 0) -> None`                                                                    | Print a tokenized sample for inspection.                 |

---

## 5. Expected Output

- `phase_06_preprocessing.py` with 6 public functions.
- Notebook cells demonstrate all functions.
- DataCollator tested with a batch of 4 samples.
- Tokenized dataset ready for handoff.

---

## 6. Files Affected

| File                                                                                    | Action            |
| --------------------------------------------------------------------------------------- | ----------------- |
| `docs/plan-doc/plan_before_process/phase_06_tokenizer_preprocessing_plan_2026-08-12.md` | Create            |
| `processing_own_phase/phase_06_preprocessing.py`                                        | Create            |
| `notebook_practice_3/practice_3.ipynb`                                                  | Add Phase 6 cells |

---

## 7. Validation / Sanity Checks

- [ ] `get_tokenizer()` returns tokenizer with `vocab_size = 30522`.
- [ ] `tokenize_dataset()` produces `input_ids` and `attention_mask` for all splits.
- [ ] All `input_ids` lengths ≤ `MAX_TOKEN_LENGTH`.
- [ ] No padding is present (all sequences have varying lengths).
- [ ] `check_tokenized_sample()` returns `all_pass = True`.
- [ ] `decode_sanity_check()` returns `word_overlap_ratio ≥ 0.8` for all samples.
- [ ] `DataCollatorWithPadding` pads batches dynamically.
- [ ] Number of samples per split is unchanged.

---

## 8. Completion Criteria

Phase 6 is complete when:

- Tokenizer is loaded.
- Dataset is tokenized successfully.
- DataCollator is created and tested.
- Sanity checks pass.
- Tokenized dataset is ready for handoff.

---

## 9. Decision Log

| Decision                  | Details                        | Reason                                                    |
| ------------------------- | ------------------------------ | --------------------------------------------------------- |
| `padding=False`           | No padding during tokenization | DataCollator handles dynamic padding, saves memory        |
| `max_length` from config  | Use `MAX_TOKEN_LENGTH` (128)   | Single source of truth, based on EDA                      |
| Use base tokenizer        | `distilbert-base-uncased`      | Same tokenizer as fine-tuned model (confirmed in Phase 3) |
| Remove `text` and `label` | After tokenization             | Reduces memory, `labels` is preserved                     |

---

## 10. Next Step

After Phase 6 plan is approved → implement `phase_06_preprocessing.py` and test in notebook.

When Phase 6 is complete → **Handoff** for Phases 7-15.

---

**End of Phase 6 Plan**
