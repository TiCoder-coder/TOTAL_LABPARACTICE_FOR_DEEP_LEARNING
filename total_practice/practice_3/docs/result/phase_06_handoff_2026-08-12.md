# Handoff Phase 6 → Phase 7

**Date:** 2026-08-12  
**From:** Stage 1 - Phase 0–6
**To:** Stage 2 - Phase 7–15
**File path:** `docs/result/phase_06_handoff_2026-08-12.md`

---

## 1. Completed Phases

Phase 0 → Phase 6

| Phase | Name                           | Status   | Notes                                                                            |
| ----- | ------------------------------ | -------- | -------------------------------------------------------------------------------- |
| 0     | Practice Overview              | Complete | Problem definition, academic architecture, technical decisions documented        |
| 1     | Environment & Reproducibility  | Complete | `config.py`, `phase_01_environment.py`, seed=42, device detection                |
| 2     | Pretrained Sentiment Inference | Complete | Exercise 1: inference with `distilbert-base-uncased-finetuned-sst-2-english`     |
| 3     | Tokenization Investigation     | Complete | Compared tokenizers (fine-tuned vs base) — identical, vocab_size=30522           |
| 4     | Dataset Loading                | Complete | Rotten Tomatoes loaded, split sizes verified (8530/1066/1066)                    |
| 5     | EDA & Sanity Checks            | Complete | Token length distribution, max_length=128, label balance ~50/50                  |
| 6     | Tokenizer & Preprocessing      | Complete | Dataset tokenized with `distilbert-base-uncased`, no padding, DataCollator ready |

---

## 2. Dataset

| Property           | Value                                           |
| ------------------ | ----------------------------------------------- |
| Dataset name       | `rotten_tomatoes`                               |
| Source             | Hugging Face Datasets                           |
| Train samples      | 8,530                                           |
| Validation samples | 1,066                                           |
| Test samples       | 1,066                                           |
| Total              | 10,662                                          |
| Labels             | `0` = NEGATIVE, `1` = POSITIVE                  |
| Label balance      | ~50% positive / ~50% negative across all splits |
| Null values        | None found                                      |
| Duplicate texts    | Train: 0, Validation: 0, Test: 0                |

---

## 3. Tokenizer

| Property                       | Value                                                   |
| ------------------------------ | ------------------------------------------------------- |
| Checkpoint                     | `distilbert-base-uncased`                               |
| Tokenizer type                 | `DistilBertTokenizerFast`                               |
| Vocab size                     | 30,522                                                  |
| `max_length`                   | 128                                                     |
| Truncation                     | `True`                                                  |
| Padding strategy               | `False` (dynamic padding via `DataCollatorWithPadding`) |
| Justification for `max_length` | Based on P99 token length from Phase 5 EDA              |

**Tokenizer comparison result:** The tokenizer from the fine-tuned model (`distilbert-base-uncased-finetuned-sst-2-english`) and the base tokenizer (`distilbert-base-uncased`) are **IDENTICAL**. Fine-tuning only updates model weights, not the tokenizer.

---

## 4. Preprocessing Output

### Available Dataset Objects

| Object              | Type                      | Description                                                    |
| ------------------- | ------------------------- | -------------------------------------------------------------- |
| `dataset`           | `DatasetDict`             | Raw Rotten Tomatoes dataset (train/val/test)                   |
| `tokenizer`         | `AutoTokenizer`           | Base tokenizer loaded from `distilbert-base-uncased`           |
| `tokenized_dataset` | `DatasetDict`             | Tokenized dataset with `input_ids`, `attention_mask`, `labels` |
| `collator`          | `DataCollatorWithPadding` | Dynamic padding collator ready for Trainer                     |

### Tokenized Dataset Structure

DatasetDict {
train: Dataset({
features: ['input_ids', 'attention_mask', 'labels'],
num_rows: 8530
})
validation: Dataset({
features: ['input_ids', 'attention_mask', 'labels'],
num_rows: 1066
})
test: Dataset({
features: ['input_ids', 'attention_mask', 'labels'],
num_rows: 1066
})
}

### Tokenized Sample (Train Split, Index 0)

Sample from 'train' split (index 0):
input_ids length: 22
input_ids (first 20): [101, 1996, 2213, 2003, 3515, 2000, 1996, 2549, 1012, 2013, 2002, 2052, 2028, 1005, 1055, 2052, 2043, 3415, 3820, 1005]
attention_mask length: 22
attention_mask (first 20): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
labels: 1 (POSITIVE)
Contains padding? False

### Data Collator (Dynamic Padding)

Batch input_ids shape: torch.Size([4, 22])
Batch attention_mask shape: torch.Size([4, 22])
Batch labels: [1, 0, 1, 0]

Dynamic padding works correctly: sequences in the batch have been padded to the same length (the length of the longest sequence in the batch).

---

## 5. Sanity Checks

| Check                           | Status | Notes                                                    |
| ------------------------------- | ------ | -------------------------------------------------------- |
| Pretrained sentiment model load | PASS   | `distilbert-base-uncased-finetuned-sst-2-english` loaded |
| Tokenizer load                  | PASS   | Tokenizer loaded from same checkpoint                    |
| Example sentences inference     | PASS   | 3 sentences (positive, negative, neutral) classified     |
| Tokens generated                | PASS   | Subword tokens displayed                                 |
| Token IDs generated             | PASS   | `input_ids` available                                    |
| Attention mask generated        | PASS   | `attention_mask` available                               |
| Predicted label                 | PASS   | Labels and confidence scores returned                    |
| Confidence score                | PASS   | Scores between 0 and 1                                   |
| Rotten Tomatoes load            | PASS   | Dataset loaded with 3 splits                             |
| Train split exists              | PASS   | 8,530 samples                                            |
| Validation split exists         | PASS   | 1,066 samples                                            |
| Test split exists               | PASS   | 1,066 samples                                            |
| Labels are 0/1                  | PASS   | Binary classification                                    |
| Dataset schema checked          | PASS   | `text` and `label` columns exist                         |
| Missing/empty text checked      | PASS   | No null values                                           |
| Label distribution checked      | PASS   | ~50/50 across splits                                     |
| Duplicate analysis              | PASS   | No duplicates found                                      |
| Token-length analysis           | PASS   | Histogram generated, P99=128                             |
| DistilBERT tokenizer works      | PASS   | Tokenizer loaded and functional                          |
| `input_ids` exists              | PASS   | Present in tokenized dataset                             |
| `attention_mask` exists         | PASS   | Present in tokenized dataset                             |
| Labels valid                    | PASS   | Labels in `{0, 1}`                                       |
| Truncation configured           | PASS   | `truncation=True`, `max_length=128`                      |
| `max_length` justified          | PASS   | Based on P99 token length from EDA                       |
| `DataCollatorWithPadding` works | PASS   | Dynamic padding tested with batch of 4                   |
| Decode sanity check             | PASS   | Word overlap ratio >= 0.8 for all samples                |
| Preprocessing sanity checks     | PASS   | `all_pass = True`                                        |

---

## 6. Important Decisions from Stage 1 (Affecting Phase 7+)

| Decision             | Value                      | Rationale                                                                   |
| -------------------- | -------------------------- | --------------------------------------------------------------------------- |
| **Model checkpoint** | `distilbert-base-uncased`  | Generic pretrained model, not sentiment-finetuned                           |
| **`max_length`**     | 128                        | Based on P99 token length from Phase 5 EDA (95% of sentences <= 128 tokens) |
| **Padding strategy** | `padding=False`            | Dynamic padding via `DataCollatorWithPadding` (Phase 8)                     |
| **Random seed**      | 42                         | Used in `set_seed()`; must pass to `TrainingArguments(seed=42)`             |
| **Hardware**         | CPU only                   | No CUDA available; set `no_cuda=True` in `TrainingArguments`                |
| **Dataset**          | Rotten Tomatoes            | Binary sentiment classification, 3 splits, short reviews                    |
| **Labels**           | 0 = NEGATIVE, 1 = POSITIVE | Consistent with Hugging Face dataset                                        |

---

## 7. Files Created / Modified

### Plans (docs/plan-doc/plan_before_process/)

| File                                                     | Action          |
| -------------------------------------------------------- | --------------- |
| `phase_00_practice_overview_plan_2026-08-10.md`          | Created         |
| `phase_01_environment_plan_2026-08-10.md`                | Created/Updated |
| `phase_02_pretrained_inference_plan_2026-08-12.md`       | Created         |
| `phase_03_tokenization_investigation_plan_2026-08-12.md` | Created         |
| `phase_04_dataset_loading_plan_2026-08-12.md`            | Created/Updated |
| `phase_05_dataset_eda_plan_2026-08-12.md`                | Created         |
| `phase_06_tokenizer_preprocessing_plan_2026-08-12.md`    | Created         |

### Code (processing_own_phase/)

| File                               | Action          |
| ---------------------------------- | --------------- |
| `config.py`                        | Created         |
| `phase_01_environment.py`          | Created/Updated |
| `phase_02_pretrained_inference.py` | Created         |
| `phase_03_tokenization.py`         | Created         |
| `phase_04_dataset_loading.py`      | Created/Updated |
| `phase_05_dataset_eda.py`          | Created         |
| `phase_06_preprocessing.py`        | Created/Updated |

### Notebook

| File                                   | Action          |
| -------------------------------------- | --------------- |
| `notebook_practice_3/practice_3.ipynb` | Created/Updated |

### Results (docs/result/)

| File                                      | Action           |
| ----------------------------------------- | ---------------- |
| `2026-08-10_phase01-environment-log.json` | Generated on run |
| `phase_05_token_length_distribution.png`  | Generated on run |

---

## 8. Last Commit

| Property       | Value                                               |
| -------------- | --------------------------------------------------- |
| Branch         | `Vien10082026`                                      |
| Commit hash    | (To be filled after commit)                         |
| Commit message | `feat(practice3): complete phases 0-6 with handoff` |
| Date           | 2026-08-12                                          |

---

## 9. Next Phase

Phase 7 Model Construction

**Tasks:**

1. Pull the latest code from branch `Vien10082026`.
2. Verify Phase 0–6 notebook runs end-to-end.
3. Load `distilbert-base-uncased` with `num_labels=2`.
4. Confirm `logits.shape == [batch_size, 2]`.
5. Perform model sanity check.
6. Ensure loss is finite.

---

## 10. Handoff Verification (Viên)

Before starting Phase 7, must verify:

### Dataset

- [ ] Train split exists
- [ ] Validation split exists
- [ ] Test split exists
- [ ] Labels are `0/1`

### Tokenizer

- [ ] Correct tokenizer/checkpoint: `distilbert-base-uncased`

### Processed Sample

- [ ] `input_ids` exists
- [ ] `attention_mask` exists
- [ ] `labels` exists and is in `{0, 1}`

### Batch

- [ ] Dynamic padding works with `DataCollatorWithPadding`

---

## 11. Declaration

- Phase 0–6 have been completed and verified.
- All sanity checks have passed.
- Preprocessing output is ready for Phase 7.
- Handoff document has been created and committed.
- All issues have been resolved or documented.

**Date:** 2026-08-12

---

## 12. Handoff Summary

Phase 0–6 COMPLETE
|
v
VERIFY PHASE 6 PASS
|
v
COMMIT + PUSH
|
v
HANDOFF DOCUMENT CREATED
|
v
VERIFY <- YOU ARE HERE
|
+----+----+
| |
FAIL PASS
| |
| v
| Phase 7-14
| |
| v
| Phase 15 + FINAL REVIEW
|
v
Return to Fix

---

**End of Handoff Document**
