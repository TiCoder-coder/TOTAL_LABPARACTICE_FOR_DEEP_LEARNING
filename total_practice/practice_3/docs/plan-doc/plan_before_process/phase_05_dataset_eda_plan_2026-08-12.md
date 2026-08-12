# Phase 5 EDA and Sanity Checks Plan

**Date:** 2026-08-12  
**Phase:** 5/15  
**File path:** `docs/plan-doc/plan_before_process/phase_05_dataset_eda_plan_2026-08-12.md`  
**Dependencies:** Phase 4 (Dataset Loading)

---

## 1. Objective

Perform exploratory data analysis (EDA) and quality checks on the Rotten Tomatoes dataset:

- Analyze text lengths (character, word, token) to determine `max_length`.
- Verify label balance (positive/negative) across splits.
- Check for duplicate texts.
- Visualize token length distribution with a histogram.
- Ensure dataset is ready for tokenization.

---

## 2. Input

| Source    | Content                                 |
| --------- | --------------------------------------- |
| Dataset   | DatasetDict from Phase 4                |
| Tokenizer | `distilbert-base-uncased` (from config) |
| Libraries | matplotlib, numpy, datasets             |

---

## 3. Processing

### 3.1 Schema Validation

- Check that `text` and `label` columns exist in all splits.
- Use `check_schema()`.

### 3.2 Text Length Analysis

- Compute lengths in 3 ways: character count, word count, token count.
- Use actual tokenizer (not word splitting) for token lengths.
- Calculate: min, max, mean, median, p95, p99.

### 3.3 Recommend max_length

- Based on real token length distribution (p95, p99, max).
- Use `recommend_max_length()` to return these values.

### 3.4 Duplicate Check

- Count exact duplicate texts in each split.
- Use `check_duplicates()`.

### 3.5 Label Balance

- Calculate positive/negative counts and percentages.
- Use `get_label_balance()`.

### 3.6 Visualization

- Plot token length histogram for the train split.
- Draw reference line at recommended `max_length` (p99).
- Save plot to `docs/result/phase_05_token_length_distribution.png`.

---

## 4. Module Functions (6 functions)

| Function                                                                                                                                                                       | Purpose                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| `check_schema(dataset: DatasetDict) -> Dict[str, bool]`                                                                                                                        | Verify `text` and `label` columns exist in each split.     |
| `compute_text_lengths(dataset: DatasetDict, tokenizer, split: str = "train") -> Dict`                                                                                          | Compute character, word, and token length statistics.      |
| `recommend_max_length(dataset: DatasetDict, tokenizer, split: str = "train") -> Dict`                                                                                          | Return p95, p99, and max token lengths.                    |
| `check_duplicates(dataset: DatasetDict) -> Dict[str, int]`                                                                                                                     | Count exact duplicate texts in each split.                 |
| `plot_token_length_distribution(dataset: DatasetDict, tokenizer, split: str = "train", max_length_reference: Optional[float] = None, save_path: Optional[str] = None) -> None` | Plot token length histogram and save to `docs/result/`.    |
| `get_label_balance(dataset: DatasetDict) -> Dict`                                                                                                                              | Return positive/negative counts and percentages per split. |

---

## 5. Expected Output

- `phase_05_dataset_eda.py` with 6 public functions.
- Notebook cells demonstrate all functions.
- Histogram saved at `docs/result/phase_05_token_length_distribution.png`.
- Decision: `max_length` is documented.

---

## 6. Files Affected

| File                                                                        | Action            |
| --------------------------------------------------------------------------- | ----------------- |
| `docs/plan-doc/plan_before_process/phase_05_dataset_eda_plan_2026-08-12.md` | Create            |
| `processing_own_phase/phase_05_dataset_eda.py`                              | Create            |
| `docs/result/phase_05_token_length_distribution.png`                        | Created on run    |
| `notebook_practice_3/practice_3.ipynb`                                      | Add Phase 5 cells |

---

## 7. Validation / Sanity Checks

- [ ] `check_schema()` returns `True` for all splits.
- [ ] `compute_text_lengths()` returns valid statistics.
- [ ] `recommend_max_length()` returns p95, p99, max values.
- [ ] `check_duplicates()` returns non-negative integers.
- [ ] `get_label_balance()` shows ~50% positive/negative in all splits.
- [ ] Histogram is saved at the correct path.
- [ ] No side effects when importing the module.

---

## 8. Completion Criteria

Phase 5 is complete when:

- Text length statistics are printed clearly.
- Histogram is saved successfully.
- `max_length` decision is documented.
- Dataset is confirmed ready for tokenization.

---

## 9. Decision Log

| Decision                          | Details                                    | Reason                                               |
| --------------------------------- | ------------------------------------------ | ---------------------------------------------------- |
| Use tokenizer for length analysis | Compute token counts with actual tokenizer | More accurate than word count for Transformer models |
| Use p99 for max_length            | Based on statistical distribution          | Covers 99% of sentences without excessive truncation |
| Save histogram                    | To `docs/result/`                          | Provides visual evidence for the report              |

---

## 10. Next Step

After Phase 5 plan is approved → implement `phase_05_dataset_eda.py` and test in notebook.

---

**End of Phase 5 Plan**
