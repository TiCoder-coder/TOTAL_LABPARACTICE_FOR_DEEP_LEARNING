# Phase 4 Dataset Loading Plan

**Date:** 2026-08-11
**Phase:** 4/15  
**File path:** `docs/plan-doc/plan_before_process/phase_04_dataset_loading_plan_2026-08-12.md`  
**Dependencies:** Phase 3 (Tokenizer Investigation)

---

## 1. Objective

Load the Rotten Tomatoes dataset from Hugging Face Hub and perform basic data quality checks:

- Dataset has exactly 3 splits: train, validation, test.
- Sample counts match expectations (8,530 / 1,066 / 1,066).
- Labels are binary (0 = Negative, 1 = Positive).
- No null values in `text` or `label` columns.

---

## 2. Input

| Source  | Content                                      |
| ------- | -------------------------------------------- |
| Dataset | `rotten_tomatoes` from Hugging Face Datasets |
| Version | datasets==5.0.1                              |

---

## 3. Processing

### 3.1 Load Dataset

- Use `load_dataset("rotten_tomatoes")`.
- Returns `DatasetDict` with 3 splits.

### 3.2 Verify Split Sizes

- Use `verify_split_sizes()` to check that train=8530, validation=1066, test=1066.
- Raise `ValueError` if any split size mismatches (not silently ignored).

### 3.3 Basic Statistics

- Count positive/negative samples in each split.
- Calculate percentages.

### 3.4 Null Value Check

- Check for null/empty values in `text` and `label` columns.
- Report counts for each column.

### 3.5 Sample Inspection

- Print first 3 rows of the train split to verify data structure.

---

## 4. Module Functions (5 functions)

| Function                                                                            | Purpose                                                                               |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `load_rotten_tomatoes() -> DatasetDict`                                             | Load dataset and return DatasetDict.                                                  |
| `verify_split_sizes(dataset: DatasetDict) -> Dict[str, int]`                        | Verify train=8530, val=1066, test=1066. Raise ValueError on mismatch.                 |
| `get_dataset_stats(dataset: DatasetDict) -> Dict`                                   | Return statistics: total samples, positive/negative counts and percentages per split. |
| `check_null_values(dataset: DatasetDict) -> Dict`                                   | Check for null/empty values in each column per split.                                 |
| `print_sample_rows(dataset: DatasetDict, split: str = "train", n: int = 3) -> None` | Print first n rows of a given split.                                                  |

---

## 5. Expected Output

- `phase_04_dataset_loading.py` with 5 public functions.
- Notebook cells:
  - `load_rotten_tomatoes()` → store as `dataset`.
  - `verify_split_sizes(dataset)` → confirm split sizes.
  - `get_dataset_stats(dataset)` → print statistics.
  - `check_null_values(dataset)` → print null report.
  - `print_sample_rows(dataset, "train", 3)` → inspect samples.

---

## 6. Files Affected

| File                                                                            | Action            |
| ------------------------------------------------------------------------------- | ----------------- |
| `docs/plan-doc/plan_before_process/phase_04_dataset_loading_plan_2026-08-12.md` | Create            |
| `processing_own_phase/phase_04_dataset_loading.py`                              | Create            |
| `notebook_practice_3/practice_3.ipynb`                                          | Add Phase 4 cells |

---

## 7. Validation / Sanity Checks

- [ ] Dataset has exactly 3 splits: `train`, `validation`, `test`.
- [ ] `verify_split_sizes()` returns `{"train": 8530, "validation": 1066, "test": 1066}`.
- [ ] `verify_split_sizes()` raises `ValueError` if sizes don't match.
- [ ] Labels are only 0 and 1.
- [ ] No null values in any column.
- [ ] Sample rows display correctly.

---

## 8. Completion Criteria

Phase 4 is complete when:

- Dataset is loaded successfully.
- Split sizes are verified.
- Statistics are printed clearly.
- No data quality issues found.

---

## 9. Decision Log

| Decision                      | Details                           | Reason                           |
| ----------------------------- | --------------------------------- | -------------------------------- |
| Don't save dataset            | Keep in notebook variable         | Simpler, avoid file I/O overhead |
| Use Hugging Face Datasets API | `load_dataset("rotten_tomatoes")` | Standard Hugging Face approach   |
| Raise ValueError on mismatch  | Strict verification               | Fail early if data is unexpected |

---

## 10. Next Step

After Phase 4 plan is approved → implement `phase_04_dataset_loading.py` and test in notebook.

---

**End of Phase 4 Plan**
