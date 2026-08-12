# Phase 1 Environment & Reproducibility Plan

**Date:** 2026-08-10  
**Phase:** 1/15  
**File path:** `docs/plan-doc/plan_before_process/phase_01_environment_plan_2026-08-10.md`  
**Dependencies:** Phase 0 (Practice Overview)

---

## 1. Objective

Set up a stable and reproducible environment for the entire Practice 3.

Specifically:

- Verify package versions match `requirements.txt`.
- Set global random seed for reproducibility.
- Detect and report device (CPU/GPU/MPS).
- Create reusable Python modules for subsequent phases.
- Ensure notebook can import modules without side effects.

---

## 2. Input

| Source                 | Content                                                                                                                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `requirements.txt`     | `transformers==5.14.1`, `datasets==5.0.1`, `evaluate==0.4.6`, `accelerate==1.14.0`, `torch==2.13.0+cpu`, `numpy==2.4.6`, `matplotlib==3.11.1`, `pandas==3.0.5`, `scikit-learn==1.9.0` |
| Current Environment    | Python 3.11.9 + venv, `torch==2.13.0+cpu`, CUDA available = False                                                                                                                     |
| Decisions from Phase 0 | CPU only                                                                                                                                                                              |
| Decisions at Phase 1   | Seed = 42                                                                                                                                                                             |
| `config.py`            | Contains: `PROJECT_ROOT`, `RESULT_DIR`, `REQUIRED_PACKAGE_VERSIONS`, `BASE_MODEL_CHECKPOINT`, `MAX_TOKEN_LENGTH`                                                                      |

---

## 3. Processing

### 3.1 Package Verification

- Check all 9 packages from `requirements.txt`.
- If version mismatch: emit `UserWarning` (continue execution).
- If package missing: raise `Exception` after checking all packages (not stopping at first missing).

### 3.2 Reproducibility Setup

- Set global seed = 42 for `random`, `numpy`, `torch`.
- If CUDA available: set `torch.backends.cudnn.deterministic = True` and `benchmark = False`.
- **Note:** This seed will be reused in Phase 8/9 for Hugging Face `TrainingArguments(seed=SEED)`.

### 3.3 Device Detection

- Auto-detect device with priority: `cuda` → `mps` (Apple Silicon) → `cpu`.
- No hard-coding of device names.
- Use `torch.cuda.is_available()` and `torch.backends.mps.is_available()`.

### 3.4 Module Functions (7 functions)

| Function                                                                              | Purpose                                                                                                                                   |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `ensure_directories() -> None`                                                        | Create required log directories using `PROJECT_ROOT`.                                                                                     |
| `set_seed(seed: int = 42) -> None`                                                    | Set seed for `random`, `numpy`, `torch`. Set cudnn deterministic if CUDA available.                                                       |
| `get_device() -> str`                                                                 | Detect and return device: `cuda` / `mps` / `cpu`. No hard-coding.                                                                         |
| `check_required_packages() -> dict`                                                   | Verify versions against `REQUIRED_PACKAGE_VERSIONS` from `config.py`. Emit warnings for mismatches; raise Exception for missing packages. |
| `check_huggingface_connectivity(checkpoint: str = "distilbert-base-uncased") -> dict` | Attempt to load a lightweight tokenizer from Hugging Face Hub. Return `{"status": "ok"/"failed", "error": ...}`.                          |
| `print_environment_info() -> dict`                                                    | Call `check_required_packages()` and `check_huggingface_connectivity()`. Print and return environment info.                               |
| `save_environment_report(info: dict) -> Path`                                         | Save environment info to `RESULT_DIR / 2026-08-10_phase01-environment-log.json`.                                                          |

### 3.5 Automatic Directory Creation

- `ensure_directories()` called inside `print_environment_info()` and `save_environment_report()`.
- Creates: `docs/result/`, `docs/plan-doc/analysis_error/`, `docs/plan-doc/plan_before_process/`, `docs/plan-doc/plan_to_refactor&fix/`, `docs/save_process_proceduce_own_phase_refactor&fix/`.

---

## 4. Expected Output

- `phase_01_environment.py` with 7 functions.
- Notebook cells import and call functions; output shows:
  - Package version table
  - Device detection result
  - Hugging Face connectivity status
  - Environment report summary
- JSON log saved at `docs/result/2026-08-10_phase01-environment-log.json`.
- No hard-coded `device = "cpu"` or `device = "cuda"`.
- No side effects when importing the module.

---

## 5. Files Affected

| File                                                                        | Action            |
| --------------------------------------------------------------------------- | ----------------- |
| `docs/plan-doc/plan_before_process/phase_01_environment_plan_2026-08-10.md` | Create            |
| `processing_own_phase/phase_01_environment.py`                              | Create            |
| `docs/result/2026-08-10_phase01-environment-log.json`                       | Created on run    |
| `notebook_practice_3/practice_3.ipynb`                                      | Add Phase 1 cells |

---

## 6. Validation / Sanity Checks

- [ ] `check_required_packages()` reports all 9 packages.
- [ ] `check_required_packages()` raises Exception if any package is missing.
- [ ] `set_seed(42)` produces identical random numbers on repeated calls.
- [ ] `get_device()` returns correct device.
- [ ] No hard-coded device names.
- [ ] `check_huggingface_connectivity()` returns `status: "ok"` or `"failed"` with error details.
- [ ] JSON log saved at correct path with correct content.
- [ ] Notebook Restart & Run All executes without errors.
- [ ] All functions have type hints and docstrings.
- [ ] No output printed when importing the module.

---

## 7. Completion Criteria

Phase 1 is complete when:

1. `phase_01_environment.py` runs correctly with all 7 functions and passes validation.
2. Notebook imports the module and displays environment information.
3. JSON log is created at the expected location.

---

## 8. Decision Log

| Decision                             | Details                                      | Reason                                       |
| ------------------------------------ | -------------------------------------------- | -------------------------------------------- |
| Use `config.py`                      | Single source of truth for constants         | Avoid duplication, easier maintenance        |
| Fixed filename for log               | `2026-08-10_phase01-environment-log.json`    | Overwrites on each run, matches plan         |
| Raise Exception for missing packages | Stop execution immediately                   | Prevents running with incomplete environment |
| Support MPS                          | Use `torch.backends.mps.is_available()`      | Team may use Mac M1/M2/M3                    |
| No side effects on import            | All demo code in `if __name__ == "__main__"` | Notebook imports cleanly                     |

---

## 9. Next Step

After Phase 1 plan is approved → implement `phase_01_environment.py` and test in notebook.

---

**End of Phase 1 Plan**
