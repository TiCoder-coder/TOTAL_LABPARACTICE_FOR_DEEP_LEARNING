# Phase 10 – Final Holdout Evaluation

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cell:
[`Cell 26` (Code)](../../notebook_practice_3/practice_3.ipynb)

Cell output:
Styled DataFrame table titled `Validation vs Holdout Metrics` comparing the locked Winner E4 performance on the Validation split versus the unseen Holdout split.

## Processing Source
- [`final_holdout_evaluation_v2_3.py`](../../processing_own_phase/final_holdout_evaluation_v2_3.py)
- Function: `guarded_holdout_evaluation()`
- [`holdout_guard_v2.py`](../../processing_own_phase/holdout_guard_v2.py)

## Artifact Sources
- [`final_holdout_metrics.json`](./practice_3_v2_3/final_holdout_metrics.json)
- [`final_validation_winner_lock.json`](./practice_3_v2_3/final_validation_winner_lock.json)
- [`validation_vs_holdout_comparison.json`](./practice_3_v2_3/validation_vs_holdout_comparison.json)

## Results
| Metric | Validation Split (E4 Lock) | Holdout Split (Final Test) | Generalization Gap |
|---|---:|---:|---:|
| **Loss** | 0.4153 | **0.3536** | -0.0617 (Better) |
| **Accuracy** | 0.8677 | **0.8531** | -0.0146 (-1.46%) |
| **Macro-F1** | 0.8678 | **0.8511** | -0.0167 (-1.67%) |
| **Precision** | — | **0.8630** | — |
| **Recall** | — | **0.8396** | — |

- **Holdout Evaluation Count**: Exactly **1** (Strictly guarded, official test set excluded).
- **Holdout Dataset Size**: 960 balanced samples (480 Negative / 480 Positive).

## Evidence
- `Validation vs Holdout Metrics` table rendered in Notebook [`Cell 26`](../../notebook_practice_3/practice_3.ipynb).
- Authoritative evaluation record in [`final_holdout_metrics.json`](./practice_3_v2_3/final_holdout_metrics.json).
- Access audit state logged in [`holdout_access_state.json`](./practice_3_v2_3/holdout_access_state.json).

## Summary
The final locked `E4` model was evaluated exactly once on the sealed Holdout dataset (960 samples), achieving **85.31% Accuracy** and an **F1 score of 0.8511**. The minimal generalization gap (1.46%) confirms strong out-of-sample robustness.
