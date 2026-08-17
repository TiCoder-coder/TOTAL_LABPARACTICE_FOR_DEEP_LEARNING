# Phase 11 – Error Analysis

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cells:
- [`Cell 28` (Code)](../../notebook_practice_3/practice_3.ipynb): Confusion matrix and confidence distribution figures.
- [`Cell 29` (Code)](../../notebook_practice_3/practice_3.ipynb): Representative error tables for False Positives, False Negatives, and High Confidence Errors.

Cell output:
Rendered figures [`final_holdout_confusion_matrix.png`](./practice_3_v2_3/figures/final_holdout_confusion_matrix.png) and [`correct_vs_incorrect_confidence.png`](./practice_3_v2_3/figures/correct_vs_incorrect_confidence.png), accompanied by 3 styled DataFrames categorizing misclassified holdout reviews.

## Processing Source
- [`final_holdout_evaluation_v2_3.py`](../../processing_own_phase/final_holdout_evaluation_v2_3.py)

## Artifact Sources
- [`holdout_errors.csv`](./practice_3_v2_3/holdout_errors.csv)
- [`high_confidence_holdout_errors.csv`](./practice_3_v2_3/high_confidence_holdout_errors.csv)
- [`holdout_predictions.csv`](./practice_3_v2_3/holdout_predictions.csv)
- [`final_holdout_confusion_matrix.png`](./practice_3_v2_3/figures/final_holdout_confusion_matrix.png)
- [`correct_vs_incorrect_confidence.png`](./practice_3_v2_3/figures/correct_vs_incorrect_confidence.png)

## Results
### Confusion Matrix Breakdown
| Metric | Count | Proportion | Interpretation |
|---|---:|---:|---|
| **True Negatives (TN)** | 416 | 43.33% | Correctly identified negative sentiment |
| **True Positives (TP)** | 403 | 41.98% | Correctly identified positive sentiment |
| **False Positives (FP)** | 64 | 6.67% | Negative reviews misclassified as Positive |
| **False Negatives (FN)** | 77 | 8.02% | Positive reviews misclassified as Negative |
| **Total Misclassifications** | **141** | **14.69%** | Overall error rate |

### Confidence & Calibration Metrics
- **Mean Overall Confidence**: `0.8489`
- **Mean Error Confidence**: `0.7483` (Errors are lower confidence near boundary)
- **Expected Calibration Error (ECE)**: `0.0422` (High probability calibration)
- **Brier Score**: `0.1092`

### Representative Misclassification Patterns
1. **Sarcasm & Subtle Tone**: Reviews containing positive words in cynical contexts (e.g., *"looks beautiful, although there is little else to recommend"*).
2. **Contrast & Mixed Signals**: Reviews with contrasting clauses where the model weights earlier clauses over final conclusions.

## Evidence
- [`final_holdout_confusion_matrix.png`](./practice_3_v2_3/figures/final_holdout_confusion_matrix.png) and [`correct_vs_incorrect_confidence.png`](./practice_3_v2_3/figures/correct_vs_incorrect_confidence.png) in Notebook [`Cell 28`](../../notebook_practice_3/practice_3.ipynb).
- False Positive, False Negative, and High Confidence error tables rendered in Notebook [`Cell 29`](../../notebook_practice_3/practice_3.ipynb).
- Raw prediction records in [`holdout_errors.csv`](./practice_3_v2_3/holdout_errors.csv).

## Summary
Error analysis shows a balanced distribution of mistakes (64 False Positives vs 77 False Negatives). Calibration analysis (ECE = 0.0422) confirms errors are predominantly low-confidence edge cases involving sarcasm or complex syntactic contrasts.
