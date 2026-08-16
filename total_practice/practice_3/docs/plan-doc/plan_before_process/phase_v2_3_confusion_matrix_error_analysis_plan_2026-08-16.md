# Phase v2.3 Confusion Matrix and Error Analysis Plan

**Date:** 2026-08-16
**Protocol:** practice_3_v2.3

## 1. Source Artifacts & Rule
All analyses will strictly consume the previously saved `docs/result/practice_3_v2_3/holdout_predictions.csv` and `final_holdout_metrics.json`. There will be **absolutely no second Holdout inference pass**. 

## 2. Confusion Matrix Plan
I will calculate the exact TN, FP, FN, TP directly from the saved CSV (True Labels vs Predicted Labels). I will generate a presentation-ready confusion matrix figure (`final_holdout_confusion_matrix.png`) illustrating these exact counts and percentages.

## 3. FP/FN & Confidence Analysis Plan
I will calculate error rates (FPR, FNR, Specificity, Recall).
I will isolate the 141 error rows into `holdout_errors.csv` containing `sample_index, text, true_label, predicted_label, probabilities, confidence, error_type`.

I will analyze False Positives (predicted positive, actually negative) and False Negatives (predicted negative, actually positive) to find:
- Mean and Median confidence of wrong predictions.
- High-confidence errors (confidence >= 0.90) into `high_confidence_holdout_errors.csv`.
- I will select ~5 representative examples for FP and FN without cherry-picking to understand qualitative errors.

## 4. Error Grouping
A qualitative analysis of the representative errors will group them into linguistic causes (e.g., subtle sentiment, sarcasm, ambiguous wording) or "UNCLEAR / OTHER". 

## 5. Presentation Outputs
I will generate:
- `correct_vs_incorrect_confidence.png` showing the distribution of confidence between correct vs. incorrect predictions.
- `final_holdout_error_analysis.md` (the comprehensive markdown presentation).
- `practice_3_v2_3_confusion_matrix_error_analysis_process_2026-08-16.md` (the execution process log).

## 6. No-Retuning Rule
Any insights discovered in this error analysis will **NOT** be used to tune hyperparameters or alter the winner. All insights are strictly diagnostic and marked for future work only.
