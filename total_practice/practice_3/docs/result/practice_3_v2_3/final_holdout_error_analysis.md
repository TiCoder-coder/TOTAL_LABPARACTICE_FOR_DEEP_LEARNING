# Final Holdout Error Analysis
**Protocol:** practice_3_v2.3
**Winner:** E4_weight_decay_0.05

## A. Final Holdout Confusion Matrix
| | Predicted Negative (0) | Predicted Positive (1) |
|---|---|---|
| **Actual Negative (0)** | TN = 416 | FP = 64 |
| **Actual Positive (1)** | FN = 77 | TP = 403 |

*Figure generated at `figures/final_holdout_confusion_matrix.png`*

## B. What TP/TN/FP/FN Mean
- **TN (416)**: Negative sentiment reviews correctly identified as Negative.
- **TP (403)**: Positive sentiment reviews correctly identified as Positive.
- **FP (64)**: Negative sentiment reviews incorrectly classified as Positive.
- **FN (77)**: Positive sentiment reviews incorrectly classified as Negative.

## C. FP vs FN Comparison
- **False Positive Rate**: 13.33%
- **False Negative Rate**: 16.04%
- **Specificity**: 86.67%
- **Recall**: 83.96%

*The model is slightly more prone to `FALSE_NEGATIVE` errors.*

## D. Representative False Positive Examples
- **Confidence**: 0.9449
  > "'synthetic' is the best description of this well-meaning , beautifully produced film that sacrifices its promise for a high-powered star pedigree ."

- **Confidence**: 0.9395
  > "whatever eyre's failings as a dramatist , he deserves credit for bringing audiences into this hard and bitter place ."

- **Confidence**: 0.7383
  > "pretty much sucks , but has a funny moment or two ."

- **Confidence**: 0.5227
  > "although purportedly a study in modern alienation , it's really little more than a particularly slanted , gay s/m fantasy , enervating and deadeningly drawn-out ."

- **Confidence**: 0.5061
  > "you can practically smell the patchouli oil ."

## E. Representative False Negative Examples
- **Confidence**: 0.9375
  > "[hawn's character]is so bluntly written , without a trace of sentimentality , and so blisteringly defined , that every other character seems overlooked and underwritten ."

- **Confidence**: 0.9162
  > "it's virtually impossible to like any of these despicable characters ."

- **Confidence**: 0.8036
  > "each punch seen through prison bars , the fights become not so much a struggle of man vs . man as brother-man vs . the man ."

- **Confidence**: 0.5181
  > "i know that i'll never listen to marvin gaye or the supremes the same way again"

- **Confidence**: 0.5045
  > "the obnoxious title character provides the drama that gives added clout to this doc ."

## F. High-Confidence Errors
- **Total High-Confidence Errors (>=0.90)**: 14
- **High-Confidence FP**: 6
- **High-Confidence FN**: 8
- **Mean Wrong Confidence**: 0.7483

## G. Confidence Analysis
- **Correct Prediction Mean Confidence**: 0.8663
- **Correct Prediction Median Confidence**: 0.9075
- **Incorrect Prediction Mean Confidence**: 0.7483
- **Incorrect Prediction Median Confidence**: 0.7593

*Figure generated at `figures/correct_vs_incorrect_confidence.png`*

## H. Common Qualitative Error Patterns
Based on qualitative sampling, the following categories appear:
1. **Subtle/Mixed Sentiment**: Reviews that start positive but end with a negative conclusion (or vice versa), confusing the classifier.
2. **Sarcasm/Irony**: Positive words used to describe a negative movie experience.
3. **Ambiguous Wording**: Reviews where even a human reader might struggle to determine the ultimate binary sentiment.
4. **UNCLEAR / OTHER**: Some errors are simply the model misinterpreting standard text.

## I. Limitations
- **Diagnostic Only**: These qualitative labels are approximate and based on representative samples rather than an exhaustive semantic tagging of all 141 errors.
- **Calibration**: The model is highly confident even when wrong, which is typical for deep learning classifiers without temperature scaling.

## J. Final Interpretation
While there are 141 incorrect predictions, the model correctly predicted 819 out of 960 reviews (~85.31% Accuracy, ~85.11% F1). The distribution of errors leans slightly towards False Negatives (missing positive sentiment), but the imbalance is very mild (77 FN vs 64 FP).

**No retuning will be performed based on this analysis.**
