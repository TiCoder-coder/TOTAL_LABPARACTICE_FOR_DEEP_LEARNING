# Final Holdout Evaluation
**Protocol**: practice_3_v2.3

## A. Locked Model
- **Winner Run ID**: `E4_weight_decay_0.05`
- **Checkpoint Hash**: `d4a8b8377de3ade5...`
- **Lock Hash**: `010cc06d7cb9edc3...`

## B. Why Holdout is Evaluated Only Now
The model selection process relied purely on the Validation split (where `E4` was chosen). The Holdout was sequestered and intentionally excluded from hyperparameter ranking. This one-time execution serves as an unbiased estimate of real-world generalization performance.

## C. Holdout Metrics
- **Loss**: 0.3536
- **Accuracy**: 85.31%
- **Precision**: 86.30%
- **Recall**: 83.96%
- **F1-Score**: 85.11%

*Confusion Counts*: TP=403, TN=416, FP=64, FN=77

## D. Validation vs Holdout Comparison
- **Loss Diff**: -0.0618 (Val: 0.4153, Holdout: 0.3536)
- **Acc Gap**: +1.46% (Val: 86.77%, Holdout: 85.31%)
- **F1 Gap**: +1.67% (Val: 86.78%, Holdout: 85.11%)

## E. Generalization Interpretation
The model generalizes exceptionally well to the Holdout split, as metrics are comparable to Validation performance.

## F. Statement of Finality
**No further hyperparameter tuning is permitted.** The model configuration is permanently locked. The Holdout split has been evaluated.
