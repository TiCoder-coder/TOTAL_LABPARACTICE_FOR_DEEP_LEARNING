# Phase v2.3 One-Time Holdout Evaluation Plan

**Date:** 2026-08-16
**Protocol:** practice_3_v2.3

## 1. Purpose of Holdout Evaluation Now
The validation phase has completely concluded and the winning hyperparameter configuration (`E4_weight_decay_0.05`) is securely locked. The `Holdout` dataset is being accessed now, and only now, to measure the model's true generalization performance on unseen data. The Holdout split was intentionally sequestered during training and hyperparameter search to prevent data leakage and selection bias.

## 2. Selection and No-Retuning Rule
- **Selection**: The winner was selected exclusively using Validation Loss performance, not Holdout.
- **No-Retuning**: This Holdout evaluation is strictly a one-way mirror. No hyperparameters (learning rate, weight decay, dropout, epochs, patience, etc.) will be changed as a result of this evaluation. The Official Test dataset remains sealed.

## 3. One-Time Evaluation Protocol
To prevent iterative "peeking," the Holdout inference will be claimed programmatically.
1. The script will first verify `winner_lock_hash` and `checkpoint_hash` match the known locked values.
2. It will write a "CLAIMED" record.
3. It will perform exactly ONE `Trainer.predict()` pass over the `Holdout` dataset. No redundant `Trainer.evaluate()` calls will be made.
4. The script will save the raw logits, probabilities, and labels into `holdout_predictions.csv` so that all future error analysis and confusion matrices can be built from this static file without re-running inference.

## 4. Artifact Outputs
The evaluation script will generate:
- `holdout_predictions.csv`: Row-level inference containing text, true_label, predicted_label, logits, and probabilities.
- `final_holdout_metrics.json`: Aggregated metrics (Loss, Accuracy, Precision, Recall, F1) + TP, TN, FP, FN counts.
- `validation_vs_holdout_comparison.json`: Gap analysis between validation and holdout metrics.
- `final_holdout_artifact_manifest.json`: SHA256 hashes of all evaluation outputs.
- `final_holdout_evaluation.md`: Presentation summary.
- `practice_3_v2_3_one_time_holdout_evaluation_process_2026-08-16.md`: Detailed process log.

## 5. Metrics Calculated
- Loss
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix Counts (TP, TN, FP, FN)
- Diagnostic: Mean confidence, wrong-prediction confidence, prediction entropy, Brier score

## 6. Official Test Exclusion & No-Training Rule
- `Trainer.train()`, `optimizer.step()`, and `loss.backward()` will NOT be called. 
- The Hugging Face `test` split (or the designated "Official Test" split) will remain unaccessed.
