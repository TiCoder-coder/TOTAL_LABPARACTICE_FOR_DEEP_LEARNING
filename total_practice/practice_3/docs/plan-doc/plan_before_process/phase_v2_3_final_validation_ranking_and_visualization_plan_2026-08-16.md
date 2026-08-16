# Phase v2.3 Final Validation Ranking and Visualization Plan

**Date:** 2026-08-16
**Protocol:** practice_3_v2.3

## 1. Purpose
The purpose of this phase is to evaluate all executed hyperparameters against the validation dataset, strictly rank them according to the locked ranking policy, exclude invalid experiments, and lock the final winner. No real training or test/holdout evaluation will occur in this phase.

## 2. Valid and Excluded Runs

### Valid Runs (Participate in Ranking)
- `E1_lr_1e-5`: Baseline LR 1e-5
- `E2_lr_2e-5`: Baseline LR 2e-5
- `E3_lr_3e-5`: Baseline LR 3e-5
- `E4_weight_decay_0.05`: Winning LR (1e-5) + Weight Decay 0.05
- `E5b_classifier_dropout_0.40`: Winning LR (1e-5) + Classifier Dropout 0.40
- `E6c_staged_finetune`: Winning LR (1e-5) + Staged fine-tuning

### Excluded Runs (Historical Evidence Only)
- `E5_classifier_dropout_0.20`: INVALID_CONTROLLED_EXPERIMENT (Baseline dropout was already 0.20, so no variable changed).
- `E6_staged_finetune`: FAILED_EXECUTION (MPS SDPA dropout incompatibility).
- `E6b_staged_finetune`: FAILED_EXECUTION (Optimizer/Scheduler mismatch after Stage 1).

## 3. Ranking Policy
The exact historical ranking semantics will be preserved:
1. Lowest Validation Loss (primary)
2. Highest Validation F1
3. Highest Validation Accuracy
4. Earlier Best Epoch (tie-breaker for faster convergence)
5. Lexical experiment ID
*Ranking tolerance applied: `1e-6`*

**Selection criteria is strictly based on Validation Loss**, not maximum Accuracy, because the model begins overfitting (loss rises) before accuracy degrades. We aggressively stop at the loss minimum to ensure generalization.

## 4. Winner Lock Policy
Once the ranking is computed programmatically from the artifacts:
- The top-ranked run will be locked. (Expected: `E4_weight_decay_0.05`).
- A `final_validation_winner_lock.json` will be generated recording the precise checkpoint hash.
- **NO further hyperparameter tuning** will be permitted. Holdout dataset remains sealed.

## 5. Visualization Plan
We will generate presentation-quality plots from the `Trainer` state histories. 
Plots to generate in `docs/result/practice_3_v2_3/figures/`:
- `winner_train_val_loss.png`
- `winner_validation_accuracy.png`
- `winner_validation_f1.png`
- `winner_learning_rate.png`
- `experiment_best_val_loss_comparison.png`
- `experiment_best_accuracy_comparison.png`
- `experiment_best_f1_comparison.png`
- `experiment_best_vs_stop_epoch.png`
- `experiment_val_loss_curves.png`

## 6. Protection & No-Training Rule
- `Holdout` remains strictly `SEALED`. Evaluation count remains `0`.
- The `Official Test` dataset is untouched.
- `Trainer.train()` will not be called. No new checkpoints will be produced.
