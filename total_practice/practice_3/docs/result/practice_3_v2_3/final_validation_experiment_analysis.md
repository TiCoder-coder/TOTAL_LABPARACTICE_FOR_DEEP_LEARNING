# Final Validation Experiment Analysis
Protocol: practice_3_v2.3

## A. Stage A — Learning Rate Search
Learning rate 1e-5 (E1) won over 2e-5 (E2) and 3e-5 (E3). 
Higher learning rates overfit earlier (E3 best epoch=1, stopped=5). 1e-5 generalized best.

## B. Stage B — Weight Decay
`E4_weight_decay_0.05` produced the lowest Validation Loss (0.4153). Improvement over E1 (0.4156) was small but measurable and consistent.

## C. Dropout
`E5_classifier_dropout_0.20` was invalid because baseline was already 0.20.
`E5b_classifier_dropout_0.40` correctly tested higher dropout but did not improve Validation Loss (0.4162).

## D. Staged Fine-tuning
`E6c_staged_finetune` delayed overfitting slightly (best epoch = 3, stopped epoch = 7) compared to baseline (best=2, stopped=6), but did not produce the lowest Validation Loss overall (0.4185).

## E. Winner
**`E4_weight_decay_0.05`** was selected by minimum Validation Loss.

### Note About Accuracy vs Loss
A later epoch may have higher Accuracy but worse Validation Loss. For example, E6c Epoch 7 reached Validation Accuracy ≈ 87.08% but Validation Loss degraded to ≈ 0.4954 (compared to 0.4185 at Epoch 3). Therefore, checkpoint selection strictly uses Validation Loss to prevent overfitting and ensure generalization.

### Parameter Search Summary
This phase thoroughly searched:
- **Learning Rate**: 1e-5, 2e-5, 3e-5
- **Weight Decay**: 0.01 (baseline), 0.05
- **Classifier Dropout**: 0.20 (baseline), 0.40
- **Strategy**: full vs staged fine-tuning
- **Budget**: max_epochs=15 (Note: 15 is the *maximum*, runs stopped early via patience=4 to prevent overfitting).

*TensorBoard logs are preserved for all valid experiments in the `runs/` directory.*
