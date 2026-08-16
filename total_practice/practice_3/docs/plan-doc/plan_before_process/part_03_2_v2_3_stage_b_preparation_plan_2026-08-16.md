# Practice 3 v2.3 — Stage B Preparation Plan

**Date:** 2026-08-16

## 1. Stage A Results
Stage A tested three learning rate variations in a strictly controlled manner under protocol `practice_3_v2.3` (15 epochs, early_stopping_patience=4, warmup_steps=720, label_smoothing=0.1).
- **E1_lr_1e-5**: Best Val Loss = 0.4156 | Val F1 = 0.8660 | Stopped Epoch = 6 (Best 2)
- **E2_lr_2e-5**: Best Val Loss = 0.4396 | Val F1 = 0.8485 | Stopped Epoch = 6 (Best 2)
- **E3_lr_3e-5**: Best Val Loss = 0.4275 | Val F1 = 0.8477 | Stopped Epoch = 5 (Best 1)

**Why 1e-5 Won:**
According to the explicit ranking protocol:
1. E1 achieved the strictly lowest Validation Loss (0.4156 vs 0.4396 / 0.4275).
2. E1 also achieved the highest Validation F1 (0.8660 vs 0.8485 / 0.8477).
Consequently, `1e-5` is locked as the optimal Learning Rate.

## 2. Why Stage B is Being Performed
While E1 (the baseline) is stable and achieves strong early performance (Val F1 ~0.866), the model still suffers from confidence overfitting as training progresses beyond epoch 2. Stage B introduces regularization and adaptation variations, strictly isolating one factor at a time, to determine if they can further mitigate this issue or push the model to a later best epoch with a narrower generalization gap.

## 3. Purpose of Experiments
- **E4 (Weight Decay = 0.05)**: Testing whether increased L2 penalty on network weights (from 0.01 to 0.05) restricts over-confident extrapolation without significantly damping the learning rate.
- **E5 (Classifier Dropout = 0.20)**: Testing whether increased pre-classifier dropout prevents the final linear layer from simply memorizing intermediate states, pushing the representation burden back into the DistilBERT layers.
- **E6 (Staged Fine-Tuning)**: Freezing the bottom layers (Embeddings, Layers 0-2) initially and only training the top representations (Layers 3-5, Classifier). This preserves generic representations and adapts only high-level features before globally unfreezing, mitigating the catastrophic forgetting and confidence spikes seen when training a large transformer with a small dataset.

## 4. Holdout Protection
The dataset split remains perfectly consistent (`4d22ccf19a61c37bad6138fcc12d40102603407fcfbc6e19a3f5e634803cbb13`). The `Holdout` subset remains `SEALED` with an evaluation count of 0. All ranking and comparison logic uses Validation metrics exclusively. The Holdout is only accessed when Stage B completes entirely and a global Winner is declared.
