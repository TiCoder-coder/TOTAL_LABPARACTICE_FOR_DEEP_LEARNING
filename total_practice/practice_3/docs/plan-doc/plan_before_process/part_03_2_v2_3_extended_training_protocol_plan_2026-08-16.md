# Practice 3 v2.3 — Extended Controlled Experiment Protocol Plan

## 1. Why v2.3 is Necessary
Practice 3 v2.2 introduced necessary regularization techniques (label smoothing factor = 0.1, warmup=480, and weight decay baseline) that improved training stability over the overfitting seen in v2.1. However, the first v2.2 experiment (`E1_lr_1e-5`) halted prematurely at Epoch 4 due to the strict `early_stopping_patience = 2`. 

While the Validation F1 score steadily improved to 0.8738, the tight patience threshold prevented the model from traversing small, normal validation fluctuations or further descending into the loss landscape. To give DistilBERT an adequate training opportunity while preserving the strict regularization controls, we are promoting a new protocol namespace: **v2.3**.

## 2. Evidence from Previous Versions
- **v2.1 Evidence**: Checkpoint audit confirmed drastic overfitting and confidence overfitting. The model's Train Loss decreased sharply while Validation Loss increased significantly.
- **v2.2 Evidence**: The model proved more stable, maintaining a narrow gap between Train and Validation Loss for longer. `E1_lr_1e-5` achieved best validation performance at Epoch 2 (Validation Loss: 0.4185, Val F1: 0.8656) and was halted at Epoch 4 because Epochs 3 and 4 showed slightly higher validation loss.

## 3. Configuration Revisions in v2.3
- **max_epochs = 15**: Increased from 10 to give the model ample runway to converge fully.
- **early_stopping_patience = 4**: Increased from 2. If the model fails to improve validation loss for 4 consecutive epochs, it halts. This prevents stopping on a local fluctuation while maintaining protection against deep overfitting. If an improvement is found at Epoch 5, the counter resets.
- **warmup_steps = 720**: Maintaining the 10% maximum steps ratio. Train size (7676) / Batch size (16) = 480 steps per epoch. `480 * 15 = 7200` total steps. 10% of 7200 = 720 steps.
- **v2.2 Results**: Cannot be mixed into v2.3 ranking because different `max_epochs` and `warmup_steps` fundamentally alter the learning rate trajectory and optimization landscape. Comparing a 10-epoch 480-warmup run with a 15-epoch 720-warmup run violates controlled experiment integrity.

## 4. Controlled Experiment Methodology
Six experiments will be created across two stages:

**Stage A — Learning Rate Search**
- E1: `learning_rate = 1e-5`
- E2: `learning_rate = 2e-5`
- E3: `learning_rate = 3e-5`
*(These differ solely in learning rate)*

**Stage B — DistilBERT-Specific Tuning**
- E4: Best Stage A LR + `weight_decay = 0.05`
- E5: Best Stage A LR + `classifier_dropout = 0.20`
- E6: Best Stage A LR + `staged_finetune` (freezing bottom DistilBERT transformer blocks, unfreezing top blocks)

Stage B remains strictly `BLOCKED_PENDING_STAGE_A` until Stage A officially completes and a Best LR is locked.

## 5. Holdout Protection & Verification
- Split fingerprint `4d22ccf19a61c37bad6138fcc12d40102603407fcfbc6e19a3f5e634803cbb13` is reused.
- Holdout remains SEALED (`evaluation_count = 0`).
- Strict strict configuration hashes and authorization protocols will guard every run.
- Zero-Training Validation (ZTV) will be performed for E1/E2/E3 during protocol creation to verify loss finiteness and logits shape. No official training will execute.
