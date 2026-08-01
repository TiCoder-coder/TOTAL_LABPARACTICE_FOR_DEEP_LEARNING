# Phase 7 — Training and Learning Curves

## Notebook location

- Notebook: [demo_practice_2.ipynb](../../notebooks/demo_practice_2.ipynb)
- Main cells: **Cells 14–15**
- Source: [train.py](../../processing_own_phase/train.py)
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Objective

Training optimizes model parameters on Train and monitors generalization on Validation across multiple epochs. The Test loader is not used in this phase.

## 2. Training loop

For each epoch, the training path:

1. calls `model.train()`;
2. reads batches from the Train DataLoader;
3. transfers images and labels to the selected device;
4. clears previous gradients;
5. performs the forward pass;
6. calculates Cross Entropy loss;
7. performs backward propagation;
8. clips gradients when configured;
9. applies the optimizer step;
10. accumulates sample-weighted loss and correct predictions.

Epoch-level Train Loss and Train Accuracy cover all processed samples rather than averaging batch accuracies without considering batch size.

## 3. Validation loop

Validation uses:

- `model.eval()`;
- `torch.no_grad()` or inference mode;
- no backward pass;
- no optimizer update;
- no random augmentation.

It produces `val_loss` and `val_accuracy` for learning-curve monitoring, checkpoint selection, scheduler behavior, and Early Stopping.

## 4. Required history fields

Every epoch records:

- `train_loss`;
- `train_accuracy` or artifact key `train_acc`;
- `val_loss`;
- `val_accuracy` or artifact key `val_acc`;
- `learning_rate` or artifact key `lr`;
- `epoch_time`.

Cell 15 reads the real artifact rather than constructing curves from hardcoded numbers.

Canonical artifacts:

- [controlled_training_history.json](../../outputs/controlled_training_history.json)
- [controlled_training_curves.png](../../reports/controlled_training_curves.png)
- [controlled_learning_rate.png](../../reports/controlled_learning_rate.png)

## 5. Epoch budget and Early Stopping

The official controlled configuration uses:

- five planned epochs;
- Early Stopping patience of three;
- Validation Accuracy as the best-model metric.

Five epochs satisfy the minimum approved multi-epoch requirement and produce meaningful curves instead of a one-point quick run. This remains a limited budget, so the comparison should be interpreted as a five-epoch controlled study rather than proof of full convergence.

Early Stopping should be reported as activated only when patience is exhausted before the planned budget. Both official runs completed all five epochs, so `early_stopping_triggered` is false.

## 6. Best checkpoint

Whenever the Validation metric improves, the pipeline saves a best checkpoint containing sufficient reconstruction metadata:

- model state;
- epoch;
- best Validation metric;
- architecture;
- training mode;
- configuration values needed for reload.

The best checkpoint is not necessarily the final epoch. Final evaluation must reload `best.pt` rather than use the model state left in memory after the last epoch.

## 7. Reading the learning curves

### Loss curves

- Falling Train Loss indicates that optimization is fitting the training data.
- Falling Validation Loss indicates improving generalization.
- Falling Train Loss with rising Validation Loss suggests overfitting.

### Accuracy curves

- A small Train–Validation gap is generally favorable.
- Strong Validation oscillation can indicate a high learning rate, insufficient data, or unstable fine-tuning.

### Learning-rate curve

This curve shows whether the scheduler actually changed the learning rate. The official five-epoch runs retained `0.001`, so the curve is flat; ReduceLROnPlateau did not reduce LR within this budget.

For E2, the best Validation result occurred at **epoch 4**. Epoch 5 declined, demonstrating why best-checkpoint persistence is necessary.

## 8. Test lock

Training and controlled history must not contain:

- `test_accuracy`;
- `test_loss`;
- Test predictions used to choose an epoch.

The Test DataLoader does not participate in `train_model()` or controlled selection.

## 9. Compute evidence

The persistent official run logs record `Using device: mps`. E1 took approximately 1,351 seconds and E2 approximately 2,273 seconds for five epochs. E2 takes longer because `partial_finetune` updates ResNet18 `layer4` in addition to the classifier.

## 10. Suggested presentation script

> Every epoch records Train and Validation loss, Accuracy, learning rate, and elapsed time. Validation runs in eval mode without gradients. The best checkpoint is selected by Validation Accuracy; E2 reached its best result at epoch 4, while epoch 5 declined, so the pipeline correctly reloads `best.pt` instead of using the final state.

## 11. Transition to the next phase

[Phase 8](phase_08_controlled_experiments.md) explains how E1 and E2 are compared fairly with every background factor held constant.
