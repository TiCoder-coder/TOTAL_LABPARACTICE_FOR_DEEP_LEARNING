# Phase 7 — Model Training Results

[Phase 6 results](phase_06_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 8 results](phase_08_results.md)

| Output | Description | Artifact |
|---|---|---|
| Cell 15 | E1 and E2 each contain five epoch records | [controlled_training_history.json](../../outputs/controlled_training_history.json) |
| Cell 15 | Train/Validation loss and Accuracy curves | [controlled_training_curves.png](../../reports/controlled_training_curves.png) |
| Cell 15 | Learning rate per epoch | [controlled_learning_rate.png](../../reports/controlled_learning_rate.png) |

Current best epochs:

- E1 `head_only`: epoch 4, Validation Accuracy 75.76%.
- E2 `partial_finetune`: epoch 4, Validation Accuracy 89.72%.
- Early Stopping did not trigger for either five-epoch run.
## Updated 25-epoch training result

The Validation-only search completed six runs on Apple MPS. Per-epoch metrics
were durably appended to `metrics.jsonl`, and the standalone live monitor can
plot loss, accuracy, macro F1, generalization gap, and head/backbone learning
rates while training is active. Five runs completed 25 epochs; the one-hidden-
layer head stopped early after nine epochs.

The winning run is `E2_resnet18_partial_2b5b94de`. Its minimum Validation loss
is `0.4303597612` at epoch 24, with Validation accuracy `94.42%` at that same
epoch. The maximum accuracy elsewhere in the run is not substituted for the
selected-checkpoint accuracy.
