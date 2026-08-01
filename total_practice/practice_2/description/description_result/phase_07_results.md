# Phase 7 — Model Training Results

[Phase 6 results](phase_06_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/demo_practice_2.ipynb) | [Phase 8 results](phase_08_results.md)

| Output | Description | Artifact |
|---|---|---|
| Cell 15 | E1 and E2 each contain five epoch records | [controlled_training_history.json](../../outputs/controlled_training_history.json) |
| Cell 15 | Train/Validation loss and Accuracy curves | [controlled_training_curves.png](../../reports/controlled_training_curves.png) |
| Cell 15 | Learning rate per epoch | [controlled_learning_rate.png](../../reports/controlled_learning_rate.png) |

Current best epochs:

- E1 `head_only`: epoch 4, Validation Accuracy 75.76%.
- E2 `partial_finetune`: epoch 4, Validation Accuracy 89.72%.
- Early Stopping did not trigger for either five-epoch run.
