# Phase 6 — Model Building Results

[Phase 5 results](phase_05_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 7 results](phase_07_results.md)

| Cell output | Description |
|---|---|
| Cell 36 | ResNet18 `head_only` baseline |
| Cell 36 | Total parameters: 11,181,642 |
| Cell 36 | Trainable classifier parameters: 5,130 |
| Cell 36 | Random ten-class baseline: 10% |

The cell builds the ImageNet-pretrained ResNet18 baseline and summarizes its parameter policy; it does not train the model. The current selected checkpoint is [winner best_val_loss.pt](../../runs/E2_resnet18_partial_2b5b94de/best_val_loss.pt), locked by [selection metadata](../../outputs/hyperparameter_selection_locked.json).
