# Phase 6 — Model Building Results

[Phase 5 results](phase_05_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 7 results](phase_07_results.md)

| Cell output | Description |
|---|---|
| Cell 13 | ResNet18 `head_only` baseline |
| Cell 13 | Total parameters: 11,181,642 |
| Cell 13 | Trainable classifier parameters: 5,130 |
| Cell 13 | Random ten-class baseline: 10% |

The cell builds the ImageNet-pretrained ResNet18 baseline and summarizes its parameter policy; it does not train the model. The official selected checkpoint is [E2 best.pt](../../runs/E2_resnet18_partial_6c5d4ec5/best.pt).
