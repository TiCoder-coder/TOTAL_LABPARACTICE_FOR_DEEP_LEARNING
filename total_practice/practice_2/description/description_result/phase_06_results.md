# Phase 6 — Model Building Results

[Phase 5 results](phase_05_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 7 results](phase_07_results.md)

| Cell output | Description |
|---|---|
| Cell 13 | ResNet18 model name and ten-class output |
| Cell 13 | Total, trainable, and frozen parameter counts |
| Cell 13 | Forward shape `[batch_size, 10]` |
| Cell 13 | Finite Cross Entropy loss |
| Cell 13 | Gradient-population sanity check |

The model returns raw logits and uses the ImageNet-pretrained ResNet18 backbone. The official selected checkpoint is [E2 best.pt](../../runs/E2_resnet18_partial_6c5d4ec5/best.pt).
