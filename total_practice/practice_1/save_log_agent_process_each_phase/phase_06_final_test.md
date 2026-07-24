# Phase 6 — Final Test Log

**Date:** 2026-07-24  
**Agent:** Main Agent (Cursor)

## Selected Model

```text
Experiment: E5_adam
Architecture: [256, 128]
Dropout: 0.2
Optimizer: Adam, lr=0.001
Best Val Acc: 0.8858
Best Epoch: 10
```

## Final Test (One Run Only)

```text
Test Loss:     0.3286
Test Accuracy: 0.8817
```

### Per-class Accuracy

| Class        | Accuracy |
|--------------|----------|
| T-shirt/top  | 0.7960   |
| Trouser      | 0.9760   |
| Pullover     | 0.7380   |
| Dress        | 0.9160   |
| Coat         | 0.8040   |
| Sandal       | 0.9720   |
| Shirt        | 0.7470   |
| Sneaker      | 0.9540   |
| Bag          | 0.9710   |
| Ankle boot   | 0.9430   |

## Error Analysis

- **Hardest classes**: Shirt (0.747), Pullover (0.738), T-shirt/top (0.796), Coat (0.804).
- **Easiest classes**: Trouser (0.976), Sandal (0.972), Bag (0.971).
- The model confuses visually similar tops (Shirt vs T-shirt, Pullover vs Coat). This is expected for an MLP that has no spatial inductive bias.

## Files Saved

- `outputs/best_model.pth` — best checkpoint (E5_adam)
- `outputs/E5_adam.pt` — full experiment checkpoint
- `outputs/confusion_matrix.png` — normalized confusion matrix
- `outputs/predictions_grid.png` — 16 test images with true vs predicted labels

## Outcome

Best model achieves **88.17% test accuracy**. Test set was used exactly once as required.
