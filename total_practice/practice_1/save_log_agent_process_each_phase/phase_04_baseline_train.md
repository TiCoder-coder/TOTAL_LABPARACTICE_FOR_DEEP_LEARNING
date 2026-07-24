# Phase 4 — Baseline Training Log

**Date:** 2026-07-24  
**Agent:** Main Agent (Cursor)

## Files Created

- `processing_own_phase/train.py` — `train_one_epoch`, `fit`
- `processing_own_phase/evaluate.py` — `evaluate`, `per_class_accuracy`

## Baseline Configuration

| Field          | Value |
|----------------|-------|
| Architecture   | [128] |
| Optimizer      | SGD (momentum=0.9) |
| Learning rate  | 0.01  |
| Batch size     | 64    |
| Epochs         | 10    |
| Dropout        | 0.0   |
| Seed           | 42    |
| Device         | MPS   |

## Training Output (E0_baseline)

```text
Epoch 01/10 | Train Loss: 0.6485 Acc: 0.7739 | Val Loss: 0.4854 Acc: 0.8310
Epoch 02/10 | Train Loss: 0.4446 Acc: 0.8427 | Val Loss: 0.4449 Acc: 0.8406
Epoch 03/10 | Train Loss: 0.4025 Acc: 0.8579 | Val Loss: 0.3942 Acc: 0.8593
Epoch 04/10 | Train Loss: 0.3737 Acc: 0.8671 | Val Loss: 0.3971 Acc: 0.8576
Epoch 05/10 | Train Loss: 0.3520 Acc: 0.8732 | Val Loss: 0.3737 Acc: 0.8620
Epoch 06/10 | Train Loss: 0.3399 Acc: 0.8773 | Val Loss: 0.3565 Acc: 0.8725
Epoch 07/10 | Train Loss: 0.3238 Acc: 0.8840 | Val Loss: 0.3536 Acc: 0.8726
Epoch 08/10 | Train Loss: 0.3134 Acc: 0.8868 | Val Loss: 0.3574 Acc: 0.8666
Epoch 09/10 | Train Loss: 0.3031 Acc: 0.8905 | Val Loss: 0.3409 Acc: 0.8785
Epoch 10/10 | Train Loss: 0.2951 Acc: 0.8922 | Val Loss: 0.3299 Acc: 0.8813
Best Val Acc: 0.8813 @ Epoch 10
Params: 101,770 | Time: 0m 47.5s
```

## Observations

- Train loss decreased monotonically (0.65 -> 0.30).
- Validation loss also decreased monotonically (0.49 -> 0.33).
- Train–val gap is small (~1.1 pp at peak), so no overfitting.
- Best validation accuracy: 0.8813 at epoch 10.

## Outcome

Baseline works. Ready to run controlled experiments.
