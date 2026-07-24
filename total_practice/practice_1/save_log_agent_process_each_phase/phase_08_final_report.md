# Phase 8 — Final Summary Report

**Date:** 2026-07-24  
**Agent:** Main Agent (Cursor)

## Objective

This exercise implements a complete PyTorch workflow for classifying FashionMNIST images using a multilayer perceptron.

## Dataset

- 28×28 grayscale images.
- 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- 60,000 training images and 10,000 test images.
- Split: 54,001 train / 5,999 validation / 10,000 test.
- Transform: `ToTensor()` (float32 in [0, 1]).

## Model

- `Flatten` -> `Linear(784, 256)` -> `ReLU` -> `Dropout(0.2)` -> `Linear(256, 128)` -> `ReLU` -> `Dropout(0.2)` -> `Linear(128, 10)`.
- 235,146 trainable parameters.
- Output: raw logits (no softmax).
- Loss: `CrossEntropyLoss`.

## Training Setup

- Optimizer: Adam, lr=0.001.
- Batch size: 64.
- Epochs: 10.
- Device: MPS (Apple Silicon).
- Seed: 42.

## Experiments

| ID          | Architecture | Optimizer | LR   | Dropout | Best Val Acc | Best Epoch |
|-------------|--------------|-----------|------|---------|--------------|------------|
| E0_baseline | [128]        | SGD       | 0.01 | 0.0     | 0.8813       | 10         |
| E1_lr_low   | [128]        | SGD       | 0.001| 0.0     | 0.8431       | 10         |
| E2_lr_high  | [128]        | SGD       | 0.1  | 0.0     | 0.8585       | 7          |
| E3_deeper   | [256, 128]   | SGD       | 0.01 | 0.0     | 0.8845       | 10         |
| E4_dropout  | [256, 128]   | SGD       | 0.01 | 0.2     | 0.8801       | 8          |
| E5_adam     | [256, 128]   | Adam      | 0.001| 0.2     | **0.8858**   | 10         |

## Results

- **Best validation accuracy**: 0.8858 (E5_adam).
- **Test accuracy**: 0.8817.
- **Test loss**: 0.3286.
- **Best epoch (validation)**: 10.
- **Total pipeline time**: 5m 19.6s.

## Error Analysis

- The MLP performs best on clothing with distinctive silhouettes (Trouser 0.976, Bag 0.971, Sandal 0.972).
- It struggles with overlapping tops (Shirt 0.747, Pullover 0.738, T-shirt/top 0.796, Coat 0.804).
- High-confidence errors are dominated by these top classes — a CNN would likely improve these because it can use spatial structure.

## Conclusion

- The best model is **E5_adam** (deeper MLP + Adam + dropout).
- It improves over the baseline by **+0.45 pp** (0.8858 vs 0.8813).
- Adding dropout (E4 vs E3) did not help here at 10 epochs.
- Adam beats SGD on this short 10-epoch run.
- A natural next step is a CNN, which should lift the tops classes significantly.

## Verification

- Reloaded model produces identical predictions on the same batch (max logit diff = 0.0).
- Test set was used exactly once.
- All numbers in this report come from the actual training run, not pre-computed values.
