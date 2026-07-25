# PyTorch FashionMNIST Classification

## Project Overview

This project implements a complete PyTorch workflow for classifying FashionMNIST images into 10 apparel categories. The primary deliverable is `practice_1.ipynb`; `processing_own_phase` provides an equivalent reusable Python package and command-line pipeline.

The workflow covers tensors, datasets, DataLoaders, transforms, neural-network construction, autograd, optimization, validation-driven hyperparameter comparison, final test evaluation, visualization, TensorBoard tracking, and model saving/loading.

## Evaluation Protocol

FashionMNIST provides 60,000 official training images and 10,000 official test images.

The official training pool is split once with seed 42 using a class-stratified 90/10 split:

| Partition | Samples | Samples per class | Purpose |
|---|---:|---:|---|
| Training subset | 54,000 | 5,400 | Optimization and controlled experiments |
| Validation subset | 6,000 | 600 | Hyperparameter and epoch selection |
| Official test set | 10,000 | 1,000 | One final generalization evaluation |

The official test set does not influence preprocessing statistics, model architecture, optimizer choice, learning rate, dropout, augmentation, or epoch selection.

After validation selects a configuration and epoch count, that configuration is rebuilt from fresh state and trained on all 60,000 official training images. The resulting model is evaluated on the official test set.

## Preprocessing

EDA uses original training images converted with `ToTensor()`.

The class-stratified split is created before final transformed datasets and DataLoaders. Mean and standard deviation are computed only from the 54,000-image training subset.

| Dataset view | Transform |
|---|---|
| Baseline training | `ToTensor()` and training-subset normalization |
| Augmentation experiment | Horizontal flip, rotation up to 10 degrees, `ToTensor()`, normalization |
| Validation | Deterministic `ToTensor()` and normalization |
| Test | Deterministic `ToTensor()` and normalization |

Separate FashionMNIST dataset instances prevent validation samples from inheriting random training augmentation.

## Model

The default baseline is:

```text
Input [B, 1, 28, 28]
Flatten
Linear(784, 128)
ReLU
Linear(128, 10)
Output [B, 10] raw logits
```

The baseline has 101,770 trainable parameters.

The model does not apply Softmax because `CrossEntropyLoss` consumes raw logits. Constructor checks validate input dimension, class count, hidden dimensions, and dropout range. A fail-fast sanity test verifies output shape, finite logits/loss/gradients, and a real optimizer update without mutating the model used for training.

## Training

Every experiment creates a fresh model, optimizer, DataLoader shuffle generator, metric history, and TensorBoard writer.

Each batch performs:

```text
zero_grad
forward
cross-entropy loss
backward
optimizer step
metric accumulation
```

Epoch loss is weighted by the number of samples in every batch. Predictions use `argmax` without `.data`. Validation runs under `torch.inference_mode()`. The selected device priority is CUDA, then Apple MPS, then CPU.

## Controlled Experiments

| ID | Hidden dimensions | Dropout | Optimizer | Learning rate | Augmentation |
|---|---|---:|---|---:|---|
| E0_baseline | `[128]` | 0.0 | Adam | 0.001 | No |
| E1_deeper | `[256, 128]` | 0.0 | Adam | 0.001 | No |
| E2_dropout | `[256, 128]` | 0.2 | Adam | 0.001 | No |
| E3_sgd | `[256, 128]` | 0.0 | SGD | 0.01 | No |
| E4_augmentation | `[256, 128]` | 0.0 | Adam | 0.001 | Yes |

All experiments use the same split, seed, batch size and 10-epoch budget. The highest validation accuracy selects the configuration; validation loss breaks an exact accuracy tie. The test set is not accepted by the training API.

## Artifacts

The pipeline produces:

```text
outputs/
  experiments/
    E0_baseline.pth
    E1_deeper.pth
    E2_dropout.pth
    E3_sgd.pth
    E4_augmentation.pth
  fashion_mnist_model.pth
  summary.json
  loss_curve.png
  accuracy_curve.png
  experiment_comparison.png
  confusion_matrix.png
  predictions_grid.png
  data_samples.png
  class_distribution.png

runs/
  <run-session>/
    <experiment-id>/
```

The final checkpoint stores weights, architecture configuration, training configuration, normalization statistics, validation selection metrics, test metrics and class names. Reload verification requires identical predictions and numerically identical logits.

## Run

Select the project virtual environment as the Jupyter kernel, restart the kernel, and run `practice_1.ipynb` from top to bottom.

Run the reusable package from the workspace root:

```bash
./venv/bin/python -m total_practice.practice_1.processing_own_phase.main
```

Run a small integration check:

```bash
./venv/bin/python -m total_practice.practice_1.processing_own_phase.main --quick
```

## Verification

The implementation is accepted only when:

- The split is 54,000/6,000/10,000 with no train/validation overlap.
- Every class contributes 5,400 training and 600 validation samples.
- Validation and test transforms are deterministic.
- Model forward, finite loss, finite gradients and parameter update checks pass.
- Repeated fresh runs with the same seed reproduce training metrics in the same environment.
- Phase 7 never iterates the official test loader.
- Test loss is sample-weighted.
- Save/load verification produces identical predictions and logits.
- Notebook code cells compile and a clean execution completes without stale outputs.

## Current Results

The current results were reproduced by both the reusable package and the notebook definitions in clean Python processes on CPU with Python 3.10.11 and PyTorch 2.13.0.

| Experiment | Best epoch | Best validation loss | Best validation accuracy | Parameters |
|---|---:|---:|---:|---:|
| E0_baseline | 10 | 0.3509 | 88.45% | 101,770 |
| E1_deeper | 10 | 0.3308 | 89.30% | 235,146 |
| E2_dropout | 7 | 0.3201 | 88.65% | 235,146 |
| E3_sgd | 7 | 0.3204 | 89.07% | 235,146 |
| E4_augmentation | 10 | 0.3440 | 87.67% | 235,146 |

`E1_deeper` was selected by validation accuracy. It was rebuilt from fresh state and trained for 10 epochs on all 60,000 official training images.

| Final metric | Value |
|---|---:|
| Test loss | 0.3320 |
| Test accuracy | 88.84% |
| Macro precision | 88.86% |
| Macro recall | 88.84% |
| Macro F1 | 88.78% |
| Maximum reload logit difference | 0.00000000 |

The detailed machine-readable report is stored in `outputs/summary.json`; independent notebook parity evidence is stored in `outputs/notebook_verification.json`.
