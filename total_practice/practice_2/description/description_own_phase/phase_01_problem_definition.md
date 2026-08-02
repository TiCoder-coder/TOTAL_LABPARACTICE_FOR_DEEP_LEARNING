# Phase 1 — Problem Definition

## Notebook location

- Presentation notebook: [demo_practice_2.ipynb](../../notebooks/demo_practice_2.ipynb)
- Main cell: **Cell 2** using 1-based numbering
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Task objective

Practice 2 addresses **supervised multiclass image classification** on CIFAR-10. Each input belongs to exactly one of ten classes:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

The original input is a `32 × 32` RGB image. The model produces ten raw logits, one per class, and the predicted class is the logit with the largest value. Because the problem is single-label multiclass classification, Cross Entropy is the appropriate loss function.

## 2. Technical questions

The notebook is designed to answer the following questions:

1. Can an ImageNet-pretrained ResNet18 classify CIFAR-10 effectively through transfer learning?
2. When every other factor is held constant, does `head_only` or `partial_finetune` perform better on Validation?
3. Can the selected checkpoint reproduce the Validation Accuracy stored during training?
4. After model decisions are locked, what is the final performance on the untouched official Test set?
5. Which classes are confused most often, and how confident is the model when it is correct or incorrect?

## 3. Scope and constraints

- Backbone: ImageNet-pretrained ResNet18.
- Dataset: CIFAR-10 loaded through TorchVision.
- Train and Validation are derived from the official 50,000-image training pool.
- The official 10,000-image Test set is excluded from selection.
- E1 and E2 differ only by fine-tuning strategy.
- Experiment and checkpoint selection use Validation only.
- Test is evaluated only after Validation Verification passes.

These constraints define a controlled comparison and reduce the risk of data leakage or Test-driven optimization.

## 4. Evaluation criteria

### Primary metrics

- **Validation Accuracy:** used for experiment and checkpoint selection.
- **Test Accuracy:** used only for final reporting.

### Supporting metrics

- Test Loss.
- Macro Precision.
- Macro Recall.
- Macro F1-score.
- Raw and normalized confusion matrices.
- Per-class classification report.

Macro metrics assign equal importance to each class. CIFAR-10 is balanced, but macro metrics are still valuable because they expose weak classes that overall Accuracy might hide.

## 5. Completion criteria

The workflow is complete only when all of the following conditions hold:

- Train, Validation, and Test are explicitly defined.
- Split indices are determined before preprocessing.
- Train and Validation do not share a transform-bearing dataset object.
- Test does not participate in model selection.
- E1 and E2 have genuine multi-epoch histories and learning curves.
- The best checkpoint is selected through Validation.
- Reloaded Validation performance matches the recorded value.
- Test is evaluated exactly once after verification.
- Metrics and figures are generated from artifacts rather than hardcoded values.

## 6. Phase output

This phase defines the experimental contract; it does not generate a metric or figure. Its output is the set of rules that prevents later phases from changing the objective or using Test improperly.

## 7. Suggested presentation script

> This project classifies ten CIFAR-10 classes with a pretrained ResNet18. I determine Train and Validation membership before preprocessing, use Validation to compare head-only and partial fine-tuning, reload and verify the selected checkpoint on Validation, and only then evaluate the official Test set once.

## 8. Transition to the next phase

[Phase 2](phase_02_environment_setup.md) establishes the environment, seed, project paths, and compute device required for reproducible execution.
