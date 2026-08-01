# Phase 5 — Preprocessing and Leakage Control

## Notebook location

- Notebook: [demo_practice_2.ipynb](../../notebooks/demo_practice_2.ipynb)
- Main cells: **Cells 10–11**
- Source: [data.py](../../processing_own_phase/data.py)
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Objective

This phase converts CIFAR-10 images into tensors compatible with ImageNet-pretrained ResNet18 while ensuring that random augmentation is restricted to Train and that Validation/Test information cannot influence learned preprocessing.

## 2. Required ordering

The implementation follows this sequence:

```text
Raw CIFAR-10 training pool
        ↓
Fix Train/Validation indices using the configured seed
        ↓
Create separate dataset objects for each split
        ↓
Attach the Train transform or deterministic evaluation transform
        ↓
Execute transforms lazily when __getitem__ accesses an image
```

The central point is that **sample membership is fixed before preprocessing is attached**. TorchVision transformations do not modify all images in advance; they run lazily when the DataLoader requests a sample.

## 3. Train preprocessing

Train uses stochastic augmentation to increase input variation during optimization:

1. Resize to approximately 1.14 times the target resolution.
2. `RandomCrop` to `224 × 224`.
3. `RandomHorizontalFlip`.
4. `ColorJitter` for controlled brightness, contrast, saturation, and hue variation.
5. `ToTensor`.
6. Normalize with ImageNet mean and standard deviation.

Because the transform is random and lazy, the same Train image can yield a slightly different tensor in different epochs while retaining the same label.

## 4. Validation and Test preprocessing

Validation and Test share the deterministic evaluation policy:

1. Resize to approximately 1.14 times the target resolution.
2. `CenterCrop` to `224 × 224`.
3. `ToTensor`.
4. Normalize using the same ImageNet constants.

They contain no `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`, or other random augmentation. Therefore, repeated evaluation is not affected by a random image transform.

## 5. Interpreting “fit preprocessing on Train only”

This project does not use StandardScaler, PCA, a learned vocabulary, or a label encoder fitted across the dataset. ImageNet mean and standard deviation are fixed constants associated with the pretrained backbone; they are not estimated from CIFAR-10 Train, Validation, or Test.

If a learned preprocessing component is introduced later, the required protocol is:

1. call `fit` on Train only;
2. transform Train, Validation, and Test using the fitted Train parameters;
3. never call `fit` or `fit_transform` on Validation or Test.

## 6. Why separate dataset objects matter

In TorchVision, `transform` normally belongs to the dataset object. If Train and Validation shared one dataset instance, assigning a transform for one split could change the other split as well.

The implementation creates independent CIFAR-10 objects for Train and Validation and then applies their separate index lists. This guarantees:

- Train augmentation cannot be applied to Validation accidentally;
- the Validation transform cannot overwrite Train augmentation;
- sample membership remains independent from transform behavior.

## 7. Cell 11 table and assertions

Cell 11 reports:

- split name;
- dataset size;
- dataset-object identity;
- transform composition;
- whether random augmentation is present;
- whether the split is used for selection.

Automated checks confirm that:

- Train contains `RandomCrop`, `RandomHorizontalFlip`, and `ColorJitter`;
- Validation contains no random augmentation;
- Validation and Test have equivalent deterministic transforms;
- Train and Validation use separate backing objects.

See [test_data.py](../../tests/test_data.py).

## 8. Visual output

Transform inspection is displayed directly in Cell 11 and is not stored as a separate canonical PNG. Visual inspection is a sanity check; the primary leakage evidence is code order, disjoint indices, dataset-object identity, and automated tests.

## 9. Residual risks

- Excessive augmentation can destroy information in a `32 × 32` source image.
- Upscaling to `224 × 224` does not create new visual detail; it only matches the pretrained backbone input convention.
- Normalization that does not match pretrained weights can reduce transfer-learning effectiveness.
- Repeatedly inspecting Test examples and changing the pipeline based on them can create human-in-the-loop leakage even if the code remains split correctly.

## 10. Suggested presentation script

> I fix Train and Validation indices before attaching transforms. Train and Validation use independent dataset objects. Train has crop, horizontal flip, and color jitter, while Validation and Test share one deterministic resize, center-crop, tensor, and normalization policy. Transforms execute only when an image is accessed, so the dataset is not preprocessed globally before splitting.

## 11. Transition to the next phase

[Phase 6](phase_06_model_building.md) constructs ResNet18 and defines exactly which parameter groups are trainable in each strategy.
