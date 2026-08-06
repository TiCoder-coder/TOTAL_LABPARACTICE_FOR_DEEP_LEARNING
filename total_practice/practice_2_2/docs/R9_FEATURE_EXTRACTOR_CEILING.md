# R9 Feature Extractor Ceiling

## Scope

R9 implements a controlled ResNet18 versus EfficientNet-B0 architecture comparison. It does not run model-quality experiments because R8 has not authorized a candidate, R6 has not selected a transform, the Train manifest remains unauthorized, and the required pretrained checkpoint authorities are unavailable.

No dataset image, Validation content, or Test content is read. No optimization step is executed. R10 is not executed.

## Architecture Registry

`A0` is the ResNet18 baseline at resolution 224. `A1` changes only the architecture to EfficientNet-B0 while retaining resolution 224, seeds `42`, `123`, and `2026`, the same split requirement, the same selected transform requirement, and the same staged transfer-learning hyperparameters from R8.

Higher-resolution input is disabled because it would be a second controlled experiment. OCR and image-text features remain disabled until the image-only baseline is stable.

## Staged Compatibility

Both architectures support head-only warmup followed by final feature-block fine-tuning. ResNet18 exposes `layer4` and `fc`; EfficientNet-B0 exposes `features.7`, `features.8`, and `classifier`. Frozen BatchNorm layers remain in evaluation mode.

Both adapters use AdamW with head learning rate `5e-4`, backbone learning rate `2e-5`, and weight decay `2e-4`. Checkpoint reload reconstructs each architecture with `weights=None`, strict-loads the complete state dict, and verifies identical output tensors offline.

## Engineering Profiles

The verifier reports total and trainable parameter counts, parameter memory, buffer memory, summed forward-activation bytes, output shape, and CPU latency for the same synthetic `1x3x224x224` input. These values are architecture engineering profiles only and are not Accuracy, Macro F1, or model-selection evidence.

## Validation Selection Contract

The model-quality selector requires all A0/A1 runs across the three declared seeds. Each architecture must share one initialization hash across its seeds. Result rows may contain only declared Train and Validation fields.

EfficientNet-B0 advances only when mean Validation Macro F1 improves, mean Validation Accuracy does not materially decrease, and the mean generalization gap decreases. A comparator that only raises Train Accuracy is rejected. Any Test field or incomplete seed coverage fails closed.

Synthetic selection fixtures verify the decision logic but cannot authorize an architecture.

## Gate Result

All R9 protocol checks pass. The overall gate remains blocked because the prerequisite R8 candidate, selected transform, authorized Train manifest, pretrained states, frozen EfficientNet checkpoint hash, and repeated-seed Validation results are unavailable.

No architecture is authorized. The canonical notebook and source images remain unchanged.
