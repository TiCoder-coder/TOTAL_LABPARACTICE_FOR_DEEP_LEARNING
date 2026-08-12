# Phase 6 — Model Building and Baseline

## Notebook location

- Notebook: [practice_2_presentation.ipynb](../../notebooks/practice_2_presentation.ipynb)
- Main cells: **Cells 12–13**
- Source: [model.py](../../processing_own_phase/model.py)
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Architecture

The model uses **ImageNet-pretrained ResNet18** as its backbone. The original 1,000-class ImageNet fully connected layer is replaced by a linear classifier with ten outputs corresponding to the ten CIFAR-10 classes.

```text
Preprocessed RGB image
       ↓
ResNet18 feature extractor
       ↓
Global average pooling
       ↓
Linear classifier: feature vector → 10 logits
       ↓
CrossEntropyLoss during training / argmax during prediction
```

The network returns raw logits. Softmax is not applied before CrossEntropyLoss because the loss includes the appropriate log-softmax computation. Softmax is used later only when prediction probabilities or confidence values are required.

## 2. Transfer-learning rationale

Pretrained weights provide reusable visual features learned from ImageNet. Although CIFAR-10 differs in resolution and class vocabulary, low-level features such as edges, textures, and shapes can transfer. Fine-tuning then adapts higher-level representation to the target dataset.

The original classifier cannot be reused because it predicts 1,000 ImageNet classes, so the project initializes a new ten-class head.

## 3. Fine-tuning strategies

The implementation supports several modes, but the official controlled comparison uses only the following two.

### E1 — `head_only`

- Freeze all backbone parameters.
- Train only the final classifier.
- Use relatively few trainable parameters.
- Reduce compute and overfitting risk.
- Limit adaptation because pretrained features remain mostly fixed.

### E2 — `partial_finetune`

- Freeze early and middle backbone blocks.
- Unfreeze ResNet18 `layer4` and the classifier.
- Adapt high-level features to CIFAR-10.
- Require more computation and memory than E1.

Full fine-tuning and additional architectures may exist for exploratory work, but they are not part of the official E1/E2 controlled comparison.

## 4. Trainable-parameter evidence

The notebook reports:

- total parameter count;
- trainable parameter count;
- frozen parameter count;
- trainable ratio;
- the layer groups enabled by the selected strategy.

These values demonstrate that E1 and E2 differ in actual model configuration, not only in an experiment label.

## 5. Model sanity checks

Before training, a synthetic batch is used to verify:

- input shape is `[batch_size, 3, 224, 224]`;
- output shape is `[batch_size, 10]`;
- logits contain finite values;
- Cross Entropy loss can be calculated and is finite;
- backward propagation produces gradients for trainable parameters;
- frozen parameters have `requires_grad=False`.

These checks catch classifier-dimension, device, and preprocessing mismatches before a multi-epoch run begins.

## 6. BatchNorm caveat

Setting `requires_grad=False` freezes trainable parameters, but it does not automatically freeze BatchNorm running statistics if the entire model is placed in `train()` mode. Therefore, `head_only` means only the head receives parameter gradients, but some frozen-backbone buffers may still change.

For a stricter experiment in future work, BatchNorm modules in frozen blocks should remain in `eval()` mode during head training. Both E1 and E2 would need to be rerun under the same declared policy before comparing new results.

This caveat does not cause Test leakage, but it affects the exact interpretation of “frozen backbone.”

## 7. Phase output

The phase produces model and parameter summaries rather than final metrics. The checkpoint becomes official only after Validation-only selection and verification in Phase 9.

Selected checkpoint:

- [selected best_val_loss.pt](../../runs/E2_resnet18_partial_2b5b94de/best_val_loss.pt)
- [locked selection metadata](../../outputs/hyperparameter_selection_locked.json)

## 8. Suggested presentation script

> I use an ImageNet-pretrained ResNet18 and replace its classifier with a ten-logit layer. E1 trains only the classifier, while E2 trains ResNet18 layer4 plus the classifier. Before training, I verify output shape, finite loss, gradient flow, and trainable parameter counts to prove that each strategy is configured correctly.

## 9. Transition to the next phase

[Phase 7](phase_07_training.md) documents the Train/Validation loop, history fields, scheduler, Early Stopping, and checkpoint behavior.
