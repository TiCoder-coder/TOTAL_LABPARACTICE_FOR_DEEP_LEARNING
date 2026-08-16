# PART 3.2 — `p3v2_lr_2e-5` No-Learning Analysis

Date: 2026-08-15  
Scope: root-cause audit only. No official run, registry, protocol, Holdout, or Test mutation.

## Observed real-run evidence

The completed run stopped at epoch 3 and selected epoch 1 / `checkpoint-480`:

| Epoch | Train loss | Validation loss | Accuracy | Precision | Recall | F1 | LR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.696617 | 0.693150 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.8004167e-5 |
| 2 | 0.695528 | 0.693152 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.6004167e-5 |
| 3 | 0.693641 | 0.693697 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.4004167e-5 |

This is chance-level binary loss/accuracy with zero positive-class recall and F1.

## Checks and evidence

### Dataset and label pipeline — PASS

- Materialized v2 Train/Validation only: 7,676 / 960.
- Train labels: class 0 = 3,838; class 1 = 3,838.
- Validation labels: class 0 = 480; class 1 = 480.
- Raw and tokenized label counts match exactly.
- Tokenized records contain `labels`; labels remain a subset of `{0,1}`.
- Fresh and checkpoint configs both map `0 -> NEGATIVE`, `1 -> POSITIVE`.
- Dataset materializer reported official Test excluded and Holdout not indexed/materialized.

No evidence of missing labels, inverted mapping, class imbalance, or loss receiving the wrong label field was found.

### Validation prediction distribution — SINGLE-CLASS COLLAPSE

Read-only inference from authoritative `checkpoint-480` over the locked Validation split produced:

- predicted class 0: **960**
- predicted class 1: **0**
- TN = 480, FP = 0, FN = 480, TP = 0

The model collapsed completely to NEGATIVE. This exactly explains accuracy 0.50 and positive-class precision/recall/F1 of zero.

### Trainable parameters — PASS

- Total parameters: 66,955,010
- Trainable parameters: 66,955,010
- Frozen parameters: 0
- Trainable DistilBERT backbone: 66,362,880
- Trainable `pre_classifier`: 590,592
- Trainable `classifier`: 1,538

All inspected model parameters had `requires_grad=True`.

### Optimizer coverage — PASS

- Reconstructed optimizer includes every trainable tensor.
- Backbone, `pre_classifier`, and `classifier` are all covered.
- Configured/initial LR is `2e-5`; no zero-LR group exists.
- Epoch-1 saved optimizer has two groups at scheduled LR `1.8e-5`, both with initial LR `2e-5`.
- Saved optimizer contains state for 104 parameter tensors.

The separate tiny diagnostic with non-fused AdamW matched fused AdamW almost exactly, so current evidence does **not** identify fused AdamW as the cause.

### Model weights changed — YES

Fresh initialization was reproduced with seed 42 and compared with `checkpoint-480`:

| Tensor | Changed elements | Total | Mean absolute change | Max absolute change |
|---|---:|---:|---:|---:|
| `classifier.weight` | 1,536 | 1,536 | 0.00022368 | 0.00113618 |
| `pre_classifier.weight` | 589,823 | 589,824 | 0.00021915 | 0.00193097 |
| `distilbert.transformer.layer.0.attention.q_lin.weight` | 589,824 | 589,824 | 0.00018908 | 0.00109638 |

Fresh/checkpoint SHA-256 fingerprints differed for every inspected tensor. Training performed updates; this is not a completely skipped optimizer-step condition.

### Training history and Early Stopping — PASS as implemented

- Epoch/global steps are 1/480, 2/960, 3/1440.
- The linear LR schedule is present and nonzero.
- Epoch 1 has the minimum Validation loss, so it is correctly selected.
- Epochs 2 and 3 do not improve Validation loss by the frozen `1e-6` threshold.
- Patience 2 is therefore exhausted after two consecutive non-improving evaluations, explaining stop at epoch 3.

Early Stopping explains *when* the failed run stopped; it does not explain why learning never started.

## Tiny-overfit diagnostic — FAIL

A disposable diagnostic used 64 balanced samples from locked v2 Train only (32/32), no Validation, Holdout, or official Test, and wrote only beneath `/private/tmp`.

With the current `2e-5`, linear schedule, seed 42 and fused AdamW for 30 epochs / 120 steps:

- initial Train accuracy/loss: 0.5000 / 0.69411
- final Train accuracy/loss: 0.578125 / 0.67864
- final predictions: class 0 = 39, class 1 = 25

The model did not overfit the tiny set. A controlled non-fused AdamW repeat kept all other factors fixed and produced essentially the same result:

- final Train accuracy/loss: 0.578125 / 0.67840
- final predictions: class 0 = 39, class 1 = 25

This excludes full-data class balance and fused-vs-non-fused AdamW as sufficient explanations. It indicates ineffective learning under the current model/training/backend combination, but does not by itself isolate one defective component.

## Suspected causes after audit

1. Effective optimization under the locked model initialization, LR/schedule and MPS execution is too weak or otherwise unhealthy.
2. A Transformers 5.14.1 / PyTorch 2.13.0 / MPS interaction remains possible and has not been isolated against CPU/CUDA.
3. The generic DistilBERT checkpoint correctly creates a new random classification head; whether the locked LR/schedule can reliably bootstrap that head in this environment needs a controlled diagnostic matrix.
4. Early Stopping is aggressive once the first three Validation losses remain at chance, but it is downstream of the no-learning behavior rather than a demonstrated primary cause.

Rejected by evidence: label imbalance/corruption, missing tokenized labels, reversed mapping, frozen parameters, absent optimizer coverage, zero LR, no weight updates, and fused AdamW alone.

## Root-cause conclusion

**Root cause identified: NO.**

The failure mode is precisely identified as single-class collapse plus ineffective learning. The audit narrowed the problem to the effective training behavior of the current model/configuration/backend, but no single cause has sufficient evidence yet.

## Recommended next action

Do **not** start `p3v2_lr_3e-5` or `p3v2_lr_5e-5` yet. Prepare and approve a separate protocol-revision diagnostic plan that uses only a temporary balanced Train subset to isolate, one variable at a time:

1. MPS versus CPU on an identical small fixture;
2. locked LR versus a diagnostic-only higher head-learning-rate control;
3. full fine-tuning versus a short classifier/head-only diagnostic;
4. initial/final logits, gradient norms, and parameter deltas by model section.

Only after one cause is reproducibly isolated should code or frozen protocol be revised. Preserve the completed run as failed-learning evidence; never reset it to `PLANNED`.

## Integrity confirmation

- `p3v2_lr_2e-5` evidence overwritten/deleted: false
- Registry changed by audit: false
- Official runs launched by audit: false
- Holdout materialized/evaluated: false
- Official Test loaded: false
- Winner selected: false
- Frozen protocol modified: false
