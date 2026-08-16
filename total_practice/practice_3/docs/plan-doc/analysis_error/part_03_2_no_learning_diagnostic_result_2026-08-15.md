# PART 3.2 — No-Learning Diagnostic Result

Date: 2026-08-15  
Status: diagnostic isolation completed; no final fix implemented.

## Fixture and isolation

- Exactly 64 locked v2 Train records: 32 class 0 and 32 class 1.
- Deterministic source-ID fingerprint: `081608817a0cd3ce881dd95f0c9dc0100b49feb7f2862ece0891de3c0fc69349`.
- Seed 42, batch 16, max length 80, 120 steps, linear schedule, AdamW, no warmup.
- No v2 Validation evaluation, Holdout materialization/evaluation, or official Test access.
- Every case used a fresh model and wrote runtime files only beneath auto-removed `/private/tmp` directories.

## Summary matrix

| Case | Final loss | Final accuracy | Prediction counts 0/1 | Tiny overfit >=90% |
|---|---:|---:|---:|---|
| MPS full, 2e-5 | 0.678655 | 0.578125 | 39 / 25 | No |
| CPU full, 2e-5 | 0.681573 | 0.578125 | 25 / 39 | No |
| MPS full, 5e-5 | 0.691740 | 0.562500 | 56 / 8 | No |
| MPS full, 1e-4 | 0.693119 | 0.500000 | 64 / 0 | No |
| MPS eager full, 1e-4 | 0.693018 | 0.500000 | 64 / 0 | No |
| MPS eager head-only, 1e-4 | 0.688885 | 0.546875 | 37 / 27 | No |

All comparable cases started at accuracy 0.50, loss approximately 0.694112, prediction counts 64/0, logits mean approximately -0.0650, standard deviation 0.02350, and mean absolute class margin 0.04510.

## A. MPS versus CPU

The device-only comparison kept fixture, seed, initialization, LR `2e-5`, batch size, optimizer, schedule, trainable scope and steps fixed.

| Measurement | MPS | CPU |
|---|---:|---:|
| Final loss | 0.678655 | 0.681573 |
| Final accuracy | 0.578125 | 0.578125 |
| Final class counts 0/1 | 39 / 25 | 25 / 39 |
| Classifier delta L2 | 0.004231 | 0.004266 |
| Pre-classifier delta L2 | 0.088524 | 0.089810 |
| Backbone delta L2 | 0.148804 | 0.118466 |
| Final classifier grad norm | 4.11050 | 4.37837 |
| Final pre-classifier grad norm | 3.68583 | 3.81726 |
| Final representative backbone grad norm | 0.17391 | 0.14029 |
| Final logits std | 0.13809 | 0.10390 |
| Final mean absolute class margin | 0.23791 | 0.17475 |

Both devices failed identically by Train accuracy criterion. Numerical trajectories differ, as expected across backends, but CPU does not restore learning. **Material MPS-versus-CPU difference: NO.** MPS alone is not supported as the main cause.

## B. Learning-rate diagnostic

MPS full fine-tuning changed only LR:

| LR | Final loss | Final accuracy | Counts 0/1 | Classifier delta L2 | Backbone delta L2 |
|---:|---:|---:|---:|---:|---:|
| 2e-5 | 0.678655 | 0.578125 | 39 / 25 | 0.004231 | 0.148804 |
| 5e-5 | 0.691740 | 0.562500 | 56 / 8 | 0.010842 | 0.256086 |
| 1e-4 | 0.693119 | 0.500000 | 64 / 0 | 0.021574 | 0.505916 |

Higher LR produces proportionally larger parameter movement but worse loss/accuracy and stronger collapse. **Higher LR materially improves learning: NO.** This is evidence against insufficient LR magnitude as the sole cause.

## C. Full fine-tuning versus head-only

The default MPS head-only attempt failed before its first optimizer step with:

```text
NotImplementedError: scaled_dot_product_attention for MPS does not support dropout
```

This is a real PyTorch/Transformers/MPS compatibility issue exposed when the backbone is frozen, but it did not occur in official full fine-tuning and therefore does not explain the completed run by itself.

To preserve a one-factor scope comparison, both full and head-only controls were repeated with the same MPS eager-attention implementation at LR `1e-4`:

| Measurement | Full | Head-only |
|---|---:|---:|
| Final loss | 0.693018 | 0.688885 |
| Final accuracy | 0.500000 | 0.546875 |
| Counts 0/1 | 64 / 0 | 37 / 27 |
| Trainable parameters | 66,955,010 | 592,130 |
| Classifier delta L2 | 0.021701 | 0.020665 |
| Pre-classifier delta L2 | 0.446246 | 0.438914 |
| Backbone delta L2 | 0.511135 | 0.000000 |

Head-only is slightly better but remains far below the 90% overfit criterion. Concentrating optimization on the randomly initialized head does not resolve the issue.

## D. Gradient, parameter and logit evidence

Initial gradient probe norms for full cases were stable across CPU/MPS:

- classifier approximately 2.5484;
- pre-classifier approximately 2.5759;
- representative backbone approximately 0.04052.

Final gradients remained finite/nonzero for every trainable section. Representative final full-model backbone gradient norms were 0.17391 (`2e-5`), 0.00925 (`5e-5`) and 0.000284 (`1e-4`). All inspected trainable tensors changed. In the head-only control, backbone gradient and delta were correctly zero while head gradients/deltas were nonzero.

Logits remained finite in every case. At higher full-model LR, logits variance and class margin collapsed instead of separating classes:

- `2e-5`: std 0.13809, mean absolute margin 0.23791;
- `5e-5`: std 0.01208, mean absolute margin 0.01860;
- `1e-4`: std 0.00692, mean absolute margin 0.01377.

This demonstrates active backward/optimization with an unhealthy representation/output trajectory, not a skipped-gradient or unchanged-weight condition.

## Additional data-order check

The fixture is stored as 32 class-0 then 32 class-1 records, but the Trainer dataloader yielded mixed batches (8/8, 6/10, 10/6, 8/8 for the inspected epoch). Missing shuffle or single-class batches are therefore rejected as the cause.

## Diagnostic conclusion

- MPS versus CPU: no material performance difference at locked LR.
- Learning-rate strength: increasing to `5e-5` and `1e-4` does not help.
- Head-only: does not overfit under the matched eager control.
- Gradient/logit path: gradients and parameter updates exist, but higher LR drives logits toward an almost constant output.
- Tiny subset successfully overfit: **NO**.
- Root cause isolated to one component: **NO**.

The most likely category is a deeper incompatibility or ineffective interaction in the installed training stack/model construction and optimization behavior (`transformers 5.14.1`, `torch 2.13.0`, generic DistilBERT MLM checkpoint plus newly initialized classification head). Evidence is insufficient to name one library or line of code as the root cause. The MPS SDPA/dropout failure is confirmed for frozen-backbone training but is a separate issue from the official full fine-tuning collapse.

## Recommended correction — not implemented

Do not run the remaining official LR experiments. First approve a new correction plan that:

1. creates a minimal known-easy Train-only fixture with an independently verifiable label signal;
2. verifies loss reduction using a direct PyTorch reference loop against Trainer on CPU;
3. compares a known-compatible stable Transformers/PyTorch environment before changing project methodology;
4. explicitly sets a supported attention implementation for any frozen-backbone diagnostic;
5. only after the training-stack fault is reproduced and isolated, proposes a version/config/code revision and a new protocol version rather than mutating `practice_3_v2.0`.

## Safety record

- Official protocol modified: false
- Real registry modified: false
- `p3v2_lr_2e-5` evidence modified/reset: false
- `p3v2_lr_3e-5` executed: false
- `p3v2_lr_5e-5` executed: false
- Holdout accessed/materialized: false
- Official Test accessed: false
- Winner selected: false
- Final fix implemented: false
