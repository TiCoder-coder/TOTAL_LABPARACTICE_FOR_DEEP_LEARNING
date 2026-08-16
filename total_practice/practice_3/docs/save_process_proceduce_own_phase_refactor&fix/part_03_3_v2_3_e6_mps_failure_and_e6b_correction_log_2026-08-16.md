# Practice 3 v2.3 — E6 MPS Failure and E6b Correction Log

**Date:** 2026-08-16

## Finding: E6_staged_finetune Failed at Step 0

### Error
`NotImplementedError: scaled_dot_product_attention for MPS does not support dropout.`

### Root Cause
Recent transformers (5.x) changed the default attention backend from `eager` to `sdpa`.
The MPS SDPA implementation does not support non-zero attention_dropout in training mode.
This manifests only during Trainer.train() — not during eval/inference — which is why
the previous ZTV did not expose it.

### Attention Backend Audit
| Run  | _attn_implementation | Outcome            |
|------|---------------------|--------------------|
| E1   | sdpa                | COMPLETED          |
| E4   | sdpa                | COMPLETED          |
| E5b  | sdpa                | COMPLETED          |
| E6   | sdpa                | FAILED @ step 0    |

### E6 Preserved as FAILED
status = FAILED
error_type = NotImplementedError
error_message = scaled_dot_product_attention for MPS does not support dropout.
runtime_seconds = 0.22
stopped_epoch = null
epochs_completed = 0

## Correction: E6b_staged_finetune

### Change Applied
attn_implementation = "eager" — an execution-backend compatibility field.
The controlled experimental variable (staged fine-tuning) is identical.

### Attention dropout unchanged: PASS
Default attn_dropout = 0.1
Eager   attn_dropout = 0.1

### E6b Configuration
- learning_rate = 1e-05
- weight_decay = 0.01
- seq_classif_dropout = 0.20 (DistilBERT default)
- attn_implementation = eager
- fine_tuning_strategy = staged
- max_epochs = 15, patience = 4, warmup_steps = 720, label_smoothing = 0.1

### Staged Strategy
Stage 1 (Epochs 0-1): Freeze distilbert.embeddings + layers 0-3
  frozen=52187136, trainable=14767874
  Optimizer coverage: PASS

Stage 2 (Epoch 2+): Full unfreeze via optimizer.add_param_group()
  trainable=66955010, frozen=0
  Optimizer coverage: PASS

Stage 3: NONE

### MPS train-mode forward: PASS
model.train() + dropout active (no backward, no optimizer.step)
loss=0.6502, logits=(16, 2), finite=True

### ZTV Policy Update
All future Transformer/MPS pre-run validations must include a model.train()
forward pass (no backward). Inference-only ZTV is insufficient for detecting
training-time dropout incompatibility on MPS.
