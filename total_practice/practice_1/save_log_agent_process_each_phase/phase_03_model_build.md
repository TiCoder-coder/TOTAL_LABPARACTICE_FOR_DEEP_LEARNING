# Phase 3 — Model Build Log

**Date:** 2026-07-24  
**Agent:** Main Agent (Cursor)

## Files Created

- `processing_own_phase/model.py` — `FashionMLP`, `build_model`, `sanity_check_model`

## Architecture

```python
Flatten(784)
  -> Linear -> ReLU -> (optional Dropout) -> ...
  -> Linear(num_classes, 10)
```

Output is raw logits (no softmax). `CrossEntropyLoss` handles log-softmax internally.

## Baseline Model

```python
FashionMLP(hidden_dims=[128], dropout=0.0)
```

- Linear(784 -> 128): 100,480 params
- ReLU: 0 params
- Linear(128 -> 10): 1,290 params
- **Total: 101,770 trainable params**

## Sanity Checks

```text
Architecture: [128] | Dropout: 0.0
Parameters: 101,770
Forward shape: (64, 10)
Loss is finite: True (loss=2.3187)
Gradients populated: True
Params changed after step: True
```

All four checks passed:
1. Forward pass produces `[B, 10]` logits.
2. Loss is finite.
3. Backward populates gradients.
4. Optimizer step changes weights.

## Outcome

Model is built correctly. Sanity checks pass. Ready to train.
