# Phase 8 — Save & Load Log

**Date:** 2026-07-24  
**Agent:** Main Agent (Cursor)

## Files Created

- `processing_own_phase/save_load.py` — `save_checkpoint`, `load_checkpoint`, `load_model_from_checkpoint`, `verify_loaded_model`

## Checkpoint Format

```python
{
    "model_state_dict": ...,
    "hidden_dims": [256, 128],
    "dropout": 0.2,
    "num_classes": 10,
    "class_names": [...],
    "config": {...},
    "best_epoch": 10,
    "best_val_acc": 0.8858,
    "best_val_loss": 0.3109,
    "extra": {"test_accuracy": 0.8817, "test_loss": 0.3286},
}
```

## Verification

```text
Saved: /Users/ticoder-coder/Documents/DEEP_LEARNING/LAB&PRACTICE/total_practice/practice_1/outputs/best_model.pth
Same predictions after reload: True
Max logit difference: 0.00e+00
```

The reloaded model produces **identical predictions** to the original on the same input batch (max logit diff = 0.0). This confirms the checkpoint round-trips perfectly.

## Files Saved

- `outputs/best_model.pth` — best model (E5_adam) with metadata
- `outputs/E0_baseline.pt` ... `E5_adam.pt` — per-experiment checkpoints
- `outputs/summary.json` — final summary dict

## Outcome

Model save/load verified. Best model is ready for downstream use.
