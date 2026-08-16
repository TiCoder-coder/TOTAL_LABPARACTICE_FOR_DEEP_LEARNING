# Process Log: Final Validation Ranking
- **Files Inspected**: All v2.3 `experiment_config.json` and `experiment_result.json` files.
- **Valid Runs**: E1, E2, E3, E4, E5b, E6c
- **Excluded Runs**: E5 (invalid dropout), E6 (MPS SDPA crash), E6b (Optimizer mismatch).
- **Ranking Result**: 1. E4, 2. E1, 3. E5b, 4. E6c, 5. E3, 6. E2.
- **Winner**: `E4_weight_decay_0.05`
- **Visualization Outputs**: 9 plots generated in `figures/`.
- **Verification**: Holdout remains SEALED (0 access). Trainer.train() was NOT called. No checkpoints regenerated.
