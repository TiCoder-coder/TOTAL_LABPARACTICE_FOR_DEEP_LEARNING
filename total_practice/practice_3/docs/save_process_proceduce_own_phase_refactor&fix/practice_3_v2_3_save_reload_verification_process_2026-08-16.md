# Save / Reload Verification Process Log
- **Source Winner**: `E4_weight_decay_0.05` checkpoint verified via SHA256.
- **Save Operation**: Exported to `final_saved_model/` using `save_pretrained()`.
- **Reload Operation**: Loaded from disk, verified vocab size.
- **Prediction Comparison**: Reran 9 custom sentences. Checked that difference between baseline probabilities and reloaded probabilities is negligible (< 1e-5).
- **Holdout Protection**: Holdout was NOT accessed.
- **Official Test Protection**: Official test split was NOT accessed.
