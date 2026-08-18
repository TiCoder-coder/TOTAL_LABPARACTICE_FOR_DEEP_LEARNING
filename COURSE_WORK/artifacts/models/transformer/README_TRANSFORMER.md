# Transformer Implementation (TRANSFORMER_IMPL-v1)

Attention-aware Transformer encoder regressor for multivariate windowed forecasting.

- Input layout `[B, L, F]` with sinusoidal positional encoding
- Post-norm `AttentionAwareEncoderLayer` stack
- `forward(x)` for training; `forward_with_attention(x)` for inspection
- Per-head attention tensors `[B, H, L, L]` without implicit averaging
- Phase 16 validates architecture only; official training starts in Phase 21
