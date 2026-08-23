# LSTM Implementation (LSTM_IMPL-v1)

Canonical unidirectional LSTM regression baseline for multivariate windowed forecasting.

- Input layout: `[B, L, F]` with `batch_first=True`
- Output layout: `[B, 1]` without output activation
- Stateless zero-initialized hidden state per forward pass
- Last-step readout and linear regression head
- Phase 15 validates architecture only; official training starts in Phase 20
