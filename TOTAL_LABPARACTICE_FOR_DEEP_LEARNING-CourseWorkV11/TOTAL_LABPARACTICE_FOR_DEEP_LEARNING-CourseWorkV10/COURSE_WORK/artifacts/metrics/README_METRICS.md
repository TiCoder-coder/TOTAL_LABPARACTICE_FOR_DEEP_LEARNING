# METRICS-v1

METRICS-v1 is the single evaluation contract for Persistence, LSTM, Transformer and every controlled experiment.

MAE and RMSE are calculated in original Wh. R2 is dimensionless, may be negative and is explicitly undefined for constant targets or fewer than two samples.

Validation RMSE in Wh is the primary model-selection metric. MAE and R2 are required secondary evidence.

YS0 predictions are already in Wh. YS1 predictions must be inverse-transformed with the frozen TRAIN-only target scaler before evaluation.

Metrics are computed once on the complete aligned split population. Mean batch RMSE and mean batch R2 are prohibited.

Residual is y_true_wh minus y_pred_wh. Positive residual means underprediction and negative residual means overprediction.

Predictions are neither clipped nor rounded before metric calculation. Arrays are evaluated on CPU as NumPy float64.

TEST metrics require FINAL_TEST mode and a final model lock identifier. Phase 12 does not materialize Test targets or Test metrics.
