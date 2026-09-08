# PERSISTENCE-v1

PERSISTENCE_LAST_VALUE is the canonical task-level temporal baseline for one-step Appliances forecasting.

The formula is y_hat(t+1) = y(t). Source and target values remain in raw Wh.

The evaluation uses the full WINDOWPOP-v1 Validation population anchored to L144 under WB0_CONTEXT_CARRY_OVER.

MAE, RMSE and R2 are computed once over the full Validation split with METRICS-v1.

No feature variant, scaler, DataLoader, training process, seed, optimizer or checkpoint participates in the prediction.

Test targets and Test metrics remain locked until Phase 47.
