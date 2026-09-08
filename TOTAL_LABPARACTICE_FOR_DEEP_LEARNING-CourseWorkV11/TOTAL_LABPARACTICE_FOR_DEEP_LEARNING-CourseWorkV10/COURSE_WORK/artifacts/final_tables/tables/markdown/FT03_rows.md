## FT03 - Rolling-Origin Temporal Robustness (DEVELOPMENT_EVIDENCE)

_Pooled outer-fold RMSE is the primary Phase 44 robustness criterion._

| model_candidate | ro1_rmse_wh | ro2_rmse_wh | ro3_rmse_wh | pooled_rmse_wh | mean_fold_rmse_wh | fold_rmse_sd_wh | worst_fold_rmse_wh | role |
|---|---|---|---|---|---|---|---|---|
| PERSISTENCE_LAST_VALUE | N/A | N/A | N/A | 66.43 | N/A | N/A | N/A | Baseline |
| LSTM_TUNED_WINNER | 51.90 | 57.63 | 67.17 | 59.24 | 58.90 | 7.72 | 67.17 | Baseline |
| TR_C0_PRIMARY | 48.30 | 59.87 | 69.94 | 60.03 | 59.37 | 10.83 | 69.94 | N/A |
| TR_C1_ALT_WEIGHT_DECAY | 50.27 | 60.51 | 68.49 | 60.22 | 59.76 | 9.14 | 68.49 | N/A |
| TR_C2_ALT_LOOKBACK | 49.76 | 59.08 | 69.16 | 59.86 | 59.33 | 9.70 | 69.16 | Final-source candidate (Phase 45) |

> DEVELOPMENT_EVIDENCE only. Pooled RMSE is the primary criterion; mean fold RMSE is secondary.
