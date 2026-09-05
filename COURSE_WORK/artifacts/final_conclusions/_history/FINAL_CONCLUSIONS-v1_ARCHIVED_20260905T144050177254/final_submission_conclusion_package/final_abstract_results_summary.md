# Abstract — Results Summary

The final Transformer was evaluated on a frozen chronological held-out Test set
(N = 2961) using three predefined seeds. It achieved a three-seed mean MAE
of 28.53 Wh, RMSE of 63.83 Wh,
and R² of 0.506. The Persistence baseline was reported
as a comparison reference. Rolling-origin analysis indicated development
robustness (development evidence). Error diagnostics showed larger errors in
high-consumption regimes. Last-query attention emphasized recent historical
positions. Head comparison showed heterogeneous temporal profiles. Across seeds,
layer-level head-mean attention was more consistent than individual heads.
These findings are descriptive for this single-household dataset; attention is
interpreted as temporal allocation rather than causal feature importance.
