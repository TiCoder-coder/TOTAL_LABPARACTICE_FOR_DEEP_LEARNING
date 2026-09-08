# Conclusions (Short Version)

The final Transformer Encoder was evaluated on a frozen chronological Held-Out
Test set (FINAL_TEST_POP-v1, N = 2961) using three predefined seeds (42,
123, 2026). It achieved a three-seed mean MAE of 28.53 Wh,
RMSE of 63.83 Wh, and R² of 0.506.
These are mean ± SD of independent runs and do not represent an ensemble.

The Persistence baseline was competitive; the Tuned LSTM was not evaluated on
FINAL_TEST_POP-v1 due to lookback mismatch. Rolling-origin evidence (Phase 44)
indicates development robustness but is not a second Test result.

Forecast errors were larger in high-consumption and rapid-change regimes.
Last-query attention emphasized recent temporal positions. Within-seed head
comparison showed non-identical profiles. Error-conditioned analysis showed
associations between attention and error magnitude, with HIGH/LOW cohorts
as post-hoc diagnostics only. Cross-seed layer head-mean attention was more
consistent than individual matched heads.

Key limitations: single household, H=1 one-step scope, WB0 boundary,
sequential tuning, three seeds, time-series dependence, and attention as a
temporal diagnostic only (not causal). Results do not generalize to other
households or support deployment claims.
