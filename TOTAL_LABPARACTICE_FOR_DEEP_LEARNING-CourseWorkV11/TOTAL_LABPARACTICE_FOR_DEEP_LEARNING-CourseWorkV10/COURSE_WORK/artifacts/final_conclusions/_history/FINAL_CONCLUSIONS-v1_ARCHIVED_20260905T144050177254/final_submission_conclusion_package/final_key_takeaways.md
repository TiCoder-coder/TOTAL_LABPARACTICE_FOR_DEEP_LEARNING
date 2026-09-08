# Key Takeaways

1. **Final Test performance** — On the frozen Held-Out Test (FINAL_TEST_POP-v1),
   the final Transformer achieved three-seed mean RMSE 63.83
   Wh and R² 0.506 (mean ± SD of seeds 42, 123, 2026,
   ddof=1; not an ensemble).

2. **Baseline comparison** — Persistence RMSE was 66.84
   Wh; the Tuned LSTM was not evaluated on FINAL_TEST_POP-v1 due to lookback
   mismatch (L36 vs L72).

3. **Temporal robustness** — Rolling-origin development evidence is available
   (FT03); it is not a second Test result.

4. **Error behavior** — Errors were larger in high-consumption and rapid-change
   regimes; worst errors are valid frozen observations.

5. **Attention** — Last-query attention concentrated on recent temporal lags
   (recent-1h, recent-6h); head profiles were non-identical within each layer.

6. **Seed stability** — Layer-level head-mean was more reproducible than
   individual matched heads; matching ambiguity and cycle consistency are reported.

7. **Limitations** — Single household; H=1; three seeds; sequential tuning;
   attention is temporal allocation only; no deployment claim.

8. **Future work** — External multi-house validation; multi-step forecasting;
   head ablation; additional seeds; prospective deployment evaluation.
