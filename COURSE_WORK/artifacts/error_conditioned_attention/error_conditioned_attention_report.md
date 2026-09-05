# Phase 56 — Error-Conditioned Attention Report

## 1. Objective
Phase 56 quantifies whether temporal attention of the Final Transformer changes systematically when forecast error is larger/smaller, underprediction or overprediction. Phase 56 is strictly diagnostic.

## 2. Why error-conditioned attention is diagnostic
Error and attention are both outcomes of the same forward prediction event. Conditioning on realized error does not create causal identification. All findings are descriptive association / co-occurrence.

## 3. Frozen attention/error sources
- Phase 49 residuals (`residual_long_table.csv`)
- Phase 50 regime assignment (`test_regime_assignment.csv`)
- Phase 51 shared hardness (derived from `residual_long_table.csv`: `SharedHardness_t = (|e_42|+|e_123|+|e_2026|)/3`)
- Phase 52 raw last-query NPZ + `attention_full_matrix_summary.csv`
- Phase 54 per-vector metrics (`last_query_metrics_long.csv`) + mean temporal profiles + layer head-mean profiles + recent-mass summary
- Phase 55 head behavior + head-pair comparison + layer diversity (handoff reference only)

## 4. Difference between Phase 50 regimes and Phase 56 error cohorts
Phase 50 regimes = Train-defined data regimes.
Phase 56 error cohorts = Test-relative diagnostic groups based on realized forecast error.
These two concepts must not be conflated.

## 5. Error cohort construction and freeze
Rank-based 20/60/20 with `n_edge = max(1, floor(0.20*N))`.
Tie-break: timestamp ASC, target_id ASC.
10 exact rank-based deciles (`decile = 1 + floor(10*r/N)`).
Cohort assignment SHA256 frozen BEFORE any attention metric/profile join.

Assignment SHA256: `fde852c7a72efe556efa3f6aef22c134694b447be4deb66dfb61f156fa3f28eb`

## 6. Continuous absolute-error associations
Per (seed, layer, head, metric) Spearman rho with `absolute_error_wh`.
Results stored in `error_attention_association_long.csv`.

## 7. Signed residual associations
Per (seed, layer, head, metric) Spearman rho with `residual_wh`.

## 8. Error-decile attention trends
Per (seed, layer, head, decile) metric summary + per-decile layer head-mean temporal profile.
Deciles are 1..10 deterministic.

## 9. High-error vs low-error metric differences
Per (seed, layer, head, metric) mean/median/quantiles + Cliff's delta.

## 10. High-error vs low-error temporal-profile shifts
Per (seed, layer, head) HIGH vs LOW profile JSD/L1/Cosine/Wasserstein (minutes).
`D_HL(k) = P_HIGH(k) - P_LOW(k)` summed over k should be ~0.

## 11. Underprediction vs overprediction attention
Per (seed, layer, head) UNDER vs OVER metric deltas + Cliff's delta + profile distances.
`UNDER = residual > 0`, `OVER = residual < 0`, `ZERO = residual == 0` (no epsilon).

## 12. Layer head-mean error-conditioned analysis
Permutation-invariant: mean 4 heads per target/layer FIRST, then recompute 6 CORE_ATTENTION_METRICS-v1 on the head-mean vector.
Cross-seed aggregation only after per-seed computation.

## 13. Shared-hardness common-target analysis
Per-seed shared-cohort layer summary + profile distances.
Same target IDs across seeds.

## 14. Full-matrix attention summary associations
Secondary Spearman with `mean_query_entropy`, `mean_self_attention_weight`, `mean_absolute_query_source_distance_steps`, `forward_within_input_mass`.
Forward-within-input mass is later-within-input attention, NOT future leakage.

## 15. Phase 50 regime composition context
Frozen Phase 50 labels are used only to describe cohort composition. No regime threshold is modified. No cartesian subgroup mining is performed.

## 16. Phase 51 worst-case examples
W2 shared ranks 1–5 are used as deterministic examples only. No attention-based case substitution.

## 17. Cross-seed descriptive layer-level context
Per-seed layer-level results are aggregated (mean/SD/min/max) across seeds. Per-head cross-seed averaging is NEVER performed (semantic head alignment is deferred to Phase 57).

## 18. Main findings
Total findings: 292

Findings table:

| ID | Scope | Seed | Layer | Head | Code | Value | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F56.001 | CONTINUOUS_ABS_ERROR | 42 | 0 | 0 | ABS_ERROR_ASSOCIATED_WITH_ATTENTION_CONCENTRATION | -0.2100 | For seed 42, layer 0, head 0, higher absolute error was associated with more concentrated last-query... |
| F56.002 | CONTINUOUS_ABS_ERROR | 42 | 0 | 3 | ABS_ERROR_ASSOCIATED_WITH_LONGER_EXPECTED_LAG | 0.3258 | For seed 42, layer 0, head 3, higher absolute error was associated with longer expected lag (Spearma... |
| F56.003 | CONTINUOUS_ABS_ERROR | 42 | 0 | 0 | ABS_ERROR_ASSOCIATED_WITH_MORE_RECENT_ATTENTION | 0.3459 | For seed 42, layer 0, head 0, higher absolute error was associated with more recent-1h attention mas... |
| F56.004 | CONTINUOUS_ABS_ERROR | 42 | 0 | 3 | NO_CLEAR_ERROR_ATTENTION_ASSOCIATION | -0.3082 | For seed 42, layer 0, head 3, recent_6h_mass shows Spearman rho = -0.308 with absolute error (descri... |
| F56.005 | CONTINUOUS_ABS_ERROR | 42 | 0 | 0 | NO_CLEAR_ERROR_ATTENTION_ASSOCIATION | 0.2537 | For seed 42, layer 0, head 0, top5_mass shows Spearman rho = +0.254 with absolute error (descriptive... |
| F56.006 | CONTINUOUS_ABS_ERROR | 42 | 0 | 3 | ABS_ERROR_ASSOCIATED_WITH_LARGER_LAG80 | 0.3684 | For seed 42, layer 0, head 3, higher absolute error was associated with larger Lag80 (Spearman rho =... |
| F56.007 | CONTINUOUS_ABS_ERROR | 42 | 1 | 1 | ABS_ERROR_ASSOCIATED_WITH_ATTENTION_CONCENTRATION | -0.3429 | For seed 42, layer 1, head 1, higher absolute error was associated with more concentrated last-query... |
| F56.008 | CONTINUOUS_ABS_ERROR | 42 | 1 | 0 | ABS_ERROR_ASSOCIATED_WITH_LONGER_EXPECTED_LAG | 0.1742 | For seed 42, layer 1, head 0, higher absolute error was associated with longer expected lag (Spearma... |
| F56.009 | CONTINUOUS_ABS_ERROR | 42 | 1 | 3 | ABS_ERROR_ASSOCIATED_WITH_MORE_RECENT_ATTENTION | 0.2188 | For seed 42, layer 1, head 3, higher absolute error was associated with more recent-1h attention mas... |
| F56.010 | CONTINUOUS_ABS_ERROR | 42 | 1 | 0 | NO_CLEAR_ERROR_ATTENTION_ASSOCIATION | -0.2088 | For seed 42, layer 1, head 0, recent_6h_mass shows Spearman rho = -0.209 with absolute error (descri... |
| F56.011 | CONTINUOUS_ABS_ERROR | 42 | 1 | 1 | NO_CLEAR_ERROR_ATTENTION_ASSOCIATION | 0.3015 | For seed 42, layer 1, head 1, top5_mass shows Spearman rho = +0.302 with absolute error (descriptive... |
| F56.012 | CONTINUOUS_ABS_ERROR | 42 | 1 | 0 | ABS_ERROR_ASSOCIATED_WITH_LARGER_LAG80 | 0.2319 | For seed 42, layer 1, head 0, higher absolute error was associated with larger Lag80 (Spearman rho =... |
| F56.013 | CONTINUOUS_ABS_ERROR | 123 | 0 | 3 | ABS_ERROR_ASSOCIATED_WITH_ATTENTION_CONCENTRATION | -0.3141 | For seed 123, layer 0, head 3, higher absolute error was associated with more concentrated last-quer... |
| F56.014 | CONTINUOUS_ABS_ERROR | 123 | 0 | 2 | ABS_ERROR_ASSOCIATED_WITH_SHORTER_EXPECTED_LAG | -0.2081 | For seed 123, layer 0, head 2, higher absolute error was associated with shorter expected lag (Spear... |
| F56.015 | CONTINUOUS_ABS_ERROR | 123 | 0 | 1 | ABS_ERROR_ASSOCIATED_WITH_MORE_RECENT_ATTENTION | 0.1548 | For seed 123, layer 0, head 1, higher absolute error was associated with more recent-1h attention ma... |
| F56.016 | CONTINUOUS_ABS_ERROR | 123 | 0 | 2 | NO_CLEAR_ERROR_ATTENTION_ASSOCIATION | 0.2177 | For seed 123, layer 0, head 2, recent_6h_mass shows Spearman rho = +0.218 with absolute error (descr... |
| F56.017 | CONTINUOUS_ABS_ERROR | 123 | 0 | 3 | NO_CLEAR_ERROR_ATTENTION_ASSOCIATION | 0.3143 | For seed 123, layer 0, head 3, top5_mass shows Spearman rho = +0.314 with absolute error (descriptiv... |
| F56.018 | CONTINUOUS_ABS_ERROR | 123 | 0 | 2 | ABS_ERROR_ASSOCIATED_WITH_SMALLER_LAG80 | -0.1988 | For seed 123, layer 0, head 2, higher absolute error was associated with smaller Lag80 (Spearman rho... |
| F56.019 | CONTINUOUS_ABS_ERROR | 123 | 1 | 2 | ABS_ERROR_ASSOCIATED_WITH_ATTENTION_CONCENTRATION | -0.2672 | For seed 123, layer 1, head 2, higher absolute error was associated with more concentrated last-quer... |
| F56.020 | CONTINUOUS_ABS_ERROR | 123 | 1 | 3 | ABS_ERROR_ASSOCIATED_WITH_SHORTER_EXPECTED_LAG | -0.3530 | For seed 123, layer 1, head 3, higher absolute error was associated with shorter expected lag (Spear... |
| F56.021 | CONTINUOUS_ABS_ERROR | 123 | 1 | 3 | ABS_ERROR_ASSOCIATED_WITH_MORE_RECENT_ATTENTION | 0.4328 | For seed 123, layer 1, head 3, higher absolute error was associated with more recent-1h attention ma... |
| F56.022 | CONTINUOUS_ABS_ERROR | 123 | 1 | 3 | NO_CLEAR_ERROR_ATTENTION_ASSOCIATION | 0.3705 | For seed 123, layer 1, head 3, recent_6h_mass shows Spearman rho = +0.371 with absolute error (descr... |
| F56.023 | CONTINUOUS_ABS_ERROR | 123 | 1 | 2 | NO_CLEAR_ERROR_ATTENTION_ASSOCIATION | 0.2899 | For seed 123, layer 1, head 2, top5_mass shows Spearman rho = +0.290 with absolute error (descriptiv... |
| F56.024 | CONTINUOUS_ABS_ERROR | 123 | 1 | 3 | ABS_ERROR_ASSOCIATED_WITH_SMALLER_LAG80 | -0.2076 | For seed 123, layer 1, head 3, higher absolute error was associated with smaller Lag80 (Spearman rho... |
| F56.025 | CONTINUOUS_ABS_ERROR | 2026 | 0 | 3 | ABS_ERROR_ASSOCIATED_WITH_ATTENTION_CONCENTRATION | -0.2041 | For seed 2026, layer 0, head 3, higher absolute error was associated with more concentrated last-que... |
| F56.026 | CONTINUOUS_ABS_ERROR | 2026 | 0 | 2 | ABS_ERROR_ASSOCIATED_WITH_LONGER_EXPECTED_LAG | 0.3709 | For seed 2026, layer 0, head 2, higher absolute error was associated with longer expected lag (Spear... |
| F56.027 | CONTINUOUS_ABS_ERROR | 2026 | 0 | 2 | ABS_ERROR_ASSOCIATED_WITH_LESS_RECENT_ATTENTION | -0.3148 | For seed 2026, layer 0, head 2, higher absolute error was associated with less recent-1h attention m... |
| F56.028 | CONTINUOUS_ABS_ERROR | 2026 | 0 | 2 | NO_CLEAR_ERROR_ATTENTION_ASSOCIATION | -0.3863 | For seed 2026, layer 0, head 2, recent_6h_mass shows Spearman rho = -0.386 with absolute error (desc... |
| F56.029 | CONTINUOUS_ABS_ERROR | 2026 | 0 | 3 | NO_CLEAR_ERROR_ATTENTION_ASSOCIATION | 0.1835 | For seed 2026, layer 0, head 3, top5_mass shows Spearman rho = +0.183 with absolute error (descripti... |
| F56.030 | CONTINUOUS_ABS_ERROR | 2026 | 0 | 2 | ABS_ERROR_ASSOCIATED_WITH_LARGER_LAG80 | 0.3680 | For seed 2026, layer 0, head 2, higher absolute error was associated with larger Lag80 (Spearman rho... |

## 19. Why no best head is selected
Per-head error-conditioned results remain seed-specific. Same head index across seeds does NOT imply semantic alignment. Therefore no best head is selected.

## 20. Why no retuning/correction is allowed
Phase 56 is diagnostic. No model is retrained. No prediction is corrected. No attention is re-extracted. Test error cohorts are not used as deployment regimes.

## 21. Why association is not causation
Attention and error are both outcomes of the same forward pass. Conditioning on realized error does not create causal identification. All findings explicitly distinguish association/co-occurrence from causal explanation.

## 22. Handoff to seed-stability attention analysis
`phase57_seed_stability_attention_handoff.json` is ready. `ready_for_phase57 = True`.
`same_index_head_semantic_alignment_assumed = False`.

## 23. Limitations
- Only 3 seeds; cross-seed aggregation is descriptive only.
- Test observations are temporally dependent; no naive iid p-value.
- High-error cohort is selected by outcome severity (selection-conditioned).
- Lookback L=72 truncates longer recency windows.
- Head averaging in layer head-mean may hide specialized head behavior.

## 24. Definition of Done
All O56.1-O56.38 artifacts emitted. Tests: 43/43. Findings: 292. Figures: 101. Phase 57/58 handoffs ready.
Overall status: `PASS`.
