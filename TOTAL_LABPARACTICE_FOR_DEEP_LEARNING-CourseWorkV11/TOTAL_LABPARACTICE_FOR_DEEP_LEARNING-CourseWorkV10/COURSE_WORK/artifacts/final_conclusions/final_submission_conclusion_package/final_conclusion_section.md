# Conclusions

## 1. Objective and Protocol

This study evaluated a Transformer Encoder for one-step-ahead multivariate energy
regression on the UCI Appliances Energy Prediction dataset. The experimental
protocol compared the final Transformer against a persistence baseline and a tuned
LSTM baseline. Temporal attention behavior was analyzed at the full attention-map,
within-seed head, error-conditioned, and cross-seed levels. All evidence was
collected on a frozen chronological Held-Out Test population (FINAL_TEST_POP-v1,
N = 2961 targets) using three predefined final seeds (42, 123, 2026).

## 2. Final Held-Out Test Performance

On the frozen chronological Held-Out Test segment (FINAL_TEST_POP-v1), the final
selected Transformer obtained three-seed mean MAE 28.53 Wh,
RMSE 63.83 Wh, and R² 0.506
across seeds 42, 123 and 2026.

These mean ± SD values summarize independent final runs and do not represent an
ensemble prediction.

The Persistence baseline achieved MAE 26.74 Wh,
RMSE 66.84 Wh, and R² 0.459.
The Tuned LSTM was not evaluated on FINAL_TEST_POP-v1 due to a lookback mismatch
(L36 vs L72), as documented in Phase 47.

## 3. Temporal Robustness and Error Behavior

Rolling-origin robustness was evaluated in Phase 44 as development evidence only.
Fold-by-fold performance variability is reported in FT03. This evidence pertains
to candidate robustness during model development and is not a second Held-Out Test.

Forecast errors were larger in high-consumption and rapid-change regimes relative
to a reference regime, as reported in FT05. The largest errors were retained
as valid Test observations and were used for diagnostic analysis rather than
excluded from final metrics.

## 4. Temporal Attention and Head Diversity

Across the final Test, last-query attention allocated substantial mass to recent
historical positions (recent-1h and recent-6h mass from FT06). Lookback was 72
steps (12 h), so the most recent 24-hour mass was truncated.

Different heads within each layer learned non-identical temporal allocation
profiles, as reported in FT07 (pairwise JSD, Wasserstein minutes, cosine,
top-1 TVD). Similarity in attention allocation does not prove functional
redundancy because value projections and downstream output transformations
can differ.

## 5. Error-Conditioned Attention and Seed Stability

Attention metrics were associated with forecast-error magnitude as reported by
FT08 (Spearman ρ, HIGH_ERROR vs LOW_ERROR differences). HIGH/LOW are
Test-relative diagnostic cohorts and are not deployment regimes; they were not
used for retuning.

Layer-level head-mean attention was more reproducible across the three predefined
final seeds than individual matched-head patterns, as reported in FT09
(permutation-invariant layer head-mean). Same numeric head indices across seeds
were not assumed to represent the same learned role. Cycle consistency was
partial (Layer 0 = 1/4; Layer 1 = 4/4).

## 6. Limitations

The findings are subject to several important limitations. The UCI Appliances
dataset covers a single household; results cannot establish generalization to
other households, buildings, or climates. The forecasting scope is H=1
(10-minute one-step-ahead) under the WB0 boundary (previously observed targets
are assumed available for subsequent predictions). The selected hyperparameters
result from sequential one-factor tuning, which does not guarantee a global
optimum. Only three final seeds (42, 123, 2026) were evaluated; the SD is
descriptive only and does not characterize the full distribution over random
initializations. Time-series dependence limits formal statistical inference;
all diagnostics are descriptive. Attention describes temporal token allocation
and is not raw-feature importance or causal attribution. No prospective
deployment or online evaluation was performed.

## 7. Future Work

Future work could extend this analysis in several directions. Evaluating on
additional households, buildings, and climates would address external validity.
Multi-step forecasting (H > 1) could be explored via direct, recursive, or
probabilistic approaches. Head ablation or feature-level attribution methods
(e.g. Integrated Gradients, SHAP) could test functional redundancy. Additional
seeds and block-bootstrap uncertainty quantification could strengthen stochastic
robustness. Online and prospective deployment evaluation remains out of scope
for the current project.

## 8. Closing Statement

Overall, the study establishes a reproducible Transformer-based one-step
forecasting pipeline and provides a cautious temporal-attention analysis,
while the observed limitations define clear directions for broader validation
and stronger attribution methods.
