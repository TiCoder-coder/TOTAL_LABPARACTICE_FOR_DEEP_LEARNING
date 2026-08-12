# PHASE 56 — ERROR-CONDITIONED ATTENTION

## Kế hoạch phân tích mối quan hệ giữa forecast error và temporal attention của Final Transformer trên Held-Out Test mà không biến Test diagnostics thành một vòng retuning mới

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Model:** Attention-Aware Transformer Encoder for regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Residual convention:** `residual = y_true - y_pred`  
**Primary attention source:** `LAST_QUERY_ATTENTION-v1` + Phase52 raw last-query arrays  
**Primary error source:** `RESIDUAL_ANALYSIS-v1`  
**Frozen regime context:** `ERROR_BY_REGIME-v1`  
**Frozen worst-case context:** `WORST_ERROR_ANALYSIS-v1`  
**Phase ID:** `PHASE_56_ERROR_CONDITIONED_ATTENTION`  
**Output version:** `ERROR_CONDITIONED_ATTENTION-v1`  
**Phase trước:** `Phase_55_Head_comparison.md`  
**Phase sau:** `Phase_57_Seed-stability_attention_check.md`

---

# 1. Vai trò của Phase 56

Phase56 trả lời câu hỏi trung tâm:

> Temporal attention của Final Transformer có thay đổi một cách có hệ thống khi forecast error lớn hơn, nhỏ hơn, underprediction hay overprediction hay không?

Phase này kết nối hai nhánh phân tích đã được khóa độc lập:

```text
Forecast error branch:
Phase47 → Phase49 → Phase50 → Phase51

Attention branch:
Phase52 → Phase53 → Phase54 → Phase55.
```

Mục tiêu:

```text
1. Verify frozen error and attention sources.
2. Freeze error-conditioning rules before joining them to attention metrics.
3. Analyze continuous association between error magnitude and attention behavior.
4. Analyze signed residual association with attention behavior.
5. Build deterministic low/mid/high error cohorts from error ranks only.
6. Build deterministic absolute-error deciles from error ranks only.
7. Compare high-error vs low-error attention metric distributions.
8. Compare high-error vs low-error temporal attention profiles.
9. Compare underprediction vs overprediction attention behavior.
10. Build shared-error cohorts from Phase51 shared hardness for cross-seed common-target context.
11. Analyze permutation-invariant layer head-mean attention under error conditioning.
12. Analyze secondary full-matrix attention summaries against error.
13. Preserve Phase50 Train-defined regimes as contextual labels only.
14. Preserve Phase51 worst cases as deterministic examples only.
15. Avoid p-value fishing and naive iid inference.
16. Avoid best-head/best-seed selection.
17. Avoid any post-Test correction, retraining or model redesign.
18. Freeze error-conditioned attention findings for Phase57 and final reporting.
```

Nguyên tắc trung tâm:

\[
\boxed{
Frozen\ Errors
+
Frozen\ Attention
+
Predeclared\ Error\ Conditioning
+
Continuous\ Associations
+
Rank\text{-}Based\ Cohorts
+
Profile\ Contrasts
+
No\ Retuning
+
No\ Best\ Head
}
\]

---

# 2. Phase56 là diagnostic analysis, không phải model selection

Forbidden:

```text
change feature set
change lookback
change Transformer architecture
change loss
change learning rate
change RevIN
change attention heads
prune heads
retrain model
bias-correct predictions
clip predictions
time-shift predictions
choose another seed.
```

Nếu Phase56 phát hiện:

```text
high error ↔ more diffuse attention
high error ↔ less recent attention
underprediction ↔ different temporal profile
```

thì kết quả được dùng để:

```text
describe
interpret cautiously
support limitations/future work
support final discussion.
```

Không được dùng để sửa final Test model trong project hiện tại.

---

# 3. Test-derived error cohorts được phép trong Phase56 nhưng chỉ với vai trò diagnostic

Điểm phương pháp luận quan trọng:

```text
Phase50 regimes
=
Train-defined data regimes.

Phase56 error cohorts
=
Test-relative diagnostic groups based on realized forecast error.
```

Hai khái niệm này không được nhập làm một.

---

# 4. Error cohort không phải deployment regime

Ví dụ:

```text
HIGH_ERROR
```

chỉ có thể biết sau khi đã có:

```text
y_true
và
y_pred.
```

Do đó nó không phải một trạng thái có thể biết trước khi forecast.

Không được gọi:

```text
high-error operating regime
```

theo nghĩa deployment.

Preferred:

```text
diagnostic high-error cohort.
```

---

# 5. Error thresholds không được feed back vào model

Forbidden:

```text
top20% error
→ add sample weight
→ retrain

high-error attention pattern
→ alter lookback
→ rerun Test

underprediction cohort
→ add bias offset.
```

---

# 6. Upstream hard gate

Required Phase55:

```text
phase_55_signoff.json
phase56_error_conditioned_attention_handoff.json
head_behavior_summary.csv
head_pair_comparison_long.csv
layer_head_diversity_summary.csv
```

Required Phase54:

```text
phase_54_signoff.json
last_query_metrics_long.csv
last_query_profile_by_lag.csv
last_query_lag_bin_mass.csv
last_query_recent_mass_summary.csv
```

Required Phase52:

```text
last_query_attention_seed42.npz
last_query_attention_seed123.npz
last_query_attention_seed2026.npz
attention_full_matrix_summary.csv
raw_attention_checksums.json
attention_test_target_order.csv
```

Required error branch:

```text
phase_49_signoff.json
residual_long_table.csv
residual_wide_table.csv

phase_50_signoff.json
test_regime_assignment.csv

phase_51_signoff.json
shared_case_hardness_all_test.csv
shared_worst_cases_top20.csv.
```

Hard:

```text
phase56_ready = true.
```

---

# 7. Numerical source hierarchy

Primary last-query metrics:

```text
Phase54 last_query_metrics_long.csv.
```

Primary raw profile source:

```text
Phase52 last_query_attention_seed*.npz.
```

Error source:

```text
Phase49 residual_long_table.csv.
```

Shared-hardness source:

```text
Phase51 shared_case_hardness_all_test.csv.
```

Regime context:

```text
Phase50 test_regime_assignment.csv.
```

Heatmaps/images:

```text
visual context only.
```

---

# 8. No new attention extraction

Hard:

```text
new checkpoint forward = forbidden
new Test prediction    = forbidden
new attention extraction = forbidden.
```

If numerical source is missing:

```text
STOP
```

rather than reconstructing from heatmap PNG.

---

# 9. No new forecast-error computation except integrity verification

Phase56 may recompute:

```text
abs(residual)
```

to audit source consistency.

It should not create an alternative error definition.

Canonical:

\[
e=y-\hat y.
\]

---

# 10. Error magnitude

Primary conditioning variable:

\[
AE=|e|.
\]

Why:

```text
sign-neutral
interpretable in Wh
same basis as MAE
directly aligned with Phase51 worst-case logic.
```

---

# 11. Signed error

Secondary conditioning variable:

\[
e=y-\hat y.
\]

Interpretation:

```text
e > 0 → underprediction
e < 0 → overprediction.
```

---

# 12. Squared error is not a separate primary conditioning variable

Because:

\[
SE=e^2
\]

is a monotonic transform of:

\[
|e|.
\]

Do not duplicate all analyses using both AE and squared error and treat them as independent evidence.

---

# 13. Shared hardness

Phase51 defines:

\[
SharedHardness_t
=
\frac{|e_{42,t}|+|e_{123,t}|+|e_{2026,t}|}{3}.
\]

This is allowed as a seed-invariant target-level difficulty diagnostic.

It is not:

```text
ensemble error.
```

---

# 14. Three conditioning systems

Phase56 freezes three complementary systems:

```text
C1 — Continuous error conditioning
C2 — Seed-specific rank-based error cohorts
C3 — Shared-hardness rank-based cohorts.
```

---

# 15. C1 — Continuous error conditioning

For every:

```text
seed
layer
head
```

compute association between attention metrics and:

```text
absolute_error_wh
signed residual_wh.
```

Primary association:

```text
Spearman rank correlation.
```

---

# 16. Why Spearman is primary

Error and attention metrics may be:

```text
nonlinear
skewed
heavy-tailed.
```

Spearman captures monotonic association and is less dominated by extreme values.

---

# 17. No primary Pearson error-attention correlation

Pearson may be stored as optional secondary diagnostic but is not required.

Core contract:

```text
Spearman only.
```

This avoids redundant metric fishing.

---

# 18. Primary attention metric set

Freeze before analysis:

```text
M1 normalized_entropy
M2 expected_lag_minutes
M3 recent_1h_mass
M4 recent_6h_mass
M5 top5_mass
M6 lag80_minutes.
```

These six are the Phase56 **CORE_ATTENTION_METRICS-v1**.

---

# 19. Why these six metrics

They cover distinct aspects:

```text
normalized_entropy
→ concentration/diffuseness

expected_lag
→ temporal center

recent_1h
→ short-term recency

recent_6h
→ broader recency

top5_mass
→ sparse concentration

Lag80
→ temporal coverage radius.
```

---

# 20. Secondary attention metric set

Allowed but not headline:

```text
effective_source_count
lag_sd_minutes
top1_weight
lag50_minutes
lag90_minutes
recent_12h_mass if fully supported
recent_24h_mass if fully supported.
```

---

# 21. Unsupported horizon policy

If final lookback does not fully support:

```text
12h
24h
```

do not use truncated values as if they were full-horizon metrics.

Primary metric set avoids this problem.

---

# 22. Continuous absolute-error association

For metric `m`:

\[
\rho^{AE}_{s,l,h,m}
=
Spearman(AE_{s,t},M_{s,t,l,h,m}).
\]

Report:

```text
rho
N
metric
seed
layer
head.
```

---

# 23. Continuous signed-residual association

\[
\rho^{R}_{s,l,h,m}
=
Spearman(e_{s,t},M_{s,t,l,h,m}).
\]

Interpretation depends on metric direction.

Example:

```text
positive rho(residual,recent1h)
```

means larger/more-positive residuals tend to co-occur with larger recent1h attention mass.

It does not imply causality.

---

# 24. No p-value as core output

Time-adjacent Test observations are dependent.

Core Phase56 reports:

```text
Spearman rho
sample count
empirical plots.
```

No naive iid p-value.

---

# 25. Optional temporal block bootstrap

Advanced optional module:

```text
TEMPORAL_BLOCK_BOOTSTRAP = OFF
```

by default.

If later enabled by protocol amendment, predeclare:

```text
block construction
block length
resample count
random seed
gap handling
```

before looking at interval results.

It is not required for Phase56 PASS.

---

# 26. Why bootstrap is OFF by default

The coursework requires analysis rather than formal inferential testing.

Adding bootstrap introduces extra methodological choices without changing the core diagnostic conclusions.

---

# 27. C2 — Seed-specific error cohorts

For each seed separately, sort all Test targets by:

```text
absolute_error_wh ascending
timestamp ascending
target_id ascending.
```

The latter two fields are deterministic tie-breakers.

---

# 28. Exact rank-based 20/60/20 cohort rule

Let:

```text
N = Test target count for one seed.
```

Define:

\[
n_{edge}=\max(1,\lfloor0.20N\rfloor).
\]

Then:

```text
LOW_ERROR:
first n_edge targets

HIGH_ERROR:
last n_edge targets

MID_ERROR:
all remaining targets.
```

---

# 29. Why rank-based instead of Test quantile values

This ensures:

```text
deterministic cohort size
stable handling of tied errors
no threshold drift.
```

It also makes clear the grouping is:

```text
Test-relative diagnostic ranking
```

not a deployment threshold.

---

# 30. Tie behavior

Equal absolute errors can land on opposite sides of a cohort boundary because deterministic ranking uses:

```text
timestamp
target_id
```

as tie-breakers.

This must be documented.

Do not adjust groups manually to keep ties together after seeing attention.

---

# 31. Exact rank-based error deciles

For ordered zero-based rank index:

```text
r = 0..N-1
```

define:

\[
Decile
=
1+\left\lfloor\frac{10r}{N}\right\rfloor
\]

capped at:

```text
10.
```

This produces approximately equal-size:

```text
ERROR_DECILE_1 ... ERROR_DECILE_10.
```

---

# 32. Error-decile interpretation

```text
Decile1
→ lowest realized absolute errors

Decile10
→ highest realized absolute errors.
```

Diagnostic only.

---

# 33. Error cohort assignment must be frozen before attention join

Workflow:

```text
residual source
→ derive rank
→ assign LOW/MID/HIGH + decile
→ freeze assignment/checksum
→ only then join attention metrics.
```

This prevents attention patterns from changing the cohort definition.

---

# 34. Seed-specific cohort labels differ across seeds

Because each seed has different errors:

```text
seed42 HIGH_ERROR target set
may differ from
seed123 HIGH_ERROR target set.
```

This is expected.

Do not force alignment.

---

# 35. C3 — Shared-hardness cohorts

Use Phase51:

```text
mean_abs_error_across_seeds.
```

Sort by:

```text
shared hardness ascending
timestamp ascending
target_id ascending.
```

Use the same exact:

```text
20/60/20
and
10-decile
```

rank rules.

---

# 36. Shared cohorts are common across seeds

This creates:

```text
same target IDs
```

for seed42/123/2026.

Useful for:

```text
cross-seed descriptive context
Phase57 follow-up.
```

Still not a semantic head alignment solution.

---

# 37. Shared cohort labels

Canonical:

```text
SHARED_LOW_ERROR
SHARED_MID_ERROR
SHARED_HIGH_ERROR

SHARED_ERROR_DECILE_1
...
SHARED_ERROR_DECILE_10.
```

---

# 38. Shared worst top20 remains separate

Phase51:

```text
W2 shared top20
```

is a case-level set.

Do not equate:

```text
SHARED_HIGH_ERROR 20%
```

with:

```text
shared top20 cases.
```

---

# 39. Error cohort assignment artifact

Create one combined:

```text
error_conditioning_assignment.csv
```

containing:

```text
target_id
timestamp
seed
absolute_error
residual
seed_error_rank
seed_error_cohort
seed_error_decile
shared_hardness
shared_error_rank
shared_error_cohort
shared_error_decile.
```

---

# 40. Error cohort assignment is frozen before attention metrics

Create checksum:

```text
error_conditioning_assignment_sha256.
```

No later regrouping.

---

# 41. Under/overprediction groups

From residual sign:

```text
UNDER
= residual > 0

OVER
= residual < 0

EXACT_ZERO
= residual == 0.
```

No epsilon.

---

# 42. Under-vs-over comparison

Primary signed group comparison:

```text
UNDER vs OVER.
```

`EXACT_ZERO` is retained in source/count tables but excluded from the two-group contrast.

---

# 43. High-error signed subgroups

Secondary:

Within:

```text
HIGH_ERROR
```

split:

```text
HIGH_UNDER
HIGH_OVER
HIGH_ZERO.
```

Use for descriptive severe-error direction context.

No new thresholds.

---

# 44. Minimum cohort-size warning

For any comparison group:

```text
N < 30
→ SMALL_N_WARNING.
```

Do not drop group.

---

# 45. No attention-based balancing

Do not subsample a larger cohort to match a smaller cohort because attention looks different.

Core summaries use all assigned members.

---

# 46. No random control sampling

No.

Rank-based LOW_ERROR is deterministic control cohort.

---

# 47. High-vs-low metric comparison

For each:

```text
seed
layer
head
metric
```

compare:

```text
HIGH_ERROR
vs
LOW_ERROR.
```

Required:

```text
N_high
N_low
mean_high
mean_low
median_high
median_low
delta_mean_high_minus_low
delta_median_high_minus_low
Cliff's delta
p05/p25/p75/p95 per cohort.
```

---

# 48. Cliff's delta

Define:

\[
\delta_C
=
P(X_{high}>X_{low})
-
P(X_{high}<X_{low}).
\]

Range:

```text
-1..1.
```

No p-value required.

---

# 49. Cliff's delta sign

Positive:

```text
attention metric tends to be larger in HIGH_ERROR.
```

Negative:

```text
attention metric tends to be smaller in HIGH_ERROR.
```

---

# 50. No small/medium/large Cliff's-delta labels

Do not apply canned magnitude thresholds as primary conclusions.

Report actual:

```text
delta.
```

---

# 51. Under-vs-over metric comparison

For each seed/layer/head/core metric:

```text
median_under
median_over
delta_median_under_minus_over
Cliff's delta
N_under
N_over.
```

---

# 52. Signed direction caveat

Under vs over groups may differ in:

```text
error magnitude
target level
rapid-change prevalence.
```

Phase56 treats this as descriptive attention association, not causal decomposition.

---

# 53. Error-decile metric trends

For each:

```text
seed/layer/head/decile/core metric
```

compute:

```text
N
mean
median
p25
p75.
```

This reveals gradual or non-monotonic patterns across error severity.

---

# 54. No line fit required

Plot decile centers/trends.

Do not force:

```text
linear trend.
```

Continuous Spearman is already the monotonic summary.

---

# 55. Raw attention profile conditioning

Metric summaries alone can hide redistribution across lag positions.

Phase56 therefore compares actual last-query temporal profiles between error cohorts.

---

# 56. High-error cohort mean profile

For seed `s`, layer `l`, head `h`, lag `k`:

\[
P^{HIGH}_{s,l,h}(k)
=
Mean_{t\in HIGH_s}
a_{s,t,l,h}(k).
\]

---

# 57. Low-error cohort mean profile

\[
P^{LOW}_{s,l,h}(k)
=
Mean_{t\in LOW_s}
a_{s,t,l,h}(k).
\]

Both must sum to:

```text
≈1.
```

---

# 58. High-minus-low profile difference

\[
D_{HL}(k)
=
P^{HIGH}(k)-P^{LOW}(k).
\]

Expected:

\[
\sum_k D_{HL}(k)\approx0.
\]

Hard audit.

---

# 59. High-vs-low profile distances

Required:

```text
Jensen–Shannon divergence
L1 distance
Wasserstein-1 distance in minutes
cosine similarity.
```

No weighted composite.

---

# 60. Why Wasserstein matters here

It quantifies how far in temporal lag the attention mass shifts between:

```text
high-error
and
low-error
```

cohorts.

---

# 61. Profile difference interpretation

Safe:

> High-error targets assigned more average attention mass to lags beyond six hours than low-error targets for this head.

Unsafe:

> Attending to old history caused the error.

---

# 62. Under-vs-over mean profiles

Similarly compute:

```text
P_UNDER(k)
P_OVER(k)
D_UO(k)=P_UNDER-P_OVER.
```

Required profile distances:

```text
JSD
L1
Wasserstein.
```

---

# 63. Exact-zero residual profile

Not required due likely small N.

Count/report only.

---

# 64. Error-decile temporal profiles

For each seed/layer/head/decile:

```text
mean attention weight by lag.
```

These profiles allow visualization of temporal-focus evolution across error severity.

---

# 65. Decile profile table can be large

Store machine-readable:

```text
error_attention_decile_profile_by_lag.csv.
```

Report figures need not show all heads simultaneously if unreadable.

---

# 66. Decile profile plotting strategy

Recommended:

```text
Decile1
Decile5/6
Decile10
```

for visualization only if cases are selected by predeclared decile identity.

Better:

```text
plot all ten with ordered alpha/line treatment
```

where readable.

Do not choose “interesting” deciles after inspection.

---

# 67. Layer head-mean error-conditioned analysis

Per target/layer/seed compute:

\[
a^{layermean}_{s,t,l}(k)
=
\frac{1}{H}\sum_h a_{s,t,l,h}(k).
\]

This is invariant to head ordering within the layer.

---

# 68. Why layer head-mean is important

It provides a stable summary for:

```text
cross-seed descriptive comparisons
```

without assuming:

```text
Head1 seed42 == Head1 seed123.
```

---

# 69. Metrics on layer head-mean vector

For every:

```text
seed/target/layer
```

compute:

```text
normalized entropy
expected lag
recent1h mass
recent6h mass
top5 mass
Lag80.
```

Use exact Phase54 formulas.

---

# 70. Do not use mean of head metrics as substitute

For nonlinear metrics:

```text
metric(mean_heads(vector))
```

can differ from:

```text
mean_heads(metric(vector)).
```

Canonical layer head-mean analysis uses:

```text
metric of head-mean vector.
```

---

# 71. Layer head-mean continuous association

Compute Spearman with:

```text
seed-specific absolute error
seed-specific residual
shared hardness.
```

This is a strong permutation-invariant summary.

---

# 72. Layer head-mean high-vs-low comparison

Use same frozen seed-specific error cohorts.

Compute:

```text
metric differences
Cliff's delta
profile JSD
profile Wasserstein.
```

---

# 73. Shared-cohort layer head-mean analysis

Use:

```text
SHARED_LOW_ERROR
SHARED_HIGH_ERROR
```

same targets across seeds.

Report each seed/layer separately.

Cross-seed aggregation may summarize:

```text
mean and sample SD across seeds
```

for layer-level metrics because layers are architecturally aligned and head permutation has been averaged out.

---

# 74. No per-head cross-seed averaging

Hard:

```text
no average of Head1 results across seeds
```

in Phase56.

Per-head outputs remain seed-specific.

---

# 75. Full-matrix summary error conditioning

Phase52 stored all-Test:

```text
attention_full_matrix_summary.csv.
```

Secondary Phase56 analysis uses:

```text
mean_query_entropy
mean_self_attention_weight
mean_absolute_query_source_distance_steps
mean_forward_within_input_mass.
```

---

# 76. Why full-matrix summaries matter

Last-query focuses only on:

```text
q=L-1.
```

Full-matrix summaries characterize:

```text
all queries
```

and are especially relevant if final pooling is:

```text
MEAN.
```

---

# 77. Full-matrix core secondary associations

For each seed/layer/head:

```text
Spearman(AE, mean_query_entropy)
Spearman(AE, mean_self_attention_weight)
Spearman(AE, mean_absolute_query_source_distance)
Spearman(AE, mean_forward_within_input_mass)

Spearman(residual, same metrics).
```

No group profile because full matrices were not stored for all targets.

---

# 78. Forward-within-input caveat

`forward_within_input_mass` means source positions later than the query **within the historical input window**.

It is not future leakage relative to forecast target.

---

# 79. Phase50 regime context

Every error cohort can be annotated with frozen Phase50 labels.

Allowed output:

```text
regime composition of HIGH_ERROR vs LOW_ERROR cohorts.
```

This helps identify confounding/context.

---

# 80. Regime composition does not redefine cohorts

Do not stratify and then tune error cohorts based on regime results.

---

# 81. Primary regime-context output

For each seed-specific error cohort report shares of:

```text
TL_LOW/TL_MID/TL_HIGH
EXTREME_HIGH
CHANGE_RAPID
DIR_DOWN/DIR_FLAT/DIR_UP
time-of-day
weekday/weekend.
```

No new regime thresholds.

---

# 82. No full regime × error × head mining

Forbidden:

```text
HIGH_ERROR
× EXTREME_HIGH
× EVENING
× Head3
```

search for strongest pattern.

This would create post-hoc subgroup mining.

---

# 83. Optional predeclared key-context view

A compact secondary table may report HIGH_ERROR cohort composition for:

```text
EXTREME_HIGH
CHANGE_RAPID
DIR_UP.
```

because they were already central Phase50/51 contexts.

No attention comparison within these intersections in core Phase56.

---

# 84. Phase51 worst-case context

W2 shared top20 and shared all-under/all-over cases remain:

```text
deterministic case examples.
```

Phase56 may attach their:

```text
last-query metrics
layer head-mean metrics
```

but must not use them as the only evidence.

---

# 85. No worst-case attention cherry-pick

Use Phase51 ranks/order.

No case substitution based on attention pattern.

---

# 86. Shared top20 vs full-cohort distinction

W2 top20:

```text
case study set.
```

HIGH_ERROR top20%:

```text
statistical diagnostic cohort.
```

Keep separate in naming and plots.

---

# 87. Cross-seed shared-hardness association

For each seed/layer/head:

\[
\rho^{SH}
=
Spearman(
SharedHardness_t,
AttentionMetric_{s,t,l,h}
).
\]

Secondary.

This uses same conditioning variable across seeds without averaging heads.

---

# 88. Cross-seed interpretation caveat

Even if same-index head associations have similar signs across seeds:

```text
do not yet claim semantic head stability.
```

Phase57 handles permutation-aware matching.

---

# 89. Association sign consistency summary

Allowed within one seed/layer:

```text
number of heads with positive rho
number with negative rho
```

for each core metric.

This is descriptive only.

---

# 90. No significance vote counting

Do not count:

```text
number of p<0.05 heads.
```

No core p-values.

---

# 91. No arbitrary correlation threshold

Do not define:

```text
|rho| > .3 = meaningful
```

after seeing results.

Report exact rho.

---

# 92. Profile-distance interpretation without threshold

Report exact:

```text
JSD
L1
Wasserstein minutes.
```

No `LOW/HIGH shift` classification threshold.

---

# 93. Cliff's delta interpretation without threshold

Report exact value and direction.

No automatic:

```text
small
medium
large.
```

---

# 94. Multiple-comparison caution

Phase56 generates many:

```text
seed × layer × head × metric
```

descriptive associations.

Therefore:

```text
do not selectively report only largest coefficients.
```

Machine tables contain all combinations.

Main report should use:

```text
predeclared core metrics
layer head-mean summaries
full coefficient matrices
```

rather than cherry-picked heads.

---

# 95. Report ordering

Always:

```text
seed
layer
head
metric
```

architectural order.

No sort by `|rho|`.

---

# 96. Error-decile ordering

Always:

```text
1 → 10
```

low error to high error.

---

# 97. High/low cohort profile scale

When plotting:

```text
P_HIGH
P_LOW
```

use same y-axis.

For difference:

```text
D_HL
```

use zero-centered y-axis with symmetric limits within the figure.

No per-head normalization.

---

# 98. Error-association heatmap

For one:

```text
seed × layer
```

rows:

```text
heads
```

columns:

```text
CORE_ATTENTION_METRICS-v1.
```

value:

```text
Spearman rho with AE.
```

Scale:

```text
-1..1
```

architectural head order.

---

# 99. Signed-residual association heatmap

Same but:

```text
rho with residual.
```

Separate from absolute-error heatmap.

---

# 100. Cliff's-delta heatmap

Rows:

```text
heads
```

columns:

```text
core metrics.
```

Value:

```text
HIGH_ERROR vs LOW_ERROR Cliff's delta.
```

Scale:

```text
-1..1.
```

---

# 101. High-vs-low profile figure

For each:

```text
seed × layer
```

create small multiples by head showing:

```text
P_LOW
P_HIGH.
```

x-axis:

```text
lag minutes
```

newest on left.

---

# 102. Profile-difference figure

For each seed/layer:

```text
rows/lines=heads
x=lag
y=P_HIGH-P_LOW.
```

Prefer separate head panels if overlap is unreadable.

---

# 103. Error-decile trend figure

For each core metric:

```text
x=error decile 1..10
y=median attention metric
```

one figure per seed/layer with architectural head series.

No smoothed regression line.

---

# 104. Under-vs-over figure

For each seed/layer/head, report:

```text
median attention metric UNDER
vs
OVER.
```

Can use grouped point plots.

No causal language.

---

# 105. Layer head-mean figure

For each seed/layer:

```text
P_HIGH_layermean
P_LOW_layermean
```

and:

```text
difference.
```

This should be one of the main report figures because it is permutation-invariant to head order.

---

# 106. Shared-cohort layer head-mean figure

Same target cohorts across seeds:

```text
SHARED_LOW
SHARED_HIGH.
```

Overlay each seed separately or use separate panels.

Do not average raw per-head identities.

---

# 107. Full-matrix association figure

Secondary compact heatmap:

```text
rows=heads
columns=full-matrix summary metrics
value=Spearman AE association.
```

---

# 108. Error cohort composition figure

Show:

```text
LOW_ERROR
MID_ERROR
HIGH_ERROR
```

regime composition for selected Phase50 labels.

This is context, not attention result.

---

# 109. No attentional “explanation score”

Forbidden:

```text
attention error score
error-attention score
composite interpretability score.
```

---

# 110. No claim that association implies mechanism

Safe:

> Larger forecast errors were associated with lower recent-one-hour attention mass for this head.

Unsafe:

> Lower recent attention caused the forecast error.

---

# 111. Reverse-causality wording

Attention and error are both outcomes of the same forward prediction event.

Do not speak of:

```text
error changing attention after the fact.
```

Better:

```text
error-conditioned attention differences
co-occurrence
association.
```

---

# 112. Cohort mean profile remains a probability distribution

For every:

```text
seed/layer/head/cohort
```

verify:

\[
\sum_k P_{cohort}(k)\approx1.
\]

No renormalization.

---

# 113. High-low difference sum audit

\[
\sum_k D_{HL}(k)\approx0.
\]

Hard.

---

# 114. Under-over difference sum audit

\[
\sum_k D_{UO}(k)\approx0.
\]

Hard.

---

# 115. Decile profile sum audit

Every:

```text
seed/layer/head/decile
```

mean profile sums:

```text
≈1.
```

---

# 116. Error-cohort coverage audit

Per seed:

```text
LOW + MID + HIGH = all Test targets
no overlap
no missing.
```

---

# 117. Error-decile coverage audit

Per seed:

```text
union decile1..10 = all Test targets
mutually exclusive.
```

---

# 118. Shared-cohort coverage audit

Same.

---

# 119. Exact-zero residual handling

Exact-zero targets remain assigned to:

```text
error magnitude cohorts
```

normally.

For signed analysis:

```text
ZERO
```

counted separately.

No arbitrary sign epsilon.

---

# 120. Error cohort counts

Machine-readable output must include:

```text
N
```

for every cohort/seed.

No hidden imbalance.

---

# 121. No dropped targets due attention metric NaN

Phase54 should have valid metrics for every target.

Any missing value:

```text
STOP/investigate.
```

Do not silently use complete cases.

---

# 122. Join contract

Join keys:

```text
seed
target_id.
```

Attention metrics include:

```text
layer
head.
```

Expected:

```text
one error row
→ multiple layer/head attention rows.
```

---

# 123. Join row-count expectation

If:

```text
N_test
N_layers
N_heads
3 seeds
```

then joined per-head table should contain:

\[
3\times N_{test}\times N_{layers}\times N_{heads}
\]

rows.

Runtime values only.

---

# 124. Join integrity

Hard:

```text
no unmatched errors
no unmatched attention targets
no duplicate error rows per seed-target
no missing head rows.
```

---

# 125. Shared-hardness join

Key:

```text
target_id
```

seed-invariant.

Each seed/layer/head row receives same:

```text
shared_hardness
shared cohort/decile.
```

---

# 126. Regime join

Key:

```text
target_id.
```

Same frozen regime labels across seeds.

---

# 127. Core continuous-association table

Create:

```text
error_attention_association_long.csv.
```

Rows:

```text
seed
layer
head
conditioning_variable
attention_metric
N
spearman_rho
status.
```

Conditioning variables:

```text
ABS_ERROR
SIGNED_RESIDUAL
SHARED_HARDNESS.
```

`SHARED_HARDNESS` is secondary.

---

# 128. No p-value column required

If library returns p-value automatically:

```text
do not use it as headline.
```

Preferred:

```text
do not store p-value in canonical core table.
```

---

# 129. Error-decile metric summary table

`error_attention_decile_metric_summary.csv`:

```text
seed
layer
head
error_decile
metric
N
mean
median
p25
p75
status.
```

---

# 130. Error-decile profile table

`error_attention_decile_profile_by_lag.csv`:

```text
seed
layer
head
error_decile
lag_steps
lag_minutes
mean_weight
profile_sum
status.
```

---

# 131. High-low metric comparison schema

`error_attention_high_low_metric_comparison.csv`:

```text
seed
layer
head
metric
N_low
N_high
mean_low
mean_high
delta_mean_high_minus_low
median_low
median_high
delta_median_high_minus_low
cliffs_delta_high_vs_low
low_p05
low_p25
low_p75
low_p95
high_p05
high_p25
high_p75
high_p95
status
```

---

# 132. High-low profile comparison schema

`error_attention_high_low_profile_comparison.csv`:

```text
seed
layer
head
N_low
N_high
jsd_high_vs_low
l1_high_vs_low
cosine_high_vs_low
wasserstein_minutes_high_vs_low
profile_sum_low
profile_sum_high
difference_sum
status
```

---

# 133. High-low profile-by-lag schema

`error_attention_high_low_profile_difference_by_lag.csv`:

```text
seed
layer
head
lag_steps
lag_minutes
low_mean_weight
high_mean_weight
high_minus_low
status
```

---

# 134. Signed metric comparison schema

`error_attention_signed_metric_comparison.csv`:

```text
seed
layer
head
metric
N_under
N_over
N_zero
mean_under
mean_over
median_under
median_over
delta_median_under_minus_over
cliffs_delta_under_vs_over
status
```

---

# 135. Signed profile comparison schema

`error_attention_signed_profile_comparison.csv`:

```text
seed
layer
head
N_under
N_over
jsd_under_vs_over
l1_under_vs_over
wasserstein_minutes_under_vs_over
profile_sum_under
profile_sum_over
difference_sum
status
```

---

# 136. Signed profile-by-lag schema

`error_attention_signed_profile_difference_by_lag.csv`:

```text
seed
layer
head
lag_steps
lag_minutes
under_mean_weight
over_mean_weight
under_minus_over
status
```

---

# 137. Error cohort assignment schema

`error_conditioning_assignment.csv`:

```text
seed
target_id
target_timestamp
residual_wh
absolute_error_wh
seed_error_rank_asc
seed_error_cohort
seed_error_decile
shared_hardness
shared_error_rank_asc
shared_error_cohort
shared_error_decile
residual_sign_group
high_error_sign_group
status
```

---

# 138. Error cohort audit schema

`error_conditioning_assignment_audit.csv`:

```text
seed
N
n_edge
low_count
mid_count
high_count
decile_min_count
decile_max_count
coverage_complete
cohorts_disjoint
deciles_disjoint
rank_tie_rule
status
```

---

# 139. Shared cohort audit schema

`shared_error_conditioning_audit.csv`:

```text
N
n_edge
shared_low_count
shared_mid_count
shared_high_count
decile_min_count
decile_max_count
same_target_assignment_all_seeds
coverage_complete
status
```

---

# 140. Error-attention join audit schema

`error_attention_join_audit.csv`:

```text
seed
error_target_count
attention_target_count
layer_count
head_count
expected_join_rows
observed_join_rows
unmatched_error_targets
unmatched_attention_targets
duplicate_error_rows
missing_head_rows
status
```

---

# 141. Layer head-mean metrics schema

`error_attention_layer_head_mean_metrics_long.csv`:

```text
seed
target_id
timestamp
layer
normalized_entropy
expected_lag_minutes
recent_1h_mass
recent_6h_mass
top5_mass
lag80_minutes
absolute_error_wh
residual_wh
shared_hardness
seed_error_cohort
seed_error_decile
shared_error_cohort
shared_error_decile
status
```

---

# 142. Layer head-mean association schema

`error_attention_layer_head_mean_association.csv`:

```text
seed
layer
conditioning_variable
attention_metric
N
spearman_rho
status
```

---

# 143. Layer head-mean cohort comparison schema

`error_attention_layer_head_mean_high_low.csv`:

```text
seed
layer
metric
N_low
N_high
median_low
median_high
delta_median_high_minus_low
cliffs_delta
status
```

Plus profile distances in companion fields/file.

---

# 144. Shared-cohort layer comparison schema

`error_attention_shared_cohort_layer_summary.csv`:

```text
seed
layer
metric
shared_low_N
shared_high_N
shared_low_median
shared_high_median
delta
cliffs_delta
profile_jsd
profile_wasserstein_minutes
status
```

---

# 145. Full-matrix association schema

`error_attention_full_matrix_association.csv`:

```text
seed
layer
head
conditioning_variable
full_matrix_metric
N
spearman_rho
status
```

Metrics:

```text
mean_query_entropy
mean_self_attention_weight
mean_absolute_query_source_distance_steps
mean_forward_within_input_mass.
```

---

# 146. Regime-composition context schema

`error_cohort_regime_composition.csv`:

```text
seed
error_cohort
regime_family
regime_label
count
cohort_share
full_test_share
share_difference
status
```

No threshold modification.

---

# 147. Worst-case context schema

`error_conditioned_worst_case_attention_context.csv`:

```text
target_id
shared_rank
selection_roles
seed
layer
head
absolute_error
residual
normalized_entropy
expected_lag_minutes
recent_1h_mass
recent_6h_mass
top5_mass
lag80_minutes
status
```

Case order from Phase51.

---

# 148. Association matrix artifact

For report convenience create:

```text
error_attention_association_matrix.csv
```

per:

```text
seed/layer/conditioning_variable
```

with:

```text
rows=heads
columns=core metrics
values=Spearman rho.
```

Architectural order.

---

# 149. Cliff's-delta matrix artifact

Similarly:

```text
error_attention_high_low_cliffs_delta_matrix.csv.
```

---

# 150. No head sorting in matrices

Rows remain:

```text
H1..HH.
```

---

# 151. Layer head-mean cross-seed summary

Because layer head-mean is head-permutation invariant, Phase56 may create:

```text
mean across seeds
sample SD across seeds
```

for each layer-level association/cohort contrast.

Do not average raw attention across seeds first.

---

# 152. Correct cross-seed aggregation order

Correct:

```text
compute layer-level result per seed
→ aggregate result across 3 seeds.
```

Incorrect:

```text
average three seeds’ attention tensors
→ compute one result.
```

---

# 153. Cross-seed layer summary schema

`error_attention_layer_cross_seed_summary.csv`:

```text
layer
analysis_type
conditioning_variable
metric
seed42_value
seed123_value
seed2026_value
mean_across_seeds
sample_sd_across_seeds
min
max
status
```

---

# 154. No cross-seed inferential claim

Three seeds are not enough for broad population inference.

Use:

```text
descriptive stability context.
```

Formal attention stability is Phase57.

---

# 155. Error-decile monotonicity summary

Optional transparent descriptor:

For each seed/layer/head/metric compute:

```text
Spearman(
decile_index 1..10,
decile_median_metric
).
```

Label:

```text
DECILE_TREND_DESCRIPTIVE.
```

Because only 10 aggregate points.

Not a replacement for target-level Spearman.

---

# 156. No thresholding decile trend

Report coefficient only.

---

# 157. Error-conditioned attention profile integrity

Every cohort mean profile:

```text
finite
nonnegative
sum≈1.
```

Hard.

---

# 158. Cliff's delta computational integrity

Ensure:

```text
ties contribute 0
delta ∈ [-1,1].
```

Can use efficient algorithm; result semantics must match definition.

---

# 159. Wasserstein integrity

Use:

```text
lag support in minutes
```

and cohort mean probability profiles.

No raw source-index units.

---

# 160. JSD integrity

Use:

```text
natural log
```

consistent with Phase55.

Range:

```text
0..ln(2).
```

---

# 161. Continuous association integrity

Spearman input arrays must have:

```text
same target IDs
same order after join
finite values.
```

---

# 162. Constant attention metric edge case

If a metric is constant within a head:

```text
Spearman undefined.
```

Return:

```text
NOT_DEFINED_CONSTANT_METRIC.
```

Do not force zero.

---

# 163. Constant error edge case

Not expected.

If all error values identical:

```text
Spearman undefined.
```

Mark clearly.

---

# 164. Under/over empty-group edge case

If:

```text
N_under=0
or
N_over=0
```

signed comparison:

```text
NOT_APPLICABLE.
```

No synthetic group.

---

# 165. High/low cohort minimum N

`n_edge >=1` by rule.

If:

```text
n_edge < 30
```

comparison proceeds with:

```text
SMALL_N_WARNING.
```

---

# 166. No statistical significance gate

Phase56 PASS does not require:

```text
strong association
large Cliff's delta
different profiles.
```

Null/weak differences are valid findings.

---

# 167. Main report hierarchy

To avoid hundreds of head-specific results, main report prioritizes:

```text
1. Layer head-mean continuous association
2. Per-head coefficient matrices for core metrics
3. High-vs-low layer head-mean profiles
4. Per-head high-vs-low profile distance matrices/tables
5. Under-vs-over layer/head summary
6. Shared-cohort layer-level results
7. Full machine tables in appendix/artifacts.
```

---

# 168. No cherry-picked strongest head in body

Do not select the head with largest `|rho|` as “representative”.

Use complete matrices or architectural order.

---

# 169. Figure ERRORATTN_56_01 — AE association heatmaps

For each seed/layer:

```text
rows=heads
columns=CORE_ATTENTION_METRICS-v1
value=Spearman rho(AE, metric).
```

Scale:

```text
-1..1.
```

---

# 170. Figure ERRORATTN_56_02 — Signed residual association heatmaps

Same.

---

# 171. Figure ERRORATTN_56_03 — High-vs-low Cliff's delta

Rows:

```text
heads
```

columns:

```text
core metrics
```

scale:

```text
-1..1.
```

---

# 172. Figure ERRORATTN_56_04 — Layer head-mean associations

Compact:

```text
seed × layer
```

for six core metrics.

Useful main-text figure.

---

# 173. Figure ERRORATTN_56_05 — High vs low temporal profiles

Per:

```text
seed × layer
```

show head-mean:

```text
LOW_ERROR
HIGH_ERROR.
```

x:

```text
lag
```

y:

```text
attention weight.
```

---

# 174. Figure ERRORATTN_56_06 — High-minus-low profile difference

Same layer head-mean profile:

```text
D_HL(k).
```

Use zero-centered y-axis.

---

# 175. Figure ERRORATTN_56_07 — Error-decile metric trends

Core metrics.

Architectural head order.

No smooth curve.

---

# 176. Figure ERRORATTN_56_08 — Error-decile temporal allocation

Can use layer head-mean non-overlapping lag-bin masses by decile.

This is readable and avoids ten dense raw profiles.

---

# 177. Figure ERRORATTN_56_09 — Under vs over metrics

Per seed/layer/head or layer mean.

Focus on:

```text
expected lag
recent1h
entropy
Lag80.
```

---

# 178. Figure ERRORATTN_56_10 — Under vs over temporal profile

Layer head-mean.

---

# 179. Figure ERRORATTN_56_11 — Shared low vs shared high

Layer head-mean, each seed.

Same target cohorts across seeds.

---

# 180. Figure ERRORATTN_56_12 — Full-matrix error association

Secondary.

Rows:

```text
heads
```

columns:

```text
full-matrix metrics.
```

---

# 181. Figure ERRORATTN_56_13 — Error cohort regime composition

Context-only.

Shows why attention differences may coincide with different data regimes.

---

# 182. Figure ERRORATTN_56_14 — Worst-case context

Use Phase51 shared ranks 1–5 only.

Plot their core attention metrics in architectural layout.

No additional case selection.

---

# 183. No plot sorting by effect magnitude

Architectural head order.

Error decile order.

Seed order:

```text
42
123
2026.
```

---

# 184. Findings codes

Possible:

```text
ABS_ERROR_ASSOCIATED_WITH_ATTENTION_CONCENTRATION
ABS_ERROR_ASSOCIATED_WITH_MORE_DIFFUSE_ATTENTION
ABS_ERROR_ASSOCIATED_WITH_MORE_RECENT_ATTENTION
ABS_ERROR_ASSOCIATED_WITH_LESS_RECENT_ATTENTION
ABS_ERROR_ASSOCIATED_WITH_LONGER_EXPECTED_LAG
ABS_ERROR_ASSOCIATED_WITH_SHORTER_EXPECTED_LAG
ABS_ERROR_ASSOCIATED_WITH_LARGER_LAG80
ABS_ERROR_ASSOCIATED_WITH_SMALLER_LAG80
HIGH_LOW_PROFILE_SHIFT_PRESENT
HIGH_LOW_PROFILE_SHIFT_SMALL_DESCRIPTIVE
HIGH_ERROR_RECENT_MASS_LOWER
HIGH_ERROR_RECENT_MASS_HIGHER
HIGH_ERROR_ENTROPY_HIGHER
HIGH_ERROR_ENTROPY_LOWER
UNDER_OVER_ATTENTION_DIFFERENCE_PRESENT
UNDER_OVER_PROFILE_SHIFT_PRESENT
ERROR_DECILE_PATTERN_MONOTONIC_DESCRIPTIVE
ERROR_DECILE_PATTERN_NONMONOTONIC
LAYER_HEAD_MEAN_ASSOCIATION_CONSISTENT_ACROSS_SEEDS
LAYER_HEAD_MEAN_ASSOCIATION_VARIES_ACROSS_SEEDS
FULL_MATRIX_ERROR_ASSOCIATION_PRESENT
REGIME_COMPOSITION_DIFFERS_BETWEEN_ERROR_COHORTS
SHARED_HARDNESS_ASSOCIATION_PRESENT
NO_CLEAR_ERROR_ATTENTION_ASSOCIATION
NO_BEST_HEAD_SELECTED
NO_HEAD_PRUNING
NO_RETUNING
NO_CAUSAL_CLAIM
READY_FOR_SEED_STABILITY_ATTENTION
```

Any qualitative code must be backed by actual values.

---

# 185. Safe findings language

Safe:

> For seed 42, higher absolute forecast error was associated with lower recent-one-hour attention mass in Layer 1 Head 2, with Spearman ρ = X.

Safe:

> The high-error cohort exhibited a larger median normalized entropy than the low-error cohort for this head, indicating a more diffuse last-query attention distribution in the high-error subset.

Safe:

> The layer head-mean temporal profile shifted toward older lags in the high-error cohort, with a Wasserstein distance of X minutes between high- and low-error profiles.

Safe:

> These associations are descriptive and do not establish that the attention pattern caused the forecast error.

---

# 186. Unsafe findings language

Do not say:

```text
attention caused the error
the model failed because it attended too far back
Head2 is bad
Head4 should be removed
high-error targets prove the model ignores recent history.
```

---

# 187. Causal caveat

Attention and forecast error are co-observed outcomes of the same forward pass.

Conditioning on realized error does not create causal identification.

---

# 188. Error cohort selection-bias caveat

HIGH_ERROR is explicitly selected by outcome severity.

Therefore any cohort difference is:

```text
diagnostic
selection-conditioned.
```

No population-causal interpretation.

---

# 189. Shared-hardness caveat

Shared hardness uses three final seeds only.

It does not represent the full distribution over random initializations.

---

# 190. Head semantic caveat

Per-head error-conditioned results remain:

```text
seed-specific.
```

Do not aggregate same head index across seeds.

---

# 191. Layer head-mean caveat

Head averaging is useful for permutation invariance but may hide specialized head behaviors.

Therefore both:

```text
per-head
and
layer head-mean
```

results are retained.

---

# 192. Mean profile caveat

Cohort mean attention can blur multimodal target-specific patterns.

Use together with:

```text
metric distributions
decile trends
per-target associations.
```

---

# 193. Multiple metrics caveat

Core metrics are mathematically related.

Do not count six correlated metrics as six independent confirmations.

---

# 194. No weighted evidence score

No.

---

# 195. Output directory

```text
artifacts/
└── error_conditioned_attention/
    ├── error_conditioned_attention_manifest.json
    ├── error_conditioned_attention_contract.json
    ├── phase56_preflight_audit.csv
    ├── error_attention_source_verification.csv
    ├── error_conditioning_assignment.csv
    ├── error_conditioning_assignment_audit.csv
    ├── shared_error_conditioning_audit.csv
    ├── error_conditioning_assignment_fingerprint.json
    ├── error_attention_join_audit.csv
    ├── error_attention_association_long.csv
    ├── error_attention_association_matrix.csv
    ├── error_attention_decile_metric_summary.csv
    ├── error_attention_decile_profile_by_lag.csv
    ├── error_attention_high_low_metric_comparison.csv
    ├── error_attention_high_low_cliffs_delta_matrix.csv
    ├── error_attention_high_low_profile_comparison.csv
    ├── error_attention_high_low_profile_difference_by_lag.csv
    ├── error_attention_signed_metric_comparison.csv
    ├── error_attention_signed_profile_comparison.csv
    ├── error_attention_signed_profile_difference_by_lag.csv
    ├── error_attention_layer_head_mean_metrics_long.csv
    ├── error_attention_layer_head_mean_association.csv
    ├── error_attention_layer_head_mean_high_low.csv
    ├── error_attention_shared_cohort_layer_summary.csv
    ├── error_attention_layer_cross_seed_summary.csv
    ├── error_attention_full_matrix_association.csv
    ├── error_cohort_regime_composition.csv
    ├── error_conditioned_worst_case_attention_context.csv
    ├── error_conditioned_attention_findings.csv
    ├── error_conditioned_attention_tests.csv
    ├── error_conditioned_attention_discrepancies.json
    ├── phase57_seed_stability_attention_handoff.json
    ├── phase58_attention_results_context_handoff.json
    ├── error_conditioned_attention_summary.json
    ├── error_conditioned_attention_report.md
    ├── figures/
    │   ├── ERRORATTN_56_01_abs_error_association_heatmaps.png
    │   ├── ERRORATTN_56_02_signed_residual_association_heatmaps.png
    │   ├── ERRORATTN_56_03_high_low_cliffs_delta.png
    │   ├── ERRORATTN_56_04_layer_head_mean_associations.png
    │   ├── ERRORATTN_56_05_high_vs_low_temporal_profiles.png
    │   ├── ERRORATTN_56_06_high_minus_low_profile_difference.png
    │   ├── ERRORATTN_56_07_error_decile_metric_trends.png
    │   ├── ERRORATTN_56_08_error_decile_lag_bin_allocation.png
    │   ├── ERRORATTN_56_09_under_vs_over_metrics.png
    │   ├── ERRORATTN_56_10_under_vs_over_temporal_profiles.png
    │   ├── ERRORATTN_56_11_shared_low_vs_shared_high.png
    │   ├── ERRORATTN_56_12_full_matrix_error_association.png
    │   ├── ERRORATTN_56_13_error_cohort_regime_composition.png
    │   └── ERRORATTN_56_14_shared_worst_case_context.png
    ├── README_ERROR_CONDITIONED_ATTENTION.md
    └── phase_56_signoff.json
```

---

# 196. Required outputs

```text
O56.1  Analysis manifest
O56.2  Analysis contract
O56.3  Preflight audit
O56.4  Source verification
O56.5  Frozen error cohort assignment
O56.6  Error cohort assignment audit
O56.7  Shared cohort audit
O56.8  Assignment fingerprint
O56.9  Error-attention join audit
O56.10 Continuous association table
O56.11 Association matrices
O56.12 Error-decile metric summaries
O56.13 Error-decile temporal profiles
O56.14 High-vs-low metric comparison
O56.15 High-vs-low Cliff's-delta matrices
O56.16 High-vs-low profile distances
O56.17 High-minus-low profile differences
O56.18 Under-vs-over metric comparison
O56.19 Under-vs-over profile distances
O56.20 Under-minus-over profile differences
O56.21 Layer head-mean per-target metrics
O56.22 Layer head-mean associations
O56.23 Layer head-mean high-vs-low comparison
O56.24 Shared-cohort layer summary
O56.25 Cross-seed layer descriptive summary
O56.26 Full-matrix error association
O56.27 Error-cohort regime composition
O56.28 Worst-case context table
O56.29 Core figures
O56.30 Findings
O56.31 Phase57 handoff
O56.32 Phase58 context handoff
O56.33 Tests
O56.34 Discrepancy log
O56.35 Summary JSON
O56.36 Human-readable report
O56.37 README
O56.38 Sign-off
```

---

# 197. Analysis manifest

`error_conditioned_attention_manifest.json`:

```text
phase=56
version=ERROR_CONDITIONED_ATTENTION-v1
source_phase55_version
source_phase54_version
source_phase52_version
source_phase49_version
source_phase50_version
source_phase51_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
residual_definition=Y_TRUE_MINUS_Y_PRED
primary_error_variable=ABS_ERROR
core_attention_metrics=[
  NORMALIZED_ENTROPY,
  EXPECTED_LAG_MINUTES,
  RECENT_1H_MASS,
  RECENT_6H_MASS,
  TOP5_MASS,
  LAG80_MINUTES
]
seed_error_cohort_rule=RANK_BASED_20_60_20
seed_error_deciles=10
shared_error_cohort_rule=RANK_BASED_20_60_20
continuous_association=SPEARMAN
high_low_effect_size=CLIFFS_DELTA
new_attention_extraction=false
new_test_inference=false
model_training=false
best_head_selection=false
best_seed_selection=false
post_test_correction=false
status
created_at
```

---

# 198. Analysis contract

`error_conditioned_attention_contract.json` must freeze:

```text
Error:
residual=y_true-y_pred
AE=abs(residual).

C1:
continuous Spearman associations.

C2:
per-seed error cohorts
rank-based bottom20/middle60/top20
plus 10 exact rank deciles.

C3:
shared-hardness cohorts
same 20/60/20 and deciles.

Core attention metrics:
normalized entropy
expected lag minutes
recent1h
recent6h
top5 mass
Lag80 minutes.

High-vs-low:
metric differences
Cliff's delta
profile JSD/L1/Cosine/Wasserstein.

Signed:
UNDER vs OVER.

Layer summary:
metric of head-mean vector.

Secondary:
full-matrix summary associations.

No:
p-value fishing
best head
head pruning
seed selection
regime retuning
cartesian subgroup mining
causal claims
retraining
prediction correction.
```

---

# 199. Preflight audit

`phase56_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Required:

```text
Phase55 approved
phase56_ready=true
Phase54 metrics complete
Phase52 raw last-query available
Phase49 residuals complete
Phase50 regime assignment frozen
Phase51 shared hardness available
same Test population
same seeds
same target IDs
same layer/head counts
core attention metrics finite
error-conditioning contract frozen before attention join
no new extraction required.
```

---

# 200. Source verification schema

`error_attention_source_verification.csv`:

```text
source_id
path
sha256_if_available
row_count
seed_count
target_count
layer_count_if_applicable
head_count_if_applicable
population_sha256
frozen
status
```

---

# 201. Assignment fingerprint

`error_conditioning_assignment_fingerprint.json`:

```text
assignment_sha256
residual_source_sha256
shared_hardness_source_sha256
test_population_sha256
cohort_rule
decile_rule
tie_break_rule
created_before_attention_join=true
status
```

---

# 202. Findings artifact

`error_conditioned_attention_findings.csv`:

```text
finding_id
scope
seed
layer
head_if_applicable
conditioning_variable
attention_metric
statistic
value
supporting_artifact
interpretation
causal_claim=false
model_change=false
status
```

---

# 203. Tests artifact

`error_conditioned_attention_tests.csv`:

```text
test_id
expected
observed
critical
status
```

Include all:

```text
source
assignment
coverage
join
profile sum
difference sum
JSD
Wasserstein
Cliff's delta
Spearman
ordering
scope
no-retuning checks.
```

---

# 204. Discrepancy taxonomy

`error_conditioned_attention_discrepancies.json`:

```text
PHASE55_NOT_APPROVED
PHASE56_HANDOFF_NOT_READY
SOURCE_TABLE_MISSING
SOURCE_SHA_MISMATCH
TEST_POPULATION_MISMATCH
SEED_MISSING
TARGET_MISSING
TARGET_DUPLICATE
LAYER_MISSING
HEAD_MISSING
RESIDUAL_CONVENTION_DRIFT
ABS_ERROR_MISMATCH
SHARED_HARDNESS_MISMATCH
ERROR_COHORT_RULE_NOT_FROZEN
ERROR_COHORT_TIE_RULE_DRIFT
ERROR_COHORT_COVERAGE_MISMATCH
ERROR_COHORT_OVERLAP
ERROR_DECILE_COVERAGE_MISMATCH
SHARED_COHORT_MISMATCH_ACROSS_SEEDS
ATTENTION_JOIN_TARGET_DROP
ATTENTION_JOIN_DUPLICATE
CORE_METRIC_MISSING
CORE_METRIC_NONFINITE
SPEARMAN_TARGET_ALIGNMENT_MISMATCH
SPEARMAN_FORCED_ZERO_ON_CONSTANT_INPUT
P_VALUE_USED_AS_HEADLINE
CLIFFS_DELTA_OUT_OF_RANGE
CLIFFS_DELTA_THRESHOLD_LABEL_ADDED_POST_HOC
PROFILE_SUM_MISMATCH
HIGH_LOW_DIFFERENCE_SUM_MISMATCH
UNDER_OVER_DIFFERENCE_SUM_MISMATCH
JSD_INVALID
WASSERSTEIN_NOT_IN_MINUTES
LAG_DIRECTION_REVERSED
HEAD_ROWS_SORTED_BY_EFFECT
BEST_HEAD_SELECTED
BEST_SEED_SELECTED
HEAD_PRUNING_ATTEMPT
HEAD_ABLATION_ATTEMPT
MODEL_RETRAINING_ATTEMPT
PREDICTION_BIAS_CORRECTION_ATTEMPT
PREDICTION_CLIPPING_ATTEMPT
PREDICTION_SHIFT_ATTEMPT
TEST_ERROR_THRESHOLD_USED_AS_DEPLOYMENT_REGIME
ERROR_COHORT_USED_FOR_RETUNING
PHASE50_REGIME_THRESHOLD_CHANGED
REGIME_ERROR_HEAD_CARTESIAN_MINING
WORST_CASE_RESELECTED_BY_ATTENTION
CROSS_SEED_HEAD_INDEX_AVERAGED
SAME_INDEX_HEAD_SEMANTIC_ALIGNMENT_ASSUMED
NEW_ATTENTION_EXTRACTION_ATTEMPT
NEW_TEST_INFERENCE_ATTEMPT
HEATMAP_IMAGE_USED_AS_NUMERIC_SOURCE
FEATURE_IMPORTANCE_CLAIM
CAUSAL_ATTRIBUTION_CLAIM
OTHER
```

---

# 205. Status model

## PASS

```text
all sources verified
error cohort rules frozen
all targets assigned
attention joins complete
continuous associations complete
error deciles complete
high-vs-low metric/profile analyses complete
under-vs-over analysis complete
layer head-mean analysis complete
shared-cohort analysis complete
full-matrix secondary analysis complete
no retuning/head selection
Phase57 handoff ready.
```

## PASS_WITH_WARNING

Possible:

```text
weak/no error-attention associations
small signed subgroup
constant metric in a head
large cohort regime-composition differences
MEAN pooling caveat
lookback truncates longer metrics
strong seed variation.
```

These are findings/limitations, not methodological failure.

## FAIL

Examples:

```text
cohorts altered after attention inspection
target rows dropped
profile integrity fails
best head chosen
error cohort fed into retuning
new Test inference.
```

---

# 206. Phase57 handoff purpose

Phase57 will ask:

> Are attention patterns and error-conditioned attention findings stable across seeds after accounting for head permutation?

Phase56 supplies:

```text
seed-specific error associations
shared-error common target cohorts
layer head-mean error-conditioned profiles
head-specific high/low profile summaries.
```

---

# 207. Phase57 handoff schema

`phase57_seed_stability_attention_handoff.json`:

```text
source_phase56_version
source_phase55_version
source_phase54_version
source_phase52_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
error_conditioning_assignment_sha256
shared_error_cohorts_common_across_seeds=true
error_attention_association_long
high_low_metric_comparison
high_low_profile_comparison
layer_head_mean_association
shared_cohort_layer_summary
raw_last_query_refs
head_behavior_refs
same_index_head_semantic_alignment_assumed=false
head_matching_not_performed_in_phase56=true
best_head_selected=false
model_retrained=false
ready_for_phase57=true
```

---

# 208. Phase58 context handoff

`phase58_attention_results_context_handoff.json`:

```text
source_phase56_version
core_attention_metrics
primary_error_conditioning_results
layer_head_mean_results
high_low_profile_results
under_over_results
shared_cohort_results
full_matrix_secondary_results
regime_context_results
findings
causal_claim=false
model_change=false
ready_for_phase58_context=true
```

---

# 209. Execution sequence

```text
1. Verify Phase55/54/52 attention sources.
2. Verify Phase49/50/51 error/context sources.
3. Freeze Phase56 contract and core metric set.
4. Build seed-specific AE ranks/cohorts/deciles using error only.
5. Build shared-hardness cohorts/deciles using shared hardness only.
6. Freeze assignment/checksum before attention join.
7. Join Phase54 attention metrics by seed/target.
8. Audit expected row counts and target coverage.
9. Compute Spearman AE associations.
10. Compute Spearman signed-residual associations.
11. Compute secondary shared-hardness associations.
12. Compute error-decile metric summaries.
13. Build decile-conditioned mean temporal profiles.
14. Compute HIGH vs LOW metric differences and Cliff's delta.
15. Compute HIGH vs LOW profile JSD/L1/Cosine/Wasserstein.
16. Build high-minus-low profile-by-lag tables.
17. Compute UNDER vs OVER metric differences.
18. Compute UNDER vs OVER profile distances/differences.
19. Reconstruct per-target layer head-mean vectors from raw Phase52 attention.
20. Compute layer head-mean metrics.
21. Compute layer head-mean continuous/high-low/shared-cohort analyses.
22. Compute secondary full-matrix summary associations.
23. Annotate error cohorts with frozen Phase50 regime composition.
24. Attach deterministic Phase51 worst-case context.
25. Generate complete matrices/figures in architectural order.
26. Write findings with explicit non-causal language.
27. Write Phase57 and Phase58 handoffs.
28. Run integrity/scope/discrepancy tests.
29. Write summary/report/README.
30. Sign off.
```

---

# 210. Recommended pseudocode

```text
p55 = load_phase55_signoff()
assert p55.overall_status in {"PASS","PASS_WITH_WARNING"}

attn_metrics = load_phase54_last_query_metrics()
raw_attn = load_phase52_last_query_raw()
residuals = load_phase49_residuals()
regimes = load_phase50_test_regime_assignment()
shared_hardness = load_phase51_shared_hardness()

verify_all_sources_same_test_population()

freeze_phase56_contract(
    core_metrics=[
        "normalized_entropy",
        "expected_lag_minutes",
        "recent_1h_mass",
        "recent_6h_mass",
        "top5_mass",
        "lag80_minutes"
    ],
    cohort_rule="RANK_20_60_20",
    decile_rule="EXACT_RANK_10",
    association="SPEARMAN",
    high_low_effect="CLIFFS_DELTA"
)

assignments = []

for seed in [42,123,2026]:

    e = get_seed_residuals(residuals, seed)

    ranked = deterministic_rank(
        rows=e,
        primary="absolute_error_wh ASC",
        tie_break=[
            "timestamp ASC",
            "target_id ASC"
        ]
    )

    n = len(ranked)
    n_edge = max(1, floor(0.20*n))

    ranked["seed_error_cohort"] = assign_20_60_20(
        ranked,
        n_edge=n_edge
    )

    ranked["seed_error_decile"] = assign_exact_rank_decile(
        rank0=range(n),
        N=n,
        K=10
    )

    ranked["residual_sign_group"] = exact_sign_group(
        residual=e.residual_wh
    )

    assignments.append(ranked)

shared_ranked = deterministic_rank(
    shared_hardness,
    primary="mean_abs_error_across_seeds ASC",
    tie_break=[
        "timestamp ASC",
        "target_id ASC"
    ]
)

shared_ranked = assign_shared_20_60_20_and_deciles(shared_ranked)

assignment = merge_seed_and_shared_assignments(assignments, shared_ranked)

freeze_and_checksum_before_attention_join(assignment)

joined = join_attention_metrics(
    attn_metrics,
    assignment,
    keys=["seed","target_id"]
)

verify_expected_join_rows(
    joined,
    seeds=3,
    targets=N_test,
    layers=N_layers,
    heads=N_heads
)

association_rows = []

for seed in [42,123,2026]:
    for layer in layers:
        for head in heads:
            df = select(joined, seed, layer, head)

            for metric in CORE_METRICS:

                association_rows.append(
                    spearman_record(
                        x=df.absolute_error_wh,
                        y=df[metric],
                        conditioning="ABS_ERROR"
                    )
                )

                association_rows.append(
                    spearman_record(
                        x=df.residual_wh,
                        y=df[metric],
                        conditioning="SIGNED_RESIDUAL"
                    )
                )

                association_rows.append(
                    spearman_record(
                        x=df.shared_hardness,
                        y=df[metric],
                        conditioning="SHARED_HARDNESS"
                    )
                )

            compute_error_decile_metric_summary(df)

            high = df[df.seed_error_cohort == "HIGH_ERROR"]
            low  = df[df.seed_error_cohort == "LOW_ERROR"]

            for metric in CORE_METRICS:
                compute_high_low_summary_and_cliffs_delta(
                    high[metric],
                    low[metric]
                )

            P_high = mean_raw_attention_profile(
                raw_attn,
                seed,
                layer,
                head,
                target_ids=high.target_id
            )

            P_low = mean_raw_attention_profile(
                raw_attn,
                seed,
                layer,
                head,
                target_ids=low.target_id
            )

            verify_probability_profile(P_high)
            verify_probability_profile(P_low)

            compare_profiles(
                P_high,
                P_low,
                metrics=[
                    "JSD",
                    "L1",
                    "COSINE",
                    "WASSERSTEIN_MINUTES"
                ]
            )

            D_hl = P_high - P_low
            assert sum(D_hl) ≈ 0

            under = df[df.residual_wh > 0]
            over  = df[df.residual_wh < 0]

            if len(under) > 0 and len(over) > 0:
                compute_under_over_metric_comparison(...)
                compute_under_over_profile_comparison(...)

# Decile profiles from raw arrays
compute_decile_conditioned_profiles(
    raw_attn,
    assignment
)

# Permutation-invariant layer head-mean vectors
layer_mean_vectors = raw_attn.mean(axis="head")

layer_mean_metrics = compute_phase54_metrics_on_layer_mean_vectors(
    layer_mean_vectors
)

analyze_layer_mean_vs_error(
    layer_mean_metrics,
    assignment
)

analyze_shared_error_cohorts_at_layer_mean(
    layer_mean_metrics,
    assignment
)

# Secondary all-query summaries
full_matrix = load_phase52_full_matrix_summary()
compute_full_matrix_error_spearman(
    full_matrix,
    assignment
)

regime_context = summarize_frozen_phase50_regimes_by_error_cohort(
    assignment,
    regimes
)

attach_phase51_worst_case_context()

generate_all_figures_in_architectural_order()

assert no_best_head
assert no_best_seed
assert no_head_pruning
assert no_model_retraining
assert no_prediction_correction
assert no_new_attention_extraction
assert no_causal_claim

write_phase57_handoff()
write_phase58_context_handoff()

signoff_phase56()
```

---

# 211. Preflight acceptance checklist

```text
[ ] Phase55 PASS/PASS_WITH_WARNING.
[ ] phase56_ready=true.
[ ] Phase54 per-target metrics available.
[ ] Phase52 raw last-query files available.
[ ] Phase52 full-matrix summary available.
[ ] Phase49 residuals available.
[ ] Phase50 regime assignment frozen.
[ ] Phase51 shared hardness available.
[ ] Same FINAL_TEST_POP-v1.
[ ] Same seed list 42/123/2026.
[ ] Same target IDs.
[ ] Same layers/heads.
[ ] Residual sign convention verified.
[ ] Core metric set frozen.
[ ] Cohort rules frozen before attention join.
[ ] No new inference/extraction required.
```

---

# 212. Cohort acceptance checklist

```text
[ ] Rank uses abs error ascending.
[ ] Tie-break timestamp ascending.
[ ] Tie-break target_id ascending.
[ ] n_edge=max(1,floor(.20N)).
[ ] LOW count=n_edge.
[ ] HIGH count=n_edge.
[ ] MID gets remainder.
[ ] Cohorts mutually exclusive.
[ ] Cohorts cover all targets.
[ ] 10 deciles cover all targets.
[ ] Shared cohorts built from Phase51 shared hardness only.
[ ] Shared cohorts identical across seeds.
[ ] Assignment frozen before attention join.
[ ] Assignment SHA generated.
```

---

# 213. Join acceptance checklist

```text
[ ] Join key seed+target_id.
[ ] No error row unmatched.
[ ] No attention target unmatched.
[ ] No duplicate error row.
[ ] Every target has all layers.
[ ] Every target has all heads.
[ ] Expected total row count reconstructed.
[ ] Shared hardness consistent across seeds.
[ ] Regime labels loaded from frozen Phase50 source.
```

---

# 214. Continuous-association acceptance checklist

```text
[ ] Six core metrics complete.
[ ] AE Spearman complete.
[ ] Signed residual Spearman complete.
[ ] Shared-hardness Spearman complete as secondary.
[ ] Exact same target IDs per association.
[ ] Constant-input cases marked NOT_DEFINED.
[ ] No forced zero.
[ ] No p-value headline.
[ ] Head order architectural.
[ ] No |rho|-based sorting.
```

---

# 215. High-vs-low metric acceptance checklist

```text
[ ] N low/high recorded.
[ ] Means.
[ ] Medians.
[ ] Quantiles.
[ ] Delta mean high-low.
[ ] Delta median high-low.
[ ] Cliff's delta.
[ ] Cliff's delta in [-1,1].
[ ] No magnitude category threshold.
[ ] Small-N warning if needed.
```

---

# 216. Profile-comparison acceptance checklist

```text
[ ] P_LOW sums≈1.
[ ] P_HIGH sums≈1.
[ ] P_HIGH-P_LOW sums≈0.
[ ] JSD valid.
[ ] L1 valid.
[ ] Cosine valid.
[ ] Wasserstein units=minutes.
[ ] Same lag support.
[ ] No smoothing.
[ ] No profile renormalization hiding failure.
```

---

# 217. Error-decile acceptance checklist

```text
[ ] Deciles 1..10.
[ ] Complete coverage.
[ ] Approximately equal counts.
[ ] Every decile metric summary complete.
[ ] Every decile profile sums≈1.
[ ] Error order low→high.
[ ] No manually selected deciles.
[ ] No fitted trend required.
```

---

# 218. Signed-error acceptance checklist

```text
[ ] UNDER=residual>0.
[ ] OVER=residual<0.
[ ] ZERO=residual==0.
[ ] No epsilon.
[ ] Counts reported.
[ ] Under-vs-over metric comparison complete if applicable.
[ ] Under/over mean profiles sum≈1.
[ ] Under-minus-over sums≈0.
[ ] High-error sign subgroup optional/contextual.
[ ] No signed-bias correction.
```

---

# 219. Layer head-mean acceptance checklist

```text
[ ] Raw vectors averaged across heads per target/layer.
[ ] No cross-seed raw averaging.
[ ] Metrics recomputed from head-mean vector.
[ ] Nonlinear metrics not replaced by mean head metrics.
[ ] AE association complete.
[ ] Residual association complete.
[ ] Shared-hardness association complete.
[ ] High-vs-low comparison complete.
[ ] Shared low/high comparison complete.
[ ] Layer-level cross-seed result aggregation occurs after per-seed computation.
```

---

# 220. Full-matrix acceptance checklist

```text
[ ] Phase52 full-matrix summary source verified.
[ ] Mean-query entropy association.
[ ] Mean self-attention association.
[ ] Mean query-source distance association.
[ ] Forward-within-input mass association.
[ ] AE and residual conditioning.
[ ] No full dense re-extraction.
[ ] Forward-within-input not mislabeled future leakage.
```

---

# 221. Regime-context acceptance checklist

```text
[ ] Phase50 labels unchanged.
[ ] Error cohort regime composition reported.
[ ] No new regime threshold.
[ ] No Test regime retuning.
[ ] No full error×regime×head mining.
[ ] Context used to qualify interpretation only.
```

---

# 222. Worst-case acceptance checklist

```text
[ ] Phase51 W2 shared top20 retained.
[ ] Shared ranks1–5 deterministic report examples.
[ ] No attention-based case replacement.
[ ] Worst cases not treated as full statistical cohort.
[ ] Error-conditioned whole-Test evidence remains primary.
```

---

# 223. Figure acceptance checklist

```text
[ ] Head rows architectural.
[ ] Seed order 42/123/2026.
[ ] Error deciles 1→10.
[ ] Spearman heatmaps scale -1..1.
[ ] Cliff's delta scale -1..1.
[ ] High/low profiles share y-scale.
[ ] Difference plots zero-centered.
[ ] Lag axis newest→older.
[ ] No smoothing.
[ ] No sorted “strongest effect” plots.
[ ] Layer head-mean figures included.
```

---

# 224. Scope acceptance checklist

```text
[ ] No best head.
[ ] No best seed.
[ ] No head pruning.
[ ] No head ablation.
[ ] No model retraining.
[ ] No Test metric recomputation after modification.
[ ] No bias correction.
[ ] No clipping.
[ ] No time shifting.
[ ] No Test error cohort used as deployment regime.
[ ] No Phase50 threshold changes.
[ ] No cartesian subgroup mining.
[ ] No cross-seed semantic head averaging.
[ ] No new attention extraction.
[ ] No feature-importance claim.
[ ] No causal claim.
```

---

# 225. Acceptance criteria

Phase56 PASS only when:

```text
All attention and error sources are verified and aligned to the exact same FINAL_TEST_POP-v1.

The residual convention remains y_true-y_pred and absolute error is the primary error-magnitude conditioning variable.

The six core attention metrics are frozen before analysis.

Seed-specific error cohorts are created solely from deterministic absolute-error ranks using a predeclared bottom20/middle60/top20 rule, and ten exact rank-based error deciles cover all Test targets.

Shared-hardness cohorts are created solely from the Phase51 shared-hardness ranking and are identical across seeds.

All cohort assignments are frozen and checksummed before any attention metric/profile join.

Continuous per-head analyses report Spearman associations between attention metrics and absolute error, signed residual and secondary shared hardness without naive iid p-value claims.

High-error vs low-error comparisons report transparent metric differences and Cliff's delta without post-hoc effect-size categories.

High-error and low-error cohort mean temporal profiles remain valid probability distributions and are compared using JSD, L1, cosine similarity and Wasserstein distance in minutes.

High-minus-low profile differences sum to approximately zero and are never interpreted causally.

Underprediction vs overprediction attention behavior is compared using the exact residual sign convention without a sign epsilon.

Error-decile attention metrics and temporal profiles are produced for all ten deterministic deciles rather than selectively showing favorable groups.

Permutation-invariant layer head-mean vectors are computed per target/layer, their metrics are recomputed from the averaged vectors, and layer-level error-conditioned analyses are retained alongside per-head results.

Shared-error layer-level comparisons are computed separately per seed and only then summarized across seeds.

Secondary all-query/full-matrix attention summaries are related to error without re-extracting dense attention.

Frozen Phase50 regimes are used only to describe error-cohort composition; no new regime thresholds or cartesian subgroup mining is introduced.

Frozen Phase51 worst cases are used only as deterministic examples and are not reselected using attention.

No best head or best seed is selected, no head is pruned/ablated, no model is retrained, no prediction is corrected and no Test result is altered.

All findings explicitly distinguish association/co-occurrence from causal explanation.

Phase57 receives seed-specific and layer head-mean error-conditioned artifacts without Phase56 assuming cross-seed head semantic alignment.
```

---

# 226. Failure conditions

Phase56 FAIL if:

```text
error cohorts are modified after viewing attention

attention metrics are used to choose cohort thresholds

Test error cohorts are treated as deployment regimes

rows are dropped during attention-error join

different target subsets are used across heads without explanation

best head is selected from association strength

same head index is averaged across seeds

a high-error attention finding triggers retraining

predictions are corrected after analysis

Phase50 regime thresholds are recomputed

post-hoc error×regime×head subgroup mining is performed

new attention is extracted

heatmap pixels are used numerically

profile sums fail

Wasserstein uses position indices but is labeled minutes

Cliff's delta is outside [-1,1]

naive p-values are used as the primary evidence

association is reported as causation.
```

---

# 227. Common mistakes

## 227.1 Dùng Test Q80 error rồi gọi đó là một “regime” giống Phase50

Không chuẩn.

Phase56 phải gọi:

```text
diagnostic HIGH_ERROR cohort.
```

## 227.2 Nhìn attention trước rồi quyết định top10% hay top20%

Sai post-hoc design.

Plan khóa:

```text
20/60/20
```

trước analysis.

## 227.3 Correlation mạnh nhất thuộc Head3 nên gọi Head3 là head quan trọng

Sai. Association với error không bằng predictive contribution.

## 227.4 Average Head1 qua seed42/123/2026

Sai vì semantic head alignment chưa được giải quyết.

## 227.5 Dùng p-value trên hàng nghìn timestamp như iid

Không chuẩn do temporal dependence.

## 227.6 HIGH_ERROR profile khác LOW_ERROR rồi kết luận profile đó gây lỗi

Sai causal interpretation.

## 227.7 High-error group có nhiều EXTREME_HIGH nên đổi Phase50 threshold

Sai.

## 227.8 Chỉ show decile1 và decile10 vì đẹp

Nếu machine analysis đủ 10 deciles, report có thể highlight endpoints nhưng không được bỏ nguồn đầy đủ hoặc chọn endpoint vì pattern.

## 227.9 Tính metric trên mean heads bằng cách average entropy của heads

Không tương đương.

Phải:

```text
mean vector first
→ recompute entropy.
```

## 227.10 Dùng squared error như một hệ conditioning độc lập khác AE

Redundant trong core Phase56.

## 227.11 Full-matrix forward mass gọi là future leakage

Sai. Đó là later-within-input attention.

## 227.12 Dùng worst top20 làm bằng chứng duy nhất

Sai. Whole-Test continuous/cohort analysis mới là primary.

---

# 228. Human-readable report structure

`error_conditioned_attention_report.md`:

```text
1. Objective
2. Why error-conditioned attention is diagnostic
3. Frozen attention/error sources
4. Difference between Phase50 regimes and Phase56 error cohorts
5. Error cohort construction and freeze
6. Continuous absolute-error associations
7. Signed residual associations
8. Error-decile attention trends
9. High-error vs low-error metric differences
10. High-error vs low-error temporal-profile shifts
11. Underprediction vs overprediction attention
12. Layer head-mean error-conditioned analysis
13. Shared-hardness common-target analysis
14. Full-matrix attention summary associations
15. Phase50 regime composition context
16. Phase51 worst-case examples
17. Cross-seed descriptive layer-level context
18. Main findings
19. Why no best head is selected
20. Why no retuning/correction is allowed
21. Why association is not causation
22. Handoff to seed-stability attention analysis
23. Limitations
24. Definition of Done
```

---

# 229. README requirements

`README_ERROR_CONDITIONED_ATTENTION.md` explains:

```text
why AE is the primary conditioning variable
why signed residual is analyzed separately
why Phase56 Test error cohorts are diagnostic rather than deployment regimes
how rank-based 20/60/20 cohorts are created
how exact rank deciles are assigned
what shared hardness means
why cohorts are frozen before attention joins
why Spearman is used
what Cliff's delta means
how high/low temporal profiles are compared
why Wasserstein is in minutes
why layer head-mean is useful
why per-head results are not averaged across seeds
why no p-value fishing is used
why no best head/retraining is allowed
why attention-error association is not causal.
```

---

# 230. Summary artifact

`error_conditioned_attention_summary.json`:

```text
version
source_phase55_version
source_phase54_version
source_phase52_version
source_phase49_version
source_phase50_version
source_phase51_version
final_lock_sha256
test_population_sha256
seed_list
core_attention_metrics
cohort_rule
assignment_sha256
association_status
decile_analysis_status
high_low_metric_status
high_low_profile_status
signed_analysis_status
layer_head_mean_status
shared_cohort_status
full_matrix_status
regime_context_status
worst_case_context_status
findings
best_head_selected=false
best_seed_selected=false
head_pruning=false
model_retrained=false
prediction_corrected=false
new_attention_extraction=false
causal_claim=false
phase57_ready
phase58_context_ready
overall_status
```

No runtime counts or effect values may be fabricated before execution.

---

# 231. Phase56 sign-off

`phase_56_signoff.json` minimum:

```text
phase=56
phase_name=Error-conditioned attention
version=ERROR_CONDITIONED_ATTENTION-v1
source_phase55_version
source_phase54_version
source_phase52_version
source_phase49_version
source_phase50_version
source_phase51_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
residual_definition=Y_TRUE_MINUS_Y_PRED
core_attention_metrics
cohort_rule=RANK_BASED_20_60_20
error_deciles=10
assignment_sha256
source_alignment_verified
cohort_coverage_verified
continuous_association_complete
error_decile_analysis_complete
high_low_metric_analysis_complete
high_low_profile_analysis_complete
signed_analysis_complete
layer_head_mean_analysis_complete
shared_cohort_analysis_complete
full_matrix_secondary_analysis_complete
regime_context_complete
worst_case_context_complete
best_head_selected=false
best_seed_selected=false
head_pruning=false
head_ablation=false
model_training=false
prediction_correction=false
new_attention_extraction=false
phase50_threshold_modified=false
cartesian_subgroup_mining=false
causal_claim=false
phase57_ready
phase58_context_ready
warnings
overall_status
created_at
```

---

# 232. Definition of Done

\[
\boxed{
Frozen\ Error\ Cohorts
+
Whole\text{-}Test\ Error\text{-}Attention\ Associations
+
Error\ Deciles
+
High/Low\ Metric\ Contrasts
+
High/Low\ Profile\ Shifts
+
Under/Over\ Analysis
+
Layer\ Head\text{-}Mean\ Analysis
+
Shared\ Hardness\ Context
+
No\ Retuning
+
No\ Best\ Head
+
Phase57\ Handoff
}
\]

---

# 233. Final status contract

```text
PHASE 56 analyzes attention conditioned on realized Test error.

Sources:
Phase49 residuals
Phase51 shared hardness
Phase54 last-query metrics
Phase52 raw last-query attention.

Error:
residual = y_true-y_pred
AE = |residual|.

Core attention metrics:
normalized entropy
expected lag minutes
recent1h mass
recent6h mass
top5 mass
Lag80 minutes.

Continuous:
Spearman vs AE
Spearman vs signed residual
secondary Spearman vs shared hardness.

Seed-specific cohorts:
bottom20% LOW_ERROR
middle60% MID_ERROR
top20% HIGH_ERROR
from exact deterministic AE ranks.

Error deciles:
1..10 exact rank-based.

Shared cohorts:
same 20/60/20 and deciles
from Phase51 shared hardness.

High vs low:
metric deltas
Cliff's delta
JSD/L1/Cosine/Wasserstein profile comparison.

Signed:
UNDER vs OVER.

Layer summary:
head-mean vector per target
metrics recomputed on head-mean vector.

Secondary:
full-matrix summary error associations.

Phase50 regimes:
context only
never redefined.

Phase51 worst cases:
deterministic examples only.

Forbidden:
p-value fishing
best head
best seed
head pruning
head ablation
retraining
prediction correction
Test error cohort as deployment regime
cartesian subgroup mining
cross-seed same-index head averaging
new attention extraction
feature-importance claim
causal claim.

After ERROR_CONDITIONED_ATTENTION-v1 PASS:
proceed to
PHASE 57 — Seed-Stability Attention Check.
```

---

# 234. Final check

Correct:

```text
verify frozen error + attention sources
→ freeze error cohorts using error only
→ checksum assignments
→ join attention
→ continuous associations
→ error deciles
→ high vs low profiles
→ under vs over
→ layer head-mean analysis
→ shared-hardness context
→ Phase57 handoff
```

Incorrect:

```text
inspect attention
→ choose error threshold that gives strongest difference
```

Incorrect:

```text
largest |rho| head
→ best head
```

Incorrect:

```text
high-error attention pattern
→ retrain final model
```

Incorrect:

```text
Head1 across seeds
→ average association
without head matching
```

Incorrect:

```text
attention differs in HIGH_ERROR
→ attention caused the error
```

Chỉ sau khi:

```text
ERROR_CONDITIONED_ATTENTION-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase57_ready = true
```

mới chuyển sang **PHASE 57 — Seed-Stability Attention Check**.
