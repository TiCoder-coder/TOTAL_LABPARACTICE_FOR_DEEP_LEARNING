# PHASE 51 — WORST-ERROR ANALYSIS

## Kế hoạch xác định, đóng băng và điều tra có hệ thống các worst-error cases của ba Final Transformer seeds trên Held-Out Test

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Residual convention:** `residual = y_true - y_pred`  
**Upstream regime analysis:** `ERROR_BY_REGIME-v1`  
**Primary error source:** frozen Phase47 Test prediction bundles + Phase49 residual tables  
**Regime annotation source:** frozen Phase50 Train-defined Test regime assignment  
**Phase ID:** `PHASE_51_WORST_ERROR_ANALYSIS`  
**Output version:** `WORST_ERROR_ANALYSIS-v1`  
**Phase trước:** `Phase_50_Error-by-regime_analysis.md`  
**Phase sau:** `Phase_52_Attention_extraction.md`

---

# 1. Vai trò của Phase 51

Phase 51 là bước **case-level failure analysis** cuối cùng trước khi chuyển sang attention analysis.

Phase47 đã trả lời:

```text
Final Test performance là bao nhiêu?
```

Phase48 trả lời:

```text
Prediction behavior nhìn chung như thế nào?
```

Phase49 trả lời:

```text
Residual có cấu trúc gì?
```

Phase50 trả lời:

```text
Model mắc lỗi nhiều hơn ở regime nào?
```

Phase51 trả lời:

> Những timestamp/case nào tạo ra các lỗi lớn nhất, lỗi đó có nhất quán qua ba seeds không, thuộc regime nào, mô hình đã nhìn thấy context nào trước khi forecast, và các case nào cần chuyển tiếp sang attention analysis?

Mục tiêu:

```text
1. Verify frozen residual/regime artifacts.
2. Freeze worst-case ranking rules trước narrative inspection.
3. Rank worst absolute-error cases cho từng seed.
4. Rank shared hard cases bằng cross-seed mean absolute error.
5. Tách worst underprediction và worst overprediction.
6. Đo overlap của worst cases giữa ba seeds.
7. Phân biệt shared systematic difficulty với seed-sensitive difficulty.
8. Annotate worst cases bằng frozen Phase50 regimes.
9. Build fixed local temporal context cho từng selected case.
10. Build model-visible input-window context summary mà không rerun model.
11. Compare Persistence/LSTM tại đúng các selected timestamps nếu source bundles tồn tại.
12. Tạo deterministic worst-case casebook.
13. Freeze worst-case IDs/checksums.
14. Handoff các case IDs và context sang Phase52 attention extraction.
15. Không chọn best seed.
16. Không sửa predictions/model sau khi nhìn worst cases.
```

Nguyên tắc trung tâm:

\[
\boxed{
Frozen\ Errors
+
Predeclared\ Ranking
+
Per\text{-}Seed\ Worst\ Cases
+
Shared\ Hard\ Cases
+
Signed\ Failure\ Cases
+
Frozen\ Regime\ Labels
+
Fixed\ Local\ Context
+
No\ Retuning
}
\]

---

# 2. Phase51 là analysis-only

Forbidden:

```text
new Test inference
model training
checkpoint tuning
feature tuning
lookback tuning
loss tuning
bias correction
prediction clipping
prediction time shifting
seed selection
regime threshold changes
```

Worst cases được dùng để:

```text
understand
document
prepare attention analysis
support limitations/future work.
```

Không dùng để sửa final Test result.

---

# 3. Source of truth

Required:

```text
phase_50_signoff.json
phase51_worst_error_regime_handoff.json
test_regime_assignment.csv
regime_thresholds_train_only.json
regime_metrics_long.csv
```

Required Phase49 residual artifacts:

```text
phase_49_signoff.json
residual_long_table.csv
residual_wide_table.csv
residual_seed_sign_consensus.csv
```

Required Phase48 seed-spread artifact:

```text
prediction_seed_spread.csv
```

Required original prediction provenance:

```text
phase_47_signoff.json
prediction_checksums.json
final_test_population_manifest.json
```

---

# 4. Why Phase51 uses Phase49 residuals as primary error vectors

Canonical error fields:

```text
residual_wh
absolute_error_wh
squared_error_wh2
```

have already been recomputed and verified against Phase47 metrics in Phase49.

Phase51 still verifies checksums/row alignment but does not independently create new model predictions.

---

# 5. Regime labels are immutable

Use exact:

```text
test_regime_assignment.csv
```

from Phase50.

Do not:

```text
recompute Q25/Q75/Q90
change rapid-change threshold
create new worst-case-specific regimes.
```

Worst cases are annotated using the already frozen labels.

---

# 6. Worst-error analysis universe

Canonical universe:

```text
FINAL_TEST_POP-v1
×
{seed42, seed123, seed2026}
```

with exactly:

```text
one residual row per seed-target.
```

No row deletion.

---

# 7. Three seeds remain symmetric

All three:

```text
42
123
2026
```

must receive the same ranking logic.

No seed is chosen as:

```text
representative
main
best
worst
```

based on Test error.

---

# 8. Ranking protocol must be frozen before case inspection

Before sorting any errors, create:

```text
worst_error_selection_contract.json
```

and checksum it.

This prevents changing:

```text
K
ranking score
tie-break
signed-case count
local context width
```

after seeing cases.

---

# 9. Primary case-selection families

Phase51 locks four required case families:

```text
W1 — Per-seed worst absolute-error cases
W2 — Shared hard cases across seeds
W3 — Worst underprediction cases
W4 — Worst overprediction cases
```

Plus two secondary diagnostics:

```text
W5 — Cross-seed overlap/consensus
W6 — Hardness vs seed-disagreement analysis.
```

---

# 10. W1 — Per-seed worst absolute errors

For each seed `s`, rank Test targets by:

\[
AE_{s,t}=|e_{s,t}|.
\]

Lock:

```text
K_ABS = 20.
```

Output exactly:

```text
top 20 seed42
top 20 seed123
top 20 seed2026
```

unless Test population has fewer than 20 cases.

---

# 11. W1 deterministic tie-break

Sort by:

```text
1. absolute_error_wh descending
2. target_timestamp ascending
3. target_id ascending.
```

No random tie resolution.

---

# 12. Why absolute error is primary case score

It is:

```text
directly interpretable in Wh
consistent with MAE
sign-neutral
```

and identifies large miss magnitude regardless of under/overprediction direction.

---

# 13. Squared-error ranking is intentionally not duplicated

Since:

\[
e^2
\]

is a strictly monotonic transformation of:

\[
|e|
\]

for nonnegative magnitudes, ranking by squared error yields the same order as absolute error.

Therefore Phase51 must **not** create a fake second “worst SSE” case list as if it were independent evidence.

Squared error remains stored for contribution/context only.

---

# 14. W2 — Shared hard cases across seeds

For each Test target `t` compute:

\[
SharedHardness_t
=
\frac{
|e_{42,t}|+
|e_{123,t}|+
|e_{2026,t}|
}{3}.
\]

Canonical name:

```text
MEAN_ABS_ERROR_ACROSS_SEEDS.
```

Lock:

```text
K_SHARED = 20.
```

---

# 15. Shared hardness is not an ensemble metric

`SharedHardness_t` measures:

```text
how difficult one Test target was across the three official seeds.
```

It is not:

```text
error of an ensemble prediction
new Test performance metric
model-selection score.
```

---

# 16. W2 deterministic tie-break

Sort:

```text
1. mean_abs_error_across_seeds descending
2. target_timestamp ascending
3. target_id ascending.
```

---

# 17. Why W2 matters

A case can be:

```text
bad for one seed only
```

or:

```text
bad for all three seeds.
```

W2 prioritizes cases with high **cross-seed shared difficulty**.

These are especially useful for:

```text
methodological failure analysis
regime annotation
attention follow-up.
```

---

# 18. W3 — Worst underprediction cases

Residual convention:

```text
e > 0 → underprediction.
```

For each seed:

```text
filter residual > 0
rank residual descending.
```

Lock:

```text
K_UNDER = 10 per seed.
```

Tie-break:

```text
residual descending
timestamp ascending
target_id ascending.
```

---

# 19. W4 — Worst overprediction cases

Residual:

```text
e < 0 → overprediction.
```

Overprediction magnitude:

\[
OE=|e|=-e.
\]

For each seed:

```text
filter residual < 0
rank absolute error descending.
```

Lock:

```text
K_OVER = 10 per seed.
```

---

# 20. Signed-case availability

If a seed has fewer than:

```text
10 positive
or
10 negative
```

residuals:

```text
return all available signed cases
and record actual count.
```

Do not fabricate additional cases.

---

# 21. W5 — Cross-seed worst-case overlap

For W1 sets:

```text
W42
W123
W2026.
```

Compute:

```text
pairwise intersection counts
triple intersection count
union size
Jaccard similarities.
```

---

# 22. Jaccard definition

For two seed worst-case sets:

\[
J(A,B)
=
\frac{|A\cap B|}{|A\cup B|}.
\]

Descriptive only.

---

# 23. Shared worst occurrence count

For each target:

```text
worst20_membership_count
∈ {0,1,2,3}.
```

This is another cross-seed failure-consistency indicator.

---

# 24. No independence claim from overlap

Overlap is descriptive.

Do not attach a statistical p-value unless a separate valid null model is predeclared, which Phase51 does not require.

---

# 25. W6 — Hardness vs seed disagreement

Phase48 already defines:

```text
seed_range_prediction
seed_std_prediction.
```

For each target join:

```text
SharedHardness_t
seed_range_prediction
seed_std_prediction
residual_sign_consensus.
```

This separates cases such as:

```text
high hardness + low seed disagreement
→ likely shared model/config difficulty

high hardness + high seed disagreement
→ difficult and seed-sensitive

low hardness + high disagreement
→ stochastic disagreement without uniformly large error.
```

These are descriptive patterns, not threshold-based classes.

---

# 26. No arbitrary quadrant thresholds

Do not invent:

```text
high hardness > X
high spread > Y
```

after Test inspection.

Use:

```text
continuous scatter
ranks
top-list overlap.
```

---

# 27. Hardness/spread association

Compute:

```text
Spearman(
  mean_abs_error_across_seeds,
  seed_range_prediction
)
```

and optionally:

```text
Spearman(
  mean_abs_error_across_seeds,
  seed_std_prediction
).
```

Diagnostic only.

---

# 28. Cross-seed residual sign consensus

Use frozen Phase49 categories:

```text
ALL_UNDER
ALL_OVER
MIXED
ALL_ZERO.
```

Do not recompute with a new epsilon.

---

# 29. Shared systematic signed cases

Create two secondary lists from W2 universe:

```text
SHARED_ALL_UNDER_TOP10
SHARED_ALL_OVER_TOP10
```

ranked by:

```text
mean_abs_error_across_seeds.
```

Lock:

```text
K_SHARED_SIGNED = 10.
```

These are especially useful for attention-conditioned analysis later.

---

# 30. Shared signed lists are secondary

Do not replace W2.

They answer:

```text
Which hard cases are consistently underpredicted by all seeds?
Which are consistently overpredicted by all seeds?
```

---

# 31. Case-selection freeze artifact

After W1–W4/W2 lists are computed:

```text
freeze all selected target IDs
write checksums
```

Before:

```text
manual narrative review
plot inspection
context extraction.
```

This prevents narrative-driven case substitution.

---

# 32. Case IDs cannot be swapped for visually nicer examples

Forbidden:

```text
rank 20 looks boring
→ replace with rank 21 because figure is clearer.
```

All selected cases remain.

If report only has space for fewer cases:

```text
use deterministic top ranks
```

such as top5, while full casebook retains all.

---

# 33. Report-display case subset

For main report figures, lock:

```text
DISPLAY_SHARED_TOP = 5
```

from W2 ranks 1–5.

Full artifact casebook still contains:

```text
K_SHARED=20.
```

---

# 34. Casebook universe

Create a deduplicated union:

```text
W1 all seeds
∪ W2
∪ W3
∪ W4
∪ shared signed top lists.
```

Every unique target receives:

```text
case_id
selection_reason(s)
rank(s).
```

---

# 35. Casebook selection reason is multi-label

Example:

```text
W2_SHARED_RANK_3
W1_SEED42_RANK_5
W3_SEED123_UNDER_RANK_2.
```

Do not force one exclusive reason.

---

# 36. Frozen Phase50 regime annotation

For every case attach:

```text
target_level_regime
extreme_high_regime
change_magnitude_regime
change_direction_regime
time_of_day_regime
day_type_regime.
```

No recomputation.

---

# 37. Regime enrichment questions

Phase51 should answer:

```text
How many W2 shared-hard cases are TL_HIGH?
How many are EXTREME_HIGH?
How many are CHANGE_RAPID?
How many are DIR_UP/DIR_DOWN?
How are they distributed by time-of-day/day-type?
```

This is case composition, not a new regime-performance metric.

---

# 38. Case composition vs population prevalence

For each regime label, compare:

```text
share among W2 top20
vs
share in full Test population.
```

This can identify overrepresentation descriptively.

---

# 39. Overrepresentation ratio

For regime `r`:

\[
CaseOverrep_r
=
\frac{
Share(r\mid W2)
}{
Share(r\mid Test)
}
\]

when denominator > 0.

This is descriptive only.

---

# 40. Small top-K caveat

With only 20 shared cases:

```text
case-composition ratios can be unstable.
```

No inferential significance claim.

---

# 41. Fixed local temporal context window

For every case use:

```text
±6 target steps
```

around target timestamp.

At 10-minute cadence:

```text
60 minutes before
target
60 minutes after.
```

Canonical:

```text
LOCAL_CONTEXT_RADIUS_STEPS = 6.
```

---

# 42. Why ±6 steps

It provides enough short-term context to inspect:

```text
pre-event trend
target jump/drop
immediate post-target continuation
```

without arbitrary case-specific zooming.

---

# 43. Local context is post-hoc descriptive

Rows after the target timestamp are:

```text
outcome context
```

and were not available to the model at forecast time.

Every case plot/report must distinguish:

```text
pre-forecast context
target
post-target outcome context.
```

---

# 44. No future context used to explain model input

Do not say:

```text
the model should have known y_{t+1...}
```

from post-target rows.

Future rows are only retrospective diagnosis.

---

# 45. Local context boundary handling

At beginning/end of Test:

```text
use all available rows within fixed radius.
```

Do not shift the window asymmetrically to force 13 rows.

Record:

```text
pre_steps_available
post_steps_available.
```

---

# 46. Local context continuity

Do not cross temporal gaps.

If a gap occurs within ±6:

```text
truncate at gap
mark CONTEXT_TRUNCATED_BY_GAP.
```

No interpolation.

---

# 47. Local temporal context table

For every selected case and nearby target:

```text
case_id
case_target_id
relative_step
relative_minutes
timestamp
is_case_target
is_pre_forecast_context
is_post_target_context
y_true_wh
y_pred_seed42
y_pred_seed123
y_pred_seed2026
residual_seed42
residual_seed123
residual_seed2026
continuity_valid
```

Prediction fields exist only where the timestamp itself belongs to frozen Test prediction population.

---

# 48. No new prediction for neighboring timestamps

If a local context timestamp does not have a Phase47 prediction:

```text
prediction field = N/A.
```

Do not rerun inference to fill it.

---

# 49. Model-visible input-window context

For each worst case, reconstruct the exact **input timestamps/features that were available to the final model** using:

```text
final feature contract
final lookback L*
WINDOWS-v1
WB0
```

No model forward pass.

---

# 50. Input-window context must be historical only

For case target `j`:

```text
input rows = j-L* ... j-1
target = j.
```

Hard:

```text
max(input_timestamp) < target_timestamp.
```

---

# 51. Model-visible context feature scope

Use only features in:

```text
final_feature_contract.json.
```

Do not add unavailable/future variables and call them model inputs.

---

# 52. Input context summary — continuous features

For every continuous final input feature:

```text
first
last
mean
std
min
max
last_minus_first.
```

Computed over the exact final input window.

---

# 53. Input context summary — cyclical features

For cyclical time features, primary useful fields:

```text
last
mean
min
max.
```

No causal interpretation.

---

# 54. Input context summary — binary features

For binary feature such as weekend:

```text
last
fraction_1_over_window.
```

---

# 55. Historical Appliances visibility

If final feature set includes historical `Appliances`:

```text
mark model_visible=true.
```

If final feature set excludes it:

```text
do not include it in model-visible feature summary.
```

---

# 56. Diagnostic target-history context

Regardless of final feature set, Phase51 may separately summarize past actual `Appliances` because it helps diagnose target dynamics.

Fields:

```text
past_target_first
past_target_last
past_target_mean
past_target_std
past_target_min
past_target_max
past_target_last_minus_first
target_minus_last_observation.
```

But include:

```text
historical_target_model_visible = true/false.
```

This prevents confusing diagnostic information with model input.

---

# 57. No target leakage in input-context reconstruction

The case target itself:

```text
must not enter model-visible context summary.
```

Hard audit.

---

# 58. Target one-step jump

For contiguous previous actual value:

\[
Jump_t=y_t-y_{t-1}.
\]

This should match frozen Phase50:

```text
delta_y_wh
change direction
change magnitude regime.
```

Audit consistency.

---

# 59. Local trend descriptor

Without fitting a new model, Phase51 may compute simple pre-target context:

```text
last 6-step actual target change
last 6-step actual target mean
last 6-step actual target std
```

using historical actual values only.

If historical target is not a model input:

```text
label DIAGNOSTIC_NOT_MODEL_VISIBLE.
```

---

# 60. No adaptive context features

Do not invent a different summary for each case.

All cases use the same fixed summary schema.

---

# 61. Model-visible context is descriptive, not attribution

If temperature was high in the input window:

Unsafe:

```text
high T2 caused the error.
```

Safe:

```text
the input window contained elevated T2 values relative to its own window summary.
```

Even that is local description, not causal explanation.

---

# 62. No automatic feature-cause ranking

Do not rank input features by:

```text
correlation with these 20 errors
```

inside Phase51.

That becomes post-hoc feature mining.

Attention analysis later examines model focus, still with caveats.

---

# 63. Baseline context at Transformer-selected worst cases

If Persistence bundle exists, join:

```text
persistence prediction/error
```

at selected Transformer case IDs.

If LSTM Test bundle exists, join frozen LSTM values.

---

# 64. Baseline context is selection-conditioned

Because cases are selected by Transformer error:

```text
baseline performance on these cases is not an unbiased global model comparison.
```

Use for:

```text
case-level context only.
```

Global model comparison remains Phase47/50.

---

# 65. Baseline case fields

For selected target:

```text
persistence_pred
persistence_residual
persistence_abs_error
lstm_pred_if_available
lstm_residual_if_available
lstm_abs_error_if_available.
```

---

# 66. Case-level baseline outcome classes

For each Transformer seed/case:

```text
TRANSFORMER_BETTER_THAN_PERSISTENCE
PERSISTENCE_BETTER_THAN_TRANSFORMER
TIE
```

using full-precision absolute error.

Similarly for LSTM if available.

Diagnostic only.

---

# 67. No baseline-based case removal

If Persistence also fails:

```text
retain case.
```

If Persistence succeeds:

```text
retain case.
```

Both are informative.

---

# 68. Case hardness vs naive baseline

Useful questions:

```text
Are shared-hard Transformer cases also hard for Persistence?
Are some Transformer worst cases easy for Persistence?
Are rapid-change cases especially different?
```

No model change.

---

# 69. Cross-seed case pattern taxonomy

Phase51 may assign deterministic descriptive tags from existing fields:

```text
ALL_SEEDS_UNDER
ALL_SEEDS_OVER
MIXED_SEED_SIGN
TOP20_ALL_THREE
TOP20_TWO_SEEDS
TOP20_ONE_SEED
EXTREME_HIGH
RAPID_CHANGE
DIR_UP
DIR_DOWN
PERSISTENCE_LOWER_AE_ALL_SEEDS
```

Each tag must be directly rule-derived.

No subjective labels like:

```text
mysterious
bad attention
sensor issue
```

without evidence.

---

# 70. Shared hard-case score table

For every Test target, not just top20, create:

```text
target_id
timestamp
mean_abs_error_across_seeds
median_abs_error_across_seeds
max_abs_error_across_seeds
min_abs_error_across_seeds
mean_squared_error_across_seeds_descriptive
seed_range_prediction
seed_std_prediction
residual_sign_consensus
worst20_membership_count.
```

This provides reproducible ranking source.

---

# 71. Mean squared error across seeds is descriptive per target

Allowed:

\[
Mean(e_s^2)
\]

for case characterization.

Do not aggregate it into a new official model score.

---

# 72. Per-seed worst ranking table

Fields:

```text
seed
rank
target_id
timestamp
y_true
y_pred
residual
absolute_error
squared_error
selection_score
```

plus frozen regimes.

---

# 73. Signed worst ranking table

Separate tables:

```text
worst_underprediction_cases.csv
worst_overprediction_cases.csv
```

with:

```text
seed
signed_rank
...
```

---

# 74. Shared worst ranking table

`shared_worst_cases.csv`:

```text
shared_rank
target_id
timestamp
y_true
pred42
pred123
pred2026
res42
res123
res2026
ae42
ae123
ae2026
mean_abs_error
median_abs_error
max_abs_error
seed_std_prediction
seed_range_prediction
sign_consensus
worst20_membership_count
Phase50 regime labels.
```

---

# 75. Worst-case overlap table

`worst_case_seed_overlap.csv`:

```text
set_a
set_b
intersection_count
union_count
jaccard
```

plus one row:

```text
TRIPLE_INTERSECTION.
```

---

# 76. Worst-case membership matrix

Optional but useful:

```text
target_id
in_seed42_top20
in_seed123_top20
in_seed2026_top20
membership_count
shared_hardness_rank.
```

---

# 77. Shared hard-case regime composition

For W2 top20:

```text
count/share by each Phase50 regime family.
```

Compare to Test prevalence.

---

# 78. Regime overrepresentation output

Fields:

```text
regime_family
regime_label
top20_shared_count
top20_shared_share
full_test_count
full_test_share
overrepresentation_ratio
small_count_warning
```

No significance p-value.

---

# 79. Shared hard-case sign composition

Report:

```text
ALL_UNDER count/share
ALL_OVER count/share
MIXED count/share
ALL_ZERO.
```

---

# 80. Per-seed signed severity summary

For each seed:

```text
largest underprediction Wh
largest overprediction magnitude Wh
median AE among top20
mean AE among top20
top20 SAE
top20 SSE
top20 share of global SAE
top20 share of global SSE.
```

---

# 81. Top-K contribution to global error

For each seed:

\[
TopK\_SSEShare
=
\frac{
\sum_{t\in W1_s}e_{s,t}^2
}{
SSE_{global,s}
}.
\]

Similarly SAE share.

This shows how concentrated Test error is in a small number of cases.

---

# 82. Top-K sample prevalence

\[
TopKSampleShare
=
\frac{K}{N_{test}}.
\]

Compare:

```text
sample share
vs
SAE/SSE share.
```

---

# 83. No top-K deletion experiment

Do not compute:

```text
RMSE after removing worst20
```

as a performance claim.

That would create a sanitized Test metric.

Top-K contribution is sufficient.

---

# 84. Error concentration curve

Optional diagnostic:

Sort each seed’s absolute errors descending and compute cumulative:

```text
sample fraction
SAE fraction
SSE fraction.
```

This is analogous to concentration analysis.

No samples removed.

---

# 85. Fixed concentration checkpoints

Report cumulative contribution at:

```text
top 1%
top 5%
top 10%
```

by sample count, if Test N is large enough.

Rounding rule:

```text
ceil(p*N)
```

with minimum 1.

This is descriptive and fixed.

---

# 86. Error concentration interpretation

Safe:

> A small fraction of Test cases accounted for a disproportionate share of squared error.

No claim that those cases should be discarded.

---

# 87. Local case figure design

For each shared top20 case generate a deterministic local plot.

Plot window:

```text
-6 ... +6 target steps
```

where available/contiguous.

Show:

```text
actual Appliances
seed42 prediction
seed123 prediction
seed2026 prediction
vertical line at case target
```

Optional:

```text
Persistence
```

with lighter/contextual role if readability allows.

---

# 88. Future-context visual separation

Shade or annotate:

```text
pre-target side
target
post-target side.
```

Caption:

```text
post-target values are retrospective outcome context and were not available to the forecast.
```

---

# 89. Local plot no rescaling trick

All case plots may use local y-axis to preserve readability, but caption should show:

```text
local axis
```

and key numeric target/prediction values.

For direct case-to-case visual magnitude comparison, also provide summary tables.

---

# 90. Deterministic report display figures

Main report includes:

```text
shared ranks 1–5
```

not manually selected cases.

Full casebook includes ranks 1–20.

---

# 91. Per-seed worst-case figures

Optional:

```text
top5 per seed
```

if not already covered by shared top5.

Use deterministic rank.

---

# 92. Hardness-vs-seed-spread figure

Scatter:

```text
x = mean_abs_error_across_seeds
y = seed_range_prediction
```

Annotate only:

```text
W2 top20 targets
```

or top5 labels for readability.

No arbitrary quadrant lines.

---

# 93. Worst-case overlap figure

Recommended:

```text
3×3 overlap/Jaccard heatmap
```

or membership bar chart.

No need for complex Venn if labels become unclear.

---

# 94. Worst-case regime composition figure

For W2 top20:

```text
shared-hard share
vs
full-Test share
```

by key regime labels.

No causal claim.

---

# 95. Signed worst-error figure

Separate:

```text
largest underprediction residuals
largest overprediction magnitudes
```

with common Wh magnitude scale where possible.

---

# 96. Error concentration figure

For each seed:

```text
x = cumulative fraction of Test cases sorted by AE descending
y = cumulative SSE share.
```

This visualizes error concentration.

---

# 97. Casebook narrative fields

Each selected case should contain machine-generated factual fields:

```text
case_id
selection reasons/ranks
target ID/timestamp
true Appliances
3 Transformer predictions
3 residuals/AEs
sign consensus
seed spread
Phase50 regimes
previous actual Appliances
delta y
local context availability
Persistence context
LSTM context if available
model-visible input feature summary reference
diagnostic target-history summary reference.
```

---

# 98. Manual case note is optional and constrained

A human-readable note may be added, but it must:

```text
describe observable facts only
avoid unsupported causal claims
avoid claiming attention explanation before Phase52
avoid proposing post-Test correction as current result.
```

---

# 99. Safe case note example

> This target belongs to the Train-defined EXTREME_HIGH and CHANGE_RAPID regimes. All three Transformer seeds underpredicted the target, and their predictions were tightly clustered relative to the size of the error. The previous actual Appliances value was substantially lower than the target value.

This is evidence-based.

---

# 100. Unsafe case note example

> The Transformer ignored temperature and therefore missed the appliance spike.

Not supported before attention/feature attribution, and even attention would not prove causality.

---

# 101. Input-window summary provenance

For every case store:

```text
input_start_timestamp
input_end_timestamp
target_timestamp
lookback_steps
feature_fingerprint
final_model_lock_sha256
```

This makes later attention alignment easier.

---

# 102. Input-window feature summary long format

`worst_case_input_feature_summary.csv`:

```text
case_id
target_id
feature_name
feature_group
model_visible=true
summary_type
summary_value
input_start
input_end
lookback_steps
```

One row per:

```text
case × feature × statistic.
```

---

# 103. Diagnostic target-history summary

`worst_case_target_history_summary.csv`:

```text
case_id
target_id
historical_target_model_visible
window_steps
first_y
last_y
mean_y
std_y
min_y
max_y
last_minus_first
target_y
target_minus_last
```

Use exact historical rows only.

---

# 104. Short context summary

Also create a compact:

```text
last_6_steps target context
```

with:

```text
last_60m_mean
last_60m_std
last_60m_min
last_60m_max
last_60m_change.
```

Again:

```text
model-visible flag
```

for historical target.

---

# 105. Context cannot use post-target statistics as input summary

Hard:

```text
model-visible input summary ends at t-1.
```

Post-target local context remains separate.

---

# 106. Attention handoff purpose

Phase52 will load the same final checkpoints and extract attention weights.

Phase51 must provide:

```text
exact target IDs
seed identities
input-window indices
regime labels
error ranks
case-selection provenance
```

so attention analysis can align attention tensors with scientifically defined failure cases.

---

# 107. Attention case set

Canonical Phase51 handoff includes:

```text
W2 shared top20 IDs
per-seed W1 top20 IDs
shared all-under top10 IDs
shared all-over top10 IDs.
```

Phase52 may extract attention for these IDs according to its own execution plan.

Phase51 does not extract attention.

---

# 108. No attention-based case replacement

Do not say:

```text
this worst case has uninteresting attention
→ replace with another.
```

Case selection is frozen independently of attention.

---

# 109. Phase52 target-ID uniqueness

Handoff should deduplicate IDs but preserve:

```text
all selection roles/ranks.
```

---

# 110. Attention input alignment fields

For each handed-off case:

```text
target_id
target_timestamp
input_start_timestamp
input_end_timestamp
lookback_steps
selection_roles
regime labels
seed-specific error fields.
```

---

# 111. Optional control-context recommendation

Phase51 may state:

```text
Phase52/56 should later compare worst-error cases with non-worst controls using a separately predeclared control-selection rule.
```

But Phase51 itself does not select controls unless requested by the later phase protocol.

---

# 112. No low-error cherry-pick in Phase51

No.

---

# 113. Output directory

```text
artifacts/
└── worst_error_analysis/
    ├── worst_error_analysis_manifest.json
    ├── worst_error_analysis_contract.json
    ├── phase51_preflight_audit.csv
    ├── worst_error_source_verification.csv
    ├── worst_error_selection_contract.json
    ├── worst_error_selection_contract_fingerprint.json
    ├── worst_error_alignment_audit.csv
    ├── worst_error_per_seed_top20.csv
    ├── shared_case_hardness_all_test.csv
    ├── shared_worst_cases_top20.csv
    ├── worst_underprediction_cases.csv
    ├── worst_overprediction_cases.csv
    ├── shared_all_under_top10.csv
    ├── shared_all_over_top10.csv
    ├── worst_case_seed_overlap.csv
    ├── worst_case_membership_matrix.csv
    ├── worst_case_error_concentration.csv
    ├── worst_case_regime_composition.csv
    ├── worst_case_regime_overrepresentation.csv
    ├── worst_case_seed_disagreement.csv
    ├── worst_case_baseline_context.csv
    ├── worst_case_casebook_index.csv
    ├── worst_case_casebook.md
    ├── worst_case_local_context.csv
    ├── worst_case_input_window_manifest.csv
    ├── worst_case_input_feature_summary.csv
    ├── worst_case_target_history_summary.csv
    ├── worst_case_context_integrity_audit.csv
    ├── worst_case_attention_handoff_cases.csv
    ├── worst_error_analysis_findings.csv
    ├── worst_error_analysis_tests.csv
    ├── worst_error_analysis_discrepancies.json
    ├── phase52_attention_extraction_handoff.json
    ├── worst_error_analysis_summary.json
    ├── worst_error_analysis_report.md
    ├── figures/
    │   ├── WORST_51_01_top20_abs_error_seed42.png
    │   ├── WORST_51_02_top20_abs_error_seed123.png
    │   ├── WORST_51_03_top20_abs_error_seed2026.png
    │   ├── WORST_51_04_shared_top20_hardness.png
    │   ├── WORST_51_05_worst_underprediction.png
    │   ├── WORST_51_06_worst_overprediction.png
    │   ├── WORST_51_07_seed_top20_overlap.png
    │   ├── WORST_51_08_hardness_vs_seed_disagreement.png
    │   ├── WORST_51_09_shared_worst_regime_composition.png
    │   ├── WORST_51_10_sample_share_vs_worst_case_share.png
    │   ├── WORST_51_11_error_concentration_sse.png
    │   ├── WORST_51_12_baseline_context_shared_worst.png
    │   └── casebook/
    │       ├── SHARED_R01_<target_id>.png
    │       ├── SHARED_R02_<target_id>.png
    │       └── ...
    ├── README_WORST_ERROR_ANALYSIS.md
    └── phase_51_signoff.json
```

Actual target IDs are runtime values and must not be fabricated in filenames before execution.

---

# 114. Required outputs

```text
O51.1  Analysis manifest
O51.2  Analysis contract
O51.3  Preflight audit
O51.4  Source verification
O51.5  Frozen selection contract
O51.6  Selection contract fingerprint
O51.7  Alignment audit
O51.8  Per-seed top20 absolute-error table
O51.9  All-Test shared hardness table
O51.10 Shared top20 table
O51.11 Worst underprediction table
O51.12 Worst overprediction table
O51.13 Shared all-under top10
O51.14 Shared all-over top10
O51.15 Seed-overlap table
O51.16 Membership matrix
O51.17 Error-concentration table
O51.18 Regime composition
O51.19 Regime overrepresentation
O51.20 Hardness vs seed-disagreement table
O51.21 Baseline context
O51.22 Casebook index
O51.23 Casebook Markdown
O51.24 Local temporal context
O51.25 Input-window manifest
O51.26 Model-visible feature summary
O51.27 Diagnostic target-history summary
O51.28 Context-integrity audit
O51.29 Figures
O51.30 Attention handoff case table
O51.31 Findings
O51.32 Phase52 handoff
O51.33 Tests
O51.34 Discrepancies
O51.35 Summary JSON
O51.36 Human-readable report
O51.37 README
O51.38 Sign-off
```

---

# 115. Analysis manifest

`worst_error_analysis_manifest.json`:

```text
phase=51
version=WORST_ERROR_ANALYSIS-v1
source_phase50_version
source_phase49_version
source_phase47_version
final_lock_sha256
test_population_sha256
regime_assignment_sha256
seed_list=[42,123,2026]
K_ABS=20
K_SHARED=20
K_UNDER=10
K_OVER=10
K_SHARED_SIGNED=10
LOCAL_CONTEXT_RADIUS_STEPS=6
DISPLAY_SHARED_TOP=5
new_inference=false
new_training=false
best_seed_selection=false
regime_threshold_changes=false
prediction_modification=false
attention_extraction=false
status
created_at
```

---

# 116. Analysis contract

`worst_error_analysis_contract.json` must state:

```text
Residual = y_true-y_pred.

W1:
top20 |residual| per seed.

W2:
top20 mean absolute error across 3 seeds.

W3:
top10 positive residual per seed.

W4:
top10 negative residual magnitude per seed.

Shared signed:
top10 ALL_UNDER
top10 ALL_OVER
ranked by mean absolute error across seeds.

Tie-break:
timestamp ascending
then target_id ascending.

Local case context:
±6 target steps
gap-safe
post-target values retrospective only.

Regime labels:
frozen Phase50 only.

No squared-error duplicate ranking.
No best seed.
No new inference.
No prediction correction.
No attention extraction.
```

---

# 117. Selection contract fingerprint

Create:

```text
worst_error_selection_contract_fingerprint.json
```

before ranking/narrative.

Fields:

```text
contract_sha256
source_residual_sha256
source_regime_assignment_sha256
test_population_sha256
created_before_case_narrative=true
status
```

---

# 118. Preflight audit

`phase51_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Checks:

```text
Phase50 approved
Phase51 handoff exists
Phase49 residual files exist
Phase48 seed spread exists
Phase50 regime assignment exists
all source checksums match
same Test population
all 3 seeds present
residual convention exact
no source row missing
final feature contract available for input context
no inference required.
```

---

# 119. Source verification

`worst_error_source_verification.csv`:

```text
source_id
path
expected_sha256
observed_sha256
row_count
population_sha256
frozen
status
```

---

# 120. Alignment audit

`worst_error_alignment_audit.csv`:

```text
check
seed42
seed123
seed2026
regime_assignment
seed_spread
expected
status
```

Required:

```text
same target IDs
same timestamps
same y_true
unique IDs
complete joins.
```

---

# 121. Per-seed top20 schema

`worst_error_per_seed_top20.csv`:

```text
seed
rank
target_id
target_timestamp
y_true_wh
y_pred_wh
residual_wh
absolute_error_wh
squared_error_wh2
target_level_regime
extreme_high_regime
change_magnitude_regime
change_direction_regime
time_of_day_regime
day_type_regime
sign_consensus
seed_range_prediction
status
```

---

# 122. Shared hardness all-Test schema

`shared_case_hardness_all_test.csv`:

```text
target_id
timestamp
y_true_wh
ae_seed42
ae_seed123
ae_seed2026
mean_abs_error_across_seeds
median_abs_error_across_seeds
min_abs_error_across_seeds
max_abs_error_across_seeds
mean_squared_error_across_seeds_descriptive
seed_std_prediction
seed_range_prediction
sign_consensus
seed42_top20_member
seed123_top20_member
seed2026_top20_member
worst20_membership_count
status
```

---

# 123. Shared top20 schema

`shared_worst_cases_top20.csv` includes:

```text
shared_rank
all fields from shared hardness table
all 3 predictions
all 3 residuals
Phase50 regime labels
Persistence/LSTM context where available.
```

---

# 124. Worst underprediction schema

`worst_underprediction_cases.csv`:

```text
seed
rank
target_id
timestamp
y_true
y_pred
residual_positive_wh
absolute_error
regime labels
sign_consensus
status
```

---

# 125. Worst overprediction schema

`worst_overprediction_cases.csv`:

```text
seed
rank
target_id
timestamp
y_true
y_pred
residual_negative_wh
overprediction_magnitude_wh
regime labels
sign_consensus
status
```

---

# 126. Shared signed top tables

`shared_all_under_top10.csv` and `shared_all_over_top10.csv`:

```text
rank
target_id
timestamp
mean_abs_error
seed-specific errors
regime labels
seed spread
status
```

---

# 127. Seed-overlap schema

`worst_case_seed_overlap.csv`:

```text
comparison
set_a
set_b
intersection_count
union_count
jaccard
status
```

Include:

```text
42_vs_123
42_vs_2026
123_vs_2026
TRIPLE_INTERSECTION.
```

Triple row has:

```text
jaccard=N/A
```

unless a clearly defined 3-set generalized measure is added; not required.

---

# 128. Membership matrix schema

`worst_case_membership_matrix.csv`:

```text
target_id
timestamp
in_seed42_top20
rank_seed42_if_any
in_seed123_top20
rank_seed123_if_any
in_seed2026_top20
rank_seed2026_if_any
worst20_membership_count
shared_hardness_rank
status
```

---

# 129. Error concentration schema

`worst_case_error_concentration.csv`:

```text
seed
selection
K_or_fraction
sample_count
sample_share
SAE
SAE_share
SSE
SSE_share
status
```

Required rows:

```text
TOP20
TOP1_PERCENT
TOP5_PERCENT
TOP10_PERCENT.
```

For percentage rows:

```text
K = ceil(p*N), minimum 1.
```

---

# 130. Regime composition schema

`worst_case_regime_composition.csv`:

```text
case_set=W2_SHARED_TOP20
regime_family
regime_label
case_count
case_share
status
```

---

# 131. Regime overrepresentation schema

`worst_case_regime_overrepresentation.csv`:

```text
regime_family
regime_label
shared_top20_count
shared_top20_share
full_test_count
full_test_share
overrepresentation_ratio
small_count_warning
status
```

---

# 132. Seed disagreement schema

`worst_case_seed_disagreement.csv`:

```text
target_id
timestamp
mean_abs_error_across_seeds
shared_hardness_rank
seed_std_prediction
seed_range_prediction
sign_consensus
worst20_membership_count
status
```

Include all Test targets for scatter reproducibility.

---

# 133. Baseline context schema

`worst_case_baseline_context.csv`:

```text
case_id
target_id
selection_reason
seed
transformer_abs_error
persistence_prediction
persistence_abs_error
transformer_vs_persistence_outcome
lstm_prediction_if_available
lstm_abs_error_if_available
transformer_vs_lstm_outcome
selection_conditioned_comparison=true
status
```

---

# 134. Casebook index schema

`worst_case_casebook_index.csv`:

```text
case_id
target_id
timestamp
selection_reasons
shared_rank_if_any
seed42_rank_if_any
seed123_rank_if_any
seed2026_rank_if_any
underprediction_ranks
overprediction_ranks
local_context_file_ref
figure_ref
input_context_ref
status
```

---

# 135. Casebook Markdown structure

`worst_case_casebook.md`:

```text
1. Selection protocol
2. Interpretation caveats
3. Shared hard cases rank 1–20
4. Per-seed-only worst cases not already covered
5. Shared all-under cases
6. Shared all-over cases
7. Cross-seed overlap summary
8. Error concentration summary
9. Regime composition summary
10. Baseline context summary
11. Handoff cases for attention analysis
```

Each case section uses factual template.

---

# 136. Case factual template

For each case:

```text
Case ID
Target ID
Timestamp

Selection:
W2 shared rank
W1 seed ranks
signed ranks

Truth/predictions:
Actual Appliances Wh
Seed42 prediction/error
Seed123 prediction/error
Seed2026 prediction/error

Cross-seed:
mean absolute error
prediction spread
sign consensus

Regimes:
R1–R6 labels

Dynamics:
previous actual
Δy
rapid/normal
direction

Baseline context:
Persistence
LSTM if available

Context:
input start/end
local ±60min plot
model-visible summary reference

Interpretation:
facts only
no causal/attention claim.
```

---

# 137. Local context schema

`worst_case_local_context.csv`:

```text
case_id
case_target_id
relative_step
relative_minutes
timestamp
same_continuity_segment
context_role
y_true_wh
y_pred_seed42_if_available
y_pred_seed123_if_available
y_pred_seed2026_if_available
residual_seed42_if_available
residual_seed123_if_available
residual_seed2026_if_available
status
```

Context role:

```text
PRE_TARGET
CASE_TARGET
POST_TARGET.
```

---

# 138. Input-window manifest schema

`worst_case_input_window_manifest.csv`:

```text
case_id
target_id
target_timestamp
lookback_steps
input_start_timestamp
input_end_timestamp
input_row_count
feature_count
feature_fingerprint
max_input_before_target
continuity_valid
target_row_excluded
status
```

Expected:

```text
input_row_count=L*
max_input_before_target=true
target_row_excluded=true.
```

---

# 139. Input feature summary schema

`worst_case_input_feature_summary.csv`:

```text
case_id
target_id
feature_name
feature_group
summary_stat
summary_value
model_visible=true
input_start_timestamp
input_end_timestamp
status
```

---

# 140. Target history summary schema

`worst_case_target_history_summary.csv`:

```text
case_id
target_id
historical_target_model_visible
history_window_steps
first_y
last_y
mean_y
std_y
min_y
max_y
last_minus_first
last_60m_mean
last_60m_std
last_60m_min
last_60m_max
last_60m_change
target_y
target_minus_last
status
```

---

# 141. Context integrity audit

`worst_case_context_integrity_audit.csv`:

```text
case_id
target_id
input_rows_exact
input_continuity_valid
all_input_timestamps_before_target
target_not_in_input
feature_order_match
local_context_gap_safe
post_target_context_labeled
no_new_predictions_generated
status
```

---

# 142. Attention handoff case table

`worst_case_attention_handoff_cases.csv`:

```text
target_id
target_timestamp
input_start_timestamp
input_end_timestamp
selection_roles
shared_rank
seed42_rank
seed123_rank
seed2026_rank
shared_all_under_rank
shared_all_over_rank
target_level_regime
extreme_high_regime
change_magnitude_regime
change_direction_regime
time_of_day_regime
day_type_regime
mean_abs_error_across_seeds
seed_range_prediction
sign_consensus
status
```

Deduplicated one row per target.

---

# 143. Phase52 handoff

`phase52_attention_extraction_handoff.json`:

```text
source_phase51_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
final_checkpoint_refs
final_scaler_refs
feature_fingerprint
lookback_steps
num_heads
num_layers
attention_shape_contract=[B,H,L,L]
worst_case_selection_contract_sha256
worst_case_attention_handoff_table
shared_top20_ids
per_seed_top20_ids
shared_all_under_top10_ids
shared_all_over_top10_ids
regime_assignment_sha256
no_attention_extracted_in_phase51=true
case_selection_independent_of_attention=true
ready_for_phase52=true
```

Runtime IDs only.

---

# 144. Findings codes

Possible:

```text
WORST_ERRORS_CONCENTRATED
WORST_ERRORS_DIFFUSE
HIGH_TOP20_SSE_CONTRIBUTION
LOW_TOP20_SSE_CONTRIBUTION
WORST_CASES_HIGHLY_OVERLAP_ACROSS_SEEDS
WORST_CASES_SEED_SPECIFIC
SHARED_HARD_CASES_LOW_SEED_SPREAD
SHARED_HARD_CASES_HIGH_SEED_SPREAD
SHARED_HARD_CASES_MOSTLY_ALL_UNDER
SHARED_HARD_CASES_MOSTLY_ALL_OVER
SHARED_HARD_CASES_MIXED_SIGN
EXTREME_HIGH_OVERREPRESENTED_IN_SHARED_WORST
RAPID_CHANGE_OVERREPRESENTED_IN_SHARED_WORST
DIR_UP_OVERREPRESENTED_IN_SHARED_WORST
DIR_DOWN_OVERREPRESENTED_IN_SHARED_WORST
TIME_BLOCK_OVERREPRESENTED_IN_SHARED_WORST
WEEKEND_OVERREPRESENTED_IN_SHARED_WORST
PERSISTENCE_ALSO_FAILS_ON_SHARED_WORST
PERSISTENCE_HANDLES_SOME_TRANSFORMER_WORST_CASES
LSTM_CONTEXT_AVAILABLE
LSTM_HANDLES_SOME_TRANSFORMER_WORST_CASES
ERROR_HARDNESS_ASSOCIATED_WITH_SEED_SPREAD
ERROR_HARDNESS_NOT_STRONGLY_ASSOCIATED_WITH_SEED_SPREAD
INPUT_CONTEXT_VERIFIED
CASE_SELECTION_FROZEN_BEFORE_NARRATIVE
NO_NEW_INFERENCE
NO_BEST_SEED
NO_POST_TEST_CORRECTION
ATTENTION_HANDOFF_READY
```

Avoid arbitrary “high/low” threshold wording unless exact quantitative context is included.

---

# 145. Safe findings language

Safe:

> The three per-seed top-20 error sets showed substantial overlap, indicating that many of the largest errors occurred at the same Test timestamps across independent final seeds.

Safe:

> Several shared hard cases belonged to the Train-defined EXTREME_HIGH and CHANGE_RAPID regimes, and all three seeds underpredicted those targets.

Safe:

> Persistence produced smaller absolute error than the Transformer on some Transformer-selected worst cases; this is a selection-conditioned case comparison and does not replace the global Test comparison.

---

# 146. Unsafe findings language

Do not say:

```text
the model failed because attention ignored feature X
```

before Phase52–57.

Do not say:

```text
rapid change caused the error
```

from regime co-occurrence alone.

Do not say:

```text
these worst cases should be removed as outliers.
```

---

# 147. No outlier deletion recommendation

Worst Test errors are part of model performance.

They cannot be removed unless independently proven to be data corruption under a pre-existing data-quality protocol.

Phase51 does not relabel difficult valid observations as outliers.

---

# 148. Data-quality anomaly handling

If a worst case appears suspicious:

```text
record DATA_QUALITY_REVIEW_REQUESTED
```

but do not alter Test metrics/data in this pipeline.

A confirmed source-data bug would require a separate contamination/amendment process.

---

# 149. No sensor-failure claim from large error alone

No.

---

# 150. Error concentration is not permission to trim

Even if top 1% explains a large SSE share:

```text
global Phase47 metrics remain unchanged.
```

---

# 151. Worst-case set size caveat

`K=20` is a casebook size for deep inspection.

It is not a natural statistical cutoff.

Do not generalize proportions from 20 cases without caution.

---

# 152. Signed top10 caveat

W3/W4 isolate severity direction.

They are not balanced samples and should not be used for inferential comparisons.

---

# 153. Shared-hardness caveat

Mean absolute error across three seeds is a descriptive target-level hardness score.

With only three seeds, it does not characterize the full randomness distribution.

---

# 154. Baseline selection-conditioned caveat

Baseline performance at Transformer-selected worst cases can appear unusually favorable by construction.

Do not compare its average on those cases to Transformer global metrics.

---

# 155. Local context caveat

Post-target values are retrospective.

They help describe event evolution but were unavailable to the forecast.

---

# 156. Model-visible feature summary caveat

Window summaries do not indicate feature importance.

They only describe the values presented to the model.

---

# 157. Attention handoff caveat

Attention weights later show internal attention allocation, not causal feature importance or guaranteed explanation.

Phase51 should not pre-interpret them.

---

# 158. Figures

Required/recommended:

```text
WORST_51_01_top20_abs_error_seed42.png
WORST_51_02_top20_abs_error_seed123.png
WORST_51_03_top20_abs_error_seed2026.png
WORST_51_04_shared_top20_hardness.png
WORST_51_05_worst_underprediction.png
WORST_51_06_worst_overprediction.png
WORST_51_07_seed_top20_overlap.png
WORST_51_08_hardness_vs_seed_disagreement.png
WORST_51_09_shared_worst_regime_composition.png
WORST_51_10_sample_share_vs_worst_case_share.png
WORST_51_11_error_concentration_sse.png
WORST_51_12_baseline_context_shared_worst.png
casebook/SHARED_Rxx_<runtime_target_id>.png
```

---

# 159. Top20 per-seed figures

Plot:

```text
rank
absolute error Wh.
```

Include target timestamp as compact labels only if readable.

Do not reorder by regime.

Rank order is fixed.

---

# 160. Shared hardness figure

For top20 shared cases show:

```text
mean AE across seeds
+
individual seed AE points/bars.
```

This reveals shared vs one-seed-dominated cases.

---

# 161. Signed figures

Underprediction:

```text
positive residual Wh
```

Overprediction:

```text
overprediction magnitude Wh.
```

Use clear sign labels.

---

# 162. Overlap figure

Use:

```text
pairwise Jaccard matrix
+
triple intersection annotation.
```

---

# 163. Hardness-vs-spread figure

Axes:

```text
x = mean_abs_error_across_seeds
y = seed_range_prediction.
```

Annotate only shared top20 or top5 labels to prevent clutter.

---

# 164. Regime composition figure

Compare:

```text
W2 top20 share
vs
full Test share.
```

Use frozen regime families.

No p-values.

---

# 165. Error concentration figure

Per seed cumulative SSE share vs cumulative sample fraction sorted by descending absolute error.

Show:

```text
1%
5%
10%
```

reference markers if applicable.

---

# 166. Baseline context figure

For shared top20 cases:

```text
mean Transformer AE across seeds
Persistence AE
LSTM AE if available.
```

Caption must say:

```text
cases selected using Transformer error; comparison is selection-conditioned.
```

---

# 167. Casebook local figures

Every W2 rank 1–20 gets one local temporal plot.

Title:

```text
Shared worst-error rank X
Target timestamp
Regime tags
```

No causal interpretation in title.

---

# 168. Main report case figures

Embed only:

```text
W2 ranks 1–5.
```

This is deterministic and report-space efficient.

---

# 169. Human-readable report structure

`worst_error_analysis_report.md`:

```text
1. Objective
2. Frozen source verification
3. Worst-case selection contract
4. Why absolute-error ranking is used
5. Why squared-error ranking is not duplicated
6. Per-seed top20 errors
7. Shared hard-case ranking
8. Worst underprediction cases
9. Worst overprediction cases
10. Cross-seed top-error overlap
11. Shared hardness vs seed disagreement
12. Error concentration
13. Regime composition of shared hard cases
14. Baseline context on selected cases
15. Local temporal context
16. Model-visible input-window context
17. Detailed shared top5 case narratives
18. Full casebook summary
19. Handoff case set for attention extraction
20. Methodological limitations
21. No-retuning statement
22. Definition of Done
```

---

# 170. README requirements

`README_WORST_ERROR_ANALYSIS.md` explains:

```text
why K=20/K=10 are fixed
why absolute error is primary
why squared-error ranking is redundant
what shared hardness means
why it is not an ensemble
how overlap is computed
how sign-specific lists work
why regimes come only from Phase50
why local context is ±6 steps
why post-target values are retrospective only
why input summaries are not feature importance
why baseline comparison is selection-conditioned
why no worst case is removed
how Phase52 consumes case IDs.
```

---

# 171. Summary artifact

`worst_error_analysis_summary.json`:

```text
version
source_phase50_version
source_phase49_version
final_lock_sha256
test_population_sha256
regime_assignment_sha256
seed_list
selection_contract_sha256
K_ABS
K_SHARED
K_UNDER
K_OVER
shared_top20_ids
per_seed_top20_ids
shared_all_under_top10_ids
shared_all_over_top10_ids
overlap_summary
error_concentration_summary
regime_composition_summary
regime_overrepresentation_summary
seed_disagreement_summary
baseline_context_summary
input_context_status
casebook_status
findings
new_inference=false
best_seed_selected=false
prediction_modified=false
regime_threshold_modified=false
attention_extracted=false
phase52_ready
overall_status
```

---

# 172. Phase51 sign-off

`phase_51_signoff.json` minimum:

```text
phase=51
phase_name=Worst-error analysis
version=WORST_ERROR_ANALYSIS-v1
source_phase50_version
source_phase49_version
final_lock_sha256
test_population_sha256
regime_assignment_sha256
seed_list=[42,123,2026]
selection_contract_sha256
K_ABS=20
K_SHARED=20
K_UNDER=10
K_OVER=10
K_SHARED_SIGNED=10
local_context_radius_steps=6
per_seed_top20_complete
shared_top20_complete
signed_case_analysis_complete
overlap_analysis_complete
error_concentration_complete
regime_annotation_complete
baseline_context_status
casebook_complete
input_context_complete
attention_handoff_case_table_complete
new_inference=false
model_training=false
best_seed_selected=false
prediction_correction_applied=false
regime_threshold_modified=false
attention_extracted=false
phase52_ready
warnings
overall_status
created_at
```

---

# 173. Discrepancy taxonomy

`worst_error_analysis_discrepancies.json`:

```text
PHASE50_NOT_APPROVED
PHASE49_SOURCE_MISSING
PHASE48_SEED_SPREAD_MISSING
SOURCE_CHECKSUM_MISMATCH
TEST_POPULATION_MISMATCH
REGIME_ASSIGNMENT_MISMATCH
RESIDUAL_CONVENTION_DRIFT
SEED_MISSING
SELECTION_CONTRACT_NOT_FROZEN
K_ABS_DRIFT
K_SHARED_DRIFT
K_UNDER_DRIFT
K_OVER_DRIFT
TIE_BREAK_DRIFT
CASE_SWAPPED_AFTER_INSPECTION
SQUARED_ERROR_RANKING_MISREPRESENTED_AS_INDEPENDENT
SHARED_HARDNESS_MISLABELED_AS_ENSEMBLE
BEST_SEED_SELECTION_ATTEMPT
REGIME_THRESHOLD_RECOMPUTED
REGIME_LABEL_CHANGED
LOCAL_CONTEXT_RADIUS_DRIFT
LOCAL_CONTEXT_CROSSES_GAP
POST_TARGET_CONTEXT_MISLABELED_AS_MODEL_INPUT
NEW_NEIGHBOR_PREDICTION_GENERATED
INPUT_WINDOW_INCLUDES_TARGET
INPUT_WINDOW_INCLUDES_FUTURE
FEATURE_ORDER_MISMATCH
NON_MODEL_FEATURE_LABELED_MODEL_VISIBLE
TARGET_HISTORY_VISIBILITY_MISLABELED
BASELINE_POPULATION_MISMATCH
BASELINE_CONTEXT_USED_AS_GLOBAL_COMPARISON
LSTM_WRONG_SCALER
LSTM_RETRAINED
WORST_CASE_REMOVAL_ATTEMPT
RMSE_AFTER_TRIMMING_COMPUTED_AS_FINAL
POST_TEST_BIAS_CORRECTION_ATTEMPT
PREDICTION_CLIPPING_ATTEMPT
PREDICTION_SHIFT_ATTEMPT
NEW_TEST_INFERENCE_ATTEMPT
MODEL_TRAINING_ATTEMPT
UNSUPPORTED_CAUSAL_CLAIM
ATTENTION_INTERPRETATION_BEFORE_EXTRACTION
ATTENTION_BASED_CASE_REPLACEMENT
OTHER
```

---

# 174. Status model

## PASS

```text
all source artifacts verified
selection protocol frozen
per-seed top20 complete
shared top20 complete
signed worst cases complete
cross-seed overlap complete
error concentration complete
regime annotations complete
local contexts complete
model-visible input summaries complete
casebook complete
attention case handoff frozen
no inference/retraining/correction
Phase52 ready.
```

## PASS_WITH_WARNING

Possible:

```text
fewer than 10 signed cases available
local context truncated by Test boundary/gap
LSTM baseline context unavailable
historical target not visible to final model
high seed-specific disagreement
casebook contains small-regime cases.
```

These do not invalidate the phase.

## FAIL

Examples:

```text
source checksum mismatch
case-selection rule changes after inspection
future rows enter model-visible context
regime labels changed
new Test inference
case trimming
best-seed selection
attention-based case substitution.
```

---

# 175. Execution sequence

```text
1. Verify Phase50/49/48/47 source artifacts.
2. Verify residual/regime/seed-spread alignment.
3. Write and freeze worst-error selection contract.
4. Compute W1 top20 per seed.
5. Compute shared hardness for every Test target.
6. Compute W2 shared top20.
7. Compute W3 top10 underprediction per seed.
8. Compute W4 top10 overprediction per seed.
9. Compute shared all-under/all-over top10.
10. Freeze all case-selection IDs/checksums.
11. Compute top20 overlap/Jaccard/membership.
12. Compute hardness-vs-seed-disagreement diagnostics.
13. Compute per-seed error concentration curves.
14. Annotate selected cases with frozen Phase50 regimes.
15. Compute shared-top20 regime composition/overrepresentation.
16. Join Persistence/LSTM case context if available.
17. Build deduplicated casebook index.
18. Build fixed ±6-step local context.
19. Reconstruct exact model-visible input windows.
20. Verify input windows exclude target/future.
21. Build feature summaries for each case.
22. Build diagnostic target-history summaries.
23. Generate deterministic case figures.
24. Write factual casebook.
25. Build attention handoff case table.
26. Write Phase52 handoff.
27. Run tests/discrepancy audit.
28. Write findings/summary/report/README.
29. Sign off.
```

---

# 176. Preflight acceptance checklist

```text
[ ] Phase50 PASS/PASS_WITH_WARNING.
[ ] Phase51 regime handoff exists.
[ ] Phase49 residual tables exist.
[ ] Phase48 seed-spread table exists.
[ ] Phase47 prediction provenance exists.
[ ] All checksums match.
[ ] Same FINAL_TEST_POP-v1.
[ ] Seed42 present.
[ ] Seed123 present.
[ ] Seed2026 present.
[ ] Residual convention y_true-y_pred.
[ ] Phase50 regime assignment frozen.
[ ] Final feature contract available.
[ ] Final lookback available.
[ ] No new inference needed.
```

---

# 177. Selection acceptance checklist

```text
[ ] Selection contract written before case narrative.
[ ] K_ABS=20.
[ ] K_SHARED=20.
[ ] K_UNDER=10.
[ ] K_OVER=10.
[ ] K_SHARED_SIGNED=10.
[ ] W1 ranks |e| descending.
[ ] W2 ranks mean |e| across seeds descending.
[ ] W3 uses residual>0.
[ ] W4 uses residual<0 magnitude.
[ ] Tie-break timestamp then target_id.
[ ] Squared-error duplicate ranking not created.
[ ] All selection IDs frozen/checksummed.
[ ] No manual case substitution.
```

---

# 178. Cross-seed acceptance checklist

```text
[ ] Pairwise top20 intersections computed.
[ ] Triple intersection computed.
[ ] Jaccard computed.
[ ] Membership count 0–3.
[ ] Shared hardness table covers all Test targets.
[ ] Seed spread joined from Phase48.
[ ] Sign consensus joined from Phase49.
[ ] Hardness-vs-spread Spearman computed.
[ ] No arbitrary high/low quadrant thresholds.
[ ] No best seed.
```

---

# 179. Regime acceptance checklist

```text
[ ] Regime labels loaded from frozen Phase50 assignment.
[ ] No thresholds recomputed.
[ ] W2 top20 regime composition computed.
[ ] Full-Test regime prevalence loaded.
[ ] Overrepresentation ratios computed where denominator>0.
[ ] Small top20 caveat documented.
[ ] No new cartesian regime mining.
```

---

# 180. Error concentration acceptance checklist

```text
[ ] Per-seed top20 SAE share.
[ ] Per-seed top20 SSE share.
[ ] Top1% concentration.
[ ] Top5% concentration.
[ ] Top10% concentration.
[ ] K percentage uses ceil(p*N).
[ ] Global SAE/SSE denominators match Phase49.
[ ] No RMSE recomputation after deleting worst cases.
[ ] No trimming.
```

---

# 181. Local context acceptance checklist

```text
[ ] Radius fixed ±6 steps.
[ ] Context gap-safe.
[ ] Test-boundary truncation documented.
[ ] Pre-target/post-target roles explicit.
[ ] Post-target values marked retrospective.
[ ] No new neighbor predictions generated.
[ ] Local plots generated for W2 top20.
[ ] Main report uses W2 ranks1–5.
[ ] No hand-picked local windows.
```

---

# 182. Input-window acceptance checklist

```text
[ ] Exact final lookback L*.
[ ] Exact final feature order.
[ ] Input rows = j-L ... j-1.
[ ] Target row excluded.
[ ] All input timestamps < target.
[ ] Exact continuity.
[ ] Feature fingerprint matches lock.
[ ] Continuous summaries generated.
[ ] Cyclical/binary summaries follow schema.
[ ] Historical target only model-visible if final feature set includes it.
[ ] Diagnostic target-history context separately labeled.
[ ] No future feature values.
```

---

# 183. Baseline-context acceptance checklist

```text
[ ] Persistence bundle common-target verified.
[ ] LSTM only if Phase47 bundle exists.
[ ] LSTM own frozen scaler provenance preserved upstream.
[ ] Baseline context restricted to selected target IDs.
[ ] Comparison labeled selection-conditioned.
[ ] No global superiority claim from worst-case subset.
[ ] No baseline retraining.
```

---

# 184. Casebook acceptance checklist

```text
[ ] Deduplicated case IDs.
[ ] Multi-label selection reasons.
[ ] Shared ranks preserved.
[ ] Per-seed ranks preserved.
[ ] Signed ranks preserved.
[ ] Truth/predictions/errors full precision in source table.
[ ] Regime tags present.
[ ] Seed consensus/spread present.
[ ] Local context references present.
[ ] Input context references present.
[ ] Baseline context references present.
[ ] Narrative factual only.
[ ] No causal attention claim.
```

---

# 185. Attention-handoff acceptance checklist

```text
[ ] W2 shared top20 IDs included.
[ ] Per-seed W1 top20 IDs included.
[ ] Shared all-under top10 IDs included.
[ ] Shared all-over top10 IDs included.
[ ] Handoff IDs deduplicated.
[ ] Selection roles retained.
[ ] Input start/end timestamps included.
[ ] Regime labels included.
[ ] Seed errors included.
[ ] Final checkpoint references included.
[ ] Feature/lookback/head/layer metadata included.
[ ] No attention extracted in Phase51.
[ ] Case selection independent of attention.
[ ] ready_for_phase52=true.
```

---

# 186. Scope acceptance checklist

```text
[ ] No new Test inference.
[ ] No training.
[ ] No best seed.
[ ] No prediction clipping.
[ ] No bias correction.
[ ] No prediction shifting.
[ ] No worst-case deletion.
[ ] No regime threshold change.
[ ] No new subgroup mining.
[ ] No attention extraction.
[ ] No attention-based case choice.
[ ] No unsupported causal claim.
```

---

# 187. Acceptance criteria

Phase51 PASS only when:

```text
All source prediction/residual/regime/seed-spread artifacts are verified and aligned to FINAL_TEST_POP-v1.

The worst-error selection contract is frozen before case narratives or manual plot inspection.

Each final Transformer seed receives a deterministic top-20 absolute-error ranking.

The cross-seed shared-hardness score is computed as mean absolute error across the three seeds and a deterministic shared top-20 case list is frozen.

Worst underprediction and worst overprediction lists are generated separately using the locked residual sign convention.

Squared-error ranking is not misrepresented as an independent ranking because it is order-equivalent to absolute-error ranking.

Cross-seed top-error overlap, Jaccard similarity and membership counts are quantified.

Error concentration is quantified without deleting or trimming any Test cases.

All selected cases retain the immutable Phase50 Train-defined regime labels.

Shared hard-case regime composition is compared descriptively with full Test prevalence without changing regime definitions.

Cross-seed hardness is compared with Phase48 seed disagreement without inventing post-hoc thresholds.

Persistence and eligible frozen LSTM predictions are joined only as selection-conditioned case context.

Every selected case uses the same fixed ±6-step local context and does not cross temporal gaps.

Model-visible input-window summaries use the exact locked final lookback/features and exclude the target/future.

Diagnostic historical target context is explicitly marked as model-visible or not depending on the final feature set.

The deterministic casebook contains all required case facts and no unsupported causal explanation.

Worst-case IDs, ranks, input-window metadata and regime labels are frozen and handed to Phase52.

No new inference, retraining, best-seed selection, prediction correction, case deletion or attention-based case replacement occurs.
```

---

# 188. Failure conditions

Phase51 FAIL if:

```text
source checksums mismatch

selection K/ranking changes after case inspection

worst cases are manually swapped

a different residual sign convention is used

a seed is omitted

shared hardness is called an ensemble score

squared-error ranking is treated as an independent case ordering

Phase50 regime thresholds are recomputed

worst cases are deleted as outliers

global Test metrics are recomputed after removing worst cases

local context crosses gaps

future/post-target rows are presented as model inputs

target row enters model input summary

a non-final feature is labeled model-visible

new neighboring predictions are generated

Persistence/LSTM worst-case context is used as a global fair ranking

model is retrained after inspecting cases

predictions are clipped/bias-corrected/time-shifted

attention is extracted and used to replace cases

unsupported causal narratives are written.
```

---

# 189. Common mistakes

## 189.1 Xếp top20 bằng squared error rồi nói đó là danh sách khác absolute error

Sai. Hai ranking giống nhau vì squaring monotonic theo |e|.

## 189.2 Chỉ xem worst20 của seed có Test RMSE thấp nhất

Sai. Cả ba seeds đều phải được phân tích.

## 189.3 Average ba predictions rồi tính error để chọn shared worst

Không theo contract. Shared hardness dùng:

```text
mean của ba absolute errors
```

không phải error của seed-mean prediction.

## 189.4 Shared top20 có nhiều HIGH nên đổi threshold HIGH

Sai. Regime labels đã frozen từ Train.

## 189.5 Thấy top20 chiếm nhiều SSE rồi loại chúng và báo RMSE mới

Sai Test trimming.

## 189.6 Dùng ±30 phút cho case này, ±2 giờ cho case khác

Sai cherry-pick. Tất cả dùng ±6 steps.

## 189.7 Dùng future outcome rows làm “input context”

Sai. Post-target rows chỉ là retrospective context.

## 189.8 Final feature set không có historical Appliances nhưng vẫn nói model đã nhìn thấy prior target

Sai. Phải đánh dấu `historical_target_model_visible=false`.

## 189.9 Persistence tốt hơn ở worst cases nên kết luận Persistence global tốt hơn

Sai selection-conditioned comparison.

## 189.10 Nhìn case plot rồi tự kết luận head attention bị lỗi

Attention chưa được extract ở Phase51.

---

# 190. Recommended report narrative

Một worst-error analysis chuẩn nên đi theo logic:

```text
How large are the worst errors?
→ W1.

Are the same timestamps hard across seeds?
→ W2/W5.

Are failures mostly underprediction or overprediction?
→ W3/W4 + sign consensus.

Do a few cases dominate global SSE?
→ concentration.

Which frozen regimes contain the shared hard cases?
→ Phase50 annotations.

Are shared hard cases also seed-sensitive?
→ hardness vs spread.

What happened immediately before/after each case?
→ fixed local context.

What exact information was visible to the model?
→ input-window summary.

How do naive/frozen baselines behave at the same timestamps?
→ selection-conditioned baseline context.

Which cases should be passed to attention analysis?
→ frozen Phase52 handoff.
```

---

# 191. Summary-writing rule

Do not turn case analysis into a model redesign section.

Final paragraph should distinguish:

```text
current evidence
limitations
future-work hypotheses
attention-analysis questions.
```

---

# 192. Phase52 questions generated by Phase51

Phase51 may generate questions such as:

```text
Do shared all-under hard cases show similar temporal attention concentration across seeds?

Does attention allocation differ between shared hard cases with low vs high seed disagreement?

Do rapid-change worst cases show concentration on the most recent input positions?

Are attention patterns stable across seed42/123/2026 for the same target?
```

These are questions only.

No answer until Phase52–57.

---

# 193. No attention causality premise

Even later, attention is not automatically explanation.

Phase51 handoff should preserve neutral wording:

```text
attention allocation
attention pattern
temporal focus
```

not:

```text
feature importance cause.
```

---

# 194. Recommended execution pseudocode

```text
p50 = load_phase50_signoff()
assert p50.overall_status in {"PASS","PASS_WITH_WARNING"}

residual_long = load_phase49_residual_long_table()
residual_wide = load_phase49_residual_wide_table()
regimes = load_frozen_phase50_test_regime_assignment()
seed_spread = load_phase48_seed_spread()

verify_all_checksums()
verify_same_FINAL_TEST_POP()

selection_contract = freeze_selection_contract(
    K_ABS=20,
    K_SHARED=20,
    K_UNDER=10,
    K_OVER=10,
    K_SHARED_SIGNED=10,
    local_context_radius_steps=6,
    tie_break=["timestamp_asc","target_id_asc"]
)

# W1
per_seed_top20 = []
for seed in [42,123,2026]:
    df_s = residual_long[seed]
    top = deterministic_sort(
        df_s,
        by=[
            abs_error_desc,
            timestamp_asc,
            target_id_asc
        ]
    ).head(20)
    per_seed_top20.append(top)

# W2
shared = build_target_level_cross_seed_table(
    residual_wide,
    seed_spread
)

shared["mean_abs_error_across_seeds"] = mean(
    [
        abs(shared.residual_seed42),
        abs(shared.residual_seed123),
        abs(shared.residual_seed2026)
    ],
    axis=1
)

shared_top20 = deterministic_sort(
    shared,
    by=[
        mean_abs_error_desc,
        timestamp_asc,
        target_id_asc
    ]
).head(20)

# W3/W4
under = {}
over = {}
for seed in [42,123,2026]:
    under[seed] = top_positive_residuals(
        residual_long,
        seed=seed,
        k=10
    )
    over[seed] = top_negative_residual_magnitude(
        residual_long,
        seed=seed,
        k=10
    )

# shared signed cases
shared_all_under = top_shared_by_consensus(
    shared,
    consensus="ALL_UNDER",
    k=10,
    score="mean_abs_error_across_seeds"
)

shared_all_over = top_shared_by_consensus(
    shared,
    consensus="ALL_OVER",
    k=10,
    score="mean_abs_error_across_seeds"
)

freeze_case_ids_before_narrative(
    per_seed_top20,
    shared_top20,
    under,
    over,
    shared_all_under,
    shared_all_over
)

overlap = compute_top20_overlap_jaccard(
    per_seed_top20
)

concentration = compute_error_concentration(
    residual_long,
    fixed_levels=["TOP20",0.01,0.05,0.10],
    percentage_k_rule="ceil"
)

shared_top20 = join_frozen_regime_labels(
    shared_top20,
    regimes
)

regime_composition = summarize_case_regime_composition(
    shared_top20,
    full_test_assignment=regimes
)

hardness_spread = compute_hardness_vs_seed_spread(
    shared
)

baseline_context = join_frozen_baseline_predictions_if_available(
    selected_case_ids=union_all_case_ids(...)
)

casebook_ids = deduplicate_preserving_selection_roles(
    per_seed_top20,
    shared_top20,
    under,
    over,
    shared_all_under,
    shared_all_over
)

local_context = build_fixed_gap_safe_context(
    casebook_ids,
    radius_steps=6,
    cadence_minutes=10,
    no_new_predictions=True
)

input_windows = reconstruct_locked_model_input_windows(
    casebook_ids,
    final_lookback=FINAL_L,
    final_feature_contract=FINAL_FEATURES,
    protocol="WB0"
)

assert all_input_timestamps_before_target(input_windows)
assert target_rows_excluded(input_windows)

feature_summary = summarize_model_visible_input_features(
    input_windows
)

target_history_summary = summarize_historical_target_context(
    casebook_ids,
    model_visible=historical_appliances_in_final_features
)

generate_deterministic_casebook_figures(
    shared_top20,
    local_context
)

write_factual_casebook(
    casebook_ids,
    local_context,
    feature_summary,
    target_history_summary,
    baseline_context
)

attention_cases = build_deduplicated_attention_handoff_table(
    shared_top20,
    per_seed_top20,
    shared_all_under,
    shared_all_over
)

write_phase52_handoff(
    cases=attention_cases,
    case_selection_independent_of_attention=True
)

assert no_new_test_inference
assert no_model_training
assert no_best_seed_selection
assert no_prediction_correction
assert no_regime_threshold_change
assert no_attention_extraction

signoff_phase51()
```

---

# 195. Definition of Done

\[
\boxed{
Frozen\ Error\ Sources
+
Frozen\ Ranking\ Contract
+
Per\text{-}Seed\ Top20
+
Shared\ Top20
+
Worst\ Under/Over
+
Cross\text{-}Seed\ Overlap
+
Error\ Concentration
+
Frozen\ Regime\ Annotation
+
Fixed\ Local\ Context
+
Exact\ Input\text{-}Window\ Context
+
Deterministic\ Casebook
+
No\ Retuning
+
Phase52\ Handoff
}
\]

---

# 196. Final status contract

```text
PHASE 51 investigates worst Test errors.

Source:
frozen Phase47 predictions
verified Phase49 residuals
frozen Phase50 regime labels
Phase48 seed spread.

Residual:
y_true - y_pred.

Selection:
W1 top20 |e| per seed.
W2 top20 mean |e| across seeds.
W3 top10 underprediction per seed.
W4 top10 overprediction per seed.
shared top10 ALL_UNDER.
shared top10 ALL_OVER.

Tie-break:
timestamp ascending
then target_id ascending.

Squared-error ranking:
not duplicated because ordering equals |e| ranking.

Cross-seed:
top20 overlap
Jaccard
membership count
hardness vs seed spread
sign consensus.

Context:
fixed ±6 target steps
gap-safe
post-target values retrospective only.

Model-visible context:
exact final lookback
exact final features
input ends at t-1
target/future excluded.

Regimes:
immutable Phase50 labels.

Baselines:
case-level context only
selection-conditioned.

Forbidden:
new Test inference
best seed
case swapping
outlier deletion
trimmed Test metric
prediction correction
regime threshold change
attention extraction
attention-based case replacement
unsupported causal claims.

Output:
frozen deterministic worst-case casebook
+
case IDs/input metadata for Phase52.

After WORST_ERROR_ANALYSIS-v1 PASS:
proceed to
PHASE 52 — Attention Extraction.
```

---

# 197. Final check

Correct:

```text
verify frozen errors/regimes
→ freeze top-K rules
→ rank all seeds
→ rank shared hard cases
→ freeze case IDs
→ overlap/concentration
→ annotate regimes
→ fixed local context
→ exact model-input context
→ deterministic casebook
→ Phase52 attention handoff
```

Incorrect:

```text
inspect errors
→ choose interesting examples manually
```

Incorrect:

```text
remove worst errors
→ report improved Test RMSE
```

Incorrect:

```text
use future rows as model-visible context
```

Incorrect:

```text
look at attention first
→ choose cases with nice heatmaps
```

Chỉ sau khi:

```text
WORST_ERROR_ANALYSIS-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase52_ready = true
```

mới chuyển sang **PHASE 52 — Attention Extraction**.
