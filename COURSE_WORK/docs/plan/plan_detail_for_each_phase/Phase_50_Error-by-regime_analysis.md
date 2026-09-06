# PHASE 50 — ERROR-BY-REGIME ANALYSIS

## Kế hoạch phân tích sai số theo các regime được định nghĩa trước bằng Train-only thresholds trên Held-Out Test

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Residual convention:** `residual = y_true - y_pred`  
**Upstream residual analysis:** `RESIDUAL_ANALYSIS-v1`  
**Primary prediction source:** frozen Phase47 Test prediction bundles  
**Regime-threshold source:** original Train only  
**Phase ID:** `PHASE_50_ERROR_BY_REGIME_ANALYSIS`  
**Output version:** `ERROR_BY_REGIME-v1`  
**Phase trước:** `Phase_49_Residual_analysis.md`  
**Phase sau:** `Phase_51_Worst-error_analysis.md`

---

# 1. Vai trò của Phase 50

Phase 50 trả lời câu hỏi:

> Final Transformer mắc lỗi nhiều hơn hoặc ít hơn trong những loại điều kiện nào?

Khác với Phase49, Phase50 không tập trung vào hình dạng residual tổng thể mà chia Held-Out Test thành một số **regime đã được định nghĩa trước** rồi so sánh sai số giữa các regime.

Điểm phương pháp luận quan trọng nhất:

```text
Test được dùng để đánh giá regime
nhưng
không được dùng để chọn ngưỡng regime.
```

Tất cả ngưỡng numeric chính phải được lấy từ:

```text
original Train only.
```

Mục tiêu:

```text
1. Khóa regime definitions trước khi xem regime-specific model error.
2. Tạo Train-only threshold artifacts có provenance rõ ràng.
3. Gán regime labels cho exact FINAL_TEST_POP-v1.
4. Freeze regime assignment trước khi join residual/model errors.
5. Tính MAE/RMSE/R²/MBE theo regime cho từng Transformer seed.
6. Aggregate regime metrics across seeds bằng mean ± sample SD.
7. Định lượng regime nào đóng góp nhiều nhất vào SAE/SSE tổng.
8. So sánh Transformer với Persistence theo từng regime.
9. So sánh với frozen LSTM Test baseline nếu Phase47 cho phép.
10. Không tạo cartesian regime mining sau khi nhìn Test.
11. Không chọn best seed.
12. Không sửa model sau khi phát hiện regime yếu.
13. Handoff regime labels và error evidence sang Phase51.
```

Nguyên tắc trung tâm:

\[
\boxed{
Train\text{-}Defined\ Regimes
+
Frozen\ Test\ Predictions
+
All\ Three\ Seeds
+
Common\ Targets
+
Descriptive\ Regime\ Metrics
+
No\ Test\text{-}Driven\ Thresholds
+
No\ Retuning
}
\]

---

# 2. Phase50 không phải một vòng model selection mới

Forbidden:

```text
retune features
retune lookback
retune loss
retune learning rate
retune RevIN
retune epoch
select another Transformer
select a seed
clip predictions
bias-correct predictions
train a regime-specific model
```

Nếu một regime cho RMSE rất cao:

```text
report
→ investigate in Phase51
→ discuss as limitation/future work
```

không được quay lại tuning.

---

# 3. Source of truth

Required upstream artifacts:

```text
phase_49_signoff.json
phase50_error_regime_handoff.json
residual_long_table.csv
residual_wide_table.csv
residual_bias_summary.csv
residual_acf_diagnostics.csv
residual_seed_sign_consensus.csv
```

Required original Test prediction provenance:

```text
phase_47_signoff.json
final_test_population_manifest.json
prediction_checksums.json
```

Required Train-side contracts:

```text
SPLIT-v1
WINDOWPOP-v1
TEMPORAL-v1
FEATURESETS-v1
```

---

# 4. Regime threshold source is original Train, not Train+Validation

Canonical reference:

```text
REGIME_REFERENCE_TRAIN-v1
```

defined as:

```text
original Train target IDs
∩ WINDOWPOP-v1
∩ temporally valid target IDs
```

using raw `Appliances` Wh.

Do not use:

```text
Validation
Test
Train+Validation final refit population
```

to estimate regime thresholds.

---

# 5. Why original Train only

Using original Train only ensures regime cut points were estimable from data available before:

```text
Validation model selection
Rolling-Origin robustness
Final Test access.
```

This creates a clean analysis contract.

---

# 6. Raw target coordinate

All target-derived regime thresholds use:

```text
raw Appliances Wh.
```

Do not use:

```text
YS1 standardized y_model
RevIN normalized coordinates.
```

This keeps regime definitions interpretable and identical across seeds/models.

---

# 7. Regime families

Phase50 locks six primary regime families:

```text
R1 — Target-level regime
R2 — Extreme-high-demand regime
R3 — One-step change-magnitude regime
R4 — One-step change-direction regime
R5 — Time-of-day regime
R6 — Day-type regime
```

No other regime family may be promoted to primary after Test inspection.

---

# 8. Why only six primary regime families

They cover:

```text
load level
extreme demand
dynamic transitions
direction of change
intraday context
weekday/weekend context
```

without creating an uncontrolled subgroup search.

---

# 9. No full regime Cartesian product

Forbidden primary mining such as:

```text
HIGH
× RAPID_CHANGE
× EVENING
× WEEKEND
```

unless predeclared in a future separate analysis.

Reason:

```text
small cells
multiple-comparison inflation
post-hoc pattern mining.
```

Phase50 analyzes regime axes separately.

---

# 10. R1 — Target-level regime

Train-only thresholds:

\[
Q25_y
=
quantile_{0.25}(y_{train})
\]

\[
Q75_y
=
quantile_{0.75}(y_{train})
\]

Canonical labels:

```text
TL_LOW
TL_MID
TL_HIGH.
```

---

# 11. Target-level assignment

Lock inequalities:

```text
TL_LOW:
y_true < Q25_y

TL_MID:
Q25_y <= y_true < Q75_y

TL_HIGH:
y_true >= Q75_y
```

This assigns every Test target exactly once.

---

# 12. Why Q25/Q75

This creates interpretable:

```text
low
central
high
```

target-load regimes while leaving a broader middle regime and avoiding Test-derived cut points.

---

# 13. Train quantile tie handling

Quantile values may coincide with repeated discrete Appliances values.

Do not perturb thresholds using Test.

If:

```text
Q25_y < Q75_y
```

use them exactly.

If:

```text
Q25_y >= Q75_y
```

mark:

```text
TARGET_LEVEL_THRESHOLD_DEGENERATE
```

and stop R1 execution pending a predefined Train-only resolution.

Recommended resolution rule, if needed:

```text
choose nearest distinct observed Train target value below/above the degenerate quantile positions
using Train values only.
```

The resolution must be recorded before joining Test residuals.

---

# 14. R2 — Extreme-high-demand regime

Train-only threshold:

\[
Q90_y
=
quantile_{0.90}(y_{train})
\]

Labels:

```text
EXTREME_HIGH
NON_EXTREME.
```

Assignment:

```text
EXTREME_HIGH:
y_true >= Q90_y

NON_EXTREME:
y_true < Q90_y.
```

---

# 15. Why Q90 for extreme-high

R1 `TL_HIGH` includes the upper quartile.

R2 separately isolates approximately the highest-demand tail using a fixed Train-derived cutoff.

This helps answer:

> Does model error increase specifically during unusually high consumption?

---

# 16. R2 is not a replacement for Phase51

R2 evaluates all Test points above the Train Q90 threshold.

Phase51 will inspect individual worst-error events.

Do not select only the largest R2 errors here.

---

# 17. R3 — One-step change-magnitude regime

Define raw target change:

\[
\Delta y_t
=
y_t-y_{t-1}
\]

only when the previous observation is exactly:

```text
10 minutes earlier
```

and belongs to the same continuity segment.

Train reference values:

\[
|\Delta y|_{train}
\]

from `REGIME_REFERENCE_TRAIN-v1`.

---

# 18. R3 Train-only threshold

\[
Q90_{\Delta}
=
quantile_{0.90}(|\Delta y|_{train})
\]

Labels:

```text
CHANGE_NORMAL
CHANGE_RAPID.
```

Assignment:

```text
CHANGE_RAPID:
|Δy_test| >= Q90_delta

CHANGE_NORMAL:
|Δy_test| < Q90_delta.
```

---

# 19. Gap-safe change regime

If Test target has no exact 10-minute predecessor:

```text
CHANGE_UNCLASSIFIED.
```

Do not fabricate:

```text
Δy=0
```

or cross a temporal gap.

---

# 20. First Test target under WB0

If the first Test target has an exactly 10-minute prior target in Validation:

```text
the prior actual observation may be used
```

for `Δy_test`.

This is consistent with the WB0 rolling one-step timeline.

---

# 21. R4 — One-step change-direction regime

Using the same gap-safe:

\[
\Delta y_t=y_t-y_{t-1}
\]

Labels:

```text
DIR_UP
DIR_FLAT
DIR_DOWN
DIR_UNCLASSIFIED.
```

Assignment:

```text
DIR_UP:
Δy > 0

DIR_FLAT:
Δy = 0

DIR_DOWN:
Δy < 0

DIR_UNCLASSIFIED:
no exact valid predecessor.
```

No epsilon.

---

# 22. Why exact direction categories

Target values are observed raw Wh values.

A threshold-free exact sign definition avoids introducing another arbitrary Test-derived cutoff.

---

# 23. R5 — Time-of-day regime

Uses raw dataset timestamp clock without timezone relocalization.

Fixed six-hour blocks:

```text
TOD_NIGHT:
00:00–05:59

TOD_MORNING:
06:00–11:59

TOD_AFTERNOON:
12:00–17:59

TOD_EVENING:
18:00–23:59.
```

---

# 24. Timestamp timezone caveat

Dataset timestamp timezone is not automatically localized.

Phase50 uses the clock value encoded in the source timestamp.

Do not claim:

```text
local solar time
occupancy schedule
official timezone
```

unless independently supported.

---

# 25. R6 — Day-type regime

Using day-of-week:

```text
DAY_WEEKDAY:
Monday–Friday

DAY_WEEKEND:
Saturday–Sunday.
```

No learned threshold.

---

# 26. No holiday regime

Dataset does not have a frozen holiday calendar contract.

Do not add:

```text
holiday/non-holiday
```

after Test inspection.

---

# 27. Regime assignment must be frozen before residual join

Execution separation:

```text
Stage A:
compute Train-only thresholds

Stage B:
using only Test target truth/timestamps,
assign Test regime labels

Stage C:
freeze regime assignment file/checksum

Stage D:
join residual/prediction bundles
and compute error metrics.
```

This prevents error values from influencing regime definition.

---

# 28. Stage A may not inspect Test error columns

Threshold script should only need:

```text
Train raw y
Train timestamps/continuity
```

No Test prediction/residual values.

---

# 29. Stage B uses Test truth only for descriptive classification

Test `y_true` is allowed because Phase47 already authorized Test evaluation.

But regime boundaries remain Train-derived/fixed.

---

# 30. Stage C assignment immutability

After:

```text
test_regime_assignment.csv
```

is generated and checksummed:

```text
do not change labels after seeing regime metrics.
```

---

# 31. Common regime assignment across seeds

Regime labels depend on:

```text
y_true
timestamp
previous actual y
Train thresholds.
```

They do **not** depend on seed prediction.

Therefore all Transformer seeds use identical regime labels.

Hard.

---

# 32. Common regime assignment across baselines

Persistence/LSTM direct regime comparisons use the same:

```text
test_regime_assignment.csv.
```

No model-specific regime cut points.

---

# 33. Minimum regime count policy

Do not drop small regimes.

Every regime reports:

```text
N.
```

Add descriptive warning:

```text
SMALL_N_WARNING
```

when:

```text
N < 30.
```

This is a reporting flag only.

It does not exclude the regime or change metrics.

---

# 34. Why small-N regimes are retained

Dropping a difficult small subgroup after observing Test performance would bias analysis.

Keep it, report uncertainty/caution.

---

# 35. R² within regimes

R² can be unstable or uninformative when within-regime target variance is small.

Therefore:

```text
MAE and RMSE = primary regime error measures
MBE = signed-bias measure
R² = supplementary.
```

If target variance denominator is zero:

```text
R² = NOT_DEFINED.
```

Do not force a numeric value.

---

# 36. Per-regime core metrics

For each seed and each regime:

```text
N
sample_share
target_mean_wh
target_std_wh
prediction_mean_wh
MAE_Wh
RMSE_Wh
R²
MBE_Wh
underprediction_fraction
overprediction_fraction
SAE
SSE.
```

---

# 37. Sample share

Within a regime family:

\[
SampleShare_r
=
\frac{N_r}
{\sum_{r'\in family}N_{r'}}
\]

For families with `UNCLASSIFIED` change targets:

Primary classified-regime shares may be reported against:

```text
classified subset
```

but unclassified count/fraction must also be shown.

---

# 38. SAE

\[
SAE_r
=
\sum_{i\in r}|e_i|.
\]

---

# 39. SSE

\[
SSE_r
=
\sum_{i\in r}e_i^2.
\]

---

# 40. Absolute-error contribution share

For disjoint regimes within a family:

\[
SAEShare_r
=
\frac{SAE_r}
{\sum_{r'}SAE_{r'}}.
\]

---

# 41. Squared-error contribution share

\[
SSEShare_r
=
\frac{SSE_r}
{\sum_{r'}SSE_{r'}}.
\]

This is especially informative because global RMSE is based on squared errors.

---

# 42. Error disproportionality

Define:

\[
SSEDisproportion_r
=
SSEShare_r-SampleShare_r.
\]

Interpretation:

```text
> 0:
regime contributes more squared error than its prevalence

< 0:
regime contributes less squared error than its prevalence.
```

---

# 43. SAE disproportionality

Optional complement:

\[
SAEDisproportion_r
=
SAEShare_r-SampleShare_r.
\]

---

# 44. Regime RMSE lift relative to seed-global Test RMSE

For seed `s`, regime `r`:

\[
RMSELift_{s,r}
=
\frac{RMSE_{s,r}}
{RMSE_{s,global}}-1.
\]

Percentage:

\[
RMSELiftPct
=
100\times RMSELift.
\]

---

# 45. Interpretation of RMSE lift

```text
positive:
regime harder than seed-global Test average

negative:
regime easier.
```

No threshold required.

---

# 46. MAE lift

Optional:

\[
MAELiftPct
=
100
\left(
\frac{MAE_{regime}}{MAE_{global}}-1
\right).
\]

---

# 47. Cross-seed aggregation

For every regime metric compute across:

```text
seed42
seed123
seed2026.
```

Primary aggregated reporting:

```text
mean
sample SD
min
max.
```

Use:

```text
ddof=1.
```

---

# 48. No 3N pooling for seed aggregate

Do not concatenate all seed residual rows and treat them as one 3N sample.

The same Test target is repeated across seeds.

Use per-seed regime metrics first, then aggregate.

---

# 49. Regime difficulty rank per seed

Within each regime family:

```text
rank regimes by RMSE
```

for each seed.

Then report:

```text
hardest-regime count across seeds.
```

No seed selection.

---

# 50. Cross-seed regime-rank stability

Create:

```text
same hardest regime across all 3 seeds?
same easiest regime across all 3 seeds?
```

Descriptive.

---

# 51. Baseline comparison — Persistence

Persistence is always authorized if Phase47 bundle exists.

For every regime:

```text
same target IDs
same regime labels
same METRICS-v1.
```

Compute:

\[
\Delta RMSE_{P\rightarrow T_s,r}
=
RMSE_{P,r}-RMSE_{T_s,r}.
\]

Positive:

```text
Transformer seed has lower regime RMSE.
```

---

# 52. Aggregate Transformer-vs-Persistence regime delta

Across three Transformer seeds:

```text
mean delta RMSE
sample SD delta RMSE
min delta
max delta
number of seeds with positive delta.
```

---

# 53. Baseline comparison — frozen LSTM

Only if:

```text
Phase47 LSTM Test bundle exists
and
common Test population is verified.
```

Use exactly the same regime assignment.

No LSTM retraining.

---

# 54. LSTM training-protocol caveat

Retain:

```text
LSTM_TUNED_DEV
```

caveat from Phase47.

The LSTM Test comparator may not share the Transformer Train+Validation final-refit protocol.

Therefore:

```text
regime-wise LSTM comparison = supplementary
Phase44 rolling-origin remains cleaner fairness evidence.
```

---

# 55. LSTM regime delta

For seed `s`, regime `r`:

\[
\Delta RMSE_{L\rightarrow T_s,r}
=
RMSE_{LSTM,r}-RMSE_{T_s,r}.
\]

Positive:

```text
Transformer lower regime RMSE.
```

---

# 56. No baseline retuning by regime

Do not train:

```text
a peak-specialized LSTM
a rapid-change Transformer
a weekend model.
```

---

# 57. No regime-specific prediction correction

If `TL_HIGH` is underpredicted:

```text
report.
```

Do not add a regime-specific bias offset.

---

# 58. No Test-derived threshold refinement

Forbidden:

```text
Q90 seems too broad
→ change to Test Q95

Q25/Q75 not balanced on Test
→ choose Test tertiles.
```

Train thresholds remain unchanged.

---

# 59. Thresholds need not create balanced Test counts

That is expected under temporal distribution shift.

Imbalanced Test regime sizes are themselves informative.

---

# 60. Train-vs-Test regime prevalence

For each threshold-based regime family, compare:

```text
Train regime sample fractions
vs
Test regime sample fractions.
```

This is a useful descriptive distribution-shift indicator.

---

# 61. Regime prevalence shift

For regime `r`:

\[
PrevalenceShift_r
=
TestShare_r-TrainShare_r.
\]

No significance test required.

---

# 62. Why prevalence shift matters

If Test contains more high/rapid-change cases than Train:

```text
global Test difficulty may partly reflect regime composition change.
```

This is descriptive evidence, not causal proof.

---

# 63. Time-of-day prevalence

Time blocks depend on dataset coverage.

Report counts.

No balancing/resampling.

---

# 64. Day-type prevalence

Report:

```text
weekday count
weekend count.
```

No weighting to equalize.

---

# 65. No weighted headline metric

Phase47 global Test metric remains authoritative.

Phase50 does not reweight regimes to produce a new “balanced Test RMSE”.

---

# 66. Optional macro-regime average

Do not use as final performance.

If shown:

```text
unweighted mean of regime RMSE
```

must be labeled:

```text
DESCRIPTIVE_MACRO_REGIME_ONLY.
```

Preferred:

```text
omit.
```

---

# 67. Target-level regime interpretation

Questions:

```text
Does RMSE increase from TL_LOW → TL_MID → TL_HIGH?
Is MBE more positive in TL_HIGH?
Does TL_HIGH contribute disproportionate SSE?
Do all seeds show the same ordering?
```

No monotonicity assumption is required.

---

# 68. Extreme-high regime interpretation

Questions:

```text
What fraction of Test targets exceed Train Q90?
What is RMSE/MAE there?
How much global SSE comes from EXTREME_HIGH?
Does Persistence become relatively stronger/weaker?
```

---

# 69. Change-magnitude interpretation

Questions:

```text
Is CHANGE_RAPID much harder than CHANGE_NORMAL?
Is Transformer gain over Persistence larger or smaller during rapid changes?
Does residual become strongly positive during rapid increases?
```

The last question involving direction is explored using R4.

---

# 70. Change-direction interpretation

Questions:

```text
Is DIR_UP harder than DIR_DOWN?
Does positive MBE dominate on DIR_UP?
Does model systematically overpredict on DIR_DOWN?
```

This can reveal underreaction to abrupt rises/falls.

---

# 71. Time-of-day interpretation

Questions:

```text
Which fixed clock block has largest RMSE?
Does bias direction change by time block?
Are errors concentrated in morning/evening periods?
```

No occupancy claim without evidence.

---

# 72. Day-type interpretation

Questions:

```text
Does weekend error differ from weekday error?
Is any difference stable across seeds?
```

No causal claim about behavior/occupancy.

---

# 73. Primary regime table hierarchy

Create six separate regime tables rather than one huge mixed table:

```text
target_level
extreme_high
change_magnitude
change_direction
time_of_day
day_type.
```

This improves interpretability and avoids mixing incompatible partitions.

---

# 74. Common seed columns

Each table should contain either long format or:

```text
regime
N
seed42_RMSE
seed123_RMSE
seed2026_RMSE
mean_RMSE
sd_RMSE
mean_MAE
sd_MAE
mean_MBE
sd_MBE
...
```

Store full-precision long table as source of truth.

---

# 75. Recommended source-of-truth format

Use:

```text
regime_metrics_long.csv
```

with rows:

```text
regime_family
regime_label
model_id
seed_if_any
N
MAE
RMSE
R²
MBE
...
```

Then create report-ready wide tables.

---

# 76. Error contribution metrics apply by seed

Because Transformer predictions differ by seed:

```text
SSE share
SAE share
disproportion
```

are computed per seed, then summarized across seeds.

---

# 77. Train regime assignment artifact

Create:

```text
train_regime_assignment.csv
```

using Train target truth and fixed definitions.

Purpose:

```text
threshold provenance
Train prevalence
audit.
```

No model predictions needed.

---

# 78. Test regime assignment artifact

Create:

```text
test_regime_assignment.csv
```

before error join.

Fields include all six regime families.

---

# 79. Test regime assignment schema

```text
target_id
target_timestamp
y_true_wh
previous_actual_y_wh_if_contiguous
delta_y_wh
abs_delta_y_wh
target_level_regime
extreme_high_regime
change_magnitude_regime
change_direction_regime
time_of_day_regime
day_type_regime
continuity_valid_for_change
threshold_version
```

No prediction/residual columns.

---

# 80. Threshold artifact

Create:

```text
regime_thresholds_train_only.json
```

with:

```text
source_region
source_population
source_target_ids_sha256
source_target_count
Q25_y
Q75_y
Q90_y
Q90_abs_delta
quantile_method
raw_unit=Wh
created_before_error_join=true
Test_values_used_for_thresholds=false
```

---

# 81. Quantile method must be frozen

Use the project’s Python/pandas/NumPy default only if explicitly recorded.

Recommended lock:

```text
quantile interpolation/method = linear
```

or the exact library equivalent.

Do not allow different threshold results across environments.

---

# 82. Quantile implementation audit

Verify:

```text
same input array
same quantile method
same thresholds on rerun.
```

Write threshold fingerprint.

---

# 83. Change threshold Train source

`Q90_abs_delta` is computed only from Train pairs satisfying:

```text
same continuity segment
exact 10-minute difference
both observations available
no split crossing outside original Train reference.
```

---

# 84. Train first target in each continuity segment

It has no valid delta.

Exclude only from `Δy` threshold estimation.

Do not exclude from target-level thresholds.

---

# 85. Test change-regime unclassified targets

Retain in assignment table.

For R3/R4 metric tables:

```text
either include UNCLASSIFIED as an explicit row
or report count separately.
```

Preferred:

```text
explicit UNCLASSIFIED row
```

with warning that it reflects missing adjacent continuity, not a behavioral regime.

---

# 86. Primary interpretation excludes UNCLASSIFIED

Do not compare it substantively as if it were:

```text
normal/rapid
up/down.
```

Use it as integrity/context row.

---

# 87. Error join

After regime assignment checksum is frozen:

```text
join residual_long_table
on target_id
```

Hard:

```text
many-to-one:
three seed residual rows
→ one regime assignment row.
```

---

# 88. Join integrity

Expected:

```text
each seed-target row matches exactly one regime assignment.
```

No dropped residual row.

No duplicate assignment.

---

# 89. Model-baseline join

Persistence/LSTM bundles join to the same assignment using:

```text
target_id.
```

No new regime calculation.

---

# 90. Per-regime metric computation

For each:

```text
regime_family
regime_label
seed/model
```

extract exact target subset and compute:

```text
N
MAE
RMSE
R²
MBE
UF
OF
SAE
SSE
sample share
SAE share
SSE share
RMSE lift.
```

---

# 91. R² guard

Within a regime:

```text
if N < 2
or
variance(y_true)==0
```

set:

```text
R2 = NOT_DEFINED.
```

Do not emit misleading zero.

---

# 92. MBE guard

Always raw Wh.

Interpretation uses project residual sign:

```text
MBE > 0:
underprediction tendency.
```

---

# 93. Under/over fraction

Use exact sign:

```text
e > 0
e < 0
e == 0.
```

No epsilon.

---

# 94. Contribution denominator

For each family, use exactly the rows classified in that family.

For complete families:

```text
R1,R2,R5,R6
```

denominator equals full Test population.

For change families:

```text
R3,R4
```

if `UNCLASSIFIED` row included, denominator still equals full Test population.

Preferred:

```text
include UNCLASSIFIED in contribution accounting
```

so shares sum to 1.

---

# 95. Contribution-share audit

Within each seed and family:

```text
sum sample_share ≈ 1
sum SAE_share ≈ 1
sum SSE_share ≈ 1.
```

Tolerance due floating point only.

---

# 96. Global metric linkage

For each seed:

\[
SSE_{global}
=
\sum_{r\in family}SSE_r
\]

for every complete disjoint regime family.

Hard audit for:

```text
R1
R2
R5
R6
```

and R3/R4 when UNCLASSIFIED included.

---

# 97. Why contribution audit matters

It verifies that:

```text
no Test samples vanished
no regime overlap occurred within a family
no double counting occurred.
```

---

# 98. Regime-wise metric aggregation across seeds

Create:

```text
mean_RMSE
sd_RMSE
mean_MAE
sd_MAE
mean_MBE
sd_MBE
mean_SSE_share
sd_SSE_share
```

with seed count:

```text
3.
```

---

# 99. No statistical superiority test across 3 seeds

Do not claim:

```text
significantly worse regime.
```

Use descriptive effect sizes.

---

# 100. No naive Test-timestamp t-tests

Time-series errors are dependent.

Phase50 uses:

```text
counts
metrics
effect sizes
seed consistency.
```

No iid t-test.

---

# 101. Optional future technique — block bootstrap

A time-block bootstrap could estimate uncertainty while respecting temporal dependence.

It is **not part of the core Phase50 protocol** because:

```text
block size would require another predeclared design choice
coursework does not require inferential regime testing.
```

Mention as future extension only.

---

# 102. Regime difficulty effect size

For regime `r`:

```text
RMSE lift vs seed-global Test RMSE
```

is the primary within-seed difficulty effect.

This is more interpretable than p-values here.

---

# 103. Pairwise regime contrasts

For R1:

```text
TL_HIGH - TL_LOW
TL_HIGH - TL_MID
```

in RMSE/MAE/MBE.

For R3:

```text
CHANGE_RAPID - CHANGE_NORMAL.
```

For R6:

```text
WEEKEND - WEEKDAY.
```

These are descriptive differences.

---

# 104. Time-of-day contrasts

Do not enumerate all six pairwise comparisons.

Report:

```text
hardest block
easiest block
range of RMSE across blocks
seed consistency.
```

No multiple-testing p-values.

---

# 105. Change-direction contrast

Useful:

```text
DIR_UP vs DIR_DOWN RMSE
DIR_UP vs DIR_DOWN MBE.
```

This can show asymmetric response to rises vs falls.

---

# 106. Extreme-high contrast

Primary:

```text
EXTREME_HIGH vs NON_EXTREME
```

RMSE/MAE/MBE and contribution shares.

---

# 107. Prevalence shift table

Create:

```text
regime_prevalence_train_vs_test.csv
```

for:

```text
R1
R2
R3
R4
R5
R6.
```

Columns:

```text
family
regime
Train N
Train share
Test N
Test share
share difference.
```

---

# 108. Train/Test change regime comparability

Train and Test change assignments use the same:

```text
Q90_abs_delta
exact sign definition
gap-safe continuity.
```

---

# 109. No train model error needed

Train regime prevalence does not require Train predictions.

Threshold/proportion analysis uses target/timestamps only.

---

# 110. Optional Validation prevalence

Do not make it primary.

Phase50 contract only requires:

```text
Train vs Test.
```

This keeps analysis compact.

---

# 111. Transformer-vs-Persistence regime win count

For each regime:

```text
0–3 Transformer seeds
```

may have lower RMSE than Persistence.

Report:

```text
transformer_seed_win_count_vs_persistence.
```

No best seed.

---

# 112. Transformer-vs-LSTM regime win count

If LSTM eligible:

```text
0–3.
```

Supplementary.

---

# 113. Baseline-delta sign convention

For baseline `B` and Transformer seed `T`:

\[
\Delta RMSE_{B\rightarrow T}
=
RMSE_B-RMSE_T.
\]

Positive:

```text
Transformer better.
```

Use same convention everywhere.

---

# 114. No percent delta when baseline RMSE near zero

Not expected, but if using percentage:

\[
100 \times \frac{RMSE_B-RMSE_T}{RMSE_B}
\]

only when:

```text
RMSE_B > 0.
```

Raw Wh delta remains primary.

---

# 115. Regime analysis of seed disagreement

Optional but useful:

For each regime:

```text
mean cross-seed prediction range
mean seed SD
```

using Phase48 seed-spread artifact.

This answers:

> Are certain regimes more stochastic across seeds?

---

# 116. Seed-spread regime analysis is secondary

Do not call it predictive uncertainty.

Label:

```text
cross-seed prediction spread by regime.
```

---

# 117. Seed residual sign consensus by regime

Using Phase49:

```text
ALL_UNDER
ALL_OVER
MIXED
```

compute fractions within each regime.

Useful particularly for:

```text
TL_HIGH
EXTREME_HIGH
CHANGE_RAPID
DIR_UP.
```

---

# 118. Shared-error interpretation

If `EXTREME_HIGH` has high `ALL_UNDER` fraction:

Safe:

```text
all three seeds tended to underpredict the same high-demand targets.
```

Do not infer cause.

---

# 119. No seed consensus threshold tuning

Consensus categories fixed from Phase49.

---

# 120. Regime analysis artifact directory

```text
artifacts/
└── error_by_regime/
    ├── error_by_regime_manifest.json
    ├── error_by_regime_contract.json
    ├── phase50_preflight_audit.csv
    ├── regime_reference_train_manifest.json
    ├── regime_thresholds_train_only.json
    ├── regime_threshold_audit.csv
    ├── regime_threshold_fingerprint.json
    ├── train_regime_assignment.csv
    ├── test_regime_assignment.csv
    ├── test_regime_assignment_audit.csv
    ├── test_regime_assignment_fingerprint.json
    ├── regime_error_join_audit.csv
    ├── regime_metrics_long.csv
    ├── regime_metrics_target_level.csv
    ├── regime_metrics_extreme_high.csv
    ├── regime_metrics_change_magnitude.csv
    ├── regime_metrics_change_direction.csv
    ├── regime_metrics_time_of_day.csv
    ├── regime_metrics_day_type.csv
    ├── regime_cross_seed_summary.csv
    ├── regime_error_contribution.csv
    ├── regime_rmse_lift.csv
    ├── regime_pairwise_contrasts.csv
    ├── regime_rank_stability.csv
    ├── regime_prevalence_train_vs_test.csv
    ├── regime_seed_spread_summary.csv
    ├── regime_seed_sign_consensus_summary.csv
    ├── regime_persistence_comparison.csv
    ├── regime_lstm_comparison.csv
    ├── regime_baseline_delta_summary.csv
    ├── regime_analysis_findings.csv
    ├── regime_analysis_tests.csv
    ├── regime_analysis_discrepancies.json
    ├── phase51_worst_error_regime_handoff.json
    ├── error_by_regime_summary.json
    ├── error_by_regime_report.md
    ├── figures/
    │   ├── REGIME_50_01_target_level_rmse.png
    │   ├── REGIME_50_02_target_level_mae.png
    │   ├── REGIME_50_03_target_level_mbe.png
    │   ├── REGIME_50_04_target_level_sse_share.png
    │   ├── REGIME_50_05_extreme_high_comparison.png
    │   ├── REGIME_50_06_change_magnitude_rmse.png
    │   ├── REGIME_50_07_change_direction_rmse_mbe.png
    │   ├── REGIME_50_08_time_of_day_rmse.png
    │   ├── REGIME_50_09_day_type_rmse.png
    │   ├── REGIME_50_10_regime_rmse_lift_heatmap.png
    │   ├── REGIME_50_11_sample_share_vs_sse_share.png
    │   ├── REGIME_50_12_train_vs_test_prevalence.png
    │   ├── REGIME_50_13_transformer_vs_persistence_delta.png
    │   ├── REGIME_50_14_transformer_vs_lstm_delta.png
    │   ├── REGIME_50_15_seed_spread_by_regime.png
    │   └── REGIME_50_16_seed_sign_consensus_by_regime.png
    ├── README_ERROR_BY_REGIME_ANALYSIS.md
    └── phase_50_signoff.json
```

If LSTM Test bundle does not exist:

```text
regime_lstm_comparison.csv
```

may contain a single `NOT_APPLICABLE` metadata row or be omitted according to project artifact policy.

---

# 121. Required outputs

```text
O50.1  Analysis manifest
O50.2  Analysis contract
O50.3  Preflight audit
O50.4  Train regime reference manifest
O50.5  Train-only threshold artifact
O50.6  Threshold audit
O50.7  Threshold fingerprint
O50.8  Train regime assignment
O50.9  Frozen Test regime assignment
O50.10 Test regime assignment audit
O50.11 Test assignment fingerprint
O50.12 Error join audit
O50.13 Long regime metrics
O50.14 Target-level metrics
O50.15 Extreme-high metrics
O50.16 Change-magnitude metrics
O50.17 Change-direction metrics
O50.18 Time-of-day metrics
O50.19 Day-type metrics
O50.20 Cross-seed regime summary
O50.21 Error contribution table
O50.22 RMSE-lift table
O50.23 Pairwise regime contrasts
O50.24 Regime rank stability
O50.25 Train-vs-Test prevalence
O50.26 Seed spread by regime
O50.27 Seed sign consensus by regime
O50.28 Persistence comparison
O50.29 LSTM comparison if eligible
O50.30 Baseline delta summary
O50.31 Figures
O50.32 Findings
O50.33 Phase51 handoff
O50.34 Tests
O50.35 Discrepancies
O50.36 Summary JSON
O50.37 Human-readable report
O50.38 README
O50.39 Sign-off
```

---

# 122. Analysis manifest

`error_by_regime_manifest.json`:

```text
phase=50
version=ERROR_BY_REGIME-v1
source_phase49_version
source_phase47_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
threshold_source=ORIGINAL_TRAIN_ONLY
threshold_coordinate=RAW_APPLIANCES_WH
primary_regime_families=[
  TARGET_LEVEL,
  EXTREME_HIGH,
  CHANGE_MAGNITUDE,
  CHANGE_DIRECTION,
  TIME_OF_DAY,
  DAY_TYPE
]
new_inference=false
new_training=false
best_seed_selection=false
Test_derived_numeric_thresholds=false
cartesian_regime_mining=false
status
created_at
```

---

# 123. Analysis contract

`error_by_regime_contract.json` must freeze:

```text
Train-only threshold source.

R1:
Q25/Q75 target level.

R2:
Q90 target extreme-high.

R3:
Q90 of gap-safe |Δy|.

R4:
exact sign of gap-safe Δy.

R5:
fixed 6-hour time blocks.

R6:
weekday/weekend.

All three Transformer seeds retained.

Regime assignment frozen before error join.

Primary metrics:
MAE/RMSE.
Supplementary:
R²/MBE/sign balance.

Contribution:
sample share
SAE share
SSE share.

No Test threshold tuning.
No regime cartesian mining.
No model correction.
No retraining.
```

---

# 124. Preflight audit

`phase50_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Checks:

```text
Phase49 approved
Phase50 handoff exists
original Train split available
WINDOWPOP-v1 available
Train raw target values available
Train timestamps/continuity available
Test regime source available
three seed residuals aligned
Persistence availability known
LSTM eligibility known
no threshold file precomputed from Test
no new inference required.
```

---

# 125. Train reference manifest

`regime_reference_train_manifest.json`:

```text
reference_id=REGIME_REFERENCE_TRAIN-v1
split=TRAIN
population_policy=WINDOWPOP-v1
target_name=Appliances
target_unit=Wh
target_count
target_ids_sha256
first_target_timestamp
last_target_timestamp
continuity_version
Test_values_used=false
Validation_values_used=false
status
```

Runtime count only after execution.

---

# 126. Train-only threshold artifact

`regime_thresholds_train_only.json`:

```text
version=REGIME_THRESHOLDS-v1
source_reference=REGIME_REFERENCE_TRAIN-v1
target_ids_sha256
target_unit=Wh
quantile_method=LINEAR

target_level:
Q25
Q75

extreme_high:
Q90

change_magnitude:
Q90_abs_delta
valid_delta_pair_count

time_of_day:
fixed clock blocks

day_type:
weekday/weekend

created_before_test_error_join=true
Test_values_used=false
Validation_values_used=false
status
```

---

# 127. Threshold audit

`regime_threshold_audit.csv`:

```text
threshold_id
source_variable
source_split
source_count
quantile
quantile_method
threshold_value
finite
nondegenerate_if_required
Test_used
Validation_used
status
```

---

# 128. Threshold fingerprint

`regime_threshold_fingerprint.json`:

```text
threshold_file_sha256
Train_target_ids_sha256
quantile_method
source_code_version_optional
status
```

---

# 129. Train regime assignment

`train_regime_assignment.csv`:

```text
target_id
timestamp
y_true_wh
previous_y_if_contiguous
delta_y
abs_delta_y
target_level_regime
extreme_high_regime
change_magnitude_regime
change_direction_regime
time_of_day_regime
day_type_regime
```

No model predictions.

---

# 130. Test regime assignment

`test_regime_assignment.csv` same schema.

Hard:

```text
no y_pred
no residual
no model ID.
```

This proves regime labels are independent of prediction errors.

---

# 131. Test regime assignment audit

`test_regime_assignment_audit.csv`:

```text
check
expected
observed
status
```

Checks:

```text
same target count as FINAL_TEST_POP-v1
unique IDs
chronological
all six family labels present
R1 exactly one label
R2 exactly one label
R5 exactly one label
R6 exactly one label
R3/R4 classified or explicit UNCLASSIFIED
threshold version exact
no prediction fields present.
```

---

# 132. Test assignment fingerprint

`test_regime_assignment_fingerprint.json`:

```text
assignment_sha256
test_population_sha256
threshold_sha256
created_before_error_join
status
```

---

# 133. Error join audit

`regime_error_join_audit.csv`:

```text
seed/model
error_row_count
assignment_row_count
matched_count
unmatched_error_rows
duplicate_assignment_matches
same_test_population
status
```

Expected:

```text
unmatched_error_rows=0
duplicate_assignment_matches=0.
```

---

# 134. Long regime metrics schema

`regime_metrics_long.csv`:

```text
regime_family
regime_label
model_id
seed
N
sample_share
target_mean_wh
target_std_wh
prediction_mean_wh
mae_wh
rmse_wh
r2
mbe_wh
underprediction_fraction
overprediction_fraction
zero_fraction
sae
sse
sae_share
sse_share
sae_disproportion
sse_disproportion
global_mae_wh
global_rmse_wh
mae_lift_pct
rmse_lift_pct
small_n_warning
status
```

For baselines:

```text
seed = N/A.
```

---

# 135. Family-specific metrics tables

Each family-specific CSV is derived from the long table and may include:

```text
per-seed rows
cross-seed aggregate rows
baseline rows.
```

No independent recalculation logic if avoidable.

---

# 136. Cross-seed summary schema

`regime_cross_seed_summary.csv`:

```text
regime_family
regime_label
N
mean_mae_wh
sd_mae_wh
mean_rmse_wh
sd_rmse_wh
mean_r2
sd_r2
mean_mbe_wh
sd_mbe_wh
mean_sse_share
sd_sse_share
mean_rmse_lift_pct
sd_rmse_lift_pct
seed_count=3
status
```

---

# 137. Error contribution schema

`regime_error_contribution.csv`:

```text
regime_family
regime_label
seed
N
sample_share
sae_share
sse_share
sae_disproportion
sse_disproportion
status
```

---

# 138. RMSE lift schema

`regime_rmse_lift.csv`:

```text
regime_family
regime_label
seed
regime_rmse
global_seed_rmse
rmse_lift_wh
rmse_lift_pct
status
```

Where:

```text
rmse_lift_wh = regime_rmse - global_seed_rmse.
```

---

# 139. Pairwise contrast schema

`regime_pairwise_contrasts.csv`:

```text
regime_family
contrast_id
seed
regime_a
regime_b
rmse_a
rmse_b
delta_rmse_a_minus_b
mae_a
mae_b
delta_mae_a_minus_b
mbe_a
mbe_b
delta_mbe_a_minus_b
status
```

Predeclared contrasts only.

---

# 140. Predeclared contrasts

```text
R1:
TL_HIGH vs TL_LOW
TL_HIGH vs TL_MID

R2:
EXTREME_HIGH vs NON_EXTREME

R3:
CHANGE_RAPID vs CHANGE_NORMAL

R4:
DIR_UP vs DIR_DOWN

R6:
DAY_WEEKEND vs DAY_WEEKDAY.
```

R5 uses hardest/easiest summary rather than all pairwise contrasts.

---

# 141. Regime rank stability schema

`regime_rank_stability.csv`:

```text
regime_family
regime_label
rank_seed42
rank_seed123
rank_seed2026
hardest_seed_count
easiest_seed_count
mean_rank
rank_range
status
```

Exclude:

```text
UNCLASSIFIED
```

from substantive hardest/easiest ranking for change families.

---

# 142. Prevalence schema

`regime_prevalence_train_vs_test.csv`:

```text
regime_family
regime_label
train_count
train_share
test_count
test_share
test_minus_train_share
status
```

---

# 143. Seed spread by regime schema

`regime_seed_spread_summary.csv`:

```text
regime_family
regime_label
N
mean_seed_std_prediction
median_seed_std_prediction
p95_seed_std_prediction
mean_seed_range_prediction
p95_seed_range_prediction
status
```

Source:

```text
Phase48 prediction_seed_spread.csv.
```

---

# 144. Seed sign consensus by regime schema

`regime_seed_sign_consensus_summary.csv`:

```text
regime_family
regime_label
N
all_under_count
all_under_fraction
all_over_count
all_over_fraction
mixed_count
mixed_fraction
all_zero_count
all_zero_fraction
status
```

Source:

```text
Phase49 residual_seed_sign_consensus.csv.
```

---

# 145. Persistence comparison schema

`regime_persistence_comparison.csv`:

```text
regime_family
regime_label
transformer_seed
N
transformer_rmse
persistence_rmse
delta_rmse_persistence_to_transformer
transformer_mae
persistence_mae
delta_mae_persistence_to_transformer
transformer_mbe
persistence_mbe
transformer_better_rmse
status
```

---

# 146. LSTM comparison schema

Same, replacing baseline.

Include:

```text
training_protocol_caveat.
```

---

# 147. Baseline delta summary

`regime_baseline_delta_summary.csv`:

```text
baseline
regime_family
regime_label
mean_delta_rmse
sd_delta_rmse
min_delta_rmse
max_delta_rmse
transformer_seed_win_count
seed_count=3
status
```

---

# 148. Core Figure REGIME_50_01

Target-level RMSE.

Preferred:

```text
x = TL_LOW/TL_MID/TL_HIGH
y = RMSE
```

show:

```text
seed42
seed123
seed2026
```

or:

```text
mean ± seed SD
```

with individual seed points.

---

# 149. Figure REGIME_50_02

Target-level MAE.

Same style.

---

# 150. Figure REGIME_50_03

Target-level MBE.

Include horizontal zero line.

Positive:

```text
underprediction.
```

---

# 151. Figure REGIME_50_04

Target-level:

```text
sample share
vs
SSE share.
```

Use cross-seed mean SSE share plus SD or per-seed points.

This visually identifies disproportionate error contribution.

---

# 152. Figure REGIME_50_05

Extreme-high comparison:

```text
NON_EXTREME
EXTREME_HIGH
```

show RMSE/MAE/MBE in a compact report-friendly form.

---

# 153. Figure REGIME_50_06

Change magnitude:

```text
CHANGE_NORMAL
CHANGE_RAPID.
```

Include RMSE and baseline context if readable.

---

# 154. Figure REGIME_50_07

Change direction:

```text
DIR_UP
DIR_FLAT
DIR_DOWN.
```

Show:

```text
RMSE
MBE
```

prefer separate panels/files if one figure becomes cluttered.

---

# 155. Figure REGIME_50_08

Fixed time-of-day RMSE:

```text
Night
Morning
Afternoon
Evening.
```

Chronological order, not score order.

Do not sort bars by RMSE because time meaning matters.

---

# 156. Figure REGIME_50_09

Weekday vs weekend RMSE/MAE.

---

# 157. Figure REGIME_50_10

Regime RMSE-lift heatmap.

Rows:

```text
regime labels
```

columns:

```text
seed42
seed123
seed2026
```

values:

```text
RMSE lift % vs that seed’s global Test RMSE.
```

Do not combine incompatible regime labels without family separators.

---

# 158. Figure REGIME_50_11

Sample share vs SSE share.

Could use:

```text
x = sample share
y = SSE share
reference y=x.
```

Points colored/grouped by regime family.

Interpretation:

```text
above line = disproportionate SSE.
```

---

# 159. Figure REGIME_50_12

Train-vs-Test regime prevalence.

Useful for:

```text
TL_HIGH
EXTREME_HIGH
CHANGE_RAPID
time blocks
weekend.
```

---

# 160. Figure REGIME_50_13

Transformer vs Persistence RMSE delta.

Positive means Transformer better.

Use mean across seeds with individual seed points.

---

# 161. Figure REGIME_50_14

Transformer vs LSTM RMSE delta if eligible.

Must include training-protocol caveat in caption.

---

# 162. Figure REGIME_50_15

Cross-seed prediction spread by regime.

Label:

```text
seed spread
not uncertainty interval.
```

---

# 163. Figure REGIME_50_16

Cross-seed residual sign consensus.

Can show:

```text
ALL_UNDER
ALL_OVER
MIXED
```

fractions for key regimes.

No model correction.

---

# 164. Figure axis fairness

When comparing same metric across regimes:

```text
same y-axis scale
```

within the figure.

No visual exaggeration.

---

# 165. No regime bar ordering by favorable result

Use semantic order:

```text
LOW → MID → HIGH
NORMAL → RAPID
DOWN → FLAT → UP or UP → FLAT → DOWN, but lock one
NIGHT → MORNING → AFTERNOON → EVENING
WEEKDAY → WEEKEND.
```

Recommended direction order:

```text
DIR_DOWN
DIR_FLAT
DIR_UP
```

to follow numeric sign.

---

# 166. Findings codes

Possible:

```text
TARGET_LEVEL_ERROR_INCREASES_WITH_LOAD
TARGET_LEVEL_ERROR_NONMONOTONIC
HIGH_LOAD_UNDERPREDICTION
HIGH_LOAD_OVERPREDICTION
EXTREME_HIGH_ERROR_ELEVATED
EXTREME_HIGH_SSE_DISPROPORTIONATE
RAPID_CHANGE_ERROR_ELEVATED
RAPID_CHANGE_NOT_SUBSTANTIALLY_HARDER_DESCRIPTIVE
DIR_UP_HARDER_THAN_DIR_DOWN
DIR_DOWN_HARDER_THAN_DIR_UP
DIR_UP_UNDERPREDICTION_PATTERN
DIR_DOWN_OVERPREDICTION_PATTERN
TIME_OF_DAY_ERROR_VARIATION
WEEKEND_ERROR_HIGHER
WEEKDAY_ERROR_HIGHER
TRAIN_TEST_REGIME_PREVALENCE_SHIFT
REGIME_ORDER_STABLE_ACROSS_SEEDS
REGIME_ORDER_VARIES_ACROSS_SEEDS
HIGH_SEED_SPREAD_IN_SPECIFIC_REGIME
ALL_SEEDS_SHARE_UNDERPREDICTION_IN_REGIME
ALL_SEEDS_SHARE_OVERPREDICTION_IN_REGIME
TRANSFORMER_GAIN_OVER_PERSISTENCE_CONCENTRATED_IN_REGIME
PERSISTENCE_STRONGER_IN_SPECIFIC_REGIME
TRANSFORMER_GAIN_OVER_LSTM_CONCENTRATED_IN_REGIME
LSTM_STRONGER_IN_SPECIFIC_REGIME
SMALL_N_REGIME_WARNING
TRAIN_ONLY_THRESHOLDS_VERIFIED
TEST_ASSIGNMENT_FROZEN_BEFORE_ERROR_JOIN
NO_CARTESIAN_REGIME_MINING
NO_MODEL_SELECTION
NO_POST_TEST_CORRECTION
```

---

# 167. Findings wording

Safe:

> Using the Train-derived Q90 consumption threshold, the extreme-high Test regime contributed a larger share of squared error than its share of Test observations.

Safe:

> Rapid one-step changes, defined using the Train-only 90th percentile of absolute target changes, showed higher RMSE than normal changes across all three final seeds.

Safe:

> Positive mean residuals in the upward-change regime indicate a tendency toward underprediction when consumption increased.

Unsafe:

> The model fails because evening users turn on appliances.

Unsafe causal claim.

---

# 168. No causal regime conclusions

Time/day regimes are observational labels.

Do not infer:

```text
occupancy
human behavior
specific appliance activity
weather cause
```

without source evidence.

---

# 169. Distribution-shift wording

Safe:

> The proportion of Test targets classified as extreme-high was larger than in Train under the fixed Train-derived threshold.

Do not say:

```text
distribution shift caused Test error
```

unless appropriately qualified:

```text
may contribute / is consistent with.
```

---

# 170. Regime-specific R² caution

Do not headline:

```text
negative R² in TL_LOW
```

without noting:

```text
within-regime target variance may be narrow.
```

MAE/RMSE are primary.

---

# 171. Contribution-share interpretation

A high RMSE regime with very low N may not dominate total error.

That is why Phase50 reports:

```text
RMSE
and
SSE share.
```

Both are necessary.

---

# 172. Error contribution vs difficulty

Two distinct questions:

```text
Difficulty:
How large is error per sample?
→ RMSE/MAE

Contribution:
How much total error comes from this regime?
→ SAE/SSE share.
```

Do not conflate them.

---

# 173. Regime prevalence vs contribution

Use:

```text
sample share
vs
SSE share
```

to see disproportionate contribution.

---

# 174. No weighted re-optimization from error contribution

Do not create sample weights for retraining after Test.

Future work only.

---

# 175. Phase51 handoff purpose

Phase51 will inspect individual worst-error timestamps and local context.

Phase50 should attach regime labels so Phase51 can answer:

```text
Do worst errors cluster in HIGH?
EXTREME_HIGH?
RAPID_CHANGE?
DIR_UP?
certain time blocks?
```

without recomputing thresholds.

---

# 176. Phase51 handoff schema

`phase51_worst_error_regime_handoff.json`:

```text
source_phase50_version
final_lock_sha256
test_population_sha256
threshold_version
threshold_sha256
test_regime_assignment_path
test_regime_assignment_sha256
regime_metrics_long_path
regime_findings
seed_list=[42,123,2026]
residual_definition=y_true-y_pred
worst_error_ranking_not_performed_in_phase50=true
regime_threshold_source=TRAIN_ONLY
ready_for_phase51=true
```

---

# 177. Phase51 must not change regime thresholds

Handoff should state:

```text
regime labels immutable.
```

Worst-error cases can be annotated with labels but not used to redefine them.

---

# 178. Discrepancy taxonomy

`regime_analysis_discrepancies.json`:

```text
PHASE49_NOT_APPROVED
TRAIN_REFERENCE_MISSING
WINDOWPOP_TRAIN_REFERENCE_MISMATCH
TRAIN_TARGET_CHECKSUM_MISMATCH
VALIDATION_USED_FOR_THRESHOLD
TEST_USED_FOR_THRESHOLD
TARGET_THRESHOLD_IN_MODEL_SPACE
QUANTILE_METHOD_UNSPECIFIED
Q25_Q75_DEGENERATE
Q90_TARGET_INVALID
Q90_DELTA_INVALID
CHANGE_THRESHOLD_GAP_UNSAFE
TIME_OF_DAY_DEFINITION_DRIFT
DAY_TYPE_DEFINITION_DRIFT
TEST_REGIME_ASSIGNMENT_AFTER_ERROR_INSPECTION
TEST_ASSIGNMENT_CHECKSUM_MISMATCH
REGIME_LABEL_DEPENDS_ON_SEED
REGIME_LABEL_DEPENDS_ON_PREDICTION
ERROR_JOIN_ROW_DROP
ERROR_JOIN_DUPLICATE
REGIME_OVERLAP_WITHIN_FAMILY
REGIME_COVERAGE_INCOMPLETE
CONTRIBUTION_SHARE_SUM_MISMATCH
GLOBAL_SSE_RECONSTRUCTION_MISMATCH
GLOBAL_METRIC_MISMATCH
R2_FOR_ZERO_VARIANCE_FORCED_NUMERIC
SMALL_N_REGIME_DROPPED
TEST_THRESHOLD_RETUNED
TEST_QUANTILE_REGIME_USED
CARTESIAN_REGIME_MINING_ATTEMPT
REGIME_SPECIFIC_MODEL_TRAINING
REGIME_SPECIFIC_BIAS_CORRECTION
BEST_SEED_SELECTION_ATTEMPT
3N_SEED_POOLING_AS_IID
PERSISTENCE_POPULATION_MISMATCH
LSTM_POPULATION_MISMATCH
LSTM_RETRAINED
LSTM_SCALER_MISMATCH
BASELINE_DELTA_SIGN_CONVENTION_MISMATCH
CAUSAL_CLAIM_UNSUPPORTED
WORST_ERROR_SCOPE_CREEP
ATTENTION_SCOPE_CREEP
OTHER
```

---

# 179. Status model

## PASS

```text
Train-only thresholds verified
Test regime assignment frozen before error join
six primary regime families complete
all three seeds evaluated
contribution shares verified
cross-seed summaries complete
Persistence comparison complete
eligible LSTM handled correctly
no Test-derived threshold
no model correction
Phase51 handoff ready.
```

## PASS_WITH_WARNING

Possible:

```text
one regime N<30
change UNCLASSIFIED count nonzero
LSTM Test baseline unavailable
R² undefined in narrow regime
strong Train/Test prevalence shift
large regime-specific seed spread.
```

These are not methodological failures.

## FAIL

Examples:

```text
Test-derived thresholds
row drops
regime overlap
threshold drift
best-seed selection
post-Test correction
baseline mismatch.
```

---

# 180. Execution sequence

```text
1. Verify Phase49 signoff and source artifacts.
2. Build REGIME_REFERENCE_TRAIN-v1.
3. Verify Train target/population checksum.
4. Compute Q25_y/Q75_y/Q90_y.
5. Compute gap-safe Train |Δy|.
6. Compute Q90_abs_delta.
7. Freeze quantile method.
8. Write/fingerprint Train-only thresholds.
9. Assign Train regimes.
10. Assign Test regimes without predictions/residuals.
11. Audit Test assignment.
12. Freeze Test assignment checksum.
13. Join Test residuals to assignment.
14. Verify no row drop/duplicate.
15. Compute per-seed regime metrics.
16. Compute SAE/SSE contribution shares.
17. Reconstruct global SAE/SSE per family.
18. Compute RMSE/MAE lift vs global.
19. Compute predeclared regime contrasts.
20. Aggregate metrics across seeds.
21. Compute regime-rank stability.
22. Compare Train-vs-Test prevalence.
23. Join Phase48 seed spread by regime.
24. Join Phase49 sign consensus by regime.
25. Evaluate Persistence per regime.
26. Evaluate eligible LSTM per regime.
27. Compute baseline deltas.
28. Generate figures.
29. Write findings.
30. Write Phase51 handoff.
31. Run tests/discrepancy audit.
32. Write report/README/signoff.
```

---

# 181. Preflight acceptance checklist

```text
[ ] Phase49 PASS/PASS_WITH_WARNING.
[ ] Phase50 handoff exists.
[ ] Original Train split available.
[ ] WINDOWPOP-v1 Train target IDs available.
[ ] Train raw Appliances values available.
[ ] Train timestamps/continuity available.
[ ] Test residual table available.
[ ] Test population SHA verified.
[ ] Three seeds aligned.
[ ] Persistence availability known.
[ ] LSTM eligibility known.
[ ] No Test-derived threshold artifact exists.
[ ] No model inference required.
```

---

# 182. Threshold acceptance checklist

```text
[ ] Reference = original Train only.
[ ] No Validation values.
[ ] No Test values.
[ ] Raw Wh coordinate.
[ ] Quantile method explicitly locked.
[ ] Q25_y finite.
[ ] Q75_y finite.
[ ] Q25_y < Q75_y or Train-only degeneracy rule invoked.
[ ] Q90_y finite.
[ ] Train |Δy| is gap-safe.
[ ] Q90_abs_delta finite.
[ ] Threshold file created before Test error join.
[ ] Threshold SHA generated.
```

---

# 183. Assignment acceptance checklist

```text
[ ] Train assignment generated.
[ ] Test assignment generated.
[ ] Test assignment has no prediction fields.
[ ] Test assignment has no residual fields.
[ ] Every Test target has one R1 label.
[ ] Every Test target has one R2 label.
[ ] Every Test target has one R5 label.
[ ] Every Test target has one R6 label.
[ ] R3/R4 classified or UNCLASSIFIED.
[ ] Same labels used all seeds/models.
[ ] Assignment SHA generated before error join.
```

---

# 184. Metric acceptance checklist

```text
[ ] Per-seed N.
[ ] Per-seed MAE.
[ ] Per-seed RMSE.
[ ] Per-seed R² supplementary.
[ ] Per-seed MBE.
[ ] Underprediction fraction.
[ ] Overprediction fraction.
[ ] SAE.
[ ] SSE.
[ ] Sample share.
[ ] SAE share.
[ ] SSE share.
[ ] SAE/SSE disproportionality.
[ ] RMSE lift vs seed global.
[ ] Small-N warning but no dropping.
```

---

# 185. Contribution acceptance checklist

```text
[ ] Shares use one disjoint family at a time.
[ ] Sample shares sum to 1 within tolerance.
[ ] SAE shares sum to 1.
[ ] SSE shares sum to 1.
[ ] Family SSE reconstructs global seed SSE.
[ ] No target double counting within family.
[ ] UNCLASSIFIED change targets retained.
```

---

# 186. Cross-seed acceptance checklist

```text
[ ] All three seeds retained.
[ ] Regime metrics aggregated after per-seed computation.
[ ] Mean across seeds.
[ ] Sample SD ddof=1.
[ ] No 3N iid pooling.
[ ] Regime rank stability computed.
[ ] No best seed.
[ ] Seed spread by regime labeled non-uncertainty.
[ ] Sign consensus by regime computed.
```

---

# 187. Baseline acceptance checklist

```text
[ ] Persistence exact same Test IDs.
[ ] Same regime assignment.
[ ] Persistence regime MAE/RMSE/R²/MBE.
[ ] Delta sign convention baseline - Transformer.
[ ] Aggregate seed deltas.
[ ] LSTM only if frozen Test bundle eligible.
[ ] LSTM not retrained.
[ ] LSTM same Test IDs.
[ ] LSTM own scaler provenance preserved.
[ ] LSTM training-protocol caveat documented.
```

---

# 188. Scope acceptance checklist

```text
[ ] No Test-derived numeric cutoff.
[ ] No Test tertile/quantile regime.
[ ] No threshold changed after errors.
[ ] No regime Cartesian product mining.
[ ] No subgroup-specific retraining.
[ ] No bias correction.
[ ] No seed selection.
[ ] No weighted final metric replacing Phase47.
[ ] No p-value significance claims from naive iid tests.
[ ] No worst individual-error ranking.
[ ] No attention interpretation.
```

---

# 189. Reporting acceptance checklist

```text
[ ] Threshold provenance reported.
[ ] Train-vs-Test prevalence reported.
[ ] MAE/RMSE primary.
[ ] R² caveat included.
[ ] MBE sign interpretation included.
[ ] Difficulty vs contribution distinction included.
[ ] SSE share vs sample share included.
[ ] Rapid-change definition documented.
[ ] Time-of-day fixed blocks documented.
[ ] Raw timestamp timezone caveat included.
[ ] No causal occupancy claims.
[ ] Baseline caveats included.
[ ] Phase51 handoff generated.
```

---

# 190. Acceptance criteria

Phase50 PASS only when:

```text
All numeric regime thresholds are derived from original Train only in raw Wh space.

The Train reference population is versioned and checksummed.

The quantile method is explicit and reproducible.

Target-level regimes use Train Q25/Q75.

Extreme-high regime uses Train Q90.

Rapid-change regime uses the Train Q90 of gap-safe absolute one-step target changes.

Change direction uses the exact sign of gap-safe one-step target change.

Time-of-day and day-type regimes use fixed deterministic calendar rules.

Test regime labels are assigned and frozen before residual/prediction errors are joined.

Regime assignment is identical across Transformer seeds and baselines.

All three final Transformer seeds are evaluated in every applicable regime.

MAE/RMSE are the primary regime performance metrics; R² is supplementary.

MBE and under/overprediction fractions preserve the y_true-y_pred residual convention.

SAE/SSE contribution shares reconstruct global error totals within every disjoint regime family.

Cross-seed regime summaries use per-seed metrics followed by mean and sample SD, not 3N iid pooling.

Train-vs-Test regime prevalence is reported without altering Test weights.

Persistence is evaluated on the same Test IDs/regime labels.

The frozen LSTM baseline is compared only if its Phase47 Test bundle is eligible and common-target aligned.

No Test-derived thresholds, post-hoc regime mining, regime-specific correction, retraining or best-seed selection occurs.

The frozen regime assignment and findings are handed to Phase51 for worst-error analysis.
```

---

# 191. Failure conditions

Phase50 FAIL if:

```text
Validation or Test is used to estimate numeric regime thresholds

threshold is computed in standardized model space

Test quantiles are used to rebalance regimes

regime labels depend on predictions

regime labels differ by seed

thresholds change after regime RMSE is observed

Test assignment is generated after error inspection and altered

rows are dropped due difficult regime

regime families overlap internally

contribution shares do not reconcile

global SSE cannot be reconstructed

R² is forced when target variance is zero

small regimes are silently omitted

3 seed rows are treated as 3N independent observations

Persistence/LSTM use different Test targets

LSTM is retrained

a regime-specific model correction is created

a new weighted headline Test metric replaces Phase47

worst-error cases are used to redesign thresholds.
```

---

# 192. Common mistakes

## 192.1 Dùng Test Q25/Q75 vì Test có phân phối khác Train

Sai. Phải dùng Train thresholds.

## 192.2 High regime ít sample nên đổi Q75 thành Q60

Sai post-Test adjustment.

## 192.3 Rapid-change threshold lấy từ Test |Δy|

Sai.

## 192.4 Tính Δy xuyên qua gap

Sai temporal integrity.

## 192.5 Dùng predicted value để định nghĩa target-demand regime

Sai. Target regime dùng raw y_true với Train-defined thresholds.

## 192.6 Gộp HIGH×RAPID×EVENING rồi tìm subgroup tệ nhất

Sai cartesian mining.

## 192.7 Regime có RMSE cao rồi thêm sample weight và retrain

Sai post-Test model modification.

## 192.8 R² âm trong một narrow regime rồi kết luận model vô dụng

Sai; xem MAE/RMSE và target variance.

## 192.9 SSE share cao nhưng quên sample share

Sai interpretation.

## 192.10 Persistence mạnh ở rapid change rồi thay model

Không. Report result.

## 192.11 Chọn seed tốt nhất trong TL_HIGH

Sai best-seed selection.

## 192.12 Dùng LSTM scaler của Transformer

Sai checkpoint/scaler semantics.

---

# 193. Recommended report structure

`error_by_regime_report.md`:

```text
1. Objective
2. Why Train-only regimes are required
3. Regime reference Train population
4. Threshold derivation
5. Regime definitions
6. Test assignment and immutability
7. Metric and contribution definitions
8. Target-level results
9. Extreme-high-demand results
10. Change-magnitude results
11. Change-direction results
12. Time-of-day results
13. Weekday/weekend results
14. Train-vs-Test prevalence
15. Cross-seed regime stability
16. Seed spread/sign consensus by regime
17. Transformer vs Persistence by regime
18. Transformer vs frozen LSTM by regime
19. Difficulty vs error-contribution interpretation
20. Main findings
21. Handoff to worst-error analysis
22. Limitations
23. Definition of Done
```

---

# 194. README requirements

`README_ERROR_BY_REGIME_ANALYSIS.md` explains:

```text
why thresholds are Train-only
why raw Wh is used
why Q25/Q75 and Q90 are fixed
how rapid changes are computed gap-safely
why direction uses exact sign
why time blocks are fixed
why no Cartesian regime mining
why MAE/RMSE are primary
why R² is supplementary
what SSE share means
why seed metrics are aggregated after per-seed calculation
why no statistical p-value claims
how Phase51 consumes regime labels.
```

---

# 195. Summary artifact

`error_by_regime_summary.json`:

```text
version
source_phase49_version
final_lock_sha256
test_population_sha256
seed_list
Train_reference_sha256
threshold_sha256
test_assignment_sha256
thresholds
regime_counts
regime_prevalence_train_vs_test
target_level_summary
extreme_high_summary
change_magnitude_summary
change_direction_summary
time_of_day_summary
day_type_summary
error_contribution_summary
rmse_lift_summary
rank_stability_summary
seed_spread_summary
seed_sign_consensus_summary
persistence_comparison
lstm_comparison_if_available
findings
Test_derived_threshold_used=false
cartesian_regime_mining=false
best_seed_selected=false
model_retrained=false
prediction_corrected=false
phase51_ready
overall_status
```

---

# 196. Phase50 sign-off

`phase_50_signoff.json` minimum:

```text
phase=50
phase_name=Error-by-regime analysis
version=ERROR_BY_REGIME-v1
source_phase49_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
Train_reference_sha256
threshold_sha256
test_regime_assignment_sha256
Q25_y
Q75_y
Q90_y
Q90_abs_delta
quantile_method
target_level_complete
extreme_high_complete
change_magnitude_complete
change_direction_complete
time_of_day_complete
day_type_complete
contribution_reconstruction_verified
cross_seed_summary_complete
persistence_comparison_complete
lstm_comparison_status
Test_derived_threshold_used=false
regime_assignment_depends_on_prediction=false
cartesian_regime_mining=false
best_seed_selected=false
model_training=false
prediction_correction_applied=false
phase51_ready
warnings
overall_status
created_at
```

---

# 197. Recommended execution pseudocode

```text
p49 = load_phase49_signoff()
assert p49.overall_status in {"PASS","PASS_WITH_WARNING"}

# Stage A — Train-only thresholds
train_ref = build_regime_reference_train(
    split="TRAIN",
    population="WINDOWPOP-v1",
    raw_target="Appliances"
)

assert no_validation_values(train_ref)
assert no_test_values(train_ref)

Q25_y = quantile(
    train_ref.y_true_wh,
    0.25,
    method="linear"
)

Q75_y = quantile(
    train_ref.y_true_wh,
    0.75,
    method="linear"
)

Q90_y = quantile(
    train_ref.y_true_wh,
    0.90,
    method="linear"
)

train_delta = compute_gap_safe_actual_target_change(
    train_ref,
    cadence_minutes=10
)

Q90_abs_delta = quantile(
    abs(train_delta.valid_values),
    0.90,
    method="linear"
)

thresholds = freeze_threshold_artifact(
    Q25_y=Q25_y,
    Q75_y=Q75_y,
    Q90_y=Q90_y,
    Q90_abs_delta=Q90_abs_delta,
    source="TRAIN_ONLY"
)

# Stage B — assign Test regimes without predictions/errors
test_truth = load_FINAL_TEST_POP_truth_and_timestamps()

test_delta = compute_gap_safe_actual_target_change(
    test_truth,
    allow_contiguous_pretest_previous_actual=True,
    cadence_minutes=10
)

test_assignment = assign_regimes(
    y=test_truth.y_true_wh,
    delta=test_delta,
    thresholds=thresholds,
    time_blocks={
        "NIGHT": [0,6],
        "MORNING": [6,12],
        "AFTERNOON": [12,18],
        "EVENING": [18,24]
    },
    weekend_rule="SAT_SUN"
)

assert no_prediction_columns(test_assignment)
assert no_residual_columns(test_assignment)

freeze_and_checksum(test_assignment)

# Stage C — error join only after assignment frozen
residuals = load_phase49_residual_long_table()
verify_same_test_population(residuals, test_assignment)

joined = many_to_one_join(
    residuals,
    test_assignment,
    on="target_id"
)

assert no_rows_dropped(joined)
assert no_duplicate_matches(joined)

# Stage D — per-seed regime metrics
families = [
    "TARGET_LEVEL",
    "EXTREME_HIGH",
    "CHANGE_MAGNITUDE",
    "CHANGE_DIRECTION",
    "TIME_OF_DAY",
    "DAY_TYPE"
]

metrics_long = []

for family in families:
    for seed in [42,123,2026]:
        family_metrics = compute_regime_metrics(
            joined,
            family=family,
            seed=seed,
            residual_definition="y_true-y_pred",
            metrics=[
                "N",
                "MAE",
                "RMSE",
                "R2",
                "MBE",
                "UNDER_FRACTION",
                "OVER_FRACTION",
                "SAE",
                "SSE"
            ]
        )

        add_sample_sae_sse_shares(family_metrics)
        add_global_rmse_mae_lift(family_metrics)

        verify_family_contribution_reconstruction(
            family_metrics,
            seed
        )

        metrics_long.append(family_metrics)

metrics_long = concat(metrics_long)

cross_seed = aggregate_per_seed_regime_metrics(
    metrics_long,
    mean=True,
    sample_sd_ddof=1
)

contrasts = compute_only_predeclared_contrasts(
    metrics_long
)

rank_stability = compute_regime_rank_stability(
    metrics_long
)

prevalence = compare_train_vs_test_regime_prevalence(
    train_assignment,
    test_assignment
)

seed_spread = summarize_phase48_seed_spread_by_regime(
    test_assignment
)

sign_consensus = summarize_phase49_sign_consensus_by_regime(
    test_assignment
)

persistence = compute_persistence_regime_metrics_on_same_assignment()

if phase47_lstm_bundle_is_eligible():
    lstm = compute_frozen_lstm_regime_metrics_on_same_assignment()
else:
    lstm = NOT_APPLICABLE

baseline_deltas = compute_regime_baseline_deltas(
    transformer=metrics_long,
    persistence=persistence,
    lstm=lstm
)

generate_regime_figures()

write_phase51_handoff(
    regime_assignment=test_assignment,
    threshold_artifact=thresholds,
    regime_metrics=metrics_long
)

assert no_test_derived_threshold
assert no_best_seed_selection
assert no_model_retraining
assert no_prediction_correction
assert no_cartesian_regime_mining

signoff_phase50()
```

---

# 198. Definition of Done

\[
\boxed{
Train\text{-}Only\ Thresholds
+
Frozen\ Test\ Regime\ Assignment
+
Six\ Predeclared\ Regime\ Families
+
All\ Three\ Transformer\ Seeds
+
MAE/RMSE/MBE
+
SAE/SSE\ Contribution
+
Cross\text{-}Seed\ Stability
+
Persistence/LSTM\ Context
+
No\ Test\ Threshold\ Tuning
+
No\ Regime\ Mining
+
No\ Retuning
+
Phase51\ Handoff
}
\]

---

# 199. Final status contract

```text
PHASE 50 analyzes error by predeclared regimes.

Threshold source:
original Train only.

Coordinate:
raw Appliances Wh.

R1 target level:
Q25/Q75
LOW / MID / HIGH.

R2 extreme high:
Train Q90 target.

R3 change magnitude:
Train Q90 of gap-safe |Δy|.

R4 change direction:
DOWN / FLAT / UP
from exact gap-safe Δy sign.

R5 time of day:
00–06
06–12
12–18
18–24.

R6 day type:
weekday/weekend.

Critical workflow:
Train thresholds
→ Test regime assignment without predictions
→ freeze assignment
→ join residuals
→ compute metrics.

Seeds:
42
123
2026
all retained.

Primary regime metrics:
MAE
RMSE.

Supplementary:
R²
MBE
under/over fractions.

Contribution:
sample share
SAE share
SSE share
disproportion.

Baselines:
Persistence always if source exists.
Frozen LSTM only if Phase47 Test bundle eligible.

Forbidden:
Test-derived thresholds
threshold changes after errors
cartesian regime mining
best-seed selection
regime-specific correction
retraining
weighted replacement for Phase47 global Test metric.

After ERROR_BY_REGIME-v1 PASS:
proceed to
PHASE 51 — Worst-Error Analysis.
```

---

# 200. Final check

Correct:

```text
original Train
→ derive Q25/Q75/Q90/Q90|Δy|
→ freeze thresholds
→ assign Test regimes without errors
→ freeze assignment
→ join all 3 seed residuals
→ compute regime metrics/contributions
→ compare Persistence/LSTM
→ Phase51 handoff
```

Incorrect:

```text
inspect Test errors
→ choose cutoffs that make regimes interesting
```

Incorrect:

```text
high-error regime
→ retrain special model
```

Incorrect:

```text
HIGH × RAPID × EVENING × WEEKEND
→ search all combinations
→ report worst subgroup
```

Incorrect:

```text
select seed with lowest high-regime RMSE
```

Chỉ sau khi:

```text
ERROR_BY_REGIME-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase51_ready = true
```

mới chuyển sang **PHASE 51 — Worst-Error Analysis**.
