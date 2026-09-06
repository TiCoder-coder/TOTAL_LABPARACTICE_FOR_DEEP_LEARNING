# PHASE 48 — PREDICTION ANALYSIS

## Kế hoạch phân tích hành vi dự báo của ba Final Transformer seeds trên Held-Out Test sau `FINAL_TEST_EVAL-v1`

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Upstream evaluation:** `FINAL_TEST_EVAL-v1`  
**Phase ID:** `PHASE_48_PREDICTION_ANALYSIS`  
**Output version:** `PREDICTION_ANALYSIS-v1`  
**Phase trước:** `Phase_47_Final_test_evaluation.md`  
**Phase sau:** `Phase_49_Residual_analysis.md`

---

# 1. Vai trò của Phase 48

Phase 48 là bước **descriptive prediction-behavior analysis** sau khi final Test predictions đã được tạo và đóng băng ở Phase47.

Mục tiêu của Phase này không phải:

```text
đánh giá lại model
chọn seed tốt nhất
chọn checkpoint mới
retune hyperparameter
thay đổi preprocessing
tạo ensemble mới
```

Mục tiêu là trả lời:

> Mô hình cuối đang dự báo như thế nào trên chuỗi Test, nó bám mức tiêu thụ thực tế ra sao, có bắt được biến động/đỉnh/đáy hay bị làm mượt, ba seed có dự báo nhất quán hay không, và những hiện tượng nào cần được chuyển tiếp sang Residual/Error Analysis?

Phase48 chỉ sử dụng các **prediction bundles đã frozen từ Phase47**.

Nguyên tắc trung tâm:

\[
\boxed{
Frozen\ Predictions
+
Descriptive\ Analysis
+
All\ Three\ Seeds
+
No\ New\ Inference
+
No\ Model\ Selection
+
No\ Retuning
}
\]

---

# 2. Scope của Phase48

Phase48 tập trung vào:

```text
prediction level
temporal tracking
prediction distribution
actual-vs-predicted relationship
peak/trough capture
change-direction behavior
prediction smoothness
seed-to-seed agreement
prediction spread across seeds
descriptive calibration-like behavior
```

Phase48 **không** đi sâu vào:

```text
residual distribution          → Phase49
error-by-regime                → Phase50
worst-error case investigation → Phase51
attention maps                 → Phase52–57
```

---

# 3. Source of truth

Required upstream artifacts:

```text
phase_47_signoff.json
final_test_evaluation_contract.json
final_test_population_manifest.json
prediction_checksums.json
final_test_metrics_by_seed.csv
transformer_seed_aggregate_metrics.csv
final_test_model_comparison.csv
phase48_prediction_analysis_handoff.json
```

Required prediction bundles:

```text
final_test_predictions_seed42.csv
final_test_predictions_seed123.csv
final_test_predictions_seed2026.csv
```

Optional contextual baselines:

```text
final_test_predictions_persistence.csv
final_test_predictions_lstm_tuned_dev.csv
```

if they exist and were eligible/frozen in Phase47.

---

# 4. Phase48 must not rerun Test inference

Hard:

```text
new model inference = forbidden
```

Use only frozen prediction bundles.

If a required prediction bundle is missing/corrupt:

```text
STOP
```

and resolve via Phase47 artifact-integrity procedure.

Do not silently reload checkpoint and create a new prediction series from Phase48.

---

# 5. Prediction bundles are immutable

Before any analysis:

```text
verify file checksums
verify population hash
verify seed identity
verify target IDs
verify row count
verify ordering
```

Any mismatch:

```text
STOP.
```

---

# 6. Three Transformer seeds remain symmetric

Phase48 analyzes:

```text
seed42
seed123
seed2026
```

with equal scientific status.

No:

```text
best seed
representative seed selected by Test RMSE
seed pruning.
```

All main plots should either:

```text
show all three seeds
```

or use a clearly labeled descriptive cross-seed summary.

---

# 7. Cross-seed descriptive prediction center

Phase48 may compute:

\[
\hat y^{mean}_t
=
\frac{
\hat y^{42}_t+
\hat y^{123}_t+
\hat y^{2026}_t
}{3}
\]

but this quantity is:

```text
SEED_MEAN_PREDICTION_DESCRIPTIVE
```

not:

```text
ensemble prediction
final model output
new Test estimator.
```

No performance metric should be used to promote it as an ensemble.

---

# 8. Cross-seed spread

For each Test target timestamp `t`, compute:

```text
seed_mean_prediction
seed_std_prediction
seed_min_prediction
seed_max_prediction
seed_range_prediction
```

Using sample SD:

```text
ddof=1.
```

This measures prediction disagreement across the three predeclared seeds.

---

# 9. Seed spread is not uncertainty interval

Do not call:

```text
95% confidence interval
predictive uncertainty
Bayesian uncertainty
calibrated uncertainty band.
```

With only three deterministic seed realizations, correct wording is:

```text
cross-seed prediction spread
seed-to-seed variation.
```

---

# 10. Prediction analysis primary questions

Phase48 should answer:

```text
Q48.1  Do predictions track the overall Test trajectory?
Q48.2  Do predictions systematically look smoother than actual Appliances?
Q48.3  Are high-demand peaks captured or compressed?
Q48.4  Are low-demand periods tracked correctly?
Q48.5  Does the model react to rapid changes with visible delay?
Q48.6  Are three seeds visually/quantitatively consistent?
Q48.7  Where is cross-seed disagreement largest?
Q48.8  Does prediction distribution cover the range of true target values?
Q48.9  Is there compression toward the mean?
Q48.10 Are prediction increments directionally aligned with actual increments?
Q48.11 Do baselines show qualitatively different prediction behavior?
Q48.12 Which observations should be handed to Phase49–51 for deeper error analysis?
```

---

# 11. No new selection metric

Phase48 can compute descriptive statistics, but no result may:

```text
select a seed
select a model
select a baseline
select a hyperparameter
```

Final Test metrics were already fixed in Phase47.

---

# 12. No Test-driven threshold tuning

Any threshold used for visualization must be:

```text
structural
predefined
or descriptive from already locked data without model-selection consequences.
```

For peak/regime thresholds, preferred:

```text
defer formal thresholds to Phase50
```

where Train-derived thresholds are used.

Phase48 should avoid inventing Test quantile thresholds for scientific claims.

---

# 13. Base aligned analysis table

Create one canonical long-form table:

```text
target_id
target_timestamp
y_true_wh
seed
y_pred_wh
```

for each of the three seeds.

Also create one wide table:

```text
target_id
target_timestamp
y_true_wh
y_pred_seed42
y_pred_seed123
y_pred_seed2026
seed_mean_prediction
seed_std_prediction
seed_min_prediction
seed_max_prediction
seed_range_prediction
```

---

# 14. Base table integrity

Hard:

```text
one row per FINAL_TEST_POP-v1 target
same target IDs as Phase47
same timestamps
same y_true
no missing seed prediction
no duplicate target
all finite.
```

---

# 15. Prediction temporal plot — full Test horizon

Core figure:

```text
Actual Appliances
Transformer seed42
Transformer seed123
Transformer seed2026
```

over the full Test timeline.

Recommended visual emphasis:

```text
actual line
three seed prediction lines
```

but avoid making plot unreadable.

Alternative allowed:

```text
actual
seed-mean descriptive center
cross-seed min/max band
```

with clear label that band is seed spread, not confidence interval.

---

# 16. Full-horizon plot purpose

Use it to inspect:

```text
trajectory tracking
persistent level shifts
peak compression
prediction smoothness
seed overlap
large temporal mismatch regions.
```

Do not use it to manually choose “good periods” for metric recomputation.

---

# 17. Zoomed temporal panels

Because the full Test may be dense, Phase48 should generate deterministic zoom windows.

Recommended:

```text
Z1 = first 24 hours of FINAL_TEST_POP-v1
Z2 = middle 24-hour block
Z3 = last 24 hours
```

if enough samples exist.

At 10-minute cadence:

```text
24h ≈ 144 targets.
```

Runtime continuity/population remains source of truth.

---

# 18. Deterministic zoom selection

Do not cherry-pick visually interesting error periods.

Use deterministic temporal positions:

```text
first
middle
last.
```

Worst-error windows belong to Phase51.

---

# 19. Optional fixed 48-hour zoom

If visual clarity requires more context:

```text
48h = approximately 288 targets
```

may be used, but lock one convention and document it.

Do not switch window lengths after seeing plots to exaggerate behavior.

---

# 20. Actual-vs-predicted scatter

For each seed create:

```text
x = y_true_wh
y = y_pred_wh
```

with reference line:

```text
y=x.
```

Purpose:

```text
range compression
over/under response at high values
global relationship
prediction saturation.
```

---

# 21. Scatter axis policy

Use same axis limits across seeds.

Recommended:

```text
common min/max from combined y_true and three prediction arrays.
```

No seed-specific autoscaling that visually changes apparent fit.

---

# 22. Seed-mean scatter

Optional descriptive plot:

```text
x = y_true
y = seed_mean_prediction
```

Label:

```text
Descriptive mean of three seed predictions
not an evaluated ensemble.
```

No new RMSE.

---

# 23. Prediction distribution comparison

Compare:

```text
distribution of y_true
distribution of each seed y_pred.
```

Useful displays:

```text
histogram
ECDF
box/violin plot.
```

Preferred:

```text
ECDF + summary table
```

because distribution shapes/ranges are easier to compare without arbitrary bin choices.

---

# 24. Distribution analysis statistics

For each series:

```text
count
mean
std
min
Q1
median
Q3
max
IQR
p05
p95
```

These are descriptive.

Do not treat Test quantiles as thresholds for model selection.

---

# 25. Distribution range coverage

Compute:

```text
prediction_min / true_min
prediction_max / true_max
prediction_IQR / true_IQR
prediction_std / true_std
```

or difference/ratio where mathematically meaningful.

Purpose:

```text
detect prediction compression or over-dispersion.
```

---

# 26. Compression ratio

Define descriptive:

\[
CR_{std,s}
=
\frac{
SD(\hat y_s)
}{
SD(y)
}
\]

Interpretation:

```text
CR < 1 → prediction series less variable than truth
CR > 1 → prediction series more variable than truth.
```

Do not call it calibration metric.

---

# 27. IQR compression ratio

\[
CR_{IQR,s}
=
\frac{
IQR(\hat y_s)
}{
IQR(y)
}
\]

Useful robust complement.

---

# 28. Prediction mean shift

For seed `s`:

\[
MeanShift_s
=
Mean(\hat y_s)
-
Mean(y)
\]

This is a prediction-level descriptive bias indicator.

Formal residual bias analysis belongs to Phase49.

---

# 29. Keep prediction-vs-residual boundary clear

Phase48 may report:

```text
prediction mean minus truth mean
```

as a distribution-comparison statistic.

Do not expand into:

```text
residual normality
residual autocorrelation
residual heteroskedasticity
```

until Phase49.

---

# 30. Temporal smoothness analysis

Compute first differences:

\[
\Delta y_t
=
y_t-y_{t-1}
\]

\[
\Delta \hat y_{s,t}
=
\hat y_{s,t}-\hat y_{s,t-1}.
\]

Use only adjacent targets with exact 10-minute continuity.

---

# 31. Gap-safe first differences

If two adjacent prediction-bundle rows are not exactly 10 minutes apart:

```text
do not compute one cross-gap difference.
```

Respect temporal integrity.

---

# 32. Change-magnitude statistics

For actual and each seed:

```text
mean |Δ|
median |Δ|
p90 |Δ|
p95 |Δ|
std(Δ)
```

Compare prediction changes with true changes.

---

# 33. Change-magnitude ratio

For seed `s`:

\[
CMR_s
=
\frac{
Mean(|\Delta \hat y_s|)
}{
Mean(|\Delta y|)
}
\]

Descriptive:

```text
CMR < 1
→ prediction trajectory changes less sharply on average.
```

---

# 34. Directional-change agreement

For valid adjacent timestamps:

```text
sign(Δy)
vs
sign(Δŷ_s).
```

Compute:

```text
direction agreement rate.
```

Need a zero-change policy.

---

# 35. Zero-change policy

Use exact sign categories:

```text
NEGATIVE
ZERO
POSITIVE.
```

Agreement only if exact category matches.

Also report an alternate coarse rate excluding actual zero changes if useful.

Do not invent epsilon after seeing data.

---

# 36. Directional agreement is descriptive only

It is not a primary forecasting metric.

Do not use it to claim superiority over Phase47 RMSE.

---

# 37. Rapid-change tracking

Formal threshold-based “rapid change” analysis should preferably use Train-derived thresholds and may be moved to Phase50.

Phase48 may simply show:

```text
distribution of |Δy|
vs
|Δŷ|.
```

No Test-driven rapid-change threshold required.

---

# 38. Temporal lag diagnostic

A common forecasting behavior is apparent response delay.

Phase48 may compute a **diagnostic lag cross-correlation** between:

```text
y_true
and
y_pred
```

over a very small fixed lag range.

Predeclare:

```text
lags = -6 ... +6 steps
```

corresponding to:

```text
-60 ... +60 minutes
```

at 10-minute cadence.

---

# 39. Lag diagnostic is not model correction

The lag yielding highest correlation is:

```text
diagnostic only.
```

Do not shift predictions by that lag and recompute final RMSE.

No post-hoc alignment correction.

---

# 40. Lag sign convention

Lock:

```text
lag k > 0
means prediction series is compared to truth shifted k future steps
```

or another explicit convention.

Document carefully to avoid opposite interpretation.

Recommended artifact should include exact formula.

---

# 41. Gap-safe lag computation

Do not correlate across discontinuities.

Use contiguous valid Test segments.

Aggregate carefully.

If only one continuous Test segment, straightforward.

---

# 42. Cross-correlation caveat

High time-series autocorrelation can make neighboring lags similar.

Do not claim causal delay solely from cross-correlation.

Use wording:

```text
apparent temporal alignment diagnostic.
```

---

# 43. Peak/trough descriptive capture without Test threshold tuning

Instead of defining peak threshold from Test quantiles, use **local extrema**.

Define local extrema on true Test series using a fixed neighborhood.

Recommended:

```text
one-step local maximum:
y_t > y_{t-1} and y_t >= y_{t+1}

one-step local minimum:
y_t < y_{t-1} and y_t <= y_{t+1}
```

for contiguous timestamps only.

---

# 44. Local extrema edge handling

First/last sample of a continuous segment:

```text
not eligible for local-extrema classification.
```

No cross-gap neighbor use.

---

# 45. Peak amplitude capture ratio

At true local maxima:

\[
PeakRatio_s
=
\frac{
Mean(\hat y_{s,t} \mid t\in true\ local\ maxima)
}{
Mean(y_t \mid t\in true\ local\ maxima)
}
\]

Descriptive only.

Because very small peaks count too, also report:

```text
number of local maxima
median actual peak
median predicted-at-peak.
```

Formal high-demand regime analysis belongs Phase50.

---

# 46. Trough capture

At true local minima compare:

```text
actual level
predicted level
```

and report:

```text
mean/median predicted-minus-actual at local minima
```

without turning it into formal residual-regime inference.

---

# 47. Peak timing diagnostic

At each true local maximum, optionally search within fixed:

```text
±1 time step
```

for nearest local prediction maximum.

Report:

```text
same-step peak rate
±1-step peak rate.
```

This is a descriptive temporal-tracking diagnostic.

Do not shift official predictions.

---

# 48. Fixed peak timing window

Hard:

```text
±1 step = ±10 minutes.
```

Do not widen after seeing results.

---

# 49. Prediction autocorrelation

Phase48 may compare ACF of:

```text
y_true
y_pred per seed
```

at a small fixed set of lags:

```text
1
6
12
36
72
144
```

where supported.

Interpretation:

```text
does predicted series preserve short/medium periodic persistence?
```

---

# 50. Prediction ACF is not residual ACF

Residual autocorrelation belongs Phase49.

---

# 51. ACF lag semantics

At 10-minute cadence:

```text
1   = 10 min
6   = 1 h
12  = 2 h
36  = 6 h
72  = 12 h
144 = 24 h
```

Only compute if enough Test observations/continuous segment length.

---

# 52. No missing-lag fabrication

If Test segment too short:

```text
mark NOT_AVAILABLE.
```

---

# 53. Cross-seed agreement — prediction correlation

For each pair:

```text
42 vs 123
42 vs 2026
123 vs 2026
```

compute:

```text
Pearson correlation
Spearman correlation
mean absolute prediction difference
RMSE between predictions
max absolute prediction difference.
```

These quantify seed agreement.

---

# 54. Cross-seed pairwise metrics are not performance metrics

They compare model outputs to each other, not to ground truth.

Label:

```text
SEED_AGREEMENT_DIAGNOSTIC.
```

---

# 55. Cross-seed disagreement per timestamp

For each target:

```text
seed_range
seed_std.
```

This supports locating timestamps where stochastic training produces materially different predictions.

---

# 56. Top disagreement timestamps

Phase48 may list top `K` timestamps by:

```text
seed_range_prediction.
```

Lock:

```text
K = 20
```

for descriptive review.

This is not worst-error ranking.

Worst actual prediction errors belong Phase51.

---

# 57. Top-disagreement table

Fields:

```text
rank
target_id
timestamp
y_true
seed42_pred
seed123_pred
seed2026_pred
seed_mean
seed_std
seed_range
```

No model selection.

---

# 58. Cross-seed spread by time

Plot:

```text
timestamp
seed_std_prediction
```

or seed range.

Can reveal periods where model stochasticity matters more.

---

# 59. Prediction agreement with actual trend

Compute rolling descriptive correlation using a fixed window only if useful.

Recommended:

```text
window = 144 samples ≈ 24h
```

but only on contiguous sequence.

Outputs:

```text
rolling correlation actual vs prediction
```

per seed.

---

# 60. Rolling correlation caveat

Edge windows naturally missing until enough observations.

Do not fill artificially.

No threshold-based “bad period” selection.

---

# 61. Rolling prediction mean comparison

Optional:

```text
24h rolling mean actual
vs
24h rolling mean prediction.
```

Useful to see level tracking.

---

# 62. Rolling variability comparison

Optional:

```text
24h rolling std actual
vs
24h rolling std prediction.
```

This is a prediction smoothness diagnostic.

---

# 63. Rolling window fixed before inspection

Use:

```text
144 consecutive 10-minute samples
```

when continuity supports.

Do not tune smoothing window to make plot favorable.

---

# 64. Prediction range saturation check

Compute for each seed:

```text
fraction of predictions exactly equal to global min
fraction exactly equal to global max
number of unique prediction values
```

Useful to detect accidental clipping/saturation.

Expected for continuous regression:

```text
no pathological saturation.
```

---

# 65. No post-hoc clipping

If saturation/negative values found:

```text
report
```

do not correct.

---

# 66. Negative-prediction audit

Record:

```text
count
fraction
minimum prediction
timestamps
```

per seed.

This is descriptive integrity.

No set-to-zero.

---

# 67. Prediction physical range check

Compare with Train/Test observed range descriptively.

Do not enforce hard physical constraints unless locked before Test.

---

# 68. Exact-target repetition behavior

Appliances values may have repeated discrete Wh values.

Prediction outputs are continuous.

Do not expect exact target-level matching.

No classification-style exact accuracy metric.

---

# 69. Calibration-like binned plot — allowed with caution

Optional descriptive analysis:

```text
bin by predicted value
compare mean prediction vs mean observed target
```

But because bins would be Test-derived and this resembles calibration analysis, if used:

```text
use fixed equal-width or fixed-count bins
no model tuning
label descriptive only.
```

Preferred:

```text
10 equal-frequency bins based on prediction ranking per seed.
```

However this can create different bins per seed.

Safer default for Phase48:

```text
omit formal calibration binning
```

unless report space requires it.

---

# 70. Baseline prediction-behavior context

If frozen baseline bundles exist:

```text
Persistence
LSTM_TUNED_DEV
```

Phase48 may create a limited qualitative comparison.

Focus:

```text
trajectory smoothness
range
peak response
lag tendency.
```

Do not redo model-family ranking beyond Phase47 metrics.

---

# 71. Persistence-specific expected behavior

Persistence:

```text
ŷ_{t+1}=y_t
```

will naturally appear one step behind abrupt changes.

This is a useful visual reference for one-step forecasting behavior.

---

# 72. Transformer vs Persistence temporal plot

Optional deterministic 24h panel:

```text
actual
Transformer seed-mean descriptive line
Persistence
```

with note:

```text
seed-mean is descriptive, not ensemble.
```

---

# 73. LSTM temporal plot

Only if Phase47 LSTM bundle exists and common-target verified.

Label:

```text
LSTM_TUNED_DEV
```

with its training-protocol caveat.

---

# 74. No baseline cherry-pick

Use same deterministic zoom windows for all compared models.

---

# 75. Prediction analysis should remain global/descriptive

Avoid creating dozens of post-hoc subgroups.

Formal subgroup/regime evaluation is reserved for Phase50 with Train-defined thresholds.

---

# 76. No demographics/external context

Dataset is one household and time series.

Do not invent occupancy labels or appliance-use events not present in data.

---

# 77. Timestamp context

Allowed deterministic derived calendar labels:

```text
hour
day of week
weekend
```

for annotation.

But do not perform full error-by-hour analysis here; Phase50 can handle regimes.

---

# 78. Optional actual/prediction heatmap by day×time

A useful descriptive visualization:

```text
rows = calendar date
columns = 10-minute time-of-day slot
value = Appliances Wh.
```

Create separately for:

```text
actual
seed-mean prediction
```

only if Test covers enough full days.

This visualizes daily pattern tracking.

---

# 79. Heatmap caveat

No difference/error heatmap in Phase48 if it becomes residual-focused.

Difference heatmaps belong Phase49/50.

---

# 80. Seed-specific heatmaps

Not necessary by default.

Use seed-mean descriptive heatmap plus spread heatmap if helpful.

---

# 81. Cross-seed spread heatmap

Optional:

```text
rows=date
columns=time slot
value=seed_std_prediction.
```

Label clearly:

```text
seed spread
not predictive uncertainty.
```

---

# 82. Prediction summary per seed

For each seed produce:

```text
prediction_mean
prediction_std
prediction_min
prediction_p05
prediction_q1
prediction_median
prediction_q3
prediction_p95
prediction_max
negative_count
compression ratios
change-magnitude ratio
direction agreement
```

No performance ranking.

---

# 83. Actual-series summary

Same relevant distribution/change descriptors for:

```text
y_true.
```

This enables direct comparison.

---

# 84. Seed-mean descriptive summary

May include:

```text
mean
std
range
```

but not official performance metrics.

---

# 85. Prediction-series fingerprint

For each seed prediction vector:

```text
SHA256 over target_id + full-precision y_pred
```

should match Phase47 bundle checksum context.

Phase48 does not rewrite source predictions.

---

# 86. Derived artifact provenance

Every Phase48 derived table should record:

```text
source prediction checksum(s)
source Test population checksum
analysis version
code version/hash if available.
```

---

# 87. No source overwrite

Never append columns directly to Phase47 frozen files.

Write new derived files under:

```text
artifacts/prediction_analysis/
```

---

# 88. Phase48 output directory

```text
artifacts/
└── prediction_analysis/
    ├── prediction_analysis_manifest.json
    ├── prediction_analysis_contract.json
    ├── phase48_preflight_audit.csv
    ├── prediction_source_verification.csv
    ├── prediction_alignment_audit.csv
    ├── prediction_wide_table.csv
    ├── prediction_long_table.csv
    ├── prediction_distribution_summary.csv
    ├── prediction_range_compression.csv
    ├── prediction_change_summary.csv
    ├── prediction_direction_agreement.csv
    ├── prediction_lag_diagnostics.csv
    ├── prediction_acf_diagnostics.csv
    ├── prediction_local_extrema_summary.csv
    ├── prediction_peak_timing_summary.csv
    ├── prediction_seed_pairwise_agreement.csv
    ├── prediction_seed_spread.csv
    ├── prediction_top_seed_disagreement.csv
    ├── prediction_rolling_tracking.csv
    ├── prediction_negative_value_audit.csv
    ├── prediction_saturation_audit.csv
    ├── prediction_baseline_context.csv
    ├── prediction_analysis_findings.csv
    ├── prediction_analysis_tests.csv
    ├── prediction_analysis_discrepancies.json
    ├── phase49_residual_analysis_handoff.json
    ├── phase50_error_regime_context_handoff.json
    ├── phase51_worst_error_context_handoff.json
    ├── prediction_analysis_summary.json
    ├── prediction_analysis_report.md
    ├── figures/
    │   ├── PRED_48_01_full_test_actual_vs_all_seeds.png
    │   ├── PRED_48_02_full_test_actual_vs_seed_mean_spread.png
    │   ├── PRED_48_03_first_24h_zoom.png
    │   ├── PRED_48_04_middle_24h_zoom.png
    │   ├── PRED_48_05_last_24h_zoom.png
    │   ├── PRED_48_06_scatter_seed42.png
    │   ├── PRED_48_07_scatter_seed123.png
    │   ├── PRED_48_08_scatter_seed2026.png
    │   ├── PRED_48_09_prediction_ecdf.png
    │   ├── PRED_48_10_change_magnitude_distribution.png
    │   ├── PRED_48_11_cross_seed_spread_over_time.png
    │   ├── PRED_48_12_pairwise_seed_prediction_scatter.png
    │   ├── PRED_48_13_lag_cross_correlation.png
    │   ├── PRED_48_14_acf_actual_vs_predictions.png
    │   ├── PRED_48_15_local_peak_capture.png
    │   ├── PRED_48_16_rolling_24h_mean_tracking.png
    │   ├── PRED_48_17_rolling_24h_std_tracking.png
    │   ├── PRED_48_18_daily_actual_heatmap.png
    │   ├── PRED_48_19_daily_seed_mean_heatmap.png
    │   └── PRED_48_20_seed_spread_heatmap.png
    ├── README_PREDICTION_ANALYSIS.md
    └── phase_48_signoff.json
```

Optional figures should only be generated when enough continuous Test data exist.

---

# 89. Required outputs

```text
O48.1  Analysis manifest
O48.2  Analysis contract
O48.3  Preflight audit
O48.4  Source verification
O48.5  Prediction alignment audit
O48.6  Wide prediction table
O48.7  Long prediction table
O48.8  Distribution summary
O48.9  Range/compression summary
O48.10 Change-magnitude summary
O48.11 Directional-change agreement
O48.12 Lag diagnostics
O48.13 Prediction ACF diagnostics
O48.14 Local extrema summary
O48.15 Peak timing summary
O48.16 Pairwise seed agreement
O48.17 Per-target seed spread
O48.18 Top seed-disagreement timestamps
O48.19 Rolling tracking diagnostics
O48.20 Negative-value audit
O48.21 Saturation audit
O48.22 Baseline context
O48.23 Core temporal figures
O48.24 Scatter figures
O48.25 Distribution figures
O48.26 Seed-spread figures
O48.27 Findings
O48.28 Phase49 handoff
O48.29 Phase50 context handoff
O48.30 Phase51 context handoff
O48.31 Tests
O48.32 Discrepancy log
O48.33 Summary JSON
O48.34 Human-readable report
O48.35 README
O48.36 Sign-off
```

---

# 90. Analysis manifest

`prediction_analysis_manifest.json`:

```text
phase=48
version=PREDICTION_ANALYSIS-v1
source_phase47_version
final_lock_sha256
test_population_sha256
source_prediction_files
source_prediction_sha256s
transformer_seeds=[42,123,2026]
new_inference=false
new_training=false
model_selection=false
seed_selection=false
ensemble=false
primary_scope=DESCRIPTIVE_PREDICTION_BEHAVIOR
status
created_at
```

---

# 91. Analysis contract

`prediction_analysis_contract.json` must state:

```text
Use only frozen Phase47 prediction bundles.

Analyze all three Transformer seeds.

Seed-mean prediction may be used only as descriptive central tendency.

Seed spread is not calibrated uncertainty.

No new inference.
No retraining.
No metric-driven seed/model selection.
No ensemble.
No prediction shifting/clipping/correction.

Formal residual statistics deferred to Phase49.
Formal Train-derived regime analysis deferred to Phase50.
Worst-error investigation deferred to Phase51.
Attention analysis deferred to Phase52+.
```

---

# 92. Preflight audit

`phase48_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Required checks:

```text
Phase47 PASS/PASS_WITH_WARNING
prediction bundles frozen
all 3 seed bundles exist
checksums match
same Test population
same target IDs/order/y_true
no rows missing
all prediction values finite
source files read-only
no inference path invoked
baseline files eligibility known.
```

---

# 93. Source verification table

`prediction_source_verification.csv`:

```text
source_id
file
expected_sha256
observed_sha256
row_count
population_sha256
frozen
verified
status
```

---

# 94. Prediction alignment audit

`prediction_alignment_audit.csv`:

```text
check
seed42
seed123
seed2026
expected
status
```

Checks:

```text
same N
same target IDs
same timestamps
same y_true
chronological
unique IDs
finite predictions.
```

---

# 95. Wide table schema

`prediction_wide_table.csv`:

```text
target_id
target_timestamp
y_true_wh
y_pred_seed42
y_pred_seed123
y_pred_seed2026
seed_mean_prediction
seed_std_prediction
seed_min_prediction
seed_max_prediction
seed_range_prediction
```

No residual columns required here.

---

# 96. Long table schema

`prediction_long_table.csv`:

```text
target_id
target_timestamp
y_true_wh
seed
y_pred_wh
population_sha256
source_prediction_sha256
```

---

# 97. Distribution summary schema

`prediction_distribution_summary.csv`:

```text
series_id
N
mean
std
min
p05
q1
median
q3
p95
max
iqr
status
```

Series:

```text
ACTUAL
SEED42
SEED123
SEED2026
SEED_MEAN_DESCRIPTIVE.
```

---

# 98. Range/compression schema

`prediction_range_compression.csv`:

```text
seed
pred_std
true_std
std_ratio
pred_iqr
true_iqr
iqr_ratio
pred_range
true_range
range_ratio
mean_shift_pred_minus_true
status
```

---

# 99. Change-summary schema

`prediction_change_summary.csv`:

```text
series_id
valid_adjacent_count
mean_abs_delta
median_abs_delta
p90_abs_delta
p95_abs_delta
std_delta
status
```

---

# 100. Direction agreement schema

`prediction_direction_agreement.csv`:

```text
seed
valid_adjacent_count
exact_three_class_agreement_count
exact_three_class_agreement_rate
actual_nonzero_count
nonzero_direction_agreement_count
nonzero_direction_agreement_rate
status
```

---

# 101. Lag diagnostics schema

`prediction_lag_diagnostics.csv`:

```text
seed
lag_steps
lag_minutes
correlation
valid_pair_count
lag_convention
status
```

Lags:

```text
-6 ... +6.
```

---

# 102. Prediction ACF schema

`prediction_acf_diagnostics.csv`:

```text
series_id
lag_steps
lag_minutes
acf
valid_pair_count
status
```

Registered lags:

```text
1,6,12,36,72,144
```

when supported.

---

# 103. Local extrema summary

`prediction_local_extrema_summary.csv`:

```text
seed
true_local_max_count
actual_peak_mean
actual_peak_median
pred_at_peak_mean
pred_at_peak_median
peak_level_ratio
true_local_min_count
actual_trough_mean
actual_trough_median
pred_at_trough_mean
pred_at_trough_median
status
```

---

# 104. Peak timing schema

`prediction_peak_timing_summary.csv`:

```text
seed
eligible_true_peaks
same_step_pred_peak_count
same_step_rate
within_plus_minus_1_step_count
within_plus_minus_1_step_rate
status
```

---

# 105. Pairwise seed agreement schema

`prediction_seed_pairwise_agreement.csv`:

```text
seed_a
seed_b
pearson_prediction_corr
spearman_prediction_corr
mean_abs_prediction_difference
rmse_between_predictions
max_abs_prediction_difference
status
```

---

# 106. Seed-spread schema

`prediction_seed_spread.csv`:

```text
target_id
timestamp
y_true_wh
seed_mean_prediction
seed_std_prediction
seed_min_prediction
seed_max_prediction
seed_range_prediction
status
```

---

# 107. Top seed-disagreement schema

`prediction_top_seed_disagreement.csv`:

```text
rank
target_id
timestamp
y_true_wh
seed42
seed123
seed2026
seed_mean
seed_std
seed_range
status
```

Top:

```text
K=20
```

by descending seed range.

---

# 108. Rolling tracking schema

`prediction_rolling_tracking.csv` may include:

```text
timestamp
seed
rolling_24h_true_mean
rolling_24h_pred_mean
rolling_24h_true_std
rolling_24h_pred_std
rolling_24h_corr
window_valid_count
status
```

Only where full contiguous 144-point window exists.

---

# 109. Negative prediction audit

`prediction_negative_value_audit.csv`:

```text
seed
negative_count
negative_fraction
minimum_prediction
first_negative_timestamp
status
```

If none:

```text
count=0.
```

---

# 110. Saturation audit

`prediction_saturation_audit.csv`:

```text
seed
unique_prediction_count
fraction_at_exact_min
fraction_at_exact_max
duplicate_rate_optional
suspected_saturation
reason
status
```

No arbitrary saturation threshold unless predeclared.

Prefer descriptive fields and qualitative finding.

---

# 111. Baseline context schema

`prediction_baseline_context.csv`:

```text
model_id
prediction_bundle_available
common_population_verified
prediction_mean
prediction_std
prediction_min
prediction_max
change_mean_abs_delta
notes
status
```

Baselines only.

---

# 112. Core Figure PRED_48_01

`full_test_actual_vs_all_seeds`

Purpose:

```text
show full trajectory
show seed overlap/divergence
show peak/trough tracking.
```

No baseline clutter in this main figure.

---

# 113. Core Figure PRED_48_02

`actual_vs_seed_mean_spread`

Plot:

```text
actual
seed mean
min–max seed band
```

or:

```text
mean ± one sample SD band
```

Preferred:

```text
min–max band
```

because n=3.

Label:

```text
cross-seed spread, not confidence interval.
```

---

# 114. Zoom figures

Use deterministic:

```text
first 24h
middle 24h
last 24h.
```

Same y-axis policy where reasonable.

---

# 115. Scatter figures

Three separate figures:

```text
seed42
seed123
seed2026
```

with common axis limits and `y=x`.

Do not combine into unreadable dense plot.

---

# 116. ECDF figure

Show:

```text
actual
3 seed predictions
```

Useful to see range compression and distribution shift.

---

# 117. Change magnitude figure

Use ECDF/histogram of:

```text
|Δ actual|
|Δ prediction|.
```

No residual.

---

# 118. Cross-seed spread figure

Plot per-target:

```text
seed_std or seed_range
```

over time.

No error color coding by default.

---

# 119. Pairwise seed prediction scatter

Could be a 3-panel figure outside artifact requirement if implementation permits separate plots.

If plotting constraints require separate images:

```text
42 vs 123
42 vs 2026
123 vs 2026.
```

Keep naming consistent.

---

# 120. Lag correlation figure

For each seed show correlation over:

```text
-6..+6 steps.
```

Mark lag 0.

Do not mark “optimal correction lag”.

---

# 121. ACF figure

Actual and predictions at registered lags.

This is not residual ACF.

---

# 122. Local peak capture figure

Use deterministic true local maxima from definition.

Plot:

```text
actual peak value
predicted value at same timestamp
```

for all peaks, or summary.

Avoid choosing only large peaks unless Train-defined threshold enters Phase50.

---

# 123. Rolling 24h mean figure

Shows slowly varying level.

No prediction correction.

---

# 124. Rolling 24h std figure

Shows whether model suppresses within-day variability.

---

# 125. Daily heatmaps

Only if enough Test days and calendar matrix not excessively sparse.

If incomplete day:

```text
leave missing cell
```

do not interpolate.

---

# 126. Heatmap value scales

Actual and seed-mean prediction heatmaps should use the same color scale for visual comparability.

---

# 127. Prediction-analysis findings

Possible findings codes:

```text
PREDICTIONS_TRACK_OVERALL_LEVEL
PREDICTIONS_SHOW_RANGE_COMPRESSION
PREDICTIONS_SHOW_OVERDISPERSION
PREDICTIONS_SMOOTHER_THAN_ACTUAL
PREDICTIONS_MORE_VOLATILE_THAN_ACTUAL
PEAK_AMPLITUDE_COMPRESSION
TROUGH_OVERPREDICTION_PATTERN
APPARENT_TEMPORAL_LAG
NO_CLEAR_TEMPORAL_LAG
HIGH_SEED_AGREEMENT
VISIBLE_SEED_DISAGREEMENT
SEED_DISAGREEMENT_CLUSTERED_IN_TIME
NEGATIVE_PREDICTIONS_PRESENT
NO_NEGATIVE_PREDICTIONS
PREDICTION_SATURATION_SUSPECTED
NO_SATURATION_SIGNAL
DIRECTIONAL_CHANGE_ALIGNMENT
DAILY_PATTERN_TRACKING
PERSISTENCE_ONE_STEP_LAG_VISIBLE
LSTM_BEHAVIOR_CONTEXT
NO_NEW_INFERENCE
NO_MODEL_SELECTION
NO_ENSEMBLE
SOURCE_BUNDLES_VERIFIED
```

Do not attach arbitrary “high/low” thresholds unless exact values are reported.

---

# 128. Findings language must be descriptive

Safe:

> Across all three seeds, the predicted series was less variable than the observed Test series, as reflected by prediction-to-target standard-deviation ratios below 1.

Safe:

> The three seed trajectories were strongly aligned, although seed spread increased during several periods of rapid target variation.

Unsafe:

> The model is statistically calibrated.

Unsafe:

> The model always misses peaks.

unless every relevant peak supports that statement.

---

# 129. No causal claims

Do not say:

```text
temperature caused the prediction miss
occupancy caused the peak
weather caused lag.
```

Phase48 does not establish causes.

---

# 130. No attention explanation

Do not use:

```text
the model predicts this because head 2 attends to...
```

until Phase52+.

---

# 131. No residual diagnostics creep

Do not compute:

```text
residual histogram
Q-Q plot
residual ACF
Ljung-Box
heteroskedasticity
```

in Phase48.

Those belong Phase49.

---

# 132. No formal error regime analysis creep

Do not produce:

```text
low/medium/high target RMSE
hourly RMSE
weekend RMSE
high-spike RMSE
```

in Phase48.

Those belong Phase50.

---

# 133. No worst-error ranking creep

Do not rank by:

```text
absolute error
squared error
```

in Phase48.

Top seed disagreement is allowed because it ranks stochastic disagreement, not prediction error.

Worst-error ranking belongs Phase51.

---

# 134. No model-quality ranking from prediction diagnostics

Even if seed42 shows:

```text
better peak timing
```

do not promote seed42.

All seeds remain final official runs.

---

# 135. Optional seed-consensus classification

For each timestamp:

```text
all 3 above truth
all 3 below truth
mixed
```

This is already residual-sign-like.

Prefer defer to Phase49.

Not part of required Phase48 scope.

---

# 136. No Test-driven new feature hypotheses for tuning

You may write discussion hypotheses for future research, but not alter current model.

Use wording:

```text
future work could investigate...
```

not:

```text
we should retrain now.
```

---

# 137. Phase48 discrepancies

`prediction_analysis_discrepancies.json` taxonomy:

```text
PHASE47_NOT_APPROVED
SOURCE_PREDICTION_FILE_MISSING
SOURCE_PREDICTION_CHECKSUM_MISMATCH
TEST_POPULATION_MISMATCH
TARGET_ID_DUPLICATE
TARGET_ORDER_MISMATCH
YTRUE_MISMATCH_ACROSS_SEEDS
NONFINITE_PREDICTION
SOURCE_FILE_MODIFIED
NEW_TEST_INFERENCE_ATTEMPT
CHECKPOINT_LOADING_ATTEMPT
MODEL_TRAINING_ATTEMPT
BEST_SEED_SELECTION_ATTEMPT
ENSEMBLE_METRIC_ATTEMPT
PREDICTION_SHIFT_CORRECTION_ATTEMPT
POST_HOC_CLIPPING_ATTEMPT
TEST_THRESHOLD_TUNING_ATTEMPT
ZOOM_CHERRY_PICK_ATTEMPT
GAP_UNSAFE_DIFFERENCING
LAG_CONVENTION_UNSPECIFIED
LAG_RANGE_DRIFT
PEAK_DEFINITION_DRIFT
PEAK_TIMING_WINDOW_DRIFT
SEED_SPREAD_MISLABELED_AS_CONFIDENCE_INTERVAL
RESIDUAL_ANALYSIS_SCOPE_CREEP
ERROR_REGIME_SCOPE_CREEP
WORST_ERROR_SCOPE_CREEP
ATTENTION_SCOPE_CREEP
BASELINE_POPULATION_MISMATCH
OTHER
```

---

# 138. Status model

## PASS

```text
all Phase47 prediction bundles verified
all three seeds analyzed
no inference/training
core temporal/distribution/seed analyses complete
derived artifacts reproducible
findings documented
Phase49 handoff ready.
```

## PASS_WITH_WARNING

Possible:

```text
optional daily heatmap unavailable
Test period too short for lag144 ACF
baseline prediction bundle unavailable
high seed spread observed
negative predictions observed.
```

These are not methodological failures unless artifact integrity is broken.

## FAIL

Examples:

```text
source checksum mismatch
target misalignment
new Test inference
best-seed selection
post-hoc correction
scope leakage into retraining.
```

---

# 139. Phase48 execution sequence

```text
1. Verify Phase47 signoff.
2. Verify frozen prediction checksums.
3. Build aligned wide/long tables.
4. Freeze prediction-analysis contract.
5. Create deterministic zoom definitions.
6. Compute prediction distribution summaries.
7. Compute range/compression descriptors.
8. Compute gap-safe first-difference descriptors.
9. Compute directional-change agreement.
10. Compute fixed lag diagnostics.
11. Compute prediction ACF diagnostics.
12. Detect fixed-definition local extrema.
13. Compute peak/trough capture descriptors.
14. Compute peak timing ±1-step descriptor.
15. Compute cross-seed pairwise agreement.
16. Compute per-target seed spread.
17. List top20 seed-disagreement timestamps.
18. Compute optional rolling-24h tracking.
19. Audit negative values/saturation.
20. Build baseline context if available.
21. Generate figures.
22. Write findings.
23. Write Phase49/50/51 handoffs.
24. Run acceptance tests/discrepancy checks.
25. Write report/README/signoff.
```

---

# 140. Preflight acceptance checklist

```text
[ ] Phase47 PASS/PASS_WITH_WARNING.
[ ] Phase48 handoff exists.
[ ] Seed42 prediction file exists.
[ ] Seed123 prediction file exists.
[ ] Seed2026 prediction file exists.
[ ] All checksums match Phase47.
[ ] All files read-only/frozen.
[ ] Same Test population fingerprint.
[ ] Same N.
[ ] Same target IDs.
[ ] Same timestamps.
[ ] Same y_true.
[ ] All predictions finite.
[ ] Persistence eligibility known.
[ ] LSTM eligibility known.
[ ] No checkpoint loading required.
[ ] No Test inference code invoked.
```

---

# 141. Core-analysis acceptance checklist

```text
[ ] Wide table generated.
[ ] Long table generated.
[ ] Seed mean marked descriptive only.
[ ] Seed spread marked non-calibrated.
[ ] Full Test temporal plot generated.
[ ] Deterministic first/middle/last zooms used.
[ ] All-seed scatter figures use same axes.
[ ] Prediction ECDF generated.
[ ] Distribution summary generated.
[ ] Std/IQR compression generated.
[ ] Gap-safe first differences generated.
[ ] Change-magnitude ratio generated.
[ ] Direction agreement generated.
[ ] Lag range fixed -6..+6.
[ ] Lag sign convention documented.
[ ] ACF only at registered lags.
[ ] Local extrema definition fixed.
[ ] Peak timing window fixed ±1 step.
[ ] Pairwise seed agreement generated.
[ ] Per-target seed spread generated.
[ ] Top20 disagreement timestamps generated.
[ ] Negative prediction audit generated.
[ ] Saturation audit generated.
```

---

# 142. Scope acceptance checklist

```text
[ ] No residual histogram.
[ ] No residual ACF.
[ ] No residual normality test.
[ ] No regime-specific RMSE.
[ ] No hour/weekend error table.
[ ] No worst absolute-error ranking.
[ ] No attention map.
[ ] No model explanation claim.
[ ] No Test-derived retuning threshold.
[ ] No best-seed selection.
[ ] No seed ensemble performance.
[ ] No prediction clipping.
[ ] No time-shift correction.
[ ] No new Test prediction.
```

---

# 143. Visualization acceptance checklist

```text
[ ] Full Test plot uses chronological order.
[ ] Seed spread label says non-confidence interval.
[ ] Zoom windows deterministic.
[ ] Common axes across seed scatter plots.
[ ] y=x line included in scatter.
[ ] Actual/prediction heatmaps share color scale if generated.
[ ] Plot titles say Held-Out Test.
[ ] No misleading truncated axes.
[ ] No cherry-picked windows.
```

---

# 144. Provenance acceptance checklist

```text
[ ] Source prediction SHA256 stored.
[ ] Test population SHA256 stored.
[ ] Derived files do not overwrite Phase47 artifacts.
[ ] Analysis version stored.
[ ] Every table/figure traceable to source bundles.
[ ] Findings reference derived artifacts.
[ ] Phase49 handoff references exact source checksums.
```

---

# 145. Acceptance criteria

Phase48 PASS only when:

```text
The three frozen Phase47 Transformer prediction bundles are verified without rerunning Test inference.

All three seeds are aligned on the exact same Test target IDs, timestamps and y_true values.

Prediction behavior is analyzed using full Test chronology and deterministic zoom windows.

Actual and predicted distributions are summarized without modifying the predictions.

Prediction range/variance compression is quantified descriptively.

Temporal change magnitude and directional tracking are computed using gap-safe adjacent timestamps.

Any lag diagnostic uses the predeclared -6 to +6 step range and is not used to shift predictions.

Local extrema analysis uses a fixed definition and does not replace the Train-derived regime analysis planned for Phase50.

Cross-seed agreement and per-target seed spread are quantified without declaring a best seed.

Seed-mean predictions, if used, are labeled descriptive and are not treated as an ensemble.

Negative/saturation behavior is reported without post-hoc clipping.

Prediction-analysis artifacts are derived from immutable Phase47 bundles and fully checksummed/provenanced.

No residual, regime, worst-error or attention analysis is improperly substituted for later phases.

No model selection, retraining, Test-driven correction or new inference occurs.

Phase49 handoff is generated.
```

---

# 146. Failure conditions

Phase48 FAIL if:

```text
prediction bundle checksum mismatch

three seeds do not share Test population

y_true differs across bundles

a source Test file is modified

new Test inference is run

a seed is chosen as representative because of Test performance

seed-mean prediction is promoted to ensemble

predictions are shifted to improve alignment

negative predictions are clipped

Test-derived thresholds are tuned

zoom windows are cherry-picked after seeing errors

gap-crossing differences are computed

lag convention is ambiguous

peak definition changes after seeing results

residual/error-regime/worst-error analysis is used to alter model decisions.
```

---

# 147. Common mistakes

## 147.1 Vẽ seed42 đẹp nhất rồi chỉ phân tích seed42

Sai. Phải giữ cả ba seeds.

## 147.2 Trung bình predictions rồi tính RMSE và gọi là final ensemble

Sai pre-Test contract.

## 147.3 Thấy model trễ 10 phút rồi shift prediction một step

Sai post-hoc correction.

## 147.4 Thấy prediction âm rồi set bằng 0

Sai.

## 147.5 Chọn một ngày model lỗi nhất làm zoom chính

Sai; Phase48 zoom phải deterministic.

## 147.6 Dùng Test p90 để định nghĩa high-energy regime

Formal regime threshold thuộc Phase50 và phải Train-derived.

## 147.7 Dùng top absolute error trong Phase48

Để Phase51.

## 147.8 Tính residual ACF luôn

Để Phase49.

## 147.9 Gọi min–max 3 seeds là confidence interval

Sai.

## 147.10 Từ correlation lag suy ra causality

Sai.

---

# 148. Prediction-analysis narrative template

Recommended report narrative structure:

```text
1. Source integrity
2. Full Test trajectory tracking
3. Prediction distribution and range
4. Prediction smoothness/change behavior
5. Temporal alignment diagnostics
6. Peak/trough capture
7. Cross-seed agreement
8. Cross-seed disagreement periods
9. Optional daily/rolling pattern tracking
10. Baseline behavioral context
11. Findings to investigate in Phase49–51
12. Limitations
```

---

# 149. Findings to hand off to Phase49

Phase49 needs signals such as:

```text
prediction variance compression observed/not observed
apparent bias direction
periods of elevated seed spread
apparent temporal lag
peak compression
negative predictions
```

But Phase49 should recompute residual diagnostics from source bundles, not trust qualitative statements alone.

---

# 150. Phase49 handoff schema

`phase49_residual_analysis_handoff.json`:

```text
source_phase48_version
final_lock_sha256
test_population_sha256
source_prediction_files
source_prediction_sha256s
wide_table_path
seed_list=[42,123,2026]
residual_convention=y_true-y_pred
prediction_behavior_findings
no_best_seed=true
no_ensemble=true
ready_for_phase49=true
```

---

# 151. Phase50 context handoff

`phase50_error_regime_context_handoff.json`:

```text
source prediction bundle refs
test population
timestamps
Train-derived threshold requirement
prediction distribution findings
change-behavior findings
seed spread refs
no Test-derived threshold authorization
ready_for_phase50_context=true
```

Phase50 will source actual regime thresholds from Train artifacts.

---

# 152. Phase51 context handoff

`phase51_worst_error_context_handoff.json`:

```text
source prediction bundles
target IDs
timestamps
seed spread table
local extrema table
no worst-error ranking performed in Phase48
ready_for_phase51_context=true
```

---

# 153. Human-readable report

`prediction_analysis_report.md`:

```text
1. Objective
2. Frozen prediction sources
3. Analysis boundaries
4. Test prediction alignment
5. Full temporal trajectory
6. Deterministic zoom views
7. Actual-vs-predicted scatter
8. Prediction distribution
9. Range and variance compression
10. Change-magnitude behavior
11. Directional-change tracking
12. Temporal lag diagnostic
13. Prediction autocorrelation
14. Peak/trough capture
15. Cross-seed agreement
16. Cross-seed spread
17. Optional rolling/daily pattern tracking
18. Baseline prediction behavior
19. Key findings
20. Handoff to residual/regime/worst-error analysis
21. Limitations
22. Definition of Done
```

---

# 154. README requirements

`README_PREDICTION_ANALYSIS.md` explains:

```text
why no new inference
why all three seeds are analyzed
what seed mean means
why seed spread is not confidence interval
why zoom windows are deterministic
why lag analysis does not shift predictions
why residual analysis is deferred
why regime thresholds are not Test-derived
why worst-error ranking is deferred
how Phase49 consumes artifacts.
```

---

# 155. Summary artifact

`prediction_analysis_summary.json`:

```text
version
source_phase47_version
final_lock_sha256
test_population_sha256
seed_list
source_prediction_sha256s
N_test
distribution_summary
compression_summary
change_summary
direction_agreement_summary
lag_summary
acf_summary
peak_summary
seed_agreement_summary
seed_spread_summary
negative_prediction_summary
saturation_summary
baseline_context
findings
new_inference=false
best_seed_selected=false
ensemble_used=false
predictions_modified=false
phase49_ready
overall_status
```

---

# 156. Phase48 sign-off

`phase_48_signoff.json` minimum:

```text
phase=48
phase_name=Prediction analysis
version=PREDICTION_ANALYSIS-v1
source_phase47_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
seed42_prediction_sha256
seed123_prediction_sha256
seed2026_prediction_sha256
source_bundles_verified
aligned_target_population
distribution_analysis_complete
change_analysis_complete
lag_analysis_complete
peak_analysis_complete
seed_agreement_complete
seed_spread_complete
negative_value_audit_complete
new_inference=false
model_training=false
best_seed_selected=false
ensemble_used=false
prediction_shift_applied=false
prediction_clipping_applied=false
source_predictions_modified=false
phase49_ready
warnings
overall_status
created_at
```

---

# 157. Recommended execution pseudocode

```text
p47 = load_phase47_signoff()
assert p47.overall_status in {"PASS","PASS_WITH_WARNING"}

handoff = load_phase48_handoff()

sources = load_frozen_prediction_bundles(
    seeds=[42,123,2026]
)

verify_checksums(sources)
verify_same_population_ids_timestamps_ytrue(sources)
assert all_predictions_finite(sources)

wide = align_to_wide_table(sources)
long = build_long_table(sources)

wide["seed_mean_prediction"] = mean(
    wide[["seed42","seed123","seed2026"]],
    axis=1
)

wide["seed_std_prediction"] = sample_std(
    wide[["seed42","seed123","seed2026"]],
    ddof=1
)

wide["seed_min_prediction"] = row_min(...)
wide["seed_max_prediction"] = row_max(...)
wide["seed_range_prediction"] = (
    wide["seed_max_prediction"]
    - wide["seed_min_prediction"]
)

distribution = summarize_distribution(
    actual=wide.y_true,
    predictions=all_seed_columns
)

compression = compute_prediction_range_compression(
    actual=wide.y_true,
    predictions=all_seed_columns
)

adjacent_mask = exact_10min_continuity_mask(
    wide.target_timestamp
)

changes = compute_gap_safe_first_difference_stats(
    wide,
    adjacent_mask
)

direction = compute_direction_agreement(
    wide,
    adjacent_mask,
    zero_policy="EXACT_ZERO"
)

lag_diag = compute_gap_safe_lag_correlations(
    wide,
    lags=range(-6,7)
)

acf_diag = compute_prediction_acf(
    wide,
    lags=[1,6,12,36,72,144]
)

extrema = detect_true_local_extrema(
    y=wide.y_true,
    continuity=adjacent_mask,
    fixed_definition=True
)

peak_summary = compare_predictions_at_extrema(
    wide,
    extrema
)

peak_timing = compute_peak_timing(
    wide,
    extrema,
    window_steps=1
)

seed_pairwise = compute_seed_agreement(
    wide.seed42,
    wide.seed123,
    wide.seed2026
)

seed_spread = build_seed_spread_table(wide)

top_disagreement = top_k(
    seed_spread,
    by="seed_range_prediction",
    k=20
)

rolling = compute_optional_rolling_tracking(
    wide,
    window=144,
    require_full_continuity=True
)

negative_audit = audit_negative_predictions(wide)
saturation_audit = audit_prediction_saturation(wide)

write_all_tables()
generate_deterministic_figures(
    zooms=["first24h","middle24h","last24h"]
)

write_findings()
write_phase49_handoff()
write_phase50_context_handoff()
write_phase51_context_handoff()

assert no_new_test_inference
assert no_best_seed_selection
assert no_ensemble_metric
assert no_prediction_modification

signoff_phase48()
```

---

# 158. Definition of Done

\[
\boxed{
Frozen\ Phase47\ Predictions
+
All\ Three\ Seeds
+
Temporal\ Tracking
+
Distribution\ Analysis
+
Change\ Dynamics
+
Lag\ Diagnostics
+
Peak/Trough\ Tracking
+
Cross\text{-}Seed\ Agreement
+
Seed\ Spread
+
No\ New\ Inference
+
No\ Selection
+
No\ Correction
+
Phase49\ Handoff
}
\]

---

# 159. Final status contract

```text
PHASE 48 analyzes predictions only.

Source:
frozen Phase47 prediction bundles.

Seeds:
42
123
2026
all retained.

Allowed:
temporal plots
scatter
distribution summaries
range compression
gap-safe first differences
direction agreement
fixed lag diagnostics
prediction ACF
fixed local extrema
peak timing ±1 step
cross-seed agreement
seed spread
deterministic zooms
rolling descriptive tracking.

Seed mean:
descriptive only
not ensemble.

Seed spread:
descriptive stochastic spread
not confidence interval.

Forbidden:
new Test inference
checkpoint reload for new predictions
best-seed selection
ensemble performance
prediction shift
prediction clipping
retuning
Test-derived regime tuning
residual analysis creep
worst-error ranking
attention analysis.

Outputs:
immutable derived tables
prediction-behavior figures
findings
Phase49/50/51 handoffs.

After PREDICTION_ANALYSIS-v1 PASS:
proceed to
PHASE 49 — Residual Analysis.
```

---

# 160. Final check

Correct:

```text
verify Phase47 prediction bundles
→ align all 3 seeds
→ analyze trajectory/distribution/change behavior
→ analyze lag/peaks
→ analyze seed agreement/spread
→ deterministic figures
→ no prediction modification
→ Phase49 handoff
```

Incorrect:

```text
load checkpoints
→ rerun Test
```

Incorrect:

```text
average seed predictions
→ calculate new ensemble RMSE
→ call it final model
```

Incorrect:

```text
shift predictions by best correlation lag
→ recompute RMSE
```

Incorrect:

```text
pick worst Test periods
→ tune model again
```

Chỉ sau khi:

```text
PREDICTION_ANALYSIS-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase49_ready = true
```

mới chuyển sang **PHASE 49 — Residual Analysis**.
