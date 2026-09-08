# PHASE 49 — RESIDUAL ANALYSIS

## Kế hoạch phân tích residual của ba Final Transformer seeds trên Held-Out Test sau `PREDICTION_ANALYSIS-v1`

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Residual convention:** `residual = y_true - y_pred`  
**Upstream prediction analysis:** `PREDICTION_ANALYSIS-v1`  
**Primary source of truth:** frozen Phase47 Test prediction bundles  
**Phase ID:** `PHASE_49_RESIDUAL_ANALYSIS`  
**Output version:** `RESIDUAL_ANALYSIS-v1`  
**Phase trước:** `Phase_48_Prediction_analysis.md`  
**Phase sau:** `Phase_50_Error-by-regime_analysis.md`

---

# 1. Vai trò của Phase 49

Phase 49 phân tích **cấu trúc sai số còn lại** của Final Transformer sau khi prediction bundles đã được tạo ở Phase47 và hành vi dự báo tổng quát đã được mô tả ở Phase48.

Mục tiêu không phải tiếp tục tối ưu model.

Mục tiêu là trả lời:

```text
1. Residual có centered quanh 0 hay còn signed bias?
2. Residual distribution có đối xứng hay lệch?
3. Residual có heavy tail / extreme tail không?
4. Error có còn temporal dependence/autocorrelation?
5. Model có tạo các chuỗi underprediction hoặc overprediction kéo dài không?
6. Độ lớn residual có thay đổi theo predicted level không?
7. Residual variance có ổn định theo thời gian hay có clustering?
8. Ba seeds có mắc lỗi theo cùng một hướng/timestamp không?
9. Các diagnostics nào cần chuyển sang Phase50 regime analysis?
10. Những error structures nào cần chuyển sang Phase51 worst-error analysis?
```

Nguyên tắc trung tâm:

\[
\boxed{
Frozen\ Predictions
+
Residual=y-\hat y
+
All\ Three\ Seeds
+
Distribution
+
Bias
+
Temporal\ Dependence
+
Variance\ Structure
+
No\ Retuning
}
\]

---

# 2. Residual convention bị khóa

Canonical:

\[
e_{s,t}
=
y_t-\hat y_{s,t}
\]

với seed `s`.

Interpretation:

```text
e > 0  → model underpredicts
e < 0  → model overpredicts
e = 0  → exact prediction.
```

Không được đổi dấu residual giữa các bảng/hình.

---

# 3. Scope của Phase49

Phase49 tập trung vào:

```text
residual distribution
signed bias
residual quantiles
tail shape
under/overprediction balance
temporal residual structure
gap-safe residual autocorrelation
sign persistence / sign runs
rolling residual mean
rolling residual variability
residual-vs-prediction structure
absolute residual-vs-prediction structure
cross-seed residual agreement
cross-seed residual sign consensus
baseline residual context
```

---

# 4. Những gì không thuộc Phase49

Không thực hiện:

```text
formal target-regime comparison      → Phase50
hour/weekend/high-demand RMSE tables → Phase50
worst-error case ranking             → Phase51
individual failure-case narratives   → Phase51
attention explanation                → Phase52–57
model retraining                     → forbidden
best-seed selection                  → forbidden
```

---

# 5. Source of truth

Primary frozen files:

```text
final_test_predictions_seed42.csv
final_test_predictions_seed123.csv
final_test_predictions_seed2026.csv
```

from:

```text
artifacts/final_test/predictions/
```

Required provenance:

```text
phase_47_signoff.json
prediction_checksums.json
final_test_population_manifest.json
phase49_residual_analysis_handoff.json
```

Phase48 artifacts are contextual/derived support, not a replacement for Phase47 prediction sources.

---

# 6. Why recompute residual from Phase47 predictions

Even if Phase47 files already contain:

```text
residual_wh
absolute_error_wh
squared_error_wh2
```

Phase49 must independently recompute:

\[
e=y-\hat y
\]

from frozen:

```text
y_true_wh
y_pred_wh
```

and verify exact/numerical agreement with stored residual fields.

This creates:

```text
RESIDUAL_RECOMPUTE_AUDIT.
```

---

# 7. No new Test inference

Hard:

```text
model loading for new Test prediction = forbidden
new Test inference                  = forbidden
training                            = forbidden.
```

If prediction bundle missing/corrupt:

```text
STOP
```

and resolve source artifact integrity.

Do not regenerate Test predictions in Phase49.

---

# 8. Source prediction files remain immutable

Do not append residual diagnostics directly into Phase47 CSVs.

Write all derived files under:

```text
artifacts/residual_analysis/
```

---

# 9. Three seeds remain scientifically symmetric

Analyze:

```text
seed42
seed123
seed2026
```

independently and jointly.

No:

```text
best residual seed
representative seed by Test error
seed deletion.
```

---

# 10. No pooled 3N pseudo-sample as primary analysis

Do not concatenate residuals from all three seeds and treat:

```text
3N
```

as independent Test observations.

The same Test targets appear three times.

Primary residual diagnostics are:

```text
per-seed
+
cross-seed summary of per-seed statistics.
```

A long table containing all seeds is acceptable for storage/plotting, but not for iid inferential sample-size claims.

---

# 11. Cross-seed residual center

For timestamp `t`, one may compute:

\[
\bar e_t
=
\frac{e_{42,t}+e_{123,t}+e_{2026,t}}{3}
\]

only as:

```text
SEED_MEAN_RESIDUAL_DESCRIPTIVE.
```

It is mathematically linked to seed-mean prediction and must not be interpreted as residual of an official ensemble.

---

# 12. Core research questions

Phase49 must answer:

```text
Q49.1  Is mean residual close to zero for each seed?
Q49.2  Is median residual close to zero?
Q49.3  What fraction of Test targets are underpredicted vs overpredicted?
Q49.4  Is residual distribution skewed?
Q49.5  Are residual tails heavy?
Q49.6  Are large positive/negative residuals asymmetric?
Q49.7  Are residuals serially autocorrelated?
Q49.8  Are there repeated runs of same-sign residuals?
Q49.9  Does residual variability change over Test time?
Q49.10 Does error magnitude increase with predicted level?
Q49.11 Do seeds make errors in similar directions at the same timestamps?
Q49.12 Which structures should be investigated in Phase50/51?
```

---

# 13. Canonical residual tables

Create:

## Long table

```text
target_id
target_timestamp
seed
y_true_wh
y_pred_wh
residual_wh
absolute_error_wh
squared_error_wh2
residual_sign
source_prediction_sha256
```

## Wide table

```text
target_id
target_timestamp
y_true_wh
residual_seed42
residual_seed123
residual_seed2026
abs_error_seed42
abs_error_seed123
abs_error_seed2026
residual_sign_seed42
residual_sign_seed123
residual_sign_seed2026
```

---

# 14. Residual sign categories

Lock:

```text
POSITIVE = residual > 0 = underprediction
ZERO     = residual == 0
NEGATIVE = residual < 0 = overprediction.
```

No arbitrary epsilon.

If floating-point exact zero is rare:

```text
report exact-zero frequency honestly.
```

---

# 15. Residual integrity checks

Hard:

```text
residual == y_true - y_pred
absolute_error == abs(residual)
squared_error == residual**2
all finite
same N all seeds
same target IDs
same timestamps
same y_true.
```

---

# 16. Metric reconstruction consistency

Using residuals, verify:

\[
MAE
=
Mean(|e|)
\]

\[
RMSE
=
\sqrt{Mean(e^2)}
\]

must match Phase47 per-seed metrics at full numerical tolerance.

Any mismatch:

```text
STOP
METRIC_RESIDUAL_INCONSISTENCY.
```

---

# 17. Distribution summary

For each seed compute:

```text
N
mean residual
median residual
std residual
MAD
min
p01
p05
p10
Q1
Q3
p90
p95
p99
max
IQR
skewness
excess kurtosis
mean absolute residual
RMSE from residual.
```

---

# 18. MAD definition

Use raw median absolute deviation around the residual median:

\[
MAD
=
median(|e-median(e)|)
\]

No normal-consistency scaling unless separately labeled.

---

# 19. Skewness definition

Use one consistent implementation and record:

```text
bias correction setting.
```

Recommended:

```text
sample skewness with bias=False
```

where library semantics are verified.

Do not mix definitions across seeds.

---

# 20. Kurtosis definition

Use:

```text
Fisher excess kurtosis
normal reference = 0
bias=False
```

and record implementation.

---

# 21. Residual distribution shape is diagnostic

Residual normality is **not** a required property for a good forecasting model.

Do not define Phase49 PASS/FAIL from:

```text
normality.
```

---

# 22. ECDF over histogram as primary distribution visual

Recommended:

```text
ECDF of residuals for all three seeds
```

with vertical line:

```text
residual=0.
```

This avoids sensitivity to histogram bins.

---

# 23. Common-bin histogram

Also useful for visual tails.

Lock:

```text
number of bins = 50
```

with one common residual range across all seeds:

```text
[min residual across all 3 seeds,
 max residual across all 3 seeds].
```

This is visualization only.

---

# 24. Histogram must not create scientific thresholds

Bin boundaries are not regimes.

Do not reuse them in Phase50.

---

# 25. Q-Q plots

Create one Q-Q plot per seed against a Gaussian reference as a **shape diagnostic**.

Purpose:

```text
visualize center/tail departure
```

not to require normality.

---

# 26. Optional normality test

Formal tests such as:

```text
Jarque–Bera
```

may be included only as secondary descriptive diagnostics.

Recommended default:

```text
Q-Q + skewness + kurtosis
```

without making normality p-values central.

Why:

```text
large time-series samples often reject normality for trivial deviations
and residual observations are temporally dependent.
```

---

# 27. Signed bias metrics

For each seed:

## Mean Bias Error

\[
MBE_s=Mean(e_s)
\]

Positive:

```text
average underprediction.
```

Negative:

```text
average overprediction.
```

---

# 28. Normalized Mean Bias Error

Optional descriptive:

\[
NMBE_s
=
100
\times
\frac{Mean(e_s)}
{Mean(y)}
\]

Because `Appliances` is positive in this dataset.

Label:

```text
descriptive percentage of mean true load.
```

Not a primary evaluation metric.

---

# 29. Median signed error

\[
MedBias_s=Median(e_s)
\]

Useful when residual distribution is skewed/heavy-tailed.

---

# 30. Underprediction fraction

\[
UF_s
=
\frac{\#(e_s>0)}{N}
\]

---

# 31. Overprediction fraction

\[
OF_s
=
\frac{\#(e_s<0)}{N}
\]

---

# 32. Exact-zero fraction

\[
ZF_s
=
\frac{\#(e_s=0)}{N}
\]

and:

```text
UF + OF + ZF = 1.
```

Hard audit.

---

# 33. No bias correction

If mean residual is non-zero:

```text
report.
```

Do not subtract the Test mean residual from predictions.

That would be post-Test calibration.

---

# 34. Positive/negative tail asymmetry

Compute separately:

```text
positive residual count
negative residual count
mean positive residual
mean absolute negative residual
p95 positive residual if available
p95 |negative residual| if available
max underprediction residual
max overprediction magnitude.
```

This helps distinguish:

```text
upward vs downward miss severity.
```

---

# 35. Tail labels are descriptive

Do not define new scientific “extreme error regime” from Test tails here.

Worst individual cases are Phase51.

---

# 36. Temporal residual plot

Core figure:

```text
timestamp vs residual
```

for all three seeds.

Horizontal:

```text
0 line.
```

Purpose:

```text
persistent bias periods
error clustering
large bursts
seed agreement.
```

---

# 37. Avoid unreadable overlay

Preferred outputs:

```text
one all-seed overview
+
one separate residual timeline per seed if needed.
```

No cherry-picked period as primary evidence.

---

# 38. Gap-safe temporal diagnostics

Never treat timestamps separated by a data gap as adjacent.

Use exact cadence:

```text
10 minutes
```

and continuity segment IDs from upstream temporal contract where available.

---

# 39. Residual autocorrelation

Primary temporal dependence diagnostic:

\[
\rho_k
=
Corr(e_t,e_{t-k})
\]

computed only on valid within-segment pairs at exact lag `k`.

---

# 40. Registered ACF lag range

Primary full curve:

```text
lags 1 ... min(144, supported max lag)
```

Key lags:

```text
1   = 10 min
6   = 1 h
12  = 2 h
36  = 6 h
72  = 12 h
144 = 24 h.
```

---

# 41. Gap-safe ACF implementation

Preferred custom pair-pair implementation:

For lag `k`, include pair only if:

```text
same continuity segment
timestamp_t - timestamp_t-k = k * 10 minutes.
```

No crossing gaps.

---

# 42. ACF confidence bands

Standard white-noise approximate bands:

\[
\pm 1.96/\sqrt{N}
\]

may be plotted only as:

```text
rough visual reference
```

because residuals are time-dependent and forecasting observations are not iid.

Do not make strong significance claims from the band.

---

# 43. Ljung–Box diagnostic

Optional/secondary but useful.

Predeclare lags:

```text
6
36
144
```

when supported.

Run only if the Test residual series is one sufficiently long contiguous segment.

If multiple segments:

```text
run per adequate segment
or mark global Ljung–Box NOT_APPLICABLE.
```

---

# 44. Ljung–Box interpretation

Safe:

```text
evidence that residual autocorrelation remains detectable at the tested lag horizon.
```

Avoid:

```text
model is statistically invalid.
```

This is a forecasting diagnostic, not a model-validity gate.

---

# 45. No post-hoc lag correction

If residual ACF shows structure:

```text
report for future work.
```

Do not retrain/shift predictions.

---

# 46. Residual sign runs

A sign run is a maximal contiguous sequence of:

```text
all POSITIVE residuals
or
all NEGATIVE residuals.
```

Break a run at:

```text
sign change
ZERO residual
temporal gap.
```

---

# 47. Sign-run diagnostics

For each seed:

```text
positive run count
negative run count
median positive run length
median negative run length
max positive run length
max negative run length
p90 run length
fraction of samples inside runs length >= 3
fraction inside runs length >= 6.
```

At 10-minute cadence:

```text
3 steps = 30 min
6 steps = 1 hour.
```

---

# 48. Why sign runs matter

Long same-sign runs suggest:

```text
persistent underprediction
or
persistent overprediction
```

rather than isolated random errors.

Still diagnostic only.

---

# 49. Residual sign transition matrix

Optional:

```text
NEG→NEG
NEG→ZERO
NEG→POS
ZERO→...
POS→...
```

computed gap-safe.

Useful for describing persistence.

---

# 50. Rolling residual mean

Use fixed:

```text
window = 144 consecutive 10-minute samples ≈ 24h.
```

Only full contiguous windows.

Compute:

```text
rolling mean residual.
```

This reveals slowly varying local bias.

---

# 51. Rolling residual variability

Same fixed 144-point window:

```text
rolling residual std
rolling median absolute residual optional.
```

This reveals temporal variance clustering.

---

# 52. Rolling windows are diagnostic

Do not select thresholds from:

```text
rolling mean
rolling std
```

to modify model.

---

# 53. No partial-window filling

Require:

```text
144 valid contiguous observations.
```

No padding/interpolation.

---

# 54. Heteroskedasticity concept

Question:

> Does residual magnitude change systematically with prediction level?

Phase49 addresses this descriptively.

---

# 55. Residual-vs-predicted scatter

For each seed:

```text
x = y_pred_wh
y = residual_wh.
```

Include:

```text
horizontal y=0.
```

Use common axes across seeds where possible.

---

# 56. Absolute residual-vs-predicted scatter

For each seed:

```text
x = y_pred_wh
y = |residual|.
```

Purpose:

```text
visualize changing error scale.
```

No causal interpretation.

---

# 57. Spearman magnitude association

Compute:

```text
Spearman(|residual|, y_pred)
Spearman(|residual|, y_true)
```

per seed.

This is descriptive evidence of error-scale association.

---

# 58. Why Spearman

Less sensitive to extreme values and nonlinear monotonic patterns than Pearson.

Still no causality.

---

# 59. Prediction-decile residual diagnostics

For each seed:

```text
sort by y_pred
split into 10 equal-frequency bins
```

where possible.

Within each bin report:

```text
count
predicted mean
true mean
mean residual
residual std
MAE
RMSE.
```

---

# 60. Deciles are diagnostic, not scientific regimes

These bins are Test-derived from predictions.

They must never be reused as Phase50 target regimes.

Phase50 uses Train-derived thresholds.

---

# 61. Equal-frequency bin edge handling

If ties prevent exactly equal sizes:

```text
use deterministic rank-based assignment
or robust qcut with duplicate-edge handling documented.
```

No adaptive number of bins based on favorable appearance.

Target:

```text
10 bins.
```

If impossible due degenerate predictions:

```text
mark reduced-bin condition explicitly.
```

---

# 62. Heteroskedasticity formal tests

Tests such as Breusch–Pagan are not central because these are deep forecasting residuals rather than OLS residuals with classical assumptions.

Recommended:

```text
residual-vs-prediction plots
absolute-error associations
prediction-decile residual spread
rolling residual std.
```

This is methodologically safer.

---

# 63. Residual-vs-actual scatter

Optional:

```text
x=y_true
y=residual.
```

Useful descriptively, but formal low/medium/high target analysis remains Phase50.

Do not bin y_true into Test-derived regimes here.

---

# 64. Cross-seed residual pairwise agreement

For pairs:

```text
42 vs 123
42 vs 2026
123 vs 2026
```

compute:

```text
Pearson residual correlation
Spearman residual correlation
mean absolute residual difference
RMSE between residual vectors
residual sign agreement rate.
```

---

# 65. Cross-seed residual correlation interpretation

High correlation means:

```text
seeds tend to miss the same targets in similar directions/magnitudes.
```

It does not prove deterministic error causes.

---

# 66. Cross-seed residual sign consensus

For each timestamp classify:

```text
ALL_UNDER
= all 3 residuals > 0

ALL_OVER
= all 3 residuals < 0

MIXED
= otherwise

ALL_ZERO
= all 3 exactly 0
```

You may optionally distinguish zero-mixed categories, but canonical summary should remain simple.

---

# 67. Consensus proportions

Report:

```text
fraction ALL_UNDER
fraction ALL_OVER
fraction MIXED
fraction ALL_ZERO.
```

This indicates whether errors are shared across seeds.

---

# 68. Per-target residual spread

Because:

\[
e_{s,t}=y_t-\hat y_{s,t}
\]

cross-seed residual spread has the same magnitude as cross-seed prediction spread.

Do not duplicate large tables unnecessarily.

Phase49 may reference:

```text
Phase48 seed-spread table
```

and focus on sign/magnitude agreement.

---

# 69. Seed-level residual statistic aggregation

For each residual statistic, create:

```text
seed42
seed123
seed2026
mean across seeds
sample SD across seeds
```

Examples:

```text
mean residual
median residual
residual std
skewness
kurtosis
underprediction fraction
lag1 residual autocorrelation
max positive run length
```

This is descriptive cross-seed stability.

---

# 70. Do not compute “mean residual series performance” as primary

No official residual model is created by averaging seeds.

---

# 71. Baseline residual context

If Phase47 baseline prediction bundles exist:

```text
Persistence
LSTM_TUNED_DEV
```

Phase49 may produce a compact baseline residual summary:

```text
mean residual
median residual
residual std
underprediction fraction
lag1 ACF
lag6 ACF
lag36 ACF
```

on exact same Test targets.

---

# 72. Baseline context is secondary

Main residual analysis is the final Transformer.

Do not open an additional model selection.

---

# 73. LSTM training-protocol caveat

If LSTM is included:

```text
retain Phase47 caveat that it is a frozen development checkpoint rather than a Train+Validation final refit.
```

---

# 74. Persistence behavior caveat

Persistence naturally induces error related to:

```text
one-step target changes.
```

Use it as a diagnostic reference, not proof of residual independence.

---

# 75. No baseline-specific Test tuning

No.

---

# 76. Error clustering over time

Phase49 may identify:

```text
intervals where rolling residual std is elevated
```

but must not define formal performance regimes here.

Phase50 handles predeclared regimes.

---

# 77. No “bad period” cherry-pick metrics

Do not select a period after seeing rolling errors and recompute a headline RMSE.

Phase51 will inspect worst cases under a fixed ranking.

---

# 78. Residual temporal heatmap

Optional descriptive figure:

```text
rows = date
columns = 10-minute time-of-day slot
value = residual.
```

Use zero-centered diverging scale symmetric around 0.

Generate per seed or one deterministic chosen format.

Because no seed may be chosen as “representative”, preferred:

```text
three separate heatmaps
```

or:

```text
seed-mean residual heatmap labeled descriptive only.
```

---

# 79. Recommended heatmap default

Prefer:

```text
three separate seed heatmaps
```

if report space allows.

This avoids creating an implicit ensemble-like residual.

---

# 80. Heatmap missing cells

Leave missing.

No interpolation.

---

# 81. Residual histogram scale

Use same x-axis and bins across seeds.

This prevents visual distortion.

---

# 82. Q-Q plot scale

Each seed separate.

Use standardized residuals only for Q-Q plotting if necessary, but raw residual summary remains authoritative.

---

# 83. Standardized residual terminology

Avoid classical standardized/studentized residual claims unless denominator is explicitly defined.

Preferred:

```text
raw residual Wh
```

for all core analyses.

---

# 84. No studentized residuals

Not needed.

---

# 85. No Cook’s distance/leverage

These are classical regression influence diagnostics and do not map directly to this sequence model.

Worst cases are handled directly in Phase51.

---

# 86. No Shapiro PASS/FAIL

Residual normality is not acceptance criterion.

---

# 87. No residual correction model

Do not fit:

```text
AR model on residuals
bias correction
post-processing regressor
```

using Test residuals.

That would be post-Test model modification.

---

# 88. No residual forecasting

No.

---

# 89. No calibration correction

No subtracting MBE from predictions.

---

# 90. No Test-derived loss redesign

If residuals show heavy positive tails:

```text
report as limitation/future work
```

do not go back to Huber tuning.

---

# 91. No Test-derived lookback redesign

If ACF remains at 24h:

```text
future-work hypothesis only.
```

Do not reopen S4.

---

# 92. No Test-derived feature redesign

If residual heatmap shows daily structure:

```text
discussion only.
```

No new features.

---

# 93. Residual analysis should be reproducible from prediction bundles alone

No model checkpoint required.

This is a hard design preference.

---

# 94. Phase49 output directory

```text
artifacts/
└── residual_analysis/
    ├── residual_analysis_manifest.json
    ├── residual_analysis_contract.json
    ├── phase49_preflight_audit.csv
    ├── residual_source_verification.csv
    ├── residual_recomputation_audit.csv
    ├── residual_metric_reconstruction_audit.csv
    ├── residual_long_table.csv
    ├── residual_wide_table.csv
    ├── residual_distribution_summary.csv
    ├── residual_bias_summary.csv
    ├── residual_tail_summary.csv
    ├── residual_sign_balance.csv
    ├── residual_acf_diagnostics.csv
    ├── residual_ljung_box_diagnostics.csv
    ├── residual_sign_run_summary.csv
    ├── residual_sign_transition_summary.csv
    ├── residual_rolling_diagnostics.csv
    ├── residual_magnitude_association.csv
    ├── residual_prediction_decile_diagnostics.csv
    ├── residual_seed_pairwise_agreement.csv
    ├── residual_seed_sign_consensus.csv
    ├── residual_cross_seed_stat_summary.csv
    ├── residual_baseline_context.csv
    ├── residual_analysis_findings.csv
    ├── residual_analysis_tests.csv
    ├── residual_analysis_discrepancies.json
    ├── phase50_error_regime_handoff.json
    ├── phase51_worst_error_handoff.json
    ├── residual_analysis_summary.json
    ├── residual_analysis_report.md
    ├── figures/
    │   ├── RESID_49_01_residual_over_time_all_seeds.png
    │   ├── RESID_49_02_residual_ecdf.png
    │   ├── RESID_49_03_residual_histogram_common_bins.png
    │   ├── RESID_49_04_qq_seed42.png
    │   ├── RESID_49_05_qq_seed123.png
    │   ├── RESID_49_06_qq_seed2026.png
    │   ├── RESID_49_07_residual_acf_all_seeds.png
    │   ├── RESID_49_08_rolling_24h_residual_mean.png
    │   ├── RESID_49_09_rolling_24h_residual_std.png
    │   ├── RESID_49_10_residual_vs_prediction_seed42.png
    │   ├── RESID_49_11_residual_vs_prediction_seed123.png
    │   ├── RESID_49_12_residual_vs_prediction_seed2026.png
    │   ├── RESID_49_13_abs_residual_vs_prediction.png
    │   ├── RESID_49_14_prediction_decile_residual_std.png
    │   ├── RESID_49_15_residual_sign_run_lengths.png
    │   ├── RESID_49_16_seed_pairwise_residual_agreement.png
    │   ├── RESID_49_17_seed_sign_consensus_over_time.png
    │   ├── RESID_49_18_residual_heatmap_seed42.png
    │   ├── RESID_49_19_residual_heatmap_seed123.png
    │   └── RESID_49_20_residual_heatmap_seed2026.png
    ├── README_RESIDUAL_ANALYSIS.md
    └── phase_49_signoff.json
```

Optional figures/tables may be marked NOT_APPLICABLE if Test continuity/length is insufficient.

---

# 95. Required outputs

```text
O49.1  Residual analysis manifest
O49.2  Residual analysis contract
O49.3  Preflight audit
O49.4  Source verification
O49.5  Residual recomputation audit
O49.6  Metric reconstruction audit
O49.7  Long residual table
O49.8  Wide residual table
O49.9  Distribution summary
O49.10 Bias summary
O49.11 Tail summary
O49.12 Sign balance
O49.13 Gap-safe ACF diagnostics
O49.14 Optional Ljung–Box diagnostics
O49.15 Sign-run summary
O49.16 Sign-transition summary
O49.17 Rolling residual diagnostics
O49.18 Residual-magnitude association
O49.19 Prediction-decile residual diagnostics
O49.20 Pairwise seed residual agreement
O49.21 Cross-seed sign consensus
O49.22 Cross-seed statistic summary
O49.23 Baseline residual context
O49.24 Core figures
O49.25 Findings
O49.26 Phase50 handoff
O49.27 Phase51 handoff
O49.28 Tests
O49.29 Discrepancy log
O49.30 Summary JSON
O49.31 Human-readable report
O49.32 README
O49.33 Sign-off
```

---

# 96. Residual analysis manifest

`residual_analysis_manifest.json`:

```text
phase=49
version=RESIDUAL_ANALYSIS-v1
source_phase47_version
source_phase48_version
final_lock_sha256
test_population_sha256
source_prediction_sha256s
seed_list=[42,123,2026]
residual_definition=Y_TRUE_MINUS_Y_PRED
residual_unit=Wh
new_inference=false
new_training=false
model_selection=false
best_seed_selection=false
prediction_modification=false
formal_regime_analysis=false
worst_error_ranking=false
status
created_at
```

---

# 97. Residual analysis contract

`residual_analysis_contract.json` must state:

```text
Use frozen Phase47 prediction bundles.

Residual:
y_true - y_pred.

Positive:
underprediction.

Negative:
overprediction.

All three seeds retained.

No 3N iid interpretation.
No best seed.
No ensemble.
No prediction correction.
No residual correction model.
No Test-driven retuning.

Primary diagnostics:
distribution
bias
tails
sign balance
gap-safe autocorrelation
sign persistence
rolling mean/std
residual-vs-prediction structure
cross-seed residual agreement.

Target-regime analysis deferred to Phase50.
Worst-error ranking deferred to Phase51.
```

---

# 98. Preflight audit

`phase49_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Required:

```text
Phase48 PASS/PASS_WITH_WARNING
Phase49 handoff exists
Phase47 source bundles exist
source checksums match
same Test population
same target IDs/order/y_true
all y_pred finite
residual convention frozen
no inference code path required
no model checkpoint required
baseline availability known.
```

---

# 99. Source verification schema

`residual_source_verification.csv`:

```text
seed
prediction_file
expected_sha256
observed_sha256
row_count
population_sha256
frozen
status
```

---

# 100. Residual recomputation audit

`residual_recomputation_audit.csv`:

```text
seed
N
max_abs_difference_recomputed_vs_stored_residual
max_abs_difference_abs_error
max_abs_difference_squared_error
all_within_tolerance
status
```

If Phase47 bundle did not store one of these fields:

```text
mark SOURCE_FIELD_NOT_PRESENT
```

and use recomputed value.

---

# 101. Metric reconstruction audit

`residual_metric_reconstruction_audit.csv`:

```text
seed
mae_from_residual
phase47_mae
mae_abs_diff
rmse_from_residual
phase47_rmse
rmse_abs_diff
r2_recomputed
phase47_r2
r2_abs_diff
within_tolerance
status
```

---

# 102. Long residual table schema

`residual_long_table.csv`:

```text
target_id
target_timestamp
seed
y_true_wh
y_pred_wh
residual_wh
absolute_error_wh
squared_error_wh2
residual_sign
continuity_segment_id
source_prediction_sha256
```

If continuity segment ID not already available, resolve from locked temporal metadata by target ID.

---

# 103. Wide residual table schema

`residual_wide_table.csv`:

```text
target_id
target_timestamp
y_true_wh
residual_seed42
residual_seed123
residual_seed2026
abs_error_seed42
abs_error_seed123
abs_error_seed2026
sign_seed42
sign_seed123
sign_seed2026
```

---

# 104. Distribution summary schema

`residual_distribution_summary.csv`:

```text
seed
N
mean_residual_wh
median_residual_wh
std_residual_wh
mad_residual_wh
min
p01
p05
p10
q1
q3
p90
p95
p99
max
iqr
skewness
excess_kurtosis
mae_from_residual
rmse_from_residual
status
```

---

# 105. Bias summary schema

`residual_bias_summary.csv`:

```text
seed
mean_bias_error_wh
normalized_mean_bias_pct
median_signed_error_wh
underprediction_count
underprediction_fraction
overprediction_count
overprediction_fraction
zero_count
zero_fraction
sign_fraction_sum
status
```

---

# 106. Tail summary schema

`residual_tail_summary.csv`:

```text
seed
positive_count
negative_count
mean_positive_residual
mean_negative_residual
mean_abs_negative_residual
p95_positive_residual_if_available
p95_abs_negative_residual_if_available
max_underprediction_residual
max_overprediction_magnitude
status
```

---

# 107. Sign balance schema

`residual_sign_balance.csv`:

```text
seed
sign
count
fraction
interpretation
status
```

Signs:

```text
POSITIVE_UNDER
ZERO
NEGATIVE_OVER.
```

---

# 108. Gap-safe ACF schema

`residual_acf_diagnostics.csv`:

```text
seed
lag_steps
lag_minutes
valid_pair_count
acf
approx_white_noise_band_upper
approx_white_noise_band_lower
gap_safe=true
status
```

---

# 109. Ljung–Box schema

`residual_ljung_box_diagnostics.csv`:

```text
seed
segment_id
lag_steps
segment_length
lb_statistic
p_value
applicable
interpretation_scope=SECONDARY_DIAGNOSTIC
status
```

No global p-value across disconnected segments unless a justified combination method is predeclared; default no combination.

---

# 110. Sign-run summary schema

`residual_sign_run_summary.csv`:

```text
seed
sign
run_count
median_run_length_steps
p90_run_length_steps
max_run_length_steps
max_run_length_minutes
sample_fraction_in_runs_ge_3
sample_fraction_in_runs_ge_6
status
```

---

# 111. Sign transition schema

`residual_sign_transition_summary.csv`:

```text
seed
from_sign
to_sign
transition_count
row_fraction
gap_safe=true
status
```

---

# 112. Rolling diagnostics schema

`residual_rolling_diagnostics.csv`:

```text
target_timestamp
seed
window_steps=144
window_minutes=1440
rolling_mean_residual
rolling_std_residual
rolling_median_abs_residual_optional
full_contiguous_window
status
```

Only valid rows where full window exists.

---

# 113. Residual magnitude association schema

`residual_magnitude_association.csv`:

```text
seed
association
coefficient
valid_count
method
status
```

Required associations:

```text
abs_residual_vs_prediction → Spearman
abs_residual_vs_y_true     → Spearman
```

Optional:

```text
squared_residual_vs_prediction → Spearman.
```

---

# 114. Prediction-decile diagnostics schema

`residual_prediction_decile_diagnostics.csv`:

```text
seed
prediction_decile
count
pred_mean
pred_min
pred_max
true_mean
mean_residual
residual_std
mae
rmse
status
```

Exactly:

```text
10 intended deciles
```

unless duplicate-value degeneracy forces documented reduction.

---

# 115. Pairwise seed agreement schema

`residual_seed_pairwise_agreement.csv`:

```text
seed_a
seed_b
pearson_residual_corr
spearman_residual_corr
mean_abs_residual_difference
rmse_between_residuals
sign_agreement_count
sign_agreement_rate
status
```

---

# 116. Cross-seed sign consensus schema

`residual_seed_sign_consensus.csv`:

```text
target_id
timestamp
y_true_wh
sign_seed42
sign_seed123
sign_seed2026
consensus_class
status
```

Summary counts/fractions can be in same file or a companion summary section.

Canonical classes:

```text
ALL_UNDER
ALL_OVER
MIXED
ALL_ZERO.
```

---

# 117. Cross-seed statistic summary

`residual_cross_seed_stat_summary.csv`:

```text
statistic
seed42
seed123
seed2026
mean_across_seeds
sample_sd_across_seeds
min
max
status
```

Include selected diagnostics:

```text
mean residual
median residual
residual std
MAD
skewness
excess kurtosis
underprediction fraction
lag1 ACF
lag6 ACF
lag36 ACF
max positive run length
max negative run length.
```

---

# 118. Baseline context schema

`residual_baseline_context.csv`:

```text
model_id
eligible
N
mean_residual
median_residual
residual_std
underprediction_fraction
lag1_acf
lag6_acf
lag36_acf
training_protocol_note
status
```

Rows:

```text
Persistence
LSTM_TUNED_DEV if available.
```

---

# 119. Core figure RESID_49_01

`residual_over_time_all_seeds`

Must include:

```text
zero line
chronological Test timestamps
all three seeds.
```

No baseline by default to avoid clutter.

---

# 120. ECDF figure

`RESID_49_02_residual_ecdf`

Show all three seeds with:

```text
vertical zero line.
```

Useful for:

```text
median location
tail comparison
overall distribution overlap.
```

---

# 121. Histogram figure

Use:

```text
50 common bins
same range
same bin edges.
```

No separate seed-specific auto-bins.

---

# 122. Q-Q figures

One per seed.

Label:

```text
Gaussian reference diagnostic only.
```

---

# 123. ACF figure

Show all seeds on same registered lag axis when readable.

Highlight key lags:

```text
1
6
12
36
72
144.
```

Approximate white-noise bands may be shown with caveat.

---

# 124. Rolling residual mean figure

All seeds:

```text
24h rolling mean residual
zero line.
```

Shows time-varying local signed bias.

---

# 125. Rolling residual std figure

All seeds:

```text
24h rolling residual std.
```

No arbitrary high-variance threshold.

---

# 126. Residual-vs-prediction figures

One per seed.

Axes:

```text
x predicted Wh
y residual Wh
```

same scale policy across seeds.

Horizontal `0`.

---

# 127. Absolute residual-vs-prediction figure

Can combine seeds or use one multi-line binned summary.

Preferred:

```text
prediction decile
vs
residual std / MAE
```

for readability.

---

# 128. Sign-run figure

Plot distributions/counts of:

```text
positive-run lengths
negative-run lengths
```

for each seed.

No cross-gap runs.

---

# 129. Pairwise residual agreement figure

Three pairwise scatter plots or a compact correlation heatmap.

If using correlation heatmap:

```text
residual correlation
```

not prediction correlation.

---

# 130. Sign-consensus timeline

Optional:

```text
timestamp
consensus class
```

Useful for identifying shared under/overprediction regions.

This still does not rank errors.

---

# 131. Residual heatmaps

If generated:

```text
same symmetric color limit across seeds:
[-M, +M]
```

where:

```text
M=max absolute residual across all 3 seeds.
```

This is visualization scaling only.

---

# 132. No color-scale manipulation by seed

Same range across all three residual heatmaps.

---

# 133. Findings codes

Possible:

```text
RESIDUAL_MEAN_NEAR_ZERO_DESCRIPTIVE
PERSISTENT_UNDERPREDICTION_BIAS
PERSISTENT_OVERPREDICTION_BIAS
RESIDUAL_DISTRIBUTION_RIGHT_SKEWED
RESIDUAL_DISTRIBUTION_LEFT_SKEWED
HEAVY_RESIDUAL_TAILS
POSITIVE_TAIL_DOMINANT
NEGATIVE_TAIL_DOMINANT
SHORT_LAG_RESIDUAL_AUTOCORRELATION
HOURLY_RESIDUAL_AUTOCORRELATION
DAILY_RESIDUAL_AUTOCORRELATION
NO_CLEAR_RESIDUAL_AUTOCORRELATION
LONG_UNDERPREDICTION_RUNS
LONG_OVERPREDICTION_RUNS
ROLLING_BIAS_DRIFT
ROLLING_VARIANCE_CLUSTERING
ERROR_MAGNITUDE_INCREASES_WITH_PREDICTION_LEVEL
NO_CLEAR_MAGNITUDE_LEVEL_ASSOCIATION
HIGH_CROSS_SEED_RESIDUAL_AGREEMENT
MIXED_CROSS_SEED_ERROR_DIRECTIONS
ALL_SEEDS_SHARE_UNDERPREDICTION_PERIODS
ALL_SEEDS_SHARE_OVERPREDICTION_PERIODS
BASELINE_RESIDUAL_CONTEXT_AVAILABLE
SOURCE_BUNDLES_VERIFIED
METRIC_RECONSTRUCTION_VERIFIED
NO_NEW_INFERENCE
NO_MODEL_SELECTION
NO_POST_TEST_CORRECTION
```

Use exact descriptive statistics in findings.

---

# 134. Findings language

Safe:

> Residuals were predominantly positive for all three seeds, indicating that underprediction occurred more frequently than overprediction on the held-out Test period.

Safe:

> Residual autocorrelation remained visible at short lags, suggesting that forecast errors were not temporally independent.

Safe:

> Error magnitude increased across higher predicted-value deciles, which is consistent with non-constant residual variance over the Test period.

Unsafe:

> Residuals prove the model is invalid.

Unsafe:

> The model should be corrected by subtracting the mean residual.

---

# 135. No arbitrary "near zero" threshold

If using phrase:

```text
near zero
```

show actual MBE relative to:

```text
mean target
RMSE
or residual SD.
```

Prefer exact values.

Do not invent a pass threshold after Test.

---

# 136. No arbitrary autocorrelation threshold

Report:

```text
actual ACF coefficients
```

rather than declaring “high” without context.

---

# 137. No p-value-driven model modification

Even if Ljung–Box is small:

```text
no retraining.
```

---

# 138. Phase50 handoff purpose

Phase50 will answer:

> In which predeclared data regimes does the model make larger or smaller errors?

It must use:

```text
Train-derived thresholds
```

not Phase49 Test quantiles.

Phase49 hands off residual/error vectors and structural findings.

---

# 139. Phase50 handoff schema

`phase50_error_regime_handoff.json`:

```text
source_phase49_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
residual_long_table_path
residual_long_table_sha256
residual_wide_table_path
source_prediction_sha256s
residual_definition=y_true-y_pred
required_regime_threshold_source=TRAIN_ONLY
forbid_Test_derived_regime_thresholds=true
bias_summary_ref
acf_summary_ref
rolling_variance_ref
seed_consensus_ref
ready_for_phase50=true
```

---

# 140. Phase51 handoff purpose

Phase51 will investigate:

```text
largest absolute/squared errors
```

using the same frozen residual data.

Phase49 itself does not create a top-error leaderboard.

---

# 141. Phase51 handoff schema

`phase51_worst_error_handoff.json`:

```text
source_phase49_version
test_population_sha256
seed_list
residual_long_table_path
residual_definition
absolute_error_field
squared_error_field
continuity metadata
seed_sign_consensus_ref
no_rows_removed=true
no_worst_error_ranking_performed=true
ready_for_phase51=true
```

---

# 142. Discrepancy taxonomy

`residual_analysis_discrepancies.json`:

```text
PHASE48_NOT_APPROVED
PHASE47_SOURCE_MISSING
SOURCE_PREDICTION_CHECKSUM_MISMATCH
TEST_POPULATION_MISMATCH
TARGET_ID_DUPLICATE
TARGET_ORDER_MISMATCH
YTRUE_MISMATCH_ACROSS_SEEDS
RESIDUAL_DEFINITION_DRIFT
RESIDUAL_RECOMPUTE_MISMATCH
ABS_ERROR_RECOMPUTE_MISMATCH
SQUARED_ERROR_RECOMPUTE_MISMATCH
PHASE47_METRIC_RECONSTRUCTION_MISMATCH
NONFINITE_RESIDUAL
SOURCE_FILE_MODIFIED
NEW_TEST_INFERENCE_ATTEMPT
CHECKPOINT_LOADING_FOR_NEW_PREDICTION
MODEL_TRAINING_ATTEMPT
BEST_SEED_SELECTION_ATTEMPT
SEED_POOLING_AS_3N_IID
ENSEMBLE_RESIDUAL_PROMOTION
POST_TEST_BIAS_CORRECTION_ATTEMPT
POST_TEST_RESIDUAL_MODEL_ATTEMPT
PREDICTION_CLIPPING_ATTEMPT
GAP_UNSAFE_ACF
GAP_UNSAFE_SIGN_RUN
ROLLING_WINDOW_CROSSES_GAP
LJUNG_BOX_ON_DISCONNECTED_SERIES
NORMALITY_TEST_USED_AS_PASS_FAIL
TEST_DERIVED_REGIME_THRESHOLD
PREDICTION_DECILE_REUSED_AS_TARGET_REGIME
WORST_ERROR_RANKING_SCOPE_CREEP
ATTENTION_SCOPE_CREEP
BASELINE_POPULATION_MISMATCH
LSTM_SCALER_MISMATCH
OTHER
```

---

# 143. Status model

## PASS

```text
source bundles verified
residual recomputation exact
Phase47 metrics reconstructed
all 3 seeds analyzed
distribution/bias/tails complete
temporal dependence diagnostics complete
variance diagnostics complete
cross-seed residual agreement complete
no inference/retraining/correction
Phase50/51 handoffs ready.
```

## PASS_WITH_WARNING

Possible:

```text
Ljung–Box not applicable due multiple segments
Test too short for lag144
baseline residual file unavailable
heavy tails observed
strong residual autocorrelation observed
large seed-shared residual structure.
```

These are findings, not methodological failure.

## FAIL

Examples:

```text
source checksum mismatch
residual metric mismatch
new inference
gap-unsafe diagnostics
post-Test correction
best-seed selection.
```

---

# 144. Phase49 execution sequence

```text
1. Verify Phase48/47 signoffs.
2. Verify prediction checksums.
3. Align all three seed prediction bundles.
4. Recompute residual/absolute/squared errors.
5. Verify stored error fields.
6. Reconstruct MAE/RMSE/R² and compare Phase47.
7. Build long/wide residual tables.
8. Compute distribution statistics.
9. Compute signed bias/sign balance.
10. Compute positive/negative tail summaries.
11. Compute gap-safe residual ACF.
12. Run optional contiguous-series Ljung–Box.
13. Compute sign-run diagnostics.
14. Compute sign-transition diagnostics.
15. Compute fixed 24h rolling residual mean/std.
16. Compute residual magnitude associations.
17. Compute fixed prediction-decile residual summaries.
18. Compute cross-seed residual pairwise agreement.
19. Compute cross-seed sign consensus.
20. Build cross-seed statistic summary.
21. Build baseline context if available.
22. Generate figures.
23. Write findings.
24. Write Phase50/51 handoffs.
25. Run tests/discrepancy audit.
26. Write summary/report/README.
27. Sign off.
```

---

# 145. Preflight acceptance checklist

```text
[ ] Phase48 PASS/PASS_WITH_WARNING.
[ ] Phase49 handoff available.
[ ] Phase47 source prediction bundles exist.
[ ] Seed42 checksum matches.
[ ] Seed123 checksum matches.
[ ] Seed2026 checksum matches.
[ ] Same Test population.
[ ] Same target IDs.
[ ] Same timestamps.
[ ] Same y_true.
[ ] All predictions finite.
[ ] Source files frozen.
[ ] Residual convention loaded.
[ ] No model inference required.
```

---

# 146. Residual integrity acceptance checklist

```text
[ ] residual = y_true-y_pred.
[ ] positive = underprediction.
[ ] negative = overprediction.
[ ] abs_error = abs(residual).
[ ] squared_error = residual².
[ ] all residuals finite.
[ ] same N all seeds.
[ ] recomputed residual matches stored field.
[ ] recomputed MAE matches Phase47.
[ ] recomputed RMSE matches Phase47.
[ ] recomputed R² matches Phase47.
[ ] no Test row removed.
```

---

# 147. Distribution acceptance checklist

```text
[ ] Mean residual.
[ ] Median residual.
[ ] Residual SD.
[ ] MAD.
[ ] p01/p05/p10.
[ ] Q1/Q3.
[ ] p90/p95/p99.
[ ] Min/max.
[ ] Skewness definition documented.
[ ] Excess kurtosis definition documented.
[ ] ECDF generated.
[ ] Common-bin histogram generated.
[ ] Q-Q plots generated.
[ ] Normality not used as PASS criterion.
```

---

# 148. Bias acceptance checklist

```text
[ ] MBE computed.
[ ] NMBE computed/marked optional.
[ ] Median signed error.
[ ] Underprediction count/fraction.
[ ] Overprediction count/fraction.
[ ] Zero count/fraction.
[ ] Fractions sum to 1.
[ ] Positive/negative tail summaries.
[ ] No Test bias correction applied.
```

---

# 149. Temporal acceptance checklist

```text
[ ] Continuity segments available.
[ ] ACF gap-safe.
[ ] Lags up to 144 where supported.
[ ] Key lags 1/6/12/36/72/144.
[ ] No cross-gap pairs.
[ ] Approximate bands labeled approximate.
[ ] Ljung–Box only on contiguous adequate segment(s).
[ ] Sign runs break at gap.
[ ] Sign runs break at zero/sign change.
[ ] Rolling window fixed 144.
[ ] Rolling window requires full continuity.
[ ] No lag-based prediction correction.
```

---

# 150. Variance-structure acceptance checklist

```text
[ ] Residual-vs-predicted plots generated.
[ ] Absolute residual-vs-predicted diagnostic generated.
[ ] Spearman |e| vs prediction.
[ ] Spearman |e| vs y_true.
[ ] 10 prediction deciles attempted.
[ ] Decile count documented.
[ ] Residual std/MAE/RMSE by prediction decile.
[ ] Deciles labeled diagnostic only.
[ ] Deciles not reused as Phase50 target regimes.
```

---

# 151. Cross-seed acceptance checklist

```text
[ ] Pairwise residual Pearson.
[ ] Pairwise residual Spearman.
[ ] Mean absolute residual differences.
[ ] Pairwise residual RMSE.
[ ] Sign agreement.
[ ] ALL_UNDER/ALL_OVER/MIXED consensus.
[ ] Cross-seed statistic summary.
[ ] No best seed.
[ ] No 3N iid inference.
[ ] No ensemble residual promoted.
```

---

# 152. Scope acceptance checklist

```text
[ ] No target-regime RMSE table.
[ ] No Test-derived target threshold.
[ ] No hour/weekend regime claims.
[ ] No top worst-error ranking.
[ ] No individual failure-case narrative.
[ ] No attention maps.
[ ] No residual correction model.
[ ] No recalibration.
[ ] No retraining.
[ ] No new Test inference.
```

---

# 153. Provenance acceptance checklist

```text
[ ] Prediction source SHA256s stored.
[ ] Test population SHA256 stored.
[ ] Residual derived files separate from Phase47.
[ ] Derived tables trace to source rows.
[ ] Analysis version stored.
[ ] Figures trace to residual tables.
[ ] Phase50 handoff references exact files/checksums.
[ ] Phase51 handoff references exact files/checksums.
```

---

# 154. Acceptance criteria

Phase49 PASS only when:

```text
Residuals are recomputed from the immutable Phase47 prediction bundles using the locked y_true-y_pred convention.

Recomputed residual, absolute-error and squared-error fields agree with upstream values where present.

MAE/RMSE/R² reconstructed from residuals agree with Phase47 metrics.

All three final seeds are analyzed on the exact same Test population without selecting a representative seed.

Residual distribution, robust location/scale, skewness and tail behavior are documented.

Underprediction/overprediction balance and signed bias are quantified.

Residual temporal dependence is analyzed using gap-safe lag pairs.

Same-sign residual run lengths are quantified without crossing temporal gaps.

Rolling 24-hour residual mean and variability use only complete contiguous windows.

Residual magnitude dependence on prediction level is analyzed descriptively without redefining model-selection regimes.

Cross-seed residual correlation and residual-sign consensus are quantified.

Any baseline residual context uses common Test targets and frozen upstream prediction bundles.

No residual correction, recalibration, Test-driven retuning, new inference or best-seed selection occurs.

Formal target-regime analysis is deferred to Phase50 using Train-derived thresholds.

Worst-error ranking and failure-case investigation are deferred to Phase51.

Phase50 and Phase51 handoffs are generated.
```

---

# 155. Failure conditions

Phase49 FAIL if:

```text
source predictions differ from frozen Phase47 bundles

residual sign convention changes

recomputed metrics do not match Phase47

a seed has a different Test population

a row is dropped due large residual

ACF/sign runs cross temporal gaps

Test prediction quantiles become model-selection regimes

residual mean is subtracted from predictions

a post-hoc residual correction model is fitted

best seed is selected

3N residuals are treated as independent Test samples for inference

new Test inference is run

model/checkpoint is retrained

worst-error investigation is used to alter model

attention interpretation is mixed into residual diagnosis.
```

---

# 156. Common mistakes

## 156.1 Dùng `y_pred - y_true` trong một số bảng

Sai. Convention toàn project là:

```text
y_true - y_pred.
```

## 156.2 Residual dương nhưng gọi overprediction

Sai. Residual dương = underprediction.

## 156.3 Gộp 3 seeds thành 3N residual độc lập

Sai dependence interpretation.

## 156.4 Thấy residual mean = +x rồi trừ x khỏi prediction

Sai post-Test correction.

## 156.5 ACF qua timestamp gap

Sai temporal analysis.

## 156.6 Ljung–Box chạy trên chuỗi bị đứt nhưng coi như contiguous

Sai.

## 156.7 Shapiro/Jarque-Bera reject rồi kết luận model thất bại

Sai. Normal residual không phải requirement.

## 156.8 Chia Test thành high/low theo Test quantile ở đây

Sai. Phase50 dùng Train-derived regime thresholds.

## 156.9 Lấy top 20 absolute residual rồi phân tích chi tiết

Để Phase51.

## 156.10 Seed có mean residual gần 0 nhất thì chọn seed đó

Sai best-seed selection.

---

# 157. Recommended report structure

`residual_analysis_report.md`:

```text
1. Objective
2. Residual definition and sign convention
3. Frozen source verification
4. Residual/metric reconstruction audit
5. Residual distribution
6. Signed bias and sign balance
7. Tail asymmetry
8. Temporal residual trajectory
9. Gap-safe residual autocorrelation
10. Ljung–Box secondary diagnostic
11. Under/overprediction sign runs
12. Rolling 24h residual mean
13. Rolling 24h residual variability
14. Residual-vs-prediction structure
15. Prediction-decile residual spread
16. Cross-seed residual agreement
17. Cross-seed residual sign consensus
18. Baseline residual context
19. Main findings
20. Handoff to error-by-regime analysis
21. Handoff to worst-error analysis
22. Limitations
23. Definition of Done
```

---

# 158. README requirements

`README_RESIDUAL_ANALYSIS.md` explains:

```text
why residual = y_true-y_pred
why positive means underprediction
why source predictions are not rerun
why all three seeds remain
why residual normality is not required
why ACF must be gap-safe
why rolling window is fixed 24h
why prediction deciles are only diagnostics
why Test-derived regimes are forbidden
why no residual correction is allowed
how Phase50/51 consume outputs.
```

---

# 159. Summary artifact

`residual_analysis_summary.json`:

```text
version
source_phase47_version
source_phase48_version
final_lock_sha256
test_population_sha256
seed_list
source_prediction_sha256s
residual_definition
metric_reconstruction_status
distribution_summary
bias_summary
tail_summary
acf_key_lags
ljung_box_status
sign_run_summary
rolling_bias_summary
rolling_variance_summary
magnitude_association_summary
prediction_decile_summary
seed_pairwise_summary
seed_sign_consensus_summary
baseline_context
findings
new_inference=false
best_seed_selected=false
prediction_corrected=false
model_retrained=false
phase50_ready
phase51_ready
overall_status
```

---

# 160. Phase49 sign-off

`phase_49_signoff.json` minimum:

```text
phase=49
phase_name=Residual analysis
version=RESIDUAL_ANALYSIS-v1
source_phase47_version
source_phase48_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
seed42_prediction_sha256
seed123_prediction_sha256
seed2026_prediction_sha256
residual_definition=Y_TRUE_MINUS_Y_PRED
residual_recomputation_verified
metric_reconstruction_verified
distribution_analysis_complete
bias_analysis_complete
tail_analysis_complete
acf_analysis_complete
sign_run_analysis_complete
rolling_analysis_complete
magnitude_association_complete
seed_agreement_complete
seed_sign_consensus_complete
new_inference=false
model_training=false
best_seed_selected=false
prediction_correction_applied=false
test_derived_regime_threshold_used=false
worst_error_ranking_performed=false
phase50_ready
phase51_ready
warnings
overall_status
created_at
```

---

# 161. Recommended execution pseudocode

```text
p48 = load_phase48_signoff()
assert p48.overall_status in {"PASS","PASS_WITH_WARNING"}

sources = load_frozen_phase47_prediction_bundles(
    seeds=[42,123,2026]
)

verify_prediction_checksums(sources)
verify_same_test_population(sources)

rows = []

for seed, df in sources.items():

    residual = df.y_true_wh - df.y_pred_wh
    abs_error = abs(residual)
    sq_error = residual ** 2

    verify_against_stored_fields(
        df,
        residual,
        abs_error,
        sq_error
    )

    verify_metrics_against_phase47(
        seed,
        mae=mean(abs_error),
        rmse=sqrt(mean(sq_error)),
        r2=compute_r2(df.y_true_wh, df.y_pred_wh)
    )

    sign = classify_sign_exact(residual)

    rows.append(
        build_residual_rows(
            df,
            seed,
            residual,
            abs_error,
            sq_error,
            sign
        )
    )

residual_long = concat(rows)
residual_wide = pivot_residual_wide(residual_long)

distribution = compute_residual_distribution_stats(
    residual_long,
    by="seed"
)

bias = compute_signed_bias_and_sign_balance(
    residual_long
)

tails = compute_positive_negative_tail_summary(
    residual_long
)

acf = compute_gap_safe_residual_acf(
    residual_long,
    lags=range(1,145),
    exact_cadence_minutes=10
)

ljung_box = run_secondary_ljung_box_only_if_applicable(
    residual_long,
    lags=[6,36,144]
)

sign_runs = compute_gap_safe_sign_runs(
    residual_long,
    break_on_zero=True,
    break_on_gap=True
)

sign_transitions = compute_gap_safe_sign_transitions(
    residual_long
)

rolling = compute_rolling_residual_diagnostics(
    residual_long,
    window_steps=144,
    require_full_continuity=True
)

magnitude_assoc = compute_spearman_associations(
    residual_long,
    pairs=[
        ("abs_residual","prediction"),
        ("abs_residual","y_true")
    ]
)

prediction_deciles = compute_fixed_prediction_decile_diagnostics(
    residual_long,
    n_bins=10
)

seed_pairwise = compute_pairwise_residual_agreement(
    residual_wide
)

seed_consensus = compute_seed_sign_consensus(
    residual_wide
)

cross_seed_summary = summarize_seed_level_residual_statistics(
    distribution,
    bias,
    acf,
    sign_runs
)

baseline_context = build_optional_frozen_baseline_residual_context()

generate_residual_figures()

write_phase50_handoff(
    residual_long=residual_long,
    require_train_only_thresholds=True
)

write_phase51_handoff(
    residual_long=residual_long,
    no_worst_ranking_done=True
)

assert no_new_test_inference
assert no_best_seed_selection
assert no_prediction_correction
assert no_model_retraining

signoff_phase49()
```

---

# 162. Definition of Done

\[
\boxed{
Frozen\ Phase47\ Predictions
+
Verified\ Residuals
+
Metric\ Reconstruction
+
Distribution
+
Bias
+
Tails
+
Gap\text{-}Safe\ ACF
+
Sign\ Runs
+
Rolling\ Residual\ Structure
+
Variance\ Diagnostics
+
Cross\text{-}Seed\ Agreement
+
No\ Correction
+
No\ Retuning
+
Phase50/51\ Handoffs
}
\]

---

# 163. Final status contract

```text
PHASE 49 analyzes residuals only.

Residual:
y_true - y_pred.

Positive:
underprediction.

Negative:
overprediction.

Source:
frozen Phase47 prediction bundles.

Seeds:
42
123
2026
all retained.

Required:
recompute residual
verify Phase47 metrics
distribution summary
MBE/median bias
under/overprediction balance
tail asymmetry
gap-safe residual ACF
optional contiguous Ljung–Box
sign-run persistence
24h rolling residual mean/std
residual magnitude vs prediction level
prediction-decile diagnostic
cross-seed residual agreement
cross-seed sign consensus.

Forbidden:
new Test inference
best-seed selection
3N iid interpretation
bias correction
residual post-processing model
retraining
Test-derived target regimes
worst-error deep dive
attention analysis.

Phase50:
uses Train-derived regimes.

Phase51:
handles worst errors.

After RESIDUAL_ANALYSIS-v1 PASS:
proceed to
PHASE 50 — Error-by-Regime Analysis.
```

---

# 164. Final check

Correct:

```text
verify frozen predictions
→ recompute residuals
→ reconstruct Phase47 metrics
→ analyze distribution/bias/tails
→ analyze gap-safe temporal dependence
→ analyze sign persistence
→ analyze rolling variance
→ analyze residual-vs-prediction structure
→ analyze cross-seed residual agreement
→ Phase50/51 handoffs
```

Incorrect:

```text
residual mean positive
→ subtract it from predictions
→ recompute Test RMSE
```

Incorrect:

```text
find strongest residual lag
→ retrain/lookback tune
```

Incorrect:

```text
use Test residual quantiles
→ define final regimes
```

Incorrect:

```text
choose seed with smallest residual bias
```

Chỉ sau khi:

```text
RESIDUAL_ANALYSIS-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase50_ready = true
```

mới chuyển sang **PHASE 50 — Error-by-Regime Analysis**.
