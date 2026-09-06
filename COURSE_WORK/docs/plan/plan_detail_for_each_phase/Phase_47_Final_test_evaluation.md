# PHASE 47 — FINAL TEST EVALUATION

## Kế hoạch mở Test lần đầu, đánh giá ba Final Transformer seeds, Persistence và frozen LSTM baseline mà không tạo Test-driven selection

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Upstream final runs:** `THREE_SEED_FINAL_RUNS-v1`  
**Final lock:** `FINAL_MODEL_LOCK-v1`  
**Phase ID:** `PHASE_47_FINAL_TEST_EVALUATION`  
**Output version:** `FINAL_TEST_EVAL-v1`  
**Phase trước:** `Phase_46_Three-seed_final_runs.md`  
**Phase sau:** `Phase_48_Prediction_analysis.md`

---

# 1. Vai trò của Phase 47

Phase 47 là **lần đầu tiên Test target values được phép truy cập** trong toàn bộ pipeline.

Đây là final held-out evaluation của Transformer sau khi:

```text
model family
candidate configuration
features
preprocessing
lookback
optimizer
loss
RevIN
boundary protocol
final epoch count
final seeds
final scalers
```

đều đã được khóa trước Test.

Mục tiêu:

```text
1. Verify Phase46 Test release.
2. Freeze final Test evaluation contract trước khi đọc Test y.
3. Materialize exact locked Test population.
4. Reuse FINAL_SCALING-v1 without any refit.
5. Strict-load all three FINAL_REFIT Transformer checkpoints.
6. Evaluate seed42, seed123, seed2026 on exactly the same Test IDs.
7. Compute per-seed MAE/RMSE/R² in raw Wh.
8. Aggregate seed-level metrics without choosing a "best seed".
9. Evaluate Persistence on exactly the same Test IDs.
10. Evaluate the frozen tuned LSTM checkpoint only if it passes the pre-Test eligibility gate; never retrain it here.
11. Produce immutable prediction bundles for Phase48–57.
12. Record first Test access and preserve complete provenance.
13. Forbid all Test-driven model/config/seed changes.
```

Nguyên tắc:

\[
\boxed{
One\ Untouched\ Test
+
Three\ Frozen\ Transformer\ Seeds
+
Same\ Test\ Targets
+
Frozen\ Scalers
+
Inference\ Only
+
No\ Selection
+
No\ Retraining
}
\]

---

# 2. Phase47 là evaluation-only

Hard:

```text
training = forbidden
backward = forbidden
optimizer step = forbidden
scaler fit = forbidden
early stopping = forbidden
checkpoint selection = forbidden
seed selection = forbidden
hyperparameter change = forbidden
```

Phase47 chỉ được:

```text
load
transform
infer
inverse-transform
compute metrics
persist predictions
report.
```

---

# 3. Test release từ Phase46 là hard gate

Required:

```text
phase47_test_release.json
```

with:

```text
released = true.
```

Also require:

```text
phase_46_signoff.overall_status
∈ {PASS, PASS_WITH_WARNING}

ready_for_phase47 = true.
```

Nếu không:

```text
STOP.
```

---

# 4. Required Phase46 artifacts

Load:

```text
phase_46_signoff.json
phase47_test_release.json
phase47_final_test_handoff.json
three_seed_checkpoint_manifest.csv
three_seed_checkpoint_metadata_audit.csv
three_seed_checkpoint_reload_tests.csv
three_seed_config_consistency_audit.csv
final_scaler_checksums.json
final_dev_population_manifest.json
three_seed_summary.json
```

---

# 5. Required Phase45 artifacts

Load:

```text
final_model_scientific_config.json
final_feature_contract.json
final_preprocessing_contract.json
final_boundary_contract.json
final_revin_contract.json
final_loss_contract.json
final_checkpoint_contract.json
final_model_lock_fingerprint.json
phase47_test_evaluation_guard.json
```

---

# 6. Test evaluation contract phải được frozen trước first Test y access

Before materializing any Test target value, write:

```text
final_test_evaluation_contract.json
```

and checksum it.

Contract must freeze:

```text
Test target population rule
three seed checkpoint IDs
metrics
seed aggregation rule
Persistence baseline rule
LSTM eligibility rule
no ensemble
no best-seed selection
prediction bundle schema
first-access logging
technical rerun policy
downstream artifact schema.
```

---

# 7. First Test access event

Create a single auditable event:

```text
TEST_FIRST_ACCESS_EVENT
```

with:

```text
phase = 47
reason = FINAL_HELD_OUT_EVALUATION
lock hash
evaluation contract hash
timestamp
authorized checkpoints
authorized models
authorized metrics.
```

Once this event occurs:

```text
scientific config becomes permanently non-editable.
```

---

# 8. No post-Test amendment for performance improvement

After first Test access, a model/config change cannot be justified as:

```text
“final tuning”
```

without invalidating Test as untouched final holdout.

If a genuine implementation bug is discovered after Test access:

```text
document contamination/invalidity
do not silently fix and re-report as untouched Test.
```

---

# 9. Models authorized for Test

Authorized primary models:

```text
FINAL_TR_SEED42
FINAL_TR_SEED123
FINAL_TR_SEED2026
```

Authorized baselines:

```text
PERSISTENCE_LAST_VALUE
LSTM_TUNED_DEV
```

where `LSTM_TUNED_DEV` is allowed only if eligibility gate passes.

No other candidate may be evaluated on Test in Phase47.

---

# 10. Transformer candidates not selected by Phase44 are forbidden on Test

Do not evaluate:

```text
TR_C0
TR_C1
TR_C2
```

as a leaderboard after Phase45 unless they are the exact final locked candidate checkpoints from Phase46.

Testing multiple rejected Transformer candidates would turn Test into a selection set.

---

# 11. Three Transformer seeds are not candidate alternatives

The three seeds are predeclared repeated materializations of one locked scientific configuration.

All three must be evaluated.

No seed is preferred before evaluation.

No seed is dropped after evaluation.

---

# 12. Final Test population ID

Define:

```text
FINAL_TEST_POP-v1
```

from the locked:

```text
WINDOWPOP-v1
Test target IDs
final candidate compatibility
WB0
H1
continuity rules.
```

Use the exact population policy carried by Phase45/46.

---

# 13. No Test population optimization

Do not alter Test population because:

```text
a target is difficult
a prediction is extreme
a model returns large error
a baseline lacks a value.
```

Eligibility must be determined before predictions are inspected.

---

# 14. Test target IDs must be common across all Transformer seeds

Hard:

```text
seed42 Test IDs
=
seed123 Test IDs
=
seed2026 Test IDs
=
FINAL_TEST_POP-v1.
```

---

# 15. Persistence uses the exact same Test target IDs

Hard.

---

# 16. LSTM baseline common-target requirement

If `LSTM_TUNED_DEV` is evaluated:

```text
its Test prediction bundle must use the exact same FINAL_TEST_POP-v1 target IDs
```

or it is not included in the direct common-target metric table.

Do not silently evaluate LSTM on a different set.

---

# 17. Final Test population materialization

Before model inference:

```text
materialize target IDs
verify uniqueness
verify chronological ordering
verify split membership = Test
verify continuity/window eligibility
compute target-ID SHA256
```

Compare with locked expected metadata.

---

# 18. Test population count is runtime output

Do not fabricate.

Persist actual:

```text
N_test
first Test target timestamp
last Test target timestamp
target-ID fingerprint.
```

---

# 19. WB0 semantics at Test boundary

Primary protocol:

```text
WB0 = context carry-over.
```

Therefore earliest Test targets may use preceding observed history from:

```text
Train+Validation
```

as long as all input timestamps precede the target.

This is not future leakage.

---

# 20. WB0 semantics inside Test

For later one-step Test targets, input may contain:

```text
actual observed Appliances values
from earlier Test timestamps
```

because the task assumes rolling one-step forecasting where past observations become known over time.

---

# 21. No recursive prediction feedback

Hard:

```text
previous model prediction
!=
historical observed Appliances.
```

For target `t+1`, historical target input uses:

```text
actual observed y_t
```

if the selected feature set includes historical Appliances.

Do not feed:

```text
ŷ_t
```

as if it were observation.

---

# 22. No future Test target in an input window

For every sample:

```text
max(input_timestamp) < target_timestamp.
```

Hard.

---

# 23. Test feature availability

Use only features available in the historical input rows.

No future weather/exogenous values beyond the input sequence.

No target-time future covariate injection.

---

# 24. Same final feature order

Load exact:

```text
final_feature_contract.json.
```

No column auto-sort.

No re-discovery.

---

# 25. Same final lookback

Hard:

```text
L = locked final L*.
```

No padding.

No shorter boundary windows.

---

# 26. Same target horizon

Hard:

```text
H=1.
```

---

# 27. Same deterministic time-feature formulas

If active:

```text
hour_sin
hour_cos
dow_sin
dow_cos
weekend.
```

No fitting.

---

# 28. Final scalers are inference-only

Load exact:

```text
FINAL_SCALING-v1
```

from Phase46.

Hard:

```text
fit = forbidden
partial_fit = forbidden
update = forbidden.
```

---

# 29. Scaler checksum verification

Before Test transform:

```text
verify X scaler bundle SHA256
verify Y scaler SHA256/identity
verify feature mapping checksum.
```

Must match all three final checkpoint metadata.

---

# 30. No Test normalization statistics for global scaler

Forbidden:

```text
fit StandardScaler on Test
recenter Test globally
normalize Test using Test mean/std.
```

---

# 31. RevIN is not global Test fitting

If RN1 active:

```text
per-window instance statistics
```

are computed from the current historical input window only.

This remains allowed because it is part of the locked model forward semantics.

---

# 32. RevIN Test safety

If RN1:

```text
mean/std only from X window
no future target
no full-Test statistics
same channel scope
same FINAL_SCALING X→raw→Y bridge.
```

---

# 33. Transformer checkpoint verification before first inference

For each seed:

```text
checkpoint SHA256
lock hash
config hash
recipe hash
population policy
feature fingerprint
scaler checksums
official epoch
checkpoint type
state schema
```

must match Phase46.

---

# 34. Official checkpoint type

Hard:

```text
FINAL_REFIT.
```

Do not accidentally load:

```text
Phase44 checkpoint
BEST checkpoint
recovery checkpoint
LAST from another run.
```

---

# 35. Strict load

For every seed:

```text
strict=True
missing_keys=0
unexpected_keys=0.
```

---

# 36. Test inference mode

Hard:

```text
model.eval()
torch.inference_mode()
```

No dropout sampling.

No Monte Carlo dropout.

---

# 37. Standard forward path only for official predictions

Official Test predictions use:

```text
standard model forward
```

without attention extraction to reduce memory and preserve the verified path.

Attention is analyzed later from the same checkpoints.

---

# 38. No attention-based seed interpretation in Phase47

No attention heatmaps.

No head analysis.

No seed selection based on attention.

---

# 39. One official inference pass per Transformer seed

Preferred:

```text
one complete ordered pass over FINAL_TEST_POP-v1
```

per checkpoint.

Persist predictions immediately.

---

# 40. Technical inference rerun

Allowed only if:

```text
process interruption
artifact write failure
hardware/runtime failure
```

and must use:

```text
same checkpoint
same Test IDs
same scalers
same code
same deterministic eval mode.
```

Never choose the more favorable output.

---

# 41. Deterministic eval rerun equivalence

If an inference pass is rerun for verification:

```text
predictions should be allclose/identical under the environment contract.
```

Mismatch:

```text
STOP and investigate.
```

---

# 42. Transformer prediction bundle schema

For each seed:

```text
seed
run_id
checkpoint_sha256
target_id
target_timestamp
y_true_wh
y_pred_wh
residual_wh
absolute_error_wh
squared_error_wh2
```

Residual convention:

```text
residual = y_true - y_pred.
```

Positive residual:

```text
underprediction.
```

---

# 43. Prediction bundle is immutable downstream source

After verified write/checksum:

```text
Phase48–51
```

must read prediction bundles rather than rerunning Test inference unless absolutely required for a separate registered artifact.

This reduces repeated Test interaction.

---

# 44. No metric computed from rounded predictions

Persist full precision.

Metrics use full-precision arrays.

---

# 45. Shared metric implementation

Use exact:

```text
METRICS-v1
```

with:

```text
MAE Wh
RMSE Wh
R².
```

---

# 46. Per-seed metrics

For each Transformer seed compute:

```text
Test MAE Wh
Test RMSE Wh
Test R²
N.
```

No seed ranking is used for selection.

---

# 47. Final primary Transformer reporting statistic

The primary multi-seed final result is:

```text
mean ± sample SD
```

across the three **per-seed Test metrics**.

For metric `m`:

\[
\bar m
=
\frac{1}{3}\sum_{s=1}^{3}m_s
\]

and sample SD:

\[
SD(m)
=
\sqrt{
\frac{
\sum_s(m_s-\bar m)^2
}{
3-1
}
}
\]

Use:

```text
ddof=1.
```

---

# 48. Why mean ± SD across seeds

The three checkpoints represent stochastic realizations of the same locked scientific model.

Reporting all three plus:

```text
mean ± SD
```

shows random-seed sensitivity without selecting the best realization.

---

# 49. Do not pool seed-target pairs as the primary score

Forbidden primary metric:

```text
concatenate seed42 + seed123 + seed2026 predictions
as 3N independent observations
→ compute one RMSE.
```

This treats repeated predictions for the same Test targets as independent data.

If ever computed as a diagnostic:

```text
label clearly NON_PRIMARY
```

but preferred:

```text
do not compute.
```

---

# 50. No seed-averaged prediction ensemble

Do not compute:

\[
\hat y_{ensemble}
=
(\hat y_{42}+\hat y_{123}+\hat y_{2026})/3
\]

as an official final model.

Ensembling was not locked before Test.

---

# 51. No best-seed reporting as final model

You may list per-seed values.

Do not say:

```text
Final model = seed with minimum Test RMSE.
```

That is Test-driven seed selection.

---

# 52. Min/max seed metric may be reported descriptively

Allowed:

```text
range
min
max
```

but clearly:

```text
descriptive variability
not selection.
```

---

# 53. Median seed metric

Optional descriptive only.

Primary remains:

```text
per-seed values + mean ± sample SD.
```

---

# 54. R² aggregation nuance

Report:

```text
mean ± SD of per-seed R²
```

not R² from averaged predictions.

---

# 55. RMSE/R² ranking consistency

Because all seeds share the same exact `y_true` array:

```text
lower SSE/RMSE
must correspond to
higher R².
```

If per-seed ordering conflicts:

```text
STOP
METRIC_OR_POPULATION_BUG.
```

MAE may rank differently.

---

# 56. Persistence baseline evaluation

For every target ID `j`:

\[
\hat y_j=y_{j-1}.
\]

Use raw Wh.

No scaler.

No training.

No seed.

---

# 57. Persistence WB0 boundary

For first Test target:

```text
y_{j-1}
```

may belong to Validation.

For later targets it may belong to earlier Test timestamps.

All are historical observations at forecast time.

---

# 58. Persistence exact target population

Use:

```text
FINAL_TEST_POP-v1.
```

Hard.

---

# 59. Persistence metrics

Compute:

```text
MAE Wh
RMSE Wh
R²
```

using METRICS-v1.

---

# 60. LSTM Test eligibility gate

Phase47 must not train/refit LSTM.

A Test LSTM baseline is allowed only if, before first Test access, there is a frozen checkpoint that satisfies:

```text
source = Phase43 LSTM_TUNED
checkpoint selected using development data only
Test never used
config frozen
strict-load valid
common FINAL_TEST_POP-v1 supported
scalers/data protocol resolvable
no post-Test tuning.
```

Canonical label:

```text
LSTM_TUNED_DEV.
```

---

# 61. Important LSTM fairness caveat

`LSTM_TUNED_DEV` from Phase43 was tuned/selected on the original development split and was not materialized as a Train+Validation final refit in Phase46.

Therefore if evaluated on Test:

```text
it is a valid frozen Test baseline
```

but it is **not training-budget/data-refit symmetric** with the final Transformer.

Report this explicitly.

---

# 62. Do not retrain LSTM after Test release

Forbidden:

```text
Test is now open
→ train LSTM on Train+Validation
→ compare.
```

Even if Test predictions were not yet inspected, training after Test authorization weakens separation and violates the current pipeline.

---

# 63. If a fair final-refit LSTM is desired

It must have been frozen/materialized before first Test access under a pre-Test protocol.

Current Phase47 plan does not create one.

---

# 64. LSTM eligibility statuses

Use:

```text
ELIGIBLE_FROZEN_DEV_BASELINE
NOT_ELIGIBLE_CHECKPOINT_MISSING
NOT_ELIGIBLE_CONFIG_MISMATCH
NOT_ELIGIBLE_POPULATION_MISMATCH
NOT_EVALUATED_BY_PROTOCOL.
```

---

# 65. If LSTM is ineligible

Do not fabricate Test LSTM metrics.

Use:

```text
Phase44 rolling-origin LSTM comparison
```

as the robustness comparison source.

Final Test comparison table marks:

```text
LSTM Test = N/A
```

with reason.

---

# 66. LSTM scaler handling

If `LSTM_TUNED_DEV` is evaluated, use the exact scaler artifacts associated with its frozen Phase43 checkpoint.

Do **not** reuse Transformer final scalers unless the LSTM checkpoint was trained with those exact artifacts.

---

# 67. Why LSTM may need different scaler artifact

A checkpoint must be evaluated in the same model-space transform used during its training.

Using a different scaler changes model semantics.

---

# 68. LSTM Test target IDs still must match

Different scaler artifacts are allowed.

Different target IDs are not allowed in the direct common-target table.

---

# 69. LSTM inference-only

Hard:

```text
strict load
eval
inference mode
no optimizer
no refit.
```

---

# 70. Baseline comparison scopes

Create two clearly separated comparison layers:

```text
A. FINAL TEST comparison
B. PRE-TEST ROLLING-ORIGIN comparison
```

Do not mix metrics in one unlabeled ranking.

---

# 71. Final Test comparison table

Recommended rows:

```text
Transformer — Seed42
Transformer — Seed123
Transformer — Seed2026
Transformer — Mean±SD summary
Persistence
LSTM_TUNED_DEV if eligible
```

---

# 72. Rolling-origin context table

Carry from Phase44:

```text
recommended Transformer
LSTM_TUNED
Persistence
```

with pooled pre-Test rolling metrics.

Label:

```text
development robustness evidence
```

not Test.

---

# 73. Transformer vs Persistence per-seed deltas

For each seed:

\[
\Delta RMSE_{P\rightarrow T_s}
=
RMSE_P-RMSE_{T_s}
\]

Positive:

```text
Transformer seed improves over Persistence.
```

Also compute MAE/R² differences as descriptive.

---

# 74. Transformer vs LSTM per-seed deltas

If LSTM eligible:

\[
\Delta RMSE_{LSTM\rightarrow T_s}
=
RMSE_{LSTM}-RMSE_{T_s}.
\]

Positive:

```text
Transformer seed lower RMSE.
```

---

# 75. Mean baseline delta across Transformer seeds

Allowed:

```text
mean of three per-seed RMSE deltas
sample SD of deltas.
```

No significance claim.

---

# 76. Baseline beat-count

Optional descriptive:

```text
number of Transformer seeds with lower RMSE than Persistence
number with lower RMSE than LSTM.
```

Do not turn this into statistical test.

---

# 77. No significance testing across three seeds

Do not report:

```text
p-value
confidence superiority
statistically significant
```

based only on three seeds.

---

# 78. No t-test treating Test timestamps as independent iid without time-series correction

Not part of Phase47.

Avoid.

---

# 79. No Diebold-Mariano test in Phase47 by default

Could be a separate registered analysis with dependence-aware design, but is not needed for coursework.

Do not add after seeing outcomes.

---

# 80. Test metric precision

Store full precision.

Human-readable tables may round for presentation, but retain underlying values.

Recommended final report display:

```text
MAE/RMSE: 2–3 decimals depending scale
R²: 3–4 decimals
```

while selection/inference uses full precision.

---

# 81. No threshold-based pass/fail on Test score

A low or disappointing Test score does not invalidate the methodology.

Do not retrain because:

```text
Test RMSE worse than expected.
```

---

# 82. Test performance can be worse than Validation

This may reflect:

```text
temporal shift
harder Test period
selection optimism
stochastic variation.
```

Report, do not fix.

---

# 83. Test performance can be worse than Persistence

Report honestly.

This is scientifically meaningful for 10-minute-ahead energy forecasting.

---

# 84. Test LSTM can outperform Transformer

Report honestly.

Do not change final Transformer checkpoint after seeing it.

---

# 85. Final performance statement

Safe:

> Across the three predeclared final seeds, the locked Transformer achieved a mean Test RMSE of X Wh with a sample standard deviation of Y Wh.

Do not fill X/Y until runtime.

---

# 86. Seed-specific statement

Safe:

> The three seed-specific Test RMSE values were ..., reflecting stochastic variation under the same locked configuration.

No “winner seed”.

---

# 87. No final model retraining after Test

Hard.

Phase47 closes the model-development loop.

---

# 88. Prediction bundle checks before metrics

For each bundle:

```text
N matches FINAL_TEST_POP-v1
target IDs exact
order exact
timestamps exact
y_true finite
y_pred finite
no duplicates.
```

---

# 89. Cross-seed y_true equality

Hard bitwise/allclose:

```text
y_true_seed42
=
y_true_seed123
=
y_true_seed2026.
```

---

# 90. Cross-seed target timestamp equality

Hard.

---

# 91. Persistence y_true equality

Hard.

---

# 92. LSTM y_true equality

Hard if included.

---

# 93. Target truth source

Use raw `Appliances` Wh from Test target rows.

Do not inverse-transform a scaler-transformed `y_true` if raw truth is already available unless exact pipeline requires it; whichever path is used must match.

Preferred:

```text
raw y_true_wh directly from locked target row
```

for metric truth.

---

# 94. Transformer prediction inverse transformation

Model outputs:

```text
y_model.
```

Convert using exact final Y transform to:

```text
y_pred_wh.
```

If RN1 active, model forward already returns y_model after internal target bridge according to lock.

---

# 95. YS0

If YS0:

```text
y_model == raw Wh coordinate
```

under locked semantics.

Still pass through standardized evaluation adapter to avoid branch-specific metric bugs.

---

# 96. Numerical checks

Before metric:

```text
all y_true finite
all y_pred finite
all residual finite
all abs error finite
all squared error finite.
```

Any failure:

```text
FAIL.
```

Do not drop bad samples.

---

# 97. No clipping of predictions

Do not force negative/high predictions into plausible range unless such clipping was locked pre-Test, which current protocol does not include.

---

# 98. Negative predictions

If model produces negative energy:

```text
retain
record
```

for honest error analysis.

Do not post-process after seeing Test.

---

# 99. No outlier removal

No Test row removed because residual is large.

Worst errors are analyzed later in Phase51.

---

# 100. No target regime stratification in Phase47

Keep Phase47 focused on global final metrics.

Detailed:

```text
regime
residual
worst error
```

analysis belongs to Phase49–51.

---

# 101. Limited descriptive Test diagnostics allowed

Phase47 may record:

```text
prediction min/max/mean
residual min/max/mean
```

as integrity diagnostics, but do not expand into exploratory tuning.

---

# 102. Avoid broad Test EDA before metric lock

Metrics and evaluation contract must be frozen before Test inspection.

---

# 103. Metric formulas

MAE:

\[
MAE
=
\frac{1}{N}\sum_i|y_i-\hat y_i|.
\]

RMSE:

\[
RMSE
=
\sqrt{
\frac{1}{N}
\sum_i(y_i-\hat y_i)^2
}.
\]

R²:

\[
R^2
=
1-
\frac{
\sum_i(y_i-\hat y_i)^2
}{
\sum_i(y_i-\bar y)^2
}.
\]

---

# 104. R² can be negative

Do not clamp.

---

# 105. No MAPE as primary

MAPE may be unstable near low values and is not part of locked primary metrics.

Do not add as a selection metric.

---

# 106. Optional MAPE

If already supported as optional descriptive metric by METRICS-v1, may be computed only if explicitly marked:

```text
OPTIONAL
NON_PRIMARY.
```

Preferred final coursework table remains:

```text
MAE
RMSE
R².
```

---

# 107. Seed aggregate artifact

Create:

```text
transformer_seed_aggregate_metrics.csv
```

with one row per metric:

```text
metric
seed42
seed123
seed2026
mean
sample_sd
min
max
range
```

No rank.

---

# 108. Per-seed metrics artifact

Create:

```text
final_test_metrics_by_seed.csv
```

with:

```text
seed
run_id
checkpoint_sha256
N
MAE_Wh
RMSE_Wh
R2
target_population_sha256
status.
```

---

# 109. Baseline metrics artifact

Create:

```text
final_test_baseline_metrics.csv
```

with:

```text
model_id
eligibility
N
MAE_Wh
RMSE_Wh
R2
checkpoint/run reference
population_sha256
notes
status.
```

---

# 110. Comparison artifact

Create:

```text
final_test_model_comparison.csv
```

Rows should distinguish:

```text
TRANSFORMER_SEED
TRANSFORMER_AGGREGATE
PERSISTENCE
LSTM_DEV_BASELINE.
```

No fake single Transformer aggregate prediction.

---

# 111. Aggregate row semantics

For Transformer aggregate row:

```text
MAE = mean of 3 per-seed MAEs
RMSE = mean of 3 per-seed RMSEs
R² = mean of 3 per-seed R²s
```

with separate SD columns.

Do not compute from averaged predictions.

---

# 112. Baseline deltas artifact

Create:

```text
final_test_baseline_deltas.csv
```

with per-seed and aggregate descriptive deltas.

---

# 113. Test access log

Create:

```text
final_test_access_log.jsonl
```

Every authorized Test action:

```text
materialize targets
build loader
run seed42 inference
run seed123 inference
run seed2026 inference
run persistence
run eligible LSTM
compute metrics
```

gets logged.

This is governance evidence.

---

# 114. No hidden Test notebook exploration

All Test interactions should go through registered Phase47 functions/scripts where practical.

Avoid ad hoc cells that inspect labels manually.

---

# 115. Test loader contract

Transformer Test loader:

```text
shuffle=false
drop_last=false
no stochastic augmentation
same feature/window implementation
same final scalers
ordered target IDs.
```

Batch size for inference may use locked B* or a deterministic memory-safe inference batch size if architecture outputs are unchanged.

Preferred:

```text
use locked B*
```

to minimize execution variation.

---

# 116. Inference batch-size changes

If memory requires a different Test batch size:

```text
predictions should be batch-independent in eval mode.
```

But since no need is expected, keep:

```text
B*
```

unless a documented technical constraint exists.

Do not change model scientific config.

---

# 117. Batch-independence probe

Before official full Test inference, use a **pre-Test probe**, not Test, to verify:

```text
prediction(sample A alone)
≈
prediction(sample A within batch).
```

Already tested upstream; Phase47 can rely on Phase46 evidence.

No need to use Test for this.

---

# 118. Official inference order

Recommended:

```text
1. seed42
2. seed123
3. seed2026
4. Persistence
5. eligible LSTM
```

But no model result may change subsequent authorized model set.

---

# 119. Do not stop after seed42 result

All three seeds must be evaluated regardless of seed42 Test score.

---

# 120. Do not stop baselines after Transformer result

If baselines are pre-authorized/eligible, evaluate them regardless of Transformer score.

---

# 121. No candidate addition after seeing baseline result

No.

---

# 122. Prediction bundle checksums

Every final Test prediction CSV/Parquet should receive:

```text
SHA256
```

and immutable status.

---

# 123. Recommended storage format

CSV is acceptable for coursework/provenance.

If Parquet used internally, also generate human-inspectable CSV summary if useful.

Do not change numerical precision on export.

---

# 124. Downstream Phase48 source of truth

Phase48 should consume:

```text
final_test_predictions_seed42.csv
final_test_predictions_seed123.csv
final_test_predictions_seed2026.csv
```

plus aggregate metric artifacts.

---

# 125. Which predictions should Phase48 analyze

Since no best seed is selected, prediction analysis should preserve all three seeds.

Recommended downstream:

```text
per-seed
+
seed-mean prediction as a descriptive statistic only if clearly not an ensemble model.
```

Phase48 will define exact visualization semantics.

---

# 126. Seed-mean prediction for analysis vs ensemble

Computing:

```text
mean prediction across seeds
```

may be useful as a descriptive central tendency for plots later.

It must never be labeled:

```text
ensemble Test model
```

unless a pre-Test ensemble protocol existed, which it does not.

Phase47 need not generate this field.

---

# 127. Error analysis source

Residual columns are persisted per seed.

Phase49 can analyze distributions without rerunning inference.

---

# 128. Regime analysis source

Target timestamps + raw truth support Phase50.

---

# 129. Worst-error analysis source

Full prediction bundles support Phase51.

---

# 130. Attention analysis source

Phase52 uses:

```text
same final checkpoints
same FINAL_TEST_POP-v1 IDs
same final scalers
```

but attention extraction is a later registered Test analysis, not part of Phase47 prediction metric selection.

---

# 131. Attention does not change Test performance metric

No.

---

# 132. Test evaluation of checkpoints is irreversible in scientific sense

After Test metrics exist:

```text
do not go back to Phase45
```

to improve score while still calling Test untouched.

---

# 133. Failed methodology vs poor performance

Differentiate:

```text
methodological FAIL
```

from:

```text
scientifically valid but poor Test RMSE.
```

Poor score does not trigger pipeline rollback.

---

# 134. Runtime failure before predictions saved

A technical rerun is allowed.

No scientific change.

---

# 135. Runtime failure after partial predictions saved

Discard incomplete bundle from official metric computation.

Rerun same deterministic inference to create complete bundle.

Retain failure log.

---

# 136. Runtime failure after complete metrics

Do not rerun just because score is surprising.

---

# 137. LSTM frozen checkpoint source

Preferred:

```text
Phase43 lstm_tuned_winner.json
```

and its exact BEST checkpoint.

Label:

```text
LSTM_TUNED_DEV
```

not:

```text
LSTM_FINAL_REFIT.
```

---

# 138. LSTM checkpoint validation

Before Test:

```text
strict-load
config match
development-only provenance
Test status previously NOT_ACCESSED
scaler/checkpoint mapping valid
```

---

# 139. LSTM target-scaling consistency

Use its own frozen `YS*` and scaler artifact.

Predictions inverse to Wh before common metric comparison.

---

# 140. LSTM feature/window semantics

Must use exact Phase43:

```text
FV*
YS*
L*
WB0
```

and same final Test target IDs.

No hidden adaptation.

---

# 141. If LSTM Phase43 target IDs/data contract differ from final Test common population

The model can still generate Test predictions if its feature/window config supports those IDs.

Direct comparison is allowed only on exact common target IDs.

If exact FINAL_TEST_POP-v1 is unsupported:

```text
mark not directly comparable.
```

Do not alter Transformer population to accommodate it.

---

# 142. Persistence information-set caveat

If final Transformer excludes historical Appliances but Persistence uses it:

```text
Persistence remains a task-level naive baseline
```

with richer autoregressive information.

Carry Phase14 caveat.

---

# 143. LSTM/Transformer training-data caveat

If LSTM is Phase43 development checkpoint and Transformer is Phase46 Train+Val refit:

```text
training data usage differs.
```

This must appear in every Test comparison table note.

---

# 144. Fair comparison statement

Safe:

> Test predictions are evaluated on identical target timestamps and metrics, while the frozen LSTM development checkpoint retains its original Phase43 training protocol and therefore is not a fully symmetric Train+Validation refit comparator.

---

# 145. No retroactive final LSTM fitting

No.

---

# 146. Pre-Test rolling-origin remains the stronger model-family fairness evidence

Because Phase44 retrained both Transformer candidates and tuned LSTM under common temporal folds, use Phase44 as the primary fairness context for model-family robustness.

Phase47 LSTM Test result, if available, is supplementary held-out baseline evidence.

---

# 147. Primary final claims focus on locked Transformer

The final Test headline should emphasize:

```text
three-seed Transformer mean ± SD
```

then baselines.

---

# 148. Final Test model-comparison interpretation

If mean Transformer RMSE < Persistence:

```text
learned model improves over naive persistence on held-out Test on average across seeds.
```

If not:

```text
persistence remains stronger under this Test period.
```

No retuning.

---

# 149. If LSTM eligible and lower than Transformer mean

Safe:

> The frozen tuned LSTM development checkpoint obtained a lower Test RMSE than the mean of the three final Transformer seed-specific RMSE values; however, its final-training protocol differs from the Transformer Train+Validation refit, so the rolling-origin comparison remains the cleaner model-family robustness comparison.

---

# 150. If Transformer lower than LSTM

Same fairness caveat.

---

# 151. Seed variability interpretation

Low SD:

```text
observed Test metric is relatively stable across these three seeds.
```

High SD:

```text
stochastic sensitivity is non-negligible across the three predeclared seeds.
```

Avoid arbitrary low/high thresholds; report actual values.

---

# 152. No inferential confidence interval from n=3 seeds by default

Mean ± SD is enough.

Do not imply population coverage from three seeds.

---

# 153. Final Test artifact directory

```text
artifacts/
└── final_test/
    ├── final_test_evaluation_manifest.json
    ├── final_test_evaluation_contract.json
    ├── phase47_preflight_audit.csv
    ├── final_test_release_verification.json
    ├── final_test_access_event.json
    ├── final_test_access_log.jsonl
    ├── final_test_population_manifest.json
    ├── final_test_population_audit.csv
    ├── final_test_feature_order_audit.csv
    ├── final_test_scaler_verification.csv
    ├── final_test_checkpoint_verification.csv
    ├── final_test_lstm_eligibility.json
    ├── final_test_common_target_audit.csv
    ├── final_test_inference_manifest.csv
    ├── predictions/
    │   ├── final_test_predictions_seed42.csv
    │   ├── final_test_predictions_seed123.csv
    │   ├── final_test_predictions_seed2026.csv
    │   ├── final_test_predictions_persistence.csv
    │   └── final_test_predictions_lstm_tuned_dev.csv
    ├── prediction_checksums.json
    ├── final_test_metrics_by_seed.csv
    ├── transformer_seed_aggregate_metrics.csv
    ├── final_test_baseline_metrics.csv
    ├── final_test_model_comparison.csv
    ├── final_test_baseline_deltas.csv
    ├── final_test_metric_consistency_audit.csv
    ├── final_test_prediction_integrity_audit.csv
    ├── final_test_summary_table.csv
    ├── final_test_findings.csv
    ├── phase48_prediction_analysis_handoff.json
    ├── phase49_residual_analysis_handoff.json
    ├── phase50_error_regime_handoff.json
    ├── phase51_worst_error_handoff.json
    ├── phase52_attention_extraction_handoff.json
    ├── final_test_tests.csv
    ├── final_test_discrepancies.json
    ├── final_test_summary.json
    ├── final_test_report.md
    ├── figures/
    │   ├── TEST_47_01_rmse_by_seed_and_baselines.png
    │   ├── TEST_47_02_mae_by_seed_and_baselines.png
    │   ├── TEST_47_03_r2_by_seed_and_baselines.png
    │   └── TEST_47_04_transformer_seed_metric_variability.png
    ├── README_FINAL_TEST_EVALUATION.md
    └── phase_47_signoff.json
```

If LSTM is ineligible:

```text
do not create fake LSTM prediction file.
```

---

# 154. Required outputs

```text
O47.1  Evaluation manifest
O47.2  Evaluation contract frozen before Test
O47.3  Preflight audit
O47.4  Test-release verification
O47.5  First Test access event
O47.6  Test access log
O47.7  Test population manifest
O47.8  Test population audit
O47.9  Feature-order audit
O47.10 Scaler verification
O47.11 Checkpoint verification
O47.12 LSTM eligibility decision
O47.13 Common-target audit
O47.14 Inference manifest
O47.15 Seed42 prediction bundle
O47.16 Seed123 prediction bundle
O47.17 Seed2026 prediction bundle
O47.18 Persistence prediction bundle
O47.19 Eligible LSTM prediction bundle
O47.20 Prediction checksums
O47.21 Per-seed metrics
O47.22 Transformer aggregate metrics
O47.23 Baseline metrics
O47.24 Model comparison
O47.25 Baseline deltas
O47.26 Metric consistency audit
O47.27 Prediction integrity audit
O47.28 Final summary table
O47.29 Figures
O47.30 Findings
O47.31 Phase48 handoff
O47.32 Phase49 handoff
O47.33 Phase50 handoff
O47.34 Phase51 handoff
O47.35 Phase52 handoff
O47.36 Tests
O47.37 Discrepancies
O47.38 Summary JSON
O47.39 Human-readable report
O47.40 README
O47.41 Sign-off
```

---

# 155. Evaluation manifest

`final_test_evaluation_manifest.json`:

```text
phase = 47
version = FINAL_TEST_EVAL-v1
source_phase46_version
final_lock_sha256
evaluation_contract_sha256
authorized_transformer_seeds = [42,123,2026]
authorized_baselines = [PERSISTENCE,LSTM_TUNED_DEV_IF_ELIGIBLE]
test_population = FINAL_TEST_POP-v1
primary_metrics = [MAE_Wh,RMSE_Wh,R2]
seed_aggregation = MEAN_PLUS_SAMPLE_SD
ensemble = false
best_seed_selection = false
training = false
scaler_fit = false
first_test_access_phase = 47
status
created_at
```

---

# 156. Evaluation contract artifact

`final_test_evaluation_contract.json` minimum:

```text
locked_before_test_access = true

Models:
3 final Transformer seeds
Persistence
frozen Phase43 LSTM if eligibility passes

Test population:
locked WINDOWPOP-v1 Test IDs
WB0
H1
exact final target-ID fingerprint

Transformer inference:
FINAL_REFIT checkpoints
FINAL_SCALING-v1
eval/inference_mode
standard forward

Metrics:
MAE Wh
RMSE Wh
R²

Seed summary:
per-seed metrics
arithmetic mean
sample SD ddof=1
min/max/range descriptive

Forbidden:
best seed
ensemble
training
scaler refit
candidate testing
post-Test tuning.
```

---

# 157. Preflight audit

`phase47_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Checks:

```text
Phase46 release true
3 checkpoints available
lock/config/recipe hashes match
same population/scalers/epochs
Test guard authorizes Phase47
evaluation contract frozen
metrics frozen
seed aggregation frozen
Persistence authorized
LSTM eligibility unresolved before access
no training objects needed
status.
```

---

# 158. Test-release verification

`final_test_release_verification.json`:

```text
release_artifact_sha256
released
required_seed_count
completed_seed_count
checkpoint_sha256s
same_lock_hash
same_config_hash
same_recipe_hash
same_population_hash
same_scaler_hashes
same_epochs
test_status_before_phase47
verified
status
```

---

# 159. First Test access event artifact

`final_test_access_event.json`:

```text
event_id
phase=47
authorized=true
final_lock_sha256
evaluation_contract_sha256
first_access_timestamp
access_reason
authorized_models
authorized_metrics
scientific_config_frozen=true
post_access_tuning_forbidden=true
status
```

---

# 160. Test population manifest

`final_test_population_manifest.json`:

```text
population_id=FINAL_TEST_POP-v1
source_population_policy=WINDOWPOP-v1
split=TEST
boundary_protocol=WB0
lookback
horizon=1
target_count
first_target_id
last_target_id
first_target_timestamp
last_target_timestamp
target_ids_sha256
continuity_version
status
```

---

# 161. Test population audit

`final_test_population_audit.csv`:

```text
check
expected
observed
status
```

Checks:

```text
all targets Test
unique
chronological
valid continuity
final lookback valid
H1
no padding
same locked population
no target omitted.
```

---

# 162. Feature-order audit

`final_test_feature_order_audit.csv`:

```text
position
locked_feature
runtime_feature
same
scaling_group
RevIN_scope_if_any
status
```

---

# 163. Scaler verification

`final_test_scaler_verification.csv`:

```text
component
expected_checksum
observed_checksum
fit_called_in_phase47
mapping_match
status
```

Expected:

```text
fit_called_in_phase47=false.
```

---

# 164. Checkpoint verification table

`final_test_checkpoint_verification.csv`:

```text
seed
run_id
checkpoint_sha256
expected_checkpoint_sha256
checkpoint_type
official_epoch
lock_hash_match
config_hash_match
recipe_hash_match
population_hash_match
scaler_ref_match
strict_load
state_schema_match
status
```

---

# 165. LSTM eligibility artifact

`final_test_lstm_eligibility.json`:

```text
model_id=LSTM_TUNED_DEV
source_phase=43
source_run_id
checkpoint_exists
checkpoint_frozen_pretest
test_previously_accessed=false
config_valid
scaler_artifacts_available
FINAL_TEST_POP_supported
common_target_comparison_possible
training_protocol_symmetric_with_transformer=false
eligible
eligibility_status
reason
status
```

---

# 166. Common-target audit

`final_test_common_target_audit.csv`:

```text
model_id
target_count
target_ids_sha256
reference_sha256
same_ids
same_order
same_ytrue
status
```

---

# 167. Inference manifest

`final_test_inference_manifest.csv`:

```text
model_id
seed_if_any
checkpoint/run
population_sha256
scaler_bundle
inference_batch_size
eval_mode
inference_mode
attention_extraction
prediction_file
prediction_sha256
attempt_number
technical_rerun
status
```

---

# 168. Prediction integrity audit

`final_test_prediction_integrity_audit.csv`:

```text
model_id
N
unique_target_ids
chronological
ytrue_finite
ypred_finite
residual_finite
no_rows_dropped
population_match
status
```

---

# 169. Per-seed metrics schema

`final_test_metrics_by_seed.csv`:

```text
seed
run_id
checkpoint_sha256
N
mae_wh
rmse_wh
r2
population_sha256
metric_version
status
```

---

# 170. Transformer aggregate schema

`transformer_seed_aggregate_metrics.csv`:

```text
metric
seed42
seed123
seed2026
mean
sample_sd
min
max
range
seed_count
aggregation_basis=PER_SEED_METRICS
status
```

---

# 171. Baseline metrics schema

`final_test_baseline_metrics.csv`:

```text
model_id
eligibility
training_protocol
N
mae_wh
rmse_wh
r2
population_sha256
scaler_reference_if_any
notes
status
```

---

# 172. Model comparison schema

`final_test_model_comparison.csv`:

```text
row_type
model_id
seed
N
mae_wh
mae_sd_if_aggregate
rmse_wh
rmse_sd_if_aggregate
r2
r2_sd_if_aggregate
training_protocol
population_sha256
directly_common_target_comparable
status
```

---

# 173. Baseline delta schema

`final_test_baseline_deltas.csv`:

```text
transformer_scope
seed_or_aggregate
baseline
transformer_rmse
baseline_rmse
delta_rmse_baseline_to_transformer
transformer_mae
baseline_mae
delta_mae_baseline_to_transformer
transformer_r2
baseline_r2
delta_r2_transformer_minus_baseline
status
```

---

# 174. Metric consistency audit

`final_test_metric_consistency_audit.csv`:

```text
check
model_a
model_b
rmse_order
r2_order
expected_consistency
observed_consistency
status
```

For same y_true:

```text
RMSE lower ↔ R² higher.
```

---

# 175. Final summary table

`final_test_summary_table.csv` should be report-ready.

Recommended rows:

```text
Transformer seed42
Transformer seed123
Transformer seed2026
Transformer mean ± SD
Persistence
LSTM_TUNED_DEV if eligible.
```

Columns:

```text
MAE
RMSE
R²
N
Notes.
```

---

# 176. Figures

Recommended only global summary figures:

```text
TEST_47_01_rmse_by_seed_and_baselines.png
TEST_47_02_mae_by_seed_and_baselines.png
TEST_47_03_r2_by_seed_and_baselines.png
TEST_47_04_transformer_seed_metric_variability.png
```

Detailed residual/prediction figures belong Phase48–51.

---

# 177. Figure rules

No truncating axes in misleading ways.

Use full labels:

```text
Held-out Test
```

and distinguish:

```text
Transformer seed-specific
Transformer mean
Persistence
LSTM_TUNED_DEV.
```

---

# 178. Do not plot aggregate Transformer mean as a model prediction

Metric bar is fine.

Do not imply an ensemble prediction exists.

---

# 179. Findings codes

`final_test_findings.csv` possible:

```text
TEST_RELEASE_VERIFIED
TEST_FIRST_ACCESS_RECORDED
FINAL_TEST_POPULATION_VERIFIED
SEED42_TEST_COMPLETED
SEED123_TEST_COMPLETED
SEED2026_TEST_COMPLETED
ALL_THREE_SEEDS_TEST_COMPLETED
TRANSFORMER_SEED_VARIABILITY
TRANSFORMER_MEAN_BEATS_PERSISTENCE
PERSISTENCE_BEATS_TRANSFORMER_MEAN
ALL_TRANSFORMER_SEEDS_BEAT_PERSISTENCE
SOME_TRANSFORMER_SEEDS_BEAT_PERSISTENCE
LSTM_TEST_ELIGIBLE
LSTM_TEST_INELIGIBLE
TRANSFORMER_MEAN_BEATS_LSTM_DEV
LSTM_DEV_BEATS_TRANSFORMER_MEAN
RMSE_R2_CONSISTENCY_VERIFIED
COMMON_TARGETS_VERIFIED
NO_BEST_SEED_SELECTION
NO_ENSEMBLE
NO_POST_TEST_TUNING
PREDICTION_BUNDLES_FROZEN
INHERITED_WARNING
```

---

# 180. Discrepancy taxonomy

`final_test_discrepancies.json`:

```text
PHASE46_NOT_RELEASED
TEST_RELEASE_MISMATCH
MISSING_FINAL_CHECKPOINT
CHECKPOINT_SHA_MISMATCH
LOCK_HASH_MISMATCH
CONFIG_HASH_MISMATCH
RECIPE_HASH_MISMATCH
SCALER_CHECKSUM_MISMATCH
FEATURE_ORDER_DRIFT
TEST_POPULATION_MISMATCH
TEST_TARGET_ID_DUPLICATE
TEST_TARGET_ORDER_MISMATCH
WB_PROTOCOL_DRIFT
LOOKBACK_DRIFT
HORIZON_DRIFT
SCALER_FIT_CALLED_ON_TEST
REVIN_SCOPE_DRIFT
FUTURE_TARGET_IN_INPUT
RECURSIVE_PREDICTION_FEEDBACK_USED
MODEL_TRAINED_IN_PHASE47
OPTIMIZER_STEP_IN_PHASE47
BACKWARD_CALLED_IN_PHASE47
DROPOUT_ACTIVE_DURING_OFFICIAL_INFERENCE
CHECKPOINT_SELECTION_AFTER_TEST
BEST_SEED_SELECTED
SEED_DROPPED
UNAUTHORIZED_TRANSFORMER_CANDIDATE_TESTED
ENSEMBLE_CREATED_POST_TEST
PREDICTION_CLIPPED_POST_HOC
TEST_ROWS_DROPPED
NONFINITE_PREDICTION
METRIC_POPULATION_MISMATCH
RMSE_R2_RANKING_INCONSISTENCY
LSTM_RETRAINED_POST_RELEASE
LSTM_WRONG_SCALER
LSTM_COMMON_TARGET_MISMATCH
PERSISTENCE_TARGET_MISMATCH
TECHNICAL_RERUN_NOT_LINKED
FAVORABLE_INFERENCE_RUN_SELECTED
POST_TEST_TUNING_ATTEMPT
OTHER
```

---

# 181. Status model

## PASS

```text
Phase46 release valid
evaluation contract frozen before first Test access
3/3 Transformer checkpoints evaluated
common Test population verified
Persistence evaluated
eligible LSTM handled correctly
all metrics verified
mean±SD seed summary generated
no selection/retraining
prediction bundles frozen
downstream handoffs ready.
```

## PASS_WITH_WARNING

Possible:

```text
LSTM Test baseline ineligible
LSTM comparison has asymmetric training protocol
high seed variability
Persistence outperforms Transformer
LSTM outperforms Transformer
environment minor warning inherited.
```

These are scientific findings, not methodological failure.

## FAIL

Examples:

```text
model/scaler changed
Test population changed
one seed omitted
unauthorized candidate tested
best seed selected
Test-driven tuning
LSTM retrained after Test release
non-finite predictions
metric population mismatch.
```

---

# 182. Execution order

Recommended:

```text
1. Verify Phase46 release.
2. Freeze/evaluate contract hash.
3. Pre-resolve LSTM eligibility using pre-Test artifacts.
4. Verify Transformer checkpoints/scalers.
5. Materialize FINAL_TEST_POP-v1.
6. Record first Test access event.
7. Build Test loader.
8. Infer seed42.
9. Infer seed123.
10. Infer seed2026.
11. Infer Persistence.
12. Infer eligible LSTM baseline.
13. Freeze/checksum prediction bundles.
14. Verify common target/y_true alignment.
15. Compute per-seed metrics.
16. Compute mean±sample SD.
17. Compute baseline metrics/deltas.
18. Run metric consistency checks.
19. Generate summary figures.
20. Write downstream handoffs.
21. Sign off.
```

---

# 183. LSTM eligibility must be decided before inspecting its Test score

Correct:

```text
pre-Test provenance gate
→ eligible/not eligible
→ only then Test inference.
```

Incorrect:

```text
run LSTM Test
→ if result useful, include it.
```

---

# 184. No adaptive model list

Authorized models are locked before first Test target access.

---

# 185. No adaptive metrics

Metrics are locked before Test access.

---

# 186. No adaptive aggregation rule

Mean±sample SD is locked before Test access.

Do not switch to median because mean looks worse.

---

# 187. No adaptive baseline choice

Persistence is always included.

LSTM inclusion depends only on pre-Test eligibility, not score.

---

# 188. Seed42 workflow

```text
verify checkpoint
load final scalers
build exact Test dataset/loader
eval + inference_mode
predict all FINAL_TEST_POP-v1
inverse to Wh
write full-precision bundle
checksum
integrity audit
```

No metric-driven action afterward.

---

# 189. Seed123 workflow

Identical except checkpoint.

---

# 190. Seed2026 workflow

Identical except checkpoint.

---

# 191. Persistence workflow

```text
for every Test target
resolve immediate prior observed Appliances
predict
write bundle
checksum
```

No train/scaler.

---

# 192. LSTM workflow

If eligible:

```text
strict-load Phase43 frozen checkpoint
load its exact development scaler artifacts
build exact compatible Test windows on FINAL_TEST_POP-v1
eval/inference
inverse to Wh
write bundle
checksum.
```

No retraining.

---

# 193. Common-target verification happens before comparison metrics

Per-model metrics may be computed after own integrity passes.

Cross-model comparison only after:

```text
same target IDs
same order
same y_true.
```

---

# 194. Per-seed metrics may differ due only to trained weights

All other Transformer evaluation elements:

```text
data
scaler
feature order
model schema
checkpoint epoch
metric code
```

are identical.

---

# 195. No metric average across rows from different populations

Hard.

---

# 196. No using original Validation to explain Test errors in a way that retunes

Comparative discussion is allowed, but no model change.

---

# 197. Downstream Phase48 handoff

`phase48_prediction_analysis_handoff.json`:

```text
final_test_version
final_lock_hash
Test population fingerprint
three Transformer prediction bundle paths/checksums
Persistence bundle
LSTM bundle if eligible
per-seed metrics
aggregate metrics
seed aggregation semantics
no_best_seed=true
no_ensemble=true
ready_for_phase48=true
```

---

# 198. Phase49 handoff

`phase49_residual_analysis_handoff.json`:

```text
prediction bundles
residual convention
target IDs
timestamps
seed identities
population fingerprint
ready_for_phase49=true.
```

---

# 199. Phase50 handoff

`phase50_error_regime_handoff.json`:

```text
prediction bundles
Train-derived regime threshold artifacts required
Test target values/predictions
population fingerprint
seed identities
ready_for_phase50=true.
```

Regime thresholds must come from Train, not be chosen from Test.

---

# 200. Phase51 handoff

`phase51_worst_error_handoff.json`:

```text
prediction bundles
seed identities
full Test IDs
no rows dropped
ready_for_phase51=true.
```

---

# 201. Phase52 handoff

`phase52_attention_extraction_handoff.json`:

```text
three final checkpoint paths/checksums
final scalers
feature order
lookback
heads/layers
Test population fingerprint
prediction bundle refs
attention compatibility status
seed policy
ready_for_phase52=true.
```

---

# 202. Downstream analyses may not change final metrics

Phase48–57 are analyses of the already fixed final model/checkpoints.

They do not reopen training.

---

# 203. Human-readable report structure

`final_test_report.md`:

```text
1. Objective
2. Test-release and no-leakage gate
3. Frozen evaluation contract
4. Final Test population
5. Final Transformer checkpoints
6. Scaler/preprocessing reuse
7. WB0 one-step Test semantics
8. Transformer Test metrics by seed
9. Seed mean ± sample SD
10. Persistence Test baseline
11. LSTM Test eligibility and result if applicable
12. Common-target fairness audit
13. Baseline deltas
14. Test vs pre-Test robustness context
15. Seed sensitivity
16. Methodological limitations
17. No best-seed/no ensemble statement
18. Frozen prediction artifacts
19. Handoff to prediction/error/attention analysis
20. Definition of Done
```

---

# 204. README requirements

`README_FINAL_TEST_EVALUATION.md` explains:

```text
why Test first opens here
why all three seeds are evaluated
why no best seed is chosen
why mean±SD is reported
why no seed ensemble
why scalers are not refit
WB0 actual-history semantics
Persistence semantics
LSTM eligibility/fairness caveat
why detailed residual/attention analysis is deferred
why Test cannot be reused for tuning.
```

---

# 205. Summary artifact

`final_test_summary.json`:

```text
version
final_lock_sha256
evaluation_contract_sha256
test_first_access_event
test_population_sha256
N_test
transformer_seed_metrics
transformer_metric_mean
transformer_metric_sample_sd
persistence_metrics
lstm_eligibility
lstm_metrics_if_any
baseline_deltas
common_target_status
metric_consistency_status
prediction_bundle_checksums
best_seed_selected=false
ensemble_used=false
training_in_phase47=false
scaler_fit_in_phase47=false
post_test_tuning=false
phase48_ready
phase52_ready
overall_status
```

---

# 206. Phase47 sign-off

`phase_47_signoff.json` minimum:

```text
phase=47
phase_name=Final test evaluation
version=FINAL_TEST_EVAL-v1
source_phase46_version
final_lock_sha256
evaluation_contract_sha256
test_first_access_authorized=true
test_population_sha256
N_test
seed_list=[42,123,2026]
seed42_checkpoint_sha256
seed123_checkpoint_sha256
seed2026_checkpoint_sha256
seed42_mae_wh
seed42_rmse_wh
seed42_r2
seed123_mae_wh
seed123_rmse_wh
seed123_r2
seed2026_mae_wh
seed2026_rmse_wh
seed2026_r2
transformer_mean_mae_wh
transformer_sd_mae_wh
transformer_mean_rmse_wh
transformer_sd_rmse_wh
transformer_mean_r2
transformer_sd_r2
persistence_mae_wh
persistence_rmse_wh
persistence_r2
lstm_eligibility
lstm_metrics_if_eligible
all_common_targets_verified
best_seed_selected=false
ensemble_used=false
training_used=false
scaler_fit_used=false
post_test_tuning=false
prediction_bundles_frozen=true
ready_for_phase48
ready_for_phase52
warnings
overall_status
created_at
```

---

# 207. Acceptance checklist — release/contract

```text
[ ] Phase46 PASS/PASS_WITH_WARNING.
[ ] phase47_test_release.released=true.
[ ] 3/3 final checkpoints verified.
[ ] Same final lock across checkpoints.
[ ] Same config/recipe/population/scalers/epochs.
[ ] Evaluation contract written before Test y access.
[ ] Metrics frozen.
[ ] Seed aggregation frozen.
[ ] Authorized model list frozen.
[ ] Persistence included.
[ ] LSTM eligibility gate defined pre-access.
[ ] No ensemble.
[ ] No best-seed rule.
```

---

# 208. Acceptance checklist — Test population

```text
[ ] FINAL_TEST_POP-v1 materialized.
[ ] Exact locked WINDOWPOP-v1 Test policy.
[ ] All IDs unique.
[ ] IDs chronological.
[ ] All target rows belong to Test.
[ ] L* exact.
[ ] H1 exact.
[ ] WB0 exact.
[ ] Continuity exact.
[ ] No padding.
[ ] No row removed post prediction.
[ ] Population SHA256 generated.
```

---

# 209. Acceptance checklist — preprocessing

```text
[ ] Final feature order exact.
[ ] Final feature fingerprint exact.
[ ] X scaler checksum exact.
[ ] Y scaler/identity exact.
[ ] Scaler fit not called.
[ ] Time features deterministic.
[ ] RN1 scope exact if active.
[ ] RN1 uses input-window stats only.
[ ] No future target statistics.
[ ] No full-Test normalization.
```

---

# 210. Acceptance checklist — Transformer inference

```text
[ ] Seed42 strict-load.
[ ] Seed123 strict-load.
[ ] Seed2026 strict-load.
[ ] checkpoint_type FINAL_REFIT.
[ ] Official epoch correct.
[ ] model.eval().
[ ] inference_mode().
[ ] Dropout inactive.
[ ] Standard forward path.
[ ] No attention extraction for official predictions.
[ ] No optimizer.
[ ] No backward.
[ ] No model mutation.
[ ] Same Test IDs all seeds.
[ ] Prediction shape valid.
[ ] All predictions finite.
```

---

# 211. Acceptance checklist — WB0 autoregressive history

```text
[ ] Inputs strictly precede target.
[ ] First Test target may use Val history.
[ ] Later Test targets may use earlier observed Test history.
[ ] Actual observations used.
[ ] No previous prediction feedback.
[ ] No target t+1 inside input.
```

---

# 212. Acceptance checklist — metrics

```text
[ ] y_true raw Wh exact.
[ ] y_pred inverse-transformed to Wh.
[ ] MAE computed globally.
[ ] RMSE computed globally.
[ ] R² computed globally.
[ ] No batch metric averaging.
[ ] Full precision used.
[ ] Same N all seeds.
[ ] RMSE/R² ordering consistency passes.
[ ] Per-seed metrics saved.
[ ] Mean across 3 seed metrics saved.
[ ] Sample SD ddof=1 saved.
[ ] No pooled 3N metric used as primary.
[ ] No best seed selected.
[ ] No ensemble metric.
```

---

# 213. Acceptance checklist — baselines

```text
[ ] Persistence exact same Test IDs.
[ ] Persistence uses raw previous observed Appliances.
[ ] Persistence metrics via METRICS-v1.
[ ] LSTM eligibility resolved before score inspection.
[ ] LSTM not retrained.
[ ] LSTM exact frozen checkpoint if eligible.
[ ] LSTM exact training scaler if eligible.
[ ] LSTM common target IDs verified if compared.
[ ] LSTM asymmetric final-training caveat reported.
```

---

# 214. Acceptance checklist — prediction artifacts

```text
[ ] Seed42 bundle complete.
[ ] Seed123 bundle complete.
[ ] Seed2026 bundle complete.
[ ] Persistence bundle complete.
[ ] Eligible LSTM bundle complete.
[ ] target IDs/timestamps saved.
[ ] y_true/y_pred full precision.
[ ] residual convention correct.
[ ] bundle checksums generated.
[ ] bundles frozen.
[ ] downstream handoffs reference exact checksums.
```

---

# 215. Acceptance checklist — no Test-driven adaptation

```text
[ ] No candidate added.
[ ] No candidate removed.
[ ] No seed dropped.
[ ] No seed selected.
[ ] No hyperparameter changed.
[ ] No epoch changed.
[ ] No scaler refit.
[ ] No retraining.
[ ] No post-hoc prediction clipping.
[ ] No Test sample removal.
[ ] No adaptive metric.
[ ] No adaptive aggregation.
[ ] No ensemble created.
```

---

# 216. Acceptance checklist — downstream

```text
[ ] Phase48 handoff generated.
[ ] Phase49 handoff generated.
[ ] Phase50 handoff generated.
[ ] Phase51 handoff generated.
[ ] Phase52 handoff generated.
[ ] No downstream phase requires rerunning official Test predictions.
[ ] Final Test sign-off generated.
```

---

# 217. Acceptance criteria

Phase47 PASS only when:

```text
Phase46 explicitly releases Test access.

The final Test evaluation contract is frozen before the first Test target value is materialized.

The exact locked FINAL_TEST_POP-v1 population is used.

The same Test target IDs and raw y_true values are used for all three Transformer seeds and Persistence.

All three FINAL_REFIT checkpoints are strict-loaded and evaluated without model mutation.

FINAL_SCALING-v1 is reused exactly and never refit on Test.

WB0 one-step observed-history semantics are preserved without recursive prediction feedback.

Every Transformer prediction is finite and persisted in an immutable full-precision bundle.

MAE, RMSE and R² are computed in raw Wh space using METRICS-v1.

All three seed-specific metrics are reported.

Transformer final performance is summarized as the arithmetic mean and sample SD of the three per-seed metrics.

No best seed is selected.

No seed-prediction ensemble is created.

Persistence is evaluated on the exact same Test targets.

The frozen Phase43 LSTM baseline is evaluated only if its pre-Test eligibility gate passes and is never retrained.

All direct model comparisons use common target IDs.

All prediction bundles are checksummed and handed to downstream analyses.

No post-Test tuning, refitting, candidate testing or sample removal occurs.
```

---

# 218. Failure conditions

Phase47 FAIL if:

```text
Test is accessed before Phase46 release

evaluation contract is written after seeing Test outputs

one final seed is omitted

an unauthorized Transformer candidate is evaluated

a checkpoint/config/scaler does not match lock

Test scaler is fitted/refitted

WB0 semantics change

future Test target enters input

previous prediction is recursively fed as observed history

model.train() remains active during official inference

dropout remains stochastic

backward/optimizer step occurs

Test rows are dropped after observing errors

predictions are clipped post hoc

best seed is selected

ensemble is created post Test

LSTM is retrained after Test access

LSTM uses wrong scaler

models are compared on different target IDs

RMSE/R² consistency fails without investigation

technical rerun is chosen by favorable score

hyperparameters/epochs/features change after Test

post-Test tuning is attempted.
```

---

# 219. Common mistakes

## 219.1 Chỉ Test seed42

Sai vì ba seeds đã được predeclared.

## 219.2 Test cả ba rồi chọn seed RMSE thấp nhất làm final

Sai Test-driven selection.

## 219.3 Average predictions của ba seed và gọi là final ensemble

Ngoài pre-Test contract.

## 219.4 Fit StandardScaler lại trên Test

Leakage.

## 219.5 Test từng old Transformer candidate

Biến Test thành model-selection set.

## 219.6 Test LSTM xong thấy tốt rồi retrain LSTM Train+Val

Sai post-Test adaptation.

## 219.7 LSTM dùng Transformer final scaler

Sai nếu checkpoint được train bằng scaler khác.

## 219.8 Persistence dùng prediction trước thay cho actual y

Sai persistence one-step semantics.

## 219.9 Loại các spike lớn rồi tính lại RMSE

Sai.

## 219.10 Prediction âm thì set 0 sau khi xem Test

Sai post-hoc clipping.

## 219.11 Mean ba RMSE nhưng gọi đó là ensemble RMSE

Sai terminology.

## 219.12 Gộp 3N seed-target rows và gọi là sample size 3N

Sai dependence interpretation.

---

# 220. Recommended execution pseudocode

```text
p46 = load_phase46_signoff()
release = load_phase47_test_release()

assert release.released is True
assert p46.ready_for_phase47

freeze_evaluation_contract_before_test_access()

authorized_seeds = [42,123,2026]
authorized_checkpoints = resolve_final_seed_checkpoints(
    authorized_seeds
)

verify_all_checkpoints_against_final_lock()
verify_final_scaler_checksums()

lstm_eligibility = resolve_lstm_eligibility_pretest_provenance()

# First Test access begins only here.
record_test_first_access_event()

test_pop = materialize_FINAL_TEST_POP_v1(
    source="WINDOWPOP-v1",
    split="TEST",
    lookback=FINAL_L,
    horizon=1,
    boundary_protocol="WB0"
)

assert valid_and_unique(test_pop)
test_population_hash = freeze_population(test_pop)

transformer_predictions = {}

for seed in [42,123,2026]:

    ckpt = authorized_checkpoints[seed]

    model = build_exact_final_model()
    strict_load(model, ckpt)

    model.eval()

    test_loader = build_final_test_loader(
        population=test_pop,
        final_scalers=FINAL_SCALING_v1,
        shuffle=False,
        drop_last=False
    )

    preds = infer_in_wh(
        model=model,
        loader=test_loader,
        inference_mode=True,
        use_standard_forward=True
    )

    assert ids(preds) == test_pop.ids
    assert finite(preds)

    bundle = persist_prediction_bundle(
        seed=seed,
        predictions=preds
    )

    freeze_and_checksum(bundle)
    transformer_predictions[seed] = bundle

persistence_bundle = evaluate_persistence_on_exact_ids(
    test_pop,
    use_actual_prior_observed_appliances=True
)
freeze_and_checksum(persistence_bundle)

if lstm_eligibility.eligible:
    lstm_bundle = evaluate_frozen_phase43_lstm(
        checkpoint=lstm_eligibility.checkpoint,
        scalers=lstm_eligibility.scalers,
        target_ids=test_pop.ids
    )
    assert common_targets(lstm_bundle, test_pop)
    freeze_and_checksum(lstm_bundle)
else:
    lstm_bundle = None

verify_common_ytrue(
    transformer_predictions,
    persistence_bundle,
    lstm_bundle
)

seed_metrics = {
    seed: METRICS_v1(bundle)
    for seed, bundle in transformer_predictions.items()
}

aggregate = aggregate_seed_metrics(
    seed_metrics,
    mean=True,
    sample_sd_ddof=1,
    min_max_range=True
)

persistence_metrics = METRICS_v1(persistence_bundle)

if lstm_bundle is not None:
    lstm_metrics = METRICS_v1(lstm_bundle)

assert rmse_r2_order_consistent(seed_metrics)

write_baseline_deltas()
write_summary_tables()
write_global_summary_figures()
write_prediction_checksums()
write_phase48_to_phase52_handoffs()

assert best_seed_selected is False
assert ensemble_used is False
assert training_used is False
assert scaler_fit_in_phase47 is False
assert post_test_tuning is False

signoff_phase47()
```

---

# 221. Definition of Done

\[
\boxed{
Authorized\ Test\ Release
+
Frozen\ Evaluation\ Contract
+
Exact\ FINAL\_TEST\_POP
+
Three\ FINAL\_REFIT\ Checkpoints
+
Frozen\ FINAL\_SCALING
+
Inference\ Only
+
Per\text{-}Seed\ MAE/RMSE/R^2
+
Mean\pm SampleSD
+
Persistence
+
Eligible\ Frozen\ LSTM
+
No\ Best\ Seed
+
No\ Ensemble
+
Frozen\ Prediction\ Bundles
+
No\ Post\text{-}Test\ Tuning
}
\]

---

# 222. Final status contract

```text
PHASE 47 is the first Test-access phase.

Prerequisite:
Phase46 release=true
3/3 final Transformer checkpoints verified.

Before Test access:
freeze evaluation contract
freeze authorized models
freeze metrics
freeze seed aggregation
resolve LSTM eligibility.

Test population:
FINAL_TEST_POP-v1
locked WINDOWPOP-v1 Test target IDs
WB0
H1
same targets for all models.

Transformer:
seed42
seed123
seed2026
FINAL_REFIT checkpoints
FINAL_SCALING-v1
eval/inference only.

Metrics:
MAE Wh
RMSE Wh
R².

Final Transformer reporting:
all three seed metrics
+
mean ± sample SD across seed metrics.

Forbidden:
best seed selection
seed ensemble
scaler refit
training
candidate testing
prediction clipping
row dropping
post-Test tuning.

Baselines:
Persistence always.
Frozen Phase43 LSTM only if pre-Test eligibility passes.
No LSTM retraining.

Outputs:
immutable full-precision prediction bundles
metrics
baseline deltas
downstream analysis handoffs.

After FINAL_TEST_EVAL-v1 PASS:
proceed to
PHASE 48 — Prediction Analysis.
```

---

# 223. Final check

Correct:

```text
Phase46 Test release
→ freeze Phase47 evaluation contract
→ verify 3 checkpoints + scalers
→ determine LSTM eligibility
→ first Test access
→ exact Test population
→ infer all 3 Transformer seeds
→ infer Persistence
→ infer eligible frozen LSTM
→ freeze prediction bundles
→ per-seed metrics
→ mean ± sample SD
→ no seed selection
→ Phase48–52 handoffs
```

Incorrect:

```text
Test seed42
→ choose it if good
→ adjust epochs
→ Test again
```

Incorrect:

```text
evaluate TR_C0/TR_C1/TR_C2 on Test
→ choose best candidate
```

Incorrect:

```text
fit scaler on Test
```

Incorrect:

```text
average seed predictions
→ declare new ensemble after seeing Test
```

Chỉ sau khi:

```text
FINAL_TEST_EVAL-v1 = PASS / PASS_WITH_WARNING
```

và prediction bundles đã được frozen mới chuyển sang **PHASE 48 — Prediction Analysis**.
