# PHASE 24 — S2 TIME-FEATURE SWEEP

## Kế hoạch controlled sweep cho engineered time features của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S1_FEATURESET-v1`  
**Sweep ID:** `S2_TIME_FEATURE`  
**Output version:** `SWEEP_S2_TIMEFEATURE-v1`  
**Phase trước:** `Phase_23_S1_Feature-set_sweep.md`

---

# 1. Vai trò của Phase 24

Phase 24 là controlled experiment thứ hai trong chuỗi Transformer sweeps. Mục tiêu duy nhất là kiểm tra liệu **engineered time features** có cải thiện Validation forecasting performance hay không, trong khi feature set đã chọn ở S1 và toàn bộ model/training/data protocol khác được giữ nguyên.

Nguyên tắc:

\[
\boxed{
One\ Factor
+
Same\ Feature\ Set
+
Same\ Samples
+
Same\ Model/Training
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

Pipeline:

```text
Phase 23
→ S1 Feature-set sweep
→ chọn current feature set

Phase 24
→ S2 TF0 vs TF1

Phase 25
→ S3 Target-scaling sweep
```

---

# 2. Câu hỏi nghiên cứu

Phase 24 phải trả lời:

```text
1. Với S1-selected FS giữ nguyên, TF1 có giảm Validation RMSE so với TF0 không?

2. Nếu có, mức giảm bao nhiêu Wh và bao nhiêu %?

3. MAE và R² có cùng chiều với RMSE không?

4. TF1 có làm thay đổi learning dynamics, gradient clipping hoặc convergence behavior không?

5. Nếu TF0 thắng, explicit calendar encoding có đang redundant hoặc không hữu ích dưới configuration hiện tại không?

6. S2 winner nào sẽ trở thành current feature variant cho Phase 25?
```

---

# 3. Source feature set từ S1

Phase 24 không hard-code `FS1`.

Phải load:

```text
s1_feature_set_winner.json
s1_reference_update.json
phase_23_signoff.json
```

để resolve:

```text
selected_feature_set_id = FSx
```

Possible:

```text
FS0
FS1
FS2
```

Comparison:

```text
FS0 winner → FS0_TF0 vs FS0_TF1
FS1 winner → FS1_TF0 vs FS1_TF1
FS2 winner → FS2_TF0 vs FS2_TF1
```

---

# 4. Time-feature definitions

Source of truth:

```text
FEATURES-v1
FEATURESETS-v1
```

## TF0

Không có:

```text
hour_sin
hour_cos
dow_sin
dow_cos
weekend
```

## TF1

Có đúng:

```text
hour_sin
hour_cos
dow_sin
dow_cos
weekend
```

Không được thêm trong Phase 24:

```text
month
season
holiday
day_of_month
absolute_time_index
target_hour
target_dow
target_weekend
raw timestamp
```

---

# 5. Formula lineage

Phase 24 không redefine feature engineering, nhưng audit phải bám contract:

\[
minute\_of\_day = 60 \times hour + minute
\]

\[
hour\_sin=\sin\left(2\pi\frac{minute\_of\_day}{1440}\right)
\]

\[
hour\_cos=\cos\left(2\pi\frac{minute\_of\_day}{1440}\right)
\]

\[
dow\_sin=\sin\left(2\pi\frac{dow}{7}\right)
\]

\[
dow\_cos=\cos\left(2\pi\frac{dow}{7}\right)
\]

```text
weekend = 1 for Saturday/Sunday
else 0
```

---

# 6. Temporal-leakage rule

TF1 chỉ xuất hiện trên historical input rows:

```text
[j-L, ..., j-1]
```

Không thêm engineered calendar features của target row `j`.

Dù target-time calendar thường có thể biết trước trong thực tế, current coursework contract đã khóa historical-sequence-only input. Phase 24 không thay formulation.

---

# 7. Working hypotheses

## H-S2-01 — Time-feature contribution

Nếu:

\[
RMSE(FSx\_TF1)<RMSE(FSx\_TF0)
\]

thì evidence support rằng explicit cyclical/calendar context cải thiện Validation forecasting trong current configuration.

Status trước run:

```text
UNTESTED
```

## H-S2-02 — Redundancy/no gain

Nếu:

```text
RMSE(TF1) >= RMSE(TF0)
```

thì TF1 không cải thiện current Validation result.

Possible explanations chỉ được giữ ở mức hypothesis:

```text
calendar effects weak
24h context already encodes temporal rhythm
sensor histories already proxy time-of-day
single-seed stochasticity
extra channels add redundancy
```

Không được khẳng định một nguyên nhân duy nhất.

---

# 8. Preconditions

Phase 24 chỉ bắt đầu khi:

```text
Phase 23 = PASS
```

hoặc `PASS_WITH_WARNING` nhưng không có unresolved CRITICAL issue, và:

```text
approved_for_phase24 = true
```

Bắt buộc verify:

```text
ENV-v1
FEATURES-v1
FEATURESETS-v1
SPLIT-v1
SCALING-v1
WINDOWS-v1
WINDOWPOP-v1
DATALOADERS-v1
METRICS-v1
EXPERIMENTS-v1
TRANSFORMER_IMPL-v1
ATTENTION_VERIFY-v1
TRAINING_ENGINE-v1
SWEEP_S1_FEATURESET-v1
```

---

# 9. Random-control warning carry-forward

Nếu S1 winner = `FS2_TF1` và Phase 23 đã flag:

```text
RANDOM_CONTROL_GAIN
```

Phase 24 phải carry warning vào:

```text
manifest
summary
report
winner artifact
Phase 25 handoff
```

Warning không tự động block S2 nếu methodology hợp lệ.

---

# 10. Swept factor duy nhất

```text
time_feature_id
```

Allowed:

```text
TF0
TF1
```

Feature set:

```text
FSx = S1 selected feature set
```

fixed.

---

# 11. Frozen factors

Mọi yếu tố sau phải giữ nguyên current reference:

```text
FSx
L144
H1
YS1
WB0
WINDOWPOP-v1
B64

d_model = 64
num_heads = 4
num_layers = 2
ffn_dim = 128
dropout = 0.1
activation = GELU
pooling = LAST_STEP
sinusoidal positional encoding
POST_NORM
no causal mask
no padding mask

AdamW
LR = 3e-4
WD = 1e-4
MSE
max_epochs = 50
patience = 10
min_delta = 0
gradient clipping max_norm = 1.0
scheduler = None
mixed precision = False
gradient accumulation = 1
seed = 42

TRAINING_ENGINE-v1
METRICS-v1
```

---

# 12. Exact feature-delta rule

For selected `FSx`:

```text
features(FSx_TF1)
-
features(FSx_TF0)
```

must equal exactly:

```text
{
  hour_sin,
  hour_cos,
  dow_sin,
  dow_cos,
  weekend
}
```

and:

```text
features(FSx_TF0)
-
features(FSx_TF1)
=
empty
```

Nếu không:

```text
S2 invalid.
```

---

# 13. Feature ordering

Canonical ordered feature lists phải lấy từ `FEATURESETS-v1`.

Không dùng:

```text
manual drop
select_dtypes
set-based ordering
hard-coded notebook list
```

làm source of truth.

---

# 14. Feature-count implication

Expected relationship:

\[
F_{TF1}=F_{TF0}+5
\]

nhưng runtime registry là authoritative.

Do not hard-code actual F.

---

# 15. Parameter-count nuance

TF1 có 5 channels thêm nên input projection:

```text
Linear(F,64)
```

có nhiều weights hơn.

Theoretical extra input-projection weights:

\[
5 \times 64 = 320
\]

under current D64, nhưng runtime parameter audit vẫn là source of truth.

Không thay d_model để parameter-match vì sẽ tạo factor thứ hai.

---

# 16. Population contract

Hard:

```text
WINDOWPOP-v1 identical
```

cho TF0 và TF1.

Train/Validation target IDs:

```text
TF0 == TF1
```

Không được drop sample riêng cho TF1.

---

# 17. Why TF1 should not create missing samples

TF1 deterministic from valid timestamps.

NaN/Inf ở TF1 là dấu hiệu:

```text
feature-engineering/timestamp bug
```

không phải lý do để drop sample.

---

# 18. Scaler contract

Use variant-specific X bundles:

```text
XSCALER__FSx_TF0
XSCALER__FSx_TF1
```

Same frozen target scaler:

```text
YSCALER__YS1
```

---

# 19. Time-feature pass-through scaling

Under `SCALING-v1`:

```text
hour_sin
hour_cos
dow_sin
dow_cos
weekend
```

không standardized.

They pass through unchanged.

TF0 không có các channels này.

---

# 20. Cyclical integrity checks

TF1 must verify:

```text
hour_sin ∈ [-1,1]
hour_cos ∈ [-1,1]
dow_sin ∈ [-1,1]
dow_cos ∈ [-1,1]
weekend ∈ {0,1}
finite values
```

and approximately:

\[
hour\_sin^2+hour\_cos^2 \approx 1
\]

\[
dow\_sin^2+dow\_cos^2 \approx 1
\]

within numerical tolerance.

---

# 21. Existing TF1 reference reuse

S1 winner is a `*_TF1` run.

If exact S2 contract match:

```text
REUSE S1 winner run
```

as TF1 reference.

Do not retrain TF1.

Possible source:

```text
Phase 21 B0 if FS1_TF1 won S1
Phase 23 FS0 run if FS0_TF1 won
Phase 23 FS2 run if FS2_TF1 won
```

---

# 22. Normal S2 run count

Typically:

```text
TF1 → reused reference
TF0 → one NEW run
```

Thus:

```text
1 new training run
+
1 reused run
```

---

# 23. Reference reuse gate

TF1 reference must exactly match:

```text
selected FS
TF1
L144
H1
YS1
WB0
WINDOWPOP-v1
B64
D64/H4/N2/FFN128
dropout .1
GELU
LAST_STEP
AdamW
LR 3e-4
WD 1e-4
MSE
E50
patience 10
clip 1
seed 42
Training Engine
Metric version
```

Any mismatch:

```text
STOP
```

rather than silently retrain.

---

# 24. Fresh TF0 run

Execution:

```text
seed 42
↓
fresh TF0 Train loader
↓
fresh TF0 Validation loader
↓
fresh Transformer
↓
correct input_size from registry
↓
MSE
↓
AdamW
↓
TRAINING_ENGINE-v1
```

No warm-start from TF1.

---

# 25. Why no warm-start

Warm-start confounds S2 because:

```text
input projection dimensions differ
weights already learned with TF1
training order matters
```

Fresh run is mandatory.

---

# 26. Experiment identity

```text
sweep_id = S2_TIME_FEATURE
experiment_family = TRANSFORMER_SWEEP_S2_TIME_FEATURE
sweep_version = SWEEP_S2_TIMEFEATURE-v1
```

Recommended labels:

```text
S2_<FSx>_TF0__L144__YS1__B64__S42
S2_<FSx>_TF1__REFERENCE_S1
```

Registry-generated `run_id` remains authoritative.

---

# 27. Test firewall

Hard:

```text
test_access_authorized = False
Test DataLoader not iterated
Test metrics absent
Test predictions absent
```

---

# 28. Primary selection metric

Hard:

```text
verified best Validation RMSE Wh
```

from `METRICS-v1`.

Secondary:

```text
MAE Wh
R²
best epoch
epochs completed
stop reason
gradient diagnostics
runtime
parameter count
```

---

# 29. S2 winner rule

\[
winner
=
argmin(RMSE_{TF0},RMSE_{TF1})
\]

Use full precision.

Exact RMSE tie:

```text
prefer TF0
```

because it is simpler and uses fewer engineered channels.

---

# 30. No arbitrary improvement threshold

Không invent:

```text
TF1 must gain >1%
TF1 must gain >1 Wh
```

Strict lower RMSE wins unless exact tie.

---

# 31. Effect formulas

Let:

```text
RMSE0 = TF0
RMSE1 = TF1
```

Then:

\[
\Delta RMSE = RMSE_0 - RMSE_1
\]

Positive:

```text
TF1 improves.
```

Relative:

\[
Improvement\%=
100\times\frac{RMSE_0-RMSE_1}{RMSE_0}
\]

Similarly:

\[
\Delta MAE = MAE_0-MAE_1
\]

\[
\Delta R^2=R^2_1-R^2_0
\]

---

# 32. Metric-ranking divergence

If RMSE and MAE rank differently:

```text
METRIC_RANKING_DIVERGENCE
```

Winner still follows RMSE.

Do not cherry-pick metric.

---

# 33. No significance testing

One seed each.

Do not:

```text
t-test epochs
bootstrap epochs as independent runs
report CI from epoch trajectory
```

---

# 34. Best-checkpoint verification

New TF0 run must:

```text
train
↓
save BEST
↓
fresh Transformer
↓
strict-load BEST
↓
full Validation
↓
inverse YS1
↓
METRICS-v1
↓
reproduce recorded metric
```

TF1 reused run must already have equivalent verification PASS.

---

# 35. Allowed config differences

Expected differences TF0 vs TF1:

```text
time_feature_id
feature_variant_id
feature_fingerprint
feature_count
x_scaler_id/checksum
input_size
trainable_parameter_count
input-projection width-derived fields
```

No other semantic difference.

---

# 36. Forbidden config differences

Must not differ:

```text
FS
L
H
YS
WB
population
batch
seed
d_model
heads
layers
FFN
dropout
activation
pooling
PE
norm
mask policy
optimizer
LR
WD
loss
epoch cap
patience
clipping
Training Engine
Metric version
```

---

# 37. S2 run matrix

Create:

```text
s2_run_matrix.csv
```

Fields:

```text
sweep_id
selected_feature_set_id
variant_id
time_feature_id
source_type
source_run_id
requires_new_training
feature_count
feature_fingerprint
x_scaler_id
y_scaler_id
lookback
horizon
population_fingerprint
batch_size
seed
model_config_id
training_config_id
status
```

---

# 38. Sweep manifest

Create:

```text
s2_time_feature_sweep_manifest.json
```

Minimum:

```text
sweep_version
sweep_id
selected_feature_set_id
source_s1_winner_run_id
reference_variant_id
candidate_variants
new_runs_required
reused_runs
swept_field = time_feature_id
frozen_fields
primary_metric
selection_direction = MIN
tie_rule = TF0_ON_EXACT_RMSE_TIE
population_version
population_fingerprint
metric_version
training_engine_version
seed = 42
inherited_random_control_warning
test_access = forbidden
status
created_at
```

---

# 39. Sweep contract

Create:

```text
s2_time_feature_sweep_contract.json
```

Must state:

```text
S1-selected FS is fixed.
Only TF0/TF1 changes.
TF1 = exactly 5 registered engineered calendar features.
No target-time feature.
Same population.
Correct variant X scalers.
Same YS1 scaler.
Same Transformer/training protocol.
Seed 42.
Validation RMSE selects winner.
Test forbidden.
```

---

# 40. Preflight audit

Create:

```text
s2_time_feature_preflight_audit.csv
```

Checks:

```text
phase23_pass
approved_for_phase24
s1_winner_valid
selected_FS_locked
TF0_registered
TF1_registered
TF1_reference_exact_match
population_fixed
Y_scaler_fixed
Training_Engine_fixed
metric_fixed
seed_fixed
test_locked
status
```

---

# 41. Definition audit

Create:

```text
s2_time_feature_definition_audit.csv
```

Rows:

```text
hour_sin
hour_cos
dow_sin
dow_cos
weekend
```

Fields:

```text
feature_name
expected_in_TF0
expected_in_TF1
feature_group
availability_semantics
model_eligible
pass_through_scaling
range_check
status
```

---

# 42. Feature-delta audit

Create:

```text
s2_feature_delta_audit.csv
```

Must prove:

```text
TF1 - TF0 = five features
TF0 - TF1 = empty
```

with canonical order metadata.

---

# 43. Population audit

Create:

```text
s2_time_feature_population_audit.csv
```

Fields:

```text
variant_id
split_id
sample_count
first_sample_id
last_sample_id
population_fingerprint
id_set_matches_reference
ordered_ids_match_reference
status
```

---

# 44. Scaler audit

Create:

```text
s2_time_feature_scaler_audit.csv
```

Fields:

```text
variant_id
time_feature_id
x_scaler_id
x_scaler_checksum
y_scaler_id
y_scaler_checksum
continuous_scaling_policy
time_feature_pass_through_verified
train_only_verified
status
```

---

# 45. Cyclical audit

Create:

```text
s2_cyclical_feature_audit.csv
```

Fields:

```text
feature_pair
min_value
max_value
max_unit_circle_error
finite
weekend_domain_valid
status
```

---

# 46. Architecture audit

Create:

```text
s2_time_feature_architecture_audit.csv
```

Fields:

```text
variant_id
input_size
d_model
num_heads
num_layers
ffn_dim
dropout
activation
pooling
norm_policy
causal_mask
padding_mask
trainable_parameters
only_allowed_differences
status
```

---

# 47. Training audit

Create:

```text
s2_time_feature_training_audit.csv
```

Fields:

```text
variant_id
batch_size
optimizer
learning_rate
weight_decay
criterion
max_epochs
patience
min_delta
gradient_clip
scheduler
mixed_precision
seed
training_engine_version
matches_reference
status
```

---

# 48. Run provenance

Create:

```text
s2_time_feature_run_provenance.csv
```

Fields:

```text
variant_id
time_feature_id
run_id
source_type
source_phase
config_fingerprint
feature_fingerprint
population_fingerprint
best_checkpoint_sha256
history_sha256
metric_artifact
prediction_artifact
status
```

---

# 49. Metrics table

Create:

```text
s2_time_feature_metrics.csv
```

Rows:

```text
FSx_TF0
FSx_TF1
```

Fields:

```text
selected_feature_set
variant_id
time_feature_id
run_id
source_type
feature_count
trainable_parameters
best_epoch
epochs_completed
stop_reason
validation_mae_wh
validation_rmse_wh
validation_r2
rmse_rank
is_empirical_winner
population_fingerprint
metric_version
status
```

---

# 50. Effect table

Create:

```text
s2_time_feature_effect.csv
```

Fields:

```text
selected_feature_set
tf0_run_id
tf1_run_id
tf0_rmse_wh
tf1_rmse_wh
rmse_delta_tf0_to_tf1_wh
rmse_improvement_pct
tf0_mae_wh
tf1_mae_wh
mae_delta_wh
tf0_r2
tf1_r2
r2_delta
rmse_winner
metric_ranking_divergence
status
```


---

# 51. Hypothesis outcomes

Create:

```text
s2_hypothesis_outcomes.csv
```

Fields:

```text
hypothesis_id
source_hypothesis_id_optional
selected_feature_set
comparison
expected_direction
tf0_rmse
tf1_rmse
observed_delta
outcome
interpretation
status
```

Allowed outcome:

```text
SUPPORTED
NOT_SUPPORTED
INCONCLUSIVE_TIE
```

Meaning only within:

```text
single-seed Validation context.
```

---

# 52. Findings artifact

Create:

```text
s2_time_feature_findings.csv
```

Possible codes:

```text
TIME_FEATURE_GAIN
TIME_FEATURE_NO_GAIN
TIME_FEATURE_EXACT_TIE
METRIC_RANKING_DIVERGENCE
CONVERGENCE_DIFFERENCE
GRADIENT_BEHAVIOR_DIFFERENCE
RUNTIME_DIFFERENCE
INHERITED_RANDOM_CONTROL_WARNING
```

Every finding should include:

```text
evidence
interpretation
confidence
severity
future-phase relevance
```

---

# 53. Winner artifact

Create:

```text
s2_time_feature_winner.json
```

Minimum fields:

```text
sweep_id
sweep_version
selected_feature_set_id
selection_metric
selection_direction
tie_rule
winner_variant_id
winner_time_feature_id
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_variant_id
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
population_fingerprint
metric_version
inherited_random_control_warning
test_status
status
```

---

# 54. Reference update for Phase 25

Create:

```text
s2_reference_update.json
```

Minimum:

```text
previous_reference_run_id
selected_feature_set_id
previous_time_feature_id = TF1
s2_winner_variant_id
selected_time_feature_id
winner_run_id
winner_config_fingerprint
winner_feature_fingerprint
winner_rmse_wh
target_scaling_state = YS1
selection_metric
inherited_random_control_warning
approved_for_phase25
```

---

# 55. Phase 25 handoff logic

Phase 25 tests:

```text
YS0
vs
YS1
```

with S2 winner feature variant fixed.

Examples:

```text
S2 winner = FS1_TF0
→ Phase 25 compares:
FS1_TF0 + YS0
vs
FS1_TF0 + YS1
```

```text
S2 winner = FS2_TF1
→ Phase 25 compares:
FS2_TF1 + YS0
vs
FS2_TF1 + YS1
```

The S2 winner already uses YS1, so Phase 25 should normally reuse it as YS1 reference and train only YS0 if all fields match.

---

# 56. Winner does not rewrite historical phases

Even if TF0 wins:

```text
S1 result remains historical S1 result.
```

Phase 24 only updates:

```text
current reference configuration.
```

Do not edit Phase 23 artifacts.

---

# 57. Sweep learning curves

Recommended figures:

```text
S2_01_validation_rmse_by_epoch.png
S2_02_validation_mae_by_epoch.png
S2_03_train_loss_by_epoch.png
S2_04_gradient_clipping_fraction.png
S2_05_best_validation_metrics.png
```

Primary figure:

```text
S2_01_validation_rmse_by_epoch.png
```

with:

```text
TF0
TF1
```

and best epoch markers.

---

# 58. Figure rules

Figures must be source-generated from:

```text
training_history.csv
s2_time_feature_metrics.csv
```

No manually typed values.

No smoothing needed for core comparison.

If smoothing is shown:

```text
visual only
raw curve remains visible
never used for winner selection.
```

---

# 59. Interpretation when TF1 wins

Safe language:

> Under the S1-selected feature set and frozen Transformer/training protocol, adding the five registered historical calendar features reduced Validation RMSE by the observed amount.

Do not claim:

```text
the Transformer definitely learned daily seasonality
```

because this sweep only measures empirical utility.

---

# 60. Interpretation when TF0 wins

Safe:

> Explicit engineered time features did not improve Validation RMSE under the current controlled configuration.

Do not claim:

```text
time-of-day or day-of-week has no relationship with energy consumption.
```

The 24-hour sequence and sensors may already encode related signals.

---

# 61. Interpretation when exact tie

Use:

```text
TF0
```

by predeclared parsimony rule.

Report:

```text
No full-precision Validation RMSE advantage was observed for TF1.
```

---

# 62. Tiny non-zero margin

If TF1 or TF0 wins by very small margin:

```text
winner rule still applies
```

but report:

```text
small single-seed Validation margin
```

and avoid strong interpretation.

---

# 63. Single-seed limitation

Mandatory:

```text
S2 conditions are represented by seed-42 runs.
```

No:

```text
mean ± std
confidence interval
significance claim.
```

---

# 64. Validation-only limitation

Mandatory:

```text
S2 winner is a Validation-selected development configuration.
```

No Test evidence.

---

# 65. Sequential selection caution

By Phase 24, Validation has been used for:

```text
S1 feature-set selection
S2 time-feature selection
```

Therefore:

```text
every tested condition must remain registered
no hidden experiments
rolling-origin robustness later
multi-seed final runs later
Test remains untouched.
```

---

# 66. No extra time-feature ablations

Do not secretly add:

```text
hour-only
dow-only
weekend-only
month
holiday
raw integer hour
one-hot weekday
target-hour
```

Current S2 factor is strictly:

```text
TF0 vs TF1.
```

Any decomposition requires a separate pre-registered experiment or protocol amendment.

---

# 67. No target-time calendar feature

Hard.

Current sequence formulation remains:

```text
historical rows only
```

even though target-time calendar may be operationally knowable.

---

# 68. No timezone reinterpretation

Use `TEMPORAL-v1`.

Do not:

```text
localize timestamps
shift timezone
reinterpret day boundary
```

inside S2.

---

# 69. No raw timestamp input

`timestamp` remains metadata.

---

# 70. No absolute time index

No.

---

# 71. No alternative cyclical encoding

Do not compare:

```text
sin/cos
vs
one-hot
vs
integer.
```

Not part of master plan.

---

# 72. No scaling change

Do not standardize time features just for TF1.

Do not apply MinMax/RobustScaler.

Use frozen `SCALING-v1`.

---

# 73. No sample weighting

No.

---

# 74. No feature importance/SHAP

Not Phase 24.

---

# 75. No attention analysis

Not Phase 24.

---

# 76. No residual regime analysis

Not Phase 24.

---

# 77. Run failure policy

If TF0 technical run fails:

```text
S2 is incomplete.
```

Do not automatically declare TF1 winner.

Resolve using Experiment Registry / Training Engine policy.

---

# 78. Score-based rerun forbidden

Do not rerun TF0 because its score looks poor.

Allowed reruns require documented technical reasons:

```text
interrupt
corrupt checkpoint
infrastructure failure
software bug.
```

If semantic config changes:

```text
new condition / protocol issue
```

not same run.

---

# 79. OOM policy

If TF0 B64 OOM:

```text
FAIL run.
```

Do not silently switch to B32.

Batch sweep is Phase 29.

---

# 80. Nonfinite policy

If:

```text
prediction
loss
gradient norm
```

contains NaN/Inf:

```text
FAIL.
```

No skip.

---

# 81. No emergency optimization edits

Do not change mid-run:

```text
LR
WD
dropout
clipping
epoch cap
batch size
loss.
```

---

# 82. No Test tie-breaker

Hard forbidden.

---

# 83. Sweep discrepancy taxonomy

```text
S1_REFERENCE_MISSING
S1_WINNER_MISMATCH
SELECTED_FEATURE_SET_DRIFT
TF_VARIANT_MISSING
TF_FEATURE_DEFINITION_MISMATCH
FEATURE_DELTA_MISMATCH
POPULATION_MISMATCH
SCALER_MISMATCH
TIME_FEATURE_SCALING_MISMATCH
CYCLICAL_INVARIANT_FAILURE
MODEL_CONFIG_DRIFT
TRAINING_CONFIG_DRIFT
SEED_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
REFERENCE_RUN_MISMATCH
RUN_FAILURE
CHECKPOINT_VERIFICATION_FAILURE
INCOMPLETE_SWEEP
RANKING_ERROR
EFFECT_CALCULATION_ERROR
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 84. Discrepancy log

Create:

```text
s2_time_feature_discrepancies.json
```

Fields:

```text
id
severity
category
variant_or_pair
expected
actual
impact
recommended_action
resolved
resolution_notes
```

Severity:

```text
CRITICAL
MAJOR
MODERATE
MINOR
INFO
```

Examples:

```text
CRITICAL:
population mismatch
Test access
Training Engine mismatch

MAJOR:
feature delta wrong
scaler wrong
checkpoint verification failure

MODERATE:
metric ranking divergence
inherited random-control warning

INFO:
runtime/parameter count differences.
```

---

# 85. Status model

## PASS

```text
S1 reference valid
TF0/TF1 represented
same population
correct scaling
feature-delta audit PASS
new TF0 run verified
winner selected
Phase 25 reference generated
Test untouched
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
metric ranking divergence
inherited RANDOM_CONTROL_GAIN warning
```

with all core contracts valid.

## FAIL

Possible:

```text
wrong selected FS
wrong TF definition
population mismatch
wrong scaler
unresolved TF0 failure
reference mismatch
Test access.
```

---

# 86. Sweep summary

Create:

```text
s2_time_feature_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
selected_feature_set
reference_run_id
tf0_run_id
tf1_run_id
new_runs
reused_runs
primary_metric
tf0_metrics
tf1_metrics
time_feature_effect
winner
winner_margin
metric_ranking_divergence
inherited_warnings
phase25_reference
test_status
overall_status
```

---

# 87. Human-readable report

Create:

```text
s2_time_feature_sweep_report.md
```

Sections:

```text
1. Objective
2. S1-selected feature set
3. Controlled variable
4. TF0/TF1 definitions
5. Frozen configuration
6. Reference provenance
7. Population fairness
8. Scaling/pass-through fairness
9. Cyclical integrity
10. Validation metrics
11. Time-feature effect
12. Learning-curve context
13. S2 winner
14. Limitations
15. Phase 25 handoff
```

Each result section should distinguish:

```text
Observed
Interpretation
Limitation
Handoff
```

---

# 88. README

Create:

```text
README_S2_TIME_FEATURE_SWEEP.md
```

Must explain:

```text
Purpose
Why S1-selected FS is frozen
TF0/TF1 definition
Why exactly 5 features differ
No target-time features
Why TF1 reference is reused
Population fairness
Scaler/pass-through semantics
Cyclical checks
Input-size/parameter-count nuance
Training protocol
Winner rule
Tie rule
Limitations
Inherited random-control warning if applicable
Phase 25 handoff
No Test.
```

---

# 89. Output directory

```text
artifacts/
└── sweeps/
    └── S2_time_feature/
        ├── s2_time_feature_sweep_manifest.json
        ├── s2_time_feature_sweep_contract.json
        ├── s2_time_feature_preflight_audit.csv
        ├── s2_run_matrix.csv
        ├── s2_time_feature_definition_audit.csv
        ├── s2_feature_delta_audit.csv
        ├── s2_time_feature_population_audit.csv
        ├── s2_time_feature_scaler_audit.csv
        ├── s2_cyclical_feature_audit.csv
        ├── s2_time_feature_architecture_audit.csv
        ├── s2_time_feature_training_audit.csv
        ├── s2_time_feature_run_provenance.csv
        ├── s2_time_feature_metrics.csv
        ├── s2_time_feature_effect.csv
        ├── s2_hypothesis_outcomes.csv
        ├── s2_time_feature_findings.csv
        ├── s2_time_feature_winner.json
        ├── s2_reference_update.json
        ├── s2_time_feature_sweep_tests.csv
        ├── s2_time_feature_discrepancies.json
        ├── s2_time_feature_sweep_summary.json
        ├── s2_time_feature_sweep_report.md
        ├── figures/
        │   ├── S2_01_validation_rmse_by_epoch.png
        │   ├── S2_02_validation_mae_by_epoch.png
        │   ├── S2_03_train_loss_by_epoch.png
        │   ├── S2_04_gradient_clipping_fraction.png
        │   └── S2_05_best_validation_metrics.png
        ├── README_S2_TIME_FEATURE_SWEEP.md
        └── phase_24_signoff.json
```

New TF0 run remains under:

```text
artifacts/runs/<run_id>/
```

No large checkpoint duplication inside sweep directory.

---

# 90. Required outputs

```text
O24.1  Sweep manifest
O24.2  Sweep contract
O24.3  Preflight audit
O24.4  Run matrix
O24.5  Time-feature definition audit
O24.6  Feature-delta audit
O24.7  Population audit
O24.8  Scaler audit
O24.9  Cyclical invariant audit
O24.10 Architecture audit
O24.11 Training-config audit
O24.12 Run provenance
O24.13 Reused TF1 reference
O24.14 Verified new TF0 run
O24.15 Metrics table
O24.16 Time-feature effect table
O24.17 Hypothesis outcomes
O24.18 Findings
O24.19 Winner artifact
O24.20 Phase 25 reference update
O24.21 Figures
O24.22 Sweep test suite
O24.23 Discrepancy log
O24.24 Sweep summary
O24.25 Human-readable report
O24.26 README
O24.27 Phase sign-off
```

---

# 91. Sweep test suite

Create:

```text
s2_time_feature_sweep_tests.csv
```

Recommended checks:

```text
S2T24-001 Phase 23 PASS/non-critical warning only
S2T24-002 approved_for_phase24 = true
S2T24-003 S1 winner artifact valid
S2T24-004 selected FS valid
S2T24-005 exactly one TF0 and one TF1 variant
S2T24-006 selected FS same in both
S2T24-007 TF0 has no 5 time features
S2T24-008 TF1 has all 5
S2T24-009 TF1-TF0 exact feature delta
S2T24-010 no unexpected removed feature
S2T24-011 no target-time feature
S2T24-012 L144 fixed
S2T24-013 H1 fixed
S2T24-014 YS1 fixed
S2T24-015 WB0 fixed
S2T24-016 WINDOWPOP-v1 fixed
S2T24-017 B64 fixed
S2T24-018 D64 fixed
S2T24-019 H4 fixed
S2T24-020 N2 fixed
S2T24-021 FFN128 fixed
S2T24-022 dropout .1 fixed
S2T24-023 GELU fixed
S2T24-024 LAST_STEP fixed
S2T24-025 AdamW fixed
S2T24-026 LR 3e-4 fixed
S2T24-027 WD 1e-4 fixed
S2T24-028 MSE fixed
S2T24-029 E50 fixed
S2T24-030 patience 10 fixed
S2T24-031 clipping 1.0 fixed
S2T24-032 seed 42 fixed
S2T24-033 Training Engine fixed
S2T24-034 Metric version fixed
S2T24-035 TF1 reference exact-match
S2T24-036 TF1 reused, not retrained
S2T24-037 TF0 feature fingerprint valid
S2T24-038 TF1 feature fingerprint valid
S2T24-039 TF0 X scaler correct
S2T24-040 TF1 X scaler correct
S2T24-041 same Y scaler
S2T24-042 time-feature pass-through verified
S2T24-043 cyclical ranges valid
S2T24-044 unit-circle checks valid
S2T24-045 weekend domain valid
S2T24-046 Train target IDs identical
S2T24-047 Validation IDs identical
S2T24-048 population fingerprint identical
S2T24-049 only allowed config differences
S2T24-050 TF0 registered before training
S2T24-051 TF0 fresh loaders/model
S2T24-052 TF0 no warm-start
S2T24-053 TF0 seed 42
S2T24-054 TF0 trained via TRAINING_ENGINE-v1
S2T24-055 TF0 best checkpoint verified
S2T24-056 TF1 best checkpoint already verified
S2T24-057 both metrics from verified BEST
S2T24-058 full-precision RMSE available
S2T24-059 RMSE effect correct
S2T24-060 relative improvement correct
S2T24-061 MAE effect correct
S2T24-062 R² effect correct
S2T24-063 winner = minimum RMSE
S2T24-064 exact tie → TF0
S2T24-065 metric divergence recorded if present
S2T24-066 hypothesis outcome generated
S2T24-067 winner points to valid run
S2T24-068 reference update generated
S2T24-069 Phase 25 preserves S2 winner FS/TF
S2T24-070 inherited random-control warning propagated
S2T24-071 no failed/SANITY run in ranking
S2T24-072 no score-based rerun
S2T24-073 no ad hoc time feature
S2T24-074 no target-time feature
S2T24-075 no Test access
S2T24-076 single-seed limitation documented
S2T24-077 Validation-only limitation documented
S2T24-078 figures source-derived
S2T24-079 summary/report generated
S2T24-080 phase sign-off generated
```

---

# 92. Recommended notebook structure

```text
Cell 24.1  Phase title
Cell 24.2  Verify Phase 23 sign-off
Cell 24.3  Declare SWEEP_S2_TIMEFEATURE-v1
Cell 24.4  Load S1 winner/reference update
Cell 24.5  Resolve selected FSx
Cell 24.6  Load FSx_TF0/FSx_TF1 registry entries
Cell 24.7  Build S2 run matrix
Cell 24.8  Audit time-feature definitions
Cell 24.9  Audit exact feature delta
Cell 24.10 Audit population identity
Cell 24.11 Audit scaler bindings
Cell 24.12 Audit cyclical/pass-through invariants
Cell 24.13 Audit architecture/training frozen fields
Cell 24.14 Verify TF1 reference reuse eligibility
Cell 24.15 Register TF0 run
Cell 24.16 Seed + fresh TF0 loaders/model
Cell 24.17 Execute TF0 via Training Engine
Cell 24.18 Verify TF0 best checkpoint
Cell 24.19 Build run provenance
Cell 24.20 Build metrics table
Cell 24.21 Compute TF0→TF1 effect
Cell 24.22 Evaluate hypotheses
Cell 24.23 Generate learning curves
Cell 24.24 Generate findings
Cell 24.25 Select S2 winner
Cell 24.26 Write winner JSON
Cell 24.27 Write Phase 25 reference update
Cell 24.28 Run S2 tests/discrepancies
Cell 24.29 Write summary/report
Cell 24.30 Register artifacts/checksums
Cell 24.31 Write README
Cell 24.32 Phase sign-off
```

---

# 93. Execution flow

```text
Verify Phase 23
        ↓
Load S1 winner
        ↓
Freeze FSx
        ↓
Resolve FSx_TF0 / FSx_TF1
        ↓
Audit exact 5-feature delta
        ↓
Audit same target population
        ↓
Audit correct scalers
        ↓
Audit cyclical/pass-through integrity
        ↓
Verify TF1 reference exact match
        ↓
Reuse TF1
        ↓
Register TF0
        ↓
Seed 42
        ↓
Fresh TF0 loaders/model
        ↓
TRAINING_ENGINE-v1
        ↓
Verify TF0 BEST
        ↓
Build two-row metric table
        ↓
Compute TF effect
        ↓
Select by full-precision RMSE
        ↓
Apply TF0 tie rule if exact tie
        ↓
Update Phase 25 reference
        ↓
Write S2 artifacts
        ↓
SWEEP_S2_TIMEFEATURE-v1 sign-off
```

---

# 94. Fail-fast order

Before training TF0:

```text
1. Phase 23 sign-off
2. S1 winner identity
3. Selected FS validity
4. TF0/TF1 registry entries
5. Exact five-feature delta
6. Population identity
7. X/Y scaler binding
8. TF1 cyclical integrity
9. Frozen architecture/training fields
10. TF1 reuse eligibility
11. Test firewall
12. Registry readiness
```

---

# 95. Why feature-delta audit must happen first

Common silent errors:

```text
TF0 still contains weekend
TF1 missing dow_cos
manual column drop changes order
wrong FS is used
target-time calendar accidentally added
```

Any one makes S2 scientifically invalid.

---

# 96. Why pass-through audit matters

If sin/cos/weekend are accidentally standardized, S2 is no longer using frozen `SCALING-v1`, so effect cannot be attributed only to time-feature presence.

---

# 97. Why population identity matters

Different targets would make RMSE difference potentially caused by evaluation set composition, not TF0/TF1.

---

# 98. Why reuse reference matters

Avoid:

```text
duplicate compute
extra stochastic realization
hidden score cherry-picking
reference ambiguity
```

---

# 99. Winner verification checklist

Before writing winner:

```text
[ ] TF0 valid completed run.
[ ] TF1 valid reused reference.
[ ] Same selected FS.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same metric version.
[ ] Same target unit Wh.
[ ] Same seed policy.
[ ] Same model/training config.
[ ] Only allowed time-feature-derived fields differ.
[ ] Both BEST checkpoints verified.
[ ] Full-precision RMSE used.
[ ] Effect computed programmatically.
[ ] Exact tie rule respected.
[ ] No Test.
```

---

# 100. Phase 25 handoff

Phase 25 receives:

```text
s2_time_feature_winner.json
s2_reference_update.json
winner_run_id
winner_config_fingerprint
winner_feature_fingerprint
selected_feature_set_id
selected_time_feature_id
```

Phase 25 must change only:

```text
target_scaling = YS0 / YS1
```

and normally reuse S2 winner as YS1 reference.

---

# 101. If TF0 wins

Current reference:

```text
FSx_TF0
YS1
```

Phase 25:

```text
same FSx_TF0
YS0 vs YS1.
```

---

# 102. If TF1 wins

Current reference:

```text
FSx_TF1
YS1
```

Phase 25:

```text
same FSx_TF1
YS0 vs YS1.
```

---

# 103. If exact tie

TF0 selected by parsimony.

Phase 25 uses TF0.

---

# 104. Relationship with later phases

Phase 24 does not test:

```text
lookback
pooling
activation
batch
LR
WD
dropout
capacity
loss
epoch cap
clipping
RevIN
boundary.
```

Those remain Phase 26–41.

---

# 105. Full experimental history preservation

Both TF0 and TF1 conditions remain in Registry.

Losing condition must not be deleted.

Phase 42 can trace full sequential search.

---

# 106. Rolling-origin importance

S2 winner is still one Validation split.

Phase 44 later checks temporal robustness.

---

# 107. Multi-seed importance

S2 is seed 42.

Do not treat tiny margin as seed-stable evidence.

---

# 108. Test remains untouched

No Test access until final evaluation phase.

---

# 109. Reproducibility metadata

New TF0 run records:

```text
run_id
seed
environment
device
code fingerprint
feature fingerprint
X scaler checksum
Y scaler checksum
population fingerprint
model config fingerprint
Training Engine fingerprint
best checkpoint checksum
history checksum
metric checksum
```

TF1 keeps existing provenance.

---

# 110. Phase sign-off

Create:

```text
phase_24_signoff.json
```

Minimum:

```text
phase = 24
phase_name = S2 Time-feature sweep
sweep_version
sweep_id
source_s1_winner_run_id
selected_feature_set_id
tf0_run_id
tf1_reference_run_id
new_run_ids
reused_run_ids
winner_variant_id
winner_time_feature_id
winner_run_id
winner_rmse_wh
population_fingerprint
metric_version
fairness_audit_status
inherited_random_control_warning
test_status
approved_for_phase25
overall_status
created_at
```

---

# 111. Acceptance checklist

```text
[ ] Phase 23 valid.
[ ] approved_for_phase24 = true.
[ ] S2 version declared.
[ ] S1 winner loaded.
[ ] Selected FS frozen.
[ ] TF0/TF1 resolved from registry.
[ ] Same FS in both.
[ ] TF0 contains no registered time feature.
[ ] TF1 contains exactly all five.
[ ] Exact five-feature delta verified.
[ ] No target-time feature.
[ ] No unexpected feature delta.
[ ] L144/H1/YS1/WB0 fixed.
[ ] WINDOWPOP-v1 fixed.
[ ] B64 fixed.
[ ] D64/H4/N2/FFN128 fixed.
[ ] Dropout/GELU/LAST_STEP fixed.
[ ] AdamW/LR/WD/MSE fixed.
[ ] E50/patience10/clip1 fixed.
[ ] Seed 42 fixed.
[ ] Training Engine fixed.
[ ] Metric version fixed.
[ ] Correct TF0 X scaler.
[ ] Correct TF1 X scaler.
[ ] Same Y scaler.
[ ] Pass-through semantics verified.
[ ] Cyclical ranges verified.
[ ] Unit-circle checks verified.
[ ] Weekend domain verified.
[ ] Train IDs identical.
[ ] Validation IDs identical.
[ ] Population fingerprint identical.
[ ] TF1 reference exact-match.
[ ] TF1 not retrained.
[ ] TF0 registered before training.
[ ] TF0 fresh loaders/model.
[ ] TF0 no warm-start.
[ ] TF0 trained via TRAINING_ENGINE-v1.
[ ] TF0 BEST verified.
[ ] TF1 BEST already verified.
[ ] Both metrics source from verified BEST.
[ ] Full-precision RMSE used.
[ ] Effect formulas correct.
[ ] Winner selected by minimum RMSE.
[ ] Exact tie → TF0.
[ ] Metric divergence recorded if present.
[ ] Hypothesis outcome recorded.
[ ] Findings recorded.
[ ] Inherited S1 warning propagated if needed.
[ ] Winner artifact generated.
[ ] Phase 25 reference update generated.
[ ] Phase 25 reuse of YS1 identified.
[ ] No hidden extra TF experiment.
[ ] No score-based rerun.
[ ] No failed/SANITY run in ranking.
[ ] No Test access.
[ ] Limitations documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancies recorded.
[ ] Phase sign-off generated.
```

---

# 112. Acceptance criteria

Phase 24 only PASS when:

```text
S1-selected FS is fixed.

Exactly TF0 and TF1 are compared.

Semantic difference is exactly five registered time features.

Same Train/Validation target IDs are used.

Correct variant-specific X scalers are used.

Same YS1 scaler is used.

TF1 time features remain pass-through.

Cyclical integrity checks pass.

TF1 reference is valid and reused.

TF0 is one fresh seed-42 run.

No warm-start occurs.

All other model/training fields match.

Best checkpoints are verified.

Validation RMSE Wh selects winner.

Exact RMSE tie selects TF0.

Phase 25 reference is created.

No hidden experiment/rerun occurs.

Test remains untouched.
```

---

# 113. Failure conditions

Phase 24 FAIL if:

```text
wrong S1 FS used
TF0/TF1 differ in multiple conceptual factors
TF1 missing/adding wrong feature
target-time feature added
timestamp interpretation changes
population differs
wrong scaler bound
pass-through policy violated
Y scaler differs
TF1 reference mismatched but reused
TF1 retrained and best rerun cherry-picked
TF0 warm-started
seed/training config drift
one condition fails but winner still declared
RMSE rounded before ranking
MAE overrides lower RMSE
Test used
score-based rerun occurs.
```

---

# 114. Common mistakes

## 114.1 Hard-code FS1

Wrong. Use S1 winner.

## 114.2 Compare FS1_TF0 against FS2_TF1

Changes feature set and time features together.

## 114.3 Keep weekend in TF0

Violates TF0.

## 114.4 Add target hour

New factor.

## 114.5 Standardize sin/cos in S2

Scaling drift.

## 114.6 Use shared X scaler

Wrong frozen scaler binding.

## 114.7 Retrain TF1 and keep better rerun

Hidden rerun bias.

## 114.8 Warm-start TF0

Confounded.

## 114.9 Try holiday/month after TF1 loses

Not S2.

## 114.10 Choose smaller model despite worse RMSE

Parsimony only exact tie.

## 114.11 Choose higher R² over lower RMSE

Wrong primary metric.

## 114.12 Use Test

Forbidden.

## 114.13 Claim “no daily pattern” if TF0 wins

Unsupported.

## 114.14 Claim “Transformer learned seasonality” if TF1 wins

Unsupported.

## 114.15 Run multiple seeds and pick best

Not Phase 24.

---

# 115. Recommended execution pseudocode

```text
load_phase23_signoff()
assert_approved_for_phase24()

s1 = load_s1_winner()
FSx = s1.selected_feature_set_id

tf0 = FEATURESETS_v1[FSx + "_TF0"]
tf1 = FEATURESETS_v1[FSx + "_TF1"]

audit_time_feature_definition(tf0, tf1)
audit_exact_five_feature_delta(tf0, tf1)
audit_same_population(tf0, tf1)
audit_scalers(tf0, tf1)
audit_cyclical_integrity(tf1)

tf1_reference = resolve_s1_winner_source_run()
assert_exact_s2_reference_match(tf1_reference)

register_tf0_run()
seed(42)

loaders = build_fresh_loaders(tf0)
model = build_fresh_transformer(
    input_size=feature_count(tf0)
)

tf0_result = TRAINING_ENGINE_v1.fit(...)
verify_best_checkpoint(tf0_result)

results = {
    "TF0": tf0_result,
    "TF1": tf1_reference
}

metrics = build_s2_metrics(results)
effect = compute_time_feature_effect(metrics)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer="TF0"
)

write_hypothesis_outcomes()
write_findings()
write_s2_winner(winner)
write_phase25_reference_update(winner)
write_sweep_summary_report_readme_signoff()
```

---

# 116. Definition of Done

\[
\boxed{
One\ Fixed\ Feature\ Set
+
Two\ Time\ Conditions
+
One\ Fresh\ TF0\ Run
+
One\ Valid\ Reused\ TF1
+
Same\ Population
+
Correct\ Scaling
+
Verified\ Metrics
+
S2\ Winner
+
Phase25\ Reference
+
No\ Test
}
\]

---

# 117. Final status contract

```text
PHASE 24 tests TIME FEATURES only.

Feature set:
comes from S1 winner.

Conditions:
FSx_TF0
FSx_TF1

TF1:
hour_sin
hour_cos
dow_sin
dow_cos
weekend

TF1 reference:
reuse S1 winner if exact match.

TF0:
one new fresh run.

Everything else:
frozen.

Population:
WINDOWPOP-v1 identical.

Scaling:
variant-specific X scaler,
same YS1 scaler,
time features pass through.

Training:
TRAINING_ENGINE-v1
seed 42
B64
AdamW
LR 3e-4
WD 1e-4
MSE
E50
patience 10
clip 1.0.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer TF0.

No new calendar feature.
No target-time feature.
No hidden rerun.
No Test.

After SWEEP_S2_TIMEFEATURE-v1 PASS:
update current reference
and proceed to
PHASE 25 — S3 Target-scaling sweep.
```

---

# 118. Final check

Correct Phase 24 order:

```text
Load S1 winner
→ Freeze FS
→ Resolve TF0/TF1
→ Audit exact 5-feature delta
→ Audit same targets
→ Audit scalers
→ Audit cyclical integrity
→ Reuse TF1
→ Train only TF0
→ Verify BEST
→ Compare full-precision Validation RMSE
→ Select S2 winner
→ Update Phase 25 reference
```

Incorrect workflow:

```text
try several calendar encodings
→ look at score
→ add target-time information
→ retrain TF1
→ keep whichever run looks best.
```

Only after `SWEEP_S2_TIMEFEATURE-v1` is signed off may the project proceed to **PHASE 25 — S3 Target-scaling sweep**.
