# PHASE 25 — S3 TARGET-SCALING SWEEP

## Kế hoạch controlled sweep cho Target Scaling của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S2_TIMEFEATURE-v1`  
**Sweep ID:** `S3_TARGET_SCALING`  
**Output version:** `SWEEP_S3_TARGETSCALING-v1`  
**Phase trước:** `Phase_24_S2_Time-feature_sweep.md`

---

# 1. Vai trò của Phase 25

Phase 25 là controlled experiment thứ ba trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant đã được S1 và S2 chọn, cùng data population, Transformer architecture, optimizer, learning rate, training engine và seed giữ nguyên, việc **scale target `Appliances` hay không** ảnh hưởng như thế nào tới optimization behavior và Validation forecasting performance?

Phase này chỉ thay đúng một conceptual factor:

```text
TARGET SCALING
```

với hai condition:

```text
YS0
YS1
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Factor
+
Same\ Inputs
+
Same\ Samples
+
Same\ Model
+
Same\ Optimization\ Hyperparameters
+
Original\text{-}Wh\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí Phase 25 trong master execution plan

```text
Phase 23
→ S1 Feature-set sweep

Phase 24
→ S2 Time-feature sweep

Phase 25
→ S3 Target-scaling sweep

Phase 26
→ S4 Lookback sweep
```

Phase 25 không chọn lại feature set hoặc time features.

Nó tiếp nhận current feature variant đã được Phase 24 khóa và chỉ kiểm:

```text
YS0 vs YS1.
```

---

# 3. Câu hỏi nghiên cứu của S3

Phase 25 phải trả lời:

```text
1. Target standardization có giúp giảm Validation RMSE Wh không?

2. YS0 và YS1 ảnh hưởng training stability như thế nào khi giữ LR/optimizer cố định?

3. Gradient magnitude/clipping behavior thay đổi như thế nào?

4. Best epoch/early stopping behavior có khác nhau không?

5. MAE và R² có đồng thuận với RMSE ranking không?

6. Target scaling có giúp optimization dễ hơn trong current Transformer configuration không?

7. S3 winner nào trở thành current target-scaling reference cho Phase 26?
```

---

# 4. Hai target-scaling conditions

Source of truth:

```text
SCALING-v1
```

## 4.1 YS0 — No target scaling

Target giữ ở original unit:

```text
Wh
```

Training target:

```text
y_model = y_raw_wh
```

Model output:

```text
prediction_model_space = Wh
```

Metric prediction:

```text
prediction_wh = prediction_model_space
```

Không fit target scaler.

Conceptually:

```text
y_scaler = IDENTITY
```

---

# 5. YS1 — Train-only StandardScaler

YS1 dùng frozen train-only target scaler từ `SCALING-v1`.

Training target:

\[
z
=
\frac{y-\mu_{train}}{\sigma_{train}}
\]

Trong đó:

```text
mu_train
sigma_train
```

được fit chỉ từ target values thuộc TRAIN protocol theo Phase 9.

Model predicts:

```text
standardized target space
```

Khi evaluate:

\[
\hat y_{Wh}
=
\hat z \sigma_{train}
+
\mu_{train}
\]

Sau inverse transform mới tính:

```text
MAE Wh
RMSE Wh
R²
```

---

# 6. Primary scientific comparison

Comparison phải diễn ra trên:

```text
original target unit = Wh
```

Không so winner bằng:

```text
training MSE
Validation model-space MSE
standardized loss
```

vì YS0 và YS1 có objective scales khác nhau.

Hard rule:

\[
\boxed{
S3\ Winner
=
\arg\min Validation\ RMSE\ Wh
}
\]

---

# 7. Working hypotheses

Các hypothesis trước execution chỉ là expectations.

## H-S3-01 — Target scaling improves optimization

Nếu:

\[
RMSE(YS1)<RMSE(YS0)
\]

thì evidence support rằng target standardization có lợi cho current training setup.

Possible reason:

```text
better-conditioned output/loss scale
```

nhưng causal mechanism cần được diễn giải thận trọng.

Status:

```text
UNTESTED
```

---

# 8. H-S3-02 — No target-scaling benefit

Nếu:

```text
RMSE(YS0) <= RMSE(YS1)
```

thì target standardization không cải thiện current Validation RMSE.

Có thể vì:

```text
AdamW handles raw target scale sufficiently

current LR works adequately in Wh-space

standardization offers little advantage

single-seed stochasticity dominates a small margin.
```

Không được khẳng định một nguyên nhân nếu chưa có controlled evidence.

---

# 9. Target scaling là một optimization intervention

Điểm phương pháp rất quan trọng:

YS0 và YS1 không chỉ khác numeric representation của target.

Với cùng:

```text
MSE
LR
optimizer
```

target scale làm thay đổi:

```text
loss magnitude
gradient magnitude
effective optimization geometry
output-layer update scale
clipping frequency
```

Do đó S3 đo **end-to-end effect of target scaling under the frozen optimizer configuration**.

Đây chính là intended experimental factor, không phải fairness violation.

---

# 10. Không được “compensate” YS0 bằng LR khác

Ví dụ không được:

```text
YS0 → LR 1e-5
YS1 → LR 3e-4
```

trong Phase 25.

Nếu làm vậy sẽ thay đồng thời:

```text
target scaling + learning rate.
```

LR sweep thuộc Phase 30.

---

# 11. Preconditions bắt buộc

Phase 25 chỉ bắt đầu khi:

```text
Phase 24 = PASS
```

hoặc `PASS_WITH_WARNING` không có unresolved critical issue.

Bắt buộc có:

```text
s2_time_feature_winner.json
s2_reference_update.json
phase_24_signoff.json
approved_for_phase25 = true
```

---

# 12. Resolve current input configuration

Phase 25 phải load Phase 24 winner để lấy:

```text
selected_feature_set_id
selected_time_feature_id
winner_variant_id
winner_run_id
winner_feature_fingerprint
```

Gọi current fixed feature variant:

```text
FV*
```

Ví dụ possible:

```text
FS0_TF0
FS0_TF1
FS1_TF0
FS1_TF1
FS2_TF0
FS2_TF1
```

Phase 25 không hard-code bất kỳ one variant nào.

---

# 13. Random-control warning carry-forward

Nếu current feature set = FS2 và upstream đã có:

```text
RANDOM_CONTROL_GAIN
```

warning phải tiếp tục xuất hiện trong:

```text
S3 manifest
S3 report
S3 winner artifact
Phase 26 reference update
```

Không tự xóa warning chỉ vì Phase 25 đang test target scaling.

---

# 14. Upstream contracts bắt buộc

Verify:

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
SWEEP_S2_TIMEFEATURE-v1
```

---

# 15. Swept factor duy nhất

```text
target_scaling_id
```

Allowed:

```text
YS0
YS1
```

---

# 16. Frozen feature/input configuration

Hard:

```text
feature_set_id = S2 winner FS
time_feature_id = S2 winner TF
feature_variant_id = S2 winner variant
feature_order = same
feature_fingerprint = same
X scaler = same
```

Target scaling không được làm thay đổi X preprocessing.

---

# 17. Frozen temporal/data factors

```text
L144
H1
WB0
WINDOWPOP-v1
same Train target IDs
same Validation target IDs
```

---

# 18. Frozen Transformer architecture

```text
input_size = same F
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
regression head = Linear(64,1)
no output activation
```

Because feature variant is fixed, model architecture and parameter count should be identical across YS0/YS1.

---

# 19. Frozen training hyperparameters

```text
batch_size = 64

optimizer = AdamW
learning_rate = 3e-4
weight_decay = 1e-4

criterion = MSE

max_epochs = 50
patience = 10
min_delta = 0

gradient clipping = ON
max_norm = 1.0

scheduler = None
mixed precision = False
gradient accumulation = 1
torch.compile = False

seed = 42

TRAINING_ENGINE-v1
METRICS-v1
```

---

# 20. Same input X is a hard requirement

YS0 and YS1 must consume exactly the same:

```text
X sample IDs
X feature order
X scaled values
X windows
```

Only target representation differs.

Recommended audit:

```text
hash selected X probes
```

or equivalent deterministic equality test.

---

# 21. Same raw target is a hard requirement

For each sample:

```text
y_raw_wh(YS0)
==
y_raw_wh(YS1)
```

YS1 only transforms the model target representation.

---

# 22. YS0 model target

For each sample:

```text
y_model_YS0 = y_raw_wh
```

Hard equality.

---

# 23. YS1 model target

For each Train/Validation sample:

```text
y_model_YS1
=
(y_raw_wh - mu_train) / sigma_train
```

using frozen train-only target scaler.

---

# 24. YS1 scaler provenance

Must record:

```text
y_scaler_id
fit_scope
fit_split
fit_target_period
mean
scale/std
checksum
SCALING-v1
```

Do not refit in Phase 25.

---

# 25. YS0 identity provenance

Create explicit identity target-transform record:

```text
target_transform_id = YS0_IDENTITY
fit_required = false
inverse_required = false
unit = Wh
```

Do not represent YS0 with a fake fitted scaler.

---

# 26. Train-only guarantee for YS1

Audit:

```text
no Validation target used in fit
no Test target used in fit
no refit before new run
```

---

# 27. Common target-population rule

Hard:

```text
Train IDs YS0 == Train IDs YS1
Validation IDs YS0 == Validation IDs YS1
population fingerprint same
```

Target scaling must never affect sample eligibility.

---

# 28. Why scaling must not change window validity

Window validity is determined by:

```text
temporal geometry
continuity
target timestamp
```

from Phase 10, not target normalization.

Any YS-specific sample count difference is:

```text
pipeline bug.
```

---

# 29. Existing YS1 reference reuse rule

The Phase 24 winner already uses:

```text
YS1
```

because S1/S2 were run under YS1.

If exact S3 contract match:

```text
REUSE S2 winner run
```

as YS1 reference.

Do not retrain YS1.

---

# 30. Normal S3 run count

Typically:

```text
YS1
→ reuse Phase 24 winner

YS0
→ one NEW training run
```

Therefore:

```text
1 new run
+
1 reused reference.
```

---

# 31. Reference reuse gate

S2 winner may be reused only when all match:

```text
current feature variant
L144
H1
WB0
WINDOWPOP-v1
YS1
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
patience10
clip1
seed42
Training Engine
Metric version
```

Mismatch:

```text
STOP.
```

---

# 32. Fresh YS0 run

For YS0:

```text
set seed 42
↓
fresh Train DataLoader
↓
fresh Validation DataLoader
↓
same X feature variant/scaler
↓
Dataset emits y_model = y_raw_wh
↓
fresh Transformer
↓
MSE
↓
AdamW
↓
TRAINING_ENGINE-v1
```

No warm-start from YS1.

---

# 33. Why no warm-start

Warm-start would confound:

```text
target scaling
+
pretrained model state.
```

Fresh initialization is mandatory.

---

# 34. Matched initialization opportunity

Khác Phase 23/24, YS0 và YS1 có:

```text
same input dimension
same architecture
same parameter shapes
same seed
```

Do đó nếu run-construction order và RNG policy được giữ đúng, có thể thực hiện **recommended matched-initialization audit**:

```text
initial_model_state_fingerprint_YS0
==
initial_model_state_fingerprint_YS1_reference
```

nếu Phase 24 source stored initial fingerprint.

Nếu upstream không lưu initial fingerprint:

```text
do not fabricate or infer equality.
```

Mark:

```text
NOT_VERIFIABLE
```

rather than fail S3.

---

# 35. Same DataLoader shuffle policy

Both conditions use:

```text
seed 42
same target IDs
same batch size
same DataLoader contract
```

If stored epoch-order fingerprints are available, recommended verify first-epoch sample order matches.

Not mandatory if upstream does not persist this detail.

---

# 36. Experiment identity

```text
sweep_id = S3_TARGET_SCALING
experiment_family = TRANSFORMER_SWEEP_S3_TARGET_SCALING
sweep_version = SWEEP_S3_TARGETSCALING-v1
```

Recommended run labels:

```text
S3_<FV*>__YS0__L144__B64__S42

S3_<FV*>__YS1__REFERENCE_S2
```

Registry run ID remains authoritative.

---

# 37. Test firewall

Hard:

```text
Test DataLoader not iterated
Test y unavailable
Test metrics unavailable
Test predictions unavailable
```

---

# 38. Training-loss comparability warning

This is one of the most important S3 rules.

YS0 train MSE is measured in:

```text
Wh²
```

YS1 train MSE is measured in:

```text
standardized-target squared units
```

Therefore:

```text
train_loss_YS0
```

and:

```text
train_loss_YS1
```

are **not directly numerically comparable**.

Same for:

```text
validation_loss_model_space.
```

---

# 39. Do not overlay raw model-space losses on one y-axis

Core S3 figures must not present YS0 vs YS1 raw training loss as if same units.

Recommended:

```text
separate panels/figures
```

with explicit unit labels.

Primary cross-condition learning curve must use:

```text
Validation RMSE Wh.
```

---

# 40. Gradient comparability nuance

Gradient norms are directly affected by target/loss scaling.

Thus:

```text
gradient norm differences are expected experimental outcomes
```

not fairness violations.

They can be reported diagnostically as part of target-scaling effect.

Do not conclude:

```text
larger gradient = worse model
```

without context.

---

# 41. Clipping comparability nuance

YS0 may trigger clipping more or less often because raw-target MSE generates different gradient scales.

This is valuable S3 diagnostic evidence.

But clipping configuration stays:

```text
max_norm = 1.0
```

for both conditions.

---

# 42. Optimization stability metrics

Record per condition:

```text
nonfinite events
max gradient norm
mean gradient norm
fraction batches clipped
best epoch
last epoch
early stopping status
Validation RMSE trajectory
```

---

# 43. Primary selection metric

Hard:

```text
best_validation_rmse_wh
```

from verified best checkpoint.

---

# 44. Secondary metrics

```text
best_validation_mae_wh
best_validation_r2
best_epoch
epochs_completed
stop_reason
gradient diagnostics
clipping fraction
runtime
```

No model-space loss in winner ranking.

---

# 45. Winner rule

\[
winner
=
\arg\min(RMSE_{YS0},RMSE_{YS1})
\]

on full-precision original-Wh Validation metrics.

---

# 46. Exact RMSE tie rule

If exact full-precision equality:

```text
prefer YS0
```

because:

```text
simpler target pipeline
no fitted target scaler
no inverse transform requirement
fewer persisted transformation artifacts
```

Tie must be explicitly recorded.

---

# 47. No arbitrary minimum gain threshold

Do not invent:

```text
YS1 must improve 1%
YS1 must improve 2 Wh
```

Strict lower RMSE wins unless exact tie.

---

# 48. Effect formulas

Let:

```text
RMSE0 = YS0
RMSE1 = YS1
```

Define YS0→YS1 effect:

\[
\Delta RMSE
=
RMSE_0-RMSE_1
\]

Positive:

```text
YS1 improves.
```

Relative:

\[
Improvement\%
=
100
\times
\frac{RMSE_0-RMSE_1}{RMSE_0}
\]

Also:

\[
\Delta MAE=MAE_0-MAE_1
\]

\[
\Delta R^2=R^2_1-R^2_0
\]

---

# 49. Metric ranking divergence

If:

```text
YS1 best RMSE
YS0 best MAE
```

record:

```text
METRIC_RANKING_DIVERGENCE.
```

Winner still RMSE-based.

---

# 50. No statistical significance testing

Only one seed per condition.

Do not use:

```text
epochs as independent observations
batch losses as independent model runs
```

for p-values.

---

# 51. No confidence intervals from epoch history

No.

---

# 52. YS0 unit semantics

For YS0:

```text
criterion = MSE
model-space target unit = Wh
loss unit = Wh²
prediction output before metric = Wh
```

---

# 53. YS1 unit semantics

For YS1:

```text
criterion = MSE
model-space target unit = standardized unit
loss unit = standardized-unit²
prediction before inverse = standardized unit
prediction after inverse = Wh
```

---

# 54. Shared metric-space contract

After conversion:

```text
YS0 prediction_wh
YS1 prediction_wh
```

must both enter exact same:

```text
METRICS-v1.
```

---

# 55. YS1 inverse-transform hard guard

Before metric calculation:

```text
assert target_scaling_id
```

and if:

```text
YS1
```

must perform frozen inverse transform exactly once.

---

# 56. Double inverse-transform guard

Never:

```text
inverse_transform twice.
```

Add artifact/test for this.

---

# 57. Missing inverse-transform guard

Never calculate RMSE Wh from standardized prediction directly.

Hard failure.

---

# 58. YS0 no-transform guard

Do not accidentally inverse-transform YS0 predictions with YS1 scaler.

Hard failure.

---

# 59. Prediction bundle consistency

Both conditions should output canonical:

```text
sample_idx
y_true_wh
y_pred_wh
residual_wh
absolute_error_wh
squared_error_wh
```

Only internal model-space metadata differs.

---

# 60. Optional prediction metadata

Recommended add:

```text
target_scaling_id
y_pred_model_space
y_true_model_space
```

to run-level diagnostic artifact, but official cross-condition prediction comparison should use Wh columns.

---

# 61. Best-checkpoint verification for new YS0

After training:

```text
fresh Transformer
↓
strict-load BEST
↓
full Validation
↓
prediction already Wh
↓
METRICS-v1
↓
verify best metrics.
```

---

# 62. YS1 reference verification

No retrain.

Verify source:

```text
best checkpoint verified
inverse transform protocol verified
same sample IDs
same metric version
exact S3 config match.
```

---

# 63. Same parameter count audit

Unlike feature sweeps, YS0/YS1 should have exact same model parameter count.

Hard expected:

```text
total_parameters_YS0
==
total_parameters_YS1
```

and:

```text
trainable_parameters_YS0
==
trainable_parameters_YS1.
```

Mismatch means model config drift.

---

# 64. Same architecture fingerprint

Recommended:

```text
architecture_fingerprint same
```

Target transform should not affect model structure.

---

# 65. Same feature fingerprint

Hard:

```text
feature_fingerprint same.
```

---

# 66. Same X scaler checksum

Hard:

```text
x_scaler_checksum same.
```

Target scaling must not alter X scaler.

---

# 67. Y scaler difference is expected

Allowed:

```text
YS0 identity transform
YS1 StandardScaler
```

This is the swept factor.

---

# 68. Config-delta audit

Expected semantic differences:

```text
target_scaling_id
target_transform_id
target_transform_type
target_scaler_id/checksum
target_model_space_unit
inverse_transform_required
```

Potential runtime-derived differences:

```text
training losses
gradient norms
best epoch
runtime
metrics.
```

Everything else must match.

---

# 69. Forbidden config differences

No differences in:

```text
feature variant
feature order
X scaler
lookback
horizon
boundary
population
batch
seed
Transformer architecture
optimizer
LR
WD
loss
epoch cap
patience
clipping
scheduler
Training Engine
Metric version.
```

---

# 70. Run matrix

Create:

```text
s3_run_matrix.csv
```

Fields:

```text
sweep_id
feature_variant_id
target_scaling_id
source_type
source_run_id
requires_new_training
feature_fingerprint
x_scaler_id
target_transform_id
target_scaler_id
inverse_transform_required
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

# 71. Sweep manifest

Create:

```text
s3_target_scaling_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S3_TARGETSCALING-v1
sweep_id = S3_TARGET_SCALING
source_s2_winner_run_id
feature_variant_id
candidate_target_scaling_ids = [YS0, YS1]
new_runs_required
reused_runs
swept_field = target_scaling_id
frozen_fields
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = YS0_ON_EXACT_RMSE_TIE
population_version
population_fingerprint
metric_version
training_engine_version
seed = 42
inherited_warnings
test_access = forbidden
status
created_at
```

---

# 72. Sweep contract

Create:

```text
s3_target_scaling_sweep_contract.json
```

Must state:

```text
S2-selected feature variant fixed.
X scaling fixed.
Only target transform changes.
YS0 = identity in Wh.
YS1 = frozen Train-only StandardScaler.
Same raw targets.
Same target IDs.
Same architecture.
Same optimizer/LR.
MSE fixed.
Metrics always computed in Wh.
Model-space losses not cross-condition comparable.
Validation RMSE Wh selects winner.
Exact tie → YS0.
Test forbidden.
```

---

# 73. Preflight audit

Create:

```text
s3_target_scaling_preflight_audit.csv
```

Checks:

```text
phase24_pass
approved_for_phase25
s2_winner_loaded
feature_variant_locked
YS0_registered
YS1_registered
YS1_reference_exact_match
population_fixed
X_scaler_fixed
Training_Engine_fixed
metric_fixed
seed_fixed
test_locked
status
```

---

# 74. Target-transform definition audit

Create:

```text
s3_target_transform_definition_audit.csv
```

Rows:

```text
YS0
YS1
```

Fields:

```text
target_scaling_id
transform_type
fit_required
fit_scope
inverse_required
model_target_unit
metric_target_unit
scaler_id
checksum
train_only_verified
status
```

---

# 75. YS1 fit-statistics audit

Create:

```text
s3_ys1_scaler_audit.json
```

Record:

```text
scaler_id
scaler_checksum
fit_split = TRAIN
fit_scope_description
mean
scale/std
n_fit_values
source_scaling_version
validation_used_in_fit = false
test_used_in_fit = false
status
```

Actual numeric values are runtime artifacts, not plan assumptions.

---

# 76. Target transform round-trip audit

For representative Train/Validation raw y values:

```text
raw
→ YS1 transform
→ inverse
```

must recover original:

\[
inverse(transform(y))
\approx y
\]

within numeric tolerance.

---

# 77. YS0 identity audit

For probe values:

```text
transform_YS0(y) == y
inverse_YS0(y) == y
```

exact or within dtype semantics.

---

# 78. Common-X audit

Create:

```text
s3_common_x_audit.csv
```

Fields:

```text
sample_probe_id
feature_variant
x_shape_ys0
x_shape_ys1
x_equal
feature_fingerprint_equal
x_scaler_checksum_equal
status
```

---

# 79. Common-target audit

Create:

```text
s3_target_identity_audit.csv
```

Fields:

```text
sample_idx
y_raw_wh_ys0
y_raw_wh_ys1
raw_equal
y_model_ys0
y_model_ys1
ys1_reconstructed_wh
roundtrip_error
status
```

Use selected probes or complete vectorized audit if lightweight.

---

# 80. Population audit

Create:

```text
s3_target_scaling_population_audit.csv
```

Fields:

```text
condition
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

# 81. Architecture audit

Create:

```text
s3_target_scaling_architecture_audit.csv
```

Fields:

```text
condition
input_size
d_model
num_heads
num_layers
ffn_dim
dropout
activation
pooling
trainable_parameters
architecture_fingerprint
matches_reference
status
```

Expected:

```text
parameter count equal.
```

---

# 82. Training-config audit

Create:

```text
s3_target_scaling_training_audit.csv
```

Fields:

```text
condition
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
matches_reference_except_target_transform
status
```

---

# 83. Initial-state audit

Create:

```text
s3_initialization_audit.csv
```

Fields:

```text
condition
initial_model_state_fingerprint
reference_available
fingerprint_matches
seed
architecture_match
status
```

If YS1 reference lacks initial fingerprint:

```text
status = NOT_VERIFIABLE
```

not FAIL.

---

# 84. Run provenance table

Create:

```text
s3_target_scaling_run_provenance.csv
```

Fields:

```text
target_scaling_id
run_id
source_type
source_phase
config_fingerprint
feature_fingerprint
x_scaler_checksum
target_transform_id
target_scaler_checksum
population_fingerprint
best_checkpoint_sha256
history_sha256
metric_artifact
prediction_artifact
status
```

---

# 85. Primary S3 metrics table

Create:

```text
s3_target_scaling_metrics.csv
```

Rows:

```text
YS0
YS1
```

Fields:

```text
feature_variant_id
target_scaling_id
run_id
source_type
target_model_space_unit
inverse_transform_required
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

Do not place raw train loss in this primary cross-condition performance table.

---

# 86. Target-scaling effect table

Create:

```text
s3_target_scaling_effect.csv
```

Fields:

```text
ys0_run_id
ys1_run_id
ys0_rmse_wh
ys1_rmse_wh
rmse_delta_ys0_to_ys1_wh
rmse_improvement_pct
ys0_mae_wh
ys1_mae_wh
mae_delta_wh
ys0_r2
ys1_r2
r2_delta
rmse_winner
metric_ranking_divergence
status
```

---

# 87. Optimization diagnostics table

Create:

```text
s3_optimization_diagnostics.csv
```

Fields:

```text
condition
target_model_space
best_epoch
last_epoch
stop_reason
global_max_grad_norm_preclip
mean_grad_norm_preclip
mean_fraction_batches_clipped
max_fraction_batches_clipped
epochs_with_any_clipping
nonfinite_events
mean_epoch_duration_seconds
status
```

Interpretation must note gradient/loss-scale dependence.

---

# 88. Loss-semantics table

Create:

```text
s3_loss_semantics.csv
```

Rows YS0/YS1:

```text
condition
criterion
target_model_space
loss_unit
directly_comparable_across_conditions
reason
```

Expected:

```text
directly_comparable_across_conditions = false
```

for model-space MSE.

---

# 89. Hypothesis outcomes

Create:

```text
s3_hypothesis_outcomes.csv
```

Fields:

```text
hypothesis_id
source_hypothesis_id_optional
comparison
expected_direction
ys0_rmse
ys1_rmse
observed_delta
outcome
interpretation
status
```

Outcome:

```text
SUPPORTED
NOT_SUPPORTED
INCONCLUSIVE_TIE
```

single-seed Validation context only.

---

# 90. Findings artifact

Create:

```text
s3_target_scaling_findings.csv
```

Possible codes:

```text
TARGET_SCALING_GAIN
TARGET_SCALING_NO_GAIN
TARGET_SCALING_EXACT_TIE
METRIC_RANKING_DIVERGENCE
GRADIENT_SCALE_DIFFERENCE
CLIPPING_BURDEN_DIFFERENCE
CONVERGENCE_DIFFERENCE
NONFINITE_YS0
NONFINITE_YS1
RUNTIME_DIFFERENCE
MATCHED_INITIALIZATION_VERIFIED
MATCHED_INITIALIZATION_NOT_VERIFIABLE
INHERITED_RANDOM_CONTROL_WARNING
```

---

# 91. Winner artifact

Create:

```text
s3_target_scaling_winner.json
```

Minimum:

```text
sweep_id
sweep_version
feature_variant_id
selection_metric
selection_direction
tie_rule
winner_target_scaling_id
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_target_scaling_id
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 92. Reference update for Phase 26

Create:

```text
s3_reference_update.json
```

Minimum:

```text
previous_reference_run_id
feature_variant_id
previous_target_scaling_id = YS1
selected_target_scaling_id
winner_run_id
winner_config_fingerprint
winner_feature_fingerprint
winner_rmse_wh
lookback_state = L144
selection_metric
inherited_warnings
approved_for_phase26
```

---

# 93. Phase 26 handoff logic

Phase 26 tests:

```text
L36
L72
L144
```

while holding:

```text
S3-selected feature variant
S3-selected target scaling
```

fixed.

The S3 winner already has:

```text
L144
```

Therefore Phase 26 should normally:

```text
reuse S3 winner as L144 reference
train only L36 and L72
```

if exact-match.

---

# 94. Common population requirement for Phase 26 reminder

Phase 10 already defined:

```text
WINDOWPOP-v1
=
common target intersection across L36/L72/L144
```

Phase 26 must preserve it.

Phase 25 handoff should include:

```text
population_fingerprint
```

for this reason.

---

# 95. Learning curves in S3

Recommended cross-condition figures:

```text
S3_01_validation_rmse_by_epoch.png
S3_02_validation_mae_by_epoch.png
S3_03_gradient_norm_by_epoch.png
S3_04_gradient_clipping_fraction.png
S3_05_best_validation_metrics.png
```

---

# 96. Training-loss figures

Because YS0/YS1 model-space loss units differ, use separate:

```text
S3_06_ys0_train_validation_loss.png
S3_07_ys1_train_validation_loss.png
```

Do not overlay raw losses on one numerical axis.

---

# 97. Primary figure

```text
S3_01_validation_rmse_by_epoch.png
```

This is valid because both trajectories are:

```text
Wh
```

after appropriate prediction transformation.

Mark best epochs.

---

# 98. Gradient figure

`S3_03_gradient_norm_by_epoch.png` may overlay YS0/YS1, but caption must state:

```text
gradient magnitude is expected to depend directly on target/loss scaling.
```

Use diagnostically, not as normalized quality metric.

---

# 99. Best metric figure

`S3_05_best_validation_metrics.png` should be generated from verified S3 metrics table.

---

# 100. No smoothing for selection

If smoothing is used visually:

```text
raw curve source of truth
winner from raw verified BEST metrics.
```

---

# 101. Interpretation if YS1 wins

Safe:

> Under the frozen optimizer and Transformer configuration, standardizing the target with Train-only statistics produced lower Validation RMSE in original Wh.

Potential supporting diagnostics:

```text
more stable RMSE trajectory
lower clipping burden
different gradient scale
earlier/later convergence
```

But do not claim one mechanism is proven.

---

# 102. Interpretation if YS0 wins

Safe:

> Under the current frozen optimizer settings, raw-Wh target training achieved lower Validation RMSE than standardized-target training.

Do not claim:

```text
target scaling is generally harmful.
```

---

# 103. Interpretation if exact tie

Choose:

```text
YS0
```

by parsimony.

Report:

```text
No full-precision Validation RMSE benefit for target standardization was observed.
```

---

# 104. Tiny non-zero margin

Winner rule still applies, but write:

```text
small single-seed Validation margin
```

and avoid strong generalization.

---

# 105. Single-seed limitation

Mandatory:

```text
S3 uses seed 42 conditions only.
```

No mean±std.

---

# 106. Validation-only limitation

Mandatory:

```text
S3 winner is a development selection on Validation.
```

No Test evidence.

---

# 107. Sequential-selection limitation

Validation has now influenced:

```text
S1
S2
S3
```

Therefore project must maintain:

```text
complete Registry
no hidden experiments
rolling-origin robustness
final multi-seed runs
untouched Test.
```

---

# 108. No target transform alternatives

Do not add:

```text
MinMax target scaling
RobustScaler
log1p target
Box-Cox
PowerTransformer
quantile transform
```

inside S3.

Current S3 is exactly:

```text
YS0 vs YS1.
```

---

# 109. No output activation change

Do not add:

```text
ReLU
Softplus
clamping
```

to keep predictions nonnegative.

That would be another factor.

---

# 110. No target clipping

No.

---

# 111. No outlier trimming

No.

---

# 112. No target log transform

No.

---

# 113. No loss change

MSE fixed.

Huber belongs Phase 37.

---

# 114. No LR tuning specifically for YS0

No.

Phase 30 later tests LR under whatever current reference survives prior sweeps.

---

# 115. No gradient clipping tuning

Clip remains 1.0.

Phase 39 later.

---

# 116. No batch change

B64 fixed.

Phase 29 later.

---

# 117. No epoch-cap extension

E50 fixed.

Phase 38 later.

---

# 118. No attention analysis

Not Phase 25.

---

# 119. No residual regime analysis

Not Phase 25.

---

# 120. No Test access

Hard.

---

# 121. Run failure policy

If YS0 run fails technically:

```text
S3 incomplete.
```

Do not automatically select YS1.

---

# 122. YS0 numerical instability

If raw target scale causes:

```text
NaN/Inf
```

under frozen training config:

```text
record scientific/engineering outcome
run = FAILED
S3 cannot complete selection until protocol decision.
```

Do not silently reduce LR.

Because a semantic config change would move beyond the pre-registered S3 condition.

---

# 123. How to handle true YS0 failure

Recommended process:

```text
1. Verify implementation is correct.

2. Verify no data/scaler bug.

3. If failure is genuinely due to frozen optimization scale:
   record YS0 incompatibility under current protocol.

4. Do not manufacture a successful YS0 by changing LR inside S3.

5. If project requires a comparable stable YS0 condition,
   create formal Protocol Amendment defining how to handle it.
```

This keeps experimental history honest.

---

# 124. Score-based rerun forbidden

Do not rerun YS0 because metric is poor.

Technical rerun only for:

```text
interrupt
corrupt artifact
hardware/software fault
```

with explicit provenance.

---

# 125. OOM policy

YS0/YS1 architecture is same.

If YS0 OOM unexpectedly while YS1 did not:

```text
investigate infrastructure anomaly.
```

Target scaling itself should not materially change model memory structure.

Do not silently switch B32.

---

# 126. Discrepancy taxonomy

```text
S2_REFERENCE_MISSING
S2_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
X_SCALER_MISMATCH
YS0_DEFINITION_MISMATCH
YS1_DEFINITION_MISMATCH
YS1_SCALER_REFIT
YS1_SCALER_LEAKAGE
TARGET_ID_MISMATCH
RAW_TARGET_MISMATCH
TARGET_ROUNDTRIP_FAILURE
POPULATION_MISMATCH
ARCHITECTURE_DRIFT
PARAMETER_COUNT_MISMATCH
TRAINING_CONFIG_DRIFT
SEED_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
REFERENCE_RUN_MISMATCH
MISSING_INVERSE_TRANSFORM
DOUBLE_INVERSE_TRANSFORM
WRONG_YS0_TRANSFORM
RUN_FAILURE
CHECKPOINT_VERIFICATION_FAILURE
INCOMPLETE_SWEEP
RANKING_ERROR
EFFECT_CALCULATION_ERROR
LOSS_COMPARABILITY_MISUSE
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 127. Discrepancy log

Create:

```text
s3_target_scaling_discrepancies.json
```

Fields:

```text
id
severity
category
condition_or_pair
expected
actual
impact
recommended_action
resolved
resolution_notes
```

---

# 128. Severity examples

```text
CRITICAL:
Test access
YS1 leakage
population mismatch
missing inverse transform
architecture drift

MAJOR:
wrong target transform
checkpoint verification failure
X scaler mismatch

MODERATE:
metric ranking divergence
strong clipping burden difference
inherited random-control warning

INFO:
runtime difference
matched initialization not verifiable
```

---

# 129. Status model

## PASS

```text
YS0 and YS1 valid
same X/population/model/training config
target transforms correct
YS1 reference valid
new YS0 verified
metrics in Wh
winner selected
Phase 26 reference generated
Test untouched
```

## PASS_WITH_WARNING

Possible:

```text
tiny margin
metric ranking divergence
large gradient/clipping differences
initialization match not verifiable
inherited FS2 warning
```

with methodology valid.

## FAIL

Examples:

```text
YS1 scaler leakage
population mismatch
raw targets differ
inverse transform wrong
architecture/config drift
unresolved YS0 failure
Test access.
```

---

# 130. Sweep summary

Create:

```text
s3_target_scaling_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
feature_variant_id
reference_run_id
ys0_run_id
ys1_run_id
new_runs
reused_runs
primary_metric
ys0_metrics
ys1_metrics
target_scaling_effect
optimization_diagnostics
winner
winner_margin
metric_ranking_divergence
inherited_warnings
phase26_reference
test_status
overall_status
```

---

# 131. Human-readable report

Create:

```text
s3_target_scaling_sweep_report.md
```

Sections:

```text
1. Objective
2. S2-selected feature variant
3. Controlled variable
4. YS0/YS1 definitions
5. Frozen configuration
6. Target-scaler provenance
7. Same-X / same-target fairness
8. Model-space loss semantics
9. Validation performance
10. Gradient/clipping diagnostics
11. Learning-curve context
12. S3 winner
13. Limitations
14. Phase 26 handoff
```

---

# 132. Report writing structure

For each result:

```text
Observed
Interpretation
Limitation
Handoff
```

---

# 133. Critical report caveat

Must explicitly state:

> Model-space MSE values for YS0 and YS1 are on different scales and therefore are not directly comparable as performance metrics.

---

# 134. README

Create:

```text
README_S3_TARGET_SCALING_SWEEP.md
```

Must explain:

```text
Purpose
S2 winner handoff
YS0 identity semantics
YS1 Train-only StandardScaler semantics
same-X/same-target requirement
why model-space losses cannot be compared
why gradient differences are expected
YS1 inverse-transform requirement
reference reuse
winner rule
tie rule
limitations
Phase 26 handoff
No Test
```

---

# 135. Output directory

```text
artifacts/
└── sweeps/
    └── S3_target_scaling/
        ├── s3_target_scaling_sweep_manifest.json
        ├── s3_target_scaling_sweep_contract.json
        ├── s3_target_scaling_preflight_audit.csv
        ├── s3_run_matrix.csv
        ├── s3_target_transform_definition_audit.csv
        ├── s3_ys1_scaler_audit.json
        ├── s3_common_x_audit.csv
        ├── s3_target_identity_audit.csv
        ├── s3_target_scaling_population_audit.csv
        ├── s3_target_scaling_architecture_audit.csv
        ├── s3_target_scaling_training_audit.csv
        ├── s3_initialization_audit.csv
        ├── s3_target_scaling_run_provenance.csv
        ├── s3_target_scaling_metrics.csv
        ├── s3_target_scaling_effect.csv
        ├── s3_optimization_diagnostics.csv
        ├── s3_loss_semantics.csv
        ├── s3_hypothesis_outcomes.csv
        ├── s3_target_scaling_findings.csv
        ├── s3_target_scaling_winner.json
        ├── s3_reference_update.json
        ├── s3_target_scaling_sweep_tests.csv
        ├── s3_target_scaling_discrepancies.json
        ├── s3_target_scaling_sweep_summary.json
        ├── s3_target_scaling_sweep_report.md
        ├── figures/
        │   ├── S3_01_validation_rmse_by_epoch.png
        │   ├── S3_02_validation_mae_by_epoch.png
        │   ├── S3_03_gradient_norm_by_epoch.png
        │   ├── S3_04_gradient_clipping_fraction.png
        │   ├── S3_05_best_validation_metrics.png
        │   ├── S3_06_ys0_train_validation_loss.png
        │   └── S3_07_ys1_train_validation_loss.png
        ├── README_S3_TARGET_SCALING_SWEEP.md
        └── phase_25_signoff.json
```

New YS0 run remains under:

```text
artifacts/runs/<run_id>/
```

Do not duplicate checkpoint into sweep folder.

---

# 136. Required outputs

```text
O25.1  Sweep manifest
O25.2  Sweep contract
O25.3  Preflight audit
O25.4  Run matrix
O25.5  Target-transform definition audit
O25.6  YS1 scaler provenance audit
O25.7  Common-X audit
O25.8  Raw-target identity/roundtrip audit
O25.9  Population audit
O25.10 Architecture audit
O25.11 Training-config audit
O25.12 Initialization audit
O25.13 Run provenance
O25.14 Reused YS1 reference
O25.15 Verified new YS0 run
O25.16 Metrics table
O25.17 Target-scaling effect table
O25.18 Optimization diagnostics
O25.19 Loss-semantics table
O25.20 Hypothesis outcomes
O25.21 Findings
O25.22 Winner artifact
O25.23 Phase 26 reference update
O25.24 Diagnostic figures
O25.25 Sweep test suite
O25.26 Discrepancy log
O25.27 Sweep summary
O25.28 Human-readable report
O25.29 README
O25.30 Phase sign-off
```

---

# 137. Sweep test suite

Create:

```text
s3_target_scaling_sweep_tests.csv
```

Recommended checks:

```text
S3T25-001 Phase 24 PASS/non-critical warning only
S3T25-002 approved_for_phase25 = true
S3T25-003 S2 winner artifact valid
S3T25-004 feature variant locked
S3T25-005 YS0 registered
S3T25-006 YS1 registered
S3T25-007 YS0 = identity transform
S3T25-008 YS0 requires no fitted statistics
S3T25-009 YS1 = StandardScaler
S3T25-010 YS1 scaler checksum valid
S3T25-011 YS1 fitted Train-only
S3T25-012 no Validation used in YS1 fit
S3T25-013 no Test used in YS1 fit
S3T25-014 YS1 transform/inverse roundtrip valid
S3T25-015 YS0 identity roundtrip valid
S3T25-016 same raw target vector
S3T25-017 same Train target IDs
S3T25-018 same Validation target IDs
S3T25-019 same population fingerprint
S3T25-020 same feature fingerprint
S3T25-021 same feature order
S3T25-022 same X scaler checksum
S3T25-023 same lookback L144
S3T25-024 same horizon H1
S3T25-025 same WB0
S3T25-026 same B64
S3T25-027 same D64
S3T25-028 same H4
S3T25-029 same N2
S3T25-030 same FFN128
S3T25-031 same dropout .1
S3T25-032 same GELU
S3T25-033 same LAST_STEP
S3T25-034 same model parameter count
S3T25-035 same architecture fingerprint
S3T25-036 same AdamW
S3T25-037 same LR 3e-4
S3T25-038 same WD 1e-4
S3T25-039 same MSE
S3T25-040 same epoch cap 50
S3T25-041 same patience 10
S3T25-042 same clipping 1
S3T25-043 same seed 42
S3T25-044 same Training Engine
S3T25-045 same Metric version
S3T25-046 YS1 reference exact-match
S3T25-047 YS1 reference reused, not retrained
S3T25-048 YS0 new run registered before training
S3T25-049 YS0 fresh loaders/model
S3T25-050 YS0 no warm-start
S3T25-051 YS0 trained via TRAINING_ENGINE-v1
S3T25-052 YS0 best checkpoint verified
S3T25-053 YS1 reference best checkpoint verified
S3T25-054 YS0 metrics calculated directly in Wh
S3T25-055 YS1 prediction inverse-transformed once
S3T25-056 no double inverse-transform
S3T25-057 no missing inverse-transform
S3T25-058 both result rows use verified BEST
S3T25-059 full-precision RMSE available
S3T25-060 RMSE effect correct
S3T25-061 relative improvement correct
S3T25-062 MAE effect correct
S3T25-063 R² effect correct
S3T25-064 winner = min RMSE
S3T25-065 exact tie → YS0
S3T25-066 model-space loss marked non-comparable
S3T25-067 no raw loss used for winner
S3T25-068 gradient diagnostics labeled scale-dependent
S3T25-069 metric divergence recorded if present
S3T25-070 hypothesis outcome generated
S3T25-071 winner points to valid run
S3T25-072 reference update generated
S3T25-073 Phase 26 preserves S3 winner
S3T25-074 L144 reference reuse identified for Phase 26
S3T25-075 inherited warnings propagated
S3T25-076 no extra target transform
S3T25-077 no LR compensation
S3T25-078 no warm-start
S3T25-079 no score-based rerun
S3T25-080 no failed/SANITY run in ranking
S3T25-081 no Test access
S3T25-082 single-seed limitation documented
S3T25-083 Validation-only limitation documented
S3T25-084 sequential-selection limitation documented
S3T25-085 figures source-derived
S3T25-086 summary/report generated
S3T25-087 phase sign-off generated
```

---

# 138. Recommended notebook structure

```text
Cell 25.1  Phase title
Cell 25.2  Verify Phase 24 sign-off
Cell 25.3  Declare SWEEP_S3_TARGETSCALING-v1
Cell 25.4  Load S2 winner/reference update
Cell 25.5  Resolve fixed feature variant
Cell 25.6  Resolve YS0/YS1 definitions
Cell 25.7  Build S3 run matrix
Cell 25.8  Audit YS0 identity
Cell 25.9  Audit frozen YS1 scaler provenance
Cell 25.10 Audit transform/inverse roundtrip
Cell 25.11 Audit same X and X scaler
Cell 25.12 Audit same raw targets
Cell 25.13 Audit population identity
Cell 25.14 Audit architecture equality
Cell 25.15 Audit training-config equality
Cell 25.16 Check optional initial-state fingerprint match
Cell 25.17 Verify YS1 reference reuse eligibility
Cell 25.18 Register YS0 run
Cell 25.19 Seed + fresh YS0 loaders/model
Cell 25.20 Execute YS0 via Training Engine
Cell 25.21 Verify YS0 best checkpoint
Cell 25.22 Build run provenance
Cell 25.23 Build Wh-space S3 metrics table
Cell 25.24 Compute target-scaling effect
Cell 25.25 Build optimization diagnostics
Cell 25.26 Build loss-semantics table
Cell 25.27 Evaluate hypotheses
Cell 25.28 Generate Wh-space learning curves
Cell 25.29 Generate separate YS0/YS1 loss figures
Cell 25.30 Generate findings
Cell 25.31 Select S3 winner
Cell 25.32 Write winner JSON
Cell 25.33 Write Phase 26 reference update
Cell 25.34 Run S3 tests/discrepancies
Cell 25.35 Write summary/report
Cell 25.36 Register artifacts/checksums
Cell 25.37 Write README
Cell 25.38 Phase sign-off
```

---

# 139. Execution flow

```text
Verify Phase 24
        ↓
Load S2 winner
        ↓
Freeze feature variant
        ↓
Resolve YS0 and YS1
        ↓
Audit YS0 identity
        ↓
Audit YS1 frozen Train-only scaler
        ↓
Audit same X / same raw y / same target IDs
        ↓
Audit identical architecture/training config
        ↓
Verify YS1 reference exact match
        ↓
Reuse YS1
        ↓
Register fresh YS0 run
        ↓
Seed 42
        ↓
Fresh loaders/model
        ↓
Train in raw Wh target space
        ↓
Verify YS0 BEST
        ↓
Convert both conditions to common Wh metric space
        ↓
Build S3 metrics/effects
        ↓
Inspect optimization diagnostics
        ↓
Select min-RMSE winner
        ↓
Apply YS0 exact-tie rule
        ↓
Update Phase 26 reference
        ↓
Write S3 artifacts
        ↓
SWEEP_S3_TARGETSCALING-v1 sign-off
```

---

# 140. Fail-fast order

Before expensive YS0 training:

```text
1. Phase 24 sign-off
2. S2 winner identity
3. fixed feature variant
4. same X scaler
5. YS0 identity definition
6. YS1 scaler provenance
7. YS1 no-leakage audit
8. transform roundtrip
9. same raw targets/IDs
10. population identity
11. model architecture equality
12. training config equality
13. YS1 reference reuse eligibility
14. Test firewall
15. Registry readiness
```

---

# 141. Why target-transform audit must happen before training

Common silent errors:

```text
YS1 scaler refitted
Validation included in scaler fit
YS0 accidentally uses standardized target
YS1 inverse transform omitted
wrong scaler from another experiment
```

Any one invalidates S3.

---

# 142. Why model-space loss cannot select winner

Suppose:

```text
YS0 Validation MSE = 4000 Wh²
YS1 Validation MSE = 0.2 standardized²
```

Those values are not directly comparable because units/scales differ.

Only after predictions are represented in Wh can performance be fairly compared.

---

# 143. Why same LR is intentional

Changing target scale changes gradients, but keeping:

```text
LR=3e-4
```

asks the controlled question:

> Does target scaling improve this exact baseline optimization pipeline?

LR adaptation is a separate later sweep.

---

# 144. Why same clipping threshold is intentional

Same logic:

```text
clip=1.0
```

allows observing whether target scaling changes clipping burden under the baseline training engine.

---

# 145. Winner verification checklist

Before writing `s3_target_scaling_winner.json`:

```text
[ ] YS0 valid completed run.
[ ] YS1 valid reused reference.
[ ] Same feature variant.
[ ] Same X scaler.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same architecture/parameter count.
[ ] Same Training Engine.
[ ] Same Metric version.
[ ] YS1 scaler is Train-only frozen.
[ ] YS1 inverse transform verified.
[ ] YS0 identity verified.
[ ] Both BEST checkpoints verified.
[ ] Wh-space RMSE values full precision.
[ ] No model-space loss used for ranking.
[ ] Effect calculated programmatically.
[ ] Exact tie rule respected.
[ ] No Test.
```

---

# 146. Phase 26 handoff

Phase 26 receives:

```text
s3_target_scaling_winner.json
s3_reference_update.json
winner_run_id
winner_config_fingerprint
feature_variant_id
selected_target_scaling_id
population_fingerprint
```

and tests only:

```text
lookback L36/L72/L144.
```

---

# 147. If YS0 wins

Current reference:

```text
FV*
YS0
L144
```

Phase 26:

```text
L36
L72
L144
```

all using:

```text
YS0.
```

Reuse S3 YS0 winner as L144 reference.

---

# 148. If YS1 wins

Current reference:

```text
FV*
YS1
L144
```

Phase 26 uses YS1 across all lookbacks and reuses YS1 winner as L144 reference.

---

# 149. If exact tie

YS0 selected by parsimony.

Phase 26 uses YS0.

---

# 150. Relationship with Phase 30 LR sweep

Even if YS0 performs poorly because gradients are huge, do not change LR here.

Phase 30 later evaluates LR on the sequentially selected current reference.

The S3 result remains historical evidence.

---

# 151. Relationship with Phase 37 Loss sweep

MSE remains fixed in S3.

Huber later may interact with target scale, but one-factor sequential design does not explore full interaction grid.

Document this limitation rather than expanding S3.

---

# 152. Interaction limitation

Target scaling may interact with:

```text
learning rate
loss
gradient clipping
```

S3 estimates its effect under current frozen settings only.

Do not claim:

```text
YS1 globally best across all optimizer settings.
```

---

# 153. Full experimental history preservation

Both YS0 and YS1 conditions remain in Registry.

Never delete loser.

---

# 154. Rolling-origin importance

S3 winner is still based on one Validation split.

Phase 44 later checks chronological robustness.

---

# 155. Multi-seed importance

S3 is seed 42.

Tiny margins may not persist across seeds.

---

# 156. Test relationship

None.

Test remains untouched until final evaluation.

---

# 157. Reproducibility metadata

New YS0 run records:

```text
run_id
seed
environment
device
feature fingerprint
X scaler checksum
target transform ID
population fingerprint
model config fingerprint
initial state fingerprint if available
Training Engine fingerprint
best checkpoint checksum
history checksum
metric checksum
```

YS1 keeps existing provenance.

---

# 158. No fabricated runtime outputs

Do not pre-fill:

```text
RMSE
MAE
R²
winner
gradient norms
clipping fraction
best epoch
runtime
scaler mean/std
```

until execution.

---

# 159. Phase sign-off

Create:

```text
phase_25_signoff.json
```

Minimum:

```text
phase = 25
phase_name = S3 Target-scaling sweep
sweep_version
sweep_id
source_s2_winner_run_id
feature_variant_id
ys0_run_id
ys1_reference_run_id
new_run_ids
reused_run_ids
winner_target_scaling_id
winner_run_id
winner_rmse_wh
population_fingerprint
metric_version
target_transform_audit_status
fairness_audit_status
matched_initialization_status
inherited_warnings
test_status
approved_for_phase26
overall_status
created_at
```

---

# 160. Acceptance checklist

```text
[ ] Phase 24 valid.
[ ] approved_for_phase25 = true.
[ ] S3 version declared.
[ ] S2 winner loaded.
[ ] Feature variant fixed.
[ ] X feature order fixed.
[ ] X scaler fixed.
[ ] YS0 defined as identity.
[ ] YS0 no fit statistics.
[ ] YS1 defined as frozen StandardScaler.
[ ] YS1 Train-only provenance verified.
[ ] Validation excluded from YS1 fit.
[ ] Test excluded from YS1 fit.
[ ] YS1 checksum valid.
[ ] YS1 roundtrip transform valid.
[ ] YS0 identity roundtrip valid.
[ ] Same raw target values.
[ ] Same Train target IDs.
[ ] Same Validation target IDs.
[ ] Same population fingerprint.
[ ] Same L144/H1/WB0.
[ ] Same B64.
[ ] Same D64/H4/N2/FFN128.
[ ] Same dropout/GELU/LAST_STEP.
[ ] Same parameter count.
[ ] Same architecture fingerprint.
[ ] Same AdamW/LR/WD/MSE.
[ ] Same E50/patience10/clip1.
[ ] Same seed 42.
[ ] Same Training Engine.
[ ] Same Metric version.
[ ] Initial-state match checked if verifiable.
[ ] YS1 reference exact-match.
[ ] YS1 not retrained.
[ ] YS0 run registered before training.
[ ] YS0 fresh loaders/model.
[ ] YS0 no warm-start.
[ ] YS0 trained via TRAINING_ENGINE-v1.
[ ] YS0 BEST verified.
[ ] YS1 BEST already verified.
[ ] YS0 metrics computed in Wh.
[ ] YS1 inverse transform exactly once.
[ ] No missing/double inverse transform.
[ ] Wh-space metrics available full precision.
[ ] Model-space losses marked non-comparable.
[ ] Model-space loss not used for ranking.
[ ] Gradient differences labeled scale-dependent.
[ ] RMSE effect computed.
[ ] Relative improvement computed.
[ ] MAE/R² effects computed.
[ ] Winner = min RMSE.
[ ] Exact tie → YS0.
[ ] Metric divergence recorded if present.
[ ] Hypothesis outcome recorded.
[ ] Findings generated.
[ ] Inherited warnings propagated.
[ ] Winner JSON generated.
[ ] Phase 26 reference update generated.
[ ] L144 reuse identified.
[ ] No extra target transform.
[ ] No LR compensation.
[ ] No clipping change.
[ ] No hidden rerun.
[ ] No failed/SANITY run in ranking.
[ ] No Test access.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] Interaction limitation documented.
[ ] Sequential-selection limitation documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancies recorded.
[ ] Phase sign-off generated.
```

---

# 161. Acceptance criteria

Phase 25 only PASS when:

```text
S2-selected feature variant is fixed.

YS0 and YS1 are correctly defined.

Same X, raw y, target IDs and population are used.

YS1 scaler is frozen and Train-only.

YS0 is a true identity transform.

Architecture and parameter count are identical.

Optimizer/training hyperparameters are identical.

YS1 reference is valid and reused.

YS0 is one fresh seed-42 run.

No warm-start occurs.

Both BEST checkpoints are verified.

All comparison metrics are computed in original Wh.

Model-space losses are not compared as performance metrics.

Validation RMSE Wh selects winner.

Exact RMSE tie selects YS0.

Phase 26 reference is generated.

No hidden optimization compensation occurs.

No Test data is accessed.
```

---

# 162. Failure conditions

Phase 25 FAIL if:

```text
wrong S2 winner used

feature variant changes

X scaler changes

YS1 scaler is refitted

Validation/Test enters scaler fit

raw targets differ

target IDs differ

population differs

YS0 is not identity

YS1 inverse transform omitted/doubled

model architecture differs

parameter counts differ

LR/WD/loss/clipping differs

YS1 reference mismatches but is reused

YS1 is retrained and best rerun cherry-picked

YS0 warm-starts

one condition fails but winner still declared

raw model-space MSE determines winner

RMSE rounded before ranking

Test used

score-based rerun occurs.
```

---

# 163. Common mistakes

## 163.1 So sánh train MSE YS0 và YS1 trực tiếp

Sai vì khác scale/unit.

## 163.2 Quên inverse-transform YS1 trước RMSE

Sai nghiêm trọng.

## 163.3 Inverse-transform YS0 bằng scaler YS1

Sai.

## 163.4 Fit YS1 scaler lại trên Train+Validation

Leakage.

## 163.5 Thấy YS0 gradient lớn rồi giảm LR riêng

Hai factors thay đổi.

## 163.6 Tắt clipping cho YS0

Không.

## 163.7 Warm-start YS0 từ YS1

Confounded.

## 163.8 Retrain YS1 rồi giữ run tốt hơn

Hidden rerun bias.

## 163.9 Dùng loss thấp hơn để chọn YS1

Loss scales khác nhau.

## 163.10 Dùng R² để override lower RMSE

Sai primary metric.

## 163.11 Log-transform target sau khi YS0 thua

Không thuộc S3.

## 163.12 Clamp negative prediction

Thay model behavior.

## 163.13 Dùng Test để xem scaling “generalize hơn”

Forbidden.

## 163.14 Kết luận YS1 luôn tốt hơn mọi LR/loss

Unsupported interaction claim.

## 163.15 Chạy nhiều seed rồi chọn seed đẹp

Không thuộc S3.

---

# 164. Recommended execution pseudocode

```text
load_phase24_signoff()
assert_approved_for_phase25()

s2_winner = load_s2_winner()

feature_variant = s2_winner.variant_id
x_scaler = resolve_frozen_x_scaler(feature_variant)

ys0 = resolve_target_transform("YS0")
ys1 = resolve_target_transform("YS1")

audit_ys0_identity(ys0)
audit_ys1_train_only_scaler(ys1)
audit_transform_roundtrip(ys1)

audit_same_x(feature_variant, x_scaler)
audit_same_raw_targets()
audit_same_population()
audit_same_architecture_training_config()

ys1_reference = resolve_s2_winner_run()
assert_exact_s3_reference_match(ys1_reference)

register_ys0_run()
seed(42)

loaders = build_fresh_loaders(
    feature_variant=feature_variant,
    target_scaling="YS0"
)

model = build_fresh_transformer(
    input_size=feature_count(feature_variant)
)

ys0_result = TRAINING_ENGINE_v1.fit(...)
verify_best_checkpoint_ys0(ys0_result)

results = {
    "YS0": ys0_result,
    "YS1": ys1_reference
}

metrics_wh = build_wh_space_metrics(results)
effect = compute_target_scaling_effect(metrics_wh)
optimization = build_scale_aware_optimization_diagnostics(results)

winner = select_min_rmse(
    metrics_wh,
    exact_tie_prefer="YS0"
)

write_hypothesis_outcomes()
write_findings()
write_s3_winner(winner)
write_phase26_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 165. Definition of Done

\[
\boxed{
One\ Fixed\ Input\ Configuration
+
Two\ Target\ Spaces
+
One\ Fresh\ YS0\ Run
+
One\ Valid\ Reused\ YS1
+
Same\ Population
+
Same\ Model
+
Correct\ Transform/Inverse
+
Wh\text{-}Space\ Metrics
+
S3\ Winner
+
Phase26\ Reference
+
No\ Test
}
\]

---

# 166. Final status contract

```text
PHASE 25 tests TARGET SCALING only.

Input feature variant:
comes from S2 winner.

Conditions:

YS0
→ identity target
→ raw Wh training target

YS1
→ frozen Train-only StandardScaler
→ standardized training target
→ inverse to Wh before metrics

YS1:
reuse S2 winner if exact match.

YS0:
one new fresh run.

Everything else:
frozen.

Same:
X
raw y
sample IDs
population
architecture
parameter count
B64
seed42
AdamW
LR3e-4
WD1e-4
MSE
E50
patience10
clip1
Training Engine
Metric version.

Critical semantic rule:
YS0/YS1 model-space MSE losses are NOT directly comparable.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer YS0.

No LR compensation.
No alternate target transform.
No warm-start.
No hidden rerun.
No Test.

After SWEEP_S3_TARGETSCALING-v1 PASS:
update current reference
and proceed to
PHASE 26 — S4 Lookback sweep.
```

---

# 167. Final check

Correct workflow:

```text
Load S2 winner
→ Freeze feature variant/X pipeline
→ Audit YS0 identity
→ Audit frozen Train-only YS1
→ Audit same X/raw targets/IDs
→ Reuse YS1
→ Train one fresh YS0
→ Verify BEST
→ Convert both to Wh metric space
→ Compare RMSE/MAE/R²
→ Diagnose scale-dependent gradients separately
→ Select min-RMSE target scaling
→ Update Phase 26 reference
```

Incorrect workflow:

```text
compare standardized MSE to Wh² MSE
→ change LR for YS0
→ rerun YS1
→ choose nicer curve
→ inspect Test.
```

Only after `SWEEP_S3_TARGETSCALING-v1` is signed off may the project proceed to **PHASE 26 — S4 Lookback sweep**.
