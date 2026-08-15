# PHASE 27 — S5 POOLING SWEEP

## Kế hoạch controlled sweep cho sequence pooling strategy của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S4_LOOKBACK-v1`  
**Sweep ID:** `S5_POOLING`  
**Output version:** `SWEEP_S5_POOLING-v1`  
**Phase trước:** `Phase_26_S4_Lookback_sweep.md`

---

# 1. Vai trò của Phase 27

Phase 27 là controlled experiment thứ năm trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target scaling, lookback length, target population, Transformer Encoder architecture, optimizer, training protocol và seed đã được khóa từ các phase trước, cách **pool sequence representation** nào phù hợp hơn cho sequence-to-one regression: lấy representation của timestep cuối cùng hay lấy trung bình representation của toàn bộ chuỗi?

Phase 27 chỉ thay đúng một conceptual factor:

```text
POOLING STRATEGY
```

với hai condition:

```text
P0 = LAST_STEP
P1 = MEAN
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Factor
+
Same\ Encoded\ Sequence
+
Same\ Samples
+
Same\ Architecture
+
Same\ Training
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí Phase 27 trong master execution plan

```text
Phase 23
→ S1 Feature-set sweep

Phase 24
→ S2 Time-feature sweep

Phase 25
→ S3 Target-scaling sweep

Phase 26
→ S4 Lookback sweep

Phase 27
→ S5 Pooling sweep

Phase 28
→ S6 Activation sweep
```

Phase 27 không được quay lại thay:

```text
feature set
time features
target scaling
lookback
batch size
learning rate
dropout
d_model
heads
layers
FFN
loss
RevIN
boundary protocol
```

Các yếu tố đó giữ nguyên current reference từ Phase 26.

---

# 3. Câu hỏi nghiên cứu của S5

Phase 27 phải trả lời:

```text
1. LAST_STEP hay MEAN tạo Validation RMSE thấp hơn?

2. Việc tổng hợp toàn bộ sequence representations có giúp hơn việc chỉ lấy encoded representation ở timestep cuối không?

3. Hai pooling strategies ảnh hưởng best epoch/early stopping như thế nào?

4. Gradient behavior có thay đổi đáng kể không?

5. MAE và R² có cùng ranking với RMSE không?

6. MEAN có làm training ổn định hơn hoặc kém ổn định hơn không?

7. Pooling strategy nào trở thành current reference cho Phase 28?
```

---

# 4. Context từ Phase 26

Phase 27 phải load:

```text
s4_lookback_winner.json
s4_reference_update.json
phase_26_signoff.json
```

để resolve current frozen configuration:

```text
FV* = selected feature variant
YS* = selected target scaling
L*  = selected lookback
```

Possible:

```text
L* ∈ {L36, L72, L144}
```

Phase 27 không hard-code `L144`.

---

# 5. Current sequence representation contract

Transformer Encoder output trước pooling có shape:

\[
[B,L^*,d_{model}]
\]

với current baseline architecture:

```text
d_model = 64
```

Do đó:

```text
encoded_sequence.shape
=
[B, L*, 64]
```

Pooling chuyển:

```text
[B, L*, 64]
→
[B, 64]
```

sau đó regression head:

```text
Linear(64,1)
```

cho output:

```text
[B,1]
```

---

# 6. P0 — LAST_STEP pooling

Canonical definition:

```python
pooled = encoded[:, -1, :]
```

Shape:

```text
[B,L,64]
→
[B,64]
```

Interpretation:

```text
use encoded representation corresponding to the most recent historical timestep.
```

Quan trọng:

`encoded[:, -1, :]` không chỉ chứa raw information của timestep cuối. Sau Transformer self-attention, representation này có thể đã contextualize information từ toàn bộ historical sequence.

---

# 7. P1 — MEAN pooling

Canonical definition:

```python
pooled = encoded.mean(dim=1)
```

Shape:

```text
[B,L,64]
→
[B,64]
```

Interpretation:

```text
average all contextualized timestep representations equally.
```

Mean được tính trên:

```text
Transformer Encoder outputs
```

không phải trên raw input features.

---

# 8. No padding simplifies MEAN

Current window contract:

```text
fixed-length windows
no padding
no padding mask
```

Do đó MEAN chính xác là:

```python
encoded.mean(dim=1)
```

Không cần:

```text
masked mean
length normalization by valid tokens
padding exclusion logic
```

Nếu code thêm masked pooling không cần thiết, phải đảm bảo output vẫn equivalent khi mask=None.

---

# 9. MEAN is not raw temporal averaging

Không được hiểu sai:

```text
MEAN pooling
```

là:

```text
average sensor values across time.
```

Correct semantics:

```text
raw X
→ input projection
→ positional encoding
→ self-attention/FFN encoder stack
→ contextualized sequence H
→ mean over H's sequence dimension
→ regression head.
```

---

# 10. LAST_STEP is not a recurrent hidden state

Transformer `LAST_STEP` không tương đương hoàn toàn với LSTM final hidden state.

Nó là:

```text
encoded token representation at final historical position
```

sau global self-attention over the historical window.

---

# 11. Why this sweep matters

Sequence-to-one Transformer cần một mechanism để map:

\[
[B,L,D]
\rightarrow
[B,D]
\]

before regression.

Two simple, interpretable choices:

```text
LAST_STEP
MEAN
```

represent different assumptions:

```text
LAST_STEP:
most recent contextualized state is sufficient readout.

MEAN:
information distributed across the sequence should be aggregated globally.
```

Phase 27 kiểm empirical utility dưới same Encoder.

---

# 12. Working hypotheses

## H-S5-01 — LAST_STEP may favor near-term forecasting

Vì task là:

```text
one-step-ahead
10-minute forecast
```

most recent contextualized timestep có thể là readout tự nhiên cho next-step regression.

Nếu:

\[
RMSE(LAST\_STEP)<RMSE(MEAN)
\]

hypothesis được support trong current Validation context.

Status:

```text
UNTESTED
```

---

# 13. H-S5-02 — MEAN may exploit distributed context

Nếu:

\[
RMSE(MEAN)<RMSE(LAST\_STEP)
\]

evidence support rằng averaging contextualized representations across the selected historical window improves the regression readout under current configuration.

Không được kết luận:

```text
all timesteps are equally important
```

vì Encoder đã transform/mix information before pooling.

---

# 14. Important interpretation caution

MEAN pooling assigns equal arithmetic weight to each **final hidden representation**, nhưng hidden representations themselves may encode non-uniform information through attention.

Do not say:

```text
MEAN means every original timestep contributes equally to prediction.
```

That statement is too strong.

---

# 15. Preconditions bắt buộc

Phase 27 chỉ bắt đầu khi:

```text
Phase 26 = PASS
```

hoặc `PASS_WITH_WARNING` không có unresolved critical issue.

Bắt buộc:

```text
approved_for_phase27 = true
```

và available:

```text
s4_lookback_winner.json
s4_reference_update.json
phase_26_signoff.json
```

---

# 16. Upstream contracts bắt buộc

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
SWEEP_S3_TARGETSCALING-v1
SWEEP_S4_LOOKBACK-v1
```

---

# 17. Carry-forward warnings

Nếu upstream có warning:

```text
RANDOM_CONTROL_GAIN
TARGET_SCALING_SMALL_MARGIN
LOOKBACK_SMALL_MARGIN
METRIC_RANKING_DIVERGENCE
```

Phase 27 phải propagate vào:

```text
manifest
summary
report
winner artifact
Phase 28 reference update
```

Không tự xóa warning.

---

# 18. Swept factor duy nhất

```text
pooling_id
```

Allowed:

```text
P0 = LAST_STEP
P1 = MEAN
```

Không có third pooling option trong S5.

---

# 19. Frozen data configuration

Hard freeze:

```text
feature_variant_id = FV*
feature order = same
feature fingerprint = same
X scaler = same

target_scaling_id = YS*
target scaler/identity = same

lookback_id = L*
horizon = H1
boundary protocol = WB0
population = WINDOWPOP-v1

same Train target IDs
same Validation target IDs
```

---

# 20. Frozen Transformer Encoder configuration

Hard:

```text
input_size = same F

input projection = Linear(F,64)

positional encoding = SINUSOIDAL

d_model = 64
num_heads = 4
num_layers = 2
ffn_dim = 128
dropout = 0.1
activation = GELU

norm_policy = POST_NORM
norm_first = False

no causal mask
no padding mask

regression head = Linear(64,1)
output activation = NONE
```

Only:

```text
sequence-to-vector pooling operator
```

changes.

---

# 21. Frozen training configuration

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

# 22. Same encoded sequence semantics

The two models must be identical up to:

```text
encoded = encoder(...)
```

Pooling is applied only after full Encoder stack.

Do not implement MEAN in one model before the final encoder layer.

---

# 23. Correct architecture paths

LAST_STEP:

```text
X
↓
Input Projection
↓
Positional Encoding
↓
Encoder Layer 1
↓
Encoder Layer 2
↓
H [B,L,D]
↓
H[:, -1, :]
↓
Linear(D,1)
```

MEAN:

```text
X
↓
Input Projection
↓
Positional Encoding
↓
Encoder Layer 1
↓
Encoder Layer 2
↓
H [B,L,D]
↓
H.mean(dim=1)
↓
Linear(D,1)
```

---

# 24. No extra LayerNorm for MEAN

Do not add:

```text
LayerNorm after mean
```

unless LAST_STEP also has same additional LayerNorm and it was already part of frozen architecture.

Current S5 intended difference is only pooling.

---

# 25. No extra dropout for MEAN

Do not add:

```text
pooling dropout
head dropout
```

to one condition only.

---

# 26. No projection after MEAN

Do not add:

```text
Linear(D,D)
```

between mean pooling and regression head.

That would change architecture/capacity.

---

# 27. No learned attention pooling

Do not use:

```text
attention pooling
weighted mean
gated pooling
learned pooling query
```

in Phase 27.

Only:

```text
LAST_STEP
MEAN
```

---

# 28. No CLS token

Do not introduce a trainable `[CLS]` token.

That is a different architecture and parameter count.

---

# 29. No MAX pooling

Not part of registered S5 options.

---

# 30. No concatenated pooling

Do not test:

```text
concat(last, mean)
```

inside S5.

---

# 31. Parameter-count equality is a hard expectation

Both pooling strategies are parameter-free.

Therefore:

```text
total_parameters_LAST_STEP
==
total_parameters_MEAN

trainable_parameters_LAST_STEP
==
trainable_parameters_MEAN
```

Any mismatch:

```text
architecture drift.
```

---

# 32. Architecture fingerprint nuance

If architecture fingerprint includes pooling ID, full config fingerprint will differ.

But a fingerprint computed over **trainable parameter schema** should match.

Recommended store both:

```text
model_semantic_config_fingerprint
trainable_parameter_schema_fingerprint
```

Expected:

```text
semantic fingerprints differ by pooling
parameter-schema fingerprints equal.
```

---

# 33. Same input tensors are a hard requirement

For every sample:

```text
X_LAST_STEP == X_MEAN
```

No data transformation difference is permitted.

---

# 34. Same raw/model targets

Hard:

```text
y_raw_wh same
y_model same
```

because target scaling is frozen.

---

# 35. Same sample population

Hard:

```text
Train IDs identical
Validation IDs identical
WINDOWPOP-v1 fingerprint identical.
```

Pooling must not affect sample eligibility.

---

# 36. Same DataLoader contract

Both conditions:

```text
same batch size
same shuffle policy
same worker policy
same drop_last=False
same seed 42
```

---

# 37. Existing LAST_STEP reference reuse

S4 winner was trained with current baseline pooling:

```text
LAST_STEP
```

If exact S5 contract match:

```text
REUSE S4 winner
```

as P0 reference.

Do not retrain LAST_STEP.

---

# 38. Normal S5 run count

Expected:

```text
LAST_STEP
→ reused S4 winner

MEAN
→ one NEW training run
```

Therefore:

```text
1 new run
+
1 reused reference.
```

---

# 39. LAST_STEP reference reuse gate

Exact match required:

```text
FV*
YS*
L*
H1
WB0
WINDOWPOP-v1
B64

D64
H4
N2
FFN128
dropout .1
GELU
LAST_STEP

AdamW
LR3e-4
WD1e-4
MSE
E50
patience10
clip1
seed42

Training Engine
Metric version
```

Any mismatch:

```text
STOP.
```

---

# 40. Fresh MEAN run

MEAN condition execution:

```text
set seed 42
↓
fresh Train DataLoader
↓
fresh Validation DataLoader
↓
same feature/scaling/window pipeline
↓
fresh Transformer with pooling=MEAN
↓
MSE
↓
AdamW
↓
TRAINING_ENGINE-v1
```

No warm-start from LAST_STEP.

---

# 41. Why no warm-start

Weights up to encoder are shape-compatible, nhưng warm-start would mix:

```text
pooling effect
+
pretrained state from another pooling strategy.
```

Fresh initialization is mandatory.

---

# 42. Matched initialization opportunity

Because pooling is parameter-free and architecture parameter shapes are identical:

```text
same seed
same model construction order
```

should allow exact initial trainable parameter matching if runtime/environment and model builder are deterministic.

Recommended:

```text
initial_trainable_state_fingerprint_MEAN
==
initial_trainable_state_fingerprint_LAST_STEP_reference
```

if reference fingerprint exists.

If not available:

```text
NOT_VERIFIABLE
```

not FAIL.

---

# 43. Stronger matched-init option for future run creation

For the NEW MEAN run, before training store:

```text
initial_model_state_fingerprint
```

This helps later reproducibility.

Do not reconstruct/reference an assumed LAST_STEP initial state after the fact.

---

# 44. Same parameter initialization does not mean same training trajectory

Pooling changes:

```text
forward readout
loss
gradients
```

from batch one.

Different trajectories are expected.

---

# 45. Experiment identity

```text
sweep_id = S5_POOLING
experiment_family = TRANSFORMER_SWEEP_S5_POOLING
sweep_version = SWEEP_S5_POOLING-v1
```

Recommended labels:

```text
S5_<FV*>__<YS*>__<L*>__LAST_STEP__REFERENCE_S4

S5_<FV*>__<YS*>__<L*>__MEAN__S42
```

Actual `run_id` from Registry.

---

# 46. Test firewall

Hard:

```text
test_access_authorized = false
Test loader not iterated
Test metrics absent
Test predictions absent
```

---

# 47. Training objective

Same:

```text
MSE
```

in current selected target model space.

---

# 48. Model-space loss comparability

Because both conditions use same:

```text
YS*
MSE
```

their model-space train/Validation losses are on the same scale and may be compared descriptively.

Still:

```text
Validation RMSE Wh
```

is the selection metric.

---

# 49. Primary S5 selection metric

Hard:

```text
best_validation_rmse_wh
```

from verified BEST checkpoint.

---

# 50. Secondary metrics

```text
best_validation_mae_wh
best_validation_r2
best_epoch
epochs_completed
stop_reason
train/Validation loss trajectory
gradient diagnostics
clipping fraction
runtime
```

---

# 51. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{LAST},
RMSE_{MEAN}
\right)
\]

Use:

```text
full-precision Validation RMSE Wh.
```

---

# 52. Exact RMSE tie rule

If exact full-precision equality:

```text
prefer LAST_STEP
```

Rationale:

```text
baseline-established readout
direct recency-aligned sequence-to-one semantics
no sequence reduction operation
preserves existing reference
```

Tie event must be explicit.

---

# 53. No arbitrary minimum gain threshold

Do not invent:

```text
MEAN must improve >1%
```

Strict lower RMSE wins unless exact tie.

---

# 54. Pooling effect formula

Let:

```text
RMSE_L = LAST_STEP
RMSE_M = MEAN
```

Define:

\[
\Delta RMSE_{LAST\rightarrow MEAN}
=
RMSE_L-RMSE_M
\]

Positive:

```text
MEAN improves.
```

Relative:

\[
Improvement\%
=
100\times
\frac{
RMSE_L-RMSE_M
}{
RMSE_L
}
\]

Also:

\[
\Delta MAE=MAE_L-MAE_M
\]

\[
\Delta R^2=R^2_M-R^2_L
\]

---

# 55. Metric ranking divergence

If:

```text
MEAN wins RMSE
LAST_STEP wins MAE
```

record:

```text
METRIC_RANKING_DIVERGENCE.
```

Winner stays RMSE-based.

---

# 56. No significance testing

Only one seed per pooling condition.

Do not use:

```text
epochs
batches
sample errors
```

as independent model-run replicates for a pooling significance test in S5.

---

# 57. Best-checkpoint verification for MEAN

After training:

```text
fresh Transformer(pooling=MEAN)
↓
strict-load BEST
↓
full common Validation
↓
inverse target transform if YS1
↓
METRICS-v1
↓
verify recorded BEST metrics.
```

---

# 58. LAST_STEP reference verification

No retrain.

Verify:

```text
BEST checkpoint already verified
same target IDs
same current feature/scaler/lookback/target-scaling config
same metric version
exact S5 reference match.
```

---

# 59. Pooling implementation unit tests before training

Before official MEAN run, add targeted S5 implementation assertions.

Synthetic encoded tensor:

```text
H.shape = [B,L,D]
```

Verify:

```text
LAST_STEP(H)
==
H[:, -1, :]
```

and:

```text
MEAN(H)
==
H.mean(dim=1)
```

within exact/appropriate float tolerance.

---

# 60. Shape tests

For:

```text
B ∈ {1, 32, 64}
L = selected L*
D = 64
```

both pooling operators return:

```text
[B,64].
```

Then regression output:

```text
[B,1].
```

---

# 61. MEAN permutation diagnostic

Mathematically, pure mean pooling itself is invariant to permutation of its input representations:

```text
mean(H[:, perm, :], dim=1)
==
mean(H, dim=1)
```

This can be unit-tested for the pooling operator only.

Do **not** infer that the full Transformer MEAN model is permutation invariant, because positional encoding and Encoder processing occur before pooling.

---

# 62. LAST_STEP permutation diagnostic

Pure LAST_STEP operator is not sequence-permutation invariant because it selects index `-1`.

Unit tests can verify operator semantics, not model quality.

---

# 63. Mean denominator audit

Because no padding:

```text
denominator = L*
```

implicitly via `torch.mean`.

Do not accidentally divide twice.

---

# 64. No feature-dimension mean

Common bug:

```python
encoded.mean(dim=2)
```

Wrong.

Correct:

```python
encoded.mean(dim=1)
```

because dimension 1 is sequence length under `[B,L,D]`.

---

# 65. No batch-dimension mean

Common bug:

```python
encoded.mean(dim=0)
```

would mix samples and cause catastrophic cross-sample leakage.

Hard unit test:

```text
pooling must preserve B dimension.
```

---

# 66. Cross-sample independence

For MEAN, changing sample 0 input must not alter sample 1 prediction in eval mode.

This should already hold, but targeted sanity test is valuable.

---

# 67. No accidental squeeze

Do not let:

```text
B=1
```

collapse output shape.

Always output:

```text
[1,1]
```

not scalar.

---

# 68. No in-place mutation of encoded sequence

Pooling should not alter encoder output tensor contents.

Recommended unit check if implementation custom.

---

# 69. Gradient flow through MEAN

Synthetic backward test should verify:

```text
loss.backward()
```

produces finite gradients through:

```text
regression head
encoder
input projection
```

with MEAN pooling.

---

# 70. Gradient distribution interpretation

MEAN distributes readout gradient through all sequence representations at the pooling operation, whereas LAST_STEP injects head gradient directly through final encoded position before attention backprop propagates further.

This is a computational distinction.

Do not claim it necessarily yields better optimization.

---

# 71. Gradient diagnostics

Record per condition:

```text
global_max_grad_norm_preclip
mean_grad_norm_preclip
mean_fraction_batches_clipped
max_fraction_batches_clipped
epochs_with_any_clipping
nonfinite_gradient_events
```

---

# 72. Runtime diagnostics

Pooling operations are computationally small compared with Transformer Encoder.

Expected runtime difference should be minor relative to total attention cost, but observed runtime is source of truth.

Do not use runtime for winner selection.

---

# 73. Parameter-count diagnostic

Expected:

```text
identical.
```

Create explicit audit rather than assume.

---

# 74. Same input/output unit contract

Both:

```text
input tensor:
[B,L*,F]

output:
[B,1]

metrics:
Wh
```

---

# 75. Same prediction population

Official sweep comparison must use exact same:

```text
Validation sample_idx
```

and canonical order.

---

# 76. Run matrix

Create:

```text
s5_run_matrix.csv
```

Fields:

```text
sweep_id
pooling_id
pooling_name
source_type
source_run_id
requires_new_training
feature_variant_id
target_scaling_id
lookback_id
population_fingerprint
feature_fingerprint
x_scaler_id
target_transform_id
batch_size
seed
model_config_id
training_config_id
status
```

---

# 77. Sweep manifest

Create:

```text
s5_pooling_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S5_POOLING-v1
sweep_id = S5_POOLING
source_s4_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
candidate_poolings = [LAST_STEP, MEAN]
new_runs_required
reused_runs
swept_field = pooling_id
frozen_fields
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = LAST_STEP_ON_EXACT_RMSE_TIE
population_version = WINDOWPOP-v1
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

# 78. Sweep contract

Create:

```text
s5_pooling_sweep_contract.json
```

Must state:

```text
Only sequence pooling changes.

LAST_STEP:
encoded[:, -1, :]

MEAN:
encoded.mean(dim=1)

Pooling occurs after complete Encoder stack.

No padding.

No masked mean required.

No CLS token.

No max/attention/weighted pooling.

Same data/sample population.

Same feature/scaling/lookback.

Same Transformer trainable layers.

Same parameter count expected.

Same optimizer/training config.

Seed 42.

Validation RMSE Wh selects winner.

Exact tie → LAST_STEP.

Test forbidden.
```

---

# 79. Preflight audit

Create:

```text
s5_pooling_preflight_audit.csv
```

Checks:

```text
phase26_pass
approved_for_phase27
s4_winner_valid
feature_variant_locked
target_scaling_locked
lookback_locked
LAST_STEP_registered
MEAN_registered
population_fixed
Training_Engine_fixed
metric_fixed
seed_fixed
test_locked
status
```

---

# 80. Pooling definition audit

Create:

```text
s5_pooling_definition_audit.csv
```

Fields:

```text
pooling_id
pooling_name
input_rank
input_shape_contract
sequence_dim
operation
output_shape_contract
requires_parameters
requires_padding_mask
occurs_after_encoder
status
```

Expected:

```text
LAST_STEP:
operation = select index -1 on dim=1
parameters = 0

MEAN:
operation = arithmetic mean on dim=1
parameters = 0
```

---

# 81. Pooling unit-test artifact

Create:

```text
s5_pooling_unit_tests.csv
```

Tests include:

```text
exact LAST_STEP selection
exact MEAN reduction
shape B1
shape B32
shape B64
correct dim=1
no cross-batch reduction
finite output
backward finite
no parameter added
B dimension preserved
full model [B,1] output
```

---

# 82. Common-data audit

Create:

```text
s5_common_data_audit.csv
```

Fields:

```text
split_id
sample_count_last
sample_count_mean
sample_ids_equal
ordered_ids_equal
feature_fingerprint_equal
x_scaler_equal
target_transform_equal
lookback_equal
population_fingerprint_equal
status
```

---

# 83. Architecture audit

Create:

```text
s5_pooling_architecture_audit.csv
```

Fields:

```text
pooling_id
input_size
lookback
d_model
num_heads
num_layers
ffn_dim
dropout
activation
norm_policy
pe_policy
regression_head
trainable_parameters
parameter_schema_fingerprint
only_pooling_differs
status
```

---

# 84. Config-delta audit

Create:

```text
s5_config_delta_audit.csv
```

Expected semantic differences:

```text
pooling_id
pooling_name
pooling_operation
```

Potential runtime-derived differences:

```text
metrics
gradients
best epoch
runtime
```

No other config difference allowed.

---

# 85. Training-config audit

Create:

```text
s5_pooling_training_audit.csv
```

Fields:

```text
pooling_id
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

# 86. Initialization audit

Create:

```text
s5_initialization_audit.csv
```

Fields:

```text
pooling_id
initial_model_state_fingerprint
reference_available
parameter_schema_match
fingerprint_matches
seed
status
```

Expected if reference supports:

```text
matched trainable initialization.
```

---

# 87. Run provenance

Create:

```text
s5_pooling_run_provenance.csv
```

Fields:

```text
pooling_id
run_id
source_type
source_phase
config_fingerprint
parameter_schema_fingerprint
feature_fingerprint
population_fingerprint
best_checkpoint_sha256
history_sha256
metric_artifact
prediction_artifact
status
```

---

# 88. Primary metrics table

Create:

```text
s5_pooling_metrics.csv
```

Rows:

```text
LAST_STEP
MEAN
```

Fields:

```text
pooling_id
pooling_name
run_id
source_type
lookback_id
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

# 89. Pooling effect table

Create:

```text
s5_pooling_effect.csv
```

Fields:

```text
last_run_id
mean_run_id
last_rmse_wh
mean_rmse_wh
rmse_delta_last_to_mean_wh
rmse_improvement_pct
last_mae_wh
mean_mae_wh
mae_delta_wh
last_r2
mean_r2
r2_delta
rmse_winner
metric_ranking_divergence
status
```

---

# 90. Optimization diagnostics

Create:

```text
s5_optimization_diagnostics.csv
```

Fields:

```text
pooling_id
best_epoch
last_epoch
stop_reason
best_rmse_wh
last_rmse_wh
global_max_grad_norm_preclip
mean_grad_norm_preclip
mean_fraction_batches_clipped
nonfinite_events
mean_epoch_duration_seconds
status
```

---

# 91. Hypothesis outcomes

Create:

```text
s5_hypothesis_outcomes.csv
```

Fields:

```text
hypothesis_id
source_hypothesis_id_optional
comparison
expected_direction
last_rmse
mean_rmse
observed_delta
outcome
interpretation
status
```

Allowed:

```text
SUPPORTED
NOT_SUPPORTED
INCONCLUSIVE_TIE
```

---

# 92. Findings artifact

Create:

```text
s5_pooling_findings.csv
```

Possible codes:

```text
LAST_STEP_GAIN
MEAN_POOLING_GAIN
POOLING_EXACT_TIE
METRIC_RANKING_DIVERGENCE
CONVERGENCE_DIFFERENCE
GRADIENT_BEHAVIOR_DIFFERENCE
RUNTIME_DIFFERENCE
PARAMETER_EQUALITY_VERIFIED
MATCHED_INITIALIZATION_VERIFIED
MATCHED_INITIALIZATION_NOT_VERIFIABLE
INHERITED_WARNING
```

---

# 93. Winner artifact

Create:

```text
s5_pooling_winner.json
```

Minimum:

```text
sweep_id
sweep_version
feature_variant_id
target_scaling_id
lookback_id
selection_metric
selection_direction
tie_rule
winner_pooling_id
winner_pooling_name
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_pooling_id
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
trainable_parameters
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 94. Reference update for Phase 28

Create:

```text
s5_reference_update.json
```

Minimum:

```text
previous_reference_run_id
feature_variant_id
target_scaling_id
lookback_id
previous_pooling_id = LAST_STEP
selected_pooling_id
winner_run_id
winner_config_fingerprint
winner_rmse_wh
activation_state = GELU
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase28
```

---

# 95. Phase 28 handoff logic

Phase 28 tests:

```text
ReLU
vs
GELU
```

while holding:

```text
S5-selected pooling
S4-selected lookback
S3-selected target scaling
S2-selected feature variant
```

fixed.

Current S5 winner uses:

```text
GELU
```

because activation has not yet been swept.

Therefore Phase 28 should normally:

```text
reuse S5 winner as GELU reference
train only ReLU
```

if exact match.

---

# 96. Learning curves

Recommended:

```text
S5_01_validation_rmse_by_epoch.png
S5_02_validation_mae_by_epoch.png
S5_03_train_loss_by_epoch.png
S5_04_gradient_clipping_fraction.png
S5_05_best_validation_metrics.png
S5_06_best_to_last_rmse.png
```

---

# 97. Primary figure

```text
S5_01_validation_rmse_by_epoch.png
```

Overlay:

```text
LAST_STEP
MEAN
```

with best epoch markers.

---

# 98. Best metric figure

```text
S5_05_best_validation_metrics.png
```

source-derived from:

```text
s5_pooling_metrics.csv
```

---

# 99. No attention visualization

Pooling sweep does not require:

```text
attention heatmaps
head comparisons
last-query attention
```

Those remain final attention phases.

---

# 100. Important relation to future attention analysis

If MEAN wins final sequential selection at S5, later attention interpretation must not pretend final prediction is read only from the final query token.

Final attention analysis should account for the actual selected pooling.

However Phase 54's planned "last-query attention" can still be an auxiliary interpretability view, but must be framed consistently with final pooling.

This should be carried in lineage.

---

# 101. Interpretation if LAST_STEP wins

Safe:

> Under the current selected feature/target/lookback configuration, using the final contextualized timestep as the sequence readout achieved lower Validation RMSE than mean pooling.

Possible interpretation:

```text
recency-focused readout may be advantageous
```

but do not claim this mechanism is proven.

---

# 102. Interpretation if MEAN wins

Safe:

> Averaging contextualized representations across the full selected historical sequence achieved lower Validation RMSE than using only the final encoded timestep.

Do not say:

```text
all timesteps contributed equally to the prediction
```

because Encoder attention already mixes information.

---

# 103. Interpretation if exact tie

Select:

```text
LAST_STEP
```

by predeclared tie rule.

Report:

```text
No full-precision Validation RMSE advantage was observed for MEAN pooling.
```

---

# 104. Tiny non-zero margin

Strict lower RMSE still wins.

Report margin transparently and note:

```text
single-seed Validation evidence
```

rather than overstate.

---

# 105. Single-seed limitation

Mandatory:

```text
S5 pooling conditions are represented by seed 42.
```

No mean±std.

---

# 106. Validation-only limitation

Mandatory:

```text
S5 winner is a development selection based on Validation.
```

No Test evidence.

---

# 107. Interaction limitation

Pooling may interact with:

```text
lookback
activation
depth
d_model
heads
FFN
dropout
```

S5 estimates pooling effect only under current frozen configuration.

Do not claim global pooling optimum.

---

# 108. Sequential-selection limitation

By Phase 27, Validation has influenced:

```text
feature set
time features
target scaling
lookback
pooling
```

Complete experiment registration and later robustness validation are increasingly important.

---

# 109. No multiple pooling variants

Do not test ad hoc:

```text
MAX
SUM
attention pooling
CLS
last+mean concat
weighted mean
learned query
```

inside Phase 27.

---

# 110. No masking change

No causal/padding mask modification.

---

# 111. No head change

Regression head stays:

```text
Linear(64,1)
```

for both.

---

# 112. No output activation

No.

---

# 113. No head dropout

No.

---

# 114. No LayerNorm addition

No.

---

# 115. No feature aggregation change

No.

---

# 116. No lookback change

No.

---

# 117. No target scaling change

No.

---

# 118. No batch/LR change

No.

---

# 119. No gradient-clipping change

No.

---

# 120. No Test access

Hard.

---

# 121. Run failure policy

If MEAN run fails technically:

```text
S5 incomplete.
```

Do not automatically declare LAST_STEP winner.

---

# 122. Numerical failure

NaN/Inf:

```text
FAIL run.
```

No skipped batches.

---

# 123. OOM policy

Pooling itself adds negligible memory compared with encoder.

If MEAN OOMs while identical LAST_STEP reference succeeded:

```text
investigate infrastructure/config bug.
```

Do not switch batch size.

---

# 124. Technical rerun policy

Allowed only for documented:

```text
interrupt
artifact corruption
hardware/software fault
```

Not because score is poor.

---

# 125. Score-based rerun forbidden

No.

---

# 126. Discrepancy taxonomy

```text
S4_REFERENCE_MISSING
S4_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
POOLING_DEFINITION_MISMATCH
POOLING_APPLIED_BEFORE_ENCODER
WRONG_POOLING_DIMENSION
BATCH_DIMENSION_REDUCTION
PARAMETER_COUNT_MISMATCH
PARAMETER_SCHEMA_MISMATCH
REGRESSION_HEAD_DRIFT
EXTRA_LAYER_OR_DROPOUT
POPULATION_MISMATCH
TARGET_ID_MISMATCH
X_SCALER_MISMATCH
TARGET_SCALER_MISMATCH
ARCHITECTURE_DRIFT
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

# 127. Discrepancy log

Create:

```text
s5_pooling_discrepancies.json
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
batch-dimension mean
wrong population
Test access
pooling before Encoder

MAJOR:
parameter count mismatch
extra head layer
checkpoint verification failure

MODERATE:
metric ranking divergence
tiny margin
gradient behavior difference

INFO:
matched initialization not verifiable
minor runtime difference
```

---

# 129. Status model

## PASS

```text
LAST_STEP reference valid
MEAN implementation audit passes
same samples/model/training
MEAN run verified
winner selected
Phase 28 reference generated
Test untouched
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
metric ranking divergence
initial-state match not verifiable
inherited upstream warning
```

with methodology valid.

## FAIL

Examples:

```text
wrong pooling dimension
population mismatch
parameter mismatch
extra architecture change
MEAN run unresolved failure
Test access
```

---

# 130. Sweep summary

Create:

```text
s5_pooling_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
feature_variant_id
target_scaling_id
lookback_id
reference_run_id
mean_run_id
new_runs
reused_runs
primary_metric
metrics_by_pooling
pooling_effect
optimization_diagnostics
parameter_equality
initialization_match
winner
winner_margin
inherited_warnings
phase28_reference
test_status
overall_status
```

---

# 131. Human-readable report

Create:

```text
s5_pooling_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S4
3. LAST_STEP definition
4. MEAN definition
5. Controlled-variable contract
6. Common data/population fairness
7. Architecture/parameter fairness
8. Pooling implementation verification
9. Run provenance
10. Validation metrics
11. Pooling effect
12. Learning-curve/gradient context
13. S5 winner
14. Interpretation cautions
15. Limitations
16. Phase 28 handoff
```

---

# 132. Report language

For each result:

```text
Observed
Interpretation
Limitation
Handoff
```

Avoid:

```text
proves
always
causes
all timesteps equally important
```

unless actually supported, which S5 alone does not.

---

# 133. README

Create:

```text
README_S5_POOLING_SWEEP.md
```

Must explain:

```text
Purpose
S4 winner handoff
LAST_STEP semantics
MEAN semantics
mean over encoded representations
why no padding simplifies mean
same parameter count
reference reuse
matched-init opportunity
winner rule
tie rule
interpretation limitations
future attention-analysis implication
Phase 28 handoff
No Test
```

---

# 134. Output directory

```text
artifacts/
└── sweeps/
    └── S5_pooling/
        ├── s5_pooling_sweep_manifest.json
        ├── s5_pooling_sweep_contract.json
        ├── s5_pooling_preflight_audit.csv
        ├── s5_run_matrix.csv
        ├── s5_pooling_definition_audit.csv
        ├── s5_pooling_unit_tests.csv
        ├── s5_common_data_audit.csv
        ├── s5_pooling_architecture_audit.csv
        ├── s5_config_delta_audit.csv
        ├── s5_pooling_training_audit.csv
        ├── s5_initialization_audit.csv
        ├── s5_pooling_run_provenance.csv
        ├── s5_pooling_metrics.csv
        ├── s5_pooling_effect.csv
        ├── s5_optimization_diagnostics.csv
        ├── s5_hypothesis_outcomes.csv
        ├── s5_pooling_findings.csv
        ├── s5_pooling_winner.json
        ├── s5_reference_update.json
        ├── s5_pooling_sweep_tests.csv
        ├── s5_pooling_discrepancies.json
        ├── s5_pooling_sweep_summary.json
        ├── s5_pooling_sweep_report.md
        ├── figures/
        │   ├── S5_01_validation_rmse_by_epoch.png
        │   ├── S5_02_validation_mae_by_epoch.png
        │   ├── S5_03_train_loss_by_epoch.png
        │   ├── S5_04_gradient_clipping_fraction.png
        │   ├── S5_05_best_validation_metrics.png
        │   └── S5_06_best_to_last_rmse.png
        ├── README_S5_POOLING_SWEEP.md
        └── phase_27_signoff.json
```

New MEAN run remains under:

```text
artifacts/runs/<run_id>/
```

Do not duplicate checkpoint inside sweep folder.

---

# 135. Required outputs

```text
O27.1  Sweep manifest
O27.2  Sweep contract
O27.3  Preflight audit
O27.4  Run matrix
O27.5  Pooling definition audit
O27.6  Pooling unit tests
O27.7  Common-data audit
O27.8  Architecture/parameter audit
O27.9  Config-delta audit
O27.10 Training-config audit
O27.11 Initialization audit
O27.12 Run provenance
O27.13 Reused LAST_STEP reference
O27.14 Verified new MEAN run
O27.15 Metrics table
O27.16 Pooling effect
O27.17 Optimization diagnostics
O27.18 Hypothesis outcomes
O27.19 Findings
O27.20 Winner artifact
O27.21 Phase 28 reference update
O27.22 Figures
O27.23 Sweep test suite
O27.24 Discrepancy log
O27.25 Sweep summary
O27.26 Human-readable report
O27.27 README
O27.28 Phase sign-off
```

---

# 136. Sweep test suite

Create:

```text
s5_pooling_sweep_tests.csv
```

Recommended checks:

```text
S5T27-001 Phase 26 PASS/non-critical warning only
S5T27-002 approved_for_phase27 = true
S5T27-003 S4 winner artifact valid
S5T27-004 feature variant fixed
S5T27-005 target scaling fixed
S5T27-006 lookback fixed
S5T27-007 LAST_STEP registered
S5T27-008 MEAN registered
S5T27-009 LAST_STEP operation = encoded[:, -1, :]
S5T27-010 MEAN operation = encoded.mean(dim=1)
S5T27-011 pooling occurs after full Encoder
S5T27-012 no pooling before final layer
S5T27-013 no CLS token
S5T27-014 no max pooling
S5T27-015 no attention pooling
S5T27-016 no weighted mean
S5T27-017 no concat pooling
S5T27-018 no extra LayerNorm
S5T27-019 no extra pooling dropout
S5T27-020 regression head unchanged
S5T27-021 output activation unchanged
S5T27-022 LAST_STEP unit test exact
S5T27-023 MEAN unit test exact
S5T27-024 pooling dimension = 1
S5T27-025 batch dimension preserved
S5T27-026 feature dimension preserved as D
S5T27-027 B1 shape correct
S5T27-028 B32 shape correct
S5T27-029 B64 shape correct
S5T27-030 full output [B,1]
S5T27-031 MEAN finite backward
S5T27-032 no additional trainable pooling params
S5T27-033 same trainable parameter count
S5T27-034 same parameter schema fingerprint
S5T27-035 same feature fingerprint
S5T27-036 same X scaler
S5T27-037 same target transform
S5T27-038 same lookback
S5T27-039 same H1
S5T27-040 same WB0
S5T27-041 same WINDOWPOP-v1
S5T27-042 Train IDs identical
S5T27-043 Validation IDs identical
S5T27-044 B64 fixed
S5T27-045 D64 fixed
S5T27-046 H4 fixed
S5T27-047 N2 fixed
S5T27-048 FFN128 fixed
S5T27-049 dropout .1 fixed
S5T27-050 GELU fixed
S5T27-051 sinusoidal PE fixed
S5T27-052 POST_NORM fixed
S5T27-053 no causal mask
S5T27-054 no padding mask
S5T27-055 AdamW fixed
S5T27-056 LR3e-4 fixed
S5T27-057 WD1e-4 fixed
S5T27-058 MSE fixed
S5T27-059 E50 fixed
S5T27-060 patience10 fixed
S5T27-061 clip1 fixed
S5T27-062 seed42 fixed
S5T27-063 Training Engine fixed
S5T27-064 Metric version fixed
S5T27-065 LAST_STEP reference exact-match
S5T27-066 LAST_STEP reference reused
S5T27-067 MEAN run registered before training
S5T27-068 MEAN fresh loaders/model
S5T27-069 MEAN no warm-start
S5T27-070 MEAN trained via Training Engine
S5T27-071 MEAN BEST verified
S5T27-072 LAST_STEP BEST already verified
S5T27-073 both rows use verified BEST
S5T27-074 full-precision RMSE available
S5T27-075 RMSE effect correct
S5T27-076 relative improvement correct
S5T27-077 MAE effect correct
S5T27-078 R² effect correct
S5T27-079 winner = minimum RMSE
S5T27-080 exact tie → LAST_STEP
S5T27-081 metric divergence recorded
S5T27-082 optimization diagnostics generated
S5T27-083 parameter equality verified
S5T27-084 initialization match checked if available
S5T27-085 hypothesis outcomes generated
S5T27-086 findings generated
S5T27-087 winner points to valid run
S5T27-088 Phase 28 reference update generated
S5T27-089 GELU reference reuse identified for Phase 28
S5T27-090 inherited warnings propagated
S5T27-091 no score-based rerun
S5T27-092 no failed/SANITY run in ranking
S5T27-093 no attention extraction added
S5T27-094 no Test access
S5T27-095 single-seed limitation documented
S5T27-096 Validation-only limitation documented
S5T27-097 interaction limitation documented
S5T27-098 figures source-derived
S5T27-099 summary/report generated
S5T27-100 phase sign-off generated
```

---

# 137. Recommended notebook structure

```text
Cell 27.1  Phase title
Cell 27.2  Verify Phase 26 sign-off
Cell 27.3  Declare SWEEP_S5_POOLING-v1
Cell 27.4  Load S4 winner/reference update
Cell 27.5  Freeze FV*/YS*/L*
Cell 27.6  Load LAST_STEP/MEAN definitions
Cell 27.7  Build S5 run matrix
Cell 27.8  Run pooling operator unit tests
Cell 27.9  Audit pooling occurs after Encoder
Cell 27.10 Audit common data/population
Cell 27.11 Audit architecture/parameter equality
Cell 27.12 Audit config delta
Cell 27.13 Audit training config
Cell 27.14 Check initial-state fingerprint if available
Cell 27.15 Verify LAST_STEP reference reuse eligibility
Cell 27.16 Register MEAN run
Cell 27.17 Seed + fresh MEAN loaders/model
Cell 27.18 Execute MEAN via Training Engine
Cell 27.19 Verify MEAN best checkpoint
Cell 27.20 Build run provenance
Cell 27.21 Build S5 metrics table
Cell 27.22 Compute LAST→MEAN effect
Cell 27.23 Build optimization diagnostics
Cell 27.24 Evaluate hypotheses
Cell 27.25 Generate learning curves
Cell 27.26 Generate findings
Cell 27.27 Select S5 winner
Cell 27.28 Write winner JSON
Cell 27.29 Write Phase 28 reference update
Cell 27.30 Run S5 tests/discrepancies
Cell 27.31 Write summary/report
Cell 27.32 Register artifacts/checksums
Cell 27.33 Write README
Cell 27.34 Phase sign-off
```

---

# 138. Execution flow

```text
Verify Phase 26
        ↓
Load S4 winner
        ↓
Freeze feature variant + target scaling + lookback
        ↓
Resolve LAST_STEP / MEAN
        ↓
Unit-test pooling semantics
        ↓
Audit same X/y/sample population
        ↓
Audit same Encoder/head/parameter schema
        ↓
Verify LAST_STEP reference exact match
        ↓
Reuse LAST_STEP
        ↓
Register MEAN
        ↓
Seed 42
        ↓
Fresh loaders/model
        ↓
Train via TRAINING_ENGINE-v1
        ↓
Verify MEAN BEST
        ↓
Build same-population Wh metrics
        ↓
Compute pooling effect
        ↓
Analyze learning/gradient context
        ↓
Select minimum-RMSE pooling
        ↓
Apply LAST_STEP exact-tie rule
        ↓
Update Phase 28 reference
        ↓
Write S5 artifacts
        ↓
SWEEP_S5_POOLING-v1 sign-off
```

---

# 139. Fail-fast order

Before expensive MEAN training:

```text
1. Phase 26 sign-off
2. S4 winner identity
3. FV*/YS*/L* lock
4. pooling definitions
5. pooling unit tests
6. same sample population
7. same X/y transforms
8. architecture/parameter equality
9. training-config equality
10. LAST_STEP reuse eligibility
11. Test firewall
12. Registry readiness
```

---

# 140. Why pooling unit tests must happen before training

A one-line implementation mistake such as:

```python
encoded.mean(dim=2)
```

can produce a tensor that still has plausible dimensions somewhere downstream but completely changes semantics.

Therefore pooling operator correctness must be verified independently before scientific training.

---

# 141. Why batch-dimension protection is critical

This bug:

```python
encoded.mean(dim=0)
```

would mix examples from different samples.

That is scientifically invalid and can create direct cross-sample leakage within a batch.

Hard fail.

---

# 142. Why no masked mean is needed

All windows have fixed `L*` and no padding.

Introducing mask logic adds complexity without benefit.

If generic model code supports mask, `None` must reduce exactly to ordinary mean.

---

# 143. Why same parameter count matters

Pooling should alter only nonparametric readout operation.

Different parameter count indicates hidden architectural change.

---

# 144. Why reuse LAST_STEP

S4 winner is already the exact current reference under LAST_STEP.

Retraining it would add an unnecessary stochastic realization and create opportunity for cherry-picking.

---

# 145. Why no warm-start for MEAN

Even if Encoder weights could be copied, the experiment must compare training each pooling strategy under equivalent fresh initialization policy.

---

# 146. Winner verification checklist

Before `s5_pooling_winner.json`:

```text
[ ] LAST_STEP valid reused run.
[ ] MEAN valid completed run.
[ ] Same FV*.
[ ] Same YS*.
[ ] Same L*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same feature/scaler/target transform.
[ ] Same Encoder/head parameters.
[ ] Same trainable parameter count.
[ ] Same training config.
[ ] Same Training Engine.
[ ] Same Metric version.
[ ] Pooling definitions unit-tested.
[ ] Both BEST checkpoints verified.
[ ] Full-precision RMSE used.
[ ] Effect computed programmatically.
[ ] Exact tie rule respected.
[ ] No Test.
```

---

# 147. Phase 28 handoff

Phase 28 receives:

```text
s5_pooling_winner.json
s5_reference_update.json
winner_run_id
winner_config_fingerprint
feature_variant_id
target_scaling_id
lookback_id
selected_pooling_id
population_fingerprint
```

and changes only:

```text
activation ReLU/GELU.
```

---

# 148. If LAST_STEP wins

Current reference:

```text
FV*
YS*
L*
LAST_STEP
GELU
```

Phase 28:

```text
ReLU
vs
GELU
```

at LAST_STEP.

---

# 149. If MEAN wins

Current reference:

```text
FV*
YS*
L*
MEAN
GELU
```

Phase 28 compares activation at MEAN pooling.

---

# 150. Reference reuse for Phase 28

S5 winner already uses:

```text
GELU
```

so normally:

```text
reuse S5 winner as GELU reference
train only ReLU.
```

---

# 151. Relationship with future attention analysis

Pooling choice affects interpretation.

If final model uses LAST_STEP:

```text
last-query attention
```

has direct readout relevance.

If final model uses MEAN:

```text
prediction is based on average of all final encoded positions
```

so focusing solely on last-query attention would be incomplete.

Final attention phases must read final pooling from model lock.

---

# 152. Relationship with Phase 34 Heads

Pooling selection occurs before head-count sweep.

Potential interaction exists, but sequential one-factor design accepts this limitation.

---

# 153. Relationship with Phase 35 Layers

Same.

Do not rerun pooling after every later architecture choice unless candidate synthesis/protocol explicitly requires it.

---

# 154. Relationship with Phase 42 candidate synthesis

All S5 conditions remain registered.

Candidate synthesis can trace:

```text
current sequential winner path
```

and compare against historical alternatives.

---

# 155. Relationship with Phase 44 rolling-origin

S5 winner is one Validation-period selection.

Rolling-origin later tests robustness.

---

# 156. Relationship with Phase 46 multi-seed

S5 uses seed 42 only.

Tiny pooling margins are not seed-robust evidence.

---

# 157. No fabricated outputs

Do not pre-fill:

```text
winner
RMSE
best epoch
gradient norm
runtime
initial-state match
```

before execution.

---

# 158. Phase sign-off

Create:

```text
phase_27_signoff.json
```

Minimum:

```text
phase = 27
phase_name = S5 Pooling sweep
sweep_version
sweep_id
source_s4_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
last_step_reference_run_id
mean_run_id
new_run_ids
reused_run_ids
winner_pooling_id
winner_pooling_name
winner_run_id
winner_rmse_wh
population_fingerprint
metric_version
pooling_unit_test_status
parameter_equality_status
fairness_audit_status
matched_initialization_status
inherited_warnings
test_status
approved_for_phase28
overall_status
created_at
```

---

# 159. Acceptance checklist

```text
[ ] Phase 26 valid.
[ ] approved_for_phase27 = true.
[ ] S5 version declared.
[ ] S4 winner loaded.
[ ] Feature variant fixed.
[ ] Target scaling fixed.
[ ] Lookback fixed.
[ ] LAST_STEP registered.
[ ] MEAN registered.
[ ] LAST_STEP = encoded[:, -1, :].
[ ] MEAN = encoded.mean(dim=1).
[ ] Pooling after full Encoder.
[ ] No pooling before final layer.
[ ] No CLS token.
[ ] No MAX pooling.
[ ] No attention pooling.
[ ] No weighted mean.
[ ] No concat pooling.
[ ] No extra LayerNorm.
[ ] No extra dropout.
[ ] Same regression head.
[ ] Same output activation policy.
[ ] Pooling unit tests PASS.
[ ] Correct sequence dimension.
[ ] Batch dimension preserved.
[ ] B1/B32/B64 shape tests PASS.
[ ] MEAN backward finite.
[ ] No trainable pooling parameters.
[ ] Same parameter count.
[ ] Same parameter schema.
[ ] Same Train IDs.
[ ] Same Validation IDs.
[ ] Same WINDOWPOP-v1.
[ ] Same feature fingerprint.
[ ] Same X scaler.
[ ] Same target transform/scaler.
[ ] Same H1/WB0.
[ ] Same B64.
[ ] Same D64/H4/N2/FFN128.
[ ] Same dropout .1.
[ ] Same GELU.
[ ] Same sinusoidal PE.
[ ] Same POST_NORM.
[ ] No causal mask.
[ ] No padding mask.
[ ] Same AdamW/LR/WD/MSE.
[ ] Same E50/patience10/clip1.
[ ] Same seed42.
[ ] Same Training Engine.
[ ] Same Metric version.
[ ] LAST_STEP reference exact-match.
[ ] LAST_STEP not retrained.
[ ] MEAN registered before training.
[ ] MEAN fresh loaders/model.
[ ] MEAN no warm-start.
[ ] MEAN trained via TRAINING_ENGINE-v1.
[ ] MEAN BEST verified.
[ ] LAST_STEP BEST already verified.
[ ] Both metrics use verified BEST.
[ ] Full-precision RMSE used.
[ ] Pooling effect computed.
[ ] MAE/R² effects computed.
[ ] Winner=min RMSE.
[ ] Exact tie=LAST_STEP.
[ ] Metric divergence recorded if present.
[ ] Optimization diagnostics generated.
[ ] Parameter equality verified.
[ ] Initial-state match checked if available.
[ ] Hypothesis outcomes recorded.
[ ] Findings generated.
[ ] Inherited warnings propagated.
[ ] Winner artifact generated.
[ ] Phase 28 reference update generated.
[ ] GELU reference reuse identified.
[ ] No ad hoc pooling experiment.
[ ] No score-based rerun.
[ ] No failed/SANITY run in ranking.
[ ] No attention extraction added.
[ ] No Test access.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] Interaction limitation documented.
[ ] Future attention interpretation note documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 160. Acceptance criteria

Phase 27 chỉ PASS khi:

```text
S4-selected feature/target/lookback configuration is fixed.

Exactly LAST_STEP and MEAN are compared.

Pooling occurs only after the full Encoder stack.

LAST_STEP and MEAN definitions are verified by unit tests.

Same X, y, sample IDs and population are used.

Same Encoder and regression head are used.

Pooling adds no trainable parameters.

Parameter counts/schemas are equal.

Training configuration and seed are identical.

LAST_STEP reference is valid and reused.

MEAN is one fresh seed-42 run.

No warm-start occurs.

Both BEST checkpoints are verified.

Validation RMSE Wh selects winner.

Exact RMSE tie selects LAST_STEP.

Phase 28 reference is generated.

No hidden pooling experiment/rerun occurs.

Test remains untouched.
```

---

# 161. Failure conditions

Phase 27 FAIL if:

```text
wrong S4 winner used

feature/target/lookback changes

MEAN averages wrong dimension

batch dimension is averaged

pooling applied before Encoder completion

CLS/max/attention pooling introduced

extra LayerNorm/dropout/projection added

parameter counts differ

sample IDs/population differ

scaler/target transform differs

training config differs

LAST_STEP reference mismatches but is reused

LAST_STEP is retrained and better rerun cherry-picked

MEAN warm-starts

MEAN run fails but LAST_STEP declared winner

RMSE rounded before ranking

MAE/R² overrides lower RMSE

attention collection added during training

Test used

score-based rerun occurs.
```

---

# 162. Common mistakes

## 162.1 `mean(dim=2)` thay vì `mean(dim=1)`

Sai pooling dimension.

## 162.2 `mean(dim=0)`

Nghiêm trọng vì mix samples trong batch.

## 162.3 Average raw X trước Transformer

Không phải MEAN pooling contract.

## 162.4 Mean after only layer 1

Không cùng Encoder architecture.

## 162.5 Add LayerNorm sau MEAN

Hidden factor.

## 162.6 Add dropout sau MEAN

Hidden factor.

## 162.7 Add learned attention pooling

Không thuộc S5.

## 162.8 Add CLS token

Không thuộc S5 và thay parameter count.

## 162.9 Retrain LAST_STEP rồi chọn run đẹp hơn

Hidden rerun bias.

## 162.10 Warm-start MEAN từ LAST_STEP

Confounded.

## 162.11 Chọn MEAN vì runtime nhanh hơn

Sai primary metric.

## 162.12 Chọn LAST_STEP vì “hợp lý hơn” dù RMSE kém hơn

Sai. Tie rule chỉ khi exact RMSE tie.

## 162.13 Kết luận MEAN thắng nghĩa là mọi timestep quan trọng bằng nhau

Không đúng.

## 162.14 Bật attention map để quyết định pooling

Không Phase 27.

## 162.15 Dùng Test làm tie-breaker

Forbidden.

---

# 163. Recommended execution pseudocode

```text
load_phase26_signoff()
assert_approved_for_phase27()

s4 = load_s4_winner()

FV = s4.feature_variant_id
YS = s4.target_scaling_id
L  = s4.lookback_id

poolings = ["LAST_STEP", "MEAN"]

audit_pooling_definitions()
run_pooling_unit_tests()
audit_same_data_population(FV, YS, L)
audit_same_encoder_and_head()
audit_parameter_schema_equality()
audit_training_config()

last_reference = resolve_s4_winner_run()
assert_exact_s5_reference_match(last_reference)

register_mean_run()
seed(42)

loaders = build_fresh_loaders(
    feature_variant=FV,
    target_scaling=YS,
    lookback=L,
    population="WINDOWPOP-v1"
)

model = build_fresh_transformer(
    input_size=feature_count(FV),
    pooling="MEAN"
)

mean_result = TRAINING_ENGINE_v1.fit(...)
verify_best_checkpoint(mean_result)

results = {
    "LAST_STEP": last_reference,
    "MEAN": mean_result
}

metrics = build_verified_s5_metrics(results)
effect = compute_pooling_effect(metrics)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer="LAST_STEP"
)

write_optimization_diagnostics()
write_hypothesis_outcomes()
write_findings()
write_s5_winner(winner)
write_phase28_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 164. Definition of Done

\[
\boxed{
One\ Fixed\ Encoder\ Setup
+
Two\ Pooling\ Strategies
+
One\ Fresh\ MEAN\ Run
+
One\ Valid\ Reused\ LAST\ Run
+
Same\ Population
+
Same\ Parameters
+
Verified\ Pooling\ Semantics
+
Verified\ BEST\ Metrics
+
S5\ Winner
+
Phase28\ Reference
+
No\ Test
}
\]

---

# 165. Final status contract

```text
PHASE 27 tests POOLING only.

Current feature variant:
comes from S2/S3 path.

Target scaling:
comes from S3.

Lookback:
comes from S4.

Conditions:

LAST_STEP
= encoded[:, -1, :]

MEAN
= encoded.mean(dim=1)

Both operate:
after full Transformer Encoder.

No padding.
No CLS.
No attention pooling.
No max pooling.
No extra layer/dropout.

LAST_STEP:
reuse S4 winner if exact match.

MEAN:
one new fresh run.

Same:
X
y
sample IDs
WINDOWPOP-v1
feature/scaler pipeline
Encoder
regression head
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

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer LAST_STEP.

No warm-start.
No hidden rerun.
No attention analysis.
No Test.

After SWEEP_S5_POOLING-v1 PASS:
update current reference
and proceed to
PHASE 28 — S6 Activation sweep.
```

---

# 166. Final check

Correct workflow:

```text
Load S4 winner
→ Freeze FV* + YS* + L*
→ Verify LAST_STEP/MEAN definitions
→ Unit-test pooling dimensions
→ Audit same samples/parameters/training
→ Reuse LAST_STEP
→ Train one fresh MEAN
→ Verify BEST
→ Compare full-precision Validation RMSE
→ Interpret pooling effect cautiously
→ Select S5 winner
→ Update Phase 28 reference
```

Incorrect workflow:

```text
try mean/max/attention/CLS
→ add LayerNorm to one option
→ retrain LAST_STEP
→ keep best-looking run
→ inspect Test.
```

Chỉ sau khi `SWEEP_S5_POOLING-v1` được sign-off mới chuyển sang **PHASE 28 — S6 Activation sweep**.
