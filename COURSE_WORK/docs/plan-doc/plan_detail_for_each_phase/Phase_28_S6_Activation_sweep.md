# PHASE 28 — S6 ACTIVATION SWEEP

## Kế hoạch controlled sweep cho FFN activation function của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S5_POOLING-v1`  
**Sweep ID:** `S6_ACTIVATION`  
**Output version:** `SWEEP_S6_ACTIVATION-v1`  
**Phase trước:** `Phase_27_S5_Pooling_sweep.md`

---

# 1. Vai trò của Phase 28

Phase 28 là controlled experiment thứ sáu trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target scaling, lookback length, pooling strategy, Transformer width/depth, optimizer, learning rate, regularization, training protocol, sample population và seed đã được khóa từ Phase 27, activation function nào trong Transformer feed-forward network phù hợp hơn cho bài toán dự báo `Appliances`: `ReLU` hay `GELU`?

Phase 28 chỉ thay đúng một conceptual factor:

```text
FFN ACTIVATION
```

với hai condition:

```text
A0 = ReLU
A1 = GELU
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Factor
+
Same\ Encoder
+
Same\ Samples
+
Same\ Parameters
+
Same\ Training
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí Phase 28 trong master execution plan

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

Phase 29
→ S7 Batch-size sweep
```

Phase 28 không được quay lại thay:

```text
feature set
time features
target scaling
lookback
pooling
batch size
learning rate
weight decay
dropout
d_model
heads
layers
FFN width
loss
epoch cap
gradient clipping
RevIN
boundary protocol
```

Các yếu tố trên giữ nguyên current reference từ S5.

---

# 3. Câu hỏi nghiên cứu của S6

Phase 28 phải trả lời:

```text
1. ReLU hay GELU tạo Validation RMSE Wh thấp hơn?

2. Activation choice có ảnh hưởng convergence speed không?

3. Activation choice có ảnh hưởng gradient stability/clipping burden không?

4. Có metric-ranking divergence giữa RMSE, MAE và R² không?

5. ReLU có tạo dấu hiệu activation sparsity mạnh hơn không?

6. GELU có cho learning trajectory mượt hơn không?

7. Activation nào trở thành current reference cho Phase 29?
```

---

# 4. Current reference từ Phase 27

Phase 28 phải load:

```text
s5_pooling_winner.json
s5_reference_update.json
phase_27_signoff.json
```

để resolve current configuration:

```text
FV* = selected feature variant
YS* = selected target scaling
L*  = selected lookback
P*  = selected pooling
```

Possible:

```text
P* ∈ {LAST_STEP, MEAN}
```

Phase 28 không hard-code `LAST_STEP`.

---

# 5. Feed-forward network contract

Mỗi Transformer Encoder layer có FFN:

\[
D
\rightarrow
D_{ff}
\rightarrow
D
\]

Current baseline:

```text
d_model = 64
ffn_dim = 128
```

Canonical form:

```text
x
→ Linear(64,128)
→ Activation
→ Dropout
→ Linear(128,64)
→ Dropout
→ Residual
→ LayerNorm
```

Activation sweep chỉ thay:

```text
Activation
```

ở vị trí này.

---

# 6. A0 — ReLU

Canonical mathematical definition:

\[
ReLU(x)=\max(0,x)
\]

Expected implementation semantic:

```text
activation = ReLU
```

ReLU:

```text
zeroes negative pre-activations
is piecewise linear
has zero derivative for negative inputs
introduces exact activation sparsity
```

Phase 28 không thay ReLU bằng:

```text
LeakyReLU
PReLU
ELU
SiLU
```

---

# 7. A1 — GELU

Canonical GELU semantic:

\[
GELU(x)=x\Phi(x)
\]

where \(\Phi(x)\) is the standard Gaussian cumulative distribution function.

Current project reference uses:

```text
activation = GELU
```

với implementation semantics đúng theo `TRANSFORMER_IMPL-v1`.

Phase 28 không được thay implementation mode của GELU giữa reference và new run.

---

# 8. GELU implementation identity matters

Nếu Transformer implementation/config stores details such as:

```text
activation_name
activation_approximation
```

Phase 28 phải verify reference GELU exactly matches the frozen model implementation contract.

Do not silently change:

```text
exact GELU
→ tanh approximation
```

or vice versa.

That would create an additional hidden implementation factor.

---

# 9. Activation location is fixed

Activation applies only inside each Encoder FFN.

Do not change:

```text
regression output activation
input projection activation
pooling activation
attention logits
LayerNorm
```

Phase 28 is **FFN activation sweep only**.

---

# 10. No activation on regression output

Regression head remains:

```text
Linear(d_model,1)
```

with:

```text
no ReLU
no GELU
no Softplus
no Sigmoid
no clamp
```

Prediction must remain unconstrained real-valued regression output.

---

# 11. Same activation in all Encoder layers

Current architecture has:

```text
num_layers = 2
```

For each condition:

```text
ReLU condition:
Layer 1 FFN = ReLU
Layer 2 FFN = ReLU

GELU condition:
Layer 1 FFN = GELU
Layer 2 FFN = GELU
```

Do not mix:

```text
Layer 1 ReLU
Layer 2 GELU
```

inside S6.

---

# 12. Working hypotheses

## H-S6-01 — GELU may provide smoother nonlinear optimization

If:

\[
RMSE(GELU)<RMSE(ReLU)
\]

this supports the empirical hypothesis that GELU is more suitable under the current Transformer configuration.

Possible mechanism:

```text
smooth nonlinear gating
```

but S6 does not causally prove that mechanism.

Status:

```text
UNTESTED
```

---

# 13. H-S6-02 — ReLU may be sufficient or preferable

If:

\[
RMSE(ReLU)\le RMSE(GELU)
\]

this indicates that the simpler piecewise-linear activation is at least as effective under the current controlled configuration.

Do not generalize to:

```text
ReLU is universally better for time-series Transformers.
```

---

# 14. Preconditions bắt buộc

Phase 28 chỉ bắt đầu khi:

```text
Phase 27 = PASS
```

hoặc `PASS_WITH_WARNING` nhưng không có unresolved critical issue.

Bắt buộc:

```text
approved_for_phase28 = true
```

và available:

```text
s5_pooling_winner.json
s5_reference_update.json
phase_27_signoff.json
```

---

# 15. Upstream contracts bắt buộc

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
SWEEP_S5_POOLING-v1
```

---

# 16. Carry-forward warnings

Nếu upstream có warning:

```text
RANDOM_CONTROL_GAIN
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
INITIALIZATION_NOT_VERIFIABLE
```

Phase 28 phải propagate vào:

```text
manifest
summary
report
winner artifact
Phase 29 reference update
```

Không tự xóa warning.

---

# 17. Swept factor duy nhất

```text
activation_id
```

Allowed:

```text
A0 = ReLU
A1 = GELU
```

Không có activation thứ ba.

---

# 18. Frozen data configuration

Hard:

```text
feature_variant_id = FV*
feature order = same
feature fingerprint = same
X scaler = same

target_scaling_id = YS*
target transform/scaler = same

lookback_id = L*
horizon = H1
boundary = WB0
population = WINDOWPOP-v1

Train target IDs = same
Validation target IDs = same
```

---

# 19. Frozen pooling configuration

Hard:

```text
pooling_id = P*
```

Possible:

```text
LAST_STEP
MEAN
```

Activation sweep cannot alter pooling.

---

# 20. Frozen Transformer architecture

```text
input_size = same F

d_model = 64
num_heads = 4
num_layers = 2
ffn_dim = 128
dropout = 0.1

pooling = P*

sinusoidal positional encoding
POST_NORM
norm_first = False

no causal mask
no padding mask

regression head = Linear(64,1)
no output activation
```

Only:

```text
FFN activation
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

# 22. Same sample population

Hard:

```text
Train IDs ReLU == Train IDs GELU
Validation IDs ReLU == Validation IDs GELU
population fingerprint identical.
```

Activation must not change Dataset/DataLoader sample eligibility.

---

# 23. Same input tensors

For a given sample:

```text
X_ReLU == X_GELU
```

exactly under same dtype/preprocessing.

---

# 24. Same target tensors

For a given sample:

```text
y_model_ReLU == y_model_GELU
y_raw_wh_ReLU == y_raw_wh_GELU
```

because target scaling is fixed.

---

# 25. Parameter-count equality

ReLU and GELU are parameter-free activations.

Therefore hard expected:

```text
total_parameters_ReLU
==
total_parameters_GELU

trainable_parameters_ReLU
==
trainable_parameters_GELU
```

Any mismatch means hidden architecture drift.

---

# 26. Parameter-schema equality

Recommended hard audit:

```text
parameter name
parameter shape
requires_grad
```

must be identical across conditions.

Activation modules themselves add zero parameters.

---

# 27. Module-tree audit

Expected module-tree difference should be limited to activation module/config identity.

No new:

```text
Linear
LayerNorm
Dropout
Parameter
Buffer with scientific meaning
```

should appear.

---

# 28. Existing GELU reference reuse

The S5 winner still uses current baseline activation:

```text
GELU
```

because activation has not yet been swept.

If exact S6 contract match:

```text
REUSE S5 winner
```

as GELU reference.

Do not retrain GELU.

---

# 29. Normal S6 run count

Typically:

```text
GELU
→ reused S5 winner

ReLU
→ one NEW training run
```

Therefore:

```text
1 new run
+
1 reused reference.
```

---

# 30. GELU reference reuse gate

Exact match required on:

```text
FV*
YS*
L*
P*
H1
WB0
WINDOWPOP-v1

B64
D64/H4/N2/FFN128
dropout .1
GELU
sinusoidal PE
POST_NORM

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

Mismatch:

```text
STOP.
```

---

# 31. Fresh ReLU run

ReLU condition:

```text
set seed 42
↓
fresh Train DataLoader
↓
fresh Validation DataLoader
↓
same data/scaler/window pipeline
↓
fresh Transformer with activation=ReLU
↓
MSE
↓
AdamW
↓
TRAINING_ENGINE-v1
```

No warm-start from GELU.

---

# 32. Why no warm-start

Even though all parameter shapes match, warm-start would compare:

```text
GELU-trained weights adapted to ReLU
```

against:

```text
GELU trained from scratch.
```

That is not a clean activation experiment.

Fresh initialization is mandatory.

---

# 33. Matched initialization opportunity

Because ReLU/GELU are parameter-free and all parameter tensors have the same shapes:

```text
same seed
same model construction order
```

should permit matched initial trainable parameters under a deterministic same-environment setup.

Recommended:

```text
initial_trainable_state_fingerprint_ReLU
==
initial_trainable_state_fingerprint_GELU_reference
```

if reference stored initial fingerprint.

If upstream reference lacks it:

```text
NOT_VERIFIABLE
```

not FAIL.

---

# 34. Why matched initialization is especially useful in S6

Unlike feature-set or time-feature sweeps:

```text
input dimensionality does not change.
```

So activation comparison can potentially isolate the factor very tightly.

Store initial fingerprint for the new ReLU run regardless.

---

# 35. Activation implementation verification before training

Before official ReLU run, verify synthetic forward behavior.

For representative tensor `z` containing:

```text
negative
zero
positive
```

values, ReLU output must satisfy:

```text
negative → 0
zero → 0
positive → unchanged positive
```

within exact float semantics.

---

# 36. GELU reference verification

Do not recompute scientific run, but implementation contract should verify GELU module identity and configuration from:

```text
TRANSFORMER_IMPL-v1
S5 source config
```

---

# 37. No activation alias ambiguity

Avoid loose config strings that could map differently.

Recommended canonical IDs:

```text
A0_RELU
A1_GELU
```

and explicit config fields:

```text
activation_id
activation_name
activation_implementation
activation_kwargs
```

---

# 38. Activation factory

Recommended centralized factory:

```text
build_activation(config)
```

rather than branching in multiple encoder locations.

Factory must return fresh modules/semantics consistently for every layer.

---

# 39. Fail on unknown activation

Do not silently default unknown strings to GELU/ReLU.

Example:

```text
activation="geluu"
```

must:

```text
RAISE CONFIG ERROR.
```

---

# 40. Activation unit tests

Create tests for:

```text
A0 resolves ReLU
A1 resolves GELU
unknown activation rejected
same activation used across all encoder layers
activation occurs in FFN location
regression head unaffected
parameter count unchanged
```

---

# 41. ReLU dead-activation diagnostic

Optional but useful diagnostic:

Measure on a small fixed Validation or synthetic probe:

```text
fraction of FFN hidden activations equal to zero
```

for ReLU.

This is diagnostic only.

Do not use it for winner selection.

---

# 42. GELU near-zero diagnostic

Do not compare:

```text
exact zero rate
```

between GELU/ReLU as if same concept.

GELU generally yields smooth nonzero outputs except exact numeric cases.

Use activation-output distribution carefully.

---

# 43. Activation instrumentation must not alter training path

If activation statistics are collected:

```text
use temporary hooks
small diagnostic batches
eval mode
after training or separate sanity pass
```

Do not keep expensive hooks during every training batch unless already planned and proven non-invasive.

Core training should remain clean.

---

# 44. Recommended activation diagnostics

Optional post-run diagnostic fields:

```text
pre_activation_mean
pre_activation_std
post_activation_mean
post_activation_std
post_activation_zero_fraction
post_activation_nonfinite_count
```

for selected fixed diagnostic batch.

Label:

```text
DIAGNOSTIC_ONLY
```

---

# 45. Same diagnostic batch

If comparing activation outputs:

```text
same fixed sample_idx
same input batch
same checkpoint type = BEST
```

must be used.

Do not compare different random batches.

---

# 46. Activation output diagnostics are not causal performance proof

If GELU has smoother output distribution and lower RMSE, do not claim distribution difference caused the RMSE improvement.

Report association only.

---

# 47. Experiment identity

```text
sweep_id = S6_ACTIVATION
experiment_family = TRANSFORMER_SWEEP_S6_ACTIVATION
sweep_version = SWEEP_S6_ACTIVATION-v1
```

Recommended labels:

```text
S6_<FV*>__<YS*>__<L*>__<P*>__GELU__REFERENCE_S5

S6_<FV*>__<YS*>__<L*>__<P*>__RELU__S42
```

Registry `run_id` remains authoritative.

---

# 48. Test firewall

Hard:

```text
test_access_authorized = false
Test loader not iterated
Test prediction absent
Test metrics absent
```

---

# 49. Training objective

Same:

```text
MSE
```

in current selected target model space.

---

# 50. Model-space loss comparability

Because:

```text
same YS*
same loss
same targets
```

ReLU/GELU model-space loss curves are directly comparable descriptively.

Still:

```text
Validation RMSE Wh
```

is the scientific selection metric.

---

# 51. Primary S6 selection metric

Hard:

```text
best_validation_rmse_wh
```

from verified BEST checkpoint.

---

# 52. Secondary metrics

```text
best_validation_mae_wh
best_validation_r2
best_epoch
epochs_completed
stop_reason
train/Validation loss curves
gradient diagnostics
clipping fraction
runtime
activation diagnostics optional
```

---

# 53. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{ReLU},
RMSE_{GELU}
\right)
\]

using full-precision Validation RMSE Wh.

---

# 54. Exact RMSE tie rule

If exact full-precision equality:

```text
prefer ReLU
```

Rationale:

```text
simpler activation
piecewise-linear operation
conceptual parsimony
```

Tie event must be explicit.

This tie rule does not override a non-zero RMSE advantage for GELU.

---

# 55. Why ReLU is tie-preferred

The tie rule is a parsimony rule only.

It is not a claim that:

```text
ReLU is inherently superior
```

or always faster on every device.

Runtime performance is not used as the scientific reason unless separately measured and reported as engineering context.

---

# 56. No arbitrary minimum improvement threshold

Do not invent:

```text
GELU must improve >1%
```

Strict lower RMSE wins unless exact tie.

---

# 57. Activation effect formula

Let:

```text
RMSE_R = ReLU
RMSE_G = GELU
```

Define ReLU→GELU effect:

\[
\Delta RMSE_{ReLU\rightarrow GELU}
=
RMSE_R-RMSE_G
\]

Positive:

```text
GELU improves.
```

Relative:

\[
Improvement\%
=
100\times
\frac{
RMSE_R-RMSE_G
}{
RMSE_R
}
\]

Also:

\[
\Delta MAE=MAE_R-MAE_G
\]

\[
\Delta R^2=R^2_G-R^2_R
\]

---

# 58. Metric ranking divergence

If:

```text
GELU wins RMSE
ReLU wins MAE
```

record:

```text
METRIC_RANKING_DIVERGENCE
```

Winner remains RMSE-based.

---

# 59. No significance testing

Only one seed condition per activation.

Do not use:

```text
epoch metrics
batch losses
sample residuals
```

as independent activation replicates for p-values.

---

# 60. Best-checkpoint verification for ReLU

After training:

```text
fresh Transformer(activation=ReLU)
↓
strict-load BEST
↓
full Validation
↓
inverse target transform if YS1
↓
METRICS-v1
↓
verify stored BEST metrics.
```

---

# 61. GELU reference verification

No retraining.

Verify:

```text
BEST already verified
same sample IDs
same data/scaler/lookback/pooling
same architecture/training config
same metric version
activation=GELU exactly
```

---

# 62. Same parameter schema audit

For ReLU/GELU models:

```text
sorted named_parameters()
```

must match on:

```text
name
shape
requires_grad
dtype class expectation
```

before device-specific runtime changes.

---

# 63. Same state-dict key set

Recommended:

```text
state_dict keys ReLU == GELU
```

because activation modules are parameter-free.

If key set differs, investigate.

---

# 64. State-dict interoperability nuance

Because parameter shapes should match, a ReLU model may technically load a GELU model state dict.

Do **not** use that to warm-start.

This property may be unit-tested for architecture schema only, but not used in scientific training.

---

# 65. Gradient-flow test

Synthetic backward:

```text
forward
MSE
backward
```

for ReLU and GELU should produce finite gradients across:

```text
input projection
attention
FFN linears
LayerNorm
regression head
```

No optimizer step needed in implementation sanity.

---

# 66. Gradient diagnostics

Official run record:

```text
global_max_grad_norm_preclip
mean_grad_norm_preclip
mean_fraction_batches_clipped
max_fraction_batches_clipped
epochs_with_any_clipping
nonfinite_gradient_events
```

---

# 67. ReLU gradient interpretation caution

Zero gradient through negative ReLU activations is expected local behavior.

Do not label that alone as:

```text
training failure
```

Need empirical evidence such as severe activation collapse or degraded metrics.

---

# 68. Activation-collapse diagnostic

If instrumentation is used, possible flag:

```text
HIGH_RELU_ZERO_FRACTION
```

only as diagnostic.

Do not define an arbitrary universal failure threshold unless pre-registered.

---

# 69. Runtime diagnostics

Record:

```text
total runtime
mean epoch duration
median epoch duration
time-to-best optional
```

Runtime does not determine S6 winner.

---

# 70. Same parameter count implies runtime difference mostly operator/backend dependent

But actual runtime can also vary due:

```text
early stopping epoch
backend kernels
system load
```

so do not infer activation computational complexity solely from total runtime.

---

# 71. Same training epoch count comparison

If comparing per-epoch runtime, use:

```text
mean/median epoch duration
```

rather than total runtime alone when runs stop at different epochs.

Still engineering context only.

---

# 72. Run matrix

Create:

```text
s6_run_matrix.csv
```

Fields:

```text
sweep_id
activation_id
activation_name
source_type
source_run_id
requires_new_training
feature_variant_id
target_scaling_id
lookback_id
pooling_id
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

# 73. Sweep manifest

Create:

```text
s6_activation_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S6_ACTIVATION-v1
sweep_id = S6_ACTIVATION
source_s5_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
candidate_activations = [RELU, GELU]
new_runs_required
reused_runs
swept_field = activation_id
frozen_fields
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = RELU_ON_EXACT_RMSE_TIE
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

# 74. Sweep contract

Create:

```text
s6_activation_sweep_contract.json
```

Must state:

```text
Only FFN activation changes.

A0 = ReLU.
A1 = GELU.

Activation applies inside every Encoder FFN.

No regression-output activation.

No change to dropout/LN/FFN width.

Same feature/scaling/lookback/pooling.

Same sample population.

Same Transformer parameter schema.

Same optimizer/training protocol.

GELU reference reused if exact match.

ReLU one fresh run.

Seed 42.

Validation RMSE Wh selects winner.

Exact tie → ReLU.

No Test.
```

---

# 75. Preflight audit

Create:

```text
s6_activation_preflight_audit.csv
```

Checks:

```text
phase27_pass
approved_for_phase28
s5_winner_valid
feature_variant_locked
target_scaling_locked
lookback_locked
pooling_locked
RELU_registered
GELU_registered
population_fixed
parameter_schema_fixed
Training_Engine_fixed
metric_fixed
seed_fixed
test_locked
status
```

---

# 76. Activation-definition audit

Create:

```text
s6_activation_definition_audit.csv
```

Fields:

```text
activation_id
activation_name
implementation_class
activation_kwargs
ffn_location_verified
same_across_layers
parameter_count
output_activation_unchanged
status
```

---

# 77. Activation unit tests artifact

Create:

```text
s6_activation_unit_tests.csv
```

Recommended tests:

```text
ReLU negative→0
ReLU zero→0
ReLU positive→identity
GELU module/config identity
unknown activation rejected
activation placed after first FFN linear
activation before FFN dropout/second linear as frozen
all encoder layers use same activation
regression output has no activation
parameter count unchanged
state_dict keys equal
synthetic forward finite
synthetic backward finite
B1/B32/B64 output shapes valid
```

---

# 78. Common-data audit

Create:

```text
s6_common_data_audit.csv
```

Fields:

```text
split_id
sample_count_relu
sample_count_gelu
sample_ids_equal
ordered_ids_equal
feature_fingerprint_equal
x_scaler_equal
target_transform_equal
lookback_equal
pooling_equal
population_fingerprint_equal
status
```

---

# 79. Architecture audit

Create:

```text
s6_activation_architecture_audit.csv
```

Fields:

```text
activation_id
input_size
lookback
pooling
d_model
num_heads
num_layers
ffn_dim
dropout
norm_policy
pe_policy
regression_head
trainable_parameters
parameter_schema_fingerprint
only_activation_differs
status
```

---

# 80. Config-delta audit

Create:

```text
s6_config_delta_audit.csv
```

Expected semantic differences only:

```text
activation_id
activation_name
activation_implementation
activation_kwargs if relevant
```

Everything else must match.

---

# 81. Training-config audit

Create:

```text
s6_activation_training_audit.csv
```

Fields:

```text
activation_id
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

# 82. Initialization audit

Create:

```text
s6_initialization_audit.csv
```

Fields:

```text
activation_id
initial_model_state_fingerprint
reference_available
parameter_schema_match
fingerprint_matches
seed
status
```

---

# 83. Run provenance

Create:

```text
s6_activation_run_provenance.csv
```

Fields:

```text
activation_id
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

# 84. Primary metrics table

Create:

```text
s6_activation_metrics.csv
```

Rows:

```text
ReLU
GELU
```

Fields:

```text
activation_id
activation_name
run_id
source_type
pooling_id
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

# 85. Activation effect table

Create:

```text
s6_activation_effect.csv
```

Fields:

```text
relu_run_id
gelu_run_id
relu_rmse_wh
gelu_rmse_wh
rmse_delta_relu_to_gelu_wh
rmse_improvement_pct
relu_mae_wh
gelu_mae_wh
mae_delta_wh
relu_r2
gelu_r2
r2_delta
rmse_winner
metric_ranking_divergence
status
```

---

# 86. Optimization diagnostics

Create:

```text
s6_optimization_diagnostics.csv
```

Fields:

```text
activation_id
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

# 87. Optional activation diagnostics

Create if instrumentation is implemented cleanly:

```text
s6_activation_output_diagnostics.csv
```

Fields:

```text
activation_id
layer_index
diagnostic_batch_fingerprint
pre_activation_mean
pre_activation_std
post_activation_mean
post_activation_std
post_activation_zero_fraction
nonfinite_count
status
```

For GELU:

```text
zero_fraction
```

should be interpreted differently than ReLU.

This file is optional and cannot block S6 unless instrumentation itself detects a real nonfinite error.

---

# 88. Hypothesis outcomes

Create:

```text
s6_hypothesis_outcomes.csv
```

Fields:

```text
hypothesis_id
source_hypothesis_id_optional
comparison
expected_direction
relu_rmse
gelu_rmse
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

within single-seed Validation context.

---

# 89. Findings artifact

Create:

```text
s6_activation_findings.csv
```

Possible codes:

```text
RELU_GAIN
GELU_GAIN
ACTIVATION_EXACT_TIE
METRIC_RANKING_DIVERGENCE
CONVERGENCE_DIFFERENCE
GRADIENT_BEHAVIOR_DIFFERENCE
CLIPPING_DIFFERENCE
RUNTIME_DIFFERENCE
RELU_ZERO_ACTIVITY_PATTERN
PARAMETER_EQUALITY_VERIFIED
MATCHED_INITIALIZATION_VERIFIED
MATCHED_INITIALIZATION_NOT_VERIFIABLE
INHERITED_WARNING
```

---

# 90. Winner artifact

Create:

```text
s6_activation_winner.json
```

Minimum:

```text
sweep_id
sweep_version
feature_variant_id
target_scaling_id
lookback_id
pooling_id
selection_metric
selection_direction
tie_rule
winner_activation_id
winner_activation_name
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_activation_id
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

# 91. Reference update for Phase 29

Create:

```text
s6_reference_update.json
```

Minimum:

```text
previous_reference_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
previous_activation_id = GELU
selected_activation_id
winner_run_id
winner_config_fingerprint
winner_rmse_wh
batch_state = B64
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase29
```

---

# 92. Phase 29 handoff logic

Phase 29 tests:

```text
B32
vs
B64
```

while holding:

```text
S6-selected activation
S5-selected pooling
S4-selected lookback
S3-selected target scaling
S2-selected feature variant
```

fixed.

Current S6 winner was trained at:

```text
B64
```

therefore Phase 29 should normally:

```text
reuse S6 winner as B64 reference
train only B32
```

if exact-match.

---

# 93. Learning curves

Recommended:

```text
S6_01_validation_rmse_by_epoch.png
S6_02_validation_mae_by_epoch.png
S6_03_train_loss_by_epoch.png
S6_04_gradient_norm_by_epoch.png
S6_05_gradient_clipping_fraction.png
S6_06_best_validation_metrics.png
```

---

# 94. Primary figure

```text
S6_01_validation_rmse_by_epoch.png
```

Overlay:

```text
ReLU
GELU
```

with best epoch markers.

---

# 95. Train-loss figure

Because same target scaling and MSE:

```text
train loss curves are numerically comparable
```

descriptively.

But winner still uses:

```text
Validation RMSE Wh.
```

---

# 96. Activation diagnostic plot optional

If activation instrumentation is used:

```text
S6_07_activation_output_distribution.png
```

can compare hidden FFN output distributions on the same fixed diagnostic batch.

Diagnostic only.

---

# 97. No activation-output instrumentation during main training by default

Keep training path minimal and comparable.

Hooks can add:

```text
memory
runtime
state
```

complexity.

Post-checkpoint diagnostics are safer.

---

# 98. Interpretation if GELU wins

Safe:

> Under the current sequentially selected Transformer configuration, GELU in the Encoder FFN achieved lower Validation RMSE than ReLU.

Possible interpretation:

```text
smooth activation may be favorable under this optimization setting
```

but not proven mechanism.

---

# 99. Interpretation if ReLU wins

Safe:

> ReLU achieved lower Validation RMSE than GELU under the frozen current configuration.

Do not generalize beyond this setup.

---

# 100. Interpretation if exact tie

Select:

```text
ReLU
```

by parsimony.

Report:

```text
No full-precision Validation RMSE advantage was observed for GELU.
```

---

# 101. Tiny non-zero margin

Strict lower RMSE wins.

Report margin honestly:

```text
small single-seed Validation margin
```

and avoid strong claims.

---

# 102. Single-seed limitation

Mandatory:

```text
S6 conditions are represented by seed 42.
```

No mean±std.

---

# 103. Validation-only limitation

Mandatory:

```text
S6 winner is development selection based on Validation.
```

No Test evidence.

---

# 104. Interaction limitation

Activation may interact with:

```text
learning rate
dropout
depth
FFN width
d_model
batch size
loss
```

S6 evaluates activation only under current frozen configuration.

Do not claim global optimum.

---

# 105. Sequential-selection limitation

By Phase 28, Validation has influenced:

```text
feature set
time features
target scaling
lookback
pooling
activation
```

This makes:

```text
Experiment Registry
rolling-origin robustness
multi-seed final runs
untouched Test
```

increasingly important.

---

# 106. No other activation

Do not add:

```text
LeakyReLU
SiLU
Swish
ELU
SELU
Tanh
Mish
```

inside S6.

---

# 107. No activation hyperparameter tuning

Do not tune:

```text
negative slope
GELU approximation
custom temperature
```

unless already frozen in implementation.

---

# 108. No output activation

Hard.

---

# 109. No loss change

MSE fixed.

Huber belongs Phase 37.

---

# 110. No batch change

B64 fixed.

Phase 29 next.

---

# 111. No LR change

Phase 30.

---

# 112. No WD change

Phase 31.

---

# 113. No dropout change

Phase 32.

---

# 114. No d_model change

Phase 33.

---

# 115. No heads/layers/FFN changes

Phases 34–36.

---

# 116. No attention analysis

Not Phase 28.

---

# 117. No residual regime analysis

Not Phase 28.

---

# 118. No Test access

Hard.

---

# 119. Run failure policy

If ReLU run fails technically:

```text
S6 incomplete.
```

Do not automatically declare GELU winner.

---

# 120. Numerical failure

NaN/Inf:

```text
FAIL run.
```

No skipped batches.

---

# 121. OOM policy

Parameter count/memory footprint should be essentially unchanged.

If ReLU OOMs while GELU reference succeeded:

```text
audit environment/config
```

rather than reducing batch size.

---

# 122. Technical rerun policy

Allowed only for documented:

```text
interrupt
corrupt artifact
hardware/software failure
```

Not:

```text
poor score
```

---

# 123. Score-based rerun forbidden

No.

---

# 124. Discrepancy taxonomy

```text
S5_REFERENCE_MISSING
S5_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
POOLING_DRIFT
ACTIVATION_DEFINITION_MISMATCH
ACTIVATION_LOCATION_MISMATCH
MIXED_LAYER_ACTIVATIONS
GELU_IMPLEMENTATION_DRIFT
OUTPUT_ACTIVATION_ADDED
UNKNOWN_ACTIVATION_FALLBACK
PARAMETER_COUNT_MISMATCH
PARAMETER_SCHEMA_MISMATCH
MODULE_TREE_DRIFT
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

# 125. Discrepancy log

Create:

```text
s6_activation_discrepancies.json
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

# 126. Severity examples

```text
CRITICAL:
output activation added
wrong population
Test access
activation used outside intended FFN path

MAJOR:
parameter schema mismatch
mixed activations across layers
wrong GELU implementation
checkpoint verification failure

MODERATE:
metric ranking divergence
tiny RMSE margin
activation diagnostic anomaly

INFO:
initialization match not verifiable
minor runtime difference
```

---

# 127. Status model

## PASS

```text
GELU reference valid
ReLU implementation audit valid
same data/model/training
ReLU run verified
winner selected
Phase 29 reference generated
Test untouched
```

## PASS_WITH_WARNING

Possible:

```text
tiny margin
metric ranking divergence
activation distribution warning
matched initialization not verifiable
inherited upstream warning
```

with methodology valid.

## FAIL

Examples:

```text
activation location wrong
parameter mismatch
population mismatch
reference mismatch
ReLU unresolved failure
Test access
```

---

# 128. Sweep summary

Create:

```text
s6_activation_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
feature_variant_id
target_scaling_id
lookback_id
pooling_id
reference_run_id
relu_run_id
new_runs
reused_runs
primary_metric
metrics_by_activation
activation_effect
optimization_diagnostics
parameter_equality
initialization_match
activation_diagnostics_optional
winner
winner_margin
inherited_warnings
phase29_reference
test_status
overall_status
```

---

# 129. Human-readable report

Create:

```text
s6_activation_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S5
3. ReLU/GELU definitions
4. FFN activation location
5. Controlled-variable contract
6. Common data/population fairness
7. Parameter/module fairness
8. Activation implementation verification
9. Run provenance
10. Validation metrics
11. Activation effect
12. Learning-curve/gradient context
13. Optional activation-output diagnostics
14. S6 winner
15. Interpretation cautions
16. Limitations
17. Phase 29 handoff
```

---

# 130. Report wording

Use:

```text
Observed
Interpretation
Limitation
Handoff
```

Avoid:

```text
GELU is always superior
ReLU causes dead neurons and therefore loses
smoothness proves better optimization
```

without direct evidence.

---

# 131. README

Create:

```text
README_S6_ACTIVATION_SWEEP.md
```

Must explain:

```text
Purpose
S5 winner handoff
ReLU/GELU definitions
activation location inside FFN
same activation across all layers
no output activation
same parameter count/schema
GELU reference reuse
matched-init opportunity
winner rule
tie rule
optional activation diagnostics
limitations
Phase 29 handoff
No Test
```

---

# 132. Output directory

```text
artifacts/
└── sweeps/
    └── S6_activation/
        ├── s6_activation_sweep_manifest.json
        ├── s6_activation_sweep_contract.json
        ├── s6_activation_preflight_audit.csv
        ├── s6_run_matrix.csv
        ├── s6_activation_definition_audit.csv
        ├── s6_activation_unit_tests.csv
        ├── s6_common_data_audit.csv
        ├── s6_activation_architecture_audit.csv
        ├── s6_config_delta_audit.csv
        ├── s6_activation_training_audit.csv
        ├── s6_initialization_audit.csv
        ├── s6_activation_run_provenance.csv
        ├── s6_activation_metrics.csv
        ├── s6_activation_effect.csv
        ├── s6_optimization_diagnostics.csv
        ├── s6_activation_output_diagnostics.csv
        ├── s6_hypothesis_outcomes.csv
        ├── s6_activation_findings.csv
        ├── s6_activation_winner.json
        ├── s6_reference_update.json
        ├── s6_activation_sweep_tests.csv
        ├── s6_activation_discrepancies.json
        ├── s6_activation_sweep_summary.json
        ├── s6_activation_sweep_report.md
        ├── figures/
        │   ├── S6_01_validation_rmse_by_epoch.png
        │   ├── S6_02_validation_mae_by_epoch.png
        │   ├── S6_03_train_loss_by_epoch.png
        │   ├── S6_04_gradient_norm_by_epoch.png
        │   ├── S6_05_gradient_clipping_fraction.png
        │   ├── S6_06_best_validation_metrics.png
        │   └── S6_07_activation_output_distribution.png
        ├── README_S6_ACTIVATION_SWEEP.md
        └── phase_28_signoff.json
```

`S6_07` và `s6_activation_output_diagnostics.csv` là optional nếu instrumentation được triển khai.

New ReLU run remains under:

```text
artifacts/runs/<run_id>/
```

No checkpoint duplication in sweep folder.

---

# 133. Required outputs

```text
O28.1  Sweep manifest
O28.2  Sweep contract
O28.3  Preflight audit
O28.4  Run matrix
O28.5  Activation-definition audit
O28.6  Activation unit tests
O28.7  Common-data audit
O28.8  Architecture/parameter audit
O28.9  Config-delta audit
O28.10 Training-config audit
O28.11 Initialization audit
O28.12 Run provenance
O28.13 Reused GELU reference
O28.14 Verified new ReLU run
O28.15 Metrics table
O28.16 Activation effect
O28.17 Optimization diagnostics
O28.18 Optional activation-output diagnostics
O28.19 Hypothesis outcomes
O28.20 Findings
O28.21 Winner artifact
O28.22 Phase 29 reference update
O28.23 Figures
O28.24 Sweep test suite
O28.25 Discrepancy log
O28.26 Sweep summary
O28.27 Human-readable report
O28.28 README
O28.29 Phase sign-off
```

---

# 134. Sweep test suite

Create:

```text
s6_activation_sweep_tests.csv
```

Recommended checks:

```text
S6T28-001 Phase 27 PASS/non-critical warning only
S6T28-002 approved_for_phase28 = true
S6T28-003 S5 winner valid
S6T28-004 feature variant fixed
S6T28-005 target scaling fixed
S6T28-006 lookback fixed
S6T28-007 pooling fixed
S6T28-008 ReLU registered
S6T28-009 GELU registered
S6T28-010 ReLU factory resolves correctly
S6T28-011 GELU factory resolves correctly
S6T28-012 unknown activation rejected
S6T28-013 activation inside FFN only
S6T28-014 activation after first FFN linear
S6T28-015 activation ordering matches frozen Encoder contract
S6T28-016 same activation across Encoder layers
S6T28-017 no mixed ReLU/GELU layers
S6T28-018 regression output has no activation
S6T28-019 no input-projection activation added
S6T28-020 ReLU negative→0
S6T28-021 ReLU zero→0
S6T28-022 ReLU positive→identity
S6T28-023 GELU implementation identity verified
S6T28-024 GELU kwargs/approximation unchanged
S6T28-025 synthetic forward ReLU finite
S6T28-026 synthetic forward GELU finite
S6T28-027 synthetic backward ReLU finite
S6T28-028 synthetic backward GELU finite
S6T28-029 B1 output shape [1,1]
S6T28-030 B32 output shape [32,1]
S6T28-031 B64 output shape [64,1]
S6T28-032 same parameter count
S6T28-033 same parameter schema
S6T28-034 same state_dict key set
S6T28-035 same Train IDs
S6T28-036 same Validation IDs
S6T28-037 same WINDOWPOP-v1
S6T28-038 same feature fingerprint
S6T28-039 same X scaler
S6T28-040 same target transform/scaler
S6T28-041 same lookback
S6T28-042 same pooling
S6T28-043 same H1
S6T28-044 same WB0
S6T28-045 B64 fixed
S6T28-046 D64 fixed
S6T28-047 H4 fixed
S6T28-048 N2 fixed
S6T28-049 FFN128 fixed
S6T28-050 dropout .1 fixed
S6T28-051 sinusoidal PE fixed
S6T28-052 POST_NORM fixed
S6T28-053 no causal mask
S6T28-054 no padding mask
S6T28-055 regression head identical
S6T28-056 AdamW fixed
S6T28-057 LR3e-4 fixed
S6T28-058 WD1e-4 fixed
S6T28-059 MSE fixed
S6T28-060 E50 fixed
S6T28-061 patience10 fixed
S6T28-062 clip1 fixed
S6T28-063 seed42 fixed
S6T28-064 Training Engine fixed
S6T28-065 Metric version fixed
S6T28-066 GELU reference exact-match
S6T28-067 GELU reference reused
S6T28-068 ReLU run registered before training
S6T28-069 ReLU fresh loaders/model
S6T28-070 ReLU no warm-start
S6T28-071 ReLU trained via Training Engine
S6T28-072 ReLU BEST verified
S6T28-073 GELU BEST already verified
S6T28-074 both rows use verified BEST
S6T28-075 full-precision RMSE available
S6T28-076 RMSE effect correct
S6T28-077 relative improvement correct
S6T28-078 MAE effect correct
S6T28-079 R² effect correct
S6T28-080 winner = minimum RMSE
S6T28-081 exact tie → ReLU
S6T28-082 metric divergence recorded if present
S6T28-083 optimization diagnostics generated
S6T28-084 parameter equality verified
S6T28-085 initialization match checked if available
S6T28-086 optional activation diagnostics use same fixed batch
S6T28-087 optional hooks not active during main training
S6T28-088 hypothesis outcomes generated
S6T28-089 findings generated
S6T28-090 winner points to valid run
S6T28-091 Phase 29 reference update generated
S6T28-092 B64 reference reuse identified for Phase 29
S6T28-093 inherited warnings propagated
S6T28-094 no extra activation experiment
S6T28-095 no score-based rerun
S6T28-096 no failed/SANITY run in ranking
S6T28-097 no Test access
S6T28-098 single-seed limitation documented
S6T28-099 Validation-only limitation documented
S6T28-100 interaction limitation documented
S6T28-101 figures source-derived
S6T28-102 summary/report generated
S6T28-103 phase sign-off generated
```

---

# 135. Recommended notebook structure

```text
Cell 28.1  Phase title
Cell 28.2  Verify Phase 27 sign-off
Cell 28.3  Declare SWEEP_S6_ACTIVATION-v1
Cell 28.4  Load S5 winner/reference update
Cell 28.5  Freeze FV*/YS*/L*/P*
Cell 28.6  Resolve ReLU/GELU definitions
Cell 28.7  Build S6 run matrix
Cell 28.8  Run activation-factory tests
Cell 28.9  Run ReLU semantic unit tests
Cell 28.10 Verify GELU implementation identity
Cell 28.11 Audit activation location across layers
Cell 28.12 Audit same common data/population
Cell 28.13 Audit parameter/module schema equality
Cell 28.14 Audit config delta
Cell 28.15 Audit training config
Cell 28.16 Check initial-state fingerprint if available
Cell 28.17 Verify GELU reference reuse eligibility
Cell 28.18 Register ReLU run
Cell 28.19 Seed + fresh ReLU loaders/model
Cell 28.20 Execute ReLU via Training Engine
Cell 28.21 Verify ReLU BEST
Cell 28.22 Build run provenance
Cell 28.23 Build S6 metrics table
Cell 28.24 Compute ReLU→GELU effect
Cell 28.25 Build optimization diagnostics
Cell 28.26 Optional activation-output diagnostic pass
Cell 28.27 Evaluate hypotheses
Cell 28.28 Generate learning curves
Cell 28.29 Generate findings
Cell 28.30 Select S6 winner
Cell 28.31 Write winner JSON
Cell 28.32 Write Phase 29 reference update
Cell 28.33 Run S6 tests/discrepancies
Cell 28.34 Write summary/report
Cell 28.35 Register artifacts/checksums
Cell 28.36 Write README
Cell 28.37 Phase sign-off
```

---

# 136. Execution flow

```text
Verify Phase 27
        ↓
Load S5 winner
        ↓
Freeze FV* + YS* + L* + P*
        ↓
Resolve ReLU/GELU
        ↓
Audit activation factory/semantics
        ↓
Audit activation location across all layers
        ↓
Audit same sample population
        ↓
Audit same parameter schema/model config
        ↓
Verify GELU reference exact match
        ↓
Reuse GELU
        ↓
Register ReLU
        ↓
Seed 42
        ↓
Fresh loaders/model
        ↓
Train via TRAINING_ENGINE-v1
        ↓
Verify ReLU BEST
        ↓
Build Wh-space metrics
        ↓
Compute activation effect
        ↓
Analyze learning/gradient context
        ↓
Optional fixed-batch activation diagnostics
        ↓
Select minimum-RMSE activation
        ↓
Apply ReLU exact-tie rule
        ↓
Update Phase 29 reference
        ↓
Write S6 artifacts
        ↓
SWEEP_S6_ACTIVATION-v1 sign-off
```

---

# 137. Fail-fast order

Before expensive ReLU training:

```text
1. Phase 27 sign-off
2. S5 winner identity
3. FV*/YS*/L*/P* lock
4. activation definitions
5. activation factory tests
6. activation placement audit
7. GELU implementation identity
8. same population/data
9. parameter/schema equality
10. training-config equality
11. GELU reference reuse eligibility
12. Test firewall
13. Registry readiness
```

---

# 138. Why activation placement audit matters

A config could say:

```text
activation = ReLU
```

while implementation accidentally applies activation:

```text
after second FFN Linear
```

or outside FFN.

Then S6 would not test the intended Transformer FFN activation.

Audit actual module/forward semantics, not config string only.

---

# 139. Why same activation across all layers matters

If only one layer changes:

```text
Layer1 ReLU
Layer2 GELU
```

the condition becomes a mixed-activation architecture, not A0.

Hard fail.

---

# 140. Why no output ReLU

Energy target is nonnegative in dataset context, but adding ReLU to final output would:

```text
constrain regression range
change loss geometry
alter architecture
```

and confound S6.

Output remains unconstrained.

---

# 141. Why parameter equality matters

Activation modules are parameter-free.

If ReLU/GELU models have different trainable parameter counts, something unintended changed.

---

# 142. Why reuse GELU

S5 winner is already the exact current configuration under GELU.

Retraining creates:

```text
unnecessary stochastic rerun
score selection opportunity
extra compute
```

Reuse is scientifically cleaner.

---

# 143. Why no warm-start ReLU

Copying GELU-trained weights would cause the activation comparison to depend on pretraining under GELU.

Fresh seed-42 ReLU training preserves one-factor intent.

---

# 144. Winner verification checklist

Before writing `s6_activation_winner.json`:

```text
[ ] GELU valid reused reference.
[ ] ReLU valid completed run.
[ ] Same FV*.
[ ] Same YS*.
[ ] Same L*.
[ ] Same P*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same feature/scaler/target transform.
[ ] Same parameter schema/count.
[ ] Same training config.
[ ] Same Training Engine.
[ ] Same Metric version.
[ ] Activation implementations verified.
[ ] ReLU/GELU only differ by activation.
[ ] Both BEST checkpoints verified.
[ ] Full-precision RMSE used.
[ ] Effect calculated programmatically.
[ ] Exact tie rule respected.
[ ] No Test.
```

---

# 145. Phase 29 handoff

Phase 29 receives:

```text
s6_activation_winner.json
s6_reference_update.json
winner_run_id
winner_config_fingerprint
feature_variant_id
target_scaling_id
lookback_id
pooling_id
selected_activation_id
population_fingerprint
```

and changes only:

```text
batch size.
```

---

# 146. If GELU wins

Current reference:

```text
FV*
YS*
L*
P*
GELU
B64
```

Phase 29 compares:

```text
B32
vs
B64
```

at GELU.

---

# 147. If ReLU wins

Current reference:

```text
FV*
YS*
L*
P*
ReLU
B64
```

Phase 29 compares batch sizes at ReLU.

---

# 148. Reference reuse for Phase 29

S6 winner already uses:

```text
B64
```

so Phase 29 normally:

```text
reuse S6 winner as B64 reference
train only B32.
```

---

# 149. Relationship with Phase 30 LR

Activation choice may interact with learning rate.

S6 winner is selected at:

```text
LR = 3e-4
```

Phase 30 later tests LR on the current selected activation.

Do not rerun S6 for each LR inside Phase 28.

---

# 150. Relationship with Phase 32 dropout

Activation may interact with dropout.

Sequential one-factor design records this limitation.

---

# 151. Relationship with Phase 36 FFN width

Activation occurs within FFN, so interaction with FFN width is possible.

Still not tested here.

---

# 152. Relationship with Phase 42 candidate synthesis

Both ReLU and GELU results remain in Registry.

No losing run is deleted.

---

# 153. Relationship with Phase 44 rolling-origin

S6 is selected on one Validation period.

Later rolling-origin robustness is required.

---

# 154. Relationship with Phase 46 multi-seed

S6 uses seed 42.

Small activation margins may not be seed-stable.

---

# 155. Reproducibility metadata

New ReLU run records:

```text
run_id
seed
environment
device
feature fingerprint
X scaler checksum
target transform checksum
lookback ID
pooling ID
activation ID
activation implementation metadata
population fingerprint
model config fingerprint
parameter schema fingerprint
initial model state fingerprint
Training Engine fingerprint
best checkpoint checksum
history checksum
metric checksum
```

GELU keeps existing provenance.

---

# 156. No fabricated runtime outputs

Do not pre-fill:

```text
winner
RMSE
MAE
R²
best epoch
gradient norms
clipping fraction
activation zero fraction
runtime
```

before execution.

---

# 157. Phase sign-off

Create:

```text
phase_28_signoff.json
```

Minimum:

```text
phase = 28
phase_name = S6 Activation sweep
sweep_version
sweep_id
source_s5_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
relu_run_id
gelu_reference_run_id
new_run_ids
reused_run_ids
winner_activation_id
winner_activation_name
winner_run_id
winner_rmse_wh
population_fingerprint
metric_version
activation_definition_audit_status
parameter_equality_status
fairness_audit_status
matched_initialization_status
inherited_warnings
test_status
approved_for_phase29
overall_status
created_at
```

---

# 158. Acceptance checklist

```text
[ ] Phase 27 valid.
[ ] approved_for_phase28 = true.
[ ] S6 version declared.
[ ] S5 winner loaded.
[ ] Feature variant fixed.
[ ] Target scaling fixed.
[ ] Lookback fixed.
[ ] Pooling fixed.
[ ] ReLU registered.
[ ] GELU registered.
[ ] ReLU factory resolves correctly.
[ ] GELU factory resolves correctly.
[ ] Unknown activation rejected.
[ ] ReLU semantic unit tests PASS.
[ ] GELU implementation identity verified.
[ ] GELU kwargs/approximation unchanged.
[ ] Activation located in FFN only.
[ ] Activation after first FFN Linear.
[ ] FFN ordering matches frozen Transformer contract.
[ ] Same activation across every Encoder layer.
[ ] No mixed-layer activation.
[ ] Regression output remains linear.
[ ] No activation on input projection.
[ ] Synthetic forward finite.
[ ] Synthetic backward finite.
[ ] B1/B32/B64 output shapes valid.
[ ] Same parameter count.
[ ] Same parameter schema.
[ ] Same state_dict key set.
[ ] Same Train IDs.
[ ] Same Validation IDs.
[ ] Same WINDOWPOP-v1.
[ ] Same feature fingerprint.
[ ] Same X scaler.
[ ] Same target transform/scaler.
[ ] Same lookback.
[ ] Same pooling.
[ ] Same H1/WB0.
[ ] Same B64.
[ ] Same D64/H4/N2/FFN128.
[ ] Same dropout .1.
[ ] Same sinusoidal PE.
[ ] Same POST_NORM.
[ ] Same mask policy.
[ ] Same regression head.
[ ] Same AdamW/LR/WD/MSE.
[ ] Same E50/patience10/clip1.
[ ] Same seed42.
[ ] Same Training Engine.
[ ] Same Metric version.
[ ] GELU reference exact-match.
[ ] GELU not retrained.
[ ] ReLU registered before training.
[ ] ReLU fresh loaders/model.
[ ] ReLU no warm-start.
[ ] ReLU trained via TRAINING_ENGINE-v1.
[ ] ReLU BEST verified.
[ ] GELU BEST already verified.
[ ] Both metrics use verified BEST.
[ ] Full-precision RMSE used.
[ ] Activation effect computed.
[ ] MAE/R² effects computed.
[ ] Winner=min RMSE.
[ ] Exact tie=ReLU.
[ ] Metric divergence recorded if present.
[ ] Optimization diagnostics generated.
[ ] Parameter equality verified.
[ ] Initial-state match checked if available.
[ ] Optional activation diagnostics use same fixed batch.
[ ] Optional hooks absent during main training.
[ ] Hypothesis outcomes recorded.
[ ] Findings generated.
[ ] Inherited warnings propagated.
[ ] Winner artifact generated.
[ ] Phase 29 reference update generated.
[ ] B64 reuse identified.
[ ] No extra activation experiment.
[ ] No activation hyperparameter change.
[ ] No score-based rerun.
[ ] No failed/SANITY run in ranking.
[ ] No Test access.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] Interaction limitation documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 159. Acceptance criteria

Phase 28 chỉ PASS khi:

```text
S5-selected feature/target/lookback/pooling configuration is fixed.

Exactly ReLU and GELU are compared.

Activation is changed only inside the Encoder FFN.

Same activation is used in every Encoder layer.

Regression output remains linear.

No extra layer/dropout/projection is introduced.

Same X, y, sample IDs and population are used.

Same parameter count/schema is verified.

Training configuration and seed are identical.

GELU reference is exact-match and reused.

ReLU is one fresh seed-42 run.

No warm-start occurs.

Both BEST checkpoints are verified.

Validation RMSE Wh selects winner.

Exact RMSE tie selects ReLU.

Phase 29 reference is generated.

No hidden activation experiment/rerun occurs.

Test remains untouched.
```

---

# 160. Failure conditions

Phase 28 FAIL if:

```text
wrong S5 winner used

feature/target/lookback/pooling changes

activation applied in wrong location

different activations used across layers

GELU implementation mode changes

output activation added

unknown activation silently defaults

parameter count/schema differs

sample IDs/population differ

scaler/target transform differs

training config differs

GELU reference mismatches but is reused

GELU retrained and best rerun chosen

ReLU warm-started from GELU

ReLU run fails but GELU declared winner

RMSE rounded before ranking

MAE/R² overrides lower RMSE

score-based rerun occurs

Test used.
```

---

# 161. Common mistakes

## 161.1 Đổi activation cả regression head

Sai. S6 chỉ đổi FFN activation.

## 161.2 Một layer ReLU, một layer GELU

Không phải registered condition.

## 161.3 ReLU + extra dropout

Hidden factor.

## 161.4 GELU approximation thay đổi

Implementation drift.

## 161.5 Unknown activation tự fallback về GELU

Nguy hiểm. Phải fail config.

## 161.6 Retrain GELU rồi giữ run tốt hơn

Hidden rerun bias.

## 161.7 Warm-start ReLU từ GELU

Confounded.

## 161.8 Chọn GELU vì curve đẹp hơn dù RMSE cao hơn

Sai primary metric.

## 161.9 Chọn ReLU vì đơn giản dù RMSE kém hơn

Parsimony chỉ exact tie.

## 161.10 Thấy ReLU zero fraction cao rồi tự tăng width

Không Phase 28.

## 161.11 Thấy GELU tốt rồi tuyên bố smoothness là nguyên nhân

Unsupported causal claim.

## 161.12 Chạy SiLU thử thêm

Không thuộc S6.

## 161.13 Đổi LR riêng cho ReLU

Hai factors thay đổi.

## 161.14 Dùng Test xác nhận activation

Forbidden.

## 161.15 Chạy nhiều seed rồi chọn seed đẹp

Không Phase 28.

---

# 162. Recommended execution pseudocode

```text
load_phase27_signoff()
assert_approved_for_phase28()

s5 = load_s5_winner()

FV = s5.feature_variant_id
YS = s5.target_scaling_id
L  = s5.lookback_id
P  = s5.pooling_id

audit_activation_factory()
audit_relu_semantics()
audit_gelu_implementation_identity()
audit_activation_location_all_layers()

audit_same_data_population(FV, YS, L)
audit_same_pooling(P)
audit_same_parameter_schema()
audit_same_training_config()

gelu_reference = resolve_s5_winner_run()
assert_exact_s6_reference_match(gelu_reference)

register_relu_run()
seed(42)

loaders = build_fresh_loaders(
    feature_variant=FV,
    target_scaling=YS,
    lookback=L,
    population="WINDOWPOP-v1"
)

model = build_fresh_transformer(
    input_size=feature_count(FV),
    pooling=P,
    activation="RELU"
)

relu_result = TRAINING_ENGINE_v1.fit(...)
verify_best_checkpoint(relu_result)

results = {
    "RELU": relu_result,
    "GELU": gelu_reference
}

metrics = build_verified_s6_metrics(results)
effect = compute_activation_effect(metrics)
optimization = build_optimization_diagnostics(results)

optional_activation_diagnostics(
    same_fixed_batch=True,
    best_checkpoints=True
)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer="RELU"
)

write_hypothesis_outcomes()
write_findings()
write_s6_winner(winner)
write_phase29_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 163. Definition of Done

\[
\boxed{
One\ Fixed\ Transformer\ Setup
+
Two\ FFN\ Activations
+
One\ Fresh\ ReLU\ Run
+
One\ Valid\ Reused\ GELU
+
Same\ Population
+
Same\ Parameter\ Schema
+
Verified\ Activation\ Semantics
+
Verified\ BEST\ Metrics
+
S6\ Winner
+
Phase29\ Reference
+
No\ Test
}
\]

---

# 164. Final status contract

```text
PHASE 28 tests FFN ACTIVATION only.

Current feature variant:
from previous sweeps.

Target scaling:
from S3.

Lookback:
from S4.

Pooling:
from S5.

Conditions:

A0 = ReLU
A1 = GELU

Activation location:
inside every Encoder FFN,
after first FFN Linear,
according to frozen Transformer implementation.

No output activation.
No extra layer.
No extra dropout.
No mixed activations.

GELU:
reuse S5 winner if exact match.

ReLU:
one new fresh run.

Same:
X
y
sample IDs
WINDOWPOP-v1
feature/scaler pipeline
lookback
pooling
parameter schema/count
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
prefer ReLU.

No warm-start.
No hidden activation.
No hidden rerun.
No Test.

After SWEEP_S6_ACTIVATION-v1 PASS:
update current reference
and proceed to
PHASE 29 — S7 Batch-size sweep.
```

---

# 165. Final check

Correct workflow:

```text
Load S5 winner
→ Freeze FV* + YS* + L* + P*
→ Verify ReLU/GELU implementations
→ Verify activation location
→ Audit same data/model/parameters/training
→ Reuse GELU
→ Train one fresh ReLU
→ Verify BEST
→ Compare full-precision Validation RMSE
→ Analyze gradients/activation diagnostics cautiously
→ Select S6 winner
→ Update Phase 29 reference
```

Incorrect workflow:

```text
try ReLU/GELU/SiLU
→ change output activation
→ retrain GELU
→ change LR for ReLU
→ select nicest curve
→ inspect Test.
```

Chỉ sau khi `SWEEP_S6_ACTIVATION-v1` được sign-off mới chuyển sang **PHASE 29 — S7 Batch-size sweep**.
