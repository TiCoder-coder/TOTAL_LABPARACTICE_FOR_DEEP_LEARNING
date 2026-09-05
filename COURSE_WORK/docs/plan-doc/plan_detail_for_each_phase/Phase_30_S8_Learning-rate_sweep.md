# PHASE 30 — S8 LEARNING-RATE SWEEP

## Kế hoạch controlled sweep cho AdamW Learning Rate của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S7_BATCH-v1`  
**Sweep ID:** `S8_LEARNING_RATE`  
**Output version:** `SWEEP_S8_LEARNINGRATE-v1`  
**Phase trước:** `Phase_29_S7_Batch_sweep.md`

---

# 1. Vai trò của Phase 30

Phase 30 là controlled experiment thứ tám trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target scaling, lookback, pooling, activation, batch size, Transformer architecture, AdamW weight decay, loss, early stopping, gradient clipping, sample population và seed đã được khóa từ Phase 29, learning rate nào phù hợp hơn cho Transformer: `1e-4`, `3e-4` hay `1e-3`?

Phase 30 chỉ thay đúng một conceptual factor:

```text
LEARNING RATE
```

với ba condition:

```text
LR1 = 1e-4
LR2 = 3e-4
LR3 = 1e-3
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Factor
+
Same\ Initialization
+
Same\ Mini\text{-}Batches
+
Same\ Optimizer
+
Same\ Training\ Budget
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí Phase 30 trong master execution plan

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
→ S7 Batch sweep

Phase 30
→ S8 Learning-rate sweep

Phase 31
→ S9 Weight-decay sweep
```

Phase 30 không được quay lại thay:

```text
feature set
time features
target scaling
lookback
pooling
activation
batch size
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

---

# 3. Câu hỏi nghiên cứu của S8

Phase 30 phải trả lời:

```text
1. LR=1e-4, 3e-4 hay 1e-3 tạo Validation RMSE Wh thấp nhất?

2. LR thấp có làm learning quá chậm dưới epoch cap hiện tại không?

3. LR cao có gây oscillation, clipping burden hoặc numerical instability không?

4. Best epoch và early stopping behavior thay đổi thế nào?

5. Gradient norm/clipping fraction thay đổi theo trajectory ra sao?

6. Parameter-update magnitude có khác nhau như kỳ vọng không?

7. MAE và R² có cùng ranking với RMSE không?

8. Learning rate nào trở thành current reference cho Phase 31?
```

---

# 4. Current reference từ Phase 29

Phase 30 phải load:

```text
s7_batch_winner.json
s7_reference_update.json
phase_29_signoff.json
```

để resolve:

```text
FV* = selected feature variant
YS* = selected target scaling
L*  = selected lookback
P*  = selected pooling
A*  = selected activation
B*  = selected training batch size
```

Possible:

```text
B* ∈ {B32, B64}
```

Phase 30 không hard-code B64.

---

# 5. Canonical learning-rate options

Registry:

```text
LR1 = 1e-4
LR2 = 3e-4
LR3 = 1e-3
```

Use exact numeric values:

```text
0.0001
0.0003
0.001
```

Không dùng rounded display string làm scientific source of truth.

---

# 6. Reference learning rate

Current sequential reference từ Phase 29 vẫn dùng:

```text
LR2 = 3e-4
```

vì learning rate chưa được sweep trước Phase 30.

Nếu exact S8 contract match:

```text
REUSE S7 winner
```

as LR2 reference.

---

# 7. Scientific meaning của learning rate

Learning rate controls step scale in optimizer updates.

Conceptually:

\[
\theta_{t+1}
=
\theta_t
-
\eta \cdot Update_t
\]

where:

```text
eta = learning rate.
```

With AdamW, actual update also depends on:

```text
first moment
second moment
epsilon
weight decay
parameter state
gradient clipping
```

Therefore S8 measures:

> effect of the AdamW learning-rate value under the frozen AdamW and training configuration.

Không được mô tả S8 như pure scalar multiplication independent of optimizer state.

---

# 8. AdamW interaction caveat

`weight_decay` coefficient remains fixed at:

```text
1e-4
```

but AdamW decoupled decay update is scaled by learning rate in the optimizer step.

Therefore changing LR also changes the absolute magnitude of decay-per-step under fixed WD coefficient.

This is an inherent LR–WD interaction.

Do not compensate WD inside S8.

Phase 31 explicitly sweeps WD after LR selection.

---

# 9. Gradient clipping interaction caveat

Gradient clipping remains:

```text
max_norm = 1.0
```

for all LR conditions.

Clipping is applied:

```text
after backward
before optimizer.step
```

Changing LR does not directly alter the current batch's pre-clip gradient, but it changes parameter trajectory and therefore future gradients.

S8 must keep clipping fixed.

---

# 10. No scheduler

Hard:

```text
scheduler = None
```

for all LR conditions.

Thus:

```text
learning rate is constant across epochs
```

unless Training Engine has another explicitly frozen behavior, which would need to be reconciled before S8.

No:

```text
cosine decay
ReduceLROnPlateau
StepLR
OneCycle
warmup
```

in Phase 30.

---

# 11. No LR warmup

Do not add warmup only for LR3 because it is larger.

That would compare:

```text
constant LR1
constant LR2
warmup+LR3
```

which is not the registered sweep.

---

# 12. No adaptive fallback

Do not implement:

```text
if loss increases:
    lower LR
```

inside an S8 run.

Each condition must remain the exact registered constant LR for the full run.

---

# 13. Fixed epoch budget

All conditions use:

```text
max_epochs = 50
patience = 10
min_delta = 0
```

This is the same development budget.

Unlike S7, batch size is now frozen, so:

```text
optimizer steps per completed epoch
```

should be identical across LR conditions.

This makes S8 especially clean.

---

# 14. Same optimizer-step schedule per epoch

Because:

```text
same B*
same Train population
same drop_last=False
same sampler policy
same gradient accumulation=1
```

expected:

```text
steps_per_epoch_LR1
=
steps_per_epoch_LR2
=
steps_per_epoch_LR3.
```

Total steps can differ because early stopping may stop at different epochs.

---

# 15. Same mini-batch sequence is highly desirable

For the two NEW runs:

```text
LR1
LR3
```

use the same:

```text
Train sample permutation
batch grouping
worker policy
loader seed
```

for corresponding epochs.

This tightly isolates LR.

For historical LR2 reference, exact batch-order fingerprint should be verified if available.

If not available:

```text
NOT_VERIFIABLE
```

rather than fabricated.

---

# 16. Same initialization is highly desirable

All three LR conditions have identical model architecture and parameter shapes.

Use the same model initialization policy:

```text
seed = 42
```

and verify initial state fingerprints when available.

For new LR1 and LR3, initial state fingerprints should match exactly under same environment/model builder.

For LR2 reference:

```text
match if provenance available
else NOT_VERIFIABLE.
```

---

# 17. Working hypotheses

## H-S8-01 — Intermediate LR may provide best tradeoff

Current B0 uses:

```text
3e-4
```

A plausible hypothesis is:

```text
1e-4 may learn too slowly
1e-3 may update too aggressively
3e-4 may balance speed/stability
```

but this remains:

```text
UNTESTED.
```

No option receives preference from this hypothesis.

---

# 18. H-S8-02 — Lower LR may improve final Validation performance

If:

\[
RMSE(1e^{-4}) < RMSE(3e^{-4}), RMSE(1e^{-3})
\]

then lower step size performs best under the fixed epoch/early-stopping budget.

Do not automatically claim it generalizes better because of smoother optimization.

---

# 19. H-S8-03 — Higher LR may converge faster but not necessarily better

If LR3 reaches a low RMSE earlier but does not have the lowest BEST RMSE:

```text
convergence speed
```

and:

```text
best Validation performance
```

must be reported separately.

Winner remains BEST Validation RMSE.

---

# 20. Preconditions bắt buộc

Phase 30 chỉ bắt đầu khi:

```text
Phase 29 = PASS
```

or `PASS_WITH_WARNING` with no unresolved critical issue.

Required:

```text
approved_for_phase30 = true
```

and:

```text
s7_batch_winner.json
s7_reference_update.json
phase_29_signoff.json
```

---

# 21. Upstream contracts bắt buộc

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
SWEEP_S6_ACTIVATION-v1
SWEEP_S7_BATCH-v1
```

---

# 22. Carry-forward warnings

Any unresolved non-critical upstream warnings must be propagated.

Examples:

```text
RANDOM_CONTROL_GAIN
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
MATCHED_INITIALIZATION_NOT_VERIFIABLE
SAMPLE_ORDER_NOT_VERIFIABLE
```

Propagate into:

```text
S8 manifest
S8 summary
S8 report
S8 winner
Phase 31 handoff.
```

---

# 23. Swept factor duy nhất

Canonical field:

```text
learning_rate
```

Allowed values only:

```text
1e-4
3e-4
1e-3
```

Aliases:

```text
LR1
LR2
LR3
```

---

# 24. Frozen data configuration

Hard:

```text
feature_variant_id = FV*
feature order/fingerprint = same
X scaler = same

target_scaling_id = YS*
target transform/scaler = same

lookback_id = L*
horizon = H1
boundary = WB0
population = WINDOWPOP-v1

same Train sample IDs
same Validation sample IDs
```

---

# 25. Frozen representation configuration

Hard:

```text
pooling_id = P*
activation_id = A*
```

No change to S5/S6 winners.

---

# 26. Frozen batch configuration

Hard:

```text
train_batch_size = B*
gradient_accumulation_steps = 1
drop_last = False
```

Evaluation batching follows frozen evaluation contract.

---

# 27. Frozen Transformer architecture

```text
input_size = same F
d_model = 64
num_heads = 4
num_layers = 2
ffn_dim = 128
dropout = 0.1
activation = A*
pooling = P*
sinusoidal positional encoding
POST_NORM
no causal mask
no padding mask
regression head = Linear(64,1)
no output activation
```

---

# 28. Frozen optimizer identity

All conditions use exact same:

```text
optimizer = AdamW
```

and exact same AdamW non-LR fields from `TRAINING_ENGINE-v1`.

Audit exact values/settings for:

```text
betas
eps
weight_decay
amsgrad
maximize
foreach
capturable
differentiable
fused
```

as applicable to the current installed PyTorch/backend.

Do not assume defaults if run config already freezes them.

Only:

```text
learning_rate
```

may differ.

---

# 29. Frozen weight decay

Hard:

```text
weight_decay = 1e-4
```

for all LR conditions.

---

# 30. Frozen criterion

```text
MSE
```

No Huber yet.

---

# 31. Frozen clipping

```text
clip_grad_norm max_norm = 1.0
```

for all LR values.

---

# 32. Frozen training engine

```text
TRAINING_ENGINE-v1
```

No LR-specific custom branch.

---

# 33. Frozen random seed

```text
seed = 42
```

for every S8 condition.

Do not run LR1 with seed 42 and LR3 with another seed.

---

# 34. Same parameter count/schema

LR changes optimizer configuration only.

Hard expected:

```text
trainable_parameters_LR1
=
trainable_parameters_LR2
=
trainable_parameters_LR3
```

and same:

```text
parameter schema
state_dict keys
model architecture fingerprint.
```

---

# 35. Existing LR2 reference reuse rule

S7 winner already uses:

```text
LR2 = 3e-4.
```

If exact S8 contract match:

```text
REUSE S7 winner
```

as LR2 reference.

Do not retrain LR2.

---

# 36. Normal S8 run count

Expected:

```text
LR2 = 3e-4
→ reused S7 winner

LR1 = 1e-4
→ NEW run

LR3 = 1e-3
→ NEW run
```

Therefore normally:

```text
2 new training runs
+
1 reused reference.
```

---

# 37. LR2 reference reuse gate

Exact match required on:

```text
FV*
YS*
L*
P*
A*
B*
H1
WB0
WINDOWPOP-v1

D64/H4/N2/FFN128
dropout .1

AdamW
LR2=3e-4
WD1e-4
MSE
E50
patience10
clip1
scheduler=None
gradient_accumulation=1
seed42

Training Engine
Metric version
```

Any mismatch:

```text
STOP.
```

---

# 38. Fresh LR1 run

Execution:

```text
reseed 42
↓
fresh Train loader
↓
fresh Validation loader
↓
fresh Transformer
↓
AdamW(lr=1e-4, all other optimizer fields frozen)
↓
MSE
↓
TRAINING_ENGINE-v1
```

No warm-start.

---

# 39. Fresh LR3 run

Execution:

```text
reseed 42
↓
fresh Train loader
↓
fresh Validation loader
↓
fresh Transformer
↓
AdamW(lr=1e-3, all other optimizer fields frozen)
↓
MSE
↓
TRAINING_ENGINE-v1
```

No warm-start.

---

# 40. Run-order independence

Scientific outcome should not depend on whether notebook runs:

```text
LR1 then LR3
```

or:

```text
LR3 then LR1.
```

Therefore each condition must:

```text
reseed
recreate loaders
recreate model
recreate criterion
recreate optimizer
```

independently.

Do not reuse mutable objects across LR runs.

---

# 41. No optimizer-state reuse

Never load:

```text
AdamW state
moment estimates
step counters
```

from LR2 into LR1/LR3.

Every condition starts with fresh optimizer state.

---

# 42. No model warm-start

Never load reference model weights into new LR runs.

All conditions use fresh initialization policy.

---

# 43. No checkpoint continuation across LR values

Do not:

```text
train 3e-4 for 10 epochs
switch to 1e-4
continue
```

That is an LR schedule, not an S8 condition.

---

# 44. Constant LR verification

For every epoch/optimizer step:

```text
optimizer.param_groups[*]["lr"]
```

must remain equal to registered LR.

Recommended record:

```text
first_step_lr
last_step_lr
unique_lr_values_observed
```

Expected one unique value per condition.

---

# 45. Multiple parameter groups audit

Preferred current baseline:

```text
single optimizer parameter group
```

or a known frozen grouping.

If multiple parameter groups exist, Phase 30 must verify their LR relationships are frozen and change consistently according to the registered condition.

Do not change only one group unless baseline contract explicitly defines that.

---

# 46. LR registry precision

Store learning rate as numeric:

```json
{"learning_rate": 0.0003}
```

and optionally display label:

```text
LR2_3E-4
```

Do not compare string formatting.

---

# 47. Optimizer config fingerprint

Optimizer config fingerprint must differ only in LR-related field(s).

Expected:

```text
LR1 fingerprint
LR2 fingerprint
LR3 fingerprint
```

different, but config-delta audit must confirm all non-LR optimizer fields identical.

---

# 48. Same sample sequence audit

For NEW LR1/LR3 runs, recommended:

```text
epoch-order fingerprints identical
```

at corresponding epochs.

Because B* is fixed, batch boundaries should also match exactly when same sample permutation is used.

This provides a very clean paired training setup.

---

# 49. Historical LR2 sample-order caveat

If S7 winner did not persist epoch sample-order fingerprint:

```text
LR2_SAMPLE_ORDER_MATCH = NOT_VERIFIABLE
```

Do not retrain LR2 just to obtain it.

Reference reuse remains preferred.

---

# 50. Same initialization audit for new runs

Before optimizer creation or before any step:

```text
initial_state_fingerprint_LR1
==
initial_state_fingerprint_LR3
```

should be verified.

If LR2 reference initial fingerprint exists:

```text
compare all three.
```

---

# 51. Why same initialization matters

LR performance can be sensitive to random initialization.

Matched initialization reduces stochastic confounding.

Still, S8 remains a single-seed experiment and does not establish seed robustness.

---

# 52. First-update diagnostic

Optional but highly informative.

On a fixed **diagnostic-only** training batch and fresh disposable model/optimizer clones, measure:

```text
pre-update parameter fingerprint
gradient norm
post-clip gradient norm
parameter delta norm after one optimizer step
relative parameter delta norm
```

for LR1/LR2/LR3.

This should never alter official run objects.

---

# 53. First-update diagnostic must be disposable

Do not consume the official run's first batch/optimizer step for instrumentation experiments.

Use:

```text
disposable cloned setup
```

or record passive statistics already emitted by the Training Engine.

Official runs must start clean.

---

# 54. Expected first-update ordering

All else equal on same model state/batch, larger LR generally implies larger parameter update magnitude under AdamW.

But exact ratio need not equal LR ratio because:

```text
Adam normalization
epsilon
weight decay
clipping
parameter scale
```

interact.

Treat as sanity/diagnostic, not hard proportionality requirement.

---

# 55. Learning-rate ratio context

Registered ratios:

```text
LR2 / LR1 = 3
LR3 / LR2 ≈ 3.3333
LR3 / LR1 = 10
```

This provides a compact logarithmic-ish sweep around baseline.

No additional LR values in S8.

---

# 56. No LR finder

Do not run:

```text
LR range test
LR finder
automatic Bayesian optimization
Optuna
```

inside Phase 30.

Those would add unregistered model-selection steps.

---

# 57. No manual intermediate LR

Do not try:

```text
2e-4
5e-4
7e-4
```

after viewing results.

S8 has exactly three registered values.

---

# 58. No scheduler workaround for LR3

If LR3 oscillates:

```text
record behavior.
```

Do not add scheduler.

---

# 59. No extra epochs for LR1

If LR1 appears to still improve at epoch 50:

```text
record EPOCH_CAP_REACHED / possible under-training.
```

Do not extend only LR1.

Phase 38 later tests epoch cap.

---

# 60. Early stopping semantics fixed

Same:

```text
monitor = Validation RMSE Wh
patience = 10
min_delta = 0
```

for all LR conditions.

---

# 61. Best-checkpoint semantics fixed

BEST checkpoint:

```text
minimum Validation RMSE Wh
```

under Training Engine contract.

Do not choose:

```text
lowest training loss
latest epoch
lowest Validation model-space loss
```

instead.

---

# 62. Primary S8 selection metric

Hard:

```text
best_validation_rmse_wh
```

from verified BEST checkpoint.

---

# 63. Secondary metrics

Record:

```text
validation_mae_wh
validation_r2
best_epoch
epochs_completed
stop_reason
sample-weighted train loss
Validation RMSE trajectory
gradient diagnostics
clipping fraction
runtime
optimizer-step count
optional update diagnostics
```

---

# 64. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{1e-4},
RMSE_{3e-4},
RMSE_{1e-3}
\right)
\]

using full-precision Validation RMSE Wh.

---

# 65. Exact RMSE tie rule

If exact full-precision tie among candidate minima:

```text
prefer the lower learning rate.
```

Tie-order:

```text
1e-4
<
3e-4
<
1e-3
```

Rationale:

```text
more conservative optimizer step size
lower aggressiveness
predeclared parsimony/stability preference
```

This rule applies only to exact RMSE ties.

---

# 66. Do not choose LR2 merely because it is the existing reference

Reference reuse prevents duplicate stochastic runs.

It does not grant LR2 selection priority if another LR has lower RMSE.

---

# 67. No arbitrary practical-improvement threshold

Do not require:

```text
>1%
>2 Wh
```

improvement.

Strict lower RMSE wins unless exact tie.

---

# 68. Pairwise effects

Required comparisons:

```text
LR1 vs LR2
LR2 vs LR3
LR1 vs LR3
```

For left→right:

\[
\Delta RMSE_{left\rightarrow right}
=
RMSE_{left}-RMSE_{right}
\]

Positive means right condition improves.

---

# 69. Relative effect

\[
Improvement\%
=
100\times
\frac{
RMSE_{left}-RMSE_{right}
}{
RMSE_{left}
}
\]

Also compute:

```text
MAE delta
R² delta
```

for each pair.

---

# 70. Trend diagnostic

With three ordered LR values, classify descriptive pattern:

```text
LOWER_IS_BETTER
INTERMEDIATE_IS_BEST
HIGHER_IS_BETTER
MONOTONIC_IMPROVEMENT_WITH_LR
MONOTONIC_DEGRADATION_WITH_LR
U_SHAPED
INVERTED_U
MIXED
EXACT_TIE_PATTERN
```

This is descriptive, not a mathematical proof of the global LR optimum.

---

# 71. Do not fit a curve and infer an untested optimal LR

No polynomial/response-surface interpolation to claim:

```text
optimal LR = 4.7e-4
```

Only registered candidates can become S8 winner.

---

# 72. Metric ranking divergence

If RMSE and MAE rank LR values differently:

```text
METRIC_RANKING_DIVERGENCE
```

must be recorded.

Winner remains RMSE-based.

---

# 73. Numerical stability monitoring

For every condition monitor:

```text
loss finite
predictions finite
gradient norms finite
parameters finite
optimizer state finite where practical
```

No `nan_to_num`.

---

# 74. High-LR instability handling

If LR3 yields NaN/Inf:

```text
do not lower LR
do not skip batch
do not restart from a better checkpoint
```

First verify:

```text
data correct
implementation correct
same config
no infrastructure fault
```

If genuine numerical instability remains:

```text
LR3 run = FAILED_NUMERICAL
S8 = INCOMPLETE under strict core protocol
```

until a formal Protocol Amendment decides how unstable candidate handling should be represented.

Do not silently rank only LR1/LR2.

---

# 75. Why strict incomplete policy is preferred

Declaring LR3 “bad” after a failed implementation could accidentally hide a bug.

Requiring explicit resolution protects experimental integrity.

If the project later pre-registers:

```text
numerically unstable LR = dominated infeasible candidate
```

that rule must be documented before re-signoff.

---

# 76. Low-LR under-training handling

If LR1 reaches epoch 50 without early stopping and Validation RMSE still trends downward:

```text
record EPOCH_CAP_REACHED
record possible under-training under E50
```

but do not extend LR1.

This is scientifically meaningful under current budget.

---

# 77. Oscillation diagnostics

Recommended derive from Validation RMSE history:

```text
best_epoch
number of post-best worsening epochs
best_to_last_rmse_gap
epoch-to-epoch RMSE change summary
```

Do not invent a universal “oscillation score” unless formula is defined in artifact.

---

# 78. Gradient clipping diagnostics

Report:

```text
fraction_batches_clipped
mean preclip grad norm
max preclip grad norm
epochs with clipping
```

Raw clipped counts may be compared because batch size is fixed and steps/epoch equal, but fractions remain preferred.

---

# 79. Parameter-update diagnostics

If Training Engine supports low-overhead logging, optional per epoch:

```text
mean parameter update norm
relative update norm
```

But do not add expensive full-parameter copies every batch unless planned and benchmarked.

A disposable first-update audit is enough for core S8.

---

# 80. Training-loss comparability

All conditions use same:

```text
YS*
MSE
same samples
same B*
```

so sample-weighted train losses are numerically comparable descriptively.

Winner still uses Validation RMSE Wh.

---

# 81. Validation metric comparability

Hard:

```text
same ordered Validation sample IDs
same Wh metric space
same METRICS-v1
```

---

# 82. Same optimizer-step count per epoch audit

Create runtime assertion:

```text
num_train_batches
```

identical for LR1/LR2/LR3.

If not:

```text
data/loader drift.
```

---

# 83. Total optimizer steps

Record:

```text
steps_per_epoch
epochs_completed
total_optimizer_steps
steps_to_best_epoch
```

Since early stopping differs, total optimizer steps may differ.

Do not equalize after the fact.

---

# 84. Time-to-best diagnostics

Record:

```text
best_epoch
steps_to_best
time_to_best_seconds optional
```

These are efficiency diagnostics only.

---

# 85. Runtime diagnostics

Record:

```text
mean epoch duration
median epoch duration
total runtime
time to best
```

Because architecture/batch are identical, per-epoch runtime should be similar, but backend/system noise exists.

Runtime is not winner criterion.

---

# 86. No attention extraction during S8

Hard:

```text
need_weights=False
```

in training/evaluation path unless normal metric verification code requires no attention.

Attention collection OFF.

---

# 87. No AMP change

Mixed precision policy remains frozen.

Do not enable AMP only for LR3 to avoid instability/performance issues.

---

# 88. Same device preferred

Use same device/backend for all new runs and reference when possible.

If reference comes from different but valid device environment, metric comparison may remain usable under project protocol, but:

```text
runtime/update-bitwise comparisons
```

must be marked limited/non-comparable.

---

# 89. Experiment identity

```text
sweep_id = S8_LEARNING_RATE
experiment_family = TRANSFORMER_SWEEP_S8_LEARNING_RATE
sweep_version = SWEEP_S8_LEARNINGRATE-v1
```

Recommended labels:

```text
S8_<FV*>__<YS*>__<L*>__<P*>__<A*>__<B*>__LR1_1E-4__S42

S8_<FV*>__<YS*>__<L*>__<P*>__<A*>__<B*>__LR2_3E-4__REFERENCE_S7

S8_<FV*>__<YS*>__<L*>__<P*>__<A*>__<B*>__LR3_1E-3__S42
```

Registry `run_id` remains authoritative.

---

# 90. Run matrix

Create:

```text
s8_run_matrix.csv
```

Fields:

```text
sweep_id
lr_id
learning_rate
source_type
source_run_id
requires_new_training
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
train_batch_size
weight_decay
population_fingerprint
feature_fingerprint
x_scaler_id
target_transform_id
seed
model_config_id
optimizer_config_id
training_config_id
status
```

---

# 91. Sweep manifest

Create:

```text
s8_learning_rate_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S8_LEARNINGRATE-v1
sweep_id = S8_LEARNING_RATE
source_s7_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
candidate_learning_rates = [1e-4,3e-4,1e-3]
new_runs_required
reused_runs
swept_field = learning_rate
budget_semantics = FIXED_EPOCH_BUDGET
scheduler = NONE
warmup = NONE
gradient_accumulation = 1
frozen_fields
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = LOWER_LR_ON_EXACT_RMSE_TIE
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

# 92. Sweep contract

Create:

```text
s8_learning_rate_sweep_contract.json
```

Must state:

```text
Only AdamW learning rate changes.

LR1=1e-4.
LR2=3e-4.
LR3=1e-3.

Same current batch size B*.
Same sample population.
Same mini-batch policy.
Same model architecture.
Same initialization policy.
Same AdamW non-LR fields.
Same WD=1e-4.
Same MSE.
Same E50/patience10.
Same clipping=1.0.
No scheduler.
No warmup.
No LR fallback.
No extra epochs per candidate.

LR2 reference reused if exact match.
LR1/LR3 fresh seed-42 runs.

Validation RMSE Wh selects winner.
Exact tie → lower LR.
Test forbidden.
```

---

# 93. Preflight audit

Create:

```text
s8_learning_rate_preflight_audit.csv
```

Checks:

```text
phase29_pass
approved_for_phase30
s7_winner_valid
FV_locked
YS_locked
L_locked
P_locked
A_locked
B_locked
LR1_registered
LR2_registered
LR3_registered
population_fixed
optimizer_identity_fixed
WD_fixed
scheduler_none
warmup_none
clip_fixed
budget_fixed
Training_Engine_fixed
Metric_fixed
seed_fixed
test_locked
status
```

---

# 94. LR-definition audit

Create:

```text
s8_learning_rate_definition_audit.csv
```

Fields:

```text
lr_id
registered_value
runtime_optimizer_lr
constant_lr_verified
unique_runtime_lr_values
scheduler_present
warmup_present
status
```

---

# 95. Optimizer config-delta audit

Create:

```text
s8_optimizer_config_delta_audit.csv
```

Fields:

```text
lr_id
optimizer_name
learning_rate
betas
eps
weight_decay
amsgrad
maximize
foreach
capturable
differentiable
fused
non_lr_fields_match_reference
only_lr_differs
status
```

Use null/not-applicable where runtime version/backend does not expose a field.

---

# 96. Common-data audit

Create:

```text
s8_common_data_audit.csv
```

Fields:

```text
split_id
sample_count_lr1
sample_count_lr2
sample_count_lr3
sample_ids_equal
ordered_ids_equal
feature_fingerprint_equal
x_scaler_equal
target_transform_equal
lookback_equal
pooling_equal
activation_equal
batch_equal
population_fingerprint_equal
status
```

---

# 97. Architecture audit

Create:

```text
s8_learning_rate_architecture_audit.csv
```

Fields:

```text
lr_id
input_size
lookback
pooling
activation
d_model
num_heads
num_layers
ffn_dim
dropout
norm_policy
regression_head
trainable_parameters
parameter_schema_fingerprint
architecture_fingerprint
matches_reference
status
```

---

# 98. Training-config audit

Create:

```text
s8_learning_rate_training_audit.csv
```

Fields:

```text
lr_id
train_batch_size
optimizer
learning_rate
weight_decay
criterion
max_epochs
patience
min_delta
gradient_clip
scheduler
warmup
gradient_accumulation
mixed_precision
seed
training_engine_version
only_lr_differs
status
```

---

# 99. Initialization audit

Create:

```text
s8_initialization_audit.csv
```

Fields:

```text
lr_id
initial_model_state_fingerprint
reference_available
parameter_schema_match
fingerprint_matches_reference
fingerprint_matches_other_new_run
seed
status
```

---

# 100. Sample-order audit

Create:

```text
s8_sample_order_audit.csv
```

Fields:

```text
epoch_or_probe
lr1_order_fingerprint
lr2_order_fingerprint
lr3_order_fingerprint
lr2_reference_available
lr1_lr3_exact_match
all_three_match_if_verifiable
status
```

New-run LR1/LR3 exact match is strongly recommended.

---

# 101. Optimizer-step budget audit

Create:

```text
s8_optimizer_budget_audit.csv
```

Fields:

```text
lr_id
train_samples_per_epoch
train_batch_size
steps_per_epoch
epochs_completed
total_optimizer_steps
best_epoch
steps_to_best_epoch
same_steps_per_completed_epoch
status
```

---

# 102. First-update diagnostic artifact

Optional but recommended:

```text
s8_first_update_diagnostics.csv
```

Fields:

```text
lr_id
diagnostic_batch_fingerprint
initial_state_fingerprint
preclip_grad_norm
postclip_grad_norm
parameter_delta_norm
relative_parameter_delta_norm
optimizer_state_fresh
official_run_untouched
status
```

Use disposable clones.

---

# 103. Constant-LR runtime audit

Create:

```text
s8_lr_runtime_audit.csv
```

Fields:

```text
lr_id
first_observed_lr
last_observed_lr
min_observed_lr
max_observed_lr
unique_lr_count
constant_lr_verified
status
```

Expected:

```text
unique_lr_count = 1
```

for each condition.

---

# 104. Run provenance

Create:

```text
s8_learning_rate_run_provenance.csv
```

Fields:

```text
lr_id
learning_rate
run_id
source_type
source_phase
config_fingerprint
optimizer_config_fingerprint
parameter_schema_fingerprint
feature_fingerprint
population_fingerprint
initial_state_fingerprint
sample_order_provenance
best_checkpoint_sha256
history_sha256
metric_artifact
prediction_artifact
status
```

---

# 105. Primary metrics table

Create:

```text
s8_learning_rate_metrics.csv
```

Rows:

```text
LR1
LR2
LR3
```

Fields:

```text
lr_id
learning_rate
run_id
source_type
best_epoch
epochs_completed
steps_per_epoch
total_optimizer_steps
steps_to_best_epoch
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

# 106. Pairwise effects table

Create:

```text
s8_learning_rate_pairwise_effects.csv
```

Rows:

```text
LR1_TO_LR2
LR2_TO_LR3
LR1_TO_LR3
```

Fields:

```text
left_lr_id
right_lr_id
left_lr
right_lr
left_rmse_wh
right_rmse_wh
rmse_delta_wh
rmse_improvement_pct
mae_delta_wh
r2_delta
lr_ratio
status
```

---

# 107. LR trend diagnostics

Create:

```text
s8_learning_rate_trend_diagnostics.json
```

Fields:

```text
rmse_lr1
rmse_lr2
rmse_lr3
pattern
best_lr_id
best_lr
is_boundary_winner
interpretation
status
```

`is_boundary_winner=true` if winner is LR1 or LR3.

---

# 108. Boundary-winner caution

If winner is:

```text
LR1
```

or:

```text
LR3
```

it is at an edge of tested range.

Do not automatically expand search in S8.

Record:

```text
BOUNDARY_WINNER
```

as a finding/limitation.

Master plan proceeds sequentially unless a formal amendment adds a follow-up LR range experiment.

---

# 109. Why not automatically extend the LR range?

Seeing a boundary winner and then trying more values would be adaptive unregistered search.

For coursework transparency:

```text
record boundary result
proceed according to master plan
```

unless explicit protocol amendment is approved.

---

# 110. Optimization diagnostics

Create:

```text
s8_optimization_diagnostics.csv
```

Fields:

```text
lr_id
best_epoch
last_epoch
stop_reason
best_rmse_wh
last_rmse_wh
mean_grad_norm_preclip
max_grad_norm_preclip
mean_fraction_batches_clipped
max_fraction_batches_clipped
epochs_with_any_clipping
nonfinite_events
best_to_last_rmse_gap
status
```

---

# 111. Convergence diagnostics

Create:

```text
s8_convergence_diagnostics.csv
```

Fields:

```text
lr_id
first_epoch_rmse_wh
best_epoch
best_rmse_wh
last_epoch
last_rmse_wh
epoch_cap_reached
early_stopped
steps_to_best
time_to_best_seconds_optional
post_best_worsening_epochs
status
```

---

# 112. Runtime diagnostics

Create:

```text
s8_runtime_diagnostics.csv
```

Fields:

```text
lr_id
device
epochs_completed
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
time_to_best_seconds_optional
runtime_comparable
status
```

---

# 113. Hypothesis outcomes

Create:

```text
s8_hypothesis_outcomes.csv
```

Fields:

```text
hypothesis_id
comparison
expected_direction_or_pattern
observed_metrics
outcome
interpretation
status
```

Allowed:

```text
SUPPORTED
NOT_SUPPORTED
INCONCLUSIVE
```

No universal proof.

---

# 114. Findings artifact

Create:

```text
s8_learning_rate_findings.csv
```

Possible codes:

```text
LR1_GAIN
LR2_GAIN
LR3_GAIN
LR_EXACT_TIE
LOW_LR_UNDERTRAINING_SIGNAL
HIGH_LR_OSCILLATION_SIGNAL
HIGH_LR_NUMERICAL_INSTABILITY
INTERMEDIATE_LR_BEST
BOUNDARY_WINNER
METRIC_RANKING_DIVERGENCE
CLIPPING_DIFFERENCE
CONVERGENCE_DIFFERENCE
PARAMETER_UPDATE_SCALE_DIFFERENCE
CONSTANT_LR_VERIFIED
MATCHED_INITIALIZATION_VERIFIED
MATCHED_INITIALIZATION_NOT_VERIFIABLE
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
INHERITED_WARNING
```

---

# 115. Winner artifact

Create:

```text
s8_learning_rate_winner.json
```

Minimum:

```text
sweep_id
sweep_version
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
selection_metric
selection_direction
tie_rule
winner_lr_id
winner_learning_rate
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_lr_id
runner_up_learning_rate
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
winner_is_boundary
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 116. Reference update for Phase 31

Create:

```text
s8_reference_update.json
```

Minimum:

```text
previous_reference_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
previous_learning_rate = 3e-4
selected_lr_id
selected_learning_rate
winner_run_id
winner_config_fingerprint
winner_optimizer_config_fingerprint
winner_rmse_wh
weight_decay_state = WD1_1E-4
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase31
```

---

# 117. Phase 31 handoff logic

Phase 31 tests:

```text
WD0 = 0
WD1 = 1e-4
WD2 = 1e-3
```

while holding S8-selected LR fixed.

S8 winner already uses:

```text
WD1 = 1e-4
```

therefore normally:

```text
reuse S8 winner as WD1 reference

train new:
WD0
WD2
```

if exact match.

---

# 118. Learning curves

Recommended figures:

```text
S8_01_validation_rmse_by_epoch.png
S8_02_validation_mae_by_epoch.png
S8_03_train_loss_by_epoch.png
S8_04_gradient_norm_by_epoch.png
S8_05_gradient_clipping_fraction.png
S8_06_best_validation_metrics.png
S8_07_validation_rmse_by_optimizer_step.png
S8_08_convergence_summary.png
```

---

# 119. Primary figure

```text
S8_01_validation_rmse_by_epoch.png
```

Overlay:

```text
1e-4
3e-4
1e-3
```

with BEST epoch markers.

---

# 120. Optimizer-step figure

Because batch size is fixed:

```text
steps_per_epoch same
```

so epoch and optimizer-step axes are linearly related within completed epochs.

Still useful to show:

```text
S8_07_validation_rmse_by_optimizer_step.png
```

for consistency with optimization diagnostics.

No interpolated observed metrics.

---

# 121. Best-metric figure

```text
S8_06_best_validation_metrics.png
```

must derive from:

```text
s8_learning_rate_metrics.csv
```

not manually typed numbers.

---

# 122. Convergence-summary figure

```text
S8_08_convergence_summary.png
```

can show:

```text
best epoch
steps to best
best-to-last RMSE gap
```

Diagnostic only.

---

# 123. No logarithmic-axis trick that hides values

LR axis may be log-scaled for display because values span 10×.

If used:

```text
show exact candidate labels
```

and do not imply continuous search.

---

# 124. Interpretation if LR1 wins

Safe:

> Under the current selected batch/model configuration and E50/patience10 budget, LR=1e-4 achieved the lowest Validation RMSE.

If it also hits epoch cap while still improving:

```text
note potential budget interaction.
```

Do not extend run in S8.

---

# 125. Interpretation if LR2 wins

Safe:

> The existing baseline LR=3e-4 remained the best Validation choice among the three registered candidates under the frozen training protocol.

This validates retention of the reference LR for the next sweep.

---

# 126. Interpretation if LR3 wins

Safe:

> LR=1e-3 achieved the lowest Validation RMSE among the registered S8 candidates under the fixed clipping/AdamW configuration.

Because LR3 is boundary/high end:

```text
record boundary-winner limitation.
```

Do not automatically test 3e-3.

---

# 127. Interpretation if LR3 is unstable

Safe:

> LR=1e-3 exhibited numerical instability under the frozen optimizer, clipping and training configuration.

Do not claim:

```text
1e-3 is universally unstable.
```

---

# 128. Exact tie interpretation

If exact RMSE tie:

```text
lower LR wins
```

by predefined conservative tie rule.

Report exact tie explicitly.

---

# 129. Tiny non-zero margin

Strict lower RMSE wins.

Report:

```text
small single-seed Validation margin
```

and avoid strong claims.

---

# 130. Single-seed limitation

Mandatory:

```text
all S8 conditions use seed 42.
```

No mean±std.

---

# 131. Validation-only limitation

Mandatory:

```text
S8 winner is a development selection on Validation.
```

No Test evidence.

---

# 132. LR–WD interaction limitation

Mandatory report point:

```text
weight_decay coefficient is fixed,
but AdamW decay/update magnitude interacts with LR.
```

Phase 31 will test WD only after S8 chooses LR.

---

# 133. LR–clipping interaction limitation

Gradient clipping remains fixed at 1.0.

S8 winner is therefore conditional on current clipping policy.

Phase 39 later tests clipping ON/OFF.

---

# 134. LR–batch interaction limitation

Batch size was selected in S7 and then frozen.

S8 does not explore a full:

```text
batch × LR
```

grid.

Thus S8 winner is conditional on selected B*.

---

# 135. LR–epoch-budget interaction limitation

LR1 may need more epochs than LR3 to converge.

Because E50 is fixed, S8 compares learning rates under the current finite training budget.

This is intended.

Phase 38 later examines epoch cap.

---

# 136. Sequential-selection limitation

By Phase 30, Validation has influenced:

```text
S1 Feature set
S2 Time features
S3 Target scaling
S4 Lookback
S5 Pooling
S6 Activation
S7 Batch
S8 Learning rate
```

Complete experiment provenance is mandatory.

---

# 137. No Test access

Hard:

```text
Test loader not iterated
Test predictions absent
Test metrics absent.
```

---

# 138. No extra LR values

Hard:

```text
only 1e-4, 3e-4, 1e-3.
```

---

# 139. No LR schedule

Hard.

---

# 140. No warmup

Hard.

---

# 141. No LR finder

Hard.

---

# 142. No optimizer change

Do not compare:

```text
Adam
SGD
RMSprop
```

inside S8.

---

# 143. No beta/eps tuning

No.

---

# 144. No WD change

No.

---

# 145. No batch change

No.

---

# 146. No clipping change

No.

---

# 147. No dropout change

No.

---

# 148. No epoch extension per LR

No.

---

# 149. No attention extraction

No.

---

# 150. Run failure policy

If LR1 or LR3 suffers a technical failure unrelated to the candidate's actual optimization behavior:

```text
S8 incomplete
```

until technical rerun is resolved.

---

# 151. Technical rerun allowed

Only for documented:

```text
hardware failure
process interruption
corrupt checkpoint/artifact
software/infrastructure fault
```

using Experiment Registry rerun policy.

---

# 152. Score-based rerun forbidden

Do not rerun an LR because:

```text
RMSE looks unusually poor.
```

---

# 153. LR2 duplicate-run prohibition

Do not retrain LR2 to “make it fair” or obtain a nicer stochastic realization.

Reuse exact reference.

---

# 154. Numerical candidate failure is not a technical rerun excuse

If LR3 genuinely diverges under correct implementation:

```text
do not rerun repeatedly hoping for a stable score.
```

Record the outcome and follow the strict failure/protocol-amendment rule.

---

# 155. Discrepancy taxonomy

```text
S7_REFERENCE_MISSING
S7_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
POOLING_DRIFT
ACTIVATION_DRIFT
BATCH_DRIFT
LR_DEFINITION_MISMATCH
LR_RUNTIME_DRIFT
SCHEDULER_PRESENT
WARMUP_PRESENT
LR_FALLBACK_PRESENT
OPTIMIZER_IDENTITY_DRIFT
OPTIMIZER_NON_LR_FIELD_DRIFT
WEIGHT_DECAY_DRIFT
GRADIENT_CLIP_DRIFT
EPOCH_BUDGET_DRIFT
SAMPLE_ORDER_POLICY_DRIFT
INITIALIZATION_MISMATCH
PARAMETER_COUNT_MISMATCH
PARAMETER_SCHEMA_MISMATCH
POPULATION_MISMATCH
TARGET_ID_MISMATCH
X_SCALER_MISMATCH
TARGET_SCALER_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
REFERENCE_RUN_MISMATCH
RUN_FAILURE
NUMERICAL_INSTABILITY
CHECKPOINT_VERIFICATION_FAILURE
INCOMPLETE_SWEEP
RANKING_ERROR
PAIRWISE_EFFECT_ERROR
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 156. Discrepancy log

Create:

```text
s8_learning_rate_discrepancies.json
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

# 157. Severity examples

```text
CRITICAL:
Test access
scheduler/warmup secretly enabled
wrong population
wrong optimizer identity
candidate LR changes during run

MAJOR:
wrong LR value
WD drift
checkpoint verification failure
new-run initialization mismatch due pipeline bug

MODERATE:
metric ranking divergence
boundary winner
sample-order match not verifiable
large clipping difference

INFO:
reference initial fingerprint unavailable
minor runtime variation
```

---

# 158. Status model

## PASS

```text
all three LR conditions valid
same data/model/batch/optimizer settings
LR2 reference valid
LR1/LR3 verified
constant LR confirmed
winner selected
Phase 31 reference generated
Test untouched
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
boundary winner
metric ranking divergence
sample-order/reference init not verifiable
large clipping/oscillation diagnostic
inherited warning
```

provided all candidates are scientifically valid.

## FAIL

Examples:

```text
hidden scheduler
wrong LR
WD drift
population mismatch
unresolved LR run failure
genuine numerical failure under strict no-infeasible-candidate rule
Test access.
```

---

# 159. Sweep summary

Create:

```text
s8_learning_rate_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
reference_run_id
new_run_ids
reused_runs
candidate_lrs
primary_metric
metrics_by_lr
pairwise_effects
trend_pattern
optimizer_budget
optimization_diagnostics
convergence_diagnostics
constant_lr_audit
initialization_match
sample_order_match
winner
winner_margin
winner_is_boundary
inherited_warnings
phase31_reference
test_status
overall_status
```

---

# 160. Human-readable report

Create:

```text
s8_learning_rate_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S7
3. LR candidate definitions
4. AdamW LR semantics
5. Frozen optimizer/training contract
6. Same data/batch/initialization fairness
7. LR2 reference provenance
8. Constant-LR verification
9. Validation metrics
10. Pairwise LR effects
11. Learning-curve/convergence diagnostics
12. Gradient/clipping diagnostics
13. Optional first-update diagnostics
14. LR trend/boundary analysis
15. S8 winner
16. Interpretation cautions
17. Interaction limitations
18. Phase 31 handoff
```

---

# 161. Report wording

Use:

```text
Observed
Interpretation
Limitation
Handoff
```

Safe example:

> LR=1e-4 achieved the lowest Validation RMSE under the selected batch size, fixed AdamW weight decay, fixed clipping and E50/patience10 budget.

Avoid:

> LR=1e-4 is the globally optimal learning rate.

---

# 162. README

Create:

```text
README_S8_LEARNING_RATE_SWEEP.md
```

Must explain:

```text
Purpose
S7 winner handoff
LR1/LR2/LR3 definitions
constant LR semantics
no scheduler/warmup
same batch and optimizer
same initialization/order goals
LR2 reference reuse
fixed epoch budget
LR-WD interaction
LR-clipping interaction
numerical-instability policy
winner rule
tie rule
boundary-winner caveat
Phase 31 handoff
No Test.
```

---

# 163. Output directory

```text
artifacts/
└── sweeps/
    └── S8_learning_rate/
        ├── s8_learning_rate_sweep_manifest.json
        ├── s8_learning_rate_sweep_contract.json
        ├── s8_learning_rate_preflight_audit.csv
        ├── s8_run_matrix.csv
        ├── s8_learning_rate_definition_audit.csv
        ├── s8_optimizer_config_delta_audit.csv
        ├── s8_common_data_audit.csv
        ├── s8_learning_rate_architecture_audit.csv
        ├── s8_learning_rate_training_audit.csv
        ├── s8_initialization_audit.csv
        ├── s8_sample_order_audit.csv
        ├── s8_optimizer_budget_audit.csv
        ├── s8_first_update_diagnostics.csv
        ├── s8_lr_runtime_audit.csv
        ├── s8_learning_rate_run_provenance.csv
        ├── s8_learning_rate_metrics.csv
        ├── s8_learning_rate_pairwise_effects.csv
        ├── s8_learning_rate_trend_diagnostics.json
        ├── s8_optimization_diagnostics.csv
        ├── s8_convergence_diagnostics.csv
        ├── s8_runtime_diagnostics.csv
        ├── s8_hypothesis_outcomes.csv
        ├── s8_learning_rate_findings.csv
        ├── s8_learning_rate_winner.json
        ├── s8_reference_update.json
        ├── s8_learning_rate_sweep_tests.csv
        ├── s8_learning_rate_discrepancies.json
        ├── s8_learning_rate_sweep_summary.json
        ├── s8_learning_rate_sweep_report.md
        ├── figures/
        │   ├── S8_01_validation_rmse_by_epoch.png
        │   ├── S8_02_validation_mae_by_epoch.png
        │   ├── S8_03_train_loss_by_epoch.png
        │   ├── S8_04_gradient_norm_by_epoch.png
        │   ├── S8_05_gradient_clipping_fraction.png
        │   ├── S8_06_best_validation_metrics.png
        │   ├── S8_07_validation_rmse_by_optimizer_step.png
        │   └── S8_08_convergence_summary.png
        ├── README_S8_LEARNING_RATE_SWEEP.md
        └── phase_30_signoff.json
```

Optional:

```text
s8_first_update_diagnostics.csv
```

can be omitted if no safe disposable instrumentation exists, but omission must be documented.

New LR1/LR3 run artifacts remain:

```text
artifacts/runs/<run_id>/
```

No checkpoint duplication inside sweep folder.

---

# 164. Required outputs

```text
O30.1  Sweep manifest
O30.2  Sweep contract
O30.3  Preflight audit
O30.4  Run matrix
O30.5  LR-definition audit
O30.6  Optimizer config-delta audit
O30.7  Common-data audit
O30.8  Architecture audit
O30.9  Training-config audit
O30.10 Initialization audit
O30.11 Sample-order audit
O30.12 Optimizer-budget audit
O30.13 Optional first-update diagnostics
O30.14 Constant-LR runtime audit
O30.15 Run provenance
O30.16 Reused LR2 reference
O30.17 Verified LR1 run
O30.18 Verified LR3 run
O30.19 Primary metrics table
O30.20 Pairwise effects table
O30.21 LR trend diagnostics
O30.22 Optimization diagnostics
O30.23 Convergence diagnostics
O30.24 Runtime diagnostics
O30.25 Hypothesis outcomes
O30.26 Findings
O30.27 Winner artifact
O30.28 Phase 31 reference update
O30.29 Figures
O30.30 Sweep test suite
O30.31 Discrepancy log
O30.32 Sweep summary
O30.33 Human-readable report
O30.34 README
O30.35 Phase sign-off
```

---

# 165. Sweep test suite

Create:

```text
s8_learning_rate_sweep_tests.csv
```

Recommended checks:

```text
S8T30-001 Phase 29 PASS/non-critical warning only
S8T30-002 approved_for_phase30 = true
S8T30-003 S7 winner valid
S8T30-004 feature variant fixed
S8T30-005 target scaling fixed
S8T30-006 lookback fixed
S8T30-007 pooling fixed
S8T30-008 activation fixed
S8T30-009 batch fixed
S8T30-010 LR1 registered = 1e-4
S8T30-011 LR2 registered = 3e-4
S8T30-012 LR3 registered = 1e-3
S8T30-013 no extra LR candidate
S8T30-014 AdamW fixed
S8T30-015 AdamW non-LR fields identical
S8T30-016 WD1e-4 fixed
S8T30-017 MSE fixed
S8T30-018 E50 fixed
S8T30-019 patience10 fixed
S8T30-020 min_delta fixed
S8T30-021 clip1 fixed
S8T30-022 scheduler=None
S8T30-023 warmup=None
S8T30-024 no adaptive LR fallback
S8T30-025 accumulation=1
S8T30-026 drop_last=False
S8T30-027 mixed precision policy fixed
S8T30-028 seed42 fixed
S8T30-029 same Train IDs
S8T30-030 same Validation IDs
S8T30-031 same WINDOWPOP-v1
S8T30-032 same feature fingerprint
S8T30-033 same X scaler
S8T30-034 same target transform/scaler
S8T30-035 same lookback
S8T30-036 same pooling
S8T30-037 same activation
S8T30-038 same batch
S8T30-039 same H1/WB0
S8T30-040 same D64
S8T30-041 same H4
S8T30-042 same N2
S8T30-043 same FFN128
S8T30-044 same dropout
S8T30-045 same PE/norm/mask policy
S8T30-046 same regression head
S8T30-047 same parameter count
S8T30-048 same parameter schema
S8T30-049 same steps per completed epoch
S8T30-050 LR1 runtime LR constant
S8T30-051 LR2 runtime LR constant from provenance
S8T30-052 LR3 runtime LR constant
S8T30-053 each LR has exactly one observed constant LR value
S8T30-054 no scheduler introduced
S8T30-055 LR1/LR3 initialization fingerprints equal
S8T30-056 LR2 initialization match checked if available
S8T30-057 LR1/LR3 sample order exact-match
S8T30-058 LR2 sample-order match checked if available
S8T30-059 LR2 reference exact-match
S8T30-060 LR2 reference reused
S8T30-061 LR1 registered before training
S8T30-062 LR3 registered before training
S8T30-063 LR1 fresh loaders/model/optimizer
S8T30-064 LR3 fresh loaders/model/optimizer
S8T30-065 no model warm-start
S8T30-066 no optimizer-state reuse
S8T30-067 LR1 trained via TRAINING_ENGINE-v1
S8T30-068 LR3 trained via TRAINING_ENGINE-v1
S8T30-069 LR1 BEST verified
S8T30-070 LR3 BEST verified
S8T30-071 LR2 BEST already verified
S8T30-072 all result rows use verified BEST
S8T30-073 all predictions finite
S8T30-074 all official metrics in Wh
S8T30-075 full-precision RMSE available
S8T30-076 pair LR1→LR2 effect correct
S8T30-077 pair LR2→LR3 effect correct
S8T30-078 pair LR1→LR3 effect correct
S8T30-079 trend pattern generated
S8T30-080 winner=min RMSE
S8T30-081 exact tie→lower LR
S8T30-082 no reference-priority override
S8T30-083 metric divergence recorded if present
S8T30-084 boundary winner flagged if LR1/LR3
S8T30-085 no automatic range extension
S8T30-086 low-LR epoch-cap signal recorded if applicable
S8T30-087 high-LR instability handled strictly
S8T30-088 no batch skip on instability
S8T30-089 no LR reduction after instability
S8T30-090 clipping diagnostics generated
S8T30-091 convergence diagnostics generated
S8T30-092 optimizer-budget diagnostics generated
S8T30-093 runtime diagnostics generated
S8T30-094 optional first-update diagnostic uses disposable setup
S8T30-095 official runs untouched by optional diagnostics
S8T30-096 hypothesis outcomes generated
S8T30-097 findings generated
S8T30-098 winner points to valid run
S8T30-099 Phase 31 reference update generated
S8T30-100 WD1 reuse identified for Phase 31
S8T30-101 inherited warnings propagated
S8T30-102 no LR finder
S8T30-103 no unregistered intermediate LR
S8T30-104 no warmup
S8T30-105 no scheduler
S8T30-106 no WD compensation
S8T30-107 no extra epochs per LR
S8T30-108 no score-based rerun
S8T30-109 no failed/SANITY run in ranking
S8T30-110 no attention extraction
S8T30-111 no Test access
S8T30-112 single-seed limitation documented
S8T30-113 Validation-only limitation documented
S8T30-114 LR-WD interaction documented
S8T30-115 LR-batch interaction documented
S8T30-116 LR-clipping interaction documented
S8T30-117 LR-epoch-budget interaction documented
S8T30-118 figures source-derived
S8T30-119 summary/report generated
S8T30-120 phase sign-off generated
```

---

# 166. Recommended notebook structure

```text
Cell 30.1  Phase title
Cell 30.2  Verify Phase 29 sign-off
Cell 30.3  Declare SWEEP_S8_LEARNINGRATE-v1
Cell 30.4  Load S7 winner/reference update
Cell 30.5  Freeze FV*/YS*/L*/P*/A*/B*
Cell 30.6  Define LR1/LR2/LR3
Cell 30.7  Build S8 run matrix
Cell 30.8  Audit common data/population
Cell 30.9  Audit architecture/parameter equality
Cell 30.10 Audit exact AdamW non-LR config
Cell 30.11 Audit WD/loss/clip/budget
Cell 30.12 Assert scheduler/warmup absent
Cell 30.13 Audit initialization policy
Cell 30.14 Audit sample-order policy
Cell 30.15 Verify LR2 reference reuse eligibility
Cell 30.16 Register LR1 run
Cell 30.17 Reseed + fresh LR1 loaders/model/optimizer
Cell 30.18 Execute LR1 via Training Engine
Cell 30.19 Verify LR1 BEST
Cell 30.20 Register LR3 run
Cell 30.21 Reseed + fresh LR3 loaders/model/optimizer
Cell 30.22 Execute LR3 via Training Engine
Cell 30.23 Verify LR3 BEST
Cell 30.24 Verify constant runtime LR values
Cell 30.25 Optional disposable first-update diagnostic
Cell 30.26 Build run provenance
Cell 30.27 Build optimizer-budget audit
Cell 30.28 Build S8 metrics table
Cell 30.29 Compute pairwise effects
Cell 30.30 Classify LR trend/boundary status
Cell 30.31 Build optimization diagnostics
Cell 30.32 Build convergence diagnostics
Cell 30.33 Build runtime diagnostics
Cell 30.34 Evaluate hypotheses
Cell 30.35 Generate learning curves
Cell 30.36 Generate findings
Cell 30.37 Select S8 winner
Cell 30.38 Write winner JSON
Cell 30.39 Write Phase 31 reference update
Cell 30.40 Run S8 tests/discrepancies
Cell 30.41 Write summary/report
Cell 30.42 Register artifacts/checksums
Cell 30.43 Write README
Cell 30.44 Phase sign-off
```

---

# 167. Execution flow

```text
Verify Phase 29
        ↓
Load S7 winner
        ↓
Freeze FV* + YS* + L* + P* + A* + B*
        ↓
Declare LR1/LR2/LR3
        ↓
Audit same data/model/batch
        ↓
Audit AdamW non-LR fields
        ↓
Audit fixed WD/loss/clip/E50
        ↓
Assert no scheduler/warmup
        ↓
Verify LR2 exact reference
        ↓
Reuse LR2
        ↓
Register LR1
        ↓
Reseed + fresh objects
        ↓
Train + verify LR1 BEST
        ↓
Register LR3
        ↓
Reseed + fresh objects
        ↓
Train + verify LR3 BEST
        ↓
Verify constant LR values
        ↓
Build common Wh metrics
        ↓
Compute pairwise effects
        ↓
Analyze convergence/clipping
        ↓
Classify LR trend
        ↓
Select minimum-RMSE LR
        ↓
Apply lower-LR exact-tie rule
        ↓
Update Phase 31 reference
        ↓
Write S8 artifacts
        ↓
SWEEP_S8_LEARNINGRATE-v1 sign-off
```

---

# 168. Fail-fast order

Before expensive training:

```text
1. Phase 29 sign-off
2. S7 winner identity
3. FV*/YS*/L*/P*/A*/B* lock
4. LR definitions
5. common sample population
6. same architecture/parameter schema
7. exact AdamW identity/non-LR config
8. WD fixed
9. clipping fixed
10. scheduler absent
11. warmup absent
12. epoch budget fixed
13. initialization/order policy
14. LR2 reuse eligibility
15. Test firewall
16. Registry readiness
```

---

# 169. Why optimizer non-LR audit matters

A sweep can accidentally change:

```text
betas
eps
fused/foreach mode
weight decay
```

while only displaying a different LR.

Then outcome cannot be attributed cleanly to learning rate.

Audit optimizer configuration directly.

---

# 170. Why constant-LR audit matters

If a scheduler or hidden framework behavior modifies LR during training, labels like:

```text
LR3 = 1e-3
```

would no longer represent the actual training path.

Therefore runtime LR logging is a core S8 guard.

---

# 171. Why same batch order matters

With same model initialization but different mini-batch sequences, stochastic gradients differ for two reasons:

```text
LR
+
data order.
```

Matching order for new LR1/LR3 isolates LR more tightly.

Reference LR2 order may remain not verifiable without invalidating the sweep.

---

# 172. Why no extra epoch for LR1

A small LR often converges more slowly.

But giving it more epochs would change the scientific budget.

S8 asks which LR is best under the currently registered budget.

---

# 173. Why no warmup for LR3

Warmup can stabilize a high LR but creates a different optimization schedule.

If future work wants warmup:

```text
new experiment.
```

Not S8.

---

# 174. Why LR tie prefers lower value

Only exact tie.

It is a predefined conservative choice:

```text
smaller update scale
```

with no claim of universal superiority.

---

# 175. Winner verification checklist

Before writing `s8_learning_rate_winner.json`:

```text
[ ] LR1 valid completed run.
[ ] LR2 valid reused reference.
[ ] LR3 valid completed run.
[ ] Same FV*/YS*/L*/P*/A*/B*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same feature/scaler/target transform.
[ ] Same architecture/parameter schema.
[ ] Same AdamW non-LR fields.
[ ] Same WD/loss/clip.
[ ] Same E50/patience.
[ ] No scheduler/warmup.
[ ] Constant runtime LR verified.
[ ] Fresh optimizer state for new runs.
[ ] No warm-start.
[ ] Matched initialization checked.
[ ] Sample-order match checked.
[ ] All BEST checkpoints verified.
[ ] Full-precision RMSE used.
[ ] Pairwise effects computed.
[ ] Exact tie rule respected.
[ ] Boundary winner flagged if needed.
[ ] No Test.
```

---

# 176. Phase 31 handoff

Phase 31 receives:

```text
s8_learning_rate_winner.json
s8_reference_update.json
winner_run_id
winner_config_fingerprint
winner_optimizer_config_fingerprint
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
selected_lr_id
selected_learning_rate
population_fingerprint
```

and changes only:

```text
weight decay.
```

---

# 177. If LR1 wins

Current reference:

```text
FV*
YS*
L*
P*
A*
B*
LR=1e-4
WD=1e-4
```

Phase 31 sweeps:

```text
WD0
WD1
WD2
```

at LR1.

---

# 178. If LR2 wins

Current reference remains LR=3e-4 and Phase 31 proceeds there.

---

# 179. If LR3 wins

Current reference becomes LR=1e-3 and Phase 31 tests WD at LR3.

This is important because WD effect is conditional on selected LR.

---

# 180. Reference reuse for Phase 31

S8 winner uses:

```text
WD1=1e-4
```

so Phase 31 normally:

```text
reuse S8 winner as WD1 reference
train WD0 and WD2.
```

---

# 181. Relationship with Phase 32 dropout

LR choice can interact with dropout, but S8 freezes dropout=.1.

Sequential design records interaction limitation.

---

# 182. Relationship with Phase 37 loss

Different losses have different gradient scales.

S8 winner is therefore conditional on MSE.

Phase 37 later tests Huber after other factors.

---

# 183. Relationship with Phase 38 epoch cap

If LR1 shows under-training, retain that finding.

Phase 38 later examines E50 vs E100 on the then-current reference, not by reopening S8.

---

# 184. Relationship with Phase 39 clipping

If LR3 causes heavy clipping, retain diagnostics.

Phase 39 later tests clipping.

---

# 185. Relationship with Phase 42 candidate synthesis

All three LR conditions remain in Registry.

Losers are not deleted.

---

# 186. Relationship with Phase 44 rolling-origin

S8 winner is selected on one Validation period.

Temporal robustness remains untested until rolling-origin phase.

---

# 187. Relationship with Phase 46 multi-seed

S8 uses seed 42.

Small LR margins are not seed-robust evidence.

---

# 188. Reproducibility metadata

New LR1/LR3 runs record:

```text
run_id
seed
environment
device
feature fingerprint
X scaler checksum
target transform checksum
lookback
pooling
activation
batch size
learning rate
AdamW non-LR fields
weight decay
scheduler state
warmup state
population fingerprint
model config fingerprint
parameter schema fingerprint
initial state fingerprint
sample-order provenance
Training Engine fingerprint
best checkpoint checksum
history checksum
metric checksum
```

---

# 189. No fabricated outputs

Do not pre-fill:

```text
winner
RMSE
best epoch
steps to best
gradient norms
clipping fraction
runtime
first-update delta
trend pattern
```

before execution.

---

# 190. Phase sign-off

Create:

```text
phase_30_signoff.json
```

Minimum:

```text
phase = 30
phase_name = S8 Learning-rate sweep
sweep_version
sweep_id
source_s7_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
lr1_run_id
lr2_reference_run_id
lr3_run_id
new_run_ids
reused_run_ids
winner_lr_id
winner_learning_rate
winner_run_id
winner_rmse_wh
winner_is_boundary
population_fingerprint
metric_version
constant_lr_audit_status
optimizer_config_audit_status
fairness_audit_status
matched_initialization_status
sample_order_match_status
inherited_warnings
test_status
approved_for_phase31
overall_status
created_at
```

---

# 191. Acceptance checklist

```text
[ ] Phase 29 valid.
[ ] approved_for_phase30 = true.
[ ] SWEEP_S8_LEARNINGRATE-v1 declared.
[ ] S7 winner loaded.
[ ] FV* fixed.
[ ] YS* fixed.
[ ] L* fixed.
[ ] P* fixed.
[ ] A* fixed.
[ ] B* fixed.
[ ] LR1 exactly 1e-4.
[ ] LR2 exactly 3e-4.
[ ] LR3 exactly 1e-3.
[ ] No extra LR candidate.
[ ] Same Train IDs.
[ ] Same Validation IDs.
[ ] Same WINDOWPOP-v1.
[ ] Same feature fingerprint.
[ ] Same X scaler.
[ ] Same target transform/scaler.
[ ] Same lookback/pooling/activation/batch.
[ ] Same H1/WB0.
[ ] Same D64/H4/N2/FFN128.
[ ] Same dropout.
[ ] Same PE/norm/mask policy.
[ ] Same regression head.
[ ] Same trainable parameter count.
[ ] Same parameter schema.
[ ] AdamW exact identity fixed.
[ ] AdamW non-LR fields equal.
[ ] WD1e-4 fixed.
[ ] MSE fixed.
[ ] E50 fixed.
[ ] Patience10 fixed.
[ ] min_delta fixed.
[ ] Clip1 fixed.
[ ] Scheduler absent.
[ ] Warmup absent.
[ ] No adaptive LR fallback.
[ ] Accumulation=1.
[ ] drop_last=False.
[ ] Mixed precision policy fixed.
[ ] Seed42 fixed.
[ ] Same steps per completed epoch.
[ ] LR1/LR3 initial states match.
[ ] LR2 initial-state match checked if available.
[ ] LR1/LR3 sample orders match.
[ ] LR2 order match checked if available.
[ ] LR2 reference exact-match.
[ ] LR2 not retrained.
[ ] LR1 registered before training.
[ ] LR3 registered before training.
[ ] LR1 fresh loaders/model/optimizer.
[ ] LR3 fresh loaders/model/optimizer.
[ ] No optimizer-state reuse.
[ ] No model warm-start.
[ ] LR1 trained via TRAINING_ENGINE-v1.
[ ] LR3 trained via TRAINING_ENGINE-v1.
[ ] LR1 BEST verified.
[ ] LR3 BEST verified.
[ ] LR2 BEST already verified.
[ ] LR1 runtime LR constant.
[ ] LR2 constant LR provenance valid.
[ ] LR3 runtime LR constant.
[ ] All official predictions finite.
[ ] All official metrics computed in Wh.
[ ] Full-precision RMSE used.
[ ] LR1/LR2 effect computed.
[ ] LR2/LR3 effect computed.
[ ] LR1/LR3 effect computed.
[ ] Trend pattern generated.
[ ] Winner=min RMSE.
[ ] Exact tie=lower LR.
[ ] Existing-reference status does not override RMSE.
[ ] Metric divergence recorded if present.
[ ] Boundary winner recorded if applicable.
[ ] No automatic search-range extension.
[ ] Epoch-cap/under-training signal recorded if applicable.
[ ] High-LR instability handled strictly.
[ ] Gradient diagnostics generated.
[ ] Convergence diagnostics generated.
[ ] Optimizer-step diagnostics generated.
[ ] Runtime diagnostics generated.
[ ] Optional first-update diagnostic is disposable.
[ ] Official runs untouched by optional diagnostics.
[ ] Hypothesis outcomes recorded.
[ ] Findings generated.
[ ] Inherited warnings propagated.
[ ] Winner artifact generated.
[ ] Phase 31 reference update generated.
[ ] WD1 reuse identified.
[ ] No LR finder.
[ ] No intermediate LR.
[ ] No scheduler.
[ ] No warmup.
[ ] No WD compensation.
[ ] No extra epochs per LR.
[ ] No score-based rerun.
[ ] No failed/SANITY run in winner ranking.
[ ] No attention extraction.
[ ] No Test access.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] LR-WD interaction documented.
[ ] LR-batch interaction documented.
[ ] LR-clipping interaction documented.
[ ] LR-budget interaction documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 192. Acceptance criteria

Phase 30 chỉ PASS khi:

```text
S7-selected data/model/batch configuration is fixed.

Exactly LR1=1e-4, LR2=3e-4 and LR3=1e-3 are represented.

Only AdamW learning rate changes.

All AdamW non-LR fields are identical.

WD=1e-4, MSE, clipping=1.0 and E50/patience10 stay fixed.

No scheduler, warmup or adaptive LR behavior exists.

Same Train/Validation sample IDs are used.

Same model parameter schema is used.

Same initialization/sample-order policy is used and audited.

LR2 exact reference is reused.

LR1/LR3 are fresh seed-42 runs with fresh optimizer state.

Runtime LR is constant for each condition.

All BEST checkpoints are verified.

Validation RMSE Wh selects winner.

Exact RMSE tie selects lower LR.

Boundary winner is documented without hidden search extension.

Phase 31 reference is generated.

No hidden LR tuning/rerun occurs.

Test remains untouched.
```

---

# 193. Failure conditions

Phase 30 FAIL if:

```text
wrong S7 winner used

feature/target/lookback/pooling/activation/batch changes

LR values differ from registry

scheduler or warmup appears

LR changes during a run

AdamW non-LR fields differ

WD changes

clipping changes

epoch budget changes

sample population differs

model architecture/parameters differ

LR2 reference mismatches but is reused

LR2 retrained and best rerun chosen

LR1/LR3 warm-start

optimizer state reused

one candidate technically fails but winner declared anyway

genuine numerical instability silently ignored

unregistered LR tested and used

RMSE rounded before ranking

MAE/R² overrides lower RMSE

runtime/convergence speed overrides lower RMSE

score-based rerun occurs

Test used.
```

---

# 194. Common mistakes

## 194.1 Dùng cosine scheduler

Không còn là constant-LR sweep.

## 194.2 Warmup cho LR=1e-3

Hidden second factor.

## 194.3 Thấy LR1 chậm rồi chạy 100 epochs

Không. E100 belongs Phase 38.

## 194.4 Thấy LR3 unstable rồi giảm clipping threshold

Không.

## 194.5 Thấy LR3 unstable rồi tự giảm LR giữa run

Không.

## 194.6 Đổi WD theo LR

Không. Phase 31 xử lý WD.

## 194.7 Dùng LR finder trước rồi thêm candidate mới

Hidden adaptive search.

## 194.8 Retrain LR2 để có “same order”

Không. Reference reuse được ưu tiên; order có thể NOT_VERIFIABLE.

## 194.9 Warm-start LR1 từ LR2 BEST

Confounded.

## 194.10 Reuse AdamW moments

Confounded.

## 194.11 Chọn LR3 vì đạt best sớm hơn dù RMSE cao hơn

Sai primary metric.

## 194.12 Chọn LR1 vì “an toàn” dù RMSE cao hơn

Tie preference only applies to exact tie.

## 194.13 Fit curve rồi chọn 5e-4

Không phải registered candidate.

## 194.14 Boundary winner rồi tự test 3e-3

Không S8.

## 194.15 Mở Test để xác nhận LR

Forbidden.

---

# 195. Recommended execution pseudocode

```text
load_phase29_signoff()
assert_approved_for_phase30()

s7 = load_s7_winner()

FV = s7.feature_variant_id
YS = s7.target_scaling_id
L  = s7.lookback_id
P  = s7.pooling_id
A  = s7.activation_id
B  = s7.batch_id

LRs = {
    "LR1": 1e-4,
    "LR2": 3e-4,
    "LR3": 1e-3
}

audit_common_data_population()
audit_same_model_architecture()
audit_adamw_non_lr_fields()
assert_fixed_weight_decay(1e-4)
assert_scheduler_none()
assert_warmup_none()
assert_clip(1.0)
assert_fixed_budget(E50, patience10)

lr2_reference = resolve_s7_winner_run()
assert_exact_s8_reference_match(lr2_reference)

results = {
    "LR2": lr2_reference
}

for lr_id in ["LR1", "LR3"]:
    register_run(lr_id)

    reseed(42)
    loaders = build_fresh_loaders(
        feature_variant=FV,
        target_scaling=YS,
        lookback=L,
        train_batch_size=B,
        population="WINDOWPOP-v1"
    )

    model = build_fresh_transformer(
        input_size=feature_count(FV),
        pooling=P,
        activation=A
    )

    optimizer = build_fresh_adamw(
        model.parameters(),
        lr=LRs[lr_id],
        weight_decay=1e-4,
        other_fields=FROZEN_ADAMW_FIELDS
    )

    result = TRAINING_ENGINE_v1.fit(...)
    verify_constant_lr(result)
    verify_best_checkpoint(result)

    results[lr_id] = result

audit_new_run_initialization_match()
audit_new_run_sample_order_match()

metrics = build_verified_s8_metrics(results)
pairwise = compute_lr_pairwise_effects(metrics)
trend = classify_lr_trend(metrics)
optimization = build_optimization_diagnostics(results)
convergence = build_convergence_diagnostics(results)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer_lower_lr=True
)

write_hypothesis_outcomes()
write_findings()
write_s8_winner(winner)
write_phase31_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 196. Definition of Done

\[
\boxed{
One\ Fixed\ Transformer\ Setup
+
Three\ Constant\ Learning\ Rates
+
Two\ Fresh\ Runs
+
One\ Valid\ Reused\ LR2
+
Same\ Initialization/Batch\ Policy
+
Same\ AdamW\ Except\ LR
+
Verified\ Constant\ LR
+
Verified\ BEST\ Metrics
+
S8\ Winner
+
Phase31\ Reference
+
No\ Test
}
\]

---

# 197. Final status contract

```text
PHASE 30 tests LEARNING RATE only.

Current:
feature variant from previous sweeps
target scaling from S3
lookback from S4
pooling from S5
activation from S6
batch from S7.

Candidates:

LR1 = 1e-4
LR2 = 3e-4
LR3 = 1e-3

LR2:
reuse S7 winner if exact match.

LR1/LR3:
fresh seed-42 runs.

Same:
Train/Validation IDs
WINDOWPOP-v1
X/y pipeline
model architecture
parameter schema
batch B*
AdamW identity
AdamW betas/eps/etc
WD=1e-4
MSE
E50
patience10
clip1
accumulation1
Training Engine
Metric version.

No:
scheduler
warmup
LR finder
adaptive fallback
WD compensation
extra epochs
warm-start
optimizer-state reuse
hidden LR candidate.

Runtime LR:
must remain constant.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer lower LR.

Boundary winner:
record, do not auto-expand search.

High-LR instability:
do not silently fix/exclude.

No Test.

After SWEEP_S8_LEARNINGRATE-v1 PASS:
update current reference
and proceed to
PHASE 31 — S9 Weight-decay sweep.
```

---

# 198. Final check

Correct workflow:

```text
Load S7 winner
→ Freeze FV* + YS* + L* + P* + A* + B*
→ Define 1e-4 / 3e-4 / 1e-3
→ Audit same data/model/AdamW non-LR config
→ Assert no scheduler/warmup
→ Reuse LR2
→ Fresh LR1 from seed42
→ Verify BEST
→ Fresh LR3 from seed42
→ Verify BEST
→ Verify constant runtime LR
→ Compare full-precision Validation RMSE
→ Analyze convergence/clipping without overriding RMSE
→ Select S8 winner
→ Apply lower-LR exact-tie rule
→ Update Phase 31 reference
```

Incorrect workflow:

```text
run LR finder
→ add warmup to 1e-3
→ give 1e-4 more epochs
→ change WD with LR
→ retrain 3e-4
→ test extra LR after seeing results
→ inspect Test
```

Chỉ sau khi `SWEEP_S8_LEARNINGRATE-v1` được sign-off mới chuyển sang **PHASE 31 — S9 Weight-decay sweep**.
