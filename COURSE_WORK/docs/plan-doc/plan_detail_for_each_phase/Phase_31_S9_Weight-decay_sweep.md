# PHASE 31 — S9 WEIGHT-DECAY SWEEP

## Kế hoạch controlled sweep cho AdamW Weight Decay của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S8_LEARNINGRATE-v1`  
**Sweep ID:** `S9_WEIGHT_DECAY`  
**Output version:** `SWEEP_S9_WEIGHTDECAY-v1`  
**Phase trước:** `Phase_30_S8_Learning-rate_sweep.md`

---

# 1. Vai trò của Phase 31

Phase 31 là controlled experiment thứ chín trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target scaling, lookback, pooling, activation, batch size, learning rate, Transformer architecture, MSE loss, epoch budget, early stopping, gradient clipping, sample population và seed đã được khóa từ Phase 30, mức AdamW weight decay nào phù hợp hơn: `0`, `1e-4` hay `1e-3`?

Phase 31 chỉ thay đúng một conceptual factor:

```text
ADAMW WEIGHT DECAY
```

với ba condition:

```text
WD0 = 0
WD1 = 1e-4
WD2 = 1e-3
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Factor
+
Same\ Parameters
+
Same\ Initialization
+
Same\ Mini\text{-}Batches
+
Same\ AdamW\ Except\ WD
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí Phase 31 trong master execution plan

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

Phase 32
→ S10 Dropout sweep
```

Phase 31 không được quay lại thay:

```text
feature set
time features
target scaling
lookback
pooling
activation
batch size
learning rate
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

# 3. Câu hỏi nghiên cứu của S9

Phase 31 phải trả lời:

```text
1. WD=0, 1e-4 hay 1e-3 tạo Validation RMSE Wh thấp nhất?

2. Không dùng weight decay có dẫn đến dấu hiệu overfitting mạnh hơn không?

3. Weight decay lớn hơn có làm model underfit dưới current training budget không?

4. Parameter norms ở BEST checkpoint thay đổi như thế nào theo WD?

5. Learning curve và best-to-last RMSE gap thay đổi ra sao?

6. Gradient norm/clipping behavior có khác nhau không?

7. MAE và R² có cùng ranking với RMSE không?

8. WD nào trở thành current reference cho Phase 32?
```

---

# 4. Current reference từ Phase 30

Phase 31 phải load:

```text
s8_learning_rate_winner.json
s8_reference_update.json
phase_30_signoff.json
```

để resolve:

```text
FV* = selected feature variant
YS* = selected target scaling
L*  = selected lookback
P*  = selected pooling
A*  = selected activation
B*  = selected batch size
LR* = selected learning rate
```

Possible:

```text
LR* ∈ {1e-4, 3e-4, 1e-3}
```

Phase 31 không được hard-code `LR=3e-4`.

---

# 5. Canonical weight-decay options

Registry:

```text
WD0 = 0
WD1 = 1e-4
WD2 = 1e-3
```

Exact numeric values:

```text
0.0
0.0001
0.001
```

Không dùng display labels làm source of truth.

---

# 6. Reference weight decay

Current sequential reference từ Phase 30 vẫn dùng:

```text
WD1 = 1e-4
```

vì weight decay chưa được sweep trước Phase 31.

Nếu exact S9 contract match:

```text
REUSE S8 winner
```

as WD1 reference.

---

# 7. Weight decay trong Phase 31 là optimizer regularization

Phase 31 test:

```text
AdamW weight_decay coefficient
```

không test:

```text
L2 penalty manually added to MSE
```

Criterion remains exactly:

```text
MSE
```

No extra regularization term may be added to the loss.

---

# 8. AdamW decoupled-decay semantics

Conceptually, AdamW separates weight-decay action from the adaptive gradient update.

A simplified no-gradient decay component can be viewed as approximately:

\[
\theta
\leftarrow
(1-\eta\lambda)\theta
\]

where:

```text
eta = selected learning rate LR*
lambda = weight_decay
```

This formula is for regularization context, not a substitute for the actual optimizer implementation.

The official run must use the frozen AdamW implementation from `TRAINING_ENGINE-v1`.

---

# 9. LR–WD interaction is fundamental

Phase 30 already selected:

```text
LR*
```

and Phase 31 freezes it.

Because AdamW decay magnitude depends on the learning rate and decay coefficient, S9 answers:

> Which weight-decay coefficient performs best under the S8-selected learning rate and all other frozen settings?

Do not claim the winning WD is globally optimal for all learning rates.

---

# 10. No WD compensation for LR

Do not adjust WD based on selected LR using:

```text
lambda / lr
constant eta*lambda
constant cumulative decay
```

The registered candidates are the raw optimizer coefficients:

```text
0
1e-4
1e-3
```

under fixed `LR*`.

---

# 11. No LR adjustment for WD

Likewise, do not change LR for WD2 because regularization is stronger.

Learning rate is frozen from S8.

---

# 12. Parameter-group scope is a critical S9 contract

Before training, Phase 31 must inspect the exact AdamW parameter-group structure used by the S8 winner.

Need identify:

```text
number of optimizer parameter groups
group membership
group names/semantic scope if available
weight_decay applied to each group
learning rate applied to each group
```

Phase 31 must preserve the same parameter-group topology and membership.

---

# 13. Do not introduce a new no-decay exclusion policy

Common Transformer recipes sometimes exclude:

```text
bias
LayerNorm weights
```

from weight decay.

Phase 31 must **not introduce this policy now** unless it was already part of the frozen optimizer builder before S9.

If S8 used one AdamW group containing all trainable parameters:

```text
keep one group
```

and apply each registered WD to that same group.

If S8 already used multiple groups:

```text
preserve exact membership/topology
```

and apply the S9 coefficient according to the already frozen parameter-group semantics.

---

# 14. Why parameter-group stability matters

Changing from:

```text
decay all parameters
```

to:

```text
decay weight matrices only
```

would change:

```text
WD coefficient
+
regularization scope
```

simultaneously.

That is not a one-factor S9 experiment.

---

# 15. Canonical WD mapping policy

Create an explicit runtime mapping artifact before training.

Recommended:

```text
optimizer_group_policy_id
group_index
group_parameter_fingerprint
is_decay_eligible_under_frozen_policy
registered_sweep_coefficient
runtime_weight_decay
```

If single-group baseline:

```text
WD0 → group WD = 0
WD1 → group WD = 1e-4
WD2 → group WD = 1e-3
```

If a pre-existing no-decay group exists:

```text
its frozen zero-decay semantics remain zero
```

while the registered candidate changes only the pre-existing decay-enabled group coefficient.

No new group may be created.

---

# 16. Weight-decay parameter scope fingerprint

Create a stable fingerprint over:

```text
parameter names
group assignment
decay-eligible flag
```

Expected:

```text
same fingerprint across WD0/WD1/WD2.
```

This is one of the strongest fairness checks in S9.

---

# 17. Working hypotheses

## H-S9-01 — Moderate WD may improve Validation generalization

A plausible hypothesis:

```text
WD0 may permit more parameter growth/overfitting
WD2 may regularize too strongly
WD1 may offer a balance
```

but this remains:

```text
UNTESTED.
```

No selection prior is allowed.

---

# 18. H-S9-02 — No decay may be sufficient

If:

\[
RMSE(WD0)<RMSE(WD1), RMSE(WD2)
\]

then explicit AdamW decay did not improve current Validation performance under selected LR/batch/model settings.

Do not generalize to all Transformer training.

---

# 19. H-S9-03 — Stronger WD may help

If:

\[
RMSE(WD2)<RMSE(WD0), RMSE(WD1)
\]

the strongest registered coefficient performs best in this controlled range.

Because WD2 is the upper tested boundary:

```text
record boundary-winner caution.
```

Do not automatically test larger values.

---

# 20. Preconditions bắt buộc

Phase 31 chỉ bắt đầu khi:

```text
Phase 30 = PASS
```

or `PASS_WITH_WARNING` with no unresolved critical issue.

Required:

```text
approved_for_phase31 = true
```

and:

```text
s8_learning_rate_winner.json
s8_reference_update.json
phase_30_signoff.json
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
SWEEP_S8_LEARNINGRATE-v1
```

---

# 22. Carry-forward warnings

Any unresolved non-critical upstream warning must be propagated.

Examples:

```text
RANDOM_CONTROL_GAIN
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
BOUNDARY_WINNER
MATCHED_INITIALIZATION_NOT_VERIFIABLE
SAMPLE_ORDER_NOT_VERIFIABLE
```

Propagate into:

```text
S9 manifest
S9 summary
S9 report
S9 winner
Phase 32 handoff.
```

---

# 23. Swept factor duy nhất

Canonical field:

```text
weight_decay
```

Allowed values only:

```text
0
1e-4
1e-3
```

Aliases:

```text
WD0
WD1
WD2
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

same Train IDs
same Validation IDs
```

---

# 25. Frozen representation configuration

```text
pooling_id = P*
activation_id = A*
```

---

# 26. Frozen batch configuration

```text
train_batch_size = B*
gradient_accumulation_steps = 1
drop_last = False
```

Same DataLoader/sampler/worker policy.

---

# 27. Frozen learning rate

Hard:

```text
learning_rate = LR*
```

resolved from S8 winner.

No hard-coded fallback to 3e-4.

---

# 28. Frozen Transformer architecture

```text
input_size = same F
d_model = 64
num_heads = 4
num_layers = 2
ffn_dim = 128
dropout = 0.1
activation = A*
pooling = P*
sinusoidal PE
POST_NORM
no causal mask
no padding mask
regression head = Linear(64,1)
no output activation
```

---

# 29. Frozen optimizer identity

All conditions:

```text
optimizer = AdamW
```

Same non-WD fields:

```text
learning rate = LR*
betas
eps
amsgrad
maximize
foreach
capturable
differentiable
fused
parameter-group topology
parameter-group membership
```

as applicable under the installed environment and frozen optimizer builder.

Only the registered decay coefficient may change.

---

# 30. Frozen loss

```text
MSE
```

No explicit L2 term.

---

# 31. Frozen epoch budget

```text
max_epochs = 50
patience = 10
min_delta = 0
```

No extra epochs for WD0/WD2.

---

# 32. Frozen clipping

```text
gradient clipping ON
max_norm = 1.0
```

---

# 33. Frozen scheduler/warmup policy

```text
scheduler = None
warmup = None
```

---

# 34. Frozen seed

```text
seed = 42
```

for every condition.

---

# 35. Same sample sequence is highly desirable

Because:

```text
same B*
same Train population
same seed/loader policy
```

new WD0 and WD2 runs should use identical Train sample permutations and batch grouping at corresponding epochs.

For historical WD1 reference:

```text
verify if provenance exists
else NOT_VERIFIABLE.
```

Do not retrain WD1 only to recover missing order metadata.

---

# 36. Same initialization is highly desirable

All models have identical parameter shapes.

For new WD0/WD2:

```text
initial_state_fingerprint_WD0
==
initial_state_fingerprint_WD2
```

should be verified.

If WD1 reference stores initial fingerprint:

```text
compare all three.
```

Otherwise mark reference match:

```text
NOT_VERIFIABLE.
```

---

# 37. Same parameter count/schema

Weight decay is optimizer-side.

Hard expected:

```text
trainable parameter count equal
parameter names equal
parameter shapes equal
state_dict key set equal
architecture fingerprint equal.
```

---

# 38. Existing WD1 reference reuse

S8 winner is already trained with:

```text
WD1 = 1e-4
```

If exact S9 contract match:

```text
REUSE S8 winner.
```

Do not retrain WD1.

---

# 39. Normal S9 run count

Expected:

```text
WD1 = 1e-4
→ reused S8 winner

WD0 = 0
→ NEW run

WD2 = 1e-3
→ NEW run
```

Therefore normally:

```text
2 new training runs
+
1 reused reference.
```

---

# 40. WD1 reference reuse gate

Exact match required on:

```text
FV*
YS*
L*
P*
A*
B*
LR*
H1
WB0
WINDOWPOP-v1

D64/H4/N2/FFN128
dropout .1

AdamW
WD1=1e-4
same optimizer group topology
same optimizer group membership
MSE
E50
patience10
clip1
scheduler=None
accumulation1
seed42

Training Engine
Metric version
```

Mismatch:

```text
STOP.
```

---

# 41. Fresh WD0 run

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
fresh AdamW with:
    lr = LR*
    weight_decay = WD0 under frozen group policy
↓
MSE
↓
TRAINING_ENGINE-v1
```

No warm-start.

---

# 42. Fresh WD2 run

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
fresh AdamW with:
    lr = LR*
    weight_decay = WD2 under frozen group policy
↓
MSE
↓
TRAINING_ENGINE-v1
```

No warm-start.

---

# 43. No model-state reuse

Never initialize WD0/WD2 from:

```text
WD1 BEST
another WD condition
```

Fresh model initialization policy only.

---

# 44. No optimizer-state reuse

Never reuse:

```text
AdamW first moments
second moments
step counters
```

from another WD condition.

---

# 45. No mid-run WD switching

Do not:

```text
start with WD1
then switch to WD0 after epoch 10
```

That is a decay schedule, not S9.

---

# 46. Constant runtime WD verification

For each optimizer parameter group, weight-decay values must remain constant throughout the run under the registered policy.

Recommended runtime log:

```text
group_index
first_observed_wd
last_observed_wd
min_observed_wd
max_observed_wd
unique_wd_count
```

Expected:

```text
unique_wd_count = 1
```

per group.

---

# 47. Parameter-group membership must not mutate

Optimizer parameter groups must not add/remove parameters mid-run.

Record:

```text
group parameter fingerprint
```

before and after training setup.

Expected stable.

---

# 48. No automatic optimizer regrouping

Do not introduce code that dynamically separates parameters based on:

```text
name contains "bias"
ndim == 1
LayerNorm type
```

only for S9.

Use the frozen optimizer builder/group policy.

---

# 49. No explicit L2 regularization term

Criterion must remain:

\[
MSE(y,\hat y)
\]

not:

\[
MSE+\lambda\|\theta\|^2
\]

Adding L2 loss would confound:

```text
AdamW WD
+
loss regularization.
```

Hard fail.

---

# 50. No model norm constraint

Do not add:

```text
max-norm constraint
spectral norm
weight normalization
```

inside S9.

---

# 51. No early-stop change

Regularization can change convergence speed.

Still:

```text
patience=10
E50
```

for all candidates.

---

# 52. No dropout compensation

Current dropout:

```text
0.1
```

remains fixed.

Dropout sweep is Phase 32.

Do not lower dropout when WD2 is strong or raise it for WD0.

---

# 53. No clipping compensation

Clip stays 1.0.

---

# 54. No batch/LR compensation

Batch and LR are frozen winners.

---

# 55. Primary S9 selection metric

Hard:

```text
best_validation_rmse_wh
```

from verified BEST checkpoint.

---

# 56. Secondary metrics

Record:

```text
validation_mae_wh
validation_r2
best_epoch
epochs_completed
stop_reason
sample-weighted train loss
Validation RMSE trajectory
gradient norms
clipping fraction
parameter norm diagnostics
runtime
optimizer-step count
```

---

# 57. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{WD0},
RMSE_{WD1},
RMSE_{WD2}
\right)
\]

using full-precision Validation RMSE Wh.

---

# 58. Exact RMSE tie rule

If exact full-precision tie among candidate minima:

```text
prefer the lower weight decay.
```

Tie order:

```text
0
<
1e-4
<
1e-3
```

Rationale:

```text
less optimizer-side regularization
simpler training assumption
predeclared parsimony rule
```

Only exact tie invokes this rule.

---

# 59. No arbitrary practical threshold

Do not require:

```text
WD must improve >1%
```

Strict lower RMSE wins unless exact tie.

---

# 60. Pairwise effects

Required:

```text
WD0 vs WD1
WD1 vs WD2
WD0 vs WD2
```

For left→right:

\[
\Delta RMSE
=
RMSE_{left}-RMSE_{right}
\]

Positive means right condition improves.

---

# 61. Relative effect

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

---

# 62. Trend classification

With three ordered WD values, classify descriptive pattern:

```text
NO_DECAY_BEST
MODERATE_DECAY_BEST
STRONG_DECAY_BEST
MONOTONIC_GAIN_WITH_DECAY
MONOTONIC_DEGRADATION_WITH_DECAY
U_SHAPED
INVERTED_U
MIXED
EXACT_TIE_PATTERN
```

Descriptive only.

Do not interpolate an untested optimal WD.

---

# 63. No fitted response curve for new WD

Do not infer:

```text
optimal WD = 3.7e-4
```

from three points.

Only registered candidates can win S9.

---

# 64. Boundary-winner caution

If winner is:

```text
WD0
```

or:

```text
WD2
```

it is at a tested range boundary.

Record:

```text
BOUNDARY_WINNER
```

but do not automatically test:

```text
1e-2
1e-5
```

inside S9.

---

# 65. Metric ranking divergence

If RMSE/MAE rank candidates differently:

```text
METRIC_RANKING_DIVERGENCE
```

must be recorded.

Winner remains RMSE-based.

---

# 66. Best-checkpoint verification for WD0

After training:

```text
fresh Transformer
↓
strict-load WD0 BEST
↓
ordered full Validation
↓
inverse target transform if YS1
↓
METRICS-v1
↓
verify recorded BEST metrics.
```

---

# 67. Best-checkpoint verification for WD2

Same procedure.

---

# 68. WD1 reference verification

No retrain.

Verify:

```text
BEST checkpoint already verified
same sample IDs
same current configuration
same LR*
same optimizer group policy
same metric version
WD=1e-4.
```

---

# 69. Parameter norm diagnostics

Weight decay is expected to influence parameter magnitude, so Phase 31 should include post-checkpoint norm diagnostics.

Recommended checkpoints:

```text
initial state
BEST
LAST
```

where available.

Compute:

\[
\|\theta\|_2
=
\sqrt{\sum_i \theta_i^2}
\]

for trainable parameters.

---

# 70. Global parameter norm is diagnostic only

Do not choose winner by:

```text
smallest parameter norm
```

Regularization strength and parameter magnitude are not equivalent to forecasting quality.

Validation RMSE remains selection criterion.

---

# 71. Module-group parameter norms

Recommended optional breakdown:

```text
input_projection
attention_modules
FFN_linear_layers
LayerNorm parameters
regression_head
```

Only if stable parameter-name grouping exists.

Do not invent inconsistent grouping across candidates.

---

# 72. Decay-eligible norm diagnostics

If optimizer policy exposes a decay-eligible parameter subset, compute:

```text
L2 norm of decay-eligible parameters
```

separately from:

```text
non-decay parameters
```

where applicable.

This is particularly useful if the frozen optimizer already has multiple groups.

---

# 73. Parameter norm ratio to initialization

For condition c:

\[
NormRatio_c
=
\frac{
\|\theta_{BEST,c}\|_2
}{
\|\theta_{INIT,c}\|_2
}
\]

Diagnostic only.

---

# 74. Parameter delta from initialization

Optional:

\[
\|\theta_{BEST}-\theta_{INIT}\|_2
\]

and relative:

\[
\frac{
\|\theta_{BEST}-\theta_{INIT}\|_2
}{
\|\theta_{INIT}\|_2
}
\]

This captures total movement, not pure decay effect.

Interpret cautiously.

---

# 75. No assumption that higher WD always yields lower global norm

Adaptive gradients, early stopping, parameter interactions and trajectory differences can make observed global norms non-monotonic.

Do not make monotonicity a hard pass condition.

---

# 76. Theoretical no-gradient decay context

For a fixed LR and K optimizer steps, a simplified decay-only multiplier is:

\[
M(K)
=
(1-\eta\lambda)^K
\]

where:

```text
eta = LR*
lambda = WD
```

Optional context artifact may report:

```text
eta*lambda
M(steps_to_best)
M(total_steps)
```

for each candidate.

Label:

```text
THEORETICAL_DECAY_ONLY_CONTEXT
```

not observed parameter shrinkage.

---

# 77. Why theoretical decay exposure is useful

It clarifies that cumulative decay depends on:

```text
LR
WD
number of optimizer steps
```

and early stopping can give different total exposure.

It must not override observed Validation metrics.

---

# 78. Optimizer steps per completed epoch

Because batch is fixed:

```text
steps_per_epoch
```

should match across WD candidates.

Total optimizer steps can differ because:

```text
best epoch
early stopping epoch
```

can differ.

---

# 79. Decay exposure at BEST vs LAST

Recommended distinguish:

```text
steps_to_best
total_steps_to_stop
```

and theoretical decay-only multiplier at each.

Winner metrics come from BEST, not LAST.

---

# 80. Gradient diagnostics

Record:

```text
mean preclip grad norm
max preclip grad norm
fraction batches clipped
epochs with clipping
nonfinite events
```

Changing WD affects future parameter trajectory and thus can affect gradients indirectly.

---

# 81. Clipping interpretation

Weight decay in AdamW is optimizer-side and separate from gradient clipping.

Do not say:

```text
clipping directly clips the WD decay term
```

unless the frozen optimizer implementation explicitly does so, which is not the intended AdamW semantics.

For the experiment, simply preserve the existing Training Engine order and report observed behavior.

---

# 82. Training-loss comparability

All S9 candidates use:

```text
same target scaling
same MSE
same samples
same batch
```

so sample-weighted train loss is comparable descriptively.

Winner remains Validation RMSE Wh.

---

# 83. Generalization-gap diagnostics

Optional descriptive measure:

```text
best_train_loss_model_space
best_validation_metric
```

But raw train MSE and Validation RMSE may have different units/scales if target scaling is YS1.

Do not subtract them directly unless transformed into compatible metric space.

---

# 84. Wh-space train diagnostic optional

If the pipeline supports full Train predictions at BEST without contaminating selection:

```text
Train RMSE Wh
Validation RMSE Wh
```

can form:

\[
Gap_{RMSE}
=
ValidationRMSE - TrainRMSE
\]

Diagnostic only.

Do not add heavy full-Train inference if unnecessary.

---

# 85. No Test-based overfitting assessment

Overfitting diagnostics use:

```text
Train
Validation
```

only.

Test remains locked.

---

# 86. Runtime diagnostics

Record:

```text
epochs completed
mean epoch duration
median epoch duration
total runtime
time-to-best optional
```

WD itself should not change model shape, but early stopping and system noise may affect total runtime.

Runtime is not winner criterion.

---

# 87. Constant runtime WD audit

For each candidate and optimizer group, verify registered WD is constant through the run.

No scheduler-like WD schedule.

---

# 88. No weight-decay schedule

Do not implement:

```text
WD warmup
WD decay
cosine WD
adaptive WD
```

inside Phase 31.

---

# 89. No zero-decay special optimizer

WD0 still uses:

```text
AdamW
```

with:

```text
weight_decay=0
```

under the same optimizer builder.

Do not switch WD0 to:

```text
Adam
```

even if mathematically similar in some settings.

Optimizer identity must stay AdamW.

---

# 90. Why WD0 must still use AdamW

Switching optimizer class would change potential implementation details and violate one-factor isolation.

S9 factor is coefficient value, not optimizer family.

---

# 91. Same device preferred

New WD0/WD2 should run under same selected backend/environment as current project.

If historical WD1 reference was produced under another valid environment:

```text
runtime comparisons may be limited
```

but scientific metric reuse follows project provenance policy.

---

# 92. No attention extraction

Training/metric path:

```text
need_weights=False
```

Attention analysis remains off.

---

# 93. Experiment identity

```text
sweep_id = S9_WEIGHT_DECAY
experiment_family = TRANSFORMER_SWEEP_S9_WEIGHT_DECAY
sweep_version = SWEEP_S9_WEIGHTDECAY-v1
```

Recommended labels:

```text
S9_<FV*>__<YS*>__<L*>__<P*>__<A*>__<B*>__<LR*>__WD0_0__S42

S9_<FV*>__<YS*>__<L*>__<P*>__<A*>__<B*>__<LR*>__WD1_1E-4__REFERENCE_S8

S9_<FV*>__<YS*>__<L*>__<P*>__<A*>__<B*>__<LR*>__WD2_1E-3__S42
```

Registry `run_id` remains authoritative.

---

# 94. Run matrix

Create:

```text
s9_run_matrix.csv
```

Fields:

```text
sweep_id
wd_id
weight_decay
source_type
source_run_id
requires_new_training
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
optimizer_group_policy_id
optimizer_group_scope_fingerprint
population_fingerprint
feature_fingerprint
seed
model_config_id
optimizer_config_id
training_config_id
status
```

---

# 95. Sweep manifest

Create:

```text
s9_weight_decay_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S9_WEIGHTDECAY-v1
sweep_id = S9_WEIGHT_DECAY
source_s8_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
candidate_weight_decays = [0,1e-4,1e-3]
new_runs_required
reused_runs
swept_field = weight_decay
optimizer_group_policy_id
optimizer_group_scope_fingerprint
budget_semantics = FIXED_EPOCH_BUDGET
scheduler = NONE
warmup = NONE
gradient_accumulation = 1
frozen_fields
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = LOWER_WD_ON_EXACT_RMSE_TIE
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

# 96. Sweep contract

Create:

```text
s9_weight_decay_sweep_contract.json
```

Must state:

```text
Only AdamW weight_decay coefficient changes.

WD0=0.
WD1=1e-4.
WD2=1e-3.

Selected learning rate LR* is fixed.

AdamW optimizer identity fixed.
Non-WD optimizer fields fixed.
Parameter-group topology/membership fixed.
No new bias/LayerNorm exclusion rule.
No explicit L2 loss penalty.
No WD schedule.

Same sample population.
Same model architecture.
Same initialization/sample-order policy.
Same batch.
Same MSE.
Same E50/patience10.
Same clipping=1.

WD1 reference reused if exact match.
WD0/WD2 fresh seed-42 runs.

Validation RMSE Wh selects winner.
Exact tie → lower WD.
Test forbidden.
```

---

# 97. Preflight audit

Create:

```text
s9_weight_decay_preflight_audit.csv
```

Checks:

```text
phase30_pass
approved_for_phase31
s8_winner_valid
FV_locked
YS_locked
L_locked
P_locked
A_locked
B_locked
LR_locked
WD0_registered
WD1_registered
WD2_registered
population_fixed
optimizer_identity_fixed
optimizer_group_policy_loaded
optimizer_group_scope_fingerprint_valid
loss_has_no_explicit_l2
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

# 98. WD-definition audit

Create:

```text
s9_weight_decay_definition_audit.csv
```

Fields:

```text
wd_id
registered_value
runtime_weight_decay_values_by_group
constant_wd_verified
optimizer_class
explicit_l2_in_loss
wd_schedule_present
status
```

---

# 99. Optimizer parameter-group audit

Create:

```text
s9_optimizer_param_group_audit.csv
```

Fields:

```text
wd_id
group_index
group_parameter_count
group_trainable_numel
group_parameter_fingerprint
group_semantic_label_optional
is_decay_eligible_under_frozen_policy
runtime_learning_rate
runtime_weight_decay
membership_matches_reference
topology_matches_reference
status
```

This is a core S9 artifact.

---

# 100. Optimizer group-policy manifest

Create:

```text
s9_optimizer_group_policy.json
```

Fields:

```text
policy_id
source_run_id
source_training_engine_version
num_groups
group_descriptions
parameter_scope_fingerprint
new_exclusion_policy_introduced = false
group_topology_frozen = true
status
```

---

# 101. Common-data audit

Create:

```text
s9_common_data_audit.csv
```

Fields:

```text
split_id
sample_count_wd0
sample_count_wd1
sample_count_wd2
sample_ids_equal
ordered_ids_equal
feature_fingerprint_equal
x_scaler_equal
target_transform_equal
lookback_equal
pooling_equal
activation_equal
batch_equal
learning_rate_equal
population_fingerprint_equal
status
```

---

# 102. Architecture audit

Create:

```text
s9_weight_decay_architecture_audit.csv
```

Fields:

```text
wd_id
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

# 103. Optimizer config-delta audit

Create:

```text
s9_optimizer_config_delta_audit.csv
```

Fields:

```text
wd_id
optimizer_name
learning_rate
weight_decay
betas
eps
amsgrad
maximize
foreach
capturable
differentiable
fused
group_policy_id
group_scope_fingerprint
non_wd_fields_match_reference
only_wd_differs
status
```

---

# 104. Training-config audit

Create:

```text
s9_weight_decay_training_audit.csv
```

Fields:

```text
wd_id
train_batch_size
learning_rate
weight_decay
criterion
explicit_l2_penalty
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
only_wd_differs
status
```

Expected:

```text
explicit_l2_penalty = 0 / false.
```

---

# 105. Initialization audit

Create:

```text
s9_initialization_audit.csv
```

Fields:

```text
wd_id
initial_model_state_fingerprint
reference_available
parameter_schema_match
fingerprint_matches_reference
fingerprint_matches_other_new_run
seed
status
```

---

# 106. Sample-order audit

Create:

```text
s9_sample_order_audit.csv
```

Fields:

```text
epoch_or_probe
wd0_order_fingerprint
wd1_order_fingerprint
wd2_order_fingerprint
wd1_reference_available
wd0_wd2_exact_match
all_three_match_if_verifiable
status
```

---

# 107. Optimizer-budget audit

Create:

```text
s9_optimizer_budget_audit.csv
```

Fields:

```text
wd_id
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

# 108. Runtime WD audit

Create:

```text
s9_wd_runtime_audit.csv
```

Fields:

```text
wd_id
group_index
first_observed_wd
last_observed_wd
min_observed_wd
max_observed_wd
unique_wd_count
constant_wd_verified
group_membership_fingerprint
status
```

---

# 109. Parameter norm diagnostics

Create:

```text
s9_parameter_norm_diagnostics.csv
```

Fields:

```text
wd_id
checkpoint_stage
parameter_scope
parameter_count
l2_norm
l1_norm_optional
max_abs_parameter
nonfinite_count
initial_norm_if_available
norm_ratio_to_initial
status
```

Where:

```text
checkpoint_stage ∈ {INIT, BEST, LAST}
```

and `parameter_scope` can include:

```text
ALL_TRAINABLE
DECAY_ELIGIBLE
NON_DECAY
```

if frozen group policy provides such distinction.

---

# 110. Parameter movement diagnostics

Optional:

```text
s9_parameter_movement_diagnostics.csv
```

Fields:

```text
wd_id
checkpoint_stage
parameter_scope
delta_l2_from_init
relative_delta_l2_from_init
status
```

Do not make this required if initial state artifact is unavailable for WD1 reference.

---

# 111. Theoretical decay exposure context

Create:

```text
s9_decay_exposure_context.csv
```

Fields:

```text
wd_id
learning_rate
weight_decay
eta_times_lambda
steps_per_epoch
steps_to_best
total_optimizer_steps
decay_only_multiplier_to_best
decay_only_multiplier_to_stop
context_only
status
```

For WD0:

```text
decay_only_multiplier = 1.
```

Use stable numeric computation where necessary.

Label every row:

```text
context_only = true
```

---

# 112. Run provenance

Create:

```text
s9_weight_decay_run_provenance.csv
```

Fields:

```text
wd_id
weight_decay
run_id
source_type
source_phase
config_fingerprint
optimizer_config_fingerprint
optimizer_group_policy_id
optimizer_group_scope_fingerprint
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

# 113. Primary metrics table

Create:

```text
s9_weight_decay_metrics.csv
```

Rows:

```text
WD0
WD1
WD2
```

Fields:

```text
wd_id
weight_decay
run_id
source_type
learning_rate
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

# 114. Pairwise effects table

Create:

```text
s9_weight_decay_pairwise_effects.csv
```

Rows:

```text
WD0_TO_WD1
WD1_TO_WD2
WD0_TO_WD2
```

Fields:

```text
left_wd_id
right_wd_id
left_weight_decay
right_weight_decay
left_rmse_wh
right_rmse_wh
rmse_delta_wh
rmse_improvement_pct
mae_delta_wh
r2_delta
wd_ratio_if_defined
status
```

For ratio involving WD0:

```text
wd_ratio_if_defined = null
```

not infinity.

---

# 115. WD trend diagnostics

Create:

```text
s9_weight_decay_trend_diagnostics.json
```

Fields:

```text
rmse_wd0
rmse_wd1
rmse_wd2
pattern
best_wd_id
best_weight_decay
is_boundary_winner
interpretation
status
```

---

# 116. Optimization diagnostics

Create:

```text
s9_optimization_diagnostics.csv
```

Fields:

```text
wd_id
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

# 117. Convergence diagnostics

Create:

```text
s9_convergence_diagnostics.csv
```

Fields:

```text
wd_id
first_epoch_rmse_wh
best_epoch
best_rmse_wh
last_epoch
last_rmse_wh
early_stopped
epoch_cap_reached
steps_to_best
time_to_best_seconds_optional
post_best_worsening_epochs
status
```

---

# 118. Runtime diagnostics

Create:

```text
s9_runtime_diagnostics.csv
```

Fields:

```text
wd_id
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

# 119. Optional Train-vs-Validation generalization diagnostic

If already cheap/available:

```text
s9_generalization_diagnostics.csv
```

Fields:

```text
wd_id
train_rmse_wh_at_best
validation_rmse_wh_at_best
rmse_gap_wh
train_mae_wh_at_best
validation_mae_wh_at_best
mae_gap_wh
status
```

This artifact is optional.

Do not access Test.

---

# 120. Hypothesis outcomes

Create:

```text
s9_hypothesis_outcomes.csv
```

Fields:

```text
hypothesis_id
comparison_or_pattern
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

# 121. Findings artifact

Create:

```text
s9_weight_decay_findings.csv
```

Possible codes:

```text
WD0_GAIN
WD1_GAIN
WD2_GAIN
WD_EXACT_TIE
NO_DECAY_BEST
MODERATE_DECAY_BEST
STRONG_DECAY_BEST
BOUNDARY_WINNER
METRIC_RANKING_DIVERGENCE
PARAMETER_NORM_DIFFERENCE
DECAY_ELIGIBLE_NORM_DIFFERENCE
GENERALIZATION_GAP_DIFFERENCE
CONVERGENCE_DIFFERENCE
CLIPPING_DIFFERENCE
CONSTANT_WD_VERIFIED
GROUP_POLICY_VERIFIED
MATCHED_INITIALIZATION_VERIFIED
MATCHED_INITIALIZATION_NOT_VERIFIABLE
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
INHERITED_WARNING
```

---

# 122. Winner artifact

Create:

```text
s9_weight_decay_winner.json
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
learning_rate
selection_metric
selection_direction
tie_rule
winner_wd_id
winner_weight_decay
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_wd_id
runner_up_weight_decay
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
winner_is_boundary
optimizer_group_policy_id
optimizer_group_scope_fingerprint
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 123. Reference update for Phase 32

Create:

```text
s9_reference_update.json
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
learning_rate
previous_weight_decay = 1e-4
selected_wd_id
selected_weight_decay
winner_run_id
winner_config_fingerprint
winner_optimizer_config_fingerprint
winner_rmse_wh
dropout_state = DR01_0P1
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase32
```

---

# 124. Phase 32 handoff logic

Phase 32 tests:

```text
DR01 = 0.1
DR02 = 0.2
DR03 = 0.3
```

while holding S9-selected WD fixed.

S9 winner already uses:

```text
dropout = 0.1
```

therefore normally:

```text
reuse S9 winner as DR01 reference

train new:
DR02
DR03
```

if exact match.

---

# 125. Learning curves

Recommended figures:

```text
S9_01_validation_rmse_by_epoch.png
S9_02_validation_mae_by_epoch.png
S9_03_train_loss_by_epoch.png
S9_04_gradient_norm_by_epoch.png
S9_05_gradient_clipping_fraction.png
S9_06_best_validation_metrics.png
S9_07_parameter_norm_at_best.png
S9_08_decay_exposure_context.png
S9_09_convergence_summary.png
```

---

# 126. Primary figure

```text
S9_01_validation_rmse_by_epoch.png
```

Overlay:

```text
WD0
WD1
WD2
```

with BEST epoch markers.

---

# 127. Parameter norm figure

```text
S9_07_parameter_norm_at_best.png
```

should show:

```text
global trainable L2 norm
```

and optionally decay-eligible norm if optimizer group policy supports it.

Diagnostic only.

---

# 128. Decay exposure figure

```text
S9_08_decay_exposure_context.png
```

must be labeled:

```text
theoretical decay-only context
```

not observed parameter shrinkage.

---

# 129. Best metric figure

`S9_06_best_validation_metrics.png` derives from:

```text
s9_weight_decay_metrics.csv.
```

---

# 130. Interpretation if WD0 wins

Safe:

> Under the S8-selected learning rate and the frozen AdamW/training configuration, disabling weight decay achieved the lowest Validation RMSE among the registered candidates.

Do not claim regularization is generally unnecessary.

---

# 131. Interpretation if WD1 wins

Safe:

> The baseline coefficient `1e-4` produced the lowest Validation RMSE under the current selected configuration.

---

# 132. Interpretation if WD2 wins

Safe:

> The strongest registered coefficient `1e-3` achieved the lowest Validation RMSE under the frozen selected learning rate and training protocol.

Also report:

```text
boundary winner
```

and do not automatically expand range.

---

# 133. Interpretation of parameter norms

Safe:

> Higher/lower WD was associated with the observed checkpoint parameter norm differences under different training trajectories.

Avoid:

> The smaller norm caused better generalization.

S9 alone does not prove causality of norm magnitude.

---

# 134. Exact tie interpretation

If exact tie:

```text
lower WD wins
```

by predeclared parsimony rule.

---

# 135. Tiny non-zero margin

Strict lower RMSE wins.

Document:

```text
small single-seed Validation margin.
```

---

# 136. Single-seed limitation

Mandatory:

```text
all S9 conditions use seed 42.
```

No mean±std.

---

# 137. Validation-only limitation

Mandatory:

```text
S9 winner is a development selection on Validation.
```

No Test evidence.

---

# 138. WD–LR interaction limitation

Mandatory:

```text
S9 winner is conditional on LR* selected in S8.
```

No full LR×WD Cartesian grid is explored.

---

# 139. WD–dropout interaction limitation

Dropout remains:

```text
0.1
```

through S9.

Phase 32 tests dropout after WD selection.

No WD×dropout grid.

---

# 140. WD–epoch-budget interaction limitation

A stronger WD can alter convergence speed.

All candidates still receive:

```text
E50/patience10
```

so S9 result is budget-conditional.

---

# 141. WD–parameter-scope limitation

Results apply to the **frozen optimizer parameter-group policy**.

If a future implementation excludes LayerNorm/bias from decay, that is a different regularization setup and S9 result should not be assumed transferable.

---

# 142. Sequential-selection limitation

By Phase 31, Validation has influenced:

```text
S1 Feature set
S2 Time features
S3 Target scaling
S4 Lookback
S5 Pooling
S6 Activation
S7 Batch
S8 Learning rate
S9 Weight decay
```

Complete lineage and no hidden experiments are mandatory.

---

# 143. No Test access

Hard:

```text
Test loader not iterated
Test metrics absent
Test predictions absent.
```

---

# 144. No extra WD candidates

Do not add:

```text
1e-5
1e-2
5e-4
```

inside S9 after seeing results.

---

# 145. No WD schedule

No.

---

# 146. No explicit L2 loss

No.

---

# 147. No optimizer switch

All candidates use AdamW.

---

# 148. No parameter-group redesign

No.

---

# 149. No bias/LayerNorm exclusion introduced

No, unless it was already frozen upstream.

---

# 150. No dropout change

No.

---

# 151. No LR change

No.

---

# 152. No batch change

No.

---

# 153. No clipping change

No.

---

# 154. No extra epochs

No.

---

# 155. No attention extraction

No.

---

# 156. Run failure policy

If WD0 or WD2 has technical failure unrelated to scientific optimization:

```text
S9 incomplete
```

until technical rerun is resolved.

---

# 157. Technical rerun allowed

Only for documented:

```text
process interruption
hardware/software failure
corrupt checkpoint
artifact-write failure
```

under Experiment Registry rerun policy.

---

# 158. Score-based rerun forbidden

Do not rerun because:

```text
RMSE looks poor.
```

---

# 159. WD1 duplicate-run prohibition

Do not retrain WD1 to get another stochastic draw.

Reuse exact S8 winner.

---

# 160. Genuine optimization failure

If one WD candidate yields NaN/Inf under correct implementation:

```text
verify config/data/optimizer first.
```

If genuinely unstable:

```text
record numerical failure
S9 remains incomplete under strict core protocol
```

until candidate-failure handling is formally resolved.

Do not silently rank only remaining candidates.

---

# 161. Discrepancy taxonomy

```text
S8_REFERENCE_MISSING
S8_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
POOLING_DRIFT
ACTIVATION_DRIFT
BATCH_DRIFT
LEARNING_RATE_DRIFT
WD_DEFINITION_MISMATCH
WD_RUNTIME_DRIFT
WD_SCHEDULE_PRESENT
OPTIMIZER_IDENTITY_DRIFT
OPTIMIZER_NON_WD_FIELD_DRIFT
OPTIMIZER_GROUP_TOPOLOGY_DRIFT
OPTIMIZER_GROUP_MEMBERSHIP_DRIFT
PARAMETER_SCOPE_FINGERPRINT_MISMATCH
NEW_NO_DECAY_EXCLUSION_POLICY
EXPLICIT_L2_LOSS_PRESENT
DROPOUT_DRIFT
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

# 162. Discrepancy log

Create:

```text
s9_weight_decay_discrepancies.json
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

# 163. Severity examples

```text
CRITICAL:
Test access
explicit L2 term added
parameter-group membership changes
new bias/LayerNorm exclusion policy
wrong population
optimizer class changes

MAJOR:
wrong WD value
runtime WD drift
LR drift
checkpoint verification failure

MODERATE:
boundary winner
metric ranking divergence
reference init/sample order not verifiable
large parameter norm difference

INFO:
runtime variation
theoretical decay context only
```

---

# 164. Status model

## PASS

```text
all three WD conditions valid
same data/model/LR/batch
same AdamW parameter-group policy
WD1 reference valid
WD0/WD2 verified
constant WD confirmed
winner selected
Phase 32 reference generated
Test untouched
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
boundary winner
metric ranking divergence
initialization/order match not verifiable for reference
large parameter-norm difference
inherited warning
```

provided methodology remains valid.

## FAIL

Examples:

```text
parameter-group redesign
explicit L2 loss
wrong WD
LR drift
population mismatch
unresolved candidate failure
Test access.
```

---

# 165. Sweep summary

Create:

```text
s9_weight_decay_sweep_summary.json
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
learning_rate
reference_run_id
new_run_ids
reused_runs
candidate_wds
optimizer_group_policy
primary_metric
metrics_by_wd
pairwise_effects
trend_pattern
optimizer_budget
parameter_norm_diagnostics
decay_exposure_context
optimization_diagnostics
convergence_diagnostics
constant_wd_audit
initialization_match
sample_order_match
winner
winner_margin
winner_is_boundary
inherited_warnings
phase32_reference
test_status
overall_status
```

---

# 166. Human-readable report

Create:

```text
s9_weight_decay_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S8
3. WD candidate definitions
4. AdamW decoupled-decay semantics
5. Frozen learning-rate context
6. Optimizer parameter-group policy
7. Controlled-variable contract
8. Same data/initialization/order fairness
9. WD1 reference provenance
10. Constant-WD verification
11. Validation metrics
12. Pairwise WD effects
13. Learning-curve/convergence diagnostics
14. Parameter norm diagnostics
15. Theoretical decay-exposure context
16. Gradient/clipping diagnostics
17. WD trend/boundary analysis
18. S9 winner
19. Interpretation cautions
20. Interaction limitations
21. Phase 32 handoff
```

---

# 167. Report wording

Use:

```text
Observed
Interpretation
Limitation
Handoff
```

Safe example:

> WD=1e-4 achieved the lowest Validation RMSE under the S8-selected learning rate and the frozen AdamW parameter-group policy.

Avoid:

> WD=1e-4 is the globally optimal regularization strength.

---

# 168. README

Create:

```text
README_S9_WEIGHT_DECAY_SWEEP.md
```

Must explain:

```text
Purpose
S8 winner handoff
WD0/WD1/WD2 definitions
AdamW vs explicit L2 loss
selected LR remains fixed
parameter-group scope policy
no new LayerNorm/bias exclusion
same initialization/order goals
WD1 reference reuse
constant WD semantics
parameter-norm diagnostics
theoretical decay exposure
winner rule
tie rule
boundary-winner caution
WD-LR interaction
WD-dropout interaction
Phase 32 handoff
No Test.
```

---

# 169. Output directory

```text
artifacts/
└── sweeps/
    └── S9_weight_decay/
        ├── s9_weight_decay_sweep_manifest.json
        ├── s9_weight_decay_sweep_contract.json
        ├── s9_weight_decay_preflight_audit.csv
        ├── s9_run_matrix.csv
        ├── s9_weight_decay_definition_audit.csv
        ├── s9_optimizer_param_group_audit.csv
        ├── s9_optimizer_group_policy.json
        ├── s9_common_data_audit.csv
        ├── s9_weight_decay_architecture_audit.csv
        ├── s9_optimizer_config_delta_audit.csv
        ├── s9_weight_decay_training_audit.csv
        ├── s9_initialization_audit.csv
        ├── s9_sample_order_audit.csv
        ├── s9_optimizer_budget_audit.csv
        ├── s9_wd_runtime_audit.csv
        ├── s9_parameter_norm_diagnostics.csv
        ├── s9_parameter_movement_diagnostics.csv
        ├── s9_decay_exposure_context.csv
        ├── s9_weight_decay_run_provenance.csv
        ├── s9_weight_decay_metrics.csv
        ├── s9_weight_decay_pairwise_effects.csv
        ├── s9_weight_decay_trend_diagnostics.json
        ├── s9_optimization_diagnostics.csv
        ├── s9_convergence_diagnostics.csv
        ├── s9_runtime_diagnostics.csv
        ├── s9_generalization_diagnostics.csv
        ├── s9_hypothesis_outcomes.csv
        ├── s9_weight_decay_findings.csv
        ├── s9_weight_decay_winner.json
        ├── s9_reference_update.json
        ├── s9_weight_decay_sweep_tests.csv
        ├── s9_weight_decay_discrepancies.json
        ├── s9_weight_decay_sweep_summary.json
        ├── s9_weight_decay_sweep_report.md
        ├── figures/
        │   ├── S9_01_validation_rmse_by_epoch.png
        │   ├── S9_02_validation_mae_by_epoch.png
        │   ├── S9_03_train_loss_by_epoch.png
        │   ├── S9_04_gradient_norm_by_epoch.png
        │   ├── S9_05_gradient_clipping_fraction.png
        │   ├── S9_06_best_validation_metrics.png
        │   ├── S9_07_parameter_norm_at_best.png
        │   ├── S9_08_decay_exposure_context.png
        │   └── S9_09_convergence_summary.png
        ├── README_S9_WEIGHT_DECAY_SWEEP.md
        └── phase_31_signoff.json
```

Optional artifacts:

```text
s9_parameter_movement_diagnostics.csv
s9_generalization_diagnostics.csv
```

may be omitted when required upstream state/predictions are unavailable, but omission must be documented.

New WD0/WD2 run artifacts remain:

```text
artifacts/runs/<run_id>/
```

No checkpoint duplication in sweep folder.

---

# 170. Required outputs

```text
O31.1  Sweep manifest
O31.2  Sweep contract
O31.3  Preflight audit
O31.4  Run matrix
O31.5  WD-definition audit
O31.6  Optimizer parameter-group audit
O31.7  Optimizer group-policy manifest
O31.8  Common-data audit
O31.9  Architecture audit
O31.10 Optimizer config-delta audit
O31.11 Training-config audit
O31.12 Initialization audit
O31.13 Sample-order audit
O31.14 Optimizer-budget audit
O31.15 Constant-WD runtime audit
O31.16 Parameter norm diagnostics
O31.17 Optional parameter movement diagnostics
O31.18 Theoretical decay exposure context
O31.19 Run provenance
O31.20 Reused WD1 reference
O31.21 Verified WD0 run
O31.22 Verified WD2 run
O31.23 Primary metrics table
O31.24 Pairwise effects table
O31.25 WD trend diagnostics
O31.26 Optimization diagnostics
O31.27 Convergence diagnostics
O31.28 Runtime diagnostics
O31.29 Optional generalization diagnostics
O31.30 Hypothesis outcomes
O31.31 Findings
O31.32 Winner artifact
O31.33 Phase 32 reference update
O31.34 Figures
O31.35 Sweep test suite
O31.36 Discrepancy log
O31.37 Sweep summary
O31.38 Human-readable report
O31.39 README
O31.40 Phase sign-off
```

---

# 171. Sweep test suite

Create:

```text
s9_weight_decay_sweep_tests.csv
```

Recommended checks:

```text
S9T31-001 Phase 30 PASS/non-critical warning only
S9T31-002 approved_for_phase31 = true
S9T31-003 S8 winner valid
S9T31-004 feature variant fixed
S9T31-005 target scaling fixed
S9T31-006 lookback fixed
S9T31-007 pooling fixed
S9T31-008 activation fixed
S9T31-009 batch fixed
S9T31-010 selected LR fixed
S9T31-011 WD0 registered = 0
S9T31-012 WD1 registered = 1e-4
S9T31-013 WD2 registered = 1e-3
S9T31-014 no extra WD candidate
S9T31-015 AdamW fixed
S9T31-016 optimizer group topology loaded from reference
S9T31-017 optimizer group membership loaded from reference
S9T31-018 group scope fingerprint created
S9T31-019 group scope identical across candidates
S9T31-020 no new optimizer parameter group
S9T31-021 no parameter removed from existing group
S9T31-022 no parameter added to another group
S9T31-023 no new bias exclusion
S9T31-024 no new LayerNorm exclusion
S9T31-025 no new ndim-based exclusion
S9T31-026 explicit L2 loss penalty absent
S9T31-027 criterion remains MSE
S9T31-028 selected LR identical across candidates
S9T31-029 AdamW non-WD fields identical
S9T31-030 E50 fixed
S9T31-031 patience10 fixed
S9T31-032 min_delta fixed
S9T31-033 clip1 fixed
S9T31-034 scheduler=None
S9T31-035 warmup=None
S9T31-036 accumulation=1
S9T31-037 drop_last=False
S9T31-038 mixed precision policy fixed
S9T31-039 seed42 fixed
S9T31-040 same Train IDs
S9T31-041 same Validation IDs
S9T31-042 same WINDOWPOP-v1
S9T31-043 same feature fingerprint
S9T31-044 same X scaler
S9T31-045 same target transform/scaler
S9T31-046 same lookback
S9T31-047 same pooling
S9T31-048 same activation
S9T31-049 same batch
S9T31-050 same H1/WB0
S9T31-051 same D64
S9T31-052 same H4
S9T31-053 same N2
S9T31-054 same FFN128
S9T31-055 same dropout=.1
S9T31-056 same PE/norm/mask policy
S9T31-057 same regression head
S9T31-058 same parameter count
S9T31-059 same parameter schema
S9T31-060 same state_dict key set
S9T31-061 same steps per completed epoch
S9T31-062 WD0 runtime WD constant
S9T31-063 WD1 constant WD provenance valid
S9T31-064 WD2 runtime WD constant
S9T31-065 runtime WD matches group policy
S9T31-066 no WD schedule
S9T31-067 WD0/WD2 initialization fingerprints equal
S9T31-068 WD1 initialization match checked if available
S9T31-069 WD0/WD2 sample order exact-match
S9T31-070 WD1 sample-order match checked if available
S9T31-071 WD1 reference exact-match
S9T31-072 WD1 reference reused
S9T31-073 WD0 registered before training
S9T31-074 WD2 registered before training
S9T31-075 WD0 fresh loaders/model/optimizer
S9T31-076 WD2 fresh loaders/model/optimizer
S9T31-077 no model warm-start
S9T31-078 no optimizer-state reuse
S9T31-079 WD0 uses AdamW, not Adam
S9T31-080 WD0 trained via TRAINING_ENGINE-v1
S9T31-081 WD2 trained via TRAINING_ENGINE-v1
S9T31-082 WD0 BEST verified
S9T31-083 WD2 BEST verified
S9T31-084 WD1 BEST already verified
S9T31-085 all result rows use verified BEST
S9T31-086 all predictions finite
S9T31-087 all official metrics in Wh
S9T31-088 full-precision RMSE available
S9T31-089 WD0→WD1 effect correct
S9T31-090 WD1→WD2 effect correct
S9T31-091 WD0→WD2 effect correct
S9T31-092 WD ratio null when denominator zero
S9T31-093 trend pattern generated
S9T31-094 winner=min RMSE
S9T31-095 exact tie→lower WD
S9T31-096 no reference-priority override
S9T31-097 metric divergence recorded if present
S9T31-098 boundary winner flagged if WD0/WD2
S9T31-099 no automatic WD-range extension
S9T31-100 parameter norm diagnostics generated
S9T31-101 parameter norms are diagnostic only
S9T31-102 decay-eligible scope norm computed if supported
S9T31-103 theoretical decay exposure labeled context only
S9T31-104 decay multiplier uses selected LR
S9T31-105 optimizer-budget diagnostics generated
S9T31-106 optimization diagnostics generated
S9T31-107 convergence diagnostics generated
S9T31-108 runtime diagnostics generated
S9T31-109 optional generalization diagnostics use Train/Validation only
S9T31-110 no Test used for overfitting assessment
S9T31-111 hypothesis outcomes generated
S9T31-112 findings generated
S9T31-113 winner points to valid run
S9T31-114 Phase 32 reference update generated
S9T31-115 DR01 reuse identified for Phase 32
S9T31-116 inherited warnings propagated
S9T31-117 no explicit L2
S9T31-118 no optimizer switch
S9T31-119 no parameter-group redesign
S9T31-120 no dropout compensation
S9T31-121 no LR compensation
S9T31-122 no extra epochs
S9T31-123 no score-based rerun
S9T31-124 no failed/SANITY run in winner ranking
S9T31-125 no attention extraction
S9T31-126 no Test access
S9T31-127 single-seed limitation documented
S9T31-128 Validation-only limitation documented
S9T31-129 WD-LR interaction documented
S9T31-130 WD-dropout interaction documented
S9T31-131 WD-budget interaction documented
S9T31-132 optimizer-scope limitation documented
S9T31-133 figures source-derived
S9T31-134 summary/report generated
S9T31-135 phase sign-off generated
```

---

# 172. Recommended notebook structure

```text
Cell 31.1  Phase title
Cell 31.2  Verify Phase 30 sign-off
Cell 31.3  Declare SWEEP_S9_WEIGHTDECAY-v1
Cell 31.4  Load S8 winner/reference update
Cell 31.5  Freeze FV*/YS*/L*/P*/A*/B*/LR*
Cell 31.6  Define WD0/WD1/WD2
Cell 31.7  Inspect S8 optimizer parameter groups
Cell 31.8  Freeze optimizer group-policy manifest
Cell 31.9  Build S9 run matrix
Cell 31.10 Audit common data/population
Cell 31.11 Audit architecture/parameter equality
Cell 31.12 Audit AdamW non-WD config
Cell 31.13 Audit no explicit L2 penalty
Cell 31.14 Audit fixed LR/dropout/clip/E50
Cell 31.15 Audit initialization policy
Cell 31.16 Audit sample-order policy
Cell 31.17 Verify WD1 reference reuse eligibility
Cell 31.18 Register WD0 run
Cell 31.19 Reseed + fresh WD0 loaders/model/optimizer
Cell 31.20 Execute WD0 via Training Engine
Cell 31.21 Verify WD0 BEST
Cell 31.22 Register WD2 run
Cell 31.23 Reseed + fresh WD2 loaders/model/optimizer
Cell 31.24 Execute WD2 via Training Engine
Cell 31.25 Verify WD2 BEST
Cell 31.26 Verify constant runtime WD/group membership
Cell 31.27 Build run provenance
Cell 31.28 Build optimizer-budget audit
Cell 31.29 Build parameter norm diagnostics
Cell 31.30 Build theoretical decay-exposure context
Cell 31.31 Build S9 metrics table
Cell 31.32 Compute pairwise effects
Cell 31.33 Classify WD trend/boundary status
Cell 31.34 Build optimization diagnostics
Cell 31.35 Build convergence diagnostics
Cell 31.36 Build runtime diagnostics
Cell 31.37 Optional Train-vs-Validation generalization diagnostic
Cell 31.38 Evaluate hypotheses
Cell 31.39 Generate learning/parameter-norm figures
Cell 31.40 Generate findings
Cell 31.41 Select S9 winner
Cell 31.42 Write winner JSON
Cell 31.43 Write Phase 32 reference update
Cell 31.44 Run S9 tests/discrepancies
Cell 31.45 Write summary/report
Cell 31.46 Register artifacts/checksums
Cell 31.47 Write README
Cell 31.48 Phase sign-off
```

---

# 173. Execution flow

```text
Verify Phase 30
        ↓
Load S8 winner
        ↓
Freeze FV* + YS* + L* + P* + A* + B* + LR*
        ↓
Declare WD0 / WD1 / WD2
        ↓
Inspect and freeze AdamW parameter-group policy
        ↓
Audit no explicit L2 loss
        ↓
Audit same data/model/batch/LR
        ↓
Verify WD1 exact reference
        ↓
Reuse WD1
        ↓
Register WD0
        ↓
Reseed + fresh loaders/model/optimizer
        ↓
Train + verify WD0 BEST
        ↓
Register WD2
        ↓
Reseed + fresh loaders/model/optimizer
        ↓
Train + verify WD2 BEST
        ↓
Verify runtime WD constant/group scope stable
        ↓
Build Wh metrics
        ↓
Build parameter norm + decay-exposure diagnostics
        ↓
Compute pairwise effects
        ↓
Analyze convergence/clipping
        ↓
Select minimum-RMSE WD
        ↓
Apply lower-WD exact-tie rule
        ↓
Update Phase 32 reference
        ↓
Write S9 artifacts
        ↓
SWEEP_S9_WEIGHTDECAY-v1 sign-off
```

---

# 174. Fail-fast order

Before expensive training:

```text
1. Phase 30 sign-off
2. S8 winner identity
3. FV*/YS*/L*/P*/A*/B*/LR* lock
4. WD definitions
5. common sample population
6. same model/parameter schema
7. inspect optimizer groups
8. freeze optimizer group scope
9. verify no new no-decay exclusions
10. verify no explicit L2 loss
11. exact AdamW non-WD config
12. fixed LR/dropout/clip/E50
13. initialization/order policy
14. WD1 reuse eligibility
15. Test firewall
16. Registry readiness
```

---

# 175. Why optimizer-group audit is the most important S9-specific check

Weight decay can be applied to different parameter subsets depending on optimizer construction.

A seemingly innocent refactor from:

```text
one group with all params
```

to:

```text
weights decay
bias/LayerNorm no decay
```

would fundamentally change the experiment.

Therefore parameter-group scope must be frozen before any S9 run.

---

# 176. Why no explicit L2 loss matters

AdamW weight decay and adding an L2 penalty to MSE are not the same experimental implementation.

Doing both would make the registered coefficient ambiguous and over-regularize the model.

S9 must use only the frozen AdamW WD mechanism.

---

# 177. Why selected LR must be loaded, not hard-coded

Phase 30 may select:

```text
1e-4
3e-4
or 1e-3.
```

WD strength in AdamW is conditional on this LR.

Hard-coding `3e-4` would break sequential handoff and invalidate S9.

---

# 178. Why WD0 remains AdamW

Using Adam for WD0 changes optimizer class.

The clean condition is:

```text
AdamW(weight_decay=0)
```

under the exact same builder/settings.

---

# 179. Why matched initialization/order matters

WD differences can be subtle.

Using identical initialization and mini-batch sequence for new WD0/WD2 reduces stochastic confounding and makes the comparison stronger.

---

# 180. Why parameter norms are not selection metrics

Weight decay intentionally influences parameter magnitude, but:

```text
smaller norm
```

does not guarantee:

```text
better forecast accuracy.
```

Parameter norms explain regularization behavior; Validation RMSE determines the winner.

---

# 181. Why theoretical decay multiplier is context only

The actual model simultaneously receives:

```text
adaptive gradient updates
weight decay
clipping-mediated gradient trajectory
```

and nonlinear parameter interactions.

Thus:

\[
(1-\eta\lambda)^K
\]

is not an observed model-norm prediction.

---

# 182. Winner verification checklist

Before writing `s9_weight_decay_winner.json`:

```text
[ ] WD0 valid completed run.
[ ] WD1 valid reused reference.
[ ] WD2 valid completed run.
[ ] Same FV*/YS*/L*/P*/A*/B*/LR*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same feature/scaler/target transform.
[ ] Same architecture/parameter schema.
[ ] Same AdamW class.
[ ] Same optimizer group topology.
[ ] Same optimizer group membership.
[ ] No new bias/LayerNorm exclusion.
[ ] No explicit L2 loss.
[ ] Same non-WD optimizer fields.
[ ] Same dropout/clip/E50.
[ ] Constant runtime WD verified.
[ ] Fresh optimizer state for new runs.
[ ] No warm-start.
[ ] Matched initialization/order audited.
[ ] All BEST checkpoints verified.
[ ] Full-precision RMSE used.
[ ] Pairwise effects computed.
[ ] Parameter norms treated as diagnostic.
[ ] Exact tie rule respected.
[ ] Boundary winner flagged if needed.
[ ] No Test.
```

---

# 183. Phase 32 handoff

Phase 32 receives:

```text
s9_weight_decay_winner.json
s9_reference_update.json
winner_run_id
winner_config_fingerprint
winner_optimizer_config_fingerprint
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
selected_wd_id
selected_weight_decay
optimizer_group_policy_id
population_fingerprint
```

and changes only:

```text
dropout.
```

---

# 184. If WD0 wins

Current reference:

```text
FV*
YS*
L*
P*
A*
B*
LR*
WD=0
DR=0.1
```

Phase 32 compares:

```text
0.1
0.2
0.3
```

at WD0.

---

# 185. If WD1 wins

Current reference retains `1e-4`.

---

# 186. If WD2 wins

Current reference becomes `1e-3`.

Phase 32 tests dropout under the stronger selected regularization coefficient.

---

# 187. Reference reuse for Phase 32

S9 winner already uses:

```text
dropout = 0.1
```

so Phase 32 normally:

```text
reuse S9 winner as DR01 reference
train DR02 and DR03.
```

---

# 188. Relationship with Phase 32 Dropout

Weight decay and dropout are both regularization mechanisms.

Their effects can interact.

Sequential design does not test full:

```text
WD × Dropout
```

grid.

S9 selects WD at dropout=.1; S10 then selects dropout at the chosen WD.

Document this limitation.

---

# 189. Relationship with Phase 37 Loss

Loss remains MSE throughout S9.

Different losses can alter parameter trajectories and regularization behavior.

S9 result is conditional on MSE.

---

# 190. Relationship with Phase 38 Epoch cap

If WD2 appears slower to converge, retain that evidence.

Do not extend WD2 here.

Phase 38 later tests epoch budget on current reference.

---

# 191. Relationship with Phase 39 Clipping

Weight decay is optimizer-side while gradient clipping acts on gradients.

S9 keeps clipping fixed.

Phase 39 later tests clipping policy.

---

# 192. Relationship with Phase 42 Candidate synthesis

All three WD conditions remain in Registry.

No losing condition is deleted.

---

# 193. Relationship with Phase 44 Rolling-origin

S9 winner is based on one Validation period.

Temporal robustness remains untested until Phase 44.

---

# 194. Relationship with Phase 46 Multi-seed

S9 uses seed 42.

Tiny WD margins are not seed-robust evidence.

---

# 195. Reproducibility metadata

New WD0/WD2 runs record:

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
weight decay
AdamW non-WD fields
optimizer parameter-group policy ID
optimizer parameter-scope fingerprint
scheduler/warmup state
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

# 196. No fabricated outputs

Do not pre-fill:

```text
winner
RMSE
parameter norms
generalization gaps
gradient norms
decay multipliers at runtime step counts
best epoch
runtime
```

before actual execution.

The theoretical formula may be defined now, but numerical exposure values depend on runtime LR/steps and must be computed from actual artifacts.

---

# 197. Phase sign-off

Create:

```text
phase_31_signoff.json
```

Minimum:

```text
phase = 31
phase_name = S9 Weight-decay sweep
sweep_version
sweep_id
source_s8_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
wd0_run_id
wd1_reference_run_id
wd2_run_id
new_run_ids
reused_run_ids
optimizer_group_policy_id
optimizer_group_scope_fingerprint
winner_wd_id
winner_weight_decay
winner_run_id
winner_rmse_wh
winner_is_boundary
population_fingerprint
metric_version
constant_wd_audit_status
optimizer_group_audit_status
explicit_l2_absent
fairness_audit_status
matched_initialization_status
sample_order_match_status
inherited_warnings
test_status
approved_for_phase32
overall_status
created_at
```

---

# 198. Acceptance checklist

```text
[ ] Phase 30 valid.
[ ] approved_for_phase31 = true.
[ ] SWEEP_S9_WEIGHTDECAY-v1 declared.
[ ] S8 winner loaded.
[ ] FV* fixed.
[ ] YS* fixed.
[ ] L* fixed.
[ ] P* fixed.
[ ] A* fixed.
[ ] B* fixed.
[ ] LR* loaded from S8 and fixed.
[ ] WD0 exactly 0.
[ ] WD1 exactly 1e-4.
[ ] WD2 exactly 1e-3.
[ ] No extra WD candidate.
[ ] AdamW fixed.
[ ] Optimizer group topology inspected.
[ ] Optimizer group membership inspected.
[ ] Optimizer group policy frozen.
[ ] Parameter-scope fingerprint created.
[ ] Same group scope across all candidates.
[ ] No new group created.
[ ] No parameter reassigned.
[ ] No new bias exclusion.
[ ] No new LayerNorm exclusion.
[ ] No ndim-based new exclusion.
[ ] Explicit L2 loss absent.
[ ] Criterion remains pure MSE.
[ ] Same Train IDs.
[ ] Same Validation IDs.
[ ] Same WINDOWPOP-v1.
[ ] Same feature fingerprint.
[ ] Same X scaler.
[ ] Same target transform/scaler.
[ ] Same lookback/pooling/activation/batch.
[ ] Same selected LR.
[ ] Same H1/WB0.
[ ] Same D64/H4/N2/FFN128.
[ ] Same dropout=.1.
[ ] Same PE/norm/mask policy.
[ ] Same regression head.
[ ] Same parameter count.
[ ] Same parameter schema.
[ ] Same state_dict key set.
[ ] Same AdamW non-WD fields.
[ ] E50 fixed.
[ ] Patience10 fixed.
[ ] min_delta fixed.
[ ] Clip1 fixed.
[ ] Scheduler absent.
[ ] Warmup absent.
[ ] Accumulation=1.
[ ] drop_last=False.
[ ] Mixed precision policy fixed.
[ ] Seed42 fixed.
[ ] Same steps per completed epoch.
[ ] WD0/WD2 initial states match.
[ ] WD1 init match checked if available.
[ ] WD0/WD2 sample order matches.
[ ] WD1 order match checked if available.
[ ] WD1 reference exact-match.
[ ] WD1 not retrained.
[ ] WD0 registered before training.
[ ] WD2 registered before training.
[ ] WD0 fresh loaders/model/optimizer.
[ ] WD2 fresh loaders/model/optimizer.
[ ] No model warm-start.
[ ] No optimizer-state reuse.
[ ] WD0 still uses AdamW.
[ ] WD0 trained via TRAINING_ENGINE-v1.
[ ] WD2 trained via TRAINING_ENGINE-v1.
[ ] WD0 BEST verified.
[ ] WD2 BEST verified.
[ ] WD1 BEST already verified.
[ ] Runtime WD constant for every optimizer group.
[ ] Runtime group membership stable.
[ ] All result rows use verified BEST.
[ ] All predictions finite.
[ ] All official metrics in Wh.
[ ] Full-precision RMSE used.
[ ] WD0/WD1 effect computed.
[ ] WD1/WD2 effect computed.
[ ] WD0/WD2 effect computed.
[ ] Zero-denominator WD ratio handled as null.
[ ] Trend pattern generated.
[ ] Winner=min RMSE.
[ ] Exact tie=lower WD.
[ ] Existing reference does not override RMSE.
[ ] Metric divergence recorded if present.
[ ] Boundary winner recorded if WD0/WD2.
[ ] No automatic range extension.
[ ] Parameter norm diagnostics generated.
[ ] Parameter norms are not selection criteria.
[ ] Decay-eligible norms computed if policy supports.
[ ] Theoretical decay exposure labeled context-only.
[ ] Theoretical exposure uses actual selected LR.
[ ] Optimizer-step diagnostics generated.
[ ] Gradient/clipping diagnostics generated.
[ ] Convergence diagnostics generated.
[ ] Runtime diagnostics generated.
[ ] Optional generalization diagnostics use Train/Validation only.
[ ] No Test overfitting assessment.
[ ] Hypothesis outcomes recorded.
[ ] Findings generated.
[ ] Inherited warnings propagated.
[ ] Winner artifact generated.
[ ] Phase 32 reference update generated.
[ ] DR01 reuse identified.
[ ] No explicit L2.
[ ] No optimizer switch.
[ ] No parameter-group redesign.
[ ] No WD schedule.
[ ] No dropout compensation.
[ ] No LR compensation.
[ ] No extra epochs.
[ ] No score-based rerun.
[ ] No failed/SANITY run in winner ranking.
[ ] No attention extraction.
[ ] No Test access.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] WD-LR interaction documented.
[ ] WD-dropout interaction documented.
[ ] WD-budget interaction documented.
[ ] Optimizer-scope limitation documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 199. Acceptance criteria

Phase 31 chỉ PASS khi:

```text
S8-selected data/model/batch/LR configuration is fixed.

Exactly WD0=0, WD1=1e-4 and WD2=1e-3 are represented.

Only AdamW weight-decay coefficient changes.

AdamW class and all non-WD optimizer fields are identical.

Optimizer parameter-group topology and membership are frozen.

No new bias/LayerNorm no-decay policy is introduced.

No explicit L2 penalty is added to MSE.

Same Train/Validation IDs and WINDOWPOP-v1 are used.

Same model parameter schema is used.

Same initialization/sample-order policy is audited.

WD1 exact S8 reference is reused.

WD0/WD2 are fresh seed-42 runs with fresh optimizer state.

Runtime WD is constant under the frozen group policy.

All BEST checkpoints are verified.

Validation RMSE Wh selects winner.

Exact RMSE tie selects lower WD.

Parameter norms/decay exposure remain diagnostic only.

Boundary winner is documented without hidden range extension.

Phase 32 reference is generated.

Test remains untouched.
```

---

# 200. Failure conditions

Phase 31 FAIL if:

```text
wrong S8 winner used

selected LR changes

WD values differ from registry

optimizer switches from AdamW

optimizer parameter-group topology changes

parameter membership changes

new bias/LayerNorm exclusion policy appears

explicit L2 term is added to MSE

WD changes during run

dropout changes

clip changes

epoch budget changes

sample population differs

architecture/parameter schema differs

WD1 reference mismatches but is reused

WD1 is retrained and best rerun chosen

WD0/WD2 warm-start

optimizer state is reused

one candidate technically fails but winner declared anyway

genuine numerical failure silently ignored

unregistered WD is tested/used

RMSE rounded before ranking

MAE/R² overrides lower RMSE

parameter norm overrides lower RMSE

runtime overrides lower RMSE

score-based rerun occurs

Test used.
```

---

# 201. Common mistakes

## 201.1 `loss = mse + wd * l2`

Sai. S9 dùng AdamW weight decay, không explicit L2 loss.

## 201.2 WD0 dùng `Adam` thay vì `AdamW(weight_decay=0)`

Sai optimizer identity.

## 201.3 Bắt đầu loại bias/LayerNorm khỏi decay trong Phase 31

Sai vì đổi parameter scope.

## 201.4 WD2 dùng LR nhỏ hơn để ổn định

Hai factors thay đổi.

## 201.5 WD0 dùng dropout cao hơn để tránh overfit

Không. Dropout sweep ở Phase 32.

## 201.6 Tăng epoch riêng cho WD2

Không.

## 201.7 Retrain WD1 để “công bằng”

Không. Exact reference reuse sạch hơn.

## 201.8 Warm-start WD0 từ WD1 BEST

Confounded.

## 201.9 Reuse AdamW state rồi đổi WD

Confounded.

## 201.10 Chọn model có parameter norm nhỏ nhất

Sai selection metric.

## 201.11 Thấy WD2 thắng rồi thử 1e-2

Hidden adaptive search.

## 201.12 Thấy WD0 thắng rồi thử 1e-5

Không thuộc S9.

## 201.13 Kết luận nhỏ norm gây generalization tốt

Unsupported causal claim.

## 201.14 Dùng Test để đánh giá overfitting

Forbidden.

## 201.15 Tạo weight-decay schedule

Không phải constant-WD sweep.

---

# 202. Recommended execution pseudocode

```text
load_phase30_signoff()
assert_approved_for_phase31()

s8 = load_s8_winner()

FV = s8.feature_variant_id
YS = s8.target_scaling_id
L  = s8.lookback_id
P  = s8.pooling_id
A  = s8.activation_id
B  = s8.batch_id
LR = s8.learning_rate

WDs = {
    "WD0": 0.0,
    "WD1": 1e-4,
    "WD2": 1e-3
}

audit_common_data_population()
audit_same_model_architecture()

group_policy = inspect_and_freeze_optimizer_group_policy(
    source_run=s8.winner_run_id
)

assert_no_new_no_decay_exclusions(group_policy)
assert_no_explicit_l2_in_loss()
audit_adamw_non_wd_fields()
assert_learning_rate(LR)
assert_dropout(0.1)
assert_clip(1.0)
assert_fixed_budget(E50, patience10)

wd1_reference = resolve_s8_winner_run()
assert_exact_s9_reference_match(wd1_reference)

results = {
    "WD1": wd1_reference
}

for wd_id in ["WD0", "WD2"]:
    register_run(wd_id)

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

    optimizer = build_fresh_adamw_with_frozen_group_policy(
        model=model,
        group_policy=group_policy,
        lr=LR,
        weight_decay=WDs[wd_id]
    )

    result = TRAINING_ENGINE_v1.fit(...)
    verify_constant_weight_decay(result, group_policy)
    verify_best_checkpoint(result)

    results[wd_id] = result

audit_new_run_initialization_match()
audit_new_run_sample_order_match()

metrics = build_verified_s9_metrics(results)
pairwise = compute_wd_pairwise_effects(metrics)
trend = classify_wd_trend(metrics)

parameter_norms = build_parameter_norm_diagnostics(results, group_policy)
decay_context = build_theoretical_decay_exposure(
    lr=LR,
    results=results
)

optimization = build_optimization_diagnostics(results)
convergence = build_convergence_diagnostics(results)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer_lower_wd=True
)

write_hypothesis_outcomes()
write_findings()
write_s9_winner(winner)
write_phase32_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 203. Definition of Done

\[
\boxed{
One\ Fixed\ Transformer\ Setup
+
Three\ Constant\ WD\ Coefficients
+
Frozen\ Optimizer\ Parameter\ Scope
+
Two\ Fresh\ Runs
+
One\ Valid\ Reused\ WD1
+
Same\ LR/Batch/Initialization
+
No\ Explicit\ L2
+
Verified\ BEST\ Metrics
+
S9\ Winner
+
Phase32\ Reference
+
No\ Test
}
\]

---

# 204. Final status contract

```text
PHASE 31 tests ADAMW WEIGHT DECAY only.

Current:
feature variant from prior sweeps
target scaling from S3
lookback from S4
pooling from S5
activation from S6
batch from S7
learning rate from S8.

Candidates:

WD0 = 0
WD1 = 1e-4
WD2 = 1e-3

WD1:
reuse S8 winner if exact match.

WD0/WD2:
fresh seed-42 runs.

Same:
Train/Validation IDs
WINDOWPOP-v1
X/y pipeline
model architecture
parameter schema
batch B*
learning rate LR*
AdamW identity
AdamW non-WD fields
optimizer parameter-group topology
optimizer parameter-group membership
MSE
dropout=.1
E50
patience10
clip1
accumulation1
Training Engine
Metric version.

No:
explicit L2 loss
new LayerNorm/bias exclusion
optimizer regrouping
WD schedule
LR compensation
dropout compensation
extra epochs
warm-start
optimizer-state reuse
hidden WD candidate.

Runtime WD:
constant under frozen group policy.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer lower WD.

Parameter norms:
diagnostic only.

Boundary winner:
record, do not auto-expand search.

No Test.

After SWEEP_S9_WEIGHTDECAY-v1 PASS:
update current reference
and proceed to
PHASE 32 — S10 Dropout sweep.
```

---

# 205. Final check

Correct workflow:

```text
Load S8 winner
→ Freeze FV* + YS* + L* + P* + A* + B* + LR*
→ Inspect/freeze AdamW parameter-group scope
→ Define WD0 / WD1 / WD2
→ Verify no explicit L2
→ Audit same data/model/optimizer non-WD config
→ Reuse WD1
→ Fresh WD0 from seed42
→ Verify BEST
→ Fresh WD2 from seed42
→ Verify BEST
→ Verify constant WD/group membership
→ Compare full-precision Validation RMSE
→ Analyze parameter norms/decay context diagnostically
→ Select S9 winner
→ Apply lower-WD exact-tie rule
→ Update Phase 32 reference
```

Incorrect workflow:

```text
add L2 penalty
→ exclude LayerNorm/bias only now
→ change LR with WD
→ retrain WD1
→ choose smallest parameter norm
→ test extra WD after seeing result
→ inspect Test
```

Chỉ sau khi `SWEEP_S9_WEIGHTDECAY-v1` được sign-off mới chuyển sang **PHASE 32 — S10 Dropout sweep**.
