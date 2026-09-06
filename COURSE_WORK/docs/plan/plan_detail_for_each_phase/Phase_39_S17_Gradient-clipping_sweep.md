# PHASE 39 — S17 GRADIENT-CLIPPING SWEEP

## Kế hoạch controlled sweep cho Gradient Clipping của Transformer Regression

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S16_EPOCHCAP-v1`  
**Sweep ID:** `S17_GRADIENT_CLIPPING`  
**Output version:** `SWEEP_S17_GRADIENTCLIP-v1`  
**Phase trước:** `Phase_38_S16_Epoch-cap_sweep.md`

---

# 1. Vai trò của Phase 39

Phase 39 là controlled experiment thứ mười bảy trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với toàn bộ data pipeline, feature configuration, target scaling, lookback, pooling, activation, batch size, optimizer, learning rate, weight decay, dropout, `d_model`, attention heads, encoder layers, FFN width, training loss, epoch cap, patience, sample population và seed đã được khóa từ Phase 38, việc sử dụng global gradient-norm clipping với `max_norm=1.0` có thực sự giúp mô hình tốt hơn hoặc ổn định hơn so với không clipping hay không?

Phase 39 chỉ thay đúng một registered factor:

```text
gradient_clipping
```

với hai condition:

```text
GC0 = clipping OFF
GC1 = global norm clipping, max_norm = 1.0
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Gradient\ Intervention
+
Same\ Gradients\ Before\ Intervention
+
Same\ Model
+
Same\ Data
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

# 2. Scientific question chính xác của S17

S17 không hỏi:

```text
gradient norm nào đẹp hơn?
```

S17 hỏi:

> Dưới exact current training configuration, việc giới hạn global gradient norm ở 1.0 trước `optimizer.step()` có tạo Validation forecasting tốt hơn không?

Đồng thời Phase phải mô tả:

```text
clipping có thực sự active không
active bao nhiêu bước
pre-clip norms như thế nào
GC0 sẽ vượt threshold bao nhiêu lần
late-training stability ra sao
```

nhưng tất cả optimization diagnostics đều là secondary evidence.

Winner vẫn do:

```text
verified BEST Validation RMSE Wh
```

quyết định.

---

# 3. Vị trí Phase 39 trong master execution plan

```text
Phase 23 → S1 Feature-set
Phase 24 → S2 Time-feature
Phase 25 → S3 Target-scaling
Phase 26 → S4 Lookback
Phase 27 → S5 Pooling
Phase 28 → S6 Activation
Phase 29 → S7 Batch
Phase 30 → S8 Learning-rate
Phase 31 → S9 Weight-decay
Phase 32 → S10 Dropout
Phase 33 → S11 d_model
Phase 34 → S12 Heads
Phase 35 → S13 Layers
Phase 36 → S14 FFN
Phase 37 → S15 Loss
Phase 38 → S16 Epoch-cap
Phase 39 → S17 Gradient-clipping
Phase 40 → S18 RevIN
```

Phase 39 không được quay lại thay:

```text
feature set
time features
target scaling
lookback
pooling
activation
batch size
learning rate
weight decay
dropout
d_model
num_heads
num_layers
ffn_dim
training loss
epoch cap
patience
min_delta
RevIN
boundary protocol
```

---

# 4. Câu hỏi nghiên cứu của S17

Phase 39 phải trả lời:

```text
1. GC0 hay GC1 tạo verified Validation RMSE Wh thấp hơn?

2. GC1 clipping có thực sự active trong reference run không?

3. Bao nhiêu optimizer steps của GC1 có pre-clip norm > 1.0?

4. GC0 có bao nhiêu steps vượt threshold 1.0 nếu xét counterfactual?

5. Khi clipping active, mức norm excess trước clipping lớn đến đâu?

6. GC1 có giảm numerical instability hoặc non-finite risk không?

7. GC0 có học nhanh hơn/chậm hơn khi không bị rescale gradient?

8. Best epoch/early stopping behavior khác nhau như thế nào?

9. Clipping ảnh hưởng thế nào đến train criterion, Validation RMSE, MAE, R²?

10. Chipping có chỉ là một safety mechanism non-binding hay là active regularizer/optimizer intervention?

11. Gradient clipping setting nào trở thành current reference cho Phase40?
```

---

# 5. Current reference từ Phase 38

Phase 39 phải load:

```text
s16_epoch_cap_winner.json
s16_reference_update.json
phase_38_signoff.json
```

để resolve:

```text
FV*      = selected feature variant
YS*      = selected target scaling
L*       = selected lookback
P*       = selected pooling
A*       = selected activation
B*       = selected batch size
LR*      = selected learning rate
WD*      = selected weight decay
DR*      = selected dropout
D*       = selected d_model
H*       = selected num_heads
HD*      = selected head_dim
N*       = selected num_layers
F*       = selected ffn_dim
LOSS*    = selected training loss
EPOCHS*  = selected max_epochs
```

Possible:

```text
EPOCHS* ∈ {50,100}
LOSS*   ∈ {MSE, Huber(delta=1.0 model-space)}
```

Phase 39 không được hard-code:

```text
E50
MSE
```

---

# 6. Canonical clipping options

Registry:

```text
GC0 = OFF
GC1 = GLOBAL_NORM_1
```

Exact semantics:

```text
GC0:
no gradient rescaling before optimizer.step()

GC1:
global gradient norm clipping
max_norm = 1.0
norm_type = 2
after backward
before optimizer.step()
```

---

# 7. Reference clipping condition

Current sequential reference từ Phase 38 already uses:

```text
GC1 = max_norm1
```

vì clipping chưa được swept trước Phase39.

If exact S17 contract match:

```text
GC1 → REUSE S16 winner
GC0 → NEW fresh run
```

Normal S17 execution:

```text
1 new GC0 scientific run
+
1 reused GC1 reference
```

---

# 8. Definition của global gradient norm

Với tất cả gradients hữu hạn của trainable parameters:

\[
g = \{g_1,g_2,\ldots,g_K\}
\]

global L2 norm được hiểu là norm của toàn bộ gradients khi xem như một vector lớn:

\[
\|g\|_2
=
\sqrt{
\sum_k \|g_k\|_2^2
}
\]

Không phải:

```text
average parameter gradient
per-layer norm
maximum single-tensor norm.
```

---

# 9. GC1 exact intervention point

Official GC1 step order:

```text
forward
↓
criterion
↓
backward
↓
compute global pre-clip gradient norm
↓
verify finite
↓
clip global norm to max_norm=1.0 when needed
↓
optimizer.step()
```

Do not clip:

```text
before backward
after optimizer.step
per layer
per tensor
per value.
```

---

# 10. GC0 exact semantics

Official GC0 step order:

```text
forward
↓
criterion
↓
backward
↓
compute global gradient norm for diagnostics/non-finite guard WITHOUT modifying gradients
↓
verify finite
↓
optimizer.step() with original raw gradients.
```

Important:

```text
GC0 = no clipping
```

does **not** mean:

```text
do not inspect gradient finiteness.
```

Fail-fast numerical safety must remain.

---

# 11. Non-finite guard is not the swept factor

Both GC0 and GC1 must fail if gradients are non-finite.

Required:

```text
NaN gradient → FAIL
Inf gradient → FAIL
```

Do not let GC0 silently step with non-finite gradients merely because clipping is off.

The difference is only:

```text
whether finite gradients are rescaled when global norm > 1.0.
```

---

# 12. GC1 clipping activation

GC1 is considered **active** on an optimizer step when:

```text
preclip_global_grad_norm > 1.0.
```

If:

```text
preclip_global_grad_norm <= 1.0
```

then clipping is non-binding for that step.

---

# 13. Exact threshold behavior

For reporting:

```text
active if norm > max_norm
inactive if norm <= max_norm
```

Do not use arbitrary tolerance to redefine “clipped”.

Numerical comparison tolerance may be used for auditing logged values, not for changing protocol classification.

---

# 14. Pre-clip norm is the primary gradient diagnostic

For both conditions, record:

```text
preclip_global_grad_norm.
```

This is the directly comparable quantity before the S17 intervention.

For GC1, optional:

```text
postclip_global_grad_norm.
```

For GC0:

```text
postclip = N/A
```

or:

```text
effective_step_grad_norm = preclip
```

if the distinction is clearly labeled.

---

# 15. Counterfactual threshold exceedance for GC0

GC0 does not clip, but Phase39 should compute:

```text
would_clip_at_1
=
(preclip_global_grad_norm > 1.0)
```

for diagnostics.

This allows:

```text
fraction_GC0_steps_above_1
```

without modifying gradients.

Do not call this:

```text
GC0 clipping fraction.
```

Correct label:

```text
GC0 threshold-exceedance fraction
or
counterfactual would-clip fraction.
```

---

# 16. GC1 actual clipping fraction

For GC1:

\[
ClipFraction
=
\frac{
\#\{steps:\|g\|_2>1\}
}{
\#optimizer\ steps
}
\]

This is an actual intervention frequency.

---

# 17. Why GC0 and GC1 gradient norms diverge over time

Before the first active clipping event, under fully matched trajectory:

```text
GC0 and GC1 should be identical.
```

After a GC1 clipping event:

```text
parameter updates can diverge
```

and therefore future predictions/gradients can differ.

This is expected.

Do not interpret all later gradient differences as direct one-step clipping effects.

---

# 18. First-active-clipping event is a key diagnostic

If GC1 telemetry supports it, record:

```text
first_clipped_optimizer_step
first_clipped_epoch
preclip_norm_at_first_clip.
```

This identifies the exact point where GC1 is expected to depart from a matched GC0 trajectory.

---

# 19. Prefix-equivalence before first active clipping

Strong scientific principle:

If:

```text
same initial state
same samples
same RNG
same model/loss/optimizer
```

then GC0 and GC1 should follow the same training trajectory until GC1 first needs to rescale a gradient.

Therefore:

```text
pre-first-clip trajectory mismatch
```

is suspicious and should trigger audit.

---

# 20. Special case: GC1 never clips

If reference GC1 has:

```text
actual_clip_count = 0
```

then GC1 never modifies gradients.

Under deterministic matched execution:

```text
GC0 and GC1 should produce the same trajectory
```

up to numerical reproducibility tolerance.

This is a powerful non-binding-clipping diagnostic.

---

# 21. Non-binding clipping interpretation

If:

```text
GC1 clip_count = 0
```

and GC0/GC1 metrics match:

Safe:

> Gradient clipping at max_norm=1.0 was non-binding under the current configuration.

This does not mean clipping is universally unnecessary under other configurations.

---

# 22. Suspicious case: GC1 never clips but GC0/GC1 differ materially

Investigate:

```text
initialization mismatch
sample order mismatch
RNG mismatch
environment nondeterminism
hidden config drift
gradient logging mutation
mode transitions
optimizer mismatch.
```

Do not attribute difference to clipping because clipping did not activate.

---

# 23. Clipping changes magnitude, not optimizer hyperparameters

GC1 changes the gradient tensor before AdamW consumes it.

It does **not** change:

```text
LR
WD
AdamW betas
AdamW eps
parameter groups.
```

---

# 24. AdamW interaction nuance

Gradient clipping occurs before:

```text
AdamW optimizer.step()
```

and therefore affects the gradient signal used to update AdamW moment estimates.

After clipping-active steps, not only the current parameter update but also future AdamW moment state may diverge.

This is an intended consequence.

---

# 25. Weight decay remains decoupled

Current optimizer is AdamW.

Do not implement clipping by modifying:

```text
weight decay term
parameter norm
parameter values.
```

Gradient clipping is not weight clipping.

---

# 26. Gradient norm clipping is not gradient value clipping

Forbidden substitution:

```text
clip_grad_value_
```

or equivalent per-element clipping.

S17 is specifically:

```text
global norm clipping vs off.
```

---

# 27. Gradient clipping is not parameter clipping

Do not clamp model weights.

---

# 28. Gradient clipping is not residual clipping

Do not clamp errors or targets.

---

# 29. Gradient clipping is not loss clipping

Do not cap criterion values.

---

# 30. Loss remains selected LOSS*

Hard:

```text
LOSS*
```

from S15.

If Huber:

```text
Huber delta = 1.0 model-space
reduction=mean.
```

If MSE:

```text
MSE reduction=mean.
```

No clipping-specific loss change.

---

# 31. Important Huber interaction context

If Huber won S15, Huber already limits growth of the derivative for large residuals relative to MSE.

Therefore GC1 may be less active.

This is a **conditional context**, not a reason to:

```text
change threshold
disable clipping without running GC0
```

S17 empirically measures the remaining value of clipping under selected loss.

---

# 32. Epoch cap remains selected EPOCHS*

Hard:

```text
max_epochs = EPOCHS*
```

from S16.

Possible:

```text
50
or
100.
```

No clipping-specific epoch budget.

---

# 33. Patience remains 10

Hard:

```text
patience = 10
min_delta = 0.
```

Same early stopping.

---

# 34. Early stopping metric remains Validation RMSE Wh

Both:

```text
monitor = validation_rmse_wh
direction = minimize.
```

No optimization-stability metric enters stopping logic.

---

# 35. BEST metric remains Validation RMSE Wh

Hard.

---

# 36. Scheduler remains None

No clipping-specific scheduler.

---

# 37. Batch size remains selected B*

Gradient norms are batch-dependent.

Therefore changing batch size in S17 would destroy factor isolation.

Hard:

```text
same B*
same drop_last=false
same sample population.
```

---

# 38. Learning rate remains selected LR*

No:

```text
higher LR for GC1
lower LR for GC0.
```

Even though clipping interacts with effective update magnitude.

---

# 39. Weight decay remains selected WD*

No clipping-specific WD.

---

# 40. Model architecture remains exactly identical

Expected:

```text
same architecture fingerprint
same parameter count
same parameter names
same tensor shapes
same buffers.
```

Clipping is training-loop only.

---

# 41. Parameter-schema equality is hard expected

Required:

```text
state_dict schema_GC0
=
state_dict schema_GC1.
```

---

# 42. Gradient-clip configuration is training metadata

Experiment Registry must distinguish:

```text
model_config
criterion_config
optimizer_config
gradient_clip_config
training_config.
```

Model/criterion/optimizer configs stay fixed.

Only `gradient_clip_config` changes.

---

# 43. Recommended registered clipping configs

```text
GC0:
enabled = false
method = none
max_norm = N/A
norm_type = N/A

GC1:
enabled = true
method = global_norm
max_norm = 1.0
norm_type = 2
error_if_nonfinite = true
```

For GC0:

```text
nonfinite_gradient_guard = true
```

must still be explicit.

---

# 44. Gradient norm measurement must be non-mutating for GC0

Use a diagnostic helper that computes the global norm from `.grad` tensors without rescaling or overwriting them.

Hard test:

```text
gradient fingerprint before norm measurement
==
gradient fingerprint after norm measurement
```

for a disposable GC0 batch.

---

# 45. GC1 clipping must occur exactly once per optimizer step

Do not accidentally clip:

```text
after every layer backward hook
twice before optimizer step
before and after accumulation
```

Current accumulation contract:

```text
gradient_accumulation = 1.
```

Therefore one backward corresponds to one optimizer step.

---

# 46. Gradient accumulation remains 1

Hard:

```text
accumulation = 1.
```

This avoids ambiguity about whether clipping occurs per microbatch or after accumulated gradient.

---

# 47. If accumulation were >1, clipping semantics would need separate lock

Not relevant in current protocol.

Do not change accumulation in S17.

---

# 48. Mixed precision policy remains frozen

If baseline contract uses:

```text
AMP = false
```

keep it false.

No clipping/GradScaler interaction is introduced.

---

# 49. No Adaptive Gradient Clipping

Do not use:

```text
AGC
unit-wise clipping
parameter-relative clipping.
```

Not registered.

---

# 50. No per-layer clipping

No.

---

# 51. No per-head clipping

No.

---

# 52. No per-parameter clipping threshold

No.

---

# 53. No dynamic threshold

Do not use:

```text
percentile-based max_norm
EMA-based threshold
schedule max_norm over epochs.
```

---

# 54. No clip warmup

Do not turn clipping on only after some epoch.

GC1 enabled from the first official backward step.

---

# 55. No “clip only if norm > X and loss high”

Standard norm criterion only.

---

# 56. Working hypothesis H-S17-01

```text
GC1 may improve stability and Validation RMSE if large gradient-norm events are frequent enough to harm optimization.
```

Status:

```text
UNTESTED.
```

---

# 57. H-S17-02

```text
GC0 may match or outperform GC1 if clipping is rarely active or if max_norm=1.0 over-constrains useful updates.
```

Status:

```text
UNTESTED.
```

---

# 58. H-S17-03

```text
If GC1 never activates, GC0 and GC1 should be functionally equivalent under reproducible execution.
```

Status:

```text
UNTESTED.
```

---

# 59. H-S17-04

```text
GC1 may reduce extreme pre-step update events without necessarily improving Validation RMSE.
```

Status:

```text
UNTESTED.
```

---

# 60. Preconditions

Required:

```text
Phase38 = PASS
```

or:

```text
PASS_WITH_WARNING
```

with no unresolved critical issue.

Also:

```text
approved_for_phase39 = true.
```

---

# 61. Required upstream artifacts

```text
s16_epoch_cap_winner.json
s16_reference_update.json
phase_38_signoff.json
```

---

# 62. Required upstream contracts

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
SWEEP_S9_WEIGHTDECAY-v1
SWEEP_S10_DROPOUT-v1
SWEEP_S11_DMODEL-v1
SWEEP_S12_HEADS-v1
SWEEP_S13_LAYERS-v1
SWEEP_S14_FFN-v1
SWEEP_S15_LOSS-v1
SWEEP_S16_EPOCHCAP-v1
```

---

# 63. Carry-forward warnings

Propagate all unresolved non-critical warnings.

Examples:

```text
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
BOUNDARY_WINNER
SAMPLE_ORDER_NOT_VERIFIABLE
INITIALIZATION_NOT_VERIFIABLE
PREFIX_REPRODUCIBILITY_LIMITATION
E100_CAP_REACHED
RANDOM_CONTROL_GAIN.
```

Include in:

```text
S17 manifest
S17 summary
S17 report
S17 winner
Phase40 reference update.
```

---

# 64. Swept factor only

Canonical:

```text
gradient_clipping
```

Allowed:

```text
OFF
GLOBAL_NORM_1
```

Aliases:

```text
GC0
GC1.
```

No alternative threshold candidate.

---

# 65. Frozen data contract

Hard:

```text
FV*
YS*
L*
H1
WB0
WINDOWPOP-v1

same Train IDs
same Validation IDs
same feature fingerprint
same X scaler
same target transform/scaler
same ordered Validation population.
```

---

# 66. Frozen architecture contract

Hard:

```text
D*
H*
HD*
N*
F*
P*
A*
DR*
sinusoidal PE
POST_NORM
no causal mask
no padding mask
same regression head.
```

---

# 67. Frozen loss contract

Hard:

```text
LOSS*
```

and if Huber:

```text
delta=1.0 model-space
target-scaler checksum identical.
```

---

# 68. Frozen optimizer contract

Hard:

```text
AdamW
LR*
WD*
same optimizer group topology
same betas/eps
same optimizer construction.
```

---

# 69. Frozen training-budget contract

Hard:

```text
max_epochs = EPOCHS*
patience=10
min_delta=0
scheduler=None
warmup=None
accumulation1
same precision policy
seed42.
```

---

# 70. GC1 reference reuse gate

Exact match required on:

```text
FV*
YS*
L*
P*
A*
B*
LR*
WD*
DR*
D*
H*
HD*
N*
F*
LOSS*
EPOCHS*
H1
WB0
WINDOWPOP-v1

GC1
max_norm=1.0
norm_type=2
patience10
min_delta0
scheduler=None
warmup=None
accumulation1
seed42

TRAINING_ENGINE-v1
METRICS-v1.
```

If Huber:

```text
delta1 model-space.
```

Mismatch:

```text
STOP.
```

---

# 71. GC1 telemetry reuse requirement

For full S17 diagnostics, attempt to retrieve from S16 reference:

```text
preclip grad norm per step/epoch
actual clipping counts
clipping fractions
nonfinite events.
```

If unavailable:

```text
GC1_CLIP_ACTIVITY = NOT_VERIFIABLE_FROM_REFERENCE.
```

This does not necessarily invalidate RMSE comparison, but weakens clipping-mechanism interpretation.

Do not retrain GC1 solely to create telemetry.

---

# 72. Fresh GC0 run

Execution:

```text
register GC0
↓
reseed42
↓
fresh Train DataLoader
↓
fresh Validation DataLoader
↓
fresh exact selected Transformer
↓
fresh exact selected criterion
↓
fresh AdamW(LR*,WD*)
↓
max_epochs=EPOCHS*
patience10
gradient clipping OFF
nonfinite gradient guard ON
↓
TRAINING_ENGINE-v1-compatible GC0 path.
```

No warm-start.

---

# 73. GC0 training-engine implementation requirement

Do not fork an entirely different training loop.

Preferred:

```text
same TRAINING_ENGINE-v1
with gradient_clip_config.enabled = false
```

and the same diagnostics hooks.

This minimizes hidden drift.

---

# 74. Training-engine code path audit

Apart from clipping branch:

```text
if clipping_enabled:
    rescale gradients
else:
    do not rescale
```

all other step logic must be identical.

Hard audit by code/config fingerprint where feasible.

---

# 75. Exact step-order contract

For GC0:

```text
move batch
zero_grad(set_to_none=True)
forward
shape/finite checks
criterion
backward
compute preclip norm non-mutating
finite-grad guard
optimizer.step
aggregate
```

For GC1:

```text
move batch
zero_grad(set_to_none=True)
forward
shape/finite checks
criterion
backward
clip_grad_norm_(..., max_norm=1.0, norm_type=2, error_if_nonfinite=True)
optimizer.step
aggregate
```

The difference must be isolated to gradient rescaling.

---

# 76. GC1 `clip_grad_norm_` return semantics

When the implementation provides the pre-clip total norm as return value, log it as:

```text
preclip_global_grad_norm.
```

Do not mislabel it as postclip norm.

---

# 77. GC0 norm helper semantics

Use exactly the same norm definition:

```text
global L2 over all available trainable gradients
```

as GC1.

Do not compare GC1 global norm with GC0 average per-layer norm.

---

# 78. `None` gradients

Parameters with:

```text
grad is None
```

are excluded from norm computation.

However unexpected persistent `None` gradients should be audited separately as optimizer/model connectivity issues.

---

# 79. Gradient coverage audit

Record:

```text
trainable parameter count
parameters with grad
parameters with grad=None
```

for sampled/official steps.

GC0/GC1 should have same gradient-connectivity pattern under same architecture.

---

# 80. No frozen-parameter drift

Same parameters must remain trainable/frozen across conditions.

---

# 81. Gradient fingerprint mutation test

Disposable unit test:

1. run backward;
2. fingerprint gradients;
3. call GC0 norm helper;
4. fingerprint again.

Expected:

```text
exactly unchanged.
```

For GC1 active test:

1. create gradients with norm >1;
2. fingerprint before;
3. clip;
4. verify gradient values changed;
5. verify direction approximately preserved under uniform global scaling.

---

# 82. GC1 direction-preservation test

For global norm clipping, when active, the whole gradient vector should be rescaled by one common scalar factor.

Therefore direction should remain approximately the same.

This unit test is implementation-level only.

Do not use it as scientific evidence.

---

# 83. Synthetic active-clipping test

Create a disposable simple model/gradient case where:

```text
global norm > 1.
```

Verify GC1:

```text
postclip norm <= 1.0 within numerical tolerance.
```

GC0:

```text
norm remains >1 and gradients unchanged.
```

---

# 84. Synthetic inactive-clipping test

Create:

```text
global norm <1.
```

Verify GC1 gradients remain unchanged within tolerance.

This protects against unconditional rescaling bugs.

---

# 85. Non-finite synthetic test

Create a disposable non-finite gradient.

Expected:

```text
GC1 → fail due error_if_nonfinite / guard
GC0 → fail due explicit finite-grad guard.
```

Neither may optimizer-step.

---

# 86. No optimizer step after non-finite detection

Hard.

---

# 87. Initial model state matching

Because model architecture and seed are identical:

```text
GC0 initial state
```

should match GC1 reference initial state if provenance exists.

If not:

```text
NOT_VERIFIABLE.
```

---

# 88. Sample-order matching

Same loader/seed policy.

If GC1 per-epoch sample-order fingerprints exist:

```text
compare.
```

Else:

```text
NOT_VERIFIABLE.
```

---

# 89. RNG policy matching

Same:

```text
Python seed
NumPy seed
PyTorch seed
device seed policy
dropout execution structure.
```

No per-batch reseeding.

---

# 90. Prefix before first GC1 clip

If GC1 clip telemetry and deterministic provenance are available, define:

```text
shared_prefix_end_step = first_GC1_clip_step - 1.
```

Compare available metrics/state probes over that prefix.

Expected:

```text
MATCH / MATCH_WITH_TOLERANCE.
```

---

# 91. If first GC1 clip occurs in first optimizer step

Then no non-trivial pre-intervention training prefix exists.

Mark:

```text
PRECLIP_PREFIX_LENGTH = 0.
```

Do not force a prefix comparison.

---

# 92. If GC1 clipping never active

Compare full common trajectory where feasible.

Expected:

```text
FUNCTIONALLY_NON_BINDING.
```

---

# 93. Clipping activity classification

GC1 reference should be classified into:

```text
NEVER_ACTIVE
RARELY_ACTIVE
INTERMITTENT_ACTIVE
FREQUENTLY_ACTIVE
UNKNOWN_TELEMETRY
```

Do not invent percentage cutoffs ad hoc.

Recommended quantitative reporting should always include the exact clipping fraction; qualitative label thresholds must be predeclared if used.

Safer default:

```text
report exact fraction
and use only NEVER_ACTIVE vs ACTIVE.
```

---

# 94. Preferred binary mechanism classification

Use:

```text
CLIPPING_NON_BINDING
if actual_clip_count == 0

CLIPPING_BINDING
if actual_clip_count > 0

CLIPPING_ACTIVITY_UNKNOWN
if telemetry unavailable.
```

This avoids arbitrary qualitative thresholds.

---

# 95. Gradient excess magnitude

For steps with:

```text
preclip_norm >1
```

compute:

```text
excess = preclip_norm - 1.0
ratio = preclip_norm / 1.0.
```

Summarize:

```text
median
p90
p95
max
```

for GC1 actual clipping events.

For GC0:

```text
counterfactual excess distribution
```

over threshold-exceeding steps.

---

# 96. Effective clipping scale diagnostic

For GC1 active steps, record if available:

```text
effective_scale ≈ postclip_norm / preclip_norm
```

or implementation-derived equivalent.

Do not rely on exact internal epsilon as a scientific result.

---

# 97. Gradient-norm distribution summary

For each condition:

```text
count optimizer steps
mean
median
p75
p90
p95
p99 optional
max
fraction >1
nonfinite count.
```

Prefer percentiles over only mean/max.

---

# 98. Per-epoch gradient summary

Store:

```text
epoch
mean preclip norm
max preclip norm
threshold exceedance fraction
actual clipping fraction if GC1
```

This supports trajectory interpretation.

---

# 99. Late-training gradient summary

If `EPOCHS*=100` and run uses epochs >50, optional segments:

```text
PRE50
POST50.
```

If E50 selected:

```text
segment split not necessary.
```

---

# 100. Gradient norm cannot select winner

Even if:

```text
GC1 max norm smaller
GC1 clipping fraction lower/higher
GC0 updates larger
```

winner remains Validation RMSE.

---

# 101. Training criterion curves

Because LOSS* is the same across GC0/GC1:

```text
raw Train criterion values are numerically comparable
```

unlike S15 loss sweep.

Still:

```text
Training criterion is secondary
Validation RMSE Wh is primary.
```

---

# 102. Validation metrics remain identical pipeline

Every epoch:

```text
model.eval()
full ordered Validation
inverse transform if YS1
MAE/RMSE/R² in Wh.
```

---

# 103. Primary metric

Hard:

```text
verified BEST Validation RMSE Wh.
```

---

# 104. Secondary scientific metrics

Record:

```text
Validation MAE Wh
Validation R²
best epoch
epochs completed
stop reason
Train criterion
Validation trajectories.
```

---

# 105. Optimization diagnostics

Record:

```text
preclip grad norms
actual clipping fraction
counterfactual GC0 would-clip fraction
first active clipping
gradient excess
nonfinite events
optimizer steps
steps to BEST.
```

---

# 106. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{GC0},
RMSE_{GC1}
\right)
\]

using full-precision verified Validation RMSE Wh.

---

# 107. Exact RMSE tie rule

If exact full-precision equality:

```text
prefer GC0.
```

Rationale:

```text
no gradient intervention
simpler training rule
no max_norm hyperparameter in active training
predeclared parsimony.
```

Only exact tie invokes this rule.

---

# 108. Safety does not override a non-tie Validation winner inside S17

If GC1 has lower RMSE:

```text
GC1 wins.
```

If GC0 has lower RMSE and remains numerically valid:

```text
GC0 wins.
```

Do not override by subjective “safer” preference.

---

# 109. Technical failure is different from lower score

If GC0 encounters genuine non-finite gradients and cannot complete under strict protocol:

```text
S17 incomplete
```

until failure policy is resolved.

Do not simply call GC1 winner from a failed-vs-completed comparison.

---

# 110. Clipping effect formula

Define GC0→GC1:

\[
\Delta RMSE
=
RMSE_{GC0}
-
RMSE_{GC1}.
\]

Positive:

```text
GC1 improves.
```

Relative:

\[
Improvement\%
=
100\times
\frac{RMSE_{GC0}-RMSE_{GC1}}
{RMSE_{GC0}}.
\]

Also:

```text
MAE delta = MAE_GC0 - MAE_GC1
R² delta = R²_GC1 - R²_GC0.
```

---

# 111. RMSE/R² consistency guard

Same Validation population means:

```text
RMSE and R² ranking should be consistent.
```

If not:

```text
investigate.
```

MAE may disagree legitimately.

---

# 112. Metric ranking divergence

If RMSE and MAE disagree:

```text
METRIC_RANKING_DIVERGENCE.
```

Winner remains RMSE-based.

---

# 113. BEST verification for GC0

After official GC0 run:

```text
fresh exact model
↓
verify full config with clipping OFF
↓
strict-load BEST
↓
model.eval()
↓
full ordered Validation
↓
inverse target transform
↓
METRICS-v1
↓
verify stored BEST metrics.
```

Clipping setting does not affect inference, but training provenance must be correct.

---

# 114. GC1 reference verification

No retraining.

Verify:

```text
exact S16 winner
GC1 max_norm1
same selected max_epochs/loss/model/data
same metric version
same Training Engine provenance
BEST already verified.
```

---

# 115. Checkpoint metadata

GC0 BEST/LAST must include:

```text
gradient_clip_id = GC0
gradient_clip_enabled = false
gradient_clip_method = none
nonfinite_gradient_guard = true
```

GC1 metadata must include:

```text
gradient_clip_id = GC1
enabled=true
method=global_norm
max_norm=1.0
norm_type=2
error_if_nonfinite=true.
```

---

# 116. Inference artifacts do not need clipping logic

Gradient clipping is training-only.

Evaluation does not call backward or clipping.

---

# 117. Same sample population hard gate

Required:

```text
Train IDs_GC0 == Train IDs_GC1
Validation IDs_GC0 == Validation IDs_GC1
WINDOWPOP-v1 equal.
```

---

# 118. Same DataLoader settings

Both:

```text
B*
shuffle Train=true
shuffle Validation=false
drop_last=false
same workers
same generator policy.
```

---

# 119. Same optimizer steps per complete epoch

Because batch/population fixed:

```text
steps_per_epoch_GC0
=
steps_per_epoch_GC1.
```

Total steps may differ through early stopping.

---

# 120. Same max epoch cap

Hard:

```text
max_epochs=EPOCHS*
```

for both.

---

# 121. Same early-stop behavior contract

Same:

```text
patience10
strict improvement
Validation RMSE Wh.
```

Different actual stop epochs are valid outcomes.

---

# 122. No clip-aware early stopping

Do not stop because:

```text
clipping fraction too high
gradient norm > X.
```

Only normal numerical failure guard can terminate for invalid gradients.

---

# 123. No adaptive LR after clipping events

Do not reduce LR when gradients clip.

No ReduceLROnPlateau.

---

# 124. No skip-step on high finite norm

GC0:

```text
high but finite norm
→ optimizer.step normally.
```

GC1:

```text
high but finite norm
→ clip then optimizer.step.
```

Do not skip finite high-norm batches.

---

# 125. No skip-batch policy

Same sample usage.

---

# 126. No gradient normalization

Do not normalize every gradient to norm1.

Clipping only rescales when threshold is exceeded.

---

# 127. No loss normalization to match gradients

No.

---

# 128. No parameter update clipping

No.

---

# 129. Runtime diagnostics

Record:

```text
mean epoch seconds
median epoch seconds
total runtime
time to BEST
samples/sec optional.
```

GC1 clipping adds some norm/reduction overhead, but both conditions already compute preclip norm for diagnostics.

Runtime difference should be small and secondary.

---

# 130. Fair runtime instrumentation

Because GC0 now also computes preclip norm for diagnostics, both conditions incur norm-measurement overhead.

GC1 additionally performs rescaling only when required by implementation.

This keeps telemetry fairness stronger.

---

# 131. Memory diagnostics

Gradient clipping should not materially change model activation memory.

Peak memory should be similar.

Large differences suggest:

```text
instrumentation/logging issues
retained gradients
backend artifacts.
```

Diagnostic only.

---

# 132. Gradient logging memory caution

Do not store full gradient tensors every step.

Store scalar summaries only:

```text
global norm
clipping flag
optional per-group scalar summaries.
```

This avoids memory/disk explosion.

---

# 133. Per-layer gradient diagnostics are optional

If already supported without training mutation:

```text
input projection
attention blocks
FFN blocks
regression head
```

can have norm summaries.

Do not make them part of winner selection.

---

# 134. No backward hooks that alter gradients

Instrumentation hooks must be read-only.

---

# 135. Single-seed limitation

Mandatory:

```text
S17 uses seed42 only.
```

No mean±std.

---

# 136. Validation-only limitation

Mandatory:

```text
S17 winner is selected using Validation only.
```

No Test evidence.

---

# 137. Clipping × LR interaction limitation

LR* fixed.

A different LR may change how often clipping is useful.

Not tested.

---

# 138. Clipping × loss interaction limitation

LOSS* fixed.

If Huber selected, S17 conclusions are conditional on Huber.

If MSE selected, conditional on MSE.

No loss×clipping factorial grid.

---

# 139. Clipping × batch interaction limitation

B* fixed.

Gradient norm distributions can change with batch size.

No batch×clip grid.

---

# 140. Clipping × model-capacity interaction limitation

D*/H*/N*/F* fixed.

Larger/deeper models may exhibit different gradient norm behavior.

---

# 141. Clipping × epoch-budget interaction limitation

EPOCHS* fixed from S16.

No cap×clipping grid.

---

# 142. Clipping × target scaling interaction limitation

YS* fixed.

Target scaling affects loss/gradient magnitudes.

Thus threshold `1.0` is conditional on the selected target preprocessing and full model/optimizer setup.

---

# 143. Threshold limitation

Only:

```text
max_norm = 1.0
```

is tested versus OFF.

A result favoring GC0 does not prove that all clipping thresholds are harmful.

A result favoring GC1 does not prove `1.0` is globally optimal.

---

# 144. Sequential-selection limitation

By Phase39, Validation has influenced all S1–S17 decisions.

Later candidate synthesis, rolling-origin robustness and multi-seed checks remain necessary.

---

# 145. No Test access

Hard:

```text
Test loader not iterated
Test metrics absent
Test predictions absent.
```

---

# 146. No GC0.5 / GC2 / GC5

Not registered.

---

# 147. No per-layer threshold search

No.

---

# 148. No adaptive clipping threshold

No.

---

# 149. No gradient value clipping

No.

---

# 150. No optimizer change

No.

---

# 151. No loss change

No.

---

# 152. No target-scale change

No.

---

# 153. No emergency LR reduction for GC0

If GC0 is unstable, do not rescue it inside S17 by changing LR.

That would create a different condition.

---

# 154. Run failure policy

If GC0 has a technical/infrastructure failure:

```text
S17 incomplete
```

until documented technical rerun succeeds.

Do not automatically declare GC1 winner.

---

# 155. Genuine numerical failure policy

If GC0 repeatedly produces non-finite gradients under valid implementation:

```text
record NUMERICAL_INSTABILITY
```

and follow project failure policy.

Do not silently:

```text
skip batch
lower LR
enable clipping mid-run
reload earlier BEST and continue.
```

---

# 156. Technical rerun allowed

Only for documented:

```text
process interruption
hardware/software failure
corrupt checkpoint
artifact-write failure.
```

Use Experiment Registry rerun semantics.

---

# 157. Score-based rerun forbidden

Do not rerun GC0 because:

```text
RMSE looks poor
gradient norm looks high
early stopping happened early.
```

Do not retrain GC1.

---

# 158. Resume semantics

If GC0 run is interrupted and strict resume is used:

```text
model
optimizer
epoch
early-stop state
RNG
DataLoader generator
```

must be restored.

Gradient clipping config must remain:

```text
OFF.
```

---

# 159. Discrepancy taxonomy

```text
S16_REFERENCE_MISSING
S16_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
POOLING_DRIFT
ACTIVATION_DRIFT
BATCH_DRIFT
LEARNING_RATE_DRIFT
WEIGHT_DECAY_DRIFT
DROPOUT_DRIFT
DMODEL_DRIFT
HEAD_COUNT_DRIFT
HEAD_DIM_DRIFT
LAYER_COUNT_DRIFT
FFN_DIM_DRIFT
LOSS_DRIFT
HUBER_DELTA_DRIFT
EPOCH_CAP_DRIFT
PATIENCE_DRIFT
GRADIENT_CLIP_DEFINITION_MISMATCH
GC1_MAX_NORM_MISMATCH
GC1_NORM_TYPE_MISMATCH
GRADIENT_VALUE_CLIPPING_USED
PER_LAYER_CLIPPING_USED
PER_PARAMETER_CLIPPING_USED
ADAPTIVE_CLIPPING_USED
CLIP_WARMUP_USED
DOUBLE_CLIPPING
CLIP_ORDER_ERROR
CLIP_AFTER_OPTIMIZER_STEP
GC0_GRADIENT_MUTATION
GC0_NONFINITE_GUARD_MISSING
HIGH_NORM_STEP_SKIPPED
NONFINITE_STEP_EXECUTED
GRADIENT_NORM_DEFINITION_MISMATCH
GRADIENT_COVERAGE_MISMATCH
MODEL_ARCHITECTURE_DRIFT
PARAMETER_COUNT_MISMATCH
STATE_DICT_SCHEMA_MISMATCH
LOSS_CONFIG_MISMATCH
OPTIMIZER_DRIFT
LR_DRIFT
WD_DRIFT
INITIALIZATION_POLICY_DRIFT
INITIAL_STATE_MISMATCH
SAMPLE_ORDER_POLICY_DRIFT
RNG_POLICY_DRIFT
PRE_FIRST_CLIP_PREFIX_MISMATCH
NON_BINDING_CLIP_TRAJECTORY_MISMATCH
POPULATION_MISMATCH
TARGET_ID_MISMATCH
X_SCALER_MISMATCH
TARGET_SCALER_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
REFERENCE_RUN_MISMATCH
GC1_TELEMETRY_MISSING
RUN_FAILURE
NUMERICAL_INSTABILITY
CHECKPOINT_CONFIG_MISMATCH
CHECKPOINT_VERIFICATION_FAILURE
INCOMPLETE_SWEEP
RANKING_ERROR
EFFECT_CALCULATION_ERROR
RMSE_R2_RANKING_INCONSISTENCY
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 160. Status model

## PASS

```text
GC1 reference valid
GC0 fresh run valid
only clipping intervention differs
same model/data/loss/optimizer/budget
GC0 non-mutating gradient diagnostics verified
GC0 BEST verified
gradient activity diagnostics produced
winner selected
Phase40 reference generated
Test untouched.
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
MAE ranking divergence
GC1 clipping telemetry unavailable
initialization/order not verifiable
backend nondeterminism
GC1 never active
large GC0 threshold-exceedance
inherited warning.
```

## FAIL

Examples:

```text
GC0 silently steps nonfinite gradients
different LR/loss/batch
wrong clipping method
gradient measurement mutates GC0
reference mismatch
population mismatch
unresolved numerical failure
Test access.
```

---

# 161. Output directory

```text
artifacts/
└── sweeps/
    └── S17_gradient_clipping/
        ├── s17_gradient_clipping_sweep_manifest.json
        ├── s17_gradient_clipping_sweep_contract.json
        ├── s17_gradient_clipping_preflight_audit.csv
        ├── s17_run_matrix.csv
        ├── s17_gradient_clip_definition_audit.csv
        ├── s17_training_step_order_audit.csv
        ├── s17_architecture_invariance_audit.csv
        ├── s17_parameter_schema_audit.csv
        ├── s17_training_config_delta_audit.csv
        ├── s17_gradient_norm_definition_audit.json
        ├── s17_gradient_clip_unit_tests.csv
        ├── s17_gradient_mutation_tests.csv
        ├── s17_nonfinite_guard_tests.csv
        ├── s17_common_data_audit.csv
        ├── s17_initialization_audit.csv
        ├── s17_sample_order_audit.csv
        ├── s17_rng_policy_audit.csv
        ├── s17_gradient_coverage_audit.csv
        ├── s17_prefix_before_first_clip_audit.csv
        ├── s17_gc1_clipping_activity.csv
        ├── s17_gc0_threshold_exceedance.csv
        ├── s17_gradient_norm_distribution.csv
        ├── s17_gradient_excess_diagnostics.csv
        ├── s17_optimizer_budget_audit.csv
        ├── s17_gradient_clipping_run_provenance.csv
        ├── s17_gradient_clipping_metrics.csv
        ├── s17_gradient_clipping_effect.csv
        ├── s17_optimization_diagnostics.csv
        ├── s17_convergence_diagnostics.csv
        ├── s17_runtime_diagnostics.csv
        ├── s17_generalization_diagnostics.csv
        ├── s17_hypothesis_outcomes.csv
        ├── s17_gradient_clipping_findings.csv
        ├── s17_gradient_clipping_winner.json
        ├── s17_reference_update.json
        ├── s17_gradient_clipping_sweep_tests.csv
        ├── s17_gradient_clipping_discrepancies.json
        ├── s17_gradient_clipping_sweep_summary.json
        ├── s17_gradient_clipping_sweep_report.md
        ├── figures/
        │   ├── S17_01_validation_rmse_by_epoch.png
        │   ├── S17_02_validation_mae_by_epoch.png
        │   ├── S17_03_train_criterion_by_epoch.png
        │   ├── S17_04_preclip_gradient_norm_by_epoch.png
        │   ├── S17_05_threshold_exceedance_fraction.png
        │   ├── S17_06_gradient_norm_distribution.png
        │   ├── S17_07_best_validation_metrics.png
        │   ├── S17_08_first_clip_and_trajectory_divergence.png
        │   ├── S17_09_convergence_summary.png
        │   └── S17_10_generalization_gap_optional.png
        ├── README_S17_GRADIENT_CLIPPING_SWEEP.md
        └── phase_39_signoff.json
```

Optional:

```text
s17_prefix_before_first_clip_audit.csv
s17_generalization_diagnostics.csv
S17_08_first_clip_and_trajectory_divergence.png
S17_10_generalization_gap_optional.png
```

may be unavailable depending on reference telemetry.

New GC0 scientific run remains under:

```text
artifacts/runs/<run_id>/
```

No checkpoint duplication.

---

# 162. Required outputs

```text
O39.1  Sweep manifest
O39.2  Sweep contract
O39.3  Preflight audit
O39.4  Run matrix
O39.5  Gradient-clipping definition audit
O39.6  Training-step order audit
O39.7  Architecture invariance audit
O39.8  Parameter-schema audit
O39.9  Training-config delta audit
O39.10 Gradient-norm definition audit
O39.11 Gradient-clipping unit tests
O39.12 Gradient-mutation tests
O39.13 Non-finite guard tests
O39.14 Common-data audit
O39.15 Initialization audit
O39.16 Sample-order audit
O39.17 RNG-policy audit
O39.18 Gradient-coverage audit
O39.19 Optional prefix-before-first-clip audit
O39.20 GC1 clipping-activity diagnostics
O39.21 GC0 threshold-exceedance diagnostics
O39.22 Gradient-norm distribution
O39.23 Gradient-excess diagnostics
O39.24 Optimizer-budget audit
O39.25 Run provenance
O39.26 Reused GC1 reference
O39.27 Verified new GC0 run
O39.28 Primary metrics table
O39.29 Clipping effect table
O39.30 Optimization diagnostics
O39.31 Convergence diagnostics
O39.32 Runtime diagnostics
O39.33 Optional generalization diagnostics
O39.34 Hypothesis outcomes
O39.35 Findings
O39.36 Winner artifact
O39.37 Phase40 reference update
O39.38 Figures
O39.39 Sweep tests
O39.40 Discrepancy log
O39.41 Sweep summary
O39.42 Human-readable report
O39.43 README
O39.44 Phase sign-off
```

---

# 163. Run matrix

Create:

```text
s17_run_matrix.csv
```

Fields:

```text
sweep_id
gradient_clip_id
gradient_clip_enabled
gradient_clip_method
max_norm
norm_type
nonfinite_gradient_guard
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
weight_decay
dropout_probability
d_model
num_heads
head_dim
num_layers
ffn_dim
loss_id
loss_name
huber_delta_if_applicable
max_epochs
patience
population_fingerprint
feature_fingerprint
seed
model_config_id
criterion_config_id
optimizer_config_id
gradient_clip_config_id
training_config_id
status
```

---

# 164. Sweep manifest

Create:

```text
s17_gradient_clipping_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S17_GRADIENTCLIP-v1
sweep_id = S17_GRADIENT_CLIPPING
source_s16_winner_run_id
selected_model_config
selected_loss_config
selected_max_epochs
candidate_clip_configs = [GC0, GC1]
gc0_enabled = false
gc1_enabled = true
gc1_method = global_norm
gc1_max_norm = 1.0
gc1_norm_type = 2
nonfinite_gradient_guard_both = true
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
dropout_probability
d_model
num_heads
head_dim
num_layers
ffn_dim
loss_id
huber_delta_if_applicable
patience = 10
new_runs_required = 1
reused_runs = 1
swept_field = gradient_clipping
architecture_equality_expected = true
parameter_count_equality_expected = true
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = GC0_ON_EXACT_RMSE_TIE
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

# 165. Sweep contract

Create:

```text
s17_gradient_clipping_sweep_contract.json
```

Must state:

```text
Only gradient clipping changes.

GC0:
clipping OFF
finite-gradient guard ON.

GC1:
global L2 norm clipping
max_norm=1.0
after backward
before optimizer.step
finite-gradient guard ON.

Same global-norm definition for diagnostics.
Same model/loss/data/optimizer/LR/WD/batch.
Same epoch cap/patience.
Same seed.

GC1 reference reused.
GC0 fresh seed42 run.

GC0 high finite gradients are not skipped.
GC1 finite gradients >1 are rescaled.

No gradient-value clipping.
No adaptive clipping.
No threshold tuning.
No LR rescue.

Validation RMSE Wh selects winner.
Exact tie → GC0.
No Test.
```

---

# 166. Preflight audit

`s17_gradient_clipping_preflight_audit.csv` checks:

```text
phase38_pass
approved_for_phase39
s16_winner_valid
all prior selected fields locked
GC0 registered
GC1 registered
GC0 off
GC1 global norm
GC1 max_norm1
GC1 norm_type2
finite guard both
selected max_epochs fixed
patience10
loss fixed
optimizer fixed
LR/WD fixed
architecture fixed
population fixed
seed fixed
Training Engine fixed
Metric fixed
GC1 reuse candidate valid
Test lock
status
```

---

# 167. Gradient-clip definition audit

`s17_gradient_clip_definition_audit.csv`:

```text
clip_id
enabled
method
max_norm
norm_type
intervention_point
finite_guard
high_finite_norm_behavior
registered
status
```

Expected:

```text
GC0:
intervention_point=NONE
high_finite_norm_behavior=STEP_RAW

GC1:
intervention_point=POST_BACKWARD_PRE_STEP
high_finite_norm_behavior=RESCALE_THEN_STEP.
```

---

# 168. Training-step order audit

`s17_training_step_order_audit.csv`:

```text
clip_id
order_index
operation
expected
observed
gradient_mutating
status
```

Must prove clipping occurs only:

```text
after backward
before optimizer.step.
```

---

# 169. Architecture invariance audit

`s17_architecture_invariance_audit.csv`:

```text
component
config_gc0
config_gc1
shape_gc0
shape_gc1
equal
status
```

All equal.

---

# 170. Parameter-schema audit

`s17_parameter_schema_audit.csv`:

```text
parameter_or_buffer
shape_gc0
shape_gc1
dtype_equal
trainable_equal
equal
status
```

Expected all equal.

---

# 171. Training config delta audit

`s17_training_config_delta_audit.csv`:

```text
field
value_gc0
value_gc1
allowed_to_differ
expected_difference
status
```

Only clipping-specific fields may differ.

---

# 172. Gradient-norm definition audit

`s17_gradient_norm_definition_audit.json`:

```text
norm_scope = all_available_trainable_gradients
norm_type = 2
none_gradients_excluded = true
finite_required = true
measurement_gc0_mutates_gradients = false
gc1_reference_measurement_semantics
threshold = 1.0
active_if = preclip_norm > threshold
status
```

---

# 173. Gradient-clipping unit tests

`s17_gradient_clip_unit_tests.csv` should cover at least:

```text
GC0 config accepted
GC1 config accepted
GC1 max_norm exactly1
GC1 norm_type2
GC0 no clipping
GC0 finite guard active
same norm definition
synthetic active GC1 clips to <=1
synthetic inactive GC1 leaves grads unchanged
GC0 synthetic high norm remains high
GC0 norm measurement does not mutate gradients
GC1 active direction approximately preserved
nonfinite GC0 fails before step
nonfinite GC1 fails before step
clip after backward
clip before optimizer.step
exactly one clip call per optimizer step
no gradient-value clipping
no per-layer clipping
no adaptive threshold
same architecture
same parameter count
same output shapes
same loss config
same optimizer config
same epoch cap
same patience.
```

---

# 174. Gradient mutation tests

`s17_gradient_mutation_tests.csv`:

```text
test_case
clip_id
pre_norm
post_norm
gradient_fingerprint_before
gradient_fingerprint_after
mutation_expected
mutation_observed
status
```

Cases:

```text
GC0_LOW_NORM
GC0_HIGH_NORM
GC1_LOW_NORM
GC1_HIGH_NORM.
```

---

# 175. Non-finite guard tests

`s17_nonfinite_guard_tests.csv`:

```text
clip_id
nonfinite_type
detected
optimizer_step_blocked
exception_or_status
status
```

Cases:

```text
NaN
+Inf
-Inf
```

where feasible.

---

# 176. Common-data audit

`s17_common_data_audit.csv`:

```text
split
sample_count_gc0
sample_count_gc1
sample_ids_equal
ordered_ids_equal
feature_fingerprint_equal
x_scaler_equal
target_transform_equal
lookback_equal
batch_equal
population_equal
status
```

---

# 177. Initialization audit

`s17_initialization_audit.csv`:

```text
clip_id
seed
model_config_fingerprint
initialization_policy_fingerprint
initial_state_fingerprint
reference_available
exact_match_expected
match
status
```

---

# 178. Sample-order audit

`s17_sample_order_audit.csv`:

```text
epoch_or_probe
gc0_order_fingerprint
gc1_order_fingerprint
gc1_reference_available
same_order
status
```

---

# 179. RNG-policy audit

`s17_rng_policy_audit.csv`:

```text
clip_id
seed
python_rng_policy
numpy_rng_policy
torch_rng_policy
device_rng_policy
manual_reseed_per_batch
dropout_scope_fingerprint
same_execution_structure_before_clip
status
```

---

# 180. Gradient-coverage audit

`s17_gradient_coverage_audit.csv`:

```text
clip_id
epoch_or_probe
trainable_parameters
parameters_with_grad
parameters_without_grad
unexpected_none_gradients
coverage_fraction
status
```

---

# 181. Prefix-before-first-clip audit

`s17_prefix_before_first_clip_audit.csv`:

```text
gc1_first_clip_step
prefix_end_step
reference_telemetry_available
initial_state_match
sample_order_match
metric_or_state_comparison_available
comparison_tolerance
prefix_match_status
notes
```

Allowed:

```text
MATCH
MATCH_WITH_TOLERANCE
NOT_APPLICABLE_FIRST_STEP_CLIP
NOT_VERIFIABLE
MISMATCH.
```

---

# 182. GC1 clipping activity

`s17_gc1_clipping_activity.csv`:

```text
run_id
epoch
optimizer_steps
actual_clipped_steps
actual_clipping_fraction
mean_preclip_norm
median_preclip_norm
p90_preclip_norm
p95_preclip_norm
max_preclip_norm
first_clip_step_if_any
first_clip_epoch_if_any
binding_status
status
```

If telemetry missing:

```text
status = NOT_VERIFIABLE.
```

---

# 183. GC0 threshold exceedance

`s17_gc0_threshold_exceedance.csv`:

```text
run_id
epoch
optimizer_steps
steps_preclip_norm_gt1
counterfactual_would_clip_fraction
mean_preclip_norm
median_preclip_norm
p90_preclip_norm
p95_preclip_norm
max_preclip_norm
nonfinite_events
status
```

---

# 184. Gradient-norm distribution

`s17_gradient_norm_distribution.csv`:

```text
clip_id
scope
count
mean
median
p75
p90
p95
p99_optional
max
fraction_gt1
nonfinite_count
status
```

Scopes may include:

```text
ALL_STEPS
PRE50
POST50
```

where applicable.

---

# 185. Gradient-excess diagnostics

`s17_gradient_excess_diagnostics.csv`:

```text
clip_id
event_type
event_count
median_excess_over1
p90_excess_over1
p95_excess_over1
max_excess_over1
median_norm_ratio_to_threshold
max_norm_ratio_to_threshold
status
```

For:

```text
GC1_ACTUAL_CLIP_EVENTS
GC0_COUNTERFACTUAL_EXCEED_EVENTS.
```

---

# 186. Optimizer-budget audit

`s17_optimizer_budget_audit.csv`:

```text
clip_id
train_samples_per_epoch
batch_size
steps_per_complete_epoch
epochs_completed
total_optimizer_steps
best_epoch
steps_to_best
same_steps_per_completed_epoch
status
```

---

# 187. Run provenance

`s17_gradient_clipping_run_provenance.csv`:

```text
clip_id
run_id
source_type
source_phase
model_config_fingerprint
criterion_config_fingerprint
optimizer_config_fingerprint
gradient_clip_config_fingerprint
training_config_fingerprint
parameter_schema_fingerprint
parameter_count
feature_fingerprint
population_fingerprint
initial_state_fingerprint
sample_order_provenance
best_checkpoint_sha256
history_sha256
metric_artifact
gradient_telemetry_artifact
status
```

---

# 188. Primary metrics table

`s17_gradient_clipping_metrics.csv`:

```text
clip_id
clip_enabled
max_norm
run_id
source_type
epochs_completed
stop_reason
best_epoch
trainable_parameters
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

# 189. Gradient-clipping effect table

`s17_gradient_clipping_effect.csv`:

```text
gc0_run_id
gc1_run_id
gc0_rmse_wh
gc1_rmse_wh
rmse_delta_gc0_to_gc1_wh
rmse_improvement_pct
gc0_mae_wh
gc1_mae_wh
mae_delta_wh
gc0_r2
gc1_r2
r2_delta
rmse_mae_ranking_divergence
rmse_r2_ranking_consistent
gc1_binding_status
gc1_actual_clip_fraction
gc0_counterfactual_exceed_fraction
winner
status
```

---

# 190. Optimization diagnostics

`s17_optimization_diagnostics.csv`:

```text
clip_id
best_epoch
last_epoch
stop_reason
best_rmse_wh
last_rmse_wh
mean_preclip_grad_norm
max_preclip_grad_norm
fraction_norm_gt1
actual_clip_fraction_if_applicable
nonfinite_events
best_to_last_gap
status
```

---

# 191. Convergence diagnostics

`s17_convergence_diagnostics.csv`:

```text
clip_id
first_epoch_rmse_wh
best_epoch
best_rmse_wh
last_epoch
last_rmse_wh
early_stopped
epoch_cap_reached
steps_to_best
post_best_worsening_epochs
status
```

---

# 192. Runtime diagnostics

`s17_runtime_diagnostics.csv`:

```text
clip_id
device
epochs_completed
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
samples_per_second_optional
time_to_best_optional
runtime_comparable
status
```

---

# 193. Optional generalization diagnostics

`s17_generalization_diagnostics.csv`:

```text
clip_id
train_rmse_wh_at_best
validation_rmse_wh_at_best
rmse_gap_wh
train_mae_wh_at_best
validation_mae_wh_at_best
mae_gap_wh
interpretation
status
```

No Test.

---

# 194. Hypothesis outcomes

`s17_hypothesis_outcomes.csv`:

```text
hypothesis_id
comparison
expected_direction_or_pattern
observed_metrics
gradient_activity_context
outcome
interpretation
status
```

Allowed:

```text
SUPPORTED
NOT_SUPPORTED
INCONCLUSIVE_TIE
INCONCLUSIVE_TELEMETRY.
```

---

# 195. Findings artifact

`s17_gradient_clipping_findings.csv` possible codes:

```text
GC0_GAIN
GC1_GAIN
GRADIENT_CLIP_EXACT_TIE
CLIPPING_NON_BINDING
CLIPPING_BINDING
CLIPPING_ACTIVITY_UNKNOWN
GC1_NEVER_ACTIVE
GC1_FIRST_CLIP_EARLY
GC1_FIRST_CLIP_LATE
GC0_HIGH_THRESHOLD_EXCEEDANCE
GC0_LOW_THRESHOLD_EXCEEDANCE
PRE_FIRST_CLIP_PREFIX_MATCH
PRE_FIRST_CLIP_PREFIX_NOT_VERIFIABLE
NON_BINDING_TRAJECTORY_MATCH
NON_BINDING_TRAJECTORY_MISMATCH
GRADIENT_NORM_DIFFERENCE
CLIPPING_REDUCED_EXTREME_NORMS
GC0_NUMERICAL_INSTABILITY
CONVERGENCE_DIFFERENCE
EARLY_STOP_DIFFERENCE
METRIC_RANKING_DIVERGENCE
RMSE_R2_RANKING_INCONSISTENCY
INITIALIZATION_MATCH_VERIFIED
INITIALIZATION_NOT_VERIFIABLE
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
INHERITED_WARNING.
```

---

# 196. Winner artifact

`s17_gradient_clipping_winner.json` minimum:

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
weight_decay
dropout_probability
d_model
num_heads
head_dim
num_layers
ffn_dim
loss_id
loss_name
huber_delta_if_applicable
max_epochs
patience
selection_metric
selection_direction
tie_rule
winner_clip_id
winner_clip_enabled
winner_max_norm_if_applicable
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_clip_id
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
gc1_binding_status
gc1_actual_clip_fraction
gc0_counterfactual_exceed_fraction
prefix_before_first_clip_status
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 197. Phase40 reference update

`s17_reference_update.json`:

```text
previous_reference_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
dropout_probability
d_model
num_heads
head_dim
num_layers
ffn_dim
loss_id
loss_name
huber_delta_if_applicable
max_epochs
patience
previous_gradient_clip_id=GC1
selected_gradient_clip_id
selected_gradient_clip_enabled
selected_max_norm_if_applicable
winner_run_id
winner_config_fingerprint
winner_rmse_wh
current_revin_id=RN0
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase40
```

---

# 198. Phase40 handoff logic

Phase40 tests:

```text
RN0 = RevIN OFF
RN1 = RevIN ON
```

while holding S17-selected clipping setting fixed.

S17 winner still uses:

```text
RN0.
```

Therefore normally:

```text
reuse S17 winner as RN0 reference
train one RN1 candidate.
```

if exact match.

---

# 199. RevIN handoff nuance

If GC0 wins:

```text
Phase40 RN0 and RN1 both use clipping OFF.
```

If GC1 wins:

```text
both use global norm max_norm1.
```

Do not revert clipping setting.

---

# 200. Figures

Recommended:

```text
S17_01_validation_rmse_by_epoch.png
S17_02_validation_mae_by_epoch.png
S17_03_train_criterion_by_epoch.png
S17_04_preclip_gradient_norm_by_epoch.png
S17_05_threshold_exceedance_fraction.png
S17_06_gradient_norm_distribution.png
S17_07_best_validation_metrics.png
S17_08_first_clip_and_trajectory_divergence.png
S17_09_convergence_summary.png
S17_10_generalization_gap_optional.png
```

---

# 201. Primary figure

```text
S17_01_validation_rmse_by_epoch.png
```

Overlay:

```text
GC0
GC1
```

with:

```text
BEST markers
stop markers.
```

---

# 202. Preclip gradient norm figure

`S17_04_preclip_gradient_norm_by_epoch.png`:

```text
epoch x
mean/p95/max preclip norm
horizontal threshold line at 1.0
```

for GC0/GC1.

Threshold is shown for GC0 as counterfactual reference, not actual clipping.

---

# 203. Threshold-exceedance figure

`S17_05_threshold_exceedance_fraction.png`:

```text
GC1 actual clipping fraction
GC0 counterfactual would-clip fraction
```

Clearly label different semantics.

---

# 204. Gradient distribution figure

Use log y/x scale only if appropriate and clearly labeled.

Do not transform values in a way that hides threshold1.

---

# 205. First-clip divergence figure

If GC1 telemetry supports first clipping:

```text
mark first GC1 clipping epoch/step
```

and show RMSE trajectories around that point.

Diagnostic only.

---

# 206. Safe interpretation if GC0 wins

> Under the frozen current model, loss and optimizer configuration, disabling global norm clipping achieved lower Validation RMSE than `max_norm=1.0`, while remaining numerically valid under the same non-finite gradient guard.

Do not claim clipping is universally harmful.

---

# 207. Safe interpretation if GC1 wins

> Global gradient-norm clipping at `max_norm=1.0` achieved lower Validation RMSE than clipping-off under the frozen current configuration.

If clipping was active, report its exact frequency as optimization context.

---

# 208. Safe interpretation if exact tie

> GC0 and GC1 produced exactly equal full-precision Validation RMSE; clipping was disabled by the predeclared parsimony tie rule.

---

# 209. Safe interpretation if GC1 never clips

> The registered `max_norm=1.0` threshold was non-binding in the available GC1 training telemetry because no optimizer step exceeded the clipping threshold.

Do not claim:

```text
all future runs will never need clipping.
```

---

# 210. Safe interpretation if GC0 has many threshold exceedances but wins

> GC0 frequently exceeded the `1.0` threshold counterfactually yet achieved lower Validation RMSE, indicating that suppressing those finite large-norm updates was not beneficial under the current selected configuration.

Conditional single-seed statement only.

---

# 211. Safe interpretation if GC0 becomes numerically unstable

Do not treat as an ordinary score comparison.

Report technical/numerical failure according to run policy.

---

# 212. Interpretation prohibitions

Do not claim:

```text
gradient norm >1 is automatically bad
clipping always stabilizes Transformers
GC0 is unstable because its norm is larger
GC1 is better because its norm is bounded
GC0 is better because updates are larger
Huber makes clipping unnecessary universally
clipping is regularization equivalent to dropout/WD
```

without evidence.

---

# 213. Sweep summary

Create:

```text
s17_gradient_clipping_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
selected_model_config
selected_loss_config
selected_epoch_cap
reference_run_id
gc0_run_id
new_runs
reused_runs
candidate_clip_configs
primary_metric
metrics_by_clip
clip_effect
gc1_clipping_activity
gc0_threshold_exceedance
gradient_norm_distribution
gradient_excess_diagnostics
prefix_before_first_clip
optimization_diagnostics
convergence_diagnostics
runtime_diagnostics
initialization_match
sample_order_match
winner
winner_margin
inherited_warnings
phase40_reference
test_status
overall_status
```

---

# 214. Human-readable report

Create:

```text
s17_gradient_clipping_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S16
3. GC0/GC1 definitions
4. Global norm semantics
5. Exact training-step intervention point
6. Non-finite gradient policy
7. Frozen-variable contract
8. Architecture/parameter invariance
9. Same data/initialization/sample-order fairness
10. GC1 reference provenance
11. GC0 run provenance
12. Gradient norm measurement integrity
13. GC1 actual clipping activity
14. GC0 counterfactual threshold exceedance
15. Prefix-before-first-clip audit
16. Validation metrics
17. Clipping effect
18. Gradient norm/excess diagnostics
19. Learning-curve/convergence analysis
20. Numerical stability analysis
21. Runtime context
22. S17 winner
23. Interpretation cautions
24. Interaction/threshold limitations
25. Phase40 handoff
```

---

# 215. README

Create:

```text
README_S17_GRADIENT_CLIPPING_SWEEP.md
```

Must explain:

```text
Purpose
S16 winner handoff
GC0 vs GC1
global L2 norm definition
max_norm1 semantics
clip after backward/before step
GC0 non-mutating norm diagnostics
finite guard in both conditions
actual clipping vs counterfactual exceedance
first active clipping concept
non-binding clipping case
same model/loss/optimizer/budget
no threshold tuning
no gradient-value/per-layer/adaptive clipping
GC1 reference reuse
winner/tie rule
interaction limitations
Phase40 handoff
No Test.
```

---

# 216. Recommended notebook structure

```text
Cell 39.1  Phase title
Cell 39.2  Verify Phase38 sign-off
Cell 39.3  Declare SWEEP_S17_GRADIENTCLIP-v1
Cell 39.4  Load S16 winner/reference
Cell 39.5  Freeze all prior selected fields
Cell 39.6  Define GC0/GC1
Cell 39.7  Lock global norm semantics/max_norm1
Cell 39.8  Build run matrix
Cell 39.9  Audit training-step order
Cell 39.10 Audit architecture/parameter invariance
Cell 39.11 Audit training-config delta
Cell 39.12 Audit gradient norm definition
Cell 39.13 Run synthetic clipping unit tests
Cell 39.14 Run GC0 mutation test
Cell 39.15 Run non-finite guard tests
Cell 39.16 Audit common population
Cell 39.17 Audit initialization/sample-order/RNG policy
Cell 39.18 Audit gradient coverage
Cell 39.19 Verify GC1 reference reuse
Cell 39.20 Extract GC1 clipping telemetry if available
Cell 39.21 Register GC0
Cell 39.22 Reseed + fresh GC0 loaders/model/criterion/optimizer
Cell 39.23 Train GC0 through same engine with clipping disabled
Cell 39.24 Verify GC0 BEST
Cell 39.25 Build run provenance
Cell 39.26 Build optimizer-budget audit
Cell 39.27 Build GC0 threshold-exceedance diagnostics
Cell 39.28 Build GC1 clipping-activity diagnostics
Cell 39.29 Build gradient-norm distribution
Cell 39.30 Build gradient-excess diagnostics
Cell 39.31 Optional prefix-before-first-clip audit
Cell 39.32 Build primary metrics
Cell 39.33 Compute GC0→GC1 effect
Cell 39.34 Build optimization diagnostics
Cell 39.35 Build convergence diagnostics
Cell 39.36 Build runtime diagnostics
Cell 39.37 Optional generalization diagnostic
Cell 39.38 Evaluate hypotheses
Cell 39.39 Generate figures
Cell 39.40 Generate findings
Cell 39.41 Select winner
Cell 39.42 Write winner JSON
Cell 39.43 Write Phase40 reference update
Cell 39.44 Run S17 tests/discrepancies
Cell 39.45 Write summary/report
Cell 39.46 Register artifacts/checksums
Cell 39.47 Write README
Cell 39.48 Phase sign-off
```

---

# 217. Execution flow

```text
Verify Phase38
→ Load S16 winner
→ Freeze model/data/loss/epoch cap/optimizer
→ Define GC0 vs GC1
→ Lock global norm max_norm1 semantics
→ Verify same population
→ Verify architecture/parameter equality
→ Unit-test clipping/non-mutation/nonfinite guards
→ Verify GC1 reference
→ Reuse GC1
→ Register GC0
→ Seed42 + fresh GC0 objects
→ Train GC0 with raw finite gradients
→ Verify GC0 BEST
→ Measure GC0 threshold exceedance
→ Recover GC1 clipping activity
→ Audit pre-first-clip/non-binding trajectory when possible
→ Compare verified Validation RMSE
→ Select min RMSE
→ Exact tie prefer GC0
→ Update Phase40 reference
→ Write artifacts/sign-off
```

---

# 218. Fail-fast order

Before expensive GC0 training:

```text
1. Phase38 sign-off
2. S16 winner identity
3. freeze all prior selected fields
4. GC0/GC1 registry
5. GC1 max_norm1/norm_type2
6. GC0 clipping truly off
7. finite gradient guard both
8. exact training-step order
9. same global norm definition
10. GC0 norm helper non-mutating
11. architecture/parameter equality
12. same loss
13. same optimizer/LR/WD
14. same max_epochs/patience
15. same population
16. initialization/sample-order/RNG policy
17. GC1 reuse eligibility
18. Test firewall
19. Registry readiness
```

---

# 219. Critical technical note: GC0 is not “no gradient checks”

GC0 still requires:

```text
finite-gradient validation.
```

The swept factor is gradient rescaling, not basic numerical safety.

---

# 220. Critical technical note: log PRE-CLIP norm for both

Without preclip norm on GC0, mechanism comparison is incomplete.

Do not log only postclip GC1 norm because it will trivially be bounded.

---

# 221. Critical technical note: actual clip fraction and counterfactual exceedance are different

Correct:

```text
GC1 actual clipping fraction
GC0 would-clip-at-1 fraction.
```

Do not call both “clipping fraction”.

---

# 222. Critical technical note: first active clip marks causal trajectory divergence point

Before that point, the two algorithms are the same if all reproducibility conditions match.

After it, trajectories may legitimately diverge.

---

# 223. Critical technical note: if clipping never activates, it cannot explain a performance difference

Any material difference then requires reproducibility/provenance interpretation.

---

# 224. Critical technical note: threshold1 is not universally meaningful

Its effect depends on:

```text
target scaling
loss
batch
model capacity
LR
training stage.
```

S17 conclusions are conditional.

---

# 225. Critical technical note: do not rescue GC0

If GC0 has high finite norms:

```text
still step normally.
```

Do not lower LR or turn clipping on halfway.

---

# 226. Critical technical note: do not skip finite high-norm steps

That would create another optimizer intervention.

---

# 227. Critical technical note: instrumentation must not change gradients

Gradient telemetry is read-only except GC1's registered clipping operation.

---

# 228. Winner verification checklist

Before writing `s17_gradient_clipping_winner.json`:

```text
[ ] GC1 valid reused reference.
[ ] GC0 valid completed scientific run.
[ ] Same FV*/YS*/L*/P*/A*/B*/LR*/WD*/DR*/D*/H*/N*/F*/LOSS*/EPOCHS*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same target scaler/checksum.
[ ] Same architecture fingerprint.
[ ] Same parameter count/schema.
[ ] Same criterion config.
[ ] Same optimizer config.
[ ] Same patience/min_delta.
[ ] GC0 clipping truly disabled.
[ ] GC1 global norm max_norm1.
[ ] Same norm definition.
[ ] Finite guard both.
[ ] GC0 norm telemetry non-mutating.
[ ] No per-layer/value/adaptive clipping.
[ ] No LR/WD/batch rescue.
[ ] Initialization policy audited.
[ ] Sample-order policy audited.
[ ] RNG policy audited.
[ ] GC1 telemetry recovered or marked unavailable.
[ ] GC0 preclip norms logged.
[ ] GC0 threshold exceedance computed.
[ ] GC1 actual clip fraction computed if telemetry exists.
[ ] First GC1 clip identified if possible.
[ ] Prefix-before-first-clip audited if possible.
[ ] GC0 BEST verified.
[ ] GC1 BEST provenance valid.
[ ] Predictions finite.
[ ] Full-precision Validation RMSE used.
[ ] RMSE/R² consistency checked.
[ ] MAE divergence recorded if present.
[ ] Gradient diagnostics not used to override RMSE.
[ ] Exact tie rule respected.
[ ] Numerical failure not silently converted into score loss.
[ ] Phase40 reference generated.
[ ] No Test.
```

---

# 229. Phase40 handoff

Phase40 receives:

```text
s17_gradient_clipping_winner.json
s17_reference_update.json
winner_run_id
winner_config_fingerprint
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
dropout_probability
d_model
num_heads
head_dim
num_layers
ffn_dim
loss_id
huber_delta_if_applicable
max_epochs
patience
selected_gradient_clip_id
selected_gradient_clip_enabled
selected_max_norm_if_applicable
current_revin=RN0
population_fingerprint
metric_version.
```

and changes only:

```text
RevIN OFF vs ON.
```

---

# 230. Reference reuse in Phase40

S17 winner already uses:

```text
RN0.
```

Therefore normally:

```text
reuse S17 winner as RN0 reference
train RN1.
```

---

# 231. Relationship with Phase40 RevIN sweep

Do not alter clipping during RevIN evaluation.

S17-selected clipping is frozen.

---

# 232. Relationship with Phase41 Boundary protocol check

Boundary protocol remains WB0 throughout S17.

No WB1 here.

---

# 233. Relationship with Phase42 Candidate synthesis

Keep both GC0/GC1 registry entries and diagnostics.

Do not delete loser.

---

# 234. Relationship with Phase44 Rolling-origin robustness

Gradient norm/clipping behavior may vary by temporal fold.

S17 winner is current Validation-selection evidence only.

---

# 235. Relationship with Phase46 Multi-seed

S17 uses seed42 only.

Clipping frequency and winner may vary across seeds.

---

# 236. Relationship with final reporting

Final Methods should report selected:

```text
gradient clipping enabled/disabled
max_norm if enabled.
```

If enabled:

```text
global L2 norm clipping before optimizer step.
```

---

# 237. Reproducibility metadata

New GC0 run records:

```text
run_id
seed
environment
device
feature fingerprint
X scaler checksum
target scaler checksum
target scaling ID
lookback
pooling
activation
batch
learning rate
weight decay
dropout
dropout scope
d_model
num_heads
head_dim
num_layers
ffn_dim
loss ID
Huber delta if applicable
max_epochs
patience
gradient_clip_id=GC0
gradient_clip_enabled=false
nonfinite_gradient_guard=true
gradient_norm_definition
optimizer group policy
architecture fingerprint
parameter schema fingerprint
initialization fingerprint
sample-order provenance
population fingerprint
Training Engine fingerprint
BEST checkpoint checksum
history checksum
metric checksum
gradient telemetry checksum
stop reason
epochs completed
best epoch.
```

---

# 238. No fabricated outputs

Do not pre-fill:

```text
GC1 clip fraction
GC0 exceedance fraction
first clipped step
gradient percentiles
winner
RMSE
best epoch
runtime
nonfinite events.
```

Allowed pre-runtime:

```text
GC0 definition
GC1 max_norm=1
norm_type=2
winner/tie policy.
```

---

# 239. Phase sign-off

Create:

```text
phase_39_signoff.json
```

Minimum:

```text
phase = 39
phase_name = S17 Gradient-clipping sweep
sweep_version
sweep_id
source_s16_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
dropout_probability
d_model
num_heads
head_dim
num_layers
ffn_dim
loss_id
loss_name
huber_delta_if_applicable
max_epochs
patience
gc0_run_id
gc1_reference_run_id
new_run_ids
reused_run_ids
gc1_max_norm
gc1_norm_type
gc1_binding_status
gc1_actual_clip_fraction
gc0_counterfactual_exceed_fraction
prefix_before_first_clip_status
gradient_norm_definition_audit_status
gc0_gradient_mutation_test_status
nonfinite_guard_status
winner_clip_id
winner_clip_enabled
winner_max_norm_if_applicable
winner_run_id
winner_rmse_wh
population_fingerprint
metric_version
architecture_invariance_status
initialization_audit_status
sample_order_match_status
inherited_warnings
test_status
approved_for_phase40
overall_status
created_at
```

---

# 240. Acceptance checklist

```text
[ ] Phase38 valid.
[ ] approved_for_phase39=true.
[ ] SWEEP_S17_GRADIENTCLIP-v1 declared.
[ ] S16 winner loaded.
[ ] All prior selected fields fixed.
[ ] GC0 exactly clipping OFF.
[ ] GC1 exactly global norm clipping.
[ ] GC1 max_norm exactly1.0.
[ ] GC1 norm_type exactly2.
[ ] No extra clipping threshold candidate.
[ ] GC0 finite guard ON.
[ ] GC1 finite guard ON.
[ ] Same global norm definition.
[ ] Parameters with grad=None handled consistently.
[ ] Gradient coverage audited.
[ ] Clip after backward.
[ ] Clip before optimizer.step.
[ ] Exactly one clip call per GC1 optimizer step.
[ ] No double clipping.
[ ] No gradient-value clipping.
[ ] No per-layer clipping.
[ ] No per-parameter clipping.
[ ] No adaptive clipping.
[ ] No clipping warmup.
[ ] GC0 norm helper read-only.
[ ] GC0 mutation test PASS.
[ ] GC1 active synthetic test PASS.
[ ] GC1 inactive synthetic test PASS.
[ ] Nonfinite GC0 step blocked.
[ ] Nonfinite GC1 step blocked.
[ ] High finite GC0 step not skipped.
[ ] Same architecture fingerprint.
[ ] Same parameter count.
[ ] Same state_dict schema.
[ ] Same loss.
[ ] Same Huber delta if applicable.
[ ] Same Train IDs.
[ ] Same Validation IDs.
[ ] Same WINDOWPOP-v1.
[ ] Same feature fingerprint.
[ ] Same X scaler.
[ ] Same target scaler.
[ ] Same batch.
[ ] Same AdamW.
[ ] Same optimizer groups.
[ ] Same LR.
[ ] Same WD.
[ ] Same dropout.
[ ] Same max_epochs from S16.
[ ] Same patience10.
[ ] Same min_delta0.
[ ] Same early-stop metric RMSE Wh.
[ ] Same BEST metric RMSE Wh.
[ ] Same scheduler=None.
[ ] Same warmup=None.
[ ] Same accumulation1.
[ ] Same precision policy.
[ ] Same seed42.
[ ] Initialization policy same.
[ ] Initial fingerprint compared if reference available.
[ ] Sample-order policy same.
[ ] RNG policy same.
[ ] GC1 reference exact-match.
[ ] GC1 not retrained.
[ ] GC1 clipping telemetry loaded or marked unavailable.
[ ] GC0 registered before training.
[ ] GC0 fresh loaders/model/criterion/optimizer.
[ ] No warm-start.
[ ] No optimizer-state reuse.
[ ] No LR rescue.
[ ] No clipping enabled mid-run.
[ ] No high-norm skip-step.
[ ] GC0 trained via same engine branch.
[ ] GC0 preclip norm logged.
[ ] GC0 threshold-exceedance fraction generated.
[ ] GC1 actual clipping fraction generated if possible.
[ ] First GC1 clip identified if possible.
[ ] Pre-first-clip prefix audit generated if possible.
[ ] Non-binding case handled correctly.
[ ] GC0 BEST verified.
[ ] GC1 BEST provenance valid.
[ ] Predictions finite.
[ ] Official metrics in Wh.
[ ] Full-precision RMSE used.
[ ] GC0→GC1 effect computed.
[ ] Winner=min RMSE.
[ ] Exact tie=GC0.
[ ] Gradient diagnostics do not override RMSE.
[ ] Runtime does not override RMSE.
[ ] RMSE/R² ranking consistency checked.
[ ] MAE divergence recorded if present.
[ ] Gradient norm distribution generated.
[ ] Gradient excess diagnostics generated.
[ ] Optimization diagnostics generated.
[ ] Convergence diagnostics generated.
[ ] Runtime diagnostics generated.
[ ] Optional generalization no Test.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] threshold limitation documented.
[ ] clip×loss interaction documented.
[ ] clip×LR interaction documented.
[ ] clip×batch interaction documented.
[ ] clip×target scaling interaction documented.
[ ] clip×capacity interaction documented.
[ ] Inherited warnings propagated.
[ ] Winner artifact generated.
[ ] Phase40 reference update generated.
[ ] RN0 reuse identified.
[ ] No threshold search.
[ ] No score-based rerun.
[ ] No failed/SANITY run ranked.
[ ] No Test access.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 241. Acceptance criteria

Phase 39 chỉ PASS khi:

```text
S16-selected model/data/loss/optimizer/epoch-cap configuration is fixed.

Exactly GC0 and GC1 are represented.

GC0 means no finite-gradient rescaling.

GC1 means global L2 norm clipping with max_norm=1.0.

Both retain a non-finite-gradient fail guard.

Clipping occurs only after backward and before optimizer.step.

GC0 gradient-norm diagnostics are proven non-mutating.

Same Train/Validation IDs and WINDOWPOP-v1 are used.

Same architecture/parameter schema are verified.

Same loss, AdamW, LR, WD, batch, epoch cap, patience and seed are used.

GC1 exact S16 reference is reused.

GC0 is one fresh seed42 run.

No adaptive/value/per-layer clipping is introduced.

No LR rescue, skip-step policy, warm-start or optimizer-state reuse occurs.

GC1 actual clipping activity is recovered or transparently marked unavailable.

GC0 counterfactual threshold exceedance is measured.

Pre-first-active-clip or non-binding trajectory behavior is audited where provenance allows.

GC0 BEST is verified.

Wh-space Validation RMSE selects winner.

Exact RMSE tie selects GC0.

Gradient stability diagnostics remain secondary.

Phase40 reference is generated.

Test remains untouched.
```

---

# 242. Failure conditions

Phase39 FAIL if:

```text
wrong S16 winner used

any prior selected setting changes

GC0 still clips gradients

GC1 uses wrong threshold/method

gradient value clipping substitutes norm clipping

GC0 norm measurement mutates gradients

nonfinite guard is disabled for GC0

finite high-norm GC0 steps are skipped

clipping occurs before backward or after optimizer step

clipping occurs twice

different LR/loss/batch/epoch cap is used

optimizer differs

sample population differs

GC1 mismatch is ignored

GC1 is retrained and favorable rerun selected

GC0 warm-starts from GC1

optimizer state is reused

adaptive threshold is introduced

RMSE rounded before ranking

gradient norm/clipping fraction overrides lower RMSE

one condition fails but winner is still declared as ordinary score winner

score-based rerun occurs

Test is accessed.
```

---

# 243. Common mistakes

## 243.1 GC0 tắt luôn non-finite check

Sai. GC0 chỉ tắt rescaling.

## 243.2 GC0 không log gradient norm

Thiếu mechanism audit.

## 243.3 Dùng postclip norm của GC1 so với preclip norm của GC0

Sai comparison.

## 243.4 Gọi GC0 fraction(norm>1) là clipping fraction

Sai. Đó là counterfactual threshold exceedance.

## 243.5 Dùng `clip_grad_value_`

Sai factor.

## 243.6 Clip từng layer riêng

Sai factor.

## 243.7 Giảm LR cho GC0 vì gradient lớn

Sai controlled sweep.

## 243.8 Skip batch khi norm >1 ở GC0

Sai. High finite norm phải được step raw.

## 243.9 Thấy GC1 clip 0 lần nhưng hai run khác nhau rồi nói clipping gây khác biệt

Sai; phải audit reproducibility.

## 243.10 Chọn GC1 vì “an toàn hơn” dù RMSE cao hơn

Sai S17 selection rule.

## 243.11 Chọn GC0 vì nhanh hơn dù RMSE cao hơn

Sai.

## 243.12 Thử max_norm .5/2 sau khi xem kết quả

Hidden adaptive search.

## 243.13 Nếu GC0 NaN thì tự bật clipping giữa run

Sai.

## 243.14 Dùng Test để chọn clipping

Forbidden.

---

# 244. Recommended execution pseudocode

```text
load_phase38_signoff()
assert_approved_for_phase39()

s16 = load_s16_winner()
freeze_all_prior_selected_fields(s16)

clip_configs = {
    "GC0": {
        "enabled": False,
        "method": None,
        "max_norm": None,
        "finite_guard": True
    },
    "GC1": {
        "enabled": True,
        "method": "global_norm",
        "max_norm": 1.0,
        "norm_type": 2,
        "finite_guard": True
    }
}

audit_common_data_population()
audit_architecture_invariance()
audit_parameter_schema_equality()
audit_loss_optimizer_budget_invariance()
audit_gradient_norm_definition()
audit_training_step_order()

run_synthetic_clipping_tests()
run_gc0_nonmutation_test()
run_nonfinite_guard_tests()

gc1_reference = resolve_s16_winner_run()

assert_exact_s17_reference_match(
    gc1_reference,
    clip_id="GC1",
    max_norm=1.0,
    norm_type=2
)

gc1_telemetry = load_gc1_gradient_telemetry_if_available()

register_gc0_run()

reseed(42)

loaders = build_fresh_loaders(
    feature_variant=FV*,
    target_scaling=YS*,
    lookback=L*,
    train_batch_size=B*,
    population="WINDOWPOP-v1"
)

model = build_fresh_transformer(
    exact_selected_architecture
)

criterion = build_exact_selected_loss(
    LOSS*,
    huber_delta_if_needed=1.0
)

optimizer = build_fresh_adamw(
    model=model,
    lr=LR*,
    weight_decay=WD*,
    group_policy=FROZEN_S9_GROUP_POLICY
)

gc0_result = TRAINING_ENGINE_v1.fit(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    max_epochs=EPOCHS*,
    patience=10,
    gradient_clip_enabled=False,
    nonfinite_gradient_guard=True,
    log_preclip_global_grad_norm=True
)

verify_best_checkpoint(
    gc0_result,
    clip_id="GC0"
)

gc0_exceed = build_gc0_threshold_exceedance(
    gc0_result,
    threshold=1.0
)

gc1_activity = build_gc1_clipping_activity(
    gc1_telemetry,
    threshold=1.0
)

if gc1_activity.first_clip_step_available:
    audit_prefix_before_first_clip(
        gc0_result,
        gc1_reference,
        gc1_activity.first_clip_step
    )

results = {
    "GC0": gc0_result,
    "GC1": gc1_reference
}

metrics = build_verified_s17_metrics(results)
effect = compute_gc0_to_gc1_effect(metrics)
assert_rmse_r2_ranking_consistent(metrics)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer="GC0"
)

write_gradient_distributions()
write_gradient_excess_diagnostics()
write_optimization_diagnostics()
write_convergence_diagnostics()
write_runtime_diagnostics()
write_hypothesis_outcomes()
write_findings()
write_s17_winner(winner)
write_phase40_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 245. Definition of Done

\[
\boxed{
One\ Fixed\ Model/Data/Loss/Budget
+
Two\ Gradient\ Policies
+
One\ Fresh\ GC0
+
One\ Valid\ Reused\ GC1
+
Same\ Finite\ Safety\ Guard
+
Non\text{-}Mutating\ GC0\ Diagnostics
+
Actual/Counterfactual\ Clip\ Analysis
+
Verified\ BEST\ Metrics
+
S17\ Winner
+
Phase40\ Reference
+
No\ Test
}
\]

---

# 246. Final status contract

```text
PHASE 39 tests gradient clipping only.

Candidates:

GC0:
clipping OFF
high finite gradient → optimizer.step raw
nonfinite gradient → FAIL

GC1:
global L2 norm clipping
max_norm=1.0
after backward
before optimizer.step
nonfinite gradient → FAIL.

Frozen:
all winners from S1–S16
same architecture
same parameters
same data/population
same loss
same AdamW
same LR
same WD
same batch
same dropout
same selected max_epochs
same patience10
same seed42
same Training Engine
same Metric version.

GC1:
reuse S16 winner if exact match.

GC0:
one fresh seed42 run.

Diagnostics:
preclip global norm both
GC1 actual clipping fraction
GC0 counterfactual would-clip fraction
first GC1 active clip
gradient excess distribution
nonfinite events
optional prefix-before-first-clip audit.

Hard:
GC0 norm measurement must not modify gradients.
No value/per-layer/adaptive clipping.
No LR rescue.
No skip finite high-norm steps.
No threshold tuning.
No warm-start.
No optimizer-state reuse.

Selection:
minimum verified BEST Validation RMSE Wh.

Exact tie:
prefer GC0.

If GC1 never clips:
clipping is non-binding under current run;
material trajectory differences require reproducibility investigation.

No Test.

After SWEEP_S17_GRADIENTCLIP-v1 PASS:
proceed to
PHASE 40 — S18 RevIN sweep.
```

---

# 247. Final check

Correct:

```text
Load S16 winner
→ freeze everything except clipping
→ GC0 vs GC1(max_norm1)
→ keep finite guard both
→ verify norm semantics
→ reuse GC1
→ fresh GC0
→ log preclip norms without mutation
→ verify GC0 BEST
→ compare GC1 actual clipping vs GC0 counterfactual exceedance
→ compare verified Validation RMSE
→ exact tie GC0
→ Phase40 reference
```

Incorrect:

```text
GC0 = no safety checks
→ lower LR for GC0
→ skip high-norm batches
→ use gradient value clipping
→ tune max_norm
→ choose by gradient norm instead of RMSE
→ inspect Test
```

Chỉ sau khi `SWEEP_S17_GRADIENTCLIP-v1` được sign-off mới chuyển sang **PHASE 40 — S18 RevIN sweep**.
