# PHASE 35 — S13 LAYER SWEEP

## Kế hoạch controlled sweep cho số lượng Transformer Encoder Layers

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S12_HEADS-v1`  
**Sweep ID:** `S13_LAYERS`  
**Output version:** `SWEEP_S13_LAYERS-v1`  
**Phase trước:** `Phase_34_S12_Head_sweep.md`

---

# 1. Vai trò của Phase 35

Phase 35 là controlled experiment thứ mười ba trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target scaling, lookback, pooling, activation, batch size, learning rate, weight decay, dropout, `d_model`, số attention heads, FFN width, loss, epoch budget, gradient clipping, sample population và seed đã được khóa từ Phase 34, Transformer Encoder nên dùng `1 layer` hay `2 layers`?

Phase 35 chỉ thay đúng một registered hyperparameter:

```text
num_layers
```

với hai condition:

```text
N1 = 1 Transformer Encoder layer
N2 = 2 Transformer Encoder layers
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Depth\ Factor
+
Same\ Width/Heads/FFN
+
Expected\ Parameter\ Count\ Change
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

# 2. Vị trí Phase 35 trong master execution plan

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
```

Phase 35 không được quay lại thay:

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
ffn_dim
loss
epoch cap
gradient clipping
RevIN
boundary protocol
```

---

# 3. Câu hỏi nghiên cứu của S13

Phase 35 phải trả lời:

```text
1. N1 hay N2 tạo Validation RMSE Wh thấp hơn?

2. Một Encoder layer đã đủ capacity/compositional depth cho task chưa?

3. Layer thứ hai có cải thiện forecasting hay tạo thêm overfitting/optimization burden?

4. Parameter count tăng bao nhiêu khi chuyển N1 → N2?

5. Checkpoint size/runtime/memory context tăng như thế nào?

6. Learning curves, best epoch và early stopping behavior thay đổi ra sao?

7. Gradient norm/clipping behavior thay đổi như thế nào?

8. MAE và R² có cùng ranking với RMSE không?

9. Depth nào trở thành current reference cho Phase 36?
```

---

# 4. Current reference từ Phase 34

Phase 35 phải load:

```text
s12_head_winner.json
s12_reference_update.json
phase_34_signoff.json
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
WD* = selected weight decay
DR* = selected dropout
D*  = selected d_model
H*  = selected num_heads
HD* = selected head_dim
```

Possible:

```text
D* ∈ {32,64}
H* ∈ {2,4}
```

Phase 35 không hard-code:

```text
D64
H4
```

---

# 5. Canonical layer options

Registry:

```text
N1 = 1
N2 = 2
```

Exact numeric values:

```text
1
2
```

No other layer count may enter S13 ranking.

---

# 6. Reference layer count

Current sequential reference từ Phase 34 still uses:

```text
N2 = 2
```

vì depth chưa được swept trước Phase 35.

If exact S13 contract match:

```text
N2 → REUSE S12 winner
N1 → NEW fresh run
```

Normal S13 execution:

```text
1 new N1 scientific run
+
1 reused N2 reference
```

---

# 7. Scientific meaning của layer sweep

Current Transformer path:

```text
X [B,L,F]
→ Input Projection
→ Positional Encoding
→ Encoder Layer 1
→ Encoder Layer 2 (N2 only)
→ Pooling
→ Regression Head
→ [B,1]
```

N1:

```text
Input Projection
→ PE
→ Encoder Layer 1
→ Pooling
→ Head
```

N2:

```text
Input Projection
→ PE
→ Encoder Layer 1
→ Encoder Layer 2
→ Pooling
→ Head
```

S13 tests:

> whether an additional full Transformer Encoder block improves forecasting under the already selected width/head/regularization/optimization configuration.

---

# 8. Extra layer means extra compositional depth, not longer temporal coverage

Each self-attention layer can already attend over the full input sequence:

```text
L*
```

because no causal mask is required and no padding mask is used.

Therefore:

```text
N2 does NOT increase the lookback horizon.
```

Both N1 and N2 receive exactly the same historical window.

The second layer increases:

```text
representation transformation depth
attention/FFN composition
nonlinearity depth
normalization/residual operations
```

not input temporal coverage.

---

# 9. Input/output contract remains unchanged

Both candidates:

```text
input:
[B,L*,F]

final prediction:
[B,1]
```

Same target:

```text
Appliances_{t+1}
```

Same horizon:

```text
H1
```

Same target population.

---

# 10. Selected d_model is frozen

Hard:

```text
d_model = D*
```

resolved from S11.

No width compensation when reducing depth.

---

# 11. Selected heads are frozen

Hard:

```text
num_heads = H*
head_dim = HD*
```

resolved from S12.

No head-count adjustment in S13.

---

# 12. Divisibility remains valid

Verify:

\[
D^* \bmod H^* = 0
\]

and:

\[
head\_dim = D^*/H^*
\]

for both N1/N2.

Layer count does not affect head geometry.

---

# 13. FFN width is frozen

Hard:

```text
ffn_dim = 128
```

for both.

Phase 36 is the FFN sweep.

---

# 14. Dropout remains frozen

Hard:

```text
dropout = DR*
```

at the same semantic sites inside every active Encoder layer.

Important:

```text
N2 contains twice as many Encoder blocks as N1
```

so there are more **instances/applications** of the same dropout-bearing submodules in the deeper architecture.

This is an inherent consequence of changing depth.

Do not compensate by lowering `p` for N2.

---

# 15. Dropout-depth exposure nuance

At fixed `DR*`:

```text
N1:
dropout modules execute inside 1 layer

N2:
dropout modules execute inside 2 layers
```

Therefore total network-level regularization exposure differs structurally with depth.

This is expected.

Do not attempt:

```text
DR_N2 < DR_N1
```

to equalize a hypothetical “total dropout”.

That would change:

```text
layers + dropout
```

simultaneously.

---

# 16. LayerNorm count changes inherently

Current custom Encoder layer includes its own normalization modules.

Therefore:

```text
N1 → one layer's LayerNorm set
N2 → two layers' LayerNorm sets
```

More LayerNorm parameters/operations are part of the depth effect.

No norm-count compensation.

---

# 17. Attention block count changes inherently

N1 contains:

```text
1 MHA block
```

N2 contains:

```text
2 MHA blocks
```

with identical per-layer architecture specification.

This is the intended capacity increase.

---

# 18. FFN block count changes inherently

N1:

```text
1 FFN block
```

N2:

```text
2 FFN blocks.
```

Again intended.

---

# 19. Residual block count changes inherently

N2 has another:

```text
attention residual path
FFN residual path
normalization sequence
```

because it has another encoder layer.

No compensation.

---

# 20. Parameter count is expected to change

Unlike S12 Head Sweep, depth changes trainable capacity.

Expected:

```text
Params_N1 < Params_N2.
```

Hard expected delta:

```text
Params_N2 - Params_N1
=
parameters of one additional Encoder layer
```

under the frozen custom Transformer implementation.

Runtime count is authoritative.

---

# 21. Canonical one-layer parameter formula

Under the established Transformer-v1 assumptions, one Encoder layer contributes approximately:

\[
P_{layer}(D,M)
=
4D^2 + 2DM + 9D + M
\]

where:

```text
D = selected d_model
M = FFN width = 128.
```

Thus:

\[
Params_{N2} - Params_{N1}
\approx
P_{layer}(D^*,128)
\]

only if runtime implementation matches the canonical assumptions.

Use as an audit reference only.

---

# 22. Reference delta for D32

If:

```text
D*=32
M=128
```

reference layer delta:

\[
4(32)^2+2(32)(128)+9(32)+128
\]

which is algebraically fixed by the formula, but the runtime model count remains the source of truth.

Do not pre-fill a runtime artifact with the formula result.

---

# 23. Reference delta for D64

If:

```text
D*=64
M=128
```

use the same formula with D=64.

Again:

```text
audit reference only.
```

---

# 24. No parameter matching

Do not change:

```text
d_model
heads
FFN
dropout
```

to make N1 and N2 have equal parameter counts.

The extra parameters are the intended depth factor.

---

# 25. Shared stem/head architecture remains unchanged

Both candidates must share identical:

```text
input projection
positional encoding policy
pooling
regression head
```

The only topology difference is the number of repeated Encoder layers.

---

# 26. Architecture role relation

N1 roles:

```text
input_projection
positional_encoding
encoder.layers.0
pooling
regression_head
```

N2 roles:

```text
input_projection
positional_encoding
encoder.layers.0
encoder.layers.1
pooling
regression_head
```

Expected:

```text
all shared roles identical
plus exactly one extra Encoder layer in N2.
```

No other module difference.

---

# 27. Architecture delta whitelist

Allowed N1/N2 differences:

```text
num_layers
presence of encoder layer index 1 in N2
parameters/buffers owned by layer 1
attention tensor list length
parameter count
checkpoint size
compute/runtime/memory
config fingerprint
```

Not allowed:

```text
different input projection
different PE
different D*
different H*
different FFN width
different activation
different pooling
different dropout probability
different regression head
```

---

# 28. State-dict key-set expectation

N2 should contain all N1 shared keys plus the extra second-layer keys, modulo any implementation-specific indexing convention.

Conceptually:

```text
keys_N1
⊂
keys_N2
```

where the set difference should correspond exactly to:

```text
encoder layer 2 parameters/buffers.
```

Create an explicit key-delta audit.

---

# 29. Shared parameter shapes should match

For every semantically shared module:

```text
input projection
PE buffer
layer 0
regression head
```

expected tensor shapes match across N1/N2.

Layer 1 exists only in N2.

---

# 30. Initialization fairness cannot use whole-model equality

Because N2 has extra parameters:

```text
whole_initial_state_fingerprint_N1
!=
whole_initial_state_fingerprint_N2
```

and direct equality is not applicable.

Instead audit:

```text
same initialization policy
same seed
same model builder code
same shared-module construction order
```

---

# 31. Shared-prefix initialization matching is highly desirable

If model builder constructs shared modules in the same deterministic order before creating the second layer, then under the same seed it may be possible to verify:

```text
N1 input projection initial tensors
==
N2 input projection initial tensors

N1 layer0 initial tensors
==
N2 layer0 initial tensors

N1 regression head initial tensors
```

may or may not match depending on whether head construction occurs before/after extra-layer initialization.

Therefore do not assume full shared-module equality blindly.

Audit actual construction order.

---

# 32. Initialization-policy fingerprint is required

Create a fingerprint over:

```text
seed policy
PyTorch version
model builder code fingerprint
custom initialization policy
layer-construction policy
bias initialization policy
LayerNorm initialization policy
```

Expected:

```text
same policy fingerprint
```

for N1/N2.

---

# 33. Optional shared-prefix init audit

If practical:

```text
compare only modules whose RNG construction sequence is guaranteed identical.
```

Allowed statuses:

```text
MATCH
NOT_MATCH_DUE_CONSTRUCTION_ORDER
NOT_VERIFIABLE
NOT_APPLICABLE
```

Do not fail S13 solely because later modules get different initial tensors due extra-layer RNG consumption.

---

# 34. No N2→N1 layer truncation

Do not create N1 by:

```text
loading N2
removing layer2
continuing training.
```

Official N1 is a fresh run.

No architecture surgery.

---

# 35. No N2→N1 warm-start

Forbidden:

```text
copy input projection
copy layer0
copy head
from N2 BEST into N1.
```

Even if shapes match.

N1 must start under the frozen initialization policy.

---

# 36. No N1→N2 layer cloning inside S13

N2 is a historical reference.

Do not create another N2 by cloning the N1 layer to form layer2.

Use exact S12 winner.

---

# 37. Layer independence contract

Custom attention-aware encoder uses independent Encoder layer instances.

For N2:

```text
layer0 parameters
and
layer1 parameters
```

must not share the same tensor objects unless weight sharing is explicitly part of Transformer-v1, which it is not.

Hard audit:

```text
no accidental parameter sharing between layers.
```

---

# 38. Why accidental layer sharing is critical

If N2 reuses the exact same layer module twice:

```text
parameter count may fail to increase
```

and S13 would no longer compare true depth capacity.

Therefore N2 reference provenance must confirm:

```text
independent layers
```

under `TRANSFORMER_IMPL-v1`.

---

# 39. Layer independence audit

For N2, verify representative parameter object identities/storage pointers differ between:

```text
layer0
layer1.
```

Also verify:

```text
parameter names distinct
both trainable
both present in optimizer groups.
```

---

# 40. Optimizer coverage

All N1 parameters must appear exactly once in optimizer groups.

For N2 reference, both layers must likewise have full optimizer coverage.

No layer may be accidentally omitted from optimization.

---

# 41. Weight-decay group policy remains frozen

Preserve S9 optimizer parameter-group semantics.

The new N1 model has fewer parameters, but the grouping rule remains the same.

Do not create layer-specific WD.

---

# 42. No layer-wise learning-rate decay

Do not introduce:

```text
lower LR for early layer
higher LR for later layer
layer-wise LR decay
```

inside S13.

All trainable parameters follow the frozen optimizer policy.

---

# 43. No layer-specific dropout

No:

```text
layer0 p=.1
layer1 p=.2
```

unless that was already part of Transformer-v1, which it is not.

Use selected global `DR*`.

---

# 44. No stochastic layer dropping

Do not use:

```text
LayerDrop
Stochastic Depth
DropPath
```

inside S13.

The layer-count factor must be structural:

```text
1 actual layer
vs
2 actual layers.
```

---

# 45. Attention API output length changes

N1:

```text
attention list length = 1
```

N2:

```text
attention list length = 2
```

Each tensor:

```text
[B,H*,L*,L*].
```

This is expected.

---

# 46. No padding/causal mask change

Both:

```text
no causal mask
no padding mask
```

under the established one-step-ahead sequence-to-one setup.

Depth does not change masking requirements.

---

# 47. POST_NORM remains frozen

Current Transformer-v1 uses:

```text
POST_NORM
```

for every active layer.

N1/N2 both use identical norm ordering per layer.

Do not switch N1 to PRE_NORM for stability.

---

# 48. Activation remains selected A*

Every active FFN block uses the same:

```text
A*
```

from S6.

No depth-specific activation.

---

# 49. Positional encoding is applied once

Current architecture:

```text
input projection
→ positional encoding
→ encoder stack.
```

Depth does not imply re-adding PE before every layer.

Hard expected:

```text
PE application count remains one.
```

Do not add PE again at layer2.

---

# 50. Pooling occurs after the final active layer

N1:

```text
pool layer0 output.
```

N2:

```text
pool layer1 output.
```

Do not pool each layer and average predictions.

That would change architecture beyond depth.

---

# 51. No deep supervision

Do not add auxiliary regression heads/losses at intermediate layers.

N2 uses one final regression head after the final encoder layer.

---

# 52. No residual connection across whole stack beyond Transformer-v1

Do not add:

```text
input-to-final skip
layer averaging
weighted layer sum
```

inside S13.

---

# 53. Working hypotheses

## H-S13-01 — N2 may improve representation composition

A second Encoder layer may refine/combine features from layer0 and reduce Validation error.

Status:

```text
UNTESTED.
```

---

# 54. H-S13-02 — N1 may be sufficient

For this dataset/task, one self-attention+FFN block may already capture useful temporal dependencies.

N1 may match or beat N2 with fewer parameters.

Status:

```text
UNTESTED.
```

---

# 55. H-S13-03 — N2 may overfit more easily

The deeper model has more trainable parameters and repeated nonlinear transformations.

It may show:

```text
lower Train error
but worse Validation RMSE
```

under the same regularization.

This is a hypothesis, not a presumption.

---

# 56. H-S13-04 — N1 may underfit

If N1 shows:

```text
high Train error
high Validation error
early plateau
```

relative to N2, that may be an underfitting-like signal.

No universal threshold.

---

# 57. Preconditions

Required:

```text
Phase 34 = PASS
```

or:

```text
PASS_WITH_WARNING
```

with no unresolved critical issue.

Also:

```text
approved_for_phase35 = true.
```

---

# 58. Required upstream artifacts

```text
s12_head_winner.json
s12_reference_update.json
phase_34_signoff.json
```

---

# 59. Required upstream contracts

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
```

---

# 60. Carry-forward warnings

Propagate all unresolved non-critical warnings.

Examples:

```text
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
BOUNDARY_WINNER
SAMPLE_ORDER_NOT_VERIFIABLE
INITIALIZATION_NOT_VERIFIABLE
RANDOM_CONTROL_GAIN
```

Include in:

```text
S13 manifest
S13 summary
S13 report
S13 winner
Phase 36 handoff.
```

---

# 61. Swept factor only

Canonical:

```text
num_layers
```

Allowed:

```text
1
2
```

Aliases:

```text
N1
N2
```

No `N3`, `N4`, `N6`.

---

# 62. Frozen data contract

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
same feature order/fingerprint
same X scaler
same target transform
same target IDs
```

---

# 63. Frozen architecture settings

Hard:

```text
D*
H*
HD*
F128
P*
A*
DR*
sinusoidal PE
POST_NORM
no masks
Linear(D*,1) final head
```

Only:

```text
num_layers
```

changes.

---

# 64. Frozen optimizer/training settings

Hard:

```text
B*
AdamW
LR*
WD*
same optimizer-group policy
MSE
E50
patience10
min_delta0
clip1
scheduler=None
warmup=None
accumulation1
mixed precision policy fixed
seed42
TRAINING_ENGINE-v1
METRICS-v1
```

---

# 65. Same sample-order policy

N1 should use:

```text
same Train population
same Train shuffle
same batch grouping
same DataLoader generator policy
same worker policy
```

as the reference training contract.

If N2 historical order fingerprint exists, compare.

Else:

```text
N2_ORDER_MATCH = NOT_VERIFIABLE.
```

Do not retrain N2.

---

# 66. N2 reference reuse gate

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
H1
WB0
WINDOWPOP-v1

N2
F128
same PE
same POST_NORM
same masks
same dropout scope

AdamW
MSE
E50
patience10
clip1
scheduler=None
accumulation1
seed42
TRAINING_ENGINE-v1
METRICS-v1
```

Mismatch:

```text
STOP.
```

---

# 67. Fresh N1 run

Execution:

```text
register N1 run
↓
reseed42
↓
fresh Train DataLoader
↓
fresh Validation DataLoader
↓
fresh Transformer(
    d_model=D*,
    num_heads=H*,
    num_layers=1,
    ffn_dim=128,
    dropout=DR*,
    activation=A*,
    pooling=P*
)
↓
fresh AdamW(LR*,WD*)
↓
MSE
↓
TRAINING_ENGINE-v1
```

No warm-start.

---

# 68. No N2 reference retraining

Do not retrain N2 to:

```text
get a better stochastic draw
match initialization metadata
match sample order metadata
```

Reuse exact S12 winner.

---

# 69. N1 geometry assertions

Runtime:

```text
d_model = D*
num_heads = H*
head_dim = HD*
num_layers = 1
ffn_dim = 128
dropout = DR*
activation = A*
pooling = P*
```

---

# 70. N2 reference geometry assertions

Runtime/reference metadata:

```text
d_model = D*
num_heads = H*
head_dim = HD*
num_layers = 2
ffn_dim = 128
dropout = DR*
activation = A*
pooling = P*
```

---

# 71. Shared input projection audit

Expected identical shape:

```text
Linear(F,D*)
```

for N1/N2.

---

# 72. Positional encoding audit

Expected identical:

```text
policy
width D*
max_seq_len
buffer shape
application location.
```

---

# 73. Layer0 geometry audit

N1 layer0 and N2 layer0 must have identical architecture geometry:

```text
MHA embed_dim D*
heads H*
head_dim HD*
FFN D*→128→D*
dropout DR*
activation A*
POST_NORM.
```

---

# 74. N2 layer1 geometry audit

N2 second layer must be architecturally identical to layer0 under Transformer-v1.

No special second-layer dimensions/activation/dropout.

---

# 75. Regression-head audit

Both:

```text
Linear(D*,1)
```

same shape.

No extra head for N2.

---

# 76. Pooling audit

Both use same `P*`, applied only after final active layer.

---

# 77. Parameter-count audit

Compute runtime:

```text
Params_N1
Params_N2
Delta_Params
```

Expected:

```text
Params_N1 < Params_N2
```

and:

```text
Delta_Params
≈ one Encoder-layer parameter count.
```

---

# 78. Layer-delta audit

Create a direct runtime comparison:

```text
N2-only parameter names
```

Expected to correspond to:

```text
encoder layer 1
```

under zero-based indexing.

Sum their `numel()`.

Expected:

```text
sum(N2-only params)
=
Params_N2 - Params_N1
```

if all shared parameter schemas otherwise align.

This is a strong S13 correctness check.

---

# 79. Shared parameter schema audit

For keys semantically shared between N1/N2:

```text
shape equality
dtype equality
trainable flag equality
```

must hold.

---

# 80. N2 layer independence audit

Verify:

```text
layer0 and layer1 parameter objects are independent
```

and not shared.

Expected:

```text
different object IDs/storage
different parameter names
both in state_dict
both optimized.
```

---

# 81. Optimizer coverage audit

For N1:

```text
every trainable parameter appears exactly once in optimizer groups.
```

For N2 reference provenance:

```text
both layers were covered.
```

No duplicate optimizer parameter references.

---

# 82. N1 forward sanity

Before official training on disposable objects:

```text
B1
small batch
selected B*
final partial batch
```

Expected prediction:

```text
[B,1].
```

Finite outputs.

---

# 83. N1 backward sanity

Synthetic/actual small batch:

```text
MSE
backward
```

must produce finite gradients.

No optimizer step in pure architecture sanity.

Official run starts from fresh objects after sanity.

---

# 84. Attention API sanity

N1:

```text
attention list length = 1
per-layer attention [B,H*,L*,L*]
prediction [B,1].
```

N2 reference contract:

```text
attention list length = 2.
```

---

# 85. Standard/inspection prediction equivalence

For N1 eval mode:

```text
model(x)
```

and:

```text
forward_with_attention(x).prediction
```

must be allclose under frozen tolerance.

---

# 86. Attention probability sanity

In eval mode:

```text
finite
nonnegative
source-axis rows ≈1.
```

No attention-based selection.

---

# 87. No attention-layer averaging in S13

Do not compare N1/N2 by averaging N2 attention layers into one and judging visualization quality.

Interpretability belongs later.

---

# 88. Primary metric

Hard:

```text
best_validation_rmse_wh
```

from verified BEST checkpoint.

---

# 89. Secondary scientific metrics

Record:

```text
Validation MAE Wh
Validation R²
best epoch
epochs completed
stop reason
train loss
gradient diagnostics
clipping fraction
convergence diagnostics.
```

---

# 90. Capacity/engineering metrics

Record:

```text
trainable parameter count
parameter increase
checkpoint size
mean epoch time
samples/sec
optional peak memory
steps to best
time to best.
```

Secondary only.

---

# 91. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{N1},
RMSE_{N2}
\right)
\]

Use full-precision Validation RMSE Wh.

---

# 92. Exact RMSE tie rule

If exact full-precision equality:

```text
prefer N1.
```

Rationale:

```text
fewer trainable parameters
shallower architecture
lower expected compute/memory burden
simpler model
predeclared parsimony.
```

Only exact tie invokes this rule.

---

# 93. No capacity override for non-tie

If N2 RMSE is lower by any non-zero full-precision amount:

```text
N2 wins.
```

Do not choose N1 merely because it is smaller/faster.

---

# 94. Effect formula

Define N1→N2:

\[
\Delta RMSE
=
RMSE_{N1}-RMSE_{N2}
\]

Positive:

```text
N2 improves.
```

Relative:

\[
Improvement\%
=
100\times
\frac{RMSE_{N1}-RMSE_{N2}}
{RMSE_{N1}}
\]

Also:

```text
MAE delta = MAE_N1 - MAE_N2
R² delta = R²_N2 - R²_N1.
```

---

# 95. Parameter increase

Runtime:

\[
\Delta Params
=
Params_{N2}-Params_{N1}
\]

\[
Increase\%
=
100\times
\frac{\Delta Params}{Params_{N1}}
\]

---

# 96. RMSE gain per extra layer

The most interpretable architecture effect is simply:

```text
one extra Encoder layer
→ ΔRMSE
→ ΔParams
→ Δruntime
```

Optional:

\[
RMSEGainPer10kExtraParams
=
\frac{RMSE_{N1}-RMSE_{N2}}
{(Params_{N2}-Params_{N1})/10000}
\]

Engineering context only.

---

# 97. Pareto context

Classify:

```text
N1_DOMINATES
N2_ACCURACY_GAIN_WITH_COST
ACCURACY_EFFICIENCY_TRADEOFF
EXACT_TIE_N1_PARSIMONY
```

Examples:

```text
N1 lower RMSE + fewer params → N1 dominates
N2 lower RMSE + more params → accuracy/cost tradeoff
```

Winner remains RMSE/tie-rule based.

---

# 98. Metric ranking divergence

If RMSE and MAE disagree:

```text
METRIC_RANKING_DIVERGENCE
```

Record it.

Winner remains RMSE-based.

---

# 99. BEST verification for N1

After official run:

```text
fresh Transformer(num_layers=1)
↓
verify full config
↓
strict-load N1 BEST
↓
model.eval()
↓
full ordered Validation
↓
inverse target transform if needed
↓
METRICS-v1
↓
verify stored BEST metrics.
```

---

# 100. N2 reference verification

No retrain.

Verify:

```text
selected D*
selected H*
N2
F128
all prior selected factors
same population
same metric version
same Training Engine provenance
BEST already verified.
```

---

# 101. Checkpoint metadata

Must include:

```text
num_layers
d_model
num_heads
head_dim
ffn_dim
dropout
activation
pooling
parameter count
model config fingerprint
architecture-role fingerprint.
```

---

# 102. State-dict config provenance

Unlike S12 where H2/H4 may share parameter shapes, N1/N2 state dict key sets differ.

Still:

```text
state_dict alone is not sufficient experiment provenance.
```

Always pair with model config.

---

# 103. Same sample population hard gate

Required:

```text
Train IDs_N1 == Train IDs_N2
Validation IDs_N1 == Validation IDs_N2
WINDOWPOP-v1 same.
```

Depth must not affect sample eligibility.

---

# 104. Same DataLoader settings

```text
B*
shuffle Train=true
shuffle Validation=false
drop_last=false
same worker policy
same generator-seed policy.
```

---

# 105. Optimizer steps per complete epoch

Because batch/population fixed:

```text
steps_per_epoch_N1
=
steps_per_epoch_N2.
```

Total steps may differ due early stopping.

---

# 106. Gradient diagnostics

Record:

```text
mean preclip grad norm
max preclip grad norm
fraction batches clipped
epochs with clipping
nonfinite events.
```

Caveat:

```text
N2 has more parameters
```

so raw global L2 gradient norm is partly dimension-sensitive.

Do not interpret larger norm as automatically less stable.

---

# 107. Optional normalized gradient context

If useful:

```text
grad_norm / sqrt(trainable_parameter_count)
```

as diagnostic only.

No selection use.

---

# 108. Convergence diagnostics

Record:

```text
first epoch RMSE
best epoch
best RMSE
last RMSE
best-to-last gap
early-stopped?
epoch-cap reached?
steps to best
post-best worsening epochs.
```

---

# 109. Underfitting-like diagnostic

Possible N1 evidence:

```text
higher Train RMSE
higher Validation RMSE
plateau earlier.
```

Do not infer underfitting from one layer alone.

---

# 110. Overfitting-like diagnostic

Possible N2 evidence:

```text
lower Train RMSE
higher Validation RMSE
larger post-best degradation.
```

Descriptive only.

---

# 111. Runtime diagnostics

Record:

```text
mean epoch seconds
median epoch seconds
total runtime
samples/sec
time to best optional.
```

N2 is expected to require more computation, but observed runtime is authoritative.

---

# 112. Checkpoint-size diagnostics

Record:

```text
BEST full checkpoint bytes
LAST full checkpoint bytes
optional model state_dict bytes.
```

N2 should generally be larger due extra layer parameters and optimizer state.

Do not mix pure model size and full training-checkpoint size.

---

# 113. Memory diagnostics

Optional:

```text
peak allocated
peak reserved
```

if backend supports it.

If unavailable:

```text
NOT_AVAILABLE.
```

No fabricated estimates.

---

# 114. Attention memory context

If attention is extracted:

```text
N2 produces two layer maps
N1 produces one.
```

Thus interpretability artifact memory roughly scales with layer count at fixed B/H/L.

But official training keeps:

```text
need_weights=False.
```

Do not use attention tensor storage as standard training-memory estimate.

---

# 115. Final attention-analysis implication

Selected layer count affects later Phases 52–57:

```text
N1 → one layer to analyze
N2 → two layers to analyze.
```

Final attention-analysis code must not hard-code 2 layers after S13.

Carry:

```text
selected_num_layers
```

into final provenance.

---

# 116. Single-seed limitation

Mandatory:

```text
S13 uses seed42 only.
```

No mean±std.

---

# 117. Validation-only limitation

Mandatory:

```text
S13 winner is selected using Validation only.
```

No Test evidence.

---

# 118. Depth × d_model interaction limitation

S13 is conditional on selected `D*`.

A shallow wider model may compare differently to a deeper narrower model, but S13 does not test such a factorial design.

---

# 119. Depth × heads interaction limitation

Selected `H*` is fixed.

No N×H grid.

---

# 120. Depth × FFN interaction limitation

F128 fixed.

Phase 36 sweeps FFN after depth selection.

No N×FFN grid.

---

# 121. Depth × dropout interaction limitation

DR* fixed.

N2 naturally experiences dropout in more layer instances.

No regularization retuning.

---

# 122. Depth × LR/WD interaction limitation

LR*/WD* fixed.

Deeper models can prefer different optimizer settings, but S13 does not retune them.

---

# 123. Depth × batch interaction limitation

B* fixed.

---

# 124. Depth × epoch-budget interaction limitation

Both get:

```text
E50/patience10.
```

A deeper model may require different training duration, but that is not changed here.

Phase 38 later tests epoch cap on the then-current reference.

---

# 125. Sequential-selection limitation

By Phase 35, Validation has influenced:

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
S10 Dropout
S11 d_model
S12 Heads
S13 Layers
```

Complete experiment lineage and later robustness stages are mandatory.

---

# 126. No Test access

Hard:

```text
Test loader not iterated
Test metrics absent
Test predictions absent.
```

---

# 127. No N0

Not registered.

---

# 128. No N3/N4

Not registered.

---

# 129. No layer sharing

No.

---

# 130. No layer-wise LR

No.

---

# 131. No layer-wise WD

No.

---

# 132. No layer-specific dropout

No.

---

# 133. No LayerDrop/DropPath

No.

---

# 134. No intermediate heads/deep supervision

No.

---

# 135. No layer averaging

No.

---

# 136. No layer-wise ensemble

No.

---

# 137. No additional PE per layer

No.

---

# 138. No optimizer-state reuse

No.

---

# 139. No warm-start

No.

---

# 140. No attention-based winner selection

No.

---

# 141. No extra epochs for N2

No.

---

# 142. Run failure policy

If N1 has a technical failure:

```text
S13 incomplete
```

until a documented technical rerun is resolved.

Do not automatically declare N2 winner.

---

# 143. Technical rerun allowed

Only for documented:

```text
process interruption
hardware failure
software failure
corrupt checkpoint
artifact-write failure.
```

Use Experiment Registry rerun policy.

---

# 144. Score-based rerun forbidden

Do not rerun N1 because:

```text
RMSE looks poor
curve looks unusual.
```

Do not retrain N2 to seek a better stochastic draw.

---

# 145. Numerical instability policy

If N1 yields NaN/Inf:

```text
audit data/config/model/device
```

first.

If genuine:

```text
record numerical failure
S13 incomplete under strict core protocol
```

until formally resolved.

---

# 146. Discrepancy taxonomy

```text
S12_REFERENCE_MISSING
S12_WINNER_MISMATCH
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
LAYER_DEFINITION_MISMATCH
FFN_DIM_DRIFT
PE_DRIFT
PE_REAPPLIED_PER_LAYER
POST_NORM_DRIFT
MASK_POLICY_DRIFT
INPUT_PROJECTION_DRIFT
REGRESSION_HEAD_DRIFT
POOLING_DRIFT_INTERNAL
UNEXPECTED_ARCHITECTURE_DELTA
PARAMETER_COUNT_UNEXPECTED_DIRECTION
LAYER_DELTA_PARAMETER_MISMATCH
SHARED_PARAMETER_SCHEMA_MISMATCH
ACCIDENTAL_LAYER_WEIGHT_SHARING
OPTIMIZER_COVERAGE_MISSING_LAYER
OPTIMIZER_DUPLICATE_PARAMETER
LAYERWISE_LR_ADDED
LAYERWISE_WD_ADDED
LAYER_SPECIFIC_DROPOUT
LAYERDROP_ADDED
DEEP_SUPERVISION_ADDED
LAYER_AVERAGING_ADDED
WARM_START_USED
LAYER_TRUNCATION_USED
OPTIMIZER_STATE_REUSE
POPULATION_MISMATCH
TARGET_ID_MISMATCH
X_SCALER_MISMATCH
TARGET_SCALER_MISMATCH
SAMPLE_ORDER_POLICY_DRIFT
INITIALIZATION_POLICY_DRIFT
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
REFERENCE_RUN_MISMATCH
RUN_FAILURE
NUMERICAL_INSTABILITY
CHECKPOINT_CONFIG_MISMATCH
CHECKPOINT_VERIFICATION_FAILURE
INCOMPLETE_SWEEP
RANKING_ERROR
EFFECT_CALCULATION_ERROR
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 147. Status model

## PASS

```text
N2 reference valid
N1 architecture valid
same data/optimization
depth-only topology delta valid
parameter increase valid
N1 BEST verified
winner selected
Phase36 reference generated
Test untouched.
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
metric ranking divergence
accuracy-efficiency tradeoff
reference init/order metadata unavailable
optional memory metric unavailable
inherited warning.
```

## FAIL

Examples:

```text
D/H/FFN changes
layer weight sharing
N2 extra layer not independently optimized
population mismatch
warm-start
deep supervision
wrong layer count
unresolved N1 failure
Test access.
```

---

# 148. Output directory

```text
artifacts/
└── sweeps/
    └── S13_layers/
        ├── s13_layer_sweep_manifest.json
        ├── s13_layer_sweep_contract.json
        ├── s13_layer_preflight_audit.csv
        ├── s13_run_matrix.csv
        ├── s13_layer_definition_audit.csv
        ├── s13_encoder_stack_geometry_audit.csv
        ├── s13_architecture_role_audit.csv
        ├── s13_state_dict_key_delta_audit.csv
        ├── s13_shared_parameter_schema_audit.csv
        ├── s13_parameter_count_audit.csv
        ├── s13_layer_delta_parameter_audit.csv
        ├── s13_layer_independence_audit.csv
        ├── s13_optimizer_coverage_audit.csv
        ├── s13_config_delta_audit.csv
        ├── s13_layer_unit_tests.csv
        ├── s13_common_data_audit.csv
        ├── s13_layer_training_audit.csv
        ├── s13_initialization_policy_audit.csv
        ├── s13_shared_prefix_initialization_audit.csv
        ├── s13_sample_order_audit.csv
        ├── s13_dropout_depth_audit.csv
        ├── s13_attention_api_audit.csv
        ├── s13_optimizer_budget_audit.csv
        ├── s13_layer_run_provenance.csv
        ├── s13_layer_metrics.csv
        ├── s13_layer_effect.csv
        ├── s13_depth_efficiency_context.csv
        ├── s13_depth_pareto_context.json
        ├── s13_optimization_diagnostics.csv
        ├── s13_convergence_diagnostics.csv
        ├── s13_runtime_capacity_diagnostics.csv
        ├── s13_generalization_diagnostics.csv
        ├── s13_hypothesis_outcomes.csv
        ├── s13_layer_findings.csv
        ├── s13_layer_winner.json
        ├── s13_reference_update.json
        ├── s13_layer_sweep_tests.csv
        ├── s13_layer_discrepancies.json
        ├── s13_layer_sweep_summary.json
        ├── s13_layer_sweep_report.md
        ├── figures/
        │   ├── S13_01_validation_rmse_by_epoch.png
        │   ├── S13_02_validation_mae_by_epoch.png
        │   ├── S13_03_train_loss_by_epoch.png
        │   ├── S13_04_gradient_clipping_fraction.png
        │   ├── S13_05_best_validation_metrics.png
        │   ├── S13_06_parameter_count_vs_rmse.png
        │   ├── S13_07_runtime_vs_rmse.png
        │   ├── S13_08_checkpoint_size_comparison.png
        │   ├── S13_09_convergence_summary.png
        │   └── S13_10_generalization_gap_optional.png
        ├── README_S13_LAYER_SWEEP.md
        └── phase_35_signoff.json
```

Optional:

```text
s13_shared_prefix_initialization_audit.csv
s13_generalization_diagnostics.csv
S13_10_generalization_gap_optional.png
peak-memory fields
```

may be omitted when not supported, but omission must be documented.

New N1 scientific run stays in:

```text
artifacts/runs/<run_id>/
```

No checkpoint duplication.

---

# 149. Required outputs

```text
O35.1  Sweep manifest
O35.2  Sweep contract
O35.3  Preflight audit
O35.4  Run matrix
O35.5  Layer definition audit
O35.6  Encoder-stack geometry audit
O35.7  Architecture-role audit
O35.8  State-dict key-delta audit
O35.9  Shared parameter-schema audit
O35.10 Runtime parameter-count audit
O35.11 Layer-delta parameter audit
O35.12 Layer-independence audit
O35.13 Optimizer-coverage audit
O35.14 Config-delta audit
O35.15 Layer unit tests
O35.16 Common-data audit
O35.17 Training-config audit
O35.18 Initialization-policy audit
O35.19 Optional shared-prefix init audit
O35.20 Sample-order audit
O35.21 Dropout-depth audit
O35.22 Attention-API audit
O35.23 Optimizer-budget audit
O35.24 Run provenance
O35.25 Reused N2 reference
O35.26 Verified new N1 run
O35.27 Primary metrics table
O35.28 Layer effect table
O35.29 Depth-efficiency context
O35.30 Pareto context
O35.31 Optimization diagnostics
O35.32 Convergence diagnostics
O35.33 Runtime/capacity diagnostics
O35.34 Optional generalization diagnostics
O35.35 Hypothesis outcomes
O35.36 Findings
O35.37 Winner artifact
O35.38 Phase36 reference update
O35.39 Figures
O35.40 Sweep tests
O35.41 Discrepancy log
O35.42 Sweep summary
O35.43 Human-readable report
O35.44 README
O35.45 Phase sign-off
```

---

# 150. Run matrix

Create:

```text
s13_run_matrix.csv
```

Fields:

```text
sweep_id
layer_id
num_layers
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
ffn_dim
population_fingerprint
feature_fingerprint
seed
model_config_id
training_config_id
status
```

---

# 151. Sweep manifest

Create:

```text
s13_layer_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S13_LAYERS-v1
sweep_id = S13_LAYERS
source_s12_winner_run_id
selected_d_model
selected_num_heads
selected_head_dim
candidate_layers = [1,2]
fixed_ffn_dim = 128
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
dropout_probability
new_runs_required
reused_runs
swept_field = num_layers
capacity_change_expected = true
parameter_count_equality_expected = false
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = N1_ON_EXACT_RMSE_TIE
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

# 152. Sweep contract

Create:

```text
s13_layer_sweep_contract.json
```

Must state:

```text
Only num_layers changes.

N1=1.
N2=2.

D*, H*, head_dim, F128 fixed.

Same PE/pooling/activation/dropout probability.
Same data/population.
Same optimizer/training budget.
Same dropout-site semantics per active layer.

N2 naturally has an additional complete Encoder block.
Parameter count is expected to increase.

No parameter matching.
No layer sharing.
No layer-wise LR/WD/dropout.
No LayerDrop.
No deep supervision.
No warm-start/truncation.

N2 reference reused.
N1 fresh seed42 run.

Validation RMSE Wh selects winner.
Exact tie → N1.
Test forbidden.
```

---

# 153. Preflight audit

`s13_layer_preflight_audit.csv` checks:

```text
phase34_pass
approved_for_phase35
s12_winner_valid
all prior selected fields locked
N1 registered
N2 registered
D* fixed
H* fixed
head_dim fixed
F128 fixed
PE fixed
dropout fixed
dropout scope fixed
POST_NORM fixed
mask policy fixed
population fixed
Training Engine fixed
Metric fixed
seed fixed
N2 layer independence provenance valid
Test lock
status
```

---

# 154. Layer definition audit

`s13_layer_definition_audit.csv`:

```text
layer_id
num_layers
d_model
num_heads
head_dim
ffn_dim
dropout
activation
pooling
pe_policy
norm_policy
expected_attention_tensor_count
status
```

---

# 155. Encoder stack geometry audit

`s13_encoder_stack_geometry_audit.csv`:

```text
layer_id
stack_index
present
embed_dim
num_heads
head_dim
ffn_dim
dropout
activation
norm_policy
independent_parameter_instance
status
```

N1 rows:

```text
layer0 present
layer1 absent
```

N2 rows:

```text
layer0 present
layer1 present.
```

---

# 156. Architecture role audit

`s13_architecture_role_audit.csv`:

```text
module_path_or_role
semantic_role
present_n1
present_n2
type_n1
type_n2
shape_n1
shape_n2
shared_role
expected_difference
status
```

---

# 157. State-dict key-delta audit

`s13_state_dict_key_delta_audit.csv`:

```text
key
present_n1
present_n2
shape_n1
shape_n2
shared_key
n2_only_expected
unexpected_difference
status
```

Expected:

```text
N2-only keys correspond only to the second encoder layer.
```

---

# 158. Shared parameter-schema audit

`s13_shared_parameter_schema_audit.csv`:

```text
semantic_parameter
n1_key
n2_key
shape_n1
shape_n2
dtype_equal
trainable_equal
shape_equal
status
```

---

# 159. Parameter-count audit

`s13_parameter_count_audit.csv`:

```text
layer_id
runtime_trainable_parameters
runtime_total_parameters
reference_formula_parameters
formula_applicable
expected_direction
direction_valid
status
```

Expected:

```text
Params_N1 < Params_N2.
```

---

# 160. Layer-delta parameter audit

`s13_layer_delta_parameter_audit.csv`:

```text
d_model
ffn_dim
runtime_n1_params
runtime_n2_params
runtime_delta
reference_one_layer_params
formula_applicable
n2_only_key_numel
delta_equals_n2_only_numel
delta_matches_reference_if_applicable
status
```

---

# 161. Layer independence audit

`s13_layer_independence_audit.csv`:

```text
n2_layer0_parameter
n2_layer1_parameter
same_shape
same_object
same_storage
shared_parameter_detected
status
```

Expected:

```text
same_object = false
same_storage = false
shared_parameter_detected = false.
```

Use representative or complete automated checks.

---

# 162. Optimizer coverage audit

`s13_optimizer_coverage_audit.csv`:

```text
layer_id
trainable_parameter_count
optimizer_parameter_reference_count
unique_optimizer_parameter_count
missing_parameters
duplicate_parameters
layer0_covered
layer1_covered_if_applicable
status
```

---

# 163. Config-delta audit

`s13_config_delta_audit.csv` allowed differences:

```text
num_layers
second-layer module presence
derived parameter count
state-dict key count
attention tensor list length
capacity/runtime/checkpoint metadata
config fingerprint
```

Everything else must match.

---

# 164. Layer unit tests

`s13_layer_unit_tests.csv` should cover at least:

```text
N1 accepted
N2 accepted
invalid num_layers rejected
N1 has exactly 1 encoder layer
N2 has exactly 2 encoder layers
D* same
H* same
head_dim same
F128 same
PE same
input projection same shape
regression head same shape
pooling same
activation same
dropout p same
POST_NORM same
mask policy same
N2 layer0/layer1 independent
N1 parameter count < N2
N2-N1 param delta valid
N2-only keys map to layer1
shared parameter shapes match
optimizer covers all N1 params
B1 output [1,1]
selected-B output [B,1]
partial batch valid
N1 forward finite
N1 backward finite
N1 attention list length 1
N2 attention contract list length 2
N1 attention [B,H*,L,L]
standard/inspection prediction allclose
no scientific attention extraction in official training
```

---

# 165. Common data audit

`s13_common_data_audit.csv`:

```text
split
sample_count_n1
sample_count_n2
sample_ids_equal
ordered_ids_equal
feature_fingerprint_equal
x_scaler_equal
target_transform_equal
lookback_equal
pooling_equal
activation_equal
batch_equal
lr_equal
wd_equal
dropout_equal
d_model_equal
num_heads_equal
ffn_dim_equal
population_equal
status
```

---

# 166. Training audit

`s13_layer_training_audit.csv`:

```text
layer_id
num_layers
batch
optimizer
learning_rate
weight_decay
dropout
criterion
max_epochs
patience
min_delta
clip
scheduler
warmup
accumulation
mixed_precision
seed
training_engine_version
only_num_layers_differs
status
```

---

# 167. Initialization-policy audit

`s13_initialization_policy_audit.csv`:

```text
layer_id
seed
initialization_policy_id
model_builder_code_fingerprint
pytorch_version
custom_initialization
whole_state_fingerprint
direct_whole_state_equality_applicable=false
policy_match
status
```

---

# 168. Shared-prefix initialization audit

Optional:

`s13_shared_prefix_initialization_audit.csv`:

```text
semantic_module
construction_order_match
shape_match
n1_initial_fingerprint
n2_initial_fingerprint
direct_match_expected
match
reason_if_not_expected
status
```

Do not use this as a winner criterion.

---

# 169. Sample-order audit

`s13_sample_order_audit.csv`:

```text
epoch_or_probe
n1_order_fingerprint
n2_order_fingerprint
n2_reference_available
same_order
status
```

---

# 170. Dropout-depth audit

`s13_dropout_depth_audit.csv`:

```text
layer_id
num_layers
dropout_probability
dropout_scope_per_layer
dropout_site_count_per_layer
active_layer_count
total_dropout_module_instances
same_p_per_layer
dropout_compensation_used=false
status
```

This artifact makes the structural regularization consequence explicit.

---

# 171. Attention API audit

`s13_attention_api_audit.csv`:

```text
layer_id
num_layers
expected_attention_tensor_count
observed_attention_tensor_count
num_heads
observed_shape_per_layer
finite
nonnegative
eval_row_sum_error_max
standard_inspection_prediction_allclose
status
```

Diagnostic only.

---

# 172. Optimizer budget audit

`s13_optimizer_budget_audit.csv`:

```text
layer_id
train_samples_per_epoch
train_batch_size
steps_per_epoch
epochs_completed
total_optimizer_steps
best_epoch
steps_to_best
same_steps_per_completed_epoch
status
```

---

# 173. Run provenance

`s13_layer_run_provenance.csv`:

```text
layer_id
num_layers
run_id
source_type
source_phase
config_fingerprint
architecture_role_fingerprint
parameter_schema_fingerprint
parameter_count
feature_fingerprint
population_fingerprint
initialization_policy_fingerprint
sample_order_provenance
dropout_scope_fingerprint
best_checkpoint_sha256
history_sha256
metric_artifact
prediction_artifact
status
```

---

# 174. Primary metrics table

`s13_layer_metrics.csv`:

```text
layer_id
num_layers
run_id
source_type
d_model
num_heads
head_dim
ffn_dim
trainable_parameters
best_epoch
epochs_completed
total_optimizer_steps
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

# 175. Layer effect table

`s13_layer_effect.csv`:

```text
n1_run_id
n2_run_id
n1_rmse_wh
n2_rmse_wh
rmse_delta_n1_to_n2_wh
rmse_improvement_pct
n1_mae_wh
n2_mae_wh
mae_delta_n1_to_n2_wh
n1_r2
n2_r2
r2_delta
n1_parameters
n2_parameters
parameter_increase
parameter_increase_pct
rmse_winner
metric_ranking_divergence
status
```

---

# 176. Depth efficiency context

`s13_depth_efficiency_context.csv`:

```text
layer_id
validation_rmse_wh
validation_mae_wh
trainable_parameters
checkpoint_bytes
mean_epoch_seconds
samples_per_second_optional
peak_memory_optional
rmse_gain_per_10k_extra_params_if_applicable
selection_metric_used=false
status
```

---

# 177. Pareto context

`s13_depth_pareto_context.json`:

```text
accuracy_metric
cost_metrics
relationship
n1_dominates
n2_dominates
accuracy_efficiency_tradeoff
official_winner
tie_rule_used
context_only=true
status
```

---

# 178. Optimization diagnostics

`s13_optimization_diagnostics.csv`:

```text
layer_id
best_epoch
last_epoch
stop_reason
best_rmse_wh
last_rmse_wh
mean_grad_norm_preclip
max_grad_norm_preclip
normalized_grad_context_optional
mean_fraction_batches_clipped
max_fraction_batches_clipped
nonfinite_events
best_to_last_gap
status
```

---

# 179. Convergence diagnostics

`s13_convergence_diagnostics.csv`:

```text
layer_id
first_epoch_rmse_wh
best_epoch
best_rmse_wh
last_epoch
last_rmse_wh
early_stopped
epoch_cap_reached
steps_to_best
time_to_best_optional
post_best_worsening_epochs
status
```

---

# 180. Runtime/capacity diagnostics

`s13_runtime_capacity_diagnostics.csv`:

```text
layer_id
device
trainable_parameters
best_checkpoint_bytes
last_checkpoint_bytes
epochs_completed
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
samples_per_second_optional
peak_allocated_optional
peak_reserved_optional
memory_metric_available
runtime_comparable
status
```

---

# 181. Optional generalization diagnostics

`s13_generalization_diagnostics.csv`:

```text
layer_id
train_rmse_wh_at_best
validation_rmse_wh_at_best
rmse_gap_wh
train_mae_wh_at_best
validation_mae_wh_at_best
mae_gap_wh
depth_interpretation
status
```

No Test.

---

# 182. Hypothesis outcomes

`s13_hypothesis_outcomes.csv`:

```text
hypothesis_id
comparison
expected_direction_or_pattern
observed_metrics
capacity_context
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

# 183. Findings artifact

`s13_layer_findings.csv` possible codes:

```text
N1_GAIN
N2_GAIN
LAYER_EXACT_TIE
N1_DOMINATES
N2_ACCURACY_GAIN_WITH_COST
ACCURACY_EFFICIENCY_TRADEOFF
METRIC_RANKING_DIVERGENCE
PARAMETER_COUNT_GROWTH
LAYER_DELTA_PARAM_VERIFIED
LAYER_INDEPENDENCE_VERIFIED
OPTIMIZER_COVERAGE_VERIFIED
DROPOUT_DEPTH_CONTEXT
CHECKPOINT_SIZE_GROWTH
RUNTIME_DIFFERENCE
MEMORY_DIFFERENCE
POSSIBLE_N1_UNDERFITTING
POSSIBLE_N2_OVERFITTING
CONVERGENCE_DIFFERENCE
CLIPPING_DIFFERENCE
INITIALIZATION_POLICY_VERIFIED
SHARED_PREFIX_INIT_MATCH
SHARED_PREFIX_INIT_NOT_VERIFIABLE
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
ATTENTION_API_VERIFIED
INHERITED_WARNING
```

---

# 184. Winner artifact

`s13_layer_winner.json` minimum:

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
ffn_dim
selection_metric
selection_direction
tie_rule
winner_layer_id
winner_num_layers
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
winner_trainable_parameters
runner_up_layer_id
runner_up_num_layers
runner_up_rmse_wh
runner_up_trainable_parameters
rmse_margin_wh
rmse_margin_pct
parameter_difference
parameter_difference_pct
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 185. Phase36 reference update

`s13_reference_update.json`:

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
previous_num_layers=2
selected_layer_id
selected_num_layers
winner_run_id
winner_config_fingerprint
winner_rmse_wh
current_ffn_dim=128
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase36
```

---

# 186. Phase36 handoff logic

Phase 36 tests:

```text
F64 = FFN width 64
F128 = FFN width 128
F256 = FFN width 256
```

while holding selected N fixed.

S13 winner already uses:

```text
F128.
```

Therefore normally:

```text
reuse S13 winner as F128 reference
train new F64
train new F256.
```

if exact match.

---

# 187. FFN handoff nuance

Phase36 must preserve:

```text
D*
H*
HD*
selected N*
DR*
A*
P*
all optimization settings.
```

Only:

```text
ffn_dim
```

changes.

---

# 188. Figures

Recommended:

```text
S13_01_validation_rmse_by_epoch.png
S13_02_validation_mae_by_epoch.png
S13_03_train_loss_by_epoch.png
S13_04_gradient_clipping_fraction.png
S13_05_best_validation_metrics.png
S13_06_parameter_count_vs_rmse.png
S13_07_runtime_vs_rmse.png
S13_08_checkpoint_size_comparison.png
S13_09_convergence_summary.png
S13_10_generalization_gap_optional.png
```

Primary:

```text
S13_01_validation_rmse_by_epoch.png.
```

All figures must derive from stored artifacts.

---

# 189. Safe interpretation if N1 wins

> Under the selected width/head configuration and frozen F128/optimization settings, one Encoder layer achieved lower Validation RMSE than two layers while using fewer trainable parameters.

If N1 has lower RMSE and fewer parameters:

```text
N1 can be described as empirically dominating N2
```

for this registered Validation comparison.

Do not claim one layer is universally sufficient.

---

# 190. Safe interpretation if N2 wins

> Two Encoder layers achieved lower Validation RMSE than one layer under the frozen selected configuration, at the cost of additional trainable parameters and compute.

This is an:

```text
accuracy-capacity tradeoff
```

unless engineering metrics unexpectedly favor N2 too.

---

# 191. Safe interpretation if exact tie

> N1 and N2 produced exactly equal full-precision Validation RMSE; N1 was selected by the predefined parsimony rule.

---

# 192. Interpretation prohibitions

Do not claim:

```text
N2 sees more historical time
N2 has a longer lookback
N2 necessarily learns more meaningful attention
N1 is underfit because it is shallow
N2 is overfit because it is deep
more layers always improve Transformers
the second layer caused a specific physical interpretation
```

without evidence.

---

# 193. Sweep summary

Create:

```text
s13_layer_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
selected_d_model
selected_num_heads
selected_head_dim
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
dropout_probability
reference_run_id
n1_run_id
new_runs
reused_runs
candidate_layers
parameter_count_audit
layer_delta_parameter_audit
layer_independence_audit
optimizer_coverage_audit
primary_metric
metrics_by_layer
layer_effect
depth_efficiency_context
pareto_context
optimization_diagnostics
convergence_diagnostics
runtime_capacity_diagnostics
initialization_policy_status
sample_order_match
attention_api_status
winner
winner_margin
inherited_warnings
phase36_reference
test_status
overall_status
```

---

# 194. Human-readable report

Create:

```text
s13_layer_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S12
3. N1/N2 definitions
4. Selected D*/H*/head_dim context
5. What depth changes scientifically
6. What depth does NOT change: lookback/horizon
7. Frozen-variable contract
8. Architecture delta whitelist
9. Layer independence audit
10. Parameter-count/layer-delta audit
11. Optimizer coverage
12. Initialization/sample-order fairness
13. N2 reference provenance
14. N1 run provenance
15. Attention API sanity
16. Validation metrics
17. Layer effect
18. Learning-curve/convergence diagnostics
19. Gradient/clipping diagnostics
20. Runtime/capacity context
21. Optional Train-vs-Validation analysis
22. S13 winner
23. Interpretation cautions
24. Interaction limitations
25. Phase36 handoff
```

---

# 195. README

Create:

```text
README_S13_LAYER_SWEEP.md
```

Must explain:

```text
Purpose
S12 winner handoff
N1/N2 definitions
selected D*/H*/head_dim
same lookback despite depth change
extra complete Encoder block semantics
parameter-count increase
one-layer parameter delta
layer independence
optimizer coverage
dropout-depth structural exposure
no layer-wise LR/WD/dropout
no LayerDrop/deep supervision
initialization-policy nuance
attention list length 1 vs2
N2 reference reuse
winner/tie rules
capacity-efficiency context
interaction limitations
Phase36 handoff
No Test.
```

---

# 196. Recommended notebook structure

```text
Cell 35.1  Phase title
Cell 35.2  Verify Phase34 sign-off
Cell 35.3  Declare SWEEP_S13_LAYERS-v1
Cell 35.4  Load S12 winner/reference
Cell 35.5  Freeze all prior selected fields
Cell 35.6  Define N1/N2
Cell 35.7  Verify D*/H*/head_dim
Cell 35.8  Build run matrix
Cell 35.9  Audit common population
Cell 35.10 Audit encoder-stack geometry
Cell 35.11 Audit architecture-role delta
Cell 35.12 Audit state-dict key delta
Cell 35.13 Audit shared parameter schemas
Cell 35.14 Audit parameter counts
Cell 35.15 Audit one-layer parameter delta
Cell 35.16 Audit N2 layer independence
Cell 35.17 Audit optimizer coverage
Cell 35.18 Audit dropout-depth semantics
Cell 35.19 Audit training config
Cell 35.20 Audit initialization policy
Cell 35.21 Optional shared-prefix init audit
Cell 35.22 Audit sample-order policy
Cell 35.23 Verify N2 reference reuse
Cell 35.24 Build disposable N1 sanity model
Cell 35.25 N1 forward/backward sanity
Cell 35.26 N1 attention API sanity
Cell 35.27 Register N1 scientific run
Cell 35.28 Reseed + fresh loaders/model/optimizer
Cell 35.29 Train N1 via TRAINING_ENGINE-v1
Cell 35.30 Verify N1 BEST
Cell 35.31 Build run provenance
Cell 35.32 Build optimizer-budget audit
Cell 35.33 Build metrics
Cell 35.34 Compute N1→N2 effect
Cell 35.35 Build depth-efficiency context
Cell 35.36 Build Pareto context
Cell 35.37 Build optimization diagnostics
Cell 35.38 Build convergence diagnostics
Cell 35.39 Build runtime/capacity diagnostics
Cell 35.40 Optional Train-vs-Validation diagnostic
Cell 35.41 Evaluate hypotheses
Cell 35.42 Generate figures
Cell 35.43 Generate findings
Cell 35.44 Select winner
Cell 35.45 Write winner JSON
Cell 35.46 Write Phase36 reference update
Cell 35.47 Run S13 tests/discrepancies
Cell 35.48 Write summary/report
Cell 35.49 Register artifacts/checksums
Cell 35.50 Write README
Cell 35.51 Phase sign-off
```

---

# 197. Execution flow

```text
Verify Phase34
→ Load S12 winner
→ Freeze D*/H*/HD* and all prior winners
→ Define N1/N2
→ Audit same population
→ Audit exact depth-only architecture delta
→ Audit parameter count/layer delta
→ Verify N2 layers are independent
→ Verify optimizer coverage
→ Audit dropout-depth context
→ Verify N2 reference
→ Reuse N2
→ Build disposable N1 sanity
→ Register N1
→ Seed42 + fresh official objects
→ Train N1
→ Verify N1 BEST
→ Same-population Validation comparison
→ Compute depth effect
→ Analyze capacity/runtime context
→ Select minimum RMSE
→ Exact tie prefer N1
→ Update Phase36 reference
→ Write artifacts/sign-off
```

---

# 198. Fail-fast order

Before expensive N1 training:

```text
1. Phase34 sign-off
2. S12 winner identity
3. freeze all previous selections
4. N1/N2 registry
5. D*/H*/head_dim validity
6. same sample population
7. encoder-stack geometry
8. architecture delta whitelist
9. state-dict key delta
10. shared parameter schema
11. parameter-count direction
12. one-layer delta audit
13. N2 layer independence
14. optimizer coverage
15. PE/pooling/head invariance
16. dropout-depth contract
17. training config
18. N1 forward/backward sanity
19. attention API sanity
20. N2 reuse eligibility
21. Test firewall
22. Registry readiness
```

---

# 199. Critical technical note: depth does not extend lookback

Both N1/N2 receive identical:

```text
L*
```

historical observations.

The second layer transforms the already globally contextualized sequence representation again.

Do not describe N2 as “seeing twice as much history”.

---

# 200. Critical technical note: parameter growth must correspond exactly to a layer

If runtime:

```text
Params_N2 - Params_N1
```

does not correspond to N2-only second-layer parameters:

```text
STOP
```

and investigate.

Possible causes:

```text
hidden architecture drift
shared weights
head/pooling difference
optimizer/model construction bug.
```

---

# 201. Critical technical note: layer sharing would invalidate the capacity sweep

A true N2 reference must contain two independent Encoder layers.

If layer0 and layer1 share parameter tensors:

```text
S13 scientific interpretation is invalid.
```

Do not proceed until reconciled with Transformer-v1.

---

# 202. Critical technical note: same p does not mean same total dropout exposure

N2 executes more dropout-bearing layer modules.

That is an inherent depth consequence.

No `p` compensation.

---

# 203. Critical technical note: initialization equality is not whole-model applicable

N1 and N2 have different state sizes.

Fairness is initialization policy, not identical whole-model weights.

---

# 204. Critical technical note: N2 reference should not be retrained

Sequential reuse protects against choosing a favorable new stochastic realization of the reference.

Use existing N2 winner if exact contract matches.

---

# 205. Winner verification checklist

Before writing `s13_layer_winner.json`:

```text
[ ] N2 valid reused reference.
[ ] N1 valid completed run.
[ ] Same FV*/YS*/L*/P*/A*/B*/LR*/WD*/DR*/D*/H*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] F128 fixed.
[ ] PE fixed and applied once.
[ ] POST_NORM fixed.
[ ] Same mask policy.
[ ] Same input projection/pooling/head.
[ ] N1 exactly one independent layer.
[ ] N2 exactly two independent layers.
[ ] N2 no layer weight sharing.
[ ] Shared module geometries match.
[ ] N2-only keys map exactly to layer1.
[ ] Params_N1 < Params_N2.
[ ] Parameter delta validated.
[ ] Optimizer covers every trainable parameter once.
[ ] No layer-wise optimizer policy.
[ ] Same dropout p per active layer.
[ ] No LayerDrop/deep supervision.
[ ] Same initialization policy.
[ ] Sample-order policy audited.
[ ] No warm-start/truncation.
[ ] N1 BEST verified.
[ ] N2 BEST provenance valid.
[ ] Full-precision Validation RMSE used.
[ ] Capacity metrics secondary only.
[ ] Exact tie rule respected.
[ ] Phase36 reference generated.
[ ] No Test.
```

---

# 206. Phase36 handoff

Phase36 receives:

```text
s13_layer_winner.json
s13_reference_update.json
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
selected_num_layers
current_ffn_dim=128
population_fingerprint
```

and changes only:

```text
ffn_dim.
```

---

# 207. Reference reuse in Phase36

S13 winner already uses:

```text
F128.
```

Therefore normally:

```text
reuse S13 winner as F128 reference
train F64
train F256.
```

---

# 208. Relationship with Phase36 FFN sweep

S13 must not change FFN width to compensate depth.

Phase36 explicitly handles the FFN capacity dimension next.

---

# 209. Relationship with Phase37 Loss sweep

Loss remains:

```text
MSE
```

through S13.

No depth-specific loss.

---

# 210. Relationship with Phase38 Epoch-cap sweep

If N2 appears to still improve at E50 or N1 converges much earlier, record it.

Do not change epochs now.

Phase38 later handles the budget factor on the then-current reference.

---

# 211. Relationship with Phase39 Gradient clipping

Clip remains:

```text
1.0
```

for both.

Clipping frequency differences are diagnostic only.

---

# 212. Relationship with Phase42 Candidate synthesis

Both N1 and N2 remain in Experiment Registry.

Do not delete the losing depth candidate.

---

# 213. Relationship with Phase44 Rolling-origin robustness

S13 winner is based on one Validation period.

Temporal robustness is not yet established.

---

# 214. Relationship with Phase46 Multi-seed

S13 uses seed42 only.

Small N1/N2 margins are not seed-robust evidence.

---

# 215. Relationship with final attention analysis

Final code must dynamically handle:

```text
selected_num_layers.
```

No hard-coded assumption of two layers after S13.

---

# 216. Reproducibility metadata

New N1 run records:

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
batch
learning rate
weight decay
dropout
dropout scope
d_model
num_heads
head_dim
num_layers=1
ffn_dim=128
PE policy
POST_NORM
parameter count
architecture-role fingerprint
state-dict key fingerprint
initialization-policy fingerprint
sample-order provenance
optimizer group policy
population fingerprint
Training Engine fingerprint
BEST checkpoint checksum
history checksum
metric checksum
```

---

# 217. No fabricated outputs

Do not pre-fill:

```text
runtime parameter counts
parameter delta
checkpoint size
runtime
memory
winner
RMSE
best epoch
gradient norms
generalization gap
```

before execution.

Only registered architecture facts may be stated before runtime.

---

# 218. Phase sign-off

Create:

```text
phase_35_signoff.json
```

Minimum:

```text
phase = 35
phase_name = S13 Layer sweep
sweep_version
sweep_id
source_s12_winner_run_id
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
n1_run_id
n2_reference_run_id
new_run_ids
reused_run_ids
n1_parameter_count
n2_parameter_count
layer_parameter_delta
layer_independence_audit_status
optimizer_coverage_audit_status
winner_layer_id
winner_num_layers
winner_run_id
winner_rmse_wh
winner_parameter_count
population_fingerprint
metric_version
architecture_delta_audit_status
attention_api_audit_status
initialization_policy_audit_status
sample_order_match_status
inherited_warnings
test_status
approved_for_phase36
overall_status
created_at
```

---

# 219. Acceptance checklist

```text
[ ] Phase34 valid.
[ ] approved_for_phase35=true.
[ ] SWEEP_S13_LAYERS-v1 declared.
[ ] S12 winner loaded.
[ ] FV*/YS*/L*/P*/A*/B*/LR*/WD*/DR* fixed.
[ ] D* fixed.
[ ] H* fixed.
[ ] head_dim fixed.
[ ] N1 exactly 1.
[ ] N2 exactly 2.
[ ] No extra layer candidate.
[ ] F128 fixed.
[ ] D*%H*=0.
[ ] PE policy fixed.
[ ] PE applied once.
[ ] POST_NORM fixed.
[ ] mask policy fixed.
[ ] input projection fixed.
[ ] pooling fixed.
[ ] regression head fixed.
[ ] activation fixed.
[ ] dropout p fixed.
[ ] same dropout semantics per layer.
[ ] no dropout compensation.
[ ] no layer-specific p.
[ ] N1 has exactly one layer.
[ ] N2 has exactly two layers.
[ ] N2 layer0/layer1 independent.
[ ] no shared parameter storage.
[ ] same per-layer geometry.
[ ] N2-only architecture delta exactly one Encoder layer.
[ ] state-dict key delta audited.
[ ] N2-only keys correspond to layer1.
[ ] shared parameter shapes match.
[ ] Params_N1 < Params_N2.
[ ] parameter delta calculated at runtime.
[ ] N2-only param numel matches delta.
[ ] reference one-layer formula used only as audit.
[ ] optimizer covers every N1 parameter.
[ ] N2 optimizer coverage provenance valid.
[ ] no duplicate optimizer references.
[ ] no layer-wise LR.
[ ] no layer-wise WD.
[ ] no LayerDrop/DropPath.
[ ] no deep supervision.
[ ] no layer averaging.
[ ] no extra PE per layer.
[ ] same Train IDs.
[ ] same Validation IDs.
[ ] same WINDOWPOP-v1.
[ ] same feature fingerprint.
[ ] same X scaler.
[ ] same target transform.
[ ] same batch.
[ ] same AdamW.
[ ] same LR.
[ ] same WD.
[ ] same MSE.
[ ] same E50/patience10.
[ ] same min_delta.
[ ] same clip1.
[ ] no scheduler/warmup.
[ ] accumulation1.
[ ] drop_last=false.
[ ] seed42.
[ ] initialization policy same.
[ ] whole-state equality marked N/A.
[ ] optional shared-prefix init handled correctly.
[ ] sample-order policy same.
[ ] N2 reference exact-match.
[ ] N2 not retrained.
[ ] N1 sanity uses disposable objects.
[ ] official N1 run starts fresh.
[ ] N1 B1 output valid.
[ ] N1 selected-B output valid.
[ ] N1 partial batch valid.
[ ] N1 forward finite.
[ ] N1 backward finite.
[ ] N1 attention list length=1.
[ ] N2 attention contract length=2.
[ ] per-layer attention [B,H*,L,L].
[ ] standard/inspection prediction allclose.
[ ] official training attention OFF.
[ ] N1 registered before training.
[ ] N1 fresh model/loaders/optimizer.
[ ] no N2→N1 warm-start.
[ ] no layer truncation.
[ ] no optimizer-state reuse.
[ ] N1 trained through TRAINING_ENGINE-v1.
[ ] N1 BEST verified.
[ ] N2 BEST provenance valid.
[ ] same optimizer steps per complete epoch.
[ ] predictions finite.
[ ] official metrics in Wh.
[ ] full-precision RMSE used.
[ ] N1→N2 RMSE effect computed.
[ ] MAE/R² effects computed.
[ ] winner=min RMSE.
[ ] exact tie=N1.
[ ] capacity/runtime does not override non-tie RMSE.
[ ] no arbitrary gain threshold.
[ ] metric divergence recorded if present.
[ ] depth-efficiency context generated.
[ ] Pareto context generated.
[ ] optimization diagnostics generated.
[ ] gradient dimension caveat documented.
[ ] convergence diagnostics generated.
[ ] runtime/checkpoint diagnostics generated.
[ ] optional memory missing handled honestly.
[ ] no underfit claim from depth alone.
[ ] no overfit claim from depth alone.
[ ] hypotheses recorded.
[ ] findings generated.
[ ] inherited warnings propagated.
[ ] winner artifact generated.
[ ] Phase36 reference update generated.
[ ] F128 reuse identified.
[ ] no N0/N3/N4.
[ ] no D/H/FFN compensation.
[ ] no extra epochs.
[ ] no score-based rerun.
[ ] no failed/SANITY run ranked.
[ ] no attention-based winner selection.
[ ] no Test access.
[ ] single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] depth×D interaction documented.
[ ] depth×H interaction documented.
[ ] depth×FFN interaction documented.
[ ] depth×dropout interaction documented.
[ ] depth×optimizer interaction documented.
[ ] depth×budget interaction documented.
[ ] figures source-generated.
[ ] summary/report/README generated.
[ ] discrepancy log generated.
[ ] phase sign-off generated.
```

---

# 220. Acceptance criteria

Phase 35 chỉ PASS khi:

```text
S12-selected d_model/head configuration and all prior selected factors are fixed.

Exactly N1 and N2 are represented.

Only num_layers changes.

N1 has one complete independent Encoder layer.

N2 has two complete independent Encoder layers.

N2 second layer is not parameter-shared with layer0.

D*, H*, head_dim and F128 stay fixed.

PE is applied once and remains identical.

POST_NORM/mask/pooling/activation/dropout semantics stay fixed.

Same Train/Validation IDs and WINDOWPOP-v1 are used.

Parameter count increase is expected and verified.

N2-N1 parameter delta maps to exactly one additional Encoder layer.

Optimizer covers all active trainable parameters exactly once.

No layer-wise LR/WD/dropout or LayerDrop is introduced.

Same initialization policy and sample-order policy are audited.

N2 exact S12 reference is reused.

N1 is one fresh seed42 run.

No warm-start, truncation, layer averaging or deep supervision is used.

N1 BEST is verified with correct num_layers=1 config.

Validation RMSE Wh selects winner.

Exact RMSE tie selects N1.

Capacity/runtime metrics remain secondary.

Phase36 reference is generated.

Test remains untouched.
```

---

# 221. Failure conditions

Phase 35 FAIL if:

```text
wrong S12 winner used

D*/H*/FFN or prior selected settings change

N1/N2 values differ from registry

N2 layers share parameters

N2 extra layer is not optimized

parameter delta cannot be explained by one extra Encoder layer

PE is applied differently

LayerNorm policy changes

LayerDrop/DropPath is introduced

layer-specific dropout/LR/WD is introduced

deep supervision/intermediate head is added

layer outputs are averaged/ensembled

sample population differs

N2 reference mismatch is ignored

N2 is retrained and best rerun selected

N1 warm-starts from N2

N2 is truncated to create N1

optimizer state is reused

unregistered layer count is tested/used

RMSE rounded before ranking

runtime/parameter count overrides lower non-tied RMSE

one candidate fails but winner is still declared

score-based rerun occurs

Test is used.
```

---

# 222. Common mistakes

## 222.1 N1 dùng D64 nhưng N2 dùng D32

Sai. D* phải cố định.

## 222.2 N1 dùng H2, N2 dùng H4

Sai. H* phải cố định.

## 222.3 Giảm FFN cho N2 để match parameter count

Sai.

## 222.4 Giảm dropout của N2 để “bù” nhiều layer hơn

Sai two-factor experiment.

## 222.5 Reuse cùng một EncoderLayer object hai lần cho N2

Sai vì weight sharing.

## 222.6 Load N2 BEST rồi bỏ layer cuối để tạo N1

Warm-start/truncation confound.

## 222.7 Pool cả hai layer rồi average

Architecture drift.

## 222.8 Gắn loss vào từng layer

Deep supervision, không thuộc S13.

## 222.9 Add positional encoding trước mỗi layer

Sai Transformer-v1 architecture.

## 222.10 Cho rằng N2 có lookback dài gấp đôi

Sai. Lookback vẫn L*.

## 222.11 Chọn N1 vì nhanh hơn dù RMSE cao hơn

Sai, trừ exact tie.

## 222.12 Chọn N2 vì “deep hơn nên tốt hơn”

Sai.

## 222.13 Thử N3 sau khi N2 thắng

Hidden adaptive search.

## 222.14 Dùng attention đẹp hơn để chọn depth

Sai winner metric.

## 222.15 Dùng Test để chọn layer count

Forbidden.

---

# 223. Recommended execution pseudocode

```text
load_phase34_signoff()
assert_approved_for_phase35()

s12 = load_s12_winner()

FV = s12.feature_variant_id
YS = s12.target_scaling_id
L  = s12.lookback_id
P  = s12.pooling_id
A  = s12.activation_id
B  = s12.batch_id
LR = s12.learning_rate
WD = s12.weight_decay
DR = s12.dropout_probability
D  = s12.d_model
H  = s12.winner_num_heads
HD = s12.winner_head_dim

candidates = {
    "N1": 1,
    "N2": 2
}

assert D % H == 0
assert HD == D // H

audit_common_data_population()
audit_encoder_stack_geometry()
audit_architecture_role_delta()
audit_state_dict_key_delta()
audit_shared_parameter_schema()
audit_parameter_count_delta()
audit_n2_layer_independence()
audit_optimizer_coverage()
audit_dropout_depth_contract()
audit_training_config()
audit_initialization_policy()
audit_sample_order_policy()

n2_reference = resolve_s12_winner_run()
assert_exact_s13_reference_match(
    n2_reference,
    num_layers=2,
    d_model=D,
    num_heads=H
)

# Disposable sanity.
reseed(42)
sanity_model = build_transformer(
    input_size=feature_count(FV),
    d_model=D,
    num_heads=H,
    num_layers=1,
    ffn_dim=128,
    dropout=DR,
    activation=A,
    pooling=P
)
run_forward_backward_sanity(sanity_model)
run_attention_api_sanity(sanity_model)

# Official N1 run starts fresh.
register_n1_run()
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
    d_model=D,
    num_heads=H,
    num_layers=1,
    ffn_dim=128,
    dropout=DR,
    activation=A,
    pooling=P
)

optimizer = build_fresh_adamw(
    model=model,
    lr=LR,
    weight_decay=WD,
    group_policy=FROZEN_S9_GROUP_POLICY
)

n1_result = TRAINING_ENGINE_v1.fit(...)

verify_best_checkpoint(
    n1_result,
    num_layers=1
)

results = {
    "N1": n1_result,
    "N2": n2_reference
}

metrics = build_verified_s13_metrics(results)
effect = compute_n1_to_n2_effect(metrics)
capacity = build_depth_efficiency_context(results)
pareto = build_depth_pareto_context(results)
optimization = build_optimization_diagnostics(results)
convergence = build_convergence_diagnostics(results)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer="N1"
)

write_hypothesis_outcomes()
write_findings()
write_s13_winner(winner)
write_phase36_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 224. Definition of Done

\[
\boxed{
One\ Fixed\ Width/Head\ Setup
+
Two\ Depths
+
One\ Fresh\ N1
+
One\ Valid\ Reused\ N2
+
Independent\ Encoder\ Layers
+
Expected\ One\text{-}Layer\ Parameter\ Delta
+
Same\ Data/Optimization
+
Verified\ BEST\ Metrics
+
S13\ Winner
+
Phase36\ Reference
+
No\ Test
}
\]

---

# 225. Final status contract

```text
PHASE 35 tests num_layers only.

Current:
all winners from S1–S12.

Candidates:
N1 = 1 Encoder layer
N2 = 2 Encoder layers.

Frozen:
D*
H*
head_dim
F128
PE
pooling
activation
dropout
batch
LR
WD
MSE
E50
patience10
clip1
Training Engine
Metric version.

N2:
reuse S12 winner if exact match.

N1:
one fresh seed42 run.

Expected differences:
num_layers
second Encoder block
parameter count
state-dict key count
attention list length
runtime/checkpoint/memory context.

Expected invariants:
same input projection
same PE
same per-layer geometry
same pooling
same regression head
same data/population
same optimizer/training protocol.

Hard:
N2 layers must be independent.
No layer weight sharing.
No layer-wise LR/WD/dropout.
No LayerDrop.
No deep supervision.
No layer averaging.
No PE reapplication.
No warm-start/truncation.

Depth does NOT change:
lookback
forecast horizon
target population.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer N1.

Capacity/runtime:
secondary context only.

No Test.

After SWEEP_S13_LAYERS-v1 PASS:
proceed to
PHASE 36 — S14 FFN sweep.
```

---

# 226. Final check

Correct:

```text
Load S12 winner
→ freeze D*/H*/all previous winners
→ N1 vs N2
→ verify depth-only topology delta
→ verify N2 layer independence
→ verify one-layer parameter delta
→ reuse N2
→ fresh N1
→ verify N1 BEST
→ same-population Validation comparison
→ select min RMSE
→ exact tie N1
→ Phase36 reference
```

Incorrect:

```text
N1 with different H/D
→ reduce FFN/dropout for N2
→ reuse same layer object twice
→ truncate N2 BEST into N1
→ add deep supervision
→ choose by attention plots
→ inspect Test
```

Chỉ sau khi `SWEEP_S13_LAYERS-v1` được sign-off mới chuyển sang **PHASE 36 — S14 FFN sweep**.
