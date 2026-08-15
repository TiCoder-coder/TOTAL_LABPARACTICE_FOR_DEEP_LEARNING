# PHASE 36 — S14 FFN SWEEP

## Kế hoạch controlled sweep cho Feed-Forward Network Width (`ffn_dim`) của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S13_LAYERS-v1`  
**Sweep ID:** `S14_FFN`  
**Output version:** `SWEEP_S14_FFN-v1`  
**Phase trước:** `Phase_35_S13_Layer_sweep.md`

---

# 1. Vai trò của Phase 36

Phase 36 là controlled experiment thứ mười bốn trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target scaling, lookback, pooling, activation, batch size, learning rate, weight decay, dropout, `d_model`, số attention heads, số encoder layers, loss, epoch budget, gradient clipping, sample population và seed đã được khóa từ Phase 35, FFN hidden width nào phù hợp hơn cho Transformer Encoder: `64`, `128` hay `256`?

Phase 36 chỉ thay đúng một registered hyperparameter:

```text
ffn_dim
```

với ba condition:

```text
F64  = FFN hidden width 64
F128 = FFN hidden width 128
F256 = FFN hidden width 256
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ FFN\ Width\ Factor
+
Fixed\ Attention\ Geometry
+
Expected\ Capacity\ Change
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

# 2. Vị trí Phase 36 trong master execution plan

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
```

Phase 36 không được quay lại thay:

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
loss
epoch cap
gradient clipping
RevIN
boundary protocol
```

---

# 3. Câu hỏi nghiên cứu của S14

Phase 36 phải trả lời:

```text
1. F64, F128 hay F256 tạo Validation RMSE Wh thấp nhất?

2. FFN bottleneck nhỏ hơn có đủ nonlinear transformation capacity không?

3. FFN rộng hơn có cải thiện forecasting hay chỉ tăng parameter/compute burden?

4. Parameter count tăng bao nhiêu giữa F64 → F128 → F256?

5. FFN expansion ratio thay đổi thế nào dưới selected d_model D*?

6. Learning curves, best epoch và early stopping behavior thay đổi ra sao?

7. Gradient norm/clipping behavior thay đổi thế nào?

8. Runtime/checkpoint/memory context thay đổi ra sao?

9. MAE và R² có cùng ranking với RMSE không?

10. FFN width nào trở thành current reference cho Phase 37?
```

---

# 4. Current reference từ Phase 35

Phase 36 phải load:

```text
s13_layer_winner.json
s13_reference_update.json
phase_35_signoff.json
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
N*  = selected num_layers
```

Possible:

```text
D* ∈ {32,64}
H* ∈ {2,4}
N* ∈ {1,2}
```

Phase 36 không hard-code:

```text
D64
H4
N2
```

---

# 5. Canonical FFN options

Registry:

```text
F64  = 64
F128 = 128
F256 = 256
```

Exact numeric values:

```text
64
128
256
```

No other FFN width may enter S14 ranking.

---

# 6. Reference FFN width

Current sequential reference từ Phase 35 still uses:

```text
F128 = 128
```

vì FFN width chưa được swept trước Phase 36.

If exact S14 contract match:

```text
F128 → REUSE S13 winner
F64  → NEW fresh run
F256 → NEW fresh run
```

Normal S14 execution:

```text
2 new scientific runs
+
1 reused F128 reference
```

---

# 7. Scientific meaning của FFN width

Trong mỗi Transformer Encoder layer, FFN path có dạng:

```text
[B,L,D*]
→ Linear(D*, M)
→ A*
→ Dropout(DR*)
→ Linear(M, D*)
→ Dropout / residual path according to Transformer-v1
→ [B,L,D*]
```

Trong đó:

```text
M = ffn_dim.
```

S14 thay:

```text
M ∈ {64,128,256}
```

while keeping attention geometry fixed.

---

# 8. FFN does not change external representation width

Regardless of M:

```text
input to FFN:
[B,L,D*]

output from FFN:
[B,L,D*]
```

Only the hidden expansion/contraction width changes.

Therefore final encoder output and regression output remain:

```text
[B,L,D*]
→ pooling
→ [B,D*]
→ Linear(D*,1)
→ [B,1].
```

---

# 9. FFN hidden geometry

F64:

```text
Linear(D*,64)
→ A*
→ Dropout
→ Linear(64,D*)
```

F128:

```text
Linear(D*,128)
→ A*
→ Dropout
→ Linear(128,D*)
```

F256:

```text
Linear(D*,256)
→ A*
→ Dropout
→ Linear(256,D*)
```

---

# 10. Selected d_model is frozen

Hard:

```text
d_model = D*
```

No d_model change to preserve a preferred FFN ratio.

---

# 11. Selected head count is frozen

Hard:

```text
num_heads = H*
head_dim = HD*
```

Attention geometry is identical across F64/F128/F256.

---

# 12. Selected layer count is frozen

Hard:

```text
num_layers = N*
```

resolved from S13.

Every active Encoder layer uses the same candidate FFN width.

No per-layer FFN heterogeneity.

---

# 13. FFN width is global across active layers

If N*=1:

```text
layer0 FFN = candidate M.
```

If N*=2:

```text
layer0 FFN = candidate M
layer1 FFN = candidate M.
```

Do not test:

```text
layer0 F64
layer1 F256
```

inside S14.

That would be a per-layer architecture experiment.

---

# 14. Expansion ratio depends on D*

Define:

\[
r_{FFN}=\frac{ffn\_dim}{d_{model}}
\]

This ratio is a derived context value, not an independently controlled hyperparameter.

---

# 15. Expansion ratios if D32 was selected

If:

```text
D*=32
```

then:

```text
F64  → ratio 2
F128 → ratio 4
F256 → ratio 8
```

---

# 16. Expansion ratios if D64 was selected

If:

```text
D*=64
```

then:

```text
F64  → ratio 1
F128 → ratio 2
F256 → ratio 4
```

These differences are inherent consequences of fixed D*.

---

# 17. Do not preserve expansion ratio by changing d_model

Forbidden:

```text
D32/F128
vs
D64/F256
```

as an S14 comparison.

That changes:

```text
d_model + ffn_dim.
```

S14 answers:

> Which absolute FFN hidden width among 64/128/256 works best under the already selected D*?

---

# 18. Activation remains frozen

Every candidate uses selected:

```text
A*
```

from S6.

No:

```text
ReLU for F64
GELU for F256.
```

---

# 19. Dropout remains frozen

Every candidate uses:

```text
DR*
```

at the exact same semantic dropout sites.

However hidden FFN tensors have different sizes:

```text
[B,L,64]
[B,L,128]
[B,L,256]
```

so elementwise dropout mask shapes differ.

This is expected.

---

# 20. No dropout compensation for FFN size

Do not change:

```text
p_F64
p_F128
p_F256
```

to equalize number of dropped activations.

Use exactly the S10-selected `DR*`.

Changing `p` would create a two-factor experiment.

---

# 21. Attention path is structurally unchanged

F64/F128/F256 must have identical:

```text
MHA embed_dim = D*
num_heads = H*
head_dim = HD*
Q/K/V projection geometry
attention dropout probability
MHA output projection
```

Only FFN sublayers change width.

---

# 22. Attention weight shape is unchanged

Per active layer:

```text
[B,H*,L*,L*]
```

for all candidates.

Number of attention tensors remains:

```text
N*.
```

No attention output geometry change due FFN width.

---

# 23. No scientific attention analysis in S14

Do not choose FFN width using:

```text
attention heatmaps
head entropy
head diversity
attention concentration
```

S14 winner is Validation RMSE-based.

Attention API sanity only.

---

# 24. Positional encoding remains unchanged

Because D* and L* remain fixed:

```text
PE policy identical
PE width identical
PE buffer shape identical
PE application count identical.
```

Any PE difference is a discrepancy.

---

# 25. Input projection remains unchanged

All:

```text
Linear(F,D*)
```

same shape and semantic role.

---

# 26. LayerNorm remains unchanged in shape/count

Because:

```text
D*
N*
```

are fixed:

```text
LayerNorm normalized_shape
LayerNorm count
LayerNorm positions
```

must be identical across candidates.

---

# 27. Regression head remains unchanged

All:

```text
Linear(D*,1)
```

same shape.

---

# 28. Pooling remains unchanged

All use selected:

```text
P*
```

after final active Encoder layer.

---

# 29. Parameter count is expected to change

FFN width changes trainable capacity.

Expected monotonicity:

```text
Params_F64
<
Params_F128
<
Params_F256.
```

This is a hard expected direction under Transformer-v1.

Runtime parameter count is authoritative.

---

# 30. Per-layer FFN parameter contribution

Ignoring FFN-independent components, one Encoder layer's FFN has:

```text
Linear(D,M):
D*M weights + M bias

Linear(M,D):
M*D weights + D bias
```

So M-dependent contribution is:

\[
2DM + M
=
(2D+1)M.
\]

The `+D` second-layer bias is constant with respect to M.

---

# 31. Parameter delta formula across FFN candidates

At fixed D and N:

\[
\Delta Params
=
N(2D+1)\Delta M
\]

under the canonical bias assumptions.

Therefore:

\[
\Delta_{64\rightarrow128}
=
N^*(2D^*+1)\times64
\]

\[
\Delta_{128\rightarrow256}
=
N^*(2D^*+1)\times128
\]

\[
\Delta_{64\rightarrow256}
=
N^*(2D^*+1)\times192.
\]

Audit reference only.

Runtime count is source of truth.

---

# 32. Full model reference formula

Under established Transformer-v1 assumptions:

\[
Params(F,D,N,M)
=
FD
+
N(4D^2+2DM+9D+M)
+
2D
+
1.
\]

For S14, only `M` changes.

Therefore the total count is affine in M under this reference implementation.

Do not hard-code formula outputs as observed results.

---

# 33. Parameter delta is easier to verify than full formula

Even if full runtime formula differs slightly due implementation details, S14 should still verify whether:

```text
Params_F128 - Params_F64
Params_F256 - Params_F128
```

are explained exactly by the FFN width changes.

This provides a strong architecture correctness check.

---

# 34. FFN-only state-dict shape changes

Expected parameter shape differences are confined to FFN Linear tensors in every active layer.

Typical changed tensors:

```text
encoder.layers.i.linear1.weight
encoder.layers.i.linear1.bias
encoder.layers.i.linear2.weight
```

and possibly implementation-equivalent names.

`linear2.bias` shape remains `[D*]` and should not change.

---

# 35. Architecture shape-delta whitelist

Allowed differences:

```text
FFN hidden width config
FFN first Linear weight shape
FFN first Linear bias shape
FFN second Linear weight shape
FFN parameter count
whole-model parameter count
checkpoint size
runtime/memory
config fingerprint
```

Not allowed:

```text
MHA shapes
LayerNorm shapes
input projection
PE
pooling
regression head
num_layers
num_heads
d_model
dropout probability
activation
```

---

# 36. State-dict key names should remain the same

Because all candidates have the same module topology and only Linear sizes change:

```text
state_dict key set_F64
==
state_dict key set_F128
==
state_dict key set_F256
```

is normally expected.

Tensor shapes for FFN-specific keys differ.

If key sets differ:

```text
investigate hidden topology drift.
```

---

# 37. Architecture-role fingerprint should remain identical

Build a semantic fingerprint over:

```text
module path
module class
semantic role
layer index
trainable flag
```

Expected:

```text
same architecture-role fingerprint
```

across F64/F128/F256.

---

# 38. Parameter-schema equality is NOT expected

FFN tensor shapes differ.

Therefore:

```text
parameter_schema_fingerprint
```

may differ.

Do not require whole parameter-schema equality.

Instead audit:

```text
same key set
only whitelisted FFN shapes differ.
```

---

# 39. Initialization fairness

Whole initial states cannot be identical because FFN tensors have different shapes.

Fairness is:

```text
same seed42
same initialization algorithm
same model builder
same PyTorch environment
same custom-init policy
same module-construction policy.
```

---

# 40. RNG construction-order nuance

Changing FFN width changes the number of random values consumed when initializing FFN tensors.

If N*>1, this can shift RNG state before construction of later Encoder layers or the regression head.

Therefore even some shape-identical later modules may initialize differently under the same seed.

This is expected under a sequential builder.

Do not falsely require all shared modules to have identical initial values.

---

# 41. Initialization-policy fingerprint

Create a fingerprint over:

```text
seed policy
model builder code fingerprint
PyTorch version
custom initialization policy
module construction order
bias policy
LayerNorm init policy.
```

Expected:

```text
same policy fingerprint
```

across candidates.

---

# 42. Optional shared-prefix initialization audit

If constructor order guarantees that modules are built before the first M-dependent tensor:

```text
input projection
possibly early attention weights
```

may initialize identically.

This can be audited where meaningful.

Allowed status:

```text
MATCH
NOT_MATCH_DUE_RNG_CONSUMPTION
NOT_VERIFIABLE
NOT_APPLICABLE.
```

Not a winner criterion.

---

# 43. Same sample-order policy

Because data and B* are fixed:

```text
same Train permutation
same batch grouping
same worker policy
same loader generator policy
```

should be used for fresh F64/F256 runs.

For historical F128 reference:

```text
compare if provenance exists
else NOT_VERIFIABLE.
```

Do not retrain F128 to recover metadata.

---

# 44. Dropout RNG masks are not expected to match

FFN hidden tensors differ in shape, so FFN dropout masks differ.

Even residual-path dropout RNG can diverge later because random-number consumption changes.

Fairness does not require identical masks.

---

# 45. No FFN weight transfer

Forbidden:

```text
slice F256 → F128
slice F128 → F64
pad F64 → F128
interpolate FFN weights
copy first hidden units
```

Official F64/F256 runs must be fresh.

---

# 46. No neuron matching

Do not attempt:

```text
match first 64 hidden neurons across models
```

as a training requirement.

The candidate is the FFN width itself.

---

# 47. No optimizer-state reuse

Every new run uses fresh:

```text
AdamW moments
step counters.
```

No F128→F64/F256 optimizer transfer.

---

# 48. No FFN-specific optimizer group

Do not introduce:

```text
special LR for FFN
special WD for FFN
```

inside S14.

Use frozen optimizer-group policy from S9.

---

# 49. Optimizer coverage

Every active trainable parameter must appear exactly once in optimizer groups.

For F64/F256:

```text
all resized FFN tensors must be covered.
```

No missing new hidden-unit parameters.

---

# 50. Working hypotheses

## H-S14-01 — F128 may remain a balanced reference

F128 may provide sufficient nonlinear capacity without excessive parameter growth.

Status:

```text
UNTESTED.
```

---

# 51. H-S14-02 — F64 may be sufficient

A narrower FFN may match or improve Validation RMSE with fewer parameters.

Status:

```text
UNTESTED.
```

---

# 52. H-S14-03 — F256 may improve capacity

A wider FFN may improve representation transformation and reduce Validation RMSE.

Status:

```text
UNTESTED.
```

---

# 53. H-S14-04 — Excessive FFN width may overfit or optimize less efficiently

F256 may:

```text
reduce Train error
but fail to improve Validation
```

under the same regularization/training budget.

Hypothesis only.

---

# 54. Preconditions

Required:

```text
Phase 35 = PASS
```

or:

```text
PASS_WITH_WARNING
```

with no unresolved critical issue.

Also:

```text
approved_for_phase36 = true.
```

---

# 55. Required upstream artifacts

```text
s13_layer_winner.json
s13_reference_update.json
phase_35_signoff.json
```

---

# 56. Required upstream contracts

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
```

---

# 57. Carry-forward warnings

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

Write into:

```text
S14 manifest
S14 summary
S14 report
S14 winner
Phase37 handoff.
```

---

# 58. Swept factor only

Canonical:

```text
ffn_dim
```

Allowed:

```text
64
128
256
```

Aliases:

```text
F64
F128
F256
```

No:

```text
F32
F192
F512
```

inside S14.

---

# 59. Frozen data contract

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
same target transform
same exact Validation ordering.
```

---

# 60. Frozen architecture settings

Hard:

```text
D*
H*
HD*
N*
P*
A*
DR*
sinusoidal PE
POST_NORM
no causal mask
no padding mask
Linear(D*,1) regression head.
```

Only `ffn_dim` changes.

---

# 61. Frozen optimizer/training settings

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
gradient accumulation=1
mixed precision policy fixed
seed42
TRAINING_ENGINE-v1
METRICS-v1.
```

---

# 62. F128 reference reuse gate

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
H1
WB0
WINDOWPOP-v1

F128
same PE
POST_NORM
same mask policy
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
METRICS-v1.
```

Mismatch:

```text
STOP.
```

---

# 63. Fresh F64 run

Execution:

```text
register F64
↓
reseed42
↓
fresh Train loader
↓
fresh Validation loader
↓
fresh Transformer(
    d_model=D*,
    num_heads=H*,
    num_layers=N*,
    ffn_dim=64,
    dropout=DR*,
    activation=A*,
    pooling=P*
)
↓
fresh AdamW(LR*,WD*)
↓
MSE
↓
TRAINING_ENGINE-v1.
```

No warm-start.

---

# 64. Fresh F256 run

Execution:

```text
register F256
↓
reseed42
↓
fresh Train loader
↓
fresh Validation loader
↓
fresh Transformer(
    d_model=D*,
    num_heads=H*,
    num_layers=N*,
    ffn_dim=256,
    dropout=DR*,
    activation=A*,
    pooling=P*
)
↓
fresh AdamW(LR*,WD*)
↓
MSE
↓
TRAINING_ENGINE-v1.
```

No warm-start.

---

# 65. Run-order independence

Each new condition independently performs:

```text
reseed
fresh loaders
fresh model
fresh criterion
fresh optimizer
fresh Training Engine state.
```

F64 run must not alter F256 initial state via shared mutable objects.

---

# 66. No F128 retraining

Do not retrain F128 to:

```text
obtain better RMSE
match initialization artifacts
match sample order metadata
```

Reuse exact S13 winner.

---

# 67. F64 geometry assertions

For every active Encoder layer:

```text
FFN first Linear:
D* → 64

FFN second Linear:
64 → D*
```

with:

```text
A*
DR*
```

unchanged.

---

# 68. F128 reference geometry assertions

Every active layer:

```text
D* → 128 → D*
```

---

# 69. F256 geometry assertions

Every active layer:

```text
D* → 256 → D*
```

---

# 70. All active layers use same FFN candidate

If N*=2:

```text
layer0 M = candidate
layer1 M = candidate.
```

Hard assert.

No mixed widths.

---

# 71. Input/output shape sanity

For all candidates:

```text
[B,L*,F]
→ [B,1]
```

for:

```text
B=1
selected B*
final partial batch.
```

---

# 72. FFN internal shape sanity

Instrument a disposable forward hook or direct module inspection to verify hidden FFN activation shapes:

```text
F64  → [...,64]
F128 → [...,128]
F256 → [...,256].
```

Do not keep heavy hooks during official training.

---

# 73. MHA geometry invariance audit

Across all candidates:

```text
embed_dim = D*
num_heads = H*
head_dim = HD*
projection shapes same
attention shape same.
```

Any MHA parameter shape drift:

```text
FAIL.
```

---

# 74. LayerNorm invariance audit

All candidate LayerNorm modules:

```text
same count
same normalized_shape D*
same affine policy.
```

---

# 75. PE invariance audit

All:

```text
same PE buffer shape
same PE values/checksum if environment/config identical.
```

Since D and L are unchanged, exact buffer equality should normally hold.

---

# 76. Regression-head invariance audit

All:

```text
Linear(D*,1)
```

same parameter shape.

---

# 77. Shared key-set audit

Expected:

```text
same state_dict key names
```

across F64/F128/F256.

Hard investigate if false.

---

# 78. FFN shape-delta audit

For every active layer, changed shapes should map exactly to FFN hidden-dimension-dependent tensors.

Expected:

```text
linear1.weight:
[M,D*]

linear1.bias:
[M]

linear2.weight:
[D*,M]

linear2.bias:
[D*] unchanged.
```

Exact module naming depends on custom encoder implementation.

---

# 79. Parameter-count monotonicity audit

Required runtime:

```text
Params_F64 < Params_F128 < Params_F256.
```

If not:

```text
STOP
→ inspect implementation.
```

---

# 80. Pairwise parameter delta audit

Compute:

```text
ΔP_64_128
ΔP_128_256
ΔP_64_256
```

Compare against:

\[
N^*(2D^*+1)\Delta M
\]

when canonical assumptions apply.

---

# 81. Optimizer coverage audit

For F64/F256:

```text
every trainable parameter included exactly once
```

and all FFN parameters included.

No stale parameter references from another candidate model.

---

# 82. Disposable sanity models

Before official training, build disposable F64/F256 models for:

```text
shape tests
parameter audits
forward/backward sanity
attention API sanity.
```

After sanity:

```text
delete/discard them
reseed
recreate official objects fresh.
```

This avoids RNG contamination.

---

# 83. Forward sanity

For F64/F256:

```text
finite output
shape [B,1].
```

---

# 84. Backward sanity

Use small Train/Val-compatible batch or synthetic input:

```text
MSE
backward
finite gradients.
```

No optimizer step needed for pure architecture sanity.

---

# 85. Attention API sanity

For each new candidate in eval mode:

```text
attention list length = N*
per-layer attention [B,H*,L*,L*]
prediction [B,1]
finite
nonnegative
row sums ≈1.
```

---

# 86. Standard/inspection prediction equivalence

For fixed candidate/input in eval:

```text
model(x)
```

and:

```text
forward_with_attention(x).prediction
```

must be allclose.

---

# 87. Official training attention path

Hard:

```text
need_weights=False.
```

No scientific attention extraction during S14 training.

---

# 88. Same target population

All three candidate results must be evaluated on exact same:

```text
Validation sample IDs
Validation order
target values
Wh-unit metrics.
```

---

# 89. Primary metric

Hard:

```text
best_validation_rmse_wh
```

from verified BEST checkpoint.

---

# 90. Secondary scientific metrics

Record:

```text
Validation MAE Wh
Validation R²
best epoch
epochs completed
stop reason
train loss
gradient/clipping diagnostics
convergence diagnostics.
```

---

# 91. Capacity/engineering metrics

Record:

```text
trainable parameter count
parameter increase
checkpoint size
mean epoch runtime
throughput
optional peak memory
time to best.
```

Secondary only.

---

# 92. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{F64},
RMSE_{F128},
RMSE_{F256}
\right)
\]

using full-precision Validation RMSE Wh.

---

# 93. Exact RMSE tie rule

If exact full-precision RMSE equality among minimum candidates:

```text
prefer smaller FFN width.
```

Tie order:

```text
F64
<
F128
<
F256.
```

Rationale:

```text
fewer trainable parameters
lower expected compute/memory
simpler model
predeclared parsimony.
```

Only exact tie invokes this rule.

---

# 94. No efficiency override for non-tie

If F256 has lower RMSE by any non-zero full-precision margin:

```text
F256 wins.
```

Do not choose F64/F128 solely because they are smaller.

---

# 95. No arbitrary practical threshold

Do not require:

```text
F256 must improve >1%
```

to win.

Strict lower RMSE wins unless exact tie.

---

# 96. Pairwise effects

Required:

```text
F64 → F128
F128 → F256
F64 → F256.
```

Define:

\[
\Delta RMSE_{left\rightarrow right}
=
RMSE_{left}-RMSE_{right}.
\]

Positive:

```text
right candidate improves.
```

---

# 97. Relative pairwise effect

\[
Improvement\%
=
100\times
\frac{RMSE_{left}-RMSE_{right}}
{RMSE_{left}}.
\]

Also compute:

```text
MAE delta
R² delta
parameter delta
runtime delta
checkpoint-size delta.
```

---

# 98. FFN trend classification

With ordered widths:

```text
64 < 128 < 256
```

classify descriptively:

```text
NARROW_BEST
MIDDLE_BEST
WIDE_BEST
MONOTONIC_GAIN_WITH_WIDTH
MONOTONIC_DEGRADATION_WITH_WIDTH
U_SHAPED
INVERTED_U
MIXED
EXACT_TIE_PATTERN.
```

Do not fit a continuous optimum beyond tested values.

---

# 99. Boundary-winner caution

F64 and F256 are boundaries of the registered S14 range.

If either wins:

```text
winner_is_boundary = true.
```

Do not automatically test:

```text
F32
F512
```

inside S14.

---

# 100. Metric ranking divergence

If RMSE winner differs from MAE winner:

```text
METRIC_RANKING_DIVERGENCE.
```

Winner remains RMSE-based.

---

# 101. BEST verification for F64

After official F64 run:

```text
fresh Transformer(ffn_dim=64)
↓
verify full config
↓
strict-load BEST
↓
model.eval()
↓
full ordered Validation
↓
inverse target transform if needed
↓
METRICS-v1
↓
verify stored BEST.
```

---

# 102. BEST verification for F256

Same with:

```text
ffn_dim=256.
```

---

# 103. F128 reference verification

No retraining.

Verify:

```text
selected D*/H*/N*
F128
all prior selected factors
same population
same metric version
same Training Engine provenance
BEST already verified.
```

---

# 104. Checkpoint metadata

Every candidate run must preserve:

```text
ffn_dim
d_model
num_heads
head_dim
num_layers
dropout
activation
pooling
parameter count
model config fingerprint
architecture-role fingerprint.
```

---

# 105. Wrong-width FFN checkpoint load

Because FFN tensors differ in shape:

```text
strict F256 state_dict → F64-config model
```

should fail.

Do not use:

```text
strict=False
slicing
partial FFN load
```

to make incompatible checkpoints load.

---

# 106. Shared non-FFN tensors may have compatible shapes

This does not authorize partial checkpoint transfer.

Official new candidates remain fresh.

---

# 107. Same DataLoader settings

All:

```text
B*
shuffle Train=true
shuffle Validation=false
drop_last=false
same workers
same generator policy.
```

---

# 108. Optimizer steps per complete epoch

Because sample count and batch size are fixed:

```text
steps_per_epoch_F64
=
steps_per_epoch_F128
=
steps_per_epoch_F256.
```

Total optimizer steps may differ through early stopping.

---

# 109. Gradient diagnostics

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
larger FFN → more trainable parameters.
```

Raw global L2 gradient norms are dimension-sensitive.

---

# 110. Optional normalized gradient context

If useful:

```text
global_grad_norm / sqrt(trainable_parameter_count)
```

diagnostic only.

Do not select winner by this.

---

# 111. Convergence diagnostics

Record:

```text
first epoch RMSE
best epoch
best RMSE
last RMSE
best-to-last gap
early stopped?
epoch cap reached?
steps to best
post-best worsening epochs.
```

---

# 112. Narrow-FFN underfitting-like signal

Possible F64 evidence:

```text
higher Train RMSE
higher Validation RMSE
early plateau.
```

Do not infer underfitting from width alone.

---

# 113. Wide-FFN overfitting-like signal

Possible F256 evidence:

```text
lower Train error
worse Validation RMSE
larger post-best degradation.
```

Descriptive only.

---

# 114. Runtime diagnostics

Record:

```text
mean epoch seconds
median epoch seconds
total runtime
samples/sec
time to best optional.
```

Expected compute increases with M, but measured runtime is source of truth.

---

# 115. Checkpoint-size diagnostics

Record:

```text
BEST full checkpoint bytes
LAST full checkpoint bytes
optional model state_dict bytes.
```

Larger FFN should generally produce larger checkpoints.

Do not mix full-checkpoint and model-only size.

---

# 116. Memory diagnostics

Optional:

```text
peak allocated
peak reserved
```

if backend supports reliable measurements.

If unavailable:

```text
NOT_AVAILABLE.
```

No fabrication.

---

# 117. Activation-memory context

The FFN hidden activation tensor scales with:

```text
B × L* × M
```

per active layer during forward/backward.

Thus F256 carries a larger FFN hidden activation footprint than F64.

This is theoretical context; actual peak memory depends on autograd/backend.

---

# 118. Optimizer-state memory context

Larger FFN means more trainable parameters, so AdamW moment-state memory also grows.

Record actual memory only when measurable.

Do not claim exact byte ratios from theory as observed results.

---

# 119. Single-seed limitation

Mandatory:

```text
S14 uses seed42 only.
```

No mean±std.

---

# 120. Validation-only limitation

Mandatory:

```text
S14 winner is selected using Validation only.
```

No Test evidence.

---

# 121. FFN × d_model interaction limitation

S14 is conditional on selected D*.

Because expansion ratios differ with D*, the best absolute M may differ under another D.

No D×FFN grid.

---

# 122. FFN × depth interaction limitation

S14 is conditional on selected N*.

If N2 is selected, the candidate FFN width is repeated in two layers.

No N×FFN grid.

---

# 123. FFN × heads interaction limitation

H* fixed.

No H×FFN grid.

---

# 124. FFN × activation interaction limitation

A* fixed.

Different FFN widths may interact with ReLU/GELU, but no joint sweep.

---

# 125. FFN × dropout interaction limitation

DR* fixed.

No FFN×dropout grid.

---

# 126. FFN × LR/WD interaction limitation

LR*/WD* fixed.

No optimizer retuning per width.

---

# 127. FFN × batch interaction limitation

B* fixed.

---

# 128. FFN × epoch-budget interaction limitation

All candidates use:

```text
E50/patience10.
```

No extra epochs for F256.

Phase 38 later tests epoch cap on current reference.

---

# 129. Sequential-selection limitation

By Phase 36, Validation has influenced:

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
S14 FFN.
```

Complete lineage and later robustness checks are mandatory.

---

# 130. No Test access

Hard:

```text
Test loader not iterated
Test metrics absent
Test predictions absent.
```

---

# 131. No F32/F512 adaptive extension

No.

---

# 132. No mixed FFN widths across layers

No.

---

# 133. No FFN bottleneck residual adapter

No.

---

# 134. No gated FFN conversion

Do not replace FFN with:

```text
GEGLU
SwiGLU
GLU
gated MLP
```

inside S14.

That changes architecture type, not only width.

---

# 135. No extra FFN Linear layer

Do not make F256 use a deeper MLP.

FFN depth remains:

```text
Linear → activation → dropout → Linear.
```

---

# 136. No FFN-specific normalization

Do not add:

```text
BatchNorm
extra LayerNorm inside FFN hidden path
```

for any candidate.

---

# 137. No activation change

No.

---

# 138. No dropout change

No.

---

# 139. No optimizer-state reuse

No.

---

# 140. No warm-start

No.

---

# 141. No parameter slicing/padding

No.

---

# 142. No attention-based selection

No.

---

# 143. Run failure policy

If F64 or F256 has a technical failure:

```text
S14 incomplete
```

until a documented technical rerun succeeds.

Do not rank only the surviving conditions.

---

# 144. Technical rerun allowed

Only for documented:

```text
process interruption
hardware/software failure
corrupt checkpoint
artifact-write failure.
```

Use Experiment Registry rerun policy.

---

# 145. Score-based rerun forbidden

Do not rerun because:

```text
RMSE looks poor
curve looks unlucky.
```

Do not retrain F128.

---

# 146. OOM policy

Under the registered candidate widths, if F256 causes OOM:

```text
verify implementation/environment
```

first.

Do not silently reduce:

```text
batch size
lookback
d_model
```

because that would break controlled comparison.

If genuine environment capacity prevents F256:

```text
record technical limitation
S14 incomplete under strict core protocol
```

unless project-wide formal candidate-failure policy authorizes another treatment.

---

# 147. Numerical instability policy

If candidate produces NaN/Inf:

```text
audit data/model/optimizer/config
```

first.

No emergency LR/dropout/clip changes.

---

# 148. Discrepancy taxonomy

```text
S13_REFERENCE_MISSING
S13_WINNER_MISMATCH
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
FFN_DEFINITION_MISMATCH
MIXED_LAYER_FFN_WIDTHS
FFN_FIRST_LINEAR_SHAPE_ERROR
FFN_SECOND_LINEAR_SHAPE_ERROR
FFN_BIAS_SHAPE_ERROR
MHA_GEOMETRY_DRIFT
PE_DRIFT
LAYERNORM_DRIFT
REGRESSION_HEAD_DRIFT
INPUT_PROJECTION_DRIFT
PARAMETER_COUNT_NONMONOTONIC
PARAMETER_DELTA_MISMATCH
UNEXPECTED_STATE_DICT_KEY_DRIFT
UNEXPECTED_NON_FFN_SHAPE_DRIFT
INITIALIZATION_POLICY_DRIFT
SAMPLE_ORDER_POLICY_DRIFT
OPTIMIZER_COVERAGE_MISSING
OPTIMIZER_DUPLICATE_PARAMETER
FFN_SPECIFIC_LR_ADDED
FFN_SPECIFIC_WD_ADDED
FFN_SPECIFIC_DROPOUT_ADDED
GATED_FFN_ADDED
EXTRA_FFN_LAYER_ADDED
FFN_NORMALIZATION_ADDED
WARM_START_USED
WEIGHT_SLICING_USED
OPTIMIZER_STATE_REUSE
POPULATION_MISMATCH
TARGET_ID_MISMATCH
X_SCALER_MISMATCH
TARGET_SCALER_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
REFERENCE_RUN_MISMATCH
RUN_FAILURE
OOM_FAILURE
NUMERICAL_INSTABILITY
CHECKPOINT_CONFIG_MISMATCH
CHECKPOINT_VERIFICATION_FAILURE
INCOMPLETE_SWEEP
RANKING_ERROR
PAIRWISE_EFFECT_ERROR
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 149. Status model

## PASS

```text
F128 reference valid
F64/F256 valid fresh runs
same data/attention/depth/training
FFN-only shape deltas valid
parameter monotonicity/deltas valid
BEST checkpoints verified
winner selected
Phase37 reference generated
Test untouched.
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
metric ranking divergence
boundary winner
accuracy-efficiency tradeoff
reference init/order metadata unavailable
optional memory unavailable
inherited warning.
```

## FAIL

Examples:

```text
D/H/N changes
mixed per-layer FFN widths
non-FFN architecture drift
warm-start/slicing
parameter delta unexplained
unresolved candidate failure
Test access.
```

---

# 150. Output directory

```text
artifacts/
└── sweeps/
    └── S14_ffn/
        ├── s14_ffn_sweep_manifest.json
        ├── s14_ffn_sweep_contract.json
        ├── s14_ffn_preflight_audit.csv
        ├── s14_run_matrix.csv
        ├── s14_ffn_definition_audit.csv
        ├── s14_ffn_geometry_audit.csv
        ├── s14_ffn_expansion_ratio_audit.csv
        ├── s14_architecture_role_audit.csv
        ├── s14_state_dict_shape_delta_audit.csv
        ├── s14_parameter_count_audit.csv
        ├── s14_parameter_delta_audit.csv
        ├── s14_mha_invariance_audit.csv
        ├── s14_non_ffn_invariance_audit.csv
        ├── s14_optimizer_coverage_audit.csv
        ├── s14_config_delta_audit.csv
        ├── s14_ffn_unit_tests.csv
        ├── s14_common_data_audit.csv
        ├── s14_ffn_training_audit.csv
        ├── s14_initialization_policy_audit.csv
        ├── s14_shared_prefix_initialization_audit.csv
        ├── s14_sample_order_audit.csv
        ├── s14_dropout_ffn_audit.csv
        ├── s14_attention_api_audit.csv
        ├── s14_optimizer_budget_audit.csv
        ├── s14_ffn_run_provenance.csv
        ├── s14_ffn_metrics.csv
        ├── s14_ffn_pairwise_effects.csv
        ├── s14_ffn_trend_diagnostics.json
        ├── s14_ffn_efficiency_context.csv
        ├── s14_ffn_pareto_context.json
        ├── s14_optimization_diagnostics.csv
        ├── s14_convergence_diagnostics.csv
        ├── s14_runtime_capacity_diagnostics.csv
        ├── s14_generalization_diagnostics.csv
        ├── s14_hypothesis_outcomes.csv
        ├── s14_ffn_findings.csv
        ├── s14_ffn_winner.json
        ├── s14_reference_update.json
        ├── s14_ffn_sweep_tests.csv
        ├── s14_ffn_discrepancies.json
        ├── s14_ffn_sweep_summary.json
        ├── s14_ffn_sweep_report.md
        ├── figures/
        │   ├── S14_01_validation_rmse_by_epoch.png
        │   ├── S14_02_validation_mae_by_epoch.png
        │   ├── S14_03_train_loss_by_epoch.png
        │   ├── S14_04_gradient_clipping_fraction.png
        │   ├── S14_05_best_validation_metrics.png
        │   ├── S14_06_parameter_count_vs_rmse.png
        │   ├── S14_07_runtime_vs_rmse.png
        │   ├── S14_08_checkpoint_size_comparison.png
        │   ├── S14_09_ffn_width_trend.png
        │   └── S14_10_generalization_gap_optional.png
        ├── README_S14_FFN_SWEEP.md
        └── phase_36_signoff.json
```

Optional:

```text
s14_shared_prefix_initialization_audit.csv
s14_generalization_diagnostics.csv
S14_10_generalization_gap_optional.png
peak-memory fields
```

may be omitted if unavailable, but omission must be documented.

New F64/F256 runs remain in:

```text
artifacts/runs/<run_id>/
```

No checkpoint duplication.

---

# 151. Required outputs

```text
O36.1  Sweep manifest
O36.2  Sweep contract
O36.3  Preflight audit
O36.4  Run matrix
O36.5  FFN definition audit
O36.6  FFN geometry audit
O36.7  Expansion-ratio audit
O36.8  Architecture-role audit
O36.9  State-dict shape-delta audit
O36.10 Runtime parameter-count audit
O36.11 Pairwise parameter-delta audit
O36.12 MHA invariance audit
O36.13 Non-FFN invariance audit
O36.14 Optimizer coverage audit
O36.15 Config-delta audit
O36.16 FFN unit tests
O36.17 Common-data audit
O36.18 Training-config audit
O36.19 Initialization-policy audit
O36.20 Optional shared-prefix initialization audit
O36.21 Sample-order audit
O36.22 Dropout/FFN audit
O36.23 Attention-API audit
O36.24 Optimizer-budget audit
O36.25 Run provenance
O36.26 Reused F128 reference
O36.27 Verified F64 run
O36.28 Verified F256 run
O36.29 Primary metrics table
O36.30 Pairwise effects
O36.31 FFN trend diagnostics
O36.32 FFN efficiency context
O36.33 Pareto context
O36.34 Optimization diagnostics
O36.35 Convergence diagnostics
O36.36 Runtime/capacity diagnostics
O36.37 Optional generalization diagnostics
O36.38 Hypothesis outcomes
O36.39 Findings
O36.40 Winner artifact
O36.41 Phase37 reference update
O36.42 Figures
O36.43 Sweep tests
O36.44 Discrepancy log
O36.45 Sweep summary
O36.46 Human-readable report
O36.47 README
O36.48 Phase sign-off
```

---

# 152. Run matrix

Create:

```text
s14_run_matrix.csv
```

Fields:

```text
sweep_id
ffn_id
ffn_dim
ffn_expansion_ratio
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
population_fingerprint
feature_fingerprint
seed
model_config_id
training_config_id
status
```

---

# 153. Sweep manifest

Create:

```text
s14_ffn_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S14_FFN-v1
sweep_id = S14_FFN
source_s13_winner_run_id
selected_d_model
selected_num_heads
selected_head_dim
selected_num_layers
candidate_ffn_dims = [64,128,256]
derived_expansion_ratios
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
swept_field = ffn_dim
capacity_change_expected = true
parameter_count_equality_expected = false
parameter_count_monotonicity_expected = true
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = SMALLEST_FFN_ON_EXACT_RMSE_TIE
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

# 154. Sweep contract

Create:

```text
s14_ffn_sweep_contract.json
```

Must state:

```text
Only ffn_dim changes.

F64=64.
F128=128.
F256=256.

D*, H*, head_dim and N* remain fixed.

Every active layer uses the same candidate FFN width.

MHA/PE/LayerNorm/pooling/regression head remain unchanged.

Activation and dropout probability remain selected A*/DR*.

FFN parameter count is expected to increase monotonically.

No parameter matching.
No mixed layer widths.
No gated FFN.
No extra FFN layer.
No FFN-specific LR/WD/dropout.
No warm-start/slicing.

F128 reference reused.
F64/F256 fresh seed42 runs.

Validation RMSE Wh selects winner.
Exact tie → smaller FFN width.
Test forbidden.
```

---

# 155. Preflight audit

`s14_ffn_preflight_audit.csv` checks:

```text
phase35_pass
approved_for_phase36
s13_winner_valid
all prior selected fields locked
F64 registered
F128 registered
F256 registered
D* fixed
H* fixed
HD* fixed
N* fixed
A* fixed
DR* fixed
same FFN width across active layers per candidate
MHA fixed
PE fixed
LayerNorm fixed
pooling fixed
regression head fixed
population fixed
Training Engine fixed
Metric fixed
seed fixed
F128 reuse candidate valid
Test lock
status
```

---

# 156. FFN definition audit

`s14_ffn_definition_audit.csv`:

```text
ffn_id
ffn_dim
d_model
num_layers
activation
dropout
expansion_ratio
registered
status
```

---

# 157. FFN geometry audit

`s14_ffn_geometry_audit.csv`:

```text
ffn_id
layer_index
first_linear_in
first_linear_out
second_linear_in
second_linear_out
first_bias_shape
second_bias_shape
activation
dropout
candidate_consistent_across_layers
status
```

---

# 158. Expansion-ratio audit

`s14_ffn_expansion_ratio_audit.csv`:

```text
ffn_id
d_model
ffn_dim
expansion_ratio
ratio_is_derived
d_model_compensation_used=false
status
```

---

# 159. Architecture-role audit

`s14_architecture_role_audit.csv`:

```text
module_path_or_role
semantic_role
type_f64
type_f128
type_f256
present_all
same_role
ffn_shape_dependent
unexpected_difference
status
```

---

# 160. State-dict shape-delta audit

`s14_state_dict_shape_delta_audit.csv`:

```text
key
shape_f64
shape_f128
shape_f256
key_present_all
expected_ffn_width_dependency
non_ffn_shape_equal
unexpected_difference
status
```

---

# 161. Parameter-count audit

`s14_parameter_count_audit.csv`:

```text
ffn_id
runtime_trainable_parameters
runtime_total_parameters
reference_formula_parameters
formula_applicable
rank_by_parameter_count
monotonicity_valid
status
```

Expected rank:

```text
F64 < F128 < F256.
```

---

# 162. Parameter-delta audit

`s14_parameter_delta_audit.csv`:

```text
left_ffn_id
right_ffn_id
delta_m
runtime_param_delta
reference_param_delta
formula_applicable
delta_matches_if_applicable
ffn_only_shape_delta_verified
status
```

Rows:

```text
F64_TO_F128
F128_TO_F256
F64_TO_F256.
```

---

# 163. MHA invariance audit

`s14_mha_invariance_audit.csv`:

```text
ffn_id
d_model
num_heads
head_dim
mha_parameter_count
mha_parameter_shape_fingerprint
mha_config_fingerprint
same_across_candidates
status
```

---

# 164. Non-FFN invariance audit

`s14_non_ffn_invariance_audit.csv`:

```text
component
shape_f64
shape_f128
shape_f256
config_fingerprint_f64
config_fingerprint_f128
config_fingerprint_f256
same_across_candidates
status
```

Components:

```text
input_projection
positional_encoding
MHA
LayerNorm
pooling
regression_head.
```

---

# 165. Optimizer coverage audit

`s14_optimizer_coverage_audit.csv`:

```text
ffn_id
trainable_parameter_count
optimizer_parameter_reference_count
unique_optimizer_parameter_count
missing_parameters
duplicate_parameters
all_ffn_parameters_covered
status
```

---

# 166. Config-delta audit

`s14_config_delta_audit.csv` allowed differences:

```text
ffn_dim
derived expansion ratio
FFN tensor shapes
parameter count
checkpoint size
runtime/memory context
config fingerprint.
```

Everything else must match.

---

# 167. FFN unit tests

`s14_ffn_unit_tests.csv` should cover at least:

```text
F64 accepted
F128 accepted
F256 accepted
invalid FFN width rejected where applicable
all layers use same M
D* fixed
H* fixed
HD* fixed
N* fixed
input projection same
MHA geometry same
PE same
LayerNorm same
pooling same
head same
activation same
dropout p same
F64 FFN D→64→D
F128 FFN D→128→D
F256 FFN D→256→D
hidden activation shape 64/128/256
state_dict key sets equal
only FFN shapes differ
Params F64<F128<F256
parameter delta formula checked
optimizer covers all params
B1 output [1,1]
selected-B output [B,1]
partial batch valid
F64 forward finite
F64 backward finite
F256 forward finite
F256 backward finite
attention API unchanged
standard/inspection prediction allclose
wrong-width strict checkpoint load fails
official training attention OFF.
```

---

# 168. Common-data audit

`s14_common_data_audit.csv`:

```text
split
sample_count_f64
sample_count_f128
sample_count_f256
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
num_layers_equal
population_equal
status
```

---

# 169. Training audit

`s14_ffn_training_audit.csv`:

```text
ffn_id
ffn_dim
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
only_ffn_dim_differs
status
```

---

# 170. Initialization-policy audit

`s14_initialization_policy_audit.csv`:

```text
ffn_id
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

# 171. Optional shared-prefix initialization audit

`s14_shared_prefix_initialization_audit.csv`:

```text
semantic_module
ffn_id
construction_order_position
shape
initial_fingerprint
cross_candidate_match_expected
match
reason
status
```

Diagnostic only.

---

# 172. Sample-order audit

`s14_sample_order_audit.csv`:

```text
epoch_or_probe
f64_order_fingerprint
f128_order_fingerprint
f256_order_fingerprint
f128_reference_available
new_runs_match
all_three_match_if_verifiable
status
```

---

# 173. Dropout/FFN audit

`s14_dropout_ffn_audit.csv`:

```text
ffn_id
dropout_probability
num_layers
ffn_hidden_width
ffn_hidden_dropout_shape_context
dropout_scope_fingerprint
same_p
same_semantic_sites
mask_shape_equality_required=false
dropout_compensation_used=false
status
```

---

# 174. Attention API audit

`s14_attention_api_audit.csv`:

```text
ffn_id
num_layers
num_heads
attention_tensor_count
attention_shape
finite
nonnegative
eval_row_sum_error_max
standard_inspection_prediction_allclose
status
```

---

# 175. Optimizer budget audit

`s14_optimizer_budget_audit.csv`:

```text
ffn_id
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

# 176. Run provenance

`s14_ffn_run_provenance.csv`:

```text
ffn_id
ffn_dim
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

# 177. Primary metrics table

`s14_ffn_metrics.csv`:

```text
ffn_id
ffn_dim
expansion_ratio
run_id
source_type
d_model
num_heads
head_dim
num_layers
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

# 178. Pairwise effects

`s14_ffn_pairwise_effects.csv`:

```text
left_ffn_id
right_ffn_id
left_ffn_dim
right_ffn_dim
left_rmse_wh
right_rmse_wh
rmse_delta_wh
rmse_improvement_pct
mae_delta_wh
r2_delta
parameter_delta
parameter_increase_pct
runtime_delta_optional
status
```

Rows:

```text
F64_TO_F128
F128_TO_F256
F64_TO_F256.
```

---

# 179. Trend diagnostics

`s14_ffn_trend_diagnostics.json`:

```text
rmse_f64
rmse_f128
rmse_f256
pattern
best_ffn_id
best_ffn_dim
winner_is_boundary
interpretation
status
```

---

# 180. FFN efficiency context

`s14_ffn_efficiency_context.csv`:

```text
ffn_id
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

# 181. Pareto context

`s14_ffn_pareto_context.json`:

```text
accuracy_metric
cost_metrics
relationship
dominated_candidates
accuracy_efficiency_tradeoffs
official_winner
tie_rule_used
context_only=true
status
```

---

# 182. Optimization diagnostics

`s14_optimization_diagnostics.csv`:

```text
ffn_id
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

# 183. Convergence diagnostics

`s14_convergence_diagnostics.csv`:

```text
ffn_id
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

# 184. Runtime/capacity diagnostics

`s14_runtime_capacity_diagnostics.csv`:

```text
ffn_id
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

# 185. Optional generalization diagnostics

`s14_generalization_diagnostics.csv`:

```text
ffn_id
train_rmse_wh_at_best
validation_rmse_wh_at_best
rmse_gap_wh
train_mae_wh_at_best
validation_mae_wh_at_best
mae_gap_wh
ffn_capacity_interpretation
status
```

No Test.

---

# 186. Hypothesis outcomes

`s14_hypothesis_outcomes.csv`:

```text
hypothesis_id
comparison_or_pattern
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
INCONCLUSIVE.
```

---

# 187. Findings artifact

`s14_ffn_findings.csv` possible codes:

```text
F64_GAIN
F128_GAIN
F256_GAIN
FFN_EXACT_TIE
NARROW_FFN_BEST
MIDDLE_FFN_BEST
WIDE_FFN_BEST
BOUNDARY_WINNER
MONOTONIC_GAIN_WITH_WIDTH
MONOTONIC_DEGRADATION_WITH_WIDTH
U_SHAPED_FFN_RESPONSE
METRIC_RANKING_DIVERGENCE
PARAMETER_COUNT_MONOTONICITY_VERIFIED
PARAMETER_DELTA_VERIFIED
MHA_INVARIANCE_VERIFIED
NON_FFN_INVARIANCE_VERIFIED
CHECKPOINT_SIZE_GROWTH
RUNTIME_DIFFERENCE
MEMORY_DIFFERENCE
POSSIBLE_F64_UNDERFITTING
POSSIBLE_F256_OVERFITTING
CONVERGENCE_DIFFERENCE
CLIPPING_DIFFERENCE
INITIALIZATION_POLICY_VERIFIED
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
ATTENTION_API_VERIFIED
INHERITED_WARNING.
```

---

# 188. Winner artifact

`s14_ffn_winner.json` minimum:

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
selection_metric
selection_direction
tie_rule
winner_ffn_id
winner_ffn_dim
winner_expansion_ratio
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
winner_trainable_parameters
runner_up_ffn_id
runner_up_ffn_dim
runner_up_rmse_wh
runner_up_trainable_parameters
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

# 189. Phase37 reference update

`s14_reference_update.json`:

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
previous_ffn_dim=128
selected_ffn_id
selected_ffn_dim
selected_ffn_expansion_ratio
winner_run_id
winner_config_fingerprint
winner_rmse_wh
current_loss=MSE
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase37
```

---

# 190. Phase37 handoff logic

Phase 37 tests:

```text
L0 = MSE
L1 = Huber
```

while holding S14-selected FFN width fixed.

S14 winner already uses:

```text
MSE.
```

Therefore normally Phase37 should:

```text
reuse S14 winner as MSE reference
train only Huber candidate.
```

if exact match.

---

# 191. Loss handoff nuance

Phase37 must preserve:

```text
D*
H*
HD*
N*
selected F*
A*
DR*
all optimization/data settings
```

and change only:

```text
training loss.
```

Evaluation remains:

```text
MAE/RMSE/R² in raw Wh.
```

---

# 192. Figures

Recommended:

```text
S14_01_validation_rmse_by_epoch.png
S14_02_validation_mae_by_epoch.png
S14_03_train_loss_by_epoch.png
S14_04_gradient_clipping_fraction.png
S14_05_best_validation_metrics.png
S14_06_parameter_count_vs_rmse.png
S14_07_runtime_vs_rmse.png
S14_08_checkpoint_size_comparison.png
S14_09_ffn_width_trend.png
S14_10_generalization_gap_optional.png
```

Primary:

```text
S14_01_validation_rmse_by_epoch.png.
```

All figures source-derived from stored artifacts.

---

# 193. Safe interpretation if F64 wins

> Under the selected Transformer width/head/depth configuration, FFN width 64 achieved the lowest Validation RMSE among the registered widths while using the fewest FFN parameters.

Because F64 is a lower boundary:

```text
record boundary-winner limitation.
```

Do not automatically test F32.

---

# 194. Safe interpretation if F128 wins

> FFN width 128 achieved the lowest Validation RMSE under the frozen current architecture and optimization protocol.

This supports retaining the existing mid-width reference under the current setup.

---

# 195. Safe interpretation if F256 wins

> FFN width 256 achieved the lowest Validation RMSE among the registered S14 candidates, at the cost of additional trainable parameters and compute.

Because F256 is upper boundary:

```text
record boundary-winner limitation.
```

Do not automatically test F512.

---

# 196. Safe interpretation if exact tie

Use smallest tied FFN by predeclared rule.

Example:

```text
F64 and F128 exact RMSE tie
→ select F64.
```

---

# 197. Tiny non-zero margin

Strict lower RMSE wins.

Report:

```text
small single-seed Validation margin
```

but do not convert into tie.

---

# 198. Interpretation prohibitions

Do not claim:

```text
larger FFN always improves Transformers
F256 learns more meaningful attention
FFN ratio itself caused the result
more hidden neurons caused better generalization
F64 is underfit solely because it is narrow
F256 is overfit solely because it is wide
```

without evidence.

---

# 199. Sweep summary

Create:

```text
s14_ffn_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
selected_d_model
selected_num_heads
selected_head_dim
selected_num_layers
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
f64_run_id
f256_run_id
new_runs
reused_runs
candidate_ffn_dims
expansion_ratios
parameter_count_audit
parameter_delta_audit
mha_invariance_audit
non_ffn_invariance_audit
primary_metric
metrics_by_ffn
pairwise_effects
trend_pattern
ffn_efficiency_context
pareto_context
optimization_diagnostics
convergence_diagnostics
runtime_capacity_diagnostics
initialization_policy_status
sample_order_match
attention_api_status
winner
winner_margin
winner_is_boundary
inherited_warnings
phase37_reference
test_status
overall_status
```

---

# 200. Human-readable report

Create:

```text
s14_ffn_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S13
3. F64/F128/F256 definitions
4. Selected D*/H*/N* context
5. FFN geometry
6. Expansion-ratio context
7. Frozen-variable contract
8. Architecture shape-delta whitelist
9. MHA/non-FFN invariance
10. Parameter-count and delta audit
11. Initialization/sample-order fairness
12. F128 reference provenance
13. F64/F256 run provenance
14. Attention API sanity
15. Validation metrics
16. Pairwise FFN effects
17. Width-trend diagnostics
18. Learning-curve/convergence analysis
19. Gradient/clipping analysis
20. Runtime/capacity context
21. Optional generalization analysis
22. S14 winner
23. Interpretation cautions
24. Interaction limitations
25. Phase37 handoff
```

---

# 201. README

Create:

```text
README_S14_FFN_SWEEP.md
```

Must explain:

```text
Purpose
S13 winner handoff
F64/F128/F256
selected D*/H*/N*
absolute FFN width vs derived expansion ratio
same M across all active layers
FFN D→M→D geometry
parameter-count monotonicity
parameter delta formula
MHA/non-FFN invariance
initialization RNG-consumption nuance
dropout-mask shape nuance
no mixed per-layer width
no gated FFN
no FFN-specific optimizer settings
F128 reference reuse
winner/tie rules
boundary-winner caution
capacity/efficiency context
interaction limitations
Phase37 handoff
No Test.
```

---

# 202. Recommended notebook structure

```text
Cell 36.1  Phase title
Cell 36.2  Verify Phase35 sign-off
Cell 36.3  Declare SWEEP_S14_FFN-v1
Cell 36.4  Load S13 winner/reference
Cell 36.5  Freeze all prior selected fields
Cell 36.6  Define F64/F128/F256
Cell 36.7  Derive expansion ratios
Cell 36.8  Build run matrix
Cell 36.9  Audit common population
Cell 36.10 Audit FFN geometry
Cell 36.11 Audit same width across active layers
Cell 36.12 Audit architecture roles
Cell 36.13 Audit state-dict shape deltas
Cell 36.14 Audit parameter counts
Cell 36.15 Audit pairwise parameter deltas
Cell 36.16 Audit MHA invariance
Cell 36.17 Audit non-FFN invariance
Cell 36.18 Audit optimizer coverage
Cell 36.19 Audit dropout/FFN semantics
Cell 36.20 Audit training config
Cell 36.21 Audit initialization policy
Cell 36.22 Optional shared-prefix init audit
Cell 36.23 Audit sample-order policy
Cell 36.24 Verify F128 reference reuse
Cell 36.25 Build disposable F64 sanity model
Cell 36.26 F64 forward/backward/attention sanity
Cell 36.27 Build disposable F256 sanity model
Cell 36.28 F256 forward/backward/attention sanity
Cell 36.29 Register F64 run
Cell 36.30 Reseed + fresh F64 objects
Cell 36.31 Train F64
Cell 36.32 Verify F64 BEST
Cell 36.33 Register F256 run
Cell 36.34 Reseed + fresh F256 objects
Cell 36.35 Train F256
Cell 36.36 Verify F256 BEST
Cell 36.37 Build run provenance
Cell 36.38 Build optimizer-budget audit
Cell 36.39 Build metrics
Cell 36.40 Compute pairwise effects
Cell 36.41 Classify width trend
Cell 36.42 Build FFN efficiency context
Cell 36.43 Build Pareto context
Cell 36.44 Build optimization diagnostics
Cell 36.45 Build convergence diagnostics
Cell 36.46 Build runtime/capacity diagnostics
Cell 36.47 Optional Train-vs-Validation diagnostic
Cell 36.48 Evaluate hypotheses
Cell 36.49 Generate figures
Cell 36.50 Generate findings
Cell 36.51 Select winner
Cell 36.52 Write winner JSON
Cell 36.53 Write Phase37 reference update
Cell 36.54 Run S14 tests/discrepancies
Cell 36.55 Write summary/report
Cell 36.56 Register artifacts/checksums
Cell 36.57 Write README
Cell 36.58 Phase sign-off
```

---

# 203. Execution flow

```text
Verify Phase35
→ Load S13 winner
→ Freeze D*/H*/HD*/N* and all prior winners
→ Define F64/F128/F256
→ Derive FFN expansion ratios
→ Audit same population
→ Audit FFN-only architecture deltas
→ Audit parameter monotonicity/deltas
→ Audit MHA/non-FFN invariance
→ Verify F128 reference
→ Reuse F128
→ Disposable F64/F256 sanity
→ Register fresh F64/F256
→ Train F64
→ Verify F64 BEST
→ Train F256
→ Verify F256 BEST
→ Same-population Validation comparison
→ Compute pairwise effects
→ Analyze width trend
→ Analyze capacity/runtime context
→ Select minimum RMSE
→ Exact tie prefer smaller FFN
→ Update Phase37 reference
→ Write artifacts/sign-off
```

---

# 204. Fail-fast order

Before expensive F64/F256 training:

```text
1. Phase35 sign-off
2. S13 winner identity
3. freeze all previous selections
4. F64/F128/F256 registry
5. D*/H*/HD*/N* validity
6. same sample population
7. FFN geometry
8. same candidate width across all active layers
9. architecture role topology
10. state-dict FFN-only shape delta
11. parameter monotonicity
12. parameter delta formula/context
13. MHA invariance
14. non-FFN invariance
15. optimizer coverage
16. dropout/FFN contract
17. training config
18. F64/F256 forward/backward sanity
19. attention API sanity
20. F128 reuse eligibility
21. Test firewall
22. Registry readiness
```

---

# 205. Critical technical note: absolute width vs expansion ratio

S14 controls:

```text
absolute ffn_dim
```

not expansion ratio.

Expansion ratio is derived from selected D*.

Do not reinterpret the sweep as a constant-ratio study.

---

# 206. Critical technical note: parameter count must grow monotonically

If runtime shows:

```text
Params_F64 >= Params_F128
or
Params_F128 >= Params_F256
```

under the intended architecture:

```text
STOP.
```

This strongly suggests implementation/config drift.

---

# 207. Critical technical note: only FFN tensor shapes may change

MHA, PE, LayerNorm, pooling and regression head shapes must stay unchanged.

This is the strongest topology fairness check for S14.

---

# 208. Critical technical note: same seed does not imply same initial shared tensors everywhere

Changing hidden width consumes different numbers of RNG draws during construction.

Later modules may therefore initialize differently under the same seed.

Audit initialization policy, not impossible whole-model equality.

---

# 209. Critical technical note: no width-based dropout compensation

A wider FFN naturally has more hidden activation elements.

Keeping `p` constant is the registered one-factor design.

Do not reduce/increase dropout to equalize absolute dropped-element count.

---

# 210. Critical technical note: no F128 retraining

Sequential reference reuse prevents adaptive rerun selection.

Reuse exact S13 winner.

---

# 211. Winner verification checklist

Before writing `s14_ffn_winner.json`:

```text
[ ] F128 valid reused reference.
[ ] F64 valid completed run.
[ ] F256 valid completed run.
[ ] Same FV*/YS*/L*/P*/A*/B*/LR*/WD*/DR*/D*/H*/N*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Candidate M is global across all active layers.
[ ] MHA geometry identical.
[ ] PE identical.
[ ] LayerNorm identical.
[ ] Pooling/head identical.
[ ] Only FFN hidden-dependent tensor shapes differ.
[ ] State-dict key sets consistent.
[ ] Params F64 < F128 < F256.
[ ] Pairwise parameter deltas audited.
[ ] Same optimizer policy.
[ ] Optimizer covers all candidate parameters.
[ ] Same initialization policy.
[ ] Same sample-order policy audited.
[ ] No warm-start/slicing.
[ ] No gated FFN/extra FFN layer.
[ ] F64 BEST verified.
[ ] F256 BEST verified.
[ ] F128 BEST provenance valid.
[ ] Full-precision Validation RMSE used.
[ ] Pairwise effects computed.
[ ] Trend pattern generated.
[ ] Exact tie rule respected.
[ ] Boundary winner flagged if F64/F256.
[ ] Capacity/runtime metrics secondary only.
[ ] Phase37 reference generated.
[ ] No Test.
```

---

# 212. Phase37 handoff

Phase37 receives:

```text
s14_ffn_winner.json
s14_reference_update.json
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
selected_ffn_dim
selected_expansion_ratio
current_loss=MSE
population_fingerprint
metric_version.
```

and changes only:

```text
training loss.
```

---

# 213. Reference reuse in Phase37

S14 winner was trained with:

```text
MSE.
```

Therefore Phase37 normally:

```text
reuse S14 winner as MSE reference
train one Huber candidate.
```

---

# 214. Relationship with Phase37 Loss sweep

S14 must not compare loss functions.

Training loss remains MSE for all FFN candidates.

---

# 215. Relationship with Phase38 Epoch-cap sweep

If F256 appears to need more epochs, record it.

Do not extend budget now.

---

# 216. Relationship with Phase39 Gradient-clipping sweep

Clip remains:

```text
1.0.
```

Any clipping difference is diagnostic only.

---

# 217. Relationship with Phase42 Candidate synthesis

All F64/F128/F256 runs remain in Experiment Registry.

Do not delete losing candidates.

---

# 218. Relationship with Phase44 Rolling-origin robustness

S14 winner is selected on one Validation period.

Temporal robustness remains untested.

---

# 219. Relationship with Phase46 Multi-seed

S14 uses seed42 only.

Tiny differences are not seed-robust evidence.

---

# 220. Relationship with final attention analysis

FFN width does not change number/shape of attention heads/maps, but it changes learned representations and therefore can indirectly change attention patterns.

Final analysis must use the final selected FFN checkpoint; S14 itself does not interpret maps.

---

# 221. Reproducibility metadata

New F64/F256 runs record:

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
num_layers
ffn_dim
expansion ratio
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
metric checksum.
```

---

# 222. No fabricated outputs

Do not pre-fill:

```text
runtime parameter counts
actual parameter deltas
checkpoint sizes
runtime
memory
winner
RMSE
best epoch
gradient norms
generalization gap
```

before execution.

Allowed pre-runtime quantities:

```text
registered F values
derived expansion ratios
symbolic parameter-delta formulas.
```

---

# 223. Phase sign-off

Create:

```text
phase_36_signoff.json
```

Minimum:

```text
phase = 36
phase_name = S14 FFN sweep
sweep_version
sweep_id
source_s13_winner_run_id
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
f64_run_id
f128_reference_run_id
f256_run_id
new_run_ids
reused_run_ids
f64_parameter_count
f128_parameter_count
f256_parameter_count
parameter_monotonicity_status
parameter_delta_audit_status
mha_invariance_audit_status
non_ffn_invariance_audit_status
winner_ffn_id
winner_ffn_dim
winner_run_id
winner_rmse_wh
winner_parameter_count
winner_is_boundary
population_fingerprint
metric_version
attention_api_audit_status
initialization_policy_audit_status
sample_order_match_status
inherited_warnings
test_status
approved_for_phase37
overall_status
created_at
```

---

# 224. Acceptance checklist

```text
[ ] Phase35 valid.
[ ] approved_for_phase36=true.
[ ] SWEEP_S14_FFN-v1 declared.
[ ] S13 winner loaded.
[ ] FV*/YS*/L*/P*/A*/B*/LR*/WD*/DR* fixed.
[ ] D* fixed.
[ ] H* fixed.
[ ] head_dim fixed.
[ ] N* fixed.
[ ] F64 exactly 64.
[ ] F128 exactly 128.
[ ] F256 exactly 256.
[ ] No extra FFN candidate.
[ ] Expansion ratios derived correctly.
[ ] Expansion ratio marked derived, not controlled.
[ ] Same candidate M used in every active layer.
[ ] No mixed per-layer widths.
[ ] Input projection fixed.
[ ] MHA geometry fixed.
[ ] PE fixed.
[ ] LayerNorm fixed.
[ ] Pooling fixed.
[ ] Regression head fixed.
[ ] Activation fixed.
[ ] Dropout p fixed.
[ ] Dropout scope fixed.
[ ] No dropout compensation.
[ ] State-dict key sets equal/expected.
[ ] Only FFN shape-dependent keys differ.
[ ] FFN first Linear shapes correct.
[ ] FFN second Linear shapes correct.
[ ] FFN biases correct.
[ ] Params F64 < F128 < F256.
[ ] Parameter counts runtime-recorded.
[ ] Pairwise parameter deltas calculated.
[ ] Reference delta formula used only as audit.
[ ] MHA invariance PASS.
[ ] Non-FFN invariance PASS.
[ ] Optimizer covers all params exactly once.
[ ] No FFN-specific LR.
[ ] No FFN-specific WD.
[ ] No gated FFN.
[ ] No extra FFN layer.
[ ] No FFN normalization addition.
[ ] Same Train IDs.
[ ] Same Validation IDs.
[ ] Same WINDOWPOP-v1.
[ ] Same feature fingerprint.
[ ] Same X scaler.
[ ] Same target transform.
[ ] Same batch.
[ ] Same AdamW.
[ ] Same LR.
[ ] Same WD.
[ ] Same MSE.
[ ] Same E50/patience10.
[ ] Same min_delta.
[ ] Same clip1.
[ ] No scheduler/warmup.
[ ] accumulation1.
[ ] drop_last=false.
[ ] seed42.
[ ] Initialization policy same.
[ ] Whole-state equality marked N/A.
[ ] RNG-consumption nuance documented.
[ ] Sample-order policy same.
[ ] Dropout masks not required to match.
[ ] F128 reference exact-match.
[ ] F128 not retrained.
[ ] F64/F256 sanity uses disposable objects.
[ ] Official new runs recreate fresh objects.
[ ] F64 B1 output valid.
[ ] F256 B1 output valid.
[ ] Partial batch valid.
[ ] F64 forward finite.
[ ] F64 backward finite.
[ ] F256 forward finite.
[ ] F256 backward finite.
[ ] FFN hidden shape sanity correct.
[ ] Attention API unchanged.
[ ] Attention shape [B,H*,L,L].
[ ] Standard/inspection predictions allclose.
[ ] Official training attention OFF.
[ ] F64 registered before training.
[ ] F256 registered before training.
[ ] F64 fresh loaders/model/optimizer.
[ ] F256 fresh loaders/model/optimizer.
[ ] No F128→candidate warm-start.
[ ] No FFN weight slicing/padding.
[ ] No optimizer-state reuse.
[ ] F64/F256 trained through TRAINING_ENGINE-v1.
[ ] F64 BEST verified.
[ ] F256 BEST verified.
[ ] F128 BEST provenance valid.
[ ] Same steps per complete epoch.
[ ] Predictions finite.
[ ] Official metrics in Wh.
[ ] Full-precision RMSE used.
[ ] F64→F128 effect computed.
[ ] F128→F256 effect computed.
[ ] F64→F256 effect computed.
[ ] Width trend generated.
[ ] Winner=min RMSE.
[ ] Exact tie=smallest FFN.
[ ] Capacity/runtime does not override non-tie RMSE.
[ ] No arbitrary gain threshold.
[ ] Metric divergence recorded if present.
[ ] Boundary winner flagged if F64/F256.
[ ] No automatic range extension.
[ ] FFN efficiency context generated.
[ ] Pareto context generated.
[ ] Optimization diagnostics generated.
[ ] Gradient dimension caveat documented.
[ ] Convergence diagnostics generated.
[ ] Runtime/checkpoint diagnostics generated.
[ ] Optional memory missing handled honestly.
[ ] No underfit claim from width alone.
[ ] No overfit claim from width alone.
[ ] Hypotheses recorded.
[ ] Findings generated.
[ ] Inherited warnings propagated.
[ ] Winner artifact generated.
[ ] Phase37 reference update generated.
[ ] MSE reuse identified for Phase37.
[ ] No F32/F512.
[ ] No D/H/N compensation.
[ ] No extra epochs.
[ ] No score-based rerun.
[ ] No failed/SANITY run ranked.
[ ] No attention-based winner selection.
[ ] No Test access.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] FFN×D interaction documented.
[ ] FFN×N interaction documented.
[ ] FFN×H interaction documented.
[ ] FFN×activation/dropout interaction documented.
[ ] FFN×optimizer interaction documented.
[ ] FFN×budget interaction documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 225. Acceptance criteria

Phase 36 chỉ PASS khi:

```text
S13-selected depth/width/head configuration and all prior selected factors are fixed.

Exactly F64/F128/F256 are represented.

Only ffn_dim changes.

Every active Encoder layer uses the same candidate FFN width.

D*, H*, head_dim and N* remain fixed.

MHA/PE/LayerNorm/pooling/regression head remain invariant.

Activation and dropout probability remain fixed.

FFN geometry is exactly D*→M→D*.

State-dict key topology remains consistent.

Only FFN-width-dependent tensor shapes differ.

Parameter count grows monotonically F64<F128<F256.

Pairwise parameter deltas are verified against runtime architecture.

No mixed widths, gated FFN, extra FFN layer, FFN-specific optimizer settings or dropout compensation are introduced.

Same Train/Validation IDs and WINDOWPOP-v1 are used.

Same initialization policy and sample-order policy are audited.

F128 exact S13 reference is reused.

F64 and F256 are fresh seed42 runs.

No warm-start, slicing/padding or optimizer-state reuse occurs.

F64/F256 BEST checkpoints are verified with exact FFN configs.

Validation RMSE Wh selects winner.

Exact RMSE tie selects smallest FFN width.

Boundary winner is documented without hidden range extension.

Capacity/runtime metrics remain secondary.

Phase37 reference is generated.

Test remains untouched.
```

---

# 226. Failure conditions

Phase 36 FAIL if:

```text
wrong S13 winner used

D*/H*/N* or prior selected settings change

FFN values differ from registry

different layers use different FFN widths

MHA/PE/LayerNorm/pooling/head change

activation/dropout change

gated FFN or extra FFN layer is introduced

parameter count is non-monotonic without justified implementation explanation

parameter deltas cannot be explained by FFN width

non-FFN parameter shapes drift

sample population differs

F128 mismatch is ignored

F128 is retrained and a favorable rerun selected

F64/F256 warm-start from reference

weights are sliced/padded/interpolated

optimizer state is reused

unregistered width is tested/used

RMSE rounded before ranking

runtime/parameter count overrides lower non-tied RMSE

one candidate fails but winner is still declared

score-based rerun occurs

Test is used.
```

---

# 227. Common mistakes

## 227.1 F64 dùng N1 nhưng F256 dùng N2

Sai. N* phải cố định.

## 227.2 Đổi d_model để giữ FFN ratio

Sai multi-factor experiment.

## 227.3 Layer0 dùng F64, layer1 dùng F128

Sai per-layer mixed width.

## 227.4 Giảm dropout cho F256 vì hidden tensor lớn

Sai.

## 227.5 Thêm SwiGLU/GEGLU cho F256

Sai architecture-type change.

## 227.6 Load F128 BEST rồi slice hidden units để tạo F64

Warm-start/compression confound.

## 227.7 Pad F128 weights để tạo F256

Sai.

## 227.8 Dùng FFN-specific LR

Sai.

## 227.9 Chọn F64 vì nhỏ nhất dù RMSE cao hơn

Sai, trừ exact tie.

## 227.10 Chọn F256 vì nhiều parameters hơn nên “mạnh hơn”

Sai.

## 227.11 Thử F512 sau khi F256 thắng

Hidden adaptive search.

## 227.12 Thử F32 sau khi F64 thắng

Không thuộc S14.

## 227.13 Dùng attention heatmap để chọn FFN

Sai winner metric.

## 227.14 Dùng Test để chọn FFN width

Forbidden.

---

# 228. Recommended execution pseudocode

```text
load_phase35_signoff()
assert_approved_for_phase36()

s13 = load_s13_winner()

FV = s13.feature_variant_id
YS = s13.target_scaling_id
L  = s13.lookback_id
P  = s13.pooling_id
A  = s13.activation_id
B  = s13.batch_id
LR = s13.learning_rate
WD = s13.weight_decay
DR = s13.dropout_probability
D  = s13.d_model
H  = s13.num_heads
HD = s13.head_dim
N  = s13.winner_num_layers

candidates = {
    "F64": 64,
    "F128": 128,
    "F256": 256
}

assert D % H == 0
assert HD == D // H

derive_expansion_ratios()

audit_common_data_population()
audit_ffn_geometry()
audit_global_candidate_width_across_layers()
audit_architecture_roles()
audit_state_dict_shape_deltas()
audit_parameter_count_monotonicity()
audit_pairwise_parameter_deltas()
audit_mha_invariance()
audit_non_ffn_invariance()
audit_optimizer_coverage()
audit_dropout_ffn_contract()
audit_training_config()
audit_initialization_policy()
audit_sample_order_policy()

f128_reference = resolve_s13_winner_run()
assert_exact_s14_reference_match(
    f128_reference,
    ffn_dim=128
)

for ffn_id in ["F64", "F256"]:
    M = candidates[ffn_id]

    reseed(42)
    sanity_model = build_transformer(
        input_size=feature_count(FV),
        d_model=D,
        num_heads=H,
        num_layers=N,
        ffn_dim=M,
        dropout=DR,
        activation=A,
        pooling=P
    )
    run_ffn_shape_sanity(sanity_model, expected_m=M)
    run_forward_backward_sanity(sanity_model)
    run_attention_api_sanity(sanity_model)

    register_run(ffn_id)

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
        num_layers=N,
        ffn_dim=M,
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

    result = TRAINING_ENGINE_v1.fit(...)
    verify_best_checkpoint(result, ffn_dim=M)

    save_result(ffn_id, result)

results = {
    "F64": load_result("F64"),
    "F128": f128_reference,
    "F256": load_result("F256")
}

metrics = build_verified_s14_metrics(results)
pairwise = compute_ffn_pairwise_effects(metrics)
trend = classify_ffn_width_trend(metrics)
capacity = build_ffn_efficiency_context(results)
pareto = build_ffn_pareto_context(results)
optimization = build_optimization_diagnostics(results)
convergence = build_convergence_diagnostics(results)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer_smallest_ffn=True
)

write_hypothesis_outcomes()
write_findings()
write_s14_winner(winner)
write_phase37_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 229. Definition of Done

\[
\boxed{
One\ Fixed\ Attention/Depth\ Setup
+
Three\ FFN\ Widths
+
Two\ Fresh\ Runs
+
One\ Valid\ Reused\ F128
+
FFN\text{-}Only\ Shape\ Changes
+
Monotonic\ Capacity\ Audit
+
Same\ Data/Optimization
+
Verified\ BEST\ Metrics
+
S14\ Winner
+
Phase37\ Reference
+
No\ Test
}
\]

---

# 230. Final status contract

```text
PHASE 36 tests ffn_dim only.

Current:
all winners from S1–S13.

Candidates:
F64 = 64
F128 = 128
F256 = 256.

Frozen:
D*
H*
head_dim
N*
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

Derived:
FFN expansion ratio = M/D*.

If D32:
F64 ratio2
F128 ratio4
F256 ratio8.

If D64:
F64 ratio1
F128 ratio2
F256 ratio4.

F128:
reuse S13 winner if exact match.

F64/F256:
fresh seed42 runs.

Expected differences:
FFN hidden width
FFN hidden tensor shapes
FFN parameter count
whole-model parameter count
checkpoint size
runtime/memory context.

Expected invariants:
same input projection
same MHA
same PE
same LayerNorm
same pooling
same regression head
same D/H/N
same activation/dropout
same data/population
same optimizer/training protocol.

Hard:
same candidate M across every active layer.
Params F64 < F128 < F256.
Only FFN-width-dependent shapes may change.
No gated FFN.
No extra FFN layer.
No FFN-specific LR/WD/dropout.
No width compensation.
No warm-start/slicing/padding.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer smallest FFN width.

Boundary winner:
record, do not auto-expand search.

Capacity/runtime:
secondary context only.

No Test.

After SWEEP_S14_FFN-v1 PASS:
proceed to
PHASE 37 — S15 Loss sweep.
```

---

# 231. Final check

Correct:

```text
Load S13 winner
→ freeze D*/H*/N*/all previous winners
→ F64 vs F128 vs F256
→ derive ratios
→ audit FFN-only shape/parameter changes
→ reuse F128
→ fresh F64/F256
→ verify BEST checkpoints
→ same-population Validation comparison
→ pairwise/trend analysis
→ select min RMSE
→ exact tie smallest FFN
→ Phase37 reference
```

Incorrect:

```text
different FFN per layer
→ change D to preserve ratio
→ add SwiGLU
→ change dropout for wide FFN
→ slice/pad F128 weights
→ choose by model size instead of RMSE
→ inspect Test
```

Chỉ sau khi `SWEEP_S14_FFN-v1` được sign-off mới chuyển sang **PHASE 37 — S15 Loss sweep**.
