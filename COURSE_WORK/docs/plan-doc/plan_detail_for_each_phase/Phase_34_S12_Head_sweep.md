# PHASE 34 — S12 HEAD SWEEP

## Kế hoạch controlled sweep cho số lượng Attention Heads của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S11_DMODEL-v1`  
**Sweep ID:** `S12_HEADS`  
**Output version:** `SWEEP_S12_HEADS-v1`  
**Phase trước:** `Phase_33_S11_d_model_sweep.md`

---

# 1. Vai trò của Phase 34

Phase 34 là controlled experiment thứ mười hai trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target scaling, lookback, pooling, activation, batch size, learning rate, weight decay, dropout, `d_model`, số encoder layers, FFN width, loss, epoch budget, gradient clipping, sample population và seed đã được khóa từ Phase 33, số lượng attention heads nào phù hợp hơn cho Transformer Encoder: `2 heads` hay `4 heads`?

Phase 34 chỉ thay đúng một registered hyperparameter:

```text
num_heads
```

với hai condition:

```text
H2 = 2 attention heads
H4 = 4 attention heads
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Head\ Count\ Factor
+
Fixed\ d_{model}
+
Valid\ Head\ Geometry
+
Same\ Parameters\ Where\ Implementation\ Allows
+
Same\ Data
+
Same\ Training\ Protocol
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí Phase 34 trong master plan

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
```

Phase 34 không được quay lại thay:

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
num_layers
ffn_dim
loss
epoch cap
gradient clipping
RevIN
boundary protocol
```

---

# 3. Câu hỏi nghiên cứu của S12

Phase 34 phải trả lời:

```text
1. H2 hay H4 tạo Validation RMSE Wh thấp hơn?

2. Chia cùng representation width D* thành ít head lớn hơn hay nhiều head nhỏ hơn phù hợp hơn?

3. Head dimension thay đổi như thế nào dưới selected d_model?

4. Parameter count có thực sự giữ nguyên giữa H2/H4 theo implementation hiện tại không?

5. Learning curves, convergence và early stopping behavior khác nhau thế nào?

6. Gradient norm/clipping behavior khác nhau ra sao?

7. Runtime/throughput context có thay đổi không?

8. MAE/R² có cùng ranking với RMSE không?

9. Head count nào trở thành current reference cho Phase 35?
```

---

# 4. Current reference từ Phase 33

Phase 34 phải load:

```text
s11_d_model_winner.json
s11_reference_update.json
phase_33_signoff.json
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
```

Possible:

```text
D* ∈ {32,64}
```

Phase 34 không được hard-code `D64`.

---

# 5. Canonical head options

Registry:

```text
H2 = 2
H4 = 4
```

Exact numeric values:

```text
2
4
```

No other head count may enter S12 ranking.

---

# 6. Reference head count

Current sequential reference từ Phase 33 still uses:

```text
H4 = 4
```

vì head count chưa được swept trước Phase 34.

If exact S12 contract match:

```text
H4 → REUSE S11 winner
H2 → NEW fresh run
```

Normal S12 execution:

```text
1 new H2 scientific run
+
1 reused H4 reference
```

---

# 7. `d_model` is frozen from S11

Hard:

```text
d_model = D*
```

Possible:

```text
D*=32
or
D*=64.
```

S12 changes head partitioning of the selected representation width.

---

# 8. Head geometry formula

Required:

\[
head\_dim=\frac{d_{model}}{num\_heads}
\]

with hard validity:

\[
d_{model}\bmod num\_heads=0.
\]

---

# 9. Geometry if D32 wins S11

If:

```text
D*=32
```

then:

```text
H2:
num_heads = 2
head_dim = 16

H4:
num_heads = 4
head_dim = 8
```

Both valid.

---

# 10. Geometry if D64 wins S11

If:

```text
D*=64
```

then:

```text
H2:
num_heads = 2
head_dim = 32

H4:
num_heads = 4
head_dim = 16
```

Both valid.

---

# 11. Divisibility is a hard preflight gate

Before building a candidate:

```python
assert D_selected % num_heads == 0
```

for both H2/H4.

If selected D somehow fails divisibility:

```text
STOP
```

Do not automatically change `d_model`.

Current registered D32/D64 both support H2/H4.

---

# 12. Scientific meaning of the head sweep

With `D*` fixed, changing heads changes:

```text
number of parallel attention subspaces
per-head representation width
Q/K/V reshape geometry
attention-score tensor head axis
scaled-dot-product head dimension
```

while total Transformer embedding width stays constant.

Safe wording:

> Effect of partitioning the selected `d_model` into two versus four attention heads.

---

# 13. Head dimension is an inherent consequence

S12 does not hold `head_dim` constant.

Changing:

```text
H2 ↔ H4
```

at fixed `D*` necessarily changes:

```text
head_dim.
```

Do not change `D*` to preserve head dimension.

That would be:

```text
head count + d_model
```

as two factors.

---

# 14. Scaled-dot-product attention scale changes inherently

Within each head, scaled dot-product attention uses a scale based on head/key dimension.

Conceptually:

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

where under the current self-attention design:

```text
d_k = head_dim.
```

Therefore H2/H4 change the implicit scale through different `head_dim`.

This is an inherent consequence of head count under fixed D*.

Do not manually compensate the scale.

---

# 15. No custom attention temperature compensation

Do not add:

```text
manual temperature
custom scaling coefficient
head-specific rescaling
```

to make H2 and H4 attention logits numerically similar.

The registered S12 factor includes the standard geometric consequences of changing `num_heads`.

---

# 16. Parameter count expectation

Under standard PyTorch-style `MultiheadAttention` with fixed `embed_dim=D*`, changing only `num_heads` generally does **not** change the shapes of Q/K/V projection parameters.

Expected canonical behavior:

```text
Params(H2) = Params(H4)
```

because:

```text
embed_dim fixed
input projection fixed
Q/K/V projection matrices fixed
out projection fixed
FFN fixed
LayerNorm fixed
regression head fixed.
```

Runtime model audit is the source of truth.

---

# 17. Parameter-count equality is a hard expected check for current implementation

For the frozen `TRANSFORMER_IMPL-v1`, expected:

```text
trainable_parameters_H2
==
trainable_parameters_H4.
```

If counts differ:

```text
investigate implementation topology
```

before training.

Do not immediately accept it as intended.

---

# 18. Parameter schema should normally remain identical

Because selected D is fixed, expected:

```text
same parameter names
same parameter tensor shapes
same state_dict keys
same trainable parameter count.
```

Only non-parameter configuration:

```text
num_heads
head_dim
```

changes.

This makes S12 a particularly clean architecture sweep.

---

# 19. Runtime implementation is authoritative

If the custom encoder implementation instantiates head-specific learned modules separately, parameter schema could differ.

If that occurs:

```text
STOP
→ inspect implementation
→ reconcile with TRANSFORMER_IMPL-v1
```

Do not silently continue.

The project's intended encoder uses standard multi-head projection semantics, so equal parameterization is expected.

---

# 20. Initialization fairness can be very strong in S12

Because parameter schema is expected identical:

```text
same seed
same model builder
same parameter shapes
same initialization algorithm
```

should normally yield:

```text
initial_state_fingerprint_H2
==
initial_state_fingerprint_H4
```

if both are freshly initialized in the same environment and model construction path.

This provides a strong controlled comparison.

---

# 21. Historical H4 initialization caveat

H4 is reused from Phase 33.

If its initial-state fingerprint was stored:

```text
compare H2 initial state with H4 reference initial state.
```

If not:

```text
H4_INIT_MATCH = NOT_VERIFIABLE
```

Do not retrain H4 only to recover this metadata.

---

# 22. Do not copy H4 learned weights into H2

Although parameter shapes may be identical, official H2 must be a fresh scientific run.

Do not:

```text
load H4 BEST
change num_heads to H2
continue training.
```

That is warm-start and invalidates the sweep.

---

# 23. Important nuance: same tensor weights can be interpreted differently by head partitioning

At the same raw projection matrices, changing `num_heads` changes how projected channels are reshaped into heads.

Therefore even if initial state tensors are identical:

```text
forward behavior H2
!=
forward behavior H4
```

in general.

This is exactly the head-partition effect being tested.

---

# 24. No cross-head weight remapping

Do not manually rearrange channels to “match” H2 heads with pairs of H4 heads.

No:

```text
head merging
head averaging
head concatenation remap
weight permutation
```

inside S12.

Use standard model construction.

---

# 25. FFN width is frozen

Hard:

```text
ffn_dim = 128
```

for H2/H4.

FFN does not depend on head count under current architecture.

---

# 26. Layers are frozen

Hard:

```text
num_layers = 2
```

for H2/H4.

Phase 35 sweeps layers.

---

# 27. Dropout remains selected DR*

All dropout sites from S10 remain:

```text
p = DR*
```

with same semantic scope.

MHA attention-weight dropout probability stays the same even though number of attention heads changes.

---

# 28. Attention returned shape changes

Attention-aware path expected per layer:

```text
H2:
[B,2,L*,L*]

H4:
[B,4,L*,L*]
```

with:

```text
average_attn_weights=False.
```

This is an intended output-shape difference in the diagnostic/interpretability path.

---

# 29. Prediction output shape remains unchanged

Both:

```text
[B,L*,F]
→
[B,1]
```

Final regression output must not depend on head count.

---

# 30. Standard training path keeps attention extraction OFF

Official training/Validation:

```text
need_weights = False
```

No full attention tensors during training.

This avoids unnecessary memory overhead and keeps alignment with `TRAINING_ENGINE-v1`.

---

# 31. No scientific attention interpretation in S12

Do not select H2/H4 using:

```text
attention heatmap aesthetics
head entropy
head diversity
last-query peaks
```

Final attention analysis belongs Phases 52–57.

S12 selection uses Validation RMSE only.

---

# 32. Attention sanity is still required

Small diagnostic before official H2 training should verify:

```text
number of layer attention tensors = N2
H2 attention shape = [B,2,L*,L*]
prediction path equivalence between standard and inspection mode in eval
finite/nonnegative attention
eval-mode row sums ≈1
```

No Test.

---

# 33. Attention train-mode row sum caveat

With MHA dropout active in training:

```text
train-mode attention rows need not sum exactly to 1
```

after dropout.

Do not fail S12 because of this.

Evaluation sanity should use:

```text
model.eval().
```

---

# 34. Positional encoding is unchanged

Because D* stays fixed:

```text
PE width is identical
PE policy identical
PE buffer shape identical
```

between H2/H4.

Any PE shape difference is a discrepancy.

---

# 35. Input projection is unchanged

Expected same:

```text
Linear(F,D*)
```

for both.

Same parameter shape and, under matched initialization, same initial tensor values where verifiable.

---

# 36. FFN is unchanged

Expected both:

```text
Linear(D*,128)
→ A*
→ Dropout(DR*)
→ Linear(128,D*)
```

No head-specific FFN.

---

# 37. LayerNorm is unchanged

Expected:

```text
normalized_shape = D*
```

for both.

---

# 38. Regression head is unchanged

Expected:

```text
Linear(D*,1)
```

for both.

---

# 39. Architecture-role topology must remain identical

Required:

```text
same module paths
same module classes
same module count
same encoder layers
same PE
same FFN
same LayerNorms
same pooling
same regression head
```

Only MHA configuration:

```text
num_heads
derived head_dim
```

may differ.

---

# 40. MHA config delta whitelist

Allowed differences:

```text
num_heads
head_dim
head-related runtime geometry
attention tensor head-axis length
config fingerprint
```

Not allowed:

```text
embed_dim
kdim
vdim
bias
batch_first
dropout p
projection policy
```

---

# 41. Working hypotheses

## H-S12-01 — More heads may help model multiple temporal relationships

H4 may provide more parallel attention subspaces than H2.

Status:

```text
UNTESTED.
```

---

# 42. H-S12-02 — Fewer wider heads may be sufficient

H2 provides larger per-head dimensionality.

It may perform as well as or better than H4 under the selected D*.

Status:

```text
UNTESTED.
```

---

# 43. H-S12-03 — Head count may have small effect

Because total D and parameter count are fixed, the difference may be modest.

This remains an empirical question.

Do not pre-declare either candidate superior.

---

# 44. Preconditions

Required:

```text
Phase 33 = PASS
```

or:

```text
PASS_WITH_WARNING
```

with no unresolved critical issue.

Also:

```text
approved_for_phase34 = true.
```

---

# 45. Required upstream artifacts

```text
s11_d_model_winner.json
s11_reference_update.json
phase_33_signoff.json
```

---

# 46. Required upstream contracts

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
```

---

# 47. Carry-forward warnings

Propagate all non-critical warnings from earlier selections.

Examples:

```text
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
BOUNDARY_WINNER
RANDOM_CONTROL_GAIN
SAMPLE_ORDER_NOT_VERIFIABLE
INITIALIZATION_NOT_VERIFIABLE
```

Write them into:

```text
S12 manifest
S12 summary
S12 report
S12 winner
Phase 35 handoff.
```

---

# 48. Swept factor only

Canonical:

```text
num_heads
```

Allowed:

```text
2
4
```

Aliases:

```text
H2
H4
```

---

# 49. Frozen data contract

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
same exact ordered Validation population
```

---

# 50. Frozen model/training selections

Hard:

```text
P*
A*
B*
LR*
WD*
DR*
D*
N2
F128
```

---

# 51. Frozen optimizer/training engine

Hard:

```text
AdamW
selected LR*
selected WD*
same optimizer group policy
MSE
E50
patience10
min_delta0
clip1
scheduler=None
warmup=None
gradient accumulation=1
seed42
TRAINING_ENGINE-v1
METRICS-v1
```

---

# 52. Same sample-order policy

H2 new run should use the same:

```text
Train population
Train shuffle policy
batch grouping
loader seed
worker policy
```

as the reference protocol.

If H4 stored exact epoch order fingerprints:

```text
compare.
```

Otherwise mark:

```text
NOT_VERIFIABLE.
```

Do not retrain H4.

---

# 53. Same model initialization policy

Required:

```text
seed42
same model construction order
same initialization functions
same custom-init policy = none
same environment.
```

Because parameter schema is expected identical, exact initial state matching is a desirable assertion when reference provenance supports it.

---

# 54. H4 reference reuse gate

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
H1
WB0
WINDOWPOP-v1

H4
N2
F128
same PE
same dropout scope
POST_NORM
same mask policy

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

# 55. Fresh H2 run

Execution:

```text
register H2 run
↓
reseed 42
↓
fresh Train DataLoader
↓
fresh Validation DataLoader
↓
fresh Transformer(
    d_model=D*,
    num_heads=2,
    num_layers=2,
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

# 56. No H4→H2 warm-start

Forbidden:

```text
load H4 BEST into H2
continue training.
```

Even if parameter shapes are compatible.

H2 is a fresh scientific run.

---

# 57. No optimizer-state reuse

Do not transfer:

```text
AdamW first moments
second moments
step counters
```

from H4.

---

# 58. No head surgery

Forbidden:

```text
merge H4 heads into H2
copy head pairs
average heads
permute QKV channels
```

Official H2 model uses standard constructor only.

---

# 59. H2 model geometry assertions

If D*=32:

```text
D=32
H=2
head_dim=16
N=2
FFN=128.
```

If D*=64:

```text
D=64
H=2
head_dim=32
N=2
FFN=128.
```

---

# 60. H4 reference geometry assertions

If D*=32:

```text
D=32
H=4
head_dim=8.
```

If D*=64:

```text
D=64
H=4
head_dim=16.
```

---

# 61. Parameter-count audit

Before training H2:

```text
count H2 trainable params
read H4 runtime count
```

Hard expected:

```text
Params_H2 == Params_H4.
```

If false:

```text
do not train
investigate architecture implementation.
```

---

# 62. Parameter-schema audit

Expected:

```text
state_dict key sets equal
parameter names equal
parameter shapes equal
buffer names/shapes equal except config metadata
```

`num_heads` itself is configuration, not normally a parameter tensor.

---

# 63. Config provenance is mandatory

Since state-dict tensor schema may be identical between H2/H4, a bare state dict cannot identify the head count.

Checkpoint/run metadata must include:

```text
num_heads
head_dim
d_model
model config fingerprint.
```

---

# 64. Important checkpoint provenance nuance

It may be possible to load H4 tensors strictly into an H2-config model because tensor shapes are identical.

This does **not** mean the model configuration is equivalent.

Therefore:

```text
state_dict load success
!=
configuration identity.
```

Never use state dict alone to infer head count.

---

# 65. Wrong-config semantic test

A diagnostic can demonstrate:

```text
same state tensors
H2 config vs H4 config
→ potentially different predictions
```

on a disposable model/input.

This is optional but educational.

Do not modify official scientific runs.

---

# 66. H2 forward sanity

Before official training, with disposable/fresh sanity model:

```text
B1 forward → [1,1]
small B forward → [B,1]
finite outputs
MSE finite
backward finite
```

Then official run must recreate fresh model/loaders after sanity.

---

# 67. H2 attention sanity

Small eval-mode batch:

```text
prediction [B,1]
attention list length = 2
per-layer attention [B,2,L*,L*]
finite
nonnegative
source-axis row sums ≈1
```

No scientific attention interpretation.

---

# 68. Standard vs inspection prediction equivalence

For H2 in eval mode:

```text
pred_standard = model(x)
pred_attention = model.forward_with_attention(x).prediction
```

Expected:

```text
allclose
```

within frozen tolerance policy.

---

# 69. H4 reference attention geometry

Historical H4 contract:

```text
[B,4,L*,L*].
```

No need to regenerate full attention just for S12 if Phase 17/previous model verification already establishes it and selected D does not break the API.

A small current-config sanity is allowed.

---

# 70. Same output target

Both candidates forecast:

```text
Appliances_{t+1}
```

same horizon and same target IDs.

---

# 71. Primary metric

Hard:

```text
best_validation_rmse_wh
```

from verified BEST checkpoint.

---

# 72. Secondary scientific metrics

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

# 73. Engineering metrics

Record:

```text
parameter count
checkpoint bytes
mean epoch time
throughput
optional peak memory
time to best
```

Since parameter count should be equal, differences mainly reflect execution geometry/backend behavior.

Secondary only.

---

# 74. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{H2},
RMSE_{H4}
\right)
\]

Use full-precision Validation RMSE Wh.

---

# 75. Exact RMSE tie rule

If exact full-precision equality:

```text
prefer H2.
```

Rationale:

```text
fewer attention partitions
simpler head structure
larger per-head representation
predeclared architectural parsimony.
```

This is only an exact-tie rule.

---

# 76. No runtime-based tie-breaking before exact RMSE tie

If H4 has lower RMSE by any non-zero full-precision margin:

```text
H4 wins.
```

If H2 has lower RMSE:

```text
H2 wins.
```

Runtime/throughput does not override RMSE.

---

# 77. Pairwise effect

Define H2→H4:

\[
\Delta RMSE
=
RMSE_{H2}-RMSE_{H4}
\]

Positive:

```text
H4 improves.
```

Relative:

\[
Improvement\%
=
100
\times
\frac{RMSE_{H2}-RMSE_{H4}}{RMSE_{H2}}
\]

Also:

```text
MAE delta = MAE_H2 - MAE_H4
R² delta = R²_H4 - R²_H2
```

---

# 78. Head efficiency context

Because parameter count should be equal, do not compute:

```text
RMSE gain per extra parameter
```

unless runtime unexpectedly finds differing parameter counts and methodology is formally reconciled.

Useful engineering context instead:

```text
RMSE vs runtime
RMSE vs peak memory
```

secondary only.

---

# 79. Metric ranking divergence

If:

```text
H4 wins RMSE
H2 wins MAE
```

record:

```text
METRIC_RANKING_DIVERGENCE.
```

Winner remains RMSE-based.

---

# 80. BEST verification for H2

After official H2 run:

```text
fresh Transformer(
    d_model=D*,
    num_heads=2
)
↓
verify full config
↓
strict-load BEST
↓
model.eval()
↓
full ordered Validation
↓
inverse y transform if needed
↓
METRICS-v1
↓
verify stored BEST metrics.
```

---

# 81. H4 reference verification

No retrain.

Verify:

```text
correct selected D*
H4
correct selected prior factors
same population
same metric version
same Training Engine provenance
BEST already verified.
```

---

# 82. Checkpoint semantic identity

H2 BEST metadata must include:

```text
d_model
num_heads=2
head_dim
num_layers=2
ffn_dim=128
dropout
activation
pooling
config fingerprint.
```

H4 metadata must correspondingly show:

```text
num_heads=4.
```

---

# 83. Same data population hard gate

Required:

```text
Train IDs_H2 == Train IDs_H4
Validation IDs_H2 == Validation IDs_H4
population fingerprint equal.
```

No head-count-dependent sample filtering.

---

# 84. Same DataLoader settings

```text
B*
shuffle Train=true
shuffle Validation=false
drop_last=false
same worker policy
same generator seed policy.
```

---

# 85. Optimizer steps per complete epoch

Because B and population are fixed:

```text
steps_per_epoch_H2
=
steps_per_epoch_H4.
```

Total optimizer steps may differ due early stopping.

---

# 86. Gradient diagnostics

Record:

```text
mean preclip grad norm
max preclip grad norm
fraction batches clipped
epochs with clipping
nonfinite events.
```

Since parameter count/schema are expected equal, global norm comparison is more directly comparable than in d_model sweep.

Still interpret as trajectory diagnostics, not winner criteria.

---

# 87. Convergence diagnostics

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

# 88. Runtime diagnostics

Record:

```text
mean epoch seconds
median epoch seconds
total runtime
samples/sec
time to best optional
peak memory optional.
```

Backend kernels may behave differently for H2/H4.

Runtime context only.

---

# 89. Attention-memory context

If attention weights were materialized:

```text
H4 attention map tensor contains about twice as many head slices as H2
```

at fixed B/L/layers.

But standard S12 training uses:

```text
need_weights=False
```

so do not use full attention-map storage cost as the official training memory model.

This matters later during attention analysis.

---

# 90. Final attention-analysis implication

Selected head count determines final analysis layout:

```text
H2 → two heads per layer
H4 → four heads per layer
```

Later Phase 55 Head Comparison must use the final selected head count and must not assume four heads.

Carry selected H into final model provenance.

---

# 91. Single-seed limitation

Mandatory:

```text
S12 uses seed42 only.
```

No mean±std.

---

# 92. Validation-only limitation

Mandatory:

```text
S12 winner is selected using Validation only.
```

No Test evidence.

---

# 93. Head-count × d_model interaction limitation

S12 is conditional on selected:

```text
D*.
```

The preferred H under D32 may differ from preferred H under D64.

No full:

```text
D × H
```

grid is explored.

---

# 94. Head-count × layers interaction limitation

N2 is fixed.

A deeper/shallower Transformer may prefer a different head count.

Phase 35 sweeps layers only after S12 winner is selected.

---

# 95. Head-count × FFN interaction limitation

F128 fixed.

No H×FFN grid.

---

# 96. Head-count × dropout interaction limitation

DR* fixed.

No H×dropout grid.

---

# 97. Head-count × LR/WD interaction limitation

LR*/WD* fixed.

No optimizer retuning per head count.

---

# 98. Head-count × lookback interaction limitation

L* fixed.

Longer/shorter context may change the usefulness of multiple heads.

Not tested in S12.

---

# 99. Sequential-selection limitation

By Phase 34, Validation has influenced:

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
```

Full provenance and later robustness stages are therefore mandatory.

---

# 100. No Test access

Hard:

```text
Test loader not iterated
Test targets not read for metrics
Test predictions absent
Test metrics absent.
```

---

# 101. No H1

Not registered.

---

# 102. No H8

Not registered.

---

# 103. No H16

Not registered.

---

# 104. No D adjustment

No.

---

# 105. No FFN adjustment

No.

---

# 106. No layer adjustment

No.

---

# 107. No LR scaling by number of heads

No.

---

# 108. No dropout scaling by number of heads

No.

---

# 109. No per-head dropout

No.

---

# 110. No head pruning

No.

---

# 111. No head regularization term

No.

---

# 112. No head diversity loss

Do not add:

```text
orthogonality penalty
attention diversity loss
head decorrelation penalty
```

inside S12.

Those would change the objective.

---

# 113. No attention entropy regularization

No.

---

# 114. No extra epochs

No.

---

# 115. No head-specific optimizer groups

No.

---

# 116. Run failure policy

If H2 has a technical failure:

```text
S12 incomplete
```

until documented technical rerun succeeds.

Do not automatically declare H4 winner.

---

# 117. Technical rerun allowed

Only for documented:

```text
process interruption
hardware/software failure
corrupt checkpoint
artifact-write failure.
```

Use Experiment Registry rerun policy.

---

# 118. Score-based rerun forbidden

Do not rerun H2 because:

```text
RMSE looks poor.
```

Do not retrain H4 for another stochastic draw.

---

# 119. Numerical instability policy

If H2 yields NaN/Inf:

```text
audit data/config/MHA geometry/device
```

first.

If genuine:

```text
record numerical failure
S12 remains incomplete under strict core protocol
```

until formally resolved.

---

# 120. Discrepancy taxonomy

```text
S11_REFERENCE_MISSING
S11_WINNER_MISMATCH
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
HEAD_DEFINITION_MISMATCH
HEAD_DIVISIBILITY_FAILURE
HEAD_DIM_ERROR
CUSTOM_ATTENTION_SCALE_ADDED
PARAMETER_COUNT_MISMATCH
PARAMETER_SCHEMA_MISMATCH
STATE_DICT_KEY_MISMATCH
INPUT_PROJECTION_DRIFT
PE_DRIFT
FFN_DRIFT
LAYERNORM_DRIFT
REGRESSION_HEAD_DRIFT
DROPOUT_SCOPE_DRIFT
LAYER_COUNT_DRIFT
MHA_EMBED_DIM_DRIFT
MHA_KDIM_VDIM_DRIFT
MHA_BIAS_POLICY_DRIFT
HEAD_SURGERY_USED
HEAD_MERGING_USED
WARM_START_USED
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

# 121. Status model

PASS:

```text
H4 reference valid
H2 geometry valid
same D/data/training
parameter schema/count equality verified
H2 BEST verified
winner selected
Phase35 reference generated
Test untouched
```

PASS_WITH_WARNING examples:

```text
tiny RMSE margin
metric ranking divergence
reference init/sample-order not verifiable
runtime difference
inherited warning
```

FAIL examples:

```text
D changed
parameter count differs unexpectedly
FFN/layers changed
custom attention temperature added
head surgery/warm-start
population mismatch
unresolved H2 failure
Test access
```

---

# 122. Output directory

```text
artifacts/
└── sweeps/
    └── S12_heads/
        ├── s12_head_sweep_manifest.json
        ├── s12_head_sweep_contract.json
        ├── s12_head_preflight_audit.csv
        ├── s12_run_matrix.csv
        ├── s12_head_definition_audit.csv
        ├── s12_mha_geometry_audit.csv
        ├── s12_architecture_role_audit.csv
        ├── s12_parameter_schema_audit.csv
        ├── s12_parameter_count_audit.csv
        ├── s12_config_delta_audit.csv
        ├── s12_head_unit_tests.csv
        ├── s12_common_data_audit.csv
        ├── s12_head_training_audit.csv
        ├── s12_initialization_audit.csv
        ├── s12_sample_order_audit.csv
        ├── s12_dropout_scope_audit.csv
        ├── s12_attention_api_audit.csv
        ├── s12_optimizer_budget_audit.csv
        ├── s12_head_run_provenance.csv
        ├── s12_head_metrics.csv
        ├── s12_head_effect.csv
        ├── s12_head_efficiency_context.csv
        ├── s12_optimization_diagnostics.csv
        ├── s12_convergence_diagnostics.csv
        ├── s12_runtime_diagnostics.csv
        ├── s12_generalization_diagnostics.csv
        ├── s12_hypothesis_outcomes.csv
        ├── s12_head_findings.csv
        ├── s12_head_winner.json
        ├── s12_reference_update.json
        ├── s12_head_sweep_tests.csv
        ├── s12_head_discrepancies.json
        ├── s12_head_sweep_summary.json
        ├── s12_head_sweep_report.md
        ├── figures/
        │   ├── S12_01_validation_rmse_by_epoch.png
        │   ├── S12_02_validation_mae_by_epoch.png
        │   ├── S12_03_train_loss_by_epoch.png
        │   ├── S12_04_gradient_clipping_fraction.png
        │   ├── S12_05_best_validation_metrics.png
        │   ├── S12_06_runtime_vs_rmse.png
        │   ├── S12_07_convergence_summary.png
        │   └── S12_08_generalization_gap_optional.png
        ├── README_S12_HEAD_SWEEP.md
        └── phase_34_signoff.json
```

Optional:

```text
s12_generalization_diagnostics.csv
S12_08_generalization_gap_optional.png
peak-memory fields
```

may be omitted if unsupported, with explicit documentation.

The H2 run remains under:

```text
artifacts/runs/<run_id>/
```

No checkpoint duplication.

---

# 123. Required outputs

```text
O34.1  Sweep manifest
O34.2  Sweep contract
O34.3  Preflight audit
O34.4  Run matrix
O34.5  Head definition audit
O34.6  MHA geometry audit
O34.7  Architecture-role audit
O34.8  Parameter-schema audit
O34.9  Parameter-count audit
O34.10 Config-delta audit
O34.11 Head unit tests
O34.12 Common-data audit
O34.13 Training-config audit
O34.14 Initialization audit
O34.15 Sample-order audit
O34.16 Dropout-scope audit
O34.17 Attention-API audit
O34.18 Optimizer-budget audit
O34.19 Run provenance
O34.20 Reused H4 reference
O34.21 Verified new H2 run
O34.22 Primary metrics table
O34.23 Head effect table
O34.24 Efficiency context
O34.25 Optimization diagnostics
O34.26 Convergence diagnostics
O34.27 Runtime diagnostics
O34.28 Optional generalization diagnostics
O34.29 Hypothesis outcomes
O34.30 Findings
O34.31 Winner artifact
O34.32 Phase35 reference update
O34.33 Figures
O34.34 Sweep tests
O34.35 Discrepancy log
O34.36 Sweep summary
O34.37 Human-readable report
O34.38 README
O34.39 Phase sign-off
```

---

# 124. Run matrix

Create:

```text
s12_run_matrix.csv
```

Fields:

```text
sweep_id
head_id
num_heads
d_model
head_dim
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
num_layers
ffn_dim
population_fingerprint
feature_fingerprint
seed
model_config_id
training_config_id
status
```

---

# 125. Sweep manifest

Create:

```text
s12_head_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S12_HEADS-v1
sweep_id = S12_HEADS
source_s11_winner_run_id
selected_d_model
candidate_heads = [2,4]
derived_head_dims
fixed_num_layers = 2
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
swept_field = num_heads
parameter_count_equality_expected = true
parameter_schema_equality_expected = true
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = H2_ON_EXACT_RMSE_TIE
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

# 126. Sweep contract

Create:

```text
s12_head_sweep_contract.json
```

Must state:

```text
Only num_heads changes.

H2=2.
H4=4.

Selected d_model D* fixed.
N2 fixed.
F128 fixed.

head_dim is derived from D*/H.

Standard attention scaling follows head_dim.
No manual temperature compensation.

Parameter count/schema expected equal under Transformer-v1.

Same data/population.
Same optimizer/training.
Same dropout and scope.
Same initialization policy.
Same sample-order policy.

H4 reference reused.
H2 fresh seed42 run.

No head surgery/warm-start.
No attention-based winner selection.

Validation RMSE Wh selects winner.
Exact tie → H2.
Test forbidden.
```

---

# 127. Preflight audit

`s12_head_preflight_audit.csv` checks:

```text
phase33_pass
approved_for_phase34
s11_winner_valid
D* locked
H2 registered
H4 registered
D%2 valid
D%4 valid
derived head dims correct
N2 fixed
F128 fixed
all prior selected fields locked
parameter count equality expectation declared
dropout scope fixed
population fixed
Training Engine fixed
Metric fixed
seed fixed
Test lock
status
```

---

# 128. Head-definition audit

`s12_head_definition_audit.csv`:

```text
head_id
num_heads
d_model
head_dim
divisible
attention_scale_context
num_layers
ffn_dim
dropout
lookback
status
```

`attention_scale_context` may record:

```text
1/sqrt(head_dim)
```

as theoretical context only.

---

# 129. MHA geometry audit

`s12_mha_geometry_audit.csv`:

```text
head_id
embed_dim
num_heads
head_dim
divisible
kdim
vdim
batch_first
mha_dropout
bias_policy
attention_output_shape
status
```

---

# 130. Architecture-role audit

`s12_architecture_role_audit.csv`:

```text
module_path
semantic_role
module_type_h2
module_type_h4
present_h2
present_h4
shape_h2
shape_h4
expected_shape_equal
role_match
status
```

Most learned tensor shapes should match.

---

# 131. Parameter-schema audit

`s12_parameter_schema_audit.csv`:

```text
parameter_name
shape_h2
shape_h4
numel_h2
numel_h4
shape_equal
dtype_equal
trainable_equal
status
```

Expected all trainable parameter rows equal.

---

# 132. Parameter-count audit

`s12_parameter_count_audit.csv`:

```text
head_id
runtime_trainable_parameters
runtime_total_parameters
reference_count
parameter_count_equal
parameter_count_difference
status
```

Expected:

```text
difference = 0.
```

---

# 133. Config-delta audit

`s12_config_delta_audit.csv` allowed differences:

```text
num_heads
head_dim
attention head-axis output size
head-related config fingerprint
```

Everything else must match.

---

# 134. Head unit tests

`s12_head_unit_tests.csv` should cover at least:

```text
H2 accepted
H4 accepted
invalid head count rejected if not divisible
D*%2=0
D*%4=0
head_dim H2 correct
head_dim H4 correct
embed_dim same
N2 same
F128 same
input projection same shape
PE same shape
FFN same shape
LayerNorm same shape
regression head same shape
parameter count equal
parameter schema equal
state_dict keys equal
B1 output [1,1]
selected-B output [B,1]
partial batch valid
H2 forward finite
H2 backward finite
H2 attention [B,2,L,L]
H4 attention contract [B,4,L,L]
eval attention rows normalized
standard/attention prediction allclose
no attention extraction in official train path
```

---

# 135. Common-data audit

`s12_common_data_audit.csv`:

```text
split
sample_count_h2
sample_count_h4
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
population_equal
status
```

---

# 136. Training audit

`s12_head_training_audit.csv`:

```text
head_id
d_model
num_heads
head_dim
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
only_num_heads_differs
status
```

---

# 137. Initialization audit

`s12_initialization_audit.csv`:

```text
head_id
seed
initialization_policy_id
parameter_schema_equal
initial_state_fingerprint
reference_available
fingerprint_matches_reference
same_builder
status
```

For new disposable H2/H4 initializations under same environment, exact equality should be testable.

For historical H4:

```text
NOT_VERIFIABLE
```

is allowed if metadata absent.

---

# 138. Sample-order audit

`s12_sample_order_audit.csv`:

```text
epoch_or_probe
h2_order_fingerprint
h4_order_fingerprint
h4_reference_available
same_order
status
```

---

# 139. Dropout-scope audit

`s12_dropout_scope_audit.csv`:

```text
head_id
dropout_probability
dropout_scope_fingerprint
site_count
site_paths_match
semantic_scope_match
mha_dropout_equal
status
```

---

# 140. Attention API audit

`s12_attention_api_audit.csv`:

```text
head_id
num_layers
attention_tensor_count
expected_heads
observed_shape
finite
nonnegative
eval_row_sum_error_max
standard_inspection_prediction_allclose
status
```

Diagnostic only.

---

# 141. Optimizer budget audit

`s12_optimizer_budget_audit.csv`:

```text
head_id
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

# 142. Run provenance

`s12_head_run_provenance.csv`:

```text
head_id
num_heads
head_dim
run_id
source_type
source_phase
config_fingerprint
parameter_schema_fingerprint
parameter_count
feature_fingerprint
population_fingerprint
initial_state_fingerprint
sample_order_provenance
dropout_scope_fingerprint
best_checkpoint_sha256
history_sha256
metric_artifact
prediction_artifact
status
```

---

# 143. Metrics table

`s12_head_metrics.csv`:

```text
head_id
num_heads
head_dim
d_model
run_id
source_type
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

# 144. Head effect table

`s12_head_effect.csv`:

```text
h2_run_id
h4_run_id
h2_rmse_wh
h4_rmse_wh
rmse_delta_h2_to_h4_wh
rmse_improvement_pct
h2_mae_wh
h4_mae_wh
mae_delta_h2_to_h4_wh
h2_r2
h4_r2
r2_delta
parameter_counts_equal
rmse_winner
metric_ranking_divergence
status
```

---

# 145. Efficiency context

`s12_head_efficiency_context.csv`:

```text
head_id
validation_rmse_wh
trainable_parameters
mean_epoch_seconds
samples_per_second_optional
peak_memory_optional
checkpoint_bytes
selection_metric_used=false
status
```

No parameter-efficiency ratio expected because parameter counts should be equal.

---

# 146. Optimization diagnostics

`s12_optimization_diagnostics.csv`:

```text
head_id
best_epoch
last_epoch
stop_reason
best_rmse_wh
last_rmse_wh
mean_grad_norm_preclip
max_grad_norm_preclip
mean_fraction_batches_clipped
max_fraction_batches_clipped
nonfinite_events
best_to_last_gap
status
```

---

# 147. Convergence diagnostics

`s12_convergence_diagnostics.csv`:

```text
head_id
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

# 148. Runtime diagnostics

`s12_runtime_diagnostics.csv`:

```text
head_id
device
epochs_completed
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
samples_per_second_optional
peak_memory_optional
runtime_comparable
status
```

---

# 149. Optional generalization diagnostics

If supported:

`s12_generalization_diagnostics.csv`:

```text
head_id
train_rmse_wh_at_best
validation_rmse_wh_at_best
rmse_gap_wh
train_mae_wh_at_best
validation_mae_wh_at_best
mae_gap_wh
status
```

No Test.

---

# 150. Hypothesis outcomes

`s12_hypothesis_outcomes.csv`:

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
INCONCLUSIVE_TIE
```

within single-seed Validation context.

---

# 151. Findings artifact

`s12_head_findings.csv` possible codes:

```text
H2_GAIN
H4_GAIN
HEAD_EXACT_TIE
PARAMETER_COUNT_EQUAL_VERIFIED
PARAMETER_SCHEMA_EQUAL_VERIFIED
HEAD_DIM_CONTEXT
ATTENTION_SCALE_CONTEXT
METRIC_RANKING_DIVERGENCE
RUNTIME_DIFFERENCE
MEMORY_DIFFERENCE
CONVERGENCE_DIFFERENCE
CLIPPING_DIFFERENCE
INITIALIZATION_MATCH_VERIFIED
INITIALIZATION_MATCH_NOT_VERIFIABLE
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
ATTENTION_API_VERIFIED
INHERITED_WARNING
```

---

# 152. Winner artifact

`s12_head_winner.json` minimum:

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
selection_metric
selection_direction
tie_rule
winner_head_id
winner_num_heads
winner_head_dim
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_head_id
runner_up_num_heads
runner_up_head_dim
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
parameter_counts_equal
parameter_count
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 153. Phase35 reference update

`s12_reference_update.json`:

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
previous_num_heads=4
selected_head_id
selected_num_heads
selected_head_dim
winner_run_id
winner_config_fingerprint
winner_rmse_wh
current_num_layers=2
ffn_dim=128
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase35
```

---

# 154. Phase 35 handoff logic

Phase 35 tests:

```text
N1 = 1 encoder layer
N2 = 2 encoder layers
```

while holding selected H fixed.

S12 winner already uses:

```text
N2.
```

Therefore normally Phase35 should:

```text
reuse S12 winner as N2 reference
train only N1.
```

if exact match.

---

# 155. Layer/head handoff geometry

Phase35 must preserve:

```text
D*
selected H
selected head_dim
F128
```

and change only:

```text
num_layers.
```

---

# 156. Figures

Recommended:

```text
S12_01_validation_rmse_by_epoch.png
S12_02_validation_mae_by_epoch.png
S12_03_train_loss_by_epoch.png
S12_04_gradient_clipping_fraction.png
S12_05_best_validation_metrics.png
S12_06_runtime_vs_rmse.png
S12_07_convergence_summary.png
S12_08_generalization_gap_optional.png
```

Primary:

```text
S12_01_validation_rmse_by_epoch.png
```

All plots must derive from stored artifacts.

---

# 157. Safe interpretation if H2 wins

> Under the selected `d_model` and frozen N2/F128/optimization configuration, two wider attention heads achieved lower Validation RMSE than four narrower heads.

Do not claim:

```text
fewer heads are universally better.
```

---

# 158. Safe interpretation if H4 wins

> Four attention heads achieved lower Validation RMSE than two heads under the fixed selected representation width and training protocol.

Do not attribute the result solely to “more diverse attention” without final interpretability evidence.

---

# 159. Safe interpretation if exact tie

> H2 and H4 had exactly equal full-precision Validation RMSE; H2 was selected by the predefined architectural-parsimony tie rule.

---

# 160. Interpretation prohibitions

Do not claim:

```text
H4 learned four different physical phenomena
H2 is simpler because it has fewer parameters
H4 has more parameters
head_dim alone caused outcome
attention heatmaps prove forecasting accuracy
more heads always means better temporal modeling
```

Under standard current MHA, parameter count is expected equal.

---

# 161. README

Create:

```text
README_S12_HEAD_SWEEP.md
```

Must explain:

```text
Purpose
S11 winner handoff
H2/H4 definitions
selected D*
head_dim geometry
divisibility
scaled-attention consequence
parameter-count equality expectation
parameter-schema equality
same-initialization opportunity
state_dict/config provenance nuance
attention output shapes
no attention-based selection
H4 reference reuse
winner/tie rules
interaction limitations
Phase35 handoff
No Test
```

---

# 162. Sweep summary

Create:

```text
s12_head_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
selected_d_model
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
h2_run_id
new_runs
reused_runs
candidate_heads
head_geometry
parameter_count_audit
parameter_schema_audit
primary_metric
metrics_by_head
head_effect
efficiency_context
optimization_diagnostics
convergence_diagnostics
runtime_diagnostics
initialization_match
sample_order_match
attention_api_status
winner
winner_margin
inherited_warnings
phase35_reference
test_status
overall_status
```

---

# 163. Human-readable report

Create:

```text
s12_head_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S11
3. Selected d_model
4. H2/H4 definitions
5. Head geometry and divisibility
6. Scaled-dot-product consequence
7. Parameter-count/schema equality audit
8. Frozen-variable contract
9. Same data/sample fairness
10. Initialization/sample-order fairness
11. H4 reference provenance
12. H2 run provenance
13. Attention API sanity
14. Validation metrics
15. Head effect
16. Learning-curve/convergence diagnostics
17. Gradient/clipping diagnostics
18. Runtime/efficiency context
19. S12 winner
20. Interpretation cautions
21. Interaction limitations
22. Phase35 handoff
```

---

# 164. Recommended notebook structure

```text
Cell 34.1  Phase title
Cell 34.2  Verify Phase33 sign-off
Cell 34.3  Declare SWEEP_S12_HEADS-v1
Cell 34.4  Load S11 winner/reference
Cell 34.5  Freeze all prior selected fields
Cell 34.6  Define H2/H4
Cell 34.7  Derive head dimensions
Cell 34.8  Verify divisibility
Cell 34.9  Build run matrix
Cell 34.10 Audit common population
Cell 34.11 Audit MHA geometry
Cell 34.12 Audit architecture roles
Cell 34.13 Audit parameter schema/count equality
Cell 34.14 Audit config delta whitelist
Cell 34.15 Run H2 unit tests
Cell 34.16 Audit training config
Cell 34.17 Audit initialization
Cell 34.18 Audit sample order
Cell 34.19 Audit dropout scope
Cell 34.20 Verify H4 reference reuse
Cell 34.21 Register H2 run
Cell 34.22 Reseed/fresh loaders/model/optimizer
Cell 34.23 H2 forward/backward sanity
Cell 34.24 H2 attention API sanity
Cell 34.25 Recreate fresh official objects
Cell 34.26 Train H2
Cell 34.27 Verify H2 BEST
Cell 34.28 Build provenance
Cell 34.29 Optimizer-budget audit
Cell 34.30 Build metrics
Cell 34.31 Compute H2→H4 effect
Cell 34.32 Efficiency context
Cell 34.33 Optimization diagnostics
Cell 34.34 Convergence diagnostics
Cell 34.35 Runtime diagnostics
Cell 34.36 Optional generalization diagnostic
Cell 34.37 Hypothesis outcomes
Cell 34.38 Generate figures
Cell 34.39 Findings
Cell 34.40 Select winner
Cell 34.41 Winner JSON
Cell 34.42 Phase35 reference update
Cell 34.43 Tests/discrepancies
Cell 34.44 Summary/report
Cell 34.45 Register checksums
Cell 34.46 README
Cell 34.47 Sign-off
```

---

# 165. Execution flow

```text
Verify Phase33
→ Load S11 winner
→ Freeze selected D* and all prior winners
→ Define H2/H4
→ Derive head_dim
→ Verify divisibility
→ Audit same population
→ Audit MHA geometry
→ Verify parameter schema/count equality
→ Verify H4 reference
→ Reuse H4
→ Register H2
→ Seed42 + fresh H2 objects
→ H2 sanity
→ Fresh official H2 objects
→ Train via TRAINING_ENGINE-v1
→ Verify H2 BEST
→ Same-population Validation comparison
→ Compute head effect
→ Analyze runtime/convergence context
→ Select min RMSE
→ Exact tie prefer H2
→ Update Phase35 reference
→ Write artifacts/sign-off
```

---

# 166. Fail-fast order

Before expensive H2 training:

```text
1. Phase33 sign-off
2. S11 winner identity
3. Freeze selected D* and previous winners
4. H2/H4 definitions
5. D/head divisibility
6. head_dim derivation
7. same population
8. MHA geometry
9. parameter schema equality
10. parameter count equality
11. PE/FFN/LayerNorm/head invariance
12. dropout scope invariance
13. training config
14. H2 forward/backward sanity
15. attention API sanity
16. H4 reuse eligibility
17. Test firewall
18. Registry readiness
```

---

# 167. Key technical note: same parameter count does not mean same model function

With fixed D*, H2 and H4 can have identical:

```text
parameter tensors
parameter shapes
parameter count
```

yet compute different functions because they reshape the projected representation into different head partitions.

This is a central methodological point of S12.

---

# 168. Key technical note: state_dict alone cannot prove head count

Because tensor schema may be identical, always preserve:

```text
model config
num_heads
head_dim
config fingerprint
```

with the checkpoint.

Never reconstruct head count from weights alone.

---

# 169. Key technical note: no manual attention scale correction

The difference in `1/sqrt(head_dim)` is part of standard attention geometry.

Compensating it would test a modified attention implementation, not just head count.

---

# 170. Key technical note: same initial weights can make S12 especially clean

If provenance permits:

```text
H2 and H4 initial tensors can be exactly matched
```

while head partitioning differs.

This is desirable because it isolates the architecture interpretation of the same initialized projections.

But the H4 historical reference must not be retrained merely to force this condition.

---

# 171. Key technical note: attention maps cannot decide winner

More visually distinct heads do not imply lower forecast error.

S12 chooses winner by Validation RMSE only.

Attention quality/interpretability is studied later.

---

# 172. Phase35 handoff requirements

Pass:

```text
selected num_heads
selected head_dim
selected D*
selected run_id
config fingerprint
N2 reference
F128
all prior selected factors
population fingerprint
metric version
warnings
approved_for_phase35
```

Phase35 changes only:

```text
N1 vs N2.
```

---

# 173. Phase sign-off schema

Create:

```text
phase_34_signoff.json
```

Minimum:

```text
phase = 34
phase_name = S12 Head sweep
sweep_version
sweep_id
source_s11_winner_run_id
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
h2_run_id
h4_reference_run_id
new_run_ids
reused_run_ids
h2_head_dim
h4_head_dim
h2_parameter_count
h4_parameter_count
parameter_count_equal
parameter_schema_equal
winner_head_id
winner_num_heads
winner_head_dim
winner_run_id
winner_rmse_wh
population_fingerprint
metric_version
mha_geometry_audit_status
attention_api_audit_status
initialization_audit_status
sample_order_match_status
inherited_warnings
test_status
approved_for_phase35
overall_status
created_at
```

---

# 174. Acceptance checklist

```text
[ ] Phase33 valid.
[ ] approved_for_phase34 = true.
[ ] SWEEP_S12_HEADS-v1 declared.
[ ] S11 winner loaded.
[ ] D* loaded dynamically.
[ ] FV*/YS*/L*/P*/A*/B*/LR*/WD*/DR* fixed.
[ ] H2 exactly 2.
[ ] H4 exactly 4.
[ ] No extra head candidate.
[ ] D* divisible by 2.
[ ] D* divisible by 4.
[ ] H2 head_dim correct.
[ ] H4 head_dim correct.
[ ] Head-dimension consequence documented.
[ ] Standard attention scaling consequence documented.
[ ] No manual attention temperature.
[ ] N2 fixed.
[ ] F128 fixed.
[ ] Dropout fixed.
[ ] Dropout scope fixed.
[ ] Input projection fixed.
[ ] PE fixed.
[ ] FFN fixed.
[ ] LayerNorm fixed.
[ ] Regression head fixed.
[ ] Same architecture roles.
[ ] Same parameter count expected.
[ ] Same parameter count verified.
[ ] Same parameter names expected.
[ ] Same parameter shapes expected.
[ ] Same state_dict keys expected.
[ ] Parameter-schema audit PASS.
[ ] Same Train IDs.
[ ] Same Validation IDs.
[ ] Same WINDOWPOP-v1.
[ ] Same feature fingerprint.
[ ] Same X scaler.
[ ] Same y transform.
[ ] Same batch.
[ ] Same LR.
[ ] Same WD.
[ ] Same activation/pooling.
[ ] Same H1/WB0.
[ ] Same AdamW.
[ ] Same optimizer-group policy.
[ ] Same MSE.
[ ] Same E50/patience10.
[ ] Same min_delta.
[ ] Same clip1.
[ ] No scheduler/warmup.
[ ] accumulation=1.
[ ] drop_last=false.
[ ] seed42.
[ ] Same initialization policy.
[ ] Exact initial-state match checked if H4 provenance available.
[ ] No H4 retrain for metadata.
[ ] Same sample-order policy.
[ ] H4 reference exact-match.
[ ] H4 not retrained.
[ ] H2 registered before training.
[ ] H2 fresh loaders/model/optimizer.
[ ] No H4→H2 warm-start.
[ ] No optimizer-state reuse.
[ ] No head merging.
[ ] No QKV channel remap.
[ ] H2 B1 forward valid.
[ ] H2 selected-B forward valid.
[ ] H2 backward finite.
[ ] H2 attention API valid.
[ ] H2 attention shape [B,2,L,L].
[ ] H4 attention contract [B,4,L,L].
[ ] Eval attention row normalization valid.
[ ] Standard/inspection predictions allclose.
[ ] Official training uses attention OFF.
[ ] H2 trained via TRAINING_ENGINE-v1.
[ ] H2 BEST verified with H2 config.
[ ] H4 BEST provenance valid.
[ ] Checkpoint metadata contains num_heads/head_dim.
[ ] State_dict alone not used as config provenance.
[ ] Same complete-epoch optimizer steps.
[ ] Predictions finite.
[ ] Metrics in Wh.
[ ] Full-precision RMSE used.
[ ] H2→H4 RMSE effect computed.
[ ] MAE/R² effects computed.
[ ] Winner=min RMSE.
[ ] Exact tie=H2.
[ ] Runtime does not override RMSE.
[ ] No arbitrary minimum improvement threshold.
[ ] Metric divergence recorded if present.
[ ] Optimization diagnostics generated.
[ ] Convergence diagnostics generated.
[ ] Runtime diagnostics generated.
[ ] Optional generalization uses Train/Validation only.
[ ] No attention aesthetics used for selection.
[ ] Hypotheses recorded.
[ ] Findings generated.
[ ] Winner artifact generated.
[ ] Phase35 reference update generated.
[ ] N2 reuse identified.
[ ] Inherited warnings propagated.
[ ] No H1/H8/H16.
[ ] No D compensation.
[ ] No FFN compensation.
[ ] No layer compensation.
[ ] No head diversity loss.
[ ] No head pruning.
[ ] No per-head dropout.
[ ] No extra epochs.
[ ] No score-based rerun.
[ ] No failed/SANITY run ranked.
[ ] No Test access.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] D×H interaction documented.
[ ] H×layers interaction documented.
[ ] H×FFN interaction documented.
[ ] H×regularization interaction documented.
[ ] H×lookback interaction documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 175. Acceptance criteria

Phase 34 chỉ PASS khi:

```text
S11-selected d_model and all prior selected factors are fixed.

Exactly H2 and H4 are represented.

Selected d_model is divisible by both head counts.

Head dimensions are correctly derived.

Only num_heads changes.

No manual attention-scale compensation is introduced.

N2 and F128 stay fixed.

Same Train/Validation IDs and WINDOWPOP-v1 are used.

Same input projection/PE/FFN/LayerNorm/regression head are used.

Parameter count and parameter schema equality are verified under Transformer-v1.

Same initialization/sample-order policy is audited.

H4 exact S11 reference is reused.

H2 is one fresh seed42 run.

No warm-start, head merging, channel remapping or optimizer-state reuse occurs.

H2 standard forward and attention-aware API pass sanity checks.

H2 BEST is verified with correct head config metadata.

Validation RMSE Wh selects winner.

Exact RMSE tie selects H2.

Runtime/attention diagnostics remain secondary.

Phase35 reference is generated.

Test remains untouched.
```

---

# 176. Failure conditions

Phase 34 FAIL if:

```text
wrong S11 winner used

D* changes

H2/H4 values differ from registry

D not divisible by H

FFN/layers change

custom attention scaling is added

parameter count differs unexpectedly

parameter schema changes unexpectedly

PE/input projection/head changes

dropout scope changes

sample population differs

H4 reference mismatch is ignored

H4 is retrained and best rerun selected

H2 warm-starts from H4

head merging/remapping is used

optimizer state is reused

attention quality overrides RMSE

unregistered head count is tested/used

RMSE is rounded before ranking

runtime overrides lower RMSE

one candidate fails but winner is still declared

score-based rerun occurs

Test is used.
```

---

# 177. Common mistakes

## 177.1 H2 dùng D32, H4 dùng D64

Sai. S12 phải cố định D*.

## 177.2 Giữ head_dim cố định bằng cách đổi D

Sai two-factor experiment.

## 177.3 Thêm temperature để H2/H4 có cùng scale

Sai implementation contract.

## 177.4 Cho rằng H4 có nhiều parameters hơn

Không đúng với standard fixed-embed-dim MHA; phải audit runtime.

## 177.5 Cho rằng state_dict giống nhau nghĩa là model giống nhau

Sai. `num_heads` thay đổi reshape semantics.

## 177.6 Load H4 BEST vào H2 rồi tiếp tục training

Warm-start confound.

## 177.7 Merge từng cặp H4 heads thành H2

Head surgery, không phải S12.

## 177.8 Chọn H4 vì heatmap đẹp hơn

Sai winner metric.

## 177.9 Chọn H2 vì chạy nhanh hơn dù RMSE cao hơn

Sai, trừ exact RMSE tie rule.

## 177.10 Thử H8 sau khi H4 thắng

Hidden adaptive search.

## 177.11 Thử H1 sau khi H2 thắng

Không thuộc S12.

## 177.12 Thêm head diversity loss

Thay objective.

## 177.13 Dùng Test để chọn số heads

Forbidden.

---

# 178. Recommended execution pseudocode

```text
load_phase33_signoff()
assert_approved_for_phase34()

s11 = load_s11_winner()

FV = s11.feature_variant_id
YS = s11.target_scaling_id
L  = s11.lookback_id
P  = s11.pooling_id
A  = s11.activation_id
B  = s11.batch_id
LR = s11.learning_rate
WD = s11.weight_decay
DR = s11.dropout_probability
D  = s11.winner_d_model

candidates = {
    "H2": 2,
    "H4": 4
}

for H in [2,4]:
    assert D % H == 0

derive_head_dims()

audit_common_data_population()
audit_mha_geometry()
audit_architecture_roles()
audit_parameter_schema_equality()
audit_parameter_count_equality()
audit_training_config()
audit_initialization_policy()
audit_sample_order_policy()
audit_dropout_scope()

h4_reference = resolve_s11_winner_run()
assert_exact_s12_reference_match(
    h4_reference,
    d_model=D,
    num_heads=4
)

register_h2_run()

reseed(42)

sanity_model = build_transformer(
    d_model=D,
    num_heads=2,
    num_layers=2,
    ffn_dim=128,
    dropout=DR,
    activation=A,
    pooling=P
)

run_h2_forward_backward_sanity(sanity_model)
run_h2_attention_api_sanity(sanity_model)

# Official run starts fresh after sanity.
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
    num_heads=2,
    num_layers=2,
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

h2_result = TRAINING_ENGINE_v1.fit(...)

verify_best_checkpoint(
    h2_result,
    d_model=D,
    num_heads=2
)

results = {
    "H2": h2_result,
    "H4": h4_reference
}

metrics = build_verified_s12_metrics(results)
effect = compute_h2_to_h4_effect(metrics)
optimization = build_optimization_diagnostics(results)
convergence = build_convergence_diagnostics(results)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer="H2"
)

write_hypothesis_outcomes()
write_findings()
write_s12_winner(winner)
write_phase35_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 179. Definition of Done

\[
\boxed{
One\ Fixed\ d_{model}
+
Two\ Head\ Counts
+
Valid\ Head\ Geometry
+
Equal\ Parameterization
+
One\ Fresh\ H2
+
One\ Valid\ Reused\ H4
+
Same\ Data/Optimization
+
Verified\ BEST\ Metrics
+
S12\ Winner
+
Phase35\ Reference
+
No\ Test
}
\]

---

# 180. Final status contract

```text
PHASE 34 tests num_heads only.

Current:
all winners from S1–S10
selected d_model from S11.

Candidates:
H2 = 2
H4 = 4.

D* fixed.

Derived:
head_dim = D*/H.

If D32:
H2 → 16
H4 → 8.

If D64:
H2 → 32
H4 → 16.

Frozen:
N2
F128
PE
pooling
activation
batch
LR
WD
dropout
dropout scope
MSE
E50
patience10
clip1
Training Engine
Metric version.

Expected:
same parameter count
same parameter names/shapes
same state_dict key set.

Allowed differences:
num_heads
head_dim
attention head-axis geometry
standard attention scaling consequence.

No:
D compensation
FFN compensation
layer compensation
custom temperature
head surgery
head merging
warm-start
optimizer-state reuse
attention-based selection
extra head counts.

H4:
reuse S11 winner if exact match.

H2:
one fresh seed42 run.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer H2.

No Test.

After SWEEP_S12_HEADS-v1 PASS:
proceed to
PHASE 35 — S13 Layer sweep.
```

---

# 181. Final check

Correct:

```text
Load S11 winner
→ freeze selected D*
→ H2 vs H4
→ derive head_dim
→ verify divisibility
→ verify equal parameterization
→ reuse H4
→ fresh H2
→ verify H2 BEST
→ same-population Validation comparison
→ select min RMSE
→ exact tie H2
→ Phase35 reference
```

Incorrect:

```text
H2 with one D
→ H4 with another D
→ add attention temperature
→ merge H4 heads
→ warm-start H2
→ choose by attention plot
→ inspect Test
```

Chỉ sau khi `SWEEP_S12_HEADS-v1` được sign-off mới chuyển sang **PHASE 35 — S13 Layer sweep**.
