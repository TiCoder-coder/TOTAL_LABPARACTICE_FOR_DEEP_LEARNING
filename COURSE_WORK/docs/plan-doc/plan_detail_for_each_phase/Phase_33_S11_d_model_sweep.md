# PHASE 33 — S11 d_model SWEEP

## Kế hoạch controlled sweep cho Transformer Representation Width (`d_model`)

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S10_DROPOUT-v1`  
**Sweep ID:** `S11_DMODEL`  
**Output version:** `SWEEP_S11_DMODEL-v1`  
**Phase trước:** `Phase_32_S10_Dropout_sweep.md`

---

# 1. Vai trò của Phase 33

Phase 33 là controlled experiment thứ mười một trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target scaling, lookback, pooling, activation, batch size, learning rate, weight decay, dropout, số attention heads, số encoder layers, FFN width, loss, epoch budget, gradient clipping, sample population và seed đã được khóa từ Phase 32, Transformer representation width nào phù hợp hơn cho bài toán dự báo `Appliances`: `d_model=32` hay `d_model=64`?

Phase 33 chỉ thay đúng một hyperparameter được đăng ký:

```text
TRANSFORMER d_model
```

với hai condition:

```text
D32 = d_model 32
D64 = d_model 64
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Registered\ Capacity\ Factor
+
Same\ Data
+
Same\ Heads/Layers/FFN
+
Expected\ Parameter\ Count\ Change
+
Same\ Training\ Protocol
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí Phase 33 trong master plan

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
```

Phase 33 không được thay:

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
num_heads
num_layers
ffn_dim
loss
epoch cap
gradient clipping
RevIN
boundary protocol
```

---

# 3. Câu hỏi nghiên cứu của S11

Phase 33 phải trả lời:

```text
1. D32 hay D64 tạo Validation RMSE Wh thấp hơn?
2. D32 có đủ representation capacity cho task hiện tại không?
3. D64 có cải thiện forecasting đủ rõ để biện minh cho model lớn hơn không?
4. Parameter count/checkpoint size/runtime/memory context thay đổi thế nào?
5. Best epoch và convergence behavior khác nhau ra sao?
6. Gradient norm/clipping behavior thay đổi thế nào?
7. MAE/R² có cùng ranking với RMSE không?
8. d_model nào trở thành current reference cho Phase 34?
```

---

# 4. Current reference từ Phase 32

Load:

```text
s10_dropout_winner.json
s10_reference_update.json
phase_32_signoff.json
```

Resolve:

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
```

Phase 33 không hard-code các winner cũ.

---

# 5. Candidate registry

```text
D32 = 32
D64 = 64
```

No:

```text
D48
D96
D128
```

inside S11.

---

# 6. Reference reuse

Current S10 winner still uses:

```text
D64
```

If exact S11 match:

```text
D64 → REUSE S10 winner
D32 → NEW fresh run
```

Normal S11 cost:

```text
1 new run
+
1 reused reference
```

---

# 7. d_model architecture semantics

Canonical model:

```text
X [B,L,F]
→ Linear(F,D)
→ Fixed Sinusoidal PE
→ Transformer Encoder
→ H [B,L,D]
→ selected pooling
→ z [B,D]
→ Linear(D,1)
→ y_hat [B,1]
```

D32:

```text
Linear(F,32)
Encoder width 32
Pooling output [B,32]
Head Linear(32,1)
```

D64:

```text
Linear(F,64)
Encoder width 64
Pooling output [B,64]
Head Linear(64,1)
```

External output contract remains:

```text
[B,1]
```

---

# 8. Dynamic input feature dimension

`F` comes from the selected `FEATURESETS-v1` variant.

Hard:

```text
input_size = runtime feature count
```

Do not hard-code `F=31`.

---

# 9. Head count is frozen

Hard:

```text
num_heads = H4 = 4
```

for both candidates.

Phase 34 is the head-count sweep.

---

# 10. Head divisibility gate

Required:

\[
d_{model}\bmod num\_heads=0
\]

Therefore:

```text
D32/H4 = 8
D64/H4 = 16
```

Hard assert before model construction:

```python
assert d_model % num_heads == 0
```

---

# 11. Head dimension changes inherently

Because H4 is frozen:

```text
D32 → head_dim 8
D64 → head_dim 16
```

This is an inherent consequence of changing `d_model`.

Do not modify heads to keep `head_dim` fixed.

Scientific wording:

> Effect of changing model width under four fixed attention heads.

Not:

> Effect of d_model with constant head dimension.

---

# 12. Layers are frozen

Hard:

```text
num_layers = N2 = 2
```

No depth compensation.

---

# 13. FFN width is frozen

Hard:

```text
ffn_dim = F128 = 128
```

Phase 36 handles FFN sweep.

---

# 14. FFN ratio changes inherently

```text
D32:
128 / 32 = 4

D64:
128 / 64 = 2
```

Do not change F128 to preserve ratio.

That would be a two-factor experiment.

S11 answers:

> D32 vs D64 under fixed absolute FFN width 128.

---

# 15. Positional encoding follows d_model

Policy remains:

```text
FIXED SINUSOIDAL
```

but width must follow D:

```text
D32 → PE [...,32]
D64 → PE [...,64]
```

No learned PE.

Hard:

```text
max_seq_len >= L*
```

for both.

---

# 16. Pooling remains frozen

`P*` from S5 remains unchanged.

LAST_STEP:

```text
[B,L,D] → [B,D] by H[:,-1,:]
```

MEAN:

```text
[B,L,D] → [B,D] by H.mean(dim=1)
```

No pooling refactor.

---

# 17. Activation remains frozen

`A*` from S6:

```text
ReLU or GELU
```

No change.

---

# 18. Dropout remains frozen

`DR*` from S10 is identical for D32/D64.

Also preserve:

```text
dropout-site topology
dropout-scope fingerprint
train/eval semantics
```

Tensor shapes differ due D, so identical dropout masks are not required.

---

# 19. Parameter count is expected to differ

`d_model` is a capacity parameter.

Expected:

```text
Params(D32) < Params(D64)
```

This is not an error.

Unlike pooling/activation/dropout sweeps, parameter-schema equality is not expected.

---

# 20. What must remain the same

Even though tensor sizes change:

```text
same semantic module roles
same number of layers
same head count
same FFN width
same bias policy
same LayerNorm policy
same PE policy
same pooling
same regression output size
same dropout sites
```

Only D-dependent shapes may differ.

---

# 21. Expected D-dependent shape whitelist

Allowed shape changes:

```text
input projection
MHA Q/K/V / in-projection
MHA out-projection
FFN first Linear input dimension
FFN second Linear output dimension
LayerNorm affine vectors
encoded representation width
pooling output width
regression head input width
PE buffer width
```

No unrelated module may appear/disappear.

---

# 22. Architecture-role fingerprint

Because parameter shapes differ, use a semantic topology fingerprint over:

```text
module path
module class
semantic role
layer index
trainable flag
dimension-dependency category
```

Expected:

```text
architecture_role_fingerprint_D32
==
architecture_role_fingerprint_D64
```

while parameter-shape fingerprints differ.

---

# 23. State-dict semantics

Recommended expectation:

```text
same state_dict key names
different shapes for D-dependent tensors
```

Wrong-width strict loading must fail.

Do not implement slicing/projection fallback.

---

# 24. Reference trainable-parameter formula

Under canonical `TRANSFORMER-v1` assumptions:

```text
Input Linear bias=True
MHA bias=True
FFN Linear biases=True
2 affine LayerNorms per layer
N encoder layers
FFN width M
Regression head bias=True
Sinusoidal PE has no trainable parameters
```

Reference formula:

\[
Params(F,D,N,M)
=
FD+D
+
N(4D^2+2DM+9D+M)
+
D+1
\]

Equivalent:

\[
Params
=
FD
+
N(4D^2+2DM+9D+M)
+
2D+1
\]

This formula is an audit reference only.

Runtime parameter count is authoritative.

---

# 25. Reference simplification for N2/F128

For:

```text
N=2
M=128
```

\[
Params(F,D)
=
8D^2+FD+532D+257
\]

Therefore, only as symbolic expectation:

```text
D32:
25,473 + 32F

D64:
67,073 + 64F
```

Do not report these as observed until runtime architecture is audited.

---

# 26. Parameter matching is forbidden

Do not change:

```text
heads
layers
FFN
```

to equalize parameter count.

The increased parameter count is part of the intended capacity effect.

---

# 27. Attention output geometry

With same:

```text
B
H=4
L*
```

attention weights remain:

```text
[B,4,L*,L*]
```

for both candidates.

Internal head width differs:

```text
8 vs 16
```

No scientific attention extraction in S11.

---

# 28. Compute/memory context

D64 is expected to use more:

```text
trainable parameters
gradient state
AdamW moment state
activations
projection compute
FFN interface compute
checkpoint storage
```

than D32.

Actual runtime/memory depends on environment and must be measured where possible.

These are secondary metrics only.

---

# 29. Working hypotheses

H-S11-01:

```text
D64 may improve Validation accuracy because it has greater representation capacity.
```

H-S11-02:

```text
D32 may be sufficient and achieve similar/better RMSE with fewer parameters.
```

H-S11-03:

```text
D32 may show underfitting-like behavior under the current frozen setup.
```

All:

```text
UNTESTED
```

before execution.

---

# 30. Preconditions

Required:

```text
Phase 32 PASS
or PASS_WITH_WARNING with no unresolved CRITICAL issue

approved_for_phase33 = true
```

Required artifacts:

```text
s10_dropout_winner.json
s10_reference_update.json
phase_32_signoff.json
```

---

# 31. Required upstream contracts

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
```

---

# 32. Carry-forward warnings

Propagate all unresolved non-critical warnings, e.g.:

```text
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
BOUNDARY_WINNER
SAMPLE_ORDER_NOT_VERIFIABLE
OPTIMIZER_SCOPE_WARNING
DROPOUT_SCOPE_WARNING
```

into:

```text
manifest
summary
report
winner
Phase 34 reference update
```

---

# 33. Frozen data contract

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
```

No model capacity-dependent sample changes.

---

# 34. Frozen optimization contract

Hard:

```text
batch = B*
optimizer = AdamW
learning_rate = LR*
weight_decay = WD*
dropout = DR*
loss = MSE
max_epochs = 50
patience = 10
min_delta = 0
gradient clip = 1.0
scheduler = None
warmup = None
gradient accumulation = 1
seed = 42
TRAINING_ENGINE-v1
METRICS-v1
```

---

# 35. Optimizer parameter-group policy

Preserve S9's frozen:

```text
optimizer group topology
parameter-group semantic policy
```

for the new D32 model.

The exact parameter names/shapes change where D-dependent, but semantic grouping rule must remain identical.

No new bias/LayerNorm exclusion policy.

---

# 36. Initialization fairness

Direct weight equality is not applicable because shapes differ.

Required instead:

```text
same seed
same PyTorch environment
same model builder code
same default initialization policy
custom initialization = none
same bias policy
same LayerNorm initialization policy
```

Create an:

```text
initialization_policy_fingerprint
```

not an equality requirement over raw model states.

---

# 37. Same sample order

Because data and B* are fixed:

```text
D32 and D64 should use the same Train permutation/batch grouping
```

if D64 historical order provenance exists.

If not:

```text
D64_ORDER_MATCH = NOT_VERIFIABLE
```

Do not retrain D64 to recover metadata.

---

# 38. Dropout RNG caveat

Even with same seed/sample order:

```text
D32/D64 dropout masks may differ
```

because tensor shapes and RNG consumption differ.

Do not manually synchronize masks.

---

# 39. D64 reference reuse gate

Exact match on:

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
H1
WB0
WINDOWPOP-v1

D64
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
seed42
TRAINING_ENGINE-v1
METRICS-v1
```

Mismatch:

```text
STOP.
```

---

# 40. Fresh D32 run

Required execution:

```text
register run
↓
reseed 42
↓
fresh Train DataLoader
↓
fresh Validation DataLoader
↓
fresh Transformer(D32,H4,N2,F128,...)
↓
fresh AdamW(LR*,WD*)
↓
MSE
↓
TRAINING_ENGINE-v1
```

No warm-start.

---

# 41. Forbidden transfer techniques

Do not:

```text
copy first 32 channels from D64
slice attention matrices
project D64 checkpoint
distill D64 into D32
reuse optimizer moments
```

These are separate compression/distillation experiments.

---

# 42. No width compensation

Do not change for D32:

```text
batch
LR
WD
dropout
heads
layers
FFN
epochs
clip
```

even if D32 could support different settings.

---

# 43. D32 model geometry assertions

Runtime expected:

```text
d_model = 32
num_heads = 4
head_dim = 8
num_layers = 2
ffn_dim = 128
FFN ratio = 4
```

---

# 44. D64 reference geometry assertions

Reference:

```text
d_model = 64
num_heads = 4
head_dim = 16
num_layers = 2
ffn_dim = 128
FFN ratio = 2
```

---

# 45. Input projection audit

Expected:

```text
D32:
weight [32,F]
bias [32]

D64:
weight [64,F]
bias [64]
```

subject to standard Linear weight layout.

---

# 46. MHA geometry audit

Expected:

```text
D32:
embed_dim=32
heads=4
head_dim=8

D64:
embed_dim=64
heads=4
head_dim=16
```

No new `kdim`/`vdim` customization.

---

# 47. FFN geometry audit

Expected:

```text
D32:
Linear(32,128)
→ A*
→ Dropout(DR*)
→ Linear(128,32)

D64:
Linear(64,128)
→ A*
→ Dropout(DR*)
→ Linear(128,64)
```

---

# 48. LayerNorm audit

Normalized shape follows D:

```text
D32 → 32
D64 → 64
```

No BatchNorm.

---

# 49. Positional encoding audit

For selected L*:

```text
D32 PE width = 32
D64 PE width = 64
max_seq_len >= L*
finite values
non-trainable
same sinusoidal policy
```

---

# 50. Pooling/head audit

Expected:

```text
D32:
[B,L,32] → [B,32] → Linear(32,1)

D64:
[B,L,64] → [B,64] → Linear(64,1)
```

Final:

```text
[B,1]
```

for both.

---

# 51. Pre-training D32 sanity

Before official run:

```text
B1 forward
selected-B forward
final-partial-batch shape
finite outputs
synthetic MSE
backward
finite gradients
attention-aware API compatibility
strict wrong-width load failure
```

No optimizer step in pure architecture sanity.

Official training starts from fresh objects afterward.

---

# 52. D32 attention API sanity

Small diagnostic only:

```text
forward_with_attention
```

should return:

```text
prediction [B,1]
2 attention tensors
each [B,4,L*,L*]
```

No attention interpretation.

---

# 53. Parameter audit policy

Record runtime:

```text
total parameters
trainable parameters
parameter groups by module role
```

Hard expected direction:

```text
D32 < D64
```

If not, investigate.

---

# 54. Reference parameter formula audit

Compare runtime count with symbolic reference formula when assumptions match.

Status can be:

```text
MATCH
EXPECTED_IMPLEMENTATION_DIFFERENCE
UNEXPECTED_DIFFERENCE
FORMULA_NOT_APPLICABLE
```

Never modify architecture merely to satisfy formula.

---

# 55. Primary metric

Hard:

```text
best_validation_rmse_wh
```

from verified BEST checkpoint.

---

# 56. Secondary metrics

```text
Validation MAE Wh
Validation R²
best epoch
epochs completed
stop reason
train loss
gradient diagnostics
clipping fraction
runtime
parameter count
checkpoint size
optional memory
```

---

# 57. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{D32},
RMSE_{D64}
\right)
\]

Use full precision.

---

# 58. Exact tie rule

If exact full-precision RMSE equality:

```text
prefer D32
```

because it has:

```text
fewer parameters
smaller internal representation
lower expected compute/memory cost
```

This rule applies only to an exact tie.

---

# 59. No efficiency override

If D64 has lower RMSE by any non-zero full-precision margin:

```text
D64 wins.
```

Do not override with model size.

---

# 60. Effect formulas

Let:

```text
RMSE32
RMSE64
```

\[
\Delta RMSE_{32\rightarrow64}
=
RMSE_{32}-RMSE_{64}
\]

Positive:

```text
D64 improves.
```

Relative:

\[
100\times\frac{RMSE_{32}-RMSE_{64}}{RMSE_{32}}
\]

Also:

```text
MAE delta = MAE32 - MAE64
R² delta = R²64 - R²32
```

---

# 61. Parameter increase metrics

Using runtime counts:

\[
\Delta Params=Params_{64}-Params_{32}
\]

\[
Increase\%=
100\times\frac{Params_{64}-Params_{32}}{Params_{32}}
\]

\[
Ratio=\frac{Params_{64}}{Params_{32}}
\]

Do not infer parameter ratio from d_model ratio.

---

# 62. Capacity-efficiency diagnostic

Optional:

\[
RMSEGainPer10kExtraParams
=
\frac{RMSE_{32}-RMSE_{64}}
{(Params_{64}-Params_{32})/10000}
\]

Label:

```text
ENGINEERING_CONTEXT_ONLY
```

Never use for winner selection.

---

# 63. Pareto context

Classify:

```text
D32_DOMINATES
D64_DOMINATES
ACCURACY_EFFICIENCY_TRADEOFF
EXACT_TIE_D32_PARSIMONY
```

Examples:

```text
D32 lower RMSE + fewer params → D32 dominates
D64 lower RMSE + more params → tradeoff
```

Official winner still follows RMSE/tie rule.

---

# 64. Metric ranking divergence

If RMSE and MAE disagree:

```text
METRIC_RANKING_DIVERGENCE
```

Record it.

Winner remains RMSE-based.

---

# 65. BEST verification for D32

After official D32 run:

```text
fresh Transformer(d_model=32)
↓
strict-load BEST
↓
model.eval()
↓
full ordered Validation
↓
inverse target transform if required
↓
METRICS-v1
↓
verify stored BEST metrics
```

---

# 66. D64 reference verification

No retrain.

Verify:

```text
correct D64 config
BEST already verified
same population
same current selections
same metric version
same Training Engine provenance
```

---

# 67. Checkpoint metadata

Each run must include:

```text
d_model
num_heads
head_dim
num_layers
ffn_dim
ffn expansion ratio
PE policy
dropout
parameter count
model config fingerprint
architecture-role fingerprint
```

---

# 68. Wrong-width load test

Expected:

```text
strict D64 checkpoint → D32 model
FAILS
```

Do not suppress shape mismatch with `strict=False` or custom truncation.

---

# 69. Same sample population

Hard:

```text
Train IDs identical
Validation IDs identical
WINDOWPOP-v1 identical
```

No capacity-dependent filtering.

---

# 70. Same DataLoader policy

```text
same B*
shuffle Train=true
shuffle eval=false
drop_last=false
same worker policy
same generator seed policy
```

---

# 71. Train coverage

Each complete D32 epoch:

```text
observed samples = expected Train population
unique samples = expected Train population
duplicates = 0
missing = 0
```

---

# 72. Optimizer-step budget

Because batch/population fixed:

```text
steps_per_epoch_D32 = steps_per_epoch_D64
```

for complete epochs.

Total steps may differ due early stopping.

---

# 73. Gradient diagnostic caveat

Global L2 gradient norm depends on model dimensionality.

Do not interpret:

```text
higher D64 grad norm
```

as automatically worse stability.

Use together:

```text
clipping fraction
nonfinite events
learning curves
Validation metrics
```

---

# 74. Optional normalized gradient context

Can record:

```text
global_grad_norm / sqrt(trainable_parameter_count)
```

as diagnostic only.

No winner selection.

---

# 75. Convergence diagnostics

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
```

---

# 76. Underfitting-like diagnostic

Possible D32 signal:

```text
higher Train error
higher Validation RMSE
plateau
```

Do not diagnose from model size alone.

---

# 77. Overfitting-like diagnostic

Possible D64 signal:

```text
lower Train error
worse Validation RMSE
larger post-best deterioration
```

Again descriptive only.

---

# 78. Runtime diagnostics

Record if available:

```text
mean epoch seconds
median epoch seconds
total runtime
time to best
samples/sec
```

Runtime must not override RMSE.

---

# 79. Checkpoint-size diagnostics

Record:

```text
BEST full checkpoint bytes
LAST full checkpoint bytes
optional pure state_dict bytes
```

Do not mix full checkpoint and state_dict sizes without labels.

---

# 80. Peak memory diagnostics

Optional depending backend support:

```text
peak allocated
peak reserved
```

If unavailable:

```text
NOT_AVAILABLE
```

No fabrication.

---

# 81. Single-seed limitation

Hard report note:

```text
S11 uses seed 42 only.
```

No mean±std.

---

# 82. Validation-only limitation

Hard:

```text
S11 winner is a development selection based on Validation.
```

No Test claim.

---

# 83. Capacity interaction limitations

Mandatory report:

```text
d_model × heads
d_model × FFN width
d_model × dropout
d_model × LR
d_model × WD
d_model × batch
d_model × epoch budget
```

are not factorially explored.

---

# 84. No Test access

Hard:

```text
Test loader not iterated
Test predictions absent
Test metrics absent
```

---

# 85. No hidden capacity search

Forbidden:

```text
D16
D48
D96
D128
```

after seeing S11 results.

---

# 86. No model transfer/compression

Forbidden:

```text
distillation
pruning
low-rank projection
channel slicing
weight interpolation
```

---

# 87. Run failure policy

If D32 has technical failure:

```text
S11 incomplete
```

until documented technical rerun succeeds.

Do not declare D64 winner automatically.

---

# 88. Numerical failure policy

NaN/Inf:

```text
audit config/data/device/model first
```

If genuine:

```text
record failure
S11 incomplete under strict core protocol
```

No silent candidate exclusion.

---

# 89. Score-based rerun forbidden

Do not rerun D32 because score looks poor.

Do not retrain D64 to seek a better stochastic draw.

---

# 90. Discrepancy taxonomy

```text
S10_REFERENCE_MISSING
S10_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
POOLING_DRIFT
ACTIVATION_DRIFT
BATCH_DRIFT
LEARNING_RATE_DRIFT
WEIGHT_DECAY_DRIFT
DROPOUT_DRIFT
DMODEL_DEFINITION_MISMATCH
HEAD_COUNT_DRIFT
DMODEL_HEAD_NOT_DIVISIBLE
HEAD_DIM_INVALID
LAYER_COUNT_DRIFT
FFN_DIM_DRIFT
FFN_COMPENSATION
POSITIONAL_ENCODING_DIM_MISMATCH
POSITIONAL_ENCODING_POLICY_DRIFT
INPUT_PROJECTION_SHAPE_ERROR
MHA_EMBED_DIM_ERROR
FFN_SHAPE_ERROR
LAYERNORM_SHAPE_ERROR
REGRESSION_HEAD_SHAPE_ERROR
OUTPUT_SHAPE_ERROR
UNEXPECTED_MODULE_TOPOLOGY_CHANGE
STATE_DICT_KEY_DRIFT
PARAMETER_COUNT_UNEXPECTED_DIRECTION
INITIALIZATION_POLICY_DRIFT
SAMPLE_ORDER_POLICY_DRIFT
POPULATION_MISMATCH
TARGET_ID_MISMATCH
X_SCALER_MISMATCH
TARGET_SCALER_MISMATCH
OPTIMIZER_DRIFT
EPOCH_BUDGET_DRIFT
GRADIENT_CLIP_DRIFT
REFERENCE_RUN_MISMATCH
WARM_START_USED
WEIGHT_SLICING_USED
DISTILLATION_USED
OPTIMIZER_STATE_REUSE
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

# 91. Status model

PASS:

```text
D64 reference valid
D32 architecture valid
same population/training
D32 BEST verified
capacity diagnostics generated
winner selected
Phase34 reference generated
Test untouched
```

PASS_WITH_WARNING examples:

```text
tiny RMSE margin
metric divergence
accuracy-efficiency tradeoff
reference order not verifiable
memory metric unavailable
inherited warning
```

FAIL examples:

```text
heads changed
FFN changed
invalid divisibility
sample mismatch
warm-start
wrong D32 geometry
unresolved run failure
Test access
```

---

# 92. Required artifact directory

```text
artifacts/
└── sweeps/
    └── S11_d_model/
        ├── s11_d_model_sweep_manifest.json
        ├── s11_d_model_sweep_contract.json
        ├── s11_d_model_preflight_audit.csv
        ├── s11_run_matrix.csv
        ├── s11_d_model_definition_audit.csv
        ├── s11_architecture_role_audit.csv
        ├── s11_architecture_shape_delta_audit.csv
        ├── s11_parameter_count_audit.csv
        ├── s11_parameter_efficiency_diagnostics.csv
        ├── s11_positional_encoding_audit.csv
        ├── s11_mha_geometry_audit.csv
        ├── s11_ffn_geometry_audit.csv
        ├── s11_d_model_unit_tests.csv
        ├── s11_common_data_audit.csv
        ├── s11_d_model_training_audit.csv
        ├── s11_initialization_policy_audit.csv
        ├── s11_sample_order_audit.csv
        ├── s11_dropout_scope_audit.csv
        ├── s11_optimizer_budget_audit.csv
        ├── s11_d_model_run_provenance.csv
        ├── s11_d_model_metrics.csv
        ├── s11_d_model_effect.csv
        ├── s11_capacity_efficiency_context.csv
        ├── s11_capacity_pareto_context.json
        ├── s11_optimization_diagnostics.csv
        ├── s11_convergence_diagnostics.csv
        ├── s11_runtime_capacity_diagnostics.csv
        ├── s11_generalization_diagnostics.csv
        ├── s11_hypothesis_outcomes.csv
        ├── s11_d_model_findings.csv
        ├── s11_d_model_winner.json
        ├── s11_reference_update.json
        ├── s11_d_model_sweep_tests.csv
        ├── s11_d_model_discrepancies.json
        ├── s11_d_model_sweep_summary.json
        ├── s11_d_model_sweep_report.md
        ├── figures/
        │   ├── S11_01_validation_rmse_by_epoch.png
        │   ├── S11_02_validation_mae_by_epoch.png
        │   ├── S11_03_train_loss_by_epoch.png
        │   ├── S11_04_gradient_clipping_fraction.png
        │   ├── S11_05_best_validation_metrics.png
        │   ├── S11_06_parameter_count_vs_rmse.png
        │   ├── S11_07_runtime_vs_rmse.png
        │   ├── S11_08_checkpoint_size_comparison.png
        │   └── S11_09_generalization_gap_optional.png
        ├── README_S11_DMODEL_SWEEP.md
        └── phase_33_signoff.json
```

`generalization` and peak-memory outputs are optional if not supported.

The new D32 scientific run remains under:

```text
artifacts/runs/<run_id>/
```

No checkpoint duplication.

---

# 93. Run matrix schema

`s11_run_matrix.csv`:

```text
sweep_id
d_model_id
d_model
num_heads
head_dim
num_layers
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
population_fingerprint
feature_fingerprint
seed
model_config_id
training_config_id
status
```

---

# 94. Sweep manifest schema

`s11_d_model_sweep_manifest.json`:

```text
sweep_version
sweep_id
source_s10_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
dropout_probability
candidate_d_models = [32,64]
fixed_num_heads = 4
fixed_num_layers = 2
fixed_ffn_dim = 128
head_dims = {D32:8,D64:16}
ffn_expansion_ratios = {D32:4,D64:2}
new_runs_required
reused_runs
swept_field = d_model
capacity_change_expected = true
parameter_count_equality_expected = false
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = D32_ON_EXACT_RMSE_TIE
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

# 95. Sweep contract schema

`s11_d_model_sweep_contract.json` must state:

```text
Only d_model changes.

D32=32.
D64=64.

H4/N2/F128 remain fixed.

D32 head_dim=8.
D64 head_dim=16.

D32 FFN ratio=4.
D64 FFN ratio=2.

Consequences are documented, not compensated.

Parameter counts expected to differ.

Same data/population.
Same optimization/regularization.
Same initialization policy.
Same sample-order policy.

D64 reused.
D32 fresh.

No warm-start/slicing/distillation.

Validation RMSE Wh selects winner.
Exact tie → D32.
Test forbidden.
```

---

# 96. Preflight audit

`s11_d_model_preflight_audit.csv` checks:

```text
phase32_pass
approved_for_phase33
s10_winner_valid
all previous selections locked
D32/D64 registered
H4 fixed
N2 fixed
F128 fixed
D/head divisibility
head dims
FFN ratios
PE capacity
dropout scope
population
Training Engine
Metric version
seed
Test lock
status
```

---

# 97. d_model definition audit

`s11_d_model_definition_audit.csv`:

```text
d_model_id
d_model
num_heads
head_dim
num_layers
ffn_dim
ffn_expansion_ratio
input_size
lookback
pe_width
pooling_output_width
regression_head_input_width
valid_divisibility
status
```

---

# 98. Architecture role audit

`s11_architecture_role_audit.csv`:

```text
module_path
semantic_role
module_type_d32
module_type_d64
present_d32
present_d64
dimension_dependency
role_match
status
```

---

# 99. Shape delta audit

`s11_architecture_shape_delta_audit.csv`:

```text
path
semantic_role
shape_d32
shape_d64
expected_to_change
reason
unexpected_difference
status
```

---

# 100. Parameter count audit

`s11_parameter_count_audit.csv`:

```text
d_model_id
runtime_feature_count
runtime_trainable_parameters
runtime_total_parameters
reference_formula_parameters
formula_applicable
formula_difference
expected_direction
direction_valid
status
```

---

# 101. Parameter efficiency artifact

`s11_parameter_efficiency_diagnostics.csv`:

```text
d32_parameters
d64_parameters
absolute_parameter_increase
parameter_increase_pct
parameter_ratio
d32_checkpoint_bytes
d64_checkpoint_bytes
checkpoint_size_ratio
status
```

---

# 102. PE audit

`s11_positional_encoding_audit.csv`:

```text
d_model_id
pe_policy
pe_trainable
max_seq_len
selected_lookback
pe_buffer_shape
pe_width_matches
lookback_supported
finite
status
```

---

# 103. MHA geometry audit

`s11_mha_geometry_audit.csv`:

```text
d_model_id
embed_dim
num_heads
head_dim
divisible
kdim
vdim
batch_first
attention_output_shape_expected
status
```

---

# 104. FFN geometry audit

`s11_ffn_geometry_audit.csv`:

```text
d_model_id
ffn_dim
first_linear_in
first_linear_out
second_linear_in
second_linear_out
expansion_ratio
activation
dropout
status
```

---

# 105. Unit tests

`s11_d_model_unit_tests.csv` should cover at least:

```text
D32 accepted
D64 accepted
invalid D/H rejected
D32/H4 valid
D64/H4 valid
correct input projection shapes
correct PE widths
correct MHA embed dims
head_dim 8/16
FFN 32→128→32
FFN 64→128→64
LayerNorm 32/64
pooling output D
regression head input D
[B,1] output
B1
selected B*
partial batch
finite forward
finite backward
attention API compatibility
strict wrong-width checkpoint load failure
D32 params < D64
```

---

# 106. Common data audit

`s11_common_data_audit.csv`:

```text
split
sample_count_d32
sample_count_d64
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
population_equal
status
```

---

# 107. Training audit

`s11_d_model_training_audit.csv`:

```text
d_model_id
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
only_registered_capacity_factor_differs
status
```

---

# 108. Initialization policy audit

`s11_initialization_policy_audit.csv`:

```text
d_model_id
seed
initialization_policy_id
custom_initialization
model_builder_code_fingerprint
pytorch_version
initial_state_fingerprint
direct_state_equality_applicable
policy_match
status
```

Expected across D:

```text
direct_state_equality_applicable = false
```

---

# 109. Sample order audit

`s11_sample_order_audit.csv`:

```text
epoch_or_probe
d32_order_fingerprint
d64_order_fingerprint
d64_reference_available
same_order
status
```

---

# 110. Dropout scope audit

`s11_dropout_scope_audit.csv`:

```text
d_model_id
dropout_probability
dropout_scope_fingerprint
site_count
site_paths_match
semantic_scope_match
mask_shape_equality_required=false
status
```

---

# 111. Optimizer budget audit

`s11_optimizer_budget_audit.csv`:

```text
d_model_id
train_samples_per_epoch
train_batch_size
steps_per_epoch
epochs_completed
total_steps
best_epoch
steps_to_best
same_steps_per_completed_epoch
status
```

---

# 112. Run provenance

`s11_d_model_run_provenance.csv`:

```text
d_model_id
d_model
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

# 113. Primary metrics table

`s11_d_model_metrics.csv`:

```text
d_model_id
d_model
run_id
source_type
num_heads
head_dim
ffn_dim
ffn_expansion_ratio
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

# 114. Effect table

`s11_d_model_effect.csv`:

```text
d32_run_id
d64_run_id
d32_rmse_wh
d64_rmse_wh
rmse_delta_d32_to_d64_wh
rmse_improvement_pct
d32_mae_wh
d64_mae_wh
mae_delta
d32_r2
d64_r2
r2_delta
d32_parameters
d64_parameters
parameter_increase
parameter_increase_pct
rmse_winner
metric_ranking_divergence
status
```

---

# 115. Capacity efficiency context

`s11_capacity_efficiency_context.csv`:

```text
d_model_id
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

# 116. Pareto context

`s11_capacity_pareto_context.json`:

```text
accuracy_metric
cost_metrics
relationship
d32_dominates
d64_dominates
accuracy_efficiency_tradeoff
official_winner
tie_rule_used
context_only=true
status
```

---

# 117. Optimization diagnostics

`s11_optimization_diagnostics.csv`:

```text
d_model_id
best_epoch
last_epoch
stop_reason
best_rmse_wh
last_rmse_wh
mean_grad_norm_preclip
max_grad_norm_preclip
normalized_grad_context_optional
mean_fraction_batches_clipped
nonfinite_events
best_to_last_gap
status
```

---

# 118. Convergence diagnostics

`s11_convergence_diagnostics.csv`:

```text
d_model_id
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

# 119. Runtime/capacity diagnostics

`s11_runtime_capacity_diagnostics.csv`:

```text
d_model_id
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

# 120. Optional generalization diagnostics

`s11_generalization_diagnostics.csv`:

```text
d_model_id
train_rmse_wh_at_best
validation_rmse_wh_at_best
rmse_gap_wh
train_mae_wh_at_best
validation_mae_wh_at_best
mae_gap_wh
capacity_interpretation
status
```

No Test.

---

# 121. Findings

`s11_d_model_findings.csv` possible codes:

```text
D32_GAIN
D64_GAIN
DMODEL_EXACT_TIE
D32_DOMINATES
D64_ACCURACY_GAIN_WITH_COST
ACCURACY_EFFICIENCY_TRADEOFF
METRIC_RANKING_DIVERGENCE
PARAMETER_COUNT_GROWTH
CHECKPOINT_SIZE_GROWTH
RUNTIME_DIFFERENCE
MEMORY_DIFFERENCE
POSSIBLE_D32_UNDERFITTING
POSSIBLE_D64_OVERFITTING
CONVERGENCE_DIFFERENCE
CLIPPING_DIFFERENCE
HEAD_DIM_CHANGE_CONTEXT
FFN_RATIO_CHANGE_CONTEXT
INITIALIZATION_POLICY_VERIFIED
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
INHERITED_WARNING
```

---

# 122. Winner artifact

`s11_d_model_winner.json` minimum:

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
selection_metric
selection_direction
tie_rule
winner_d_model_id
winner_d_model
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
winner_trainable_parameters
runner_up_d_model_id
runner_up_d_model
runner_up_rmse_wh
runner_up_trainable_parameters
rmse_margin_wh
rmse_margin_pct
parameter_difference
parameter_difference_pct
head_count
winner_head_dim
ffn_dim
winner_ffn_expansion_ratio
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 123. Phase 34 reference update

`s11_reference_update.json`:

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
previous_d_model=64
selected_d_model_id
selected_d_model
winner_run_id
winner_config_fingerprint
winner_rmse_wh
current_num_heads=4
current_head_dim
ffn_dim=128
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase34
```

---

# 124. Phase 34 handoff

Phase 34 tests:

```text
H2 = 2 heads
H4 = 4 heads
```

at selected D.

If D32 wins:

```text
H2 → head_dim16
H4 → head_dim8
```

If D64 wins:

```text
H2 → head_dim32
H4 → head_dim16
```

Both D32/D64 are divisible by H2 and H4.

Normal Phase 34 strategy:

```text
reuse S11 winner as H4 reference
train only H2.
```

---

# 125. Figures

Recommended:

```text
S11_01_validation_rmse_by_epoch.png
S11_02_validation_mae_by_epoch.png
S11_03_train_loss_by_epoch.png
S11_04_gradient_clipping_fraction.png
S11_05_best_validation_metrics.png
S11_06_parameter_count_vs_rmse.png
S11_07_runtime_vs_rmse.png
S11_08_checkpoint_size_comparison.png
S11_09_generalization_gap_optional.png
```

Primary scientific figure:

```text
S11_01_validation_rmse_by_epoch.png
```

Capacity figures are secondary context.

---

# 126. Safe result interpretation

If D32 wins:

> Under the frozen H4/N2/F128 and selected optimization settings, D32 achieved lower Validation RMSE while using fewer trainable parameters.

If D64 wins:

> D64 achieved lower Validation RMSE under fixed H4/N2/F128, at the cost of additional model capacity.

If exact tie:

> No full-precision Validation RMSE advantage was observed for D64; D32 was selected by the predefined parsimony tie rule.

---

# 127. Interpretation prohibitions

Do not claim:

```text
larger dimensions always improve Transformers
more parameters caused better RMSE
head_dim alone caused the outcome
FFN ratio alone caused the outcome
D32 is underfit solely because it is smaller
D64 is overfit solely because it is larger
```

---

# 128. Required outputs

```text
O33.1  Sweep manifest
O33.2  Sweep contract
O33.3  Preflight audit
O33.4  Run matrix
O33.5  d_model definition audit
O33.6  Architecture-role audit
O33.7  Shape-delta audit
O33.8  Parameter-count audit
O33.9  Parameter-efficiency diagnostics
O33.10 Positional encoding audit
O33.11 MHA geometry audit
O33.12 FFN geometry audit
O33.13 Unit tests
O33.14 Common-data audit
O33.15 Training audit
O33.16 Initialization-policy audit
O33.17 Sample-order audit
O33.18 Dropout-scope audit
O33.19 Optimizer-budget audit
O33.20 Run provenance
O33.21 Reused D64 reference
O33.22 Verified new D32 run
O33.23 Metrics table
O33.24 d_model effect table
O33.25 Capacity-efficiency context
O33.26 Pareto context
O33.27 Optimization diagnostics
O33.28 Convergence diagnostics
O33.29 Runtime/capacity diagnostics
O33.30 Optional generalization diagnostics
O33.31 Hypothesis outcomes
O33.32 Findings
O33.33 Winner artifact
O33.34 Phase 34 reference update
O33.35 Figures
O33.36 Sweep tests
O33.37 Discrepancy log
O33.38 Sweep summary
O33.39 Human-readable report
O33.40 README
O33.41 Phase sign-off
```

---

# 129. Sweep acceptance tests

`s11_d_model_sweep_tests.csv` should include at least:

```text
S11T33-001 Phase32 valid
S11T33-002 approved_for_phase33 true
S11T33-003 S10 winner valid
S11T33-004 previous selected fields fixed
S11T33-005 D32=32
S11T33-006 D64=64
S11T33-007 no extra D candidate
S11T33-008 H4 fixed
S11T33-009 N2 fixed
S11T33-010 F128 fixed
S11T33-011 D32%4=0
S11T33-012 D64%4=0
S11T33-013 head_dim D32=8
S11T33-014 head_dim D64=16
S11T33-015 FFN ratio D32=4
S11T33-016 FFN ratio D64=2
S11T33-017 no head compensation
S11T33-018 no FFN compensation
S11T33-019 no layer compensation
S11T33-020 Train IDs equal
S11T33-021 Validation IDs equal
S11T33-022 WINDOWPOP equal
S11T33-023 feature fingerprint equal
S11T33-024 X scaler equal
S11T33-025 target transform equal
S11T33-026 lookback equal
S11T33-027 pooling equal
S11T33-028 activation equal
S11T33-029 batch equal
S11T33-030 LR equal
S11T33-031 WD equal
S11T33-032 dropout equal
S11T33-033 dropout scope equal
S11T33-034 H1/WB0 equal
S11T33-035 PE policy equal
S11T33-036 PE width follows D
S11T33-037 PE supports L*
S11T33-038 norm policy equal
S11T33-039 mask policy equal
S11T33-040 final output size equal
S11T33-041 input projection shapes correct
S11T33-042 MHA embed_dim correct
S11T33-043 MHA heads=4
S11T33-044 FFN geometries correct
S11T33-045 LayerNorm widths correct
S11T33-046 pooling widths correct
S11T33-047 regression heads correct
S11T33-048 B1 output valid
S11T33-049 partial batch valid
S11T33-050 D32 forward finite
S11T33-051 D32 backward finite
S11T33-052 attention API compatible
S11T33-053 wrong-width strict load fails
S11T33-054 no slicing fallback
S11T33-055 architecture roles equal
S11T33-056 expected shape whitelist passes
S11T33-057 no unexpected modules
S11T33-058 runtime param counts stored
S11T33-059 parameter equality not required
S11T33-060 D32 params < D64
S11T33-061 formula is audit-only
S11T33-062 runtime counts source of truth
S11T33-063 AdamW equal
S11T33-064 optimizer group policy equal
S11T33-065 MSE equal
S11T33-066 E50 equal
S11T33-067 patience10 equal
S11T33-068 clip1 equal
S11T33-069 no scheduler/warmup
S11T33-070 accumulation1
S11T33-071 drop_last false
S11T33-072 seed42
S11T33-073 initialization policy equal
S11T33-074 raw init equality N/A
S11T33-075 no custom D32 init
S11T33-076 sample-order checked
S11T33-077 dropout-mask equality not required
S11T33-078 D64 reference exact
S11T33-079 D64 reused
S11T33-080 D32 registered before train
S11T33-081 D32 fresh objects
S11T33-082 no warm-start
S11T33-083 no optimizer-state reuse
S11T33-084 no distillation
S11T33-085 D32 via Training Engine
S11T33-086 D32 BEST verified
S11T33-087 D64 BEST valid
S11T33-088 verified BEST rows only
S11T33-089 predictions finite
S11T33-090 metrics in Wh
S11T33-091 full-precision RMSE
S11T33-092 RMSE effect correct
S11T33-093 MAE effect correct
S11T33-094 R² effect correct
S11T33-095 winner=min RMSE
S11T33-096 exact tie=D32
S11T33-097 efficiency no non-tie override
S11T33-098 no arbitrary gain threshold
S11T33-099 metric divergence recorded
S11T33-100 parameter diagnostics generated
S11T33-101 checkpoint-size diagnostics generated
S11T33-102 runtime diagnostics generated
S11T33-103 memory missing handled honestly
S11T33-104 capacity context non-selection
S11T33-105 Pareto context generated
S11T33-106 gradient dimension caveat
S11T33-107 optimization diagnostics generated
S11T33-108 convergence diagnostics generated
S11T33-109 no size-only underfit claim
S11T33-110 no size-only overfit claim
S11T33-111 hypotheses recorded
S11T33-112 findings generated
S11T33-113 winner valid
S11T33-114 Phase34 reference update
S11T33-115 H4 reuse identified
S11T33-116 selected D valid H2/H4
S11T33-117 inherited warnings propagated
S11T33-118 no D48
S11T33-119 no D128
S11T33-120 no width-specific compensation
S11T33-121 no extra epochs
S11T33-122 no score-based rerun
S11T33-123 no SANITY/failed run ranked
S11T33-124 no scientific attention extraction
S11T33-125 no Test
S11T33-126 single-seed limitation
S11T33-127 Validation-only limitation
S11T33-128 head interaction documented
S11T33-129 FFN interaction documented
S11T33-130 regularization interaction documented
S11T33-131 budget limitation documented
S11T33-132 figures source-derived
S11T33-133 summary/report generated
S11T33-134 phase sign-off generated
```

---

# 130. Recommended notebook structure

```text
Cell 33.1  Phase title
Cell 33.2  Verify Phase32
Cell 33.3  Declare SWEEP_S11_DMODEL-v1
Cell 33.4  Load S10 winner/reference
Cell 33.5  Freeze previous selected fields
Cell 33.6  Define D32/D64
Cell 33.7  Assert H4/N2/F128
Cell 33.8  Derive head_dim/FFN ratios
Cell 33.9  Build run matrix
Cell 33.10 Audit common population
Cell 33.11 Audit architecture roles
Cell 33.12 Audit shape delta whitelist
Cell 33.13 Audit PE
Cell 33.14 Audit MHA
Cell 33.15 Audit FFN
Cell 33.16 Run D32 unit tests
Cell 33.17 Parameter count audit
Cell 33.18 Training config audit
Cell 33.19 Initialization policy audit
Cell 33.20 Sample-order audit
Cell 33.21 Dropout scope audit
Cell 33.22 Verify D64 reuse
Cell 33.23 Register D32
Cell 33.24 Reseed/fresh loaders/model
Cell 33.25 D32 sanity
Cell 33.26 Train D32
Cell 33.27 Verify D32 BEST
Cell 33.28 Build provenance
Cell 33.29 Optimizer budget
Cell 33.30 Metrics table
Cell 33.31 Compute effect
Cell 33.32 Capacity efficiency
Cell 33.33 Runtime/checkpoint/memory context
Cell 33.34 Pareto context
Cell 33.35 Optimization diagnostics
Cell 33.36 Convergence diagnostics
Cell 33.37 Optional generalization diagnostic
Cell 33.38 Hypotheses
Cell 33.39 Figures
Cell 33.40 Findings
Cell 33.41 Select winner
Cell 33.42 Winner JSON
Cell 33.43 Phase34 reference update
Cell 33.44 Tests/discrepancies
Cell 33.45 Summary/report
Cell 33.46 Register checksums
Cell 33.47 README
Cell 33.48 Sign-off
```

---

# 131. Execution flow

```text
Verify Phase32
→ Load S10 winner
→ Freeze prior selected config
→ Define D32/D64
→ Freeze H4/N2/F128
→ Verify divisibility
→ Audit same population
→ Audit architecture role/shape geometry
→ Audit PE/MHA/FFN
→ Verify D64 reference
→ Reuse D64
→ Register D32
→ Seed42 + fresh objects
→ D32 sanity
→ Train D32 via TRAINING_ENGINE-v1
→ Verify D32 BEST
→ Build runtime param counts
→ Build Wh metrics
→ Compute D effect
→ Build capacity/runtime context
→ Select min RMSE
→ Exact tie prefer D32
→ Update Phase34 reference
→ Write all artifacts/sign-off
```

---

# 132. Fail-fast order

Before training D32:

```text
1. Phase32 sign-off
2. S10 winner identity
3. Freeze previous selections
4. D32/D64 registry
5. H4/N2/F128 locks
6. D/head divisibility
7. same sample population
8. architecture role topology
9. D-dependent shape whitelist
10. PE
11. MHA geometry
12. FFN geometry
13. D32 forward/backward
14. training/optimizer config
15. initialization policy
16. D64 reuse eligibility
17. Test firewall
18. Registry readiness
```

---

# 133. Critical technical notes

## 133.1 Do not preserve head_dim by changing heads

That would be two factors.

## 133.2 Do not preserve FFN ratio by changing FFN width

F128 must remain fixed.

## 133.3 Parameter counts should differ

Capacity change is intended.

## 133.4 Raw initial weights cannot be equal

Shapes differ; audit initialization policy instead.

## 133.5 PE must not be hard-coded to 64

It must follow D.

## 133.6 D32 head must be `Linear(32,1)`

Not `Linear(64,1)`.

## 133.7 LayerNorm must follow D

Normalized dimension 32 vs64.

## 133.8 MHA embed_dim must follow D

Head count remains four.

## 133.9 Do not transfer D64 checkpoint into D32

Fresh run only.

## 133.10 Do not select model by parameter count/runtime except exact RMSE tie

Primary metric remains RMSE.

---

# 134. Phase 34 handoff requirements

Phase 34 must receive:

```text
selected_d_model
selected run_id
selected config fingerprint
current H4
current head_dim
F128
all previous selected config
population fingerprint
metric version
warnings
approved_for_phase34
```

Then Phase34 changes only:

```text
H2 vs H4
```

and normally:

```text
reuse H4 reference
train H2.
```

---

# 135. Phase sign-off schema

Create:

```text
phase_33_signoff.json
```

Minimum:

```text
phase = 33
phase_name = S11 d_model sweep
sweep_version
sweep_id
source_s10_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
dropout_probability
d32_run_id
d64_reference_run_id
new_run_ids
reused_run_ids
d32_parameter_count
d64_parameter_count
winner_d_model_id
winner_d_model
winner_run_id
winner_rmse_wh
winner_parameter_count
population_fingerprint
metric_version
architecture_role_audit_status
shape_delta_audit_status
mha_geometry_audit_status
ffn_geometry_audit_status
positional_encoding_audit_status
initialization_policy_audit_status
sample_order_match_status
capacity_efficiency_context_status
inherited_warnings
test_status
approved_for_phase34
overall_status
created_at
```

---

# 136. Definition of Done

\[
\boxed{
One\ Fixed\ Data/Training\ Setup
+
Two\ Model\ Widths
+
H4/N2/F128\ Frozen
+
Valid\ D\text{-}Dependent\ Geometry
+
One\ Fresh\ D32\ Run
+
One\ Valid\ Reused\ D64
+
Expected\ Capacity\ Difference
+
Verified\ BEST\ Metrics
+
Capacity\ Context
+
S11\ Winner
+
Phase34\ Reference
+
No\ Test
}
\]

---

# 137. Final status contract

```text
PHASE 33 tests d_model only.

Candidates:
D32 = 32
D64 = 64

Frozen:
H4
N2
F128
all prior selected data/optimization settings.

Consequences:
D32 head_dim=8
D64 head_dim=16
D32 FFN ratio=4
D64 FFN ratio=2.

These are documented, not compensated.

D64:
reuse S10 winner if exact match.

D32:
one fresh seed42 run.

Same:
Train/Validation IDs
WINDOWPOP-v1
X/y pipeline
lookback
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

Expected to differ:
internal representation width
D-dependent tensor shapes
head_dim
FFN ratio
parameter count
checkpoint size
runtime/memory context.

Not required:
same parameter count
same raw initial tensors
same dropout masks.

Required:
same initialization policy
same semantic architecture topology
same sample-order policy.

No:
head/FFN/layer compensation
warm-start
weight slicing
distillation
width-specific LR/WD/dropout/batch changes
extra D values.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer D32.

Engineering/capacity metrics:
secondary context only.

No Test.

After SWEEP_S11_DMODEL-v1 PASS:
proceed to
PHASE 34 — S12 Head sweep.
```

---

# 138. Final check

Correct:

```text
Load S10 winner
→ Freeze previous winners
→ D32 vs D64
→ H4/N2/F128 fixed
→ geometry audits
→ reuse D64
→ fresh D32
→ verify D32 BEST
→ same-population Validation comparison
→ capacity context
→ select min RMSE
→ tie D32
→ Phase34 reference
```

Incorrect:

```text
D32+H2
→ D64+H4
→ resize FFN
→ copy D64 weights
→ match parameter count
→ select by speed instead of RMSE
→ inspect Test
```

Chỉ sau khi `SWEEP_S11_DMODEL-v1` được sign-off mới chuyển sang **PHASE 34 — S12 Head sweep**.
