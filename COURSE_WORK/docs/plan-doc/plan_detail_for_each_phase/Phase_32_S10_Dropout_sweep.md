# PHASE 32 — S10 DROPOUT SWEEP

## Kế hoạch controlled sweep cho Transformer Encoder Dropout Probability

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S9_WEIGHTDECAY-v1`  
**Sweep ID:** `S10_DROPOUT`  
**Output version:** `SWEEP_S10_DROPOUT-v1`  
**Phase trước:** `Phase_31_S9_Weight-decay_sweep.md`

---

# 1. Vai trò của Phase 32

Phase 32 là controlled experiment thứ mười trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target scaling, lookback, pooling, activation, batch size, learning rate, weight decay, Transformer width/depth, MSE loss, epoch budget, early stopping, gradient clipping, sample population và seed đã được khóa từ Phase 31, dropout probability nào phù hợp hơn cho Transformer Encoder: `0.1`, `0.2` hay `0.3`?

Phase 32 chỉ thay đúng một conceptual factor:

```text
TRANSFORMER ENCODER DROPOUT PROBABILITY
```

với ba condition:

```text
DR01 = 0.1
DR02 = 0.2
DR03 = 0.3
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Dropout\ Factor
+
Frozen\ Dropout\ Sites
+
Same\ Parameters
+
Same\ Initialization
+
Same\ Mini\text{-}Batches
+
Correct\ Train/Eval\ Semantics
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí Phase 32 trong master execution plan

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

Phase 33
→ S11 d_model sweep
```

Phase 32 không được quay lại thay:

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

# 3. Câu hỏi nghiên cứu của S10

Phase 32 phải trả lời:

```text
1. Dropout 0.1, 0.2 hay 0.3 tạo Validation RMSE Wh thấp nhất?

2. Tăng dropout có cải thiện Validation generalization dưới WD/LR đã chọn không?

3. Dropout mạnh hơn có tạo dấu hiệu underfitting hoặc convergence chậm hơn không?

4. Learning curves và best-to-last RMSE gap thay đổi như thế nào?

5. Gradient norm/clipping behavior thay đổi ra sao theo learned trajectory?

6. Train-mode stochastic behavior và eval-mode deterministic behavior có đúng contract không?

7. MAE và R² có cùng ranking với RMSE không?

8. Dropout nào trở thành current reference cho Phase 33?
```

---

# 4. Current reference từ Phase 31

Phase 32 phải load:

```text
s9_weight_decay_winner.json
s9_reference_update.json
phase_31_signoff.json
```

để resolve current configuration:

```text
FV* = selected feature variant
YS* = selected target scaling
L*  = selected lookback
P*  = selected pooling
A*  = selected activation
B*  = selected batch size
LR* = selected learning rate
WD* = selected weight decay
```

Possible:

```text
WD* ∈ {0, 1e-4, 1e-3}
```

Phase 32 không hard-code `WD=1e-4`.

---

# 5. Canonical dropout options

Registry:

```text
DR01 = 0.1
DR02 = 0.2
DR03 = 0.3
```

Exact numeric values:

```text
0.1
0.2
0.3
```

No additional dropout value may enter S10 ranking.

---

# 6. Reference dropout

Current sequential reference từ Phase 31 vẫn uses:

```text
DR01 = 0.1
```

vì dropout chưa được swept trước Phase 32.

Nếu exact S10 contract match:

```text
REUSE S9 winner
```

as DR01 reference.

---

# 7. Dropout phải được định nghĩa theo implementation scope, không chỉ theo một con số

Một lỗi phổ biến là config ghi:

```text
dropout = 0.2
```

nhưng không rõ nó được áp dụng ở đâu.

S10 phải freeze **dropout-site scope** trước khi train.

Phase 32 không chỉ audit:

```text
p = 0.1 / 0.2 / 0.3
```

mà còn phải audit:

```text
site nào dùng p
site nào không dùng dropout
site nào có dropout cố định khác
```

---

# 8. Source of truth cho dropout sites

Source hierarchy:

```text
1. TRANSFORMER_IMPL-v1 actual implementation
2. current model config schema
3. S9 winner config
4. documentation/comments
```

Actual code/config semantics là source of truth.

Không suy đoán dropout placement chỉ từ tên hyperparameter.

---

# 9. Expected dropout scope từ Transformer-v1 contract

Theo Transformer implementation contract hiện tại, global encoder dropout được kỳ vọng ảnh hưởng các dropout locations đã có sẵn trong encoder, bao gồm các nhóm như:

```text
D1 — MultiheadAttention attention-weight dropout
D2 — self-attention output/residual-branch dropout
D3 — FFN hidden dropout after activation
D4 — FFN output/residual-branch dropout
```

Exact module count và exact code paths phải được xác minh runtime từ `TRANSFORMER_IMPL-v1`.

Nếu implementation thực tế khác, audit actual implementation; không tự thêm site để ép khớp danh sách expected.

---

# 10. MultiheadAttention dropout semantics

Nếu custom encoder sử dụng:

```text
nn.MultiheadAttention(..., dropout=p)
```

thì `p` ở đây là dropout probability trên attention weights theo PyTorch API.

S10 phải verify:

```text
MHA dropout p
```

được cập nhật đúng theo registered candidate nếu MHA dropout nằm trong frozen global-dropout scope.

---

# 11. Residual/FFN dropout semantics

Các `nn.Dropout` modules thuộc frozen encoder design phải nhận đúng candidate `p`.

Ví dụ expected pattern:

```text
self_attention(...)
→ residual dropout
→ residual add
→ LayerNorm

Linear(D,FFN)
→ Activation
→ FFN hidden dropout
→ Linear(FFN,D)
→ residual dropout
→ residual add
→ LayerNorm
```

Exact ordering phải giữ nguyên từ Transformer-v1.

---

# 12. Không thêm input dropout

Current baseline contract không có separate input-projection dropout.

Phase 32 không được thêm:

```text
dropout immediately after raw input
dropout after input projection
dropout after positional encoding
```

nếu các site đó không tồn tại trong frozen implementation.

---

# 13. Không thêm regression-head dropout

Current regression head remains:

```text
Linear(d_model,1)
```

without head dropout.

Do not add:

```text
Dropout before regression head
Dropout after pooling
```

only because S10 is a dropout sweep.

---

# 14. Không thêm pooling dropout

Pooling strategy `P*` remains exactly as selected in S5.

No additional dropout may be inserted inside:

```text
LAST_STEP
MEAN
```

readout path.

---

# 15. Dropout-site topology phải giống nhau

Across DR01/DR02/DR03:

```text
same number of dropout sites
same module locations
same call order
same MHA/dropout structure
```

Only registered probability value changes at existing global-dropout sites.

---

# 16. Dropout-site fingerprint

Create a stable fingerprint over:

```text
site_id
module path
site type
semantic location
controlled_by_global_dropout
```

Expected:

```text
same scope fingerprint
```

for all three conditions.

This is one of the most important S10 fairness checks.

---

# 17. Sites not controlled by S10 must remain frozen

If actual implementation has a dropout-like field not controlled by the global encoder dropout, such as a hypothetical:

```text
input_dropout = 0
head_dropout = 0
```

it must remain unchanged.

Do not sweep multiple independent dropout dimensions under one S10 label.

---

# 18. Canonical elementwise Dropout context

For standard elementwise dropout during training:

```text
drop probability = p
keep probability = q = 1-p
```

Registered keep probabilities:

```text
DR01:
p = 0.1
q = 0.9

DR02:
p = 0.2
q = 0.8

DR03:
p = 0.3
q = 0.7
```

For standard inverted dropout, retained activations are scaled during training so evaluation can use identity behavior.

This context applies to standard dropout semantics; do not multiply these keep probabilities across the whole network to claim a global “effective dropout”.

---

# 19. Train/eval semantics là hard contract

During training:

```text
model.train()
```

Dropout must be active.

During Validation and BEST-checkpoint verification:

```text
model.eval()
+
torch.inference_mode()
```

Dropout must be disabled according to module semantics.

A Validation pass accidentally executed in train mode invalidates the scientific metric.

---

# 20. Why Validation mode is critical in S10

If Validation is run with dropout active:

```text
predictions become stochastic
Validation RMSE depends on random masks
early stopping becomes noisy in an unintended way
winner comparison becomes invalid
```

Therefore:

```text
TRAIN mode for training only
EVAL mode for Validation
```

is a hard gate.

---

# 21. Train-mode restoration after Validation

Training Engine must correctly alternate:

```text
model.train()
→ training epoch

model.eval()
→ full Validation

model.train()
→ next training epoch
```

A model left in `eval()` after Validation would silently disable dropout for later training.

This is a critical S10-specific failure mode.

---

# 22. Validation repeated-forward determinism

For a fixed BEST checkpoint and identical Validation input:

```text
model.eval()
```

repeated forwards should be numerically stable/allclose under the active device tolerance policy.

Do not require universal cross-device bitwise equality.

---

# 23. Train-mode stochasticity sanity

On a disposable model/input:

```text
model.train()
```

dropout-enabled forward behavior should show the registered stochastic path.

Do not require that every pair of consecutive outputs must differ; stochastic coincidence is possible.

Use multiple calls or isolated dropout-module tests.

---

# 24. Do not reset RNG before every training batch

Never do:

```text
torch.manual_seed(42)
```

inside each batch loop to force reproducible dropout masks.

That would create repeated/correlated masks and materially change training.

Seed once at run startup according to `TRAINING_ENGINE-v1`.

---

# 25. Same seed does not mean same dropout mask semantics across p

All candidates use seed 42, but:

```text
p differs intentionally
```

so the realized masks differ semantically.

Do not require:

```text
DR01 mask == DR02 mask == DR03 mask
```

or nested-mask identity.

What must match is:

```text
initial RNG policy
model call topology
sample ordering policy
seed
```

not the final binary masks.

---

# 26. Dropout RNG audit

Recommended record:

```text
run seed
RNG initialization policy
model initialization fingerprint
DataLoader generator policy
dropout-site call topology fingerprint
```

Do not attempt to store every dropout mask from official training.

---

# 27. No Monte Carlo Dropout

Validation and final downstream inference are deterministic eval-mode inference.

Do not use:

```text
MC Dropout
multiple stochastic inference passes
predictive uncertainty via train-mode dropout
```

in S10.

That is a separate uncertainty-estimation experiment.

---

# 28. Attention-weight dropout nuance

If MHA dropout is part of S10 scope:

```text
training-time attention weights may be dropout-affected.
```

Therefore train-mode attention row sums are not a valid probability-normalization check under dropout.

This is consistent with `ATTENTION_VERIFY-v1`.

Do not fail S10 because train-mode returned attention weights do not sum exactly to one.

---

# 29. No scientific attention extraction in S10

Even though MHA dropout may be swept:

```text
need_weights = False
```

remains the standard training/Validation path.

No attention heatmaps/head interpretation in Phase 32.

Final attention analysis remains Phase 52–57.

---

# 30. Final attention interpretation uses eval mode

Later attention analysis will use final locked checkpoint in eval mode.

Thus selected dropout affects:

```text
learned parameters
```

but dropout itself is disabled during standard final attention extraction.

Carry final selected dropout as provenance.

---

# 31. Working hypotheses

## H-S10-01 — Moderate dropout may improve generalization

A plausible hypothesis:

```text
0.1 may regularize too little
0.3 may regularize too strongly
0.2 may provide a balance
```

but this remains:

```text
UNTESTED.
```

No candidate receives prior preference.

---

# 32. H-S10-02 — Low dropout may remain best after selected WD

Because Phase 31 already selects weight decay, additional dropout regularization may be unnecessary.

If:

\[
RMSE(DR01)<RMSE(DR02),RMSE(DR03)
\]

then the lowest registered dropout performs best under `WD*`.

Do not conclude dropout is generally unnecessary.

---

# 33. H-S10-03 — Strong dropout may help

If:

\[
RMSE(DR03)<RMSE(DR01),RMSE(DR02)
\]

the strongest registered probability performs best under the current model.

Since DR03 is upper boundary:

```text
record BOUNDARY_WINNER.
```

Do not automatically test 0.4/0.5.

---

# 34. Preconditions bắt buộc

Phase 32 chỉ bắt đầu khi:

```text
Phase 31 = PASS
```

or `PASS_WITH_WARNING` with no unresolved critical issue.

Required:

```text
approved_for_phase32 = true
```

and:

```text
s9_weight_decay_winner.json
s9_reference_update.json
phase_31_signoff.json
```

---

# 35. Upstream contracts bắt buộc

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
```

---

# 36. Carry-forward warnings

Any unresolved non-critical upstream warnings must be propagated.

Examples:

```text
RANDOM_CONTROL_GAIN
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
BOUNDARY_WINNER
MATCHED_INITIALIZATION_NOT_VERIFIABLE
SAMPLE_ORDER_NOT_VERIFIABLE
OPTIMIZER_SCOPE_WARNING
```

Propagate into:

```text
S10 manifest
S10 summary
S10 report
S10 winner
Phase 33 handoff.
```

---

# 37. Swept factor duy nhất

Canonical field:

```text
dropout
```

Allowed:

```text
0.1
0.2
0.3
```

Aliases:

```text
DR01
DR02
DR03
```

---

# 38. Frozen data configuration

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

# 39. Frozen representation configuration

```text
pooling_id = P*
activation_id = A*
```

---

# 40. Frozen batch configuration

```text
train_batch_size = B*
gradient_accumulation = 1
drop_last = False
```

Same DataLoader/sampler/worker/device-loading policy.

---

# 41. Frozen optimizer configuration

Hard:

```text
optimizer = AdamW
learning_rate = LR*
weight_decay = WD*
```

and all non-LR/WD fields remain exactly as frozen upstream.

Optimizer parameter-group topology/scope from Phase 31 remains unchanged.

---

# 42. Frozen Transformer capacity

```text
d_model = 64
num_heads = 4
num_layers = 2
ffn_dim = 128
```

No capacity change in S10.

---

# 43. Frozen remaining architecture

```text
input_size = same F
activation = A*
pooling = P*
sinusoidal positional encoding
POST_NORM
norm_first = False
no causal mask
no padding mask
regression head = Linear(64,1)
no output activation
```

---

# 44. Frozen loss

```text
MSE
```

No explicit L2 loss.

---

# 45. Frozen epoch/early-stop budget

```text
max_epochs = 50
patience = 10
min_delta = 0
```

No extra epochs for DR03 if it learns more slowly.

---

# 46. Frozen clipping

```text
gradient clipping ON
max_norm = 1.0
```

---

# 47. Frozen scheduler/warmup policy

```text
scheduler = None
warmup = None
```

---

# 48. Frozen seed

```text
seed = 42
```

for all candidates.

---

# 49. Same parameter count/schema

Dropout modules are parameter-free in the current design.

Hard expected:

```text
trainable_parameters_DR01
=
trainable_parameters_DR02
=
trainable_parameters_DR03
```

and same:

```text
parameter names
parameter shapes
state_dict key set
trainable parameter schema.
```

---

# 50. Full semantic config fingerprint may differ

Expected semantic model config differs by:

```text
dropout
```

while trainable parameter schema remains equal.

Store both:

```text
model_config_fingerprint
parameter_schema_fingerprint
dropout_scope_fingerprint
```

---

# 51. State dict does not substitute for dropout config provenance

Dropout probability is configuration, not a learned tensor parameter.

Therefore a state dict alone is insufficient to prove which dropout probability was used during training.

Every checkpoint/run must preserve:

```text
dropout ID/value
dropout-scope fingerprint
model config fingerprint
```

---

# 52. BEST checkpoint must be re-instantiated with correct dropout config

Even though Validation runs in eval mode, checkpoint verification must instantiate:

```text
fresh model with the candidate's correct dropout config
```

before strict state-dict load.

Do not load DR03 weights into a DR01-config model and call the provenance equivalent merely because eval predictions happen to match.

---

# 53. Existing DR01 reference reuse

S9 winner is already trained with:

```text
dropout = 0.1.
```

If exact S10 contract match:

```text
REUSE S9 winner.
```

Do not retrain DR01.

---

# 54. Normal S10 run count

Expected:

```text
DR01 = 0.1
→ reused S9 winner

DR02 = 0.2
→ NEW run

DR03 = 0.3
→ NEW run
```

Therefore normally:

```text
2 new training runs
+
1 reused reference.
```

---

# 55. DR01 reference reuse gate

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
H1
WB0
WINDOWPOP-v1

D64/H4/N2/FFN128
dropout=0.1
dropout-site topology/scope
sinusoidal PE
POST_NORM

AdamW
selected LR*
selected WD*
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

# 56. Fresh DR02 run

Execution:

```text
reseed 42
↓
fresh Train loader
↓
fresh Validation loader
↓
fresh Transformer(dropout=0.2)
↓
fresh AdamW with LR*/WD*
↓
MSE
↓
TRAINING_ENGINE-v1
```

No warm-start.

---

# 57. Fresh DR03 run

Execution:

```text
reseed 42
↓
fresh Train loader
↓
fresh Validation loader
↓
fresh Transformer(dropout=0.3)
↓
fresh AdamW with LR*/WD*
↓
MSE
↓
TRAINING_ENGINE-v1
```

No warm-start.

---

# 58. No model-state reuse

Never initialize DR02/DR03 from:

```text
DR01 BEST
another dropout candidate.
```

---

# 59. No optimizer-state reuse

Every new run uses fresh:

```text
AdamW state
moment estimates
step counters.
```

---

# 60. Run-order independence

Each new condition must independently execute:

```text
reseed
fresh loaders
fresh model
fresh criterion
fresh optimizer
fresh training state
```

so running DR02 before DR03 or vice versa does not inherit mutable state.

---

# 61. Same initialization audit

Dropout p does not alter parameter shapes or initialization.

For new DR02/DR03:

```text
initial_state_fingerprint_DR02
==
initial_state_fingerprint_DR03
```

should be verified under same environment/model builder.

If DR01 reference has initial-state fingerprint:

```text
compare all three.
```

Else:

```text
DR01_INIT_MATCH = NOT_VERIFIABLE.
```

---

# 62. Same sample-order audit

New DR02/DR03 should use identical Train sample permutation/batch grouping at corresponding epochs.

For DR01 reference:

```text
verify if provenance exists
else NOT_VERIFIABLE.
```

Do not retrain DR01 just to recover missing order metadata.

---

# 63. Same RNG startup policy

New candidates:

```text
seed42
same DataLoader RNG initialization
same model initialization sequence
same worker policy
```

must match.

Do not manually synchronize dropout masks across candidates.

---

# 64. Dropout-site runtime audit

Before official run starts, inspect model and record every registered site.

Fields should include:

```text
dropout site path
site type
candidate p
controlled_by_global_dropout
train/eval behavior expectation
```

All global-dropout sites must show candidate p.

---

# 65. Runtime site value must remain constant

No dropout schedule.

For each run, configured site values remain:

```text
0.1
0.2
or 0.3
```

from start to stop.

No epoch-dependent p changes.

---

# 66. No dropout warmup

Do not start at:

```text
p=0
```

and ramp to p=0.3.

That is a dropout schedule, not S10.

---

# 67. No DropPath/Stochastic Depth

Do not introduce:

```text
DropPath
stochastic depth
layer drop
```

inside S10.

These are different regularization mechanisms.

---

# 68. No attention-dropout-only experiment

Do not set:

```text
MHA dropout = 0.3
residual/FFN dropout = 0.1
```

unless that exact multi-field behavior was already what the single global dropout config means, which is not the intended S10 contract.

S10 tests the frozen global dropout field.

---

# 69. No residual-dropout-only experiment

Same rule.

---

# 70. No input-dropout-only experiment

Same rule.

---

# 71. No head-dropout experiment

Same rule.

---

# 72. No dropout-site ablation

Do not turn off individual sites to study importance.

That would be a separate ablation.

---

# 73. Dropout implementation unit tests

Before official new runs, test the centralized model/config path:

```text
DR01 builds expected site p=0.1
DR02 builds expected site p=0.2
DR03 builds expected site p=0.3
unknown dropout ID rejected
p<0 rejected
p>=1 rejected
```

according to model config validation.

---

# 74. Standard Dropout module train/eval sanity

For each registered p, on a large all-ones synthetic tensor using an isolated standard dropout module:

Train mode should show:

```text
some values zeroed
retained values scaled under standard inverted-dropout semantics
```

and empirical zero fraction should be broadly consistent with p.

Eval mode should behave as identity.

This is a semantic diagnostic, not a scientific model metric.

---

# 75. Avoid ReLU-zero confounding in dropout tests

Do not estimate dropout rate by counting zeros after:

```text
ReLU + Dropout
```

because ReLU itself creates zeros.

Use isolated dropout on nonzero synthetic input if checking mask behavior.

---

# 76. No hard universal empirical-zero tolerance without declaration

Random zero fraction fluctuates.

If an empirical p diagnostic is used:

```text
predeclare tensor size
seed
tolerance/statistical rule
```

and label it diagnostic.

Configuration inspection remains the authoritative p check.

---

# 77. Eval identity test

For isolated standard dropout module and finite tensor `x`:

```text
dropout.eval()
dropout(x)
```

should reproduce `x` under the module's evaluation semantics.

This test confirms mode control.

---

# 78. Model-level eval repeatability

For the same checkpoint/input in eval mode:

```text
prediction_1
prediction_2
...
```

should be allclose.

This confirms no unintended active dropout during Validation.

---

# 79. Model-level train stochasticity diagnostic

On a disposable model with p>0:

```text
model.train()
```

multiple forward passes on same bounded input may differ.

Use diagnostic summary over multiple calls rather than asserting every pair differs.

---

# 80. Training Engine mode-transition audit

Create an explicit audit proving:

```text
training batches:
model.training == True

Validation:
model.training == False

after Validation before next Train batch:
model.training == True
```

This is core S10 infrastructure validation.

---

# 81. Validation prediction repeatability from BEST checkpoint

For each candidate BEST:

```text
full Validation pass A
full Validation pass B
```

in eval mode should yield same ordered IDs and allclose predictions.

One repeated full pass may be optional if expensive; at minimum use a fixed Validation subset plus BEST metric re-verification.

---

# 82. No stochastic Validation averaging

Do not perform:

```text
5 eval passes and average prediction
```

because standard eval dropout is off and one deterministic pass is the contract.

---

# 83. No stochastic train-mode Validation

Hard fail.

---

# 84. Primary S10 selection metric

Hard:

```text
best_validation_rmse_wh
```

from verified BEST checkpoint.

---

# 85. Secondary metrics

Record:

```text
validation_mae_wh
validation_r2
best_epoch
epochs_completed
stop_reason
sample-weighted train loss
Validation RMSE trajectory
gradient norm
clipping fraction
runtime
optimizer steps
dropout-mode diagnostics
optional Train-vs-Validation gap
```

---

# 86. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{DR01},
RMSE_{DR02},
RMSE_{DR03}
\right)
\]

using full-precision Validation RMSE Wh.

---

# 87. Exact RMSE tie rule

If exact full-precision tie among candidate minima:

```text
prefer lower dropout.
```

Tie order:

```text
0.1
<
0.2
<
0.3
```

Rationale:

```text
less stochastic regularization
closer to current baseline
predeclared parsimony rule
```

Only exact ties invoke this rule.

---

# 88. No arbitrary practical-improvement threshold

Do not require:

```text
DR02 must improve >1%
```

Strict lower RMSE wins unless exact tie.

---

# 89. Pairwise effects

Required:

```text
DR01 vs DR02
DR02 vs DR03
DR01 vs DR03
```

For left→right:

\[
\Delta RMSE
=
RMSE_{left}-RMSE_{right}
\]

Positive means right candidate improves.

---

# 90. Relative effect

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

# 91. Trend classification

With three ordered p values, classify descriptively:

```text
LOW_DROPOUT_BEST
MODERATE_DROPOUT_BEST
HIGH_DROPOUT_BEST
MONOTONIC_GAIN_WITH_DROPOUT
MONOTONIC_DEGRADATION_WITH_DROPOUT
U_SHAPED
INVERTED_U
MIXED
EXACT_TIE_PATTERN
```

Do not fit a continuous curve to infer an untested optimum.

---

# 92. Boundary-winner caution

Both:

```text
DR01 = lower tested boundary
DR03 = upper tested boundary
```

If either wins:

```text
winner_is_boundary = true
```

Record limitation.

Do not automatically test:

```text
0
0.4
0.5
```

inside S10.

---

# 93. Metric ranking divergence

If:

```text
DR02 wins RMSE
DR01 wins MAE
```

record:

```text
METRIC_RANKING_DIVERGENCE.
```

Winner remains RMSE-based.

---

# 94. Best-checkpoint verification for DR02

After training:

```text
fresh Transformer(dropout=0.2)
↓
verify dropout-scope/config fingerprint
↓
strict-load BEST state_dict
↓
model.eval()
↓
full ordered Validation
↓
inverse target transform if YS1
↓
METRICS-v1
↓
verify recorded BEST metrics.
```

---

# 95. Best-checkpoint verification for DR03

Same with:

```text
dropout=0.3.
```

---

# 96. DR01 reference verification

No retraining.

Verify:

```text
BEST already verified
same sample population
same FV*/YS*/L*/P*/A*/B*/LR*/WD*
same dropout scope
dropout=0.1
same metric version
same Training Engine provenance.
```

---

# 97. Dropout configuration must be stored outside state_dict

Because dropout p is not learned state, official run/checkpoint metadata must contain:

```text
dropout ID
dropout value
dropout scope fingerprint
model config fingerprint
```

A bare `.pt` state dict is insufficient provenance.

---

# 98. Checkpoint resume semantics

If a DR02/DR03 run resumes from interruption:

```text
resume same run
same dropout value
same scope fingerprint
same RNG state where supported
same optimizer state
same loader generator state
```

No resume checkpoint from another dropout candidate.

---

# 99. Training stochasticity is intended, not noise to “remove”

Dropout intentionally introduces training stochasticity.

Do not disable dropout temporarily because training curves look noisy.

The registered p must remain active throughout train-mode batches.

---

# 100. Gradient diagnostics

Record:

```text
mean preclip grad norm
max preclip grad norm
fraction batches clipped
epochs with any clipping
nonfinite events
```

Dropout can alter learned trajectory and gradient distribution.

Interpret association only.

---

# 101. Clipping fraction is preferred normalized measure

Same batch size means raw clipped counts are more comparable than in S7, but:

```text
fraction_batches_clipped
```

remains the preferred summary.

---

# 102. Training-loss comparability

All candidates use:

```text
same target scaling
same MSE
same Train population
same batch size
```

so sample-weighted train loss is directly comparable descriptively.

Winner remains Validation RMSE Wh.

---

# 103. Generalization-gap diagnostics

Because dropout is a regularizer, optional Train-vs-Validation diagnostics are useful.

Preferred if available:

```text
Train RMSE Wh at BEST
Validation RMSE Wh at BEST
```

and:

\[
Gap_{RMSE}
=
ValidationRMSE-TrainRMSE
\]

Diagnostic only.

Do not select winner by smallest gap.

---

# 104. Why smallest generalization gap is not automatically best

A heavily underfit model can have:

```text
small Train–Validation gap
```

but poor Validation RMSE.

Primary metric remains actual forecasting performance.

---

# 105. Underfitting signal

Possible descriptive signal:

```text
high train loss
high Validation RMSE
small gap
```

for stronger dropout.

Do not define a universal threshold without pre-registration.

---

# 106. Overfitting signal

Possible descriptive signal:

```text
low train error
worse Validation error
larger gap
```

for lower dropout.

Again, descriptive under current run only.

---

# 107. No Test-based regularization diagnosis

Overfitting/generalization diagnostics use:

```text
Train
Validation
```

only.

Test remains locked.

---

# 108. Convergence diagnostics

Create summaries for:

```text
first epoch RMSE
best epoch
best RMSE
last RMSE
best-to-last gap
early stop
epoch cap
post-best worsening epochs
```

---

# 109. Stronger dropout may converge differently

If DR03 reaches best later:

```text
record it.
```

Do not give DR03 additional epochs beyond E50.

Phase 38 later handles epoch-cap sensitivity.

---

# 110. Runtime diagnostics

Record:

```text
epochs completed
total runtime
mean epoch time
median epoch time
time to best optional
```

Per-epoch compute shape is the same across p values.

Differences may reflect backend/system noise.

Runtime is not winner criterion.

---

# 111. Theoretical dropout context

Create simple context fields:

```text
drop_probability = p
keep_probability = 1-p
standard_inverted_scale = 1/(1-p)
```

For candidates:

```text
DR01:
q = 0.9
scale ≈ 1.111111

DR02:
q = 0.8
scale = 1.25

DR03:
q = 0.7
scale ≈ 1.428571
```

Label:

```text
STANDARD_DROPOUT_CONTEXT_ONLY
```

Do not treat this as a global model scaling factor.

---

# 112. No multiplication of keep probabilities across sites

Do not compute:

```text
q_global = q^number_of_dropout_sites
```

and interpret it as probability an input survives the network.

Dropout is applied to different transformed tensors/branches and such a simple product is not a valid model-wide interpretation.

---

# 113. No claim that p=0.3 means 30% of information is lost

Dropout zeros elements at particular training-time tensors.

The network, residual paths, attention mixing and inverted scaling make “information lost = 30%” an invalid simplification.

---

# 114. Model parameter norms optional

S9 already focused on parameter norms.

S10 may reuse parameter-norm diagnostics if readily available, but they are not core outputs.

Do not turn S10 into another WD analysis phase.

---

# 115. No dropout-mask artifact storage

Do not save massive per-batch dropout masks.

Not needed for coursework reproducibility.

Store:

```text
seed policy
dropout config/scope
RNG provenance
```

instead.

---

# 116. No deterministic-algorithm hack that disables intended dropout

Deterministic/reproducibility settings may constrain kernels but must not disable dropout itself during training.

Dropout stochasticity under seeded RNG is part of the intended model.

---

# 117. Device/backend reproducibility caution

Exact stochastic sequences may differ across:

```text
PyTorch versions
devices
backends
```

even with same seed.

Do not claim universal bitwise reproducibility.

Use environment fingerprint and within-environment reproducibility policy.

---

# 118. Same device preferred

New DR02/DR03 runs should use same backend/device as current development environment.

If DR01 reference was produced under another valid environment:

```text
runtime comparisons may be limited
```

while metric provenance remains governed by project contract.

---

# 119. Experiment identity

```text
sweep_id = S10_DROPOUT
experiment_family = TRANSFORMER_SWEEP_S10_DROPOUT
sweep_version = SWEEP_S10_DROPOUT-v1
```

Recommended labels:

```text
S10_<FV*>__<YS*>__<L*>__<P*>__<A*>__<B*>__<LR*>__<WD*>__DR01_0P1__REFERENCE_S9

S10_<FV*>__<YS*>__<L*>__<P*>__<A*>__<B*>__<LR*>__<WD*>__DR02_0P2__S42

S10_<FV*>__<YS*>__<L*>__<P*>__<A*>__<B*>__<LR*>__<WD*>__DR03_0P3__S42
```

Registry `run_id` remains authoritative.

---

# 120. Run matrix

Create:

```text
s10_run_matrix.csv
```

Fields:

```text
sweep_id
dropout_id
dropout_probability
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
dropout_scope_fingerprint
population_fingerprint
feature_fingerprint
seed
model_config_id
optimizer_config_id
training_config_id
status
```

---

# 121. Sweep manifest

Create:

```text
s10_dropout_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S10_DROPOUT-v1
sweep_id = S10_DROPOUT
source_s9_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
candidate_dropouts = [0.1,0.2,0.3]
new_runs_required
reused_runs
swept_field = dropout
dropout_scope_fingerprint
dropout_site_count
training_mode_required = true
validation_eval_mode_required = true
mc_dropout = false
budget_semantics = FIXED_EPOCH_BUDGET
frozen_fields
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = LOWER_DROPOUT_ON_EXACT_RMSE_TIE
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

# 122. Sweep contract

Create:

```text
s10_dropout_sweep_contract.json
```

Must state:

```text
Only frozen global Transformer dropout probability changes.

DR01=0.1.
DR02=0.2.
DR03=0.3.

Dropout sites are frozen before training.
No site is added/removed.
No input/head/pooling dropout is introduced.
MHA dropout follows the frozen global-dropout scope if wired there by Transformer-v1.

model.train() for Train.
model.eval() for Validation.
No MC Dropout.

Same sample population.
Same model parameter schema.
Same initialization/sample-order policy.
Same batch/LR/WD/optimizer/loss.
Same E50/patience10/clip1.

DR01 reference reused if exact match.
DR02/DR03 fresh seed-42 runs.

Validation RMSE Wh selects winner.
Exact tie → lower dropout.
Test forbidden.
```

---

# 123. Preflight audit

Create:

```text
s10_dropout_preflight_audit.csv
```

Checks:

```text
phase31_pass
approved_for_phase32
s9_winner_valid
FV_locked
YS_locked
L_locked
P_locked
A_locked
B_locked
LR_locked
WD_locked
DR01_registered
DR02_registered
DR03_registered
dropout_scope_loaded
dropout_scope_fingerprint_valid
no_new_dropout_site
no_removed_dropout_site
train_eval_semantics_ready
population_fixed
Training_Engine_fixed
Metric_fixed
seed_fixed
test_locked
status
```

---

# 124. Dropout-site registry

Create:

```text
s10_dropout_site_registry.csv
```

Fields:

```text
site_id
module_path
module_type
semantic_location
layer_index
controlled_by_global_dropout
reference_p
expected_candidate_p_source
train_active
eval_active
parameter_count
status
```

Expected:

```text
eval_active = false
```

for standard dropout sites.

For MHA internal attention-weight dropout, represent semantic behavior appropriately.

---

# 125. Dropout-scope manifest

Create:

```text
s10_dropout_scope_manifest.json
```

Fields:

```text
scope_version
source_transformer_impl_version
source_reference_run_id
site_count
sites
scope_fingerprint
input_dropout_in_scope
head_dropout_in_scope
pooling_dropout_in_scope
mha_attention_weight_dropout_in_scope
scope_frozen = true
status
```

---

# 126. Dropout definition audit

Create:

```text
s10_dropout_definition_audit.csv
```

Fields:

```text
dropout_id
registered_p
keep_probability
standard_inverted_scale
runtime_site_count
all_global_sites_match_p
no_unregistered_site_change
valid_probability_range
status
```

---

# 127. Train/eval semantics audit

Create:

```text
s10_dropout_mode_audit.csv
```

Fields:

```text
dropout_id
context
model_training_flag
dropout_expected_active
prediction_repeatability_check
max_abs_repeat_diff
status
```

Contexts:

```text
TRAIN_SANITY
VALIDATION_SANITY
BEST_CHECKPOINT_EVAL
```

---

# 128. Dropout unit tests artifact

Create:

```text
s10_dropout_unit_tests.csv
```

Recommended tests:

```text
DR01 resolves 0.1
DR02 resolves 0.2
DR03 resolves 0.3
unknown dropout ID rejected
p<0 rejected
p>=1 rejected
scope fingerprint stable
site count stable
all controlled sites receive candidate p
non-controlled sites unchanged
no input dropout added
no head dropout added
no pooling dropout added
standard dropout train behavior valid
standard dropout eval identity valid
model eval repeated forward allclose
model train stochastic path active
B1 output shape valid
B32/B64-compatible shape valid
synthetic backward finite
parameter count unchanged
state_dict keys unchanged
```

---

# 129. Common-data audit

Create:

```text
s10_common_data_audit.csv
```

Fields:

```text
split_id
sample_count_dr01
sample_count_dr02
sample_count_dr03
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
weight_decay_equal
population_fingerprint_equal
status
```

---

# 130. Architecture audit

Create:

```text
s10_dropout_architecture_audit.csv
```

Fields:

```text
dropout_id
input_size
lookback
pooling
activation
d_model
num_heads
num_layers
ffn_dim
dropout
dropout_scope_fingerprint
norm_policy
pe_policy
regression_head
trainable_parameters
parameter_schema_fingerprint
state_dict_key_fingerprint
only_dropout_differs
status
```

---

# 131. Config-delta audit

Create:

```text
s10_config_delta_audit.csv
```

Allowed scientific differences:

```text
dropout_id
dropout probability at frozen controlled sites
model config fingerprint as consequence
```

Everything else must match.

---

# 132. Training-config audit

Create:

```text
s10_dropout_training_audit.csv
```

Fields:

```text
dropout_id
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
only_dropout_differs
status
```

---

# 133. Initialization audit

Create:

```text
s10_initialization_audit.csv
```

Fields:

```text
dropout_id
initial_model_state_fingerprint
reference_available
parameter_schema_match
fingerprint_matches_reference
fingerprint_matches_other_new_run
seed
status
```

---

# 134. Sample-order audit

Create:

```text
s10_sample_order_audit.csv
```

Fields:

```text
epoch_or_probe
dr01_order_fingerprint
dr02_order_fingerprint
dr03_order_fingerprint
dr01_reference_available
dr02_dr03_exact_match
all_three_match_if_verifiable
status
```

---

# 135. RNG-policy audit

Create:

```text
s10_rng_policy_audit.json
```

Fields:

```text
seed
environment_fingerprint
python_rng_policy
numpy_rng_policy
torch_rng_policy
dataloader_generator_policy
worker_seed_policy
dropout_seed_reset_per_batch = false
dropout_masks_persisted = false
cross_candidate_mask_identity_required = false
status
```

---

# 136. Training Engine mode-transition audit

Create:

```text
s10_training_mode_transition_audit.csv
```

Fields:

```text
dropout_id
epoch
stage
expected_model_training
observed_model_training
status
```

Stages can be summarized:

```text
TRAIN_START
VALIDATION_START
POST_VALIDATION_TRAIN_RESTORE
```

Do not add intrusive per-batch logging if it materially slows training; strategic probes are sufficient.

---

# 137. Runtime dropout config audit

Create:

```text
s10_dropout_runtime_audit.csv
```

Fields:

```text
dropout_id
site_id
runtime_p_start
runtime_p_end
min_runtime_p
max_runtime_p
unique_p_count
scope_membership_stable
constant_dropout_verified
status
```

Expected:

```text
unique_p_count = 1
```

at each controlled site.

---

# 138. Eval repeatability audit

Create:

```text
s10_eval_repeatability_audit.csv
```

Fields:

```text
dropout_id
checkpoint_run_id
sample_set_fingerprint
repeat_count
max_abs_prediction_diff_wh
rmse_diff
mae_diff
r2_diff
rtol
atol
allclose
status
```

Use Validation or a fixed Validation subset.

No Test.

---

# 139. Optimizer-budget audit

Create:

```text
s10_optimizer_budget_audit.csv
```

Fields:

```text
dropout_id
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

Batch is fixed, so steps per completed epoch should match.

---

# 140. Run provenance

Create:

```text
s10_dropout_run_provenance.csv
```

Fields:

```text
dropout_id
dropout_probability
run_id
source_type
source_phase
config_fingerprint
dropout_scope_fingerprint
parameter_schema_fingerprint
feature_fingerprint
population_fingerprint
initial_state_fingerprint
sample_order_provenance
rng_policy_id_or_fingerprint
best_checkpoint_sha256
history_sha256
metric_artifact
prediction_artifact
status
```

---

# 141. Primary metrics table

Create:

```text
s10_dropout_metrics.csv
```

Rows:

```text
DR01
DR02
DR03
```

Fields:

```text
dropout_id
dropout_probability
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

# 142. Pairwise effects table

Create:

```text
s10_dropout_pairwise_effects.csv
```

Rows:

```text
DR01_TO_DR02
DR02_TO_DR03
DR01_TO_DR03
```

Fields:

```text
left_dropout_id
right_dropout_id
left_p
right_p
left_rmse_wh
right_rmse_wh
rmse_delta_wh
rmse_improvement_pct
mae_delta_wh
r2_delta
p_difference
status
```

---

# 143. Dropout trend diagnostics

Create:

```text
s10_dropout_trend_diagnostics.json
```

Fields:

```text
rmse_dr01
rmse_dr02
rmse_dr03
pattern
best_dropout_id
best_dropout_probability
is_boundary_winner
interpretation
status
```

---

# 144. Optimization diagnostics

Create:

```text
s10_optimization_diagnostics.csv
```

Fields:

```text
dropout_id
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

# 145. Convergence diagnostics

Create:

```text
s10_convergence_diagnostics.csv
```

Fields:

```text
dropout_id
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

# 146. Runtime diagnostics

Create:

```text
s10_runtime_diagnostics.csv
```

Fields:

```text
dropout_id
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

# 147. Standard dropout context artifact

Create:

```text
s10_dropout_theoretical_context.csv
```

Fields:

```text
dropout_id
p
keep_probability
standard_inverted_scale
context_only
not_global_effective_dropout
status
```

Every row:

```text
context_only = true
not_global_effective_dropout = true
```

---

# 148. Optional generalization diagnostics

If full Train BEST-checkpoint inference is already supported and affordable:

```text
s10_generalization_diagnostics.csv
```

Fields:

```text
dropout_id
train_rmse_wh_at_best
validation_rmse_wh_at_best
rmse_gap_wh
train_mae_wh_at_best
validation_mae_wh_at_best
mae_gap_wh
status
```

Optional only.

Do not access Test.

---

# 149. Hypothesis outcomes

Create:

```text
s10_hypothesis_outcomes.csv
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

within single-seed Validation context.

---

# 150. Findings artifact

Create:

```text
s10_dropout_findings.csv
```

Possible codes:

```text
DR01_GAIN
DR02_GAIN
DR03_GAIN
DROPOUT_EXACT_TIE
LOW_DROPOUT_BEST
MODERATE_DROPOUT_BEST
HIGH_DROPOUT_BEST
BOUNDARY_WINNER
METRIC_RANKING_DIVERGENCE
TRAIN_EVAL_MODE_VERIFIED
EVAL_REPEATABILITY_VERIFIED
DROPOUT_SCOPE_VERIFIED
CONSTANT_DROPOUT_VERIFIED
CONVERGENCE_DIFFERENCE
CLIPPING_DIFFERENCE
GENERALIZATION_GAP_DIFFERENCE
POSSIBLE_UNDERFITTING_SIGNAL
POSSIBLE_OVERFITTING_SIGNAL
MATCHED_INITIALIZATION_VERIFIED
MATCHED_INITIALIZATION_NOT_VERIFIABLE
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
INHERITED_WARNING
```

---

# 151. Winner artifact

Create:

```text
s10_dropout_winner.json
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
weight_decay
selection_metric
selection_direction
tie_rule
winner_dropout_id
winner_dropout_probability
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_dropout_id
runner_up_dropout_probability
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
winner_is_boundary
dropout_scope_fingerprint
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 152. Reference update for Phase 33

Create:

```text
s10_reference_update.json
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
weight_decay
previous_dropout = 0.1
selected_dropout_id
selected_dropout_probability
winner_run_id
winner_config_fingerprint
winner_dropout_scope_fingerprint
winner_rmse_wh
d_model_state = D64
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase33
```

---

# 153. Phase 33 handoff logic

Phase 33 tests:

```text
D32 = d_model 32
D64 = d_model 64
```

while holding S10-selected dropout fixed.

S10 winner already uses:

```text
D64
```

because d_model has not yet been swept.

Therefore normally Phase 33 should:

```text
reuse S10 winner as D64 reference
train only D32
```

if exact match.

---

# 154. Learning curves

Recommended figures:

```text
S10_01_validation_rmse_by_epoch.png
S10_02_validation_mae_by_epoch.png
S10_03_train_loss_by_epoch.png
S10_04_gradient_norm_by_epoch.png
S10_05_gradient_clipping_fraction.png
S10_06_best_validation_metrics.png
S10_07_best_to_last_rmse.png
S10_08_generalization_gap_optional.png
S10_09_dropout_context.png
```

---

# 155. Primary figure

```text
S10_01_validation_rmse_by_epoch.png
```

Overlay:

```text
DR01
DR02
DR03
```

with BEST epoch markers.

---

# 156. Best metric figure

```text
S10_06_best_validation_metrics.png
```

must be generated from:

```text
s10_dropout_metrics.csv
```

not manually typed values.

---

# 157. Generalization-gap figure

Optional:

```text
S10_08_generalization_gap_optional.png
```

only if Train BEST-checkpoint metrics are computed consistently.

Do not present gap alone without absolute Train/Validation RMSE.

---

# 158. Dropout-context figure

`S10_09_dropout_context.png` may display:

```text
p
keep probability
```

as theoretical context.

Do not display a fabricated “effective model dropout”.

---

# 159. Interpretation if DR01 wins

Safe:

> Under the S9-selected weight decay, selected learning rate and frozen Transformer configuration, dropout `0.1` achieved the lowest Validation RMSE among the registered candidates.

Because DR01 is the lower boundary:

```text
record boundary-winner limitation.
```

Do not automatically test dropout 0.

---

# 160. Interpretation if DR02 wins

Safe:

> Dropout `0.2` achieved the lowest Validation RMSE under the frozen current configuration, providing the best empirical tradeoff among the three registered probabilities.

Do not claim it is universally optimal.

---

# 161. Interpretation if DR03 wins

Safe:

> Dropout `0.3` achieved the lowest Validation RMSE among the registered S10 candidates under the selected WD/LR configuration.

Record:

```text
upper boundary winner.
```

No automatic 0.4/0.5 search.

---

# 162. Exact tie interpretation

If exact full-precision RMSE tie:

```text
lower dropout wins
```

by predeclared rule.

---

# 163. Tiny non-zero margin

Strict lower RMSE wins.

Document:

```text
small single-seed Validation margin.
```

Do not overstate.

---

# 164. Single-seed limitation

Mandatory:

```text
all S10 conditions use seed 42.
```

No mean±std.

---

# 165. Validation-only limitation

Mandatory:

```text
S10 winner is development selection based on Validation.
```

No Test evidence.

---

# 166. Dropout–WD interaction limitation

Weight decay `WD*` is already selected/frozen.

S10 does not explore:

```text
WD × Dropout
```

grid.

The winner is conditional on `WD*`.

---

# 167. Dropout–LR interaction limitation

Learning rate `LR*` is frozen from S8.

Different dropout probabilities may prefer different LRs, but S10 does not test that interaction.

---

# 168. Dropout–batch interaction limitation

Batch `B*` is frozen from S7.

No Batch×Dropout grid.

---

# 169. Dropout–capacity interaction limitation

Current capacity remains:

```text
D64
H4
N2
FFN128.
```

Later phases change:

```text
d_model
heads
layers
FFN width.
```

A dropout optimum can depend on capacity.

S10 winner is therefore local to current capacity.

---

# 170. Dropout–activation/pooling/lookback interaction limitation

These are already selected and frozen.

No factorial grid is explored.

---

# 171. Dropout–epoch-budget interaction limitation

Higher dropout may converge more slowly.

All candidates still use:

```text
E50/patience10.
```

Phase 38 later tests epoch cap on the then-current reference.

---

# 172. Sequential-selection limitation

By Phase 32, Validation has influenced:

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
```

Complete Registry/provenance and later robustness checks are mandatory.

---

# 173. No Test access

Hard:

```text
Test loader not iterated
Test metrics absent
Test predictions absent.
```

---

# 174. No dropout 0

Not a registered S10 option.

Do not test it after seeing DR01 win.

---

# 175. No dropout >0.3

Not registered.

---

# 176. No per-layer dropout values

Do not test:

```text
layer1=.1
layer2=.3
```

inside S10.

---

# 177. No separate attention/FFN dropout tuning

No.

---

# 178. No dropout schedule

No.

---

# 179. No MC Dropout

No.

---

# 180. No DropPath/Stochastic Depth

No.

---

# 181. No head/input/pooling dropout additions

No.

---

# 182. No LR/WD compensation

No.

---

# 183. No extra epochs

No.

---

# 184. No gradient clipping change

No.

---

# 185. No architecture change

No.

---

# 186. No attention extraction during training

No.

---

# 187. Run failure policy

If DR02 or DR03 has a technical failure unrelated to the scientific candidate:

```text
S10 incomplete
```

until technical rerun is resolved.

---

# 188. Technical rerun allowed

Only for documented:

```text
process interruption
hardware/software failure
corrupt checkpoint
artifact-write failure
```

under Experiment Registry rerun policy.

---

# 189. Score-based rerun forbidden

Do not rerun because:

```text
RMSE looks poor
curve looks noisy
dropout seems unlucky.
```

Training stochasticity is part of the registered run.

---

# 190. DR01 duplicate-run prohibition

Do not retrain DR01 to obtain a new stochastic draw.

Reuse exact S9 winner.

---

# 191. Genuine numerical instability

If a candidate produces NaN/Inf under correct implementation:

```text
verify data/config/mode/optimizer
```

first.

If genuine:

```text
record numerical failure
S10 remains incomplete under strict core protocol
```

until candidate-failure handling is formally resolved.

Do not silently rank only remaining conditions.

---

# 192. Discrepancy taxonomy

```text
S9_REFERENCE_MISSING
S9_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
POOLING_DRIFT
ACTIVATION_DRIFT
BATCH_DRIFT
LEARNING_RATE_DRIFT
WEIGHT_DECAY_DRIFT
DROPOUT_DEFINITION_MISMATCH
DROPOUT_SCOPE_MISMATCH
DROPOUT_SITE_ADDED
DROPOUT_SITE_REMOVED
DROPOUT_SITE_VALUE_MISMATCH
DROPOUT_RUNTIME_DRIFT
DROPOUT_SCHEDULE_PRESENT
INPUT_DROPOUT_ADDED
HEAD_DROPOUT_ADDED
POOLING_DROPOUT_ADDED
PER_LAYER_DROPOUT_DRIFT
ATTENTION_FFN_DROPOUT_SPLIT
TRAIN_MODE_DISABLED
VALIDATION_TRAIN_MODE
POST_VALIDATION_TRAIN_NOT_RESTORED
MC_DROPOUT_USED
DROPPATH_ADDED
RNG_RESET_PER_BATCH
PARAMETER_COUNT_MISMATCH
PARAMETER_SCHEMA_MISMATCH
POPULATION_MISMATCH
TARGET_ID_MISMATCH
X_SCALER_MISMATCH
TARGET_SCALER_MISMATCH
OPTIMIZER_DRIFT
EPOCH_BUDGET_DRIFT
GRADIENT_CLIP_DRIFT
SAMPLE_ORDER_POLICY_DRIFT
INITIALIZATION_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
REFERENCE_RUN_MISMATCH
RUN_FAILURE
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

# 193. Discrepancy log

Create:

```text
s10_dropout_discrepancies.json
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

# 194. Severity examples

```text
CRITICAL:
Validation in train mode
Test access
dropout site topology changes
MC Dropout used for scientific Validation
wrong population
per-batch RNG reset

MAJOR:
wrong p
new input/head dropout
checkpoint config mismatch
post-Validation train mode not restored
checkpoint verification failure

MODERATE:
boundary winner
metric ranking divergence
reference init/order not verifiable
large convergence difference

INFO:
minor runtime difference
theoretical dropout context only
```

---

# 195. Status model

## PASS

```text
all three dropout conditions valid
same data/model/optimizer except p
dropout scope frozen
train/eval mode semantics verified
DR01 reference valid
DR02/DR03 verified
constant dropout confirmed
winner selected
Phase 33 reference generated
Test untouched
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
boundary winner
metric ranking divergence
reference initialization/order not verifiable
possible under/overfitting signal
inherited warning
```

provided methodology remains valid.

## FAIL

Examples:

```text
Validation train mode
dropout site drift
hidden dropout schedule
new dropout site
wrong p
population mismatch
unresolved candidate failure
Test access.
```

---

# 196. Sweep summary

Create:

```text
s10_dropout_sweep_summary.json
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
weight_decay
reference_run_id
new_run_ids
reused_runs
candidate_dropouts
dropout_scope
train_eval_mode_status
primary_metric
metrics_by_dropout
pairwise_effects
trend_pattern
optimizer_budget
optimization_diagnostics
convergence_diagnostics
generalization_diagnostics_optional
eval_repeatability
constant_dropout_audit
initialization_match
sample_order_match
winner
winner_margin
winner_is_boundary
inherited_warnings
phase33_reference
test_status
overall_status
```

---

# 197. Human-readable report

Create:

```text
s10_dropout_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S9
3. DR01/DR02/DR03 definitions
4. Frozen dropout-site scope
5. Train/eval dropout semantics
6. Controlled-variable contract
7. Same data/initialization/order fairness
8. DR01 reference provenance
9. Runtime dropout/site verification
10. Validation metrics
11. Pairwise dropout effects
12. Learning-curve/convergence diagnostics
13. Gradient/clipping diagnostics
14. Optional Train-vs-Validation gap
15. Eval repeatability
16. Dropout trend/boundary analysis
17. S10 winner
18. Interpretation cautions
19. Interaction limitations
20. Phase 33 handoff
```

---

# 198. Report wording

Use:

```text
Observed
Interpretation
Limitation
Handoff
```

Safe example:

> Dropout 0.2 achieved the lowest Validation RMSE under the S9-selected weight decay, S8-selected learning rate, fixed capacity and deterministic eval-mode protocol.

Avoid:

> 20% of information should be dropped for this dataset.

That is not supported by S10.

---

# 199. README

Create:

```text
README_S10_DROPOUT_SWEEP.md
```

Must explain:

```text
Purpose
S9 winner handoff
DR01/DR02/DR03 definitions
dropout-site scope
MHA attention-weight dropout relationship
no new input/head/pooling dropout
training vs evaluation mode
no MC Dropout
same initialization/order goals
DR01 reference reuse
state_dict/config provenance nuance
constant dropout
generalization diagnostics
winner rule
tie rule
boundary-winner caution
dropout-WD/LR/capacity interactions
Phase 33 handoff
No Test.
```

---

# 200. Output directory

```text
artifacts/
└── sweeps/
    └── S10_dropout/
        ├── s10_dropout_sweep_manifest.json
        ├── s10_dropout_sweep_contract.json
        ├── s10_dropout_preflight_audit.csv
        ├── s10_run_matrix.csv
        ├── s10_dropout_site_registry.csv
        ├── s10_dropout_scope_manifest.json
        ├── s10_dropout_definition_audit.csv
        ├── s10_dropout_mode_audit.csv
        ├── s10_dropout_unit_tests.csv
        ├── s10_common_data_audit.csv
        ├── s10_dropout_architecture_audit.csv
        ├── s10_config_delta_audit.csv
        ├── s10_dropout_training_audit.csv
        ├── s10_initialization_audit.csv
        ├── s10_sample_order_audit.csv
        ├── s10_rng_policy_audit.json
        ├── s10_training_mode_transition_audit.csv
        ├── s10_dropout_runtime_audit.csv
        ├── s10_eval_repeatability_audit.csv
        ├── s10_optimizer_budget_audit.csv
        ├── s10_dropout_run_provenance.csv
        ├── s10_dropout_metrics.csv
        ├── s10_dropout_pairwise_effects.csv
        ├── s10_dropout_trend_diagnostics.json
        ├── s10_optimization_diagnostics.csv
        ├── s10_convergence_diagnostics.csv
        ├── s10_runtime_diagnostics.csv
        ├── s10_dropout_theoretical_context.csv
        ├── s10_generalization_diagnostics.csv
        ├── s10_hypothesis_outcomes.csv
        ├── s10_dropout_findings.csv
        ├── s10_dropout_winner.json
        ├── s10_reference_update.json
        ├── s10_dropout_sweep_tests.csv
        ├── s10_dropout_discrepancies.json
        ├── s10_dropout_sweep_summary.json
        ├── s10_dropout_sweep_report.md
        ├── figures/
        │   ├── S10_01_validation_rmse_by_epoch.png
        │   ├── S10_02_validation_mae_by_epoch.png
        │   ├── S10_03_train_loss_by_epoch.png
        │   ├── S10_04_gradient_norm_by_epoch.png
        │   ├── S10_05_gradient_clipping_fraction.png
        │   ├── S10_06_best_validation_metrics.png
        │   ├── S10_07_best_to_last_rmse.png
        │   ├── S10_08_generalization_gap_optional.png
        │   └── S10_09_dropout_context.png
        ├── README_S10_DROPOUT_SWEEP.md
        └── phase_32_signoff.json
```

Optional:

```text
s10_generalization_diagnostics.csv
S10_08_generalization_gap_optional.png
```

may be omitted if full Train BEST inference is not part of current low-cost diagnostics.

New DR02/DR03 run artifacts remain:

```text
artifacts/runs/<run_id>/
```

Do not duplicate checkpoints in sweep folder.

---

# 201. Required outputs

```text
O32.1  Sweep manifest
O32.2  Sweep contract
O32.3  Preflight audit
O32.4  Run matrix
O32.5  Dropout-site registry
O32.6  Dropout-scope manifest/fingerprint
O32.7  Dropout-definition audit
O32.8  Train/eval mode audit
O32.9  Dropout unit tests
O32.10 Common-data audit
O32.11 Architecture audit
O32.12 Config-delta audit
O32.13 Training-config audit
O32.14 Initialization audit
O32.15 Sample-order audit
O32.16 RNG-policy audit
O32.17 Training-mode transition audit
O32.18 Runtime dropout audit
O32.19 Eval repeatability audit
O32.20 Optimizer-budget audit
O32.21 Run provenance
O32.22 Reused DR01 reference
O32.23 Verified DR02 run
O32.24 Verified DR03 run
O32.25 Primary metrics table
O32.26 Pairwise effects table
O32.27 Dropout trend diagnostics
O32.28 Optimization diagnostics
O32.29 Convergence diagnostics
O32.30 Runtime diagnostics
O32.31 Theoretical dropout context
O32.32 Optional generalization diagnostics
O32.33 Hypothesis outcomes
O32.34 Findings
O32.35 Winner artifact
O32.36 Phase 33 reference update
O32.37 Figures
O32.38 Sweep test suite
O32.39 Discrepancy log
O32.40 Sweep summary
O32.41 Human-readable report
O32.42 README
O32.43 Phase sign-off
```

---

# 202. Sweep test suite

Create:

```text
s10_dropout_sweep_tests.csv
```

Recommended checks:

```text
S10T32-001 Phase 31 PASS/non-critical warning only
S10T32-002 approved_for_phase32 = true
S10T32-003 S9 winner valid
S10T32-004 feature variant fixed
S10T32-005 target scaling fixed
S10T32-006 lookback fixed
S10T32-007 pooling fixed
S10T32-008 activation fixed
S10T32-009 batch fixed
S10T32-010 selected LR fixed
S10T32-011 selected WD fixed
S10T32-012 DR01 registered = 0.1
S10T32-013 DR02 registered = 0.2
S10T32-014 DR03 registered = 0.3
S10T32-015 no extra dropout candidate
S10T32-016 dropout scope loaded from Transformer-v1/S9 reference
S10T32-017 dropout site registry generated
S10T32-018 dropout scope fingerprint generated
S10T32-019 site count same across candidates
S10T32-020 module paths same across candidates
S10T32-021 no controlled site removed
S10T32-022 no controlled site added
S10T32-023 all controlled DR01 sites use 0.1
S10T32-024 all controlled DR02 sites use 0.2
S10T32-025 all controlled DR03 sites use 0.3
S10T32-026 non-controlled dropout fields unchanged
S10T32-027 no new input dropout
S10T32-028 no new head dropout
S10T32-029 no new pooling dropout
S10T32-030 no DropPath/stochastic depth
S10T32-031 no per-layer dropout split
S10T32-032 no attention-only dropout sweep
S10T32-033 no FFN-only dropout sweep
S10T32-034 MHA dropout follows frozen scope
S10T32-035 standard Dropout train semantic sanity passes
S10T32-036 standard Dropout eval identity sanity passes
S10T32-037 empirical zero-rate diagnostic avoids ReLU confound
S10T32-038 unknown dropout ID rejected
S10T32-039 invalid p<0 rejected
S10T32-040 invalid p>=1 rejected
S10T32-041 same Train IDs
S10T32-042 same Validation IDs
S10T32-043 same WINDOWPOP-v1
S10T32-044 same feature fingerprint
S10T32-045 same X scaler
S10T32-046 same target transform/scaler
S10T32-047 same lookback
S10T32-048 same pooling
S10T32-049 same activation
S10T32-050 same batch
S10T32-051 same selected LR
S10T32-052 same selected WD
S10T32-053 same H1/WB0
S10T32-054 same D64
S10T32-055 same H4
S10T32-056 same N2
S10T32-057 same FFN128
S10T32-058 same positional encoding
S10T32-059 same POST_NORM
S10T32-060 no causal mask
S10T32-061 no padding mask
S10T32-062 same regression head
S10T32-063 same parameter count
S10T32-064 same parameter schema
S10T32-065 same state_dict key set
S10T32-066 same AdamW identity
S10T32-067 same optimizer parameter-group policy
S10T32-068 MSE fixed
S10T32-069 E50 fixed
S10T32-070 patience10 fixed
S10T32-071 min_delta fixed
S10T32-072 clip1 fixed
S10T32-073 scheduler=None
S10T32-074 warmup=None
S10T32-075 accumulation=1
S10T32-076 drop_last=False
S10T32-077 mixed precision policy fixed
S10T32-078 seed42 fixed
S10T32-079 no RNG reset per batch
S10T32-080 cross-candidate mask identity not required
S10T32-081 DR02/DR03 initial states match
S10T32-082 DR01 init match checked if available
S10T32-083 DR02/DR03 sample order matches
S10T32-084 DR01 order match checked if available
S10T32-085 DR01 reference exact-match
S10T32-086 DR01 reference reused
S10T32-087 DR02 registered before training
S10T32-088 DR03 registered before training
S10T32-089 DR02 fresh loaders/model/optimizer
S10T32-090 DR03 fresh loaders/model/optimizer
S10T32-091 no model warm-start
S10T32-092 no optimizer-state reuse
S10T32-093 DR02 trained via TRAINING_ENGINE-v1
S10T32-094 DR03 trained via TRAINING_ENGINE-v1
S10T32-095 Train stages use model.train()
S10T32-096 Validation stages use model.eval()
S10T32-097 post-Validation train mode restored
S10T32-098 no MC Dropout Validation
S10T32-099 no stochastic Validation averaging
S10T32-100 eval repeated predictions allclose
S10T32-101 DR02 BEST verified with dropout=0.2 config
S10T32-102 DR03 BEST verified with dropout=0.3 config
S10T32-103 DR01 BEST already verified
S10T32-104 checkpoint metadata contains dropout value
S10T32-105 checkpoint metadata contains dropout scope fingerprint
S10T32-106 state_dict alone not treated as config provenance
S10T32-107 runtime DR01 p constant from provenance
S10T32-108 runtime DR02 p constant
S10T32-109 runtime DR03 p constant
S10T32-110 no dropout schedule
S10T32-111 same steps per completed epoch
S10T32-112 optimizer-budget audit generated
S10T32-113 all result rows use verified BEST
S10T32-114 all predictions finite
S10T32-115 all official metrics in Wh
S10T32-116 full-precision RMSE available
S10T32-117 DR01→DR02 effect correct
S10T32-118 DR02→DR03 effect correct
S10T32-119 DR01→DR03 effect correct
S10T32-120 trend pattern generated
S10T32-121 winner=min RMSE
S10T32-122 exact tie→lower dropout
S10T32-123 no reference-priority override
S10T32-124 metric divergence recorded if present
S10T32-125 boundary winner flagged if DR01/DR03
S10T32-126 no automatic dropout-range extension
S10T32-127 gradient diagnostics generated
S10T32-128 clipping diagnostics generated
S10T32-129 convergence diagnostics generated
S10T32-130 runtime diagnostics generated
S10T32-131 theoretical keep-probability context generated
S10T32-132 theoretical context labeled non-selection
S10T32-133 no global effective-dropout product claimed
S10T32-134 optional generalization diagnostics use Train/Validation only
S10T32-135 smallest generalization gap not used as winner rule
S10T32-136 hypothesis outcomes generated
S10T32-137 findings generated
S10T32-138 winner points to valid run
S10T32-139 Phase 33 reference update generated
S10T32-140 D64 reuse identified for Phase 33
S10T32-141 inherited warnings propagated
S10T32-142 no dropout 0
S10T32-143 no dropout >0.3
S10T32-144 no DropPath
S10T32-145 no MC Dropout
S10T32-146 no LR compensation
S10T32-147 no WD compensation
S10T32-148 no extra epochs
S10T32-149 no score-based rerun
S10T32-150 no failed/SANITY run in winner ranking
S10T32-151 no scientific attention extraction
S10T32-152 no Test access
S10T32-153 single-seed limitation documented
S10T32-154 Validation-only limitation documented
S10T32-155 dropout-WD interaction documented
S10T32-156 dropout-LR interaction documented
S10T32-157 dropout-batch interaction documented
S10T32-158 dropout-capacity interaction documented
S10T32-159 dropout-budget interaction documented
S10T32-160 figures source-derived
S10T32-161 summary/report generated
S10T32-162 phase sign-off generated
```

---

# 203. Recommended notebook structure

```text
Cell 32.1  Phase title
Cell 32.2  Verify Phase 31 sign-off
Cell 32.3  Declare SWEEP_S10_DROPOUT-v1
Cell 32.4  Load S9 winner/reference update
Cell 32.5  Freeze FV*/YS*/L*/P*/A*/B*/LR*/WD*
Cell 32.6  Define DR01/DR02/DR03
Cell 32.7  Inspect Transformer dropout sites
Cell 32.8  Freeze dropout-scope manifest/fingerprint
Cell 32.9  Build S10 run matrix
Cell 32.10 Run dropout config/unit tests
Cell 32.11 Run isolated train/eval Dropout semantic sanity
Cell 32.12 Audit common data/population
Cell 32.13 Audit architecture/parameter equality
Cell 32.14 Audit optimizer/training config
Cell 32.15 Audit initialization policy
Cell 32.16 Audit sample-order/RNG policy
Cell 32.17 Verify DR01 reference reuse eligibility
Cell 32.18 Register DR02 run
Cell 32.19 Reseed + fresh DR02 loaders/model/optimizer
Cell 32.20 Verify DR02 dropout sites before training
Cell 32.21 Execute DR02 via Training Engine
Cell 32.22 Verify mode transitions
Cell 32.23 Verify DR02 BEST
Cell 32.24 Register DR03 run
Cell 32.25 Reseed + fresh DR03 loaders/model/optimizer
Cell 32.26 Verify DR03 dropout sites before training
Cell 32.27 Execute DR03 via Training Engine
Cell 32.28 Verify mode transitions
Cell 32.29 Verify DR03 BEST
Cell 32.30 Verify constant runtime dropout/site scope
Cell 32.31 Run eval-repeatability audit
Cell 32.32 Build run provenance
Cell 32.33 Build optimizer-budget audit
Cell 32.34 Build S10 metrics table
Cell 32.35 Compute pairwise effects
Cell 32.36 Classify dropout trend/boundary status
Cell 32.37 Build optimization diagnostics
Cell 32.38 Build convergence diagnostics
Cell 32.39 Build runtime diagnostics
Cell 32.40 Build theoretical dropout context
Cell 32.41 Optional Train-vs-Validation diagnostic
Cell 32.42 Evaluate hypotheses
Cell 32.43 Generate figures
Cell 32.44 Generate findings
Cell 32.45 Select S10 winner
Cell 32.46 Write winner JSON
Cell 32.47 Write Phase 33 reference update
Cell 32.48 Run S10 tests/discrepancies
Cell 32.49 Write summary/report
Cell 32.50 Register artifacts/checksums
Cell 32.51 Write README
Cell 32.52 Phase sign-off
```

---

# 204. Execution flow

```text
Verify Phase 31
        ↓
Load S9 winner
        ↓
Freeze FV* + YS* + L* + P* + A* + B* + LR* + WD*
        ↓
Declare DR01 / DR02 / DR03
        ↓
Inspect all Transformer dropout sites
        ↓
Freeze dropout-scope fingerprint
        ↓
Verify Train/Eval mode semantics
        ↓
Audit same data/model/optimizer
        ↓
Verify DR01 exact reference
        ↓
Reuse DR01
        ↓
Register DR02
        ↓
Reseed + fresh objects
        ↓
Verify p=0.2 at frozen sites
        ↓
Train via TRAINING_ENGINE-v1
        ↓
Verify DR02 BEST in eval mode
        ↓
Register DR03
        ↓
Reseed + fresh objects
        ↓
Verify p=0.3 at frozen sites
        ↓
Train via TRAINING_ENGINE-v1
        ↓
Verify DR03 BEST in eval mode
        ↓
Verify constant dropout/site scope
        ↓
Verify eval repeatability
        ↓
Build Wh-space metrics
        ↓
Compute pairwise effects
        ↓
Analyze convergence/clipping/generalization context
        ↓
Select minimum-RMSE dropout
        ↓
Apply lower-dropout exact-tie rule
        ↓
Update Phase 33 reference
        ↓
Write S10 artifacts
        ↓
SWEEP_S10_DROPOUT-v1 sign-off
```

---

# 205. Fail-fast order

Before expensive DR02/DR03 training:

```text
1. Phase 31 sign-off
2. S9 winner identity
3. FV*/YS*/L*/P*/A*/B*/LR*/WD* lock
4. DR01/DR02/DR03 definitions
5. inspect dropout sites
6. freeze dropout scope
7. verify no new/removed sites
8. verify model.train()/eval() control
9. common sample population
10. same architecture/parameter schema
11. same optimizer/training config
12. initialization/order/RNG policy
13. DR01 reuse eligibility
14. Test firewall
15. Registry readiness
```

---

# 206. Why dropout-scope audit is the most important S10-specific check

A single field:

```text
dropout=0.2
```

can represent very different models depending on whether it affects:

```text
attention weights
residual branches
FFN hidden activations
head/input path
```

Therefore S10 is only scientifically interpretable when **where dropout acts** is frozen and traceable.

---

# 207. Why Validation eval mode is equally critical

Dropout is deliberately stochastic in training.

If Validation remains in train mode, the model-selection target itself becomes an unintended stochastic estimator.

That would contaminate:

```text
RMSE
early stopping
BEST checkpoint choice
sweep winner
```

and invalidate S10.

---

# 208. Why post-Validation train restoration matters

A subtle bug is:

```text
epoch 1 Train: dropout active
Validation: dropout off
epoch 2 Train: model accidentally still eval
```

Then subsequent epochs train without dropout.

S10 therefore explicitly audits mode transitions.

---

# 209. Why no cross-candidate mask matching

The scientific candidate is the dropout probability itself.

Requiring identical masks across p values is neither necessary nor a stable framework-level contract.

Fairness is achieved through:

```text
same seed policy
same initialization
same sample order
same call topology
```

not identical masks.

---

# 210. Why no MC Dropout

S10 asks which training regularization probability produces the best standard deterministic Validation predictor.

MC Dropout asks a different question about stochastic inference/uncertainty.

Do not mix them.

---

# 211. Why state_dict provenance is insufficient

Dropout has no learned weights.

Two model objects with different p values can have the same state-dict key schema.

Therefore:

```text
checkpoint config
dropout scope
run manifest
```

must accompany learned weights.

---

# 212. Why lower-dropout exact tie rule is reasonable

Only if full-precision Validation RMSE is exactly equal.

The rule favors:

```text
less stochastic regularization
simpler continuation
```

without overriding empirical differences.

---

# 213. Winner verification checklist

Before writing `s10_dropout_winner.json`:

```text
[ ] DR01 valid reused reference.
[ ] DR02 valid completed run.
[ ] DR03 valid completed run.
[ ] Same FV*/YS*/L*/P*/A*/B*/LR*/WD*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same architecture/parameter schema.
[ ] Same optimizer and training settings.
[ ] Dropout site topology frozen.
[ ] Scope fingerprint identical.
[ ] DR01 sites p=0.1.
[ ] DR02 sites p=0.2.
[ ] DR03 sites p=0.3.
[ ] No extra input/head/pooling dropout.
[ ] Train uses model.train().
[ ] Validation uses model.eval().
[ ] Train mode restored after Validation.
[ ] No MC Dropout.
[ ] No per-batch seed reset.
[ ] Fresh optimizer state for new runs.
[ ] No warm-start.
[ ] Initialization/order audits completed.
[ ] All BEST checkpoints instantiated with correct dropout config.
[ ] All BEST metrics verified.
[ ] Eval repeatability valid.
[ ] Full-precision RMSE used.
[ ] Pairwise effects computed.
[ ] Exact tie rule respected.
[ ] Boundary winner flagged if needed.
[ ] No Test.
```

---

# 214. Phase 33 handoff

Phase 33 receives:

```text
s10_dropout_winner.json
s10_reference_update.json
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
selected_dropout_id
selected_dropout_probability
dropout_scope_fingerprint
population_fingerprint
```

and changes only:

```text
d_model.
```

---

# 215. If DR01 wins

Current reference:

```text
FV*
YS*
L*
P*
A*
B*
LR*
WD*
dropout=0.1
D64
```

Phase 33 compares:

```text
D32
D64
```

at dropout .1.

---

# 216. If DR02 wins

Current reference uses:

```text
dropout=0.2
D64.
```

Phase 33 changes only d_model.

---

# 217. If DR03 wins

Current reference uses:

```text
dropout=0.3
D64.
```

Phase 33 changes only d_model.

---

# 218. Reference reuse for Phase 33

S10 winner already has:

```text
d_model = 64
```

therefore Phase 33 normally:

```text
reuse S10 winner as D64 reference
train only D32.
```

---

# 219. Relationship with Phase 33 d_model

Dropout may interact with capacity.

A smaller D32 model may prefer less dropout than D64, but sequential design does not re-open S10 automatically.

Document interaction limitation.

---

# 220. Relationship with Phase 34 Heads

Head count remains H4 during S10.

Later head sweep occurs under selected dropout.

---

# 221. Relationship with Phase 35 Layers

Depth can change regularization needs.

S10 winner is conditional on N2.

---

# 222. Relationship with Phase 36 FFN width

FFN capacity also interacts with dropout.

No factorial grid.

---

# 223. Relationship with Phase 37 Loss

MSE remains frozen in S10.

Different losses can alter gradients and regularization response.

---

# 224. Relationship with Phase 38 Epoch cap

If DR03 appears undertrained at E50, retain that signal.

Do not extend now.

---

# 225. Relationship with Phase 39 Gradient clipping

Dropout can affect clipping frequency through trajectory.

Clip remains 1.0 now.

Phase 39 later tests clipping.

---

# 226. Relationship with Phase 42 Candidate synthesis

All DR01/DR02/DR03 runs remain in Experiment Registry.

No losing candidate is deleted.

---

# 227. Relationship with Phase 44 Rolling-origin

S10 winner is selected on one Validation period.

Rolling-origin later tests temporal robustness.

---

# 228. Relationship with Phase 46 Multi-seed

S10 uses seed 42.

Dropout is stochastic, so small one-seed margins are especially important to treat cautiously.

Final seed robustness comes later.

---

# 229. Relationship with final attention analysis

Final attention artifacts must record:

```text
selected dropout
dropout scope fingerprint
checkpoint identity
model.eval() status
```

so readers know dropout shaped training but was inactive during standard interpretability inference.

---

# 230. Reproducibility metadata

New DR02/DR03 runs record:

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
dropout ID
dropout p
dropout scope fingerprint
dropout site registry version
train/eval mode contract
RNG policy
optimizer parameter-group policy
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

# 231. No fabricated outputs

Do not pre-fill:

```text
winner
RMSE
best epoch
train/Validation gap
gradient norms
clipping fraction
runtime
empirical zero fraction
```

before actual execution.

Theoretical `p`, `q`, `1/q` values may be defined because they follow directly from registered candidate values.

---

# 232. Phase sign-off

Create:

```text
phase_32_signoff.json
```

Minimum:

```text
phase = 32
phase_name = S10 Dropout sweep
sweep_version
sweep_id
source_s9_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
dr01_reference_run_id
dr02_run_id
dr03_run_id
new_run_ids
reused_run_ids
dropout_scope_fingerprint
dropout_site_count
winner_dropout_id
winner_dropout_probability
winner_run_id
winner_rmse_wh
winner_is_boundary
population_fingerprint
metric_version
dropout_scope_audit_status
train_eval_mode_audit_status
eval_repeatability_status
constant_dropout_audit_status
fairness_audit_status
matched_initialization_status
sample_order_match_status
inherited_warnings
test_status
approved_for_phase33
overall_status
created_at
```

---

# 233. Acceptance checklist

```text
[ ] Phase 31 valid.
[ ] approved_for_phase32 = true.
[ ] SWEEP_S10_DROPOUT-v1 declared.
[ ] S9 winner loaded.
[ ] FV* fixed.
[ ] YS* fixed.
[ ] L* fixed.
[ ] P* fixed.
[ ] A* fixed.
[ ] B* fixed.
[ ] LR* fixed.
[ ] WD* fixed.
[ ] DR01 exactly .1.
[ ] DR02 exactly .2.
[ ] DR03 exactly .3.
[ ] No extra p value.
[ ] Dropout sites inspected from actual Transformer implementation.
[ ] Dropout-site registry created.
[ ] Dropout scope fingerprint created.
[ ] Same site count across candidates.
[ ] Same module paths across candidates.
[ ] No dropout site added.
[ ] No dropout site removed.
[ ] All global sites use candidate p.
[ ] Non-controlled dropout fields unchanged.
[ ] MHA dropout follows frozen scope.
[ ] No input dropout added.
[ ] No head dropout added.
[ ] No pooling dropout added.
[ ] No DropPath.
[ ] No per-layer custom p.
[ ] No attention-only dropout split.
[ ] No FFN-only dropout split.
[ ] Dropout config validation works.
[ ] Standard Dropout train semantic sanity PASS.
[ ] Standard Dropout eval identity sanity PASS.
[ ] Empirical p test avoids ReLU-zero confound.
[ ] Same Train IDs.
[ ] Same Validation IDs.
[ ] Same WINDOWPOP-v1.
[ ] Same feature fingerprint.
[ ] Same X scaler.
[ ] Same target transform/scaler.
[ ] Same lookback/pooling/activation/batch.
[ ] Same LR/WD.
[ ] Same H1/WB0.
[ ] Same D64/H4/N2/FFN128.
[ ] Same PE/norm/mask policy.
[ ] Same regression head.
[ ] Same parameter count.
[ ] Same parameter schema.
[ ] Same state_dict key set.
[ ] Same AdamW identity/group policy.
[ ] Same MSE.
[ ] Same E50.
[ ] Same patience10.
[ ] Same min_delta.
[ ] Same clip1.
[ ] Scheduler absent.
[ ] Warmup absent.
[ ] Accumulation=1.
[ ] drop_last=False.
[ ] Same mixed precision policy.
[ ] Seed42 fixed.
[ ] No RNG reset per batch.
[ ] Cross-p mask identity not required.
[ ] DR02/DR03 initial states match.
[ ] DR01 init match checked if available.
[ ] DR02/DR03 sample orders match.
[ ] DR01 order match checked if available.
[ ] DR01 reference exact-match.
[ ] DR01 not retrained.
[ ] DR02 registered before training.
[ ] DR03 registered before training.
[ ] DR02 fresh loaders/model/optimizer.
[ ] DR03 fresh loaders/model/optimizer.
[ ] No warm-start.
[ ] No optimizer-state reuse.
[ ] DR02/DR03 trained via TRAINING_ENGINE-v1.
[ ] Training stages use model.train().
[ ] Validation stages use model.eval().
[ ] Train mode restored after Validation.
[ ] No MC Dropout.
[ ] No stochastic Validation averaging.
[ ] DR02 BEST loaded with correct p config.
[ ] DR03 BEST loaded with correct p config.
[ ] DR01 BEST provenance valid.
[ ] Checkpoint metadata contains p.
[ ] Checkpoint metadata contains scope fingerprint.
[ ] State dict alone not accepted as dropout provenance.
[ ] Runtime dropout p constant at all controlled sites.
[ ] No dropout schedule.
[ ] Eval repeatability verified.
[ ] Same steps per completed epoch.
[ ] Optimizer-budget audit generated.
[ ] All result rows use verified BEST.
[ ] All predictions finite.
[ ] Official metrics in Wh.
[ ] Full-precision RMSE used.
[ ] DR01/DR02 effect computed.
[ ] DR02/DR03 effect computed.
[ ] DR01/DR03 effect computed.
[ ] Trend pattern generated.
[ ] Winner=min RMSE.
[ ] Exact tie=lower dropout.
[ ] Existing reference does not override RMSE.
[ ] Metric divergence recorded if present.
[ ] Boundary winner recorded if DR01/DR03.
[ ] No automatic p-range extension.
[ ] Gradient/clipping diagnostics generated.
[ ] Convergence diagnostics generated.
[ ] Runtime diagnostics generated.
[ ] Theoretical p/q context generated.
[ ] Theoretical context marked diagnostic.
[ ] No global effective-dropout formula claimed.
[ ] Optional generalization diagnostics use Train/Validation only.
[ ] Smallest gap not used as winner rule.
[ ] Hypothesis outcomes recorded.
[ ] Findings generated.
[ ] Inherited warnings propagated.
[ ] Winner artifact generated.
[ ] Phase 33 reference update generated.
[ ] D64 reuse identified.
[ ] No dropout 0 experiment.
[ ] No dropout >.3 experiment.
[ ] No per-site ablation.
[ ] No LR/WD compensation.
[ ] No extra epochs.
[ ] No score-based rerun.
[ ] No failed/SANITY run in winner ranking.
[ ] No scientific attention extraction.
[ ] No Test access.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] Dropout-WD interaction documented.
[ ] Dropout-LR interaction documented.
[ ] Dropout-batch interaction documented.
[ ] Dropout-capacity interaction documented.
[ ] Dropout-budget interaction documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 234. Acceptance criteria

Phase 32 chỉ PASS khi:

```text
S9-selected data/model/optimizer configuration is fixed.

Exactly DR01=.1, DR02=.2 and DR03=.3 are represented.

The actual dropout-site scope is inspected and frozen.

Only p changes at already-controlled global dropout sites.

No site is added/removed.

No input/head/pooling dropout is introduced.

Same Train/Validation IDs and WINDOWPOP-v1 are used.

Same model parameter schema is used.

Same initialization/sample-order/RNG startup policy is audited.

DR01 exact S9 reference is reused.

DR02/DR03 are fresh seed-42 runs with fresh optimizer state.

Train uses model.train().

Validation and checkpoint verification use model.eval().

Training mode is restored after each Validation.

No MC Dropout or stochastic Validation averaging is used.

Runtime p remains constant.

All BEST checkpoints are instantiated with the correct dropout config and verified.

Validation RMSE Wh selects winner.

Exact RMSE tie selects lower dropout.

Boundary winner is documented without hidden range extension.

Phase 33 reference is generated.

Test remains untouched.
```

---

# 235. Failure conditions

Phase 32 FAIL if:

```text
wrong S9 winner used

LR or WD changes

dropout values differ from registry

dropout site scope changes

new input/head/pooling dropout is added

MHA and FFN/residual dropout are split into unregistered independent values

per-layer p differs

DropPath/stochastic depth is added

dropout schedule is introduced

Train runs in eval mode

Validation runs in train mode

train mode is not restored after Validation

MC Dropout is used for scientific Validation

RNG is reset every batch

sample population differs

model architecture/parameter schema differs

DR01 reference mismatches but is reused

DR01 is retrained and a better rerun chosen

DR02/DR03 warm-start

optimizer state is reused

candidate checkpoint is verified with wrong config provenance

one candidate fails but winner is declared anyway

unregistered dropout candidate is tested/used

RMSE rounded before ranking

MAE/R² overrides lower RMSE

generalization gap overrides lower RMSE

runtime overrides lower RMSE

score-based rerun occurs

Test is used.
```

---

# 236. Common mistakes

## 236.1 Chỉ đổi `nn.Dropout` nhưng quên MHA dropout

Nếu MHA nằm trong global dropout scope thì đây là incomplete candidate implementation.

## 236.2 Chỉ đổi MHA dropout nhưng residual/FFN dropout giữ .1

Không còn là registered global-dropout sweep.

## 236.3 Thêm head dropout vì “regularization mạnh hơn”

Hidden factor.

## 236.4 Thêm input dropout

Hidden factor.

## 236.5 Validation quên `model.eval()`

Critical scientific error.

## 236.6 Sau Validation quên `model.train()`

Các epoch sau sẽ train với dropout disabled.

## 236.7 Reset seed mỗi batch để mask “công bằng”

Sai training semantics.

## 236.8 Yêu cầu dropout masks giữa p=.1/.2/.3 giống nhau

Không phải fairness requirement.

## 236.9 Dùng MC Dropout rồi average Validation prediction

Không thuộc S10.

## 236.10 Thấy DR03 học chậm rồi tăng epoch riêng

Không.

## 236.11 Thấy DR03 tốt nhất rồi thử .4/.5

Hidden adaptive search.

## 236.12 Thấy DR01 tốt nhất rồi thử p=0

Không thuộc S10.

## 236.13 Chọn smallest Train–Validation gap

Sai primary metric.

## 236.14 Gọi p=.3 là “mất 30% thông tin”

Sai diễn giải.

## 236.15 Tính “effective dropout” bằng nhân keep probabilities

Không hợp lệ.

## 236.16 Load state_dict vào model p khác rồi coi là cùng checkpoint config

Sai provenance.

## 236.17 Retrain DR01 để có cùng RNG metadata

Không. Reference reuse được ưu tiên.

## 236.18 Bật attention extraction trong training để xem dropout effect

Không Phase 32.

## 236.19 Dùng Test để chọn regularization

Forbidden.

---

# 237. Recommended execution pseudocode

```text
load_phase31_signoff()
assert_approved_for_phase32()

s9 = load_s9_winner()

FV = s9.feature_variant_id
YS = s9.target_scaling_id
L  = s9.lookback_id
P  = s9.pooling_id
A  = s9.activation_id
B  = s9.batch_id
LR = s9.learning_rate
WD = s9.weight_decay

dropouts = {
    "DR01": 0.1,
    "DR02": 0.2,
    "DR03": 0.3
}

dropout_scope = inspect_transformer_dropout_scope(
    source_impl="TRANSFORMER_IMPL-v1",
    source_reference=s9.winner_run_id
)
freeze_dropout_scope(dropout_scope)

run_dropout_unit_tests(dropout_scope)
audit_train_eval_mode_contract()

audit_common_data_population()
audit_same_model_parameter_schema()
audit_same_optimizer_training_config(
    lr=LR,
    wd=WD
)

dr01_reference = resolve_s9_winner_run()
assert_exact_s10_reference_match(
    dr01_reference,
    dropout=0.1,
    scope=dropout_scope
)

results = {
    "DR01": dr01_reference
}

for dropout_id in ["DR02", "DR03"]:
    p = dropouts[dropout_id]

    register_run(dropout_id)

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
        activation=A,
        dropout=p
    )

    verify_dropout_scope_and_values(
        model=model,
        expected_scope=dropout_scope,
        expected_p=p
    )

    optimizer = build_fresh_adamw(
        model=model,
        lr=LR,
        weight_decay=WD,
        group_policy=FROZEN_S9_GROUP_POLICY
    )

    result = TRAINING_ENGINE_v1.fit(...)
    verify_mode_transitions(result)
    verify_constant_dropout(result, expected_p=p)
    verify_best_checkpoint_with_correct_config(
        result,
        dropout=p,
        dropout_scope=dropout_scope
    )

    results[dropout_id] = result

audit_new_run_initialization_match()
audit_new_run_sample_order_match()
audit_eval_repeatability(results)

metrics = build_verified_s10_metrics(results)
pairwise = compute_dropout_pairwise_effects(metrics)
trend = classify_dropout_trend(metrics)
optimization = build_optimization_diagnostics(results)
convergence = build_convergence_diagnostics(results)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer_lower_dropout=True
)

write_hypothesis_outcomes()
write_findings()
write_s10_winner(winner)
write_phase33_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 238. Definition of Done

\[
\boxed{
One\ Fixed\ Transformer\ Setup
+
Three\ Dropout\ Probabilities
+
Frozen\ Dropout\ Site\ Scope
+
Correct\ Train/Eval\ Modes
+
Two\ Fresh\ Runs
+
One\ Valid\ Reused\ DR01
+
Same\ LR/WD/Batch/Initialization
+
Verified\ BEST\ Metrics
+
S10\ Winner
+
Phase33\ Reference
+
No\ Test
}
\]

---

# 239. Final status contract

```text
PHASE 32 tests TRANSFORMER DROPOUT only.

Current:
feature variant from prior sweeps
target scaling from S3
lookback from S4
pooling from S5
activation from S6
batch from S7
learning rate from S8
weight decay from S9.

Candidates:

DR01 = 0.1
DR02 = 0.2
DR03 = 0.3

Before training:
inspect and freeze actual dropout sites.

Only:
dropout probability changes
at the already-controlled global dropout sites.

No:
new input dropout
new head dropout
new pooling dropout
per-layer custom p
attention-only/FFN-only split
DropPath
dropout schedule
MC Dropout.

DR01:
reuse S9 winner if exact match.

DR02/DR03:
fresh seed-42 runs.

Same:
Train/Validation IDs
WINDOWPOP-v1
X/y pipeline
model capacity D64/H4/N2/FFN128
parameter schema
batch B*
LR*
WD*
AdamW group policy
MSE
E50
patience10
clip1
accumulation1
Training Engine
Metric version.

Mode:
Train → model.train()
Validation → model.eval()
after Validation → restore model.train() for next epoch.

No RNG reset per batch.

Checkpoint provenance:
must store dropout p + scope fingerprint;
state_dict alone is insufficient.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer lower dropout.

Boundary winner:
record, do not auto-expand search.

No Test.

After SWEEP_S10_DROPOUT-v1 PASS:
update current reference
and proceed to
PHASE 33 — S11 d_model sweep.
```

---

# 240. Final check

Correct workflow:

```text
Load S9 winner
→ Freeze FV* + YS* + L* + P* + A* + B* + LR* + WD*
→ Inspect/freeze dropout-site scope
→ Define .1 / .2 / .3
→ Verify Train/Eval semantics
→ Audit same data/model/optimizer
→ Reuse DR01
→ Fresh DR02 from seed42
→ Verify p=.2 at exact frozen sites
→ Train + verify BEST in eval mode
→ Fresh DR03 from seed42
→ Verify p=.3 at exact frozen sites
→ Train + verify BEST in eval mode
→ Verify constant p and mode transitions
→ Compare full-precision Validation RMSE
→ Analyze convergence/generalization context without overriding RMSE
→ Select S10 winner
→ Apply lower-dropout exact-tie rule
→ Update Phase 33 reference
```

Incorrect workflow:

```text
change only some dropout sites
→ add head/input dropout
→ Validation in train mode
→ reset RNG every batch
→ use MC Dropout
→ retrain DR01
→ test p=.4 after seeing results
→ inspect Test
```

Chỉ sau khi `SWEEP_S10_DROPOUT-v1` được sign-off mới chuyển sang **PHASE 33 — S11 d_model sweep**.

---

# 241. Technical references used to verify dropout semantics

Phase 32 should remain aligned with the actual installed PyTorch version frozen by `ENV-v1`.

The plan's technical assumptions were cross-checked against current official PyTorch documentation for:

```text
torch.nn.Dropout
torch.nn.functional.dropout
torch.nn.MultiheadAttention
torch.nn.Module.train()
torch.nn.Module.eval()
```

Key implementation facts to preserve in the project:

```text
standard Dropout randomly zeroes elements during training
standard Dropout uses evaluation-time identity behavior
MultiheadAttention exposes a dropout probability on attention weights
train()/eval() control modules whose behavior depends on training mode
```

At execution time, the frozen project environment and audited implementation remain the operational source of truth.
