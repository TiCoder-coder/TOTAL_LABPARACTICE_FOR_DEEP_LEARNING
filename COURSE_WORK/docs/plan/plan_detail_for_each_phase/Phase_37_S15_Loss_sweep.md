# PHASE 37 — S15 LOSS SWEEP

## Kế hoạch controlled sweep cho Training Loss của Transformer Regression

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S14_FFN-v1`  
**Sweep ID:** `S15_LOSS`  
**Output version:** `SWEEP_S15_LOSS-v1`  
**Phase trước:** `Phase_36_S14_FFN_sweep.md`

---

# 1. Vai trò của Phase 37

Phase 37 là controlled experiment thứ mười lăm trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với toàn bộ data pipeline, feature configuration, target scaling, lookback, pooling, activation, batch size, optimizer, learning rate, weight decay, dropout, `d_model`, attention heads, encoder layers, FFN width, epoch budget, gradient clipping, sample population và seed đã được khóa từ Phase 36, training objective nào phù hợp hơn cho Transformer regression: Mean Squared Error hay Huber Loss?

Phase 37 chỉ thay đúng một registered factor:

```text
training_loss
```

với hai condition:

```text
L0 = MSE
L1 = Huber
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Loss\ Function\ Factor
+
Same\ Predictions/Targets
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

# 2. Vị trí Phase 37 trong master execution plan

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
```

Phase 37 không được quay lại thay:

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
epoch cap
gradient clipping
RevIN
boundary protocol
```

---

# 3. Câu hỏi nghiên cứu của S15

Phase 37 phải trả lời:

```text
1. MSE hay Huber tạo verified Validation RMSE Wh thấp hơn?

2. Huber có giúp Transformer bớt nhạy với residual lớn trong training không?

3. Gradient norm và clipping frequency có thay đổi khi dùng Huber không?

4. Huber có làm convergence ổn định hơn hay chậm hơn không?

5. Best epoch/early stopping behavior khác nhau ra sao?

6. MAE và R² có cùng ranking với RMSE không?

7. Huber residuals nằm trong quadratic/linear regime với tỷ lệ bao nhiêu?

8. Training-loss curve có thể được so sánh như thế nào mà không phạm sai lầm về scale/objective?

9. Loss nào trở thành current reference cho Phase 38?
```

---

# 4. Current reference từ Phase 36

Phase 37 phải load:

```text
s14_ffn_winner.json
s14_reference_update.json
phase_36_signoff.json
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
F*  = selected ffn_dim
```

Phase 37 không được hard-code bất kỳ winner nào từ sweep trước.

---

# 5. Canonical loss options

Registry:

```text
L0 = MSE
L1 = HUBER
```

Canonical implementation configuration:

```text
L0:
criterion = MSELoss
reduction = "mean"

L1:
criterion = HuberLoss
delta = 1.0
reduction = "mean"
```

`delta=1.0` là một phần bắt buộc của `L1` contract.

Không được tune delta trong S15.

---

# 6. Huber delta contract

Khóa:

```text
HUBER_DELTA_MODEL_SPACE = 1.0
```

Đây là `delta` trong **target model space hiện tại**, tức là sau target transform đã được chọn từ S3.

Không được diễn giải `delta=1.0` là luôn luôn `1 Wh`.

---

# 7. Huber delta semantics nếu YS0 được chọn

Nếu:

```text
YS* = YS0
```

target model space chính là raw Wh.

Khi đó:

```text
Huber delta = 1.0 Wh.
```

Điều này có thể làm nhiều residual thực tế rơi vào linear regime.

Đó là hệ quả của protocol đã đăng ký, không phải lý do để tự ý đổi delta.

---

# 8. Huber delta semantics nếu YS1 được chọn

Nếu:

```text
YS* = YS1
```

target model space là standardized target:

\[
z=\frac{y-\mu_{train}}{\sigma_{train}}
\]

và:

```text
Huber delta = 1.0 standardized target unit.
```

Equivalent raw-Wh threshold:

\[
\delta_{Wh}
=
1.0 \times \sigma_{train}
\]

với `sigma_train` lấy từ frozen train-only target scaler.

Không refit scaler.

---

# 9. Mandatory Huber delta audit

Phase 37 phải ghi rõ:

```text
target_scaling_id
model_space_delta = 1.0
raw_wh_equivalent_delta
target_scaler_checksum
delta_source = S15_PROTOCOL
delta_tuned = false
```

Nếu YS0:

```text
raw_wh_equivalent_delta = 1.0
```

Nếu YS1:

```text
raw_wh_equivalent_delta = y_scaler.scale_[0]
```

theo frozen scaler runtime.

---

# 10. Không được chọn delta dựa trên Validation

Forbidden:

```text
try delta=.5
try delta=1
try delta=2
choose best Validation delta
```

S15 chỉ là:

```text
MSE
vs
Huber(delta=1.0 model-space)
```

Nếu sau này muốn nghiên cứu delta riêng:

```text
Protocol Amendment / separate experiment
```

không nằm trong S15.

---

# 11. MSE definition

Với residual:

\[
e_i = \hat y_i - y_i
\]

MSE sample contribution:

\[
\ell_{MSE}(e)=e^2.
\]

Batch criterion:

\[
L_{MSE}
=
\frac{1}{B}
\sum_i e_i^2.
\]

Exact implementation must use the frozen mean-reduction semantics.

---

# 12. Huber definition

Với `delta > 0`:

\[
\ell_{\delta}(e)
=
\begin{cases}
\frac{1}{2}e^2, & |e|\le\delta \\
\delta\left(|e|-\frac{1}{2}\delta\right), & |e|>\delta
\end{cases}
\]

và:

```text
delta = 1.0 model-space unit.
```

Batch criterion:

\[
L_{Huber}
=
\frac{1}{B}
\sum_i \ell_{\delta}(e_i).
\]

---

# 13. Quadratic vs linear Huber regimes

Huber có hai vùng:

```text
|e| <= delta
→ quadratic regime

|e| > delta
→ linear regime.
```

Phase 37 phải hiểu đây là training-objective behavior, không phải error-regime definition cuối cùng của project.

Final error-regime analysis vẫn thuộc Phase 50.

---

# 14. Huber is not “MSE with outliers removed”

Huber không xóa sample và không xóa outlier.

Mọi training sample vẫn tham gia loss.

Khác biệt là contribution của residual lớn chuyển từ quadratic sang linear growth.

Không được viết:

```text
Huber removes outliers.
```

Safe:

```text
Huber reduces the influence growth of large residuals relative to MSE.
```

---

# 15. Loss functions optimize model-space targets

Cả MSE và Huber được tính trên:

```text
y_model
```

không phải `y_raw_wh` trực tiếp nếu YS1 đang active.

DataLoader contract vẫn giữ:

```text
y_model
y_raw_wh
```

cho Train/Validation.

Training criterion uses:

```text
y_model.
```

Evaluation metrics use predictions inverse-transformed to:

```text
Wh.
```

---

# 16. Primary selection remains Wh-space RMSE

Dù training objective thay đổi:

```text
winner metric = Validation RMSE Wh.
```

Không chọn winner bằng:

```text
lowest MSE training loss
lowest Huber training loss
lowest model-space validation criterion.
```

---

# 17. Raw loss values are not cross-condition comparable

MSE và Huber có functional form khác nhau.

Ngay cả trên cùng target space:

```text
MSE value
and
Huber value
```

không cùng numerical scale theo cách đủ để ranking trực tiếp.

Ví dụ trong quadratic region Huber dùng:

```text
0.5 * e^2
```

thay vì:

```text
e^2.
```

Do đó:

```text
train_loss_MSE = 0.5
train_loss_Huber = 0.3
```

không chứng minh Huber tốt hơn.

---

# 18. Training-loss curve comparability contract

Allowed:

```text
plot MSE training loss trajectory riêng
plot Huber training loss trajectory riêng
analyze convergence shape within each objective
```

Not allowed:

```text
rank models by raw criterion magnitude across MSE vs Huber.
```

Primary cross-loss learning curve:

```text
Validation RMSE Wh by epoch.
```

---

# 19. Recommended loss-curve visualization

Use:

```text
S15_01 Validation RMSE Wh by epoch
```

as main overlay.

For criterion curves:

```text
S15_03a MSE train criterion
S15_03b Huber train criterion
```

or one clearly annotated figure with:

```text
NOT NUMERICALLY CROSS-LOSS COMPARABLE
```

Do not imply common objective magnitude.

---

# 20. Same target transform is mandatory

Phase 37 must preserve S3-selected:

```text
YS*
```

for both MSE and Huber.

Do not run:

```text
MSE + YS1
Huber + YS0.
```

That would mix loss and target scaling.

---

# 21. Same predictions architecture

Loss is not a model-architecture factor.

Expected:

```text
same model config
same parameter count
same parameter names
same parameter shapes
same buffers
same output shape.
```

Only training criterion changes.

---

# 22. Parameter count equality is hard expected

Required:

```text
Params_MSE
=
Params_Huber.
```

Loss module itself should have no trainable model parameters.

If model parameter count differs:

```text
STOP.
```

---

# 23. State-dict schema equality is hard expected

Expected:

```text
same model state_dict keys
same parameter shapes
same buffer shapes.
```

Loss function should not alter model state schema.

---

# 24. Architecture fingerprint equality

Hard:

```text
architecture_fingerprint_MSE
=
architecture_fingerprint_Huber.
```

No model topology change.

---

# 25. Criterion configuration must be separate from model configuration

Experiment Registry should distinguish:

```text
model_config_fingerprint
criterion_config_fingerprint
training_config_fingerprint.
```

Model config identical.

Criterion config differs.

---

# 26. Criterion must not be stored ambiguously

Checkpoint/run metadata must include:

```text
loss_id
loss_name
reduction
huber_delta_if_applicable
target_scaling_id
target_model_space
criterion_config_fingerprint.
```

For MSE:

```text
huber_delta = N/A.
```

For Huber:

```text
huber_delta = 1.0.
```

---

# 27. Same initialization is strongly expected

Because model architecture is identical and seed/model builder are identical:

```text
initial_state_fingerprint_MSE
==
initial_state_fingerprint_Huber
```

should be expected when provenance exists.

This provides a strong controlled comparison.

---

# 28. Historical MSE reference initialization caveat

MSE condition is reused from Phase 36.

If its initial-state fingerprint exists:

```text
compare with fresh Huber initial state.
```

If not:

```text
MSE_INIT_MATCH = NOT_VERIFIABLE.
```

Do not retrain MSE just to recover the fingerprint.

---

# 29. Same sample order is strongly expected

Because:

```text
same Dataset
same batch size
same Train shuffle seed
same workers
```

Huber and MSE should ideally receive the same batch/sample order by epoch.

If historical MSE order fingerprints exist:

```text
compare.
```

Else:

```text
NOT_VERIFIABLE.
```

---

# 30. Dropout RNG trajectory nuance

If initial model state and sample order are identical, MSE vs Huber still produce different parameter updates after the first backward pass.

However the sequence of dropout RNG calls should remain structurally identical because model architecture and tensor shapes are identical.

With identical RNG state and execution path, masks can often be matched batch-by-batch.

Do not make exact mask matching a hard requirement unless the project already records such fingerprints.

Fairness requirement is:

```text
same RNG policy
same model mode transitions
same seed
same execution structure.
```

---

# 31. Loss affects gradients inherently

For MSE:

```text
per-example gradient magnitude grows with |e|.
```

For Huber outside delta:

```text
loss derivative saturates in magnitude relative to residual size.
```

Therefore Huber may naturally change:

```text
global gradient norms
clip frequency
parameter update trajectory.
```

This is the scientific effect being tested.

---

# 32. Do not compensate Huber gradient behavior

Forbidden:

```text
different gradient clip threshold
different LR
different WD
different batch size
different optimizer
```

for Huber.

Use exact selected current training settings.

---

# 33. Gradient clipping stays fixed

Hard:

```text
max_norm = 1.0
norm_type = 2
```

for both conditions.

Phase 39 later sweeps clipping.

---

# 34. Why clipping must remain fixed

If Huber reduces clipping frequency, that is meaningful evidence of objective-gradient behavior.

Do not turn clipping off for Huber because it “needs it less”.

That would create:

```text
loss + clipping
```

as two changed factors.

---

# 35. Learning rate remains selected LR*

No:

```text
Huber-specific LR.
```

Even if Huber gradients have different scale.

The sweep asks which loss works under the already selected optimizer settings.

---

# 36. Weight decay remains selected WD*

No loss-specific regularization change.

---

# 37. Batch size remains selected B*

Same number of samples per criterion reduction step.

Both use:

```text
reduction="mean".
```

No criterion-specific batch adaptation.

---

# 38. Epoch cap remains E50

Hard:

```text
max_epochs = 50
patience = 10
min_delta = 0
```

for both.

Phase 38 later examines epoch cap.

---

# 39. Early stopping metric remains Validation RMSE Wh

Critical:

```text
early stopping must not switch to Huber validation loss
```

for L1.

Both conditions:

```text
monitor = Validation RMSE Wh.
```

This preserves comparability and project selection policy.

---

# 40. BEST checkpoint selection remains Validation RMSE Wh

Hard:

```text
BEST = earliest strict minimum Validation RMSE Wh
```

not criterion loss.

No condition-specific checkpoint rule.

---

# 41. No scheduler/warmup changes

Hard:

```text
scheduler = None
warmup = None.
```

---

# 42. No label clipping/winsorization

Do not combine Huber with:

```text
target clipping
residual clipping
outlier removal
winsorization.
```

Huber itself is the only loss factor.

---

# 43. No sample weighting

All samples remain equally weighted under the dataset/training engine contract.

No:

```text
higher weights on peaks
lower weights on outliers.
```

---

# 44. No residual-based resampling

No hard-example mining.

No curriculum.

No adaptive sampling.

---

# 45. No quantile loss

Not part of S15.

---

# 46. No MAE/L1 candidate

Master registry only includes:

```text
MSE
Huber.
```

Do not add MAE as third condition.

---

# 47. No log-cosh

Not registered.

---

# 48. No custom asymmetric loss

No.

---

# 49. No target transformation change

No log transform, Box-Cox, clipping or differencing.

Target scaling remains `YS*`.

---

# 50. Working hypothesis H-S15-01

```text
Huber may improve Validation RMSE if large training residuals disproportionately influence MSE updates.
```

Status:

```text
UNTESTED.
```

---

# 51. H-S15-02

```text
MSE may remain superior because the project ultimately selects by RMSE and squared-error optimization directly aligns with that emphasis.
```

Status:

```text
UNTESTED.
```

This is not guaranteed because optimization/generalization dynamics matter.

---

# 52. H-S15-03

```text
Huber may reduce gradient clipping frequency relative to MSE.
```

Status:

```text
UNTESTED.
```

---

# 53. H-S15-04

```text
Huber may improve MAE without improving RMSE.
```

If observed:

```text
METRIC_RANKING_DIVERGENCE
```

must be reported.

---

# 54. Preconditions

Required:

```text
Phase 36 = PASS
```

or:

```text
PASS_WITH_WARNING
```

with no unresolved critical issue.

Also:

```text
approved_for_phase37 = true.
```

---

# 55. Required upstream artifacts

```text
s14_ffn_winner.json
s14_reference_update.json
phase_36_signoff.json
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
SWEEP_S14_FFN-v1
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

Include in:

```text
S15 manifest
S15 summary
S15 report
S15 winner
Phase38 reference update.
```

---

# 58. Swept factor only

Canonical:

```text
training_loss
```

Allowed:

```text
MSE
HUBER_DELTA_1
```

Aliases:

```text
L0
L1.
```

No additional loss.

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
same feature order/fingerprint
same X scaler
same target transform/scaler
same target IDs
same Validation ordering.
```

---

# 60. Frozen architecture contract

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

Loss must not modify architecture.

---

# 61. Frozen optimizer/training contract

Hard:

```text
B*
AdamW
LR*
WD*
same parameter-group policy
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
METRICS-v1.
```

Only criterion changes.

---

# 62. MSE reference reuse gate

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
H1
WB0
WINDOWPOP-v1

MSE
same PE
POST_NORM
same mask policy
same dropout scope

AdamW
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

# 63. Fresh Huber run

Execution:

```text
register Huber run
↓
reseed42
↓
fresh Train DataLoader
↓
fresh Validation DataLoader
↓
fresh Transformer with exact S14-selected architecture
↓
criterion = HuberLoss(delta=1.0, reduction="mean")
↓
fresh AdamW(LR*,WD*)
↓
TRAINING_ENGINE-v1
```

No warm-start.

---

# 64. No MSE retraining

Do not retrain MSE to:

```text
obtain a better seed42 draw
match Huber initialization metadata
match sample-order logs
```

Reuse exact S14 winner.

---

# 65. Huber criterion construction audit

Required runtime:

```text
criterion class = HuberLoss
delta = 1.0
reduction = mean.
```

No wrapper that silently rescales Huber loss unless already part of registered implementation.

---

# 66. MSE criterion construction audit

Reference:

```text
criterion class = MSELoss
reduction = mean.
```

No legacy sum reduction.

---

# 67. Huber delta positivity

Hard:

```text
delta > 0.
```

For S15:

```text
delta == 1.0.
```

Exact protocol assertion.

---

# 68. Loss input shape

Both conditions receive:

```text
prediction [B,1]
target y_model [B,1].
```

No flattening bug that broadcasts incorrectly.

---

# 69. Broadcast safety

Before criterion:

```text
assert pred.shape == y_model.shape
```

for every batch.

Do not rely on implicit broadcasting.

---

# 70. Finite-loss guard

Every official training batch:

```text
prediction finite
target finite
criterion finite.
```

No NaN skip.

Use `TRAINING_ENGINE-v1` fail policy.

---

# 71. Sample-weighted epoch criterion aggregation

Batch criterion uses mean reduction.

Epoch criterion history must aggregate sample-weighted:

\[
L_{epoch}
=
\frac{\sum_b n_b L_b}{\sum_b n_b}
\]

so the final partial batch does not receive equal weight to a full batch.

Applies to both MSE and Huber.

---

# 72. Criterion history field must carry loss identity

History rows should include:

```text
loss_id
criterion_name
huber_delta
train_criterion_value
```

so raw values are never misread without objective context.

---

# 73. Validation metric computation remains objective-independent

At each epoch:

```text
model.eval()
→ full ordered Validation predictions
→ inverse transform to Wh
→ MAE/RMSE/R².
```

Same code path for MSE and Huber.

---

# 74. Optional validation criterion value

You may compute the active criterion on Validation in model-space for diagnostic purposes.

If stored:

```text
validation_criterion_active
```

must be labeled objective-specific.

Do not use it for cross-loss winner ranking.

---

# 75. Do not compute MSE condition with Huber validation criterion and vice versa as selection signals

Cross-evaluating both losses on both prediction sets can be an optional analysis, but it introduces unnecessary complexity and is not required.

Primary diagnostics should remain:

```text
Wh-space MAE/RMSE/R²
gradient behavior
Huber regime occupancy.
```

---

# 76. Huber regime occupancy diagnostic

For Huber BEST checkpoint, calculate on Train and Validation separately:

```text
fraction |e_model| <= 1.0
fraction |e_model| > 1.0
```

where:

```text
e_model = prediction_model - target_model.
```

This describes how often Huber operates quadratically vs linearly.

---

# 77. Regime occupancy is not a winner metric

Do not say:

```text
more quadratic samples = better
more linear samples = better.
```

Use only to explain optimization behavior.

---

# 78. Huber residual magnitude summary

Recommended diagnostic in model-space:

```text
median |e|
p75 |e|
p90 |e|
p95 |e|
max |e|
fraction above delta.
```

Train and Validation separately.

No Test.

---

# 79. Raw-Wh residual summary remains optional

For interpretability, BEST predictions can also be summarized in Wh:

```text
median absolute error
p90 absolute error
```

but full error-by-regime analysis remains Phase 50.

Do not expand S15 into final residual analysis.

---

# 80. Gradient influence diagnostic

Recommended to record per epoch:

```text
mean preclip grad norm
max preclip grad norm
fraction clipped
nonfinite events.
```

Compare MSE vs Huber as objective-gradient diagnostics.

---

# 81. Per-batch gradient matching is not required

After the first optimizer update:

```text
model parameters diverge
```

so gradient differences reflect both loss derivative and trajectory.

Do not over-interpret batch-level causal purity beyond early training.

---

# 82. Optional first-batch controlled gradient probe

A useful diagnostic before official training may:

```text
use same disposable initial model state
same first Train batch
compute MSE gradient norm
reset to identical initial state
compute Huber gradient norm
```

with no optimizer step.

This isolates the immediate criterion-gradient difference.

Label:

```text
SANITY_DIAGNOSTIC_ONLY.
```

Discard models afterward.

---

# 83. First-batch gradient probe must not contaminate official runs

After diagnostic:

```text
discard model
reseed
recreate DataLoaders
recreate official model
recreate optimizer.
```

No state reuse.

---

# 84. Optional per-parameter-group gradient summary

If existing engine supports it safely:

```text
attention block gradient norm
FFN gradient norm
input projection gradient norm
regression head gradient norm.
```

Diagnostic only.

Do not add instrumentation that changes training.

---

# 85. Gradient clipping order remains fixed

Official order:

```text
zero_grad
forward
criterion
backward
clip_grad_norm_(max_norm=1.0, error_if_nonfinite=True)
optimizer.step.
```

Same for MSE/Huber.

---

# 86. No residual clipping before Huber

Do not do:

```text
e = clamp(e, ...)
Huber(e).
```

Huber itself defines the robust transition.

---

# 87. No detach in loss path

Prediction must remain connected to autograd.

No accidental:

```text
pred.detach()
```

before criterion.

---

# 88. No inverse transform before training loss

Training criterion remains in model-space.

Do not inverse-transform y to Wh inside backward path.

Evaluation only.

---

# 89. Why model-space Huber delta must be explicit

If YS1 is active, `delta=1` represents one standardized target unit.

If YS0 is active, it represents one Wh.

Without this metadata, the Huber experiment cannot be reproduced or interpreted.

Therefore the target-space/delta contract is a critical S15 artifact.

---

# 90. Loss-scale comparability audit

Create an explicit artifact confirming:

```text
raw criterion values are NOT cross-loss comparable
Validation RMSE Wh IS cross-loss comparable
Validation MAE Wh IS cross-loss comparable
Validation R² IS cross-loss comparable.
```

This prevents a common reporting mistake.

---

# 91. Primary metric

Hard:

```text
best_validation_rmse_wh.
```

---

# 92. Secondary scientific metrics

Record:

```text
Validation MAE Wh
Validation R²
best epoch
epochs completed
stop reason
Validation RMSE trajectory
Validation MAE trajectory.
```

---

# 93. Optimization diagnostics

Record:

```text
objective-specific train criterion
gradient norms
clipping fraction
Huber regime occupancy
convergence speed
early stopping.
```

---

# 94. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{MSE},
RMSE_{Huber}
\right)
\]

using full-precision verified Validation RMSE Wh.

---

# 95. Exact RMSE tie rule

If exact full-precision equality:

```text
prefer MSE.
```

Rationale:

```text
simpler baseline objective
direct alignment with squared-error evaluation emphasis
no extra delta hyperparameter
predeclared parsimony.
```

Only exact tie invokes this rule.

---

# 96. No MAE override

If Huber improves MAE but MSE has lower RMSE:

```text
MSE wins S15.
```

Record metric divergence.

---

# 97. No training-loss override

If Huber raw criterion appears numerically much smaller:

```text
ignore for ranking.
```

Winner remains Wh-space Validation RMSE.

---

# 98. No gradient-stability override

Even if Huber has fewer clipped batches:

```text
do not choose it unless RMSE wins
or exact tie rule says otherwise.
```

Gradient diagnostics explain behavior but do not define winner.

---

# 99. Loss effect formula

Define MSE→Huber:

\[
\Delta RMSE
=
RMSE_{MSE}
-
RMSE_{Huber}.
\]

Positive:

```text
Huber improves RMSE.
```

Relative:

\[
Improvement\%
=
100\times
\frac{RMSE_{MSE}-RMSE_{Huber}}
{RMSE_{MSE}}.
\]

Also:

```text
MAE delta = MAE_MSE - MAE_Huber
R² delta = R²_Huber - R²_MSE.
```

---

# 100. Gradient clipping effect

Define:

```text
clip_fraction_delta
=
clip_fraction_MSE - clip_fraction_Huber.
```

Positive:

```text
Huber clipped less often.
```

Diagnostic only.

---

# 101. Best-epoch effect

Record:

```text
best_epoch_MSE
best_epoch_Huber
difference
```

Do not infer faster convergence solely from lower best epoch if early trajectories differ substantially.

Use trajectory/context.

---

# 102. Metric ranking divergence

If:

```text
RMSE winner != MAE winner
```

record:

```text
METRIC_RANKING_DIVERGENCE.
```

If R² ranking differs from RMSE in a same-population setting, investigate because RMSE and R² should usually induce consistent ordering when target set is identical and R² denominator is fixed.

A discrepancy may indicate:

```text
population mismatch
metric implementation issue
rounding/reporting issue.
```

---

# 103. R² consistency guard

Since both conditions use exact same Validation targets:

\[
R^2
=
1-\frac{SSE}{SST}
\]

and RMSE is monotonic in SSE for fixed N.

Therefore full-precision RMSE and R² rankings should be consistent.

Hard diagnostic:

```text
if RMSE ranking and R² ranking conflict
→ investigate metric/population pipeline.
```

MAE may legitimately disagree.

---

# 104. BEST verification for Huber

After official Huber run:

```text
fresh exact model
↓
criterion identity metadata verified
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
verify stored BEST RMSE/MAE/R².
```

---

# 105. MSE reference verification

No retraining.

Verify:

```text
exact S14 winner
MSE criterion
same selected architecture
same population
same metric version
same Training Engine provenance
BEST already verified.
```

---

# 106. Checkpoint metadata

Huber BEST/LAST checkpoint metadata must include:

```text
loss_id = L1
loss_name = HuberLoss
huber_delta_model_space = 1.0
huber_delta_raw_wh_equivalent
target_scaling_id
target_scaler_checksum
reduction = mean
selection_metric = validation_rmse_wh.
```

---

# 107. Same sample population hard gate

Required:

```text
Train IDs_MSE == Train IDs_Huber
Validation IDs_MSE == Validation IDs_Huber
WINDOWPOP-v1 equal.
```

---

# 108. Same DataLoader settings

Both:

```text
B*
shuffle Train=true
shuffle Validation=false
drop_last=false
same worker policy
same generator policy.
```

---

# 109. Optimizer steps per complete epoch

Because data/batch fixed:

```text
steps_per_epoch_MSE
=
steps_per_epoch_Huber.
```

Total steps may differ due early stopping.

---

# 110. Parameter-update magnitude is allowed to differ

Different loss derivatives mean:

```text
optimizer updates differ
```

even with identical AdamW hyperparameters.

This is the intended factor effect.

Do not normalize gradients to force equal update magnitudes.

---

# 111. No gradient rescaling to match MSE

Do not multiply Huber by arbitrary constants to mimic MSE gradient scale.

Canonical Huber implementation remains as registered.

---

# 112. Loss reduction must remain mean

Do not use:

```text
MSE mean
Huber sum.
```

Hard fail.

---

# 113. Batch-size interaction limitation

Because both use mean reduction, B* remains fixed.

S15 does not test loss × batch interaction.

---

# 114. Target-scaling interaction limitation

The effect of Huber depends strongly on target scale because delta is expressed in model-space.

S15 is conditional on selected `YS*`.

No:

```text
YS0 × loss
YS1 × loss
```

factorial grid.

This limitation is mandatory in report.

---

# 115. Huber-delta limitation

Only:

```text
delta=1.0
```

is tested.

Therefore a result that MSE beats Huber does not prove all Huber deltas are inferior.

Safe:

> MSE outperformed Huber with the registered delta of 1.0 in the selected target model-space.

---

# 116. Loss × LR interaction limitation

LR* fixed.

Different objectives may prefer different LR.

Not tested.

---

# 117. Loss × clipping interaction limitation

Clip1 fixed.

Phase39 later explores clipping under then-current loss.

---

# 118. Loss × WD interaction limitation

WD* fixed.

---

# 119. Loss × architecture interaction limitation

D*/H*/N*/F* fixed.

Preferred loss may differ for another model capacity.

Not tested.

---

# 120. Loss × dropout interaction limitation

DR* fixed.

---

# 121. Loss × epoch-budget interaction limitation

Both receive:

```text
E50/patience10.
```

A robust loss may converge at different speed.

No budget adaptation here.

---

# 122. Single-seed limitation

Mandatory:

```text
S15 uses seed42 only.
```

No mean±std.

---

# 123. Validation-only limitation

Mandatory:

```text
S15 winner is selected using Validation only.
```

No Test evidence.

---

# 124. Sequential-selection limitation

By Phase37, Validation has influenced:

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
S14 FFN
S15 Loss.
```

Later rolling-origin/multi-seed stages remain necessary.

---

# 125. No Test access

Hard:

```text
Test loader not iterated
Test metrics absent
Test predictions absent.
```

---

# 126. No delta tuning

No.

---

# 127. No SmoothL1 substitution

Do not silently use another robust loss implementation under the label Huber.

Criterion identity must be exact and logged.

If codebase already wraps Huber, verify mathematical equivalence and configuration before official run.

---

# 128. No MAE/L1

No.

---

# 129. No log-cosh

No.

---

# 130. No quantile loss

No.

---

# 131. No asymmetric penalty

No.

---

# 132. No outlier removal

No.

---

# 133. No target clipping

No.

---

# 134. No residual clipping

No.

---

# 135. No sample weighting

No.

---

# 136. No hard-example mining

No.

---

# 137. No loss scheduler

Do not transition:

```text
Huber → MSE
MSE → Huber
```

across epochs.

Loss identity stays fixed for the entire run.

---

# 138. No delta scheduler

No annealing:

```text
delta 2 → 1 → .5.
```

---

# 139. No mixed objective

No:

```text
alpha*MSE + beta*Huber.
```

---

# 140. No criterion-dependent model output activation

Final output remains linear.

---

# 141. Run failure policy

If Huber has technical failure:

```text
S15 incomplete
```

until documented technical rerun succeeds.

Do not automatically declare MSE winner.

---

# 142. Technical rerun allowed

Only for documented:

```text
process interruption
hardware/software failure
corrupt checkpoint
artifact-write failure.
```

Use Experiment Registry rerun policy.

---

# 143. Score-based rerun forbidden

Do not rerun Huber because:

```text
RMSE looks poor
best epoch seems unlucky
clipping is high.
```

Do not retrain MSE.

---

# 144. Numerical instability policy

If Huber produces NaN/Inf:

```text
audit prediction/target shape
delta
data
optimizer
device
implementation.
```

Do not emergency-change:

```text
LR
clip
delta
batch.
```

---

# 145. Discrepancy taxonomy

```text
S14_REFERENCE_MISSING
S14_WINNER_MISMATCH
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
LOSS_DEFINITION_MISMATCH
HUBER_DELTA_MISMATCH
HUBER_DELTA_NOT_MODEL_SPACE
HUBER_DELTA_TUNED
LOSS_REDUCTION_MISMATCH
WRONG_CRITERION_CLASS
SMOOTHL1_SUBSTITUTION
MIXED_OBJECTIVE_USED
LOSS_SCHEDULER_USED
DELTA_SCHEDULER_USED
MODEL_ARCHITECTURE_DRIFT
PARAMETER_COUNT_MISMATCH
STATE_DICT_SCHEMA_MISMATCH
INITIALIZATION_POLICY_DRIFT
SAMPLE_ORDER_POLICY_DRIFT
TARGET_TRANSFORM_DRIFT
TARGET_SCALER_MISMATCH
TRAINING_LOSS_ON_RAW_WH_WHEN_YS1
INVERSE_TRANSFORM_IN_BACKWARD_PATH
BROADCASTING_LOSS_SHAPE_ERROR
EARLY_STOP_METRIC_DRIFT
BEST_SELECTION_METRIC_DRIFT
GRADIENT_CLIP_DRIFT
OPTIMIZER_DRIFT
LOSS_SPECIFIC_LR_USED
LOSS_SPECIFIC_WD_USED
LOSS_SPECIFIC_BATCH_USED
OUTLIER_REMOVAL_USED
TARGET_CLIPPING_USED
RESIDUAL_CLIPPING_USED
SAMPLE_WEIGHTING_USED
POPULATION_MISMATCH
TARGET_ID_MISMATCH
X_SCALER_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
REFERENCE_RUN_MISMATCH
RUN_FAILURE
NUMERICAL_INSTABILITY
CHECKPOINT_CONFIG_MISMATCH
CHECKPOINT_VERIFICATION_FAILURE
RAW_LOSS_CROSS_COMPARISON_USED
RANKING_ERROR
EFFECT_CALCULATION_ERROR
RMSE_R2_RANKING_INCONSISTENCY
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 146. Status model

## PASS

```text
MSE reference valid
Huber exact definition valid
same model/data/optimizer/budget
delta semantics documented
Huber BEST verified
Wh-space metric comparison valid
winner selected
Phase38 reference generated
Test untouched.
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
MAE ranking divergence
Huber mostly linear regime
Huber mostly quadratic regime
reference init/order not verifiable
large clipping difference
inherited warning.
```

## FAIL

Examples:

```text
delta tuned
target scaling changes
loss-specific LR/clip change
raw criterion values used for winner
architecture drift
population mismatch
wrong Huber implementation
Test access.
```

---

# 147. Output directory

```text
artifacts/
└── sweeps/
    └── S15_loss/
        ├── s15_loss_sweep_manifest.json
        ├── s15_loss_sweep_contract.json
        ├── s15_loss_preflight_audit.csv
        ├── s15_run_matrix.csv
        ├── s15_loss_definition_audit.csv
        ├── s15_huber_delta_audit.json
        ├── s15_target_space_loss_audit.csv
        ├── s15_loss_scale_comparability_audit.csv
        ├── s15_architecture_invariance_audit.csv
        ├── s15_parameter_schema_audit.csv
        ├── s15_criterion_unit_tests.csv
        ├── s15_common_data_audit.csv
        ├── s15_loss_training_audit.csv
        ├── s15_initialization_audit.csv
        ├── s15_sample_order_audit.csv
        ├── s15_dropout_rng_policy_audit.csv
        ├── s15_first_batch_gradient_probe.csv
        ├── s15_optimizer_budget_audit.csv
        ├── s15_loss_run_provenance.csv
        ├── s15_loss_metrics.csv
        ├── s15_loss_effect.csv
        ├── s15_huber_regime_diagnostics.csv
        ├── s15_gradient_diagnostics.csv
        ├── s15_clipping_diagnostics.csv
        ├── s15_optimization_diagnostics.csv
        ├── s15_convergence_diagnostics.csv
        ├── s15_runtime_diagnostics.csv
        ├── s15_generalization_diagnostics.csv
        ├── s15_hypothesis_outcomes.csv
        ├── s15_loss_findings.csv
        ├── s15_loss_winner.json
        ├── s15_reference_update.json
        ├── s15_loss_sweep_tests.csv
        ├── s15_loss_discrepancies.json
        ├── s15_loss_sweep_summary.json
        ├── s15_loss_sweep_report.md
        ├── figures/
        │   ├── S15_01_validation_rmse_by_epoch.png
        │   ├── S15_02_validation_mae_by_epoch.png
        │   ├── S15_03_objective_specific_train_loss.png
        │   ├── S15_04_gradient_norm_by_epoch.png
        │   ├── S15_05_clipping_fraction_by_epoch.png
        │   ├── S15_06_best_validation_metrics.png
        │   ├── S15_07_huber_regime_occupancy.png
        │   ├── S15_08_convergence_summary.png
        │   └── S15_09_generalization_gap_optional.png
        ├── README_S15_LOSS_SWEEP.md
        └── phase_37_signoff.json
```

Optional:

```text
s15_first_batch_gradient_probe.csv
s15_generalization_diagnostics.csv
S15_09_generalization_gap_optional.png
```

may be omitted if unsupported, with explicit reason.

New Huber scientific run remains under:

```text
artifacts/runs/<run_id>/
```

No checkpoint duplication.

---

# 148. Required outputs

```text
O37.1  Sweep manifest
O37.2  Sweep contract
O37.3  Preflight audit
O37.4  Run matrix
O37.5  Loss definition audit
O37.6  Huber delta audit
O37.7  Target-space/loss audit
O37.8  Loss-scale comparability audit
O37.9  Architecture invariance audit
O37.10 Parameter-schema audit
O37.11 Criterion unit tests
O37.12 Common-data audit
O37.13 Training-config audit
O37.14 Initialization audit
O37.15 Sample-order audit
O37.16 Dropout/RNG policy audit
O37.17 Optional first-batch gradient probe
O37.18 Optimizer-budget audit
O37.19 Run provenance
O37.20 Reused MSE reference
O37.21 Verified new Huber run
O37.22 Primary metrics table
O37.23 Loss effect table
O37.24 Huber regime diagnostics
O37.25 Gradient diagnostics
O37.26 Clipping diagnostics
O37.27 Optimization diagnostics
O37.28 Convergence diagnostics
O37.29 Runtime diagnostics
O37.30 Optional generalization diagnostics
O37.31 Hypothesis outcomes
O37.32 Findings
O37.33 Winner artifact
O37.34 Phase38 reference update
O37.35 Figures
O37.36 Sweep tests
O37.37 Discrepancy log
O37.38 Sweep summary
O37.39 Human-readable report
O37.40 README
O37.41 Phase sign-off
```

---

# 149. Run matrix

Create:

```text
s15_run_matrix.csv
```

Fields:

```text
sweep_id
loss_id
loss_name
huber_delta_model_space
huber_delta_raw_wh_equivalent
reduction
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
population_fingerprint
feature_fingerprint
seed
model_config_id
criterion_config_id
training_config_id
status
```

---

# 150. Sweep manifest

Create:

```text
s15_loss_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S15_LOSS-v1
sweep_id = S15_LOSS
source_s14_winner_run_id
selected_architecture_config
candidate_losses = [MSE, HUBER]
huber_delta_model_space = 1.0
huber_delta_tuned = false
loss_reduction = mean
target_scaling_id
target_model_space
feature_variant_id
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
new_runs_required = 1
reused_runs = 1
swept_field = training_loss
model_parameter_count_equality_expected = true
architecture_equality_expected = true
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = MSE_ON_EXACT_RMSE_TIE
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

# 151. Sweep contract

Create:

```text
s15_loss_sweep_contract.json
```

Must state:

```text
Only training loss changes.

L0 = MSELoss(reduction=mean).
L1 = HuberLoss(delta=1.0, reduction=mean).

Huber delta is 1.0 in selected target model-space.
Delta is not tuned.

Same YS* for both.
Same architecture and parameter schema.
Same model initialization policy.
Same data/sample order policy.
Same optimizer/LR/WD/batch/dropout.
Same E50/patience10/clip1.
Same early-stop and BEST metric = Validation RMSE Wh.

Raw criterion values are not used for cross-loss ranking.

MSE reference reused.
Huber fresh seed42 run.

Validation RMSE Wh selects winner.
Exact tie → MSE.
No Test.
```

---

# 152. Preflight audit

`s15_loss_preflight_audit.csv` checks:

```text
phase36_pass
approved_for_phase37
s14_winner_valid
all prior selected fields locked
MSE registered
Huber registered
Huber delta exactly1
Huber delta target space explicit
reduction mean both
same YS*
same architecture
same parameter count expectation
same optimizer
same batch
same clip
same epoch budget
same early-stop metric
same BEST metric
population fixed
Training Engine fixed
Metric fixed
seed fixed
MSE reuse candidate valid
Test lock
status
```

---

# 153. Loss definition audit

`s15_loss_definition_audit.csv`:

```text
loss_id
loss_name
criterion_class
delta
reduction
input_space
formula_id
trainable_parameters
registered
status
```

Expected:

```text
criterion trainable_parameters = 0
```

for both.

---

# 154. Huber delta audit

`s15_huber_delta_audit.json`:

```text
loss_id = L1
delta_model_space = 1.0
target_scaling_id
target_model_space
target_scaler_type
target_scaler_checksum
target_scaler_scale_wh_if_applicable
delta_raw_wh_equivalent
delta_source = S15_PROTOCOL
delta_tuned = false
delta_positive = true
status
```

---

# 155. Target-space/loss audit

`s15_target_space_loss_audit.csv`:

```text
loss_id
target_scaling_id
training_target_tensor
training_target_space
training_prediction_space
inverse_transform_used_in_backward
evaluation_inverse_transform
evaluation_unit
same_target_transform
status
```

Expected:

```text
inverse_transform_used_in_backward = false
evaluation_unit = Wh.
```

---

# 156. Loss-scale comparability audit

`s15_loss_scale_comparability_audit.csv`:

```text
quantity
mse_condition_meaning
huber_condition_meaning
cross_condition_numeric_comparable
eligible_for_winner_selection
notes
```

Rows:

```text
train_criterion
validation_active_criterion
validation_mae_wh
validation_rmse_wh
validation_r2
gradient_norm
clipping_fraction.
```

Expected:

```text
Validation RMSE Wh → comparable, winner eligible
Validation MAE Wh → comparable, secondary
Validation R² → comparable, secondary
raw criterion → not winner comparable.
```

---

# 157. Architecture invariance audit

`s15_architecture_invariance_audit.csv`:

```text
component
config_mse
config_huber
shape_mse
shape_huber
equal
status
```

Components:

```text
input projection
PE
each MHA
each FFN
LayerNorm
pooling
regression head
dropout modules.
```

All equal.

---

# 158. Parameter-schema audit

`s15_parameter_schema_audit.csv`:

```text
parameter_or_buffer_name
shape_mse
shape_huber
dtype_mse
dtype_huber
trainable_mse
trainable_huber
equal
status
```

Expected all equal.

---

# 159. Criterion unit tests

`s15_criterion_unit_tests.csv` should cover at least:

```text
MSE criterion class correct
MSE reduction mean
Huber criterion class correct
Huber delta exactly1
Huber reduction mean
delta positive
same prediction/target shape
no broadcasting
both criteria finite on finite input
zero residual gives zero loss
Huber small residual quadratic formula
Huber boundary |e|=delta correct
Huber large residual linear formula
Huber symmetry for ±e
MSE symmetry for ±e
Huber no trainable params
MSE no trainable params
architecture parameter count equal
state_dict schema equal
Huber delta raw-Wh equivalent derivation valid
training target space correct
no inverse transform in backward
Validation metrics inverse transform correct
early stop metric RMSE Wh
BEST metric RMSE Wh.
```

---

# 160. Common-data audit

`s15_common_data_audit.csv`:

```text
split
sample_count_mse
sample_count_huber
sample_ids_equal
ordered_ids_equal
feature_fingerprint_equal
x_scaler_equal
target_transform_equal
target_scaler_checksum_equal
lookback_equal
pooling_equal
activation_equal
batch_equal
lr_equal
wd_equal
dropout_equal
architecture_equal
population_equal
status
```

---

# 161. Training audit

`s15_loss_training_audit.csv`:

```text
loss_id
criterion
delta
reduction
batch
optimizer
learning_rate
weight_decay
dropout
max_epochs
patience
min_delta
clip
scheduler
warmup
accumulation
mixed_precision
seed
early_stop_metric
best_checkpoint_metric
training_engine_version
only_loss_differs
status
```

---

# 162. Initialization audit

`s15_initialization_audit.csv`:

```text
loss_id
seed
model_config_fingerprint
initialization_policy_fingerprint
initial_state_fingerprint
reference_available
exact_match_expected
match
status
```

For architecture-identical fresh probes:

```text
exact_match_expected = true.
```

For historical MSE artifact:

```text
reference_available may be false.
```

---

# 163. Sample-order audit

`s15_sample_order_audit.csv`:

```text
epoch_or_probe
mse_order_fingerprint
huber_order_fingerprint
mse_reference_available
same_order
status
```

---

# 164. Dropout/RNG policy audit

`s15_dropout_rng_policy_audit.csv`:

```text
loss_id
seed
dropout_probability
dropout_scope_fingerprint
model_mode_transitions
rng_policy_id
manual_seed_reset_per_batch
same_execution_structure
status
```

Expected:

```text
manual_seed_reset_per_batch = false.
```

---

# 165. Optional first-batch gradient probe

`s15_first_batch_gradient_probe.csv`:

```text
loss_id
same_initial_state
same_batch
criterion_value
preclip_global_grad_norm
fraction_or_count_parameters_nonzero_grad
finite
optimizer_step_performed=false
diagnostic_only=true
status
```

This is a disposable sanity diagnostic only.

---

# 166. Optimizer budget audit

`s15_optimizer_budget_audit.csv`:

```text
loss_id
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

# 167. Run provenance

`s15_loss_run_provenance.csv`:

```text
loss_id
loss_name
huber_delta_model_space
run_id
source_type
source_phase
model_config_fingerprint
criterion_config_fingerprint
training_config_fingerprint
parameter_schema_fingerprint
parameter_count
feature_fingerprint
population_fingerprint
target_scaler_checksum
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

# 168. Primary metrics table

`s15_loss_metrics.csv`:

```text
loss_id
loss_name
huber_delta_model_space
huber_delta_raw_wh_equivalent
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
mae_rank
r2_rank
is_empirical_winner
population_fingerprint
metric_version
status
```

---

# 169. Loss effect table

`s15_loss_effect.csv`:

```text
mse_run_id
huber_run_id
mse_rmse_wh
huber_rmse_wh
rmse_delta_mse_to_huber_wh
rmse_improvement_pct
mse_mae_wh
huber_mae_wh
mae_delta_mse_to_huber_wh
mse_r2
huber_r2
r2_delta
rmse_mae_ranking_divergence
rmse_r2_ranking_consistent
winner
status
```

---

# 170. Huber regime diagnostics

`s15_huber_regime_diagnostics.csv`:

```text
split
run_id
target_model_space
delta_model_space
delta_raw_wh_equivalent
sample_count
fraction_quadratic
fraction_linear
median_abs_residual_model
p75_abs_residual_model
p90_abs_residual_model
p95_abs_residual_model
max_abs_residual_model
status
```

Rows:

```text
TRAIN_BEST optional
VALIDATION_BEST required diagnostic if feasible.
```

No Test.

---

# 171. Gradient diagnostics

`s15_gradient_diagnostics.csv`:

```text
loss_id
epoch
mean_preclip_grad_norm
median_preclip_grad_norm_optional
max_preclip_grad_norm
nonfinite_grad_events
optimizer_steps
status
```

---

# 172. Clipping diagnostics

`s15_clipping_diagnostics.csv`:

```text
loss_id
epoch
clipped_batches
total_batches
clipping_fraction
max_preclip_grad_norm
status
```

Summary should include:

```text
mean clipping fraction
max clipping fraction
total clipped steps.
```

---

# 173. Optimization diagnostics

`s15_optimization_diagnostics.csv`:

```text
loss_id
best_epoch
last_epoch
stop_reason
best_rmse_wh
last_rmse_wh
objective_specific_train_loss_at_best_or_epoch
mean_grad_norm_preclip
max_grad_norm_preclip
mean_clipping_fraction
nonfinite_events
best_to_last_rmse_gap
status
```

Raw criterion field must be explicitly objective-specific.

---

# 174. Convergence diagnostics

`s15_convergence_diagnostics.csv`:

```text
loss_id
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

# 175. Runtime diagnostics

`s15_runtime_diagnostics.csv`:

```text
loss_id
device
epochs_completed
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
samples_per_second_optional
runtime_comparable
status
```

Runtime is secondary.

---

# 176. Optional generalization diagnostics

`s15_generalization_diagnostics.csv`:

```text
loss_id
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

# 177. Hypothesis outcomes

`s15_hypothesis_outcomes.csv`:

```text
hypothesis_id
comparison
expected_direction_or_pattern
observed_metrics
gradient_context
huber_regime_context
outcome
interpretation
status
```

Allowed:

```text
SUPPORTED
NOT_SUPPORTED
INCONCLUSIVE_TIE.
```

---

# 178. Findings artifact

`s15_loss_findings.csv` possible codes:

```text
MSE_GAIN
HUBER_GAIN
LOSS_EXACT_TIE
METRIC_RANKING_DIVERGENCE
RMSE_R2_RANKING_INCONSISTENCY
HUBER_MOSTLY_QUADRATIC
HUBER_MOSTLY_LINEAR
HUBER_MIXED_REGIME
HUBER_REDUCED_CLIPPING
HUBER_INCREASED_CLIPPING
GRADIENT_NORM_DIFFERENCE
CONVERGENCE_DIFFERENCE
EARLY_STOP_DIFFERENCE
GENERALIZATION_GAP_DIFFERENCE
LOSS_SCALE_NONCOMPARABILITY_VERIFIED
TARGET_SPACE_DELTA_VERIFIED
INITIALIZATION_MATCH_VERIFIED
INITIALIZATION_NOT_VERIFIABLE
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
INHERITED_WARNING.
```

---

# 179. Winner artifact

`s15_loss_winner.json` minimum:

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
selection_metric
selection_direction
tie_rule
winner_loss_id
winner_loss_name
winner_huber_delta_if_applicable
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_loss_id
runner_up_loss_name
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
metric_ranking_divergence
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 180. Phase38 reference update

`s15_reference_update.json`:

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
previous_loss=MSE
selected_loss_id
selected_loss_name
selected_huber_delta_if_applicable
winner_run_id
winner_config_fingerprint
winner_rmse_wh
current_max_epochs=50
current_patience=10
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase38
```

---

# 181. Phase38 handoff logic

Phase38 tests epoch-cap sensitivity:

```text
E50
vs
E100
```

while holding S15-selected loss fixed.

Current S15 winner already uses:

```text
E50.
```

Therefore normally:

```text
reuse S15 winner as E50 reference
train E100 candidate
```

if exact match.

---

# 182. Epoch-cap handoff nuance

Phase38 must preserve selected loss exactly.

If Huber wins:

```text
Huber delta remains 1.0 model-space.
```

No delta retuning during epoch-cap sweep.

If MSE wins:

```text
MSE remains active.
```

---

# 183. Figures

Recommended:

```text
S15_01_validation_rmse_by_epoch.png
S15_02_validation_mae_by_epoch.png
S15_03_objective_specific_train_loss.png
S15_04_gradient_norm_by_epoch.png
S15_05_clipping_fraction_by_epoch.png
S15_06_best_validation_metrics.png
S15_07_huber_regime_occupancy.png
S15_08_convergence_summary.png
S15_09_generalization_gap_optional.png
```

---

# 184. Primary figure

```text
S15_01_validation_rmse_by_epoch.png
```

Overlay:

```text
MSE
Huber
```

using raw Wh Validation RMSE.

This is the most important cross-loss learning curve.

---

# 185. Objective-specific training-loss figure

`S15_03_objective_specific_train_loss.png` must clearly state:

```text
Raw MSE and Huber criterion magnitudes are not directly comparable.
```

Preferred visualization:

```text
two panels
or
two normalized-within-objective trajectories.
```

If normalized trajectories are used, label them diagnostic only.

---

# 186. Huber regime figure

`S15_07_huber_regime_occupancy.png` may show:

```text
quadratic fraction
linear fraction
```

for Train/Validation BEST.

No Test.

---

# 187. Safe interpretation if MSE wins

> Under the selected target scaling, architecture and optimization configuration, MSE achieved lower Validation RMSE than Huber with `delta=1.0` in model-space.

Do not generalize to all Huber configurations.

---

# 188. Safe interpretation if Huber wins

> Huber with `delta=1.0` in the selected target model-space achieved lower Validation RMSE than MSE under the frozen current configuration.

Can additionally report observed gradient/clipping differences as supporting diagnostics.

Do not state Huber “removed outliers”.

---

# 189. Safe interpretation if exact tie

> MSE and Huber produced exactly equal full-precision Validation RMSE; MSE was retained under the predeclared parsimony rule.

---

# 190. Safe interpretation of Huber regime occupancy

Example:

> A substantial fraction of Validation residuals exceeded the registered Huber delta, indicating that the linear branch of the Huber objective was frequently active.

Do not say this proves robustness or causality by itself.

---

# 191. Interpretation prohibitions

Do not claim:

```text
Huber removes outliers
Huber is always robust and therefore better
MSE is always optimal for RMSE
smaller raw Huber training loss means better model
fewer clipped batches automatically means better model
delta=1 means 1 Wh when YS1 is active
MSE beats all possible Huber deltas
Huber wins because of a specific outlier without controlled evidence.
```

---

# 192. Sweep summary

Create:

```text
s15_loss_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
selected_architecture
target_scaling_id
target_model_space
huber_delta_model_space
huber_delta_raw_wh_equivalent
reference_run_id
huber_run_id
new_runs
reused_runs
candidate_losses
primary_metric
metrics_by_loss
loss_effect
loss_scale_comparability_status
gradient_diagnostics
clipping_diagnostics
huber_regime_diagnostics
optimization_diagnostics
convergence_diagnostics
runtime_diagnostics
initialization_match
sample_order_match
winner
winner_margin
metric_ranking_divergence
inherited_warnings
phase38_reference
test_status
overall_status
```

---

# 193. Human-readable report

Create:

```text
s15_loss_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S14
3. MSE and Huber definitions
4. Huber delta contract
5. Selected target model-space
6. Raw-Wh equivalent delta
7. Frozen-variable contract
8. Model/parameter invariance
9. Same data/sample fairness
10. Initialization/sample-order fairness
11. Loss-scale non-comparability
12. MSE reference provenance
13. Huber run provenance
14. Validation metrics
15. Loss effect
16. Learning-curve analysis
17. Gradient/clipping diagnostics
18. Huber quadratic/linear regime diagnostics
19. Convergence/early-stop analysis
20. Optional generalization analysis
21. S15 winner
22. Interpretation cautions
23. Interaction limitations
24. Phase38 handoff
```

---

# 194. README

Create:

```text
README_S15_LOSS_SWEEP.md
```

Must explain:

```text
Purpose
S14 winner handoff
MSE definition
Huber definition
delta=1.0 model-space
YS0 vs YS1 delta semantics
raw-Wh equivalent delta
no delta tuning
same architecture/model parameters
same optimizer/batch/LR/WD/clip
Validation RMSE Wh as shared selection metric
raw criterion values not cross-loss comparable
Huber quadratic/linear regimes
gradient/clipping implications
no outlier removal
MSE reference reuse
winner/tie rule
target-scaling interaction limitation
delta limitation
Phase38 handoff
No Test.
```

---

# 195. Recommended notebook structure

```text
Cell 37.1  Phase title
Cell 37.2  Verify Phase36 sign-off
Cell 37.3  Declare SWEEP_S15_LOSS-v1
Cell 37.4  Load S14 winner/reference
Cell 37.5  Freeze all prior selected fields
Cell 37.6  Define MSE/Huber
Cell 37.7  Lock Huber delta=1.0 model-space
Cell 37.8  Resolve raw-Wh equivalent delta
Cell 37.9  Build run matrix
Cell 37.10 Audit same target space
Cell 37.11 Audit loss-scale comparability
Cell 37.12 Audit architecture/parameter invariance
Cell 37.13 Audit criterion definitions
Cell 37.14 Run criterion unit tests
Cell 37.15 Audit common population
Cell 37.16 Audit training config
Cell 37.17 Audit initialization
Cell 37.18 Audit sample-order policy
Cell 37.19 Audit dropout/RNG policy
Cell 37.20 Verify MSE reference reuse
Cell 37.21 Optional controlled first-batch gradient probe
Cell 37.22 Register Huber run
Cell 37.23 Reseed + fresh Huber loaders/model/optimizer
Cell 37.24 Train Huber via TRAINING_ENGINE-v1
Cell 37.25 Verify Huber BEST
Cell 37.26 Build run provenance
Cell 37.27 Build optimizer-budget audit
Cell 37.28 Build Wh-space metrics
Cell 37.29 Compute MSE→Huber effect
Cell 37.30 Build gradient diagnostics
Cell 37.31 Build clipping diagnostics
Cell 37.32 Build Huber regime diagnostics
Cell 37.33 Build optimization diagnostics
Cell 37.34 Build convergence diagnostics
Cell 37.35 Build runtime diagnostics
Cell 37.36 Optional Train-vs-Validation diagnostic
Cell 37.37 Evaluate hypotheses
Cell 37.38 Generate figures
Cell 37.39 Generate findings
Cell 37.40 Select winner
Cell 37.41 Write winner JSON
Cell 37.42 Write Phase38 reference update
Cell 37.43 Run S15 tests/discrepancies
Cell 37.44 Write summary/report
Cell 37.45 Register artifacts/checksums
Cell 37.46 Write README
Cell 37.47 Phase sign-off
```

---

# 196. Execution flow

```text
Verify Phase36
→ Load S14 winner
→ Freeze full selected architecture/data/optimizer
→ Define MSE vs Huber
→ Lock Huber delta=1 model-space
→ Resolve raw-Wh equivalent delta
→ Audit same population/target transform
→ Audit identical architecture/parameters
→ Audit criterion definitions
→ Verify MSE reference
→ Reuse MSE
→ Optional disposable gradient probe
→ Register Huber
→ Seed42 + fresh Huber objects
→ Train via TRAINING_ENGINE-v1
→ Verify Huber BEST
→ Compare Wh-space Validation RMSE
→ Analyze gradient/clipping behavior
→ Analyze Huber regime occupancy
→ Select min RMSE
→ Exact tie prefer MSE
→ Update Phase38 reference
→ Write artifacts/sign-off
```

---

# 197. Fail-fast order

Before expensive Huber training:

```text
1. Phase36 sign-off
2. S14 winner identity
3. freeze all previous selections
4. MSE/Huber registry
5. Huber delta exactly1
6. target model-space resolved
7. raw-Wh equivalent delta resolved
8. same population
9. same target scaling
10. architecture equality
11. parameter-count/schema equality
12. criterion classes/reduction
13. no broadcasting
14. Training Engine selection metric still RMSE Wh
15. same optimizer/LR/WD/batch/clip
16. initialization policy
17. sample-order policy
18. MSE reuse eligibility
19. Test firewall
20. Registry readiness
```

---

# 198. Critical technical note: Huber delta belongs to target model-space

This is the most important S15 reproducibility detail.

Always record:

```text
delta_model_space
target_scaling_id
delta_raw_wh_equivalent.
```

Without these, `delta=1` is ambiguous.

---

# 199. Critical technical note: raw loss values cannot rank MSE vs Huber

Different objective functions have different scales and formulas.

Cross-condition comparison must use:

```text
Validation RMSE Wh
MAE Wh
R².
```

---

# 200. Critical technical note: early stopping must not change with criterion

Both:

```text
monitor Validation RMSE Wh.
```

If Huber run early-stops on Huber Validation criterion instead:

```text
S15 is invalid.
```

---

# 201. Critical technical note: gradient differences are expected

Huber changes the derivative for large residuals.

Different gradient norms/clipping rates are not fairness violations.

Changing LR/clip to compensate would be.

---

# 202. Critical technical note: same architecture means same parameters

MSE vs Huber must not alter:

```text
model state
parameter count
state_dict schema.
```

Any difference indicates hidden configuration drift.

---

# 203. Critical technical note: Huber does not delete large-error samples

Every sample still contributes.

Only the growth of its loss/gradient changes beyond delta.

---

# 204. Critical technical note: YS0 may make delta=1 very small in Wh

If S3 selected YS0:

```text
delta=1 Wh
```

may place many residuals in Huber's linear regime.

Do not secretly change delta.

Document the regime occupancy and conditional interpretation.

---

# 205. Critical technical note: YS1 makes delta scale-relative

If YS1 selected:

```text
delta=1 standardized unit
```

corresponds to the frozen train target standard deviation in raw Wh.

This often makes the threshold interpretable relative to train target variability.

Still no tuning.

---

# 206. Winner verification checklist

Before writing `s15_loss_winner.json`:

```text
[ ] MSE valid reused reference.
[ ] Huber valid completed run.
[ ] Same FV*/YS*/L*/P*/A*/B*/LR*/WD*/DR*/D*/H*/N*/F*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same target scaler/checksum.
[ ] Huber delta exactly1 model-space.
[ ] Raw-Wh equivalent delta recorded.
[ ] No delta tuning.
[ ] MSE/Huber reduction both mean.
[ ] Same architecture fingerprint.
[ ] Same parameter count.
[ ] Same parameter schema.
[ ] Same output shape.
[ ] Same optimizer/parameter-group policy.
[ ] Same clip threshold.
[ ] Same E50/patience10.
[ ] Same early-stop metric RMSE Wh.
[ ] Same BEST metric RMSE Wh.
[ ] Same initialization policy.
[ ] Sample-order policy audited.
[ ] No outlier removal/clipping/weighting.
[ ] No loss-specific LR/WD/batch.
[ ] No mixed/scheduled loss.
[ ] No warm-start/optimizer-state reuse.
[ ] Huber BEST verified.
[ ] MSE BEST provenance valid.
[ ] Raw criterion values not used for ranking.
[ ] Full-precision Validation RMSE used.
[ ] RMSE/R² ranking consistency checked.
[ ] MAE divergence recorded if present.
[ ] Gradient/clipping diagnostics secondary only.
[ ] Huber regime occupancy diagnostic generated.
[ ] Exact tie rule respected.
[ ] Phase38 reference generated.
[ ] No Test.
```

---

# 207. Phase38 handoff

Phase38 receives:

```text
s15_loss_winner.json
s15_reference_update.json
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
selected_loss
selected_huber_delta_if_applicable
current_max_epochs=50
current_patience=10
population_fingerprint
metric_version.
```

and changes only:

```text
max_epochs.
```

---

# 208. Reference reuse in Phase38

S15 winner already uses:

```text
E50.
```

Therefore normally:

```text
reuse S15 winner as E50 reference
train E100.
```

---

# 209. Relationship with Phase38 Epoch-cap sweep

If Huber converges later than MSE, record it.

Do not give Huber extra epochs inside S15.

Phase38 applies to selected loss only.

---

# 210. Relationship with Phase39 Gradient clipping sweep

If Huber wins and clipping is rare:

```text
still keep clip1 through Phase38.
```

Phase39 later tests clipping on the current selected configuration.

---

# 211. Relationship with Phase42 Candidate synthesis

Both MSE and Huber remain in Experiment Registry.

Do not delete the losing objective.

---

# 212. Relationship with Phase44 Rolling-origin robustness

S15 winner is based on one Validation period.

Loss robustness across time folds remains untested.

---

# 213. Relationship with Phase46 Multi-seed

S15 uses seed42 only.

A small MSE/Huber margin may reverse across seeds.

---

# 214. Relationship with Phase49 Residual analysis

S15 only includes limited Huber regime diagnostics.

Full residual distribution analysis remains Phase49.

---

# 215. Relationship with Phase50 Error-by-regime

Do not replace Phase50 with Huber's quadratic/linear regime.

They answer different questions:

```text
Huber regime:
defined by training-loss delta in model-space.

Phase50 error regimes:
project-defined analysis based on train-derived target/error context.
```

---

# 216. Reproducibility metadata

New Huber run records:

```text
run_id
seed
environment
device
feature fingerprint
X scaler checksum
target scaler checksum
target scaling ID
target model-space
Huber delta model-space
Huber raw-Wh equivalent delta
reduction
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
PE policy
POST_NORM
parameter count
architecture fingerprint
criterion config fingerprint
initialization fingerprint
sample-order provenance
optimizer-group policy
population fingerprint
Training Engine fingerprint
BEST checkpoint checksum
history checksum
metric checksum.
```

---

# 217. No fabricated outputs

Do not pre-fill:

```text
winner
RMSE
MAE
R²
best epoch
clipping fraction
gradient norms
Huber regime occupancy
runtime
raw-Wh delta for YS1 without loading scaler
```

before execution.

Allowed pre-runtime:

```text
Huber delta model-space = 1.0
loss definitions
selection/tie rules.
```

---

# 218. Phase sign-off

Create:

```text
phase_37_signoff.json
```

Minimum:

```text
phase = 37
phase_name = S15 Loss sweep
sweep_version
sweep_id
source_s14_winner_run_id
feature_variant_id
target_scaling_id
target_model_space
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
mse_reference_run_id
huber_run_id
huber_delta_model_space
huber_delta_raw_wh_equivalent
new_run_ids
reused_run_ids
parameter_count_equal
architecture_invariance_status
loss_scale_comparability_status
winner_loss_id
winner_loss_name
winner_huber_delta_if_applicable
winner_run_id
winner_rmse_wh
population_fingerprint
metric_version
gradient_diagnostics_status
clipping_diagnostics_status
huber_regime_diagnostics_status
initialization_audit_status
sample_order_match_status
inherited_warnings
test_status
approved_for_phase38
overall_status
created_at
```

---

# 219. Acceptance checklist

```text
[ ] Phase36 valid.
[ ] approved_for_phase37=true.
[ ] SWEEP_S15_LOSS-v1 declared.
[ ] S14 winner loaded.
[ ] All prior selected fields fixed.
[ ] L0=MSE exactly.
[ ] L1=Huber exactly.
[ ] Huber delta=1.0.
[ ] Delta is model-space.
[ ] Delta not tuned.
[ ] Raw-Wh equivalent delta recorded.
[ ] YS* same across losses.
[ ] Target scaler checksum same.
[ ] MSE reduction=mean.
[ ] Huber reduction=mean.
[ ] Correct MSE class.
[ ] Correct Huber class.
[ ] No SmoothL1 substitution.
[ ] Criterion formulas unit-tested.
[ ] Prediction/target shapes equal.
[ ] No loss broadcasting.
[ ] Training criterion uses y_model.
[ ] No inverse transform in backward path.
[ ] Evaluation inverse-transforms to Wh.
[ ] Same architecture fingerprint.
[ ] Same parameter count.
[ ] Same state_dict schema.
[ ] Same input projection.
[ ] Same MHA.
[ ] Same PE.
[ ] Same FFN.
[ ] Same LayerNorm.
[ ] Same pooling.
[ ] Same regression head.
[ ] Same Train IDs.
[ ] Same Validation IDs.
[ ] Same WINDOWPOP-v1.
[ ] Same feature fingerprint.
[ ] Same X scaler.
[ ] Same target transform.
[ ] Same batch.
[ ] Same AdamW.
[ ] Same optimizer-group policy.
[ ] Same LR.
[ ] Same WD.
[ ] Same dropout.
[ ] Same clip1.
[ ] Same E50/patience10.
[ ] Same min_delta.
[ ] Same scheduler/warmup policy.
[ ] Same accumulation1.
[ ] Same mixed precision policy.
[ ] Same seed42.
[ ] Same early-stop metric=Validation RMSE Wh.
[ ] Same BEST metric=Validation RMSE Wh.
[ ] Initialization policy same.
[ ] Exact init match checked if MSE provenance exists.
[ ] Sample-order policy same.
[ ] Dropout/RNG policy same.
[ ] No manual seed reset per batch.
[ ] MSE reference exact-match.
[ ] MSE not retrained.
[ ] Optional gradient probe uses disposable state.
[ ] Huber registered before training.
[ ] Huber fresh loaders/model/optimizer.
[ ] No warm-start.
[ ] No optimizer-state reuse.
[ ] No target clipping.
[ ] No residual clipping.
[ ] No outlier removal.
[ ] No sample weighting.
[ ] No hard-example mining.
[ ] No loss scheduler.
[ ] No delta scheduler.
[ ] No mixed objective.
[ ] No loss-specific LR.
[ ] No loss-specific WD.
[ ] No loss-specific batch.
[ ] Huber trained via TRAINING_ENGINE-v1.
[ ] Huber BEST verified.
[ ] MSE BEST provenance valid.
[ ] Predictions finite.
[ ] Raw criterion values marked non-cross-comparable.
[ ] Validation RMSE Wh cross-comparable.
[ ] Validation MAE Wh cross-comparable.
[ ] Validation R² cross-comparable.
[ ] Full-precision RMSE used.
[ ] MSE→Huber effect computed.
[ ] RMSE winner correct.
[ ] Exact tie=MSE.
[ ] MAE cannot override RMSE.
[ ] Gradient/clipping cannot override RMSE.
[ ] RMSE/R² ranking consistency checked.
[ ] MAE divergence recorded if present.
[ ] Huber regime diagnostics generated.
[ ] Quadratic/linear fractions use model-space residuals.
[ ] Huber regime not used as winner metric.
[ ] Gradient diagnostics generated.
[ ] Clipping diagnostics generated.
[ ] Convergence diagnostics generated.
[ ] Runtime diagnostics generated.
[ ] Optional generalization no Test.
[ ] No claim Huber removes outliers.
[ ] No claim delta1 always means1Wh.
[ ] Target-scaling interaction documented.
[ ] Delta limitation documented.
[ ] Loss×LR interaction documented.
[ ] Loss×clip interaction documented.
[ ] Loss×architecture interaction documented.
[ ] Loss×budget interaction documented.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] Inherited warnings propagated.
[ ] Winner artifact generated.
[ ] Phase38 reference update generated.
[ ] E50 reuse identified.
[ ] No MAE/LogCosh/Quantile additional candidate.
[ ] No score-based rerun.
[ ] No failed/SANITY run ranked.
[ ] No Test access.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 220. Acceptance criteria

Phase 37 chỉ PASS khi:

```text
S14-selected model/data/optimizer configuration is fixed.

Exactly MSE and Huber(delta=1.0 model-space) are represented.

Only training loss changes.

Huber delta is explicitly tied to the selected target model-space.

Raw-Wh equivalent delta is documented without Test leakage.

Delta is not tuned.

Both criteria use mean reduction.

Same YS*, Train/Validation IDs and WINDOWPOP-v1 are used.

Same architecture, parameter count and state_dict schema are verified.

Same optimizer/LR/WD/batch/dropout/clip/budget are used.

Early stopping and BEST selection remain Validation RMSE Wh for both.

Raw objective values are not used for cross-loss ranking.

MSE exact S14 reference is reused.

Huber is one fresh seed42 run.

No warm-start, optimizer-state reuse, outlier removal, clipping, weighting or loss scheduling occurs.

Huber BEST is verified.

Wh-space RMSE/MAE/R² are computed through METRICS-v1.

RMSE/R² ranking consistency is checked.

Huber regime and gradient/clipping diagnostics are generated as secondary evidence.

Validation RMSE Wh selects winner.

Exact RMSE tie selects MSE.

Phase38 reference is generated.

Test remains untouched.
```

---

# 221. Failure conditions

Phase 37 FAIL if:

```text
wrong S14 winner used

target scaling differs between conditions

Huber delta differs from 1.0

delta is tuned on Validation

delta target-space is undocumented

Huber implementation differs from registered definition

reduction differs

architecture/parameter schema differs

loss-specific LR/WD/batch/clip is used

early stopping metric changes to criterion loss

BEST checkpoint metric changes

raw MSE/Huber loss magnitude is used to choose winner

prediction/target broadcasting occurs

training loss is computed on raw Wh while YS1 is active

inverse transform appears in backward path

outliers are removed/clipped/weighted

loss/delta schedule is used

MSE is retrained and favorable rerun selected

Huber warm-starts from MSE

optimizer state is reused

population differs

one candidate fails but winner is still declared

RMSE is rounded before ranking

MAE or gradient stability overrides lower RMSE

Test is accessed.
```

---

# 222. Common mistakes

## 222.1 Huber delta không ghi target-space

Sai. `delta=1` vô nghĩa nếu không biết YS0/YS1.

## 222.2 Nếu YS1 thì nói delta=1 Wh

Sai.

## 222.3 Thử nhiều delta rồi chọn tốt nhất

Sai hidden hyperparameter tuning.

## 222.4 So sánh raw MSE loss với raw Huber loss

Sai scale/objective comparison.

## 222.5 Early stop MSE bằng RMSE nhưng Huber bằng Huber val loss

Sai fairness contract.

## 222.6 Huber dùng LR khác

Sai.

## 222.7 Tắt gradient clipping cho Huber

Sai.

## 222.8 Loại outlier trước khi dùng Huber

Sai multi-factor data change.

## 222.9 Dùng MAE thấp hơn để chọn Huber dù RMSE cao hơn

Sai primary metric.

## 222.10 Dùng `SmoothL1Loss` nhưng ghi là Huber mà không audit semantics

Sai provenance.

## 222.11 Thấy Huber thắng rồi tune delta

Không thuộc S15.

## 222.12 Thấy MSE thắng rồi thêm MAE candidate

Không thuộc S15.

## 222.13 Cho rằng Huber “không quan tâm outlier”

Sai. Large residuals vẫn đóng góp tuyến tính.

## 222.14 Dùng Test để quyết định loss

Forbidden.

---

# 223. Recommended execution pseudocode

```text
load_phase36_signoff()
assert_approved_for_phase37()

s14 = load_s14_winner()

freeze_all_prior_selected_fields(s14)

losses = {
    "L0": {
        "name": "MSE",
        "reduction": "mean"
    },
    "L1": {
        "name": "Huber",
        "delta": 1.0,
        "reduction": "mean"
    }
}

resolve_target_model_space(YS*)
resolve_huber_delta_raw_wh_equivalent(
    delta_model=1.0,
    target_scaling=YS*,
    frozen_y_scaler=...
)

audit_common_data_population()
audit_target_space()
audit_loss_scale_comparability()
audit_architecture_invariance()
audit_parameter_schema_equality()
audit_criterion_definitions()
run_criterion_unit_tests()
audit_training_config()
audit_initialization_policy()
audit_sample_order_policy()
audit_dropout_rng_policy()

mse_reference = resolve_s14_winner_run()
assert_exact_s15_reference_match(
    mse_reference,
    loss="MSE"
)

optional_first_batch_gradient_probe()

register_huber_run()

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

criterion = HuberLoss(
    delta=1.0,
    reduction="mean"
)

optimizer = build_fresh_adamw(
    model=model,
    lr=LR*,
    weight_decay=WD*,
    group_policy=FROZEN_S9_GROUP_POLICY
)

huber_result = TRAINING_ENGINE_v1.fit(
    model=model,
    criterion=criterion,
    early_stop_metric="validation_rmse_wh",
    best_checkpoint_metric="validation_rmse_wh",
    ...
)

verify_best_checkpoint(
    huber_result,
    loss="Huber",
    delta=1.0
)

results = {
    "MSE": mse_reference,
    "Huber": huber_result
}

metrics = build_verified_s15_wh_metrics(results)
effect = compute_mse_to_huber_effect(metrics)
gradient_diag = build_gradient_diagnostics(results)
clipping_diag = build_clipping_diagnostics(results)
huber_regimes = build_huber_regime_diagnostics(
    huber_result,
    delta_model=1.0
)
convergence = build_convergence_diagnostics(results)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer="MSE"
)

assert_rmse_r2_ranking_consistent(metrics)

write_hypothesis_outcomes()
write_findings()
write_s15_winner(winner)
write_phase38_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 224. Definition of Done

\[
\boxed{
One\ Fixed\ Model/Data\ Setup
+
Two\ Training\ Objectives
+
Explicit\ Huber\ Delta\ Semantics
+
One\ Fresh\ Huber\ Run
+
One\ Valid\ Reused\ MSE
+
Same\ Optimizer/Budget
+
Wh\text{-}Space\ Shared\ Evaluation
+
Verified\ BEST\ Metrics
+
Gradient/Regime\ Diagnostics
+
S15\ Winner
+
Phase38\ Reference
+
No\ Test
}
\]

---

# 225. Final status contract

```text
PHASE 37 tests training loss only.

Candidates:

L0:
MSELoss
reduction=mean

L1:
HuberLoss
delta=1.0 model-space
reduction=mean.

Frozen:
all winners from S1–S14
same YS*
same architecture
same parameter count/schema
same data/population
same batch
same AdamW
same LR
same WD
same dropout
same clip1
E50
patience10
seed42
Training Engine
Metric version.

Huber delta:
never tuned.

If YS0:
delta raw equivalent = 1 Wh.

If YS1:
delta raw equivalent = frozen train target scaler scale.

Training criterion:
model-space.

Evaluation:
raw Wh.

Cross-loss selection:
Validation RMSE Wh only.

Raw MSE vs Huber criterion values:
NOT numerically rank-comparable.

Early stopping:
Validation RMSE Wh for both.

BEST:
Validation RMSE Wh for both.

MSE:
reuse S14 winner if exact match.

Huber:
one fresh seed42 run.

Expected scientific differences:
gradient influence
clipping behavior
optimization trajectory
convergence
Validation performance.

No:
outlier removal
target/residual clipping
sample weighting
delta tuning
loss schedule
mixed objective
loss-specific LR/WD/batch/clip
warm-start
optimizer-state reuse.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer MSE.

No Test.

After SWEEP_S15_LOSS-v1 PASS:
proceed to
PHASE 38 — S16 Epoch-cap sweep.
```

---

# 226. Final check

Correct:

```text
Load S14 winner
→ freeze everything except loss
→ MSE vs Huber(delta1 model-space)
→ audit target-space/delta semantics
→ reuse MSE
→ fresh Huber
→ same RMSE-based early stopping
→ verify Huber BEST
→ compare Wh-space Validation RMSE
→ diagnose gradients/clipping/Huber regimes
→ select min RMSE
→ exact tie MSE
→ Phase38 reference
```

Incorrect:

```text
tune Huber delta
→ change LR/clip for Huber
→ remove outliers
→ compare raw train losses
→ early-stop on different metrics
→ choose by MAE despite worse RMSE
→ inspect Test
```

Chỉ sau khi `SWEEP_S15_LOSS-v1` được sign-off mới chuyển sang **PHASE 38 — S16 Epoch-cap sweep**.
