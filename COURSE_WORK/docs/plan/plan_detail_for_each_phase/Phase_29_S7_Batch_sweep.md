# PHASE 29 — S7 BATCH SWEEP

## Kế hoạch controlled sweep cho Training Mini-Batch Size của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S6_ACTIVATION-v1`  
**Sweep ID:** `S7_BATCH`  
**Output version:** `SWEEP_S7_BATCH-v1`  
**Phase trước:** `Phase_28_S6_Activation_sweep.md`

---

# 1. Vai trò của Phase 29

Phase 29 là controlled experiment thứ bảy trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target scaling, lookback, pooling, activation, Transformer architecture, optimizer, learning rate, weight decay, loss, early stopping, sample population và seed đã được khóa từ Phase 28, training mini-batch size nào phù hợp hơn: `B32` hay `B64`?

Phase 29 chỉ thay đúng một conceptual factor:

```text
TRAINING MINI-BATCH SIZE
```

với hai condition:

```text
B32 = 32 samples / training optimizer batch
B64 = 64 samples / training optimizer batch
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Factor
+
Same\ Samples
+
Same\ Model
+
Same\ Optimizer\ Hyperparameters
+
Controlled\ Batch\ Grouping
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí Phase 29 trong master execution plan

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
```

Phase 29 không được quay lại thay:

```text
feature set
time features
target scaling
lookback
pooling
activation
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

---

# 3. Câu hỏi nghiên cứu của S7

Phase 29 phải trả lời:

```text
1. B32 hay B64 tạo Validation RMSE Wh thấp hơn?

2. Batch size nhỏ hơn có tạo gradient updates nhiều/noisy hơn và giúp generalization không?

3. Batch size lớn hơn có tạo optimization trajectory ổn định hơn không?

4. Gradient norm và clipping fraction thay đổi thế nào?

5. Số optimizer updates trên mỗi epoch và toàn run khác nhau bao nhiêu?

6. Runtime trên mỗi epoch và time-to-best khác nhau thế nào?

7. MAE/R² có cùng ranking với RMSE không?

8. Batch size nào trở thành current reference cho Phase 30?
```

---

# 4. Current reference từ Phase 28

Phase 29 phải load:

```text
s6_activation_winner.json
s6_reference_update.json
phase_28_signoff.json
```

để resolve:

```text
FV* = selected feature variant
YS* = selected target scaling
L*  = selected lookback
P*  = selected pooling
A*  = selected activation
```

Phase 29 không hard-code:

```text
FS1_TF1
YS1
L144
LAST_STEP
GELU
```

vì các giá trị current phải đến từ sequential winner path.

---

# 5. Canonical S7 conditions

```text
B32
→ nominal training mini-batch size = 32

B64
→ nominal training mini-batch size = 64
```

Current reference từ Phase 28 đang ở:

```text
B64
```

nên đây là reference condition.

---

# 6. Scientific meaning của batch-size sweep

Batch size thay đổi cách cùng một Train population được chia thành các gradient-estimation groups.

Với dataset có `N_train` samples:

\[
steps\_per\_epoch(B)
=
\left\lceil
\frac{N_{train}}{B}
\right\rceil
\]

vì:

```text
drop_last = False
```

Do đó thông thường:

```text
B32
→ nhiều optimizer steps/epoch hơn

B64
→ ít optimizer steps/epoch hơn.
```

Đây là một phần tự nhiên của S7.

---

# 7. Important budget semantics

S7 sử dụng **fixed epoch budget**, không fixed optimizer-step budget.

Hard:

```text
max_epochs = 50
patience = 10
```

cho cả B32/B64.

Do đó:

```text
B32 và B64 có thể thực hiện số optimizer updates khác nhau.
```

Điều này không phải bug; nó là consequence của batch size dưới epoch-based training protocol.

---

# 8. Không equalize optimizer steps

Không được đổi:

```text
epochs_B32
epochs_B64
```

để cố làm total optimizer steps bằng nhau.

Nếu làm vậy sẽ thay thêm:

```text
training budget policy.
```

Current S7 question là:

> Batch size nào tốt hơn dưới cùng epoch/early-stopping protocol?

---

# 9. Không linear-scale learning rate

Hard:

```text
LR = 3e-4
```

cho cả B32 và B64.

Không áp dụng:

```text
linear scaling rule
square-root scaling rule
LR_B32 = LR_B64 / 2
```

trong Phase 29.

Learning rate sweep thuộc Phase 30.

---

# 10. Không compensate weight decay

Hard:

```text
weight_decay = 1e-4
```

cho cả hai.

AdamW applies parameter updates per optimizer step, nên batch size có thể interact với cumulative per-epoch optimization/regularization exposure.

Đây là interaction inherent dưới frozen optimizer settings.

Không điều chỉnh WD trong S7.

---

# 11. Không dùng gradient accumulation

Hard:

```text
gradient_accumulation_steps = 1
```

cho cả B32/B64.

Nếu dùng:

```text
B32 + accumulation 2
```

để tạo effective B64 thì đó không còn là B32 condition theo S7.

---

# 12. Effective batch size semantics

Với:

```text
gradient_accumulation_steps = 1
```

nominal effective training batch:

```text
B32 → 32
B64 → 64
```

trừ final partial batch khi:

```text
N_train mod B != 0.
```

---

# 13. `drop_last=False` phải giữ nguyên

Hard từ `DATALOADERS-v1`:

```text
drop_last = False
```

cho Train và evaluation.

Không được đổi sang:

```text
drop_last=True
```

để làm mọi batch đều cùng kích thước.

Mọi Train sample phải được nhìn thấy mỗi epoch.

---

# 14. Final partial batch là expected behavior

For each condition:

\[
last\_batch\_size
=
\begin{cases}
B, & N_{train}\bmod B = 0 \\
N_{train}\bmod B, & otherwise
\end{cases}
\]

Record actual runtime value.

Không được drop partial batch.

---

# 15. Train-loss aggregation

Do batch sizes và final partial batch khác nhau, epoch train loss phải dùng:

```text
sample-weighted aggregation
```

theo `TRAINING_ENGINE-v1`.

Không lấy simple mean của batch losses nếu batch sizes không đều.

Canonical:

\[
EpochLoss
=
\frac{
\sum_b n_b L_b
}{
\sum_b n_b
}
\]

nếu `L_b` là mean loss của batch `b`.

---

# 16. Validation metric aggregation

Validation metrics phải được tính global trên complete prediction bundle:

```text
all Validation samples
```

Không average batch RMSE/R².

Đây là hard contract từ `METRICS-v1`.

---

# 17. Evaluation batch-size semantics

Scientific swept factor của S7 là:

```text
training mini-batch size
```

Evaluation batch size không phải model hyperparameter.

Preferred design:

```text
Train:
B32 or B64

Validation:
fixed deterministic evaluation batch configuration
matching existing reference evaluation contract
```

Nếu current implementation dùng một shared `batch_size` field cho cả Train và Validation, Phase 29 phải audit và document exact semantics; không được âm thầm refactor giữa conditions.

Regardless of evaluation batching:

```text
Validation sample IDs
predictions
global metrics
```

phải independent of batch partition within numerical tolerance.

---

# 18. Evaluation batch-invariance check

Recommended on a fixed checkpoint:

```text
evaluate same model with eval B32
evaluate same model with eval B64
```

using same ordered Validation IDs.

Verify:

```text
sample_idx identical
predictions allclose
MAE/RMSE/R² allclose
```

This is an engineering sanity check.

It must not use Test.

---

# 19. Why evaluation-batch invariance is expected

Current model uses:

```text
LayerNorm
```

not BatchNorm.

No stateful recurrence or cross-sample operation exists.

Therefore eval predictions for a sample should not depend materially on how other samples are grouped in the batch.

---

# 20. No BatchNorm

Hard architecture context:

```text
Transformer uses LayerNorm
```

so no batch-statistics normalization is present.

This helps isolate batch size primarily to optimization rather than forward normalization semantics.

---

# 21. Working hypotheses

## H-S7-01 — Smaller batch may improve generalization

If:

\[
RMSE(B32)<RMSE(B64)
\]

this supports the empirical hypothesis that smaller-batch optimization is beneficial under current frozen LR/WD/training settings.

Potential mechanism:

```text
noisier gradient estimates
more optimizer updates per epoch
```

but S7 alone cannot identify which mechanism caused the difference.

Status:

```text
UNTESTED
```

---

# 22. H-S7-02 — Larger batch may be sufficiently stable/effective

If:

\[
RMSE(B64)\le RMSE(B32)
\]

this supports retaining the current larger batch under this training setup.

Do not conclude:

```text
B64 is universally optimal.
```

---

# 23. Preconditions bắt buộc

Phase 29 chỉ bắt đầu khi:

```text
Phase 28 = PASS
```

hoặc `PASS_WITH_WARNING` without unresolved critical issue.

Bắt buộc:

```text
approved_for_phase29 = true
```

and:

```text
s6_activation_winner.json
s6_reference_update.json
phase_28_signoff.json
```

---

# 24. Upstream contracts bắt buộc

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
```

---

# 25. Carry-forward warnings

Any unresolved non-critical upstream warning must be propagated, including possible:

```text
RANDOM_CONTROL_GAIN
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
MATCHED_INITIALIZATION_NOT_VERIFIABLE
```

Store in:

```text
S7 manifest
summary
report
winner
Phase 30 handoff.
```

---

# 26. Swept factor duy nhất

Canonical scientific field:

```text
train_batch_size
```

Allowed:

```text
32
64
```

Registry aliases:

```text
B32
B64
```

---

# 27. Frozen data configuration

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

# 28. Frozen model configuration

Hard:

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
Linear(d_model,1) regression head
no output activation
```

---

# 29. Frozen optimizer/training configuration

Except batch size:

```text
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

# 30. Same parameter count

Batch size does not modify model structure.

Hard expected:

```text
trainable_parameters_B32
==
trainable_parameters_B64
```

and same:

```text
parameter schema
state_dict keys
architecture fingerprint
```

---

# 31. Same model initialization opportunity

All model shapes/config fields are identical.

Recommended:

```text
initial_trainable_state_fingerprint_B32
==
initial_trainable_state_fingerprint_B64_reference
```

if B64 reference stores initial fingerprint.

If unavailable:

```text
NOT_VERIFIABLE
```

not FAIL.

---

# 32. Same sample order policy

Both conditions must use:

```text
same Train dataset order before shuffle
same sampler class
same shuffle=True
same seed policy
same worker policy
```

The only intended difference is grouping of that sample stream into batches of 32 vs 64.

---

# 33. Epoch-level permutation matching

Preferred high-quality S7 design:

For each comparable epoch, the ordered sequence of Train `sample_idx` before batch grouping should be identical across B32/B64.

Example:

```text
permutation:
[17, 5, 91, 2, ...]

B32:
first 32 | next 32 | ...

B64:
first 64 | next 64 | ...
```

This tightly isolates batching.

---

# 34. DataLoader permutation audit feasibility

If `DATALOADERS-v1` and run artifacts expose reproducible sample-order fingerprints, verify:

```text
epoch_1 order fingerprint
```

and ideally each completed common epoch.

If reference history does not retain permutation fingerprints:

```text
NOT_VERIFIABLE
```

rather than fabricating.

Future S7 new run should persist them if lightweight.

---

# 35. Do not change sample order to force artificial pairing unless compatible with DATALOADERS-v1

Phase 29 cannot silently replace:

```text
shuffle=True RandomSampler
```

with a new custom sampler if that changes the previously frozen loader semantics.

Any loader-contract change requires explicit protocol/version handling.

---

# 36. Worker policy fixed

Use exactly the existing correctness/reference worker configuration.

Baseline recommended from Phase 11:

```text
num_workers = 0
```

unless current reference contract explicitly differs.

Do not change workers to compensate runtime for B32.

---

# 37. Pin-memory/device loader settings fixed

Batch sweep must not change:

```text
pin_memory
persistent_workers
prefetch_factor
worker count
device transfer policy
```

between conditions.

---

# 38. Same device

Preferred hard fairness:

```text
B32 and B64 run on same device class/backend
```

for runtime comparison.

If scientific metric run must move device because environment changes:

```text
record discrepancy
runtime comparison becomes non-comparable
```

Metric comparison may remain valid only if environment protocol approves.

---

# 39. Existing B64 reference reuse

S6 winner is already trained at:

```text
B64
```

If exact S7 contract match:

```text
REUSE S6 winner
```

as B64 reference.

Do not retrain B64.

---

# 40. Normal S7 run count

Typically:

```text
B64
→ reused S6 winner

B32
→ one NEW training run
```

Therefore:

```text
1 new training run
+
1 reused reference.
```

---

# 41. B64 reference reuse gate

Exact match required on:

```text
FV*
YS*
L*
P*
A*
H1
WB0
WINDOWPOP-v1

B64
D64/H4/N2/FFN128
dropout .1

AdamW
LR3e-4
WD1e-4
MSE
E50
patience10
clip1
accumulation1
seed42

Training Engine
Metric version
```

Any mismatch:

```text
STOP.
```

---

# 42. Fresh B32 run

Execution:

```text
set seed 42
↓
fresh Train DataLoader with train batch=32
↓
fresh Validation DataLoader under fixed evaluation contract
↓
fresh Transformer
↓
same MSE
↓
same AdamW
↓
TRAINING_ENGINE-v1
```

No warm-start from B64.

---

# 43. Why no warm-start

Warm-start would compare:

```text
B32 fine-tuning from a B64-trained solution
```

against:

```text
B64 from scratch.
```

That does not isolate batch size.

---

# 44. No gradient accumulation

Re-emphasized:

```text
B32 remains actual B32 optimizer batch.
```

No accumulation to emulate B64.

---

# 45. No LR rescaling

Re-emphasized:

```text
same 3e-4
```

because LR is next sweep.

---

# 46. No epoch-cap rescaling

Do not run:

```text
B32 for 25 epochs
B64 for 50 epochs
```

to equalize approximate step count.

Max epochs remains 50.

---

# 47. Early stopping semantics fixed

Same:

```text
primary = Validation RMSE Wh
patience = 10
min_delta = 0
```

Batch size can change best epoch/stop epoch naturally.

---

# 48. Optimizer-step accounting

For each run record:

```text
train_samples_per_epoch
nominal_batch_size
actual_num_batches_per_epoch
last_batch_size
optimizer_steps_per_epoch
epochs_completed
total_optimizer_steps
samples_seen_total
```

If every complete epoch sees all samples:

\[
samples\_seen\_total
=
N_{train}\times epochs\_completed
\]

subject to no interrupted incomplete epoch entering official history.

---

# 49. Why total optimizer steps matters

If B32 wins, S7 alone cannot distinguish whether the advantage came from:

```text
smaller gradient batch
higher gradient noise
more updates per epoch
different Adam moment evolution
more clipping opportunities
different cumulative WD application
```

These are bundled consequences of batch size under fixed epoch-based training.

Document this explicitly.

---

# 50. AdamW interaction caveat

AdamW performs updates per optimizer step.

With different number of steps/epoch:

```text
cumulative update dynamics differ.
```

Therefore S7 result should be phrased:

> effect of batch size under fixed AdamW LR/WD and epoch-based training budget

not:

> pure effect of gradient noise only.

---

# 51. Gradient clipping comparison

Raw number of clipped batches is not directly comparable because B32 has more batches.

Primary clipping diagnostic:

```text
fraction_batches_clipped
```

not:

```text
count_clipped_batches
```

Count may also be reported with denominator.

---

# 52. Gradient-norm comparison

Gradient norm can vary with batch size and averaging.

Record:

```text
mean preclip norm
median preclip norm if available
max preclip norm
fraction clipped
```

Interpret cautiously.

---

# 53. Train loss comparability

Same:

```text
target scaling
criterion
samples
```

so sample-weighted train loss is comparable descriptively.

Do not compare simple mean-of-batch-loss if batch sizes differ.

---

# 54. Validation loss/metrics comparability

Validation metrics must be on identical sample IDs and original Wh space.

Hard.

---

# 55. Primary S7 selection metric

```text
best_validation_rmse_wh
```

from verified BEST checkpoint.

---

# 56. Secondary metrics

```text
validation_mae_wh
validation_r2
best_epoch
epochs_completed
stop_reason
sample-weighted train loss
gradient diagnostics
clipping fraction
optimizer steps
runtime
```

---

# 57. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{B32},
RMSE_{B64}
\right)
\]

Use full precision.

---

# 58. Exact RMSE tie rule

If exact full-precision equality:

```text
prefer B64
```

Rationale:

```text
fewer optimizer steps per epoch
lower training overhead in typical execution
existing reference retained
larger throughput potential
```

This is a predefined parsimony/engineering tie rule only.

Do not override a non-zero RMSE advantage for B32.

---

# 59. No arbitrary practical threshold

Do not invent:

```text
B32 must improve >1%
```

Strict lower RMSE wins unless exact tie.

---

# 60. Batch effect formula

Let:

```text
RMSE_32
RMSE_64
```

Define B64→B32 improvement:

\[
\Delta RMSE_{B64\rightarrow B32}
=
RMSE_{64}-RMSE_{32}
\]

Positive:

```text
B32 improves.
```

Relative:

\[
Improvement\%
=
100\times
\frac{RMSE_{64}-RMSE_{32}}
{RMSE_{64}}
\]

Also:

\[
\Delta MAE=MAE_{64}-MAE_{32}
\]

\[
\Delta R^2=R^2_{32}-R^2_{64}
\]

---

# 61. Metric ranking divergence

If:

```text
B32 wins RMSE
B64 wins MAE
```

record:

```text
METRIC_RANKING_DIVERGENCE
```

Winner remains RMSE-based.

---

# 62. No significance testing

Only one seed per batch condition.

Do not use:

```text
batches
epochs
sample residuals
```

as independent training-run replicates to test B32 vs B64 significance.

---

# 63. Best-checkpoint verification for B32

After training:

```text
fresh Transformer
↓
strict-load B32 BEST
↓
fresh ordered Validation loader
↓
full Validation population
↓
inverse target transform if YS1
↓
METRICS-v1
↓
verify BEST metrics.
```

---

# 64. B64 reference verification

No retrain.

Verify:

```text
BEST already verified
same sample population
same data/model/training config
B64 exact semantics
same metric version.
```

---

# 65. Evaluation batch invariance verification

Recommended use B32 BEST checkpoint:

```text
evaluate with eval batch 32
evaluate with eval batch 64
```

same order.

Verify prediction vectors allclose.

This test confirms evaluation batching itself does not materially alter metrics.

---

# 66. Numerical tolerance

Use existing model/device tolerance policy.

Do not demand universal bitwise equality across all backends.

Record:

```text
rtol
atol
max_abs_prediction_diff
metric_diff
```

---

# 67. No batch-dependent normalization state

Verify model module tree contains no:

```text
BatchNorm1d
BatchNorm2d
BatchNorm3d
SyncBatchNorm
```

in scientific Transformer.

LayerNorm is expected.

This strengthens interpretation that batch size acts through training mini-batches rather than normalization statistics.

---

# 68. Batch-shape sanity tests

Before official B32 run, verify model/training engine handles:

```text
B=32
B=64
B=1
final partial batch
```

with output shape:

```text
[B,1]
```

and finite loss/gradients.

---

# 69. Final partial-batch sanity

Build a synthetic/loader case where:

```text
N % B != 0
```

Verify:

```text
last batch processed
loss sample-weighted correctly
all sample IDs covered once
no duplicate/drop.
```

---

# 70. Coverage audit

For each training epoch:

```text
count(sample_idx) = N_train
unique(sample_idx) = N_train
```

No missing/duplicate sample within an epoch.

For Validation:

```text
complete ordered coverage
```

once per evaluation.

---

# 71. Optimizer step count audit

Expected:

```text
steps_per_epoch = number of Train batches
```

because:

```text
accumulation = 1
one optimizer.step per batch
```

except technical skipped/failed batches, which are forbidden.

---

# 72. Zero-grad/step ordering fixed

Training Engine must preserve:

```text
zero_grad
forward
loss
backward
clip
optimizer.step
```

same as prior runs.

Batch sweep cannot alter ordering.

---

# 73. Gradient clipping timing fixed

Still:

```text
after backward
before optimizer.step
```

for both B32/B64.

---

# 74. No scheduler

Hard:

```text
scheduler = None.
```

Otherwise differing steps per epoch could create scheduler-semantic confounding.

Current absence of scheduler keeps S7 interpretation cleaner.

---

# 75. If scheduler were unexpectedly present

Hard failure:

```text
TRAINING_CONFIG_DRIFT
```

because per-step vs per-epoch scheduler behavior would interact strongly with batch count.

---

# 76. Runtime measurement

Record separately:

```text
data loading time optional
train epoch duration
validation duration
total run duration
time to BEST
```

Use same device/environment if runtime comparison is reported.

---

# 77. Throughput diagnostics

Recommended:

```text
train_samples_per_second
optimizer_steps_per_second
mean_batch_duration_ms
```

if instrumentation is already available and low overhead.

Engineering only.

---

# 78. Do not benchmark with attention extraction

Attention collection must remain OFF.

---

# 79. Device synchronization for timing

If accurate CUDA/MPS timing is implemented, use existing environment/runtime timing policy.

Do not add backend-specific synchronization logic to only one condition.

If timing is coarse Python wall-clock, document it.

---

# 80. Experiment identity

```text
sweep_id = S7_BATCH
experiment_family = TRANSFORMER_SWEEP_S7_BATCH
sweep_version = SWEEP_S7_BATCH-v1
```

Recommended labels:

```text
S7_<FV*>__<YS*>__<L*>__<P*>__<A*>__B64__REFERENCE_S6

S7_<FV*>__<YS*>__<L*>__<P*>__<A*>__B32__S42
```

Registry `run_id` remains authoritative.

---

# 81. Run matrix

Create:

```text
s7_run_matrix.csv
```

Fields:

```text
sweep_id
batch_id
train_batch_size
eval_batch_size
source_type
source_run_id
requires_new_training
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
population_fingerprint
feature_fingerprint
x_scaler_id
target_transform_id
gradient_accumulation_steps
seed
model_config_id
training_config_id
status
```

---

# 82. Sweep manifest

Create:

```text
s7_batch_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S7_BATCH-v1
sweep_id = S7_BATCH
source_s6_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
candidate_batches = [B32,B64]
new_runs_required
reused_runs
swept_field = train_batch_size
evaluation_batch_semantics
budget_semantics = FIXED_EPOCH_BUDGET
gradient_accumulation = 1
drop_last = false
frozen_fields
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = B64_ON_EXACT_RMSE_TIE
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

# 83. Sweep contract

Create:

```text
s7_batch_sweep_contract.json
```

Must state:

```text
Only training mini-batch size changes.

B32 = 32.
B64 = 64.

Fixed epoch budget.
No optimizer-step equalization.
No LR scaling.
No WD compensation.
No gradient accumulation.
drop_last=False.
All Train samples seen each complete epoch.

Same data/sample population.
Same architecture.
Same optimizer hyperparameters.
Same seed policy.

B64 reference reused if exact match.
B32 one fresh run.

Selection:
verified Validation RMSE Wh.

Exact tie:
B64.

Test forbidden.
```

---

# 84. Preflight audit

Create:

```text
s7_batch_preflight_audit.csv
```

Checks:

```text
phase28_pass
approved_for_phase29
s6_winner_valid
FV_locked
YS_locked
L_locked
P_locked
A_locked
B32_registered
B64_registered
population_fixed
drop_last_false
gradient_accumulation_one
LR_fixed
WD_fixed
epoch_budget_fixed
Training_Engine_fixed
Metric_fixed
seed_fixed
test_locked
status
```

---

# 85. Batch-definition audit

Create:

```text
s7_batch_definition_audit.csv
```

Fields:

```text
batch_id
nominal_train_batch_size
gradient_accumulation_steps
nominal_effective_batch_size
drop_last
train_sample_count
steps_per_epoch_expected
last_batch_size_expected
budget_semantics
status
```

Runtime expected counts derived from actual `N_train`.

---

# 86. Common-data audit

Create:

```text
s7_common_data_audit.csv
```

Fields:

```text
split_id
sample_count_b32
sample_count_b64
sample_ids_equal
ordered_base_dataset_ids_equal
feature_fingerprint_equal
x_scaler_equal
target_transform_equal
lookback_equal
pooling_equal
activation_equal
population_fingerprint_equal
status
```

---

# 87. DataLoader audit

Create:

```text
s7_dataloader_audit.csv
```

Fields:

```text
batch_id
train_batch_size
eval_batch_size
shuffle_train
shuffle_eval
drop_last_train
drop_last_eval
num_workers
pin_memory
sampler_type
generator_seed
worker_seed_policy
gradient_accumulation
status
```

---

# 88. Sample-order audit

Create:

```text
s7_sample_order_audit.csv
```

Fields:

```text
epoch_or_probe
b32_order_fingerprint
b64_order_fingerprint
reference_available
same_underlying_sample_sequence
batch_grouping_only_difference
status
```

If exact B64 historical order unavailable:

```text
NOT_VERIFIABLE
```

not FAIL.

---

# 89. Batch coverage audit

Create:

```text
s7_batch_coverage_audit.csv
```

Fields:

```text
batch_id
epoch
expected_samples
observed_samples
unique_samples
duplicate_count
missing_count
num_batches
last_batch_size
coverage_complete
status
```

Full per-epoch table may be large but manageable.

---

# 90. Architecture audit

Create:

```text
s7_batch_architecture_audit.csv
```

Fields:

```text
batch_id
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
batchnorm_module_count
architecture_matches_reference
status
```

Expected:

```text
batchnorm_module_count = 0.
```

---

# 91. Training-config audit

Create:

```text
s7_batch_training_audit.csv
```

Fields:

```text
batch_id
train_batch_size
optimizer
learning_rate
weight_decay
criterion
max_epochs
patience
min_delta
gradient_clip
gradient_accumulation
scheduler
mixed_precision
seed
training_engine_version
only_batch_differs
status
```

---

# 92. Initialization audit

Create:

```text
s7_initialization_audit.csv
```

Fields:

```text
batch_id
initial_model_state_fingerprint
reference_available
parameter_schema_match
fingerprint_matches
seed
status
```

---

# 93. Evaluation-batch invariance artifact

Create:

```text
s7_eval_batch_invariance_audit.csv
```

Fields:

```text
checkpoint_run_id
eval_batch_a
eval_batch_b
sample_ids_equal
max_abs_prediction_diff_wh
mae_diff
rmse_diff
r2_diff
rtol
atol
allclose
status
```

This is a diagnostic/sanity artifact, not the S7 winner table.

---

# 94. Optimizer-budget diagnostics

Create:

```text
s7_optimizer_budget_diagnostics.csv
```

Fields:

```text
batch_id
train_samples_per_epoch
nominal_batch_size
num_batches_per_epoch
last_batch_size
optimizer_steps_per_epoch
epochs_completed
total_optimizer_steps
samples_seen_total
updates_per_1000_samples
status
```

---

# 95. Why `updates_per_1000_samples` helps

Descriptive:

\[
updates\_per\_1000
=
1000
\times
\frac{optimizer\_steps\_per\_epoch}{N_{train}}
\]

This makes update-frequency difference explicit.

Do not use as winner criterion.

---

# 96. Run provenance

Create:

```text
s7_batch_run_provenance.csv
```

Fields:

```text
batch_id
run_id
source_type
source_phase
config_fingerprint
parameter_schema_fingerprint
feature_fingerprint
population_fingerprint
initial_state_fingerprint
best_checkpoint_sha256
history_sha256
metric_artifact
prediction_artifact
status
```

---

# 97. Primary metrics table

Create:

```text
s7_batch_metrics.csv
```

Rows:

```text
B32
B64
```

Fields:

```text
batch_id
train_batch_size
run_id
source_type
optimizer_steps_per_epoch
best_epoch
epochs_completed
total_optimizer_steps_to_stop
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

# 98. Batch effect table

Create:

```text
s7_batch_effect.csv
```

Fields:

```text
b32_run_id
b64_run_id
b32_rmse_wh
b64_rmse_wh
rmse_delta_b64_to_b32_wh
rmse_improvement_pct
b32_mae_wh
b64_mae_wh
mae_delta_b64_to_b32_wh
b32_r2
b64_r2
r2_delta_b64_to_b32
rmse_winner
metric_ranking_divergence
status
```

---

# 99. Optimization diagnostics

Create:

```text
s7_optimization_diagnostics.csv
```

Fields:

```text
batch_id
best_epoch
last_epoch
stop_reason
best_rmse_wh
last_rmse_wh
mean_grad_norm_preclip
median_grad_norm_preclip_optional
max_grad_norm_preclip
mean_fraction_batches_clipped
max_fraction_batches_clipped
clipped_batch_count
total_batch_count
nonfinite_events
status
```

Interpret fractions before raw counts.

---

# 100. Runtime diagnostics

Create:

```text
s7_runtime_diagnostics.csv
```

Fields:

```text
batch_id
device
epochs_completed
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
mean_train_samples_per_second_optional
mean_optimizer_steps_per_second_optional
time_to_best_seconds_optional
runtime_comparable
status
```

---

# 101. Update-normalized learning diagnostics

Recommended:

```text
s7_update_axis_diagnostics.csv
```

At each completed epoch:

```text
batch_id
epoch
cumulative_optimizer_steps
validation_rmse_wh
validation_mae_wh
validation_r2
```

This enables plotting Validation RMSE versus cumulative optimizer steps.

Diagnostic only.

---

# 102. Why update-axis plots are useful

Epoch-axis plot answers:

> Under the same number of passes over data, which condition learns better?

Update-axis plot gives context:

> At comparable optimizer-update counts, how do trajectories look?

But because Validation is only measured at epoch boundaries, exact step-matched comparison may not exist.

Do not interpolate a scientific winner.

---

# 103. No interpolation of Validation metrics by step

Do not create fake:

```text
RMSE at step 1000
```

by interpolation between epochs and treat it as observed.

Plots may connect points visually, but analysis must know observations occur at epoch boundaries.

---

# 104. Hypothesis outcomes

Create:

```text
s7_hypothesis_outcomes.csv
```

Fields:

```text
hypothesis_id
source_hypothesis_id_optional
comparison
expected_direction
b32_rmse
b64_rmse
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

single-seed Validation context only.

---

# 105. Findings artifact

Create:

```text
s7_batch_findings.csv
```

Possible codes:

```text
B32_GAIN
B64_GAIN
BATCH_EXACT_TIE
METRIC_RANKING_DIVERGENCE
MORE_UPDATES_B32
GRADIENT_NORM_DIFFERENCE
CLIPPING_FRACTION_DIFFERENCE
CONVERGENCE_DIFFERENCE
RUNTIME_THROUGHPUT_DIFFERENCE
EVAL_BATCH_INVARIANCE_VERIFIED
PARAMETER_EQUALITY_VERIFIED
MATCHED_INITIALIZATION_VERIFIED
MATCHED_INITIALIZATION_NOT_VERIFIABLE
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
INHERITED_WARNING
```

---

# 106. Winner artifact

Create:

```text
s7_batch_winner.json
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
selection_metric
selection_direction
tie_rule
budget_semantics
winner_batch_id
winner_train_batch_size
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_batch_id
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
optimizer_steps_per_epoch_winner
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 107. Reference update for Phase 30

Create:

```text
s7_reference_update.json
```

Minimum:

```text
previous_reference_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
previous_batch_id = B64
selected_batch_id
winner_run_id
winner_config_fingerprint
winner_rmse_wh
learning_rate_state = LR2_3E-4
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase30
```

---

# 108. Phase 30 handoff logic

Phase 30 tests:

```text
LR1 = 1e-4
LR2 = 3e-4
LR3 = 1e-3
```

while holding S7 winner batch fixed.

Current S7 winner already uses:

```text
LR2 = 3e-4
```

Therefore Phase 30 should normally:

```text
reuse S7 winner as LR2 reference

train new:
LR1
LR3
```

if exact match.

---

# 109. Learning curves

Recommended figures:

```text
S7_01_validation_rmse_by_epoch.png
S7_02_validation_mae_by_epoch.png
S7_03_train_loss_by_epoch.png
S7_04_gradient_norm_by_epoch.png
S7_05_gradient_clipping_fraction.png
S7_06_validation_rmse_by_optimizer_step.png
S7_07_best_validation_metrics.png
S7_08_runtime_throughput_context.png
```

---

# 110. Primary figure

```text
S7_01_validation_rmse_by_epoch.png
```

Overlay:

```text
B32
B64
```

with best epoch markers.

---

# 111. Update-axis diagnostic figure

```text
S7_06_validation_rmse_by_optimizer_step.png
```

Plot observed epoch-end points against:

```text
cumulative_optimizer_steps
```

No interpolation-based ranking.

---

# 112. Runtime figure

```text
S7_08_runtime_throughput_context.png
```

Engineering context only.

Do not select winner by throughput.

---

# 113. No attention visualization

Not Phase 29.

Attention extraction stays OFF during training.

---

# 114. Interpretation if B32 wins

Safe:

> Under the frozen AdamW learning rate, weight decay and epoch-based training budget, batch size 32 achieved lower Validation RMSE than batch size 64.

Do not reduce this automatically to:

```text
gradient noise caused better generalization.
```

B32 also had more optimizer updates per epoch.

---

# 115. Interpretation if B64 wins

Safe:

> Under the same epoch-based training protocol and optimizer hyperparameters, batch size 64 achieved lower Validation RMSE.

Do not claim larger batches are universally better.

---

# 116. Interpretation if exact tie

Choose:

```text
B64
```

by predefined engineering/parsimony tie rule.

Report no full-precision RMSE advantage for B32.

---

# 117. Tiny non-zero margin

Strict lower RMSE wins.

Document:

```text
small single-seed Validation margin
```

and avoid strong claims.

---

# 118. Single-seed limitation

Mandatory:

```text
S7 conditions represented by seed 42.
```

No mean±std.

---

# 119. Validation-only limitation

Mandatory:

```text
S7 winner is a development selection based on Validation.
```

No Test evidence.

---

# 120. Interaction limitation

Batch size interacts with:

```text
learning rate
weight decay
gradient clipping
dropout
optimizer dynamics
```

S7 estimates the effect under current frozen settings only.

This is especially important because:

```text
Phase 30 = LR sweep
Phase 31 = WD sweep
Phase 39 = clipping sweep.
```

---

# 121. Sequential-selection limitation

By Phase 29, Validation has influenced:

```text
S1 Feature set
S2 Time features
S3 Target scaling
S4 Lookback
S5 Pooling
S6 Activation
S7 Batch
```

Thus all experiments must remain transparent and registered.

---

# 122. No B16/B128

Do not add:

```text
B16
B128
B256
```

inside S7.

Registered options are exactly:

```text
B32
B64.
```

---

# 123. No dynamic batch size

Do not change batch size by epoch.

---

# 124. No batch-size warmup

No.

---

# 125. No gradient accumulation sweep

No.

---

# 126. No LR scaling

No.

---

# 127. No WD scaling

No.

---

# 128. No epoch-budget normalization by steps

No.

---

# 129. No drop_last change

No.

---

# 130. No loss change

MSE fixed.

---

# 131. No clipping change

Clip 1.0 fixed.

---

# 132. No model architecture change

Hard.

---

# 133. No Test access

Hard.

---

# 134. Run failure policy

If B32 run fails technically:

```text
S7 incomplete.
```

Do not automatically declare B64 winner.

---

# 135. Numerical failure

NaN/Inf:

```text
FAIL run.
```

No skipped batch.

---

# 136. OOM policy

B32 should not require more activation memory per batch than B64.

If B32 OOMs while B64 reference succeeded:

```text
investigate implementation/environment anomaly.
```

Do not alter config.

---

# 137. Technical rerun policy

Allowed only for documented:

```text
interrupt
hardware/software failure
corrupt checkpoint/artifact
```

Not because score is poor.

---

# 138. Score-based rerun forbidden

No retraining B32/B64 until a favorable score appears.

---

# 139. B64 duplicate-run prohibition

Do not create another B64 run merely because current B64 reference score seems weak.

Reuse exact reference.

---

# 140. Discrepancy taxonomy

```text
S6_REFERENCE_MISSING
S6_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
POOLING_DRIFT
ACTIVATION_DRIFT
BATCH_DEFINITION_MISMATCH
GRADIENT_ACCUMULATION_DRIFT
DROP_LAST_DRIFT
SAMPLE_COVERAGE_FAILURE
SAMPLE_ORDER_POLICY_DRIFT
EVAL_BATCH_SEMANTICS_DRIFT
PARAMETER_COUNT_MISMATCH
PARAMETER_SCHEMA_MISMATCH
BATCHNORM_PRESENT
POPULATION_MISMATCH
TARGET_ID_MISMATCH
X_SCALER_MISMATCH
TARGET_SCALER_MISMATCH
ARCHITECTURE_DRIFT
LEARNING_RATE_DRIFT
WEIGHT_DECAY_DRIFT
EPOCH_BUDGET_DRIFT
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
TRAIN_LOSS_AGGREGATION_ERROR
OPTIMIZER_STEP_ACCOUNTING_ERROR
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 141. Discrepancy log

Create:

```text
s7_batch_discrepancies.json
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

# 142. Severity examples

```text
CRITICAL:
different sample population
drop_last=True
LR scaling in only one condition
Test access
train sample coverage failure

MAJOR:
wrong batch size
gradient accumulation !=1
train loss aggregation incorrect
checkpoint verification failure

MODERATE:
sample-order exact match unavailable
metric ranking divergence
large runtime difference

INFO:
initial-state fingerprint unavailable
evaluation batch invariance tolerance note
```

---

# 143. Status model

## PASS

```text
B64 reference valid
B32 exact condition valid
same samples/model/optimizer config
coverage complete
optimizer budget recorded
B32 BEST verified
winner selected
Phase 30 reference generated
Test untouched
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
metric ranking divergence
sample-order match not verifiable
initialization match not verifiable
large throughput difference
inherited upstream warning
```

with core methodology valid.

## FAIL

Examples:

```text
sample population mismatch
drop_last drift
LR/WD compensation
gradient accumulation changed
wrong batch definition
B32 unresolved failure
Test access.
```

---

# 144. Sweep summary

Create:

```text
s7_batch_sweep_summary.json
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
reference_run_id
b32_run_id
new_runs
reused_runs
batch_semantics
budget_semantics
primary_metric
metrics_by_batch
batch_effect
optimizer_budget_diagnostics
optimization_diagnostics
runtime_diagnostics
evaluation_batch_invariance
parameter_equality
initialization_match
sample_order_match
winner
winner_margin
inherited_warnings
phase30_reference
test_status
overall_status
```

---

# 145. Human-readable report

Create:

```text
s7_batch_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S6
3. B32/B64 definitions
4. Fixed-epoch budget semantics
5. Controlled-variable contract
6. Same population/data fairness
7. DataLoader/batch coverage audit
8. Parameter/model fairness
9. Optimizer-step accounting
10. Validation metrics
11. Batch-size effect
12. Gradient/clipping diagnostics
13. Epoch-axis learning curves
14. Update-axis diagnostic view
15. Runtime/throughput context
16. S7 winner
17. Interpretation cautions
18. Limitations
19. Phase 30 handoff
```

---

# 146. Report wording

Use:

```text
Observed
Interpretation
Limitation
Handoff
```

Example safe wording:

> B32 achieved lower Validation RMSE under the same AdamW LR/WD and fixed epoch budget, while also performing more optimizer updates per epoch.

Avoid:

> B32 wins because noisy gradients generalize better.

The mechanism is not isolated.

---

# 147. README

Create:

```text
README_S7_BATCH_SWEEP.md
```

Must explain:

```text
Purpose
S6 winner handoff
B32/B64 definitions
training vs evaluation batch semantics
fixed-epoch budget
optimizer-step difference
no LR scaling
no WD compensation
no accumulation
drop_last=False
sample-weighted train loss
same population
reference reuse
winner/tie rules
interaction limitations
Phase 30 handoff
No Test
```

---

# 148. Output directory

```text
artifacts/
└── sweeps/
    └── S7_batch/
        ├── s7_batch_sweep_manifest.json
        ├── s7_batch_sweep_contract.json
        ├── s7_batch_preflight_audit.csv
        ├── s7_run_matrix.csv
        ├── s7_batch_definition_audit.csv
        ├── s7_common_data_audit.csv
        ├── s7_dataloader_audit.csv
        ├── s7_sample_order_audit.csv
        ├── s7_batch_coverage_audit.csv
        ├── s7_batch_architecture_audit.csv
        ├── s7_batch_training_audit.csv
        ├── s7_initialization_audit.csv
        ├── s7_eval_batch_invariance_audit.csv
        ├── s7_optimizer_budget_diagnostics.csv
        ├── s7_batch_run_provenance.csv
        ├── s7_batch_metrics.csv
        ├── s7_batch_effect.csv
        ├── s7_optimization_diagnostics.csv
        ├── s7_runtime_diagnostics.csv
        ├── s7_update_axis_diagnostics.csv
        ├── s7_hypothesis_outcomes.csv
        ├── s7_batch_findings.csv
        ├── s7_batch_winner.json
        ├── s7_reference_update.json
        ├── s7_batch_sweep_tests.csv
        ├── s7_batch_discrepancies.json
        ├── s7_batch_sweep_summary.json
        ├── s7_batch_sweep_report.md
        ├── figures/
        │   ├── S7_01_validation_rmse_by_epoch.png
        │   ├── S7_02_validation_mae_by_epoch.png
        │   ├── S7_03_train_loss_by_epoch.png
        │   ├── S7_04_gradient_norm_by_epoch.png
        │   ├── S7_05_gradient_clipping_fraction.png
        │   ├── S7_06_validation_rmse_by_optimizer_step.png
        │   ├── S7_07_best_validation_metrics.png
        │   └── S7_08_runtime_throughput_context.png
        ├── README_S7_BATCH_SWEEP.md
        └── phase_29_signoff.json
```

New B32 run remains under:

```text
artifacts/runs/<run_id>/
```

Do not duplicate checkpoints in sweep folder.

---

# 149. Required outputs

```text
O29.1  Sweep manifest
O29.2  Sweep contract
O29.3  Preflight audit
O29.4  Run matrix
O29.5  Batch-definition audit
O29.6  Common-data audit
O29.7  DataLoader audit
O29.8  Sample-order audit
O29.9  Batch-coverage audit
O29.10 Architecture audit
O29.11 Training-config audit
O29.12 Initialization audit
O29.13 Evaluation-batch invariance audit
O29.14 Optimizer-budget diagnostics
O29.15 Run provenance
O29.16 Reused B64 reference
O29.17 Verified new B32 run
O29.18 Metrics table
O29.19 Batch effect table
O29.20 Optimization diagnostics
O29.21 Runtime diagnostics
O29.22 Update-axis diagnostics
O29.23 Hypothesis outcomes
O29.24 Findings
O29.25 Winner artifact
O29.26 Phase 30 reference update
O29.27 Figures
O29.28 Sweep test suite
O29.29 Discrepancy log
O29.30 Sweep summary
O29.31 Human-readable report
O29.32 README
O29.33 Phase sign-off
```

---

# 150. Sweep test suite

Create:

```text
s7_batch_sweep_tests.csv
```

Recommended checks:

```text
S7T29-001 Phase 28 PASS/non-critical warning only
S7T29-002 approved_for_phase29 = true
S7T29-003 S6 winner valid
S7T29-004 feature variant fixed
S7T29-005 target scaling fixed
S7T29-006 lookback fixed
S7T29-007 pooling fixed
S7T29-008 activation fixed
S7T29-009 B32 registered
S7T29-010 B64 registered
S7T29-011 B32 train batch = 32
S7T29-012 B64 train batch = 64
S7T29-013 gradient accumulation = 1 for both
S7T29-014 nominal effective batch semantics correct
S7T29-015 drop_last=False
S7T29-016 fixed epoch budget declared
S7T29-017 max_epochs=50 fixed
S7T29-018 patience=10 fixed
S7T29-019 no optimizer-step budget equalization
S7T29-020 no LR scaling
S7T29-021 LR3e-4 fixed
S7T29-022 no WD compensation
S7T29-023 WD1e-4 fixed
S7T29-024 MSE fixed
S7T29-025 clip1 fixed
S7T29-026 scheduler=None
S7T29-027 mixed precision fixed
S7T29-028 seed42 fixed
S7T29-029 same Train IDs
S7T29-030 same Validation IDs
S7T29-031 same WINDOWPOP-v1
S7T29-032 same feature fingerprint
S7T29-033 same X scaler
S7T29-034 same target transform/scaler
S7T29-035 same lookback
S7T29-036 same pooling
S7T29-037 same activation
S7T29-038 same H1
S7T29-039 same WB0
S7T29-040 same D64
S7T29-041 same H4
S7T29-042 same N2
S7T29-043 same FFN128
S7T29-044 same dropout .1
S7T29-045 same PE
S7T29-046 same norm policy
S7T29-047 no causal mask
S7T29-048 no padding mask
S7T29-049 same regression head
S7T29-050 same parameter count
S7T29-051 same parameter schema
S7T29-052 no BatchNorm modules
S7T29-053 same sampler class
S7T29-054 same shuffle policy
S7T29-055 same worker policy
S7T29-056 same pin-memory/device loading policy
S7T29-057 B32 coverage complete
S7T29-058 B64 reference coverage complete if artifact available
S7T29-059 no duplicated Train sample within epoch
S7T29-060 no missing Train sample within epoch
S7T29-061 final partial batch processed
S7T29-062 train loss sample-weighted
S7T29-063 Validation metrics globally aggregated
S7T29-064 optimizer steps/epoch computed correctly
S7T29-065 total optimizer steps computed correctly
S7T29-066 B32 has expected grouping semantics
S7T29-067 sample-order match checked if available
S7T29-068 initialization match checked if available
S7T29-069 B64 reference exact-match
S7T29-070 B64 reference reused
S7T29-071 B32 registered before training
S7T29-072 B32 fresh loaders/model
S7T29-073 B32 no warm-start
S7T29-074 B32 trained via TRAINING_ENGINE-v1
S7T29-075 B32 BEST verified
S7T29-076 B64 BEST already verified
S7T29-077 evaluation-batch invariance checked
S7T29-078 both result rows use verified BEST
S7T29-079 full-precision RMSE available
S7T29-080 RMSE effect correct
S7T29-081 relative improvement correct
S7T29-082 MAE effect correct
S7T29-083 R² effect correct
S7T29-084 winner=min RMSE
S7T29-085 exact tie=B64
S7T29-086 metric divergence recorded if present
S7T29-087 clipping fraction used before raw clipping count
S7T29-088 optimization diagnostics generated
S7T29-089 runtime diagnostics generated
S7T29-090 update-axis diagnostics generated
S7T29-091 no interpolated metric used for selection
S7T29-092 hypotheses recorded
S7T29-093 findings generated
S7T29-094 winner points to valid run
S7T29-095 Phase 30 reference update generated
S7T29-096 LR2 reuse identified for Phase 30
S7T29-097 inherited warnings propagated
S7T29-098 no B16/B128 experiment
S7T29-099 no dynamic batch size
S7T29-100 no score-based rerun
S7T29-101 no failed/SANITY run in ranking
S7T29-102 no attention extraction during training
S7T29-103 no Test access
S7T29-104 single-seed limitation documented
S7T29-105 Validation-only limitation documented
S7T29-106 batch/LR/WD interaction limitation documented
S7T29-107 epoch-vs-step budget caveat documented
S7T29-108 figures source-derived
S7T29-109 summary/report generated
S7T29-110 phase sign-off generated
```

---

# 151. Recommended notebook structure

```text
Cell 29.1  Phase title
Cell 29.2  Verify Phase 28 sign-off
Cell 29.3  Declare SWEEP_S7_BATCH-v1
Cell 29.4  Load S6 winner/reference update
Cell 29.5  Freeze FV*/YS*/L*/P*/A*
Cell 29.6  Resolve B32/B64 definitions
Cell 29.7  Declare fixed-epoch budget semantics
Cell 29.8  Build S7 run matrix
Cell 29.9  Audit same data/population
Cell 29.10 Audit model/parameter equality
Cell 29.11 Audit DataLoader configuration
Cell 29.12 Audit drop_last/coverage semantics
Cell 29.13 Compute expected batches/epoch
Cell 29.14 Audit fixed LR/WD/clip/accumulation
Cell 29.15 Check sample-order fingerprint if available
Cell 29.16 Check initialization fingerprint if available
Cell 29.17 Verify B64 reference reuse eligibility
Cell 29.18 Register B32 run
Cell 29.19 Seed + fresh B32 loaders/model
Cell 29.20 Execute B32 via Training Engine
Cell 29.21 Verify B32 BEST
Cell 29.22 Run evaluation-batch invariance sanity
Cell 29.23 Build run provenance
Cell 29.24 Build optimizer-budget diagnostics
Cell 29.25 Build S7 metrics table
Cell 29.26 Compute B64→B32 effect
Cell 29.27 Build gradient/clipping diagnostics
Cell 29.28 Build runtime/throughput diagnostics
Cell 29.29 Build update-axis diagnostics
Cell 29.30 Evaluate hypotheses
Cell 29.31 Generate learning curves
Cell 29.32 Generate findings
Cell 29.33 Select S7 winner
Cell 29.34 Write winner JSON
Cell 29.35 Write Phase 30 reference update
Cell 29.36 Run S7 tests/discrepancies
Cell 29.37 Write summary/report
Cell 29.38 Register artifacts/checksums
Cell 29.39 Write README
Cell 29.40 Phase sign-off
```

---

# 152. Execution flow

```text
Verify Phase 28
        ↓
Load S6 winner
        ↓
Freeze FV* + YS* + L* + P* + A*
        ↓
Resolve B32/B64
        ↓
Declare fixed-epoch budget
        ↓
Audit same population/model/optimizer config
        ↓
Audit drop_last=False and accumulation=1
        ↓
Compute expected update counts
        ↓
Verify B64 exact reference
        ↓
Reuse B64
        ↓
Register B32
        ↓
Seed 42
        ↓
Fresh B32 loaders/model
        ↓
TRAINING_ENGINE-v1
        ↓
Verify B32 BEST
        ↓
Audit complete sample coverage
        ↓
Check eval-batch invariance
        ↓
Build Wh-space metrics
        ↓
Build optimizer-step diagnostics
        ↓
Compare gradients/clipping/runtime
        ↓
Select minimum-RMSE batch
        ↓
Apply B64 exact-tie rule
        ↓
Update Phase 30 reference
        ↓
Write S7 artifacts
        ↓
SWEEP_S7_BATCH-v1 sign-off
```

---

# 153. Fail-fast order

Before expensive B32 training:

```text
1. Phase 28 sign-off
2. S6 winner identity
3. FV*/YS*/L*/P*/A* lock
4. B32/B64 definitions
5. same sample population
6. DataLoader semantics
7. drop_last=False
8. gradient accumulation=1
9. same architecture/parameter schema
10. fixed LR/WD/loss/clip
11. fixed epoch budget
12. B64 reuse eligibility
13. Test firewall
14. Registry readiness
```

---

# 154. Why fixed-epoch budget must be explicit

Without documenting budget semantics, a reviewer may incorrectly assume B32/B64 receive equal optimizer updates.

They do not.

S7 must transparently state:

```text
equal data passes allowed by same epoch/early-stop protocol
not equal optimizer-step budget.
```

---

# 155. Why no LR scaling matters

Batch size and LR often interact.

But changing LR together with batch would make S7 unable to isolate the predefined factor.

Therefore:

```text
LR fixed now
LR swept next.
```

---

# 156. Why no WD compensation matters

With AdamW, more optimizer steps can change cumulative regularization/update dynamics.

This is an interaction limitation to document, not something to silently correct inside S7.

---

# 157. Why sample-weighted train loss matters

Suppose final B32 batch has 7 samples and earlier batches have 32.

Simple average of batch mean losses would overweight that final batch.

Therefore epoch loss must weight by actual batch size.

---

# 158. Why clipping fraction is better than clipped count

Example conceptually:

```text
B32 → 100 batches, 20 clipped
B64 → 50 batches, 10 clipped
```

Raw counts differ, but both are:

```text
20% clipped.
```

Fraction is the normalized comparison.

---

# 159. Why evaluation batch should not decide model quality

Evaluation batching is an implementation detail.

Metrics must be based on the same prediction set.

If eval batch partition changes predictions materially, that is a pipeline/model bug to audit, not an S7 effect.

---

# 160. Why B64 is exact-tie preferred

Only for exact RMSE equality.

The tie rule favors the existing larger batch because it usually requires fewer optimizer iterations per epoch and preserves the current reference.

Observed throughput must still be reported rather than assumed as a scientific result.

---

# 161. Winner verification checklist

Before writing `s7_batch_winner.json`:

```text
[ ] B64 valid reused reference.
[ ] B32 valid completed run.
[ ] Same FV*/YS*/L*/P*/A*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same feature/scaler/target transform.
[ ] Same model architecture/parameters.
[ ] Same LR/WD/loss/clip.
[ ] Same epoch budget.
[ ] Accumulation=1.
[ ] drop_last=False.
[ ] B32 coverage complete.
[ ] B64 coverage provenance valid.
[ ] Optimizer steps correctly counted.
[ ] Sample-weighted train loss used.
[ ] Both BEST checkpoints verified.
[ ] Evaluation batching does not alter scientific metrics.
[ ] Full-precision RMSE used.
[ ] Effect computed programmatically.
[ ] Exact tie rule respected.
[ ] No Test.
```

---

# 162. Phase 30 handoff

Phase 30 receives:

```text
s7_batch_winner.json
s7_reference_update.json
winner_run_id
winner_config_fingerprint
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
selected_batch_id
population_fingerprint
```

and changes only:

```text
learning rate.
```

---

# 163. If B32 wins

Current reference becomes:

```text
FV*
YS*
L*
P*
A*
B32
LR2=3e-4
```

Phase 30 compares:

```text
1e-4
3e-4
1e-3
```

at B32.

---

# 164. If B64 wins

Current reference remains B64 and Phase 30 sweeps LR at B64.

---

# 165. Reference reuse for Phase 30

S7 winner already has:

```text
LR2 = 3e-4.
```

Therefore normally:

```text
reuse S7 winner as LR2 reference
train only LR1=1e-4 and LR3=1e-3.
```

---

# 166. Relationship with Phase 31 WD

Batch-size/WD interaction exists.

Phase 31 later tunes WD after LR has been selected.

Do not return to S7 automatically.

---

# 167. Relationship with Phase 39 clipping

Batch size may affect clipping frequency.

Phase 39 later tests clipping ON/OFF on the then-current reference.

S7 diagnostic should preserve clipping evidence for later interpretation.

---

# 168. Relationship with Phase 42 candidate synthesis

Both B32 and B64 runs remain in Registry.

No losing condition is deleted.

---

# 169. Relationship with Phase 44 rolling-origin

S7 winner uses one Validation period.

Later rolling-origin checks temporal robustness.

---

# 170. Relationship with Phase 46 multi-seed

S7 uses seed 42.

Small batch-size margins may not be seed-stable.

---

# 171. Reproducibility metadata

New B32 run records:

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
train batch size
eval batch size
drop_last
gradient accumulation
sampler/worker policy
population fingerprint
model config fingerprint
parameter schema fingerprint
initial model state fingerprint
Training Engine fingerprint
best checkpoint checksum
history checksum
metric checksum
```

---

# 172. No fabricated runtime outputs

Do not pre-fill:

```text
N_train
steps per epoch
last batch size
winner
RMSE
gradient norms
runtime
throughput
sample order match
```

before actual execution.

---

# 173. Phase sign-off

Create:

```text
phase_29_signoff.json
```

Minimum:

```text
phase = 29
phase_name = S7 Batch sweep
sweep_version
sweep_id
source_s6_winner_run_id
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
b32_run_id
b64_reference_run_id
new_run_ids
reused_run_ids
winner_batch_id
winner_train_batch_size
winner_run_id
winner_rmse_wh
population_fingerprint
metric_version
budget_semantics
drop_last
gradient_accumulation
coverage_audit_status
optimizer_budget_status
eval_batch_invariance_status
fairness_audit_status
matched_initialization_status
sample_order_match_status
inherited_warnings
test_status
approved_for_phase30
overall_status
created_at
```

---

# 174. Acceptance checklist

```text
[ ] Phase 28 valid.
[ ] approved_for_phase29 = true.
[ ] S7 version declared.
[ ] S6 winner loaded.
[ ] FV* fixed.
[ ] YS* fixed.
[ ] L* fixed.
[ ] P* fixed.
[ ] A* fixed.
[ ] B32 registered as train batch 32.
[ ] B64 registered as train batch 64.
[ ] Scientific swept field = train_batch_size.
[ ] Evaluation batch semantics documented.
[ ] Fixed-epoch budget declared.
[ ] No optimizer-step equalization.
[ ] Gradient accumulation = 1.
[ ] drop_last=False.
[ ] LR3e-4 fixed.
[ ] WD1e-4 fixed.
[ ] MSE fixed.
[ ] E50 fixed.
[ ] Patience10 fixed.
[ ] Clip1 fixed.
[ ] Scheduler=None.
[ ] Mixed precision fixed.
[ ] Seed42 fixed.
[ ] Same Train IDs.
[ ] Same Validation IDs.
[ ] Same WINDOWPOP-v1.
[ ] Same feature fingerprint.
[ ] Same X scaler.
[ ] Same target transform/scaler.
[ ] Same lookback/pooling/activation.
[ ] Same H1/WB0.
[ ] Same D64/H4/N2/FFN128.
[ ] Same dropout.
[ ] Same PE/norm/mask policy.
[ ] Same regression head.
[ ] Same parameter count.
[ ] Same parameter schema.
[ ] No BatchNorm present.
[ ] Same sampler policy.
[ ] Same shuffle policy.
[ ] Same worker/pin-memory policy.
[ ] Same device for runtime comparison or runtime marked non-comparable.
[ ] Expected steps per epoch derived from actual N_train.
[ ] Last partial batch semantics computed.
[ ] B32 Train coverage complete.
[ ] B64 coverage provenance available or documented.
[ ] No missing samples.
[ ] No duplicate samples.
[ ] Train loss sample-weighted.
[ ] Validation metrics globally aggregated.
[ ] Optimizer-step counts recorded.
[ ] Samples-seen counts recorded.
[ ] Sample-order match checked if available.
[ ] Initial-state match checked if available.
[ ] B64 reference exact-match.
[ ] B64 not retrained.
[ ] B32 registered before training.
[ ] B32 fresh loaders/model.
[ ] B32 no warm-start.
[ ] B32 trained via TRAINING_ENGINE-v1.
[ ] B32 BEST verified.
[ ] B64 BEST already verified.
[ ] Eval batch invariance checked.
[ ] Both result rows use verified BEST.
[ ] Full-precision RMSE used.
[ ] Batch effect computed.
[ ] MAE/R² effects computed.
[ ] Winner=min RMSE.
[ ] Exact tie=B64.
[ ] Metric divergence recorded if present.
[ ] Clipping fractions normalized.
[ ] Optimizer-budget diagnostics generated.
[ ] Update-axis diagnostics generated.
[ ] No interpolated Validation metric used for winner.
[ ] Runtime/throughput diagnostics generated.
[ ] Hypothesis outcomes recorded.
[ ] Findings generated.
[ ] Inherited warnings propagated.
[ ] Winner artifact generated.
[ ] Phase 30 reference update generated.
[ ] LR2 reuse identified.
[ ] No B16/B128.
[ ] No dynamic batch.
[ ] No LR scaling.
[ ] No WD compensation.
[ ] No accumulation.
[ ] No score-based rerun.
[ ] No failed/SANITY run in ranking.
[ ] No attention extraction.
[ ] No Test access.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] Batch/LR/WD/clipping interaction limitation documented.
[ ] Epoch-vs-step budget caveat documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 175. Acceptance criteria

Phase 29 chỉ PASS khi:

```text
S6-selected data/model configuration is fixed.

Exactly B32 and B64 are compared.

Only training mini-batch size changes scientifically.

Same Train/Validation target IDs are used.

drop_last=False and complete sample coverage are preserved.

gradient accumulation remains 1.

LR/WD/loss/clipping remain fixed.

Fixed epoch/early-stopping budget is explicit.

Optimizer-step differences are measured, not hidden.

Train loss is sample-weighted.

Validation metrics are globally aggregated.

Model architecture and parameter schema are identical.

B64 reference is exact-match and reused.

B32 is one fresh seed-42 run.

No warm-start occurs.

Both BEST checkpoints are verified.

Validation RMSE Wh selects winner.

Exact RMSE tie selects B64.

Phase 30 reference is generated.

No hidden batch/LR compensation occurs.

Test remains untouched.
```

---

# 176. Failure conditions

Phase 29 FAIL if:

```text
wrong S6 winner used

feature/target/lookback/pooling/activation changes

B32/B64 definitions wrong

gradient accumulation differs

drop_last changes

Train sample coverage differs

LR is scaled with batch

WD is compensated

epoch cap is changed to equalize steps

architecture/parameters differ

sample population differs

B64 reference mismatches but is reused

B64 retrained and best rerun chosen

B32 warm-started

B32 run fails but B64 declared winner

simple mean of unequal batch losses used as epoch loss

batch RMSEs averaged into global RMSE

RMSE rounded before ranking

runtime overrides RMSE

score-based rerun occurs

Test used.
```

---

# 177. Common mistakes

## 177.1 B32 + LR1.5e-4

Sai. Hai factors thay đổi.

## 177.2 B32 + gradient accumulation 2

Khi đó effective batch gần B64, không phải intended B32.

## 177.3 B64 drop_last=True nhưng B32 false

Population exposure differs.

## 177.4 Chạy B32 ít epoch hơn để equalize update count

Không theo fixed-epoch S7 contract.

## 177.5 Dùng batch-loss mean rồi average các batch

Sai khi final partial batch khác size.

## 177.6 So raw clipped-batch counts

Phải ưu tiên fraction.

## 177.7 Thấy B32 nhiều optimizer steps hơn rồi gọi đó là unfair bug

Đây là intended consequence của batch size dưới fixed epoch budget.

## 177.8 Thấy B32 thắng rồi kết luận gradient noise là nguyên nhân

Không đủ evidence.

## 177.9 Retrain B64 để “công bằng”

Không. Exact reference reuse sạch hơn.

## 177.10 Warm-start B32 từ B64

Confounded.

## 177.11 Đổi num_workers cho B32 để tăng tốc

Hidden implementation difference.

## 177.12 Chọn B64 vì nhanh hơn dù RMSE kém

Sai. Runtime chỉ tie/context.

## 177.13 Dùng Test để kiểm batch nào ổn định hơn

Forbidden.

## 177.14 Bật attention weights trong B32 run

Thay runtime/memory path không cần thiết.

## 177.15 Chạy thêm B128 sau khi thấy trend

Không thuộc S7.

---

# 178. Recommended execution pseudocode

```text
load_phase28_signoff()
assert_approved_for_phase29()

s6 = load_s6_winner()

FV = s6.feature_variant_id
YS = s6.target_scaling_id
L  = s6.lookback_id
P  = s6.pooling_id
A  = s6.activation_id

audit_batch_definitions(B32=32, B64=64)
declare_budget_semantics("FIXED_EPOCH_BUDGET")

audit_same_data_population(FV, YS, L)
audit_same_model(FV, L, P, A)
audit_dataloader_semantics()
assert_drop_last_false()
assert_gradient_accumulation(1)
audit_fixed_optimizer_config(
    lr=3e-4,
    wd=1e-4,
    loss="MSE",
    clip=1.0
)

b64_reference = resolve_s6_winner_run()
assert_exact_s7_reference_match(b64_reference)

register_b32_run()
seed(42)

loaders = build_fresh_loaders(
    feature_variant=FV,
    target_scaling=YS,
    lookback=L,
    train_batch_size=32,
    evaluation_batch_semantics=FROZEN_REFERENCE,
    population="WINDOWPOP-v1"
)

model = build_fresh_transformer(
    input_size=feature_count(FV),
    pooling=P,
    activation=A
)

b32_result = TRAINING_ENGINE_v1.fit(...)
verify_best_checkpoint(b32_result)

audit_train_coverage(b32_result)
audit_eval_batch_invariance(b32_result.best_checkpoint)

results = {
    "B32": b32_result,
    "B64": b64_reference
}

metrics = build_verified_s7_metrics(results)
budget = build_optimizer_budget_diagnostics(results)
effect = compute_batch_effect(metrics)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer="B64"
)

write_optimization_runtime_update_axis_diagnostics()
write_hypothesis_outcomes()
write_findings()
write_s7_winner(winner)
write_phase30_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 179. Definition of Done

\[
\boxed{
One\ Fixed\ Transformer\ Configuration
+
Two\ Training\ Batch\ Sizes
+
One\ Fresh\ B32\ Run
+
One\ Valid\ Reused\ B64
+
Same\ Population
+
Complete\ Coverage
+
Fixed\ Epoch\ Budget
+
Measured\ Update\ Budget
+
Verified\ BEST\ Metrics
+
S7\ Winner
+
Phase30\ Reference
+
No\ Test
}
\]

---

# 180. Final status contract

```text
PHASE 29 tests TRAINING BATCH SIZE only.

Current:
feature variant from previous sweeps
target scaling from S3
lookback from S4
pooling from S5
activation from S6.

Conditions:

B32
= training mini-batch 32

B64
= training mini-batch 64

Budget:
same max epochs / early stopping
NOT equal optimizer-step budget.

drop_last:
False.

gradient accumulation:
1.

No LR scaling.
No WD compensation.
No batch-specific epoch cap.

B64:
reuse S6 winner if exact match.

B32:
one new fresh run.

Same:
sample IDs
WINDOWPOP-v1
X/y transforms
model architecture
parameter schema
AdamW
LR3e-4
WD1e-4
MSE
E50
patience10
clip1
seed42
Training Engine
Metric version.

Train loss:
sample-weighted.

Validation metrics:
global full-population.

Selection:
minimum verified Validation RMSE Wh.

Exact tie:
prefer B64.

Record:
steps/epoch
total optimizer steps
last batch size
coverage
gradient clipping fraction
runtime/throughput.

No warm-start.
No hidden B16/B128.
No score-based rerun.
No attention extraction.
No Test.

After SWEEP_S7_BATCH-v1 PASS:
update current reference
and proceed to
PHASE 30 — S8 Learning-rate sweep.
```

---

# 181. Final check

Correct workflow:

```text
Load S6 winner
→ Freeze FV* + YS* + L* + P* + A*
→ Define B32/B64
→ Lock fixed-epoch budget
→ Audit same samples/model/optimizer config
→ Verify drop_last=False
→ Verify accumulation=1
→ Reuse B64
→ Train one fresh B32
→ Verify complete sample coverage
→ Verify BEST checkpoint
→ Record optimizer-step budget
→ Compare full-precision Validation RMSE
→ Diagnose gradients/runtime without overriding RMSE
→ Select S7 winner
→ Update Phase 30 reference
```

Incorrect workflow:

```text
B32 + scaled LR
→ B64 + different drop_last
→ equalize total steps by changing epochs
→ retrain B64
→ choose fastest run
→ inspect Test
```

Chỉ sau khi `SWEEP_S7_BATCH-v1` được sign-off mới chuyển sang **PHASE 30 — S8 Learning-rate sweep**.
