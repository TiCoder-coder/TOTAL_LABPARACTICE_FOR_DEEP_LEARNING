# PHASE 26 — S4 LOOKBACK SWEEP

## Kế hoạch controlled sweep cho Lookback Length của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S3_TARGETSCALING-v1`  
**Sweep ID:** `S4_LOOKBACK`  
**Output version:** `SWEEP_S4_LOOKBACK-v1`  
**Phase trước:** `Phase_25_S3_Target-scaling_sweep.md`

---

# 1. Vai trò của Phase 26

Phase 26 là controlled experiment thứ tư trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với feature variant, target-scaling policy, model architecture, optimizer, training protocol, data population và seed đã được khóa từ Phase 25, độ dài lịch sử đầu vào bao nhiêu là phù hợp hơn cho bài toán dự báo `Appliances` 10 phút tiếp theo?

Phase 26 chỉ thay đúng một conceptual factor:

```text
LOOKBACK LENGTH
```

với ba condition:

```text
L36
L72
L144
```

tương ứng:

```text
6 giờ
12 giờ
24 giờ
```

ở sampling interval 10 phút.

Nguyên tắc trung tâm:

\[
\boxed{
One\ Factor
+
Same\ Targets
+
Same\ Inputs\ Per\ Feature\ Channel
+
Same\ Training
+
Common\ Population
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí Phase 26 trong master execution plan

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
```

Phase 26 không được quay lại thay:

```text
feature set
time features
target scaling
optimizer
model depth
dropout
pooling
```

Mọi thứ đó giữ nguyên current reference từ S3.

---

# 3. Câu hỏi nghiên cứu của S4

Phase 26 phải trả lời:

```text
1. 6h, 12h hay 24h lịch sử tạo Validation RMSE thấp nhất?

2. Tăng lookback có giúp model tận dụng temporal context tốt hơn không?

3. Có dấu hiệu diminishing returns khi tăng từ 12h lên 24h không?

4. Lookback dài hơn có làm optimization khó hơn không?

5. Best epoch và early stopping behavior thay đổi như thế nào?

6. Runtime/memory tăng bao nhiêu khi sequence length tăng?

7. Gradient behavior có thay đổi theo L không?

8. Lookback nào trở thành current reference cho Phase 27?
```

---

# 4. Lookback definitions

Sampling interval đã khóa:

```text
10 minutes
```

Do đó:

```text
6 samples/hour
```

và:

```text
L36  = 36 steps  = 6 hours
L72  = 72 steps  = 12 hours
L144 = 144 steps = 24 hours
```

---

# 5. Forecast horizon giữ nguyên

Hard:

```text
H1
```

tức:

```text
10 minutes ahead.
```

Cho target index `j`:

```text
input = [j-L, ..., j-1]
target = j
```

Không đổi H trong S4.

---

# 6. Exact temporal geometry

General formula:

```text
target position = j
input_end = j - H
input_start = input_end - L + 1
```

Với `H=1`:

```text
input_end = j - 1
input_start = j - L
```

Các condition:

```text
L36:
[j-36, ..., j-1] → j

L72:
[j-72, ..., j-1] → j

L144:
[j-144, ..., j-1] → j
```

---

# 7. Relative lag semantics

Với sequence position `p` từ `0` đến `L-1`:

\[
LagSteps_p
=
H + (L-1-p)
\]

Với `H=1`:

```text
last input position
→ lag 1 step = 10 minutes

first input position:
L36  → 36 steps = 360 minutes
L72  → 72 steps = 720 minutes
L144 → 144 steps = 1440 minutes
```

---

# 8. Working hypotheses

## H-S4-01 — Longer historical context can improve forecasting

Nếu:

```text
RMSE(L144) < RMSE(L72) < RMSE(L36)
```

hoặc L144 tốt nhất, evidence support rằng longer context hữu ích dưới current configuration.

Status:

```text
UNTESTED
```

## H-S4-02 — Moderate context may be sufficient

Nếu:

```text
L72 <= L144
```

và L72 tốt hơn hoặc gần tương đương, có thể gợi ý:

```text
12h context captures most useful short-term dynamics
24h may add redundant/noisy context
```

chỉ ở mức hypothesis.

## H-S4-03 — Short context may generalize better

Nếu L36 thắng:

```text
shorter local context may be sufficient or easier to optimize
```

nhưng không được kết luận nguyên nhân duy nhất.

---

# 9. Preconditions bắt buộc

Phase 26 chỉ bắt đầu khi:

```text
Phase 25 = PASS
```

hoặc `PASS_WITH_WARNING` không có unresolved critical issue.

Bắt buộc có:

```text
s3_target_scaling_winner.json
s3_reference_update.json
phase_25_signoff.json
approved_for_phase26 = true
```

---

# 10. Resolve current configuration từ S3

Phase 26 phải load:

```text
feature_variant_id
selected_target_scaling_id
winner_run_id
winner_config_fingerprint
winner_feature_fingerprint
population_fingerprint
```

Gọi:

```text
FV* = S3-selected feature variant
YS* = S3-selected target scaling
```

Possible:

```text
FV* ∈ {
FS0_TF0, FS0_TF1,
FS1_TF0, FS1_TF1,
FS2_TF0, FS2_TF1
}

YS* ∈ {YS0, YS1}
```

Phase 26 không hard-code `FS1_TF1 + YS1`.

---

# 11. Carry-forward warnings

Nếu upstream có warning như:

```text
RANDOM_CONTROL_GAIN
METRIC_RANKING_DIVERGENCE
TARGET_SCALING_SMALL_MARGIN
```

Phase 26 phải carry vào:

```text
manifest
summary
report
winner artifact
Phase 27 reference update
```

Không tự xóa warning.

---

# 12. Upstream contracts bắt buộc

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
```

---

# 13. Swept factor duy nhất

```text
lookback_id
```

Allowed:

```text
L36
L72
L144
```

---

# 14. Frozen factors

Hard freeze:

```text
feature_variant_id = FV*
target_scaling_id = YS*
H1
WB0
WINDOWPOP-v1
B64

d_model = 64
num_heads = 4
num_layers = 2
ffn_dim = 128
dropout = 0.1
activation = GELU
pooling = LAST_STEP
sinusoidal positional encoding
POST_NORM
no causal mask
no padding mask

AdamW
LR = 3e-4
WD = 1e-4
MSE
max_epochs = 50
patience = 10
min_delta = 0
gradient clip max_norm = 1.0
scheduler = None
mixed precision = False
gradient accumulation = 1
seed = 42

TRAINING_ENGINE-v1
METRICS-v1
```

---

# 15. Common population là hard requirement quan trọng nhất của S4

Phase 10 đã khóa:

```text
WINDOWPOP-v1
=
intersection of valid target IDs
across L36, L72, L144
under H1/WB0.
```

Phase 26 phải dùng chính population này.

Không được dùng native per-lookback sample sets để chọn winner.

---

# 16. Native population vs common population

Phase 10 có thể có:

```text
native_valid_targets_L36
native_valid_targets_L72
native_valid_targets_L144
```

và:

```text
common_targets
=
intersection(...)
```

S4 scientific comparison phải dùng:

```text
common_targets only.
```

Native counts chỉ được báo engineering/context, không dùng để tính sweep winner.

---

# 17. Vì sao common population quan trọng?

Nếu dùng native sets:

```text
L36 có thể có nhiều target hợp lệ hơn L144
```

do cần ít lịch sử hơn.

Khi đó RMSE khác nhau có thể phản ánh:

```text
different target timestamps
different difficulty distribution
```

không chỉ lookback length.

Do đó:

\[
\boxed{
Same\ Target\ IDs
}
\]

là fairness condition bắt buộc.

---

# 18. Population identity audit

Hard:

```text
Train IDs L36
=
Train IDs L72
=
Train IDs L144

Validation IDs L36
=
Validation IDs L72
=
Validation IDs L144
```

under `WINDOWPOP-v1`.

---

# 19. Do not redefine WINDOWPOP-v1 in Phase 26

Không recompute intersection bằng logic mới nếu frozen artifact đã có.

Phase 26 phải:

```text
load
verify
use
```

Không:

```text
rebuild ad hoc
```

unless checksum/contract requires regeneration from canonical Phase 10 code.

---

# 20. Window continuity rules giữ nguyên

Mỗi lookback phải satisfy:

```text
all adjacent input deltas = 10 min
target-input_end delta = 10 min
same continuity segment
no duplicate/off-grid timestamp
no gap crossing
no padding
no interpolation
```

---

# 21. No padding

S4 không được:

```text
pad L36 to 144
pad L72 to 144
```

để giữ same shape.

Mỗi condition có actual sequence length:

```text
[B,36,F]
[B,72,F]
[B,144,F]
```

Transformer already supports variable L by construction.

---

# 22. No truncation trick inside a shared L144 tensor as source of truth

Có thể implementation optimize bằng slicing nếu exact same canonical window semantics được đảm bảo, nhưng scientific contract phải coi mỗi L là một registered window condition.

Không được accidentally:

```text
take first 36 rows of L144
```

thay vì most recent 36 rows.

Correct L36 relative to target phải là:

```text
[j-36, ..., j-1]
```

tức **right-aligned recent context**.

---

# 23. Right-alignment audit

For a common target `j`:

```text
L36 input
=
last 36 positions of L72 input
=
last 36 positions of L144 input
```

and:

```text
L72 input
=
last 72 positions of L144 input
```

provided same target and continuity.

This is a very useful high-value S4 assertion.

---

# 24. Timestamp alignment audit

For common sample:

```text
input_end timestamp identical across L36/L72/L144
target timestamp identical across all
```

Only `input_start` changes.

Expected:

```text
L36 starts 6h before target
L72 starts 12h before target
L144 starts 24h before target
```

at 10-minute granularity.

---

# 25. Feature configuration phải giống nhau

Hard:

```text
same feature list
same feature order
same feature fingerprint
same X scaler
```

Lookback does not alter feature channels.

---

# 26. X scaler reuse

Unlike feature sweeps:

```text
same FV*
```

therefore all lookbacks use exact same frozen X scaler bundle.

Hard:

```text
x_scaler_checksum_L36
=
x_scaler_checksum_L72
=
x_scaler_checksum_L144
```

---

# 27. Target scaling reuse

All lookbacks use:

```text
YS*
```

selected from S3.

If:

```text
YS1
```

same frozen target scaler across all L.

If:

```text
YS0
```

same identity target transform.

---

# 28. Target scaling independent of lookback

This is important because Phase 9 deliberately defined YS fit independent of lookback.

Do not fit a different y scaler per L.

---

# 29. Architecture invariants

Transformer architecture remains:

```text
input_size = same F
d_model = 64
heads = 4
layers = 2
FFN = 128
dropout = .1
GELU
LAST_STEP
```

Trainable parameter count should be identical across lookbacks.

---

# 30. Why parameter count stays same

Sequence length changes compute/memory, not learned layer dimensions.

Therefore hard expected:

```text
trainable_parameters_L36
=
trainable_parameters_L72
=
trainable_parameters_L144.
```

Mismatch suggests architecture/config drift.

---

# 31. Positional encoding capacity audit

Transformer implementation must support at least:

```text
L144
```

positions.

Phase 26 should verify:

```text
PE capacity >= 144
```

for all runs.

No need to create separate PE tables per L if implementation slices a shared buffer correctly.

---

# 32. Positional encoding indexing semantics

Each lookback should use positions:

```text
0 ... L-1
```

for its own sequence.

Current B0 contract does not require absolute timestamps as positional indices.

Do not offset L36 positions to:

```text
108...143
```

unless implementation contract explicitly defined that earlier. Default remains local sequence positions.

---

# 33. Attention complexity note

Self-attention memory/compute scales approximately with:

\[
O(L^2)
\]

for attention score matrices.

Relative score-matrix sizes:

```text
L36  → 36²   = 1,296
L72  → 72²   = 5,184
L144 → 144²  = 20,736
```

So L144 has:

```text
16×
```

the attention score elements of L36, and:

```text
4×
```

those of L72.

This is engineering context, not winner criterion.

---

# 34. No attention extraction during S4 training

Normal training remains:

```text
need_weights = False
attention collection OFF
```

Do not turn on attention just because lookback changes.

---

# 35. Existing L144 reference reuse rule

S3 winner already uses:

```text
L144
```

If exact S4 contract match:

```text
REUSE S3 winner run
```

as L144 reference.

Do not retrain L144.

---

# 36. Normal S4 run count

Expected:

```text
L144
→ reused S3 winner

L36
→ NEW run

L72
→ NEW run
```

Therefore typically:

```text
2 new runs
+
1 reused reference.
```

---

# 37. L144 reference reuse gate

Exact match required on:

```text
FV*
YS*
L144
H1
WB0
WINDOWPOP-v1
B64
D64/H4/N2/FFN128
dropout .1
GELU
LAST_STEP
AdamW
LR3e-4
WD1e-4
MSE
E50
patience10
clip1
seed42
Training Engine
Metric version
```

Mismatch:

```text
STOP.
```

---

# 38. Fresh L36 and L72 runs

For each new lookback:

```text
set seed 42
↓
fresh lookback-specific Train loader
↓
fresh lookback-specific Validation loader
↓
same feature channels/scaler
↓
same target transform
↓
fresh Transformer
↓
MSE
↓
AdamW
↓
TRAINING_ENGINE-v1
```

No warm-start from L144 or each other.

---

# 39. No warm-start

Hard:

```text
L36 fresh
L72 fresh
```

Do not initialize one from another condition.

---

# 40. Matched initialization opportunity

Because architecture/parameter shapes are identical across lookbacks:

```text
same seed
same model config
```

it is recommended to verify:

```text
initial_model_state_fingerprint
```

matches across L36/L72/L144 if upstream reference stored it.

If L144 initial fingerprint unavailable:

```text
NOT_VERIFIABLE
```

not FAIL.

---

# 41. DataLoader shuffle equivalence nuance

All conditions use same target IDs and seed 42.

If DataLoader datasets expose identical sample ordering before shuffle, same generator seed should yield same **sample-index permutation** per epoch even though `X` tensors differ in L.

Recommended verify if infrastructure supports:

```text
first-epoch sample_idx order fingerprint
```

across L36/L72/new runs and reference.

If not stored for reference:

```text
NOT_VERIFIABLE.
```

---

# 42. Batch size fixed

```text
B64
```

for all L.

No automatic batch reduction for L144/L72.

---

# 43. OOM policy

If a new lookback run OOMs under B64:

```text
run FAILS.
```

Do not silently change B32.

Batch sweep belongs Phase 29.

Because L144 B64 already exists successfully under reference, L36/L72 should normally be no more memory-demanding; an OOM likely indicates infrastructure/config anomaly.

---

# 44. Training Engine fixed

All new runs use:

```text
TRAINING_ENGINE-v1.
```

No lookback-specific custom loop.

---

# 45. Training objective fixed

```text
MSE
```

in current selected target model space `YS*`.

---

# 46. Primary S4 selection metric

Hard:

```text
best_validation_rmse_wh
```

from each verified BEST checkpoint.

---

# 47. Secondary metrics

Record:

```text
best_validation_mae_wh
best_validation_r2
best_epoch
epochs_completed
stop_reason
gradient diagnostics
runtime
memory diagnostics optional
```

---

# 48. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{L36},
RMSE_{L72},
RMSE_{L144}
\right)
\]

using full-precision Wh metrics.

---

# 49. Exact tie rule

If exact full-precision RMSE tie:

```text
prefer shorter lookback
```

Parsimony order:

```text
L36 < L72 < L144
```

Rationale:

```text
less historical dependency
lower compute/memory
shorter startup history requirement
simpler deployment context
```

Tie event must be explicitly recorded.

---

# 50. No arbitrary improvement threshold

Do not require:

```text
1% improvement
5 Wh margin
```

Current experimental rule is strict lower RMSE.

---

# 51. Pairwise effect formulas

Required comparisons:

```text
L36 → L72
L72 → L144
L36 → L144
```

For A→B:

\[
\Delta RMSE_{A\rightarrow B}
=
RMSE_A-RMSE_B
\]

Positive means B improves.

Relative:

\[
Improvement\%
=
100
\times
\frac{RMSE_A-RMSE_B}{RMSE_A}
\]

Also:

\[
\Delta MAE=MAE_A-MAE_B
\]

\[
\Delta R^2=R^2_B-R^2_A
\]

---

# 52. Diminishing-return diagnostic

Create explicit descriptive quantity:

```text
gain_36_to_72
gain_72_to_144
```

If both positive, compare magnitude:

\[
GainRatio
=
\frac{
\Delta RMSE_{72\rightarrow144}
}{
\Delta RMSE_{36\rightarrow72}
}
\]

only if denominator is meaningfully non-zero.

Do not turn this into a formal significance measure.

---

# 53. Context-efficiency diagnostic

Can report:

```text
RMSE gain per additional hour
```

for descriptive engineering context:

```text
36→72 adds 6h
72→144 adds 12h
```

But this must not override winner selection.

---

# 54. Runtime scaling diagnostics

Record per lookback:

```text
mean epoch duration
total training duration
time to best epoch
```

Expected:

```text
longer L may be slower
```

but observed runtime is source of truth.

---

# 55. Memory diagnostics

If available from device/runtime:

```text
peak allocated memory
peak reserved memory
```

may be recorded.

Do not require cross-device-comparable exact values.

This is engineering context only.

---

# 56. Gradient diagnostics

Record:

```text
max grad norm
mean grad norm
fraction batches clipped
nonfinite events
```

per L.

Lookback may affect gradients, but do not infer causality from one seed.

---

# 57. Model-space loss comparability

Unlike S3, all S4 runs use same:

```text
YS*
MSE
```

so model-space losses are on same scale and may be compared descriptively.

Still:

```text
Validation RMSE Wh
```

remains primary selection metric.

---

# 58. Best-checkpoint verification for L36/L72

After each training:

```text
fresh Transformer
↓
same lookback config
↓
strict-load BEST
↓
full common-population Validation
↓
inverse target transform if YS1
↓
METRICS-v1
↓
verify recorded metrics.
```

---

# 59. L144 reference verification

No retrain.

Verify existing source:

```text
best checkpoint verification PASS
same common Validation IDs
same metric version
exact S4 config compatibility.
```

---

# 60. Do not compare native-population metrics

If run infrastructure also emits native-L metrics, label:

```text
DIAGNOSTIC_ONLY
NOT_ELIGIBLE_FOR_S4_SELECTION
```

S4 winner table must contain only:

```text
WINDOWPOP-v1 common-population metrics.
```

---

# 61. Common-population metric guard

For each S4 result:

```text
selection_population_id = WINDOWPOP-v1
```

Hard.

Any result from:

```text
NATIVE_L36
NATIVE_L72
NATIVE_L144
```

cannot enter ranking.

---

# 62. Run matrix

Create:

```text
s4_run_matrix.csv
```

Fields:

```text
sweep_id
lookback_id
lookback_steps
lookback_hours
source_type
source_run_id
requires_new_training
feature_variant_id
target_scaling_id
feature_fingerprint
x_scaler_id
target_transform_id
window_index_version
population_fingerprint
batch_size
seed
model_config_id
training_config_id
status
```

---

# 63. Sweep manifest

Create:

```text
s4_lookback_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S4_LOOKBACK-v1
sweep_id = S4_LOOKBACK
source_s3_winner_run_id
feature_variant_id
target_scaling_id
candidate_lookbacks = [L36,L72,L144]
new_runs_required
reused_runs
swept_field = lookback_id
frozen_fields
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = SHORTEST_LOOKBACK_ON_EXACT_RMSE_TIE
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

# 64. Sweep contract

Create:

```text
s4_lookback_sweep_contract.json
```

Must state:

```text
Only lookback length changes.

L36 = 6h.
L72 = 12h.
L144 = 24h.

H1 fixed.

Same S3-selected feature variant.
Same S3-selected target scaling.
Same X scaler.
Same y scaler/identity.
Same common target IDs from WINDOWPOP-v1.
No native-population ranking.
Same Transformer architecture.
Same optimizer/training config.
Seed 42.
Validation RMSE Wh selects winner.
Exact tie → shortest lookback.
Test forbidden.
```

---

# 65. Preflight audit

Create:

```text
s4_lookback_preflight_audit.csv
```

Checks:

```text
phase25_pass
approved_for_phase26
s3_winner_valid
feature_variant_locked
target_scaling_locked
L36_registered
L72_registered
L144_registered
WINDOWPOP_v1_valid
population_fingerprint_valid
L144_reference_exact_match
Training_Engine_fixed
metric_fixed
seed_fixed
test_locked
status
```

---

# 66. Lookback-definition audit

Create:

```text
s4_lookback_definition_audit.csv
```

Fields:

```text
lookback_id
steps
expected_hours
sampling_minutes
computed_hours
horizon_steps
horizon_minutes
input_start_relation
input_end_relation
status
```

Expected:

```text
L36 36 6h
L72 72 12h
L144 144 24h
```

---

# 67. Window geometry audit

Create:

```text
s4_window_geometry_audit.csv
```

For selected sample probes:

```text
sample_idx
lookback_id
input_start_idx
input_end_idx
target_idx
input_start_timestamp
input_end_timestamp
target_timestamp
adjacent_delta_valid
target_delta_valid
continuity_segment_id
no_gap
status
```

---

# 68. Right-alignment audit

Create:

```text
s4_window_alignment_audit.csv
```

For common probes:

```text
sample_idx
l36_equals_last36_of_l72
l36_equals_last36_of_l144
l72_equals_last72_of_l144
input_end_equal
target_equal
status
```

This is one of the most important S4 correctness artifacts.

---

# 69. Common-population audit

Create:

```text
s4_lookback_population_audit.csv
```

Fields:

```text
lookback_id
split_id
native_sample_count
common_sample_count
population_version
population_fingerprint
common_ids_match_reference
ordered_common_ids_match_reference
eligible_for_selection
status
```

---

# 70. Feature/scaler audit

Create:

```text
s4_feature_scaler_audit.csv
```

Fields:

```text
lookback_id
feature_variant_id
feature_count
feature_fingerprint
x_scaler_id
x_scaler_checksum
target_scaling_id
target_scaler_id
target_scaler_checksum
all_match_reference
status
```

---

# 71. Architecture audit

Create:

```text
s4_lookback_architecture_audit.csv
```

Fields:

```text
lookback_id
input_size
sequence_length
d_model
num_heads
num_layers
ffn_dim
dropout
activation
pooling
pe_capacity
trainable_parameters
architecture_fingerprint
only_sequence_length_differs
status
```

---

# 72. Training-config audit

Create:

```text
s4_lookback_training_audit.csv
```

Fields:

```text
lookback_id
batch_size
optimizer
learning_rate
weight_decay
criterion
max_epochs
patience
min_delta
gradient_clip
scheduler
mixed_precision
seed
training_engine_version
matches_reference
status
```

---

# 73. Initialization audit

Create:

```text
s4_initialization_audit.csv
```

Fields:

```text
lookback_id
initial_model_state_fingerprint
reference_available
fingerprint_matches
architecture_match
seed
status
```

Recommended expected match if all source fingerprints available.

---

# 74. DataLoader order audit

Optional but recommended:

```text
s4_dataloader_order_audit.csv
```

Fields:

```text
lookback_id
split_id
epoch_or_probe
sample_order_fingerprint
reference_available
matches_reference
status
```

Not-verifiable reference should not fail phase.

---

# 75. Run provenance table

Create:

```text
s4_lookback_run_provenance.csv
```

Fields:

```text
lookback_id
run_id
source_type
source_phase
config_fingerprint
feature_fingerprint
window_index_version
population_fingerprint
best_checkpoint_sha256
history_sha256
metric_artifact
prediction_artifact
status
```

---

# 76. Primary S4 metrics table

Create:

```text
s4_lookback_metrics.csv
```

Rows:

```text
L36
L72
L144
```

Fields:

```text
lookback_id
lookback_steps
lookback_hours
run_id
source_type
selection_population
trainable_parameters
best_epoch
epochs_completed
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

# 77. Pairwise effects table

Create:

```text
s4_lookback_pairwise_effects.csv
```

Rows:

```text
L36_TO_L72
L72_TO_L144
L36_TO_L144
```

Fields:

```text
left_lookback
right_lookback
left_rmse_wh
right_rmse_wh
rmse_delta_wh
rmse_improvement_pct
mae_delta_wh
r2_delta
additional_steps
additional_hours
rmse_gain_per_added_hour
status
```

---

# 78. Diminishing-return artifact

Create:

```text
s4_diminishing_return_diagnostics.json
```

Fields:

```text
gain_l36_to_l72_wh
gain_l72_to_l144_wh
gain_ratio_if_defined
winner
pattern
interpretation
status
```

Possible pattern:

```text
MONOTONIC_GAIN
DIMINISHING_GAIN
NO_ADDITIONAL_GAIN
REVERSAL
MIXED
INCONCLUSIVE
```

Descriptive only.

---

# 79. Runtime diagnostics

Create:

```text
s4_runtime_diagnostics.csv
```

Fields:

```text
lookback_id
sequence_length
epochs_completed
best_epoch
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
time_to_best_seconds_optional
peak_memory_optional
status
```

---

# 80. Compute-scaling context

Create optional:

```text
s4_attention_complexity_context.csv
```

Fields:

```text
lookback_id
L
L_squared
relative_to_L36
relative_to_L72
relative_to_L144
```

This is theoretical context, not observed runtime.

---

# 81. Gradient diagnostics

Create:

```text
s4_gradient_diagnostics.csv
```

Fields:

```text
lookback_id
global_max_grad_norm_preclip
mean_grad_norm_preclip
mean_fraction_batches_clipped
max_fraction_batches_clipped
epochs_with_any_clipping
nonfinite_gradient_events
status
```

---

# 82. Hypothesis outcomes

Create:

```text
s4_hypothesis_outcomes.csv
```

Fields:

```text
hypothesis_id
source_hypothesis_id_optional
comparison
expected_direction
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

# 83. Findings artifact

Create:

```text
s4_lookback_findings.csv
```

Possible codes:

```text
SHORT_CONTEXT_GAIN
MEDIUM_CONTEXT_GAIN
LONG_CONTEXT_GAIN
MONOTONIC_CONTEXT_GAIN
DIMINISHING_CONTEXT_GAIN
NO_LONG_CONTEXT_GAIN
LOOKBACK_EXACT_TIE
METRIC_RANKING_DIVERGENCE
CONVERGENCE_DIFFERENCE
GRADIENT_BEHAVIOR_DIFFERENCE
RUNTIME_SCALING
MEMORY_SCALING
COMMON_POPULATION_VERIFIED
INITIALIZATION_MATCH_VERIFIED
INITIALIZATION_NOT_VERIFIABLE
INHERITED_WARNING
```

---

# 84. Winner artifact

Create:

```text
s4_lookback_winner.json
```

Minimum:

```text
sweep_id
sweep_version
feature_variant_id
target_scaling_id
selection_metric
selection_direction
tie_rule
winner_lookback_id
winner_steps
winner_hours
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_lookback_id
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 85. Reference update for Phase 27

Create:

```text
s4_reference_update.json
```

Minimum:

```text
previous_reference_run_id
feature_variant_id
target_scaling_id
previous_lookback_id = L144
selected_lookback_id
winner_run_id
winner_config_fingerprint
winner_rmse_wh
pooling_state = LAST_STEP
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase27
```

---

# 86. Phase 27 handoff logic

Phase 27 tests:

```text
LAST_STEP
vs
MEAN
```

while holding:

```text
S4-selected lookback
S3-selected target scaling
S2-selected feature variant
```

fixed.

Current winner uses:

```text
LAST_STEP
```

therefore Phase 27 should normally:

```text
reuse S4 winner as LAST_STEP reference
train only MEAN pooling condition
```

if exact-match.

---

# 87. Learning curves in S4

Recommended:

```text
S4_01_validation_rmse_by_epoch.png
S4_02_validation_mae_by_epoch.png
S4_03_train_loss_by_epoch.png
S4_04_gradient_clipping_fraction.png
S4_05_best_validation_metrics.png
S4_06_runtime_by_lookback.png
```

---

# 88. Primary figure

```text
S4_01_validation_rmse_by_epoch.png
```

Overlay:

```text
L36
L72
L144
```

using common-population Validation metrics and best epoch markers.

---

# 89. Best metrics figure

```text
S4_05_best_validation_metrics.png
```

show:

```text
RMSE Wh
```

for all three.

Could include MAE as secondary separate view.

---

# 90. Runtime figure

```text
S4_06_runtime_by_lookback.png
```

engineering context only.

No winner selection from runtime.

---

# 91. No attention heatmaps in S4

Even though sequence length changes attention matrix size, no scientific attention analysis here.

Final attention work remains Phase 52–57.

---

# 92. No positional encoding sweep

Sinusoidal PE remains fixed.

---

# 93. No causal mask

Still:

```text
is_causal = False
```

because entire input window is historical.

---

# 94. No padding-mask use

No padding exists.

---

# 95. No recursive forecast

Still one-step ahead with observed history.

---

# 96. No horizon change

No H3/H6/etc.

---

# 97. No stride change

Window target population contract remains as Phase 10.

Do not sub-sample target windows differently per L.

---

# 98. No downsampling

Do not aggregate 10-minute data to hourly just to fit L144.

Not S4.

---

# 99. No resampling

No.

---

# 100. No feature aggregation by lookback

Do not add:

```text
mean_last_6h
max_last_12h
```

inside S4.

That would be another feature-engineering factor.

---

# 101. No memory workaround that changes semantics

If compute is slow, do not:

```text
truncate random positions
sample attention tokens
change batch size
gradient accumulate differently
```

without protocol amendment.

---

# 102. No winner by runtime

Hard.

---

# 103. No winner by best epoch

Hard.

---

# 104. No winner by train loss

Hard.

---

# 105. No winner by parameter count

All should be same anyway.

---

# 106. No winner by native sample count

Hard.

---

# 107. No winner by Test

Forbidden.

---

# 108. Winner by common-population Validation RMSE only

Hard central rule.

---

# 109. Exact tie parsimony rule

If exact tie among:

```text
L36 and L72
```

choose L36.

If tie among:

```text
L72 and L144
```

choose L72.

If all tie:

```text
choose L36.
```

---

# 110. Tiny non-zero margin

Strict lower RMSE still wins.

Report:

```text
small single-seed Validation margin
```

and avoid strong conclusions.

---

# 111. Interpretation if L144 wins

Safe:

> Under the current feature/target/training configuration, a full 24-hour historical window achieved the lowest Validation RMSE on the common target population.

Potential implication:

```text
longer daily context is empirically useful
```

but not proof of daily periodic mechanism.

---

# 112. Interpretation if L72 wins

Safe:

> The 12-hour lookback produced the lowest Validation RMSE, while extending to 24 hours did not provide additional improvement under this configuration.

Could indicate diminishing/redundant context, but keep causal language cautious.

---

# 113. Interpretation if L36 wins

Safe:

> The 6-hour window achieved the lowest Validation RMSE under the current configuration.

Could suggest local dynamics dominate or shorter sequences optimize more effectively, but not proven.

---

# 114. Single-seed limitation

Mandatory:

```text
all S4 conditions represented by seed 42.
```

No mean±std.

---

# 115. Validation-only limitation

Mandatory:

```text
S4 winner is Validation-selected development evidence.
```

No Test.

---

# 116. Sequential-selection limitation

By Phase 26, Validation has selected:

```text
S1 feature set
S2 time features
S3 target scaling
S4 lookback
```

Therefore complete Registry and later robustness checks are essential.

---

# 117. Interaction limitation

Lookback may interact with:

```text
pooling
capacity
dropout
learning rate
RevIN
```

S4 measures lookback effect only under current frozen settings.

Do not claim winner is globally best for all later architectures.

---

# 118. Why sequential design still acceptable

Coursework objective favors:

```text
controlled, interpretable one-factor sweeps
```

over full Cartesian search.

Limitations must be documented honestly.

---

# 119. Run failure policy

If L36 or L72 run fails:

```text
S4 incomplete.
```

Do not select from remaining runs.

---

# 120. Technical rerun policy

Allowed only for:

```text
interrupt
corrupt checkpoint
software/hardware fault
```

with Registry provenance.

Not allowed:

```text
score looked bad
```

as rerun reason.

---

# 121. Numerical failure

NaN/Inf:

```text
FAIL run.
```

No skip.

---

# 122. OOM anomaly

Since shorter L should not exceed L144 attention memory in same config, OOM at L36/L72 while L144 reference succeeded should trigger:

```text
infrastructure/config audit
```

not batch-size change.

---

# 123. Discrepancy taxonomy

```text
S3_REFERENCE_MISSING
S3_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DEFINITION_MISMATCH
WINDOW_GEOMETRY_MISMATCH
WINDOW_ALIGNMENT_FAILURE
NATIVE_POPULATION_USED_FOR_SELECTION
COMMON_POPULATION_MISMATCH
TARGET_ID_MISMATCH
FEATURE_FINGERPRINT_MISMATCH
X_SCALER_MISMATCH
TARGET_SCALER_MISMATCH
ARCHITECTURE_DRIFT
PARAMETER_COUNT_MISMATCH
PE_CAPACITY_FAILURE
TRAINING_CONFIG_DRIFT
SEED_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
REFERENCE_RUN_MISMATCH
RUN_FAILURE
CHECKPOINT_VERIFICATION_FAILURE
INCOMPLETE_SWEEP
RANKING_ERROR
PAIRWISE_EFFECT_ERROR
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 124. Discrepancy log

Create:

```text
s4_lookback_discrepancies.json
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

# 125. Severity examples

```text
CRITICAL:
native population used for ranking
common population mismatch
Test access
window geometry wrong

MAJOR:
wrong lookback definition
wrong scaler
checkpoint verification failure

MODERATE:
metric ranking divergence
tiny RMSE margin
runtime/memory scaling warning

INFO:
initialization fingerprint not verifiable
```

---

# 126. Status model

## PASS

```text
all three L conditions valid
same common targets
window geometry correct
reference reuse valid
L36/L72 verified
winner selected
Phase 27 reference generated
Test untouched
```

## PASS_WITH_WARNING

Possible:

```text
tiny margin
metric ranking divergence
large runtime increase
inherited upstream warning
```

with methodological validity intact.

## FAIL

Examples:

```text
common-population mismatch
native metrics used for winner
window alignment wrong
wrong scaler
architecture drift
unresolved run failure
Test access
```

---

# 127. Sweep summary

Create:

```text
s4_lookback_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
feature_variant_id
target_scaling_id
reference_run_id
new_runs
reused_runs
candidate_lookbacks
common_population
primary_metric
metrics_by_lookback
pairwise_effects
diminishing_return_pattern
runtime_diagnostics
gradient_diagnostics
winner
winner_margin
inherited_warnings
phase27_reference
test_status
overall_status
```

---

# 128. Human-readable report

Create:

```text
s4_lookback_sweep_report.md
```

Sections:

```text
1. Objective
2. S3-selected current reference
3. Lookback definitions
4. Frozen configuration
5. Common-population fairness
6. Window geometry/right alignment
7. Scaler/model fairness
8. Run provenance
9. Validation metrics
10. Pairwise lookback effects
11. Diminishing returns
12. Learning-curve context
13. Runtime/memory context
14. S4 winner
15. Limitations
16. Phase 27 handoff
```

---

# 129. Report wording structure

Each conclusion should separate:

```text
Observed
Interpretation
Limitation
Handoff
```

---

# 130. README

Create:

```text
README_S4_LOOKBACK_SWEEP.md
```

Must explain:

```text
Purpose
L36/L72/L144 definitions
why common population is mandatory
native vs common populations
right-aligned windows
same target timestamps
same feature/scaler/architecture
reference reuse
attention O(L²) context
winner rule
tie rule
limitations
Phase 27 handoff
No Test.
```

---

# 131. Output directory

```text
artifacts/
└── sweeps/
    └── S4_lookback/
        ├── s4_lookback_sweep_manifest.json
        ├── s4_lookback_sweep_contract.json
        ├── s4_lookback_preflight_audit.csv
        ├── s4_run_matrix.csv
        ├── s4_lookback_definition_audit.csv
        ├── s4_window_geometry_audit.csv
        ├── s4_window_alignment_audit.csv
        ├── s4_lookback_population_audit.csv
        ├── s4_feature_scaler_audit.csv
        ├── s4_lookback_architecture_audit.csv
        ├── s4_lookback_training_audit.csv
        ├── s4_initialization_audit.csv
        ├── s4_dataloader_order_audit.csv
        ├── s4_lookback_run_provenance.csv
        ├── s4_lookback_metrics.csv
        ├── s4_lookback_pairwise_effects.csv
        ├── s4_diminishing_return_diagnostics.json
        ├── s4_runtime_diagnostics.csv
        ├── s4_attention_complexity_context.csv
        ├── s4_gradient_diagnostics.csv
        ├── s4_hypothesis_outcomes.csv
        ├── s4_lookback_findings.csv
        ├── s4_lookback_winner.json
        ├── s4_reference_update.json
        ├── s4_lookback_sweep_tests.csv
        ├── s4_lookback_discrepancies.json
        ├── s4_lookback_sweep_summary.json
        ├── s4_lookback_sweep_report.md
        ├── figures/
        │   ├── S4_01_validation_rmse_by_epoch.png
        │   ├── S4_02_validation_mae_by_epoch.png
        │   ├── S4_03_train_loss_by_epoch.png
        │   ├── S4_04_gradient_clipping_fraction.png
        │   ├── S4_05_best_validation_metrics.png
        │   └── S4_06_runtime_by_lookback.png
        ├── README_S4_LOOKBACK_SWEEP.md
        └── phase_26_signoff.json
```

New run artifacts remain under:

```text
artifacts/runs/<run_id>/
```

No duplicate checkpoints inside sweep folder.

---

# 132. Required outputs

```text
O26.1  Sweep manifest
O26.2  Sweep contract
O26.3  Preflight audit
O26.4  Run matrix
O26.5  Lookback definition audit
O26.6  Window geometry audit
O26.7  Right-alignment audit
O26.8  Common-population audit
O26.9  Feature/scaler audit
O26.10 Architecture audit
O26.11 Training-config audit
O26.12 Initialization audit
O26.13 DataLoader order audit
O26.14 Run provenance
O26.15 Reused L144 reference
O26.16 Verified L36 run
O26.17 Verified L72 run
O26.18 Metrics table
O26.19 Pairwise effects
O26.20 Diminishing-return diagnostics
O26.21 Runtime diagnostics
O26.22 Attention complexity context
O26.23 Gradient diagnostics
O26.24 Hypothesis outcomes
O26.25 Findings
O26.26 Winner artifact
O26.27 Phase 27 reference update
O26.28 Figures
O26.29 Sweep tests
O26.30 Discrepancy log
O26.31 Sweep summary
O26.32 Human-readable report
O26.33 README
O26.34 Phase sign-off
```

---

# 133. Sweep test suite

Create:

```text
s4_lookback_sweep_tests.csv
```

Recommended checks:

```text
S4T26-001 Phase 25 PASS/non-critical warning only
S4T26-002 approved_for_phase26 = true
S4T26-003 S3 winner artifact valid
S4T26-004 feature variant fixed
S4T26-005 target scaling fixed
S4T26-006 L36 registered
S4T26-007 L72 registered
S4T26-008 L144 registered
S4T26-009 L36 = 36 steps
S4T26-010 L72 = 72 steps
S4T26-011 L144 = 144 steps
S4T26-012 sampling interval = 10 min
S4T26-013 L36 = 6 hours
S4T26-014 L72 = 12 hours
S4T26-015 L144 = 24 hours
S4T26-016 H1 fixed
S4T26-017 WB0 fixed
S4T26-018 WINDOWPOP-v1 loaded
S4T26-019 common population fingerprint valid
S4T26-020 Train common target IDs equal
S4T26-021 Validation common target IDs equal
S4T26-022 no native-population result enters ranking
S4T26-023 no gap crossing L36
S4T26-024 no gap crossing L72
S4T26-025 no gap crossing L144
S4T26-026 target-input_end delta = 10min for all
S4T26-027 L36 right-aligned to L72
S4T26-028 L36 right-aligned to L144
S4T26-029 L72 right-aligned to L144
S4T26-030 target timestamp identical across L
S4T26-031 input_end timestamp identical across L
S4T26-032 no padding
S4T26-033 no interpolation
S4T26-034 feature fingerprint same
S4T26-035 feature order same
S4T26-036 X scaler checksum same
S4T26-037 target transform same
S4T26-038 target scaler checksum same if YS1
S4T26-039 B64 fixed
S4T26-040 D64 fixed
S4T26-041 H4 fixed
S4T26-042 N2 fixed
S4T26-043 FFN128 fixed
S4T26-044 dropout .1 fixed
S4T26-045 GELU fixed
S4T26-046 LAST_STEP fixed
S4T26-047 PE policy fixed
S4T26-048 PE capacity >=144
S4T26-049 no causal mask
S4T26-050 no padding mask
S4T26-051 trainable parameter counts equal
S4T26-052 architecture fingerprints equal
S4T26-053 AdamW fixed
S4T26-054 LR3e-4 fixed
S4T26-055 WD1e-4 fixed
S4T26-056 MSE fixed
S4T26-057 E50 fixed
S4T26-058 patience10 fixed
S4T26-059 clip1 fixed
S4T26-060 seed42 fixed
S4T26-061 Training Engine fixed
S4T26-062 Metric version fixed
S4T26-063 L144 reference exact-match
S4T26-064 L144 reference reused
S4T26-065 L36 registered before training
S4T26-066 L72 registered before training
S4T26-067 L36 fresh loaders/model
S4T26-068 L72 fresh loaders/model
S4T26-069 no warm-start
S4T26-070 L36 trained via Training Engine
S4T26-071 L72 trained via Training Engine
S4T26-072 L36 BEST verified
S4T26-073 L72 BEST verified
S4T26-074 L144 BEST already verified
S4T26-075 all sweep rows use common population
S4T26-076 all sweep rows use verified BEST
S4T26-077 full-precision RMSE available
S4T26-078 ranking by RMSE correct
S4T26-079 exact tie → shortest L
S4T26-080 L36→L72 effect correct
S4T26-081 L72→L144 effect correct
S4T26-082 L36→L144 effect correct
S4T26-083 diminishing-return diagnostics generated
S4T26-084 runtime diagnostics generated
S4T26-085 gradient diagnostics generated
S4T26-086 theoretical complexity context labeled non-selection
S4T26-087 hypothesis outcomes generated
S4T26-088 winner points to valid run
S4T26-089 Phase 27 reference update generated
S4T26-090 LAST_STEP reuse identified for Phase 27
S4T26-091 inherited warnings propagated
S4T26-092 no score-based rerun
S4T26-093 no failed/SANITY run in ranking
S4T26-094 no Test access
S4T26-095 single-seed limitation documented
S4T26-096 Validation-only limitation documented
S4T26-097 interaction limitation documented
S4T26-098 figures source-derived
S4T26-099 summary/report generated
S4T26-100 phase sign-off generated
```

---

# 134. Recommended notebook structure

```text
Cell 26.1  Phase title
Cell 26.2  Verify Phase 25 sign-off
Cell 26.3  Declare SWEEP_S4_LOOKBACK-v1
Cell 26.4  Load S3 winner/reference update
Cell 26.5  Freeze FV* and YS*
Cell 26.6  Load L36/L72/L144 definitions
Cell 26.7  Load WINDOWPOP-v1 common population
Cell 26.8  Build S4 run matrix
Cell 26.9  Audit lookback definitions
Cell 26.10 Audit common target IDs
Cell 26.11 Audit temporal geometry
Cell 26.12 Audit right alignment
Cell 26.13 Audit continuity/no-gap rules
Cell 26.14 Audit feature/scaler equality
Cell 26.15 Audit architecture/parameter equality
Cell 26.16 Audit PE capacity
Cell 26.17 Audit training config
Cell 26.18 Check initialization/order fingerprints if available
Cell 26.19 Verify L144 reference reuse eligibility
Cell 26.20 Register L36 run
Cell 26.21 Seed + fresh L36 loaders/model
Cell 26.22 Train L36 via Training Engine
Cell 26.23 Verify L36 BEST
Cell 26.24 Register L72 run
Cell 26.25 Seed + fresh L72 loaders/model
Cell 26.26 Train L72 via Training Engine
Cell 26.27 Verify L72 BEST
Cell 26.28 Build run provenance
Cell 26.29 Build common-population metrics table
Cell 26.30 Compute pairwise effects
Cell 26.31 Compute diminishing-return diagnostics
Cell 26.32 Build runtime/memory diagnostics
Cell 26.33 Build gradient diagnostics
Cell 26.34 Evaluate hypotheses
Cell 26.35 Generate learning curves
Cell 26.36 Generate findings
Cell 26.37 Select S4 winner
Cell 26.38 Write winner JSON
Cell 26.39 Write Phase 27 reference update
Cell 26.40 Run S4 tests/discrepancies
Cell 26.41 Write summary/report
Cell 26.42 Register artifacts/checksums
Cell 26.43 Write README
Cell 26.44 Phase sign-off
```

---

# 135. Execution flow

```text
Verify Phase 25
        ↓
Load S3 winner
        ↓
Freeze feature variant + target scaling
        ↓
Load L36/L72/L144 contracts
        ↓
Load WINDOWPOP-v1
        ↓
Audit same target IDs
        ↓
Audit exact window geometry
        ↓
Audit right alignment
        ↓
Audit same feature/scaler/model/training config
        ↓
Verify L144 reference exact match
        ↓
Reuse L144
        ↓
Register L36
        ↓
Fresh seed/loaders/model
        ↓
Train + verify L36 BEST
        ↓
Register L72
        ↓
Fresh seed/loaders/model
        ↓
Train + verify L72 BEST
        ↓
Build common-population metrics
        ↓
Compute pairwise effects
        ↓
Analyze diminishing returns
        ↓
Analyze runtime/gradient context
        ↓
Select minimum-RMSE lookback
        ↓
Apply shortest-L exact-tie rule
        ↓
Update Phase 27 reference
        ↓
Write S4 artifacts
        ↓
SWEEP_S4_LOOKBACK-v1 sign-off
```

---

# 136. Fail-fast order

Before expensive training:

```text
1. Phase 25 sign-off
2. S3 winner identity
3. FV*/YS* lock
4. L definitions
5. WINDOWPOP-v1 validity
6. common target IDs
7. temporal geometry
8. right alignment
9. same feature/scaler
10. architecture equality
11. PE capacity
12. training config
13. L144 reuse eligibility
14. Test firewall
15. Registry readiness
```

---

# 137. Why common-population audit must happen first

This is the most dangerous S4 methodological mistake.

Without same target IDs:

```text
L36
L72
L144
```

would be evaluated on different observations.

Then lower RMSE cannot be attributed cleanly to lookback length.

---

# 138. Why right-alignment audit matters

Common bug:

```text
L36 = first 36 rows of L144
```

instead of:

```text
most recent 36 rows before target.
```

Correct:

```text
L36 == last 36 rows of L144 window.
```

---

# 139. Why same scaler matters

Lookback changes temporal extent, not feature scale definition.

Refitting X scaler per L would add:

```text
scaling-statistics factor.
```

Phase 9 already defined one frozen scaler per feature variant.

---

# 140. Why same y scaler matters

Same target transform keeps S4 centered on temporal context only.

---

# 141. Why parameter count must match

Lookback modifies sequence dimension only.

Any parameter count mismatch means something else changed.

---

# 142. Why no batch adaptation

If batch changed with L, comparison would mix:

```text
lookback
+
optimization batch size.
```

Batch sweep is Phase 29.

---

# 143. Why no attention extraction

Attention matrices grow as `L²`; collecting them during training would alter memory/runtime and make comparison less clean.

Attention stays OFF.

---

# 144. Winner verification checklist

Before writing `s4_lookback_winner.json`:

```text
[ ] L36 valid completed run.
[ ] L72 valid completed run.
[ ] L144 valid reused reference.
[ ] Same feature variant.
[ ] Same target scaling.
[ ] Same Train common IDs.
[ ] Same Validation common IDs.
[ ] Same population fingerprint.
[ ] Same X scaler.
[ ] Same target scaler/identity.
[ ] Same architecture/parameters.
[ ] Same Training Engine.
[ ] Same Metric version.
[ ] Window geometry verified.
[ ] Right alignment verified.
[ ] All BEST checkpoints verified.
[ ] All metrics use WINDOWPOP-v1.
[ ] No native metric enters ranking.
[ ] Full-precision RMSE used.
[ ] Pairwise effects computed.
[ ] Exact tie rule respected.
[ ] No Test.
```

---

# 145. Phase 27 handoff

Phase 27 receives:

```text
s4_lookback_winner.json
s4_reference_update.json
winner_run_id
winner_config_fingerprint
feature_variant_id
target_scaling_id
selected_lookback_id
population_fingerprint
```

and changes only:

```text
pooling.
```

---

# 146. If L36 wins

Current reference:

```text
FV*
YS*
L36
LAST_STEP
```

Phase 27 compares:

```text
LAST_STEP
vs
MEAN
```

at L36.

---

# 147. If L72 wins

Same but L72 fixed.

---

# 148. If L144 wins

Same but L144 fixed.

---

# 149. Reference reuse for Phase 27

S4 winner is already:

```text
LAST_STEP
```

so Phase 27 normally:

```text
reuse S4 winner
train only MEAN pooling.
```

---

# 150. Relationship with Phase 33/35/36 capacity sweeps

Lookback may interact with model capacity.

S4 winner is selected under:

```text
D64/H4/N2/FFN128
```

Later capacity sweeps may change relative optimum in theory.

Sequential design limitation must be documented.

---

# 151. Relationship with Phase 44 rolling-origin

S4 winner is still selected from one Validation period.

Rolling-origin later checks temporal robustness.

---

# 152. Relationship with Phase 46 multi-seed

S4 seed = 42.

Tiny margins may be stochastic.

---

# 153. Relationship with final attention analysis

Final attention analysis must use final selected lookback, which determines:

```text
attention matrix size
lag interpretation
heatmap axes.
```

Therefore Phase 26 selected L becomes important downstream lineage.

---

# 154. Lag-label handoff

Winner artifact should expose:

```text
lookback_steps
lookback_hours
sampling_minutes
```

so later attention Phase 54 can map positions to:

```text
minutes/hours before target.
```

---

# 155. Reproducibility metadata

New L36/L72 runs record:

```text
run_id
seed
environment
device
feature fingerprint
X scaler checksum
target transform checksum
window version
population fingerprint
lookback ID
model config fingerprint
initial state fingerprint if available
Training Engine fingerprint
best checkpoint checksum
history checksum
metric checksum
```

---

# 156. No fabricated runtime outputs

Do not pre-fill:

```text
winner
RMSE
best epochs
runtime
memory
gradient norms
native counts
common counts
```

before execution.

---

# 157. Phase sign-off

Create:

```text
phase_26_signoff.json
```

Minimum:

```text
phase = 26
phase_name = S4 Lookback sweep
sweep_version
sweep_id
source_s3_winner_run_id
feature_variant_id
target_scaling_id
l36_run_id
l72_run_id
l144_reference_run_id
new_run_ids
reused_run_ids
winner_lookback_id
winner_steps
winner_hours
winner_run_id
winner_rmse_wh
population_version
population_fingerprint
metric_version
window_alignment_status
fairness_audit_status
inherited_warnings
test_status
approved_for_phase27
overall_status
created_at
```

---

# 158. Acceptance checklist

```text
[ ] Phase 25 valid.
[ ] approved_for_phase26 = true.
[ ] S4 version declared.
[ ] S3 winner loaded.
[ ] Feature variant fixed.
[ ] Target scaling fixed.
[ ] L36/L72/L144 registered.
[ ] L36=36 steps=6h.
[ ] L72=72 steps=12h.
[ ] L144=144 steps=24h.
[ ] H1 fixed.
[ ] WB0 fixed.
[ ] WINDOWPOP-v1 loaded.
[ ] Common population fingerprint valid.
[ ] Same Train common IDs.
[ ] Same Validation common IDs.
[ ] Native counts recorded only as context.
[ ] Native metrics excluded from ranking.
[ ] No gaps crossed.
[ ] No padding.
[ ] No interpolation.
[ ] target-input_end = 10min.
[ ] L36 right-aligned inside L72.
[ ] L36 right-aligned inside L144.
[ ] L72 right-aligned inside L144.
[ ] Same target timestamps.
[ ] Same input_end timestamps.
[ ] Same feature order/fingerprint.
[ ] Same X scaler checksum.
[ ] Same target transform/scaler.
[ ] B64 fixed.
[ ] D64/H4/N2/FFN128 fixed.
[ ] Dropout .1 fixed.
[ ] GELU fixed.
[ ] LAST_STEP fixed.
[ ] Sinusoidal PE fixed.
[ ] PE capacity >=144.
[ ] No causal mask.
[ ] No padding mask.
[ ] Same parameter count.
[ ] Same architecture fingerprint.
[ ] AdamW/LR/WD/MSE fixed.
[ ] E50/patience10/clip1 fixed.
[ ] Seed42 fixed.
[ ] Training Engine fixed.
[ ] Metric version fixed.
[ ] L144 reference exact-match.
[ ] L144 not retrained.
[ ] L36 registered before training.
[ ] L72 registered before training.
[ ] L36 fresh loaders/model.
[ ] L72 fresh loaders/model.
[ ] No warm-start.
[ ] L36 trained via Training Engine.
[ ] L72 trained via Training Engine.
[ ] L36 BEST verified.
[ ] L72 BEST verified.
[ ] L144 BEST already verified.
[ ] All metrics use common population.
[ ] Full-precision RMSE used.
[ ] Winner=min RMSE.
[ ] Exact tie=shorter L.
[ ] Pairwise effects computed.
[ ] Diminishing-return diagnostics computed.
[ ] Runtime diagnostics computed.
[ ] Gradient diagnostics computed.
[ ] Complexity context labeled non-selection.
[ ] Hypothesis outcomes recorded.
[ ] Findings generated.
[ ] Inherited warnings propagated.
[ ] Winner artifact generated.
[ ] Phase 27 reference update generated.
[ ] LAST_STEP reuse identified.
[ ] No batch adaptation.
[ ] No attention extraction.
[ ] No score-based rerun.
[ ] No failed/SANITY run in ranking.
[ ] No Test access.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] Interaction limitation documented.
[ ] Figures source-generated.
[ ] Summary/report/README generated.
[ ] Discrepancy log generated.
[ ] Phase sign-off generated.
```

---

# 159. Acceptance criteria

Phase 26 chỉ PASS khi:

```text
S3-selected feature variant and target scaling are fixed.

L36/L72/L144 are defined correctly.

All three conditions use the exact same WINDOWPOP-v1 target IDs.

Window geometry is correct.

Right alignment across lookbacks is verified.

No padding/interpolation/gap crossing occurs.

Feature channels/scalers are identical.

Target transform is identical.

Architecture and parameter count are identical.

Training protocol and seed are identical.

L144 reference is valid and reused.

L36/L72 are fresh runs.

No warm-start occurs.

All BEST checkpoints are verified.

Only common-population Wh metrics enter ranking.

Validation RMSE Wh selects winner.

Exact tie selects shortest lookback.

Phase 27 reference is created.

No Test data is accessed.
```

---

# 160. Failure conditions

Phase 26 FAIL if:

```text
wrong S3 winner used

feature variant changes

target scaling changes

L definitions wrong

native populations used for ranking

common target IDs differ

window right alignment wrong

window crosses temporal gap

padding/interpolation used

feature order/scaler differs

architecture/parameter count differs

PE cannot support L144

batch/LR/dropout/etc differs

L144 reference mismatches but is reused

L144 retrained and best rerun cherry-picked

L36/L72 warm-start

one condition fails but winner still declared

RMSE rounded before ranking

runtime/native count used to override RMSE

Test used

score-based rerun occurs.
```

---

# 161. Common mistakes

## 161.1 Dùng native sample population

Sai nghiêm trọng cho S4 selection.

## 161.2 So L36 và L144 trên target timestamps khác nhau

Invalid.

## 161.3 Lấy first 36 rows của L144

Sai; phải lấy most recent 36 rows.

## 161.4 Pad L36 thành 144

Thay semantic input.

## 161.5 Fit scaler riêng theo lookback

Không.

## 161.6 Fit target scaler riêng theo L

Không.

## 161.7 Giảm batch ở L144

Thay factor.

## 161.8 Retrain L144 rồi chọn run tốt hơn

Hidden rerun bias.

## 161.9 Warm-start L72 từ L144

Confounded.

## 161.10 Dùng runtime để chọn L72 dù L144 RMSE tốt hơn

Sai primary metric.

## 161.11 Chọn L36 vì ít compute dù RMSE kém hơn

Parsimony chỉ exact tie.

## 161.12 Thêm causal mask vì sequence dài hơn

Không.

## 161.13 Bật attention collection để “xem L nào tốt”

Không Phase 26.

## 161.14 Thêm rolling mean features cho L36

New factor.

## 161.15 Mở Test để quyết định lookback

Forbidden.

---

# 162. Recommended execution pseudocode

```text
load_phase25_signoff()
assert_approved_for_phase26()

s3 = load_s3_winner()

FV = s3.feature_variant_id
YS = s3.target_scaling_id

lookbacks = ["L36", "L72", "L144"]

common_population = load_WINDOWPOP_v1()

audit_lookback_definitions()
audit_common_target_ids(common_population)
audit_window_geometry()
audit_right_alignment()
audit_same_feature_scaler_target_transform()
audit_same_architecture_training_config()
audit_pe_capacity(min_required=144)

l144_reference = resolve_s3_winner_run()
assert_exact_s4_reference_match(l144_reference)

results = {
    "L144": l144_reference
}

for L in ["L36", "L72"]:
    register_run(L)
    seed(42)

    loaders = build_fresh_loaders(
        feature_variant=FV,
        target_scaling=YS,
        lookback=L,
        population="WINDOWPOP-v1"
    )

    model = build_fresh_transformer(
        input_size=feature_count(FV)
    )

    result = TRAINING_ENGINE_v1.fit(...)
    verify_best_checkpoint(
        result,
        validation_population="WINDOWPOP-v1"
    )

    results[L] = result

metrics = build_common_population_metrics(results)
effects = compute_pairwise_lookback_effects(metrics)
diminishing = analyze_diminishing_returns(effects)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer_shortest=True
)

write_findings()
write_s4_winner(winner)
write_phase27_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 163. Definition of Done

\[
\boxed{
Three\ Lookbacks
+
Same\ Target\ IDs
+
Correct\ Window\ Geometry
+
Two\ Fresh\ Runs
+
One\ Valid\ Reused\ L144
+
Same\ Model/Training
+
Verified\ BEST\ Metrics
+
S4\ Winner
+
Phase27\ Reference
+
No\ Test
}
\]

---

# 164. Final status contract

```text
PHASE 26 tests LOOKBACK only.

Conditions:

L36
= 36 steps
= 6 hours

L72
= 72 steps
= 12 hours

L144
= 144 steps
= 24 hours

H1 fixed.

Feature variant:
comes from S2/S3 path.

Target scaling:
comes from S3 winner.

Population:
WINDOWPOP-v1
same target IDs for all L.

L144:
reuse S3 winner if exact match.

L36/L72:
fresh seed-42 runs.

Same:
feature channels
X scaler
target transform
architecture
parameter count
B64
AdamW
LR3e-4
WD1e-4
MSE
E50
patience10
clip1
Training Engine
Metric version.

Windowing:
right-aligned recent history,
no padding,
no interpolation,
no gap crossing.

Selection:
minimum verified common-population Validation RMSE Wh.

Exact tie:
prefer shorter lookback.

No native-population ranking.
No batch adaptation.
No warm-start.
No attention extraction.
No hidden rerun.
No Test.

After SWEEP_S4_LOOKBACK-v1 PASS:
update current reference
and proceed to
PHASE 27 — S5 Pooling sweep.
```

---

# 165. Final check

Correct workflow:

```text
Load S3 winner
→ Freeze FV* + YS*
→ Load L36/L72/L144
→ Load WINDOWPOP-v1
→ Audit same target IDs
→ Audit exact right-aligned windows
→ Audit same scalers/model/training
→ Reuse L144
→ Train fresh L36
→ Verify BEST
→ Train fresh L72
→ Verify BEST
→ Compare only common-population Wh metrics
→ Analyze pairwise gains/diminishing returns
→ Select min-RMSE lookback
→ Apply shortest-L exact-tie rule
→ Update Phase 27 reference
```

Incorrect workflow:

```text
use each lookback's native samples
→ change batch for long L
→ retrain L144
→ choose fastest/shortest model
→ inspect Test
```

Chỉ sau khi `SWEEP_S4_LOOKBACK-v1` được sign-off mới chuyển sang **PHASE 27 — S5 Pooling sweep**.
