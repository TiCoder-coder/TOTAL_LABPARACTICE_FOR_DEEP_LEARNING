# PHASE 23 — S1 FEATURE-SET SWEEP

## Kế hoạch controlled sweep cho Feature Set của Transformer Encoder

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference model:** `TRANSFORMER_B0-v1`  
**Sweep ID:** `S1_FEATURE_SET`  
**Output version:** `SWEEP_S1_FEATURESET-v1`  
**Phase trước:** `Phase_22_Learning-curve_diagnostics.md`

---

# 1. Vai trò của Phase 23

Phase 23 là controlled experiment đầu tiên trong chuỗi Transformer sweeps.

Mục tiêu duy nhất của phase này là trả lời:

> Với toàn bộ data protocol, windowing, scaling, Transformer architecture, optimizer, training protocol và seed được giữ nguyên, việc thay đổi **feature set** ảnh hưởng như thế nào tới Validation performance?

Phase 23 phải cô lập đúng một conceptual factor:

```text
FEATURE SET
```

và giữ tất cả yếu tố còn lại cố định.

Nguyên tắc trung tâm:

\[
\boxed{
One\ Factor
+
Same\ Samples
+
Same\ Training\ Protocol
+
Same\ Seed
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vị trí của Phase 23 trong master plan

```text
Phase 21
→ Transformer B0 official run

Phase 22
→ Learning-curve diagnostics

Phase 23
→ S1 Feature-set sweep

Phase 24
→ S2 Time-feature sweep
```

Phase 23 là sweep đầu tiên vì feature composition quyết định model đang nhìn thấy loại tín hiệu nào trước khi mình tối ưu các training/model hyperparameters sâu hơn.

---

# 3. Câu hỏi nghiên cứu của S1

Phase 23 cần trả lời ba câu hỏi chính.

## 3.1 Historical target có giúp dự báo không?

So:

```text
FS0_TF1
vs
FS1_TF1
```

Khác biệt duy nhất:

```text
FS1 thêm historical Appliances.
```

Đây là phép ablation quan trọng nhất của S1.

## 3.2 Random-control features có đóng góp gì không?

So:

```text
FS1_TF1
vs
FS2_TF1
```

Khác biệt duy nhất:

```text
FS2 thêm rv1 + rv2.
```

## 3.3 Feature set nào tạo Validation RMSE thấp nhất dưới B0 training protocol?

So toàn bộ:

```text
FS0_TF1
FS1_TF1
FS2_TF1
```

Primary metric:

```text
Validation RMSE Wh.
```

---

# 4. Feature-set definitions

Feature registry từ `FEATURESETS-v1` là source of truth.

Không viết lại feature lists thủ công trong Phase 23.

## 4.1 FS0 — Exogenous-only

```text
FS0
=
raw exogenous predictors
```

Không có:

```text
historical Appliances
rv1
rv2
```

TF1 vẫn được bật.

Canonical variant:

```text
FS0_TF1
```

## 4.2 FS1 — Autoregressive + Exogenous

```text
FS1
=
FS0
+
historical Appliances
```

Không có:

```text
rv1
rv2.
```

Canonical variant:

```text
FS1_TF1
```

Đây là B0 reference từ Phase 21.

## 4.3 FS2 — FS1 + Random Controls

```text
FS2
=
FS1
+
rv1
+
rv2
```

Canonical variant:

```text
FS2_TF1
```

---

# 5. Time-feature configuration phải cố định

Tất cả S1 runs dùng:

```text
TF1
```

gồm đúng:

```text
hour_sin
hour_cos
dow_sin
dow_cos
weekend
```

Không được so TF0 trong Phase 23.

TF0 vs TF1 thuộc:

```text
Phase 24.
```

---

# 6. Controlled comparison matrix

| Variant | Historical Appliances | rv1/rv2 | Time features |
|---|---:|---:|---:|
| FS0_TF1 | No | No | TF1 |
| FS1_TF1 | Yes | No | TF1 |
| FS2_TF1 | Yes | Yes | TF1 |

Conceptual deltas:

```text
FS0 → FS1
isolates historical target contribution.

FS1 → FS2
isolates random-control contribution.
```

`FS0 → FS2` không phải one-component contrast vì thay đổi đồng thời historical target và random controls.

---

# 7. Working hypotheses

Các hypothesis này là pre-analysis expectations, không phải kết quả.

## H-S1-01 — Historical target contribution

\[
RMSE(FS1\_TF1)
<
RMSE(FS0\_TF1)
\]

would support the hypothesis that recent historical energy consumption contains useful autoregressive information beyond exogenous features.

Status trước execution:

```text
UNTESTED.
```

## H-S1-02 — Random-control robustness

Nếu:

```text
FS2_TF1
```

không cải thiện hoặc làm xấu Validation RMSE so với FS1_TF1, kết quả phù hợp với kỳ vọng rằng random-control variables không mang tín hiệu dự báo ổn định.

Nếu FS2 thắng:

```text
không được tự động gọi rv1/rv2 là meaningful predictors.
```

Phải flag:

```text
RANDOM_CONTROL_GAIN
```

và giữ kết quả để kiểm chứng robustness ở các phase sau.

## H-S1-03 — Feature-set effect exists

Nếu Validation RMSE khác nhau giữa variants, feature composition có ảnh hưởng thực nghiệm dưới B0 protocol.

Không làm significance test trong Phase 23 vì mỗi variant mới chỉ có một seed.

---

# 8. Preconditions bắt buộc

Phase 23 chỉ bắt đầu khi:

```text
Phase 21 = PASS
Phase 22 = PASS
```

và:

```text
TRANSFORMER_B0-v1 = valid
LEARNING_DIAGNOSTICS-v1 = PASS hoặc PASS_WITH_WARNING không critical
```

Không được bắt đầu nếu Phase 22 có unresolved:

```text
CRITICAL
PIPELINE_INTEGRITY_ANOMALY.
```

---

# 9. Upstream contracts bắt buộc

Phải verify:

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
TRANSFORMER_B0-v1
LEARNING_DIAGNOSTICS-v1
```

---

# 10. Không được sửa upstream artifacts

Phase 23 không được:

```text
recompute split
refit scalers
change windows
drop samples
change feature ordering
edit Transformer implementation
change Training Engine
```

Nếu upstream artifact mismatch:

```text
STOP.
```

---

# 11. Frozen factors trong S1

Mọi yếu tố sau phải giống B0:

```text
TF1

L144

H1

YS1

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

learning rate = 3e-4

weight decay = 1e-4

MSE

max epochs = 50

patience = 10

min_delta = 0

gradient clipping max_norm = 1.0

scheduler = None

mixed precision = False

seed = 42

TRAINING_ENGINE-v1

METRICS-v1
```

---

# 12. Swept factor duy nhất

```text
feature_variant_id
```

Allowed values:

```text
FS0_TF1
FS1_TF1
FS2_TF1
```

No other semantic field may differ.

---

# 13. Existing B0 reuse rule

`FS1_TF1` đã được train chính thức ở Phase 21 dưới B0.

Nếu và chỉ nếu Phase 21 run có exact matching:

```text
data config
model config
training config
seed
Training Engine
metric version
population fingerprint
```

thì Phase 23 **reuse** Phase 21 run làm reference row.

Không retrain FS1_TF1 chỉ để tạo sweep table.

---

# 14. Vì sao phải reuse B0 reference?

Reusing avoids:

```text
duplicate scientific run
extra stochastic realization
hidden rerun bias
unnecessary compute
```

Reference identity phải được khóa bằng:

```text
run_id
config fingerprint
checkpoint checksum.
```

---

# 15. Khi nào không được reuse B0?

Nếu B0 không match S1 contract do:

```text
different seed
different batch
different scaler version
different Training Engine
different metric version
different population
different architecture
```

thì:

```text
S1 cannot silently reuse it.
```

Phải resolve protocol/version issue trước.

---

# 16. Number of runs Phase 23 thực sự cần train

Expected:

```text
FS1_TF1
→ REUSE Phase 21 B0

FS0_TF1
→ NEW official sweep run

FS2_TF1
→ NEW official sweep run
```

Do đó Phase 23 thông thường tạo:

```text
2 new training runs
+
1 reused reference run.
```

Không hard-code đây là 3 new runs.

---

# 17. Sweep identity

Canonical:

```text
sweep_id = S1_FEATURE_SET
```

Recommended experiment family:

```text
TRANSFORMER_SWEEP_S1_FEATURE_SET
```

Output version:

```text
SWEEP_S1_FEATURESET-v1
```

---

# 18. Run labels

Recommended:

```text
S1_FS0_TF1__L144__YS1__B64__S42

S1_FS1_TF1__REFERENCE_B0

S1_FS2_TF1__L144__YS1__B64__S42
```

Actual unique IDs vẫn do Registry quản lý.

---

# 19. Same sample-population requirement

Hard:

```text
WINDOWPOP-v1
```

must be identical for all three variants.

For each split:

```text
Train target IDs identical
Validation target IDs identical.
```

---

# 20. Feature variants không được drop sample riêng

Nếu FS2 có nonfinite feature và code muốn drop target:

```text
FAIL.
```

Không được tạo population riêng cho FS2.

---

# 21. Population identity audit

Explicitly verify:

```text
FS0 target IDs
==
FS1 target IDs
==
FS2 target IDs
```

for Train and Validation.

---

# 22. Why common population matters?

Nếu sample sets khác nhau, Validation RMSE difference có thể do:

```text
different targets
```

rather than feature-set effect.

---

# 23. X-scaler rule theo feature variant

Mỗi variant dùng đúng frozen X scaler bundle từ `SCALING-v1`:

```text
XSCALER__FS0_TF1

XSCALER__FS1_TF1

XSCALER__FS2_TF1
```

Không dùng chung scaler superset rồi subset thủ công.

---

# 24. Y scaler phải giống nhau

Tất cả S1 runs dùng:

```text
YSCALER__YS1
```

same identity/checksum.

---

# 25. Why X scalers differ nhưng sweep vẫn fair?

Feature sets có channels khác nhau, nên mỗi feature variant cần scaler mapping đúng feature list của nó.

Fairness được giữ vì:

```text
mỗi X scaler fit Train-only
same scaling policy
same split
same scaler version
```

Chỉ feature composition khác.

---

# 26. Feature-order fingerprint

Mỗi run phải bind đúng:

```text
feature_fingerprint
```

của variant.

Hard fail nếu runtime X order không match Registry.

---

# 27. Runtime feature count

Expected counts from earlier design may be:

```text
FS0_TF1 ≈ 30
FS1_TF1 ≈ 31
FS2_TF1 ≈ 33
```

nhưng Phase 23 không hard-code các count này.

Source of truth:

```text
FEATURESETS-v1 runtime registry.
```

---

# 28. Input projection implication

Changing F changes first projection:

```text
Linear(F,64)
```

Therefore trainable parameter count will differ slightly across feature sets.

Đây là inherent consequence của feature-set sweep, không phải hidden architecture change.

---

# 29. Parameter-count fairness note

Record:

```text
trainable_parameters
```

for each variant.

Do not normalize RMSE by parameter count.

Do not attempt to parameter-match models by changing d_model.

---

# 30. Model architecture beyond input projection must be identical

Hard verify:

```text
d_model
heads
layers
FFN
dropout
activation
pooling
norm
PE
mask policy
regression head semantics
```

same across variants.

---

# 31. Fresh run rule

For each NEW S1 run:

```text
set seed 42
↓
create fresh variant-specific Train DataLoader
↓
create fresh variant-specific Validation DataLoader
↓
instantiate fresh Transformer
↓
move to selected device
↓
build MSE
↓
build AdamW
↓
run TRAINING_ENGINE-v1.
```

---

# 32. Không warm-start từ B0

Hard:

```text
FS0 and FS2 start from fresh initialization.
```

Do not load Phase 21 Transformer weights.

---

# 33. Why no warm-start?

Warm-start would make feature-set comparison depend on:

```text
training order
B0 learned representation
incompatible input projection
```

and confound the feature-set effect.

---

# 34. Same seed semantics

Every new run uses:

```text
seed 42.
```

This controls initialization/shuffle policy as much as practical under fixed environment.

---

# 35. Same seed does not mean identical parameter tensors across F

Because input projection shapes differ, RNG consumption and total parameter layout can differ.

Do not claim perfect matched initialization.

Fairness means:

```text
same seed policy and same training protocol.
```

---

# 36. Experiment Registry lifecycle

Each new variant:

```text
REGISTERED
→ RUNNING
→ COMPLETED
```

or:

```text
FAILED.
```

---

# 37. Run eligibility

New sweep runs:

```text
execution_type = TRAINING
experiment_family = TRANSFORMER_SWEEP_S1_FEATURE_SET
sweep_id = S1_FEATURE_SET
eligible_for_model_selection = False initially
```

Set eligible only after run verification.

---

# 38. B0 reused reference registry role

Reference row:

```text
source = REUSED_BASELINE
source_run_id = Phase21 run_id
```

Do not duplicate metric artifact.

Sweep table may point to same source run.

---

# 39. Test firewall

Hard:

```text
Test DataLoader not created/iterated.
Test target locked.
Test metric unavailable.
```

S1 selection uses Validation only.

---

# 40. Training Engine contract

All NEW runs must use:

```text
TRAINING_ENGINE-v1
```

No variant-specific custom training loop.

---

# 41. Training objective

Same:

```text
MSE in YS1 model target space.
```

---

# 42. Primary sweep metric

Hard:

```text
best_validation_rmse_wh
```

from verified best checkpoint.

---

# 43. Secondary metrics

```text
best_validation_mae_wh
best_validation_r2
best_epoch
epochs_completed
stop_reason
gradient diagnostics
runtime
parameter count
```

---

# 44. Winner selection rule

Primary:

\[
\boxed{
winner
=
argmin(
Validation\ RMSE\ Wh
)
}
\]

using verified best checkpoints and full-precision values.

---

# 45. Exact-tie rule

Nếu two variants have exactly equal full-precision RMSE:

```text
prefer fewer model input features
```

parsimony order:

```text
FS0
<
FS1
<
FS2
```

This tie rule applies only to exact equality.

Record tie event explicitly.

---

# 46. Không dùng MAE để overturn lower RMSE

If RMSE values differ:

```text
lower RMSE wins
```

even if MAE ranking differs.

MAE/R² are interpretation signals only.

---

# 47. No minimum-improvement threshold required for winner

Do not invent:

```text
must improve by 1%
```

unless master protocol is amended.

Strict minimum RMSE determines empirical S1 winner.

---

# 48. Empirical winner vs interpretation

If:

```text
FS2 wins
```

record:

```text
S1 empirical winner = FS2_TF1
```

but also flag random-control dependence for later robustness interpretation.

Do not silently discard the result.

---

# 49. Random-control warning policy

If FS2 beats FS1:

```text
RANDOM_CONTROL_GAIN = True
```

Create a diagnostic note:

```text
Improvement may reflect chance correlation/noise exploitation and requires downstream robustness scrutiny.
```

Do not claim rv1/rv2 are physically meaningful.

---

# 50. Phase 24 feature-set reference

After S1 selection:

```text
S1 winner feature set
```

becomes the reference FS for Phase 24 Time-feature sweep, unless a formal protocol rule says otherwise.

Phase 24 compares:

```text
FSx_TF0
vs
FSx_TF1
```

with `FSx` fixed to S1 winner.

---

# 51. Reference-update artifact

Create:

```text
s1_reference_update.json
```

Fields:

```text
previous_reference = FS1_TF1
s1_winner_variant
selected_feature_set_id
time_feature_state = TF1
selection_metric
winner_run_id
winner_rmse_wh
selection_rule
random_control_warning
approved_for_phase24
```

---

# 52. Do not mutate Phase 21 B0 identity

Even if S1 winner differs:

```text
TRANSFORMER_B0-v1
```

remains historical baseline.

Phase 23 creates new:

```text
current_reference_config
```

for next sweep.

---

# 53. Controlled-delta audit

For every comparison verify only intended feature fields differ.

## FS0 vs FS1

Expected delta:

```text
+ historical Appliances
```

Nothing else.

## FS1 vs FS2

Expected delta:

```text
+ rv1
+ rv2
```

Nothing else.

---

# 54. Config-delta artifact

Create:

```text
s1_config_delta_audit.csv
```

Fields:

```text
comparison_pair
config_field
left_value
right_value
expected_to_differ
observed_diff
status
```

---

# 55. Hard forbidden differences

Examples:

```text
lookback differs
batch differs
seed differs
LR differs
dropout differs
Training Engine differs
population differs
YS scaler differs
metric version differs
```

Any unexpected difference:

```text
comparison invalid.
```

---

# 56. Preflight sweep matrix

Create:

```text
s1_run_matrix.csv
```

Fields:

```text
sweep_id
variant_id
feature_set_id
time_feature_id
source_type
source_run_id
requires_new_training
feature_count
feature_fingerprint
x_scaler_id
y_scaler_id
lookback
horizon
population_fingerprint
batch_size
seed
model_config_id
training_config_id
status
```

---

# 57. Recommended run order

To minimize accidental dependence:

```text
1. Verify/reuse FS1_TF1 B0 reference.

2. Run FS0_TF1.

3. Run FS2_TF1.
```

Order must not affect configs.

Could run FS0/FS2 in reverse or parallel if infrastructure supports independent reproducible runs.

---

# 58. Parallel execution

Allowed only if:

```text
run IDs independent
artifact directories independent
device resources sufficient
same environment
no shared mutable state.
```

Sequential execution is simpler and preferred for notebook workflow.

---

# 59. No cross-run state reuse

After each run:

```text
delete/discard model object
discard optimizer
discard DataLoaders
reset/reseed for next run.
```

---

# 60. Config freeze before each run

Variant-specific config is generated from canonical B0 + one feature-set substitution.

Recommended pattern:

```text
base_config = frozen B0 config
variant_config = deep-copy(base_config)
variant_config.feature_variant_id = ...
```

Then audit diff.

---

# 61. Do not hand-edit many config fields

Use centralized config builder.

This reduces hidden drift.

---

# 62. Variant-specific model input size

Derived from feature registry after config freeze.

Example:

```text
model.input_size = feature_count(variant)
```

This is allowed because it is mechanically required by feature-set change.

---

# 63. Variant-specific X scaler

Derived from variant registry.

Not an independently tuned choice.

---

# 64. Best-checkpoint verification per new run

After each new run:

```text
fresh Transformer
↓
strict load BEST
↓
full Validation
↓
inverse YS1
↓
METRICS-v1
↓
verify recorded best metrics.
```

---

# 65. New-run critical artifacts

Each NEW run must produce standard Phase 19 run artifacts:

```text
config.json
status.json
training_history.csv
training.log
best_checkpoint.pt
last_checkpoint.pt
best_validation_metrics.json
best_validation_predictions.csv
runtime_summary.json
```

plus diagnostics per run.

---

# 66. Sweep does not need to copy B0 files

For reused FS1:

```text
store references
```

not duplicate large checkpoints.

---

# 67. Sweep-level metric table

Create:

```text
s1_feature_set_metrics.csv
```

Rows:

```text
FS0_TF1
FS1_TF1
FS2_TF1
```

Fields:

```text
variant_id
run_id
source_type
feature_count
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

# 68. Ranking rule

Sort ascending:

```text
validation_rmse_wh
```

Tie:

```text
fewer features first.
```

---

# 69. Pairwise effect table

Create:

```text
s1_feature_set_pairwise_effects.csv
```

Required comparisons:

```text
FS0_TF1 → FS1_TF1
FS1_TF1 → FS2_TF1
FS0_TF1 → FS2_TF1
```

---

# 70. RMSE effect formula

For left/reference A and right B:

\[
\Delta RMSE_{A\rightarrow B}
=
RMSE_A-RMSE_B
\]

Positive:

```text
B improves RMSE.
```

---

# 71. Relative improvement

\[
Improvement\%
=
100
\times
\frac{RMSE_A-RMSE_B}{RMSE_A}
\]

if:

```text
RMSE_A > 0.
```

---

# 72. MAE effect

Analogous:

\[
\Delta MAE
=
MAE_A-MAE_B.
\]

---

# 73. R² effect

\[
\Delta R^2
=
R^2_B-R^2_A.
\]

No percentage.

---

# 74. FS0 → FS1 interpretation

This comparison directly measures the empirical value of adding:

```text
historical Appliances
```

under current model/training protocol.

Do not say:

```text
historical Appliances causes better forecasts
```

in causal sense.

Say:

```text
including historical Appliances was associated with lower/higher Validation error in this controlled configuration.
```

---

# 75. FS1 → FS2 interpretation

This comparison measures effect of adding:

```text
rv1
rv2
```

to otherwise identical FS1.

If no improvement:

```text
consistent with random controls not providing useful predictive signal.
```

If improvement:

```text
flag for robustness.
```

---

# 76. FS0 → FS2 interpretation

Use as total feature-set contrast only.

Do not attribute entire difference to any one added component.

---

# 77. No statistical significance claim

One seed each.

Do not calculate t-test using epochs as samples.

Do not call a 0.3% difference “statistically significant”.

---

# 78. Practical effect reporting

Report:

```text
absolute RMSE difference Wh
relative RMSE difference %
MAE difference Wh
R² difference
```

without arbitrary practical-significance threshold.

---

# 79. Learning-curve comparison inside S1

Can create sweep learning curves for diagnostic context.

But winner selection remains:

```text
best Validation RMSE Wh
```

per run.

---

# 80. No checkpoint cherry-picking across variants

Each variant contributes only:

```text
its own verified BEST checkpoint
```

according to Training Engine selection rule.

---

# 81. No “same epoch” comparison requirement

Variants may early-stop at different epochs.

Compare:

```text
each variant's best verified Validation checkpoint.
```

---

# 82. Why this is valid?

Early stopping is identical protocol across runs.

Best epoch is an outcome, not a controlled factor.

---

# 83. Epoch-cap caveat

All use:

```text
max_epochs 50
patience 10.
```

If one variant still improves at epoch 50, record; do not extend it here.

---

# 84. Gradient diagnostics

Record per variant:

```text
global max grad norm
mean fraction clipped
nonfinite events
```

No config changes.

---

# 85. Runtime diagnostics

Record:

```text
training duration
mean epoch duration
```

but do not use runtime to choose S1 winner.

---

# 86. Parameter count diagnostics

Record actual counts.

No winner selection based on parameter count unless exact RMSE tie.

---

# 87. Random-control sensitivity finding

Create boolean:

```text
random_control_gain
```

and fields:

```text
rmse_gain_vs_fs1_wh
rmse_gain_vs_fs1_pct
```

if FS2 wins FS1.

---

# 88. Historical-target contribution finding

Create:

```text
historical_target_gain
```

and:

```text
rmse_gain_fs0_to_fs1_wh
rmse_gain_fs0_to_fs1_pct
```

---

# 89. Hypothesis outcome artifact

Create:

```text
s1_hypothesis_outcomes.csv
```

Fields:

```text
hypothesis_id
comparison
expected_direction
observed_left_rmse
observed_right_rmse
observed_delta
outcome
interpretation
status
```

Allowed `outcome`:

```text
SUPPORTED
NOT_SUPPORTED
INCONCLUSIVE_TIE
```

This is descriptive support under one-seed Validation context, not universal proof.

---

# 90. Learning Diagnostics integration

If Phase 22 had S1-related hypothesis:

```text
source_hypothesis_id
```

should be referenced in Phase 23 outcome table.

Do not change S1 candidates based on the hypothesis.

---

# 91. Sweep winner artifact

Create:

```text
s1_feature_set_winner.json
```

Minimum:

```text
sweep_id
sweep_version
selection_metric
selection_direction
tie_rule
winner_variant_id
winner_feature_set_id
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_variant_id
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
random_control_warning
population_fingerprint
metric_version
test_status
status
```

---

# 92. Winner must be based on verified source runs only

No:

```text
FAILED
SANITY
unverified checkpoint
```

may enter ranking.

---

# 93. Sweep completeness gate

S1 cannot select winner until:

```text
FS0 valid
FS1 reference valid
FS2 valid
```

all three metrics are available.

---

# 94. Failure of one variant

If one new run fails technically:

```text
do not select winner from remaining two.
```

Resolve/rerun the failed config under same run-resume policy or create a properly logged technical rerun.

---

# 95. Score-based rerun forbidden

Do not rerun variant merely because its score is poor.

---

# 96. Technical rerun allowed

Only for:

```text
OOM due infrastructure anomaly
interrupt
corrupt checkpoint
documented software failure
```

If changing batch/config:

```text
not a rerun of same S1 condition.
```

Requires protocol decision.

---

# 97. OOM policy

If FS0 or FS2 B64 normal training OOM:

```text
run FAILS.
```

Do not silently switch B32.

B32 belongs Phase 29.

---

# 98. Nonfinite policy

NaN/Inf:

```text
FAIL run
```

No skipping.

---

# 99. No emergency LR change

No.

---

# 100. No emergency dropout change

No.

---

# 101. No hidden early-stop change

No.

---

# 102. Test firewall remains absolute

Phase 23 does not:

```text
load Test targets
calculate Test predictions
calculate Test RMSE
inspect Test ranking
```

---

# 103. No use of Test to break Validation tie

Hard forbidden.

---

# 104. Validation reuse policy

Validation is used for S1 model-selection decision, which is expected.

This is development selection, not final evaluation.

Later robustness/rolling-origin steps reduce over-reliance on one Validation split.

---

# 105. Multiple sequential sweeps caution

Because many later sweeps use the same Validation set, project must:

```text
log every tested option
avoid hidden experiments
use Experiment Registry
perform rolling-origin robustness later
keep final Test untouched.
```

Phase 23 begins this controlled sequence.

---

# 106. Sweep manifest

Create:

```text
s1_feature_set_sweep_manifest.json
```

Minimum fields:

```text
sweep_version = SWEEP_S1_FEATURESET-v1
sweep_id = S1_FEATURE_SET
experiment_family
reference_run_id
reference_variant = FS1_TF1
candidate_variants
new_runs_required
reused_runs
frozen_fields
swept_field = feature_variant_id
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule
population_version
population_fingerprint
metric_version
training_engine_version
seed = 42
test_access = forbidden
sweep_status
created_at
```

---

# 107. Sweep contract

Create:

```text
s1_feature_set_sweep_contract.json
```

Must explicitly state:

```text
Only feature-set composition may differ.
TF1 fixed.
WINDOWPOP-v1 fixed.
YS1 fixed.
Transformer architecture fixed except mechanically required input projection width.
Training protocol fixed.
Seed fixed.
Validation RMSE Wh selects winner.
Test forbidden.
```

---

# 108. Preflight audit

Create:

```text
s1_feature_set_preflight_audit.csv
```

Checks:

```text
phase22_pass
B0_reference_valid
feature_registry_valid
all_three_variants_registered
TF1_fixed
population_fixed
Y_scaler_fixed
Training_Engine_fixed
metric_fixed
seed_fixed
test_locked
status
```

---

# 109. Population audit

Create:

```text
s1_feature_set_population_audit.csv
```

Fields:

```text
variant_id
split_id
sample_count
first_sample_id
last_sample_id
population_fingerprint
id_set_matches_reference
ordered_ids_match_reference
status
```

---

# 110. Scaler audit

Create:

```text
s1_feature_set_scaler_audit.csv
```

Fields:

```text
variant_id
x_scaler_id
x_scaler_checksum
feature_fingerprint
y_scaler_id
y_scaler_checksum
x_train_only_verified
y_train_only_verified
status
```

---

# 111. Architecture audit

Create:

```text
s1_feature_set_architecture_audit.csv
```

Fields:

```text
variant_id
input_size
d_model
num_heads
num_layers
ffn_dim
dropout
activation
pooling
norm_policy
causal_mask
padding_mask
trainable_parameters
only_expected_differences
status
```

---

# 112. Training-config audit

Create:

```text
s1_feature_set_training_audit.csv
```

Fields:

```text
variant_id
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

# 113. Run provenance table

Create:

```text
s1_feature_set_run_provenance.csv
```

Fields:

```text
variant_id
run_id
source_type
config_fingerprint
feature_fingerprint
population_fingerprint
best_checkpoint_sha256
history_sha256
metric_artifact
prediction_artifact
status
```

---

# 114. Sweep comparison table

Create:

```text
s1_feature_set_metrics.csv
```

This is the primary machine-readable result table.

---

# 115. Pairwise-effect table

Create:

```text
s1_feature_set_pairwise_effects.csv
```

---

# 116. Hypothesis outcomes

Create:

```text
s1_hypothesis_outcomes.csv
```

---

# 117. Winner JSON

Create:

```text
s1_feature_set_winner.json
```

---

# 118. Reference update

Create:

```text
s1_reference_update.json
```

---

# 119. Diagnostic findings

Create:

```text
s1_feature_set_findings.csv
```

Possible finding codes:

```text
HISTORICAL_TARGET_GAIN
HISTORICAL_TARGET_NO_GAIN
RANDOM_CONTROL_GAIN
RANDOM_CONTROL_NO_GAIN
FEATURE_SET_RANKING_STABLE
METRIC_RANKING_DIVERGENCE
GRADIENT_BEHAVIOR_DIFFERENCE
RUNTIME_DIFFERENCE
INCONCLUSIVE_TIE
```

---

# 120. Metric-ranking divergence

Example:

```text
FS1 best RMSE
FS0 best MAE
```

Record:

```text
METRIC_RANKING_DIVERGENCE.
```

Winner still follows RMSE.

---

# 121. Sweep learning curves

Recommended figures:

```text
S1_01_validation_rmse_by_epoch.png
S1_02_validation_mae_by_epoch.png
S1_03_train_loss_by_epoch.png
S1_04_gradient_clipping_fraction.png
S1_05_best_validation_metrics.png
```

---

# 122. Primary figure

`S1_01_validation_rmse_by_epoch.png`

show:

```text
FS0_TF1
FS1_TF1
FS2_TF1
```

with each best epoch marked.

---

# 123. Best-metric figure

`S1_05_best_validation_metrics.png`

should clearly show best-checkpoint Validation RMSE across three variants.

---

# 124. Figures are secondary to metrics table

Winner always computed from:

```text
s1_feature_set_metrics.csv
```

not visually estimated.

---

# 125. No smoothing for selection

Raw curves only for primary source.

Optional smoothing display is unnecessary here.

---

# 126. Human-readable report

Create:

```text
s1_feature_set_sweep_report.md
```

Sections:

```text
1. Objective
2. Controlled variable
3. Frozen configuration
4. Feature-set definitions
5. Run provenance
6. Population/scaling fairness
7. Validation metrics
8. Pairwise feature effects
9. Historical-target interpretation
10. Random-control interpretation
11. Learning-curve context
12. S1 winner
13. Limitations
14. Phase 24 reference handoff
```

---

# 127. Report must distinguish empirical result vs interpretation

Example:

```text
Observed:
FS1 RMSE is X Wh lower than FS0.

Interpretation:
Adding historical Appliances improved Validation forecasting performance under this configuration.

Limitation:
Single seed and one Validation split.
```

No causal/general universal claim.

---

# 128. Single-seed limitation

Mandatory:

```text
All new S1 runs use seed 42 only.
```

No mean±std.

---

# 129. Validation-selection limitation

Mandatory:

```text
S1 winner is a Validation-selected development configuration.
```

It is not final Test winner.

---

# 130. Sequential-search limitation

Mandatory:

```text
S1 is the first of many controlled Validation-based sweeps.
```

Later:

```text
rolling-origin robustness
multi-seed final evaluation
untouched Test
```

are required before final conclusions.

---

# 131. Output directory

```text
artifacts/
└── sweeps/
    └── S1_feature_set/
        ├── s1_feature_set_sweep_manifest.json
        ├── s1_feature_set_sweep_contract.json
        ├── s1_feature_set_preflight_audit.csv
        ├── s1_run_matrix.csv
        ├── s1_config_delta_audit.csv
        ├── s1_feature_set_population_audit.csv
        ├── s1_feature_set_scaler_audit.csv
        ├── s1_feature_set_architecture_audit.csv
        ├── s1_feature_set_training_audit.csv
        ├── s1_feature_set_run_provenance.csv
        ├── s1_feature_set_metrics.csv
        ├── s1_feature_set_pairwise_effects.csv
        ├── s1_hypothesis_outcomes.csv
        ├── s1_feature_set_findings.csv
        ├── s1_feature_set_winner.json
        ├── s1_reference_update.json
        ├── s1_feature_set_sweep_tests.csv
        ├── s1_feature_set_discrepancies.json
        ├── s1_feature_set_sweep_summary.json
        ├── s1_feature_set_sweep_report.md
        ├── figures/
        │   ├── S1_01_validation_rmse_by_epoch.png
        │   ├── S1_02_validation_mae_by_epoch.png
        │   ├── S1_03_train_loss_by_epoch.png
        │   ├── S1_04_gradient_clipping_fraction.png
        │   └── S1_05_best_validation_metrics.png
        ├── README_S1_FEATURE_SET_SWEEP.md
        └── phase_23_signoff.json
```

Individual NEW run artifacts remain under:

```text
artifacts/runs/<run_id>/
```

Do not duplicate them inside sweep directory.

---

# 132. Required outputs

```text
O23.1  S1 sweep registry/manifest
O23.2  Frozen sweep contract
O23.3  Preflight audit
O23.4  Run matrix
O23.5  Config-delta audit
O23.6  Population audit
O23.7  Scaler audit
O23.8  Architecture audit
O23.9  Training-config audit
O23.10 Run provenance table
O23.11 Verified FS0 run
O23.12 Reused FS1 B0 reference
O23.13 Verified FS2 run
O23.14 Metrics/ranking table
O23.15 Pairwise-effect table
O23.16 Hypothesis outcomes
O23.17 Findings table
O23.18 Winner artifact
O23.19 Reference-update artifact
O23.20 Diagnostic figures
O23.21 Sweep tests
O23.22 Discrepancy log
O23.23 Sweep summary
O23.24 Human-readable sweep report
O23.25 README
O23.26 Phase sign-off
```

---

# 133. Sweep summary JSON

Create:

```text
s1_feature_set_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
reference_run_id
variants
new_runs
reused_runs
primary_metric
ranking
winner
runner_up
pairwise_effects
historical_target_gain
random_control_gain
critical_warnings
phase24_reference
test_status
overall_status
```

---

# 134. Sweep test suite

Create:

```text
s1_feature_set_sweep_tests.csv
```

Recommended IDs:

```text
S1T23-001 ...
```

---

# 135. S1T23-001

Phase 22 PASS or non-critical PASS_WITH_WARNING.

---

# 136. S1T23-002

S1 sweep version declared.

---

# 137. S1T23-003

Exactly three feature variants registered.

---

# 138. S1T23-004

TF1 fixed across all variants.

---

# 139. S1T23-005

L144 fixed.

---

# 140. S1T23-006

H1 fixed.

---

# 141. S1T23-007

YS1 fixed.

---

# 142. S1T23-008

WB0 fixed.

---

# 143. S1T23-009

WINDOWPOP-v1 fixed.

---

# 144. S1T23-010

B64 fixed.

---

# 145. S1T23-011

D64 fixed.

---

# 146. S1T23-012

H4 fixed.

---

# 147. S1T23-013

N2 fixed.

---

# 148. S1T23-014

FFN128 fixed.

---

# 149. S1T23-015

Dropout .1 fixed.

---

# 150. S1T23-016

GELU fixed.

---

# 151. S1T23-017

LAST_STEP fixed.

---

# 152. S1T23-018

AdamW fixed.

---

# 153. S1T23-019

LR 3e-4 fixed.

---

# 154. S1T23-020

WD 1e-4 fixed.

---

# 155. S1T23-021

MSE fixed.

---

# 156. S1T23-022

Epoch cap 50 fixed.

---

# 157. S1T23-023

Patience 10 fixed.

---

# 158. S1T23-024

Gradient clipping 1.0 fixed.

---

# 159. S1T23-025

Seed 42 fixed.

---

# 160. S1T23-026

Training Engine version fixed.

---

# 161. S1T23-027

Metric version fixed.

---

# 162. S1T23-028

B0 FS1 reference config exactly matches sweep contract.

---

# 163. S1T23-029

B0 reference reused rather than silently retrained.

---

# 164. S1T23-030

FS0 feature fingerprint valid.

---

# 165. S1T23-031

FS1 feature fingerprint valid.

---

# 166. S1T23-032

FS2 feature fingerprint valid.

---

# 167. S1T23-033

FS0 X scaler bound correctly.

---

# 168. S1T23-034

FS1 X scaler bound correctly.

---

# 169. S1T23-035

FS2 X scaler bound correctly.

---

# 170. S1T23-036

Same Y scaler across all variants.

---

# 171. S1T23-037

Train target ID sets identical.

---

# 172. S1T23-038

Validation target ID sets identical.

---

# 173. S1T23-039

Population fingerprint identical.

---

# 174. S1T23-040

FS0→FS1 delta only historical Appliances.

---

# 175. S1T23-041

FS1→FS2 delta only rv1/rv2.

---

# 176. S1T23-042

No unexpected config difference.

---

# 177. S1T23-043

FS0 run registered before training.

---

# 178. S1T23-044

FS2 run registered before training.

---

# 179. S1T23-045

FS0 fresh seed/loaders/model.

---

# 180. S1T23-046

FS2 fresh seed/loaders/model.

---

# 181. S1T23-047

No warm-start from B0.

---

# 182. S1T23-048

FS0 trained through TRAINING_ENGINE-v1.

---

# 183. S1T23-049

FS2 trained through TRAINING_ENGINE-v1.

---

# 184. S1T23-050

FS0 best checkpoint verified.

---

# 185. S1T23-051

FS2 best checkpoint verified.

---

# 186. S1T23-052

All sweep result rows use verified best checkpoints.

---

# 187. S1T23-053

Validation RMSE Wh full precision available for all variants.

---

# 188. S1T23-054

Ranking ascending RMSE correct.

---

# 189. S1T23-055

Tie rule implemented correctly.

---

# 190. S1T23-056

Pairwise RMSE deltas correct.

---

# 191. S1T23-057

Pairwise percentage improvements correct.

---

# 192. S1T23-058

MAE deltas correct.

---

# 193. S1T23-059

R² deltas correct.

---

# 194. S1T23-060

Historical-target hypothesis outcome computed.

---

# 195. S1T23-061

Random-control hypothesis outcome computed.

---

# 196. S1T23-062

FS2 gain triggers warning if observed.

---

# 197. S1T23-063

Winner JSON points to valid completed run.

---

# 198. S1T23-064

Reference update points to winner feature set.

---

# 199. S1T23-065

No Test artifact accessed.

---

# 200. S1T23-066

No score-based rerun occurred.

---

# 201. S1T23-067

No hidden hyperparameter changed.

---

# 202. S1T23-068

No failed/sanity run enters ranking.

---

# 203. S1T23-069

Single-seed limitation documented.

---

# 204. S1T23-070

Validation-only limitation documented.

---

# 205. S1T23-071

Figures generated from metrics/history sources.

---

# 206. S1T23-072

Phase 24 handoff config created.

---

# 207. S1T23-073

Phase sign-off created.

---

# 208. Sweep discrepancy taxonomy

```text
REFERENCE_RUN_MISMATCH
FEATURE_REGISTRY_MISMATCH
FEATURE_FINGERPRINT_MISMATCH
UNEXPECTED_FEATURE_DELTA
POPULATION_MISMATCH
SCALER_MISMATCH
Y_SCALER_MISMATCH
MODEL_CONFIG_DRIFT
TRAINING_CONFIG_DRIFT
SEED_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
RUN_FAILURE
CHECKPOINT_VERIFICATION_FAILURE
INCOMPLETE_SWEEP
RANKING_ERROR
PAIRWISE_EFFECT_ERROR
RANDOM_CONTROL_GAIN_WARNING
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 209. Discrepancy log

Create:

```text
s1_feature_set_discrepancies.json
```

Fields:

```text
id
severity
category
variant_or_pair
expected
actual
impact
recommended_action
resolved
resolution_notes
```

---

# 210. Severity rules

```text
CRITICAL
MAJOR
MODERATE
MINOR
INFO
```

Examples:

```text
CRITICAL:
population mismatch
Training Engine mismatch
Test access

MAJOR:
wrong feature fingerprint
checkpoint verification failure

MODERATE:
random-control gain warning

INFO:
runtime differences.
```

---

# 211. Sweep status model

## PASS

```text
all three variants valid
fairness audits pass
winner selected
reference update generated
Test untouched
```

## PASS_WITH_WARNING

Example:

```text
FS2 wins and RANDOM_CONTROL_GAIN is flagged
```

while all methodological contracts remain valid.

## FAIL

Examples:

```text
one variant failed
population differs
unexpected config drift
reference B0 mismatch
Test accessed
```

---

# 212. Critical rule về FS2

`RANDOM_CONTROL_GAIN` by itself không làm sweep invalid.

Nó là:

```text
scientific/robustness warning
```

not pipeline failure.

---

# 213. Winner acceptance

Winner is empirical Validation winner under S1.

Do not call it:

```text
final optimal feature set.
```

Correct language:

```text
S1-selected feature set
```

or:

```text
current feature-set reference.
```

---

# 214. Experiment history preservation

All losing variants remain in Registry.

Do not delete checkpoints/results because they lost.

---

# 215. No retroactive B0 change

If FS0 wins:

```text
Phase 21 B0 is still FS1_TF1.
```

Phase 23 simply updates future reference.

---

# 216. No redefinition of FS IDs

FS0, FS1, FS2 definitions remain locked from Phase 7.

---

# 217. No new engineered feature in Phase 23

If diagnostic suggests adding another lag/calendar feature:

```text
not allowed here.
```

Requires protocol amendment/another dedicated experiment.

---

# 218. No target-time feature addition

No.

---

# 219. No manual lag columns

Historical context remains sequence axis from Phase 10.

---

# 220. No PCA/feature selection algorithm

This is predefined feature-set ablation, not automated feature selection.

---

# 221. No SHAP/feature importance

Not Phase 23.

---

# 222. No attention interpretation

Not Phase 23.

---

# 223. No feature-standardization changes

Same SCALING-v1 policy.

---

# 224. No robust scaling

Not here.

---

# 225. No changing missing-value policy

No.

---

# 226. No sample weighting

No.

---

# 227. No augmentation

No.

---

# 228. Recommended notebook structure

```text
Cell 23.1  Phase title
Cell 23.2  Verify Phase 22 sign-off
Cell 23.3  Declare SWEEP_S1_FEATURESET-v1
Cell 23.4  Load B0 reference run
Cell 23.5  Load FEATURESETS-v1 registry
Cell 23.6  Build S1 run matrix
Cell 23.7  Verify FS0/FS1/FS2 definitions
Cell 23.8  Verify TF1 fixed
Cell 23.9  Population identity audit
Cell 23.10 Scaler binding audit
Cell 23.11 Config-delta audit
Cell 23.12 Verify B0 reuse eligibility
Cell 23.13 Register FS0 run
Cell 23.14 Seed + build fresh FS0 loaders/model
Cell 23.15 Execute FS0 through Training Engine
Cell 23.16 Verify FS0 best checkpoint
Cell 23.17 Register FS2 run
Cell 23.18 Seed + build fresh FS2 loaders/model
Cell 23.19 Execute FS2 through Training Engine
Cell 23.20 Verify FS2 best checkpoint
Cell 23.21 Build run provenance table
Cell 23.22 Build S1 metrics table
Cell 23.23 Rank variants
Cell 23.24 Compute pairwise effects
Cell 23.25 Evaluate hypotheses
Cell 23.26 Generate learning curves
Cell 23.27 Analyze random-control warning
Cell 23.28 Write winner artifact
Cell 23.29 Write Phase 24 reference update
Cell 23.30 Build tests/discrepancies
Cell 23.31 Write sweep summary/report
Cell 23.32 Register artifacts/checksums
Cell 23.33 Write README
Cell 23.34 Phase sign-off
```

---

# 229. Execution flow

```text
Verify Phase 22
        ↓
Load frozen B0 reference
        ↓
Load feature-set registry
        ↓
Build S1 matrix
        ↓
Verify one-factor deltas
        ↓
Verify same target population
        ↓
Verify scaler bindings
        ↓
Reuse FS1 B0 reference
        ↓
Fresh FS0 run
        ↓
Verify FS0 best checkpoint
        ↓
Fresh FS2 run
        ↓
Verify FS2 best checkpoint
        ↓
Build three-row metrics table
        ↓
Pairwise feature effects
        ↓
Rank by Validation RMSE
        ↓
Select S1 winner
        ↓
Flag random-control warning if needed
        ↓
Update current reference for Phase 24
        ↓
Write sweep artifacts
        ↓
SWEEP_S1_FEATURESET-v1 sign-off
```

---

# 230. Fail-fast order

Before launching expensive training, verify in this order:

```text
1. Phase 22 status

2. B0 reference identity

3. Feature registry

4. Population identity

5. X/Y scaler bindings

6. Frozen config equality

7. Only intended feature differences

8. Test firewall

9. Registry readiness
```

Only then train FS0/FS2.

---

# 231. Why config-delta audit trước training?

It prevents wasting an entire run due to silent differences such as:

```text
wrong dropout
wrong seed
wrong scaler
wrong batch.
```

---

# 232. Best-checkpoint verification workflow

For each new variant:

```text
training complete
↓
load config from run artifact
↓
fresh model
↓
strict-load best checkpoint
↓
fresh sequential Validation loader
↓
full prediction
↓
inverse YS1
↓
METRICS-v1
↓
compare to stored best metrics
↓
PASS
```

---

# 233. Population-comparison workflow

For all variants:

```text
load ordered Validation sample_idx
↓
assert unique
↓
assert same count
↓
assert set equal
↓
assert canonical ordered IDs equal after sort/order rule
↓
assert population fingerprint equal
```

---

# 234. Pairwise-effect interpretation workflow

## FS0 → FS1

Ask:

```text
Did adding historical Appliances improve RMSE?
By how many Wh?
By what percentage?
What happened to MAE and R²?
```

## FS1 → FS2

Ask:

```text
Did adding random controls improve RMSE?
If yes, how large is the gain?
Does MAE agree?
Should a robustness warning be emitted?
```

---

# 235. Learning-curve use inside S1

Compare curves only to understand:

```text
training stability
best epoch
plateau behavior
gradient behavior
```

Do not override winner selection.

---

# 236. No winner based on convergence speed

A model that reaches best RMSE earlier is not automatically better.

---

# 237. No winner based on parameter count except exact RMSE tie

Correct.

---

# 238. No winner based on runtime

Correct.

---

# 239. No winner based on training loss

Correct.

---

# 240. No winner based on R² if RMSE ranking differs

Correct.

---

# 241. Winner verification checklist

Before writing `s1_feature_set_winner.json`:

```text
[ ] All three variants available.
[ ] All runs verified.
[ ] Same Validation IDs.
[ ] Same metric version.
[ ] Same target unit.
[ ] Same seed policy.
[ ] No unexpected config drift.
[ ] RMSE values full precision.
[ ] Rank computed programmatically.
[ ] Tie rule applied if needed.
[ ] Winner run status valid.
[ ] Test untouched.
```

---

# 242. Phase 24 handoff rules

`Phase 24 — S2 Time-feature sweep` receives:

```text
s1_feature_set_winner.json
s1_reference_update.json
winner run_id
winner config fingerprint
selected feature_set_id
```

Phase 24 then tests:

```text
same selected FS
TF0 vs TF1
```

while holding everything else fixed.

---

# 243. If S1 winner = FS0_TF1

Phase 24 compares:

```text
FS0_TF0
vs
FS0_TF1.
```

---

# 244. If S1 winner = FS1_TF1

Phase 24 compares:

```text
FS1_TF0
vs
FS1_TF1.
```

---

# 245. If S1 winner = FS2_TF1

Phase 24 compares:

```text
FS2_TF0
vs
FS2_TF1.
```

and carry:

```text
random_control_warning
```

into handoff.

---

# 246. Reuse rule for Phase 24

Whichever `*_TF1` run wins S1 can be reused as TF1 reference in Phase 24 if all other Phase 24 config fields match exactly.

This avoids unnecessary duplicate runs.

---

# 247. Experiment Registry handoff

Phase 24 must know:

```text
reference_run_id
reference_variant
reference_config_fingerprint
sweep_parent = S1_FEATURE_SET
```

---

# 248. Relationship with Phase 42 candidate synthesis

All S1 variants remain available in Registry.

Even though Phase 24 follows the S1 winner, Phase 42 can later inspect full experimental history and robustness context.

Do not delete non-winners.

---

# 249. Relationship with Phase 44 rolling-origin

If FS2/random controls win, rolling-origin robustness becomes particularly important to assess whether the gain persists over alternate chronological origins.

---

# 250. Relationship with Phase 46 multi-seed

Single-seed S1 winner is not seed-robust evidence.

Final multi-seed runs occur only after final config lock.

---

# 251. Relationship with Phase 47 Test

No Test access occurs in S1.

The S1 winner remains:

```text
Validation-selected development choice.
```

---

# 252. Phase 23 limitations section bắt buộc

Report must state:

```text
single seed

single Validation split

sequential model-selection process

feature-set differences alter input dimensionality and therefore input-projection parameter count

FS2 contains random-control variables and any gain requires cautious interpretation

no Test evidence.
```

---

# 253. Reproducibility metadata

Every NEW S1 run records:

```text
run_id
seed
environment ID
device
code fingerprint
feature fingerprint
scaler checksum
population fingerprint
model config fingerprint
Training Engine fingerprint
best checkpoint checksum
history checksum
metric artifact checksum
```

---

# 254. No fabricated runtime output

Before execution, plan must not invent:

```text
metrics
best epochs
sample counts
run duration
parameter counts
gradient norms
winner
```

All are runtime outputs.

---

# 255. README content

Create:

```text
README_S1_FEATURE_SET_SWEEP.md
```

Must explain:

```text
Purpose

Why S1 is one-factor controlled

FS0/FS1/FS2 definitions

Why TF1 is fixed

Why B0 FS1 is reused

Population fairness

Scaler fairness

Parameter-count nuance

Training protocol

Winner rule

Random-control warning

Hypothesis outcomes

Limitations

Phase 24 handoff

No Test access.
```

---

# 256. Phase sign-off

Create:

```text
phase_23_signoff.json
```

Minimum:

```text
phase = 23
phase_name = S1 Feature-set sweep
sweep_version
sweep_id
source_b0_run_id
new_run_ids
winner_variant
winner_run_id
winner_rmse_wh
population_fingerprint
metric_version
fairness_audit_status
random_control_warning
test_status
approved_for_phase24
overall_status
created_at
```

---

# 257. Phase 23 acceptance checklist

```text
[ ] Phase 21 PASS.
[ ] Phase 22 PASS/no unresolved critical issue.
[ ] SWEEP_S1_FEATURESET-v1 declared.
[ ] S1_FEATURE_SET sweep ID declared.
[ ] FS0/FS1/FS2 loaded from registry.
[ ] TF1 fixed.
[ ] L144 fixed.
[ ] H1 fixed.
[ ] YS1 fixed.
[ ] WB0 fixed.
[ ] WINDOWPOP-v1 fixed.
[ ] B64 fixed.
[ ] D64/H4/N2/FFN128 fixed.
[ ] Dropout .1 fixed.
[ ] GELU fixed.
[ ] LAST_STEP fixed.
[ ] AdamW fixed.
[ ] LR 3e-4 fixed.
[ ] WD 1e-4 fixed.
[ ] MSE fixed.
[ ] E50 fixed.
[ ] Patience 10 fixed.
[ ] Gradient clipping 1 fixed.
[ ] Seed 42 fixed.
[ ] Training Engine fixed.
[ ] Metric version fixed.
[ ] Test locked.
[ ] FS1 B0 exact-match reuse verified.
[ ] FS1 not silently retrained.
[ ] FS0 feature fingerprint valid.
[ ] FS1 feature fingerprint valid.
[ ] FS2 feature fingerprint valid.
[ ] X scalers variant-specific and correct.
[ ] Y scaler identical.
[ ] Train target IDs identical.
[ ] Validation target IDs identical.
[ ] Population fingerprint identical.
[ ] FS0→FS1 delta only historical Appliances.
[ ] FS1→FS2 delta only rv1/rv2.
[ ] No unexpected config differences.
[ ] FS0 run registered before training.
[ ] FS0 uses fresh loaders/model.
[ ] FS0 uses seed 42.
[ ] FS0 no warm-start.
[ ] FS0 trained via Training Engine.
[ ] FS0 best checkpoint verified.
[ ] FS2 run registered before training.
[ ] FS2 uses fresh loaders/model.
[ ] FS2 uses seed 42.
[ ] FS2 no warm-start.
[ ] FS2 trained via Training Engine.
[ ] FS2 best checkpoint verified.
[ ] No failed/sanity run enters sweep table.
[ ] All three verified metrics available.
[ ] Full-precision RMSE used.
[ ] Ranking generated programmatically.
[ ] Tie rule implemented.
[ ] Pairwise effects calculated correctly.
[ ] Historical-target hypothesis evaluated.
[ ] Random-control hypothesis evaluated.
[ ] Random-control warning emitted if required.
[ ] Winner artifact generated.
[ ] Winner points to valid run.
[ ] Winner remains Validation-selected only.
[ ] Reference-update artifact generated.
[ ] Phase 24 feature-set reference defined.
[ ] Learning curves source-generated.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] Parameter-count nuance documented.
[ ] No hidden rerun.
[ ] No ad hoc hyperparameter change.
[ ] No Test access.
[ ] Sweep summary generated.
[ ] Report generated.
[ ] README generated.
[ ] Discrepancies recorded.
[ ] Phase 23 sign-off generated.
```

---

# 258. Acceptance criteria

Phase 23 chỉ PASS khi:

```text
All three predefined feature variants are represented.

Phase 21 FS1 reference is reused only if exact-match valid.

FS0 and FS2 are trained from fresh seed-42 runs.

Only feature-set composition differs.

Train and Validation target IDs are identical.

Variant-specific X scalers and common Y scaler are correct.

All runs use the same Transformer/training protocol.

Best checkpoints are verified.

Validation RMSE Wh is the sole primary selection metric.

Pairwise effects are computed from source artifacts.

Random-control gain is transparently flagged if present.

A single S1 winner/current reference is recorded.

No hidden rerun/tuning occurs.

Test remains untouched.

Phase 24 handoff is complete.
```

---

# 259. Khi nào Phase 23 FAIL?

```text
FS1 reference does not exactly match sweep contract.

FS1 is retrained without explicit reason and best rerun is cherry-picked.

Feature lists are hand-written and drift from FEATURESETS-v1.

Different target IDs across variants.

Different population fingerprint.

Wrong scaler bound to a variant.

YS scaler differs.

TF0 used in one run.

Lookback differs.

Seed differs.

Model/training hyperparameter differs unexpectedly.

FS0 or FS2 warm-start from B0.

One variant fails but winner is still selected.

Failed/SANITY run enters ranking.

RMSE is rounded before ranking.

MAE overrides a lower RMSE.

Test is used as tie-breaker.

Score-based reruns occur.

FS2 gain is automatically interpreted as meaningful physical signal.
```

---

# 260. Các lỗi thường gặp

## 260.1 Train lại FS1 nhiều lần rồi chọn run tốt nhất

Sai. Reuse official B0 if exact match.

## 260.2 Dùng một X scaler chung cho FS0/FS1/FS2

Không theo SCALING-v1.

## 260.3 FS0 có fewer features nên model parameter count khác rồi cố thay d_model để equalize

Sai. Điều đó tạo thêm factor.

## 260.4 Thấy FS0 yếu nên tăng LR riêng FS0

Không.

## 260.5 Thấy FS2 mạnh nên kết luận rv1/rv2 quan trọng

Không. Đây là random-control warning.

## 260.6 Dùng cùng model object và thay input projection giữa runs

Không. Fresh model mỗi run.

## 260.7 Warm-start FS2 từ FS1

Confounded.

## 260.8 Drop sample có NaN riêng cho FS2

Population drift.

## 260.9 So RMSE trên khác target IDs

Invalid.

## 260.10 Dùng Test để xác nhận FS winner

Forbidden.

## 260.11 Chọn winner bằng train loss

Sai.

## 260.12 Chọn winner bằng best epoch sớm nhất

Sai.

## 260.13 Chọn winner bằng runtime nhanh nhất

Sai.

## 260.14 Thêm time feature mới vì nghĩ FS0 cần hỗ trợ

Phase 24 hoặc protocol amendment.

## 260.15 Chạy nhiều seed trong S1 rồi chọn seed tốt

Không. Final seed robustness later.

---

# 261. Recommended execution pseudocode

```text
load_phase22_signoff()
assert_no_critical_findings()

load_feature_set_registry()
load_b0_reference()

build_s1_matrix(
    variants = [FS0_TF1, FS1_TF1, FS2_TF1]
)

audit_population_identity()
audit_scaler_bindings()
audit_config_deltas()

assert_b0_reference_exact_match()

results = {}
results["FS1_TF1"] = reuse_b0_result()

for variant in ["FS0_TF1", "FS2_TF1"]:
    register_run(variant)
    seed(42)
    loaders = build_fresh_loaders(variant)
    model = build_fresh_transformer(input_size=registry_F(variant))
    result = TRAINING_ENGINE_v1.fit(...)
    verify_best_checkpoint(result)
    results[variant] = result

metrics = build_verified_metrics_table(results)
effects = compute_pairwise_effects(metrics)
winner = select_min_rmse(metrics)

write_hypothesis_outcomes()
write_random_control_warning_if_needed()
write_reference_update(winner)
write_sweep_report()
write_signoff()
```

---

# 262. Definition of Done

Phase 23 hoàn thành khi:

\[
\boxed{
Three\ Registered\ Feature\ Conditions
+
Two\ Fresh\ Runs
+
One\ Valid\ Reused\ B0
+
Same\ Population
+
Same\ Training
+
Verified\ Metrics
+
S1\ Winner
+
Phase24\ Reference
+
No\ Test
}
\]

---

# 263. Final status contract

```text
PHASE 23 tests FEATURE SET only.

Variants:
FS0_TF1
FS1_TF1
FS2_TF1

FS1_TF1:
reuse official Transformer B0 if exact match.

FS0_TF1:
new fresh run.

FS2_TF1:
new fresh run.

Everything else:
frozen to B0.

Population:
WINDOWPOP-v1 identical.

Scaling:
variant-specific frozen X scaler,
same YS1 scaler.

Training:
TRAINING_ENGINE-v1
seed 42
B64
AdamW
LR 3e-4
WD 1e-4
MSE
E50
patience 10
clip 1.0.

Selection:
minimum verified Validation RMSE Wh.

FS2 gain:
record honestly,
flag random-control warning,
do not claim physical meaning.

No Test.

After SWEEP_S1_FEATURESET-v1 PASS:
update current reference
and proceed to
PHASE 24 — S2 Time-feature sweep.
```

---

# 264. Final check

Thứ tự đúng của Phase 23 là:

```text
Freeze
→ Audit
→ Reuse valid B0
→ Train only missing variants
→ Verify checkpoints
→ Compare same targets
→ Compute feature effects
→ Select by RMSE
→ Record warnings
→ Update Phase 24 reference
```

Không phải:

```text
train thử
→ nhìn score
→ sửa config
→ train lại
→ chọn run đẹp nhất.
```

Chỉ sau khi `SWEEP_S1_FEATURESET-v1` được sign-off mới chuyển sang **PHASE 24 — S2 Time-feature sweep**.
