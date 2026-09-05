# PHASE 45 — FINAL MODEL LOCK

## Kế hoạch khóa Final Transformer configuration, final-fit protocol và reproducibility contract trước Three-Seed Final Runs

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Upstream robustness:** `ROLLING_ORIGIN-v1`  
**Phase ID:** `PHASE_45_FINAL_MODEL_LOCK`  
**Output version:** `FINAL_MODEL_LOCK-v1`  
**Phase trước:** `Phase_44_Rolling-origin_robustness.md`  
**Phase sau:** `Phase_46_Three-seed_final_runs.md`

---

# 1. Vai trò của Phase 45

Phase 45 là **governance gate cuối cùng trước final model training**.

Phase này không train model mới, không mở thêm sweep, không xem Test, và không thay đổi candidate dựa trên cảm tính.

Mục tiêu:

```text
1. Nhận recommended Transformer từ Phase44.
2. Xác minh toàn bộ lineage S1–S19 + Phase42–44.
3. Freeze exact final Transformer scientific configuration.
4. Freeze exact final data/preprocessing contract.
5. Freeze exact final training recipe cho Phase46.
6. Chuyển từ development-time early stopping sang final full-pretest refit bằng epoch count đã khóa.
7. Freeze three final seeds.
8. Freeze checkpoint/artifact semantics.
9. Freeze no-Test/no-retuning rules.
10. Tạo immutable Final Model Lock package cho Phase46–57.
```

Nguyên tắc trung tâm:

\[
\boxed{
No\ New\ Search
+
One\ Recommended\ Transformer
+
Full\ Lineage\ Verification
+
Predeclared\ Final\ Refit\ Recipe
+
Immutable\ Configuration
+
No\ Test
}
\]

---

# 2. Phase45 không phải training phase

Hard:

```text
new_training_runs = 0
new_validation_runs = 0
new_test_runs = 0
```

Phase45 chỉ:

```text
read
verify
resolve
freeze
fingerprint
sign off.
```

---

# 3. Vì sao cần Final Model Lock riêng

Sau Phase44, ta đã có:

```text
best Transformer candidate theo rolling-origin robustness
```

nhưng chưa đủ để chạy final seeds một cách khoa học.

Phải khóa thêm:

```text
final pre-Test training population
final scaler-fit population
final epoch count
final seed set
final run identity
final checkpoint semantics
final prediction/evaluation prohibition
final attention compatibility
```

Nếu những thứ này bị quyết định trong lúc đang chạy Phase46 hoặc sau khi thấy Test, sẽ tạo hidden researcher degrees of freedom.

---

# 4. Inputs bắt buộc từ Phase44

Load:

```text
phase_44_signoff.json
rolling_origin_recommended_transformer.json
phase45_final_model_lock_handoff.json
transformer_robustness_ranking.csv
rolling_origin_fold_metrics.csv
rolling_origin_pooled_metrics.csv
rolling_origin_inner_best_epochs.csv
model_family_robustness_comparison.csv
rolling_origin_summary.json
```

Hard:

```text
approved_for_phase45 = true.
```

---

# 5. Inputs bắt buộc từ upstream lineage

Phase45 phải có access machine-readable tới:

```text
Phase42 frozen Transformer shortlist
Phase41 S19 boundary sensitivity
Phase40 RevIN outcome
S1–S18 winner/reference artifacts
SCALING-v1
WINDOWS-v1
WINDOWPOP-v1
METRICS-v1
EXPERIMENTS-v1
TRANSFORMER_IMPL-v1
TRAINING_ENGINE-v1
```

Không được chỉ tin một report prose.

---

# 6. Final model family

Hard:

```text
model_family = Transformer Encoder for regression.
```

Dù Phase44 tuned LSTM có thể có pooled RMSE thấp hơn, coursework primary model vẫn là Transformer.

Nếu LSTM tốt hơn:

```text
record as comparison finding
```

không tự động đổi final model family.

---

# 7. Final candidate source

Final Transformer configuration phải lấy chính xác từ:

```text
rolling_origin_recommended_transformer.json
```

Canonical ID:

```text
FINAL_TR_CANDIDATE
```

Source candidate có thể là:

```text
TR_C0_PRIMARY
TR_C1_LOCAL_ALT
TR_C2_LOCAL_ALT
```

tùy Phase44 ranking runtime.

Không hard-code `TR_C0`.

---

# 8. Final candidate selection không được làm lại

Phase45 không được:

```text
rerank candidates bằng MAE
rerank bằng runtime
rerank bằng parameter count
rerank bằng original Validation
rerank bằng attention
rerank bằng LSTM comparison
```

Phase44 ranking đã frozen.

---

# 9. Phase44 ranking rule phải được verify

Expected:

```text
1. lowest pooled outer RMSE Wh
2. exact tie → lower worst-fold RMSE
3. exact tie → lower fold-RMSE SD
4. exact tie → earlier Phase42 shortlist position
```

Phase45 chỉ kiểm rule được thực thi đúng.

---

# 10. Final scientific configuration fields

Freeze exact values runtime cho:

```text
feature_set_id
time_feature_id
feature_variant_id
feature_names
feature_order
target_scaling_id
lookback_id
lookback_steps
horizon = 1
pooling_id
activation_id
train_batch_size
learning_rate_id
learning_rate
weight_decay_id
weight_decay
dropout_id
dropout_probability
d_model_id
d_model
head_id
num_heads
head_dim
layer_id
num_layers
ffn_id
ffn_dim
loss_id
loss_name
huber_delta_if_applicable
epoch_cap_id
max_epochs
patience
min_delta
gradient_clip_id
gradient_clip_enabled
gradient_clip_max_norm_if_applicable
RevIN_id
RevIN_enabled
RevIN exact config if active
boundary_protocol = WB0
positional_encoding
norm_style
attention mask policy
output head semantics
Training Engine version
Metric version.
```

---

# 11. Final task contract

Hard:

```text
X_{t-L+1:t}
→
Appliances_{t+1}
```

with:

```text
H=1 = 10 minutes ahead.
```

No task reformulation after lock.

---

# 12. Final boundary protocol

Hard:

```text
WB0
```

because S19 treated WB1 as strict-isolation sensitivity unless a formal amendment says otherwise.

Phase45 must include S19 evidence but not switch to WB1 silently.

---

# 13. S19 sensitivity evidence is part of final lock context

Record:

```text
WB0 common-pop RMSE
WB1 common-pop RMSE
coverage loss under WB1
sensitivity interpretation
protocol_amendment_required
```

Hard:

```text
protocol_amendment_required = false
```

before final lock.

---

# 14. Final feature information contract

Use exact final candidate feature configuration.

Historical Appliances:

```text
included or excluded exactly according to final candidate.
```

No additional feature added for final training.

---

# 15. Final target scaling

Use exact final candidate:

```text
YS0
or
YS1.
```

Phase46 cannot switch.

---

# 16. Final lookback

Use exact final candidate:

```text
L36
L72
or
L144.
```

No re-evaluation.

---

# 17. Final Transformer architecture

Freeze:

```text
input projection
fixed sinusoidal positional encoding
d_model*
heads*
layers*
FFN*
activation*
dropout*
POST_NORM
pooling*
Linear(D,1)
no decoder
no output activation
no causal mask
no padding mask
attention-aware implementation.
```

---

# 18. Final attention compatibility contract

Final model must preserve:

```text
standard forward path
+
inspection forward path
```

with:

```text
per-layer attention [B,H,L,L]
average_attn_weights=False.
```

This is required for Phase52–57.

No final refactor may remove attention extraction.

---

# 19. Final RevIN contract

If:

```text
RN0
```

RevIN disabled.

If:

```text
RN1
```

freeze exact S18 semantics:

```text
eps=1e-5
affine=True
mean centering
unbiased=False
stats detached
continuous-signal scope
time-feature pass-through
historical Appliances target denorm
X-target → raw Wh → y_model bridge.
```

---

# 20. Final loss contract

Use candidate’s exact:

```text
MSE
or
Huber.
```

If Huber:

```text
delta=1.0 y_model-space.
```

No final loss switch.

---

# 21. Final optimizer contract

Use exact:

```text
AdamW
LR*
WD*
same parameter-group policy
same betas/eps
```

from candidate.

No scheduler/warmup unless candidate explicitly contains it; current protocol expects:

```text
scheduler=None
warmup=None.
```

---

# 22. Final gradient clipping

Use exact:

```text
GC0
or
GC1 max_norm1.
```

No final safety-based override.

Both still maintain non-finite gradient guard.

---

# 23. Final epoch-cap field is not final refit epoch count

Important distinction:

```text
candidate max_epochs
```

is the maximum allowed during development inner selection.

Phase46 full pre-Test final refit needs:

```text
FINAL_REFIT_EPOCHS
```

precomputed and frozen in Phase45.

These are not the same concept.

---

# 24. Why final refit should use Train + Validation

After model/config selection is fully frozen, original Validation no longer needs to remain a model-selection holdout.

Final three-seed training should use:

```text
all pre-Test development history
=
original Train + original Validation
```

to maximize information before final Test evaluation.

Test remains untouched.

---

# 25. Final fit dataset

Define:

```text
FINAL_DEV_REGION-v1
=
original Train rows
+
original Validation rows.
```

Target IDs:

```text
FINAL_DEV_TARGET_IDS
```

must be based on:

```text
final candidate lookback
H1
WB0
continuity rules
```

and exclude original Test targets.

---

# 26. No Test rows in final training inputs

Hard:

```text
max final training target timestamp
<
first Test target timestamp.
```

Input windows for final training targets also cannot extend into Test.

---

# 27. Final pre-Test scaler fitting

For Phase46:

```text
fit X scaler on FINAL_DEV_REGION-v1 allowed rows
fit Y scaler on FINAL_DEV_REGION-v1 allowed training target/history basis
```

according to candidate YS/Phase9 semantics.

This creates:

```text
FINAL_SCALING-v1
```

for final three-seed refit.

---

# 28. Why final scaler is refit

After model lock, all pre-Test development data are legally available for final model fitting.

Using only original Train scaler would discard information.

Using Train+Validation:

```text
does not leak Test.
```

---

# 29. Final scaler must be identical across seeds

Scaler fitting is deterministic and independent of seed.

Phase46 three seeds use exactly the same:

```text
X scaler checksums
Y scaler checksum/identity.
```

Hard.

---

# 30. Final epoch-count source

Phase45 derives `FINAL_REFIT_EPOCHS` only from the Phase44 recommended Transformer’s **three inner best epochs**.

Let:

```text
e1 = recommended candidate inner best epoch in RO1
e2 = recommended candidate inner best epoch in RO2
e3 = recommended candidate inner best epoch in RO3.
```

---

# 31. Final epoch aggregation rule

Lock:

\[
E_{final}
=
median(e_1,e_2,e_3)
\]

Since K=3:

```text
sort [e1,e2,e3]
take middle integer.
```

Canonical field:

```text
FINAL_REFIT_EPOCHS = median_RO_inner_best_epochs.
```

---

# 32. Why median epoch is used

Median:

```text
1. uses only pre-Test robustness evidence
2. is resistant to one unusually early/late fold
3. avoids Test/Validation reselection during final fit
4. gives one fixed epoch budget shared by all final seeds
5. remains within candidate max-epoch cap.
```

---

# 33. No rounding issue for median

All inner best epochs are integers.

For exactly three values:

```text
median is an observed integer epoch.
```

No averaging/rounding required.

---

# 34. Epoch validity assertions

Hard:

```text
1 <= e1,e2,e3 <= candidate.max_epochs

1 <= FINAL_REFIT_EPOCHS <= candidate.max_epochs.
```

---

# 35. Missing inner epoch is critical

If Phase44 recommended candidate does not have all 3 valid inner best epochs:

```text
STOP.
```

Do not improvise a final epoch count.

---

# 36. No original Validation early stopping in Phase46

Once Phase45 locks:

```text
FINAL_REFIT_EPOCHS
```

Phase46 trains on all pre-Test development data for exactly that many epochs.

No separate Validation holdout.

No early stopping.

---

# 37. No Test early stopping

Absolutely forbidden.

---

# 38. No seed-specific epoch selection

All seeds:

```text
42
123
2026
```

train exact same `FINAL_REFIT_EPOCHS`.

Do not allow:

```text
seed42 28 epochs
seed123 34 epochs
seed2026 21 epochs
```

based on any holdout.

---

# 39. Final seed set

Freeze exactly:

```text
FINAL_SEEDS-v1 = [42, 123, 2026]
```

No additional seed after seeing results.

---

# 40. Why three seeds

Purpose:

```text
quantify stochastic training sensitivity
```

before final Test reporting and attention stability analysis.

---

# 41. Seed roles

Seeds are symmetric.

No:

```text
primary seed
backup seed
best seed.
```

Phase46 runs all three.

---

# 42. No seed selection by Train loss

No.

---

# 43. No seed selection before Test

No final single “best seed” is chosen by training statistics.

All 3 checkpoints remain official.

---

# 44. Phase47 test evaluation across seeds

Phase45 must predeclare that Phase47 will evaluate all three final-seed checkpoints on the same untouched Test population.

Final performance should be summarized across seeds according to Phase47 plan.

No seed is dropped.

---

# 45. Final Test population remains locked

Phase45 records:

```text
TEST_LOCKED
```

but does not inspect Test targets.

Allowed metadata:

```text
Test boundary
candidate window eligibility schema
expected protocol
```

No values/metrics.

---

# 46. Final Test boundary context

Phase47 will use primary:

```text
WB0
```

so first Test targets may use preceding observed pre-Test history from Train+Validation.

This is consistent with the deployment assumption.

---

# 47. Final final-fit run semantics

For each Phase46 seed:

```text
fresh seed
fresh model
fresh optimizer
same final scalers
same final data
train exactly FINAL_REFIT_EPOCHS
no validation
no early stopping
save FINAL checkpoint at final epoch
```

---

# 48. Final checkpoint naming

Recommended:

```text
FINAL_SEED_42
FINAL_SEED_123
FINAL_SEED_2026
```

Checkpoint type:

```text
FINAL_REFIT
```

not:

```text
BEST.
```

There is no Validation selection during Phase46.

---

# 49. No BEST checkpoint semantics in Phase46

Do not save a checkpoint as “best” based on:

```text
Train loss
Test score
ad hoc probe.
```

Official checkpoint is:

```text
epoch = FINAL_REFIT_EPOCHS.
```

---

# 50. LAST vs FINAL

Recommended:

```text
FINAL = official final epoch checkpoint
LAST = same epoch/checkpoint
```

If implementation writes both:

```text
checksums should match or semantic equivalence documented.
```

---

# 51. Final training loss can be logged

Allowed:

```text
epoch Train criterion
gradient norms
runtime
```

but not used to select epochs/seeds.

---

# 52. Numerical failure policy in Phase46

If a seed has genuine numerical failure:

```text
Phase46 incomplete
```

Do not replace that seed with another seed unless formal Protocol Amendment.

---

# 53. No rescue hyperparameter changes

If seed123 fails:

```text
do not lower LR
do not enable clipping
do not reduce dropout
do not change epoch count.
```

That would break locked config.

---

# 54. Technical rerun policy

Allowed only for:

```text
hardware interruption
process kill
artifact-write failure
checkpoint corruption.
```

Must use same seed/config and documented rerun lineage.

---

# 55. No score-based rerun

No.

---

# 56. Final data population fingerprint

Phase45 must define expected canonical metadata for:

```text
FINAL_DEV_REGION-v1
FINAL_DEV_TARGET_IDS
```

The actual final population fingerprint should be produced deterministically before Phase46 first model run.

Phase45 can either compute metadata-only population fingerprint now or require Phase46 preflight to generate it.

Preferred:

```text
compute/freeze target-ID fingerprint now
without training.
```

---

# 57. Final preprocessing reproducibility

Freeze:

```text
feature order
scaling group rules
time-feature formulas
target-scaler semantics
RevIN scope/index mapping
lookback
window builder
boundary protocol.
```

---

# 58. Final candidate config fingerprint

Create canonical:

```text
final_model_scientific_config.json
```

and SHA256:

```text
FINAL_MODEL_CONFIG_SHA256.
```

---

# 59. Final training recipe fingerprint

Create separately:

```text
final_training_recipe.json
```

including:

```text
FINAL_DEV population
scaler-fit protocol
FINAL_REFIT_EPOCHS
seed list
optimizer
batch
loss
clipping
precision
DataLoader
checkpoint semantics.
```

Fingerprint:

```text
FINAL_TRAINING_RECIPE_SHA256.
```

---

# 60. Final lock fingerprint

Create a combined immutable hash over:

```text
scientific config fingerprint
training recipe fingerprint
lineage fingerprint
```

Canonical:

```text
FINAL_MODEL_LOCK_SHA256.
```

---

# 61. Why separate config vs recipe fingerprints

Scientific config answers:

```text
what model is it?
```

Training recipe answers:

```text
how is final model materialized?
```

Keeping them separate improves provenance.

---

# 62. Final lineage fingerprint

Build from:

```text
S1–S18 selected lineage
S19 boundary sensitivity
Phase42 shortlist
Phase43 LSTM context
Phase44 robustness recommendation.
```

LSTM context is evidence, not part of Transformer config.

---

# 63. Final lock immutability

After:

```text
phase_45_signoff.json
```

Phase46 must refuse execution if any locked artifact checksum changes.

---

# 64. Protocol Amendment rule

Any change after lock to:

```text
features
YS
L
architecture
loss
optimizer
LR
WD
dropout
epoch count
clipping
RevIN
WB0
seeds
final data region
scaler protocol
```

requires:

```text
Protocol Amendment
+
new lock version.
```

---

# 65. No silent hotfix

Even a seemingly harmless change such as:

```text
dropout 0.1 → 0.0
batch64 → batch32
```

invalidates lock.

---

# 66. Code bug exception

If Phase46 discovers a genuine implementation bug that means final lock does not correspond to intended protocol:

```text
STOP
fix bug
invalidate affected upstream evidence if necessary
document amendment
re-lock.
```

Do not patch and continue silently.

---

# 67. Final software/environment lock

Phase45 records approved environment from `ENV-v1`:

```text
Python
PyTorch
device policy
critical package versions
precision policy.
```

Phase46 must compare environment fingerprint.

---

# 68. Device policy

Use current project device-selection contract:

```text
CUDA → MPS → CPU
```

or exact approved Phase1 environment.

Final seeds should ideally all run on the same backend/device class.

If device differs:

```text
record warning
```

and avoid claiming pure seed-only variation.

---

# 69. Precision policy

Freeze:

```text
AMP=false
```

unless exact selected Training Engine config says otherwise.

No final mixed-precision optimization.

---

# 70. DataLoader contract

Freeze:

```text
Map-style lazy dataset
Train shuffle=true
drop_last=false
candidate batch B*
same worker policy
seed_worker policy
separate Generator
```

No final worker-count tuning.

---

# 71. Final training population and shuffle

All three seeds use:

```text
same target IDs
same dataset length
same feature order
```

but Train shuffle order differs deterministically by seed.

That is intended seed variation.

---

# 72. Final scaler and seeds

Scalers are fit once deterministically from full pre-Test data and reused for all seeds.

Do not refit separately in a way that could generate floating-point/environment variation unless exact deterministic equivalence is guaranteed.

Preferred:

```text
fit once
persist
checksum
reuse.
```

---

# 73. Final RevIN affine parameters

If RN1:

```text
gamma/beta are freshly initialized for each seed
```

and trained as model parameters.

Per-window instance stats remain dynamic.

---

# 74. Final positional encoding

Same deterministic fixed sinusoidal PE.

No learned final PE.

---

# 75. Final attention path

Standard training forward:

```text
attention weights disabled/not retained
```

for efficiency.

Inspection path remains available after checkpoint.

---

# 76. Final parameter count

Phase45 may resolve expected runtime parameter count from recommended candidate existing configuration.

Phase46 must verify exact count for each seed.

Expected:

```text
same parameter count all seeds.
```

---

# 77. State-dict schema

All seeds must have:

```text
same keys
same shapes
same dtypes
```

after training.

Values differ.

---

# 78. Final checkpoint metadata

Each Phase46 final checkpoint must embed/reference:

```text
FINAL_MODEL_LOCK_SHA256
candidate ID
seed
FINAL_REFIT_EPOCHS
feature fingerprint
scaler checksums
config fingerprint
training recipe fingerprint
population fingerprint
environment fingerprint
metric version
attention config.
```

---

# 79. No Test metric in checkpoint metadata before Phase47

Hard.

---

# 80. Final run family

Experiment Registry family:

```text
FINAL_SEED_RUN
```

Expected run count:

```text
3.
```

Exactly.

---

# 81. Final run IDs

Recommended:

```text
FINAL_TR_SEED42
FINAL_TR_SEED123
FINAL_TR_SEED2026
```

Actual Registry IDs may include timestamp/UUID, but logical aliases should be fixed.

---

# 82. Phase46 run matrix

Phase45 creates a planned matrix:

```text
seed
candidate ID
config fingerprint
recipe fingerprint
epochs
population
scaler bundle
status=PLANNED
```

No metrics.

---

# 83. Final training order

Recommended deterministic execution order:

```text
42
123
2026.
```

Run order must not affect configs.

---

# 84. No early analysis after seed42

Do not inspect seed42 and change anything before seed123/2026.

Phase46 should complete all three locked runs before model-quality interpretation.

---

# 85. No Test between seeds

Absolutely forbidden:

```text
train seed42
→ test seed42
→ decide whether to run other seeds.
```

Correct:

```text
train all 3
→ freeze all 3
→ Phase47 evaluates all 3.
```

---

# 86. Phase45 must carry rolling-origin evidence

Record recommended candidate:

```text
RO1 RMSE
RO2 RMSE
RO3 RMSE
pooled RMSE
pooled MAE
pooled R²
mean fold RMSE
SD
worst fold
fold wins.
```

These are development robustness evidence, not final Test results.

---

# 87. Carry model-family context

Record:

```text
LSTM_TUNED pooled metrics
Persistence pooled metrics
```

to preserve final reporting context.

Do not affect final Transformer config.

---

# 88. Carry Phase42 source factor context

If recommended candidate is a local alternative:

```text
record changed factor
changed_from
changed_to
source local regret
Phase42 candidate role.
```

This makes final selection explainable.

---

# 89. Final selection narrative

Phase45 should be able to answer:

```text
Why this Transformer?
```

with:

```text
It was the top frozen Transformer candidate under pre-Test three-fold rolling-origin pooled RMSE, using the predeclared tie rules, after one-factor candidate synthesis and boundary sensitivity checks.
```

No claim about Test yet.

---

# 90. Final epoch narrative

Should answer:

```text
Why this number of epochs?
```

with:

```text
It is the median of the recommended candidate’s three fold-specific inner best epochs from nested rolling-origin robustness, frozen before final seed training.
```

---

# 91. Final data narrative

Should answer:

```text
Why Train+Validation?
```

with:

```text
All model-selection decisions are locked, so all pre-Test development history can be used to fit the final model while keeping Test untouched.
```

---

# 92. No final hyperparameter selection from Phase46

Phase46 is materialization/reproducibility only.

It cannot change final lock.

---

# 93. No final hyperparameter selection from Phase47

Test evaluation cannot change final model.

---

# 94. No test-driven seed selection

All three seeds remain official, regardless of Test score differences.

---

# 95. Final test report policy preview

Phase45 should predeclare:

```text
Phase47 reports per-seed Test metrics
+
aggregate across seeds
```

but Phase45 must not specify observed values.

Exact aggregation mechanics belong to Phase47 plan.

---

# 96. Attention seed policy preview

Phase52–57 may use:

```text
all final-seed checkpoints
```

for attention stability.

No single seed chosen by Test score.

If a representative seed is needed for visualization later, selection must be based on a predeclared non-Test rule in attention phase.

---

# 97. Error-analysis compatibility

Final prediction bundles in Phase47 must preserve:

```text
target IDs
timestamps
raw Wh predictions
seed
```

for Phase48–51.

Phase45 records this downstream requirement.

---

# 98. Final lock artifact directory

```text
artifacts/
└── final_model_lock/
    ├── final_model_lock_manifest.json
    ├── final_model_lock_contract.json
    ├── phase45_preflight_audit.csv
    ├── final_candidate_source_audit.csv
    ├── final_lineage_audit.csv
    ├── final_model_scientific_config.json
    ├── final_feature_contract.json
    ├── final_preprocessing_contract.json
    ├── final_boundary_contract.json
    ├── final_revin_contract.json
    ├── final_optimizer_contract.json
    ├── final_loss_contract.json
    ├── final_epoch_policy.json
    ├── final_epoch_source_audit.csv
    ├── final_data_region_contract.json
    ├── final_scaling_contract.json
    ├── final_seed_contract.json
    ├── final_training_recipe.json
    ├── final_checkpoint_contract.json
    ├── final_attention_compatibility_audit.csv
    ├── final_environment_contract.json
    ├── final_three_seed_run_matrix.csv
    ├── final_model_config_fingerprint.json
    ├── final_training_recipe_fingerprint.json
    ├── final_lineage_fingerprint.json
    ├── final_model_lock_fingerprint.json
    ├── rolling_origin_selection_evidence.json
    ├── boundary_sensitivity_evidence.json
    ├── baseline_context_evidence.json
    ├── phase46_three_seed_handoff.json
    ├── phase47_test_evaluation_guard.json
    ├── final_model_lock_findings.csv
    ├── final_model_lock_tests.csv
    ├── final_model_lock_discrepancies.json
    ├── final_model_lock_summary.json
    ├── final_model_lock_report.md
    ├── README_FINAL_MODEL_LOCK.md
    └── phase_45_signoff.json
```

No checkpoint created here.

---

# 99. Required outputs

```text
O45.1  Lock manifest
O45.2  Lock contract
O45.3  Preflight audit
O45.4  Candidate-source audit
O45.5  Full lineage audit
O45.6  Final scientific config
O45.7  Feature contract
O45.8  Preprocessing contract
O45.9  Boundary contract
O45.10 RevIN contract
O45.11 Optimizer contract
O45.12 Loss contract
O45.13 Epoch policy
O45.14 Epoch-source audit
O45.15 Final data-region contract
O45.16 Final scaling contract
O45.17 Final seed contract
O45.18 Final training recipe
O45.19 Final checkpoint contract
O45.20 Attention compatibility audit
O45.21 Environment contract
O45.22 Planned three-seed run matrix
O45.23 Config fingerprint
O45.24 Recipe fingerprint
O45.25 Lineage fingerprint
O45.26 Combined lock fingerprint
O45.27 Rolling-origin selection evidence
O45.28 Boundary sensitivity evidence
O45.29 Baseline context
O45.30 Phase46 handoff
O45.31 Phase47 Test guard
O45.32 Findings
O45.33 Tests
O45.34 Discrepancies
O45.35 Summary
O45.36 Human-readable report
O45.37 README
O45.38 Phase sign-off
```

---

# 100. Lock manifest

`final_model_lock_manifest.json`:

```text
phase = 45
version = FINAL_MODEL_LOCK-v1
source_phase44_version
source_recommended_candidate_id
source_candidate_fingerprint
model_family = TRANSFORMER
new_training_runs = 0
new_validation_evaluations = 0
test_access = forbidden
final_data_region = TRAIN_PLUS_VALIDATION
final_epoch_policy = MEDIAN_RO_INNER_BEST_EPOCHS
final_seed_set = [42,123,2026]
boundary_protocol = WB0
lock_immutable_after_signoff = true
status
created_at
```

---

# 101. Final lock contract

`final_model_lock_contract.json` must state:

```text
One Phase44-recommended Transformer.
No candidate reranking.
No new tuning.
No Test.

Final fit:
original Train + Validation
fold-independent final scalers fit on all pre-Test development data
fixed final epoch count
no validation
no early stopping
three fixed seeds
fresh model/optimizer each seed
same config and scalers every seed.

Final epoch:
median of RO1/RO2/RO3 inner best epochs
for recommended Transformer.

All locked artifacts immutable after signoff.
Any change requires Protocol Amendment.
```

---

# 102. Candidate-source audit

`final_candidate_source_audit.csv`:

```text
field
phase44_recommended_value
phase42_candidate_value
resolved_final_value
match
status
```

Include all candidate scientific fields.

---

# 103. Full lineage audit

`final_lineage_audit.csv`:

```text
source_phase
decision
selected_value
source_artifact
source_run_id
carried_to_final
final_value
consistent
warning
status
```

Cover:

```text
S1–S18
S19
Phase42
Phase44.
```

Phase43 is baseline context, not Transformer decision.

---

# 104. Final scientific config schema

`final_model_scientific_config.json`:

```text
model_family
candidate_id
feature_set
time_features
feature_variant
feature_names_ordered
target_scaling
lookback
horizon
pooling
activation
batch_size
learning_rate
weight_decay
dropout
d_model
num_heads
head_dim
num_layers
ffn_dim
loss
Huber delta if applicable
max_epochs_development
patience_development
gradient_clipping
RevIN
boundary_protocol
positional_encoding
norm_style
mask_policy
output_head
attention_extraction_supported
config_version
```

Do not include observed metrics.

---

# 105. Final feature contract

`final_feature_contract.json`:

```text
feature_names
feature_order
feature_count_expected_from_runtime
historical_Appliances_included
rv1_rv2_included
time_feature_names
feature_fingerprint
target_name
availability_contract
```

---

# 106. Final preprocessing contract

`final_preprocessing_contract.json`:

```text
raw schema version
time-feature formulas
scaling groups
continuous features
passthrough cyclical features
passthrough binary features
target scaling
window construction
continuity policy
no padding
no interpolation
```

---

# 107. Final boundary contract

`final_boundary_contract.json`:

```text
protocol = WB0
target_assigned_by_target_timestamp
past cross-boundary context allowed
future input forbidden
actual observed history semantics
recursive prediction feedback forbidden
S19 sensitivity reference
protocol amendment required=false
```

---

# 108. Final RevIN contract

If RN0:

```text
enabled=false
```

If RN1 include all exact:

```text
scope
indices resolved by names
eps
affine
centering
variance convention
stats detach
target channel
time passthrough
coordinate bridge.
```

---

# 109. Final optimizer contract

`final_optimizer_contract.json`:

```text
optimizer=AdamW
LR
WD
betas
eps
parameter-group policy
scheduler=None
warmup=None
gradient accumulation=1
precision policy
```

---

# 110. Final loss contract

`final_loss_contract.json`:

```text
loss_id
loss_name
reduction
target_space = y_model
Huber_delta_if_applicable
evaluation_space = Wh
```

---

# 111. Final epoch policy artifact

`final_epoch_policy.json`:

```text
policy_id = MEDIAN_RO_INNER_BEST_EPOCHS-v1
source_phase = 44
source_candidate_id
RO1_inner_best_epoch
RO2_inner_best_epoch
RO3_inner_best_epoch
sorted_epochs
final_refit_epochs
candidate_max_epochs
within_cap
no_test_dependency = true
no_seed_dependency = true
status
```

---

# 112. Epoch-source audit

`final_epoch_source_audit.csv`:

```text
fold
candidate_id
inner_run_id
best_epoch
best_inner_val_rmse_wh
candidate_max_epochs
verified
status
```

plus summary row:

```text
MEDIAN
```

---

# 113. Final data-region contract

`final_data_region_contract.json`:

```text
region_id = FINAL_DEV_REGION-v1
included_splits = [TRAIN, VALIDATION]
excluded_splits = [TEST]
split_version
first_allowed_timestamp
last_allowed_training_target_timestamp
first_test_target_timestamp_metadata
target_population_rule
WB0
continuity
lookback
horizon
target_ids_fingerprint
test_target_values_accessed=false
```

---

# 114. Final scaling contract

`final_scaling_contract.json`:

```text
version = FINAL_SCALING-v1
fit_region = FINAL_DEV_REGION-v1
X_scaler_semantics = SCALING-v1
Y_scaler_semantics = selected YS*
fit_once = true
reuse_all_seeds = true
time_features_passthrough
binary_passthrough
RevIN fold-independent global-scaler bridge if active
Test rows used = false
expected checksum fields
```

---

# 115. Final seed contract

`final_seed_contract.json`:

```text
version = FINAL_SEEDS-v1
seeds = [42,123,2026]
run_order = [42,123,2026]
all_seeds_required = true
seed_replacement_forbidden = true
same_config_all_seeds = true
same_data_all_seeds = true
same_scalers_all_seeds = true
same_epochs_all_seeds = true
```

---

# 116. Final training recipe

`final_training_recipe.json`:

```text
model_config_fingerprint
final_data_region
final_population_fingerprint
final_scaling_contract
batch
optimizer
LR
WD
loss
gradient clipping
RevIN
epochs = FINAL_REFIT_EPOCHS
early_stopping = false
validation_loader = none
scheduler
warmup
accumulation
precision
shuffle
drop_last
worker policy
seed list
checkpoint type = FINAL_REFIT
attention retention during training = false
```

---

# 117. Final checkpoint contract

`final_checkpoint_contract.json`:

```text
checkpoint_type = FINAL_REFIT
official_epoch = FINAL_REFIT_EPOCHS
BEST_semantics = not applicable
LAST_semantics
required metadata
strict load required
config fingerprint required
recipe fingerprint required
lock fingerprint required
scaler checksums required
population fingerprint required
seed required
attention inspection compatibility required
```

---

# 118. Attention compatibility audit

`final_attention_compatibility_audit.csv`:

```text
check
expected
source
verified
status
```

Checks:

```text
attention-aware encoder retained
per-head weights available
shape contract valid
standard/inspection prediction equivalence contract exists
no mask change
final lookback supported
RevIN compatible if active
```

No attention extraction run required in Phase45.

---

# 119. Final environment contract

`final_environment_contract.json`:

```text
ENV-v1 fingerprint
Python version
PyTorch version
device policy
precision
determinism settings
worker policy
critical library versions
environment drift policy
```

---

# 120. Planned three-seed run matrix

`final_three_seed_run_matrix.csv`:

```text
logical_run_id
seed
candidate_id
config_fingerprint
training_recipe_fingerprint
lock_fingerprint
final_refit_epochs
data_region_id
population_fingerprint
x_scaler_bundle_id
y_scaler_bundle_id
checkpoint_type
status = PLANNED
```

Rows exactly 3.

---

# 121. Config fingerprint artifact

`final_model_config_fingerprint.json`:

```text
canonical_json_sha256
canonicalization_rules
source_config_file
```

---

# 122. Training recipe fingerprint

`final_training_recipe_fingerprint.json`:

```text
canonical_json_sha256
source_recipe_file
```

---

# 123. Final lineage fingerprint

`final_lineage_fingerprint.json`:

```text
selected_lineage_sha256
source_artifact_checksums
```

---

# 124. Final combined lock fingerprint

`final_model_lock_fingerprint.json`:

```text
final_model_config_sha256
final_training_recipe_sha256
final_lineage_sha256
combined_lock_sha256
algorithm=SHA256
```

---

# 125. Rolling-origin selection evidence

`rolling_origin_selection_evidence.json`:

```text
recommended_candidate
shortlist
fold_metrics
pooled_metrics
macro metrics
ranking rule
tie break
recommended rank
all folds complete
```

---

# 126. Boundary sensitivity evidence

`boundary_sensitivity_evidence.json`:

```text
WB0 primary
WB1 sensitivity
common delta
coverage delta
interpretation
protocol amendment false
```

---

# 127. Baseline context evidence

`baseline_context_evidence.json`:

```text
LSTM_TUNED config/run
LSTM rolling pooled metrics
Persistence rolling pooled metrics
recommended Transformer rolling pooled metrics
comparison notes
```

No final Test comparison.

---

# 128. Phase46 handoff

`phase46_three_seed_handoff.json` must contain everything required to run Phase46 without further scientific decision.

Fields:

```text
final_lock_version
final_lock_sha256
candidate_id
scientific_config
config_fingerprint
training_recipe
recipe_fingerprint
FINAL_REFIT_EPOCHS
FINAL_DEV_REGION-v1
target_ids_fingerprint
final scaling contract
seed list
planned run IDs
checkpoint contract
environment contract
test_locked=true
no_validation=true
no_early_stopping=true
ready_for_phase46=true
```

---

# 129. Phase47 Test guard

`phase47_test_evaluation_guard.json`:

```text
test_access_allowed_in_phase45=false
test_access_allowed_in_phase46=false
test_access_first_allowed_phase=47
all_three_seed_checkpoints_required_before_test=true
seed_specific_test_peeking_forbidden=true
test_population_must_be_identical_across_seeds=true
model_config_must_not_change_after_test=true
```

---

# 130. Final findings artifact

`final_model_lock_findings.csv` possible codes:

```text
FINAL_CANDIDATE_TR_C0
FINAL_CANDIDATE_TR_C1
FINAL_CANDIDATE_TR_C2
ROLLING_PRIMARY_CHANGED
ROLLING_PRIMARY_RETAINED
LSTM_ROLLING_GAIN_CONTEXT
TRANSFORMER_ROLLING_GAIN_CONTEXT
PERSISTENCE_STRONG_CONTEXT
WB0_PRIMARY_LOCKED
WB1_SENSITIVITY_CARRIED
RN0_LOCKED
RN1_LOCKED
FINAL_EPOCH_MEDIAN_POLICY_LOCKED
FINAL_DEV_TRAIN_PLUS_VAL_LOCKED
FINAL_SCALERS_REFIT_PRETEST
FINAL_SEEDS_LOCKED
ATTENTION_COMPATIBILITY_VERIFIED
CONFIG_FINGERPRINT_LOCKED
RECIPE_FINGERPRINT_LOCKED
TEST_FIREWALL_PRESERVED
INHERITED_WARNING
```

---

# 131. Discrepancy taxonomy

`final_model_lock_discrepancies.json`:

```text
PHASE44_NOT_APPROVED
RECOMMENDED_CANDIDATE_MISSING
PHASE44_RANKING_RULE_MISMATCH
RECOMMENDED_CANDIDATE_CONFIG_MISMATCH
SHORTLIST_FINGERPRINT_MISMATCH
LINEAGE_BREAK
S19_PROTOCOL_AMENDMENT_PENDING
BOUNDARY_PROTOCOL_DRIFT
FEATURE_CONFIG_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
ARCHITECTURE_DRIFT
LOSS_DRIFT
OPTIMIZER_DRIFT
DROPOUT_DRIFT
CLIPPING_DRIFT
REVIN_DRIFT
ATTENTION_SUPPORT_LOST
RO_INNER_EPOCH_MISSING
RO_INNER_EPOCH_OUT_OF_RANGE
FINAL_EPOCH_POLICY_MISMATCH
FINAL_DATA_REGION_INCLUDES_TEST
FINAL_SCALER_PROTOCOL_INVALID
FINAL_SEED_SET_MISMATCH
EARLY_STOPPING_LEFT_ENABLED
FINAL_VALIDATION_SELECTION_ENABLED
ENVIRONMENT_CONTRACT_MISSING
FINGERPRINT_MISMATCH
NEW_TUNING_ATTEMPT
NEW_TRAINING_ATTEMPT
NEW_VALIDATION_EVALUATION_ATTEMPT
TEST_LABEL_ACCESSED
TEST_PREDICTION_GENERATED
TEST_METRIC_COMPUTED
OTHER
```

---

# 132. Phase45 status model

## PASS

```text
Phase44 recommendation valid
lineage valid
final config frozen
final epoch derived/verified
final Train+Val recipe frozen
three seeds frozen
fingerprints generated
Phase46 handoff ready
Test untouched.
```

## PASS_WITH_WARNING

Possible:

```text
LSTM stronger in rolling-origin
Persistence strong
small Transformer robustness margin
fold variability high
S19 sensitivity warning
environment minor noncritical warning.
```

Warnings do not change locked config.

## FAIL

Examples:

```text
missing RO epoch
lineage mismatch
candidate reranking
Test access
unresolved protocol amendment
invalid final-fit recipe.
```

---

# 133. Preflight sequence

Before locking:

```text
1. Verify Phase44 signoff.
2. Verify approved_for_phase45.
3. Verify 3/3 folds complete for recommended candidate.
4. Verify Transformer robustness ranking.
5. Verify recommended candidate fingerprint.
6. Verify Phase42 shortlist fingerprint.
7. Verify S19 protocol amendment false.
8. Reconstruct S1–S18 lineage.
9. Verify final scientific config.
10. Extract RO1/RO2/RO3 inner best epochs.
11. Compute median final epoch.
12. Define FINAL_DEV_REGION-v1.
13. Verify Test exclusion.
14. Define FINAL_SCALING-v1.
15. Freeze seeds [42,123,2026].
16. Freeze checkpoint semantics.
17. Verify attention compatibility.
18. Verify environment contract.
19. Generate fingerprints.
20. Generate Phase46/Test guard handoffs.
```

---

# 134. No runtime outputs may be fabricated

Do not pre-fill:

```text
final scaler checksums
final population count
seed runtimes
seed train losses
Test metrics
Test ranking
attention results
```

until corresponding phases execute.

Allowed locked values:

```text
candidate config
epoch rule/result from existing Phase44 artifacts
seed list
data-region semantics
training recipe.
```

---

# 135. Final epoch source must be exact recommended candidate

Do not use inner epochs from:

```text
TR_C0
```

if Phase44 recommended:

```text
TR_C1.
```

Hard candidate-ID match.

---

# 136. No averaging epochs across different candidates

No.

---

# 137. No use of LSTM best epochs

No.

---

# 138. No use of outer fold RMSE to choose epoch

No.

---

# 139. No use of Phase21 B0 best epoch

No.

---

# 140. Why no early stopping in final seed runs

There is no unused pre-Test validation block left once Train+Validation are combined.

Using Test for early stopping is forbidden.

Thus fixed epochs are methodologically clean.

---

# 141. Why final full-data refit starts fresh

Do not load:

```text
Phase44 RO3 refit checkpoint
S18 winner checkpoint
Phase21 B0
```

and continue training.

Final seeds must represent fresh materializations of the locked recipe.

---

# 142. No transfer of optimizer moments

No.

---

# 143. No final warm-start

No.

---

# 144. No model averaging in Phase46

Three seeds remain separate checkpoints.

No ensemble unless explicitly introduced in a later pre-Test protocol, which current plan does not.

---

# 145. No seed ensemble for Test by default

Phase47 should evaluate individual seeds and aggregate metrics descriptively.

Do not average predictions into an ensemble unless Phase47 plan explicitly registers it before Test access.

Safer default:

```text
no ensemble.
```

---

# 146. Final model terminology

Use:

```text
Final configuration
Final-seed checkpoints
```

rather than prematurely saying:

```text
the one final trained model
```

because three seed-specific checkpoints will exist.

---

# 147. Representative checkpoint terminology

Do not designate a representative checkpoint in Phase45.

All seeds symmetric.

---

# 148. Attention analysis downstream

Phase52–57 may analyze:

```text
all three final checkpoints
```

and report seed stability.

This is why final attention-aware implementation must remain locked.

---

# 149. Error analysis downstream

Phase48–51 operates on final Test predictions generated only in Phase47.

Phase45 must not precompute any such result.

---

# 150. Phase45 human-readable report

`final_model_lock_report.md` sections:

```text
1. Objective
2. Inputs from Rolling-Origin Robustness
3. Recommended Transformer verification
4. Full selected lineage
5. S19 boundary sensitivity context
6. Final scientific configuration
7. Final feature/preprocessing contract
8. Final architecture
9. Final loss/optimizer/clipping/RevIN
10. Why Train+Validation are used for final refit
11. Final scaler-fit protocol
12. Final epoch policy
13. RO inner epoch evidence
14. Final seed policy
15. Final checkpoint semantics
16. Attention-analysis compatibility
17. Reproducibility/fingerprints
18. No-retuning/no-Test rules
19. Phase46 handoff
20. Phase47 Test guard
21. Limitations
22. Definition of Done
```

---

# 151. README requirements

`README_FINAL_MODEL_LOCK.md` explains:

```text
what is locked
what is not executed here
why recommended Transformer comes from Phase44
why final data = Train+Validation
why scalers refit on pre-Test data
why epoch = median rolling inner best epochs
why fixed epochs/no early stopping
why seeds = 42/123/2026
why all seeds must finish before Test
why config cannot change after lock
Phase46/47 handoff.
```

---

# 152. Summary artifact

`final_model_lock_summary.json`:

```text
version
recommended_candidate_id
config_fingerprint
training_recipe_fingerprint
lineage_fingerprint
final_lock_fingerprint
final_refit_epochs
epoch_source_values
final_data_region
final_seed_set
boundary_protocol
RevIN state
attention compatible
rolling_origin_metrics
baseline context
S19 sensitivity context
test_status
phase46_ready
overall_status
```

---

# 153. Planned Phase46 run-count assertion

Hard:

```text
planned_final_seed_runs = 3.
```

No extra exploratory final run.

---

# 154. Planned Phase46 training count

Exactly:

```text
3 scientific final refits.
```

Technical reruns do not count as new scientific configurations and must be lineage-linked.

---

# 155. Phase46 no-validation assertion

Hard:

```text
validation_loader = NONE
```

for official final training.

If Training Engine API requires a Validation object, Phase46 must use a dedicated final-refit mode rather than silently passing Test or reusing Validation.

---

# 156. Required final-refit engine mode

Preferred new execution mode:

```text
FINAL_REFIT_MODE-v1
```

semantics:

```text
fixed epochs
Train only
no validation
no early stopping
no BEST selection
save final epoch
same gradient safety/logging.
```

This is not a new scientific algorithm; it is a controlled execution mode.

---

# 157. Final-refit engine must be tested before Phase46

Phase45 can specify tests; Phase46 preflight executes them.

Tests:

```text
does not require val loader
does not call METRICS-v1 during training
does not access Test
runs exact N epochs
saves final checkpoint
preserves candidate optimizer/loss/clipping.
```

---

# 158. No Phase46 model comparison

Phase46 should not rank seeds by training curves.

---

# 159. No Phase46 convergence intervention

If Train loss still decreasing at final epoch:

```text
do not extend epochs.
```

Epoch count is locked.

---

# 160. No Phase46 manual stop

Except technical/numerical failure.

---

# 161. Final development data includes historical context before first Validation target

Since Train+Validation are one final development region, normal WB0 windows are built over the continuous pre-Test timeline.

No artificial split boundary is retained between Train and Validation for final training.

---

# 162. Final-dev window population

Unlike original Train/Val training distinction, final fit uses:

```text
all valid final candidate target IDs before Test
```

subject to:

```text
lookback
continuity
H1
WB0
feature availability.
```

---

# 163. `WINDOWPOP-v1` nuance

`WINDOWPOP-v1` was created for controlled L36/L72/L144 comparisons.

For final candidate refit:

Preferred conservative policy:

```text
retain WINDOWPOP-v1-compatible target universe
```

for provenance continuity unless current implementation defines an exact candidate-specific final population contract.

To avoid changing sample population at the last stage, lock:

```text
FINAL_DEV_TARGET_IDS
=
pre-Test IDs from WINDOWPOP-v1
compatible with final candidate.
```

---

# 164. Why retain common population in final fit

This avoids an unplanned final-stage sample expansion that was never used during selection.

It keeps final training population semantics aligned with development comparisons.

---

# 165. No candidate-specific extra training targets

Even if final L* could support additional targets beyond WINDOWPOP-v1:

```text
do not add them in Phase46
```

under `FINAL_MODEL_LOCK-v1`.

Any expansion would be a data-protocol change.

---

# 166. Final Test population policy preview

Phase47 should use the corresponding locked:

```text
WINDOWPOP-v1 Test IDs
```

under WB0 so final evaluation remains directly comparable to development protocol.

Exact Phase47 plan will audit this.

---

# 167. Final scaler fit rows vs target IDs distinction

X scaler may fit on all allowed pre-Test rows according to Phase9 semantics even though training target IDs use WINDOWPOP-v1.

Y scaler follows locked Phase9 target-fit semantics adapted to final pre-Test region.

Document both separately.

---

# 168. Final Y scaler semantics

If YS1:

```text
fit target StandardScaler using all Appliances values in the final pre-Test training target period according to the same design principle used in Phase9,
without Test.
```

Do not fit only minibatch target IDs unless Phase9 semantics require it.

---

# 169. RevIN + final scalers

If RN1, rebuild target X→Y coordinate bridge using:

```text
FINAL_SCALING-v1
```

checksums.

---

# 170. Final test leakage guard for scaler

Hard:

```text
no row whose timestamp belongs to Test
```

may contribute to scaler fit.

---

# 171. Environment drift warning

If Phase46 environment differs from Phase44 in a potentially meaningful way:

```text
ENVIRONMENT_DRIFT_WARNING
```

and decide whether current ENV-v1 still permits execution.

Do not silently compare as pure seed variation.

---

# 172. Reproducibility scope

Three seeds deliberately vary:

```text
initial weights
dropout RNG
DataLoader shuffle RNG
other seeded stochastic operations.
```

They do not vary:

```text
data
scalers
epochs
hyperparameters
architecture
environment intentionally.
```

---

# 173. Final parameter schema equality across seeds

Phase46 must assert:

```text
same state-dict key set
same tensor shapes
same parameter count.
```

---

# 174. Final data fingerprint equality across seeds

Hard.

---

# 175. Final scaler checksum equality across seeds

Hard.

---

# 176. Final recipe fingerprint equality across seeds

Hard except:

```text
seed field
```

which belongs to run identity, not scientific recipe if recipe seed list is global.

---

# 177. Run config fingerprint

Each seed gets:

```text
run_fingerprint
=
hash(final lock + seed).
```

---

# 178. No accidental reseeding every batch

Same existing RNG contract.

---

# 179. No seed-order dependence

Each seed starts from clean process/state or explicit full reseed.

---

# 180. Final lock audit before every Phase46 run

Each run must re-read:

```text
FINAL_MODEL_LOCK_SHA256
```

and compare.

If mismatch:

```text
STOP.
```

---

# 181. Final lock audit before Phase47

Phase47 must verify all three checkpoints reference the same lock hash.

---

# 182. Final lock audit before attention extraction

Phase52 similarly verifies checkpoints.

---

# 183. Acceptance checklist — Phase44 source

```text
[ ] Phase44 PASS/PASS_WITH_WARNING.
[ ] approved_for_phase45=true.
[ ] Recommended Transformer exists.
[ ] Recommended candidate has 3/3 folds.
[ ] Phase44 pooled metric verified.
[ ] Phase44 tie-break verified.
[ ] Candidate fingerprint matches Phase42.
[ ] Shortlist fingerprint matches Phase42 signoff.
[ ] No candidate reranking in Phase45.
```

---

# 184. Acceptance checklist — lineage

```text
[ ] S1 feature decision traced.
[ ] S2 time-feature decision traced.
[ ] S3 target-scaling decision traced.
[ ] S4 lookback traced.
[ ] S5 pooling traced.
[ ] S6 activation traced.
[ ] S7 batch traced.
[ ] S8 LR traced.
[ ] S9 WD traced.
[ ] S10 dropout traced.
[ ] S11 d_model traced.
[ ] S12 heads traced.
[ ] S13 layers traced.
[ ] S14 FFN traced.
[ ] S15 loss traced.
[ ] S16 epoch cap traced.
[ ] S17 clipping traced.
[ ] S18 RevIN traced/skip handled.
[ ] S19 WB0 primary verified.
[ ] Phase42 candidate synthesis traced.
[ ] Phase44 recommendation traced.
[ ] No unresolved lineage mismatch.
```

---

# 185. Acceptance checklist — final configuration

```text
[ ] Exact feature names/order frozen.
[ ] Historical target inclusion frozen.
[ ] Time features frozen.
[ ] YS frozen.
[ ] L frozen.
[ ] H1 frozen.
[ ] Pooling frozen.
[ ] Activation frozen.
[ ] B frozen.
[ ] LR frozen.
[ ] WD frozen.
[ ] Dropout frozen.
[ ] D frozen.
[ ] Heads/head_dim frozen.
[ ] Layers frozen.
[ ] FFN frozen.
[ ] Loss frozen.
[ ] Huber delta frozen if applicable.
[ ] Development max_epochs recorded.
[ ] Clipping frozen.
[ ] RevIN frozen.
[ ] WB0 frozen.
[ ] PE/norm/mask/head semantics frozen.
[ ] Attention extraction support retained.
```

---

# 186. Acceptance checklist — final epoch

```text
[ ] RO1 inner best epoch found.
[ ] RO2 inner best epoch found.
[ ] RO3 inner best epoch found.
[ ] All belong to recommended candidate.
[ ] All verified source run IDs.
[ ] All within candidate max_epochs.
[ ] Median computed deterministically.
[ ] FINAL_REFIT_EPOCHS integer.
[ ] Final epoch <= candidate max_epochs.
[ ] No Test dependency.
[ ] No seed dependency.
[ ] No manual override.
```

---

# 187. Acceptance checklist — final data/scaling

```text
[ ] FINAL_DEV_REGION = Train+Validation.
[ ] Test excluded.
[ ] Final target IDs from locked WINDOWPOP-v1 pre-Test universe.
[ ] No candidate-specific extra targets.
[ ] Final lookback/H1/WB0 applied.
[ ] Continuity rules preserved.
[ ] X scaler fit protocol defined.
[ ] Y scaler fit protocol defined.
[ ] Time features pass through.
[ ] Binary features pass through.
[ ] RevIN bridge rules defined if active.
[ ] Scalers fit once and reused across seeds.
[ ] Test rows forbidden in scaler fit.
```

---

# 188. Acceptance checklist — seeds/training

```text
[ ] Seeds exactly 42,123,2026.
[ ] Exactly 3 final scientific runs planned.
[ ] Same config all seeds.
[ ] Same data all seeds.
[ ] Same scalers all seeds.
[ ] Same fixed epochs all seeds.
[ ] Fresh model each seed.
[ ] Fresh optimizer each seed.
[ ] No warm-start.
[ ] No optimizer state reuse.
[ ] No validation.
[ ] No early stopping.
[ ] No seed-specific epoch changes.
[ ] No seed replacement.
[ ] No score-based rerun.
```

---

# 189. Acceptance checklist — checkpoints

```text
[ ] Checkpoint type FINAL_REFIT.
[ ] Official epoch fixed.
[ ] No BEST semantics.
[ ] Lock hash metadata required.
[ ] Config hash required.
[ ] Recipe hash required.
[ ] Population fingerprint required.
[ ] Scaler checksums required.
[ ] Seed required.
[ ] Attention compatibility required.
[ ] Same schema expected all seeds.
```

---

# 190. Acceptance checklist — Test firewall

```text
[ ] Test labels not read.
[ ] Test predictions not generated.
[ ] Test metrics not computed.
[ ] No Test used for epoch.
[ ] No Test used for scaler.
[ ] No Test used for seed choice.
[ ] No Test used for hyperparameter changes.
[ ] Phase47 guard artifact generated.
```

---

# 191. Acceptance checklist — fingerprints/handoff

```text
[ ] Config SHA256 generated.
[ ] Training recipe SHA256 generated.
[ ] Lineage SHA256 generated.
[ ] Combined lock SHA256 generated.
[ ] Phase46 handoff references lock SHA.
[ ] Phase46 planned run matrix has 3 rows.
[ ] Phase47 guard complete.
[ ] Lock immutable after signoff.
[ ] Protocol Amendment rule documented.
```

---

# 192. Acceptance criteria

Phase45 PASS only when:

```text
Phase44 has a valid recommended Transformer from complete rolling-origin evaluation.

The recommended candidate configuration matches its Phase42 frozen specification.

The full S1–S19 and Phase42–44 lineage is internally consistent.

WB0 remains the primary protocol with no pending amendment.

The exact final Transformer scientific configuration is frozen.

The final development region is Train+Validation only, excluding Test.

Final training target IDs preserve the locked WINDOWPOP-v1 population policy.

Final X/Y scaler fitting is restricted to pre-Test development data.

The recommended candidate’s three Phase44 inner best epochs are valid.

FINAL_REFIT_EPOCHS is the median of those three epochs.

The final seed set is exactly [42,123,2026].

Phase46 is specified as fresh fixed-epoch Train+Validation refit with no Validation and no early stopping.

All final seeds share the same data, scalers, epochs and scientific config.

Checkpoint semantics are FINAL_REFIT, not BEST.

Attention extraction compatibility is preserved.

All lock fingerprints are generated.

Phase46 and Phase47 guard handoffs are complete.

Test remains untouched.
```

---

# 193. Failure conditions

Phase45 FAIL if:

```text
Phase44 recommendation incomplete

recommended candidate is changed manually

lineage mismatch unresolved

WB1 promoted without amendment

Test included in final fit/scaling

final target population expanded ad hoc

final epoch uses outer Test/Validation metric post hoc

median epoch computed from wrong candidate

epoch manually overridden

seed set changed

validation/early stopping left enabled for final refit

RevIN/loss/optimizer/config changed

attention-aware implementation removed

new training is run in Phase45

new Validation evaluation is run in Phase45

Test labels/predictions/metrics are accessed

lock fingerprints cannot be reproduced.
```

---

# 194. Common mistakes

## 194.1 Phase44 xong rồi train final luôn mà chưa khóa epochs

Sai; dễ phát sinh ad hoc stopping.

## 194.2 Train+Validation nhưng vẫn early-stop trên Validation cũ

Validation đã nằm trong training population.

## 194.3 Dùng Test để early stop final model

Leakage nghiêm trọng.

## 194.4 Chọn số epoch bằng best epoch của original Validation

Có thể dùng development evidence nếu predeclared, nhưng Phase45 đã khóa rule tốt hơn: median three rolling inner epochs.

## 194.5 Dùng mean 3 best epochs rồi round

Không theo lock; phải median.

## 194.6 Mỗi seed dùng một số epochs khác nhau

Sai seed-comparison fairness.

## 194.7 Seed42 Test tốt nhất rồi gọi nó là final seed

Sai Test-driven selection.

## 194.8 Fit scaler riêng từng seed

Không cần; scaler phải deterministic và shared.

## 194.9 Dùng Train+Val+đầu Test để scaler “ổn định hơn”

Leakage.

## 194.10 Load RO3 checkpoint rồi train tiếp

Sai warm-start.

## 194.11 LSTM rolling tốt hơn nên đổi final model sang LSTM

Không đúng primary coursework model contract.

## 194.12 Candidate local alternative thắng Phase44 nhưng vẫn dùng TR_C0 vì quen

Sai Phase44 recommendation.

---

# 195. Recommended Phase45 execution flow

```text
Verify Phase44
→ resolve recommended Transformer
→ verify Phase42 fingerprint
→ reconstruct full lineage
→ carry S19 boundary sensitivity
→ freeze scientific config
→ extract recommended candidate RO1/2/3 inner best epochs
→ median → FINAL_REFIT_EPOCHS
→ define Train+Validation final region
→ freeze WINDOWPOP-v1 final target IDs
→ define final scaler-fit recipe
→ freeze seeds 42/123/2026
→ freeze fixed-epoch no-validation training mode
→ freeze checkpoint semantics
→ verify attention compatibility
→ hash config/recipe/lineage
→ combined lock hash
→ Phase46 handoff
→ Phase47 Test guard
→ sign off.
```

---

# 196. Recommended pseudocode

```text
p44 = load_phase44_signoff()
assert p44.approved_for_phase45
assert p44.test_status == "NOT_ACCESSED"

ro_rec = load_rolling_origin_recommended_transformer()
phase42 = load_frozen_transformer_shortlist()

assert exact_candidate_match(
    ro_rec.candidate_id,
    phase42
)

assert all_3_folds_complete(ro_rec)

lineage = reconstruct_full_transformer_lineage()
assert lineage.valid

s19 = load_boundary_sensitivity()
assert s19.primary_protocol == "WB0"
assert not s19.protocol_amendment_required

final_config = canonicalize(
    ro_rec.exact_scientific_config
)

inner_epochs = load_phase44_inner_best_epochs(
    candidate_id=ro_rec.candidate_id
)

assert len(inner_epochs) == 3
assert all_valid_and_verified(inner_epochs)

FINAL_REFIT_EPOCHS = median(
    [
        inner_epochs["RO1"],
        inner_epochs["RO2"],
        inner_epochs["RO3"]
    ]
)

assert 1 <= FINAL_REFIT_EPOCHS <= final_config.max_epochs

final_dev = define_final_dev_region(
    include=["TRAIN","VALIDATION"],
    exclude=["TEST"],
    population="WINDOWPOP-v1",
    candidate=final_config
)

assert no_test_target_or_row_used(final_dev)

final_scaling = define_final_scaling_contract(
    fit_region=final_dev,
    semantics="SCALING-v1",
    reuse_across_seeds=True
)

seed_contract = {
    "seeds": [42,123,2026],
    "all_required": True
}

training_recipe = {
    "config": final_config,
    "data": final_dev,
    "scaling": final_scaling,
    "epochs": FINAL_REFIT_EPOCHS,
    "early_stopping": False,
    "validation": None,
    "seeds": seed_contract,
    "checkpoint_type": "FINAL_REFIT"
}

verify_attention_compatibility(final_config)

config_hash = sha256(canonical(final_config))
recipe_hash = sha256(canonical(training_recipe))
lineage_hash = sha256(canonical(lineage))
lock_hash = sha256(
    config_hash + recipe_hash + lineage_hash
)

write_final_lock_artifacts(
    final_config,
    training_recipe,
    lineage,
    lock_hash
)

write_phase46_handoff(
    lock_hash=lock_hash,
    seeds=[42,123,2026],
    epochs=FINAL_REFIT_EPOCHS
)

write_phase47_test_guard(
    test_first_allowed_phase=47,
    require_all_seed_checkpoints=True
)

assert new_training_runs == 0
assert new_validation_evaluations == 0
assert test_not_accessed

signoff_phase45()
```

---

# 197. Phase46 handoff contract

Phase46 may execute only if:

```text
phase_45_signoff.overall_status
∈ {PASS, PASS_WITH_WARNING}

phase46_three_seed_handoff.ready_for_phase46 = true

all lock fingerprints match.
```

---

# 198. Phase46 expected behavior

For each:

```text
42
123
2026
```

Phase46:

```text
load same FINAL_MODEL_LOCK
fit/load same final scalers
fresh model
fresh optimizer
train FINAL_REFIT_EPOCHS
save FINAL_REFIT checkpoint
verify no Test.
```

---

# 199. Phase47 activation condition

Phase47 Test access must not begin until:

```text
3/3 final seed runs completed
3/3 checkpoints verified
same lock hash
same population
same scalers
same epochs
no scientific config drift.
```

---

# 200. Final model lock limitations

Mandatory report:

```text
1. Recommended candidate was selected using pre-Test development data.
2. Rolling-origin robustness uses only three folds.
3. Final epoch count derives from median of three inner selections.
4. Final seed count is three.
5. Train+Validation final refit removes an internal validation set by design.
6. Final unbiased generalization is unknown until Phase47.
```

---

# 201. Phase45 report language

Safe:

> The final Transformer configuration was fixed before Test evaluation using the candidate with the lowest pre-Test rolling-origin pooled RMSE under the predefined ranking rule.

Safe:

> The final refit duration was fixed as the median of the candidate’s three rolling-origin inner best epochs, allowing all pre-Test development data to be used for training without using Test for early stopping.

Safe:

> Three fixed random seeds will materialize the same locked scientific configuration; no seed-specific tuning is permitted.

---

# 202. Reporting prohibitions

Do not say:

```text
final model generalizes best
```

before Test.

Do not say:

```text
seed42 is the best model
```

before/after Test based on Test selection.

Do not say:

```text
Train+Validation guarantees better performance.
```

It simply uses all available pre-Test development data.

---

# 203. Definition of Done

\[
\boxed{
Phase44\ Recommended\ Transformer
+
Verified\ Full\ Lineage
+
Frozen\ Scientific\ Config
+
Train\!+\!Validation\ Final\ Region
+
PreTest\ Scalers
+
Median\ Rolling\ Epoch\ Lock
+
Seeds\{42,123,2026\}
+
Fixed\text{-}Epoch\ No\text{-}Validation\ Refit
+
Immutable\ Fingerprints
+
No\ Test
+
Phase46\ Handoff
}
\]

---

# 204. Final status contract

```text
PHASE 45 freezes the final Transformer.

Source:
Phase44 rolling-origin recommended Transformer.

No:
new training
new sweep
candidate reranking
new Validation evaluation
Test.

Final data:
original Train + Validation
Test excluded
locked WINDOWPOP-v1 target policy.

Final scalers:
fit on pre-Test development region
reuse same artifacts across all seeds.

Final epochs:
median(
  RO1 inner best epoch,
  RO2 inner best epoch,
  RO3 inner best epoch
)
for the recommended Transformer.

Final seeds:
42
123
2026.

Phase46:
fresh model + optimizer per seed
same data
same scalers
same config
same epochs
no validation
no early stopping
FINAL_REFIT checkpoint.

Phase47:
Test first becomes accessible only after all 3 final seed checkpoints exist and verify the same lock hash.

Lock:
scientific config SHA256
training recipe SHA256
lineage SHA256
combined FINAL_MODEL_LOCK_SHA256.

After FINAL_MODEL_LOCK-v1 PASS:
proceed to
PHASE 46 — Three-seed Final Runs.
```

---

# 205. Final check

Correct:

```text
Phase44 recommendation
→ verify lineage
→ freeze final config
→ median rolling inner epochs
→ Train+Validation final-data contract
→ pre-Test scaler contract
→ seeds 42/123/2026
→ immutable fingerprints
→ Phase46
```

Incorrect:

```text
Phase44 recommendation
→ try a few more configs
→ use Test to choose epoch
→ train seed42 first
→ inspect Test
→ change config
```

Chỉ sau khi:

```text
FINAL_MODEL_LOCK-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
ready_for_phase46 = true
```

mới chuyển sang **PHASE 46 — Three-Seed Final Runs**.
