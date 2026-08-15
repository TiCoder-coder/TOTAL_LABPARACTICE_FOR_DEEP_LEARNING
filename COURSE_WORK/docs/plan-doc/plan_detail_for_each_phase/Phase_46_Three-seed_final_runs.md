# PHASE 46 — THREE-SEED FINAL RUNS

## Kế hoạch materialize ba Final Transformer checkpoints từ `FINAL_MODEL_LOCK-v1` trước khi mở Test

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Upstream lock:** `FINAL_MODEL_LOCK-v1`  
**Phase ID:** `PHASE_46_THREE_SEED_FINAL_RUNS`  
**Output version:** `THREE_SEED_FINAL_RUNS-v1`  
**Phase trước:** `Phase_45_Final_model_lock.md`  
**Phase sau:** `Phase_47_Final_test_evaluation.md`

---

# 1. Vai trò của Phase 46

Phase 46 là phase **materialization và reproducibility**, không phải phase model selection.

Phase45 đã khóa:

```text
final scientific configuration
final data/preprocessing contract
final Train+Validation region
final scaler protocol
FINAL_REFIT_EPOCHS
final seed set
checkpoint semantics
environment/reproducibility contract
Test firewall
```

Phase46 chỉ thực thi đúng lock đó để tạo ba checkpoint độc lập:

```text
FINAL_TR_SEED42
FINAL_TR_SEED123
FINAL_TR_SEED2026
```

Mục tiêu chính:

```text
1. Verify FINAL_MODEL_LOCK-v1 trước khi chạy.
2. Materialize FINAL_DEV_REGION-v1.
3. Fit FINAL_SCALING-v1 đúng một lần trên pre-Test data.
4. Freeze scaler artifacts/checksums trước model run đầu tiên.
5. Train seed42 từ fresh initialization.
6. Train seed123 từ fresh initialization.
7. Train seed2026 từ fresh initialization.
8. Mỗi run dùng cùng scientific config, data, scalers và epoch count.
9. Không Validation, không Early Stopping, không BEST selection.
10. Save official FINAL_REFIT checkpoint ở đúng FINAL_REFIT_EPOCHS.
11. Verify checkpoint/config/data/scaler/lock consistency.
12. Quantify training-process variability mà không chọn “best seed”.
13. Chỉ khi 3/3 checkpoints hợp lệ mới mở gate cho Phase47.
14. Không truy cập Test.
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Locked\ Configuration
+
One\ Locked\ Data\ Population
+
One\ Locked\ Scaler\ Bundle
+
One\ Locked\ Epoch\ Count
+
Three\ Fixed\ Seeds
+
Fresh\ Independent\ Training
+
No\ Validation
+
No\ Test
}
\]

---

# 2. Phase 46 không được quyết định khoa học mới

Không được:

```text
đổi candidate
đổi feature
đổi scaler semantics
đổi lookback
đổi architecture
đổi loss
đổi LR
đổi WD
đổi dropout
đổi clipping
đổi RevIN
đổi epoch count
đổi seed list
đổi boundary protocol
thêm scheduler
thêm warmup
thêm augmentation
thêm ensemble
```

Nếu cần bất kỳ thay đổi nào:

```text
STOP
→ Protocol Amendment
→ regenerate Final Model Lock
→ chạy lại Phase46 từ đầu.
```

---

# 3. Inputs bắt buộc từ Phase45

Load chính xác:

```text
phase_45_signoff.json
phase46_three_seed_handoff.json
final_model_lock_manifest.json
final_model_lock_contract.json
final_model_scientific_config.json
final_feature_contract.json
final_preprocessing_contract.json
final_boundary_contract.json
final_revin_contract.json
final_optimizer_contract.json
final_loss_contract.json
final_epoch_policy.json
final_data_region_contract.json
final_scaling_contract.json
final_seed_contract.json
final_training_recipe.json
final_checkpoint_contract.json
final_environment_contract.json
final_three_seed_run_matrix.csv
final_model_config_fingerprint.json
final_training_recipe_fingerprint.json
final_lineage_fingerprint.json
final_model_lock_fingerprint.json
phase47_test_evaluation_guard.json
```

Hard:

```text
phase_45_signoff.overall_status
∈ {PASS, PASS_WITH_WARNING}

phase46_three_seed_handoff.ready_for_phase46 = true.
```

---

# 4. Lock verification là bước đầu tiên

Before any data fit/model construction:

```text
recompute FINAL_MODEL_CONFIG_SHA256
recompute FINAL_TRAINING_RECIPE_SHA256
recompute FINAL_LINEAGE_SHA256
recompute FINAL_MODEL_LOCK_SHA256
```

Compare against Phase45.

Hard:

```text
all hashes must match.
```

Mismatch:

```text
STOP
LOCK_FINGERPRINT_MISMATCH.
```

---

# 5. Source scientific configuration

Phase46 không tự reconstruct config từ notebooks.

Use:

```text
final_model_scientific_config.json
```

as authoritative scientific config.

This config must match the Phase44 recommended Transformer lineage already verified in Phase45.

---

# 6. Source training recipe

Use:

```text
final_training_recipe.json
```

as authoritative execution recipe.

Canonical values include:

```text
FINAL_DEV_REGION-v1
FINAL_REFIT_EPOCHS
batch*
AdamW
LR*
WD*
loss*
clipping*
RevIN*
no Validation
no Early Stopping
seed list
checkpoint type FINAL_REFIT
```

---

# 7. Seed set

Hard:

```text
FINAL_SEEDS-v1
=
[42, 123, 2026].
```

No additional seed.

No missing seed.

No replacement seed.

---

# 8. Run order

Preferred logical order:

```text
1. seed42
2. seed123
3. seed2026
```

Run order is operational only and must not influence configuration.

If infrastructure executes in another order:

```text
record actual run order
```

but keep same fixed seed set.

---

# 9. Exactly three scientific final runs

Expected:

```text
scientific_final_run_count = 3.
```

Technical rerun of a failed infrastructure execution may exist, but:

```text
must retain original failed record
must use same seed/config
must be linked by rerun_of
must not count as new scientific configuration.
```

---

# 10. Final data region

Phase46 materializes:

```text
FINAL_DEV_REGION-v1
=
original Train
+
original Validation
```

Test is excluded.

Hard:

```text
no Test target ID
no Test target value
no Test row in scaler fit
no Test prediction
```

---

# 11. Final target population

Use exactly the target-ID policy locked in Phase45:

```text
FINAL_DEV_TARGET_IDS
=
pre-Test IDs from locked WINDOWPOP-v1
compatible with final candidate.
```

Do not expand target population because final candidate lookback could support extra targets.

---

# 12. Why no final sample expansion

Changing target population after selection would create a new data-protocol condition.

Therefore Phase46 uses the population locked before Test access.

---

# 13. Target population fingerprint

Before training:

```text
materialize FINAL_DEV_TARGET_IDS
verify chronological order
verify uniqueness
compute SHA256 fingerprint
```

Compare against Phase45 expected/locked fingerprint.

If Phase45 stored only a semantic contract and not exact runtime fingerprint, Phase46 must:

```text
compute once
freeze before first seed
write artifact
reuse across all seeds.
```

No per-seed regeneration with inconsistent IDs.

---

# 14. Final full timeline

Construct one immutable pre-Test timeline view with:

```text
row ID
timestamp
split membership
continuity segment
raw feature values
derived deterministic time features
target availability metadata
```

No Test target values.

If implementation loads a full raw CSV containing Test rows in memory for schema/index purposes, hard guards must prevent Test rows from entering:

```text
scaler fit
training window IDs
target arrays
model evaluation.
```

Preferred:

```text
explicit pre-Test mask before target/scaler extraction.
```

---

# 15. Final feature contract

Use exact ordered final feature list.

Hard:

```text
same feature names
same feature order
same feature count
```

for all seeds.

No feature auto-discovery that could reorder columns.

---

# 16. Historical Appliances policy

If final feature config includes:

```text
historical Appliances
```

use only observed past target values inside input windows.

Never target `t+1`.

If absent:

```text
do not add it.
```

---

# 17. Time feature contract

Recompute using exact locked formulas:

```text
hour_sin
hour_cos
dow_sin
dow_cos
weekend
```

if TF1 active.

No new date features.

---

# 18. Window contract

For target position `j`, H=1:

```text
input rows = j-L* ... j-1
target row = j.
```

Hard:

```text
input [L*,F]
target [1]
```

No padding.

No interpolation.

Exact continuity rules preserved.

---

# 19. Boundary protocol in final dev training

Final dev region combines Train+Validation.

The old Train/Validation boundary is no longer a fitting boundary.

Within final pre-Test region:

```text
build normal continuous WB0-compatible windows
```

using locked target IDs.

No artificial isolation between former Train and former Validation.

---

# 20. Test boundary

Hard:

```text
last final training target timestamp
<
first Test target timestamp.
```

No training target lies in Test.

No input window for a training target extends beyond its target or into future Test.

---

# 21. Final scaling is a Phase46 pre-training artifact

Before seed42:

```text
fit FINAL_SCALING-v1
```

once.

Then persist and checksum it.

All 3 seeds use the same bundle.

---

# 22. X-scaler fit

Follow `final_scaling_contract.json`.

Use only allowed pre-Test rows.

Respect Phase9 scaling groups:

```text
continuous signals → fit/transform
cyclical calendar → pass-through
binary calendar → pass-through
metadata → not model input.
```

---

# 23. Y-scaler fit

If final:

```text
YS0
```

use identity artifact.

If:

```text
YS1
```

fit final target StandardScaler using the exact pre-Test target-fit semantics locked from Phase9/45.

No Test value.

---

# 24. Scaler fit once

Hard:

```text
fit scaler once
persist once
reuse same scaler artifacts for seed42/123/2026.
```

Do not refit independently per seed.

---

# 25. Final scaler artifact immutability

After scaler bundle is created:

```text
write checksums
set logical status = FROZEN
```

Subsequent mutation:

```text
STOP.
```

---

# 26. Scaler verification

Run before model training:

```text
finite means/scales
non-zero scale safeguards
expected feature-channel mapping
time/binary pass-through
YS inverse round-trip
feature order match
no Test rows used
```

---

# 27. RevIN final bridge verification

If RN1 active:

```text
resolve historical Appliances index by feature name
resolve final X-target scaler
resolve final Y transform
build FINAL_SCALING-v1 X→raw→Y bridge
```

Run synthetic/pre-Test round-trip tests.

No Test data required.

---

# 28. Final dataset implementation

Use the same:

```text
Map-style lazy Dataset
```

semantics from Phase11 where practical.

Each item:

```text
x [L,F] float32
y_model [1]
y_raw_wh [1]
sample_idx int64
```

Test-specific target lock is irrelevant because Phase46 dataset is pre-Test only.

---

# 29. Same dataset object/config across seeds

Preferred:

```text
same immutable dataset specification
```

with new DataLoader generators per seed.

No sample-content difference.

---

# 30. DataLoader contract

For final training:

```text
shuffle = true
drop_last = false
batch_size = B*
num_workers = locked environment value
worker seeding = existing contract
pin_memory = according to device policy
```

No Validation loader.

---

# 31. Why no Validation loader

All pre-Test data have been allocated to final fitting.

Official Phase46 mode must not need:

```text
Validation predictions
Validation metric
BEST checkpoint
early stopping.
```

---

# 32. Required execution mode

Use:

```text
FINAL_REFIT_MODE-v1
```

with semantics:

```text
fixed number of epochs
Train only
no Validation
no Early Stopping
no BEST checkpoint selection
save final epoch
same numerical safety
same loss aggregation
same optimizer behavior.
```

---

# 33. FINAL_REFIT_MODE preflight tests

Before first official run:

```text
no validation loader required
no metric evaluator called during training
no Test accessor invoked
exact epoch loop respected
final checkpoint saved
no hidden early-stop state
no BEST path selected
no scheduler unless locked
same gradient safety.
```

Disposable test model only.

Discard before seed42.

---

# 34. FINAL_REFIT_EPOCHS

Load exact locked integer:

```text
FINAL_REFIT_EPOCHS
```

from Phase45.

Do not recompute differently.

Can recompute median only as verification:

```text
median(RO1,RO2,RO3 inner best epochs)
==
locked FINAL_REFIT_EPOCHS.
```

---

# 35. Same epochs all seeds

Hard:

```text
epochs(seed42)
=
epochs(seed123)
=
epochs(seed2026)
=
FINAL_REFIT_EPOCHS.
```

---

# 36. No seed-specific stopping

No:

```text
stop if loss plateaus
stop if gradient unstable but finite
extend if loss still falls
```

unless genuine numerical failure requires abort.

---

# 37. Scientific config equality across seeds

All fields equal except:

```text
seed.
```

Hard audit.

---

# 38. Fresh model per seed

For each run:

```text
reset global RNG
construct fresh Transformer
verify config
move to device
```

No checkpoint loading from another seed.

---

# 39. Fresh optimizer per seed

Construct fresh:

```text
AdamW
```

with exact locked parameter-group policy.

No state reuse.

---

# 40. Fresh DataLoader RNG per seed

Train ordering is seed-dependent by design.

Each seed gets deterministic generator derived from that seed.

---

# 41. Seed controls

At run start set:

```text
Python RNG
NumPy RNG
PyTorch CPU RNG
CUDA RNG if applicable
DataLoader Generator
worker seeds.
```

Respect ENV-v1 determinism contract.

---

# 42. No reseeding per batch

Set seed at run setup, not every optimization step.

---

# 43. No hidden stochasticity change

Do not change:

```text
dropout sites
data order policy
backend
precision
worker policy
```

between seeds.

---

# 44. Device consistency

Preferred:

```text
same backend/device class all 3 runs.
```

If seed42 runs MPS and seed123 runs CPU:

```text
DEVICE_DRIFT_WARNING
```

because variation is no longer seed-only.

If exact reproducibility contract requires same device:

```text
STOP
```

until same environment available.

---

# 45. Environment fingerprint

Before each run:

```text
capture environment fingerprint
compare with final_environment_contract.
```

Hard/soft mismatches classified before training.

---

# 46. Model construction verification

For each seed before optimizer:

```text
parameter count
state-dict key schema
tensor shapes
feature input size
D/H divisibility
layers/FFN
pooling
activation
dropout
PE
mask policy
RevIN state
output [B,1].
```

---

# 47. Parameter count equality

Expected:

```text
param_count_42
=
param_count_123
=
param_count_2026.
```

Hard.

---

# 48. State-dict schema equality

Expected exact same:

```text
keys
shapes
dtypes
```

across seeds.

Values differ.

---

# 49. Initial-state fingerprint semantics

Different seeds should generally yield:

```text
different initial tensor values
```

while same schema/policy.

Do not require whole-state equality.

Record:

```text
initial_state_fingerprint
```

per seed.

---

# 50. Initial-state collision audit

If two different seeds produce exact same full initial-state fingerprint unexpectedly:

```text
investigate seed application.
```

Not automatically impossible, but highly suspicious.

---

# 51. Optimizer coverage

For every seed:

```text
all trainable params exactly once
no missing
no duplicates
same group topology
same LR/WD.
```

If RN1:

```text
RevIN affine params included according to locked policy.
```

---

# 52. Training step order

Use locked Training Engine semantics:

```text
model.train()
move batch
zero_grad(set_to_none=True)
forward
shape/finite assertions
criterion
backward
gradient finite guard
gradient clipping if GC1
optimizer.step
sample-weighted accumulation
```

No Validation phase after epoch.

---

# 53. Training loss aggregation

Use sample-weighted aggregation over the entire epoch.

Do not average batch losses naively if last batch size differs.

---

# 54. Loss coordinate

Criterion uses:

```text
y_model
```

exactly as locked.

Evaluation in Wh is not needed for model selection during Phase46.

---

# 55. Optional Train RMSE/MAE

Avoid expensive full-dataset prediction metrics every epoch unless already part of locked final recipe.

Preferred minimal logging:

```text
Train criterion
gradient diagnostics
runtime
```

No new scientific decision from training metrics.

---

# 56. No training metric used to choose checkpoint

Hard.

---

# 57. Numerical safety

Every batch:

```text
prediction finite
loss finite
gradient finite
```

Non-finite:

```text
abort run
record failure.
```

No silent skip.

---

# 58. Gradient clipping

If GC1:

```text
compute preclip global L2 norm
clip max_norm=1.0
optimizer.step.
```

If GC0:

```text
no gradient rescaling
but nonfinite guard remains.
```

No policy change by seed.

---

# 59. Gradient telemetry

Recommended:

```text
epoch mean preclip grad norm
epoch p50/p90/p95/max if available
clip fraction if GC1
counterfactual threshold exceedance if GC0
nonfinite count.
```

Diagnostic only.

---

# 60. Training epoch loop

For:

```text
epoch = 1 ... FINAL_REFIT_EPOCHS
```

every epoch must complete.

No patience counter.

No validation hook.

---

# 61. Checkpoint timing

Official final checkpoint saved:

```text
after optimizer updates for epoch FINAL_REFIT_EPOCHS complete.
```

No off-by-one.

---

# 62. Official checkpoint type

Hard metadata:

```text
checkpoint_type = FINAL_REFIT
```

Never:

```text
BEST.
```

---

# 63. Optional interim checkpoints

For crash recovery, interim epoch-boundary checkpoints may be stored if needed.

But they are:

```text
RECOVERY_ONLY
```

not scientific alternatives.

---

# 64. Recovery checkpoint policy

Resume only if exact:

```text
seed
lock hash
epoch
model state
optimizer state
RNG state
DataLoader generator state
scalers
population
```

match.

Preferred final-run correctness:

```text
restart seed from scratch
```

if recovery equivalence is uncertain.

---

# 65. Recovery cannot change final result selection

No choosing between:

```text
fresh run
resumed run
```

based on better training loss.

---

# 66. FINAL checkpoint metadata

Each checkpoint must include/reference:

```text
logical_run_id
Registry run_id
seed
checkpoint_type
official_epoch
FINAL_REFIT_EPOCHS
FINAL_MODEL_LOCK_SHA256
FINAL_MODEL_CONFIG_SHA256
FINAL_TRAINING_RECIPE_SHA256
FINAL_LINEAGE_SHA256
population_fingerprint
feature_fingerprint
X scaler checksum(s)
Y scaler checksum/identity
model config
optimizer config
loss config
clipping config
RevIN config
environment fingerprint
parameter count
state schema fingerprint
created_at
```

No Test metrics.

---

# 67. Final checkpoint strict-load verification

After saving:

```text
destroy/discard in-memory model
construct fresh exact model
strict-load checkpoint
```

Hard:

```text
strict=True
```

No missing/unexpected keys.

---

# 68. Post-load deterministic forward sanity

Use a fixed **pre-Test diagnostic probe**.

For loaded checkpoint:

```text
model.eval()
inference mode
prediction finite
shape [B,1]
```

No Test sample.

---

# 69. Standard vs attention-inspection path

For each final checkpoint on small pre-Test probe:

```text
standard prediction
≈
inspection-path prediction
```

within fixed tolerance.

This confirms downstream Phase52 compatibility.

Do not save attention maps yet except minimal shape/check status.

---

# 70. Attention shape verification

If inspection path invoked:

```text
per-layer attention shape [B,H*,L*,L*]
```

for small probe.

No interpretation.

No heatmap.

---

# 71. RevIN checkpoint verification

If RN1:

```text
affine keys present
scope metadata match
target-channel mapping match
FINAL_SCALING bridge checksums match.
```

---

# 72. Same pre-Test probe across seeds

Use identical fixed sample IDs for checkpoint sanity.

Probe serves only functional verification.

No seed ranking from probe errors.

---

# 73. Probe selection

Predeclare deterministic:

```text
first N valid FINAL_DEV_TARGET_IDS
```

or fixed hashed selection.

Do not choose “interesting” samples after training.

---

# 74. No pseudo-validation from probe

Probe predictions cannot be used to:

```text
select seed
select epoch
change config.
```

---

# 75. Training-process variability analysis

After all 3 runs complete, summarize:

```text
final Train criterion
min Train criterion
epoch-to-epoch trajectory
gradient norm summaries
clip fraction
runtime
parameter count
checkpoint size.
```

This is engineering/reproducibility evidence.

Not model-selection evidence.

---

# 76. No “best seed” from Train criterion

Forbidden.

---

# 77. No average weights

Do not average checkpoint parameters.

---

# 78. No ensemble predictions

No final/Test predictions in Phase46.

---

# 79. No checkpoint pruning

Keep all 3 official final checkpoints.

Do not delete the seed with highest Train loss.

---

# 80. Seed completion gate

Phase46 scientific completion requires:

```text
3/3 final runs COMPLETED
3/3 FINAL_REFIT checkpoints verified
3/3 same lock hash
3/3 same config fingerprint
3/3 same recipe fingerprint
3/3 same population fingerprint
3/3 same scaler checksums
3/3 same official epoch
```

---

# 81. Cross-seed lock consistency matrix

Create a table comparing every locked field across seeds.

Only allowed difference:

```text
seed
run_id
initial-state fingerprint
trained parameter values
training trajectory
runtime
checkpoint checksum.
```

---

# 82. Cross-seed data equality

Hard:

```text
same FINAL_DEV_TARGET_IDS fingerprint.
```

---

# 83. Cross-seed feature equality

Hard:

```text
same feature fingerprint.
```

---

# 84. Cross-seed scaler equality

Hard.

---

# 85. Cross-seed epoch equality

Hard.

---

# 86. Cross-seed config equality

Hard except seed.

---

# 87. Cross-seed optimizer equality

Hard except optimizer learned state trajectory.

Initial optimizer config same.

---

# 88. Cross-seed checkpoint schema equality

Hard.

---

# 89. Cross-seed environment equality

Preferred hard where environment contract requires; otherwise warning.

---

# 90. Phase47 gate

Phase47 may open Test only when Phase46 creates:

```text
phase47_test_release.json
```

with:

```text
released = true
```

and all scientific prerequisites pass.

---

# 91. Test release cannot depend on training score

Gate is based on:

```text
completion
consistency
provenance
```

not:

```text
Train loss quality.
```

---

# 92. Test release must include all checkpoint IDs

List:

```text
seed42 checkpoint
seed123 checkpoint
seed2026 checkpoint
```

and checksums.

---

# 93. Phase47 must evaluate all seeds

Phase46 handoff forbids:

```text
only evaluate seed with lowest Train loss.
```

---

# 94. Phase47 seed equality requirement

All three seed checkpoints correspond to one common scientific config.

If one seed differs:

```text
no Test release.
```

---

# 95. Test population remains untouched

Throughout Phase46:

```text
Test targets = NOT_ACCESSED
Test predictions = NOT_GENERATED
Test metrics = NOT_COMPUTED.
```

---

# 96. Test code guard

Recommended software-level guard:

```text
if current_phase < 47:
    forbid_test_target_materialization()
```

and explicit:

```text
FINAL_REFIT_MODE cannot receive TEST_LOCKED dataset.
```

---

# 97. Test loader should not be constructed

Preferred:

```text
no Test DataLoader object exists in Phase46 runtime.
```

This reduces accidental access risk.

---

# 98. No final-test “smoke test”

Do not use even one Test sample to check shape.

Use pre-Test probe.

---

# 99. No Test scaler transform check

No need.

Final scaler transformation on Test happens Phase47.

---

# 100. Experiment Registry family

All 3 official runs:

```text
family = FINAL_SEED_RUN
```

Run metadata:

```text
phase=46
final_lock_hash
seed
status.
```

---

# 101. Registry states

Expected:

```text
PLANNED
→ REGISTERED
→ RUNNING
→ COMPLETED
```

or:

```text
FAILED.
```

No deleting failed entries.

---

# 102. Run aliases

Logical aliases:

```text
FINAL_TR_SEED42
FINAL_TR_SEED123
FINAL_TR_SEED2026.
```

Registry run IDs remain unique.

---

# 103. Duplicate run guard

If Phase46 rerun is triggered accidentally for an already completed seed:

```text
STOP
```

unless explicitly a documented verification/technical rerun.

Do not pick the better duplicate.

---

# 104. No favorable duplicate selection

If two complete runs for same seed/config exist because of technical incident:

```text
use predetermined valid lineage rule
```

not lower Train loss.

Prefer first valid official run unless rerun was required because first was invalid.

---

# 105. Run-level output directory

Each official run under:

```text
artifacts/runs/<run_id>/
```

contains:

```text
config.json
lock_reference.json
training_history.csv
gradient_history.csv
runtime.json
FINAL_REFIT.pt
checkpoint_metadata.json
verification.json
status.json
```

Exact naming may follow existing Registry convention.

---

# 106. Phase-level output directory

```text
artifacts/
└── final_seed_runs/
    ├── three_seed_manifest.json
    ├── three_seed_contract.json
    ├── phase46_preflight_audit.csv
    ├── final_lock_verification.json
    ├── final_dev_population_manifest.json
    ├── final_dev_population_audit.csv
    ├── final_feature_order_audit.csv
    ├── final_scaler_fit_manifest.json
    ├── final_scaler_fit_audit.csv
    ├── final_scaler_checksums.json
    ├── final_scaler_roundtrip_tests.csv
    ├── final_revin_bridge_tests.csv
    ├── final_refit_engine_tests.csv
    ├── three_seed_run_matrix.csv
    ├── three_seed_config_consistency_audit.csv
    ├── three_seed_environment_audit.csv
    ├── three_seed_initialization_audit.csv
    ├── three_seed_optimizer_coverage_audit.csv
    ├── three_seed_sample_order_audit.csv
    ├── three_seed_epoch_completion_audit.csv
    ├── three_seed_training_history_summary.csv
    ├── three_seed_gradient_diagnostics.csv
    ├── three_seed_runtime_diagnostics.csv
    ├── three_seed_checkpoint_manifest.csv
    ├── three_seed_checkpoint_schema_audit.csv
    ├── three_seed_checkpoint_metadata_audit.csv
    ├── three_seed_checkpoint_reload_tests.csv
    ├── three_seed_forward_sanity_tests.csv
    ├── three_seed_attention_compatibility_tests.csv
    ├── three_seed_reproducibility_summary.csv
    ├── three_seed_findings.csv
    ├── phase47_test_release.json
    ├── phase47_final_test_handoff.json
    ├── three_seed_tests.csv
    ├── three_seed_discrepancies.json
    ├── three_seed_summary.json
    ├── three_seed_report.md
    ├── figures/
    │   ├── FINAL_46_01_training_criterion_by_seed.png
    │   ├── FINAL_46_02_gradient_norm_by_seed.png
    │   ├── FINAL_46_03_clipping_fraction_by_seed.png
    │   ├── FINAL_46_04_runtime_by_seed.png
    │   └── FINAL_46_05_final_training_criterion_summary.png
    ├── README_THREE_SEED_FINAL_RUNS.md
    └── phase_46_signoff.json
```

No Test predictions folder.

---

# 107. Required outputs

```text
O46.1  Three-seed manifest
O46.2  Three-seed contract
O46.3  Preflight audit
O46.4  Final lock verification
O46.5  Final-dev population manifest
O46.6  Final-dev population audit
O46.7  Feature-order audit
O46.8  Final scaler fit manifest
O46.9  Scaler fit audit
O46.10 Scaler checksums
O46.11 Scaler round-trip tests
O46.12 RevIN bridge tests if applicable
O46.13 FINAL_REFIT_MODE tests
O46.14 Official run matrix
O46.15 Cross-seed config consistency audit
O46.16 Environment audit
O46.17 Initialization audit
O46.18 Optimizer coverage audit
O46.19 Sample-order audit
O46.20 Epoch-completion audit
O46.21 Training-history summary
O46.22 Gradient diagnostics
O46.23 Runtime diagnostics
O46.24 Three official FINAL_REFIT checkpoints
O46.25 Checkpoint manifest
O46.26 Checkpoint schema audit
O46.27 Checkpoint metadata audit
O46.28 Checkpoint reload tests
O46.29 Forward sanity tests
O46.30 Attention compatibility tests
O46.31 Reproducibility summary
O46.32 Findings
O46.33 Phase47 Test release
O46.34 Phase47 handoff
O46.35 Tests
O46.36 Discrepancies
O46.37 Summary
O46.38 Human-readable report
O46.39 Figures
O46.40 README
O46.41 Sign-off
```

---

# 108. Three-seed manifest

`three_seed_manifest.json` minimum:

```text
phase = 46
version = THREE_SEED_FINAL_RUNS-v1
source_lock_version = FINAL_MODEL_LOCK-v1
final_lock_sha256
candidate_id
config_sha256
recipe_sha256
lineage_sha256
final_refit_epochs
final_dev_region = FINAL_DEV_REGION-v1
seed_contract = FINAL_SEEDS-v1
seeds = [42,123,2026]
planned_scientific_run_count = 3
validation_used = false
early_stopping_used = false
test_access = forbidden
status
created_at
```

---

# 109. Three-seed execution contract

`three_seed_contract.json` must state:

```text
Exactly three fixed seeds.
Same final model configuration.
Same final-dev target IDs.
Same final scaler bundle.
Same epoch count.
Same optimizer/loss/clipping/RevIN.
Fresh model/optimizer/loaders for every seed.
No Validation.
No Early Stopping.
No BEST checkpoint.
Official checkpoint = FINAL_REFIT at locked epoch.
No Test.
No seed selection.
No ensemble.
```

---

# 110. Preflight audit schema

`phase46_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Required checks:

```text
Phase45 approved
ready_for_phase46
lock hashes match
seed list exact
epoch count valid
Test guard active
environment valid
population contract valid
scaler contract valid
model config valid
FINAL_REFIT_MODE available
Registry available.
```

---

# 111. Final lock verification artifact

`final_lock_verification.json`:

```text
stored_config_sha256
recomputed_config_sha256
stored_recipe_sha256
recomputed_recipe_sha256
stored_lineage_sha256
recomputed_lineage_sha256
stored_lock_sha256
recomputed_lock_sha256
all_match
status
```

---

# 112. Final-dev population manifest

`final_dev_population_manifest.json`:

```text
region_id
included_splits
excluded_splits
lookback
horizon
boundary_protocol
population_policy
target_count_runtime
first_target_id
last_target_id
first_target_timestamp
last_target_timestamp
target_ids_sha256
Test_target_count_included = 0
status
```

No fabricated count before runtime.

---

# 113. Final-dev population audit

`final_dev_population_audit.csv`:

```text
check
expected
observed
status
```

Checks:

```text
unique IDs
chronological
all before Test
WINDOWPOP-v1 membership
continuity valid
lookback valid
feature availability valid
no Test target.
```

---

# 114. Final feature-order audit

`final_feature_order_audit.csv`:

```text
position
locked_feature_name
runtime_feature_name
same
feature_group
scaling_group
RevIN_scope_if_any
status
```

---

# 115. Final scaler fit manifest

`final_scaler_fit_manifest.json`:

```text
version = FINAL_SCALING-v1
fit_once = true
fit_region_id
X scaler bundle paths
Y scaler path/identity
feature mappings
fit_row_range
Test_rows_used=false
frozen=true
status
```

---

# 116. Scaler fit audit

`final_scaler_fit_audit.csv`:

```text
scaler_component
fit_start_timestamp
fit_end_timestamp
first_Test_timestamp_metadata
fit_end_before_Test
row_count_runtime
feature_names
Test_rows_used
status
```

---

# 117. Scaler checksums

`final_scaler_checksums.json`:

```text
X_bundle_sha256
per_component_sha256
Y_scaler_sha256_or_identity
feature_mapping_sha256
frozen_at
status
```

---

# 118. Scaler round-trip tests

`final_scaler_roundtrip_tests.csv`:

```text
test_case
component
max_abs_error
max_rel_error
finite
pass
status
```

Use pre-Test samples/synthetic values.

---

# 119. RevIN bridge tests

If RN1:

`final_revin_bridge_tests.csv`:

```text
test_case
target_channel
X_scaler_checksum
Y_scaler_checksum
roundtrip_error
future_target_used
pass
status
```

If RN0:

```text
artifact may contain one NOT_APPLICABLE row.
```

---

# 120. FINAL_REFIT_MODE tests

`final_refit_engine_tests.csv`:

```text
test_case
expected
observed
pass
status
```

Test cases:

```text
accepts Train-only loader
validation callback absent
early-stop state absent
exact fixed epochs
FINAL checkpoint semantics
no BEST selection
gradient safety active
no Test accessor
checkpoint metadata complete.
```

---

# 121. Official run matrix

`three_seed_run_matrix.csv`:

```text
logical_run_id
seed
registry_run_id
candidate_id
config_sha256
recipe_sha256
lock_sha256
population_sha256
X_scaler_sha256
Y_scaler_sha256_or_identity
final_refit_epochs
device
environment_fingerprint
checkpoint_type
status
```

Exactly three scientific rows.

---

# 122. Cross-seed config consistency

`three_seed_config_consistency_audit.csv`:

```text
field
seed42
seed123
seed2026
allowed_to_differ
all_equal_where_required
status
```

Allowed differences include only runtime identity/stochastic outputs.

---

# 123. Environment audit

`three_seed_environment_audit.csv`:

```text
seed
python_version
torch_version
device_type
device_name
precision
worker_policy
environment_fingerprint
matches_lock
warning
status
```

---

# 124. Initialization audit

`three_seed_initialization_audit.csv`:

```text
seed
model_schema_fingerprint
initial_state_fingerprint
parameter_count
fresh_initialization
loaded_checkpoint_before_train
expected_unique_vs_other_seeds
status
```

Expected:

```text
loaded_checkpoint_before_train=false.
```

---

# 125. Optimizer coverage audit

`three_seed_optimizer_coverage_audit.csv`:

```text
seed
trainable_parameter_count
optimizer_reference_count
unique_optimizer_parameter_count
missing
duplicate
group_count
group_fingerprint
LR
WD
status
```

---

# 126. Sample-order audit

`three_seed_sample_order_audit.csv`:

```text
seed
epoch_or_probe
sample_order_fingerprint
deterministic_for_seed
population_fingerprint
status
```

Across seeds order may differ.

Within exact technical rerun same seed should reproduce according to environment contract.

---

# 127. Epoch completion audit

`three_seed_epoch_completion_audit.csv`:

```text
seed
locked_final_refit_epochs
epochs_started
epochs_completed
early_stopping_triggered
validation_used
official_checkpoint_epoch
exact_match
status
```

Expected:

```text
early_stopping_triggered=false
validation_used=false
exact_match=true.
```

---

# 128. Training history summary

`three_seed_training_history_summary.csv`:

```text
seed
epoch
train_criterion
samples_seen
optimizer_steps
learning_rate
weight_decay
nonfinite_events
status
```

No Validation columns.

---

# 129. Gradient diagnostics

`three_seed_gradient_diagnostics.csv`:

```text
seed
epoch
mean_preclip_grad_norm
p50_preclip
p90_preclip
p95_preclip
max_preclip
actual_clip_fraction_if_GC1
counterfactual_exceedance_if_GC0
nonfinite_events
status
```

---

# 130. Runtime diagnostics

`three_seed_runtime_diagnostics.csv`:

```text
seed
device
epochs
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
samples_per_second_optional
peak_memory_optional
checkpoint_size_bytes
status
```

---

# 131. Checkpoint manifest

`three_seed_checkpoint_manifest.csv`:

```text
seed
logical_run_id
run_id
checkpoint_path
checkpoint_type
official_epoch
checkpoint_sha256
config_sha256
recipe_sha256
lock_sha256
population_sha256
X_scaler_sha256
Y_scaler_sha256_or_identity
state_schema_sha256
parameter_count
verified
status
```

---

# 132. Checkpoint schema audit

`three_seed_checkpoint_schema_audit.csv`:

```text
key
shape_seed42
shape_seed123
shape_seed2026
dtype_seed42
dtype_seed123
dtype_seed2026
same_shape
same_dtype
status
```

Aggregate summary:

```text
same key set
same shape schema
same dtype schema.
```

---

# 133. Checkpoint metadata audit

`three_seed_checkpoint_metadata_audit.csv`:

```text
field
seed42
seed123
seed2026
expected
allowed_difference
status
```

Verify:

```text
lock hash
epochs
candidate
data/scalers
checkpoint type
seed.
```

---

# 134. Checkpoint reload tests

`three_seed_checkpoint_reload_tests.csv`:

```text
seed
strict_load
missing_keys
unexpected_keys
parameter_count_match
config_match
scaler_refs_match
lock_hash_match
status
```

---

# 135. Forward sanity tests

`three_seed_forward_sanity_tests.csv`:

```text
seed
probe_fingerprint
batch_shape
prediction_shape
prediction_finite
eval_repeatability
status
```

Expected:

```text
prediction_shape=[B,1].
```

No Test.

---

# 136. Attention compatibility tests

`three_seed_attention_compatibility_tests.csv`:

```text
seed
probe_fingerprint
standard_prediction_shape
inspection_prediction_shape
predictions_allclose
layer_count
attention_shape_expected
attention_shape_observed
finite
status
```

No interpretation.

---

# 137. Reproducibility summary

`three_seed_reproducibility_summary.csv`:

```text
dimension
seed42
seed123
seed2026
required_equal
observed_equal
status
```

Dimensions:

```text
config
population
feature order
scalers
epochs
parameter count
state schema
environment
checkpoint metadata
attention compatibility.
```

---

# 138. Training criterion cross-seed summary

Can compute:

```text
final_train_criterion
mean final criterion
std final criterion
range
```

but label:

```text
TRAINING_PROCESS_DIAGNOSTIC_ONLY.
```

No model ranking.

---

# 139. Gradient cross-seed summary

Can summarize:

```text
clip fraction variation
grad norm variation
```

No seed selection.

---

# 140. Runtime cross-seed summary

Engineering only.

---

# 141. No performance claims before Test

Do not say:

```text
seed42 performs better
seed123 generalizes worse
```

from Train criterion.

Only:

```text
training trajectory differs.
```

---

# 142. Figures

Recommended:

```text
FINAL_46_01_training_criterion_by_seed.png
FINAL_46_02_gradient_norm_by_seed.png
FINAL_46_03_clipping_fraction_by_seed.png
FINAL_46_04_runtime_by_seed.png
FINAL_46_05_final_training_criterion_summary.png
```

No prediction/error plots.

No Test plots.

---

# 143. Training criterion figure

Overlay all seeds:

```text
epoch
vs
Train criterion.
```

Useful for reproducibility/stability.

---

# 144. Gradient norm figure

If telemetry exists.

No interpretation as generalization quality.

---

# 145. Clipping figure

If GC1:

```text
actual clip fraction by epoch/seed.
```

If GC0:

```text
counterfactual exceedance
```

clearly labeled.

---

# 146. Runtime figure

Secondary.

---

# 147. Findings codes

`three_seed_findings.csv` possible:

```text
FINAL_LOCK_VERIFIED
FINAL_DEV_POPULATION_VERIFIED
FINAL_SCALERS_FROZEN
FINAL_SCALERS_SHARED_ACROSS_SEEDS
SEED42_COMPLETED
SEED123_COMPLETED
SEED2026_COMPLETED
ALL_THREE_SEEDS_COMPLETED
PARAMETER_SCHEMA_EQUAL
CHECKPOINT_SCHEMA_EQUAL
ENVIRONMENT_EQUAL
ENVIRONMENT_DRIFT_WARNING
INITIALIZATION_DISTINCT
TRAINING_TRAJECTORY_VARIABILITY
GRADIENT_VARIABILITY
CLIPPING_VARIABILITY
ATTENTION_PATH_VERIFIED_ALL_SEEDS
RN1_BRIDGE_VERIFIED
FINAL_REFIT_EPOCHS_VERIFIED
NO_VALIDATION_USED
NO_EARLY_STOPPING_USED
TEST_FIREWALL_PRESERVED
TECHNICAL_RERUN_OCCURRED
INHERITED_WARNING
```

---

# 148. Discrepancy taxonomy

`three_seed_discrepancies.json`:

```text
PHASE45_NOT_APPROVED
PHASE46_HANDOFF_NOT_READY
LOCK_CONFIG_HASH_MISMATCH
LOCK_RECIPE_HASH_MISMATCH
LOCK_LINEAGE_HASH_MISMATCH
LOCK_COMBINED_HASH_MISMATCH
SEED_SET_MISMATCH
EXTRA_SEED_ATTEMPT
MISSING_SEED
FINAL_EPOCH_MISMATCH
FINAL_DEV_REGION_MISMATCH
TARGET_POPULATION_MISMATCH
TARGET_POPULATION_EXPANDED
TEST_TARGET_INCLUDED
FEATURE_ORDER_DRIFT
FEATURE_CONFIG_DRIFT
SCALER_FIT_TEST_LEAKAGE
SCALER_REFIT_PER_SEED
SCALER_CHECKSUM_MISMATCH
REVIN_BRIDGE_MISMATCH
MODEL_CONFIG_DRIFT
PARAMETER_COUNT_MISMATCH
STATE_SCHEMA_MISMATCH
OPTIMIZER_GROUP_DRIFT
LOSS_DRIFT
LR_DRIFT
WD_DRIFT
DROPOUT_DRIFT
CLIPPING_DRIFT
REVIN_DRIFT
BOUNDARY_PROTOCOL_DRIFT
VALIDATION_LOADER_USED
EARLY_STOPPING_USED
BEST_CHECKPOINT_USED
OFFICIAL_EPOCH_OFF_BY_ONE
WARM_START_USED
OPTIMIZER_STATE_REUSE
CHECKPOINT_FROM_OTHER_SEED_LOADED
RNG_SETUP_INVALID
SEED_RESEEDED_PER_BATCH
DEVICE_DRIFT
ENVIRONMENT_DRIFT
NONFINITE_PREDICTION
NONFINITE_LOSS
NONFINITE_GRADIENT
BATCH_SKIPPED
SCORE_BASED_RERUN
FAVORABLE_DUPLICATE_SELECTION
CHECKPOINT_STRICT_LOAD_FAILURE
ATTENTION_PATH_FAILURE
TEST_LOADER_CONSTRUCTED
TEST_LABEL_ACCESSED
TEST_PREDICTION_GENERATED
TEST_METRIC_COMPUTED
PHASE47_RELEASE_PREMATURE
OTHER
```

---

# 149. Status model

## PASS

```text
lock verified
final population verified
scalers fitted/frozen correctly
3/3 seeds completed
3/3 official checkpoints verified
cross-seed equality contracts pass
attention compatibility pass
Test untouched
Phase47 release generated.
```

## PASS_WITH_WARNING

Possible:

```text
environment minor drift allowed by contract
large training trajectory variation
high gradient/clipping variability
technical rerun required but exact lineage preserved
inherited noncritical warnings.
```

No warning may involve:

```text
scientific config drift
Test access
missing seed.
```

## FAIL

Examples:

```text
one seed missing
different epoch count
different scaler
different population
warm-start
Validation used
Test accessed
lock mismatch.
```

---

# 150. Run-level workflow — seed42

Exact:

```text
verify lock
verify frozen scalers
seed_all(42)
build fresh loaders
build fresh Transformer
verify schema/count
build fresh optimizer
register run
train FINAL_REFIT_EPOCHS
save FINAL_REFIT
strict reload
pre-Test forward sanity
attention-path sanity
mark COMPLETED
```

No interpretation yet.

---

# 151. Run-level workflow — seed123

Same, replacing only:

```text
seed=123.
```

No use of seed42 weights/state.

---

# 152. Run-level workflow — seed2026

Same, replacing only:

```text
seed=2026.
```

No use of prior seed weights/state.

---

# 153. No cross-seed checkpoint dependency

Hard:

```text
seed123 does not load seed42
seed2026 does not load seed42/123.
```

---

# 154. No cross-seed optimizer dependency

Hard.

---

# 155. No cross-seed DataLoader RNG dependency

Each generator initialized independently from current seed.

---

# 156. Cross-seed result interpretation only after all runs

Do not generate scientific conclusion after first or second run.

Phase-level summaries occur after:

```text
3/3 completed.
```

---

# 157. Training-run count audit

Report:

```text
planned scientific runs
completed scientific runs
failed scientific runs
technical reruns
```

No hidden runs.

---

# 158. Checkpoint byte-level equality is not expected

Different seeds should usually generate different weights/checksums.

Expected equality is in:

```text
schema/config/recipe/data/scalers/epochs.
```

---

# 159. Final training criterion equality is not expected

No.

---

# 160. Same optimizer step count expected

Given same:

```text
sample count
batch size
drop_last=false
epochs
```

expected:

```text
same optimizer steps across seeds.
```

Hard audit.

---

# 161. Steps per epoch formula

Runtime:

\[
steps\_per\_epoch
=
\left\lceil
\frac{N_{train}}{B}
\right\rceil
\]

since:

```text
drop_last=false.
```

Then:

\[
total\_steps
=
steps\_per\_epoch
\times
FINAL\_REFIT\_EPOCHS.
\]

Audit runtime count.

---

# 162. No gradient accumulation

Unless final recipe says otherwise; current expected:

```text
accumulation=1.
```

---

# 163. Learning rate constancy

If scheduler none:

```text
LR constant every optimizer step.
```

Audit first/last/unique LR values.

---

# 164. Weight decay constancy

Same optimizer config every seed.

---

# 165. Dropout semantics

Training:

```text
model.train()
dropout active.
```

Checkpoint sanity:

```text
model.eval()
dropout disabled.
```

---

# 166. Train/eval mode audit

During training epochs:

```text
model.training = true.
```

During reload/forward sanity:

```text
model.training = false.
```

---

# 167. No attention retention during training

Use standard forward path.

Do not store `[B,H,L,L]` attention tensors during full final training.

---

# 168. Memory safety

If attention-aware model has optional inspection:

```text
need_weights=False
```

during training.

This must not alter predictions compared with verified implementation semantics.

---

# 169. RevIN context safety

If RN1:

```text
each forward has its own instance context
no cross-batch cache
no future target stats.
```

Same Phase40 guarantees.

---

# 170. Dataset mutability guard

Training Dataset must not mutate:

```text
feature arrays
target arrays
scalers
sample indices
```

across epochs/seeds.

---

# 171. Data hashing

Recommended hashes:

```text
target IDs
feature names
scaler bundle
window index arrays
```

No need to hash every float tensor repeatedly if source/checksum lineage sufficient.

---

# 172. Input batch sanity per seed

First official batch:

```text
shape [B_or_partial,L,F]
finite
target [B_or_partial,1]
sample IDs in population.
```

Do not use this to alter config.

---

# 173. Final partial batch

`drop_last=false`.

Last batch may be smaller.

Sample-weighted loss aggregation must handle it.

---

# 174. No training sample duplication from DataLoader logic

Shuffle permutes, does not resample.

No weighted sampler.

---

# 175. No validation-style deterministic order requirement for Train

Shuffle required.

Reproducible under seed, not chronological.

---

# 176. No curriculum

No.

---

# 177. No target clipping/outlier removal

No.

---

# 178. No data augmentation

No.

---

# 179. No missing-value imputation added

No new preprocessing.

---

# 180. No AMP if locked false

Hard.

---

# 181. No `torch.compile` if not in lock

No final optimization change.

---

# 182. No EMA/SWA

No.

---

# 183. No gradient checkpointing unless locked

No.

---

# 184. No mixed-device execution within a run

Model/data/optimizer remain on one approved device.

---

# 185. Checkpoint serialization

Prefer atomic write:

```text
temp path
fsync/close
rename
compute SHA256.
```

No partially written official checkpoint.

---

# 186. Checkpoint corruption test

After checksum:

```text
reload from disk
strict-load
forward sanity.
```

---

# 187. Artifact atomicity

Run status becomes:

```text
COMPLETED
```

only after:

```text
checkpoint
metadata
history
verification
checksums
```

all written successfully.

---

# 188. Registry completion timing

Do not mark completed immediately after last optimizer step.

Completion follows verification.

---

# 189. Phase47 release logic

Pseudo:

```text
if all 3 runs COMPLETED
and all checkpoints verified
and same lock/config/recipe/population/scalers/epochs
and Test untouched:
    release=true
else:
    release=false
```

---

# 190. Phase47 release artifact

`phase47_test_release.json` minimum:

```text
release_version
released
reason
required_seed_count=3
completed_seed_count
checkpoint_ids
checkpoint_sha256s
all_same_lock_hash
all_same_config_hash
all_same_recipe_hash
all_same_population_hash
all_same_scaler_hashes
all_same_epochs
test_status_before_release=NOT_ACCESSED
released_at
status
```

---

# 191. Phase47 final-test handoff

`phase47_final_test_handoff.json`:

```text
phase46_version
final_lock_hash
final_candidate_id
final scientific config
FINAL_REFIT_EPOCHS
final scaler artifact paths/checksums
final feature fingerprint
final population policy
boundary protocol=WB0
seed_checkpoint_map
checkpoint metadata refs
attention compatibility status
Test first-access authorization=true
all_three_seeds_required=true
no_seed_selection=true
metric_version
prediction bundle schema requirement
ready_for_phase47=true
```

Only generate `ready_for_phase47=true` after release.

---

# 192. Phase47 must not retrain

Handoff should explicitly state:

```text
model training forbidden in Phase47.
```

Only checkpoint loading + Test inference/evaluation.

---

# 193. Phase47 must not refit scaler

Use exact final scalers from Phase46.

---

# 194. Phase47 must not choose a seed before evaluation

Evaluate all three.

---

# 195. Phase47 aggregate semantics deferred

Phase46 does not decide observed winner/aggregate performance.

Phase47 plan will lock metric aggregation/reporting before Test execution.

---

# 196. Human-readable report

`three_seed_report.md` sections:

```text
1. Objective
2. Final Model Lock verification
3. Final scientific configuration
4. Final development population
5. Final scaler fitting
6. Final fixed-epoch training protocol
7. Seed contract
8. FINAL_REFIT_MODE verification
9. Seed42 execution
10. Seed123 execution
11. Seed2026 execution
12. Cross-seed config/data/scaler consistency
13. Training-trajectory diagnostics
14. Gradient/clipping diagnostics
15. Runtime diagnostics
16. Checkpoint verification
17. Attention-path compatibility
18. Reproducibility findings
19. No-validation/no-early-stopping proof
20. Test firewall
21. Phase47 release decision
22. Limitations
23. Definition of Done
```

---

# 197. README requirements

`README_THREE_SEED_FINAL_RUNS.md` must explain:

```text
why 3 seeds
why same locked model
why Train+Validation
why scaler fit once
why no Validation
why no Early Stopping
why official checkpoint is FINAL_REFIT
why all seeds use same epoch count
why no best-seed selection
why no Test before all checkpoints complete
how Phase47 consumes artifacts.
```

---

# 198. Summary artifact

`three_seed_summary.json`:

```text
version
lock_hash
candidate_id
seed_list
final_refit_epochs
population_fingerprint
feature_fingerprint
scaler_checksums
runs
completed_count
failed_count
technical_rerun_count
parameter_schema_equal
optimizer_step_count_equal
environment_consistency
training_trajectory_summary
gradient_summary
checkpoint_verification_status
attention_compatibility_status
test_status
phase47_released
overall_status
```

---

# 199. Phase46 sign-off

`phase_46_signoff.json` minimum:

```text
phase=46
phase_name=Three-seed final runs
version=THREE_SEED_FINAL_RUNS-v1
source_final_lock_version
final_lock_sha256
candidate_id
config_sha256
recipe_sha256
population_sha256
feature_sha256
X_scaler_sha256
Y_scaler_sha256_or_identity
final_refit_epochs
seed_list=[42,123,2026]
scientific_run_count=3
completed_seed_count
seed42_run_id
seed123_run_id
seed2026_run_id
seed42_checkpoint_sha256
seed123_checkpoint_sha256
seed2026_checkpoint_sha256
all_same_config
all_same_recipe
all_same_population
all_same_scalers
all_same_epochs
all_same_parameter_schema
attention_compatibility_all_seeds
validation_used=false
early_stopping_used=false
test_status=NOT_ACCESSED
phase47_released
ready_for_phase47
warnings
overall_status
created_at
```

---

# 200. Acceptance checklist — lock

```text
[ ] Phase45 PASS/PASS_WITH_WARNING.
[ ] ready_for_phase46=true.
[ ] Config SHA recomputed/matches.
[ ] Recipe SHA recomputed/matches.
[ ] Lineage SHA recomputed/matches.
[ ] Combined lock SHA recomputed/matches.
[ ] No Protocol Amendment pending.
[ ] Scientific config loaded only from lock.
[ ] Training recipe loaded only from lock.
```

---

# 201. Acceptance checklist — data

```text
[ ] FINAL_DEV_REGION includes Train+Validation only.
[ ] Test excluded.
[ ] FINAL_DEV_TARGET_IDS materialized.
[ ] Target IDs unique.
[ ] Target IDs chronological.
[ ] Population policy matches Phase45.
[ ] No ad hoc sample expansion.
[ ] Feature names/order exact.
[ ] Lookback exact.
[ ] H1 exact.
[ ] Continuity exact.
[ ] WB0 exact.
[ ] No padding.
[ ] No future target in input.
```

---

# 202. Acceptance checklist — scalers

```text
[ ] FINAL_SCALING-v1 fitted before seed42.
[ ] X scaler uses pre-Test rows only.
[ ] Y scaler/identity exact.
[ ] Time features pass-through.
[ ] Binary features pass-through.
[ ] Scaler mappings exact.
[ ] Round-trip tests pass.
[ ] RN1 bridge passes if active.
[ ] Scaler bundle frozen.
[ ] Scaler checksums generated.
[ ] Same scaler bundle used all 3 seeds.
[ ] No per-seed refit.
```

---

# 203. Acceptance checklist — FINAL_REFIT_MODE

```text
[ ] No Validation loader.
[ ] No Early Stopping.
[ ] No BEST checkpoint selection.
[ ] Fixed epochs only.
[ ] Same loss mechanics.
[ ] Same optimizer mechanics.
[ ] Same gradient safety.
[ ] FINAL checkpoint saved at exact epoch.
[ ] No Test accessor.
[ ] Disposable engine tests discarded before official run.
```

---

# 204. Acceptance checklist — seeds

```text
[ ] Seed list exactly 42,123,2026.
[ ] No extra seed.
[ ] No replacement seed.
[ ] Run order recorded.
[ ] Fresh RNG setup each seed.
[ ] Fresh model each seed.
[ ] Fresh optimizer each seed.
[ ] Fresh DataLoader generator each seed.
[ ] No checkpoint warm-start.
[ ] No optimizer-state reuse.
[ ] No per-batch reseeding.
```

---

# 205. Acceptance checklist — training

```text
[ ] Same batch all seeds.
[ ] Same LR all seeds.
[ ] Same WD all seeds.
[ ] Same loss all seeds.
[ ] Same dropout all seeds.
[ ] Same clipping all seeds.
[ ] Same RevIN all seeds.
[ ] Same parameter-group policy.
[ ] Same FINAL_REFIT_EPOCHS all seeds.
[ ] drop_last=false.
[ ] sample-weighted epoch loss.
[ ] prediction finite.
[ ] loss finite.
[ ] gradient finite.
[ ] no skipped invalid batch.
[ ] no scheduler/warmup unless locked.
[ ] no AMP if locked false.
```

---

# 206. Acceptance checklist — official checkpoints

```text
[ ] Seed42 FINAL_REFIT exists.
[ ] Seed123 FINAL_REFIT exists.
[ ] Seed2026 FINAL_REFIT exists.
[ ] All official epoch exact.
[ ] checkpoint_type=FINAL_REFIT.
[ ] No BEST semantics.
[ ] All checkpoint SHA256 generated.
[ ] All strict-load successfully.
[ ] Same state schema.
[ ] Same parameter count.
[ ] Same config hash.
[ ] Same recipe hash.
[ ] Same lock hash.
[ ] Same population hash.
[ ] Same scaler checksums.
[ ] Seed metadata correct.
```

---

# 207. Acceptance checklist — sanity/attention

```text
[ ] Fixed pre-Test probe selected deterministically.
[ ] Probe identical all seeds.
[ ] Loaded prediction [B,1].
[ ] Predictions finite.
[ ] Eval repeatability verified.
[ ] Standard vs inspection prediction allclose.
[ ] Attention layer count correct.
[ ] Attention shape [B,H,L,L].
[ ] No Test probe.
[ ] No attention interpretation.
```

---

# 208. Acceptance checklist — cross-seed consistency

```text
[ ] Config equality except seed.
[ ] Population equality.
[ ] Feature equality.
[ ] Scaler equality.
[ ] Epoch equality.
[ ] Parameter schema equality.
[ ] Optimizer step-count equality.
[ ] Environment consistency audited.
[ ] Different stochastic trajectories allowed.
[ ] No best-seed designation.
[ ] All checkpoints retained.
```

---

# 209. Acceptance checklist — Test firewall

```text
[ ] Test loader not constructed.
[ ] Test target values not materialized.
[ ] Test predictions not generated.
[ ] Test metrics not computed.
[ ] No Test smoke sample.
[ ] No Test scaler fit.
[ ] No Test seed selection.
[ ] Phase47 release remains false until 3/3 verified.
[ ] Phase47 release true only after all gates pass.
```

---

# 210. Acceptance criteria

Phase46 PASS only when:

```text
FINAL_MODEL_LOCK-v1 is byte/semantically verified.

FINAL_DEV_REGION-v1 is materialized without Test.

The locked final target population is used exactly.

FINAL_SCALING-v1 is fitted once on pre-Test development data, frozen and shared across seeds.

FINAL_REFIT_MODE-v1 is verified to use fixed epochs with no Validation, no Early Stopping and no BEST checkpoint selection.

The exact seeds [42,123,2026] are run.

Every seed starts from a fresh model, optimizer and DataLoader RNG state.

Every seed uses the same scientific config, data population, scalers and FINAL_REFIT_EPOCHS.

All three runs complete all locked epochs without hidden scientific changes.

All three FINAL_REFIT checkpoints are persisted, checksummed, strict-reloaded and functionally verified.

All checkpoint schemas and locked metadata agree across seeds.

The attention inspection path remains compatible for all checkpoints.

No seed is selected or dropped based on training diagnostics.

Test labels, predictions and metrics remain untouched.

Phase47 Test release is granted only after 3/3 checkpoints satisfy every consistency gate.
```

---

# 211. Failure conditions

Phase46 FAIL if:

```text
final lock hash mismatches

scientific config changes

training recipe changes

target population changes

Test row enters scaler or training data

scaler differs between seeds

feature order differs

epoch count differs

seed set differs

one seed missing

warm-start is used

optimizer state reused

Validation is used

Early Stopping is used

BEST checkpoint selection is used

official checkpoint saved at wrong epoch

a non-finite batch is silently skipped

score-based rerun occurs

different device/environment violates lock

checkpoint strict-load fails

attention inspection path fails

Test loader is constructed/accessed

Test prediction or metric is generated

Phase47 is released before all three runs verify.
```

---

# 212. Common mistakes

## 212.1 Fit scaler lại cho mỗi seed

Sai. Scaler không phải stochastic component.

## 212.2 Train seed42 xong rồi xem Test

Sai Test firewall.

## 212.3 Seed42 train loss thấp nhất nên chỉ giữ seed42

Sai.

## 212.4 Seed123 loss vẫn giảm nên cho thêm 10 epochs

Sai fixed-epoch lock.

## 212.5 Seed2026 bị NaN nên giảm LR riêng

Sai scientific config.

## 212.6 Dùng old Validation để Early Stop

Sai final Train+Validation recipe.

## 212.7 Save “BEST” bằng minimum Train loss

Sai checkpoint semantics.

## 212.8 Load seed42 checkpoint rồi dùng làm init seed123

Sai independent seed run.

## 212.9 Test một sample để check shape

Không cần; dùng pre-Test probe.

## 212.10 Tạo ensemble ba seed ngay Phase46

Ngoài scope.

## 212.11 Xóa seed có trajectory xấu

Sai reproducibility reporting.

## 212.12 Dùng attention map để chọn seed

Sai; attention analysis diễn ra sau final Test evaluation.

---

# 213. Recommended execution notebook/script flow

```text
Step 46.1  Verify Phase45 signoff
Step 46.2  Recompute all Final Lock hashes
Step 46.3  Load scientific config/recipe
Step 46.4  Verify seed contract
Step 46.5  Materialize FINAL_DEV_REGION-v1
Step 46.6  Materialize/freeze FINAL_DEV_TARGET_IDS
Step 46.7  Audit feature order/window contract
Step 46.8  Fit FINAL_SCALING-v1 once
Step 46.9  Freeze scaler checksums
Step 46.10 Run scaler round-trip tests
Step 46.11 Run RevIN bridge tests if active
Step 46.12 Run FINAL_REFIT_MODE disposable tests
Step 46.13 Freeze official run matrix
Step 46.14 Execute seed42
Step 46.15 Verify seed42 checkpoint
Step 46.16 Execute seed123
Step 46.17 Verify seed123 checkpoint
Step 46.18 Execute seed2026
Step 46.19 Verify seed2026 checkpoint
Step 46.20 Build cross-seed config audit
Step 46.21 Build environment audit
Step 46.22 Build initialization/sample-order audits
Step 46.23 Build epoch/optimizer-step audit
Step 46.24 Build gradient/training/runtime diagnostics
Step 46.25 Build checkpoint schema/metadata audits
Step 46.26 Run common pre-Test forward sanity
Step 46.27 Run attention-path compatibility sanity
Step 46.28 Build reproducibility summary
Step 46.29 Run Test-firewall audit
Step 46.30 Decide Phase47 release
Step 46.31 Write Phase47 handoff
Step 46.32 Write findings/tests/discrepancies
Step 46.33 Write summary/report/README
Step 46.34 Sign off
```

---

# 214. Recommended execution pseudocode

```text
p45 = load_phase45_signoff()
assert p45.overall_status in {"PASS","PASS_WITH_WARNING"}

handoff = load_phase46_handoff()
assert handoff.ready_for_phase46

verify_all_final_lock_hashes()

cfg = load_final_scientific_config()
recipe = load_final_training_recipe()

assert recipe.seeds == [42,123,2026]
assert recipe.early_stopping is False
assert recipe.validation is None

final_dev = materialize_final_dev_region(
    splits=["TRAIN","VALIDATION"],
    exclude=["TEST"],
    population_policy="WINDOWPOP-v1",
    candidate=cfg
)

population_hash = freeze_target_ids(final_dev.target_ids)

assert no_test_targets(final_dev)
assert feature_order_matches_lock(final_dev)

final_scalers = fit_final_scalers_once(
    data=final_dev,
    contract="FINAL_SCALING-v1"
)

freeze_and_checksum(final_scalers)

run_scaler_tests(final_scalers)

if cfg.revin_enabled:
    run_final_revin_bridge_tests(
        final_scalers,
        cfg
    )

run_disposable_final_refit_engine_tests()
discard_disposable_objects()

run_records = []

for seed in [42,123,2026]:

    verify_lock_unchanged()
    verify_scalers_unchanged()

    seed_all(seed)

    train_loader = build_fresh_final_train_loader(
        final_dev,
        final_scalers,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        seed=seed
    )

    model = build_fresh_final_transformer(cfg)

    assert model_matches_locked_schema(model)

    optimizer = build_fresh_locked_adamw(
        model,
        cfg
    )

    assert optimizer_coverage_valid(model, optimizer)

    run_id = register_final_seed_run(
        seed=seed,
        lock_hash=FINAL_MODEL_LOCK_SHA256
    )

    result = FINAL_REFIT_MODE.fit(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        loss=cfg.loss,
        epochs=recipe.FINAL_REFIT_EPOCHS,
        validation=None,
        early_stopping=False,
        gradient_clip=cfg.gradient_clip
    )

    assert result.epochs_completed == recipe.FINAL_REFIT_EPOCHS

    ckpt = save_atomic_final_refit_checkpoint(
        model=model,
        optimizer=optimizer,
        seed=seed,
        epoch=recipe.FINAL_REFIT_EPOCHS,
        metadata=locked_metadata
    )

    verify_strict_reload(ckpt)

    probe = load_fixed_pretest_probe()

    verify_forward_sanity(
        checkpoint=ckpt,
        probe=probe
    )

    verify_attention_compatibility(
        checkpoint=ckpt,
        probe=probe
    )

    mark_run_completed(run_id)

    run_records.append(
        build_verified_run_record(...)
    )

assert len(run_records) == 3
assert seeds(run_records) == {42,123,2026}

verify_cross_seed_consistency(run_records)

assert test_loader_was_never_constructed()
assert test_targets_not_accessed()
assert test_predictions_not_generated()
assert test_metrics_not_computed()

release = build_phase47_release(
    released=all_scientific_gates_pass(run_records)
)

if release.released:
    write_phase47_handoff(
        checkpoints=run_records,
        scalers=final_scalers,
        lock_hash=FINAL_MODEL_LOCK_SHA256
    )

write_phase46_artifacts()
signoff_phase46()
```

---

# 215. Phase47 release hard gate

Release only when:

```text
3/3 seeds complete
+
3/3 checkpoints verified
+
same lock
+
same config
+
same recipe
+
same target population
+
same feature fingerprint
+
same scalers
+
same epochs
+
same parameter/state schema
+
attention compatibility PASS
+
Test status NOT_ACCESSED.
```

---

# 216. Phase47 release failure examples

Do not release if:

```text
seed2026 still running
seed123 checkpoint corrupt
seed42 used different scaler
one run had 1 fewer epoch
one run used CPU against disallowed environment contract
attention inspection breaks for one checkpoint
Test was accidentally touched.
```

---

# 217. Downstream Phase47 requirements

Phase47 receives immutable:

```text
3 checkpoint paths/checksums
FINAL_MODEL_LOCK_SHA256
scientific config
final scalers
feature order
Test population policy
WB0
metric version.
```

It must not retrain.

---

# 218. Downstream Phase48–51 requirements

Phase47 prediction bundles should preserve:

```text
seed
target ID
timestamp
y_true Wh
y_pred Wh
residual
```

for error analysis.

Phase46 does not generate them.

---

# 219. Downstream Phase52–57 requirements

All three final checkpoints must remain available.

Attention analysis later needs:

```text
same final model schema
same feature semantics
same lookback
same head/layer layout
seed-specific checkpoints.
```

---

# 220. Reproducibility limitation

Three seeds provide limited stochastic robustness evidence.

They do not fully characterize all possible random initialization outcomes.

Do not make distributional claims beyond:

```text
these three predeclared seeds.
```

---

# 221. Training-criterion limitation

Train loss across seeds is not a generalization metric.

Do not infer final Test ordering from it.

---

# 222. Environment limitation

Hardware/backend nondeterminism may cause small differences even under same seed if exact environment is not deterministic.

Record environment contract and warnings.

---

# 223. Fixed epoch limitation

Using one common median-derived epoch count prioritizes pre-Test protocol consistency over seed-specific convergence optimization.

This is intentional.

---

# 224. Full Train+Validation limitation

There is no internal holdout during final refit by design.

Generalization remains unknown until Test evaluation.

---

# 225. Safe reporting language

Safe:

> Three independently initialized Transformer models were trained using the same locked configuration, the same full pre-Test development population, the same preprocessing artifacts, and the same fixed number of epochs.

Safe:

> The three runs differ only in the predeclared random seed and the stochastic training trajectory induced by that seed.

Safe:

> No Validation or Test data were used for checkpoint selection during final refitting; the official checkpoint for each seed is the model state after the locked final epoch.

---

# 226. Reporting prohibitions

Do not write:

```text
seed42 is the best
seed123 generalizes worst
three-seed mean Test RMSE
```

in Phase46.

Do not write:

```text
final model performance
```

before Phase47.

Use:

```text
final training runs completed
final checkpoints verified.
```

---

# 227. Definition of Done

\[
\boxed{
Verified\ Final\ Lock
+
Frozen\ PreTest\ Population
+
One\ Shared\ Final\ Scaler\ Bundle
+
Seed42
+
Seed123
+
Seed2026
+
Fresh\ Independent\ Training
+
Exact\ Fixed\ Epochs
+
No\ Validation
+
Three\ Verified\ FINAL\_REFIT\ Checkpoints
+
Cross\text{-}Seed\ Consistency
+
No\ Test
+
Phase47\ Release
}
\]

---

# 228. Final status contract

```text
PHASE 46 materializes the locked final Transformer.

Inputs:
FINAL_MODEL_LOCK-v1
FINAL_DEV_REGION-v1
FINAL_SCALING-v1
FINAL_REFIT_EPOCHS
FINAL_SEEDS-v1.

Seeds:
42
123
2026.

Before seed42:
materialize final pre-Test population
fit final scalers once
freeze/checksum scalers
verify FINAL_REFIT_MODE.

Every seed:
verify lock unchanged
fresh RNG
fresh DataLoader
fresh model
fresh optimizer
same data
same scalers
same config
same fixed epochs
no Validation
no Early Stopping
no BEST checkpoint
save FINAL_REFIT
strict reload
pre-Test sanity
attention-path sanity.

Forbidden:
feature/config changes
per-seed scaler fit
per-seed epoch changes
warm-start
optimizer-state reuse
best-seed selection
ensemble
Test access.

Completion:
3/3 seed runs
3/3 checkpoints
same lock/config/recipe/population/scalers/epochs/schema.

Only then:
phase47_test_release.released=true

After THREE_SEED_FINAL_RUNS-v1 PASS:
proceed to
PHASE 47 — Final Test Evaluation.
```

---

# 229. Final check

Correct:

```text
verify Final Lock
→ materialize Train+Validation population
→ fit/freeze final scalers once
→ seed42 fresh fixed-epoch run
→ verify
→ seed123 fresh fixed-epoch run
→ verify
→ seed2026 fresh fixed-epoch run
→ verify
→ cross-seed consistency
→ release Test gate
→ Phase47
```

Incorrect:

```text
seed42
→ Test
→ adjust epochs/LR
→ seed123
→ choose best seed
```

Incorrect:

```text
fit a different scaler per seed
```

Incorrect:

```text
use original Validation for final Early Stopping
```

Chỉ sau khi:

```text
THREE_SEED_FINAL_RUNS-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase47_test_release.released = true
```

mới chuyển sang **PHASE 47 — Final Test Evaluation**.
