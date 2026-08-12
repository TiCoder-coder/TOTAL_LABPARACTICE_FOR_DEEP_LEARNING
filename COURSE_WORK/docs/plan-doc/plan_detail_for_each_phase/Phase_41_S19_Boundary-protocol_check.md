# PHASE 41 — S19 BOUNDARY PROTOCOL CHECK

## Kế hoạch sensitivity check cho Boundary Context Protocol của Transformer Regression

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S18_REVIN-v1`  
**Sweep ID:** `S19_BOUNDARY_PROTOCOL`  
**Output version:** `SWEEP_S19_BOUNDARY-v1`  
**Phase trước:** `Phase_40_S18_RevIN_sweep.md`

---

# 1. Vai trò của Phase 41

Phase 41 là controlled sensitivity check thứ mười chín và là bước cuối cùng của chuỗi S1–S19 trước khi đi sang Candidate Synthesis.

Mục tiêu duy nhất:

> Kiểm tra mức độ nhạy của mô hình cuối chuỗi sweep đối với cách xử lý historical context tại biên Train/Validation/Test split, bằng cách so sánh protocol chính `WB0 = context carry-over` với protocol nhạy cảm `WB1 = strict split isolation`.

Phase 41 chỉ thay đúng một registered protocol factor:

```text
window_boundary_protocol
```

với hai condition:

```text
WB0 = context carry-over
WB1 = strict split isolation
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Boundary\ Protocol
+
Same\ Forecasting\ Task
+
Same\ Selected\ Model
+
Same\ Split\ Boundaries
+
No\ Future\ Leakage
+
Validation\ Sensitivity\ Analysis
+
No\ Test\ Selection
}
\]

---

# 2. Đây là sensitivity check, không phải ordinary hyperparameter tuning

WB0 đã được khóa từ Phase 10 như primary forecasting protocol.

WB1 được pre-registered như strict-isolation sensitivity.

Mục đích S19:

```text
không phải tìm “window protocol tốt nhất” theo nghĩa hyperparameter thông thường,
mà kiểm tra kết luận mô hình có phụ thuộc mạnh vào cross-split historical context hay không.
```

Do đó report phải ưu tiên:

```text
robustness / sensitivity interpretation
```

thay vì chỉ viết:

```text
WB0 thắng WB1.
```

---

# 3. Vị trí Phase 41 trong master execution plan

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
Phase 39 → S17 Gradient-clipping
Phase 40 → S18 RevIN
Phase 41 → S19 Boundary protocol check
Phase 42 → Candidate synthesis
```

S19 là bước chuyển từ:

```text
sequential controlled sweeps
```

sang:

```text
candidate synthesis + robustness evaluation.
```

---

# 4. Hai boundary protocols

## 4.1 WB0 — Context carry-over

Primary protocol:

```text
forecast samples assigned by target timestamp
```

và Validation/Test windows được phép dùng immediately preceding observed history từ split trước nếu lịch sử đó nằm trước target timestamp.

Ví dụ:

```text
Train ends at t0
Validation starts at t0+10min

first Validation target:
y_{t0+10}

its input may include:
... historical Train rows up to t0
```

nếu window hoàn toàn precedes target.

WB0 phản ánh rolling one-step deployment:

```text
tại prediction time,
past observations trước target được xem là known history.
```

---

# 5. WB0 không phải leakage

WB0 chỉ hợp lệ nếu:

```text
all input timestamps < target timestamp
```

và không có:

```text
future target
future covariates beyond observation time
Validation/Test label in input
post-target row
```

Cross-split history:

```text
past Train row used for Validation context
```

không phải future leakage.

Nó chỉ là protocol choice về historical context availability.

---

# 6. WB1 — Strict split isolation

WB1 cấm một forecast sample sử dụng input history từ split khác.

For Validation:

```text
all L* input rows
and
target row
must belong to Validation split.
```

For Test:

```text
all L* input rows
and
target row
must belong to Test split.
```

For Train:

```text
all inputs + target remain Train.
```

WB1 do đó làm mất các target samples ở đầu Validation/Test vì chưa có đủ L* local historical rows.

---

# 7. Exact WB1 eligibility

Với target row position `j`, horizon `H=1`, selected lookback `L*`:

```text
input rows:
j-L* ... j-1

target:
j
```

WB1 sample valid khi:

```text
all input rows exist
exact cadence/continuity valid
same continuity segment
all input split labels == target split label
```

Hard:

```text
no cross-split input row.
```

---

# 8. WB0 eligibility

WB0 sample valid khi:

```text
target belongs to requested split
all input rows precede target
continuity valid
no gaps/duplicates/off-grid
```

Input split membership may differ from target split for boundary context.

No future leakage.

---

# 9. Key methodological consequence: sample populations differ

WB0 và WB1 không có identical Validation target population ở boundary.

WB1 normally removes approximately up to:

```text
L*
```

initial target opportunities from Validation/Test depending on continuity.

Therefore direct RMSE difference can combine two effects:

```text
A. boundary context protocol
B. target population difference.
```

S19 phải tách hai câu hỏi này.

---

# 10. Two-level comparison design

S19 phải tạo hai comparison views:

```text
VIEW A — Native Protocol Population
WB0_native vs WB1_native

VIEW B — Common Target Population
WB0_common vs WB1_common
```

Trong đó:

```text
common target population
=
intersection of valid target IDs under WB0 and WB1
```

cho cùng split.

---

# 11. Native population comparison

Purpose:

```text
what performance would each protocol produce under its own natural eligible target set?
```

This is operational/protocol-level comparison.

Nhưng nó không hoàn toàn isolate context effect do target sets differ.

Report phải nói rõ.

---

# 12. Common population comparison

Purpose:

```text
compare WB0 and WB1 on exactly the same target timestamps.
```

Đây là fairness view quan trọng nhất để tách:

```text
protocol/model-training effect
```

khỏi:

```text
different evaluated target composition.
```

Hard:

```text
same target IDs
same ordering
same raw y_true
same metrics implementation.
```

---

# 13. Important nuance: on common targets, input windows may become identical

Với WB1-common targets đủ xa split boundary:

```text
their last L* rows are fully inside split.
```

For those same targets, WB0 naturally uses the same most recent L* rows.

Do đó:

```text
WB0 input window
may equal
WB1 input window
```

for all common target IDs.

Nếu vậy, sensitivity primarily arises from:

```text
different training procedure/sample population
and
availability of early-boundary targets in native evaluation
```

rather than different input windows on common target IDs.

S19 phải audit điều này thay vì giả định.

---

# 14. Window fingerprint equality audit

For common target IDs:

```text
fingerprint WB0 input row IDs/timestamps
fingerprint WB1 input row IDs/timestamps
```

Expected under standard fixed-L formulation:

```text
often equal.
```

If not equal:

```text
document exact reason
```

and verify no bug.

---

# 15. Training population question

WB0/WB1 effect on Train is normally minimal or none because Train is the first split and cannot borrow history from a prior split in the same dataset partition.

However initial Train targets may still be unavailable until enough historical data exist.

Audit:

```text
Train target IDs WB0
vs
Train target IDs WB1.
```

Expected:

```text
equal
```

under current split definition.

If not:

```text
investigate window-builder implementation.
```

---

# 16. Validation population effect

Expected:

```text
WB1 Validation native sample count
<
or equal
WB0 Validation native sample count.
```

Typically:

```text
WB1 loses first L* valid targets after Validation boundary.
```

Actual runtime count is source of truth due gap/continuity constraints.

---

# 17. Test population effect

Phase41 must define Test populations but **must not evaluate Test labels**.

Allowed:

```text
construct/index Test eligibility metadata
count candidate target IDs
compare WB0/WB1 Test target-ID sets
```

Not allowed:

```text
read Test y values
compute Test predictions
compute Test metrics
inspect Test residuals.
```

This preserves Test firewall.

---

# 18. Test metadata access boundary

Allowed Test metadata:

```text
timestamps
split membership
row IDs
continuity segment IDs
window eligibility
target ID counts
```

Forbidden until Phase47:

```text
target values
predictions
metrics.
```

---

# 19. Current reference từ Phase40

Phase41 phải load:

```text
s18_revin_winner.json
s18_reference_update.json
phase_40_signoff.json
```

or valid skip-equivalent S18 outcome.

Resolve:

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
LOSS*
EPOCHS*
GC*
RN*
```

If S18 was:

```text
SKIPPED_NOT_APPLICABLE
```

then:

```text
RN* = RN0
```

carried forward.

---

# 20. Frozen settings

S19 must freeze all S1–S18 selections.

Only:

```text
WB0 vs WB1
```

may differ.

No retuning after seeing WB1.

---

# 21. Primary S19 question

Main sensitivity question:

> Does strict split isolation materially change Validation conclusions relative to the primary context-carry-over protocol?

This should be answered with:

```text
common-population metric difference
native-population metric difference
sample-count loss
boundary sample analysis
training convergence context
```

not just one RMSE.

---

# 22. Secondary research questions

S19 should answer:

```text
1. How many Validation targets are lost under WB1?
2. How many Test target opportunities would be lost under WB1?
3. Are common-target input windows identical?
4. Is Train population identical?
5. Does retraining under WB1 alter Validation RMSE on common targets?
6. Does WB0’s additional early-boundary Validation region have different error difficulty?
7. Is the final selected configuration robust to stricter isolation?
8. Should candidate synthesis keep WB0 as primary and WB1 as sensitivity evidence?
```

---

# 23. Why WB1 requires fresh scientific run

Even if Train sample population happens to be identical, Phase41 should treat WB1 as a new protocol condition.

Preferred:

```text
WB0 → reuse S18 selected run
WB1 → one fresh seed42 run
```

because DataLoader/window protocol lineage differs and the project should preserve explicit scientific provenance.

If runtime audit proves Train datasets are bitwise identical, RN0/RN1 etc. unchanged, retraining may mathematically reproduce the same training path, but official S19 still records a distinct WB1 run for protocol traceability.

---

# 24. No warm-start

WB1 run:

```text
fresh seed42
fresh loaders
fresh model
fresh criterion
fresh optimizer
```

No load WB0 BEST.

---

# 25. No optimizer-state reuse

No.

---

# 26. Same selected RevIN

If:

```text
RN*=RN0
```

WB0/WB1 both RevIN OFF.

If:

```text
RN*=RN1
```

WB0/WB1 both use exact S18 RevIN config.

---

# 27. RevIN with WB1

If RN1 selected, per-instance statistics must use:

```text
only each WB1 valid local input window.
```

No pre-Validation Train rows may be used for:

```text
mean
std
target-channel stats
```

in WB1 Validation.

---

# 28. Same global scalers

WB1 does not refit:

```text
X scaler
Y scaler.
```

Use same train-only frozen scaling artifacts from Phase9.

This preserves exact preprocessing comparability.

---

# 29. Same split timestamps

Hard:

```text
Train/Validation/Test cut points
```

unchanged.

WB1 changes eligibility, not split boundaries.

---

# 30. Same chronological ordering

No shuffle in Validation.

No altered target order.

---

# 31. Same forecasting horizon

Hard:

```text
H=1.
```

No shift.

---

# 32. Same lookback length

Hard:

```text
L=L*.
```

WB1 cannot shorten windows at split start.

---

# 33. No padding

Do not create WB1 early targets using:

```text
zero padding
repeat padding
mask padding
short windows.
```

If insufficient local split history:

```text
target not eligible.
```

---

# 34. No synthetic context

No interpolation across boundary solely to make WB1 windows.

---

# 35. No reinitializing history at split with target leakage

Do not preload first Validation target as history.

---

# 36. Continuity contract remains

Both protocols still require:

```text
exact 10-minute cadence
same continuity segment
gap-safe windows
no invalid duplicates/off-grid.
```

Boundary policy does not override temporal integrity.

---

# 37. Common population construction

For split `S`:

```text
WB0_ids_S
WB1_ids_S
```

define:

```text
COMMON_IDS_S = sorted intersection(WB0_ids_S, WB1_ids_S)
```

Hard:

```text
unique IDs
chronological order
same target timestamps.
```

---

# 38. Boundary-only population

Define:

```text
WB0_ONLY_IDS_S = WB0_ids_S - WB1_ids_S
```

Typically boundary targets.

This subset is useful for descriptive Validation sensitivity.

For Validation only, metrics may be computed separately.

For Test:

```text
count/timestamps metadata only
no labels/metrics.
```

---

# 39. Validation WB0-only metrics

Allowed:

```text
WB0 reference predictions on Validation WB0-only targets
```

if predictions are available or regenerated from Validation.

Compute:

```text
MAE/RMSE/R² only where mathematically valid.
```

R² on very small subset may be unstable/undefined; report carefully.

Purpose:

```text
describe difficulty of boundary-only region.
```

Not selection metric.

---

# 40. No WB1 metrics on WB0-only targets

WB1 has no valid window there.

Do not fabricate prediction.

---

# 41. Common-population model comparison

On `COMMON_IDS_VAL`, evaluate both WB0 and WB1 model checkpoints.

Required:

```text
same IDs
same target values
same ordering
same Wh-space metrics.
```

This is the cleanest model/protocol sensitivity comparison.

---

# 42. Native-population model comparison

WB0:

```text
all WB0-native Validation targets
```

WB1:

```text
all WB1-native Validation targets.
```

Report sample counts adjacent to metrics.

Never place native RMSE numbers side-by-side without noting different N.

---

# 43. Which metric selects S19 outcome?

S19 is a sensitivity check, so primary sensitivity metric should be:

```text
Common-population Validation RMSE Wh
```

because it controls target population.

However WB0 remains the project’s primary deployment protocol unless a formal protocol amendment changes it.

S19 should not silently rewrite the master forecasting contract based only on WB1.

---

# 44. Recommended S19 decision semantics

Use:

```text
ROBUST
SENSITIVE_BUT_ACCEPTABLE
MATERIAL_SENSITIVITY
INCONCLUSIVE
FAIL
```

based on evidence, but avoid arbitrary numeric thresholds unless predeclared.

Since no practical threshold was pre-registered, preferred classification should be evidence-based/descriptive rather than inventing a percentage cutoff.

At minimum report:

```text
exact delta RMSE
relative delta
sample-count delta
boundary-only behavior
warning flags.
```

---

# 45. No arbitrary “5% sensitivity threshold”

Do not create after seeing results.

If the project later needs a formal threshold:

```text
Protocol Amendment.
```

---

# 46. Reference protocol remains WB0

Default outcome:

```text
current primary protocol = WB0
```

and WB1 serves as sensitivity evidence.

Only a formal Protocol Amendment may promote WB1 as new primary protocol.

This protects sequential protocol consistency.

---

# 47. S19 does not optimize boundary protocol

Therefore there is no ordinary “exact tie → parsimony winner” rule.

Instead:

```text
WB0 = primary reference
WB1 = strict-isolation sensitivity.
```

Metrics quantify sensitivity.

---

# 48. Fresh WB1 run selection checkpoint

WB1 training still uses:

```text
BEST Validation RMSE Wh
```

on its **native WB1 Validation population** for internal early stopping/checkpointing, because that is the actual Validation set available under its protocol.

This must be recorded.

---

# 49. Important metric distinction

WB1 BEST checkpoint is selected by:

```text
WB1 native Validation RMSE.
```

WB0 reference BEST was selected by:

```text
WB0 native Validation RMSE.
```

After both checkpoints are fixed, S19 re-evaluates both on:

```text
common Validation target population.
```

This avoids reselecting checkpoints using common-pop metrics post hoc.

---

# 50. No checkpoint reselection on common population

Critical:

```text
do not scan historical epochs
and choose a different checkpoint
that looks best on COMMON_IDS_VAL.
```

Use the official BEST checkpoint of each protocol.

---

# 51. Why this matters

Otherwise S19 would introduce:

```text
new hidden checkpoint selection criterion
```

and break reproducibility.

---

# 52. Training fairness

Freeze:

```text
seed42
model architecture
loss
optimizer
LR
WD
batch
dropout
epoch cap
patience
clipping
RevIN
```

Only window boundary protocol changes.

---

# 53. Train sample population audit

Before training WB1:

```text
compare WB0 Train target IDs
vs
WB1 Train target IDs.
```

If equal:

```text
TRAIN_POPULATION_EQUAL = true.
```

If not:

```text
document exact difference
```

and verify strict-isolation logic.

---

# 54. If Train populations are identical

Then with deterministic initialization/order:

```text
WB0 and WB1 training trajectories may be identical
```

because only Validation eligibility differs.

But early stopping can still diverge because:

```text
Validation native sets differ.
```

Therefore:

```text
checkpoint/stop epoch can differ
```

even with identical Train updates up to divergence in stop timing.

---

# 55. If Train trajectories are identical before one run stops

This is expected and is useful evidence that S19 isolates Validation boundary protocol.

Optional audit:

```text
per-epoch train criterion
initial state
sample-order fingerprints
gradient summaries.
```

---

# 56. Validation set difference can alter early stopping

This is an intended S19 effect.

WB1 may:

```text
stop earlier/later
select different BEST epoch
```

because its Validation population excludes boundary targets.

Do not force same checkpoint epoch.

---

# 57. Common-pop comparison after checkpoint fixation

For final S19 sensitivity:

```text
load WB0 official BEST
load WB1 official BEST
evaluate both on COMMON_IDS_VAL
```

No further training.

---

# 58. Native-vs-common result matrix

Required table:

```text
                 Native Val        Common Val
WB0              metric_N0         metric_C0
WB1              metric_N1         metric_C1
```

plus:

```text
N_native
N_common
```

for every metric row.

---

# 59. Boundary sample count audit

Record:

```text
wb0_val_count
wb1_val_count
common_val_count
wb0_only_val_count
percentage_val_removed_by_wb1

wb0_test_count_metadata
wb1_test_count_metadata
common_test_count_metadata
wb0_only_test_count_metadata
```

No Test y.

---

# 60. Expected approximate removed count

Without gaps, WB1 may remove roughly:

```text
L* initial target opportunities
```

from each later split.

But do not hard-code this as observed.

Runtime continuity rules determine exact count.

---

# 61. First eligible target timestamps

Record for each split/protocol:

```text
first_target_timestamp
last_target_timestamp
```

This makes boundary effect transparent.

---

# 62. Last eligible target should normally match

WB0/WB1 usually differ only at split start.

If last eligible target differs unexpectedly:

```text
investigate continuity/window logic.
```

---

# 63. Target range audit

For Validation:

```text
WB0 first timestamp
WB1 first timestamp
difference in minutes/hours
last timestamps
```

For Test metadata similarly.

---

# 64. Window containment audit

Random/fixed probes near split boundary:

For WB0 Validation sample:

```text
target in Validation
some input rows may be Train
all inputs < target.
```

For WB1:

```text
target in Validation
all inputs in Validation.
```

Hard PASS evidence.

---

# 65. Cross-split row count per WB0 sample

For boundary-region WB0 targets, optionally record:

```text
number of input rows from previous split
number from current split.
```

This declines from L* toward 0 as targets move inward.

Useful figure/table.

---

# 66. WB1 strict containment proof

For every WB1 sample:

```text
unique(input_split_labels) == {target_split_label}
```

Hard assertion.

---

# 67. No target row inside input

Both protocols:

```text
max(input_timestamp) < target_timestamp
```

Hard assertion.

---

# 68. No off-by-one

For H1:

```text
last input = t
target = t+10min
```

not:

```text
last input = target row.
```

---

# 69. Scaler fairness

Same frozen:

```text
X scaler checksum
Y scaler checksum/identity
```

across WB0/WB1.

No scaler fit from WB1 Validation.

---

# 70. Feature-engineering fairness

Same deterministic time features.

No boundary-specific feature transformation.

---

# 71. RevIN scope fairness

If RN1:

```text
same channel list
same indices
same eps
same affine config
same coordinate bridge.
```

Only available input window differs by protocol.

---

# 72. No cross-boundary RevIN context under WB1

Even if adapter caches context, all stats must come from current window.

Hard leakage/context audit.

---

# 73. DataLoader settings

WB1:

```text
Train shuffle=true
Validation shuffle=false
drop_last=false
same B*
same worker policy
same seed policy.
```

---

# 74. Common-population evaluator

Create a dedicated evaluator that accepts:

```text
checkpoint
protocol-specific Dataset
target_id whitelist = COMMON_IDS_VAL
```

and verifies exact ordering.

Do not rebuild targets inconsistently.

---

# 75. Prediction bundle requirements

Common evaluation should store:

```text
target_id
target_timestamp
y_true_wh
y_pred_wh
protocol_id
checkpoint_run_id
```

for Validation only.

---

# 76. Native prediction bundles

Store separate:

```text
wb0_native_validation_predictions
wb1_native_validation_predictions
```

No Test prediction bundle.

---

# 77. Metrics

For each valid Validation view:

```text
MAE Wh
RMSE Wh
R²
```

Primary sensitivity:

```text
Common Val RMSE Wh.
```

Native metrics secondary/operational.

---

# 78. Delta definitions

Common-pop effect:

\[
\Delta RMSE_{common}
=
RMSE_{WB0,common}
-
RMSE_{WB1,common}
\]

Positive:

```text
WB1 lower RMSE.
```

Native effect:

\[
\Delta RMSE_{native}
=
RMSE_{WB0,native}
-
RMSE_{WB1,native}
\]

but label:

```text
different target populations.
```

---

# 79. Sample-loss effect

Define:

\[
RemovedRate_{Val}
=
100
\times
\frac{N_{WB0}-N_{WB1}}{N_{WB0}}
\]

This quantifies strict-isolation coverage cost.

---

# 80. Boundary-region difficulty

For WB0-only Validation targets:

```text
MAE/RMSE
```

can be compared descriptively against WB0-common targets.

Purpose:

```text
are boundary-only targets systematically harder/easier?
```

No causal overclaim.

---

# 81. Boundary-region target distribution

Optional descriptive:

```text
mean/median/std/min/max y_true Wh
```

for:

```text
WB0-only Val
COMMON Val
```

Validation only.

This helps explain native metric shifts.

---

# 82. No threshold tuning from boundary subset

Do not change model/config based on this descriptive analysis.

---

# 83. Working hypothesis H-S19-01

```text
WB0 and WB1 common-population performance may be similar if model training is robust and common-target windows are effectively identical.
```

Status:

```text
UNTESTED.
```

---

# 84. H-S19-02

```text
Native WB0/WB1 metrics may differ partly because WB1 removes early-boundary targets.
```

Status:

```text
UNTESTED.
```

---

# 85. H-S19-03

```text
WB0 should provide broader immediate post-split forecast coverage under rolling one-step deployment assumptions.
```

Protocol expectation.

---

# 86. H-S19-04

```text
If WB1 produces materially different common-population results, sequential model conclusions may be sensitive to split-boundary context assumptions.
```

Status:

```text
UNTESTED.
```

---

# 87. Preconditions

Required:

```text
Phase40 overall status ∈
{PASS, PASS_WITH_WARNING, SKIPPED_NOT_APPLICABLE}
```

and:

```text
approved_for_phase41=true.
```

---

# 88. Required upstream artifacts

```text
s18_revin_winner.json or skip-equivalent outcome
s18_reference_update.json
phase_40_signoff.json
```

---

# 89. Carry-forward warnings

Propagate:

```text
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
BOUNDARY_WINNER
REVIN_NOT_APPLICABLE
SAMPLE_ORDER_NOT_VERIFIABLE
INITIALIZATION_NOT_VERIFIABLE
RANDOM_CONTROL_GAIN
other unresolved noncritical warnings.
```

---

# 90. WB0 reference reuse gate

Exact match required on:

```text
all S1-S18 selected configuration
WB0
WINDOWPOP-v1
seed42
Training Engine
Metric version
current RevIN config
current clipping config
current epoch cap/loss/etc.
```

Mismatch:

```text
STOP.
```

---

# 91. WB1 new run config

Only delta:

```text
window_boundary_protocol:
WB0 → WB1.
```

Everything else identical.

---

# 92. Experiment Registry

Family:

```text
TRANSFORMER_SENSITIVITY_S19_BOUNDARY
```

Recommended run tags:

```text
S19_WB0_REFERENCE
S19_WB1_STRICT_ISOLATION.
```

---

# 93. Preflight artifact directory

```text
artifacts/sweeps/S19_boundary_protocol/
```

---

# 94. Output directory

```text
artifacts/
└── sweeps/
    └── S19_boundary_protocol/
        ├── s19_boundary_sweep_manifest.json
        ├── s19_boundary_sweep_contract.json
        ├── s19_boundary_preflight_audit.csv
        ├── s19_run_matrix.csv
        ├── s19_boundary_definition_audit.csv
        ├── s19_split_boundary_audit.csv
        ├── s19_window_containment_tests.csv
        ├── s19_target_population_audit.csv
        ├── s19_train_population_audit.csv
        ├── s19_validation_population_audit.csv
        ├── s19_test_population_metadata_audit.csv
        ├── s19_common_population_audit.csv
        ├── s19_window_fingerprint_audit.csv
        ├── s19_boundary_context_depth_audit.csv
        ├── s19_scaler_invariance_audit.csv
        ├── s19_feature_invariance_audit.csv
        ├── s19_revin_boundary_audit.csv
        ├── s19_training_config_delta_audit.csv
        ├── s19_initialization_audit.csv
        ├── s19_sample_order_audit.csv
        ├── s19_optimizer_budget_audit.csv
        ├── s19_boundary_run_provenance.csv
        ├── s19_native_validation_predictions_wb0.csv
        ├── s19_native_validation_predictions_wb1.csv
        ├── s19_common_validation_predictions_wb0.csv
        ├── s19_common_validation_predictions_wb1.csv
        ├── s19_boundary_only_validation_predictions_wb0.csv
        ├── s19_native_validation_metrics.csv
        ├── s19_common_validation_metrics.csv
        ├── s19_boundary_only_validation_metrics.csv
        ├── s19_boundary_effects.csv
        ├── s19_coverage_effects.csv
        ├── s19_learning_curve_diagnostics.csv
        ├── s19_convergence_diagnostics.csv
        ├── s19_runtime_diagnostics.csv
        ├── s19_hypothesis_outcomes.csv
        ├── s19_boundary_findings.csv
        ├── s19_boundary_sensitivity_conclusion.json
        ├── s19_reference_update.json
        ├── s19_boundary_sweep_tests.csv
        ├── s19_boundary_discrepancies.json
        ├── s19_boundary_sweep_summary.json
        ├── s19_boundary_sweep_report.md
        ├── figures/
        │   ├── S19_01_validation_rmse_native_vs_common.png
        │   ├── S19_02_validation_mae_native_vs_common.png
        │   ├── S19_03_target_population_coverage.png
        │   ├── S19_04_first_eligible_target_timeline.png
        │   ├── S19_05_boundary_context_depth.png
        │   ├── S19_06_learning_curves.png
        │   ├── S19_07_common_population_prediction_scatter.png
        │   ├── S19_08_common_population_residual_difference.png
        │   └── S19_09_boundary_only_vs_common_error.png
        ├── README_S19_BOUNDARY_PROTOCOL.md
        └── phase_41_signoff.json
```

No Test prediction/metric files.

---

# 95. Required outputs

```text
O41.1  Manifest
O41.2  Contract
O41.3  Preflight audit
O41.4  Run matrix
O41.5  Boundary definition audit
O41.6  Split boundary audit
O41.7  Window containment tests
O41.8  Target population audit
O41.9  Train population audit
O41.10 Validation population audit
O41.11 Test population metadata audit
O41.12 Common population audit
O41.13 Window fingerprint audit
O41.14 Boundary context-depth audit
O41.15 Scaler invariance audit
O41.16 Feature invariance audit
O41.17 RevIN boundary audit if RN1
O41.18 Training-config delta audit
O41.19 Initialization audit
O41.20 Sample-order audit
O41.21 Optimizer-budget audit
O41.22 Run provenance
O41.23 Reused WB0 reference
O41.24 Fresh WB1 run
O41.25 Native Validation prediction bundles
O41.26 Common Validation prediction bundles
O41.27 WB0-only boundary prediction bundle
O41.28 Native Validation metrics
O41.29 Common Validation metrics
O41.30 Boundary-only metrics
O41.31 Boundary effects
O41.32 Coverage effects
O41.33 Learning-curve diagnostics
O41.34 Convergence diagnostics
O41.35 Runtime diagnostics
O41.36 Hypothesis outcomes
O41.37 Findings
O41.38 Sensitivity conclusion
O41.39 Phase42 reference update
O41.40 Figures
O41.41 Tests
O41.42 Discrepancies
O41.43 Summary
O41.44 Report
O41.45 README
O41.46 Sign-off
```

---

# 96. Manifest schema

`s19_boundary_sweep_manifest.json`:

```text
sweep_id = S19_BOUNDARY_PROTOCOL
sweep_version = SWEEP_S19_BOUNDARY-v1
source_s18_run_id
candidate_protocols = [WB0,WB1]
primary_protocol = WB0
sensitivity_protocol = WB1
common_population_required = true
native_population_required = true
test_labels_forbidden = true
all_selected_S1_S18_fields
population_version
metric_version
training_engine_version
seed=42
new_runs_required=1
reused_runs=1
status
```

---

# 97. Contract artifact

`s19_boundary_sweep_contract.json` must state:

```text
WB0:
target split determines sample split;
past context may originate from previous split.

WB1:
all input rows and target must belong to same split.

No future leakage in either.

Same split cut points.
Same lookback/horizon.
Same scalers/features/model/training.
WB0 reused.
WB1 fresh.

Native populations differ.
Common-population comparison required.

WB0 remains primary protocol.
WB1 is sensitivity evidence.
No Test labels/predictions/metrics.
```

---

# 98. Boundary definition audit

`s19_boundary_definition_audit.csv`:

```text
protocol
target_assignment_rule
cross_split_past_context_allowed
all_inputs_same_split_required
future_input_allowed
padding_allowed
continuity_required
status
```

Expected:

```text
future_input_allowed=false
padding_allowed=false
continuity_required=true.
```

---

# 99. Split boundary audit

Record:

```text
train_start
train_end
validation_start
validation_end
test_start
test_end
```

with half-open interval semantics.

No change from Phase8.

---

# 100. Window containment tests

`s19_window_containment_tests.csv` should include:

```text
Train boundary cases
first WB0 Validation target
first WB1 Validation target
near-boundary Validation
deep Validation target
first WB0 Test target metadata
first WB1 Test target metadata
```

Fields:

```text
protocol
split
target_id
target_timestamp
input_start
input_end
input_split_labels
target_split
all_inputs_before_target
all_inputs_same_split
continuity_valid
expected_valid
observed_valid
status
```

No Test target values.

---

# 101. Target population audit

For each split/protocol:

```text
native_count
first_target_id/timestamp
last_target_id/timestamp
fingerprint
```

No hidden target values needed.

---

# 102. Train population audit

Expected:

```text
WB0_train_ids == WB1_train_ids
```

If unequal, explain.

---

# 103. Validation population audit

Required:

```text
WB0 IDs
WB1 IDs
intersection
WB0-only
WB1-only
```

Expected generally:

```text
WB1-only = empty
```

because strict isolation should be subset of carry-over eligibility.

If WB1-only non-empty:

```text
STOP and investigate.
```

---

# 104. Test population metadata audit

Same set logic but without labels.

Expected:

```text
WB1_test_ids ⊆ WB0_test_ids.
```

---

# 105. Set-inclusion hard assertions

For Validation/Test:

```text
WB1_ids ⊆ WB0_ids
```

under same L/H/continuity.

Violation indicates protocol implementation bug.

---

# 106. Common-population fingerprint

Store:

```text
common_val_target_ids_sha256
common_test_target_ids_sha256
```

Test fingerprint uses IDs only.

---

# 107. Window fingerprint audit

For `COMMON_IDS_VAL`:

```text
input row ID sequence hash under WB0
input row ID sequence hash under WB1
```

per target or aggregate.

Record:

```text
equal_windows_count
different_windows_count
fraction_equal.
```

---

# 108. Context depth audit

For WB0-only boundary targets:

```text
previous_split_rows_in_window
current_split_rows_in_window
```

Expected monotonic transition toward:

```text
0 previous-split rows
```

as target moves inward.

---

# 109. Scaler invariance audit

Check checksums:

```text
X scaler
Y scaler
```

equal across protocols.

---

# 110. Feature invariance audit

Check:

```text
feature names/order
TF setting
RevIN scope if active.
```

---

# 111. RevIN boundary audit

If RN1 selected:

For WB1 Validation:

```text
RevIN context input row IDs
```

must all belong to Validation.

For WB0:

```text
may include Train rows
but all before target.
```

No future label.

---

# 112. Training config delta audit

Only allowed diff:

```text
window_boundary_protocol.
```

If Train population/index arrays differ as a consequence, record as derived effect, not second manually changed config.

---

# 113. Initialization audit

WB1 fresh run:

```text
seed42
same initialization policy
shared exact model config.
```

Compare initial-state fingerprint with WB0 reference if available.

Else:

```text
NOT_VERIFIABLE.
```

---

# 114. Sample-order audit

If Train IDs equal and WB0 historical order fingerprint exists:

```text
compare epoch order.
```

If unavailable:

```text
NOT_VERIFIABLE.
```

---

# 115. Optimizer-budget audit

Fields:

```text
protocol
train_samples_per_epoch
steps_per_epoch
epochs_completed
total_optimizer_steps
best_epoch
steps_to_best
```

If Train sets equal:

```text
steps_per_epoch expected equal.
```

---

# 116. WB1 fresh run training

Execute:

```text
register WB1
reseed42
build WB1 Train/Val datasets
fresh loaders
fresh selected model
fresh selected loss
fresh AdamW
selected GC*
selected RN*
train with TRAINING_ENGINE-v1
```

No Test.

---

# 117. WB1 early stopping

Uses:

```text
WB1 native Validation RMSE Wh.
```

Same patience/min_delta.

---

# 118. WB1 BEST verification

After run:

```text
fresh exact model
strict-load WB1 BEST
evaluate on full native WB1 Validation
verify stored metrics.
```

Then separately evaluate common population.

---

# 119. WB0 common evaluation

Load existing WB0 BEST checkpoint.

Evaluate only:

```text
COMMON_IDS_VAL.
```

Do not retrain.

---

# 120. WB1 common evaluation

Use fixed WB1 BEST.

Evaluate:

```text
COMMON_IDS_VAL.
```

---

# 121. Prediction identity alignment

Before common metrics:

```text
WB0 target IDs == WB1 target IDs == COMMON_IDS_VAL
```

exact same order.

Hard.

---

# 122. Common target truth equality

Verify:

```text
y_true_wh_WB0 == y_true_wh_WB1
```

for common IDs.

Hard.

---

# 123. Native metrics table

`s19_native_validation_metrics.csv`:

```text
protocol
run_id
sample_count
first_target
last_target
mae_wh
rmse_wh
r2
best_epoch
stop_reason
population_fingerprint
status
```

---

# 124. Common metrics table

`s19_common_validation_metrics.csv`:

```text
protocol
run_id
sample_count
common_population_fingerprint
mae_wh
rmse_wh
r2
status
```

---

# 125. Boundary-only metrics table

For WB0-only Validation targets:

```text
sample_count
first_target
last_target
mae_wh
rmse_wh
r2_if_valid
target_mean_wh
target_std_wh
status
```

No WB1 row unless explicit N/A.

---

# 126. Boundary effects table

`s19_boundary_effects.csv`:

```text
metric
wb0_native
wb1_native
native_delta
wb0_common
wb1_common
common_delta
interpretation_scope
```

---

# 127. Coverage effects table

`s19_coverage_effects.csv`:

```text
split
wb0_count
wb1_count
common_count
wb0_only_count
wb1_only_count
removed_by_wb1
removed_rate_pct
first_target_delay_minutes
status
```

Test row metadata only.

---

# 128. Learning curves

Overlay:

```text
WB0 official history
WB1 official history
```

but remember Validation metrics are on different native populations.

Label plot:

```text
Native-protocol Validation RMSE; populations differ.
```

Do not interpret small curve gap as pure model effect.

---

# 129. Common-pop metrics are checkpoint-level only

Unless you pre-register recomputing common metrics at every epoch for both runs, do not generate retrospective per-epoch common curves by checkpoint fishing.

BEST-level common evaluation is sufficient and cleaner.

---

# 130. Convergence diagnostics

Record:

```text
best epoch
last epoch
stop reason
epochs completed
best-to-last gap
```

and explain Validation-population difference.

---

# 131. Runtime diagnostics

Secondary:

```text
epoch seconds
total runtime
time to BEST
```

WB1 Validation has fewer samples and may evaluate slightly faster.

Do not interpret runtime as model quality.

---

# 132. Sensitivity conclusion artifact

Create:

```text
s19_boundary_sensitivity_conclusion.json
```

Minimum:

```text
primary_protocol = WB0
sensitivity_protocol = WB1
common_population_rmse_wb0
common_population_rmse_wb1
common_delta_rmse_wh
common_delta_rmse_pct
native_rmse_wb0
native_rmse_wb1
native_delta_rmse_wh
validation_coverage_removed_pct
test_coverage_removed_pct_metadata
common_window_fraction_equal
train_population_equal
boundary_only_rmse_wb0
sensitivity_interpretation
primary_protocol_changed = false
protocol_amendment_required = false/true
warnings
status
```

---

# 133. Sensitivity interpretation categories

Recommended:

```text
ROBUST
SENSITIVITY_OBSERVED
MATERIAL_SENSITIVITY_REQUIRES_REVIEW
INCONCLUSIVE
```

Do not attach arbitrary numeric thresholds.

Use narrative evidence:

```text
common delta magnitude
coverage loss
metric ranking changes
checkpoint/convergence changes
boundary-only difficulty
```

---

# 134. Protocol amendment trigger

If WB1 reveals severe contradiction that calls the primary forecasting assumption into question:

```text
do not silently select WB1.
```

Instead:

```text
protocol_amendment_required = true
```

and stop before Candidate Synthesis until amendment is documented.

Examples:

```text
unexpected leakage discovered in WB0
window containment bug
common-population conclusions radically inconsistent due implementation issue.
```

---

# 135. WB0 remains primary if methodologically valid

Even if WB1 RMSE is numerically lower, that does not automatically mean WB0 should be replaced.

WB0 corresponds to the predeclared rolling one-step forecasting assumption.

WB1 is stricter but sacrifices initial split coverage.

The choice is methodological, not merely metric-driven.

---

# 136. No “winner” file in ordinary sweep sense

Prefer:

```text
s19_boundary_sensitivity_conclusion.json
```

rather than a simplistic:

```text
s19_winner.json
```

because this phase is a robustness check.

---

# 137. Phase42 reference update

`s19_reference_update.json` should normally carry:

```text
primary_protocol = WB0
primary_reference_run_id = S18 selected WB0 run
WB1_sensitivity_run_id
common_population_findings
coverage_findings
sensitivity_status
protocol_amendment_required
approved_for_phase42
```

If no amendment required:

```text
approved_for_phase42=true.
```

If critical methodological issue:

```text
approved_for_phase42=false.
```

---

# 138. Candidate Synthesis handoff

Phase42 should receive:

```text
all S1–S18 selected configuration
WB0 primary reference
WB1 sensitivity evidence
sample-population audit
common-population delta
coverage cost
warnings
```

Candidate synthesis must not treat WB1 as an extra hyperparameter winner unless protocol amendment explicitly changes the contract.

---

# 139. Figures

Recommended:

```text
S19_01_validation_rmse_native_vs_common.png
S19_02_validation_mae_native_vs_common.png
S19_03_target_population_coverage.png
S19_04_first_eligible_target_timeline.png
S19_05_boundary_context_depth.png
S19_06_learning_curves.png
S19_07_common_population_prediction_scatter.png
S19_08_common_population_residual_difference.png
S19_09_boundary_only_vs_common_error.png
```

---

# 140. First eligible target timeline figure

Show:

```text
Train boundary
Validation start
WB0 first Validation target
WB1 first Validation target
```

and analogous Test metadata boundary without target values.

This is one of the clearest protocol illustrations.

---

# 141. Boundary context-depth figure

For initial Validation WB0-only targets:

```text
x = target timestamp
y = number of previous-split rows in lookback
```

Should decrease toward zero.

---

# 142. Common prediction scatter

Validation only.

Use same common target IDs.

No Test.

---

# 143. Residual difference plot

For common Validation IDs:

```text
residual_WB0
residual_WB1
difference
```

Descriptive only.

Do not perform Phase49 full residual analysis here.

---

# 144. Test firewall checklist

Hard:

```text
[ ] no Test y read
[ ] no Test prediction
[ ] no Test metric
[ ] no Test residual
[ ] no Test plot using target values
[ ] no Test-based decision
```

Allowed:

```text
Test target ID/timestamp eligibility metadata only.
```

---

# 145. Common mistakes

## 145.1 So sánh native RMSE như thể cùng target set

Sai.

## 145.2 Không tạo common-population evaluation

Sai.

## 145.3 Cho WB1 dùng short windows ở split start

Sai.

## 145.4 Padding đầu Validation cho WB1

Sai.

## 145.5 Refit scaler cho WB1

Sai.

## 145.6 Dùng Test metrics để “check robustness”

Forbidden.

## 145.7 WB1 RMSE tốt hơn rồi tự đổi primary protocol

Sai; cần protocol-level reasoning/amendment.

## 145.8 Chọn checkpoint lại theo common population

Sai hidden selection.

## 145.9 Report WB0 cross-split past context là leakage

Không chính xác nếu mọi inputs precede target và deployment assumes past observations available.

## 145.10 Nói WB1 luôn “more correct”

Không đúng tuyệt đối; đây là stricter isolation assumption, đổi coverage/deployment semantics.

---

# 146. Recommended notebook structure

```text
Cell 41.1  Phase title
Cell 41.2  Verify Phase40 sign-off
Cell 41.3  Declare SWEEP_S19_BOUNDARY-v1
Cell 41.4  Load S18 selected config
Cell 41.5  Freeze S1-S18 fields
Cell 41.6  Declare WB0/WB1 contracts
Cell 41.7  Load split boundaries
Cell 41.8  Build WB0 eligibility indices
Cell 41.9  Build WB1 eligibility indices
Cell 41.10 Run set-inclusion assertions
Cell 41.11 Audit Train population equality
Cell 41.12 Audit Validation native/common/boundary-only populations
Cell 41.13 Audit Test metadata populations
Cell 41.14 Build window containment tests
Cell 41.15 Build common-window fingerprints
Cell 41.16 Build context-depth audit
Cell 41.17 Audit scaler/features/RevIN invariance
Cell 41.18 Verify WB0 reference reuse
Cell 41.19 Register WB1 run
Cell 41.20 Reseed + fresh WB1 loaders/model/optimizer
Cell 41.21 Train WB1 via TRAINING_ENGINE-v1
Cell 41.22 Verify WB1 BEST native metrics
Cell 41.23 Evaluate WB0 BEST on COMMON_IDS_VAL
Cell 41.24 Evaluate WB1 BEST on COMMON_IDS_VAL
Cell 41.25 Evaluate WB0 boundary-only Validation
Cell 41.26 Build prediction bundles
Cell 41.27 Build native metrics
Cell 41.28 Build common metrics
Cell 41.29 Build boundary-only metrics
Cell 41.30 Compute boundary effects
Cell 41.31 Compute coverage effects
Cell 41.32 Build convergence/runtime diagnostics
Cell 41.33 Evaluate hypotheses
Cell 41.34 Generate figures
Cell 41.35 Write findings
Cell 41.36 Write sensitivity conclusion
Cell 41.37 Determine protocol amendment requirement
Cell 41.38 Write Phase42 reference update
Cell 41.39 Run acceptance tests
Cell 41.40 Write discrepancy log
Cell 41.41 Write summary/report
Cell 41.42 Register artifacts/checksums
Cell 41.43 Write README
Cell 41.44 Phase sign-off
```

---

# 147. Execution flow

```text
Verify Phase40
→ Load selected model/config
→ Freeze S1-S18
→ Build WB0 and WB1 eligibility
→ Assert WB1 subset of WB0
→ Build common target population
→ Audit boundary containment
→ Audit scaler/feature/RevIN invariance
→ Reuse WB0
→ Fresh WB1 seed42 run
→ Verify WB1 BEST on native WB1 Validation
→ Freeze both checkpoints
→ Evaluate both on common Validation IDs
→ Analyze WB0-only boundary region
→ Quantify Validation/Test coverage loss
→ Write robustness conclusion
→ keep WB0 primary unless formal amendment
→ handoff Phase42
```

---

# 148. Fail-fast order

Before WB1 training:

```text
1. Phase40 signoff valid
2. S18 selected config valid
3. split boundaries unchanged
4. WB0 definition valid
5. WB1 strict-containment definition valid
6. same H/L
7. no padding
8. continuity rules same
9. WB1 Val IDs subset WB0 Val IDs
10. WB1 Test IDs subset WB0 Test IDs
11. common population constructed
12. Train population audited
13. scalers identical
14. features identical
15. RevIN config identical if active
16. training config delta only boundary protocol
17. WB0 reuse exact
18. Test label firewall
19. Registry ready
```

---

# 149. Acceptance checklist

```text
[ ] Phase40 valid.
[ ] approved_for_phase41=true.
[ ] SWEEP_S19_BOUNDARY-v1 declared.
[ ] S18 selected configuration loaded.
[ ] All S1-S18 fields frozen.
[ ] WB0 exact definition locked.
[ ] WB1 exact strict-isolation definition locked.
[ ] Same split boundaries.
[ ] Same H1.
[ ] Same L*.
[ ] No padding.
[ ] Same continuity/gap rules.
[ ] No future target in either protocol.
[ ] WB1 inputs all same split as target.
[ ] WB0 cross-split rows are strictly past.
[ ] Train populations audited.
[ ] Validation WB1 IDs subset WB0 IDs.
[ ] Test WB1 IDs subset WB0 IDs.
[ ] Common Val IDs built.
[ ] Common Test metadata IDs built.
[ ] WB0-only Val IDs built.
[ ] Test labels never read.
[ ] First eligible timestamps recorded.
[ ] Population fingerprints generated.
[ ] Common-window fingerprints audited.
[ ] Context-depth audit generated.
[ ] Same X scaler checksums.
[ ] Same Y scaler/identity.
[ ] Same features/order.
[ ] Same RN*.
[ ] If RN1, WB1 RevIN uses only strict local window.
[ ] Same architecture.
[ ] Same loss.
[ ] Same AdamW.
[ ] Same LR/WD.
[ ] Same dropout.
[ ] Same batch.
[ ] Same epoch cap.
[ ] Same patience/min_delta.
[ ] Same clipping.
[ ] Same seed42.
[ ] WB0 reference exact match.
[ ] WB0 not retrained.
[ ] WB1 registered.
[ ] WB1 fresh initialization.
[ ] No warm-start.
[ ] No optimizer-state reuse.
[ ] WB1 native early stopping uses WB1 native Val RMSE.
[ ] WB1 BEST verified.
[ ] WB0 official BEST fixed.
[ ] No checkpoint reselection on common population.
[ ] Both BEST checkpoints evaluated on COMMON_IDS_VAL.
[ ] Common target IDs/order identical.
[ ] Common y_true Wh identical.
[ ] Native metrics generated with sample counts.
[ ] Common metrics generated.
[ ] Boundary-only Validation metrics generated.
[ ] Coverage effects generated.
[ ] Common RMSE delta generated.
[ ] Native RMSE delta labeled population-different.
[ ] RMSE/R² consistency checked on common set.
[ ] Boundary-only metrics not used for model selection.
[ ] Runtime not used as robustness criterion.
[ ] No arbitrary sensitivity threshold invented.
[ ] WB0 remains primary unless amendment.
[ ] Protocol amendment flag set appropriately.
[ ] Phase42 reference update written.
[ ] No Test prediction.
[ ] No Test metric.
[ ] No Test residual.
[ ] Summary/report/README generated.
[ ] Sign-off generated.
```

---

# 150. Acceptance criteria

Phase41 PASS only when:

```text
WB0 and WB1 definitions are implemented exactly.

WB1 strict isolation uses only same-split input rows.

Both protocols preserve causal ordering and continuity rules.

Same split boundaries, lookback, horizon, scalers, features, model and training configuration are used.

WB1 Validation/Test target IDs are subsets of WB0 target IDs.

Native and common target populations are both explicitly audited.

WB0 reference is reused.

WB1 is one fresh seed42 run.

WB1 BEST is selected only by its native Validation RMSE and verified.

Official WB0/WB1 BEST checkpoints are frozen before common-pop evaluation.

Both checkpoints are evaluated on exactly the same common Validation target IDs.

Boundary-only WB0 Validation region is analyzed descriptively.

Validation/Test coverage loss under WB1 is quantified.

No Test labels/predictions/metrics are accessed.

Sensitivity conclusion is documented without silently redefining primary protocol.

Phase42 handoff is generated.
```

---

# 151. Failure conditions

Phase41 FAIL if:

```text
split boundaries change

WB1 uses previous-split rows

WB1 uses padding/short windows

future target appears in input

continuity rules differ

scalers are refit

feature/model/training settings drift

WB1 Val/Test IDs are not subsets of WB0

common population is omitted

native metrics are compared as if same population

checkpoint is reselected on common population

WB0 is retrained and favorable rerun chosen

WB1 warm-starts from WB0

optimizer state reused

Test labels/predictions/metrics accessed

WB1 numeric RMSE automatically replaces WB0 primary protocol without amendment.
```

---

# 152. Sensitivity conclusion guidance

## Case A — Common metrics very similar, coverage reduced under WB1

Interpretation:

```text
model conclusion robust to stricter boundary isolation;
WB0 preserves broader immediate post-split coverage.
```

## Case B — Native metrics differ but common metrics similar

Interpretation:

```text
difference is largely associated with changed evaluation population/boundary region.
```

## Case C — Common metrics differ meaningfully

Interpretation:

```text
strict-isolation protocol affects trained checkpoint/generalization behavior;
record sensitivity warning.
```

## Case D — WB0 implementation leakage discovered

Interpretation:

```text
critical protocol issue;
do not proceed to Phase42 without amendment/fix.
```

---

# 153. Safe reporting examples

Safe:

> WB0 allows strictly historical pre-boundary context for the earliest Validation forecasts, whereas WB1 requires all lookback rows to lie within the same split. Both protocols remain causal.

Safe:

> Because WB1 removes early-boundary targets, native Validation RMSE values are not strictly population-matched; therefore we additionally compare both fixed checkpoints on the intersection of valid Validation targets.

Safe:

> WB0 remains the predeclared primary forecasting protocol, while WB1 is reported as a strict-isolation sensitivity analysis.

---

# 154. Reporting prohibitions

Do not write:

```text
WB0 leaks data
```

unless actual future information was found.

Do not write:

```text
WB1 is always more scientifically correct.
```

Do not write:

```text
WB1 lower RMSE means WB1 is automatically final protocol.
```

Do not hide:

```text
sample-count/coverage loss.
```

---

# 155. Phase42 handoff contract

Phase42 receives:

```text
primary run = WB0 S18-selected reference
strict-isolation sensitivity run = WB1
all S1-S18 selected fields
WB0/WB1 native metrics
common-population metrics
coverage effect
boundary-only diagnostics
protocol amendment flag
warnings
population fingerprints
```

If:

```text
protocol_amendment_required=false
```

then:

```text
approved_for_phase42=true.
```

---

# 156. Phase sign-off

Create:

```text
phase_41_signoff.json
```

Minimum:

```text
phase = 41
phase_name = S19 Boundary protocol check
sweep_version
sweep_id
source_s18_run_id
all selected S1-S18 fields
wb0_reference_run_id
wb1_run_id
wb0_val_native_count
wb1_val_native_count
common_val_count
wb0_only_val_count
test_counts_metadata_only
train_population_equal
wb1_subset_wb0_validation
wb1_subset_wb0_test
common_window_fraction_equal
wb0_native_rmse_wh
wb1_native_rmse_wh
wb0_common_rmse_wh
wb1_common_rmse_wh
common_delta_rmse_wh
validation_coverage_removed_pct
test_coverage_removed_pct
sensitivity_interpretation
primary_protocol = WB0
protocol_amendment_required
test_label_status = NOT_ACCESSED
test_prediction_status = NOT_GENERATED
approved_for_phase42
overall_status
created_at
```

---

# 157. Recommended discrepancy taxonomy

```text
S18_REFERENCE_MISSING
S18_REFERENCE_MISMATCH
SPLIT_BOUNDARY_DRIFT
WB0_DEFINITION_MISMATCH
WB1_DEFINITION_MISMATCH
WB1_CROSS_SPLIT_INPUT
FUTURE_INPUT_LEAKAGE
PADDING_USED
CONTINUITY_RULE_DRIFT
LOOKBACK_DRIFT
HORIZON_DRIFT
FEATURE_DRIFT
SCALER_DRIFT
REVIN_DRIFT
MODEL_CONFIG_DRIFT
TRAINING_CONFIG_DRIFT
WB1_NOT_SUBSET_OF_WB0
TRAIN_POPULATION_UNEXPECTED_DIFFERENCE
COMMON_POPULATION_MISSING
COMMON_TARGET_ORDER_MISMATCH
COMMON_YTRUE_MISMATCH
WINDOW_FINGERPRINT_UNEXPECTED_DIFFERENCE
CHECKPOINT_RESELECTION_ON_COMMON_SET
WB0_RETRAINED
WARM_START_USED
OPTIMIZER_STATE_REUSE
NATIVE_POPULATION_COMPARISON_MISLABELED
TEST_LABEL_ACCESSED
TEST_PREDICTION_GENERATED
TEST_METRIC_COMPUTED
PRIMARY_PROTOCOL_CHANGED_WITHOUT_AMENDMENT
OTHER
```

---

# 158. Recommended findings codes

```text
WB0_PRIMARY_VALID
WB1_STRICT_ISOLATION_VALID
TRAIN_POPULATIONS_EQUAL
TRAIN_POPULATIONS_DIFFER
WB1_VALIDATION_SUBSET_VERIFIED
WB1_TEST_SUBSET_VERIFIED
COMMON_WINDOW_INPUTS_IDENTICAL
COMMON_WINDOW_INPUTS_DIFFER
BOUNDARY_COVERAGE_LOSS
NATIVE_METRIC_DIFFERENCE
COMMON_METRIC_STABLE
COMMON_METRIC_SENSITIVITY
BOUNDARY_ONLY_TARGETS_HARDER
BOUNDARY_ONLY_TARGETS_EASIER
BOUNDARY_ONLY_TARGETS_SIMILAR
EARLY_STOP_DIFFERENCE
BEST_EPOCH_DIFFERENCE
PROTOCOL_ROBUST
PROTOCOL_SENSITIVE
PROTOCOL_AMENDMENT_REQUIRED
TEST_FIREWALL_PRESERVED
INHERITED_WARNING
```

---

# 159. Recommended execution pseudocode

```text
load_phase40_signoff()
assert approved_for_phase41

cfg = load_s18_reference_update()
freeze_S1_to_S18(cfg)

split = load_SPLIT_v1()

wb0_indices = build_windows(
    protocol="WB0",
    L=L*,
    H=1,
    continuity="TEMPORAL-v1"
)

wb1_indices = build_windows(
    protocol="WB1",
    L=L*,
    H=1,
    continuity="TEMPORAL-v1",
    require_all_rows_same_split=True
)

assert wb1_val_ids <= wb0_val_ids
assert wb1_test_ids <= wb0_test_ids

common_val = intersection(wb0_val_ids, wb1_val_ids)
wb0_only_val = wb0_val_ids - wb1_val_ids
common_test_metadata = intersection(wb0_test_ids, wb1_test_ids)

audit_train_population()
audit_window_containment()
audit_window_fingerprints(common_val)
audit_context_depth(wb0_only_val)
audit_scalers_features_revin()

wb0 = resolve_exact_s18_reference()

register_wb1_run()
reseed(42)

train_loader_wb1 = build_fresh_train_loader(wb1_indices.train)
val_loader_wb1   = build_fresh_val_loader(wb1_indices.validation)

model = build_fresh_selected_model()
criterion = build_selected_loss()
optimizer = build_selected_adamw()

wb1 = TRAINING_ENGINE_v1.fit(
    model,
    train_loader_wb1,
    val_loader_wb1,
    max_epochs=EPOCHS*,
    patience=10,
    gradient_clip=GC*
)

verify_wb1_best_on_native_val()

pred_wb0_common = evaluate_checkpoint(
    wb0.best_checkpoint,
    protocol="WB0",
    ids=common_val
)

pred_wb1_common = evaluate_checkpoint(
    wb1.best_checkpoint,
    protocol="WB1",
    ids=common_val
)

assert same_target_ids_and_ytrue(
    pred_wb0_common,
    pred_wb1_common
)

metrics_common = compute_shared_metrics()
metrics_native = collect_native_metrics()

pred_wb0_boundary = evaluate_checkpoint(
    wb0.best_checkpoint,
    protocol="WB0",
    ids=wb0_only_val
)

boundary_metrics = compute_boundary_metrics_if_nonempty()

coverage = compute_coverage_effects_without_test_labels()

conclusion = build_sensitivity_conclusion(
    primary="WB0",
    sensitivity="WB1",
    common_metrics=metrics_common,
    native_metrics=metrics_native,
    coverage=coverage
)

if critical_protocol_issue:
    conclusion.protocol_amendment_required = True
    approved_for_phase42 = False
else:
    approved_for_phase42 = True

write_all_artifacts()
write_phase42_reference_update()
write_signoff()
```

---

# 160. Definition of Done

\[
\boxed{
One\ Fixed\ S1\text{-}S18\ Configuration
+
Two\ Boundary\ Protocols
+
Causal\ WB0
+
Strict\ WB1
+
Native\ Population\ Audit
+
Common\ Population\ Audit
+
One\ Reused\ WB0
+
One\ Fresh\ WB1
+
Verified\ Native\ BEST
+
Common\ Validation\ Comparison
+
Coverage\ Sensitivity
+
No\ Test\ Labels
+
Phase42\ Handoff
}
\]

---

# 161. Final status contract

```text
PHASE 41 checks boundary protocol sensitivity only.

WB0:
target assigned by target timestamp
past context may cross from previous split
all input strictly before target
primary deployment protocol.

WB1:
all input rows and target must belong to same split
strict isolation
no padding
no cross-split history.

Frozen:
all S1-S18 selected configuration
same split boundaries
same L*
same H1
same scalers
same features
same model
same loss
same optimizer
same LR/WD/batch/dropout
same epoch cap/patience
same clipping
same RevIN
same seed42.

Expected:
WB1 Val/Test IDs subset of WB0 IDs.

Comparison:
Native populations
+
Common target population.

Checkpoint rule:
each protocol selects BEST on its own native Validation RMSE.
Then freeze checkpoints.
Then compare both on common Validation IDs.
No checkpoint reselection.

WB0:
reuse S18 selected reference.

WB1:
one fresh seed42 run.

Test:
eligibility metadata only.
No Test labels.
No Test predictions.
No Test metrics.

Decision:
WB0 remains primary protocol unless formal protocol amendment is required.
WB1 is sensitivity evidence.

After SWEEP_S19_BOUNDARY-v1 PASS:
proceed to
PHASE 42 — Candidate synthesis.
```

---

# 162. Final check

Correct:

```text
Load S18 selected model
→ freeze everything
→ build WB0/WB1 target populations
→ assert WB1 subset
→ build common Validation population
→ reuse WB0
→ fresh WB1
→ verify WB1 native BEST
→ freeze both checkpoints
→ evaluate both on common Val IDs
→ quantify boundary coverage loss
→ preserve Test firewall
→ write sensitivity conclusion
→ Phase42
```

Incorrect:

```text
WB1 short-window padding
→ refit scaler
→ compare native RMSE without sample counts
→ reselect checkpoint on common subset
→ inspect Test metrics
→ automatically replace WB0 because WB1 RMSE lower
```

Chỉ sau khi `SWEEP_S19_BOUNDARY-v1` được sign-off và:

```text
approved_for_phase42 = true
```

mới chuyển sang **PHASE 42 — Candidate synthesis**.
