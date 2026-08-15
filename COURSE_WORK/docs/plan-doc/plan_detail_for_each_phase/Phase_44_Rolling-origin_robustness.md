# PHASE 44 — ROLLING-ORIGIN ROBUSTNESS

## Kế hoạch đánh giá temporal robustness cho Transformer shortlist, tuned LSTM và Persistence trước Final Model Lock

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Upstream Transformer shortlist:** `CANDIDATE_SYNTHESIS-v1`  
**Upstream LSTM:** `LSTM_TUNING-v1`  
**Phase ID:** `PHASE_44_ROLLING_ORIGIN_ROBUSTNESS`  
**Output version:** `ROLLING_ORIGIN-v1`  
**Phase trước:** `Phase_43_LSTM_tuning.md`  
**Phase sau:** `Phase_45_Final_model_lock.md`

---

# 1. Vai trò của Phase 44

Phase 44 kiểm tra liệu các model/configuration đã được chọn ở một Validation period cố định có duy trì chất lượng khi forecast origin dịch chuyển theo thời gian hay không.

Đây là bước temporal robustness quan trọng trước khi khóa final Transformer configuration.

Phase44 không mở thêm hyperparameter search.

Nó chỉ đánh giá:

```text
Frozen Transformer shortlist từ Phase42
+
Frozen LSTM_TUNED từ Phase43
+
Persistence baseline
```

trên một bộ rolling-origin folds được predefine trước khi chạy.

Mục tiêu chính:

```text
1. Kiểm tra temporal stability của Transformer candidates.
2. Chọn Transformer candidate có robustness tốt nhất.
3. So sánh Transformer với tuned LSTM và Persistence trên cùng outer-fold targets.
4. Tách epoch selection khỏi outer-fold evaluation.
5. Không sử dụng Test.
6. Tạo handoff đủ mạnh cho Phase45 Final Model Lock.
```

Nguyên tắc:

\[
\boxed{
Frozen\ Candidates
+
Expanding\ Origins
+
Nested\ Time\ Validation
+
Fold\text{-}Local\ Scaling
+
Same\ Outer\ Targets
+
Pooled\ Out\text{-}of\text{-}Time\ RMSE
+
No\ Test
}
\]

---

# 2. Đây là robustness evaluation, không phải tuning phase mới

Không được thêm:

```text
new Transformer candidate
new LSTM candidate
new hyperparameter value
new feature set
new lookback
new loss
new clipping threshold
new RevIN option
```

sau khi Phase44 bắt đầu.

Candidate universe phải được frozen từ upstream.

---

# 3. Inputs bắt buộc từ Phase42

Load:

```text
phase_42_signoff.json
transformer_candidate_shortlist.json
phase44_rolling_origin_handoff.json
boundary_sensitivity_context.json
baseline_anchor_context.json
```

Hard:

```text
transformer_shortlist_frozen = true.
```

---

# 4. Inputs bắt buộc từ Phase43

Load:

```text
phase_43_signoff.json
lstm_tuned_winner.json
phase44_rolling_origin_lstm_handoff.json
```

Hard:

```text
phase44_handoff_ready = true.
```

---

# 5. Model families được đánh giá

## 5.1 Transformer candidates

Maximum:

```text
TR_C0_PRIMARY
TR_C1_LOCAL_ALT
TR_C2_LOCAL_ALT
```

hoặc ít hơn nếu Phase42 shortlist nhỏ hơn.

## 5.2 LSTM

Exactly:

```text
LSTM_TUNED
```

from Phase43.

## 5.3 Persistence

Exactly:

```text
PERSISTENCE_LAST_VALUE
```

with:

\[
\hat y_{t+1}=y_t.
\]

---

# 6. Mục tiêu selection của Phase44

Phase44 có hai nhiệm vụ khác nhau:

```text
A. Transformer candidate selection
B. Model-family robustness comparison
```

Primary selection:

```text
chọn Transformer candidate cho Phase45
```

không thay assignment main model bằng LSTM chỉ vì LSTM có score tốt hơn.

LSTM/Persistence là baselines để contextualize robustness.

---

# 7. Primary robustness score

Transformer candidate được xếp hạng bằng:

```text
POOLED OUTER-FOLD RMSE Wh
```

trên toàn bộ outer-fold predictions.

Nếu outer folds có prediction bundles:

\[
\mathcal{P}
=
\bigcup_{k=1}^{K}
\mathcal{P}_k
\]

với disjoint outer target IDs, thì:

\[
RMSE_{pooled}
=
\sqrt{
\frac{
\sum_{i\in\mathcal{P}}
(y_i-\hat y_i)^2
}{
|\mathcal{P}|
}
}
\]

Đây là primary Phase44 Transformer ranking metric.

---

# 8. Secondary robustness metrics

Record:

```text
Pooled MAE Wh
Pooled R²
Macro mean fold RMSE
Fold RMSE standard deviation
Worst-fold RMSE
Best-fold RMSE
Fold-win count
Pairwise fold deltas
Pooled delta vs Persistence
Pooled delta vs LSTM
```

Không metric nào override pooled outer RMSE trừ exact-tie logic đã predeclare.

---

# 9. Vì sao dùng pooled outer RMSE

Pooled score:

```text
weights every forecast target equally
```

và tránh việc một fold nhỏ có weight bằng một fold lớn chỉ vì macro averaging.

Macro fold statistics vẫn cần để đánh giá stability.

---

# 10. Exact tie rule cho Transformer candidates

Nếu hai Transformer candidates có exactly equal full-precision pooled outer RMSE:

```text
1. lower worst-fold RMSE wins
2. if exact tie, lower fold-RMSE standard deviation wins
3. if exact tie, earlier Phase42 shortlist position wins
```

Tức:

```text
TR_C0 trước TR_C1 trước TR_C2
```

ở final exact tie.

Không rounding trước tie check.

---

# 11. Không đặt arbitrary robustness threshold

Không tự tạo:

```text
1%
5 Wh
2 folds out of 3
```

làm threshold bắt buộc sau khi xem kết quả.

Report exact effect sizes.

---

# 12. Number of rolling folds

Lock:

```text
K = 3
```

under:

```text
RO3_EXPANDING_PRETEST-v1.
```

Ba folds này dùng dữ liệu trước Test.

---

# 13. Không sử dụng Test để tạo rolling folds

Rolling-origin robustness chỉ sử dụng:

```text
original Train
+
original Validation
```

Không sử dụng target values trong original Test.

Test vẫn dành cho Phase47.

---

# 14. Base pre-Test regions

From `SPLIT-v1`:

```text
TRAIN_REGION
VALIDATION_REGION
TEST_REGION
```

Phase44 builds:

```text
robustness initial history = original Train
robustness rolling evaluation region = original Validation
```

Test remains untouched.

---

# 15. Base target population

Use the existing registered:

```text
WINDOWPOP-v1
```

as base target universe.

This preserves common target eligibility across registered lookbacks.

For Phase44 additionally intersect with all frozen candidate requirements if needed.

Define:

```text
ROBASE-v1
```

as the ordered set of target IDs that:

```text
belong to Train or Validation
are valid under temporal integrity
are valid under WINDOWPOP-v1
are usable by every frozen learned candidate
```

Persistence is restricted to these IDs for fairness.

---

# 16. Why common target population is mandatory

Transformer alternatives may differ in:

```text
lookback
feature set
target scaling
architecture
training choices.
```

LSTM uses the shared selected data configuration.

Phase44 must compare predictions on exactly the same outer target timestamps.

Hard:

```text
outer target IDs identical across
all Transformer candidates
LSTM_TUNED
Persistence.
```

---

# 17. Original Validation partition

Take ordered target IDs:

```text
RVAL_IDS
=
ROBASE-v1 target IDs whose target timestamp lies in original Validation.
```

Partition them chronologically into exactly:

```text
V1
V2
V3
```

using deterministic balanced contiguous split semantics equivalent to:

```text
array_split(RVAL_IDS, 3)
```

Properties:

```text
chronological
non-overlapping
union = RVAL_IDS
sizes differ by at most 1 target ID
no randomization.
```

---

# 18. Fold definitions

## Fold RO1

```text
Outer history:
original Train

Outer evaluation:
V1
```

## Fold RO2

```text
Outer history:
original Train + V1

Outer evaluation:
V2
```

## Fold RO3

```text
Outer history:
original Train + V1 + V2

Outer evaluation:
V3
```

This is expanding-origin evaluation.

---

# 19. Fold origin timestamp

For each fold:

```text
origin_k
=
timestamp of first outer evaluation target.
```

Everything used for fold training/scaler fitting must be strictly before:

```text
origin_k.
```

except deterministic known calendar transformations.

---

# 20. No model update inside outer evaluation block

For each fold:

```text
train/refit once before origin
freeze model
forecast all outer targets
```

No:

```text
gradient update
optimizer step
scaler refit
model fine-tuning
```

after outer block begins.

---

# 21. One-step rolling context inside outer block is allowed

Because project forecasting contract is one-step rolling prediction, later outer targets may use actual observations from earlier outer timestamps as historical input.

For target:

```text
Appliances_{t+1}
```

input may contain:

```text
actual observed Appliances up to t
```

including observations that occurred earlier in the same outer evaluation block.

This is consistent with WB0.

---

# 22. This is not recursive multi-step forecasting

The model does not recursively feed its own predictions as future historical targets.

Hard:

```text
historical target input = actual observed past value
not previous model prediction.
```

---

# 23. Boundary protocol remains WB0

All Phase44 model candidates use:

```text
WB0
```

as primary protocol carried from Phase41/42.

WB1 remains separate sensitivity evidence.

Do not include WB1 as another rolling candidate.

---

# 24. Why nested epoch selection is required

If outer evaluation block were also used for:

```text
early stopping
BEST checkpoint selection
```

then the outer robustness score would be optimistically selected on itself.

Phase44 therefore separates:

```text
Inner validation
```

from:

```text
Outer evaluation.
```

---

# 25. Two-stage training inside each fold

For every learned model candidate and fold:

```text
STAGE A — Inner epoch selection
STAGE B — Full-history refit
STAGE C — Outer evaluation
```

Persistence skips A/B.

---

# 26. Inner validation construction

For each fold, define inner validation as the immediately preceding block with approximately the same target count as the outer block.

## RO1

```text
Outer eval = V1

Inner validation:
last |V1| eligible target IDs from original Train

Inner train:
all earlier original Train eligible IDs.
```

## RO2

```text
Outer eval = V2
Inner validation = V1
Inner train = original Train
```

## RO3

```text
Outer eval = V3
Inner validation = V2
Inner train = original Train + V1
```

---

# 27. RO1 inner validation size

Use:

```text
n_inner_val_RO1 = len(V1)
```

and select the last exactly `n_inner_val_RO1` eligible `ROBASE-v1` target IDs from original Train.

Hard assertions:

```text
inner_train non-empty
inner_val non-empty
all inner_train targets < all inner_val targets
all inner_val targets < outer origin.
```

---

# 28. Why inner validation is immediately preceding history

This provides:

```text
temporal proximity
```

while keeping outer evaluation untouched during checkpoint/epoch selection.

No random inner split.

---

# 29. Stage A — inner training

For candidate `m` and fold `k`:

```text
fit fold-inner scalers on inner-train history only
build inner-train windows
build inner-validation windows
fresh model
fresh optimizer
train with candidate max_epochs/patience
select BEST by inner Validation RMSE Wh.
```

Output:

```text
best_epoch_inner(m,k).
```

---

# 30. Stage A uses candidate’s frozen training config

Transformer candidate fields may differ:

```text
batch
LR
WD
dropout
loss
epoch cap
clipping
RevIN
architecture.
```

Use exact Phase42 candidate specification.

LSTM uses exact Phase43 tuned specification.

Do not homogenize optimizer hyperparameters across model candidates.

---

# 31. Stage A early stopping

Use:

```text
candidate-specific max_epochs
patience from candidate
min_delta from candidate
monitor = inner_validation_rmse_wh
```

Transformer inherited settings exactly.

LSTM uses:

```text
MSE
E50
P10
GC1
```

from Phase43 winner.

---

# 32. Stage A outer block firewall

No outer evaluation target value may be:

```text
read for scaler fitting
used in loss
used in early stopping
used in checkpoint selection.
```

Outer target IDs may be known as metadata to define block membership, but target values remain evaluation-only until Stage C.

---

# 33. Stage B — full-history refit

After Stage A locks:

```text
best_epoch_inner
```

discard Stage A model for outer scoring.

Then:

```text
fit fresh fold-outer-history scalers
fresh model from seed42
fresh optimizer
train on ALL eligible outer-history training targets
for exactly best_epoch_inner complete epochs
no early stopping
no outer validation access
save final epoch model.
```

---

# 34. Why refit on full outer history

This lets the model use all observations available before forecast origin while keeping outer evaluation labels untouched.

The inner validation block is reincorporated into training only after:

```text
epoch count
```

has been selected.

---

# 35. Stage B checkpoint semantics

The official outer-fold model is:

```text
REFIT_FINAL at epoch = best_epoch_inner.
```

Not:

```text
BEST on outer evaluation.
```

No outer checkpoint search.

---

# 36. Stage B scaler fitting

Scalers are refit using all data available before fold origin:

```text
outer history only.
```

No outer-evaluation rows.

---

# 37. Stage A scaler fitting

Scalers use:

```text
inner training history only.
```

No inner-validation rows.

This prevents preprocessing leakage into inner epoch selection.

---

# 38. Fold-local scaling contract

Phase44 introduces:

```text
RO_SCALING-v1
```

which preserves Phase9 semantics but fits per fold/stage.

For each candidate/fold/stage:

```text
X scaler fit only on allowed training-history rows
Y scaler fit only on allowed training target/history period
cyclical time features pass through
binary time features pass through
no future fold data.
```

---

# 39. Candidate-specific X scalers

If Transformer candidates use different feature variants:

```text
each candidate receives its correct fold-local X scaler bundle.
```

Do not force one feature scaler across different feature sets.

---

# 40. Candidate-specific Y scaling

If a candidate uses:

```text
YS0
```

then identity.

If:

```text
YS1
```

fit fold-local target StandardScaler on allowed training history only.

---

# 41. RevIN inside rolling folds

If candidate has:

```text
RN1
```

keep exact Phase40 config.

RevIN statistics are:

```text
per input window
only from observed historical rows
```

after fold-local global scaling.

No fold-level RevIN fitting beyond learned affine parameters.

---

# 42. RevIN target bridge

If RN1 active:

```text
fold-local X-target scaler
→ raw Wh
→ fold-local Y transform
```

must be used by the RN1 target coordinate bridge.

Hard audit per fold.

---

# 43. LSTM scaling

LSTM uses the same fold-local scaling semantics for its frozen:

```text
FV*
YS*
L*
```

configuration.

No LSTM-specific data leakage.

---

# 44. Persistence baseline

Persistence requires no scaling/training.

For every outer target ID:

\[
\hat y_{j}=y_{j-1}
\]

under H1 semantics.

Use actual prior observed Appliances.

Restrict to exact common outer target IDs.

---

# 45. No persistence inner stage

Persistence does not:

```text
early stop
fit scaler
train.
```

It is evaluated directly at Stage C.

---

# 46. Common outer population per fold

Define:

```text
RO_OUTER_IDS_k
```

as exact frozen target IDs of `Vk`.

Before model runs, assert every frozen candidate can construct valid windows for all these IDs.

If not:

```text
STOP
```

and investigate candidate/population compatibility.

Do not silently remove candidate-specific targets after training.

---

# 47. Common inner population

For each fold define:

```text
RO_INNER_TRAIN_IDS_k
RO_INNER_VAL_IDS_k
RO_OUTER_TRAIN_IDS_k
```

from `ROBASE-v1`.

All learned candidates use same target IDs for each corresponding role.

Candidate lookback differences are handled because ROBASE is compatibility-filtered.

---

# 48. Common outer y_true

For each fold:

```text
y_true_wh
```

must be identical across all candidate prediction bundles.

Hard assertion.

---

# 49. Candidate feature availability

All dynamic observed signals used by a candidate must be available at each input timestamp.

No future exogenous values beyond the target observation cutoff.

This follows original task’s historical multivariate input semantics.

---

# 50. Candidate shortlist immutability

Phase44 must verify Phase42 shortlist checksum before execution.

After first run:

```text
no candidate may be added/removed/replaced.
```

---

# 51. LSTM immutability

Verify Phase43 tuned winner checksum/config.

No LSTM retuning.

---

# 52. No candidate rescue after fold result

If candidate performs poorly on RO1:

```text
continue remaining folds
```

unless genuine technical/numerical failure blocks execution.

Do not replace it.

---

# 53. No candidate pruning mid-robustness

All valid frozen candidates must complete all 3 folds.

---

# 54. Candidate failure policy

If a learned candidate has genuine numerical failure on any fold:

```text
Phase44 is incomplete for ranking
```

until failure is resolved according to protocol.

Do not rank only surviving candidates silently.

---

# 55. Technical rerun policy

Allowed only for:

```text
process interruption
hardware failure
artifact corruption
non-scientific infrastructure issue.
```

Must retain failed run record and link rerun.

---

# 56. Score-based rerun forbidden

No rerun because:

```text
fold RMSE looks bad
candidate lost
training curve looks unusual.
```

---

# 57. Seeds

Phase44 uses:

```text
seed = 42
```

for every learned candidate/fold Stage A and Stage B.

Purpose:

```text
isolate temporal-origin robustness
```

rather than seed robustness.

Multi-seed evaluation belongs later.

---

# 58. Stage A/B seed semantics

Both Stage A and Stage B start fresh with:

```text
seed42.
```

Stage B is not initialized from Stage A.

---

# 59. No RNG state carry from candidate to candidate

Each run:

```text
reset/reseed
fresh loaders
fresh model.
```

---

# 60. DataLoader behavior

Inner/refit Train:

```text
shuffle=true
drop_last=false
candidate batch size
same worker-seeding contract.
```

Inner validation/outer evaluation:

```text
shuffle=false
drop_last=false.
```

---

# 61. Stage B has no validation loader for selection

Optional training-loss monitoring is allowed.

No outer eval iteration during Stage B training.

---

# 62. Outer evaluation happens exactly once per refit model

Preferred:

```text
one official Stage C inference
```

after refit final checkpoint fixed.

If rerun for artifact verification, predictions must match within tolerance and must not influence model state.

---

# 63. Outer prediction bundle

For each learned model/fold:

```text
fold_id
candidate_id
target_id
target_timestamp
y_true_wh
y_pred_wh
residual_wh
model_run_id
refit_epoch
```

Residual convention:

```text
residual = y_true - y_pred.
```

---

# 64. Persistence prediction bundle

Same schema where applicable:

```text
model_run_id = N/A
refit_epoch = N/A.
```

---

# 65. No Test row in prediction bundles

Hard.

---

# 66. Fold-level metrics

For each model/fold:

```text
MAE Wh
RMSE Wh
R²
N
```

computed globally from prediction bundle.

No average batch RMSE.

---

# 67. Pooled metrics

Concatenate all 3 disjoint outer-fold prediction bundles per model.

Then compute:

```text
pooled MAE
pooled RMSE
pooled R²
```

with `METRICS-v1`.

---

# 68. Macro fold metrics

Compute:

```text
mean fold RMSE
sample standard deviation of fold RMSE (ddof=1)
min fold RMSE
max fold RMSE
mean fold MAE
```

K=3.

---

# 69. Fold ranking

For each fold, rank:

```text
Transformer candidates
LSTM
Persistence
```

by RMSE.

Also create Transformer-only rank.

---

# 70. Fold-win count

For each Transformer candidate:

```text
number of folds with lowest Transformer RMSE.
```

Secondary only.

---

# 71. Worst-fold robustness

Record:

```text
max(RMSE_fold1, RMSE_fold2, RMSE_fold3).
```

Used only in exact pooled-RMSE tie.

---

# 72. Variability

Use:

```text
sample SD of fold RMSE
```

not population SD.

With K=3, this is descriptive only.

No strong statistical inference.

---

# 73. No t-test over 3 folds

Folds are:

```text
temporally ordered
not independent random samples
K small.
```

Do not report conventional independent t-test significance.

---

# 74. No unsupported Diebold-Mariano claim

A formal forecast test requires additional assumptions and dependence handling.

Not needed for coursework robustness phase.

Use descriptive paired error deltas instead.

---

# 75. Pairwise common-target comparison

For every pair of models on pooled outer IDs:

```text
same target IDs
paired absolute errors
paired squared errors
```

may be summarized.

Do not use Test.

---

# 76. Transformer candidate primary ranking

Create:

```text
transformer_robustness_ranking.csv
```

ordered by:

```text
1. pooled_outer_rmse_wh ascending
2. exact tie → worst_fold_rmse ascending
3. exact tie → fold_rmse_sd ascending
4. exact tie → Phase42 shortlist position ascending.
```

---

# 77. Recommended Transformer candidate

Top row becomes:

```text
RO_RECOMMENDED_TRANSFORMER
```

for Phase45 consideration.

This is not yet final model lock.

---

# 78. LSTM/Persistence do not enter Transformer candidate tie-break

They are comparison anchors only.

---

# 79. Model-family robustness comparison

Create a separate table:

```text
TR recommended
LSTM_TUNED
Persistence
```

with:

```text
pooled metrics
macro/worst-fold metrics
fold wins
```

This supports final conclusions later.

---

# 80. If LSTM beats all Transformers

Report clearly.

Do not change assignment’s primary Transformer requirement automatically.

Safe conclusion:

```text
The tuned LSTM showed stronger rolling-origin predictive performance under this robustness protocol, while the Transformer remains the primary model family required by the coursework and proceeds to final model lock using the best Transformer candidate.
```

---

# 81. If Persistence beats learned models

Report clearly.

Do not suppress.

This is important evidence about strong short-horizon autocorrelation.

---

# 82. If Transformer beats Persistence/LSTM

Report conditionally:

```text
under pre-Test rolling-origin folds
```

not final Test superiority.

---

# 83. Original Validation reuse limitation

S1–S18 candidate selection already used original Validation.

Phase44 outer folds partition the pre-Test Validation region.

Therefore rolling-origin results are:

```text
robustness/model-selection evidence
```

not a completely independent unbiased generalization estimate.

Final unbiased evaluation remains Phase47 Test.

This limitation must be stated.

---

# 84. Nested epoch selection reduces but does not erase selection reuse

Inner/outer separation prevents:

```text
same outer block from selecting epoch and scoring itself.
```

But candidate configurations were already chosen using historical Validation evidence.

State this clearly.

---

# 85. Fold-local scaler rationale

Reusing original global Train scaler in RO2/RO3 would not use future data, but would fail to exploit newly available training history and make rolling refit less faithful.

Using full pre-Test scaler would leak future fold information.

Therefore:

```text
fit scaler separately at each fold stage.
```

---

# 86. No scaler incremental update shortcuts

Correctness baseline:

```text
refit from scratch on allowed fold history.
```

No partial_fit unless separately verified exactly equivalent.

---

# 87. No model warm-start across origins

Even though origins expand:

```text
RO2 model does not load RO1 weights.
RO3 does not load RO2 weights.
```

Each fold trains from scratch.

This isolates configuration robustness from continual-learning effects.

---

# 88. No optimizer state carry across origins

No.

---

# 89. No scaler carry across origins

Refit per fold.

---

# 90. No online fine-tuning inside outer block

No.

---

# 91. Calendar features

Recompute deterministic time features using original timestamp rules.

These require no fit.

---

# 92. Feature-set candidates

If candidate differs by feature set:

```text
use exact candidate feature list/order
```

with fold-local X scaling for eligible continuous channels.

No hidden carry of primary feature set.

---

# 93. Target-scaling candidates

If candidate differs by YS:

```text
apply candidate-specific fold-local target transformation.
```

Evaluation always inverse to Wh.

---

# 94. Lookback candidates

If candidate differs by L:

```text
use exact candidate L.
```

All outer target IDs remain common due robustness population gate.

No padding.

---

# 95. Pooling/activation/architecture candidates

Use exact Phase42 specification.

No adaptation per fold.

---

# 96. Batch/LR/WD candidates

Use exact values.

No fold-specific retuning.

---

# 97. Dropout candidate

Use exact registered dropout.

---

# 98. Loss candidate

Use exact registered:

```text
MSE
or
Huber(delta=1.0 y_model-space)
```

candidate semantics.

---

# 99. Epoch-cap candidate

Stage A uses candidate max cap.

Stage B refits for:

```text
inner best epoch
```

which must satisfy:

```text
1 <= best_epoch <= candidate max_epochs.
```

---

# 100. Gradient clipping candidate

Use exact GC0/GC1 semantics.

No fold-specific changes.

---

# 101. RevIN candidate

Use exact RN0/RN1 semantics and fold-local target bridge.

---

# 102. LSTM exact config

Use frozen Phase43:

```text
hidden*
layers*
dropout*
B*
LR*
WD*
MSE
E50
P10
GC1
RevIN OFF
WB0.
```

---

# 103. Persistence exact semantics

Latest observed raw Appliances value before each target.

No model-space calculation required.

---

# 104. Fold definitions must be materialized before any model run

Create:

```text
rolling_origin_fold_manifest.json
```

and checksum it.

No changing fold boundaries after seeing scores.

---

# 105. Fold manifest minimum fields

For each fold:

```text
fold_id
inner_train_target_ids_fingerprint
inner_val_target_ids_fingerprint
outer_train_target_ids_fingerprint
outer_eval_target_ids_fingerprint
inner_train_first/last timestamp
inner_val_first/last timestamp
outer_train_first/last timestamp
outer_eval_first/last timestamp
counts
origin_timestamp
```

No Test values.

---

# 106. Temporal ordering assertions

For every fold:

```text
max(inner_train_timestamp)
<
min(inner_val_timestamp)

max(inner_val_timestamp)
<
min(outer_eval_timestamp)

max(outer_train_timestamp)
<
min(outer_eval_timestamp).
```

RO2/RO3 outer train includes prior evaluation blocks but never current/future block.

---

# 107. Fold disjointness

Outer eval sets:

```text
V1 ∩ V2 = empty
V1 ∩ V3 = empty
V2 ∩ V3 = empty
```

and:

```text
V1 ∪ V2 ∪ V3 = RVAL_IDS.
```

Hard.

---

# 108. Pooled OOF semantics

Because outer eval blocks are disjoint, each target appears exactly once in pooled rolling-origin predictions per model.

Hard:

```text
duplicate target ID count = 0.
```

---

# 109. Training-history expansion assertions

Expected:

```text
RO1 outer train ⊂ RO2 outer train ⊂ RO3 outer train
```

by target IDs.

Hard.

---

# 110. No future block contamination

For RO1:

```text
V2/V3 unavailable
```

during all Stage A/B operations.

For RO2:

```text
V3 unavailable.
```

---

# 111. Inner stage target populations

RO2 inner val = V1.

RO3 inner val = V2.

These labels are known because they occur before current outer origin, so using them for epoch selection is temporally valid.

---

# 112. Outer evaluation target labels

Current fold outer labels are hidden from all fitting/selection steps until Stage C.

---

# 113. Experiment registry families

Recommended:

```text
ROLLING_ORIGIN_INNER_SELECTION
ROLLING_ORIGIN_REFIT
ROLLING_ORIGIN_PERSISTENCE
```

Tags include:

```text
candidate_id
fold_id
stage=A/B/C
```

---

# 114. Inner-stage run ID

Recommended identity:

```text
RO_<fold>_<candidate>_INNER
```

---

# 115. Refit run ID

Recommended:

```text
RO_<fold>_<candidate>_REFIT
```

---

# 116. Persistence evaluation ID

Recommended:

```text
RO_<fold>_PERSISTENCE
```

not a training run.

---

# 117. Stage A artifacts

Per learned candidate/fold:

```text
inner config
inner scaler checksums
learning history
BEST checkpoint
best epoch
BEST inner Validation metrics
sample-order fingerprint
gradient diagnostics
status.
```

---

# 118. Stage B artifacts

Per learned candidate/fold:

```text
outer-history scaler checksums
fresh initialization fingerprint
exact refit epoch count
training history
REFIT_FINAL checkpoint
no-validation-selection flag
status.
```

---

# 119. Stage C artifacts

Per model/fold:

```text
outer prediction bundle
outer metrics
population fingerprint.
```

---

# 120. No Stage A checkpoint used for outer official score

The Stage A checkpoint is only for selecting epoch count.

Official outer score uses Stage B refit model.

---

# 121. No BEST notion inside Stage B

Final epoch is official by construction.

Do not secretly pick lowest training loss epoch.

---

# 122. Stage B exact epoch audit

Hard:

```text
epochs_completed == best_epoch_inner.
```

No early stop.

---

# 123. Candidate/fold comparability

For each outer fold:

```text
same exact outer target IDs
same raw y_true
same METRICS-v1
same residual convention.
```

---

# 124. Model-specific training cost is allowed to differ

Frozen configs may have different:

```text
batch
architecture
epoch count selected
runtime.
```

This is part of their practical robustness profile.

Do not equalize optimizer steps artificially.

---

# 125. Runtime is secondary

Record:

```text
inner runtime
refit runtime
total fold runtime
time per epoch
peak memory optional.
```

Do not select by runtime unless exact metric tie has already passed defined tie-breaks; runtime still does not enter tie rule.

---

# 126. Learning-curve diagnostics

Stage A only:

```text
Train criterion
inner Validation RMSE
best epoch
stop epoch.
```

Stage B:

```text
Train criterion only
```

because no validation selection.

---

# 127. Gradient diagnostics

For learned models:

```text
preclip norm
clip fraction if applicable
nonfinite events.
```

Diagnostic only.

---

# 128. Pooled prediction order

Sort:

```text
fold order RO1→RO3
then target timestamp ascending.
```

Store target IDs.

---

# 129. Pooled metric verification

Recompute from pooled prediction bundle.

Do not average fold RMSE to claim pooled RMSE.

---

# 130. Macro metric verification

Compute separately from fold metric table.

---

# 131. Pairwise delta formulas

For model A relative to B:

\[
\Delta RMSE_{B\rightarrow A}
=
RMSE_B-RMSE_A.
\]

Positive:

```text
A is better.
```

Use consistent naming.

---

# 132. Per-fold baseline deltas

For each learned model:

```text
Persistence RMSE - model RMSE
LSTM RMSE - Transformer RMSE
```

where relevant.

Positive means the named model improves over comparator.

---

# 133. Pooled baseline deltas

Same on pooled prediction bundles.

---

# 134. Fold robustness table

Required:

```text
model
RO1_RMSE
RO2_RMSE
RO3_RMSE
pooled_RMSE
mean_fold_RMSE
sd_fold_RMSE
worst_fold_RMSE
fold_wins
```

---

# 135. Candidate ranking must only use complete candidates

A candidate must have:

```text
3/3 valid outer folds
```

before ranking.

---

# 136. No partial-score extrapolation

Do not average 2 folds and compare with 3-fold candidates.

---

# 137. No fold weighting by subjective difficulty

Pooled score already target-weights observations.

No manual weights.

---

# 138. No removal of “bad” fold

All three folds retained.

---

# 139. No outlier-fold deletion

No.

---

# 140. No error trimming

No.

---

# 141. No target clipping

No.

---

# 142. No metric change

Primary remains RMSE Wh.

---

# 143. Robustness findings categories

Recommended:

```text
TRANSFORMER_PRIMARY_REMAINS_BEST
TRANSFORMER_LOCAL_ALT_OVERTAKES_PRIMARY
TRANSFORMER_RANK_UNSTABLE_ACROSS_FOLDS
TRANSFORMER_RANK_STABLE
LSTM_BEATS_RECOMMENDED_TRANSFORMER_POOLED
TRANSFORMER_BEATS_LSTM_POOLED
PERSISTENCE_BEATS_LEARNED_MODELS
LEARNED_MODELS_BEAT_PERSISTENCE
HIGH_FOLD_VARIABILITY
LOW_FOLD_VARIABILITY
FOLD_SPECIFIC_FAILURE
INNER_EPOCH_VARIABILITY
SCALER_DRIFT_CONTEXT
WB1_SENSITIVITY_CARRIED
TEST_FIREWALL_PRESERVED
```

Do not attach arbitrary “high/low” thresholds unless purely descriptive text; prefer exact SD/range.

---

# 144. Temporal robustness interpretation

A candidate is more robust descriptively if it shows:

```text
low pooled RMSE
reasonable worst-fold RMSE
consistent fold ranking
low fold-to-fold variability
```

But official Transformer recommendation remains defined by primary pooled RMSE + exact tie rules.

---

# 145. No statistical significance claim from three folds

Avoid:

```text
significantly better
statistically superior
```

unless a separately valid statistical analysis is later performed.

Use:

```text
lower pooled RMSE
more consistent across folds
```

instead.

---

# 146. Phase44 preflight

Before any run verify:

```text
Phase42 shortlist frozen
Phase43 tuned LSTM frozen
Phase41 primary protocol WB0
no protocol amendment pending
SPLIT-v1 unchanged
WINDOWPOP-v1 available
Validation target IDs available
K=3 fold definition frozen
Test firewall active
candidate compatibility with ROBASE
Experiment Registry ready.
```

---

# 147. Candidate compatibility preflight

For every Transformer candidate:

```text
config fingerprint valid
feature registry valid
lookback supported
d_model/head valid
loss semantics valid
RevIN applicability valid
boundary WB0
Training Engine supported.
```

For LSTM:

```text
Phase43 winner valid.
```

---

# 148. Scaling preflight

Verify fold-local scaler constructors reproduce Phase9 semantics.

Unit tests:

```text
no eval row used in fit
transform/inverse transform finite
YS1 inverse roundtrip
RN1 X→Y bridge correct
TF pass-through correct.
```

---

# 149. Fold-population preflight

Verify:

```text
RVAL_IDS non-empty
V1/V2/V3 non-empty
RO1 inner train non-empty
all outer targets valid for all candidates
outer sets disjoint
union equals RVAL_IDS.
```

---

# 150. Stage A/B leakage tests

For sampled/complete metadata:

```text
max scaler-fit timestamp < inner-val start
max inner-training target < inner-val start
max Stage A available timestamp < outer origin
max Stage B scaler-fit timestamp < outer origin
max Stage B training target < outer origin
```

---

# 151. Outer evaluation no-update audit

During Stage C:

```text
model.eval()
inference mode
optimizer absent/not stepped
no backward
no scaler fit
```

Hard.

---

# 152. Required artifact directory

```text
artifacts/
└── rolling_origin/
    ├── rolling_origin_manifest.json
    ├── rolling_origin_contract.json
    ├── phase44_preflight_audit.csv
    ├── rolling_origin_fold_manifest.json
    ├── rolling_origin_fold_table.csv
    ├── rolling_origin_population_audit.csv
    ├── rolling_origin_candidate_matrix.csv
    ├── rolling_origin_candidate_compatibility_audit.csv
    ├── rolling_origin_scaling_contract.json
    ├── rolling_origin_scaler_fit_audit.csv
    ├── rolling_origin_temporal_leakage_tests.csv
    ├── rolling_origin_common_target_audit.csv
    ├── rolling_origin_inner_selection_runs.csv
    ├── rolling_origin_inner_best_epochs.csv
    ├── rolling_origin_refit_runs.csv
    ├── rolling_origin_refit_epoch_audit.csv
    ├── rolling_origin_initialization_audit.csv
    ├── rolling_origin_sample_order_audit.csv
    ├── rolling_origin_gradient_diagnostics.csv
    ├── rolling_origin_runtime_diagnostics.csv
    ├── rolling_origin_fold_metrics.csv
    ├── rolling_origin_pooled_metrics.csv
    ├── rolling_origin_macro_metrics.csv
    ├── rolling_origin_pairwise_effects.csv
    ├── rolling_origin_fold_ranks.csv
    ├── transformer_robustness_ranking.csv
    ├── model_family_robustness_comparison.csv
    ├── rolling_origin_robustness_findings.csv
    ├── rolling_origin_recommended_transformer.json
    ├── phase45_final_model_lock_handoff.json
    ├── rolling_origin_tests.csv
    ├── rolling_origin_discrepancies.json
    ├── rolling_origin_summary.json
    ├── rolling_origin_report.md
    ├── predictions/
    │   ├── RO1_<candidate>_outer_predictions.csv
    │   ├── RO2_<candidate>_outer_predictions.csv
    │   ├── RO3_<candidate>_outer_predictions.csv
    │   ├── RO1_LSTM_TUNED_outer_predictions.csv
    │   ├── RO2_LSTM_TUNED_outer_predictions.csv
    │   ├── RO3_LSTM_TUNED_outer_predictions.csv
    │   ├── RO1_PERSISTENCE_outer_predictions.csv
    │   ├── RO2_PERSISTENCE_outer_predictions.csv
    │   ├── RO3_PERSISTENCE_outer_predictions.csv
    │   └── pooled_<model>_outer_predictions.csv
    ├── figures/
    │   ├── RO_44_01_fold_timeline.png
    │   ├── RO_44_02_fold_rmse_by_model.png
    │   ├── RO_44_03_pooled_rmse_comparison.png
    │   ├── RO_44_04_transformer_candidate_ranking.png
    │   ├── RO_44_05_fold_rmse_variability.png
    │   ├── RO_44_06_inner_best_epoch_by_fold.png
    │   ├── RO_44_07_pooled_prediction_scatter.png
    │   ├── RO_44_08_paired_error_deltas.png
    │   ├── RO_44_09_runtime_by_model_fold.png
    │   └── RO_44_10_baseline_delta_by_fold.png
    ├── README_ROLLING_ORIGIN_ROBUSTNESS.md
    └── phase_44_signoff.json
```

Per-run checkpoints/history remain under:

```text
artifacts/runs/<run_id>/
```

---

# 153. Required outputs

```text
O44.1  Manifest
O44.2  Contract
O44.3  Preflight audit
O44.4  Frozen fold manifest
O44.5  Fold table
O44.6  Population audit
O44.7  Candidate matrix
O44.8  Candidate compatibility audit
O44.9  Fold-local scaling contract
O44.10 Scaler-fit audit
O44.11 Temporal leakage tests
O44.12 Common-target audit
O44.13 Inner-selection run registry
O44.14 Inner best-epoch table
O44.15 Refit run registry
O44.16 Refit epoch audit
O44.17 Initialization audit
O44.18 Sample-order audit
O44.19 Gradient diagnostics
O44.20 Runtime diagnostics
O44.21 Per-fold outer predictions
O44.22 Pooled outer predictions
O44.23 Fold metrics
O44.24 Pooled metrics
O44.25 Macro robustness metrics
O44.26 Pairwise effects
O44.27 Fold ranks
O44.28 Transformer robustness ranking
O44.29 Model-family robustness comparison
O44.30 Findings
O44.31 Recommended Transformer
O44.32 Phase45 handoff
O44.33 Figures
O44.34 Tests
O44.35 Discrepancies
O44.36 Summary
O44.37 Human-readable report
O44.38 README
O44.39 Sign-off
```

---

# 154. Manifest schema

`rolling_origin_manifest.json`:

```text
version = ROLLING_ORIGIN-v1
phase = 44
fold_protocol = RO3_EXPANDING_PRETEST-v1
K = 3
initial_history = original Train
outer_eval_region = original Validation
test_region_used = false
nested_epoch_selection = true
full_history_refit = true
outer_eval_used_for_selection = false
primary_metric = pooled_outer_rmse_wh
seed = 42
transformer_shortlist_fingerprint
lstm_winner_fingerprint
persistence_version
boundary_protocol = WB0
population_base = WINDOWPOP-v1
fold_local_scaling = true
test_status = NOT_ACCESSED
status
created_at
```

---

# 155. Rolling-origin contract

`rolling_origin_contract.json` must state:

```text
Transformer shortlist frozen.
LSTM tuned config frozen.
Persistence fixed.

Three expanding origins.
Original Validation split partitioned V1/V2/V3.
No Test.

Each learned model/fold:
inner epoch selection
→ fresh full-history refit
→ one outer evaluation.

Outer block never selects epoch/checkpoint.

Scalers fit only on allowed training history.
Refit scalers fit on full pre-origin history.

WB0 rolling one-step context.
Actual prior observations allowed.
No recursive prediction feedback.
No online model updates.

All models use same outer target IDs.

Transformer recommendation by pooled outer RMSE.
```

---

# 156. Fold table schema

`rolling_origin_fold_table.csv`:

```text
fold_id
origin_timestamp
inner_train_count
inner_val_count
outer_train_count
outer_eval_count
inner_train_first_target
inner_train_last_target
inner_val_first_target
inner_val_last_target
outer_train_first_target
outer_train_last_target
outer_eval_first_target
outer_eval_last_target
inner_train_fingerprint
inner_val_fingerprint
outer_train_fingerprint
outer_eval_fingerprint
status
```

---

# 157. Population audit

`rolling_origin_population_audit.csv`:

```text
fold_id
role
target_count
target_ids_unique
chronological
candidate_coverage_count
all_candidates_supported
population_fingerprint
status
```

---

# 158. Candidate matrix

`rolling_origin_candidate_matrix.csv`:

```text
model_id
model_family
candidate_role
config_fingerprint
feature_variant
target_scaling
lookback
batch
loss
max_epochs
patience
clipping
revin
requires_training
source_phase
status
```

Persistence fields N/A where appropriate.

---

# 159. Candidate compatibility audit

Fields:

```text
model_id
fold_id
all_outer_ids_valid
all_inner_ids_valid
feature_config_valid
scaler_config_valid
lookback_valid
revin_valid
loss_valid
training_engine_valid
WB0_valid
status
```

---

# 160. Scaling contract artifact

`rolling_origin_scaling_contract.json`:

```text
version = RO_SCALING-v1
fit_stage_A_on = inner_train_history_only
fit_stage_B_on = outer_train_history_only
fit_stage_C = false
validation/eval_fit = forbidden
X_scaler_semantics = SCALING-v1
Y_scaler_semantics = SCALING-v1
time_features_passthrough
binary_features_passthrough
candidate_specific_feature_scalers = true
candidate_specific_target_scalers = true
RN1_bridge_fold_local = true
status
```

---

# 161. Scaler-fit audit

`rolling_origin_scaler_fit_audit.csv`:

```text
fold_id
candidate_id
stage
fit_start
fit_end
next_validation_or_eval_start
fit_end_before_next_block
x_scaler_checksum
y_scaler_checksum_or_identity
future_rows_used
status
```

Expected:

```text
future_rows_used=false.
```

---

# 162. Temporal leakage test schema

`rolling_origin_temporal_leakage_tests.csv`:

```text
fold_id
candidate_id
stage
test_case
max_allowed_timestamp
max_observed_training_timestamp
max_scaler_fit_timestamp
outer_target_value_accessed
future_input_accessed
pass
status
```

---

# 163. Common-target audit

`rolling_origin_common_target_audit.csv`:

```text
fold_id
model_id
outer_target_count
target_ids_fingerprint
reference_fingerprint
same_target_ids
same_order
same_ytrue_wh
status
```

---

# 164. Inner-selection run table

`rolling_origin_inner_selection_runs.csv`:

```text
fold_id
model_id
run_id
seed
inner_train_count
inner_val_count
max_epochs
patience
best_epoch
best_inner_val_rmse_wh
stop_epoch
stop_reason
scaler_fingerprints
status
```

---

# 165. Inner best epochs

`rolling_origin_inner_best_epochs.csv`:

```text
fold_id
model_id
best_epoch_inner
candidate_max_epochs
within_cap
source_inner_run_id
status
```

---

# 166. Refit run table

`rolling_origin_refit_runs.csv`:

```text
fold_id
model_id
run_id
seed
outer_train_count
refit_epochs
early_stopping_used=false
validation_selection_used=false
scaler_fingerprints
final_checkpoint_sha256
status
```

---

# 167. Refit epoch audit

`rolling_origin_refit_epoch_audit.csv`:

```text
fold_id
model_id
selected_inner_epoch
refit_epochs_completed
exact_match
outer_eval_seen_before_refit_complete
status
```

Expected:

```text
exact_match=true
outer_eval_seen_before_refit_complete=false.
```

---

# 168. Initialization audit

`rolling_origin_initialization_audit.csv`:

```text
fold_id
model_id
stage
seed
model_config_fingerprint
initial_state_fingerprint
fresh_initialization
warm_start_used
status
```

---

# 169. Sample-order audit

`rolling_origin_sample_order_audit.csv`:

```text
fold_id
model_id
stage
epoch_or_probe
sample_order_fingerprint
expected_seed_policy
status
```

No need for equal order across candidates with different batch/config, but each run must be reproducible.

---

# 170. Gradient diagnostics

`rolling_origin_gradient_diagnostics.csv`:

```text
fold_id
model_id
stage
epoch
mean_preclip_norm
max_preclip_norm
clip_fraction_if_applicable
nonfinite_events
status
```

---

# 171. Runtime diagnostics

`rolling_origin_runtime_diagnostics.csv`:

```text
fold_id
model_id
inner_runtime_seconds
refit_runtime_seconds
outer_eval_runtime_seconds
total_runtime_seconds
parameter_count
peak_memory_optional
status
```

---

# 172. Fold metrics table

`rolling_origin_fold_metrics.csv`:

```text
fold_id
model_id
model_family
outer_count
mae_wh
rmse_wh
r2
rank_all_models
rank_transformer_only
refit_epoch
status
```

---

# 173. Pooled metrics table

`rolling_origin_pooled_metrics.csv`:

```text
model_id
model_family
pooled_count
pooled_mae_wh
pooled_rmse_wh
pooled_r2
all_fold_predictions_complete
population_fingerprint
status
```

---

# 174. Macro metrics table

`rolling_origin_macro_metrics.csv`:

```text
model_id
fold_count
mean_fold_mae_wh
mean_fold_rmse_wh
sd_fold_rmse_wh
best_fold_rmse_wh
worst_fold_rmse_wh
fold_win_count_all
fold_win_count_transformer
status
```

---

# 175. Pairwise effects

`rolling_origin_pairwise_effects.csv`:

```text
scope = POOLED | RO1 | RO2 | RO3
model_a
model_b
rmse_a
rmse_b
delta_rmse_b_to_a
mae_a
mae_b
delta_mae_b_to_a
same_target_ids
status
```

---

# 176. Fold ranks

`rolling_origin_fold_ranks.csv`:

```text
fold_id
model_id
rmse_wh
rank_all
rank_transformer
status
```

---

# 177. Transformer robustness ranking

`transformer_robustness_ranking.csv`:

```text
rank
candidate_id
phase42_shortlist_position
pooled_rmse_wh
pooled_mae_wh
pooled_r2
mean_fold_rmse_wh
sd_fold_rmse_wh
worst_fold_rmse_wh
fold_win_count
all_3_folds_complete
tie_break_stage
recommended_for_phase45
status
```

---

# 178. Model-family comparison

`model_family_robustness_comparison.csv`:

```text
model_id
family
role
pooled_rmse_wh
pooled_mae_wh
pooled_r2
mean_fold_rmse
sd_fold_rmse
worst_fold_rmse
fold_wins
parameter_count_if_applicable
status
```

Include:

```text
recommended Transformer
LSTM_TUNED
Persistence.
```

---

# 179. Recommended Transformer artifact

`rolling_origin_recommended_transformer.json`:

```text
phase44_version
candidate_id
candidate_config_fingerprint
phase42_shortlist_position
source_phase42_role
pooled_outer_rmse_wh
pooled_outer_mae_wh
pooled_outer_r2
mean_fold_rmse_wh
sd_fold_rmse_wh
worst_fold_rmse_wh
fold_win_count
fold_metrics
ranking_rule
tie_break_used
all_folds_complete
boundary_protocol=WB0
test_status=NOT_ACCESSED
status
```

---

# 180. Phase45 handoff

`phase45_final_model_lock_handoff.json`:

```text
source_phase44_version
recommended_transformer_candidate_id
recommended_transformer_config
recommended_transformer_fingerprint
rolling_origin_pooled_metrics
rolling_origin_fold_metrics
transformer_ranking
lstm_tuned_context
persistence_context
model_family_comparison
boundary_sensitivity_context_from_S19
candidate_shortlist_fingerprint
rolling_fold_manifest_fingerprint
test_status=NOT_ACCESSED
approved_for_phase45
warnings
```

---

# 181. Findings artifact

`rolling_origin_robustness_findings.csv` possible codes:

```text
TR_C0_REMAINS_TOP
TR_C1_BECOMES_TOP
TR_C2_BECOMES_TOP
TRANSFORMER_RANK_STABLE
TRANSFORMER_RANK_CHANGES_BY_FOLD
LSTM_POOLED_GAIN
TRANSFORMER_POOLED_GAIN_OVER_LSTM
PERSISTENCE_POOLED_GAIN
LEARNED_GAIN_OVER_PERSISTENCE
INNER_BEST_EPOCH_STABLE
INNER_BEST_EPOCH_VARIES
FOLD_RMSE_VARIABILITY
WORST_FOLD_REVERSAL
SCALER_REFIT_VERIFIED
NO_OUTER_SELECTION_LEAKAGE
COMMON_TARGETS_VERIFIED
WB0_ONE_STEP_CONTEXT_VERIFIED
WB1_SENSITIVITY_CARRIED
CANDIDATE_NUMERICAL_FAILURE
TEST_FIREWALL_PRESERVED
INHERITED_WARNING
```

---

# 182. Discrepancy taxonomy

`rolling_origin_discrepancies.json`:

```text
PHASE42_NOT_APPROVED
TRANSFORMER_SHORTLIST_NOT_FROZEN
PHASE43_NOT_APPROVED
LSTM_WINNER_MISMATCH
PROTOCOL_AMENDMENT_PENDING
FOLD_MANIFEST_CHANGED
RVAL_PARTITION_ERROR
OUTER_FOLD_OVERLAP
OUTER_FOLD_UNION_MISMATCH
INNER_TEMPORAL_ORDER_ERROR
OUTER_TRAIN_EXPANSION_ERROR
CANDIDATE_OUTER_TARGET_MISMATCH
COMMON_YTRUE_MISMATCH
WINDOWPOP_MISMATCH
FEATURE_CONFIG_DRIFT
LOOKBACK_DRIFT
LOSS_DRIFT
OPTIMIZER_DRIFT
REVIN_DRIFT
WB_PROTOCOL_DRIFT
SCALER_FUTURE_LEAKAGE
INNER_VAL_USED_IN_SCALER_FIT
OUTER_EVAL_USED_IN_SCALER_FIT
OUTER_EVAL_USED_FOR_EARLY_STOP
OUTER_EVAL_USED_FOR_CHECKPOINT_SELECTION
REFIT_EPOCH_MISMATCH
WARM_START_USED
OPTIMIZER_STATE_REUSE
MODEL_CARRY_ACROSS_ORIGIN
ONLINE_FINE_TUNING_USED
PREDICTION_FEEDBACK_USED_AS_ACTUAL_HISTORY
CANDIDATE_ADDED_AFTER_FREEZE
CANDIDATE_DROPPED_AFTER_BAD_FOLD
PARTIAL_FOLD_RANKING
SCORE_BASED_RERUN
POOLED_RMSE_COMPUTED_AS_MEAN_FOLD_RMSE
ROUNDING_RANK_ERROR
TEST_LABEL_ACCESSED
TEST_PREDICTION_GENERATED
TEST_METRIC_COMPUTED
OTHER
```

---

# 183. Status model

## PASS

```text
3 folds frozen
nested epoch selection valid
fold-local scaling valid
all frozen candidates complete 3 folds
same outer target IDs
pooled metrics verified
Transformer ranking generated
Phase45 handoff ready
Test untouched.
```

## PASS_WITH_WARNING

Possible:

```text
large fold variability
LSTM beats Transformer
Persistence beats learned model
candidate ranking changes across folds
small pooled margin
initialization/sample-order diagnostic limitation
inherited S19 sensitivity warning.
```

## FAIL

Examples:

```text
outer evaluation leaks into selection
scaler future leakage
candidate missing fold
partial ranking
candidate shortlist changed
Test access
invalid refit.
```

---

# 184. Phase44 execution sequence

```text
1. Verify Phase42/43 signoffs.
2. Freeze model candidate universe.
3. Build ROBASE-v1.
4. Partition original Validation into V1/V2/V3.
5. Build inner/outer fold roles.
6. Freeze fold manifest/checksum.
7. Validate candidate compatibility.
8. Validate fold-local scaling implementation.
9. For each learned candidate/fold:
   a. Stage A inner epoch selection.
   b. Lock best epoch.
   c. Stage B fresh full-history refit.
   d. Freeze final refit model.
   e. Stage C outer evaluation once.
10. Evaluate Persistence on same outer IDs.
11. Verify common target/y_true alignment.
12. Create fold metrics.
13. Pool outer predictions.
14. Create pooled/macro metrics.
15. Compute pairwise effects.
16. Rank Transformer candidates.
17. Build model-family comparison.
18. Generate findings/figures.
19. Select recommended Transformer for Phase45.
20. Write Phase45 handoff.
21. Run tests/discrepancy audit.
22. Sign off.
```

---

# 185. Recommended execution order by fold

For reproducibility, preferred:

```text
RO1:
  all Transformer candidates
  LSTM
  Persistence

RO2:
  all Transformer candidates
  LSTM
  Persistence

RO3:
  all Transformer candidates
  LSTM
  Persistence
```

Within each candidate:

```text
Stage A
→ Stage B
→ Stage C.
```

Run order does not influence scores; record actual order.

---

# 186. Alternative execution order

Candidate-major order is acceptable if:

```text
fold manifest frozen
seed reset per run
no state carry
```

Record run schedule.

---

# 187. Preflight acceptance checklist

```text
[ ] Phase42 PASS/PASS_WITH_WARNING.
[ ] Transformer shortlist frozen.
[ ] Candidate fingerprints verified.
[ ] Phase43 PASS/PASS_WITH_WARNING.
[ ] LSTM_TUNED fingerprint verified.
[ ] Persistence contract verified.
[ ] WB0 primary protocol.
[ ] No protocol amendment pending.
[ ] SPLIT-v1 unchanged.
[ ] WINDOWPOP-v1 available.
[ ] RVAL_IDS built.
[ ] V1/V2/V3 deterministic.
[ ] Outer folds disjoint.
[ ] Outer fold union equals RVAL_IDS.
[ ] RO1 inner Train tail exists.
[ ] RO2 inner val=V1.
[ ] RO3 inner val=V2.
[ ] All temporal ordering assertions pass.
[ ] All candidates support all outer IDs.
[ ] Test firewall active.
[ ] Fold manifest frozen before training.
```

---

# 188. Data/scaling acceptance checklist

```text
[ ] ROBASE-v1 fingerprint generated.
[ ] Same outer target IDs across all models.
[ ] Same raw y_true across all models.
[ ] Stage A scalers fit inner train only.
[ ] Stage B scalers fit outer history only.
[ ] No outer row in scaler fit.
[ ] No full-pretest scaler leakage.
[ ] Candidate-specific feature scalers correct.
[ ] Candidate-specific YS correct.
[ ] TF pass-through correct.
[ ] RN1 fold-local bridge correct if active.
[ ] LSTM fold-local scaling correct.
[ ] Persistence raw Wh semantics correct.
```

---

# 189. Training acceptance checklist

```text
[ ] Every learned candidate Stage A fresh.
[ ] Every Stage A seed42.
[ ] Candidate max_epochs/patience exact.
[ ] Inner Validation RMSE Wh selects best epoch.
[ ] Outer eval never iterated during Stage A.
[ ] Stage B model fresh.
[ ] Stage B optimizer fresh.
[ ] Stage B scalers fresh/refit.
[ ] Stage B seed42.
[ ] Stage B trains exactly selected epoch count.
[ ] Stage B early stopping disabled.
[ ] Stage B outer eval unseen.
[ ] No warm-start.
[ ] No optimizer-state reuse.
[ ] No model carry across folds.
[ ] No online fine-tuning.
[ ] No score-based rerun.
[ ] Non-finite failures not silently skipped.
```

---

# 190. Evaluation acceptance checklist

```text
[ ] Stage C model.eval()/inference.
[ ] No backward.
[ ] No optimizer step.
[ ] Exact outer IDs.
[ ] Actual observed history used under WB0.
[ ] No prediction feedback as actual history.
[ ] Predictions finite.
[ ] Predictions inverse-transformed to Wh.
[ ] Residual = y_true-y_pred.
[ ] Fold MAE/RMSE/R² computed globally.
[ ] Persistence exact same IDs.
[ ] No outer target duplication.
[ ] Pooled prediction bundles complete.
[ ] Pooled RMSE recomputed from pooled errors.
[ ] Pooled RMSE not mean fold RMSE.
[ ] Macro RMSE/SD computed separately.
```

---

# 191. Ranking acceptance checklist

```text
[ ] Every Transformer has 3/3 valid folds.
[ ] Primary rank metric pooled outer RMSE Wh.
[ ] Full precision ranking.
[ ] Exact tie → lower worst fold.
[ ] Next exact tie → lower fold RMSE SD.
[ ] Next exact tie → Phase42 shortlist order.
[ ] Runtime not used for rank.
[ ] Parameter count not used for rank.
[ ] LSTM/Persistence not used to alter Transformer candidate tie rule.
[ ] Recommended Transformer artifact generated.
```

---

# 192. Reporting acceptance checklist

```text
[ ] Fold timeline documented.
[ ] Inner/outer distinction documented.
[ ] Fold-local scaling documented.
[ ] WB0 one-step observed-history assumption documented.
[ ] Candidate configuration frozen statement included.
[ ] Original Validation reuse limitation included.
[ ] Single-seed limitation included.
[ ] No statistical-significance overclaim.
[ ] LSTM comparison reported honestly.
[ ] Persistence comparison reported honestly.
[ ] S19 WB1 sensitivity context carried.
[ ] No Test conclusion.
[ ] Phase45 handoff generated.
```

---

# 193. Acceptance criteria

Phase44 PASS only when:

```text
The Transformer shortlist from Phase42 is unchanged.

The tuned LSTM from Phase43 is unchanged.

Three rolling-origin folds are frozen before model execution.

All folds lie entirely in the pre-Test Train+Validation region.

Original Validation target IDs are partitioned into three chronological, non-overlapping outer blocks.

Every learned model uses nested inner epoch selection.

Outer evaluation targets do not influence scaler fitting, early stopping, checkpoint selection or refit.

Stage B retrains from scratch on all history available before the fold origin.

Stage B trains for exactly the inner-selected epoch count.

Fold-local scalers use only allowed historical data.

WB0 rolling one-step context is preserved without recursive prediction feedback.

All model families are evaluated on exactly the same outer target IDs.

All Transformer candidates complete all three folds.

Fold and pooled MAE/RMSE/R² are produced in Wh.

Pooled outer RMSE ranks Transformer candidates.

Tie rules are respected.

LSTM/Persistence are reported as robustness baselines.

The recommended Transformer is handed to Phase45.

Test labels/predictions/metrics remain untouched.
```

---

# 194. Failure conditions

Phase44 FAIL if:

```text
candidate list changes after signoff

outer folds overlap

outer fold target union is wrong

Test enters any fold

outer targets are used for early stopping

outer rows are used to fit scalers

Stage B uses Stage A model weights

Stage B optimizer state reused

Stage B epoch count differs from inner selected epoch

model is fine-tuned within outer block

previous prediction replaces actual observed history

different models see different outer target IDs

candidate missing one fold but still ranked

bad fold is removed

pooled RMSE is computed by averaging fold RMSE

ranking uses rounded values

new hyperparameter is introduced

score-based rerun occurs

Test prediction/metric is computed.
```

---

# 195. Common mistakes

## 195.1 Dùng outer fold làm early stopping

Sai vì outer robustness score đã tham gia selection.

## 195.2 Reuse original Train scaler cho mọi fold

Không future leak ở RO1 nhưng không đúng full rolling refit semantics ở later origins.

## 195.3 Fit scaler trên Train+Validation toàn bộ

Leak future fold information.

## 195.4 Stage B load Stage A BEST rồi train tiếp

Sai. Refit phải fresh.

## 195.5 Cho RO2 load RO1 model

Sai warm-start across origins.

## 195.6 Trong outer block dùng prediction trước làm historical Appliances

Sai task contract; phải dùng actual observed prior value.

## 195.7 Mỗi model có outer target IDs khác nhau

Sai fairness.

## 195.8 Average ba fold RMSE rồi gọi là pooled RMSE

Sai.

## 195.9 Candidate thua RO1 rồi loại khỏi RO2/RO3

Sai.

## 195.10 LSTM thắng rồi đổi assignment final model thành LSTM ngay

Sai Phase44 role.

## 195.11 Dùng Test làm fold thứ 4

Forbidden.

---

# 196. Figures

Recommended:

```text
RO_44_01_fold_timeline.png
RO_44_02_fold_rmse_by_model.png
RO_44_03_pooled_rmse_comparison.png
RO_44_04_transformer_candidate_ranking.png
RO_44_05_fold_rmse_variability.png
RO_44_06_inner_best_epoch_by_fold.png
RO_44_07_pooled_prediction_scatter.png
RO_44_08_paired_error_deltas.png
RO_44_09_runtime_by_model_fold.png
RO_44_10_baseline_delta_by_fold.png
```

---

# 197. Fold timeline figure

Must visualize:

```text
Original Train
V1
V2
V3
Test locked
```

and for each origin:

```text
inner train
inner val
outer refit history
outer eval.
```

This is the key methodology figure.

---

# 198. Fold RMSE figure

Show:

```text
RO1
RO2
RO3
```

for all candidates.

No Test bar.

---

# 199. Pooled RMSE figure

Show:

```text
Transformer shortlist
LSTM_TUNED
Persistence.
```

Label:

```text
Pre-Test Rolling-Origin Pooled RMSE
```

not final Test RMSE.

---

# 200. Candidate ranking figure

Transformer-only.

Sort by pooled RMSE.

---

# 201. Inner best epoch figure

Useful to detect:

```text
training-budget sensitivity across temporal origins.
```

No candidate selection beyond already defined pooled score.

---

# 202. Pooled prediction scatter

Validation rolling outer predictions only.

Do not mix Test.

---

# 203. Report structure

`rolling_origin_report.md`:

```text
1. Objective
2. Frozen candidate universe
3. Why rolling-origin robustness is required
4. Pre-Test data boundary
5. RO3 fold construction
6. Nested inner/outer design
7. Fold-local scaling
8. WB0 rolling one-step context
9. Candidate-specific training contracts
10. LSTM/Persistence baseline contracts
11. Temporal leakage safeguards
12. Inner epoch-selection results
13. Refit verification
14. Fold-level outer metrics
15. Pooled outer metrics
16. Transformer robustness ranking
17. Fold-to-fold stability
18. Transformer vs tuned LSTM
19. Learned models vs Persistence
20. Pairwise effects
21. S19 boundary sensitivity context
22. Limitations
23. Recommended Transformer for Phase45
24. Test firewall
25. Definition of Done
```

---

# 204. README

`README_ROLLING_ORIGIN_ROBUSTNESS.md` explains:

```text
why 3 folds
how V1/V2/V3 are built
inner vs outer validation
why full-history refit is fresh
why scalers are fold-local
WB0 actual-history semantics
why no warm-start
same outer target rule
pooled vs macro RMSE
Transformer ranking rule
LSTM/Persistence roles
why Test is excluded
Phase45 handoff.
```

---

# 205. Summary artifact

`rolling_origin_summary.json`:

```text
version
fold_protocol
fold_count
transformer_candidate_count
learned_model_count
outer_target_total_count
fold_manifest_fingerprint
candidate_shortlist_fingerprint
lstm_fingerprint
fold_metrics
pooled_metrics
macro_metrics
transformer_ranking
recommended_transformer
lstm_comparison
persistence_comparison
inner_epoch_summary
scaler_audit_status
temporal_leakage_status
boundary_sensitivity_status
test_status
phase45_handoff
overall_status
```

---

# 206. Phase44 sign-off

`phase_44_signoff.json` minimum:

```text
phase = 44
phase_name = Rolling-origin robustness
version = ROLLING_ORIGIN-v1
fold_protocol = RO3_EXPANDING_PRETEST-v1
fold_count = 3
phase42_shortlist_fingerprint
phase43_lstm_fingerprint
fold_manifest_fingerprint
transformer_candidate_ids
lstm_model_id
persistence_id
all_candidates_complete
common_outer_targets_verified
nested_epoch_selection_verified
fold_local_scaling_verified
outer_selection_leakage = false
warm_start_used = false
online_update_used = false
recommended_transformer_candidate_id
recommended_transformer_fingerprint
recommended_pooled_rmse_wh
recommended_worst_fold_rmse_wh
recommended_fold_rmse_sd
lstm_pooled_rmse_wh
persistence_pooled_rmse_wh
test_status = NOT_ACCESSED
approved_for_phase45
warnings
overall_status
created_at
```

---

# 207. Phase45 handoff requirement

Phase45 must receive enough evidence to freeze final Transformer without rerunning robustness.

Required:

```text
recommended candidate exact config
candidate fingerprint
all 3 fold metrics
pooled metrics
tie-break evidence if used
shortlist rankings
LSTM/Persistence comparison
S19 WB1 sensitivity context
known warnings
no-Test declaration.
```

---

# 208. Phase45 must not reinterpret incomplete candidate

Hard:

```text
approved_for_phase45=false
```

if recommended ranking is based on incomplete folds.

---

# 209. Robustness caveat: candidate selection history

Transformer shortlist was generated from earlier Validation sweeps.

Therefore Phase44 is best described as:

```text
temporal robustness assessment within the pre-Test development data
```

rather than independent final generalization.

---

# 210. Robustness caveat: K=3

Three folds provide:

```text
useful temporal variation evidence
```

but not a large-sample distribution of performance across origins.

Do not overstate stability.

---

# 211. Robustness caveat: one seed

Differences can include stochastic training variation.

Phase46 multi-seed later addresses seed sensitivity for final locked configuration.

---

# 212. Robustness caveat: expanding history

Fold difficulty changes with:

```text
time regime
training-set size
distribution shift.
```

This is intended.

Do not assume fold RMSE differences are caused by only one factor.

---

# 213. Robustness caveat: actual past target context

For FS1/FS2 and Persistence:

```text
past observed Appliances
```

is used under rolling one-step deployment.

This differs from autonomous multi-step forecasting.

State explicitly in final methodology.

---

# 214. Robustness caveat: Persistence and FS0

If a learned candidate excludes historical Appliances but Persistence uses it:

```text
Persistence remains a task-level naive baseline
```

not an exact information-set baseline.

Carry the nuance from Phase14.

---

# 215. Recommended high-level flow

```text
Phase42 frozen Transformer shortlist
+
Phase43 frozen LSTM
+
Persistence
↓
build ROBASE-v1
↓
partition Validation → V1/V2/V3
↓
freeze folds
↓
for each learned model × fold:
    inner epoch selection
    fresh full-history refit
    outer evaluation
↓
Persistence outer evaluation
↓
same-target verification
↓
fold metrics
↓
pooled metrics
↓
Transformer robustness ranking
↓
model-family comparison
↓
recommend Transformer
↓
Phase45 handoff
```

---

# 216. Recommended implementation pseudocode

```text
assert phase42.shortlist_frozen
assert phase43.phase44_handoff_ready
assert phase41.primary_protocol == "WB0"
assert no_protocol_amendment_pending
assert test_locked

transformers = load_phase42_transformer_shortlist()
lstm = load_phase43_tuned_lstm()
persistence = PERSISTENCE_LAST_VALUE

robase = build_robustness_base_population(
    source="WINDOWPOP-v1",
    regions=["TRAIN", "VALIDATION"],
    candidates=transformers + [lstm]
)

rtrain_ids = robase.train_ids
rval_ids = robase.validation_ids

V1, V2, V3 = deterministic_array_split(
    rval_ids,
    n_splits=3
)

folds = {
    "RO1": {
        "outer_train": rtrain_ids,
        "inner_val": tail(rtrain_ids, len(V1)),
        "inner_train": rtrain_ids[:-len(V1)],
        "outer_eval": V1
    },
    "RO2": {
        "outer_train": concat(rtrain_ids, V1),
        "inner_train": rtrain_ids,
        "inner_val": V1,
        "outer_eval": V2
    },
    "RO3": {
        "outer_train": concat(rtrain_ids, V1, V2),
        "inner_train": concat(rtrain_ids, V1),
        "inner_val": V2,
        "outer_eval": V3
    }
}

validate_and_freeze_fold_manifest(folds)

learned_models = transformers + [lstm]

for fold in folds:
    for candidate in learned_models:

        # Stage A
        seed_all(42)

        inner_scalers = fit_candidate_scalers(
            candidate=candidate,
            training_history=fold.inner_train
        )

        inner_train_loader = build_loader(
            candidate,
            ids=fold.inner_train,
            scalers=inner_scalers,
            shuffle=True
        )

        inner_val_loader = build_loader(
            candidate,
            ids=fold.inner_val,
            scalers=inner_scalers,
            shuffle=False
        )

        inner_model = build_fresh_model(candidate)
        inner_optimizer = build_fresh_optimizer(candidate)

        inner_result = TRAINING_ENGINE.fit(
            model=inner_model,
            train=inner_train_loader,
            val=inner_val_loader,
            max_epochs=candidate.max_epochs,
            patience=candidate.patience,
            monitor="validation_rmse_wh"
        )

        selected_epoch = inner_result.best_epoch

        assert 1 <= selected_epoch <= candidate.max_epochs

        # Stage B
        seed_all(42)

        refit_scalers = fit_candidate_scalers(
            candidate=candidate,
            training_history=fold.outer_train
        )

        refit_train_loader = build_loader(
            candidate,
            ids=fold.outer_train,
            scalers=refit_scalers,
            shuffle=True
        )

        refit_model = build_fresh_model(candidate)
        refit_optimizer = build_fresh_optimizer(candidate)

        refit_result = train_exact_epochs(
            model=refit_model,
            optimizer=refit_optimizer,
            train=refit_train_loader,
            epochs=selected_epoch,
            early_stopping=False
        )

        assert refit_result.epochs_completed == selected_epoch

        # Stage C
        outer_loader = build_loader(
            candidate,
            ids=fold.outer_eval,
            scalers=refit_scalers,
            shuffle=False
        )

        predictions = evaluate_once_in_wh(
            refit_model,
            outer_loader
        )

        save_outer_predictions(candidate, fold, predictions)

    persistence_predictions = evaluate_persistence(
        target_ids=fold.outer_eval,
        use_actual_prior_observed_appliances=True
    )

    save_persistence_predictions(
        fold,
        persistence_predictions
    )

verify_same_outer_target_ids_and_ytrue_all_models()

fold_metrics = compute_fold_metrics()
pooled_predictions = concatenate_disjoint_outer_predictions()
pooled_metrics = compute_metrics_from_pooled_predictions()
macro_metrics = compute_fold_macro_statistics()

transformer_ranking = rank_transformers(
    primary="pooled_rmse_wh",
    tie1="worst_fold_rmse_wh",
    tie2="sd_fold_rmse_wh",
    tie3="phase42_shortlist_position"
)

recommended = transformer_ranking[0]

write_model_family_comparison(
    recommended_transformer=recommended,
    lstm=lstm,
    persistence=persistence
)

write_phase45_handoff(recommended)

assert test_not_accessed
signoff_phase44()
```

---

# 217. Definition of Done

\[
\boxed{
Frozen\ Transformer\ Shortlist
+
Frozen\ Tuned\ LSTM
+
Persistence
+
Three\ Expanding\ Origins
+
Nested\ Epoch\ Selection
+
Fresh\ Full\text{-}History\ Refit
+
Fold\text{-}Local\ Scaling
+
Same\ Outer\ Targets
+
Pooled\ Outer\ RMSE
+
Recommended\ Transformer
+
No\ Test
+
Phase45\ Handoff
}
\]

---

# 218. Final status contract

```text
PHASE 44 performs temporal robustness evaluation.

Data:
original Train + Validation only.
Test untouched.

Folds:
K=3
original Validation target IDs
→ V1/V2/V3 contiguous chronological blocks.

RO1:
train history = original Train
outer eval = V1.

RO2:
train history = Train + V1
outer eval = V2.

RO3:
train history = Train + V1 + V2
outer eval = V3.

Each learned model/fold:
Stage A:
inner train
→ inner validation
→ select best epoch.

Stage B:
fresh model
fresh optimizer
fold-local full-history scalers
train all pre-origin history
exactly selected epoch count.

Stage C:
freeze
evaluate outer block once.

No:
outer early stopping
outer checkpoint selection
warm-start
cross-origin model carry
online fine-tuning
prediction feedback
full-pretest scaler leakage
candidate changes
Test.

Models:
Phase42 frozen Transformer shortlist
Phase43 LSTM_TUNED
Persistence.

Fairness:
same outer target IDs and y_true.

Primary Transformer ranking:
pooled outer-fold RMSE Wh.

Exact tie:
worst-fold RMSE
→ fold RMSE SD
→ Phase42 shortlist order.

Output:
recommended Transformer for Phase45
plus LSTM/Persistence robustness context.

After ROLLING_ORIGIN-v1 PASS:
proceed to
PHASE 45 — Final Model Lock.
```

---

# 219. Final check

Correct:

```text
freeze candidates
→ freeze RO1/RO2/RO3
→ inner epoch selection
→ fresh full-history refit
→ outer evaluation once
→ same-target check
→ pooled RMSE
→ rank Transformer candidates
→ compare LSTM/Persistence
→ Phase45 handoff
```

Incorrect:

```text
outer fold used for early stopping
→ refit scaler on future rows
→ warm-start across origins
→ drop bad fold
→ change candidate after RO1
→ use Test as RO4
```

Chỉ sau khi:

```text
ROLLING_ORIGIN-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
approved_for_phase45 = true
```

mới chuyển sang **PHASE 45 — Final Model Lock**.
