# PHASE 38 — S16 EPOCH-CAP SWEEP

## Kế hoạch controlled sweep cho Maximum Training Epochs của Transformer Regression

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Reference source:** `SWEEP_S15_LOSS-v1`  
**Sweep ID:** `S16_EPOCH_CAP`  
**Output version:** `SWEEP_S16_EPOCHCAP-v1`  
**Phase trước:** `Phase_37_S15_Loss_sweep.md`

---

# 1. Vai trò của Phase 38

Phase 38 là controlled experiment thứ mười sáu trong chuỗi Transformer development sweeps.

Mục tiêu duy nhất:

> Với toàn bộ data pipeline, feature configuration, target scaling, lookback, pooling, activation, batch size, optimizer, learning rate, weight decay, dropout, `d_model`, attention heads, encoder layers, FFN width, training loss, patience, gradient clipping, sample population và seed đã được khóa từ Phase 37, giới hạn tối đa `50 epochs` có đang cắt ngắn quá trình học hay có cần mở rộng lên `100 epochs`?

Phase 38 chỉ thay đúng một registered hyperparameter:

```text
max_epochs
```

với hai condition:

```text
E50  = max_epochs 50
E100 = max_epochs 100
```

Nguyên tắc trung tâm:

\[
\boxed{
One\ Epoch\ Cap\ Factor
+
Same\ Early\ Stopping
+
Same\ Model
+
Same\ Data
+
Same\ Optimizer
+
Same\ Seed
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Ý nghĩa chính xác của “Epoch-cap sweep”

Phase 38 **không** so sánh:

```text
model được train đúng 50 epochs
vs
model được train đúng 100 epochs
```

vì project đang dùng Early Stopping.

Phase 38 thực sự so sánh:

```text
maximum allowed epochs = 50
vs
maximum allowed epochs = 100
```

while keeping:

```text
patience = 10
min_delta = 0
early-stop metric = Validation RMSE Wh
```

unchanged.

Do đó:

```text
actual_epochs_completed <= max_epochs.
```

Đây là distinction bắt buộc trong toàn bộ report.

---

# 3. Vị trí Phase 38 trong master execution plan

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
```

Phase 38 không được quay lại thay:

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
training loss
patience
min_delta
gradient clipping
RevIN
boundary protocol
```

---

# 4. Câu hỏi nghiên cứu của S16

Phase 38 phải trả lời:

```text
1. E50 hay E100 tạo verified Validation RMSE Wh thấp hơn?

2. E50 cap có thực sự binding hay Early Stopping đã dừng trước cap?

3. E100 có thực sự sử dụng epochs > 50 hay cũng dừng sớm?

4. Nếu E100 đi qua epoch50, BEST checkpoint có xuất hiện sau epoch50 không?

5. Có Validation improvement mới nào sau epoch50 không?

6. E100 có chỉ kéo dài training nhưng BEST vẫn nằm trong first 50 epochs không?

7. Prefix trajectory của E100 có tái hiện E50 trước epoch50 không?

8. E100 có làm tăng overfitting-like post-best deterioration không?

9. Runtime/optimizer-step cost của extra budget là bao nhiêu?

10. Epoch cap nào trở thành current reference cho Phase39?
```

---

# 5. Current reference từ Phase 37

Phase 38 phải load:

```text
s15_loss_winner.json
s15_reference_update.json
phase_37_signoff.json
```

để resolve:

```text
FV*   = selected feature variant
YS*   = selected target scaling
L*    = selected lookback
P*    = selected pooling
A*    = selected activation
B*    = selected batch size
LR*   = selected learning rate
WD*   = selected weight decay
DR*   = selected dropout
D*    = selected d_model
H*    = selected num_heads
HD*   = selected head_dim
N*    = selected num_layers
F*    = selected ffn_dim
LOSS* = selected training loss
```

Nếu:

```text
LOSS* = HUBER
```

thì phải giữ nguyên:

```text
Huber delta = 1.0 model-space
reduction = mean.
```

Phase 38 không được hard-code MSE.

---

# 6. Canonical epoch-cap options

Registry:

```text
E50  = 50
E100 = 100
```

Exact numeric values:

```text
50
100
```

No other epoch cap enters S16 ranking.

---

# 7. Reference epoch cap

Current sequential reference từ Phase 37 uses:

```text
E50.
```

If exact S16 contract match:

```text
E50  → REUSE S15 winner
E100 → NEW fresh run
```

Normal S16 execution:

```text
1 new E100 scientific run
+
1 reused E50 reference
```

---

# 8. Maximum epochs is a ceiling, not a guaranteed budget

For each condition:

```text
training terminates at the earlier of:
1. Early Stopping trigger
2. max_epochs cap.
```

Formally:

\[
E_{actual}
=
\min(E_{early-stop}, E_{cap})
\]

when an early-stop trigger exists.

Do not write:

```text
E100 always receives twice the optimization steps of E50.
```

That may be false.

---

# 9. Early Stopping remains fixed

Hard:

```text
patience = 10
min_delta = 0
monitor = Validation RMSE Wh
direction = minimize
```

Same strict-improvement semantics from `TRAINING_ENGINE-v1`.

No patience change in S16.

---

# 10. BEST checkpoint policy remains fixed

Both conditions:

```text
BEST = earliest strict minimum Validation RMSE Wh.
```

Changing max_epochs must not change:

```text
selection metric
tie semantics
checkpoint overwrite rule.
```

---

# 11. Why patience must not scale with epoch cap

Do not use:

```text
E50  → patience10
E100 → patience20
```

That would change:

```text
max_epochs + patience
```

simultaneously.

S16 tests only the ceiling.

---

# 12. Why scheduler=None is critical

Current training contract:

```text
scheduler = None
warmup = None.
```

Therefore changing:

```text
max_epochs 50 → 100
```

does not alter a schedule defined as a fraction of total epochs.

This makes S16 a clean epoch-cap experiment.

---

# 13. Scheduler-dependent training would confound this sweep

If code unexpectedly uses:

```text
CosineAnnealing(T_max=max_epochs)
OneCycle(total_steps=max_epochs*steps)
linear decay tied to max_epochs
warmup fraction of max_epochs
```

then E50/E100 would already differ before epoch50.

That would no longer isolate the cap.

Hard audit:

```text
no epoch-cap-dependent scheduler.
```

---

# 14. Loss remains selected LOSS*

Hard:

```text
training loss = LOSS*
```

from S15.

If MSE:

```text
MSELoss(reduction=mean).
```

If Huber:

```text
HuberLoss(delta=1.0 model-space, reduction=mean).
```

No loss change.

---

# 15. Model architecture is exactly identical

Expected:

```text
same architecture fingerprint
same parameter count
same parameter names
same parameter tensor shapes
same buffer shapes.
```

`max_epochs` is a training-loop budget field only.

It must not alter model construction.

---

# 16. Optimizer is exactly identical

Hard:

```text
AdamW
LR*
WD*
same parameter-group policy
same betas
same eps
same foreach/fused policy if explicitly frozen
```

No budget-specific optimizer setting.

---

# 17. Gradient clipping remains fixed

Hard:

```text
GC1
max_norm = 1.0
norm_type = 2
```

for both E50/E100.

Phase39 handles clipping.

---

# 18. Batch size remains selected B*

No:

```text
larger batch for E100 to save time.
```

Same optimizer steps per completed epoch.

---

# 19. Same data population is mandatory

Both conditions use exact same:

```text
Train sample IDs
Validation sample IDs
WINDOWPOP-v1
feature/scaler pipeline.
```

Max epoch cap cannot affect sample eligibility.

---

# 20. Same seed is mandatory

Hard:

```text
seed = 42.
```

The new E100 official run must start from the same seed policy as E50.

---

# 21. E100 must be a fresh registered run

Preferred official S16 implementation:

```text
reseed42
fresh DataLoaders
fresh model
fresh criterion
fresh optimizer
fresh early-stop state
max_epochs=100
train from epoch1.
```

Do **not** define the primary E100 candidate as:

```text
load E50 BEST
continue training.
```

---

# 22. Why E50 BEST must never be used to continue E100

E50 BEST may come from:

```text
epoch < 50
```

and does not preserve the exact uninterrupted optimization trajectory.

Continuing BEST would:

```text
rewind model parameters
lose later optimizer trajectory
change patience semantics
```

and invalidate the cap comparison.

---

# 23. Resume-from-LAST is not the primary S16 protocol

In principle, continuing from exact E50 LAST with:

```text
optimizer state
RNG state
DataLoader state
early-stop counter
epoch index
```

could reproduce an uninterrupted extension.

However the clean registered S16 comparison is:

```text
fresh E100 run from epoch1
```

under the same seed/config.

This allows direct prefix-reproducibility auditing.

---

# 24. No early-stopping counter reset at epoch50

E100 is one continuous training run.

Do not implement:

```text
train 50
reset patience counter
train another 50.
```

The patience counter evolves continuously from epoch1.

---

# 25. No optimizer reset at epoch50

E100 must not:

```text
reset AdamW moments at epoch50.
```

No milestone reset.

---

# 26. No RNG reset at epoch50

Do not:

```text
torch.manual_seed(42)
```

again at epoch50.

That would alter dropout/sample-order behavior.

---

# 27. No DataLoader reseeding at epoch50

E100 must follow the normal epoch-to-epoch generator evolution under `DATALOADERS-v1`.

No “second stage” loader.

---

# 28. Prefix-equivalence principle

Because E50/E100 differ only in the stopping ceiling, their training process **before the E50 boundary** should be the same under deterministic/reproducible execution.

Conceptually:

```text
E100 epochs 1..k
should match
E50 epochs 1..k
```

for all epochs both runs actually share, assuming:

```text
same initial state
same sample order
same dropout RNG
same environment/backend determinism
same criterion/optimizer.
```

---

# 29. Prefix equivalence is a powerful hidden-drift detector

If E50 and E100 diverge substantially before epoch50 despite full reproducibility metadata, investigate:

```text
different initialization
different Train order
different environment
hidden config drift
scheduler tied to cap
RNG contamination
non-deterministic backend operation
metric/history mismatch.
```

Do not interpret such divergence as an epoch-cap effect.

---

# 30. Historical reference limitation for prefix matching

E50 is reused from Phase37.

Exact prefix equality may be impossible to verify if E50 lacks:

```text
initial-state fingerprint
per-epoch sample-order fingerprint
RNG provenance
full epoch history
same environment hash.
```

Allowed:

```text
PREFIX_MATCH = NOT_VERIFIABLE.
```

Do not retrain E50 solely to manufacture perfect matching metadata.

---

# 31. Prefix-comparison scope

Compare all common observed epochs:

```text
1 .. min(E50_epochs_completed, E100_epochs_completed, 50)
```

for available deterministic quantities:

```text
Validation RMSE Wh
Validation MAE Wh
Train criterion value
learning rate
clipping fraction
optional checkpoint hash/state fingerprint.
```

Prediction/weight equality can be checked only if artifacts exist and environment supports it.

---

# 32. Strong prefix match vs metric prefix match

Two levels:

```text
LEVEL 1 — METRIC_PREFIX
per-epoch metrics allclose.

LEVEL 2 — STATE_PREFIX
model/optimizer states or checkpoints match at selected epochs.
```

State-prefix is stronger but optional.

Do not claim state identity from metric identity alone.

---

# 33. Prefix tolerance policy

If exact bitwise reproduction is not guaranteed by backend:

```text
use predeclared numeric tolerance
```

from environment/reproducibility policy.

Do not invent a tolerance after seeing the difference.

Record:

```text
comparison_type
atol
rtol
backend
determinism_status.
```

---

# 34. E50 cap-binding classification

Phase38 must classify the E50 reference.

## Case A — Early Stopping before epoch50

If:

```text
stop_reason = EARLY_STOP
epochs_completed < 50
```

then:

```text
E50 cap was NON_BINDING.
```

The run stopped for patience, not the cap.

---

# 35. E50 cap-binding classification: reached cap

If:

```text
stop_reason = MAX_EPOCHS
epochs_completed = 50
```

then the cap was reached.

This means:

```text
cap boundary was operationally binding
```

but it does **not** by itself prove that more epochs would improve Validation RMSE.

E100 answers that empirical question.

---

# 36. Cap reached vs cap harmful

Important distinction:

```text
CAP_REACHED
!=
CAP_HARMFUL.
```

E50 can hit epoch50 yet its BEST may be much earlier.

If E100 continues but never improves BEST:

```text
E50 cap was reached
but extending it did not improve selected performance.
```

---

# 37. Best-at-boundary diagnostic

Record whether E50 BEST occurred:

```text
epoch50
within last K epochs
far before boundary.
```

Recommended descriptive:

```text
K = min(5, epochs_completed)
```

consistent with earlier diagnostics.

No arbitrary significance threshold.

---

# 38. Patience state at E50 boundary

If E50 stopped by cap, record if available:

```text
patience_counter_at_epoch50
best_epoch
epochs_since_best.
```

This shows whether Early Stopping would likely have triggered shortly after 50 absent further improvement.

Diagnostic only.

---

# 39. E100 extra-budget utilization

Required:

```text
epochs_completed_E100
epochs_beyond_50 = max(0, epochs_completed_E100 - 50)
optimizer_steps_beyond_50
runtime_beyond_50 if measurable.
```

This tells whether the extra ceiling was actually used.

---

# 40. E100 early-stop-before-50 case

If E100:

```text
EARLY_STOP at epoch < 50
```

then E100 never used the extra budget.

If reproducibility is valid, this should align with an equivalent E50 trajectory.

Record:

```text
EXTRA_BUDGET_UNUSED.
```

---

# 41. E100 early-stop-between-51-and-100 case

If:

```text
50 < epochs_completed_E100 < 100
stop_reason=EARLY_STOP
```

then the extra budget was partly used.

Analyze whether BEST:

```text
occurred <=50
or
>50.
```

---

# 42. E100 reaches epoch100

If:

```text
stop_reason=MAX_EPOCHS
epochs_completed=100
```

then E100 cap itself is reached.

Record:

```text
E100_BOUNDARY_REACHED.
```

Do not automatically extend to E150/E200.

---

# 43. No automatic cap-range extension

Even if E100 wins at epoch100:

```text
do not add E150/E200
```

inside S16.

That would be adaptive hidden search.

Record:

```text
BOUNDARY_WINNER / E100_CAP_REACHED
```

for later interpretation.

---

# 44. Post-50 improvement diagnostic

For E100, calculate:

```text
best_rmse_epoch_1_50
best_rmse_epoch_51_100_observed
```

when applicable.

Define:

\[
Gain_{post50}
=
RMSE^{best}_{1:50}
-
RMSE^{best}_{51:100}
\]

Positive:

```text
a post-50 checkpoint improved over the first-50 best.
```

---

# 45. If E100 stops before epoch51

Then:

```text
post50_best = N/A
post50_gain = N/A
extra_budget_used = false.
```

Do not fill zeros that imply observed no improvement after50.

Use:

```text
NOT_OBSERVED / NOT_APPLICABLE.
```

---

# 46. First post-50 improvement epoch

If E100 gets a strict new BEST after50, record:

```text
first_new_best_after_50
```

This reveals when the extended budget first became useful.

---

# 47. Number of post-50 new BEST events

Record:

```text
new_best_count_epochs_51_plus.
```

Diagnostic only.

---

# 48. Final BEST location

For E100:

```text
best_epoch <=50
or
best_epoch >50.
```

This is a key S16 result.

---

# 49. E100 improvement without later BEST is impossible under strict BEST definition

If report claims:

```text
post50 Validation RMSE improved
```

but no new BEST after50, clarify “improved relative to immediately prior epoch” versus “improved over global best”.

S16's important quantity is:

```text
new global BEST after50.
```

---

# 50. Best checkpoint not last checkpoint

Always compare:

```text
verified BEST RMSE
```

not:

```text
epoch50 RMSE
epoch100 RMSE
last RMSE.
```

The cap sweep is still model-selection by BEST Validation performance.

---

# 51. Why last-epoch comparison is misleading

E100 may:

```text
find best at epoch63
then worsen
stop at epoch73.
```

Comparing epoch73 vs E50 last would misrepresent the selected model.

Use BEST.

---

# 52. Winner rule

\[
winner
=
\arg\min
\left(
RMSE_{E50}^{BEST},
RMSE_{E100}^{BEST}
\right)
\]

using full-precision verified Validation RMSE Wh.

---

# 53. Exact RMSE tie rule

If exact full-precision equality:

```text
prefer E50.
```

Rationale:

```text
lower maximum allowed budget
lower potential training cost
simpler current protocol
same selected Validation performance
predeclared parsimony.
```

Only exact tie invokes this rule.

---

# 54. No runtime override for non-tie

If E100 achieves lower RMSE by any non-zero full-precision margin:

```text
E100 wins.
```

Do not choose E50 only because it trains faster.

---

# 55. No arbitrary minimum gain threshold

Do not require:

```text
>1% improvement
>1 Wh improvement
```

for E100 to win.

Strict lower verified RMSE wins.

---

# 56. Epoch-cap effect formula

Define:

\[
\Delta RMSE_{50\rightarrow100}
=
RMSE_{E50}
-
RMSE_{E100}
\]

Positive:

```text
E100 improves.
```

Relative:

\[
Improvement\%
=
100
\times
\frac{RMSE_{E50}-RMSE_{E100}}
{RMSE_{E50}}.
\]

Also:

```text
MAE delta = MAE_E50 - MAE_E100
R² delta = R²_E100 - R²_E50.
```

---

# 57. RMSE/R² consistency guard

Same Validation population means:

```text
RMSE and R² ranking should be consistent.
```

If not:

```text
investigate metric/population/rounding issue.
```

MAE may legitimately differ.

---

# 58. Metric ranking divergence

If:

```text
RMSE winner != MAE winner
```

record:

```text
METRIC_RANKING_DIVERGENCE.
```

Winner remains RMSE-based.

---

# 59. Stop-reason audit

Every run must have exact:

```text
stop_reason
```

from allowed values such as:

```text
EARLY_STOP
MAX_EPOCHS
FAILURE
INTERRUPTED
```

A completed scientific run cannot have ambiguous stop reason.

---

# 60. Epoch-count audit

Hard:

```text
E50 epochs_completed <= 50
E100 epochs_completed <= 100.
```

No epoch100+1 off-by-one.

---

# 61. Epoch indexing convention

Choose and lock one display convention:

```text
human epoch numbers = 1..N
```

even if internal loop indexes from0.

Artifacts must not mix:

```text
epoch_index0
epoch_number1
```

without explicit columns.

Recommended:

```text
epoch_number = 1-based canonical reporting field.
```

---

# 62. Patience counting semantics must remain unchanged

`TRAINING_ENGINE-v1` strict logic:

```text
strict RMSE improvement
→ reset non-improvement counter

otherwise
→ increment

counter >= patience
→ early stop
```

Do not change off-by-one semantics in E100.

---

# 63. min_delta remains zero

Hard:

```text
min_delta = 0.
```

No practical-change threshold added.

---

# 64. No special grace period after epoch50

E100 does not receive:

```text
“at least 10 extra epochs”
```

unless patience state naturally permits it.

The early-stop counter carries continuously.

---

# 65. No checkpoint reset at epoch50

BEST remains global across the entire E100 run.

Do not reset:

```text
best_rmse
best_epoch
patience counter
```

at the boundary.

---

# 66. Global BEST continuity

For E100:

```text
BEST_{1:100}
=
minimum over all observed Validation epochs.
```

Not:

```text
minimum first50
then separate minimum second50.
```

Post50 analysis is diagnostic only.

---

# 67. Checkpoint retention

E100 official run still maintains:

```text
BEST
LAST
```

under `TRAINING_ENGINE-v1`.

Optional milestone:

```text
EPOCH50_SNAPSHOT
```

may be saved for diagnostics if predeclared, but it must not replace BEST/LAST semantics.

---

# 68. Optional epoch50 snapshot

Useful for exact prefix/state auditing:

```text
checkpoint_epoch_0050
```

if E100 reaches50.

This snapshot is diagnostic only.

It may store:

```text
model state
optimizer state
early-stop state
RNG state
DataLoader generator state.
```

No requirement if storage policy prefers not to.

---

# 69. Milestone snapshot must not affect training

Saving epoch50 checkpoint must not:

```text
reseed RNG
reload model
reset optimizer
reset patience
change mode transitions.
```

Pure serialization only.

---

# 70. Same Train shuffle policy

Each E100 epoch:

```text
Train shuffle = true
```

under the same generator progression policy.

No deterministic “same order every epoch” unless already specified by `DATALOADERS-v1`.

---

# 71. Same Validation order

Validation remains:

```text
shuffle = false
full ordered population
```

every epoch.

---

# 72. Same sample-weighted training loss aggregation

No change.

---

# 73. Same Validation metric aggregation

Global split metrics, not average batch RMSE.

No change.

---

# 74. Same target inverse-transform policy

Evaluation still:

```text
model-space prediction
→ inverse transform if YS1
→ raw Wh
→ METRICS-v1.
```

No budget-specific metric path.

---

# 75. Same dropout train/eval mode semantics

Every epoch:

```text
Train:
model.train()

Validation:
model.eval()

Next Train:
model.train().
```

No mode drift in longer run.

---

# 76. Longer run increases opportunity for mode-transition bugs

E100 repeats more train/eval cycles.

Audit:

```text
mode at train start
mode at validation start
mode after validation before next epoch.
```

No validation in `train()`.

---

# 77. Gradient clipping diagnostics continue through E100

Record per epoch:

```text
mean preclip grad norm
max preclip grad norm
clipping fraction
nonfinite events.
```

This can reveal late-training stability.

Do not change clip based on diagnostics.

---

# 78. Gradient norm late-training analysis

Recommended:

```text
epochs1_50 summary
epochs51_plus summary
```

for E100 if extra budget used.

Diagnostic only.

---

# 79. Learning rate audit

Since scheduler=None:

```text
LR should remain LR*
```

for every optimizer step in both conditions.

Hard:

```text
no LR drift after epoch50.
```

---

# 80. Weight decay audit

Same `WD*` throughout.

---

# 81. Optimizer-state continuity

For E100, AdamW state must evolve uninterrupted from epoch1.

No reinitialization at boundary.

---

# 82. Optimizer steps per epoch

At fixed B*/population:

```text
steps_per_complete_epoch
```

must remain constant.

Required:

```text
expected_steps_per_epoch
observed_steps_per_epoch.
```

---

# 83. Extra optimizer-step cost

If E100 completes epochs >50:

\[
ExtraSteps
=
Steps_{E100}
-
50\times StepsPerEpoch
\]

when all first50 complete epochs occurred.

More generally calculate from recorded optimizer-step counters.

Do not infer if early stopping occurs before50.

---

# 84. Runtime cost

Record:

```text
total runtime E50
total runtime E100
runtime beyond epoch50 if measurable
time to BEST.
```

Runtime is secondary.

---

# 85. Time-to-BEST is more informative than total runtime alone

If E100 wins with:

```text
best_epoch=63
```

report:

```text
time_to_best_E100
```

when available.

Do not imply entire run time was required to discover the selected checkpoint if early stopping continued afterward.

---

# 86. Overfitting-like late-training diagnostic

For E100, if:

```text
BEST <=50
```

and epochs >50 show worsening Validation RMSE, record:

```text
NO_POST50_GAIN
LATE_VALIDATION_DETERIORATION
```

without claiming universal overfitting.

---

# 87. Late improvement diagnostic

If:

```text
BEST >50
```

record:

```text
POST50_NEW_BEST.
```

This is the clearest evidence that E50 cap potentially truncated a better reachable checkpoint under the same training protocol.

---

# 88. Strong evidence E50 cap was performance-limiting

The strongest S16 pattern is:

```text
E50 hit MAX_EPOCHS
E100 prefix is consistent
E100 obtains strict new BEST after50
E100 final verified BEST RMSE < E50 BEST RMSE.
```

Safe conclusion:

> Under the frozen training protocol, the 50-epoch cap limited access to a better Validation checkpoint.

---

# 89. Evidence E50 cap was not performance-limiting

Examples:

```text
E50 early-stopped before50
or
E100 never reaches >50
or
E100 reaches >50 but no new global BEST
or
E100 exact same BEST as E50.
```

Safe:

> Extending the maximum epoch allowance to 100 did not improve the verified Validation optimum under the current early-stopping policy.

---

# 90. Boundary winner caution

If E100 wins and:

```text
best_epoch=100
or
stop_reason=MAX_EPOCHS at100
```

record:

```text
BOUNDARY_WINNER
E100_CAP_REACHED.
```

Do not extend search automatically.

---

# 91. E100 wins with BEST before50

If E100 reports lower BEST but:

```text
best_epoch <=50
```

than E50 reference, this is suspicious under a deterministic cap-only experiment.

Investigate:

```text
prefix mismatch
initialization mismatch
sample order
environment nondeterminism
hidden config differences.
```

Do not casually attribute the improvement to max_epochs.

---

# 92. Why this suspicious pattern matters

`max_epochs=100` should not affect computation at epoch20 when:

```text
no scheduler/warmup depends on cap
all other configs equal.
```

So a different pre50 best is evidence of stochastic/provenance differences, not of extra budget itself.

---

# 93. Handling legitimate nondeterminism

If environment is not bitwise deterministic:

```text
E100 may diverge slightly before50.
```

Then S16 remains a seed42 controlled run but the causal isolation is weaker.

Record:

```text
PREFIX_REPRODUCIBILITY_LIMITATION.
```

Do not claim pure deterministic continuation.

---

# 94. No multiple E100 seeds

S16 remains:

```text
seed42 only.
```

Do not run extra E100 seeds to average out nondeterminism here.

Phase46 handles multi-seed final runs.

---

# 95. No cherry-picking E100 reruns

If multiple technical reruns occur, use Registry policy and documented valid run resolution.

Do not choose the E100 rerun with the lowest RMSE.

---

# 96. Working hypothesis H-S16-01

```text
E50 may already be sufficient because patience10 terminates training before the maximum cap becomes relevant.
```

Status:

```text
UNTESTED.
```

---

# 97. H-S16-02

```text
If E50 hits its cap while Validation is still capable of new improvements, E100 may discover a better BEST checkpoint after epoch50.
```

Status:

```text
UNTESTED.
```

---

# 98. H-S16-03

```text
E100 may only increase compute without improving BEST Validation RMSE.
```

Status:

```text
UNTESTED.
```

---

# 99. H-S16-04

```text
A longer allowed budget may expose additional late-training instability or overfitting-like Validation deterioration.
```

Diagnostic hypothesis only.

---

# 100. Preconditions

Required:

```text
Phase37 = PASS
```

or:

```text
PASS_WITH_WARNING
```

with no unresolved critical issue.

Also:

```text
approved_for_phase38 = true.
```

---

# 101. Required upstream artifacts

```text
s15_loss_winner.json
s15_reference_update.json
phase_37_signoff.json
```

---

# 102. Required upstream contracts

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
SWEEP_S15_LOSS-v1
```

---

# 103. Carry-forward warnings

Propagate all unresolved non-critical warnings.

Examples:

```text
SMALL_SELECTION_MARGIN
METRIC_RANKING_DIVERGENCE
BOUNDARY_WINNER
SAMPLE_ORDER_NOT_VERIFIABLE
INITIALIZATION_NOT_VERIFIABLE
PREFIX_REPRODUCIBILITY_LIMITATION
RANDOM_CONTROL_GAIN.
```

Include in:

```text
S16 manifest
S16 summary
S16 report
S16 winner
Phase39 reference update.
```

---

# 104. Swept factor only

Canonical:

```text
max_epochs
```

Allowed:

```text
50
100
```

Aliases:

```text
E50
E100.
```

No E75/E150/E200.

---

# 105. Frozen data contract

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
same feature fingerprint
same X scaler
same target transform/scaler
same Validation ordering.
```

---

# 106. Frozen architecture contract

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

---

# 107. Frozen loss contract

Hard:

```text
LOSS*
```

from S15.

If Huber:

```text
delta=1.0 model-space
reduction=mean
same target-scaler checksum.
```

If MSE:

```text
reduction=mean.
```

---

# 108. Frozen optimizer contract

Hard:

```text
AdamW
LR*
WD*
same parameter-group policy
same betas/eps
same gradient accumulation
same precision policy.
```

---

# 109. Frozen stopping contract except cap

Hard:

```text
patience=10
min_delta=0
monitor=Validation RMSE Wh
strict improvement semantics
BEST selection=Validation RMSE Wh.
```

Only:

```text
max_epochs
```

changes.

---

# 110. E50 reference reuse gate

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
LOSS*
H1
WB0
WINDOWPOP-v1

E50
patience10
min_delta0
clip1
scheduler=None
warmup=None
accumulation1
seed42

TRAINING_ENGINE-v1
METRICS-v1.
```

If Huber:

```text
delta1 model-space
target-scaler checksum
```

must match.

Mismatch:

```text
STOP.
```

---

# 111. Fresh E100 run

Execution:

```text
register E100
↓
reseed42
↓
fresh Train DataLoader
↓
fresh Validation DataLoader
↓
fresh exact selected Transformer
↓
fresh exact selected criterion
↓
fresh AdamW(LR*,WD*)
↓
max_epochs=100
patience=10
clip1
↓
TRAINING_ENGINE-v1
```

No resume/warm-start.

---

# 112. Disposable preflight sanity

Architecture does not change, so no need for expensive new architecture sanity.

Still verify:

```text
model config fingerprint
criterion config
optimizer config
max_epochs=100
patience10
scheduler=None
selection metric RMSE Wh.
```

No official optimizer step before registered run.

---

# 113. Early-stop state initialization

At E100 epoch1:

```text
best_metric = uninitialized/+inf according to engine
best_epoch = none
non_improvement_counter = 0
```

same as E50 run-start semantics.

---

# 114. History must support 100 epochs

Artifact/history storage must not have hard-coded:

```text
50-row arrays
50 x-axis limits
epoch<=50 assertions.
```

Preflight code should verify dynamic handling through epoch100.

---

# 115. Plotting code must support epoch100

No plot truncation at 50.

Recommended:

```text
x-axis observed epoch range.
```

Mark:

```text
vertical line at epoch50
```

on E100 learning-curve figures to show the original cap boundary.

---

# 116. Registry config identity

Run config must explicitly distinguish:

```text
max_epochs=50
max_epochs=100
```

and no other controlled field.

Create a config-delta audit.

---

# 117. Parameter count equality hard expected

Required:

```text
Params_E50 == Params_E100.
```

Any model parameter difference is hidden drift.

---

# 118. State-dict schema equality hard expected

Same:

```text
model state_dict keys/shapes
buffers
architecture fingerprint.
```

---

# 119. Criterion config equality hard expected

Same selected loss.

Only training cap differs.

---

# 120. Initial model state matching

Because model architecture/seed are identical:

```text
fresh E100 initial-state fingerprint
```

should match E50 initial fingerprint if reference metadata exists.

If missing:

```text
NOT_VERIFIABLE.
```

---

# 121. Initial optimizer state matching

Fresh AdamW state before first step should be semantically identical.

Optimizer state dict may initially contain empty per-parameter state until first step.

Compare config/group semantics rather than fragile serialized ordering when appropriate.

---

# 122. DataLoader order matching

If E50 per-epoch order fingerprints exist:

```text
compare E100 first shared epochs.
```

If not:

```text
NOT_VERIFIABLE.
```

---

# 123. Learning-rate prefix audit

Every shared optimizer step:

```text
LR_E50 == LR_E100 == LR*
```

because scheduler=None.

Any difference is critical.

---

# 124. Validation metric prefix audit

For common epochs, compare full-precision:

```text
RMSE
MAE
R².
```

If exact reproducibility expected, require predeclared tolerance.

---

# 125. Train criterion prefix audit

Compare same objective-specific Train criterion values over common epochs.

If mismatch but Validation metrics match, investigate aggregation/provenance.

---

# 126. Clipping prefix audit

Optional but useful:

```text
fraction clipped by epoch
mean grad norm.
```

Matching adds confidence that optimization trajectories are equivalent before cap.

---

# 127. Stop reason consistency

If E50 early-stopped at epoch e<50 and E100 trajectory is deterministic-equivalent:

```text
E100 should also early-stop at the same e.
```

If E100 proceeds beyond e, investigate prefix/early-stop state mismatch.

---

# 128. Cap-nonbinding proof pattern

Strong:

```text
E50 EARLY_STOP <50
E100 same prefix
E100 EARLY_STOP same epoch
same BEST.
```

Conclusion:

```text
50-epoch cap is clearly non-binding under current protocol.
```

---

# 129. Cap-binding exploration pattern

If E50:

```text
MAX_EPOCHS at50
```

then E100 provides new evidence by allowing continuation.

Analyze:

```text
extra epochs
new BEST after50
best epoch
early-stop/cap at end.
```

---

# 130. E100 budget utilization classification

Allowed categories:

```text
NO_EXTRA_BUDGET_USED
PARTIAL_EXTRA_BUDGET_USED
FULL_EXTRA_BUDGET_USED.
```

Definitions:

```text
NO_EXTRA:
epochs_completed <=50

PARTIAL_EXTRA:
51..99

FULL_EXTRA:
100.
```

Separate from winner status.

---

# 131. E100 post50 benefit classification

Allowed:

```text
POST50_NOT_OBSERVED
POST50_NO_NEW_GLOBAL_BEST
POST50_NEW_GLOBAL_BEST
POST50_FINAL_WINNER.
```

`POST50_FINAL_WINNER` means:

```text
E100 selected winner
and E100 BEST epoch >50.
```

---

# 132. E50 cap status classification

Allowed:

```text
NON_BINDING_EARLY_STOP
REACHED_CAP_BEST_EARLY
REACHED_CAP_BEST_NEAR_BOUNDARY
REACHED_CAP_BEST_AT_BOUNDARY
UNKNOWN_INCOMPLETE_METADATA.
```

Descriptive categories must use explicit rules stored in code/report.

---

# 133. “Near boundary” rule

If used, predeclare:

```text
K = min(5, E50_epochs_completed)
near_boundary = best_epoch >= E50_epochs_completed-K+1.
```

This is diagnostic only, not selection.

---

# 134. Tail trend diagnostic

Optional descriptive metrics for E50 and E100:

```text
last K Validation RMSE values
tail min/max
tail slope from simple least-squares
```

with:

```text
K=min(5,N).
```

Do not use slope as winner criterion or statistical proof.

---

# 135. No smoothing for selection

Learning curves may be visually smoothed only if raw trajectory remains visible and smoothing is clearly diagnostic.

Never choose BEST from smoothed curve.

---

# 136. Primary metric

Hard:

```text
verified BEST Validation RMSE Wh.
```

---

# 137. Secondary scientific metrics

Record:

```text
Validation MAE Wh
Validation R²
best epoch
epochs completed
stop reason
epochs after best
best beyond50?
post50 new BEST count
Validation trajectories.
```

---

# 138. Optimization diagnostics

Record:

```text
optimizer steps
steps to best
gradient norms
clipping fraction
patience counter where available
late-training behavior.
```

---

# 139. Engineering diagnostics

Record:

```text
total runtime
runtime to best
extra runtime after50
checkpoint sizes
history size
optional peak memory.
```

Memory should not materially change due cap for a single step, though total execution time does.

Do not claim peak memory doubles with E100.

---

# 140. Peak memory expectation

At fixed batch/lookback/model:

```text
peak training memory should be similar
```

between E50/E100 because epochs execute sequentially and optimizer/model states are same size.

Small backend allocator differences are possible.

This is not a selection metric.

---

# 141. Disk/storage growth

Longer history/checkpoint logging may increase:

```text
history file size
optional milestone snapshot storage
logs.
```

Model checkpoint tensor size itself should be same.

---

# 142. BEST checkpoint file-size equality expectation

If checkpoint payload structure is the same, model/optimizer state size should be broadly similar across E50/E100.

Differences can come from:

```text
history metadata
RNG payload
serialization details.
```

Do not treat file-byte equality as a hard scientific requirement.

---

# 143. Epoch100 OOM is unlikely to be cap-caused

Because each epoch has same tensor geometry, a true OOM appearing only late suggests:

```text
memory leak
retained graph
logging accumulation
attention tensor retention
backend fragmentation.
```

Investigate implementation.

Do not reduce batch for E100.

---

# 144. Long-run memory-leak audit

Recommended E100 diagnostic:

```text
memory at selected epoch milestones:
1
10
25
50
75
100 if reached
```

where backend supports it.

Look for monotonic unexplained growth.

Diagnostic only.

---

# 145. No attention tensor retention

Official training remains:

```text
need_weights=False.
```

A 100-epoch run must not accidentally accumulate attention tensors in history.

---

# 146. No prediction-history retention for Train

Do not store every batch prediction across 100 epochs unless already required.

Use aggregate training statistics.

Validation predictions per epoch should be released after metrics unless explicitly needed.

---

# 147. Checkpoint write policy

Continue:

```text
BEST on strict improvement
LAST every epoch
atomic writes
SHA256.
```

Longer run must not accumulate 100 full checkpoints unless policy explicitly requests it.

---

# 148. No best-checkpoint deletion during longer run

When a new BEST occurs:

```text
atomically replace/update BEST
```

according to engine.

Final BEST must be loadable after early stopping.

---

# 149. E100 BEST reload verification

After official run:

```text
fresh exact model
↓
strict-load E100 BEST
↓
model.eval()
↓
full ordered Validation
↓
inverse transform
↓
METRICS-v1
↓
verify recorded BEST metrics.
```

---

# 150. E50 reference verification

No retraining.

Verify:

```text
exact S15 winner
E50
same selected loss
same model/data/optimizer
same metric version
BEST already verified.
```

---

# 151. Prefix-state optional checkpoint verification

If E50 and E100 both have an epoch-k snapshot for a common k:

```text
compare model state fingerprint
optimizer state fingerprint
early-stop state
```

when deterministic reproduction is expected.

This is optional but very strong.

---

# 152. No reference mutation

Do not modify E50 artifacts/history/checkpoint in place to add S16 metadata.

S16 may create derived audit artifacts pointing to E50.

Upstream run remains immutable.

---

# 153. Same experiment registry semantics

E100 gets a new run ID.

Do not rename E50 run or change its original registered config.

---

# 154. Reuse provenance

S16 must record:

```text
E50 source_phase = 37
E50 source_type = REUSED_REFERENCE
E100 source_type = NEW_RUN.
```

---

# 155. Run failure policy

If E100 has a technical failure:

```text
S16 incomplete
```

until a documented technical rerun succeeds.

Do not automatically declare E50 winner.

---

# 156. Technical rerun allowed

Only for documented:

```text
process interruption
hardware/software failure
corrupt checkpoint
artifact-write failure.
```

Use Experiment Registry rerun rules.

---

# 157. Resume after infrastructure interruption

If E100 is interrupted and engine supports strict resume:

```text
resume only from valid LAST
with model
optimizer
epoch
early-stop state
RNG state
DataLoader generator state.
```

This is a technical continuation of the same registered run, not a new scientific condition.

---

# 158. Resume must not reset patience

Critical.

A resumed E100 run must continue:

```text
best metric
best epoch
non-improvement counter
```

exactly.

---

# 159. Resume must not reset LR/optimizer state

Critical.

---

# 160. Score-based rerun forbidden

Do not restart E100 from epoch1 because:

```text
late RMSE looks poor
no post50 improvement
best is early.
```

---

# 161. E100 early-stop is not a failure

If E100 early-stops:

```text
that is a valid scientific outcome.
```

Do not force it to reach 100.

---

# 162. E100 not reaching 100 is often the point

The question is whether higher cap helps under the same early-stopping policy.

If patience stops earlier:

```text
higher cap was unnecessary.
```

This is valid evidence.

---

# 163. Discrepancy taxonomy

```text
S15_REFERENCE_MISSING
S15_WINNER_MISMATCH
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
LOSS_DRIFT
HUBER_DELTA_DRIFT
EPOCH_CAP_DEFINITION_MISMATCH
PATIENCE_DRIFT
MIN_DELTA_DRIFT
EARLY_STOP_METRIC_DRIFT
BEST_SELECTION_METRIC_DRIFT
SCHEDULER_ADDED
CAP_DEPENDENT_SCHEDULER
WARMUP_ADDED
GRADIENT_CLIP_DRIFT
OPTIMIZER_DRIFT
LR_DRIFT
WD_DRIFT
MODEL_ARCHITECTURE_DRIFT
PARAMETER_COUNT_MISMATCH
STATE_DICT_SCHEMA_MISMATCH
INITIALIZATION_POLICY_DRIFT
INITIAL_STATE_MISMATCH
SAMPLE_ORDER_POLICY_DRIFT
PREFIX_TRAJECTORY_MISMATCH
PREFIX_STATE_MISMATCH
RNG_POLICY_DRIFT
MANUAL_RESEED_AT_EPOCH50
DATALOADER_RESET_AT_EPOCH50
OPTIMIZER_RESET_AT_EPOCH50
EARLY_STOP_RESET_AT_EPOCH50
BEST_RESET_AT_EPOCH50
CONTINUE_FROM_BEST_USED
NON_STRICT_RESUME_USED
POPULATION_MISMATCH
TARGET_ID_MISMATCH
X_SCALER_MISMATCH
TARGET_SCALER_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
REFERENCE_RUN_MISMATCH
RUN_FAILURE
NUMERICAL_INSTABILITY
MEMORY_LEAK_SUSPECTED
CHECKPOINT_CONFIG_MISMATCH
CHECKPOINT_VERIFICATION_FAILURE
INCOMPLETE_SWEEP
RANKING_ERROR
EFFECT_CALCULATION_ERROR
RMSE_R2_RANKING_INCONSISTENCY
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
HIDDEN_CAP_EXTENSION
OTHER
```

---

# 164. Status model

## PASS

```text
E50 reference valid
E100 fresh run valid
only max_epochs differs
same early stopping/optimizer/model/data
BEST checkpoints verified
cap-binding diagnostics valid
winner selected
Phase39 reference generated
Test untouched.
```

## PASS_WITH_WARNING

Possible:

```text
tiny RMSE margin
metric ranking divergence
prefix match not verifiable
backend nondeterminism
E100 boundary winner
E100 cap reached
suspected late memory growth
inherited warning.
```

## FAIL

Examples:

```text
patience changed
scheduler tied to cap
E100 resumed from E50 BEST
counter reset at50
population mismatch
hidden architecture/optimizer drift
unresolved E100 failure
Test access.
```

---

# 165. Output directory

```text
artifacts/
└── sweeps/
    └── S16_epoch_cap/
        ├── s16_epoch_cap_sweep_manifest.json
        ├── s16_epoch_cap_sweep_contract.json
        ├── s16_epoch_cap_preflight_audit.csv
        ├── s16_run_matrix.csv
        ├── s16_epoch_cap_definition_audit.csv
        ├── s16_early_stopping_contract_audit.csv
        ├── s16_scheduler_independence_audit.csv
        ├── s16_architecture_invariance_audit.csv
        ├── s16_parameter_schema_audit.csv
        ├── s16_training_config_delta_audit.csv
        ├── s16_common_data_audit.csv
        ├── s16_initialization_audit.csv
        ├── s16_sample_order_audit.csv
        ├── s16_prefix_reproducibility_audit.csv
        ├── s16_prefix_metric_differences.csv
        ├── s16_stop_reason_audit.csv
        ├── s16_cap_binding_audit.csv
        ├── s16_extra_budget_utilization.csv
        ├── s16_post50_improvement_diagnostics.csv
        ├── s16_patience_state_audit.csv
        ├── s16_optimizer_budget_audit.csv
        ├── s16_mode_transition_audit.csv
        ├── s16_gradient_diagnostics.csv
        ├── s16_memory_leak_diagnostics.csv
        ├── s16_epoch_cap_run_provenance.csv
        ├── s16_epoch_cap_metrics.csv
        ├── s16_epoch_cap_effect.csv
        ├── s16_learning_curve_diagnostics.csv
        ├── s16_convergence_diagnostics.csv
        ├── s16_runtime_budget_diagnostics.csv
        ├── s16_generalization_diagnostics.csv
        ├── s16_hypothesis_outcomes.csv
        ├── s16_epoch_cap_findings.csv
        ├── s16_epoch_cap_winner.json
        ├── s16_reference_update.json
        ├── s16_epoch_cap_sweep_tests.csv
        ├── s16_epoch_cap_discrepancies.json
        ├── s16_epoch_cap_sweep_summary.json
        ├── s16_epoch_cap_sweep_report.md
        ├── figures/
        │   ├── S16_01_validation_rmse_by_epoch.png
        │   ├── S16_02_validation_mae_by_epoch.png
        │   ├── S16_03_train_criterion_by_epoch.png
        │   ├── S16_04_prefix_rmse_difference.png
        │   ├── S16_05_post50_validation_rmse.png
        │   ├── S16_06_gradient_clipping_fraction.png
        │   ├── S16_07_best_validation_metrics.png
        │   ├── S16_08_epoch_budget_utilization.png
        │   ├── S16_09_runtime_vs_best_rmse.png
        │   ├── S16_10_convergence_summary.png
        │   └── S16_11_generalization_gap_optional.png
        ├── README_S16_EPOCH_CAP_SWEEP.md
        └── phase_38_signoff.json
```

Optional:

```text
s16_memory_leak_diagnostics.csv
s16_generalization_diagnostics.csv
S16_11_generalization_gap_optional.png
epoch50 milestone checkpoint
prefix state-level audit
```

may be unavailable, but omission must be documented.

New E100 run stays under:

```text
artifacts/runs/<run_id>/
```

No duplicate scientific checkpoint trees in sweep folder.

---

# 166. Required outputs

```text
O38.1  Sweep manifest
O38.2  Sweep contract
O38.3  Preflight audit
O38.4  Run matrix
O38.5  Epoch-cap definition audit
O38.6  Early-stopping contract audit
O38.7  Scheduler-independence audit
O38.8  Architecture invariance audit
O38.9  Parameter-schema audit
O38.10 Training config delta audit
O38.11 Common-data audit
O38.12 Initialization audit
O38.13 Sample-order audit
O38.14 Prefix reproducibility audit
O38.15 Prefix metric-difference table
O38.16 Stop-reason audit
O38.17 Cap-binding audit
O38.18 Extra-budget utilization
O38.19 Post-50 improvement diagnostics
O38.20 Patience-state audit
O38.21 Optimizer-budget audit
O38.22 Mode-transition audit
O38.23 Gradient diagnostics
O38.24 Optional memory-leak diagnostics
O38.25 Run provenance
O38.26 Reused E50 reference
O38.27 Verified new E100 run
O38.28 Primary metrics table
O38.29 Epoch-cap effect table
O38.30 Learning-curve diagnostics
O38.31 Convergence diagnostics
O38.32 Runtime-budget diagnostics
O38.33 Optional generalization diagnostics
O38.34 Hypothesis outcomes
O38.35 Findings
O38.36 Winner artifact
O38.37 Phase39 reference update
O38.38 Figures
O38.39 Sweep tests
O38.40 Discrepancy log
O38.41 Sweep summary
O38.42 Human-readable report
O38.43 README
O38.44 Phase sign-off
```

---

# 167. Run matrix

Create:

```text
s16_run_matrix.csv
```

Fields:

```text
sweep_id
epoch_cap_id
max_epochs
patience
min_delta
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
loss_id
loss_name
huber_delta_if_applicable
gradient_clip
scheduler
warmup
population_fingerprint
feature_fingerprint
seed
model_config_id
criterion_config_id
training_config_id
status
```

---

# 168. Sweep manifest

Create:

```text
s16_epoch_cap_sweep_manifest.json
```

Minimum:

```text
sweep_version = SWEEP_S16_EPOCHCAP-v1
sweep_id = S16_EPOCH_CAP
source_s15_winner_run_id
selected_model_config
selected_loss_config
candidate_epoch_caps = [50,100]
fixed_patience = 10
fixed_min_delta = 0
fixed_early_stop_metric = validation_rmse_wh
fixed_best_metric = validation_rmse_wh
scheduler = none
warmup = none
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
loss_id
huber_delta_if_applicable
gradient_clip = 1.0
new_runs_required = 1
reused_runs = 1
swept_field = max_epochs
architecture_equality_expected = true
parameter_count_equality_expected = true
prefix_equivalence_expected_if_reproducible = true
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = E50_ON_EXACT_RMSE_TIE
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

# 169. Sweep contract

Create:

```text
s16_epoch_cap_sweep_contract.json
```

Must state:

```text
Only max_epochs changes.

E50=50.
E100=100.

max_epochs is a ceiling, not guaranteed actual epochs.

patience=10 fixed.
min_delta=0 fixed.
early-stop metric=Validation RMSE Wh fixed.
BEST metric=Validation RMSE Wh fixed.

scheduler=None.
warmup=None.

Same model/loss/data/optimizer/clip/seed.

E50 reference reused.
E100 fresh seed42 run from epoch1.

No resume from E50 BEST.
No patience reset at epoch50.
No optimizer/RNG/DataLoader reset at epoch50.
No cap-dependent schedule.

Prefix trajectories should match when reproducibility permits.

Validation RMSE Wh selects winner.
Exact tie → E50.
No Test.
```

---

# 170. Preflight audit

`s16_epoch_cap_preflight_audit.csv` checks:

```text
phase37_pass
approved_for_phase38
s15_winner_valid
all prior selected fields locked
E50 registered
E100 registered
patience10 fixed
min_delta0 fixed
early_stop_metric fixed
BEST metric fixed
scheduler none
warmup none
clip1 fixed
loss fixed
Huber delta fixed if applicable
same architecture
same parameter count expected
same population
same seed
Training Engine fixed
Metric version fixed
history supports100
plotting supports100
Test lock
status
```

---

# 171. Epoch-cap definition audit

`s16_epoch_cap_definition_audit.csv`:

```text
epoch_cap_id
max_epochs
patience
actual_epochs_guaranteed
early_stopping_active
registered
status
```

Expected:

```text
actual_epochs_guaranteed = false.
```

---

# 172. Early-stopping contract audit

`s16_early_stopping_contract_audit.csv`:

```text
epoch_cap_id
patience
min_delta
monitor_metric
direction
strict_improvement
counter_reset_rule
trigger_rule
best_checkpoint_metric
tie_policy
same_across_conditions
status
```

---

# 173. Scheduler-independence audit

`s16_scheduler_independence_audit.csv`:

```text
epoch_cap_id
scheduler
warmup
lr_depends_on_max_epochs
optimizer_step_depends_on_total_budget
lr_initial
lr_expected_constant
status
```

Hard:

```text
lr_depends_on_max_epochs=false.
```

---

# 174. Architecture invariance audit

`s16_architecture_invariance_audit.csv`:

```text
component
config_e50
config_e100
shape_e50
shape_e100
equal
status
```

All components equal.

---

# 175. Parameter-schema audit

`s16_parameter_schema_audit.csv`:

```text
parameter_or_buffer
shape_e50
shape_e100
dtype_equal
trainable_equal
equal
status
```

Expected all equal.

---

# 176. Training config delta audit

`s16_training_config_delta_audit.csv`:

```text
field
value_e50
value_e100
allowed_to_differ
expected_difference
status
```

Only:

```text
max_epochs
```

may differ.

---

# 177. Common-data audit

`s16_common_data_audit.csv`:

```text
split
sample_count_e50
sample_count_e100
sample_ids_equal
ordered_ids_equal
feature_fingerprint_equal
x_scaler_equal
target_transform_equal
lookback_equal
batch_equal
population_equal
status
```

---

# 178. Initialization audit

`s16_initialization_audit.csv`:

```text
epoch_cap_id
seed
initialization_policy_fingerprint
initial_state_fingerprint
reference_available
exact_match_expected
match
status
```

---

# 179. Sample-order audit

`s16_sample_order_audit.csv`:

```text
epoch_number
e50_order_fingerprint
e100_order_fingerprint
reference_available
common_epoch
same_order
status
```

Only common epochs.

---

# 180. Prefix reproducibility audit

`s16_prefix_reproducibility_audit.csv`:

```text
comparison_level
common_epoch_start
common_epoch_end
backend
determinism_status
atol
rtol
initial_state_match
sample_order_match
lr_match
train_criterion_match
validation_rmse_match
validation_mae_match
validation_r2_match
gradient_summary_match_optional
state_match_optional
overall_prefix_status
notes
```

Allowed overall:

```text
MATCH
MATCH_WITH_TOLERANCE
NOT_VERIFIABLE
MISMATCH_CRITICAL
MISMATCH_NONDETERMINISTIC_LIMITATION.
```

---

# 181. Prefix metric differences

`s16_prefix_metric_differences.csv`:

```text
epoch_number
rmse_e50
rmse_e100
rmse_abs_diff
mae_e50
mae_e100
mae_abs_diff
r2_e50
r2_e100
r2_abs_diff
train_criterion_e50
train_criterion_e100
train_criterion_abs_diff
within_tolerance
status
```

---

# 182. Stop-reason audit

`s16_stop_reason_audit.csv`:

```text
epoch_cap_id
max_epochs
epochs_completed
stop_reason
early_stopped
cap_reached
best_epoch
epochs_after_best
patience_counter_at_end_if_available
valid_stop_state
status
```

---

# 183. Cap-binding audit

`s16_cap_binding_audit.csv`:

```text
epoch_cap_id
max_epochs
epochs_completed
stop_reason
cap_reached
best_epoch
best_at_boundary
best_near_boundary
early_stop_before_cap
cap_binding_classification
status
```

---

# 184. Extra-budget utilization

`s16_extra_budget_utilization.csv`:

```text
e100_epochs_completed
epochs_beyond50
optimizer_steps_beyond50
runtime_beyond50_optional
budget_utilization_class
extra_budget_used
status
```

---

# 185. Post50 improvement diagnostics

`s16_post50_improvement_diagnostics.csv`:

```text
e100_run_id
observed_epochs_1_50
observed_epochs_51_plus
best_rmse_1_50
best_epoch_1_50
best_rmse_51_plus
best_epoch_51_plus
post50_global_new_best
first_new_best_after50
new_best_count_after50
post50_gain_wh
post50_gain_pct
final_best_after50
status
```

Use:

```text
N/A
```

when post50 epochs were not observed.

---

# 186. Patience-state audit

`s16_patience_state_audit.csv`:

```text
epoch_cap_id
epoch_number
validation_rmse_wh
global_best_rmse_so_far
global_best_epoch_so_far
strict_improvement
non_improvement_counter
early_stop_triggered
status
```

At minimum record:

```text
final E50 boundary state
final E100 state
```

if complete per-epoch reconstruction unavailable.

---

# 187. Optimizer budget audit

`s16_optimizer_budget_audit.csv`:

```text
epoch_cap_id
train_samples_per_epoch
batch_size
steps_per_complete_epoch
epochs_completed
total_optimizer_steps
best_epoch
steps_to_best
steps_after_best
steps_beyond50
status
```

---

# 188. Mode-transition audit

`s16_mode_transition_audit.csv`:

```text
epoch_number
train_started_in_train_mode
validation_started_in_eval_mode
next_epoch_train_mode_restored
attention_collection_off
status
```

Can be sampled/aggregated rather than storing 100 verbose rows if engine logs mode transitions robustly.

---

# 189. Gradient diagnostics

`s16_gradient_diagnostics.csv`:

```text
epoch_cap_id
epoch_number
mean_preclip_grad_norm
max_preclip_grad_norm
clipped_batches
total_batches
clipping_fraction
nonfinite_events
epoch_segment
status
```

`epoch_segment`:

```text
PRE50
POST50.
```

---

# 190. Optional memory-leak diagnostics

`s16_memory_leak_diagnostics.csv`:

```text
epoch_number
device
allocated_memory_optional
reserved_memory_optional
rss_memory_optional
milestone
unexpected_monotonic_growth
status
```

Milestones:

```text
1
10
25
50
75
100
```

when reached.

---

# 191. Run provenance

`s16_epoch_cap_run_provenance.csv`:

```text
epoch_cap_id
max_epochs
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
initial_state_fingerprint
sample_order_provenance
best_checkpoint_sha256
last_checkpoint_sha256
history_sha256
metric_artifact
stop_reason
status
```

---

# 192. Primary metrics table

`s16_epoch_cap_metrics.csv`:

```text
epoch_cap_id
max_epochs
run_id
source_type
epochs_completed
stop_reason
best_epoch
epochs_after_best
best_after50
trainable_parameters
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

# 193. Epoch-cap effect table

`s16_epoch_cap_effect.csv`:

```text
e50_run_id
e100_run_id
e50_rmse_wh
e100_rmse_wh
rmse_delta_e50_to_e100_wh
rmse_improvement_pct
e50_mae_wh
e100_mae_wh
mae_delta_wh
e50_r2
e100_r2
r2_delta
rmse_mae_ranking_divergence
rmse_r2_ranking_consistent
e50_epochs_completed
e100_epochs_completed
e100_best_after50
winner
status
```

---

# 194. Learning-curve diagnostics

`s16_learning_curve_diagnostics.csv`:

```text
epoch_cap_id
epoch_number
train_criterion
validation_rmse_wh
validation_mae_wh
validation_r2
is_global_best
patience_counter_optional
epoch_segment
status
```

---

# 195. Convergence diagnostics

`s16_convergence_diagnostics.csv`:

```text
epoch_cap_id
first_epoch_rmse_wh
best_epoch
best_rmse_wh
last_epoch
last_rmse_wh
early_stopped
cap_reached
epochs_after_best
steps_to_best
post50_new_best
tail_k
tail_rmse_slope_optional
status
```

---

# 196. Runtime-budget diagnostics

`s16_runtime_budget_diagnostics.csv`:

```text
epoch_cap_id
device
epochs_completed
total_optimizer_steps
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
time_to_best_optional
runtime_after50_optional
steps_after50
extra_budget_cost
runtime_comparable
status
```

---

# 197. Optional generalization diagnostics

`s16_generalization_diagnostics.csv`:

```text
epoch_cap_id
train_rmse_wh_at_best
validation_rmse_wh_at_best
rmse_gap_wh
train_mae_wh_at_best
validation_mae_wh_at_best
mae_gap_wh
best_epoch
status
```

No Test.

---

# 198. Hypothesis outcomes

`s16_hypothesis_outcomes.csv`:

```text
hypothesis_id
comparison_or_pattern
expected_direction_or_pattern
observed_metrics
cap_binding_context
post50_context
outcome
interpretation
status
```

Allowed:

```text
SUPPORTED
NOT_SUPPORTED
INCONCLUSIVE.
```

---

# 199. Findings artifact

`s16_epoch_cap_findings.csv` possible codes:

```text
E50_GAIN
E100_GAIN
EPOCH_CAP_EXACT_TIE
E50_CAP_NON_BINDING
E50_CAP_REACHED
E50_BEST_AT_BOUNDARY
E50_BEST_NEAR_BOUNDARY
EXTRA_BUDGET_UNUSED
PARTIAL_EXTRA_BUDGET_USED
FULL_EXTRA_BUDGET_USED
POST50_NOT_OBSERVED
POST50_NO_NEW_GLOBAL_BEST
POST50_NEW_GLOBAL_BEST
POST50_FINAL_WINNER
E100_CAP_REACHED
BOUNDARY_WINNER
PREFIX_MATCH_VERIFIED
PREFIX_MATCH_WITH_TOLERANCE
PREFIX_MATCH_NOT_VERIFIABLE
PREFIX_TRAJECTORY_MISMATCH
PREFIX_REPRODUCIBILITY_LIMITATION
LATE_VALIDATION_DETERIORATION
LATE_GRADIENT_INSTABILITY
METRIC_RANKING_DIVERGENCE
RMSE_R2_RANKING_INCONSISTENCY
RUNTIME_COST_INCREASE
MEMORY_LEAK_SUSPECTED
INHERITED_WARNING.
```

---

# 200. Winner artifact

`s16_epoch_cap_winner.json` minimum:

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
loss_id
loss_name
huber_delta_if_applicable
patience
min_delta
selection_metric
selection_direction
tie_rule
winner_epoch_cap_id
winner_max_epochs
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
winner_epochs_completed
winner_best_epoch
runner_up_epoch_cap_id
runner_up_max_epochs
runner_up_rmse_wh
rmse_margin_wh
rmse_margin_pct
e50_cap_binding_classification
e100_extra_budget_utilization
e100_post50_new_best
prefix_reproducibility_status
winner_is_boundary
population_fingerprint
metric_version
inherited_warnings
test_status
status
```

---

# 201. Phase39 reference update

`s16_reference_update.json`:

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
loss_id
loss_name
huber_delta_if_applicable
previous_max_epochs=50
selected_epoch_cap_id
selected_max_epochs
winner_run_id
winner_config_fingerprint
winner_rmse_wh
current_patience=10
current_gradient_clip_id=GC1
current_gradient_clip_max_norm=1.0
selection_metric
population_fingerprint
inherited_warnings
approved_for_phase39
```

---

# 202. Phase39 handoff logic

Phase39 tests:

```text
GC0 = gradient clipping OFF
GC1 = max_norm 1.0
```

while holding S16-selected epoch cap fixed.

S16 winner already uses:

```text
GC1.
```

Therefore normally:

```text
reuse S16 winner as GC1 reference
train only GC0.
```

if exact match.

---

# 203. Important Phase39 handoff nuance

If E100 wins:

```text
Phase39 uses max_epochs=100
patience10.
```

If E50 wins:

```text
Phase39 uses max_epochs=50
patience10.
```

Do not revert to E50 automatically.

---

# 204. Figures

Recommended:

```text
S16_01_validation_rmse_by_epoch.png
S16_02_validation_mae_by_epoch.png
S16_03_train_criterion_by_epoch.png
S16_04_prefix_rmse_difference.png
S16_05_post50_validation_rmse.png
S16_06_gradient_clipping_fraction.png
S16_07_best_validation_metrics.png
S16_08_epoch_budget_utilization.png
S16_09_runtime_vs_best_rmse.png
S16_10_convergence_summary.png
S16_11_generalization_gap_optional.png
```

---

# 205. Primary figure

`S16_01_validation_rmse_by_epoch.png`:

```text
E50 curve
E100 curve
BEST markers
vertical line at epoch50
actual stop markers.
```

If E50 early-stops before50, its curve ends naturally.

Do not pad it to50.

---

# 206. Prefix-difference figure

`S16_04_prefix_rmse_difference.png`:

```text
x = common epoch
y = RMSE_E100 - RMSE_E50
```

with zero reference line.

Only if prefix comparison is meaningful.

Label tolerance/reproducibility status.

---

# 207. Post50 figure

`S16_05_post50_validation_rmse.png` should focus on:

```text
epochs 45 onward
```

or observed late window and show:

```text
original E50 cap boundary
E100 new BEST events.
```

Diagnostic only.

---

# 208. Epoch-budget utilization figure

`S16_08_epoch_budget_utilization.png` may show:

```text
cap
actual epochs completed
best epoch
```

for E50/E100.

This visually distinguishes:

```text
allowed budget
used budget
useful budget.
```

---

# 209. Safe interpretation if E50 wins

Possible:

> Extending the maximum epoch allowance from 50 to 100 did not improve the verified Validation RMSE under the fixed patience-10 early-stopping policy, so the 50-epoch cap was retained.

If E50 early-stopped before50, additionally:

> The 50-epoch cap was non-binding because Early Stopping terminated training before the cap.

---

# 210. Safe interpretation if E100 wins with BEST >50

> Under the fixed training protocol, E100 discovered a new Validation BEST after epoch50 and achieved lower verified RMSE than the E50 reference, indicating that the 50-epoch ceiling constrained access to a better checkpoint.

Conditional on prefix/provenance quality.

---

# 211. Safe interpretation if E100 wins with BEST <=50

Do **not** say:

```text
extra epochs improved performance.
```

Instead:

> E100 produced a lower Validation BEST within the first 50 epochs; because max_epochs should not affect the pre-50 trajectory under the frozen no-scheduler protocol, this difference requires interpretation in light of stochastic/reproducibility differences rather than the additional epoch allowance itself.

This pattern should carry a warning.

---

# 212. Safe interpretation if E100 reaches100 and wins at100

> E100 achieved the best registered Validation RMSE, but the optimum occurred at the upper boundary of the tested budget. The result therefore supports E100 within the registered S16 range without establishing that 100 epochs is globally sufficient.

Do not auto-test E200.

---

# 213. Safe interpretation if exact tie

> E50 and E100 produced exactly equal full-precision Validation BEST RMSE; E50 was retained by the predefined lower-budget parsimony rule.

---

# 214. Interpretation prohibitions

Do not claim:

```text
E100 trains exactly twice as long
E100 always has twice the optimizer steps
E100 is better because it trained longer
E50 is undertrained solely because it hit epoch50
E100 is overfit solely because it has a larger cap
later last-epoch RMSE determines winner
E100 pre50 improvement is caused by max_epochs
100 epochs is globally optimal
```

without evidence.

---

# 215. Sweep summary

Create:

```text
s16_epoch_cap_sweep_summary.json
```

Structure:

```text
sweep_id
sweep_version
selected_model_config
selected_loss_config
reference_run_id
e100_run_id
new_runs
reused_runs
candidate_epoch_caps
fixed_patience
fixed_min_delta
primary_metric
metrics_by_cap
epoch_cap_effect
stop_reasons
e50_cap_binding
e100_budget_utilization
post50_improvement
prefix_reproducibility
optimizer_budget
learning_curve_diagnostics
gradient_diagnostics
convergence_diagnostics
runtime_budget_diagnostics
winner
winner_margin
winner_is_boundary
inherited_warnings
phase39_reference
test_status
overall_status
```

---

# 216. Human-readable report

Create:

```text
s16_epoch_cap_sweep_report.md
```

Sections:

```text
1. Objective
2. Current reference from S15
3. E50/E100 definitions
4. Max epochs vs actual epochs distinction
5. Fixed early-stopping contract
6. Scheduler-independence proof
7. Frozen-variable contract
8. Same model/data/optimizer invariance
9. E50 reference provenance
10. E100 run provenance
11. Prefix reproducibility audit
12. Stop-reason analysis
13. E50 cap-binding analysis
14. E100 extra-budget utilization
15. Post-50 new-BEST analysis
16. Validation metrics
17. Epoch-cap effect
18. Learning-curve/convergence diagnostics
19. Gradient/clipping diagnostics
20. Runtime-budget cost
21. Optional memory-leak/generalization diagnostics
22. S16 winner
23. Interpretation cautions
24. Sequential/single-seed limitations
25. Phase39 handoff
```

---

# 217. README

Create:

```text
README_S16_EPOCH_CAP_SWEEP.md
```

Must explain:

```text
Purpose
S15 winner handoff
E50/E100 definitions
max_epochs as ceiling
patience10 unchanged
min_delta0 unchanged
RMSE-based early stop/BEST
scheduler=None significance
fresh E100 from epoch1
why not continue from E50 BEST
no counter/optimizer/RNG reset at50
prefix-equivalence concept
cap-binding classification
extra-budget utilization
post50 new BEST logic
winner/tie rules
E100 boundary caution
single-seed/nondeterminism limitations
Phase39 handoff
No Test.
```

---

# 218. Recommended notebook structure

```text
Cell 38.1  Phase title
Cell 38.2  Verify Phase37 sign-off
Cell 38.3  Declare SWEEP_S16_EPOCHCAP-v1
Cell 38.4  Load S15 winner/reference
Cell 38.5  Freeze all selected fields
Cell 38.6  Define E50/E100
Cell 38.7  Lock patience10/min_delta0
Cell 38.8  Audit scheduler=None/warmup=None
Cell 38.9  Build run matrix
Cell 38.10 Audit architecture/parameter invariance
Cell 38.11 Audit same loss/optimizer/clip
Cell 38.12 Audit common population
Cell 38.13 Inspect E50 stop reason/cap binding
Cell 38.14 Audit E50 reference reuse
Cell 38.15 Audit history/plot support through100
Cell 38.16 Register E100 run
Cell 38.17 Reseed + fresh E100 loaders/model/optimizer
Cell 38.18 Execute E100 via TRAINING_ENGINE-v1
Cell 38.19 Verify E100 BEST
Cell 38.20 Build run provenance
Cell 38.21 Build initialization audit
Cell 38.22 Build sample-order audit
Cell 38.23 Build prefix reproducibility audit
Cell 38.24 Build prefix metric differences
Cell 38.25 Build stop-reason audit
Cell 38.26 Build cap-binding audit
Cell 38.27 Build extra-budget utilization
Cell 38.28 Build post50 improvement diagnostics
Cell 38.29 Build patience-state audit
Cell 38.30 Build optimizer-budget audit
Cell 38.31 Build mode-transition audit
Cell 38.32 Build gradient diagnostics
Cell 38.33 Optional memory-leak diagnostics
Cell 38.34 Build primary metrics
Cell 38.35 Compute E50→E100 effect
Cell 38.36 Build learning-curve diagnostics
Cell 38.37 Build convergence diagnostics
Cell 38.38 Build runtime-budget diagnostics
Cell 38.39 Optional generalization diagnostic
Cell 38.40 Evaluate hypotheses
Cell 38.41 Generate figures
Cell 38.42 Generate findings
Cell 38.43 Select winner
Cell 38.44 Write winner JSON
Cell 38.45 Write Phase39 reference update
Cell 38.46 Run S16 tests/discrepancies
Cell 38.47 Write summary/report
Cell 38.48 Register artifacts/checksums
Cell 38.49 Write README
Cell 38.50 Phase sign-off
```

---

# 219. Execution flow

```text
Verify Phase37
→ Load S15 winner
→ Freeze model/data/loss/optimizer/patience/clip
→ Define E50 vs E100
→ Prove scheduler independent of cap
→ Inspect E50 stop reason
→ Reuse E50
→ Register E100
→ Seed42 + fresh official objects
→ Train E100 from epoch1
→ Verify E100 BEST
→ Audit shared prefix
→ Classify E50 cap binding
→ Measure E100 extra-budget usage
→ Detect post50 new BEST
→ Compare verified BEST Validation RMSE
→ Select minimum RMSE
→ Exact tie prefer E50
→ Update Phase39 reference
→ Write artifacts/sign-off
```

---

# 220. Fail-fast order

Before expensive E100 training:

```text
1. Phase37 sign-off
2. S15 winner identity
3. freeze all prior selected fields
4. E50/E100 registry
5. patience10
6. min_delta0
7. early-stop metric RMSE Wh
8. BEST metric RMSE Wh
9. scheduler=None
10. warmup=None
11. same selected loss
12. same architecture/parameter schema
13. same optimizer/LR/WD
14. same clip1
15. same population
16. same seed/init policy
17. history supports100
18. plotting supports100
19. E50 reuse eligibility
20. Test firewall
21. Registry readiness
```

---

# 221. Critical technical note: max_epochs is not actual epochs

This is the most important reporting rule.

Always distinguish:

```text
max_epochs
epochs_completed
best_epoch
stop_reason.
```

Never use the terms interchangeably.

---

# 222. Critical technical note: patience is part of the cap experiment context

E100 is not “100 fixed epochs”.

It is:

```text
up to100 epochs
subject to patience10 early stopping.
```

---

# 223. Critical technical note: E100 must not be a second training stage

No:

```text
first stage 1..50
second stage 51..100.
```

It is one continuous registered run.

---

# 224. Critical technical note: fresh E100 enables prefix audit

Because S16 changes only a loop ceiling and no scheduler depends on it:

```text
pre50 trajectory should be reproducible
```

when environment determinism/provenance permit.

This is one of the best audits for hidden drift in the whole sweep chain.

---

# 225. Critical technical note: E50 hit cap does not imply E100 must win

Only a new verified BEST after50 can show added budget was useful for selection.

---

# 226. Critical technical note: E100 winning before50 is not evidence for extra budget

Such a result points to stochastic/reproducibility differences.

Report it cautiously.

---

# 227. Critical technical note: do not reset early stopping at50

Resetting patience would artificially favor E100.

---

# 228. Critical technical note: do not use LAST checkpoint for winner

Always verified BEST Validation RMSE.

---

# 229. Critical technical note: no hidden E150/E200

Even if E100 hits the boundary.

Record boundary limitation and stop S16.

---

# 230. Winner verification checklist

Before writing `s16_epoch_cap_winner.json`:

```text
[ ] E50 valid reused reference.
[ ] E100 valid completed scientific run.
[ ] Same FV*/YS*/L*/P*/A*/B*/LR*/WD*/DR*/D*/H*/N*/F*/LOSS*.
[ ] Same Train/Validation IDs.
[ ] Same population fingerprint.
[ ] Same target scaler/checksum.
[ ] Same architecture fingerprint.
[ ] Same parameter count/schema.
[ ] Same criterion config.
[ ] Same optimizer config.
[ ] Same clip1.
[ ] Same patience10.
[ ] Same min_delta0.
[ ] Same early-stop metric RMSE Wh.
[ ] Same BEST metric RMSE Wh.
[ ] Scheduler=None.
[ ] Warmup=None.
[ ] E100 fresh from epoch1.
[ ] No continue-from-BEST.
[ ] No optimizer reset at50.
[ ] No patience reset at50.
[ ] No RNG/DataLoader reset at50.
[ ] Initialization policy audited.
[ ] Sample-order policy audited.
[ ] Prefix reproducibility audited or marked NOT_VERIFIABLE.
[ ] Stop reasons valid.
[ ] Epoch counts within caps.
[ ] E50 cap-binding classified.
[ ] E100 extra-budget utilization classified.
[ ] Post50 new-BEST analysis generated.
[ ] E100 BEST verified.
[ ] E50 BEST provenance valid.
[ ] Full-precision Validation RMSE used.
[ ] RMSE/R² ranking consistency checked.
[ ] MAE divergence recorded if present.
[ ] Exact tie rule respected.
[ ] E100 boundary winner flagged if applicable.
[ ] No hidden E150/E200.
[ ] Runtime/steps secondary only.
[ ] Phase39 reference generated.
[ ] No Test.
```

---

# 231. Phase39 handoff

Phase39 receives:

```text
s16_epoch_cap_winner.json
s16_reference_update.json
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
loss_id
huber_delta_if_applicable
selected_max_epochs
patience=10
current_gradient_clip=1.0
population_fingerprint
metric_version.
```

and changes only:

```text
gradient clipping.
```

---

# 232. Reference reuse in Phase39

S16 winner already uses:

```text
GC1 = max_norm1.
```

Therefore normally:

```text
reuse S16 winner as GC1 reference
train GC0 = clipping off.
```

---

# 233. Relationship with Phase39 Gradient-clipping sweep

Do not turn clipping off inside S16 even if E100 clipping fraction becomes very small.

Phase39 owns that question.

---

# 234. Relationship with Phase40 RevIN sweep

RevIN remains off during S16.

---

# 235. Relationship with Phase42 Candidate synthesis

Both E50/E100 remain in Registry.

Do not delete the losing budget condition.

---

# 236. Relationship with Phase44 Rolling-origin robustness

S16 winner is based on current Validation interval.

A longer budget may behave differently across temporal folds.

Not tested yet.

---

# 237. Relationship with Phase46 Multi-seed

S16 uses seed42 only.

Small E50/E100 margins are not seed-robust evidence.

---

# 238. Relationship with final reporting

Report:

```text
max_epochs selected
actual epochs completed
best epoch
early stopping behavior.
```

Do not report only:

```text
trained for 50/100 epochs
```

unless actual epochs completed exactly equal cap.

---

# 239. Reproducibility metadata

New E100 run records:

```text
run_id
seed
environment
device
feature fingerprint
X scaler checksum
target scaler checksum
target scaling ID
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
loss ID
Huber delta if applicable
max_epochs=100
patience10
min_delta0
early-stop metric
BEST metric
gradient clip1
scheduler none
warmup none
parameter count
architecture fingerprint
criterion fingerprint
initialization fingerprint
sample-order provenance
population fingerprint
Training Engine fingerprint
BEST checkpoint checksum
LAST checkpoint checksum
history checksum
metric checksum
stop reason
epochs completed
best epoch.
```

---

# 240. No fabricated outputs

Do not pre-fill:

```text
E50 stop reason
E100 epochs completed
E100 best epoch
post50 gain
prefix match
runtime
winner
RMSE
gradient norms
cap binding
```

before execution.

Allowed pre-runtime:

```text
E50 cap = 50
E100 cap = 100
patience = 10
selection/tie rules
scheduler=None contract.
```

---

# 241. Phase sign-off

Create:

```text
phase_38_signoff.json
```

Minimum:

```text
phase = 38
phase_name = S16 Epoch-cap sweep
sweep_version
sweep_id
source_s15_winner_run_id
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
loss_id
loss_name
huber_delta_if_applicable
e50_reference_run_id
e100_run_id
new_run_ids
reused_run_ids
patience
min_delta
gradient_clip
e50_epochs_completed
e50_stop_reason
e50_best_epoch
e50_cap_binding_classification
e100_epochs_completed
e100_stop_reason
e100_best_epoch
e100_extra_budget_utilization
e100_post50_new_best
prefix_reproducibility_status
winner_epoch_cap_id
winner_max_epochs
winner_run_id
winner_rmse_wh
winner_best_epoch
winner_epochs_completed
winner_is_boundary
population_fingerprint
metric_version
architecture_invariance_status
scheduler_independence_status
initialization_audit_status
sample_order_match_status
inherited_warnings
test_status
approved_for_phase39
overall_status
created_at
```

---

# 242. Acceptance checklist

```text
[ ] Phase37 valid.
[ ] approved_for_phase38=true.
[ ] SWEEP_S16_EPOCHCAP-v1 declared.
[ ] S15 winner loaded.
[ ] All prior selected fields fixed.
[ ] E50 exactly50.
[ ] E100 exactly100.
[ ] No extra epoch-cap candidate.
[ ] max_epochs described as ceiling.
[ ] patience exactly10 both.
[ ] min_delta exactly0 both.
[ ] early-stop metric Validation RMSE Wh both.
[ ] BEST metric Validation RMSE Wh both.
[ ] same strict-improvement semantics.
[ ] scheduler=None both.
[ ] warmup=None both.
[ ] no cap-dependent LR schedule.
[ ] selected loss same.
[ ] Huber delta same if applicable.
[ ] architecture identical.
[ ] parameter count identical.
[ ] state_dict schema identical.
[ ] same Train IDs.
[ ] same Validation IDs.
[ ] same WINDOWPOP-v1.
[ ] same feature fingerprint.
[ ] same X scaler.
[ ] same target scaler/transform.
[ ] same batch.
[ ] same AdamW.
[ ] same LR.
[ ] same WD.
[ ] same dropout.
[ ] same clip1.
[ ] same accumulation.
[ ] same precision policy.
[ ] same seed42.
[ ] initialization policy same.
[ ] E50 initial fingerprint compared if available.
[ ] sample-order policy same.
[ ] history code supports100.
[ ] plotting code supports100.
[ ] E50 reference exact-match.
[ ] E50 not retrained.
[ ] E100 registered before training.
[ ] E100 starts fresh at epoch1.
[ ] no warm-start from E50.
[ ] no continue-from-E50-BEST.
[ ] no early-stop counter reset at50.
[ ] no optimizer reset at50.
[ ] no RNG reseed at50.
[ ] no DataLoader reset at50.
[ ] no BEST reset at50.
[ ] no LR change after50.
[ ] train/eval modes correct across long run.
[ ] attention collection OFF during official training.
[ ] E100 trained through TRAINING_ENGINE-v1.
[ ] epochs_completed E100<=100.
[ ] valid stop reason.
[ ] E100 BEST verified.
[ ] E50 BEST provenance valid.
[ ] prefix common range calculated correctly.
[ ] prefix reproducibility audited.
[ ] prefix tolerance predeclared.
[ ] prefix mismatch investigated if present.
[ ] E50 cap-binding classified.
[ ] E50 cap reached distinguished from cap harmful.
[ ] E100 extra-budget utilization classified.
[ ] epochs beyond50 computed.
[ ] optimizer steps beyond50 computed.
[ ] post50 BEST diagnostics generated.
[ ] first new BEST after50 recorded if exists.
[ ] post50 gain N/A if no post50 epochs.
[ ] patience state audited where possible.
[ ] full-precision Validation RMSE used.
[ ] E50→E100 effect computed.
[ ] winner=min BEST RMSE.
[ ] exact tie=E50.
[ ] runtime does not override RMSE.
[ ] no arbitrary gain threshold.
[ ] RMSE/R² ranking consistency checked.
[ ] MAE divergence recorded if present.
[ ] E100 winning before50 triggers reproducibility warning/investigation.
[ ] E100 boundary winner flagged if applicable.
[ ] no E150/E200 extension.
[ ] no score-based rerun.
[ ] early stop before100 accepted as valid outcome.
[ ] optimizer-budget diagnostics generated.
[ ] gradient/clipping diagnostics generated.
[ ] runtime-budget diagnostics generated.
[ ] optional memory leak audit handled honestly.
[ ] optional generalization uses no Test.
[ ] no claim E100 always trains twice as long.
[ ] no claim E50 undertrained solely from cap reached.
[ ] single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] cap×patience context documented.
[ ] cap×scheduler dependency ruled out.
[ ] cap×loss/optimizer interactions documented as fixed.
[ ] inherited warnings propagated.
[ ] winner artifact generated.
[ ] Phase39 reference update generated.
[ ] GC1 reuse identified for Phase39.
[ ] no failed/SANITY run ranked.
[ ] no Test access.
[ ] figures source-generated.
[ ] summary/report/README generated.
[ ] discrepancy log generated.
[ ] phase sign-off generated.
```

---

# 243. Acceptance criteria

Phase 38 chỉ PASS khi:

```text
S15-selected model/data/loss/optimizer configuration is fixed.

Exactly E50 and E100 are represented.

Only max_epochs changes.

max_epochs is treated as a ceiling, not guaranteed actual training length.

patience10/min_delta0 remain unchanged.

Early stopping and BEST checkpoint selection both use Validation RMSE Wh.

No scheduler/warmup depends on max_epochs.

Same Train/Validation IDs and WINDOWPOP-v1 are used.

Same architecture/parameter count/state schema are verified.

Same optimizer/LR/WD/batch/dropout/clip/seed are used.

E50 exact S15 reference is reused.

E100 is one fresh run from epoch1.

No continuation from E50 BEST occurs.

No optimizer/patience/RNG/DataLoader/BEST reset occurs at epoch50.

E100 BEST is verified.

Prefix reproducibility is audited or transparently marked not verifiable.

E50 cap-binding status is classified.

E100 extra-budget utilization is measured.

Post-50 new-BEST behavior is analyzed.

Validation BEST RMSE Wh selects winner.

Exact RMSE tie selects E50.

E100 boundary status is documented without hidden search extension.

Phase39 reference is generated.

Test remains untouched.
```

---

# 244. Failure conditions

Phase 38 FAIL if:

```text
wrong S15 winner used

any prior selected architecture/data/loss setting changes

E50/E100 values differ from registry

patience changes

min_delta changes

early-stop metric changes

BEST metric changes

scheduler/warmup tied to max_epochs is introduced

gradient clipping changes

optimizer/LR/WD/batch changes

E100 continues from E50 BEST

early-stop counter resets at50

optimizer resets at50

RNG/DataLoader resets at50

BEST state resets at50

E100 uses a second-stage training protocol

sample population differs

E50 mismatch is ignored

E50 is retrained and a favorable rerun selected

E100 failure is treated as automatic E50 win

LAST checkpoint is used instead of BEST for ranking

RMSE rounded before ranking

runtime overrides lower non-tied RMSE

E150/E200 is added after seeing results

score-based rerun occurs

Test is accessed.
```

---

# 245. Common mistakes

## 245.1 Nói E100 là “train 100 epochs”

Sai nếu Early Stopping dừng trước100.

## 245.2 E100 = load E50 BEST rồi train tiếp

Sai.

## 245.3 Reset patience ở epoch50

Sai và ưu ái E100.

## 245.4 Tăng patience lên20 cho E100

Sai two-factor experiment.

## 245.5 Dùng cosine schedule với T_max=max_epochs

Sai vì first50 trajectory thay đổi.

## 245.6 E50 hit cap rồi kết luận chắc chắn undertrained

Sai. Phải xem E100 có new BEST sau50 không.

## 245.7 E100 last epoch tốt hơn E50 last epoch nên chọn E100

Sai. So BEST checkpoint.

## 245.8 E100 best ở epoch30 thấp hơn E50 best rồi nói “100 epochs tốt hơn”

Sai causal interpretation; extra budget chưa được dùng tại epoch30.

## 245.9 E100 thắng ở epoch100 rồi tự thử E200

Hidden adaptive search.

## 245.10 Early Stop E100 ở epoch43 rồi coi run thất bại

Sai. Đây là valid evidence cap100 không cần thiết.

## 245.11 Reset optimizer tại epoch50

Sai second-stage training.

## 245.12 Dùng batch lớn hơn cho E100 để tiết kiệm thời gian

Sai.

## 245.13 Chọn E50 vì nhanh hơn dù E100 RMSE thấp hơn

Sai, trừ exact RMSE tie.

## 245.14 Dùng Test để quyết định epoch budget

Forbidden.

---

# 246. Recommended execution pseudocode

```text
load_phase37_signoff()
assert_approved_for_phase38()

s15 = load_s15_winner()
freeze_all_prior_selected_fields(s15)

caps = {
    "E50": 50,
    "E100": 100
}

assert patience == 10
assert min_delta == 0
assert early_stop_metric == "validation_rmse_wh"
assert best_checkpoint_metric == "validation_rmse_wh"
assert scheduler is None
assert warmup is None
assert gradient_clip == 1.0

audit_common_data_population()
audit_architecture_invariance()
audit_parameter_schema_equality()
audit_loss_invariance()
audit_optimizer_invariance()
audit_scheduler_independence()
audit_history_plot_capacity()

e50_reference = resolve_s15_winner_run()
assert_exact_s16_reference_match(
    e50_reference,
    max_epochs=50
)

inspect_e50_cap_binding(e50_reference)

register_e100_run()

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

criterion = build_selected_loss(
    LOSS*,
    huber_delta_if_needed=1.0
)

optimizer = build_fresh_adamw(
    model=model,
    lr=LR*,
    weight_decay=WD*,
    group_policy=FROZEN_S9_GROUP_POLICY
)

e100_result = TRAINING_ENGINE_v1.fit(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    max_epochs=100,
    patience=10,
    min_delta=0,
    early_stop_metric="validation_rmse_wh",
    gradient_clip=1.0,
    scheduler=None
)

verify_best_checkpoint(
    e100_result,
    max_epochs=100
)

audit_prefix_reproducibility(
    e50_reference,
    e100_result
)

cap_binding = build_cap_binding_audit(
    e50_reference,
    e100_result
)

extra_budget = build_extra_budget_utilization(
    e100_result,
    boundary=50
)

post50 = build_post50_improvement_diagnostics(
    e100_result
)

results = {
    "E50": e50_reference,
    "E100": e100_result
}

metrics = build_verified_s16_metrics(results)
effect = compute_e50_to_e100_effect(metrics)
assert_rmse_r2_ranking_consistent(metrics)

winner = select_min_rmse(
    metrics,
    exact_tie_prefer="E50"
)

write_gradient_diagnostics()
write_convergence_diagnostics()
write_runtime_budget_diagnostics()
write_hypothesis_outcomes()
write_findings()
write_s16_winner(winner)
write_phase39_reference_update(winner)
write_summary_report_readme_signoff()
```

---

# 247. Definition of Done

\[
\boxed{
One\ Fixed\ Model/Data/Loss\ Setup
+
Two\ Epoch\ Ceilings
+
Same\ Patience
+
One\ Fresh\ E100
+
One\ Valid\ Reused\ E50
+
Prefix\ Audit
+
Cap\text{-}Binding\ Analysis
+
Post50\ New\text{-}BEST\ Analysis
+
Verified\ BEST\ Metrics
+
S16\ Winner
+
Phase39\ Reference
+
No\ Test
}
\]

---

# 248. Final status contract

```text
PHASE 38 tests max_epochs only.

Candidates:

E50:
max_epochs=50

E100:
max_epochs=100.

Frozen:
all winners from S1–S15
same architecture
same parameter count/schema
same data/population
same loss
same AdamW
same LR
same WD
same batch
same dropout
same gradient clip1
same seed42
same Training Engine
same Metric version.

Early stopping:
patience10
min_delta0
Validation RMSE Wh
unchanged.

BEST:
Validation RMSE Wh
unchanged.

Scheduler:
None.

Warmup:
None.

Therefore:
max_epochs changes only the maximum allowed loop length.

E50:
reuse S15 winner if exact match.

E100:
one fresh run from epoch1.

No:
continue from E50 BEST
patience reset at50
optimizer reset at50
RNG reseed at50
DataLoader reset at50
BEST reset at50
cap-dependent LR schedule
extra E150/E200.

Key diagnostics:
E50 cap binding
E100 actual epochs
epochs beyond50
prefix equivalence
new BEST after50
best epoch
stop reason
optimizer-step/runtime cost.

Selection:
minimum verified BEST Validation RMSE Wh.

Exact tie:
prefer E50.

If E100 wins with BEST >50:
evidence extra allowed budget was useful.

If E100 wins with BEST <=50:
investigate stochastic/prefix differences;
do not attribute benefit to extra epochs.

If E100 hits100:
flag boundary;
do not auto-extend search.

No Test.

After SWEEP_S16_EPOCHCAP-v1 PASS:
proceed to
PHASE 39 — S17 Gradient-clipping sweep.
```

---

# 249. Final check

Correct:

```text
Load S15 winner
→ freeze everything except max_epochs
→ E50 vs E100
→ keep patience10
→ prove no cap-dependent scheduler
→ reuse E50
→ fresh E100 from epoch1
→ verify E100 BEST
→ audit shared prefix
→ classify E50 cap binding
→ measure epochs beyond50
→ detect new global BEST after50
→ compare verified BEST RMSE
→ exact tie E50
→ Phase39 reference
```

Incorrect:

```text
E50 BEST
→ continue another50
→ reset patience
→ change scheduler
→ compare LAST epochs
→ choose by runtime
→ add E200
→ inspect Test
```

Chỉ sau khi `SWEEP_S16_EPOCHCAP-v1` được sign-off mới chuyển sang **PHASE 39 — S17 Gradient-clipping sweep**.
