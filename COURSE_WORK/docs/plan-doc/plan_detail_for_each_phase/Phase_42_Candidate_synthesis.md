# PHASE 42 — CANDIDATE SYNTHESIS

## Kế hoạch tổng hợp Transformer candidates sau chuỗi S1–S19 và chuẩn bị cho LSTM tuning + Rolling-Origin Robustness

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Upstream sensitivity:** `SWEEP_S19_BOUNDARY-v1`  
**Phase ID:** `PHASE_42_CANDIDATE_SYNTHESIS`  
**Output version:** `CANDIDATE_SYNTHESIS-v1`  
**Phase trước:** `Phase_41_S19_Boundary-protocol_check.md`  
**Phase sau:** `Phase_43_LSTM_tuning.md`  
**Robustness consumer:** `Phase_44_Rolling-origin_robustness.md`

---

# 1. Vai trò của Phase 42

Phase 42 là bước chuyển từ:

```text
sequential one-factor Validation sweeps
```

sang:

```text
shortlisted candidate robustness evaluation.
```

Mục tiêu của Phase này không phải train thêm model, không phải tiếp tục tune hyperparameter, và không phải mở một Cartesian search mới.

Mục tiêu là:

```text
1. Reconstruct chính xác configuration lineage sau S1–S19.
2. Xác nhận primary Transformer configuration hiện tại.
3. Tổng hợp toàn bộ local sweep evidence.
4. Tìm các tuning decisions còn yếu/không chắc chắn nhất.
5. Tạo một shortlist nhỏ, deterministic, reproducible.
6. Không tạo "Frankenstein search" không kiểm soát.
7. Chuẩn bị candidate specifications cho Phase44 Rolling-Origin Robustness.
8. Chuẩn bị fairness context cho Phase43 LSTM tuning.
9. Giữ nguyên Test firewall.
```

Nguyên tắc trung tâm:

\[
\boxed{
No\ New\ Training
+
No\ New\ Validation\ Search
+
Registry\text{-}Grounded\ Evidence
+
One\text{-}Factor\ Local\ Alternatives
+
Small\ Deterministic\ Shortlist
+
No\ Test
}
\]

---

# 2. Phase 42 không phải một sweep

Không có:

```text
S20
```

trong Phase42.

Không được chạy:

```text
new optimizer experiments
new architecture experiments
new feature experiments
new seeds
new Validation evaluations
```

Phase42 chỉ đọc artifacts đã được tạo hợp lệ từ S1–S19 và tạo candidate specifications.

---

# 3. Vì sao Candidate Synthesis là cần thiết

Chuỗi S1–S18 sử dụng sequential one-factor sweeps:

```text
winner của phase trước
→ reference của phase sau.
```

Điều này tạo ra một **greedy selected chain** rất rõ ràng, nhưng không đồng nghĩa:

```text
mọi runner-up từ phase cũ
đã từng được test dưới toàn bộ final downstream settings.
```

Ví dụ:

```text
runner-up từ S8 Learning Rate
```

được đánh giá ở configuration tồn tại tại Phase30, trước khi S9–S18 được chọn.

Do đó Phase42 phải phân biệt:

```text
local sweep evidence
```

với:

```text
final-context candidate performance.
```

Không được gán local historical RMSE cho một synthesized final-context candidate chưa từng train.

---

# 4. Primary output của Phase42

Phase42 phải tạo:

```text
Transformer candidate shortlist
```

với tối đa:

```text
3 candidates
```

theo mặc định:

```text
TR_C0_PRIMARY
TR_C1_LOCAL_ALT
TR_C2_LOCAL_ALT
```

Cấu trúc:

```text
TR_C0_PRIMARY
= exact current greedy-selected final Transformer configuration

TR_C1_LOCAL_ALT
= TR_C0 nhưng thay đúng 1 factor bằng local runner-up
  từ decision có evidence gap nhỏ nhất và candidate hợp lệ

TR_C2_LOCAL_ALT
= TR_C0 nhưng thay đúng 1 factor khác bằng local runner-up
  có evidence gap nhỏ kế tiếp và candidate hợp lệ
```

Không bắt buộc phải có đủ 3 nếu không có đủ admissible alternatives.

---

# 5. Vì sao giới hạn tối đa 3 Transformer candidates

Phase44 Rolling-Origin Robustness sẽ tốn nhiều lần training theo temporal folds.

Nếu Candidate Synthesis tạo:

```text
10–20 candidates
```

thì Phase44 trở thành một hidden hyperparameter search rất lớn và phá mục đích robustness.

Do đó khóa:

```text
MAX_TRANSFORMER_ROBUSTNESS_CANDIDATES = 3
```

bao gồm:

```text
1 primary
+
tối đa 2 one-factor alternatives.
```

---

# 6. Không tạo full Cartesian product

Forbidden:

```text
all feature sets
× all lookbacks
× all learning rates
× all dropout
× all d_model
× all layers
...
```

Không được dùng Phase42 để quay lại grid search.

---

# 7. Không combine nhiều runner-up trong cùng một synthesized candidate

Candidate alternative phải khác `TR_C0_PRIMARY` đúng **một scientific factor**.

Allowed:

```text
TR_C1:
same as primary
except LR2 → LR1
```

Forbidden:

```text
same candidate:
LR2 → LR1
and
DR01 → DR02
and
N2 → N1.
```

Lý do:

```text
multi-factor recombination làm mất khả năng attribution
và mở hidden combinatorial search.
```

---

# 8. Boundary protocol không phải candidate hyperparameter

S19:

```text
WB0 = primary protocol
WB1 = strict-isolation sensitivity.
```

Phase42 phải giữ:

```text
TR_C0/T1/T2 boundary protocol = WB0
```

trừ khi Phase41 explicitly sets:

```text
protocol_amendment_required = true
```

và amendment đã được giải quyết.

WB1 không được tự động đưa vào candidate shortlist như một model hyperparameter.

---

# 9. Preconditions

Phase42 chỉ bắt đầu nếu:

```text
Phase41 overall status = PASS
or
PASS_WITH_WARNING
```

và:

```text
approved_for_phase42 = true
```

Nếu:

```text
protocol_amendment_required = true
```

thì:

```text
STOP
```

cho đến khi amendment được document và upstream reference được cập nhật.

---

# 10. Required upstream artifacts

Phase42 phải load:

```text
phase_41_signoff.json
s19_boundary_sensitivity_conclusion.json
s19_reference_update.json
```

và toàn bộ winner/reference artifacts từ S1–S18.

Required lineage sources:

```text
S1  feature-set winner/reference
S2  time-feature winner/reference
S3  target-scaling winner/reference
S4  lookback winner/reference
S5  pooling winner/reference
S6  activation winner/reference
S7  batch winner/reference
S8  learning-rate winner/reference
S9  weight-decay winner/reference
S10 dropout winner/reference
S11 d_model winner/reference
S12 heads winner/reference
S13 layers winner/reference
S14 FFN winner/reference
S15 loss winner/reference
S16 epoch-cap winner/reference
S17 clipping winner/reference
S18 RevIN winner/reference or valid SKIPPED_NOT_APPLICABLE outcome
S19 boundary sensitivity conclusion/reference
```

---

# 11. Machine-readable artifacts là source of truth

Priority:

```text
1. winner JSON
2. reference_update JSON
3. phase signoff JSON
4. Experiment Registry
5. metrics/effect tables
6. human-readable reports
```

Không lấy values từ:

```text
manual notes
screenshots
rounded report text
```

nếu machine-readable full-precision artifact tồn tại.

---

# 12. Reconstruct selected lineage

Phase42 phải reconstruct selected configuration từ S1 đến S18.

Canonical fields:

```text
feature_set_id
time_feature_id
feature_variant_id
target_scaling_id
lookback_id
lookback_steps
horizon
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
gradient_clip_id
gradient_clip_enabled
gradient_clip_max_norm_if_applicable
revin_id
revin_enabled
revin_config_if_applicable
boundary_protocol
seed
population_version
metric_version
training_engine_version.
```

---

# 13. Primary Transformer candidate

`TR_C0_PRIMARY` phải là exact S18-selected WB0 primary configuration carried through S19.

Conceptually:

```text
S1 winner
→ S2 winner
→ ...
→ S18 selected RevIN state
→ WB0 primary protocol.
```

Không dùng WB1 làm primary candidate nếu không có Protocol Amendment.

---

# 14. Primary candidate must map to an existing completed run

`TR_C0_PRIMARY` phải có:

```text
existing_run_id
BEST checkpoint
verified Validation metrics
WB0
valid registry status
```

normally là:

```text
S18 selected run
```

hoặc S17-carried RN0 run nếu S18 `SKIPPED_NOT_APPLICABLE`.

Hard:

```text
TR_C0_PRIMARY requires existing valid run.
```

---

# 15. Primary candidate fingerprint

Create canonical config JSON với deterministic ordering rồi:

```text
SHA256
```

to generate:

```text
candidate_config_fingerprint.
```

Fingerprint không include:

```text
run_id
runtime
best epoch
Validation metric
timestamp
```

vì đây không phải scientific config fields.

---

# 16. Sweep evidence ledger

Phase42 phải tạo một row cho mỗi S1–S18 scientific decision.

Fields tối thiểu:

```text
phase
sweep_id
factor_name
winner_value
runner_up_value
winner_run_id
runner_up_run_id
winner_validation_rmse_wh
runner_up_validation_rmse_wh
absolute_local_regret_wh
relative_local_regret_pct
same_population_verified
same_metric_verified
warning_flags
empirical_comparison_performed
runner_up_available
factor_current_value_matches_winner
candidate_substitution_feasible
candidate_substitution_reason
evidence_context
```

---

# 17. Local regret definition

For each sweep:

\[
LocalRegret
=
RMSE_{runner-up}
-
RMSE_{winner}
\]

Expected:

```text
LocalRegret >= 0
```

under full-precision winner selection.

Relative:

\[
LocalRegret\%
=
100\times
\frac{RMSE_{runner-up}-RMSE_{winner}}
{RMSE_{winner}}
\]

This measures local decision separation at the phase where the factor was tested.

---

# 18. Local regret is not final-context candidate RMSE

Hard reporting rule:

```text
local regret
≠
expected final candidate performance penalty.
```

Do not write:

```text
TR_C1 RMSE = old runner-up RMSE
```

unless exact full candidate config maps to that historical run.

---

# 19. Multi-level sweeps: runner-up definition

For sweeps with >2 candidates:

```text
S4 lookback
S8 learning rate
S9 weight decay
S10 dropout
...
```

runner-up is:

```text
the condition with second-lowest verified full-precision Validation RMSE
```

after valid-run filtering.

Tie resolution for the original sweep remains according to original sweep rule.

Candidate synthesis does not re-rank the original sweep with new tie rules.

---

# 20. Invalid/failed runs cannot be runner-up

Exclude:

```text
FAILED
FAILED_NUMERICAL
INCOMPLETE
SANITY_ONLY
DEBUG
invalid reference
population mismatch
Test-contaminated run
protocol violation.
```

---

# 21. S18 not-applicable handling

If:

```text
RN1 not applicable
```

then S18 does not have an empirical runner-up.

Record:

```text
empirical_comparison_performed = false
runner_up_available = false
candidate_substitution_feasible = false
reason = RN1_NOT_APPLICABLE.
```

Do not treat RN1 as a near-tie candidate.

---

# 22. S19 is not included in local-regret ranking

Boundary protocol:

```text
WB0/WB1
```

goes into:

```text
sensitivity evidence ledger
```

not:

```text
candidate-alternative factor ranking.
```

---

# 23. Candidate source factors

Eligible one-factor substitutions may come from:

```text
S1–S18
```

subject to:

```text
empirical runner-up exists
final-context candidate is valid
no protocol constraint violated
no information-set mutation side effect
```

---

# 24. Final-context substitution

For an eligible factor `f`:

```text
Candidate_f
=
copy(TR_C0_PRIMARY)
set f = local_runner_up_value
keep all other fields fixed.
```

This is a **candidate specification**, not necessarily an already-trained run.

---

# 25. Candidate must remain internally valid

After substitution run compatibility checks.

Examples:

```text
d_model % num_heads == 0

RevIN RN1 requires historical Appliances

Huber retains delta=1.0 in current candidate y_model space

lookback exists in WINDOWPOP-v1 design

pooling supported

FFN > 0

dropout valid

batch supported

feature variant/time-feature composition valid

gradient clipping semantics valid.
```

---

# 26. No automatic second-factor repair

If changing one factor makes current config invalid:

```text
candidate is INADMISSIBLE.
```

Do not fix by changing another factor.

Example:

```text
TR_C0 uses RN1
candidate changes feature set to FS0
→ historical Appliances removed
→ RN1 invalid
```

Forbidden repair:

```text
also switch RN1 → RN0.
```

That would be a two-factor candidate.

Correct:

```text
exclude that feature-set substitution.
```

---

# 27. Candidate feasibility audit

For every local runner-up substitution, record:

```text
candidate_id_provisional
source_sweep
factor
primary_value
alternative_value
one_factor_only
schema_valid
feature_information_valid
RevIN_valid
d_model_heads_valid
loss_target_space_valid
windowing_valid
optimizer_config_valid
training_engine_supported
boundary_protocol_valid
admissible
reason
```

---

# 28. Candidate duplicate detection

Different source logic might generate identical full configs in edge cases.

Canonical fingerprint every provisional candidate.

Hard:

```text
no duplicate fingerprints in shortlist.
```

If duplicate:

```text
keep higher-priority source
record duplicate mapping.
```

---

# 29. Search Experiment Registry for exact existing run

For each synthesized final-context candidate:

```text
query registry by full candidate config fingerprint.
```

Possible:

```text
EXACT_EXISTING_RUN
SYNTHESIZED_NOT_PREVIOUSLY_RUN.
```

Do not assume the runner-up run itself is exact final-context match.

---

# 30. Existing-run exact match does not eliminate Phase44 retraining

Even if an exact current-Validation run exists:

```text
Phase44 still retrains candidate per rolling-origin fold.
```

Existing run is lineage evidence only.

---

# 31. Candidate evidence class

Every candidate must carry:

```text
PRIMARY_EXISTING
LOCAL_ALT_EXACT_EXISTING
LOCAL_ALT_SYNTHESIZED
```

No ambiguous provenance.

---

# 32. Factor uncertainty ranking

Purpose:

```text
select at most 2 alternative factors
```

for `TR_C1`, `TR_C2`.

Use a deterministic lexicographic ranking.

Primary key:

```text
smallest absolute_local_regret_wh
```

because smallest separation means decision was least decisively separated on its local sweep.

Secondary:

```text
smallest relative_local_regret_pct
```

Tertiary:

```text
higher phase number preferred
```

because later sweep evidence is closer to final selected context.

Final deterministic tie-break:

```text
factor_name ascending
```

---

# 33. Why no arbitrary weighted uncertainty score

Do not invent:

```text
0.4*RMSE_gap + 0.3*warning + 0.3*complexity
```

without methodological basis.

Lexicographic ordering is transparent and reproducible.

---

# 34. Warning flags do not automatically override RMSE-gap ranking

Warnings are carried as context:

```text
METRIC_RANKING_DIVERGENCE
BOUNDARY_WINNER
SMALL_SELECTION_MARGIN
...
```

but should not be converted into arbitrary numeric penalties.

Candidate selection remains deterministic from admissibility + local RMSE separation.

---

# 35. Exact local tie naturally gets highest alternative priority

If original sweep had:

```text
winner RMSE == runner-up RMSE
```

then:

```text
LocalRegret = 0
```

and that factor becomes a strong robustness candidate, provided final-context substitution is valid.

---

# 36. Boundary winner flag

If a sweep winner lies on tested boundary:

```text
LR1 or LR3
F64 or F256
E100 at boundary
...
```

record warning.

Do not create an untested outside-range candidate.

Example forbidden:

```text
LR winner = 1e-3
→ synthesize 3e-3
```

No.

---

# 37. Candidate shortlist selection algorithm

Algorithm:

```text
1. Add TR_C0_PRIMARY.
2. Build eligible local-runner-up substitution table from S1–S18.
3. Remove inadmissible candidates.
4. Remove duplicate fingerprints.
5. Sort by:
   a. local_regret_wh ascending
   b. local_regret_pct ascending
   c. source_phase descending
   d. factor_name ascending
6. Select first candidate → TR_C1_LOCAL_ALT.
7. Select next candidate from a different factor → TR_C2_LOCAL_ALT.
8. Stop at max total candidates = 3.
```

---

# 38. Shortlist can contain fewer than 3 candidates

If only one valid alternative exists:

```text
TR_C0
TR_C1
```

If none:

```text
TR_C0 only.
```

Do not fabricate candidates to fill quota.

---

# 39. Candidate IDs

Recommended:

```text
TR_C0_PRIMARY

TR_C1_ALT_<FACTOR_TOKEN>

TR_C2_ALT_<FACTOR_TOKEN>
```

Examples only:

```text
TR_C1_ALT_DROPOUT
TR_C2_ALT_LR
```

Runtime factor determines actual IDs.

---

# 40. Candidate description must state exact delta

Every local alternative:

```text
source_factor
primary_value
alternative_value
changed_fields
unchanged_fields_count
source_local_regret
evidence_class.
```

Hard:

```text
changed scientific factor count = 1.
```

---

# 41. Derived fields may change but do not count as extra factors

Example:

```text
num_heads changes
→ head_dim = d_model / num_heads changes.
```

`head_dim` is a derived architecture field.

Candidate still counts as one factor change:

```text
HEADS.
```

Similarly:

```text
feature set changes
→ input feature count changes.
```

This is derived from one factor.

---

# 42. Scientific factor vs derived field audit

Create mapping:

```text
factor_name
primary_setting
alternative_setting
derived_fields_expected_to_change
derived_fields_forbidden_to_change.
```

This prevents false multi-factor alarms.

---

# 43. Candidate parameter count may differ

Alternatives in:

```text
d_model
heads indirectly not parameter count much via same D
layers
FFN
RevIN
feature set
```

may change parameter count.

This is inherent.

Do not parameter-match candidates.

---

# 44. Candidate feature count may differ

Feature-set/time-feature alternatives may change input dimension.

Allowed if substitution is valid.

No hidden feature changes.

---

# 45. Candidate target space may differ

YS alternative changes target model-space.

If candidate uses Huber:

```text
Huber delta remains exactly 1.0 candidate y_model-space.
```

Record changed semantic raw-Wh equivalent if Phase44 materializes that candidate.

Do not precompute without scaler runtime.

---

# 46. Candidate lookback may differ

Lookback alternative:

```text
L36/L72/L144
```

must use registered `WINDOWPOP-v1` common target design in robustness stage where applicable.

Phase42 records windowing requirement; no windows are rebuilt here unless metadata checks are needed.

---

# 47. Candidate boundary protocol remains WB0

All Transformer robustness candidates:

```text
boundary_protocol = WB0.
```

WB1 evidence is stored separately.

---

# 48. Primary protocol sensitivity record

Candidate manifest must include S19 evidence:

```text
wb1_sensitivity_run_id
common_delta_rmse
validation_coverage_removed
test_coverage_removed_metadata
protocol_amendment_required
sensitivity_interpretation.
```

This travels into Phase44/45 context.

---

# 49. No Test access

Phase42 may use only existing Test eligibility metadata from S19.

Forbidden:

```text
Test y
Test predictions
Test metrics
Test residuals
Test-based ranking.
```

Candidate shortlist is Validation-derived only.

---

# 50. No new prediction generation

Phase42 normally should not call model inference.

It is an artifact/registry synthesis phase.

If a consistency check needs existing metric artifacts:

```text
read stored metrics
```

rather than rerun Validation predictions.

---

# 51. No new training

Hard:

```text
new_training_runs = 0.
```

Any candidate marked synthesized stays:

```text
NOT_YET_MATERIALIZED.
```

until robustness stage.

---

# 52. No new seed

No seed123/2026 here.

Final multi-seed later.

---

# 53. No new checkpoint selection

No.

---

# 54. No score-based manual candidate insertion

Do not manually add:

```text
“candidate này nhìn curve đẹp”
```

unless it satisfies the deterministic synthesis rule or is separately declared via Protocol Amendment before robustness execution.

---

# 55. No complexity-based override

Candidate inclusion is not based on:

```text
parameter count
runtime
memory
```

except feasibility.

These are annotations, not shortlist ranking criteria.

---

# 56. Current primary candidate should never be dropped

`TR_C0_PRIMARY` is always included if Phase41 permits Phase42.

Even if a local runner-up had tiny regret.

---

# 57. Candidate synthesis does not re-open winners

Phase42 does not change:

```text
selected S1–S18 winner chain.
```

It only identifies alternatives for robustness testing.

---

# 58. Baseline anchors

Carry existing references:

```text
Persistence baseline
LSTM B0
Transformer primary
```

but do not include Persistence/LSTM B0 in Transformer candidate count.

Phase43 will produce:

```text
tuned LSTM candidate.
```

---

# 59. Handoff to Phase43 LSTM tuning

Phase42 must create a fairness context for LSTM tuning.

Recommended fields:

```text
forecast task H1
selected feature variant
selected target scaling
selected lookback
boundary protocol WB0
sample population policy
batch/training metric policy where architecture-independent
Validation metric = RMSE Wh
Test locked
```

Phase43 may tune LSTM architecture/training within its own registered scope, but it should not silently use a different forecasting task.

---

# 60. LSTM fairness nuance

Transformer-specific settings do not transfer:

```text
heads
FFN
RevIN necessarily
Transformer pooling.
```

Phase43 plan will define LSTM tuning.

Phase42 only hands over shared data/task contract.

---

# 61. Handoff to Phase44 Rolling-Origin Robustness

Phase44 receives:

```text
Transformer shortlist from Phase42
+
tuned LSTM candidate from Phase43
+
Persistence baseline
```

and evaluates robustness across temporal folds according to Phase44 protocol.

---

# 62. Candidate status before Phase44

Possible:

```text
READY_EXISTING_PRIMARY
READY_SYNTHESIZED_SPEC
READY_EXACT_EXISTING_ALT
INADMISSIBLE
DUPLICATE_EXCLUDED
NOT_SELECTED_BY_SHORTLIST_CAP.
```

Only READY candidates enter shortlist.

---

# 63. Candidate synthesis matrix

Create a matrix containing all eligible source sweeps, not only selected T1/T2.

This ensures transparent exclusion.

Fields:

```text
source_phase
source_sweep
factor
winner_value
runner_up_value
local_regret_wh
local_regret_pct
final_context_substitution_valid
provisional_candidate_fingerprint
registry_exact_match
selection_rank
selected_for_shortlist
exclusion_reason.
```

---

# 64. Lineage consistency audit

Need verify:

```text
winner from phase n
matches reference input of phase n+1
```

for the selected field chain where relevant.

If lineage break is found:

```text
STOP.
```

Do not synthesize candidates from inconsistent history.

---

# 65. Reference-update consistency

For every phase:

```text
winner_run_id
selected value
reference_update next phase
```

must agree.

Allowed exception:

```text
S18 SKIPPED_NOT_APPLICABLE
```

where RN0 is carried by applicability constraint.

---

# 66. Registry run integrity

Primary and local evidence runs must have:

```text
COMPLETED
verified BEST
metric artifact
config fingerprint
no Test contamination.
```

---

# 67. Metric unit consistency

All local regrets must use:

```text
Validation RMSE Wh
```

not:

```text
model-space MSE
scaled RMSE
rounded report metric.
```

---

# 68. Population consistency within each sweep

Before computing local regret:

```text
winner and runner-up same target population
```

must be verified from that sweep’s contract.

If not:

```text
local regret inadmissible for uncertainty ranking.
```

---

# 69. Do not compare absolute RMSE across unrelated historical sweeps

A Phase23 run RMSE and Phase38 run RMSE come from different configs.

Candidate ranking uses:

```text
within-sweep winner–runner-up separation
```

not global sorting of every historical run by RMSE.

Forbidden:

```text
sort all registry runs by validation_rmse
take top3.
```

That would ignore sequential context/population/config differences.

---

# 70. Why within-sweep regret is safer

Each S1–S18 sweep was designed as:

```text
one factor
same population
same current reference
same metric.
```

Therefore its local winner-vs-runner-up gap is the cleanest available evidence about uncertainty in that specific decision.

---

# 71. Local evidence still has interaction limitations

Even local regret is conditional on the configuration at that phase.

Phase42 must mark:

```text
FINAL_CONTEXT_PERFORMANCE_UNKNOWN
```

for synthesized alternatives not exactly executed under final configuration.

Phase44 rolling-origin retraining resolves this partially.

---

# 72. Candidate evidence confidence

Recommended qualitative field:

```text
DIRECT_EXACT_FINAL_CONTEXT
DIRECT_LOCAL_HISTORICAL_CONTEXT
SYNTHESIZED_FINAL_CONTEXT
```

No arbitrary numeric confidence score.

---

# 73. Exact final-context existing alternative

If Registry contains a run whose entire config equals a synthesized candidate:

```text
evidence_class = DIRECT_EXACT_FINAL_CONTEXT
```

Still do not let its old Validation score replace rolling-origin evaluation.

---

# 74. Candidate config canonicalization

Before fingerprint:

```text
sort keys
normalize enums
numeric values canonical
remove null fields not applicable consistently
include all scientific config fields
include boundary protocol
include RevIN details if enabled
include target/loss semantics.
```

---

# 75. Candidate identity must not include seed for architecture/config identity

Recommended two fingerprints:

```text
candidate_config_fingerprint
run_config_fingerprint
```

Candidate config excludes seed.

Run config includes:

```text
seed
fold
runtime training settings.
```

This helps Phase44 materialize candidate across folds.

---

# 76. Candidate spec must include immutable scientific config

Minimum:

```text
candidate_id
candidate_role
source_type
base_candidate_id
changed_factor
changed_from
changed_to
feature_variant_id
target_scaling_id
lookback
horizon
pooling
activation
batch
learning_rate
weight_decay
dropout
d_model
heads
layers
ffn
loss
epoch cap
patience
clipping
RevIN
boundary protocol
training engine
metric version
candidate_config_fingerprint.
```

---

# 77. Candidate spec must include provenance

```text
source_sweep
source_runner_up_run_id
source_local_regret_wh
source_local_regret_pct
source_warning_flags
exact_existing_run_id_if_any
final_context_performance_available
```

---

# 78. Candidate spec must include admissibility evidence

```text
one_factor_only
derived_fields
compatibility_tests
status.
```

---

# 79. Primary candidate evidence

For TR_C0:

```text
source_type = PRIMARY_EXISTING
source_run_id = S18 selected WB0 run
changed_factor = NONE
final_context_performance_available = true
```

Its Validation metric is for lineage context only, not Phase44 winner selection.

---

# 80. Alternative candidate evidence

For T1/T2 synthesized:

```text
source_type = LOCAL_ALT_SYNTHESIZED
changed_factor = one factor
final_context_performance_available = false
```

unless exact registry match exists.

---

# 81. Candidate shortlist must be frozen before Phase43/44 outcomes

Once Phase42 sign-off:

```text
Transformer shortlist is immutable.
```

Do not change T1/T2 after seeing:

```text
LSTM tuning result
rolling-origin partial folds
runtime behavior.
```

Any change requires:

```text
Protocol Amendment.
```

---

# 82. No candidate replacement after first rolling fold

Critical.

If T2 performs badly in first fold:

```text
do not swap in T3.
```

Phase44 must evaluate the locked shortlist.

---

# 83. No candidate pruning based on Phase43 LSTM result

Even if tuned LSTM is very strong:

```text
keep Transformer shortlist unchanged.
```

---

# 84. Compute budget metadata

Phase42 should estimate planned robustness run count symbolically:

```text
N_transformer_candidates
× N_rolling_folds
```

but do not fabricate `N_rolling_folds` if Phase44 has not locked it yet.

Use:

```text
planned_transformer_candidate_count
```

and leave fold count for Phase44.

---

# 85. Candidate complexity annotations

Record runtime metadata from existing related runs when available:

```text
trainable parameter count
mean epoch runtime
peak memory
```

but label:

```text
ENGINEERING_CONTEXT_ONLY.
```

Synthesized candidate runtime may be:

```text
UNKNOWN until Phase44.
```

---

# 86. No estimated RMSE interpolation

Do not predict:

```text
final candidate RMSE
```

by adding local regrets to TR_C0 RMSE.

Forbidden:

```text
RMSE_candidate ≈ RMSE_primary + local_regret.
```

Interactions make this unsupported.

---

# 87. No model ensembling

Candidate synthesis does not create:

```text
ensemble
weighted average
stacking.
```

Not in coursework contract.

---

# 88. No architecture averaging

No.

---

# 89. No checkpoint soup

No.

---

# 90. No candidate mutation from S19 WB1

WB1 run is stored as sensitivity evidence only.

---

# 91. Candidate selection edge case: runner-up equals winner value

Should not occur if values are unique.

If duplicated aliases point to same actual value:

```text
deduplicate factor conditions before runner-up selection.
```

---

# 92. Candidate selection edge case: exact tie

Original sweep winner chosen by its own tie rule.

Runner-up may have same RMSE.

Candidate alternative remains valid with:

```text
local_regret=0.
```

---

# 93. Candidate selection edge case: runner-up technical warning

Non-critical warning:

```text
retain candidate evidence
```

with warning.

Critical protocol violation:

```text
exclude.
```

---

# 94. Candidate selection edge case: source sweep not comparable

If sweep artifact shows:

```text
population mismatch
metric mismatch
reference mismatch
```

then:

```text
do not use its regret for alternative ranking.
```

The selected winner may remain in historical chain if already formally accepted, but Phase42 records a serious inherited warning and may require review.

---

# 95. Candidate selection edge case: S18 skip

Handled as no RevIN alternative.

---

# 96. Candidate selection edge case: feature-set alternative incompatible with RN1

Exclude rather than modify RN1.

---

# 97. Candidate selection edge case: d_model/head invalid

If final heads do not divide alternative d_model:

```text
exclude d_model substitution
```

rather than also changing heads.

Likewise alternate heads must divide final d_model.

---

# 98. Candidate selection edge case: lookback/window population

Alternative L must be one of registered:

```text
L36
L72
L144
```

and supported by `WINDOWPOP-v1`.

If final downstream artifact requires a population unavailable for candidate:

```text
exclude / STOP depending on cause.
```

---

# 99. Candidate selection edge case: target scaling + RevIN bridge

If RN1 active and YS alternative is synthesized:

```text
candidate remains valid only if X-target→new Y target bridge is formally supported.
```

If implementation supports Phase9 YS0/YS1 both:

```text
admissible.
```

Otherwise:

```text
exclude.
```

No hidden fallback.

---

# 100. Candidate selection edge case: Huber + YS alternative

Huber delta remains:

```text
1.0 candidate y_model-space.
```

Candidate spec must flag:

```text
HUBER_DELTA_RAW_EQUIVALENT_RECOMPUTE_REQUIRED_IN_PHASE44.
```

---

# 101. Candidate selection edge case: batch alternative

Same architecture but train batch changes.

Phase44 fold training uses candidate’s selected batch exactly.

No LR scaling.

---

# 102. Candidate selection edge case: epoch-cap alternative

Candidate uses alternative E50/E100 with same:

```text
patience10.
```

Max epochs is ceiling.

---

# 103. Candidate selection edge case: clipping alternative

GC0/GC1 uses exact S17 semantics.

No threshold tuning.

---

# 104. Candidate selection edge case: RevIN alternative

Only if empirical S18 comparison exists and final selected feature set supports target channel.

If primary RN0 and runner-up RN1:

```text
RN1 candidate valid if applicability true.
```

If primary RN1 and runner-up RN0:

```text
valid.
```

---

# 105. Sweep evidence priority does not include Persistence/LSTM

Persistence and LSTM are model-family baselines, not Transformer factor alternatives.

Keep separate.

---

# 106. Expected Phase42 artifact tree

```text
artifacts/
└── candidate_synthesis/
    ├── candidate_synthesis_manifest.json
    ├── candidate_synthesis_contract.json
    ├── phase42_preflight_audit.csv
    ├── selected_lineage.json
    ├── selected_lineage_audit.csv
    ├── sweep_evidence_ledger.csv
    ├── sweep_runner_up_table.csv
    ├── local_regret_ranking.csv
    ├── candidate_admissibility_audit.csv
    ├── candidate_derived_field_audit.csv
    ├── candidate_duplicate_audit.csv
    ├── candidate_registry_match_audit.csv
    ├── transformer_candidate_pool.csv
    ├── transformer_candidate_shortlist.csv
    ├── transformer_candidate_shortlist.json
    ├── candidate_config_fingerprints.csv
    ├── boundary_sensitivity_context.json
    ├── baseline_anchor_context.json
    ├── phase43_lstm_tuning_handoff.json
    ├── phase44_rolling_origin_handoff.json
    ├── candidate_synthesis_findings.csv
    ├── candidate_synthesis_tests.csv
    ├── candidate_synthesis_discrepancies.json
    ├── candidate_synthesis_summary.json
    ├── candidate_synthesis_report.md
    ├── README_CANDIDATE_SYNTHESIS.md
    └── phase_42_signoff.json
```

No new checkpoint folder.

---

# 107. Required outputs

```text
O42.1  Candidate synthesis manifest
O42.2  Candidate synthesis contract
O42.3  Preflight audit
O42.4  Selected lineage JSON
O42.5  Lineage consistency audit
O42.6  Sweep evidence ledger
O42.7  Runner-up table
O42.8  Local-regret ranking
O42.9  Candidate admissibility audit
O42.10 Derived-field audit
O42.11 Duplicate audit
O42.12 Registry exact-match audit
O42.13 Full Transformer candidate pool
O42.14 Final Transformer shortlist CSV
O42.15 Final Transformer shortlist JSON
O42.16 Candidate fingerprints
O42.17 Boundary sensitivity context
O42.18 Baseline anchor context
O42.19 Phase43 LSTM tuning handoff
O42.20 Phase44 Rolling-Origin handoff
O42.21 Findings
O42.22 Acceptance tests
O42.23 Discrepancy log
O42.24 Summary
O42.25 Human-readable report
O42.26 README
O42.27 Phase sign-off
```

---

# 108. Candidate synthesis manifest

`candidate_synthesis_manifest.json`:

```text
phase=42
version=CANDIDATE_SYNTHESIS-v1
source_phase41_signoff
source_primary_protocol=WB0
source_primary_run_id
max_transformer_candidates=3
candidate_strategy=PRIMARY_PLUS_TWO_ONE_FACTOR_LOCAL_ALTERNATIVES
ranking_strategy=LEXICOGRAPHIC_LOCAL_REGRET
new_training_runs=0
new_validation_evaluations=0
new_test_access=0
candidate_shortlist_frozen_on_signoff=true
test_status=NOT_ACCESSED
status
created_at
```

---

# 109. Candidate synthesis contract

Must state:

```text
No training.
No new model evaluation.
No Test.
No Cartesian recombination.

Primary candidate = exact current WB0 selected Transformer.

Alternative candidate:
copy primary
change exactly one factor
to registered local runner-up value
from S1–S18.

Runner-up ranking:
smallest local Validation RMSE regret
then relative regret
then later source phase
then factor name.

Only admissible candidates.
No automatic second-factor repair.

Maximum 3 Transformer candidates total.

S19 WB1 remains sensitivity evidence, not candidate factor.

Shortlist becomes immutable after Phase42 signoff.
```

---

# 110. Preflight audit

`phase42_preflight_audit.csv`:

```text
phase41_pass
approved_for_phase42
protocol_amendment_required_false
primary_protocol_WB0
S1_S18_artifacts_available
registry_available
metric_version_consistent
selected_lineage_reconstructable
primary_run_valid
primary_checkpoint_verified
test_firewall
new_training_forbidden
status
```

---

# 111. Selected lineage JSON

`selected_lineage.json` must include:

```text
phase_by_phase selections
source winner artifact
source run ID
selected value
reference update link/fingerprint
final current value
inherited warning
```

and final canonical config.

---

# 112. Lineage audit

`selected_lineage_audit.csv`:

```text
phase
factor
winner_artifact_value
reference_update_value
next_phase_reference_value
final_config_value_where_applicable
consistent
exception_reason
status
```

---

# 113. Sweep evidence ledger schema

`sweep_evidence_ledger.csv`:

```text
source_phase
sweep_id
factor
winner_value
runner_up_value
winner_run_id
runner_up_run_id
winner_rmse_wh
runner_up_rmse_wh
local_regret_wh
local_regret_pct
population_match
metric_match
empirical_comparison
warning_flags
final_context_interaction_notes
eligible_for_candidate_ranking
status
```

---

# 114. Runner-up table schema

`sweep_runner_up_table.csv`:

```text
factor
candidate_values_tested
winner
runner_up
runner_up_rank_basis
runner_up_valid
source_metric_precision
status
```

---

# 115. Local-regret ranking table

`local_regret_ranking.csv`:

```text
rank
source_phase
factor
winner_value
runner_up_value
local_regret_wh
local_regret_pct
later_phase_tiebreak
admissible
provisional_candidate_id
selected_for_shortlist
exclusion_reason
```

Only admissible rows receive shortlist selection.

---

# 116. Candidate admissibility audit

`candidate_admissibility_audit.csv`:

```text
provisional_candidate_id
source_phase
factor
primary_value
alternative_value
one_factor_change
derived_fields
feature_contract_valid
revin_valid
dmodel_heads_valid
window_valid
target_loss_space_valid
optimizer_supported
training_engine_supported
WB0_preserved
test_independent
admissible
reason
```

---

# 117. Derived-field audit

`candidate_derived_field_audit.csv`:

```text
candidate_id
changed_factor
derived_field
primary_value
candidate_value
expected_change
scientific_factor_count_increment
status
```

Example:

```text
HEADS changes
→ head_dim changes
→ expected_change=true
→ scientific_factor_count_increment=0.
```

---

# 118. Duplicate audit

`candidate_duplicate_audit.csv`:

```text
provisional_candidate_id
fingerprint
duplicate_of
kept
reason
status
```

---

# 119. Registry exact-match audit

`candidate_registry_match_audit.csv`:

```text
candidate_id
candidate_config_fingerprint
exact_registry_match_found
matching_run_ids
valid_completed_match_exists
selected_evidence_class
status
```

---

# 120. Candidate pool

`transformer_candidate_pool.csv` includes:

```text
TR_C0
all admissible local alternatives
```

not only final top 3.

Fields:

```text
candidate_id
role
source_phase
source_factor
changed_from
changed_to
local_regret_wh
local_regret_pct
config_fingerprint
evidence_class
exact_existing_run_id
parameter_count_if_known
final_context_performance_available
shortlist_rank
selected
status
```

---

# 121. Final shortlist CSV

`transformer_candidate_shortlist.csv`:

```text
shortlist_position
candidate_id
candidate_role
changed_factor
changed_from
changed_to
source_phase
source_sweep
source_local_regret_wh
source_local_regret_pct
evidence_class
exact_existing_run_id
candidate_config_fingerprint
ready_for_phase44
status
```

---

# 122. Final shortlist JSON

`transformer_candidate_shortlist.json`:

```text
version
frozen=true
max_candidates=3
primary_candidate_id
candidate_count
candidates:[...full configs...]
selection_algorithm
excluded_candidates_summary
boundary_protocol=WB0
source_phase41_sensitivity_context
test_status
approved_for_phase44_after_phase43
```

---

# 123. Boundary sensitivity context

`boundary_sensitivity_context.json`:

```text
primary_protocol=WB0
sensitivity_protocol=WB1
wb0_reference_run_id
wb1_run_id
common_validation_rmse_wb0
common_validation_rmse_wb1
common_delta_rmse
validation_coverage_removed
test_coverage_removed_metadata
train_population_equal
common_window_fraction_equal
sensitivity_interpretation
protocol_amendment_required
warnings
```

No Test metrics.

---

# 124. Baseline anchor context

`baseline_anchor_context.json`:

```text
persistence_run_or_artifact
lstm_b0_run_id
transformer_primary_run_id
shared_task_contract
metric_version
population_notes
test_status
```

No candidate ranking across these families yet.

---

# 125. Phase43 LSTM handoff

`phase43_lstm_tuning_handoff.json`:

```text
forecast_task
horizon=1
selected_feature_variant
selected_time_features
selected_target_scaling
selected_lookback
boundary_protocol=WB0
window_population_policy
split_version
scaler checksums
metric_version
primary_validation_metric=RMSE_Wh
test_locked=true
transformer_primary_candidate_id
transformer_primary_run_id
phase42_shortlist_fingerprint
```

---

# 126. Phase44 Rolling-Origin handoff

`phase44_rolling_origin_handoff.json`:

```text
transformer_shortlist_frozen=true
transformer_candidate_ids
candidate_config_fingerprints
candidate_specs
primary_candidate_id
boundary_protocol=WB0
boundary_sensitivity_context_ref
shared forecasting task
metric version
test locked
requires_phase43_tuned_lstm=true
persistence_anchor_ref
phase42_signoff_ref
```

---

# 127. Candidate findings

`candidate_synthesis_findings.csv` possible codes:

```text
PRIMARY_LINEAGE_VERIFIED
LINEAGE_INCONSISTENCY
PRIMARY_RUN_VERIFIED
LOCAL_TIE_PRESENT
SMALL_LOCAL_REGRET
RUNNER_UP_UNAVAILABLE
RUNNER_UP_INVALID
S18_NOT_APPLICABLE
ALTERNATIVE_INADMISSIBLE
REVIN_FEATURESET_CONFLICT
DMODEL_HEAD_CONFLICT
LOOKBACK_POPULATION_CONFLICT
DUPLICATE_CANDIDATE
EXACT_EXISTING_ALT_FOUND
SYNTHESIZED_ALT_CREATED
TR_C1_SELECTED
TR_C2_SELECTED
SHORTLIST_ONLY_PRIMARY
SHORTLIST_TWO_CANDIDATES
SHORTLIST_THREE_CANDIDATES
WB1_SENSITIVITY_CARRIED
TEST_FIREWALL_PRESERVED
INHERITED_WARNING
```

---

# 128. Phase42 discrepancies

`candidate_synthesis_discrepancies.json` taxonomy:

```text
PHASE41_NOT_APPROVED
PROTOCOL_AMENDMENT_PENDING
MISSING_SWEEP_ARTIFACT
LINEAGE_WINNER_REFERENCE_MISMATCH
FINAL_CONFIG_RECONSTRUCTION_FAILURE
PRIMARY_RUN_MISSING
PRIMARY_RUN_INVALID
PRIMARY_CHECKPOINT_UNVERIFIED
METRIC_VERSION_MISMATCH
LOCAL_SWEEP_POPULATION_MISMATCH
LOCAL_RUNNER_UP_INVALID
NEGATIVE_LOCAL_REGRET
ROUNDING_RANK_ERROR
RUNNER_UP_SELECTION_ERROR
CANDIDATE_MULTI_FACTOR_CHANGE
CANDIDATE_AUTOREPAIR_ATTEMPT
CANDIDATE_REVIN_INVALID
CANDIDATE_DMODEL_HEAD_INVALID
CANDIDATE_WINDOW_INVALID
CANDIDATE_TARGET_SPACE_INVALID
CANDIDATE_DUPLICATE
CANDIDATE_FINGERPRINT_COLLISION
UNEXPECTED_REGISTRY_MATCH
GLOBAL_HISTORICAL_RMSE_SORT_USED
CARTESIAN_SEARCH_ATTEMPT
NEW_TRAINING_ATTEMPT
NEW_VALIDATION_EVALUATION_ATTEMPT
WB1_PROMOTED_WITHOUT_AMENDMENT
TEST_LABEL_ACCESSED
TEST_METRIC_ACCESSED
SHORTLIST_CHANGED_AFTER_SIGNOFF
OTHER
```

---

# 129. Acceptance tests — lineage

Required:

```text
[ ] Phase41 approved.
[ ] No pending protocol amendment.
[ ] S1–S18 winner artifacts found.
[ ] S1–S18 reference updates found.
[ ] S19 sensitivity artifacts found.
[ ] Every selected factor reconstructs.
[ ] Every winner→next-reference link checked.
[ ] S18 skip semantics handled if needed.
[ ] Final primary config canonicalized.
[ ] Primary run ID exists.
[ ] Primary run COMPLETED/valid.
[ ] Primary BEST verified.
[ ] Primary protocol WB0.
```

---

# 130. Acceptance tests — evidence

```text
[ ] Every empirical S1–S18 sweep has winner.
[ ] Valid runner-up identified where possible.
[ ] Failed/debug/sanity runs excluded.
[ ] Full-precision Validation RMSE used.
[ ] Local regret computed within sweep only.
[ ] Population match checked within each sweep.
[ ] Metric version checked within each sweep.
[ ] No global sorting of all historical RMSE.
[ ] S19 excluded from candidate regret ranking.
[ ] S18 not-applicable not treated as empirical runner-up.
```

---

# 131. Acceptance tests — candidate synthesis

```text
[ ] TR_C0 always present.
[ ] Every alternative copies primary config.
[ ] Every alternative changes one factor only.
[ ] Derived changes explicitly whitelisted.
[ ] No automatic second-factor repair.
[ ] RevIN applicability checked.
[ ] d_model/head compatibility checked.
[ ] Window/lookback compatibility checked.
[ ] Target/loss-space compatibility checked.
[ ] WB0 fixed for all candidates.
[ ] Candidate fingerprints generated.
[ ] Duplicate candidates removed.
[ ] Registry exact-match search performed.
[ ] Synthesized candidates not assigned fake RMSE.
[ ] Local historical RMSE clearly separated from candidate performance.
```

---

# 132. Acceptance tests — ranking

```text
[ ] Only admissible alternatives ranked.
[ ] Rank primary key = local_regret_wh ascending.
[ ] Secondary = local_regret_pct ascending.
[ ] Tertiary = later phase preferred.
[ ] Final tiebreak = factor name.
[ ] No arbitrary weighted score.
[ ] No runtime/parameter-count override.
[ ] Max total Transformer candidates = 3.
[ ] At most one alternative per factor.
[ ] No fabricated candidate to fill quota.
```

---

# 133. Acceptance tests — handoff

```text
[ ] Transformer shortlist JSON frozen.
[ ] Phase43 shared data/task context written.
[ ] Phase44 candidate specs written.
[ ] Persistence anchor carried.
[ ] LSTM B0 anchor carried.
[ ] WB1 sensitivity context carried.
[ ] Test status NOT_ACCESSED.
[ ] No new training run created.
[ ] No new Validation inference created.
[ ] No Test inference created.
[ ] Phase42 signoff generated.
```

---

# 134. Primary candidate validation context

Phase42 may record existing:

```text
primary Validation RMSE
MAE
R²
best epoch
```

for documentation.

Do not use this metric to globally compare synthesized alternatives with no current-context metric.

---

# 135. Alternative local evidence context

For each T1/T2:

```text
source sweep winner RMSE
source sweep runner-up RMSE
local regret
source configuration at that time
```

must be preserved.

This lets Phase44 interpret why candidate was selected.

---

# 136. Candidate "uncertainty" language

Safe:

```text
This factor had one of the smallest local winner–runner-up Validation RMSE separations among admissible sweeps.
```

Unsafe:

```text
This candidate is expected to be only X Wh worse than primary.
```

unless actually evaluated.

---

# 137. Candidate shortlist reporting

Recommended table:

```text
Candidate | Role | Changed factor | Primary value | Alt value | Local regret | Evidence class | Existing exact run?
```

No fake final-context RMSE column for synthesized candidates.

---

# 138. Human-readable report structure

`candidate_synthesis_report.md`:

```text
1. Purpose
2. Why synthesis is needed after sequential sweeps
3. Phase41 boundary sensitivity gate
4. Reconstructed S1–S18 selected lineage
5. Primary Transformer configuration
6. Sweep evidence ledger
7. Local runner-up methodology
8. Local-regret ranking
9. Candidate admissibility rules
10. Final-context interaction constraints
11. Duplicate/exact-registry-match audit
12. Candidate pool
13. Final shortlist
14. Evidence limitations
15. Persistence/LSTM anchor context
16. Phase43 LSTM tuning handoff
17. Phase44 Rolling-Origin handoff
18. Test firewall
19. Definition of Done
```

---

# 139. README content

`README_CANDIDATE_SYNTHESIS.md` must explain:

```text
No training in Phase42.
Why not globally sort all historical RMSE.
How local regret is computed.
Why candidates change one factor only.
Why maximum shortlist is 3.
Why synthesized candidate has no final-context Validation score.
How incompatibility excludes a candidate.
Why WB1 is sensitivity evidence, not candidate hyperparameter.
How shortlist freezes before rolling-origin.
Where Phase43/Phase44 consume outputs.
```

---

# 140. Phase42 sign-off

Create:

```text
phase_42_signoff.json
```

Minimum:

```text
phase=42
phase_name=Candidate synthesis
version=CANDIDATE_SYNTHESIS-v1
phase41_approved
protocol_amendment_required=false
primary_protocol=WB0
primary_candidate_id=TR_C0_PRIMARY
primary_run_id
primary_candidate_fingerprint
transformer_candidate_count
transformer_candidate_ids
candidate_fingerprints
selected_alt_source_phases
selected_alt_factors
new_training_runs=0
new_validation_evaluations=0
test_status=NOT_ACCESSED
shortlist_frozen=true
phase43_handoff_ready
phase44_handoff_ready
overall_status
created_at
```

---

# 141. Status model

## PASS

```text
Lineage verified
Primary valid
Evidence ledger complete
Admissibility complete
Shortlist deterministic
No new training/evaluation
No Test
Handoffs ready
```

## PASS_WITH_WARNING

Possible:

```text
some historical initialization fingerprints unavailable
some runner-up exact config not in registry
only one admissible alternative
only primary candidate available
inherited noncritical warnings
boundary sensitivity warning but no amendment required.
```

## FAIL

Examples:

```text
lineage inconsistent
primary run invalid
protocol amendment pending
candidate ranking uses incomparable metrics
candidate contains multiple factor changes
Test accessed
new hidden tuning performed.
```

---

# 142. No Test firewall

Hard:

```text
Test labels     = NOT_ACCESSED
Test predictions= NOT_GENERATED
Test metrics    = NOT_COMPUTED
```

S19 Test eligibility metadata may be copied, but no values beyond metadata.

---

# 143. No external data

No new dataset.

No additional weather.

No external benchmark.

---

# 144. No new EDA

No.

---

# 145. No new feature engineering

No.

---

# 146. No new target transform

No.

---

# 147. No new hyperparameter values

Candidate alternatives can only use:

```text
values actually registered/tested in S1–S18.
```

No untested:

```text
new LR
new D
new F
new dropout
new lookback
new RevIN variant.
```

---

# 148. No outside-range extrapolation

If local winner is boundary:

```text
record boundary warning
```

but do not create outside-range candidate.

---

# 149. No candidate based solely on MAE

Local runner-up ranking uses original primary sweep metric:

```text
Validation RMSE Wh.
```

MAE divergence is warning only.

---

# 150. No candidate based on attention

No.

---

# 151. No candidate based on runtime

No.

---

# 152. No candidate based on parameter count

No, except exact-tie behavior already resolved in source sweep.

---

# 153. No candidate based on narrative preference

No.

---

# 154. Phase42 must preserve original sweep decisions

The primary chain remains the official sequential selection result.

T1/T2 do not “undo” winners.

They are:

```text
robustness challengers.
```

---

# 155. Candidate shortlist semantics

Recommended roles:

```text
TR_C0_PRIMARY
= greedy sequential winner

TR_C1_LOCAL_ALT
= closest admissible local decision challenger

TR_C2_LOCAL_ALT
= second closest admissible local decision challenger.
```

---

# 156. Why candidate alternatives are useful

Rolling-origin robustness can reveal:

```text
a tiny one-period Validation advantage
may not persist across time folds.
```

Therefore T1/T2 focus robustness compute on decisions that were locally least separated.

---

# 157. Why not choose worst/best complexity extremes

That would answer a different question.

Phase42 focuses:

```text
near-decision uncertainty
```

not model scaling laws.

---

# 158. Why later-phase tiebreak is reasonable

If two factors have identical local regret:

```text
prefer later source phase
```

because its controlled comparison was performed under a configuration closer to current final selected chain.

This is only a tie-break, not a weighted priority.

---

# 159. Phase43 should not modify Transformer shortlist

Hard.

---

# 160. Phase44 should not modify Transformer shortlist

Hard.

---

# 161. Phase44 may mark candidate training failure

If a locked candidate fails on a fold:

```text
record failure
do not substitute another candidate.
```

Exact handling belongs to Phase44.

---

# 162. Candidate artifacts must be immutable after sign-off

Generate checksums.

If modified:

```text
new version
or Protocol Amendment.
```

---

# 163. Recommended canonical candidate JSON structure

```text
{
  "candidate_id": "...",
  "candidate_role": "...",
  "base_candidate_id": "...",
  "changed_factor": "...",
  "changed_from": "...",
  "changed_to": "...",
  "scientific_config": {...},
  "derived_config": {...},
  "source_evidence": {...},
  "admissibility": {...},
  "registry_match": {...},
  "candidate_config_fingerprint": "...",
  "status": "READY"
}
```

---

# 164. Candidate scientific config vs derived config

Scientific:

```text
feature set
time features
YS
L
pooling
activation
batch
LR
WD
dropout
D
heads
layers
FFN
loss
epoch cap
clipping
RevIN
WB0.
```

Derived:

```text
feature count
head_dim
parameter count
Huber raw-equivalent delta
steps/epoch
RevIN channel count.
```

Derived runtime fields may need recomputation in Phase44.

---

# 165. Do not pre-fill unknown derived runtime fields

Use:

```text
TO_BE_RESOLVED_IN_PHASE44
```

rather than inventing values.

Examples:

```text
parameter count of synthesized candidate
runtime
fold sample count
Huber raw-equivalent delta if scaler context changes.
```

---

# 166. Candidate registry search safety

Exact match requires all scientific fields equal.

Do not consider:

```text
“close enough”
```

run as exact.

---

# 167. Candidate local source run vs candidate full config

Every alternative must store both:

```text
source_runner_up_run_config
```

and:

```text
synthesized_candidate_config.
```

This prevents confusion.

---

# 168. Validation evidence disclaimer

Mandatory:

> The local runner-up metric was observed under the sweep context in which that factor was originally tested. A synthesized final-context candidate has not yet been evaluated as a complete configuration; its performance will be assessed during the registered robustness stage.

---

# 169. S19 sensitivity disclaimer

Mandatory:

> WB1 is a strict-isolation sensitivity protocol and is not treated as a candidate model hyperparameter in Phase42. WB0 remains the predeclared primary forecasting protocol unless a formal Protocol Amendment states otherwise.

---

# 170. Phase43 handoff readiness

Phase42 PASS requires:

```text
phase43_lstm_tuning_handoff.json valid.
```

---

# 171. Phase44 handoff readiness

Phase42 may mark:

```text
phase44_handoff_ready=true
```

for Transformer candidates, but Phase44 execution additionally waits for:

```text
Phase43 tuned LSTM output.
```

---

# 172. Candidate synthesis summary

`candidate_synthesis_summary.json`:

```text
primary_candidate
candidate_count
selected_alt_factors
lineage_status
evidence_rows
admissible_alt_count
excluded_alt_count
duplicate_count
exact_existing_alt_count
synthesized_alt_count
boundary_sensitivity_status
protocol_amendment_required
test_status
phase43_handoff
phase44_handoff
overall_status
```

---

# 173. Recommended Phase42 execution notebook/script flow

```text
Step 42.1  Verify Phase41 signoff
Step 42.2  Verify no protocol amendment pending
Step 42.3  Load S1–S18 winner/reference artifacts
Step 42.4  Reconstruct selected lineage
Step 42.5  Audit winner→reference continuity
Step 42.6  Build canonical TR_C0 config
Step 42.7  Verify TR_C0 existing run/checkpoint
Step 42.8  Build sweep evidence ledger
Step 42.9  Resolve valid runner-up for every S1–S18 sweep
Step 42.10 Compute local regrets
Step 42.11 Generate one-factor provisional candidate specs
Step 42.12 Run final-context admissibility checks
Step 42.13 Canonicalize/fingerprint candidates
Step 42.14 Remove duplicates
Step 42.15 Search Registry for exact matches
Step 42.16 Rank admissible alternatives
Step 42.17 Select T1/T2 up to shortlist cap
Step 42.18 Freeze shortlist
Step 42.19 Attach S19 boundary sensitivity context
Step 42.20 Attach Persistence/LSTM anchors
Step 42.21 Write Phase43 handoff
Step 42.22 Write Phase44 handoff
Step 42.23 Run acceptance tests
Step 42.24 Write discrepancy log
Step 42.25 Write summary/report/README
Step 42.26 Hash artifacts
Step 42.27 Phase sign-off
```

---

# 174. Recommended execution pseudocode

```text
assert phase41.approved_for_phase42
assert not phase41.protocol_amendment_required

artifacts = load_s1_to_s19_machine_readable_artifacts()

lineage = reconstruct_selected_lineage(artifacts)
assert_lineage_consistency(lineage)

primary = build_primary_candidate(
    lineage=lineage,
    boundary_protocol="WB0"
)

assert_existing_valid_run(primary)

evidence = []

for sweep in S1_to_S18:
    if not sweep.empirical_comparison_performed:
        record_no_runner_up(sweep)
        continue

    valid_conditions = filter_valid_conditions(sweep)

    winner = resolve_original_winner(sweep)
    runner_up = second_best_full_precision_rmse(valid_conditions)

    assert_same_population(winner, runner_up)
    assert_same_metric_version(winner, runner_up)

    local_regret = runner_up.rmse - winner.rmse

    provisional = copy(primary)
    provisional.set_factor(
        sweep.factor,
        runner_up.factor_value
    )

    admissibility = validate_one_factor_candidate(
        primary,
        provisional
    )

    evidence.append({
        "sweep": sweep,
        "runner_up": runner_up,
        "local_regret": local_regret,
        "candidate": provisional,
        "admissibility": admissibility
    })

eligible = [
    e for e in evidence
    if e.admissibility.pass
]

canonicalize_and_fingerprint(eligible)
remove_duplicates(eligible)
search_exact_registry_matches(eligible)

ranked = sort(
    eligible,
    key=(
        local_regret_wh ASC,
        local_regret_pct ASC,
        source_phase DESC,
        factor_name ASC
    )
)

shortlist = [primary]

for item in ranked:
    if len(shortlist) == 3:
        break
    if item.factor not already used:
        shortlist.append(item.candidate)

assert each_alt_changes_exactly_one_factor(shortlist)
assert all_candidate_boundary_protocol == "WB0"

freeze_shortlist(shortlist)

write_boundary_sensitivity_context(S19)
write_baseline_anchor_context()
write_phase43_handoff()
write_phase44_handoff()

assert no_new_training_runs
assert no_new_validation_evaluations
assert test_not_accessed

write_all_phase42_artifacts()
signoff()
```

---

# 175. Definition of Done

\[
\boxed{
Verified\ S1\text{-}S19\ Lineage
+
Exact\ Primary\ Transformer
+
Sweep\ Evidence\ Ledger
+
Local\ Runner\text{-}Up\ Regrets
+
One\text{-}Factor\ Admissibility
+
Deterministic\ Top\text{-}3\ Shortlist
+
No\ New\ Training
+
No\ Test
+
Phase43\ Handoff
+
Phase44\ Handoff
}
\]

---

# 176. Final status contract

```text
PHASE 42 does not tune.

It synthesizes candidates.

Primary:
TR_C0 = exact current S1–S18 selected Transformer
        under WB0 primary protocol.

Alternatives:
copy TR_C0
change exactly one factor
to that factor's registered local runner-up value.

Evidence:
within-sweep full-precision Validation RMSE regret only.

Do not:
sort all historical runs globally
combine multiple alternatives
repair invalid candidate with second factor
invent new hyperparameter values
assign fake RMSE to synthesized candidates
train models
evaluate Validation again
access Test.

Ranking:
1. smallest local regret Wh
2. smallest local regret %
3. later phase
4. factor name

Shortlist:
max 3 Transformer candidates total.

WB1:
sensitivity evidence only.

After sign-off:
Transformer shortlist frozen.

Phase43:
tune LSTM under shared task/data fairness context.

Phase44:
rolling-origin robustness evaluates
locked Transformer shortlist
+ tuned LSTM
+ persistence baseline.

No Test.
```

---

# 177. Final check

Correct:

```text
Phase41 PASS
→ reconstruct full winner lineage
→ build exact primary candidate
→ find each sweep's local runner-up
→ compute within-sweep regret
→ synthesize one-factor final-context challengers
→ reject invalid combinations
→ rank deterministically
→ choose max two alternatives
→ freeze shortlist
→ handoff Phase43/44
```

Incorrect:

```text
sort every historical run by RMSE
→ choose top3
```

Incorrect:

```text
combine LR runner-up + dropout runner-up + layer runner-up
```

Incorrect:

```text
run extra Validation experiments to decide shortlist
```

Incorrect:

```text
use Test to choose candidate
```

Chỉ sau khi:

```text
CANDIDATE_SYNTHESIS-v1 = PASS / PASS_WITH_WARNING
```

và Transformer shortlist đã được frozen mới chuyển sang **PHASE 43 — LSTM tuning**.
