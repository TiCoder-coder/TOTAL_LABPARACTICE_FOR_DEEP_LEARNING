# PHASE 52 — ATTENTION EXTRACTION

## Kế hoạch trích xuất, kiểm định, lưu trữ và đóng băng temporal self-attention của ba Final Transformer seeds phục vụ Phase53–57

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Model:** Attention-Aware Transformer Encoder for regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Primary attention object:** Encoder self-attention weights  
**Attention tensor contract:** `[B, H, Q, S]`, self-attention `[B, H, L, L]`  
**Upstream worst-case analysis:** `WORST_ERROR_ANALYSIS-v1`  
**Final checkpoints:** `FINAL_TR_SEED42`, `FINAL_TR_SEED123`, `FINAL_TR_SEED2026`  
**Phase ID:** `PHASE_52_ATTENTION_EXTRACTION`  
**Output version:** `ATTENTION_EXTRACTION-v1`  
**Phase trước:** `Phase_51_Worst-error_analysis.md`  
**Phase sau:** `Phase_53_Attention_heatmaps.md`

---

# 1. Vai trò của Phase 52

Phase52 là **extraction and integrity phase** cho attention.

Phase này không có nhiệm vụ quyết định:

```text
attention head nào tốt nhất
attention pattern nào “đúng”
feature nào quan trọng nhất
case nào nên giữ/bỏ
model nào tốt hơn
```

Phase52 chỉ:

```text
1. Xác minh ba final checkpoints và exact Test-input contract.
2. Xác minh inspection forward path vẫn tương đương standard forward.
3. Trích xuất per-layer/per-head attention với đúng shape semantics.
4. Lưu full dense attention cho scientifically frozen case set từ Phase51.
5. Lưu last-query attention cho toàn bộ FINAL_TEST_POP-v1.
6. Stream-compute all-Test attention diagnostics mà không phải lưu mọi [L,L] matrix.
7. Kiểm tra probability/integrity constraints.
8. Tạo position ↔ timestamp ↔ lag mappings.
9. Đóng băng raw attention artifacts bằng checksum.
10. Handoff đúng source artifacts sang Phase53–57.
```

Nguyên tắc:

\[
\boxed{
Frozen\ Final\ Checkpoints
+
Exact\ Test\ Inputs
+
Per\text{-}Head\ Raw\ Attention
+
Full\ Dense\ Case\ Maps
+
All\text{-}Test\ Last\text{-}Query\ Attention
+
Strict\ Integrity\ Verification
+
No\ Interpretation\ Yet
}
\]

---

# 2. Phase52 không phải attention-interpretation phase

Phase52 không kết luận:

```text
“head 2 quan trọng nhất”
“model chú ý vào 1 giờ gần nhất”
“attention giải thích lỗi”
“feature X gây prediction”
```

Các phase downstream:

```text
Phase53 → attention heatmaps
Phase54 → last-query attention
Phase55 → head comparison
Phase56 → error-conditioned attention
Phase57 → seed-stability attention check.
```

Phase52 tạo source-of-truth artifacts cho các phase đó.

---

# 3. Attention trong project này là temporal token-to-token attention

Transformer input:

```text
[B,L,F]
→ Linear(F,D)
→ positional encoding
→ Transformer Encoder.
```

Mỗi token tương ứng với:

```text
một timestamp lịch sử
```

sau khi toàn bộ feature vector ở timestamp đó đã được project vào embedding.

Do đó raw self-attention trả lời:

```text
query time position
đang phân bổ attention tới
source/key time positions nào.
```

---

# 4. Attention không phải feature importance

Hard interpretation rule:

```text
attention axis = temporal token position
not raw feature dimension.
```

Vì:

```text
Linear(F,D)
```

đã trộn các feature trước self-attention.

Không được đọc một attention weight rồi kết luận:

```text
T2 có importance = ...
RH_1 có attention = ...
```

Phase52 phải ghi caveat này vào metadata/report.

---

# 5. Attention cũng không tự động là causal explanation

Ngay cả temporal attention cao tại một lag:

```text
không chứng minh
timestamp đó gây ra prediction.
```

Attention analysis là:

```text
internal allocation diagnostic.
```

Không phải causal attribution.

---

# 6. Upstream hard gate

Required:

```text
phase_51_signoff.json
phase52_attention_extraction_handoff.json
worst_case_attention_handoff_cases.csv
worst_error_selection_contract_fingerprint.json
```

Hard:

```text
phase52_ready = true.
```

Phase51 status:

```text
PASS
or
PASS_WITH_WARNING.
```

---

# 7. Required final-model artifacts

Load:

```text
final_model_scientific_config.json
final_feature_contract.json
final_preprocessing_contract.json
final_boundary_contract.json
final_revin_contract.json
final_model_lock_fingerprint.json
```

Required Phase46/47:

```text
three_seed_checkpoint_manifest.csv
final_scaler_checksums.json
phase_47_signoff.json
final_test_population_manifest.json
prediction_checksums.json
```

---

# 8. Authorized checkpoints

Exactly:

```text
FINAL_TR_SEED42
FINAL_TR_SEED123
FINAL_TR_SEED2026.
```

No Phase21 B0 checkpoint.

No Phase44 rolling checkpoint.

No rejected candidate.

No LSTM attention.

---

# 9. Checkpoint scientific equality

All three final checkpoints must share:

```text
same FINAL_MODEL_LOCK_SHA256
same scientific config
same feature order
same lookback L*
same d_model
same heads H*
same layers N*
same FFN
same pooling
same RevIN config
same final scalers
same Test population policy.
```

Only trained parameters/seed differ.

---

# 10. Checkpoint strict-load gate

For every seed:

```text
strict=True
missing_keys=0
unexpected_keys=0.
```

Verify:

```text
checkpoint SHA256
config SHA256
recipe SHA256
lock SHA256
state schema
parameter count.
```

Any mismatch:

```text
STOP.
```

---

# 11. Exact attention-aware implementation required

The model used in Phase52 must be the same attention-aware implementation verified in:

```text
ATTENTION_VERIFY-v1
```

with public `MultiheadAttention` behavior:

```text
need_weights=True
average_attn_weights=False.
```

Do not swap to a different implementation merely for visualization.

---

# 12. Inspection path contract

Canonical API concept:

```text
prediction, attention_layers = model(
    x,
    return_attention=True
)
```

where:

```text
attention_layers
=
list length N_layers

attention_layers[layer]
shape
=
[B,H,L,L].
```

Exact API name may differ, but semantics cannot.

---

# 13. Standard path vs inspection path

For the same:

```text
checkpoint
input
eval mode
```

predictions from:

```text
standard forward
```

and:

```text
attention inspection forward
```

must be numerically equivalent within the frozen tolerance.

Attention extraction cannot silently alter predictions.

---

# 14. Prediction-equivalence reference

Phase52 compares inspection-path prediction against the **frozen Phase47 prediction bundle** for the same:

```text
seed
target_id.
```

This is stronger than comparing two fresh forwards only.

---

# 15. Comparison coordinate

Preferred:

```text
model-space output
```

to avoid scaler amplification.

If Phase47 prediction is stored in raw Wh only:

```text
map frozen Phase47 y_pred_wh
back to locked y_model coordinate
using FINAL_SCALING-v1
```

and compare there.

No scaler fitting.

---

# 16. Prediction-equivalence tolerance

First priority:

```text
reuse exact tolerance frozen in ATTENTION_VERIFY-v1
```

if machine-readable.

Fallback canonical if unavailable:

```text
rtol = 1e-5
atol = 1e-5
```

in model-output coordinate.

Tolerance must be locked **before extraction**.

Do not widen tolerance after observing a failure.

---

# 17. Model mode

Hard:

```text
model.eval()
torch.inference_mode()
```

No gradients.

No optimizer.

No dropout sampling.

No MC dropout.

---

# 18. Attention dropout semantics

Because model is in eval mode:

```text
attention dropout is inactive.
```

Therefore returned attention rows should behave like normalized probabilities over source positions, subject only to numerical precision.

---

# 19. Mask contract

Final Transformer uses:

```text
causal mask = None
padding mask = None.
```

Phase52 must not add a mask.

---

# 20. Why no causal mask is correct here

All input tokens:

```text
occur before forecast target.
```

Encoder self-attention is allowed to mix all positions within the historical input window.

For an early query position `q`, attention to a later input position is not leakage with respect to the forecasting target because the entire historical window is observed before predicting `t+1`.

---

# 21. Upper-triangle attention is therefore expected

Full heatmaps may contain:

```text
source position > query position
```

weights.

Do not interpret upper-triangle mass as target leakage.

---

# 22. Attention tensor axis semantics

Canonical:

```text
A[b,h,q,s]
```

where:

```text
b = batch sample
h = attention head
q = query position
s = source/key position.
```

For self-attention:

```text
Q = S = L.
```

---

# 23. Storage axis order is locked

Raw dense arrays must be stored as:

```text
[case, layer, head, query, source]
```

Full-Test last-query arrays:

```text
[target, layer, head, source].
```

No implicit transposition.

---

# 24. Display indexing vs storage indexing

Storage:

```text
layer_idx0 = 0..N-1
head_idx0  = 0..H-1
position0  = 0..L-1.
```

Human-readable display IDs:

```text
Layer 1 = layer_idx0 0
Head 1  = head_idx0 0.
```

Every table should make this explicit.

---

# 25. Temporal order of positions

Input position:

```text
0 = oldest timestamp
L-1 = newest historical timestamp.
```

No reversal during extraction or plotting.

---

# 26. Forecast target position

The forecast target itself:

```text
is not an attention token.
```

Attention source/query positions cover only:

```text
historical input rows.
```

---

# 27. Last-query attention

Canonical:

```text
A[b,h,L-1,:]
```

not:

```text
A[b,h,:,L-1].
```

This vector answers:

> For the newest historical token, how is attention distributed across the historical source positions?

---

# 28. Position-to-lag mapping

For source position `p`:

\[
LagSteps_p
=
H+(L-1-p).
\]

With project:

```text
H=1.
```

Therefore:

\[
LagSteps_p=L-p.
\]

And:

\[
LagMinutes_p=10(L-p).
\]

---

# 29. Lag examples

If final:

```text
L=144
```

then:

```text
source position 143 → lag 1   → 10 min
source position 138 → lag 6   → 60 min
source position 108 → lag 36  → 6 h
source position 72  → lag 72  → 12 h
source position 0   → lag 144 → 24 h.
```

If L differs, mapping follows formula.

---

# 30. Query-position timestamp mapping

For a case with target timestamp `T_target` and H=1:

```text
input source/query position p
=
T_target - (L-p)*10 minutes
```

subject to actual locked timeline/index mapping.

Preferred:

```text
derive from Window Builder indices
```

rather than reconstruct timestamp arithmetic alone.

---

# 31. Position map artifact

Create one global relative map:

```text
attention_relative_position_map.csv
```

Fields:

```text
position_idx0
position_display
lag_steps_from_target
lag_minutes_from_target
relative_time_label.
```

---

# 32. Case-specific timestamp map

For every dense attention case:

```text
case_id
target_id
target_timestamp
position_idx0
input_timestamp
lag_steps_from_target
lag_minutes_from_target
```

stored in:

```text
attention_case_position_map.csv.
```

---

# 33. Why both relative and case-specific maps are needed

Relative lag supports:

```text
aggregation across samples.
```

Case timestamp mapping supports:

```text
case heatmaps and narrative alignment.
```

---

# 34. Extraction design has two storage tiers

To avoid unnecessary multi-gigabyte raw storage while still supporting Phase53–57:

```text
Tier A — Dense case attention
Tier B — All-Test last-query attention + streaming head summaries.
```

---

# 35. Tier A — Dense case attention

Source case set:

```text
worst_case_attention_handoff_cases.csv
```

from Phase51.

This table is already deduplicated by target ID while preserving selection roles.

---

# 36. Dense case set content

It should include union of:

```text
W2 shared top20
per-seed W1 top20
shared ALL_UNDER top10
shared ALL_OVER top10.
```

Potential overlap is deduplicated.

Runtime case count:

```text
K_ATTN_CASES
```

must be measured, not fabricated.

---

# 37. Same dense case set for all seeds

Critical:

```text
seed42
seed123
seed2026
```

must extract dense attention for the **same deduplicated target IDs**.

Even if a case entered the set because it was worst only for seed42, attention for that case is still extracted from all three checkpoints.

This is required for Phase57 seed-stability comparison.

---

# 38. Dense attention storage

For each seed store:

```text
dense_case_attention_seed42.npz
dense_case_attention_seed123.npz
dense_case_attention_seed2026.npz
```

Canonical tensor:

```text
[K_ATTN_CASES,N_layers,N_heads,L,L]
```

dtype:

```text
float32.
```

---

# 39. Why float32 raw storage

Use the model attention precision without lossy down-casting.

Do not store raw scientific source as:

```text
float16
uint8
image pixels.
```

Plots can be rendered later from float32 source.

---

# 40. Compression

Compressed NPZ/Zarr/HDF5 may be used.

Whichever format is selected must preserve:

```text
float32 values
axis order
target ID ordering
metadata.
```

For this plan, canonical file naming assumes:

```text
NPZ.
```

---

# 41. Tier B — Full-Test last-query attention

For every target in:

```text
FINAL_TEST_POP-v1
```

and every seed, persist:

```text
A_last[target,layer,head,source]
=
A[target,layer,head,L-1,source].
```

Canonical shape:

```text
[N_test,N_layers,N_heads,L].
```

---

# 42. Full-Test last-query files

```text
last_query_attention_seed42.npz
last_query_attention_seed123.npz
last_query_attention_seed2026.npz.
```

These support:

```text
Phase54 last-query analysis
Phase55 head comparison
Phase56 error-conditioned analysis
Phase57 seed stability.
```

---

# 43. Why full-Test last-query is storage-efficient

Full dense attention scales as:

\[
O(NHL^2)
\]

while last-query storage scales as:

\[
O(NHL).
\]

This preserves the forecasting-relevant newest-query distribution without retaining every query×source matrix for every Test sample.

---

# 44. Full-Test dense attention is not required by default

Do not store all:

```text
[N_test,N_layers,N_heads,L,L]
```

for all seeds unless a later formally approved analysis requires it.

Phase52 streams full matrices batch-by-batch and discards non-case dense matrices after deriving required summaries.

---

# 45. Streaming extraction workflow

For each batch and seed:

```text
1. inspection forward → predictions + full attention tensors
2. verify prediction equivalence
3. verify attention integrity
4. extract last-query vector for all samples
5. compute all-query/head summary statistics
6. if target is in frozen dense case set:
       persist full [layer,head,L,L]
7. discard non-case full [L,L] tensors
8. continue.
```

This is the preferred memory/storage pattern.

---

# 46. Dedicated attention extraction batch size

Canonical default:

```text
ATTN_EXTRACT_BATCH = 8.
```

Reason:

```text
attention memory scales with B × H × L² × layers.
```

This extraction batch size is a technical parameter, not a scientific model hyperparameter.

---

# 47. OOM fallback policy

If batch 8 cannot execute due device memory:

```text
8 → 4 → 2 → 1.
```

Use the first size that runs.

This is allowed only as a technical extraction fallback.

Must record:

```text
attempted batch sizes
failure type
final extraction batch size.
```

---

# 48. No favorable batch-size selection

Do not compare attention outputs from several batch sizes and choose one because plots look cleaner.

The fallback is memory-driven only.

---

# 49. Batch-independence verification

Before official extraction, on a small deterministic **pre-Test probe** or first authorized extraction IDs:

compare same sample:

```text
alone
vs
inside extraction batch.
```

Expected predictions and attention:

```text
allclose
```

under frozen tolerance.

If exact environment yields small numeric differences, document and use one fixed batch size for all three seeds.

---

# 50. Same extraction batch size across seeds

Preferred hard contract:

```text
same ATTN_EXTRACT_BATCH
```

for seed42/123/2026.

If one seed uniquely OOMs under identical model schema, investigate environment before accepting per-seed batch differences.

---

# 51. Test loader order

Use:

```text
shuffle=false
drop_last=false
```

and exact:

```text
FINAL_TEST_POP-v1 target ID order.
```

---

# 52. Input preprocessing

Use exactly:

```text
final feature order
FINAL_SCALING-v1
final RevIN semantics
final lookback
WB0.
```

No scaler fitting.

No feature changes.

---

# 53. Y target not required for extraction

Attention extraction requires:

```text
x
target_id
timestamp.
```

Error/regime metadata comes from frozen downstream analysis tables.

No need to recompute Test metrics.

---

# 54. Prediction verification still required

Although `y_true` is unnecessary for extraction, inspection-path:

```text
y_pred
```

must be compared to frozen Phase47 prediction values.

This detects implementation/path drift.

---

# 55. All-Test target ordering manifest

Create:

```text
attention_test_target_order.csv
```

with:

```text
attention_row_idx
target_id
target_timestamp
Phase47_prediction_row_idx/reference
```

Same ordering across all three seed last-query files.

---

# 56. Dense-case ordering manifest

Create:

```text
attention_dense_case_order.csv
```

with:

```text
case_row_idx
target_id
target_timestamp
selection_roles
shared_rank
seed-specific ranks
signed ranks.
```

Same ordering across three dense attention files.

---

# 57. Raw attention integrity — shape

For every layer tensor:

```text
ndim = 4
shape[0] = actual batch size
shape[1] = N_heads
shape[2] = L
shape[3] = L.
```

Layer-list length:

```text
N_layers.
```

---

# 58. Raw attention integrity — finite

Hard:

```text
all attention values finite.
```

NaN/Inf:

```text
FAIL extraction run.
```

No row/sample skipping.

---

# 59. Raw attention integrity — nonnegative

Because weights follow softmax in eval:

```text
min attention weight >= 0
```

up to only tiny numerical tolerance.

Canonical audit:

```text
min_weight >= -1e-7.
```

A material negative value:

```text
STOP.
```

---

# 60. Raw attention integrity — row sums

For every:

```text
sample
layer
head
query
```

verify:

\[
\sum_s A[b,h,q,s]\approx1.
\]

Use frozen attention-integrity tolerance.

Preferred:

```text
reuse ATTENTION_VERIFY-v1.
```

Fallback:

```text
atol=1e-5
rtol=1e-5.
```

---

# 61. No renormalization to hide a failure

Do not:

```text
A = A / A.sum(...)
```

before the audit.

Raw extracted weights must pass integrity on their own.

Derived downstream copies may only be renormalized if a later analysis mathematically requires it and clearly labels that transformation; primary source remains raw.

---

# 62. Attention max bound

Expected:

```text
max <= 1 + numerical tolerance.
```

Audit.

---

# 63. Layer/head count integrity

All seeds:

```text
same N_layers
same N_heads.
```

Hard.

---

# 64. Attention-source immutability

Raw files:

```text
dense_case_attention_seed*.npz
last_query_attention_seed*.npz
```

become read-only scientific sources after checksum verification.

Phase53–57 must derive from them without overwrite.

---

# 65. No head averaging in raw source

Preserve:

```text
every head separately.
```

Do not save only:

```text
mean across heads.
```

Head averaging belongs downstream derived analysis.

---

# 66. No layer averaging in raw source

Preserve every layer separately.

---

# 67. No seed averaging in raw source

Preserve every checkpoint separately.

---

# 68. Whole-matrix streaming summaries

Although dense full matrices are discarded for non-case Test targets, Phase52 can compute compact per-sample/per-layer/per-head summaries before discard.

Required:

```text
mean_query_entropy
mean_self_attention_weight
mean_source_position
mean_absolute_query_source_distance
backward_or_same_mass
forward_within_input_mass
```

These are extraction diagnostics/support, not scientific conclusions.

---

# 69. Mean query entropy

For each query row:

\[
H_q=-\sum_s A_{q,s}\log(A_{q,s}+\epsilon_H).
\]

Use numerical epsilon:

```text
epsilon_H = machine-safe constant
```

only inside log, e.g. `1e-12`.

Then:

\[
MeanQueryEntropy
=
Mean_q(H_q).
\]

---

# 70. Normalized entropy

Also compute:

\[
H^{norm}_q
=
\frac{H_q}{\log L}
\]

for `L>1`.

Range:

```text
approximately 0..1.
```

This makes entropy comparable if lookback differs across experiments, though final L is fixed here.

---

# 71. Entropy is concentration diagnostic

Low entropy:

```text
more concentrated distribution.
```

High entropy:

```text
more diffuse distribution.
```

Do not call entropy “attention quality”.

---

# 72. Mean self-attention weight

For a head matrix:

\[
SelfMassMean
=
\frac{1}{L}\sum_q A[q,q].
\]

This measures average token self-attention weight.

No causal interpretation.

---

# 73. Forward-within-input mass

For each query `q`:

```text
source > q
```

is later within the already-observed historical input window.

Compute:

\[
ForwardMass_q
=
\sum_{s>q}A[q,s].
\]

Then average over queries.

---

# 74. Backward-or-same mass

\[
BackwardSameMass_q
=
\sum_{s\le q}A[q,s].
\]

In no-mask Encoder:

```text
ForwardMass + BackwardSameMass ≈ 1.
```

Audit can use this identity.

---

# 75. Do not call forward-within-input mass “future leakage”

It is not future relative to forecast target.

Use exact label:

```text
forward-within-input.
```

---

# 76. Query-source distance

For each row:

\[
D_q
=
\sum_s A[q,s]|q-s|.
\]

Average:

```text
mean_absolute_query_source_distance.
```

This describes temporal mixing within the historical window.

---

# 77. Last-query derived summaries

For every:

```text
target
layer
head
```

compute from:

```text
a_s=A[L-1,s].
```

Required:

```text
entropy
normalized_entropy
expected_lag_steps
expected_lag_minutes
top1_source_position
top1_lag_steps
top1_lag_minutes
top1_weight
top5_mass
recent_1h_mass
recent_6h_mass
recent_12h_mass
recent_24h_mass
```

subject to final L support.

---

# 78. Expected lag

\[
E[LagSteps]
=
\sum_s a_s LagSteps_s.
\]

And:

\[
E[LagMinutes]
=
10\times E[LagSteps].
\]

---

# 79. Top-5 mass

Let top five weights of last-query vector be:

```text
a_(1)...a_(5).
```

Then:

\[
Top5Mass=\sum_{k=1}^{min(5,L)}a_{(k)}.
\]

No source-position adjacency requirement.

---

# 80. Recent-window masses

Define by lag steps:

```text
recent_1h:
lag <= min(6,L)

recent_6h:
lag <= min(36,L)

recent_12h:
lag <= min(72,L)

recent_24h:
lag <= min(144,L).
```

If final lookback is shorter than a named horizon:

```text
record effective steps
and
coverage_truncated=true.
```

Do not imply the model sees beyond L.

---

# 81. Last-query vector sum audit

For every row:

\[
\sum_s a_s\approx1.
\]

Hard.

---

# 82. Top1 deterministic tie rule

If several source positions have exactly equal max weight:

```text
choose newest source among tied maxima
```

for the single `top1_source_position` field.

Also record:

```text
top1_tie_count.
```

This tie rule is purely reporting metadata.

---

# 83. Why newest tied source

It provides deterministic resolution aligned with forecasting recency.

It does not alter raw attention.

---

# 84. Head IDs across seeds are not guaranteed semantically aligned

Critical caveat:

```text
Head 1 in seed42
```

and:

```text
Head 1 in seed123
```

share architecture index but may learn functionally permuted roles.

Therefore Phase52 preserves raw head index but does **not** assume cross-seed same-index semantic equivalence.

---

# 85. Phase57 implication

Seed-stability analysis must consider:

```text
same-index comparison
and/or
explicit head matching based on pattern similarity
```

before strong claims.

Phase52 handoff records this caveat.

---

# 86. Layer indices are architecturally aligned

Layer 1/2 positions in the stack are structurally aligned across seeds, unlike head functional identities which can permute within a layer.

Still, learned behavior may differ.

---

# 87. Dense case metadata

For every dense case:

```text
target_id
target_timestamp
selection roles
Phase50 regimes
Phase51 shared hardness
seed-specific errors
seed spread
sign consensus
input start/end
```

should accompany attention arrays.

Do not embed all metadata ambiguously inside NPZ only; provide CSV/JSON manifest.

---

# 88. Full-Test last-query metadata

For every target:

```text
target_id
timestamp
Phase49 residual/error per seed
Phase50 regimes
Phase51 shared hardness rank if any.
```

May be joined downstream rather than duplicated in NPZ.

Raw attention metadata should retain:

```text
target IDs + ordering.
```

---

# 89. Case selection independent of attention

Hard:

```text
attention_case_set
```

comes only from frozen Phase51 selection.

No case may be added/removed because its attention map is:

```text
interesting
flat
ugly
clear.
```

---

# 90. Optional extraction QA case

A small deterministic Test or pre-Test sample may be used for extraction smoke testing.

It is not added to scientific dense case set unless already selected.

---

# 91. No attention-conditioned data filtering in Phase52

No.

---

# 92. No raw-attention thresholding

Do not zero:

```text
small weights
```

before storage.

No top-k sparsification of raw source.

---

# 93. No smoothing

Do not temporally smooth attention vectors/matrices before storage.

---

# 94. No interpolation

No.

---

# 95. No min-max normalization for source

No.

Heatmap color scaling will be handled in Phase53.

---

# 96. No log transform for raw source

No.

---

# 97. Attention extraction reproducibility

For the same:

```text
checkpoint
input
eval mode
batch size
environment
```

repeated extraction should be deterministic/allclose under ENV-v1.

Phase52 runs a small duplicate extraction audit.

---

# 98. Duplicate extraction audit

Select deterministic small set:

```text
first 2 dense cases
```

or fewer if case count <2.

Run inspection twice without mutation.

Compare:

```text
prediction
attention tensors.
```

Expected:

```text
allclose.
```

Do not rerun full Test merely for duplication.

---

# 99. Cross-device rerun is not required

No need to prove CPU/MPS/CUDA attention bit equivalence.

Record actual extraction environment.

---

# 100. Extraction environment audit

For each seed:

```text
Python version
PyTorch version
device type
device name
precision
extraction batch size
environment fingerprint.
```

Prefer same environment for all seeds.

---

# 101. AMP policy

Use locked final precision policy:

```text
AMP=false
```

unless the final model lock explicitly says otherwise.

Do not enable AMP only for extraction.

---

# 102. Raw tensor device handling

Recommended:

```text
attention.detach()
→ cpu()
→ float32
→ validated
→ persisted.
```

No gradient graph exists under inference mode.

---

# 103. Do not detach after modifying

Raw attention must be captured before any derived transformations.

---

# 104. Storage atomicity

For each raw file:

```text
write temp
close/fsync
rename atomically
compute SHA256
reload file
verify shape/order/checksum.
```

Mark artifact complete only after reload verification.

---

# 105. Partial extraction recovery

Because full-Test extraction can be long, chunked temporary storage is allowed.

But official raw files become authoritative only after complete target coverage.

Do not compute downstream analyses from partial files.

---

# 106. Resume policy

If extraction interrupted:

```text
resume by exact target IDs/chunks
```

only if:

```text
same checkpoint
same lock
same scaler
same batch semantics
same environment contract
same extraction code/version.
```

Otherwise restart seed extraction.

---

# 107. No duplicate target rows on resume

Hard dedup audit.

---

# 108. Full-Test coverage audit

For each seed last-query file:

```text
N_rows = N_test
unique target IDs = N_test
order = attention_test_target_order.csv
no missing target
no duplicate target.
```

---

# 109. Dense-case coverage audit

For each seed:

```text
N_dense_cases = K_ATTN_CASES
exact same case IDs/order.
```

---

# 110. Dense/full last-query consistency

For every dense case:

extract:

```text
dense[layer,head,L-1,:]
```

and compare to corresponding:

```text
full-Test last_query_attention
```

for same:

```text
seed
target
layer
head.
```

Hard allclose.

This is a powerful storage consistency test.

---

# 111. Relative-lag consistency

For every stored source position:

```text
position 0 → largest lag
position L-1 → lag1.
```

Hard mapping audit.

---

# 112. Target timestamp consistency

For each case:

```text
input_end_timestamp = target_timestamp - 10 min
```

under H1 and exact cadence, unless source timeline metadata shows a valid equivalent mapping.

Since windows require continuity:

```text
this should hold.
```

Audit.

---

# 113. Attention extraction does not alter checkpoint

No model parameter mutation.

Optionally hash state_dict before and after extraction.

Expected:

```text
same checkpoint/model parameter fingerprint.
```

---

# 114. Model-mutation audit

For each seed:

```text
state fingerprint before extraction
state fingerprint after extraction
match=true.
```

Buffers that are intentionally runtime-mutated should not exist under current model semantics; any mismatch requires investigation.

---

# 115. RevIN context

If RN1 enabled:

```text
RevIN instance statistics
```

are computed exactly as locked during the forward.

Attention is therefore extracted from the actual normalized input path used by the final model.

Do not disable RevIN “for interpretability”.

---

# 116. RevIN does not change attention axis meaning

Tokens still correspond to historical timestamps.

But token values entering projection have been normalized according to the frozen RevIN semantics.

Record RN state in metadata.

---

# 117. Pooling independence caveat

Attention maps come from encoder layers before final pooling.

If final pooling is:

```text
LAST_STEP
```

last-query attention has a natural connection to the token used by regression head.

If final pooling is:

```text
MEAN
```

the final prediction uses all encoded positions, so last-query attention is only one diagnostic view.

Phase52 must record actual pooling.

---

# 118. Pooling-specific interpretation flag

Metadata:

```text
last_query_directly_corresponds_to_pooled_token
=
true if LAST_STEP
false if MEAN.
```

This will matter in Phase54.

---

# 119. Full dense attention remains required for MEAN pooling cases

Because mean pooling uses all encoder positions, dense case maps are especially important if final P1 mean pooling won.

Do not rely only on last-query in that scenario.

---

# 120. Layer normalization / residual path caveat

Attention weights are only one part of each encoder block.

Prediction also depends on:

```text
value projections
residual connections
LayerNorm
FFN
later layers
pooling
regression head.
```

Therefore attention magnitude is not equivalent to direct contribution magnitude.

---

# 121. Value-vector caveat

A high attention weight can multiply a value vector with different content/magnitude.

Raw attention weight alone does not equal output contribution.

This caveat must appear in README/report.

---

# 122. Multi-layer caveat

Attention in earlier layers can be transformed by later layers.

Do not treat Layer1 attention as direct final decision explanation.

---

# 123. Multi-head caveat

Heads are combined through output projection.

No single head is independently the prediction.

---

# 124. Extraction summary statistics are secondary

Required raw artifacts are primary.

Derived summary CSVs can be regenerated from raw source.

---

# 125. Output directory

```text
artifacts/
└── attention_extraction/
    ├── attention_extraction_manifest.json
    ├── attention_extraction_contract.json
    ├── phase52_preflight_audit.csv
    ├── attention_source_verification.csv
    ├── attention_checkpoint_verification.csv
    ├── attention_environment_audit.csv
    ├── attention_extraction_batch_audit.csv
    ├── attention_prediction_equivalence_audit.csv
    ├── attention_tensor_integrity_audit.csv
    ├── attention_probability_audit.csv
    ├── attention_model_mutation_audit.csv
    ├── attention_reproducibility_audit.csv
    ├── attention_test_target_order.csv
    ├── attention_dense_case_order.csv
    ├── attention_relative_position_map.csv
    ├── attention_case_position_map.csv
    ├── attention_case_metadata.csv
    ├── raw/
    │   ├── dense_case_attention_seed42.npz
    │   ├── dense_case_attention_seed123.npz
    │   ├── dense_case_attention_seed2026.npz
    │   ├── last_query_attention_seed42.npz
    │   ├── last_query_attention_seed123.npz
    │   └── last_query_attention_seed2026.npz
    ├── raw_attention_checksums.json
    ├── attention_dense_last_query_consistency.csv
    ├── attention_last_query_summary.csv
    ├── attention_full_matrix_summary.csv
    ├── attention_recent_mass_summary.csv
    ├── attention_top_source_summary.csv
    ├── attention_extraction_findings.csv
    ├── attention_extraction_tests.csv
    ├── attention_extraction_discrepancies.json
    ├── phase53_attention_heatmaps_handoff.json
    ├── phase54_last_query_attention_handoff.json
    ├── phase55_head_comparison_handoff.json
    ├── phase56_error_conditioned_attention_handoff.json
    ├── phase57_seed_stability_attention_handoff.json
    ├── attention_extraction_summary.json
    ├── attention_extraction_report.md
    ├── README_ATTENTION_EXTRACTION.md
    └── phase_52_signoff.json
```

---

# 126. Required outputs

```text
O52.1  Extraction manifest
O52.2  Extraction contract
O52.3  Preflight audit
O52.4  Source verification
O52.5  Checkpoint verification
O52.6  Environment audit
O52.7  Extraction-batch audit
O52.8  Prediction-equivalence audit
O52.9  Tensor-shape/integrity audit
O52.10 Probability audit
O52.11 Model-mutation audit
O52.12 Reproducibility audit
O52.13 Full-Test target ordering
O52.14 Dense-case ordering
O52.15 Relative position/lag map
O52.16 Case-specific timestamp map
O52.17 Case metadata
O52.18 Three dense case raw-attention files
O52.19 Three all-Test last-query raw-attention files
O52.20 Raw attention checksums
O52.21 Dense/last-query consistency audit
O52.22 Last-query summary
O52.23 Full-matrix streaming summary
O52.24 Recent-mass summary
O52.25 Top-source summary
O52.26 Findings
O52.27 Phase53 handoff
O52.28 Phase54 handoff
O52.29 Phase55 handoff
O52.30 Phase56 handoff
O52.31 Phase57 handoff
O52.32 Tests
O52.33 Discrepancies
O52.34 Summary JSON
O52.35 Human-readable report
O52.36 README
O52.37 Sign-off
```

---

# 127. Extraction manifest

`attention_extraction_manifest.json` minimum:

```text
phase=52
version=ATTENTION_EXTRACTION-v1
source_phase51_version
final_lock_sha256
test_population_sha256
worst_case_selection_contract_sha256
seed_list=[42,123,2026]
lookback_steps
num_layers
num_heads
pooling
RevIN_state
attention_tensor_semantics=[B,H,Q,S]
self_attention_shape=[B,H,L,L]
raw_dense_storage_shape=[K,Layers,Heads,L,L]
last_query_storage_shape=[N_test,Layers,Heads,L]
raw_dtype=float32
default_extraction_batch=8
new_training=false
model_selection=false
case_selection_changed=false
attention_interpretation=false
status
created_at
```

Runtime `K`/`N_test` populated only during execution.

---

# 128. Extraction contract

`attention_extraction_contract.json` must freeze:

```text
Authorized checkpoints:
42,123,2026 final refit only.

Model:
same attention-aware final Transformer.

Mode:
eval + inference_mode.

Attention:
need_weights=true
average_attn_weights=false
no masks.

Raw axes:
B,H,Q,S.

Dense case storage:
case,layer,head,query,source.

All-Test last-query:
target,layer,head,source.

Dense cases:
exact frozen Phase51 case set
same IDs for all seeds.

Last-query:
entire FINAL_TEST_POP-v1
all seeds.

No:
head averaging
layer averaging
seed averaging
thresholding
smoothing
renormalizing
case changes
interpretation.
```

---

# 129. Preflight audit

`phase52_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Required:

```text
Phase51 approved
phase52_ready=true
Phase51 case table exists
case-selection fingerprint matches
3 final checkpoints exist
checkpoint checksums match
same final lock
same model schema
final scalers match
final feature fingerprint matches
Test population matches
attention-aware API available
inspection mode available
no-mask contract matches
pooling identified
RevIN state identified
attention tolerance frozen
output directories ready.
```

---

# 130. Source verification

`attention_source_verification.csv`:

```text
source_id
path
expected_sha256
observed_sha256
population_sha256_if_applicable
frozen
status
```

Include:

```text
Phase51 case table
Phase47 prediction bundles
final checkpoint manifest
final scaler bundle
feature contract.
```

---

# 131. Checkpoint verification schema

`attention_checkpoint_verification.csv`:

```text
seed
run_id
checkpoint_path
checkpoint_sha256
expected_sha256
checkpoint_type
official_epoch
lock_hash_match
config_hash_match
scaler_refs_match
strict_load
parameter_count
state_schema_sha256
status
```

---

# 132. Environment audit

`attention_environment_audit.csv`:

```text
seed
python_version
torch_version
device_type
device_name
precision
AMP
environment_fingerprint
matches_final_contract
status
```

---

# 133. Extraction-batch audit

`attention_extraction_batch_audit.csv`:

```text
seed
attempted_batch_size
result
failure_type_if_any
selected_batch_size
memory_fallback_used
same_as_other_seeds
status
```

---

# 134. Prediction-equivalence audit

`attention_prediction_equivalence_audit.csv`:

```text
seed
N_checked
comparison_space
rtol
atol
max_abs_difference
max_relative_difference
allclose_fraction
all_pass
status
```

Preferred:

```text
N_checked = entire FINAL_TEST_POP-v1
```

because inspection forward runs for all Test targets anyway.

---

# 135. Tensor integrity audit

`attention_tensor_integrity_audit.csv`:

```text
seed
layer_idx0
expected_heads
observed_heads
expected_L
observed_Q
observed_S
finite
min_weight
max_weight
shape_valid
status
```

Could aggregate across batches after checking every batch.

---

# 136. Probability audit

`attention_probability_audit.csv`:

```text
seed
layer_idx0
head_idx0
row_count_checked
min_row_sum
max_row_sum
mean_row_sum
max_abs_row_sum_minus_1
min_weight
max_weight
nonnegative_pass
row_sum_pass
status
```

No renormalization.

---

# 137. Model-mutation audit

`attention_model_mutation_audit.csv`:

```text
seed
state_fingerprint_before
state_fingerprint_after
same
parameter_mutation_detected
status
```

Expected:

```text
same=true.
```

---

# 138. Reproducibility audit

`attention_reproducibility_audit.csv`:

```text
seed
target_id
repeat_index
prediction_max_abs_diff
attention_max_abs_diff
rtol
atol
allclose
status
```

Small deterministic subset only.

---

# 139. Full-Test target-order schema

`attention_test_target_order.csv`:

```text
attention_row_idx0
target_id
target_timestamp
population_sha256
status
```

Same file applies to all three seeds.

---

# 140. Dense-case order schema

`attention_dense_case_order.csv`:

```text
case_row_idx0
target_id
target_timestamp
selection_roles
shared_rank_if_any
seed42_rank_if_any
seed123_rank_if_any
seed2026_rank_if_any
shared_all_under_rank_if_any
shared_all_over_rank_if_any
status
```

---

# 141. Relative position map schema

`attention_relative_position_map.csv`:

```text
position_idx0
position_display
lag_steps_from_forecast_target
lag_minutes_from_forecast_target
is_newest_input_position
is_oldest_input_position
status
```

Expected:

```text
position L-1 → lag1
position 0   → lagL.
```

---

# 142. Case position map schema

`attention_case_position_map.csv`:

```text
target_id
target_timestamp
position_idx0
input_timestamp
lag_steps_from_target
lag_minutes_from_target
same_continuity_segment
status
```

---

# 143. Case metadata schema

`attention_case_metadata.csv`:

```text
target_id
target_timestamp
selection_roles
shared_rank
seed42_rank
seed123_rank
seed2026_rank
mean_abs_error_across_seeds
seed_range_prediction
residual_sign_consensus
target_level_regime
extreme_high_regime
change_magnitude_regime
change_direction_regime
time_of_day_regime
day_type_regime
input_start_timestamp
input_end_timestamp
status
```

---

# 144. Dense raw NPZ schema

Each file should contain:

```text
attention
target_ids
target_timestamps_or_reference
layer_count
head_count
lookback_steps
axis_order
dtype
checkpoint_sha256
final_lock_sha256
case_order_sha256
position_map_sha256.
```

Primary:

```text
attention.shape
=
[K,Layers,Heads,L,L].
```

---

# 145. Last-query NPZ schema

Each file:

```text
last_query_attention
target_ids
layer_count
head_count
lookback_steps
axis_order
checkpoint_sha256
final_lock_sha256
target_order_sha256
position_map_sha256.
```

Shape:

```text
[N_test,Layers,Heads,L].
```

---

# 146. Raw attention checksums

`raw_attention_checksums.json`:

```text
dense_seed42_sha256
dense_seed123_sha256
dense_seed2026_sha256
last_query_seed42_sha256
last_query_seed123_sha256
last_query_seed2026_sha256
case_order_sha256
target_order_sha256
relative_position_map_sha256
case_position_map_sha256
status
```

---

# 147. Dense/last-query consistency schema

`attention_dense_last_query_consistency.csv`:

```text
seed
target_id
layer_idx0
head_idx0
max_abs_difference
allclose
status
```

Expected:

```text
dense[...,L-1,:]
==
last_query corresponding vector.
```

---

# 148. Last-query summary schema

`attention_last_query_summary.csv`:

```text
seed
target_id
target_timestamp
layer_idx0
layer_display
head_idx0
head_display
entropy
normalized_entropy
expected_lag_steps
expected_lag_minutes
top1_source_position_idx0
top1_lag_steps
top1_lag_minutes
top1_weight
top1_tie_count
top5_mass
status
```

---

# 149. Recent-mass schema

`attention_recent_mass_summary.csv`:

```text
seed
target_id
layer_idx0
head_idx0
mass_recent_1h
effective_steps_1h
coverage_truncated_1h
mass_recent_6h
effective_steps_6h
coverage_truncated_6h
mass_recent_12h
effective_steps_12h
coverage_truncated_12h
mass_recent_24h
effective_steps_24h
coverage_truncated_24h
status
```

---

# 150. Full-matrix summary schema

`attention_full_matrix_summary.csv`:

```text
seed
target_id
layer_idx0
head_idx0
mean_query_entropy
mean_normalized_query_entropy
mean_self_attention_weight
mean_absolute_query_source_distance_steps
mean_backward_or_same_mass
mean_forward_within_input_mass
status
```

Generated streaming for all Test targets.

---

# 151. Top-source summary

`attention_top_source_summary.csv` may contain compact top-k source positions for last-query:

```text
seed
target_id
layer_idx0
head_idx0
rank
source_position_idx0
lag_steps
lag_minutes
weight
status
```

Lock:

```text
TOP_K_SOURCES = 5.
```

This table is derived; raw vector remains authoritative.

---

# 152. No case ranking by attention in Phase52

Top-k here means:

```text
top source positions within one attention vector
```

not:

```text
top cases.
```

Case selection remains Phase51.

---

# 153. Extraction findings codes

Possible:

```text
ATTENTION_API_VERIFIED
PREDICTION_EQUIVALENCE_VERIFIED
ATTENTION_SHAPE_VERIFIED
ATTENTION_ROWS_NORMALIZED
ATTENTION_NONNEGATIVE
NO_MODEL_MUTATION
EXTRACTION_REPRODUCIBLE
DENSE_CASE_COVERAGE_COMPLETE
FULL_TEST_LAST_QUERY_COVERAGE_COMPLETE
DENSE_LAST_QUERY_CONSISTENCY_VERIFIED
POSITION_LAG_MAPPING_VERIFIED
POOLING_LAST_STEP
POOLING_MEAN
RN0_EXTRACTION
RN1_EXTRACTION
OOM_BATCH_FALLBACK_USED
NO_OOM_BATCH_FALLBACK
RAW_ATTENTION_FROZEN
HEAD_ID_SEMANTIC_ALIGNMENT_NOT_ASSUMED
ATTENTION_IS_TEMPORAL_NOT_FEATURE_IMPORTANCE
NO_ATTENTION_INTERPRETATION
NO_CASE_SELECTION_CHANGE
READY_FOR_HEATMAPS
READY_FOR_LAST_QUERY
READY_FOR_HEAD_COMPARISON
READY_FOR_ERROR_CONDITIONED
READY_FOR_SEED_STABILITY
```

---

# 154. Discrepancy taxonomy

`attention_extraction_discrepancies.json`:

```text
PHASE51_NOT_APPROVED
PHASE52_HANDOFF_NOT_READY
CASE_SELECTION_FINGERPRINT_MISMATCH
CASE_SET_CHANGED
FINAL_CHECKPOINT_MISSING
CHECKPOINT_SHA_MISMATCH
FINAL_LOCK_MISMATCH
CONFIG_MISMATCH
SCALER_CHECKSUM_MISMATCH
FEATURE_ORDER_MISMATCH
LOOKBACK_MISMATCH
HEAD_COUNT_MISMATCH
LAYER_COUNT_MISMATCH
ATTENTION_AWARE_IMPLEMENTATION_MISMATCH
INSPECTION_PATH_UNAVAILABLE
STANDARD_INSPECTION_PREDICTION_MISMATCH
PHASE47_PREDICTION_MISMATCH
ATTENTION_SHAPE_MISMATCH
ATTENTION_NAN_INF
ATTENTION_NEGATIVE_WEIGHT
ATTENTION_WEIGHT_ABOVE_ONE
ATTENTION_ROW_SUM_MISMATCH
ATTENTION_RENORMALIZED_TO_HIDE_FAILURE
CAUSAL_MASK_ADDED
PADDING_MASK_ADDED
UPPER_TRIANGLE_MISLABELED_AS_LEAKAGE
QUERY_SOURCE_AXES_SWAPPED
LAST_QUERY_AXIS_ERROR
POSITION_ORDER_REVERSED
LAG_MAPPING_ERROR
TARGET_TOKEN_IN_ATTENTION_MAP
FULL_TEST_TARGET_MISSING
FULL_TEST_TARGET_DUPLICATE
DENSE_CASE_MISSING
DENSE_CASE_ORDER_MISMATCH
DENSE_LAST_QUERY_MISMATCH
RAW_ATTENTION_DOWNSAMPLED
RAW_ATTENTION_FLOAT16
RAW_ATTENTION_THRESHOLDING
RAW_ATTENTION_SMOOTHING
HEAD_AVERAGED_BEFORE_STORAGE
LAYER_AVERAGED_BEFORE_STORAGE
SEED_AVERAGED_BEFORE_STORAGE
MODEL_PARAMETER_MUTATED
EXTRACTION_NONDETERMINISTIC
OOM_FALLBACK_NOT_DOCUMENTED
DIFFERENT_BATCH_SIZE_UNJUSTIFIED
NEW_MODEL_TRAINING
OPTIMIZER_CREATED
GRADIENT_ENABLED
DROPOUT_ACTIVE
MC_DROPOUT_USED
REVIN_DISABLED_FOR_EXTRACTION
ATTENTION_MISLABELED_AS_FEATURE_IMPORTANCE
ATTENTION_MISLABELED_AS_CAUSAL_EXPLANATION
HEAD_INDEX_ASSUMED_SEMANTICALLY_ALIGNED_ACROSS_SEEDS
ATTENTION_BASED_CASE_SELECTION
OTHER
```

---

# 155. Status model

## PASS

```text
all three final checkpoints verified
inspection predictions match frozen Phase47 predictions
all attention shapes valid
all weights finite/nonnegative
all rows normalized
dense frozen Phase51 cases complete
all-Test last-query vectors complete
full-matrix streaming summaries complete
position/lag mappings valid
raw files frozen/checksummed
Phase53–57 handoffs ready
no interpretation/model changes.
```

## PASS_WITH_WARNING

Possible:

```text
OOM batch fallback used
final pooling is MEAN, so last-query is not direct pooled-token view
minor permitted environment drift
attention extraction storage large
head semantic alignment across seeds unresolved by design.
```

## FAIL

Examples:

```text
prediction path mismatch
row-sum failure
shape mismatch
case set changed
raw files lossy
model mutation
wrong position mapping
missing Test targets.
```

---

# 156. Phase53 handoff

`phase53_attention_heatmaps_handoff.json`:

```text
source_phase52_version
final_lock_sha256
seed_list=[42,123,2026]
dense_attention_files
dense_attention_sha256s
dense_case_order_path
dense_case_order_sha256
relative_position_map
case_position_map
case_metadata
attention_axes=[case,layer,head,query,source]
heatmap_y_axis=query
heatmap_x_axis=source
no_mask=true
raw_float32=true
case_selection_independent_of_attention=true
ready_for_phase53=true
```

---

# 157. Phase54 handoff

`phase54_last_query_attention_handoff.json`:

```text
source_phase52_version
last_query_files
last_query_sha256s
target_order_path
position_lag_map
last_query_axis=[target,layer,head,source]
last_query_definition=query_idx=L-1
lag_formula=H+(L-1-p)
horizon=1
pooling
last_query_directly_corresponds_to_pooled_token
summary_files
ready_for_phase54=true
```

---

# 158. Phase55 handoff

`phase55_head_comparison_handoff.json`:

```text
source_phase52_version
last_query_files
full_matrix_summary
recent_mass_summary
top_source_summary
seed_list
layer_count
head_count
head_index_semantic_alignment_across_seeds=false
raw_head_identity_preserved=true
ready_for_phase55=true
```

---

# 159. Phase56 handoff

`phase56_error_conditioned_attention_handoff.json`:

```text
source_phase52_version
last_query_files
target_order
Phase49 residual/error refs
Phase50 regime refs
Phase51 shared hardness refs
dense worst-case files
case metadata
seed_list
no_attention_based_case_selection=true
all_test_last_query_available=true
ready_for_phase56=true
```

This allows Phase56 to define error-conditioned cohorts without rerunning full attention extraction.

---

# 160. Phase57 handoff

`phase57_seed_stability_attention_handoff.json`:

```text
source_phase52_version
seed_list=[42,123,2026]
last_query_files
dense_case_files
same_target_order=true
same_dense_case_order=true
layer_count
head_count
head_index_semantic_alignment_not_guaranteed=true
head_matching_may_be_required=true
checkpoint_refs
ready_for_phase57=true
```

---

# 161. Phase52 execution sequence

```text
1. Verify Phase51 signoff/handoff.
2. Verify frozen attention case IDs/fingerprint.
3. Verify all final checkpoint/scaler/config hashes.
4. Build exact ordered FINAL_TEST_POP-v1 attention dataset.
5. Build dense-case membership lookup.
6. Freeze extraction contract/tolerances.
7. Determine fixed extraction batch size.
8. Load seed42 checkpoint strictly.
9. Run inspection forward over full Test.
10. Compare predictions to frozen Phase47 seed42 predictions.
11. Validate attention shape/probability each batch.
12. Persist all-Test last-query vectors.
13. Persist full dense maps only for frozen case IDs.
14. Stream-compute full-matrix and last-query summaries.
15. Freeze/checksum/reload seed42 raw artifacts.
16. Repeat identical procedure for seed123.
17. Repeat identical procedure for seed2026.
18. Verify same target/case ordering across seeds.
19. Verify dense last-query slices equal all-Test last-query files.
20. Verify relative/case position-lag mappings.
21. Run small duplicate extraction reproducibility audit.
22. Verify model states unchanged.
23. Freeze raw source artifacts.
24. Write Phase53–57 handoffs.
25. Run acceptance/discrepancy audits.
26. Write report/README/signoff.
```

---

# 162. Seed-level extraction workflow

For one seed:

```text
verify checkpoint
→ strict load
→ eval
→ inference_mode
→ iterate ordered Test batches
→ inspection forward
→ prediction equivalence
→ attention integrity
→ extract last query
→ compute summaries
→ save dense case map if selected
→ discard non-case dense full map
→ finalize files
→ checksum
→ reload verify.
```

---

# 163. Recommended pseudocode

```text
p51 = load_phase51_signoff()
assert p51.overall_status in {"PASS","PASS_WITH_WARNING"}

handoff = load_phase52_attention_handoff()
assert handoff.ready_for_phase52

case_table = load_frozen_attention_case_table()
verify_case_selection_fingerprint(case_table)

test_order = load_FINAL_TEST_POP_v1_order()
position_map = build_relative_lag_map(
    L=FINAL_L,
    H=1,
    cadence_minutes=10
)

freeze_extraction_contract(
    need_weights=True,
    average_attn_weights=False,
    mask=None,
    dtype="float32",
    default_batch=8
)

dense_case_ids = set(case_table.target_id)

for seed in [42,123,2026]:

    checkpoint = resolve_final_checkpoint(seed)
    verify_checkpoint_against_lock(checkpoint)

    model = build_exact_final_attention_aware_transformer()
    strict_load(model, checkpoint)
    model.eval()

    state_hash_before = hash_model_state(model)

    batch_size = resolve_attention_batch_size(
        preferred=[8,4,2,1],
        memory_failure_only=True
    )

    loader = build_ordered_test_loader(
        target_ids=test_order,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        final_scalers=FINAL_SCALING_v1
    )

    last_query_store = initialize_last_query_store(...)
    dense_case_store = initialize_dense_case_store(...)
    summary_rows = []

    with torch.inference_mode():

        for batch in loader:

            pred_inspect, layer_attn = model(
                batch.x,
                return_attention=True
            )

            frozen_pred = load_phase47_predictions(
                seed=seed,
                target_ids=batch.target_ids
            )

            verify_prediction_equivalence(
                pred_inspect,
                frozen_pred,
                comparison_space="y_model",
                frozen_tolerance=True
            )

            assert len(layer_attn) == FINAL_N_LAYERS

            for layer_idx, A in enumerate(layer_attn):

                assert A.shape == [
                    batch_size_actual,
                    FINAL_N_HEADS,
                    FINAL_L,
                    FINAL_L
                ]

                verify_finite(A)
                verify_nonnegative(A)
                verify_row_sums_one(A)

                # all-Test last query
                A_last = A[:, :, FINAL_L-1, :]

                append_last_query(
                    last_query_store,
                    batch.target_ids,
                    layer_idx,
                    A_last.float().cpu()
                )

                # streaming summaries before full tensor discard
                append_full_matrix_summaries(
                    summary_rows,
                    A,
                    target_ids=batch.target_ids,
                    layer_idx=layer_idx
                )

                append_last_query_summaries(
                    summary_rows,
                    A_last,
                    lag_map=position_map
                )

                # full dense maps only for frozen cases
                selected_mask = target_ids_in(
                    batch.target_ids,
                    dense_case_ids
                )

                if any(selected_mask):
                    append_dense_case_attention(
                        dense_case_store,
                        selected_target_ids,
                        layer_idx,
                        A[selected_mask].float().cpu()
                    )

            discard(layer_attn)

    finalize_raw_files_in_frozen_order(
        last_query_store,
        dense_case_store
    )

    checksum_and_reload_verify_all(seed)

    state_hash_after = hash_model_state(model)
    assert state_hash_after == state_hash_before

verify_all_seeds_same_target_order()
verify_all_seeds_same_dense_case_order()

verify_dense_last_query_consistency(
    all_seeds=True
)

run_small_repeat_extraction_audit()

write_phase53_57_handoffs()

assert no_training
assert no_case_selection_change
assert no_attention_interpretation
assert raw_attention_float32
assert per_head_preserved

signoff_phase52()
```

---

# 164. Preflight acceptance checklist

```text
[ ] Phase51 PASS/PASS_WITH_WARNING.
[ ] phase52_ready=true.
[ ] Phase51 case selection fingerprint matches.
[ ] Dense attention case table deduplicated.
[ ] Three final checkpoints available.
[ ] Checkpoint SHA256s match Phase46/47.
[ ] Same final lock.
[ ] Same final config.
[ ] Same final scaler refs.
[ ] Feature fingerprint matches.
[ ] Lookback known.
[ ] Layers known.
[ ] Heads known.
[ ] Pooling known.
[ ] RevIN state known.
[ ] Attention-aware inspection API verified.
[ ] No masks expected.
[ ] Attention tolerance frozen.
[ ] Extraction contract written before raw extraction.
```

---

# 165. Checkpoint/path acceptance checklist

```text
[ ] Seed42 strict-load.
[ ] Seed123 strict-load.
[ ] Seed2026 strict-load.
[ ] checkpoint_type=FINAL_REFIT.
[ ] Correct official epoch.
[ ] Correct config hash.
[ ] Correct lock hash.
[ ] Correct state schema.
[ ] Model eval mode.
[ ] inference_mode active.
[ ] No optimizer.
[ ] No backward.
[ ] No dropout sampling.
```

---

# 166. Prediction-equivalence acceptance checklist

```text
[ ] Frozen Phase47 predictions loaded by exact target ID.
[ ] Comparison coordinate documented.
[ ] Tolerance frozen.
[ ] Seed42 inspection predictions allclose.
[ ] Seed123 inspection predictions allclose.
[ ] Seed2026 inspection predictions allclose.
[ ] No prediction row skipped.
[ ] Max differences recorded.
[ ] Tolerance not widened post hoc.
```

---

# 167. Attention tensor acceptance checklist

```text
[ ] Layer list length = N_layers.
[ ] Every tensor 4D.
[ ] Axis order B,H,Q,S verified.
[ ] H = final num_heads.
[ ] Q = L.
[ ] S = L.
[ ] All finite.
[ ] All nonnegative within numerical tolerance.
[ ] Max weights within probability bound.
[ ] Every query row sums to ~1.
[ ] No renormalization used.
[ ] No causal mask.
[ ] No padding mask.
[ ] Upper triangle allowed.
```

---

# 168. Position/lag acceptance checklist

```text
[ ] Position0 = oldest input.
[ ] Position L-1 = newest input.
[ ] Last query = query L-1.
[ ] Source L-1 = lag1 = 10 min.
[ ] Source0 = lag L.
[ ] Lag formula H+(L-1-p).
[ ] H=1.
[ ] Case input timestamps map exactly.
[ ] Input end = latest historical row.
[ ] Forecast target not included in attention positions.
```

---

# 169. Dense-case acceptance checklist

```text
[ ] Dense case IDs exactly equal frozen Phase51 handoff set.
[ ] Same dense case IDs for all seeds.
[ ] Same dense case order for all seeds.
[ ] No case added because attention is interesting.
[ ] No case removed because attention is flat.
[ ] Full tensor stored per layer/head/query/source.
[ ] dtype float32.
[ ] No head averaging.
[ ] No layer averaging.
[ ] No thresholding.
[ ] No smoothing.
[ ] Dense file checksum generated.
```

---

# 170. All-Test last-query acceptance checklist

```text
[ ] Every FINAL_TEST_POP-v1 target included.
[ ] Same target order all seeds.
[ ] No duplicate target.
[ ] No missing target.
[ ] Shape [N,Layers,Heads,L].
[ ] Last-query row sums ~1.
[ ] float32 raw source.
[ ] Files checksummed.
[ ] Files reload successfully.
[ ] Dense case last-query slices match.
```

---

# 171. Derived summary acceptance checklist

```text
[ ] Last-query entropy.
[ ] Normalized entropy.
[ ] Expected lag steps/minutes.
[ ] Top1 source/lag/weight.
[ ] Top1 tie count.
[ ] Top5 mass.
[ ] Recent 1h mass.
[ ] Recent 6h mass.
[ ] Recent 12h mass.
[ ] Recent 24h mass.
[ ] Truncated horizon flags where L is shorter.
[ ] Mean-query entropy from full matrices.
[ ] Mean self-attention weight.
[ ] Mean query-source distance.
[ ] Backward-or-same mass.
[ ] Forward-within-input mass.
```

---

# 172. Reproducibility/storage acceptance checklist

```text
[ ] Default extraction batch=8 attempted.
[ ] OOM fallback documented if used.
[ ] Same final batch size across seeds where possible.
[ ] Small batch-independence test passes.
[ ] Small duplicate-extraction test passes.
[ ] Model state unchanged pre/post extraction.
[ ] Atomic writes used.
[ ] Raw files reloaded and verified.
[ ] Raw SHA256s generated.
[ ] Partial files not treated as authoritative.
```

---

# 173. Interpretation-scope acceptance checklist

```text
[ ] Attention labeled temporal token-to-token.
[ ] Not labeled raw feature importance.
[ ] Not labeled causal explanation.
[ ] Last-query caveat reflects pooling type.
[ ] Value/residual/FFN caveats documented.
[ ] Head same-index semantic equivalence across seeds not assumed.
[ ] No attention-based case ranking.
[ ] No heatmap interpretation in Phase52.
[ ] No head winner selected.
```

---

# 174. Handoff acceptance checklist

```text
[ ] Phase53 dense attention handoff complete.
[ ] Phase54 full-Test last-query handoff complete.
[ ] Phase55 head comparison handoff complete.
[ ] Phase56 error-conditioned handoff complete.
[ ] Phase57 seed-stability handoff complete.
[ ] Every handoff references exact raw SHA256s.
[ ] Every handoff references position/lag mapping.
[ ] Every handoff preserves seed identity.
[ ] ready flags true only after raw verification PASS.
```

---

# 175. Acceptance criteria

Phase52 PASS only when:

```text
The exact three Phase46 final Transformer checkpoints are strict-loaded without mutation.

The attention-aware inspection path is the same implementation verified upstream and preserves need_weights=true, average_attn_weights=false semantics.

Inspection-path predictions agree with the frozen Phase47 predictions under the predeclared numerical tolerance for the complete Test population.

Every extracted attention tensor follows [B,H,Q,S] and [B,H,L,L] self-attention semantics.

All attention values are finite, nonnegative within numerical tolerance and every query row sums to approximately 1 in eval mode.

No causal or padding mask is introduced.

The temporal position convention is verified: position 0 is oldest and position L-1 is newest.

The last-query vector is taken from A[:,:,L-1,:], not the source-last column.

The position-to-lag map uses H+(L-1-p), with H=1.

Full dense per-layer/per-head [L,L] attention is stored in float32 for the exact deduplicated Phase51 frozen case set, using the same cases/order for all three seeds.

Per-layer/per-head last-query attention is stored in float32 for every FINAL_TEST_POP-v1 target for all three seeds.

All-Test full matrices are streamed to produce compact diagnostics and are not unnecessarily retained outside the frozen case set.

Dense-case last-query slices exactly match the corresponding full-Test last-query artifacts.

Raw attention artifacts are not head-averaged, layer-averaged, seed-averaged, smoothed, thresholded, quantized or renormalized.

Raw files are atomically written, checksummed and reload-verified.

Model states are unchanged before and after extraction.

Attention is explicitly documented as temporal allocation rather than raw-feature importance or causal attribution.

Same-index head semantics are not assumed to be functionally aligned across independently trained seeds.

All Phase53–57 handoffs reference the frozen raw attention artifacts and pass their readiness gates.
```

---

# 176. Failure conditions

Phase52 FAIL if:

```text
wrong checkpoint is loaded

inspection path changes prediction materially

attention shape is averaged [B,L,L] instead of per-head [B,H,L,L]

query/source axes are swapped

last-query vector is extracted as A[:,:,:,L-1]

position order is reversed

forecast target appears as an attention token

attention rows are not normalized in eval mode

NaN/Inf exists

material negative weights exist

attention is renormalized to hide integrity failure

case set changes after seeing attention

dense case IDs differ across seeds

full-Test last-query coverage is incomplete

raw attention is stored only after head averaging

raw attention is downcast to float16

attention is thresholded/smoothed

causal mask is added

RevIN is disabled for interpretability

dropout remains active

gradient/training occurs

model parameters mutate

raw file checksum/reload fails

attention is reported as raw-feature importance

same head index is treated as guaranteed same semantic head across seeds

attention interpretation begins before extraction signoff.
```

---

# 177. Common mistakes

## 177.1 Dùng `average_attn_weights=True`

Sai vì sẽ mất head dimension.

Project cần:

```text
[B,H,L,L].
```

## 177.2 Lấy `A[:,:,:, -1]` và gọi last-query

Sai. Đây là source/key cuối cùng qua mọi query.

Last-query đúng:

```text
A[:,:,-1,:].
```

## 177.3 Thấy upper triangle khác 0 rồi nghĩ leakage

Sai. Encoder không dùng causal mask và toàn bộ input window nằm trước forecast target.

## 177.4 Average heads rồi mới save raw

Sai. Mất thông tin head-specific.

## 177.5 Chỉ extract attention cho seed có Test RMSE thấp nhất

Sai. Phải đủ cả ba seeds.

## 177.6 Case seed42 worst thì chỉ extract seed42

Sai. Same dense case set phải extract cả ba seeds để Phase57 so sánh.

## 177.7 Save heatmap PNG rồi bỏ raw tensor

Sai. PNG không phải scientific source.

## 177.8 Store float16 để giảm dung lượng

Không dùng cho raw canonical attention.

## 177.9 Gọi temporal attention là feature importance

Sai axis semantics.

## 177.10 Tắt RevIN khi extract để heatmap dễ hiểu

Sai model semantics.

## 177.11 Thấy row sum lệch rồi normalize lại

Sai integrity protocol.

## 177.12 So Head1 seed42 với Head1 seed123 rồi coi là cùng chức năng

Chưa chắc do head permutation symmetry.

---

# 178. Human-readable report structure

`attention_extraction_report.md`:

```text
1. Objective
2. Upstream frozen case/test sources
3. Final checkpoint verification
4. Attention-aware implementation contract
5. Attention tensor axis semantics
6. Temporal position and lag mapping
7. Why attention is temporal, not feature importance
8. Extraction storage design
9. Dense-case extraction
10. Full-Test last-query extraction
11. Streaming full-matrix summaries
12. Prediction-equivalence verification
13. Attention probability/integrity verification
14. Reproducibility and model-mutation audit
15. Raw storage/checksum verification
16. Pooling-specific interpretation caveat
17. Multi-layer/multi-head/value-path caveats
18. Head identity caveat across seeds
19. Phase53–57 handoffs
20. Limitations
21. Definition of Done
```

---

# 179. README requirements

`README_ATTENTION_EXTRACTION.md` must explain:

```text
what raw attention tensor means
axis order B,H,Q,S
why position0 is oldest
why last-query is A[:,:,L-1,:]
how source position maps to lag
why no causal mask is used
why upper triangle is valid
why full dense attention is only stored for frozen cases
why all-Test last-query is stored
why float32/per-head raw source is required
why raw attention is not feature importance
why attention is not causal explanation
why head IDs may permute across seeds
how Phase53–57 consume artifacts.
```

---

# 180. Summary artifact

`attention_extraction_summary.json`:

```text
version
source_phase51_version
final_lock_sha256
test_population_sha256
case_selection_contract_sha256
seed_list
lookback
layers
heads
pooling
RevIN_state
extraction_batch_size
dense_case_count
test_target_count
dense_attention_shapes
last_query_attention_shapes
prediction_equivalence_status
probability_integrity_status
position_mapping_status
dense_last_query_consistency_status
model_mutation_status
reproducibility_status
raw_attention_checksums
head_semantic_alignment_across_seeds_guaranteed=false
attention_feature_importance_claim=false
attention_causal_claim=false
phase53_ready
phase54_ready
phase55_ready
phase56_ready
phase57_ready
overall_status
```

No runtime counts/shapes should be fabricated before extraction.

---

# 181. Phase52 sign-off

`phase_52_signoff.json` minimum:

```text
phase=52
phase_name=Attention extraction
version=ATTENTION_EXTRACTION-v1
source_phase51_version
final_lock_sha256
test_population_sha256
case_selection_contract_sha256
seed_list=[42,123,2026]
lookback_steps
num_layers
num_heads
pooling
RevIN_state
attention_axis_contract=B,H,Q,S
dense_case_count
full_test_target_count
seed42_dense_sha256
seed123_dense_sha256
seed2026_dense_sha256
seed42_last_query_sha256
seed123_last_query_sha256
seed2026_last_query_sha256
prediction_equivalence_verified
attention_shape_verified
attention_probability_verified
position_lag_mapping_verified
dense_last_query_consistency_verified
model_mutation=false
new_training=false
case_selection_changed=false
head_averaging_before_storage=false
layer_averaging_before_storage=false
seed_averaging_before_storage=false
attention_interpretation_performed=false
phase53_ready
phase54_ready
phase55_ready
phase56_ready
phase57_ready
warnings
overall_status
created_at
```

---

# 182. Definition of Done

\[
\boxed{
Three\ Frozen\ Final\ Checkpoints
+
Verified\ Inspection\ Path
+
Per\text{-}Head\ [L,L]\ Case\ Attention
+
All\text{-}Test\ Last\text{-}Query\ Attention
+
Exact\ Lag\ Mapping
+
Probability\ Integrity
+
Raw\ Float32\ Storage
+
Checksums
+
No\ Interpretation
+
Phase53\text{–}57\ Handoffs
}
\]

---

# 183. Final status contract

```text
PHASE 52 extracts attention only.

Models:
FINAL_TR_SEED42
FINAL_TR_SEED123
FINAL_TR_SEED2026.

Mode:
eval
inference_mode
no gradients
no masks.

Attention API:
need_weights=true
average_attn_weights=false.

Raw semantics:
A[b,h,q,s]
self-attention [B,H,L,L].

Temporal positions:
0 oldest
L-1 newest.

Last query:
A[:,:,L-1,:].

Source lag:
H+(L-1-p)
with H=1.

Storage Tier A:
full [L,L] attention
for exact frozen Phase51 case set
all layers
all heads
all three seeds.

Storage Tier B:
last-query [L]
for every FINAL_TEST_POP-v1 target
all layers
all heads
all three seeds.

Streaming:
derive compact full-matrix summaries
discard non-case dense matrices.

Integrity:
inspection prediction == frozen Phase47 prediction
finite
nonnegative
row sums ≈1
exact shapes
exact case/target coverage
model unchanged.

Raw source:
float32
no head averaging
no layer averaging
no seed averaging
no thresholding
no smoothing
no renormalization.

Interpretation:
temporal token attention
not feature importance
not causal explanation.

Head identity:
same numeric head index across seeds
does not guarantee same learned semantic role.

After ATTENTION_EXTRACTION-v1 PASS:
proceed to
PHASE 53 — Attention Heatmaps.
```

---

# 184. Final check

Correct:

```text
verify Phase51 case set
→ verify final checkpoints
→ ordered Test inputs
→ inspection forward
→ verify prediction equivalence
→ verify [B,H,L,L]
→ save full dense maps for frozen cases
→ save last-query vectors for all Test targets
→ derive compact summaries
→ checksum/reload
→ Phase53–57 handoffs
```

Incorrect:

```text
average heads
→ save one [L,L] map
```

Incorrect:

```text
choose cases after seeing pretty attention
```

Incorrect:

```text
A[:,:,:,L-1]
→ call it last-query
```

Incorrect:

```text
attention high at one timestamp
→ conclude feature/cause importance
```

Chỉ sau khi:

```text
ATTENTION_EXTRACTION-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase53_ready = true
```

mới chuyển sang **PHASE 53 — Attention Heatmaps**.
