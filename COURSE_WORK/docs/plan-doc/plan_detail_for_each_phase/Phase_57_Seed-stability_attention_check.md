# PHASE 57 — SEED-STABILITY ATTENTION CHECK

## Kế hoạch kiểm tra độ ổn định của temporal attention qua ba Final Transformer seeds bằng phân tích permutation-invariant và permutation-aware head matching

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Model:** Attention-Aware Transformer Encoder for regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Official final seeds:** `42`, `123`, `2026`  
**Primary raw attention source:** `ATTENTION_EXTRACTION-v1`  
**Primary last-query source:** `LAST_QUERY_ATTENTION-v1`  
**Within-seed head behavior source:** `HEAD_COMPARISON-v1`  
**Error-conditioned source:** `ERROR_CONDITIONED_ATTENTION-v1`  
**Primary stability principle:** same-index heads are **not** assumed semantically equivalent across independently trained seeds  
**Phase ID:** `PHASE_57_SEED_STABILITY_ATTENTION_CHECK`  
**Output version:** `SEED_STABILITY_ATTENTION-v1`  
**Phase trước:** `Phase_56_Error-conditioned_attention.md`  
**Phase sau:** `Phase_58_Final_tables.md`

---

# 1. Vai trò của Phase 57

Phase57 là bước cuối của nhánh attention analysis.

Phase52–56 đã lần lượt trả lời:

```text
Phase52:
Attention được extract đúng chưa?

Phase53:
Attention maps nhìn như thế nào?

Phase54:
Newest-query attention tập trung ở những lag nào?

Phase55:
Các heads trong cùng seed/layer khác nhau thế nào?

Phase56:
Attention behavior có liên hệ thế nào với forecast error?
```

Phase57 trả lời:

> Những attention patterns và error-conditioned attention findings đó có ổn định qua ba stochastic final seeds hay không, sau khi đã xử lý đúng vấn đề head permutation?

Mục tiêu:

```text
1. Verify toàn bộ seed-level attention sources/checksums.
2. Tách rõ permutation-invariant stability và head-level permutation-aware stability.
3. Kiểm tra layer head-mean attention stability trên toàn Test mà không cần head matching.
4. Xây dựng canonical one-to-one head matching trong từng layer bằng full-Test mean temporal profiles.
5. Không dùng same-index head như một giả định semantic.
6. Kiểm tra độ rõ/ambiguous của head matching.
7. Kiểm tra cycle consistency giữa ba pairwise seed mappings.
8. So sánh canonical JSD matching với Wasserstein matching như sensitivity audit.
9. Đánh giá matched-head mean-profile stability.
10. Đánh giá matched-head per-target last-query stability trên toàn Test.
11. Đánh giá top1-lag distribution và head behavior metric stability sau matching.
12. Đánh giá full dense attention stability trên frozen Phase51 case set.
13. Đánh giá error-conditioned attention finding stability từ Phase56.
14. Đánh giá layer-level error-conditioned findings theo cách permutation-invariant.
15. Liên hệ attention disagreement với prediction seed spread như secondary diagnostic.
16. Tạo consensus/matched-head profiles chỉ sau matching, không trước.
17. Không chọn best seed.
18. Không chọn best head.
19. Không retune/retrain/prune model.
20. Freeze toàn bộ attention robustness evidence cho Phase58 Final Tables.
```

Nguyên tắc trung tâm:

\[
\boxed{
Same\ Targets
+
Three\ Official\ Seeds
+
Permutation\text{-}Invariant\ Layer\ Checks
+
Permutation\text{-}Aware\ Head\ Matching
+
Per\text{-}Target\ Stability
+
Error\text{-}Finding\ Stability
+
No\ Best\ Seed
+
No\ Retuning
}
\]

---

# 2. Vấn đề phương pháp luận trung tâm: head permutation

Trong Multi-Head Attention, các heads trong cùng layer không có semantic label cố định như:

```text
Head 1 = short-term head
Head 2 = daily head
...
```

Hai model độc lập có thể học các vai trò tương tự nhưng đặt chúng vào các head indices khác nhau.

Vì vậy:

```text
Layer1 Head1 seed42
```

không được mặc định tương đương với:

```text
Layer1 Head1 seed123.
```

---

# 3. Same-index comparison chỉ là architectural-index comparison

Allowed:

```text
same-index diagnostic
```

nhưng phải ghi:

```text
semantic_alignment_not_guaranteed=true.
```

Không được dùng same-index comparison làm bằng chứng chính cho seed stability.

---

# 4. Phase57 sử dụng hai nhánh stability

Canonical:

```text
S57-A — Permutation-invariant layer stability
S57-B — Permutation-aware matched-head stability.
```

Hai nhánh bổ sung cho nhau.

---

# 5. S57-A — Permutation-invariant layer stability

Average attention across all heads within a layer:

\[
a^{layermean}_{s,t,l}(k)
=
\frac{1}{H}
\sum_{h=1}^{H}
a_{s,t,l,h}(k).
\]

Head order/permutation không ảnh hưởng giá trị này.

Đây là **primary robust cross-seed stability view**.

---

# 6. S57-B — Permutation-aware matched-head stability

Heads trong cùng layer được one-to-one match giữa seeds dựa trên:

```text
full-Test mean temporal last-query profile.
```

Sau matching mới so:

```text
matched head profiles
per-target attention
head metrics
error-conditioned findings.
```

---

# 7. Phase57 không dùng model performance để match heads

Forbidden matching cost:

```text
Test RMSE
absolute error
prediction spread
error-attention rho
worst-case behavior.
```

Canonical head matching dựa **chỉ** trên temporal attention profiles.

Điều này tránh circularity với Phase56 error-conditioned conclusions.

---

# 8. Upstream hard gate

Required Phase56:

```text
phase_56_signoff.json
phase57_seed_stability_attention_handoff.json
error_attention_association_long.csv
error_attention_high_low_metric_comparison.csv
error_attention_high_low_profile_comparison.csv
error_attention_layer_head_mean_association.csv
error_attention_shared_cohort_layer_summary.csv
error_attention_layer_cross_seed_summary.csv
error_conditioning_assignment.csv
```

Required Phase55:

```text
phase_55_signoff.json
phase57_seed_stability_head_context_handoff.json
head_behavior_summary.csv
head_pair_comparison_long.csv
layer_head_diversity_summary.csv
```

Required Phase54:

```text
phase_54_signoff.json
phase57_seed_stability_attention_context_handoff.json
last_query_metrics_long.csv
last_query_profile_by_lag.csv
last_query_layer_head_mean_profile.csv
last_query_top1_lag_frequency.csv
```

Required Phase52 raw:

```text
phase_52_signoff.json
last_query_attention_seed42.npz
last_query_attention_seed123.npz
last_query_attention_seed2026.npz
dense_case_attention_seed42.npz
dense_case_attention_seed123.npz
dense_case_attention_seed2026.npz
attention_test_target_order.csv
attention_dense_case_order.csv
attention_relative_position_map.csv
raw_attention_checksums.json
```

Secondary source:

```text
Phase48 prediction_seed_spread.csv
```

for attention-disagreement vs prediction-disagreement analysis.

Hard:

```text
phase57_ready=true.
```

---

# 9. Official seeds are immutable

Exactly:

```text
42
123
2026.
```

No seed replacement.

No extra seed added because matching is weak.

No seed removed because attention is unusual.

---

# 10. Seed42 is allowed only as deterministic alignment anchor

For three-way matched-head tables, define:

```text
ALIGNMENT_REFERENCE_SEED = 42.
```

Reason:

```text
42 is the first seed in the predeclared FINAL_SEEDS-v1 list.
```

It is **not** selected because it has:

```text
lower Test RMSE
cleaner attention
better matching.
```

---

# 11. All three seed pairs are still analyzed

Required pair set:

```text
42 ↔ 123
42 ↔ 2026
123 ↔ 2026.
```

The seed42 anchor is only for constructing three-way canonical matched groups.

---

# 12. Same target order is mandatory

All raw last-query files must use the exact:

```text
FINAL_TEST_POP-v1
```

target order.

Hard:

```text
same N
same target IDs
same timestamps
same lag support.
```

No target intersection fallback.

---

# 13. Same dense case order is mandatory

Dense attention files must use the exact same:

```text
Phase51 frozen attention case set
```

across all three seeds.

No case dropping.

---

# 14. Same architecture is mandatory

All three final seeds must have:

```text
same N_layers
same N_heads
same L
same feature contract
same pooling
same final model lock.
```

If not:

```text
Phase57 FAIL
```

because they are not stochastic realizations of the same final configuration.

---

# 15. No new attention extraction

Hard:

```text
new model forward
new Test inference
new attention extraction
```

are forbidden.

Phase57 reads frozen Phase52 raw artifacts.

---

# 16. No new training

No:

```text
fine-tuning
head alignment training
projection fitting
attention distillation
head permutation learning.
```

Head matching is a deterministic post-hoc assignment problem only.

---

# 17. Core Phase57 questions

```text
Q57.1  Are layer head-mean temporal profiles similar across seeds?
Q57.2  Are per-target layer head-mean attentions stable across seeds?
Q57.3  Can heads in each layer be matched one-to-one using temporal profiles?
Q57.4  Is the canonical head matching unambiguous or weak/ambiguous?
Q57.5  Is pairwise head matching cycle-consistent across the three seeds?
Q57.6  Does canonical JSD matching agree with a Wasserstein-based sensitivity matching?
Q57.7  After matching, are mean head profiles stable across seeds?
Q57.8  After matching, are per-target head attention vectors stable?
Q57.9  Are top1 lag preferences and head behavior metrics stable after matching?
Q57.10 Are dense full attention maps stable on the frozen worst-case set?
Q57.11 Are Phase56 error-attention associations directionally consistent across seeds?
Q57.12 Are high-vs-low attention profile shifts stable across seeds?
Q57.13 Is layer-level error-conditioned attention more stable than individual-head attention?
Q57.14 Does attention disagreement co-occur with prediction seed disagreement?
Q57.15 What attention robustness evidence should be carried into Phase58?
```

---

# 18. Stability is continuous, not a binary arbitrary threshold

Do not invent:

```text
JSD < 0.05 = stable
JSD > 0.2 = unstable
```

after seeing results.

Phase57 reports:

```text
exact distances/similarities
pairwise distributions
matching ambiguity
sign agreement
range/SD across seeds.
```

---

# 19. PASS does not require attention to be highly stable

A scientifically valid result can be:

```text
attention varies substantially across seeds.
```

Methodological PASS means:

```text
analysis protocol executed correctly.
```

---

# 20. Part A — Layer head-mean profile stability

Source:

```text
Phase54 layer head-mean temporal profiles
```

or recompute from Phase52 raw last-query vectors for verification.

For every:

```text
seed
layer
lag
```

obtain:

\[
P^{layer}_{s,l}(k).
\]

---

# 21. Layer head-mean profile integrity

Each:

```text
P_layer
```

must satisfy:

```text
nonnegative
sum ≈ 1.
```

No renormalization.

---

# 22. Pairwise layer-profile stability metrics

For each:

```text
layer
seed_pair
```

compute:

```text
Jensen–Shannon divergence
L1 distance
L2 distance
cosine similarity
Pearson correlation
Spearman correlation
Wasserstein-1 distance in minutes.
```

---

# 23. Wasserstein support

Use exact:

```text
lag_minutes
```

from Phase52.

No source-position units mislabeled as minutes.

---

# 24. Layer-profile three-seed summary

For each layer:

```text
mean pairwise JSD
max pairwise JSD
mean pairwise Wasserstein
max pairwise Wasserstein
mean pairwise cosine
min pairwise cosine.
```

Descriptive only.

---

# 25. No layer stability score composite

Do not combine:

```text
JSD + Wasserstein + cosine
```

into one weighted score.

---

# 26. Per-target layer head-mean stability

Mean profiles can hide target-specific instability.

For every:

```text
target
layer
seed_pair
```

compare layer head-mean last-query vectors.

Required:

```text
JSD
cosine
L1
Wasserstein minutes.
```

---

# 27. Per-target layer stability summary

For each:

```text
layer
seed_pair
metric
```

report across Test targets:

```text
N
mean
sample SD
median
p05
p25
p75
p95
min
max.
```

No iid confidence interval.

---

# 28. Three-seed per-target layer disagreement

For each:

```text
target
layer
```

aggregate the three pairwise distances:

```text
mean pairwise JSD
max pairwise JSD
mean pairwise Wasserstein
max pairwise Wasserstein.
```

This becomes a seed-invariant target-level attention-disagreement measure.

---

# 29. Layer head-mean metric stability

From Phase56/Phase54 per-target layer head-mean metrics, compare same metric values across seed pairs for:

```text
normalized entropy
expected lag minutes
recent1h mass
recent6h mass
top5 mass
Lag80 minutes.
```

Required:

```text
Spearman across targets
median absolute difference
mean absolute difference.
```

---

# 30. Part B — Canonical head matching

Head matching occurs separately for each:

```text
encoder layer.
```

No matching across layers.

---

# 31. Matching representation

For each seed/layer/head:

\[
p_{s,l,h}(k)
=
Mean_t[a_{s,t,l,h}(k)].
\]

This is the **full-Test mean last-query temporal profile**.

---

# 32. Why full-Test mean profile is canonical representation

It is:

```text
independent of one particular worst case
independent of error cohort
defined for every head
a valid probability distribution
directly aligned to project temporal-attention question.
```

---

# 33. Canonical matching cost

For heads:

```text
h_a from seed A
h_b from seed B
```

define:

\[
C(h_a,h_b)
=
JSD(p_{A,h_a},p_{B,h_b}).
\]

Canonical matching minimizes:

\[
\sum_{h} C(h,\pi(h))
\]

over one-to-one permutations `π`.

---

# 34. Why JSD is the canonical cost

It is:

```text
symmetric
bounded
designed for probability distributions
already used in Phase55
does not depend on forecast error.
```

---

# 35. No weighted composite matching cost

Forbidden:

```text
0.5*JSD + 0.3*Wasserstein + 0.2*entropy_diff.
```

Canonical objective is:

```text
JSD only.
```

---

# 36. Final head-count search space makes exhaustive matching practical

The locked project head search is:

```text
H2
H4.
```

Therefore final:

```text
H ∈ {2,4}
```

under current protocol.

Number of permutations:

```text
2! = 2
4! = 24.
```

Phase57 can exhaustively enumerate all one-to-one head permutations.

This is preferred over relying on opaque assignment tie behavior.

---

# 37. Protocol amendment fallback

If a formal upstream Protocol Amendment produced:

```text
H > 4
```

Phase57 may use:

```text
Hungarian / linear_sum_assignment
```

with the same JSD objective and deterministic tie audit.

Under current locked search space, exhaustive enumeration is canonical.

---

# 38. Canonical matching tie-break

For every possible permutation compute:

```text
total_JSD
total_Wasserstein_minutes.
```

Select:

```text
1. minimum total JSD
2. if total JSD ties within MATCH_TIE_TOL:
      minimum total Wasserstein
3. if still tied:
      lexicographically smallest permutation.
```

---

# 39. Matching tie tolerance

Freeze before matching:

```text
MATCH_TIE_TOL = 1e-12
```

on total JSD.

Do not change it after seeing mappings.

---

# 40. Wasserstein is only a tie-break here

It is not part of weighted canonical matching objective.

The canonical scientific matching remains:

```text
minimum total JSD.
```

---

# 41. Pairwise seed matching

Required mappings per layer:

```text
M_42_123
M_42_2026
M_123_2026.
```

All computed independently.

---

# 42. Canonical three-seed matched groups

Use seed42 as deterministic anchor.

For seed42 head `h`:

```text
canonical_group h
=
(
seed42 head h,
matched seed123 head via M_42_123,
matched seed2026 head via M_42_2026
).
```

Name:

```text
CANONICAL_MATCH_GROUP_1
...
CANONICAL_MATCH_GROUP_H.
```

Do not call it:

```text
semantic Head1
```

without qualification.

---

# 43. Anchor choice is non-performance-based

Metadata must contain:

```text
alignment_anchor_seed=42
alignment_anchor_reason=FIRST_PREDECLARED_FINAL_SEED
performance_based_anchor=false.
```

---

# 44. Head matching cost matrix

For every:

```text
layer
seed pair
```

store all:

```text
head_a
head_b
JSD
Wasserstein
cosine
L1.
```

Canonical assignment uses JSD only.

---

# 45. Matching ambiguity audit

A mapping can be mathematically valid but weakly identifiable.

Required diagnostics:

```text
best_total_JSD
second_best_total_JSD
assignment_gap_JSD
best_total_Wasserstein
number_of_assignments_within_tie_tolerance
ambiguous_match_warning.
```

---

# 46. Assignment gap

\[
Gap
=
Cost_{second-best}
-
Cost_{best}.
\]

Report exact value.

No threshold-based “good/bad matching” category except:

```text
exact/tolerance tie
→ AMBIGUOUS_MATCH_WARNING.
```

---

# 47. Pairwise matched-edge margin

For each canonical matched edge also record:

```text
matched JSD
row next-best JSD
row margin
column next-best JSD
column margin.
```

Descriptive only.

---

# 48. Cycle consistency

Direct mapping:

```text
123 → 2026
```

can be compared against anchor-induced mapping:

```text
123 → 42 → 2026.
```

For every layer/head:

```text
cycle_consistent = true/false.
```

---

# 49. Cycle consistency rate

For each layer:

\[
CycleConsistencyRate
=
\frac{
\# consistent\ heads
}{
H
}.
\]

This is transparent and threshold-free.

---

# 50. Why cycle consistency matters

If pairwise optimal matches conflict:

```text
head identities may be weakly identifiable
or
multiple heads may have similar temporal profiles.
```

This is a legitimate stability finding.

---

# 51. Wasserstein-only matching sensitivity

As a sensitivity audit, independently find the one-to-one mapping minimizing:

```text
total Wasserstein distance in minutes.
```

Call:

```text
WASSERSTEIN_MATCH_SENSITIVITY.
```

---

# 52. Wasserstein matching is not allowed to replace canonical JSD matching post hoc

Report agreement/disagreement.

Do not choose whichever mapping yields more stable downstream results.

---

# 53. Matching-method agreement

For each:

```text
layer
seed pair
```

compute:

```text
matched_pair_agreement_count
matched_pair_agreement_fraction.
```

between:

```text
canonical JSD match
Wasserstein sensitivity match.
```

---

# 54. No cosine-based third matching search in core Phase57

Avoid method fishing.

Canonical:

```text
JSD.
```

Sensitivity:

```text
Wasserstein.
```

Enough.

---

# 55. Part C — Matched-head mean-profile stability

After canonical matching, for each:

```text
layer
canonical matched group
seed pair
```

compare mean profiles with:

```text
JSD
Wasserstein minutes
cosine
Pearson
Spearman
L1
L2.
```

---

# 56. Three-seed matched-group summary

For each:

```text
layer
canonical group
```

aggregate three seed-pair values:

```text
mean pairwise JSD
max pairwise JSD
mean Wasserstein
max Wasserstein
mean cosine
min cosine.
```

No composite score.

---

# 57. Matched-head consensus profile

Only **after matching**, create:

\[
P^{consensus}_{l,g}(k)
=
\frac{
P_{42,l,g}(k)
+
P_{123,l,g}(k)
+
P_{2026,l,g}(k)
}{3}.
\]

This is a derived descriptive profile.

---

# 58. Consensus profile integrity

Because mean of probability profiles:

```text
sum ≈ 1.
```

Hard audit.

---

# 59. Per-lag cross-seed variability

For each:

```text
layer
canonical matched group
lag
```

compute:

```text
mean weight across seeds
sample SD across seeds
min
max.
```

Only three seeds, so use descriptive wording.

---

# 60. No confidence interval across 3 seeds

No.

---

# 61. Part D — Matched-head per-target stability

Mean-profile stability alone may hide target-specific variation.

After global head matching is frozen, apply the same mapping to every Test target.

---

# 62. Head matching is not target-specific

Forbidden:

```text
re-match heads separately for every target.
```

Canonical mapping is learned once from full-Test mean profiles.

This preserves a stable head correspondence.

---

# 63. Why target-specific matching is forbidden

It would artificially inflate apparent stability by allowing head identities to change from target to target.

---

# 64. Per-target matched-head metrics

For every:

```text
target
layer
canonical matched group
seed pair
```

compare last-query vectors with:

```text
JSD
Wasserstein minutes
cosine
L1.
```

---

# 65. Per-target matched-head summary

For each:

```text
layer
canonical group
seed pair
```

report across Test targets:

```text
N
mean
sample SD
median
p05
p25
p75
p95
min
max.
```

---

# 66. Three-seed per-target matched-head disagreement

For each:

```text
target
layer
canonical group
```

compute:

```text
mean pairwise JSD
max pairwise JSD
mean pairwise Wasserstein
max pairwise Wasserstein.
```

This supports case-level stability analysis.

---

# 67. Matched-head metric stability

From Phase54 per-target head metrics, after matching compare same canonical group across seed pairs for:

```text
normalized entropy
expected lag minutes
recent1h mass
recent6h mass
top5 mass
Lag80 minutes.
```

Required:

```text
Spearman across targets
mean absolute difference
median absolute difference.
```

---

# 68. Top1 lag stability after matching

Use Phase54 top1 lag frequencies.

For each matched head seed pair:

```text
TVD
JSD
modal lag comparison.
```

---

# 69. Top1 modal lag agreement

Report:

```text
same_modal_lag=true/false
```

but do not overinterpret exact single-mode agreement.

---

# 70. Part E — Dense full-attention stability on frozen cases

Use Phase52:

```text
dense_case_attention_seed*.npz
```

for exact Phase51 frozen case IDs.

Shape:

```text
[case,layer,head,query,source].
```

---

# 71. Apply canonical global head mapping

Do not re-match dense-case heads case-by-case.

Use the exact full-Test canonical head mapping.

---

# 72. Full-map rowwise JSD

For one:

```text
case
layer
matched head
seed pair
query q
```

each source row is a probability distribution.

Compute:

\[
JSD_q
=
JSD(A^{seedA}_{q,:},A^{seedB}_{q,:}).
\]

Then case/head summary:

```text
mean_query_JSD
median_query_JSD
max_query_JSD.
```

---

# 73. Full-map rowwise cosine

For each query:

```text
cosine(A_q_seedA, A_q_seedB)
```

then:

```text
mean_query_cosine
min_query_cosine.
```

---

# 74. Full-map normalized Frobenius distance

For matrices `A,B`:

\[
D_F
=
\frac{
\|A-B\|_F
}{
\sqrt{L}
}.
\]

Because each of `L` rows is a probability vector, dividing by:

```text
sqrt(L)
```

provides a row-count-scaled matrix distance.

Label:

```text
NORMALIZED_FROBENIUS_DESCRIPTIVE.
```

---

# 75. No image-pixel similarity

Do not compare:

```text
PNG heatmap pixels
SSIM
image color distance.
```

Use raw numerical attention matrices only.

---

# 76. Dense-case stability is selection-conditioned

The Phase51 case set is enriched for:

```text
worst/shared hard cases.
```

Therefore dense full-map stability findings are:

```text
case-study stability
```

not unbiased full-Test dense-map stability.

---

# 77. Last-query full-Test analysis remains primary

Dense-case full matrices are supplementary.

---

# 78. Dense case ordering

Preserve exact:

```text
Phase51 / Phase52 case order.
```

No attention-stability-driven case selection.

---

# 79. Part F — Error-conditioned finding stability

Phase56 produced:

```text
AE Spearman associations
signed residual associations
high-vs-low metric differences
Cliff's delta
high-vs-low profile distances
layer head-mean results
shared cohort results.
```

Phase57 checks whether those findings persist across seeds.

---

# 80. Layer head-mean error-conditioned stability is primary

Because head averaging removes permutation.

For each:

```text
layer
conditioning variable
core attention metric
```

collect seed-specific:

```text
Spearman rho.
```

Report:

```text
seed42 rho
seed123 rho
seed2026 rho
mean
sample SD
min
max
sign agreement count.
```

---

# 81. Sign agreement count

For three seed values:

```text
positive
negative
zero/undefined.
```

Required:

```text
positive_count
negative_count
undefined_count.
```

No p-value voting.

---

# 82. Unanimous sign flag

Allowed:

```text
all_defined_same_sign=true/false.
```

This is transparent.

Do not call it statistical replication.

---

# 83. High-vs-low layer result stability

For each layer/core metric collect seed-specific:

```text
delta_median_high_minus_low
Cliff's delta.
```

Report:

```text
mean
sample SD
range
sign agreement.
```

---

# 84. High-vs-low layer profile-shift stability

For each seed/layer:

```text
JSD HIGH vs LOW
Wasserstein HIGH vs LOW
```

already computed in Phase56.

Compare magnitudes descriptively across seeds.

No cross-seed averaging of raw cohort profiles required.

---

# 85. Shared-cohort advantage

Phase56 shared cohorts use identical target IDs across seeds.

Thus layer head-mean:

```text
SHARED_LOW vs SHARED_HIGH
```

is especially useful for seed stability.

Report seed-specific effect metrics side by side.

---

# 86. Matched-head error-conditioned stability

After canonical matching, map Phase56 per-head findings into canonical matched groups.

For each:

```text
layer
canonical group
core metric
conditioning variable
```

compare seed-specific:

```text
Spearman rho
high-low Cliff's delta
high-low median difference.
```

---

# 87. Matched-head error associations remain secondary

Because head matching itself can be ambiguous.

Primary robustness statement should prioritize:

```text
layer head-mean
```

then matched-head results.

---

# 88. Ambiguous matching caveat propagates

If a layer/seed-pair mapping is flagged:

```text
AMBIGUOUS_MATCH_WARNING
```

then matched-head error-conditioned comparisons involving that mapping receive:

```text
MATCH_AMBIGUITY_WARNING.
```

Do not suppress them.

---

# 89. No head-level effect averaging before matching

Hard:

```text
Head1 seed42 + Head1 seed123 + Head1 seed2026
```

must not be averaged without canonical mapping.

---

# 90. Part G — Attention disagreement vs prediction seed spread

Secondary diagnostic source:

```text
Phase48 prediction_seed_spread.csv.
```

For every target, Phase48 contains:

```text
prediction range
prediction SD
```

across final seeds.

---

# 91. Permutation-invariant attention disagreement variable

For every:

```text
target
layer
```

use:

```text
mean pairwise JSD of layer head-mean attention
mean pairwise Wasserstein of layer head-mean attention.
```

These are seed-invariant and require no head matching.

---

# 92. Prediction-disagreement association

Compute Spearman:

\[
\rho(
PredictionSeedRange_t,
AttentionDisagreement_{t,l}
)
\]

and:

\[
\rho(
PredictionSeedSD_t,
AttentionDisagreement_{t,l}
).
\]

No p-values as core.

---

# 93. Matched-head secondary variant

For each canonical group/target, compute mean pairwise matched-head JSD and compare to:

```text
prediction seed range.
```

Secondary only.

---

# 94. Interpretation caveat

Safe:

> Targets with larger prediction spread also tended to show larger layer-level attention disagreement.

Unsafe:

> Attention instability caused prediction instability.

---

# 95. No thresholding prediction spread

Continuous association only.

Do not invent:

```text
high prediction instability group.
```

---

# 96. Part H — Stability figures

Figures must preserve:

```text
seed pair order
layer order
canonical matched-group order.
```

No sorting by best stability.

---

# 97. Seed-pair order

Canonical:

```text
42–123
42–2026
123–2026.
```

---

# 98. Layer order

Canonical:

```text
Layer1
Layer2
...
LayerN.
```

---

# 99. Canonical matched-group order

Anchored to seed42 architectural head order:

```text
Group1 = seed42 Head1
Group2 = seed42 Head2
...
```

This is not stability ranking.

---

# 100. Figure SEEDATTN_57_01 — Layer head-mean profile stability

For each layer:

```text
three seed head-mean temporal profiles
```

over lag.

Same y-axis.

No smoothing.

---

# 101. Figure SEEDATTN_57_02 — Layer pairwise stability matrix

Rows:

```text
layers
```

columns/markers:

```text
seed pairs
```

show:

```text
JSD
Wasserstein
```

in separate panels.

---

# 102. Figure SEEDATTN_57_03 — Per-target layer stability distributions

Box/ECDF of:

```text
per-target layer head-mean JSD
```

for all seed pairs/layers.

No iid inferential labels.

---

# 103. Figure SEEDATTN_57_04 — Head matching cost matrices

For each:

```text
layer × seed pair
```

show JSD cost matrix.

Rows/columns architectural head indices.

Overlay canonical matched pairs.

---

# 104. Head matching matrix scale

JSD:

```text
0..ln(2).
```

Same scale across matrices.

---

# 105. Figure SEEDATTN_57_05 — Canonical head mapping diagram

Show for each layer:

```text
seed42 heads
→ seed123 matched heads
→ seed2026 matched heads.
```

Include:

```text
cycle consistency
ambiguity warning
```

without visual implication of quality.

---

# 106. Figure SEEDATTN_57_06 — Matching sensitivity agreement

Compare:

```text
JSD mapping
vs
Wasserstein mapping.
```

Report agreement fractions.

---

# 107. Figure SEEDATTN_57_07 — Matched-head mean profiles

For each:

```text
layer × canonical group
```

overlay three seed profiles.

No same-index assumption because mapping is explicit.

---

# 108. Figure SEEDATTN_57_08 — Matched-head pairwise JSD/Wasserstein

Architectural canonical-group order.

No ranking.

---

# 109. Figure SEEDATTN_57_09 — Matched-head per-target stability

Distribution of per-target:

```text
JSD
Wasserstein
```

by canonical group.

---

# 110. Figure SEEDATTN_57_10 — Dense-case full-map stability

For frozen Phase51 cases:

```text
mean query JSD
```

by:

```text
case
layer
canonical group
seed pair.
```

Report maybe shared ranks1–5 in body; all cases in artifact.

---

# 111. Figure SEEDATTN_57_11 — Layer error-conditioned effect stability

Show three seeds side-by-side for:

```text
AE Spearman
high-low Cliff's delta
```

on six core metrics.

---

# 112. Figure SEEDATTN_57_12 — Shared-cohort layer stability

Show seed-specific:

```text
SHARED_HIGH vs SHARED_LOW
```

layer head-mean effects.

---

# 113. Figure SEEDATTN_57_13 — Prediction spread vs attention disagreement

Scatter:

```text
x = prediction seed range
y = layer mean pairwise attention JSD.
```

One panel per layer.

No regression-causal claim.

---

# 114. Figure SEEDATTN_57_14 — Stability evidence summary

A compact table-like figure showing:

```text
layer profile distances
matching agreement
cycle consistency
matched-head distances
error-effect sign agreement.
```

No weighted aggregate stability score.

---

# 115. No “overall attention stability score”

Forbidden:

```text
0.4*profile + 0.3*matching + ...
```

Use transparent separate evidence.

---

# 116. Layer head-mean stability table

Create:

```text
layer_head_mean_seed_stability.csv.
```

Schema:

```text
layer_idx0
seed_a
seed_b
jsd
l1
l2
cosine
pearson
spearman
wasserstein_minutes
status
```

---

# 117. Layer head-mean three-seed summary schema

`layer_head_mean_seed_stability_summary.csv`:

```text
layer_idx0
pair_count=3
mean_pairwise_jsd
max_pairwise_jsd
mean_pairwise_wasserstein_minutes
max_pairwise_wasserstein_minutes
mean_pairwise_cosine
min_pairwise_cosine
status
```

---

# 118. Per-target layer stability schema

`layer_head_mean_per_target_stability.csv`:

```text
target_id
target_timestamp
layer_idx0
seed_a
seed_b
jsd
l1
cosine
wasserstein_minutes
status
```

---

# 119. Per-target layer three-seed disagreement schema

`layer_attention_disagreement_by_target.csv`:

```text
target_id
target_timestamp
layer_idx0
mean_pairwise_jsd
max_pairwise_jsd
mean_pairwise_wasserstein_minutes
max_pairwise_wasserstein_minutes
status
```

---

# 120. Layer metric stability schema

`layer_attention_metric_seed_stability.csv`:

```text
layer_idx0
metric
seed_a
seed_b
N
spearman_across_targets
mean_absolute_difference
median_absolute_difference
status
```

Metrics:

```text
normalized_entropy
expected_lag_minutes
recent1h_mass
recent6h_mass
top5_mass
lag80_minutes.
```

---

# 121. Head matching cost schema

`head_matching_cost_matrices.csv`:

```text
layer_idx0
seed_a
seed_b
head_a_idx0
head_b_idx0
jsd_cost
wasserstein_minutes
cosine
l1
status
```

---

# 122. Canonical assignment schema

`head_matching_assignments.csv`:

```text
layer_idx0
seed_a
seed_b
head_a_idx0
head_b_idx0
assignment_method=MIN_TOTAL_JSD
matched
matched_edge_jsd
matched_edge_wasserstein
total_assignment_jsd
total_assignment_wasserstein
assignment_rank
status
```

Rows for all head pairs can include `matched=false`; preferred companion matched-only table also allowed.

---

# 123. Matching ambiguity schema

`head_matching_ambiguity_audit.csv`:

```text
layer_idx0
seed_a
seed_b
head_count
permutation_count
best_total_jsd
second_best_total_jsd
assignment_gap_jsd
best_total_wasserstein
num_assignments_within_tie_tolerance
match_tie_tolerance
ambiguous_match_warning
status
```

---

# 124. Matched-edge margin schema

`head_matching_edge_margin_audit.csv`:

```text
layer_idx0
seed_a
seed_b
head_a_idx0
matched_head_b_idx0
matched_jsd
next_best_row_jsd
row_margin
next_best_column_jsd
column_margin
status
```

---

# 125. Cycle consistency schema

`head_matching_cycle_consistency.csv`:

```text
layer_idx0
seed123_head_idx0
direct_matched_seed2026_head
anchor_induced_seed2026_head
cycle_consistent
status
```

Layer summary:

```text
consistent_count
head_count
cycle_consistency_fraction.
```

---

# 126. Matching sensitivity schema

`head_matching_wasserstein_sensitivity.csv`:

```text
layer_idx0
seed_a
seed_b
head_a_idx0
jsd_match_head_b
wasserstein_match_head_b
pair_agrees
status
```

Summary:

```text
agreement_count
head_count
agreement_fraction.
```

---

# 127. Canonical three-seed group schema

`canonical_matched_head_groups.csv`:

```text
layer_idx0
canonical_group
anchor_seed=42
seed42_head_idx0
seed123_head_idx0
seed2026_head_idx0
mapping_42_123_ambiguous
mapping_42_2026_ambiguous
cycle_consistent
status
```

---

# 128. Matched-head mean-profile stability schema

`matched_head_profile_seed_stability.csv`:

```text
layer_idx0
canonical_group
seed_a
seed_b
head_a_idx0
head_b_idx0
jsd
wasserstein_minutes
cosine
pearson
spearman
l1
l2
match_ambiguity_warning
status
```

---

# 129. Matched-head three-seed profile summary

`matched_head_profile_stability_summary.csv`:

```text
layer_idx0
canonical_group
mean_pairwise_jsd
max_pairwise_jsd
mean_pairwise_wasserstein_minutes
max_pairwise_wasserstein_minutes
mean_pairwise_cosine
min_pairwise_cosine
any_match_ambiguity
cycle_consistent
status
```

---

# 130. Matched-head consensus profile schema

`matched_head_consensus_profile.csv`:

```text
layer_idx0
canonical_group
lag_steps
lag_minutes
seed42_weight
seed123_weight
seed2026_weight
mean_weight
sample_sd_weight
min_weight
max_weight
consensus_profile_sum
status
```

---

# 131. Per-target matched-head stability schema

`matched_head_per_target_stability.csv`:

```text
target_id
target_timestamp
layer_idx0
canonical_group
seed_a
seed_b
jsd
wasserstein_minutes
cosine
l1
match_ambiguity_warning
status
```

---

# 132. Per-target matched-group disagreement schema

`matched_head_disagreement_by_target.csv`:

```text
target_id
target_timestamp
layer_idx0
canonical_group
mean_pairwise_jsd
max_pairwise_jsd
mean_pairwise_wasserstein_minutes
max_pairwise_wasserstein_minutes
status
```

---

# 133. Matched-head metric stability schema

`matched_head_metric_seed_stability.csv`:

```text
layer_idx0
canonical_group
metric
seed_a
seed_b
N
spearman_across_targets
mean_absolute_difference
median_absolute_difference
match_ambiguity_warning
status
```

---

# 134. Matched-head top1 stability schema

`matched_head_top1_lag_stability.csv`:

```text
layer_idx0
canonical_group
seed_a
seed_b
tvd
jsd
modal_lag_seed_a
modal_lag_seed_b
same_modal_lag
match_ambiguity_warning
status
```

---

# 135. Dense case stability schema

`dense_case_attention_seed_stability.csv`:

```text
case_row_idx0
target_id
target_timestamp
selection_roles
layer_idx0
canonical_group
seed_a
seed_b
head_a_idx0
head_b_idx0
mean_query_jsd
median_query_jsd
max_query_jsd
mean_query_cosine
min_query_cosine
normalized_frobenius_distance
match_ambiguity_warning
status
```

---

# 136. Dense case summary schema

`dense_case_attention_stability_summary.csv`:

```text
layer_idx0
canonical_group
seed_a
seed_b
case_count
mean_case_mean_query_jsd
median_case_mean_query_jsd
p05
p95
mean_normalized_frobenius
status
```

Selection-conditioned caveat stored.

---

# 137. Layer error-conditioned stability schema

`layer_error_conditioned_seed_stability.csv`:

```text
layer_idx0
analysis_type
conditioning_variable
attention_metric
seed42_value
seed123_value
seed2026_value
mean_across_seeds
sample_sd_across_seeds
min
max
positive_count
negative_count
undefined_count
all_defined_same_sign
status
```

Analysis types:

```text
SPEARMAN
HIGH_LOW_MEDIAN_DELTA
HIGH_LOW_CLIFFS_DELTA
SHARED_HIGH_LOW_CLIFFS_DELTA
PROFILE_JSD
PROFILE_WASSERSTEIN.
```

---

# 138. Matched-head error-conditioned stability schema

`matched_head_error_conditioned_stability.csv`:

```text
layer_idx0
canonical_group
analysis_type
conditioning_variable
attention_metric
seed42_value
seed123_value
seed2026_value
mean_across_seeds
sample_sd_across_seeds
positive_count
negative_count
undefined_count
all_defined_same_sign
any_match_ambiguity
status
```

---

# 139. Prediction-attention disagreement schema

`prediction_attention_disagreement_association.csv`:

```text
layer_idx0
prediction_spread_metric
attention_disagreement_metric
N
spearman_rho
status
```

Prediction metrics:

```text
prediction_range
prediction_std.
```

Attention disagreement:

```text
mean_pairwise_layer_jsd
mean_pairwise_layer_wasserstein.
```

---

# 140. Matched-head prediction-disagreement secondary schema

`matched_head_prediction_disagreement_association.csv`:

```text
layer_idx0
canonical_group
prediction_spread_metric
attention_disagreement_metric
N
spearman_rho
any_match_ambiguity
status
```

Secondary only.

---

# 141. Stability evidence summary table

Create:

```text
attention_seed_stability_evidence_summary.csv.
```

Rows:

```text
LAYER_PROFILE
LAYER_PER_TARGET
HEAD_MATCHING
MATCH_CYCLE
MATCH_METHOD_SENSITIVITY
MATCHED_HEAD_PROFILE
MATCHED_HEAD_PER_TARGET
DENSE_CASE_FULL_MAP
ERROR_CONDITIONED_LAYER
ERROR_CONDITIONED_MATCHED_HEAD
PREDICTION_ATTENTION_DISAGREEMENT.
```

Columns contain transparent metric summaries only.

No weighted score.

---

# 142. No stable/unstable categorical column by arbitrary threshold

Allowed status fields:

```text
PASS
WARNING
NOT_APPLICABLE
```

for methodological/availability conditions.

Do not assign:

```text
STABLE / UNSTABLE
```

from ad hoc metric thresholds.

---

# 143. Matching reproducibility

Given:

```text
same mean profiles
same tie tolerance
same enumeration order
```

canonical mapping must be deterministic.

---

# 144. Permutation enumeration order

Canonical:

```text
lexicographic permutations of target-seed head indices.
```

Store enumeration/ranking metadata.

---

# 145. No result-dependent anchor change

If seed42 mapping is ambiguous:

```text
do not switch anchor to seed123
```

because it “looks cleaner”.

Keep seed42 anchor and record ambiguity.

---

# 146. No result-dependent matching metric change

If Wasserstein mapping looks more stable:

```text
do not replace canonical JSD mapping.
```

Report sensitivity disagreement.

---

# 147. No target-specific rematching

Hard.

---

# 148. No error-specific rematching

Do not create:

```text
HIGH_ERROR head match
LOW_ERROR head match.
```

Matching is frozen globally before error-conditioned stability checks.

---

# 149. No regime-specific rematching

No.

---

# 150. No case-specific rematching

No.

---

# 151. Matching is frozen before applying Phase56 findings

Workflow:

```text
Phase54 mean profiles
→ canonical matching
→ freeze matching SHA
→ only then map Phase56 head-level results.
```

This avoids outcome-conditioned alignment.

---

# 152. Matching fingerprint

Create:

```text
head_matching_fingerprint.json
```

including:

```text
profile source SHA
cost metric
tie tolerance
seed anchor
pairwise assignments
assignment SHA.
```

---

# 153. Error-conditioned matching independence audit

Artifact:

```text
head_matching_independence_audit.csv.
```

Hard checks:

```text
forecast error not used in matching
Phase56 effects not used in matching
worst-case IDs not used in matching
regimes not used in matching
prediction spread not used in matching.
```

---

# 154. Phase48 prediction-spread analysis remains secondary

Do not use prediction disagreement to adjust head matching.

---

# 155. Same target independence caveat

Test timestamps are temporally dependent.

Per-target stability distributions are empirical descriptive distributions.

No iid confidence intervals.

---

# 156. Three-seed limitation

Only:

```text
3 final seeds
```

are available.

Therefore mean/SD across seeds describe the chosen stochastic realizations, not the full seed population.

---

# 157. Head matching limitation

Matching on mean temporal profiles captures similarity in:

```text
last-query attention allocation.
```

It does not guarantee equality of:

```text
Q/K/V parameters
value outputs
functional contribution
full-matrix behavior.
```

---

# 158. Dense-map stability limitation

Dense full maps are only available for frozen Phase51 case set.

Whole-Test full-matrix stability is not measured directly.

Whole-Test last-query stability is primary.

---

# 159. Pooling caveat

If final pooling:

```text
LAST_STEP
```

last-query stability is particularly relevant to the token consumed by regression head.

If final pooling:

```text
MEAN
```

last-query stability is incomplete; layer/full-matrix summaries must receive greater interpretive weight.

---

# 160. Value-path caveat

Even stable attention weights do not prove stable head outputs because value projections/representations can differ.

---

# 161. Error-conditioned caveat

Stable error-attention association does not imply causal mechanism.

---

# 162. Prediction-spread caveat

Correlation between attention disagreement and prediction spread does not establish that attention instability caused prediction instability.

---

# 163. No statistical significance requirement

Phase57 does not need:

```text
p<0.05
confidence interval excluding zero.
```

Core evidence is descriptive stability.

---

# 164. Optional bootstrap is not required

No need to add block bootstrap at this late phase unless formally amended.

---

# 165. Output directory

```text
artifacts/
└── seed_stability_attention/
    ├── seed_stability_attention_manifest.json
    ├── seed_stability_attention_contract.json
    ├── phase57_preflight_audit.csv
    ├── seed_stability_source_verification.csv
    ├── seed_pair_manifest.csv
    ├── layer_head_mean_seed_stability.csv
    ├── layer_head_mean_seed_stability_summary.csv
    ├── layer_head_mean_per_target_stability.csv
    ├── layer_attention_disagreement_by_target.csv
    ├── layer_attention_metric_seed_stability.csv
    ├── head_matching_cost_matrices.csv
    ├── head_matching_assignments.csv
    ├── head_matching_ambiguity_audit.csv
    ├── head_matching_edge_margin_audit.csv
    ├── head_matching_cycle_consistency.csv
    ├── head_matching_wasserstein_sensitivity.csv
    ├── canonical_matched_head_groups.csv
    ├── head_matching_fingerprint.json
    ├── head_matching_independence_audit.csv
    ├── matched_head_profile_seed_stability.csv
    ├── matched_head_profile_stability_summary.csv
    ├── matched_head_consensus_profile.csv
    ├── matched_head_per_target_stability.csv
    ├── matched_head_disagreement_by_target.csv
    ├── matched_head_metric_seed_stability.csv
    ├── matched_head_top1_lag_stability.csv
    ├── dense_case_attention_seed_stability.csv
    ├── dense_case_attention_stability_summary.csv
    ├── layer_error_conditioned_seed_stability.csv
    ├── matched_head_error_conditioned_stability.csv
    ├── prediction_attention_disagreement_association.csv
    ├── matched_head_prediction_disagreement_association.csv
    ├── attention_seed_stability_evidence_summary.csv
    ├── seed_stability_attention_findings.csv
    ├── seed_stability_attention_tests.csv
    ├── seed_stability_attention_discrepancies.json
    ├── phase58_final_tables_handoff.json
    ├── phase59_conclusions_context_handoff.json
    ├── seed_stability_attention_summary.json
    ├── seed_stability_attention_report.md
    ├── figures/
    │   ├── SEEDATTN_57_01_layer_head_mean_profiles.png
    │   ├── SEEDATTN_57_02_layer_pairwise_stability.png
    │   ├── SEEDATTN_57_03_layer_per_target_stability.png
    │   ├── SEEDATTN_57_04_head_matching_cost_matrices.png
    │   ├── SEEDATTN_57_05_canonical_head_mapping.png
    │   ├── SEEDATTN_57_06_matching_sensitivity_agreement.png
    │   ├── SEEDATTN_57_07_matched_head_mean_profiles.png
    │   ├── SEEDATTN_57_08_matched_head_pairwise_distances.png
    │   ├── SEEDATTN_57_09_matched_head_per_target_stability.png
    │   ├── SEEDATTN_57_10_dense_case_full_map_stability.png
    │   ├── SEEDATTN_57_11_error_conditioned_layer_stability.png
    │   ├── SEEDATTN_57_12_shared_cohort_layer_stability.png
    │   ├── SEEDATTN_57_13_prediction_spread_vs_attention_disagreement.png
    │   └── SEEDATTN_57_14_stability_evidence_summary.png
    ├── README_SEED_STABILITY_ATTENTION.md
    └── phase_57_signoff.json
```

---

# 166. Required outputs

```text
O57.1  Analysis manifest
O57.2  Analysis contract
O57.3  Preflight audit
O57.4  Source verification
O57.5  Seed-pair manifest
O57.6  Layer head-mean profile stability
O57.7  Layer head-mean stability summary
O57.8  Per-target layer stability
O57.9  Per-target three-seed layer disagreement
O57.10 Layer attention metric stability
O57.11 Head matching cost matrices
O57.12 Canonical head assignments
O57.13 Matching ambiguity audit
O57.14 Matched-edge margin audit
O57.15 Cycle-consistency audit
O57.16 Wasserstein matching sensitivity
O57.17 Canonical three-seed matched-head groups
O57.18 Matching fingerprint
O57.19 Matching-independence audit
O57.20 Matched-head mean-profile stability
O57.21 Matched-head stability summary
O57.22 Matched-head consensus profiles
O57.23 Per-target matched-head stability
O57.24 Per-target matched-group disagreement
O57.25 Matched-head metric stability
O57.26 Matched-head top1-lag stability
O57.27 Dense full-map case stability
O57.28 Dense-case stability summary
O57.29 Layer error-conditioned seed stability
O57.30 Matched-head error-conditioned stability
O57.31 Prediction-attention disagreement association
O57.32 Stability evidence summary
O57.33 Figures
O57.34 Findings
O57.35 Phase58 handoff
O57.36 Phase59 context handoff
O57.37 Tests
O57.38 Discrepancies
O57.39 Summary JSON
O57.40 Human-readable report
O57.41 README
O57.42 Sign-off
```

---

# 167. Analysis manifest

`seed_stability_attention_manifest.json`:

```text
phase=57
version=SEED_STABILITY_ATTENTION-v1
source_phase56_version
source_phase55_version
source_phase54_version
source_phase52_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
alignment_anchor_seed=42
alignment_anchor_reason=FIRST_PREDECLARED_FINAL_SEED
canonical_matching_cost=JSD_MEAN_TEMPORAL_PROFILE
matching_method=EXHAUSTIVE_PERMUTATION
match_tie_tolerance=1e-12
matching_sensitivity=WASSERSTEIN_MINUTES
primary_permutation_invariant_view=LAYER_HEAD_MEAN
new_attention_extraction=false
new_test_inference=false
model_training=false
best_seed_selection=false
best_head_selection=false
head_pruning=false
status
created_at
```

---

# 168. Analysis contract

`seed_stability_attention_contract.json` must freeze:

```text
Seeds:
42,123,2026.

Seed pairs:
42-123
42-2026
123-2026.

Primary:
layer head-mean permutation-invariant stability.

Canonical head matching:
within each layer
full-Test mean last-query temporal profiles
minimum total JSD
exhaustive permutations
tie tolerance 1e-12
Wasserstein tie-break
lexicographic final tie-break.

Sensitivity:
minimum total Wasserstein matching.

Anchor:
seed42 only for canonical group naming.

Matched-head analyses:
mean profile
per-target last-query vectors
behavior metrics
top1 lag distributions
dense case full maps
Phase56 error-conditioned effects.

No:
same-index semantic assumption
target-specific rematching
error-specific rematching
regime-specific rematching
best seed
best head
pruning
retraining
weighted stability score.
```

---

# 169. Preflight audit

`phase57_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Required:

```text
Phase56 approved
phase57_ready=true
Phase55 approved
Phase54 approved
Phase52 approved
3 official seeds present
same final lock
same Test population
same target order
same dense-case order
same layers
same heads
same lookback
raw attention checksums match
Phase54 profiles complete
Phase56 layer/per-head effects complete
Phase48 prediction spread available for secondary analysis
matching contract frozen before matching.
```

---

# 170. Source verification schema

`seed_stability_source_verification.csv`:

```text
source_id
path
expected_sha256_if_available
observed_sha256
seed_count
target_count
layer_count
head_count
population_sha256
frozen
status
```

---

# 171. Seed-pair manifest

`seed_pair_manifest.csv`:

```text
pair_id
seed_a
seed_b
pair_order
same_architecture
same_target_order
same_lag_support
status
```

Order:

```text
P1 42-123
P2 42-2026
P3 123-2026.
```

---

# 172. Head matching fingerprint

`head_matching_fingerprint.json`:

```text
source_profile_sha256
final_lock_sha256
seed_list
layers
heads
cost_metric=JSD_NATURAL_LOG
matching_method=EXHAUSTIVE_PERMUTATION
tie_tolerance=1e-12
tie_break_2=WASSERSTEIN_MINUTES
tie_break_3=LEXICOGRAPHIC
anchor_seed=42
pairwise_assignment_sha256
canonical_group_sha256
created_before_error_effect_mapping=true
status
```

---

# 173. Matching-independence audit

`head_matching_independence_audit.csv`:

```text
check
used_in_matching
expected
status
```

Required:

```text
mean temporal profiles             true
forecast error                     false
residual sign                      false
Phase50 regimes                    false
Phase51 worst-case membership      false
Phase56 effect sizes               false
prediction seed spread             false
Test RMSE                          false.
```

---

# 174. Layer head-mean stability integrity tests

For every seed/layer:

```text
profile sum≈1
finite
nonnegative.
```

For pairwise matrices:

```text
JSD >=0
JSD <= ln2
Wasserstein >=0
cosine valid.
```

---

# 175. Matching cost integrity

For every cost matrix:

```text
shape H×H
all finite
JSD in [0,ln2]
Wasserstein >=0
same lag support.
```

---

# 176. Permutation count audit

Expected:

\[
H!
\]

Under current protocol:

```text
H=2 → 2
H=4 → 24.
```

Actual enumeration count must match.

---

# 177. Assignment integrity

Each canonical mapping must be bijective:

```text
every source head matched exactly once
every target-seed head used exactly once.
```

Hard.

---

# 178. Canonical group integrity

For each layer:

```text
number of canonical groups = H
```

and each seed contributes:

```text
exactly H unique heads.
```

---

# 179. Cycle consistency integrity

Every anchor-induced path uses frozen canonical pairwise mappings.

No rematching.

---

# 180. Matched profile integrity

Every matched mean profile:

```text
same lag length
sum≈1
```

before pairwise metrics.

---

# 181. Consensus profile integrity

For each canonical group:

```text
sum(mean_weight over lag)≈1.
```

---

# 182. Per-target matched-head integrity

For every target/group/seed:

```text
exact one vector
same lag support
vector sum≈1.
```

---

# 183. Dense-map integrity

For every dense case/matched head:

```text
[L,L]
finite
each query row sums≈1.
```

No renormalization.

---

# 184. Error-effect alignment integrity

Phase56 per-head effects are reindexed by:

```text
canonical matched group
```

using frozen matching.

No effect value recomputed with different cohorts.

---

# 185. Shared-cohort integrity

Phase56 shared cohorts must have:

```text
same target IDs across seeds.
```

Verify before cross-seed layer comparison.

---

# 186. Prediction-spread alignment integrity

Phase48:

```text
prediction_seed_spread
```

must map one-to-one by target ID to Phase57 attention-disagreement rows.

No dropped targets.

---

# 187. Findings codes

Possible:

```text
LAYER_HEAD_MEAN_PROFILES_HIGHLY_SIMILAR_DESCRIPTIVE
LAYER_HEAD_MEAN_PROFILES_DIFFER_DESCRIPTIVELY
LAYER_PER_TARGET_ATTENTION_STABLE_DESCRIPTIVE
LAYER_PER_TARGET_ATTENTION_VARIABLE_DESCRIPTIVE
HEAD_MATCHING_UNAMBIGUOUS
HEAD_MATCHING_AMBIGUOUS
HEAD_MATCHING_CYCLE_CONSISTENT
HEAD_MATCHING_CYCLE_INCONSISTENT
JSD_WASSERSTEIN_MATCHING_AGREE
JSD_WASSERSTEIN_MATCHING_DIFFER
MATCHED_HEAD_PROFILES_SIMILAR_DESCRIPTIVE
MATCHED_HEAD_PROFILES_VARIABLE_DESCRIPTIVE
MATCHED_HEAD_TARGET_LEVEL_STABILITY_HIGHER_DESCRIPTIVE
MATCHED_HEAD_TARGET_LEVEL_STABILITY_LOWER_DESCRIPTIVE
DENSE_CASE_ATTENTION_STABLE_DESCRIPTIVE
DENSE_CASE_ATTENTION_VARIABLE_DESCRIPTIVE
ERROR_ATTENTION_SIGN_CONSISTENT_ACROSS_SEEDS
ERROR_ATTENTION_SIGN_VARIES_ACROSS_SEEDS
SHARED_COHORT_EFFECT_CONSISTENT_ACROSS_SEEDS
SHARED_COHORT_EFFECT_VARIES_ACROSS_SEEDS
LAYER_LEVEL_EFFECTS_MORE_STABLE_THAN_HEAD_LEVEL_DESCRIPTIVE
PREDICTION_SPREAD_ASSOCIATED_WITH_ATTENTION_DISAGREEMENT
NO_CLEAR_PREDICTION_ATTENTION_DISAGREEMENT_ASSOCIATION
NO_BEST_SEED_SELECTED
NO_BEST_HEAD_SELECTED
NO_RETUNING
NO_CAUSAL_CLAIM
ATTENTION_ROBUSTNESS_EVIDENCE_READY_FOR_FINAL_TABLES
```

Any “similar/variable” wording must be accompanied by exact distances, not threshold-only labels.

---

# 188. Safe findings language

Safe:

> The Layer 1 head-mean temporal profiles were similar across the three final seeds, with pairwise JSD values of X, Y and Z and Wasserstein distances of A, B and C minutes.

Safe:

> Canonical JSD matching mapped seed42 Head 1 to seed123 Head 3 and seed2026 Head 2, illustrating why same-index comparison would have been inappropriate for this layer.

Safe:

> The direct 123–2026 mapping disagreed with the seed42-induced mapping for one head, indicating that head identity was not fully cycle-consistent.

Safe:

> The sign of the layer-level association between absolute error and recent-one-hour attention mass was the same across all three seeds.

---

# 189. Unsafe findings language

Do not say:

```text
seed42 learned the correct attention
seed123 is the unstable seed
Head2 is the true semantic daily head
matching proves functional equivalence
stable attention proves explainability
attention instability causes prediction spread.
```

---

# 190. No seed ranking

Do not compute:

```text
seed attention stability score
```

and rank:

```text
42 > 123 > 2026.
```

All three are official stochastic realizations.

---

# 191. No head ranking

No.

---

# 192. No head pruning

Even highly similar matched/within-seed profiles do not authorize pruning.

---

# 193. No model redesign

Phase57 is the end of attention analysis.

Any architecture redesign belongs:

```text
future work
```

not current Test cycle.

---

# 194. No Test retuning

Hard.

---

# 195. Discrepancy taxonomy

`seed_stability_attention_discrepancies.json`:

```text
PHASE56_NOT_APPROVED
PHASE57_HANDOFF_NOT_READY
PHASE55_SOURCE_MISSING
PHASE54_SOURCE_MISSING
PHASE52_RAW_SOURCE_MISSING
RAW_SHA_MISMATCH
FINAL_LOCK_MISMATCH
TEST_POPULATION_MISMATCH
TARGET_ORDER_MISMATCH
DENSE_CASE_ORDER_MISMATCH
LAYER_COUNT_MISMATCH
HEAD_COUNT_MISMATCH
LOOKBACK_MISMATCH
SEED_MISSING
EXTRA_SEED_ADDED
SEED_DROPPED
LAYER_PROFILE_SUM_MISMATCH
PER_TARGET_VECTOR_SUM_MISMATCH
HEAD_MATCH_PROFILE_SUM_MISMATCH
MATCH_COST_NONFINITE
JSD_COST_INVALID
WASSERSTEIN_COST_INVALID
PERMUTATION_COUNT_MISMATCH
MATCH_NOT_BIJECTIVE
MATCH_TIE_TOL_DRIFT
MATCHING_METRIC_CHANGED_POST_HOC
ANCHOR_CHANGED_POST_HOC
PERFORMANCE_BASED_ANCHOR_SELECTION
ERROR_USED_IN_HEAD_MATCHING
REGIME_USED_IN_HEAD_MATCHING
WORST_CASE_USED_IN_HEAD_MATCHING
PREDICTION_SPREAD_USED_IN_HEAD_MATCHING
TARGET_SPECIFIC_REMATCHING
ERROR_SPECIFIC_REMATCHING
REGIME_SPECIFIC_REMATCHING
CASE_SPECIFIC_REMATCHING
WASSERSTEIN_SENSITIVITY_REPLACED_CANONICAL_MAPPING
CYCLE_CONSISTENCY_COMPUTATION_ERROR
CANONICAL_GROUP_DUPLICATE_HEAD
CONSENSUS_PROFILE_SUM_MISMATCH
MATCHED_HEAD_TARGET_ALIGNMENT_MISMATCH
TOP1_DISTRIBUTION_MISMATCH
DENSE_ATTENTION_ROW_SUM_MISMATCH
IMAGE_PIXEL_SIMILARITY_USED
PHASE56_EFFECT_RECOMPUTED_WITH_CHANGED_COHORT
SHARED_COHORT_MISMATCH
SAME_INDEX_HEAD_SEMANTIC_ALIGNMENT_ASSUMED
CROSS_SEED_HEAD_AVERAGED_BEFORE_MATCHING
BEST_SEED_SELECTED
BEST_HEAD_SELECTED
HEAD_PRUNING_ATTEMPT
HEAD_ABLATION_ATTEMPT
MODEL_RETRAINING_ATTEMPT
NEW_TEST_INFERENCE_ATTEMPT
NEW_ATTENTION_EXTRACTION_ATTEMPT
WEIGHTED_STABILITY_SCORE_CREATED
P_VALUE_VOTING_USED
CAUSAL_ATTRIBUTION_CLAIM
OTHER
```

---

# 196. Status model

## PASS

```text
all three seed sources verified
layer head-mean stability complete
canonical head matching complete
ambiguity/cycle/sensitivity audits complete
matched-head mean-profile stability complete
matched-head per-target stability complete
dense-case stability complete
error-conditioned stability complete
prediction-attention disagreement secondary check complete
no seed/head selection
Phase58 handoff ready.
```

## PASS_WITH_WARNING

Possible:

```text
head matching ambiguous
cycle inconsistency
JSD/Wasserstein matching disagreement
substantial seed attention variation
dense-case stability differs from full-Test last-query stability
error-conditioned effects vary by seed
MEAN pooling limitation.
```

These are substantive findings, not methodological failure.

## FAIL

Examples:

```text
same-index heads averaged without matching
matching uses error
target-specific rematching
seed dropped
raw attention mismatch
model retrained.
```

---

# 197. Phase58 handoff purpose

Phase58 will create final report tables/figures.

Phase57 should provide a compact, frozen set of attention robustness evidence.

---

# 198. Phase58 handoff schema

`phase58_final_tables_handoff.json`:

```text
source_phase57_version
source_phase56_version
source_phase55_version
source_phase54_version
source_phase53_version
source_phase52_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
layer_head_mean_stability_summary
head_matching_assignments
head_matching_ambiguity
cycle_consistency
matching_sensitivity
matched_head_profile_stability_summary
matched_head_metric_stability
dense_case_attention_stability_summary
layer_error_conditioned_seed_stability
prediction_attention_disagreement_association
attention_seed_stability_evidence_summary
recommended_attention_tables=[
  LAYER_TEMPORAL_PROFILE_SUMMARY,
  HEAD_BEHAVIOR_COMPARISON,
  ERROR_CONDITIONED_ATTENTION,
  SEED_STABILITY_ATTENTION
]
recommended_attention_figures=[
  ATTENTION_HEATMAP,
  LAST_QUERY_PROFILE,
  ERROR_CONDITIONED_PROFILE,
  SEED_STABILITY_PROFILE
]
best_seed_selected=false
best_head_selected=false
causal_claim=false
ready_for_phase58=true
```

Actual final report content selection remains Phase58's responsibility.

---

# 199. Phase59 context handoff

`phase59_conclusions_context_handoff.json`:

```text
source_phase57_version
main_stability_findings
matching_limitations
seed_variability_findings
error_attention_robustness_findings
prediction_attention_disagreement_findings
attention_is_temporal_not_feature_importance=true
attention_is_not_causal_explanation=true
three_seed_limit=true
no_model_change_after_test=true
ready_for_phase59_context=true
```

---

# 200. Execution sequence

```text
1. Verify Phase56/55/54/52 signoffs and sources.
2. Verify three seed raw checksums and exact target/case ordering.
3. Freeze Phase57 stability/matching contract.
4. Build seed-pair manifest.
5. Recompute/verify layer head-mean profiles from raw Phase52 last-query arrays.
6. Compute layer mean-profile pairwise stability.
7. Compute per-target layer head-mean stability.
8. Compute layer attention metric stability across seeds.
9. Build full-Test mean profile for every seed/layer/head.
10. Build JSD matching cost matrices.
11. Enumerate all head permutations per layer/seed pair.
12. Freeze minimum-total-JSD assignments with deterministic tie-break.
13. Write matching fingerprint before using Phase56 effects.
14. Audit matching ambiguity and edge margins.
15. Compute direct vs anchor-induced cycle consistency.
16. Compute Wasserstein-only sensitivity matching.
17. Freeze canonical seed42-anchored matched groups.
18. Compute matched-head mean-profile stability.
19. Build matched-head consensus profiles.
20. Apply frozen matching to every Test target.
21. Compute per-target matched-head stability.
22. Compute matched-head behavior metric/top1-lag stability.
23. Apply frozen matching to dense Phase51 case maps.
24. Compute raw full-map rowwise stability metrics.
25. Map Phase56 per-head effects into canonical matched groups.
26. Compute layer head-mean error-conditioned stability across seeds.
27. Compute shared-cohort effect stability.
28. Join Phase48 prediction spread.
29. Compute attention-disagreement vs prediction-spread associations.
30. Build complete figures/tables in canonical order.
31. Write findings with no best seed/head.
32. Write Phase58 and Phase59 handoffs.
33. Run integrity/scope/discrepancy tests.
34. Write summary/report/README.
35. Sign off.
```

---

# 201. Recommended pseudocode

```text
p56 = load_phase56_signoff()
assert p56.overall_status in {"PASS","PASS_WITH_WARNING"}
assert p56.phase57_ready

raw = {
    42: load_phase52_last_query(42),
    123: load_phase52_last_query(123),
    2026: load_phase52_last_query(2026)
}

dense = {
    42: load_phase52_dense_cases(42),
    123: load_phase52_dense_cases(123),
    2026: load_phase52_dense_cases(2026)
}

verify_raw_checksums()
verify_same_test_target_order()
verify_same_dense_case_order()
verify_same_architecture()

freeze_phase57_contract(
    seeds=[42,123,2026],
    anchor_seed=42,
    canonical_cost="JSD",
    matching="EXHAUSTIVE_PERMUTATION",
    tie_tol=1e-12,
    tie_break_2="WASSERSTEIN_MINUTES",
    tie_break_3="LEXICOGRAPHIC",
    sensitivity_matching="WASSERSTEIN_MINUTES"
)

# A. Permutation-invariant layer stability
layer_mean = {
    seed: mean(raw[seed], axis="head")
    for seed in [42,123,2026]
}
# [N,Layers,L]

for layer in layers:
    for seed_a, seed_b in seed_pairs:
        P_a = mean(layer_mean[seed_a][:,layer,:], axis="target")
        P_b = mean(layer_mean[seed_b][:,layer,:], axis="target")

        verify_probability_profile(P_a)
        verify_probability_profile(P_b)

        compute_profile_stability(
            P_a, P_b,
            metrics=[
                "JSD","L1","L2","COSINE",
                "PEARSON","SPEARMAN",
                "WASSERSTEIN_MINUTES"
            ]
        )

        for target_id in all_test_targets:
            compute_per_target_layer_stability(
                layer_mean[seed_a][target_id,layer,:],
                layer_mean[seed_b][target_id,layer,:]
            )

# B. Build canonical head profiles
mean_head_profiles = {}
for seed in [42,123,2026]:
    mean_head_profiles[seed] = mean(
        raw[seed],
        axis="target"
    )
    # [Layers,Heads,L]

# C. Pairwise canonical matching
pairwise_matches = {}

for layer in layers:
    for seed_a, seed_b in seed_pairs:

        cost_jsd = zeros([H,H])
        cost_wass = zeros([H,H])

        for ha in heads:
            for hb in heads:
                pa = mean_head_profiles[seed_a][layer,ha,:]
                pb = mean_head_profiles[seed_b][layer,hb,:]

                cost_jsd[ha,hb] = JSD(pa,pb)
                cost_wass[ha,hb] = wasserstein(
                    support=lag_minutes,
                    weights_a=pa,
                    weights_b=pb
                )

        permutations = lexicographic_permutations(range(H))
        assert len(permutations) == factorial(H)

        scored = []

        for perm in permutations:
            total_jsd = sum(
                cost_jsd[ha,perm[ha]]
                for ha in heads
            )
            total_wass = sum(
                cost_wass[ha,perm[ha]]
                for ha in heads
            )
            scored.append(
                (perm,total_jsd,total_wass)
            )

        canonical = choose_assignment(
            scored,
            primary="MIN_TOTAL_JSD",
            tie_tol=1e-12,
            secondary="MIN_TOTAL_WASSERSTEIN",
            tertiary="LEXICOGRAPHIC"
        )

        pairwise_matches[layer,seed_a,seed_b] = canonical

        audit_assignment_ambiguity(scored)
        audit_edge_margins(cost_jsd, canonical)

# Freeze matching before mapping error-conditioned results
write_matching_fingerprint(pairwise_matches)

# D. Cycle consistency
audit_cycle_consistency(
    M_42_123,
    M_42_2026,
    M_123_2026
)

# E. Wasserstein sensitivity mapping
compute_wasserstein_only_matching_and_agreement()

# F. Canonical three-seed groups anchored at seed42
canonical_groups = build_anchor_groups(
    anchor=42,
    match_42_123=M_42_123,
    match_42_2026=M_42_2026
)

# G. Mean-profile matched-head stability
for layer, group in canonical_groups:
    profiles = get_three_seed_group_profiles(...)
    compute_all_three_pairwise_stability(profiles)
    build_consensus_profile(profiles)

# H. Per-target matched-head stability
for target in all_test_targets:
    for layer, group in canonical_groups:
        vectors = get_three_seed_group_vectors(
            raw,
            target,
            layer,
            group
        )
        compute_three_pairwise_target_stability(vectors)

# I. Matched-head behavior metric stability
metrics54 = load_phase54_metrics()
compare_matched_head_metrics_across_targets(
    metrics54,
    canonical_groups
)

compare_matched_head_top1_distributions(
    phase54_top1_frequency,
    canonical_groups
)

# J. Dense case full-map stability
for case in frozen_dense_cases:
    for layer, group in canonical_groups:
        maps = get_three_seed_dense_maps(
            dense,
            case,
            layer,
            group
        )
        compute_rowwise_jsd_cosine_and_frobenius(maps)

# K. Phase56 finding stability
phase56 = load_phase56_effects()

compute_layer_error_conditioned_seed_stability(
    phase56.layer_head_mean_results
)

map_phase56_head_effects_to_canonical_groups(
    phase56.per_head_results,
    canonical_groups
)

compute_matched_head_error_effect_stability()

# L. Prediction disagreement secondary diagnostic
prediction_spread = load_phase48_prediction_seed_spread()

attention_disagreement = build_layer_attention_disagreement_by_target()

compute_spearman(
    prediction_spread,
    attention_disagreement
)

# M. Final handoff
assert no_best_seed
assert no_best_head
assert no_head_pruning
assert no_retraining
assert no_new_attention_extraction
assert no_causal_claim

write_phase58_handoff()
write_phase59_context_handoff()

signoff_phase57()
```

---

# 202. Preflight acceptance checklist

```text
[ ] Phase56 PASS/PASS_WITH_WARNING.
[ ] phase57_ready=true.
[ ] Phase55 PASS/PASS_WITH_WARNING.
[ ] Phase54 PASS/PASS_WITH_WARNING.
[ ] Phase52 PASS/PASS_WITH_WARNING.
[ ] Seed42 raw last-query exists.
[ ] Seed123 raw last-query exists.
[ ] Seed2026 raw last-query exists.
[ ] Three dense-case raw files exist.
[ ] All raw SHA256s match.
[ ] Same final lock.
[ ] Same FINAL_TEST_POP-v1.
[ ] Same target order.
[ ] Same dense-case order.
[ ] Same layers.
[ ] Same heads.
[ ] Same lookback.
[ ] Phase56 effects complete.
[ ] Phase48 prediction spread available.
[ ] Matching contract frozen before results.
```

---

# 203. Permutation-invariant layer acceptance checklist

```text
[ ] Head-mean vector computed from raw heads per seed/target/layer.
[ ] No cross-seed averaging before per-seed computation.
[ ] Every head-mean vector sums≈1.
[ ] Mean layer profile sums≈1.
[ ] All three seed pairs analyzed.
[ ] JSD complete.
[ ] L1/L2 complete.
[ ] Cosine complete.
[ ] Pearson/Spearman complete.
[ ] Wasserstein in minutes complete.
[ ] Per-target layer stability complete.
[ ] Per-target three-seed disagreement complete.
[ ] No stability threshold invented.
```

---

# 204. Matching acceptance checklist

```text
[ ] Matching separately by layer.
[ ] Representation = full-Test mean last-query profile.
[ ] Error not used.
[ ] Regime not used.
[ ] Prediction spread not used.
[ ] Cost = JSD.
[ ] JSD uses natural log.
[ ] All H! permutations enumerated.
[ ] Permutation count correct.
[ ] Tie tolerance=1e-12.
[ ] Secondary tie-break=Wasserstein minutes.
[ ] Final tie-break=lexicographic.
[ ] Mapping one-to-one.
[ ] Mapping frozen before Phase56 effect mapping.
[ ] Matching fingerprint written.
```

---

# 205. Ambiguity acceptance checklist

```text
[ ] Best total JSD recorded.
[ ] Second-best total JSD recorded.
[ ] Assignment gap recorded.
[ ] Number of tied/near-tied assignments recorded.
[ ] Edge margins recorded.
[ ] Ambiguous matches warned.
[ ] Anchor not changed due ambiguity.
[ ] Canonical cost not changed due ambiguity.
```

---

# 206. Cycle/sensitivity acceptance checklist

```text
[ ] Direct 123→2026 mapping computed.
[ ] Anchor-induced 123→42→2026 mapping computed.
[ ] Per-head cycle consistency recorded.
[ ] Layer cycle consistency fraction recorded.
[ ] Wasserstein-only sensitivity mapping computed.
[ ] JSD-vs-Wasserstein agreement recorded.
[ ] Sensitivity mapping does not replace canonical mapping.
```

---

# 207. Canonical-group acceptance checklist

```text
[ ] Anchor seed=42.
[ ] Anchor reason non-performance-based.
[ ] H canonical groups per layer.
[ ] Every seed contributes each head exactly once.
[ ] No duplicate head within a seed/layer.
[ ] Ambiguity flags propagated.
[ ] Cycle consistency included.
```

---

# 208. Matched-head profile acceptance checklist

```text
[ ] All canonical groups analyzed.
[ ] All three seed pairs analyzed.
[ ] JSD.
[ ] Wasserstein minutes.
[ ] Cosine.
[ ] Pearson.
[ ] Spearman.
[ ] L1/L2.
[ ] Three-seed summary.
[ ] Consensus profile generated only after matching.
[ ] Consensus profile sums≈1.
[ ] Per-lag seed SD recorded.
[ ] No confidence interval claim.
```

---

# 209. Per-target matched-head acceptance checklist

```text
[ ] Global frozen matching reused for every target.
[ ] No target-specific rematching.
[ ] Same target IDs.
[ ] JSD complete.
[ ] Wasserstein complete.
[ ] Cosine complete.
[ ] L1 complete.
[ ] Per-target three-seed disagreement complete.
[ ] Distribution summaries complete.
[ ] No row drops.
```

---

# 210. Matched-head metric/top1 acceptance checklist

```text
[ ] Entropy stability.
[ ] Expected-lag stability.
[ ] Recent1h stability.
[ ] Recent6h stability.
[ ] Top5 stability.
[ ] Lag80 stability.
[ ] Spearman across same targets.
[ ] Mean/median absolute differences.
[ ] Top1 TVD.
[ ] Top1 JSD.
[ ] Modal lag agreement.
[ ] No best matched group.
```

---

# 211. Dense-case acceptance checklist

```text
[ ] Exact frozen Phase51 dense cases.
[ ] Same case order.
[ ] Canonical global head mapping reused.
[ ] No case-specific rematching.
[ ] Every row remains probability distribution.
[ ] Mean query JSD computed.
[ ] Median/max query JSD computed.
[ ] Mean/min query cosine computed.
[ ] Normalized Frobenius computed from raw matrix.
[ ] No PNG/image similarity.
[ ] Selection-conditioned caveat documented.
```

---

# 212. Error-conditioned stability acceptance checklist

```text
[ ] Layer head-mean AE associations compared across 3 seeds.
[ ] Layer head-mean signed-residual associations compared.
[ ] Shared-hardness associations compared.
[ ] High-low median deltas compared.
[ ] High-low Cliff's deltas compared.
[ ] Shared-cohort effects compared.
[ ] Profile JSD/Wasserstein effect magnitudes compared.
[ ] Sign counts recorded.
[ ] Unanimous-sign flag recorded.
[ ] No p-value voting.
[ ] Matched-head effects mapped only after matching freeze.
[ ] Ambiguous-match warnings propagated.
```

---

# 213. Prediction-attention disagreement acceptance checklist

```text
[ ] Phase48 prediction range/std source verified.
[ ] Same target IDs.
[ ] Layer mean pairwise attention disagreement computed.
[ ] Spearman with prediction range.
[ ] Spearman with prediction SD.
[ ] No prediction-spread threshold.
[ ] No causal claim.
[ ] Matched-head variant clearly secondary.
```

---

# 214. Scope acceptance checklist

```text
[ ] No best seed.
[ ] No best head.
[ ] No head pruning.
[ ] No head ablation.
[ ] No retraining.
[ ] No prediction correction.
[ ] No new Test inference.
[ ] No new attention extraction.
[ ] No same-index semantic assumption.
[ ] No target-specific rematching.
[ ] No error-specific rematching.
[ ] No regime-specific rematching.
[ ] No weighted stability score.
[ ] No causal claim.
```

---

# 215. Provenance acceptance checklist

```text
[ ] Phase52 raw SHA references stored.
[ ] Phase54 profile source stored.
[ ] Phase56 effect source stored.
[ ] Final lock SHA stored.
[ ] Test population SHA stored.
[ ] Matching fingerprint stored.
[ ] Anchor reason stored.
[ ] Assignment matrices reproducible.
[ ] Phase58 handoff references exact artifacts.
[ ] Phase59 context handoff complete.
```

---

# 216. Acceptance criteria

Phase57 PASS only when:

```text
All three official final Transformer seed attention sources are verified against the same final lock, Test population, target order, architecture and lag support.

Permutation-invariant layer head-mean attention is analyzed first and provides a seed-stability view that does not depend on head identity.

Layer head-mean mean profiles and per-target vectors are compared across all three seed pairs using transparent probability-profile and temporal-distance metrics.

Canonical head matching is performed independently within each encoder layer using only full-Test mean last-query temporal profiles and minimum total Jensen–Shannon divergence.

Under the locked H2/H4 head search space, all possible one-to-one head permutations are exhaustively enumerated.

Matching ties are resolved by the predeclared JSD tolerance, Wasserstein secondary tie-break and lexicographic final tie-break without changing the canonical objective.

The matching process is demonstrably independent of forecast error, regimes, worst-case membership, prediction spread and Phase56 effect sizes.

All pairwise seed mappings are retained, while seed42 serves only as the first-predeclared deterministic anchor for naming three-seed matched groups.

Matching ambiguity, assignment gap, matched-edge margins and cycle consistency are explicitly audited rather than hidden.

A Wasserstein-only one-to-one matching is computed only as a sensitivity analysis, and disagreement with the canonical JSD mapping is reported without replacing the canonical mapping.

After matching is frozen, matched-head mean-profile stability, per-target last-query stability, behavior-metric stability and top1-lag stability are quantified across seeds.

Consensus matched-head profiles are created only after matching and preserve probability-mass integrity.

The exact same global head mapping is applied to all Test targets and dense Phase51 cases; no target-specific, error-specific, regime-specific or case-specific rematching occurs.

Dense full-attention stability is computed from raw [L,L] matrices using rowwise probability-distribution comparisons and raw numerical distances rather than image-pixel similarity.

Phase56 error-conditioned findings are compared across seeds first at the permutation-invariant layer head-mean level and secondarily at the matched-head level.

Shared-error cohorts use identical targets across seeds and provide a clean descriptive robustness view of error-conditioned attention effects.

Attention disagreement is optionally related to Phase48 prediction seed spread using permutation-invariant layer-level disagreement metrics without asserting causality.

No best seed or best head is selected, no head is pruned/ablated, no model is retrained and no prediction/Test metric is changed.

No weighted overall stability score or arbitrary stable/unstable threshold is introduced.

All findings explicitly preserve the distinction between attention allocation stability, functional equivalence and causal explanation.

A frozen attention robustness package is handed to Phase58 Final Tables and Phase59 Conclusions.
```

---

# 217. Failure conditions

Phase57 FAIL if:

```text
one final seed is omitted

raw attention sources do not share identical final lock/Test population

same-index heads are averaged across seeds before matching

matching uses forecast error or Phase56 effects

matching is recomputed separately for high/low error groups

target-specific rematching is used

case-specific rematching is used

anchor seed is changed after seeing matching quality

canonical matching metric is changed after seeing results

Wasserstein sensitivity mapping replaces JSD mapping because it looks better

matching is not bijective

permutation count is incomplete under H2/H4

consensus profile is built before head alignment

matched-head effects are averaged without ambiguity warning

PNG heatmap pixels are used for numeric stability

best seed/head is selected

head pruning/ablation occurs

model is retrained

new Test inference/attention extraction occurs

a weighted overall stability score is created

attention stability is treated as proof of functional equivalence

attention disagreement is reported as cause of prediction disagreement.
```

---

# 218. Common mistakes

## 218.1 So sánh Head1 seed42 với Head1 seed123 trực tiếp và gọi đó là stability

Không chuẩn.

Head permutation phải được xử lý.

## 218.2 Match heads bằng Test error behavior

Sai circularity.

Canonical matching chỉ dùng temporal attention profiles.

## 218.3 Match lại heads cho từng target

Sai vì sẽ làm stability trông tốt giả tạo.

## 218.4 Match lại riêng HIGH_ERROR và LOW_ERROR

Sai outcome-conditioned alignment.

## 218.5 Seed42 làm anchor rồi gọi seed42 là seed chuẩn/best

Sai.

Seed42 chỉ là deterministic first-seed anchor.

## 218.6 JSD matching không đẹp nên đổi sang Wasserstein

Sai.

Wasserstein chỉ là sensitivity audit.

## 218.7 Mapping 42→123 và 42→2026 đẹp nhưng bỏ qua direct 123→2026

Sai.

Cần cycle consistency.

## 218.8 Mean profile giống nhau rồi kết luận attention luôn stable

Sai.

Phải kiểm tra per-target stability.

## 218.9 Last-query stable rồi kết luận full attention stable

Sai.

Dense full-map case analysis là supplementary check.

## 218.10 Attention profiles giống nhau rồi kết luận heads functionally identical

Sai.

Value projections/output paths vẫn có thể khác.

## 218.11 Cross-seed effect signs giống nhau rồi gọi causal replication

Sai.

Chỉ là descriptive seed consistency.

## 218.12 Prediction spread correlate với attention disagreement rồi nói attention instability gây prediction instability

Sai causal inference.

---

# 219. Human-readable report structure

`seed_stability_attention_report.md`:

```text
1. Objective
2. Why head permutation matters
3. Frozen sources and three-seed contract
4. Permutation-invariant layer head-mean stability
5. Per-target layer attention stability
6. Canonical head-matching methodology
7. Matching cost matrices
8. Matching ambiguity and edge margins
9. Cycle-consistency audit
10. Wasserstein matching sensitivity
11. Canonical three-seed matched-head groups
12. Matched-head mean-profile stability
13. Matched-head per-target stability
14. Matched-head metric/top1 stability
15. Dense worst-case full-map stability
16. Error-conditioned layer-level stability
17. Error-conditioned matched-head stability
18. Shared-cohort seed consistency
19. Prediction-spread vs attention-disagreement analysis
20. Main attention robustness findings
21. Why no best seed/head is selected
22. Why attention stability is not functional equivalence
23. Why stability is not causal explanation
24. Three-seed limitation
25. Handoff to Final Tables
26. Definition of Done
```

---

# 220. README requirements

`README_SEED_STABILITY_ATTENTION.md` explains:

```text
why same-index heads cannot be assumed equivalent
why layer head-mean is permutation-invariant
why JSD is the canonical matching cost
why H2/H4 allows exhaustive permutation matching
how the deterministic tie-break works
why seed42 is only an alignment anchor
what matching ambiguity means
what cycle consistency means
why Wasserstein matching is sensitivity-only
why matching is frozen before error-conditioned analysis
how per-target matched-head stability is computed
why dense-case stability is supplementary
why error-conditioned stability prioritizes layer head-mean
why prediction-attention disagreement is non-causal
why no stability score/best seed/best head is created.
```

---

# 221. Summary artifact

`seed_stability_attention_summary.json`:

```text
version
source_phase56_version
source_phase55_version
source_phase54_version
source_phase52_version
final_lock_sha256
test_population_sha256
seed_list
anchor_seed
layers
heads
lookback
layer_stability_status
per_target_layer_stability_status
canonical_matching_status
matching_ambiguity_summary
cycle_consistency_summary
wasserstein_sensitivity_agreement
canonical_groups
matched_head_profile_stability_status
matched_head_per_target_stability_status
dense_case_stability_status
error_conditioned_layer_stability_status
error_conditioned_matched_head_stability_status
prediction_attention_disagreement_status
findings
best_seed_selected=false
best_head_selected=false
head_pruning=false
model_retrained=false
new_attention_extraction=false
causal_claim=false
phase58_ready
phase59_context_ready
overall_status
```

No runtime match assignments/effect values may be fabricated before execution.

---

# 222. Phase57 sign-off

`phase_57_signoff.json` minimum:

```text
phase=57
phase_name=Seed-stability attention check
version=SEED_STABILITY_ATTENTION-v1
source_phase56_version
source_phase55_version
source_phase54_version
source_phase52_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
alignment_anchor_seed=42
canonical_matching_cost=JSD
matching_method=EXHAUSTIVE_PERMUTATION
match_tie_tolerance=1e-12
matching_sensitivity=WASSERSTEIN_MINUTES
layer_head_mean_stability_complete
per_target_layer_stability_complete
canonical_head_matching_complete
matching_ambiguity_audit_complete
cycle_consistency_complete
matching_sensitivity_complete
matched_head_profile_stability_complete
matched_head_per_target_stability_complete
matched_head_metric_stability_complete
dense_case_stability_complete
error_conditioned_layer_stability_complete
error_conditioned_matched_head_stability_complete
prediction_attention_disagreement_complete
matching_independent_of_error=true
same_index_semantic_alignment_assumed=false
target_specific_rematching=false
error_specific_rematching=false
best_seed_selected=false
best_head_selected=false
head_pruning=false
head_ablation=false
model_training=false
new_test_inference=false
new_attention_extraction=false
weighted_stability_score=false
causal_claim=false
phase58_ready
phase59_context_ready
warnings
overall_status
created_at
```

---

# 223. Definition of Done

\[
\boxed{
Three\ Official\ Seeds
+
Permutation\text{-}Invariant\ Layer\ Stability
+
Deterministic\ Head\ Matching
+
Matching\ Ambiguity
+
Cycle\ Consistency
+
Matching\ Sensitivity
+
Matched\text{-}Head\ Profile\ Stability
+
Per\text{-}Target\ Stability
+
Dense\ Case\ Stability
+
Error\text{-}Finding\ Stability
+
No\ Best\ Seed/Head
+
Phase58\ Handoff
}
\]

---

# 224. Final status contract

```text
PHASE 57 checks attention stability across seeds.

Seeds:
42
123
2026.

Primary seed-stability view:
layer head-mean attention
permutation-invariant.

Head-level view:
permutation-aware matching.

Canonical matching representation:
full-Test mean last-query temporal profile.

Canonical matching cost:
Jensen–Shannon divergence.

Matching:
within layer only
all H! permutations under H2/H4
one-to-one
minimum total JSD.

Tie:
JSD tolerance 1e-12
then minimum Wasserstein
then lexicographic.

Anchor:
seed42
only because it is first predeclared seed
not because of performance.

Required pairwise mappings:
42-123
42-2026
123-2026.

Required audits:
ambiguity
assignment gap
edge margins
cycle consistency
Wasserstein matching sensitivity.

After matching:
mean-profile stability
per-target last-query stability
metric stability
top1-lag stability
dense full-map worst-case stability
error-conditioned effect stability.

Primary error-effect robustness:
layer head-mean.

Secondary:
matched-head effects
prediction spread vs attention disagreement.

Forbidden:
same-index semantic assumption
target-specific rematching
error-specific rematching
case-specific rematching
best seed
best head
head pruning
head ablation
retraining
new Test inference
new attention extraction
weighted stability score
causal claim.

After SEED_STABILITY_ATTENTION-v1 PASS:
proceed to
PHASE 58 — Final Tables.
```

---

# 225. Final check

Correct:

```text
verify 3 seed attention sources
→ permutation-invariant layer stability
→ build JSD head matching
→ freeze matching
→ ambiguity/cycle/sensitivity audits
→ matched-head stability
→ per-target stability
→ dense-case stability
→ error-effect stability
→ Phase58 handoff
```

Incorrect:

```text
Head1 seed42
vs
Head1 seed123
→ call seed stability
```

Incorrect:

```text
match heads separately for every target
```

Incorrect:

```text
choose matching method
that produces the nicest stability
```

Incorrect:

```text
seed with smallest attention disagreement
→ best seed
```

Incorrect:

```text
stable attention
→ causal explanation proven
```

Chỉ sau khi:

```text
SEED_STABILITY_ATTENTION-v1
=
PASS / PASS_WITH_WARNING
```

và:

```text
phase58_ready=true
```

mới chuyển sang **PHASE 58 — Final Tables**.
