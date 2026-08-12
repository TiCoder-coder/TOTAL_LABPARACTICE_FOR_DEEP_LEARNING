# PHASE 55 — HEAD COMPARISON

## Kế hoạch so sánh có kiểm soát hành vi temporal attention giữa các attention heads trong Final Transformer, không biến head analysis thành model selection

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Model:** Attention-Aware Transformer Encoder for regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Primary quantitative source:** `LAST_QUERY_ATTENTION-v1`  
**Raw fallback source:** Phase52 `last_query_attention_seed*.npz`  
**Primary comparison unit:** attention head within a fixed `seed × layer`  
**Cross-seed semantic alignment:** not assumed by head index  
**Phase ID:** `PHASE_55_HEAD_COMPARISON`  
**Output version:** `HEAD_COMPARISON-v1`  
**Phase trước:** `Phase_54_Last-query_attention.md`  
**Phase sau:** `Phase_56_Error-conditioned_attention.md`

---

# 1. Vai trò của Phase 55

Phase55 là bước **head-behavior comparison**.

Phase54 đã mô tả từng head bằng:

```text
entropy
normalized entropy
effective source count
expected lag
lag SD
top1 lag
top1 weight
top5 mass
recent 1h/6h/12h/24h mass
Lag50/Lag80/Lag90
mean temporal profile
lag-bin mass
top1 lag frequency.
```

Phase55 dùng các metric/profile đó để trả lời:

> Các attention heads trong cùng một encoder layer có thật sự học các temporal allocation patterns khác nhau hay chỉ gần như trùng lặp? Sự khác biệt đó nằm ở recency, concentration, lag coverage hay profile shape nào? Có dấu hiệu head redundancy hoặc head diversity hay không?

Mục tiêu:

```text
1. Verify Phase54 metrics/profiles.
2. Freeze head-comparison contract before comparison.
3. Compare heads only within the same seed and same layer as primary analysis.
4. Quantify pairwise profile similarity/divergence.
5. Quantify pairwise metric differences.
6. Quantify head redundancy/diversity descriptively.
7. Build within-layer pairwise similarity matrices.
8. Build layer-level diversity summaries.
9. Create deterministic head behavior tables.
10. Optionally derive descriptive head archetypes using predeclared rules, not unsupervised cherry-picking.
11. Preserve all heads; no pruning.
12. Do not select a “best head”.
13. Do not use forecast error to define head quality; Phase56 handles error conditioning.
14. Do not assume same-index heads are semantically aligned across seeds; Phase57 handles seed stability/head matching.
15. Handoff head-comparison artifacts to Phase56/57.
```

Nguyên tắc trung tâm:

\[
\boxed{
Compare\ Heads
+
Within\ Fixed\ Seed/Layer
+
Behavioral\ Diversity
+
Redundancy\ Diagnostics
+
No\ Best\ Head
+
No\ Pruning
+
No\ Error\ Conditioning
}
\]

---

# 2. Phase55 không phải head pruning phase

Forbidden:

```text
remove head
zero head
mask head
retrain model with fewer heads
select only “useful” heads
recompute Test performance after head ablation.
```

Head redundancy observed here is:

```text
descriptive internal-analysis evidence
```

not permission to modify final model.

---

# 3. Phase55 không phải predictive head-quality analysis

Không được định nghĩa:

```text
head tốt
head xấu
head đóng góp nhiều nhất vào RMSE
```

chỉ từ attention statistics.

Một head có:

```text
low entropy
high top1 mass
recent focus
```

không đồng nghĩa predictive superiority.

---

# 4. Error-conditioned head analysis thuộc Phase56

Phase55 không dùng:

```text
absolute error
residual
worst-case membership
regime-specific error
```

để đánh giá head.

Phase56 mới trả lời:

```text
attention behavior thay đổi thế nào khi error cao/thấp?
```

---

# 5. Seed-stability/head matching thuộc Phase57

Phase55 primary comparison:

```text
within one seed
within one layer
across heads.
```

Cross-seed:

```text
same-index head comparison
```

không được diễn giải là semantic equality.

Phase57 mới làm:

```text
cross-seed head matching
head permutation-aware stability.
```

---

# 6. Upstream hard gate

Required:

```text
phase_54_signoff.json
phase55_head_comparison_handoff.json
last_query_metrics_long.csv
last_query_metric_summary_by_head.csv
last_query_profile_by_lag.csv
last_query_layer_head_mean_profile.csv
last_query_lag_bin_mass.csv
last_query_recent_mass_summary.csv
last_query_coverage_radius_summary.csv
last_query_top1_lag_frequency.csv
```

Raw fallback:

```text
Phase52 last_query_attention_seed*.npz
```

Hard:

```text
phase55_ready = true.
```

Phase54 status:

```text
PASS
or
PASS_WITH_WARNING.
```

---

# 7. Source hierarchy

Primary:

```text
Phase54 derived quantitative tables.
```

Secondary verification:

```text
Phase52 raw last-query arrays.
```

Phase53 heatmaps:

```text
visual context only.
```

Do not digitize images.

---

# 8. Unit of comparison

Primary comparison unit:

```text
(seed, layer, head_a, head_b).
```

No primary pairwise comparison across:

```text
different layers
different seeds
```

because layer depth and seed identity alter context.

---

# 9. Within-layer head pair count

For a layer with `H` heads:

\[
P
=
\binom{H}{2}
=
\frac{H(H-1)}{2}.
\]

Runtime `H` is authoritative.

Do not hard-code B0 `H=4`.

---

# 10. Primary head-comparison questions

```text
Q55.1  Do heads within the same layer have different temporal profiles?
Q55.2  Which head pairs are highly similar vs highly distinct?
Q55.3  Is diversity driven by recent-vs-old allocation or concentration?
Q55.4  Are some heads broad/diffuse while others are narrow/concentrated?
Q55.5  Are some heads centered on systematically different expected lags?
Q55.6  Do some heads repeatedly attend to similar top1 lag regions?
Q55.7  Is one encoder layer more internally diverse than another?
Q55.8  Are there near-redundant heads by profile similarity?
Q55.9  Do qualitative heatmap differences agree with quantitative profile metrics?
Q55.10 Which head-comparison artifacts should Phase56/57 consume?
```

---

# 11. No global “best head” question

Not allowed:

```text
Which head is best?
Which head contributes most to prediction?
Which head should be removed?
```

Those are unsupported by attention weights alone.

---

# 12. Comparison dimensions

Phase55 locks five head-behavior dimensions:

```text
D1 — Temporal profile shape
D2 — Recency allocation
D3 — Concentration/diffuseness
D4 — Temporal coverage/spread
D5 — Top-lag preference
```

---

# 13. D1 — Temporal profile shape

Source:

```text
last_query_profile_by_lag.csv
```

For each:

```text
seed
layer
head
```

profile:

\[
p_h(k)
=
Mean_t[a_{t,h}(k)].
\]

Each profile:

```text
p_h(k) >= 0
sum_k p_h(k) ≈ 1.
```

---

# 14. Pairwise profile similarity metrics

Required:

```text
Pearson correlation
Spearman correlation
cosine similarity
Jensen–Shannon divergence
L1 distance
L2 distance.
```

All computed on same lag vector.

---

# 15. Why multiple profile metrics

They capture different notions:

```text
Pearson
→ linear shape association

Spearman
→ rank/order similarity

cosine
→ directional similarity

Jensen–Shannon
→ probability-distribution divergence

L1
→ absolute probability-mass redistribution

L2
→ Euclidean difference.
```

No one metric is declared universally superior.

---

# 16. Jensen–Shannon divergence

For probability vectors `P,Q`:

\[
M=\frac{P+Q}{2}
\]

\[
JSD(P,Q)
=
\frac{1}{2}KL(P\|M)
+
\frac{1}{2}KL(Q\|M).
\]

Use:

```text
natural logarithm
```

and machine-safe epsilon only in log operations.

Because mean profiles should be nonnegative normalized distributions.

---

# 17. JSD bounds

With natural log:

```text
0 <= JSD <= ln(2)
```

approximately.

Do not call it `0..1` unless normalized.

Canonical project:

```text
raw JSD in natural-log units.
```

---

# 18. No renormalization unless integrity requires check

Mean profile should already sum to ~1.

If within frozen tolerance:

```text
use raw profile.
```

Do not normalize silently.

If profile sum materially fails:

```text
STOP.
```

---

# 19. Cosine similarity caveat

Probability vectors are nonnegative, so cosine similarity is usually positive/high.

Interpret together with:

```text
JSD
L1
expected-lag differences.
```

---

# 20. Correlation caveat

Two profiles can have high correlation yet different absolute concentration.

Therefore:

```text
correlation alone is insufficient.
```

---

# 21. D2 — Recency allocation

Use Phase54:

```text
recent_1h_mass
recent_6h_mass
recent_12h_mass
recent_24h_mass
non-overlap lag-bin masses.
```

Compare per-head distributions across Test targets.

---

# 22. Primary recency comparison

For same:

```text
seed/layer
```

compare heads using:

```text
median
mean
IQR
p05/p95
```

of each recent-mass metric.

No p-value required in core analysis.

---

# 23. Pairwise recency differences

For head pair `(a,b)`:

\[
\Delta R_w
=
Median(R_{a,w})
-
Median(R_{b,w})
\]

for windows:

```text
1h
6h
12h
24h.
```

Store raw difference.

No interpretation threshold yet.

---

# 24. Non-overlap lag-bin difference

For each temporal bin:

\[
\Delta Mass_{bin}
=
Mean(Mass_{a,bin})
-
Mean(Mass_{b,bin}).
\]

This directly shows where probability mass differs.

---

# 25. D3 — Concentration/diffuseness

Use:

```text
normalized entropy
effective source count
top1 weight
top5 mass.
```

These are related but complementary.

---

# 26. Concentration direction

Typical interpretation:

```text
lower normalized entropy
lower effective source count
higher top1 weight
higher top5 mass
→ more concentrated attention.
```

Do not call concentrated attention better.

---

# 27. D4 — Temporal coverage/spread

Use:

```text
expected lag
lag SD
Lag50
Lag80
Lag90.
```

These quantify:

```text
where attention centers
how broadly it spans time
how much recent-history radius is needed to cover mass.
```

---

# 28. D5 — Top-lag preference

Use:

```text
top1 lag frequency.
```

For pair of heads compare:

```text
top1 lag distributions.
```

---

# 29. Top1 lag distribution comparison

Required metrics:

```text
Total Variation Distance
Jensen–Shannon divergence.
```

For discrete lag-frequency distributions.

---

# 30. Total Variation Distance

\[
TVD(P,Q)
=
\frac{1}{2}\sum_k |P_k-Q_k|.
\]

Range:

```text
0..1.
```

Interpretation:

```text
0 identical top1-lag distributions
1 disjoint distributions.
```

---

# 31. No top1 lag “importance” claim

Top1 frequency only captures:

```text
which lag had maximum single weight.
```

It ignores remaining probability mass.

Always interpret with full profile.

---

# 32. Within-layer head diversity matrix

For each:

```text
seed
layer
```

create square matrices:

```text
Pearson similarity
Spearman similarity
cosine similarity
JSD
L1 distance
expected-lag difference
normalized-entropy difference
recent-1h difference.
```

Diagonal:

```text
identity/zero as appropriate.
```

---

# 33. Matrix ordering

Rows/columns in:

```text
architectural head order
H1,H2,...,HH.
```

No reordering by similarity in the canonical matrix.

This prevents implicit head clustering.

---

# 34. Optional reordered matrix

A similarity-clustered visualization may be generated only as:

```text
SECONDARY_DIAGNOSTIC
```

if Phase55 explicitly performs a predeclared clustering.

Canonical machine outputs remain architectural order.

Preferred core:

```text
no clustering.
```

---

# 35. Head redundancy diagnostic

Phase55 may quantify **potential redundancy** using profile similarity.

But no single arbitrary cutoff like:

```text
cosine > 0.95 = redundant
```

should be invented after viewing results.

---

# 36. Continuous redundancy evidence

For each pair report:

```text
cosine similarity
JSD
L1 distance
expected-lag difference
entropy difference.
```

Then describe:

```text
more similar / more distinct
```

with actual values.

---

# 37. Layer diversity summary

For each seed/layer compute pairwise aggregate statistics:

```text
mean pairwise JSD
median pairwise JSD
max pairwise JSD
mean pairwise cosine
min pairwise cosine
mean pairwise L1
mean pairwise expected-lag absolute difference.
```

This yields a layer-level head-diversity summary.

---

# 38. No layer quality ranking

If Layer2 has higher pairwise JSD than Layer1:

Safe:

```text
Layer2 heads are more diverse in average last-query temporal profiles.
```

Unsafe:

```text
Layer2 is better.
```

---

# 39. Head-to-layer-mean deviation

For each head:

\[
D_{h,layermean}
=
JSD(
p_h,
p_{layermean}
).
\]

Also compute:

```text
L1 distance to layer head-mean profile
cosine similarity to layer head-mean profile.
```

This shows whether a head is:

```text
close to layer consensus
or
distinct from layer average.
```

---

# 40. No “outlier head” cutoff

Do not label a head an outlier using post-hoc threshold.

Report continuous deviation metrics.

---

# 41. Temporal centroid difference

For pair `(a,b)`:

\[
\Delta Lag_{mean}
=
|E[Lag]_a-E[Lag]_b|.
\]

Use head-level aggregate expected lag summary.

---

# 42. Distribution-level comparison across targets

Head mean profiles can hide target-to-target variability.

Therefore Phase55 also compares per-target metric distributions for selected metrics:

```text
normalized entropy
expected lag
recent 1h mass
Lag80.
```

---

# 43. Paired per-target structure

Within one seed/layer, heads are observed on the **same Test targets**.

Thus per-target metric differences can be computed pairwise:

\[
d_t
=
metric_{h_a,t}
-
metric_{h_b,t}.
\]

This is a paired descriptive comparison.

---

# 44. Required paired-difference summaries

For each pair and metric:

```text
N
mean difference
median difference
sample SD
p05
p25
p75
p95
fraction positive
fraction zero
fraction negative.
```

No iid inferential test required.

---

# 45. Why paired differences are useful

They answer:

> Does head A consistently allocate more recent mass than head B on the same Test targets, or is the difference target-dependent?

---

# 46. No paired t-test as core

Time-series targets are dependent.

Use descriptive paired-difference distribution.

No classical iid significance claim.

---

# 47. Optional sign consistency

For paired metric difference:

```text
fraction_positive
fraction_negative.
```

This quantifies directional consistency.

Do not call it statistical significance.

---

# 48. Head profile crossing

Two mean temporal profiles may cross across lag.

Phase55 may report:

```text
lag-bin mass differences
```

rather than one scalar “which is more recent” if patterns are complex.

---

# 49. Head behavior card

For every:

```text
seed/layer/head
```

create one behavior-summary row:

```text
mean normalized entropy
median normalized entropy
mean effective source count
median expected lag minutes
median lag SD minutes
median top1 weight
median top5 mass
median recent1h mass
median recent6h mass
median Lag50
median Lag80
median Lag90
top1 modal lag
top1 modal lag fraction
JSD to layer mean
L1 to layer mean
cosine to layer mean.
```

Architectural order only.

---

# 50. Head behavior card is not a leaderboard

No:

```text
score
rank
grade.
```

---

# 51. Optional descriptive head typology

Phase55 may derive a **rule-based descriptive typology** only if all thresholds are predeclared before viewing head results.

To avoid arbitrary post-hoc thresholds, default:

```text
TYPOLOGY = OFF.
```

This is the preferred core plan.

---

# 52. Why typology is OFF by default

Labels like:

```text
recent specialist
long-range head
diffuse head
focused head
```

require thresholds that can be arbitrary.

Better to report exact continuous metrics.

---

# 53. No unsupervised head clustering in core

Do not run:

```text
k-means
spectral clustering
hierarchical clustering
```

as the main method.

Potential future extension only.

---

# 54. Head pair summary table

Create one long row per:

```text
seed
layer
head_a
head_b.
```

Include:

```text
profile similarities
profile distances
expected-lag diff
entropy diff
recent-mass diffs
lag80 diff
top1-distribution distance.
```

---

# 55. Symmetry rule

For undirected pairwise similarities:

```text
store only head_a < head_b
```

in long pair table.

Square matrices are derived.

---

# 56. Difference sign convention

For signed metric differences:

\[
\Delta(A-B)
=
metric_A-metric_B.
\]

Head order:

```text
head_a < head_b.
```

Document clearly.

---

# 57. Absolute-difference fields

Also store:

```text
abs_delta_expected_lag
abs_delta_entropy
abs_delta_recent1h.
```

Useful for diversity magnitude independent of direction.

---

# 58. Profile-based similarity source

Use:

```text
mean temporal profile by lag
```

not single-case profiles.

Case-level head comparison remains illustrative only.

---

# 59. All-Test case independence caveat

Mean profiles are aggregated over temporally dependent Test targets.

No confidence interval or p-value claims without block-aware methods.

---

# 60. Empirical variability

Allowed:

```text
sample SD
quantiles
paired-difference distributions.
```

Descriptive only.

---

# 61. Heatmap visual integration

Phase53 raw attention heatmaps can be cross-referenced to illustrate heads identified as:

```text
quantitatively similar
quantitatively distinct
```

but case/image selection must be deterministic.

---

# 62. Deterministic illustrative pair selection

To show one “most similar” and one “most different” pair visually would itself be a ranking.

Phase55 may do this **only after** pairwise metrics are frozen and if clearly labeled:

```text
descriptive extremal pair by predeclared metric
```

However preferred main report:

```text
show full similarity matrices
```

instead of cherry-picking pairs.

---

# 63. Recommended report figures

Use:

```text
pairwise similarity heatmaps
per-head metric distributions
layer diversity summary
profile overlays
```

rather than selected pretty head pairs.

---

# 64. Canonical profile similarity metric for matrix headline

Use:

```text
Jensen–Shannon divergence
```

as the primary probability-profile divergence view.

Why:

```text
symmetric
bounded
well-suited to probability distributions.
```

But Pearson/cosine/L1 remain reported.

---

# 65. No JSD threshold classification

No:

```text
JSD < 0.02 = redundant
```

unless a future predeclared rule is introduced.

---

# 66. Pairwise JSD matrix

For each seed/layer:

```text
rows=heads
columns=heads
value=JSD(mean temporal profiles).
```

Canonical matrix order:

```text
H1...HH.
```

---

# 67. Pairwise cosine matrix

Same.

---

# 68. Pairwise expected-lag difference matrix

Value:

```text
absolute difference in median expected lag minutes
```

or mean.

Canonical project:

```text
median expected lag
```

because per-target lag distributions may be skewed.

---

# 69. Median vs mean lock

For head-level behavioral center:

```text
median
```

is primary for:

```text
expected lag
entropy
recent mass
coverage radii.
```

Mean remains secondary.

This is robust to skewed per-target distributions.

---

# 70. Layer diversity uses mean pairwise distance

Layer summary primary:

```text
mean pairwise JSD.
```

Secondary:

```text
median pairwise JSD
max pairwise JSD.
```

No ranking across layers as quality.

---

# 71. Pairwise profile centroid shift

Optional:

```text
Wasserstein-1 distance over lag axis
```

is a useful temporal-profile distance because it accounts for how far probability mass moves in time.

Phase55 may include it as a required advanced metric.

---

# 72. Wasserstein-1 distance

For discrete lag distributions:

\[
W_1(P,Q)
\]

on support:

```text
lag_minutes.
```

Units:

```text
minutes.
```

Interpretation:

> Average temporal distance required to move one profile’s attention mass into the other profile.

This is highly interpretable for temporal attention.

---

# 73. Why Wasserstein is valuable

Two profiles shifted by one hour may have:

```text
similar shape
but temporal displacement.
```

JSD/L1 detect difference but not temporal distance.

Wasserstein adds lag geometry.

---

# 74. Wasserstein computation

Use identical lag support:

```text
10,20,...,L×10 minutes
```

with probability weights from mean profiles.

No normalization beyond verified profile sums.

---

# 75. Pairwise metrics required final set

Canonical:

```text
Pearson
Spearman
Cosine
JSD
L1
L2
Wasserstein minutes
absolute median expected-lag difference
absolute median normalized-entropy difference
absolute median recent1h-mass difference
top1 lag TVD
top1 lag JSD.
```

---

# 76. Metric redundancy caveat

Some pairwise measures are correlated.

Do not combine them into one arbitrary weighted score.

---

# 77. No “head diversity score” weighted composite

Forbidden:

```text
0.4*JSD + 0.3*entropy_diff + ...
```

No theoretical basis.

Use transparent separate metrics.

---

# 78. Layer diversity summary metric family

For each seed/layer:

```text
mean pairwise JSD
mean pairwise L1
mean pairwise Wasserstein minutes
mean pairwise abs expected-lag diff
mean pairwise top1 TVD
```

This yields multi-dimensional diversity evidence.

---

# 79. No single-layer diversity winner

No rank.

---

# 80. Head redundancy graph

Optional derived visualization:

```text
nodes=heads
edge width=similarity
```

but requires choosing a threshold to show edges.

Preferred core:

```text
omit graph
```

to avoid threshold arbitrariness.

---

# 81. Full pairwise matrices are sufficient

Use exact values.

---

# 82. Head metric correlation matrix

Optional:

Within one seed/layer/head metrics across Test targets, calculate:

```text
correlation between entropy, expected lag, recent mass, top5 mass, Lag80.
```

This describes metric relationships, not head relationships.

Not required for core Phase55.

---

# 83. Avoid scope inflation

Keep focus:

```text
compare heads
not analyze every attention metric relationship.
```

---

# 84. Cross-layer same-head-index comparison

Do not treat:

```text
Layer1 Head1
Layer2 Head1
```

as same head evolving through depth.

They are different parameter sets.

Layer comparison can use:

```text
layer head-mean profiles
layer diversity summaries.
```

---

# 85. Same-index across layers is not semantic identity

Hard caveat.

---

# 86. Per-head figure ordering

All figures:

```text
Head1
Head2
...
HeadH
```

architectural order.

No sorting by metric.

---

# 87. Pairwise matrix color scales

Within same metric across:

```text
seed/layer matrices
```

use same color range where meaningful.

Examples:

```text
JSD: 0..ln(2)
Cosine: 0..1
TVD: 0..1
```

For Wasserstein/lag differences, use common max across all seed/layer matrices for that metric.

Record scale.

---

# 88. Diverging scale for signed differences

If plotting:

```text
recent1h head_a-head_b
```

a zero-centered diverging scale is valid.

But pairwise matrices are usually symmetric/absolute and can use sequential scale.

---

# 89. JSD matrix diagonal

Exactly:

```text
0.
```

within numerical tolerance.

Hard audit.

---

# 90. Cosine matrix diagonal

Exactly:

```text
1.
```

within tolerance.

---

# 91. Pairwise symmetry audit

For matrices:

```text
JSD
L1
L2
Wasserstein
TVD
```

must satisfy:

```text
M[i,j] ≈ M[j,i].
```

Hard.

---

# 92. Correlation symmetry

Same.

---

# 93. Pairwise distance nonnegativity

Hard:

```text
distance >= 0.
```

---

# 94. Wasserstein units audit

Ensure:

```text
minutes
```

not raw position indices.

---

# 95. Top1 lag distribution completeness

For each seed/layer/head:

```text
sum lag frequencies = 1.
```

Before TVD/JSD.

---

# 96. Head pair paired-difference source

Use:

```text
last_query_metrics_long.csv
```

matched by:

```text
seed
target_id
layer.
```

Hard one-to-one head values per target.

---

# 97. Pairwise target alignment

For any head pair:

```text
same exact target IDs
same N.
```

No row dropping.

If metric unavailable for some target:

```text
investigate source
```

rather than silently intersecting subsets.

---

# 98. Truncated recent-window metrics

If lookback truncates 12h/24h:

```text
do not compare those named windows as full 12h/24h semantics.
```

Use only effective supported windows or mark truncated.

---

# 99. Preferred recency metric for head comparison

Always available:

```text
recent_1h
recent_6h
```

only if final `L>=6/36` respectively.

Since candidate lookbacks are at least 36, 1h and 6h should normally be supported, but runtime contract is authoritative.

---

# 100. Head comparison on unsupported metric

Set:

```text
NOT_APPLICABLE.
```

No artificial zeros.

---

# 101. Report-ready head behavior table

For each:

```text
seed/layer
```

generate a table with heads as rows and columns:

```text
median normalized entropy
median expected lag
median recent1h mass
median recent6h mass
median Lag80
modal top1 lag
JSD to layer mean.
```

This is compact and interpretable.

---

# 102. No best/worst labels in table

Only values.

---

# 103. Head pair table can be large

Store full machine-readable:

```text
head_pair_comparison_long.csv.
```

Report only summary matrices/selected metrics.

---

# 104. Output directory

```text
artifacts/
└── head_comparison/
    ├── head_comparison_manifest.json
    ├── head_comparison_contract.json
    ├── phase55_preflight_audit.csv
    ├── head_comparison_source_verification.csv
    ├── head_profile_integrity_audit.csv
    ├── head_target_alignment_audit.csv
    ├── head_pair_comparison_long.csv
    ├── head_pair_profile_similarity.csv
    ├── head_pair_metric_difference.csv
    ├── head_pair_paired_difference_summary.csv
    ├── head_pair_top1_distribution_distance.csv
    ├── head_pair_wasserstein_distance.csv
    ├── head_behavior_summary.csv
    ├── head_to_layer_mean_distance.csv
    ├── layer_head_diversity_summary.csv
    ├── head_similarity_matrix_jsd.csv
    ├── head_similarity_matrix_cosine.csv
    ├── head_similarity_matrix_pearson.csv
    ├── head_similarity_matrix_spearman.csv
    ├── head_distance_matrix_l1.csv
    ├── head_distance_matrix_wasserstein.csv
    ├── head_expected_lag_difference_matrix.csv
    ├── head_recent1h_difference_matrix.csv
    ├── head_top1_tvd_matrix.csv
    ├── head_comparison_findings.csv
    ├── head_comparison_tests.csv
    ├── head_comparison_discrepancies.json
    ├── phase56_error_conditioned_attention_handoff.json
    ├── phase57_seed_stability_head_context_handoff.json
    ├── head_comparison_summary.json
    ├── head_comparison_report.md
    ├── figures/
    │   ├── HEAD_55_01_jsd_matrices.png
    │   ├── HEAD_55_02_cosine_matrices.png
    │   ├── HEAD_55_03_wasserstein_matrices.png
    │   ├── HEAD_55_04_expected_lag_difference_matrices.png
    │   ├── HEAD_55_05_top1_tvd_matrices.png
    │   ├── HEAD_55_06_entropy_by_head.png
    │   ├── HEAD_55_07_expected_lag_by_head.png
    │   ├── HEAD_55_08_recent1h_mass_by_head.png
    │   ├── HEAD_55_09_recent6h_mass_by_head.png
    │   ├── HEAD_55_10_lag80_by_head.png
    │   ├── HEAD_55_11_head_to_layer_mean_jsd.png
    │   ├── HEAD_55_12_layer_head_diversity_summary.png
    │   ├── HEAD_55_13_mean_temporal_profiles_by_head.png
    │   └── HEAD_55_14_paired_difference_distributions.png
    ├── README_HEAD_COMPARISON.md
    └── phase_55_signoff.json
```

---

# 105. Required outputs

```text
O55.1  Analysis manifest
O55.2  Analysis contract
O55.3  Preflight audit
O55.4  Source verification
O55.5  Profile integrity audit
O55.6  Target alignment audit
O55.7  Full head-pair comparison table
O55.8  Profile-similarity table
O55.9  Metric-difference table
O55.10 Paired per-target difference summaries
O55.11 Top1-distribution distance table
O55.12 Wasserstein-distance table
O55.13 Head behavior summary
O55.14 Head-to-layer-mean distance table
O55.15 Layer diversity summary
O55.16 JSD matrices
O55.17 Cosine matrices
O55.18 Pearson/Spearman matrices
O55.19 L1/Wasserstein matrices
O55.20 Expected-lag difference matrices
O55.21 Recent1h difference matrices
O55.22 Top1 TVD matrices
O55.23 Core figures
O55.24 Findings
O55.25 Phase56 handoff
O55.26 Phase57 context handoff
O55.27 Tests
O55.28 Discrepancies
O55.29 Summary JSON
O55.30 Human-readable report
O55.31 README
O55.32 Sign-off
```

---

# 106. Analysis manifest

`head_comparison_manifest.json`:

```text
phase=55
version=HEAD_COMPARISON-v1
source_phase54_version
source_phase52_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
lookback_steps
num_layers
num_heads
primary_comparison_scope=WITHIN_SEED_WITHIN_LAYER_ACROSS_HEADS
pairwise_metrics=[
  PEARSON,
  SPEARMAN,
  COSINE,
  JSD,
  L1,
  L2,
  WASSERSTEIN_MINUTES,
  ABS_EXPECTED_LAG_DIFF,
  ABS_ENTROPY_DIFF,
  ABS_RECENT1H_DIFF,
  TOP1_TVD,
  TOP1_JSD
]
best_head_selection=false
head_pruning=false
error_conditioning=false
cross_seed_head_matching=false
status
created_at
```

---

# 107. Analysis contract

`head_comparison_contract.json` must freeze:

```text
Primary comparison:
heads within same seed/layer.

Profiles:
Phase54 mean temporal profiles.

Head order:
architectural.

Pairwise profile metrics:
Pearson
Spearman
Cosine
JSD
L1
L2
Wasserstein minutes.

Pairwise behavioral differences:
median expected lag
median normalized entropy
median recent1h
median recent6h
median Lag80.

Top1 distribution:
TVD
JSD.

Paired target-level differences:
entropy
expected lag
recent1h
Lag80.

Layer diversity:
mean/median/max pairwise JSD
mean L1
mean Wasserstein
mean abs expected-lag diff.

No:
best head
weighted diversity score
head pruning
head ablation
head clustering core
error conditioning
cross-seed semantic head assumption.
```

---

# 108. Preflight audit

`phase55_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Required:

```text
Phase54 approved
phase55_ready=true
all Phase54 tables exist
same seed list
same layers
same heads
same target population
mean profiles sum≈1
top1 frequency distributions sum≈1
no missing head
no missing target
raw fallback refs available
head comparison contract frozen before results.
```

---

# 109. Source verification schema

`head_comparison_source_verification.csv`:

```text
source_id
path
sha256_if_available
row_count
seed_count
layer_count
head_count
target_count_if_applicable
status
```

---

# 110. Profile integrity audit

`head_profile_integrity_audit.csv`:

```text
seed
layer_idx0
head_idx0
lag_count
min_weight
max_weight
profile_sum
nonnegative
sum_pass
status
```

---

# 111. Target alignment audit

`head_target_alignment_audit.csv`:

```text
seed
layer_idx0
metric
head_a
head_b
N_head_a
N_head_b
matched_target_count
missing_a
missing_b
exact_alignment
status
```

Expected:

```text
missing_a=0
missing_b=0.
```

---

# 112. Head-pair comparison schema

`head_pair_comparison_long.csv`:

```text
seed
layer_idx0
head_a_idx0
head_a_display
head_b_idx0
head_b_display
pearson_profile
spearman_profile
cosine_profile
jsd_profile
l1_profile
l2_profile
wasserstein_minutes
median_expected_lag_a
median_expected_lag_b
delta_expected_lag_a_minus_b
abs_delta_expected_lag
median_norm_entropy_a
median_norm_entropy_b
delta_entropy_a_minus_b
abs_delta_entropy
median_recent1h_a
median_recent1h_b
delta_recent1h_a_minus_b
abs_delta_recent1h
median_recent6h_a
median_recent6h_b
delta_recent6h_a_minus_b
median_lag80_a
median_lag80_b
delta_lag80_a_minus_b
top1_tvd
top1_jsd
status
```

Store only:

```text
head_a_idx0 < head_b_idx0.
```

---

# 113. Profile similarity schema

`head_pair_profile_similarity.csv`:

```text
seed
layer_idx0
head_a
head_b
pearson
spearman
cosine
jsd
l1
l2
wasserstein_minutes
status
```

---

# 114. Metric difference schema

`head_pair_metric_difference.csv`:

```text
seed
layer_idx0
head_a
head_b
metric
center_type=MEDIAN
value_a
value_b
delta_a_minus_b
abs_delta
status
```

Metrics:

```text
normalized_entropy
expected_lag_minutes
recent_1h_mass
recent_6h_mass
lag80_minutes
top5_mass.
```

---

# 115. Paired target-difference schema

`head_pair_paired_difference_summary.csv`:

```text
seed
layer_idx0
head_a
head_b
metric
N
mean_difference
median_difference
sample_sd_difference
p05
p25
p75
p95
fraction_positive
fraction_zero
fraction_negative
status
```

Required metrics:

```text
normalized_entropy
expected_lag_minutes
recent_1h_mass
lag80_minutes.
```

---

# 116. Top1 distribution distance schema

`head_pair_top1_distribution_distance.csv`:

```text
seed
layer_idx0
head_a
head_b
tvd
jsd
lag_support_count
status
```

---

# 117. Wasserstein schema

`head_pair_wasserstein_distance.csv`:

```text
seed
layer_idx0
head_a
head_b
wasserstein_minutes
lag_min_minutes
lag_max_minutes
profile_sum_a
profile_sum_b
status
```

---

# 118. Head behavior summary schema

`head_behavior_summary.csv`:

```text
seed
layer_idx0
head_idx0
head_display
median_normalized_entropy
mean_normalized_entropy
median_effective_source_count
median_expected_lag_minutes
median_lag_sd_minutes
median_top1_weight
median_top5_mass
median_recent1h_mass
median_recent6h_mass
median_recent12h_mass_if_supported
median_recent24h_mass_if_supported
median_lag50_minutes
median_lag80_minutes
median_lag90_minutes
modal_top1_lag_minutes
modal_top1_lag_fraction
jsd_to_layer_mean
l1_to_layer_mean
cosine_to_layer_mean
status
```

No rank.

---

# 119. Head-to-layer-mean schema

`head_to_layer_mean_distance.csv`:

```text
seed
layer_idx0
head_idx0
jsd_to_layer_head_mean_profile
l1_to_layer_head_mean_profile
l2_to_layer_head_mean_profile
cosine_to_layer_head_mean_profile
wasserstein_to_layer_head_mean_minutes
status
```

---

# 120. Layer diversity summary schema

`layer_head_diversity_summary.csv`:

```text
seed
layer_idx0
head_count
pair_count
mean_pairwise_jsd
median_pairwise_jsd
max_pairwise_jsd
mean_pairwise_l1
mean_pairwise_l2
mean_pairwise_wasserstein_minutes
mean_pairwise_abs_expected_lag_diff_minutes
mean_pairwise_top1_tvd
min_pairwise_cosine
mean_pairwise_cosine
status
```

No overall weighted score.

---

# 121. Matrix schemas

Each matrix CSV:

```text
seed
layer_idx0
row_head
column_head
value
metric
status
```

or pivot format with architectural head columns.

Long form is preferred for audit; wide pivot can be report artifact.

---

# 122. JSD matrix audit

For every seed/layer:

```text
diagonal≈0
symmetric
nonnegative
<=ln(2)+tolerance.
```

---

# 123. Cosine matrix audit

```text
diagonal≈1
symmetric
range 0..1+tolerance.
```

Because profiles are nonnegative.

---

# 124. Pearson/Spearman range audit

```text
-1..1.
```

---

# 125. Wasserstein matrix audit

```text
diagonal=0
symmetric
units=minutes
nonnegative.
```

---

# 126. Expected-lag difference matrix audit

```text
diagonal=0
symmetric
nonnegative.
```

Use absolute difference matrix.

---

# 127. Recent1h difference matrix

For a symmetric magnitude matrix use:

```text
absolute median recent1h difference.
```

If signed matrix is desired, store separately.

Canonical matrix:

```text
absolute difference.
```

---

# 128. Top1 TVD matrix audit

```text
diagonal=0
symmetric
0..1.
```

---

# 129. Figure HEAD_55_01 — JSD matrices

One matrix per:

```text
seed × layer.
```

Use consistent scale:

```text
0..ln(2).
```

Architectural order.

---

# 130. Figure HEAD_55_02 — Cosine matrices

Scale:

```text
0..1.
```

Do not reorder heads.

---

# 131. Figure HEAD_55_03 — Wasserstein matrices

Use one common max scale across all:

```text
seed × layer
```

matrices.

Units:

```text
minutes.
```

---

# 132. Figure HEAD_55_04 — Expected-lag difference matrices

Common scale.

Units:

```text
minutes.
```

---

# 133. Figure HEAD_55_05 — Top1 TVD matrices

Scale:

```text
0..1.
```

---

# 134. Figure HEAD_55_06 — Entropy by head

Show empirical distributions or summary bars:

```text
normalized entropy
```

by architectural head order.

No ranking labels.

---

# 135. Figure HEAD_55_07 — Expected lag by head

Use:

```text
minutes/hours
```

with architectural order.

---

# 136. Figure HEAD_55_08 — Recent 1h mass by head

No recency-quality judgment.

---

# 137. Figure HEAD_55_09 — Recent 6h mass by head

Same.

---

# 138. Figure HEAD_55_10 — Lag80 by head

Shows history radius needed to accumulate 80% mass.

---

# 139. Figure HEAD_55_11 — Distance to layer mean

Use:

```text
JSD to layer head-mean profile.
```

No outlier threshold.

---

# 140. Figure HEAD_55_12 — Layer diversity summary

Plot transparent separate measures, e.g.:

```text
mean pairwise JSD
mean Wasserstein
```

in separate panels/figures.

Do not combine different units on one arbitrary normalized score.

---

# 141. Figure HEAD_55_13 — Mean temporal profiles

Per seed/layer:

```text
all heads over lag.
```

This can reuse Phase54 profiles but Phase55 contextualizes comparison.

---

# 142. Figure HEAD_55_14 — Paired-difference distributions

For selected core metric:

```text
recent1h mass
expected lag
```

show all head pairs in architectural pair order.

No “winning pair”.

---

# 143. Head pair order

Canonical pair list:

```text
H1-H2
H1-H3
...
H1-HH
H2-H3
...
```

lexicographic by head indices.

---

# 144. Findings codes

Possible:

```text
HEAD_PROFILES_HIGHLY_SIMILAR_DESCRIPTIVE
HEAD_PROFILES_DIVERSE_DESCRIPTIVE
HEAD_PAIR_LOW_JSD
HEAD_PAIR_HIGH_JSD
HEAD_PAIR_SMALL_WASSERSTEIN
HEAD_PAIR_LARGE_WASSERSTEIN
HEAD_RECENCY_ALLOCATION_DIFFERS
HEAD_CONCENTRATION_DIFFERS
HEAD_EXPECTED_LAG_DIFFERS
HEAD_TOP1_LAG_DISTRIBUTIONS_DIFFERS
HEADS_CLOSE_TO_LAYER_MEAN
HEADS_DEVIATE_FROM_LAYER_MEAN
LAYER_HEAD_DIVERSITY_HIGHER_DESCRIPTIVE
LAYER_HEAD_DIVERSITY_LOWER_DESCRIPTIVE
PAIRWISE_DIFFERENCES_TARGET_DEPENDENT
PAIRWISE_DIFFERENCES_DIRECTIONALLY_CONSISTENT
NO_CLEAR_HEAD_DIVERSITY
NO_BEST_HEAD_SELECTED
NO_HEAD_PRUNING
NO_ERROR_CONDITIONING
NO_CROSS_SEED_HEAD_MATCHING
ATTENTION_TEMPORAL_NOT_FEATURE_IMPORTANCE
READY_FOR_ERROR_CONDITIONED_ATTENTION
READY_FOR_SEED_STABILITY_CONTEXT
```

Avoid “HIGH/LOW” wording without actual metric values in report.

---

# 145. Safe findings language

Safe:

> Within seed 42, Layer 1 heads showed non-identical mean temporal profiles, with pairwise JSD ranging from X to Y and Wasserstein distances ranging from A to B minutes.

Safe:

> Head 2 allocated a larger median fraction of last-query attention to the most recent hour than Head 4 on the same Test targets.

Safe:

> Layer 2 displayed greater average pairwise profile divergence than Layer 1 for this seed, indicating stronger internal head diversity in temporal allocation.

Unsafe:

> Head 2 is more useful than Head 4.

Unsafe:

> Head 3 can be removed because it is redundant.

---

# 146. Redundancy wording

Use:

```text
near-similar temporal attention behavior
potential redundancy in last-query allocation
```

not:

```text
parameter redundancy proven
head unnecessary.
```

Attention profile similarity does not prove identical contribution because:

```text
Q/K/V projections differ
value content differs
output projection differs.
```

---

# 147. Value-path caveat

Two heads with nearly identical attention weights can still output different values because:

```text
their value projections are different.
```

Therefore head redundancy from attention weights is only:

```text
attention-allocation redundancy
```

not functional redundancy.

---

# 148. Output-projection caveat

Multi-head outputs are concatenated/projected.

No individual head’s attention map directly equals its contribution to final prediction.

---

# 149. No predictive head attribution

No.

---

# 150. Phase56 handoff purpose

Phase56 will ask:

> Do these head behaviors systematically change under high error, under/overprediction, rapid change, extreme-high regimes, etc.?

Phase55 hands off head metric definitions and within-layer comparison context.

---

# 151. Phase56 handoff schema

`phase56_error_conditioned_attention_handoff.json`:

```text
source_phase55_version
source_phase54_version
source_phase52_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
head_behavior_summary
head_pair_comparison_long
layer_diversity_summary
last_query_metrics_long_ref
raw_last_query_refs
Phase49_residual_refs
Phase50_regime_refs
Phase51_worst_case_refs
head_order=ARCHITECTURAL
best_head_selected=false
head_pruning=false
error_conditioning_performed_in_phase55=false
ready_for_phase56=true
```

---

# 152. Phase57 context handoff purpose

Phase57 will perform:

```text
cross-seed attention stability
head matching
permutation-aware comparison.
```

Phase55 provides within-seed head diversity, useful for determining whether multiple heads are easy/hard to match.

---

# 153. Phase57 context handoff schema

`phase57_seed_stability_head_context_handoff.json`:

```text
source_phase55_version
within_seed_pairwise_jsd
within_seed_pairwise_wasserstein
head_to_layer_mean_distance
layer_head_diversity_summary
per_seed_head_profiles_ref
head_count
layer_count
same_index_semantic_alignment_assumed=false
head_matching_performed=false
ready_for_phase57_context=true
```

---

# 154. Discrepancy taxonomy

`head_comparison_discrepancies.json`:

```text
PHASE54_NOT_APPROVED
PHASE55_HANDOFF_NOT_READY
SOURCE_TABLE_MISSING
SOURCE_CHECKSUM_MISMATCH
SEED_MISSING
LAYER_MISSING
HEAD_MISSING
TARGET_ALIGNMENT_MISMATCH
PROFILE_SUM_MISMATCH
TOP1_DISTRIBUTION_SUM_MISMATCH
PAIR_COUNT_MISMATCH
HEAD_PAIR_DUPLICATE
HEAD_PAIR_MISSING
JSD_NEGATIVE
JSD_ABOVE_LN2
JSD_ASYMMETRIC
COSINE_OUT_OF_RANGE
COSINE_DIAGONAL_MISMATCH
PEARSON_OUT_OF_RANGE
SPEARMAN_OUT_OF_RANGE
L1_NEGATIVE
L2_NEGATIVE
WASSERSTEIN_NEGATIVE
WASSERSTEIN_UNITS_NOT_MINUTES
DISTANCE_MATRIX_ASYMMETRIC
TVD_OUT_OF_RANGE
TOP1_JSD_INVALID
SIGNED_DELTA_CONVENTION_DRIFT
HEAD_ORDER_REORDERED_BY_METRIC
HEADS_CLUSTERED_WITHOUT_PREDECLARATION
REDUNDANCY_THRESHOLD_INVENTED_POST_HOC
WEIGHTED_HEAD_DIVERSITY_SCORE_CREATED
BEST_HEAD_SELECTED
HEAD_PRUNING_ATTEMPT
HEAD_ABLATION_ATTEMPT
ERROR_CONDITIONING_SCOPE_CREEP
SEED_STABILITY_SCOPE_CREEP
CROSS_SEED_HEAD_MATCHING_SCOPE_CREEP
SAME_INDEX_HEAD_SEMANTIC_EQUIVALENCE_ASSUMED
LAYER_SAME_HEAD_IDENTITY_ASSUMED
ATTENTION_REDUNDANCY_MISLABELED_FUNCTIONAL_REDUNDANCY
FEATURE_IMPORTANCE_CLAIM
CAUSAL_CLAIM
NEW_ATTENTION_EXTRACTION_ATTEMPT
NEW_TEST_INFERENCE_ATTEMPT
OTHER
```

---

# 155. Status model

## PASS

```text
all Phase54 sources verified
all head pairs complete
profile similarities complete
Wasserstein/TVD/JSD valid
paired target-difference summaries complete
head behavior cards complete
head-to-layer-mean distances complete
layer diversity summaries complete
architectural ordering preserved
no best head/pruning
Phase56 handoff ready.
```

## PASS_WITH_WARNING

Possible:

```text
some heads have near-identical attention profiles
some heads highly divergent
lookback truncates longer-window metrics
pairwise metrics strongly redundant
MEAN pooling caveat remains
large artifact volume.
```

## FAIL

Examples:

```text
missing head pairs
invalid profile sums
head order re-sorted by score
post-hoc redundancy threshold
best-head selection
cross-seed semantic matching performed
error conditioning performed.
```

---

# 156. Execution sequence

```text
1. Verify Phase54 signoff/handoff.
2. Verify profiles/metrics/top1 distributions.
3. Freeze Phase55 comparison contract.
4. Build canonical architectural head-pair list per seed/layer.
5. Verify pair count H(H-1)/2.
6. Compute pairwise profile Pearson/Spearman/Cosine.
7. Compute pairwise JSD/L1/L2.
8. Compute temporal Wasserstein distance in minutes.
9. Compute head-level median metric differences.
10. Compute paired per-target metric difference summaries.
11. Compute top1 lag TVD/JSD.
12. Compute head-to-layer-mean profile distances.
13. Compute head behavior cards.
14. Compute layer-level diversity summaries.
15. Build square similarity/distance matrices.
16. Run symmetry/diagonal/range audits.
17. Generate comparison figures in architectural order.
18. Write findings with no head winner.
19. Write Phase56 handoff.
20. Write Phase57 context handoff.
21. Run tests/discrepancy audit.
22. Write summary/report/README.
23. Sign off.
```

---

# 157. Recommended pseudocode

```text
p54 = load_phase54_signoff()
assert p54.overall_status in {"PASS","PASS_WITH_WARNING"}

profiles = load_last_query_profile_by_lag()
metrics = load_last_query_metrics_long()
top1_freq = load_top1_lag_frequency()
layer_mean_profiles = load_layer_head_mean_profiles()

verify_sources()
freeze_head_comparison_contract()

pair_rows = []
paired_diff_rows = []

for seed in [42,123,2026]:

    for layer in range(N_layers):

        heads = architectural_heads(layer)

        assert len(heads) == N_heads

        expected_pair_count = N_heads * (N_heads - 1) // 2

        pairs = [
            (a,b)
            for a in heads
            for b in heads
            if a < b
        ]

        assert len(pairs) == expected_pair_count

        for head_a, head_b in pairs:

            p_a = get_mean_profile(
                profiles,
                seed,
                layer,
                head_a
            )

            p_b = get_mean_profile(
                profiles,
                seed,
                layer,
                head_b
            )

            verify_probability_profile(p_a)
            verify_probability_profile(p_b)

            pearson = corr_pearson(p_a, p_b)
            spearman = corr_spearman(p_a, p_b)
            cosine = cosine_similarity(p_a, p_b)

            jsd = jensen_shannon_divergence(
                p_a,
                p_b,
                log_base="natural"
            )

            l1 = sum(abs(p_a - p_b))
            l2 = sqrt(sum((p_a - p_b)**2))

            wasserstein_min = wasserstein_1d(
                support=lag_minutes,
                weights_a=p_a,
                weights_b=p_b
            )

            metric_a = summarize_head_medians(
                metrics,
                seed,
                layer,
                head_a
            )

            metric_b = summarize_head_medians(
                metrics,
                seed,
                layer,
                head_b
            )

            top1_a = get_top1_distribution(
                top1_freq,
                seed,
                layer,
                head_a
            )

            top1_b = get_top1_distribution(
                top1_freq,
                seed,
                layer,
                head_b
            )

            tvd = 0.5 * sum(abs(top1_a - top1_b))
            top1_jsd = jensen_shannon(top1_a, top1_b)

            pair_rows.append(...)

            for metric_name in [
                "normalized_entropy",
                "expected_lag_minutes",
                "recent_1h_mass",
                "lag80_minutes"
            ]:

                paired_a = target_aligned_values(
                    metrics,
                    seed,
                    layer,
                    head_a,
                    metric_name
                )

                paired_b = target_aligned_values(
                    metrics,
                    seed,
                    layer,
                    head_b,
                    metric_name
                )

                assert exact_same_target_ids(
                    paired_a,
                    paired_b
                )

                d = paired_a - paired_b

                paired_diff_rows.append(
                    summarize_paired_difference(
                        d,
                        seed,
                        layer,
                        head_a,
                        head_b,
                        metric_name
                    )
                )

        # head-to-layer-mean
        p_layer = get_layer_head_mean_profile(
            layer_mean_profiles,
            seed,
            layer
        )

        for head in heads:
            p_h = get_mean_profile(...)
            compute_distance_to_layer_mean(
                p_h,
                p_layer
            )

        compute_layer_diversity_summary(
            pair_rows_for_seed_layer
        )

build_behavior_summary()

build_pairwise_matrices(
    metrics=[
        "JSD",
        "COSINE",
        "PEARSON",
        "SPEARMAN",
        "L1",
        "WASSERSTEIN",
        "ABS_EXPECTED_LAG_DIFF",
        "ABS_RECENT1H_DIFF",
        "TOP1_TVD"
    ],
    head_order="ARCHITECTURAL"
)

verify_matrix_symmetry_and_diagonals()
generate_figures()

assert no_best_head_selection
assert no_head_pruning
assert no_error_conditioning
assert no_cross_seed_head_matching

write_phase56_handoff()
write_phase57_context_handoff()

signoff_phase55()
```

---

# 158. Preflight acceptance checklist

```text
[ ] Phase54 PASS/PASS_WITH_WARNING.
[ ] phase55_ready=true.
[ ] Per-head metric summary exists.
[ ] Per-target metrics exist.
[ ] Mean profile table exists.
[ ] Layer head-mean profile exists.
[ ] Top1 lag frequency exists.
[ ] Same seeds.
[ ] Same layers.
[ ] Same heads.
[ ] Same target population.
[ ] All profiles sum≈1.
[ ] Top1 distributions sum≈1.
[ ] Contract frozen before pairwise results.
```

---

# 159. Pair construction acceptance checklist

```text
[ ] Heads kept in architectural order.
[ ] Only within-seed within-layer pairs.
[ ] head_a < head_b.
[ ] Pair count = H(H-1)/2 per seed/layer.
[ ] No duplicate pairs.
[ ] No missing pairs.
[ ] No cross-layer identity assumptions.
[ ] No cross-seed semantic matching.
```

---

# 160. Profile similarity acceptance checklist

```text
[ ] Pearson computed.
[ ] Spearman computed.
[ ] Cosine computed.
[ ] JSD computed with natural log.
[ ] JSD within [0,ln2].
[ ] L1 computed.
[ ] L2 computed.
[ ] Wasserstein computed in minutes.
[ ] No profile renormalization hiding integrity failure.
[ ] All metrics use exact same lag support.
```

---

# 161. Matrix acceptance checklist

```text
[ ] JSD diagonal=0.
[ ] JSD symmetric.
[ ] Cosine diagonal=1.
[ ] Cosine symmetric.
[ ] Pearson diagonal=1.
[ ] Spearman diagonal=1.
[ ] L1 diagonal=0.
[ ] Wasserstein diagonal=0.
[ ] TVD diagonal=0.
[ ] All distances nonnegative.
[ ] Architectural head order preserved.
[ ] No clustering/reordering in canonical matrices.
```

---

# 162. Metric-difference acceptance checklist

```text
[ ] Median expected lag differences.
[ ] Median entropy differences.
[ ] Median recent1h differences.
[ ] Median recent6h differences.
[ ] Median Lag80 differences.
[ ] Delta convention A-B documented.
[ ] Absolute delta fields available.
[ ] Unsupported truncated metrics marked N/A.
```

---

# 163. Paired-target acceptance checklist

```text
[ ] Exact same target IDs across pair.
[ ] N unchanged.
[ ] Mean difference.
[ ] Median difference.
[ ] SD.
[ ] p05/p25/p75/p95.
[ ] Fraction positive/zero/negative.
[ ] No iid significance test.
[ ] No row drops.
```

---

# 164. Top1-distribution acceptance checklist

```text
[ ] Lag support identical.
[ ] Frequency sum=1 per head.
[ ] TVD in 0..1.
[ ] JSD valid.
[ ] Frozen newest-tie rule inherited.
[ ] No top1-only interpretation replacing full profile.
```

---

# 165. Head behavior acceptance checklist

```text
[ ] Every head has behavior row.
[ ] Architectural order.
[ ] No score.
[ ] No rank.
[ ] JSD to layer mean.
[ ] L1 to layer mean.
[ ] Cosine to layer mean.
[ ] No outlier threshold.
```

---

# 166. Layer diversity acceptance checklist

```text
[ ] Pair count correct.
[ ] Mean pairwise JSD.
[ ] Median/max JSD.
[ ] Mean L1.
[ ] Mean Wasserstein.
[ ] Mean abs expected-lag difference.
[ ] Mean top1 TVD.
[ ] No weighted composite score.
[ ] No layer “winner”.
```

---

# 167. Scope acceptance checklist

```text
[ ] No best head.
[ ] No head pruning.
[ ] No head ablation.
[ ] No model retraining.
[ ] No Test metric recomputation.
[ ] No error conditioning.
[ ] No regime conditioning.
[ ] No worst-case statistical comparison.
[ ] No cross-seed head matching.
[ ] No same-index semantic assumption across seeds.
[ ] No same-index semantic assumption across layers.
[ ] No feature importance claim.
[ ] No causal claim.
```

---

# 168. Provenance acceptance checklist

```text
[ ] Source Phase54 version stored.
[ ] Raw Phase52 refs preserved.
[ ] Final lock SHA stored.
[ ] Test population SHA stored.
[ ] Head/layer counts stored.
[ ] Pairwise outputs trace to source profiles.
[ ] Phase56 handoff references exact outputs.
[ ] Phase57 context handoff references exact outputs.
```

---

# 169. Acceptance criteria

Phase55 PASS only when:

```text
All Phase54 head metrics, temporal profiles and top1-lag distributions are verified and aligned to the same Held-Out Test population.

Primary head comparison is restricted to heads within the same seed and same encoder layer.

Every within-layer head pair is generated exactly once, with expected pair count H(H-1)/2.

Pairwise temporal-profile comparison includes Pearson, Spearman, cosine similarity, Jensen–Shannon divergence, L1/L2 distance and temporal Wasserstein distance in minutes.

All profile-comparison metrics use exactly the same frozen lag support and verified probability profiles.

Top1-lag distributions are compared using TVD and JSD while preserving the Phase52 deterministic tie rule.

Behavioral differences in entropy, expected lag, recent-history mass and coverage radius are summarized both at the head level and through paired same-target differences.

Paired target-level comparisons use identical target IDs and report descriptive difference distributions rather than naive iid significance tests.

Every head receives a transparent behavior summary without a composite score or ranking.

Head-to-layer-mean distances are computed without declaring outlier heads.

Layer-level head diversity is summarized using transparent pairwise metrics rather than a weighted diversity score.

Canonical similarity/distance matrices preserve architectural head order and satisfy symmetry/diagonal/range audits.

Potentially similar heads are described only as attention-allocation redundancy, not proven functional redundancy.

No best head is selected, no head is pruned/ablated, no model is retrained and no final Test performance is recomputed.

No error-conditioned analysis is performed; Phase56 receives the frozen head behavior metrics.

No cross-seed head matching or seed-stability conclusion is performed; Phase57 receives the within-seed diversity context.

Attention is still described as temporal allocation rather than raw-feature importance or causal attribution.
```

---

# 170. Failure conditions

Phase55 FAIL if:

```text
head pairs are missing or duplicated

profiles do not sum to one

target alignment differs between heads

pairwise metric uses different lag support

JSD/TVD/Wasserstein integrity fails

head matrices are reordered by similarity as the canonical source

a redundancy threshold is invented after seeing results

a weighted “head score” is created

best head is declared

head is pruned/ablated

model is retrained

Test predictions are recomputed

error-conditioned comparison is introduced

cross-seed same-index heads are treated as same semantic role

cross-layer same-index heads are treated as same head

attention-profile redundancy is claimed as functional redundancy

feature importance or causal claims are made.
```

---

# 171. Common mistakes

## 171.1 Cosine cao rồi kết luận hai heads redundant hoàn toàn

Sai. Chỉ cho thấy mean temporal profiles giống nhau theo cosine.

Value projections vẫn khác.

## 171.2 JSD nhỏ rồi prune một head

Sai. Phase55 không pruning.

## 171.3 Sắp xếp heads theo entropy thấp→cao

Tạo implicit leaderboard. Giữ architectural order.

## 171.4 Head1 Layer1 và Head1 Layer2 là cùng một head

Sai. Khác parameter set.

## 171.5 Head1 seed42 và Head1 seed123 có cùng semantics

Không được giả định.

## 171.6 Dùng một weighted diversity score

Không cần và khó biện minh.

## 171.7 Chỉ dùng correlation

Correlation cao không đảm bảo absolute probability allocation giống nhau.

## 171.8 Dùng only top1 lag frequency

Bỏ qua toàn bộ remaining attention mass.

## 171.9 Chạy t-test trên từng timestamp như iid

Không chuẩn vì time dependence.

## 171.10 Head nào recent hơn thì gọi head tốt hơn

Sai. Recency là behavior, không phải quality.

## 171.11 Compare heads theo RMSE

Một head không có independent prediction output trong current architecture.

## 171.12 Ablate head để xem RMSE

Không thuộc Phase55.

---

# 172. Human-readable report structure

`head_comparison_report.md`:

```text
1. Objective
2. Why head comparison is behavioral, not predictive ranking
3. Upstream sources and integrity
4. Within-seed/within-layer comparison scope
5. Temporal profile similarity metrics
6. Why JSD and Wasserstein are useful
7. Head-level behavior summaries
8. Pairwise profile similarity results
9. Recency-allocation differences
10. Concentration differences
11. Temporal coverage differences
12. Top1-lag distribution differences
13. Paired same-target behavioral differences
14. Head-to-layer-mean deviation
15. Layer-level head diversity
16. Attention-allocation redundancy caveat
17. Why no best head is selected
18. Why no head pruning is performed
19. Handoff to error-conditioned attention
20. Handoff to seed-stability analysis
21. Limitations
22. Definition of Done
```

---

# 173. README requirements

`README_HEAD_COMPARISON.md` explains:

```text
why comparisons are within seed/layer
why same-index heads across seeds are not assumed identical
what Pearson/Spearman/Cosine/JSD/L1/L2 mean
why Wasserstein uses lag minutes
what TVD on top1 lag distribution means
why paired same-target differences are useful
why no iid p-values are central
why architectural head order is preserved
why redundancy is only attention-allocation redundancy
why no best head/pruning is allowed
how Phase56/57 consume outputs.
```

---

# 174. Summary artifact

`head_comparison_summary.json`:

```text
version
source_phase54_version
source_phase52_version
final_lock_sha256
test_population_sha256
seed_list
lookback
layers
heads
pair_count_per_layer
profile_integrity_status
target_alignment_status
pairwise_similarity_status
wasserstein_status
top1_distribution_status
head_behavior_summary_status
layer_diversity_status
findings
best_head_selected=false
head_pruning=false
head_ablation=false
error_conditioning=false
cross_seed_head_matching=false
feature_importance_claim=false
causal_claim=false
phase56_ready
phase57_context_ready
overall_status
```

Do not fabricate runtime pair counts before execution.

---

# 175. Phase55 sign-off

`phase_55_signoff.json` minimum:

```text
phase=55
phase_name=Head comparison
version=HEAD_COMPARISON-v1
source_phase54_version
source_phase52_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
lookback_steps
num_layers
num_heads
expected_pair_count_per_layer
actual_pair_count_per_layer
profile_integrity_verified
target_alignment_verified
pairwise_profile_metrics_complete
pairwise_wasserstein_complete
paired_target_differences_complete
top1_distribution_comparison_complete
head_behavior_summary_complete
head_to_layer_mean_complete
layer_diversity_complete
architectural_order_preserved
best_head_selected=false
head_pruning=false
head_ablation=false
model_training=false
test_metric_recomputation=false
error_conditioning=false
cross_seed_head_matching=false
same_index_semantic_alignment_assumed=false
feature_importance_claim=false
causal_claim=false
phase56_ready
phase57_context_ready
warnings
overall_status
created_at
```

---

# 176. Definition of Done

\[
\boxed{
Verified\ Head\ Profiles
+
All\ Within\text{-}Layer\ Head\ Pairs
+
Profile\ Similarity
+
JSD/L1/L2
+
Temporal\ Wasserstein
+
Recency/Entropy/Coverage\ Differences
+
Top1\ Distribution\ Comparison
+
Paired\ Same\text{-}Target\ Differences
+
Layer\ Diversity
+
No\ Best\ Head
+
No\ Pruning
+
Phase56\ Handoff
}
\]

---

# 177. Final status contract

```text
PHASE 55 compares attention heads behaviorally.

Primary scope:
within same seed
within same layer
across heads.

Core comparison:
mean temporal profiles.

Required pairwise metrics:
Pearson
Spearman
Cosine
JSD
L1
L2
Wasserstein minutes
expected-lag difference
entropy difference
recent1h difference
top1 TVD/JSD.

Per-target paired metrics:
entropy
expected lag
recent1h
Lag80.

Layer diversity:
mean/median/max pairwise JSD
mean L1
mean Wasserstein
mean abs expected-lag diff
mean top1 TVD.

Head behavior cards:
architectural order
no rank
no score.

Critical caveats:
similar attention profiles
do not prove functional redundancy.

Same-index heads:
not assumed semantically equivalent
across seeds or layers.

Forbidden:
best head
head pruning
head ablation
weighted head score
error conditioning
seed matching
retraining
Test metric recomputation
feature-importance claim
causal claim.

After HEAD_COMPARISON-v1 PASS:
proceed to
PHASE 56 — Error-Conditioned Attention.
```

---

# 178. Final check

Correct:

```text
verify Phase54 profiles
→ build all within-layer head pairs
→ compare profile shape
→ compare temporal recency/concentration/coverage
→ paired same-target differences
→ layer diversity
→ no winner
→ Phase56 handoff
```

Incorrect:

```text
entropy thấp nhất
→ best head
```

Incorrect:

```text
JSD gần 0
→ prune head
```

Incorrect:

```text
Head1 across seeds
→ same learned function
```

Incorrect:

```text
compare attention with error
→ Phase55 result
```

Chỉ sau khi:

```text
HEAD_COMPARISON-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase56_ready = true
```

mới chuyển sang **PHASE 56 — Error-Conditioned Attention**.
