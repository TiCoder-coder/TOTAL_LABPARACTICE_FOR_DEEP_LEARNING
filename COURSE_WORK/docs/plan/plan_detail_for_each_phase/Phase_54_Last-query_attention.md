# PHASE 54 — LAST-QUERY ATTENTION

## Kế hoạch phân tích định lượng temporal attention của newest historical query trên toàn bộ Held-Out Test từ raw artifacts `ATTENTION_EXTRACTION-v1`

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Model:** Attention-Aware Transformer Encoder for regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Primary raw source:** Phase52 `last_query_attention_seed*.npz`  
**Last-query definition:** `A[:, :, L-1, :]`  
**Last-query tensor contract:** `[target, layer, head, source]`  
**Source position semantics:** historical input positions only  
**Phase ID:** `PHASE_54_LAST_QUERY_ATTENTION`  
**Output version:** `LAST_QUERY_ATTENTION-v1`  
**Phase trước:** `Phase_53_Attention_heatmaps.md`  
**Phase sau:** `Phase_55_Head_comparison.md`

---

# 1. Vai trò của Phase 54

Phase54 là bước **quantitative temporal-focus analysis** cho hàng attention của query mới nhất trong input window:

\[
a_{s,t,l,h,p}
=
A_{s,t,l,h,q=L-1,p}
\]

trong đó:

```text
s = seed
t = Test target
l = encoder layer
h = attention head
p = source/key historical position.
```

Phase54 trả lời:

> Khi newest historical token được Encoder xử lý, attention của nó phân bố về những lag lịch sử nào, mức độ tập trung/phân tán ra sao, và cấu trúc temporal focus tổng thể trên Held-Out Test như thế nào?

Mục tiêu:

```text
1. Verify Phase52 raw last-query files and target/lag mappings.
2. Reconstruct and verify per-target last-query probability vectors.
3. Quantify recency vs long-history attention allocation.
4. Quantify attention concentration using entropy/top-k mass.
5. Quantify expected temporal lag.
6. Quantify recency coverage radii such as lag50/lag80/lag90.
7. Build per-seed/per-layer/per-head average temporal profiles.
8. Build layer-level head-mean profiles without selecting heads.
9. Characterize top-attended lag distributions.
10. Produce all-Test last-query plots and tables.
11. Provide deterministic case-level last-query views for Phase51 shared ranks 1–5.
12. Preserve head identity and avoid best-head ranking.
13. Avoid error-conditioned grouping; that belongs Phase56.
14. Avoid formal head-to-head ranking; that belongs Phase55.
15. Avoid cross-seed stability claims; that belongs Phase57.
16. Handoff standardized last-query metrics and profiles to Phase55–57.
```

Nguyên tắc trung tâm:

\[
\boxed{
Frozen\ Last\text{-}Query\ Raw\ Attention
+
All\ Test\ Targets
+
Per\ Seed/Layer/Head
+
Temporal\ Lag\ Profiles
+
Concentration\ Metrics
+
No\ Head\ Winner
+
No\ Error\ Conditioning
}
\]

---

# 2. Phase54 sử dụng raw Phase52, không đọc số từ heatmap

Numerical source bắt buộc:

```text
last_query_attention_seed42.npz
last_query_attention_seed123.npz
last_query_attention_seed2026.npz.
```

Phase53 heatmaps chỉ là:

```text
visual context.
```

Forbidden:

```text
digitize heatmap pixels
read bottom-row colors
estimate attention from PNG.
```

---

# 3. Không re-extract attention

Hard:

```text
checkpoint load for new attention = forbidden
new Test inference              = forbidden
new attention extraction        = forbidden.
```

Nếu Phase52 raw last-query artifact thiếu/corrupt:

```text
STOP
→ resolve Phase52 artifact integrity.
```

---

# 4. Upstream hard gate

Required:

```text
phase_52_signoff.json
phase54_last_query_attention_handoff.json
raw_attention_checksums.json
attention_test_target_order.csv
attention_relative_position_map.csv
attention_last_query_summary.csv
attention_recent_mass_summary.csv
attention_top_source_summary.csv
```

Phase53 context:

```text
phase_53_signoff.json
phase54_last_query_attention_context_handoff.json
attention_heatmap_report_cases.csv
```

Hard:

```text
phase54_ready = true
```

from Phase52 numerical handoff.

Phase53 visual context may be:

```text
PASS
or
PASS_WITH_WARNING.
```

---

# 5. Raw last-query tensor semantics

Per seed:

```text
last_query_attention
shape
=
[N_test,N_layers,N_heads,L].
```

Canonical axis order:

```text
[target, layer, head, source].
```

For every vector:

\[
\sum_{p=0}^{L-1} a_p \approx 1.
\]

---

# 6. Last-query definition bị khóa

Correct:

```text
A[:, :, L-1, :]
```

Incorrect:

```text
A[:, :, :, L-1].
```

Phase54 never recomputes last-query from PNG.

---

# 7. Why newest query matters

If final pooling is:

```text
LAST_STEP
```

the newest encoded token is directly passed to the regression head, so last-query attention is a particularly relevant internal diagnostic.

If final pooling is:

```text
MEAN
```

the final prediction pools all encoded positions; therefore last-query attention is still informative but **not a complete view** of the attention contributing to the pooled representation.

---

# 8. Pooling interpretation flag

Load exact:

```text
last_query_directly_corresponds_to_pooled_token
```

from Phase52 handoff.

Report:

```text
true  → LAST_STEP pooling
false → MEAN pooling.
```

No change to analysis formulas.

---

# 9. Attention is temporal, not raw-feature attention

Each source position corresponds to:

```text
one historical timestamp token
```

after feature projection.

Phase54 may say:

```text
attention mass at 60-minute lag
attention concentrated on recent historical tokens.
```

Phase54 may not say:

```text
attention to T2
attention to humidity
feature X importance.
```

---

# 10. Attention is not causal attribution

A large attention weight at lag `k` does not prove:

```text
the observation k steps ago caused the forecast.
```

The model output also depends on:

```text
value projections
residual paths
LayerNorm
FFN
later layers
pooling
regression head.
```

---

# 11. Same-index head semantics across seeds are not guaranteed

Keep:

```text
Head1 seed42
Head1 seed123
Head1 seed2026
```

as architectural indices only.

Do not claim they learned the same role.

Formal head matching/stability belongs Phase57.

---

# 12. Phase54 does not select a “best head”

No ranking by:

```text
lowest entropy
highest recent mass
largest top1 weight
smallest expected lag
visual sharpness.
```

Phase55 can compare head behaviors, but not “best” in predictive quality unless separately justified.

---

# 13. Phase54 does not condition on forecast error

Do not split attention by:

```text
low error / high error
underprediction / overprediction
rapid-change error
worst cases.
```

Those group-level comparisons belong Phase56.

Phase54 can show deterministic Phase51 shared-top cases as examples only.

---

# 14. Phase54 does not perform seed-stability inference

Do not compute:

```text
cross-seed head matching
seed stability score
same-head correlation as a stability conclusion.
```

That belongs Phase57.

---

# 15. Temporal position mapping

Use frozen:

```text
attention_relative_position_map.csv.
```

For source position `p`:

\[
LagSteps_p
=
H+(L-1-p).
\]

With:

```text
H=1.
```

Thus:

\[
LagSteps_p=L-p.
\]

---

# 16. Lag minute mapping

At 10-minute cadence:

\[
LagMinutes_p
=
10\times LagSteps_p.
\]

Expected:

```text
source L-1 → 10 min
source 0   → L×10 min.
```

---

# 17. Recency order

For cumulative recency analyses, sort source positions by:

```text
lag ascending
```

which means:

```text
newest → oldest.
```

This is the opposite direction of raw array position order:

```text
position 0 → oldest
position L-1 → newest.
```

The distinction must be explicit.

---

# 18. Canonical recency vector

Define:

```text
a_recency[k]
```

where:

```text
k=1 corresponds lag1/newest
k=L corresponds lagL/oldest.
```

Then:

```text
a_recency[k]
=
raw_attention[position=L-k].
```

No numerical change, only deterministic re-indexing for recency analysis.

---

# 19. Raw vector remains authoritative

Store raw-position vector unchanged.

Any recency-ordered representation is:

```text
derived.
```

It must include mapping metadata.

---

# 20. Per-vector integrity recheck

For every seed/target/layer/head:

```text
finite
nonnegative within tolerance
sum ≈ 1
length = L.
```

Reuse Phase52 tolerance.

No renormalization.

---

# 21. Phase52 summary verification

Phase54 should recompute from raw vectors and verify Phase52:

```text
entropy
normalized entropy
expected lag
top1 source
top1 lag
top1 weight
top5 mass
recent-window masses.
```

Create:

```text
last_query_phase52_summary_reconstruction_audit.csv.
```

Any material mismatch:

```text
STOP.
```

---

# 22. Core per-vector metrics

For every:

```text
seed
target
layer
head
```

Phase54 uses the following metrics:

```text
entropy
normalized entropy
effective source count
expected lag steps
expected lag minutes
lag standard deviation
top1 lag
top1 weight
top1 tie count
top5 mass
recent 1h mass
recent 6h mass
recent 12h mass
recent 24h mass
lag50 coverage
lag80 coverage
lag90 coverage.
```

---

# 23. Entropy

\[
H(a)
=
-\sum_p a_p\log(a_p+\epsilon).
\]

Use same numerical epsilon as Phase52:

```text
1e-12
```

or exact Phase52 contract value.

---

# 24. Normalized entropy

\[
H_{norm}
=
\frac{H}{\log L}.
\]

For valid probability vector:

```text
approximately 0..1.
```

Interpretation:

```text
closer to 0 → more concentrated
closer to 1 → more diffuse.
```

No quality judgment.

---

# 25. Effective source count

Define:

\[
N_{eff}
=
\exp(H).
\]

Interpretation:

> Number of uniformly weighted source positions that would produce the same Shannon entropy.

Bounds:

```text
approximately 1..L.
```

This is an interpretable concentration statistic.

---

# 26. Expected lag

\[
E[LagSteps]
=
\sum_p a_p LagSteps_p.
\]

\[
E[LagMinutes]
=
10\times E[LagSteps].
\]

---

# 27. Attention-weighted lag variance

\[
Var(Lag)
=
\sum_p a_p
(LagSteps_p-E[LagSteps])^2.
\]

\[
SD(Lag)
=
\sqrt{Var(Lag)}.
\]

This measures temporal spread around expected lag.

---

# 28. Expected lag caveat

A bimodal distribution can have an expected lag between two peaks where little actual attention exists.

Therefore expected lag must be interpreted together with:

```text
entropy
top sources
cumulative recency profile.
```

---

# 29. Top1 lag

From Phase52 deterministic tie rule:

```text
if exact max tie:
choose newest tied source
and record tie count.
```

No alternative tie rule in Phase54.

---

# 30. Top5 mass

\[
Top5Mass
=
\sum_{\text{five largest }a_p}a_p.
\]

This measures concentration on a small set of source positions.

---

# 31. Fixed recent-window masses

Use:

```text
recent_1h  → lag <= 6
recent_6h  → lag <= 36
recent_12h → lag <= 72
recent_24h → lag <= 144
```

with:

```text
effective_steps = min(requested_steps,L).
```

---

# 32. Coverage-truncated flag

If final `L` is shorter than named horizon:

```text
coverage_truncated=true.
```

Example:

```text
L=36
recent_12h actually covers only 6h.
```

Do not label it simply “12h mass” without truncation metadata.

---

# 33. Recency cumulative mass

Sort by lag ascending:

\[
C(k)
=
\sum_{j=1}^{k}a_{recency,j}.
\]

Then:

```text
C(1) = lag1 mass
C(L) ≈ 1.
```

---

# 34. Lag50 coverage radius

Define:

\[
Lag50
=
\min\{k:C(k)\ge0.50\}.
\]

Interpretation:

> Smallest most-recent history radius containing at least 50% of last-query attention mass.

---

# 35. Lag80 coverage radius

\[
Lag80
=
\min\{k:C(k)\ge0.80\}.
\]

---

# 36. Lag90 coverage radius

\[
Lag90
=
\min\{k:C(k)\ge0.90\}.
\]

---

# 37. Coverage radii are recency radii, not quantiles of source positions in raw order

Always compute in:

```text
newest → oldest
```

recency order.

---

# 38. Coverage-radius interpretation

Smaller:

```text
Lag50/Lag80/Lag90
```

means more attention mass is contained in recent history.

Larger means more history is needed to accumulate the same mass.

No claim that smaller is better.

---

# 39. Optional lag25

Not required.

Keep Phase54 core focused on:

```text
50%
80%
90%.
```

---

# 40. Per-target long-form metrics table

Create one row for every:

```text
target × seed × layer × head.
```

This is the quantitative source for Phase55–57.

---

# 41. All-Test average temporal profile per seed/layer/head

For each:

```text
seed s
layer l
head h
source lag k
```

compute:

\[
\bar a_{s,l,h,k}
=
\frac{1}{N}
\sum_t a_{s,t,l,h,k}.
\]

This is:

```text
mean attention weight at lag k across Test targets.
```

---

# 42. Average profile remains a valid probability distribution

Because it is the mean of normalized vectors:

\[
\sum_k\bar a_{s,l,h,k}\approx1.
\]

Hard audit.

---

# 43. Average profile is not “average causal importance”

It is:

```text
mean temporal attention allocation.
```

---

# 44. Per-lag variability across Test targets

For each:

```text
seed/layer/head/lag
```

compute:

```text
mean
median
sample std
p25
p75
p05
p95.
```

This shows whether attention at a lag is stable or target-dependent.

---

# 45. No confidence interval claim

The `p05–p95` range is:

```text
empirical target distribution
```

not a confidence interval.

Test timestamps are temporally dependent.

---

# 46. Layer-level head-mean profile

Within one seed/layer:

\[
\bar a_{s,l,\cdot,k}
=
\frac{1}{H}
\sum_h\bar a_{s,l,h,k}.
\]

This is permutation-invariant to head ordering within the layer.

Allowed as:

```text
LAYER_HEAD_MEAN_PROFILE.
```

---

# 47. Why layer head-mean is useful

It summarizes:

```text
overall temporal allocation of one encoder layer
```

without asserting one head is superior.

---

# 48. Layer head-mean is derived, not raw

Preserve per-head profiles.

Do not replace them.

---

# 49. No global layer ranking

Phase54 may show Layer1 vs Layer2 temporal profiles descriptively.

Do not declare:

```text
Layer2 is better
Layer1 is useless.
```

---

# 50. Seed-level overall head/layer mean

Optional compact summary:

\[
\bar a_{s,k}
=
\frac{1}{N_{layers}N_{heads}}
\sum_{l,h}\bar a_{s,l,h,k}.
\]

Label:

```text
SEED_OVERALL_TEMPORAL_PROFILE_DESCRIPTIVE.
```

Useful for one high-level plot.

Do not average seeds into one primary profile.

---

# 51. No cross-seed head averaging

Forbidden:

```text
Head1 across seeds mean profile
```

as if heads are semantically matched.

---

# 52. Cross-seed layer-head-mean visual overlay

Because averaging over heads removes head permutation, Phase54 may plot:

```text
Layer1 head-mean profile for each seed
Layer2 head-mean profile for each seed.
```

This is descriptive visual context.

Formal seed stability belongs Phase57.

---

# 53. Top-lag frequency

For each:

```text
seed/layer/head
```

count how often each lag is:

```text
top1_lag.
```

Output:

```text
count
fraction.
```

---

# 54. Top-lag frequency interpretation

This answers:

> Which lag most often receives the single largest last-query weight for this head?

It does not measure total attention mass.

Must be interpreted with mean profile and recent-window masses.

---

# 55. Top-lag tie handling

Use frozen Phase52:

```text
newest tied source
```

for top1 frequency.

Also report:

```text
fraction targets with top1 tie_count > 1.
```

---

# 56. Lag-bin allocation

For report readability, define non-overlapping recency bins:

```text
BIN_10M_TO_1H
lags 1..6

BIN_GT1H_TO_6H
lags 7..36

BIN_GT6H_TO_12H
lags 37..72

BIN_GT12H_TO_24H
lags 73..144

BIN_BEYOND_24H
lags 145..L
```

Only bins with supported lags exist.

---

# 57. Why non-overlapping bins

Phase52 recent masses are cumulative/overlapping:

```text
1h
6h
12h
24h.
```

Phase54 lag bins provide mutually exclusive temporal allocation components that sum to 1.

---

# 58. Lookback-dependent bin handling

If `L < upper bound`:

```text
truncate final supported bin.
```

Do not create empty impossible bins as scientific rows unless marked:

```text
NOT_APPLICABLE.
```

---

# 59. Lag-bin mass sum audit

For every last-query vector:

\[
\sum_{bins}Mass_{bin}\approx1.
\]

Hard.

---

# 60. Non-overlapping bin labels must state effective lag range

Example if:

```text
L=36
```

only:

```text
10m–1h
>1h–6h
```

exist.

---

# 61. Aggregate metric summaries

For each:

```text
seed/layer/head
```

summarize across Test targets:

```text
mean
sample SD
median
p05
p25
p75
p95
```

for:

```text
normalized entropy
effective source count
expected lag minutes
lag SD minutes
top1 weight
top5 mass
recent masses
lag50
lag80
lag90.
```

---

# 62. No statistical p-values in Phase54

Phase54 is descriptive temporal-attention characterization.

No:

```text
t-test
ANOVA
head significance test
seed significance test.
```

Head comparison is next phase.

---

# 63. No head ranking from aggregate summaries

Sort report tables by:

```text
seed
layer index
head index.
```

Not by metric value.

---

# 64. Profile normalization audit

Every average temporal profile:

```text
sum of mean weights by lag ≈ 1.
```

No post-hoc renormalization except a display-only derived copy if tiny numerical error exists; preferred no renormalization and simply verify tolerance.

---

# 65. Profile temporal orientation

Profile plots should use:

```text
x = lag minutes
```

with:

```text
10 min on left
older history to the right.
```

This is recency-oriented and intuitive.

---

# 66. Raw position plot is optional

Primary profile plot should use lag, not raw source index.

If raw source position is shown:

```text
position 0 oldest
position L-1 newest.
```

Do not mix axis conventions between figures.

---

# 67. Primary profile x-axis

Canonical:

```text
Lag from forecast target
```

in:

```text
minutes or hours.
```

Use minutes for machine tables and readable hour labels on figures.

---

# 68. Key lag markers

Plot vertical reference markers at supported:

```text
10 min
1 h
6 h
12 h
24 h.
```

These are context guides, not threshold claims.

---

# 69. Average profile figure per seed/layer

For each:

```text
seed × layer
```

plot all heads as separate lines:

```text
weight vs lag.
```

Do not emphasize one line as best.

---

# 70. Layer head-mean overlay

Add one separate figure:

```text
head-mean temporal profile
```

per seed/layer.

Avoid overlaying head mean on already crowded per-head figure unless clear.

---

# 71. Top-lag frequency figure

For each:

```text
seed × layer
```

show head-specific top1 lag frequency distributions.

Preferred representation:

```text
heatmap:
rows=heads
columns=lag bins/positions
value=top1 frequency.
```

This is a quantitative heatmap, not the raw attention heatmap from Phase53.

---

# 72. Entropy distribution figure

For each:

```text
seed × layer
```

show normalized entropy distributions for all heads.

Use:

```text
boxplots/violin/ECDF
```

without ranking.

---

# 73. Expected-lag distribution figure

For each seed/layer:

```text
per-head expected lag distribution.
```

Common x-axis across heads.

---

# 74. Recent-mass figure

For each seed/layer/head show mean:

```text
1h
6h
12h
24h
```

cumulative mass.

Because these are nested, clearly label:

```text
cumulative recent-history mass.
```

---

# 75. Non-overlap lag-bin figure

Prefer stacked bars:

```text
10m–1h
>1h–6h
>6h–12h
>12h–24h
>beyond24h
```

for each head.

Stack should sum to 1.

---

# 76. Coverage-radius figure

For each seed/layer:

```text
Lag50
Lag80
Lag90
```

distribution by head.

Units:

```text
steps
minutes/hours.
```

---

# 77. Case-level deterministic views

Use exact Phase53/Phase51 report case set:

```text
shared worst ranks 1–5.
```

No new case selection.

---

# 78. Case-level last-query plot

For each selected case:

```text
seed
layer
head
```

plot:

```text
attention weight vs lag.
```

This is a line/stem plot of raw last-query vector.

---

# 79. Case-level grid layout

Recommended:

```text
rows = layers
columns = heads
```

one grid per:

```text
case × seed.
```

Same architectural arrangement as Phase53.

---

# 80. Cross-seed case-level plot

Optional report view:

```text
rows = seeds
columns = heads
```

for a fixed:

```text
case × layer.
```

Must retain caveat that same-index heads are not semantically guaranteed across seeds.

---

# 81. Case-level views are examples, not error-conditioned analysis

Do not aggregate shared worst cases and compare them statistically with non-worst cases in Phase54.

That belongs Phase56.

---

# 82. No low-error control group in Phase54

No.

---

# 83. No regime-specific attention table in Phase54

No:

```text
EXTREME_HIGH attention
RAPID_CHANGE attention
DIR_UP attention.
```

Phase56 handles conditioned attention.

---

# 84. Phase54 primary questions

```text
Q54.1  How much last-query attention mass lies in recent history?
Q54.2  How diffuse/concentrated are last-query distributions?
Q54.3  What is the expected historical lag attended by each head?
Q54.4  How much temporal history is needed to accumulate 50/80/90% attention mass?
Q54.5  Which lags most often receive top1 attention within each head?
Q54.6  How do layer-level temporal profiles look across the final Test?
Q54.7  Do heads show diverse temporal allocation patterns?
Q54.8  Are some heads predominantly recent while others allocate more mass to older history?
Q54.9  How does the pattern vary across Test targets?
Q54.10 What exact quantitative artifacts should Phase55 compare across heads?
```

Question 8 can be described quantitatively without declaring superiority.

---

# 85. No “important historical lag” terminology without caveat

Prefer:

```text
high-attention lag
frequently top-attended lag
lag receiving larger average attention mass.
```

Avoid:

```text
most important lag
causal lag.
```

---

# 86. Daily seasonality caveat

If lag near:

```text
24h
```

receives elevated attention, safe wording:

> Attention mass is elevated around the 24-hour lag.

Do not immediately claim:

```text
the model learned daily seasonality
```

without broader pattern evidence.

A cautious statement:

```text
consistent with a daily-timescale temporal pattern
```

may be used later if repeated across heads/seeds and aligned with other evidence.

---

# 87. 12-hour/6-hour lag caveat

Same.

Do not overinterpret individual peaks.

---

# 88. Recent-history caveat

High recent mass may reflect:

```text
forecast recency dependence
```

but attention weight alone does not prove contribution magnitude.

---

# 89. Profile smoothing forbidden

Do not smooth average temporal attention curves with:

```text
moving average
Savitzky–Golay
spline.
```

Plot raw lag-level mean profile.

---

# 90. Profile interpolation forbidden

No.

---

# 91. Optional display aggregation into lag bins

For compact report figures, non-overlapping lag-bin masses are allowed because they are explicitly defined sums.

Do not interpolate between lags.

---

# 92. No rescaling each head profile to max=1

Forbidden.

Each profile retains actual probability weights.

---

# 93. Common y-axis within a figure

When plotting multiple heads:

```text
same attention-weight y-axis.
```

No per-head axis scaling.

---

# 94. Cross-figure scale

For average per-lag profile plots, preferred:

```text
same y-limit across heads within seed/layer.
```

Across seeds, use consistent scale where possible.

---

# 95. Cumulative mass plots

Optional:

```text
x=lag radius
y=cumulative recency mass.
```

All curves end at:

```text
~1.
```

Good for visualizing Lag50/80/90.

---

# 96. Cumulative mass direction

Start:

```text
10 min
```

and move older.

Do not plot from oldest toward newest while labeling it “recent coverage”.

---

# 97. Top-source lag distribution

For each seed/layer/head:

```text
top1 lag count/fraction
```

must include all supported lags, even if count 0 in machine-readable source.

Report plots can omit zero-only labels for readability while preserving full table.

---

# 98. Tie-frequency audit

Report:

```text
top1_tie_fraction.
```

If material ties are common, top1-frequency interpretation should mention the deterministic newest-tie rule.

---

# 99. Head metric independence caveat

Metrics like:

```text
entropy
expected lag
top5 mass
recent mass
```

are mathematically related.

Do not treat them as independent evidence.

---

# 100. Effective-source-count caveat

`exp(entropy)` assumes entropy interpretation of equivalent uniform support.

It is not a literal count of nonzero attended positions.

---

# 101. Expected-lag SD caveat

Large lag SD can arise from:

```text
diffuse profile
or
multi-modal separated attention peaks.
```

Use together with actual profile.

---

# 102. All-Test target dependence

Attention vectors across nearby Test timestamps are temporally dependent.

Do not treat:

```text
N_test vectors
```

as iid replicates for classical confidence intervals.

---

# 103. Empirical quantiles are allowed

Per-head:

```text
p05/p25/median/p75/p95
```

across Test targets are descriptive.

No confidence-interval wording.

---

# 104. Layer-level head-mean metric summary

For each seed/layer, compute head-mean attention vector per target:

\[
a^{layermean}_{t,k}
=
\frac{1}{H}\sum_h a_{t,h,k}.
\]

Then compute:

```text
entropy
expected lag
recent masses
Lag50/80/90.
```

Label:

```text
HEAD_MEAN_LAYER_DERIVED.
```

---

# 105. Why compute metrics from head-mean vector instead of averaging metric values

These are not always equivalent:

\[
Entropy(mean_h(a_h))
\ne
mean_h(Entropy(a_h)).
\]

Therefore store both concepts separately if needed:

```text
A. mean of head metric values
B. metric of head-mean vector.
```

Do not conflate.

---

# 106. Canonical layer aggregate

For temporal profile visualization:

```text
metric of head-mean vector
```

is preferred.

For head distribution summary:

```text
mean of per-head metrics
```

may be reported separately.

---

# 107. No layer aggregation across seeds as primary

Keep seed identity.

---

# 108. Output directory

```text
artifacts/
└── last_query_attention/
    ├── last_query_attention_manifest.json
    ├── last_query_attention_contract.json
    ├── phase54_preflight_audit.csv
    ├── last_query_source_verification.csv
    ├── last_query_integrity_audit.csv
    ├── last_query_phase52_summary_reconstruction_audit.csv
    ├── last_query_target_order_audit.csv
    ├── last_query_lag_mapping_audit.csv
    ├── last_query_metrics_long.csv
    ├── last_query_metric_summary_by_head.csv
    ├── last_query_profile_by_lag.csv
    ├── last_query_layer_head_mean_profile.csv
    ├── last_query_seed_overall_profile.csv
    ├── last_query_lag_bin_mass.csv
    ├── last_query_recent_mass_summary.csv
    ├── last_query_coverage_radius_summary.csv
    ├── last_query_top1_lag_frequency.csv
    ├── last_query_top1_tie_summary.csv
    ├── last_query_report_case_manifest.csv
    ├── last_query_report_case_metrics.csv
    ├── last_query_analysis_findings.csv
    ├── last_query_attention_tests.csv
    ├── last_query_attention_discrepancies.json
    ├── phase55_head_comparison_handoff.json
    ├── phase56_error_conditioned_attention_context_handoff.json
    ├── phase57_seed_stability_attention_context_handoff.json
    ├── last_query_attention_summary.json
    ├── last_query_attention_report.md
    ├── figures/
    │   ├── LASTQ_54_01_mean_profiles_seed42_layer01.png
    │   ├── LASTQ_54_02_mean_profiles_seed42_layer02.png
    │   ├── LASTQ_54_03_mean_profiles_seed123_layer01.png
    │   ├── LASTQ_54_04_mean_profiles_seed123_layer02.png
    │   ├── LASTQ_54_05_mean_profiles_seed2026_layer01.png
    │   ├── LASTQ_54_06_mean_profiles_seed2026_layer02.png
    │   ├── LASTQ_54_07_layer_head_mean_profiles.png
    │   ├── LASTQ_54_08_normalized_entropy_by_head.png
    │   ├── LASTQ_54_09_expected_lag_by_head.png
    │   ├── LASTQ_54_10_top5_mass_by_head.png
    │   ├── LASTQ_54_11_recent_mass_by_head.png
    │   ├── LASTQ_54_12_nonoverlap_lag_bins.png
    │   ├── LASTQ_54_13_lag50_lag80_lag90.png
    │   ├── LASTQ_54_14_top1_lag_frequency.png
    │   ├── LASTQ_54_15_cumulative_recency_profiles.png
    │   └── report_cases/
    │       ├── SHARED_R01_SEED42_LAST_QUERY.png
    │       ├── SHARED_R01_SEED123_LAST_QUERY.png
    │       ├── SHARED_R01_SEED2026_LAST_QUERY.png
    │       └── ...
    ├── README_LAST_QUERY_ATTENTION.md
    └── phase_54_signoff.json
```

Figure names depending on actual number of layers must be generated dynamically; the examples above illustrate naming only.

---

# 109. Required outputs

```text
O54.1  Analysis manifest
O54.2  Analysis contract
O54.3  Preflight audit
O54.4  Raw-source verification
O54.5  Last-query integrity audit
O54.6  Phase52 summary reconstruction audit
O54.7  Test target-order audit
O54.8  Lag-mapping audit
O54.9  Per-target per-head metrics long table
O54.10 Head-level metric summary
O54.11 Mean temporal profile by lag
O54.12 Layer head-mean profile
O54.13 Seed overall descriptive profile
O54.14 Non-overlapping lag-bin mass
O54.15 Recent cumulative mass summary
O54.16 Lag50/Lag80/Lag90 coverage summary
O54.17 Top1 lag frequency
O54.18 Top1 tie summary
O54.19 Report-case manifest
O54.20 Report-case last-query metrics/views
O54.21 Core figures
O54.22 Findings
O54.23 Phase55 handoff
O54.24 Phase56 context handoff
O54.25 Phase57 context handoff
O54.26 Tests
O54.27 Discrepancy log
O54.28 Summary JSON
O54.29 Human-readable report
O54.30 README
O54.31 Sign-off
```

---

# 110. Analysis manifest

`last_query_attention_manifest.json`:

```text
phase=54
version=LAST_QUERY_ATTENTION-v1
source_phase52_version
source_phase53_version
final_lock_sha256
test_population_sha256
target_order_sha256
position_map_sha256
seed_list=[42,123,2026]
lookback_steps
num_layers
num_heads
pooling
last_query_definition=A[:,:,L-1,:]
raw_axis_order=[target,layer,head,source]
new_attention_extraction=false
new_test_inference=false
head_ranking=false
error_conditioning=false
seed_stability_inference=false
status
created_at
```

---

# 111. Analysis contract

`last_query_attention_contract.json` must freeze:

```text
Raw source:
Phase52 last-query NPZ only.

Last query:
q=L-1.

Source:
historical input positions only.

Lag:
H+(L-1-p), H=1.

Per-vector metrics:
entropy
normalized entropy
effective source count
expected lag
lag SD
top1 lag/weight/tie
top5 mass
recent cumulative masses
Lag50/Lag80/Lag90.

Non-overlap lag bins:
1–6
7–36
37–72
73–144
145–L where supported.

All-Test aggregation:
per seed/layer/head
no head winner.

Layer aggregate:
head-mean profile allowed.

Report examples:
Phase51 shared ranks1–5 only.

No:
new extraction
head ranking
error-conditioned groups
seed-stability claims
raw-feature importance
causal attribution.
```

---

# 112. Preflight audit

`phase54_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Checks:

```text
Phase52 approved
phase54_ready=true
3 last-query NPZ files available
checksums match
same target order
same target count
same layers
same heads
same lookback
position map matches
Phase52 summary artifacts exist
Phase53 context available
report cases equal Phase51 shared ranks1–5
no new extraction required.
```

---

# 113. Source verification schema

`last_query_source_verification.csv`:

```text
seed
raw_file
expected_sha256
observed_sha256
shape
dtype
target_count
layer_count
head_count
lookback
target_order_match
position_map_match
status
```

---

# 114. Integrity audit schema

`last_query_integrity_audit.csv`:

```text
seed
layer_idx0
head_idx0
vector_count
vector_length
finite
min_weight
max_weight
min_vector_sum
max_vector_sum
max_abs_sum_minus_1
nonnegative_pass
sum_pass
status
```

---

# 115. Phase52 reconstruction audit schema

`last_query_phase52_summary_reconstruction_audit.csv`:

```text
seed
target_id
layer_idx0
head_idx0
metric
phase52_value
phase54_recomputed_value
abs_difference
tolerance
pass
status
```

Required metrics:

```text
entropy
normalized_entropy
expected_lag_steps
top1_lag_steps
top1_weight
top5_mass
recent_1h_mass
recent_6h_mass
recent_12h_mass
recent_24h_mass.
```

Could be very large; a compact aggregated version plus full mismatch file is acceptable, but all metrics must be verified programmatically.

---

# 116. Target-order audit

`last_query_target_order_audit.csv`:

```text
row_idx0
target_id
target_timestamp
seed42_match
seed123_match
seed2026_match
Phase52_order_match
status
```

---

# 117. Lag-mapping audit

`last_query_lag_mapping_audit.csv`:

```text
source_position_idx0
lag_steps
lag_minutes
recency_index
expected_position_from_lag
mapping_consistent
status
```

Hard:

```text
lag1 ↔ position L-1
lagL ↔ position0.
```

---

# 118. Per-target metric table schema

`last_query_metrics_long.csv`:

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
effective_source_count
expected_lag_steps
expected_lag_minutes
lag_sd_steps
lag_sd_minutes
top1_source_position_idx0
top1_lag_steps
top1_lag_minutes
top1_weight
top1_tie_count
top5_mass
recent_1h_mass
recent_6h_mass
recent_12h_mass
recent_24h_mass
recent_1h_truncated
recent_6h_truncated
recent_12h_truncated
recent_24h_truncated
lag50_steps
lag50_minutes
lag80_steps
lag80_minutes
lag90_steps
lag90_minutes
status
```

---

# 119. Head-level summary schema

`last_query_metric_summary_by_head.csv`:

```text
seed
layer_idx0
head_idx0
metric
N
mean
sample_sd
median
p05
p25
p75
p95
min
max
status
```

Metrics include all primary metrics.

Rows remain in architectural order.

---

# 120. Temporal profile schema

`last_query_profile_by_lag.csv`:

```text
seed
layer_idx0
head_idx0
source_position_idx0
lag_steps
lag_minutes
mean_weight
median_weight
sample_sd_weight
p05_weight
p25_weight
p75_weight
p95_weight
target_count
profile_sum_mean_weights
status
```

Hard:

```text
sum_lag(mean_weight) ≈ 1
```

for every seed/layer/head.

---

# 121. Layer head-mean profile schema

`last_query_layer_head_mean_profile.csv`:

```text
seed
layer_idx0
lag_steps
lag_minutes
head_mean_weight
target_mean_weight
profile_sum
status
```

Recommended computation:

```text
mean across all targets and all heads
for each lag
```

which is equivalent to mean of per-head mean profiles.

---

# 122. Seed overall profile schema

`last_query_seed_overall_profile.csv`:

```text
seed
lag_steps
lag_minutes
mean_weight_across_layers_heads_targets
profile_sum
status
```

Descriptive only.

---

# 123. Lag-bin mass schema

`last_query_lag_bin_mass.csv`:

```text
seed
target_id
layer_idx0
head_idx0
lag_bin
lag_start_steps
lag_end_steps
effective_start
effective_end
mass
applicable
status
```

Canonical bins:

```text
1–6
7–36
37–72
73–144
145–L.
```

Only supported bins are applicable.

---

# 124. Recent-mass summary schema

`last_query_recent_mass_summary.csv`:

```text
seed
layer_idx0
head_idx0
window
effective_steps
truncated
mean_mass
sample_sd_mass
median_mass
p05_mass
p95_mass
status
```

Windows:

```text
1h
6h
12h
24h.
```

---

# 125. Coverage-radius summary schema

`last_query_coverage_radius_summary.csv`:

```text
seed
layer_idx0
head_idx0
coverage_level
N
mean_steps
sample_sd_steps
median_steps
p05_steps
p95_steps
mean_minutes
median_minutes
status
```

Coverage levels:

```text
0.50
0.80
0.90.
```

---

# 126. Top1 lag frequency schema

`last_query_top1_lag_frequency.csv`:

```text
seed
layer_idx0
head_idx0
lag_steps
lag_minutes
count
fraction
total_targets
status
```

Fractions per seed/layer/head sum to:

```text
1.
```

---

# 127. Top1 tie summary schema

`last_query_top1_tie_summary.csv`:

```text
seed
layer_idx0
head_idx0
target_count
tie_count_gt1
tie_fraction
max_tie_count
tie_rule=NEWEST_SOURCE
status
```

---

# 128. Report-case manifest

`last_query_report_case_manifest.csv`:

```text
report_order
shared_rank
target_id
target_timestamp
selection_rule=PHASE51_SHARED_RANK_1_TO_5
Phase53_heatmap_refs
status
```

No case substitutions.

---

# 129. Report-case metrics schema

`last_query_report_case_metrics.csv`:

```text
target_id
seed
layer_idx0
head_idx0
entropy
expected_lag_minutes
top1_lag_minutes
top1_weight
top5_mass
recent_1h_mass
recent_6h_mass
lag50_minutes
lag80_minutes
lag90_minutes
status
```

This is case-specific descriptive data only.

---

# 130. Figure LASTQ_54_01+ — mean profiles

For each:

```text
seed × layer
```

plot all heads:

```text
x=lag
y=mean attention weight.
```

Rules:

```text
lag1/newest at left
older lags to right
same y-axis within figure
no smoothing
no head ranking.
```

---

# 131. Layer head-mean figure

Plot:

```text
head-mean profile
```

for each:

```text
seed/layer.
```

Optional overlay across seeds per layer.

If overlay across seeds:

```text
no stability conclusion.
```

---

# 132. Entropy figure

Show normalized entropy distributions by:

```text
seed
layer
head.
```

Architectural order.

Do not sort by median entropy.

---

# 133. Expected-lag figure

Same.

Use:

```text
minutes/hours
```

for readable axis.

---

# 134. Top5 mass figure

Show distributions/means by head.

Top5 mass close to 1 means concentration in few source positions, but not predictive superiority.

---

# 135. Recent-mass figure

Show cumulative masses:

```text
1h
6h
12h
24h
```

with truncation labels if applicable.

---

# 136. Lag-bin stacked bars

For each seed/layer/head:

```text
non-overlapping temporal bins
```

stack to 1.

This is often the clearest high-level temporal allocation figure.

---

# 137. Coverage-radius figure

Plot:

```text
Lag50
Lag80
Lag90
```

per head.

Avoid connecting heads with lines that imply ordinal continuity between head indices.

Use grouped points/bars.

---

# 138. Top1 lag-frequency heatmap

Rows:

```text
heads
```

columns:

```text
lag positions or lag bins.
```

Value:

```text
fraction of Test targets.
```

One figure per seed/layer.

This is not raw query-source heatmap.

---

# 139. Cumulative recency profile

For each seed/layer/head:

\[
C(k)
=
\sum_{j=1}^{k} a_{recency,j}.
\]

Average across targets:

```text
mean cumulative recency curve.
```

Plot:

```text
x=lag radius
y=cumulative attention mass.
```

Markers at:

```text
50%
80%
90%.
```

---

# 140. Mean cumulative curve caveat

`mean(C_t(k))` is preferred.

Do not compute coverage radius from the averaged profile and assume it equals mean per-target Lag50/80/90.

Store both if needed:

```text
A. average cumulative profile
B. distribution of per-target coverage radii.
```

---

# 141. Case-level line plots

For deterministic shared ranks 1–5:

```text
one grid per case×seed
rows=layers
columns=heads.
```

Each panel:

```text
x=lag
y=raw last-query weight.
```

No smoothing.

---

# 142. Case line-plot y-scale

Within one case×seed grid, use common y-axis across panels where practical.

For cross-seed comparison, prefer same y-scale for same case.

Do not auto-normalize each head.

---

# 143. Case plot link to Phase53

Caption may reference corresponding:

```text
Phase53 dense heatmap.
```

But numeric values come only from Phase52 raw last-query arrays.

---

# 144. Findings codes

Possible:

```text
LAST_QUERY_SOURCE_VERIFIED
LAST_QUERY_SUMS_NORMALIZED
PHASE52_SUMMARY_RECONSTRUCTED
RECENT_HISTORY_MASS_DOMINANT
LONGER_HISTORY_MASS_SUBSTANTIAL
LAST_QUERY_ATTENTION_DIFFUSE
LAST_QUERY_ATTENTION_CONCENTRATED
EXPECTED_LAG_RECENT
EXPECTED_LAG_LONGER_HORIZON
TOP1_LAG_CLUSTERED_AT_RECENT_POSITIONS
TOP1_LAG_DISTRIBUTED_ACROSS_HISTORY
LAYER_TEMPORAL_PROFILES_DIFFER_DESCRIPTIVELY
HEAD_TEMPORAL_PROFILES_DIVERSE
HEAD_MEAN_PROFILE_RECENCY_DOMINANT
COVERAGE_RADIUS_SMALL
COVERAGE_RADIUS_LARGE
TOP1_TIES_RARE
TOP1_TIES_PRESENT
LOOKBACK_TRUNCATES_12H_WINDOW
LOOKBACK_TRUNCATES_24H_WINDOW
REPORT_CASE_VIEWS_COMPLETE
NO_HEAD_SELECTION
NO_ERROR_CONDITIONING
NO_SEED_STABILITY_CLAIM
ATTENTION_TEMPORAL_NOT_FEATURE_IMPORTANCE
READY_FOR_HEAD_COMPARISON
```

Avoid labels such as `RECENT`/`LARGE` without actual numbers in accompanying finding text.

---

# 145. Findings language

Safe:

> Across this head’s Test targets, the median expected lag was X minutes and the median mass assigned to the most recent hour was Y.

Safe:

> The average temporal profile assigned substantial probability mass to both recent and older source positions.

Safe:

> Different heads within the same layer exhibited distinct temporal allocation profiles.

Unsafe:

> Head 3 is the best forecasting head.

Unsafe:

> The model uses the 24-hour lag because daily seasonality causes energy consumption.

---

# 146. No arbitrary recency threshold beyond predefined windows

The windows:

```text
1h
6h
12h
24h
```

are fixed structural summaries.

Do not invent:

```text
“recent head” if recent_1h_mass > 0.63
```

after viewing data.

Phase55 may define descriptive head typologies only if predeclared there.

---

# 147. No head cluster analysis in Phase54

Do not run:

```text
k-means on heads
hierarchical head clustering
head archetype mining.
```

That would belong Phase55 if desired and properly predeclared.

---

# 148. No attention-performance association in Phase54

Do not correlate head metrics with:

```text
RMSE
absolute error
residual.
```

That belongs Phase56.

---

# 149. No cross-seed attention similarity metric in Phase54

No.

Phase57.

---

# 150. No head ablation

Do not zero/disable a head to test prediction impact.

That is a new intervention not part of Phase54.

---

# 151. No attention rollout

No multi-layer attention rollout in Phase54.

Project requirement is attention-map analysis, and rollout would introduce additional assumptions.

Could be future extension only.

---

# 152. No gradient attribution

No:

```text
Integrated Gradients
saliency
gradient×input.
```

Separate method, not Phase54.

---

# 153. Discrepancy taxonomy

`last_query_attention_discrepancies.json`:

```text
PHASE52_NOT_APPROVED
PHASE54_HANDOFF_NOT_READY
RAW_LAST_QUERY_FILE_MISSING
RAW_SHA_MISMATCH
RAW_SHAPE_MISMATCH
RAW_DTYPE_MISMATCH
TARGET_ORDER_MISMATCH
TARGET_DUPLICATE
TARGET_MISSING
LAYER_COUNT_MISMATCH
HEAD_COUNT_MISMATCH
LOOKBACK_MISMATCH
POSITION_MAP_MISMATCH
LAST_QUERY_AXIS_MISMATCH
RAW_VECTOR_NONFINITE
RAW_VECTOR_NEGATIVE
RAW_VECTOR_SUM_MISMATCH
ATTENTION_RENORMALIZED
PHASE52_SUMMARY_RECONSTRUCTION_MISMATCH
LAG_DIRECTION_REVERSED
LAG1_MAPPING_ERROR
RECENCY_REINDEX_ERROR
EXPECTED_LAG_FORMULA_ERROR
ENTROPY_FORMULA_DRIFT
EFFECTIVE_SOURCE_COUNT_FORMULA_ERROR
TOP1_TIE_RULE_DRIFT
RECENT_WINDOW_DEFINITION_DRIFT
TRUNCATED_WINDOW_MISLABELED
LAG50_DEFINITION_ERROR
LAG80_DEFINITION_ERROR
LAG90_DEFINITION_ERROR
LAG_BIN_OVERLAP
LAG_BIN_GAP
LAG_BIN_MASS_SUM_MISMATCH
AVERAGE_PROFILE_SUM_MISMATCH
HEAD_METRICS_SORTED_AS_RANKING
BEST_HEAD_SELECTED
HEAD_CLUSTERING_SCOPE_CREEP
ERROR_CONDITIONING_SCOPE_CREEP
SEED_STABILITY_SCOPE_CREEP
HEAD_ABLATION_ATTEMPT
NEW_ATTENTION_EXTRACTION_ATTEMPT
NEW_TEST_INFERENCE_ATTEMPT
HEATMAP_IMAGE_USED_AS_NUMERIC_SOURCE
RAW_FEATURE_IMPORTANCE_CLAIM
CAUSAL_ATTRIBUTION_CLAIM
SAME_INDEX_HEAD_SEMANTIC_EQUIVALENCE_ASSUMED
REPORT_CASE_CHANGED
OTHER
```

---

# 154. Status model

## PASS

```text
all raw last-query files verified
all target/layer/head vectors complete
Phase52 metrics reconstructed
lag mapping verified
per-target metrics complete
per-head summaries complete
mean temporal profiles complete
lag-bin/recent-mass/coverage metrics complete
top1 frequency complete
report shared-top5 views complete
no head ranking
no error conditioning
Phase55 handoff ready.
```

## PASS_WITH_WARNING

Possible:

```text
final lookback truncates 12h/24h summaries
top1 ties occur
MEAN pooling makes last-query incomplete as prediction view
some profiles highly diffuse
artifact volume large.
```

## FAIL

Examples:

```text
wrong last-query axis
lag mapping reversed
profile sums fail
Phase52 reconstruction mismatch
report cases changed
head ranking performed
new extraction.
```

---

# 155. Phase55 handoff purpose

Phase55 will compare head-level behaviors using standardized Phase54 metrics/profiles.

Phase54 supplies:

```text
per-target head metrics
head aggregate summaries
mean temporal profiles
lag-bin masses
top1 lag distributions.
```

Phase55 should not need to recompute raw last-query basics.

---

# 156. Phase55 handoff schema

`phase55_head_comparison_handoff.json`:

```text
source_phase54_version
source_phase52_version
final_lock_sha256
test_population_sha256
seed_list=[42,123,2026]
lookback
layers
heads
last_query_metrics_long
metric_summary_by_head
profile_by_lag
lag_bin_mass
recent_mass_summary
coverage_radius_summary
top1_lag_frequency
head_order=ARCHITECTURAL_INDEX
head_semantic_alignment_across_seeds=false
head_ranking_performed=false
ready_for_phase55=true
```

---

# 157. Phase56 context handoff

`phase56_error_conditioned_attention_context_handoff.json`:

```text
source_phase54_version
last_query_metrics_long
profile_by_lag
raw_last_query_refs=PHASE52
Phase49_residual_refs
Phase50_regime_refs
Phase51_worst_case_refs
error_conditioning_performed_in_phase54=false
ready_for_phase56_context=true
```

---

# 158. Phase57 context handoff

`phase57_seed_stability_attention_context_handoff.json`:

```text
source_phase54_version
per_seed_head_profiles
per_seed_head_metrics
raw_last_query_refs=PHASE52
same_target_order=true
same_layers_heads=true
same_index_semantic_alignment_not_guaranteed=true
head_matching_not_performed_in_phase54=true
seed_stability_inference_performed=false
ready_for_phase57_context=true
```

---

# 159. Execution sequence

```text
1. Verify Phase52 numerical handoff.
2. Verify Phase53 visual context.
3. Verify 3 raw last-query files/checksums.
4. Verify target ordering, shapes and lag map.
5. Freeze Phase54 analysis contract.
6. Recheck vector probability integrity.
7. Recompute Phase52 metrics and verify.
8. Convert raw-position vectors to recency order for cumulative analyses.
9. Compute effective source count.
10. Compute expected lag and lag SD.
11. Compute Lag50/Lag80/Lag90.
12. Compute non-overlapping lag-bin masses.
13. Build full per-target metrics table.
14. Aggregate metric distributions by seed/layer/head.
15. Compute mean/median/per-lag temporal profiles.
16. Verify average profile sums.
17. Compute layer head-mean profiles.
18. Compute seed overall descriptive profiles.
19. Compute top1 lag frequencies/tie summaries.
20. Generate all-Test profile/concentration/coverage figures.
21. Render deterministic shared ranks1–5 last-query case views.
22. Write findings without head ranking.
23. Write Phase55/56/57 handoffs.
24. Run acceptance/discrepancy audits.
25. Write summary/report/README.
26. Sign off.
```

---

# 160. Recommended pseudocode

```text
p52 = load_phase52_signoff()
assert p52.overall_status in {"PASS","PASS_WITH_WARNING"}

handoff = load_phase54_numerical_handoff()
assert handoff.ready_for_phase54

raw = {
    42: load_last_query_seed42(),
    123: load_last_query_seed123(),
    2026: load_last_query_seed2026()
}

verify_raw_checksums(raw)
verify_same_target_order(raw)
verify_shapes(raw)
verify_position_map()

freeze_phase54_contract()

metrics_rows = []
profile_rows = []
bin_rows = []
top1_rows = []

for seed in [42,123,2026]:

    A = raw[seed]
    # shape [N,Layers,Heads,L]

    verify_all_vectors_probability_valid(A)

    for target_idx in range(N_test):
        target_id = target_order[target_idx]

        for layer in range(N_layers):
            for head in range(N_heads):

                a_raw = A[target_idx,layer,head,:]

                # Raw positions: oldest -> newest
                # Recency order: newest -> oldest
                a_recency = reverse_by_lag_mapping(a_raw)

                lag_steps = frozen_lag_steps_in_raw_order()
                lag_minutes = lag_steps * 10

                entropy = shannon_entropy(a_raw, eps=1e-12)
                norm_entropy = entropy / log(L)
                effective_sources = exp(entropy)

                expected_lag_steps = sum(
                    a_raw * lag_steps
                )

                lag_var = sum(
                    a_raw *
                    (lag_steps - expected_lag_steps)**2
                )

                lag_sd_steps = sqrt(lag_var)

                top1 = resolve_top1_using_phase52_rule(
                    a_raw,
                    tie_rule="NEWEST_SOURCE"
                )

                top5_mass = sum_top_k(a_raw, k=min(5,L))

                recent_masses = compute_fixed_recent_masses(
                    a_raw,
                    lag_steps,
                    windows=[6,36,72,144],
                    truncate_to_L=True
                )

                cumulative_recency = cumsum(a_recency)

                lag50 = first_k(cumulative_recency >= 0.50)
                lag80 = first_k(cumulative_recency >= 0.80)
                lag90 = first_k(cumulative_recency >= 0.90)

                lag_bins = compute_nonoverlap_lag_bins(
                    a_raw,
                    lag_steps,
                    bins=[
                        (1,6),
                        (7,36),
                        (37,72),
                        (73,144),
                        (145,L)
                    ],
                    include_only_supported=True
                )

                assert abs(sum(lag_bins.values()) - 1) <= tolerance

                metrics_rows.append(...)
                bin_rows.append(...)

verify_against_phase52_summary(metrics_rows)

profiles = aggregate_by_lag(
    raw,
    stats=["mean","median","sd","p05","p25","p75","p95"]
)

for seed,layer,head in profiles:
    assert sum(profiles.mean_weight) ≈ 1

layer_head_mean_profiles = compute_layer_head_mean_profiles(raw)
seed_overall_profiles = compute_seed_overall_profiles(raw)

top1_frequency = aggregate_top1_lag_frequency(metrics_rows)
top1_ties = aggregate_top1_tie_summary(metrics_rows)
head_metric_summary = summarize_metrics_by_head(metrics_rows)
recent_mass_summary = summarize_recent_masses(metrics_rows)
coverage_summary = summarize_coverage_radii(metrics_rows)

report_cases = load_phase51_shared_ranks_1_to_5()
assert report_cases unchanged

generate_global_last_query_figures(
    profiles,
    head_metric_summary,
    lag_bins,
    top1_frequency,
    coverage_summary
)

generate_report_case_last_query_plots(
    raw,
    report_cases,
    no_smoothing=True,
    no_head_normalization=True
)

write_phase55_handoff()
write_phase56_context_handoff()
write_phase57_context_handoff()

assert no_new_extraction
assert no_head_ranking
assert no_error_conditioning
assert no_seed_stability_inference
assert no_feature_importance_claim

signoff_phase54()
```

---

# 161. Preflight acceptance checklist

```text
[ ] Phase52 PASS/PASS_WITH_WARNING.
[ ] phase54_ready=true.
[ ] Seed42 last-query NPZ exists.
[ ] Seed123 last-query NPZ exists.
[ ] Seed2026 last-query NPZ exists.
[ ] All raw SHA256s match.
[ ] Same target count.
[ ] Same target IDs/order.
[ ] Same layers.
[ ] Same heads.
[ ] Same lookback.
[ ] Raw dtype correct.
[ ] Position/lag map verified.
[ ] Pooling mode known.
[ ] Phase53 visual context available.
[ ] Report case set frozen from Phase51.
[ ] No extraction/model load required.
```

---

# 162. Integrity acceptance checklist

```text
[ ] Every vector length=L.
[ ] All values finite.
[ ] All values nonnegative within tolerance.
[ ] Every vector sums to ~1.
[ ] No renormalization applied.
[ ] Target order exact.
[ ] Layer/head axis exact.
[ ] Lag1 maps to source L-1.
[ ] LagL maps to source0.
[ ] Recency reorder verified.
```

---

# 163. Phase52 reconstruction acceptance checklist

```text
[ ] Entropy matches.
[ ] Normalized entropy matches.
[ ] Expected lag matches.
[ ] Top1 source/lag matches.
[ ] Top1 weight matches.
[ ] Top1 tie rule matches.
[ ] Top5 mass matches.
[ ] Recent 1h mass matches.
[ ] Recent 6h mass matches.
[ ] Recent 12h mass matches.
[ ] Recent 24h mass matches.
[ ] No tolerance widened after mismatch.
```

---

# 164. Metric acceptance checklist

```text
[ ] Effective source count computed as exp(entropy).
[ ] Lag SD computed from attention-weighted variance.
[ ] Lag50 uses newest→oldest cumulative mass.
[ ] Lag80 uses newest→oldest cumulative mass.
[ ] Lag90 uses newest→oldest cumulative mass.
[ ] Coverage radii within 1..L.
[ ] Non-overlap lag bins defined exactly.
[ ] Bin mass sums to ~1.
[ ] Truncated windows explicitly flagged.
```

---

# 165. Profile acceptance checklist

```text
[ ] Mean profile by lag for every seed/layer/head.
[ ] Median profile by lag.
[ ] Per-lag SD.
[ ] Per-lag p05/p25/p75/p95.
[ ] Mean profile sums ~1.
[ ] Layer head-mean profile generated.
[ ] Layer head-mean profile sums ~1.
[ ] Seed overall descriptive profile generated.
[ ] No cross-seed head averaging.
[ ] No smoothing.
[ ] No per-head max normalization.
```

---

# 166. Top-lag acceptance checklist

```text
[ ] Top1 lag frequency generated.
[ ] Fractions sum to 1 per seed/layer/head.
[ ] Top1 tie fraction generated.
[ ] Newest-source tie rule preserved.
[ ] All supported lags retained in machine table.
[ ] No top-lag interpreted as causal.
```

---

# 167. Figure acceptance checklist

```text
[ ] Lag1/newest shown on left.
[ ] Older history extends rightward.
[ ] Key lag markers derived from mapping.
[ ] Common y-axis within head profile figures.
[ ] No smoothing/interpolation.
[ ] No sorted head order.
[ ] Entropy plot architectural order.
[ ] Expected-lag plot architectural order.
[ ] Lag-bin stacks sum to 1.
[ ] Coverage-radius units clear.
[ ] Report cases exactly shared ranks1–5.
[ ] Case plots use raw last-query vectors.
```

---

# 168. Scope acceptance checklist

```text
[ ] No best head.
[ ] No head clustering.
[ ] No head ablation.
[ ] No error-conditioned attention groups.
[ ] No regime-conditioned attention groups.
[ ] No low-error control group.
[ ] No cross-seed stability score.
[ ] No head matching across seeds.
[ ] No new Test inference.
[ ] No new attention extraction.
[ ] No feature-importance claim.
[ ] No causal claim.
```

---

# 169. Provenance acceptance checklist

```text
[ ] Raw last-query SHA256s stored.
[ ] Target-order SHA stored.
[ ] Position-map SHA stored.
[ ] Phase54 contract/version stored.
[ ] All derived tables reference source version.
[ ] Report cases trace to Phase51.
[ ] Phase55 handoff references exact derived sources.
[ ] Phase56/57 context handoffs reference Phase52 raw source.
```

---

# 170. Acceptance criteria

Phase54 PASS only when:

```text
All three Phase52 full-Test last-query raw attention files are verified by checksum, shape, target ordering and lag mapping.

The last-query vector is treated exactly as A[:,:,L-1,:] over historical source positions.

All vectors are finite, nonnegative within numerical tolerance and sum to approximately 1 without renormalization.

The Phase52 last-query summaries are independently reconstructed from raw vectors and agree under the frozen tolerance.

Raw source positions are mapped to lag steps using H+(L-1-p), with H=1, and recency cumulative analyses correctly reorder newest-to-oldest.

Per-target per-seed per-layer per-head metrics include entropy, normalized entropy, effective source count, expected lag, lag dispersion, top1/top5 concentration, recent-window masses and Lag50/Lag80/Lag90 coverage radii.

Recent 1h/6h/12h/24h summaries explicitly record when the final lookback truncates a named window.

Non-overlapping lag-bin masses cover the supported lookback without overlaps/gaps and sum to approximately 1.

All-Test mean temporal attention profiles are produced for every seed/layer/head and each mean profile sums to approximately 1.

Layer-level head-mean profiles are produced as permutation-invariant descriptive summaries without replacing per-head profiles.

Top1 lag frequency and top1 tie frequency are computed using the frozen newest-source tie rule.

All aggregate metric summaries preserve architectural head order and do not rank heads by desirability.

Deterministic Phase51 shared worst-error ranks 1–5 are used only as illustrative case-level last-query views, not as an error-conditioned statistical comparison.

No new attention extraction, new Test inference, head clustering, head ablation, error conditioning, seed-stability claim, raw-feature importance claim or causal attribution occurs.

Phase55 receives standardized head metrics/profiles, while Phase56 and Phase57 receive context without Phase54 pre-empting their analyses.
```

---

# 171. Failure conditions

Phase54 FAIL if:

```text
wrong raw last-query source is loaded

target order differs across seeds

last-query axis is wrong

lag direction is reversed

position0 is treated as newest

vectors are renormalized to pass probability checks

Phase52 summary cannot be reconstructed

recent-window definitions drift

12h/24h truncated windows are mislabeled as fully covered

Lag50/80/90 accumulate oldest-to-newest

lag bins overlap or leave supported lags uncovered

lag-bin masses do not sum to one

mean profile does not reconstruct a probability distribution

head rows are sorted into a best-to-worst ranking

best head is selected

error/regime-conditioned analysis is performed

cross-seed head matching is performed

same-index head semantics are assumed equivalent across seeds

new attention is extracted

heatmap images are used as numerical source

attention is described as raw-feature importance

attention is treated as causal explanation.
```

---

# 172. Common mistakes

## 172.1 Tính cumulative attention từ position0 đến positionL-1 rồi gọi Lag50 recency

Sai, vì raw position0 là oldest.

Recency coverage phải:

```text
newest → oldest.
```

## 172.2 Expected lag nhỏ thì kết luận head tốt

Sai. Đây chỉ là temporal allocation characteristic.

## 172.3 Entropy thấp thì gọi head quan trọng

Sai.

## 172.4 `recent_24h_mass` khi L=36 nhưng vẫn nói model dùng 24h

Sai. Chỉ có 6h context; phải flag truncation.

## 172.5 Average Head1 qua ba seeds

Không hợp lệ nếu coi là cùng semantic role.

## 172.6 Sắp xếp heads theo expected lag trong table chính

Sai vì tạo implicit ranking.

Giữ architectural order.

## 172.7 Chia high-error/low-error rồi so attention ngay

Để Phase56.

## 172.8 Tính head similarity giữa seeds

Để Phase57.

## 172.9 Đọc bottom heatmap row từ PNG

Sai. Dùng raw Phase52 NPZ.

## 172.10 Smooth mean temporal profile

Sai canonical analysis.

## 172.11 Lag top1 gần 24h rồi kết luận daily seasonality chắc chắn

Quá mạnh. Chỉ có thể nói elevated attention near 24h.

## 172.12 Gọi last-query attention là feature importance

Sai.

---

# 173. Human-readable report structure

`last_query_attention_report.md`:

```text
1. Objective
2. Why last-query attention is analyzed
3. Pooling-dependent interpretation
4. Frozen numerical sources
5. Last-query and lag semantics
6. Raw-source integrity verification
7. Phase52 summary reconstruction
8. Concentration metrics
9. Expected-lag metrics
10. Recent-history cumulative mass
11. Non-overlapping temporal allocation bins
12. Lag50/Lag80/Lag90 coverage radii
13. Mean temporal profiles by head
14. Layer head-mean profiles
15. Top1 lag-frequency patterns
16. Deterministic shared-worst case views
17. Main descriptive findings
18. Why no head winner is selected
19. Why attention is not raw-feature importance
20. Why error-conditioning is deferred to Phase56
21. Why seed-stability is deferred to Phase57
22. Limitations
23. Handoff to head comparison
24. Definition of Done
```

---

# 174. README requirements

`README_LAST_QUERY_ATTENTION.md` explains:

```text
what A[:,:,L-1,:] means
why source positions are historical only
how source position maps to forecast lag
why raw order and recency order differ
what entropy means
what effective source count means
what expected lag means
what recent 1h/6h/12h/24h mass means
what Lag50/Lag80/Lag90 mean
how non-overlapping lag bins work
why mean profiles sum to 1
why heads are not ranked
why same-index heads across seeds may differ semantically
why error-conditioned analysis is deferred
why raw attention is temporal allocation, not feature importance.
```

---

# 175. Summary artifact

`last_query_attention_summary.json`:

```text
version
source_phase52_version
source_phase53_version
final_lock_sha256
test_population_sha256
target_order_sha256
position_map_sha256
seed_list
lookback
layers
heads
pooling
raw_last_query_sha256s
integrity_status
phase52_reconstruction_status
per_target_metric_count
profile_count
lag_bin_status
coverage_radius_status
top1_frequency_status
report_case_count
new_attention_extraction=false
new_test_inference=false
best_head_selected=false
head_clustering=false
error_conditioning=false
seed_stability_inference=false
attention_feature_importance_claim=false
attention_causal_claim=false
phase55_ready
phase56_context_ready
phase57_context_ready
overall_status
```

Do not fabricate runtime counts.

---

# 176. Phase54 sign-off

`phase_54_signoff.json` minimum:

```text
phase=54
phase_name=Last-query attention
version=LAST_QUERY_ATTENTION-v1
source_phase52_version
source_phase53_version
final_lock_sha256
test_population_sha256
target_order_sha256
position_map_sha256
seed_list=[42,123,2026]
lookback_steps
num_layers
num_heads
pooling
last_query_definition=A[:,:,L-1,:]
raw_source_verified
probability_integrity_verified
phase52_summary_reconstructed
lag_mapping_verified
recency_order_verified
per_target_metrics_complete
per_head_summaries_complete
temporal_profiles_complete
layer_head_mean_profiles_complete
lag_bins_complete
coverage_radii_complete
top1_frequency_complete
report_shared_top5_complete
new_attention_extraction=false
new_test_inference=false
best_head_selected=false
head_clustering=false
error_conditioning=false
seed_stability_inference=false
feature_importance_claim=false
causal_claim=false
phase55_ready
phase56_context_ready
phase57_context_ready
warnings
overall_status
created_at
```

---

# 177. Definition of Done

\[
\boxed{
Verified\ Raw\ Last\text{-}Query\ Attention
+
Exact\ Lag\ Mapping
+
Per\text{-}Target\ Head\ Metrics
+
Temporal\ Profiles
+
Recency\ Mass
+
Lag50/80/90
+
Top\text{-}Lag\ Frequencies
+
Layer\ Head\text{-}Mean\ Profiles
+
No\ Head\ Winner
+
No\ Error\ Conditioning
+
Phase55\ Handoff
}
\]

---

# 178. Final status contract

```text
PHASE 54 quantitatively analyzes newest-query attention.

Source:
Phase52 raw last-query NPZ only.

Last query:
A[:,:,L-1,:].

Axes:
[target,layer,head,source].

Lag mapping:
H+(L-1-p)
H=1.

Raw order:
oldest→newest.

Recency analyses:
newest→oldest.

Required per-vector metrics:
entropy
normalized entropy
effective source count
expected lag
lag SD
top1 lag/weight
top5 mass
recent 1h/6h/12h/24h mass
Lag50/Lag80/Lag90.

Required global outputs:
per-head temporal profile
layer head-mean profile
non-overlap lag-bin mass
top1 lag frequency
empirical metric distributions.

Report examples:
Phase51 shared worst ranks1–5 only.

Forbidden:
new extraction
head winner
head clustering
head ablation
error-conditioned comparison
seed-stability inference
cross-seed semantic head assumption
feature-importance claim
causal claim.

After LAST_QUERY_ATTENTION-v1 PASS:
proceed to
PHASE 55 — Head Comparison.
```

---

# 179. Final check

Correct:

```text
verify Phase52 last-query raw files
→ verify lag mapping
→ reconstruct Phase52 summaries
→ compute entropy/expected lag/coverage radii
→ aggregate temporal profiles
→ top1 lag distributions
→ deterministic shared-top5 case views
→ Phase55 handoff
```

Incorrect:

```text
read heatmap PNG
→ estimate bottom-row attention
```

Incorrect:

```text
entropy lowest
→ declare best head
```

Incorrect:

```text
group by worst errors
→ compare attention
```

Incorrect:

```text
compare Head1 across seeds
→ call it stable semantic head
```

Chỉ sau khi:

```text
LAST_QUERY_ATTENTION-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase55_ready = true
```

mới chuyển sang **PHASE 55 — Head Comparison**.
