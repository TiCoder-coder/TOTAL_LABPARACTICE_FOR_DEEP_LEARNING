# PHASE 58 — FINAL TABLES

## Kế hoạch tổng hợp, kiểm định, chuẩn hóa và đóng băng toàn bộ bảng kết quả cuối cùng của coursework từ các artifacts đã được duyệt ở Phase 0–57

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Required model:** Transformer Encoder for regression  
**Required comparison:** LSTM baseline  
**Required interpretability extension:** Attention analysis  
**Primary forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Primary horizon:** `H=1`  
**Final evaluation source:** Frozen Held-Out Test results from Phase47  
**Final attention source:** Frozen attention-analysis artifacts from Phase52–57  
**Phase ID:** `PHASE_58_FINAL_TABLES`  
**Output version:** `FINAL_TABLES-v1`  
**Phase trước:** `Phase_57_Seed-stability_attention_check.md`  
**Phase sau:** `Phase_59_Final_conclusions.md`

---

# 1. Vai trò của Phase 58

Phase58 là **reporting synthesis and governance phase**.

Phase này không còn:

```text
train model
tune hyperparameter
select candidate
select seed
extract attention
redefine cohort
recompute Test after a model change.
```

Phase58 nhận các kết quả đã được khóa từ những phase trước và biến chúng thành một bộ bảng cuối cùng:

```text
đúng source
đúng population
đúng units
đúng precision
đúng seed semantics
đúng model identity
đúng metric semantics
đúng footnotes
đúng provenance
không cherry-pick
không duplicate evidence
không tạo conclusion mới vượt quá source.
```

Mục tiêu:

```text
1. Verify toàn bộ upstream signoffs/handoffs cần cho final reporting.
2. Xây dựng một source-of-truth ledger cho từng con số sẽ xuất hiện trong report.
3. Freeze final table inventory trước khi nhìn vào việc bảng nào “đẹp”.
4. Tách rõ development evidence, Held-Out Test evidence và post-Test diagnostics.
5. Chuẩn hóa model labels và model-order trong mọi bảng.
6. Chuẩn hóa metric labels, units, direction và rounding.
7. Tạo bảng final configuration của Transformer.
8. Tạo bảng final Held-Out Test performance giữa Persistence, tuned LSTM và Final Transformer.
9. Tạo bảng temporal robustness từ rolling-origin evaluation.
10. Tạo bảng prediction/residual/error diagnostics.
11. Tạo bảng error-by-regime và worst-error evidence.
12. Tạo bảng last-query attention.
13. Tạo bảng head comparison.
14. Tạo bảng error-conditioned attention.
15. Tạo bảng seed-stability attention.
16. Tạo appendix tables đầy đủ để mọi main-table value có thể audit.
17. Tạo machine-readable CSV, report-ready Markdown và LaTeX variants.
18. Tạo cross-table consistency audits.
19. Tạo final table catalog và source lineage.
20. Tạo Phase59 handoff chỉ chứa các findings đã được upstream support.
```

Nguyên tắc trung tâm:

\[
\boxed{
Frozen\ Upstream\ Evidence
+
Source\text{-}of\text{-}Truth\ Ledger
+
Transparent\ Table\ Semantics
+
Consistent\ Rounding
+
No\ New\ Selection
+
No\ New\ Scientific\ Analysis
}
\]

---

# 2. Phase58 không phải analysis phase mới

Hard rule:

```text
Phase58 may:
copy
join
reshape
format
audit
derive display-only aggregate already authorized upstream.

Phase58 may not:
invent a new metric
change a cohort
change a threshold
rerun model
rerank candidates on Test
perform a new hypothesis test
perform new head matching
perform new error-attention analysis.
```

Nếu cần một scientific quantity chưa tồn tại upstream:

```text
STOP
→ identify missing upstream artifact
→ do not silently derive a new result in Phase58.
```

---

# 3. Những phép tính được phép trong Phase58

Chỉ được phép các phép **reporting-preserving**:

```text
verify checksum
verify row count
join metadata
recompute a previously defined aggregate for integrity
compute display mean±SD from the exact three already-frozen final seed metrics
round for display
convert lag minutes to readable hours for labels
calculate table row ordering
build footnotes
build cross references.
```

Không được dùng phép tính mới để tạo scientific finding mới.

---

# 4. Upstream evidence taxonomy

Phase58 phải phân loại evidence thành ba lớp.

## 4.1 DEVELOPMENT_EVIDENCE

Ví dụ:

```text
Validation sweeps S1–S19
rolling-origin robustness
candidate synthesis
LSTM tuning.
```

Dùng để giải thích:

```text
model selection process
pre-Test robustness.
```

Không gọi là final generalization result.

## 4.2 HELD_OUT_TEST_EVIDENCE

Ví dụ:

```text
final Test MAE
final Test RMSE
final Test R²
per-seed Transformer Test metrics
Persistence/LSTM Test metrics
final Test prediction spread.
```

Đây là primary final performance evidence.

## 4.3 POST_TEST_DIAGNOSTIC_EVIDENCE

Ví dụ:

```text
residual analysis
error-by-regime
worst-error analysis
attention heatmaps
last-query attention
head comparison
error-conditioned attention
seed-stability attention.
```

Dùng để:

```text
understand
characterize
discuss limitations.
```

Không được dùng để quay lại chọn model.

---

# 5. Final tables phải ghi rõ evidence class

Mỗi table manifest row phải có:

```text
DEVELOPMENT_EVIDENCE
HELD_OUT_TEST_EVIDENCE
POST_TEST_DIAGNOSTIC_EVIDENCE
MIXED_WITH_EXPLICIT_SEPARATION.
```

Nếu một bảng chứa cả development và Test:

```text
phải có section/panel rõ ràng
```

hoặc tốt hơn:

```text
tách thành hai bảng.
```

---

# 6. Upstream hard gate

Required upstream signoffs:

```text
phase_44_signoff.json
phase_45_signoff.json
phase_46_signoff.json
phase_47_signoff.json
phase_48_signoff.json
phase_49_signoff.json
phase_50_signoff.json
phase_51_signoff.json
phase_52_signoff.json
phase_53_signoff.json
phase_54_signoff.json
phase_55_signoff.json
phase_56_signoff.json
phase_57_signoff.json.
```

For each:

```text
PASS
or
PASS_WITH_WARNING.
```

Phase57:

```text
phase58_ready=true.
```

Any required upstream:

```text
FAIL
or
missing
```

means:

```text
BLOCK Phase58 final signoff.
```

---

# 7. Source-of-truth hierarchy

For final tables:

```text
Final model config:
Phase45 final lock.

Three-seed run identity/checkpoints:
Phase46.

Held-Out Test metrics:
Phase47.

Prediction spread:
Phase48.

Residuals:
Phase49.

Error regimes:
Phase50.

Worst cases:
Phase51.

Raw attention:
Phase52.

Attention figures:
Phase53.

Last-query metrics:
Phase54.

Head comparison:
Phase55.

Error-conditioned attention:
Phase56.

Seed-stability attention:
Phase57.

Rolling-origin development robustness:
Phase44.

LSTM tuning lineage:
Phase43.
```

Later phase artifacts override duplicated earlier summaries only if they are explicitly derived from the same frozen source and not scientifically changed.

---

# 8. Source precedence rule

If the same field appears in multiple artifacts:

```text
1. authoritative phase-specific source
2. later handoff/summary as verification
3. report-ready table is derived copy only.
```

Example:

```text
Final Test RMSE
→ authoritative Phase47 metric artifact
not Phase48 narrative
not Phase57 report.
```

---

# 9. No value may come only from narrative prose

Every final table cell containing a scientific value must trace to:

```text
CSV
JSON
NPZ-derived frozen summary
or other machine-readable upstream artifact.
```

Human-readable `.md` report text is secondary evidence only.

---

# 10. Final table source ledger

Create:

```text
final_table_source_ledger.csv
```

One row per scientific table field or field family.

Required columns:

```text
table_id
field_id
display_label
source_phase
source_version
source_artifact
source_column
source_filter
aggregation
population
unit
evidence_class
rounding_rule
footnote_rule
status.
```

---

# 11. Cell-level lineage for critical tables

For:

```text
FT02 Final Test Performance
FT03 Rolling-Origin Robustness
FT06 Last-Query Attention Summary
FT08 Error-Conditioned Attention
FT09 Seed-Stability Attention
```

create:

```text
final_table_cell_lineage.csv
```

with one row per rendered numeric cell where practical.

Fields:

```text
table_id
row_key
column_key
display_value
raw_value
source_artifact
source_row_key
source_field
aggregation
status.
```

---

# 12. No manually typed final metric

Main scientific values should be programmatically generated from source ledger.

Manual typing is allowed only for:

```text
caption
footnote
label
interpretive note.
```

---

# 13. Final table inventory phải freeze trước formatting

Create:

```text
final_table_inventory.json
```

before final table rendering.

This prevents:

```text
see weak result
→ omit table
see strong result
→ add new table.
```

---

# 14. Canonical main-table inventory

Phase58 freezes the following main-report-ready tables:

```text
FT01 — Final Experimental and Model Configuration
FT02 — Final Held-Out Test Performance
FT03 — Rolling-Origin Temporal Robustness
FT04 — Final Prediction and Residual Diagnostics
FT05 — Error-by-Regime and Worst-Error Summary
FT06 — Last-Query Temporal Attention Summary
FT07 — Within-Seed Head Comparison Summary
FT08 — Error-Conditioned Attention Summary
FT09 — Seed-Stability Attention Summary
FT10 — Evidence and Limitation Summary
```

No table may be removed merely because its result is unfavorable.

---

# 15. Appendix table inventory

Canonical appendix:

```text
FA01 — Per-Seed Final Test Metrics
FA02 — Rolling-Origin Fold-Level Metrics
FA03 — Residual Distribution Details
FA04 — Full Error-by-Regime Results
FA05 — Shared Worst-Error Cases
FA06 — Full Last-Query Head Metrics
FA07 — Full Head Pairwise Comparison
FA08 — Full Error-Conditioned Attention Coefficients
FA09 — Head Matching and Ambiguity Details
FA10 — Attention Seed-Stability Detailed Metrics
FA11 — Table Provenance and Population Audit
FA12 — Upstream Warnings and Reporting Caveats
```

Appendix count can be extended only for a genuine formatting need, not result selection.

---

# 16. Main vs appendix principle

Main tables answer:

```text
What was the final model?
How well did it perform?
Was it robust temporally?
Where did it fail?
What temporal attention behavior did it learn?
Were heads diverse?
Did attention differ with error?
Was attention stable across seeds?
```

Appendix provides:

```text
full detail
full heads
full regimes
full seed-pairs
full case lists
full provenance.
```

---

# 17. Final figure inventory is also frozen as context

Phase58 is named Final Tables, but report tables must reference a stable set of already-created figures.

Create:

```text
final_figure_inventory.json
```

No new scientific plotting required.

Recommended figure references:

```text
FF01 — Final prediction time-series view from Phase48
FF02 — Residual diagnostic figure from Phase49
FF03 — Error-by-regime figure from Phase50
FF04 — Attention heatmap from Phase53
FF05 — Last-query temporal profile from Phase54
FF06 — Head comparison matrix from Phase55
FF07 — Error-conditioned attention profile from Phase56
FF08 — Seed-stability attention figure from Phase57.
```

Phase58 selects by predeclared source role, not by which plot looks most favorable.

---

# 18. Model label contract

Canonical display labels:

```text
Persistence Baseline
Tuned LSTM Baseline
Final Transformer — Seed 42
Final Transformer — Seed 123
Final Transformer — Seed 2026
Final Transformer — Three-Seed Summary.
```

If upstream artifact uses other internal IDs:

```text
map them through final_model_label_map.csv.
```

---

# 19. Transformer three-seed summary is not an ensemble

Critical footnote:

```text
“Three-Seed Summary” reports the descriptive mean ± sample SD of metrics
across the three independently trained final Transformer runs.
It is not an ensemble prediction.
```

Do not call it:

```text
ensemble
average model
combined prediction.
```

---

# 20. No best-seed row

All three seed rows remain official.

Do not:

```text
keep only lowest-RMSE seed
bold best seed
call one seed primary.
```

---

# 21. Baseline order contract

Every performance comparison table:

```text
1. Persistence Baseline
2. Tuned LSTM Baseline
3. Final Transformer seed rows
4. Final Transformer Three-Seed Summary.
```

Order is methodological, not metric rank.

---

# 22. No sort-by-performance

No table sorted by:

```text
RMSE ascending
MAE ascending
R² descending.
```

This avoids implicit Test-based ranking.

---

# 23. Metric contract

Primary final metrics:

```text
MAE [Wh]
RMSE [Wh]
R² [dimensionless].
```

Selection metric historically:

```text
Validation RMSE [Wh].
```

But FT02 is Held-Out Test reporting, not selection.

---

# 24. Metric formulas remain Phase12 definitions

Do not redefine.

Final table captions must use:

```text
MAE
RMSE
R²
```

with raw-unit evaluation.

---

# 25. R² may be negative

Never clamp:

```text
R² < 0
```

to zero.

Negative R² is valid evidence.

---

# 26. No MAPE added

MAPE was not a locked shared metric.

Do not introduce it in Phase58 because it is common in forecasting.

---

# 27. No new accuracy percentage

No.

---

# 28. Units contract

```text
Appliances target: Wh
MAE: Wh
RMSE: Wh
residual: Wh
absolute error: Wh
expected attention lag: minutes or hours, clearly labeled
Wasserstein attention distance: minutes
JSD: dimensionless
cosine: dimensionless
Spearman rho: dimensionless
Cliff's delta: dimensionless
attention mass: 0–1 or % only if explicitly converted for display.
```

---

# 29. Percentage display rule

If attention mass or cohort share is displayed as percentage:

```text
raw value remains 0..1 in machine CSV
display table may multiply by 100
header must include (%).
```

Do not mix fraction and percent in same column.

---

# 30. Display precision contract

Phase58 freezes report rounding.

## 30.1 Counts

```text
integer
no decimal.
```

## 30.2 MAE/RMSE/error values in Wh

```text
2 decimal places.
```

## 30.3 R²

```text
3 decimal places.
```

## 30.4 Correlation / JSD / cosine / Cliff's delta / attention mass

```text
3 decimal places.
```

## 30.5 Wasserstein / expected lag in minutes

```text
1 decimal place
```

unless exact integer lag is specifically reported.

## 30.6 Hours

```text
2 decimal places
```

when converted from minutes.

## 30.7 Percentages

```text
1 decimal place.
```

---

# 31. Raw precision is never discarded

Machine-readable tables store:

```text
full available precision.
```

Only report-ready:

```text
Markdown
LaTeX
```

use display rounding.

---

# 32. Never aggregate rounded values

Correct:

```text
full precision metrics
→ mean/SD
→ round display.
```

Incorrect:

```text
round each seed RMSE
→ mean rounded numbers.
```

---

# 33. Negative-zero display rule

If display rounding produces:

```text
-0.000
```

render:

```text
0.000.
```

Raw value remains unchanged.

---

# 34. Mean ± SD semantics

For final Transformer metrics:

\[
\bar{x}
=
\frac{x_{42}+x_{123}+x_{2026}}{3}
\]

Sample SD:

\[
s
=
\sqrt{
\frac{
\sum_i(x_i-\bar{x})^2
}{
3-1
}
}.
\]

Use:

```text
ddof=1.
```

---

# 35. Three-seed SD is descriptive only

Footnote:

```text
SD reflects variability across the three predefined final seeds
and is not a confidence interval.
```

---

# 36. Baselines do not receive fake ±SD

If Persistence or tuned LSTM upstream has only one official final Test realization:

```text
report single value.
```

Do not invent:

```text
±0.
```

If upstream actually evaluated multiple frozen seeds, use exact upstream protocol only.

---

# 37. FT01 — Final Experimental and Model Configuration

Purpose:

> Freeze the exact final scientific configuration used for the official Transformer results.

Source:

```text
Phase45 final_model_scientific_config.json
Phase45 final_training_recipe.json
Phase45 final_feature_contract.json
Phase45 final_boundary_contract.json
Phase45 final_seed_contract.json.
```

Evidence class:

```text
METHOD / LOCKED_CONFIG
```

not a performance table.

---

# 38. FT01 required fields

Recommended two-column table:

```text
Category
Final setting.
```

Rows:

```text
Task formulation
Forecast horizon
Sampling interval
Final feature variant
Time-feature setting
Target scaling
Lookback
Boundary protocol
Window population policy
Input feature count
Transformer d_model
Attention heads
Head dimension
Encoder layers
FFN dimension
Activation
Dropout
Positional encoding
Normalization style
Pooling
Loss
Optimizer
Learning rate
Weight decay
Batch size
Gradient clipping
RevIN
Final refit epoch count
Final seeds
Final training region
Test population policy.
```

Use exact runtime-selected values from Phase45.

No value is hard-coded from B0 unless B0 actually remained final.

---

# 39. FT01 must distinguish development settings vs final refit recipe

Important:

```text
Development max_epochs/patience
```

may differ conceptually from:

```text
Final refit fixed epoch count
no validation
no early stopping.
```

FT01 should clearly display the **final official refit recipe**.

Optional footnote can mention development selection separately.

---

# 40. FT02 — Final Held-Out Test Performance

This is the **primary final performance table**.

Evidence class:

```text
HELD_OUT_TEST_EVIDENCE.
```

Source:

```text
Phase47 final Test metrics.
```

---

# 41. FT02 columns

Canonical:

```text
Model
Seed / summary
MAE (Wh)
RMSE (Wh)
R²
Evaluation population
Notes.
```

Rows:

```text
Persistence Baseline
Tuned LSTM Baseline
Final Transformer — Seed42
Final Transformer — Seed123
Final Transformer — Seed2026
Final Transformer — Three-Seed Summary.
```

---

# 42. FT02 fairness rule

Direct baseline comparison is allowed only if:

```text
same final Test target IDs
same raw y_true
same metric version
same horizon
same population policy.
```

If not:

```text
do not place metrics side-by-side as if directly comparable
→ use matched-target view or explicit warning.
```

---

# 43. FT02 population fingerprint

Table metadata must include:

```text
FINAL_TEST_POP_SHA256
METRICS-v1
H=1
WB0
```

or exact final boundary policy if formally amended.

---

# 44. FT02 no best-value bolding by default

Recommended:

```text
no bold lowest RMSE
no color-coded winner.
```

The table reports Held-Out Test evidence without creating post-Test selection.

---

# 45. FT02 allowed comparison deltas

Only include deltas if they were already frozen upstream, e.g.:

```text
Transformer vs Persistence ΔRMSE
Transformer vs LSTM ΔRMSE.
```

If not already an upstream official result:

```text
do not add a new headline delta in Phase58.
```

A simple descriptive difference can be computed for integrity but should not create a new finding.

---

# 46. FT03 — Rolling-Origin Temporal Robustness

Purpose:

> Show pre-Test temporal robustness evidence that supported final model lock.

Evidence class:

```text
DEVELOPMENT_EVIDENCE.
```

Source:

```text
Phase44 rolling-origin outputs
Phase45 final source candidate.
```

---

# 47. FT03 canonical columns

```text
Model / candidate
RO1 RMSE (Wh)
RO2 RMSE (Wh)
RO3 RMSE (Wh)
Pooled RMSE (Wh)
Mean fold RMSE (Wh)
Fold RMSE SD (Wh)
Worst-fold RMSE (Wh)
Role.
```

Rows:

```text
Frozen Transformer candidates from Phase42
Tuned LSTM
Persistence
```

as actually available upstream.

---

# 48. FT03 primary metric semantics

Primary robustness ranking upstream used:

```text
pooled outer-fold RMSE.
```

Table caption must explicitly say:

```text
Pooled RMSE is the primary Phase44 robustness criterion;
mean fold RMSE is secondary and is not equivalent to pooled RMSE.
```

---

# 49. FT03 final Transformer marker

Mark the Phase45-selected source candidate with:

```text
Final-source candidate
```

based on Phase45/44 handoff.

Do not choose row based on Phase58 reinspection.

---

# 50. FT04 — Final Prediction and Residual Diagnostics

Purpose:

> Compactly summarize final prediction behavior after Test evaluation.

Evidence class:

```text
POST_TEST_DIAGNOSTIC_EVIDENCE.
```

Sources:

```text
Phase48
Phase49.
```

---

# 51. FT04 possible two-panel structure

## Panel A — Prediction seed variability

Rows:

```text
prediction spread summary
```

Fields from Phase48 such as frozen:

```text
prediction range summary
prediction SD summary
cross-seed prediction agreement
```

only if upstream defined them.

## Panel B — Residual diagnostics

Rows/fields from Phase49:

```text
mean residual
median residual
residual SD
MAE-compatible absolute error summary
underprediction share
overprediction share
residual quantiles.
```

Do not invent new residual statistics.

---

# 52. FT04 seed treatment

Prefer:

```text
per-seed rows
+
three-seed descriptive summary only where upstream semantics support it.
```

Do not average residuals across seeds into an “ensemble residual”.

---

# 53. FT05 — Error-by-Regime and Worst-Error Summary

Purpose:

> Show where errors are concentrated and whether some data regimes are especially difficult.

Evidence class:

```text
POST_TEST_DIAGNOSTIC_EVIDENCE.
```

Source:

```text
Phase50
Phase51.
```

---

# 54. FT05 Panel A — Error by regime

Use frozen Phase50 regime families:

```text
target-level
EXTREME_HIGH
change magnitude
direction
time of day
weekday/weekend
```

Do not redefine thresholds.

---

# 55. FT05 Panel A columns

Canonical:

```text
Regime family
Regime
N
Share (%)
MAE (Wh)
RMSE (Wh)
R² if upstream applicable
Notes.
```

If Phase50 reports seed-specific values, choose a predeclared representation:

```text
layer? no
model seed-specific or three-seed descriptive summary
```

according to exact upstream artifact.

No new cross-seed ensemble metric.

---

# 56. FT05 Panel B — Worst-error concentration

Use Phase51 frozen:

```text
Top20 SAE share
Top20 SSE share
Top1% SSE share
Top5% SSE share
Top10% SSE share
shared worst-case overlap
```

only if upstream output contains them.

---

# 57. FT05 case-study footnote

Worst cases:

```text
remain valid Test cases
were not removed
were not used to recompute a better RMSE.
```

---

# 58. FT06 — Last-Query Temporal Attention Summary

Purpose:

> Provide a compact quantitative summary of where attention is allocated in historical time.

Evidence class:

```text
POST_TEST_DIAGNOSTIC_EVIDENCE.
```

Source:

```text
Phase54.
```

---

# 59. FT06 row unit

Primary:

```text
seed × layer × head
```

in main-ready compact table.

If table becomes too large:

```text
main table uses layer head-mean summary
appendix FA06 contains all heads.
```

Main table selection rule must be frozen before rendering.

Recommended main table:

```text
layer head-mean
for seed42,123,2026
+
cross-seed descriptive row if Phase54/57 upstream supports it.
```

---

# 60. FT06 columns

Canonical core:

```text
Seed
Layer
Normalized entropy
Expected lag (min)
Recent 1h mass
Recent 6h mass
Top-5 mass
Lag80 (min)
Interpretation scope.
```

No `importance` label.

---

# 61. FT06 attention footnote

Mandatory:

```text
Attention values describe temporal token-to-token allocation.
They are not direct raw-feature importance and do not establish causal contribution.
```

---

# 62. FT06 pooling footnote

If final pooling is:

```text
LAST_STEP
```

state:

```text
last-query corresponds to newest encoded token used by the regression head.
```

If:

```text
MEAN
```

state:

```text
last-query is only one view because prediction pools all encoded positions.
```

---

# 63. FT07 — Within-Seed Head Comparison Summary

Purpose:

> Show whether heads within a layer have diverse or near-similar temporal allocation behavior.

Source:

```text
Phase55.
```

Evidence class:

```text
POST_TEST_DIAGNOSTIC_EVIDENCE.
```

---

# 64. FT07 recommended level

Main table should avoid all pair rows.

Use:

```text
layer_head_diversity_summary.csv
```

with one row per:

```text
seed × layer.
```

---

# 65. FT07 columns

Canonical:

```text
Seed
Layer
Head count
Mean pairwise JSD
Median pairwise JSD
Mean pairwise Wasserstein (min)
Mean pairwise |Δ expected lag| (min)
Mean pairwise top1 TVD
Mean pairwise cosine
Notes.
```

No diversity score.

---

# 66. FT07 footnote

Mandatory:

```text
Similarity in attention allocation does not prove functional redundancy,
because value projections and downstream output projections may differ.
```

---

# 67. FT08 — Error-Conditioned Attention Summary

Purpose:

> Show whether attention behavior co-varies with realized Test forecast error.

Source:

```text
Phase56.
```

Evidence class:

```text
POST_TEST_DIAGNOSTIC_EVIDENCE.
```

---

# 68. FT08 primary level

Main report prioritizes permutation-invariant:

```text
layer head-mean results
```

from Phase56.

Per-head full coefficients go to:

```text
FA08.
```

---

# 69. FT08 primary columns

Recommended:

```text
Seed
Layer
Attention metric
Spearman ρ with |error|
HIGH vs LOW median difference
HIGH vs LOW Cliff's δ
HIGH vs LOW profile JSD
HIGH vs LOW profile Wasserstein (min)
Direction note.
```

Rows for the six frozen core attention metrics.

If profile distance is layer-level rather than metric-specific:

```text
use separate Panel B.
```

Do not force incompatible fields into one row.

---

# 70. FT08 panel design

Recommended:

## Panel A — Metric-level error association

```text
Seed
Layer
Metric
Spearman ρ(|error|, metric)
High-low median delta
Cliff's delta.
```

## Panel B — Temporal profile shift

```text
Seed
Layer
JSD(high,low)
L1(high,low)
Wasserstein(high,low) [min].
```

---

# 71. FT08 error cohort footnote

Mandatory:

```text
HIGH/LOW are Test-relative diagnostic cohorts defined from frozen rank-based realized errors.
They are not deployment regimes and were not used for retuning.
```

---

# 72. FT08 causal footnote

Mandatory:

```text
Error-attention relationships are associative diagnostics and do not establish causality.
```

---

# 73. FT09 — Seed-Stability Attention Summary

Purpose:

> Provide final evidence that attention interpretation was checked across all three official final seeds.

Source:

```text
Phase57.
```

Evidence class:

```text
POST_TEST_DIAGNOSTIC_EVIDENCE.
```

---

# 74. FT09 recommended multi-panel structure

## Panel A — Permutation-invariant layer stability

```text
Layer
Mean pairwise JSD
Max pairwise JSD
Mean pairwise Wasserstein (min)
Max pairwise Wasserstein (min)
Mean pairwise cosine.
```

## Panel B — Head-matching robustness

```text
Layer
Head count
Cycle consistency fraction
JSD/Wasserstein matching agreement fraction
Ambiguous mapping warning
Mean matched-head pairwise JSD
Mean matched-head Wasserstein.
```

## Panel C — Error-effect seed consistency

```text
Layer
Metric
Seed42 effect
Seed123 effect
Seed2026 effect
Sign agreement.
```

Use exact upstream fields.

---

# 75. FT09 head-matching footnote

Mandatory:

```text
Same-index heads were not assumed semantically equivalent across seeds.
Head-level comparison used the frozen permutation-aware matching protocol from Phase57.
```

---

# 76. FT09 no stable/unstable categorical label

Report exact continuous stability evidence.

Do not add:

```text
Stable
Unstable
```

based on a Phase58 threshold.

---

# 77. FT10 — Evidence and Limitation Summary

Purpose:

> Give the final report a compact traceability table linking each major claim to its evidence class and limitation.

This is not a scientific metric table.

---

# 78. FT10 columns

```text
Question
Primary evidence
Evidence class
What can be concluded
What cannot be concluded
Source phase.
```

Rows:

```text
Does Transformer generalize on held-out Test?
Does it outperform/compare with LSTM?
Is performance temporally robust?
Where are largest errors?
Which regimes are difficult?
What temporal lags receive attention?
Are heads diverse?
Does attention co-vary with error?
Is attention stable across seeds?
Can attention be interpreted causally?
```

---

# 79. FT10 wording rule

Only use findings supported by frozen upstream artifacts.

If upstream is inconclusive:

```text
state inconclusive / mixed
```

rather than forcing a positive result.

---

# 80. Appendix FA01 — Per-Seed Final Test Metrics

Contains exact:

```text
seed42
seed123
seed2026
```

plus full-precision:

```text
MAE
RMSE
R²
target count
population fingerprint
checkpoint SHA
final lock SHA.
```

No baseline needed if FT02 already has it.

---

# 81. Appendix FA02 — Rolling-Origin Fold-Level Metrics

All Phase44 folds/candidates needed for audit.

No reranking.

---

# 82. Appendix FA03 — Residual Distribution Details

Full Phase49 final tables.

No new metrics.

---

# 83. Appendix FA04 — Full Error-by-Regime Results

All frozen Phase50 regime rows.

No report-space-based omission.

---

# 84. Appendix FA05 — Shared Worst-Error Cases

Use Phase51 deterministic:

```text
W2 shared top20
shared all-under
shared all-over
```

as applicable.

Columns can include:

```text
timestamp
actual
seed predictions
residuals
shared hardness
regimes.
```

---

# 85. Appendix FA06 — Full Last-Query Head Metrics

All:

```text
seed × layer × head
```

Phase54 metrics.

No head ranking.

---

# 86. Appendix FA07 — Full Head Pairwise Comparison

All Phase55 pair metrics:

```text
JSD
Wasserstein
cosine
expected-lag differences
top1 TVD.
```

Architectural order.

---

# 87. Appendix FA08 — Full Error-Conditioned Attention Coefficients

All Phase56:

```text
seed × layer × head × core metric
```

Spearman and high/low effects.

No sorting by effect magnitude.

---

# 88. Appendix FA09 — Head Matching and Ambiguity Details

Phase57:

```text
canonical mapping
costs
assignment gaps
cycle consistency
Wasserstein sensitivity.
```

---

# 89. Appendix FA10 — Attention Seed-Stability Details

Full Phase57 matched-head/per-target summaries as reportable.

---

# 90. Appendix FA11 — Provenance and Population Audit

Required fields:

```text
table ID
population
target count
population SHA
source phase
source artifact
source checksum
metric version
final lock SHA.
```

This table is critical for reproducibility.

---

# 91. Appendix FA12 — Upstream Warnings and Reporting Caveats

Collect all:

```text
PASS_WITH_WARNING
```

warnings from required upstream phases.

Examples may include:

```text
small cohort
matching ambiguity
MEAN pooling caveat
seed variability
lookback truncation.
```

Do not fabricate warnings not present upstream.

---

# 92. Cross-table model identity contract

Every row labeled Final Transformer must refer to the exact:

```text
FINAL_MODEL_LOCK_SHA256.
```

No Phase21 B0 result may accidentally appear as final.

---

# 93. Cross-table Test population contract

All Held-Out Test and post-Test diagnostic tables must reference:

```text
FINAL_TEST_POP-v1
```

or exact final population fingerprint.

No mixture with:

```text
rolling-origin outer folds
original validation
strict WB1 sensitivity
unless table explicitly says so.
```

---

# 94. WB0/WB1 reporting

Primary final Test protocol follows:

```text
final locked boundary protocol
```

expected to be WB0 unless formal amendment.

If WB1 sensitivity is reported:

```text
label as sensitivity only
```

and do not mix WB1 values into primary final table.

---

# 95. No population-mismatch averaging

Do not compute mean across metrics from different target populations.

---

# 96. No metric averaging across samples at table stage

Phase12/47 already define global split metrics.

Do not reconstruct RMSE by averaging per-batch/per-regime RMSE.

---

# 97. Pooled vs macro safeguard

FT03:

```text
pooled rolling-origin RMSE
```

must come from Phase44 authoritative pooled output.

Do not average fold RMSE and relabel it pooled RMSE.

---

# 98. R² aggregation safeguard

Do not average:

```text
per-regime R²
```

to obtain overall R².

Use upstream overall metric only.

---

# 99. Seed metric summary safeguard

Three-seed summary:

```text
mean of three seed-level metrics
```

is allowed as descriptive run variability.

It is not:

```text
metric of mean prediction.
```

---

# 100. No ensemble reconstruction

Hard:

```text
do not average seed predictions
→ calculate new ensemble RMSE
```

because no ensemble was predeclared.

---

# 101. No “best of three seeds” summary

No:

```text
min RMSE
max R²
best seed.
```

as headline final result.

Range can be reported descriptively only if upstream/Phase47 already uses it.

---

# 102. No significance test between models in Phase58

Do not run:

```text
paired t-test
Wilcoxon
Diebold-Mariano
bootstrap comparison
```

unless such test was already a frozen upstream artifact.

Phase58 cannot introduce it.

---

# 103. No new confidence intervals

No.

---

# 104. Table ordering rules

Canonical:

```text
models: methodological order
seeds: 42,123,2026
layers: Layer1...LayerN
heads: Head1...HeadH
seed pairs: 42–123, 42–2026, 123–2026
error deciles: 1...10
regimes: Phase50 frozen order
worst cases: Phase51 frozen rank
matched groups: seed42-anchor canonical order.
```

---

# 105. Table sorting is part of scientific presentation

Do not sort rows by:

```text
metric magnitude
effect size
correlation strength
error severity
```

unless ranking itself is the frozen upstream object, such as:

```text
Phase51 worst-case rank.
```

---

# 106. Missing-value display policy

Machine tables:

```text
NaN / null with reason field.
```

Report tables:

```text
N/A
```

with footnote.

Never replace missing value by:

```text
0.
```

---

# 107. Not-applicable policy

Examples:

```text
12h mass when lookback <72
24h mass when lookback <144
signed comparison if one sign group empty.
```

Display:

```text
N/A
```

with exact reason.

---

# 108. Warning propagation

Any upstream warning relevant to a table must appear in:

```text
table footnote
or
FA12.
```

Critical warnings should appear in both.

---

# 109. Caption contract

Every final table caption should answer:

```text
What is being reported?
Which population?
Which evidence class?
What are the units?
What aggregation is used?
What is the main caveat?
```

---

# 110. Footnote contract

Footnotes are mandatory when relevant for:

```text
three-seed summary != ensemble
development vs Test distinction
pooled vs macro distinction
error cohort diagnostic nature
attention != feature importance
attention != causality
head matching semantics
selection-conditioned worst cases
three-seed SD is descriptive.
```

---

# 111. Table title style

Recommended concise academic format:

```text
Table FT02. Final held-out test performance.
```

No promotional wording like:

```text
Superior Transformer performance.
```

---

# 112. No result claim in table title

Titles describe content only.

---

# 113. Report-ready Markdown format

Generate:

```text
tables/markdown/FT01_....md
...
```

Use standard Markdown tables.

Do not embed source hashes in main visual columns; hashes go into metadata/appendix.

---

# 114. Report-ready LaTeX format

Generate:

```text
tables/latex/FT01_....tex
...
```

Use:

```text
booktabs-compatible
```

format if available in report environment.

Do not rely on custom packages unless documented.

---

# 115. Machine-readable CSV format

Generate:

```text
tables/csv/FT01_....csv
...
```

Full precision.

---

# 116. Metadata JSON per table

For every table:

```text
tables/metadata/FTxx_....json
```

Required:

```text
table_id
title
evidence_class
source_phases
source_artifacts
population
population_sha
metric_version
raw_precision
display_precision
row_order_rule
footnotes
warnings
status.
```

---

# 117. Table fingerprint

For every final table:

```text
source fingerprint
machine CSV SHA256
Markdown SHA256
LaTeX SHA256.
```

Store in:

```text
final_table_checksums.json.
```

---

# 118. Final tables manifest

Create:

```text
final_tables_manifest.json
```

containing all tables and checksums.

---

# 119. Report table catalog

Create:

```text
FINAL_TABLE_CATALOG.md
```

For each table:

```text
ID
purpose
evidence class
source phase
main/appendix
report insertion recommendation
important footnotes.
```

No result interpretation beyond source support.

---

# 120. Table-to-claim matrix

Create:

```text
table_claim_traceability.csv
```

Example:

```text
Claim ID
Allowed claim
Supporting table
Supporting upstream finding
Limitation.
```

This supports Phase59 conclusions.

---

# 121. Claim types

Canonical:

```text
C1 final Test performance
C2 Transformer vs LSTM comparison
C3 temporal robustness
C4 error concentration
C5 regime-dependent error
C6 temporal attention allocation
C7 head diversity
C8 error-attention association
C9 seed attention stability
C10 interpretability limitation.
```

No unsupported claim.

---

# 122. Phase59 must not infer from formatting

Phase58 handoff should carry:

```text
raw evidence
allowed claim scope
caveats.
```

Not:

```text
bold row → winner
color → importance.
```

---

# 123. Display highlighting policy

Canonical main tables:

```text
no metric-cell color
no winner color
no best-value bold.
```

Allowed bold:

```text
table headings
section labels
Final Transformer model label
```

only if not based on Test performance.

Preferred:

```text
no row-specific emphasis.
```

---

# 124. Arrow policy

Avoid:

```text
↑
↓
```

in metric headers unless explained.

Use explicit:

```text
Lower is better
Higher is better
```

only in notes if needed.

No arrows required.

---

# 125. Decimal alignment

LaTeX may use numeric alignment if environment supports it.

Markdown uses consistent decimal places.

---

# 126. Long-table handling

If FT06–FT09 exceed main-report readability:

```text
main = aggregated panel
appendix = full rows.
```

Aggregation must already exist upstream.

Do not create a new selective subset.

---

# 127. Table width governance

Main tables should avoid more than roughly:

```text
7–9 visible numeric/text columns
```

where possible.

Use panels rather than unreadably wide tables.

This is formatting guidance, not scientific filtering.

---

# 128. Panel design

Allowed:

```text
Panel A
Panel B
Panel C
```

when:

```text
same scientific topic
different metric family
same source phase.
```

---

# 129. Do not combine unrelated evidence just to save space

Example:

```text
Test RMSE
and
attention JSD
```

should not be in one row/table.

---

# 130. FT02 validation audit

Required programmatic checks:

```text
all three Transformer seed metrics exactly match Phase47
three-seed mean recomputed full precision
three-seed sample SD ddof=1
no ensemble prediction
same Test population
baseline population match
same metric version
display rounding only after aggregate.
```

---

# 131. FT03 validation audit

```text
pooled RMSE from Phase44
fold metrics exact
candidate IDs exact
final-source marker matches Phase45
no Test metric present
no reranking.
```

---

# 132. FT04 validation audit

```text
prediction spread values from Phase48
residual convention y_true-y_pred
no ensemble residual
no row drops
all units Wh
no new diagnostics.
```

---

# 133. FT05 validation audit

```text
regime labels exactly Phase50
thresholds unchanged
worst-case ranks exactly Phase51
error concentration exact
no worst-case deletion
no new cohort.
```

---

# 134. FT06 validation audit

```text
Phase54 source exact
attention metrics not recalculated using images
lag units correct
layer/head semantics correct
no head ranking
pooling caveat correct.
```

---

# 135. FT07 validation audit

```text
Phase55 layer diversity exact
no composite diversity score
no functional redundancy claim
architectural ordering retained.
```

---

# 136. FT08 validation audit

```text
Phase56 core metric set exact
error cohorts unchanged
Spearman values exact
Cliff's delta exact
profile-distance fields exact
no p-value fishing
no causal wording
no best head.
```

---

# 137. FT09 validation audit

```text
Phase57 layer stability exact
head matching fingerprint exact
same-index heads not assumed aligned
cycle consistency exact
matching ambiguity propagated
no stability threshold
no best seed.
```

---

# 138. Cross-table metric consistency

If `RMSE seed42` appears in:

```text
FT02
FA01
FT10 narrative support
```

all raw values must be identical before rounding.

Create:

```text
final_table_cross_consistency_audit.csv.
```

---

# 139. Cross-table population consistency

Create:

```text
final_table_population_audit.csv.
```

For each table:

```text
population name
N
population SHA
allowed evidence class.
```

---

# 140. Cross-table model-lock consistency

Create:

```text
final_table_model_lock_audit.csv.
```

Every final Transformer result:

```text
FINAL_MODEL_LOCK_SHA256
```

must match.

---

# 141. Cross-table seed consistency

Create:

```text
final_table_seed_audit.csv.
```

Expected official final seed set:

```text
{42,123,2026}.
```

No missing or additional seed in final Transformer evidence.

---

# 142. Cross-table unit consistency

Create:

```text
final_table_unit_audit.csv.
```

No field may have:

```text
Wh in one table
kWh in another
```

without explicit conversion.

Canonical target unit remains:

```text
Wh.
```

---

# 143. Cross-table rounding consistency

Same field family always uses same display precision.

Example:

```text
RMSE everywhere = 2 decimals.
```

---

# 144. Cross-table label consistency

Canonical labels via:

```text
final_label_dictionary.csv.
```

Fields:

```text
internal_name
display_name
unit
definition
source_phase
notes.
```

---

# 145. Scientific notation policy

If a dimensionless value is extremely small:

```text
do not automatically use scientific notation in main table
```

unless rounding to 3 decimals would erase meaningful value.

For such fields:

```text
use 4–6 significant digits
and document exception.
```

Expected attention JSD/correlation usually does not need this.

---

# 146. No hidden precision claims

Displaying:

```text
0.123456789
```

suggests unjustified precision.

Use frozen display rules.

---

# 147. Table generation must be deterministic

Given:

```text
same upstream artifacts
same table inventory
same formatting config
```

generated CSV/Markdown/LaTeX should be deterministic.

---

# 148. Table generation config

Create:

```text
final_table_render_config.json
```

Freeze:

```text
table order
row order
column order
display precision
missing-value label
percentage style
mean±SD format
caption style
footnote style
LaTeX escaping
Markdown escaping.
```

---

# 149. Mean±SD display format

Canonical:

```text
123.45 ± 6.78
```

No parenthesized ambiguity.

Machine CSV should store:

```text
mean
sd
```

as separate numeric columns where relevant.

---

# 150. Confidence interval symbol is forbidden for seed SD

Do not write:

```text
95% CI
```

unless actual upstream CI exists.

---

# 151. Main table evidence completeness

Each major coursework requirement must appear at least once:

```text
energy regression performance
Transformer final configuration
LSTM baseline comparison
temporal robustness
prediction/error analysis
attention analysis
attention head comparison
error-conditioned attention
attention seed stability.
```

---

# 152. Coursework requirement coverage audit

Create:

```text
coursework_requirement_table_coverage.csv.
```

Rows:

```text
Task formulation
Dataset
Transformer model
Regression metrics
LSTM baseline
Attention analysis
Error analysis
Reproducibility
Seed robustness.
```

Columns:

```text
covered_by_table
covered_by_figure
source_phase
status.
```

---

# 153. Dataset facts in final tables

FT01 may include:

```text
Dataset: UCI Appliances Energy Prediction
sampling: 10 minutes
forecast horizon: 10 minutes
```

Use locked project contract.

No need to repeat all raw schema details.

---

# 154. No unsupported dataset generalization

Do not claim:

```text
multi-house generalization
cross-climate generalization
```

because dataset is a single-house setting.

This is more Phase59 limitation, but FT10 can carry it.

---

# 155. Warnings from Phase57

If Phase57 reports:

```text
head matching ambiguity
cycle inconsistency
seed attention variation.
```

FT09 must surface it rather than hide it in appendix only.

---

# 156. Warnings from Phase56

If:

```text
weak/no error-attention association
small signed subgroup
```

FT08 must not imply strong evidence.

---

# 157. Warnings from Phase50/51

If a regime has small N:

```text
N shown
warning footnote.
```

---

# 158. No conclusion from non-independent rows

Attention head metrics across targets are temporally dependent.

Tables report descriptive quantities.

No inferential significance language.

---

# 159. Table footnote wording for descriptive attention

Recommended:

```text
All attention statistics are descriptive diagnostics on the frozen Held-Out Test.
They were not used to modify the final model.
```

---

# 160. Table footnote wording for rolling-origin

Recommended:

```text
Rolling-origin results are pre-Test development robustness evidence;
the Held-Out Test remained untouched until final evaluation.
```

---

# 161. Table footnote wording for Test

Recommended:

```text
Held-Out Test results were evaluated after final model lock and were not used for model selection or retuning.
```

---

# 162. Table footnote wording for seed summary

Recommended:

```text
Mean ± SD summarizes the three predefined Transformer seeds (42, 123, 2026);
it does not represent an ensemble forecast.
```

---

# 163. Table footnote wording for worst cases

Recommended:

```text
Worst-error cases are retained valid Test observations and were not removed from final metrics.
```

---

# 164. Table footnote wording for head matching

Recommended:

```text
Cross-seed head-level comparisons use the frozen permutation-aware matching protocol;
same numeric head indices are not assumed semantically equivalent.
```

---

# 165. Table footnote wording for causality

Recommended:

```text
Attention associations are descriptive and do not establish causal feature or temporal importance.
```

---

# 166. No table should exceed source support

If upstream does not provide:

```text
a metric
a cohort
a summary
```

Phase58 marks:

```text
NOT_AVAILABLE_FROM_FROZEN_SOURCE
```

rather than guessing.

---

# 167. Table build status model

Per table:

```text
READY
READY_WITH_WARNING
BLOCKED_SOURCE_MISSING
BLOCKED_CONSISTENCY_FAILURE.
```

---

# 168. Phase58 overall PASS rule

All `FT01–FT10` must be:

```text
READY
or
READY_WITH_WARNING.
```

Appendix source gaps can be warning only if they are genuinely optional and do not undermine main-table traceability.

---

# 169. No silent source repair

If a source mismatch occurs:

```text
do not edit the source CSV in Phase58.
```

Record discrepancy and return to responsible phase.

---

# 170. Output directory

```text
artifacts/
└── final_tables/
    ├── final_tables_manifest.json
    ├── final_tables_contract.json
    ├── phase58_preflight_audit.csv
    ├── final_table_inventory.json
    ├── final_figure_inventory.json
    ├── final_table_render_config.json
    ├── final_table_source_ledger.csv
    ├── final_table_cell_lineage.csv
    ├── final_model_label_map.csv
    ├── final_label_dictionary.csv
    ├── final_table_cross_consistency_audit.csv
    ├── final_table_population_audit.csv
    ├── final_table_model_lock_audit.csv
    ├── final_table_seed_audit.csv
    ├── final_table_unit_audit.csv
    ├── final_table_rounding_audit.csv
    ├── coursework_requirement_table_coverage.csv
    ├── table_claim_traceability.csv
    ├── final_table_checksums.json
    ├── tables/
    │   ├── csv/
    │   │   ├── FT01_final_experimental_model_configuration.csv
    │   │   ├── FT02_final_heldout_test_performance.csv
    │   │   ├── FT03_rolling_origin_temporal_robustness.csv
    │   │   ├── FT04_final_prediction_residual_diagnostics.csv
    │   │   ├── FT05_error_by_regime_worst_error_summary.csv
    │   │   ├── FT06_last_query_temporal_attention_summary.csv
    │   │   ├── FT07_within_seed_head_comparison_summary.csv
    │   │   ├── FT08_error_conditioned_attention_summary.csv
    │   │   ├── FT09_seed_stability_attention_summary.csv
    │   │   ├── FT10_evidence_limitation_summary.csv
    │   │   ├── FA01_per_seed_final_test_metrics.csv
    │   │   ├── FA02_rolling_origin_fold_level_metrics.csv
    │   │   ├── FA03_residual_distribution_details.csv
    │   │   ├── FA04_full_error_by_regime_results.csv
    │   │   ├── FA05_shared_worst_error_cases.csv
    │   │   ├── FA06_full_last_query_head_metrics.csv
    │   │   ├── FA07_full_head_pairwise_comparison.csv
    │   │   ├── FA08_full_error_conditioned_attention_coefficients.csv
    │   │   ├── FA09_head_matching_ambiguity_details.csv
    │   │   ├── FA10_attention_seed_stability_details.csv
    │   │   ├── FA11_provenance_population_audit.csv
    │   │   └── FA12_upstream_warnings_reporting_caveats.csv
    │   ├── markdown/
    │   │   ├── FT01_final_experimental_model_configuration.md
    │   │   ├── FT02_final_heldout_test_performance.md
    │   │   └── ...
    │   ├── latex/
    │   │   ├── FT01_final_experimental_model_configuration.tex
    │   │   ├── FT02_final_heldout_test_performance.tex
    │   │   └── ...
    │   └── metadata/
    │       ├── FT01_metadata.json
    │       ├── FT02_metadata.json
    │       └── ...
    ├── FINAL_TABLE_CATALOG.md
    ├── final_tables_findings.csv
    ├── final_tables_tests.csv
    ├── final_tables_discrepancies.json
    ├── phase59_final_conclusions_handoff.json
    ├── final_tables_summary.json
    ├── final_tables_report.md
    ├── README_FINAL_TABLES.md
    └── phase_58_signoff.json
```

Ellipses in the directory example mean dynamically generated equivalent files for the remaining frozen table IDs; the actual artifact tree must contain explicit filenames.

---

# 171. Required outputs

```text
O58.1  Final tables manifest
O58.2  Final tables contract
O58.3  Preflight audit
O58.4  Frozen table inventory
O58.5  Frozen figure inventory
O58.6  Render config
O58.7  Source ledger
O58.8  Critical cell lineage
O58.9  Model label map
O58.10 Label dictionary
O58.11 FT01 configuration table
O58.12 FT02 final Test performance table
O58.13 FT03 rolling-origin robustness table
O58.14 FT04 prediction/residual diagnostics table
O58.15 FT05 regime/worst-error table
O58.16 FT06 last-query attention table
O58.17 FT07 head-comparison table
O58.18 FT08 error-conditioned attention table
O58.19 FT09 seed-stability attention table
O58.20 FT10 evidence/limitation table
O58.21 FA01–FA12 appendix tables
O58.22 CSV versions
O58.23 Markdown versions
O58.24 LaTeX versions
O58.25 Per-table metadata JSON
O58.26 Table checksums
O58.27 Cross-consistency audit
O58.28 Population audit
O58.29 Final-model-lock audit
O58.30 Seed audit
O58.31 Unit audit
O58.32 Rounding audit
O58.33 Coursework requirement coverage
O58.34 Table-to-claim traceability
O58.35 Table catalog
O58.36 Findings
O58.37 Tests
O58.38 Discrepancy log
O58.39 Phase59 handoff
O58.40 Summary JSON
O58.41 Human-readable report
O58.42 README
O58.43 Sign-off
```

---

# 172. Final tables manifest schema

`final_tables_manifest.json`:

```text
phase=58
version=FINAL_TABLES-v1
source_phase57_version
source_phase56_version
source_phase55_version
source_phase54_version
source_phase53_version
source_phase52_version
source_phase51_version
source_phase50_version
source_phase49_version
source_phase48_version
source_phase47_version
source_phase45_version
source_phase44_version
final_lock_sha256
final_test_population_sha256
seed_list=[42,123,2026]
main_table_ids=[FT01..FT10]
appendix_table_ids=[FA01..FA12]
new_training=false
new_test_inference=false
new_metric=false
new_model_selection=false
best_seed_selection=false
attention_reextraction=false
status
created_at
```

---

# 173. Final tables contract

`final_tables_contract.json` freezes:

```text
Main tables:
FT01..FT10.

Appendix:
FA01..FA12.

Model order:
Persistence
Tuned LSTM
Transformer seed42
Transformer seed123
Transformer seed2026
Transformer three-seed summary.

Seed order:
42,123,2026.

Metrics:
MAE Wh
RMSE Wh
R².

Three-seed summary:
metric mean ± sample SD
ddof=1
not ensemble.

Rounding:
Wh 2dp
R² 3dp
dimensionless attention/effect metrics 3dp
minutes 1dp
percentages 1dp.

No:
best-value highlighting
Test reranking
new metric
new significance test
ensemble reconstruction
new attention analysis
new cohort
new threshold
causal claim.
```

---

# 174. Preflight audit

`phase58_preflight_audit.csv`:

```text
check
expected
observed
critical
status
```

Required:

```text
Phase44 approved
Phase45 approved
Phase46 approved
Phase47 approved
Phase48 approved
Phase49 approved
Phase50 approved
Phase51 approved
Phase52 approved
Phase53 approved
Phase54 approved
Phase55 approved
Phase56 approved
Phase57 approved
phase58_ready=true
final lock exists
final Test population fingerprint exists
seed set exactly 42/123/2026
table inventory frozen
render config frozen
source ledger buildable
no new analysis requested.
```

---

# 175. Model label map schema

`final_model_label_map.csv`:

```text
internal_id
run_id_if_applicable
model_family
seed
display_label
role
final_lock_sha
status
```

---

# 176. Label dictionary schema

`final_label_dictionary.csv`:

```text
field_key
display_label
definition
unit
direction_if_applicable
display_precision
source_phase
notes
status.
```

---

# 177. Final table source ledger schema

`final_table_source_ledger.csv`:

```text
table_id
panel_id
field_id
display_label
source_phase
source_version
source_artifact
source_column
source_filter
aggregation
population
population_sha
unit
evidence_class
rounding_rule
footnote_rule
critical
status
```

---

# 178. Cross-consistency audit schema

`final_table_cross_consistency_audit.csv`:

```text
field_family
source_value
table_a
value_a_raw
table_b
value_b_raw
difference
tolerance
consistent
status.
```

---

# 179. Population audit schema

`final_table_population_audit.csv`:

```text
table_id
panel_id
population_name
N
population_sha
expected_population
same_as_expected
evidence_class
status.
```

---

# 180. Model-lock audit schema

`final_table_model_lock_audit.csv`:

```text
table_id
row_key
model_label
observed_final_lock_sha
expected_final_lock_sha
match
status.
```

---

# 181. Seed audit schema

`final_table_seed_audit.csv`:

```text
table_id
expected_seeds
observed_seeds
missing_seeds
extra_seeds
seed_order_correct
status.
```

---

# 182. Unit audit schema

`final_table_unit_audit.csv`:

```text
table_id
field
expected_unit
observed_unit
conversion_applied
conversion_rule
status.
```

---

# 183. Rounding audit schema

`final_table_rounding_audit.csv`:

```text
table_id
field
raw_value
display_value
expected_precision
aggregation_before_rounding
negative_zero_fixed
status.
```

---

# 184. Coursework coverage schema

`coursework_requirement_table_coverage.csv`:

```text
requirement
main_table
appendix_table
figure
source_phase
coverage_status
notes.
```

---

# 185. Table claim traceability schema

`table_claim_traceability.csv`:

```text
claim_id
claim_topic
allowed_claim_template
supporting_table
supporting_source_phase
supporting_artifact
required_caveat
prohibited_overclaim
status.
```

---

# 186. Table findings artifact

`final_tables_findings.csv`:

```text
finding_id
topic
source_phase
source_finding_id_if_available
supported_statement
supporting_table
caveat
ready_for_phase59
status.
```

No new substantive finding generated purely by Phase58 formatting.

---

# 187. Tests artifact

`final_tables_tests.csv`:

```text
test_id
scope
expected
observed
critical
status.
```

---

# 188. Discrepancy taxonomy

`final_tables_discrepancies.json`:

```text
UPSTREAM_SIGNOFF_MISSING
UPSTREAM_PHASE_FAILED
PHASE58_HANDOFF_NOT_READY
FINAL_LOCK_MISSING
FINAL_LOCK_MISMATCH
TEST_POPULATION_MISSING
TEST_POPULATION_MISMATCH
SEED_SET_MISMATCH
SEED_ORDER_MISMATCH
SOURCE_ARTIFACT_MISSING
SOURCE_FIELD_MISSING
SOURCE_CHECKSUM_MISMATCH
MANUAL_VALUE_WITHOUT_SOURCE
NARRATIVE_ONLY_VALUE_USED
DEVELOPMENT_TEST_EVIDENCE_MIXED
WRONG_EVIDENCE_CLASS
MODEL_LABEL_MISMATCH
B0_MISLABELED_AS_FINAL
BASELINE_POPULATION_MISMATCH
METRIC_VERSION_MISMATCH
RMSE_UNIT_MISMATCH
R2_CLAMPED
UNLOCKED_METRIC_ADDED
MAPE_ADDED
ENSEMBLE_RECONSTRUCTED
BEST_SEED_SELECTED
TEST_ROWS_SORTED_BY_PERFORMANCE
BEST_VALUE_HIGHLIGHTED
SEED_SUMMARY_USED_DDOF0
SEED_SUMMARY_CALCULATED_AFTER_ROUNDING
BASELINE_FAKE_SD
POOLED_RMSE_REPLACED_BY_MEAN_FOLD_RMSE
R2_AVERAGED_ACROSS_REGIMES
REGIME_THRESHOLD_CHANGED
WORST_CASE_RESELECTED
WORST_CASE_REMOVED_FROM_METRIC
ATTENTION_RECOMPUTED_FROM_IMAGE
ATTENTION_FEATURE_IMPORTANCE_CLAIM
ATTENTION_CAUSAL_CLAIM
HEAD_SEMANTIC_ALIGNMENT_ASSUMED_BY_INDEX
HEAD_STABILITY_THRESHOLD_ADDED
ERROR_COHORT_REDEFINED
ERROR_COHORT_USED_AS_DEPLOYMENT_REGIME
P_VALUE_ADDED
NEW_HYPOTHESIS_TEST_ADDED
NEW_CONFIDENCE_INTERVAL_ADDED
NEW_SCIENTIFIC_METRIC_DERIVED
NEW_MODEL_SELECTION
NEW_TRAINING
NEW_TEST_INFERENCE
ROUNDING_INCONSISTENT
UNIT_INCONSISTENT
NEGATIVE_ZERO_DISPLAY
MISSING_VALUE_REPLACED_WITH_ZERO
WARNING_NOT_PROPAGATED
TABLE_CELL_LINEAGE_MISSING
TABLE_CHECKSUM_MISSING
OTHER
```

---

# 189. Status model

## PASS

```text
all upstream sources verified
FT01–FT10 built
FA01–FA12 built as required
all critical table cells traceable
Test population consistent
final model lock consistent
seed set complete
rounding/unit rules consistent
no new selection/analysis
Phase59 handoff ready.
```

## PASS_WITH_WARNING

Possible:

```text
some upstream PASS_WITH_WARNING caveats
optional appendix table partially unavailable
report-width limitation requiring panel split
attention matching ambiguity propagated
small diagnostic cohort warning
three-seed limitation.
```

## FAIL

Examples:

```text
final Test metric source mismatch
missing seed
wrong population
new ensemble metric
best seed selection
untraceable numeric cell
development/Test evidence mixed.
```

---

# 190. Phase59 handoff purpose

Phase59 will write final conclusions.

Phase58 must hand off:

```text
exact final tables
supported claims
caveats
limitations
no unsupported inference.
```

---

# 191. Phase59 handoff schema

`phase59_final_conclusions_handoff.json`:

```text
source_phase58_version
final_lock_sha256
final_test_population_sha256
seed_list=[42,123,2026]
main_table_catalog
appendix_table_catalog
figure_inventory
table_claim_traceability
final_tables_findings
upstream_warnings
allowed_claims=[
  FINAL_TEST_PERFORMANCE,
  LSTM_COMPARISON,
  TEMPORAL_ROBUSTNESS,
  ERROR_REGIME_BEHAVIOR,
  WORST_ERROR_CONCENTRATION,
  LAST_QUERY_TEMPORAL_ATTENTION,
  HEAD_DIVERSITY,
  ERROR_ATTENTION_ASSOCIATION,
  ATTENTION_SEED_STABILITY
]
prohibited_claims=[
  CAUSAL_ATTENTION_EXPLANATION,
  RAW_FEATURE_IMPORTANCE_FROM_TEMPORAL_ATTENTION,
  BEST_SEED_SELECTION,
  POST_TEST_RETUNING,
  MULTI_HOUSE_GENERALIZATION
]
new_analysis_performed=false
ready_for_phase59=true
```

---

# 192. Execution sequence

```text
1. Load/verify all required upstream signoffs.
2. Load Phase57 → Phase58 handoff.
3. Freeze main and appendix table inventory.
4. Freeze figure inventory.
5. Freeze render/rounding config.
6. Build final model label map.
7. Build label dictionary.
8. Build source-of-truth ledger.
9. Verify final lock and Test population lineage.
10. Build FT01 directly from Phase45 lock.
11. Build FT02 directly from Phase47 final Test metrics.
12. Recompute only the authorized three-seed mean±sample-SD integrity fields.
13. Build FT03 from Phase44 rolling-origin outputs.
14. Build FT04 from Phase48/49.
15. Build FT05 from Phase50/51.
16. Build FT06 from Phase54.
17. Build FT07 from Phase55.
18. Build FT08 from Phase56.
19. Build FT09 from Phase57.
20. Build FT10 from table-to-claim traceability.
21. Build FA01–FA12 from frozen upstream detail artifacts.
22. Generate full-precision CSV tables.
23. Generate rounded Markdown tables.
24. Generate rounded LaTeX tables.
25. Generate per-table metadata JSON.
26. Run cell lineage checks.
27. Run cross-table consistency checks.
28. Run population/model-lock/seed/unit/rounding audits.
29. Generate table checksums.
30. Generate FINAL_TABLE_CATALOG.md.
31. Build coursework requirement coverage audit.
32. Build Phase59 claim handoff.
33. Run no-new-analysis/no-selection scope checks.
34. Write summary/report/README.
35. Sign off Phase58.
```

---

# 193. Recommended pseudocode

```text
required_phases = [
    44,45,46,47,48,49,50,51,52,53,54,55,56,57
]

for phase in required_phases:
    signoff = load_signoff(phase)
    assert signoff.status in {"PASS","PASS_WITH_WARNING"}

p57 = load_phase57_handoff()
assert p57.ready_for_phase58

freeze_table_inventory(
    main=["FT01","FT02","FT03","FT04","FT05","FT06","FT07","FT08","FT09","FT10"],
    appendix=["FA01","FA02","FA03","FA04","FA05","FA06","FA07","FA08","FA09","FA10","FA11","FA12"]
)

freeze_render_config(
    model_order=[
        "Persistence Baseline",
        "Tuned LSTM Baseline",
        "Final Transformer — Seed 42",
        "Final Transformer — Seed 123",
        "Final Transformer — Seed 2026",
        "Final Transformer — Three-Seed Summary"
    ],
    seed_order=[42,123,2026],
    precision={
        "wh":2,
        "r2":3,
        "dimensionless":3,
        "minutes":1,
        "percentage":1
    },
    seed_sd_ddof=1,
    missing_label="N/A",
    best_value_highlighting=False
)

labels = build_final_label_dictionary()
source_ledger = build_source_of_truth_ledger()

verify_final_lock_everywhere()
verify_test_population_everywhere()
verify_seed_set_everywhere()

# FT01
ft01 = build_from_phase45_final_lock()

# FT02
phase47_metrics = load_phase47_final_test_metrics()

transformer_seed_metrics = select_exact_final_transformer_seeds(
    phase47_metrics,
    seeds=[42,123,2026]
)

for metric in ["MAE","RMSE","R2"]:
    raw = transformer_seed_metrics[metric].full_precision_values

    summary_mean = mean(raw)
    summary_sd = sample_sd(raw, ddof=1)

    verify_no_ensemble_semantics()

ft02 = build_final_test_table(
    persistence=load_phase47_persistence_metrics(),
    lstm=load_phase47_tuned_lstm_metrics(),
    transformer_seeds=transformer_seed_metrics,
    transformer_seed_summary=authorized_metric_summary
)

assert same_test_population_all_rows(ft02)

# FT03
ft03 = build_from_phase44_rolling_origin()
assert pooled_rmse_source_is_authoritative(ft03)
mark_phase45_final_source_candidate(ft03)

# FT04
ft04 = build_from_phase48_49_without_new_metrics()

# FT05
ft05 = build_from_phase50_51(
    regimes_frozen=True,
    worst_case_ranks_frozen=True
)

# FT06
ft06 = build_from_phase54_last_query()

# FT07
ft07 = build_from_phase55_layer_diversity()

# FT08
ft08 = build_from_phase56_layer_head_mean_results()

# FT09
ft09 = build_from_phase57_stability_results()

# FT10
ft10 = build_evidence_limitation_table(
    upstream_findings_only=True
)

appendix_tables = build_appendix_from_frozen_sources()

all_csv = write_full_precision_csvs()
all_md  = write_rounded_markdown()
all_tex = write_rounded_latex()

build_cell_lineage()
run_cross_table_consistency()
run_population_audit()
run_model_lock_audit()
run_seed_audit()
run_unit_audit()
run_rounding_audit()

assert no_new_metric
assert no_new_hypothesis_test
assert no_new_model_selection
assert no_best_seed
assert no_ensemble_reconstruction
assert no_new_test_inference
assert no_attention_reextraction

write_checksums()
write_catalog()
write_claim_traceability()
write_phase59_handoff()

signoff_phase58()
```

---

# 194. Preflight acceptance checklist

```text
[ ] Phase44 PASS/PASS_WITH_WARNING.
[ ] Phase45 PASS/PASS_WITH_WARNING.
[ ] Phase46 PASS/PASS_WITH_WARNING.
[ ] Phase47 PASS/PASS_WITH_WARNING.
[ ] Phase48 PASS/PASS_WITH_WARNING.
[ ] Phase49 PASS/PASS_WITH_WARNING.
[ ] Phase50 PASS/PASS_WITH_WARNING.
[ ] Phase51 PASS/PASS_WITH_WARNING.
[ ] Phase52 PASS/PASS_WITH_WARNING.
[ ] Phase53 PASS/PASS_WITH_WARNING.
[ ] Phase54 PASS/PASS_WITH_WARNING.
[ ] Phase55 PASS/PASS_WITH_WARNING.
[ ] Phase56 PASS/PASS_WITH_WARNING.
[ ] Phase57 PASS/PASS_WITH_WARNING.
[ ] phase58_ready=true.
[ ] Final lock SHA known.
[ ] Final Test population SHA known.
[ ] Final seed set exactly 42/123/2026.
[ ] Table inventory frozen.
[ ] Figure inventory frozen.
[ ] Render config frozen.
[ ] No new scientific analysis needed.
```

---

# 195. Source-ledger acceptance checklist

```text
[ ] Every main table field mapped.
[ ] Every metric has authoritative source.
[ ] Source phase/version stored.
[ ] Source artifact stored.
[ ] Source field stored.
[ ] Population stored.
[ ] Unit stored.
[ ] Evidence class stored.
[ ] Aggregation stored.
[ ] Rounding rule stored.
[ ] Footnote rule stored.
[ ] No narrative-only scientific value.
```

---

# 196. FT01 acceptance checklist

```text
[ ] Final config from Phase45 only.
[ ] Runtime selected feature variant.
[ ] Runtime selected lookback.
[ ] Runtime selected architecture.
[ ] Runtime selected optimizer/loss settings.
[ ] Final fixed epoch count.
[ ] Final seeds.
[ ] Final Train+Validation refit region.
[ ] Test protocol.
[ ] No B0 value assumed unless actually final.
[ ] No development early-stopping recipe mislabeled final refit.
```

---

# 197. FT02 acceptance checklist

```text
[ ] Persistence row exact.
[ ] Tuned LSTM row exact.
[ ] Transformer seed42 exact.
[ ] Transformer seed123 exact.
[ ] Transformer seed2026 exact.
[ ] Three-seed mean full precision.
[ ] Three-seed sample SD ddof=1.
[ ] No ensemble prediction.
[ ] Same target IDs.
[ ] Same y_true.
[ ] Same metric version.
[ ] MAE/RMSE Wh.
[ ] R² unclamped.
[ ] No best seed.
[ ] No performance sorting.
[ ] No best-value bold.
[ ] Mean±SD footnote.
```

---

# 198. FT03 acceptance checklist

```text
[ ] Phase44 source only.
[ ] RO1/RO2/RO3 exact.
[ ] Pooled RMSE exact.
[ ] Mean fold RMSE clearly secondary.
[ ] Fold SD exact.
[ ] Worst fold exact.
[ ] Final-source candidate matches Phase45.
[ ] Development evidence label.
[ ] No Test value mixed.
```

---

# 199. FT04 acceptance checklist

```text
[ ] Phase48 prediction diagnostics exact.
[ ] Phase49 residual diagnostics exact.
[ ] residual=y_true-y_pred.
[ ] No ensemble residual.
[ ] No new statistic.
[ ] No row drops.
[ ] Units correct.
```

---

# 200. FT05 acceptance checklist

```text
[ ] Phase50 regime thresholds unchanged.
[ ] All shown regimes trace to frozen labels.
[ ] Counts displayed.
[ ] Small-N warnings propagated.
[ ] Phase51 error concentration exact.
[ ] Worst-case ranking unchanged.
[ ] No worst-case deletion.
[ ] No post-hoc new regime.
```

---

# 201. FT06 acceptance checklist

```text
[ ] Phase54 source exact.
[ ] Last-query definition preserved.
[ ] Lag units correct.
[ ] Main aggregation predeclared.
[ ] Per-head details in appendix.
[ ] No head ranking.
[ ] Attention temporal-allocation footnote.
[ ] Pooling caveat.
[ ] No feature-importance wording.
```

---

# 202. FT07 acceptance checklist

```text
[ ] Phase55 source exact.
[ ] Seed/layer rows complete.
[ ] Mean pairwise JSD.
[ ] Wasserstein minutes.
[ ] Expected-lag differences.
[ ] Top1 TVD.
[ ] No composite diversity score.
[ ] No head pruning implication.
[ ] Redundancy caveat.
```

---

# 203. FT08 acceptance checklist

```text
[ ] Phase56 source exact.
[ ] Six core metrics preserved.
[ ] Error cohorts unchanged.
[ ] Spearman exact.
[ ] High-low median delta exact.
[ ] Cliff's delta exact.
[ ] Profile JSD/Wasserstein exact.
[ ] Error cohort diagnostic footnote.
[ ] No p-values added.
[ ] No causal language.
[ ] No best head.
```

---

# 204. FT09 acceptance checklist

```text
[ ] Phase57 source exact.
[ ] Layer head-mean stability present.
[ ] Head matching summary present.
[ ] Cycle consistency present.
[ ] Matching sensitivity present.
[ ] Ambiguity warning propagated.
[ ] Error-effect seed consistency present.
[ ] Same-index semantic caveat.
[ ] No stable/unstable threshold.
[ ] No best seed.
```

---

# 205. FT10 acceptance checklist

```text
[ ] Every question linked to a table.
[ ] Evidence class correct.
[ ] Allowed conclusion supported.
[ ] Prohibited overclaim explicit.
[ ] Single-house generalization limitation represented if relevant.
[ ] Attention causality limitation represented.
[ ] Three-seed limitation represented.
[ ] No unsupported positive spin.
```

---

# 206. Appendix acceptance checklist

```text
[ ] FA01 full seed metrics.
[ ] FA02 full rolling folds.
[ ] FA03 residual details.
[ ] FA04 all regime results.
[ ] FA05 frozen worst cases.
[ ] FA06 all head metrics.
[ ] FA07 all head-pair results.
[ ] FA08 all error-conditioned coefficients.
[ ] FA09 matching ambiguity.
[ ] FA10 stability detail.
[ ] FA11 provenance.
[ ] FA12 warnings/caveats.
[ ] No result-driven omission.
```

---

# 207. Rounding acceptance checklist

```text
[ ] Machine CSV full precision.
[ ] Wh 2dp in report.
[ ] R² 3dp.
[ ] Dimensionless 3dp.
[ ] Minutes 1dp.
[ ] Percent 1dp.
[ ] Aggregation before rounding.
[ ] ddof=1.
[ ] Negative zero cleaned only in display.
[ ] Same field same precision everywhere.
```

---

# 208. Provenance acceptance checklist

```text
[ ] Table metadata JSON exists.
[ ] Source phase recorded.
[ ] Source artifact recorded.
[ ] Population SHA recorded.
[ ] Final lock SHA recorded where applicable.
[ ] Source checksum recorded.
[ ] Table CSV checksum.
[ ] Markdown checksum.
[ ] LaTeX checksum.
[ ] Critical cell lineage complete.
```

---

# 209. Cross-table acceptance checklist

```text
[ ] Same metric value identical across tables before rounding.
[ ] Same model label everywhere.
[ ] Same seed labels everywhere.
[ ] Same units everywhere.
[ ] Same Test population everywhere for Test evidence.
[ ] Development values never relabeled Test.
[ ] Test diagnostics never relabeled model-selection evidence.
[ ] No macro/pooled confusion.
[ ] No ensemble semantics.
```

---

# 210. Scope acceptance checklist

```text
[ ] No new training.
[ ] No new Test inference.
[ ] No new attention extraction.
[ ] No new metric.
[ ] No new error cohort.
[ ] No new threshold.
[ ] No new head matching.
[ ] No new hypothesis test.
[ ] No new confidence interval.
[ ] No best seed.
[ ] No Test reranking.
[ ] No ensemble reconstruction.
[ ] No causal claim.
```

---

# 211. Acceptance criteria

Phase58 PASS only when:

```text
Every required upstream phase from the final robustness/evaluation/diagnostic chain is approved and its relevant machine-readable artifacts are available.

The final table inventory is frozen before report rendering and includes the full core set FT01–FT10 plus the predefined appendix tables.

Every scientific numeric field in a final table has a machine-readable source in the final table source ledger.

The exact Phase45 final model lock is used to populate final configuration and model identity throughout all tables.

The exact Phase47 Held-Out Test metrics are used for final performance reporting without rerunning Test inference or reconstructing an unplanned ensemble.

All three predefined final Transformer seeds are reported symmetrically, and the three-seed summary is computed only as mean ± sample SD of the three seed-level metrics with ddof=1.

The three-seed metric summary is explicitly distinguished from an ensemble forecast.

Persistence, tuned LSTM and Transformer are compared only on a verified common final Test population and shared metric contract.

Development rolling-origin robustness is presented separately from Held-Out Test performance and retains pooled outer-fold RMSE as its primary robustness metric.

Prediction/residual, error-by-regime and worst-error tables use only the frozen Phase48–51 definitions and do not change regimes, cases or error conventions.

Attention tables use only frozen Phase54–57 quantitative artifacts; no value is digitized from heatmap images.

Last-query attention is labeled as temporal token allocation rather than raw-feature importance.

Head-comparison tables report transparent diversity/similarity metrics without creating a head score, head ranking or pruning recommendation.

Error-conditioned attention tables preserve the frozen diagnostic cohort definitions and explicitly state that error-attention associations are non-causal post-Test diagnostics.

Seed-stability attention tables preserve the permutation-aware Phase57 matching protocol and do not assume same-index head semantic equivalence.

All table units, metric definitions, populations, seeds, rounding rules and model labels are consistent across the final table package.

Full-precision machine-readable CSVs are generated before display rounding, and no aggregate is computed from rounded source values.

Report-ready Markdown and LaTeX versions are generated deterministically from the same full-precision source tables.

Cross-table consistency, population, model-lock, seed, unit and rounding audits all pass.

Upstream PASS_WITH_WARNING conditions are propagated into table metadata/footnotes rather than hidden.

No best seed is selected, no Test result is used to retune or rerank the final model, no new scientific metric or hypothesis test is introduced and no new causal claim is created.

Phase59 receives a claim-traceability handoff that states both what each table supports and what it does not support.
```

---

# 212. Failure conditions

Phase58 FAIL if:

```text
a final Test value cannot be traced to Phase47

a Final Transformer row uses a non-final checkpoint/config

a seed is omitted because its metric is worse

seed predictions are averaged to create a new ensemble result

the three-seed summary uses ddof=0 without protocol

mean/SD is calculated from rounded metrics

baseline and Transformer use different Test populations without disclosure

rolling-origin mean fold RMSE is mislabeled pooled RMSE

Validation/development results are mixed into Held-Out Test table

R² is clamped or averaged incorrectly

a new MAPE/accuracy metric is introduced

Phase50 regimes are redefined

Phase51 worst cases are reselected or removed

attention numbers are read from PNG heatmaps

same-index heads across seeds are treated as semantically identical

a new stability threshold is invented

Phase56 diagnostic high-error group is called a deployment regime

p-values or new hypothesis tests are added

best-value highlighting is used to imply post-Test model selection

a scientific cell is manually typed without source lineage

missing values are silently converted to zero

upstream warnings are hidden

a causal attention claim is introduced.
```

---

# 213. Common mistakes

## 213.1 Chỉ report seed có RMSE thấp nhất

Sai.

Ba seeds là official final stochastic realizations.

## 213.2 Lấy trung bình prediction của 3 seeds rồi tính RMSE

Sai nếu ensemble không được predeclare.

Three-seed summary là:

```text
mean ± SD của metrics
```

không phải metric của mean prediction.

## 213.3 Mean fold RMSE rồi gọi pooled RMSE

Sai Phase44 semantics.

## 213.4 Gộp rolling-origin và Test trong cùng cột “Performance”

Dễ làm người đọc nhầm development với final generalization.

## 213.5 R² âm thì sửa thành 0

Sai.

## 213.6 Thêm MAPE vì “forecasting thường có MAPE”

Sai locked metrics.

## 213.7 Bold số Test đẹp nhất

Không cần và dễ tạo impression of post-Test winner selection.

## 213.8 Report head entropy thấp nhất như “best head”

Sai.

## 213.9 Gọi attention là feature importance

Sai axis semantics.

## 213.10 Gọi HIGH_ERROR là operating regime

Sai. Đây là Test-relative diagnostic cohort.

## 213.11 Dùng table summary để che upstream warning

Sai governance.

## 213.12 Copy số thủ công từ Markdown report vào table

Dễ drift. Dùng machine-readable source.

---

# 214. Human-readable report structure

`final_tables_report.md`:

```text
1. Objective
2. Upstream evidence freeze
3. Evidence-class separation
4. Final table inventory
5. Source-of-truth ledger
6. Final model label and metric contracts
7. Rounding and unit contracts
8. FT01 Final configuration
9. FT02 Held-Out Test performance
10. FT03 Rolling-origin robustness
11. FT04 Prediction/residual diagnostics
12. FT05 Error-by-regime/worst-error summary
13. FT06 Last-query attention
14. FT07 Head comparison
15. FT08 Error-conditioned attention
16. FT09 Seed-stability attention
17. FT10 Evidence/limitation summary
18. Appendix package
19. Cross-table consistency audits
20. Warnings/caveats
21. Phase59 claim handoff
22. Definition of Done
```

---

# 215. README requirements

`README_FINAL_TABLES.md` explains:

```text
which tables are main vs appendix
which phases are authoritative for which values
how to regenerate tables
why full precision CSV precedes report rounding
why three-seed summary is not an ensemble
why development and Test evidence are separated
why no best-value highlighting is used
why attention tables are diagnostic
why Phase56 error cohorts are not deployment regimes
why same-index attention heads are not assumed aligned across seeds
how table checksums/provenance work
how Phase59 should use the claim-traceability file.
```

---

# 216. Summary artifact

`final_tables_summary.json`:

```text
version
final_lock_sha256
final_test_population_sha256
seed_list=[42,123,2026]
main_table_ids
appendix_table_ids
main_tables_ready_count
appendix_tables_ready_count
source_ledger_status
cell_lineage_status
performance_table_status
rolling_origin_table_status
error_tables_status
attention_tables_status
seed_stability_table_status
cross_consistency_status
population_audit_status
model_lock_audit_status
seed_audit_status
unit_audit_status
rounding_audit_status
coursework_coverage_status
upstream_warning_count
new_training=false
new_test_inference=false
new_metric=false
ensemble_reconstructed=false
best_seed_selected=false
causal_claim=false
phase59_ready
overall_status.
```

Runtime counts must be populated from execution, not fabricated in advance.

---

# 217. Phase58 sign-off

`phase_58_signoff.json` minimum:

```text
phase=58
phase_name=Final tables
version=FINAL_TABLES-v1
final_lock_sha256
final_test_population_sha256
seed_list=[42,123,2026]
main_table_inventory_frozen
appendix_table_inventory_frozen
figure_inventory_frozen
source_ledger_complete
critical_cell_lineage_complete
FT01_ready
FT02_ready
FT03_ready
FT04_ready
FT05_ready
FT06_ready
FT07_ready
FT08_ready
FT09_ready
FT10_ready
appendix_package_ready
csv_package_ready
markdown_package_ready
latex_package_ready
table_checksums_complete
cross_consistency_verified
population_consistency_verified
model_lock_consistency_verified
seed_consistency_verified
unit_consistency_verified
rounding_consistency_verified
coursework_requirement_coverage_verified
upstream_warnings_propagated
new_training=false
new_test_inference=false
new_attention_extraction=false
new_metric=false
new_hypothesis_test=false
new_model_selection=false
ensemble_reconstructed=false
best_seed_selected=false
best_head_selected=false
causal_claim=false
phase59_ready
warnings
overall_status
created_at.
```

---

# 218. Definition of Done

\[
\boxed{
Frozen\ Upstream\ Results
+
FT01\text{–}FT10
+
FA01\text{–}FA12
+
Full\text{-}Precision\ CSV
+
Report\text{-}Ready\ Markdown/LaTeX
+
Source\ Ledger
+
Cell\ Lineage
+
Population/Lock/Seed/Unit\ Audits
+
No\ New\ Selection
+
Phase59\ Handoff
}
\]

---

# 219. Final status contract

```text
PHASE 58 builds the final reporting tables.

Authoritative final performance:
Phase47.

Final config:
Phase45.

Rolling-origin robustness:
Phase44.

Prediction/residual:
Phase48–49.

Error regimes/worst cases:
Phase50–51.

Attention:
Phase54–57
with raw source lineage to Phase52.

Main tables:
FT01 Final config
FT02 Held-Out Test performance
FT03 Rolling-origin robustness
FT04 Prediction/residual diagnostics
FT05 Error-by-regime/worst-error
FT06 Last-query attention
FT07 Head comparison
FT08 Error-conditioned attention
FT09 Seed-stability attention
FT10 Evidence/limitations.

Official seeds:
42,123,2026.

Three-seed summary:
mean ± sample SD of seed-level metrics
ddof=1
not an ensemble.

Display precision:
Wh 2dp
R² 3dp
dimensionless 3dp
minutes 1dp
percent 1dp.

Machine source:
full precision first.

Forbidden:
new training
new Test inference
new metric
new cohort
new threshold
new head matching
new significance test
ensemble reconstruction
best seed
best head
post-Test model selection
attention feature-importance claim
causal claim.

After FINAL_TABLES-v1 PASS:
proceed to
PHASE 59 — Final Conclusions.
```

---

# 220. Final check

Correct:

```text
verify all frozen upstream evidence
→ freeze final table inventory
→ build source ledger
→ build FT01–FT10
→ build appendix
→ full-precision CSV
→ display rounding
→ Markdown/LaTeX
→ cross-table audits
→ claim traceability
→ Phase59 handoff
```

Incorrect:

```text
look at Test metrics
→ keep best seed only
```

Incorrect:

```text
average predictions across seeds
→ create unplanned ensemble RMSE
```

Incorrect:

```text
add MAPE/significance test
because final report looks more complete
```

Incorrect:

```text
read attention heatmap colors
→ manually enter numeric table
```

Incorrect:

```text
format a result strongly
→ turn it into a stronger scientific claim
```

Chỉ sau khi:

```text
FINAL_TABLES-v1
=
PASS / PASS_WITH_WARNING
```

và:

```text
phase59_ready=true
```

mới chuyển sang **PHASE 59 — Final Conclusions**.
